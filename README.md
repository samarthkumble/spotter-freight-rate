# Freight Rate Prediction

Predicts the posted rate ($) of a truckload from its lane, distance, equipment, weight and date.
LightGBM on log(rate), trained on Jan-Oct 2025 loads and checked with forward-in-time backtests.

## Results (rolling backtest, mean of 3 forward folds)

| Model | MAE, all test rows | MAE, clean rows | MAPE, clean rows |
|---|---|---|---|
| Baseline: median $/mile by equipment and distance band | $158 | $105 | 4.54% |
| LightGBM trained on uncleaned labels | $144 | $91 | 4.08% |
| LightGBM with market_index | $121 | $67 | 2.90% |
| **LightGBM, final** | **$113** | **$59** | **2.52%** |

On the most recent fold (train Jan-Aug, test Sep-Oct) the final model's MAPE is 1.44%.
Loads that touch a city the model never trained on: 1.51% MAPE.

## Setup

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the four provided CSV files in `data/` (they are not committed):

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python eda.py        # data-quality findings + figures/
python train.py      # backtest (results/) + final model (models/lgbm_model.txt)
python predict.py    # validation_predictions.csv + fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` is Spotter's scorer; it validates both files and writes `scorer_results/candidate_december.png`.

## Files

| File | What it does |
|---|---|
| `freight.py` | Shared cleaning, features, model training and metrics |
| `eda.py` | Data-quality checks and figures |
| `train.py` | Rolling backtest against the baseline and alternatives, then the final model |
| `predict.py` | Final predictions for validation.csv and the December chart |
| `validation_predictions.csv` | Submission: `load_id,predicted_rate` for all 12,000 loads |
| `results/` | Backtest tables, feature importance, December predictions |

## Key decisions

- **Time-based validation.** The loads to predict (Nov-Dec) come after all training data, so every test fold is later than its training data. A random split would overstate accuracy.
- **Corrupted labels removed.** 677 rates (1.4%) sit at about 0.2-0.45x or 2.2-5x the expected rate, with no loads between 0.6x and 1.6x. They are dropped from training using those cut-offs.
- **Negative weights** (292 train, 145 validation) are sign errors and are made positive. Missing weights are left to LightGBM.
- **market_index is not used.** It improves two older folds but makes the most recent fold much worse (1.44% to 3.79% MAPE); its link to rates is not stable over time.
- **Coordinates instead of city names.** 8 cities in validation.csv never appear in training, so the model locates them by latitude/longitude.
- **December chart.** The fixed inputs have no quote_signal, so the training median is used. The only date feature is day of week, so the chart shows a weekly cycle (Thursday highest, Sunday lowest, about 1.2% apart).
