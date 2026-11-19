"""Backtest the model forward in time, then train the final model.

Run:  python train.py
Writes: results/backtest.csv, results/unseen_cities.csv, models/lgbm_model.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

import freight

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

df = freight.clean(freight.load("data/train_test.csv"))
print(f"Loaded {len(df):,} loads, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

# ---------------------------------------------------------------------------
# Why the split is by date, not random:
# validation.csv is Nov-Dec 2025, after every training row (Jan-Oct).
# A random split would let the model see loads from the same days it is
# tested on, which is easier than the real task. So each fold trains only on
# the past and tests on the next two months (rolling-origin backtest).
# ---------------------------------------------------------------------------
FOLDS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),  # most recent: closest to Nov-Dec
]
DISTANCE_BANDS = [0, 150, 300, 500, 800, 1200, 1800, 2500, 4000]


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median rate-per-mile for (equipment, distance band) x distance."""
    band = lambda d: pd.cut(d["distance"], DISTANCE_BANDS)
    rpm = (train["posted_rate"] / train["distance"]).groupby(
        [train["equipment"], band(train)], observed=True).median()
    keys = pd.MultiIndex.from_arrays([test["equipment"], band(test)])
    return test["distance"].to_numpy() * rpm.reindex(keys).to_numpy()


rows = []
for start, end in FOLDS:
    past = df[df["date"] < start]
    test = df[(df["date"] >= start) & (df["date"] < end)]

    # Corrupted labels are found using the training window only.
    past_bad = freight.flag_outliers(past)
    train = past[~past_bad]
    # For scoring, flag test rows with a model that never saw them.
    huber = freight.fit(train, params={**freight.LGB_PARAMS, "objective": "huber", "alpha": 0.2})
    ratio = test["posted_rate"] / freight.predict(huber, test)
    test_clean = test[(ratio >= freight.OUTLIER_LOW) & (ratio <= freight.OUTLIER_HIGH)]

    candidates = {
        "Baseline (median $/mile)": lambda tr, te: baseline_predict(tr, te),
        "LightGBM + market_index": lambda tr, te: freight.predict(
            freight.fit(tr, freight.FEATURES + ["market_index"]), te, freight.FEATURES + ["market_index"]),
        "LightGBM, uncleaned labels": lambda tr, te: freight.predict(freight.fit(past), te),
        "LightGBM (final)": lambda tr, te: freight.predict(freight.fit(tr), te),
    }
    for name, run in candidates.items():
        pred = run(train, test)
        pred_clean = pd.Series(pred, index=test.index).loc[test_clean.index].to_numpy()
        m_all = freight.metrics(test["posted_rate"], pred)
        m_clean = freight.metrics(test_clean["posted_rate"], pred_clean)
        rows.append({
            "test_period": f"{start} to {end}",
            "model": name,
            "MAE_all_rows": round(m_all["MAE"], 1),
            "MAE_clean_rows": round(m_clean["MAE"], 1),
            "MAPE_clean_%": round(m_clean["MAPE_%"], 2),
            "median_APE_clean_%": round(m_clean["median_APE_%"], 2),
        })
    print(f"  fold {start}..{end}: train {len(train):,} (dropped {past_bad.sum()} bad labels), test {len(test):,}")

backtest = pd.DataFrame(rows)
backtest.to_csv("results/backtest.csv", index=False)
summary = backtest.groupby("model", sort=False)[["MAE_all_rows", "MAE_clean_rows", "MAPE_clean_%"]].mean().round(2)
print("\nRolling backtest (mean of 3 forward folds):")
print(summary.to_string())

# ---------------------------------------------------------------------------
# Unseen-city check: 12% of validation.csv rows involve 8 cities that never
# appear in train_test.csv. Simulate that by hiding 8 cities from training.
# ---------------------------------------------------------------------------
bad_all = freight.flag_outliers(df)
clean_df = df[~bad_all]
rng = np.random.default_rng(0)
cities = sorted(set(df["pickup"]) | set(df["delivery"]))
city_rows = []
for repeat in range(3):
    hidden = set(rng.choice(cities, 8, replace=False))
    touches = clean_df["pickup"].isin(hidden) | clean_df["delivery"].isin(hidden)
    train = clean_df[~touches & (clean_df["date"] < "2025-09-01")]
    test = clean_df[touches & (clean_df["date"] >= "2025-09-01")]
    m = freight.metrics(test["posted_rate"], freight.predict(freight.fit(train), test))
    city_rows.append({"repeat": repeat + 1, "hidden_cities": ", ".join(sorted(hidden)),
                      "test_rows": len(test), "MAPE_%": round(m["MAPE_%"], 2)})
unseen = pd.DataFrame(city_rows)
unseen.to_csv("results/unseen_cities.csv", index=False)
print(f"\nUnseen-city test, mean MAPE: {unseen['MAPE_%'].mean():.2f}%")

# ---------------------------------------------------------------------------
# Final model: all Jan-Oct data, corrupted labels removed.
# ---------------------------------------------------------------------------
print(f"\nFinal training: {len(clean_df):,} loads ({bad_all.sum()} corrupted labels removed)")
model = freight.fit(clean_df)
model.save_model("models/lgbm_model.txt")
importance = pd.Series(model.feature_importance("gain"), index=freight.FEATURES)
(importance / importance.sum() * 100).round(2).sort_values(ascending=False).to_csv(
    "results/feature_importance.csv", header=["gain_%"])
print("Saved models/lgbm_model.txt")
