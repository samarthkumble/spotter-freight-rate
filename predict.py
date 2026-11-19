"""Predict the 12,000 validation loads and the 31 fixed December rows.

Run:  python predict.py   (after python train.py)
Writes: validation_predictions.csv, data/december_chart_inputs.csv (filled),
        results/december_predictions.csv (copy kept in the repo)
"""
import lightgbm as lgb
import pandas as pd

import freight

model = lgb.Booster(model_file="models/lgbm_model.txt")
train = freight.clean(freight.load("data/train_test.csv"))

# --- 1. Validation loads -----------------------------------------------------
validation = freight.clean(freight.load("data/validation.csv"))
validation["predicted_rate"] = freight.predict(model, validation).round(2)

template = pd.read_csv("data/validation_predictions_template.csv")
out = template[["load_id"]].merge(validation[["load_id", "predicted_rate"]], on="load_id", how="left")
assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
out.to_csv("validation_predictions.csv", index=False)
print(f"Wrote validation_predictions.csv: {len(out):,} rows, "
      f"rates ${out['predicted_rate'].min():,.0f} to ${out['predicted_rate'].max():,.0f}")

# --- 2. Fixed December chart -------------------------------------------------
# The chart file has no coordinates, quote_signal or load_id. Coordinates come
# from the city lookup in the data. quote_signal is a per-load value with no
# date pattern, so a typical value (training median) is used for every day.
december = pd.read_csv("data/december_chart_inputs.csv")
columns = list(december.columns)

both = pd.concat([train, validation])
pickup_xy = both.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
delivery_xy = both.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()

rows = december.copy()
rows["date"] = pd.to_datetime(rows["date"])
rows = rows.join(pickup_xy, on="pickup").join(delivery_xy, on="delivery")
rows["quote_signal"] = train["quote_signal"].median()
rows["market_index"] = float("nan")  # not used by the final model

december["predicted_rate"] = freight.predict(model, rows).round(2)
december = december[columns]
december.to_csv("data/december_chart_inputs.csv", index=False)
december.to_csv("results/december_predictions.csv", index=False)
print(f"Filled data/december_chart_inputs.csv: ${december['predicted_rate'].min():,.2f} "
      f"to ${december['predicted_rate'].max():,.2f}")
