"""Shared helpers: loading, cleaning, features and the LightGBM model.

Every script (eda.py, train.py, predict.py) imports from here, so the
cleaning and feature logic is written once and used the same way for
training, backtesting and final predictions.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]
COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# Final feature set. market_index is deliberately NOT used: it looks useful
# inside the training months but makes forward-in-time predictions worse
# (see train.py backtest). City names are not used either: 8 cities in
# validation.csv never appear in training, so the model locates cities by
# their coordinates instead.
FEATURES = [
    "distance",
    "log_distance",
    "equipment",
    "weight",
    "quote_signal",
    "day_of_week",
    *COORDS,
]

# A rate is treated as a corrupted label when it is below 0.6x or above
# 1.6x what a robust model expects. Real loads sit within about +/-5%;
# the corrupted ones sit at about 0.2-0.45x or 2.2-5x, with nothing in between.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.6

LGB_PARAMS = {
    "objective": "regression",  # L2 loss on log(rate)
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}
NUM_ROUNDS = 1800


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix input problems that also exist in validation.csv."""
    df = df.copy()
    # Negative weights are sign errors: their absolute values have the same
    # distribution as the positive weights (5,000-47,500 lb).
    df["weight"] = df["weight"].abs()
    # Missing weights stay NaN; LightGBM learns where to send missing values.
    return df


def build_features(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)
    X["weight"] = df["weight"]
    X["quote_signal"] = df["quote_signal"]
    X["day_of_week"] = df["date"].dt.dayofweek
    for col in COORDS:
        X[col] = df[col]
    X["market_index"] = df["market_index"]  # only used by the backtest comparison
    return X[features]


def fit(df: pd.DataFrame, features: list[str] = FEATURES, params: dict | None = None,
        rounds: int = NUM_ROUNDS) -> lgb.Booster:
    """Train on log(rate): errors become relative, and predictions stay positive."""
    data = lgb.Dataset(build_features(df, features), label=np.log(df["posted_rate"]))
    return lgb.train(params or LGB_PARAMS, data, num_boost_round=rounds)


def predict(model: lgb.Booster, df: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    return np.exp(model.predict(build_features(df, features)))


def flag_outliers(df: pd.DataFrame) -> pd.Series:
    """Return True for rows whose posted_rate looks like a corrupted label.

    A Huber-loss model is fitted first; Huber loss limits how much the
    corrupted rows can pull the fit, so the expected rate stays reliable.
    """
    params = {**LGB_PARAMS, "objective": "huber", "alpha": 0.2, "learning_rate": 0.05}
    model = fit(df, params=params, rounds=500)
    ratio = df["posted_rate"] / predict(model, df)
    return (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    ape = np.abs(predicted - actual) / actual
    return {
        "MAE": float(np.mean(np.abs(predicted - actual))),
        "MAPE_%": float(100 * ape.mean()),
        "median_APE_%": float(100 * np.median(ape)),
    }
