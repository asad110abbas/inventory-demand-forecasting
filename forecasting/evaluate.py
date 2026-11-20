"""Rolling-origin backtesting and error metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))))


def bias(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float((pred - actual).sum() / max(actual.sum(), 1e-9))


def backtest(series: pd.DataFrame, model_cls, folds: int = 8, horizon: int = 7) -> pd.DataFrame:
    """Train on everything before each cut-off, forecast the next `horizon` days."""
    rows = []
    n = len(series)
    for k in range(folds, 0, -1):
        cut = n - k * horizon
        train, test = series.iloc[:cut], series.iloc[cut: cut + horizon]
        pred = model_cls().fit(train).predict(test[["date", "promo"]])
        rows.append({"model": model_cls.name, "cutoff": train["date"].iloc[-1],
                     "wape": wape(test["units"], pred), "mae": mae(test["units"], pred),
                     "bias": bias(test["units"], pred)})
    return pd.DataFrame(rows)
