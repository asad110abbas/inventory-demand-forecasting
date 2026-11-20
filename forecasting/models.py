"""Three forecasters with a common interface: fit(history) -> predict(horizon dates)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import add_features, feature_columns


class SeasonalNaive:
    """Baseline: demand on the same weekday last week."""
    name = "seasonal_naive"

    def fit(self, hist: pd.DataFrame):
        self.last_week = hist["units"].to_numpy()[-7:]
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        reps = int(np.ceil(len(future) / 7))
        return np.tile(self.last_week, reps)[: len(future)].astype(float)


class HoltWinters:
    """Additive trend + weekly seasonality exponential smoothing."""
    name = "holt_winters"

    def fit(self, hist: pd.DataFrame):
        self.model = ExponentialSmoothing(hist["units"].astype(float).to_numpy(), trend="add",
                                          seasonal="add", seasonal_periods=7).fit()
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.forecast(len(future)), 0, None)


class GradientBoosting:
    """Gradient-boosted trees on calendar, promo, lag and rolling features (direct 7-day-ahead)."""
    name = "gradient_boosting"

    def fit(self, hist: pd.DataFrame):
        feats = add_features(hist).dropna()
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        self.model.fit(feats[feature_columns()], feats["units"])
        self.hist = hist
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        if len(future) > 7:
            raise ValueError("lag features support a horizon of at most 7 days")
        full = pd.concat([self.hist, future.assign(units=np.nan)], ignore_index=True)
        feats = add_features(full).iloc[-len(future):]
        return np.clip(self.model.predict(feats[feature_columns()]), 0, None)


MODELS = [SeasonalNaive, HoltWinters, GradientBoosting]
