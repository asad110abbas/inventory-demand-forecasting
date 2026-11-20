"""Feature engineering for the ML forecaster: calendar features, lags and rolling statistics."""
from __future__ import annotations

import pandas as pd

LAGS = (7, 14, 28)
WINDOWS = (7, 28)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects one SKU's history sorted by date with columns date, units, promo."""
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    for lag in LAGS:
        out[f"lag_{lag}"] = out["units"].shift(lag)
    for w in WINDOWS:
        # shift(7) so a 7-day-ahead forecast never sees future data
        out[f"roll_mean_{w}"] = out["units"].shift(7).rolling(w).mean()
        out[f"roll_std_{w}"] = out["units"].shift(7).rolling(w).std()
    out["promo"] = out["promo"].astype(int)
    return out


def feature_columns() -> list[str]:
    return (["dow", "month", "dayofyear", "is_weekend", "promo"]
            + [f"lag_{lag}" for lag in LAGS]
            + [f"roll_{s}_{w}" for w in WINDOWS for s in ("mean", "std")])
