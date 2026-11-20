"""Synthetic daily demand for a set of SKUs: trend + weekly/annual seasonality + promotions + noise."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demand(n_skus: int = 12, start: str = "2023-01-01", periods: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")
    t = np.arange(periods)
    frames = []
    for i in range(n_skus):
        base = rng.uniform(20, 200)
        trend = rng.uniform(-0.02, 0.08) * t
        weekly = base * rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * (dates.dayofweek.to_numpy() - rng.integers(0, 7)) / 7)
        annual = base * rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - rng.integers(0, 365)) / 365.25)
        promo = rng.random(periods) < 0.04
        lift = np.where(promo, base * rng.uniform(0.3, 0.8), 0)
        noise = rng.normal(0, base * 0.08, periods)
        demand = np.clip(base + trend + weekly + annual + lift + noise, 0, None).round()
        frames.append(pd.DataFrame({"date": dates, "sku": f"SKU-{i + 1:03d}", "units": demand.astype(int),
                                    "promo": promo, "unit_cost": round(float(rng.uniform(5, 120)), 2)}))
    return pd.concat(frames, ignore_index=True)
