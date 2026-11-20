"""Turn forecasts into inventory decisions: safety stock, reorder point, order quantity."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from scipy.stats import norm


@dataclass
class ReorderPolicy:
    safety_stock: float
    reorder_point: float
    eoq: float


def reorder_policy(daily_forecast: float, daily_std: float, lead_time_days: float, service_level: float = 0.95,
                   order_cost: float = 50.0, unit_cost: float = 10.0, holding_rate: float = 0.25) -> ReorderPolicy:
    """Classic (s, Q) policy.

    safety stock = z * sigma_daily * sqrt(lead time)
    reorder point = expected demand during lead time + safety stock
    EOQ = sqrt(2 * annual demand * order cost / annual holding cost per unit)
    """
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1")
    z = norm.ppf(service_level)
    ss = z * daily_std * sqrt(lead_time_days)
    rop = daily_forecast * lead_time_days + ss
    annual = daily_forecast * 365
    eoq = sqrt(2 * annual * order_cost / (unit_cost * holding_rate)) if annual > 0 else 0.0
    return ReorderPolicy(round(ss, 1), round(rop, 1), round(eoq, 1))
