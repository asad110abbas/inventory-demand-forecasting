"""Run the full pipeline: generate data -> backtest models -> pick best per SKU -> reorder plan.

    python -m forecasting.cli --skus 12 --out reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import make_demand
from .evaluate import backtest
from .inventory import reorder_policy
from .models import MODELS


def run(n_skus: int, out: Path, lead_time: int = 10, service_level: float = 0.95) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    demand = make_demand(n_skus=n_skus)
    demand.to_csv(out / "demand.csv", index=False)

    scores, plan = [], []
    for sku, s in demand.groupby("sku"):
        s = s.sort_values("date").reset_index(drop=True)
        res = pd.concat([backtest(s, m) for m in MODELS])
        res.insert(0, "sku", sku)
        scores.append(res)
        summary = res.groupby("model")["wape"].mean()
        best = summary.idxmin()
        model_cls = next(m for m in MODELS if m.name == best)
        future = pd.DataFrame({"date": pd.date_range(s["date"].iloc[-1] + pd.Timedelta(days=1), periods=7),
                               "promo": False})
        fc = model_cls().fit(s).predict(future)
        daily_std = s["units"].tail(56).std()
        pol = reorder_policy(fc.mean(), daily_std, lead_time, service_level, unit_cost=s["unit_cost"].iloc[0])
        plan.append({"sku": sku, "best_model": best, "wape": round(summary[best], 3),
                     "next7_forecast": int(fc.sum()), "safety_stock": pol.safety_stock,
                     "reorder_point": pol.reorder_point, "eoq": pol.eoq})

    scores = pd.concat(scores)
    scores.to_csv(out / "backtest_scores.csv", index=False)
    plan = pd.DataFrame(plan)
    plan.to_csv(out / "reorder_plan.csv", index=False)
    leaderboard = scores.groupby("model")[["wape", "mae", "bias"]].mean().sort_values("wape").round(3)
    (out / "leaderboard.md").write_text(leaderboard.to_markdown() if hasattr(leaderboard, "to_markdown")
                                        else leaderboard.to_string(), encoding="utf-8")
    print(leaderboard)
    print(plan)
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skus", type=int, default=12)
    ap.add_argument("--lead-time", type=int, default=10)
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    a = ap.parse_args()
    run(a.skus, a.out, a.lead_time, a.service_level)


if __name__ == "__main__":
    main()
