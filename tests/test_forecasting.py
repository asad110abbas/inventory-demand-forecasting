import numpy as np
import pandas as pd
import pytest

from forecasting.data import make_demand
from forecasting.evaluate import backtest, wape
from forecasting.inventory import reorder_policy
from forecasting.models import MODELS, GradientBoosting


@pytest.fixture(scope="module")
def series():
    d = make_demand(n_skus=1, periods=400, seed=1)
    return d.sort_values("date").reset_index(drop=True)


def test_demand_shape_and_non_negative():
    d = make_demand(n_skus=3, periods=100)
    assert len(d) == 300 and (d["units"] >= 0).all()


def test_wape_perfect_and_known():
    assert wape([10, 10], [10, 10]) == 0
    assert wape([10, 10], [5, 15]) == pytest.approx(0.5)


@pytest.mark.parametrize("model_cls", MODELS)
def test_each_model_forecasts_7_days(series, model_cls):
    future = pd.DataFrame({"date": pd.date_range(series["date"].iloc[-1] + pd.Timedelta(days=1), periods=7),
                           "promo": False})
    pred = model_cls().fit(series).predict(future)
    assert len(pred) == 7 and np.all(pred >= 0)


def test_gradient_boosting_rejects_long_horizon(series):
    future = pd.DataFrame({"date": pd.date_range("2030-01-01", periods=8), "promo": False})
    with pytest.raises(ValueError):
        GradientBoosting().fit(series).predict(future)


def test_backtest_returns_one_row_per_fold(series):
    res = backtest(series, MODELS[0], folds=4)
    assert len(res) == 4 and set(res.columns) >= {"wape", "mae", "bias"}


def test_reorder_policy_math():
    p = reorder_policy(daily_forecast=100, daily_std=20, lead_time_days=4, service_level=0.95)
    # z(0.95) ~= 1.645 -> SS = 1.645 * 20 * 2 = 65.8
    assert p.safety_stock == pytest.approx(65.8, abs=0.2)
    assert p.reorder_point == pytest.approx(465.8, abs=0.2)
    assert p.eoq > 0
