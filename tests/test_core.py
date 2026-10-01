"""Unit tests for features, metrics, and rolling-origin helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import rolling_origins
from src.baselines import moving_average_forecast, naive_forecast, seasonal_naive_forecast
from src.features import FEATURE_COLUMNS, build_feature_frame, drop_warmup
from src.metrics import inventory_service_cost, mae, mape, mape_positive, summarize_errors


def _toy_panel(n_days: int = 60, n_skus: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for sku in [f"S{i}" for i in range(n_skus)]:
        # weekly seasonality + noise
        base = 20 + 10 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
        demand = np.maximum(base + rng.normal(0, 2, size=n_days), 0)
        for d, y in zip(dates, demand):
            rows.append({"date": d, "sku": sku, "demand": float(y)})
    return pd.DataFrame(rows)


def test_lag_features_are_shifted_not_current():
    panel = _toy_panel()
    feat = build_feature_frame(panel)
    # For each sku, lag_1[t] == demand[t-1]
    for sku, g in feat.groupby("sku"):
        g = g.sort_values("date")
        assert np.allclose(g["lag_1"].iloc[1:].to_numpy(), g["demand"].iloc[:-1].to_numpy())


def test_rolling_mean_uses_past_only():
    panel = _toy_panel(n_skus=1)
    feat = build_feature_frame(panel)
    g = feat.sort_values("date")
    # roll_mean_7 at t uses demand[t-7:t] (shifted), not demand[t]
    t = 20
    expected = g["demand"].iloc[t - 7 : t].mean()  # indices t-7..t-1 inclusive = 7 values
    assert abs(g["roll_mean_7"].iloc[t] - expected) < 1e-9


def test_drop_warmup_removes_nan_feature_rows():
    panel = _toy_panel()
    feat = build_feature_frame(panel)
    clean = drop_warmup(feat)
    assert clean[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert len(clean) < len(feat)


def test_baselines_shapes_and_naive_value():
    hist = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert naive_forecast(hist, 3).tolist() == [7.0, 7.0, 7.0]
    seas = seasonal_naive_forecast(hist, 3, season=7)
    assert seas.tolist() == [1.0, 2.0, 3.0]
    ma = moving_average_forecast(hist, 2, window=3)
    assert ma.tolist() == [pytest.approx(6.0), pytest.approx(6.0)]


def test_inventory_cost_asymmetric():
    y = np.array([10.0, 10.0])
    # under-forecast by 2 vs over-forecast by 2
    under = inventory_service_cost(y, np.array([8.0, 8.0]), underage_cost=5.0, overage_cost=1.0)
    over = inventory_service_cost(y, np.array([12.0, 12.0]), underage_cost=5.0, overage_cost=1.0)
    assert under == pytest.approx(10.0)  # 5 * 2
    assert over == pytest.approx(2.0)  # 1 * 2
    assert under > over


def test_mape_handles_zeros():
    y = np.array([0.0, 10.0])
    p = np.array([1.0, 10.0])
    # with eps=1, first term |0-1|/1=1, second 0 → mean 0.5 → 50%
    assert mape(y, p, eps=1.0) == pytest.approx(50.0)


def test_rolling_origins_monotonic_and_count():
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    origins = rolling_origins(dates, horizon=7, n_origins=5, step=7, min_train_days=100)
    assert len(origins) == 5
    assert origins == sorted(origins)
    # each origin leaves room for horizon
    assert origins[-1] <= dates[-1] - pd.Timedelta(days=7)


def test_summarize_errors_keys():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.5, 2.0])
    out = summarize_errors(y, p)
    assert set(out) == {
        "mae",
        "mape",
        "mape_positive",
        "smape",
        "inventory_service_cost",
    }
    assert out["mae"] == pytest.approx(mae(y, p))
    assert out["mape_positive"] == pytest.approx(mape_positive(y, p))


@pytest.mark.slow
def test_end_to_end_backtest_smoke():
    from pathlib import Path

    from src.backtest import aggregate_results, run_backtest
    from src.data import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    cfg = {
        **cfg,
        "backtest": {**cfg["backtest"], "n_origins": 2, "horizon": 7, "min_train_days": 40},
        "baselines": ["naive", "seasonal_naive"],
        "models": {
            "hist_gradient_boosting": {
                **cfg["models"]["hist_gradient_boosting"],
                "max_iter": 40,
            }
        },
    }
    panel = _toy_panel(n_days=80, n_skus=3)
    results, details = run_backtest(panel, cfg)
    summary = aggregate_results(results)
    assert not summary.empty
    assert "hist_gradient_boosting" in set(summary["model"])
    assert details
