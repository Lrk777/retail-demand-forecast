"""Forecast accuracy and business-flavored inventory cost metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Mean absolute percentage error with floor eps in the denominator.

    Retail series are often zero-inflated; raw MAPE is undefined at 0.
    Using max(|y|, eps) keeps the metric defined while remaining interpretable
    on positive-demand days. Prefer `mape_positive` or sMAPE when zeros are common.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mape_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE evaluated only on days with strictly positive actual demand."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def inventory_service_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    underage_cost: float = 5.0,
    overage_cost: float = 1.0,
) -> float:
    """
    Proxy expected unit cost under a newsvendor-style asymmetry.

    underage (stockout / missed service): actual > forecast
    overage (holding / excess): forecast > actual

    Mean cost per SKU-day. Not a full inventory simulator — a decision-flavored
    ranking metric for CTC / ops interviews.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    short = np.maximum(y_true - y_pred, 0.0)
    excess = np.maximum(y_pred - y_true, 0.0)
    return float(np.mean(underage_cost * short + overage_cost * excess))


def summarize_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    underage_cost: float = 5.0,
    overage_cost: float = 1.0,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mape_positive": mape_positive(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "inventory_service_cost": inventory_service_cost(
            y_true, y_pred, underage_cost=underage_cost, overage_cost=overage_cost
        ),
    }
