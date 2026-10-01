"""Rolling-origin (walk-forward) backtesting for panel demand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import BASELINE_FNS
from .features import FEATURE_COLUMNS, build_feature_frame, drop_warmup
from .metrics import summarize_errors
from .models import make_hgbr


@dataclass
class OriginResult:
    origin_date: pd.Timestamp
    model: str
    y_true: np.ndarray
    y_pred: np.ndarray
    sku: np.ndarray
    dates: np.ndarray


def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(panel["date"].unique()))


def rolling_origins(
    dates: pd.DatetimeIndex,
    horizon: int,
    n_origins: int,
    step: int,
    min_train_days: int,
) -> list[pd.Timestamp]:
    """
    Choose forecast origins so each origin has >= min_train_days of history
    and horizon days of future labels after the origin day.
    Origin is the last training day (inclusive).
    """
    if len(dates) < min_train_days + horizon:
        raise ValueError("Not enough dates for configured backtest")
    last_origin_idx = len(dates) - 1 - horizon
    first_origin_idx = min_train_days - 1
    origins: list[pd.Timestamp] = []
    idx = last_origin_idx
    while idx >= first_origin_idx and len(origins) < n_origins:
        origins.append(pd.Timestamp(dates[idx]))
        idx -= step
    origins.reverse()
    if not origins:
        raise ValueError("No valid rolling origins under current config")
    return origins


def _baseline_predictions(
    hist: pd.DataFrame,
    future: pd.DataFrame,
    name: str,
    horizon: int,
) -> np.ndarray:
    preds = []
    for sku, fut in future.groupby("sku", sort=False):
        h = hist.loc[hist["sku"] == sku, "demand"]
        fn = BASELINE_FNS[name]
        pred = fn(h, horizon=len(fut))
        preds.append(pd.Series(pred, index=fut.index))
    return pd.concat(preds).sort_index().to_numpy(dtype=float)


def _fit_predict_hgbr(
    feat: pd.DataFrame,
    origin: pd.Timestamp,
    future_idx: pd.Index,
    seed: int,
    model_params: dict,
) -> np.ndarray:
    train = feat[feat["date"] <= origin]
    test = feat.loc[future_idx]
    X_train = train[FEATURE_COLUMNS]
    y_train = train["demand"].to_numpy(dtype=float)
    model = make_hgbr(model_params, seed=seed)
    model.fit(X_train, y_train)
    pred = model.predict(test[FEATURE_COLUMNS])
    return np.maximum(pred, 0.0)  # demand cannot be negative


def run_backtest(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[OriginResult]]:
    """
    Rolling-origin backtest across baselines + hist gradient boosting.

    Features for boosting are built on the full panel with lag/rolling shifts,
    so rows after `origin` only use information from <= origin via lag construction.
    Baselines use raw history up to origin only.
    """
    seed = int(cfg["seed"])
    bt = cfg["backtest"]
    feat_cfg = cfg["features"]
    cost = cfg["business_cost"]
    model_params = cfg["models"]["hist_gradient_boosting"]

    feat = build_feature_frame(
        panel,
        lags=list(feat_cfg["lags"]),
        roll_windows=list(feat_cfg["roll_windows"]),
    )
    feat = drop_warmup(feat)

    dates = _panel_dates(panel)
    origins = rolling_origins(
        dates,
        horizon=int(bt["horizon"]),
        n_origins=int(bt["n_origins"]),
        step=int(bt["step"]),
        min_train_days=int(bt["min_train_days"]),
    )

    model_names = list(cfg["baselines"]) + ["hist_gradient_boosting"]
    rows: list[dict] = []
    details: list[OriginResult] = []

    for origin in origins:
        future_mask = (panel["date"] > origin) & (
            panel["date"] <= origin + pd.Timedelta(days=int(bt["horizon"]))
        )
        future = panel.loc[future_mask].sort_values(["sku", "date"])
        if future.empty:
            continue
        hist = panel[panel["date"] <= origin]
        future_feat = feat[
            (feat["date"] > origin)
            & (feat["date"] <= origin + pd.Timedelta(days=int(bt["horizon"])))
        ].sort_values(["sku", "date"])

        for name in model_names:
            if name == "hist_gradient_boosting":
                y_pred = _fit_predict_hgbr(
                    feat, origin, future_feat.index, seed=seed, model_params=model_params
                )
                y_true = future_feat["demand"].to_numpy(dtype=float)
                skus = future_feat["sku"].to_numpy()
                dts = future_feat["date"].to_numpy()
            else:
                y_pred = _baseline_predictions(hist, future, name, horizon=int(bt["horizon"]))
                y_true = future["demand"].to_numpy(dtype=float)
                skus = future["sku"].to_numpy()
                dts = future["date"].to_numpy()

            metrics = summarize_errors(
                y_true,
                y_pred,
                underage_cost=float(cost["underage_cost"]),
                overage_cost=float(cost["overage_cost"]),
            )
            rows.append(
                {
                    "origin": origin,
                    "model": name,
                    **metrics,
                    "n": int(len(y_true)),
                }
            )
            details.append(
                OriginResult(
                    origin_date=origin,
                    model=name,
                    y_true=y_true,
                    y_pred=y_pred,
                    sku=skus,
                    dates=dts,
                )
            )

    results = pd.DataFrame(rows)
    return results, details


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics across origins, weighted equally per origin."""
    return (
        results.groupby("model", as_index=False)[
            ["mae", "mape", "mape_positive", "smape", "inventory_service_cost"]
        ]
        .mean()
        .sort_values("inventory_service_cost")
        .reset_index(drop=True)
    )
