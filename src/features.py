"""Calendar, lag, and rolling features for demand forecasting (leakage-safe)."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "dow",
    "weekofyear",
    "month",
    "is_weekend",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("sku")["demand"].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling stats use only past demand (shift 1 before rolling)."""
    out = df.copy()
    grouped = out.groupby("sku")["demand"]
    shifted = grouped.shift(1)
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.groupby(out["sku"]).transform(
            lambda s, window=w: s.rolling(window, min_periods=max(1, window // 2)).mean()
        )
        if w == 7:
            out[f"roll_std_{w}"] = shifted.groupby(out["sku"]).transform(
                lambda s, window=w: s.rolling(window, min_periods=max(1, window // 2)).std()
            )
    return out


def build_feature_frame(
    panel: pd.DataFrame,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    lags = lags or [1, 7, 14, 28]
    roll_windows = roll_windows or [7, 14, 28]
    out = panel.sort_values(["sku", "date"]).reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_lag_features(out, lags)
    out = add_rolling_features(out, roll_windows)
    return out


def drop_warmup(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLUMNS
    return df.dropna(subset=cols).reset_index(drop=True)
