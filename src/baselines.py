"""Naive and seasonal baselines for demand."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed demand for `horizon` steps."""
    if history.empty:
        return np.zeros(horizon)
    last = float(history.iloc[-1])
    return np.full(horizon, last, dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the value from the same weekday `season` days ago, cycling forward."""
    if len(history) < season:
        return naive_forecast(history, horizon)
    tail = history.iloc[-season:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / season))
    return np.tile(tail, reps)[:horizon]


def moving_average_forecast(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Flat forecast equal to the trailing mean."""
    if history.empty:
        return np.zeros(horizon)
    w = min(window, len(history))
    level = float(history.iloc[-w:].mean())
    return np.full(horizon, level, dtype=float)


BASELINE_FNS = {
    "naive": naive_forecast,
    "seasonal_naive": lambda h, horizon: seasonal_naive_forecast(h, horizon, season=7),
    "moving_average": lambda h, horizon: moving_average_forecast(h, horizon, window=7),
}
