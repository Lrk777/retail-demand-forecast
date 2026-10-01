"""Gradient boosting demand model (sklearn HistGradientBoosting)."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor


def make_hgbr(params: dict | None = None, seed: int = 42) -> HistGradientBoostingRegressor:
    p = {
        "max_depth": 6,
        "learning_rate": 0.08,
        "max_iter": 250,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": seed,
    }
    if params:
        p.update(params)
        p["random_state"] = seed
    return HistGradientBoostingRegressor(**p)
