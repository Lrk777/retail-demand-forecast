"""Tests for src/uncertainty.py."""


def test_paired_bootstrap_detects_a_clear_winner():
    import numpy as np
    import pandas as pd

    from src.uncertainty import cost_sensitivity, paired_bootstrap

    rng = np.random.default_rng(0)
    rows = []
    for o in range(4):
        for s in range(10):
            for d in range(7):
                y = float(rng.poisson(20))
                rows.append({"origin": o, "sku": s, "date": d, "model": "good", "y_true": y, "y_pred": y + rng.normal(0, 1)})
                rows.append({"origin": o, "sku": s, "date": d, "model": "bad", "y_true": y, "y_pred": y + rng.normal(0, 8)})
    preds = pd.DataFrame(rows)
    r = paired_bootstrap(preds, "good", "bad", "mae", {"underage_cost": 5, "overage_cost": 1}, n_boot=300)
    assert r["ci_high"] < 0 and r["n_blocks"] == 40
    sens = cost_sensitivity(preds, ratios=[1, 5])
    assert set(sens[sens["rank"] == 1].model) == {"good"}
