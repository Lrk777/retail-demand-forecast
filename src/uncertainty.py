"""How sure are we that gradient boosting beats the baselines?

Two checks on the rolling-origin backtest:

1. Block bootstrap of the paired error difference. Blocks are (origin, SKU) pairs, so the 7 days a
   model forecasts for one SKU from one origin stay together (their errors are correlated).
2. Cost-ratio sensitivity: which model has the lowest inventory/service cost as the stockout
   penalty moves from 1x to 10x the overstock penalty.

Usage: python -m src.uncertainty --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import load_config, prepare_dataset


def prediction_table(details) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {"origin": d.origin_date, "model": d.model, "sku": d.sku, "date": d.dates,
             "y_true": d.y_true, "y_pred": d.y_pred}
        )
        for d in details
    ]
    return pd.concat(frames, ignore_index=True)


def unit_cost(y_true, y_pred, under: float, over: float) -> np.ndarray:
    short = np.maximum(y_true - y_pred, 0.0)
    excess = np.maximum(y_pred - y_true, 0.0)
    return under * short + over * excess


def paired_bootstrap(
    preds: pd.DataFrame, model: str, baseline: str, metric: str, cost: dict,
    n_boot: int = 2000, seed: int = 42,
) -> dict:
    """95% interval for mean(metric[model] - metric[baseline]) over SKU-days. Negative = model better."""
    key = ["origin", "sku", "date"]
    a = preds[preds.model == model].set_index(key)
    b = preds[preds.model == baseline].set_index(key)
    j = a.join(b, lsuffix="_m", rsuffix="_b", how="inner")
    if metric == "mae":
        diff = np.abs(j.y_true_m - j.y_pred_m) - np.abs(j.y_true_b - j.y_pred_b)
    else:
        u, o = float(cost["underage_cost"]), float(cost["overage_cost"])
        diff = unit_cost(j.y_true_m, j.y_pred_m, u, o) - unit_cost(j.y_true_b, j.y_pred_b, u, o)
    d = diff.reset_index()
    d["block"] = d.origin.astype(str) + "|" + d.sku.astype(str)
    sums = d.groupby("block")[0].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    s, c = sums["sum"].to_numpy(), sums["count"].to_numpy()
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "model": model, "baseline": baseline, "metric": metric,
        "mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "share_of_boots_model_better": float((boots < 0).mean()),
        "n_sku_days": int(len(d)), "n_blocks": int(len(sums)),
    }


def cost_sensitivity(preds: pd.DataFrame, ratios=range(1, 11)) -> pd.DataFrame:
    rows = []
    for r in ratios:
        for m, g in preds.groupby("model"):
            rows.append({"underage_to_overage": r, "model": m,
                         "mean_cost": float(unit_cost(g.y_true, g.y_pred, r, 1.0).mean())})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("underage_to_overage")["mean_cost"].rank(method="first")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    panel = prepare_dataset(cfg, project_root=root)
    _, details = run_backtest(panel, cfg)
    preds = prediction_table(details)

    out_dir = root / cfg["reports"]["run_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = [
        paired_bootstrap(preds, "hist_gradient_boosting", b, m, cfg["business_cost"], args.n_boot, cfg["seed"])
        for b in ["moving_average", "seasonal_naive", "naive"]
        for m in ["mae", "inventory_service_cost"]
    ]
    sens = cost_sensitivity(preds)
    sens.to_csv(out_dir / "cost_ratio_sensitivity.csv", index=False)
    winners = sens[sens["rank"] == 1][["underage_to_overage", "model"]].to_dict(orient="records")
    (out_dir / "uncertainty.json").write_text(
        json.dumps({"bootstrap_95ci": tests, "winner_by_cost_ratio": winners}, indent=2), encoding="utf-8"
    )

    from .plotting import plt  # reuse the repo's matplotlib setup

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "lrk"})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, g in sens.groupby("model"):
        ax.plot(g.underage_to_overage, g.mean_cost, marker="o", ms=4, lw=2, label=m.replace("_", " "))
    ax.set_xlabel("Stockout cost ÷ overstock cost per unit")
    ax.set_ylabel("Mean cost per SKU-day")
    ax.set_title("Which model is cheapest as stockouts get more expensive", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(root / cfg["reports"]["figures_dir"] / "cost_ratio_sensitivity.svg", dpi=150)

    for t in tests:
        print(f"{t['metric']:>24} HGB − {t['baseline']:<15} {t['mean_diff']:8.2f}  "
              f"95% CI [{t['ci_low']:.2f}, {t['ci_high']:.2f}]  P(HGB better)={t['share_of_boots_model_better']:.3f}")
    print("Winner by cost ratio:", {w["underage_to_overage"]: w["model"] for w in winners})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
