"""CLI: prepare data, run rolling-origin backtest, write metrics + figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .backtest import aggregate_results, run_backtest
from .data import load_config, prepare_dataset
from .plotting import (
    plot_example_forecast,
    plot_metric_bars,
    plot_origin_mae,
    plot_seasonality,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retail demand forecast backtest")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    args = parser.parse_args(argv)

    root = _project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)

    panel = prepare_dataset(cfg, project_root=root)
    results, details = run_backtest(panel, cfg)
    summary = aggregate_results(results)

    run_dir = root / cfg["reports"]["run_dir"]
    fig_dir = root / cfg["reports"]["figures_dir"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(run_dir / "backtest_by_origin.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    shutil.copy(cfg_path, run_dir / "config.yaml")

    best = summary.iloc[0]["model"]
    payload = {
        "dataset": cfg["data"]["source"],
        "source_url": cfg["data"]["source_url"],
        "n_skus": int(panel["sku"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "n_origins": int(results["origin"].nunique()),
        "horizon": int(cfg["backtest"]["horizon"]),
        "best_model_by_inventory_service_cost": best,
        "summary": summary.to_dict(orient="records"),
        "business_cost": cfg["business_cost"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_metric_bars(summary, fig_dir)
    plot_origin_mae(results, fig_dir)
    plot_seasonality(panel, fig_dir)
    plot_example_forecast(panel, details, fig_dir, model=best)

    print("\nRolling-origin summary (mean across origins)")
    print(summary.to_string(index=False))
    print(f"\nBest by inventory/service cost: {best}")
    print(f"Wrote {run_dir / 'metrics.json'}")
    print(f"Figures → {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
