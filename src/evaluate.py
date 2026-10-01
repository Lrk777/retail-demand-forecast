"""CLI: pretty-print the latest backtest metrics table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print demand forecast metrics")
    parser.add_argument("--run-dir", default="reports/latest")
    args = parser.parse_args(argv)

    root = _project_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}; run: python -m src.train")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)

    print(f"Dataset: {payload['dataset']}")
    print(f"Window:  {payload['date_min']} → {payload['date_max']}")
    print(f"SKUs:    {payload['n_skus']}  | origins: {payload['n_origins']}  | H={payload['horizon']}")
    print(f"Costs:   underage={payload['business_cost']['underage_cost']}  "
          f"overage={payload['business_cost']['overage_cost']}")
    print()
    cols = ["model", "mae", "mape_positive", "smape", "inventory_service_cost"]
    view = summary[cols].copy()
    for c in ["mae", "mape_positive", "smape", "inventory_service_cost"]:
        view[c] = view[c].map(lambda x: f"{x:8.3f}")
    print(view.to_string(index=False))
    print(f"\n* best by inventory/service cost: {payload['best_model_by_inventory_service_cost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
