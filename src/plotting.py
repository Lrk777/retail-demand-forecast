"""Reporting plots for demand forecast backtests."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .backtest import OriginResult


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_metric_bars(summary: pd.DataFrame, figures_dir: Path) -> Path:
    _setup_style()
    metrics = ["mae", "mape_positive", "inventory_service_cost"]
    titles = ["MAE (units)", "MAPE on y>0 (%)", "Inventory/service cost"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title in zip(axes, metrics, titles):
        order = summary.sort_values(metric)
        sns.barplot(data=order, x="model", y=metric, ax=ax, color="#2F5D50")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "metric_bars.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_origin_mae(results: pd.DataFrame, figures_dir: Path) -> Path:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_df = results.copy()
    plot_df["origin"] = pd.to_datetime(plot_df["origin"])
    sns.lineplot(data=plot_df, x="origin", y="mae", hue="model", marker="o", ax=ax)
    ax.set_title("Rolling-origin MAE by model")
    ax.set_xlabel("Forecast origin")
    ax.set_ylabel("MAE")
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "rolling_origin_mae.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_example_forecast(
    panel: pd.DataFrame,
    details: list[OriginResult],
    figures_dir: Path,
    model: str = "hist_gradient_boosting",
) -> Path | None:
    """Plot one SKU's history + last-origin forecast vs actual."""
    model_details = [d for d in details if d.model == model]
    if not model_details:
        return None
    last = model_details[-1]
    # pick a mid-volume SKU for readability
    sku_means = (
        pd.DataFrame({"sku": last.sku, "y": last.y_true})
        .groupby("sku")["y"]
        .mean()
        .sort_values()
    )
    sku = sku_means.index[len(sku_means) // 2]
    mask = last.sku == sku
    hist = panel[panel["sku"] == sku].sort_values("date")
    # show last ~90 days
    cutoff = pd.Timestamp(last.origin_date) - pd.Timedelta(days=90)
    hist = hist[hist["date"] >= cutoff]

    _setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist["date"], hist["demand"], color="#4A4A4A", label="Actual", linewidth=1.6)
    ax.plot(
        last.dates[mask],
        last.y_true[mask],
        color="#4A4A4A",
        linewidth=1.6,
    )
    ax.plot(
        last.dates[mask],
        last.y_pred[mask],
        color="#C45C26",
        marker="o",
        label=f"{model} forecast",
        linewidth=2,
    )
    ax.axvline(last.origin_date, color="#888888", linestyle="--", linewidth=1, label="Origin")
    ax.set_title(f"SKU {sku}: history and {model} forecast")
    ax.set_ylabel("Daily demand")
    ax.legend(fontsize=9)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "example_forecast.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seasonality(panel: pd.DataFrame, figures_dir: Path) -> Path:
    _setup_style()
    tmp = panel.copy()
    tmp["dow"] = tmp["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.barplot(
        data=tmp,
        x="dow",
        y="demand",
        order=order,
        estimator=np.mean,
        errorbar=None,
        color="#2F5D50",
        ax=ax,
    )
    ax.set_title("Average daily demand by weekday (panel)")
    ax.set_xlabel("")
    ax.set_ylabel("Mean demand")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "weekday_seasonality.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
