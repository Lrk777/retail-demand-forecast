"""Load and prepare the UCI Online Retail SKU demand panel."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_uci_online_retail(raw_dir: Path, url: str = UCI_ZIP_URL) -> Path:
    """Download the UCI Online Retail zip and extract the Excel workbook."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "online_retail.zip"
    xlsx_path = raw_dir / "Online Retail.xlsx"
    if xlsx_path.exists():
        return xlsx_path
    with urlopen(url, timeout=180) as resp:
        zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    if not xlsx_path.exists():
        # Handle alternate casing / nested paths from the archive.
        matches = list(raw_dir.rglob("*.xlsx"))
        if not matches:
            raise FileNotFoundError("Online Retail.xlsx not found after extract")
        return matches[0]
    return xlsx_path


def _clean_transactions(df: pd.DataFrame, country: str) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Country"] == country]
    out = out[out["Quantity"] > 0]
    out = out[~out["InvoiceNo"].astype(str).str.startswith("C")]
    out = out.dropna(subset=["StockCode", "InvoiceDate"])
    out["date"] = pd.to_datetime(out["InvoiceDate"]).dt.normalize()
    out["sku"] = out["StockCode"].astype(str)
    return out


def build_sku_panel(
    transactions: pd.DataFrame,
    n_skus: int = 20,
    min_selling_days: int = 150,
    max_spike_ratio: float = 30.0,
) -> pd.DataFrame:
    """
    Aggregate to daily units by SKU, keep regularly selling SKUs, zero-fill calendar.

    Zero-fill treats missing invoice days as zero demand for the panel SKUs.
    That is a modeling choice (common for dense retail panels) and understates
    true demand if stockouts prevented sales — see README limitations.
    """
    daily = (
        transactions.groupby(["date", "sku"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "demand"})
    )
    stats = daily.groupby("sku").agg(
        selling_days=("date", "nunique"),
        total=("demand", "sum"),
        mean=("demand", "mean"),
        mx=("demand", "max"),
    )
    regular = stats[
        (stats["selling_days"] >= min_selling_days)
        & (stats["mx"] < stats["mean"] * max_spike_ratio)
    ].sort_values(["selling_days", "total"], ascending=False)
    if regular.empty:
        raise ValueError("No SKUs met regularity filters; relax config thresholds")
    skus = regular.head(n_skus).index.tolist()
    panel = daily[daily["sku"].isin(skus)].copy()
    dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([dates, skus], names=["date", "sku"])
    full = (
        panel.set_index(["date", "sku"])
        .reindex(idx, fill_value=0)
        .reset_index()
        .sort_values(["sku", "date"])
        .reset_index(drop=True)
    )
    full["demand"] = full["demand"].astype(float)
    return full


def prepare_dataset(cfg: dict, project_root: Path | None = None) -> pd.DataFrame:
    """Return the processed panel, rebuilding from UCI if needed."""
    root = Path(project_root) if project_root else Path.cwd()
    data_cfg = cfg["data"]
    processed = root / data_cfg["processed_path"]
    if processed.exists() and not data_cfg.get("force_rebuild", False):
        panel = pd.read_csv(processed, parse_dates=["date"])
        return panel.sort_values(["sku", "date"]).reset_index(drop=True)

    raw_dir = root / data_cfg["raw_dir"]
    xlsx = download_uci_online_retail(raw_dir, url=data_cfg.get("source_url", UCI_ZIP_URL))
    raw = pd.read_excel(xlsx)
    clean = _clean_transactions(raw, country=data_cfg["country"])
    panel = build_sku_panel(
        clean,
        n_skus=int(data_cfg["n_skus"]),
        min_selling_days=int(data_cfg["min_selling_days"]),
        max_spike_ratio=float(data_cfg["max_spike_ratio"]),
    )
    processed.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed, index=False)
    return panel
