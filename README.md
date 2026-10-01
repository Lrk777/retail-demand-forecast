# retail-demand-forecast

SKU-level retail demand forecasting with **rolling-origin backtests**, seasonal baselines, gradient boosting, and a **proxy inventory/service cost** metric — aimed at analytics / new-grad roles in CTC, retail, and ops.

## Problem

Given daily unit sales for regularly selling SKUs, can a lag-and-seasonality feature model beat naive baselines on a walk-forward protocol — and does the winner change when we score with an asymmetric stockout vs holding cost instead of MAE alone?

## Data source

**UCI Online Retail** (dataset 352) — a UK-centric e-commerce transaction extract (Dec 2010–Dec 2011).

| | |
|--|--|
| Archive | https://archive.ics.uci.edu/dataset/352/online+retail |
| Zip URL | https://archive.ics.uci.edu/static/public/352/online+retail.zip |
| Citation | Chen, D. S., Sibley, C., & Blundell, J. Online Retail. UCI Machine Learning Repository. |

**This repo’s panel** (checked in at `data/processed/uk_sku_daily.csv`):

1. Keep **United Kingdom** invoices only; drop cancellations (`InvoiceNo` starting with `C`) and non-positive quantities.
2. Aggregate to **daily units by `StockCode` (SKU)**.
3. Select **20 regularly selling SKUs** (≥150 selling days; filter extreme one-day spikes).
4. **Zero-fill** missing calendar days (no invoice that day → demand 0).

This is a dense SKU demand panel suitable for supply-chain style forecasting demos. It is **not** M5 store-level POS; see Limitations.

## Demo

```text
$ python -m src.train --config configs/default.yaml
$ python -m src.evaluate --run-dir reports/latest

Dataset: UCI Online Retail (dataset 352)
Window:  2010-12-01 → 2011-12-09
SKUs:    20  | origins: 8  | H=7
Costs:   underage=5.0  overage=1.0

                 model      mae mape_positive    smape inventory_service_cost
hist_gradient_boosting   55.721       231.344   83.713                173.128 *
        moving_average   57.864       185.248   93.158                183.337
        seasonal_naive   59.966       174.920   73.783                189.645
                 naive   84.612       287.465  103.619                223.879
```

Figures in `reports/figures/`: weekday seasonality, metric bars, rolling-origin MAE, example SKU forecast.

## Method

1. **Features (leakage-safe):** lags `{1,7,14,28}`, rolling mean/std on **shifted** history, calendar (`dow`, week, month, weekend).
2. **Baselines:** last-value naive, **seasonal naive (period=7)**, 7-day moving average — flat or seasonal forecasts from history ≤ origin.
3. **Model:** `HistGradientBoostingRegressor` (sklearn) on the feature matrix; predictions clipped at 0.
4. **Evaluation:** **rolling-origin / walk-forward** — 8 origins, horizon **H=7**, step 7, min train 120 days. Metrics averaged across origins.
5. **Business metric:** mean newsvendor-style unit cost  
   `underage_cost * max(y − ŷ, 0) + overage_cost * max(ŷ − y, 0)`  
   with default **underage=5**, **overage=1** (stockouts cost more than excess).

Config: `configs/default.yaml` (seed, SKU filters, horizon, costs, HGBR hyperparameters).

## Results

Checked-in backtest (seed **42**, 8 origins × 7-day horizon):

| Model | MAE ↓ | MAPE (y>0) ↓ | sMAPE ↓ | Inventory/service cost ↓ |
|-------|------:|-------------:|--------:|-------------------------:|
| **hist_gradient_boosting** | **55.7** | 231.3 | 83.7 | **173.1** |
| moving_average | 57.9 | 185.2 | 93.2 | 183.3 |
| seasonal_naive | 60.0 | **174.9** | **73.8** | 189.6 |
| naive | 84.6 | 287.5 | 103.6 | 223.9 |

**Talk track:** Gradient boosting wins on **MAE** and **asymmetric inventory cost**. Seasonal naive wins **MAPE (y>0)** and **sMAPE** on this zero-inflated panel — a useful interview point that the “best” model depends on the decision metric, which is why ops roles often care about stockout-weighted cost, not percentage error alone.

## Stack

- Python 3.11+
- pandas, NumPy, scikit-learn, matplotlib, seaborn, PyYAML, openpyxl, pytest

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m src.train --config configs/default.yaml
python -m src.evaluate --run-dir reports/latest
pytest -q -m "not slow"
```

- First run uses the checked-in panel under `data/processed/`.
- To rebuild from UCI: set `data.force_rebuild: true` in the config (downloads ~23 MB Excel into `data/raw/`).

## Project layout

```text
configs/default.yaml     # seeds, backtest, costs
data/processed/          # UK SKU daily panel (committed)
src/data.py              # UCI download + panel build
src/features.py          # lags / rolling / calendar
src/baselines.py         # naive, seasonal naive, MA
src/models.py            # HistGradientBoosting
src/backtest.py          # rolling-origin harness
src/metrics.py           # MAE, MAPE, sMAPE, inventory cost
src/train.py             # CLI entry
src/evaluate.py          # print metrics table
src/plotting.py          # report figures
reports/figures/         # plots
reports/latest/          # metrics.json, CSVs
tests/                   # unit tests (no network)
```

## Limitations

- **Zero-fill ≠ lost sales.** Missing days are treated as zero demand; true demand under stockouts is unidentified.
- **No Saturday activity** in this UK extract (source quirk) — weekly seasonality is still informative but not a modern omnichannel calendar.
- **Not store-level POS / M5.** One marketplace, SKU panel, ~1 year — good for method, not for claiming production WMAPE.
- **Inventory cost is a proxy**, not a full multi-echelon simulator (no lead times, capacity, or substitution).
- **Intermittent / spiky SKUs** remain hard; MAPE with zeros is misleading — use `mape_positive` / sMAPE / cost.
- Metrics can drift slightly across sklearn minor versions; seed + config are fixed.

## License

MIT for this repository’s code. UCI Online Retail retains its original UCI terms — cite the dataset when publishing results.

## Cite

```text
Chen, D. S., Sibley, C., & Blundell, J. Online Retail.
UCI Machine Learning Repository (dataset 352).
https://archive.ics.uci.edu/dataset/352/online+retail
```
