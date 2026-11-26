# Futures Market Data Quality & Analytics Platform

> **v2:** reconciliation, controls, holiday calendar, completeness-aware
> scoring and a redesigned UI. See [CHANGES.md](CHANGES.md).

A business-facing application that ingests historical futures market data
(CSV or Parquet), **identifies data quality issues**, produces **market
analytics** (daily OHLCV bars, rolling 15-minute VWAP) and turns findings into
**insights and suggested controls** for traders, risk managers and analysts.

![layers](https://img.shields.io/badge/tests-62%20passing-brightgreen)

---

## 1. Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py        # real minute bars, vendor daily bars + manifest from Hugging Face
python scripts/make_sample_data.py     # demo CSV with known injected defects -> data/sample/
python scripts/make_recon_copy.py      # demo "downstream copy" with 3 silent differences

streamlit run app/dashboard.py         # UI at http://localhost:8501
python -m pytest                       # unit + integration tests
```

Headless / batch use (no UI):

```bash
PYTHONPATH=src python -m mdq.cli data/sample/dirty_futures_sample.csv --out reports/demo
PYTHONPATH=src python -m mdq.cli data/raw/*.parquet --contracts ESZ25 --start 2025-11-01 --end 2025-11-30
PYTHONPATH=src python -m mdq.cli data/sample/dirty_futures_sample.csv --reconcile-against data/sample/dirty_futures_copy.csv
PYTHONPATH=src python -m mdq.cli data/raw/*.parquet --vendor-daily data/daily/*.parquet --manifest data/manifest/files.csv
```

Outputs: `dq_scorecard.csv`, `dq_issues.csv`, `rejected_records.csv`,
`daily_ohlcv.csv`, `vwap_15m.parquet`, `insights.csv`.

## 2. Data

* **Real data** – [lynx1231/historical-futures-data-sample](https://huggingface.co/datasets/lynx1231/historical-futures-data-sample):
  minute bars for 8 roots (CL, ES, GC, SB, SR3, VX, ZC, ZN). One contract per
  root is downloaded by default (~1.4M rows); `--all` fetches all 40.
* **Demo data** – `data/sample/dirty_futures_sample.csv` is one week of ESZ25
  + CLH26 in the brief's column layout (`Contract, Timestamp, Open, High, Low,
  Close, Volume`) with **known injected defects**. `injected_defects.json` is
  the ground truth; an integration test proves every defect is detected with
  the exact count.

> The real data turned out to be clean on hard validity (no nulls, duplicates
> or OHLC breaches), so the demo file is needed to show the controls working.
> The real data still produces meaningful findings: gaps, outliers and
> liquidity patterns.

## 3. What it does (mapped to the brief)

| Requirement | Implementation |
|---|---|
| Accept CSV or Parquet | `ingestion.read_raw` (file paths or uploads in the UI) |
| Contract, Timestamp, OHLCV | Column aliases map vendor names (e.g. `contract_symbol`, `timestamp_chicago_wall`) to one canonical schema |
| Handle missing / duplicates / malformed | Malformed → **quarantined** with reason code; missing → kept & flagged; duplicates → exact copies dropped, conflicting versions resolved (last-received wins) |
| Daily OHLCV bars | `analytics.daily_ohlcv` – CME **session** trade date (17:00 CT rolls to next day) or calendar date |
| Rolling 15-min VWAP | `analytics.rolling_vwap` – time-based window `(t-15m, t]`, typical price (H+L+C)/3, per contract |
| Filter by contract & date range | `analytics.filter_data`; sidebar filters in UI; `--contracts/--start/--end` in CLI |
| Missing timestamps & gaps | Missing/unparseable timestamps quarantined; `GapRule` finds holes **inside trading hours only** |
| Duplicates | `DUPLICATE_EXACT`, `DUPLICATE_CONFLICT` |
| Invalid prices / volumes | `INVALID_PRICE`, `OHLC_INCONSISTENT`, `INVALID_VOLUME`, `ZERO_VOLUME_PRICE_MOVE`, `TIMESTAMP_INVALID` |
| Statistical outliers | `PRICE_OUTLIER` – robust z-score (rolling median/MAD) with spike-and-revert “bad tick” labelling |
| Recurring patterns | Concentration by contract, time-of-day clustering, recurrence across days, simultaneous cross-contract gaps, contract-lifecycle liquidity |
| Suggest cleansing/validation rules | Playbook of proposed controls + owner, prioritised by severity and evidence |
| UI / dashboard | Streamlit: scorecard, issue charts & log, candlesticks, VWAP chart with DQ markers, insights, downloads |

## 4. Data quality rules

| Rule | Severity | Effect |
|---|---|---|
| `MALFORMED_RECORD` | ERROR | Unparseable row quarantined at ingestion (reason code kept) |
| `MISSING_VALUE` | ERROR (price) / WARNING (volume) | Missing price excluded; missing volume kept |
| `DUPLICATE_EXACT` | WARNING | Extra copy dropped (would double-count volume) |
| `DUPLICATE_CONFLICT` | ERROR | Superseded version excluded; latest version retained |
| `INVALID_PRICE` | ERROR | Price ≤ 0 (switchable for products that can trade negative) |
| `OHLC_INCONSISTENT` | ERROR | High < max(O,C) or Low > min(O,C) |
| `INVALID_VOLUME` | ERROR | Negative or fractional volume |
| `ZERO_VOLUME_PRICE_MOVE` | WARNING | Price range with no volume (indicative prices mixed in?) |
| `TIMESTAMP_INVALID` | ERROR / WARNING | Future-dated (error); Saturday (warning) |
| `DATA_GAP` | WARNING | Missing in-session minutes ≥ threshold (default 30) |
| `PRICE_OUTLIER` | WARNING | Extreme return vs recent behaviour; labelled if it reverses next bar |
| `SCHEDULED_CLOSURE` | INFO | Gap overlapping an exchange holiday / early close: shown, not counted |

**ERROR** records never reach analytics. **WARNING** records are kept but
flagged, because removing real market moves (e.g. around an economic release)
would be worse than flagging them.

## 5. Key design decisions & edge cases

* **Never silently drop data.** Every record is accounted for:
  `received = clean + excluded + quarantined` (tested).
* **Lineage.** Every issue carries `source_file` + `row_id` back to the original line.
* **Timezones.** The sample's ms timestamps are *Chicago wall-clock labels,
  not UTC*. Treating them as UTC would shift every bar by 5–6 hours and break
  daily bars. All times are kept in exchange time; ISO strings with an offset
  are converted to Chicago time.
* **Trading date ≠ calendar date.** CME sessions start at 17:00 CT the previous
  evening, so a Sunday-evening bar belongs to Monday's daily bar.
* **Gaps vs closures.** A gap only counts if it covers minutes when the
  contract *normally trades*. That session is learned per contract from its
  own history (15-min buckets of the week, ≥3 weeks of data), falling back to
  a default Globex calendar. Weekends, the 16:00–17:00 break and grain/softs
  hours are therefore not false alarms.
* **Illiquid ≠ broken.** Deferred contracts often have no trade for a minute.
  Insights separate liquidity-driven gaps from outages (gaps across several
  contracts at once).
* **Outliers in flat markets.** MAD = 0 on a flat contract would make every
  one-tick move “infinitely” extreme; the rule falls back to std-dev and needs
  a minimum absolute move.
* **Conflicting duplicates.** Latest-received wins (vendor correction
  semantics). This is a **business decision to confirm** with data owners.
* **VWAP.** Time-based window, so it never reaches back across a gap or
  session break; zero-volume window → blank rather than a misleading number.
* **Negative prices** are configurable (WTI April 2020, calendar spreads).

## 6. Assumptions & open questions (for the business)

1. Bar timestamps are bar-start labels in exchange-local time.
2. VWAP uses typical price (H+L+C)/3 because only bars, not trades, are available.
3. Which duplicate wins: latest vendor correction, or a designated golden source?
4. Gap tolerance by product/maturity: should deferred months have looser thresholds?
5. Should DQ scores gate downstream publication (e.g. block if < 99%)?
6. Holiday calendar: holidays currently show as gaps; an exchange calendar would remove them.

## 7. Project layout

```
src/mdq/
  schema.py      canonical columns + vendor aliases
  ingestion.py   CSV/Parquet reading, parsing, quarantine
  quality.py     rule engine + rules + scorecard
  analytics.py   daily OHLCV, rolling VWAP, filters, summary
  insights.py    pattern detection + suggested controls (pluggable provider)
  pipeline.py    ingest -> validate -> analyse -> explain
  cli.py         batch reports
app/dashboard.py Streamlit UI
scripts/         data download + demo data generator
tests/           unit + integration (incl. UI smoke test)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design overview.
