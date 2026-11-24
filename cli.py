"""Command-line entry point: produce all reports without the UI.

Example:
    python -m mdq.cli data/sample/dirty_futures_sample.csv --out reports
"""
from __future__ import annotations

import argparse
import os
import sys

from .analytics import CALENDAR, SESSION, rolling_vwap
from .insights import insights_frame
from .pipeline import run_pipeline
from .quality import QualityConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Futures market data quality & analytics")
    p.add_argument("files", nargs="+", help="CSV or Parquet files")
    p.add_argument("--out", default="reports", help="output directory")
    p.add_argument("--contracts", nargs="*", help="filter to these contracts")
    p.add_argument("--start", help="start date/time (inclusive)")
    p.add_argument("--end", help="end date/time (inclusive)")
    p.add_argument("--convention", choices=[SESSION, CALENDAR], default=SESSION)
    p.add_argument("--gap-minutes", type=int, default=30)
    p.add_argument("--allow-negative-prices", action="store_true")
    a = p.parse_args(argv)

    cfg = QualityConfig(gap_threshold_minutes=a.gap_minutes, allow_negative_prices=a.allow_negative_prices)
    res = run_pipeline(a.files, cfg=cfg, convention=a.convention,
                       contracts=a.contracts, start=a.start, end=a.end)
    os.makedirs(a.out, exist_ok=True)
    q = res.quality
    q.scorecard.to_csv(os.path.join(a.out, "dq_scorecard.csv"), index=False)
    q.issues.to_csv(os.path.join(a.out, "dq_issues.csv"), index=False)
    res.ingest.rejected.to_csv(os.path.join(a.out, "rejected_records.csv"), index=False)
    res.daily.to_csv(os.path.join(a.out, "daily_ohlcv.csv"), index=False)
    rolling_vwap(q.clean).to_parquet(os.path.join(a.out, "vwap_15m.parquet"), index=False)
    insights_frame(res.insights).to_csv(os.path.join(a.out, "insights.csv"), index=False)

    print(f"Ingested {res.ingest.total_rows:,} rows from {len(res.ingest.files)} file(s); "
          f"{len(res.ingest.rejected):,} quarantined as malformed.")
    print(f"{len(q.issues):,} issues; {len(q.clean):,} clean records used for analytics.")
    print(q.counts().to_string(index=False))
    print(f"\nReports written to {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
