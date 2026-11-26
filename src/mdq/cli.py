"""Command-line entry point: produce all reports without the UI.

Example:
    python -m mdq.cli data/sample/dirty_futures_sample.csv --out reports
"""
from __future__ import annotations

import argparse
import os
import sys

from . import recon
from .analytics import CALENDAR, SESSION, rolling_vwap
from .ingestion import ingest
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
    p.add_argument("--reconcile-against", metavar="TARGET",
                   help="reconcile the input against this downstream copy (key, values, control totals)")
    p.add_argument("--vendor-daily", nargs="*", metavar="PARQUET",
                   help="vendor daily-bar files to reconcile our daily bars against")
    p.add_argument("--manifest", help="vendor files.csv: verify checksum, row count and volume")
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
    res.ingest.file_checks.to_csv(os.path.join(a.out, "file_checks.csv"), index=False)
    if a.manifest:
        recon.verify_manifest(a.files, a.manifest).to_csv(os.path.join(a.out, "manifest_check.csv"), index=False)
    if a.vendor_daily:
        summary, detail = recon.reconcile_daily(res.daily, recon.load_vendor_daily(a.vendor_daily))
        summary.to_csv(os.path.join(a.out, "recon_vs_vendor_daily.csv"), index=False)
        detail.to_csv(os.path.join(a.out, "recon_vs_vendor_daily_detail.csv"), index=False)
    if a.reconcile_against:
        rr = recon.reconcile(res.ingest.data, ingest([a.reconcile_against], max_malformed_pct=100).data)
        rr.checks.to_csv(os.path.join(a.out, "recon_checks.csv"), index=False)
        rr.breaks.to_csv(os.path.join(a.out, "recon_breaks.csv"), index=False)
        print("\nReconciliation vs", a.reconcile_against)
        print(rr.checks[["check", "source", "target", "result"]].to_string(index=False))

    print(f"Ingested {res.ingest.total_rows:,} rows from {len(res.ingest.files)} file(s); "
          f"{len(res.ingest.rejected):,} quarantined as malformed.")
    print(f"{len(q.issues):,} issues; {len(q.clean):,} clean records used for analytics.")
    print(q.counts().to_string(index=False))
    print("\nVerdict by contract (publication gate):")
    print(q.scorecard[["contract", "verdict", "validity_pct", "completeness_pct", "dq_score",
                       "verdict_reason"]].to_string(index=False))
    print(f"\nReports written to {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
