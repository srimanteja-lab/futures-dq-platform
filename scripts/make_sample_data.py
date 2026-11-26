"""Build a demo CSV from the real Hugging Face sample with *known* injected
defects, plus a JSON manifest of what was injected (ground truth for tests).

Usage:  python scripts/make_sample_data.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "sample")
CONTRACTS = ["ESZ25", "CLH26"]
START, END = "2025-11-09 17:00", "2025-11-14 16:00"  # Sun evening -> Fri close
GAP_CONTRACT, GAP_START, GAP_END = "ESZ25", "2025-11-12 10:00", "2025-11-12 11:29"


def main() -> int:
    rng = np.random.default_rng(42)
    frames = []
    for c in CONTRACTS:
        d = pd.read_parquet(os.path.join(RAW, f"{c}_minute.parquet"))
        d = d[(d.timestamp_chicago_wall >= START) & (d.timestamp_chicago_wall <= END)]
        frames.append(d)
    base = pd.concat(frames)
    df = pd.DataFrame({
        "Contract": base.contract_symbol.values,
        "Timestamp": base.timestamp_chicago_wall.dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "Open": base.open.values, "High": base.high.values,
        "Low": base.low.values, "Close": base.close.values,
        "Volume": base.volume.astype("int64").values,
    }).astype(object)

    # 1) Outage: remove a 90-minute block of ES during US hours.
    ts = pd.to_datetime(df["Timestamp"])
    gap = (df.Contract == GAP_CONTRACT) & (ts >= GAP_START) & (ts <= GAP_END)
    df = df[~gap].reset_index(drop=True)

    manifest = {"DATA_GAP": {"contract": GAP_CONTRACT, "start": GAP_START, "end": GAP_END}}
    # Pick distinct victim rows away from the gap and from each other.
    pool = rng.permutation(np.arange(200, len(df) - 200))
    take = iter(pool)

    def pick(n):
        return [int(next(take)) for _ in range(n)]

    # 2) Missing close prices.
    rows = pick(8)
    df.loc[rows, "Close"] = ""
    manifest["MISSING_VALUE"] = len(rows)
    # 3) High < Low (swapped).
    rows = pick(5)
    hi = df.loc[rows, "High"].astype(float)
    lo = df.loc[rows, "Low"].astype(float)
    df.loc[rows, "High"] = (lo - 0.5).values
    manifest["OHLC_INCONSISTENT"] = len(rows)
    # 4) Negative / zero prices.
    rows = pick(3)
    df.loc[rows, "Low"] = -1.0
    manifest["INVALID_PRICE"] = len(rows)
    # 5) Negative volume.
    rows = pick(2)
    df.loc[rows, "Volume"] = -50
    manifest["INVALID_VOLUME"] = len(rows)
    # 6) Spike-and-revert bad ticks on the close (+6%).
    rows = pick(3)
    for r in rows:
        c = float(df.loc[r, "Close"]) * 1.06
        df.loc[r, ["Close", "High"]] = [round(c, 2), round(c, 2)]
    manifest["PRICE_OUTLIER"] = len(rows)
    # 7) Malformed records (quarantined at ingestion).
    rows = pick(4)
    df.loc[rows[:2], "Timestamp"] = "not-a-date"
    df.loc[rows[2:], "Open"] = "abc"
    rows = pick(2)
    df.loc[rows, "Contract"] = ""
    manifest["MALFORMED_RECORD"] = 6
    # 8) Exact duplicates (e.g. vendor re-sent a batch).
    dup_rows = pick(15)
    exact = df.loc[dup_rows].copy()
    # 9) Conflicting duplicates (a correction with a different close).
    conf_rows = pick(4)
    conflict = df.loc[conf_rows].copy()
    # Corrected close stays inside the bar's range so only the conflict fires.
    conflict["Close"] = [
        (lo if float(c) != float(lo) else hi)
        for c, lo, hi in zip(conflict["Close"], conflict["Low"], conflict["High"])
    ]
    conflict = conflict[conflict["Close"].astype(float) != df.loc[conflict.index, "Close"].astype(float)]
    df = pd.concat([df, exact, conflict]).reset_index(drop=True)
    manifest["DUPLICATE_EXACT"] = len(exact)
    manifest["DUPLICATE_CONFLICT"] = len(conflict)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "dirty_futures_sample.csv")
    df.to_csv(out, index=False)
    with open(os.path.join(OUT_DIR, "injected_defects.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out} ({len(df):,} rows) and injected_defects.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
