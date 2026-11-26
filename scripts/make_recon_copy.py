"""Create a 'downstream copy' of the demo file with three silent differences,
for the reconciliation demo:

* one record dropped      (ESZ25 2025-11-12 09:45)
* one record added        (ESZ25 2025-11-12 10:30, inside the outage window)
* one volume changed +11  (CLH26 2025-11-13 10:00)

The row count is unchanged, so a count-only check passes. The key match,
value check and control total all break.

Usage: python scripts/make_recon_copy.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "sample", "dirty_futures_sample.csv")
OUT = os.path.join(ROOT, "data", "sample", "dirty_futures_copy.csv")


def main():
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    drop = df.index[(df.Contract == "ESZ25") & (df.Timestamp == "2025-11-12 09:45:00")][0]
    added = df.loc[[drop]].copy()
    added["Timestamp"] = "2025-11-12 10:30:00"
    change = df.index[(df.Contract == "CLH26") & (df.Timestamp == "2025-11-13 10:00:00")][0]
    df.loc[change, "Volume"] = str(int(float(df.loc[change, "Volume"])) + 11)
    out = pd.concat([df.drop(index=drop), added], ignore_index=True)
    out.to_csv(OUT, index=False)
    with open(OUT.replace(".csv", "_changes.json"), "w") as f:
        json.dump({"dropped": "ESZ25 2025-11-12 09:45", "added": "ESZ25 2025-11-12 10:30",
                   "volume_changed": "CLH26 2025-11-13 10:00 (+11)"}, f, indent=2)
    print(f"wrote {OUT}: {len(out)} rows (source {len(df)})")


if __name__ == "__main__":
    main()
