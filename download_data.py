"""Download minute-bar Parquet files from the public Hugging Face sample
(lynx1231/historical-futures-data-sample) into data/raw/.

Usage:
    python scripts/download_data.py            # one contract per root (8 files)
    python scripts/download_data.py --all      # all 40 minute files
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPO = "lynx1231/historical-futures-data-sample"
API = f"https://huggingface.co/api/datasets/{REPO}"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw")
DEFAULT = ["ESZ25", "CLH26", "GCH26", "ZNH26", "ZCH26", "VXF26", "SR3H26", "SBH26"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    with urllib.request.urlopen(API) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"] if s["rfilename"].startswith("data/minute/")]
    if not a.all:
        files = [f for f in files if os.path.basename(f)[:-8] in DEFAULT]
    os.makedirs(OUT, exist_ok=True)
    for f in files:
        dest = os.path.join(OUT, os.path.basename(f).replace(".parquet", "_minute.parquet"))
        if os.path.exists(dest):
            print("exists ", dest)
            continue
        urllib.request.urlretrieve(BASE + f, dest)
        print("fetched", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
