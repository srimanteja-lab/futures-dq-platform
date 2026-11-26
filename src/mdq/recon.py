"""Reconciliation: prove two data sets agree, not just that their counts do.

Three reconciliations, each answering a business question:

1. ``reconcile``             - "Is this copy of the data the same as the source?"
   Count, key match (full outer join), value check, control totals.
   A matching row count is never enough: one dropped row plus one added row
   keeps the count identical.
2. ``reconcile_daily``       - "Do our daily bars agree with the vendor's own
   daily bars?" Key = contract + trading date. Checks completeness (missing
   days, volume coverage), range containment and open price. Close is shown
   against the vendor *settlement* price - a definition difference, not a break.
3. ``verify_manifest``       - "Did the file arrive intact?" SHA-256, row count
   and total volume checked against the vendor's published manifest.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import schema as S

MATCHED = "MATCHED"
MISSING = "MISSING_IN_TARGET"
UNEXPECTED = "UNEXPECTED_IN_TARGET"
VALUE_BREAK = "VALUE_BREAK"
DUP_KEY = "DUPLICATE_KEY"
PASS, BREAK, INFO = "Pass", "Break", "Info"


@dataclass
class ReconResult:
    checks: pd.DataFrame              # one row per check: name, source, target, result
    breaks: pd.DataFrame              # one row per broken key, with a status
    control_totals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        return bool((self.checks["result"] != BREAK).all())


def _check(name, src, tgt, ok, note=""):
    return {"check": name, "source": src, "target": tgt, "result": PASS if ok else BREAK, "note": note}


# --------------------------------------------------------------- 1. two copies
def reconcile(source: pd.DataFrame, target: pd.DataFrame,
              keys: Iterable[str] = (S.CONTRACT, S.TIMESTAMP),
              value_cols: Iterable[str] = tuple(S.NUMERIC_COLS),
              total_col: str = S.VOLUME, tolerance: float = 0.0) -> ReconResult:
    """Reconcile a target copy against its source, record by record."""
    keys, value_cols = list(keys), [c for c in value_cols if c in source and c in target]
    checks: List[dict] = [
        _check("Row count", len(source), len(target), len(source) == len(target),
               "Necessary, never sufficient"),
    ]
    dup_s = source[source.duplicated(keys, keep="first")]
    dup_t = target[target.duplicated(keys, keep="first")]
    # Duplicates present equally on both sides were inherited from the source
    # (a data-quality issue, reported by the DQ rules), not a copy break.
    same_dups = len(dup_s) == len(dup_t)
    checks.append(_check("Duplicate keys", len(dup_s), len(dup_t), same_dups,
                         "Inherited from source; see DQ issue log" if same_dups and len(dup_s)
                         else "Same contract + timestamp more than once"))
    s = source.drop_duplicates(keys)
    t = target.drop_duplicates(keys)
    j = s.merge(t, on=keys, how="outer", suffixes=("_src", "_tgt"), indicator=True)
    missing = j[j["_merge"] == "left_only"]
    unexpected = j[j["_merge"] == "right_only"]
    both = j[j["_merge"] == "both"]
    diff_mask = pd.Series(False, index=both.index)
    for c in value_cols:
        a, b = both[f"{c}_src"], both[f"{c}_tgt"]
        diff_mask |= ~((a - b).abs().le(tolerance) | (a.isna() & b.isna()))
    value_breaks = both[diff_mask]
    checks += [
        _check("Keys missing in target", len(missing), "-", missing.empty, "Full outer join on " + " + ".join(keys)),
        _check("Unexpected keys in target", "-", len(unexpected), unexpected.empty, "In target, not in source"),
        _check("Value differences on matched keys", "-", len(value_breaks), value_breaks.empty,
               "Field by field: " + ", ".join(value_cols)),
    ]
    ctl = pd.DataFrame()
    if total_col in source and total_col in target:
        ss, tt = source[total_col].sum(), target[total_col].sum()
        checks.append(_check(f"Control total: sum of {total_col}", f"{ss:,.0f}", f"{tt:,.0f}",
                             abs(ss - tt) <= tolerance, f"difference {tt - ss:+,.0f}"))
        if S.CONTRACT in source and S.TIMESTAMP in source:
            g = lambda d: d.groupby([d[S.CONTRACT], d[S.TIMESTAMP].dt.date.rename("date")])[total_col].sum()
            ctl = pd.concat([g(source).rename("source"), g(target).rename("target")], axis=1).fillna(0)
            ctl["difference"] = ctl["target"] - ctl["source"]
            ctl = ctl[ctl["difference"] != 0].reset_index()

    def frame(df, status):
        out = df[keys].copy()
        out["status"] = status
        out["detail"] = ""
        return out

    vb = frame(value_breaks, VALUE_BREAK)
    if not value_breaks.empty:
        vb["detail"] = [
            "; ".join(f"{c}: {r[f'{c}_src']} -> {r[f'{c}_tgt']}" for c in value_cols
                      if not (abs(r[f"{c}_src"] - r[f"{c}_tgt"]) <= tolerance
                              or (pd.isna(r[f"{c}_src"]) and pd.isna(r[f"{c}_tgt"]))))
            for _, r in value_breaks.iterrows()
        ]
    dups = (pd.DataFrame(columns=keys + ["status", "detail"]) if same_dups else
            pd.concat([frame(dup_s, DUP_KEY).assign(detail="in source"),
                       frame(dup_t, DUP_KEY).assign(detail="in target")]))
    parts = [p for p in (frame(missing, MISSING), frame(unexpected, UNEXPECTED), vb, dups) if not p.empty]
    breaks = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=keys + ["status", "detail"]))
    return ReconResult(checks=pd.DataFrame(checks), breaks=breaks, control_totals=ctl)


# ------------------------------------------------- 2. our daily bars vs vendor
def load_vendor_daily(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        frames.append(pd.DataFrame({
            "contract": d["contract_symbol"].astype(str).str.upper(),
            "date": pd.to_datetime(d["date"]).dt.normalize(),
            "open": d["open"].astype(float), "high": d["high"].astype(float),
            "low": d["low"].astype(float), "settle": d["close"].astype(float),
            "volume": d["volume"].astype(float),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reconcile_daily(ours: pd.DataFrame, vendor: pd.DataFrame, tick: float = 1e-9):
    """Compare daily bars built from minute data with the vendor's daily file.

    Only vendor days inside the period our minute data covers are expected.
    Returns (summary per contract, detail per contract-day).
    """
    o = ours.rename(columns={"trading_date": "date"})[
        ["contract", "date", "open", "high", "low", "close", "volume"]].copy()
    o["date"] = pd.to_datetime(o["date"]).dt.normalize()
    rows, details = [], []
    for c, oc in o.groupby("contract"):
        vc = vendor[(vendor["contract"] == c)
                    & vendor["date"].between(oc["date"].min(), oc["date"].max())]
        if vc.empty:
            continue
        j = oc.merge(vc, on=["contract", "date"], how="outer", suffixes=("", "_v"), indicator=True)
        both = j[j["_merge"] == "both"].copy()
        both["open_match"] = (both["open"] - both["open_v"]).abs() <= tick
        both["range_break"] = (both["high"] > both["high_v"] + tick) | (both["low"] < both["low_v"] - tick)
        both["volume_coverage_pct"] = 100 * both["volume"] / both["volume_v"].replace(0, np.nan)
        both["close_minus_settle"] = both["close"] - both["settle"]
        status = np.select(
            [j["_merge"] == "right_only", j["_merge"] == "left_only"],
            ["MISSING_IN_MINUTE_DATA", "NOT_IN_VENDOR_DAILY"], default=MATCHED)
        j["status"] = status
        j.loc[both.index[both["range_break"]], "status"] = "RANGE_BREAK"
        det = j[["contract", "date", "status"]].join(
            both[["open_match", "volume_coverage_pct", "close_minus_settle"]])
        details.append(det)
        vol_cov = 100 * both["volume"].sum() / both["volume_v"].sum() if both["volume_v"].sum() else np.nan
        rows.append({
            "contract": c,
            "vendor_days": len(vc),
            "matched_days": len(both),
            "days_missing_in_minute_data": int((j["_merge"] == "right_only").sum()),
            "days_not_in_vendor": int((j["_merge"] == "left_only").sum()),
            "open_match_pct": round(100 * both["open_match"].mean(), 1) if len(both) else np.nan,
            "range_breaks": int(both["range_break"].sum()),
            "volume_coverage_pct": round(vol_cov, 1),
            "median_abs_close_vs_settle": round(both["close_minus_settle"].abs().median(), 4) if len(both) else np.nan,
        })
    summary = pd.DataFrame(rows)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detail


# ------------------------------------------------------ 3. vendor manifest
_NAME = re.compile(r"^(?P<sym>[A-Z0-9]+?)_(?P<freq>minute|daily)\.parquet$", re.I)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(paths: Iterable[str], manifest_csv: str) -> pd.DataFrame:
    """Check each local file against the vendor manifest (files.csv)."""
    man = pd.read_csv(manifest_csv) if os.path.exists(manifest_csv) else pd.DataFrame()
    out = []
    for p in paths:
        name = os.path.basename(p)
        m = _NAME.match(name)
        row: Dict[str, object] = {"file": name, "status": "Not in manifest", "rows": None,
                                  "manifest_rows": None, "checksum": None, "volume_matches": None}
        if m is None or man.empty:
            out.append(row)
            continue
        hit = man[(man["contract_symbol"].str.upper() == m["sym"].upper()) & (man["frequency"] == m["freq"].lower())]
        if hit.empty:
            out.append(row)
            continue
        h = hit.iloc[0]
        df = pd.read_parquet(p, columns=["volume"])
        sha_ok = sha256_file(p) == h["sha256"]
        rows_ok = len(df) == int(h["row_count"])
        vol_ok = int(df["volume"].fillna(0).sum()) == int(h["reported_total_volume"])
        row.update(rows=len(df), manifest_rows=int(h["row_count"]),
                   checksum="match" if sha_ok else "MISMATCH",
                   volume_matches=vol_ok,
                   status="Verified" if (sha_ok and rows_ok and vol_ok) else "MISMATCH")
        out.append(row)
    return pd.DataFrame(out)
