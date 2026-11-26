"""Ingestion layer: read CSV/Parquet, map to the canonical schema, quarantine
malformed records.

Design choices
--------------
* Nothing is silently dropped. A record that cannot be parsed is moved to a
  *rejected* (quarantine) table with a reason code, so it can be reported and
  reconciled with the vendor.
* A *missing* value (blank cell) is different from a *malformed* value
  ("abc" in a price column). Missing prices/volumes are kept and flagged by the
  quality layer; malformed values are quarantined here.
* Every record keeps ``row_id`` and ``source_file`` for lineage.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Union

import numpy as np
import pandas as pd

from . import schema as S


class SchemaError(ValueError):
    """Raised when a file is missing a required column entirely."""


REJECT_COLUMNS = [S.SOURCE, S.ROW_ID, "reason", "raw_record"]
FILE_CHECK_COLUMNS = ["file", "rows", "malformed", "malformed_pct", "status"]
FILE_ACCEPTED, FILE_REJECTED = "Accepted", "Rejected"
# Preventive control: a file with more malformed rows than this is rejected
# whole - a broken extract shouldn't be half-loaded into analytics.
DEFAULT_MAX_MALFORMED_PCT = 1.0


@dataclass
class IngestResult:
    data: pd.DataFrame
    rejected: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=REJECT_COLUMNS)
    )
    files: List[str] = field(default_factory=list)
    # One row per input file: the file-level control (preventive).
    file_checks: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=FILE_CHECK_COLUMNS)
    )

    @property
    def total_rows(self) -> int:
        return len(self.data) + len(self.rejected)


FileLike = Union[str, os.PathLike, io.IOBase]


def read_raw(source: FileLike, filename: str = None) -> pd.DataFrame:
    """Read a CSV or Parquet file (path or file-like object) as-is."""
    name = filename or (str(source) if not hasattr(source, "read") else getattr(source, "name", ""))
    lower = name.lower()
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(source)
    if lower.endswith((".csv", ".txt")) or lower.endswith(".csv.gz"):
        # Read everything as text so we can tell "missing" from "malformed".
        return pd.read_csv(source, dtype=str, keep_default_na=False, skipinitialspace=True)
    raise ValueError(f"Unsupported file type: {name!r} (expected .csv or .parquet)")


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse to tz-naive exchange-local (America/Chicago) wall-clock time.

    * Integer epochs are interpreted as milliseconds (the sample data stores
      Chicago wall-clock labels as ms, NOT true UTC; see README).
    * Strings with an explicit offset are converted to Chicago time.
    * Naive strings are assumed to already be exchange-local.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        ts = values
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(S.EXCHANGE_TZ).dt.tz_localize(None)
        return ts

    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce")

    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    is_epoch = numeric.notna() & text.str.fullmatch(r"-?\d{9,}").fillna(False)

    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if is_epoch.any():
        out[is_epoch] = pd.to_datetime(numeric[is_epoch].astype("int64"), unit="ms")

    rest = text[~is_epoch & text.notna() & (text != "")]
    if len(rest):
        has_offset = rest.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        naive = rest[~has_offset]
        aware = rest[has_offset]
        if len(naive):
            out[naive.index] = pd.to_datetime(naive, errors="coerce", format="mixed")
        if len(aware):
            parsed = pd.to_datetime(aware, errors="coerce", utc=True, format="mixed")
            out[aware.index] = parsed.dt.tz_convert(S.EXCHANGE_TZ).dt.tz_localize(None)
    return out


def _is_blank(series: pd.Series) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        s = series.astype("string").str.strip()
        return s.isna() | s.isin(["", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"])
    return series.isna()


def normalise(raw: pd.DataFrame, source_name: str = "<memory>") -> IngestResult:
    """Map a raw frame onto the canonical schema and quarantine bad records."""
    mapping = S.resolve_columns(raw.columns)
    found = set(mapping.values())
    required = {S.CONTRACT, S.TIMESTAMP} | set(S.PRICE_COLS)
    missing = sorted(required - found)
    if missing:
        raise SchemaError(
            f"{source_name}: missing required column(s) {missing}. "
            f"Found columns: {list(raw.columns)}"
        )

    df = raw[list(mapping)].rename(columns=mapping).copy()
    if S.VOLUME not in df:
        df[S.VOLUME] = np.nan
    df.insert(0, S.ROW_ID, np.arange(1, len(df) + 1))
    df.insert(0, S.SOURCE, os.path.basename(source_name))

    reasons = pd.Series("", index=df.index, dtype=object)

    def add_reason(mask, code):
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask] == "", reasons.loc[mask] + ";") + code

    # Contract
    contract_blank = _is_blank(df[S.CONTRACT])
    add_reason(contract_blank, "MISSING_CONTRACT")
    df[S.CONTRACT] = df[S.CONTRACT].astype("string").str.strip().str.upper()

    # Timestamp
    ts_blank = _is_blank(df[S.TIMESTAMP])
    parsed_ts = _parse_timestamps(df[S.TIMESTAMP])
    add_reason(ts_blank, "MISSING_TIMESTAMP")
    add_reason(~ts_blank & parsed_ts.isna(), "MALFORMED_TIMESTAMP")
    df[S.TIMESTAMP] = parsed_ts

    # Numerics: blank -> NaN (kept, flagged later); garbage -> quarantined.
    for col in S.NUMERIC_COLS:
        blank = _is_blank(df[col])
        parsed = pd.to_numeric(
            df[col].astype("string").str.strip().str.replace(",", "", regex=False)
            if not pd.api.types.is_numeric_dtype(df[col])
            else df[col],
            errors="coerce",
        ).astype("float64")
        bad = ~blank & (parsed.isna() | ~np.isfinite(parsed.fillna(0)))
        add_reason(bad, f"MALFORMED_{col.upper()}")
        df[col] = parsed.where(np.isfinite(parsed))

    is_rejected = reasons != ""
    rejected = pd.DataFrame(
        {
            S.SOURCE: df.loc[is_rejected, S.SOURCE],
            S.ROW_ID: df.loc[is_rejected, S.ROW_ID],
            "reason": reasons[is_rejected],
            "raw_record": raw.loc[is_rejected].astype(str).agg(" | ".join, axis=1)
            if is_rejected.any()
            else pd.Series(dtype=str),
        }
    ).reset_index(drop=True)

    clean = df.loc[~is_rejected].reset_index(drop=True)
    clean[S.CONTRACT] = clean[S.CONTRACT].astype(str)
    return IngestResult(data=clean, rejected=rejected, files=[os.path.basename(source_name)])


def ingest(sources: Iterable, filenames: Iterable[str] = None,
           max_malformed_pct: float = DEFAULT_MAX_MALFORMED_PCT) -> IngestResult:
    """Ingest one or more files (paths or uploaded file objects).

    File-level control: if more than ``max_malformed_pct`` % of a file's rows
    are malformed, the whole file is rejected. Its parsed rows are moved to
    quarantine with reason FILE_REJECTED, so record accounting still balances.
    """
    sources = list(sources)
    filenames = list(filenames) if filenames is not None else [None] * len(sources)
    frames, rejects, files, checks = [], [], [], []
    for src, fname in zip(sources, filenames):
        name = fname or getattr(src, "name", None) or str(src)
        raw = read_raw(src, name)
        result = normalise(raw, name)
        n = len(raw)
        bad = len(result.rejected)
        pct = 100.0 * bad / n if n else 0.0
        status = FILE_REJECTED if pct > max_malformed_pct else FILE_ACCEPTED
        checks.append({"file": os.path.basename(name), "rows": n, "malformed": bad,
                       "malformed_pct": round(pct, 3), "status": status})
        if status == FILE_REJECTED and not result.data.empty:
            moved = pd.DataFrame({
                S.SOURCE: result.data[S.SOURCE].values,
                S.ROW_ID: result.data[S.ROW_ID].values,
                "reason": f"FILE_REJECTED ({pct:.1f}% malformed > {max_malformed_pct:g}%)",
                "raw_record": raw.iloc[result.data[S.ROW_ID].values - 1].astype(str).agg(" | ".join, axis=1).values,
            })
            rejects += [result.rejected, moved]
            frames.append(result.data.iloc[0:0])
        else:
            frames.append(result.data)
            rejects.append(result.rejected)
        files += result.files
    if not frames:
        raise ValueError("No input files provided")
    data = pd.concat(frames, ignore_index=True)
    rejected = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=REJECT_COLUMNS)
    rejected = rejected.sort_values([S.SOURCE, S.ROW_ID]).reset_index(drop=True) if not rejected.empty else rejected
    return IngestResult(data=data, rejected=rejected, files=files,
                        file_checks=pd.DataFrame(checks, columns=FILE_CHECK_COLUMNS))
