"""Canonical schema shared by every layer of the platform.

All inputs (CSV, Parquet, vendor-specific column names) are mapped to this
schema at ingestion, so analytics and quality rules never deal with vendor
quirks.
"""
from __future__ import annotations

CONTRACT = "contract"
TIMESTAMP = "timestamp"
OPEN, HIGH, LOW, CLOSE, VOLUME = "open", "high", "low", "close", "volume"

PRICE_COLS = [OPEN, HIGH, LOW, CLOSE]
NUMERIC_COLS = PRICE_COLS + [VOLUME]
CANONICAL_COLS = [CONTRACT, TIMESTAMP] + NUMERIC_COLS

# Lineage columns added at ingestion (audit trail back to the source record).
ROW_ID = "row_id"
SOURCE = "source_file"

# Accepted source column names, in priority order, per canonical column.
# Matching is case-insensitive and ignores spaces/underscores/hyphens.
COLUMN_ALIASES = {
    CONTRACT: ["contract", "contract_symbol", "symbol", "ticker", "instrument"],
    TIMESTAMP: [
        "timestamp",
        "timestamp_chicago_wall",
        "datetime",
        "date_time",
        "time",
        "ts",
        "timestamp_ms",
        "date",
    ],
    OPEN: ["open", "open_price", "o"],
    HIGH: ["high", "high_price", "h"],
    LOW: ["low", "low_price", "l"],
    CLOSE: ["close", "close_price", "last", "c"],
    VOLUME: ["volume", "vol", "v", "qty"],
}

# Exchange-local timezone for the sample data (CME/CBOT/NYMEX/COMEX/CFE).
EXCHANGE_TZ = "America/Chicago"


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_columns(columns) -> dict:
    """Return {source_column: canonical_column} for the best alias matches."""
    by_norm = {}
    for col in columns:
        by_norm.setdefault(_norm(col), col)
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            src = by_norm.get(_norm(alias))
            if src is not None and src not in mapping:
                mapping[src] = canonical
                break
    return mapping
