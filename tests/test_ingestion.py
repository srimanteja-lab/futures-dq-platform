import io

import pandas as pd
import pytest

from mdq.ingestion import SchemaError, ingest, normalise


def _csv(text, name="t.csv", max_malformed_pct=100.0):
    # Row-level tests use tiny files that are mostly malformed on purpose, so
    # the file-level rejection control is switched off unless a test sets it.
    buf = io.BytesIO(text.encode())
    return ingest([buf], [name], max_malformed_pct=max_malformed_pct)


def test_file_over_malformed_threshold_is_rejected_whole_and_accounted():
    text = "Contract,Timestamp,Open,High,Low,Close,Volume\n" + "".join(
        f"ES,{ts:%Y-%m-%d %H:%M},1,2,0.5,1,10\n"
        for ts in pd.date_range("2025-11-10 09:00", periods=98, freq="1min")
    ) + "ES,2025-11-10 11:00,abc,2,0.5,1,10\nES,bad,1,2,0.5,1,10\n"  # 2% malformed
    r = _csv(text, max_malformed_pct=1.0)
    assert r.file_checks.status.tolist() == ["Rejected"]
    assert r.data.empty
    assert (r.rejected.reason.str.startswith("FILE_REJECTED")).sum() == 98
    assert r.total_rows == 100  # nothing silently dropped
    ok = _csv(text, max_malformed_pct=5.0)
    assert ok.file_checks.status.tolist() == ["Accepted"] and len(ok.data) == 98


def test_maps_brief_column_names_and_types():
    r = _csv("Contract,Timestamp,Open,High,Low,Close,Volume\nesz25,2025-11-10 09:30:00,1,2,0.5,1.5,10\n")
    row = r.data.iloc[0]
    assert row.contract == "ESZ25"  # trimmed + upper-cased
    assert row.timestamp == pd.Timestamp("2025-11-10 09:30")
    assert (row.open, row.high, row.low, row.close, row.volume) == (1, 2, 0.5, 1.5, 10)
    assert row.row_id == 1 and row.source_file == "t.csv"


def test_maps_vendor_parquet_schema(tmp_path):
    raw = pd.DataFrame({
        "contract_symbol": ["ZNH26"], "timestamp_ms": [1751016600000],
        "timestamp_chicago_wall": [pd.Timestamp("2025-06-27 09:30")],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [3],
    })
    p = tmp_path / "x.parquet"
    raw.to_parquet(p)
    r = ingest([str(p)])
    # wall-clock column preferred over the integer ms label
    assert r.data.timestamp.iloc[0] == pd.Timestamp("2025-06-27 09:30")


def test_missing_value_is_kept_but_malformed_is_quarantined():
    r = _csv(
        "Contract,Timestamp,Open,High,Low,Close,Volume\n"
        "ES,2025-11-10 09:30,1,2,0.5,,10\n"        # missing close -> kept
        "ES,2025-11-10 09:31,abc,2,0.5,1,10\n"     # malformed open -> rejected
        "ES,not-a-date,1,2,0.5,1,10\n"             # malformed ts -> rejected
        ",2025-11-10 09:33,1,2,0.5,1,10\n"         # missing contract -> rejected
        "ES,,1,2,0.5,1,10\n"                       # missing ts -> rejected
    )
    assert len(r.data) == 1 and pd.isna(r.data.close.iloc[0])
    reasons = set(r.rejected.reason)
    assert reasons == {"MALFORMED_OPEN", "MALFORMED_TIMESTAMP", "MISSING_CONTRACT", "MISSING_TIMESTAMP"}
    assert r.total_rows == 5


def test_multiple_reasons_are_combined():
    r = _csv("Contract,Timestamp,Open,High,Low,Close,Volume\n,bad,x,2,1,1,1\n")
    assert r.rejected.reason.iloc[0] == "MISSING_CONTRACT;MALFORMED_TIMESTAMP;MALFORMED_OPEN"


def test_timezone_aware_strings_converted_to_chicago():
    r = _csv("Contract,Timestamp,Open,High,Low,Close,Volume\nES,2025-11-10T15:30:00Z,1,1,1,1,1\n")
    assert r.data.timestamp.iloc[0] == pd.Timestamp("2025-11-10 09:30")  # UTC-6 in November


def test_epoch_millis_strings_parsed():
    r = _csv("Contract,Timestamp,Open,High,Low,Close,Volume\nES,1751016600000,1,1,1,1,1\n")
    assert r.data.timestamp.iloc[0] == pd.Timestamp("2025-06-27 09:30")


def test_missing_required_column_raises():
    with pytest.raises(SchemaError, match="close"):
        normalise(pd.DataFrame({"Contract": ["ES"], "Timestamp": ["2025-01-01"], "Open": [1], "High": [1], "Low": [1]}))


def test_volume_optional():
    r = _csv("Contract,Timestamp,Open,High,Low,Close\nES,2025-11-10 09:30,1,1,1,1\n")
    assert pd.isna(r.data.volume.iloc[0])


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        ingest([io.BytesIO(b"x")], ["data.xlsx"])
