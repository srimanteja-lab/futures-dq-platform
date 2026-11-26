"""Reconciliation: counts are necessary, never sufficient."""
import hashlib

import pandas as pd

from conftest import make_bars
from mdq import recon


def test_count_matches_but_key_match_catches_drop_plus_add():
    src = make_bars(n=10)
    tgt = src.drop(index=3)                                    # one row dropped...
    extra = src.iloc[[0]].assign(timestamp=pd.Timestamp("2025-11-10 12:00"))
    tgt = pd.concat([tgt, extra], ignore_index=True)           # ...one row added
    r = recon.reconcile(src, tgt)
    checks = r.checks.set_index("check")["result"]
    assert checks["Row count"] == recon.PASS                   # 10 == 10
    assert checks["Keys missing in target"] == recon.BREAK
    assert checks["Unexpected keys in target"] == recon.BREAK
    assert not r.passed
    assert set(r.breaks.status) == {recon.MISSING, recon.UNEXPECTED}


def test_value_break_and_control_total():
    src = make_bars(n=10)
    tgt = src.copy()
    tgt.loc[5, "volume"] += 11
    r = recon.reconcile(src, tgt)
    checks = r.checks.set_index("check")["result"]
    assert checks["Value differences on matched keys"] == recon.BREAK
    assert checks["Control total: sum of volume"] == recon.BREAK
    vb = r.breaks[r.breaks.status == recon.VALUE_BREAK]
    assert len(vb) == 1 and "volume: 10.0 -> 21.0" in vb.detail.iloc[0]
    assert r.control_totals["difference"].tolist() == [11]


def test_identical_copies_pass_and_duplicates_are_reported():
    src = make_bars(n=10)
    assert recon.reconcile(src, src.copy()).passed
    dup = pd.concat([src, src.iloc[[2]]], ignore_index=True)
    r = recon.reconcile(src, dup)
    assert (r.breaks.status == recon.DUP_KEY).sum() == 1


def test_daily_reconciliation_flags_missing_days_and_range_breaks():
    ours = pd.DataFrame({
        "contract": "ESZ25", "trading_date": pd.to_datetime(["2025-11-10", "2025-11-12"]),
        "open": [100.0, 102.0], "high": [105.0, 110.0], "low": [99.0, 101.0],
        "close": [104.0, 103.0], "volume": [900, 500]})
    vendor = pd.DataFrame({
        "contract": "ESZ25", "date": pd.to_datetime(["2025-11-10", "2025-11-11", "2025-11-12"]),
        "open": [100.0, 101.0, 102.0], "high": [105.0, 104.0, 106.0], "low": [99.0, 100.0, 101.0],
        "settle": [104.5, 102.0, 103.0], "volume": [1000.0, 800.0, 500.0]})
    summary, detail = recon.reconcile_daily(ours, vendor)
    row = summary.iloc[0]
    assert row.days_missing_in_minute_data == 1          # 11 Nov absent from minute data
    assert row.range_breaks == 1                          # our high 110 > vendor 106
    assert row.open_match_pct == 100.0
    assert row.volume_coverage_pct == round(100 * 1400 / 1500, 1)
    assert "RANGE_BREAK" in set(detail.status)


def test_manifest_verification(tmp_path):
    p = tmp_path / "ESZ25_minute.parquet"
    pd.DataFrame({"volume": [1, 2, 3]}).to_parquet(p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    man = tmp_path / "files.csv"
    pd.DataFrame([{"contract_symbol": "ESZ25", "frequency": "minute", "row_count": 3,
                   "reported_total_volume": 6, "sha256": sha}]).to_csv(man, index=False)
    assert recon.verify_manifest([str(p)], str(man)).status.tolist() == ["Verified"]
    pd.DataFrame([{"contract_symbol": "ESZ25", "frequency": "minute", "row_count": 4,
                   "reported_total_volume": 6, "sha256": "x"}]).to_csv(man, index=False)
    out = recon.verify_manifest([str(p)], str(man)).iloc[0]
    assert out.status == "MISMATCH" and out.checksum == "MISMATCH"
