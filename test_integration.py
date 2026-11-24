"""End-to-end tests: real files through the whole pipeline, CLI and UI."""
import glob
import json
import os

import pandas as pd
import pytest

from mdq.cli import main as cli_main
from mdq.pipeline import run_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRTY = os.path.join(ROOT, "data", "sample", "dirty_futures_sample.csv")
MANIFEST = os.path.join(ROOT, "data", "sample", "injected_defects.json")
RAW = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.parquet")))


@pytest.fixture(scope="module")
def dirty():
    if not os.path.exists(DIRTY):
        pytest.skip("run scripts/make_sample_data.py first")
    return run_pipeline([DIRTY])


def test_every_injected_defect_is_detected_exactly(dirty):
    with open(MANIFEST) as f:
        truth = json.load(f)
    counts = dirty.quality.issues.rule_id.value_counts().to_dict()
    for rule_id, expected in truth.items():
        if rule_id == "DATA_GAP":
            continue
        assert counts.get(rule_id, 0) == expected, rule_id


def test_injected_outage_found_with_exact_window(dirty):
    with open(MANIFEST) as f:
        g = json.load(f)["DATA_GAP"]
    gaps = dirty.quality.issues.query("rule_id == 'DATA_GAP' and contract == @g['contract']")
    hit = gaps[pd.to_datetime(gaps.timestamp) == pd.Timestamp(g["start"])]
    assert len(hit) == 1 and hit.gap_minutes.iloc[0] == 90


def test_no_error_record_reaches_analytics(dirty):
    q = dirty.quality
    assert q.clean[["open", "high", "low", "close"]].notna().all().all()
    assert (q.clean[["open", "high", "low", "close"]] > 0).all().all()
    assert (q.clean.high >= q.clean[["open", "close", "low"]].max(axis=1)).all()
    assert not q.clean.duplicated(["contract", "timestamp"]).any()
    assert (q.clean.volume.dropna() >= 0).all()


def test_daily_bars_and_insights_produced(dirty):
    assert set(dirty.daily.contract) == {"ESZ25", "CLH26"}
    assert len(dirty.daily) == 10  # 5 sessions x 2 contracts
    assert any(i.title == "Spike-and-revert ticks detected" for i in dirty.insights)


def test_record_accounting_balances(dirty):
    ing, q = dirty.ingest, dirty.quality
    excluded = len(ing.data) - len(q.clean)
    assert ing.total_rows == len(q.clean) + excluded + len(ing.rejected)


def test_cli_writes_all_reports(tmp_path):
    if not os.path.exists(DIRTY):
        pytest.skip("demo file missing")
    assert cli_main([DIRTY, "--out", str(tmp_path), "--contracts", "ESZ25"]) == 0
    for name in ["dq_scorecard.csv", "dq_issues.csv", "rejected_records.csv",
                 "daily_ohlcv.csv", "vwap_15m.parquet", "insights.csv"]:
        assert (tmp_path / name).exists(), name
    assert set(pd.read_csv(tmp_path / "daily_ohlcv.csv").contract) == {"ESZ25"}


@pytest.mark.skipif(not RAW, reason="real parquet not downloaded")
def test_real_parquet_file_end_to_end():
    gc = [p for p in RAW if "GCH26" in p] or RAW[:1]
    res = run_pipeline(gc, start="2026-01-01", end="2026-01-31")
    assert not res.daily.empty
    assert res.ingest.rejected.empty
    assert (res.daily.high >= res.daily.low).all()


def test_dashboard_renders_without_errors():
    if not os.path.exists(DIRTY):
        pytest.skip("demo file missing")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "app", "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("Futures Market Data Quality")
    assert len(at.tabs) == 4
