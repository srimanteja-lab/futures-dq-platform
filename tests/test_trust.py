"""Fixes for mistakes found in the AI-built logic, each with a test."""
import datetime as dt

import pandas as pd

from conftest import make_bars
from mdq import exchange_calendar as XC
from mdq.insights import generate_insights
from mdq.quality import BLOCKED, CAUTION, INFO, TRUSTED, QualityConfig, run_quality


def test_calendar_knows_thanksgiving_week_and_moving_holidays():
    assert XC.lookup(dt.date(2025, 11, 27))[0] == "Thanksgiving"
    assert XC.lookup(dt.date(2025, 11, 28))[0] == "Day after Thanksgiving"
    assert XC.lookup(dt.date(2025, 12, 24))[0] == "Christmas Eve"
    assert XC.lookup(dt.date(2024, 3, 29)) == ("Good Friday", XC.CLOSED)
    assert XC.lookup(dt.date(2025, 11, 26)) is None


def _with_gap(start, gap_from, gap_to, n=600):
    b = make_bars(start=start, n=n)
    return b[(b.timestamp < pd.Timestamp(gap_from)) | (b.timestamp >= pd.Timestamp(gap_to))]


def test_black_friday_early_close_is_scheduled_not_a_data_gap():
    # Bars stop at 12:15 CT on Black Friday 2025 and resume after.
    bars = _with_gap("2025-11-28 09:00", "2025-11-28 12:15", "2025-11-28 14:00")
    rep = run_quality(bars, QualityConfig(learn_session_from_data=False))
    assert (rep.issues.rule_id == "DATA_GAP").sum() == 0
    sched = rep.issues[rep.issues.rule_id == "SCHEDULED_CLOSURE"]
    assert len(sched) == 1 and sched.severity.iloc[0] == INFO
    assert "Day after Thanksgiving" in sched.detail.iloc[0]
    assert rep.scorecard.completeness_pct.iloc[0] == 100.0   # closure doesn't count


def test_same_gap_on_an_ordinary_day_is_still_a_data_gap():
    bars = _with_gap("2025-11-26 09:00", "2025-11-26 12:15", "2025-11-26 14:00")
    rep = run_quality(bars, QualityConfig(learn_session_from_data=False))
    assert (rep.issues.rule_id == "DATA_GAP").sum() == 1


def test_score_reflects_completeness_and_verdict_explains_itself():
    bars = _with_gap("2025-11-26 08:00", "2025-11-26 10:00", "2025-11-26 13:00", n=480)
    rep = run_quality(bars, QualityConfig(learn_session_from_data=False))
    sc = rep.scorecard.iloc[0]
    assert sc.validity_pct == 100.0
    assert sc.completeness_pct < 95 and sc.dq_score < 100   # used to show 100.0
    assert sc.verdict == CAUTION and "completeness" in sc.verdict_reason


def test_verdict_trusted_and_blocked():
    good = run_quality(make_bars(n=60)).scorecard.iloc[0]
    assert good.verdict == TRUSTED
    bad = make_bars(n=60)
    bad.loc[:5, "high"] = 0.1                                  # 10% OHLC breaches
    assert run_quality(bad).scorecard.iloc[0].verdict == BLOCKED


def test_liquidity_explained_gaps_are_not_sent_to_the_vendor():
    # Thin contract: many short gaps every day, tiny volume.
    frames = []
    for day in pd.date_range("2025-11-03", periods=10, freq="B"):
        b = make_bars(contract="SBH26", start=f"{day:%Y-%m-%d} 08:00", n=360, volume=3)
        keep = (b.timestamp.dt.minute < 20) | (b.timestamp.dt.minute > 55)
        frames.append(b[keep])
    data = pd.concat(frames, ignore_index=True)
    rep = run_quality(data, QualityConfig(learn_session_from_data=False))
    titles = [i.title for i in generate_insights(rep, data)]
    assert "SBH26 gaps look liquidity-driven" in titles
    assert "Recurring DATA_GAP on SBH26" not in titles


def test_dominant_gap_advice_does_not_claim_a_single_root_cause():
    from mdq.insights import RuleBasedInsights
    issues = pd.DataFrame({"rule_id": ["DATA_GAP"] * 9 + ["PRICE_OUTLIER"]})
    ins = RuleBasedInsights()._dominant_issue(issues)[0]
    assert "liquidity-aware" in ins.recommendation and "one root cause" in ins.recommendation
