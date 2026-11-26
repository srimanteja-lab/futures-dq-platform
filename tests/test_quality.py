import numpy as np
import pandas as pd

from conftest import make_bars
from mdq.quality import (
    ERROR, WARNING, QualityConfig, default_session_mask, learn_session_mask, run_quality,
)

CFG = QualityConfig(max_timestamp=pd.Timestamp("2030-01-01"))


def rules(rep, rule_id):
    return rep.issues[rep.issues.rule_id == rule_id]


def test_clean_data_has_no_issues(bars):
    rep = run_quality(bars, CFG)
    assert rep.issues.empty
    assert len(rep.clean) == len(bars)
    assert rep.scorecard.dq_score.iloc[0] == 100


def test_missing_price_is_error_and_excluded(bars):
    bars.loc[5, "close"] = np.nan
    bars.loc[6, "volume"] = np.nan
    rep = run_quality(bars, CFG)
    mv = rules(rep, "MISSING_VALUE")
    assert set(zip(mv.row_id, mv.severity)) == {(6, ERROR), (7, WARNING)}
    assert len(rep.clean) == len(bars) - 1  # missing-volume row kept


def test_exact_and_conflicting_duplicates(bars):
    exact = bars.iloc[[3]]
    corr = bars.iloc[[4]].assign(close=bars.close[4] - 0.1)
    df = pd.concat([bars, exact, corr], ignore_index=True)
    rep = run_quality(df, CFG)
    assert rules(rep, "DUPLICATE_EXACT").row_id.tolist() == [31]
    conflict = rules(rep, "DUPLICATE_CONFLICT")
    assert conflict.row_id.tolist() == [5]  # original superseded by the later correction
    kept = rep.clean[rep.clean.timestamp == bars.timestamp[4]]
    assert len(kept) == 1 and kept.close.iloc[0] == bars.close[4] - 0.1
    assert len(rep.clean) == len(bars)


def test_invalid_prices_and_negative_allowed_switch(bars):
    bars.loc[2, "low"] = -1.0
    bars.loc[3, ["open", "high", "low", "close"]] = 0.0
    rep = run_quality(bars, CFG)
    assert sorted(rules(rep, "INVALID_PRICE").row_id) == [3, 4]
    rep2 = run_quality(bars, QualityConfig(allow_negative_prices=True, max_timestamp=CFG.max_timestamp))
    assert rules(rep2, "INVALID_PRICE").empty


def test_ohlc_consistency(bars):
    bars.loc[1, "high"] = bars.low[1] - 1  # high below low
    bars.loc[2, "close"] = bars.high[2] + 1  # close above high
    bars.loc[3, "low"] = bars.open[3] + 0.01  # low above open
    rep = run_quality(bars, CFG)
    assert sorted(rules(rep, "OHLC_INCONSISTENT").row_id) == [2, 3, 4]


def test_volume_rules(bars):
    bars.loc[1, "volume"] = -5
    bars.loc[2, "volume"] = 1.5
    bars.loc[3, "volume"] = 0  # with a price range -> warning
    bars.loc[4, ["volume", "open", "high", "low", "close"]] = [0, 100, 100, 100, 100]  # flat -> fine
    rep = run_quality(bars, CFG)
    assert sorted(rules(rep, "INVALID_VOLUME").row_id) == [2, 3]
    assert rules(rep, "ZERO_VOLUME_PRICE_MOVE").row_id.tolist() == [4]


def test_future_and_saturday_timestamps():
    df = pd.concat([make_bars(n=3), make_bars(start="2031-01-06 09:00", n=1), make_bars(start="2025-11-15 10:00", n=1)],
                   ignore_index=True)
    rep = run_quality(df, CFG)
    ts = rules(rep, "TIMESTAMP_INVALID")
    assert set(zip(ts.row_id, ts.severity)) == {(4, ERROR), (5, WARNING)}


def test_gap_inside_session_is_flagged():
    df = make_bars(n=200)
    df = df.drop(index=range(50, 110)).reset_index(drop=True)  # 60-minute hole
    rep = run_quality(df, CFG)
    gaps = rules(rep, "DATA_GAP")
    assert len(gaps) == 1
    assert gaps.gap_minutes.iloc[0] == 60
    assert gaps.timestamp.iloc[0] == pd.Timestamp("2025-11-10 09:50")


def test_gap_below_threshold_not_flagged():
    df = make_bars(n=100).drop(index=range(40, 60)).reset_index(drop=True)  # 20 min
    assert rules(run_quality(df, CFG), "DATA_GAP").empty


def test_scheduled_closures_are_not_gaps():
    # Thu 15:00-15:59, then Thu 17:00 (daily break 16-17) ... Fri 15:00-15:59, Sun 17:00 open.
    df = pd.concat([
        make_bars(start="2025-11-13 15:00", n=60),
        make_bars(start="2025-11-13 17:00", n=60),
        make_bars(start="2025-11-14 15:00", n=60),
        make_bars(start="2025-11-16 17:00", n=60),
    ], ignore_index=True)
    gaps = rules(run_quality(df, CFG), "DATA_GAP")
    # Only the overnight Thu 18:00 -> Fri 15:00 hole is in-session.
    assert len(gaps) == 1
    assert gaps.timestamp.iloc[0] == pd.Timestamp("2025-11-13 18:00")


def test_default_session_mask_shape():
    m = default_session_mask()
    assert m.size == 7 * 1440
    assert not m[5 * 1440 + 600]  # Saturday 10:00
    assert m[0 * 1440 + 600]  # Monday 10:00
    assert not m[2 * 1440 + 16 * 60 + 30]  # Wednesday 16:30 maintenance


def test_learned_session_needs_history():
    one_week = make_bars(n=60)["timestamp"]
    assert learn_session_mask(one_week, CFG) is None
    weeks = pd.concat([make_bars(start=f"2025-11-{d} 09:00", n=60)["timestamp"] for d in (3, 10, 17, 24)])
    m = learn_session_mask(weeks, CFG)
    assert m[0 * 1440 + 9 * 60 + 30] and not m[0 * 1440 + 12 * 60]


def test_outlier_flags_spike_once_and_labels_bad_tick():
    df = make_bars(n=120, step=0.0)
    rng = np.random.default_rng(0)
    df["close"] = 100 + rng.normal(0, 0.02, len(df)).cumsum()
    df["high"] = df.close + 0.05
    df["low"] = df.close - 0.05
    df["open"] = df.close
    df.loc[80, ["close", "high"]] = df.close[80] * 1.05
    out = rules(run_quality(df, CFG), "PRICE_OUTLIER")
    assert out.row_id.tolist() == [81]  # spike only, not the reversal bar
    assert "likely bad tick" in out.detail.iloc[0]


def test_flat_prices_do_not_create_infinite_z_outliers():
    df = make_bars(n=120, step=0.0)
    df.loc[60, "close"] = df.close[60] + 0.25  # one tick move in a flat market
    assert rules(run_quality(df, CFG), "PRICE_OUTLIER").empty


def test_rejected_records_reported_as_issues(bars):
    rejected = pd.DataFrame({"source_file": ["f.csv"], "row_id": [9], "reason": ["MALFORMED_OPEN"], "raw_record": ["x"]})
    rep = run_quality(bars, CFG, rejected=rejected)
    m = rules(rep, "MALFORMED_RECORD")
    assert len(m) == 1 and m.severity.iloc[0] == ERROR


def test_custom_rule_can_be_plugged_in(bars):
    from mdq.quality import DEFAULT_RULES, Rule, _issues

    class BigVolume(Rule):
        rule_id, category, severity = "BIG_VOLUME", "Custom", WARNING

        def check(self, df, cfg):
            return _issues(df, df.volume > 1000, self, "large print")

    bars.loc[0, "volume"] = 5000
    rep = run_quality(bars, CFG, rules=DEFAULT_RULES + [BigVolume()])
    assert rules(rep, "BIG_VOLUME").row_id.tolist() == [1]


def test_scorecard_counts(bars):
    bars.loc[0, "close"] = np.nan
    df = pd.concat([bars, bars.iloc[[5]]], ignore_index=True)
    sc = run_quality(df, CFG).scorecard.iloc[0]
    assert (sc.records, sc.error_records, sc.warning_records, sc.clean_records) == (31, 1, 1, 29)
