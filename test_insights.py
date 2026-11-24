import numpy as np
import pandas as pd

from conftest import make_bars
from mdq.insights import RuleBasedInsights, insights_frame
from mdq.quality import QualityConfig, run_quality

CFG = QualityConfig(max_timestamp=pd.Timestamp("2030-01-01"))


def _titles(df):
    rep = run_quality(df, CFG)
    return insights_frame(RuleBasedInsights().generate(rep, df))


def test_clean_data_gets_no_issue_recommendation():
    f = _titles(make_bars())
    assert f.title.tolist() == ["No issues found"]


def test_every_rule_hit_gets_a_recommendation():
    df = make_bars(n=60)
    df.loc[1, "close"] = np.nan
    df.loc[2, "high"] = df.low[2] - 1
    f = _titles(df)
    recs = f[f.kind == "recommendation"]
    assert set(recs.rule_id) == {"MISSING_VALUE", "OHLC_INCONSISTENT"}
    assert (recs.priority == "High").all()


def test_simultaneous_gaps_across_contracts_suggest_outage():
    frames = []
    for c in ("A", "B"):
        d = make_bars(contract=c, n=200)
        frames.append(d.drop(index=range(60, 120)))
    f = _titles(pd.concat(frames, ignore_index=True))
    assert "Gaps occur simultaneously across contracts" in f.title.tolist()


def test_error_concentration_in_one_contract():
    a = make_bars(contract="A", n=100)
    b = make_bars(contract="B", n=100)
    a.loc[:9, "close"] = np.nan
    f = _titles(pd.concat([a, b], ignore_index=True))
    assert "Errors concentrated in A" in f.title.tolist()


def test_recurring_daily_issue_detected():
    days = [make_bars(start=f"2025-11-{d:02d} 09:00", n=30) for d in (3, 4, 5, 6, 7, 10)]
    for d in days:
        d.loc[d.index[0], "high"] = d.low.iloc[0] - 1
    f = _titles(pd.concat(days, ignore_index=True))
    assert any(t.startswith("Recurring OHLC_INCONSISTENT") for t in f.title)
    assert any(t.startswith("OHLC_INCONSISTENT clusters around 09:00") for t in f.title) is False  # <10 issues


def test_patterns_listed_before_recommendations():
    days = [make_bars(start=f"2025-11-{d:02d} 09:00", n=30) for d in (3, 4, 5, 6, 7, 10)]
    for d in days:
        d.loc[d.index[0], "high"] = d.low.iloc[0] - 1
    f = _titles(pd.concat(days, ignore_index=True))
    kinds = f.kind.tolist()
    assert kinds == sorted(kinds, key=lambda k: k != "pattern")
