import numpy as np
import pandas as pd
import pytest

from conftest import make_bars
from mdq.analytics import CALENDAR, SESSION, daily_ohlcv, filter_data, market_summary, rolling_vwap, trading_date


def test_daily_ohlcv_values():
    df = pd.DataFrame({
        "contract": "ES",
        "timestamp": pd.to_datetime(["2025-11-10 09:00", "2025-11-10 09:01", "2025-11-10 09:02"]),
        "open": [10.0, 11.0, 12.0], "high": [12.0, 15.0, 13.0],
        "low": [9.0, 10.0, 8.0], "close": [11.0, 12.0, 9.0], "volume": [1.0, 2.0, 3.0],
    })
    d = daily_ohlcv(df.sample(frac=1, random_state=1)).iloc[0]  # order-independent
    assert (d.open, d.high, d.low, d.close, d.volume, d.bar_count) == (10, 15, 8, 9, 6, 3)
    tp = (df.high + df.low + df.close) / 3
    assert d.vwap == pytest.approx((tp * df.volume).sum() / 6)


def test_session_convention_rolls_evening_bars_forward():
    ts = pd.Series(pd.to_datetime(["2025-11-09 17:00", "2025-11-10 15:59", "2025-11-10 17:00"]))
    assert trading_date(ts, SESSION).dt.day.tolist() == [10, 10, 11]
    assert trading_date(ts, CALENDAR).dt.day.tolist() == [9, 10, 10]


def test_daily_session_vs_calendar_bucketing():
    df = pd.concat([make_bars(start="2025-11-09 17:00", n=10), make_bars(start="2025-11-10 09:00", n=10)])
    assert len(daily_ohlcv(df, SESSION)) == 1
    assert len(daily_ohlcv(df, CALENDAR)) == 2


def test_rolling_vwap_matches_hand_calculation():
    df = make_bars(n=20)
    df["volume"] = np.arange(1, 21, dtype=float)
    v = rolling_vwap(df, price="close")
    # Bar 19 (index) covers bars 5..19 (15 bars)
    w = df.iloc[5:20]
    assert v.vwap_15m.iloc[19] == pytest.approx((w.close * w.volume).sum() / w.volume.sum())
    # First bar is just itself
    assert v.vwap_15m.iloc[0] == pytest.approx(df.close.iloc[0])


def test_rolling_vwap_is_time_based_across_gaps():
    df = pd.concat([make_bars(n=5, price=100), make_bars(start="2025-11-10 10:00", n=1, price=200)], ignore_index=True)
    v = rolling_vwap(df, price="close")
    assert v.vwap_15m.iloc[-1] == pytest.approx(200)  # earlier bars are >15 min old


def test_rolling_vwap_zero_volume_is_nan_and_contracts_independent():
    a = make_bars(contract="A", n=3, volume=0)
    b = make_bars(contract="B", n=3, price=50)
    v = rolling_vwap(pd.concat([a, b]))
    assert v[v.contract == "A"].vwap_15m.isna().all()
    assert v[v.contract == "B"].vwap_15m.notna().all()
    assert v[v.contract == "B"].vwap_15m.max() < 60  # not polluted by A


def test_filter_by_contract_and_inclusive_dates():
    df = pd.concat([make_bars(contract="A", start="2025-11-10 09:00"), make_bars(contract="B", start="2025-11-12 09:00")])
    assert set(filter_data(df, ["A"]).contract) == {"A"}
    assert len(filter_data(df, None, "2025-11-12", "2025-11-12")) == 30  # whole end day included
    assert filter_data(df, ["A"], "2025-11-11").empty


def test_empty_inputs():
    empty = make_bars().iloc[0:0]
    assert daily_ohlcv(empty).empty
    assert rolling_vwap(empty).empty


def test_market_summary():
    df = pd.concat([make_bars(start=f"2025-11-1{d} 09:00", n=10, price=100 + d) for d in range(3)])
    s = market_summary(daily_ohlcv(df)).iloc[0]
    assert s.trading_days == 3 and s.period_return_pct > 0
