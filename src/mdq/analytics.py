"""Market data analytics: daily OHLCV bars, rolling VWAP and filtering.

All functions expect the *clean* canonical frame produced by the quality
layer (errors removed), so a bad tick can never leak into a bar or VWAP.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import schema as S

SESSION = "session"    # exchange trade date: 17:00 CT (prev day) -> 16:00 CT
CALENDAR = "calendar"  # midnight-to-midnight in exchange time


def trading_date(ts: pd.Series, convention: str = SESSION) -> pd.Series:
    """Assign each bar to a trading date.

    ``session``: CME/CFE Globex trade date - bars from 17:00 CT onwards belong
    to the next day's session (a Sunday 17:00 bar is Monday's trade date).
    Shifting by +7h achieves this and is also correct for products whose
    session never crosses 17:00 (e.g. ICE sugar, CBOT grains day session).
    """
    if convention == SESSION:
        return (ts + pd.Timedelta(hours=7)).dt.normalize()
    if convention == CALENDAR:
        return ts.dt.normalize()
    raise ValueError(f"Unknown convention {convention!r}")


def filter_data(
    df: pd.DataFrame,
    contracts: Optional[Iterable[str]] = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Filter by contract list and inclusive date range (dates or datetimes).

    A date-only ``end`` (e.g. 2025-01-31) includes that whole day.
    """
    mask = pd.Series(True, index=df.index)
    if contracts:
        mask &= df[S.CONTRACT].isin(list(contracts))
    if start is not None:
        mask &= df[S.TIMESTAMP] >= pd.Timestamp(start)
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= df[S.TIMESTAMP] <= end_ts
    return df.loc[mask]


def daily_ohlcv(df: pd.DataFrame, convention: str = SESSION) -> pd.DataFrame:
    """Aggregate intraday bars to one OHLCV bar per contract per trading date."""
    cols = ["contract", "trading_date", "open", "high", "low", "close",
            "volume", "bar_count", "vwap", "first_bar", "last_bar"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.sort_values([S.CONTRACT, S.TIMESTAMP]).copy()
    d["trading_date"] = trading_date(d[S.TIMESTAMP], convention)
    d["_pv"] = _typical_price(d) * d[S.VOLUME].fillna(0)
    g = d.groupby([S.CONTRACT, "trading_date"], sort=True)
    out = g.agg(
        open=(S.OPEN, "first"),
        high=(S.HIGH, "max"),
        low=(S.LOW, "min"),
        close=(S.CLOSE, "last"),
        volume=(S.VOLUME, "sum"),
        bar_count=(S.CLOSE, "size"),
        _pv=("_pv", "sum"),
        first_bar=(S.TIMESTAMP, "min"),
        last_bar=(S.TIMESTAMP, "max"),
    ).reset_index()
    out["vwap"] = np.where(out["volume"] > 0, out["_pv"] / out["volume"].replace(0, np.nan), np.nan)
    out["volume"] = out["volume"].astype("int64")
    return out[cols]


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df[S.HIGH] + df[S.LOW] + df[S.CLOSE]) / 3.0


def rolling_vwap(df: pd.DataFrame, window: str = "15min", price: str = "typical") -> pd.DataFrame:
    """Rolling time-window VWAP per contract.

    For each bar at time t the VWAP covers bars with timestamp in
    (t - window, t], i.e. the last 15 one-minute bars when data is complete.
    The window is *time*-based, not count-based, so after a gap it only uses
    the bars that actually fall inside the last 15 minutes and never reaches
    back across a session break. A window with zero volume yields NaN rather
    than a misleading number.

    ``price``: ``typical`` = (H+L+C)/3 (default) or ``close``.
    """
    if df.empty:
        return df.assign(vwap_15m=pd.Series(dtype=float), window_volume=pd.Series(dtype=float))
    d = df.sort_values([S.CONTRACT, S.TIMESTAMP]).reset_index(drop=True)
    px = _typical_price(d) if price == "typical" else d[S.CLOSE]
    vol = d[S.VOLUME].fillna(0).astype(float)
    tmp = pd.DataFrame({"pv": px * vol, "v": vol, S.TIMESTAMP: d[S.TIMESTAMP], S.CONTRACT: d[S.CONTRACT]})
    parts = []
    for _, g in tmp.groupby(S.CONTRACT, sort=False):
        r = g.set_index(S.TIMESTAMP)[["pv", "v"]].rolling(window).sum()
        r.index = g.index
        parts.append(r)
    sums = pd.concat(parts).sort_index()
    d["window_volume"] = sums["v"]
    d["vwap_15m"] = (sums["pv"] / sums["v"].replace(0, np.nan)).astype(float)
    return d


def market_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-contract headline stats derived from daily bars."""
    if daily.empty:
        return pd.DataFrame()
    d = daily.sort_values(["contract", "trading_date"]).copy()
    d["ret"] = d.groupby("contract")["close"].pct_change()
    g = d.groupby("contract")
    return pd.DataFrame(
        {
            "first_date": g["trading_date"].min(),
            "last_date": g["trading_date"].max(),
            "trading_days": g.size(),
            "last_close": g["close"].last(),
            "period_return_pct": (g["close"].last() / g["close"].first() - 1) * 100,
            "avg_daily_volume": g["volume"].mean().round(0),
            "ann_volatility_pct": g["ret"].std() * np.sqrt(252) * 100,
        }
    ).round(2).reset_index()
