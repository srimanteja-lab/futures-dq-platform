"""US futures exchange holiday & early-close calendar (CME Group style).

Why this exists: on the real data the insight engine ranked Thanksgiving,
Black Friday and Christmas Eve early closes as HIGH-priority "feed outages".
A gap that overlaps a scheduled closure is not a data-quality problem, so the
gap rule labels it SCHEDULED_CLOSURE (INFO) instead of DATA_GAP (WARNING).

The rules below are generated, not hand-typed per year, so they cover any
range of history. Exact halt times differ by product; the business should
confirm the golden-source calendar (see README "open questions").
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Dict, Tuple

CLOSED, EARLY = "closed", "early close"

# One-off market closures / early closes not covered by the yearly rules.
SPECIAL_DAYS: Dict[dt.date, Tuple[str, str]] = {
    dt.date(2025, 1, 9): ("National Day of Mourning (President Carter)", EARLY),
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _easter(year: int) -> dt.date:
    # Anonymous Gregorian algorithm
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


@lru_cache(maxsize=None)
def year_calendar(year: int) -> Dict[dt.date, Tuple[str, str]]:
    cal: Dict[dt.date, Tuple[str, str]] = {}
    ny = dt.date(year, 1, 1)
    if ny.weekday() == 6:
        ny += dt.timedelta(days=1)
    if ny.weekday() < 5:
        cal[ny] = ("New Year's Day", CLOSED)
    cal[_nth_weekday(year, 1, 0, 3)] = ("Martin Luther King Jr. Day", EARLY)
    cal[_nth_weekday(year, 2, 0, 3)] = ("Presidents' Day", EARLY)
    cal[_easter(year) - dt.timedelta(days=2)] = ("Good Friday", CLOSED)
    cal[_last_weekday(year, 5, 0)] = ("Memorial Day", EARLY)
    if year >= 2022:
        cal[_observed(dt.date(year, 6, 19))] = ("Juneteenth", EARLY)
    july4 = _observed(dt.date(year, 7, 4))
    cal[july4] = ("Independence Day", EARLY)
    eve = july4 - dt.timedelta(days=1)
    if eve.weekday() < 5:
        cal.setdefault(eve, ("Day before Independence Day", EARLY))
    cal[_nth_weekday(year, 9, 0, 1)] = ("Labor Day", EARLY)
    thanks = _nth_weekday(year, 11, 3, 4)
    cal[thanks] = ("Thanksgiving", EARLY)
    cal[thanks + dt.timedelta(days=1)] = ("Day after Thanksgiving", EARLY)
    xmas = _observed(dt.date(year, 12, 25))
    cal[xmas] = ("Christmas Day", CLOSED)
    xeve = dt.date(year, 12, 24)
    if xeve.weekday() < 5 and xeve != xmas:
        cal.setdefault(xeve, ("Christmas Eve", EARLY))
    for d, v in SPECIAL_DAYS.items():
        if d.year == year:
            cal[d] = v
    return cal


def lookup(day: dt.date):
    """Return (name, kind) if ``day`` is a holiday or early close, else None."""
    return year_calendar(day.year).get(day)


def closure_between(start, end):
    """First scheduled closure on any calendar day from ``start`` to ``end``
    (inclusive). Returns (date, name, kind) or None."""
    d, last = start.date(), end.date()
    while d <= last:
        hit = lookup(d)
        if hit:
            return d, hit[0], hit[1]
        d += dt.timedelta(days=1)
    return None
