"""Data quality layer: a pluggable rule engine.

Each rule is a small class that inspects the canonical frame and returns
*issues* (one row per finding). Severity drives what happens to a record:

* ERROR   - record is excluded from analytics (it would corrupt OHLCV/VWAP).
* WARNING - record is kept but flagged for review (e.g. outliers, gaps).
* INFO    - observation only.

To add a rule: subclass :class:`Rule`, implement ``check`` and append an
instance to ``DEFAULT_RULES`` (or pass a custom list to ``run_quality``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import exchange_calendar as XC
from . import schema as S

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"

# Publication-gate verdicts (plain English for traders / risk).
TRUSTED, CAUTION, BLOCKED = "Trusted", "Use with caution", "Blocked"
ISSUE_COLUMNS = [
    "rule_id", "category", "severity", S.CONTRACT, S.TIMESTAMP,
    S.SOURCE, S.ROW_ID, "detail",
]

MINUTES_PER_WEEK = 7 * 24 * 60
# Longest gap a scheduled closure can explain (e.g. Thu 12:00 -> Sun 17:00).
MAX_CLOSURE_SPAN = pd.Timedelta(days=4)
BUCKET_MINUTES = 15
_MONDAY_EPOCH = pd.Timestamp("1970-01-05")  # a Monday


@dataclass
class QualityConfig:
    # Gap detection
    gap_threshold_minutes: int = 30
    learn_session_from_data: bool = True
    min_weeks_for_learned_session: int = 3
    session_presence_ratio: float = 0.5
    # Outlier detection (robust z-score on 1-bar log returns)
    detect_outliers: bool = True
    outlier_window: int = 60
    outlier_z: float = 10.0
    outlier_min_abs_return: float = 0.003
    # Price validation. Some futures CAN trade negative (WTI crude, April
    # 2020; calendar spreads), so this is configurable rather than hard-coded.
    allow_negative_prices: bool = False
    # Anything later than "now" is impossible for historical data.
    max_timestamp: Optional[pd.Timestamp] = None
    # Gaps that overlap an exchange holiday / early close are scheduled, not defects.
    use_holiday_calendar: bool = True
    # Publication gate (business-owned thresholds, in %).
    gate_min_validity: float = 99.0       # below -> Blocked
    gate_min_completeness: float = 95.0   # below -> Use with caution


def _issues(df: pd.DataFrame, mask, rule: "Rule", detail, severity=None) -> pd.DataFrame:
    hit = df.loc[mask]
    if hit.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    if callable(detail):
        detail = detail(hit)
    return pd.DataFrame(
        {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "severity": severity if severity is not None else rule.severity,
            S.CONTRACT: hit[S.CONTRACT].values,
            S.TIMESTAMP: hit[S.TIMESTAMP].values,
            S.SOURCE: hit[S.SOURCE].values if S.SOURCE in hit else None,
            S.ROW_ID: hit[S.ROW_ID].values if S.ROW_ID in hit else None,
            "detail": detail if not isinstance(detail, pd.Series) else detail.values,
        }
    )


def _concat(frames) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ISSUE_COLUMNS)


class Rule(ABC):
    rule_id: str = ""
    category: str = ""
    severity: str = WARNING
    description: str = ""

    @abstractmethod
    def check(self, df: pd.DataFrame, cfg: QualityConfig) -> pd.DataFrame:
        ...


# --------------------------------------------------------------- completeness
class MissingValueRule(Rule):
    rule_id, category, severity = "MISSING_VALUE", "Completeness", ERROR
    description = "Required price field is blank (volume-only blanks are warnings)."

    def check(self, df, cfg):
        price_null = df[S.PRICE_COLS].isna()
        any_price = price_null.any(axis=1)
        vol_only = ~any_price & df[S.VOLUME].isna()

        def cols(hit):
            return price_null.loc[hit.index].apply(
                lambda r: "missing " + ",".join(c for c, v in r.items() if v), axis=1
            )

        return _concat([
            _issues(df, any_price, self, cols),
            _issues(df, vol_only, self, "missing volume", severity=WARNING),
        ])


# ----------------------------------------------------------------- uniqueness
class ExactDuplicateRule(Rule):
    rule_id, category, severity = "DUPLICATE_EXACT", "Uniqueness", WARNING
    description = "Identical record received more than once; extra copies dropped."

    def check(self, df, cfg):
        mask = df.duplicated(subset=S.CANONICAL_COLS, keep="first")
        return _issues(df, mask, self, "exact duplicate of an earlier record (dropped)")


class ConflictingDuplicateRule(Rule):
    rule_id, category, severity = "DUPLICATE_CONFLICT", "Uniqueness", ERROR
    description = (
        "Same contract+timestamp with different values. The last-received "
        "version is retained (vendor-correction semantics); earlier ones excluded."
    )

    def check(self, df, cfg):
        uniq = df.drop_duplicates(subset=S.CANONICAL_COLS, keep="first")
        key = [S.CONTRACT, S.TIMESTAMP]
        dup_key = uniq.duplicated(subset=key, keep=False)
        superseded = dup_key & uniq.duplicated(subset=key, keep="last")
        mask = df.index.isin(uniq.index[superseded])
        return _issues(df, mask, self, "conflicting duplicate; superseded by a later record")


# ------------------------------------------------------------------- validity
class InvalidPriceRule(Rule):
    rule_id, category, severity = "INVALID_PRICE", "Validity", ERROR
    description = "Price is zero/negative (unless negative prices are allowed)."

    def check(self, df, cfg):
        if cfg.allow_negative_prices:
            return pd.DataFrame(columns=ISSUE_COLUMNS)
        bad = df[S.PRICE_COLS] <= 0
        mask = bad.any(axis=1)
        return _issues(
            df, mask, self,
            lambda h: bad.loc[h.index].apply(
                lambda r: "non-positive " + ",".join(c for c, v in r.items() if v), axis=1
            ),
        )


class OHLCConsistencyRule(Rule):
    rule_id, category, severity = "OHLC_INCONSISTENT", "Validity", ERROR
    description = "High must be >= Open/Close/Low and Low must be <= Open/Close."

    def check(self, df, cfg):
        o, h, l, c = (df[x] for x in S.PRICE_COLS)
        complete = df[S.PRICE_COLS].notna().all(axis=1)
        mask = complete & (
            (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
        )
        return _issues(
            df, mask, self,
            lambda x: "O=" + x[S.OPEN].astype(str) + " H=" + x[S.HIGH].astype(str)
            + " L=" + x[S.LOW].astype(str) + " C=" + x[S.CLOSE].astype(str),
        )


class InvalidVolumeRule(Rule):
    rule_id, category, severity = "INVALID_VOLUME", "Validity", ERROR
    description = "Volume must be a non-negative whole number."

    def check(self, df, cfg):
        v = df[S.VOLUME]
        mask = v.notna() & ((v < 0) | (v != np.floor(v)))
        return _issues(df, mask, self, lambda x: "volume=" + x[S.VOLUME].astype(str))


class ZeroVolumePriceMoveRule(Rule):
    rule_id, category, severity = "ZERO_VOLUME_PRICE_MOVE", "Validity", WARNING
    description = (
        "Bar shows a price range but zero volume. Flat zero-volume bars are "
        "normal (settlement-style marks on deferred contracts) and not flagged."
    )

    def check(self, df, cfg):
        mask = (df[S.VOLUME] == 0) & (df[S.HIGH] > df[S.LOW])
        return _issues(df, mask, self, "price range with zero volume")


# ----------------------------------------------------------------- timeliness
class TimestampValidityRule(Rule):
    rule_id, category, severity = "TIMESTAMP_INVALID", "Timeliness", ERROR
    description = "Timestamp is in the future, or falls on a Saturday (no futures session)."

    def check(self, df, cfg):
        now = cfg.max_timestamp or pd.Timestamp.now(tz=S.EXCHANGE_TZ).tz_localize(None)
        future = df[S.TIMESTAMP] > now
        saturday = df[S.TIMESTAMP].dt.dayofweek == 5
        return _concat([
            _issues(df, future, self, "timestamp is in the future"),
            _issues(df, saturday & ~future, self, "timestamp falls on a Saturday", severity=WARNING),
        ])


def _minute_of_week(ts: pd.Series) -> np.ndarray:
    return ((ts - _MONDAY_EPOCH) // pd.Timedelta(minutes=1)).to_numpy(dtype="int64")


def default_session_mask() -> np.ndarray:
    """Generic CME Globex-style week (exchange time): Sun 17:00 -> Fri 16:00,
    with a daily 16:00-17:00 maintenance break. Used when there is not enough
    history to learn a contract's real trading hours."""
    m = np.ones(MINUTES_PER_WEEK, dtype=bool)
    day = 1440
    for d in range(7):  # 0=Mon
        m[d * day + 16 * 60: d * day + 17 * 60] = False  # daily break
    m[4 * day + 16 * 60: 5 * day] = False  # Fri after 16:00
    m[5 * day: 6 * day] = False  # Saturday
    m[6 * day: 6 * day + 17 * 60] = False  # Sunday before 17:00
    return m


def learn_session_mask(ts: pd.Series, cfg: QualityConfig) -> Optional[np.ndarray]:
    """Learn when a contract normally trades: a 15-minute bucket of the week
    is 'in session' if the contract printed in it in at least
    ``session_presence_ratio`` of its active weeks. Returns None when there is
    too little history."""
    mow = _minute_of_week(ts)
    week = mow // MINUTES_PER_WEEK
    n_weeks = len(np.unique(week))
    if n_weeks < cfg.min_weeks_for_learned_session:
        return None
    bucket = (mow % MINUTES_PER_WEEK) // BUCKET_MINUTES
    pairs = np.unique(np.stack([week, bucket]), axis=1)
    counts = np.bincount(pairs[1], minlength=MINUTES_PER_WEEK // BUCKET_MINUTES)
    active = counts / n_weeks >= cfg.session_presence_ratio
    return np.repeat(active, BUCKET_MINUTES)


def _expected_minutes_between(start_m: np.ndarray, end_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Count in-session minutes in [start, end) using a weekly prefix sum."""
    cum = np.concatenate([[0], np.cumsum(mask)])
    per_week = cum[-1]

    def E(m):
        return (m // MINUTES_PER_WEEK) * per_week + cum[m % MINUTES_PER_WEEK]

    return E(end_m) - E(start_m)


class GapRule(Rule):
    rule_id, category, severity = "DATA_GAP", "Completeness", WARNING
    description = (
        "Missing bars during the contract's normal trading hours (learned from "
        "its own history, or a default Globex calendar). Scheduled closures, "
        "weekends and the daily break are not reported."
    )

    def check(self, df, cfg):
        out = []
        for contract, g in df.groupby(S.CONTRACT, sort=False):
            ts = g[S.TIMESTAMP].drop_duplicates().sort_values()
            if len(ts) < 2:
                continue
            mask = learn_session_mask(ts, cfg) if cfg.learn_session_from_data else None
            source = "learned" if mask is not None else "default Globex"
            if mask is None:
                mask = default_session_mask()
            m = _minute_of_week(ts)
            start = m[:-1] + 1  # first missing minute
            end = m[1:]  # next bar (exclusive)
            missing_total = end - start
            candidates = missing_total >= cfg.gap_threshold_minutes
            if not candidates.any():
                continue
            expected = _expected_minutes_between(start[candidates], end[candidates], mask)
            flag = expected >= cfg.gap_threshold_minutes
            if not flag.any():
                continue
            idx = np.flatnonzero(candidates)[flag]
            prev_ts = ts.iloc[idx].to_numpy()
            next_ts = ts.iloc[idx + 1].to_numpy()
            exp = expected[flag]
            first_rows = g.drop_duplicates(S.TIMESTAMP).set_index(S.TIMESTAMP)
            nxt = first_rows.loc[next_ts]
            # Scheduled closures (holidays / early closes) are not data defects.
            # Only short gaps can be explained by a closure: a gap spanning
            # months that happens to contain a holiday is still a real gap.
            closures = [
                XC.closure_between(pd.Timestamp(a), pd.Timestamp(b))
                if cfg.use_holiday_calendar and pd.Timestamp(b) - pd.Timestamp(a) <= MAX_CLOSURE_SPAN
                else None
                for a, b in zip(prev_ts, next_ts)
            ]
            rule_ids, sevs, details = [], [], []
            for a, b, e, c in zip(prev_ts, next_ts, exp, closures):
                window = (f"no bars {pd.Timestamp(a):%Y-%m-%d %H:%M} -> {pd.Timestamp(b):%Y-%m-%d %H:%M} "
                          f"({int(e)} in-session minutes")
                if c:
                    rule_ids.append(ScheduledClosureRule.rule_id)
                    sevs.append(INFO)
                    details.append(f"{window}; {c[1]} {c[0]:%d %b} ({c[2]}), not a data issue)")
                else:
                    rule_ids.append(self.rule_id)
                    sevs.append(self.severity)
                    details.append(f"{window} missing; {source} session)")
            out.append(
                pd.DataFrame(
                    {
                        "rule_id": rule_ids,
                        "category": self.category,
                        "severity": sevs,
                        S.CONTRACT: contract,
                        S.TIMESTAMP: pd.to_datetime(prev_ts) + pd.Timedelta(minutes=1),
                        S.SOURCE: nxt[S.SOURCE].to_numpy() if S.SOURCE in nxt else None,
                        S.ROW_ID: nxt[S.ROW_ID].to_numpy() if S.ROW_ID in nxt else None,
                        "detail": details,
                        "gap_minutes": exp.astype(int),
                    }
                )
            )
        if not out:
            return pd.DataFrame(columns=ISSUE_COLUMNS + ["gap_minutes"])
        return pd.concat(out, ignore_index=True)


class ScheduledClosureRule(Rule):
    """Not run on its own: GapRule emits it for gaps that overlap an exchange
    holiday or early close. Listed so it appears in the rule catalogue."""
    rule_id, category, severity = "SCHEDULED_CLOSURE", "Completeness", INFO
    description = ("Gap that overlaps an exchange holiday or early close "
                   "(e.g. Thanksgiving). Reported for transparency, not a defect.")

    def check(self, df, cfg):
        return pd.DataFrame(columns=ISSUE_COLUMNS)


# ---------------------------------------------------------------- consistency
class OutlierRule(Rule):
    rule_id, category, severity = "PRICE_OUTLIER", "Consistency", WARNING
    description = (
        "Bar-to-bar close return is extreme versus the contract's recent "
        "behaviour (robust z-score using rolling median/MAD). Spikes that "
        "immediately reverse are labelled as likely bad ticks."
    )

    def check(self, df, cfg):
        if not cfg.detect_outliers:
            return pd.DataFrame(columns=ISSUE_COLUMNS)
        d = df[[S.CONTRACT, S.TIMESTAMP, S.CLOSE]].dropna()
        d = d[d[S.CLOSE] > 0].sort_values([S.CONTRACT, S.TIMESTAMP])
        if d.empty:
            return pd.DataFrame(columns=ISSUE_COLUMNS)
        r = np.log(d[S.CLOSE]).groupby(d[S.CONTRACT]).diff()
        grp = r.groupby(d[S.CONTRACT])
        w = cfg.outlier_window
        minp = max(10, w // 3)
        med = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=minp).median())
        mad = (r - med).abs().groupby(d[S.CONTRACT]).transform(
            lambda x: x.shift(1).rolling(w, min_periods=minp).median()
        )
        std = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=minp).std())
        scale = (1.4826 * mad).where(mad > 0, std)
        z = (r - med) / scale.replace(0, np.nan)
        flag = (z.abs() > cfg.outlier_z) & (r.abs() > cfg.outlier_min_abs_return)
        # Spike-and-revert: next return has opposite sign and similar size.
        nxt = grp.shift(-1)
        revert = flag & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > 0.5 * r.abs())
        # The bar after a bad tick merely returns to normal - don't flag it too.
        after_bad_tick = revert.groupby(d[S.CONTRACT]).shift(1, fill_value=False).astype(bool)
        flag = flag & ~after_bad_tick
        hit_idx = d.index[flag.fillna(False).to_numpy()]
        detail = pd.Series(
            [
                f"return {r[i]:+.2%} (robust z={z[i]:.1f})"
                + ("; reverses next bar - likely bad tick" if bool(revert[i]) else "")
                for i in hit_idx
            ],
            index=hit_idx,
            dtype=object,
        )
        mask = df.index.isin(hit_idx)
        return _issues(df, mask, self, lambda h: detail.reindex(h.index))


DEFAULT_RULES: List[Rule] = [
    MissingValueRule(),
    ExactDuplicateRule(),
    ConflictingDuplicateRule(),
    InvalidPriceRule(),
    OHLCConsistencyRule(),
    InvalidVolumeRule(),
    ZeroVolumePriceMoveRule(),
    TimestampValidityRule(),
    GapRule(),
    OutlierRule(),
]


def rule_catalogue(rules: List[Rule] = None) -> pd.DataFrame:
    rules = rules or DEFAULT_RULES + [ScheduledClosureRule()]
    return pd.DataFrame(
        [
            {"rule_id": r.rule_id, "category": r.category,
             "default_severity": r.severity, "description": r.description}
            for r in rules
        ]
    )


@dataclass
class QualityReport:
    issues: pd.DataFrame
    clean: pd.DataFrame
    scorecard: pd.DataFrame
    config: QualityConfig = field(default_factory=QualityConfig)

    def counts(self) -> pd.DataFrame:
        if self.issues.empty:
            return pd.DataFrame(columns=["rule_id", "severity", "count"])
        return (
            self.issues.groupby(["rule_id", "severity"]).size()
            .rename("count").reset_index().sort_values("count", ascending=False)
        )


def rejected_to_issues(rejected: pd.DataFrame) -> pd.DataFrame:
    """Express ingestion rejections as ERROR issues so they appear in reports."""
    if rejected is None or rejected.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    return pd.DataFrame(
        {
            "rule_id": "MALFORMED_RECORD",
            "category": "Ingestion",
            "severity": ERROR,
            S.CONTRACT: None,
            S.TIMESTAMP: pd.NaT,
            S.SOURCE: rejected[S.SOURCE].values,
            S.ROW_ID: rejected[S.ROW_ID].values,
            "detail": (rejected["reason"] + ": " + rejected["raw_record"].astype(str).str.slice(0, 120)).values,
        }
    )


def _verdict(row, cfg: QualityConfig):
    """Publication gate: plain-English verdict plus the reasons behind it."""
    reasons = []
    if row["validity_pct"] < cfg.gate_min_validity:
        reasons.append(f"validity {row['validity_pct']:.2f}% < {cfg.gate_min_validity:g}%")
    if row["completeness_pct"] < cfg.gate_min_completeness:
        reasons.append(f"completeness {row['completeness_pct']:.1f}% < {cfg.gate_min_completeness:g}%")
    if row["validity_pct"] < cfg.gate_min_validity:
        return BLOCKED, "; ".join(reasons)
    if reasons:
        return CAUTION, "; ".join(reasons)
    extra = []
    if row["error_records"]:
        extra.append(f"{row['error_records']} bad record(s) excluded")
    if row["warning_records"]:
        extra.append(f"{row['warning_records']} flagged")
    return TRUSTED, "passes all gates" + (" (" + ", ".join(extra) + ")" if extra else "")


def _scorecard(df: pd.DataFrame, issues: pd.DataFrame, clean: pd.DataFrame,
               cfg: QualityConfig = None) -> pd.DataFrame:
    cfg = cfg or QualityConfig()
    total = df.groupby(S.CONTRACT).size().rename("records")
    kept = clean.groupby(S.CONTRACT).size().rename("clean_records")
    rec_issues = issues[issues[S.ROW_ID].notna() & (issues["rule_id"] != "DATA_GAP")]
    key = rec_issues[S.SOURCE].astype(str) + "#" + rec_issues[S.ROW_ID].astype(str)
    err = rec_issues[rec_issues["severity"] == ERROR].assign(k=key).groupby(S.CONTRACT)["k"].nunique().rename("error_records")
    warn = rec_issues[rec_issues["severity"] == WARNING].assign(k=key).groupby(S.CONTRACT)["k"].nunique().rename("warning_records")
    gaps = issues[issues["rule_id"] == "DATA_GAP"]
    gap_n = gaps.groupby(S.CONTRACT).size().rename("gaps")
    gap_m = (gaps.groupby(S.CONTRACT)["gap_minutes"].sum() if "gap_minutes" in gaps else pd.Series(dtype=float)).rename("gap_minutes")
    first = df.groupby(S.CONTRACT)[S.TIMESTAMP].min().rename("first_bar")
    last = df.groupby(S.CONTRACT)[S.TIMESTAMP].max().rename("last_bar")
    sc = pd.concat([total, kept, err, warn, gap_n, gap_m, first, last], axis=1).fillna(
        {"clean_records": 0, "error_records": 0, "warning_records": 0, "gaps": 0, "gap_minutes": 0}
    )
    for c in ["records", "clean_records", "error_records", "warning_records", "gaps", "gap_minutes"]:
        sc[c] = sc[c].astype(int)
    sc["validity_pct"] = (100 * (1 - sc["error_records"] / sc["records"])).round(2)
    # Completeness: bars present vs bars expected while the contract normally
    # trades (present + in-session minutes inside reported gaps). Scheduled
    # holiday closures are excluded, so they don't count against the feed.
    sc["completeness_pct"] = (100 * sc["records"] / (sc["records"] + sc["gap_minutes"])).round(1)
    # Score = record quality (errors hurt fully, warnings a quarter) x completeness.
    # The original score ignored completeness, so a contract with 135k missing
    # minutes still scored 100.0 - a trader would over-trust it.
    record_quality = (1 - (sc["error_records"] + 0.25 * sc["warning_records"]) / sc["records"]).clip(lower=0)
    sc["dq_score"] = (100 * record_quality * sc["completeness_pct"] / 100).round(1)
    gate = sc.apply(lambda r: _verdict(r, cfg), axis=1, result_type="expand")
    sc["verdict"], sc["verdict_reason"] = gate[0], gate[1]
    return sc.reset_index().rename(columns={"index": S.CONTRACT}).sort_values("dq_score")


def run_quality(
    df: pd.DataFrame,
    cfg: QualityConfig = None,
    rules: List[Rule] = None,
    rejected: pd.DataFrame = None,
) -> QualityReport:
    cfg = cfg or QualityConfig()
    rules = rules if rules is not None else DEFAULT_RULES
    df = df.reset_index(drop=True)
    if S.SOURCE not in df:
        df[S.SOURCE] = "<memory>"
    if S.ROW_ID not in df:
        df[S.ROW_ID] = np.arange(1, len(df) + 1)

    found = [r.check(df, cfg) for r in rules]
    found.append(rejected_to_issues(rejected))
    issues = _concat(found)

    # Build the clean set: drop ERROR records and exact duplicates.
    rec = issues[issues[S.ROW_ID].notna() & ~issues["rule_id"].isin(["DATA_GAP", "MALFORMED_RECORD"])]
    drop = rec[(rec["severity"] == ERROR) | (rec["rule_id"] == "DUPLICATE_EXACT")]
    drop_idx = pd.MultiIndex.from_arrays([drop[S.SOURCE].astype(str), drop[S.ROW_ID].astype("int64")])
    all_idx = pd.MultiIndex.from_arrays([df[S.SOURCE].astype(str), df[S.ROW_ID].astype("int64")])
    keep = ~all_idx.isin(drop_idx)
    clean = df.loc[keep].sort_values([S.CONTRACT, S.TIMESTAMP]).reset_index(drop=True)

    sev_order = {ERROR: 0, WARNING: 1, INFO: 2}
    if not issues.empty:
        issues = issues.assign(_s=issues["severity"].map(sev_order)).sort_values(
            ["_s", "rule_id", S.CONTRACT, S.TIMESTAMP], na_position="last"
        ).drop(columns="_s").reset_index(drop=True)
    return QualityReport(issues=issues, clean=clean,
                         scorecard=_scorecard(df, issues, clean, cfg), config=cfg)
