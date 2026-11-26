"""Intelligent insights: recurring data quality patterns and suggested
cleansing/validation rules.

The default provider is *rule-based*: deterministic, explainable, testable and
auditable, which matters for market data used by trading and risk. The
``InsightProvider`` interface lets an AI/LLM provider be plugged in later
(e.g. to write narrative summaries) without changing the rest of the system;
it would receive only aggregated statistics, never raw records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from . import schema as S
from .quality import ERROR, INFO, QualityReport

HIGH, MEDIUM, LOW = "High", "Medium", "Low"
_PRIORITY_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}


@dataclass
class Insight:
    kind: str  # "pattern" | "recommendation"
    priority: str
    title: str
    evidence: str
    recommendation: str = ""
    rule_id: str = ""


class InsightProvider:
    name = "base"

    def generate(self, report: QualityReport, data: pd.DataFrame) -> List[Insight]:
        raise NotImplementedError


# Suggested control for each rule: (proposed validation / cleansing rule, owner).
PLAYBOOK = {
    "MALFORMED_RECORD": (
        "Enforce a schema contract at the vendor boundary (ISO-8601 timestamps "
        "with timezone, numeric OHLCV). Quarantine bad rows and fail the whole "
        "file if more than 1% of rows are malformed.",
        "Data Engineering / vendor",
    ),
    "MISSING_VALUE": (
        "Reject bars with missing O/H/L/C (never forward-fill prices for "
        "trading or risk use); request a resend. Treat missing volume as 0 for "
        "VWAP but keep the flag.",
        "Market Data Ops",
    ),
    "DUPLICATE_EXACT": (
        "Make loads idempotent: deduplicate on (contract, timestamp) at "
        "ingestion so retries and replays cannot double-count volume.",
        "Data Engineering",
    ),
    "DUPLICATE_CONFLICT": (
        "Agree a golden-source precedence rule with the business: latest vendor "
        "correction wins, keep superseded versions in an audit table, alert if "
        "corrections exceed a daily tolerance.",
        "Market Data Ops + Business owner",
    ),
    "INVALID_PRICE": (
        "Hard validation price > 0, configurable per root (allow negative for "
        "products/spreads that can trade below zero, e.g. WTI in Apr-2020).",
        "Data Engineering",
    ),
    "OHLC_INCONSISTENT": (
        "Hard validation High >= max(Open, Close) and Low <= min(Open, Close); "
        "reject and reconcile against exchange settlement/time & sales.",
        "Market Data Ops",
    ),
    "INVALID_VOLUME": (
        "Validate volume is a non-negative integer; reject otherwise.",
        "Data Engineering",
    ),
    "ZERO_VOLUME_PRICE_MOVE": (
        "Check whether indicative/settlement prices are mixed into trade bars; "
        "tag bar type (trade vs indicative) at source.",
        "Market Data Ops / vendor",
    ),
    "TIMESTAMP_INVALID": (
        "Validate timestamps against the exchange trading calendar and carry an "
        "explicit timezone; reject future-dated records.",
        "Data Engineering",
    ),
    "DATA_GAP": (
        "Separate genuine no-trade minutes (illiquid contracts) from feed "
        "outages: use liquidity-aware gap thresholds, backfill outages from a "
        "secondary source and mark filled bars so users know.",
        "Market Data Ops",
    ),
    "PRICE_OUTLIER": (
        "Bad-tick filter: auto-reject when the move is extreme AND reverses on "
        "the next bar; otherwise keep but flag (real moves happen around "
        "economic releases and must not be removed).",
        "Market Data Ops + Trading desk",
    ),
}


class RuleBasedInsights(InsightProvider):
    name = "rule-based"

    def __init__(self, min_count: int = 3, hour_share: float = 0.3,
                 recurring_day_ratio: float = 0.5, min_recurring_days: int = 5,
                 min_for_time_pattern: int = 10):
        self.min_count = min_count
        self.hour_share = hour_share
        self.recurring_day_ratio = recurring_day_ratio
        self.min_recurring_days = min_recurring_days
        self.min_for_time_pattern = min_for_time_pattern

    # -------------------------------------------------------------- patterns
    def _dominant_issue(self, issues):
        out = []
        counts = issues["rule_id"].value_counts()
        if counts.empty:
            return out
        top, n = counts.index[0], int(counts.iloc[0])
        share = n / counts.sum()
        if share >= 0.4 and n >= self.min_count:
            out.append(Insight(
                "pattern", MEDIUM, f"{top} dominates the issue log",
                f"{n:,} of {counts.sum():,} issues ({share:.0%}) are {top}.",
                ("Most gaps on thin or deferred contracts are no-trade minutes, not one root cause: "
                 "use liquidity-aware thresholds so real outages stand out." if top == "DATA_GAP"
                 else "Fixing this one root cause would remove most of the noise."), top,
            ))
        return out

    def _contract_concentration(self, issues, data):
        out = []
        errs = issues[(issues["severity"] == ERROR) & issues[S.CONTRACT].notna()]
        if errs.empty or data[S.CONTRACT].nunique() < 2:
            return out
        err_share = errs[S.CONTRACT].value_counts(normalize=True)
        rec_share = data[S.CONTRACT].value_counts(normalize=True)
        for contract, es in err_share.items():
            rs = rec_share.get(contract, 0)
            n = int((errs[S.CONTRACT] == contract).sum())
            if n >= self.min_count and rs > 0 and es >= 2 * rs and es >= 0.25:
                out.append(Insight(
                    "pattern", HIGH, f"Errors concentrated in {contract}",
                    f"{contract} has {es:.0%} of all errors but only {rs:.0%} of records ({n} errors).",
                    f"Check the {contract} feed/mapping specifically rather than the whole pipeline.",
                ))
        return out

    def _time_of_day(self, issues):
        out = []
        timed = issues[issues[S.TIMESTAMP].notna()]
        for rule_id, g in timed.groupby("rule_id"):
            if len(g) < self.min_for_time_pattern:
                continue
            hours = pd.to_datetime(g[S.TIMESTAMP]).dt.hour.value_counts(normalize=True)
            h, share = int(hours.index[0]), float(hours.iloc[0])
            if share >= self.hour_share:
                out.append(Insight(
                    "pattern", MEDIUM, f"{rule_id} clusters around {h:02d}:00 CT",
                    f"{share:.0%} of {len(g):,} {rule_id} issues occur in the {h:02d}:00-{h:02d}:59 hour "
                    f"(uniform would be ~4%).",
                    "Time-specific cause likely (session open/close, vendor batch, "
                    "economic release). Add a time-aware control or calendar exception.",
                    rule_id,
                ))
        return out

    def _recurring(self, issues, data):
        out = []
        timed = issues[issues[S.TIMESTAMP].notna() & issues[S.CONTRACT].notna()].copy()
        if timed.empty:
            return out
        timed["d"] = pd.to_datetime(timed[S.TIMESTAMP]).dt.normalize()
        days = data.assign(d=data[S.TIMESTAMP].dt.normalize()).groupby(S.CONTRACT)["d"].nunique()
        hits = timed.groupby(["rule_id", S.CONTRACT])["d"].nunique()
        for (rule_id, contract), n_days in hits.items():
            total = days.get(contract, 0)
            if n_days >= self.min_recurring_days and n_days / total >= self.recurring_day_ratio:
                out.append(Insight(
                    "pattern", HIGH if rule_id in ("OHLC_INCONSISTENT", "DUPLICATE_CONFLICT", "MISSING_VALUE") else MEDIUM,
                    f"Recurring {rule_id} on {contract}",
                    f"Seen on {n_days} of {total} trading days ({n_days / total:.0%}) - systemic, not a one-off.",
                    "Raise with the data vendor / upstream owner as a recurring defect.",
                    rule_id,
                ))
        return out

    def _simultaneous_gaps(self, issues, data):
        gaps = issues[issues["rule_id"] == "DATA_GAP"]
        if gaps.empty or data[S.CONTRACT].nunique() < 2:
            return []
        g = gaps.assign(bucket=pd.to_datetime(gaps[S.TIMESTAMP]).dt.floor("15min"))
        per = g.groupby("bucket")[S.CONTRACT].nunique()
        shared = per[per >= 2]
        if shared.empty:
            return []
        example = shared.sort_values(ascending=False).index[0]
        return [Insight(
            "pattern", HIGH, "Gaps occur simultaneously across contracts",
            f"{len(shared)} time window(s) where 2+ contracts stop printing together "
            f"(e.g. {example:%Y-%m-%d %H:%M}, {int(shared[example])} contracts).",
            "Simultaneous gaps point to a feed/infrastructure outage or exchange "
            "halt, not illiquidity. Check vendor incident logs and backfill.",
            "DATA_GAP",
        )]

    def _illiquidity(self, report, data):
        out = []
        sc = report.scorecard
        if sc.empty or "gaps" not in sc:
            return out
        days = data.assign(d=data[S.TIMESTAMP].dt.normalize()).groupby(S.CONTRACT)["d"].nunique()
        avg_vol = data.groupby(S.CONTRACT)[S.VOLUME].mean()
        for _, row in sc.iterrows():
            c = row[S.CONTRACT]
            d = max(int(days.get(c, 1)), 1)
            if row["gaps"] / d >= 2 and avg_vol.get(c, np.inf) < 50:
                out.append(Insight(
                    "pattern", LOW, f"{c} gaps look liquidity-driven",
                    f"{int(row['gaps'])} gaps over {d} days with average volume "
                    f"{avg_vol.get(c, 0):.1f} per bar - consistent with a thinly traded "
                    "(e.g. deferred) contract rather than missing data.",
                    "Use a liquidity-aware gap threshold for deferred months so "
                    "real outages are not lost in noise. Don't raise these with "
                    "the vendor as a recurring defect.",
                    "DATA_GAP",
                ))
        return out

    def _lifecycle(self, issues, data, recent_days: int = 60):
        """Deferred contracts trade thinly until they approach front month."""
        gaps = issues[issues["rule_id"] == "DATA_GAP"]
        out = []
        if gaps.empty:
            return out
        days = data.assign(d=data[S.TIMESTAMP].dt.normalize()).groupby(S.CONTRACT)["d"].agg(["min", "max", "nunique"])
        for c, row in days.iterrows():
            if row["nunique"] < recent_days * 2:
                continue
            cutoff = row["max"] - pd.Timedelta(days=recent_days)
            g = pd.to_datetime(gaps.loc[gaps[S.CONTRACT] == c, S.TIMESTAMP])
            d = data.loc[data[S.CONTRACT] == c, S.TIMESTAMP].dt.normalize()
            early_days = d[d <= cutoff].nunique()
            late_days = d[d > cutoff].nunique()
            if not early_days or not late_days:
                continue
            early = (g <= cutoff).sum() / early_days
            late = (g > cutoff).sum() / late_days
            if early >= 3 * max(late, 0.05) and early >= 1:
                out.append(Insight(
                    "pattern", LOW, f"{c}: gaps fall as the contract nears expiry",
                    f"{early:.1f} gaps/day early in its life vs {late:.1f} gaps/day in the last {recent_days} days.",
                    "Gaps on deferred months are mostly no-trade periods. Report DQ "
                    "for front/active months separately so real issues stand out.",
                    "DATA_GAP",
                ))
        return out

    def _bad_ticks(self, issues):
        o = issues[issues["rule_id"] == "PRICE_OUTLIER"]
        if o.empty:
            return []
        rev = o["detail"].astype(str).str.contains("likely bad tick").sum()
        if rev == 0:
            return []
        return [Insight(
            "pattern", HIGH, "Spike-and-revert ticks detected",
            f"{rev} of {len(o)} price outliers reverse on the very next bar.",
            "These are almost certainly erroneous prints; promote the bad-tick "
            "filter to auto-reject for this pattern.",
            "PRICE_OUTLIER",
        )]

    # ------------------------------------------------------- recommendations
    def _recommendations(self, report: QualityReport, issues: pd.DataFrame = None) -> List[Insight]:
        out = []
        issues = report.issues if issues is None else issues
        if issues.empty:
            return [Insight(
                "recommendation", LOW, "No issues found",
                "All checks passed on the selected data.",
                "Keep the current controls; consider tightening thresholds "
                "(e.g. lower gap threshold) to increase sensitivity.",
            )]
        stats = issues.groupby("rule_id").agg(
            n=("rule_id", "size"),
            errors=("severity", lambda s: int((s == ERROR).sum())),
            contracts=(S.CONTRACT, lambda s: s.dropna().nunique()),
        )
        for rule_id, row in stats.iterrows():
            text, owner = PLAYBOOK.get(rule_id, ("Review and define a control.", "Data owner"))
            prio = HIGH if row["errors"] > 0 else (MEDIUM if row["n"] >= 10 else LOW)
            scope = f" across {row['contracts']} contract(s)" if row["contracts"] else ""
            out.append(Insight(
                "recommendation", prio, f"Control for {rule_id}",
                f"{row['n']:,} issue(s){scope}; {row['errors']:,} are errors excluded from analytics.",
                f"{text} Suggested owner: {owner}.",
                rule_id,
            ))
        return out

    def generate(self, report: QualityReport, data: pd.DataFrame) -> List[Insight]:
        # INFO items (e.g. scheduled holiday closures) are context, not defects.
        issues = report.issues
        if not issues.empty:
            issues = issues[issues["severity"] != INFO]
        patterns: List[Insight] = []
        illiquid = self._illiquidity(report, data)
        lifecycle = self._lifecycle(issues, data) if not issues.empty else []
        # Contracts whose gaps are already explained by thin trading or by
        # being a deferred month: not a vendor defect.
        illiquid_contracts = ({i.title.split(" ")[0] for i in illiquid}
                              | {i.title.split(":")[0] for i in lifecycle})
        if not issues.empty:
            patterns += self._dominant_issue(issues)
            patterns += self._contract_concentration(issues, data)
            # Don't tell Ops to chase the vendor for gaps that thin trading
            # already explains (the two insights used to contradict each other).
            patterns += [
                i for i in self._recurring(issues, data)
                if not (i.rule_id == "DATA_GAP" and i.title.split(" on ")[-1] in illiquid_contracts)
            ]
            patterns += self._time_of_day(issues)
            patterns += self._simultaneous_gaps(issues, data)
            patterns += self._bad_ticks(issues)
            patterns += lifecycle
        patterns += illiquid
        result = patterns + self._recommendations(report, issues)
        return sorted(result, key=lambda i: (i.kind != "pattern", _PRIORITY_ORDER[i.priority]))


def insights_frame(insights: List[Insight]) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in insights],
                        columns=["kind", "priority", "title", "evidence", "recommendation", "rule_id"])


def generate_insights(report: QualityReport, data: pd.DataFrame,
                      provider: Optional[InsightProvider] = None) -> List[Insight]:
    return (provider or RuleBasedInsights()).generate(report, data)
