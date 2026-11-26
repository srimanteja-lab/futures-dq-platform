"""Streamlit dashboard: "Can I trust this data, and how does its quality
affect my analytics?"

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import glob
import io
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mdq import recon  # noqa: E402
from mdq import schema as S  # noqa: E402
from mdq.analytics import CALENDAR, SESSION, daily_ohlcv, filter_data, market_summary, rolling_vwap, trading_date  # noqa: E402
from mdq.ingestion import DEFAULT_MAX_MALFORMED_PCT, SchemaError, ingest  # noqa: E402
from mdq.insights import generate_insights, insights_frame  # noqa: E402
from mdq.quality import BLOCKED, CAUTION, ERROR, INFO, TRUSTED, WARNING, QualityConfig, rule_catalogue, run_quality  # noqa: E402

DEMO_FILE = os.path.join(ROOT, "data", "sample", "dirty_futures_sample.csv")
DEMO_COPY = os.path.join(ROOT, "data", "sample", "dirty_futures_copy.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")
DAILY_DIR = os.path.join(ROOT, "data", "daily")
MANIFEST = os.path.join(ROOT, "data", "manifest", "files.csv")

# Plain-English names for rule ids (the ids stay visible for traceability).
FRIENDLY = {
    "MALFORMED_RECORD": "Unreadable row",
    "MISSING_VALUE": "Missing price / volume",
    "DUPLICATE_EXACT": "Exact duplicate",
    "DUPLICATE_CONFLICT": "Conflicting versions",
    "INVALID_PRICE": "Zero / negative price",
    "OHLC_INCONSISTENT": "Impossible OHLC",
    "INVALID_VOLUME": "Bad volume",
    "ZERO_VOLUME_PRICE_MOVE": "Price move, no volume",
    "TIMESTAMP_INVALID": "Bad timestamp",
    "DATA_GAP": "Feed gap",
    "PRICE_OUTLIER": "Price spike",
    "SCHEDULED_CLOSURE": "Holiday / early close",
}
VERDICT_ICON = {TRUSTED: "✅", CAUTION: "⚠️", BLOCKED: "⛔"}
SEV_COLOR = {ERROR: "#C0392B", WARNING: "#E67E22", INFO: "#5D8AA8"}
DEFAULTS = {
    "gap": 30, "z": 10.0, "min_ret": 0.003, "neg": False, "holidays": True,
    "gate_validity": 99.0, "gate_complete": 95.0, "max_malformed": DEFAULT_MAX_MALFORMED_PCT,
    "convention": SESSION,
}
ROLES = ["Trader / risk", "Market Data Ops", "Data owner"]

st.set_page_config(page_title="Can I trust this data? · Futures DQ", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.verdict {border-radius: 12px; padding: 14px 16px; margin-bottom: 8px; border: 1px solid #0001;}
.verdict h4 {margin: 0 0 4px 0; font-size: 1.05rem;}
.verdict p {margin: 0; font-size: 0.86rem; color: #333;}
.v-ok {background: #E8F4EC;} .v-warn {background: #FDF1E0;} .v-bad {background: #FBE3E0;}
.acct {font-size: 1.0rem; padding: 10px 14px; border-radius: 10px; background: #EEF3F8;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ caching
@st.cache_data(show_spinner="Reading files…")
def _ingest_paths(paths: tuple, max_malformed: float):
    r = ingest(list(paths), max_malformed_pct=max_malformed)
    return r.data, r.rejected, r.file_checks


@st.cache_data(show_spinner="Reading uploads…")
def _ingest_bytes(blobs: tuple, max_malformed: float):
    r = ingest([io.BytesIO(b) for _, b in blobs], [n for n, _ in blobs], max_malformed_pct=max_malformed)
    return r.data, r.rejected, r.file_checks


@st.cache_data(show_spinner="Running data quality checks…")
def _quality(data: pd.DataFrame, rejected: pd.DataFrame, cfg_items: tuple):
    rep = run_quality(data, QualityConfig(**dict(cfg_items)), rejected=rejected)
    ins = insights_frame(generate_insights(rep, data))
    return rep.issues, rep.clean, rep.scorecard, ins


@st.cache_data(show_spinner=False)
def _daily(clean: pd.DataFrame, convention: str):
    return daily_ohlcv(clean, convention)


@st.cache_data(show_spinner="Calculating 15-minute VWAP…")
def _vwap(clean_contract: pd.DataFrame):
    return rolling_vwap(clean_contract)


@st.cache_data(show_spinner="Checking files against the vendor manifest…")
def _manifest(paths: tuple):
    return recon.verify_manifest(list(paths), MANIFEST)


@st.cache_data(show_spinner=False)
def _vendor_daily(paths: tuple):
    return recon.load_vendor_daily(list(paths))


def _label(rule_id: str) -> str:
    return FRIENDLY.get(rule_id, rule_id)


def _mask(text: str) -> str:
    """Restricted view: keep the reason code, hide the raw record content."""
    head = str(text).split(":")[0]
    return f"{head}: •••• restricted (data owner role)"


def _reset_rules():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Futures data trust")
st.sidebar.caption("Can I trust this data, and how does its quality affect my analytics?")

st.sidebar.subheader("① Choose data")
source = st.sidebar.radio(
    "Data source",
    ["Demo file (known defects)", "Real vendor data (Parquet)", "Upload CSV / Parquet"],
    captions=["1 week of ES + CL with injected defects", "8 CME/CFE/ICE contracts, 1.4M bars",
              "Contract, Timestamp, Open, High, Low, Close, Volume"],
    label_visibility="collapsed",
)


ss = st.session_state
max_malformed = float(ss.max_malformed)
file_paths: tuple = ()
try:
    if source.startswith("Demo"):
        if not os.path.exists(DEMO_FILE):
            st.error("Demo file missing - run `python scripts/make_sample_data.py`.")
            st.stop()
        file_paths = (DEMO_FILE,)
        data, rejected, file_checks = _ingest_paths(file_paths, max_malformed)
    elif source.startswith("Real"):
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
        if not files:
            st.error("No Parquet files in data/raw - run `python scripts/download_data.py`.")
            st.stop()
        names = [os.path.basename(f) for f in files]
        chosen = st.sidebar.multiselect("Files", names, default=names[:2])
        if st.sidebar.checkbox("Use all files", value=False):
            chosen = names
        if not chosen:
            st.info("Choose at least one file in the sidebar.")
            st.stop()
        file_paths = tuple(os.path.join(RAW_DIR, n) for n in chosen)
        data, rejected, file_checks = _ingest_paths(file_paths, max_malformed)
    else:
        ups = st.sidebar.file_uploader("CSV or Parquet", type=["csv", "parquet"], accept_multiple_files=True)
        if not ups:
            st.info("Upload one or more files with Contract, Timestamp, Open, High, Low, Close, Volume columns.")
            st.stop()
        data, rejected, file_checks = _ingest_bytes(tuple((u.name, u.getvalue()) for u in ups), max_malformed)
except SchemaError as e:
    st.error(f"File rejected: {e}")
    st.stop()

if data.empty:
    st.error("No usable records: every file was rejected or unreadable. See the file checks below.")
    st.dataframe(file_checks, width="stretch", hide_index=True)
    st.stop()

st.sidebar.subheader("② Filter")
all_contracts = sorted(data[S.CONTRACT].unique())
contracts = st.sidebar.multiselect("Contracts", all_contracts, default=all_contracts)
dmin, dmax = data[S.TIMESTAMP].min().date(), data[S.TIMESTAMP].max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))

with st.sidebar.expander("③ Business rules (owned by the business)", expanded=False):
    st.caption("Tolerances are business decisions. Change one and every number shows its impact vs the defaults.")
    st.number_input("Feed gap: minutes of silence while trading", 5, 1440, step=5, key="gap")
    st.checkbox("Ignore exchange holidays & early closes", key="holidays")
    st.slider("Price spike sensitivity (robust z-score)", 4.0, 30.0, step=0.5, key="z")
    st.number_input("Price spike: minimum move", 0.0, 0.1, step=0.001, format="%.3f", key="min_ret")
    st.checkbox("Allow negative prices", key="neg", help="e.g. WTI crude, 20 Apr 2020; calendar spreads")
    st.number_input("Reject a whole file above this % of unreadable rows", 0.0, 100.0, step=0.5, key="max_malformed")
    st.number_input("Gate: block below validity %", 90.0, 100.0, step=0.1, key="gate_validity")
    st.number_input("Gate: caution below completeness %", 50.0, 100.0, step=1.0, key="gate_complete")
    st.radio("Trading date", [SESSION, CALENDAR], key="convention", horizontal=True,
             help="session = CME trade date (17:00 CT starts the next day). calendar = midnight to midnight.")
    st.button("Reset to defaults", on_click=_reset_rules)

role = st.sidebar.radio("Viewing as", ROLES, help="Raw rows are visible to the data owner role only.")

cfg = dict(gap_threshold_minutes=int(ss.gap), outlier_z=float(ss.z), outlier_min_abs_return=float(ss.min_ret),
           allow_negative_prices=bool(ss.neg), use_holiday_calendar=bool(ss.holidays),
           gate_min_validity=float(ss.gate_validity), gate_min_completeness=float(ss.gate_complete))
default_cfg = dict(gap_threshold_minutes=DEFAULTS["gap"], outlier_z=DEFAULTS["z"],
                   outlier_min_abs_return=DEFAULTS["min_ret"], allow_negative_prices=DEFAULTS["neg"],
                   use_holiday_calendar=DEFAULTS["holidays"], gate_min_validity=DEFAULTS["gate_validity"],
                   gate_min_completeness=DEFAULTS["gate_complete"])
convention = ss.convention
changed = [k for k in DEFAULTS if ss[k] != DEFAULTS[k]]

view = filter_data(data, contracts, start, end)
if view.empty:
    st.warning("No data for the selected filters.")
    st.stop()
issues, clean, scorecard, insights = _quality(view, rejected, tuple(sorted(cfg.items())))
daily = _daily(clean, convention)

# ------------------------------------------------------------------ header
st.title("Can I trust this data?")
st.caption(f"{len(contracts)} contract(s) · {start} → {end} · exchange time (America/Chicago) · "
           f"trading date: {convention} · viewing as {role}")

if changed:
    st.info(f"**Impact preview:** {len(changed)} business rule(s) differ from the defaults "
            f"({', '.join(changed)}). Numbers show the change vs the default rules.", icon="🧪")
    b_issues, b_clean, b_scorecard, _ = _quality(view, rejected, tuple(sorted(default_cfg.items())))
else:
    b_issues = b_clean = b_scorecard = None

st.subheader("Verdict by contract")
cards = scorecard.sort_values("contract")
cols = st.columns(min(4, len(cards)))
css = {TRUSTED: "v-ok", CAUTION: "v-warn", BLOCKED: "v-bad"}
for i, (_, r) in enumerate(cards.iterrows()):
    with cols[i % len(cols)]:
        st.markdown(
            f"<div class='verdict {css[r.verdict]}'><h4>{VERDICT_ICON[r.verdict]} {r.contract} · {r.verdict}</h4>"
            f"<p>{r.verdict_reason}</p>"
            f"<p>Validity <b>{r.validity_pct:.2f}%</b> · Completeness <b>{r.completeness_pct:.1f}%</b> · "
            f"Score <b>{r.dq_score:.1f}</b></p></div>", unsafe_allow_html=True)

received = len(view) + len(rejected)
excluded = len(view) - len(clean)
n_warn = int((issues["severity"] == WARNING).sum()) if not issues.empty else 0
balanced = received == len(clean) + excluded + len(rejected)
st.markdown(
    f"<div class='acct'>🧮 <b>{received:,}</b> received = <b>{len(clean):,}</b> clean + "
    f"<b>{excluded:,}</b> excluded + <b>{len(rejected):,}</b> quarantined "
    f"{'✓ nothing silently dropped' if balanced else '✗ DOES NOT BALANCE'}</div>", unsafe_allow_html=True)
st.write("")


def _delta(now, base):
    return None if base is None else f"{now - base:+,} vs default rules"


k = st.columns(5)
k[0].metric("Records received", f"{received:,}", help="Every row in the selected files and filters.")
k[1].metric("Usable for analytics", f"{len(clean) / max(received, 1):.2%}",
            _delta(len(clean), None if b_clean is None else len(b_clean)),
            help="Share of records that passed every ERROR rule. Only these feed bars and VWAP.")
k[2].metric("Excluded (errors)", f"{excluded:,}",
            _delta(excluded, None if b_clean is None else len(view) - len(b_clean)), delta_color="inverse",
            help="Records that would corrupt a bar or a VWAP: bad prices, impossible OHLC, duplicates.")
k[3].metric("Flagged for review", f"{n_warn:,}",
            _delta(n_warn, None if b_issues is None else int((b_issues["severity"] == WARNING).sum())),
            delta_color="inverse", help="Kept in analytics but marked: gaps, price spikes. Real moves aren't deleted.")
k[4].metric("Quarantined (unreadable)", f"{len(rejected):,}", help="Rows that could not be parsed, kept with a reason code.")

tab_trust, tab_mkt, tab_rec, tab_ins, tab_data = st.tabs(
    ["✅ Can I trust it?", "📊 Impact on analytics", "🔁 Reconciliation", "💡 Insights & actions", "🗂 Records & downloads"])

issues_v = issues.copy()
if not issues_v.empty:
    issues_v.insert(0, "issue", issues_v["rule_id"].map(_label))
    if role != "Data owner":
        m = issues_v["rule_id"] == "MALFORMED_RECORD"
        issues_v.loc[m, "detail"] = issues_v.loc[m, "detail"].map(_mask)

# ------------------------------------------------------------- trust tab
with tab_trust:
    st.markdown("**Scorecard.** *Validity* = records with no errors. *Completeness* = bars present while the "
                "contract normally trades (holidays excluded). *Score* = record quality × completeness.")
    show = scorecard[["contract", "verdict", "verdict_reason", "records", "clean_records", "error_records",
                      "warning_records", "gaps", "gap_minutes", "validity_pct", "completeness_pct", "dq_score",
                      "first_bar", "last_bar"]]
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "verdict": st.column_config.TextColumn("Verdict"),
        "verdict_reason": st.column_config.TextColumn("Why", width="large"),
        "dq_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "validity_pct": st.column_config.NumberColumn("Validity %", format="%.2f"),
        "completeness_pct": st.column_config.NumberColumn("Completeness %", format="%.1f"),
    })
    if issues.empty:
        st.success("No data quality issues found for the current selection.")
    else:
        c1, c2 = st.columns(2)
        counts = issues_v.groupby(["issue", "severity"]).size().reset_index(name="count")
        fig = go.Figure()
        for sev in [ERROR, WARNING, INFO]:
            s = counts[counts.severity == sev].sort_values("count")
            if not s.empty:
                fig.add_bar(y=s.issue, x=s["count"], name={ERROR: "Error: excluded", WARNING: "Warning: flagged",
                                                            INFO: "Info: context"}[sev],
                            orientation="h", marker_color=SEV_COLOR[sev])
        fig.update_layout(title="What went wrong", barmode="stack", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.15))
        c1.plotly_chart(fig, use_container_width=True)
        timed = issues_v[issues_v[S.TIMESTAMP].notna() & (issues_v.severity != INFO)].copy()
        if not timed.empty:
            timed["hour"] = pd.to_datetime(timed[S.TIMESTAMP]).dt.hour
            heat = timed.groupby(["issue", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
            fig2 = go.Figure(go.Heatmap(z=heat.values, x=[f"{h:02d}" for h in heat.columns], y=heat.index,
                                        colorscale="OrRd"))
            fig2.update_layout(title="When does it happen? (hour of day, CT)", height=380,
                               margin=dict(l=10, r=10, t=40, b=10))
            c2.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Issue log.** Every issue traces back to its source file and row.")
        f1, f2 = st.columns(2)
        sev_f = f1.multiselect("Severity", [s for s in [ERROR, WARNING, INFO] if s in set(issues.severity)],
                               default=[s for s in [ERROR, WARNING] if s in set(issues.severity)])
        rule_f = f2.multiselect("Issue type", sorted(issues_v.issue.unique()), default=sorted(issues_v.issue.unique()))
        shown = issues_v[issues_v.severity.isin(sev_f) & issues_v.issue.isin(rule_f)]
        st.dataframe(shown.drop(columns=[c for c in ["gap_minutes", "category"] if c in shown]),
                     width="stretch", hide_index=True, height=340)

    with st.expander("Rule catalogue: what each check does"):
        cat = rule_catalogue()
        cat.insert(0, "issue", cat.rule_id.map(_label))
        st.dataframe(cat, width="stretch", hide_index=True)

# --------------------------------------------------------- analytics tab
with tab_mkt:
    if clean.empty:
        st.warning("No clean records left after quality checks.")
    else:
        st.info(f"Bars and VWAP below are built **only from records that passed the rules**: "
                f"{excluded:,} record(s) were kept out. A red ✕ marks a bar with a data-quality issue; "
                f"hover it for the reason.", icon="🛡️")
        st.dataframe(market_summary(daily), width="stretch", hide_index=True)
        contract = st.selectbox("Contract", sorted(clean[S.CONTRACT].unique()))
        d = daily[daily.contract == contract]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=d.trading_date, open=d.open, high=d.high, low=d.low, close=d.close,
                                     name="Daily OHLC"), 1, 1)
        fig.add_trace(go.Scatter(x=d.trading_date, y=d.vwap, name="Daily VWAP", line=dict(width=1, dash="dot")), 1, 1)
        fig.add_trace(go.Bar(x=d.trading_date, y=d.volume, name="Volume", marker_color="#8da0cb"), 2, 1)
        fig.update_layout(title=f"{contract} daily bars ({convention} trading date)", height=480,
                          xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Intraday: close vs rolling 15-minute VWAP")
        cc = clean[clean[S.CONTRACT] == contract]
        v = _vwap(cc)
        days = sorted(d.trading_date.dt.date.unique())
        day = st.select_slider("Trading date", options=days, value=days[-1])
        vd = v[trading_date(v[S.TIMESTAMP], convention).dt.date == day]
        fig3 = go.Figure()
        fig3.add_scatter(x=vd[S.TIMESTAMP], y=vd[S.CLOSE], name="Close", line=dict(width=1))
        fig3.add_scatter(x=vd[S.TIMESTAMP], y=vd["vwap_15m"], name="VWAP 15m", line=dict(width=2))
        day_issues = pd.DataFrame()
        if not issues_v.empty and not vd.empty:
            iss = issues_v[(issues_v[S.CONTRACT] == contract) & issues_v[S.TIMESTAMP].notna()]
            iss = iss[pd.to_datetime(iss[S.TIMESTAMP]).between(vd[S.TIMESTAMP].min(), vd[S.TIMESTAMP].max())]
            day_issues = iss
            if not iss.empty:
                ref = vd.set_index(S.TIMESTAMP)[S.CLOSE]
                y = ref.reindex(pd.to_datetime(iss[S.TIMESTAMP]), method="nearest")
                fig3.add_scatter(x=pd.to_datetime(iss[S.TIMESTAMP]), y=y.values, mode="markers", name="DQ issue",
                                 marker=dict(color="red", size=10, symbol="x"),
                                 text=iss.issue + ": " + iss.detail.astype(str), hoverinfo="text")
        fig3.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        if not day_issues.empty:
            st.markdown(f"**Issues affecting {contract} on {day}**")
            st.dataframe(day_issues[["issue", "severity", S.TIMESTAMP, "detail"]], width="stretch", hide_index=True)
        c = st.columns(2)
        c[0].download_button("Download daily bars (CSV)", daily.to_csv(index=False), "daily_ohlcv.csv", "text/csv")
        c[1].download_button(f"Download {contract} 15m VWAP (CSV)", v.to_csv(index=False),
                             f"{contract}_vwap_15m.csv", "text/csv")

# ----------------------------------------------------- reconciliation tab
with tab_rec:
    st.markdown("A matching row count is **necessary, never sufficient**: one dropped row plus one added row "
                "keeps the count identical. Each check below answers a different question.")

    st.subheader("1 · Did the files arrive intact?")
    st.caption("Preventive control: a file with too many unreadable rows is rejected whole, before analytics.")
    st.dataframe(file_checks, width="stretch", hide_index=True)
    if file_paths and os.path.exists(MANIFEST):
        man = _manifest(file_paths)
        if (man.status == "Not in manifest").all():
            st.caption("This file isn't in the vendor manifest, so no checksum check is possible.")
        else:
            st.caption("Checked against the vendor's published manifest: SHA-256 checksum, row count and total volume.")
            st.dataframe(man, width="stretch", hide_index=True)

    st.subheader("2 · Do our daily bars agree with the vendor's daily bars?")
    vendor_files = tuple(sorted(glob.glob(os.path.join(DAILY_DIR, "*_daily.parquet"))))
    if source.startswith("Real") and vendor_files:
        vendor = _vendor_daily(vendor_files)
        rsum, rdet = recon.reconcile_daily(daily, vendor)
        if rsum.empty:
            st.info("No vendor daily bars for the selected contracts.")
        else:
            st.caption(f"Key = contract + trading date ({convention}). Close is compared with the vendor's "
                       "settlement price, a different definition, so it's shown for information only.")
            st.dataframe(rsum, width="stretch", hide_index=True, column_config={
                "open_match_pct": st.column_config.NumberColumn("Open matches %", format="%.1f"),
                "volume_coverage_pct": st.column_config.NumberColumn("Volume coverage %", format="%.1f",
                                                                     help="Our minute volume / vendor daily volume"),
                "days_missing_in_minute_data": st.column_config.NumberColumn("Vendor days missing from minute data"),
                "range_breaks": st.column_config.NumberColumn("Range breaks",
                                                              help="Our high/low outside the vendor's daily range"),
            })
            st.caption("Try switching *Trading date* to calendar in Business rules: open matches drop and range breaks "
                       "jump, which shows why the session convention is the right definition.")
            with st.expander("Breaks by day"):
                st.dataframe(rdet[rdet.status != recon.MATCHED], width="stretch", hide_index=True)
    else:
        st.caption("Available for the real vendor data, where the vendor also publishes daily bars.")

    st.subheader("3 · Is a downstream copy the same as the source?")
    st.caption("Key = contract + timestamp. Full outer join, field-by-field values and volume control totals.")
    tgt_up = st.file_uploader("Target copy to reconcile (optional)", type=["csv", "parquet"], key="recon_target")
    src_data = target = None
    if tgt_up is not None:
        src_data = data
        target = ingest([io.BytesIO(tgt_up.getvalue())], [tgt_up.name], max_malformed_pct=100).data
        st.caption(f"Source: the loaded data · Target: {tgt_up.name}")
    elif source.startswith("Demo") and os.path.exists(DEMO_COPY):
        src_data = data
        target = _ingest_paths((DEMO_COPY,), 100.0)[0]
        st.caption("Source: demo file · Target: `dirty_futures_copy.csv`, a downstream copy with three silent "
                   "differences (see `scripts/make_recon_copy.py`).")
    if target is not None:
        rr = recon.reconcile(src_data, target)
        chk = rr.checks.copy()
        chk.insert(0, "", chk["result"].map({recon.PASS: "✅", recon.BREAK: "❌"}))
        st.dataframe(chk, width="stretch", hide_index=True)
        if rr.passed:
            st.success("Source and target reconcile: same keys, same values, same totals.")
        else:
            st.error(f"{len(rr.breaks)} break(s). Each needs a reason, an owner and an age.")
            st.dataframe(rr.breaks, width="stretch", hide_index=True)
            if not rr.control_totals.empty:
                st.markdown("**Control totals that differ (volume by contract and day)**")
                st.dataframe(rr.control_totals, width="stretch", hide_index=True)
    else:
        st.caption("Upload a target copy to reconcile it against the loaded data.")

# ----------------------------------------------------------- insights tab
with tab_ins:
    pats = insights[insights.kind == "pattern"]
    recs = insights[insights.kind == "recommendation"]
    st.subheader("Recurring patterns")
    if pats.empty:
        st.info("No recurring patterns detected for the current selection.")
    icon = {"High": "🔴", "Medium": "🟠", "Low": "🔵"}
    for _, r in pats.iterrows():
        with st.container(border=True):
            st.markdown(f"**{icon[r.priority]} {r.title}**  \n{r.evidence}  \n_→ {r.recommendation}_")
    st.subheader("Suggested controls, with an owner")
    recs = recs.assign(issue=recs.rule_id.map(_label))
    st.dataframe(recs[["priority", "issue", "rule_id", "evidence", "recommendation"]], width="stretch", hide_index=True)
    st.caption("Insights are rule-based: deterministic and auditable. An AI provider can be plugged in through "
               "InsightProvider and would receive aggregated statistics only, never raw records.")

# --------------------------------------------------------------- data tab
with tab_data:
    st.subheader("Quarantined records (could not be read)")
    q = rejected.copy()
    if role != "Data owner" and not q.empty:
        q["raw_record"] = "•••• restricted (data owner role)"
        st.caption("Raw rows are hidden for your role. They can contain identifiers the feed shouldn't expose.")
    st.dataframe(q, width="stretch", hide_index=True)
    st.subheader("Clean data preview")
    st.dataframe(clean.head(1000), width="stretch", hide_index=True)
    c = st.columns(3)
    c[0].download_button("Issue log (CSV)", issues_v.to_csv(index=False), "dq_issues.csv", "text/csv")
    c[1].download_button("Scorecard (CSV)", scorecard.to_csv(index=False), "dq_scorecard.csv", "text/csv")
    c[2].download_button("Insights (CSV)", insights.to_csv(index=False), "insights.csv", "text/csv")
