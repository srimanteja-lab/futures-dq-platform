# What changed after review (v2)

Each change starts from a weakness found by checking the v1 output against
real data or the business question, and each comes with a test.
Tests: 49 → **62**, all passing.

## 1. Fixes to mistakes found in the AI-built logic

| Found in v1 | Why it mattered | v2 fix | Test |
|---|---|---|---|
| Thanksgiving, Black Friday and Christmas Eve early closes ranked as HIGH "simultaneous feed outages" | Ops would chase the vendor for a scheduled closure | `exchange_calendar.py`: US exchange holidays and early closes, generated per year. Short gaps (≤ 4 days) overlapping one become `SCHEDULED_CLOSURE` (INFO), excluded from scores and insights. 355 real gaps relabelled. | `test_black_friday_early_close_is_scheduled_not_a_data_gap`, `test_same_gap_on_an_ordinary_day_is_still_a_data_gap` |
| Every real contract scored 100.0, e.g. ESZ25 with 135k in-session minutes missing | A trader would over-trust a patchy series | `completeness_pct` added. Score = record quality × completeness. Real ESZ25 now scores 46.2. | `test_score_reflects_completeness_and_verdict_explains_itself` |
| The same contract was called a "recurring vendor defect" **and** "liquidity-driven" | Contradictory advice to a vendor | Recurring-gap insight suppressed when thin trading or the contract lifecycle explains it. "One root cause" advice corrected for gaps. | `test_liquidity_explained_gaps_are_not_sent_to_the_vendor`, `test_dominant_gap_advice_does_not_claim_a_single_root_cause` |

## 2. Reconciliation (`recon.py`, new **Reconciliation** tab)

* **Did the files arrive intact?** SHA-256, row count and total volume checked
  against the vendor's published manifest (`data/manifest/files.csv`). All 16
  files verify.
* **Do our daily bars agree with the vendor's?** Key = contract + trading date.
  Missing days, open match, high/low range containment, volume coverage, and
  close vs settlement (a definition difference, shown for information).
  Result: on the CME **session** date, opens match the vendor 97.6–100% on all
  8 contracts. On calendar dates, 7 of 8 drop to 14.5–33.4%, with up to 200
  range breaks. Sugar matches either way because it trades a day session only.
  Real data proves the trading-date decision.
* **Is a downstream copy the same as the source?** Count, duplicate keys, full
  outer join on contract + timestamp, field-by-field values, volume control
  totals by contract-day. `scripts/make_recon_copy.py` builds a demo copy with
  one row dropped, one added and one volume changed: **count passes, four
  checks break.**

Tests: `tests/test_recon.py`.

## 3. Controls (preventive and detective)

| Control | Type | Where |
|---|---|---|
| File rejected whole above 1% unreadable rows (rows go to quarantine, accounting still balances) | Preventive | `ingestion.ingest(max_malformed_pct=…)` |
| Publication gate: **Trusted / Use with caution / Blocked**, with the reason | Preventive | `quality._verdict` (thresholds in `QualityConfig`) |
| Manifest check, vendor daily reconciliation, copy reconciliation | Detective | `recon.py`, CLI flags, Reconciliation tab |

Test: `test_file_over_malformed_threshold_is_rejected_whole_and_accounted`,
`test_verdict_trusted_and_blocked`.

## 4. A more intuitive UI

* Opens on the question **"Can I trust this data?"** with a verdict card per
  contract and the record accounting in one line.
* Plain-English issue names ("Impossible OHLC") next to the rule ids.
* Sidebar in order: ① choose data → ② filter → ③ business rules (with *Reset*).
* **Impact preview:** change any business rule and every headline number shows
  its change vs the default rules.
* **Viewing as** Trader / Ops / Data owner: raw quarantined rows are masked
  unless you are the data owner.
* Intraday chart lists the issues affecting the selected day.
* Plotly deprecation banner removed (`use_container_width=True`).

## 5. CLI

`--manifest`, `--vendor-daily`, `--reconcile-against`, plus `file_checks.csv`
and a verdict per contract in the console output.
