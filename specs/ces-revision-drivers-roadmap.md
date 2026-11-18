# ces-revision-drivers — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

Derived 2026-11-15 by derive-roadmap's New-Project Entry. It resets the parked `ces-revisions` roadmap so that each driver gets a usable answer on built data first. The [reset review](findings/ces-revision-drivers-reset-review.md) records what was built and why the old order was slow. The owner approved this partition on 2026-11-15.

## Outcomes

Source: the owner's words, 2026-11-15: "First, measure how the number of days between the reference date and the release date affects revisions", with "Accept defaults" for the reset questions and "Keep all, list order" for the ranking. Background: [the old spec](completed/ces-revisions.md), [the research brief](completed/ces-revisions-prompt.md), the research drafts ([chatgpt](completed/ces-revisions-research-chatgpt.md), [claude](completed/ces-revisions-research-claude.md), [gemini](completed/ces-revisions-research-gemini.md)), [the old roadmap](completed/ces-revisions-roadmap.md), parked 2026-11-15, and [the reset review](findings/ces-revision-drivers-reset-review.md).

Choices made:

- Stack: Python 3.14, the uv 0.13 series, float64 JAX with NumPyro, and Polars (CLAUDE.md).
- GPUs: rented by hand from Lambda. Only the owner rents, pays for and terminates them.
- Built inputs (owner, 2026-11-15): the Kalman engine (plan 1), the review document and its inventory (plan 2), the vintage panel, release index and accounting decomposition in `data/panel/` (plan 5), the annual tables in `data/annual/panel/` (plan 6), and the operators in `data/operators/` (plan 8). Each rebuilds offline with its `build` command in CLAUDE.md.
- Plan 11 (owner, 2026-11-15): it stays on branch `claude/ces-revisions-stage-6-bd9ab7` until Stage 2 finishes it.
- Plan IDs (2026-11-15): the next new plan takes ID 12. Plans 9 and 10 were on the abandoned `codex/plan-10-cloud-gpu-runner` branch, which this repository no longer holds, and plan 11 is on a local branch that was never pushed. writing-plans' `entry_check.py` reports 12 only where that branch exists. A clone without it, such as a cloud session or a Lambda host, reports 9.
- Day count `D_t` (owner, 2026-11-15): federal business days from the first business day after the 12th, up to but not including the actual release date. Holidays are those of 5 U.S.C. 6103 as observed each year, computed by rule. Lapse days count as business days, and the affected releases are flagged. Reference months 2025-09 and 2025-10 are dropped. `D_t` stands alone, with no window to the second print.
- Smallest standard for O1 (owner, 2026-11-15): total nonfarm, NSA and SA, 1979–2026. Mean absolute F→S and F→T revision for each `D_t` value, and one regression with year and calendar-month controls. March 2020 to December 2021 reported apart. One findings file with the table and no figure.
- Target for O1 (owner, 2026-11-15): one session, on the same day.
- Process for Stage 1 (owner, 2026-11-15): lighter. Test-driven development for the day count and one whole-branch Opus review stay. Brainstorming, code blocks in the plan, per-task reviewers and subagent dispatch are skipped.
- Finish for O5 (owner, 2026-11-15): without plan 11's GPU work, which moves to O6: the Lambda measurement of Task 13 and the GPU rule of Task 14.
- Track A after Stage 1 (owner, 2026-11-15, at the resume): O3 is promoted to Stage 9, and the Track A Ranked list keeps O2, then O4. Stage 1 found no clear `D_t` effect within 12 to 19 business days, and O2 would reuse that narrow range.
- Resume of 2026-11-17 (owner, "Approve all"): O4 records BLS's refusal and stays ranked, and the Stage 8 and Stage 9 entries cite the Stage 1 work that landed after plan 12's Handoff (commit a3a54a1 and PR 21).
- First closing (owner, 2026-11-17): first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release. No public BLS page states it. By this rule the window to first closing is `D_t` less 4 business days, or less 3 when a holiday falls in the release's final week.
- Stage 9 after the first-closing rule (owner, 2026-11-17, at the same resume): O4 joins Stage 9, because the rule computes its dates and Stage 9 needs the window to first closing for the C1 rate. Stage 9 covers reference months 2003–2026, the scale model's sample, which the owner chose on 2026-11-17 because of the probability-sample redesign. This supersedes the line above for O4.

| # | Outcome | Track | Rank | Cost (estimate; basis) |
|---|---|---|---|---|
| O1 | How mean absolute revision (F→S, F→T) changes with the length of the collection window, total nonfarm, 1979–2026 | A | 1 | about one session: every input is on disk, and the holiday rule is computed offline |
| O2 | The same for each of the 11 supersectors, 2003–2026 | A | 3 | small after O1: the data is on disk, 12 series × 280 months |
| O3 | Whether a longer window raises the first-closing collection rate, and how revision size moves with that rate, 2003–2026 (1981–2026 until the resume of 2026-11-17) | A | 2 | one LABSTAT fetch for C1–C3. First-published values need Internet Archive pages, which throttle bursts |
| O4 | The O1 answer with first-closing dates in place of the release-date proxy, 2003–2026 | A | 4 | small, in Stage 9: the owner's first-closing rule computes the dates. BLS declined on 2026-09-14 to share them (`specs/deferred_items.md`, retired 2026-11-16; Resolved at resume, 2026-11-17) |
| O5 | How much of each total-nonfarm revision is news and how much is noise, 1979–2026 | B | 1 | the fit is done on plan 11's branch. About one session to finish without GPUs, and up to \$40 more with them |
| O6 | The news/noise split for each supersector, with benchmark and seasonal revisions inside the model | B | 2 | about 64 h per fit on the Mac, above plan 11's 48 h cap. This is arithmetic from the measured Mac grid, not a measurement. GPUs are not yet measured |
| O7 | Each driver's effect on revision variance, estimated jointly in the model with the news/noise split | B | 3 | after O6. Only a real run can tell |
| O8 | Which model predicts revision size best in real time, and a predicted revision scale for each new release | B | 4 | about 280 refits at past release dates (old roadmap) |
| O9 | The share of SA revision variance that comes from changes in seasonal adjustment, by sector, transition and regime | C | 1 | small: the table exists (110 rows) and is not yet reported |
| O10 | The same share with seasonal factors reproduced from BLS's archived X-13 files, for the 29 releases with complete archives | C | 2 | only a real run can tell. The inputs are rounded, so reproduction may fail |
| O11 | How revision size moves with BLS funding and staffing, FY2009–2026 | D | 1 | a fiscal-year panel keyed by hand from budget documents. The review document names the sources |
| O12 | The same back to FY1979 | D | 2 | archived budget justifications keyed by hand, with definition breaks |
| O13 | How birth–death forecast error relates to benchmark revisions, by supersector | E | 1 | the data is on disk. It waits on the open question below |
| O14 | How revision size moves with sample coverage and its RSE, by supersector and year | F | 1 | the data is on disk, annual only. Coverage is a proxy based on active reports |
| O15 | A verified literature review by driver, with an annotated bibliography | G | 1 | the inventory half exists in `docs/ces-revisions-review.md` |
| O16 | A written report of the answers, with intervals and stated limits | G | 2 | depends on which stages tick |
| O17 | Full validation suites: posterior predictive checks and full-panel SBC for the supersector and joint models | B | 5 | SBC multiplies each fit's cost. The aggregate pilot's replicates took a median of 523 s each on the Mac, 100 replicates |
| O18 | Traceability: each funding and staffing value traced to its budget document and page | D | 3 | small once O11's table exists: one page reference for each keyed value |

Tracks: A — Collection window (research). B — Revision model (research). Split from A because the news/noise split stands without A's answer, and O7's joint estimate serves a different use from A's descriptive one. C — Seasonal adjustment, D — Institutional capacity, E — Birth–death, and F — Sample coverage (each research). Each is its own driver, and each answer stands without the others. G — Written finding (known progression: the review, then the report). Split because it is a document for readers, not an estimate. Track order (owner, 2026-11-15): A, B, C, D, E, F, G.

## Open questions

- Req 11 of the old spec dates the birth–death method changes differently from the documented `birth_death_forecast` row of `docs/ces-revisions-review.md`. The owner has not decided whether to amend Req 11. Stage 5 needs that decision before its plan.
- BLS's [response-rate page](https://www.bls.gov/osmr/response-rates/) gives "10–16 business days" of collection before the first preliminary release. The owner's first-closing rule gives 8 to 15 business days for the 278 releases of 2003–2026 that no lapse delayed, counted from the 13th through the closing day. The page does not say where its count starts. It is also open whether the rule holds for a release that a lapse delayed. Stage 9 needs both settled before its plan.

## Stages

The order follows the owner's track ranking. Each core stage is a cheap answer on data that is already built, so the core stages come first. This diverges from the old roadmap, which put the model before every driver answer. Stages 1–7 are independent of one another and may run in any order. Stage 8 reads the findings of the stages that have ticked. Stage 9 was promoted from the Track A Ranked list at the resume after Stage 1, and it is independent of Stages 2–7.

- [x] Stage 1: Collection window and revision size, total nonfarm\
      Objective: Measure how mean absolute F→S and F→T revisions of total nonfarm change with the day count `D_t`, 1979–2026, at the owner's smallest standard.\
      Outcomes: O1\
      Track: A\
      Consumes: Built inputs (plan 5): `data/panel/release_index.parquet` (572 releases, 1979-02-02 to 2026-09-04) and the `cesnaicsrev` rows of `data/panel/panel.parquet` (total nonfarm F, S and T same-release changes, 1979-01 to 2026-08, SA and NSA), rebuilt offline by `scripts/vintage_sources.py build`. Built inputs (plan 2): the `collection_business_days` and `federal_holiday` rows of `docs/ces-revisions-review.md`. The `D_t` and smallest-standard lines under Choices made.\
      Produces: `specs/ces-revision-drivers-stage-1-findings.md`, with the table, the regression and the measured cost. A `D_t` value for every release, which later stages may join.\
      Exit: the findings file holds the mean absolute F→S and F→T revision for each `D_t` value, NSA and SA, with March 2020 to December 2021 apart, and the `D_t` coefficient with its standard error from the regression with year and calendar-month controls (O1). It records the wall time of the real run and the sessions used.\
      Leaves out: full validation suites (SBC, prior and posterior predictive checks): O17 in the Track B Ranked list. Point-in-time awareness (`observable_at` for each value): O8 in the Track B Ranked list. Traceability: Stage 8. Manifests and recorded fixtures: none, because the stage adds no source. O3's fetch follows CLAUDE.md's rules for `data/raw/`. The GPU measurement: O6 in the Track B Ranked list.\
      ROUTING: writing-plans

> Stage 1: COMPLETE (2026-11-15) — implemented by plan 12 (specs/plans/completed/12-ces-revision-drivers-stage-1.md). Its Handoff section names what Stage 8 consumes.
> Next: resume the roadmap.

- [x] Stage 2: Finish the aggregate news/noise pilot\
      Objective: Finish plan 11 without its GPU work, so that the total-nonfarm news/noise split and the run harness reach `main`.\
      Outcomes: O5\
      Track: B\
      Consumes: Plan 11 on branch `claude/ces-revisions-stage-6-bd9ab7`, 22 commits ahead of `main` and 2 behind. Its Tasks 1–12 are committed, with the pilot fit that meets Req 18, the 100-replicate SBC, the prior predictive, the Mac timing grid and the seasonal-layout comparison under `docs/runs/stage-6/`. Its run directories, with the pilot's `inference_data.nc`, are gitignored in `data/runs/` of the main checkout. They moved there on 2026-11-15 from the `ces-revisions-stage-6-bd9ab7` worktree, which was then archived. A worktree that checks out the branch must copy them in. Its `Roadmap:` header still names the old roadmap's Stage 6. Its Task 14 applies the GPU rule first and stops as `unmeasured` without a priced GPU record, so this stage re-scopes Task 14 to the seasonal-state decision alone, recorded as a plan Deviation. Its Task 1 amends `specs/ces-revisions.md`, which this reset moved to `specs/completed/`, so the merge carries that amendment to the new path. Built inputs from plans 1, 5, 6 and 8.\
      Produces: plan 11 retired, with the pilot model (`ces_revisions.pilot`), the run harness (`ces_revisions.runs`) and the seasonal-state decision on `main`, that decision written from the seasonal-layout comparison and the Mac timing grid. `specs/ces-revision-drivers-stage-2-findings.md`, with the news shares and the measured cost.\
      Exit: the findings file reports the posterior news share of the S and T revisions (`share_s`, `share_t`) of total nonfarm, with intervals, from the pilot fit's run directory, or from a refit if that directory is gone (O5). The committed JSON evidence does not hold these shares. Plan 11's header names this roadmap's Stage 2. The plan 1 dispersed-starts item and the plan 5 repeated-EMPLOY item in `specs/deferred_items.md` are ticked. The file records the fit's sampling time and the sessions used.\
      Leaves out: the GPU measurement and the GPU choice (plan 11's Task 13, up to \$40, and the GPU rule and `docs/runs/stage-6/gpu-choice.json` of its Task 14): O6 in the Track B Ranked list. Full validation suites at supersector scale: O17 in the Track B Ranked list.\
      ROUTING: writing-plans

> Stage 2: COMPLETE (2026-11-15) — implemented by plan 13 (specs/plans/completed/13-ces-revision-drivers-stage-2.md) with plan 11 (specs/plans/completed/11-ces-revisions.md). Its Handoff section names what Stage 8 consumes.
> Next: resume the roadmap.

- [ ] Stage 3: Seasonal share of SA revisions\
      Objective: Report the share of SA revision variance that comes from changes in seasonal adjustment, from the accounting decomposition that is already built.\
      Outcomes: O9\
      Track: C\
      Consumes: Built input (plan 5): `data/panel/accounting_decomposition.parquet`, 110 rows for 12 sectors, the F→S, S→T and T→B transitions, and the regimes before 2003, 2003–2019, 2020–2022 and 2023 on, with its Req 9 identity tests.\
      Produces: `specs/ces-revision-drivers-stage-3-findings.md`, with the shares and the measured cost.\
      Exit: the findings file gives, for each sector, transition and regime in the table, the share of SA revision variance from the change in seasonal adjustment, with the identity gap shown (O9). It records the sessions used.\
      Leaves out: nothing. Its input already carries provenance (plan 5).\
      ROUTING: writing-plans

- [ ] Stage 4: Revision size and BLS funding and staffing, FY2009–2026\
      Objective: Measure how revision size moves with BLS's real budget authority and staffing over the fiscal years 2009–2026.\
      Outcomes: O11\
      Track: D\
      Consumes: Built input (plan 2): the institutional rows and the FTE ruling of `docs/ces-revisions-review.md`, which name each series, its source documents and its definition breaks. Built input (plan 5): the stage changes in `data/panel/`. Background: Req 3's institutional row and Req 13 of the old spec.\
      Produces: a fiscal-year table of funding and staffing, FY2009–2026, with each value's source document. `specs/ces-revision-drivers-stage-4-findings.md`, with the association and the measured cost.\
      Exit: the findings file reports the mean absolute F→S and F→T revision in each fiscal year beside each funding and staffing series, and their association with its uncertainty (O11). It records the sessions used and the documents keyed.\
      Leaves out: traceability to the document and page of each value: O18 in the Track D Ranked list. Point-in-time awareness: O8 in the Track B Ranked list.\
      ROUTING: brainstorming

- [ ] Stage 5: Birth–death forecast error and benchmark revisions\
      Objective: Measure how the error of the birth–death forecast relates to the benchmark revision, by supersector and benchmark year.\
      Outcomes: O13\
      Track: E\
      Consumes: Built inputs (plan 6): `data/annual/panel/birth_death.parquet` (7,074 rows, with forecast-versus-realized rows, zero for government) and `data/annual/panel/benchmarks.parquet`. The owner's answer to the open question on Req 11.\
      Produces: `specs/ces-revision-drivers-stage-5-findings.md`, with the association and the measured cost.\
      Exit: the findings file reports, for each supersector and benchmark year, the birth–death forecast error beside the benchmark revision, with the method-change dates the owner chose marked, and their association with its uncertainty (O13). It records the sessions used.\
      Leaves out: nothing. Its inputs already carry provenance and `observable_at` (plan 6).\
      ROUTING: brainstorming

- [ ] Stage 6: Revision size and sample coverage\
      Objective: Measure how revision size moves with sample coverage and its relative standard error, by supersector and year.\
      Outcomes: O14\
      Track: F\
      Consumes: Built inputs (plan 6): `data/annual/panel/sample_panel.parquet` (288 rows, Table 1's coverage proxy, RSE with definition-break flags, March 2012 missing). Built input (plan 5): `data/panel/stage_changes.parquet` (12 sectors, 2003-05 to 2026-08).\
      Produces: `specs/ces-revision-drivers-stage-6-findings.md`, with the association and the measured cost.\
      Exit: the findings file reports the mean absolute F→S and F→T revision for each supersector and year beside coverage and RSE, with the RSE breaks and the missing 2012 table marked, and their association with its uncertainty (O14). It records the sessions used.\
      Leaves out: nothing. Its inputs already carry provenance and `observable_at` (plans 5 and 6).\
      ROUTING: brainstorming

- [ ] Stage 7: Verified literature review\
      Objective: Complete the literature by driver, the gap table and the annotated bibliography in `docs/ces-revisions-review.md`, each claim checked against primary sources.\
      Outcomes: O15\
      Track: G\
      Consumes: Built input (plan 2): `docs/ces-revisions-review.md`, with its inventory sections, rulings and structural tests in `tests/test_review_document.py`. Background: Req 20 of the old spec, the research brief and the three research drafts.\
      Produces: the finished review document. `specs/ces-revision-drivers-stage-7-findings.md`, with the measured cost.\
      Exit: the literature-by-driver, gap-table and bibliography sections exist, every numeric claim, date and citation carries a primary-source citation or an unverified flag, and `uv run pytest tests/test_review_document.py` and `uv run ruff format --check` pass (O15). The findings file records the sessions used and the sources checked.\
      Leaves out: nothing.\
      ROUTING: writing-plans

- [ ] Stage 8: Report\
      Objective: Write the report of the answers that the ticked stages found, with intervals and stated limits.\
      Outcomes: O16\
      Track: G\
      Consumes: The findings file of each ticked stage, and of any stage promoted from a Ranked list. From Stage 7: the finished review document. From Stage 1: the findings file and the command that regenerates its tables (plan 12, Handoff). Those tables rebuild from the committed `data/raw/`, so they meet this stage's rule for generated numbers (Resolved at resume, 2026-11-15). After that Handoff, the Stage 1 findings gained a 95% interval for the change from 12 to 19 days and a fit with the 5-week control (commit a3a54a1), and the scale model's report, `docs/runs/stage-1-scale-model/report.md`, was added beside them (PR 21). Where that report's result lands is the owner's open decision in `specs/deferred_items.md`, § stage-1-scale-model (Resolved at resume, 2026-11-17). From Stage 2: the findings file and the command that regenerates its table from the committed JSON, with no run directory (plan 13, Handoff; Resolved at resume, 2026-11-17). From Stage 9: the findings file and `scripts/first_closing.py report`, which runs the archive check and then rewrites its nine tables from `data/panel/` after the vintage build, so only the prose around them is hand-written. That Handoff names the traps. The C1 item it deferred to § stage-1-scale-model of `specs/deferred_items.md` meets the same open decision on the scale model's report (plan 15, Handoff; Resolved at resume, 2026-11-18).\
      Produces: the report under `docs/`.\
      Exit: the report has a section for each outcome whose stage has ticked, with its limits and with the interval or uncertainty that its findings file reports. Where a findings file reports none, as Stage 3's descriptive shares may not, the section says the result is descriptive. Every number and table in it is generated by code from committed artifacts, which places traceability for the report here. `uv run ruff format --check` passes on it (O16).\
      Leaves out: nothing.\
      ROUTING: brainstorming

- [x] Stage 9: Collection window and the first-closing collection rate\
      Objective: Measure whether a longer window to first closing raises the first-closing collection rate, how revision size moves with that rate, and how the Stage 1 answer changes when the window ends at first closing, 2003–2026.\
      Outcomes: O3, O4\
      Track: A\
      Consumes: From Stage 1: the day count `D_t` and the lapse flag of each release in `data/panel/collection_days.parquet`, which the vintage build writes, and `absolute_revisions` in `ces_revisions.collection_window` (plan 12, Handoff; Resolved at resume, 2026-11-15). Built inputs (plan 5): `data/panel/release_index.parquet` and the `cesnaicsrev` rows of `data/panel/panel.parquet`. Added after plan 12's Handoff (Resolved at resume, 2026-11-17): `five_week_interval` (commit a3a54a1) and `signed_revisions` (PR 21) in `ces_revisions.collection_window`, and the Stage 1 scale model in `ces_revisions.scale_model` (PR 21). The first-closing rule and the 2003–2026 sample under Choices made (Resolved at resume, 2026-11-17). Background: Req 3's collection rows of the old spec, the parked Stage 12 item in `specs/deferred_items.md`, and the scale model's `docs/runs/stage-1-scale-model/report.md` and `specs/stage-1-scale-model/recommendation.md`.\
      Produces: the collection-rate sources in `data/raw/`, from one LABSTAT fetch and the first-published values on Internet Archive pages, each fetched and validated under CLAUDE.md's rules for `data/raw/`, with its manifest row. `specs/ces-revision-drivers-stage-9-findings.md`, with the associations and the measured cost.\
      Exit: the findings file reports how the first-closing collection rate moves with the window to first closing, and how the mean absolute F→S and F→T revisions move with that rate, each with its uncertainty (O3). It reports the Stage 1 regression with the window to first closing in place of `D_t`, with its uncertainty (O4). It records the measured cost of the real run and the sessions used.\
      Leaves out: point-in-time awareness: O8 in the Track B Ranked list. Traceability: Stage 8.\
      ROUTING: brainstorming

> Stage 9: COMPLETE (2026-11-18) — implemented by plan 15 (specs/plans/completed/15-ces-revision-drivers-stage-9.md). Its Handoff section names what Stage 8 consumes.
> Next: resume the roadmap.

## Ranked list — Track A

1. O2 — small after Stage 1. The data is on disk.

O4 left this list for Stage 9 at the resume of 2026-11-17.

## Ranked list — Track B

1. O6 — about 64 h per fit on the Mac (an estimate). It picks up the GPU measurement and the GPU choice: plan 11's Task 13 with its \$40 cap, and the GPU rule of its Task 14.
2. O7 — after O6. Only a real run can tell.
3. O8 — about 280 refits at past release dates. It picks up point-in-time awareness: `observable_at` for every input, the day count and the funding and staffing values included.
4. O17 — full validation suites, approved 2026-11-15. SBC multiplies each fit's cost.

## Ranked list — Track C

1. O10 — only a real run can tell.

## Ranked list — Track D

1. O12 — archived budget justifications keyed by hand.
2. O18 — traceability, approved 2026-11-15. Small once O11's table exists.

## Ranked list — Track E

None.

## Ranked list — Track F

None.

## Stage-spec stamp

Every stage spec's Rollout note carries this line, which writing-plans copies verbatim into the stage plan's header:

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage N — on plan completion, tick the stage and
> re-validate later stages against what shipped.

On completion the stamp goes under the stage's entry and is authoritative:

> Stage N: COMPLETE (YYYY-MM-DD) — implemented by plan <id> (path). Its Handoff section names what <stages> consume.
> Next: resume the roadmap.

## Completion

Retirement is gated behind a conformance audit of the accumulated system. Re-run derive-roadmap's gap rubric over the Outcomes list, with evidence per verdict: the implementing stage and plan, its findings file, Deviation notes and `specs/deferred_items.md` entries. Ranked-list items that the owner never promoted have parked. Unmet outcomes exit exactly two ways: a new stage, or a conscious deferral with a written why. On retirement, this roadmap and the reset review move to `specs/completed/`.
