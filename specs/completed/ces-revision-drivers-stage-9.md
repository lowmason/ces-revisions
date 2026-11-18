# Stage 9: collection window and the first-closing collection rate — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: writing-plans. This spec is one plan.

Status: COMPLETE (2026-11-18), implemented by plan 15 (`specs/plans/completed/15-ces-revision-drivers-stage-9.md`). Approved design, 2026-11-17.\
Source: Stage 9 of `specs/ces-revision-drivers-roadmap.md` (outcomes O3 and O4), brainstormed on 2026-11-17.\
Owner decisions: the ones in [Decisions](#decisions), all of 2026-11-17.

## Intent

Stage 1 found no clear effect of the collection window `D_t` on revision size within the 12 to 19 business days of BLS's normal schedule. Its variation is thin: after year, month and 5-week controls, `D_t` keeps a residual standard deviation of about 1.3 business days. The scale model's report (`docs/runs/stage-1-scale-model/report.md`, Limitations and Threats, item 3) names the first-closing collection rate as the sharper test of the mechanism.

Stage 9 tests that mechanism on reference months 2003–2026:

- **O3 (a).** Does a longer window to first closing raise C1, the first-closing collection rate (`CEU00000000C1`)?
- **O3 (b).** How does the mean absolute F→S and F→T revision move with C1?
- **O4.** How does the Stage 1 regression change when the window ends at first closing instead of at the release?

BLS's former collection-rate page states that the number of collection days explains much of the month-to-month variation in the first preliminary collection rate. So O3 (a) measures the size of a link that BLS describes. O3 (b) asks whether that rate carries through to revisions.

## Background

### The first-closing rule

First closing falls one week before the release (owner, 2026-11-17). No public BLS page states it. The roadmap's Choices made records it in these words. Cite it only as the owner's rule, in those words, in every committed text: this spec, the plan, the findings, code comments, commit messages and PR text.

Derived from the rule (not the owner's words): the closing day is the release date less 7 calendar days, and the window to first closing `W_t` is `D_t` less 4 business days, or less 3 when a holiday falls in the release's final week.

### The window to first closing, computed offline on 2026-11-17

`W_t` counts the federal business days from the 13th of the reference month through the closing day. `business_days(start, end)` in `src/ces_revisions/vintages/federal_calendar.py` excludes `end`, so pass the closing day plus one day as `end`. Passing the closing day itself gives 7 to 14 days, which is wrong.

For the 278 releases of reference months 2003-01 to 2026-08 that no lapse delayed:

- `W_t` runs from 8 to 15. One release has 8 (reference month 2026-06, released Thursday 2026-07-02). The counts are 26 at 9, 59 at 10, 36 at 11, 42 at 12, 52 at 13, 42 at 14 and 20 at 15.
- `D_t − W_t` is 4 in 232 releases and 3 in 46. The 46 fall only in reference months May (13), June (13), August (10) and December (10): Memorial Day, Independence Day, Labor Day and New Year's Day in the release's final week.
- In 6 releases the closing day is itself a federal holiday: the December data of 2004, 2009, 2010, 2015, 2020 and 2021 (December 31 or January 1 as observed). The business-day count is unaffected.

The four lapse-delayed releases in the sample have these windows by the rule:

| Reference month | Release | Closing day by the rule | `W_t` |
|---|---|---|---:|
| 2013-09 | 2013-10-22 | 2013-10-15, inside the lapse of October 1–16, 2013, when BLS suspended all survey operations | 22 |
| 2013-10 | 2013-11-08 | 2013-11-01 | 14 |
| 2025-11 | 2025-12-16 | 2025-12-09 | 18 |
| 2026-01 | 2026-02-11 | 2026-02-04, the day after the lapse ended | 16 |

September 2025 is also lapse-delayed (`W_t` = 42), but Stage 1 drops it, and Stage 9 does too.

BLS's [response-rate page](https://www.bls.gov/osmr/response-rates/) gives "10–16 business days" of collection before the first preliminary release. No start rule tried reproduces that range: from the 13th the window is 8–15, from the 12th counted 9–15, from the Monday of the reference week 13–18, and from the Sunday after the reference week 8–13. A start that shifts every release by the same number of days changes no slope, so the start is a labeling choice (Decision D3).

### C1 sources: what the recon fetch found on 2026-11-17

The owner approved a read-only fetch of three files into `/tmp`. Nothing was committed.

- **LABSTAT** `https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries`: 352,998,862 bytes, `Last-Modified` Tue, 10 Nov 2026 13:30:00 GMT. It holds C1 from 1981-01 to 2026-10. October 2025 has no value. September and November 2025 carry footnote code `E`, whose meaning was not checked.
- **Internet Archive copy of 2025-03-19** of `https://www.bls.gov/web/empsit/cesregrec.htm`: one table, "Registry Receipts, by Month and by Release", with three sections (first preliminary, second preliminary and third release), each by year from 1981 and by month. It holds C1 from 1981-01 to 2025-02. Every one of its 530 C1 values equals LABSTAT's, and so do its 526 C2 and 527 C3 values. Its note says these data were previously published only on that page, and that the page ends with the February 2025 data of March 7, 2025.
- **Internet Archive copy of 2007-02-16** of `http://www.bls.gov/web/cesregrec.htm`: no data. It says BLS plans to post collection rates from 1981 forward later that year.

So each copy of the page is a full snapshot of the history as BLS showed it on that day. One unknown remains: whether BLS changed any published value between 2007 and 2025. Decision D8 sets the check.

In 2003–2026, C1 has a median of 70.5 percent and runs from 47.4 to 86.1. Its yearly mean was 65.3 in 2003, 77.3 in 2013 and 60.4 in 2024. The lapse-delayed months have C1 of 77.9 (2013-09), 83.5 (2013-10), 73.8 (2025-11, footnote `E`) and 69.5 (2026-01).

## Goals

1. Compute `W_t` for every release, offline, by the owner's rule.
2. Commit the C1 sources to `data/raw/` under CLAUDE.md's rules for that tree: a LABSTAT extract and the Internet Archive copies of `cesregrec.htm`, each with its manifest row.
3. Check, before any fit, that BLS never changed a published C1 value (D8).
4. Report O3 (a), O3 (b) and O4 with their uncertainty in `specs/ces-revision-drivers-stage-9-findings.md`, from generated tables.
5. Record the measured cost of the real run and the sessions used.

## Non-goals

- 1981–2002 (D5). The sample is reference months 2003–2026 (roadmap Choices made).
- Any analysis of C2, C3 or `CEU05000000RR`. They are committed with C1 because the same fetch brings them (D9).
- C1 as a covariate in the Bayesian scale model (Approach C). The plan adds this as a deferred item in `specs/deferred_items.md`, under the `stage-1-scale-model` section, at plan completion.
- The calendar-predicted and residual split of C1 from the old spec's Req 3 (Approach B).
- Supersectors (O2). C1 is published for total nonfarm only.
- Point-in-time awareness (`observable_at`): O8. Traceability for the report: Stage 8.
- Any change to Stage 1's tables, to the scale model's evidence, to the owner's open decision at `specs/deferred_items.md` § stage-1-scale-model ("Where the scale model's result lands"), or to `docs/ces-revisions-review.md` (Stage 7 owns it).

## Anti-goals

- Any committed text that names or hints at a source for the first-closing rule other than the owner.
- A fetch that refreshes the existing sources in `data/raw/`. A new `cesnaicsrev.htm` or `cesvinall.zip` would move Stage 1's generated tables and the scale model's evidence.
- A fit run before this spec's rules are fixed. Every sample rule, band and contrast below is set in advance.
- Reading O3 (b) as a causal effect of C1. It is an association.

## Constraints

- **Network gate.** Every run of a network command needs the owner's explicit yes, each time (owner's brief, 2026-11-17). Stage 9 has two kinds of run: the LABSTAT fetch and the Internet Archive fetch.
- **`data/raw/` rules (CLAUDE.md).** Validate every payload. Complete the fetch in a temporary tree before promoting any file. Give each file a manifest row. Never hand-edit a fetched file.
- **Contact address.** Requests to bls.gov hosts send `BLS_CONTACT_EMAIL` from `.project.env` through `archive_inventory.open_url`, which every fetcher in `scripts/` uses. Never print or commit the address. A worktree passes the main checkout's `.project.env` by absolute path.
- **Internet Archive limits.** It throttles bursts. Pause between requests, as `scripts/annual_sources.py` does (`PAUSE_SECONDS`).
- **Untrusted downloads.** Parse fetched pages with plain parsing code, and run any one-off reading of a download with `python -I`.
- **Writing.** Markdown in `specs/` stays GitHub-renderable (CLAUDE.md): display math in ```` ```math ```` fences and inline math as `` $`…`$ ``.

## Approaches

Part (a) has one sensible form: C1 on `W_t` with Stage 1's controls. The choice was for part (b).

- **A. Two OLS fits in Stage 1's form (chosen, D10).** (a) C1 on `W_t`. (b) Absolute revision on C1, four fits. Both with year and month effects, the 5-week control and CR1 errors clustered by reference year. Each part also has a descriptive table. It reuses `ols_clustered` and the design code in `ces_revisions.collection_window`, and it reports mean absolute revision, the roadmap exit's words, so O1, O3 and O4 read on one scale. Its limit: (b) is an association, because C1 moves for reasons other than the window.
- **B. Calendar-predicted and residual C1.** Predict C1 from the calendar, then fit the revision on both parts. The predicted part's coefficient is an instrumental-variable ratio: O4's slope over (a)'s slope. It inherits the thin window variation that Stage 1 found. The residual part equals A's (b) with `W_t` added as a control. It also needs standard errors for a generated regressor. Rejected: it adds little to A and O4.
- **C. C1 in the Bayesian scale model.** `specs/stage-1-scale-model/recommendation.md` built the model so that C1 can enter its log scale. It reads spread with heavy tails. But it needs a prior check, SBC and four NUTS fits, it does not report the exit's mean absolute revision, and it runs into the owner's open decision on where the scale model's result lands. Deferred (Non-goals).

## Decisions

All by the owner on 2026-11-17.

- **D1 — The first-closing rule and how text names its source.** As in [The first-closing rule](#the-first-closing-rule).
- **D2 — Lapse-delayed releases.** The four reference months of the sample whose first release a lapse delayed (2013-09, 2013-10, 2025-11, 2026-01) stay out of every fit, for both `D_t` and `W_t`. The findings list them in a table of their own with `D_t`, `W_t` by the rule, the closing day, C1 and their revisions. With no lapse month in the fits, the fits have no lapse indicator. A revision whose later estimate comes from a lapse-delayed release (for example August 2013's F→S) stays in the fits, unflagged, as in Stage 1 (owner, plan 12, final review).
- **D3 — Window start.** `W_t` starts on the 13th, as `D_t` does, so the two differ only at the end. The findings state that BLS's "10–16 business days" is not reproduced, with the ranges of [the window section](#the-window-to-first-closing-computed-offline-on-2026-11-17).
- **D4 — O4's fit.** O4 repeats Stage 1's regression with the 5-week control. It is fit twice on one sample, once with `D_t` and once with `W_t`.
- **D5 — 1981–2002.** Left out, with no descriptive table.
- **D6 — Closing day on a holiday.** Keep the rule as stated. The findings list the 6 releases.
- **D7 — Stage 1 findings.** The Stage 9 plan rewrites the first Limits bullet of `specs/ces-revision-drivers-stage-1-findings.md` (line 100: "Outcome O4 replaces it with BLS's dates when they arrive"). The new text points to the Stage 9 findings and cites the rule as the owner's.
- **D8 — Check that C1 never changed.** Fetch the last Internet Archive copy of `cesregrec.htm` in each calendar year from 2008 to 2024, beside the copy of 2025-03-19, and compare each copy's values with LABSTAT. Each fetch run needs the owner's yes.
- **D9 — C2 and C3.** Committed with C1 in the LABSTAT extract and present in the archived pages. Only C1 is analyzed.
- **D10 — O3's approach.** Approach A.
- **Recon fetch.** The owner approved the read-only fetch of 2026-11-17 whose results are in [C1 sources](#c1-sources-what-the-recon-fetch-found-on-2026-11-17).

## Design

### Window to first closing

Add the closing day and `W_t` to `data/panel/collection_days.parquet`, written by the vintage build, beside `collection_days` (`D_t`) and `lapse_delayed`. Compute them for every release in the release index. Stage 9 uses 2003–2026 only.

- `first_closing_date` = release date less 7 calendar days.
- `first_closing_days` = `business_days(13th of the reference month, first_closing_date + 1 day)`.

Tests (test first): for every non-lapse release of 2003–2026, `D_t − W_t` is 4 or 3, and it is 3 exactly when a federal holiday falls in the release's final week (closing day plus one through release day less one). Pin the 8 at 2026-06, the 6 holiday closing days, and the lapse windows of the table above.

### C1 sources in `data/raw/`

My calls, which the builder may change if a constraint forces it:

- **LABSTAT extract.** Do not commit the 353 MB file. Commit `data/raw/bls/ce-collection-rates.tsv`, the header line and every line of `ce.data.0.AllCESSeries` whose `series_id` is `CEU00000000C1`, `CEU00000000C2`, `CEU00000000C3` or `CEU05000000RR`, byte for byte. Its manifest row gives the LABSTAT URL in `derived_from`, the SHA-256 of the full download in `derived_from_sha256`, and the full file's `Last-Modified`. `bls/cesvin00-comments.csv` already follows this pattern.
- **Footnotes.** Commit `ce.footnote` from the same directory (small), so the findings can state what footnote `E` means.
- **Archive copies.** List the copies of both page addresses (`www.bls.gov/web/cesregrec.htm` and `www.bls.gov/web/empsit/cesregrec.htm`) with one request to the Internet Archive's CDX index. Take the last copy with HTTP status 200 in each year 2008–2024. Fetch each in its original form (the `id_` address) and commit it as `data/raw/internet-archive/cesregrec/<14-digit timestamp>.htm`, with the copy of 2025-03-19 (timestamp `20250319165303`). A copy is valid only if its table parses into the three sections. If a year's last copy is not valid, take the next latest copy in that year. If no copy in a year is valid, record the year as missing. Commit each copy whole: they are public BLS pages, as the archived copies under `data/annual/raw/internet-archive/` are. Before any copy is promoted, read each staged copy's prose. If one describes when collection for the first preliminary estimate ends, stop and report it to the owner, because it bears on how D1's text reads (my call).
- **Fetch command.** A network command that adds only these files and their manifest rows. It never refetches the existing sources (Anti-goals). Whether it is a new subcommand of `scripts/vintage_sources.py` or its own script is the builder's choice. `scripts/vintage_sources.py manifest` must still rehash the whole tree offline. The existing `scripts/vintage_sources.py fetch` writes `manifest.csv` from its own rows only, so it must be changed to keep the Stage 9 rows when it next runs.

### C1 table

The vintage build (offline) parses the LABSTAT extract and the archived copies, and writes `data/panel/collection_rates.parquet`: one row per series, reference month and source (LABSTAT or the copy's timestamp), with the value and any footnote code. Tests build it from `data/raw/` with the other artifacts through `tests/vintage_data.py`.

### Check that C1 never changed (D8)

Compare every value that appears in both LABSTAT and an archive copy, for C1, C2 and C3, as printed to one decimal. Then:

- **No value differs:** LABSTAT's C1 is the first-published C1, and the fits use it. The findings report the copies compared and the number of values checked.
- **Any value differs:** stop before any fit, and report the differences to the owner. The design assumed values never change, so the owner decides how to proceed.

This rule is my call. It is set here so that it is fixed before the data are seen.

### Samples

- **Main period:** reference months 2003-01 to 2026-08 with a release in the release index. Exclude September and October 2025 (dropped, as in Stage 1), March 2020 to December 2021 (reported apart, as in Stage 1), and the four lapse-delayed months (D2).
- **Part (a):** every main-period month with both C1 and `W_t`.
- **Part (b) and O4:** every main-period month and transition with a published revision, from `absolute_revisions` in `ces_revisions.collection_window`.
- **March 2020 to December 2021:** one descriptive table, as Stage 1 reports this period apart. No fit.

### Part (a): C1 on the window

```math
C1_t = \alpha + \beta W_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

`F_t` is the 5-week dummy (`five_week_interval`), `μ` a reference-year effect and `ν` a calendar-month effect. Standard errors cluster by reference year with the CR1 factor (`ols_clustered`). The findings give β with its standard error, and a 95% interval for `7β`, the change in C1 from 8 to 15 days, with a t quantile on G − 1 degrees of freedom for G clusters. Seven days matches Stage 1's contrast from 12 to 19 days in `D_t`.

Descriptive table: the number of months and the mean C1 for each value of `W_t`.

### Part (b): revision size on C1

```math
|r_t| = \alpha + \theta\, C1_t + \delta F_t + \mu_{y(t)} + \nu_{m(t)} + \varepsilon_t
```

Fit four times: NSA and SA, F→S and F→T. Same errors and degrees of freedom. The findings give θ with its standard error, and a 95% interval for `10θ`, the change in mean absolute revision for 10 percentage points of C1.

Descriptive table: C1 less its reference-year mean, cut into quintiles over the part (a) sample. For each quintile: the number of months, the range of the deviation, and the mean absolute revision for each status and transition. Quintiles of the within-year deviation follow the fit, which uses only within-year variation.

### O4: the Stage 1 regression with the window to first closing

On the part (b) sample, fit Stage 1's regression with the 5-week control twice: once with `D_t` and once with `W_t`, with no lapse indicator (D2). `days_regression` always adds the lapse column, which would be all zeros here and make the design singular, so O4 needs its own design. The table gives, for each status and transition, n, the number of years, and for each window its coefficient, standard error and the 95% interval for a 7-day change. A note says that `W_t − D_t` is constant except in the 46 releases where a holiday falls in the final week, so the two slopes can differ only through those releases.

The `D_t` fit on 2003–2026 belongs to Stage 9 only. It does not replace the Stage 1 tables.

### Lapse-delayed releases

One table with the four months of D2: reference month, release date, closing day by the rule, `D_t`, `W_t`, C1 with any footnote, and the four absolute revisions.

### Findings file and report command

`specs/ces-revision-drivers-stage-9-findings.md` follows the Stage 1 findings' layout: Question, Answer, Data and window, Results, Limits, Measured cost. A report command (offline) writes every table into `<!-- BEGIN GENERATED … -->` blocks with `archive_inventory.replace_generated`, as `scripts/collection_window.py report` does. The prose sections are written by hand from the tables.

The Limits section states at least these:

- The rule is the owner's (in the words of D1). BLS's "10–16 business days" is not reproduced (D3).
- The 6 holiday closing days (D6), and the 46 releases with a holiday in the final week.
- The four lapse-delayed months are out of the fits (D2), and the rule puts the 2013-09 closing inside the 2013 lapse.
- O3 (b) is an association. C1 moves for reasons other than the window, and C1 covers total nonfarm only.
- The meaning of footnote `E`, from `ce.footnote`.
- Stage 1's limits that carry over: later prints from a lapse-delayed release (D2), the second release's window, benchmark effects in November's F→T and December's F→S, the year clusters, and rounding to thousands.

Measured cost: the wall time of the real run, measured as Stage 1 measured it, the network fetches timed apart, and the sessions used.

## Sequence

1. `W_t` in `collection_days.parquet`, test first.
2. The fetch command, then a run on the owner's yes, then the committed sources and manifest rows.
3. `collection_rates.parquet` in the vintage build, with parser tests on the committed files.
4. The C1 check of D8, before any fit.
5. The analysis functions for (a), (b) and O4, with tests on small synthetic frames.
6. The report command and the findings file.
7. The rewrite of the Stage 1 findings' first Limits bullet (D7).
8. The measured cost, the deferred item for Approach C, and the whole-branch review.

## Left to the builder

- Module names and where the Stage 9 analysis code lives, for example a new module beside `ces_revisions.collection_window`.
- Whether the fetch is a subcommand of `scripts/vintage_sources.py` or its own script, within the Constraints.
- The plan's process: how many reviewers, and whether to dispatch subagents. Stage 1's lighter process was the owner's choice for Stage 1 only.

## Rollout

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 9 — on plan completion, tick the stage and
> re-validate later stages against what shipped.
