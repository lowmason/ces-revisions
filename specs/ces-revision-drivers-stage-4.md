# Stage 4: revision size and BLS funding and staffing, FY2009–2026 — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: writing-plans. This spec is one plan.

Status: awaiting the owner's approval.\
Source: Stage 4 of `specs/ces-revision-drivers-roadmap.md` (outcome O11), brainstormed on 2026-11-18.\
Owner decisions: the ones in [Decisions](#decisions), all of 2026-11-18. Every other choice is marked "my call".

## Intent

O11 asks how revision size moves with BLS funding and staffing. The research brief (`specs/completed/ces-revisions-prompt.md`) warns that appropriations and staffing are slow-moving, trending and collinear. It asks for candor about what such data cannot identify.

So Stage 4 settles one question (D1). Over FY2009–FY2025, were BLS's leaner fiscal years also its years of larger revisions? A leaner year has less real budget authority or fewer actual FTE (full-time equivalent employment). The answer also states how large an association the data can rule out.

The answer is an interval in thousands of jobs, and it reads like Stage 1's answer. It does not settle whether the 2025 cuts made revisions larger. The stage also builds a keyed fiscal-year panel of funding and staffing that O7, O12 and O18 can extend.

## Background

### What the fits can use

Funding and staffing take one value for each fiscal year. Stage 1 and Stage 9 controlled for the reference year, but here a year effect would absorb the covariate. So the association rests only on variation between fiscal years.

Counted from the built `data/panel/` on 2026-11-18:

- **Main sample.** Reference months 2008-10 to 2025-09, less the months that Stage 1 drops or reports apart, and less the lapse-delayed months of D2. Each status and transition (NSA and SA, F→S and F→T) has 179 months in 16 fiscal years.
- **Short years.** FY2021 has no months, because Stage 1 reports March 2020 to December 2021 apart. FY2020 has 5 months (2019-10 to 2020-02), and FY2022 has 9 (2022-01 to 2022-09). FY2013, FY2014 and FY2025 have 11 each.
- **FY2026.** It keeps 2025-12 and 2026-02 to 2026-07: 7 F→S months and 6 F→T months. Its actual FTE will first appear in the FY2028 budget justification, which does not exist yet. The 1,773 FTE in the FY2027 justification is a plan.

### Sources, as the review documents them

The `### Institutional` rows and the FTE ruling of `docs/ces-revisions-review.md` (lines 86–101 and 657–673) name the sources:

- **Budget authority.** The Total row of each Congressional Budget Justification (CBJ), BLS volume, in its table "Summary Budget Authority and FTE by Activity". It is General Funds plus the Unemployment Trust Funds transfer, in nominal thousands of dollars.
- **Actual FTE.** Only the first column of that table reports actual FTE, as a note under the table says ("NOTE: FY 2025 reflects actual FTE."). The column is headed "Enacted" or "Revised Enacted". The other columns are plans. The FTE column of the appropriation history is not actual FTE. A year's actual FTE first appears in the CBJ two budget years later: FY2009's in the FY2011 CBJ, FY2025's in the FY2027 CBJ.
- **Transition brief.** The DOL BLS transition brief gives authorized and actual FTE for FY2009–FY2016 on slide 37.
- **OPM headcount.** The Office of Personnel Management (OPM) counts the people on the rolls on the last day of each month. That is a headcount, not FTE. Each monthly employment file holds over 2 million rows.
- **Deflator.** The GDP implicit price deflator (`GDPDEF`) is quarterly, and BEA revises it. ALFRED keeps its vintages.

On disk in `data/annual/raw/dol/`: the FY2024 and FY2027 CBJs only, each as a PDF with its `pdftotext -layout` text beside it.

### What the two CBJs on disk show

- **FY2024 CBJ.** Its first column is FY2022 Revised Enacted: 1,951 FTE and \$687,952 thousand. That total includes a Headquarters Relocation line of \$28,470 thousand, with 0 FTE, which does not fund survey work.
- **FY2027 CBJ.** Its first column is FY2025 Enacted: 2,019 FTE and \$703,952 thousand. Its FY2026 Enacted column gives 1,773 FTE and \$708,500 thousand.
- **Activity lines.** The FY2024 CBJ has Labor Force Statistics, Prices and Cost of Living, Compensation and Working Conditions, Productivity and Technology, Executive Direction and Staff Services, and Headquarters Relocation. In the FY2027 CBJ, Labor Force Statistics becomes Employment and Unemployment Statistics, and the trust-fund money moves to its own line, Labor Market Information, with 0 FTE.

## Goals

1. Fetch and commit the missing sources under the network gate: 15 CBJ BLS volumes, the transition brief and one ALFRED vintage of `GDPDEF`.
2. Key one fiscal-year table of funding and staffing, FY2009–FY2026, with each value's source document.
3. Check the keyed values against their sources, and check the FTE concept, before any fit.
4. Report the association and its uncertainty (D5), with the descriptive tables, in `specs/ces-revision-drivers-stage-4-findings.md`, from generated tables.
5. Record the sessions used, the documents keyed and the measured cost.

## Non-goals

- Whether the 2025 cuts made revisions larger (D1). FY2025 and FY2026 are shown, not tested.
- The four `shutdown_*` rows (D2) and the three continuing-resolution rows (D3). Both go to O7.
- Budget and FTE by activity (my call). They are not in the exit, and keying them brings the activity rename and the trust-fund split into this stage. They are left to O7 or O12.
- Page references for each value (O18), `observable_at` for each value (O8), and the years before FY2009 (O12).
- A fit with both covariates, with C1 or with a business-cycle control. The old spec's Req 13 rules out unrestricted joint slopes for budget and staffing.
- A Bayesian fit. The joint model of O7 covers it, on Lambda.
- CES workyears. The review's `ces_workyears` row has one year only (FY2015).
- Any change to `docs/ces-revisions-review.md` (Stage 7 owns it), to the Stage 1 or Stage 9 tables, or to any committed source in `data/raw/` or `data/annual/raw/`.

## Anti-goals

- Reading the association as causal.
- A fit run before this spec's rules are fixed, or a rule changed after the data are seen. Every sample rule, contrast and sensitivity below is set in advance.
- A fetch that refreshes committed sources. `scripts/annual_sources.py fetch` downloads every `available` row of `data/annual/raw/source-catalog.csv` again, so adding rows and running it would replace the committed CBJs and every other annual source.
- One FTE series that mixes concepts: a plan or an authorized figure in the actual series, or an OPM headcount in either.
- Committed text outside `specs/` that names this work "Stage 4" alone (see Constraints).

## Constraints

- **Network gate.** Every network run needs the owner's explicit yes, each time. Stage 4 has up to four runs: the CBJs and the transition brief from dol.gov, the `GDPDEF` vintage from ALFRED, the OPM recon request, and, if the OPM rule passes, the OPM fetch.
- **Rules for `data/annual/raw/` (CLAUDE.md).** Validate every payload. Complete the fetch in a temporary tree before promoting any file. Give each new file a manifest row and a catalog row. Store a `pdftotext -layout` text beside each PDF, as the committed CBJs have. Never hand-edit a fetched file.
- **Manifest stamps (my call).** `refresh_manifest` stamps every row of `data/annual/raw/manifest.csv` with the run's time, through `raw.manifest_frame`. Stage 4 adds rows to that manifest and edits a manual table, so change it to keep the `fetched_at` of each file whose SHA-256 is unchanged. The existing rows then keep their `sha256`, `bytes` and `fetched_at`.
- **Contact address.** Requests to bls.gov hosts send `BLS_CONTACT_EMAIL` through `archive_inventory.open_url`. No Stage 4 host is a bls.gov host (dol.gov, ALFRED, data.opm.gov), so every request gets the generic `User-Agent`. Never print or commit the address.
- **Untrusted downloads.** Parse fetched files with plain parsing code. Run any one-off reading of a download with `python -I`.
- **Naming.** Committed text outside `specs/` names this work by O11 or by its findings file, never as "Stage 4" alone. CLAUDE.md's "Stage 4" means `src/ces_revisions/annual/`, the parked roadmap's stage. This is plan 16's convention, not an owner ruling.
- **Revision input.** The roadmap's Consumes line names "the stage changes in `data/panel/`". Stage 4 uses the `cesnaicsrev` rows of `data/panel/panel.parquet` through `absolute_revisions` in `ces_revisions.collection_window`, as Stage 1 did. `stage_changes.parquet` is by sector and starts in 2003-05, so Stage 4 does not need it.
- **No interpolation.** The old spec's Req 3 never interpolates an annual series to monthly. Each month takes its fiscal year's value as a step.
- **No model run.** Stage 4 fits OLS only, so the roadmap's Where models run line sends nothing to Lambda.
- **Tests and style.** New tests are hermetic and unmarked, and the gate is `uv run pytest -m "not slow and not network"`. `uv run ruff check` and `uv run ruff format --check` pass. Markdown in `specs/` follows CLAUDE.md's GitHub-renderable rules.

## Approaches

Each approach fits real budget authority and actual FTE separately (Non-goals).

- **A. Monthly fit with a fiscal-year covariate (chosen, D5).** It regresses each month's absolute revision on its fiscal year's value. Stage 1's 5-week control and calendar-month effects take out within-year noise. Errors cluster by fiscal year. The answer reads in thousands of jobs per 10% of real budget or per 100 FTE, which is what "how large an association the data rule out" needs. It reuses `ols_clustered`. Its limit: with no trend, the slope absorbs any drift, and with a trend, it rests on a few years' deviations. So both are reported.
- **B. Rank association of the fiscal-year means.** Spearman's correlation of the 16 means with each series, with a permutation interval. It is robust to an outlying year, such as FY2009 in the recession. Rejected: it gives no units, so it cannot say how large an effect is ruled out.
- **C. Year-to-year changes.** The change in the fiscal-year mean regressed on the change in log real budget. It removes any shared trend. Rejected: about 14 noisy differences give a very wide interval.
- **Bayesian fit.** A model run on Lambda. Rejected for this stage: O7's joint model covers it.

## Decisions

All by the owner on 2026-11-18.

- **D1 — What O11 settles.** The association over FY2009–FY2025, with an interval that shows how large an association the data rule out. The main fits use FY2009–FY2025. FY2026 goes in the tables with its enacted budget, and its FTE is marked as a plan. The 2025 cuts stay an open question: FY2025 and FY2026 are shown descriptively, and the findings say why they cannot settle it. OPM headcount is a cross-check only, and only if a cheap download exists ([OPM rule](#opm-rule)). The keyed panel is built for reuse by O7, O12 and O18.
- **D2 — Lapses.** The four lapse-delayed reference months 2013-09, 2013-10, 2025-11 and 2026-01 leave every fiscal-year mean and fit, as in Stage 9 (D2 of its spec). They get a table of their own. The four `shutdown_*` rows are not keyed. They stay with O7, as the parked Stage 13 item in `specs/deferred_items.md` assigns them. This confirms the default that the handoff flagged.
- **D3 — Continuing-resolution rows.** `cr_days`, `cr_extensions` and `full_year_cr` are not keyed. They go to O7. This spec's commit records that in the parked Stage 13 item.
- **D4 — Deflator.** The GDP implicit price deflator, `GDPDEF`, from one ALFRED vintage. Each fiscal year averages its four quarters.
- **D5 — Approach.** Approach A.

## Design

### Fiscal year of a month (my call)

A reference month `m` belongs to fiscal year `year(m) + 1` when its month is October or later, and to `year(m)` otherwise. The mapping uses the reference month, not the release date. Collection for September's first print runs into October, the next fiscal year, and the findings state this.

### Sources to fetch

- **CBJ volumes.** `https://www.dol.gov/sites/dolgov/files/general/budget/<YYYY>/CBJ-<YYYY>-V3-01.pdf` for YYYY 2011–2023, 2025 and 2026, committed as `data/annual/raw/dol/cbj-<YYYY>-v3-01.pdf` with its text file, as the two on disk are. A payload is valid only if it is a PDF whose text names the Bureau of Labor Statistics and holds the summary table of budget authority and FTE by activity. If an address fails that check, stop and report to the owner. Do not search for another address without the owner.
- **Transition brief.** `https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf`, committed as `data/annual/raw/dol/bls-transition-brief.pdf` with its text file, in the same run as the CBJs.
- **Deflator.** One ALFRED vintage of `GDPDEF`: the latest vintage on the fetch date, committed under `data/annual/raw/alfred/` with the vintage date in its file name. Its catalog row records the vintage date.
- **Fetch command.** A network command that adds only these files, their catalog rows and their manifest rows. It never downloads a committed source again (Anti-goals). `scripts/annual_sources.py manifest` must still rehash the whole tree offline, with the stamp rule of Constraints.

### OPM rule

My call, within D1's "only if cheap".

1. One recon request, on the owner's yes: does data.opm.gov return BLS's on-board count for one month without the full employment file? BLS is the agency subelement BUREAU OF LABOR STATISTICS.
2. If it does, then on a second yes, key the count of September 30 for each fiscal year FY2009–FY2025 that the source covers. For FY2026, key the latest month, labeled with its date. OPM keeps several versions of a month, so take the latest version on the fetch date and record which it is. The series is `opm_headcount`, and its concept is headcount. It goes in the tables, never in a fit.
3. If it does not, leave OPM out. The findings say why.

### Keyed table

One CSV under `data/annual/raw/manual/`, rehashed by `scripts/annual_sources.py manifest`. It has one row per fiscal year and series, with the value, its unit, its concept, its source file, a source locator and a note. The locator names the table and column, for example "Summary Budget Authority and FTE by Activity, FY 2022 Revised Enacted". It never gives a page, which is O18's work. The note holds the text of any footnote, "comparable" heading or other basis marker that the summary table attaches to the value, and is empty when there is none (my call).

| Series | Fiscal years | Source | Concept |
|---|---|---|---|
| `budget_authority_total` | FY2009–FY2026 | Total row, Amount, first column of the CBJ two budget years later. FY2026: the FY2027 CBJ's FY2026 Enacted column | nominal \$ thousands |
| `budget_authority_general_funds`, `budget_authority_trust_funds` | FY2009–FY2026 | the two rows under Total | nominal \$ thousands |
| `budget_authority_one_time` | FY2009–FY2026 | the sum of the one-time lines in the same column, 0 when there are none | nominal \$ thousands |
| `fte_actual` | FY2009–FY2025 | Total row, FTE, first column of the CBJ two budget years later | actual FTE |
| `fte_plan` | FY2026 | the FY2027 CBJ's FY2026 Enacted column (1,773) | planned FTE |
| `opm_headcount` | as the OPM rule allows | OPM | headcount |

**Standing and one-time lines (my call).** The standing activity lines are Labor Force Statistics (named Employment and Unemployment Statistics in the FY2027 CBJ), Labor Market Information (the trust-fund line), Prices and Cost of Living, Compensation and Working Conditions, Productivity and Technology, and Executive Direction and Staff Services. Headquarters Relocation is a one-time line. If a first column holds any other line, stop and report it to the owner before any fit. Older CBJs may name an activity that the two on disk do not, and the owner decides whether it is standing.

### Checks before any fit

Each check is a test on the committed files. A failure stops the work, and the stop rules go to the owner.

- **Ties to the source text.** Each value keyed from a CBJ appears in the first column of its line in the summary table of that CBJ's text file: Total, General Funds, Unemployment Trust Funds and each one-time line.
- **Identity.** In each year, General Funds plus Unemployment Trust Funds equals the Total.
- **Actual FTE concept (my call).** The arithmetic checks catch typos. They cannot catch a first column that holds a plan or an authorized figure. So a year's `fte_actual` is accepted in one of two cases. Either the note under its summary table says that the first column reflects actual FTE, and, for FY2009–FY2016, the value equals the transition brief's actual FTE. Or the note is missing, the year is in FY2009–FY2016, and the value equals the brief's actual FTE. In every other case, stop and report to the owner.
- **Definition breaks (my call).** Each fiscal year's value comes from the first column of a different CBJ, so each sits on the basis of its own CBJ. The review records that CBJs restate earlier years, with footnotes for changes such as comparative transfers. If the summary table marks a first-column Total amount or FTE as restated or comparable, or a footnote records a transfer of a function or of FTE into or out of BLS, stop and report it to the owner. The owner decides how the series treats it before any fit. The fits never gain a break term, so the 16 fits of [Fit](#fit) stay fixed.
- **One-time money inside a standing line (my call).** The subtraction of [Real budget authority](#real-budget-authority) uses only one-time lines of the summary table. If a footnote of that table names one-time money inside a standing line for the first-column year, stop and report it to the owner. Keying reads the summary table and its footnotes only. It does not search the narrative for such money, and the findings say so.

### Real budget authority

- The deflator `d_f` for fiscal year `f` is the mean of `GDPDEF` over Q4 of calendar year `f − 1` and Q1 to Q3 of calendar year `f`, all from the committed vintage.
- The fitted series is `(budget_authority_total − budget_authority_one_time) × d_2025 / d_f`, in FY2025 dollars.
- **Deviation from the review (my call).** The review's `budget_authority` is the Total row. The fit uses the Total less the one-time lines, because relocation money does not fund survey work. Both are keyed, both are in the tables, and the findings state which one the fit uses.
- If the vintage lacks a quarter of FY2026 (Q3 2026), FY2026's real value is left blank and marked. FY2026 is descriptive only.
- The tables also show the nominal Total and the one-time amount.

### Samples

- **Revisions.** `absolute_revisions` (Stage 1): total nonfarm, NSA and SA, F→S and F→T, in thousands.
- **Main sample.** Reference months 2008-10 to 2025-09 with Stage 1's period `main`. That leaves out September and October 2025 (dropped) and March 2020 to December 2021 (reported apart). Also leave out 2013-09 and 2013-10 (D2). The result is 179 months for each status and transition, in 16 fiscal years.
- **FY2026.** 2025-12 and 2026-02 to 2026-07 (D2 removes 2025-11 and 2026-01). Tables only.
- **March 2020 to December 2021.** One table by fiscal year: FY2020 (7 months), FY2021 (12) and FY2022 (3). No fit.
- **Lapse-delayed months.** One table. No fit.

### Fit

```math
|r_t| = \alpha + \theta\, x_{f(t)} + \delta F_t + \nu_{m(t)} + \varepsilon_t
```

- `x` is either the natural log of real budget authority or actual FTE divided by 100, in separate fits. `F_t` is the 5-week dummy (`five_week_interval`), and `ν` is a calendar-month effect.
- Standard errors cluster by fiscal year with the CR1 factor (`ols_clustered`). With G = 16 clusters, 95% intervals use a t quantile on 15 degrees of freedom.
- **Contrasts.** For budget: 10% less real budget authority, $`\theta \ln 0.9`$. For staffing: 100 fewer actual FTE, $`-\theta`$. Both are in thousands of jobs of mean absolute revision, so a positive value means leaner years had larger revisions.
- **Trend sensitivity (D5).** The same fit with a linear trend in the fiscal year added. Both forms are reported.
- Four series × two covariates × two forms gives 16 fits. The table gives, for each: n, G, the contrast, its standard error and its 95% interval.
- **Leave one year out (my call).** With 16 clusters and a covariate that is constant within each cluster, one year can carry the slope. So each of the 16 fits is refit 16 times, once without each fiscal year (G = 15, t on 14 degrees of freedom). The table gives each fit's smallest and largest contrast across the 16 refits, and the year whose removal moves the contrast most. No other variant is fit.
- **What the Answer says is ruled out (my call).** For each series and covariate, the Answer gives the no-trend and the trend interval side by side. Its statement of how large an association the data rule out uses the span of the two 95% intervals together: from the lower of their two lower limits to the higher of their two upper limits. The leave-one-year-out table shows how fragile the result is. It does not bound the statement.

### Tables

All are generated.

1. **By fiscal year, FY2009–FY2026** (the exit's table). Each fiscal year's months in the sample and its mean absolute revision for the four series, beside nominal budget authority, the one-time amount, real budget authority, actual FTE (FY2026's plan, marked) and, if keyed, OPM headcount with its date. FY2021 shows no months.
2. **Sources.** Each fiscal year and series, with its value and its source document.
3. **Fits.** The 16 fits of [Fit](#fit).
4. **Leave one year out.**
5. **March 2020 to December 2021, by fiscal year.**
6. **Lapse-delayed months.** Reference month, fiscal year, release date and the four absolute revisions.

### Findings file and report command

`specs/ces-revision-drivers-stage-4-findings.md` follows the Stage 1 findings' layout: Question, Answer, Data and sources, Results, Limits, Measured cost. Data and sources lists every document keyed. A report command (offline) writes every table into `<!-- BEGIN GENERATED … -->` blocks with `archive_inventory.replace_generated`, as `scripts/collection_window.py report` does. The prose is written by hand from the tables.

The Limits section states at least these:

- The association is not causal. Funding and staffing trend together with other things. The yearly mean first-closing collection rate fell from 77.3 percent in 2013 to 60.4 in 2024 (Stage 9 spec). FY2009 falls in the 2007–2009 recession, when revisions run large.
- A covariate that is constant within each of 16 clusters can make CR1 understate the interval.
- Actual FTE counts all of BLS. CES is a small part of it: about 595 FTE in FY2015, federal, state and contract together (Robertson 2016, the review's `ces_workyears` row).
- FY2026 is a partial year, and its FTE is a plan. The 2025 cuts are not settled (D1).
- The fit subtracts the one-time lines, which the review's Total definition includes. One-time money that a CBJ folds into a standing line, with no footnote, stays in the fitted series.
- Each fiscal year's value sits on the basis of the CBJ it comes from, not on one common basis.
- `GDPDEF` measures general prices, not the cost of BLS's staff and contracts.
- Fiscal years follow the reference month, and September's collection runs into the next fiscal year.
- The lapse-delayed months are out of the fits (D2). The shutdown and continuing-resolution rows are with O7 (D2, D3).
- Stage 1's limits that carry over: later prints that come from a lapse-delayed release stay in the sample, benchmark effects in some months' F→T and F→S revisions, and rounding to thousands.

Measured cost: the wall time of the report run, measured as Stage 1 measured it, the network fetches timed apart, the keying time, and the sessions used.

## Sequence

1. The fetch command and the manifest-stamp change. Then the CBJ and brief run and the ALFRED run, each on the owner's yes. Then the committed sources, catalog rows and manifest rows.
2. The OPM recon on the owner's yes, then the [OPM rule](#opm-rule).
3. The keyed table, with the checks of [Checks before any fit](#checks-before-any-fit). Every stop rule is settled before step 5.
4. The fiscal-year mapping and the deflator average, test first.
5. The fit and table functions, with tests on small synthetic frames.
6. The report command and the findings file.
7. The measured cost and the whole-branch review.

## Left to the builder

- Module and script names, for example a new module beside `ces_revisions.collection_window`.
- Whether the fetch is a new subcommand of `scripts/annual_sources.py` or its own script, within Constraints.
- The keyed table's file name, its exact columns and the catalog's role names, within [Keyed table](#keyed-table).
- The ALFRED endpoint that delivers the chosen vintage.
- The plan's process: how many reviewers, and whether to dispatch subagents.

## Rollout

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 4 — on plan completion, tick the stage and
> re-validate later stages against what shipped.
