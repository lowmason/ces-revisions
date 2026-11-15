# Stage 1 — Collection Window and Revision Size, Total Nonfarm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via executing-plans (the owner chose inline execution, D2). Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 1 — on plan completion, tick the stage and re-validate later stages against what shipped.

Process: lighter. Keep TDD for the day count and one whole-branch Opus review. Skip brainstorming, code blocks in the plan, per-task reviewers and subagent dispatch.

**Goal:** Measure how the mean absolute F→S and F→T revisions of total nonfarm change with the collection-window day count `D_t`, 1979–2026, NSA and SA, and write the answer to `specs/ces-revision-drivers-stage-1-findings.md`.

**Architecture:** Two small modules in `ces_revisions.vintages` compute the federal holiday calendar by rule and one `D_t` per release. The vintage build writes the day counts as `data/panel/collection_days.parquet`. A new module, `ces_revisions.collection_window`, joins `D_t` to the revision table's estimates in `data/panel/panel.parquet`. It computes the table of mean absolute revisions and one OLS specification with year and calendar-month fixed effects and year-clustered standard errors. `scripts/collection_window.py report` writes the tables into generated blocks of the findings file.

**Tech Stack:** Python 3.14, Polars, NumPy (OLS and clustered standard errors by hand), pytest, ruff. No new dependency.

**Spec:** none. The plan's source is the Stage 1 entry of the roadmap (`specs/ces-revision-drivers-roadmap.md:57-65`) and its Stage 1 choices (`:17`, `:19-23`). Executors read both.

## Global Constraints

- **`D_t` (roadmap `:20`, verbatim):** "federal business days from the first business day after the 12th, up to but not including the actual release date. Holidays are those of 5 U.S.C. 6103 as observed each year, computed by rule. Lapse days count as business days, and the affected releases are flagged. Reference months 2025-09 and 2025-10 are dropped. `D_t` stands alone, with no window to the second print." So `D_t` counts the weekdays `d` with the 12th < `d` < release date that are not holidays.
- **Holiday rule:** the legal public holidays of 5 U.S.C. 6103(a). These are New Year's Day (January 1), Martin Luther King Jr.'s Birthday (third Monday in January, from 1986), Washington's Birthday (third Monday in February), Memorial Day (last Monday in May), Juneteenth (June 19, from 2021), Independence Day (July 4), Labor Day (first Monday in September), Columbus Day (second Monday in October), Veterans Day (November 11), Thanksgiving Day (fourth Thursday in November) and Christmas Day (December 25). Under 6103(b) and its executive order, a Saturday holiday is observed the Friday before and a Sunday holiday the Monday after, so New Year's Day can fall on December 31 of the year before. The rule holds from 1978, when Veterans Day returned to November 11.
- **Not holidays (defaults F2, F3):** Inauguration Day (6103(c)), and closures by executive order, such as Christmas Eve and days of mourning.
- **Smallest standard (roadmap `:21`, verbatim):** "total nonfarm, NSA and SA, 1979–2026. Mean absolute F→S and F→T revision for each `D_t` value, and one regression with year and calendar-month controls. March 2020 to December 2021 reported apart. One findings file with the table and no figure."
- **Lapse flag (D1):** exactly the reference months 1995-12, 2013-09, 2013-10, 2025-11 and 2026-01. The release index's `off_schedule` is not a lapse flag: it also marks 1998-10 and the canceled 2025-10.
- **Sample:** the `cesnaicsrev` change rows, reference months 1979-01 to 2026-08, in thousands. A revision is |later − first| and exists only where both estimates are non-null. The `A` and `NA` markers and the ends of the S and T series drop out, never as zeros. Reference months 2025-09 and 2025-10 drop. The pandemic period, 2020-03 to 2021-12, is reported apart.
- **Regression (defaults F4–F6):** one specification, fit four times (F→S and F→T, each NSA and SA), on the main period only. Absolute revision on `D_t`, the lapse indicator, an intercept, reference-year dummies and calendar-month dummies. Standard errors cluster by reference year with the CR1 factor `G/(G−1)·(N−1)/(N−K)`.
- **Sources:** no new source, and `data/raw/` stays untouched. The report reads the built `data/panel/`.
- **Tests:** new tests are hermetic and unmarked. The gate is `uv run pytest -m "not slow and not network"`. Task 2 also runs `uv run pytest -m slow tests/test_vintage_build.py`.
- **Style:** `uv run ruff check` and `uv run ruff format --check` pass. The findings file follows CLAUDE.md's GitHub-renderable Markdown rules.

## Review Focus

1. A December window across New Year's Day: when January 1 falls on a Saturday, it is observed on December 31. December 2010 and December 2021 count 17 days, not 18 (Task 1).
2. A 13th that is a holiday: Columbus Day fell on October 13 in 2003, 2008, 2014 and 2025, so the window starts on the 14th. October 2014 counts 18 days (Task 1).
3. Holidays that began inside the sample: January 1985 counts 14 days with no King holiday, January 1986 counts 18 with it, and June 2021 counts 13 with Juneteenth's first observance on June 18 (Task 1).
4. Missing estimates: 2003-03, 2003-04, the 2025 lapse cells and the ends of the S and T series give no revision row, and F→S keeps more rows than F→T (Task 3).
5. Lapse releases: only the five D1 months carry the flag, 1998-10 does not, and the canceled 2025-10 release has no day count (Task 2).

## Decisions

- **D1** (owner, brief, 2026-11-15): Flag the five releases that a lapse delayed (December 1995, September and October 2013, November 2025 and January 2026), and keep them in the table and the regression with an indicator. The sources are BLS's footnote 1 for December 1995 (`data/raw/bls/histreleasedates.txt:61-62`, keyed at `data/raw/manual/es-reschedules.csv:2`) and the `shutdown_release_delayed` row (`docs/ces-revisions-review.md:100`). The January 2018 and 2018–19 lapses delayed no release, so they get no flag. Applied in: Global Constraints, Task 2, Task 3.
- **D2** (owner, brief, 2026-11-15): Execute inline in the planning session with executing-plans and no subagents, so Stage 1 takes one session. For Stage 1 only, this replaces the global routing to Sonnet in a fresh session. Applied in: header.
- **D3** (owner, brief, 2026-11-15): Keep executing-plans' Codex seat at the final review and its mutant gate at each task. The one whole-branch Opus review is executing-plans' final code-reviewer dispatch. Applied in: Tasks 1–3, final review.
- **D4** (owner, brief, 2026-11-15): Commit on a branch cut from local `main` at `0b9324b` (`claude/stage-1-collection-window-plan-5c7756`), merged later through finishing-a-development-branch. Nothing is committed on `main`. Applied in: every commit step.

## Defaults settled in this plan (for the owner's review)

- **F1** (brief): a test step gives the test's name and its assertions as one-line inline code, with no fenced blocks. Test-first covers the day count. The Review Focus pins add tests in Tasks 2 and 3, and the table and regression get verification steps.
- **F2:** Inauguration Day is not a holiday. 6103(c) covers only employees in the Washington area, and CES collection runs nationwide. Counting it would lower January `D_t` by 1 in 1981, 1985, 1989, 1993, 2005, 2009, 2017 and 2021.
- **F3:** Closures by executive order are not holidays, because the owner chose "computed by rule".
- **F4:** "One regression" is one specification fit four times, as the Regression constraint states.
- **F5:** The pandemic months stay out of the regression. 22 months cannot carry year and month dummies, and they would dominate a pooled fit.
- **F6:** Standard errors cluster by reference year (CR1), because absolute revisions are serially correlated inside a year.
- **F7:** From 2004 on, the table's November third estimate and December second estimate come from the January benchmark release (plan 5's reconciliation, `specs/plans/completed/5-ces-revisions.md:38`). So November F→T and December F→S include benchmark effects. The specification stays as the owner gave it, and the findings state this limit. Before 2003 it is unchecked.
- **F8:** `D_t` is a new `collection_days` frame of `VintageBuild`, not a column of the release index, whose design holds no closing dates.
- **F9:** The measured wall time is `scripts/vintage_sources.py build` plus `scripts/collection_window.py report`, each timed with `time -p` from a worktree with no `data/panel/`.

---

### Task 1: The federal holiday rule and the day count

**Files:**
- Create: `src/ces_revisions/vintages/federal_calendar.py`
- Create: `src/ces_revisions/vintages/collection_days.py`
- Test: `tests/test_federal_calendar.py`, `tests/test_collection_days.py`

**Interfaces:**
- Produces: `federal_holidays(year: int) -> tuple[date, ...]`, the sorted observed dates of that year's 6103(a) holidays, which can include December 31 of the year before. It raises `ValueError` for a year before 1978. `business_days(start: date, end: date) -> int`, the weekdays `d` with `start <= d < end` that are not holidays. `collection_days(reference_month: date, release_date: date) -> int`, which is `business_days` from the 13th of the reference month to the release date.

- [ ] **Step 1: Write the failing tests.** In `tests/test_federal_calendar.py`:
  - `test_federal_holidays_2026_match_opm`: `federal_holidays(2026) == (date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,5,25), date(2026,6,19), date(2026,7,3), date(2026,9,7), date(2026,10,12), date(2026,11,11), date(2026,11,26), date(2026,12,25))`
  - `test_saturday_new_years_day_is_observed_the_friday_before`: `date(2010,12,31) in federal_holidays(2011)` and `date(2010,12,31) not in federal_holidays(2010)`
  - `test_sunday_holiday_is_observed_the_monday_after`: `date(2021,7,5) in federal_holidays(2021)`
  - `test_holidays_follow_the_statute_in_force`: `[len(federal_holidays(y)) for y in (1985, 1986, 2020, 2021)] == [9, 10, 10, 11]`, `date(1985,1,21) not in federal_holidays(1985)`, `date(1986,1,20) in federal_holidays(1986)`, `date(2021,6,18) in federal_holidays(2021)`
  - `test_inauguration_and_executive_order_closures_are_not_holidays`: `date(2021,1,20) not in federal_holidays(2021)` and `date(2019,12,24) not in federal_holidays(2019)`
  - `test_years_before_1978_are_refused`: `pytest.raises(ValueError)` around `federal_holidays(1977)`
  - `test_business_days_skip_weekends_and_holidays`: `business_days(date(2026,8,13), date(2026,9,4)) == 16`, `business_days(date(2026,11,26), date(2026,11,27)) == 0`, `business_days(date(2026,9,4), date(2026,9,4)) == 0`

  In `tests/test_collection_days.py`:
  - `test_window_runs_from_after_the_12th_to_before_the_release`: `collection_days(date(2026,8,1), date(2026,9,4)) == 16`
  - `test_december_window_skips_observed_saturday_holidays`: `collection_days(date(2010,12,1), date(2011,1,7)) == 17` and `collection_days(date(2021,12,1), date(2022,1,7)) == 17`
  - `test_window_starts_after_a_columbus_day_on_the_13th`: `collection_days(date(2014,10,1), date(2014,11,7)) == 18`
  - `test_holidays_count_from_their_first_observance`: `collection_days(date(1985,1,1), date(1985,2,1)) == 14`, `collection_days(date(1986,1,1), date(1986,2,7)) == 18`, `collection_days(date(2021,6,1), date(2021,7,2)) == 13`, `collection_days(date(2026,6,1), date(2026,7,2)) == 12`
  - `test_lapse_days_count_as_business_days`: `collection_days(date(2026,1,1), date(2026,2,11)) == 20` and `collection_days(date(2013,9,1), date(2013,10,22)) == 26`

- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_federal_calendar.py tests/test_collection_days.py -q`. Expected: collection errors, `ModuleNotFoundError: No module named 'ces_revisions.vintages.federal_calendar'`.
- [ ] **Step 3: Implement `federal_calendar.py`** with `federal_holidays` and `business_days`, and **the scalar `collection_days`** in `collection_days.py`. Name the first years as constants: `FIRST_RULE_YEAR = 1978`, `KING_BIRTHDAY_FIRST_YEAR = 1986`, `JUNETEENTH_FIRST_YEAR = 2021`. `business_days` takes the holidays of `start.year` through `end.year + 1`, so a December 31 observance of the next New Year's Day is seen. `numpy.busday_count` or a day loop are both fine.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: 12 passed.
- [ ] **Step 5: Prove the tests.** Run executing-plans' mutant gate on this task's diff. Expected: every mutant killed, or each survivor ruled in the ledger.
- [ ] **Step 6: Commit.** `git add src/ces_revisions/vintages/federal_calendar.py src/ces_revisions/vintages/collection_days.py tests/test_federal_calendar.py tests/test_collection_days.py` then `git commit -m "Add the federal holiday rule and the collection-window day count"`.

### Task 2: One `D_t` for every release, in the vintage build

**Files:**
- Modify: `src/ces_revisions/vintages/collection_days.py`
- Modify: `src/ces_revisions/vintages/build.py` (a `collection_days` field after `accounting_decomposition`)
- Modify: `tests/vintage_data.py` (a cached `collection_days()`), `tests/test_vintage_build.py` (its key in `composed()`)
- Modify: `CLAUDE.md` (the build's artifact list in the `data/raw/` bullet)
- Test: `tests/test_collection_days.py`

**Interfaces:**
- Consumes: `collection_days(reference_month, release_date)` (Task 1). The release index frame from `release_index.build_release_index`.
- Produces: `LAPSE_DELAYED_MONTHS: tuple[date, ...]`, the five D1 months in order. `build_collection_days(index: pl.DataFrame) -> pl.DataFrame` with the columns `reference_month: Date`, `release_date: Date`, `collection_days: Int32` and `lapse_delayed: Boolean`. It has one row for each index row with `status == "released"`, sorted by `reference_month`, and keeps 2025-09, which only the analysis drops. `VintageBuild.collection_days`, written as `data/panel/collection_days.parquet`.

- [ ] **Step 1: Write the failing tests** in `tests/test_collection_days.py`, on `vintage_data.collection_days()`:
  - `test_every_release_has_one_day_count`: `frame.height == 571`, `frame["reference_month"].is_unique().all()`, `date(2025,10,1) not in frame["reference_month"].to_list()`, `frame["collection_days"].null_count() == 0`
  - `test_only_the_five_lapse_delayed_releases_are_flagged`: `LAPSE_DELAYED_MONTHS == (date(1995,12,1), date(2013,9,1), date(2013,10,1), date(2025,11,1), date(2026,1,1))` and `frame.filter("lapse_delayed")["reference_month"].to_list() == list(LAPSE_DELAYED_MONTHS)`
  - `test_built_day_counts_match_hand_counts`: the rows for 1995-12, 2013-09, 2025-09 and 2026-08 carry 24, 26, 46 and 16
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_collection_days.py -q`. Expected: a collection error, `ImportError: cannot import name 'LAPSE_DELAYED_MONTHS'`.
- [ ] **Step 3: Implement** `LAPSE_DELAYED_MONTHS` and `build_collection_days`. Cite D1's sources in a comment beside the constant. Add the frame to `VintageBuild` and `build()`, the cached accessor, the `composed()` key and the CLAUDE.md list entry.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `uv run pytest tests/test_collection_days.py -q` and `uv run pytest -m slow tests/test_vintage_build.py -q`. Expected: all pass, with the slow tier's parametrized test now covering `collection_days`.
- [ ] **Step 5: Prove the tests.** Run the mutant gate on this task's diff. Expected: every mutant killed, or each survivor ruled in the ledger.
- [ ] **Step 6: Commit.** `git add` the five files, then `git commit -m "Write one collection-window day count per release in the vintage build"`.

### Task 3: The table, the regression and the findings file

**Files:**
- Create: `src/ces_revisions/collection_window.py`
- Create: `scripts/collection_window.py`
- Create: `specs/ces-revision-drivers-stage-1-findings.md`
- Modify: `CLAUDE.md` (one Commands line for the report)
- Test: `tests/test_collection_window.py`

**Interfaces:**
- Consumes: the `cesnaicsrev` change rows of `data/panel/panel.parquet` (`reference_month`, `seasonal_status`, `release_stage`, `value_thousands`), and `data/panel/collection_days.parquet` (Task 2). In tests, `vintage_data.table_changes()` and `vintage_data.collection_days()`.
- Produces, in `ces_revisions.collection_window`:
  - `PANDEMIC_MONTHS = (date(2020,3,1), date(2021,12,1))`, a closed range, and `DROPPED_MONTHS = (date(2025,9,1), date(2025,10,1))`.
  - `absolute_revisions(changes: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame`: one row per reference month, seasonal status and transition with both estimates present, from the rows with `source == "cesnaicsrev"` only. Its columns are `reference_month`, `seasonal_status`, `transition` (`"F_S"` or `"F_T"`, the accounting module's naming), `absolute_revision` (Int64), `collection_days`, `lapse_delayed` and `period` (`"main"` or `"pandemic"`).
  - `revisions_by_days(revisions: pl.DataFrame) -> pl.DataFrame`: one row per `period`, `seasonal_status`, `transition` and `collection_days`, with `n`, `lapse_n` and `mean_absolute_revision`.
  - `ols_clustered(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: the OLS coefficients and their CR1 standard errors.
  - `days_regression(revisions: pl.DataFrame) -> pl.DataFrame`: one row per `seasonal_status` and `transition`, from the main period, with `n`, `clusters`, `days_coefficient`, `days_se`, `lapse_coefficient` and `lapse_se`.
  - `scripts/collection_window.py report`: reads `data/panel/`, renders the generated blocks `stage-1-main`, `stage-1-pandemic` and `stage-1-regression` with `archive_inventory.replace_generated`, writes them into the findings file and prints its wall time. With no `data/panel/`, it exits non-zero and names `scripts/vintage_sources.py build`.

- [ ] **Step 1: Write the Review Focus tests** in `tests/test_collection_window.py`, on `absolute_revisions(vintage_data.table_changes(), vintage_data.collection_days())`:
  - `test_missing_estimates_drop_rather_than_count_as_zero`: no row for (2003-04, `F_S`), (2003-03, `F_T`) or (2003-04, `F_T`); no row in `DROPPED_MONTHS`; the latest `F_S` month is 2026-07 and the latest `F_T` month 2026-06; for each seasonal status the `F_S` count exceeds the `F_T` count
  - `test_pandemic_months_are_reported_apart`: the `"pandemic"` months are exactly the 22 months from 2020-03 to 2021-12
  - `test_lapse_flag_reaches_the_revisions`: the sorted unique months with `lapse_delayed` equal `list(LAPSE_DELAYED_MONTHS)`
- [ ] **Step 2: Run them to verify they fail.** Run: `uv run pytest tests/test_collection_window.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.collection_window'`.
- [ ] **Step 3: Implement `ces_revisions.collection_window`** as its Interfaces state. In `days_regression`, the design is the intercept, `collection_days`, `lapse_delayed`, reference-year dummies without the first year and calendar-month dummies without January.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: 3 passed.
- [ ] **Step 5: Write the script and the findings skeleton.** The findings file has the sections Question, Data and day count, Results (the three generated blocks, each between `<!-- BEGIN GENERATED <name> -->` and `<!-- END GENERATED <name> -->`), Limits and Measured cost. The tables give means to one decimal and coefficients to two decimals, in thousands.
- [ ] **Step 6: Make the real run.** Run: `time -p uv run python scripts/vintage_sources.py build`, then `time -p uv run python scripts/collection_window.py report`. Expected: the build manifest lists `collection_days` with 571 rows. The main tables hold `D_t` values from 12 to 26, with no 46. The regression block has four rows, each with 47 clusters (1979–2020 and 2022–2026).
- [ ] **Step 7: Cross-check the regression.** In `/tmp`, refit the four designs with `uv run --with statsmodels python` and `OLS(...).fit(cov_type="cluster", cov_kwds={"groups": year})`. Expected: coefficients and standard errors agree with `days_regression` to 1e-8. If statsmodels does not install on Python 3.14, compare the coefficients with `np.linalg.lstsq`, and record the fallback in the ledger.
- [ ] **Step 8: Write the findings prose.** Give the answer in two or three sentences, from the generated numbers. Limits name F7's benchmark effect, the cells with n = 1, the five lapse releases, and `D_t` as a release-date proxy for first closing (O4 replaces it). Measured cost gives the two wall times from Step 6 and "1 session".
- [ ] **Step 9: Verify.** Run: `uv run pytest -m "not slow and not network" -q`, `uv run ruff check` and `uv run ruff format --check`. Expected: all pass. Rerun the report and check that `shasum` of the findings file is unchanged, which shows the generated blocks are stable.
- [ ] **Step 10: Prove the tests.** Run the mutant gate on this task's diff. Expected: every mutant killed, or each survivor ruled in the ledger.
- [ ] **Step 11: Commit.** `git add` the five files, then `git commit -m "Measure revision size against the collection window for total nonfarm"`.
