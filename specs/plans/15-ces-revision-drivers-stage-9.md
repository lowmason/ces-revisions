# Stage 9 — Collection Window and the First-Closing Collection Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via executing-plans (default F1, approved at plan review on 2026-11-17). Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 9 — on plan completion, tick the stage and re-validate later stages against what shipped.

Process (default F1): Stage 1's lighter process. Test first for `W_t` and for the source parsers, one task-reviewer pass on Task 3, and one whole-branch Opus review at the end. No per-task reviewers otherwise, no subagent dispatch, and no code blocks in this plan.

**Goal:** Add the window to first closing `W_t` to every release, commit the C1 sources to `data/raw/`, check that no published C1 value changed, and report O3 (a), O3 (b) and O4 with their uncertainty in `specs/ces-revision-drivers-stage-9-findings.md`.

**Architecture:** `collection_days.py` gains the closing day and `W_t`, so `data/panel/collection_days.parquet` carries them. A new script, `scripts/collection_rate_sources.py`, stages the LABSTAT extract, `ce.footnote` and the Internet Archive copies of `cesregrec.htm` in a gitignored folder. It promotes them into `data/raw/` only after an offline validity check and the prose reading. A new module, `ces_revisions.vintages.collection_rates`, parses those sources into `data/panel/collection_rates.parquet` and holds the D8 comparison. A new module, `ces_revisions.first_closing`, builds the samples and the fits. `scripts/first_closing.py` runs the D8 check and writes the findings' generated blocks.

**Tech Stack:** Python 3.14, Polars, NumPy, SciPy (t quantiles), pytest, ruff. No new dependency. `statsmodels` appears only in a `/tmp` cross-check through `uv run --with statsmodels`.

**Spec:** `specs/ces-revision-drivers-stage-9.md`, approved 2026-11-17, as of commit c146cb5. Executors read the spec beside this plan.

## Global Constraints

- **The first-closing rule (D1).** Cite it only as the owner's rule, in the roadmap's words: "first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release" (owner, 2026-11-17). This holds for every committed text: code comments, tests, commit messages, the findings and PR text. No committed text names or hints at any other source for the rule.
- **The window (spec, verbatim).** "`first_closing_date` = release date less 7 calendar days." "`first_closing_days` = `business_days(13th of the reference month, first_closing_date + 1 day)`." Passing the closing day itself as `end` is wrong.
- **Samples (spec § Samples).** Main period: reference months 2003-01 to 2026-08 with a release in the index. It excludes 2025-09 and 2025-10, March 2020 to December 2021 (reported apart) and the four lapse-delayed months 2013-09, 2013-10, 2025-11 and 2026-01 (D2). The fits have no lapse indicator.
- **Fits (spec, D4 and D10).** OLS with year and calendar-month effects and the 5-week dummy `five_week_interval`. CR1 errors clustered by reference year (`ols_clustered`), and 95% intervals with a t quantile on G − 1 degrees of freedom. The contrasts are `7β` for part (a), `10θ` for part (b), and 7 days for O4.
- **Network gate.** Each run of `fetch-labstat` or `fetch-archive` needs the owner's explicit yes in chat, each time. Never run `scripts/vintage_sources.py fetch`, `scripts/annual_sources.py fetch` or any `archive_inventory.py` network command.
- **Contact address.** Requests to bls.gov hosts go through `archive_inventory.open_url`, with `BLS_CONTACT_EMAIL` exported by `uv run --env-file <main checkout>/.project.env`. Never print, log or commit the address.
- **Internet Archive.** Pause `archive_inventory.PAUSE_SECONDS` before each request, and retry with `archive_inventory.retrying`.
- **`data/raw/` (CLAUDE.md).** Validate every payload. Stage it before any file is promoted. Give each new file a manifest row. Never hand-edit a fetched file. Never change or replace an existing source (Anti-goals).
- **Owner stops (spec).** (1) A staged archive copy whose prose describes when collection for the first preliminary estimate ends: stop before `promote` and report to the owner. Record the outcome of that reading in the ledger only, never in committed text. (2) Any C1, C2 or C3 value that differs between LABSTAT and an archive copy: stop before any fit and report to the owner.
- **Stage 1 stays unchanged.** Do not change the columns of `absolute_revisions` or `signed_revisions`, the Stage 1 tables, the scale model's evidence, or `docs/ces-revisions-review.md`. Do not edit `specs/ces-revision-drivers-roadmap.md:56`: the next roadmap resume settles it.
- **Manifest hash drift is expected.** New rows change the SHA-256 of `data/raw/manifest.csv`. Later runs record the new `stage3` hash in provenance, and committed evidence keeps the old one. `pilot_run.py` compares panel-array hashes, which the new sources do not change.
- **Tests.** New tests are hermetic and unmarked. The gate is `uv run pytest -m "not slow and not network"`. Tasks that touch the vintage build also run `uv run pytest -m slow tests/test_vintage_build.py`.
- **Style.** `uv run ruff check` and `uv run ruff format --check` pass. The findings file follows CLAUDE.md's GitHub-renderable Markdown rules.

## Review Focus

1. LABSTAT cells that are blank, `-`, or in period `M13`: no row, never a zero (Task 2).
2. Numbers or section names in a page's prose, such as the 2025 copy's note: no values, and no section starts from them (Task 2).
3. A LABSTAT download cut off before `Content-Length`, or ended by `http.client.IncompleteRead`: an error, and nothing staged (Task 3).
4. A later run of `vintage_sources.py fetch` or of `promote`: the Stage 9 manifest rows survive, and no existing file is overwritten (Task 3).
5. A main-period month with no C1: refused with the month named, never dropped in silence (Task 6).

## Decisions

None.

## Defaults settled in this plan

The owner approved the plan with these defaults on 2026-11-17 ("Approve as is").

- **F1 (brief): process.** Stage 1's process (roadmap `:23`), which was the owner's choice for Stage 1 only: executing-plans inline, test first for `W_t`, executing-plans' final whole-branch review, no per-task reviewers or subagents, and test steps as names with inline assertions. Two additions the owner has not seen: parser tests on synthetic fixtures before the gated fetch (Task 2), and one task-reviewer pass on Task 3, which writes `data/raw/` and changes the manifest code of `scripts/vintage_sources.py`.
- **F2: fetch shape.** The fetch is its own script, `scripts/collection_rate_sources.py`, with two network commands (`fetch-labstat`, `fetch-archive`) and three offline commands (`status`, `prose`, `promote`). Staging persists in the gitignored `data/cache/collection-rates/`, so the prose reading and a parser fix happen offline, between the fetch and `promote`. The first `fetch-archive` run fetches only the last copy of each year, which tests the parser on the real layouts before any walk. A later run, with its own yes, walks each year whose staged copies are all invalid down the year's captures until one parses or none remain. This follows the spec's walk-down rule in two or three yeses, and it never walks a year on a parser that has not met the real layouts.
- **F3: LABSTAT download.** The 353 MB file streams to staging. It is hashed while streaming and checked against `Content-Length`. It is deleted after the extract is written. A cut-off stream raises `ValueError`, so `archive_inventory.retrying` retries it.
- **F4: D8 comparison.** Values compare as integer tenths, `round(10 · value)`. The copies table gives, for each copy, the values compared, the values that differ and the values only in the copy. `scripts/first_closing.py report` runs the check first and refuses on any difference.
- **F5: quintiles.** The deviation is C1 less the mean C1 of its reference year over the part (a) months. Edges are `np.quantile(deviation, [0.2, 0.4, 0.6, 0.8])` with linear interpolation. Quintile = `np.searchsorted(edges, deviation, side="left") + 1`, so a value on an edge goes to the lower quintile.
- **F6: pandemic table.** One row per `W_t` value of March 2020 to December 2021, then an All row: months, mean C1, and the four mean absolute revisions.
- **F7: names.** `ces_revisions.vintages.collection_rates` (sources), `ces_revisions.first_closing` (analysis), `scripts/collection_rate_sources.py` (fetch and promote), `scripts/first_closing.py` (`check` and `report`).
- **F8: missing C1 is refused.** The spec's part (a) takes "every main-period month with both C1 and `W_t`". By the recon, no main-period month lacks C1, so a missing value is a data fault. `month_sample` raises and names the month instead of dropping it.
- **F9: expected counts.** Computed offline on 2026-11-17 from the committed `data/raw/`: part (a) has 256 months in 23 year clusters (2003–2020 and 2022–2026). Part (b) and O4 have 254 F→S and 252 F→T revisions for each status, in 23 clusters.
- **F10: measured cost.** Measured as Stage 1 measured it: `/usr/bin/time -p` on `scripts/vintage_sources.py build` and `scripts/first_closing.py report`, from a worktree with no `data/panel/`. Each network run is timed apart with `/usr/bin/time -p`.
- **F11: two CDX requests, not one.** The spec lists both page addresses "with one request to the Internet Archive's CDX index". The CDX `url=` parameter takes one address, and a prefix match on `bls.gov/web/` would list the whole directory. So `fetch-archive` makes one listing request for each address, as `archive_inventory` does for its two ZIP addresses.
- **F12: missing years in the findings.** A year from 2008 to 2024 with no committed copy is missing. `promote` refuses while a year is pending, so a year without a committed copy had every capture staged and none valid. The copies block gives that year a row that reads "no valid copy", derived from the committed timestamps.

---

### Task 1: The window to first closing in `collection_days.parquet`

**Files:**
- Modify: `src/ces_revisions/vintages/federal_calendar.py` (one function)
- Modify: `src/ces_revisions/vintages/collection_days.py` (module docstring, schema, two functions, `build_collection_days`)
- Modify: `CLAUDE.md` (the `data/raw/` bullet's artifact list: "the collection-window day count `D_t` and the window to first closing `W_t` of each release")
- Test: `tests/test_federal_calendar.py`, `tests/test_collection_days.py`

**Interfaces:**
- Consumes: `business_days(start, end)` and `federal_holidays(year)` in `ces_revisions.vintages.federal_calendar`.
- Produces: `is_federal_holiday(day: date) -> bool`, which checks the lists of `day.year` and `day.year + 1`. `federal_holidays` lists a Saturday New Year's Day under its own year, observed on December 31 of the year before, so `federal_holidays(day.year)` alone misses three of the six holiday closing days. `FIRST_CLOSING_LEAD_DAYS = 7`. `first_closing_date(release_date: date) -> date`. `first_closing_days(reference_month: date, release_date: date) -> int`. `SCHEMA` and the built frame gain `first_closing_date: pl.Date` and `first_closing_days: pl.Int32` after `lapse_delayed`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_federal_calendar.py`:
  - `test_is_federal_holiday_sees_a_new_years_day_observed_the_year_before`: `is_federal_holiday(date(2010,12,31))`, `is_federal_holiday(date(2010,1,1))`, and `not is_federal_holiday(date(2010,12,30))`

  In `tests/test_collection_days.py`:
  - `test_first_closing_is_one_week_before_the_release`: `first_closing_date(date(2026,7,2)) == date(2026,6,25)`
  - `test_window_to_first_closing_counts_through_the_closing_day`: `first_closing_days(date(2026,6,1), date(2026,7,2)) == 8` and `first_closing_days(date(2026,8,1), date(2026,9,4)) == 12`
  - `test_lapse_delayed_windows_follow_the_rule`: on `vintage_data.collection_days()`, the rows for 2013-09, 2013-10, 2025-09, 2025-11 and 2026-01 carry `first_closing_date` 2013-10-15, 2013-11-01, 2025-11-13, 2025-12-09 and 2026-02-04, and `first_closing_days` 22, 14, 42, 18 and 16
  - `test_window_to_first_closing_is_four_days_short_or_three_with_a_holiday`: for every row with `reference_month` from 2003-01 to 2026-08 and not `lapse_delayed` (278 rows), `collection_days − first_closing_days` is 3 or 4, and it is 3 exactly when `is_federal_holiday` holds for a date from `first_closing_date` + 1 to `release_date` − 1
  - `test_window_to_first_closing_counts_2003_2026`: on the same 278 rows, the counts of `first_closing_days` are `{8: 1, 9: 26, 10: 59, 11: 36, 12: 42, 13: 52, 14: 42, 15: 20}`, the one 8 is 2026-06, the difference is 4 in 232 rows and 3 in 46, and the 46 fall by calendar month as `{5: 13, 6: 13, 8: 10, 12: 10}`
  - `test_six_closing_days_fall_on_a_holiday`: on the same 278 rows, the reference months whose `first_closing_date` satisfies `is_federal_holiday` are exactly the Decembers of 2004, 2009, 2010, 2015, 2020 and 2021
  - Update `test_day_counts_come_sorted_with_the_stated_schema` to the six-column schema.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_federal_calendar.py tests/test_collection_days.py -q`. Expected: collection errors, `ImportError: cannot import name 'is_federal_holiday'` and `ImportError: cannot import name 'first_closing_date'`.
- [ ] **Step 3: Implement** `is_federal_holiday`, the two functions and the two columns. Rewrite the module docstring's sentence on the proxy: `D_t` keeps the release date as its end, and `first_closing_date` and `first_closing_days` follow the owner's rule in D1's words, cited as Global Constraints state.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `uv run pytest tests/test_federal_calendar.py tests/test_collection_days.py -q` and `uv run pytest -m slow tests/test_vintage_build.py -q`. Expected: all pass.
- [ ] **Step 5: Check that Stage 1 is unchanged.** Run: `uv run python scripts/vintage_sources.py build`, then `uv run python scripts/collection_window.py report`, then `git diff --exit-code specs/ces-revision-drivers-stage-1-findings.md`. Expected: exit 0.
- [ ] **Step 6: Commit.** `git add` the five files, then `git commit -m "Add the window to first closing to each release's day counts"`.

### Task 2: Parsers for the C1 sources, and the D8 comparison

**Files:**
- Modify: `src/ces_revisions/vintages/raw.py` (three path constants)
- Create: `src/ces_revisions/vintages/collection_rates.py`
- Create: `tests/fixtures/vintages/cesregrec-synthetic.htm`, written by hand (not an excerpt of a real copy)
- Test: `tests/test_collection_rates.py`

**Interfaces:**
- Produces, in `raw.py`: `COLLECTION_RATES = "bls/ce-collection-rates.tsv"`, `CE_FOOTNOTES = "bls/ce.footnote"`, `REGISTRY_COPIES = "internet-archive/cesregrec"`.
- Produces, in `collection_rates.py`:
  - `SERIES_BY_SECTION = {"first preliminary": "CEU00000000C1", "second preliminary": "CEU00000000C2", "third": "CEU00000000C3"}`, `LABSTAT_SOURCE = "labstat"`, `ARCHIVE_YEARS = range(2008, 2025)` and `RECON_COPY = "20250319165303"`.
  - `SCHEMA = {"series_id": pl.String, "reference_month": pl.Date, "source": pl.String, "value": pl.Float64, "footnote_code": pl.String}`.
  - `labstat_values(text: str) -> pl.DataFrame`: the rows of a LABSTAT `ce.data` text, tab-separated, with fields stripped of padding. It reads periods `M01` to `M12` with a numeric value. It skips the header line, `M13`, and blank or non-numeric values. Columns as `SCHEMA` with `source` = `LABSTAT_SOURCE`, and `footnote_code` null when blank.
  - `registry_page_values(page: str, source: str) -> pl.DataFrame`: values from `<tr>` rows only. A row with one non-empty cell whose lowercased text starts with a `SERIES_BY_SECTION` key starts that section. A row whose first cell is a four-digit year, inside a section, gives up to 12 monthly values in calendar order. A cell's value is its leading decimal number, and a cell without one gives no row. Reuse `raw.cell_text` and the row and cell regexes of `raw.py`. It raises `ValueError` naming the missing sections when any of the three sections has no value. `footnote_code` is null.
  - `build_collection_rates(raw_dir: Path = raw.RAW_DIR) -> pl.DataFrame`: the LABSTAT rows and the rows of every `REGISTRY_COPIES/<timestamp>.htm`, with `source` = that timestamp, sorted by `source`, `series_id` and `reference_month`.
  - `archive_comparison(rates: pl.DataFrame) -> pl.DataFrame`: one row per archive copy, sorted by `source`, with `compared`, `differing` and `only_in_copy` (all `pl.Int64`), over C1, C2 and C3 values as integer tenths (F4).
  - `archive_differences(rates: pl.DataFrame) -> pl.DataFrame`: `series_id`, `reference_month`, `source`, `copy_value`, `labstat_value` for each value whose tenths differ.

- [ ] **Step 1: Write the fixture.** One page with a prose paragraph above the table that holds the words "first preliminary" and the numbers "March 7, 2025". Then one `<table>`: a header row of month names, and for each of the three sections a one-cell section row and two year rows (1981 with a blank cell, and 2003 with all 12 months).
- [ ] **Step 2: Write the failing tests** in `tests/test_collection_rates.py`:
  - `test_labstat_values_read_padded_fields_and_skip_annual_and_blank_cells`: from an inline text with a header line, a padded `CEU00000000C1  \t2003\tM01\t     65.2\t` line, an `M13` line, a `-` value line and a blank value line, the frame has one row, `("CEU00000000C1", date(2003,1,1), "labstat", 65.2, None)`
  - `test_labstat_values_keep_footnote_codes`: a `2025\tM09\t 70.1\tE` line gives `footnote_code == "E"`
  - `test_registry_page_values_read_three_sections_into_c1_c2_c3`: the fixture gives 11 values for 1981 and 12 for 2003 in each of the three series, and `source` is the passed timestamp
  - `test_registry_page_values_ignore_numbers_and_section_names_in_prose`: no row has a year other than 1981 or 2003, and the C1 count equals the fixture's table count
  - `test_registry_page_without_three_sections_is_refused`: `pytest.raises(ValueError, match="third")` on the fixture with its third section removed, and on a page with prose and no table
  - `test_collection_rates_combine_labstat_and_each_copy`: in a `tmp_path` raw directory with a LABSTAT extract and two copies, the frame has three `source` values and the schema `SCHEMA`
  - `test_archive_comparison_counts_values_by_tenths`: LABSTAT 70.5 against a copy's 70.50 is compared and does not differ, 70.4 against 70.5 differs, a copy value with no LABSTAT value is `only_in_copy`, and `CEU05000000RR` rows are not compared
  - `test_archive_differences_list_each_changed_value`: the 70.4 case gives one row with `copy_value == 70.4` and `labstat_value == 70.5`
- [ ] **Step 3: Run the tests to verify they fail.** Run: `uv run pytest tests/test_collection_rates.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.vintages.collection_rates'`.
- [ ] **Step 4: Implement** the constants and the module as the Interfaces state.
- [ ] **Step 5: Run the tests to verify they pass.** Run: the Step 3 command. Expected: 8 passed.
- [ ] **Step 6: Commit.** `git add` the four files, then `git commit -m "Parse the LABSTAT collection rates and the archived registry pages"`.

### Task 3: The fetch and promote commands, and the existing fetch's manifest

**Files:**
- Create: `scripts/collection_rate_sources.py`
- Modify: `scripts/vintage_sources.py` (`fetch_sources` keeps the manifest rows of files it does not write)
- Modify: `CLAUDE.md` (the five Commands lines below, and a sentence in the `data/raw/` bullet: the Stage 9 sources, their two network commands, `collection_rate_sources.py promote` as the only offline command that adds files to `data/raw/`, and that `vintage_sources.py fetch` keeps their rows)
- Test: `tests/test_collection_rate_sources.py`, `tests/test_vintage_sources.py`

**Interfaces:**
- Consumes: `labstat_values`, `registry_page_values`, `ARCHIVE_YEARS`, `RECON_COPY` and the `raw.py` constants (Task 2). `archive_inventory.open_url`, `fetch`, `parse_cdx`, `retrying`, `CDX_URL`, `REPLAY_URL`, `PAUSE_SECONDS`, `read_csv` and `write_csv`. `vintage_sources.MANIFEST_COLUMNS` and `vintage_sources.manifest_row`.
- Produces, in `scripts/collection_rate_sources.py`:
  - `LABSTAT_URL = "https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries"`, `FOOTNOTE_URL = "https://download.bls.gov/pub/time.series/ce/ce.footnote"`, `SERIES = ("CEU00000000C1", "CEU00000000C2", "CEU00000000C3", "CEU05000000RR")`, `PAGE_KEYS = ("bls.gov/web/cesregrec.htm", "bls.gov/web/empsit/cesregrec.htm")`, `STAGING_DIR = DATA_DIR / "cache" / "collection-rates"`.
  - The staging layout: `raw/` mirrors `data/raw/`, `cdx/<n>.txt` holds each listing's raw text, and `provenance.csv` holds `file`, `url`, `last_modified`, `fetched_at`, `derived_from` and `derived_from_sha256` for each staged file.
  - `extract_series(lines: Iterable[bytes], series: tuple[str, ...]) -> bytes`: the first line, and every later line whose first tab field, stripped, is in `series`, byte for byte.
  - `stream_download(url: str, target: Path) -> tuple[str, str]`: the SHA-256 hex of the body and its `Last-Modified`. It raises `ValueError` when the bytes written differ from `Content-Length`, or when `http.client.IncompleteRead` ends the stream.
  - `fetch_labstat(now: datetime, staging: Path = STAGING_DIR) -> int`: exits 1 without `BLS_CONTACT_EMAIL`. It streams the full file to `staging/download/`, writes the extract to `raw/COLLECTION_RATES` and deletes the full file. It also fetches `FOOTNOTE_URL` to `raw/CE_FOOTNOTES`. Validation: the extract's header starts with `series_id`, each `SERIES` member has at least one line, and `labstat_values` reads C1 rows. The footnote header starts with `footnote_code`. The extract's provenance row: `derived_from` = `f"{LABSTAT_URL}, lines of {' '.join(SERIES)}"`, `derived_from_sha256` = the full file's hash, `last_modified` = the full file's header.
  - `year_candidates(listing: list[tuple[datetime, str, str, str]]) -> dict[int, list[tuple[str, str]]]`: for each year in `ARCHIVE_YEARS`, the `(timestamp, original)` pairs of its status-`200` captures from either address, latest first, with no repeated timestamp.
  - `year_status(candidates: dict[int, list[tuple[str, str]]], staged: dict[str, bool]) -> dict[int, str]`: for each year, the latest staged timestamp whose copy is valid, else `"missing"` when every candidate is staged and none is valid, else `"pending"`. `staged` maps a staged copy's timestamp to whether `registry_page_values` reads it.
  - `fetch_archive(now: datetime, staging: Path = STAGING_DIR) -> int`: lists each of the `PAGE_KEYS` with one `CDX_URL` request (F11), and saves each listing's text to `cdx/` before parsing it with `parse_cdx`. For a `"pending"` year with no staged copy, it fetches the latest candidate only. For a `"pending"` year whose staged copies are all invalid, it fetches unstaged candidates, latest first, until one parses or none remain. It fetches `RECON_COPY` while unstaged. Each replay URL is `REPLAY_URL` with the capture's timestamp and original. It exits 1 when the listing has no `RECON_COPY` capture.
  - `page_prose(page: str) -> str`: the page's text with every `<table>`, `<script>` and `<style>` element removed, tags stripped and whitespace collapsed.
  - Commands: `status` prints each year's status, the staged LABSTAT and footnote files, and every staged copy that does not parse, and exits 0 only when no year is `"pending"` and every source is staged. `prose` prints `page_prose` for `RECON_COPY` and each year's chosen copy. `promote` copies those files, the extract and the footnote into `data/raw/` and adds their manifest rows. It refuses (exit 1, writing nothing) while `status` would exit 1, and when any target file or manifest row already exists.
- Produces, in `scripts/vintage_sources.py`: `fetch_sources` writes the staged manifest with its own rows plus every existing row whose file it does not write and which still exists in `raw_dir`.

- [ ] **Step 1: Write the failing tests** in `tests/test_collection_rate_sources.py`, with `collection_rate_sources.archive_inventory.open_url` patched to return a fake response that has a body and headers, and with `tmp_path` staging. `tests/fake_https.py` answers with empty bodies only, so it does not fit here.
  - `test_extract_keeps_the_header_and_the_four_series_byte_for_byte`: padded lines with `\r\n` endings for C1, C2, C3, RR and two other series give the header and the four lines, unchanged
  - `test_short_download_stages_nothing`: a body shorter than its `Content-Length` raises `ValueError`, and an `http.client.IncompleteRead` raises `ValueError`. After `fetch_labstat` meets either one on every attempt, no `raw/` file is staged.
  - `test_fetch_labstat_refuses_without_a_contact_address`: exit 1, and no request made
  - `test_fetch_labstat_stages_the_extract_footnotes_and_provenance`: the staged extract equals `extract_series` of the fake body, `staging/download/` is empty, and the provenance row's `derived_from` starts with `LABSTAT_URL`, its `derived_from_sha256` is the fake body's SHA-256, and its `last_modified` is the fake header
  - `test_year_candidates_take_200_captures_of_either_address_latest_first`: a listing with captures of both addresses in 2007, 2010 and 2025 and with statuses `200`, `301` and `404` gives 2010's `200` captures only, latest first
  - `test_year_status_walks_down_to_the_next_copy_and_marks_exhausted_years_missing`: a valid latest copy is chosen, an invalid latest copy with an unstaged older one is `"pending"`, and a year with every candidate staged and invalid is `"missing"`
  - `test_fetch_archive_fetches_only_the_latest_copy_of_a_year_with_none_staged`: with nothing staged and three captures in 2010, a run fetches 2010's latest capture only, and pauses before each request
  - `test_fetch_archive_walks_down_a_year_whose_staged_copies_are_invalid`: with 2010 resolved and 2011's latest copy staged and invalid, a run fetches nothing for 2010. In 2011 it fetches the next copy, which is invalid, then the one after, which parses, and stops there.
  - `test_prose_drops_tables_and_tags`: `page_prose` of the Task 2 fixture holds the paragraph's words and none of the table's values
  - `test_promote_adds_rows_and_keeps_every_existing_row`: with a `tmp_path` raw directory holding one existing source and its row, `promote` copies the chosen files and adds one manifest row for each. The existing row is unchanged, the rows are sorted by `file`, and `vintage_sources.refresh_manifest(raw_dir)` then succeeds.
  - `test_promote_refuses_while_a_year_is_pending`: exit 1, and the raw directory unchanged
  - `test_promote_refuses_to_overwrite_an_existing_source`: with the extract already in the raw directory, exit 1, and its bytes unchanged

  In `tests/test_vintage_sources.py`:
  - `test_fetch_keeps_the_manifest_rows_of_sources_it_does_not_fetch`: as in `test_fetch_promotes_a_complete_validated_snapshot`, with `bls/ce-collection-rates.tsv` and its row present before the fetch. After the fetch, that row is unchanged and the file's bytes are unchanged.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_collection_rate_sources.py tests/test_vintage_sources.py -q`. Expected: `ModuleNotFoundError: No module named 'collection_rate_sources'`. With that module stubbed, the new `test_vintage_sources.py` test fails because the row is dropped.
- [ ] **Step 3: Implement** the script and the `fetch_sources` change as the Interfaces state. Add the Commands lines to CLAUDE.md:
  - `uv run --env-file <main checkout>/.project.env python scripts/collection_rate_sources.py fetch-labstat  # network, owner's yes: stage the LABSTAT C1–C3 extract and ce.footnote`
  - `uv run python scripts/collection_rate_sources.py fetch-archive  # network, owner's yes: stage the next Internet Archive copy of cesregrec.htm for each unresolved year`
  - `uv run python scripts/collection_rate_sources.py status  # offline: which staged copies parse, and which years remain`
  - `uv run python scripts/collection_rate_sources.py prose  # offline: print each chosen copy's text outside its tables`
  - `uv run python scripts/collection_rate_sources.py promote  # offline: add the staged sources to data/raw/ with manifest rows`
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command, then `uv run pytest -m "not slow and not network" -q`. Expected: all pass.
- [ ] **Step 5: Commit.** `git add scripts/collection_rate_sources.py scripts/vintage_sources.py CLAUDE.md tests/test_collection_rate_sources.py tests/test_vintage_sources.py`, then `git commit -m "Add the collection-rate fetch and promote commands, and keep their manifest rows"`.
- [ ] **Step 6: Review.** Dispatch one task-reviewer with this task's brief, the Step 4 test output and the diff of the Step 5 commit. The review checks the Global Constraints on `data/raw/`, the contact address and the Internet Archive pauses. Fix each finding test first, or rule it in the ledger, before Task 4.

### Task 4: The gated runs, the prose reading and the committed sources

This task writes `data/raw/`. It is a mutant task only if Step 3 changes the parser.

**Files:**
- Create: `data/raw/bls/ce-collection-rates.tsv`, `data/raw/bls/ce.footnote`, `data/raw/internet-archive/cesregrec/<timestamp>.htm` (by `promote` only)
- Modify: `data/raw/manifest.csv` (by `promote` only)
- Maybe modify: `src/ces_revisions/vintages/collection_rates.py`, with `tests/fixtures/vintages/cesregrec-<timestamp>-excerpt.htm` and its test (Step 3)
- Test: `tests/test_vintage_sources.py`

**Interfaces:**
- Consumes: Task 3's commands.
- Produces: the committed sources and their manifest rows, which Task 5 reads.

- [ ] **Step 1: Fetch LABSTAT.** Ask the owner for a yes to run `fetch-labstat`. On the yes only, run: `/usr/bin/time -p uv run --env-file <main checkout>/.project.env python scripts/collection_rate_sources.py fetch-labstat`. Expected: exit 0, and a staged extract with lines for all four series. Record the wall time in the ledger.
- [ ] **Step 2: Fetch the archive copies.** Ask the owner for a yes to run `fetch-archive`. On the yes only, run: `/usr/bin/time -p uv run python scripts/collection_rate_sources.py fetch-archive`. Expected: two listings, and up to 18 copies staged (the last `200` copy of each year from 2008 to 2024, and `RECON_COPY`). Record the wall time.
- [ ] **Step 3: Check validity offline.** Run: `uv run python scripts/collection_rate_sources.py status`. For each staged copy that does not parse, read it with a one-off `uv run python -I` script in `/tmp`.
  - If it holds the three sections in a layout the parser misses, copy a short excerpt to `tests/fixtures/vintages/cesregrec-<timestamp>-excerpt.htm`. Add a failing test that the excerpt gives its values, extend `registry_page_values` until the test passes, and rerun `status`.
  - If it truly lacks the table (a stub or an error page), leave it. Once the parser reads every real layout, a year that is still `"pending"` needs one more Step 2 run, with its own yes. That run walks the year down to its first valid copy, or marks it `"missing"`.
- [ ] **Step 4: Read the prose.** Run: `uv run python scripts/collection_rate_sources.py prose`, and read every copy's text. If any copy describes when collection for the first preliminary estimate ends, stop. Report the copy's timestamp and the passage to the owner in chat, and promote nothing until the owner rules. Record the outcome in the ledger only.
- [ ] **Step 5: Promote.** Run: `uv run python scripts/collection_rate_sources.py promote`, then `uv run python scripts/vintage_sources.py manifest`. Expected: `git diff data/raw/manifest.csv` shows only added lines, and the second command changes nothing more. `git status` lists only new files under `data/raw/bls/` and `data/raw/internet-archive/cesregrec/`.
- [ ] **Step 6: Pin the provenance.** In `tests/test_vintage_sources.py`, add `test_collection_rate_sources_record_their_origin`. The extract's row has `derived_from` starting with `LABSTAT_URL`, a 64-character `derived_from_sha256` and a `last_modified` ending in ` GMT`. The footnote's row has `url == FOOTNOTE_URL`. Each copy's row has a `url` that starts with `https://web.archive.org/web/<its timestamp>id_/`. The copies include `RECON_COPY`, and no year from 2008 to 2024 has more than one copy.
- [ ] **Step 7: Verify.** Run: `uv run pytest -m "not slow and not network" -q`. Expected: all pass, `test_the_manifest_lists_every_committed_source_with_its_hash_and_size` included.
- [ ] **Step 8: Commit.** `git add data/raw tests/test_vintage_sources.py`, and any Step 3 parser files. Then `git commit -m "Commit the collection-rate sources: the LABSTAT extract, ce.footnote and the archived registry pages"`.

### Task 5: The collection-rate table in the vintage build, and the D8 check

**Files:**
- Modify: `src/ces_revisions/vintages/build.py` (a `collection_rates` field after `collection_days`)
- Modify: `tests/vintage_data.py` (a cached `collection_rates()`), `tests/test_vintage_build.py` (its key in `composed()`)
- Create: `scripts/first_closing.py` (the `check` command, with `report` to come in Task 7)
- Modify: `CLAUDE.md` (the build's artifact list, a Commands line for `check`, and one layout bullet for Stage 9's modules and scripts)
- Test: `tests/test_collection_rates.py`

**Interfaces:**
- Consumes: `build_collection_rates`, `archive_comparison` and `archive_differences` (Task 2). The committed sources (Task 4).
- Produces: `VintageBuild.collection_rates`, written as `data/panel/collection_rates.parquet`. `scripts/first_closing.py check`, which reads that file, prints `archive_comparison` and every row of `archive_differences`, and exits 1 on any difference or when the file is missing (naming `scripts/vintage_sources.py build`).

- [ ] **Step 1: Write the failing tests** in `tests/test_collection_rates.py`, on `vintage_data.collection_rates()`:
  - `test_labstat_c1_runs_from_1981_without_october_2025`: the LABSTAT C1 months start at 1981-01, end no earlier than 2026-10, and lack 2025-10. BLS replaces the LABSTAT file each month, so the end month depends on the fetch date. Record the committed end month in the ledger. No sample depends on it, because every sample ends at 2026-08.
  - `test_footnote_e_marks_september_and_november_2025`: LABSTAT C1 has `footnote_code == "E"` at 2025-09 and 2025-11
  - `test_c1_matches_the_recon_values`: LABSTAT C1 is 77.9 at 2013-09, 83.5 at 2013-10, 73.8 at 2025-11 and 69.5 at 2026-01, and its yearly means round to 65.3 in 2003, 77.3 in 2013 and 60.4 in 2024
  - `test_the_2025_copy_matches_labstat`: source `RECON_COPY` has 530 C1, 526 C2 and 527 C3 values, runs from 1981-01 to 2025-02, and has no row in `archive_differences`
  - `test_check_refuses_without_a_built_panel`: with `first_closing.PANEL_DIR` patched to an empty `tmp_path`, `first_closing.main(["check"]) == 1`, and stderr names `vintage_sources.py build`
  - `test_check_prints_each_difference_and_exits_1`: with `vintage_data.collection_rates()` written to a patched `PANEL_DIR` after one copy's C1 value is raised by 0.1, `main(["check"]) == 1`, and stdout names that copy and month
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_collection_rates.py -q`. Expected: `AttributeError: module 'vintage_data' has no attribute 'collection_rates'`, and `ModuleNotFoundError: No module named 'first_closing'` for the two command tests.
- [ ] **Step 3: Implement** the build field, the cached accessor, the `composed()` key, `scripts/first_closing.py check` and the CLAUDE.md lines.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `uv run pytest tests/test_collection_rates.py -q` and `uv run pytest -m slow tests/test_vintage_build.py -q`. Expected: all pass. If a recon value differs, stop and report it to the owner. The recon read the same sources, so a difference means a parse fault or a changed source.
- [ ] **Step 5: Run the D8 check.** Run: `uv run python scripts/vintage_sources.py build`, then `uv run python scripts/first_closing.py check`. If it exits 1, stop before any fit, and report the differences to the owner (spec § Check that C1 never changed). If it exits 0, add `test_no_archived_value_differs_from_labstat`: `archive_differences` is empty, and `archive_comparison` lists every committed copy with its `compared` count as printed.
- [ ] **Step 6: Check that Stage 1 is unchanged.** Run: `uv run python scripts/collection_window.py report`, then `git diff --exit-code specs/ces-revision-drivers-stage-1-findings.md`. Expected: exit 0.
- [ ] **Step 7: Commit.** `git add` the changed files, then `git commit -m "Build the collection-rate table and check the archived values against LABSTAT"`.

### Task 6: The Stage 9 analysis functions

**Files:**
- Create: `src/ces_revisions/first_closing.py`
- Test: `tests/test_first_closing.py`

**Interfaces:**
- Consumes: `absolute_revisions`, `five_week_interval`, `ols_clustered`, `PANDEMIC_MONTHS`, `DROPPED_MONTHS` and `CONFIDENCE` from `ces_revisions.collection_window`. `is_federal_holiday` (Task 1), for `holiday_closings`. The frames of `collection_days.parquet` and `collection_rates.parquet`.
- Produces, in `ces_revisions.first_closing`:
  - `FIRST_MONTH = date(2003, 1, 1)`, `LAST_MONTH = date(2026, 8, 1)`, `C1_SERIES = "CEU00000000C1"`, `WINDOW_CONTRAST_DAYS = 7`, `RATE_CONTRAST_POINTS = 10`, `WINDOWS = ("collection_days", "first_closing_days")`.
  - `month_sample(days: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame`: one row per release with `reference_month` from `FIRST_MONTH` to `LAST_MONTH`, less `DROPPED_MONTHS`. Columns: `reference_month`, `release_date`, `first_closing_date`, `collection_days`, `first_closing_days`, `c1`, `c1_footnote` (LABSTAT C1), `five_week`, and `period` (`"pandemic"` inside `PANDEMIC_MONTHS`, `"lapse"` when `lapse_delayed`, else `"main"`). It raises `ValueError` naming each `"main"` month with no C1 (F8).
  - `revision_sample(revisions: pl.DataFrame, months: pl.DataFrame) -> pl.DataFrame`: the rows of `absolute_revisions` for the sample's months, joined on `reference_month` to `first_closing_days`, `c1` and `period` from `months`. The revision frame's own `period` and `lapse_delayed` are dropped.
  - `rates_by_window(months) -> pl.DataFrame`: per `first_closing_days` value of the main period, `n` and `mean_c1`.
  - `rate_regression(months) -> pl.DataFrame`: one row with `n`, `clusters`, `coefficient`, `se`, `contrast_low`, `contrast_high`. It fits C1 on `first_closing_days` over the main period, and the contrast covers `WINDOW_CONTRAST_DAYS`.
  - `revision_regression(revisions) -> pl.DataFrame`: one row per `seasonal_status` and `transition`, sorted, with the same columns. It fits the absolute revision on `c1` over the main period, and the contrast covers `RATE_CONTRAST_POINTS`.
  - `window_regressions(revisions) -> pl.DataFrame`: one row per `seasonal_status`, `transition` and `window` in `WINDOWS`, with the same columns. It fits the absolute revision on that window over the main period, and the contrast covers `WINDOW_CONTRAST_DAYS`.
  - `rate_quintiles(months) -> pl.DataFrame`: `reference_month`, `deviation` and `quintile` (`pl.Int8`, 1 to 5) for the main period, by F5.
  - `revisions_by_quintile(revisions, quintiles) -> pl.DataFrame`: one row per `quintile`, `seasonal_status` and `transition`, with `months` (main months in the quintile), `deviation_low`, `deviation_high`, `n` and `mean_absolute_revision`.
  - `lapse_releases(months, revisions) -> pl.DataFrame`: one row per `"lapse"` month with `reference_month`, `release_date`, `first_closing_date`, `collection_days`, `first_closing_days`, `c1`, `c1_footnote`, and `NSA_F_S`, `NSA_F_T`, `SA_F_S` and `SA_F_T` (absolute revisions, null where none).
  - `pandemic_by_window(months, revisions) -> pl.DataFrame`: per `first_closing_days` value of the pandemic period, `months`, `mean_c1` and the four mean absolute revisions, named as in `lapse_releases`.
  - `holiday_closings(months) -> pl.DataFrame`: the `"main"` and `"pandemic"` months whose `first_closing_date` is a federal holiday, with `reference_month`, `release_date` and `first_closing_date`.
  - `final_week_holidays(months) -> pl.DataFrame`: per calendar month, over the `"main"` and `"pandemic"` months, `releases` and `with_holiday`, the count with `collection_days − first_closing_days == 3`.
  - The fits share one private helper. Its design columns are the intercept, the regressor, a dummy for each reference year after the first, a dummy for each calendar month after January, and `five_week` last. It returns n, clusters, β, its CR1 standard error, and the contrast interval from the t quantile on clusters − 1 degrees of freedom.

- [ ] **Step 1: Write the failing tests** in `tests/test_first_closing.py`, on small synthetic frames built in the test file:
  - `test_month_sample_labels_each_period_and_drops_the_dropped_months`: one month each of main, pandemic and lapse, and 2025-10, give three rows with `period` main, pandemic and lapse
  - `test_month_sample_refuses_a_main_month_without_c1`: `pytest.raises(ValueError, match="2004-05")`
  - `test_revision_sample_takes_period_and_rate_from_the_months`: a revision row of a lapse month gets `period == "lapse"` and the month's `c1`
  - `test_fit_recovers_an_exact_slope`: `rate_regression` on 3 years of 12 main months, with `first_closing_days` varying within each year and `c1 = 2 × first_closing_days + year effect + month effect + 5 × five_week`, gives `coefficient == pytest.approx(2)`, `n == 36` and `clusters == 3`
  - `test_contrast_interval_uses_a_t_quantile_on_clusters_less_one`: the midpoint is `contrast × coefficient`, and the half-width is `contrast × t(0.975, clusters − 1) × se`, for both `rate_regression` and `revision_regression`
  - `test_only_the_main_period_enters_the_fits`: adding pandemic and lapse rows with outliers leaves every fit unchanged
  - `test_window_regressions_fit_each_window_on_the_same_rows`: both windows report the same `n`, and an outcome built on `first_closing_days` gives slope 2 for that window
  - `test_quintile_edges_put_a_tie_in_the_lower_quintile`: deviations 1 to 10 in one year (mean 0 after centering) give two months in each quintile, and a value equal to an edge goes to the lower quintile
  - `test_quintile_deviation_is_taken_within_the_reference_year`: two years with C1 means 60 and 80 give deviations centered on 0 in each
  - `test_lapse_releases_keep_missing_revisions_null`: a lapse month with no F→T row has null `NSA_F_T` and `SA_F_T`
  - `test_holiday_closings_and_final_week_counts`: a December release on 2011-01-07 has its closing day on 2010-12-31, a holiday, and counts once in `with_holiday`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_first_closing.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.first_closing'`.
- [ ] **Step 3: Implement** the module as the Interfaces state.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: 11 passed.
- [ ] **Step 5: Commit.** `git add src/ces_revisions/first_closing.py tests/test_first_closing.py`, then `git commit -m "Add the Stage 9 samples and fits on the window to first closing and C1"`.

### Task 7: The report command, the findings file and the Stage 1 Limits bullet

Steps 6 to 9 are not a mutant task.

**Files:**
- Modify: `scripts/first_closing.py` (the `report` command)
- Create: `specs/ces-revision-drivers-stage-9-findings.md`
- Modify: `specs/ces-revision-drivers-stage-1-findings.md` (D7, the first Limits bullet only)
- Modify: `CLAUDE.md` (a Commands line for `report`)
- Test: `tests/test_first_closing.py`

**Interfaces:**
- Consumes: every Task 6 function, `archive_comparison`, `archive_differences` and `archive_inventory.replace_generated`.
- Produces: `scripts/first_closing.py report`. It reads `collection_days.parquet`, `panel.parquet` and `collection_rates.parquet` from `data/panel/`, and exits 1 naming `scripts/vintage_sources.py build` when one is missing. It runs the D8 check first, and exits 1 printing the differences when any differ. It writes each block with `replace_generated(document, name, f"\n{body}\n\n")`, and prints the counts and the wall time. The blocks and their columns:
  - `stage-9-copies`: one row for each year of `ARCHIVE_YEARS` and one for `RECON_COPY`, with the copy's timestamp, values compared, differing and only in the copy. A year with no committed copy reads "no valid copy" (F12).
  - `stage-9-holidays`: the six holiday closing days (reference month, release, closing day), then the final-week counts by calendar month
  - `stage-9-rate-by-window`: `W_t`, months, mean C1, and an All row
  - `stage-9-rate-regression`: n, years, `W_t` coefficient, SE, and 8→15 days, 95% CI
  - `stage-9-revision-by-rate`: quintile, months, deviation range, and the mean absolute revision of NSA F→S, NSA F→T, SA F→S and SA F→T
  - `stage-9-revision-regression`: status, transition, n, years, C1 coefficient, SE, and +10 points, 95% CI
  - `stage-9-window-regression`: status, transition, n, years, then for `D_t` and for `W_t` the coefficient, SE and 7-day 95% CI
  - `stage-9-lapse`: the `lapse_releases` columns, with C1 followed by its footnote code
  - `stage-9-pandemic`: the `pandemic_by_window` columns, and an All row

  Means and C1 to one decimal, coefficients to two decimals, and intervals to one decimal. Revisions are in thousands, C1 in percent.

- [ ] **Step 1: Write the failing tests** in `tests/test_first_closing.py`:
  These tests follow Stage 1's `test_report_fills_the_four_generated_blocks`: they write `vintage_data.table_changes()`, `collection_days()` and `collection_rates()` to a patched `PANEL_DIR`, and patch `FINDINGS` to a `tmp_path` file with the nine block markers.
  - `test_report_refuses_without_a_built_panel`: with `PANEL_DIR` patched to an empty `tmp_path`, exit 1, and stderr names `vintage_sources.py build`
  - `test_report_refuses_when_an_archived_value_differs`: with one copy's C1 value raised by 0.1, exit 1, the difference printed, and the findings file unchanged
  - `test_report_fills_every_generated_block`: exit 0, every block is filled, the text outside the blocks is unchanged, and stdout reports 256 part (a) months
  - `test_render_window_regression_sets_both_windows_side_by_side`: the header and one row of the rendered O4 table, as exact strings
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_first_closing.py -q`. Expected: the four new tests fail with `AttributeError` for `report`.
- [ ] **Step 3: Implement** `report` and the renderers. Write the findings skeleton with the sections Question, Answer, Data and window, Results (the nine blocks, each under its own `###` heading with the formula of its fit in a ```` ```math ```` fence where it has one), Limits and Measured cost. Add the CLAUDE.md Commands line: `uv run python scripts/first_closing.py report  # rewrite the Stage 9 findings tables from data/panel/ (offline; run the vintage build first)`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: all pass.
- [ ] **Step 5: Commit.** `git add scripts/first_closing.py specs/ces-revision-drivers-stage-9-findings.md CLAUDE.md tests/test_first_closing.py`, then `git commit -m "Add the Stage 9 report command and the findings skeleton"`.
- [ ] **Step 6: Make the real run.** Run `rm -rf data/panel` (gitignored, and rebuilt by the next command). Then run `/usr/bin/time -p uv run python scripts/vintage_sources.py build`, then `/usr/bin/time -p uv run python scripts/first_closing.py report`. Expected: part (a) has n = 256 in 23 years, and part (b) and O4 have 254 F→S and 252 F→T rows for each status, in 23 years (F9). The `W_t` values of part (a) run from 8 to 15. Record both wall times.
- [ ] **Step 7: Cross-check the fits.** In `/tmp`, refit part (a), the four part (b) fits and the eight O4 fits with `uv run --with statsmodels python` and `OLS(...).fit(cov_type="cluster", cov_kwds={"groups": year})`. Expected: coefficients and standard errors agree to 1e-6. Pin the reference values in `tests/test_first_closing.py`, on `vintage_data`, as `test_fits_match_an_independent_statsmodels_fit`.
- [ ] **Step 8: Write the findings prose and the D7 bullet.**
  - Answer: two or three sentences each for O3 (a), O3 (b) and O4, from the generated numbers. State that O3 (b) is an association.
  - Data and window: the sources, the D8 result (copies compared, values checked), `W_t` by the owner's rule with D1's citation, the samples, and D3: BLS's "10–16 business days" is not reproduced, with the spec's ranges for the four start rules.
  - Results, beside the O4 table: `W_t − D_t` is constant except in the 46 releases with a holiday in the final week, so the two slopes can differ only through those releases. The `D_t` fit on 2003–2026 belongs to Stage 9 only, and it does not replace the Stage 1 tables.
  - Limits: at least the six items of spec § Findings file and report command. Take the meaning of footnote `E` from `data/raw/bls/ce.footnote`. Name the four lapse-delayed months, and state that the rule puts the 2013-09 closing inside the 2013 lapse.
  - Measured cost: the two wall times of Step 6, each network run's wall time from Task 4, and the sessions used.
  - D7: in `specs/ces-revision-drivers-stage-1-findings.md`, replace "Outcome O4 replaces it with BLS's dates when they arrive." with: "The [Stage 9 findings](ces-revision-drivers-stage-9-findings.md) repeat the regression with the window to first closing. That window follows the owner's rule (owner, 2026-11-17): first closing, when BLS stops collecting survey responses for the first estimate, falls one week before the release." Change nothing else in that file.
- [ ] **Step 9: Verify.** Run: `uv run pytest -m "not slow and not network" -q`, `uv run ruff check` and `uv run ruff format --check`. Expected: all pass. Rerun `scripts/first_closing.py report`, and check that `git diff --exit-code specs/ces-revision-drivers-stage-9-findings.md` exits 0 after a commit, which shows the blocks are stable. Rerun `scripts/collection_window.py report`, and check that the Stage 1 findings differ from `main` only in the D7 bullet.
- [ ] **Step 10: Commit.** `git add` the findings files and the test file, then `git commit -m "Report the first-closing collection rate, revision size and the window to first closing"`.

## Completion

After Task 7: run executing-plans' final whole-branch review, then writing-plans' Plan Completion Protocol. Its step 3 adds this item under § stage-1-scale-model of `specs/deferred_items.md` (after the last item at `:416` onward), not under a section for plan 15 (spec § Non-goals):

- [ ] C1, the first-closing collection rate, as a covariate on the log scale of the Stage 1 scale model (Approach C of `specs/ces-revision-drivers-stage-9.md`, deferred by its Non-goals). `specs/stage-1-scale-model/recommendation.md` built the model so that C1 can enter its log scale, and `data/panel/collection_rates.parquet` now supplies C1 from 1981. The fit needs a prior predictive check, SBC and four NUTS fits. It also meets the open decision above on where the scale model's result lands. Size: plan. Done when: the scale model's report gives C1's scale coefficient with its interval, or the owner drops the item.

The ticking pass of step 3 leaves the item "Parked Stage 12, collection-window panel, first-stage C1 model and macro controls" open: this plan delivers its O3 half, and its `Done when` also waits on O7.

Step 3a ticks Stage 9 in the roadmap and writes the Handoff section. Leave the roadmap's open question at `:56` to the next resume.
