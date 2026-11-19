# Stage 4 — Revision Size and BLS Funding and Staffing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via executing-plans (default F1). Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revision-drivers-roadmap.md, Stage 4 — on plan completion, tick the stage and re-validate later stages against what shipped.

Process (default F1): executing-plans inline, as plans 15 and 16. Test first for the summary-table reader, the fetch commands, the fiscal-year mapping and deflator, the fits and the report. executing-plans' mutant gate at each code task and its final whole-branch Opus review, with its Codex seat. One task-reviewer pass on Task 2. No subagent dispatch, and no code blocks in this plan.

**Goal:** Commit the missing funding and staffing sources, key one checked fiscal-year table of BLS funding and staffing for FY2009–FY2026, and report how revision size moved with real budget authority and actual FTE over FY2009–FY2025, with its uncertainty, in `specs/ces-revision-drivers-stage-4-findings.md` (O11).

**Architecture:** A new script, `scripts/funding_sources.py`, stages the 15 CBJ volumes, the transition brief and one ALFRED vintage of `GDPDEF` in the gitignored `data/annual/cache/o11/`. Its offline `promote` adds them to `data/annual/raw/` with catalog and manifest rows. A new module, `ces_revisions.annual.funding_tables`, reads a CBJ's summary table, the hand-keyed `data/annual/raw/manual/bls-funding-staffing.csv` and the `GDPDEF` vintage, and tests tie every keyed value to its source text. A new module, `ces_revisions.funding_staffing`, maps months to fiscal years, deflates, builds the samples and runs the 16 OLS fits and their leave-one-year-out refits. `scripts/funding_staffing.py report` writes six generated tables into the findings file. The annual build and its tables stay unchanged.

**Tech Stack:** Python 3.14, Polars, NumPy, SciPy (t quantiles), pytest, ruff, `pdftotext` (poppler). No new dependency. `statsmodels` appears only in a `/tmp` cross-check through `uv run --with statsmodels`.

**Spec:** `specs/ces-revision-drivers-stage-4.md`, approved 2026-11-18, as of commit b715a0c. Executors read the spec beside this plan.

## Global Constraints

- **Network gate (spec).** Each network run needs the owner's explicit yes in chat, each time: `funding_sources.py fetch-dol`, the ALFRED run (Task 3), the OPM recon and, if the OPM rule passes, `fetch-opm` (Task 4). Never run `scripts/annual_sources.py fetch`, `scripts/vintage_sources.py fetch`, or any network command of `archive_inventory.py` or `collection_rate_sources.py`.
- **Contact address.** No O11 host is a bls.gov host. Every O11 request goes through `archive_inventory.open_url` with `contact=""`, so it carries the bare `ces-revisions/0.1.0`, a redirect to bls.gov included. The O11 commands never read `.project.env` or `BLS_CONTACT_EMAIL`. Never print or commit the address.
- **`data/annual/raw/` (CLAUDE.md, spec Constraints).** Validate every payload, and stage it before any file is promoted. Give each new payload a catalog row and each new file a manifest row. Store the `pdftotext -layout` text beside each PDF. Never hand-edit a fetched file. Never replace or download again a committed source. Leave the `cbj-2024` and `cbj-2027` catalog rows (role `sample_metadata`) as they are.
- **Manifest stamps (spec).** A rehash keeps the `fetched_at` of every file whose SHA-256 is unchanged.
- **Existing annual tables (spec).** After every change to `data/annual/raw/`, `uv run python scripts/annual_sources.py build` writes the same `artifacts` (rows and content hashes) into `data/annual/panel/manifest.json`. Only `annual_source_manifest_sha256` changes. The funding table never enters `AnnualBuild`.
- **Untrusted downloads (spec).** Parse fetched files with plain parsing code. Any one-off reading of a download runs as `uv run python -I` on a script in `/tmp`.
- **Keying (brief).** Type each value from the CBJ's text file, and from its PDF page where the text layout is unclear. Never copy `summary_table` output into the table: the tie tests are the check on the keying. Key only the summary table's Total, General Funds, Unemployment Trust Funds and one-time lines. The Amounts Available table, the appropriation history and the object-class FTE give other figures for the same year (FY2022: 1,949 and 2,119 FTE). They are other concepts.
- **Owner stops (spec § Checks before any fit, § Sources to fetch).** A CBJ or brief address that fails its payload check stops the run at once, and goes to the owner. The keying stops (an unlisted activity line, the actual-FTE concept, a definition break, and one-time money inside a standing line) are batched (F2). No fit runs before every stop is ruled.
- **Revisions (spec, brief).** `absolute_revisions` from `ces_revisions.collection_window`, on `panel.parquet` and `collection_days.parquet`. The data label transitions `F_S` and `F_T` (`collection_window.TRANSITIONS`). The arrow forms exist only in renderers, and a filter on them returns an empty frame with no error.
- **Fiscal year (spec).** Reference month `m` belongs to fiscal year `year(m) + 1` when its month is October or later, else to `year(m)`. The reference month decides, never the release date.
- **Samples (spec § Samples).** Main: reference months 2008-10 to 2025-09, Stage 1 period `main`, not `lapse_delayed`. FY2026: its months that are not `lapse_delayed`, tables only. Pandemic: Stage 1 period `pandemic`, by fiscal year, no fit. Lapse: the `lapse_delayed` months of FY2009–FY2026 (2013-09, 2013-10, 2025-11, 2026-01), one table, no fit.
- **Fits (spec § Fit).** OLS of the absolute revision on `x`, the 5-week dummy `five_week` and calendar-month effects, with `x` the natural log of real budget authority or actual FTE divided by 100, in separate fits. The trend form adds a linear term in the fiscal year. No year effects: they would absorb `x`. CR1 errors cluster by fiscal year (`ols_clustered`), and 95% intervals take a t quantile on G − 1 degrees of freedom. Contrasts: $`\theta \ln 0.9`$ for 10% less real budget authority, and $`-\theta`$ for 100 fewer actual FTE.
- **Real budget authority (spec).** $`(\text{total} - \text{one-time}) \cdot d_{2025} / d_f`$, with $`d_f`$ the mean of `GDPDEF` over Q4 of calendar year $`f - 1`$ and Q1 to Q3 of calendar year $`f`$, from the one committed vintage. No interpolation to months: each month takes its fiscal year's value.
- **No model run.** O11 fits OLS only, so the roadmap's Where models run line sends nothing to Lambda.
- **Naming.** Committed text outside `specs/` names this work O11 or by its findings file, never "Stage 4" alone: docstrings, block markers (`o11-…`), the staging folder, commit messages and CLAUDE.md lines. The existing annual code already says "Stage 4" for the parked roadmap's stage.
- **Leave alone (brief).** The untracked `CES Revisions Methodology.pdf` at the main checkout's root, and the main checkout's ignored `data/runs/`, `.sdd/` ledgers and `data/cache/collection-rates/`. The plan needs none of them.
- **Tests and style.** New tests are hermetic and unmarked. The gate is `uv run pytest -m "not slow and not network"`. `uv run ruff check` and `uv run ruff format --check` pass. The findings file follows CLAUDE.md's GitHub-renderable Markdown rules.

## Review Focus

1. FY2021, which has no month in any fitted or FY2026 sample: the by-year table keeps its row with 0 months and `–` for each mean, never `NaN` or a dropped row, and no fit sees it (Task 7).
2. A `GDPDEF` vintage without Q3 2026: FY2026's real budget authority is blank and marked, never an average of three quarters (Tasks 6 and 8).
3. OPM left out by its rule: the report still runs, and the by-year table shows `–` in the OPM column (Task 8).
4. A summary-table line that is neither a standing line, a one-time line nor a fund line, such as an activity an older CBJ names: the check fails and names the CBJ and the line. The line is never counted as standing in silence (Task 5).
5. A payload whose text holds the summary table's title only in its table of contents, as line 15 of the FY2027 CBJ's text does, or an HTML error page saved as a PDF: refused at fetch, and nothing is staged for it (Task 2).

## Decisions

None. The owner's decisions of 2026-11-18 are D1–D5 in the spec.

## Defaults settled in this plan

Each is for the owner's review at the plan review.

- **F1: process.** executing-plans inline, as plans 15 and 16, because Tasks 3 to 5 stop for the owner's yes or ruling, which a subagent cannot ask for. Plan 15's one task-reviewer pass returns, on Task 2, which writes `data/annual/raw/`, changes the manifest code of every annual source and sends network requests. Effect if wrong: more or less review than the owner wants.
- **F2 (brief): batched keying stops.** Key all 17 first columns and the FY2027 CBJ's FY2026 Enacted column first. Then put each kind of keying stop to the owner as one question, listing every CBJ and line it covers, before any fit (Task 5, Step 6). The fetch stop cannot wait: a failed address stops `fetch-dol` at once. Recurring stops are a guess, not a finding: `docs/ces-revisions-review.md:90` records restatement footnotes in the appropriation histories, but the summary tables of the two CBJs on disk carry none. Effect if wrong: a stop that recurs in many CBJs interrupts the owner once per CBJ, not once per kind.
- **F3: names.** `scripts/funding_sources.py` (`fetch-dol`, `fetch-alfred`, `fetch-opm` if OPM passes, `promote`). `ces_revisions.annual.funding_tables` (source readers, beside the other readers of `data/annual/raw/`). `ces_revisions.funding_staffing` (analysis, beside `collection_window`). `scripts/funding_staffing.py report`. The keyed table `data/annual/raw/manual/bls-funding-staffing.csv`. Catalog roles `budget_justification`, `transition_brief`, `price_deflator` and `opm_headcount`. Block markers `o11-by-year`, `o11-sources`, `o11-fits`, `o11-leave-one-out`, `o11-pandemic` and `o11-lapse`. The reader module and the fetch script have different names because tests import scripts by bare name.
- **F4: fetch shape.** Staging persists in the gitignored `data/annual/cache/o11/`, so a stop on one address keeps the payloads already staged, and a rerun after the owner's ruling fetches only the rest. `promote` adds whole groups only (`dol`, `alfred`, `opm`) and refuses to overwrite. `annual_sources.py fetch` skips the four O11 roles, so a later run cannot replace the pinned `GDPDEF` vintage or download the CBJs again.
- **F5: the ALFRED endpoint.** ALFRED's download form for `GDPDEF`. The planning session read that form page once on 2026-11-18, through the assistant's web fetch. It listed vintage dates up to 2026-10-29 and offered Excel and Zipped CSV, and it named no API key. Its action URL and field names are not yet known, so the ALFRED run starts with one read of the form's HTML (Task 3, Step 3). The command asks for one vintage, the latest in the form's list, in Zipped CSV. It commits the zip and its CSV, and takes the vintage date from the payload. A key, a sign-in or a cookie stops the run.
- **F6: the transition brief's actual FTE as check-only rows.** The spec's actual-FTE check compares FY2009–FY2016 with the brief, but its series table has no row for the brief. So `brief_fte_actual` rows, FY2009–FY2016, carry the brief's actual FTE, tie to the brief's text and enter only the check and the Sources table. Effect if wrong: the brief's values live outside the keyed table, or reach a fit.
- **F7: the keyed table's columns.** `fiscal_year`, `series`, `value` (integer), `unit`, `concept`, `reference_date` (OPM rows only), `source_file` (the PDF or other payload under `data/annual/raw/`), `source_locator` (table, column and line, never a page or slide number), `note` (the text of any footnote, "comparable" heading or other basis marker the summary table attaches to the value) and `owner_ruling` (the owner's ruling on a stop, ending `(owner, YYYY-MM-DD)`). A row with a non-empty `note` needs a ruling. A failed check passes only for a row with a ruling.
- **F8: the OPM recon.** A `/tmp` one-off under `uv run python -I`, not a committed command. On the owner's yes, it reads OPM's API page and makes one request for BLS's on-board count for September 2025, which the review records as 2,165. Two branches follow (Task 4).
- **F9: the fits table.** One row per series and covariate (8 rows), with the no-trend and the trend fit side by side: n, G, contrast, SE and 95% interval for each, then the span of the two intervals, which the Answer's ruled-out statement quotes (spec § Fit). The 16 fits stay 16. The table only pairs them.
- **F10: the year that moves the contrast most.** In the leave-one-year-out refits, the fiscal year whose removal gives the largest absolute change from the full fit's contrast.
- **F11: measured cost.** As plan 16's F6: `uv sync`, `rm -rf data/panel`, then `/usr/bin/time -p` on `scripts/vintage_sources.py build` and on `scripts/funding_staffing.py report`. Each network run is timed apart with `/usr/bin/time -p`, and the keying time is recorded in the ledger.
- **F12: expected counts.** Computed in the planning worktree on 2026-11-18 from the built `data/panel/`: the main sample has 179 rows for each status and transition, in 16 fiscal years: FY2009–FY2012 12 each, FY2013 11, FY2014 11, FY2015–FY2019 12 each, FY2020 5, FY2022 9, FY2023–FY2024 12 each and FY2025 11. FY2026 has 7 F→S months (2025-12 and 2026-02 to 2026-07) and 6 F→T months (to 2026-06). Pandemic: FY2020 7, FY2021 12, FY2022 3. Lapse: 4 months × 2 statuses × 2 transitions = 16 rows. The build took 28.0 s of wall time.

---

### Task 1: The CBJ summary-table reader

**Files:**
- Create: `src/ces_revisions/annual/funding_tables.py`
- Test: `tests/test_funding_tables.py`

**Interfaces:**
- Produces:
  - `SUMMARY_HEADING: re.Pattern[str]`: a whole line holding `SUMMARY BUDGET AUTHORITY AND FTE BY ACTIVITY` in capitals, with any whitespace between words. Case-sensitive, so a contents line never matches.
  - `TOTAL = "Total"`, `FUND_LINES = ("General Funds", "Unemployment Trust Funds")`.
  - `SummaryLine`, a frozen dataclass: `label: str` (marker removed), `marker: str` (a trailing footnote marker such as `1/`, else `""`), `parent: str` (for a fund line, the activity or Total line above it; else its own label) and `values: tuple[int, ...]` (every number on the line, in order: FTE and amount for each column).
  - `SummaryTable`, a frozen dataclass: `years: tuple[int, ...]` (each distinct four-digit year after `FY` in the header lines, in order of first appearance), `lines: tuple[SummaryLine, ...]`, `notes: tuple[str, ...]` (each non-blank line after the last value line, stripped). Its method `cell(label: str, *, parent: str | None = None, column: int = 0) -> tuple[int, int]` gives `(values[2·column], values[2·column + 1])` of the one line with that label (and parent, when given). It raises `KeyError` naming the label when no line or more than one line matches.
  - `summary_table(text: str) -> SummaryTable`. The block runs from the `SUMMARY_HEADING` line to the first later line that holds only `BLS - <n>`, or to the end. Header lines are the block's lines before its first value line. A value line ends in at least two number tokens. A number token is whitespace-delimited, may start with `-`, `$` or `-$`, may hold thousands commas, and parses to an `int`. A label is the text before the first number token, stripped, with a trailing `\d+/` split off into `marker`. It raises `ValueError("no summary table of budget authority and FTE by activity")` when no line matches `SUMMARY_HEADING`, and a `ValueError` naming the heading when the block has no `Total` line.

- [ ] **Step 1: Write the failing tests** in `tests/test_funding_tables.py`. The first three read the committed text files under `raw.RAW_DIR`:
  - `test_reads_the_fy2024_cbj_first_column`: on `dol/cbj-2024-v3-01.txt`, `years == (2022, 2023, 2024)`, `cell("Total") == (1951, 687952)`, `cell("General Funds", parent="Total") == (1951, 619952)`, `cell("Unemployment Trust Funds", parent="Total") == (0, 68000)`, `cell("Headquarters Relocation") == (0, 28470)`, and `"NOTE: FY 2022 reflects actual FTE." in notes`
  - `test_reads_dollar_amounts_and_a_later_column`: on `dol/cbj-2027-v3-01.txt`, `years[:2] == (2025, 2026)`, `cell("Total") == (2019, 703952)`, `cell("Total", column=1) == (1773, 708500)`, `cell("General Funds", parent="Total") == (2019, 635952)`, `cell("Labor Market Information") == (0, 68000)`, and the Employment and Unemployment Statistics line's `values[-2:] == (-6, -21603)`
  - `test_activity_labels_of_the_committed_cbjs`: the labels of lines whose `parent` is their own label, `Total` left out, are the six of spec § What the two CBJs on disk show for FY2024, in table order, and Employment and Unemployment Statistics, Labor Market Information, Prices and Cost of Living, Compensation and Working Conditions, Productivity and Technology, and Executive Direction and Staff Services for FY2027
  - `test_a_contents_line_alone_is_no_summary_table`: a text holding only `Summary Budget Authority and FTE by Activity ........ 5` raises `ValueError` matching `no summary table`
  - `test_totals_of_other_tables_stay_outside_the_block`: a synthetic text with a `Total Change   2,023   +$697,952` line before the heading, a block with its own `Total`, a `BLS - 6` footer, and a `Total   9   9` line after the footer: `cell("Total")` returns the block's own pair
  - `test_a_footnote_marker_is_split_from_its_label`: a block line `Total 1/     2,000    $600,000` gives `label == "Total"`, `marker == "1/"` and `values == (2000, 600000)`
  - `test_an_ambiguous_label_is_refused`: on the FY2024 text, `cell("General Funds")` without `parent` raises `KeyError`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_funding_tables.py -q`. Expected: a collection error, `ModuleNotFoundError: No module named 'ces_revisions.annual.funding_tables'`.
- [ ] **Step 3: Implement** the module as the Interfaces state. Its docstring names O11 and the spec.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: 7 passed.
- [ ] **Step 5: Commit.** `git add src/ces_revisions/annual/funding_tables.py tests/test_funding_tables.py`, then `git commit -m "Read the summary table of a BLS budget justification (O11)"`.
- [ ] **Step 6: Prove the tests.** Run executing-plans' mutant gate on this task's diff. Expected: every mutant killed, or each survivor ruled in the ledger.

### Task 2: The manifest stamp rule, and the O11 fetch and promote commands

**Files:**
- Modify: `src/ces_revisions/annual/raw.py` (`manifest_frame`)
- Modify: `src/ces_revisions/annual/funding_tables.py` (`SOURCE_ROLES`)
- Modify: `scripts/annual_sources.py` (`refresh_manifest`, `fetch_sources`)
- Create: `scripts/funding_sources.py`
- Modify: `CLAUDE.md` (two Commands lines, and the `annual_sources.py fetch` bullet: O11's sources come from `scripts/funding_sources.py`, which sends no contact address, `annual_sources.py fetch` skips them, and `manifest` keeps each unchanged file's stamp)
- Test: `tests/test_annual_raw.py`, `tests/test_annual_sources.py`, `tests/test_funding_sources.py`

**Interfaces:**
- Consumes: `SUMMARY_HEADING` (Task 1). `annual_sources.retrying_download`, `validate_payload` and `refresh_manifest`. `raw.RAW_DIR`, `CACHE_DIR`, `MANIFEST`, `SOURCE_CATALOG`, `manifest_frame` and `read_source_catalog`.
- Produces, in `raw.py`: `manifest_frame(raw_dir: Path = RAW_DIR, *, fetched_at: str, previous: pl.DataFrame | None = None) -> pl.DataFrame`. As before, except that a file whose `file` and `sha256` match a row of `previous` keeps that row's `fetched_at`.
- Produces, in `funding_tables.py`: `SOURCE_ROLES = frozenset({"budget_justification", "transition_brief", "price_deflator", "opm_headcount"})`.
- Produces, in `annual_sources.py`: `refresh_manifest` passes the current manifest as `previous` when it exists. `fetch_sources` fetches only `available` rows whose role is not in `SOURCE_ROLES`.
- Produces, in `scripts/funding_sources.py`:
  - `STAGING_DIR = raw.CACHE_DIR / "o11"`. Layout: `raw/` mirrors `data/annual/raw/`. `provenance.csv` holds `group`, `source_id`, `role`, `url`, `file`, `source_locator`, `last_modified` and `fetched_at` for each staged payload. `unvouched/` keeps a payload that failed its check, for reading only.
  - `Payload`, a frozen dataclass: `group`, `source_id`, `role`, `url`, `file`, `source_locator`.
  - `CBJ_YEARS = (*range(2011, 2024), 2025, 2026)`. `dol_payloads() -> list[Payload]`: for each year, `cbj-<year>`, role `budget_justification`, URL `https://www.dol.gov/sites/dolgov/files/general/budget/<year>/CBJ-<year>-V3-01.pdf`, file `dol/cbj-<year>-v3-01.pdf`, locator `FY<year> BLS budget justification`. Then `bls-transition-brief`, role `transition_brief`, URL `https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf`, file `dol/bls-transition-brief.pdf`, locator `DOL BLS presidential transition brief`. All in group `dol`.
  - `pdf_to_text(pdf: Path) -> Path`: runs `pdftotext -layout` and returns the `.txt` path beside the PDF. Tests patch it.
  - `validate_cbj_text(text: str) -> None`: `ValueError` unless the text names "Bureau of Labor Statistics" in any case and holds a `SUMMARY_HEADING` line. `validate_brief_text(text: str) -> None`: `ValueError` unless it names "Bureau of Labor Statistics" in any case and holds `FTE`.
  - `fetch_dol(now: datetime, staging: Path = STAGING_DIR) -> int`: exits 1 naming poppler when `pdftotext` is missing. Takes each payload not yet staged, in order: `retrying_download(url, "")`, `validate_payload`, `pdf_to_text`, then the text check. On the first failure it moves the payload to `unvouched/`, prints the URL and the reason, and exits 1. Exits 0 when all 16 are staged.
  - `promote(groups: list[str], now: datetime, staging: Path = STAGING_DIR, raw_dir: Path = raw.RAW_DIR) -> int`: refuses with exit 1, writing nothing, when a named group lacks a staged payload or its derived file, when a target file exists in `raw_dir`, or when the catalog already has the `source_id` or `file`. Otherwise it copies each payload and its derived file, appends one catalog row per payload (`status` `available`, with `source_id`, `role`, `url`, `file` and `source_locator`, the other columns empty), and rewrites the manifest with `manifest_frame(fetched_at=<now>, previous=<current manifest>)`. Each new file's `fetched_at` is then set to its payload's staged fetch time. The catalog's own row takes the new hash and `now`.
  - `main(argv: list[str] | None = None) -> int` with `fetch-dol` (network) and `promote GROUP...` (offline).

- [ ] **Step 1: Write the failing tests.**

  In `tests/test_annual_raw.py`:
  - `test_manifest_frame_keeps_the_stamp_of_unchanged_files`: with `previous` stamping files `a` and `b` at `2026-09-16T14:54:07Z`, where `b`'s bytes have since changed and `c` is new, a frame stamped `2026-11-18T12:00:00Z` keeps `a`'s old stamp and gives `b` and `c` the new one

  In `tests/test_annual_sources.py`:
  - `test_refresh_manifest_keeps_existing_rows`: in a `tmp_path` raw directory with a manifest, adding one file and calling `refresh_manifest` leaves every earlier row unchanged, field for field, and adds one row with the new stamp
  - `test_fetch_skips_the_o11_sources`: as in `test_fetch_validates_every_payload_before_replacing_sources`, a catalog with a `budget_justification` row and a `technical_notes` row: the patched `download` sees the second URL only

  In `tests/test_funding_sources.py`, with `retrying_download` and `pdf_to_text` patched and `tmp_path` staging and raw directories:
  - `test_dol_payloads_are_the_fifteen_cbjs_and_the_brief`: 16 payloads; the CBJ years are 2011–2023, 2025 and 2026; the URLs and files are those of spec § Sources to fetch
  - `test_o11_requests_never_carry_the_contact`: with `BLS_CONTACT_EMAIL` set, `fetch_dol` passes `""` as the contact on every request, and through `fake_https.serve_https` with a redirect from a dol.gov URL to a bls.gov URL, `open_url(url, timeout=5, contact="")` sends the bare `ces-revisions/0.1.0` on both hops
  - `test_a_contents_line_alone_fails_the_cbj_check`: `validate_cbj_text` raises on a text that names the bureau and holds only the contents line, and passes on the committed FY2024 text
  - `test_an_html_page_saved_as_a_pdf_stages_nothing`: a payload body `<html>error</html>`: exit 1, the URL printed, nothing in `raw/`, no provenance row
  - `test_a_failed_payload_stops_the_run_and_keeps_the_earlier_ones`: the first two payloads valid, the third with text lacking the heading: exit 1, the third URL printed, the first two staged with provenance rows, the third in `unvouched/`, and the fourth never requested
  - `test_a_rerun_fetches_only_unstaged_payloads`: after that run, with the third payload now valid, a rerun requests the third payload onward, never the first two
  - `test_promote_adds_files_catalog_and_manifest_rows`: with a raw directory holding one source, its catalog row and its manifest row, `promote(["dol"])` copies each PDF and its text, appends 16 catalog rows after the existing one, and adds 32 manifest rows stamped with the staged fetch times. The existing catalog row and the existing source's manifest row are unchanged, the catalog's manifest row has its new hash and the run's stamp, and the manifest lists every file with its SHA-256 and size.
  - `test_promote_refuses_an_incomplete_group`: one payload unstaged: exit 1, and the raw directory unchanged
  - `test_promote_refuses_to_overwrite`: a target already in the raw directory: exit 1, and its bytes unchanged
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_annual_raw.py tests/test_annual_sources.py tests/test_funding_sources.py -q`. Expected: `ModuleNotFoundError: No module named 'funding_sources'`, and the three new annual tests fail.
- [ ] **Step 3: Implement** the changes and the script as the Interfaces state. Add the CLAUDE.md lines:
  - `uv run python scripts/funding_sources.py fetch-dol  # network, owner's yes: stage the O11 budget justifications and the BLS transition brief in data/annual/cache/o11/`
  - `uv run python scripts/funding_sources.py promote dol alfred  # offline: add staged O11 sources to data/annual/raw/ with catalog and manifest rows`
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command, then `uv run pytest -m "not slow and not network" -q`. Expected: all pass, `test_manifest_matches_every_committed_annual_source` included.
- [ ] **Step 5: Commit.** `git add` the five changed or new source files, CLAUDE.md and the three test files, then `git commit -m "Add the O11 fetch and promote commands, and keep the stamps of unchanged annual sources"`.
- [ ] **Step 6: Review.** Dispatch one task-reviewer with this task's brief, the Step 4 output and the diff of the Step 5 commit. It checks the Global Constraints on `data/annual/raw/`, the contact address and the manifest stamps. Fix each finding test first, or rule it in the ledger, before Task 3.
- [ ] **Step 7: Prove the tests.** Run executing-plans' mutant gate on this task's diff. Expected: every mutant killed, or each survivor ruled in the ledger.

### Task 3: The dol.gov and ALFRED runs, and the committed sources

This task writes `data/annual/raw/`.

**Files:**
- Modify: `scripts/funding_sources.py` (`fetch-alfred` and the `alfred` group)
- Maybe modify: `scripts/archive_inventory.py` (Step 4 only)
- Create, by `promote` only: `data/annual/raw/dol/cbj-<year>-v3-01.pdf` and `.txt` for the 15 years, `data/annual/raw/dol/bls-transition-brief.pdf` and `.txt`, `data/annual/raw/alfred/gdpdef-<vintage>.zip` and `.csv`
- Modify, by `promote` only: `data/annual/raw/source-catalog.csv`, `data/annual/raw/manifest.csv`
- Modify: `CLAUDE.md` (one Commands line)
- Test: `tests/test_funding_sources.py`, `tests/test_annual_sources.py`

**Interfaces:**
- Consumes: Task 2's commands and helpers.
- Produces, in `scripts/funding_sources.py`:
  - `ALFRED_FORM_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=GDPDEF"`, and `alfred_url(vintage: date) -> str`, which asks the form's target for `GDPDEF` in Levels, the one vintage, all observations, output "Observations by Vintage Date, All Observations" and file format Zipped CSV, with the field names that Step 3 reads.
  - `alfred_payload(zipped: bytes, vintage: date) -> bytes`: the zip's one CSV member, byte for byte. `ValueError` unless the zip holds exactly one `.csv` member whose header names `GDPDEF` and the vintage, with quarterly rows from 1947-01-01 through at least 2026-04-01.
  - `fetch_alfred(vintage: date, now: datetime, staging: Path = STAGING_DIR) -> int`: one `retrying_download(alfred_url(vintage), "")`, then stages `alfred/gdpdef-<YYYY-MM-DD>.zip` and its `.csv`, with source id `gdpdef-<YYYY-MM-DD>`, role `price_deflator` and locator `GDPDEF, ALFRED vintage <YYYY-MM-DD>`, in group `alfred`. A failed check moves the zip to `unvouched/` and exits 1.
  - The command `fetch-alfred --vintage YYYY-MM-DD` (network).

- [ ] **Step 1: Baseline the annual tables.** Run: `uv run python scripts/annual_sources.py build`, then `cp data/annual/panel/manifest.json /tmp/o11-annual-before.json`.
- [ ] **Step 2: Fetch from dol.gov.** Ask the owner for a yes to run `fetch-dol`. On the yes only, run: `/usr/bin/time -p uv run python scripts/funding_sources.py fetch-dol`. Expected: exit 0, and 16 payloads staged. On exit 1, stop and report the URL and the reason to the owner. Do not search for another address (spec § Sources to fetch). A rerun after the ruling needs its own yes. Record each run's wall time in the ledger.
- [ ] **Step 3: Read the ALFRED form.** Ask the owner for one yes that covers the ALFRED run: one GET of `ALFRED_FORM_URL` now, and one download in Step 5. On the yes only, fetch the form page with a `/tmp` script under `uv run python -I` that sends the bare `User-Agent: ces-revisions/0.1.0` through `urllib`, and read its HTML: the form's action, its method, the field names and values for Levels, the output format, Zipped CSV and the observation range, and the latest vintage date in its list. Record them in the ledger. If the download needs an API key, a sign-in or a cookie, stop and report to the owner: a key is a secret and never enters the repo.
- [ ] **Step 4: Write `fetch-alfred` test first.** In `tests/test_funding_sources.py`:
  - `test_alfred_url_asks_for_one_vintage_of_gdpdef`: `alfred_url(date(2026,10,29))` equals the exact URL that Step 3's fields give
  - `test_alfred_payload_returns_the_csv_of_its_vintage`: a synthetic zip with one CSV from 1947-01-01 to 2026-07-01, its header naming `GDPDEF` and the vintage, returns that CSV's bytes
  - `test_alfred_payload_refuses_another_vintage_or_a_short_series`: a header with another vintage, a series that ends at 2026-01-01, and a zip with two members each raise `ValueError`
  - `test_fetch_alfred_stages_the_zip_and_its_csv`: with `retrying_download` patched, the two files and one provenance row are staged, and the contact passed is `""`

  The synthetic header follows the payload layout that Step 3's form promises (`GDPDEF_<YYYYMMDD>` columns when the form says so). If the form submits by POST, first add `data: bytes | None = None` to `archive_inventory.open_url`, test first: a POST through `fake_https.serve_https` keeps the bare User-Agent. Run the tests to see them fail, implement, and run them to see them pass. Add the CLAUDE.md line: `uv run python scripts/funding_sources.py fetch-alfred --vintage YYYY-MM-DD  # network, owner's yes: stage one ALFRED vintage of GDPDEF`. Commit with `git commit -m "Add the O11 GDPDEF vintage fetch"`.
- [ ] **Step 5: Fetch the vintage.** Under Step 3's yes, run: `/usr/bin/time -p uv run python scripts/funding_sources.py fetch-alfred --vintage <the latest date from Step 3>`. Expected: exit 0. If the real payload's layout differs from Step 4's fixture, it lands in `unvouched/`. Fix `alfred_payload` test first on an excerpt of it, and rerun with a new yes. Record the wall time.
- [ ] **Step 6: Promote.** Run: `uv run python scripts/funding_sources.py promote dol alfred` (`main` passes the run's time as `now`), then `uv run python scripts/annual_sources.py manifest`. Expected: the second command changes nothing more. `git status` lists only new files under `data/annual/raw/dol/` and `data/annual/raw/alfred/`, and the two changed CSVs. `git diff data/annual/raw/manifest.csv` shows added lines only.
- [ ] **Step 7: Check the existing annual tables.** Run: `uv run python scripts/annual_sources.py build`, then compare `data/annual/panel/manifest.json` with `/tmp/o11-annual-before.json` in a `/tmp` script. Expected: the `artifacts` objects are equal, and only `annual_source_manifest_sha256` differs.
- [ ] **Step 8: Pin the sources.** In `tests/test_annual_sources.py`, add `test_o11_sources_are_catalogued_and_archived`: the catalog has 15 `budget_justification` rows (`cbj-2011` to `cbj-2023`, `cbj-2025`, `cbj-2026`), one `transition_brief` row and one `price_deflator` row whose locator holds the date in its file name. Each new PDF has its text beside it, each CBJ text passes `validate_cbj_text`, and every manifest row of a file that existed before, `source-catalog.csv` aside, keeps `2026-09-16T14:54:07Z`.
- [ ] **Step 9: Verify.** Run: `uv run pytest -m "not slow and not network" -q`. Expected: all pass.
- [ ] **Step 10: Commit.** `git add data/annual/raw tests/test_annual_sources.py`, and any Step 5 fix. Then `git commit -m "Commit the O11 sources: 15 budget justifications, the BLS transition brief and one GDPDEF vintage"`.

### Task 4: The OPM recon and the OPM rule

**Files (pass branch only):**
- Modify: `scripts/funding_sources.py` (`fetch-opm` and the `opm` group), `CLAUDE.md` (one Commands line)
- Create, by `promote` only: `data/annual/raw/opm/bls-onboard-<YYYY-MM>.<ext>`, with catalog and manifest rows
- Test: `tests/test_funding_sources.py`

**Interfaces (pass branch only):**
- Produces, in `scripts/funding_sources.py`: `opm_months(latest: date) -> list[date]`, the Septembers of 2009–2025 that the source covers and `latest` for FY2026. `opm_count(payload: bytes) -> int`, which raises `ValueError` unless the payload names `BUREAU OF LABOR STATISTICS` and gives one count. `fetch_opm(now, staging)`, which stages one response per month in group `opm`, role `opm_headcount`, with the version that OPM labels it in its locator. The request and the payload layout follow Step 1's response.

- [ ] **Step 1: Run the recon (F8).** Ask the owner for a yes to the recon. On the yes only, run a `/tmp` script under `uv run python -I` with the bare User-Agent: read `https://data.opm.gov/get-data/api-access`, then make one request for the on-board count of agency subelement BUREAU OF LABOR STATISTICS for September 2025. The rule passes when one response gives that count, with no API key and without the full employment file. A count of 2,165 matches the review's figure for September 30, 2025, which confirms the concept. Record the request, the outcome and the wall time in the ledger.
- [ ] **Step 2: Fail branch.** Record "OPM left out" and the reason in the ledger, for the findings' Data and sources and Limits (Task 8). Key no `opm_headcount` row, and go to Task 5.
- [ ] **Step 3: Pass branch.** Write `fetch-opm` test first, with synthetic payloads in the layout of Step 1's response: `test_opm_months_are_the_septembers_and_the_latest_month`, `test_opm_count_refuses_a_payload_without_bls`, and `test_fetch_opm_stages_one_response_per_month`. Then ask the owner for a second yes, and run `/usr/bin/time -p uv run python scripts/funding_sources.py fetch-opm`. Then `promote opm`, `annual_sources.py manifest`, the Task 3 Step 7 check, the gate, and `git commit -m "Commit OPM's BLS on-board counts for O11"`.

### Task 5: The keyed table and its checks

**Files:**
- Create: `data/annual/raw/manual/bls-funding-staffing.csv`
- Modify: `src/ces_revisions/annual/funding_tables.py`
- Modify: `data/annual/raw/manifest.csv` (by `scripts/annual_sources.py manifest` only)
- Test: `tests/test_funding_tables.py`

**Interfaces:**
- Consumes: `summary_table` (Task 1). The committed sources (Tasks 3 and 4).
- Produces, in `funding_tables.py`:
  - `FUNDING_TABLE = "manual/bls-funding-staffing.csv"`, and `FUNDING_SCHEMA` with the F7 columns: `fiscal_year` `pl.Int32`, `value` `pl.Int64`, `reference_date` `pl.Date`, the rest `pl.String`.
  - `SERIES: dict[str, tuple[str, str]]`, each series' `(unit, concept)`: `budget_authority_total`, `budget_authority_general_funds`, `budget_authority_trust_funds` and `budget_authority_one_time` as `("thousands of dollars", "nominal")`; `fte_actual` and `brief_fte_actual` as `("FTE", "actual FTE")`; `fte_plan` as `("FTE", "planned FTE")`; `opm_headcount` as `("persons", "headcount")`.
  - `STANDING_LINES = frozenset({"Labor Force Statistics", "Employment and Unemployment Statistics", "Labor Market Information", "Prices and Cost of Living", "Compensation and Working Conditions", "Productivity and Technology", "Executive Direction and Staff Services"})` and `ONE_TIME_LINES = frozenset({"Headquarters Relocation"})`.
  - `read_funding_table(raw_dir: Path = raw.RAW_DIR) -> pl.DataFrame`: the table with `FUNDING_SCHEMA`, sorted by `series` and `fiscal_year`. `ValueError` on a repeated `fiscal_year` and `series`, a series not in `SERIES`, or a unit or concept that differs from `SERIES`.
- The rows: each budget series for FY2009–FY2026 (72 rows), `fte_actual` for FY2009–FY2025 (17), `fte_plan` for FY2026 (1), `brief_fte_actual` for FY2009–FY2016 (8), and `opm_headcount` as Task 4 allows. Fiscal year `f` comes from the first column of `dol/cbj-<f + 2>-v3-01.pdf`. FY2026 comes from column 1 of `dol/cbj-2027-v3-01.pdf`, so a CBJ row's column is `f − (cbj year − 2)`. The one-time amount is the sum of that column's `ONE_TIME_LINES` lines, or 0. A locator names the table, the column heading and the line, for example `Summary Budget Authority and FTE by Activity, FY 2022 Revised Enacted, Total`, and the brief's locator names its slide title and column.

- [ ] **Step 1: Write the failing tests** in `tests/test_funding_tables.py`, on the committed table unless named synthetic:
  - `test_one_row_per_year_and_series`: the row counts and fiscal-year ranges above
  - `test_cbj_values_tie_to_their_summary_table`: for each row from a CBJ, with `table = summary_table(<the PDF's .txt>)` and its column, `table.years[column] == fiscal_year`. Total and FTE rows equal `table.cell("Total", column=column)`. The fund rows equal `cell(<fund line>, parent="Total", column=column)`. The one-time row equals the sum over its `ONE_TIME_LINES` lines, or 0.
  - `test_general_and_trust_funds_add_to_the_total`: in each fiscal year
  - `test_every_activity_line_is_listed`: in each CBJ the table cites, each line whose `parent` is its own label, `Total` aside, is in `STANDING_LINES | ONE_TIME_LINES`. The failure message names the CBJ file and the line (Review Focus 4).
  - `test_fte_actual_is_the_actual_concept`: for each `fte_actual` row, either its CBJ's `notes` hold `FY <f> reflects actual FTE` (any spacing and case) and, for FY2009–FY2016, the value equals `brief_fte_actual`; or no such note exists, the year is in FY2009–FY2016, and the value equals `brief_fte_actual`. A row with an `owner_ruling` passes.
  - `test_brief_values_appear_in_the_brief`: each `brief_fte_actual` value, with or without its thousands comma, is in the brief's text
  - `test_basis_markers_carry_a_note_and_a_ruling`: where a first-column Total line has a `marker`, or the header lines hold "comparable" or "restated" in any case, that year's Total and FTE rows have a `note`. Every row with a `note` has an `owner_ruling`, and every `owner_ruling` ends `(owner, YYYY-MM-DD)`.
  - `test_the_review_figures`: independent of the reader, from `docs/ces-revisions-review.md`. FY2009: Total 597,182 and trust funds 78,264, and brief actual FTE 2,210. FY2022: actual FTE 1,951, Total 687,952 and one-time 28,470. FY2023 actual FTE 2,036. FY2024 actual FTE 2,062. FY2025: actual FTE 2,019 and Total 703,952. FY2026: plan 1,773 and Total 708,500.
  - `test_read_funding_table_refuses_bad_rows`: synthetic `tmp_path` tables with a repeated year and series, an unknown series, and a wrong unit each raise `ValueError`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_funding_tables.py -q`. Expected: `ImportError` for `read_funding_table`.
- [ ] **Step 3: Implement** the constants and `read_funding_table`.
- [ ] **Step 4: Key the table.** Read each CBJ's summary table in its text file, and its PDF page where the layout is unclear, and type the values (Global Constraints, Keying). For each first column, read the column heading, the lines and the footnotes below the table, and copy the text of any basis marker into `note`. Read the brief for FY2009–FY2016 actual FTE. Add the OPM rows if Task 4 passed. Record the keying time in the ledger.
- [ ] **Step 5: Run the tests.** Run: the Step 2 command. A CBJ that `summary_table` cannot read (a table across a page break, an older layout) gets a failing test on its committed text first, then the reader fix. List every other failure as a stop, and fix none of them.
- [ ] **Step 6: Ask the batched question (F2).** If Step 5 left stops, put one question to the owner for each kind (an unlisted line, the FTE concept, a definition break, one-time money inside a standing line). Each question lists every CBJ, year and line it covers, quotes the source text, and gives a recommended default. Record each ruling where it applies: an activity line in `STANDING_LINES` or `ONE_TIME_LINES` with a comment citing the ruling, otherwise in the rows' `owner_ruling`. Task 6 may run while the answer is pending. Task 7 may not start before every stop is ruled.
- [ ] **Step 7: Rehash.** Run: `uv run python scripts/annual_sources.py manifest`. Expected: `git diff data/annual/raw/manifest.csv` adds the table's row only. Then repeat Task 3 Step 7. Expected: the same result.
- [ ] **Step 8: Verify.** Run: `uv run pytest -m "not slow and not network" -q`. Expected: all pass.
- [ ] **Step 9: Commit.** `git add data/annual/raw src/ces_revisions/annual/funding_tables.py tests/test_funding_tables.py`, then `git commit -m "Key BLS funding and staffing for FY2009-2026 and check it against the sources (O11)"`.
- [ ] **Step 10: Prove the tests.** Run executing-plans' mutant gate on this task's code diff. Expected: every mutant killed, or each survivor ruled in the ledger.

### Task 6: Fiscal years, the deflator and real budget authority

**Files:**
- Modify: `src/ces_revisions/annual/funding_tables.py` (`read_gdpdef`)
- Create: `src/ces_revisions/funding_staffing.py`
- Test: `tests/test_funding_tables.py`, `tests/test_funding_staffing.py`

**Interfaces:**
- Consumes: the committed `GDPDEF` vintage (Task 3). `read_funding_table` (Task 5), in tests on synthetic frames of its schema.
- Produces, in `funding_tables.py`: `read_gdpdef(raw_dir: Path = raw.RAW_DIR) -> tuple[date, pl.DataFrame]`: the vintage date from the one `alfred/gdpdef-<YYYY-MM-DD>.csv` file name, and a frame of `quarter_start` (`pl.Date`) and `gdpdef` (`pl.Float64`), sorted. `ValueError` unless exactly one such file exists.
- Produces, in `ces_revisions.funding_staffing`:
  - `BASE_YEAR = 2025`.
  - `fiscal_year(column: str | pl.Expr) -> pl.Expr`: the fiscal year of a date, as `pl.Int32`, by the Global Constraints rule.
  - `fiscal_year_deflator(gdpdef: pl.DataFrame) -> pl.DataFrame`: `fiscal_year`, `quarters` (count) and `deflator`, the mean of its quarters when there are 4, else null. A quarter belongs to the fiscal year of its start month.
  - `real_budget_authority(table: pl.DataFrame, deflators: pl.DataFrame) -> pl.DataFrame`: per fiscal year, `budget_authority_total`, `budget_authority_one_time` and `real_budget_authority` (`pl.Float64`, in FY2025 thousands of dollars), null when that year's deflator is null. `ValueError` when the FY2025 deflator is null.

- [ ] **Step 1: Write the failing tests.**
  - `test_fiscal_year_follows_the_reference_month`: 2008-10 gives 2009, 2009-09 gives 2009, 2025-09 gives 2025, 2025-10 gives 2026, and 2026-07 gives 2026
  - `test_the_deflator_averages_q4_of_the_year_before_and_q1_to_q3`: quarters starting 2024-07, 2024-10, 2025-01, 2025-04 and 2025-07 with values 9, 1, 2, 3 and 4 give FY2025 a deflator of 2.5, and the 2024-07 quarter goes to FY2024
  - `test_a_year_missing_a_quarter_has_no_deflator`: FY2026 with three quarters has `quarters == 3` and a null deflator (Review Focus 2)
  - `test_real_budget_authority_subtracts_one_time_and_rebases_to_fy2025`: total 700, one-time 100, $`d_f`$ 2 and $`d_{2025}`$ 4 give 1200.0. In FY2025 the real value equals total less one-time. A year with a null deflator keeps its total and has a null real value.
  - In `tests/test_funding_tables.py`, `test_read_gdpdef_reads_the_committed_vintage`: the date equals the one in the `price_deflator` catalog row, the first quarter is 1947-01-01, and the last is 2026-04-01 or later
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_funding_staffing.py tests/test_funding_tables.py -q`. Expected: `ModuleNotFoundError: No module named 'ces_revisions.funding_staffing'`.
- [ ] **Step 3: Implement** the functions. The module docstring names O11, the spec and the fit, as `ces_revisions.collection_window`'s does.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: all pass.
- [ ] **Step 5: Commit.** `git add` the four files, then `git commit -m "Map months to fiscal years and deflate BLS budget authority (O11)"`.
- [ ] **Step 6: Prove the tests.** Run executing-plans' mutant gate on this task's diff.

### Task 7: The samples, the fits and the tables

**Files:**
- Modify: `src/ces_revisions/funding_staffing.py`
- Test: `tests/test_funding_staffing.py`

**Interfaces:**
- Consumes: `absolute_revisions`'s columns, `ols_clustered` and `CONFIDENCE` from `ces_revisions.collection_window`. Task 6's functions. `read_funding_table`'s schema.
- Produces, in `ces_revisions.funding_staffing`:
  - `FIRST_YEAR = 2009`, `LAST_FIT_YEAR = 2025`, `PLAN_YEAR = 2026`, `SAMPLES = ("main", "fy2026", "pandemic", "lapse")`, `COVARIATES = ("budget", "fte")`, `FORMS = ("no_trend", "trend")`, `REVISION_COLUMNS = ("NSA_F_S", "NSA_F_T", "SA_F_S", "SA_F_T")`.
  - `revision_sample(revisions: pl.DataFrame, days: pl.DataFrame) -> pl.DataFrame`: the rows of `absolute_revisions` with fiscal year 2009 to 2026, adding `fiscal_year`, `release_date` (joined from `days`) and `sample`: `lapse` when `lapse_delayed`, else `pandemic` when `period` is `pandemic`, else `fy2026` in FY2026, else `main`.
  - `covariates(table: pl.DataFrame, deflators: pl.DataFrame) -> pl.DataFrame`: one row per fiscal year 2009–2026, with `budget_authority_total`, `budget_authority_one_time`, `real_budget_authority`, `fte` (`fte_actual`, or `fte_plan` for FY2026), `fte_is_plan`, `opm_headcount` and `opm_date` (null when not keyed), `deflator_quarters`, `log_real_budget` and `fte_hundreds`.
  - `fiscal_year_table(sample: pl.DataFrame, covariates: pl.DataFrame) -> pl.DataFrame`: always 18 rows, FY2009–FY2026. `n_f_s` and `n_f_t` count the year's `main` or `fy2026` months, and `ValueError` names the year when NSA and SA counts differ. The four `REVISION_COLUMNS` hold mean absolute revisions, null for a year with no months. The covariate columns follow.
  - `funding_fits(sample: pl.DataFrame, covariates: pl.DataFrame) -> pl.DataFrame`: 16 rows, sorted by `seasonal_status`, `transition`, `covariate` and `form`, with `n`, `clusters`, `coefficient`, `se`, `contrast`, `contrast_se`, `contrast_low` and `contrast_high`. `main` rows only. The contrast is $`\theta \ln 0.9`$ on `log_real_budget`, or $`-\theta`$ on `fte_hundreds`, and `contrast_se` is the contrast factor's absolute value times `se`.
  - One private fit helper. Its design columns are the intercept, the covariate, `five_week`, a dummy for each calendar month after January, and, in the trend form, the fiscal year last. It clusters by fiscal year and takes the t quantile on clusters − 1.
  - `leave_one_year_out(sample: pl.DataFrame, covariates: pl.DataFrame) -> pl.DataFrame`: 16 rows with the fit keys, `contrast` (the full fit's), `min_contrast`, `max_contrast`, `influential_year` (F10) and `influential_contrast`, over refits that each drop one fiscal year of the main sample.
  - `ruled_out(fits: pl.DataFrame) -> pl.DataFrame`: per `seasonal_status`, `transition` and `covariate`, `low` (the lower of the two forms' lower limits) and `high` (the higher of their upper limits).
  - `pandemic_table(sample: pl.DataFrame) -> pl.DataFrame`: per fiscal year of the `pandemic` rows, `months` and the four `REVISION_COLUMNS` means.
  - `lapse_table(sample: pl.DataFrame) -> pl.DataFrame`: per `lapse` month, `reference_month`, `fiscal_year`, `release_date` and the four `REVISION_COLUMNS` absolute revisions, null where none.

- [ ] **Step 1: Write the failing tests** in `tests/test_funding_staffing.py`, on small synthetic frames built in the file:
  - `test_revision_sample_labels_each_month`: 2008-09 is left out, 2008-10 is `main` in FY2009, 2013-09 is `lapse`, 2020-05 is `pandemic`, 2025-12 is `fy2026`, and 2026-01 is `lapse`
  - `test_fits_recover_an_exact_slope`: 4 fiscal years of 12 months each, a covariate that is not linear in the year, and an outcome of 3 × covariate + a calendar-month effect + 5 × `five_week`, give `coefficient == pytest.approx(3)`, `n == 48` and `clusters == 4` in both forms. The trend form also recovers 3 when 2 × fiscal year is added to the outcome.
  - `test_contrasts_and_their_intervals`: the budget contrast is the coefficient times `math.log(0.9)`, the FTE contrast is minus the coefficient, and the half-width is `stats.t.ppf(0.975, clusters - 1) × contrast_se`
  - `test_only_main_rows_enter_the_fits`: added `fy2026`, `pandemic` and `lapse` rows with outlying revisions leave every fit unchanged
  - `test_leave_one_year_out_names_the_year_that_moves_the_contrast_most`: with one outlying fiscal year, `influential_year` is that year, `min_contrast <= max_contrast`, and every refit has `clusters − 1` clusters
  - `test_ruled_out_spans_both_intervals`: intervals (−3, 1) and (−1, 4) give `low == -3` and `high == 4`
  - `test_a_year_without_months_keeps_its_row`: FY2021 has `n_f_s == 0`, `n_f_t == 0` and null means, and the table has 18 rows (Review Focus 1)
  - `test_fy2026_takes_the_plan_and_is_marked`: FY2026's `fte` is the `fte_plan` value and `fte_is_plan` is true, and no other year is a plan
  - `test_lapse_table_keeps_missing_revisions_null`: a lapse month with no F→T row has null `NSA_F_T` and `SA_F_T`
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_funding_staffing.py -q`. Expected: `ImportError` for `revision_sample`.
- [ ] **Step 3: Implement** the functions as the Interfaces state.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: all pass.
- [ ] **Step 5: Commit.** `git add src/ces_revisions/funding_staffing.py tests/test_funding_staffing.py`, then `git commit -m "Add the O11 samples, fits and tables"`.
- [ ] **Step 6: Prove the tests.** Run executing-plans' mutant gate on this task's diff.

### Task 8: The report command and the findings file

**Files:**
- Create: `scripts/funding_staffing.py`
- Create: `specs/ces-revision-drivers-stage-4-findings.md`
- Modify: `CLAUDE.md` (one Commands line, and one layout bullet for O11's modules, scripts, staging folder, keyed table and checks)
- Test: `tests/test_funding_staffing.py`

**Interfaces:**
- Consumes: Tasks 5 to 7. `replace_generated` from `archive_inventory`. `PANEL_DIR` and `ROOT` from `ces_revisions.vintages.raw`.
- Produces, in `scripts/funding_staffing.py`: `FINDINGS = ROOT / "specs" / "ces-revision-drivers-stage-4-findings.md"`, `INPUTS = ("panel.parquet", "collection_days.parquet")`, a renderer for each block, and `report() -> int` and `main(argv: list[str] | None = None) -> int` with the one command `report`. With an input missing, it prints a line naming `uv run python scripts/vintage_sources.py build` to stderr and returns 1. It writes each block with `replace_generated(document, name, f"\n{body}\n\n")`, and prints `report: <n> revisions, <s> s`. The blocks:
  - `o11-by-year`: fiscal year, n F→S, n F→T, the four means, budget authority, one-time, real budget authority (a year without a deflator as `– (<n> quarters)`), actual FTE (FY2026 as `1,773 (plan)`), and OPM headcount with its date, `–` when absent
  - `o11-sources`: fiscal year, series, value, unit, source document and locator, one row per keyed row
  - `o11-fits`: F9's 8 rows
  - `o11-leave-one-out`: status, transition, covariate, form, full contrast, smallest, largest, and the year that moves it most with its contrast
  - `o11-pandemic`: fiscal year, months and the four means
  - `o11-lapse`: reference month, fiscal year, release date and the four absolute revisions

  Means, contrasts, standard errors and limits to one decimal, in thousands of jobs. Dollars in thousands with commas, and FTE as integers with commas. A missing value prints `–`.

- [ ] **Step 1: Write the failing tests** in `tests/test_funding_staffing.py`, importing the script as `import funding_staffing as report_script`. The report tests write `vintage_data.table_changes()` and `vintage_data.collection_days()` to a patched `PANEL_DIR`, and patch `FINDINGS` to a `tmp_path` file with the six marker pairs, as Stage 1's `test_report_fills_the_four_generated_blocks` does:
  - `test_report_refuses_without_a_built_panel`: exit 1, and stderr names `scripts/vintage_sources.py build`
  - `test_report_fills_every_generated_block`: exit 0, each block equals its renderer's output, the text outside the blocks is unchanged, and stdout starts `report: `
  - `test_by_year_rows_for_fy2021_and_fy2026`: the FY2021 row starts `| 2021 | 0 | 0 | – | – | – | – |`, and the FY2026 row starts `| 2026 | 7 | 6 |` and holds `1,773 (plan)`
  - `test_without_opm_the_column_shows_a_dash`: with no `opm_headcount` rows, every row's OPM cell is `–` (Review Focus 3)
  - `test_a_year_without_a_deflator_is_marked`: rendered from covariates in which FY2026 has a null real value and `deflator_quarters == 3`, its real budget cell is `– (3 quarters)` (Review Focus 2)
- [ ] **Step 2: Run the tests to verify they fail.** Run: `uv run pytest tests/test_funding_staffing.py -q`. Expected: `ModuleNotFoundError: No module named 'funding_staffing'`.
- [ ] **Step 3: Implement** the script after the shape of `scripts/collection_window.py`. Write the findings skeleton: an opening paragraph naming O11, the roadmap's Stage 4, plan 17 and the command, and saying not to edit the tables by hand. Then the sections Question, Answer, Data and sources, Results (the six blocks, each under its own `###` heading, with the fit's formula in a ```` ```math ```` fence above `o11-fits`), Limits and Measured cost. Add the CLAUDE.md lines, the Commands line reading `uv run python scripts/funding_staffing.py report  # rewrite the O11 findings tables from data/panel/ and the keyed funding table (offline; run the vintage build first)`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: the Step 2 command. Expected: all pass.
- [ ] **Step 5: Commit.** `git add scripts/funding_staffing.py specs/ces-revision-drivers-stage-4-findings.md CLAUDE.md tests/test_funding_staffing.py`, then `git commit -m "Add the O11 report command and the findings skeleton"`.
- [ ] **Step 6: Make the real run (F11).** Run: `uv sync`, `rm -rf data/panel`, `/usr/bin/time -p uv run python scripts/vintage_sources.py build`, then `/usr/bin/time -p uv run python scripts/funding_staffing.py report`. Expected: the counts of F12 in the by-year and pandemic blocks, 16 rows in the lapse block, and G = 16 in every fit. Record both wall times.
- [ ] **Step 7: Cross-check the fits.** In `/tmp`, refit the 16 fits with `uv run --with statsmodels python` and `OLS(...).fit(cov_type="cluster", cov_kwds={"groups": fiscal_year})`. Expected: coefficients and standard errors agree to 1e-6. Pin the reference values on `vintage_data` and the committed sources as `test_fits_match_an_independent_statsmodels_fit`, which needs no statsmodels at run time.
- [ ] **Step 8: Write the findings prose** from the generated numbers.
  - Answer: for each series and covariate, the no-trend and trend intervals side by side, and what the data rule out, quoted from the span column (spec § Fit). Say that the association is not causal and that the 2025 cuts are not settled (D1). Give the leave-one-year-out result as fragility, not as a bound.
  - Data and sources: every document keyed, the `GDPDEF` vintage date, the OPM outcome with its reason, the fiscal-year rule and that September's collection runs into the next fiscal year, the samples, the one-time subtraction as a deviation from the review's Total (spec § Real budget authority), and each owner ruling from Task 5.
  - Limits: at least the ten items of spec § Findings file and report command.
  - Measured cost: the two wall times of Step 6, each network run's wall time from Tasks 3 and 4, the keying time, and the sessions used.
- [ ] **Step 9: Verify.** Run: `uv run pytest -m "not slow and not network" -q`, `uv run ruff check` and `uv run ruff format --check`. Expected: all pass. Rerun the report, and check that `shasum` of the findings file is unchanged, which shows the blocks are stable.
- [ ] **Step 10: Commit.** `git add` the findings file and the test file, then `git commit -m "Report revision size against BLS funding and staffing (O11)"`.
- [ ] **Step 11: Prove the tests.** Run executing-plans' mutant gate on this task's code diff.

## Completion

After Task 8, run executing-plans' final whole-branch review, then writing-plans' Plan Completion Protocol. Its retire commit moves the spec to `specs/completed/`, so that commit must also re-point `specs/deferred_items.md:168`, the only other reference to the spec's live path (brief). The ticking pass leaves the parked Stage 13 item open: its `Done when` also waits on O7 and O16. Step 3a ticks Stage 4 in the roadmap and writes the Handoff section: Stage 8 consumes the findings file, and the keyed table feeds O12, O18 and O7 when the roadmap stages them.
