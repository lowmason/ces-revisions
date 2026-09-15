# Stage 3 — Vintage Panel, Stage Labels, and Same-Release Differencing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revisions-roadmap.md, Stage 3 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build Req 1's vintage panel for total nonfarm and the eleven supersectors from BLS's vintage files, with the total-nonfarm leg from January 1979 read from BLS's revision table and the Philadelphia Fed's EMPLOY vintages; label every reference month's F, S, T, B, and M vintages under Req 2; date every Employment Situation release from January 1979; and ship the same-release differencing operator that reproduces BLS's revision table, with Req 9's accounting decomposition.

**Architecture:** Stage 3's source files are committed under `data/raw/` with a SHA-256 manifest. `scripts/vintage_sources.py fetch` (network) refreshes them, and a new subpackage, `ces_revisions.vintages`, rebuilds every derived artifact from them offline in a few seconds, writing parquet to the gitignored `data/panel/`. The subpackage reads each source cell by cell into an immutable raw-value table, derives a long panel whose rows each cite their raw cell and a named transformation, labels stages on the release clock the vintage files' own rows define, and differences levels only within one release file. Tests build the artifacts once per session from the committed sources, so every exit criterion runs in the default tier without the network.

**Tech Stack:** Python 3.14; Polars 1.44.2; fastexcel, a new runtime dependency and Polars' calamine Excel engine, whose `cp310-abi3` wheels import on CPython 3.14; the standard library's `zipfile`, `csv`, `html`, `fractions`, and `zoneinfo`; pytest 9.1.1; ruff 0.16.7; and `pdftotext` (poppler), for the fetch step only.

**Source:** [`specs/ces-revisions.md`](../ces-revisions.md) Req 1, Req 2, Req 4, Req 5, Req 9 (the accounting identity), Req 14 (the composite 2003 regime), and Verification bullet 1; [`specs/ces-revisions-roadmap.md`](../ces-revisions-roadmap.md) Stage 3; the pre-May-1999 item of [`specs/deferred_items.md`](../deferred_items.md).

**Retirement:** When this plan retires to `specs/plans/completed/`, `specs/ces-revisions.md` does **not** retire with it. The spec, roadmap, prompt, and research drafts retire together under the roadmap's Completion section. At completion, tick Stage 3 in the roadmap and append this authoritative stamp to the spec's Rollout note:\
`Stage 3: COMPLETE (YYYY-MM-DD) — implemented by plan 5 (specs/plans/completed/5-ces-revisions.md). Next: resume the roadmap.`

## Planning evidence (2026-09-14)

This plan's code and tests were written in a scratch clone of `9f1efe5`, against copies of every source downloaded on 2026-09-14. The nine tasks were then replayed in order on a fresh clone, with the downloaded copies standing in for `fetch`:
- each task's new tests failed first as its steps state, then passed;
- `ruff format --check` and `ruff check` were clean after every task;
- the fast tier grew from 103 to 197 passed, and the slow tier passed with 3.

The live `fetch` step was not run, because its output is this stage's committed evidence. Each source was downloaded and inspected by hand:

- **Vintage files.** [`cesvinall.zip`](https://www.bls.gov/web/empsit/cesvinall.zip) is 3,095,848 bytes, SHA-256 `c8a0d98cecd10d1ed35097d75a9c07344c2f15abb472d9d0164a2a164e4ab9fc`, `Last-Modified: Fri, 06 Mar 2026 12:06:34 GMT`, with 226 CSV members at the archive root (162 MB extracted). Each member `tri_{code}_{NSA|SA}.csv` has columns `year,month`, naming a release by the reference month it first estimates, then one column per reference month, `Jan_39` through `Jan_26` for the twelve members read here. The twelve members for total nonfarm and the supersectors (codes `000000`, `100000`, `200000`, `300000`, `400000`, `500000`, `550000`, `600000`, `650000`, `700000`, `800000`, `900000`) each have 273 release rows, (2003, 5) through (2026, 1), and no decimal point.
- **Annual refresh.** BLS refreshes the vintage files once a year, weeks after the benchmark release. The Internet Archive's copies of `cesvinall.zip` keep one digest from 2025-03-18 to 2026-02-15 and another from 2026-05-02 to 2026-08-19, and all 13 supersector workbooks `cesvin*.xlsx` carry the same March 6, 2026 date. The files' last row is the January 2026 benchmark release, while `docs/inventory/es-vintages.csv` lists releases through August 2026.
- **Append-only history.** A March 2025 copy of all 226 members, with rows through (2025, 1), equals the March 2026 copy on every shared release row: 32,353,818 cells, none different.
- **The lapse placeholder.** Row (2025, 10) exists although no October 2025 release was made. It holds `-1` in `Sep_25` and `Oct_25` in every member (452 cells), and its other values equal those of the November 2025 release, which BLS made on December 16, 2025. BLS's revision table counts estimates on the same clock: October 2025's first estimate is unavailable, its value from December 16 is the second estimate, and August 2025's third estimate is the value in row (2025, 10).
- **Benchmark releases.** From (2004, 1) each January row revises the not seasonally adjusted values of the 21 months from April two years earlier through the prior December, and the seasonally adjusted values of the five prior calendar years. Other rows revise only their two prior months, except (2010, 2), which also revised April to July 2009, and revisions outside those windows occur only in January rows.
- **Stage-rule check.** With B the first January release after the third estimate's release and not-seasonally-adjusted M the second, the value moves between B and M in 1,775 of 1,836 April-to-October unit-months (the next benchmark's wedge), but in only 46 of 756 January-to-March and 35 of 504 November-to-December unit-months, 73 of them in years with a Comments entry.
- **Sums.** The eleven supersectors sum exactly to total nonfarm in every release row for every reference month back to 1939, seasonally adjusted and not: 496,310 cells, largest difference 0.
- **Revision table.** [`cesnaicsrev.htm`](https://www.bls.gov/web/empsit/cesnaicsrev.htm) (401,161 bytes, "Last Modified Date: September 4, 2026") holds two summary tables and one table per year from 1979 to 2026, each with twelve month rows labeled `Jan.`, `Feb.`, `Mar.`, `Apr.`, `May`, `Jun.`, `Jul.`, `Aug.`, `Sep.`, `Oct.`, `Nov.`, `Dec.` and two average rows. Of its 6,912 monthly cells, 6,824 hold an integer, 64 are blank (estimates not yet published), 12 read `-(A)` (September and October 2025), 2 read `-105 (B)` and `679 (B)` (October 2025's second estimates), and 10 read `NA***` or `NA` (revisions of March and April 2003). The page prints zeros for March 2003's third estimate and April 2003's second and third; April 2003's first estimate, whose revisions are also all NA, is a published change. The 2025 averages carry `(c)`.
- **Reconciliation.** Same-release differencing of the total nonfarm members reproduces the table's estimate for every reference month from May 2003 whose release the files hold: 814 cells seasonally adjusted and 813 not. The one exception is November 2003's not seasonally adjusted third estimate, 46 in the table: the same-release change in the February 2004 benchmark release is 147, and 46 is November in that release minus October in the January 2004 release. November's third estimate reconciles in every later year.
- **Averages.** All 564 published yearly averages and all 36 summary averages equal the exact average of the page's own monthly revisions, rounded to an integer over the periods of its footnote **, with 61 yearly averages at an exact half: 37 rounded up and 24 down.
- **One inconsistent cell.** September 2023's seasonally adjusted third-minus-second revision reads −34, while its estimates, 297 and 262, differ by −35.
- **Historical release dates.** BLS's [`histreleasedates.pdf`](https://www.bls.gov/bls/histreleasedates.pdf) (30,838 bytes, `Last-Modified: Thu, 06 Feb 2025 14:46:20 GMT`) tabulates Employment Situation release dates for reference months from June 1957 to December 2000, with footnotes for December 1995 (scheduled January 5, released January 19, 1996, during a lapse in appropriations) and October 1998 (scheduled November 6, released officially at 1:30 PM EST on November 5 after a premature posting). `pdftotext -layout` renders it as one line per year. The Philadelphia Fed's [`Release_-Dates-Employment_Situation-BLS.xls`](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/documentation/Release_-Dates-Employment_Situation-BLS.xls) (34,816 bytes; titled "Release dates for the Employment Situation, 1966-2010", with document metadata reading "Not verified") agrees with the PDF on all 420 overlapping reference months and with `es-vintages.csv` on every overlapping release but December 1999. Its last date is November 2010's release.
- **December 1999.** `es-vintages.csv` dates December 1999's release 2000-01-19, from the date in its archive link, but the archived release's embargo line reads "Friday, January 7, 2000", as do the PDF and the workbook.
- **Reschedules.** BLS's [2025 lapse page](https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm) gives September 2025's release as scheduled for October 3 and made November 20, 2025; October 2025's as scheduled for November 7 and canceled, its establishment data published with November's; November 2025's as scheduled for December 5 and made December 16; December 2025's as unchanged on January 9, 2026; and January 2026's as scheduled for February 6 and made February 11. The [September 2013 release](https://www.bls.gov/news.release/archives/empsit_10222013.htm) says it came "about 2 weeks later than originally scheduled", without the original date, and that October 2013's release moved from November 1 to November 8.
- **RTDSM.** [`employMvMd.xlsx`](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/employMvMd.xlsx) (2,228,010 bytes) has one sheet, `employ`, with header `DATE`, `EMPLOY64M12`, `EMPLOY65M1`, …, `EMPLOY26M8`, observation rows `1939:01` through `2026:07`, and `#N/A` in empty cells. Every vintage month from February 1979 holds the release made in that calendar month, and no calendar month since 1979 has two releases; October 2025 had none, and its vintage repeats September 2025's. Its same-vintage seasonally adjusted changes differ from the revision table in 10 cells between 1981 and 1996, for example March 1981's third estimate (53 in the table, 52 in EMPLOY), besides the 2003 gap and the 2025 lapse, so the 1979 leg's changes come from the table.
- **Comments.** [`cesvin00.xlsx`](https://www.bls.gov/web/empsit/cesvin00.xlsx) (27,361,622 bytes, `Last-Modified: Fri, 06 Mar 2026 12:06:28 GMT`, internal "Last Modified Date: March 6, 2026") has a sheet `Data Usage and Comments` whose table, under a `Publication Date` and `Adjustment` header in row 10, holds 16 entries in rows 11 to 26 describing 13 releases: May 2003, January 2008, 2011, 2012, 2014, 2016, 2018, 2019, 2020, 2023, 2025, and 2026, and the canceled October 2025 release. The March 2025 copy's list differs, so BLS rewrites the sheet rather than appending to it.
- **Headers and hosts.** Only BLS's static files (the ZIP, the workbooks, and the PDF) send `Last-Modified`; the revision table page, the release index, and the Philadelphia Fed files send none. Both hosts served every file to a client identifying the project.
- **Dependencies.** `uv add fastexcel` resolves on Python 3.14, and fastexcel 0.20.2 and 0.21.0 both import on CPython 3.14.0. In Polars 1.44.2, `pl.format` panicked on a String column of a filtered frame where `pl.concat_str` did not, so the code builds that identifier with `pl.concat_str`.
- **Build cost.** The raw-value table has 6,742,659 cells (5,955,768 from the vintage files, 8,100 from the revision table, and 778,791 from EMPLOY) and builds in 2.8 seconds; the panel has 6,342,530 rows; the whole build takes about 6 seconds, and Stage 3's 94 fast tests add about 7 seconds to the fast tier.

## Deviations recorded at planning

1. **The vintage-file frontier.** BLS refreshes `cesvinall.zip` once a year, so its last release row is January 2026 while the release index runs through August 2026. The exit criterion "reproduces every row of the BLS 1979–present revision table" is met by differencing for every estimate whose release the files hold, and inside the table for the rest, where every published revision and average is recomputed from the page's own estimates. Stage labels mark the later releases `beyond_frontier`. No other vintage source is spliced in, because an Employment Situation release's Table B-1 omits the not seasonally adjusted level three months back, so the third estimate's same-release change cannot be formed from it. Absorbing BLS's next refresh is deferred at completion, and the At completion section gives the item's text.
2. **Two cells of the revision table are BLS's own.** November 2003's not seasonally adjusted third estimate is a cross-release change in the table, and September 2023's seasonally adjusted third-minus-second revision disagrees with its estimates by 1. Tests pin both, and accept a yearly average at an exact half rounded either way.
3. **December 1999 in `es-vintages.csv`.** The index takes January 7, 2000, from the release itself, BLS's historical dates, and the Philadelphia Fed copy, and agrees with `es-vintages.csv` on every other release from May 1999. `es-vintages.csv` is Stage 2's generated evidence and is not edited.
4. **The index's reach and its publication dates.** The index covers every reference month from January 1979, discharging the pre-May-1999 deferred item. The roadmap's "publication dates" are read as `published_date`, the day a release's values became public, which differs from `release_date` only for the canceled October 2025 release, and `observable_at`, the instant they did. Preliminary benchmark announcements are separate publications, which Stage 4's benchmark table dates, and the index holds only Employment Situation releases, so Stage 4's exit clause "every reconstruction event and benchmark publication date exists in Stage 3's release-date index" cannot hold for them as written. The roadmap resume at completion re-validates that clause (At completion). `benchmark_release` is null before May 2003, where no Stage 3 source dates benchmarks.
5. **Req 2's B and M counted from T.** Read literally, "B the first release incorporating the final March benchmark" and "M … the vintage of the second annual benchmark after the reference month" would give November's third estimate and December's second, which arrive in the January benchmark release, two stages each, and would make January's first estimate its own B. The labels count benchmark releases after T: B is the first January release after T's release, and not-seasonally-adjusted M the second, so every vintage serves one stage, and B→M carries the next benchmark's wedge for April to October (Planning evidence). A `benchmark_release` flag marks the closing-stage vintages that are benchmark releases. Seasonally adjusted M is the January release six years after the reference year, as Req 2 defines it, and a later revision of any M value sets `revised_after_m`, Req 2's "later reconstructions separately flagged".
6. **The panel's scope.** The panel reads the 12 whole-thousand members Req 5 names; the other 214 stay unread in the committed archive. The 1979 leg keeps the Philadelphia Fed's seasonally adjusted level vintages from February 1979 beside the table's changes.
7. **Extra artifacts.** The roadmap's Produces line names no fetch script, manifest, or hand-keyed table. This plan adds `scripts/vintage_sources.py`, `data/raw/manifest.csv`, the cited `data/raw/manual/es-reschedules.csv`, and the fastexcel dependency.
8. **The nonstandard-release flag.** Req 1's "nonstandard releases flagged from the file comments" reads the Comments sheet of the vintage workbooks, which lists reconstructions and the 2025 lapse by release, and flags those 13 releases. Off-schedule timing is the index's `off_schedule`, which the roadmap requires to stay distinct.

## Execution notes

- **Branch.** Create `stage-3-vintage-panel` from `main` (`9f1efe5` or later) and execute there. Other sessions switch branches in the main checkout without notice, so when executing in a worktree (using-git-worktrees), create the branch inside the worktree and never commit in the main checkout.
- **Network.** Task 1 fetches from www.bls.gov and www.philadelphiafed.org, and Task 1 and Task 9 run network tests.
  - Before any network step, export `BLS_CONTACT_EMAIL`, the contact address your human partner chose; the gitignored `.project.env` holds it, and `export BLS_CONTACT_EMAIL="$(grep '^BLS_CONTACT_EMAIL=' .project.env | cut -d= -f2-)"` sets it. Never write the address into a committed file.
  - The fetch needs `pdftotext`; install it with `brew install poppler` if `command -v pdftotext` prints nothing.
  - Do not fetch on an Employment Situation release day between 8:00 and 9:00 a.m. Eastern, so the revision table and the release index describe the same release.
- **Shell.** The Bash tool runs zsh: never name a variable `status` or `path`, and write guards as `if … then … fi` checks, because `set -e` does not stop a command list there.
- **Commits.** Every commit step first runs `uv run ruff format` and `uv run ruff check`, then stages files by explicit path; the repo root can hold an untracked `.DS_Store`, so never `git add -A`.
- **Stop conditions.** Stop and report rather than improvise when:
  - the fetched `cesvinall.zip` has a `Last-Modified` other than `Fri, 06 Mar 2026 12:06:34 GMT`, or the Comments sheet no longer has 16 entries: BLS has refreshed the vintage files, and the frontier, the comments, and several counts in this plan change;
  - a test pinned to a BLS cell (the November 2003 estimate, the September 2023 revision, the 61 exact halves) fails after a fetch, which means BLS corrected its table;
  - the supersector sums differ from total nonfarm in any cell;
  - any count in a step's Expected output differs from what you observe.

  Never loosen a test to make it pass.

## Global Constraints

- **Python and dependencies:** `requires-python = ">=3.14"`. CLAUDE.md: "confirm Python 3.14 support for any package you add." fastexcel is the only new dependency, with an import test in `tests/test_stack.py`.
- **Req 1, the panel:** "One long Polars table keyed by `(sector, reference_month, release_date, vintage_id, release_stage, seasonal_status, value_thousands, concept_regime)`, with an immutable raw-value table and a separate transformations table so every derived number is reproducible".
- **Req 1, the sources:** "BLS CES vintage files (SA and NSA, supersector detail, every Employment Situation release from the May 2003 publication vintage; nonstandard releases flagged from the file comments); the BLS 1979–present revision table and the Philadelphia Fed RTDSM `EMPLOY` matrix for the total-nonfarm aggregate leg".
- **Req 1, differencing:** "Over-the-month changes are constructed only with a **same-release differencing operator** that subtracts the prior month's value in the same release file; stage labels are never differenced across release files".
- **Req 2:** "`F` first release; `S` second; `T` third/final sample-based; `B` the first release incorporating the **final** March benchmark; `M` a **fixed** mature horizon — for NSA the vintage of the second annual benchmark after the reference month; for SA the first benchmark vintage in which the reference month has left BLS's five-year SA revision window — with later reconstructions separately flagged." "The preliminary August/September benchmark is **not** a sixth vintage". "Reference months whose `M` vintage does not yet exist are right-censored".
- **Req 4:** "Supersector NSA and SA vintages from the May 2003 publication vintage forward, plus the total-nonfarm F/S/T revision series from January 1979 appended as a partially observed upper level with one composite pre/post-May-2003 regime shift in intercept and scale". "Pre-2003 sector vintages are integrated out, never fabricated."
- **Req 5:** "The eleven mutually exclusive BLS supersectors, government included, in one hierarchy".
- **Req 9, the identity:** for same-release monthly changes $`d^N,d^S`$ and implied adjustment $`a=d^N-d^S`$,

  ```math
  r^S_{s,t,j\to j+1}=r^N_{s,t,j\to j+1}-\left(a_{s,t,j+1}-a_{s,t,j}\right),
  \qquad
  \operatorname{Var}(r^S)=\operatorname{Var}(r^N)+\operatorname{Var}(\Delta_j a)
  -2\operatorname{Cov}(r^N,\Delta_j a),
  ```

  "estimated conditionally on sector, stage, and regime".
- **Req 14:** "May 2003 as one composite regime shift, never labeled NAICS, probability-sample, or concurrent-SA separately". The regime values are `pre_2003_05` and `from_2003_05`.
- **Verification bullet 1:** "The same-release differencing operator applied to the Req 1 panel reproduces every row of BLS's 1979–present revision table (SA and NSA, all three pairwise MARs) to publication rounding, including the May 2003 gap handling."
- **Data layout (user decision, 2026-09-14):** commit the fetched sources under `data/raw/` with a SHA-256 manifest; gitignore `data/cache/` and the built `data/panel/`; keep the 27 MB workbook out of git and commit its Comments entries as a CSV beside the workbook's hash; every exit test runs in the default tier without the network, on the Mac and on the VM.
- **Never hand-edit:** `docs/inventory/*.csv` (Stage 2's generated evidence) or `data/raw/`, apart from `data/raw/manual/es-reschedules.csv`, whose rows cite their sources.
- **Tests:** live fetches carry the `network` marker and end-to-end runs the `slow` marker; the fast hermetic tier is `uv run pytest -m "not slow and not network"`.
- **Ruff:** the default rules plus `extend-select = ["I", "B", "UP"]`; `ruff format` also formats Python blocks inside Markdown.
- **Markdown:** CLAUDE.md, the README, and this plan stay GitHub-renderable under CLAUDE.md's conventions.
- **Scope:** Stage 3 does not build the benchmark, birth–death, QCEW revision, or sample tables (Stage 4), the operators (Stage 5), the collection-window covariates (Stage 12), or preliminary benchmark dates (Stage 4), and adds nothing to `docs/ces-revisions-review.md`.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `.gitignore` | Ignore `data/cache/` and `data/panel/` | 1 |
| `pyproject.toml`, `uv.lock` | The fastexcel runtime dependency | 1 |
| `tests/test_stack.py` | fastexcel's Python 3.14 import test | 1 |
| `src/ces_revisions/vintages/__init__.py` | The Stage 3 subpackage | 1 |
| `src/ces_revisions/vintages/raw.py` | Data-layout constants and file hashing; then the source readers and the raw-value table | 1, 2 |
| `data/raw/manual/es-reschedules.csv` | The eight off-schedule releases, each cited | 1 |
| `data/raw/bls/`, `data/raw/philadelphiafed/`, `data/raw/manifest.csv` | The committed sources and their manifest | 1 |
| `scripts/vintage_sources.py` | The `fetch` command; then the `build` command | 1, 9 |
| `tests/test_vintage_sources.py` | Script helpers, the manifest, and the refresh canary | 1 |
| `tests/vintage_data.py` | Session-cached Stage 3 artifacts for tests | 2–8 |
| `tests/fixtures/vintages/cesnaicsrev-excerpt.htm` | An excerpt of the revision table's markup | 2 |
| `tests/test_vintage_raw.py` | Raw-value table tests | 2 |
| `src/ces_revisions/vintages/months.py` | Month arithmetic | 3 |
| `src/ces_revisions/vintages/release_index.py` | The release-date index | 3 |
| `tests/fixtures/vintages/histreleasedates-excerpt.txt` | An excerpt of the historical release dates | 3 |
| `tests/test_release_index.py` | Release-date index tests | 3 |
| `src/ces_revisions/vintages/revision_table.py` | Revision table cells, estimates, and averages | 4 |
| `tests/test_revision_table.py` | Revision table tests | 4 |
| `src/ces_revisions/vintages/panel.py` | Transformations and the three source frames; then the assembled panel | 5, 6 |
| `tests/test_vintage_panel.py` | Source-frame tests; then assembled-panel tests | 5, 6 |
| `src/ces_revisions/vintages/stages.py` | Stage labels | 6 |
| `tests/test_stage_labels.py` | Stage-label tests | 6 |
| `src/ces_revisions/vintages/differencing.py` | Same-release differencing and reconciliation | 7 |
| `tests/test_differencing.py` | Differencing and reconciliation tests | 7 |
| `src/ces_revisions/vintages/accounting.py` | The Req 9 identity and variance decomposition | 8 |
| `tests/test_accounting.py` | Accounting tests | 8 |
| `src/ces_revisions/vintages/build.py` | Build and write every artifact | 9 |
| `tests/test_vintage_build.py` | The slow end-to-end build test | 9 |
| `CLAUDE.md`, `README.md` | Layout, commands, and dependency notes | 9 |

---

### Task 1: Committed sources, their manifest, and the fastexcel dependency

**Files:**
- Modify: `tests/test_stack.py` (`STACK_MODULES`)
- Modify: `pyproject.toml` and `uv.lock`, through `uv add fastexcel`
- Modify: `.gitignore`
- Create: `src/ces_revisions/vintages/__init__.py`
- Create: `src/ces_revisions/vintages/raw.py`
- Create: `data/raw/manual/es-reschedules.csv`
- Create: `scripts/vintage_sources.py`
- Test: `tests/test_vintage_sources.py`
- Create, by running `fetch`: `data/raw/manifest.csv`; `data/raw/bls/cesvinall.zip`, `cesnaicsrev.htm`, `histreleasedates.pdf`, `histreleasedates.txt`, `cesvin00-comments.csv`, and `empsit-releases.csv`; and `data/raw/philadelphiafed/employMvMd.xlsx` and `release-dates-employment-situation.xls`

**Interfaces:**
- Consumes: from `scripts/archive_inventory.py`, `user_agent() -> str` (which reads `BLS_CONTACT_EMAIL`), `RELEASE_INDEX_URL`, `parse_release_index(html: str) -> list[Vintage]`, `select_vintages(vintages, *, now: datetime) -> list[Vintage]`, `vintage_rows(vintages) -> list[dict[str, str]]`, `VINTAGE_COLUMNS`, and `write_csv(path, rows, columns) -> None`.
- Produces:
  - `ces_revisions.vintages.raw`: `ROOT`, `DATA_DIR`, `RAW_DIR`, and `PANEL_DIR` (`Path`); the paths under `RAW_DIR` as `str` constants `VINTAGE_FILES = "bls/cesvinall.zip"`, `REVISION_TABLE = "bls/cesnaicsrev.htm"`, `HISTORICAL_RELEASE_DATES = "bls/histreleasedates.txt"`, `EMPSIT_RELEASES = "bls/empsit-releases.csv"`, `VINTAGE_COMMENTS = "bls/cesvin00-comments.csv"`, `RTDSM_LEVELS = "philadelphiafed/employMvMd.xlsx"`, `RTDSM_RELEASE_DATES = "philadelphiafed/release-dates-employment-situation.xls"`, `RESCHEDULES = "manual/es-reschedules.csv"`, and `MANIFEST = "manifest.csv"`; and `file_sha256(path: Path) -> str`.
  - `scripts/vintage_sources.py`: `Download(file: str, url: str)`, `DOWNLOADS`, `WORKBOOK_URL`, `WORKBOOK_PATH`, `COMMENTS_SHEET`, `COMMENT_COLUMNS`, `MANIFEST_COLUMNS`, `user_agent_for(url: str) -> str`, `download(url: str) -> tuple[bytes, str]`, `comment_rows(rows: list[tuple[str | None, ...]]) -> list[dict[str, str]]`, `extract_comments(workbook: Path) -> list[dict[str, str]]`, `manifest_row(file: str, fetched_at: str, **fields: str) -> dict[str, str]`, `fetch_sources(now: datetime) -> int`, and `main(argv: list[str] | None = None) -> int`. Task 9 adds the `build` command.
  - The committed CSVs: `manifest.csv` (`file,url,sha256,bytes,last_modified,fetched_at,derived_from,derived_from_sha256`, one row per file under `data/raw/`, sorted by `file`); `bls/cesvin00-comments.csv` (`sheet_row,publication_label,adjustment`); `bls/empsit-releases.csv` (`reference_month,release_date,release_url`, the format of `docs/inventory/es-vintages.csv`); and `manual/es-reschedules.csv` (`reference_month,scheduled_date,release_date,release_time_et,status,published_with,citation,note`).

- [ ] **Step 1: Write the failing import test**

In `tests/test_stack.py`, replace the comment above `STACK_MODULES` and the list itself with:

```python
# One importable module per pinned distribution: runtime jax, numpy, numpyro, arviz,
# polars, and fastexcel, the Excel reader behind Stage 3's workbooks; dynamax from the dev
# group, as engine-determination evidence only.
STACK_MODULES = [
    "jax",
    "numpy",
    "numpyro",
    "arviz",
    "polars",
    "fastexcel",
    "dynamax.linear_gaussian_ssm",
]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stack.py -q`
Expected: FAIL. `test_stack_module_imports[fastexcel]` fails with `ModuleNotFoundError: No module named 'fastexcel'`: 1 failed, 8 passed.

- [ ] **Step 3: Add fastexcel**

Run: `uv add fastexcel`, then `uv run pytest tests/test_stack.py -q`
Expected: `pyproject.toml`'s `dependencies` gains a `fastexcel>=` entry (0.21.0 or later), `uv.lock` changes, and the tests PASS: 9 passed.

- [ ] **Step 4: Ignore the rebuilt data**

Append to `.gitignore`, after a blank line:

```gitignore
# Stage 3 data: data/raw/ is committed; the workbook cache and the built panel are rebuilt
data/cache/
data/panel/
```

- [ ] **Step 5: Create the subpackage and its data layout**

Create `src/ces_revisions/vintages/__init__.py`:

```python
"""Roadmap Stage 3's vintage data: sources, release dates, the long panel, stages, revisions."""
```

Create `src/ces_revisions/vintages/raw.py`. Task 2 replaces it with the full module:

```python
"""The committed source files in data/raw/: where each lives, and how its hash is taken."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"

VINTAGE_FILES = "bls/cesvinall.zip"
REVISION_TABLE = "bls/cesnaicsrev.htm"
HISTORICAL_RELEASE_DATES = "bls/histreleasedates.txt"
EMPSIT_RELEASES = "bls/empsit-releases.csv"
VINTAGE_COMMENTS = "bls/cesvin00-comments.csv"
RTDSM_LEVELS = "philadelphiafed/employMvMd.xlsx"
RTDSM_RELEASE_DATES = "philadelphiafed/release-dates-employment-situation.xls"
RESCHEDULES = "manual/es-reschedules.csv"
MANIFEST = "manifest.csv"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 6: Key the off-schedule releases by hand**

Create `data/raw/manual/es-reschedules.csv`. Each row cites the BLS page its dates come from, and Planning evidence quotes each page:

```csv
reference_month,scheduled_date,release_date,release_time_et,status,published_with,citation,note
1995-12,1996-01-05,1996-01-19,08:30,released,,https://www.bls.gov/bls/histreleasedates.pdf,"Footnote 1: delayed two weeks because many federal agencies, BLS included, were closed for lack of appropriations."
1998-10,1998-11-06,1998-11-05,13:30,released,,https://www.bls.gov/bls/histreleasedates.pdf,"Footnote 2: some payroll estimates were released prematurely on the Internet, so the report was released officially at 1:30 PM EST a day early."
2013-09,,2013-10-22,08:30,released,,https://www.bls.gov/news.release/archives/empsit_10222013.htm,"The release says it came about 2 weeks later than originally scheduled because of the partial federal shutdown, without giving the original date."
2013-10,2013-11-01,2013-11-08,08:30,released,,https://www.bls.gov/news.release/archives/empsit_10222013.htm,"The September 2013 release says the October release was originally scheduled for November 1, 2013."
2025-09,2025-10-03,2025-11-20,08:30,released,,https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm,Rescheduled after the 2025 lapse in appropriations.
2025-10,2025-11-07,,,canceled,2025-11,https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm,"Canceled; October's establishment survey data were published with November's."
2025-11,2025-12-05,2025-12-16,08:30,released,,https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm,Rescheduled after the 2025 lapse in appropriations.
2026-01,2026-02-06,2026-02-11,08:30,released,,https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm,Rescheduled after the 2026 lapse in appropriations.
```

- [ ] **Step 7: Write the failing source tests**

Create `tests/test_vintage_sources.py`:

```python
"""scripts/vintage_sources.py and the manifest of the committed sources in data/raw/."""

import csv
import urllib.request

import pytest
import vintage_sources

from ces_revisions.vintages import raw


def manifest() -> list[dict[str, str]]:
    with (raw.RAW_DIR / raw.MANIFEST).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_bls_requests_name_a_contact_and_other_hosts_do_not(monkeypatch):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    assert (
        vintage_sources.user_agent_for("https://www.bls.gov/web/empsit/cesvinall.zip")
        == "ces-revisions/0.1.0 (someone@example.org)"
    )
    assert (
        vintage_sources.user_agent_for("https://www.philadelphiafed.org/data.xlsx")
        == "ces-revisions/0.1.0"
    )
    assert (
        vintage_sources.user_agent_for("https://bls.gov.example.com/")
        == "ces-revisions/0.1.0"
    )


def test_comment_rows_keep_the_entries_below_the_header():
    rows = [
        ("NOTE ON DATA USAGE: the purpose of providing first and third", None),
        (None, None),
        ("Publication Date", "Adjustment"),
        ("June 2003", "With the release of May 2003 data on June 6, 2003, the CES"),
        (None, "Due to the 2025 lapse in appropriations, no estimates for October"),
        (None, None),
    ]
    assert vintage_sources.comment_rows(rows) == [
        {
            "sheet_row": "4",
            "publication_label": "June 2003",
            "adjustment": "With the release of May 2003 data on June 6, 2003, the CES",
        },
        {
            "sheet_row": "5",
            "publication_label": "",
            "adjustment": "Due to the 2025 lapse in appropriations, no estimates for October",
        },
    ]


def test_the_manifest_lists_every_committed_source_with_its_hash_and_size():
    rows = manifest()
    committed = sorted(
        str(path.relative_to(raw.RAW_DIR))
        for path in raw.RAW_DIR.rglob("*")
        if path.is_file() and path.name not in (raw.MANIFEST, ".DS_Store")
    )
    assert [row["file"] for row in rows] == committed
    for row in rows:
        path = raw.RAW_DIR / row["file"]
        assert row["sha256"] == raw.file_sha256(path), row["file"]
        assert row["bytes"] == str(path.stat().st_size), row["file"]


def test_downloads_record_their_origin_and_bls_file_dates():
    rows = {row["file"]: row for row in manifest()}
    for item in vintage_sources.DOWNLOADS:
        assert rows[item.file]["url"] == item.url
        assert rows[item.file]["fetched_at"], item.file
    # BLS dates its static files; its HTML pages and the Philadelphia Fed's files carry no
    # Last-Modified header, so their manifest rows leave it empty.
    for file in ("bls/cesvinall.zip", "bls/histreleasedates.pdf"):
        assert rows[file]["last_modified"].endswith(" GMT"), file
    comments = rows[raw.VINTAGE_COMMENTS]
    assert comments["derived_from"].startswith(vintage_sources.WORKBOOK_URL)
    assert len(comments["derived_from_sha256"]) == 64
    assert rows[raw.RESCHEDULES]["derived_from"].startswith("hand-keyed")


@pytest.mark.network
def test_bls_has_not_refreshed_the_vintage_files_since_the_manifest():
    """BLS refreshes cesvinall.zip about once a year; when this fails, fetch and rebuild."""
    rows = {row["file"]: row for row in manifest()}
    url = "https://www.bls.gov/web/empsit/cesvinall.zip"
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": vintage_sources.user_agent_for(url)}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        last_modified = response.headers["Last-Modified"]
    assert last_modified == rows["bls/cesvinall.zip"]["last_modified"]
```

- [ ] **Step 8: Run the tests to verify they fail**

Run: `uv run pytest tests/test_vintage_sources.py -q -m "not network"`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'vintage_sources'`.

- [ ] **Step 9: Write the fetch script**

Create `scripts/vintage_sources.py`:

```python
"""Fetch roadmap Stage 3's source files into data/raw/, with a manifest of their hashes.

    uv run python scripts/vintage_sources.py fetch   # network: refresh data/raw/ and its manifest

BLS keeps one current copy of each file and overwrites it in place, so the committed files are
the raw archive Req 1 asks for. data/raw/manifest.csv records each file's origin, SHA-256, size,
and Last-Modified header. The 27 MB workbook holding the vintage files' Comments sheet stays in
the gitignored data/cache/; its entries are committed as a CSV beside the workbook's hash.
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import archive_inventory
import fastexcel

from ces_revisions.vintages.raw import (
    DATA_DIR,
    EMPSIT_RELEASES,
    HISTORICAL_RELEASE_DATES,
    MANIFEST,
    RAW_DIR,
    RESCHEDULES,
    VINTAGE_COMMENTS,
    file_sha256,
)

PHILADELPHIA_FED = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "real-time-data/data-files"
)


@dataclass(frozen=True)
class Download:
    file: str  # a path under data/raw/
    url: str


DOWNLOADS = (
    Download("bls/cesvinall.zip", "https://www.bls.gov/web/empsit/cesvinall.zip"),
    Download("bls/cesnaicsrev.htm", "https://www.bls.gov/web/empsit/cesnaicsrev.htm"),
    Download(
        "bls/histreleasedates.pdf", "https://www.bls.gov/bls/histreleasedates.pdf"
    ),
    Download(
        "philadelphiafed/employMvMd.xlsx", f"{PHILADELPHIA_FED}/xlsx/employMvMd.xlsx"
    ),
    Download(
        "philadelphiafed/release-dates-employment-situation.xls",
        f"{PHILADELPHIA_FED}/documentation/Release_-Dates-Employment_Situation-BLS.xls",
    ),
)
WORKBOOK_URL = "https://www.bls.gov/web/empsit/cesvin00.xlsx"
WORKBOOK_PATH = DATA_DIR / "cache" / "cesvin00.xlsx"
COMMENTS_SHEET = "Data Usage and Comments"
COMMENT_COLUMNS = ["sheet_row", "publication_label", "adjustment"]
MANIFEST_COLUMNS = [
    "file",
    "url",
    "sha256",
    "bytes",
    "last_modified",
    "fetched_at",
    "derived_from",
    "derived_from_sha256",
]


def user_agent_for(url: str) -> str:
    """BLS asks automated clients for a contact address; other hosts get none."""
    host = urlsplit(url).hostname or ""
    if host == "bls.gov" or host.endswith(".bls.gov"):
        return archive_inventory.user_agent()
    return "ces-revisions/0.1.0"


def download(url: str) -> tuple[bytes, str]:
    """The payload and its Last-Modified header."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent_for(url)})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def comment_rows(rows: list[tuple[str | None, ...]]) -> list[dict[str, str]]:
    """The entries below the Comments sheet's "Publication Date" header, with sheet rows."""
    header = next(
        index for index, row in enumerate(rows) if row[0] == "Publication Date"
    )
    return [
        {
            "sheet_row": str(index + 1),
            "publication_label": row[0] or "",
            "adjustment": row[1],
        }
        for index, row in enumerate(rows)
        if index > header and row[1]
    ]


def extract_comments(workbook: Path) -> list[dict[str, str]]:
    sheet = fastexcel.read_excel(workbook).load_sheet(
        COMMENTS_SHEET, header_row=None, dtypes="string"
    )
    return comment_rows(sheet.to_polars().rows())


def manifest_row(file: str, fetched_at: str, **fields: str) -> dict[str, str]:
    path = RAW_DIR / file
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        file=file,
        sha256=file_sha256(path),
        bytes=str(path.stat().st_size),
        fetched_at=fetched_at,
        **fields,
    )
    return row


def fetch_sources(now: datetime) -> int:
    """Network step: refresh every source file, its derived text, and the manifest."""
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for item in DOWNLOADS:
        payload, last_modified = download(item.url)
        target = RAW_DIR / item.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        rows.append(
            manifest_row(item.file, stamp, url=item.url, last_modified=last_modified)
        )
    pdf = RAW_DIR / "bls/histreleasedates.pdf"
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(RAW_DIR / HISTORICAL_RELEASE_DATES)],
        check=True,
    )
    rows.append(
        manifest_row(
            HISTORICAL_RELEASE_DATES,
            stamp,
            derived_from="pdftotext -layout bls/histreleasedates.pdf",
            derived_from_sha256=file_sha256(pdf),
        )
    )
    payload, last_modified = download(WORKBOOK_URL)
    WORKBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKBOOK_PATH.write_bytes(payload)
    archive_inventory.write_csv(
        RAW_DIR / VINTAGE_COMMENTS, extract_comments(WORKBOOK_PATH), COMMENT_COLUMNS
    )
    rows.append(
        manifest_row(
            VINTAGE_COMMENTS,
            stamp,
            last_modified=last_modified,
            derived_from=f"{WORKBOOK_URL}, sheet {COMMENTS_SHEET}",
            derived_from_sha256=file_sha256(WORKBOOK_PATH),
        )
    )
    index = download(archive_inventory.RELEASE_INDEX_URL)[0].decode("utf-8", "replace")
    vintages = archive_inventory.select_vintages(
        archive_inventory.parse_release_index(index), now=now
    )
    archive_inventory.write_csv(
        RAW_DIR / EMPSIT_RELEASES,
        archive_inventory.vintage_rows(vintages),
        archive_inventory.VINTAGE_COLUMNS,
    )
    rows.append(
        manifest_row(
            EMPSIT_RELEASES,
            stamp,
            derived_from=archive_inventory.RELEASE_INDEX_URL,
        )
    )
    rows.append(
        manifest_row(
            RESCHEDULES,
            "",
            derived_from="hand-keyed from the citation on each row",
        )
    )
    archive_inventory.write_csv(
        RAW_DIR / MANIFEST, sorted(rows, key=lambda row: row["file"]), MANIFEST_COLUMNS
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch"])
    parser.parse_args(argv)
    return fetch_sources(datetime.now(UTC).replace(microsecond=0))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Run the hermetic tests before fetching**

Run: `uv run pytest tests/test_vintage_sources.py -q -m "not network"`
Expected: 2 failed, 2 passed, 1 deselected. Both helper tests pass, and both manifest tests fail with `FileNotFoundError` for `data/raw/manifest.csv`.

- [ ] **Step 11: Fetch the sources**

```bash
export BLS_CONTACT_EMAIL="$(grep '^BLS_CONTACT_EMAIL=' .project.env | cut -d= -f2-)"
if [ -z "$BLS_CONTACT_EMAIL" ]; then echo "STOP: set BLS_CONTACT_EMAIL first"; fi
if ! command -v pdftotext; then echo "STOP: brew install poppler"; fi
uv run python scripts/vintage_sources.py fetch
grep '^bls/cesvinall.zip,' data/raw/manifest.csv
uv run python -c "import polars as pl; print(pl.read_csv('data/raw/manifest.csv').height, pl.read_csv('data/raw/bls/cesvin00-comments.csv').height)"
tail -1 data/raw/bls/empsit-releases.csv
```

Expected:
- `fetch` exits 0 within a few minutes, and `data/raw/` holds the files listed under **Files**.
- The `cesvinall.zip` row shows `3095848` bytes and `Fri, 06 Mar 2026 12:06:34 GMT`. Any other date means BLS has refreshed the vintage files: stop and report.
- The counts print `9 16`: nine manifest rows, and the 16 Comments entries. Any other number of entries is also a refresh: stop and report.
- The release list's last line is the latest Employment Situation release, `2026-08,2026-09-04,…` at planning, or a later one.

- [ ] **Step 12: Run the source tests with the refresh canary**

Run: `uv run pytest tests/test_vintage_sources.py tests/test_stack.py -q`
Expected: PASS: 14 passed, the network canary included.

- [ ] **Step 13: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: both ruff commands are clean, and the fast tier passes: 108 passed, 5 deselected.

- [ ] **Step 14: Commit**

```bash
uv run ruff format scripts src tests
uv run ruff check scripts src tests
git add .gitignore pyproject.toml uv.lock tests/test_stack.py tests/test_vintage_sources.py scripts/vintage_sources.py src/ces_revisions/vintages/__init__.py src/ces_revisions/vintages/raw.py data/raw
git status --short --ignored data
git commit -m "Commit Stage 3's source files with a manifest, and the script that fetches them"
```

Expected from `git status`: ten files under `data/raw/` staged, `manifest.csv` among them, and `data/cache/` ignored.

---

### Task 2: The immutable raw-value table

**Files:**
- Create: `tests/fixtures/vintages/cesnaicsrev-excerpt.htm`
- Create: `tests/vintage_data.py`
- Test: `tests/test_vintage_raw.py`
- Modify: `src/ces_revisions/vintages/raw.py` (replace)

**Interfaces:**
- Consumes: Task 1's `raw` constants and `file_sha256`, and the committed `data/raw/` files.
- Produces, in `ces_revisions.vintages.raw`:
  - `SECTORS: dict[str, str]`, vintage-file code to two-digit CES supersector code, from `"000000": "00"` (total nonfarm) to `"900000": "90"` (government); `SUPERSECTORS`, the eleven codes other than `"00"`; and `SEASONAL_STATUSES = ("NSA", "SA")`.
  - `ESTIMATE_COLUMNS`, the revision table's twelve column keys `sa_1st`, `sa_2nd`, `sa_3rd`, `sa_2nd_minus_1st`, `sa_3rd_minus_2nd`, and `sa_3rd_minus_1st`, and the same six with `nsa_`; `SUMMARY_COLUMNS`, its six `_minus_` keys; and `TABLE_MONTHS`, the page's month labels to month numbers.
  - `vintage_file_members() -> list[str]`; `triangle_cells(payload: bytes, member: str) -> pl.DataFrame`; `cell_text(fragment: str) -> str`; `revision_table_cells(page: str) -> pl.DataFrame`, whose row keys are `YYYY-MM` for months, `YYYY:mean` and `YYYY:mean_absolute` for yearly averages, and `summary:mean:<period>` and `summary:mean_absolute:<period>` for the summaries; `rtdsm_cells(wide: pl.DataFrame) -> pl.DataFrame`; and `read_rtdsm_matrix(raw_dir: Path = RAW_DIR) -> pl.DataFrame`.
  - `raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame`, the raw-value table: `cell_id` (Int64, from 0), `source` (`cesvinall`, `cesnaicsrev`, or `rtdsm_employ`), `file` (such as `bls/cesvinall.zip:tri_000000_NSA.csv`, `bls/cesnaicsrev.htm`, or `philadelphiafed/employMvMd.xlsx:employ`), `row_key`, `column_key`, and `text`, sorted by source, file, row key, and column key.
  - `read_vintage_comments(raw_dir: Path = RAW_DIR) -> pl.DataFrame` and `content_sha256(frame: pl.DataFrame, *, chunk_rows: int = 500_000) -> str`.
- Test helper: `tests/vintage_data.py` with `raw_values()`, built once per session. Tasks 3 to 8 add builders to it.

- [ ] **Step 1: Record the revision table fixture**

Create `tests/fixtures/vintages/cesnaicsrev-excerpt.htm`. It keeps the page's markup for both summary tables, four 2025 rows with the lapse markers and that year's averages, and five 2003 rows with the redesign's zeros and that year's averages:

```html
<!-- Excerpt of https://www.bls.gov/web/empsit/cesnaicsrev.htm as served on 2026-09-14: both
     summary tables, the 2003 table's January to May rows and averages, and the 2025 table's
     August to November rows and averages, with the page's markup. -->
<table id="Summary" class="regular">
<caption><span class="tableTitle">Summary of MEAN revisions between nonfarm payroll employment over-the-month estimates, 1979-present</span></caption>
<thead>
<tr> <th class="stubhead" scope="col" rowspan="2" colspan="1">Time Period**</th> <th scope="col" rowspan="1" colspan="3">Seasonally adjusted</th> <th scope="col" rowspan="1" colspan="3">Not seasonally adjusted</th> </tr>
<tr> <th scope="col" rowspan="1" colspan="1">2nd - 1st</th> <th scope="col" rowspan="1" colspan="1">3rd - 2nd</th> <th scope="col" rowspan="1" colspan="1">3rd - 1st</th> <th scope="col" rowspan="1" colspan="1">2nd - 1st</th> <th scope="col" rowspan="1" colspan="1">3rd - 2nd</th> <th scope="col" rowspan="1" colspan="1">3rd - 1st</th> </tr>
</thead>
<tbody>
<tr> <th scope="row"><p class="sub0">1979 - 2003</p></th> <td>-3</td> <td>17</td> <td>14</td> <td>-4</td> <td>16</td> <td>12</td> </tr>
<tr class="greenbar"> <th scope="row"><p class="sub0">2003 - present</p></th> <td>7</td> <td>0</td> <td>7</td> <td>10</td> <td>-2</td> <td>8</td> </tr>
<tr> <th scope="row"><p class="sub0">Total All Periods</p></th> <td>2</td> <td>9</td> <td>11</td> <td>3</td> <td>7</td> <td>10</td> </tr>
</tbody>
</table>
<table id="SummaryAbsolute" class="regular">
<caption><span class="tableTitle">Summary of ABSOLUTE MEAN revisions between nonfarm payroll employment over-the-month estimates, 1979-present</span></caption>
<thead>
<tr> <th class="stubhead" scope="col" rowspan="2" colspan="1">Time Period**</th> <th scope="col" rowspan="1" colspan="3">Seasonally adjusted</th> <th scope="col" rowspan="1" colspan="3">Not seasonally adjusted</th> </tr>
<tr> <th scope="col" rowspan="1" colspan="1">2nd - 1st</th> <th scope="col" rowspan="1" colspan="1">3rd - 2nd</th> <th scope="col" rowspan="1" colspan="1">3rd - 1st</th> <th scope="col" rowspan="1" colspan="1">2nd - 1st</th> <th scope="col" rowspan="1" colspan="1">3rd - 2nd</th> <th scope="col" rowspan="1" colspan="1">3rd - 1st</th> </tr>
</thead>
<tbody>
<tr> <th scope="row"><p class="sub0">1979 - 2003</p></th> <td>48</td> <td>29</td> <td>61</td> <td>46</td> <td>53</td> <td>83</td> </tr>
<tr class="greenbar"> <th scope="row"><p class="sub0">2003 - present</p></th> <td>33</td> <td>34</td> <td>51</td> <td>46</td> <td>18</td> <td>53</td> </tr>
<tr> <th scope="row"><p class="sub0">Total All Periods</p></th> <td>41</td> <td>31</td> <td>57</td> <td>46</td> <td>36</td> <td>68</td> </tr>
</tbody>
</table>
<table id="2025" class="regular" cellspacing="0" cellpadding="0" xborder="1">
<caption><span class="tableTitle">Nonfarm Payroll Employment: Revisions between over-the-month estimates, 2025</span></caption>
<thead>
<tr> <th class="stubhead" scope="col" rowspan="3" colspan="1">Month</th> <th scope="col" rowspan="3" colspan="1">Year</th> <th scope="col" rowspan="1" colspan="6">Seasonally adjusted</th> <th scope="col" rowspan="1" colspan="6">Not seasonally adjusted</th> </tr>
<tr> <th scope="col" rowspan="1" colspan="3">Over-the-month change</th> <th scope="col" rowspan="1" colspan="3">Revision* in over-the-month change</th> <th scope="col" rowspan="1" colspan="3">Over-the-month change</th> <th scope="col" rowspan="1" colspan="3">Revision* in over-the-month change</th> </tr>
<tr> <th>1st</th> <th>2nd</th> <th>3rd</th> <th>2nd - 1st</th> <th>3rd - 2nd</th> <th>3rd - 1st</th> <th>1st</th> <th>2nd</th> <th>3rd</th> <th>2nd - 1st</th> <th>3rd - 2nd</th> <th>3rd - 1st</th> </tr>
</thead>
<tbody>
<tr class="greenbar"> <th scope="row"><p class="sub0">Aug.</p></th> <td>2025</td> <td>22</td> <td>-4</td> <td>-26</td> <td>-26</td> <td>-22</td> <td>-48</td> <td>200</td> <td>196</td> <td>185</td> <td>-4</td> <td>-11</td> <td>-15</td> </tr>
<tr> <th scope="row"><p class="sub0">Sep.</p></th> <td>2025</td> <td>119</td> <td>-(A)</td> <td>108</td> <td>-(A)</td> <td>-(A)</td> <td>-11</td> <td>317</td> <td>-(A)</td> <td>328</td> <td>-(A)</td> <td>-(A)</td> <td>11</td> </tr>
<tr class="greenbar"> <th scope="row"><p class="sub0">Oct.</p></th> <td>2025</td> <td>-(A)</td> <td>-105 (B)</td> <td>-173</td> <td>-(A)</td> <td>-68</td> <td>-(A)</td> <td>-(A)</td> <td>679 (B)</td> <td>665</td> <td>-(A)</td> <td>-14</td> <td>-(A)</td> </tr>
<tr> <th scope="row"><p class="sub0">Nov.</p></th> <td>2025</td> <td>64</td> <td>56</td> <td>41</td> <td>-8</td> <td>-15</td> <td>-23</td> <td>241</td> <td>243</td> <td>216</td> <td>2</td> <td>-27</td> <td>-25</td> </tr>
<tr> <th scope="row"><p class="sub0">Mean revision</p></th> <td>2025</td> <td></td> <td></td> <td></td> <td>-28</td> <td>-37</td> <td>-58</td> <td></td> <td></td> <td></td> <td>-32</td> <td>-12</td> <td>-39</td> </tr>
<tr class="greenbar"> <th scope="row"><p class="sub0">Mean absolute revision</p></th> <td>2025</td> <td></td> <td></td> <td></td> <td>31(c) </td> <td>39(c)</td> <td>58(c)</td> <td></td> <td></td> <td></td> <td>42(c)</td> <td>15(c)</td> <td>51(c)</td> </tr>
</tbody>
</table>
<table id="2003" class="regular" cellspacing="0" cellpadding="0" xborder="1">
<caption><span class="tableTitle">Nonfarm Payroll Employment: Revisions between over-the-month estimates, 2003</span></caption>
<thead>
<tr> <th class="stubhead" rowspan="3">Month</th> <th rowspan="3">Year</th> <th colspan="6">Seasonally adjusted</th> <th colspan="6">Not seasonally adjusted</th> </tr>
<tr> <th colspan="3">Over-the-month change</th> <th colspan="3">Revision* in over-the-month change</th> <th colspan="3">Over-the-month change</th> <th colspan="3">Revision* in over-the-month change</th> </tr>
<tr> <th>1st</th> <th>2nd</th> <th>3rd</th> <th>2nd - 1st</th> <th>3rd - 2nd</th> <th>3rd - 1st</th> <th>1st</th> <th>2nd</th> <th>3rd</th> <th>2nd - 1st</th> <th>3rd - 2nd</th> <th>3rd - 1st</th> </tr>
</thead>
<tbody>
<tr> <th id="tnfrevisiontables-separatetabsforweb.xls.r.1" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Jan.</p></th> <td>2003</td> <td>143</td> <td>185</td> <td>203</td> <td>42</td> <td>18</td> <td>60</td> <td>-2722</td> <td>-2684</td> <td>-2669</td> <td>38</td> <td>15</td> <td>53</td> </tr>
<tr class="greenbar"> <th id="tnfrevisiontables-separatetabsforweb.xls.r.2" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Feb.</p></th> <td>2003</td> <td>-308</td> <td>-357</td> <td>-353</td> <td>-49</td> <td>4</td> <td>-45</td> <td>315</td> <td>274</td> <td>276</td> <td>-41</td> <td>2</td> <td>-39</td> </tr>
<tr> <th id="tnfrevisiontables-separatetabsforweb.xls.r.3" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Mar.</p></th> <td>2003</td> <td>-108</td> <td>-124</td> <td>0</td> <td>-16</td> <td>NA***</td> <td>NA***</td> <td>496</td> <td>484</td> <td>0</td> <td>-12</td> <td>NA</td> <td>NA</td> </tr>
<tr class="greenbar"> <th id="tnfrevisiontables-separatetabsforweb.xls.r.4" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Apr.</p></th> <td>2003</td> <td>-48</td> <td>0</td> <td>0</td> <td>NA***</td> <td>NA***</td> <td>NA***</td> <td>714</td> <td>0</td> <td>0</td> <td>NA</td> <td>NA</td> <td>NA</td> </tr>
<tr> <th id="tnfrevisiontables-separatetabsforweb.xls.r.5" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">May</p></th> <td>2003</td> <td>-17</td> <td>-70</td> <td>-76</td> <td>-53</td> <td>-6</td> <td>-59</td> <td>729</td> <td>684</td> <td>680</td> <td>-45</td> <td>-4</td> <td>-49</td> </tr>
<tr> <th id="tnfrevisiontables-separatetabsforweb.xls.r.13" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Mean revision</p></th> <td>2003</td> <td> </td> <td> </td> <td> </td> <td>1</td> <td>4</td> <td>7</td> <td> </td> <td> </td> <td> </td> <td>11</td> <td>-12</td> <td>2</td> </tr>
<tr class="greenbar"> <th id="tnfrevisiontables-separatetabsforweb.xls.r.14" headers="tnfrevisiontables-separatetabsforweb.xls.r"><p class="sub0">Mean absolute revision</p></th> <td>2003</td> <td> </td> <td> </td> <td> </td> <td>33</td> <td>23</td> <td>46</td> <td> </td> <td> </td> <td> </td> <td>38</td> <td>21</td> <td>55</td> </tr>
</tbody>
</table>
```

- [ ] **Step 2: Add the session cache for tests**

Create `tests/vintage_data.py`:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import raw


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_vintage_raw.py`:

```python
"""The committed sources and the immutable raw-value table of Req 1."""

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"


def excerpt_cells() -> dict[tuple[str, str], str]:
    page = (FIXTURES / "cesnaicsrev-excerpt.htm").read_text(encoding="utf-8")
    frame = raw.revision_table_cells(page)
    return {
        (row_key, column_key): text
        for row_key, column_key, text in frame.select(
            "row_key", "column_key", "text"
        ).iter_rows()
    }


def member_rows(member: str) -> list[list[str]]:
    with (
        zipfile.ZipFile(raw.RAW_DIR / raw.VINTAGE_FILES) as archive,
        archive.open(member) as handle,
    ):
        return list(csv.reader(io.TextIOWrapper(handle, encoding="utf-8")))


def test_vintage_file_members_are_total_nonfarm_and_supersectors_both_ways():
    members = raw.vintage_file_members()
    assert len(members) == 24
    assert members[:2] == ["tri_000000_NSA.csv", "tri_000000_SA.csv"]
    assert raw.SUPERSECTORS == (
        "10",
        "20",
        "30",
        "40",
        "50",
        "55",
        "60",
        "65",
        "70",
        "80",
        "90",
    )


def test_triangle_cells_key_each_nonempty_cell_by_release_row_and_month_column():
    payload = b"year,month,Jan_39,Feb_39\n2003,5,29296,-1\n2003,6,,29394\n"
    cells = raw.triangle_cells(payload, "tri_000000_NSA.csv")
    assert cells.select("file", "row_key", "column_key", "text").rows() == [
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-05", "Jan_39", "29296"),
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-05", "Feb_39", "-1"),
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-06", "Feb_39", "29394"),
    ]


def test_revision_table_cells_read_months_averages_and_summaries():
    cells = excerpt_cells()
    assert cells[("2003-03", "sa_3rd")] == "0"
    assert cells[("2003-03", "sa_3rd_minus_2nd")] == "NA***"
    assert cells[("2003-03", "nsa_3rd_minus_2nd")] == "NA"
    assert cells[("2025-10", "sa_2nd")] == "-105 (B)"
    assert cells[("2025-09", "nsa_2nd")] == "-(A)"
    assert cells[("2003:mean_absolute", "nsa_3rd_minus_1st")] == "55"
    assert cells[("2003:mean_absolute", "sa_1st")] == ""
    assert cells[("2025:mean_absolute", "sa_2nd_minus_1st")] == "31(c)"
    assert cells[("summary:mean_absolute:2003 - present", "sa_3rd_minus_1st")] == "51"
    assert cells[("summary:mean:Total All Periods", "nsa_2nd_minus_1st")] == "3"


def test_an_unrecognized_revision_table_row_is_an_error():
    page = (
        "<table><caption>Nonfarm Payroll Employment: Revisions between over-the-month "
        "estimates, 2003</caption><tr><th>Annual</th><td>2003</td>"
        + "<td>1</td>" * 12
        + "</tr></table>"
    )
    with pytest.raises(ValueError, match="unrecognized revision table row 'Annual'"):
        raw.revision_table_cells(page)


def test_rtdsm_cells_keep_every_nonempty_cell_as_text():
    wide = pl.DataFrame(
        {
            "DATE": ["1979:01", "1979:02"],
            "EMPLOY79M2": ["88808", "#N/A"],
            "EMPLOY79M3": ["88810", None],
        }
    )
    assert raw.rtdsm_cells(wide).select("row_key", "column_key", "text").rows() == [
        ("1979:01", "EMPLOY79M2", "88808"),
        ("1979:02", "EMPLOY79M2", "#N/A"),
        ("1979:01", "EMPLOY79M3", "88810"),
    ]


def test_content_sha256_hashes_the_csv_serialization_in_chunks():
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = hashlib.sha256(b"a,b\n1,x\n2,y\n").hexdigest()
    assert raw.content_sha256(frame, chunk_rows=1) == expected


# --- The committed sources ------------------------------------------------------------------


def test_raw_values_hold_every_nonempty_cell_of_the_twelve_vintage_files():
    expected = sum(
        1
        for member in raw.vintage_file_members()
        for row in member_rows(member)[1:]
        for text in row[2:]
        if text
    )
    held = vintage_data.raw_values().filter(pl.col("source") == "cesvinall")
    assert held.height == expected


def test_raw_values_match_a_vintage_file_cell_for_cell():
    member = "tri_900000_SA.csv"
    header, *rows = member_rows(member)
    expected = {
        (f"{row[0]}-{int(row[1]):02d}", column, text)
        for row in rows
        for column, text in zip(header[2:], row[2:], strict=True)
        if text
    }
    held = vintage_data.raw_values().filter(
        pl.col("file") == f"{raw.VINTAGE_FILES}:{member}"
    )
    assert set(held.select("row_key", "column_key", "text").iter_rows()) == expected


def test_the_twelve_vintage_files_hold_whole_thousands():
    texts = vintage_data.raw_values().filter(pl.col("source") == "cesvinall")["text"]
    assert texts.str.contains(r"^-?\d+$").all()


def test_revision_table_cells_cover_every_month_and_average_row():
    cells = vintage_data.raw_values().filter(pl.col("source") == "cesnaicsrev")
    months = cells.filter(pl.col("row_key").str.contains(r"^\d{4}-\d{2}$"))
    years = sorted({key[:4] for key in months["row_key"]})
    assert years[0] == "1979"
    assert months.height == len(years) * 12 * 12
    averages = cells.filter(pl.col("row_key").str.contains(r"^\d{4}:mean"))
    assert averages.height == len(years) * 2 * 12
    assert cells.filter(pl.col("row_key").str.starts_with("summary:")).height == 36


def test_rtdsm_cells_are_keyed_by_observation_month_and_vintage():
    cells = vintage_data.raw_values().filter(pl.col("source") == "rtdsm_employ")
    assert cells["row_key"].str.contains(r"^\d{4}:\d{2}$").all()
    assert cells["column_key"].str.contains(r"^EMPLOY\d{2}M\d{1,2}$").all()


def test_cell_ids_number_the_raw_values_from_zero():
    ids = vintage_data.raw_values()["cell_id"]
    assert (ids.min(), ids.max(), ids.n_unique()) == (0, ids.len() - 1, ids.len())
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_vintage_raw.py -q`
Expected: FAIL. All 12 tests fail with `AttributeError`, such as `module 'ces_revisions.vintages.raw' has no attribute 'vintage_file_members'`, and likewise for `triangle_cells`, `revision_table_cells`, `rtdsm_cells`, `content_sha256`, and `raw_values`: 12 failed.

- [ ] **Step 5: Write the raw-value table**

Replace `src/ces_revisions/vintages/raw.py` with:

```python
"""The committed source files in data/raw/ and the immutable raw-value table read from them.

Req 1 keeps every published number in an immutable raw-value table beside a separate
transformations table. A raw cell is the text one cell of a source file holds, so every panel
value traces to the file, row, and column it was read from.
"""

import hashlib
import html
import re
import zipfile
from pathlib import Path

import fastexcel
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"

VINTAGE_FILES = "bls/cesvinall.zip"
REVISION_TABLE = "bls/cesnaicsrev.htm"
HISTORICAL_RELEASE_DATES = "bls/histreleasedates.txt"
EMPSIT_RELEASES = "bls/empsit-releases.csv"
VINTAGE_COMMENTS = "bls/cesvin00-comments.csv"
RTDSM_LEVELS = "philadelphiafed/employMvMd.xlsx"
RTDSM_RELEASE_DATES = "philadelphiafed/release-dates-employment-situation.xls"
RESCHEDULES = "manual/es-reschedules.csv"
MANIFEST = "manifest.csv"

# Total nonfarm and the eleven supersectors of Req 5, keyed by vintage-file code. These twelve
# members hold whole thousands; the other 214 carry decimals (docs/ces-revisions-review.md,
# Unrounded NSA inputs) and stay unread in the committed archive.
SECTORS = {
    "000000": "00",
    "100000": "10",
    "200000": "20",
    "300000": "30",
    "400000": "40",
    "500000": "50",
    "550000": "55",
    "600000": "60",
    "650000": "65",
    "700000": "70",
    "800000": "80",
    "900000": "90",
}
SUPERSECTORS = tuple(code for code in SECTORS.values() if code != "00")
SEASONAL_STATUSES = ("NSA", "SA")

# The revision table's twelve estimate columns, and the six revision columns of its summaries.
ESTIMATE_COLUMNS = tuple(
    f"{status}_{column}"
    for status in ("sa", "nsa")
    for column in (
        "1st",
        "2nd",
        "3rd",
        "2nd_minus_1st",
        "3rd_minus_2nd",
        "3rd_minus_1st",
    )
)
SUMMARY_COLUMNS = tuple(column for column in ESTIMATE_COLUMNS if "_minus_" in column)
TABLE_MONTHS = {
    "Jan.": 1,
    "Feb.": 2,
    "Mar.": 3,
    "Apr.": 4,
    "May": 5,
    "Jun.": 6,
    "Jul.": 7,
    "Aug.": 8,
    "Sep.": 9,
    "Oct.": 10,
    "Nov.": 11,
    "Dec.": 12,
}
_STATISTIC_ROWS = {"Mean revision": "mean", "Mean absolute revision": "mean_absolute"}
_TABLE = re.compile(r"(?s)<table\b.*?</table>")
_CAPTION = re.compile(r"(?s)<caption\b.*?</caption>")
_ROW = re.compile(r"(?s)<tr\b.*?</tr>")
_CELL = re.compile(r"(?s)<t[hd]\b[^>]*>(.*?)</t[hd]>")
_TAG = re.compile(r"<[^>]+>")
_SUMMARY_PERIOD = re.compile(r"\d{4} - (?:\d{4}|present)|Total All Periods")


def vintage_file_members() -> list[str]:
    return [
        f"tri_{code}_{status}.csv" for code in SECTORS for status in SEASONAL_STATUSES
    ]


def triangle_cells(payload: bytes, member: str) -> pl.DataFrame:
    """Every non-empty cell of one vintage-file member, keyed by release row and month column."""
    wide = pl.read_csv(payload, infer_schema_length=0)
    return (
        wide.unpivot(
            index=["year", "month"], variable_name="column_key", value_name="text"
        )
        .filter(pl.col("text").is_not_null())
        .select(
            source=pl.lit("cesvinall"),
            file=pl.lit(f"{VINTAGE_FILES}:{member}"),
            row_key=pl.format("{}-{}", "year", pl.col("month").str.zfill(2)),
            column_key="column_key",
            text="text",
        )
    )


def cell_text(fragment: str) -> str:
    """An HTML cell's text: tags dropped, entities decoded, and whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", fragment))).strip()


def revision_table_cells(page: str) -> pl.DataFrame:
    """Every data cell of the revision table page: months, yearly averages, and summaries."""
    records = []
    for table in _TABLE.findall(page):
        caption = _CAPTION.search(table)
        title = cell_text(caption.group(0)) if caption else ""
        rows = [
            [cell_text(cell) for cell in _CELL.findall(row)]
            for row in _ROW.findall(table)
        ]
        if "Revisions between over-the-month estimates," in title:
            for cells in rows:
                if len(cells) != 14 or not cells[1].isdigit():
                    continue
                if cells[0] in TABLE_MONTHS:
                    key = f"{cells[1]}-{TABLE_MONTHS[cells[0]]:02d}"
                elif cells[0] in _STATISTIC_ROWS:
                    key = f"{cells[1]}:{_STATISTIC_ROWS[cells[0]]}"
                else:
                    raise ValueError(f"unrecognized revision table row {cells[0]!r}")
                records += [
                    (key, column, text)
                    for column, text in zip(ESTIMATE_COLUMNS, cells[2:], strict=True)
                ]
        elif title.startswith("Summary of"):
            kind = "mean_absolute" if "ABSOLUTE" in title else "mean"
            for cells in rows:
                if len(cells) == 7 and _SUMMARY_PERIOD.fullmatch(cells[0]):
                    records += [
                        (f"summary:{kind}:{cells[0]}", column, text)
                        for column, text in zip(SUMMARY_COLUMNS, cells[1:], strict=True)
                    ]
    frame = pl.DataFrame(
        records, schema=["row_key", "column_key", "text"], orient="row"
    )
    return frame.select(
        source=pl.lit("cesnaicsrev"),
        file=pl.lit(REVISION_TABLE),
        row_key="row_key",
        column_key="column_key",
        text="text",
    )


def rtdsm_cells(wide: pl.DataFrame) -> pl.DataFrame:
    """Every non-empty cell of the EMPLOY matrix: observation rows by vintage columns."""
    return (
        wide.unpivot(index="DATE", variable_name="column_key", value_name="text")
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
        .select(
            source=pl.lit("rtdsm_employ"),
            file=pl.lit(f"{RTDSM_LEVELS}:employ"),
            row_key="DATE",
            column_key="column_key",
            text="text",
        )
    )


def read_rtdsm_matrix(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    sheet = fastexcel.read_excel(raw_dir / RTDSM_LEVELS).load_sheet(
        "employ", header_row=0, dtypes="string"
    )
    return sheet.to_polars()


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """The immutable raw-value table: every cell Stage 3 reads, each with a stable cell_id."""
    with zipfile.ZipFile(raw_dir / VINTAGE_FILES) as archive:
        parts = [
            triangle_cells(archive.read(member), member)
            for member in vintage_file_members()
        ]
    parts.append(
        revision_table_cells((raw_dir / REVISION_TABLE).read_text(encoding="utf-8"))
    )
    parts.append(rtdsm_cells(read_rtdsm_matrix(raw_dir)))
    return (
        pl.concat(parts)
        .sort("source", "file", "row_key", "column_key")
        .with_row_index("cell_id")
        .with_columns(pl.col("cell_id").cast(pl.Int64))
    )


def read_vintage_comments(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / VINTAGE_COMMENTS, infer_schema_length=0)


def content_sha256(frame: pl.DataFrame, *, chunk_rows: int = 500_000) -> str:
    """SHA-256 of the frame's CSV serialization with its header, hashed chunk by chunk."""
    digest = hashlib.sha256((",".join(frame.columns) + "\n").encode())
    for chunk in frame.iter_slices(chunk_rows):
        digest.update(chunk.write_csv(include_header=False).encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vintage_raw.py -q`
Expected: PASS: 12 passed.

- [ ] **Step 7: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 120 passed, 5 deselected.

- [ ] **Step 8: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/raw.py tests/vintage_data.py tests/test_vintage_raw.py tests/fixtures/vintages/cesnaicsrev-excerpt.htm
git commit -m "Read the vintage files, the revision table, and EMPLOY into an immutable raw-value table"
```

---

### Task 3: The release-date index

**Files:**
- Create: `tests/fixtures/vintages/histreleasedates-excerpt.txt`
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_release_index.py`
- Create: `src/ces_revisions/vintages/months.py`
- Create: `src/ces_revisions/vintages/release_index.py`

**Interfaces:**
- Consumes: Task 1's `raw` paths; the committed `bls/histreleasedates.txt`, `bls/empsit-releases.csv`, `philadelphiafed/release-dates-employment-situation.xls`, and `manual/es-reschedules.csv`; and, for the agreement test, `docs/inventory/es-vintages.csv`.
- Produces:
  - `ces_revisions.vintages.months`: `MONTH_NAMES` (`"January"` to `"December"`), `add_months(month: date, count: int) -> date`, and `month_range(first: date, last: date) -> list[date]`, both ends included. Stage 3 holds every month as a date on its first day.
  - `ces_revisions.vintages.release_index`: `FIRST_REFERENCE_MONTH = date(1979, 1, 1)`, `LAST_HISTORICAL_MONTH = date(2000, 12, 1)`, `FIRST_VINTAGE_FILE_MONTH = date(2003, 5, 1)`, `FIRST_JANUARY_BENCHMARK = date(2004, 1, 1)`, `EASTERN`, `EMBARGO = time(8, 30)`, `PUBLICATION = "employment_situation"`, and `INDEX_SCHEMA`; `parse_historical_release_dates(text: str) -> dict[date, date]`; `parse_rtdsm_release_dates(rows: list[tuple[str | None, ...]]) -> dict[date, date]`; `read_historical_release_dates(raw_dir: Path = RAW_DIR)`, `read_rtdsm_release_dates(raw_dir: Path = RAW_DIR)`, and `read_release_list(path: Path)`, each returning a `dict[date, date]` from reference month to release date; `read_reschedules(raw_dir: Path = RAW_DIR) -> dict[date, dict[str, str]]`; and `build_release_index(raw_dir: Path = RAW_DIR) -> pl.DataFrame`.
  - The index: one row per reference month from January 1979 through the latest release in `empsit-releases.csv`, with `reference_month`, `publication`, `scheduled_date`, `release_date` (null for the canceled October 2025 release), `published_date` (the carrying November 2025 release's date for that one, the release date otherwise), `observable_at` (UTC), `status` (`released` or `canceled`), `off_schedule`, `benchmark_release` (null before May 2003, true for the January releases from 2004), and `date_source` (`histreleasedates` through December 2000, `empsit_releases` after).
- Test helper: `vintage_data.index()`.

- [ ] **Step 1: Record the historical release-dates fixture**

Create `tests/fixtures/vintages/histreleasedates-excerpt.txt`, lines of `pdftotext -layout` output from BLS's PDF: the Employment Situation table's header, four year rows (1959 with unknown dates, 1995 with a footnote digit after the year, 1998 after the stray footnote-mark line, and 2000), its source notes and footnotes, and the start of the Consumer Price Index table, which the parser must not read:

```text
HISTORICAL RELEASE DATES FOR SELECTED BUREAU OF
LABOR STATISTICS NEWS RELEASES, 2000 AND EARLIER

               Employment Situation,
               Consumer Price Index,
Release dates for national employment and unemployment estimates, 1957-2000

                                                                                               Reference month
   Year        January         February        March          April           May           June            July           August      September      October       November        December
   1959      February 10      March 11         April 7       May 11         June 10        July 14        August 11           --      October 13   November 11          --               --
   1995       February 3      March 10         April 7        May 5          June 2         July 7        August 4       September 1   October 6    November 3     December 8    January 19, 19961
                                                                                                                                                               2
   1998       February 6       March 6         April 3        May 8          June 5         July 2        August 7       September 4   October 2    November 5     December 4     January 8, 1999
   2000       February 4       March 3         April 7        May 5          June 2         July 7        August 4       September 1   October 6    November 3     December 8     January 5, 2001
SOURCE: U.S. Department of Labor, Bureau of Labor Statistics, Office of Employment and Unemployment Statistics
NOTE: Dash indicates that release date is unknown. BLS began announcing its scheduled release dates in advance in 1961.
1
  The estimates for December 1995 originally were scheduled for release on Friday, January 5, 1996, but the release was delayed two weeks.
2
  The estimates for October 1998 originally were scheduled to be released at 8:30 AM EST on Friday, November 6.
Release dates for the Consumer Price Index, 1953-2000
                                                                                                               Reference month
    Year          January         February            March             April             May             June              July          August         September         October    November          December
    1953        February 27       March 26           April 22          May 22           June 23          July 22        August 26      September 23      October 28      November 25 December 23     January 22, 1954
```

- [ ] **Step 2: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    raw,
    release_index,
)


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_release_index.py`:

```python
"""The release-date index of Employment Situation releases from January 1979."""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, release_index
from ces_revisions.vintages.months import add_months, month_range

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"
ES_VINTAGES = raw.ROOT / "docs" / "inventory" / "es-vintages.csv"
RESCHEDULED = [
    date(1995, 12, 1),
    date(1998, 10, 1),
    date(2013, 9, 1),
    date(2013, 10, 1),
    date(2025, 9, 1),
    date(2025, 10, 1),
    date(2025, 11, 1),
    date(2026, 1, 1),
]


def test_add_months_crosses_year_boundaries_both_ways():
    assert add_months(date(2025, 11, 1), 2) == date(2026, 1, 1)
    assert add_months(date(2003, 5, 1), -1) == date(2003, 4, 1)
    assert add_months(date(2004, 1, 1), -13) == date(2002, 12, 1)


def test_month_range_includes_both_ends_and_is_empty_when_reversed():
    assert month_range(date(2025, 11, 1), date(2026, 1, 1)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
    ]
    assert month_range(date(2026, 2, 1), date(2026, 1, 1)) == []


def test_historical_release_dates_read_the_employment_table_only():
    text = (FIXTURES / "histreleasedates-excerpt.txt").read_text(encoding="utf-8")
    dates = release_index.parse_historical_release_dates(text)
    assert dates[date(1959, 1, 1)] == date(1959, 2, 10)
    assert date(1959, 8, 1) not in dates
    assert dates[date(1995, 12, 1)] == date(1996, 1, 19)
    assert dates[date(1998, 10, 1)] == date(1998, 11, 5)
    assert dates[date(2000, 12, 1)] == date(2001, 1, 5)
    assert (min(dates), max(dates), len(dates)) == (
        date(1959, 1, 1),
        date(2000, 12, 1),
        45,
    )


def test_a_historical_row_without_twelve_dates_is_an_error():
    text = (
        "Release dates for national employment and unemployment estimates, 1957-2000\n"
        "   1999       February 5       March 5\n"
        "Release dates for the Consumer Price Index, 1953-2000\n"
    )
    with pytest.raises(ValueError, match="expected 12 release dates"):
        release_index.parse_historical_release_dates(text)


def test_philadelphia_fed_release_dates_read_iso_and_marked_cells():
    rows = [
        ("Release dates for the Employment Situation, 1966-2010", *[None] * 12),
        ("Year", "Jan.", "Feb.", *[None] * 10),
        ("1995", *["1995-02-03 00:00:00"] * 11, "1/19/96*"),
        ("1998", *[None] * 9, "11/5/98**", None, None),
        (
            "* Delayed due to Government shutdown and weather-related closing.",
            *[None] * 12,
        ),
    ]
    dates = release_index.parse_rtdsm_release_dates(rows)
    assert dates[date(1995, 1, 1)] == date(1995, 2, 3)
    assert dates[date(1995, 12, 1)] == date(1996, 1, 19)
    assert dates[date(1998, 10, 1)] == date(1998, 11, 5)
    assert len(dates) == 13


# --- The committed sources ------------------------------------------------------------------


def test_bls_historical_dates_agree_with_the_philadelphia_fed_copy_from_1979():
    historical = release_index.read_historical_release_dates()
    copy = release_index.read_rtdsm_release_dates()
    months = month_range(date(1979, 1, 1), date(2000, 12, 1))
    assert [historical[month] for month in months] == [copy[month] for month in months]


def test_archive_dates_agree_with_the_philadelphia_fed_copy_from_2001_on():
    archive = release_index.read_release_list(raw.RAW_DIR / raw.EMPSIT_RELEASES)
    copy = release_index.read_rtdsm_release_dates()
    # The copy's title says 1966-2010, but its last date is November 2010's release.
    assert (min(copy), max(copy)) == (date(1966, 1, 1), date(2010, 11, 1))
    months = month_range(date(2001, 1, 1), date(2010, 11, 1))
    assert [archive[month] for month in months] == [copy[month] for month in months]


def test_the_index_has_one_row_per_month_from_january_1979():
    latest = max(release_index.read_release_list(raw.RAW_DIR / raw.EMPSIT_RELEASES))
    months = vintage_data.index()["reference_month"].to_list()
    assert months == month_range(date(1979, 1, 1), latest)


def test_the_index_agrees_with_es_vintages_except_the_december_1999_repost():
    """The archive index links December 1999's release as a file of January 19, 2000; the
    release's own embargo line, BLS's historical dates, and the Philadelphia Fed copy all give
    January 7, 2000."""
    index = dict(
        vintage_data.index().select("reference_month", "release_date").iter_rows()
    )
    listed = release_index.read_release_list(ES_VINTAGES)
    differences = {
        month: (index[month], released)
        for month, released in listed.items()
        if index[month] != released
    }
    assert differences == {date(1999, 12, 1): (date(2000, 1, 7), date(2000, 1, 19))}


def test_february_1995_to_april_1999_releases_come_from_bls_historical_dates():
    rows = vintage_data.index().filter(
        pl.col("reference_month").is_between(date(1995, 2, 1), date(1999, 4, 1))
    )
    assert rows.height == 51
    assert rows["date_source"].unique().to_list() == ["histreleasedates"]
    assert rows["release_date"].null_count() == 0


def test_rescheduled_releases_keep_their_scheduled_dates():
    index = vintage_data.index()
    moved = index.filter(pl.col("off_schedule"))
    assert moved["reference_month"].to_list() == RESCHEDULED
    rows = {row["reference_month"]: row for row in moved.iter_rows(named=True)}
    october = rows[date(2025, 10, 1)]
    assert (
        october["status"],
        october["scheduled_date"],
        october["release_date"],
        october["published_date"],
    ) == ("canceled", date(2025, 11, 7), None, date(2025, 12, 16))
    assert rows[date(2013, 9, 1)]["scheduled_date"] is None
    assert rows[date(1995, 12, 1)]["scheduled_date"] == date(1996, 1, 5)
    assert index.filter(pl.col("status") == "canceled").height == 1
    unmoved = index.filter(~pl.col("off_schedule"))
    assert (unmoved["scheduled_date"] == unmoved["release_date"]).all()


def test_observable_at_is_the_release_time_in_eastern_time():
    rows = dict(
        vintage_data.index().select("reference_month", "observable_at").iter_rows()
    )
    assert rows[date(1998, 10, 1)] == datetime(1998, 11, 5, 18, 30, tzinfo=UTC)
    assert rows[date(2025, 8, 1)] == datetime(2025, 9, 5, 12, 30, tzinfo=UTC)
    assert rows[date(2026, 1, 1)] == datetime(2026, 2, 11, 13, 30, tzinfo=UTC)


def test_benchmark_releases_are_the_january_releases_from_2004():
    index = vintage_data.index()
    early = index.filter(pl.col("reference_month") < date(2003, 5, 1))
    assert early["benchmark_release"].null_count() == early.height
    later = index.filter(pl.col("reference_month") >= date(2003, 5, 1))
    assert later["benchmark_release"].null_count() == 0
    flagged = later.filter(pl.col("benchmark_release"))["reference_month"].to_list()
    assert flagged == [date(year, 1, 1) for year in range(2004, flagged[-1].year + 1)]


def test_no_calendar_month_holds_two_releases():
    released = vintage_data.index().filter(pl.col("status") == "released")
    assert released["release_date"].dt.truncate("1mo").n_unique() == released.height


def test_each_rescheduled_release_cites_a_bls_page():
    for row in release_index.read_reschedules().values():
        assert row["citation"].startswith("https://www.bls.gov/"), row
        assert row["status"] in ("released", "canceled"), row
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_release_index.py -q`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'release_index' from 'ces_revisions.vintages'`, raised through `tests/vintage_data.py`.

- [ ] **Step 5: Write the month helpers**

Create `src/ces_revisions/vintages/months.py`:

```python
"""Month arithmetic on dates that stand for a month by its first day."""

from datetime import date

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def add_months(month: date, count: int) -> date:
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def month_range(first: date, last: date) -> list[date]:
    """Every month from `first` through `last`, both included."""
    months = []
    month = first
    while month <= last:
        months.append(month)
        month = add_months(month, 1)
    return months
```

- [ ] **Step 6: Write the release-date index**

Create `src/ces_revisions/vintages/release_index.py`:

```python
"""The release-date index: when each Employment Situation release was scheduled and published.

Release dates for reference months through December 2000 come from BLS's table of historical
release dates (bls/histreleasedates.txt, the text of histreleasedates.pdf), and later ones from
the Employment Situation archive index (bls/empsit-releases.csv). A hand-keyed table of the
eight releases that left their schedule, each row cited, adds the scheduled dates, the canceled
October 2025 release, and the one release not made at 8:30 a.m. Eastern. BLS publishes no
closing dates (Req 3), so the index holds none.
"""

import csv
import re
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import fastexcel
import polars as pl

from ces_revisions.vintages.months import MONTH_NAMES, month_range
from ces_revisions.vintages.raw import (
    EMPSIT_RELEASES,
    HISTORICAL_RELEASE_DATES,
    RAW_DIR,
    RESCHEDULES,
    RTDSM_RELEASE_DATES,
)

FIRST_REFERENCE_MONTH = date(1979, 1, 1)
LAST_HISTORICAL_MONTH = date(2000, 12, 1)
# The May 2003 publication vintage, the first release in the vintage files.
FIRST_VINTAGE_FILE_MONTH = date(2003, 5, 1)
# From the March 2003 benchmark on, benchmark revisions arrive with January estimates: every
# January release from 2004 in the vintage files revises the 21 months through the prior
# December, and no release from May to December 2003 revises more than two.
FIRST_JANUARY_BENCHMARK = date(2004, 1, 1)
EASTERN = ZoneInfo("America/New_York")
EMBARGO = time(8, 30)
PUBLICATION = "employment_situation"

_SECTION_START = "Release dates for national employment and unemployment estimates"
_SECTION_END = "Release dates for the Consumer Price Index"
_YEAR_ROW = re.compile(r"^\s*(\d{4})\s+(\S.*)$")
_DATE_TOKEN = re.compile(
    r"--|(" + "|".join(MONTH_NAMES) + r") (\d{1,2})(?:, (\d{4})\d?)?"
)

INDEX_SCHEMA = {
    "reference_month": pl.Date,
    "publication": pl.String,
    "scheduled_date": pl.Date,
    "release_date": pl.Date,
    "published_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "status": pl.String,
    "off_schedule": pl.Boolean,
    "benchmark_release": pl.Boolean,
    "date_source": pl.String,
}


def parse_historical_release_dates(text: str) -> dict[date, date]:
    """Reference month to release date from the Employment Situation table of the BLS PDF.

    Each row holds a year and twelve cells, one per reference month, such as "February 2" or,
    in December's column, "January 11, 1980"; a dash marks an unknown date, and a footnote
    digit can trail a year, as in "January 19, 19961".
    """
    start = text.index(_SECTION_START)
    section = text[start : text.index(_SECTION_END, start)]
    dates = {}
    for line in section.splitlines():
        row = _YEAR_ROW.match(line)
        if row is None:
            continue
        tokens = list(_DATE_TOKEN.finditer(row.group(2)))
        if len(tokens) != 12:
            raise ValueError(f"expected 12 release dates in {line!r}")
        year = int(row.group(1))
        for month, token in enumerate(tokens, start=1):
            if token.group(0) == "--":
                continue
            released = date(
                int(token.group(3) or year),
                MONTH_NAMES.index(token.group(1)) + 1,
                int(token.group(2)),
            )
            dates[date(year, month, 1)] = released
    return dates


def parse_rtdsm_release_dates(rows: list[tuple[str | None, ...]]) -> dict[date, date]:
    """Reference month to release date from the Philadelphia Fed's copy of BLS's date file.

    Year rows hold twelve cells as ISO dates, except the two releases the file marks with
    asterisks, which read "1/19/96*" and "11/5/98**".
    """
    dates = {}
    for row in rows:
        head = (row[0] or "").strip()
        if not re.fullmatch(r"\d{4}", head):
            continue
        for month, cell in enumerate(row[1:13], start=1):
            text = (cell or "").strip().rstrip("*")
            if not text:
                continue
            if "/" in text:
                month_part, day, year = (int(part) for part in text.split("/"))
                released = date(1900 + year, month_part, day)
            else:
                released = date.fromisoformat(text[:10])
            dates[date(int(head), month, 1)] = released
    return dates


def read_historical_release_dates(raw_dir: Path = RAW_DIR) -> dict[date, date]:
    text = (raw_dir / HISTORICAL_RELEASE_DATES).read_text(encoding="utf-8")
    return parse_historical_release_dates(text)


def read_rtdsm_release_dates(raw_dir: Path = RAW_DIR) -> dict[date, date]:
    sheet = fastexcel.read_excel(raw_dir / RTDSM_RELEASE_DATES).load_sheet(
        "Release dates", header_row=None, dtypes="string"
    )
    return parse_rtdsm_release_dates(sheet.to_polars().rows())


def read_release_list(path: Path) -> dict[date, date]:
    """Reference month to release date from a CSV in the format of es-vintages.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            date.fromisoformat(f"{row['reference_month']}-01"): date.fromisoformat(
                row["release_date"]
            )
            for row in csv.DictReader(handle)
        }


def read_reschedules(raw_dir: Path = RAW_DIR) -> dict[date, dict[str, str]]:
    with (raw_dir / RESCHEDULES).open(newline="", encoding="utf-8") as handle:
        return {
            date.fromisoformat(f"{row['reference_month']}-01"): row
            for row in csv.DictReader(handle)
        }


def _optional_date(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


def build_release_index(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """One row per reference month from January 1979 through the latest archived release."""
    historical = read_historical_release_dates(raw_dir)
    archive = read_release_list(raw_dir / EMPSIT_RELEASES)
    moved = read_reschedules(raw_dir)
    rows, times = {}, {}
    for month in month_range(FIRST_REFERENCE_MONTH, max(archive)):
        if month <= LAST_HISTORICAL_MONTH:
            source, released = "histreleasedates", historical.get(month)
        else:
            source, released = "empsit_releases", archive.get(month)
        scheduled, status, times[month] = released, "released", EMBARGO
        change = moved.get(month)
        if change is not None:
            scheduled, status = (
                _optional_date(change["scheduled_date"]),
                change["status"],
            )
            if status == "canceled":
                released = None
            elif released != date.fromisoformat(change["release_date"]):
                raise ValueError(
                    f"{month:%Y-%m}: {source} gives {released}, "
                    f"{RESCHEDULES} {change['release_date']}"
                )
            else:
                times[month] = time.fromisoformat(change["release_time_et"])
        elif released is None:
            raise ValueError(f"no release date for {month:%Y-%m}")
        rows[month] = {
            "reference_month": month,
            "publication": PUBLICATION,
            "scheduled_date": scheduled,
            "release_date": released,
            "status": status,
            "off_schedule": change is not None,
            "benchmark_release": None
            if month < FIRST_VINTAGE_FILE_MONTH
            else month.month == 1 and month >= FIRST_JANUARY_BENCHMARK,
            "date_source": source,
        }
    for month, row in rows.items():
        carrier = month
        if row["status"] == "canceled":
            published_with = moved[month]["published_with"]
            carrier = date.fromisoformat(f"{published_with}-01")
        row["published_date"] = rows[carrier]["release_date"]
        row["observable_at"] = datetime.combine(
            row["published_date"], times[carrier], EASTERN
        ).astimezone(UTC)
    return pl.DataFrame(list(rows.values()), schema=INDEX_SCHEMA)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_release_index.py -q`
Expected: PASS: 15 passed.

- [ ] **Step 8: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 135 passed, 5 deselected.

- [ ] **Step 9: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/months.py src/ces_revisions/vintages/release_index.py tests/vintage_data.py tests/test_release_index.py tests/fixtures/vintages/histreleasedates-excerpt.txt
git commit -m "Date every Employment Situation release from January 1979"
```

---

### Task 4: The revision table's estimates and published averages

**Files:**
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_revision_table.py`
- Create: `src/ces_revisions/vintages/revision_table.py`

**Interfaces:**
- Consumes: Task 2's `raw_values()`, `revision_table_cells`, and revision table fixture.
- Produces, in `ces_revisions.vintages.revision_table`:
  - `STAGE_OF_ESTIMATE = {"1st": "F", "2nd": "S", "3rd": "T"}`; `REVISION_TERMS`, each revision column to the later and earlier estimates it compares; `REVISIONS_OF_ESTIMATE`; and `SUMMARY_PERIODS`, the periods of the page's footnote **.
  - `Cell(value: int | None, marker: str | None)` and `parse_cell(text: str) -> Cell`, with markers `"A"`, `"B"`, `"c"`, `"NA"`, `"blank"`, or `None`.
  - `monthly_cells(raw: pl.DataFrame) -> pl.DataFrame`: `cell_id`, `reference_month`, `seasonal_status` (`SA` or `NSA`), `column` (`1st` to `3rd_minus_1st`), `text`, `value`, and `marker`.
  - `estimates(cells: pl.DataFrame) -> pl.DataFrame`: `cell_id`, `reference_month`, `seasonal_status`, `release_stage` (`F`, `S`, or `T`), `value`, and `marker`, with the 2003 zeros set missing and marked `NA`.
  - `revision_checks(cells) -> pl.DataFrame`: `reference_month`, `seasonal_status`, `column`, `published`, `difference`, and `consistent`.
  - `average(values: list[int], kind: str) -> Fraction`, `rounds_to(published: int, exact: Fraction) -> bool`, and `average_checks(raw) -> pl.DataFrame`: `row_key`, `seasonal_status`, `column`, `kind`, `published`, `months`, `exact` (the fraction as text), and `reproduced`.
- Test helpers: `vintage_data.table_cells()` and `vintage_data.table_estimates()`.

- [ ] **Step 1: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    raw,
    release_index,
    revision_table,
)


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_revision_table.py`:

```python
"""BLS's revision table as data: cells, markers, estimates, and published averages."""

from datetime import date
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, revision_table
from ces_revisions.vintages.revision_table import Cell, parse_cell, rounds_to

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"


def excerpt_estimates() -> pl.DataFrame:
    page = (FIXTURES / "cesnaicsrev-excerpt.htm").read_text(encoding="utf-8")
    cells = raw.revision_table_cells(page).with_row_index("cell_id")
    return revision_table.estimates(revision_table.monthly_cells(cells))


def estimate(frame: pl.DataFrame, month: date, status: str, stage: str) -> tuple:
    row = frame.filter(
        (pl.col("reference_month") == month)
        & (pl.col("seasonal_status") == status)
        & (pl.col("release_stage") == stage)
    )
    return row["value"].item(), row["marker"].item()


@pytest.mark.parametrize(
    ("text", "cell"),
    [
        ("143", Cell(143, None)),
        ("-2722", Cell(-2722, None)),
        ("-105 (B)", Cell(-105, "B")),
        ("31(c)", Cell(31, "c")),
        ("-(A)", Cell(None, "A")),
        ("NA***", Cell(None, "NA")),
        ("NA", Cell(None, "NA")),
        ("", Cell(None, "blank")),
    ],
)
def test_parse_cell_reads_values_and_footnote_markers(text, cell):
    assert parse_cell(text) == cell


def test_an_unrecognized_cell_is_an_error():
    with pytest.raises(ValueError, match="unrecognized revision table cell"):
        parse_cell("12.5")


def test_the_2003_zeros_are_missing_estimates():
    frame = excerpt_estimates()
    assert estimate(frame, date(2003, 4, 1), "SA", "F") == (-48, None)
    assert estimate(frame, date(2003, 3, 1), "SA", "S") == (-124, None)
    assert estimate(frame, date(2003, 3, 1), "SA", "T") == (None, "NA")
    assert estimate(frame, date(2003, 4, 1), "NSA", "S") == (None, "NA")
    assert estimate(frame, date(2003, 4, 1), "NSA", "T") == (None, "NA")
    assert estimate(frame, date(2003, 5, 1), "NSA", "T") == (680, None)


def test_october_2025_has_no_first_estimate_and_a_marked_second():
    frame = excerpt_estimates()
    assert estimate(frame, date(2025, 10, 1), "SA", "F") == (None, "A")
    assert estimate(frame, date(2025, 10, 1), "SA", "S") == (-105, "B")
    assert estimate(frame, date(2025, 9, 1), "NSA", "S") == (None, "A")
    assert estimate(frame, date(2025, 9, 1), "NSA", "T") == (328, None)


@pytest.mark.parametrize(
    ("published", "exact", "expected"),
    [
        (78, Fraction(157, 2), True),
        (79, Fraction(157, 2), True),
        (77, Fraction(157, 2), False),
        (-4, Fraction(-9, 2), True),
        (-5, Fraction(-9, 2), True),
        (48, Fraction(12019, 250), True),
        (49, Fraction(12019, 250), False),
    ],
)
def test_rounds_to_takes_either_neighbor_of_an_exact_half(published, exact, expected):
    assert rounds_to(published, exact) is expected


# --- The committed page ---------------------------------------------------------------------


def test_every_published_average_reproduces_from_its_monthly_revisions():
    checks = revision_table.average_checks(vintage_data.raw_values())
    assert checks.filter(~pl.col("reproduced")).height == 0
    assert checks.filter(pl.col("row_key").str.starts_with("summary:")).height == 36


def test_bls_rounds_exact_half_yearly_averages_both_ways():
    checks = revision_table.average_checks(vintage_data.raw_values()).filter(
        pl.col("row_key").str.contains(r"^(?:19\d\d|20[01]\d|202[0-5]):")
    )
    ties = [
        (published, Fraction(exact))
        for published, exact in checks.select("published", "exact").iter_rows()
        if Fraction(exact).denominator == 2
    ]
    rounded_up = sum(published == exact + Fraction(1, 2) for published, exact in ties)
    assert (len(ties), rounded_up) == (61, 37)


def test_published_revisions_equal_their_estimates_difference_but_once():
    checks = revision_table.revision_checks(vintage_data.table_cells())
    inconsistent = checks.filter(~pl.col("consistent")).select(
        "reference_month", "seasonal_status", "column", "published", "difference"
    )
    assert inconsistent.rows() == [(date(2023, 9, 1), "SA", "3rd_minus_2nd", -34, -35)]


def test_markers_sit_only_where_the_page_footnotes_put_them():
    cells = vintage_data.table_cells()

    def months(marker: str) -> list[date]:
        rows = cells.filter(pl.col("marker") == marker)
        return sorted(set(rows["reference_month"]))

    assert months("NA") == [date(2003, 3, 1), date(2003, 4, 1)]
    assert months("A") == [date(2025, 9, 1), date(2025, 10, 1)]
    marked_b = cells.filter(pl.col("marker") == "B")
    assert sorted(marked_b.select("seasonal_status", "column").iter_rows()) == [
        ("NSA", "2nd"),
        ("SA", "2nd"),
    ]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_revision_table.py -q`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'revision_table' from 'ces_revisions.vintages'`.

- [ ] **Step 4: Write the revision table module**

Create `src/ces_revisions/vintages/revision_table.py`:

```python
"""BLS's revision table as data: estimates with their markers, and the averages it publishes.

For each month since January 1979 the page gives total nonfarm over-the-month changes at the
first, second, and third estimates, seasonally adjusted and not, their three pairwise
revisions, and each year's mean and mean absolute revision; two summary tables average the
revisions over the periods its footnote ** defines. Markers carry the page's footnotes: `A` for
values the 2025 lapse left unavailable, `B` for October 2025's first published estimates, which
the page counts as second estimates, `c` for corrected averages, and `NA` for the 2003 redesign.
"""

import re
from dataclasses import dataclass
from datetime import date
from fractions import Fraction

import polars as pl

STAGE_OF_ESTIMATE = {"1st": "F", "2nd": "S", "3rd": "T"}
# Each revision column and the later and earlier estimates it compares.
REVISION_TERMS = {
    "2nd_minus_1st": ("2nd", "1st"),
    "3rd_minus_2nd": ("3rd", "2nd"),
    "3rd_minus_1st": ("3rd", "1st"),
}
# The page prints zeros for March 2003's third estimate and April 2003's second and third, and
# NA in every revision they enter; a zero estimate whose every revision is NA is a placeholder.
# April 2003's first estimate also has only NA revisions, but it is a published change.
REVISIONS_OF_ESTIMATE = {
    estimate: tuple(
        column for column, terms in REVISION_TERMS.items() if estimate in terms
    )
    for estimate in STAGE_OF_ESTIMATE
}
# The periods of the page's footnote **. The later period, and the total, run through the
# latest month each revision is published for.
SUMMARY_PERIODS = {
    "1979 - 2003": {
        "2nd_minus_1st": (date(1979, 1, 1), date(2003, 3, 1)),
        "3rd_minus_2nd": (date(1979, 1, 1), date(2003, 2, 1)),
        "3rd_minus_1st": (date(1979, 1, 1), date(2003, 2, 1)),
    },
    "2003 - present": dict.fromkeys(REVISION_TERMS, (date(2003, 5, 1), None)),
    "Total All Periods": dict.fromkeys(REVISION_TERMS, (date(1979, 1, 1), None)),
}


@dataclass(frozen=True)
class Cell:
    value: int | None
    marker: str | None


def parse_cell(text: str) -> Cell:
    if text == "":
        return Cell(None, "blank")
    if text == "-(A)":
        return Cell(None, "A")
    if re.fullmatch(r"NA(?:\*\*\*)?", text):
        return Cell(None, "NA")
    match = re.fullmatch(r"(-?\d+)(?:\s*\((B|c)\))?", text)
    if match is None:
        raise ValueError(f"unrecognized revision table cell {text!r}")
    return Cell(int(match.group(1)), match.group(2))


def monthly_cells(raw: pl.DataFrame) -> pl.DataFrame:
    """One row per month and column of the page, with the parsed value and marker."""
    frame = raw.filter(
        (pl.col("source") == "cesnaicsrev")
        & pl.col("row_key").str.contains(r"^\d{4}-\d{2}$")
    )
    parsed = [parse_cell(text) for text in frame["text"]]
    return frame.select(
        "cell_id",
        reference_month=(pl.col("row_key") + "-01").str.to_date("%Y-%m-%d"),
        seasonal_status=pl.col("column_key")
        .str.extract(r"^(n?sa)_")
        .str.to_uppercase(),
        column=pl.col("column_key").str.extract(r"^n?sa_(.+)$"),
        text="text",
        value=pl.Series([cell.value for cell in parsed], dtype=pl.Int64),
        marker=pl.Series([cell.marker for cell in parsed], dtype=pl.String),
    )


def estimates(cells: pl.DataFrame) -> pl.DataFrame:
    """The first, second, and third estimates, with the 2003 placeholders set to missing."""
    revision_na = cells.filter(pl.col("column").is_in(list(REVISION_TERMS))).select(
        "reference_month",
        "seasonal_status",
        "column",
        na=(pl.col("marker") == "NA").fill_null(False),
    )
    parts = []
    for estimate, stage in STAGE_OF_ESTIMATE.items():
        placeholders = (
            revision_na.filter(pl.col("column").is_in(REVISIONS_OF_ESTIMATE[estimate]))
            .group_by("reference_month", "seasonal_status")
            .agg(placeholder=pl.col("na").all())
        )
        parts.append(
            cells.filter(pl.col("column") == estimate)
            .join(placeholders, on=["reference_month", "seasonal_status"], how="left")
            .with_columns(
                placeholder=pl.col("placeholder")
                & (pl.col("value") == 0).fill_null(False)
            )
            .select(
                "cell_id",
                "reference_month",
                "seasonal_status",
                release_stage=pl.lit(stage),
                value=pl.when(pl.col("placeholder"))
                .then(pl.lit(None, dtype=pl.Int64))
                .otherwise("value"),
                marker=pl.when(pl.col("placeholder"))
                .then(pl.lit("NA"))
                .otherwise("marker"),
            )
        )
    return pl.concat(parts).sort("reference_month", "seasonal_status", "release_stage")


def revision_checks(cells: pl.DataFrame) -> pl.DataFrame:
    """Each published revision beside the difference of the two estimates it compares."""
    values = cells.pivot(
        on="column", index=["reference_month", "seasonal_status"], values="value"
    )
    parts = [
        values.select(
            "reference_month",
            "seasonal_status",
            column=pl.lit(column),
            published=pl.col(column),
            difference=pl.col(later) - pl.col(earlier),
        )
        for column, (later, earlier) in REVISION_TERMS.items()
    ]
    return (
        pl.concat(parts)
        .drop_nulls(["published", "difference"])
        .with_columns(consistent=pl.col("published") == pl.col("difference"))
        .sort("reference_month", "seasonal_status", "column")
    )


def average(values: list[int], kind: str) -> Fraction:
    numbers = [abs(value) for value in values] if kind == "mean_absolute" else values
    return Fraction(sum(numbers), len(numbers))


def rounds_to(published: int, exact: Fraction) -> bool:
    """Whether `published` is `exact` rounded to an integer, taking either neighbor of a half.

    BLS rounds exact halves both ways: of the 61 yearly averages on the page that fall exactly
    halfway between two integers, 37 are rounded up and 24 down.
    """
    floor = exact.numerator // exact.denominator
    remainder = exact - floor
    if remainder == Fraction(1, 2):
        return published in (floor, floor + 1)
    return published == (floor + 1 if remainder > Fraction(1, 2) else floor)


def average_checks(raw: pl.DataFrame) -> pl.DataFrame:
    """Each published yearly or summary average beside the exact average of its revisions."""
    revisions = monthly_cells(raw).filter(
        pl.col("column").is_in(list(REVISION_TERMS)) & pl.col("value").is_not_null()
    )
    published = raw.filter(
        (pl.col("source") == "cesnaicsrev") & pl.col("row_key").str.contains(":mean")
    )
    rows = []
    for row_key, column_key, text in published.select(
        "row_key", "column_key", "text"
    ).iter_rows():
        cell = parse_cell(text)
        if cell.value is None:
            continue
        status, column = column_key.split("_", 1)
        if row_key.startswith("summary:"):
            _, kind, period = row_key.split(":", 2)
            first, last = SUMMARY_PERIODS[period][column]
        else:
            year, kind = row_key.split(":")
            first, last = date(int(year), 1, 1), date(int(year), 12, 1)
        window = revisions.filter(
            (pl.col("seasonal_status") == status.upper())
            & (pl.col("column") == column)
            & (pl.col("reference_month") >= first)
            & ((pl.col("reference_month") <= last) if last else pl.lit(True))
        )
        exact = average(window["value"].to_list(), kind)
        rows.append(
            {
                "row_key": row_key,
                "seasonal_status": status.upper(),
                "column": column,
                "kind": kind,
                "published": cell.value,
                "months": window.height,
                "exact": str(exact),
                "reproduced": rounds_to(cell.value, exact),
            }
        )
    return pl.DataFrame(rows)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_revision_table.py -q`
Expected: PASS: 22 passed.

- [ ] **Step 6: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 157 passed, 5 deselected.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/revision_table.py tests/vintage_data.py tests/test_revision_table.py
git commit -m "Parse the revision table's estimates and reproduce its published averages"
```

---

### Task 5: Transformations and the panel's three sources

**Files:**
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_vintage_panel.py`
- Create: `src/ces_revisions/vintages/panel.py`

**Interfaces:**
- Consumes: `raw_values()` and `raw.SECTORS` (Task 2); the index's `reference_month`, `release_date`, `published_date`, and `status` (Task 3); and `monthly_cells` and `estimates` (Task 4).
- Produces, in `ces_revisions.vintages.panel`:
  - `REGIME_SHIFT_MONTH = date(2003, 5, 1)`, `VINTAGE_FILE_YEAR_PIVOT = 39`, `RTDSM_YEAR_PIVOT = 64`, `MONTH_ABBREVIATIONS`, and `STAGE_OFFSETS`.
  - `TRANSFORMATIONS`, a frame of `transformation`, `source`, `description`, and `parameters` (JSON) for `vintage_file_level`, `vintage_file_lapse_sentinel`, `rtdsm_level`, `revision_table_estimate`, and `revision_table_missing_estimate`.
  - `vintage_file_levels(raw, index)`, `rtdsm_levels(raw, index)`, and `revision_table_changes(raw, index)`, each returning a `pl.DataFrame` with `cell_id`, `source`, `sector`, `reference_month`, `release_month` (the reference month of the Employment Situation release whose values the row holds), `seasonal_status`, `measure` (`level` or `change`), `value_thousands`, `marker`, `transformation`, `release_date` (the index's `published_date`, or for EMPLOY the release's date), `vintage_id` (`cesvinall:YYYY-MM`, `rtdsm:EMPLOYyyMm`, or `cesnaicsrev:YYYY-MM`), and `concept_regime` (`pre_2003_05` or `from_2003_05`); the table's frame also has `release_stage`.
- Test helpers: `vintage_data.levels()`, `vintage_data.rtdsm()`, and `vintage_data.table_changes()`.

- [ ] **Step 1: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    panel,
    raw,
    release_index,
    revision_table,
)


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())


@cache
def levels() -> pl.DataFrame:
    return panel.vintage_file_levels(raw_values(), index())


@cache
def rtdsm() -> pl.DataFrame:
    return panel.rtdsm_levels(raw_values(), index())


@cache
def table_changes() -> pl.DataFrame:
    return panel.revision_table_changes(raw_values(), index())
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_vintage_panel.py`. Task 6 appends a section for the assembled panel:

```python
"""The long vintage panel of Req 1: transformations, lineage, sums, and the aggregate leg."""

import json
import re
from datetime import date

import polars as pl
import vintage_data

from ces_revisions.vintages import panel, raw
from ces_revisions.vintages.revision_table import parse_cell

KEY = [
    "source",
    "sector",
    "reference_month",
    "vintage_id",
    "seasonal_status",
    "measure",
]
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
MONTHS += ("Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
STAGE_OF_COLUMN = {"1st": ("F", 0), "2nd": ("S", 1), "3rd": ("T", 2)}


def month_after(month: date, count: int) -> date:
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def rebuilt_key(file: str, row_key: str, column_key: str) -> tuple:
    """A panel row's key from its raw cell's keys, derived apart from the panel module."""
    if file.startswith(raw.VINTAGE_FILES):
        code, status = re.search(r"tri_(\d{6})_(N?SA)\.csv$", file).groups()
        two_digit = int(column_key[4:])
        year = 1900 + two_digit if two_digit >= 39 else 2000 + two_digit
        month = date(year, MONTHS.index(column_key[:3]) + 1, 1)
        return raw.SECTORS[code], month, status, date.fromisoformat(f"{row_key}-01")
    if file.startswith(raw.RTDSM_LEVELS):
        return "00", date.fromisoformat(f"{row_key.replace(':', '-')}-01"), "SA"
    status, column = column_key.split("_", 1)
    stage, offset = STAGE_OF_COLUMN[column]
    month = date.fromisoformat(f"{row_key}-01")
    return "00", month, status.upper(), month_after(month, offset), stage


def sources() -> dict[str, pl.DataFrame]:
    return {
        "cesvinall": vintage_data.levels(),
        "rtdsm_employ": vintage_data.rtdsm(),
        "cesnaicsrev": vintage_data.table_changes(),
    }


def test_transformations_name_and_parameterize_every_rule():
    names = panel.TRANSFORMATIONS["transformation"].to_list()
    assert names == [
        "vintage_file_level",
        "vintage_file_lapse_sentinel",
        "rtdsm_level",
        "revision_table_estimate",
        "revision_table_missing_estimate",
    ]
    for parameters in panel.TRANSFORMATIONS["parameters"]:
        assert isinstance(json.loads(parameters), dict)


# --- The committed sources ------------------------------------------------------------------


def test_the_sources_use_every_named_transformation_and_no_other():
    used = set()
    for frame in sources().values():
        used |= set(frame["transformation"].unique())
    assert used == set(panel.TRANSFORMATIONS["transformation"])


def test_source_keys_are_unique():
    for name, frame in sources().items():
        assert frame.select(KEY).n_unique() == frame.height, name


def test_every_value_rebuilds_from_its_raw_cell():
    columns = ["cell_id", "value_thousands", "marker", "transformation"]
    joined = pl.concat([frame.select(columns) for frame in sources().values()]).join(
        vintage_data.raw_values().select("cell_id", "text"), on="cell_id", how="left"
    )
    assert joined["text"].null_count() == 0
    levels = joined.filter(
        pl.col("transformation").is_in(["vintage_file_level", "rtdsm_level"])
    )
    assert (levels["value_thousands"] == levels["text"].cast(pl.Int64)).all()
    sentinels = joined.filter(pl.col("transformation") == "vintage_file_lapse_sentinel")
    assert (sentinels["text"] == "-1").all()
    assert sentinels["value_thousands"].null_count() == sentinels.height
    estimates = joined.filter(pl.col("transformation") == "revision_table_estimate")
    assert estimates["value_thousands"].to_list() == [
        parse_cell(text).value for text in estimates["text"]
    ]
    missing = joined.filter(
        pl.col("transformation") == "revision_table_missing_estimate"
    )
    assert missing["value_thousands"].null_count() == missing.height
    assert set(missing["marker"]) == {"A", "NA"}


def test_keys_rebuild_from_their_raw_cells():
    raw_keys = vintage_data.raw_values().select(
        "cell_id", "file", "row_key", "column_key"
    )
    columns = {
        "cesvinall": ["sector", "reference_month", "seasonal_status", "release_month"],
        "rtdsm_employ": ["sector", "reference_month", "seasonal_status"],
        "cesnaicsrev": [
            "sector",
            "reference_month",
            "seasonal_status",
            "release_month",
            "release_stage",
        ],
    }
    for name, frame in sources().items():
        sample = frame.join(raw_keys, on="cell_id").sort("cell_id").gather_every(97)
        for row in sample.iter_rows(named=True):
            expected = tuple(row[column] for column in columns[name])
            assert (
                rebuilt_key(row["file"], row["row_key"], row["column_key"]) == expected
            )


def test_the_eleven_supersectors_sum_to_total_nonfarm_in_every_release_file():
    """Req 5. BLS publishes these aggregates as exact sums of the supersectors, so a nonzero
    difference is a finding to report, not a tolerance to widen."""
    frame = vintage_data.levels().filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    total = frame.filter(pl.col("sector") == "00").select(
        *keys, total="value_thousands"
    )
    parts = (
        frame.filter(pl.col("sector") != "00")
        .group_by(keys)
        .agg(parts=pl.col("value_thousands").sum(), supersectors=pl.len())
    )
    joined = total.join(parts, on=keys, how="left")
    assert (joined["supersectors"] == 11).all()
    assert (joined["parts"] == joined["total"]).all()
    releases = frame["release_month"].n_unique()
    assert joined.select("release_month", "seasonal_status").n_unique() == 2 * releases


def test_vintage_file_releases_begin_with_the_may_2003_publication_vintage():
    frame = vintage_data.levels()
    assert frame["release_month"].min() == date(2003, 5, 1)
    assert frame["concept_regime"].unique().to_list() == ["from_2003_05"]


def test_the_aggregate_leg_carries_concept_regime():
    frame = pl.concat(
        [
            vintage_data.rtdsm().select("source", "release_month", "concept_regime"),
            vintage_data.table_changes().select(
                "source", "release_month", "concept_regime"
            ),
        ]
    )
    expected = (
        pl.when(pl.col("release_month") < date(2003, 5, 1))
        .then(pl.lit("pre_2003_05"))
        .otherwise(pl.lit("from_2003_05"))
    )
    assert frame["concept_regime"].null_count() == 0
    assert frame.select((pl.col("concept_regime") == expected).all()).item()
    regimes = set(frame.select("source", "concept_regime").unique().iter_rows())
    assert regimes == {
        ("cesnaicsrev", "pre_2003_05"),
        ("cesnaicsrev", "from_2003_05"),
        ("rtdsm_employ", "pre_2003_05"),
        ("rtdsm_employ", "from_2003_05"),
    }


def test_the_canceled_october_2025_row_is_published_with_november_and_holds_sentinels():
    row = vintage_data.levels().filter(pl.col("vintage_id") == "cesvinall:2025-10")
    assert row["release_date"].unique().to_list() == [date(2025, 12, 16)]
    missing = row.filter(pl.col("value_thousands").is_null())
    assert missing.height == 48
    assert sorted(set(missing["reference_month"])) == [
        date(2025, 9, 1),
        date(2025, 10, 1),
    ]


def test_rtdsm_vintages_end_with_the_month_their_release_first_estimates():
    frame = vintage_data.rtdsm()
    vintages = frame.group_by("vintage_id", "release_month").agg(
        last=pl.col("reference_month").max()
    )
    assert (vintages["last"] == vintages["release_month"]).all()
    assert vintages["vintage_id"].n_unique() == vintages.height
    october = frame.filter(pl.col("vintage_id") == "rtdsm:EMPLOY25M10")
    assert october["release_month"].unique().to_list() == [date(2025, 8, 1)]
    assert frame["release_month"].min() == date(1979, 1, 1)


def test_revision_table_changes_are_keyed_to_the_releases_that_made_them():
    rows = {
        (row["reference_month"], row["seasonal_status"], row["release_stage"]): row
        for row in vintage_data.table_changes().iter_rows(named=True)
    }
    october = rows[(date(2025, 10, 1), "SA", "S")]
    assert (
        october["release_month"],
        october["release_date"],
        october["value_thousands"],
        october["marker"],
    ) == (date(2025, 11, 1), date(2025, 12, 16), -105, "B")
    august = rows[(date(2025, 8, 1), "NSA", "T")]
    assert (
        august["release_month"],
        august["release_date"],
        august["value_thousands"],
    ) == (date(2025, 10, 1), date(2025, 12, 16), 185)
    assert rows[(date(1979, 1, 1), "NSA", "F")]["release_date"] == date(1979, 2, 2)


def test_the_revision_table_and_the_release_index_end_with_the_same_release():
    """A fetch that straddles a release's embargo can leave the page a release ahead."""
    first = vintage_data.table_changes().filter(
        (pl.col("release_stage") == "F") & pl.col("value_thousands").is_not_null()
    )
    assert (
        first["reference_month"].max() == vintage_data.index()["reference_month"].max()
    )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_vintage_panel.py -q`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'panel' from 'ces_revisions.vintages'`.

- [ ] **Step 4: Write the transformations and source frames**

Create `src/ces_revisions/vintages/panel.py`. Task 6 appends `PANEL_COLUMNS` and `assemble_panel`:

```python
"""The long vintage panel of Req 1, and the named transformations that read it from raw cells.

Every panel row keeps the cell_id of the raw cell it was read from and the name of the
transformation that read it; TRANSFORMATIONS describes each one with its parameters. The panel
holds three sources: vintage-file levels for total nonfarm and the eleven supersectors, the
Philadelphia Fed's seasonally adjusted total nonfarm level vintages, and the revision table's
total nonfarm estimates of over-the-month change, which reach back to January 1979.
"""

import json
from datetime import date

import polars as pl

from ces_revisions.vintages.raw import SECTORS
from ces_revisions.vintages.revision_table import estimates, monthly_cells

# Req 4's composite regime shift: releases from the May 2003 publication vintage on.
REGIME_SHIFT_MONTH = date(2003, 5, 1)
VINTAGE_FILE_YEAR_PIVOT = 39  # column "Jan_39" is January 1939, "Jan_26" January 2026
RTDSM_YEAR_PIVOT = 64  # column "EMPLOY64M12" is December 1964, "EMPLOY26M8" August 2026
MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
STAGE_OFFSETS = {"F": "0mo", "S": "1mo", "T": "2mo"}

TRANSFORMATIONS = pl.DataFrame(
    [
        {
            "transformation": "vintage_file_level",
            "source": "cesvinall",
            "description": (
                "A vintage-file cell's text is a level in whole thousands. The member name "
                "gives the sector and seasonal status, the row key the release month, and the "
                "column key the reference month, two-digit years from the pivot on read as "
                "19xx and earlier ones as 20xx."
            ),
            "parameters": json.dumps(
                {"sectors": SECTORS, "year_pivot": VINTAGE_FILE_YEAR_PIVOT}
            ),
        },
        {
            "transformation": "vintage_file_lapse_sentinel",
            "source": "cesvinall",
            "description": (
                "The vintage files write -1 where the 2025 lapse left an estimate unpublished: "
                "September 2025's second and October 2025's first, in the row for the "
                "canceled October 2025 release. The level is missing."
            ),
            "parameters": json.dumps({"sentinel": "-1"}),
        },
        {
            "transformation": "rtdsm_level",
            "source": "rtdsm_employ",
            "description": (
                "An EMPLOY cell's text is a seasonally adjusted total nonfarm level in "
                "thousands. Column EMPLOYyyMm is the vintage of calendar month m of year yy, "
                "which holds the Employment Situation release made in that month or, when "
                "none was, the latest earlier one. #N/A cells and vintages before the first "
                "release in the release-date index are not read."
            ),
            "parameters": json.dumps(
                {"missing": "#N/A", "year_pivot": RTDSM_YEAR_PIVOT}
            ),
        },
        {
            "transformation": "revision_table_estimate",
            "source": "cesnaicsrev",
            "description": (
                "A first, second, or third estimate of the total nonfarm over-the-month "
                "change in thousands, published by the release zero, one, or two months after "
                "the reference month's own. A (B) or (c) marker is kept beside the value."
            ),
            "parameters": json.dumps({"release_offsets": STAGE_OFFSETS}),
        },
        {
            "transformation": "revision_table_missing_estimate",
            "source": "cesnaicsrev",
            "description": (
                "An estimate the page marks unavailable after the 2025 lapse (A) or leaves "
                "out for the 2003 redesign (NA), including the zeros it prints where every "
                "revision of the estimate is NA. The change is missing."
            ),
            "parameters": json.dumps({"markers": ["A", "NA"]}),
        },
    ]
)


def _regime() -> pl.Expr:
    return (
        pl.when(pl.col("release_month") < REGIME_SHIFT_MONTH)
        .then(pl.lit("pre_2003_05"))
        .otherwise(pl.lit("from_2003_05"))
    )


def _year(two_digit: pl.Expr, pivot: int) -> pl.Expr:
    return (
        pl.when(two_digit >= pivot).then(two_digit + 1900).otherwise(two_digit + 2000)
    )


def _published(frame: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    releases = index.select(
        release_month="reference_month", release_date="published_date"
    )
    return frame.join(releases, on="release_month", how="left")


def vintage_file_levels(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """Every vintage-file level of total nonfarm and the eleven supersectors."""
    two_digit = pl.col("column_key").str.slice(4, 2).cast(pl.Int32)
    month = (
        pl.col("column_key")
        .str.slice(0, 3)
        .replace_strict(
            {name: number for number, name in enumerate(MONTH_ABBREVIATIONS, start=1)},
            return_dtype=pl.Int32,
        )
    )
    sentinel = pl.col("text") == "-1"
    levels = raw.filter(pl.col("source") == "cesvinall").select(
        "cell_id",
        source=pl.lit("cesvinall"),
        sector=pl.col("file")
        .str.extract(r"tri_(\d{6})_N?SA\.csv$", 1)
        .replace_strict(SECTORS),
        reference_month=pl.date(_year(two_digit, VINTAGE_FILE_YEAR_PIVOT), month, 1),
        release_month=(pl.col("row_key") + "-01").str.to_date("%Y-%m-%d"),
        seasonal_status=pl.col("file").str.extract(r"tri_\d{6}_(N?SA)\.csv$", 1),
        measure=pl.lit("level"),
        value_thousands=pl.when(sentinel)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("text").cast(pl.Int64)),
        marker=pl.when(sentinel)
        .then(pl.lit("lapse_sentinel"))
        .otherwise(pl.lit(None, dtype=pl.String)),
        transformation=pl.when(sentinel)
        .then(pl.lit("vintage_file_lapse_sentinel"))
        .otherwise(pl.lit("vintage_file_level")),
    )
    return _published(levels, index).with_columns(
        vintage_id=pl.format(
            "cesvinall:{}", pl.col("release_month").dt.strftime("%Y-%m")
        ),
        concept_regime=_regime(),
    )


def rtdsm_levels(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """The Philadelphia Fed's seasonally adjusted total nonfarm level vintages from 1979."""
    two_digit = pl.col("column_key").str.extract(r"^EMPLOY(\d{2})M", 1).cast(pl.Int32)
    month = pl.col("column_key").str.extract(r"M(\d{1,2})$", 1).cast(pl.Int32)
    levels = raw.filter(
        (pl.col("source") == "rtdsm_employ") & (pl.col("text") != "#N/A")
    ).select(
        "cell_id",
        source=pl.lit("rtdsm_employ"),
        sector=pl.lit("00"),
        reference_month=(pl.col("row_key").str.replace(":", "-") + "-01").str.to_date(
            "%Y-%m-%d"
        ),
        vintage_month=pl.date(_year(two_digit, RTDSM_YEAR_PIVOT), month, 1),
        vintage_id=pl.concat_str([pl.lit("rtdsm:"), pl.col("column_key")]),
        seasonal_status=pl.lit("SA"),
        measure=pl.lit("level"),
        value_thousands=pl.col("text").cast(pl.Int64),
        marker=pl.lit(None, dtype=pl.String),
        transformation=pl.lit("rtdsm_level"),
    )
    releases = (
        index.filter(pl.col("status") == "released")
        .select(
            release_month="reference_month",
            release_date="release_date",
            released_in=pl.col("release_date").dt.truncate("1mo"),
        )
        .sort("released_in")
    )
    vintages = (
        levels.select("vintage_month")
        .unique()
        .sort("vintage_month")
        .join_asof(
            releases,
            left_on="vintage_month",
            right_on="released_in",
            strategy="backward",
        )
        .drop_nulls("release_month")
        .drop("released_in")
    )
    return (
        levels.join(vintages, on="vintage_month", how="inner")
        .drop("vintage_month")
        .with_columns(concept_regime=_regime())
    )


def revision_table_changes(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """The revision table's total nonfarm estimates, each keyed by the release that made it."""
    missing = pl.col("marker").is_in(["A", "NA"])
    changes = (
        estimates(monthly_cells(raw))
        .filter(pl.col("marker").fill_null("") != "blank")
        .select(
            "cell_id",
            "reference_month",
            "seasonal_status",
            "release_stage",
            source=pl.lit("cesnaicsrev"),
            sector=pl.lit("00"),
            release_month=pl.col("reference_month").dt.offset_by(
                pl.col("release_stage").replace_strict(STAGE_OFFSETS)
            ),
            measure=pl.lit("change"),
            value_thousands="value",
            marker="marker",
            transformation=pl.when(missing)
            .then(pl.lit("revision_table_missing_estimate"))
            .otherwise(pl.lit("revision_table_estimate")),
        )
    )
    return _published(changes, index).with_columns(
        vintage_id=pl.format(
            "cesnaicsrev:{}", pl.col("release_month").dt.strftime("%Y-%m")
        ),
        concept_regime=_regime(),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vintage_panel.py -q`
Expected: PASS: 12 passed.

- [ ] **Step 6: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 169 passed, 5 deselected.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/panel.py tests/vintage_data.py tests/test_vintage_panel.py
git commit -m "Derive the panel's level and change rows through named transformations"
```

---

### Task 6: Stage labels and the assembled panel

**Files:**
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_stage_labels.py`
- Modify: `tests/test_vintage_panel.py` (append)
- Create: `src/ces_revisions/vintages/stages.py`
- Modify: `src/ces_revisions/vintages/panel.py` (append)

**Interfaces:**
- Consumes: `levels()`, `table_changes()`, and `panel.REGIME_SHIFT_MONTH` (Task 5); `index()` and `release_index.FIRST_REFERENCE_MONTH` (Task 3); `MONTH_NAMES`, `add_months`, and `month_range` (Task 3); and `raw.read_vintage_comments()`, `SECTORS`, and `SEASONAL_STATUSES` (Task 2).
- Produces:
  - `ces_revisions.vintages.stages`: `STAGES = ("F", "S", "T", "B", "M")`, `CLOSING_STAGES = ("F", "S", "T")`, `FIRST_SECTOR_MONTH = date(2003, 5, 1)`, `STATUSES`, and `STAGE_LABEL_COLUMNS`; `stage_release_months(reference_month: date, seasonal_status: str) -> dict[str, date]`; `comment_release_month(adjustment: str) -> date`; and `vintage_file_stage_labels(levels, index, comments)`, `revision_table_stage_labels(changes, index, comments)`, and `build_stage_labels(levels, changes, index, comments)`, each returning a `pl.DataFrame`.
  - The stage-label table: `source`, `sector`, `reference_month`, `seasonal_status`, `release_stage`, `release_month`, `vintage_id` (null for a stage the vintage files or table do not hold), `published_date` (null for a missing vintage or a release still to come), `status` (`observed`, `missing_vintage`, `beyond_frontier`, or `right_censored`), `benchmark_release`, `nonstandard_release`, `revised_after_m` (set only on observed vintage-file M rows), and `concept_regime`. The vintage files give five rows per sector, seasonal status, and reference month from May 2003, and the table three per seasonal status and reference month from January 1979.
  - `ces_revisions.vintages.panel`: `PANEL_COLUMNS` and `assemble_panel(levels, rtdsm, changes, stage_labels) -> pl.DataFrame`, the long panel, whose vintage-file levels carry the stage their vintage serves.
- Test helpers: `vintage_data.labels()` and `vintage_data.long_panel()`.

- [ ] **Step 1: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())


@cache
def levels() -> pl.DataFrame:
    return panel.vintage_file_levels(raw_values(), index())


@cache
def rtdsm() -> pl.DataFrame:
    return panel.rtdsm_levels(raw_values(), index())


@cache
def table_changes() -> pl.DataFrame:
    return panel.revision_table_changes(raw_values(), index())


@cache
def labels() -> pl.DataFrame:
    return stages.build_stage_labels(
        levels(), table_changes(), index(), raw.read_vintage_comments()
    )


@cache
def long_panel() -> pl.DataFrame:
    return panel.assemble_panel(levels(), rtdsm(), table_changes(), labels())
```

- [ ] **Step 2: Write the failing stage-label tests**

Create `tests/test_stage_labels.py`:

```python
"""Req 2's stage labels for the vintage files and for BLS's revision table."""

from datetime import date

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, stages
from ces_revisions.vintages.stages import comment_release_month, stage_release_months

UNIT = ["source", "sector", "reference_month", "seasonal_status"]
LAPSE_GAPS = {(date(2025, 9, 1), "S"), (date(2025, 10, 1), "F")}
REDESIGN_GAPS = {
    (date(2003, 3, 1), "T"),
    (date(2003, 4, 1), "S"),
    (date(2003, 4, 1), "T"),
}
COMMENTED_RELEASES = [
    date(2003, 5, 1),
    date(2008, 1, 1),
    date(2011, 1, 1),
    date(2012, 1, 1),
    date(2014, 1, 1),
    date(2016, 1, 1),
    date(2018, 1, 1),
    date(2019, 1, 1),
    date(2020, 1, 1),
    date(2023, 1, 1),
    date(2025, 1, 1),
    date(2025, 10, 1),
    date(2026, 1, 1),
]


@pytest.mark.parametrize(
    ("month", "status", "expected"),
    [
        (
            date(2024, 4, 1),
            "NSA",
            ((2024, 4), (2024, 5), (2024, 6), (2025, 1), (2026, 1)),
        ),
        (
            date(2024, 11, 1),
            "NSA",
            ((2024, 11), (2024, 12), (2025, 1), (2026, 1), (2027, 1)),
        ),
        (
            date(2024, 12, 1),
            "SA",
            ((2024, 12), (2025, 1), (2025, 2), (2026, 1), (2030, 1)),
        ),
        (
            date(2024, 1, 1),
            "SA",
            ((2024, 1), (2024, 2), (2024, 3), (2025, 1), (2030, 1)),
        ),
    ],
)
def test_benchmark_stages_count_from_the_third_estimate(month, status, expected):
    releases = stage_release_months(month, status)
    assert list(releases) == list(stages.STAGES)
    assert tuple((value.year, value.month) for value in releases.values()) == expected


@pytest.mark.parametrize(
    ("text", "month"),
    [
        (
            "With the release of May 2003 data on June 6, 2003, the CES national nonfarm",
            date(2003, 5, 1),
        ),
        (
            "With the 2024 benchmark, CES reconstructed several series.",
            date(2025, 1, 1),
        ),
        (
            "Due to the 2025 lapse in appropriations, no estimates for October first",
            date(2025, 10, 1),
        ),
    ],
)
def test_comment_entries_name_their_release(text, month):
    assert comment_release_month(text) == month


def test_a_comment_without_a_release_rule_is_an_error():
    with pytest.raises(ValueError, match="no release rule"):
        comment_release_month("Historical revisions, reconstructions, or adjustments")


# --- The committed sources ------------------------------------------------------------------


def test_the_committed_comments_describe_thirteen_releases():
    comments = raw.read_vintage_comments()
    assert comments.height == 16
    months = {comment_release_month(text) for text in comments["adjustment"]}
    assert sorted(months) == COMMENTED_RELEASES


def test_every_unit_carries_each_of_its_stages_once_on_distinct_releases():
    labels = vintage_data.labels()
    assert labels.select(*UNIT, "release_stage").n_unique() == labels.height
    units = labels.group_by(UNIT).agg(
        stages=pl.col("release_stage").sort().str.join(""),
        releases=pl.col("release_month").n_unique(),
    )
    expected = {"cesvinall": ("BFMST", 5), "cesnaicsrev": ("FST", 3)}
    for source, (names, count) in expected.items():
        frame = units.filter(pl.col("source") == source)
        assert frame["stages"].unique().to_list() == [names]
        assert frame["releases"].unique().to_list() == [count]


def test_missing_vintages_are_the_2025_lapse_and_the_2003_redesign():
    missing = vintage_data.labels().filter(pl.col("status") == "missing_vintage")
    files = missing.filter(pl.col("source") == "cesvinall")
    assert (
        set(files.select("reference_month", "release_stage").iter_rows()) == LAPSE_GAPS
    )
    assert files.height == len(LAPSE_GAPS) * 12 * 2
    assert files["published_date"].null_count() == files.height
    table = missing.filter(pl.col("source") == "cesnaicsrev")
    gaps = LAPSE_GAPS | REDESIGN_GAPS
    assert set(table.select("reference_month", "release_stage").iter_rows()) == gaps
    assert table.height == len(gaps) * 2


def test_right_censoring_and_the_frontier_follow_release_timing():
    labels = vintage_data.labels()
    latest = vintage_data.index()["reference_month"].max()
    frontier = vintage_data.levels()["release_month"].max()
    # cesvinall.zip of 2026-03-06 ends with the January 2026 benchmark release.
    assert frontier == date(2026, 1, 1)
    for source, last_held in (("cesvinall", frontier), ("cesnaicsrev", latest)):
        frame = labels.filter(pl.col("source") == source)
        censored = frame.filter(pl.col("status") == "right_censored")
        assert censored.height == frame.filter(pl.col("release_month") > latest).height
        assert (censored["release_month"] > latest).all()
        beyond = frame.filter(pl.col("status") == "beyond_frontier")
        held_later = frame.filter(pl.col("release_month").is_between(last_held, latest))
        assert (
            beyond.height
            == held_later.height
            - held_later.filter(pl.col("release_month") == last_held).height
        )


def test_m_is_right_censored_exactly_where_its_benchmark_release_is_still_to_come():
    latest = vintage_data.index()["reference_month"].max()
    mature = vintage_data.labels().filter(pl.col("release_stage") == "M")
    censored = mature["status"] == "right_censored"
    assert (censored == (mature["release_month"] > latest)).all()


def test_every_stage_is_an_employment_situation_release_so_no_preliminary_benchmark_is():
    labels = vintage_data.labels()
    assert set(labels["release_stage"].unique()) == set(stages.STAGES)
    releases = vintage_data.index().select(
        "publication", release_month="reference_month"
    )
    held = labels.filter(pl.col("status") != "right_censored").join(
        releases, on="release_month", how="left"
    )
    assert held["publication"].unique().to_list() == ["employment_situation"]


def test_no_vintage_file_stage_precedes_the_may_2003_publication_vintage():
    files = vintage_data.labels().filter(pl.col("source") == "cesvinall")
    assert files["reference_month"].min() == date(2003, 5, 1)
    assert files["release_month"].min() == date(2003, 5, 1)


def test_benchmark_releases_carry_january_first_december_second_november_third():
    closing = vintage_data.labels().filter(
        (pl.col("source") == "cesvinall")
        & pl.col("release_stage").is_in(list(stages.CLOSING_STAGES))
        & pl.col("benchmark_release")
    )
    pairs = closing.select(pl.col("reference_month").dt.month(), "release_stage")
    assert set(pairs.unique().iter_rows()) == {(1, "F"), (12, "S"), (11, "T")}


def test_nonstandard_releases_are_the_releases_the_comments_describe():
    flagged = vintage_data.labels().filter(pl.col("nonstandard_release"))
    assert sorted(set(flagged["release_month"])) == COMMENTED_RELEASES


def test_revised_after_m_marks_only_observed_m_rows():
    labels = vintage_data.labels()
    marked = labels.filter(pl.col("revised_after_m").is_not_null())
    assert set(
        marked.select("source", "release_stage", "status").unique().iter_rows()
    ) == {("cesvinall", "M", "observed")}
    observed_m = labels.filter(
        (pl.col("release_stage") == "M") & (pl.col("status") == "observed")
    )
    assert marked.height == observed_m.height


def test_panel_levels_carry_the_stage_their_vintage_serves():
    staged = vintage_data.long_panel().filter(
        (pl.col("source") == "cesvinall") & pl.col("release_stage").is_not_null()
    )
    labeled = vintage_data.labels().filter(
        (pl.col("source") == "cesvinall") & pl.col("vintage_id").is_not_null()
    )
    assert staged.height == labeled.height
```

- [ ] **Step 3: Append the failing assembled-panel test**

Append to `tests/test_vintage_panel.py`, after two blank lines:

```python
# --- Task 6: the assembled panel ------------------------------------------------------------


def test_the_assembled_panel_holds_every_source_row_once_under_req_1_columns():
    frame = vintage_data.long_panel()
    assert frame.columns == panel.PANEL_COLUMNS
    assert frame.select(KEY).n_unique() == frame.height
    assert frame.height == sum(source.height for source in sources().values())
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stage_labels.py tests/test_vintage_panel.py -q`
Expected: FAIL. Collection stops in both modules with `ImportError: cannot import name 'stages' from 'ces_revisions.vintages'`.

- [ ] **Step 5: Write the stage labels**

Create `src/ces_revisions/vintages/stages.py`:

```python
"""Stage labels: each reference month's F, S, T, B, and M vintages under Req 2.

F, S, and T are the releases zero, one, and two months after a reference month on the release
clock. The vintage files keep a row for every month's release, the canceled October 2025 one
included, and BLS's revision table counts estimates the same way. B is the first benchmark
release after T, and M, for not seasonally adjusted values, the second; benchmark releases
carry January estimates from 2004. Counting from T gives every vintage one stage: November's
third estimate and December's second arrive in a benchmark release, which `benchmark_release`
marks. Seasonally adjusted M is the January release six years after the reference year, the
first benchmark release after the month leaves BLS's five-year seasonal revision window.

A stage whose release is still to come is right-censored, a release after the vintage files'
last row is beyond the frontier, and a value the 2025 lapse left unpublished, or an estimate
the revision table omits for the 2003 redesign, is a missing vintage.
"""

import re
from datetime import date

import polars as pl

from ces_revisions.vintages.months import MONTH_NAMES, add_months, month_range
from ces_revisions.vintages.panel import REGIME_SHIFT_MONTH
from ces_revisions.vintages.raw import SEASONAL_STATUSES, SECTORS
from ces_revisions.vintages.release_index import FIRST_REFERENCE_MONTH

STAGES = ("F", "S", "T", "B", "M")
CLOSING_STAGES = ("F", "S", "T")
FIRST_SECTOR_MONTH = date(2003, 5, 1)
STATUSES = ("observed", "missing_vintage", "beyond_frontier", "right_censored")
STAGE_LABEL_COLUMNS = [
    "source",
    "sector",
    "reference_month",
    "seasonal_status",
    "release_stage",
    "release_month",
    "vintage_id",
    "published_date",
    "status",
    "benchmark_release",
    "nonstandard_release",
    "revised_after_m",
    "concept_regime",
]
_KEYS = ["sector", "reference_month", "seasonal_status"]
_RELEASE_OF = re.compile(
    r"^With the release of (" + "|".join(MONTH_NAMES) + r") (\d{4}) data on"
)
_BENCHMARK_OF = re.compile(r"^With the (\d{4}) benchmark")
_LAPSE = re.compile(r"^Due to the 2025 lapse in appropriations")


def stage_release_months(
    reference_month: date, seasonal_status: str
) -> dict[str, date]:
    """The release month of each stage, as the reference month that release first estimates."""
    third = add_months(reference_month, 2)
    mature = (
        date(third.year + 2, 1, 1)
        if seasonal_status == "NSA"
        else date(reference_month.year + 6, 1, 1)
    )
    return {
        "F": reference_month,
        "S": add_months(reference_month, 1),
        "T": third,
        "B": date(third.year + 1, 1, 1),
        "M": mature,
    }


def comment_release_month(adjustment: str) -> date:
    """The release a Comments-sheet entry describes, read from its opening words."""
    if match := _RELEASE_OF.match(adjustment):
        return date(int(match.group(2)), MONTH_NAMES.index(match.group(1)) + 1, 1)
    if match := _BENCHMARK_OF.match(adjustment):
        return date(int(match.group(1)) + 1, 1, 1)
    if _LAPSE.match(adjustment):
        return date(2025, 10, 1)
    raise ValueError(f"no release rule for the comment {adjustment[:60]!r}")


def _calendar(months: list[date], sectors, stages) -> pl.DataFrame:
    records = [
        (sector, month, status, stage, stage_release_months(month, status)[stage])
        for sector in sectors
        for status in SEASONAL_STATUSES
        for month in months
        for stage in stages
    ]
    return pl.DataFrame(
        records,
        schema={
            "sector": pl.String,
            "reference_month": pl.Date,
            "seasonal_status": pl.String,
            "release_stage": pl.String,
            "release_month": pl.Date,
        },
        orient="row",
    )


def _finish(
    labeled: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    commented = sorted({comment_release_month(text) for text in comments["adjustment"]})
    releases = index.select(
        "published_date", "benchmark_release", release_month="reference_month"
    )
    return (
        labeled.join(releases, on="release_month", how="left")
        .with_columns(
            published_date=pl.when(pl.col("status") == "missing_vintage")
            .then(pl.lit(None, dtype=pl.Date))
            .otherwise("published_date"),
            nonstandard_release=pl.col("release_month").is_in(commented),
            concept_regime=pl.when(pl.col("release_month") < REGIME_SHIFT_MONTH)
            .then(pl.lit("pre_2003_05"))
            .otherwise(pl.lit("from_2003_05")),
        )
        .select(STAGE_LABEL_COLUMNS)
    )


def vintage_file_stage_labels(
    levels: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    """Five stage rows per sector, seasonal status, and reference month from May 2003."""
    latest = index["reference_month"].max()
    frontier = levels["release_month"].max()
    calendar = _calendar(
        month_range(FIRST_SECTOR_MONTH, latest), SECTORS.values(), STAGES
    )
    cells = levels.select(*_KEYS, "release_month", "vintage_id", "value_thousands")
    labeled = calendar.join(
        cells, on=[*_KEYS, "release_month"], how="left"
    ).with_columns(
        source=pl.lit("cesvinall"),
        status=pl.when(pl.col("release_month") > latest)
        .then(pl.lit("right_censored"))
        .when(pl.col("release_month") > frontier)
        .then(pl.lit("beyond_frontier"))
        .when(pl.col("value_thousands").is_null())
        .then(pl.lit("missing_vintage"))
        .otherwise(pl.lit("observed")),
    )
    mature = labeled.filter(
        (pl.col("release_stage") == "M") & (pl.col("status") == "observed")
    ).select(*_KEYS, mature_month="release_month", mature_value="value_thousands")
    later = (
        levels.join(mature, on=_KEYS, how="inner")
        .filter(pl.col("release_month") > pl.col("mature_month"))
        .group_by(_KEYS)
        .agg(revised=(pl.col("value_thousands") != pl.col("mature_value")).any())
    )
    revised = mature.join(later, on=_KEYS, how="left").select(
        *_KEYS,
        release_stage=pl.lit("M"),
        revised_after_m=pl.col("revised").fill_null(False),
    )
    return _finish(
        labeled.join(revised, on=[*_KEYS, "release_stage"], how="left"), index, comments
    )


def revision_table_stage_labels(
    changes: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    """Three stage rows per seasonal status and reference month of the revision table."""
    latest = index["reference_month"].max()
    calendar = _calendar(
        month_range(FIRST_REFERENCE_MONTH, latest), ["00"], CLOSING_STAGES
    )
    values = changes.select(*_KEYS, "release_stage", "vintage_id", "value_thousands")
    labeled = calendar.join(
        values, on=[*_KEYS, "release_stage"], how="left"
    ).with_columns(
        source=pl.lit("cesnaicsrev"),
        revised_after_m=pl.lit(None, dtype=pl.Boolean),
        status=pl.when(pl.col("release_month") > latest)
        .then(pl.lit("right_censored"))
        .when(pl.col("value_thousands").is_null())
        .then(pl.lit("missing_vintage"))
        .otherwise(pl.lit("observed")),
    )
    return _finish(labeled, index, comments)


def build_stage_labels(
    levels: pl.DataFrame,
    changes: pl.DataFrame,
    index: pl.DataFrame,
    comments: pl.DataFrame,
) -> pl.DataFrame:
    return pl.concat(
        [
            vintage_file_stage_labels(levels, index, comments),
            revision_table_stage_labels(changes, index, comments),
        ]
    ).sort("source", "sector", "seasonal_status", "reference_month", "release_stage")
```

- [ ] **Step 6: Append the panel assembly**

Append to `src/ces_revisions/vintages/panel.py`, after two blank lines:

```python
PANEL_COLUMNS = [
    "source",
    "sector",
    "reference_month",
    "release_month",
    "vintage_id",
    "seasonal_status",
    "measure",
    "release_date",
    "release_stage",
    "value_thousands",
    "marker",
    "concept_regime",
    "cell_id",
    "transformation",
]


def assemble_panel(
    levels: pl.DataFrame,
    rtdsm: pl.DataFrame,
    changes: pl.DataFrame,
    stage_labels: pl.DataFrame,
) -> pl.DataFrame:
    """The long panel: the three sources, with vintage-file levels labeled by stage."""
    stages = stage_labels.filter(
        (pl.col("source") == "cesvinall") & pl.col("vintage_id").is_not_null()
    ).select(
        "sector", "reference_month", "seasonal_status", "vintage_id", "release_stage"
    )
    labeled = levels.join(
        stages,
        on=["sector", "reference_month", "seasonal_status", "vintage_id"],
        how="left",
    )
    return pl.concat(
        [
            labeled.select(PANEL_COLUMNS),
            rtdsm.with_columns(release_stage=pl.lit(None, dtype=pl.String)).select(
                PANEL_COLUMNS
            ),
            changes.select(PANEL_COLUMNS),
        ]
    ).sort("source", "sector", "seasonal_status", "reference_month", "vintage_id")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stage_labels.py tests/test_vintage_panel.py -q`
Expected: PASS: 32 passed.

- [ ] **Step 8: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 189 passed, 5 deselected.

- [ ] **Step 9: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/stages.py src/ces_revisions/vintages/panel.py tests/vintage_data.py tests/test_stage_labels.py tests/test_vintage_panel.py
git commit -m "Label each reference month's F, S, T, B, and M vintages and assemble the long panel"
```

---

### Task 7: Same-release differencing and its reconciliation with the revision table

**Files:**
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_differencing.py`
- Create: `src/ces_revisions/vintages/differencing.py`

**Interfaces:**
- Consumes: `levels()` (Task 5); `labels()` and `stages.FIRST_SECTOR_MONTH` (Task 6); and `table_estimates()`, `table_cells()`, `REVISION_TERMS`, and `STAGE_OF_ESTIMATE` (Task 4).
- Produces, in `ces_revisions.vintages.differencing`:
  - `UNCOMPARED = ("beyond_frontier", "right_censored")`.
  - `same_release_changes(levels) -> pl.DataFrame`: `source`, `sector`, `seasonal_status`, `vintage_id`, `release_month`, `reference_month`, and `change_thousands`.
  - `stage_changes(changes, stage_labels) -> pl.DataFrame`: `source`, `sector`, `reference_month`, `seasonal_status`, `release_stage`, `release_month`, `vintage_id`, `status`, and `change_thousands`, one row per vintage-file stage label.
  - `reconcile(stage_changes, table_estimates) -> pl.DataFrame`: `reference_month`, `seasonal_status`, `release_stage`, `status`, `panel_thousands`, `table_thousands`, `table_marker`, and `outcome`, one of `reproduced`, `differs`, `missing_in_both`, `beyond_frontier`, or `right_censored`.
  - `revision_reconciliation(stage_changes, table_cells) -> pl.DataFrame`: `reference_month`, `seasonal_status`, `column`, `panel_thousands`, `table_thousands`, `table_marker`, and `outcome` (`reproduced` or `differs`).
- Test helper: `vintage_data.stage_changes()`.

- [ ] **Step 1: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    differencing,
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)
from ces_revisions.vintages.months import add_months


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())


@cache
def levels() -> pl.DataFrame:
    return panel.vintage_file_levels(raw_values(), index())


@cache
def rtdsm() -> pl.DataFrame:
    return panel.rtdsm_levels(raw_values(), index())


@cache
def table_changes() -> pl.DataFrame:
    return panel.revision_table_changes(raw_values(), index())


@cache
def labels() -> pl.DataFrame:
    return stages.build_stage_labels(
        levels(), table_changes(), index(), raw.read_vintage_comments()
    )


@cache
def long_panel() -> pl.DataFrame:
    return panel.assemble_panel(levels(), rtdsm(), table_changes(), labels())


@cache
def stage_changes() -> pl.DataFrame:
    first = add_months(stages.FIRST_SECTOR_MONTH, -1)
    within = differencing.same_release_changes(
        levels().filter(pl.col("reference_month") >= first)
    )
    return differencing.stage_changes(within, labels())
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_differencing.py`:

```python
"""Req 1's same-release differencing and its reconciliation with BLS's revision table."""

from datetime import date

import polars as pl
import vintage_data

from ces_revisions.vintages import differencing


def level(vintage: str, month: date, value: int | None) -> dict:
    return {
        "source": "cesvinall",
        "sector": "00",
        "seasonal_status": "NSA",
        "vintage_id": f"cesvinall:{vintage}",
        "release_month": date.fromisoformat(f"{vintage}-01"),
        "reference_month": month,
        "value_thousands": value,
    }


def test_changes_difference_levels_within_one_vintage_only():
    levels = pl.DataFrame(
        [
            level("2003-06", date(2003, 4, 1), 100),
            level("2003-06", date(2003, 5, 1), 130),
            level("2003-06", date(2003, 6, 1), None),
            level("2003-07", date(2003, 5, 1), 131),
            level("2003-07", date(2003, 6, 1), 140),
        ]
    )
    changes = differencing.same_release_changes(levels).sort(
        "vintage_id", "reference_month"
    )
    assert changes.select(
        "vintage_id", "reference_month", "change_thousands"
    ).rows() == [
        ("cesvinall:2003-06", date(2003, 5, 1), 30),
        ("cesvinall:2003-06", date(2003, 6, 1), None),
        ("cesvinall:2003-07", date(2003, 6, 1), 9),
    ]


# --- The committed sources ------------------------------------------------------------------


def test_same_release_changes_reproduce_every_table_estimate_within_the_frontier():
    frame = differencing.reconcile(
        vintage_data.stage_changes(), vintage_data.table_estimates()
    )
    differs = frame.filter(pl.col("outcome") == "differs").select(
        "reference_month",
        "seasonal_status",
        "release_stage",
        "panel_thousands",
        "table_thousands",
    )
    assert differs.rows() == [(date(2003, 11, 1), "NSA", "T", 147, 46)]
    both_missing = frame.filter(pl.col("outcome") == "missing_in_both")
    assert set(both_missing.select("reference_month", "release_stage").iter_rows()) == {
        (date(2025, 9, 1), "S"),
        (date(2025, 10, 1), "F"),
    }
    outcomes = dict(frame.group_by("outcome").len().iter_rows())
    assert outcomes["reproduced"] == 1627
    assert set(outcomes) <= {
        "reproduced",
        "differs",
        "missing_in_both",
        "beyond_frontier",
        "right_censored",
    }


def test_the_table_takes_november_2003_third_estimate_across_two_releases():
    """November 2003's third estimate arrived in the February 2004 benchmark release. BLS's
    table differences it against October 2003 in the January 2004 release; Req 1's
    same-release change uses October in the February release."""
    levels = {
        (row["vintage_id"], row["reference_month"]): row["value_thousands"]
        for row in vintage_data.levels()
        .filter(
            (pl.col("sector") == "00")
            & (pl.col("seasonal_status") == "NSA")
            & pl.col("reference_month").is_between(date(2003, 10, 1), date(2003, 11, 1))
        )
        .iter_rows(named=True)
    }
    november = levels[("cesvinall:2004-01", date(2003, 11, 1))]
    assert november - levels[("cesvinall:2004-01", date(2003, 10, 1))] == 147
    assert november - levels[("cesvinall:2003-12", date(2003, 10, 1))] == 46


def test_panel_revisions_reproduce_the_table_revisions_but_three_cells():
    frame = differencing.revision_reconciliation(
        vintage_data.stage_changes(), vintage_data.table_cells()
    )
    differs = frame.filter(pl.col("outcome") == "differs")
    assert sorted(
        differs.select("reference_month", "seasonal_status", "column").iter_rows()
    ) == [
        (date(2003, 11, 1), "NSA", "3rd_minus_1st"),
        (date(2003, 11, 1), "NSA", "3rd_minus_2nd"),
        (date(2023, 9, 1), "SA", "3rd_minus_2nd"),
    ]
    assert frame.filter(pl.col("outcome") == "reproduced").height == 1617
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_differencing.py -q`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'differencing' from 'ces_revisions.vintages'`.

- [ ] **Step 4: Write the differencing module**

Create `src/ces_revisions/vintages/differencing.py`:

```python
"""The same-release differencing operator of Req 1, and its reconciliation with BLS's table.

An over-the-month change is a month's level minus the prior month's level in the same release
file; levels from different vintages are never differenced, and stage labels are never
differenced across release files. Reconciliation sets each total nonfarm estimate in the
revision table beside the same-release change at its stage.
"""

import polars as pl

from ces_revisions.vintages.revision_table import REVISION_TERMS, STAGE_OF_ESTIMATE

_VINTAGE = ["source", "sector", "seasonal_status", "vintage_id", "release_month"]
_CELL = ["reference_month", "seasonal_status", "release_stage"]
UNCOMPARED = ("beyond_frontier", "right_censored")


def same_release_changes(levels: pl.DataFrame) -> pl.DataFrame:
    """Each month's level minus the prior month's in the same vintage; missing if either is."""
    prior = levels.select(
        *_VINTAGE,
        reference_month=pl.col("reference_month").dt.offset_by("1mo"),
        prior_thousands="value_thousands",
    )
    return (
        levels.select(*_VINTAGE, "reference_month", "value_thousands")
        .join(prior, on=[*_VINTAGE, "reference_month"], how="inner")
        .select(
            *_VINTAGE,
            "reference_month",
            change_thousands=pl.col("value_thousands") - pl.col("prior_thousands"),
        )
    )


def stage_changes(changes: pl.DataFrame, stage_labels: pl.DataFrame) -> pl.DataFrame:
    """The same-release change at every vintage-file stage label."""
    return (
        stage_labels.filter(pl.col("source") == "cesvinall")
        .join(
            changes.select(
                "sector",
                "seasonal_status",
                "vintage_id",
                "reference_month",
                "change_thousands",
            ),
            on=["sector", "seasonal_status", "vintage_id", "reference_month"],
            how="left",
        )
        .select(
            "source",
            "sector",
            "reference_month",
            "seasonal_status",
            "release_stage",
            "release_month",
            "vintage_id",
            "status",
            "change_thousands",
        )
    )


def reconcile(
    stage_changes: pl.DataFrame, table_estimates: pl.DataFrame
) -> pl.DataFrame:
    """Each total nonfarm F, S, and T change from May 2003 beside the table's estimate."""
    panel = stage_changes.filter(
        (pl.col("sector") == "00") & pl.col("release_stage").is_in(list("FST"))
    ).select(*_CELL, "status", panel_thousands="change_thousands")
    table = table_estimates.select(
        *_CELL, table_thousands="value", table_marker="marker"
    )
    return panel.join(table, on=_CELL, how="left").with_columns(
        outcome=pl.when(pl.col("status").is_in(UNCOMPARED))
        .then("status")
        .when(pl.col("panel_thousands").is_null() & pl.col("table_thousands").is_null())
        .then(pl.lit("missing_in_both"))
        .when(pl.col("panel_thousands") == pl.col("table_thousands"))
        .then(pl.lit("reproduced"))
        .otherwise(pl.lit("differs"))
    )


def revision_reconciliation(
    stage_changes: pl.DataFrame, table_cells: pl.DataFrame
) -> pl.DataFrame:
    """Each total nonfarm revision between observed stages beside the table's revision cell."""
    wide = stage_changes.filter(
        (pl.col("sector") == "00")
        & pl.col("release_stage").is_in(list("FST"))
        & (pl.col("status") == "observed")
    ).pivot(
        on="release_stage",
        index=["reference_month", "seasonal_status"],
        values="change_thousands",
    )
    panel = pl.concat(
        [
            wide.select(
                "reference_month",
                "seasonal_status",
                column=pl.lit(column),
                panel_thousands=pl.col(STAGE_OF_ESTIMATE[later])
                - pl.col(STAGE_OF_ESTIMATE[earlier]),
            )
            for column, (later, earlier) in REVISION_TERMS.items()
        ]
    ).drop_nulls("panel_thousands")
    table = table_cells.filter(pl.col("column").is_in(list(REVISION_TERMS))).select(
        "reference_month",
        "seasonal_status",
        "column",
        table_thousands="value",
        table_marker="marker",
    )
    return panel.join(
        table, on=["reference_month", "seasonal_status", "column"], how="left"
    ).with_columns(
        outcome=pl.when(pl.col("panel_thousands") == pl.col("table_thousands"))
        .then(pl.lit("reproduced"))
        .otherwise(pl.lit("differs"))
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_differencing.py -q`
Expected: PASS: 4 passed.

- [ ] **Step 6: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 193 passed, 5 deselected.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/differencing.py tests/vintage_data.py tests/test_differencing.py
git commit -m "Difference levels within each release and reconcile them with BLS's revision table"
```

---

### Task 8: The Req 9 accounting decomposition

**Files:**
- Modify: `tests/vintage_data.py` (replace)
- Test: `tests/test_accounting.py`
- Create: `src/ces_revisions/vintages/accounting.py`

**Interfaces:**
- Consumes: `stage_changes()` (Task 7) and `table_estimates()` (Task 4).
- Produces, in `ces_revisions.vintages.accounting`:
  - `TRANSITIONS = (("F", "S"), ("S", "T"), ("T", "B"))` and `REGIMES`, Req 18's regimes by reference month: `pre_2003` through April 2003, `2003_2019` from May 2003, `2020_2022`, and `2023_present`.
  - `accounting_changes(stage_changes, table_estimates) -> pl.DataFrame`: `source`, `sector`, `reference_month`, `seasonal_status`, `release_stage`, and `change_thousands`, from the vintage files for reference months from May 2003 and from the revision table before.
  - `identity_terms(changes) -> pl.DataFrame`: `source`, `sector`, `reference_month`, `nsa_start`, `sa_start`, `nsa_end`, `sa_end`, `transition`, `revision_nsa`, `revision_sa`, `adjustment_change`, `regime`, and `identity_residual`.
  - `decomposition(terms) -> pl.DataFrame`: `source`, `sector`, `transition`, `regime`, `months`, `var_revision_sa`, `var_revision_nsa`, `var_adjustment_change`, `cov_revision_nsa_adjustment_change`, and `variance_identity_gap`.
- Test helper: `vintage_data.identity_terms()`.

- [ ] **Step 1: Extend the session cache**

Replace `tests/vintage_data.py` with:

```python
"""Stage 3 artifacts, each built at most once per test session from the sources in data/raw/."""

from functools import cache

import polars as pl

from ces_revisions.vintages import (
    accounting,
    differencing,
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)
from ces_revisions.vintages.months import add_months


@cache
def raw_values() -> pl.DataFrame:
    return raw.raw_values()


@cache
def index() -> pl.DataFrame:
    return release_index.build_release_index()


@cache
def table_cells() -> pl.DataFrame:
    return revision_table.monthly_cells(raw_values())


@cache
def table_estimates() -> pl.DataFrame:
    return revision_table.estimates(table_cells())


@cache
def levels() -> pl.DataFrame:
    return panel.vintage_file_levels(raw_values(), index())


@cache
def rtdsm() -> pl.DataFrame:
    return panel.rtdsm_levels(raw_values(), index())


@cache
def table_changes() -> pl.DataFrame:
    return panel.revision_table_changes(raw_values(), index())


@cache
def labels() -> pl.DataFrame:
    return stages.build_stage_labels(
        levels(), table_changes(), index(), raw.read_vintage_comments()
    )


@cache
def long_panel() -> pl.DataFrame:
    return panel.assemble_panel(levels(), rtdsm(), table_changes(), labels())


@cache
def stage_changes() -> pl.DataFrame:
    first = add_months(stages.FIRST_SECTOR_MONTH, -1)
    within = differencing.same_release_changes(
        levels().filter(pl.col("reference_month") >= first)
    )
    return differencing.stage_changes(within, labels())


@cache
def identity_terms() -> pl.DataFrame:
    return accounting.identity_terms(
        accounting.accounting_changes(stage_changes(), table_estimates())
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_accounting.py`:

```python
"""Req 9's accounting identity and the variance decomposition of published revisions."""

from datetime import date

import polars as pl
import vintage_data

from ces_revisions.vintages import accounting


def change(stage: str, status: str, value: int) -> dict:
    return {
        "source": "cesvinall",
        "sector": "00",
        "reference_month": date(2010, 3, 1),
        "seasonal_status": status,
        "release_stage": stage,
        "change_thousands": value,
    }


def test_identity_terms_split_the_adjusted_revision_into_its_parts():
    changes = pl.DataFrame(
        [
            change("F", "NSA", 300),
            change("F", "SA", 120),
            change("S", "NSA", 340),
            change("S", "SA", 150),
        ]
    )
    terms = accounting.identity_terms(changes)
    assert terms.select(
        "transition",
        "revision_nsa",
        "revision_sa",
        "adjustment_change",
        "identity_residual",
        "regime",
    ).rows() == [("F_S", 40, 30, 10, 0, "2003_2019")]


# --- The committed sources ------------------------------------------------------------------


def test_the_identity_holds_exactly_on_every_same_release_pair():
    terms = vintage_data.identity_terms()
    assert terms["identity_residual"].abs().max() == 0
    assert set(terms.select("source", "transition").unique().iter_rows()) == {
        ("cesvinall", "F_S"),
        ("cesvinall", "S_T"),
        ("cesvinall", "T_B"),
        ("cesnaicsrev", "F_S"),
        ("cesnaicsrev", "S_T"),
    }


def test_the_variance_decomposition_closes_to_float_precision():
    table = accounting.decomposition(vintage_data.identity_terms())
    scale = table["var_revision_sa"].max()
    assert table["variance_identity_gap"].abs().max() <= 1e-9 * scale


def test_the_decomposition_covers_every_sector_transition_and_regime():
    table = accounting.decomposition(vintage_data.identity_terms())
    files = table.filter(pl.col("source") == "cesvinall")
    assert files.height == 12 * 3 * 3
    assert set(files["regime"]) == {"2003_2019", "2020_2022", "2023_present"}
    table_rows = table.filter(pl.col("source") == "cesnaicsrev")
    assert table_rows.select("sector", "transition", "regime").rows() == [
        ("00", "F_S", "pre_2003"),
        ("00", "S_T", "pre_2003"),
    ]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_accounting.py -q`
Expected: FAIL. Collection stops with `ImportError: cannot import name 'accounting' from 'ces_revisions.vintages'`.

- [ ] **Step 4: Write the accounting module**

Create `src/ces_revisions/vintages/accounting.py`:

```python
"""The Req 9 accounting identity, and the variance decomposition of published revisions.

For same-release changes d^N and d^S at stages j and k of one reference month, with implied
adjustment a = d^N - d^S, the seasonally adjusted revision is the unadjusted revision less the
change in implied adjustment, r^S = r^N - (a_k - a_j). Within any group of reference months,
therefore, Var(r^S) = Var(r^N) + Var(Δa) - 2 Cov(r^N, Δa) exactly.
"""

from datetime import date

import polars as pl

TRANSITIONS = (("F", "S"), ("S", "T"), ("T", "B"))
# Req 18's regimes, by reference month; the pre-2003 regime is the revision table's alone.
REGIMES = (
    ("pre_2003", date(1979, 1, 1), date(2003, 4, 1)),
    ("2003_2019", date(2003, 5, 1), date(2019, 12, 1)),
    ("2020_2022", date(2020, 1, 1), date(2022, 12, 1)),
    ("2023_present", date(2023, 1, 1), date(9999, 12, 1)),
)
_UNIT = ["source", "sector", "reference_month"]


def accounting_changes(
    stage_changes: pl.DataFrame, table_estimates: pl.DataFrame
) -> pl.DataFrame:
    """Observed stage changes: vintage files from May 2003, the table's before May 2003."""
    files = stage_changes.filter(
        (pl.col("status") == "observed") & pl.col("release_stage").is_in(list("FSTB"))
    ).select(*_UNIT, "seasonal_status", "release_stage", "change_thousands")
    table = table_estimates.filter(
        (pl.col("reference_month") < REGIMES[1][1]) & pl.col("value").is_not_null()
    ).select(
        source=pl.lit("cesnaicsrev"),
        sector=pl.lit("00"),
        reference_month="reference_month",
        seasonal_status="seasonal_status",
        release_stage="release_stage",
        change_thousands="value",
    )
    return pl.concat([files, table])


def _regime() -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for name, first, last in reversed(REGIMES):
        expression = (
            pl.when(pl.col("reference_month").is_between(first, last))
            .then(pl.lit(name))
            .otherwise(expression)
        )
    return expression


def identity_terms(changes: pl.DataFrame) -> pl.DataFrame:
    """One row per unit and transition whose two stages are observed seasonally adjusted and not."""
    wide = changes.pivot(
        on="seasonal_status", index=[*_UNIT, "release_stage"], values="change_thousands"
    )
    parts = []
    for earlier, later in TRANSITIONS:
        start = wide.filter(pl.col("release_stage") == earlier).select(
            *_UNIT, nsa_start="NSA", sa_start="SA"
        )
        end = wide.filter(pl.col("release_stage") == later).select(
            *_UNIT, nsa_end="NSA", sa_end="SA"
        )
        parts.append(
            start.join(end, on=_UNIT, how="inner").with_columns(
                transition=pl.lit(f"{earlier}_{later}")
            )
        )
    return (
        pl.concat(parts)
        .drop_nulls(["nsa_start", "sa_start", "nsa_end", "sa_end"])
        .with_columns(
            revision_nsa=pl.col("nsa_end") - pl.col("nsa_start"),
            revision_sa=pl.col("sa_end") - pl.col("sa_start"),
            adjustment_change=(pl.col("nsa_end") - pl.col("sa_end"))
            - (pl.col("nsa_start") - pl.col("sa_start")),
            regime=_regime(),
        )
        .with_columns(
            identity_residual=pl.col("revision_sa")
            - (pl.col("revision_nsa") - pl.col("adjustment_change"))
        )
    )


def decomposition(terms: pl.DataFrame) -> pl.DataFrame:
    """Variances of the identity's terms by source, sector, transition, and regime."""
    return (
        terms.group_by("source", "sector", "transition", "regime")
        .agg(
            months=pl.len(),
            var_revision_sa=pl.col("revision_sa").var(),
            var_revision_nsa=pl.col("revision_nsa").var(),
            var_adjustment_change=pl.col("adjustment_change").var(),
            cov_revision_nsa_adjustment_change=pl.cov(
                "revision_nsa", "adjustment_change"
            ),
        )
        .with_columns(
            variance_identity_gap=pl.col("var_revision_sa")
            - (
                pl.col("var_revision_nsa")
                + pl.col("var_adjustment_change")
                - 2 * pl.col("cov_revision_nsa_adjustment_change")
            )
        )
        .sort("source", "sector", "transition", "regime")
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_accounting.py -q`
Expected: PASS: 4 passed.

- [ ] **Step 6: Check the whole tree**

Run: `uv run ruff format --check . && uv run ruff check && uv run pytest -m "not slow and not network" -q`
Expected: ruff is clean, and the fast tier passes: 197 passed, 5 deselected.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src tests
uv run ruff check src tests
git add src/ces_revisions/vintages/accounting.py tests/vintage_data.py tests/test_accounting.py
git commit -m "Compute the Req 9 accounting identity and its variance decomposition"
```

---

### Task 9: The build, the build command, documentation, and final verification

**Files:**
- Test: `tests/test_vintage_build.py`
- Create: `src/ces_revisions/vintages/build.py`
- Modify: `scripts/vintage_sources.py` (replace)
- Modify: `CLAUDE.md` (Purpose, Layout and tooling, Commands)
- Modify: `README.md` (Status, Getting started)

**Interfaces:**
- Consumes: every earlier module.
- Produces:
  - `ces_revisions.vintages.build`: `VintageBuild`, a frozen dataclass of `raw_values`, `transformations`, `release_index`, `panel`, `stage_labels`, `stage_changes`, `reconciliation`, `revision_reconciliation`, `average_checks`, and `accounting_decomposition`, all `pl.DataFrame`; `build(raw_dir: Path = RAW_DIR) -> VintageBuild`; and `write(result: VintageBuild, out_dir: Path = PANEL_DIR) -> dict[str, dict]`, which writes `<field>.parquet` for each frame and a `manifest.json` of row counts and content hashes.
  - `scripts/vintage_sources.py build`, which writes `data/panel/`, and `build_panel() -> int`.

- [ ] **Step 1: Write the failing build test**

Create `tests/test_vintage_build.py`:

```python
"""The Stage 3 build writes every artifact from the committed sources, with a manifest."""

import json
from dataclasses import fields

import polars as pl
import pytest

from ces_revisions.vintages.build import VintageBuild, build, write


@pytest.mark.slow
def test_build_writes_every_artifact_and_its_manifest(tmp_path):
    result = build()
    manifest = write(result, tmp_path)
    assert (
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    for field in fields(VintageBuild):
        frame = getattr(result, field.name)
        assert manifest[field.name]["rows"] == frame.height
        written = pl.read_parquet(tmp_path / f"{field.name}.parquet")
        assert written.equals(frame), field.name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_vintage_build.py -q -m slow`
Expected: FAIL. Collection stops with `ModuleNotFoundError: No module named 'ces_revisions.vintages.build'`.

- [ ] **Step 3: Write the build**

Create `src/ces_revisions/vintages/build.py`:

```python
"""Build every Stage 3 artifact from the committed sources, and write them to data/panel/."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.vintages import (
    accounting,
    differencing,
    panel,
    raw,
    release_index,
    revision_table,
    stages,
)
from ces_revisions.vintages.months import add_months


@dataclass(frozen=True)
class VintageBuild:
    raw_values: pl.DataFrame
    transformations: pl.DataFrame
    release_index: pl.DataFrame
    panel: pl.DataFrame
    stage_labels: pl.DataFrame
    stage_changes: pl.DataFrame
    reconciliation: pl.DataFrame
    revision_reconciliation: pl.DataFrame
    average_checks: pl.DataFrame
    accounting_decomposition: pl.DataFrame


def build(raw_dir: Path = raw.RAW_DIR) -> VintageBuild:
    values = raw.raw_values(raw_dir)
    index = release_index.build_release_index(raw_dir)
    cells = revision_table.monthly_cells(values)
    table_estimates = revision_table.estimates(cells)
    levels = panel.vintage_file_levels(values, index)
    changes = panel.revision_table_changes(values, index)
    labels = stages.build_stage_labels(
        levels, changes, index, raw.read_vintage_comments(raw_dir)
    )
    first = add_months(stages.FIRST_SECTOR_MONTH, -1)
    within = differencing.same_release_changes(
        levels.filter(pl.col("reference_month") >= first)
    )
    at_stages = differencing.stage_changes(within, labels)
    terms = accounting.identity_terms(
        accounting.accounting_changes(at_stages, table_estimates)
    )
    return VintageBuild(
        raw_values=values,
        transformations=panel.TRANSFORMATIONS,
        release_index=index,
        panel=panel.assemble_panel(
            levels, panel.rtdsm_levels(values, index), changes, labels
        ),
        stage_labels=labels,
        stage_changes=at_stages,
        reconciliation=differencing.reconcile(at_stages, table_estimates),
        revision_reconciliation=differencing.revision_reconciliation(at_stages, cells),
        average_checks=revision_table.average_checks(values),
        accounting_decomposition=accounting.decomposition(terms),
    )


def write(result: VintageBuild, out_dir: Path = raw.PANEL_DIR) -> dict[str, dict]:
    """Write each frame as parquet, with a manifest of row counts and content hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for field in fields(result):
        frame = getattr(result, field.name)
        frame.write_parquet(out_dir / f"{field.name}.parquet")
        manifest[field.name] = {
            "rows": frame.height,
            "sha256": raw.content_sha256(frame),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
```

- [ ] **Step 4: Add the build command**

Replace `scripts/vintage_sources.py` with:

```python
"""Fetch roadmap Stage 3's source files into data/raw/, or build the vintage panel from them.

    uv run python scripts/vintage_sources.py fetch   # network: refresh data/raw/ and its manifest
    uv run python scripts/vintage_sources.py build   # offline: write data/panel/ from data/raw/

BLS keeps one current copy of each file and overwrites it in place, so the committed files are
the raw archive Req 1 asks for. data/raw/manifest.csv records each file's origin, SHA-256, size,
and Last-Modified header. The 27 MB workbook holding the vintage files' Comments sheet stays in
the gitignored data/cache/; its entries are committed as a CSV beside the workbook's hash.
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import archive_inventory
import fastexcel

from ces_revisions.vintages.build import build, write
from ces_revisions.vintages.raw import (
    DATA_DIR,
    EMPSIT_RELEASES,
    HISTORICAL_RELEASE_DATES,
    MANIFEST,
    PANEL_DIR,
    RAW_DIR,
    RESCHEDULES,
    VINTAGE_COMMENTS,
    file_sha256,
)

PHILADELPHIA_FED = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "real-time-data/data-files"
)


@dataclass(frozen=True)
class Download:
    file: str  # a path under data/raw/
    url: str


DOWNLOADS = (
    Download("bls/cesvinall.zip", "https://www.bls.gov/web/empsit/cesvinall.zip"),
    Download("bls/cesnaicsrev.htm", "https://www.bls.gov/web/empsit/cesnaicsrev.htm"),
    Download(
        "bls/histreleasedates.pdf", "https://www.bls.gov/bls/histreleasedates.pdf"
    ),
    Download(
        "philadelphiafed/employMvMd.xlsx", f"{PHILADELPHIA_FED}/xlsx/employMvMd.xlsx"
    ),
    Download(
        "philadelphiafed/release-dates-employment-situation.xls",
        f"{PHILADELPHIA_FED}/documentation/Release_-Dates-Employment_Situation-BLS.xls",
    ),
)
WORKBOOK_URL = "https://www.bls.gov/web/empsit/cesvin00.xlsx"
WORKBOOK_PATH = DATA_DIR / "cache" / "cesvin00.xlsx"
COMMENTS_SHEET = "Data Usage and Comments"
COMMENT_COLUMNS = ["sheet_row", "publication_label", "adjustment"]
MANIFEST_COLUMNS = [
    "file",
    "url",
    "sha256",
    "bytes",
    "last_modified",
    "fetched_at",
    "derived_from",
    "derived_from_sha256",
]


def user_agent_for(url: str) -> str:
    """BLS asks automated clients for a contact address; other hosts get none."""
    host = urlsplit(url).hostname or ""
    if host == "bls.gov" or host.endswith(".bls.gov"):
        return archive_inventory.user_agent()
    return "ces-revisions/0.1.0"


def download(url: str) -> tuple[bytes, str]:
    """The payload and its Last-Modified header."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent_for(url)})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def comment_rows(rows: list[tuple[str | None, ...]]) -> list[dict[str, str]]:
    """The entries below the Comments sheet's "Publication Date" header, with sheet rows."""
    header = next(
        index for index, row in enumerate(rows) if row[0] == "Publication Date"
    )
    return [
        {
            "sheet_row": str(index + 1),
            "publication_label": row[0] or "",
            "adjustment": row[1],
        }
        for index, row in enumerate(rows)
        if index > header and row[1]
    ]


def extract_comments(workbook: Path) -> list[dict[str, str]]:
    sheet = fastexcel.read_excel(workbook).load_sheet(
        COMMENTS_SHEET, header_row=None, dtypes="string"
    )
    return comment_rows(sheet.to_polars().rows())


def manifest_row(file: str, fetched_at: str, **fields: str) -> dict[str, str]:
    path = RAW_DIR / file
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        file=file,
        sha256=file_sha256(path),
        bytes=str(path.stat().st_size),
        fetched_at=fetched_at,
        **fields,
    )
    return row


def fetch_sources(now: datetime) -> int:
    """Network step: refresh every source file, its derived text, and the manifest."""
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for item in DOWNLOADS:
        payload, last_modified = download(item.url)
        target = RAW_DIR / item.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        rows.append(
            manifest_row(item.file, stamp, url=item.url, last_modified=last_modified)
        )
    pdf = RAW_DIR / "bls/histreleasedates.pdf"
    subprocess.run(
        ["pdftotext", "-layout", str(pdf), str(RAW_DIR / HISTORICAL_RELEASE_DATES)],
        check=True,
    )
    rows.append(
        manifest_row(
            HISTORICAL_RELEASE_DATES,
            stamp,
            derived_from="pdftotext -layout bls/histreleasedates.pdf",
            derived_from_sha256=file_sha256(pdf),
        )
    )
    payload, last_modified = download(WORKBOOK_URL)
    WORKBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKBOOK_PATH.write_bytes(payload)
    archive_inventory.write_csv(
        RAW_DIR / VINTAGE_COMMENTS, extract_comments(WORKBOOK_PATH), COMMENT_COLUMNS
    )
    rows.append(
        manifest_row(
            VINTAGE_COMMENTS,
            stamp,
            last_modified=last_modified,
            derived_from=f"{WORKBOOK_URL}, sheet {COMMENTS_SHEET}",
            derived_from_sha256=file_sha256(WORKBOOK_PATH),
        )
    )
    index = download(archive_inventory.RELEASE_INDEX_URL)[0].decode("utf-8", "replace")
    vintages = archive_inventory.select_vintages(
        archive_inventory.parse_release_index(index), now=now
    )
    archive_inventory.write_csv(
        RAW_DIR / EMPSIT_RELEASES,
        archive_inventory.vintage_rows(vintages),
        archive_inventory.VINTAGE_COLUMNS,
    )
    rows.append(
        manifest_row(
            EMPSIT_RELEASES,
            stamp,
            derived_from=archive_inventory.RELEASE_INDEX_URL,
        )
    )
    rows.append(
        manifest_row(
            RESCHEDULES,
            "",
            derived_from="hand-keyed from the citation on each row",
        )
    )
    archive_inventory.write_csv(
        RAW_DIR / MANIFEST, sorted(rows, key=lambda row: row["file"]), MANIFEST_COLUMNS
    )
    return 0


def build_panel() -> int:
    """Offline step: write every Stage 3 artifact to data/panel/."""
    manifest = write(build(RAW_DIR), PANEL_DIR)
    for name, entry in manifest.items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "build"])
    command = parser.parse_args(argv).command
    if command == "fetch":
        return fetch_sources(datetime.now(UTC).replace(microsecond=0))
    return build_panel()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vintage_build.py tests/test_vintage_sources.py -q -m "not network"`
Expected: PASS: 5 passed, 1 deselected.

- [ ] **Step 6: Build the panel from the committed sources**

```bash
uv run python scripts/vintage_sources.py build
ls data/panel
git status --short --ignored data
```

Expected:
- `build` prints one line per artifact: `raw_values: 6742659 rows`, `transformations: 5 rows`, `release_index: 572 rows`, `panel: 6342530 rows`, `stage_labels: 37032 rows`, `stage_changes: 33600 rows`, `reconciliation: 1680 rows`, `revision_reconciliation: 1620 rows`, `average_checks: 600 rows`, and `accounting_decomposition: 110 rows`. Those are the counts from the sources of 2026-09-14. Each Employment Situation release since then adds 1 row to `release_index`, 126 to `stage_labels`, 120 to `stage_changes`, and 6 to `reconciliation`, and newer EMPLOY vintages and revision-table estimates add rows to `raw_values` and `panel`.
- `data/panel/` holds the ten parquet files and `manifest.json`.
- `git status` reports `data/cache/` and `data/panel/` as ignored and nothing else under `data/`.

- [ ] **Step 7: Document the layout in CLAUDE.md**

In `CLAUDE.md`, make four edits.

1. **Purpose.** Replace

   ```markdown
   The Python package so far holds the marginalized Kalman engine (`src/ces_revisions/kalman.py`);
   ```

   with

   ```markdown
   The Python package holds the marginalized Kalman engine (`src/ces_revisions/kalman.py`) and roadmap Stage 3's vintage panel (`src/ces_revisions/vintages/`), built from the source files committed in `data/raw/`;
   ```

2. **Dependencies.** In the Python 3.14 bullet of Layout and tooling, replace

   ```markdown
   and Polars; Dynamax is a dev-only dependency
   ```

   with

   ```markdown
   Polars, and fastexcel (Polars' Excel reader, for Stage 3's workbooks); Dynamax is a dev-only dependency
   ```

3. **Layout.** After the bullet that begins ``- `scripts/` holds research tools outside the package``, add:

   ```markdown
   - `data/raw/` holds roadmap Stage 3's source files, committed with `manifest.csv` (each file's origin, SHA-256, size, and BLS's `Last-Modified` date): BLS's vintage files and revision table, BLS's historical release dates, the Employment Situation release list, the entries of the vintage workbooks' Comments sheet, the Philadelphia Fed's EMPLOY vintages and release dates, and `manual/es-reschedules.csv`, the one hand-keyed file, whose rows cite their sources. `scripts/vintage_sources.py fetch` (network) refreshes them, and `scripts/vintage_sources.py build` (offline) writes the artifacts of `ces_revisions.vintages` (the raw-value table, transformations, release-date index, long panel, stage labels, reconciliation, and accounting decomposition) to the gitignored `data/panel/`; tests build the same artifacts from `data/raw/` once per session through `tests/vintage_data.py`. BLS refreshes the vintage files about once a year, weeks after its February benchmark release, and the network canary in `tests/test_vintage_sources.py` fails when it has. Never hand-edit `data/raw/` apart from `manual/es-reschedules.csv`.
   ```

4. **Commands.** After the line `uv run python scripts/archive_inventory.py inventory   # rebuild the archive inventory from docs/inventory/ (offline)`, add:

   ```bash
   uv run python scripts/vintage_sources.py build  # rebuild data/panel/ from the committed sources in data/raw/ (offline)
   uv run python scripts/vintage_sources.py fetch  # refresh data/raw/ and its manifest (network; BLS_CONTACT_EMAIL, pdftotext)
   ```

- [ ] **Step 8: Update the README**

In `README.md`, make three edits.

1. **Status.** Replace `the Python package holds the marginalized Kalman engine that later model stages build on.` with `the Python package holds the marginalized Kalman engine and the vintage panel of CES revisions that later model stages build on.`
2. **Layout.** Replace the line

   ```text
   src/ces_revisions/                    Python package (src layout); kalman.py is the state-space engine
   ```

   with

   ```text
   src/ces_revisions/                    Python package (src layout); kalman.py is the state-space engine, vintages/ the vintage panel
   data/raw/                             committed source files for the vintage panel, with a manifest of their hashes
   ```

3. **Getting started.** After the line `uv run ces-revisions  # run the entry point`, add `uv run python scripts/vintage_sources.py build  # build the vintage panel in data/panel/`, and replace `The modeling stack is JAX, NumPyro, ArviZ, and Polars on Python 3.14.` with `The modeling stack is JAX, NumPyro, ArviZ, and Polars on Python 3.14, with fastexcel to read the source workbooks.`

- [ ] **Step 9: Run the complete verification**

```bash
uv run ruff format --check .
uv run ruff check
uv run pytest -m "not slow and not network" -q
uv run pytest -m network -q
uv run pytest -m slow -q
```

Expected:
- Both ruff commands are clean, the Markdown of CLAUDE.md, the README, and this plan included.
- The fast tier passes: 197 passed, 6 deselected.
- The network tier passes: 3 passed, the two archive-inventory canaries and the vintage-file refresh canary. The Internet Archive answers the archive-inventory canaries slowly; rerun once on a 503 before treating it as a failure.
- The slow tier passes: 3 passed, the synthetic pilot's two fits and the Stage 3 build.

Then check the roadmap's Stage 3 exit clause by clause against the tests that carry it:

| Exit clause | Tests |
|---|---|
| The differencing module reproduces every row of the BLS 1979–present revision table, seasonally adjusted and not, with all three pairwise MARs, to rounding, including the 2003 gap (bullet 1) | `test_differencing.py`: `test_same_release_changes_reproduce_every_table_estimate_within_the_frontier`, `test_panel_revisions_reproduce_the_table_revisions_but_three_cells`, and `test_the_table_takes_november_2003_third_estimate_across_two_releases`; `test_revision_table.py`: `test_every_published_average_reproduces_from_its_monthly_revisions`, `test_published_revisions_equal_their_estimates_difference_but_once`, and `test_the_2003_zeros_are_missing_estimates`. Deviations 1 and 2 record the frontier and BLS's two cells. |
| Re-ingest reproduces every panel value from raw plus transformations | `test_vintage_panel.py`: `test_every_value_rebuilds_from_its_raw_cell` and `test_keys_rebuild_from_their_raw_cells`; `test_vintage_build.py`: `test_build_writes_every_artifact_and_its_manifest` |
| Every stage-label row carries exactly one stage, M is right-censored where its vintage does not exist, and the preliminary benchmark is not a stage (Req 2) | `test_stage_labels.py`: `test_every_unit_carries_each_of_its_stages_once_on_distinct_releases`, `test_m_is_right_censored_exactly_where_its_benchmark_release_is_still_to_come`, and `test_every_stage_is_an_employment_situation_release_so_no_preliminary_benchmark_is` |
| No sector vintage precedes May 2003, and the aggregate leg carries `concept_regime` (Req 4) | `test_vintage_panel.py`: `test_vintage_file_releases_begin_with_the_may_2003_publication_vintage` and `test_the_aggregate_leg_carries_concept_regime`; `test_stage_labels.py`: `test_no_vintage_file_stage_precedes_the_may_2003_publication_vintage` |
| The eleven sector values sum to the published total within rounding in every release file (Req 5) | `test_vintage_panel.py`: `test_the_eleven_supersectors_sum_to_total_nonfarm_in_every_release_file` |
| The Req 9 identity holds on every same-release pair | `test_accounting.py`: `test_the_identity_holds_exactly_on_every_same_release_pair` and `test_the_variance_decomposition_closes_to_float_precision` |
| Live fetches carry the `network` marker, and the default tier passes on fixtures | `test_vintage_sources.py`: `test_bls_has_not_refreshed_the_vintage_files_since_the_manifest`; the fast tier above |
| The release-date index agrees with `es-vintages.csv` on every release from May 1999 on, and covers the February 1995–April 1999 releases with checked dates | `test_release_index.py`: `test_the_index_agrees_with_es_vintages_except_the_december_1999_repost`, `test_february_1995_to_april_1999_releases_come_from_bls_historical_dates`, and `test_bls_historical_dates_agree_with_the_philadelphia_fed_copy_from_1979`. Deviation 3 records December 1999. |

- [ ] **Step 10: Commit**

```bash
uv run ruff format src scripts tests CLAUDE.md README.md
uv run ruff check src scripts tests
git add src/ces_revisions/vintages/build.py scripts/vintage_sources.py tests/test_vintage_build.py CLAUDE.md README.md
git commit -m "Build and write Stage 3's artifacts, and document the vintage data layout"
```

---

## At completion

After the final whole-branch review, the Plan Completion Protocol does three things for this plan besides the Retirement steps at its top.

**Tick the pre-May-1999 item.** In `specs/deferred_items.md`, under `2-ces-revisions`, mark "Review Minor: releases before May 1999 in `es-vintages.csv`" as `- [x] … → done in plan 5`. The index covers the February 1995 to April 1999 releases with BLS's historical dates, checked against the Philadelphia Fed copy (Task 3).

**Defer absorbing BLS's next vintage-file refresh** (Deviation 1). Append this item under the plan's `## 5-ces-revisions — YYYY-MM-DD` section, beside any items the gate defers:

```markdown
- [ ] BLS's next vintage-file refresh (plan 5, Deviation 1). The committed
      `data/raw/bls/cesvinall.zip` of 2026-03-06 ends with the January 2026
      benchmark release, so Stage 3 labels later vintage-file stages
      `beyond_frontier`, and their estimates reconcile only inside the revision
      table. BLS refreshes the files about once a year, weeks after the
      benchmark release. Absorbing a refresh: run
      `scripts/vintage_sources.py fetch` and `build`, check that every newly
      held table estimate reproduces, teach `comment_release_month` in
      `src/ces_revisions/vintages/stages.py` any new Comments wording, and
      re-pin the frontier and the Comments entries in
      `tests/test_stage_labels.py` and the 1627 and 1617 reproduced counts in
      `tests/test_differencing.py`. See
      `specs/plans/completed/5-ces-revisions.md`. Size: quick-fix. Revisit if:
      `test_bls_has_not_refreshed_the_vintage_files_since_the_manifest` fails
      under `uv run pytest -m network`.
```

**Flag Stage 4 for the roadmap resume** (Deviation 4). Stage 4's exit requires that "every reconstruction event and benchmark publication date exists in Stage 3's release-date index", but the index holds only Employment Situation releases, and preliminary benchmark announcements are separate publications. The resume re-validates that clause before it routes Stage 4.
