# Stage 2 — Data and Archive Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revisions-roadmap.md, Stage 2 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Write the inventory half of the Req 20 finding in `docs/ces-revisions-review.md`. It comprises a data-availability inventory with one row per Req 3 series and a first-publication-lag column, a per-vintage inventory of BLS's archived seasonal-adjustment files from May 2003 that discharges Req 9's (open) item, and cited rulings on the five draft disagreements Req 20 names. The literature sections are stubbed for Stage 23.

**Architecture:** The review document is Markdown under test. `tests/test_review_document.py` enforces its sections, CLAUDE.md's rendering conventions, `ruff format --check`, a primary-source host allowlist for citations, and the evidence-label vocabulary. The archive inventory's roughly 280 rows are generated, not typed. `scripts/archive_inventory.py captures` records network evidence (the release index, BLS's current ZIP files, and every Internet Archive copy of them) in committed CSVs under `docs/inventory/`. `scripts/archive_inventory.py inventory` then derives the per-vintage statuses offline and regenerates marked blocks of the document, and tests re-derive the inventory from the committed evidence. The data inventory and the rulings are researched by hand against primary sources and checked structurally.

**Tech Stack:** Python 3.14 standard library (`urllib`, `zipfile`, `csv`, `zoneinfo`), Polars 1.44.2 for one-off LABSTAT computations, pytest 9.1.1, ruff 0.16.7, `pdftotext` (poppler), and the Internet Archive CDX API.

**Source:** [`specs/ces-revisions.md`](../ces-revisions.md) Req 20, Req 9 (the (open) archive item), Req 3, the Rollout note, and Verification bullets 4 and 13 (inventory half); [`specs/ces-revisions-roadmap.md`](../ces-revisions-roadmap.md) Stage 2.

**Retirement:** When this plan retires to `specs/plans/completed/`, `specs/ces-revisions.md` does **not** retire with it. The spec, roadmap, prompt, and research drafts retire together under the roadmap's Completion section. At completion, tick Stage 2 in the roadmap and append this authoritative stamp to the spec's Rollout note:\
`Stage 2: COMPLETE (YYYY-MM-DD) — implemented by plan 2 (specs/plans/completed/2-ces-revisions.md). Next: resume the roadmap.`

## Planning evidence (2026-09-13)

This plan's script and tests were run in a scratch clone of `cd835c5`:
- `ruff check` and `ruff format --check` were clean.
- The 30 hermetic archive tests passed, and the 14 skeleton tests passed on the Task 1 document.
- The offline `inventory` step regenerated the document's blocks from synthetic evidence CSVs, after which all 21 skeleton and archive document tests passed.
- The Task 5–7 checkers accepted a well-formed section. They rejected a placeholder cell, an uncited row, a missing draft locator, a secondary-source link, an unknown evidence label, a primary-sources bullet without a link, and source bullets separated from their lead-in by a blank line. The Task 8 marker test caught a leftover plan-step reference and matched no `Stage N` text.

The live `captures` step was not run, because its output is this stage's deliverable. The live observations below were each fetched on 2026-09-13:

- **BLS keeps no historical seasonal-adjustment inputs.** [`cesseasadj.htm`](https://www.bls.gov/web/empsit/cesseasadj.htm) serves only the current files:
  - `ces.spec.ae.zip`, `ces.spec.aehe.zip`, and `ces.spec.nonae.zip` (first preliminary);
  - `ces.spec.ae2.zip`, `ces.spec.aehe2.zip`, and `ces.spec.nonae2.zip` (second preliminary, independent series);
  - `ces.spec.other.zip`, which holds the calendar regressor `.dat` files, `outliers.xlsx`, and `prior_adjustment_file.xlsx`.

  ZIP entries are dated 2026-02-08 for the specifications and 2026-09-01 for the other inputs. The page's section on updates and the availability of historical input files gives only the cadence (specifications and prior adjustments annually, prior adjustments and recent outliers monthly) and links no archive. Historical model specification *tables* for 2003–2013 are reached through [`cesbmkarch.htm`](https://www.bls.gov/web/empsit/cesbmkarch.htm), which lists the benchmark articles `ces-benchmark-revision-2002.pdf` through `-2025.pdf` and the table pages `cesbmart18-tables.htm` through `cesbmart25-tables.htm`.
- **Unrounded inputs are not published.** [`cesseasadjtn.htm`](https://www.bls.gov/web/empsit/cesseasadjtn.htm) says seasonal adjustment runs on unrounded data while the data published on the BLS website are rounded. No input data file is in any ZIP, whether the current copies or the 2014 capture. The page's figure 2 footnote says the prior-adjustment file contains unrounded data, and the 2014 readme says that file holds every prior adjustment back to 1975.
- **Published precision.** `CEU0000000001` for 2026 M01 is `156728` in `ce.data.00a.TotalNonfarm.Employment`, a value in whole thousands. The alt-nfp repo's local copy of `tri_000000_NSA.csv` from `cesvinall.zip` holds integers. Supersector precision was not checked.
- **Internet Archive coverage.** The CDX index holds status-200 copies of `ces.spec.ae.zip` in 2014–2017, 2019–2021, and 2024–2026. It holds 33 captures of `ces.spec.other.zip` from 2014-07-19 to 2026-08-19, among them 301 redirects and one 403. The Internet Archive's 2011-03-12 copy of `cesseasadj.htm` links the pre-2012 location `ftp://ftp.bls.gov/pub/suppl/empsit.ces.spec.{ae,aehe,nonae,other}.zip`, which the CDX index holds no copies of. A burst of seven CDX queries took over three minutes, and two of them answered with HTML pages instead of CDX lines.
- **Release index.** [`empsit.htm`](https://www.bls.gov/bls/news-release/empsit.htm) lists 12 releases a year for 2000–2024 and 11 for 2025. There was no October 2025 release; the September 2025 release came on 2025-11-20 and the November 2025 release on 2025-12-16. The index also links releases not yet published (`empsit_10022026.htm` returned 404). 2003 items link only `history/empsit_MMDDYYYY.txt` and `archives/empsit_MMDDYYYY.pdf` and carry the title in the list-item text; later items put the title in the `.htm` link.
- **Collection and response series.** LABSTAT `ce.series` gives these spans:
  - `CEU00000000C1` 1981 M01 to 2026 M08;
  - `CEU00000000C2` 1981 M03 to 2026 M07;
  - `CEU00000000C3` 1981 M01 to 2026 M06;
  - `CEU05000000RR`, the third closing response rate for total private, 2009 M04 to 2026 M06.

  `ce.datatype` names C1–C3 the first, second, and third closing collection rates, and RR the third closing response rate.
- **Benchmarks.** [`cestn.htm`](https://www.bls.gov/web/empsit/cestn.htm) Table 5 is "CES total nonfarm preliminary and final benchmark revisions", in thousands:

  | March benchmark | Final | Preliminary | Difference | Footnote |
  |---|---|---|---|---|
  | 2023 | −187 (−0.1%) | −306 (−0.2%) | 119 | none |
  | 2024 | −598 (−0.4%) | −818 (−0.5%) | 220 | 11 |
  | 2025 | −861 (−0.5%) | −911 (−0.6%) | 50 | 12 |

  Table 1 gives benchmark coverage for March 2025 only, and Table 4 gives standard and relative standard errors.
- **Birth–death history.** [`cesbdhst.htm`](https://www.bls.gov/web/empsit/cesbdhst.htm) holds 51 forecast tables, reaching back to April 1999–March 2000, with separate April–December tables from 2019 on.
- **URLs checked.**
  - HTTP 200: the BLS pages above; the 2024 collection-rate notice; the OSMR response-rate page; the BLS 2018–19 lapse page; the 2025 lapse revised-release-dates page; the 2013 shutdown FAQ; the CES notices index; the Employment Situation schedule and archived schedules; the Handbook data chapter; the 2022 MLR seasonal-adjustment article; the CES strike report; work stoppages; the Robertson (2016) OSMR paper; the DOL budget page; the FY2027 CBJ BLS volume; the DOL BLS transition brief and contingency plan; OPM federal holidays; the OMB circulars page; GAO-26-107538; the House and Senate party-division pages; NBER cycle dates; DOL weekly claims; NOAA Storm Events; the Philadelphia Fed EMPLOY page; and `ce.data.0.AllCESSeries`.
  - HTTP 403 to non-browser clients: congress.gov CRS PDFs.
  - HEAD 405: fedscope.opm.gov.
  - 404: guessed archive indexes for the preliminary benchmark release. The current release, `https://www.bls.gov/news.release/prebmk.nr0.htm`, returned 200.
- **Existing code.**
  - alt-nfp's `nfp_download.release_dates` scrapes the same index, with `START_YEAR = 2003`.
  - bls-stats' `releases/calendar.py` takes the release date from the `MMDDYYYY` in each link and overlays `2025-lapse-revised-release-dates.htm`.
  - `bls-stats-specs-ideal.md` §11.2–§11.3 holds that a knowability date is not a value vintage, and that the Internet Archive is a legitimate source for historical BLS index pages.

  Neither repo is a dependency: both are Python 3.12 workspaces whose dependencies this project does not pin. The script ports the link-date rule. alt-nfp's data (`benchmark_revisions.py`, `release_dates.parquet`) serves as a cross-check at most and is never cited: alt-nfp's CLAUDE.md calls its reference "not an oracle".

## Deviations recorded at planning

1. **C1–C3 start in 1981.** Req 3 and the spec's Motivation date C1–C3 "monthly from January 2000", but LABSTAT begins them in 1981. The inventory records the LABSTAT dates and the months present by decade, and Stage 12 inherits them. This plan does not edit the spec; the completion report flags the discrepancy.
2. **Extra artifacts.** The roadmap's Stage 2 Produces line names only the document. This plan also ships `scripts/archive_inventory.py`, its tests and fixture, and `docs/inventory/{es-vintages,archive-captures,archive-inventory}.csv`, because roughly 280 generated rows are auditable only with a reproducible derivation. `es-vintages.csv` overlaps Stage 3's release-date index, so resume re-validates Stage 3 to consume or reconcile it.
3. **Internet Archive copies count.** BLS publishes no archive of these files, so the archive inventory counts Internet Archive copies of BLS's files as an access route. They are labeled `internet_archive`, apart from `bls_current`.
4. **All-employees specifications only.** Only the all-employees specification set is inventoried. Hours and earnings are out of scope, and the second-preliminary sets adjust detailed series that do not aggregate to total nonfarm.

## Execution notes

- **Before Task 1.** This plan is committed on `stage-2-inventory`. If `git status` still shows the roadmap's 2026-09-13 resume edits uncommitted (`M specs/ces-revisions-roadmap.md`), ask your human partner to commit them on the branch first, so the Stage 2 tick lands on the re-validated roadmap.
- **Branch.** Execute on `stage-2-inventory`, branched from `main` at `cd835c5`; `main` is the default branch. A branch can be checked out in only one place, so to execute in a worktree (using-git-worktrees), first switch the checkout at the repository root back to `main`, then add a worktree for the existing `stage-2-inventory` branch rather than creating a new one.
- **Importing the script.** `scripts/` is not a package. Task 2 adds `pythonpath = ["scripts"]` to `[tool.pytest.ini_options]`, so tests import `archive_inventory` by bare name, just as they import the `tests/` helpers. Do not add `__init__.py` to `scripts/` or `tests/`.
- **Network.** Tasks 3 to 7 fetch from www.bls.gov, download.bls.gov, www.dol.gov, and web.archive.org.
  - Before any network step, export `BLS_CONTACT_EMAIL`, set to an address your human partner chooses. BLS asks automated clients for a contact address, and download.bls.gov refuses large flat files without one. Never write the address into a file.
  - curl steps use `UA="ces-revisions/0.1.0 ($BLS_CONTACT_EMAIL)"`.
- **congress.gov.** It answers non-browser clients with HTTP 403. Read CRS products in a browser, or settle the point from BLS, DOL, GAO, or OMB. Never cite a news article or a draft instead.
- **Capture runtime.** `captures` pauses between Internet Archive requests and retries throttled ones, so it runs for several minutes; run it in the background. Do not run it on an Employment Situation release day before 8:30 a.m. Eastern.
- **Research discipline.**
  - The drafts are inputs: read them for the claims to test, and never cite them.
  - Every table cell and ruling rests on a page you fetched during execution, cited with its access date.
  - Quote rarely and briefly; paraphrase otherwise.
  - Write dollar amounts in the document as `\$`.
- **Commits.** Every commit step first runs `uv run ruff format` and `uv run ruff check`, then stages files by explicit path. The repo root has an untracked `.DS_Store`, so never `git add -A`.
- **Stop conditions.** Stop and report rather than improvise when:
  - `captures` exits 1 because a ZIP member matches no known pattern, and Task 3 Step 7 does not resolve it;
  - a primary source contradicts a Planning evidence bullet;
  - a ruling's decisive primary source cannot be reached, and no other primary source settles the question;
  - the vintage list has a reference month without a release other than 2025-10;
  - any count or quoted figure in this plan differs from what you observe.

  Never loosen a test, the allowlist, or a vocabulary to make a row pass. Mark the cell *unverified*, or label the row `not found`, instead.

## Global Constraints

- **Python and dependencies:** `requires-python = ">=3.14"`. Add no dependency. The script uses only the standard library, and Polars (1.44.2) is already pinned. CLAUDE.md: "confirm Python 3.14 support for any package you add."
- **Req 20 evidence rule:** "every numeric claim, date, and citation is verified against a primary source (BLS, CRS, GAO, DOL/OMB, the journal) or explicitly marked *unverified*; each claim carries one of the evidence labels *documented / supported / asserted / contested / not found*".
- **Req 20 rulings:** "at minimum the collection-versus-response conflation, the 2018–19 lapse, whether 2025 program cuts touched CES, the FTE authorized-versus-actual and FTE-versus-headcount concepts, and the 2024/2025 preliminary-versus-final benchmark figures."
- **Req 20 inventory:** "the data-availability inventory per driver (series, vintage coverage, frequency, earliest date, access route, hand-build constraints)", plus the roadmap's first-publication-lag column.
- **Req 9 (open):** "Which vintages have archived specification, prior-adjustment, and outlier files, and whether the unrounded NSA inputs are recoverable, is **(open — resolved by verification, not argument)**".
- **Verification bullet 4:** "an inventory of BLS's archived seasonal-adjustment files by vintage (specification, prior-adjustment, outlier files; unrounded inputs present or absent), committed with the written finding."
- **Verification bullet 13 (inventory half):** "`docs/ces-revisions-review.md` exists; every numeric claim carries a primary-source citation or an *unverified* flag; the five named draft disagreements are resolved on the record; `uv run ruff format --check` passes on it".
- **Req 3:** "annual series are **never** interpolated to monthly". Institutional reach is "FY2009–present in v1".
- **Req 1 and Req 4:** vintages start with "the May 2003 publication vintage".
- **Drafts (CLAUDE.md):** "They are unverified drafts: figures, dates, and citations have not been checked against primary BLS sources".
- **Markdown:** the document, CLAUDE.md, and this plan stay GitHub-renderable, per CLAUDE.md:
  - display math in ```` ```math ```` fences and inline math as `` $`…`$ ``; never `\(…\)`, `\[…\]`, `$$…$$`, or bare `$…$`;
  - literal dollars written as `\$`;
  - real headings, not bold-only lines;
  - hard breaks written as a trailing `\`;
  - pseudo-math containing `_` in code spans.

  The rule binds the review document. This plan's own `**Files:**` and `**Interfaces:**` labels follow the writing-plans task template, as plan 1's do.
- **Tests:** network tests carry the `network` marker. The fast hermetic tier is `uv run pytest -m "not slow and not network"`.
- **Ruff:** the default rules plus `extend-select = ["I", "B", "UP"]`. `ruff format` also formats Python blocks inside Markdown.
- **Scope:** Stage 2 ships the document (inventory sections, rulings, and Stage 23 stubs), the archive-inventory script with its evidence CSVs, their tests, and CLAUDE.md's layout note. It does not:
  - store the seasonal-adjustment files (Stage 14);
  - build the release-date index with closing dates (Stage 3);
  - determine link-relative recoverability (Stage 5);
  - write the literature review (Stage 23);
  - add anything under `src/`.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `tests/review_document.py` | Parse the review: sections, tables, links, the primary-host allowlist, heading anchors | 1 |
| `tests/test_review_document.py` | Structure, conventions, archive, inventory, ruling, and summary tests | 1, 4–8 |
| `docs/ces-revisions-review.md` | The Req 20 finding | 1, 4–8 |
| `pyproject.toml` | `pythonpath = ["scripts"]` for pytest | 2 |
| `scripts/archive_inventory.py` | Vintages, network, captures, inventory, command line | 2, 3, 4 |
| `tests/test_archive_inventory.py` | Hermetic tests and two network canaries for the script | 2, 3, 4 |
| `tests/fixtures/archive_inventory/empsit-index-excerpt.html` | Recorded excerpt of the release index | 2 |
| `docs/inventory/es-vintages.csv` | Every release in the index through the capture date | 3 |
| `docs/inventory/archive-captures.csv` | Every BLS and Internet Archive copy, with fingerprints | 3 |
| `docs/inventory/archive-inventory.csv` | One row per reference month from May 2003 | 4 |
| `CLAUDE.md` | Layout note for `scripts/` and `docs/inventory/`; the inventory command | 8 |

---

### Task 1: Review document skeleton and its structural tests

**Files:**
- Create: `tests/review_document.py`
- Create: `tests/test_review_document.py`
- Create: `docs/ces-revisions-review.md`

**Interfaces:**
- Consumes: the spec, the roadmap, CLAUDE.md's Markdown conventions.
- Produces:
  - `review_document` helpers: `REVIEW_PATH: Path`, `EVIDENCE_LABELS: tuple[str, ...]`, `PRIMARY_HOSTS: tuple[str, ...]`, `read_review() -> str`, `outside_fences(text) -> str`, `headings(text, level) -> list[str]`, `section(text, title, level) -> str`, `tables(text) -> list[list[dict[str, str]]]`, `links(text) -> list[str]`, `is_primary(url) -> bool`, `github_anchor(heading) -> str`.
  - In `tests/test_review_document.py`, the constants `SECTIONS`, `STAGE_23_STUBS`, `REQ3_SERIES` (group heading to series slugs), `RULINGS`, and `CONVENTION_BREAKS`. Later tasks import nothing from this file; they append to it.
  - The document's headings. Each unfinished section carries a `**Status:** in progress — plan 2, Task N.` line, which the task that fills the section removes.

- [ ] **Step 1: Create the document helper module**

Create `tests/review_document.py`:

```python
"""Read docs/ces-revisions-review.md for structural tests: headings, sections, tables, and links."""

import re
from pathlib import Path
from urllib.parse import urlsplit

REVIEW_PATH = Path(__file__).resolve().parents[1] / "docs" / "ces-revisions-review.md"

EVIDENCE_LABELS = ("documented", "supported", "asserted", "contested", "not found")

# Req 20's primary sources are BLS, CRS, GAO, DOL/OMB, and the journal. The inventory also cites
# the custodians of other primary records: Congress and the Government Publishing Office, the
# National Archives, OPM, BEA, the Census Bureau, NBER's business-cycle dates, NOAA, and the
# Federal Reserve's real-time data sets. DOI links resolve journal articles.
PRIMARY_HOSTS = (
    "bls.gov",
    "dol.gov",
    "doleta.gov",
    "gao.gov",
    "congress.gov",
    "govinfo.gov",
    "house.gov",
    "senate.gov",
    "whitehouse.gov",
    "archives.gov",
    "opm.gov",
    "bea.gov",
    "census.gov",
    "nber.org",
    "noaa.gov",
    "federalreserve.gov",
    "philadelphiafed.org",
    "stlouisfed.org",
    "doi.org",
)
ARCHIVE_HOST = "web.archive.org"

_FENCE = re.compile(r"(?ms)^```.*?^```[ \t]*$")
_HEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")
_LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
_ARCHIVED = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(https?://.+)$")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def read_review() -> str:
    return REVIEW_PATH.read_text(encoding="utf-8")


def outside_fences(text: str) -> str:
    """`text` with fenced blocks blanked line for line, so they never read as headings or tables."""
    return _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def headings(text: str, level: int) -> list[str]:
    return [
        match.group(2)
        for line in outside_fences(text).splitlines()
        if (match := _HEADING.match(line)) and len(match.group(1)) == level
    ]


def section(text: str, title: str, level: int) -> str:
    """The body under heading `title` at `level`, up to the next heading at that level or above."""
    lines = outside_fences(text).splitlines()
    marker = f"{'#' * level} {title}"
    assert marker in lines, f"no heading {marker!r}"
    start = lines.index(marker) + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if (match := _HEADING.match(lines[index])) and len(match.group(1)) <= level
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def tables(text: str) -> list[list[dict[str, str]]]:
    """Every pipe table in `text`, each a list of rows keyed by the header cells."""
    found, block = [], []
    for line in [*outside_fences(text).splitlines(), ""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2:
            header = _cells(block[0])
            found.append(
                [dict(zip(header, _cells(row), strict=True)) for row in block[2:]]
            )
        block = []
    return found


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in _UNESCAPED_PIPE.split(line.strip().strip("|"))]


def links(text: str) -> list[str]:
    return _LINK.findall(text)


def is_primary(url: str) -> bool:
    """Whether `url` is on a primary-source host, or is an Internet Archive copy of such a page."""
    host = urlsplit(url).hostname or ""
    if host == ARCHIVE_HOST:
        archived = _ARCHIVED.match(url)
        return archived is not None and is_primary(archived.group(1))
    return any(
        host == primary or host.endswith(f".{primary}") for primary in PRIMARY_HOSTS
    )


def github_anchor(heading: str) -> str:
    """The fragment GitHub gives a heading: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
```

`tables` zips cells with `strict=True`, so an unescaped `|` inside a cell raises instead of silently shifting columns.

- [ ] **Step 2: Write the failing structural tests**

Create `tests/test_review_document.py`:

```python
"""docs/ces-revisions-review.md, the Req 20 finding: structure, conventions, and evidence."""

import re
import subprocess
import sys

import pytest
from review_document import REVIEW_PATH, headings, outside_fences, read_review, section

SECTIONS = [
    "Evidence labels and citations",
    "Summary of findings",
    "Literature by driver",
    "Gap table",
    "Data-availability inventory",
    "Seasonal-adjustment archive inventory",
    "Draft disagreements resolved",
    "Annotated bibliography",
]
STAGE_23_STUBS = ["Literature by driver", "Gap table", "Annotated bibliography"]

# Every series Req 3 names, grouped as Req 3 groups them. Calendar-predicted and residual C1 are
# derived in roadmap Stage 12 from rows here, so they are not series of their own.
REQ3_SERIES = {
    "Collection window": [
        "collection_business_days",
        "federal_holiday",
        "nonstandard_release",
        "appropriations_lapse",
        "CEU00000000C1",
        "CEU00000000C2",
        "CEU00000000C3",
        "CEU05000000RR",
    ],
    "Seasonal": [
        "four_five_week_interval",
        "easter_labor_day",
        "outlier_flags",
        "specification_regime",
        "covid_interventions",
    ],
    "Net birth–death": ["birth_death_forecast", "birth_death_forecast_vs_realized"],
    "Sample": ["linked_coverage", "relative_standard_error"],
    "Institutional": [
        "budget_authority",
        "budget_deflator",
        "fte",
        "opm_headcount",
        "ces_workyears",
        "cr_days",
        "cr_extensions",
        "full_year_cr",
        "shutdown_bls_closed",
        "shutdown_collection_stopped",
        "shutdown_release_delayed",
        "shutdown_window_extended",
    ],
    "Party composition": ["house_majority", "senate_majority"],
    "Macro and tail controls": [
        "recession_dates",
        "ui_claims",
        "strikes",
        "severe_weather",
    ],
}

RULINGS = [
    "Collection rate versus response rate",
    "The December 2018 to January 2019 lapse in appropriations",
    "Whether the 2025 program cuts reached CES",
    "FTE concepts: authorized versus actual, and FTE versus headcount",
    "The 2024 and 2025 preliminary and final benchmark revisions",
]

# CLAUDE.md's GitHub-rendering conventions, as patterns that must not occur outside code fences.
CONVENTION_BREAKS = {
    "$$ display math": re.compile(r"\$\$"),
    r"\( or \) math": re.compile(r"\\[()]"),
    r"\[ or \] math": re.compile(r"\\[\[\]]"),
    "bare dollar sign: write \\$ or $`…`$": re.compile(r"(?<![\\`])\$(?!`)"),
    "two trailing spaces: write a trailing backslash": re.compile(
        r" {2,}$", re.MULTILINE
    ),
    "bold-only line: write a heading": re.compile(r"^\*\*[^*]+\*\*$", re.MULTILINE),
    "setext underline": re.compile(
        r"^(?![ \t]*\|)[^\n]*\S[^\n]*\n[ \t]*(?:=+|-+)[ \t]*$", re.MULTILINE
    ),
}


# --- Task 1: skeleton and conventions ------------------------------------------------------


def test_top_level_sections_appear_in_order():
    assert headings(read_review(), level=2) == SECTIONS


@pytest.mark.parametrize("title", STAGE_23_STUBS)
def test_literature_sections_are_stubs_that_stage_23_writes(title):
    body = section(read_review(), title, level=2)
    assert "**Status:** stub" in body
    assert "Stage 23" in body


def test_inventory_groups_are_the_req_3_groups():
    body = section(read_review(), "Data-availability inventory", level=2)
    assert headings(body, level=3) == list(REQ3_SERIES)


def test_rulings_are_the_five_named_by_req_20():
    body = section(read_review(), "Draft disagreements resolved", level=2)
    assert headings(body, level=3) == RULINGS


@pytest.mark.parametrize("name", list(CONVENTION_BREAKS))
def test_review_follows_the_github_markdown_conventions(name):
    text = outside_fences(read_review())
    found = [
        text[max(0, m.start() - 40) : m.end() + 40]
        for m in CONVENTION_BREAKS[name].finditer(text)
    ]
    assert not found, found[:3]


def test_ruff_format_accepts_the_review():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", str(REVIEW_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q`
Expected: FAIL. Every test errors with `FileNotFoundError` for `docs/ces-revisions-review.md`, except `test_ruff_format_accepts_the_review`, which fails because ruff cannot find the file.

- [ ] **Step 4: Create the document skeleton**

Create `docs/ces-revisions-review.md`:

```markdown
# CES payroll revisions: data inventory, archive inventory, and draft rulings

This document is the written finding that Req 20 of [`specs/ces-revisions.md`](../specs/ces-revisions.md) requires. Roadmap Stage 2 wrote its inventory half: the data-availability inventory, the seasonal-adjustment archive inventory, and rulings on the five disagreements among the research drafts that Req 20 names. Roadmap Stage 23 completes the literature by driver, the gap table, and the annotated bibliography.

## Evidence labels and citations

The three research drafts in `specs/` are inputs to this document, never its sources. Every numeric claim, date, and citation here was checked against a primary source (BLS, CRS, GAO, DOL or OMB, or the journal) or is marked *unverified*, and each cited page carries the date it was accessed. Each claim carries one evidence label, in the sense `specs/ces-revisions.md` adopts from the drafts:

- *documented*: a primary source reports the rule, fact, or statistic directly.
- *supported*: an explicit empirical design backs it, or it is computed from primary data.
- *asserted*: a source states it without evidence that would establish it.
- *contested*: the evidence conflicts, or does not support the claimed reading.
- *not found*: no primary source was located, which does not prove none exists.

## Summary of findings

**Status:** in progress — plan 2, Task 8.

## Literature by driver

**Status:** stub — roadmap Stage 23 writes this section: the literature on the magnitude of CES revisions, organized by the five candidate drivers (seasonality, collection interval, BLS funding, sample, and staffing), with the decomposition of revision sources running through it.

## Gap table

**Status:** stub — roadmap Stage 23 writes this section: one row per open question, giving the question, the closest existing work, why it falls short, the data it needs, and its feasibility from public sources.

## Data-availability inventory

**Status:** in progress — plan 2, Tasks 5 to 7.

Each subsection below is one group of the series Req 3 names, with one row per series. The columns are:

- **Vintage coverage**: whether values as first published can still be recovered, and from where.
- **Frequency**: the native frequency. Req 3 never interpolates an annual series to monthly.
- **Earliest date**: the first observation the access route provides.
- **Access route**: where the series is obtained.
- **Hand-build constraint**: what must be assembled by hand, and why.
- **First-publication lag**: when a value first became public relative to its reference period, which sets its `observable_at` date in roadmap Stages 4, 12 to 14, and 20.
- **Evidence** and **Citation**: the row's evidence label, and its primary sources or *unverified*.

### Collection window

### Seasonal

### Net birth–death

### Sample

### Institutional

### Party composition

### Macro and tail controls

## Seasonal-adjustment archive inventory

**Status:** in progress — plan 2, Tasks 3 and 4.

## Draft disagreements resolved

Req 20 names five disagreements among the research drafts. Each subsection states what each draft says, cites the primary sources that settle the question, and gives the ruling, its evidence label, and its consequence for later stages.

### Collection rate versus response rate

**Status:** in progress — plan 2, Task 5.

### The December 2018 to January 2019 lapse in appropriations

**Status:** in progress — plan 2, Task 7.

### Whether the 2025 program cuts reached CES

**Status:** in progress — plan 2, Task 6.

### FTE concepts: authorized versus actual, and FTE versus headcount

**Status:** in progress — plan 2, Task 7.

### The 2024 and 2025 preliminary and final benchmark revisions

**Status:** in progress — plan 2, Task 6.

## Annotated bibliography

**Status:** stub — roadmap Stage 23 writes this section: full citations, each with a one-line note on its relevance and labeled peer-reviewed, agency documentation, or grey literature.
```

The heading `Net birth–death` uses an en dash, matching `REQ3_SERIES`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_review_document.py -q`
Expected: PASS, 14 passed.

- [ ] **Step 6: Commit**

```bash
uv run ruff format tests docs/ces-revisions-review.md
uv run ruff check tests
git add tests/review_document.py tests/test_review_document.py docs/ces-revisions-review.md
git commit -m "Add the Req 20 review skeleton and its structural tests"
```

### Task 2: Vintage enumeration from the Employment Situation release index

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`)
- Create: `tests/fixtures/archive_inventory/empsit-index-excerpt.html`
- Create: `tests/test_archive_inventory.py`
- Create: `scripts/archive_inventory.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, in `scripts/archive_inventory.py`:
  - Constants: `ROOT`, `INVENTORY_DIR`, `REVIEW_PATH`, `VINTAGES_PATH`, `CAPTURES_PATH`, `INVENTORY_PATH` (all `Path`); `FIRST_REFERENCE_MONTH = date(2003, 5, 1)`; `RELEASE_INDEX_URL: str`; `VINTAGE_COLUMNS: list[str]`.
  - `@dataclass(frozen=True) Vintage(reference_month: date, release_date: date, release_url: str)`; `reference_month` is the first of the month.
  - `parse_release_index(html: str) -> list[Vintage]`, sorted by reference month, raising `ValueError` on two release dates for one month.
  - `select_vintages(vintages, *, as_of: date) -> list[Vintage]`.
  - `next_month(month: date) -> date`.
  - `missing_reference_months(vintages, *, start=FIRST_REFERENCE_MONTH) -> list[date]`.
  - `release_instant(release_date: date) -> datetime`, 8:30 a.m. Eastern as an aware UTC datetime.
  - `vintage_rows(vintages) -> list[dict[str, str]]` and `vintages_from_rows(rows) -> list[Vintage]`.
  - `write_csv(path, rows, columns) -> None` and `read_csv(path) -> list[dict[str, str]]`.
  - `user_agent() -> str` and `fetch(url: str) -> bytes`.
- Test helpers: `index_excerpt() -> str` and `vintage(reference_month: str, released: str) -> Vintage`. Tasks 3 and 4 add more.

- [ ] **Step 1: Put `scripts/` on the test path**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, insert these two lines directly after `testpaths = ["tests"]`:

```toml
# scripts/ holds research tools that tests import by bare name, like the helpers in tests/.
pythonpath = ["scripts"]
```

- [ ] **Step 2: Record the index fixture**

Create `tests/fixtures/archive_inventory/empsit-index-excerpt.html`. The list items copy the live index's markup, including a release scheduled after 2026-09-13, the missing October 2025 release, and 2003's text-and-PDF-only items:

```html
<!-- Excerpt of https://www.bls.gov/bls/news-release/empsit.htm as served on 2026-09-13: list
     items for 2026 (including a release scheduled after that date), 2025 (no October release),
     and 2003 (text and PDF links only), plus one navigation item. -->
<h4 id="2026">2026 Employment Situation</h4>
<ul>
<li><a href="/news.release/archives/empsit_10022026.htm">September 2026 Employment Situation</a> (<a aria-label="September 2026 Employment Situation PDF" href="/news.release/archives/empsit_10022026.pdf">PDF</a>)</li>
<li><a href="/news.release/archives/empsit_09042026.htm">August 2026 Employment Situation</a> (<a aria-label="August 2026 Employment Situation PDF" href="/news.release/archives/empsit_09042026.pdf">PDF</a>)</li>
</ul>
<h4 id="2025">2025 Employment Situation</h4>
<ul>
<li><a href="/news.release/archives/empsit_12162025.htm">November 2025 Employment Situation</a> (<a aria-label="November 2025 Employment Situation PDF" href="/news.release/archives/empsit_12162025.pdf">PDF</a>)</li>
<li><a href="/news.release/archives/empsit_11202025.htm">September 2025 Employment Situation</a> (<a aria-label="September 2025 Employment Situation PDF" href="/news.release/archives/empsit_11202025.pdf">PDF</a>)</li>
<li><a href="/news.release/archives/empsit_09052025.htm">August 2025 Employment Situation</a> (<a aria-label="August 2025 Employment Situation PDF" href="/news.release/archives/empsit_09052025.pdf">PDF</a>)</li>
</ul>
<h4 id="2003">2003 Employment Situation</h4>
<ul>
<li>May 2003 Employment Situation (<a aria-label="May 2003 Employment Situation TXT" href="/news.release/history/empsit_06062003.txt">TXT</a>) (<a aria-label="May 2003 Employment Situation PDF" href="/news.release/archives/empsit_06062003.pdf">PDF</a>)</li>
<li>April 2003 Employment Situation (<a aria-label="April 2003 Employment Situation TXT" href="/news.release/history/empsit_05022003.txt">TXT</a>) (<a aria-label="April 2003 Employment Situation PDF" href="/news.release/archives/empsit_05022003.pdf">PDF</a>)</li>
</ul>
<ul class="nav"><li><a href="/bls/news-release/empsit.htm">The Employment Situation</a></li></ul>
```

- [ ] **Step 3: Write the failing vintage tests**

Create `tests/test_archive_inventory.py`:

```python
"""scripts/archive_inventory.py: vintages, Internet Archive captures, and per-vintage statuses."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from archive_inventory import (
    FIRST_REFERENCE_MONTH,
    RELEASE_INDEX_URL,
    Vintage,
    fetch,
    missing_reference_months,
    parse_release_index,
    release_instant,
    select_vintages,
    vintage_rows,
    vintages_from_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "archive_inventory"


def index_excerpt() -> str:
    return (FIXTURES / "empsit-index-excerpt.html").read_text(encoding="utf-8")


def vintage(reference_month: str, released: str) -> Vintage:
    return Vintage(
        date.fromisoformat(f"{reference_month}-01"),
        date.fromisoformat(released),
        f"https://www.bls.gov/news.release/archives/empsit_{reference_month}.htm",
    )


# --- Vintages ------------------------------------------------------------------------------


def test_release_index_yields_one_vintage_per_reference_month():
    vintages = parse_release_index(index_excerpt())
    assert [
        (f"{v.reference_month:%Y-%m}", v.release_date.isoformat()) for v in vintages
    ] == [
        ("2003-04", "2003-05-02"),
        ("2003-05", "2003-06-06"),
        ("2025-08", "2025-09-05"),
        ("2025-09", "2025-11-20"),
        ("2025-11", "2025-12-16"),
        ("2026-08", "2026-09-04"),
        ("2026-09", "2026-10-02"),
    ]


def test_release_urls_prefer_html_then_text_over_pdf():
    urls = {
        f"{v.reference_month:%Y-%m}": v.release_url
        for v in parse_release_index(index_excerpt())
    }
    assert (
        urls["2003-05"]
        == "https://www.bls.gov/news.release/history/empsit_06062003.txt"
    )
    assert (
        urls["2025-09"]
        == "https://www.bls.gov/news.release/archives/empsit_11202025.htm"
    )


def test_two_release_dates_for_one_reference_month_are_an_error():
    html = (
        '<li><a href="/news.release/archives/empsit_09052025.htm">'
        "August 2025 Employment Situation</a></li>"
        '<li><a href="/news.release/archives/empsit_09122025.htm">'
        "August 2025 Employment Situation</a></li>"
    )
    with pytest.raises(ValueError, match="two releases for 2025-08"):
        parse_release_index(html)


def test_releases_scheduled_after_the_as_of_date_are_dropped():
    kept = select_vintages(
        parse_release_index(index_excerpt()), as_of=date(2026, 9, 13)
    )
    assert kept[-1].reference_month == date(2026, 8, 1)


def test_missing_reference_months_finds_the_absent_october_2025_release():
    vintages = [
        vintage("2025-08", "2025-09-05"),
        vintage("2025-09", "2025-11-20"),
        vintage("2025-11", "2025-12-16"),
    ]
    assert missing_reference_months(vintages, start=date(2025, 8, 1)) == [
        date(2025, 10, 1)
    ]


@pytest.mark.parametrize(
    ("released", "instant"),
    [
        ("2025-09-05", "2025-09-05T12:30:00+00:00"),
        ("2026-02-11", "2026-02-11T13:30:00+00:00"),
    ],
    ids=["daylight time", "standard time"],
)
def test_the_embargo_lifts_at_eight_thirty_eastern(released, instant):
    assert release_instant(date.fromisoformat(released)) == datetime.fromisoformat(
        instant
    )


def test_vintages_round_trip_through_csv_rows():
    vintages = parse_release_index(index_excerpt())
    assert vintages_from_rows(vintage_rows(vintages)) == vintages


@pytest.mark.network
def test_live_release_index_starts_the_modern_vintages_on_june_6_2003():
    html = fetch(RELEASE_INDEX_URL).decode("utf-8", "replace")
    vintages = select_vintages(
        parse_release_index(html), as_of=datetime.now(UTC).date()
    )
    first = next(v for v in vintages if v.reference_month == FIRST_REFERENCE_MONTH)
    assert first.release_date == date(2003, 6, 6)
    missing = missing_reference_months(vintages)
    assert date(2025, 10, 1) in missing
    assert all(month >= date(2025, 10, 1) for month in missing)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'archive_inventory'`.

- [ ] **Step 5: Write the vintage, CSV, and network sections of the script**

Create `scripts/archive_inventory.py`. The module docstring already names the `captures` and `inventory` subcommands, which Tasks 3 and 4 add:

```python
"""Enumerate CES vintages and inventory the seasonal-adjustment files that survive for each.

Roadmap Stage 2 (specs/ces-revisions-roadmap.md) answers Req 9's open question: which vintages
have archived specification, prior-adjustment, and outlier files. BLS serves only the current
files (https://www.bls.gov/web/empsit/cesseasadj.htm), so a past vintage's files survive only
where the Internet Archive copied them. Network capture and offline derivation are separate
subcommands, so the inventory can be rebuilt from the committed evidence without the network:

    uv run python scripts/archive_inventory.py captures    # writes the evidence CSVs
    uv run python scripts/archive_inventory.py inventory   # derives the inventory from them

`captures` writes docs/inventory/es-vintages.csv and docs/inventory/archive-captures.csv.
`inventory` writes docs/inventory/archive-inventory.csv and regenerates the inventory tables in
docs/ces-revisions-review.md.
"""

import csv
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DIR = ROOT / "docs" / "inventory"
REVIEW_PATH = ROOT / "docs" / "ces-revisions-review.md"
VINTAGES_PATH = INVENTORY_DIR / "es-vintages.csv"
CAPTURES_PATH = INVENTORY_DIR / "archive-captures.csv"
INVENTORY_PATH = INVENTORY_DIR / "archive-inventory.csv"

# Req 1 and Req 4: the vintage panel starts with the May 2003 publication vintage.
FIRST_REFERENCE_MONTH = date(2003, 5, 1)
RELEASE_INDEX_URL = "https://www.bls.gov/bls/news-release/empsit.htm"
MONTHS = (
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
EASTERN = ZoneInfo("America/New_York")
# Employment Situation releases are embargoed until 8:30 a.m. Eastern time.
EMBARGO = time(8, 30)


# --- Vintages ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Vintage:
    """One Employment Situation release: the reference month it first estimates, and when."""

    reference_month: date  # the first day of the month
    release_date: date
    release_url: str


_ITEM = re.compile(r"(?s)<li(?:\s[^>]*)?>(.*?)</li>")
_TAG = re.compile(r"<[^>]+>")
_RELEASE_HREF = re.compile(
    r'href="(/news\.release/(?:archives|history)/empsit_(\d{2})(\d{2})(\d{4})\.(htm|txt|pdf))"'
)
_TITLE = re.compile(r"\b(" + "|".join(MONTHS) + r") (\d{4}) Employment Situation\b")
_FORMAT_PREFERENCE = {"htm": 0, "txt": 1, "pdf": 2}


def parse_release_index(html: str) -> list[Vintage]:
    """Parse the Employment Situation archive index into one vintage per reference month.

    The release date is the MMDDYYYY in each link, the index's only stable key. The reference
    month comes from the item's title, because delayed releases (September 2013 and September
    2025) break any fixed lag between reference month and release.
    """
    vintages: dict[date, Vintage] = {}
    for item in _ITEM.findall(html):
        title = _TITLE.search(_TAG.sub(" ", item))
        links = _RELEASE_HREF.findall(item)
        if title is None or not links:
            continue
        reference = date(int(title.group(2)), MONTHS.index(title.group(1)) + 1, 1)
        path, month, day, year, _ = min(
            links, key=lambda link: _FORMAT_PREFERENCE[link[4]]
        )
        vintage = Vintage(
            reference,
            date(int(year), int(month), int(day)),
            f"https://www.bls.gov{path}",
        )
        earlier = vintages.get(reference)
        if earlier is not None and earlier.release_date != vintage.release_date:
            raise ValueError(
                f"two releases for {reference:%Y-%m}: "
                f"{earlier.release_date} and {vintage.release_date}"
            )
        vintages[reference] = vintage
    return sorted(vintages.values(), key=lambda vintage: vintage.reference_month)


def select_vintages(vintages: list[Vintage], *, as_of: date) -> list[Vintage]:
    """Drop releases after `as_of`: the index links scheduled releases before they happen."""
    return [vintage for vintage in vintages if vintage.release_date <= as_of]


def next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)


def missing_reference_months(
    vintages: list[Vintage], *, start: date = FIRST_REFERENCE_MONTH
) -> list[date]:
    """Reference months from `start` through the latest vintage that no release first estimated."""
    released = {vintage.reference_month for vintage in vintages}
    missing, month = [], start
    while month <= vintages[-1].reference_month:
        if month not in released:
            missing.append(month)
        month = next_month(month)
    return missing


def release_instant(release_date: date) -> datetime:
    """When a release's embargo lifted, in UTC."""
    return datetime.combine(release_date, EMBARGO, tzinfo=EASTERN).astimezone(UTC)


VINTAGE_COLUMNS = ["reference_month", "release_date", "release_url"]


def vintage_rows(vintages: list[Vintage]) -> list[dict[str, str]]:
    return [
        {
            "reference_month": f"{vintage.reference_month:%Y-%m}",
            "release_date": vintage.release_date.isoformat(),
            "release_url": vintage.release_url,
        }
        for vintage in vintages
    ]


def vintages_from_rows(rows: list[dict[str, str]]) -> list[Vintage]:
    return [
        Vintage(
            date.fromisoformat(f"{row['reference_month']}-01"),
            date.fromisoformat(row["release_date"]),
            row["release_url"],
        )
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Network -------------------------------------------------------------------------------


def user_agent() -> str:
    """BLS asks automated clients for a contact address, which BLS_CONTACT_EMAIL supplies."""
    contact = os.environ.get("BLS_CONTACT_EMAIL")
    return f"ces-revisions/0.1.0 ({contact})" if contact else "ces-revisions/0.1.0"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: PASS, 8 passed, 1 deselected.

- [ ] **Step 7: Run the live index canary**

Run: `uv run pytest tests/test_archive_inventory.py -q -m network` (with `BLS_CONTACT_EMAIL` exported)
Expected: PASS, 1 passed. On a failure, open the live index. Changed list-item markup is a parser fix plus a re-recorded fixture. A reference month without a release other than 2025-10 is a stop condition.

- [ ] **Step 8: Commit**

```bash
uv run ruff format scripts tests
uv run ruff check scripts tests
git add pyproject.toml scripts/archive_inventory.py tests/test_archive_inventory.py tests/fixtures/archive_inventory/empsit-index-excerpt.html
git commit -m "Enumerate Employment Situation vintages from the BLS release index"
```

### Task 3: Capture evidence — BLS and Internet Archive copies of the seasonal-adjustment files

**Files:**
- Modify: `scripts/archive_inventory.py` (replace the import block; append the Captures and Command line sections)
- Modify: `tests/test_archive_inventory.py` (replace the import block; add `make_zip`; append the capture tests)
- Create (generated): `docs/inventory/es-vintages.csv`, `docs/inventory/archive-captures.csv`

**Interfaces:**
- Consumes (Task 2): `Vintage`, `parse_release_index`, `select_vintages`, `vintage_rows`, `VINTAGE_COLUMNS`, `write_csv`, `fetch`, `RELEASE_INDEX_URL`, `VINTAGES_PATH`, `CAPTURES_PATH`.
- Produces:
  - `FILE_TYPES = ("specification", "prior_adjustment", "outliers")`.
  - `inspect_zip(payload: bytes) -> dict[str, str]`. Its keys are the three file types plus `"unexpected"`. A value is a 16-hex-digit fingerprint, or `""` when the file type is absent; `unexpected` is a `;`-joined list of member names.
  - `@dataclass(frozen=True) Capture(artifact: str, source: str, timestamp: datetime, url: str, status: str, digest: str, specification: str = "", prior_adjustment: str = "", outliers: str = "", unexpected: str = "")`. `source` is `"bls_current"` or `"internet_archive"`. `status` is a CDX status code or `"unreplayable"`.
  - `CAPTURE_COLUMNS`; `capture_row(capture) -> dict[str, str]`; `capture_from_row(row) -> Capture`. Timestamps use the format `%Y-%m-%dT%H:%M:%SZ`.
  - `parse_cdx(text) -> list[tuple[datetime, str, str, str]]`, raising `ValueError` on any non-CDX line.
  - `@dataclass(frozen=True) Artifact(name: str, current_url: str, archive_keys: tuple[str, ...])` and `ARTIFACTS`.
  - `retrying(attempt: Callable[[], T], *, what: str) -> T`, raising `RuntimeError` after `ATTEMPTS` (5) tries.
  - `list_captures(key) -> list[tuple[datetime, str, str, str]]`.
  - `collect_captures(now: datetime) -> list[Capture]` and `capture_evidence() -> int`.
  - `main(argv=None) -> int`, accepting only `captures` until Task 4.
- Test helper: `make_zip(members: dict[str, bytes], date_time=...) -> bytes`.

- [ ] **Step 1: Write the failing capture tests**

In `tests/test_archive_inventory.py`, replace the import lines between the module docstring and `FIXTURES = ...` with:

```python
import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import archive_inventory
import pytest
from archive_inventory import (
    FIRST_REFERENCE_MONTH,
    RELEASE_INDEX_URL,
    Capture,
    Vintage,
    capture_from_row,
    capture_row,
    fetch,
    inspect_zip,
    list_captures,
    missing_reference_months,
    parse_cdx,
    parse_release_index,
    release_instant,
    retrying,
    select_vintages,
    vintage_rows,
    vintages_from_rows,
)
```

Add this helper directly after the `vintage` helper:

```python
def make_zip(members: dict[str, bytes], date_time=(2026, 9, 1, 15, 33, 0)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=date_time), payload)
    return buffer.getvalue()
```

Append at the end of the file:

```python
# --- Captures ------------------------------------------------------------------------------

# Member names in the Internet Archive's 2014-07-19 capture of ces.spec.other.zip.
OTHER_INPUTS_2014 = [
    "Fdumw96.dat",
    "NEW.readme.announcement.txt",
    "outliers.xls",
    "prior_adjustment_file.xls",
    "readme.other.txt",
    "DUMlp06.dat",
    "DUMlpel6.dat",
    "FDUM8606.dat",
    "Fdumel06.dat",
    "Fdumel96.dat",
    "Fdumpc96.dat",
    "Fdumpcw6.dat",
]


def test_inspect_zip_finds_prior_adjustments_and_outliers_among_the_2014_other_inputs():
    found = inspect_zip(make_zip(dict.fromkeys(OTHER_INPUTS_2014, b"x")))
    assert found["prior_adjustment"] and found["outliers"]
    assert found["specification"] == ""
    assert found["unexpected"] == ""


def test_inspect_zip_finds_specification_files_inside_a_folder():
    found = inspect_zip(make_zip({"ces.spec.ae/AE1011330000.spc": b"series{}"}))
    assert found["specification"]
    assert found["prior_adjustment"] == found["outliers"] == found["unexpected"] == ""


def test_inspect_zip_reports_a_member_no_readme_explains():
    found = inspect_zip(
        make_zip({"prior_adjustment_file.xlsx": b"x", "ces_input.dat": b"1 2"})
    )
    assert found["unexpected"] == "ces_input.dat"


def test_fingerprints_ignore_zip_timestamps_but_not_contents():
    members = {"outliers.xlsx": b"2020 AO"}
    first = inspect_zip(make_zip(members, date_time=(2025, 1, 1, 0, 0, 0)))
    rezipped = inspect_zip(make_zip(members, date_time=(2026, 1, 1, 0, 0, 0)))
    changed = inspect_zip(make_zip({"outliers.xlsx": b"2020 LS"}))
    assert first["outliers"] == rezipped["outliers"] != changed["outliers"]


def test_parse_cdx_reads_utc_timestamps_and_keeps_every_status():
    text = (
        "20140719134601 http://www.bls.gov/web/empsit/ces.spec.other.zip 200 "
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH\n"
        "20170102045740 http://www.bls.gov/web/empsit/ces.spec.other.zip 301 "
        "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ\n"
    )
    rows = parse_cdx(text)
    assert rows[0] == (
        datetime(2014, 7, 19, 13, 46, 1, tzinfo=UTC),
        "http://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH",
    )
    assert [status for _, _, status, _ in rows] == ["200", "301"]


def test_parse_cdx_of_no_captures_is_empty():
    assert parse_cdx("") == []


def test_parse_cdx_rejects_a_throttling_page():
    with pytest.raises(ValueError, match="not a CDX line"):
        parse_cdx("<html><body><p>Too many requests</p></body></html>")


def test_captures_round_trip_through_csv_rows():
    original = Capture(
        "other_inputs",
        "internet_archive",
        datetime(2014, 7, 19, 13, 46, 1, tzinfo=UTC),
        "https://web.archive.org/web/20140719134601id_/http://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "ZSL4N7HMBW6G454PPITPXXHNHP34TVXH",
        prior_adjustment="ab12",
        outliers="cd34",
    )
    assert capture_from_row(capture_row(original)) == original


def test_retrying_returns_once_an_attempt_succeeds(monkeypatch):
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    outcomes = iter([OSError("connection reset"), ValueError("throttled"), "payload"])

    def attempt():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert retrying(attempt, what="test fetch") == "payload"


def test_retrying_gives_up_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)

    def attempt():
        raise OSError("down")

    with pytest.raises(RuntimeError, match="test fetch failed after 5 attempts"):
        retrying(attempt, what="test fetch")


def test_collect_captures_records_an_unreplayable_copy_instead_of_failing(monkeypatch):
    artifact = archive_inventory.Artifact(
        "other_inputs",
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        ("bls.gov/web/empsit/ces.spec.other.zip",),
    )
    original = "http://www.bls.gov/web/empsit/ces.spec.other.zip"
    listing = [
        (datetime(2015, 3, 21, 6, 17, 57, tzinfo=UTC), original, "200", "KEPT"),
        (datetime(2017, 1, 2, 4, 57, 40, tzinfo=UTC), original, "301", "MOVED"),
        (datetime(2020, 10, 18, 12, 35, 3, tzinfo=UTC), original, "200", "LOST"),
    ]
    payload = make_zip({"outliers.xlsx": b"AO"})

    def fake_fetch(url):
        if "20201018123503" in url:
            raise OSError("replay unavailable")
        return payload

    monkeypatch.setattr(archive_inventory, "ARTIFACTS", (artifact,))
    monkeypatch.setattr(archive_inventory, "list_captures", lambda key: listing)
    monkeypatch.setattr(archive_inventory, "fetch", fake_fetch)
    monkeypatch.setattr(archive_inventory, "sleep", lambda seconds: None)
    captures = archive_inventory.collect_captures(datetime(2026, 9, 13, 15, tzinfo=UTC))
    assert [(c.source, c.status, bool(c.outliers)) for c in captures] == [
        ("bls_current", "200", True),
        ("internet_archive", "200", True),
        ("internet_archive", "301", False),
        ("internet_archive", "unreplayable", False),
    ]
    assert (
        captures[1].url == f"https://web.archive.org/web/20150321061757id_/{original}"
    )


@pytest.mark.network
def test_live_cdx_lists_2014_captures_of_the_other_inputs_zip():
    rows = list_captures("bls.gov/web/empsit/ces.spec.other.zip")
    assert any(moment.year == 2014 and status == "200" for moment, _, status, _ in rows)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: FAIL at collection with `ImportError: cannot import name 'Capture' from 'archive_inventory'`.

- [ ] **Step 3: Replace the script's import block**

In `scripts/archive_inventory.py`, replace the import block (from `import csv` through `from zoneinfo import ZoneInfo`) with:

```python
import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, time
from functools import partial
from pathlib import Path, PurePosixPath
from time import sleep
from zoneinfo import ZoneInfo
```

`datetime.time` is imported as `time` and `time.sleep` as `sleep`, so the two never collide.

- [ ] **Step 4: Append the Captures section**

Append after the Network section:

```python
# --- Captures ------------------------------------------------------------------------------

FILE_TYPES = ("specification", "prior_adjustment", "outliers")
_MEMBER_PATTERNS = {
    "specification": re.compile(r"\.spc$"),
    "prior_adjustment": re.compile(r"^prior_adjustment_file\.xlsx?$"),
    "outliers": re.compile(r"^outliers\.xlsx?$"),
}
# Members the ZIPs' readme files account for: the readmes and the calendar regressor files.
_EXPLAINED = re.compile(
    r"^(readme[.\w]*\.txt|new\.readme\.announcement\.txt|f?dum\w*\.dat)$"
)


def inspect_zip(payload: bytes) -> dict[str, str]:
    """Fingerprint each Req 9 file type in a seasonal-adjustment ZIP and list unexplained members.

    A fingerprint hashes member names and CRC-32s, so re-zipping identical files under new
    timestamps keeps it and any change of contents alters it. An empty fingerprint means the
    file type is absent.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [
            (PurePosixPath(info.filename).name.lower(), info.CRC)
            for info in archive.infolist()
            if not info.is_dir()
        ]
    result = {}
    for file_type, pattern in _MEMBER_PATTERNS.items():
        matched = sorted((name, crc) for name, crc in members if pattern.search(name))
        listing = "\n".join(f"{name} {crc:08x}" for name, crc in matched)
        result[file_type] = (
            hashlib.sha256(listing.encode()).hexdigest()[:16] if matched else ""
        )
    result["unexpected"] = ";".join(
        sorted(
            name
            for name, _ in members
            if not _EXPLAINED.match(name)
            and not any(pattern.search(name) for pattern in _MEMBER_PATTERNS.values())
        )
    )
    return result


@dataclass(frozen=True)
class Capture:
    """One copy of a seasonal-adjustment ZIP: an Internet Archive capture or BLS's current file."""

    artifact: str
    source: str  # "internet_archive" or "bls_current"
    timestamp: datetime  # UTC: when the copy was made, or when BLS's file was fetched
    url: str  # where these bytes can be fetched again
    status: str
    # The base-32 SHA-1 of the payload, the Internet Archive's own digest format.
    digest: str
    specification: str = ""
    prior_adjustment: str = ""
    outliers: str = ""
    unexpected: str = ""


CAPTURE_COLUMNS = [field.name for field in fields(Capture)]
_TIMESTAMP = "%Y-%m-%dT%H:%M:%SZ"


def capture_row(capture: Capture) -> dict[str, str]:
    return {**asdict(capture), "timestamp": capture.timestamp.strftime(_TIMESTAMP)}


def capture_from_row(row: dict[str, str]) -> Capture:
    stamp = datetime.strptime(row["timestamp"], _TIMESTAMP).replace(tzinfo=UTC)
    return Capture(**{**row, "timestamp": stamp})


_CDX_LINE = re.compile(r"^(\d{14}) (\S+) (\S+) (\S+)$")


def parse_cdx(text: str) -> list[tuple[datetime, str, str, str]]:
    """Parse CDX lines of `timestamp original statuscode digest`, rejecting anything else.

    The CDX server can answer a throttled request with an HTML page, which must fail loudly
    rather than read as an empty list of captures.
    """
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _CDX_LINE.match(line.strip())
        if match is None:
            raise ValueError(f"not a CDX line: {line[:80]!r}")
        stamp, original, status, digest = match.groups()
        moment = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        rows.append((moment, original, status, digest))
    return rows


@dataclass(frozen=True)
class Artifact:
    name: str
    current_url: str
    archive_keys: tuple[str, ...]  # CDX URL keys, including the pre-2012 FTP location


# Only the all-employees specification set is inventoried: hours and earnings are out of scope,
# and the second-preliminary sets adjust detailed series that do not aggregate to total nonfarm.
ARTIFACTS = (
    Artifact(
        "all_employees_specifications",
        "https://www.bls.gov/web/empsit/ces.spec.ae.zip",
        (
            "bls.gov/web/empsit/ces.spec.ae.zip",
            "ftp.bls.gov/pub/suppl/empsit.ces.spec.ae.zip",
        ),
    ),
    Artifact(
        "other_inputs",
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        (
            "bls.gov/web/empsit/ces.spec.other.zip",
            "ftp.bls.gov/pub/suppl/empsit.ces.spec.other.zip",
        ),
    ),
)
CDX_URL = "https://web.archive.org/cdx/search/cdx?url={key}&fl=timestamp,original,statuscode,digest"
REPLAY_URL = "https://web.archive.org/web/{stamp}id_/{original}"
ATTEMPTS = 5
PAUSE_SECONDS = 3.0


def retrying[T](attempt: Callable[[], T], *, what: str) -> T:
    """Retry `attempt` with a growing pause, because the Internet Archive throttles bursts."""
    retryable = (OSError, ValueError, zipfile.BadZipFile)
    for number in range(1, ATTEMPTS):
        try:
            return attempt()
        except retryable:
            sleep(PAUSE_SECONDS * number)
    try:
        return attempt()
    except retryable as error:
        raise RuntimeError(f"{what} failed after {ATTEMPTS} attempts") from error


def list_captures(key: str) -> list[tuple[datetime, str, str, str]]:
    return parse_cdx(fetch(CDX_URL.format(key=key)).decode("utf-8", "replace"))


def payload_digest(payload: bytes) -> str:
    return base64.b32encode(
        hashlib.sha1(payload, usedforsecurity=False).digest()
    ).decode()


def fetch_and_inspect(url: str) -> tuple[bytes, dict[str, str]]:
    payload = fetch(url)
    return payload, inspect_zip(payload)


def collect_captures(now: datetime) -> list[Capture]:
    """Fetch BLS's current copy and every Internet Archive capture of each artifact."""
    captures = []
    for artifact in ARTIFACTS:
        payload, found = retrying(
            partial(fetch_and_inspect, artifact.current_url), what=artifact.current_url
        )
        captures.append(
            Capture(
                artifact.name,
                "bls_current",
                now,
                artifact.current_url,
                "200",
                payload_digest(payload),
                **found,
            )
        )
        for key in artifact.archive_keys:
            for moment, original, status, digest in retrying(
                partial(list_captures, key), what=f"CDX listing of {key}"
            ):
                replay = REPLAY_URL.format(
                    stamp=f"{moment:%Y%m%d%H%M%S}", original=original
                )
                found, outcome = {}, status
                if status == "200":
                    sleep(PAUSE_SECONDS)
                    try:
                        _, found = retrying(
                            partial(fetch_and_inspect, replay), what=replay
                        )
                    except RuntimeError as error:
                        print(f"{error}; recorded as unreplayable", file=sys.stderr)
                        outcome = "unreplayable"
                captures.append(
                    Capture(
                        artifact.name,
                        "internet_archive",
                        moment,
                        replay,
                        outcome,
                        digest,
                        **found,
                    )
                )
    return captures
```

Test doubles replace `fetch`, `list_captures`, and `sleep` through module attributes, so each helper must look those names up at call time, as written above. Do not bind them to locals.

- [ ] **Step 5: Append the command line**

Append at the end of the file:

```python
# --- Command line --------------------------------------------------------------------------


def capture_evidence() -> int:
    """Network step: write the vintage list and every copy's fingerprints."""
    now = datetime.now(UTC).replace(microsecond=0)
    index = retrying(partial(fetch, RELEASE_INDEX_URL), what=RELEASE_INDEX_URL)
    vintages = select_vintages(
        parse_release_index(index.decode("utf-8", "replace")), as_of=now.date()
    )
    write_csv(VINTAGES_PATH, vintage_rows(vintages), VINTAGE_COLUMNS)
    captures = collect_captures(now)
    write_csv(
        CAPTURES_PATH, [capture_row(capture) for capture in captures], CAPTURE_COLUMNS
    )
    unexplained = [capture for capture in captures if capture.unexpected]
    for capture in unexplained:
        print(
            f"unexplained members in {capture.url}: {capture.unexpected}",
            file=sys.stderr,
        )
    return 1 if unexplained else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("captures",))
    parser.parse_args(argv)
    return capture_evidence()


if __name__ == "__main__":
    raise SystemExit(main())
```

The evidence CSVs are written before the exit code is decided, so an unexplained member never costs the capture.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: PASS, 19 passed, 2 deselected.

Run: `uv run pytest tests/test_archive_inventory.py -q -m network` (with `BLS_CONTACT_EMAIL` exported)
Expected: PASS, 2 passed.

- [ ] **Step 7: Capture the evidence**

Run in the background, with `BLS_CONTACT_EMAIL` exported:

```bash
uv run python scripts/archive_inventory.py captures
```

Expected: exit 0 after several minutes, with `docs/inventory/es-vintages.csv` and `docs/inventory/archive-captures.csv` written. Lines on stderr that end in `recorded as unreplayable` are allowed.

On exit 1, read the `unexplained members` lines:
- **A documented non-data input.** The member is a seasonal-adjustment input that `cesseasadjtn.htm` documents and that is not NSA data, such as a metafile or a readme under another name. Add its pattern to `_EXPLAINED` with a comment citing the technical notes. Add a test asserting `inspect_zip` explains that member name, then rerun `captures`.
- **Anything that could hold NSA input data.** Stop and report: it would change the unrounded-inputs finding.

- [ ] **Step 8: Summarize the evidence**

```bash
uv run python - <<'EOF'
import sys
from collections import Counter

sys.path.insert(0, "scripts")
from archive_inventory import (
    CAPTURES_PATH,
    FILE_TYPES,
    VINTAGES_PATH,
    missing_reference_months,
    read_csv,
    vintages_from_rows,
)

captures = read_csv(CAPTURES_PATH)
print(Counter((row["artifact"], row["source"], row["status"]) for row in captures))
for file_type in FILE_TYPES:
    stamps = sorted(
        row["timestamp"] for row in captures if row["status"] == "200" and row[file_type]
    )
    print(file_type, len(stamps), stamps[:1], stamps[-1:])
vintages = vintages_from_rows(read_csv(VINTAGES_PATH))
print(len(vintages), vintages[0].reference_month, vintages[-1])
print([f"{month:%Y-%m}" for month in missing_reference_months(vintages)])
EOF
```

Expected:
- One `bls_current` row with status `200` per artifact, and no rows from the `ftp.bls.gov` keys.
- The earliest specification-holding copy falls in 2014, and the earliest prior-adjustment and outlier copy is 2014-07-19.
- The missing-months list prints `['2025-10']`.
- The last vintage's release date is on or before today.

Any other outcome is a stop condition.

- [ ] **Step 9: Commit**

```bash
uv run ruff format scripts tests
uv run ruff check scripts tests
git add scripts/archive_inventory.py tests/test_archive_inventory.py docs/inventory/es-vintages.csv docs/inventory/archive-captures.csv
git commit -m "Capture BLS and Internet Archive copies of the seasonal-adjustment files"
```

### Task 4: Per-vintage archive inventory and the archive section

**Files:**
- Modify: `scripts/archive_inventory.py` (insert the Inventory section before `# --- Command line`; replace `main`)
- Modify: `tests/test_archive_inventory.py` (imports, the `capture` helper, the inventory tests)
- Modify: `tests/test_review_document.py` (imports, `ARCHIVE_SUBSECTIONS`, the archive tests)
- Modify: `docs/ces-revisions-review.md` (the archive section)
- Create (generated): `docs/inventory/archive-inventory.csv`

**Interfaces:**
- Consumes:
  - From Task 2: `Vintage`, `release_instant`, `next_month`, `read_csv`, `write_csv`, `vintages_from_rows`, `FIRST_REFERENCE_MONTH`, `REVIEW_PATH`, `INVENTORY_PATH`.
  - From Task 3: `Capture`, `FILE_TYPES`, `capture_from_row`, `capture_evidence`, `VINTAGES_PATH`, `CAPTURES_PATH`, and the two evidence CSVs.
- Produces:
  - `PRESENT = ("bls_current", "internet_archive")`.
  - `STATUSES`, which adds `"conflicting_captures"`, `"not_archived"`, and `"no_release"` to `PRESENT`.
  - `INVENTORY_COLUMNS`: `reference_month`, `release_date`, the three file types, `unrounded_nsa_inputs`, `files_complete`, and one `<file type>_evidence` column per file type.
  - `live_windows(vintages, *, annual: bool) -> dict[date, tuple[datetime, datetime | None]]`.
  - `file_status(captures, file_type, window) -> tuple[str, str]`.
  - `inventory_rows(vintages, captures) -> list[dict[str, str]]`.
  - `render_blocks(rows) -> dict[str, str]`, keyed `archive-findings`, `archive-coverage`, and `archive-vintages`.
  - `generated_block(document, name) -> str` and `replace_generated(document, name, body) -> str`.
  - `derive_inventory() -> int`.
  - `main`, accepting `captures` and `inventory`.
  - The document's generated-block markers, `<!-- BEGIN GENERATED <name> -->` and `<!-- END GENERATED <name> -->`.
- Test helper: `capture(timestamp: str, **fingerprints: str) -> Capture`.

- [ ] **Step 1: Write the failing inventory tests**

In `tests/test_archive_inventory.py`, add `file_status`, `generated_block`, `inventory_rows`, `live_windows`, `render_blocks`, and `replace_generated` to the `from archive_inventory import (...)` list. `ruff check --fix` sorts the list.

Add this helper directly after `make_zip`:

```python
def capture(timestamp: str, **fingerprints: str) -> Capture:
    return Capture(
        "other_inputs",
        "internet_archive",
        datetime.fromisoformat(timestamp),
        f"https://web.archive.org/web/{timestamp}",
        "200",
        "DIGEST",
        **fingerprints,
    )
```

Append at the end of the file:

```python
# --- Inventory -----------------------------------------------------------------------------


def test_monthly_windows_run_from_one_release_to_the_next():
    vintages = [
        vintage("2025-07", "2025-08-01"),
        vintage("2025-08", "2025-09-05"),
        vintage("2025-09", "2025-11-20"),
    ]
    windows = live_windows(vintages, annual=False)
    assert windows[date(2025, 8, 1)] == (
        release_instant(date(2025, 9, 5)),
        release_instant(date(2025, 11, 20)),
    )
    assert windows[date(2025, 9, 1)] == (release_instant(date(2025, 11, 20)), None)


def test_specification_windows_run_from_benchmark_release_to_benchmark_release():
    vintages = [
        vintage("2024-12", "2025-01-10"),
        vintage("2025-01", "2025-02-07"),
        vintage("2025-02", "2025-03-07"),
        vintage("2025-12", "2026-01-09"),
        vintage("2026-01", "2026-02-11"),
    ]
    windows = live_windows(vintages, annual=True)
    benchmark_2025 = (
        release_instant(date(2025, 2, 7)),
        release_instant(date(2026, 2, 11)),
    )
    assert windows[date(2025, 1, 1)] == benchmark_2025
    assert windows[date(2025, 12, 1)] == benchmark_2025
    assert windows[date(2026, 1, 1)] == (release_instant(date(2026, 2, 11)), None)


def test_a_capture_a_minute_before_the_embargo_belongs_to_the_previous_vintage():
    windows = live_windows(
        [vintage("2025-07", "2025-08-01"), vintage("2025-08", "2025-09-05")],
        annual=False,
    )
    before = capture("2025-09-05T12:29:00+00:00", outliers="a")
    after = capture("2025-09-05T12:31:00+00:00", outliers="b")
    assert file_status([before, after], "outliers", windows[date(2025, 7, 1)]) == (
        "internet_archive",
        before.url,
    )
    assert file_status([before, after], "outliers", windows[date(2025, 8, 1)]) == (
        "internet_archive",
        after.url,
    )


def test_file_status_prefers_the_bls_copy_and_ignores_captures_without_content():
    window = (datetime(2026, 9, 4, 12, 30, tzinfo=UTC), None)
    archived = capture("2026-09-05T00:00:00+00:00", outliers="same")
    current = Capture(
        "other_inputs",
        "bls_current",
        datetime(2026, 9, 13, 15, 0, tzinfo=UTC),
        "https://www.bls.gov/web/empsit/ces.spec.other.zip",
        "200",
        "DIGEST",
        outliers="same",
    )
    redirect = Capture(
        "other_inputs",
        "internet_archive",
        datetime(2026, 9, 6, tzinfo=UTC),
        "https://web.archive.org/web/redirect",
        "301",
        "DIGEST",
    )
    assert file_status([archived, current, redirect], "outliers", window) == (
        "bls_current",
        current.url,
    )


def test_file_status_flags_copies_that_disagree_within_one_window():
    window = (
        datetime(2021, 2, 5, 13, 30, tzinfo=UTC),
        datetime(2021, 3, 5, 13, 30, tzinfo=UTC),
    )
    first = capture("2021-02-10T00:00:00+00:00", prior_adjustment="a")
    second = capture("2021-02-20T00:00:00+00:00", prior_adjustment="b")
    assert file_status([second, first], "prior_adjustment", window) == (
        "conflicting_captures",
        f"{first.url} {second.url}",
    )


def test_file_status_without_a_copy_is_not_archived():
    window = (
        datetime(2008, 2, 1, 13, 30, tzinfo=UTC),
        datetime(2008, 3, 7, 13, 30, tzinfo=UTC),
    )
    assert file_status([], "specification", window) == ("not_archived", "")


def test_inventory_rows_cover_every_reference_month_and_mark_the_gap():
    vintages = [
        vintage("2003-04", "2003-05-02"),
        vintage("2003-05", "2003-06-06"),
        vintage("2003-06", "2003-07-03"),
        vintage("2003-08", "2003-09-05"),
    ]
    rows = inventory_rows(vintages, [])
    assert [row["reference_month"] for row in rows] == [
        "2003-05",
        "2003-06",
        "2003-07",
        "2003-08",
    ]
    gap = rows[2]
    assert (gap["release_date"], gap["specification"], gap["unrounded_nsa_inputs"]) == (
        "",
        "no_release",
        "no_release",
    )
    assert (rows[0]["prior_adjustment"], rows[0]["unrounded_nsa_inputs"]) == (
        "not_archived",
        "not_published",
    )


def test_files_complete_needs_all_three_file_types():
    vintages = [vintage("2003-05", "2003-06-06")]
    all_three = capture(
        "2003-06-10T00:00:00+00:00",
        specification="s",
        prior_adjustment="p",
        outliers="o",
    )
    two = capture("2003-06-10T00:00:00+00:00", specification="s", prior_adjustment="p")
    assert inventory_rows(vintages, [all_three])[0]["files_complete"] == "true"
    assert inventory_rows(vintages, [two])[0]["files_complete"] == "false"


def test_generated_blocks_are_replaced_in_place():
    document = (
        "intro\n<!-- BEGIN GENERATED archive-coverage -->\nold\n"
        "<!-- END GENERATED archive-coverage -->\noutro\n"
    )
    updated = replace_generated(document, "archive-coverage", "new\n")
    assert updated == document.replace("\nold\n", "\nnew\n")
    assert generated_block(updated, "archive-coverage") == "new\n"


def test_a_missing_generated_block_is_an_error():
    with pytest.raises(ValueError, match="archive-vintages"):
        replace_generated("no markers here", "archive-vintages", "body\n")


def test_rendered_vintage_rows_link_the_copy_that_evidences_them():
    kept = capture("2003-06-10T00:00:00+00:00", outliers="o")
    rows = inventory_rows([vintage("2003-05", "2003-06-06")], [kept])
    assert (
        f"| 2003-05 | 2003-06-06 | `not_archived` | `not_archived` | [`internet_archive`]({kept.url}) "
        "| `not_published` | false |"
    ) in render_blocks(rows)["archive-vintages"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: FAIL at collection with `ImportError: cannot import name 'file_status' from 'archive_inventory'`.

- [ ] **Step 3: Insert the Inventory section**

In `scripts/archive_inventory.py`, insert this section between the Captures section and `# --- Command line`:

```python
# --- Inventory -----------------------------------------------------------------------------

PRESENT = ("bls_current", "internet_archive")
STATUSES = (*PRESENT, "conflicting_captures", "not_archived", "no_release")
INVENTORY_COLUMNS = [
    "reference_month",
    "release_date",
    *FILE_TYPES,
    "unrounded_nsa_inputs",
    "files_complete",
    *(f"{file_type}_evidence" for file_type in FILE_TYPES),
]
_FILE_LABELS = {
    "specification": "Specification files",
    "prior_adjustment": "Prior-adjustment files",
    "outliers": "Outlier files",
}


def live_windows(
    vintages: list[Vintage], *, annual: bool
) -> dict[date, tuple[datetime, datetime | None]]:
    """When each vintage's files were the ones BLS served.

    Prior-adjustment and outlier files change with every release, so a vintage's window runs
    from its release to the next. Specification files change only with the annual benchmark,
    released with January estimates, so their window runs from one benchmark release to the
    next. A vintage before the first benchmark release in the list opens at the list's start.
    """
    opens = [release_instant(vintage.release_date) for vintage in vintages]
    boundaries = [
        index
        for index, vintage in enumerate(vintages)
        if not annual or vintage.reference_month.month == 1
    ]
    windows = {}
    for index, vintage in enumerate(vintages):
        start = max(
            (boundary for boundary in boundaries if boundary <= index), default=0
        )
        end = min(
            (boundary for boundary in boundaries if boundary > index), default=None
        )
        windows[vintage.reference_month] = (
            opens[start],
            None if end is None else opens[end],
        )
    return windows


def file_status(
    captures: list[Capture], file_type: str, window: tuple[datetime, datetime | None]
) -> tuple[str, str]:
    """One file type's status for one vintage, with the URL of the copy that evidences it."""
    start, end = window
    holding = sorted(
        (
            capture
            for capture in captures
            if capture.status == "200"
            and getattr(capture, file_type)
            and start <= capture.timestamp
            and (end is None or capture.timestamp < end)
        ),
        key=lambda capture: capture.timestamp,
    )
    if not holding:
        return "not_archived", ""
    if len({getattr(capture, file_type) for capture in holding}) > 1:
        return "conflicting_captures", " ".join(capture.url for capture in holding)
    chosen = min(holding, key=lambda capture: capture.source != "bls_current")
    return chosen.source, chosen.url


def inventory_rows(
    vintages: list[Vintage], captures: list[Capture]
) -> list[dict[str, str]]:
    """One row per reference month from May 2003 through the latest vintage."""
    windows = {
        "specification": live_windows(vintages, annual=True),
        "prior_adjustment": live_windows(vintages, annual=False),
        "outliers": live_windows(vintages, annual=False),
    }
    released = {vintage.reference_month: vintage for vintage in vintages}
    rows = []
    month = FIRST_REFERENCE_MONTH
    while month <= vintages[-1].reference_month:
        row = dict.fromkeys(INVENTORY_COLUMNS, "")
        row["reference_month"] = f"{month:%Y-%m}"
        vintage = released.get(month)
        if vintage is None:
            row.update(dict.fromkeys(FILE_TYPES, "no_release"))
            row.update(unrounded_nsa_inputs="no_release", files_complete="false")
        else:
            row["release_date"] = vintage.release_date.isoformat()
            for file_type in FILE_TYPES:
                row[file_type], row[f"{file_type}_evidence"] = file_status(
                    captures, file_type, windows[file_type][month]
                )
            # cesseasadjtn.htm: X-13 runs on unrounded NSA data, and BLS publishes rounded data.
            row["unrounded_nsa_inputs"] = "not_published"
            complete = all(row[file_type] in PRESENT for file_type in FILE_TYPES)
            row["files_complete"] = "true" if complete else "false"
        rows.append(row)
        month = next_month(month)
    return rows


def _status_cell(row: dict[str, str], file_type: str) -> str:
    status = row[file_type]
    if status in PRESENT:
        return f"[`{status}`]({row[f'{file_type}_evidence']})"
    return f"`{status}`"


def render_findings(rows: list[dict[str, str]]) -> str:
    released = [row for row in rows if row["release_date"]]
    gaps = [row["reference_month"] for row in rows if not row["release_date"]]
    lines = [
        (
            f"- Releases inventoried: {len(released)}, estimating reference months "
            f"{released[0]['reference_month']} through {released[-1]['reference_month']} "
            f"(released {released[0]['release_date']} to {released[-1]['release_date']})."
        ),
        f"- Reference months without a release of their own: {', '.join(gaps) or 'none'}.",
    ]
    for file_type, label in _FILE_LABELS.items():
        present = [
            row["reference_month"] for row in released if row[file_type] in PRESENT
        ]
        earliest = present[0] if present else "none"
        lines.append(
            f"- {label} survive for {len(present)} of {len(released)} releases; "
            f"the earliest such release estimates {earliest}."
        )
    complete = sum(row["files_complete"] == "true" for row in released)
    conflicts = sum(
        row[file_type] == "conflicting_captures"
        for row in released
        for file_type in FILE_TYPES
    )
    lines.append(
        f"- All three file types survive for {complete} of {len(released)} releases, "
        f"and {conflicts} cells are `conflicting_captures`."
    )
    lines.append("- Unrounded NSA inputs are `not_published` for every release.")
    return "\n".join(lines) + "\n"


def render_coverage(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Reference year | Releases | Specification | Prior adjustment | Outliers | All three |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    years = sorted({row["reference_month"][:4] for row in rows})
    for year in [*years, "Total"]:
        subset = [
            row
            for row in rows
            if row["release_date"] and year in ("Total", row["reference_month"][:4])
        ]
        counts = [str(sum(row[t] in PRESENT for row in subset)) for t in FILE_TYPES]
        complete = sum(row["files_complete"] == "true" for row in subset)
        lines.append(f"| {year} | {len(subset)} | {' | '.join(counts)} | {complete} |")
    return "\n".join(lines) + "\n"


def render_vintages(rows: list[dict[str, str]]) -> str:
    lines = [
        "<details>",
        "<summary>One row per reference month from May 2003</summary>",
        "",
        (
            "| Reference month | Release | Specification | Prior adjustment | Outliers "
            "| Unrounded NSA inputs | All three |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row["reference_month"],
            row["release_date"] or "none",
            *(_status_cell(row, file_type) for file_type in FILE_TYPES),
            f"`{row['unrounded_nsa_inputs']}`",
            row["files_complete"],
        ]
        lines.append(f"| {' | '.join(cells)} |")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def render_blocks(rows: list[dict[str, str]]) -> dict[str, str]:
    """The generated blocks of the review's archive section, keyed by marker name."""
    blocks = {
        "archive-findings": render_findings(rows),
        "archive-coverage": render_coverage(rows),
        "archive-vintages": render_vintages(rows),
    }
    # Blank lines keep each table or list a block of its own beside the HTML-comment markers.
    return {name: f"\n{body}\n" for name, body in blocks.items()}


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED {name} -->\n", f"<!-- END GENERATED {name} -->"


def _block_span(document: str, name: str) -> tuple[int, int]:
    begin, end = _markers(name)
    start = document.find(begin)
    stop = document.find(end, start) if start >= 0 else -1
    if stop < 0:
        raise ValueError(f"document has no generated block named {name!r}")
    return start + len(begin), stop


def generated_block(document: str, name: str) -> str:
    start, stop = _block_span(document, name)
    return document[start:stop]


def replace_generated(document: str, name: str, body: str) -> str:
    start, stop = _block_span(document, name)
    return document[:start] + body + document[stop:]
```

A conflict is flagged whenever the fingerprints inside one window differ, even when one of the copies is BLS's current file, so a mid-year specification change can never hide behind the annual window.

- [ ] **Step 4: Add the offline command**

In `scripts/archive_inventory.py`, replace `def main(...)` and its body with:

```python
def derive_inventory() -> int:
    """Offline step: derive the inventory from the evidence and regenerate the review's tables."""
    vintages = vintages_from_rows(read_csv(VINTAGES_PATH))
    captures = [capture_from_row(row) for row in read_csv(CAPTURES_PATH)]
    rows = inventory_rows(vintages, captures)
    write_csv(INVENTORY_PATH, rows, INVENTORY_COLUMNS)
    document = REVIEW_PATH.read_text(encoding="utf-8")
    for name, body in render_blocks(rows).items():
        document = replace_generated(document, name, body)
    REVIEW_PATH.write_text(document, encoding="utf-8")
    return 0


COMMANDS = {"captures": capture_evidence, "inventory": derive_inventory}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    return COMMANDS[parser.parse_args(argv).command]()
```

Keep the `if __name__ == "__main__":` block that follows.

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_archive_inventory.py -q -m "not network"`
Expected: PASS, 30 passed, 2 deselected.

- [ ] **Step 6: Write the failing document tests**

In `tests/test_review_document.py`, replace the `from review_document import ...` line with:

```python
from archive_inventory import (
    CAPTURES_PATH,
    FILE_TYPES,
    FIRST_REFERENCE_MONTH,
    INVENTORY_COLUMNS,
    INVENTORY_PATH,
    STATUSES,
    VINTAGES_PATH,
    capture_from_row,
    generated_block,
    inventory_rows,
    next_month,
    read_csv,
    render_blocks,
    vintages_from_rows,
)
from review_document import (
    REVIEW_PATH,
    headings,
    is_primary,
    links,
    outside_fences,
    read_review,
    section,
)
```

Add this constant directly after `RULINGS`:

```python
ARCHIVE_SUBSECTIONS = [
    "What BLS publishes",
    "Method",
    "Findings",
    "Coverage by year",
    "Per-vintage inventory",
    "Unrounded NSA inputs",
    "Consequences for later stages",
]
```

Append at the end of the file:

```python
# --- Task 4: seasonal-adjustment archive inventory -----------------------------------------


def archive_rows() -> list[dict[str, str]]:
    return read_csv(INVENTORY_PATH)


def test_archive_section_has_its_subsections():
    body = section(read_review(), "Seasonal-adjustment archive inventory", level=2)
    assert headings(body, level=3) == ARCHIVE_SUBSECTIONS


def test_archive_inventory_has_one_row_per_reference_month_from_may_2003():
    rows = archive_rows()
    expected, month = [], FIRST_REFERENCE_MONTH
    while len(expected) < len(rows):
        expected.append(f"{month:%Y-%m}")
        month = next_month(month)
    assert [row["reference_month"] for row in rows] == expected
    assert list(rows[0]) == INVENTORY_COLUMNS
    assert {row[file_type] for row in rows for file_type in FILE_TYPES} <= set(STATUSES)
    assert {row["unrounded_nsa_inputs"] for row in rows} <= {
        "not_published",
        "no_release",
    }


def test_archive_inventory_is_derived_from_the_committed_evidence():
    vintages = vintages_from_rows(read_csv(VINTAGES_PATH))
    captures = [capture_from_row(row) for row in read_csv(CAPTURES_PATH)]
    assert inventory_rows(vintages, captures) == archive_rows()


def test_no_copy_of_a_seasonal_adjustment_zip_holds_an_unexplained_file():
    captures = read_csv(CAPTURES_PATH)
    assert [row["url"] for row in captures if row["unexpected"]] == []


@pytest.mark.parametrize(
    "name", ["archive-findings", "archive-coverage", "archive-vintages"]
)
def test_archive_tables_in_the_review_match_the_inventory(name):
    assert generated_block(read_review(), name) == render_blocks(archive_rows())[name]


def test_archive_section_cites_primary_sources_only():
    urls = links(
        section(read_review(), "Seasonal-adjustment archive inventory", level=2)
    )
    assert urls
    assert [url for url in urls if not is_primary(url)] == []
```

- [ ] **Step 7: Run the document tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q`
Expected: FAIL. `test_archive_section_has_its_subsections` fails on `[] == ARCHIVE_SUBSECTIONS`. The CSV-based tests error with `FileNotFoundError` for `archive-inventory.csv`, and the generated-block tests with `ValueError: document has no generated block`. The 14 Task 1 tests, and `test_no_copy_of_a_seasonal_adjustment_zip_holds_an_unexplained_file` on Task 3's evidence, pass.

- [ ] **Step 8: Re-verify the statements the section cites**

With `BLS_CONTACT_EMAIL` exported:

```bash
UA="ces-revisions/0.1.0 ($BLS_CONTACT_EMAIL)"
curl -s -A "$UA" https://www.bls.gov/web/empsit/cesseasadj.htm -o /tmp/cesseasadj.htm
curl -s -A "$UA" https://www.bls.gov/web/empsit/cesseasadjtn.htm -o /tmp/cesseasadjtn.htm
curl -s -A "$UA" https://www.bls.gov/web/empsit/cesbmkarch.htm -o /tmp/cesbmkarch.htm
curl -s 'https://web.archive.org/web/20110312035934id_/http://www.bls.gov:80/web/empsit/cesseasadj.htm' -o /tmp/cesseasadj-2011.htm
uv run python - <<'EOF'
import re
from pathlib import Path


def page_text(path):
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).lower()


checks = {
    "/tmp/cesseasadj.htm": [
        "availability of historical input files",
        "model specifications tables for 2003 to 2013",
    ],
    "/tmp/cesseasadjtn.htm": [
        "uses unrounded data when running seasonal adjustment",
        "remain fixed during the year",
        "contains unrounded data",
    ],
}
for path, phrases in checks.items():
    for phrase in phrases:
        print(phrase in page_text(path), "|", phrase)
print(sorted(set(re.findall(r"ces-benchmark-revision-\d{4}\.pdf", Path("/tmp/cesbmkarch.htm").read_text()))))
print(sorted(set(re.findall(r"ftp://ftp\.bls\.gov/pub/suppl/[\w.]+\.zip", Path("/tmp/cesseasadj-2011.htm").read_text()))))
EOF
```

Expected:
- Five `True` lines.
- The benchmark-article list runs from `ces-benchmark-revision-2002.pdf` to at least `-2025.pdf`.
- Four FTP links, `empsit.ces.spec.{ae,aehe,nonae,other}.zip`.

A `False` line or a missing link is a stop condition. The section below states those facts and must not survive a failed check.

- [ ] **Step 9: Measure the published precision of NSA levels**

```bash
curl -s -A "$UA" https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries -o /tmp/ce.data.0.AllCESSeries
curl -s -A "$UA" https://www.bls.gov/web/empsit/cesvinall.zip -o /tmp/cesvinall.zip
uv run python - <<'EOF'
import re
import zipfile

import polars as pl

# Total nonfarm and the eleven supersectors, all employees, not seasonally adjusted.
CODES = ["00", "10", "20", "30", "40", "50", "55", "60", "65", "70", "80", "90"]
frame = (
    pl.read_csv("/tmp/ce.data.0.AllCESSeries", separator="\t", infer_schema=False)
    .rename(str.strip)
    .with_columns(pl.all().str.strip_chars())
    .filter(
        pl.col("series_id").is_in([f"CEU{code}00000001" for code in CODES]),
        pl.col("year").cast(pl.Int32) >= 2003,
    )
)
print(
    frame.group_by("series_id")
    .agg(
        pl.col("value").str.contains(r"\.\d*[1-9]").sum().alias("nonzero_decimals"),
        pl.len().alias("values"),
    )
    .sort("series_id")
)
with zipfile.ZipFile("/tmp/cesvinall.zip") as archive:
    names = [name for name in archive.namelist() if name.endswith("_NSA.csv")]
    decimal = [n for n in names if re.search(rb",-?\d+\.\d*[1-9]", archive.read(n))]
print(len(names), "NSA vintage files;", len(decimal), "with nonzero decimals:", decimal[:12])
EOF
```

Expected: twelve series rows, each with a `nonzero_decimals` count, then a count of NSA vintage files. Planning saw whole thousands for total nonfarm only. Record both outputs for Step 12.

- [ ] **Step 10: Write the archive section**

In `docs/ces-revisions-review.md`, replace

```markdown
## Seasonal-adjustment archive inventory

**Status:** in progress — plan 2, Tasks 3 and 4.
```

with the text below. Set every `accessed` date to the day Step 8 fetched the pages. Step 12 replaces the two paragraphs marked `PRECISION` and `CHANNEL-B`.

```markdown
## Seasonal-adjustment archive inventory

Req 9 leaves open which vintages have archived specification, prior-adjustment, and outlier files, and whether the unrounded NSA inputs are recoverable. This section settles both for every Employment Situation release from the May 2003 publication vintage onward, from BLS's pages and the Internet Archive's copies of BLS's files.

### What BLS publishes

The [CES seasonal adjustment files page](https://www.bls.gov/web/empsit/cesseasadj.htm) (accessed 2026-09-13) serves only the input files now in use. Three ZIP files hold the X-13ARIMA-SEATS specification files for series adjusted at the first preliminary estimates, among them `ces.spec.ae.zip` for all employees. Three more hold the files for series adjusted independently at the second preliminary estimates, and `ces.spec.other.zip` holds the calendar regressor files, the prior-adjustment file, and the recent-outliers file. The page's section on the availability of historical input files says only when the files change: the specification and prior-adjustment files each year, and the prior-adjustment and recent-outlier files with every monthly release. It links no earlier versions. The model specification tables for 2003 through 2013, which give each series' adjustment mode and calendar treatments but not its specification file, are in the [archived benchmark articles](https://www.bls.gov/web/empsit/cesbmkarch.htm) (accessed 2026-09-13). In March 2011 the page linked the same four ZIP files at `ftp.bls.gov/pub/suppl/`, as the [Internet Archive's copy of the page](https://web.archive.org/web/20110312035934/http://www.bls.gov:80/web/empsit/cesseasadj.htm) shows. Evidence: *documented*.

### Method

[`scripts/archive_inventory.py`](../scripts/archive_inventory.py) builds the inventory in two steps, whose outputs are committed in [`docs/inventory/`](inventory/):

1. `captures` reads the [Employment Situation archive index](https://www.bls.gov/bls/news-release/empsit.htm) into `es-vintages.csv`, one release per reference month, dropping links to releases scheduled after the day it runs. It fetches BLS's current `ces.spec.ae.zip` and `ces.spec.other.zip`, lists every copy of each that the Internet Archive holds at the current and pre-2012 locations, and downloads each copy. It writes each file's fingerprints of its specification, prior-adjustment, and outlier members, by name and CRC-32, to `archive-captures.csv`, and marks a listed copy the Internet Archive cannot serve as `unreplayable`.
2. `inventory` works offline from those two files. A release's prior-adjustment and outlier files were current from its 8:30 a.m. Eastern embargo until the next release. Its specification files were current from the benchmark release, which carries January estimates, until the next benchmark release, because the [seasonal adjustment technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm) (accessed 2026-09-13) say all controllable variables remain fixed during the year. A copy made inside a release's window evidences that release's file.

Each file type gets one status per release: `bls_current` when BLS serves the file today, `internet_archive` when a copy from inside the window holds it, `conflicting_captures` when copies inside one window hold different contents, `not_archived` when no copy does, and `no_release` for a reference month with no release of its own. Only the all-employees specification set is inventoried, because hours and earnings are out of scope and the second-preliminary sets adjust detailed series that do not aggregate to total nonfarm.

### Findings

<!-- BEGIN GENERATED archive-findings -->
<!-- END GENERATED archive-findings -->

### Coverage by year

Releases by the year of their reference month, with how many keep each file type as a `bls_current` or `internet_archive` copy.

<!-- BEGIN GENERATED archive-coverage -->
<!-- END GENERATED archive-coverage -->

### Per-vintage inventory

Each surviving file's status links the copy that evidences it, and [`archive-inventory.csv`](inventory/archive-inventory.csv) holds the same rows with the evidence URLs.

<!-- BEGIN GENERATED archive-vintages -->
<!-- END GENERATED archive-vintages -->

### Unrounded NSA inputs

BLS does not publish them. The [seasonal adjustment technical notes](https://www.bls.gov/web/empsit/cesseasadjtn.htm) describe the input data file of NSA estimates that X-13ARIMA-SEATS reads, from which BLS first removes strikes and other prior adjustments, and they say seasonal adjustment runs on unrounded data while the data BLS publishes are rounded. No copy of either ZIP file, current or archived, holds an input data file: the `unexpected` column of [`archive-captures.csv`](inventory/archive-captures.csv) is empty, so every member is a specification file, a calendar regressor file, a readme, the prior-adjustment file, or the outlier file. The prior-adjustment file is the one unrounded input BLS does publish, according to the technical notes' footnote to figure 2. Evidence: *documented*.

PRECISION

### Consequences for later stages

- **Req 9 channel (b), roadmap Stages 25 and 26.** CHANNEL-B
- **Roadmap Stage 14.** The seasonal-adjustment file store fetches exactly the copies [`archive-inventory.csv`](inventory/archive-inventory.csv) marks `bls_current` or `internet_archive`, so its coverage equals this inventory, and it records every other release as missing rather than silently absent.
- **Roadmap Stage 3.** [`es-vintages.csv`](inventory/es-vintages.csv) lists release dates only. Stage 3's release-date index adds closing and publication dates and must agree with it on every release, including the September 2025 release on 2025-11-20 and the absence of an October 2025 release.
- **The seasonal-flag panel of Req 3.** Calendar regressors and specification regimes are observable release by release only where specification files survive. Elsewhere the model specification tables in the benchmark articles are the fallback, and they cover 2003 through 2013.
```

- [ ] **Step 11: Generate the inventory tables**

Run: `uv run python scripts/archive_inventory.py inventory`
Expected: exit 0. It writes `docs/inventory/archive-inventory.csv` and fills the three generated blocks. Print the findings with:

```bash
sed -n '/^### Findings/,/^### Coverage by year/p' docs/ces-revisions-review.md
```

- [ ] **Step 12: Replace the two marked paragraphs**

Replace the line `PRECISION` from the Step 9 output:
- **When every one of the twelve series shows `nonzero_decimals` 0 and no NSA vintage file has nonzero decimals**, write:

  ```markdown
  The NSA employment levels BLS publishes are whole thousands. Since 2003, no value of total nonfarm or of the eleven supersectors in [`ce.data.0.AllCESSeries`](https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries) carries a nonzero decimal, and neither does any NSA file in the [CES vintage data](https://www.bls.gov/web/empsit/cesvindata.htm) ZIP file (both accessed 2026-09-13). That matches Req 15's treatment of publication rounding as independent error with variance 1/12, in thousands squared. Evidence: *supported*.
  ```
- **Otherwise**, write a paragraph of the same shape. Name each series and vintage file with nonzero decimals, and its count, from the Step 9 output. State that Req 15's variance of 1/12 holds only for levels published in whole thousands, and that a level published to one decimal place has rounding variance 1/1200. End with `Evidence: *supported*.`, and add the finding to the completion report for roadmap Stages 6 and 7.

Replace `CHANNEL-B` from the findings block's line `All three file types survive for N of M releases`:
- **When N is 0**, write:

  ```markdown
  No release keeps all three file types and none has unrounded inputs, so no vintage-specific X-13 run can be reproduced from public files. The published NSA and SA pair of channel (a) is the only observation channel for the factor path, and resume should park Stages 25 and 26.
  ```
- **Otherwise**, write:

  ```markdown
  Only the releases counted above as keeping all three file types can seed a vintage-specific X-13 reproduction, and none of them has unrounded inputs, so Stage 25 can at best match published SA values to rounding from rounded inputs. Every other release carries the channel-(a) flag.
  ```

- [ ] **Step 13: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS, 82 passed, 4 deselected. That is 30 from before this plan, 30 archive tests, and 22 document tests.

- [ ] **Step 14: Commit**

```bash
uv run ruff format scripts tests docs/ces-revisions-review.md
uv run ruff check scripts tests
git add scripts/archive_inventory.py tests/test_archive_inventory.py tests/test_review_document.py docs/ces-revisions-review.md docs/inventory/archive-inventory.csv
git commit -m "Derive the per-vintage archive inventory and write its section"
```

### Task 5: Collection-window and seasonal inventory, and the collection-versus-response ruling

**Files:**
- Modify: `tests/test_review_document.py` (imports, constants, the two checkers, two tests)
- Modify: `docs/ces-revisions-review.md` (the Collection window and Seasonal tables; the first ruling)

**Interfaces:**
- Consumes (Task 1): `REQ3_SERIES`, `RULINGS`, `EVIDENCE_LABELS`, `tables`, `links`, `is_primary`, `section`.
- Produces:
  - Test constants `INVENTORY_TABLE_COLUMNS`, `PLACEHOLDER_CELLS`, `RULING_PARAGRAPHS`, and `_DRAFT_LOCATOR`.
  - Checkers `check_inventory_group(group: str) -> None` and `check_ruling(title: str) -> None`, which Tasks 6 and 7 call.
  - Table format: one pipe table per group with the columns in `INVENTORY_TABLE_COLUMNS`. The `Series` cell is the slug in a code span, `Evidence` is one label, and `Citation` holds only primary links, each followed by its access date, or exactly `*unverified*`.
  - Ruling format: paragraphs opening `**Drafts.**`, `**Primary sources.**` (with lead-in text on the same line, then a bullet list with no blank line between, each bullet carrying at least one link; a bold-only line breaks the conventions test, and a blank line detaches the bullets from the paragraph the checker reads), `**Ruling.**`, `**Evidence:**`, and `**Consequence.**`, in that order, with no other bold-led paragraph.

- [ ] **Step 1: Write the failing inventory and ruling tests**

In `tests/test_review_document.py`, add `EVIDENCE_LABELS` and `tables` to the `from review_document import (...)` list.

Add these constants directly after `REQ3_SERIES`:

```python
INVENTORY_TABLE_COLUMNS = [
    "Series",
    "Vintage coverage",
    "Frequency",
    "Earliest date",
    "Access route",
    "Hand-build constraint",
    "First-publication lag",
    "Evidence",
    "Citation",
]
PLACEHOLDER_CELLS = {"", "-", "—", "?", "n/a", "tbd", "todo"}
```

Add these directly after `RULINGS`:

```python
RULING_PARAGRAPHS = [
    "Drafts.",
    "Primary sources.",
    "Ruling.",
    "Evidence:",
    "Consequence.",
]
_DRAFT_LOCATOR = re.compile(r"\((chatgpt|claude|gemini) §")
```

Append at the end of the file:

```python
# --- Tasks 5 to 7: data-availability inventory and rulings ---------------------------------


def check_inventory_group(group: str) -> None:
    inventory = section(read_review(), "Data-availability inventory", level=2)
    found = tables(section(inventory, group, level=3))
    assert len(found) == 1, f"{group}: expected one table, found {len(found)}"
    table = found[0]
    assert list(table[0]) == INVENTORY_TABLE_COLUMNS, group
    assert [row["Series"].strip("`") for row in table] == REQ3_SERIES[group]
    for row in table:
        series = row["Series"]
        placeholders = [
            column
            for column, cell in row.items()
            if cell.strip("*_ ").lower() in PLACEHOLDER_CELLS
        ]
        assert not placeholders, (series, placeholders)
        assert row["Evidence"].strip("*_ ") in EVIDENCE_LABELS, series
        cited = links(row["Citation"])
        verified = bool(cited) and all(is_primary(url) for url in cited)
        assert verified or row["Citation"] == "*unverified*", (series, row["Citation"])


def check_ruling(title: str) -> None:
    rulings = section(read_review(), "Draft disagreements resolved", level=2)
    body = section(rulings, title, level=3)
    blocks = [block.strip() for block in body.split("\n\n") if block.strip()]
    paragraphs = {
        block.split("**")[1]: block for block in blocks if block.startswith("**")
    }
    assert list(paragraphs) == RULING_PARAGRAPHS, (title, list(paragraphs))
    assert set(_DRAFT_LOCATOR.findall(paragraphs["Drafts."])) == {
        "chatgpt",
        "claude",
        "gemini",
    }
    _, *bullets = paragraphs["Primary sources."].splitlines()
    assert bullets, title
    assert [bullet for bullet in bullets if not links(bullet)] == [], title
    sources = links(paragraphs["Primary sources."])
    assert [url for url in sources if not is_primary(url)] == [], title
    label = paragraphs["Evidence:"].removeprefix("**Evidence:**").strip(" .*_")
    assert label in EVIDENCE_LABELS, (title, label)


def test_collection_window_and_seasonal_inventory():
    check_inventory_group("Collection window")
    check_inventory_group("Seasonal")


def test_ruling_on_collection_rate_versus_response_rate():
    check_ruling("Collection rate versus response rate")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q -k "collection"`
Expected: FAIL.
- `test_collection_window_and_seasonal_inventory` fails with `Collection window: expected one table, found 0`.
- `test_ruling_on_collection_rate_versus_response_rate` fails because its only paragraph is `Status:`.

- [ ] **Step 3: Measure the collection- and response-rate series**

With `BLS_CONTACT_EMAIL` exported:

```bash
UA="ces-revisions/0.1.0 ($BLS_CONTACT_EMAIL)"
curl -s -A "$UA" https://download.bls.gov/pub/time.series/ce/ce.series | grep -E '^CEU00000000C[123]|^CEU05000000RR'
curl -s -A "$UA" https://download.bls.gov/pub/time.series/ce/ce.datatype | grep -E '^(C1|C2|C3|RR)\s'
[ -f /tmp/ce.data.0.AllCESSeries ] || curl -s -A "$UA" https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries -o /tmp/ce.data.0.AllCESSeries
uv run python - <<'EOF'
import polars as pl

IDS = ["CEU00000000C1", "CEU00000000C2", "CEU00000000C3", "CEU05000000RR"]
frame = (
    pl.read_csv("/tmp/ce.data.0.AllCESSeries", separator="\t", infer_schema=False)
    .rename(str.strip)
    .with_columns(pl.all().str.strip_chars())
    .filter(pl.col("series_id").is_in(IDS), pl.col("period") != "M13")
    .with_columns(
        pl.col("year").cast(pl.Int32), pl.col("value").cast(pl.Float64, strict=False)
    )
)
decade = (pl.col("year") // 10 * 10).alias("decade")
with pl.Config(tbl_rows=200):
    print(frame.group_by("series_id", decade).agg(pl.len().alias("months")).sort("series_id", "decade"))
    print(frame.filter(pl.col("year") == 2024).group_by("series_id").agg(pl.col("value").mean().round(1), pl.len()).sort("series_id"))
    print(
        frame.filter(pl.col("year") >= 2015, pl.col("series_id").is_in(["CEU00000000C1", "CEU05000000RR"]))
        .group_by("series_id", "year")
        .agg(pl.col("value").mean().round(1))
        .sort("series_id", "year")
    )
EOF
```

Expected: the `ce.series` lines match Planning evidence. The output then prints months present per series and decade, the 2024 means of all four series, and the yearly means of C1 and RR since 2015. Keep all three tables: they fill the C1–C3 and RR rows and the ruling.

- [ ] **Step 4: Fetch the collection-window sources**

```bash
page() { uv run python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); print(re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",t)))' "$1"; }
fetch_to() { curl -s -L -A "$UA" "$1" -o "$2"; echo "$(wc -c < "$2") bytes  $1"; }
fetch_to https://www.bls.gov/ces/notices/2024/ces-collection-rates-have-been-added-to-the-online-public-database.htm /tmp/c-notice.htm
fetch_to https://www.bls.gov/osmr/response-rates/ /tmp/c-osmr.htm
fetch_to https://files.gao.gov/reports/GAO-26-107538/index.html /tmp/c-gao.htm
fetch_to https://www.bls.gov/opub/hom/ces/data.htm /tmp/c-hom-data.htm
fetch_to https://www.bls.gov/schedule/news_release/empsit.htm /tmp/c-schedule.htm
fetch_to https://www.bls.gov/bls/archived_sched.htm /tmp/c-archived-sched.htm
fetch_to https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/ /tmp/c-opm-holidays.htm
fetch_to https://www.bls.gov/bls/shutdown_2013_empsit_qa.pdf /tmp/c-shutdown-2013.pdf && pdftotext /tmp/c-shutdown-2013.pdf /tmp/c-shutdown-2013.txt
fetch_to https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm /tmp/c-lapse-2025.htm
fetch_to https://www.bls.gov/bls/what-impact-did-the-lapse-of-appropriation-for-some-federal-agencies-have-on-january-employment-data.htm /tmp/c-lapse-2019.htm
fetch_to https://www.bls.gov/web/empsit/cesvininfo.htm /tmp/c-vininfo.htm
fetch_to "$(awk -F, '$1=="2019-04"{print $3}' docs/inventory/es-vintages.csv)" /tmp/c-empsit-2019-04.htm
page /tmp/c-notice.htm | grep -o -i '.\{0,300\}\(collection rate\|response rate\).\{0,300\}' | head -8
page /tmp/c-hom-data.htm | grep -o -i '.\{0,250\}clos\(e\|ing\).\{0,250\}' | head -8
page /tmp/c-empsit-2019-04.htm | grep -o -i '.\{0,200\}collection rate.\{0,200\}' | head -3
```

From these pages, establish:
- **Definitions.** The collection-rate and response-rate definitions, from the notice and the OSMR page.
- **GAO figures.** GAO's fiscal-year CES response rates.
- **The window.** The collection period and the first-closing rule, from the Handbook data chapter. If no primary page states when first closing falls, the `collection_business_days` row says so, with Evidence `not found`.
- **Schedule depth.** The first year of the archived release schedules, and when a year's schedule is posted.
- **Holidays.** The years OPM's holiday tables cover.
- **Lapses.** Each lapse's dates and release changes, from the lapse pages.
- **Earlier publication.** Whether the April 2019 release text reports a collection rate, which shows whether rates were public before the December 2024 database addition. Where a release reports one, cite that release.

- [ ] **Step 5: Inspect the seasonal-adjustment inputs**

```bash
fetch_to https://www.bls.gov/web/empsit/ces.spec.other.zip /tmp/s-other.zip
fetch_to https://www.bls.gov/web/empsit/ces.spec.ae.zip /tmp/s-ae.zip
fetch_to https://www.bls.gov/opub/mlr/2022/article/the-challenges-of-seasonal-adjustment-for-the-current-employment-statistics-survey-during-the-covid-19-pandemic.htm /tmp/s-mlr2022.htm
rm -rf /tmp/sa && mkdir -p /tmp/sa && unzip -q -o /tmp/s-other.zip -d /tmp/sa/other && unzip -q -o /tmp/s-ae.zip -d /tmp/sa/ae
cat /tmp/sa/other/readme*.txt
for f in /tmp/sa/other/*.dat; do echo "== $f"; head -2 "$f"; done
first_spec="$(find /tmp/sa/ae -name '*.spc' | sort | head -1)"; cat "$first_spec"
find /tmp/sa/ae -name '*.spc' -exec grep -h -o -i -E '\b(ao|ls|tc)[0-9]{4}\.[a-z0-9]+\b' {} + | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -20
page /tmp/s-mlr2022.htm | grep -o -i '.\{0,250\}\(additive outlier\|level shift\|temporary change\).\{0,250\}' | head -6
page /tmp/cesseasadj.htm | grep -o -i 'table [0-9]*\. model specifications[^|]\{0,80\}' | head -9
```

From these, establish:
- **Calendar regressors.** Which regressor files carry the four/five-week and the Easter and Labor Day treatments, and the first year of data in them.
- **Treatments in the specs.** How a `.spc` file names its regression and outlier treatments.
- **The 2020–21 outliers.** The additive-outlier, level-shift, and temporary-change regressors dated 2020–2021 in the current specifications.
- **The MLR account.** When the 2022 article says the COVID intervention treatments were adopted.
- **Current tables.** Which model specification tables `cesseasadj.htm` lists now.

- [ ] **Step 6: Write the Collection window and Seasonal tables**

Under `### Collection window` and `### Seasonal` in `docs/ces-revisions-review.md`, write one table each. Both start with this header and separator:

```markdown
| Series | Vintage coverage | Frequency | Earliest date | Access route | Hand-build constraint | First-publication lag | Evidence | Citation |
|---|---|---|---|---|---|---|---|---|
```

Write one row per series, in the order below.

Cell rules:
- **Series.** The slug in a code span.
- **Vintage coverage.** Whether values as first published survive, and where.
- **Earliest date.** The first observation from the access route. Where Step 3 measured months present by decade, give them.
- **Access route.** A link.
- **First-publication lag.** The event that first makes a value public and the typical delay, for example "with the first release after the reference month, about five weeks".
- **Evidence.** One label.
- **Citation.** Primary links, each followed by `accessed YYYY-MM-DD`, or exactly `*unverified*`.

Write `\|` for a literal pipe and `\$` for a dollar sign. Never leave a cell empty or `n/a`; state the fact instead, for example "none: hand-built from the release schedule".

| Series | What the row must establish | Verify against |
|---|---|---|
| `collection_business_days` | Federal business days from the first business day after the 12th through first closing, hand-built per release. Give the first-closing rule and its source; the earliest year of archived schedules; that the count is known once the year's schedule is posted, with when that happens. | Handbook data chapter; Employment Situation schedule and archived schedules; OPM federal holidays |
| `federal_holiday` | Holiday dates that fall in each release's collection window. Give the years OPM's tables cover and the statutory basis. The count is known in advance. | OPM federal holidays; 5 U.S.C. 6103 at uscode.house.gov |
| `nonstandard_release` | Releases off their scheduled date or covering two months, from `es-vintages.csv` checked against archived schedules and lapse pages. Say that the vintage files' comments cover nonstandard *revisions*, a different flag. Known at release. | archived schedules; 2013 shutdown FAQ; 2025 lapse page; `cesvininfo.htm` |
| `appropriations_lapse` | Each lapse since 2003 with its dates and whether BLS was affected. Known when it happens. | the three BLS lapse pages; DOL contingency plan |
| `CEU00000000C1` | First closing collection rate, total nonfarm. Earliest 1981-01, with months present by decade from Step 3. The LABSTAT file holds current values; say whether earlier-published values survive (Step 4). Lag: with the first release after the reference month. Say when the series became public in the database, and whether release texts carried it earlier. | `ce.series`; `ce.datatype`; `ce.data.0.AllCESSeries`; the 2024 notice; the April 2019 release |
| `CEU00000000C2` | As C1, beginning 1981-03, published with the second release. | as C1 |
| `CEU00000000C3` | As C1, beginning 1981-01, published with the third release. | as C1 |
| `CEU05000000RR` | Third closing response rate, total private, beginning 2009-04, published with the third release. Not interchangeable with C3; say how the definitions differ. | `ce.series`; OSMR response rates; the 2024 notice |
| `four_five_week_interval` | The regressor files that carry the four/five-week treatment, and the first year of data in them. Past years survive only where the archive inventory keeps specification files. Known in advance; which series use it changes at the benchmark release. | `ces.spec.other.zip` readme; technical notes table 1; model specification tables |
| `easter_labor_day` | The same, for the Easter and Labor Day regressor files. | as above |
| `outlier_flags` | The recent-outliers file (monthly) and the AO, LS, and TC regressors in `.spc` files. Past coverage per the archive inventory. Lag: with each release for recent outliers, and with the benchmark release for specification outliers. | `cesseasadj.htm`; technical notes; `ces.spec.ae.zip` |
| `specification_regime` | Adjustment mode and calendar treatments per series per benchmark year. Annual. Tables for 2003–2013 are in the benchmark articles and the current year's is on `cesseasadj.htm`; other years only where specification files survive. Lag: with the benchmark release. | model specification tables; `cesbmkarch.htm` |
| `covid_interventions` | The 2020–2021 AO, LS, and TC treatments and when BLS adopted them. Annual, from 2020. Lag: with the benchmark release. | Hudson, Mercurio, and Kropf (2022), Monthly Labor Review; `.spc` files |

- [ ] **Step 7: Write the collection-versus-response ruling**

Under `### Collection rate versus response rate`, replace the status line with the text below. Fill each `from Step N` element with the observed value, and set each access date to the day you fetched the page.

```markdown
**Drafts.** The chatgpt draft reports 2024 average collection rates of 60.4, 89.0, and 90.9 percent at the three closings, treats the third-release response rate as a separate concept that fell from about 62 percent in fiscal 2015 to 42 percent in fiscal 2025, and dates the monthly C1–C3 history to January 2000 (chatgpt §2.2; chatgpt §3.3). The claude draft gives the same 2024 collection rates, a response rate falling from about 58 percent before 2020 to 43 percent in 2024 from `CEU05000000RR`, and collection-rate history from about 2000 (claude §2, Driver 2; claude §3). The gemini draft reports a secular decline in first-closing response rates from above 70 percent in 1999 to about 40 percent after the pandemic (gemini §2, Collection Interval).

**Primary sources.** Each accessed on the date shown.
- [LABSTAT `ce.series`](https://download.bls.gov/pub/time.series/ce/ce.series) and [`ce.datatype`](https://download.bls.gov/pub/time.series/ce/ce.datatype), accessed YYYY-MM-DD: series titles, data-type names, and begin and end periods.
- [LABSTAT `ce.data.0.AllCESSeries`](https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries), accessed YYYY-MM-DD: the 2024 means and yearly means from Step 3.
- [BLS notice adding collection rates to the database](https://www.bls.gov/ces/notices/2024/ces-collection-rates-have-been-added-to-the-online-public-database.htm), accessed YYYY-MM-DD: the definitions from Step 4.
- [BLS survey response rates](https://www.bls.gov/osmr/response-rates/), accessed YYYY-MM-DD: the response-rate definition from Step 4.
- [GAO-26-107538](https://files.gao.gov/reports/GAO-26-107538/index.html), accessed YYYY-MM-DD: the fiscal-year response rates from Step 4.

**Ruling.** Write the ruling from the Step 3 and Step 4 evidence:
- State both definitions as BLS gives them.
- Give the 2024 means of C1, C2, C3, and RR, and the yearly means of C1 and RR since 2020.
- If RR's post-2020 yearly means sit near 40 percent while C1's sit near 60 percent, rule that the gemini figure describes the third closing response rate, not a first-closing rate, and that the chatgpt and claude drafts keep the concepts apart. If the data contradict any draft figure, give the figure the data support.
- Rule that C1 and C3 begin in January 1981 and C2 in March 1981, not January 2000, with the months present by decade.

**Evidence:** documented.

**Consequence.** Roadmap Stage 12 builds `CEU00000000C1`, `CEU00000000C2`, `CEU00000000C3`, and `CEU05000000RR` as four distinct series, from the start dates and first-publication lags in the Collection window table. Req 3's January 2000 start is superseded by the LABSTAT dates.
```

The `**Ruling.**` paragraph in the document is your written ruling, not the bullet list above, which states what the ruling must cover. If the definitions or start dates cannot be read from BLS files, use `**Evidence:** supported` or `not found`, whichever fits, and say why.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS, 84 passed, 4 deselected.

- [ ] **Step 9: Commit**

```bash
uv run ruff format tests docs/ces-revisions-review.md
uv run ruff check tests
git add tests/test_review_document.py docs/ces-revisions-review.md
git commit -m "Inventory the collection-window and seasonal series and rule on collection versus response"
```

### Task 6: Birth–death, sample, and macro inventory, and the rulings on 2025 program cuts and the 2024–2025 benchmarks

**Files:**
- Modify: `tests/test_review_document.py` (three tests)
- Modify: `docs/ces-revisions-review.md` (the Net birth–death, Sample, and Macro and tail controls tables; two rulings)

**Interfaces:**
- Consumes (Task 5): `check_inventory_group`, `check_ruling`, and the table and ruling formats.
- Produces: the three tables and two rulings. Stage 4 reads the benchmark ruling and the birth–death and sample rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_review_document.py`:

```python
def test_birth_death_sample_and_macro_inventory():
    check_inventory_group("Net birth–death")
    check_inventory_group("Sample")
    check_inventory_group("Macro and tail controls")


def test_ruling_on_the_2025_program_cuts():
    check_ruling("Whether the 2025 program cuts reached CES")


def test_ruling_on_the_2024_and_2025_benchmark_revisions():
    check_ruling("The 2024 and 2025 preliminary and final benchmark revisions")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q -k "birth_death or program_cuts or benchmark_revisions"`
Expected: FAIL.
- The inventory test fails with `Net birth–death: expected one table, found 0`.
- Both ruling tests fail because their only paragraph is `Status:`.

- [ ] **Step 3: Fetch the birth–death and sample sources**

With `BLS_CONTACT_EMAIL` exported:

```bash
UA="ces-revisions/0.1.0 ($BLS_CONTACT_EMAIL)"
page() { uv run python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); print(re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",t)))' "$1"; }
fetch_to() { curl -s -L -A "$UA" "$1" -o "$2"; echo "$(wc -c < "$2") bytes  $1"; }
fetch_to https://www.bls.gov/web/empsit/cesbd.htm /tmp/b-bd.htm
fetch_to https://www.bls.gov/web/empsit/cesbdhst.htm /tmp/b-bdhst.htm
fetch_to https://www.bls.gov/web/empsit/cesbdqa.htm /tmp/b-bdqa.htm
fetch_to https://www.bls.gov/web/empsit/cestn.htm /tmp/b-cestn.htm
fetch_to https://www.bls.gov/ces/publications/benchmark/cesbmart25-tables.htm /tmp/b-bmart25-tables.htm
for year in 2003 2013 2024 2025; do
  fetch_to "https://www.bls.gov/ces/publications/benchmark/ces-benchmark-revision-$year.pdf" "/tmp/b-bmk-$year.pdf"
  pdftotext -layout "/tmp/b-bmk-$year.pdf" "/tmp/b-bmk-$year.txt"
  echo "== $year"; head -15 "/tmp/b-bmk-$year.txt"
  grep -n -i -E '^ *table [0-9]+' "/tmp/b-bmk-$year.txt" | head -20
done
uv run python - <<'EOF'
import re
from pathlib import Path

html = Path("/tmp/b-bdhst.htm").read_text(errors="replace")
captions = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", c)).strip() for c in re.findall(r"(?s)<caption[^>]*>(.*?)</caption>", html)]
print(len(captions), captions[:3], captions[-3:])
html = Path("/tmp/b-cestn.htm").read_text(errors="replace")
print([re.sub(r"\s+", " ", re.sub("<[^>]+>", "", c)).strip() for c in re.findall(r"(?s)<caption[^>]*>(.*?)</caption>", html)])
EOF
page /tmp/b-bdqa.htm | grep -o -i '.\{0,250\}\(quarterly\|current sample\|2026\|forecast\).\{0,250\}' | head -10
```

Establish, for the Net birth–death and Sample rows:
- **Forecast history on `cesbdhst.htm`.** The span of its forecast tables, whether they show forecasts as first published, and what the April–December tables that begin in 2019 contain.
- **Forecast timing.** When `cesbdqa.htm` says a forecast is first published: the quarterly update and the 2025 and 2026 method changes.
- **Forecast-versus-realized tables.** In each benchmark article: the table title, its unit, and the first benchmark year it appears. First read each article's opening lines to confirm which March benchmark it covers.
- **Linked coverage.** Which `cestn.htm` and benchmark-article tables give usable linked coverage and relative standard errors by supersector, their first year, and each change in the unit counted (organizations, UI accounts, worksites).

- [ ] **Step 4: Fetch the macro and tail-control sources**

```bash
fetch_to https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions /tmp/m-nber.htm
fetch_to https://oui.doleta.gov/unemploy/claims.asp /tmp/m-claims.htm
fetch_to "https://alfred.stlouisfed.org/series?seid=ICSA" /tmp/m-alfred-icsa.htm
fetch_to https://www.bls.gov/ces/publications/strike-report.htm /tmp/m-strike-report.htm
fetch_to https://www.bls.gov/wsp/ /tmp/m-wsp.htm
fetch_to https://www.ncei.noaa.gov/stormevents/ /tmp/m-storm-events.htm
fetch_to "$(awk -F, '$1=="2024-01"{print $3}' docs/inventory/es-vintages.csv)" /tmp/m-empsit-2024-01.htm
page /tmp/m-nber.htm | grep -o -i '.\{0,200\}announcement.\{0,200\}' | head -4
page /tmp/m-empsit-2024-01.htm | grep -o -i '.\{0,250\}weather.\{0,250\}' | head -3
```

Establish, for the Macro and tail controls rows:
- **Recession dates.** NBER announcement dates, which are when each turning point became public.
- **Initial claims.** Their release day, revision in the following week, and earliest date, plus whether the ALFRED page resolves as a vintage archive of `ICSA`.
- **Strikes.** What the CES strike report lists and how it relates to major work stoppages.
- **Storm events.** What Storm Events says about its update lag.
- **Weather in the release.** Whether the January 2024 release text states a weather effect.

A source that does not resolve is cited as *unverified*, and its row explains what was tried.

- [ ] **Step 5: Fetch the 2025 program-cut sources**

```bash
fetch_to https://www.bls.gov/ces/notices/ /tmp/p-ces-notices.htm
fetch_to https://www.bls.gov/sae/ /tmp/p-sae.htm
fetch_to https://www.bls.gov/cpi/notices/ /tmp/p-cpi-notices.htm
fetch_to https://www.bls.gov/ppi/notices/ /tmp/p-ppi-notices.htm
fetch_to https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf /tmp/p-cbj-2027.pdf && pdftotext -layout /tmp/p-cbj-2027.pdf /tmp/p-cbj-2027.txt
grep -o 'href="[^"]*notices[^"]*20\(25\|26\)[^"]*"' /tmp/p-ces-notices.htm /tmp/p-cpi-notices.htm /tmp/p-ppi-notices.htm | sort -u
grep -o 'href="[^"]*notices[^"]*"' /tmp/p-sae.htm | sort -u | head -5
grep -n -E '24,511|22,049' /tmp/p-cbj-2027.txt
grep -n -i -E 'sample (size|reduc)|discontinu|suspend' /tmp/p-cbj-2027.txt | head -20
grep -n -i -E 'discontinu|sample' /tmp/b-bmk-2025.txt | head -20
```

Open every 2025 and 2026 notice the `grep` lists for CES, for State and Area CES (from the SAE notices index linked on `/sae/`), for CPI, and for PPI. For each, record its date, what it reduces, suspends, or discontinues, and the reason BLS gives.

- [ ] **Step 6: Fetch the benchmark sources**

```bash
uv run python - <<'EOF'
import re
from pathlib import Path

html = Path("/tmp/b-cestn.htm").read_text(errors="replace")
start = html.find("Table 5.")
segment = html[start : start + 80000]
text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", segment))
for year in ("2023", "2024", "2025"):
    match = re.search(rf"{year}\s*\(?\d*\)?\s*(-?[\d.]+\s+-?[\d,]+\s+-?[\d.]+\s+-?[\d,]+\s+-?[\d.]+\s+-?[\d,]+)", text)
    print(year, match.group(1) if match else "not found")
for note in ("(11)", "(12)"):
    at = text.find(note, text.find("2025"))
    print(note, text[at : at + 400])
EOF
fetch_to https://www.bls.gov/news.release/prebmk.nr0.htm /tmp/p-prebmk.htm
page /tmp/p-prebmk.htm | head -c 1500; echo
grep -n -E '862,000|861,000|898,000|598,000|589,000' /tmp/b-bmk-2024.txt /tmp/b-bmk-2025.txt
```

Expected: rows for 2023, 2024, and 2025 matching Planning evidence, then the text of footnotes 11 and 12. If the extraction misreads the table, read Table 5 in a browser.

Establish:
- **Figures.** Each 2024 and 2025 figure (preliminary, final NSA, final SA) with its source.
- **Footnote 12.** What it says about the 2025 final figure.
- **The −862 figure.** Whether the article states −862 thousand, and why it differs from −861.
- **The current preliminary release.** The March 2026 preliminary figure and its release date.

- [ ] **Step 7: Write the three tables**

Write one table under each of `### Net birth–death`, `### Sample`, and `### Macro and tail controls`. Use the header, separator, and cell rules from Task 5 Step 6, with one row per series in this order:

| Series | What the row must establish | Verify against |
|---|---|---|
| `birth_death_forecast` | Monthly net birth–death forecasts by supersector: the span on `cesbdhst.htm`; whether they are as first published; the April–December tables since 2019. Zero for government. Lag from `cesbdqa.htm`. | `cesbd.htm`; `cesbdhst.htm`; `cesbdqa.htm` |
| `birth_death_forecast_vs_realized` | The benchmark-article table of forecast against realized net birth–death: title, unit, first benchmark year. Annual; published with the benchmark article. | benchmark articles |
| `linked_coverage` | Usable linked coverage by supersector per March benchmark: current table, earlier years, first year, and each definition switch with its years. Annual; lag to the benchmark article. | `cestn.htm` Table 1; benchmark articles and table pages |
| `relative_standard_error` | RSE of the one-month change by supersector: where the current and earlier years are published. Annual; lag. | `cestn.htm` Table 4; benchmark articles |
| `recession_dates` | NBER peaks and troughs; the lag is each announcement date. | NBER business-cycle dates |
| `ui_claims` | Weekly initial claims: release timing, revision in the following week, earliest date, vintage archive. | DOL weekly claims; ALFRED `ICSA` |
| `strikes` | Strikes affecting CES counts, and how the CES strike report relates to major work stoppages; lag. | CES strike report; BLS work stoppages |
| `severe_weather` | Storm Events coverage and update lag; weather statements in releases, citing one if found. | NOAA Storm Events; the January 2024 release |

- [ ] **Step 8: Write the ruling on 2025 program cuts**

Under `### Whether the 2025 program cuts reached CES`, replace the status line with the text below. Fill each `from Step N` element with the observed value, link each source that a bullet names without a link, and set each access date to the day you fetched the page. Keep the benchmark-article bullet only if Step 3 confirmed that the article covers the March 2025 benchmark.

```markdown
**Drafts.** The claude draft lists 2025 reductions in CPI collection (Lincoln, Nebraska, and Provo, Utah, in April; Buffalo, New York, in June), discontinued PPI indexes, and suspended restricted-use datasets, and finds no public documentation of a CES sample reduction or suspended CES series beyond the routine February 2025 sample review (claude §2, Driver 3; claude §6). The chatgpt draft finds no documented budget-driven reduction in the national CES probability sample, while noting that published national, state, and area CES estimates fell from 24,511 in fiscal 2022 to 22,049 in fiscal 2025 (chatgpt §2.3). The gemini draft does not address 2025; it says only that the 2013 and 2018–19 shutdowns forced collection delays and suspended series (gemini §2, BLS Funding).

**Primary sources.** Each accessed on the date shown.
- [CES notices](https://www.bls.gov/ces/notices/), accessed YYYY-MM-DD: every 2025 and 2026 notice, from Step 5.
- The State and Area CES, CPI, and PPI notices from Step 5, each linked and dated.
- [DOL FY2027 Congressional Budget Justification, BLS volume](https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf), accessed YYYY-MM-DD: the published-estimate counts and any stated program reductions, with page numbers.
- [CES benchmark article covering the March 2025 benchmark](https://www.bls.gov/ces/publications/benchmark/ces-benchmark-revision-2025.pdf), accessed YYYY-MM-DD: the annual review of published series.

**Ruling.** Write the ruling from the Step 5 notices:
- Rule separately on national CES and on State and Area CES.
- A notice that reduces the sample, suspends collection, or discontinues series counts as a cut. The routine review of published series at the benchmark counts only if BLS ties it to resources.
- Confirm or correct the CPI and PPI reductions the claude draft lists, citing their notices.
- Say whether the CBJ carries the 24,511 and 22,049 counts, with the page, or that they were not found.

**Evidence:** documented.

**Consequence.** Roadmap Stages 13 and 14 code a 2025 CES program change only from a notice this ruling cites. A fall in the number of published estimates is a definition note for the sample rows of Req 3, not a scale covariate.
```

As in Task 5, the document's `**Ruling.**` paragraph is prose that covers the bullets above.

Choose the label on the `**Evidence:**` line:
- **`documented`** when a notice announces a CES reduction, or when BLS states that CES was unaffected.
- **`not found`** when no notice addresses CES at all. State in the ruling that no BLS notice documents a 2025 CES reduction.

- [ ] **Step 9: Write the benchmark ruling**

Under `### The 2024 and 2025 preliminary and final benchmark revisions`, replace the status line with the text below. Fill each `from Step N` element with the observed value, link each source that a bullet names without a link, and set each access date to the day you fetched the page.

```markdown
**Drafts.** The claude draft gives the March 2024 benchmark as −818,000 preliminary and −598,000 final, and the March 2025 benchmark as −911,000 preliminary with a final of −862,000 not seasonally adjusted, −861,000 after a data reconstruction, and −898,000 seasonally adjusted, warning that −598,000 belongs to 2024 (claude §2; claude §6). The chatgpt draft gives −598,000 (−0.4 percent) for 2024 and, for 2025, −861,000 after a scope adjustment beside a −862,000 headline not-seasonally-adjusted difference, pairing them with preliminary figures of −818,000 and −911,000 (chatgpt §2.6). The gemini draft quotes no benchmark figures (gemini §3).

**Primary sources.** Each accessed on the date shown.
- [CES technical notes, Table 5](https://www.bls.gov/web/empsit/cestn.htm), accessed YYYY-MM-DD: final and preliminary total nonfarm benchmark revisions, with footnotes 11 and 12.
- The benchmark articles covering the March 2024 and March 2025 benchmarks, from Step 3, each linked with its access date: the not-seasonally-adjusted and seasonally adjusted level revisions.
- [Current preliminary benchmark release](https://www.bls.gov/news.release/prebmk.nr0.htm), accessed YYYY-MM-DD: the March 2026 preliminary figure, from Step 6.

**Ruling.** Write the ruling from the Step 6 evidence:
- Give each figure with its source and its NSA or SA basis: the 2024 preliminary and final, the 2025 preliminary, the 2025 final NSA as Table 5 and the article state it, and the 2025 final SA.
- Explain the −862 versus −861 difference from footnote 12 and the article, in your own words.
- Say which draft figures match the sources, and that −598 is the 2024 final.
- Give the March 2026 preliminary figure and its release date.

**Evidence:** documented.

**Consequence.** Roadmap Stage 4's benchmark table carries Table 5's final and preliminary figures with their footnotes and publication dates. The reconstruction that footnote 12 describes is an event in Stage 4's reconstruction-events table.
```

If Table 5 or the articles disagree with Planning evidence, stop and report.

- [ ] **Step 10: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS, 87 passed, 4 deselected.

- [ ] **Step 11: Commit**

```bash
uv run ruff format tests docs/ces-revisions-review.md
uv run ruff check tests
git add tests/test_review_document.py docs/ces-revisions-review.md
git commit -m "Inventory the birth-death, sample, and macro series and rule on 2025 cuts and benchmarks"
```

### Task 7: Institutional and party inventory, and the rulings on the 2018–19 lapse and FTE concepts

**Files:**
- Modify: `tests/test_review_document.py` (three tests)
- Modify: `docs/ces-revisions-review.md` (the Institutional and Party composition tables, two rulings, and the inventory's status line)

**Interfaces:**
- Consumes:
  - From Task 5: `check_inventory_group`, `check_ruling`, and the table and ruling formats.
  - From Task 3: `docs/inventory/es-vintages.csv`, for release URLs.
- Produces: the last two tables and two rulings. Stage 13 reads the shutdown and FTE rulings. Stage 27 reads the FTE ruling.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_review_document.py`:

```python
def test_institutional_and_party_inventory():
    check_inventory_group("Institutional")
    check_inventory_group("Party composition")


def test_ruling_on_the_2018_to_2019_lapse():
    check_ruling("The December 2018 to January 2019 lapse in appropriations")


def test_ruling_on_fte_concepts():
    check_ruling("FTE concepts: authorized versus actual, and FTE versus headcount")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q -k "institutional or 2018 or fte"`
Expected: FAIL.
- The inventory test fails with `Institutional: expected one table, found 0`.
- Both ruling tests fail because their only paragraph is `Status:`.

- [ ] **Step 3: Fetch the budget, FTE, and headcount sources**

With `BLS_CONTACT_EMAIL` exported:

```bash
UA="ces-revisions/0.1.0 ($BLS_CONTACT_EMAIL)"
page() { uv run python -c 'import re,sys; t=open(sys.argv[1],errors="replace").read(); print(re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",t)))' "$1"; }
fetch_to() { curl -s -L -A "$UA" "$1" -o "$2"; echo "$(wc -c < "$2") bytes  $1"; }
fetch_to https://www.dol.gov/general/budget /tmp/i-dol-budget.htm
grep -o -i 'href="[^"]*\(cbj\|budget/20[0-9][0-9]\)[^"]*"' /tmp/i-dol-budget.htm | sort -u | head -40
[ -f /tmp/p-cbj-2027.txt ] || { fetch_to https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf /tmp/p-cbj-2027.pdf && pdftotext -layout /tmp/p-cbj-2027.pdf /tmp/p-cbj-2027.txt; }
grep -n -i -E 'full-time equivalent|FTE' /tmp/p-cbj-2027.txt | head -30
grep -n -E '1,773|2,058|2,019' /tmp/p-cbj-2027.txt | head -20
grep -n -i 'unemployment trust fund' /tmp/p-cbj-2027.txt | head -10
fetch_to https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf /tmp/i-transition.pdf && pdftotext -layout /tmp/i-transition.pdf /tmp/i-transition.txt
grep -n -i -E 'FTE|full-time' /tmp/i-transition.txt | head -20
fetch_to https://www.whitehouse.gov/omb/information-resources/guidance/circulars/ /tmp/i-omb-circulars.htm
grep -o -i 'href="[^"]*a11[^"]*"' /tmp/i-omb-circulars.htm | sort -u | head
curl -s -L -A "$UA" -o /tmp/i-fedscope.htm -w 'fedscope HTTP %{http_code}\n' https://www.fedscope.opm.gov/
fetch_to https://www.bls.gov/osmr/research-papers/2016/st160050.htm /tmp/i-robertson-2016.htm
page /tmp/i-robertson-2016.htm | grep -o -i '.\{0,250\}workyear.\{0,250\}' | head -4
```

From the DOL budget page, open the BLS volume of the FY2026 CBJ and of the earliest CBJ that reports FY2009 actuals. Download each with `fetch_to` and convert it with `pdftotext -layout`. From the OMB circulars page, open the current Circular A-11 and find the definition of FTE employment in section 85. If fedscope.opm.gov does not serve data, find OPM's current workforce-data site from `https://www.opm.gov/` and record the route you used.

Establish:
- **Budget authority.** Which CBJ table and column gives BLS budget authority, and how the Unemployment Trust Fund transfer appears.
- **FTE columns.** Which CBJ columns report actual FTE and which report plans (enacted, estimate, request), from their headings.
- **The FTE definition.** Section 85's definition of FTE.
- **Headcount.** OPM's BLS on-board headcount route, and its snapshot dates.
- **Workyears.** The CES workyears that Robertson (2016) reports, and whether any CBJ reports them.

- [ ] **Step 4: Fetch the appropriations, shutdown, and party sources**

```bash
for month in 2013-09 2013-10 2018-12 2019-01 2025-09 2025-11 2025-12 2026-01; do
  url="$(awk -F, -v m="$month" '$1==m{print $3}' docs/inventory/es-vintages.csv)"
  fetch_to "$url" "/tmp/i-empsit-$month.htm"
  page "/tmp/i-empsit-$month.htm" | grep -o -i '.\{0,250\}\(lapse\|shutdown\|furlough\|collection period\).\{0,250\}' | head -4
done
fetch_to https://www.bls.gov/bls/archived_sched.htm /tmp/i-archived-sched.htm
grep -o -i 'href="[^"]*\(2013\|2019\|2025\|2026\)[^"]*"' /tmp/i-archived-sched.htm | sort -u | head
fetch_to https://www.bls.gov/bls/what-impact-did-the-lapse-of-appropriation-for-some-federal-agencies-have-on-january-employment-data.htm /tmp/i-lapse-2019.htm
fetch_to https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm /tmp/i-lapse-2025.htm
fetch_to https://www.bls.gov/bls/shutdown_2013_empsit_qa.pdf /tmp/i-shutdown-2013.pdf && pdftotext /tmp/i-shutdown-2013.pdf /tmp/i-shutdown-2013.txt
fetch_to https://www.dol.gov/sites/dolgov/files/general/plans/dol-contingency-plan.pdf /tmp/i-dol-contingency.pdf && pdftotext -layout /tmp/i-dol-contingency.pdf /tmp/i-dol-contingency.txt
fetch_to https://www.govinfo.gov/app/details/PLAW-115publ245 /tmp/i-plaw-115-245.htm
fetch_to https://history.house.gov/Institution/Party-Divisions/Party-Divisions/ /tmp/i-house-parties.htm
fetch_to https://www.senate.gov/history/partydiv.htm /tmp/i-senate-parties.htm
page /tmp/i-lapse-2019.htm | head -c 3000; echo
grep -n -i -E 'bureau of labor statistics|BLS' /tmp/i-dol-contingency.txt | head -10
page /tmp/i-plaw-115-245.htm | grep -o -i '.\{0,200\}\(approved\|enacted\|date\).\{0,120\}' | head -3
```

Read the continuing-resolution history in CRS R46595 at `https://www.congress.gov/crs_external_products/R/PDF/R46595/R46595.7.pdf` in a browser, because congress.gov refuses non-browser clients. Cite the enacted statutes on govinfo.gov for the counts it gives.

Establish:
- **Lapses.** Every lapse in appropriations since 2003 that BLS or DOL documents, with its dates.
- **The four shutdown fields.** For each lapse: whether BLS closed, whether CES collection stopped, the scheduled and actual release dates, and whether a release says the collection window was extended.
- **FY2019 appropriations.** The enactment date of the FY2019 Labor appropriations, and what BLS says about the December 2018 to January 2019 lapse.
- **Continuing resolutions.** Days under CRs, counts of CRs, and full-year status for each fiscal year from FY2009.
- **Party control.** House and Senate majorities for each Congress since 2003, including any mid-Congress Senate change.

- [ ] **Step 5: Write the two tables**

Write one table under `### Institutional` and one under `### Party composition`. Use the header, separator, and cell rules from Task 5 Step 6, with one row per series in this order:

| Series | What the row must establish | Verify against |
|---|---|---|
| `budget_authority` | BLS budget authority by fiscal year from FY2009, General Fund plus the Unemployment Trust Fund transfer: the table and column, and how continuing resolutions, sequestration, and rescissions appear. Nominal. Lag: the enacted amount is known at enactment, and actuals first appear in a later CBJ. | CBJ BLS volumes; DOL BLS transition brief |
| `budget_deflator` | The deflator candidates, such as BEA's GDP implicit price deflator: frequency, revisions, vintage archive. Stage 13 makes the choice. | BEA NIPA tables; ALFRED |
| `fte` | BLS FTE by fiscal year per the CBJs, with Ruling 4's concept: which column is actual and which a plan. Lag: when a year's actual FTE first appears. | CBJs; OMB Circular A-11 section 85 |
| `opm_headcount` | BLS on-board headcount from OPM: snapshot dates, earliest date, access route. Not FTE. | OPM workforce data |
| `ces_workyears` | Federal, state, and contractor workyears for CES, for every year a source reports them, and whether any CBJ reports them. | Robertson (2016), BLS OSMR; CBJs |
| `cr_days` | Days of each fiscal year under continuing resolutions, from FY2009, and when the count becomes final. | CRS R46595 (read in a browser); govinfo.gov statutes |
| `cr_extensions` | The number of continuing resolutions and extensions per fiscal year. | as above |
| `full_year_cr` | Whether BLS's fiscal-year appropriation was a full-year continuing resolution, citing the statute. | as above |
| `shutdown_bls_closed` | For each documented lapse since 2003, whether BLS itself closed. | BLS lapse pages; DOL contingency plan |
| `shutdown_collection_stopped` | Whether CES collection stopped during each lapse, and for how long. | BLS lapse pages; releases after each lapse |
| `shutdown_release_delayed` | Whether each lapse moved an Employment Situation release, with its scheduled and actual dates. | archived schedules; `es-vintages.csv`; BLS lapse pages |
| `shutdown_window_extended` | Whether BLS extended the collection window during a lapse, and the release that says so. | releases after each lapse |
| `house_majority` | The House majority party for each Congress since 2003, with dates. Descriptive only (Req 3). Known when each Congress convenes. | House history, party divisions |
| `senate_majority` | The Senate majority party for each Congress since 2003, including mid-Congress changes. Descriptive only. | Senate history, party division |

In the Institutional table, the Hand-build constraint cell of each `shutdown_*` row lists the lapses whose field no BLS or DOL page documents.

- [ ] **Step 6: Write the ruling on the 2018–19 lapse**

Under `### The December 2018 to January 2019 lapse in appropriations`, replace the status line with the text below. Fill each `from Step N` element with the observed value, link each source that a bullet names without a link, and set each access date to the day you fetched the page.

```markdown
**Drafts.** The gemini draft says the 2013 and 2018–2019 shutdowns forced collection delays and suspended series (gemini §2, BLS Funding). The chatgpt draft says the Department of Labor already had full-year appropriations during the December 2018 to January 2019 partial lapse, so BLS did not shut down, while other agencies' furloughs entered CES as real employment changes (chatgpt §2.3). The claude draft does not discuss that lapse; its funding discussion covers the FY2013 sequestration, the 2025 program cuts, and the 2025 shutdown (claude §1; claude §2, Driver 3).

**Primary sources.** Each accessed on the date shown.
- [BLS on the lapse's effect on January 2019 employment data](https://www.bls.gov/bls/what-impact-did-the-lapse-of-appropriation-for-some-federal-agencies-have-on-january-employment-data.htm), accessed YYYY-MM-DD.
- The January 2019 Employment Situation release, linked from `es-vintages.csv`, accessed YYYY-MM-DD: its release date and what it says about the lapse.
- The 2019 release schedule from [BLS's archived schedules](https://www.bls.gov/bls/archived_sched.htm), accessed YYYY-MM-DD: the scheduled date.
- The FY2019 Labor appropriations act on govinfo.gov, accessed YYYY-MM-DD: its enactment date.

**Ruling.** Write the ruling from the Step 4 evidence:
- State whether the Department of Labor, and so BLS, was funded throughout the lapse.
- Compare the January 2019 release's actual date with its scheduled date.
- Code the four Stage 13 fields for this lapse (BLS closed, collection stopped, release delayed, window extended), each yes or no with its source.
- Rule on the gemini draft's claim for 2018–19, and say what the release says about furloughed federal workers, if anything.

**Evidence:** documented.

**Consequence.** Roadmap Stage 13's shutdown-events table codes this lapse with the four values above, and no generic federal-shutdown indicator stands in for them.
```

- [ ] **Step 7: Write the FTE ruling**

Under `### FTE concepts: authorized versus actual, and FTE versus headcount`, replace the status line with the text below. Fill each `from Step N` element with the observed value, link each source that a bullet names without a link, and set each access date to the day you fetched the page.

```markdown
**Drafts.** The claude draft says FY2026 enacted budget authority of about \$708.5 million came with on-board FTE of 1,773, down about 14 percent from FY2024's 2,058, and cites the American Statistical Association's estimate of a 22 percent staffing loss since FY2024. Its caveats add that FTE figures mix authorized and actual concepts across budget justifications and that the ASA figure reflects on-board staffing (claude §1; claude §2, Driver 3; claude §6). The chatgpt draft calls its FTE series the budget justifications' fiscal-year-average concept rather than OPM headcount, marks FY2026's 1,773 as planned, and separately reports OPM on-board counts of 2,330 in FY2023, 2,321 in FY2024, 2,165 in FY2025, and 1,846 by July 1, 2026 (chatgpt §2.3; chatgpt §2.5). The gemini draft says OPM FedScope tracks statistical-agency FTEs and lists FedScope FTEs as quarterly (gemini §2, Staffing; gemini §3).

**Primary sources.** Each accessed on the date shown.
- OMB Circular A-11, section 85, from the [OMB circulars page](https://www.whitehouse.gov/omb/information-resources/guidance/circulars/), accessed YYYY-MM-DD: the definition of FTE employment.
- [DOL FY2027 Congressional Budget Justification, BLS volume](https://www.dol.gov/sites/dolgov/files/general/budget/2027/CBJ-2027-V3-01.pdf), accessed YYYY-MM-DD: column headings and FTE rows for FY2025 to FY2027, with page numbers.
- The FY2026 CBJ BLS volume from the [DOL budget page](https://www.dol.gov/general/budget), accessed YYYY-MM-DD: FY2024 actual FTE.
- OPM's workforce data for BLS, from Step 3, accessed YYYY-MM-DD: on-board counts and snapshot dates.
- [DOL BLS transition brief](https://www.dol.gov/sites/dolgov/files/general/foia/presidential-transition-docs/bls.pdf), accessed YYYY-MM-DD: its FTE table.

**Ruling.** Write the ruling from the Step 3 evidence:
- Define FTE in your own words, from section 85. Distinguish it from on-board headcount and from authorized positions or ceilings.
- For each CBJ column used, say whether it reports actual FTE or a plan, from its heading.
- Classify each draft figure: 2,058 for FY2024; 1,773 for FY2026; OPM's 2,330, 2,321, 2,165, and 1,846; and the ASA's 22 percent. The ASA figure is grey literature, so treat its claim as asserted and do not list it as a primary source.
- Rule on the claude draft's "on-board FTE" and the gemini draft's "FedScope FTEs".

**Evidence:** documented.

**Consequence.** Roadmap Stage 13's institutional panel keeps actual FTE, from the CBJ columns that report completed years, and OPM on-board headcount as separate series with a definition flag. Stage 16's capacity factor loads actual FTE, never a plan, and Stage 27 applies the same concepts to pre-2009 budget justifications.
```

- [ ] **Step 8: Close the inventory section**

In `## Data-availability inventory`, delete the line `**Status:** in progress — plan 2, Tasks 5 to 7.` and the blank line that follows it.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest -m "not slow and not network" -q`
Expected: PASS, 90 passed, 4 deselected.

- [ ] **Step 10: Commit**

```bash
uv run ruff format tests docs/ces-revisions-review.md
uv run ruff check tests
git add tests/test_review_document.py docs/ces-revisions-review.md
git commit -m "Inventory the institutional and party series and rule on the 2018-19 lapse and FTE concepts"
```

### Task 8: Summary of findings, project notes, and final verification

**Files:**
- Modify: `tests/test_review_document.py` (import `github_anchor`; four tests, one of them parametrized over four markers)
- Modify: `docs/ces-revisions-review.md` (the Summary of findings)
- Modify: `CLAUDE.md` (Purpose, Layout and tooling, Commands)

**Interfaces:**
- Consumes: every earlier section of the document, and the GitHub heading anchors of `## Seasonal-adjustment archive inventory`'s subsections, the inventory groups, and the rulings.
- Produces: the finished Stage 2 document and CLAUDE.md's notes on `scripts/` and `docs/inventory/`.

- [ ] **Step 1: Write the failing completion tests**

In `tests/test_review_document.py`, add `github_anchor` to the `from review_document import (...)` list, then append:

```python
# --- Task 8: summary and completion --------------------------------------------------------


def test_no_section_is_left_in_progress():
    assert "**Status:** in progress" not in read_review()


def test_every_summary_finding_links_its_evidence():
    body = section(read_review(), "Summary of findings", level=2)
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(bullets) >= 9
    for bullet in bullets:
        assert re.search(r"\]\((?:#[\w-]+|https?://[^)\s]+)\)", bullet), bullet


def test_internal_links_resolve_to_headings():
    text = read_review()
    anchors = {
        github_anchor(title)
        for level in range(1, 7)
        for title in headings(text, level=level)
    }
    targets = set(re.findall(r"\]\(#([\w-]+)\)", text))
    assert targets <= anchors, sorted(targets - anchors)


@pytest.mark.parametrize(
    "marker", ["YYYY-MM-DD", "PRECISION", "CHANNEL-B", r"\bSteps? \d"]
)
def test_no_parameterized_marker_survives(marker):
    text = read_review()
    found = [
        text[max(0, m.start() - 40) : m.end() + 40] for m in re.finditer(marker, text)
    ]
    assert not found, found[:3]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review_document.py -q -k "in_progress or summary_finding or internal_links or marker"`
Expected: FAIL. `test_no_section_is_left_in_progress` fails on the Summary's status line, and `test_every_summary_finding_links_its_evidence` fails on `assert 0 >= 9`. `test_internal_links_resolve_to_headings` passes, because the document has no internal links yet. `test_no_parameterized_marker_survives` passes unless an earlier task left an access-date placeholder, a `PRECISION` or `CHANNEL-B` marker, or a plan-step reference such as `from Step 3` in the document; fix any it reports before going on.

- [ ] **Step 3: Write the summary**

Under `## Summary of findings`, replace the status line with one bullet per finding below, in this order. Each bullet states the determination the linked section records, in one or two sentences, and ends with the link shown:

1. **Archived seasonal-adjustment files.** Give the number of releases that keep each file type, the earliest such release, and the number that keep all three, from the findings block. Link `[archive findings](#findings)`.
2. **Unrounded NSA inputs and published precision.** State the non-publication finding and the Step 12 precision result of Task 4. Link `[unrounded NSA inputs](#unrounded-nsa-inputs)`.
3. **Channel (b).** State the Stage 25 and 26 consequence chosen in Task 4 Step 12. Link `[consequences](#consequences-for-later-stages)`.
4. **Vintages.** Give the number of releases from May 2003, that October 2025 had no release of its own, and the date of the delayed September 2025 release. Link `[method](#method)`.
5. **Collection-rate history.** State that C1–C3 begin in 1981, not January 2000, and that RR begins in April 2009. Link `[collection window](#collection-window)`.
6. **Collection versus response.** Give the ruling. Link `[ruling](#collection-rate-versus-response-rate)`.
7. **The 2018–19 lapse.** Give the ruling and the four field values. Link `[ruling](#the-december-2018-to-january-2019-lapse-in-appropriations)`.
8. **2025 program cuts.** Give the ruling. Link `[ruling](#whether-the-2025-program-cuts-reached-ces)`.
9. **FTE concepts.** Give the ruling. Link `[ruling](#fte-concepts-authorized-versus-actual-and-fte-versus-headcount)`.
10. **2024 and 2025 benchmarks.** Give the figures and the −862 versus −861 explanation. Link `[ruling](#the-2024-and-2025-preliminary-and-final-benchmark-revisions)`.
11. **Unverified rows.** Count the inventory rows labeled `not found` or cited *unverified*, and name them. Link `[data inventory](#data-availability-inventory)`.

Write each bullet as `- **Label.** Determination ([link text](#anchor)).`

- [ ] **Step 4: Note the tooling in CLAUDE.md**

In `CLAUDE.md`, make three edits.

1. **Purpose.** Replace

   ```markdown
   and decision records in `docs/decisions/`.
   ```

   with

   ```markdown
   decision records in `docs/decisions/`, and the Req 20 written finding in `docs/ces-revisions-review.md`, with its evidence in `docs/inventory/`.
   ```

2. **Layout and tooling.** After the bullet that begins ``- `src/ces_revisions/kalman.py` is the hand-written``, add:

   ```markdown
   - `scripts/` holds research tools outside the package; `pythonpath = ["scripts"]` in `[tool.pytest.ini_options]` lets tests import them by bare name. `scripts/archive_inventory.py captures` (network) records every Employment Situation release and every BLS and Internet Archive copy of the seasonal-adjustment files in `docs/inventory/`, and `scripts/archive_inventory.py inventory` (offline) derives the archive inventory from that evidence and regenerates its tables in `docs/ces-revisions-review.md`. Never hand-edit those CSVs or the generated blocks.
   ```

3. **Commands.** After the line `uv run ruff format                              # format (--check to verify without writing)`, add:

   ```bash
   uv run python scripts/archive_inventory.py inventory   # rebuild the archive inventory from docs/inventory/ (offline)
   ```

- [ ] **Step 5: Run the complete verification**

```bash
uv run ruff format --check scripts tests docs CLAUDE.md
uv run ruff check
uv run pytest -m "not slow and not network" -q
uv run pytest -m network -q
cp docs/ces-revisions-review.md /tmp/review-before.md
cp docs/inventory/archive-inventory.csv /tmp/inventory-before.csv
uv run python scripts/archive_inventory.py inventory
diff /tmp/review-before.md docs/ces-revisions-review.md
diff /tmp/inventory-before.csv docs/inventory/archive-inventory.csv
```

Expected:
- Both ruff commands are clean.
- The fast tier passes with 97 passed and 4 deselected.
- The network tier passes with 2 passed.
- Regenerating the inventory changes neither the document nor `archive-inventory.csv`: both `diff` commands print nothing and exit 0. The copies are compared, not `git diff`, because Step 3's summary is not yet committed.

Then check the roadmap's Stage 2 exit clause by clause:
- the archive inventory lists every vintage from May 2003 with its three file types and unrounded-input status;
- the data inventory has one row per Req 3 series, with a first-publication-lag column;
- each of the five rulings has its own cited subsection;
- `uv run ruff format --check docs/ces-revisions-review.md` passes.

- [ ] **Step 6: Commit**

```bash
uv run ruff format tests docs/ces-revisions-review.md CLAUDE.md
uv run ruff check tests
git add tests/test_review_document.py docs/ces-revisions-review.md CLAUDE.md
git commit -m "Summarize the Stage 2 findings and document the inventory tooling"
```

---

## Completion notes

Run the Plan Completion Protocol after Task 8 and the whole-branch review. It includes these items specific to Stage 2:

- **Roadmap and spec.** Tick Stage 2 in `specs/ces-revisions-roadmap.md` and append the stamp in **Retirement** to the spec's Rollout note.
- **Completion report.** Include:
  - Deviation 1 (C1–C3 start in 1981 rather than January 2000), for the spec's Req 3 and Motivation;
  - the Task 4 Step 12 outcomes: the precision result for Req 15, and whether resume should park Stages 25 and 26;
  - Deviation 2's overlap between `es-vintages.csv` and Stage 3's release-date index.
- **Deferred items.** The `specs/deferred_items.md` ticking pass still runs, though this plan closes neither plan-1 item. Defer any row left *unverified* only if it can state a closure condition, for example "Done when: a primary source for `<series>` is cited, or Stage 13 records the series as unavailable".
