# Stage 4 — Benchmark, Birth–Death, QCEW Revision, and Sample Tables Implementation Plan

**Status: COMPLETE (2026-09-15)** — executed via executing-plans; nothing deferred

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/ces-revisions-roadmap.md, Stage 4 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build the annual-source data layer that later benchmark operators and scale covariates consume: dated preliminary and final CES benchmark revisions, sector benchmark contributions, documented reconstruction events, monthly and annual birth–death rows, the 2017+ national QCEW revision sequence with a March precision proxy, and the annual sample coverage/RSE panel.

**Architecture:** Add a focused `ces_revisions.annual` subpackage backed by a committed, content-hashed source archive under `data/annual/raw/`. Source cells remain printed text with stable string keys; derived tables carry those keys (or a namespaced Stage 3 release-index key), a named transformation, and an `observable_at` timestamp. Employment Situation publication dates are joined from Stage 3's release index, while preliminary benchmark announcements and QCEW releases use a Stage 4 publication catalog whose rows cite the release that printed each date. PDF tables are transcribed into auditable cell CSVs and checked by row identities; HTML and the QCEW CSV are parsed directly.

**Tech Stack:** Python 3.14; Polars 1.44.2; `python-dotenv>=1.0` for the gitignored project environment; standard-library HTML, CSV, hashing, URL, datetime, and zoneinfo tools; pytest 9.1.1; ruff 0.16.7; and `pdftotext` (poppler) in the network fetch only.

## Global Constraints

- **Scope:** Implement only roadmap Stage 4. The wedge, post-March recursion, benchmark likelihood, relative birth–death covariates, and scale model remain in Stages 5, 8, 14, and 16.
- **Python:** Keep `requires-python = ">=3.14"`; add only `python-dotenv>=1.0`, matching the configuration pattern in `/Users/lowell/Projects/bls-stats/src/bls_stats/core/config.py`.
- **HTTP identity:** Load `.project.env` with `python-dotenv` before network work, require `BLS_CONTACT_EMAIL`, and include that address in every Stage 4 HTTP `User-Agent`. Never print, persist, or commit the address.
- **BLS concepts:** Benchmark and birth–death quantities are not seasonally adjusted. Never add an NSA birth–death amount to an SA change.
- **Units:** CES levels and revisions are stored in thousands. QCEW source employment stays in jobs; only a named transformation may divide it by 1,000.
- **QCEW scope:** Ingest the public 2017+ national aggregate employment revision sequence only. Do not reconstruct the full historical QCEW vintage cube.
- **As-of correctness:** Every derived row carries the first date and UTC timestamp at which that exact value was observable. A later page snapshot never backdates a revised value.
- **Provenance:** Every non-null published number points to one or more stable raw-cell keys and one named transformation. Structural zeros and explicit missing rows name their rule and evidence.
- **Sample frequency:** Sample coverage and RSE remain annual. Never interpolate or repeat either across monthly rows.
- **Coverage meaning:** Label Table 1's employee share as a proxy for usable linked coverage because it counts active sample reports, not the matched sample.
- **RSE meaning:** Preserve the published measure, closing, and unit. Set `rse_definition_break` when the measure changes; do not harmonize unlike RSEs into a synthetic series.
- **Government:** Materialize a government birth–death row of exactly zero for every birth–death period and value kind.
- **Missing evidence:** Materialize March 2012 sample rows with null published values and an explicit archive-gap reason. Do not interpolate them.
- **2025 benchmark ruling:** Table 5's final total is −861 thousand. Store the article's level difference of −862 thousand beside it and label the one-thousand difference `unexplained`.
- **2025 program-cuts ruling:** Do not create a national-CES program-cut covariate. A falling count of published series is definition metadata only.
- **Network tests:** Every live fetch or live-layout canary carries `@pytest.mark.network`; the default tier uses committed sources and fixtures only.
- **Markdown:** Keep GitHub math/rendering conventions from `CLAUDE.md`; literal dollar amounts are escaped.

---

## Source and numbering

This plan implements [`specs/ces-revisions.md`](../../ces-revisions.md) Req 1's benchmark sources, Req 3's sample rows, Req 10's preliminary/final and QCEW inputs, and Req 11's data half. It implements Stage 4 of [`specs/ces-revisions-roadmap.md`](../../ces-revisions-roadmap.md), using the findings in [`docs/ces-revisions-review.md`](../../../docs/ces-revisions-review.md) and the coverage item in [`specs/deferred_items.md`](../../deferred_items.md).

The plan ID is **6**, although this is roadmap Stage 4. Plans 3 and 4 are reserved by the cloud-GPU amendment on branch `claude/cloud-gpu-ces-revisions-ee5804`; plan 5 implemented Stage 3 and is retired at `specs/plans/completed/5-ces-revisions.md`.

**Retirement:** This shared spec does not retire with this plan. At completion, tick Stage 4 in the roadmap and append this authoritative stamp to the spec's Rollout note:\
`Stage 4: COMPLETE (YYYY-MM-DD) — implemented by plan 6 (specs/plans/completed/6-ces-revisions.md). Next: resume the roadmap.`

## Scope check

Stage 4 names four datasets, but they share three contracts that make one plan coherent: the annual benchmark publication clock, one immutable-source/provenance model, and one build manifest consumed together by later stages. Each dataset remains a separate module and reviewer-sized task. Splitting them into four plans would duplicate the publication and provenance work and would leave Stage 5 unable to consume an atomic benchmark/birth–death handoff.

## Planning evidence (2026-09-15)

- Stage 3 is already retired in commit `39defc9`; its `build_release_index()` returns one row per Employment Situation reference month with `published_date`, UTC `observable_at`, and a benchmark-release flag from the May 2003 carrier onward. Final benchmark year 2002 is carried by the May 2003 release; benchmark year 2003 onward is carried by the following January release.
- The current [CES technical notes](https://www.bls.gov/web/empsit/cestn.htm) Table 5 prints final total-nonfarm revisions for 1979–2025 and preliminary revisions for 2000–2025. The [March 2026 preliminary release](https://www.bls.gov/news.release/prebmk.htm) adds −79 thousand, published August 28, 2026; its final is due with January 2027 data in February 2027.
- Table 5's footnotes identify exceptional scope/reconstruction work in benchmark years 2002, 2010, 2011, 2013, 2015, 2017, 2019, 2022, 2024, and 2025. Footnote 12 contains two distinct 2025 events: the taxi-and-limousine benchmark-level exclusion and the monetary-authorities/commercial-banking reconstruction.
- The 2025 article prints −861 thousand in its summary but its Table 3 levels differ by −862 thousand. No primary source resolves the one-thousand discrepancy; the output therefore preserves both, without choosing a causal explanation.
- The [historical birth–death page](https://www.bls.gov/web/empsit/cesbdhst.htm) carries initial January–December schedules for 2004–2025, post-benchmark April–December schedules for 2003–2025, and older April–March tables. The live page repeats the 2025 post-benchmark schedule and carries the partial 2026 preliminary schedule. Its aggregate is the sum of private supersectors and has no government row.
- Benchmark articles for 2009–2025 contain the annual April–March forecast-versus-realized table, under changing labels (unnumbered; exhibits 7 and 6; table 7; then table 8). These PDF cells should be transcribed, not inferred from prose totals.
- The public QCEW revisions CSV observed on 2026-09-15 had 9,990 data rows and the exact header `Year,Quarter,Area,Field,Initial Value,First Revised Value,Second Revised Value,Third Revised Value,Fourth Revised Value,Final Value`. The national March-employment rows begin in 2017. First-quarter values have five publications (initial plus four revisions); `Final Value` repeats the fourth-revised value rather than adding a sixth publication.
- QCEW employment values are person-count jobs, not CES thousands. The precision proxy uses March-employment revision increments after converting the increments to thousands; the raw long sequence stays in jobs.
- Technical-note Table 1 survives for March 2002–2011 and 2013–2025, but no March 2012 copy was found. Its percentage is sample employees divided by benchmark employment and its footnote says the counts reflect active sample reports.
- The RSE measure changes. Through the 2009 copy, Table 2-E reports RSE of levels at first closing. The 2010–2016 technical-note copies do not carry that table. From the February 2017 copy onward, Table 4 reports an RSE for a one-month change. The panel must preserve those regimes and the gap.

## Execution notes

- **Plan artifact:** Before creating the execution worktree, ensure this file is tracked on the starting branch. If the planning session left it untracked, make a plan-only commit with `git add specs/plans/6-ces-revisions.md` followed by `git commit -m "docs: plan Stage 4 annual tables"`; otherwise the new worktree will not contain the handoff.
- **Isolation:** Before Task 1, use `using-git-worktrees` and create branch `codex/stage-4-annual-tables` from the then-current `main`. Do not execute in the user's main checkout.
- **Network:** Task 1's fetch contacts BLS and the Internet Archive. Load the existing gitignored `BLS_CONTACT_EMAIL` setting without printing it. Do not fetch on an Employment Situation or QCEW release morning while a live source can change underneath the run.
- **Source capture:** Commit source bytes and their SHA-256 hashes. A later BLS overwrite is a new source vintage, never an in-place unexplained test update.
- **PDF transcription:** Use `pdftotext -layout` to locate cells, preserve the printed string in the transcription CSV, and have a second pass compare every printed row total/difference with the transcribed row. Do not parse prose summaries in place of tables.
- **HTML archives:** The source catalog selects one Internet Archive capture after each applicable benchmark publication and before the next one. Record the capture timestamp and original URL separately. An archive timestamp is evidence of the copy, not the publication date.
- **Commands:** Run `uv run ruff format` and `uv run ruff check` before every commit, and stage paths explicitly. Never use `git add -A`.
- **Stop conditions:** Stop and report instead of loosening a test when a source header changes, a planned table is absent, two candidate archive captures disagree, a QCEW sequence has the wrong number of releases, sector contributions or birth–death totals fail their printed identities, a final benchmark carrier cannot be resolved uniquely in Stage 3, or an `observable_at` would have to be guessed.

## File structure

### Create

- `src/ces_revisions/annual/__init__.py` — Stage 4 package boundary.
- `src/ces_revisions/annual/raw.py` — paths, schemas, raw-cell keys, numeric parsing, hashes, and named transformations.
- `src/ces_revisions/annual/html_tables.py` — small standard-library HTML table reader shared by BLS pages.
- `src/ces_revisions/annual/publications.py` — final-benchmark, preliminary-announcement, and QCEW publication calendar.
- `src/ces_revisions/annual/benchmarks.py` — Table 5 anchors and PDF-transcribed sector rows.
- `src/ces_revisions/annual/reconstructions.py` — structured exceptional scope/reconstruction events.
- `src/ces_revisions/annual/birth_death.py` — monthly initial/post-benchmark schedules and annual forecast/realized rows.
- `src/ces_revisions/annual/qcew.py` — national aggregate revision sequence and March precision proxy.
- `src/ces_revisions/annual/sample.py` — annual Table 1 coverage and RSE panel.
- `src/ces_revisions/annual/build.py` — complete Stage 4 build and artifact writer.
- `src/ces_revisions/annual/contract.py` — executable cross-artifact Stage 4 exit gate.
- `scripts/annual_sources.py` — network fetch, offline manifest refresh, and build CLI.
- `tests/annual_data.py` — session-cached Stage 4 artifacts for hermetic exit tests.
- `tests/test_annual_raw.py` — source archive, parser, raw-cell, and transformation tests.
- `tests/test_annual_sources.py` — fetch/catalog/manifest tests and live canaries.
- `tests/test_annual_publications.py` — publication-clock tests.
- `tests/test_benchmarks.py` — benchmark completeness, sector identities, and 2025 discrepancy tests.
- `tests/test_reconstructions.py` — event completeness and footnote-12 tests.
- `tests/test_birth_death.py` — schedule parsing, totals, timing, annual identity, and government-zero tests.
- `tests/test_qcew.py` — CSV contract, sequence, units, dates, and precision tests.
- `tests/test_sample_panel.py` — coverage proxy, missing 2012, RSE regimes, and annual-frequency tests.
- `tests/test_annual_build.py` — artifact/manifest round-trip tests.
- `tests/test_annual_contract.py` — cross-artifact provenance and `observable_at` exit gate.
- `tests/fixtures/annual/*.htm` and `tests/fixtures/annual/*.csv` — minimal source-layout fixtures copied from committed sources.
- `data/annual/raw/manifest.csv` — hashes and acquisition metadata for every Stage 4 source/transcription.
- `data/annual/raw/source-catalog.csv` — role, reference year, source URL, local file, and capture/publication metadata.
- `data/annual/raw/manual/publication-dates.csv` — cited preliminary-benchmark and QCEW release dates.
- `data/annual/raw/manual/benchmark-sector-cells.csv` — printed benchmark-article sector cells, 2003–2025.
- `data/annual/raw/manual/reconstruction-events.csv` — structured rows tied to Table 5 footnotes/articles.
- `data/annual/raw/manual/birth-death-annual-cells.csv` — printed forecast/realized PDF cells, 2009–2025.
- `data/annual/raw/manual/birth-death-rules.csv` — cited structural rules, including the government zero.
- `data/annual/raw/manual/sample-source-regimes.csv` — one row per benchmark year describing the selected coverage/RSE table and measure.
- `data/annual/raw/bls/**` and `data/annual/raw/internet-archive/**` — downloaded BLS originals and selected archive captures.

### Modify

- `.gitignore` — ignore `data/annual/cache/` and `data/annual/panel/`, not the raw archive.
- `pyproject.toml` and `uv.lock` — add and lock `python-dotenv>=1.0` for `.project.env` loading.
- `README.md` — document the annual tables and offline build command.
- `CLAUDE.md` — document the Stage 4 source/panel contract and commands.
- `specs/deferred_items.md` — at completion, close the coverage design item implemented here; preserve the open BLS-request item.
- `specs/ces-revisions-roadmap.md` and `specs/ces-revisions.md` — completion markup only after the plan gate passes.

---

### Task 1: Create the Annual Source Archive and Its Manifest

**Files:**

- Create: `src/ces_revisions/annual/__init__.py`
- Create: `src/ces_revisions/annual/raw.py`
- Create: `scripts/annual_sources.py`
- Create: `tests/test_annual_sources.py`
- Create: `tests/test_annual_raw.py`
- Create: `data/annual/raw/source-catalog.csv`
- Create: `data/annual/raw/manual/publication-dates.csv`
- Create: `data/annual/raw/manifest.csv`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**

- Consumes: the gitignored `.project.env`, its `BLS_CONTACT_EMAIL` key, `python-dotenv.load_dotenv()`, and `ces_revisions.vintages.raw.file_sha256()`'s hashing convention.
- Produces: `annual.raw.RAW_DIR`, `PANEL_DIR`, `MANIFEST`, `SOURCE_CATALOG`, `PUBLICATION_DATES`, `TRANSFORMATIONS`, `cell_key()`, `parse_number()`, `raw_frame()`, and `manifest_frame()`; `annual_sources.load_contact_email(env_file)`, `user_agent(contact_email)`, `download(url, contact_email)`, `fetch_sources(now, env_file=...)`, `refresh_manifest(now)`, and `main(argv)`.

- [x] **Step 1: Write the failing path, parsing, and source-manifest tests**

Create `tests/test_annual_raw.py`:

```python
"""Stage 4 source cells and transformations."""

from pathlib import Path

import polars as pl
import pytest

from ces_revisions.annual import raw


def test_annual_paths_do_not_mutate_stage_3s_archive():
    assert raw.RAW_DIR == raw.ROOT / "data" / "annual" / "raw"
    assert raw.PANEL_DIR == raw.ROOT / "data" / "annual" / "panel"
    assert raw.RAW_DIR != raw.ROOT / "data" / "raw"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-861", -861.0),
        ("−862", -862.0),
        ("83,200", 83200.0),
        ("< 0.05", 0.05),
        ("Not yet published", None),
        ("Not applicable", None),
        ("", None),
    ],
)
def test_parse_number_preserves_missingness_and_normalizes_printed_numbers(
    text, expected
):
    assert raw.parse_number(text) == expected


def test_cell_keys_are_stable_and_include_the_printed_location():
    assert raw.cell_key("bls/cestn.htm", "table_5", "2025", "final_thousands") == (
        "bls/cestn.htm::table_5::2025::final_thousands"
    )


def test_raw_frame_rejects_duplicate_cell_keys():
    records = [
        {
            "source": "bls",
            "file": "bls/cestn.htm",
            "table_key": "table_5",
            "row_key": "2025",
            "column_key": "final_thousands",
            "text": "-861",
        }
    ]
    with pytest.raises(ValueError, match="duplicate raw cell keys"):
        raw.raw_frame(records + records)


def test_transformations_name_every_non_identity_derivation():
    names = set(raw.TRANSFORMATIONS["transformation"])
    assert names == {
        "identity",
        "parse_published_number",
        "parse_thousands",
        "benchmark_article_comparison",
        "qcew_jobs_to_thousands",
        "successive_difference",
        "rms_qcew_revision",
        "realized_minus_forecast",
        "government_structural_zero",
        "explicit_archive_gap",
    }


def test_manifest_frame_hashes_every_file_but_the_manifest(tmp_path: Path):
    (tmp_path / "bls").mkdir()
    (tmp_path / "bls" / "page.htm").write_text("source", encoding="utf-8")
    (tmp_path / "source-catalog.csv").write_text("source_id\npage\n", encoding="utf-8")
    frame = raw.manifest_frame(tmp_path, fetched_at="2026-09-15T12:00:00Z")
    assert frame["file"].to_list() == ["bls/page.htm", "source-catalog.csv"]
    assert frame["sha256"].str.len_chars().eq(64).all()
    assert frame.schema["bytes"] == pl.Int64
```

Create `tests/test_annual_sources.py`:

```python
"""The Stage 4 fetch/build script and committed source manifest."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

import annual_sources
from ces_revisions.annual import raw


def manifest_rows() -> list[dict[str, str]]:
    with (raw.RAW_DIR / raw.MANIFEST).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_dotenv_supplies_the_contact_without_exposing_it(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    env_file = tmp_path / ".project.env"
    env_file.write_text("BLS_CONTACT_EMAIL=someone@example.org\n", encoding="utf-8")
    contact = annual_sources.load_contact_email(env_file)
    assert contact == "someone@example.org"
    assert annual_sources.user_agent(contact) == (
        "ces-revisions/0.1.0 (someone@example.org)"
    )


def test_exported_contact_takes_precedence_over_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "exported@example.org")
    env_file = tmp_path / ".project.env"
    env_file.write_text("BLS_CONTACT_EMAIL=file@example.org\n", encoding="utf-8")
    assert annual_sources.load_contact_email(env_file) == "exported@example.org"


def test_contact_setting_must_look_like_an_email(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "anonymous-client")
    assert annual_sources.load_contact_email(tmp_path / "missing.env") is None


def test_fetch_stops_without_the_bls_contact_setting(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.delenv("BLS_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(
        annual_sources,
        "download",
        lambda url, contact: (_ for _ in ()).throw(
            AssertionError(f"downloaded {url} as {contact}")
        ),
    )
    result = annual_sources.fetch_sources(
        datetime(2026, 9, 15, tzinfo=UTC),
        env_file=tmp_path / "missing.env",
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "BLS_CONTACT_EMAIL" in captured.err
    assert "@" not in captured.err


def test_payload_validation_rejects_an_html_error_saved_as_a_pdf():
    with pytest.raises(ValueError, match="PDF signature"):
        annual_sources.validate_payload(
            "https://www.bls.gov/ces/publications/benchmark/example.pdf",
            b"<html>temporarily unavailable</html>",
        )


def test_fetch_validates_every_payload_before_replacing_sources(
    monkeypatch, tmp_path: Path
):
    raw_dir = tmp_path / "raw"
    existing = raw_dir / "bls" / "one.htm"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old source")
    catalog = pl.DataFrame(
        {
            "status": ["available", "available"],
            "url": ["https://www.bls.gov/one.htm", "https://www.bls.gov/two.pdf"],
            "file": ["bls/one.htm", "bls/two.pdf"],
        }
    )
    monkeypatch.setattr(raw, "RAW_DIR", raw_dir)
    monkeypatch.setattr(raw, "read_source_catalog", lambda: catalog)
    monkeypatch.setattr(annual_sources.shutil, "which", lambda command: "/pdftotext")
    monkeypatch.setattr(
        annual_sources,
        "download",
        lambda url, contact: (
            (b"<html>new source</html>" if url.endswith(".htm") else b"error"),
            "",
        ),
    )
    monkeypatch.setenv("BLS_CONTACT_EMAIL", "someone@example.org")
    with pytest.raises(ValueError, match="PDF signature"):
        annual_sources.fetch_sources(
            datetime(2026, 9, 15, tzinfo=UTC),
            env_file=tmp_path / "missing.env",
        )
    assert existing.read_bytes() == b"old source"


def test_source_catalog_has_one_local_path_per_available_source():
    catalog = raw.read_source_catalog()
    available = catalog.filter(pl.col("status") == "available")
    assert available["source_id"].n_unique() == available.height
    assert available["file"].str.len_chars().gt(0).all()
    assert available["url"].str.starts_with("https://").all()


def test_manifest_matches_every_committed_annual_source():
    rows = manifest_rows()
    files = sorted(
        str(path.relative_to(raw.RAW_DIR))
        for path in raw.RAW_DIR.rglob("*")
        if path.is_file() and path.name not in {raw.MANIFEST, ".DS_Store"}
    )
    assert [row["file"] for row in rows] == files
    for row in rows:
        path = raw.RAW_DIR / row["file"]
        assert row["sha256"] == raw.file_sha256(path)
        assert row["bytes"] == str(path.stat().st_size)


@pytest.mark.network
def test_live_qcew_header_has_not_changed():
    contact = annual_sources.load_contact_email()
    assert contact is not None
    payload, _ = annual_sources.download(annual_sources.QCEW_REVISIONS_URL, contact)
    assert payload.splitlines()[0].decode() == (
        "Year,Quarter,Area,Field,Initial Value,First Revised Value,"
        "Second Revised Value,Third Revised Value,Fourth Revised Value,Final Value"
    )
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_annual_raw.py tests/test_annual_sources.py -q -m "not network"`

Expected: collection fails because `ces_revisions.annual` and `annual_sources` do not exist yet; no test body runs.

- [x] **Step 3: Create the package boundary and raw contracts**

Create `src/ces_revisions/annual/__init__.py`:

```python
"""Roadmap Stage 4's annual benchmark, birth-death, QCEW, and sample tables."""
```

Create `src/ces_revisions/annual/raw.py`:

```python
"""Paths, source cells, hashes, and named transformations for Stage 4."""

import hashlib
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "annual"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST = "manifest.csv"
SOURCE_CATALOG = "source-catalog.csv"
PUBLICATION_DATES = "manual/publication-dates.csv"

RAW_SCHEMA = {
    "cell_key": pl.String,
    "source": pl.String,
    "file": pl.String,
    "table_key": pl.String,
    "row_key": pl.String,
    "column_key": pl.String,
    "text": pl.String,
}

TRANSFORMATIONS = pl.DataFrame(
    [
        ("identity", "Copy a published numeric value without rescaling."),
        (
            "parse_published_number",
            "Parse a published count or percentage without changing its unit.",
        ),
        ("parse_thousands", "Parse a CES value already printed in thousands."),
        (
            "benchmark_article_comparison",
            "Parse the Table 5 anchor and subtract it from the separately published article total.",
        ),
        (
            "qcew_jobs_to_thousands",
            "Divide a QCEW person-count employment value by 1,000.",
        ),
        (
            "successive_difference",
            "Subtract the preceding publication of the same QCEW cell.",
        ),
        (
            "rms_qcew_revision",
            "Root mean square of the four finalized Q1 March revision increments, in thousands.",
        ),
        (
            "realized_minus_forecast",
            "Subtract annual forecast birth-death from its published realized value.",
        ),
        (
            "government_structural_zero",
            "Set birth-death to zero because the model applies only to private industries.",
        ),
        (
            "explicit_archive_gap",
            "Materialize a null row where the source table is documented absent.",
        ),
    ],
    schema=["transformation", "description"],
    orient="row",
)

_MISSING = {"", "-", "--", "N/A", "NA", "Not applicable", "Not yet published"}


def cell_key(file: str, table_key: str, row_key: str, column_key: str) -> str:
    """A stable, human-readable key for one printed source cell."""
    return "::".join((file, table_key, row_key, column_key))


def parse_number(text: str) -> float | None:
    """Parse BLS numeric text while preserving documented missing values."""
    cleaned = re.sub(r"\([A-Za-z0-9]+\)$", "", text.strip())
    cleaned = cleaned.replace("−", "-").replace(",", "").strip()
    if cleaned in _MISSING:
        return None
    if cleaned.startswith("< "):
        cleaned = cleaned[2:]
    return float(cleaned)


def raw_frame(records: list[dict[str, str]]) -> pl.DataFrame:
    """Build a sorted immutable raw-cell frame and reject ambiguous locations."""
    rows = []
    for record in records:
        row = dict(record)
        row["cell_key"] = cell_key(
            row["file"], row["table_key"], row["row_key"], row["column_key"]
        )
        rows.append(row)
    frame = pl.DataFrame(rows, schema=RAW_SCHEMA).sort("cell_key")
    if frame["cell_key"].n_unique() != frame.height:
        raise ValueError("duplicate raw cell keys")
    return frame


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(frame: pl.DataFrame) -> str:
    payload = frame.write_csv().encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_frame(raw_dir: Path = RAW_DIR, *, fetched_at: str) -> pl.DataFrame:
    """Hash every source/transcription below raw_dir except manifest.csv itself."""
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name in {MANIFEST, ".DS_Store"}:
            continue
        rows.append(
            {
                "file": str(path.relative_to(raw_dir)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "fetched_at": fetched_at,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "file": pl.String,
            "sha256": pl.String,
            "bytes": pl.Int64,
            "fetched_at": pl.String,
        },
    )


def read_source_catalog(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / SOURCE_CATALOG, infer_schema_length=0)
```

- [x] **Step 4: Create the source catalog and publication-date evidence table**

Create `data/annual/raw/source-catalog.csv` with this exact header:

```csv
source_id,role,benchmark_year,status,url,file,capture_timestamp,original_url,source_locator
```

Populate it under these exhaustive rules:

1. `cestn-current`: the current technical notes, used for Table 5 and the current sample tables.
2. `cesbd-history`, `cesbd-current`, and `cesbd-faq`: the two birth–death tables pages and the FAQ that documents the private-industry scope used for the government structural zero.
3. `qcew-revisions`: the revisions CSV.
4. `benchmark-article-YYYY`: one BLS PDF for every benchmark year 2002–2025.
5. `coverage-YYYY`: one post-publication technical-note capture for every year 2002–2011 and 2013–2024, plus the current 2025 page. Add `coverage-2012` with `status=missing`, blank `file`, and the CDX query locator that demonstrates the gap.
6. `rse-level-YYYY`: each surviving Table 2-E/all-employee level-RSE copy through 2009.
7. `rse-change-YYYY`: one post-publication Table 4 copy for every benchmark year 2016–2025; the first was visible in the February 2017 copy.
8. One cited evidence source for every row later placed in `manual/publication-dates.csv`.
9. `cbj-2024` and `cbj-2027`: the two DOL BLS budget-justification PDFs that define the non-comparable FY2022 and FY2025 published-series counts retained as sample metadata.

For an Internet Archive row, `url` is the replay URL, `original_url` is the BLS URL, and `capture_timestamp` is the 14-digit CDX timestamp. For a BLS row, leave the latter two fields blank. No two available rows may point to the same local file.

Create `data/annual/raw/manual/publication-dates.csv` with this exact header:

```csv
publication_id,publication_kind,benchmark_year,qcew_year,qcew_quarter,release_order,publication_date,release_time_et,source_file,source_locator,source_url
```

Transcribe dates printed by cited BLS releases, using these completeness rules:

- `benchmark_preliminary_2000` through `benchmark_preliminary_2026`, one row each, with `publication_kind=benchmark_preliminary`, `release_time_et=10:00:00`, and blank QCEW keys.
- For every release represented in the QCEW CSV, one `qcew_YYYY_qQ_ORDER` row per actual publication order: Q1 orders 0–4, Q2 orders 0–3, Q3 orders 0–2, and Q4 orders 0–1. Use the BLS QCEW release date, not the month-end or archive-capture date.

The first and last preliminary rows must be source-supported; the last row is exactly:

```csv
benchmark_preliminary_2026,benchmark_preliminary,2026,,,,2026-08-28,10:00:00,bls/preliminary/prebmk-2026.htm,release header,https://www.bls.gov/news.release/archives/prebmk_08282026.htm
```

- [x] **Step 5: Write the fetch and manifest CLI**

Add and lock the sole new runtime dependency:

```bash
uv add "python-dotenv>=1.0"
```

Expected: `pyproject.toml` contains `python-dotenv>=1.0`, `uv.lock` is refreshed, and no `.project.env` content appears in either file.

Create `scripts/annual_sources.py`:

```python
"""Fetch Stage 4 sources, refresh their manifest, or build the annual tables."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl
from dotenv import load_dotenv

from ces_revisions.annual import raw

QCEW_REVISIONS_URL = "https://www.bls.gov/cew/revisions/qcew-revisions.csv"
PROJECT_ENV = raw.ROOT / ".project.env"
USER_AGENT_PRODUCT = "ces-revisions/0.1.0"


def load_contact_email(env_file: Path = PROJECT_ENV) -> str | None:
    """Load project-local configuration; an exported value keeps precedence."""
    load_dotenv(env_file)
    value = os.environ.get("BLS_CONTACT_EMAIL", "").strip()
    return value if "@" in value and not value.startswith("@") else None


def user_agent(contact_email: str) -> str:
    """Return the descriptive identity sent with every Stage 4 request."""
    return f"{USER_AGENT_PRODUCT} ({contact_email})"


def download(url: str, contact_email: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent(contact_email)}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get("Last-Modified", "")


def validate_payload(url: str, payload: bytes) -> None:
    """Reject common BLS error pages before any committed source is replaced."""
    if not payload:
        raise ValueError(f"{url} returned an empty payload")
    suffix = Path(urlsplit(url).path).suffix.lower()
    prefix = payload.lstrip()[:4096].lower()
    if suffix == ".pdf" and not payload.startswith(b"%PDF-"):
        raise ValueError(f"{url} does not have a PDF signature")
    if suffix in {".htm", ".html"} and b"<html" not in prefix:
        raise ValueError(f"{url} does not have an HTML signature")
    if suffix == ".csv" and b"," not in payload.splitlines()[0]:
        raise ValueError(f"{url} does not have a CSV header")


def refresh_manifest(now: datetime, raw_dir: Path = raw.RAW_DIR) -> int:
    stamp = (
        now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    frame = raw.manifest_frame(raw_dir, fetched_at=stamp)
    frame.write_csv(raw_dir / raw.MANIFEST)
    return frame.height


def fetch_sources(now: datetime, env_file: Path = PROJECT_ENV) -> int:
    contact_email = load_contact_email(env_file)
    if not contact_email:
        print(
            "set BLS_CONTACT_EMAIL to the contact address BLS asks automated clients for",
            file=sys.stderr,
        )
        return 1
    if shutil.which("pdftotext") is None:
        print("pdftotext (poppler) is required: brew install poppler", file=sys.stderr)
        return 1
    catalog = raw.read_source_catalog()
    with tempfile.TemporaryDirectory(prefix="ces-stage4-fetch-") as directory:
        staging = Path(directory)
        for row in catalog.filter(pl.col("status") == "available").iter_rows(
            named=True
        ):
            target = staging / row["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            payload, _ = download(row["url"], contact_email)
            validate_payload(row["url"], payload)
            target.write_bytes(payload)
            if target.suffix.lower() == ".pdf":
                text_target = target.with_suffix(".txt")
                subprocess.run(
                    ["pdftotext", "-layout", str(target), str(text_target)],
                    check=True,
                )
                if not text_target.read_text(encoding="utf-8").strip():
                    raise ValueError(f"pdftotext produced no text for {row['url']}")
        for staged in sorted(path for path in staging.rglob("*") if path.is_file()):
            target = raw.RAW_DIR / staged.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
    count = refresh_manifest(now)
    print(f"manifest: {count} source files")
    return 0


def build_tables() -> int:
    from ces_revisions.annual.build import build, write

    manifest = write(build())
    for name, entry in manifest["artifacts"].items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "manifest", "build"])
    command = parser.parse_args(argv).command
    now = datetime.now(UTC)
    if command == "fetch":
        return fetch_sources(now)
    if command == "manifest":
        print(f"manifest: {refresh_manifest(now)} source files")
        return 0
    return build_tables()


if __name__ == "__main__":
    raise SystemExit(main())
```

The local import inside `build_tables()` keeps Task 1 test collection independent of the build module written in Task 7. This follows `/Users/lowell/Projects/bls-stats/src/bls_stats/core/config.py`: `load_dotenv()` runs before the environment lookup and does not override an explicitly exported variable. Unlike the older Stage 3 helper, every Stage 4 request carries the contact address. The real address is never printed or written to the source manifest.

- [x] **Step 6: Ignore only rebuilt annual artifacts**

Append to `.gitignore`:

```gitignore

# Stage 4 annual data: raw sources are committed; cache and built parquet are rebuilt
data/annual/cache/
data/annual/panel/
```

- [x] **Step 7: Materialize the first manifest and run the unit tests**

Run:

```bash
uv run python scripts/annual_sources.py manifest
uv run pytest tests/test_annual_raw.py tests/test_annual_sources.py -q -m "not network"
```

Expected: all non-network tests pass against the catalog, manual publication table, and then-current contents of `data/annual/raw/`; the network fetch itself remains isolated to Step 8.

- [x] **Step 8: Fetch and inspect the committed source archive**

Run:

```bash
uv run python scripts/annual_sources.py fetch
uv run python scripts/annual_sources.py manifest
git status --short data/annual
```

Expected: every `status=available` catalog path exists; every PDF has a sibling `.txt`; `manifest.csv` lists every source, catalog, transcription, and derived PDF text file; no cache or panel file is staged.

Inspect, without changing parser tests:

```bash
head -n 1 data/annual/raw/bls/qcew-revisions.csv
rg -n "Table 5|2025|Footnotes" data/annual/raw/bls/cestn.htm
rg -n "Net Birth-Death Forecast" data/annual/raw/bls/cesbd-history.htm
```

Expected: the QCEW header equals the pinned header in Step 1; Table 5 contains years 1979 and 2025 plus footnote 12; the birth–death page contains initial and post-benchmark captions.

- [x] **Step 9: Run the committed-source and live tests**

Run:

```bash
uv run pytest tests/test_annual_raw.py tests/test_annual_sources.py -q -m "not network"
uv run pytest tests/test_annual_sources.py -q -m network
uv run ruff format src/ces_revisions/annual scripts/annual_sources.py tests/test_annual_raw.py tests/test_annual_sources.py
uv run ruff check src/ces_revisions/annual scripts/annual_sources.py tests/test_annual_raw.py tests/test_annual_sources.py
```

Expected: all tests pass and ruff is clean.

- [x] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml uv.lock src/ces_revisions/annual/__init__.py src/ces_revisions/annual/raw.py scripts/annual_sources.py tests/test_annual_raw.py tests/test_annual_sources.py tests/fixtures/annual data/annual/raw
git commit -m "Archive and hash Stage 4 annual sources"
```

---

### Task 2: Parse HTML Tables and Build the Publication Clock

**Files:**

- Create: `src/ces_revisions/annual/html_tables.py`
- Create: `src/ces_revisions/annual/publications.py`
- Create: `tests/test_annual_publications.py`
- Modify: `tests/test_annual_raw.py`

**Interfaces:**

- Consumes: `annual.raw.raw_frame()`, `vintages.release_index.build_release_index()`, and `manual/publication-dates.csv`.
- Produces: `HtmlTable(caption, rows)`, `parse_tables(page)`, `find_table(page, caption)`, `table_raw_cells(...)`, `final_carrier_month(year)`, `catalog_publications(raw_dir)`, `final_benchmark_publications(release_index)`, and `build_publication_calendar(release_index, raw_dir)`.

- [x] **Step 1: Write the failing HTML parser tests**

Append to `tests/test_annual_raw.py`:

```python
from ces_revisions.annual import html_tables


def test_html_tables_keep_caption_rows_and_nested_text():
    page = """
    <table><caption>Table 5. Benchmark <em>revisions</em></caption>
      <tr><th>Year</th><th>Final</th></tr>
      <tr><td>2025<sup>(12)</sup></td><td>−861</td></tr>
    </table>
    """
    table = html_tables.find_table(page, r"Table 5\. Benchmark")
    assert table.caption == "Table 5. Benchmark revisions"
    assert table.rows == (("Year", "Final"), ("2025 (12)", "−861"))


def test_find_table_requires_exactly_one_matching_caption():
    page = "<table><caption>A</caption></table><table><caption>A</caption></table>"
    with pytest.raises(ValueError, match="expected one table"):
        html_tables.find_table(page, "A")


def test_table_raw_cells_preserve_printed_text():
    table = html_tables.HtmlTable("Table 1", (("Industry", "Coverage"), ("00", "26")))
    cells = html_tables.table_raw_cells(
        table, source="bls", file="bls/cestn.htm", table_key="coverage_2025"
    )
    assert cells.select("row_key", "column_key", "text").rows() == [
        ("0", "0", "Industry"),
        ("0", "1", "Coverage"),
        ("1", "0", "00"),
        ("1", "1", "26"),
    ]
```

- [x] **Step 2: Write the failing publication tests**

Create `tests/test_annual_publications.py`:

```python
"""Stage 4 publication dates and as-of timestamps."""

from datetime import date

import polars as pl

from ces_revisions.annual import publications
from ces_revisions.vintages.release_index import build_release_index


def test_final_carrier_changes_after_benchmark_year_2002():
    assert publications.final_carrier_month(1979) == date(1980, 5, 1)
    assert publications.final_carrier_month(2002) == date(2003, 5, 1)
    assert publications.final_carrier_month(2003) == date(2004, 1, 1)
    assert publications.final_carrier_month(2025) == date(2026, 1, 1)


def test_final_publications_match_stage_3s_release_index():
    index = build_release_index()
    finals = publications.final_benchmark_publications(index)
    assert finals["benchmark_year"].to_list() == list(range(1979, 2026))
    expected = index.filter(pl.col("reference_month") == date(2026, 1, 1)).row(
        0, named=True
    )
    row = finals.filter(pl.col("benchmark_year") == 2025).row(0, named=True)
    assert row["publication_date"] == expected["published_date"] == date(2026, 2, 11)
    assert row["observable_at"] == expected["observable_at"]


def test_preliminary_publications_are_complete_and_separate():
    calendar = publications.build_publication_calendar(build_release_index())
    preliminary = calendar.filter(pl.col("publication_kind") == "benchmark_preliminary")
    assert preliminary["benchmark_year"].to_list() == list(range(2000, 2027))
    assert preliminary.filter(pl.col("benchmark_year") == 2026).row(0, named=True)[
        "publication_date"
    ] == date(2026, 8, 28)
    assert preliminary["source_file"].str.len_chars().gt(0).all()


def test_every_catalog_publication_has_a_utc_observable_timestamp():
    calendar = publications.build_publication_calendar(build_release_index())
    assert calendar["publication_date"].is_not_null().all()
    assert calendar["observable_at"].is_not_null().all()
    assert (
        str(calendar.schema["observable_at"])
        == "Datetime(time_unit='us', time_zone='UTC')"
    )
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_annual_raw.py tests/test_annual_publications.py -q`

Expected: collection fails because `html_tables` and `publications` do not exist.

- [x] **Step 4: Write the HTML table reader**

Create `src/ces_revisions/annual/html_tables.py`:

```python
"""Minimal HTML table extraction for the stable BLS tables Stage 4 reads."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser

import polars as pl

from ces_revisions.annual.raw import raw_frame


@dataclass(frozen=True)
class HtmlTable:
    caption: str
    rows: tuple[tuple[str, ...], ...]


def _clean(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[HtmlTable] = []
        self.in_table = False
        self.in_caption = False
        self.in_cell = False
        self.caption_parts: list[str] = []
        self.cell_parts: list[str] = []
        self.rows: list[tuple[str, ...]] = []
        self.row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table" and not self.in_table:
            self.in_table = True
            self.caption_parts, self.rows = [], []
        elif self.in_table and tag == "caption":
            self.in_caption = True
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_caption:
            self.caption_parts.append(data)
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.in_cell:
            if self.row is None:
                raise ValueError("table cell outside a row")
            self.row.append(_clean(self.cell_parts))
            self.in_cell = False
        elif tag == "tr" and self.in_table:
            if self.row:
                self.rows.append(tuple(self.row))
            self.row = None
        elif tag == "caption":
            self.in_caption = False
        elif tag == "table" and self.in_table:
            self.tables.append(HtmlTable(_clean(self.caption_parts), tuple(self.rows)))
            self.in_table = False


def parse_tables(page: str) -> tuple[HtmlTable, ...]:
    parser = _Parser()
    parser.feed(page)
    return tuple(parser.tables)


def find_table(page: str, caption: str) -> HtmlTable:
    pattern = re.compile(caption, re.IGNORECASE)
    matches = [table for table in parse_tables(page) if pattern.search(table.caption)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one table matching {caption!r}, found {len(matches)}"
        )
    return matches[0]


def table_raw_cells(
    table: HtmlTable, *, source: str, file: str, table_key: str
) -> pl.DataFrame:
    records = [
        {
            "source": source,
            "file": file,
            "table_key": table_key,
            "row_key": str(row_index),
            "column_key": str(column_index),
            "text": text,
        }
        for row_index, row in enumerate(table.rows)
        for column_index, text in enumerate(row)
    ]
    return raw_frame(records)
```

- [x] **Step 5: Write the publication calendar**

> **Deviation (source correction):** The shipped calendar uses the actual historical
> preliminary-announcement dates and times, including the earlier 2011–2018 notices,
> rather than deriving them from later annual carriers. QCEW news-release and full-data
> availability are represented as separate product events because BLS sometimes published
> them on different dates.

Create `src/ces_revisions/annual/publications.py`:

```python
"""Publication dates for Stage 4 values."""

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ces_revisions.annual.raw import PUBLICATION_DATES, RAW_DIR, cell_key

EASTERN = ZoneInfo("America/New_York")

PUBLICATION_SCHEMA = {
    "publication_id": pl.String,
    "publication_kind": pl.String,
    "benchmark_year": pl.Int32,
    "qcew_year": pl.Int32,
    "qcew_quarter": pl.Int8,
    "release_order": pl.Int8,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
}


def _utc_observable(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, EASTERN).astimezone(UTC)


def final_carrier_month(benchmark_year: int) -> date:
    if benchmark_year <= 2002:
        return date(benchmark_year + 1, 5, 1)
    return date(benchmark_year + 1, 1, 1)


def catalog_publications(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(
        raw_dir / PUBLICATION_DATES,
        schema_overrides={
            "benchmark_year": pl.Int32,
            "qcew_year": pl.Int32,
            "qcew_quarter": pl.Int8,
            "release_order": pl.Int8,
            "publication_date": pl.Date,
        },
        try_parse_dates=True,
    )
    rows = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        clock = time.fromisoformat(row["release_time_et"])
        rows.append(
            {
                "publication_id": row["publication_id"],
                "publication_kind": row["publication_kind"],
                "benchmark_year": row["benchmark_year"],
                "qcew_year": row["qcew_year"],
                "qcew_quarter": row["qcew_quarter"],
                "release_order": row["release_order"],
                "publication_date": row["publication_date"],
                "observable_at": _utc_observable(row["publication_date"], clock),
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        PUBLICATION_DATES,
                        "publication_dates",
                        str(row_number),
                        "publication_date",
                    )
                ],
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def final_benchmark_publications(release_index: pl.DataFrame) -> pl.DataFrame:
    by_month = {
        row["reference_month"]: row for row in release_index.iter_rows(named=True)
    }
    rows = []
    for year in range(1979, 2026):
        carrier = final_carrier_month(year)
        release = by_month.get(carrier)
        if release is None:
            raise ValueError(f"no Stage 3 release-index row for benchmark year {year}")
        rows.append(
            {
                "publication_id": f"benchmark_final_{year}",
                "publication_kind": "benchmark_final",
                "benchmark_year": year,
                "qcew_year": None,
                "qcew_quarter": None,
                "release_order": None,
                "publication_date": release["published_date"],
                "observable_at": release["observable_at"],
                "source_file": "data/panel/release_index.parquet",
                "source_locator": f"reference_month={carrier:%Y-%m}",
                "source_keys": [f"stage3::release_index::{carrier:%Y-%m}"],
            }
        )
    return pl.DataFrame(rows, schema=PUBLICATION_SCHEMA)


def build_publication_calendar(
    release_index: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    frame = pl.concat(
        [final_benchmark_publications(release_index), catalog_publications(raw_dir)]
    ).sort("publication_date", "publication_id")
    if frame["publication_id"].n_unique() != frame.height:
        raise ValueError("duplicate publication_id")
    preliminary = frame.filter(pl.col("publication_kind") == "benchmark_preliminary")
    if preliminary["benchmark_year"].to_list() != list(range(2000, 2027)):
        raise ValueError("preliminary benchmark publication calendar is incomplete")
    return frame
```

- [x] **Step 6: Add publication-date cells to the raw-value contract**

Append this test to `tests/test_annual_publications.py`:

```python
def test_publication_catalog_dates_are_raw_values():
    raw_values = publications.publication_raw_values()
    expected = publications.catalog_publications()["source_keys"].explode()
    assert set(expected) <= set(raw_values["cell_key"])
```

Append this function to `src/ces_revisions/annual/publications.py`:

```python
def publication_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / PUBLICATION_DATES, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": PUBLICATION_DATES,
                    "table_key": "publication_dates",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)
```

Add `raw_frame` to the import from `ces_revisions.annual.raw`.

- [x] **Step 7: Run the tests and the fast tier**

Run:

```bash
uv run pytest tests/test_annual_raw.py tests/test_annual_publications.py -q
uv run ruff format src/ces_revisions/annual tests/test_annual_raw.py tests/test_annual_publications.py
uv run ruff check src/ces_revisions/annual tests/test_annual_raw.py tests/test_annual_publications.py
uv run pytest -m "not slow and not network" -q
```

Expected: the focused tests and existing fast tier pass.

- [x] **Step 8: Commit**

```bash
git add src/ces_revisions/annual/html_tables.py src/ces_revisions/annual/publications.py tests/test_annual_raw.py tests/test_annual_publications.py data/annual/raw/manual/publication-dates.csv data/annual/raw/manifest.csv
git commit -m "Date annual-source publications on the Stage 3 release clock"
```

---

### Task 3: Build Benchmark Anchors, Sector Contributions, and Reconstruction Events

**Files:**

- Create: `src/ces_revisions/annual/benchmarks.py`
- Create: `src/ces_revisions/annual/reconstructions.py`
- Create: `tests/test_benchmarks.py`
- Create: `tests/test_reconstructions.py`
- Create: `data/annual/raw/manual/benchmark-sector-cells.csv`
- Create: `data/annual/raw/manual/reconstruction-events.csv`
- Modify: `data/annual/raw/manifest.csv`

**Interfaces:**

- Consumes: `find_table()`, `table_raw_cells()`, `parse_number()`, and the publication calendar from Task 2.
- Produces: `benchmarks.table5_raw_values(raw_dir)`, `benchmark_sector_raw_values(raw_dir)`, `build_benchmarks(calendar, raw_dir)`, `reconstructions.raw_values(raw_dir)`, and `build_reconstruction_events(calendar, raw_dir)`.

- [x] **Step 1: Transcribe the benchmark-article sector cells**

Create `data/annual/raw/manual/benchmark-sector-cells.csv` with this exact header:

```csv
benchmark_year,sector,industry_title,revision_text,percent_text,source_file,source_locator
```

For each benchmark year 2003–2025, transcribe the article's NSA major-industry revision table for sector `00` and the eleven sectors `10,20,30,40,50,55,60,65,70,80,90`. Preserve the printed minus sign, commas, decimals, and less-than sign in the two text columns. Use the article PDF path in `source_file` and its printed table/exhibit name plus page in `source_locator`. Do not transcribe the indented wholesale, retail, transportation, or utilities rows as separate supersectors.

Before continuing, check each year's sector identity in a one-off Polars expression: the eleven sector revisions must sum to the article's total to printed rounding. Where an article explicitly explains a scope/reconstruction difference, record that in the reconstruction table, not by changing a sector cell.

- [x] **Step 2: Transcribe the documented event rows**

Create `data/annual/raw/manual/reconstruction-events.csv` with this exact header:

```csv
event_id,benchmark_year,event_type,sectors,series_codes,reference_start,reference_end,effect_thousands_text,footnote,description,source_file,source_locator
```

Create these rows, one per independently described operation:

| `event_id` | Year | Type | Required contents |
|---|---:|---|---|
| `2002-remove-animal-support` | 2002 | `scope_change` | NAICS 1152 removed from CES scope |
| `2002-federal-benchmark-source` | 2002 | `source_change` | federal government moved from OPM end-of-month counts to QCEW-derived levels |
| `2010-census-temporary-workers` | 2010 | `processing_correction` | 91-999900 reconstructed; approximately +4 thousand |
| `2011-noncovered-expansion` | 2011 | `scope_change` | 13 industries added; +95 thousand |
| `2013-qcew-financial-recoding` | 2013 | `classification_reconstruction` | substantial nonrandom changes to NAICS 525 |
| `2013-private-households-recoding` | 2013 | `classification_reconstruction` | about 466 thousand moved from NAICS 814 to 62412 |
| `2015-elderly-services-california` | 2015 | `classification_reconstruction` | 65-624120 rebuilt to January 2000; total −27 thousand |
| `2017-security-microdata-error` | 2017 | `processing_correction` | 60-561613 rebuilt to October 2016; total +3 thousand |
| `2019-rail-double-count` | 2019 | `processing_correction` | 43-482000 rebuilt to January 1990; total −16 thousand |
| `2022-shopping-to-management` | 2022 | `classification_reconstruction` | about 68 thousand moved from 42-454100 to 60-551114; history to January 1990 |
| `2022-elderly-services-imputation` | 2022 | `benchmark_level_adjustment` | 624120 March level imputed; approximately +83 thousand versus reported QCEW |
| `2024-computer-to-management` | 2024 | `classification_reconstruction` | about 50 thousand moved from 31-334100 to 60-551114; history to January 2005 |
| `2025-taxi-exclusion` | 2025 | `benchmark_level_adjustment` | approximately 83.2 thousand in 43-485300 excluded from the benchmark revision amount |
| `2025-central-bank-commercial-bank` | 2025 | `classification_reconstruction` | employment moved 55-521000 → 55-522110; ratio-adjusted to 1990 and re-estimated from March 2024 |

Use Table 5 footnotes 3–12 and the linked benchmark-article sections as `source_file`/`source_locator`. `sectors` and `series_codes` are semicolon-delimited lists. Leave an effect blank only when BLS prints none; never derive an unprinted effect from the benchmark discrepancy.

- [x] **Step 3: Write the failing benchmark tests**

Create `tests/test_benchmarks.py`:

```python
"""Final/preliminary benchmark anchors and major-industry contributions."""

from datetime import date

import polars as pl

from ces_revisions.annual import benchmarks, publications
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return benchmarks.build_benchmarks(calendar)


def test_total_anchor_years_are_complete():
    frame = built().filter(
        (pl.col("sector") == "00") & (pl.col("row_kind") == "anchor")
    )
    final = frame.filter(pl.col("benchmark_status") == "final")
    preliminary = frame.filter(pl.col("benchmark_status") == "preliminary")
    assert final["benchmark_year"].to_list() == list(range(1979, 2026))
    assert preliminary["benchmark_year"].to_list() == list(range(2000, 2027))


def test_march_2026_is_preliminary_only():
    frame = built().filter(pl.col("benchmark_year") == 2026)
    assert frame.select("benchmark_status", "revision_thousands").rows() == [
        ("preliminary", -79.0)
    ]
    assert frame["publication_date"].item() == date(2026, 8, 28)


def test_march_2026_value_is_parsed_from_the_preliminary_release(tmp_path):
    source = tmp_path / "bls" / "preliminary"
    source.mkdir(parents=True)
    (source / "prebmk-2026.htm").write_text(
        """
        <table>
          <caption>Table 1. National Current Employment Statistics March 2026
          Preliminary Benchmark Revisions by Major Industry Sector</caption>
          <tr><th>Industry</th><th>Benchmark revision (in thousands)</th>
              <th>Percent benchmark revision</th></tr>
          <tr><td>Total nonfarm</td><td>-80</td><td>-0.1</td></tr>
        </table>
        """,
        encoding="utf-8",
    )
    revision, percent, keys = benchmarks.preliminary_2026_values(tmp_path)
    assert (revision, percent) == (-80.0, -0.1)
    assert len(keys) == 2


def test_march_2025_preserves_the_unexplained_article_difference():
    row = (
        built()
        .filter(
            (pl.col("benchmark_year") == 2025)
            & (pl.col("benchmark_status") == "final")
            & (pl.col("sector") == "00")
            & (pl.col("row_kind") == "anchor")
        )
        .row(0, named=True)
    )
    assert row["revision_thousands"] == -861.0
    assert row["article_revision_thousands"] == -862.0
    assert row["article_difference_thousands"] == -1.0
    assert row["discrepancy_status"] == "unexplained"
    assert 12 in row["footnotes"]


def test_table_5_footnotes_remain_structured_metadata():
    anchors = built().filter(
        (pl.col("row_kind") == "anchor") & (pl.col("benchmark_status") == "final")
    )
    expected = {
        2002: 3,
        2010: 4,
        2011: 5,
        2013: 6,
        2015: 7,
        2017: 8,
        2019: 9,
        2022: 10,
        2024: 11,
        2025: 12,
    }
    observed = {
        row["benchmark_year"]: row["footnotes"] for row in anchors.iter_rows(named=True)
    }
    assert all(1 in notes for notes in observed.values())
    for year, footnote in expected.items():
        assert footnote in observed[year]


def test_sector_contributions_cover_2003_through_2025_and_sum_to_total():
    sectors = built().filter(pl.col("row_kind") == "sector_contribution")
    assert sectors["benchmark_year"].unique().sort().to_list() == list(
        range(2003, 2026)
    )
    assert sectors.group_by("benchmark_year").len()["len"].unique().to_list() == [11]
    totals = (
        sectors.group_by("benchmark_year")
        .agg(pl.col("revision_thousands").sum().alias("sector_sum"))
        .join(
            built()
            .filter(
                (pl.col("row_kind") == "anchor")
                & (pl.col("benchmark_status") == "final")
            )
            .select("benchmark_year", total=pl.col("article_revision_thousands")),
            on="benchmark_year",
        )
    )
    assert (totals["sector_sum"] - totals["total"]).abs().le(1.0).all()


def test_all_benchmark_rows_are_nsa_and_dated():
    frame = built()
    assert frame["seasonal_status"].unique().to_list() == ["NSA"]
    assert frame["publication_date"].is_not_null().all()
    assert frame["observable_at"].is_not_null().all()
```

- [x] **Step 4: Write the failing reconstruction tests**

Create `tests/test_reconstructions.py`:

```python
"""The sole structured source of documented benchmark reconstructions."""

import polars as pl

from ces_revisions.annual import publications, reconstructions
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return reconstructions.build_reconstruction_events(calendar)


def test_expected_documented_events_exist_once():
    expected = {
        "2002-remove-animal-support",
        "2002-federal-benchmark-source",
        "2010-census-temporary-workers",
        "2011-noncovered-expansion",
        "2013-qcew-financial-recoding",
        "2013-private-households-recoding",
        "2015-elderly-services-california",
        "2017-security-microdata-error",
        "2019-rail-double-count",
        "2022-shopping-to-management",
        "2022-elderly-services-imputation",
        "2024-computer-to-management",
        "2025-taxi-exclusion",
        "2025-central-bank-commercial-bank",
    }
    frame = built()
    assert set(frame["event_id"]) == expected
    assert frame["event_id"].n_unique() == frame.height


def test_footnote_12_is_two_events_not_an_explanation_for_the_one_thousand():
    rows = built().filter(pl.col("footnote") == 12)
    assert set(rows["event_id"]) == {
        "2025-taxi-exclusion",
        "2025-central-bank-commercial-bank",
    }
    assert rows["description"].str.contains("unexplained").not_().all()


def test_reconstruction_publications_match_final_benchmark_publications():
    frame = built()
    calendar = publications.build_publication_calendar(build_release_index()).filter(
        pl.col("publication_kind") == "benchmark_final"
    )
    joined = frame.join(
        calendar.select("benchmark_year", expected=pl.col("observable_at")),
        on="benchmark_year",
    )
    assert (joined["observable_at"] == joined["expected"]).all()
```

- [x] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/test_benchmarks.py tests/test_reconstructions.py -q`

Expected: collection fails because the two modules do not exist.

- [x] **Step 6: Write the benchmark builder**

Create `src/ces_revisions/annual/benchmarks.py` with these public contracts and schema:

```python
"""CES final/preliminary benchmark revisions and major-industry contributions."""

import re
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

TECHNICAL_NOTES = "bls/cestn.htm"
PRELIMINARY_2026 = "bls/preliminary/prebmk-2026.htm"
SECTOR_CELLS = "manual/benchmark-sector-cells.csv"

BENCHMARK_SCHEMA = {
    "benchmark_year": pl.Int32,
    "benchmark_status": pl.String,
    "row_kind": pl.String,
    "sector": pl.String,
    "seasonal_status": pl.String,
    "footnotes": pl.List(pl.Int16),
    "revision_thousands": pl.Float64,
    "percent_revision": pl.Float64,
    "article_revision_thousands": pl.Float64,
    "article_difference_thousands": pl.Float64,
    "discrepancy_status": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _year(text: str) -> int | None:
    match = re.match(r"^(\d{4})", text)
    return int(match.group(1)) if match else None


def _footnotes(*texts: str) -> list[int]:
    return sorted(
        {int(value) for text in texts for value in re.findall(r"\((\d+)\)", text)}
    )


def _table5(raw_dir: Path) -> html_tables.HtmlTable:
    page = (raw_dir / TECHNICAL_NOTES).read_text(encoding="utf-8")
    return html_tables.find_table(page, r"Table 5\..*preliminary and final benchmark")


def table5_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    preliminary = html_tables.find_table(
        (raw_dir / PRELIMINARY_2026).read_text(encoding="utf-8"),
        r"Table 1\..*March 2026.*Preliminary Benchmark",
    )
    return pl.concat(
        [
            html_tables.table_raw_cells(
                _table5(raw_dir),
                source="bls",
                file=TECHNICAL_NOTES,
                table_key="table_5",
            ),
            html_tables.table_raw_cells(
                preliminary,
                source="bls",
                file=PRELIMINARY_2026,
                table_key="preliminary_2026_table_1",
            ),
        ]
    )


def preliminary_2026_values(
    raw_dir: Path = RAW_DIR,
) -> tuple[float, float, list[str]]:
    page = (raw_dir / PRELIMINARY_2026).read_text(encoding="utf-8")
    table = html_tables.find_table(
        page, r"Table 1\..*March 2026.*Preliminary Benchmark"
    )
    matches = [
        (row_index, row)
        for row_index, row in enumerate(table.rows)
        if row and row[0] == "Total nonfarm"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one 2026 total-nonfarm row, found {len(matches)}")
    row_index, row = matches[0]
    revision = parse_number(row[1])
    percent = parse_number(row[2])
    if revision is None or percent is None:
        raise ValueError("the 2026 total-nonfarm preliminary cells are not numeric")
    return (
        revision,
        percent,
        [
            cell_key(
                PRELIMINARY_2026,
                "preliminary_2026_table_1",
                str(row_index),
                column,
            )
            for column in ("1", "2")
        ],
    )


def benchmark_sector_raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": SECTOR_CELLS,
                    "table_key": "benchmark_sector_cells",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)


def _publication(calendar: pl.DataFrame, publication_id: str) -> dict:
    rows = calendar.filter(pl.col("publication_id") == publication_id)
    if rows.height != 1:
        raise ValueError(
            f"expected one publication {publication_id}, found {rows.height}"
        )
    return rows.row(0, named=True)


def _anchor_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    table = _table5(raw_dir)
    rows = []
    article = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    article_totals = {}
    for article_row_number, article_row in enumerate(
        article.iter_rows(named=True), start=2
    ):
        if article_row["sector"] == "00":
            article_totals[int(article_row["benchmark_year"])] = (
                parse_number(article_row["revision_text"]),
                article_row_number,
            )
    for row_index, cells in enumerate(table.rows):
        year = _year(cells[0]) if cells else None
        if year is None:
            continue
        final_percent = parse_number(cells[1]) if len(cells) > 2 else None
        final_revision = parse_number(cells[2]) if len(cells) > 2 else None
        preliminary_percent = parse_number(cells[3]) if len(cells) > 4 else None
        preliminary_revision = parse_number(cells[4]) if len(cells) > 4 else None
        for status, percent, revision, columns in (
            ("final", final_percent, final_revision, ("1", "2")),
            ("preliminary", preliminary_percent, preliminary_revision, ("3", "4")),
        ):
            if revision is None:
                continue
            publication = _publication(calendar, f"benchmark_{status}_{year}")
            article_entry = article_totals.get(year) if status == "final" else None
            article_revision = article_entry[0] if article_entry else None
            difference = (
                article_revision - revision if article_revision is not None else None
            )
            keys = [
                cell_key(TECHNICAL_NOTES, "table_5", str(row_index), column)
                for column in ("0", *columns)
            ]
            if article_entry:
                keys.append(
                    cell_key(
                        SECTOR_CELLS,
                        "benchmark_sector_cells",
                        str(article_entry[1]),
                        "revision_text",
                    )
                )
            keys.extend(publication["source_keys"])
            rows.append(
                {
                    "benchmark_year": year,
                    "benchmark_status": status,
                    "row_kind": "anchor",
                    "sector": "00",
                    "seasonal_status": "NSA",
                    "footnotes": sorted(
                        {
                            1,
                            *_footnotes(
                                cells[0],
                                *(cells[int(column)] for column in columns),
                            ),
                        }
                    ),
                    "revision_thousands": revision,
                    "percent_revision": percent,
                    "article_revision_thousands": article_revision,
                    "article_difference_thousands": difference,
                    "discrepancy_status": (
                        "unexplained" if year == 2025 and status == "final" else None
                    ),
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": TECHNICAL_NOTES,
                    "source_locator": f"Table 5, row {year}, {status}",
                    "source_keys": keys,
                    "transformation": (
                        "benchmark_article_comparison"
                        if article_entry
                        else "parse_thousands"
                    ),
                }
            )
    if not any(
        row["benchmark_year"] == 2026 and row["benchmark_status"] == "preliminary"
        for row in rows
    ):
        revision, percent, keys = preliminary_2026_values(raw_dir)
        publication = _publication(calendar, "benchmark_preliminary_2026")
        keys.extend(publication["source_keys"])
        rows.append(
            {
                "benchmark_year": 2026,
                "benchmark_status": "preliminary",
                "row_kind": "anchor",
                "sector": "00",
                "seasonal_status": "NSA",
                "footnotes": [],
                "revision_thousands": revision,
                "percent_revision": percent,
                "article_revision_thousands": None,
                "article_difference_thousands": None,
                "discrepancy_status": None,
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": PRELIMINARY_2026,
                "source_locator": "Table 1, Total nonfarm",
                "source_keys": keys,
                "transformation": "parse_thousands",
            }
        )
    return rows


def _sector_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    frame = pl.read_csv(raw_dir / SECTOR_CELLS, infer_schema_length=0)
    rows = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        if row["sector"] == "00":
            continue
        year = int(row["benchmark_year"])
        publication = _publication(calendar, f"benchmark_final_{year}")
        rows.append(
            {
                "benchmark_year": year,
                "benchmark_status": "final",
                "row_kind": "sector_contribution",
                "sector": row["sector"],
                "seasonal_status": "NSA",
                "footnotes": [],
                "revision_thousands": parse_number(row["revision_text"]),
                "percent_revision": parse_number(row["percent_text"]),
                "article_revision_thousands": None,
                "article_difference_thousands": None,
                "discrepancy_status": None,
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        SECTOR_CELLS,
                        "benchmark_sector_cells",
                        str(row_number),
                        column,
                    )
                    for column in ("revision_text", "percent_text")
                ]
                + publication["source_keys"],
                "transformation": "parse_thousands",
            }
        )
    return rows


def build_benchmarks(calendar: pl.DataFrame, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    rows = _anchor_rows(calendar, raw_dir) + _sector_rows(calendar, raw_dir)
    frame = pl.DataFrame(rows, schema=BENCHMARK_SCHEMA).sort(
        "benchmark_year", "benchmark_status", "row_kind", "sector"
    )
    keys = ["benchmark_year", "benchmark_status", "row_kind", "sector"]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate benchmark rows")
    return frame
```

- [x] **Step 7: Write the reconstruction builder**

Create `src/ces_revisions/annual/reconstructions.py`:

```python
"""Documented benchmark scope changes and reconstructions."""

from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

EVENTS = "manual/reconstruction-events.csv"


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / EVENTS, infer_schema_length=0)
    records = []
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        for column in frame.columns:
            records.append(
                {
                    "source": "manual_transcription",
                    "file": EVENTS,
                    "table_key": "reconstruction_events",
                    "row_key": str(row_number),
                    "column_key": column,
                    "text": row[column] or "",
                }
            )
    return raw_frame(records)


def _optional_date(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


def build_reconstruction_events(
    calendar: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    source = pl.read_csv(raw_dir / EVENTS, infer_schema_length=0)
    publications = {
        row["benchmark_year"]: row
        for row in calendar.filter(
            pl.col("publication_kind") == "benchmark_final"
        ).iter_rows(named=True)
    }
    rows = []
    for row_number, row in enumerate(source.iter_rows(named=True), start=2):
        year = int(row["benchmark_year"])
        publication = publications[year]
        rows.append(
            {
                "event_id": row["event_id"],
                "benchmark_year": year,
                "event_type": row["event_type"],
                "sectors": row["sectors"].split(";") if row["sectors"] else [],
                "series_codes": (
                    row["series_codes"].split(";") if row["series_codes"] else []
                ),
                "reference_start": _optional_date(row["reference_start"]),
                "reference_end": _optional_date(row["reference_end"]),
                "effect_thousands": parse_number(row["effect_thousands_text"]),
                "footnote": int(row["footnote"]),
                "description": row["description"],
                "publication_date": publication["publication_date"],
                "observable_at": publication["observable_at"],
                "source_file": row["source_file"],
                "source_locator": row["source_locator"],
                "source_keys": [
                    cell_key(
                        EVENTS,
                        "reconstruction_events",
                        str(row_number),
                        column,
                    )
                    for column in (
                        "event_type",
                        "effect_thousands_text",
                        "description",
                    )
                ]
                + publication["source_keys"],
                "transformation": "parse_thousands",
            }
        )
    return pl.DataFrame(rows).sort("benchmark_year", "event_id")
```

- [x] **Step 8: Run the focused tests and repair only source-layout differences**

Run:

```bash
uv run pytest tests/test_benchmarks.py tests/test_reconstructions.py -q
uv run ruff format src/ces_revisions/annual tests/test_benchmarks.py tests/test_reconstructions.py
uv run ruff check src/ces_revisions/annual tests/test_benchmarks.py tests/test_reconstructions.py
```

Expected: all focused tests pass. If the fetched HTML's row/column structure differs from the fixture, adapt `_anchor_rows()` to the observed header labels and add the observed excerpt to the fixture; do not change any expected year, value, or discrepancy.

- [x] **Step 9: Refresh the source manifest and run the fast tier**

Run:

```bash
uv run python scripts/annual_sources.py manifest
uv run pytest -m "not slow and not network" -q
```

Expected: the manifest includes both transcription CSVs and the full fast tier passes.

- [x] **Step 10: Commit**

```bash
git add src/ces_revisions/annual/benchmarks.py src/ces_revisions/annual/reconstructions.py tests/test_benchmarks.py tests/test_reconstructions.py tests/fixtures/annual data/annual/raw/manual/benchmark-sector-cells.csv data/annual/raw/manual/reconstruction-events.csv data/annual/raw/manifest.csv
git commit -m "Build dated benchmark and reconstruction tables"
```

---

### Task 4: Build Monthly and Forecast-versus-Realized Birth–Death Rows

**Files:**

- Create: `src/ces_revisions/annual/birth_death.py`
- Create: `tests/test_birth_death.py`
- Create: `data/annual/raw/manual/birth-death-annual-cells.csv`
- Create: `data/annual/raw/manual/birth-death-rules.csv`
- Modify: `data/annual/raw/manifest.csv`

**Interfaces:**

- Consumes: `html_tables.parse_tables(page)`, `raw.parse_number(text)`, the Task 2 publication calendar, and Stage 3's release index.
- Produces: `birth_death.raw_values(raw_dir) -> pl.DataFrame`, `birth_death.build_birth_death(calendar, release_index, raw_dir) -> pl.DataFrame`, and `BIRTH_DEATH_SCHEMA`. The table key is `(series_kind, benchmark_year, reference_month, frequency, sector, value_kind)`.

- [x] **Step 1: Create the two audited manual inputs**

Create `data/annual/raw/manual/birth-death-annual-cells.csv` with this exact header:

```csv
benchmark_year,period_label,reference_month,frequency,realized_text,forecast_text,difference_text,source_file,source_locator
```

Transcribe the forecast-versus-actual table from each benchmark article for benchmark years 2009–2025. Use 13 rows per year: the 12 printed months from April of `benchmark_year - 1` through March of `benchmark_year`, followed by the printed `Total` row with a blank `reference_month` and `frequency=annual`. Use `frequency=monthly` for month rows. Preserve commas and minus signs in the three text columns. `difference_text` is BLS's “Difference” row, whose sign is **realized minus forecast**. The source locators are the printed exhibit/table name and PDF page; do not use the surrounding prose total.

Create `data/annual/raw/manual/birth-death-rules.csv` with this exact content:

```csv
rule_id,value_text,description,source_file,source_locator
government-structural-zero,0,CES net birth-death applies only to private-industry employment; government therefore contributes zero,bls/cesbdqa.htm,FAQ scope statement and industry tables
```

After transcription, run this independent arithmetic check:

```bash
uv run python -c 'import polars as pl; p="data/annual/raw/manual/birth-death-annual-cells.csv"; f=pl.read_csv(p, infer_schema_length=0).filter(pl.col("frequency")=="monthly").with_columns(*(pl.col(c).str.replace_all(",", "").cast(pl.Int64).alias(c[:-5]) for c in ("realized_text", "forecast_text", "difference_text"))); assert f.filter(pl.col("difference") != pl.col("realized") - pl.col("forecast")).is_empty(); totals=f.group_by("benchmark_year").agg(pl.col("realized").sum(), pl.col("forecast").sum(), pl.col("difference").sum()); printed=pl.read_csv(p, infer_schema_length=0).filter(pl.col("frequency")=="annual").with_columns(*(pl.col(c).str.replace_all(",", "").cast(pl.Int64).alias(c[:-5]) for c in ("realized_text", "forecast_text", "difference_text"))); assert totals.join(printed, on="benchmark_year", suffix="_printed").select(*(pl.col(c)==pl.col(f"{c}_printed") for c in ("realized", "forecast", "difference"))).to_numpy().all()'
```

Expected: exit 0. A mismatch stops the task and is resolved against the printed PDF before writing parser code.

- [x] **Step 2: Write the failing birth–death tests**

Create `tests/test_birth_death.py`:

```python
"""Monthly and forecast-versus-realized CES net birth-death values."""

from datetime import date

import polars as pl

from ces_revisions.annual import birth_death, publications
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    index = build_release_index()
    calendar = publications.build_publication_calendar(index)
    return birth_death.build_birth_death(calendar, index)


def test_modern_schedule_vintages_are_complete():
    frame = built().filter(pl.col("series_kind") == "published_schedule")
    initial = frame.filter(
        (pl.col("schedule_kind") == "preliminary") & (pl.col("sector") == "00")
    )
    revised = frame.filter(
        (pl.col("schedule_kind") == "post_benchmark") & (pl.col("sector") == "00")
    )
    assert initial["reference_month"].dt.year().unique().sort().to_list() == list(
        range(2004, 2027)
    )
    assert revised["reference_month"].dt.year().unique().sort().to_list() == list(
        range(2003, 2026)
    )
    assert revised.group_by(pl.col("reference_month").dt.year()).len()[
        "len"
    ].unique().to_list() == [9]


def test_monthly_total_is_the_sum_of_private_supersectors():
    frame = built().filter(
        (pl.col("series_kind") == "published_schedule")
        & pl.col("sector").is_in(
            ["00", "10", "20", "30", "40", "50", "55", "60", "65", "70", "80"]
        )
    )
    totals = frame.filter(pl.col("sector") == "00").select(
        "schedule_kind", "reference_month", published=pl.col("value_thousands")
    )
    summed = (
        frame.filter(pl.col("sector") != "00")
        .group_by("schedule_kind", "reference_month")
        .agg(calculated=pl.col("value_thousands").sum())
    )
    checked = totals.join(summed, on=["schedule_kind", "reference_month"])
    assert (checked["published"] == checked["calculated"]).all()


def test_preliminary_months_use_their_first_employment_situation_release():
    index = build_release_index()
    frame = built()
    row = frame.filter(
        (pl.col("schedule_kind") == "preliminary")
        & (pl.col("reference_month") == date(2025, 7, 1))
        & (pl.col("sector") == "00")
    ).row(0, named=True)
    release = index.filter(pl.col("reference_month") == date(2025, 7, 1)).row(
        0, named=True
    )
    assert row["publication_date"] == release["published_date"]
    assert row["observable_at"] == release["observable_at"]


def test_post_benchmark_months_share_the_final_benchmark_publication():
    frame = built().filter(
        (pl.col("schedule_kind") == "post_benchmark")
        & (pl.col("benchmark_year") == 2025)
    )
    assert frame["publication_date"].unique().to_list() == [date(2026, 2, 11)]


def test_forecast_realized_rows_cover_2009_through_2025_and_balance():
    frame = built().filter(
        (pl.col("series_kind") == "forecast_vs_realized") & (pl.col("sector") == "00")
    )
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2009, 2026))
    wide = frame.pivot(
        on="value_kind",
        index=["benchmark_year", "reference_month", "frequency"],
        values="value_thousands",
    )
    assert (
        wide["realized_minus_forecast"] == wide["realized"] - wide["forecast"]
    ).all()


def test_government_is_a_structural_zero_for_every_birth_death_key():
    frame = built()
    key = [
        "series_kind",
        "benchmark_year",
        "reference_month",
        "period_start",
        "period_end",
        "frequency",
        "schedule_kind",
        "value_kind",
    ]
    total_keys = frame.filter(pl.col("sector") == "00").select(key).unique()
    government = frame.filter(pl.col("sector") == "90")
    assert government["value_thousands"].eq(0.0).all()
    assert government["transformation"].unique().to_list() == [
        "government_structural_zero"
    ]
    assert government.select(key).unique().sort(key).equals(total_keys.sort(key))


def test_every_birth_death_row_is_nsa_dated_and_in_thousands():
    frame = built()
    assert frame["seasonal_status"].unique().to_list() == ["NSA"]
    assert frame["unit"].unique().to_list() == ["thousands"]
    assert frame["observable_at"].is_not_null().all()
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_birth_death.py -q`

Expected: collection fails with `ImportError: cannot import name 'birth_death'`.

- [x] **Step 4: Write the schedule parser and schema**

Create `src/ces_revisions/annual/birth_death.py` with the following constants, schema, and parsing helpers:

```python
"""Audited CES net birth-death schedules and forecast-versus-realized values."""

import re
from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

HISTORY = "bls/cesbd-history.htm"
CURRENT = "bls/cesbd.htm"
ANNUAL_CELLS = "manual/birth-death-annual-cells.csv"
RULES = "manual/birth-death-rules.csv"
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
SECTOR_BY_TITLE = {
    "mining and logging": "10",
    "natural resources and mining": "10",
    "construction": "20",
    "manufacturing": "30",
    "trade, transportation, and utilities": "40",
    "information": "50",
    "financial activities": "55",
    "professional and business services": "60",
    "private education and health services": "65",
    "education and health services": "65",
    "leisure and hospitality": "70",
    "other services": "80",
    "total nonfarm birth-death forecast": "00",
    "total private net birth-death forecast": "00",
}
BIRTH_DEATH_SCHEMA = {
    "series_kind": pl.String,
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "frequency": pl.String,
    "schedule_kind": pl.String,
    "sector": pl.String,
    "seasonal_status": pl.String,
    "value_kind": pl.String,
    "value_thousands": pl.Float64,
    "unit": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _normalized(text: str) -> str:
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _schedule_kind(header: tuple[str, ...]) -> str | None:
    label = _normalized(header[0]) if header else ""
    if label.startswith("preliminary"):
        return "preliminary"
    if label.startswith("post-benchmark"):
        return "post_benchmark"
    return None


def _table_year(table: html_tables.HtmlTable) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", table.caption)
    return int(match.group()) if match else None


def _selected_tables(raw_dir: Path):
    for file in (HISTORY, CURRENT):
        page = (raw_dir / file).read_text(encoding="utf-8")
        for table_number, table in enumerate(html_tables.parse_tables(page)):
            year = _table_year(table)
            header = next((row for row in table.rows if set(row) & set(MONTHS)), None)
            if year is None or header is None:
                continue
            kind = (
                "preliminary"
                if file == CURRENT and year == 2026
                else _schedule_kind(header)
            )
            if kind is None:
                continue
            if file == HISTORY and (
                (kind == "preliminary" and 2004 <= year <= 2025)
                or (kind == "post_benchmark" and 2003 <= year <= 2025)
            ):
                yield file, table_number, year, kind, table, header
            elif file == CURRENT and year == 2026 and kind == "preliminary":
                yield file, table_number, year, kind, table, header


def _sector(text: str) -> str | None:
    return SECTOR_BY_TITLE.get(_normalized(text))


def _monthly_cells(raw_dir: Path):
    for file, table_number, year, kind, table, header in _selected_tables(raw_dir):
        header_index = table.rows.index(header)
        month_columns = {
            column: MONTHS[label]
            for column, label in enumerate(header)
            if label in MONTHS
        }
        table_key = f"birth_death_{year}_{kind}"
        for row_index, row in enumerate(
            table.rows[header_index + 1 :], start=header_index + 1
        ):
            title_column = 1 if row and re.fullmatch(r"\d{2}-\d{6}.*", row[0]) else 0
            sector = _sector(row[title_column]) if len(row) > title_column else None
            if sector is None:
                continue
            for column, month in month_columns.items():
                if column >= len(row) or parse_number(row[column]) is None:
                    continue
                yield {
                    "file": file,
                    "table_number": table_number,
                    "table_key": table_key,
                    "row_index": row_index,
                    "column": column,
                    "year": year,
                    "schedule_kind": kind,
                    "reference_month": date(year, month, 1),
                    "sector": sector,
                    "text": row[column],
                }


def _csv_raw_values(path: Path, file: str, table_key: str) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0)
    records = [
        {
            "source": "manual_transcription",
            "file": file,
            "table_key": table_key,
            "row_key": str(row_number),
            "column_key": column,
            "text": row[column] or "",
        }
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2)
        for column in frame.columns
    ]
    return raw_frame(records)


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    schedule_records = [
        {
            "source": "bls",
            "file": cell["file"],
            "table_key": cell["table_key"],
            "row_key": str(cell["row_index"]),
            "column_key": str(cell["column"]),
            "text": cell["text"],
        }
        for cell in _monthly_cells(raw_dir)
    ]
    return pl.concat(
        [
            raw_frame(schedule_records),
            _csv_raw_values(raw_dir / ANNUAL_CELLS, ANNUAL_CELLS, "birth_death_annual"),
            _csv_raw_values(raw_dir / RULES, RULES, "birth_death_rules"),
        ]
    ).sort("cell_key")
```

The source selection is deliberate: the historical page owns complete preliminary schedules for 2004–2025 and post-benchmark schedules for 2003–2025; the live page contributes only the non-null 2026 preliminary months. It excludes the detailed `41`–`44` rows so they cannot be added again beneath sector `40`.

- [x] **Step 5: Write the dated builders and identity checks**

Append to `src/ces_revisions/annual/birth_death.py`:

```python
def _one(frame: pl.DataFrame, **filters) -> dict:
    expression = pl.lit(True)
    for column, value in filters.items():
        expression &= pl.col(column) == value
    rows = frame.filter(expression)
    if rows.height != 1:
        raise ValueError(f"expected one row for {filters}, found {rows.height}")
    return rows.row(0, named=True)


def _schedule_rows(
    calendar: pl.DataFrame, release_index: pl.DataFrame, raw_dir: Path
) -> list[dict]:
    rows = []
    for cell in _monthly_cells(raw_dir):
        year = cell["year"]
        month = cell["reference_month"]
        if cell["schedule_kind"] == "preliminary":
            publication = _one(release_index, reference_month=month)
            publication_date = publication["published_date"]
            observable_at = publication["observable_at"]
            benchmark_year = year - 1
            publication_keys = [f"stage3::release_index::{month:%Y-%m}"]
        else:
            publication = _one(
                calendar,
                publication_id=f"benchmark_final_{year}",
            )
            publication_date = publication["publication_date"]
            observable_at = publication["observable_at"]
            benchmark_year = year
            publication_keys = publication["source_keys"]
        rows.append(
            {
                "series_kind": "published_schedule",
                "benchmark_year": benchmark_year,
                "reference_month": month,
                "period_start": month,
                "period_end": month,
                "frequency": "monthly",
                "schedule_kind": cell["schedule_kind"],
                "sector": cell["sector"],
                "seasonal_status": "NSA",
                "value_kind": (
                    "initial_forecast"
                    if cell["schedule_kind"] == "preliminary"
                    else "revised_forecast"
                ),
                "value_thousands": parse_number(cell["text"]),
                "unit": "thousands",
                "publication_date": publication_date,
                "observable_at": observable_at,
                "source_file": cell["file"],
                "source_locator": (
                    f"table {cell['table_number'] + 1}, "
                    f"{cell['schedule_kind']}, {month:%Y-%m}"
                ),
                "source_keys": [
                    cell_key(
                        cell["file"],
                        cell["table_key"],
                        str(cell["row_index"]),
                        str(cell["column"]),
                    )
                ]
                + publication_keys,
                "transformation": "parse_thousands",
            }
        )
    return rows


def _annual_rows(calendar: pl.DataFrame, raw_dir: Path) -> list[dict]:
    source = pl.read_csv(raw_dir / ANNUAL_CELLS, infer_schema_length=0)
    rows = []
    for row_number, row in enumerate(source.iter_rows(named=True), start=2):
        year = int(row["benchmark_year"])
        publication = _one(calendar, publication_id=f"benchmark_final_{year}")
        reference_month = (
            date.fromisoformat(row["reference_month"])
            if row["reference_month"]
            else None
        )
        period_start = reference_month or date(year - 1, 4, 1)
        period_end = reference_month or date(year, 3, 1)
        realized = parse_number(row["realized_text"])
        forecast = parse_number(row["forecast_text"])
        printed_difference = parse_number(row["difference_text"])
        if None in (realized, forecast, printed_difference):
            raise ValueError(
                f"missing annual birth-death value on CSV row {row_number}"
            )
        difference = realized - forecast
        if difference != printed_difference:
            raise ValueError(
                f"birth-death identity fails on CSV row {row_number}: "
                f"{realized} - {forecast} != {printed_difference}"
            )
        for value_kind, value, columns, transformation in (
            ("realized", realized, ("realized_text",), "parse_thousands"),
            ("forecast", forecast, ("forecast_text",), "parse_thousands"),
            (
                "realized_minus_forecast",
                difference,
                ("realized_text", "forecast_text", "difference_text"),
                "realized_minus_forecast",
            ),
        ):
            rows.append(
                {
                    "series_kind": "forecast_vs_realized",
                    "benchmark_year": year,
                    "reference_month": reference_month,
                    "period_start": period_start,
                    "period_end": period_end,
                    "frequency": row["frequency"],
                    "schedule_kind": None,
                    "sector": "00",
                    "seasonal_status": "NSA",
                    "value_kind": value_kind,
                    "value_thousands": value,
                    "unit": "thousands",
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": row["source_file"],
                    "source_locator": row["source_locator"],
                    "source_keys": [
                        cell_key(
                            ANNUAL_CELLS,
                            "birth_death_annual",
                            str(row_number),
                            column,
                        )
                        for column in columns
                    ]
                    + publication["source_keys"],
                    "transformation": transformation,
                }
            )
    return rows


def _assert_private_totals(frame: pl.DataFrame) -> None:
    schedules = frame.filter(
        (pl.col("series_kind") == "published_schedule") & (pl.col("sector") != "90")
    )
    totals = schedules.filter(pl.col("sector") == "00").select(
        "schedule_kind", "reference_month", total=pl.col("value_thousands")
    )
    calculated = (
        schedules.filter(pl.col("sector") != "00")
        .group_by("schedule_kind", "reference_month")
        .agg(total_from_sectors=pl.col("value_thousands").sum())
    )
    failures = totals.join(calculated, on=["schedule_kind", "reference_month"]).filter(
        pl.col("total") != pl.col("total_from_sectors")
    )
    if failures.height:
        raise ValueError(f"birth-death sector totals fail for {failures.rows()}")


def _government_rows(frame: pl.DataFrame) -> pl.DataFrame:
    rule_key = cell_key(RULES, "birth_death_rules", "2", "value_text")
    return frame.filter(pl.col("sector") == "00").with_columns(
        sector=pl.lit("90"),
        value_thousands=pl.lit(0.0),
        source_file=pl.lit(RULES),
        source_locator=pl.lit("government-structural-zero"),
        source_keys=pl.concat_list(
            pl.lit([rule_key], dtype=pl.List(pl.String)), pl.col("source_keys")
        ),
        transformation=pl.lit("government_structural_zero"),
    )


def build_birth_death(
    calendar: pl.DataFrame,
    release_index: pl.DataFrame,
    raw_dir: Path = RAW_DIR,
) -> pl.DataFrame:
    rows = _schedule_rows(calendar, release_index, raw_dir)
    rows.extend(_annual_rows(calendar, raw_dir))
    frame = pl.DataFrame(rows, schema=BIRTH_DEATH_SCHEMA)
    _assert_private_totals(frame)
    frame = pl.concat([frame, _government_rows(frame)]).sort(
        "series_kind",
        "benchmark_year",
        "reference_month",
        "frequency",
        "sector",
        "value_kind",
    )
    keys = [
        "series_kind",
        "benchmark_year",
        "reference_month",
        "frequency",
        "sector",
        "value_kind",
    ]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate birth-death rows")
    return frame
```

The `preliminary` label here is BLS's label for the original monthly schedule; it is unrelated to the preliminary annual benchmark and remains `series_kind=published_schedule`. The schedule value is dated to the Employment Situation that first used that reference month. A `post_benchmark` value is dated to the final benchmark release that recalculated the April–December period.

- [x] **Step 6: Run the focused tests and inspect the source identities**

Run:

```bash
uv run pytest tests/test_birth_death.py -q
uv run ruff format src/ces_revisions/annual/birth_death.py tests/test_birth_death.py
uv run ruff check src/ces_revisions/annual/birth_death.py tests/test_birth_death.py
```

Expected: all tests pass. If a historical caption or title fails to map, add its literal title to `SECTOR_BY_TITLE` or the caption fixture; do not broaden matching until detailed `41`–`44` rows can leak into the supersector sum.

- [x] **Step 7: Refresh the manifest and run the fast tier**

Run:

```bash
uv run python scripts/annual_sources.py manifest
uv run pytest -m "not slow and not network" -q
```

Expected: `manifest.csv` hashes both manual files and all tests pass.

- [x] **Step 8: Commit**

```bash
git add src/ces_revisions/annual/birth_death.py tests/test_birth_death.py data/annual/raw/manual/birth-death-annual-cells.csv data/annual/raw/manual/birth-death-rules.csv data/annual/raw/manifest.csv
git commit -m "Build dated CES birth-death tables"
```

---

### Task 5: Build the National QCEW Revision Sequence and Precision Proxy

**Files:**

- Create: `src/ces_revisions/annual/qcew.py`
- Create: `tests/test_qcew.py`
- Modify: `data/annual/raw/manifest.csv`

**Interfaces:**

- Consumes: `bls/qcew-revisions.csv`, Task 2's QCEW publication rows, `raw.cell_key()`, and `raw.parse_number()`.
- Produces: `qcew.raw_values(raw_dir) -> pl.DataFrame`, `qcew.build_qcew_revisions(calendar, raw_dir) -> pl.DataFrame`, `qcew.build_qcew_precision(revisions) -> pl.DataFrame`, `QCEW_REVISION_SCHEMA`, and `QCEW_PRECISION_SCHEMA`.

- [x] **Step 1: Write the failing QCEW tests**

Create `tests/test_qcew.py`:

```python
"""The public national QCEW revision sequence and its March precision proxy."""

from math import sqrt

import polars as pl

from ces_revisions.annual import publications, qcew
from ces_revisions.vintages.release_index import build_release_index


def revisions() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return qcew.build_qcew_revisions(calendar)


def test_only_2017_plus_national_monthly_employment_is_ingested():
    frame = revisions()
    assert frame["area"].unique().to_list() == ["United States"]
    assert frame["reference_month"].min().year == 2017
    assert frame["field"].str.ends_with(" Employment").all()
    assert frame["reference_month"].n_unique() < 12 * 20


def test_completed_quarters_have_the_documented_release_count():
    frame = revisions().filter(pl.col("sequence_status") == "complete")
    counts = frame.group_by("qcew_year", "qcew_quarter", "field").agg(
        releases=pl.col("release_order").n_unique(),
        last_order=pl.col("release_order").max(),
    )
    expected = pl.DataFrame(
        {"qcew_quarter": [1, 2, 3, 4], "expected": [5, 4, 3, 2]},
        schema={"qcew_quarter": pl.Int8, "expected": pl.UInt32},
    )
    checked = counts.join(expected, on="qcew_quarter")
    assert (checked["releases"] == checked["expected"]).all()
    assert (checked["last_order"] == checked["expected"] - 1).all()


def test_final_value_is_an_alias_not_a_sixth_release():
    source = qcew.read_source()
    q1_complete = source.filter(
        (pl.col("Quarter") == 1)
        & pl.col("Final Value").cast(pl.String).str.contains(r"^-?\d")
    )
    assert (
        q1_complete["Final Value"].cast(pl.String)
        == q1_complete["Fourth Revised Value"].cast(pl.String)
    ).all()
    assert revisions().filter(pl.col("qcew_quarter") == 1)["release_order"].max() == 4


def test_values_and_successive_revisions_remain_in_jobs():
    frame = revisions()
    assert frame["unit"].unique().to_list() == ["jobs"]
    assert (frame["employment_jobs"].drop_nulls() % 1 == 0).all()
    group = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("field") == "March Employment")
    ).sort("release_order")
    expected = group["employment_jobs"].diff()
    assert group["revision_jobs"].to_list() == expected.to_list()


def test_each_revision_uses_the_qcew_publication_clock():
    calendar = publications.build_publication_calendar(build_release_index())
    frame = revisions()
    row = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("release_order") == 4)
    ).row(0, named=True)
    expected = calendar.filter(
        (pl.col("publication_kind") == "qcew")
        & (pl.col("qcew_year") == 2017)
        & (pl.col("qcew_quarter") == 1)
        & (pl.col("release_order") == 4)
    ).row(0, named=True)
    assert row["publication_date"] == expected["publication_date"]
    assert row["observable_at"] == expected["observable_at"]


def test_qcew_calendar_has_exactly_the_releases_present_in_the_source():
    calendar = publications.build_publication_calendar(build_release_index())
    keys = ["qcew_year", "qcew_quarter", "release_order"]
    expected = revisions().select(keys).unique().sort(keys)
    actual = (
        calendar.filter(pl.col("publication_kind") == "qcew")
        .select(keys)
        .unique()
        .sort(keys)
    )
    assert actual.equals(expected)
    qcew_dates = calendar.filter(pl.col("publication_kind") == "qcew")
    for group in qcew_dates.partition_by("qcew_year", "qcew_quarter"):
        assert group.sort("release_order")["publication_date"].is_sorted()


def test_march_precision_is_rms_of_four_revision_increments_in_thousands():
    frame = revisions()
    precision = qcew.build_qcew_precision(frame)
    march = frame.filter(
        (pl.col("qcew_year") == 2017)
        & (pl.col("field") == "March Employment")
        & (pl.col("release_order") > 0)
    )
    expected = sqrt(sum((value / 1_000) ** 2 for value in march["revision_jobs"]) / 4)
    row = precision.filter(pl.col("benchmark_year") == 2017).row(0, named=True)
    assert row["sequence_status"] == "complete"
    assert row["release_count"] == 5
    assert row["revision_precision_thousands"] == expected


def test_latest_incomplete_march_sequence_is_right_censored():
    precision = qcew.build_qcew_precision(revisions())
    incomplete = precision.filter(pl.col("release_count") < 5)
    if incomplete.height:
        assert incomplete["sequence_status"].unique().to_list() == ["right_censored"]
        assert incomplete["revision_precision_thousands"].is_null().all()
    assert precision["observable_at"].is_not_null().all()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_qcew.py -q`

Expected: collection fails with `ImportError: cannot import name 'qcew'`.

- [x] **Step 3: Write the source reader and long revision sequence**

> **Deviation (as-of correction):** QCEW quarter-end initial employment uses the news
> release, while other initial-month values and every revised value use the full-data
> release. This preserves the actual first-observable date when the two products diverge.

Create `src/ces_revisions/annual/qcew.py`:

```python
"""National aggregate QCEW revisions and the March precision proxy."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl

from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

SOURCE = "bls/qcew-revisions.csv"
EXPECTED_COLUMNS = [
    "Year",
    "Quarter",
    "Area",
    "Field",
    "Initial Value",
    "First Revised Value",
    "Second Revised Value",
    "Third Revised Value",
    "Fourth Revised Value",
    "Final Value",
]
RELEASE_COLUMNS = [
    "Initial Value",
    "First Revised Value",
    "Second Revised Value",
    "Third Revised Value",
    "Fourth Revised Value",
]
RELEASE_COUNT = {1: 5, 2: 4, 3: 3, 4: 2}
MONTH = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
QCEW_REVISION_SCHEMA = {
    "qcew_year": pl.Int32,
    "qcew_quarter": pl.Int8,
    "area": pl.String,
    "field": pl.String,
    "reference_month": pl.Date,
    "release_order": pl.Int8,
    "release_label": pl.String,
    "employment_jobs": pl.Int64,
    "revision_jobs": pl.Int64,
    "unit": pl.String,
    "seasonal_status": pl.String,
    "sequence_status": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_file": pl.String,
    "source_locator": pl.String,
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}
QCEW_PRECISION_SCHEMA = {
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "release_count": pl.Int8,
    "sequence_status": pl.String,
    "revision_precision_thousands": pl.Float64,
    "unit": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def read_source(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = pl.read_csv(raw_dir / SOURCE, infer_schema_length=0).with_row_index(
        "source_row", offset=2
    )
    if frame.columns[1:] != EXPECTED_COLUMNS:
        raise ValueError(f"QCEW revision header changed: {frame.columns[1:]!r}")
    return frame.with_columns(
        pl.col("Year").cast(pl.Int32),
        pl.col("Quarter").cast(pl.Int8),
    ).filter(
        (pl.col("Area") == "United States")
        & (pl.col("Year").cast(pl.Int32) >= 2017)
        & pl.col("Field").str.ends_with(" Employment")
    )


def _available_values(row: dict) -> list[tuple[int, str, int]]:
    quarter = int(row["Quarter"])
    available = []
    missing_seen = False
    for order, column in enumerate(RELEASE_COLUMNS[: RELEASE_COUNT[quarter]]):
        value = parse_number(row[column] or "")
        if value is None:
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError(
                f"QCEW row {row['source_row']} has a revision after a missing release"
            )
        if not value.is_integer():
            raise ValueError(f"QCEW row {row['source_row']} is not whole jobs")
        available.append((order, column, int(value)))
    final = parse_number(row["Final Value"] or "")
    if final is not None and (not available or int(final) != available[-1][2]):
        raise ValueError(
            f"QCEW Final Value is not the last applicable revision on row "
            f"{row['source_row']}"
        )
    return available


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = read_source(raw_dir)
    records = [
        {
            "source": "bls_qcew_revisions",
            "file": SOURCE,
            "table_key": "qcew_revisions",
            "row_key": str(row["source_row"]),
            "column_key": column,
            "text": "" if row[column] is None else str(row[column]),
        }
        for row in frame.iter_rows(named=True)
        for column in EXPECTED_COLUMNS
    ]
    return raw_frame(records)


def _publication(calendar: pl.DataFrame, year: int, quarter: int, order: int) -> dict:
    rows = calendar.filter(
        (pl.col("publication_kind") == "qcew")
        & (pl.col("qcew_year") == year)
        & (pl.col("qcew_quarter") == quarter)
        & (pl.col("release_order") == order)
    )
    if rows.height != 1:
        raise ValueError(
            f"expected one QCEW publication for {(year, quarter, order)}, "
            f"found {rows.height}"
        )
    return rows.row(0, named=True)


def build_qcew_revisions(
    calendar: pl.DataFrame, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    rows = []
    for source in read_source(raw_dir).iter_rows(named=True):
        year = int(source["Year"])
        quarter = int(source["Quarter"])
        month_name = source["Field"].removesuffix(" Employment")
        month = MONTH.get(month_name)
        if month is None or (month - 1) // 3 + 1 != quarter:
            raise ValueError(
                f"QCEW row {source['source_row']} has an invalid field/quarter pair"
            )
        available = _available_values(source)
        complete = len(available) == RELEASE_COUNT[quarter]
        previous = None
        previous_column = None
        for order, column, value in available:
            publication = _publication(calendar, year, quarter, order)
            keys = [
                cell_key(
                    SOURCE,
                    "qcew_revisions",
                    str(source["source_row"]),
                    column,
                )
            ]
            revision = None
            transformation = "parse_published_number"
            if previous is not None:
                revision = value - previous
                transformation = "successive_difference"
                keys.append(
                    cell_key(
                        SOURCE,
                        "qcew_revisions",
                        str(source["source_row"]),
                        previous_column,
                    )
                )
            keys.extend(publication["source_keys"])
            rows.append(
                {
                    "qcew_year": year,
                    "qcew_quarter": quarter,
                    "area": "United States",
                    "field": source["Field"],
                    "reference_month": date(year, month, 1),
                    "release_order": order,
                    "release_label": column,
                    "employment_jobs": value,
                    "revision_jobs": revision,
                    "unit": "jobs",
                    "seasonal_status": "NSA",
                    "sequence_status": "complete" if complete else "right_censored",
                    "publication_date": publication["publication_date"],
                    "observable_at": publication["observable_at"],
                    "source_file": SOURCE,
                    "source_locator": (f"CSV row {source['source_row']}, {column}"),
                    "source_keys": keys,
                    "transformation": transformation,
                }
            )
            previous, previous_column = value, column
    frame = pl.DataFrame(rows, schema=QCEW_REVISION_SCHEMA).sort(
        "reference_month", "release_order"
    )
    keys = ["reference_month", "release_order"]
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError("duplicate national QCEW revision rows")
    return frame
```

- [x] **Step 4: Write the complete/right-censored March precision builder**

Append to `src/ces_revisions/annual/qcew.py`:

```python
def build_qcew_precision(revisions: pl.DataFrame) -> pl.DataFrame:
    rows = []
    march = revisions.filter(pl.col("field") == "March Employment")
    for year in sorted(march["qcew_year"].unique()):
        group = march.filter(pl.col("qcew_year") == year).sort("release_order")
        complete = group.height == RELEASE_COUNT[1]
        increments = group.filter(pl.col("release_order") > 0)["revision_jobs"]
        precision = None
        if complete:
            precision = math.sqrt(sum((value / 1_000) ** 2 for value in increments) / 4)
        latest = group.tail(1).row(0, named=True)
        rows.append(
            {
                "benchmark_year": year,
                "reference_month": date(year, 3, 1),
                "release_count": group.height,
                "sequence_status": "complete" if complete else "right_censored",
                "revision_precision_thousands": precision,
                "unit": "thousands",
                "publication_date": latest["publication_date"],
                "observable_at": latest["observable_at"],
                "source_keys": group["source_keys"].explode().unique().sort().to_list(),
                "transformation": "rms_qcew_revision",
            }
        )
    return pl.DataFrame(rows, schema=QCEW_PRECISION_SCHEMA).sort("benchmark_year")
```

Do not use `Final Value` as another observation. It is retained in `raw_values()` so the alias check is reproducible, but the long table has only orders 0–4/0–3/0–2/0–1 for quarters 1–4.

- [x] **Step 5: Run the focused tests and inspect the national slice**

Run:

```bash
uv run pytest tests/test_qcew.py -q
uv run python -c 'from ces_revisions.annual import publications,qcew; from ces_revisions.vintages.release_index import build_release_index; i=build_release_index(); f=qcew.build_qcew_revisions(publications.build_publication_calendar(i)); print(f.select("qcew_year","qcew_quarter","reference_month","release_order","employment_jobs","revision_jobs").head(8))'
uv run ruff format src/ces_revisions/annual/qcew.py tests/test_qcew.py
uv run ruff check src/ces_revisions/annual/qcew.py tests/test_qcew.py
```

Expected: all tests pass; the inspection prints only United States employment rows, with order 0 revision null and later revisions in whole jobs.

- [x] **Step 6: Refresh the manifest and run the fast tier**

Run:

```bash
uv run python scripts/annual_sources.py manifest
uv run pytest -m "not slow and not network" -q
```

Expected: the QCEW source hash remains pinned and the full fast tier passes.

- [x] **Step 7: Commit**

```bash
git add src/ces_revisions/annual/qcew.py tests/test_qcew.py data/annual/raw/manifest.csv
git commit -m "Build the national QCEW revision sequence"
```

---

### Task 6: Build the Annual Coverage-Proxy and RSE Panel

**Files:**

- Create: `src/ces_revisions/annual/sample.py`
- Create: `tests/test_sample_panel.py`
- Create: `data/annual/raw/manual/sample-source-regimes.csv`
- Modify: `data/annual/raw/manifest.csv`

**Interfaces:**

- Consumes: Task 1's selected technical-note captures, Task 2's final-benchmark publication calendar, and `html_tables.find_table()`.
- Produces: `sample.raw_values(raw_dir) -> pl.DataFrame`, `sample.build_sample_panel(calendar, raw_dir) -> pl.DataFrame`, `SAMPLE_SCHEMA`, and the fixed `SECTORS` tuple. The unique key is `(benchmark_year, sector)`; it is intentionally annual and is never expanded to months.

- [x] **Step 1: Create the auditable source-regime map**

Create `data/annual/raw/manual/sample-source-regimes.csv` with this exact header:

```csv
benchmark_year,coverage_status,coverage_file,coverage_caption_regex,rse_status,rse_measure,rse_closing,rse_unit,rse_file,rse_caption_regex,rse_column_index,published_series_note,published_series_source_files,published_series_source_locators,evidence_observed_at
```

Populate exactly one row for every benchmark year 2002–2025:

- Coverage years 2002–2011 and 2013–2024 point to the selected Internet Archive technical-note copy; 2025 points to `bls/cestn.htm`. Their `coverage_status` is `published`, and their caption regex selects Table 1 only.
- The 2012 row has `coverage_status=archive_gap`, blank coverage file/caption, and `evidence_observed_at=2026-09-13T16:00:00Z`, the documented inventory check. This is evidence timing, not a fabricated 2012 publication.
- Surviving level-RSE tables through 2009 use `rse_status=published`, `rse_measure=level`, `rse_closing=first_closing`, and the exact percent column index in `rse_column_index`.
- Years 2010–2015 use `rse_status=not_published` with the measure, closing, file, caption, and column blank.
- Years 2016–2025 use `rse_status=published`, `rse_measure=one_month_change`, `rse_closing=not_stated`, `rse_unit=percent`, and point to the Table 4 copy first published with the February 2017 benchmark release and its successors.
- `published_series_note` is blank unless a source states a change. For 2025, record: `FY2022's 24,511 counts national series maintained, published and unpublished; FY2025's 22,049 counts national series published, so the apparent fall is not definition-comparable; the March 2025 benchmark article reports no series-structure change for sample coverage or disclosure review.` Point the semicolon-delimited source fields to the FY2024 CBJ pages BLS-36–38, FY2027 CBJ pages BLS-29–31, and the 2025 benchmark article's series section. Do not turn either count into a covariate.

Every `published` file must be an available row of `source-catalog.csv`; every caption regex must match exactly one table in that file.

- [x] **Step 2: Write the failing sample-panel tests**

Create `tests/test_sample_panel.py`:

```python
"""Annual CES Table 1 coverage proxy and published RSE regimes."""

from datetime import date

import polars as pl
import pytest

from ces_revisions.annual import publications, sample
from ces_revisions.vintages.release_index import build_release_index


def built() -> pl.DataFrame:
    calendar = publications.build_publication_calendar(build_release_index())
    return sample.build_sample_panel(calendar)


def test_required_table_1_counts_cannot_be_silently_missing():
    assert sample._required_integer("115,195", "active UI accounts") == 115195
    with pytest.raises(ValueError, match="active UI accounts"):
        sample._required_integer("", "active UI accounts")


def test_panel_has_one_annual_row_per_supersector_and_year():
    frame = built()
    assert frame.height == 24 * len(sample.SECTORS)
    assert frame["benchmark_year"].unique().sort().to_list() == list(range(2002, 2026))
    assert frame.group_by("benchmark_year").len()["len"].unique().to_list() == [
        len(sample.SECTORS)
    ]
    assert frame["frequency"].unique().to_list() == ["annual"]
    assert (
        frame["reference_month"]
        == pl.Series([date(year, 3, 1) for year in frame["benchmark_year"]])
    ).all()


def test_coverage_is_explicitly_the_active_report_proxy():
    frame = built()
    assert frame["coverage_measure"].unique().to_list() == [
        "table_1_active_report_employee_share_proxy"
    ]
    published = frame.filter(pl.col("coverage_status") == "published")
    calculated = (
        published["sample_employees_thousands"]
        / published["benchmark_employment_thousands"]
        * 100
    ).round(0)
    assert (calculated == published["coverage_percent"]).all()
    assert published["coverage_note"].str.contains("not the matched sample").all()


def test_march_2012_is_an_explicit_archive_gap_not_an_interpolation():
    rows = built().filter(pl.col("benchmark_year") == 2012)
    assert rows["coverage_status"].unique().to_list() == ["archive_gap"]
    assert (
        rows.select(
            "benchmark_employment_thousands",
            "active_ui_accounts",
            "active_establishments",
            "sample_employees_thousands",
            "coverage_percent",
        )
        .null_count()
        .row(0)
        == (len(rows),) * 5
    )
    assert rows["missing_reason"].str.contains("March 2012 Table 1").all()
    assert rows["observable_at"].is_not_null().all()


def test_rse_regimes_and_definition_break_are_preserved():
    frame = built()
    level = frame.filter(pl.col("rse_measure") == "level")
    change = frame.filter(pl.col("rse_measure") == "one_month_change")
    gap = frame.filter(pl.col("benchmark_year").is_between(2010, 2015))
    assert level["benchmark_year"].unique().sort().to_list() == list(range(2002, 2010))
    assert level["rse_closing"].unique().to_list() == ["first_closing"]
    assert change["benchmark_year"].unique().sort().to_list() == list(range(2016, 2026))
    assert change["rse_closing"].unique().to_list() == ["not_stated"]
    assert gap["rse_value"].is_null().all()
    assert frame.filter(pl.col("rse_definition_break"))[
        "benchmark_year"
    ].unique().to_list() == [2016]


def test_detailed_trade_rows_are_not_double_counted():
    assert set(built()["sector"]) == set(sample.SECTORS)
    assert not {"41", "42", "43", "44"} & set(built()["sector"])


def test_published_series_change_is_metadata_not_a_covariate():
    frame = built()
    assert "program_cut_covariate" not in frame.columns
    note = frame.filter(pl.col("benchmark_year") == 2025)[
        "published_series_note"
    ].unique()
    assert len(note) == 1
    assert "not definition-comparable" in note.item()
    assert "no series-structure change" in note.item()


def test_every_sample_row_has_an_as_of_timestamp():
    assert built()["observable_at"].is_not_null().all()
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sample_panel.py -q`

Expected: collection fails with `ImportError: cannot import name 'sample'`.

- [x] **Step 4: Write the source readers and stable annual schema**

Create `src/ces_revisions/annual/sample.py`:

```python
"""CES Table 1 active-report coverage proxy and published RSE regimes."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from ces_revisions.annual import html_tables
from ces_revisions.annual.raw import RAW_DIR, cell_key, parse_number, raw_frame

REGIMES = "manual/sample-source-regimes.csv"
SECTORS = ("00", "10", "20", "30", "40", "50", "55", "60", "65", "70", "80", "90")
COVERAGE_NOTE = (
    "CES Table 1 counts active sample reports; its employee share is a proxy for "
    "usable linked coverage, not the matched sample used by the estimator."
)
SAMPLE_SCHEMA = {
    "benchmark_year": pl.Int32,
    "reference_month": pl.Date,
    "frequency": pl.String,
    "sector": pl.String,
    "industry_title": pl.String,
    "benchmark_employment_thousands": pl.Float64,
    "active_ui_accounts": pl.Int64,
    "active_establishments": pl.Int64,
    "sample_employees_thousands": pl.Float64,
    "coverage_percent": pl.Float64,
    "coverage_measure": pl.String,
    "coverage_status": pl.String,
    "coverage_note": pl.String,
    "rse_value": pl.Float64,
    "rse_measure": pl.String,
    "rse_closing": pl.String,
    "rse_unit": pl.String,
    "rse_status": pl.String,
    "rse_definition_break": pl.Boolean,
    "missing_reason": pl.String,
    "published_series_note": pl.String,
    "publication_date": pl.Date,
    "observable_at": pl.Datetime("us", "UTC"),
    "source_files": pl.List(pl.String),
    "source_locators": pl.List(pl.String),
    "source_keys": pl.List(pl.String),
    "transformation": pl.String,
}


def _source_rows(raw_dir: Path) -> pl.DataFrame:
    return pl.read_csv(raw_dir / REGIMES, infer_schema_length=0)


def _table(raw_dir: Path, file: str, caption: str) -> html_tables.HtmlTable:
    page = (raw_dir / file).read_text(encoding="utf-8")
    return html_tables.find_table(page, caption)


def _sector_row(
    table: html_tables.HtmlTable, sector: str
) -> tuple[int, tuple[str, ...]]:
    matches = []
    for index, row in enumerate(table.rows):
        if not row:
            continue
        match = re.match(r"^(\d{2})-\d{6}", row[0])
        if match and match.group(1) == sector:
            matches.append((index, row))
    if len(matches) != 1:
        raise ValueError(
            f"expected one sector {sector} row in {table.caption!r}, found {len(matches)}"
        )
    return matches[0]


def _table_cells(
    raw_dir: Path, file: str, caption: str, table_key: str
) -> pl.DataFrame:
    return html_tables.table_raw_cells(
        _table(raw_dir, file, caption),
        source="bls",
        file=file,
        table_key=table_key,
    )


def _regime_raw_values(raw_dir: Path) -> pl.DataFrame:
    frame = _source_rows(raw_dir)
    records = [
        {
            "source": "manual_transcription",
            "file": REGIMES,
            "table_key": "sample_source_regimes",
            "row_key": str(row_number),
            "column_key": column,
            "text": row[column] or "",
        }
        for row_number, row in enumerate(frame.iter_rows(named=True), start=2)
        for column in frame.columns
    ]
    return raw_frame(records)


def raw_values(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    parts = [_regime_raw_values(raw_dir)]
    seen = set()
    for row in _source_rows(raw_dir).iter_rows(named=True):
        year = int(row["benchmark_year"])
        for kind in ("coverage", "rse"):
            if row[f"{kind}_status"] != "published":
                continue
            file = row[f"{kind}_file"]
            caption = row[f"{kind}_caption_regex"]
            key = (file, caption, kind, year)
            if key in seen:
                continue
            seen.add(key)
            parts.append(_table_cells(raw_dir, file, caption, f"sample_{kind}_{year}"))
    return pl.concat(parts).sort("cell_key")
```

The Table 1 data columns are fixed by the source contract: code, title, benchmark employment (thousands), active UI-account reports, establishments, sample employees (thousands), and employee coverage percent. The output names the UI count `active_ui_accounts` because Table 1 footnote 1 says the counts reflect active sample reports.

- [x] **Step 5: Write the coverage/RSE builder and regime-break logic**

Append to `src/ces_revisions/annual/sample.py`:

```python
def _publication(calendar: pl.DataFrame, year: int) -> dict:
    rows = calendar.filter(pl.col("publication_id") == f"benchmark_final_{year}")
    if rows.height != 1:
        raise ValueError(f"expected one final benchmark publication for {year}")
    return rows.row(0, named=True)


def _published_at(
    row: dict, calendar: pl.DataFrame
) -> tuple[date | None, datetime, list[str]]:
    if row["coverage_status"] == "archive_gap":
        instant = datetime.fromisoformat(
            row["evidence_observed_at"].replace("Z", "+00:00")
        ).astimezone(UTC)
        return None, instant, []
    publication = _publication(calendar, int(row["benchmark_year"]))
    return (
        publication["publication_date"],
        publication["observable_at"],
        publication["source_keys"],
    )


def _required_integer(text: str, label: str) -> int:
    value = parse_number(text)
    if value is None or not value.is_integer():
        raise ValueError(
            f"{label} must be a published whole-number count, got {text!r}"
        )
    return int(value)


def _coverage(
    row: dict, sector: str, raw_dir: Path
) -> tuple[dict[str, object], list[str]]:
    if row["coverage_status"] != "published":
        return (
            {
                "industry_title": None,
                "benchmark_employment_thousands": None,
                "active_ui_accounts": None,
                "active_establishments": None,
                "sample_employees_thousands": None,
                "coverage_percent": None,
            },
            [],
        )
    table = _table(raw_dir, row["coverage_file"], row["coverage_caption_regex"])
    row_index, cells = _sector_row(table, sector)
    if len(cells) != 7:
        raise ValueError(
            f"Table 1 for {row['benchmark_year']} has {len(cells)} columns, expected 7"
        )
    values = {
        "industry_title": cells[1],
        "benchmark_employment_thousands": parse_number(cells[2]),
        "active_ui_accounts": _required_integer(cells[3], "active UI accounts"),
        "active_establishments": _required_integer(cells[4], "active establishments"),
        "sample_employees_thousands": parse_number(cells[5]),
        "coverage_percent": parse_number(cells[6]),
    }
    keys = [
        cell_key(
            row["coverage_file"],
            f"sample_coverage_{row['benchmark_year']}",
            str(row_index),
            str(column),
        )
        for column in range(1, 7)
    ]
    return values, keys


def _rse(row: dict, sector: str, raw_dir: Path) -> tuple[float | None, list[str]]:
    if row["rse_status"] != "published":
        return None, []
    table = _table(raw_dir, row["rse_file"], row["rse_caption_regex"])
    row_index, cells = _sector_row(table, sector)
    column = int(row["rse_column_index"])
    if column >= len(cells):
        raise ValueError(
            f"RSE column {column} absent for {row['benchmark_year']} sector {sector}"
        )
    return parse_number(cells[column]), [
        cell_key(
            row["rse_file"],
            f"sample_rse_{row['benchmark_year']}",
            str(row_index),
            str(column),
        )
    ]


def _break_years(source: pl.DataFrame) -> set[int]:
    years = []
    previous = None
    for row in source.sort("benchmark_year").iter_rows(named=True):
        if row["rse_status"] != "published":
            continue
        definition = (row["rse_measure"], row["rse_closing"], row["rse_unit"])
        if previous is not None and definition != previous:
            years.append(int(row["benchmark_year"]))
        previous = definition
    return set(years)


def build_sample_panel(calendar: pl.DataFrame, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    source = _source_rows(raw_dir)
    breaks = _break_years(source)
    rows = []
    for row_number, source_row in enumerate(source.iter_rows(named=True), start=2):
        year = int(source_row["benchmark_year"])
        publication_date, observable_at, publication_keys = _published_at(
            source_row, calendar
        )
        for sector in SECTORS:
            coverage, coverage_keys = _coverage(source_row, sector, raw_dir)
            rse_value, rse_keys = _rse(source_row, sector, raw_dir)
            regime_keys = [
                cell_key(
                    REGIMES,
                    "sample_source_regimes",
                    str(row_number),
                    column,
                )
                for column in (
                    "coverage_status",
                    "rse_status",
                    "rse_measure",
                    "rse_closing",
                    "rse_unit",
                    "published_series_note",
                    "published_series_source_files",
                    "published_series_source_locators",
                    "evidence_observed_at",
                )
            ]
            missing_reason = None
            if source_row["coverage_status"] == "archive_gap":
                missing_reason = "March 2012 Table 1 was not found in the archive"
            rows.append(
                {
                    "benchmark_year": year,
                    "reference_month": date(year, 3, 1),
                    "frequency": "annual",
                    "sector": sector,
                    **coverage,
                    "coverage_measure": "table_1_active_report_employee_share_proxy",
                    "coverage_status": source_row["coverage_status"],
                    "coverage_note": COVERAGE_NOTE,
                    "rse_value": rse_value,
                    "rse_measure": source_row["rse_measure"] or None,
                    "rse_closing": source_row["rse_closing"] or None,
                    "rse_unit": source_row["rse_unit"] or None,
                    "rse_status": source_row["rse_status"],
                    "rse_definition_break": year in breaks,
                    "missing_reason": missing_reason,
                    "published_series_note": source_row["published_series_note"]
                    or None,
                    "publication_date": publication_date,
                    "observable_at": observable_at,
                    "source_files": sorted(
                        {
                            file
                            for file in (
                                source_row["coverage_file"],
                                source_row["rse_file"],
                                REGIMES,
                                *(
                                    source_row["published_series_source_files"].split(
                                        ";"
                                    )
                                    if source_row["published_series_source_files"]
                                    else []
                                ),
                            )
                            if file
                        }
                    ),
                    "source_locators": [
                        locator
                        for locator in (
                            source_row["coverage_caption_regex"],
                            source_row["rse_caption_regex"],
                            *(
                                source_row["published_series_source_locators"].split(
                                    ";"
                                )
                                if source_row["published_series_source_locators"]
                                else []
                            ),
                        )
                        if locator
                    ],
                    "source_keys": sorted(
                        set(coverage_keys + rse_keys + regime_keys + publication_keys)
                    ),
                    "transformation": (
                        "explicit_archive_gap"
                        if source_row["coverage_status"] == "archive_gap"
                        else "parse_published_number"
                    ),
                }
            )
    frame = pl.DataFrame(rows, schema=SAMPLE_SCHEMA).sort("benchmark_year", "sector")
    if frame.select("benchmark_year", "sector").n_unique() != frame.height:
        raise ValueError("duplicate annual sample rows")
    return frame
```

- [x] **Step 6: Run the focused tests and validate all selected tables**

Run:

```bash
uv run pytest tests/test_sample_panel.py -q
uv run python -c 'from ces_revisions.annual import publications,sample; from ces_revisions.vintages.release_index import build_release_index; i=build_release_index(); f=sample.build_sample_panel(publications.build_publication_calendar(i)); print(f.group_by("coverage_status","rse_measure").len().sort("coverage_status","rse_measure"))'
uv run ruff format src/ces_revisions/annual/sample.py tests/test_sample_panel.py
uv run ruff check src/ces_revisions/annual/sample.py tests/test_sample_panel.py
```

Expected: all tests pass; the inspection shows the coverage gap only in 2012 and the two non-null RSE definitions on their documented year ranges.

- [x] **Step 7: Refresh the manifest and run the fast tier**

Run:

```bash
uv run python scripts/annual_sources.py manifest
uv run pytest -m "not slow and not network" -q
```

Expected: `sample-source-regimes.csv` is hashed and all tests pass.

- [x] **Step 8: Commit**

```bash
git add src/ces_revisions/annual/sample.py tests/test_sample_panel.py data/annual/raw/manual/sample-source-regimes.csv data/annual/raw/manifest.csv
git commit -m "Build annual CES coverage and RSE panels"
```

---

### Task 7: Assemble and Write the Stage 4 Artifacts

**Files:**

- Create: `src/ces_revisions/annual/build.py`
- Create: `tests/annual_data.py`
- Create: `tests/test_annual_build.py`
- Modify: `tests/test_annual_sources.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**

- Consumes: every Task 2–6 builder plus `vintages.release_index.build_release_index(stage3_raw_dir)`.
- Produces: immutable `AnnualBuild`, `annual.build.build(raw_dir=..., stage3_raw_dir=...) -> AnnualBuild`, `annual.build.write(result, out_dir=..., raw_dir=..., stage3_raw_dir=...) -> dict`, nine parquet artifacts, and `data/annual/panel/manifest.json`.

- [x] **Step 1: Write the failing assembly and round-trip tests**

Create `tests/test_annual_build.py`:

```python
"""Stage 4's complete build and content-hashed artifact manifest."""

import json
from dataclasses import fields

import polars as pl

from ces_revisions.annual import build, raw
from ces_revisions.vintages import raw as vintage_raw


def test_build_exposes_every_stage_4_artifact():
    result = build.build()
    assert [field.name for field in fields(result)] == [
        "raw_values",
        "transformations",
        "publication_calendar",
        "benchmarks",
        "reconstruction_events",
        "birth_death",
        "qcew_revisions",
        "qcew_precision",
        "sample_panel",
    ]
    for field in fields(result):
        frame = getattr(result, field.name)
        assert isinstance(frame, pl.DataFrame)
        assert not frame.is_empty(), field.name
    assert result.raw_values["cell_key"].n_unique() == result.raw_values.height


def test_write_round_trips_every_frame_and_hashes_both_source_archives(tmp_path):
    result = build.build()
    out = tmp_path / "panel"
    manifest = build.write(result, out_dir=out)
    assert manifest["annual_source_manifest_sha256"] == raw.file_sha256(
        raw.RAW_DIR / raw.MANIFEST
    )
    assert manifest["stage3_source_manifest_sha256"] == vintage_raw.file_sha256(
        vintage_raw.RAW_DIR / vintage_raw.MANIFEST
    )
    for field in fields(result):
        expected = getattr(result, field.name)
        actual = pl.read_parquet(out / f"{field.name}.parquet")
        assert actual.equals(expected), field.name
        assert manifest["artifacts"][field.name] == {
            "rows": expected.height,
            "sha256": raw.content_sha256(expected),
        }
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_built_outputs_live_only_below_the_gitignored_panel_directory():
    assert raw.PANEL_DIR == raw.ROOT / "data" / "annual" / "panel"
    assert not str(raw.RAW_DIR).startswith(str(raw.PANEL_DIR))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_annual_build.py -q`

Expected: collection fails with `ImportError: cannot import name 'build'` from `ces_revisions.annual`.

- [x] **Step 3: Write the build object and artifact writer**

Create `src/ces_revisions/annual/build.py`:

```python
"""Build every Stage 4 artifact from the two committed source archives."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

from ces_revisions.annual import (
    benchmarks,
    birth_death,
    publications,
    qcew,
    raw,
    reconstructions,
    sample,
)
from ces_revisions.vintages import raw as vintage_raw
from ces_revisions.vintages.release_index import build_release_index


@dataclass(frozen=True)
class AnnualBuild:
    raw_values: pl.DataFrame
    transformations: pl.DataFrame
    publication_calendar: pl.DataFrame
    benchmarks: pl.DataFrame
    reconstruction_events: pl.DataFrame
    birth_death: pl.DataFrame
    qcew_revisions: pl.DataFrame
    qcew_precision: pl.DataFrame
    sample_panel: pl.DataFrame


def _raw_values(raw_dir: Path) -> pl.DataFrame:
    parts = [
        publications.publication_raw_values(raw_dir),
        benchmarks.table5_raw_values(raw_dir),
        benchmarks.benchmark_sector_raw_values(raw_dir),
        reconstructions.raw_values(raw_dir),
        birth_death.raw_values(raw_dir),
        qcew.raw_values(raw_dir),
        sample.raw_values(raw_dir),
    ]
    frame = pl.concat(parts).sort("cell_key")
    duplicates = frame.group_by("cell_key").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"duplicate Stage 4 raw cells: {duplicates.rows()}")
    return frame


def build(
    raw_dir: Path = raw.RAW_DIR,
    stage3_raw_dir: Path = vintage_raw.RAW_DIR,
) -> AnnualBuild:
    release_index = build_release_index(stage3_raw_dir)
    calendar = publications.build_publication_calendar(release_index, raw_dir)
    qcew_revisions = qcew.build_qcew_revisions(calendar, raw_dir)
    return AnnualBuild(
        raw_values=_raw_values(raw_dir),
        transformations=raw.TRANSFORMATIONS,
        publication_calendar=calendar,
        benchmarks=benchmarks.build_benchmarks(calendar, raw_dir),
        reconstruction_events=reconstructions.build_reconstruction_events(
            calendar, raw_dir
        ),
        birth_death=birth_death.build_birth_death(calendar, release_index, raw_dir),
        qcew_revisions=qcew_revisions,
        qcew_precision=qcew.build_qcew_precision(qcew_revisions),
        sample_panel=sample.build_sample_panel(calendar, raw_dir),
    )


def write(
    result: AnnualBuild,
    out_dir: Path = raw.PANEL_DIR,
    raw_dir: Path = raw.RAW_DIR,
    stage3_raw_dir: Path = vintage_raw.RAW_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for field in fields(result):
        frame = getattr(result, field.name)
        frame.write_parquet(out_dir / f"{field.name}.parquet")
        artifacts[field.name] = {
            "rows": frame.height,
            "sha256": raw.content_sha256(frame),
        }
    manifest = {
        "annual_source_manifest_sha256": raw.file_sha256(raw_dir / raw.MANIFEST),
        "stage3_source_manifest_sha256": vintage_raw.file_sha256(
            stage3_raw_dir / vintage_raw.MANIFEST
        ),
        "artifacts": artifacts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
```

This mirrors Stage 3's `VintageBuild` convention but pins two source manifests: Stage 4's annual archive and Stage 3's release-date archive. The output directory remains rebuilt and gitignored; raw source bytes, transcriptions, and their manifest remain committed.

- [x] **Step 4: Add session-cached test access and the CLI contract test**

Create `tests/annual_data.py`:

```python
"""One cached Stage 4 build shared by contract tests."""

from functools import cache

import polars as pl

from ces_revisions.annual.build import AnnualBuild, build


@cache
def result() -> AnnualBuild:
    return build()


def raw_values() -> pl.DataFrame:
    return result().raw_values


def artifact(name: str) -> pl.DataFrame:
    return getattr(result(), name)
```

Append to `tests/test_annual_sources.py`:

```python
def test_build_command_reports_each_written_artifact(monkeypatch, capsys):
    expected = {"benchmarks": {"rows": 70}}
    monkeypatch.setattr("ces_revisions.annual.build.build", lambda: object())
    monkeypatch.setattr(
        "ces_revisions.annual.build.write", lambda result: {"artifacts": expected}
    )
    assert annual_sources.build_tables() == 0
    assert capsys.readouterr().out == "benchmarks: 70 rows\n"
```

- [x] **Step 5: Document the annual archive and offline build**

Append this section after the vintage-panel section in `README.md`:

```markdown
### Annual benchmark-source tables

Roadmap Stage 4 archives BLS benchmark articles, technical-note snapshots,
birth–death pages, and the national QCEW revisions CSV under
`data/annual/raw/`. The committed `manifest.csv` pins every source and manual
table transcription by SHA-256. Built parquet is reproducible and remains
gitignored under `data/annual/panel/`.

Set `BLS_CONTACT_EMAIL` in the gitignored `.project.env`; every network
request identifies that address in its `User-Agent` without recording it in
logs or manifests. Then run:

```bash
uv run python scripts/annual_sources.py fetch     # network + PDF text extraction
uv run python scripts/annual_sources.py manifest  # offline rehash
uv run python scripts/annual_sources.py build     # offline parquet build
```

The build writes dated benchmark, reconstruction, birth–death, QCEW-revision,
QCEW-precision, and annual sample/RSE tables. CES quantities are in thousands;
the long QCEW sequence remains in jobs.
```

Add these bullets to `CLAUDE.md` under “Layout and tooling”:

```markdown
- `src/ces_revisions/annual/` implements roadmap Stage 4. Its raw sources and
  manual table transcriptions are committed under `data/annual/raw/`; rebuilt
  parquet lives under the gitignored `data/annual/panel/`. Every derived value
  carries raw-cell provenance, a named transformation, and `observable_at`.
- `scripts/annual_sources.py fetch` loads `BLS_CONTACT_EMAIL` from the
  gitignored `.project.env` via `python-dotenv` and includes it in every HTTP
  `User-Agent`. Never print or commit the address. The `manifest` and `build`
  commands are offline.
- Stage 4 stores CES benchmark and birth–death values in thousands and the raw
  QCEW revision sequence in jobs. Sample coverage/RSE is annual, the coverage
  share is explicitly an active-report proxy, and the RSE definitions are not
  harmonized across their documented break.
```

Add these commands to `CLAUDE.md`'s command block:

```bash
uv run python scripts/annual_sources.py manifest  # rehash committed Stage 4 sources (offline)
uv run python scripts/annual_sources.py build     # rebuild data/annual/panel/ (offline)
```

- [x] **Step 6: Build twice and verify deterministic artifacts**

Run:

```bash
uv run pytest tests/test_annual_build.py tests/test_annual_sources.py -q -m "not network"
uv run python scripts/annual_sources.py build
cp data/annual/panel/manifest.json /tmp/ces-revisions-stage4-manifest.json
uv run python scripts/annual_sources.py build
cmp /tmp/ces-revisions-stage4-manifest.json data/annual/panel/manifest.json
git status --short data/annual/panel
uv run ruff format src/ces_revisions/annual/build.py tests/annual_data.py tests/test_annual_build.py tests/test_annual_sources.py README.md CLAUDE.md
uv run ruff check src/ces_revisions/annual/build.py tests/annual_data.py tests/test_annual_build.py tests/test_annual_sources.py
```

Expected: focused tests pass; both manifest files are byte-identical; `git status` prints nothing for the ignored panel; ruff is clean.

- [x] **Step 7: Run the full fast tier**

Run: `uv run pytest -m "not slow and not network" -q`

Expected: all existing and Stage 4 fast tests pass.

- [x] **Step 8: Commit**

```bash
git add src/ces_revisions/annual/build.py tests/annual_data.py tests/test_annual_build.py tests/test_annual_sources.py README.md CLAUDE.md
git commit -m "Assemble reproducible Stage 4 annual artifacts"
```

---

### Task 8: Enforce the Cross-Artifact Contract and Pass the Stage Gate

**Files:**

- Create: `src/ces_revisions/annual/contract.py`
- Create: `tests/test_annual_contract.py`
- Modify: `src/ces_revisions/annual/build.py`
- Modify: `data/annual/raw/manifest.csv`

**Interfaces:**

- Consumes: `AnnualBuild`, Stage 3's release index, all stable annual raw-cell keys, and the namespaced `stage3::release_index::YYYY-MM` keys from Task 2.
- Produces: `contract.validate(result, release_index) -> None`; `build()` calls it before returning, so CLI and tests share the same exit gate.

- [x] **Step 1: Write the failing cross-artifact tests**

Create `tests/test_annual_contract.py`:

```python
"""Roadmap Stage 4's executable exit criteria across all annual artifacts."""

from dataclasses import replace

import polars as pl
import pytest

import annual_data
from ces_revisions.annual import contract
from ces_revisions.vintages.release_index import build_release_index


def test_complete_build_passes_the_stage_4_contract():
    contract.validate(annual_data.result(), build_release_index())


def test_contract_rejects_an_undated_value():
    result = annual_data.result()
    broken = (
        result.benchmarks.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(None)
            .otherwise(pl.col("observable_at"))
            .alias("observable_at")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="benchmarks has null observable_at"):
        contract.validate(replace(result, benchmarks=broken), build_release_index())


def test_contract_rejects_an_unknown_source_key():
    result = annual_data.result()
    broken = (
        result.qcew_precision.with_row_index("row")
        .with_columns(
            pl.when(pl.col("row") == 0)
            .then(pl.lit(["unknown::cell"], dtype=pl.List(pl.String)))
            .otherwise(pl.col("source_keys"))
            .alias("source_keys")
        )
        .drop("row")
    )
    with pytest.raises(ValueError, match="unknown source keys"):
        contract.validate(replace(result, qcew_precision=broken), build_release_index())


def test_final_and_reconstruction_dates_equal_the_stage_3_carriers():
    result = annual_data.result()
    finals = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_final"
    )
    joined = result.reconstruction_events.join(
        finals.select(
            "benchmark_year",
            final_date=pl.col("publication_date"),
            final_observable=pl.col("observable_at"),
        ),
        on="benchmark_year",
    )
    assert (joined["publication_date"] == joined["final_date"]).all()
    assert (joined["observable_at"] == joined["final_observable"]).all()


def test_preliminary_benchmarks_use_only_stage_4_publication_evidence():
    preliminary = annual_data.artifact("publication_calendar").filter(
        pl.col("publication_kind") == "benchmark_preliminary"
    )
    assert preliminary["source_keys"].list.len().gt(0).all()
    assert preliminary["source_keys"].explode().str.starts_with("stage3::").not_().all()


def test_no_annual_sample_measure_was_repeated_across_months():
    sample = annual_data.artifact("sample_panel")
    assert sample.select("benchmark_year", "sector").n_unique() == sample.height
    assert sample["frequency"].unique().to_list() == ["annual"]


def test_qcew_long_values_are_jobs_and_precision_is_thousands():
    assert annual_data.artifact("qcew_revisions")["unit"].unique().to_list() == ["jobs"]
    assert annual_data.artifact("qcew_precision")["unit"].unique().to_list() == [
        "thousands"
    ]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_annual_contract.py -q`

Expected: collection fails with `ImportError: cannot import name 'contract'`.

- [x] **Step 3: Implement the executable contract**

> **Deviation (review hardening):** The contract also enforces required derived schemas,
> non-null/non-empty provenance elements, registered transformations, complete benchmark
> and sample domains, government structural zeros, and the exact 2025 ruling/event set.

Create `src/ces_revisions/annual/contract.py`:

```python
"""Cross-artifact invariants that define roadmap Stage 4 completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ces_revisions.annual.publications import final_carrier_month

if TYPE_CHECKING:
    from ces_revisions.annual.build import AnnualBuild

DERIVED_ARTIFACTS = (
    "publication_calendar",
    "benchmarks",
    "reconstruction_events",
    "birth_death",
    "qcew_revisions",
    "qcew_precision",
    "sample_panel",
)


def _all_source_keys(frame: pl.DataFrame) -> set[str]:
    if "source_keys" not in frame.columns:
        return set()
    if (
        frame["source_keys"].is_null().any()
        or frame["source_keys"].list.len().eq(0).any()
    ):
        raise ValueError("a derived row has no source keys")
    return set(frame["source_keys"].explode().drop_nulls())


def _validate_provenance(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    annual_keys = set(result.raw_values["cell_key"])
    stage3_keys = {
        f"stage3::release_index::{month:%Y-%m}"
        for month in release_index["reference_month"]
    }
    allowed = annual_keys | stage3_keys
    used_transformations = set()
    for name in DERIVED_ARTIFACTS:
        frame = getattr(result, name)
        unknown = _all_source_keys(frame) - allowed
        if unknown:
            raise ValueError(f"{name} has unknown source keys: {sorted(unknown)[:5]}")
        if "transformation" in frame.columns:
            used_transformations.update(frame["transformation"].drop_nulls())
    registered = set(result.transformations["transformation"])
    if not used_transformations <= registered:
        raise ValueError(
            f"unregistered transformations: {sorted(used_transformations - registered)}"
        )


def _validate_dates(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    for name in DERIVED_ARTIFACTS:
        frame = getattr(result, name)
        if frame["observable_at"].is_null().any():
            raise ValueError(f"{name} has null observable_at")
    by_month = {
        row["reference_month"]: row for row in release_index.iter_rows(named=True)
    }
    finals = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_final"
    )
    final_by_year = {}
    for row in finals.iter_rows(named=True):
        expected = by_month[final_carrier_month(row["benchmark_year"])]
        if (
            row["publication_date"] != expected["published_date"]
            or row["observable_at"] != expected["observable_at"]
        ):
            raise ValueError(
                f"benchmark year {row['benchmark_year']} does not match Stage 3"
            )
        final_by_year[row["benchmark_year"]] = row
    final_artifacts = {
        "benchmarks": result.benchmarks.filter(pl.col("benchmark_status") == "final"),
        "reconstruction_events": result.reconstruction_events,
    }
    for name, frame in final_artifacts.items():
        for row in frame.iter_rows(named=True):
            expected = final_by_year[row["benchmark_year"]]
            if (
                row["publication_date"] != expected["publication_date"]
                or row["observable_at"] != expected["observable_at"]
            ):
                raise ValueError(
                    f"{name} benchmark year {row['benchmark_year']} has the wrong carrier"
                )
    prelim = result.publication_calendar.filter(
        pl.col("publication_kind") == "benchmark_preliminary"
    )
    if prelim["source_keys"].explode().str.starts_with("stage3::").any():
        raise ValueError("preliminary benchmark dates must use Stage 4 evidence")
    prelim_by_year = {
        row["benchmark_year"]: row for row in prelim.iter_rows(named=True)
    }
    for row in result.benchmarks.filter(
        pl.col("benchmark_status") == "preliminary"
    ).iter_rows(named=True):
        expected = prelim_by_year[row["benchmark_year"]]
        if (
            row["publication_date"] != expected["publication_date"]
            or row["observable_at"] != expected["observable_at"]
        ):
            raise ValueError(
                f"preliminary benchmark year {row['benchmark_year']} has the wrong date"
            )


def _validate_domain_contracts(result: AnnualBuild) -> None:
    government = result.birth_death.filter(pl.col("sector") == "90")
    if government.is_empty() or government["value_thousands"].ne(0.0).any():
        raise ValueError("government birth-death is not structurally zero")
    if set(result.qcew_revisions["unit"]) != {"jobs"}:
        raise ValueError("QCEW revision values must remain in jobs")
    if set(result.qcew_precision["unit"]) != {"thousands"}:
        raise ValueError("QCEW precision must be in thousands")
    sample = result.sample_panel
    if set(sample["frequency"]) != {"annual"}:
        raise ValueError("sample measures must remain annual")
    missing_2012 = sample.filter(pl.col("benchmark_year") == 2012)
    if set(missing_2012["coverage_status"]) != {"archive_gap"}:
        raise ValueError("March 2012 coverage must remain an archive gap")


def validate(result: AnnualBuild, release_index: pl.DataFrame) -> None:
    """Raise on any Stage 4 exit-criterion violation."""
    _validate_provenance(result, release_index)
    _validate_dates(result, release_index)
    _validate_domain_contracts(result)
```

Modify `build()` in `src/ces_revisions/annual/build.py` so the final construction is assigned, validated, and returned:

```python
def build(
    raw_dir: Path = raw.RAW_DIR,
    stage3_raw_dir: Path = vintage_raw.RAW_DIR,
) -> AnnualBuild:
    release_index = build_release_index(stage3_raw_dir)
    calendar = publications.build_publication_calendar(release_index, raw_dir)
    qcew_revisions = qcew.build_qcew_revisions(calendar, raw_dir)
    result = AnnualBuild(
        raw_values=_raw_values(raw_dir),
        transformations=raw.TRANSFORMATIONS,
        publication_calendar=calendar,
        benchmarks=benchmarks.build_benchmarks(calendar, raw_dir),
        reconstruction_events=reconstructions.build_reconstruction_events(
            calendar, raw_dir
        ),
        birth_death=birth_death.build_birth_death(calendar, release_index, raw_dir),
        qcew_revisions=qcew_revisions,
        qcew_precision=qcew.build_qcew_precision(qcew_revisions),
        sample_panel=sample.build_sample_panel(calendar, raw_dir),
    )
    contract.validate(result, release_index)
    return result
```

Add `contract` to the grouped import from `ces_revisions.annual`. Replace the Task 7 direct `return AnnualBuild(...)` block with the block above; do not retain both.

- [x] **Step 4: Run the focused exit gate**

Run:

```bash
uv run pytest tests/test_annual_contract.py -q
uv run python scripts/annual_sources.py build
uv run ruff format src/ces_revisions/annual/contract.py src/ces_revisions/annual/build.py tests/test_annual_contract.py
uv run ruff check src/ces_revisions/annual/contract.py src/ces_revisions/annual/build.py tests/test_annual_contract.py
```

Expected: all contract tests pass, the offline build succeeds, and ruff is clean.

- [x] **Step 5: Run the source, deterministic-build, and full-suite gates**

Run in this order:

```bash
uv sync --locked
uv run python scripts/annual_sources.py manifest
uv run pytest -m "not slow and not network" -q
uv run pytest -m slow -q
uv run pytest tests/test_annual_sources.py -m network -q
uv run ruff format --check .
uv run ruff check .
git diff --check
git status --short
```

Expected: the locked environment resolves; the manifest is already current after regeneration; the fast, slow, and network tiers pass; both ruff commands and `git diff --check` exit 0. `git status` shows only the Stage 4 implementation, committed source archive/transcriptions, and any pre-existing user files. If a live source changed, stop and perform Task 1's capture/inspection flow; never update a fixture alone.

- [x] **Step 6: Request and resolve the final whole-branch review**

> **Deviation (review routing):** The dedicated `code-reviewer` role was unavailable on
> two dispatch attempts, so an isolated reviewer was given the same whole-plan rubric.
> Its four Important and two Minor findings were fixed, and follow-up review confirmed
> that no findings remain open.

Use `requesting-code-review` with the base commit from immediately before Task 1 and the current `HEAD`. The review brief is:

```text
Review roadmap Stage 4 against specs/plans/6-ces-revisions.md. Prioritize as-of correctness, BLS units and definitions, QCEW publication counts, the 2025 benchmark ruling, government structural zeros, raw-cell provenance, and accidental persistence of BLS_CONTACT_EMAIL. Confirm every exit criterion in roadmap lines 104–110 has executable evidence.
```

Resolve every Important finding before continuing. Fix Minor findings that affect correctness, provenance, secrets, or reproducibility; put only genuinely non-blocking leftovers through the Plan Completion Protocol's resolve-before-defer gate. Re-run the commands from Step 5 after the last fix.

- [x] **Step 7: Commit the gate**

```bash
git add src/ces_revisions/annual/contract.py src/ces_revisions/annual/build.py tests/test_annual_contract.py data/annual/raw/manifest.csv
git commit -m "Enforce the Stage 4 annual-data contract"
```

---

## At completion

Run this only after all eight tasks and the final review are resolved.

1. Apply the `writing-plans` **Plan Completion Protocol**. Collect skipped steps and unresolved review findings, ask one batched set of genuinely blocking questions, implement anything unblocked by the answers, and defer only self-contained non-blockers with a size and closure condition.
2. Mark every completed checkbox in this file. Add a one-line deviation/skipped note under any step that did not execute as written. Add one of these exact headers below the title:

   ```markdown
   **Status: COMPLETE (YYYY-MM-DD)** — executed via subagent-driven-development; nothing deferred
   ```

   or, if the gate created backlog items:

   ```markdown
   **Status: COMPLETE (YYYY-MM-DD)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md
   ```

   Use `executing-plans` in that line instead if the user selected inline execution.
3. Update `specs/deferred_items.md`. Tick the plan 2 “usable linked coverage is not published” item and append `→ done in plan 6`; this plan implements its coverage proxy, 2012 archive gap, and RSE break. Leave the separate BLS request for matched-sample counts and first-closing dates open. Append a `## 6-ces-revisions — YYYY-MM-DD` section only when this execution actually deferred something.
4. Report backlog health by running:

   ```bash
   uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py
   ```

   Surface its open count, closure rate, and aged tail. If it reports at least 20 open items or any aged tail, run steps 1–4 of `writing-plans/references/deferred-backlog.md`, present the grouped proposal, note that `/deferred` applies dispositions, and continue; triage does not block retirement.
5. In `specs/ces-revisions-roadmap.md`, change only Stage 4's checkbox to `[x]`. Re-read and re-validate Stages 5, 6, 8, 14, 15, 16, and 20 against the artifact names, units, timing, and annual-frequency contracts that shipped; amend their `Consumes`/exit prose in the same retirement commit only if an actual interface differs from this plan.
6. Append this exact authoritative stamp to the Rollout note in `specs/ces-revisions.md`, substituting the real date:

   ```markdown
   Stage 4: COMPLETE (YYYY-MM-DD) — implemented by plan 6 (specs/plans/completed/6-ces-revisions.md). Next: resume the roadmap.
   ```

   The shared spec remains live because later roadmap stages still implement it.
7. Retire this plan and repair its now-one-level-deeper relative links: `../ces-revisions.md` becomes `../../ces-revisions.md`, `../ces-revisions-roadmap.md` becomes `../../ces-revisions-roadmap.md`, `../deferred_items.md` becomes `../../deferred_items.md`, and `../../docs/ces-revisions-review.md` becomes `../../../docs/ces-revisions-review.md`. Then commit completion atomically:

   ```bash
   git mv specs/plans/6-ces-revisions.md specs/plans/completed/6-ces-revisions.md
   git add specs/plans/completed/6-ces-revisions.md specs/deferred_items.md specs/ces-revisions-roadmap.md specs/ces-revisions.md
   git commit -m "chore(specs): retire plan 6"
   ```

8. Use `finishing-a-development-branch` to present merge/PR/keep/discard options only after the retirement commit and final verification are both present.
