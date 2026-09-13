"""docs/ces-revisions-review.md, the Req 20 finding: structure, conventions, and evidence."""

import re
import subprocess
import sys

import pytest
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
    EVIDENCE_LABELS,
    REVIEW_PATH,
    headings,
    is_primary,
    links,
    outside_fences,
    read_review,
    section,
    tables,
)

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

RULINGS = [
    "Collection rate versus response rate",
    "The December 2018 to January 2019 lapse in appropriations",
    "Whether the 2025 program cuts reached CES",
    "FTE concepts: authorized versus actual, and FTE versus headcount",
    "The 2024 and 2025 preliminary and final benchmark revisions",
]
RULING_PARAGRAPHS = [
    "Drafts.",
    "Primary sources.",
    "Ruling.",
    "Evidence:",
    "Consequence.",
]
_DRAFT_LOCATOR = re.compile(r"\((chatgpt|claude|gemini) §")

ARCHIVE_SUBSECTIONS = [
    "What BLS publishes",
    "Method",
    "Findings",
    "Coverage by year",
    "Per-vintage inventory",
    "Unrounded NSA inputs",
    "Consequences for later stages",
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


def test_birth_death_sample_and_macro_inventory():
    check_inventory_group("Net birth–death")
    check_inventory_group("Sample")
    check_inventory_group("Macro and tail controls")


def test_ruling_on_the_2025_program_cuts():
    check_ruling("Whether the 2025 program cuts reached CES")


def test_ruling_on_the_2024_and_2025_benchmark_revisions():
    check_ruling("The 2024 and 2025 preliminary and final benchmark revisions")
