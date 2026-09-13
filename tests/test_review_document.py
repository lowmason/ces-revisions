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
