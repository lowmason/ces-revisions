"""Req 9's accounting identity and the variance decomposition of published revisions."""

from datetime import date

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import accounting

RELEASES = {"F": date(2010, 3, 1), "S": date(2010, 4, 1)}


def change(stage: str, status: str, value: int, release: date | None = None) -> dict:
    return {
        "source": "cesvinall",
        "sector": "00",
        "reference_month": date(2010, 3, 1),
        "seasonal_status": status,
        "release_stage": stage,
        "release_month": release or RELEASES[stage],
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


def test_identity_terms_reject_a_stage_whose_values_come_from_two_releases():
    changes = pl.DataFrame(
        [
            change("F", "NSA", 300),
            change("F", "SA", 120, release=date(2010, 4, 1)),
            change("S", "NSA", 340),
            change("S", "SA", 150),
        ]
    )
    with pytest.raises(ValueError, match="different releases"):
        accounting.identity_terms(changes)


# --- The committed sources ------------------------------------------------------------------


def test_the_identity_holds_exactly_on_every_same_release_pair():
    """The residual is zero by construction. The substance is the same-release premise:
    identity_terms raises unless each stage's unadjusted and adjusted changes share a release."""
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
