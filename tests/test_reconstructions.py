"""The sole structured source of documented benchmark reconstructions."""

from datetime import date

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
        "2018-state-ownership-change",
        "2018-wholesale-recoding",
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


def test_2018_reconstructions_cover_both_documented_sector_transfers():
    rows = built().filter(pl.col("benchmark_year") == 2018)
    assert rows.select("event_id", "sectors", "effect_thousands").rows() == [
        ("2018-state-ownership-change", ["65", "90"], 17.0),
        ("2018-wholesale-recoding", ["40", "60"], 336.0),
    ]
    assert rows["reference_start"].to_list() == [
        date(2018, 1, 1),
        date(1990, 1, 1),
    ]
    assert rows["footnote"].null_count() == 2


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
