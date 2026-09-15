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
