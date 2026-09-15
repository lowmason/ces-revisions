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


def test_publication_catalog_dates_are_raw_values():
    raw_values = publications.publication_raw_values()
    expected = publications.catalog_publications()["source_keys"].explode()
    assert set(expected.to_list()) <= set(raw_values["cell_key"].to_list())
