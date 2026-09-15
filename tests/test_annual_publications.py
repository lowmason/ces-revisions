"""Stage 4 publication dates and as-of timestamps."""

from datetime import UTC, date, datetime

import polars as pl

from ces_revisions.annual import publications
from ces_revisions.vintages.release_index import build_release_index


def test_final_carrier_preserves_the_historical_release_exceptions():
    assert publications.final_carrier_month(1979) == date(1980, 6, 1)
    assert publications.final_carrier_month(1980) == date(1981, 6, 1)
    assert publications.final_carrier_month(1981) == date(1982, 5, 1)
    assert publications.final_carrier_month(1989) == date(1990, 8, 1)
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


def test_preliminary_dates_use_the_announcement_not_a_later_jobs_report():
    calendar = publications.build_publication_calendar(build_release_index())
    preliminary = calendar.filter(pl.col("publication_kind") == "benchmark_preliminary")
    expected = {
        2011: (date(2011, 9, 29), datetime(2011, 9, 29, 12, 30, tzinfo=UTC)),
        2012: (date(2012, 9, 27), datetime(2012, 9, 27, 12, 30, tzinfo=UTC)),
        2013: (date(2013, 9, 26), datetime(2013, 9, 26, 14, 0, tzinfo=UTC)),
        2018: (date(2018, 8, 22), datetime(2018, 8, 22, 14, 0, tzinfo=UTC)),
    }
    observed = {
        row["benchmark_year"]: (row["publication_date"], row["observable_at"])
        for row in preliminary.filter(
            pl.col("benchmark_year").is_in(expected)
        ).iter_rows(named=True)
    }
    assert observed == expected


def test_qcew_calendar_distinguishes_news_from_full_data():
    calendar = publications.build_publication_calendar(build_release_index())
    rows = calendar.filter(
        (pl.col("qcew_year") == 2018) & (pl.col("qcew_quarter") == 1)
    ).sort("publication_kind")
    assert rows.select("publication_kind", "publication_date").rows() == [
        ("qcew_full_data", date(2018, 9, 5)),
        ("qcew_news", date(2018, 8, 22)),
    ]


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
    expected = publications.catalog_publications()["source_keys"].explode(
        empty_as_null=True
    )
    assert set(expected.to_list()) <= set(raw_values["cell_key"].to_list())
