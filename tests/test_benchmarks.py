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
