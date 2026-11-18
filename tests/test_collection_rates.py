"""The first-closing collection rate C1 and its sources: LABSTAT and the archived registry pages."""

from datetime import date
from pathlib import Path

import first_closing
import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw
from ces_revisions.vintages.collection_rates import (
    LABSTAT_SOURCE,
    RECON_COPY,
    SCHEMA,
    archive_comparison,
    archive_differences,
    build_collection_rates,
    labstat_values,
    registry_page_values,
)

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"
COPY = "20100115120000"
LABSTAT_HEADER = (
    "series_id                     \tyear\tperiod\t       value\tfootnote_codes\n"
)


def synthetic_page() -> str:
    return (FIXTURES / "cesregrec-synthetic.htm").read_text(encoding="utf-8")


def test_labstat_values_read_padded_fields_and_skip_annual_and_blank_cells():
    text = (
        LABSTAT_HEADER
        + "CEU00000000C1                 \t2003\tM01\t        65.2\t\n"
        + "CEU00000000C1                 \t2003\tM13\t        66.0\t\n"
        + "CEU00000000C1                 \t2003\tM02\t           -\t\n"
        + "CEU00000000C1                 \t2003\tM03\t            \t\n"
    )
    frame = labstat_values(text)
    assert frame.rows() == [("CEU00000000C1", date(2003, 1, 1), "labstat", 65.2, None)]
    assert frame.schema == SCHEMA


def test_labstat_values_keep_footnote_codes():
    text = LABSTAT_HEADER + "CEU00000000C1                 \t2025\tM09\t 70.1\tE\n"
    assert labstat_values(text)["footnote_code"].to_list() == ["E"]


def test_registry_page_values_read_three_sections_into_c1_c2_c3():
    frame = registry_page_values(synthetic_page(), COPY)
    counts = frame.group_by(
        "series_id", pl.col("reference_month").dt.year().alias("year")
    ).len()
    assert sorted(counts.rows()) == [
        ("CEU00000000C1", 1981, 11),
        ("CEU00000000C1", 2003, 12),
        ("CEU00000000C2", 1981, 11),
        ("CEU00000000C2", 2003, 12),
        ("CEU00000000C3", 1981, 11),
        ("CEU00000000C3", 2003, 12),
    ]
    assert frame["source"].unique().to_list() == [COPY]
    first_of_2003 = frame.filter(pl.col("reference_month") == date(2003, 1, 1))
    assert sorted(first_of_2003.select("series_id", "value").rows()) == [
        ("CEU00000000C1", 55.1),
        ("CEU00000000C2", 75.1),
        ("CEU00000000C3", 95.1),
    ]
    december_2003 = frame.filter(
        pl.col("series_id") == "CEU00000000C3",
        pl.col("reference_month") == date(2003, 12, 1),
    )
    assert december_2003["value"].to_list() == [96.2]
    assert date(1981, 7, 1) not in frame["reference_month"].to_list()
    assert frame.schema == SCHEMA


def test_registry_page_values_ignore_numbers_and_section_names_in_prose():
    frame = registry_page_values(synthetic_page(), COPY)
    assert set(frame["reference_month"].dt.year().to_list()) == {1981, 2003}
    assert frame.filter(pl.col("series_id") == "CEU00000000C1").height == 23
    assert 99.9 not in frame["value"].to_list()


def test_registry_page_without_three_sections_is_refused():
    page = synthetic_page()
    without_third = page[: page.index('<tr><td colspan="13"><strong>Third')] + (
        "</table></body></html>"
    )
    with pytest.raises(ValueError, match="third"):
        registry_page_values(without_third, COPY)
    prose_only = (
        "<p>The first preliminary, second preliminary and third rates.</p>"
        "<p>2003 65.2 70.1 80.3</p>"
    )
    with pytest.raises(ValueError, match="third"):
        registry_page_values(prose_only, COPY)


def test_collection_rates_combine_labstat_and_each_copy(tmp_path):
    (tmp_path / "bls").mkdir()
    (tmp_path / raw.COLLECTION_RATES).write_text(
        LABSTAT_HEADER + "CEU00000000C1                 \t2003\tM01\t        55.1\t\n",
        encoding="utf-8",
    )
    copies = tmp_path / raw.REGISTRY_COPIES
    copies.mkdir(parents=True)
    for timestamp in ("20100115120000", "20111230120000"):
        (copies / f"{timestamp}.htm").write_text(synthetic_page(), encoding="utf-8")
    frame = build_collection_rates(tmp_path)
    assert frame["source"].unique().sort().to_list() == [
        "20100115120000",
        "20111230120000",
        LABSTAT_SOURCE,
    ]
    assert frame.schema == SCHEMA
    assert frame.equals(frame.sort("source", "series_id", "reference_month"))


def rate_rows(rows: list[tuple[str, date, str, float]]) -> pl.DataFrame:
    return pl.DataFrame([(*row, None) for row in rows], schema=SCHEMA, orient="row")


def comparison_rates() -> pl.DataFrame:
    c1, c2, rr = "CEU00000000C1", "CEU00000000C2", "CEU05000000RR"
    return rate_rows(
        [
            (c1, date(2010, 1, 1), LABSTAT_SOURCE, 70.5),
            (c1, date(2010, 2, 1), LABSTAT_SOURCE, 70.5),
            (c2, date(2010, 1, 1), LABSTAT_SOURCE, 80.0),
            (rr, date(2010, 1, 1), LABSTAT_SOURCE, 1.0),
            (c1, date(2010, 1, 1), COPY, 70.50),
            (c1, date(2010, 2, 1), COPY, 70.4),
            (c2, date(2010, 1, 1), COPY, 80.0),
            (c2, date(1980, 12, 1), COPY, 79.0),
            (rr, date(2010, 1, 1), COPY, 2.0),
            (c1, date(2010, 1, 1), "20111230120000", 70.5),
        ]
    )


def test_archive_comparison_counts_values_by_tenths():
    comparison = archive_comparison(comparison_rates())
    assert comparison.rows() == [
        (COPY, 3, 1, 1),
        ("20111230120000", 1, 0, 0),
    ]
    assert comparison.columns == ["source", "compared", "differing", "only_in_copy"]
    assert comparison.schema["compared"] == pl.Int64
    assert comparison.schema["differing"] == pl.Int64
    assert comparison.schema["only_in_copy"] == pl.Int64


def test_archive_differences_list_each_changed_value():
    differences = archive_differences(comparison_rates())
    assert differences.columns == [
        "series_id",
        "reference_month",
        "source",
        "copy_value",
        "labstat_value",
    ]
    assert differences.rows() == [("CEU00000000C1", date(2010, 2, 1), COPY, 70.4, 70.5)]


def test_a_note_row_inside_the_table_starts_no_section():
    page = synthetic_page()
    first_2003 = page.index("<tr><th>2003</th><td>55.1")
    note = (
        '<tr><td colspan="13">Note: the second preliminary and third rates '
        "follow below.</td></tr>\n"
    )
    frame = registry_page_values(page[:first_2003] + note + page[first_2003:], COPY)
    c1_2003 = frame.filter(
        pl.col("series_id") == "CEU00000000C1",
        pl.col("reference_month").dt.year() == 2003,
    )
    assert c1_2003.height == 12


def test_archive_comparison_rounds_both_values_to_one_decimal():
    c1 = "CEU00000000C1"
    rates = rate_rows(
        [
            (c1, date(2010, 1, 1), LABSTAT_SOURCE, 70.5),
            (c1, date(2010, 2, 1), LABSTAT_SOURCE, 70.5),
            (c1, date(2010, 1, 1), COPY, 70.54),
            (c1, date(2010, 2, 1), COPY, 70.56),
        ]
    )
    assert archive_comparison(rates).rows() == [(COPY, 2, 1, 0)]
    assert archive_differences(rates)["reference_month"].to_list() == [date(2010, 2, 1)]


def test_a_year_row_before_any_section_gives_no_value():
    page = synthetic_page()
    first_section = page.index('<tr><td colspan="13"><strong>First')
    stray = "<tr><th>1999</th>" + "<td>12.3</td>" * 12 + "</tr>\n"
    frame = registry_page_values(
        page[:first_section] + stray + page[first_section:], COPY
    )
    assert 1999 not in frame["reference_month"].dt.year().to_list()
    assert frame["series_id"].null_count() == 0


def test_a_registry_table_inside_a_layout_table_gives_its_values():
    page = (FIXTURES / "cesregrec-20191212004825-excerpt.htm").read_text(
        encoding="utf-8"
    )
    values = registry_page_values(page, "20191212004825")
    counts = dict(values.group_by("series_id").len().iter_rows())
    assert counts == {"CEU00000000C1": 24, "CEU00000000C2": 21, "CEU00000000C3": 23}
    value = {
        (series, month): number
        for series, month, number in values.select(
            "series_id", "reference_month", "value"
        ).iter_rows()
    }
    assert value["CEU00000000C1", date(1981, 1, 1)] == 34.9
    assert value["CEU00000000C1", date(1982, 12, 1)] == 45.3
    assert value["CEU00000000C2", date(1981, 3, 1)] == 74.3
    assert ("CEU00000000C2", date(1982, 5, 1)) not in value
    assert value["CEU00000000C3", date(1982, 12, 1)] == 87.9


C1 = "CEU00000000C1"


def labstat_c1() -> pl.DataFrame:
    rates = vintage_data.collection_rates()
    return rates.filter(
        (pl.col("source") == LABSTAT_SOURCE) & (pl.col("series_id") == C1)
    )


def test_labstat_c1_runs_from_1981_without_october_2025():
    months = labstat_c1()["reference_month"]
    assert months.min() == date(1981, 1, 1)
    assert months.max() >= date(2026, 10, 1)
    assert date(2025, 10, 1) not in months.to_list()


def test_footnote_e_marks_september_and_november_2025():
    marked = labstat_c1().filter(pl.col("footnote_code") == "E")["reference_month"]
    assert date(2025, 9, 1) in marked.to_list()
    assert date(2025, 11, 1) in marked.to_list()


def test_c1_matches_the_recon_values():
    c1 = labstat_c1()
    value = dict(c1.select("reference_month", "value").iter_rows())
    assert value[date(2013, 9, 1)] == 77.9
    assert value[date(2013, 10, 1)] == 83.5
    assert value[date(2025, 11, 1)] == 73.8
    assert value[date(2026, 1, 1)] == 69.5
    means = dict(
        c1.group_by(pl.col("reference_month").dt.year().alias("year"))
        .agg(pl.col("value").mean())
        .iter_rows()
    )
    assert round(means[2003], 1) == 65.3
    assert round(means[2013], 1) == 77.3
    assert round(means[2024], 1) == 60.4


def test_the_2025_copy_matches_labstat():
    rates = vintage_data.collection_rates()
    copy = rates.filter(pl.col("source") == RECON_COPY)
    counts = dict(copy.group_by("series_id").len().iter_rows())
    assert counts == {C1: 530, "CEU00000000C2": 526, "CEU00000000C3": 527}
    assert copy["reference_month"].min() == date(1981, 1, 1)
    assert copy["reference_month"].max() == date(2025, 2, 1)
    assert archive_differences(rates).filter(pl.col("source") == RECON_COPY).is_empty()


def test_check_refuses_without_a_built_panel(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(first_closing, "PANEL_DIR", tmp_path)
    assert first_closing.main(["check"]) == 1
    assert "vintage_sources.py build" in capsys.readouterr().err


def test_check_prints_each_difference_and_exits_1(monkeypatch, tmp_path, capsys):
    rates = vintage_data.collection_rates()
    raised = (
        (pl.col("source") == RECON_COPY)
        & (pl.col("series_id") == C1)
        & (pl.col("reference_month") == date(2013, 9, 1))
    )
    rates.with_columns(
        value=pl.when(raised).then(pl.col("value") + 0.1).otherwise(pl.col("value"))
    ).write_parquet(tmp_path / "collection_rates.parquet")
    monkeypatch.setattr(first_closing, "PANEL_DIR", tmp_path)
    assert first_closing.main(["check"]) == 1
    printed = capsys.readouterr().out
    assert f"{C1} 2013-09 {RECON_COPY}" in printed


def test_check_accepts_only_the_differences_the_owner_ruled_on(
    monkeypatch, tmp_path, capsys
):
    vintage_data.collection_rates().write_parquet(tmp_path / "collection_rates.parquet")
    monkeypatch.setattr(first_closing, "PANEL_DIR", tmp_path)
    assert first_closing.main(["check"]) == 0
    printed = capsys.readouterr().out
    assert (
        f"{C1} 2012-06 20121114143627: 64.6 against LABSTAT 72.9, accepted" in printed
    )
    assert "20121114143627: 1139 compared, 1 differ, 0 only in the copy" in printed
    monkeypatch.setattr(first_closing, "ACCEPTED_DIFFERENCES", frozenset())
    assert first_closing.main(["check"]) == 1


COMPARED = {
    "20081204164446": 995,
    "20090831195859": 1022,
    "20100615144206": 1052,
    "20111205012811": 1106,
    "20121114143627": 1139,
    "20131109043233": 1175,
    "20141110094945": 1211,
    "20150905111214": 1241,
    "20161214012544": 1286,
    "20170523103601": 1301,
    "20180719105627": 1343,
    "20191212004825": 1394,
    "20201219101554": 1430,
    "20211228210841": 1466,
    "20221209230559": 1502,
    "20231207233658": 1535,
    "20241231185714": 1574,
    "20250319165303": 1583,
}


def test_the_only_archived_difference_is_the_one_the_owner_accepted():
    rates = vintage_data.collection_rates()
    differences = {
        (series, month, source, round(copy_value, 1), round(labstat_value, 1))
        for series, month, source, copy_value, labstat_value in archive_differences(
            rates
        ).iter_rows()
    }
    assert differences == first_closing.ACCEPTED_DIFFERENCES
    committed = sorted(
        path.stem for path in (raw.RAW_DIR / raw.REGISTRY_COPIES).glob("*.htm")
    )
    comparison = archive_comparison(rates)
    assert comparison["source"].to_list() == committed
    assert dict(comparison.select("source", "compared").iter_rows()) == COMPARED
    assert comparison["only_in_copy"].sum() == 0
