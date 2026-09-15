"""The committed sources and the immutable raw-value table of Req 1."""

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw

FIXTURES = Path(__file__).parent / "fixtures" / "vintages"


def excerpt_cells() -> dict[tuple[str, str], str]:
    page = (FIXTURES / "cesnaicsrev-excerpt.htm").read_text(encoding="utf-8")
    frame = raw.revision_table_cells(page)
    return {
        (row_key, column_key): text
        for row_key, column_key, text in frame.select(
            "row_key", "column_key", "text"
        ).iter_rows()
    }


def member_rows(member: str) -> list[list[str]]:
    with (
        zipfile.ZipFile(raw.RAW_DIR / raw.VINTAGE_FILES) as archive,
        archive.open(member) as handle,
    ):
        return list(csv.reader(io.TextIOWrapper(handle, encoding="utf-8")))


def test_vintage_file_members_are_total_nonfarm_and_supersectors_both_ways():
    members = raw.vintage_file_members()
    assert len(members) == 24
    assert members[:2] == ["tri_000000_NSA.csv", "tri_000000_SA.csv"]
    assert raw.SUPERSECTORS == (
        "10",
        "20",
        "30",
        "40",
        "50",
        "55",
        "60",
        "65",
        "70",
        "80",
        "90",
    )


def test_triangle_cells_key_each_nonempty_cell_by_release_row_and_month_column():
    payload = b"year,month,Jan_39,Feb_39\n2003,5,29296,-1\n2003,6,,29394\n"
    cells = raw.triangle_cells(payload, "tri_000000_NSA.csv")
    assert cells.select("file", "row_key", "column_key", "text").rows() == [
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-05", "Jan_39", "29296"),
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-05", "Feb_39", "-1"),
        ("bls/cesvinall.zip:tri_000000_NSA.csv", "2003-06", "Feb_39", "29394"),
    ]


def test_revision_table_cells_read_months_averages_and_summaries():
    cells = excerpt_cells()
    assert cells[("2003-03", "sa_3rd")] == "0"
    assert cells[("2003-03", "sa_3rd_minus_2nd")] == "NA***"
    assert cells[("2003-03", "nsa_3rd_minus_2nd")] == "NA"
    assert cells[("2025-10", "sa_2nd")] == "-105 (B)"
    assert cells[("2025-09", "nsa_2nd")] == "-(A)"
    assert cells[("2003:mean_absolute", "nsa_3rd_minus_1st")] == "55"
    assert cells[("2003:mean_absolute", "sa_1st")] == ""
    assert cells[("2025:mean_absolute", "sa_2nd_minus_1st")] == "31(c)"
    assert cells[("summary:mean_absolute:2003 - present", "sa_3rd_minus_1st")] == "51"
    assert cells[("summary:mean:Total All Periods", "nsa_2nd_minus_1st")] == "3"


def test_an_unrecognized_revision_table_row_is_an_error():
    page = (
        "<table><caption>Nonfarm Payroll Employment: Revisions between over-the-month "
        "estimates, 2003</caption><tr><th>Annual</th><td>2003</td>"
        + "<td>1</td>" * 12
        + "</tr></table>"
    )
    with pytest.raises(ValueError, match="unrecognized revision table row 'Annual'"):
        raw.revision_table_cells(page)


def test_rtdsm_cells_keep_every_nonempty_cell_as_text():
    wide = pl.DataFrame(
        {
            "DATE": ["1979:01", "1979:02"],
            "EMPLOY79M2": ["88808", "#N/A"],
            "EMPLOY79M3": ["88810", None],
        }
    )
    assert raw.rtdsm_cells(wide).select("row_key", "column_key", "text").rows() == [
        ("1979:01", "EMPLOY79M2", "88808"),
        ("1979:02", "EMPLOY79M2", "#N/A"),
        ("1979:01", "EMPLOY79M3", "88810"),
    ]


def test_content_sha256_hashes_the_csv_serialization_in_chunks():
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = hashlib.sha256(b"a,b\n1,x\n2,y\n").hexdigest()
    assert raw.content_sha256(frame, chunk_rows=1) == expected


# --- The committed sources ------------------------------------------------------------------


def test_raw_values_hold_every_nonempty_cell_of_the_twelve_vintage_files():
    expected = sum(
        1
        for member in raw.vintage_file_members()
        for row in member_rows(member)[1:]
        for text in row[2:]
        if text
    )
    held = vintage_data.raw_values().filter(pl.col("source") == "cesvinall")
    assert held.height == expected


def test_raw_values_match_a_vintage_file_cell_for_cell():
    member = "tri_900000_SA.csv"
    header, *rows = member_rows(member)
    expected = {
        (f"{row[0]}-{int(row[1]):02d}", column, text)
        for row in rows
        for column, text in zip(header[2:], row[2:], strict=True)
        if text
    }
    held = vintage_data.raw_values().filter(
        pl.col("file") == f"{raw.VINTAGE_FILES}:{member}"
    )
    assert set(held.select("row_key", "column_key", "text").iter_rows()) == expected


def test_the_twelve_vintage_files_hold_whole_thousands():
    texts = vintage_data.raw_values().filter(pl.col("source") == "cesvinall")["text"]
    assert texts.str.contains(r"^-?\d+$").all()


def test_revision_table_cells_cover_every_month_and_average_row():
    cells = vintage_data.raw_values().filter(pl.col("source") == "cesnaicsrev")
    months = cells.filter(pl.col("row_key").str.contains(r"^\d{4}-\d{2}$"))
    years = sorted({key[:4] for key in months["row_key"]})
    assert years[0] == "1979"
    assert months.height == len(years) * 12 * 12
    averages = cells.filter(pl.col("row_key").str.contains(r"^\d{4}:mean"))
    assert averages.height == len(years) * 2 * 12
    assert cells.filter(pl.col("row_key").str.starts_with("summary:")).height == 36


def test_rtdsm_cells_are_keyed_by_observation_month_and_vintage():
    cells = vintage_data.raw_values().filter(pl.col("source") == "rtdsm_employ")
    assert cells["row_key"].str.contains(r"^\d{4}:\d{2}$").all()
    assert cells["column_key"].str.contains(r"^EMPLOY\d{2}M\d{1,2}$").all()


def test_cell_ids_number_the_raw_values_from_zero():
    ids = vintage_data.raw_values()["cell_id"]
    assert (ids.min(), ids.max(), ids.n_unique()) == (0, ids.len() - 1, ids.len())
