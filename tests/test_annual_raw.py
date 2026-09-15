"""Stage 4 source cells and transformations."""

from pathlib import Path

import polars as pl
import pytest

from ces_revisions.annual import html_tables, raw


def test_annual_paths_do_not_mutate_stage_3s_archive():
    assert raw.RAW_DIR == raw.ROOT / "data" / "annual" / "raw"
    assert raw.PANEL_DIR == raw.ROOT / "data" / "annual" / "panel"
    assert raw.RAW_DIR != raw.ROOT / "data" / "raw"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-861", -861.0),
        ("−862", -862.0),
        ("83,200", 83200.0),
        ("< 0.05", 0.05),
        ("Not yet published", None),
        ("Not applicable", None),
        ("", None),
    ],
)
def test_parse_number_preserves_missingness_and_normalizes_printed_numbers(
    text, expected
):
    assert raw.parse_number(text) == expected


def test_cell_keys_are_stable_and_include_the_printed_location():
    assert raw.cell_key("bls/cestn.htm", "table_5", "2025", "final_thousands") == (
        "bls/cestn.htm::table_5::2025::final_thousands"
    )


def test_raw_frame_rejects_duplicate_cell_keys():
    records = [
        {
            "source": "bls",
            "file": "bls/cestn.htm",
            "table_key": "table_5",
            "row_key": "2025",
            "column_key": "final_thousands",
            "text": "-861",
        }
    ]
    with pytest.raises(ValueError, match="duplicate raw cell keys"):
        raw.raw_frame(records + records)


def test_transformations_name_every_non_identity_derivation():
    names = set(raw.TRANSFORMATIONS["transformation"])
    assert names == {
        "identity",
        "parse_published_number",
        "parse_thousands",
        "benchmark_article_comparison",
        "qcew_jobs_to_thousands",
        "successive_difference",
        "rms_qcew_revision",
        "realized_minus_forecast",
        "government_structural_zero",
        "explicit_archive_gap",
    }


def test_manifest_frame_hashes_every_file_but_the_manifest(tmp_path: Path):
    (tmp_path / "bls").mkdir()
    (tmp_path / "bls" / "page.htm").write_text("source", encoding="utf-8")
    (tmp_path / "source-catalog.csv").write_text("source_id\npage\n", encoding="utf-8")
    frame = raw.manifest_frame(tmp_path, fetched_at="2026-09-15T12:00:00Z")
    assert frame["file"].to_list() == ["bls/page.htm", "source-catalog.csv"]
    assert frame["sha256"].str.len_chars().eq(64).all()
    assert frame.schema["bytes"] == pl.Int64


def test_html_tables_keep_caption_rows_and_nested_text():
    page = """
    <table><caption>Table 5. Benchmark <em>revisions</em></caption>
      <tr><th>Year</th><th>Final</th></tr>
      <tr><td>2025<sup>(12)</sup></td><td>−861</td></tr>
    </table>
    """
    table = html_tables.find_table(page, r"Table 5\. Benchmark")
    assert table.caption == "Table 5. Benchmark revisions"
    assert table.rows == (("Year", "Final"), ("2025 (12)", "−861"))


def test_find_table_requires_exactly_one_matching_caption():
    page = "<table><caption>A</caption></table><table><caption>A</caption></table>"
    with pytest.raises(ValueError, match="expected one table"):
        html_tables.find_table(page, "A")


def test_table_raw_cells_preserve_printed_text():
    table = html_tables.HtmlTable("Table 1", (("Industry", "Coverage"), ("00", "26")))
    cells = html_tables.table_raw_cells(
        table, source="bls", file="bls/cestn.htm", table_key="coverage_2025"
    )
    assert cells.select("row_key", "column_key", "text").rows() == [
        ("0", "0", "Industry"),
        ("0", "1", "Coverage"),
        ("1", "0", "00"),
        ("1", "1", "26"),
    ]
