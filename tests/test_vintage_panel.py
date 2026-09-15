"""The long vintage panel of Req 1: transformations, lineage, sums, and the aggregate leg."""

import json
import re
from datetime import date

import polars as pl
import vintage_data

from ces_revisions.vintages import panel, raw
from ces_revisions.vintages.revision_table import parse_cell

KEY = [
    "source",
    "sector",
    "reference_month",
    "vintage_id",
    "seasonal_status",
    "measure",
]
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")
MONTHS += ("Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
STAGE_OF_COLUMN = {"1st": ("F", 0), "2nd": ("S", 1), "3rd": ("T", 2)}


def month_after(month: date, count: int) -> date:
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def rebuilt_key(file: str, row_key: str, column_key: str) -> tuple:
    """A panel row's key from its raw cell's keys, derived apart from the panel module."""
    if file.startswith(raw.VINTAGE_FILES):
        code, status = re.search(r"tri_(\d{6})_(N?SA)\.csv$", file).groups()
        two_digit = int(column_key[4:])
        year = 1900 + two_digit if two_digit >= 39 else 2000 + two_digit
        month = date(year, MONTHS.index(column_key[:3]) + 1, 1)
        return raw.SECTORS[code], month, status, date.fromisoformat(f"{row_key}-01")
    if file.startswith(raw.RTDSM_LEVELS):
        return "00", date.fromisoformat(f"{row_key.replace(':', '-')}-01"), "SA"
    status, column = column_key.split("_", 1)
    stage, offset = STAGE_OF_COLUMN[column]
    month = date.fromisoformat(f"{row_key}-01")
    return "00", month, status.upper(), month_after(month, offset), stage


def sources() -> dict[str, pl.DataFrame]:
    return {
        "cesvinall": vintage_data.levels(),
        "rtdsm_employ": vintage_data.rtdsm(),
        "cesnaicsrev": vintage_data.table_changes(),
    }


def test_transformations_name_and_parameterize_every_rule():
    names = panel.TRANSFORMATIONS["transformation"].to_list()
    assert names == [
        "vintage_file_level",
        "vintage_file_lapse_sentinel",
        "rtdsm_level",
        "revision_table_estimate",
        "revision_table_missing_estimate",
    ]
    for parameters in panel.TRANSFORMATIONS["parameters"]:
        assert isinstance(json.loads(parameters), dict)


# --- The committed sources ------------------------------------------------------------------


def test_the_sources_use_every_named_transformation_and_no_other():
    used = set()
    for frame in sources().values():
        used |= set(frame["transformation"].unique())
    assert used == set(panel.TRANSFORMATIONS["transformation"])


def test_source_keys_are_unique():
    for name, frame in sources().items():
        assert frame.select(KEY).n_unique() == frame.height, name


def test_every_value_rebuilds_from_its_raw_cell():
    columns = ["cell_id", "value_thousands", "marker", "transformation"]
    joined = pl.concat([frame.select(columns) for frame in sources().values()]).join(
        vintage_data.raw_values().select("cell_id", "text"), on="cell_id", how="left"
    )
    assert joined["text"].null_count() == 0
    levels = joined.filter(
        pl.col("transformation").is_in(["vintage_file_level", "rtdsm_level"])
    )
    assert (levels["value_thousands"] == levels["text"].cast(pl.Int64)).all()
    sentinels = joined.filter(pl.col("transformation") == "vintage_file_lapse_sentinel")
    assert (sentinels["text"] == "-1").all()
    assert sentinels["value_thousands"].null_count() == sentinels.height
    estimates = joined.filter(pl.col("transformation") == "revision_table_estimate")
    assert estimates["value_thousands"].to_list() == [
        parse_cell(text).value for text in estimates["text"]
    ]
    missing = joined.filter(
        pl.col("transformation") == "revision_table_missing_estimate"
    )
    assert missing["value_thousands"].null_count() == missing.height
    assert set(missing["marker"]) == {"A", "NA"}


def test_keys_rebuild_from_their_raw_cells():
    raw_keys = vintage_data.raw_values().select(
        "cell_id", "file", "row_key", "column_key"
    )
    columns = {
        "cesvinall": ["sector", "reference_month", "seasonal_status", "release_month"],
        "rtdsm_employ": ["sector", "reference_month", "seasonal_status"],
        "cesnaicsrev": [
            "sector",
            "reference_month",
            "seasonal_status",
            "release_month",
            "release_stage",
        ],
    }
    for name, frame in sources().items():
        sample = frame.join(raw_keys, on="cell_id").sort("cell_id").gather_every(97)
        for row in sample.iter_rows(named=True):
            expected = tuple(row[column] for column in columns[name])
            assert (
                rebuilt_key(row["file"], row["row_key"], row["column_key"]) == expected
            )


def test_the_eleven_supersectors_sum_to_total_nonfarm_in_every_release_file():
    """Req 5. BLS publishes these aggregates as exact sums of the supersectors, so a nonzero
    difference is a finding to report, not a tolerance to widen."""
    frame = vintage_data.levels().filter(pl.col("value_thousands").is_not_null())
    keys = ["release_month", "seasonal_status", "reference_month"]
    total = frame.filter(pl.col("sector") == "00").select(
        *keys, total="value_thousands"
    )
    parts = (
        frame.filter(pl.col("sector") != "00")
        .group_by(keys)
        .agg(parts=pl.col("value_thousands").sum(), supersectors=pl.len())
    )
    joined = total.join(parts, on=keys, how="left")
    assert (joined["supersectors"] == 11).all()
    assert (joined["parts"] == joined["total"]).all()
    releases = frame["release_month"].n_unique()
    assert joined.select("release_month", "seasonal_status").n_unique() == 2 * releases


def test_vintage_file_releases_begin_with_the_may_2003_publication_vintage():
    frame = vintage_data.levels()
    assert frame["release_month"].min() == date(2003, 5, 1)
    assert frame["concept_regime"].unique().to_list() == ["from_2003_05"]


def test_the_aggregate_leg_carries_concept_regime():
    frame = pl.concat(
        [
            vintage_data.rtdsm().select("source", "release_month", "concept_regime"),
            vintage_data.table_changes().select(
                "source", "release_month", "concept_regime"
            ),
        ]
    )
    expected = (
        pl.when(pl.col("release_month") < date(2003, 5, 1))
        .then(pl.lit("pre_2003_05"))
        .otherwise(pl.lit("from_2003_05"))
    )
    assert frame["concept_regime"].null_count() == 0
    assert frame.select((pl.col("concept_regime") == expected).all()).item()
    regimes = set(frame.select("source", "concept_regime").unique().iter_rows())
    assert regimes == {
        ("cesnaicsrev", "pre_2003_05"),
        ("cesnaicsrev", "from_2003_05"),
        ("rtdsm_employ", "pre_2003_05"),
        ("rtdsm_employ", "from_2003_05"),
    }


def test_the_canceled_october_2025_row_is_published_with_november_and_holds_sentinels():
    row = vintage_data.levels().filter(pl.col("vintage_id") == "cesvinall:2025-10")
    assert row["release_date"].unique().to_list() == [date(2025, 12, 16)]
    missing = row.filter(pl.col("value_thousands").is_null())
    assert missing.height == 48
    assert sorted(set(missing["reference_month"])) == [
        date(2025, 9, 1),
        date(2025, 10, 1),
    ]


def test_rtdsm_vintages_end_with_the_month_their_release_first_estimates():
    frame = vintage_data.rtdsm()
    vintages = frame.group_by("vintage_id", "release_month").agg(
        last=pl.col("reference_month").max()
    )
    assert (vintages["last"] == vintages["release_month"]).all()
    assert vintages["vintage_id"].n_unique() == vintages.height
    october = frame.filter(pl.col("vintage_id") == "rtdsm:EMPLOY25M10")
    assert october["release_month"].unique().to_list() == [date(2025, 8, 1)]
    assert frame["release_month"].min() == date(1979, 1, 1)


def test_revision_table_changes_are_keyed_to_the_releases_that_made_them():
    rows = {
        (row["reference_month"], row["seasonal_status"], row["release_stage"]): row
        for row in vintage_data.table_changes().iter_rows(named=True)
    }
    october = rows[(date(2025, 10, 1), "SA", "S")]
    assert (
        october["release_month"],
        october["release_date"],
        october["value_thousands"],
        october["marker"],
    ) == (date(2025, 11, 1), date(2025, 12, 16), -105, "B")
    august = rows[(date(2025, 8, 1), "NSA", "T")]
    assert (
        august["release_month"],
        august["release_date"],
        august["value_thousands"],
    ) == (date(2025, 10, 1), date(2025, 12, 16), 185)
    assert rows[(date(1979, 1, 1), "NSA", "F")]["release_date"] == date(1979, 2, 2)


def test_the_revision_table_and_the_release_index_end_with_the_same_release():
    """A fetch that straddles a release's embargo can leave the page a release ahead."""
    first = vintage_data.table_changes().filter(
        (pl.col("release_stage") == "F") & pl.col("value_thousands").is_not_null()
    )
    assert (
        first["reference_month"].max() == vintage_data.index()["reference_month"].max()
    )


# --- Task 6: the assembled panel ------------------------------------------------------------


def test_the_assembled_panel_holds_every_source_row_once_under_req_1_columns():
    frame = vintage_data.long_panel()
    assert frame.columns == panel.PANEL_COLUMNS
    assert frame.select(KEY).n_unique() == frame.height
    assert frame.height == sum(source.height for source in sources().values())
