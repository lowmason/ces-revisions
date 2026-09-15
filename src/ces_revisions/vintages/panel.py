"""The long vintage panel of Req 1, and the named transformations that read it from raw cells.

Every panel row keeps the cell_id of the raw cell it was read from and the name of the
transformation that read it; TRANSFORMATIONS describes each one with its parameters. The panel
holds three sources: vintage-file levels for total nonfarm and the eleven supersectors, the
Philadelphia Fed's seasonally adjusted total nonfarm level vintages, and the revision table's
total nonfarm estimates of over-the-month change, which reach back to January 1979.
"""

import json
from datetime import date

import polars as pl

from ces_revisions.vintages.raw import SECTORS
from ces_revisions.vintages.revision_table import estimates, monthly_cells

# Req 4's composite regime shift: releases from the May 2003 publication vintage on.
REGIME_SHIFT_MONTH = date(2003, 5, 1)
VINTAGE_FILE_YEAR_PIVOT = 39  # column "Jan_39" is January 1939, "Jan_26" January 2026
RTDSM_YEAR_PIVOT = 64  # column "EMPLOY64M12" is December 1964, "EMPLOY26M8" August 2026
MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
STAGE_OFFSETS = {"F": "0mo", "S": "1mo", "T": "2mo"}

TRANSFORMATIONS = pl.DataFrame(
    [
        {
            "transformation": "vintage_file_level",
            "source": "cesvinall",
            "description": (
                "A vintage-file cell's text is a level in whole thousands. The member name "
                "gives the sector and seasonal status, the row key the release month, and the "
                "column key the reference month, two-digit years from the pivot on read as "
                "19xx and earlier ones as 20xx."
            ),
            "parameters": json.dumps(
                {"sectors": SECTORS, "year_pivot": VINTAGE_FILE_YEAR_PIVOT}
            ),
        },
        {
            "transformation": "vintage_file_lapse_sentinel",
            "source": "cesvinall",
            "description": (
                "The vintage files write -1 where the 2025 lapse left an estimate unpublished: "
                "September 2025's second and October 2025's first, in the row for the "
                "canceled October 2025 release. The level is missing."
            ),
            "parameters": json.dumps({"sentinel": "-1"}),
        },
        {
            "transformation": "rtdsm_level",
            "source": "rtdsm_employ",
            "description": (
                "An EMPLOY cell's text is a seasonally adjusted total nonfarm level in "
                "thousands. Column EMPLOYyyMm is the vintage of calendar month m of year yy, "
                "which holds the Employment Situation release made in that month or, when "
                "none was, the latest earlier one. #N/A cells and vintages before the first "
                "release in the release-date index are not read."
            ),
            "parameters": json.dumps(
                {"missing": "#N/A", "year_pivot": RTDSM_YEAR_PIVOT}
            ),
        },
        {
            "transformation": "revision_table_estimate",
            "source": "cesnaicsrev",
            "description": (
                "A first, second, or third estimate of the total nonfarm over-the-month "
                "change in thousands, published by the release zero, one, or two months after "
                "the reference month's own. A (B) or (c) marker is kept beside the value."
            ),
            "parameters": json.dumps({"release_offsets": STAGE_OFFSETS}),
        },
        {
            "transformation": "revision_table_missing_estimate",
            "source": "cesnaicsrev",
            "description": (
                "An estimate the page marks unavailable after the 2025 lapse (A) or leaves "
                "out for the 2003 redesign (NA), including the zeros it prints where every "
                "revision of the estimate is NA. The change is missing."
            ),
            "parameters": json.dumps({"markers": ["A", "NA"]}),
        },
    ]
)


def _regime() -> pl.Expr:
    return (
        pl.when(pl.col("release_month") < REGIME_SHIFT_MONTH)
        .then(pl.lit("pre_2003_05"))
        .otherwise(pl.lit("from_2003_05"))
    )


def _year(two_digit: pl.Expr, pivot: int) -> pl.Expr:
    return (
        pl.when(two_digit >= pivot).then(two_digit + 1900).otherwise(two_digit + 2000)
    )


def _published(frame: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    releases = index.select(
        release_month="reference_month", release_date="published_date"
    )
    return frame.join(releases, on="release_month", how="left")


def vintage_file_levels(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """Every vintage-file level of total nonfarm and the eleven supersectors."""
    two_digit = pl.col("column_key").str.slice(4, 2).cast(pl.Int32)
    month = (
        pl.col("column_key")
        .str.slice(0, 3)
        .replace_strict(
            {name: number for number, name in enumerate(MONTH_ABBREVIATIONS, start=1)},
            return_dtype=pl.Int32,
        )
    )
    sentinel = pl.col("text") == "-1"
    levels = raw.filter(pl.col("source") == "cesvinall").select(
        "cell_id",
        source=pl.lit("cesvinall"),
        sector=pl.col("file")
        .str.extract(r"tri_(\d{6})_N?SA\.csv$", 1)
        .replace_strict(SECTORS),
        reference_month=pl.date(_year(two_digit, VINTAGE_FILE_YEAR_PIVOT), month, 1),
        release_month=(pl.col("row_key") + "-01").str.to_date("%Y-%m-%d"),
        seasonal_status=pl.col("file").str.extract(r"tri_\d{6}_(N?SA)\.csv$", 1),
        measure=pl.lit("level"),
        value_thousands=pl.when(sentinel)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("text").cast(pl.Int64)),
        marker=pl.when(sentinel)
        .then(pl.lit("lapse_sentinel"))
        .otherwise(pl.lit(None, dtype=pl.String)),
        transformation=pl.when(sentinel)
        .then(pl.lit("vintage_file_lapse_sentinel"))
        .otherwise(pl.lit("vintage_file_level")),
    )
    return _published(levels, index).with_columns(
        vintage_id=pl.format(
            "cesvinall:{}", pl.col("release_month").dt.strftime("%Y-%m")
        ),
        concept_regime=_regime(),
    )


def rtdsm_levels(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """The Philadelphia Fed's seasonally adjusted total nonfarm level vintages from 1979."""
    two_digit = pl.col("column_key").str.extract(r"^EMPLOY(\d{2})M", 1).cast(pl.Int32)
    month = pl.col("column_key").str.extract(r"M(\d{1,2})$", 1).cast(pl.Int32)
    levels = raw.filter(
        (pl.col("source") == "rtdsm_employ") & (pl.col("text") != "#N/A")
    ).select(
        "cell_id",
        source=pl.lit("rtdsm_employ"),
        sector=pl.lit("00"),
        reference_month=(pl.col("row_key").str.replace(":", "-") + "-01").str.to_date(
            "%Y-%m-%d"
        ),
        vintage_month=pl.date(_year(two_digit, RTDSM_YEAR_PIVOT), month, 1),
        vintage_id=pl.concat_str([pl.lit("rtdsm:"), pl.col("column_key")]),
        seasonal_status=pl.lit("SA"),
        measure=pl.lit("level"),
        value_thousands=pl.col("text").cast(pl.Int64),
        marker=pl.lit(None, dtype=pl.String),
        transformation=pl.lit("rtdsm_level"),
    )
    releases = (
        index.filter(pl.col("status") == "released")
        .select(
            release_month="reference_month",
            release_date="release_date",
            released_in=pl.col("release_date").dt.truncate("1mo"),
        )
        .sort("released_in")
    )
    vintages = (
        levels.select("vintage_month")
        .unique()
        .sort("vintage_month")
        .join_asof(
            releases,
            left_on="vintage_month",
            right_on="released_in",
            strategy="backward",
        )
        .drop_nulls("release_month")
        .drop("released_in")
    )
    return (
        levels.join(vintages, on="vintage_month", how="inner")
        .drop("vintage_month")
        .with_columns(concept_regime=_regime())
    )


def revision_table_changes(raw: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
    """The revision table's total nonfarm estimates, each keyed by the release that made it."""
    missing = pl.col("marker").is_in(["A", "NA"])
    changes = (
        estimates(monthly_cells(raw))
        .filter(pl.col("marker").fill_null("") != "blank")
        .select(
            "cell_id",
            "reference_month",
            "seasonal_status",
            "release_stage",
            source=pl.lit("cesnaicsrev"),
            sector=pl.lit("00"),
            release_month=pl.col("reference_month").dt.offset_by(
                pl.col("release_stage").replace_strict(STAGE_OFFSETS)
            ),
            measure=pl.lit("change"),
            value_thousands="value",
            marker="marker",
            transformation=pl.when(missing)
            .then(pl.lit("revision_table_missing_estimate"))
            .otherwise(pl.lit("revision_table_estimate")),
        )
    )
    return _published(changes, index).with_columns(
        vintage_id=pl.format(
            "cesnaicsrev:{}", pl.col("release_month").dt.strftime("%Y-%m")
        ),
        concept_regime=_regime(),
    )
