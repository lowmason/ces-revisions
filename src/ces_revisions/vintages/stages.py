"""Stage labels: each reference month's F, S, T, B, and M vintages under Req 2.

F, S, and T are the releases zero, one, and two months after a reference month on the release
clock. The vintage files keep a row for every month's release, the canceled October 2025 one
included, and BLS's revision table counts estimates the same way. B is the first benchmark
release after T, and M, for not seasonally adjusted values, the second; benchmark releases
carry January estimates from 2004. Counting from T gives every vintage one stage: November's
third estimate and December's second arrive in a benchmark release, which `benchmark_release`
marks. Seasonally adjusted M is the January release six years after the reference year, the
first benchmark release after the month leaves BLS's five-year seasonal revision window.

A stage whose release is still to come is right-censored, a release after the vintage files'
last row is beyond the frontier, and a value the 2025 lapse left unpublished, or an estimate
the revision table omits for the 2003 redesign, is a missing vintage.
"""

import re
from datetime import date

import polars as pl

from ces_revisions.vintages.months import MONTH_NAMES, add_months, month_range
from ces_revisions.vintages.panel import REGIME_SHIFT_MONTH
from ces_revisions.vintages.raw import SEASONAL_STATUSES, SECTORS
from ces_revisions.vintages.release_index import FIRST_REFERENCE_MONTH

STAGES = ("F", "S", "T", "B", "M")
CLOSING_STAGES = ("F", "S", "T")
FIRST_SECTOR_MONTH = date(2003, 5, 1)
STATUSES = ("observed", "missing_vintage", "beyond_frontier", "right_censored")
STAGE_LABEL_COLUMNS = [
    "source",
    "sector",
    "reference_month",
    "seasonal_status",
    "release_stage",
    "release_month",
    "vintage_id",
    "published_date",
    "status",
    "benchmark_release",
    "nonstandard_release",
    "revised_after_m",
    "concept_regime",
]
_KEYS = ["sector", "reference_month", "seasonal_status"]
_RELEASE_OF = re.compile(
    r"^With the release of (" + "|".join(MONTH_NAMES) + r") (\d{4}) data on"
)
_BENCHMARK_OF = re.compile(r"^With the (\d{4}) benchmark")
_LAPSE = re.compile(r"^Due to the 2025 lapse in appropriations")


def stage_release_months(
    reference_month: date, seasonal_status: str
) -> dict[str, date]:
    """The release month of each stage, as the reference month that release first estimates."""
    third = add_months(reference_month, 2)
    mature = (
        date(third.year + 2, 1, 1)
        if seasonal_status == "NSA"
        else date(reference_month.year + 6, 1, 1)
    )
    return {
        "F": reference_month,
        "S": add_months(reference_month, 1),
        "T": third,
        "B": date(third.year + 1, 1, 1),
        "M": mature,
    }


def comment_release_month(adjustment: str) -> date:
    """The release a Comments-sheet entry describes, read from its opening words."""
    if match := _RELEASE_OF.match(adjustment):
        return date(int(match.group(2)), MONTH_NAMES.index(match.group(1)) + 1, 1)
    if match := _BENCHMARK_OF.match(adjustment):
        return date(int(match.group(1)) + 1, 1, 1)
    if _LAPSE.match(adjustment):
        return date(2025, 10, 1)
    raise ValueError(f"no release rule for the comment {adjustment[:60]!r}")


def _calendar(months: list[date], sectors, stages) -> pl.DataFrame:
    records = [
        (sector, month, status, stage, stage_release_months(month, status)[stage])
        for sector in sectors
        for status in SEASONAL_STATUSES
        for month in months
        for stage in stages
    ]
    return pl.DataFrame(
        records,
        schema={
            "sector": pl.String,
            "reference_month": pl.Date,
            "seasonal_status": pl.String,
            "release_stage": pl.String,
            "release_month": pl.Date,
        },
        orient="row",
    )


def _finish(
    labeled: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    commented = sorted({comment_release_month(text) for text in comments["adjustment"]})
    releases = index.select(
        "published_date", "benchmark_release", release_month="reference_month"
    )
    return (
        labeled.join(releases, on="release_month", how="left")
        .with_columns(
            published_date=pl.when(pl.col("status") == "missing_vintage")
            .then(pl.lit(None, dtype=pl.Date))
            .otherwise("published_date"),
            nonstandard_release=pl.col("release_month").is_in(commented),
            concept_regime=pl.when(pl.col("release_month") < REGIME_SHIFT_MONTH)
            .then(pl.lit("pre_2003_05"))
            .otherwise(pl.lit("from_2003_05")),
        )
        .select(STAGE_LABEL_COLUMNS)
    )


def vintage_file_stage_labels(
    levels: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    """Five stage rows per sector, seasonal status, and reference month from May 2003."""
    latest = index["reference_month"].max()
    frontier = levels["release_month"].max()
    calendar = _calendar(
        month_range(FIRST_SECTOR_MONTH, latest), SECTORS.values(), STAGES
    )
    cells = levels.select(*_KEYS, "release_month", "vintage_id", "value_thousands")
    labeled = calendar.join(
        cells, on=[*_KEYS, "release_month"], how="left"
    ).with_columns(
        source=pl.lit("cesvinall"),
        status=pl.when(pl.col("release_month") > latest)
        .then(pl.lit("right_censored"))
        .when(pl.col("release_month") > frontier)
        .then(pl.lit("beyond_frontier"))
        .when(pl.col("value_thousands").is_null())
        .then(pl.lit("missing_vintage"))
        .otherwise(pl.lit("observed")),
    )
    mature = labeled.filter(
        (pl.col("release_stage") == "M") & (pl.col("status") == "observed")
    ).select(*_KEYS, mature_month="release_month", mature_value="value_thousands")
    later = (
        levels.join(mature, on=_KEYS, how="inner")
        .filter(pl.col("release_month") > pl.col("mature_month"))
        .group_by(_KEYS)
        .agg(revised=(pl.col("value_thousands") != pl.col("mature_value")).any())
    )
    revised = mature.join(later, on=_KEYS, how="left").select(
        *_KEYS,
        release_stage=pl.lit("M"),
        # No vintage file follows the frontier, so an M there cannot be checked yet.
        revised_after_m=pl.when(pl.col("mature_month") >= frontier)
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("revised").fill_null(False)),
    )
    return _finish(
        labeled.join(revised, on=[*_KEYS, "release_stage"], how="left"), index, comments
    )


def revision_table_stage_labels(
    changes: pl.DataFrame, index: pl.DataFrame, comments: pl.DataFrame
) -> pl.DataFrame:
    """Three stage rows per seasonal status and reference month of the revision table."""
    latest = index["reference_month"].max()
    calendar = _calendar(
        month_range(FIRST_REFERENCE_MONTH, latest), ["00"], CLOSING_STAGES
    )
    values = changes.select(*_KEYS, "release_stage", "vintage_id", "value_thousands")
    labeled = calendar.join(
        values, on=[*_KEYS, "release_stage"], how="left"
    ).with_columns(
        source=pl.lit("cesnaicsrev"),
        revised_after_m=pl.lit(None, dtype=pl.Boolean),
        status=pl.when(pl.col("release_month") > latest)
        .then(pl.lit("right_censored"))
        .when(pl.col("value_thousands").is_null())
        .then(pl.lit("missing_vintage"))
        .otherwise(pl.lit("observed")),
    )
    return _finish(labeled, index, comments)


def build_stage_labels(
    levels: pl.DataFrame,
    changes: pl.DataFrame,
    index: pl.DataFrame,
    comments: pl.DataFrame,
) -> pl.DataFrame:
    return pl.concat(
        [
            vintage_file_stage_labels(levels, index, comments),
            revision_table_stage_labels(changes, index, comments),
        ]
    ).sort("source", "sector", "seasonal_status", "reference_month", "release_stage")
