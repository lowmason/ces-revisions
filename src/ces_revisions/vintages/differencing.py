"""The same-release differencing operator of Req 1, and its reconciliation with BLS's table.

An over-the-month change is a month's level minus the prior month's level in the same release
file; levels from different vintages are never differenced, and stage labels are never
differenced across release files. Reconciliation sets each total nonfarm estimate in the
revision table beside the same-release change at its stage.
"""

import polars as pl

from ces_revisions.vintages.revision_table import REVISION_TERMS, STAGE_OF_ESTIMATE

_VINTAGE = ["source", "sector", "seasonal_status", "vintage_id", "release_month"]
_CELL = ["reference_month", "seasonal_status", "release_stage"]
UNCOMPARED = ("beyond_frontier", "right_censored")


def same_release_changes(levels: pl.DataFrame) -> pl.DataFrame:
    """Each month's level minus the prior month's in the same vintage; missing if either is."""
    prior = levels.select(
        *_VINTAGE,
        reference_month=pl.col("reference_month").dt.offset_by("1mo"),
        prior_thousands="value_thousands",
    )
    return (
        levels.select(*_VINTAGE, "reference_month", "value_thousands")
        .join(prior, on=[*_VINTAGE, "reference_month"], how="inner")
        .select(
            *_VINTAGE,
            "reference_month",
            change_thousands=pl.col("value_thousands") - pl.col("prior_thousands"),
        )
    )


def stage_changes(changes: pl.DataFrame, stage_labels: pl.DataFrame) -> pl.DataFrame:
    """The same-release change at every vintage-file stage label."""
    return (
        stage_labels.filter(pl.col("source") == "cesvinall")
        .join(
            changes.select(
                "sector",
                "seasonal_status",
                "vintage_id",
                "reference_month",
                "change_thousands",
            ),
            on=["sector", "seasonal_status", "vintage_id", "reference_month"],
            how="left",
        )
        .select(
            "source",
            "sector",
            "reference_month",
            "seasonal_status",
            "release_stage",
            "release_month",
            "vintage_id",
            "status",
            "change_thousands",
        )
    )


def reconcile(
    stage_changes: pl.DataFrame, table_estimates: pl.DataFrame
) -> pl.DataFrame:
    """Each total nonfarm F, S, and T change from May 2003 beside the table's estimate."""
    panel = stage_changes.filter(
        (pl.col("sector") == "00") & pl.col("release_stage").is_in(list("FST"))
    ).select(*_CELL, "status", panel_thousands="change_thousands")
    table = table_estimates.select(
        *_CELL, table_thousands="value", table_marker="marker"
    )
    return panel.join(table, on=_CELL, how="left").with_columns(
        outcome=pl.when(pl.col("status").is_in(UNCOMPARED))
        .then("status")
        .when(pl.col("panel_thousands").is_null() & pl.col("table_thousands").is_null())
        .then(pl.lit("missing_in_both"))
        .when(pl.col("panel_thousands") == pl.col("table_thousands"))
        .then(pl.lit("reproduced"))
        .otherwise(pl.lit("differs"))
    )


def revision_reconciliation(
    stage_changes: pl.DataFrame, table_cells: pl.DataFrame
) -> pl.DataFrame:
    """Each total nonfarm revision between observed stages beside the table's revision cell."""
    wide = stage_changes.filter(
        (pl.col("sector") == "00")
        & pl.col("release_stage").is_in(list("FST"))
        & (pl.col("status") == "observed")
    ).pivot(
        on="release_stage",
        index=["reference_month", "seasonal_status"],
        values="change_thousands",
    )
    panel = pl.concat(
        [
            wide.select(
                "reference_month",
                "seasonal_status",
                column=pl.lit(column),
                panel_thousands=pl.col(STAGE_OF_ESTIMATE[later])
                - pl.col(STAGE_OF_ESTIMATE[earlier]),
            )
            for column, (later, earlier) in REVISION_TERMS.items()
        ]
    ).drop_nulls("panel_thousands")
    table = table_cells.filter(pl.col("column").is_in(list(REVISION_TERMS))).select(
        "reference_month",
        "seasonal_status",
        "column",
        table_thousands="value",
        table_marker="marker",
    )
    return panel.join(
        table, on=["reference_month", "seasonal_status", "column"], how="left"
    ).with_columns(
        outcome=pl.when(pl.col("panel_thousands") == pl.col("table_thousands"))
        .then(pl.lit("reproduced"))
        .otherwise(pl.lit("differs"))
    )
