"""The Req 9 accounting identity, and the variance decomposition of published revisions.

For same-release changes d^N and d^S at stages j and k of one reference month, with implied
adjustment a = d^N - d^S, the seasonally adjusted revision is the unadjusted revision less the
change in implied adjustment, r^S = r^N - (a_k - a_j). Within any group of reference months,
therefore, Var(r^S) = Var(r^N) + Var(Δa) - 2 Cov(r^N, Δa) exactly.
"""

from datetime import date

import polars as pl

from ces_revisions.vintages.panel import STAGE_OFFSETS

# B to M has no transition: the identity needs each stage's two changes from one release, and
# seasonally adjusted M waits for the five-year seasonal revision window, so unadjusted and
# adjusted M are never one release.
TRANSITIONS = (("F", "S"), ("S", "T"), ("T", "B"))
# Req 18's regimes, by reference month; the pre-2003 regime is the revision table's alone.
REGIMES = (
    ("pre_2003", date(1979, 1, 1), date(2003, 4, 1)),
    ("2003_2019", date(2003, 5, 1), date(2019, 12, 1)),
    ("2020_2022", date(2020, 1, 1), date(2022, 12, 1)),
    ("2023_present", date(2023, 1, 1), date(9999, 12, 1)),
)
_UNIT = ["source", "sector", "reference_month"]


def accounting_changes(
    stage_changes: pl.DataFrame, table_estimates: pl.DataFrame
) -> pl.DataFrame:
    """Observed stage changes: vintage files from May 2003, the table's before May 2003."""
    files = stage_changes.filter(
        (pl.col("status") == "observed") & pl.col("release_stage").is_in(list("FSTB"))
    ).select(
        *_UNIT, "seasonal_status", "release_stage", "release_month", "change_thousands"
    )
    table = table_estimates.filter(
        (pl.col("reference_month") < REGIMES[1][1]) & pl.col("value").is_not_null()
    ).select(
        source=pl.lit("cesnaicsrev"),
        sector=pl.lit("00"),
        reference_month="reference_month",
        seasonal_status="seasonal_status",
        release_stage="release_stage",
        release_month=pl.col("reference_month").dt.offset_by(
            pl.col("release_stage").replace_strict(STAGE_OFFSETS)
        ),
        change_thousands="value",
    )
    return pl.concat([files, table])


def _regime() -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for name, first, last in reversed(REGIMES):
        expression = (
            pl.when(pl.col("reference_month").is_between(first, last))
            .then(pl.lit(name))
            .otherwise(expression)
        )
    return expression


def identity_terms(changes: pl.DataFrame) -> pl.DataFrame:
    """One row per unit and transition whose two stages are observed seasonally adjusted and not.

    The identity is exact only for same-release changes, so a stage whose unadjusted and adjusted
    changes come from different releases is an error.
    """
    per_stage = changes.group_by(*_UNIT, "release_stage").agg(
        releases=pl.col("release_month").n_unique()
    )
    mixed = per_stage.filter(pl.col("releases") > 1).sort(*_UNIT, "release_stage")
    if mixed.height:
        source, sector, month, stage, _ = mixed.row(0)
        raise ValueError(
            f"{source} {sector} {month:%Y-%m} {stage}: unadjusted and adjusted changes "
            "come from different releases"
        )
    wide = changes.pivot(
        on="seasonal_status", index=[*_UNIT, "release_stage"], values="change_thousands"
    )
    parts = []
    for earlier, later in TRANSITIONS:
        start = wide.filter(pl.col("release_stage") == earlier).select(
            *_UNIT, nsa_start="NSA", sa_start="SA"
        )
        end = wide.filter(pl.col("release_stage") == later).select(
            *_UNIT, nsa_end="NSA", sa_end="SA"
        )
        parts.append(
            start.join(end, on=_UNIT, how="inner").with_columns(
                transition=pl.lit(f"{earlier}_{later}")
            )
        )
    return (
        pl.concat(parts)
        .drop_nulls(["nsa_start", "sa_start", "nsa_end", "sa_end"])
        .with_columns(
            revision_nsa=pl.col("nsa_end") - pl.col("nsa_start"),
            revision_sa=pl.col("sa_end") - pl.col("sa_start"),
            adjustment_change=(pl.col("nsa_end") - pl.col("sa_end"))
            - (pl.col("nsa_start") - pl.col("sa_start")),
            regime=_regime(),
        )
        .with_columns(
            identity_residual=pl.col("revision_sa")
            - (pl.col("revision_nsa") - pl.col("adjustment_change"))
        )
    )


def decomposition(terms: pl.DataFrame) -> pl.DataFrame:
    """Variances of the identity's terms by source, sector, transition, and regime."""
    return (
        terms.group_by("source", "sector", "transition", "regime")
        .agg(
            months=pl.len(),
            var_revision_sa=pl.col("revision_sa").var(),
            var_revision_nsa=pl.col("revision_nsa").var(),
            var_adjustment_change=pl.col("adjustment_change").var(),
            cov_revision_nsa_adjustment_change=pl.cov(
                "revision_nsa", "adjustment_change"
            ),
        )
        .with_columns(
            variance_identity_gap=pl.col("var_revision_sa")
            - (
                pl.col("var_revision_nsa")
                + pl.col("var_adjustment_change")
                - 2 * pl.col("cov_revision_nsa_adjustment_change")
            )
        )
        .sort("source", "sector", "transition", "regime")
    )
