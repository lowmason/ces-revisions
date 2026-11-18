"""The share of SA revision variance that comes from the change in seasonal adjustment (O9).

Within a group, `Var(r^S) = Var(r^N) + Var(Δa) − 2 Cov(r^N, Δa)` (the identity of
`ces_revisions.vintages.accounting`). With V = `var_revision_sa` and C the covariance of the NSA
revision with the adjustment change, three shares describe it:

- `share_ratio = Var(Δa) / V`, the plain ratio.
- `share_adjustment = (Var(Δa) − C) / V`, the covariance split, which is `Cov(r^S, −Δa) / V`.
- `share_nsa = (Var(r^N) − C) / V`, its complement.

The two splits add to `1 − variance_identity_gap / V`, so together they partition `Var(r^S)`. None
of the three is bounded in [0, 1]: a covariance can make one term of a split negative, and the plain
ratio ignores the covariance. Rows are never clipped, dropped or flagged for their value.
"""

from datetime import date

import polars as pl

from ces_revisions.vintages.accounting import REGIMES, TRANSITIONS


def _regime_label(first: date, last: date) -> str:
    if last.year == date.max.year:
        return f"{first:%Y-%m} on"
    return f"{first:%Y-%m} to {last:%Y-%m}"


REGIME_LABELS = {name: _regime_label(first, last) for name, first, last in REGIMES}

_TRANSITION_RANK = {
    f"{earlier}_{later}": rank for rank, (earlier, later) in enumerate(TRANSITIONS)
}
_REGIME_RANK = {name: rank for rank, (name, _, _) in enumerate(REGIMES)}


def seasonal_shares(table: pl.DataFrame) -> pl.DataFrame:
    """The decomposition with its three shares, in transition, sector and regime order.

    A group whose SA revision variance is null or not above 0 has no share, and raises.
    """
    refused = table.filter(~(pl.col("var_revision_sa").fill_null(0) > 0))
    if refused.height:
        row = refused.row(0, named=True)
        raise ValueError(
            f"{row['source']} {row['sector']} {row['transition']} {row['regime']}: "
            "SA revision variance is null or not above 0, so it has no share"
        )
    variance = pl.col("var_revision_sa")
    covariance = pl.col("cov_revision_nsa_adjustment_change")
    return table.with_columns(
        share_ratio=pl.col("var_adjustment_change") / variance,
        share_adjustment=(pl.col("var_adjustment_change") - covariance) / variance,
        share_nsa=(pl.col("var_revision_nsa") - covariance) / variance,
    ).sort(
        pl.col("transition").replace_strict(_TRANSITION_RANK),
        "sector",
        pl.col("regime").replace_strict(_REGIME_RANK),
    )
