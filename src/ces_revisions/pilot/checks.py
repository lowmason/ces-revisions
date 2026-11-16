"""Posterior checks of the fitted pilot on the panel's most extreme month.

April 2020 moved total nonfarm by about twenty million jobs, two orders of magnitude more
than any month's revision. The crisis multiplier lets the S and T news grow with the level
shock that month (decision D2), so the smoothed latent change should follow the third print
rather than be pulled toward the trend, and the panel should pin it down rather than leave
it to the prior. ``april_2020`` reads the smoothed change, the smoothed news, and the
published prints there, for a subsample of posterior draws.
"""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np

from ces_revisions.kalman import kalman_filter, kalman_smoother
from ces_revisions.pilot.layout import NEWS_STATES, Layout, seasonal_selector
from ces_revisions.pilot.model import SAMPLED_SITES, pilot_ssm
from ces_revisions.pilot.panel import CLOSING_STAGES, COLUMNS, AggregatePanel
from ces_revisions.pilot.sbc import thin

APRIL_2020 = date(2020, 4, 1)
# Pre-registered: every subsampled draw filters to finite moments, and the smoothed latent
# change's central 90% interval, its mean plus or minus INTERVAL_Z total posterior SDs,
# lies within 5% of the third print. A mean inside the band alone would pass a change the
# panel leaves almost unidentified.
RATIO_RANGE = (0.95, 1.05)
INTERVAL_Z = 1.6448536269514722  # the standard normal's 0.95 quantile
# Draws filtered and smoothed together under lax.map; bounds the (batch, T, n, n) buffers.
BATCH_SIZE = 20


def change_row(layout: Layout) -> np.ndarray:
    """The (n,) row reading the latent change dx_t = level_t + gamma_t - lagged_value_t."""
    names = layout.state_names
    row = seasonal_selector(layout)
    row[names.index("level")] += 1.0
    row[names.index("lagged_value")] -= 1.0
    return row


def news_row(layout: Layout) -> np.ndarray:
    """The (n,) row summing the month's S, T, and post-T news states."""
    row = np.zeros(layout.state_dim)
    row[[layout.state_names.index(name) for name in NEWS_STATES]] = 1.0
    return row


def april_2020(
    samples: dict,
    panel: AggregatePanel,
    layout: Layout,
    *,
    num_draws: int = 200,
    seed: int,
) -> dict:
    """The smoothed latent change and news at April 2020 against the published prints.

    ``samples`` maps every sampled site to its draws, with any leading (chain, draw)
    shape. ``num_draws`` evenly spaced draws of the pooled chains are filtered and
    smoothed. A quantity's ``sd`` is its total posterior SD, the smoother's own variance
    averaged over draws plus the variance of the smoothed means across draws;
    ``sd_between_draws`` is the second part alone. ``seed`` is recorded for the run's
    provenance; the subsample itself is deterministic.
    """
    step = panel.reference_months.index(APRIL_2020)
    flat = {name: np.asarray(samples[name]).reshape(-1) for name in SAMPLED_SITES}
    total = len(flat[SAMPLED_SITES[0]])
    kept = thin(np.arange(total), min(num_draws, total))
    draws = {name: jnp.asarray(values[kept]) for name, values in flat.items()}
    rows = jnp.asarray(np.stack([change_row(layout), news_row(layout)]))
    y = jnp.asarray(panel.y)

    def moments(params):
        ssm = pilot_ssm(params, panel, layout)
        smoothed = kalman_smoother(ssm, kalman_filter(ssm, y))
        mean = rows @ smoothed.smoothed_mean[step]
        variance = jnp.einsum("ri,ij,rj->r", rows, smoothed.smoothed_cov[step], rows)
        return mean, variance

    mapped = jax.jit(lambda params: jax.lax.map(moments, params, batch_size=BATCH_SIZE))
    means, variances = (np.asarray(array) for array in mapped(draws))
    finite = np.isfinite(means).all(axis=1) & np.isfinite(variances).all(axis=1)
    observed = {
        stage: _optional(panel.y[step, COLUMNS.index(stage)])
        for stage in CLOSING_STAGES
    }
    summaries = {
        name: _summary(means[finite, k], variances[finite, k])
        for k, name in enumerate(("dx", "news_sum"))
    }
    third = observed["T"]
    check = (
        band_check(summaries["dx"]["mean"], summaries["dx"]["sd"], third)
        if third is not None and finite.any()
        else {"ratio": None, "interval_90": None, "band": None, "passed": False}
    )
    # A draw the filter cannot evaluate fails the check rather than drop out of it.
    check["passed"] = check["passed"] and bool(finite.all())
    return {
        "reference_month": APRIL_2020.isoformat(),
        "layout": layout.name,
        "seed": int(seed),
        "num_draws": len(kept),
        "draw_indices": kept.tolist(),
        "non_finite_draws": int((~finite).sum()),
        **summaries,
        "observed": observed,
        "ratio_range": list(RATIO_RANGE),
        **check,
    }


def band_check(mean: float, sd: float, third: float) -> dict:
    """Whether mean plus or minus INTERVAL_Z sd lies inside RATIO_RANGE times the print."""
    interval = [mean - INTERVAL_Z * sd, mean + INTERVAL_Z * sd]
    band = sorted(third * bound for bound in RATIO_RANGE)
    return {
        "ratio": mean / third,
        "interval_90": interval,
        "band": band,
        "passed": band[0] <= interval[0] and interval[1] <= band[1],
    }


def _summary(means: np.ndarray, variances: np.ndarray) -> dict:
    if means.size == 0:
        return {"mean": None, "sd": None, "sd_between_draws": None}
    between = float(means.var()) if means.size > 1 else 0.0
    return {
        "mean": float(means.mean()),
        "sd": float(np.sqrt(variances.mean() + between)),
        "sd_between_draws": float(np.sqrt(between)),
    }


def _optional(value: float) -> float | None:
    return None if np.isnan(value) else float(value)
