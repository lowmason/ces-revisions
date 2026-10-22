"""Synthetic panels from the pilot model, and Req 15's prior-predictive monthly check.

``simulate_panel`` runs the model's equations forward as an explicit NumPy recursion, with
its own seasonal recursions, so it shares no code with ``model.pilot_ssm`` beyond the
priors' derived quantities and the initial-state constants; a test checks its moments
against the dense oracle. Rounding enters as Gaussian noise with the model's rounding
variances, exactly as the model treats it.

``prior_predictive`` draws parameters from the prior, simulates a complete panel per draw
on the real panel's flags, and records each draw's F-to-T revision statistics;
``monthly_constraint`` applies Req 15's monthly constraint at total nonfarm to them.
"""

import math

import jax
import numpy as np
import numpyro.distributions as dist

from ces_revisions.pilot.layout import PERIOD, Layout
from ces_revisions.pilot.model import (
    INITIAL_MEAN,
    INITIAL_SD,
    MONTHS_PER_YEAR,
    NOISE_RAW_SITES,
    derived,
    draw_distributions,
    prior_distributions,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CLOSING_ROUNDING_VARIANCE,
    COLUMNS,
    AggregatePanel,
)

F_COLUMN, T_COLUMN = COLUMNS.index("F"), COLUMNS.index("T")
# Req 15's monthly constraint at total nonfarm, operationalized: at least 80% of prior
# draws put the median |F->T| revision in [20, 100] thousand, and the pooled share of
# months with |F->T| above 250 thousand lies in [0.1%, 5%].
MEDIAN_RANGE = (20.0, 100.0)
MEDIAN_SHARE_MIN = 0.80
LARGE_REVISION = 250.0
LARGE_SHARE_RANGE = (0.001, 0.05)
# Req 15's sensitivity rerun doubles every LogNormal prior's log-scale SD. The noise SDs
# are LogNormal through their non-centered standard normal sites, so those double too.
SENSITIVITY_FACTOR = 2.0
# The standard normal draws of each month, in draw order; the seasonal shocks follow.
_MONTHLY_DRAWS = (
    "eps_s",
    "eps_t",
    "eps_post",
    "eta_level",
    "eta_slope",
    "z_f",
    "z_s",
    "z_t",
    "rounding_f",
    "rounding_s",
    "rounding_t",
    "anchor",
)


def _seasonal_gamma(layout: Layout, seasonal: np.ndarray) -> np.ndarray:
    """gamma from a (draws, S) seasonal block."""
    if layout.harmonics is None:
        return seasonal[:, 0]
    keep = [not name.endswith("_sin") for name in layout.seasonal_names]
    return seasonal[:, keep].sum(axis=1)


def _seasonal_step(
    layout: Layout, seasonal: np.ndarray, shocks: np.ndarray
) -> np.ndarray:
    """One month of the seasonal recursion; shocks are (draws, number of shocks), scaled."""
    if layout.harmonics is None:
        # gamma_t = -(gamma_{t-1} + ... + gamma_{t-11}) + w_t, then every slot ages a month.
        gamma = -seasonal.sum(axis=1) + shocks[:, 0]
        return np.column_stack([gamma, seasonal[:, :-1]])
    moved = np.empty_like(seasonal)
    for j in range(1, min(layout.harmonics, PERIOD // 2 - 1) + 1):
        angle = 2.0 * math.pi * j / PERIOD
        cos, sin = seasonal[:, 2 * j - 2], seasonal[:, 2 * j - 1]
        moved[:, 2 * j - 2] = math.cos(angle) * cos + math.sin(angle) * sin
        moved[:, 2 * j - 1] = -math.sin(angle) * cos + math.cos(angle) * sin
    if layout.harmonics == PERIOD // 2:
        moved[:, -1] = -seasonal[:, -1]
    return moved + shocks


def _simulate_complete(
    params: dict[str, np.ndarray], panel: AggregatePanel, layout: Layout, rng
) -> np.ndarray:
    """(draws, T, 4) panels: every closing cell, and anchors at the panel's anchor months."""
    p = {name: np.asarray(value, dtype=np.float64) for name, value in params.items()}
    noise = derived(p)
    num_draws, num_steps = p["sigma_level"].shape[0], panel.num_steps
    seasonal_shocks = 1 if layout.harmonics is None else layout.seasonal_dim

    def initial(name, size=None):
        return rng.normal(INITIAL_MEAN[name], INITIAL_SD[name], size)

    # lagged_value and the news states at t = -1 enter no later state, so only the
    # level, slope, and seasonal block of x_{-1} are drawn.
    level = initial("level", num_draws)
    slope = initial("slope", num_draws)
    seasonal = np.column_stack(
        [initial(name, num_draws) for name in layout.seasonal_names]
    )
    value = level + _seasonal_gamma(layout, seasonal)

    rounding_sd = math.sqrt(CLOSING_ROUNDING_VARIANCE)
    anchor_sd = np.sqrt(p["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE)
    y = np.full((num_draws, num_steps, len(COLUMNS)), np.nan)
    anchored = ~np.isnan(panel.y[:, ANCHOR_COLUMN])
    for t in range(num_steps):
        pre, crisis = float(panel.pre_2003[t]), float(panel.crisis[t])
        crisis_scale = np.exp(p["kappa_crisis"] * crisis)
        shift_scale = np.exp(p["shift_log_scale"] * pre)
        z = dict(
            zip(
                _MONTHLY_DRAWS,
                rng.standard_normal((len(_MONTHLY_DRAWS), num_draws)),
                strict=True,
            )
        )
        seasonal_z = rng.standard_normal((num_draws, seasonal_shocks))

        # D2: the crisis multiplier scales the S and T news and the non-news level shock.
        news_s = (
            p["mean_news_s"]
            + p["shift_mean"] * pre
            + p["sigma_news_s"] * shift_scale * crisis_scale * z["eps_s"]
        )
        news_t = (
            p["mean_news_t"]
            + p["shift_mean"] * pre
            + p["sigma_news_t"] * shift_scale * crisis_scale * z["eps_t"]
        )
        news_post = (
            p["mean_news_annual"] / MONTHS_PER_YEAR
            + noise["sigma_news_post"] * z["eps_post"]
        )
        eta_level = p["sigma_level"] * crisis_scale * z["eta_level"]
        level, slope = (
            level + slope + eta_level + news_s + news_t + news_post,
            slope + p["sigma_slope"] * z["eta_slope"],
        )
        seasonal = _seasonal_step(
            layout, seasonal, p["sigma_seasonal"][:, None] * seasonal_z
        )
        previous, value = value, level + _seasonal_gamma(layout, seasonal)
        change = value - previous

        # Req 7: e_f = z_f, e_s = rho e_f + z_s, e_t = rho e_s + z_t.
        e_f = noise["sigma_noise_f"] * shift_scale * z["z_f"]
        e_s = p["rho_noise"] * e_f + noise["sigma_noise_s"] * shift_scale * z["z_s"]
        e_t = p["rho_noise"] * e_s + noise["sigma_noise_t"] * shift_scale * z["z_t"]
        y[:, t, 0] = (
            change - news_s - news_t - news_post + e_f + rounding_sd * z["rounding_f"]
        )
        y[:, t, 1] = change - news_t - news_post + e_s + rounding_sd * z["rounding_s"]
        y[:, t, 2] = change - news_post + e_t + rounding_sd * z["rounding_t"]
        if anchored[t]:
            y[:, t, ANCHOR_COLUMN] = (
                value + panel.anchor_offset[t] + anchor_sd * z["anchor"]
            )
    return y


def simulate_panel(
    params: dict[str, float],
    panel: AggregatePanel,
    layout: Layout,
    seed: int,
    *,
    complete: bool = False,
) -> np.ndarray:
    """One (T, 4) panel drawn from the model at params, with x_{-1} from its prior.

    NaN wherever ``panel.y`` is NaN, unless ``complete``: then every closing cell of
    every month is filled, and anchors appear at the panel's anchor months only. The
    same seed draws the same values either way.
    """
    y = simulate_complete_panels(params, panel, layout, seed, 1)[0]
    if not complete:
        y[np.isnan(panel.y)] = np.nan
    return y


def simulate_complete_panels(
    params: dict[str, float],
    panel: AggregatePanel,
    layout: Layout,
    seed: int,
    num_panels: int,
) -> np.ndarray:
    """num_panels independent complete (T, 4) panels at the same params, in one pass.

    The draws depend on num_panels: the first panel is not ``simulate_panel``'s.
    """
    batch = {name: np.full(num_panels, float(value)) for name, value in params.items()}
    return _simulate_complete(batch, panel, layout, np.random.default_rng(seed))


def revision_statistics(y: np.ndarray) -> dict[str, float]:
    """F-to-T revision statistics over the rows where both F and T are observed."""
    both = ~np.isnan(y[:, F_COLUMN]) & ~np.isnan(y[:, T_COLUMN])
    revision = y[both, T_COLUMN] - y[both, F_COLUMN]
    size = np.abs(revision)
    return {
        "median_abs": float(np.median(size)),
        "share_above_250": float(np.mean(size > LARGE_REVISION)),
        "mean": float(revision.mean()),
        "sd": float(revision.std(ddof=1)),
        "count": int(both.sum()),
    }


def _widened(priors: dict[str, dist.Distribution]) -> dict[str, dist.Distribution]:
    widened = {}
    for name, prior in priors.items():
        if isinstance(prior, dist.LogNormal):
            widened[name] = dist.LogNormal(prior.loc, SENSITIVITY_FACTOR * prior.scale)
        elif name in NOISE_RAW_SITES:
            widened[name] = dist.Normal(prior.loc, SENSITIVITY_FACTOR * prior.scale)
        else:
            widened[name] = prior
    return widened


def prior_predictive(
    panel: AggregatePanel,
    layout: Layout,
    num_draws: int,
    seed: int,
    *,
    doubled: bool = False,
) -> dict:
    """F-to-T revision statistics of one complete simulated panel per prior draw.

    ``doubled`` doubles every LogNormal prior's log-scale SD and the SD of each
    ``NOISE_RAW_SITES`` prior (Req 15's sensitivity rerun).
    Returns the draws' parameters and statistics as arrays of shape (num_draws,).
    """
    priors = prior_distributions(layout)
    if doubled:
        priors = _widened(priors)
    params = {
        name: np.asarray(value)
        for name, value in draw_distributions(
            priors, jax.random.PRNGKey(seed), num_draws
        ).items()
    }
    panels = _simulate_complete(params, panel, layout, np.random.default_rng(seed))
    rows = [revision_statistics(y) for y in panels]
    return {
        "layout": layout.name,
        "seed": seed,
        "doubled": doubled,
        "num_draws": num_draws,
        "params": params,
        "statistics": {name: np.array([row[name] for row in rows]) for name in rows[0]},
    }


def monthly_constraint(prior_predictive_result: dict) -> dict:
    """Req 15's monthly constraint at total nonfarm, applied to a prior_predictive result."""
    statistics = prior_predictive_result["statistics"]
    median = statistics["median_abs"]
    low, high = MEDIAN_RANGE
    share_median_in_range = float(np.mean((median >= low) & (median <= high)))
    counts = statistics["count"]
    pooled = float(np.sum(statistics["share_above_250"] * counts) / np.sum(counts))
    passed = (
        share_median_in_range >= MEDIAN_SHARE_MIN
        and LARGE_SHARE_RANGE[0] <= pooled <= LARGE_SHARE_RANGE[1]
    )
    return {
        "share_median_in_range": share_median_in_range,
        "pooled_share_above_250": pooled,
        "passed": bool(passed),
    }
