"""The Stage 6 pilot model: priors, the state-space arrays, and the NumPyro model.

``pilot_ssm`` turns one draw of the 17 sampled scalars into the time-varying
``LinearGaussianSSM`` over the panel's reference months, and ``pilot_model`` samples them
and adds the panel's marginal log likelihood with ``kalman_factor``, so no state is ever a
sampled site. Shock scales vary in time through two flags: before May 2003 the news and
noise SDs are multiplied by ``exp(shift_log_scale)`` and the S and T news means move by
``shift_mean`` (Req 4's 2003 regime shift); in March-June 2020 the non-news level shock and
the S and T news shocks are multiplied by ``exp(kappa_crisis)`` (decision D2). The post-T
news is not: the March anchors pin only its yearly total, so a multiplied post-T shock would
let the crisis months absorb the whole benchmark year's news. Revision noise follows Req 7's
within-month recursion e_f = z_f, e_s = rho e_f + z_s, e_t = rho e_s + z_t.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ces_revisions.kalman import (
    FilterResult,
    LinearGaussianSSM,
    kalman_factor,
    kalman_filter,
)
from ces_revisions.pilot.layout import (
    LAYOUTS,
    NEWS_SHOCKS,
    NEWS_STATES,
    SEASONAL_PRIOR_SCALE,
    TREND_SHOCKS,
    Layout,
    observation_matrix,
    shock_loading,
    shock_names,
    transition_matrix,
)
from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CLOSING_ROUNDING_VARIANCE,
    CLOSING_STAGES,
    AggregatePanel,
)

MONTHS_PER_YEAR = 12
# NUTS's target acceptance for this model, for the fit and every SBC replicate alike. At
# NumPyro's 0.8 the real-panel fit diverged where sigma_news_t is small and trades off
# against noise_t_raw; at 0.95 it met Req 18 (plan 11, planning evidence).
TARGET_ACCEPT_PROB = 0.95

SAMPLED_SITES = (
    "sigma_level",
    "sigma_slope",
    "sigma_seasonal",
    "sigma_news_s",
    "sigma_news_t",
    "sigma_news_annual",
    "noise_f_raw",
    "noise_s_raw",
    "noise_t_raw",
    "rho_noise",
    "mean_news_s",
    "mean_news_t",
    "mean_news_annual",
    "sigma_anchor",
    "kappa_crisis",
    "shift_log_scale",
    "shift_mean",
)

# The state at December 1978 (x_{-1}), as independent normals, in thousands of jobs.
# lagged_value and the news states at t = -1 enter no later state or observation.
_SEASONAL_STATES = sorted(
    {name for layout in LAYOUTS.values() for name in layout.seasonal_names}
)
INITIAL_MEAN = {
    # Dec 1978 NSA total nonfarm in vintage cesvinall:2026-01.
    "level": 89_408.0,
    # Between the 1979-2019 average monthly gain (129) and 1979's (167), cesvinall:2026-01.
    "slope": 150.0,
    # x_{-2}, Nov 1978, is within a few hundred thousand of Dec 1978.
    "lagged_value": 89_408.0,
    # Seasonal effects sum to about zero over a year.
    **dict.fromkeys(_SEASONAL_STATES, 0.0),
    # The news states hold only the current month's shocks; x_{-1}'s enter nothing.
    **dict.fromkeys(NEWS_STATES, 0.0),
}
INITIAL_SD = {
    # Covers any vintage's Dec 1978 level, including every benchmark revision since.
    "level": 5_000.0,
    # Twice the mean: covers 1978's 359 thousand a month and a flat trend alike.
    "slope": 300.0,
    # As the level.
    "lagged_value": 5_000.0,
    # NSA seasonal swings in total nonfarm are one to two million.
    **dict.fromkeys(_SEASONAL_STATES, 2_000.0),
    # Wide against any monthly news SD; it enters nothing later.
    **dict.fromkeys(NEWS_STATES, 100.0),
}

# Req 15 noise row: a stage's noise SD is centered on half the SD of the news that next
# revises it, with a log-scale spread of 0.7. F's next news is S's and S's is T's. T's is
# the annual post-T news, which only the March anchors pin, so T's noise is centered on
# T's own news SD, as S's is (plan 11's reading of Req 15's sigma_{s,j}).
NOISE_CENTER = 0.5
NOISE_LOG_SCALE = 0.7
_NOISE_NEWS = {
    "sigma_noise_f": ("noise_f_raw", "sigma_news_s"),
    "sigma_noise_s": ("noise_s_raw", "sigma_news_t"),
    "sigma_noise_t": ("noise_t_raw", "sigma_news_t"),
}
# The standard normal sites behind the noise SDs: NOISE_LOG_SCALE times one is its SD's
# log-scale deviation from the center.
NOISE_RAW_SITES = tuple(raw for raw, _ in _NOISE_NEWS.values())


def prior_distributions(layout: Layout) -> dict[str, dist.Distribution]:
    """The prior of every sampled site, in ``SAMPLED_SITES`` order."""
    priors = {
        "sigma_level": dist.HalfNormal(150.0),  # Req 15 local-level row, non-news part
        "sigma_slope": dist.HalfNormal(15.0),
        "sigma_seasonal": dist.HalfNormal(SEASONAL_PRIOR_SCALE[layout.name]),
        "sigma_news_s": dist.LogNormal(math.log(40.0), 0.40),
        "sigma_news_t": dist.LogNormal(math.log(35.0), 0.45),
        "sigma_news_annual": dist.LogNormal(math.log(300.0), 0.60),  # post-T, per year
        "noise_f_raw": dist.Normal(0.0, 1.0),  # non-centered noise SDs; see derived()
        "noise_s_raw": dist.Normal(0.0, 1.0),
        "noise_t_raw": dist.Normal(0.0, 1.0),
        "rho_noise": dist.TruncatedNormal(0.0, 0.20, low=-0.8, high=0.8),
        "mean_news_s": dist.Normal(0.0, 15.0),
        "mean_news_t": dist.Normal(0.0, 15.0),
        "mean_news_annual": dist.Normal(0.0, 100.0),  # post-T, per year
        "sigma_anchor": dist.HalfNormal(100.0),
        "kappa_crisis": dist.Normal(math.log(50.0), 0.5),
        "shift_log_scale": dist.Normal(0.0, 0.5),
        "shift_mean": dist.Normal(0.0, 15.0),
    }
    return {name: priors[name] for name in SAMPLED_SITES}


def _namespace(params: dict):
    """jax.numpy when any value is a JAX array (or tracer), NumPy otherwise."""
    return jnp if any(isinstance(value, jax.Array) for value in params.values()) else np


def derived(params: dict) -> dict:
    """The deterministic quantities of a draw: noise SDs, post-T news SD, news shares.

    Works elementwise on scalars or arrays of draws, NumPy or JAX. The shares are the
    news variance's share of each revision's variance at post-2003 scale, without the
    rounding variance: Var(e_s - e_f) = (rho - 1)^2 s_f^2 + s_s^2 and
    Var(e_t - e_s) = (rho - 1)^2 (rho^2 s_f^2 + s_s^2) + s_t^2.
    """
    xp = _namespace(params)
    noise = {
        name: NOISE_CENTER * params[news] * xp.exp(NOISE_LOG_SCALE * params[raw])
        for name, (raw, news) in _NOISE_NEWS.items()
    }
    rho = params["rho_noise"]
    var_f, var_s, var_t = (noise[name] ** 2 for name in _NOISE_NEWS)
    noise_var_s = (rho - 1.0) ** 2 * var_f + var_s
    noise_var_t = (rho - 1.0) ** 2 * (rho**2 * var_f + var_s) + var_t
    news_var_s, news_var_t = params["sigma_news_s"] ** 2, params["sigma_news_t"] ** 2
    return {
        **noise,
        "sigma_news_post": params["sigma_news_annual"] / math.sqrt(MONTHS_PER_YEAR),
        "share_s": news_var_s / (news_var_s + noise_var_s),
        "share_t": news_var_t / (news_var_t + noise_var_t),
    }


def _closing_covariance(rho: jax.Array, sd: jax.Array) -> jax.Array:
    """Req 7's (T, 3, 3) closing-noise covariance M diag(sd^2) M' plus rounding."""
    zero, one = jnp.zeros_like(rho), jnp.ones_like(rho)
    recursion = jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([rho, one, zero]),
            jnp.stack([rho**2, rho, one]),
        ]
    )
    scaled = recursion[None] * sd[:, None, :]  # M diag(sd), per step
    rounding = CLOSING_ROUNDING_VARIANCE * jnp.eye(len(CLOSING_STAGES))
    return scaled @ jnp.swapaxes(scaled, 1, 2) + rounding


def pilot_ssm(
    params: dict[str, jax.Array], panel: AggregatePanel, layout: Layout
) -> LinearGaussianSSM:
    """The pilot's time-varying state-space model for one draw of the sampled sites."""
    num_steps, state_dim = panel.num_steps, layout.state_dim
    params = {
        name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()
    }
    noise = derived(params)
    pre_2003 = jnp.asarray(panel.pre_2003, dtype=jnp.float64)
    crisis_scale = jnp.exp(
        params["kappa_crisis"] * jnp.asarray(panel.crisis, jnp.float64)
    )
    shift_scale = jnp.exp(params["shift_log_scale"] * pre_2003)

    # Shock SDs in shock_names order: eta_level, eta_slope, seasonal shocks, eps_s, eps_t,
    # eps_post. D2: the crisis multiplier scales the non-news level shock and the S and T
    # news shocks, never the slope, seasonal, or post-T news shocks.
    seasonal_shocks = len(shock_names(layout)) - len(TREND_SHOCKS) - len(NEWS_SHOCKS)
    ones = jnp.ones(num_steps)
    shock_sd = jnp.stack(
        [
            params["sigma_level"] * crisis_scale,
            params["sigma_slope"] * ones,
            *[params["sigma_seasonal"] * ones] * seasonal_shocks,
            params["sigma_news_s"] * shift_scale * crisis_scale,
            params["sigma_news_t"] * shift_scale * crisis_scale,
            noise["sigma_news_post"] * ones,
        ],
        axis=1,
    )
    loading = jnp.asarray(shock_loading(layout))
    transition_cov = jnp.einsum("ik,tk,jk->tij", loading, shock_sd**2, loading)

    # News enters the truth with its mean, so the level's offset is the news offsets' sum.
    news_mean = jnp.stack(
        [
            params["mean_news_s"] + params["shift_mean"] * pre_2003,
            params["mean_news_t"] + params["shift_mean"] * pre_2003,
            params["mean_news_annual"] / MONTHS_PER_YEAR * ones,
        ],
        axis=1,
    )
    names = layout.state_names
    transition_offset = (
        jnp.zeros((num_steps, state_dim))
        .at[:, names.index("level")]
        .set(news_mean.sum(axis=1))
        .at[:, [names.index(state) for state in NEWS_STATES]]
        .set(news_mean)
    )

    noise_sd = (
        jnp.stack([noise[name] for name in _NOISE_NEWS])[None, :] * shift_scale[:, None]
    )
    closing_cov = _closing_covariance(params["rho_noise"], noise_sd)
    anchor_var = params["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE
    num_closing = len(CLOSING_STAGES)
    observation_cov = (
        jnp.zeros((num_steps, ANCHOR_COLUMN + 1, ANCHOR_COLUMN + 1))
        .at[:, :num_closing, :num_closing]
        .set(closing_cov)
        .at[:, ANCHOR_COLUMN, ANCHOR_COLUMN]
        .set(anchor_var)
    )
    observation_offset = (
        jnp.zeros((num_steps, ANCHOR_COLUMN + 1))
        .at[:, ANCHOR_COLUMN]
        .set(jnp.asarray(panel.anchor_offset))
    )

    def over_steps(matrix):
        return jnp.broadcast_to(jnp.asarray(matrix), (num_steps, *matrix.shape))

    return LinearGaussianSSM(
        initial_mean=jnp.array([INITIAL_MEAN[name] for name in names]),
        initial_cov=jnp.diag(jnp.array([INITIAL_SD[name] for name in names]) ** 2),
        transition_matrix=over_steps(transition_matrix(layout)),
        transition_offset=transition_offset,
        transition_cov=transition_cov,
        observation_matrix=over_steps(observation_matrix(layout)),
        observation_offset=observation_offset,
        observation_cov=observation_cov,
    )


def pilot_model(panel: AggregatePanel, layout: Layout) -> FilterResult:
    """Sample the 17 scalars; add the panel's marginal log likelihood as site "panel"."""
    params = {
        name: numpyro.sample(name, prior)
        for name, prior in prior_distributions(layout).items()
    }
    return kalman_factor(
        "panel", pilot_ssm(params, panel, layout), jnp.asarray(panel.y)
    )


def sample_prior(
    key: jax.Array, num_samples: int, layout: Layout
) -> dict[str, jax.Array]:
    """Independent prior draws of every sampled site, shape (num_samples,) each."""
    return draw_distributions(prior_distributions(layout), key, num_samples)


def draw_distributions(
    priors: dict[str, dist.Distribution], key: jax.Array, num_samples: int
) -> dict[str, jax.Array]:
    """Independent draws from each named distribution, one split key per name."""
    keys = jax.random.split(key, len(priors))
    return {
        name: prior.sample(site_key, (num_samples,))
        for (name, prior), site_key in zip(priors.items(), keys, strict=True)
    }


def step_log_likelihood(
    params: dict[str, jax.Array], panel: AggregatePanel, layout: Layout
) -> jax.Array:
    """The (T,) per-step log likelihood log p(y_t | y_<t) of one draw."""
    ssm = pilot_ssm(params, panel, layout)
    return kalman_filter(ssm, jnp.asarray(panel.y)).step_log_likelihood
