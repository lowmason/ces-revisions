"""The location-scale Student-t model of one series' signed revisions (Fit 1).

    r_t ~ StudentT(nu, mu_t, sigma_t)
    mu_t = m + p[month] + gamma * window_t + phi * five_week_t
    log sigma_t = a + tau_y * z_y[year] + c[month] + beta * window_t
                  + kappa * year_window[year] + delta * five_week_t + lam * lapse_t

``beta`` is the per-day effect of the collection window on log scale, and ``scale_ratio`` =
exp(7 beta) is the ratio of revision scale at 19 business days to 12. The window enters the
location too, so a window effect on the mean is not read as a change in scale. The year
effects are partly pooled and non-centered. The month effects are unpooled and sum to zero,
because the window's mean and the 5-week share vary by calendar month.

Deviation from the recommendation: it has no benchmark-cell term omega. From 2004, the cell
it flags (December for F->S, November for F->T) is the same calendar month in every year
but 2003, so one observation would identify omega against that month's effect.
Its month effects on log scale have prior SD 0.4, not 0.5, after the prior predictive check.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from ces_revisions.collection_window import CONTRAST_DAYS
from ces_revisions.scale_model.data import NUM_MONTHS, ScaleData

# The Stage 6 pilot's lesson: at NumPyro's default of 0.8 the pilot diverged.
TARGET_ACCEPT_PROB = 0.95
SAMPLED_SITES = (
    "m",
    "p",
    "gamma",
    "phi",
    "a",
    "tau_y",
    "z_y",
    "c",
    "beta",
    "kappa",
    "delta",
    "lam",
    "nu",
)
OBSERVED_SITE = "r"
# The prior SD of the month effects on log scale (the recommendation had 0.5).
MONTH_SCALE_SD = 0.4


@dataclass(frozen=True)
class Priors:
    """The prior settings that the sensitivity refits vary."""

    beta_sd: float = 0.1  # about 3 times the data-only SE of 0.033 per day


DEFAULT_PRIORS = Priors()


def location_scale(params: dict, data: ScaleData) -> tuple[jax.Array, jax.Array]:
    """mu_t and log sigma_t at one parameter draw, each (T,)."""
    window = jnp.asarray(data.window)
    five_week = jnp.asarray(data.five_week)
    mu = (
        params["m"]
        + jnp.asarray(params["p"])[data.month]
        + params["gamma"] * window
        + params["phi"] * five_week
    )
    log_sigma = (
        params["a"]
        + params["tau_y"] * jnp.asarray(params["z_y"])[data.year]
        + jnp.asarray(params["c"])[data.month]
        + params["beta"] * window
        + params["kappa"] * jnp.asarray(data.year_window)[data.year]
        + params["delta"] * five_week
        + params["lam"] * jnp.asarray(data.lapse)
    )
    return mu, log_sigma


def pointwise_log_likelihood(params: dict, data: ScaleData) -> jax.Array:
    """log p(r_t | theta) for each reference month, (T,)."""
    mu, log_sigma = location_scale(params, data)
    return dist.StudentT(params["nu"], mu, jnp.exp(log_sigma)).log_prob(
        jnp.asarray(data.revision)
    )


def scale_model(
    data: ScaleData, priors: Priors = DEFAULT_PRIORS, revision=None
) -> None:
    """The model; ``revision=None`` draws the revisions, as for a predictive check."""
    # Location, in thousands. The NSA mean is about +9 with month means within about +-10.
    m = numpyro.sample("m", dist.Normal(0.0, 10.0))
    p = numpyro.sample("p", dist.ZeroSumNormal(10.0, event_shape=(NUM_MONTHS,)))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 5.0))  # per day of window
    phi = numpyro.sample("phi", dist.Normal(0.0, 10.0))  # 5-week interval
    # Log scale. Observed sigma is 36-62 thousand, so 40 thousand is the centre.
    a = numpyro.sample("a", dist.Normal(math.log(40.0), 0.5))
    # Era means vary by about +-0.15 in log; non-centered, as 23 years hold ~11 months each.
    tau_y = numpyro.sample("tau_y", dist.HalfNormal(0.3))
    with numpyro.plate("year", len(data.years)):
        z_y = numpyro.sample("z_y", dist.Normal(0.0, 1.0))
    numpyro.deterministic("u", tau_y * z_y)
    # Unpooled month effects. The observed month means spread with a log SD of 0.21-0.34.
    # The recommendation's 0.5 put the prior's central 90% of adjacent-month scale
    # ratios at 0.27-3.7, outside its own target of about 0.3-3; 0.4 meets it.
    c = numpyro.sample(
        "c", dist.ZeroSumNormal(MONTH_SCALE_SD, event_shape=(NUM_MONTHS,))
    )
    # The estimand: per-day window effect on log scale, weakly informative.
    beta = numpyro.sample("beta", dist.Normal(0.0, priors.beta_sd))
    kappa = numpyro.sample("kappa", dist.Normal(0.0, 0.1))  # between-year (Mundlak)
    delta = numpyro.sample("delta", dist.Normal(0.0, 0.25))  # 5-week interval on scale
    # Only 4 lapse months, so the prior regularizes.
    lam = numpyro.sample("lam", dist.Normal(0.0, 0.5))
    # Juarez-Steel; the measured kurtosis of 4-5 suggests nu near 5-10.
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    numpyro.deterministic("scale_ratio", jnp.exp(CONTRAST_DAYS * beta))

    params = {
        "m": m,
        "p": p,
        "gamma": gamma,
        "phi": phi,
        "a": a,
        "tau_y": tau_y,
        "z_y": z_y,
        "c": c,
        "beta": beta,
        "kappa": kappa,
        "delta": delta,
        "lam": lam,
    }
    mu, log_sigma = location_scale(params, data)
    with numpyro.plate("reference_month", len(data.reference_months)):
        numpyro.sample(
            OBSERVED_SITE,
            dist.StudentT(nu, mu, jnp.exp(log_sigma)),
            obs=None if revision is None else jnp.asarray(revision),
        )


def sample_prior(
    key: jax.Array, num_samples: int, data: ScaleData, priors: Priors = DEFAULT_PRIORS
) -> dict[str, np.ndarray]:
    """``num_samples`` prior draws of every sampled site, draws on the leading axis."""
    draws = Predictive(
        scale_model, num_samples=num_samples, return_sites=SAMPLED_SITES
    )(key, data, priors)
    return {name: np.asarray(draws[name]) for name in SAMPLED_SITES}
