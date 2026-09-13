"""A synthetic pilot for the Kalman engine: CES-shaped structure and six sampled scales.

A local linear trend, in thousands of jobs, is observed by three closing rows (first,
second, and third print) that measure month-over-month changes before LEVEL_REGIME_START
and levels from then on (a time-varying emission row), and by an annual benchmark row
observed only in March. The panel has scattered missing cells, a ragged real-time edge, and
one fully missing month, and a known crisis window scales the level innovations and the
closing-row noise. Only the six scales are sampled; ``kalman_factor`` integrates out the
states.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ces_revisions.kalman import LinearGaussianSSM, kalman_factor

NUM_STEPS = 120  # ten years of reference months; t = 0 is a January
MARCH = 2  # benchmark months are the steps with t % 12 == MARCH
LEVEL_REGIME_START = 60  # closing rows observe changes before this step, levels after
CRISIS_STEPS = range(86, 90)  # a known window standing in for March–June 2020
CRISIS_PROCESS_MULTIPLIER = 10.0  # on the level-innovation SD inside the window
CRISIS_NOISE_MULTIPLIER = 3.0  # on the closing-row noise SD inside the window
LAPSE_STEP = 100  # a month with no release at all
MISSING_CELL_RATE = 0.05  # share of closing-row cells dropped at random
INITIAL_SD = (50.0, 5.0, 50.0)  # zero-mean prior SDs of the state at t = -1
CLOSING_ROWS = ("sigma_first", "sigma_second", "sigma_third")
BENCHMARK_ROW = len(CLOSING_ROWS)

TRUTH = {
    "sigma_level": 20.0,
    "sigma_slope": 2.0,
    "sigma_first": 30.0,
    "sigma_second": 20.0,
    "sigma_third": 12.0,
    "sigma_benchmark": 8.0,
}
# Broad LogNormal(log median, 1) priors, not centered on TRUTH; one SD is a factor of e.
PRIOR_MEDIANS = {
    "sigma_level": 15.0,
    "sigma_slope": 3.0,
    "sigma_first": 15.0,
    "sigma_second": 15.0,
    "sigma_third": 15.0,
    "sigma_benchmark": 15.0,
}


def _in_crisis(steps):
    return (steps >= CRISIS_STEPS.start) & (steps < CRISIS_STEPS.stop)


def simulate_panel(truth, seed):
    """Draw the (NUM_STEPS, 4) panel by explicit recursion, independently of pilot_ssm."""
    rng = np.random.default_rng(seed)
    steps = np.arange(NUM_STEPS)
    crisis = _in_crisis(steps)
    level_sd = truth["sigma_level"] * np.where(crisis, CRISIS_PROCESS_MULTIPLIER, 1.0)
    level_prev = rng.normal(0.0, INITIAL_SD[0])
    slope_prev = rng.normal(0.0, INITIAL_SD[1])
    levels, changes = np.empty(NUM_STEPS), np.empty(NUM_STEPS)
    for t in steps:
        level = level_prev + slope_prev + rng.normal(0.0, level_sd[t])
        slope = slope_prev + rng.normal(0.0, truth["sigma_slope"])
        levels[t], changes[t] = level, level - level_prev
        level_prev, slope_prev = level, slope

    y = np.full((NUM_STEPS, BENCHMARK_ROW + 1), np.nan)
    closing_target = np.where(steps < LEVEL_REGIME_START, changes, levels)
    noise_scale = np.where(crisis, CRISIS_NOISE_MULTIPLIER, 1.0)
    for row, name in enumerate(CLOSING_ROWS):
        y[:, row] = closing_target + rng.normal(0.0, truth[name] * noise_scale)
    march = steps % 12 == MARCH
    benchmark_noise = rng.normal(0.0, truth["sigma_benchmark"], march.sum())
    y[march, BENCHMARK_ROW] = levels[march] + benchmark_noise

    closing = y[:, :BENCHMARK_ROW]  # a view, so these writes land in y
    closing[rng.random(closing.shape) < MISSING_CELL_RATE] = np.nan
    closing[-1, 1:] = np.nan  # ragged edge: the latest month has only its first print
    closing[-2, 2] = np.nan  # and the month before it has no third print yet
    y[LAPSE_STEP] = np.nan
    return y


def pilot_ssm(scales):
    """The pilot as a LinearGaussianSSM over the state (level_t, slope_t, level_{t-1})."""
    steps = jnp.arange(NUM_STEPS)
    crisis = _in_crisis(steps)
    transition = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    process_sd = jnp.stack(
        [
            scales["sigma_level"] * jnp.where(crisis, CRISIS_PROCESS_MULTIPLIER, 1.0),
            jnp.broadcast_to(scales["sigma_slope"], (NUM_STEPS,)),
            jnp.zeros(NUM_STEPS),  # the lagged level is carried exactly
        ],
        axis=1,
    )

    level_row = jnp.array([1.0, 0.0, 0.0])
    change_row = jnp.array([1.0, 0.0, -1.0])
    measures_changes = (steps < LEVEL_REGIME_START)[:, None]
    closing_row = jnp.where(measures_changes, change_row, level_row)
    benchmark_row = jnp.broadcast_to(level_row, (NUM_STEPS, 3))
    observation_rows = [closing_row] * len(CLOSING_ROWS) + [benchmark_row]
    noise_scale = jnp.where(crisis, CRISIS_NOISE_MULTIPLIER, 1.0)
    observation_sd = jnp.stack(
        [
            *(scales[name] * noise_scale for name in CLOSING_ROWS),
            jnp.broadcast_to(scales["sigma_benchmark"], (NUM_STEPS,)),
        ],
        axis=1,
    )
    return LinearGaussianSSM(
        initial_mean=jnp.zeros(3),
        initial_cov=jnp.diag(jnp.square(jnp.array(INITIAL_SD))),
        transition_matrix=jnp.broadcast_to(transition, (NUM_STEPS, 3, 3)),
        transition_offset=jnp.zeros((NUM_STEPS, 3)),
        transition_cov=jax.vmap(jnp.diag)(process_sd**2),
        observation_matrix=jnp.stack(observation_rows, axis=1),
        observation_offset=jnp.zeros((NUM_STEPS, BENCHMARK_ROW + 1)),
        observation_cov=jax.vmap(jnp.diag)(observation_sd**2),
    )


def pilot_model(y):
    """Sample the six scales; add the panel's marginal log likelihood as site "panel"."""
    scales = {
        name: numpyro.sample(name, dist.LogNormal(math.log(median), 1.0))
        for name, median in PRIOR_MEDIANS.items()
    }
    kalman_factor("panel", pilot_ssm(scales), y)
