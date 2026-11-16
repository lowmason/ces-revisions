"""A small synthetic aggregate panel for the Stage 6 pilot, and an independent model.

``reference_ssm`` writes the pilot's state-space arrays entry by entry from the model's
written definition (state names, transitions, shock covariances, observation rows, and
Req 7's noise recursion), sharing no code with ``ces_revisions.pilot.layout`` or
``ces_revisions.pilot.model``. Tests compare ``pilot_ssm`` with it array by array, and run
the dense oracle on it.

``synthetic_panel`` packs every feature the pilot must handle into a short window: a
pre-2003 to post-2003 boundary, two crisis steps, March anchors with nonzero offsets, a
fully missing step, F-only steps like a benchmark carrier's, scattered missing cells, and a
ragged edge. Its values are drawn from ``reference_ssm`` by the dense oracle's simulator,
not by the pilot's own ``simulate_panel``. The crisis flags are set by step, not by date:
the real crisis (2020) and the 2003 boundary cannot share a short window.
"""

import math
from datetime import date

import numpy as np
import polars as pl
from dense_reference import simulate

from ces_revisions.pilot.panel import (
    ANCHOR_COLUMN,
    ANCHOR_ROUNDING_VARIANCE,
    CELL_SCHEMA,
    CLOSING_ROUNDING_VARIANCE,
    COLUMNS,
    REGIME_SHIFT_MONTH,
    AggregatePanel,
    benchmark_year_of,
)

SEED = sum(map(ord, "ces-revisions-stage-6-synthetic-panel"))
FIRST_MONTH = date(2002, 1, 1)  # steps 0-15 are before REGIME_SHIFT_MONTH (2003-05)
NUM_STEPS = 30
CRISIS_STEPS = (20, 21)
MISSING_STEP = 9  # no cell observed at all
CARRIER_STEPS = (5, 17)  # F observed, S and T masked as a benchmark carrier's would be
MISSING_CELLS = ((12, 2), (23, 0))  # (step, column) gaps
# The anchor offset at each successive March: earlier reconstructions add up.
ANCHOR_OFFSETS = (0.0, -312.0, 175.0, 268.0, -41.0)

# A draw away from the prior medians, so a swapped or mis-scaled term shows up.
TRUTH = {
    "sigma_level": 60.0,
    "sigma_slope": 8.0,
    "sigma_seasonal": 30.0,
    "sigma_news_s": 45.0,
    "sigma_news_t": 30.0,
    "sigma_news_annual": 350.0,
    "noise_f_raw": 0.3,
    "noise_s_raw": -0.2,
    "noise_t_raw": 0.1,
    "rho_noise": 0.25,
    "mean_news_s": 5.0,
    "mean_news_t": -3.0,
    "mean_news_annual": 40.0,
    "sigma_anchor": 80.0,
    "kappa_crisis": math.log(30.0),
    "shift_log_scale": 0.3,
    "shift_mean": -6.0,
}

_TREND = ("level", "slope", "lagged_value")
_NEWS = ("news_s", "news_t", "news_post")
_PAIRS = {"trig": 5, "trig4": 4}  # cos/sin harmonic pairs j = 1..pairs
STATE_NAMES = {
    "dummy": (*_TREND, *(f"seasonal_{k}" for k in range(11)), *_NEWS),
    "trig": (
        *_TREND,
        *(f"harmonic_{j}_{part}" for j in range(1, 6) for part in ("cos", "sin")),
        "harmonic_6",
        *_NEWS,
    ),
    "trig4": (
        *_TREND,
        *(f"harmonic_{j}_{part}" for j in range(1, 5) for part in ("cos", "sin")),
        *_NEWS,
    ),
}


def _months(first: date, num_steps: int) -> tuple[date, ...]:
    index = first.year * 12 + first.month - 1
    return tuple(
        date((index + t) // 12, (index + t) % 12 + 1, 1) for t in range(num_steps)
    )


def _gamma_states(layout_name: str) -> tuple[str, ...]:
    """The seasonal states whose sum is gamma_t."""
    if layout_name == "dummy":
        return ("seasonal_0",)
    names = STATE_NAMES[layout_name]
    return tuple(
        name for name in names if name.endswith("_cos") or name == "harmonic_6"
    )


def reference_ssm(params: dict, panel: AggregatePanel, layout_name: str) -> dict:
    """The pilot's arrays, written entry by entry, as a dense_reference model dict."""
    names = STATE_NAMES[layout_name]
    at = {name: k for k, name in enumerate(names)}
    n, num_steps = len(names), panel.num_steps
    seasonal = [name for name in names if name not in (*_TREND, *_NEWS)]

    a = np.zeros((n, n))
    a[at["level"], at["level"]] = a[at["level"], at["slope"]] = 1.0
    a[at["slope"], at["slope"]] = 1.0
    a[at["lagged_value"], at["level"]] = 1.0
    for name in _gamma_states(layout_name):
        a[at["lagged_value"], at[name]] = 1.0
    if layout_name == "dummy":
        for k in range(11):
            a[at["seasonal_0"], at[f"seasonal_{k}"]] = -1.0
        for k in range(1, 11):
            a[at[f"seasonal_{k}"], at[f"seasonal_{k - 1}"]] = 1.0
    else:
        for j in range(1, _PAIRS[layout_name] + 1):
            cos, sin = f"harmonic_{j}_cos", f"harmonic_{j}_sin"
            angle = 2 * math.pi * j / 12
            a[at[cos], at[cos]] = a[at[sin], at[sin]] = math.cos(angle)
            a[at[cos], at[sin]] = math.sin(angle)
            a[at[sin], at[cos]] = -math.sin(angle)
        if layout_name == "trig":
            a[at["harmonic_6"], at["harmonic_6"]] = -1.0

    offset = np.zeros((num_steps, n))
    shock_cov = np.zeros((num_steps, n, n))
    obs_cov = np.zeros((num_steps, 4, 4))
    noise_sd = {
        "f": 0.5 * params["sigma_news_s"] * math.exp(0.7 * params["noise_f_raw"]),
        "s": 0.5 * params["sigma_news_t"] * math.exp(0.7 * params["noise_s_raw"]),
        "t": 0.5 * params["sigma_news_t"] * math.exp(0.7 * params["noise_t_raw"]),
    }
    rho = params["rho_noise"]
    for t in range(num_steps):
        pre, crisis = float(panel.pre_2003[t]), float(panel.crisis[t])
        crisis_scale = math.exp(params["kappa_crisis"] * crisis)
        shift_scale = math.exp(params["shift_log_scale"] * pre)
        news_mean = {
            "news_s": params["mean_news_s"] + params["shift_mean"] * pre,
            "news_t": params["mean_news_t"] + params["shift_mean"] * pre,
            "news_post": params["mean_news_annual"] / 12,
        }
        news_var = {
            "news_s": (params["sigma_news_s"] * shift_scale * crisis_scale) ** 2,
            "news_t": (params["sigma_news_t"] * shift_scale * crisis_scale) ** 2,
            "news_post": params["sigma_news_annual"] ** 2 / 12,
        }
        level_var = (params["sigma_level"] * crisis_scale) ** 2
        offset[t, at["level"]] = sum(news_mean.values())
        shock_cov[t, at["level"], at["level"]] = level_var + sum(news_var.values())
        for name in _NEWS:
            offset[t, at[name]] = news_mean[name]
            shock_cov[t, at[name], at[name]] = news_var[name]
            shock_cov[t, at[name], at["level"]] = news_var[name]
            shock_cov[t, at["level"], at[name]] = news_var[name]
        shock_cov[t, at["slope"], at["slope"]] = params["sigma_slope"] ** 2
        shocked = ["seasonal_0"] if layout_name == "dummy" else seasonal
        for name in shocked:
            shock_cov[t, at[name], at[name]] = params["sigma_seasonal"] ** 2

        var_f, var_s, var_t = ((noise_sd[k] * shift_scale) ** 2 for k in "fst")
        e_s_var = rho**2 * var_f + var_s
        obs_cov[t, :3, :3] = [
            [var_f, rho * var_f, rho**2 * var_f],
            [rho * var_f, e_s_var, rho * e_s_var],
            [rho**2 * var_f, rho * e_s_var, rho**2 * e_s_var + var_t],
        ]
        obs_cov[t, :3, :3] += CLOSING_ROUNDING_VARIANCE * np.eye(3)
        obs_cov[t, 3, 3] = params["sigma_anchor"] ** 2 + ANCHOR_ROUNDING_VARIANCE

    z = np.zeros((4, n))
    for row in range(4):
        z[row, at["level"]] = 1.0
        for name in _gamma_states(layout_name):
            z[row, at[name]] = 1.0
    for row, missed in enumerate((_NEWS, _NEWS[1:], _NEWS[2:])):
        z[row, at["lagged_value"]] = -1.0
        for name in missed:
            z[row, at[name]] = -1.0
    obs_offset = np.zeros((num_steps, 4))
    obs_offset[:, 3] = panel.anchor_offset

    mean0 = {"level": 89_408.0, "slope": 150.0, "lagged_value": 89_408.0}
    sd0 = {"level": 5_000.0, "slope": 300.0, "lagged_value": 5_000.0}
    sd0 |= dict.fromkeys(seasonal, 2_000.0) | dict.fromkeys(_NEWS, 100.0)
    return {
        "initial_mean": np.array([mean0.get(name, 0.0) for name in names]),
        "initial_cov": np.diag([sd0[name] ** 2 for name in names]),
        "transition_matrix": np.broadcast_to(a, (num_steps, n, n)).copy(),
        "transition_offset": offset,
        "transition_cov": shock_cov,
        "observation_matrix": np.broadcast_to(z, (num_steps, 4, n)).copy(),
        "observation_offset": obs_offset,
        "observation_cov": obs_cov,
    }


def shell_panel(
    num_steps: int = NUM_STEPS,
    first_month: date = FIRST_MONTH,
    crisis_steps: tuple[int, ...] = CRISIS_STEPS,
) -> AggregatePanel:
    """A panel with every flag set and 0.0 placeholders in each cell that will be observed.

    Anchors sit at every March, with ANCHOR_OFFSETS in turn; the closing cells are all
    zero, so a simulator with ``complete=True`` fills them.
    """
    months = _months(first_month, num_steps)
    march = np.array([month.month == 3 for month in months])
    anchor_offset = np.zeros(num_steps)
    anchor_offset[march] = ANCHOR_OFFSETS[: march.sum()]
    y = np.zeros((num_steps, len(COLUMNS)))
    y[~march, ANCHOR_COLUMN] = np.nan
    return _panel(months, y, anchor_offset, crisis_steps)


def synthetic_panel(
    layout_name: str = "dummy",
    num_steps: int = NUM_STEPS,
    *,
    params: dict = TRUTH,
    seed: int = SEED,
) -> AggregatePanel:
    """A panel drawn from ``reference_ssm`` and then masked with every gap pattern."""
    shell = shell_panel(num_steps)
    draws = simulate(
        reference_ssm(params, shell, layout_name), np.random.default_rng(seed)
    )
    y = np.where(np.isnan(shell.y), np.nan, draws)
    y[MISSING_STEP] = np.nan
    for step in CARRIER_STEPS:
        y[step, 1:ANCHOR_COLUMN] = np.nan
    for step, column in MISSING_CELLS:
        y[step, column] = np.nan
    y[-1, 1:ANCHOR_COLUMN] = np.nan  # ragged edge: the latest month has only F
    y[-2, 2] = np.nan  # and the month before it has no T yet
    return _panel(shell.reference_months, y, shell.anchor_offset, CRISIS_STEPS)


# A window around the 2020 crisis for the April 2020 check: 2019-07 through 2020-12, with
# the crisis flags on March-June 2020 by date and an anchor at March 2020.
CRISIS_FIRST_MONTH = date(2019, 7, 1)
CRISIS_NUM_STEPS = 18
CRISIS_WINDOW_STEPS = (8, 9, 10, 11)


def crisis_panel(
    layout_name: str = "dummy", *, params: dict = TRUTH, seed: int = SEED
) -> AggregatePanel:
    """A fully observed panel over CRISIS_FIRST_MONTH's 18 months, from ``reference_ssm``.

    The latest month has only its F print, like the real panel's ragged edge.
    """
    shell = shell_panel(CRISIS_NUM_STEPS, CRISIS_FIRST_MONTH, CRISIS_WINDOW_STEPS)
    draws = simulate(
        reference_ssm(params, shell, layout_name), np.random.default_rng(seed)
    )
    y = np.where(np.isnan(shell.y), np.nan, draws)
    y[-1, 1:ANCHOR_COLUMN] = np.nan
    return _panel(shell.reference_months, y, shell.anchor_offset, CRISIS_WINDOW_STEPS)


def _panel(months, y, anchor_offset, crisis_steps) -> AggregatePanel:
    num_steps = len(months)
    crisis = np.zeros(num_steps, dtype=bool)
    crisis[list(crisis_steps)] = True
    rows = [
        {
            "reference_month": month,
            "column": column,
            "source": "synthetic",
            "value_thousands": None if np.isnan(y[t, k]) else float(y[t, k]),
            "status": "synthetic",
            "mask_reason": "synthetic_gap" if np.isnan(y[t, k]) else None,
            "published_date": None,
        }
        for t, month in enumerate(months)
        for k, column in enumerate(COLUMNS)
    ]
    return AggregatePanel(
        reference_months=months,
        y=y.astype(np.float64),
        anchor_offset=anchor_offset.astype(np.float64),
        pre_2003=np.array([month < REGIME_SHIFT_MONTH for month in months]),
        crisis=crisis,
        benchmark_year=np.array([benchmark_year_of(month) for month in months]),
        cells=pl.DataFrame(rows, schema=CELL_SCHEMA),
    )
