"""The pilot's state layouts: the state vector, its transition, and its observation rows.

The state is ``level, slope, lagged_value, <seasonal block>, news_s, news_t, news_post``.
``level`` and ``slope`` form a local linear trend; ``lagged_value`` copies last month's
value ``x_{t-1} = level_{t-1} + gamma_{t-1}``, so a same-release change is a difference of
two states; and the three news states are white noise holding this month's second-print,
third-print, and post-third-print news, the same shocks that move ``level``
(Jacobs and van Norden's news inside the truth's innovation). The seasonal block is the
only part that differs between layouts: a dummy seasonal ("dummy"), a full trigonometric
seasonal ("trig"), or a trigonometric seasonal with four harmonics ("trig4").

All matrices are NumPy float64 and time-invariant; ``model.pilot_ssm`` broadcasts them over
the reference months and adds the time-varying shock and noise scales.
"""

import math
from dataclasses import dataclass

import numpy as np

from ces_revisions.pilot.panel import COLUMNS

PERIOD = 12  # months in a seasonal cycle
TREND_STATES = ("level", "slope", "lagged_value")
NEWS_STATES = ("news_s", "news_t", "news_post")
TREND_SHOCKS = ("eta_level", "eta_slope")
NEWS_SHOCKS = ("eps_s", "eps_t", "eps_post")
# Req 15's HalfNormal scale for the dummy seasonal shock SD.
DUMMY_SEASONAL_PRIOR_SCALE = 50.0


@dataclass(frozen=True)
class Layout:
    """One seasonal layout of the pilot's state vector.

    ``harmonics`` is None for the dummy seasonal and the highest harmonic for the
    trigonometric layouts (6 includes the Nyquist harmonic, a single state).
    """

    name: str
    harmonics: int | None
    state_names: tuple[str, ...]

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def seasonal_dim(self) -> int:
        return self.state_dim - len(TREND_STATES) - len(NEWS_STATES)

    @property
    def seasonal_names(self) -> tuple[str, ...]:
        first = len(TREND_STATES)
        return self.state_names[first : first + self.seasonal_dim]


def _trigonometric_names(harmonics: int) -> tuple[str, ...]:
    pairs = min(harmonics, PERIOD // 2 - 1)
    names = [
        f"harmonic_{j}_{part}" for j in range(1, pairs + 1) for part in ("cos", "sin")
    ]
    if harmonics == PERIOD // 2:
        names.append(f"harmonic_{PERIOD // 2}")
    return tuple(names)


def _layout(name: str, harmonics: int | None, seasonal: tuple[str, ...]) -> Layout:
    return Layout(name, harmonics, (*TREND_STATES, *seasonal, *NEWS_STATES))


LAYOUTS = {
    "dummy": _layout("dummy", None, tuple(f"seasonal_{k}" for k in range(PERIOD - 1))),
    "trig": _layout("trig", PERIOD // 2, _trigonometric_names(PERIOD // 2)),
    "trig4": _layout("trig4", 4, _trigonometric_names(4)),
}


def _index(layout: Layout, name: str) -> int:
    return layout.state_names.index(name)


def _seasonal_slice(layout: Layout) -> slice:
    first = len(TREND_STATES)
    return slice(first, first + layout.seasonal_dim)


def _seasonal_transition(layout: Layout) -> np.ndarray:
    """The seasonal block's own transition, (S, S)."""
    size = layout.seasonal_dim
    block = np.zeros((size, size))
    if layout.harmonics is None:
        # gamma_t = -(gamma_{t-1} + ... + gamma_{t-11}); the other slots shift down by one.
        block[0, :] = -1.0
        block[np.arange(1, size), np.arange(size - 1)] = 1.0
        return block
    for j in range(1, min(layout.harmonics, PERIOD // 2 - 1) + 1):
        frequency = 2.0 * math.pi * j / PERIOD
        cos, sin = math.cos(frequency), math.sin(frequency)
        first = 2 * (j - 1)
        block[first : first + 2, first : first + 2] = [[cos, sin], [-sin, cos]]
    if layout.harmonics == PERIOD // 2:
        block[-1, -1] = -1.0  # the Nyquist harmonic alternates sign every month
    return block


def _seasonal_selector_block(layout: Layout) -> np.ndarray:
    """The (S,) row that reads gamma_t off the seasonal block."""
    if layout.harmonics is None:
        selector = np.zeros(layout.seasonal_dim)
        selector[0] = 1.0
        return selector
    return np.array(
        [0.0 if name.endswith("_sin") else 1.0 for name in layout.seasonal_names]
    )


def _seasonal_loading_block(layout: Layout) -> np.ndarray:
    """The (S, number of seasonal shocks) loading of the seasonal shocks."""
    if layout.harmonics is None:
        loading = np.zeros((layout.seasonal_dim, 1))
        loading[0, 0] = 1.0  # one shock, on the current month's slot
        return loading
    return np.eye(layout.seasonal_dim)  # every harmonic state has its own shock


def seasonal_selector(layout: Layout) -> np.ndarray:
    """The (n,) row over the whole state that picks gamma_t."""
    selector = np.zeros(layout.state_dim)
    selector[_seasonal_slice(layout)] = _seasonal_selector_block(layout)
    return selector


def shock_names(layout: Layout) -> tuple[str, ...]:
    """The shock vector loaded by ``shock_loading``, in column order."""
    count = _seasonal_loading_block(layout).shape[1]
    seasonal = tuple(f"eta_seasonal_{k}" for k in range(count))
    return (*TREND_SHOCKS, *seasonal, *NEWS_SHOCKS)


def transition_matrix(layout: Layout) -> np.ndarray:
    """The (n, n) transition A: x_t = A x_{t-1} + c_t + L w_t."""
    level, slope, lagged = (_index(layout, name) for name in TREND_STATES)
    seasonal = _seasonal_slice(layout)
    matrix = np.zeros((layout.state_dim, layout.state_dim))
    matrix[level, [level, slope]] = 1.0  # level_t = level_{t-1} + slope_{t-1} + shocks
    matrix[slope, slope] = 1.0
    matrix[lagged, level] = 1.0  # lagged_value_t = level_{t-1} + gamma_{t-1}
    matrix[lagged, seasonal] = _seasonal_selector_block(layout)
    matrix[seasonal, seasonal] = _seasonal_transition(layout)
    # The news rows stay zero: each month's news is a fresh shock.
    return matrix


def shock_loading(layout: Layout) -> np.ndarray:
    """The (n, k) loading L of the shocks in ``shock_names`` order onto the states."""
    names = shock_names(layout)
    level, slope = _index(layout, "level"), _index(layout, "slope")
    loading = np.zeros((layout.state_dim, len(names)))
    loading[level, names.index("eta_level")] = 1.0
    loading[slope, names.index("eta_slope")] = 1.0
    seasonal_block = _seasonal_loading_block(layout)
    first_shock = len(TREND_SHOCKS)
    loading[
        _seasonal_slice(layout), first_shock : first_shock + seasonal_block.shape[1]
    ] = seasonal_block
    for state, shock in zip(NEWS_STATES, NEWS_SHOCKS, strict=True):
        # Each news shock is both this month's news state and part of the level's move.
        loading[_index(layout, state), names.index(shock)] = 1.0
        loading[level, names.index(shock)] = 1.0
    return loading


def observation_matrix(layout: Layout) -> np.ndarray:
    """The (4, n) rows of the F, S, T same-release changes and the March anchor.

    With dx_t = level_t + gamma_t - lagged_value_t, a print is the true change less the
    news still to come: F misses all three news shocks, S misses T's and post-T's, and T
    misses post-T's. The anchor observes the level x_t itself.
    """
    gamma = seasonal_selector(layout)
    value = gamma.copy()
    value[_index(layout, "level")] = 1.0
    change = value.copy()
    change[_index(layout, "lagged_value")] = -1.0
    news = {
        state: np.eye(layout.state_dim)[_index(layout, state)] for state in NEWS_STATES
    }
    rows = {
        "F": change - news["news_s"] - news["news_t"] - news["news_post"],
        "S": change - news["news_t"] - news["news_post"],
        "T": change - news["news_post"],
        "anchor": value,
    }
    return np.stack([rows[column] for column in COLUMNS])


def seasonal_variance_factor(layout: Layout) -> float:
    """Var(gamma_t - gamma_{t-12}) per unit seasonal shock variance, exactly.

    Every layout's seasonal block C satisfies C^12 = I (its eigenvalues are twelfth roots
    of unity), so gamma_t - gamma_{t-12} = s' sum_{h=0}^{11} C^h L w_{t-h} whatever the
    state at t - 12, and its variance per unit shock variance is sum_h ||s' C^h L||^2.
    Dummy: gamma_t - gamma_{t-12} = w_t - w_{t-1}, so 2. Trigonometric: each harmonic
    adds 12 (a rotation keeps the shock's norm for 12 months), so 72 for six harmonics
    and 48 for four.
    """
    block = _seasonal_transition(layout)
    if not np.allclose(
        np.linalg.matrix_power(block, PERIOD), np.eye(layout.seasonal_dim)
    ):
        raise ValueError(f"layout {layout.name!r} is not periodic with period {PERIOD}")
    selector = _seasonal_selector_block(layout)
    loading = _seasonal_loading_block(layout)
    return float(
        sum(
            np.sum((selector @ np.linalg.matrix_power(block, lag) @ loading) ** 2)
            for lag in range(PERIOD)
        )
    )


# Each layout's sigma_seasonal prior implies the dummy's Var(gamma_t - gamma_{t-12}) per
# unit shock: scale = 50 * sqrt(v(dummy) / v(layout)), with v(dummy) = 2, v(trig) = 72,
# and v(trig4) = 48 (``seasonal_variance_factor``), so trig = 50 / 6 and
# trig4 = 50 / sqrt(24).
SEASONAL_PRIOR_SCALE = {
    name: DUMMY_SEASONAL_PRIOR_SCALE
    * math.sqrt(
        seasonal_variance_factor(LAYOUTS["dummy"]) / seasonal_variance_factor(layout)
    )
    for name, layout in LAYOUTS.items()
}
