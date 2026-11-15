"""State and cell counts of the pilot and of the Stage 7-9 models, for the GPU measurement.

The engine's cost per step is cubic in the state dimension and grows with the number of
observed cells, so the Stage 6 cost measurement times the engine at the dimensions the
later models will have. Each count below names its terms; ``S`` is the seasonal block's
size under the chosen layout, which is the largest share of each sector's state.
"""

# Seasonal states per series: the dummy recursion and full trigonometric form carry 11,
# the reduced trigonometric form (harmonics 1-4, no Nyquist term) carries 8.
SEASONAL_DIMS = {"dummy": 11, "trig": 11, "trig4": 8}
# The pilot's cells per month: the F, S, and T closing changes and the March anchor.
PILOT_CELLS = 4
# Level, slope, and the lagged value x_{t-1} that turns levels into same-release changes.
TREND_STATES = 3
# Per-series news states at Stage 6 and 7: S news, T news, and post-T news.
CLOSING_NEWS_STATES = 3
# From Stage 8 the post-T news splits into its benchmark (B) and post-benchmark (M) parts.
BENCHMARK_NEWS_STATES = 4
# Stage 9's sector-specific factor news n^A, one state per news stage (S, T, B, M).
FACTOR_NEWS_STATES = 4
SECTORS = 11  # Req 5's eleven supersectors
SERIES = 12  # the eleven supersectors and total nonfarm
COMMON_FACTORS = 1  # the common level factor
CLOSINGS = 3  # F, S, and T rows per series
# Wedge "high" carries the benchmark's backward window as 11 lagged months per sector.
WEDGE_WINDOW = 11
STAGES = (7, 8, 9)
WEDGES = ("low", "high")
# Every Stage 7-9 model the GPU measurement times: Stage 7 has no wedge rows, and Stage 8
# chooses the wedge reading that Stage 9 inherits, so both readings are timed.
STAGE_CONFIGURATIONS = ((7, "low"), (8, "low"), (8, "high"), (9, "low"), (9, "high"))
# 1979-01 through 2026-08, the aggregate leg's full span, and 2003-05 through 2026-08, the
# sector panel's span (Req 4: no sector vintage precedes May 2003).
FULL_STEPS = 572
SECTOR_STEPS = 280


def pilot_dimensions(layout_name: str, num_steps: int = FULL_STEPS) -> dict:
    """The Stage 6 pilot's dimensions: one series with trend, seasonal, and news states."""
    seasonal = _seasonal_dim(layout_name)
    return {
        "steps": num_steps,
        "states": TREND_STATES + seasonal + CLOSING_NEWS_STATES,
        "cells": PILOT_CELLS,
        "stage": 6,
        "layout": layout_name,
        "wedge": None,
    }


def stage_dimensions(
    layout_name: str, stage: int, wedge: str = "low", num_steps: int = FULL_STEPS
) -> dict:
    """The Stage 7, 8, or 9 model's state and cell counts under a layout and wedge.

    Stage 7 has no wedge rows, so it accepts only ``wedge="low"``, which adds nothing.
    """
    seasonal = _seasonal_dim(layout_name)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage}")
    if wedge not in WEDGES:
        raise ValueError(f"wedge must be one of {WEDGES}, not {wedge!r}")
    if stage == 7 and wedge != "low":
        raise ValueError("Stage 7 has no wedge rows: wedge must be 'low'")

    # Every series carries its F, S, and T closing rows; total nonfarm adds its anchor.
    closing_cells = SERIES * CLOSINGS + 1
    if stage == 7:
        per_sector = TREND_STATES + seasonal + CLOSING_NEWS_STATES
        # One common factor and the two cross-sector news factors, for S and T.
        factors = COMMON_FACTORS + 2
        states = SECTORS * per_sector + factors
        cells = closing_cells
    else:
        per_sector = TREND_STATES + seasonal + BENCHMARK_NEWS_STATES
        if stage == 9:
            per_sector += FACTOR_NEWS_STATES
        # One common factor and the four cross-sector news factors, for S, T, B, and M.
        factors = COMMON_FACTORS + BENCHMARK_NEWS_STATES
        states = SECTORS * per_sector + factors
        # Per series: the B and M rows, then the preliminary and final benchmark rows.
        cells = closing_cells + SERIES * 2 + SERIES * 2
        if wedge == "high":
            states += SECTORS * WEDGE_WINDOW
            # The benchmark's backward window enters as an 11-month B vector per series.
            cells += SERIES * WEDGE_WINDOW
        if stage == 9:
            # Stage 9 adds the seasonally adjusted F, S, T, B, and M rows per series.
            cells += SERIES * 5
    return {
        "steps": num_steps,
        "states": states,
        "cells": cells,
        "stage": stage,
        "layout": layout_name,
        "wedge": wedge,
    }


def matches_layout(entry: dict, layout_name: str) -> bool:
    """Whether a probe-grid entry has a layout's Stage 7-9 dimensions.

    The entry's states and cells must equal ``stage_dimensions`` at its own stage, wedge,
    and steps. ``trig`` and the dummy have equal dimensions, so each reads the other's
    entries; an entry without a Stage 7-9 configuration matches no layout.
    """
    try:
        expected = stage_dimensions(
            layout_name, entry.get("stage"), entry.get("wedge"), entry["steps"]
        )
    except ValueError:
        return False
    return all(entry[key] == expected[key] for key in ("states", "cells"))


def _seasonal_dim(layout_name: str) -> int:
    if layout_name not in SEASONAL_DIMS:
        raise ValueError(
            f"layout must be one of {tuple(SEASONAL_DIMS)}, not {layout_name!r}"
        )
    return SEASONAL_DIMS[layout_name]


# The configurations the Stage 6 cost measurement times on each GPU type: the dummy (whose
# dimensions ``trig`` shares) and ``trig4``, every Stage 7-9 model, at the full span (a
# fit) and the sector span (an SBC replicate).
MEASUREMENT_GRID = tuple(
    stage_dimensions(layout_name, stage, wedge, num_steps)
    for layout_name in ("dummy", "trig4")
    for stage, wedge in STAGE_CONFIGURATIONS
    for num_steps in (FULL_STEPS, SECTOR_STEPS)
)
