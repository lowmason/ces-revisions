"""The aggregate total-nonfarm observation panel of the Stage 6 pilot.

One row per reference month from January 1979, and four cells per row: the first,
second, and third same-release over-the-month changes of not seasonally adjusted total
nonfarm employment, and the final March benchmark anchor, all in thousands. A NaN cell
is not observed: never published, missing in the 2003 gap or the 2025 lapse, beyond the
release index, a non-March anchor cell, or masked because a benchmark release carried it.
"""

import itertools
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ces_revisions.vintages.panel import REGIME_SHIFT_MONTH

FIRST_MONTH = date(1979, 1, 1)
CLOSING_STAGES = ("F", "S", "T")
COLUMNS = (*CLOSING_STAGES, "anchor")
ANCHOR_COLUMN = COLUMNS.index("anchor")
# Req 6's known crisis window, March through June 2020.
CRISIS_MONTHS = tuple(date(2020, month, 1) for month in range(3, 7))
# Benchmark years with a final anchor on the vintage-file basis (Stage 5's wedge fixtures).
ANCHOR_YEARS = tuple(range(2003, 2026))
# A same-release change differences two independently rounded whole-thousand levels;
# an anchor is one rounded level (Req 15's 1/12).
CLOSING_ROUNDING_VARIANCE = 2.0 / 12.0
ANCHOR_ROUNDING_VARIANCE = 1.0 / 12.0
CELL_SCHEMA = {
    "reference_month": pl.Date,
    "column": pl.String,
    "source": pl.String,
    "value_thousands": pl.Float64,
    "status": pl.String,
    "mask_reason": pl.String,
    "published_date": pl.Date,
}

__all__ = [
    "ANCHOR_COLUMN",
    "ANCHOR_ROUNDING_VARIANCE",
    "ANCHOR_YEARS",
    "CELL_SCHEMA",
    "CLOSING_ROUNDING_VARIANCE",
    "CLOSING_STAGES",
    "COLUMNS",
    "CRISIS_MONTHS",
    "FIRST_MONTH",
    "REGIME_SHIFT_MONTH",
    "AggregatePanel",
    "benchmark_year_of",
]


def benchmark_year_of(month: date) -> int:
    """The benchmark year whose April-to-March window holds ``month`` (Req 10's I_y)."""
    return month.year + 1 if month.month >= 4 else month.year


@dataclass(frozen=True)
class AggregatePanel:
    """The pilot's observations and the per-month flags the model reads.

    ``y`` holds the published values exactly; the anchor cell's offset for earlier
    reconstructions is kept apart in ``anchor_offset`` so the model adds it as an
    observation offset. ``cells`` is the audit table: one row per (month, column).
    """

    reference_months: tuple[date, ...]
    y: np.ndarray  # (T, 4) float64
    anchor_offset: np.ndarray  # (T,) float64; zero outside anchor months
    pre_2003: np.ndarray  # (T,) bool; reference month before REGIME_SHIFT_MONTH
    crisis: np.ndarray  # (T,) bool; reference month in CRISIS_MONTHS
    benchmark_year: np.ndarray  # (T,) int64
    cells: pl.DataFrame  # CELL_SCHEMA

    def __post_init__(self) -> None:
        num_steps = len(self.reference_months)
        if self.y.shape != (num_steps, len(COLUMNS)) or self.y.dtype != np.float64:
            raise ValueError(
                f"y must be float64 of shape ({num_steps}, {len(COLUMNS)})"
            )
        for name, dtype in (
            ("anchor_offset", np.float64),
            ("pre_2003", np.bool_),
            ("crisis", np.bool_),
            ("benchmark_year", np.int64),
        ):
            array = getattr(self, name)
            if array.shape != (num_steps,) or array.dtype != dtype:
                raise ValueError(
                    f"{name} must be {np.dtype(dtype)} of shape ({num_steps},)"
                )
        months = self.reference_months
        if any(
            (later.year - earlier.year) * 12 + later.month - earlier.month != 1
            for earlier, later in itertools.pairwise(months)
        ):
            raise ValueError("reference_months must be consecutive months")
        if dict(self.cells.schema) != CELL_SCHEMA:
            raise ValueError(f"cells must have schema {CELL_SCHEMA}")

    @property
    def num_steps(self) -> int:
        return len(self.reference_months)

    def observed_any(self) -> np.ndarray:
        """Steps with at least one observed cell; LOO drops the rest (they add zero)."""
        return ~np.isnan(self.y).all(axis=1)

    def window(self, first: date, last: date) -> AggregatePanel:
        """The panel restricted to reference months first..last, both included."""
        keep = np.array([first <= month <= last for month in self.reference_months])
        return AggregatePanel(
            reference_months=tuple(
                month
                for month, kept in zip(self.reference_months, keep, strict=True)
                if kept
            ),
            y=self.y[keep].copy(),
            anchor_offset=self.anchor_offset[keep].copy(),
            pre_2003=self.pre_2003[keep].copy(),
            crisis=self.crisis[keep].copy(),
            benchmark_year=self.benchmark_year[keep].copy(),
            cells=self.cells.filter(pl.col("reference_month").is_between(first, last)),
        )
