"""One series' design arrays for the scale model, from the rows of ``signed_revisions``.

The sample is the main period from January 2003 on (the owner's choice of 2026-11-17,
because of the probability-sample redesign). The window ``D_t`` enters twice: centered within
its reference year, so the window effect rests on within-year variation only, and as the
year's mean less the sample mean (Mundlak 1978). Both means are taken over the months whose
release was not delayed by a lapse in appropriations: a lapse month's window of 18 to 26 days
would otherwise move every other month of its year by up to a day. A lapse month's own
within-year window is set to 0 (the owner's choice of 2026-11-17): a shutdown, not the
calendar, set its length, so it informs neither window effect: on log scale the lapse
indicator absorbs it, and on the mean it keeps only its month and 5-week terms.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

SAMPLE_START = date(2003, 1, 1)
SERIES = (("NSA", "F_S"), ("SA", "F_S"), ("NSA", "F_T"), ("SA", "F_T"))
PRIMARY_SERIES = ("NSA", "F_S")
NUM_MONTHS = 12


@dataclass(frozen=True, eq=False)
class ScaleData:
    """The arrays one series' model reads, one entry per reference month in time order."""

    series: tuple[str, str]
    reference_months: tuple[date, ...]
    years: tuple[int, ...]  # the reference years present, in order
    revision: np.ndarray  # (T,) float64, thousands: later estimate less first
    collection_days: np.ndarray  # (T,) float64, D_t
    window: np.ndarray  # (T,) float64, D_t less its year's non-lapse mean; 0 at a lapse
    year_window: np.ndarray  # (Y,) float64, the year's non-lapse mean less the sample's
    five_week: np.ndarray  # (T,) float64, 1 for a 5-week survey interval
    lapse: np.ndarray  # (T,) float64, 1 for a release delayed by a lapse
    year: np.ndarray  # (T,) int, index into years
    month: np.ndarray  # (T,) int, calendar month less one


def scale_data(revisions: pl.DataFrame, series: tuple[str, str]) -> ScaleData:
    """The design arrays of one (seasonal status, transition) series in the sample."""
    if series not in SERIES:
        raise ValueError(f"series must be one of {SERIES}; got {series}")
    status, transition = series
    rows = revisions.filter(
        pl.col("seasonal_status") == status,
        pl.col("transition") == transition,
        pl.col("period") == "main",
        pl.col("reference_month") >= SAMPLE_START,
    ).sort("reference_month")
    reference_months = tuple(rows["reference_month"].to_list())
    calendar_year = rows["reference_month"].dt.year().to_numpy()
    days = rows["collection_days"].to_numpy().astype(np.float64)
    lapse = rows["lapse_delayed"].to_numpy()
    years, year = np.unique(calendar_year, return_inverse=True)

    year_means = np.empty(len(years))
    for index, label in enumerate(years):
        normal = (year == index) & ~lapse
        if not normal.any():
            raise ValueError(f"reference year {label} has no month without a lapse")
        year_means[index] = days[normal].mean()
    sample_mean = days[~lapse].mean()
    return ScaleData(
        series=series,
        reference_months=reference_months,
        years=tuple(int(label) for label in years),
        revision=rows["revision"].to_numpy().astype(np.float64),
        collection_days=days,
        window=np.where(lapse, 0.0, days - year_means[year]),
        year_window=year_means - sample_mean,
        five_week=rows["five_week"].to_numpy().astype(np.float64),
        lapse=lapse.astype(np.float64),
        year=year.astype(np.int64),
        month=rows["reference_month"].dt.month().to_numpy().astype(np.int64) - 1,
    )
