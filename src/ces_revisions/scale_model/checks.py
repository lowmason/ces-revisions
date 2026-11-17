"""Predictive statistics of the scale model, their p-values, and the reference-year blocks.

The statistics are the recommendation's posterior predictive checks: the kurtosis of the
revision standardized by each draw's location and scale, mean absolute revision by calendar
month, by survey interval and by window cell, and the autocorrelation of r and |r| at lags 1
and 12. A lag pairs months that are exactly that many calendar months apart, so a pair never
spans the held-out March 2020 to December 2021 as if the months were neighbours.
"""

from datetime import date

import numpy as np

from ces_revisions.scale_model.data import NUM_MONTHS, ScaleData

ACF_LAGS = (1, 12)
# Window cells in business days; the ends pool the few months outside 13 to 19.
DAY_CELLS = {
    "le13": (None, 13),
    "14": (14, 14),
    "15": (15, 15),
    "16": (16, 16),
    "17": (17, 17),
    "18": (18, 18),
    "ge19": (19, None),
}


def month_gaps(
    reference_months: tuple[date, ...], lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i, j) with month j exactly ``lag`` calendar months after month i."""
    ordinal = np.array([month.year * 12 + month.month for month in reference_months])
    position = {value: index for index, value in enumerate(ordinal)}
    pairs = [
        (index, position[value + lag])
        for index, value in enumerate(ordinal)
        if value + lag in position
    ]
    first, second = zip(*pairs, strict=True) if pairs else ((), ())
    return np.array(first, dtype=np.int64), np.array(second, dtype=np.int64)


def acf_at_lag(x: np.ndarray, reference_months: tuple[date, ...], lag: int):
    """Autocorrelation over the month pairs ``lag`` apart, along the last axis.

    The series is centered on its own mean over every month, and the lagged products are
    averaged over the pairs and divided by the variance over every month.
    """
    x = np.asarray(x, dtype=np.float64)
    first, second = month_gaps(reference_months, lag)
    if first.size == 0:
        return np.full(x.shape[:-1], np.nan) if x.ndim > 1 else np.nan
    centered = x - x.mean(axis=-1, keepdims=True)
    products = (centered[..., first] * centered[..., second]).mean(axis=-1)
    return products / (centered**2).mean(axis=-1)


def pearson_kurtosis(x: np.ndarray):
    """The fourth standardized moment along the last axis (3 for a normal)."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    return (centered**4).mean(axis=-1) / (centered**2).mean(axis=-1) ** 2


def _masked_mean(values: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return np.full(values.shape[:-1], np.nan) if values.ndim > 1 else np.nan
    return values[..., mask].mean(axis=-1)


def predictive_statistics(
    revision: np.ndarray, mu: np.ndarray, sigma: np.ndarray, data: ScaleData
) -> dict:
    """Each check statistic of ``revision``, one value per leading index (draw).

    ``mu`` and ``sigma`` standardize the revision for the kurtosis; pass the same draw's
    values for the observed and the replicated revisions.
    """
    revision = np.asarray(revision, dtype=np.float64)
    absolute = np.abs(revision)
    values = {
        "kurtosis_standardized": pearson_kurtosis(
            (revision - np.asarray(mu)) / np.asarray(sigma)
        )
    }
    for lag in ACF_LAGS:
        values[f"acf_r_lag{lag}"] = acf_at_lag(revision, data.reference_months, lag)
        values[f"acf_abs_lag{lag}"] = acf_at_lag(absolute, data.reference_months, lag)
    for month in range(NUM_MONTHS):
        values[f"mean_abs_month_{month + 1:02d}"] = _masked_mean(
            absolute, data.month == month
        )
    values["mean_abs_five_week"] = _masked_mean(absolute, data.five_week == 1)
    values["mean_abs_four_week"] = _masked_mean(absolute, data.five_week == 0)
    for name, (low, high) in DAY_CELLS.items():
        days = data.collection_days
        mask = np.ones(days.shape, dtype=bool)
        if low is not None:
            mask &= days >= low
        if high is not None:
            mask &= days <= high
        values[f"mean_abs_days_{name}"] = _masked_mean(absolute, mask)
    return values


def predictive_p_values(observed: dict, replicated: dict) -> dict:
    """P(T(y_rep) >= T(y)) for each statistic; None where the observed value is undefined.

    An observed value is a scalar or one value per draw (a discrepancy that depends on the
    draw's parameters), paired with the replicated value of the same draw.
    """
    p_values = {}
    for name, value in observed.items():
        value = np.asarray(value, dtype=np.float64)
        if np.all(np.isnan(value)):
            p_values[name] = None
            continue
        p_values[name] = float(np.mean(np.asarray(replicated[name]) >= value))
    return p_values


def sum_by_year(pointwise: np.ndarray, data: ScaleData) -> np.ndarray:
    """The pointwise log likelihood summed within each reference year, (..., Y)."""
    pointwise = np.asarray(pointwise, dtype=np.float64)
    return np.stack(
        [
            pointwise[..., data.year == index].sum(axis=-1)
            for index in range(len(data.years))
        ],
        axis=-1,
    )


# The recommendation's prior predictive targets: simulated |r| should cover about 2 to
# 300 thousand, and month-to-month scale ratios should stay within about 0.3 to 3.
ABS_TARGET = (2.0, 300.0)
RATIO_TARGET = (0.3, 3.0)
# Thousands: above every total-nonfarm F->S or F->T revision of 1979-2026 (largest 839).
IMPLAUSIBLE_ABS = 1000.0
ABS_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
RATIO_QUANTILES = (0.05, 0.5, 0.95)


def prior_predictive_record(
    revision: np.ndarray, log_sigma: np.ndarray, data: ScaleData
) -> dict:
    """Prior predictive |r| and lag-1 scale ratios against the recommendation's targets.

    ``revision`` and ``log_sigma`` are (draws, T). |r| covers its target when its 1%
    quantile is at most the low end and its 99% quantile at least the high end; the
    ratios stay within theirs when the central 90% interval does.
    """
    absolute = np.abs(np.asarray(revision, dtype=np.float64)).ravel()
    first, second = month_gaps(data.reference_months, 1)
    log_sigma = np.asarray(log_sigma, dtype=np.float64)
    ratio = np.exp(log_sigma[:, second] - log_sigma[:, first]).ravel()
    abs_q = {str(q): float(np.quantile(absolute, q)) for q in ABS_QUANTILES}
    ratio_q = {str(q): float(np.quantile(ratio, q)) for q in RATIO_QUANTILES}
    return {
        "num_draws": int(np.shape(revision)[0]),
        "abs_revision_quantiles": abs_q,
        "share_abs_above_1000": float(np.mean(absolute > IMPLAUSIBLE_ABS)),
        "scale_ratio_lag1_quantiles": ratio_q,
        "targets": {
            "abs_revision": list(ABS_TARGET),
            "scale_ratio": list(RATIO_TARGET),
        },
        "abs_covers_target": abs_q["0.01"] <= ABS_TARGET[0]
        and abs_q["0.99"] >= ABS_TARGET[1],
        "ratio_within_target": ratio_q["0.05"] >= RATIO_TARGET[0]
        and ratio_q["0.95"] <= RATIO_TARGET[1],
    }


# Half the data-only SE of beta, about 0.033 per day (recommendation, section 5).
SENSITIVITY_TOLERANCE = 0.017


def beta_sensitivity(summaries: dict[float, dict]) -> dict:
    """The recommendation's prior-sensitivity rule for beta.

    ``summaries`` maps beta's prior SD to its posterior summary. The rule passes when the
    posterior mean moves by less than SENSITIVITY_TOLERANCE between the 0.1 and 0.2 runs.
    The 0.05 run shrinks the posterior SD by construction, so it is reported, not judged.
    """
    if 0.1 not in summaries or 0.2 not in summaries:
        raise ValueError("beta sensitivity needs the runs at prior SD 0.1 and 0.2")
    shift = abs(summaries[0.2]["mean"] - summaries[0.1]["mean"])
    return {
        "by_prior_sd": {str(sd): summary for sd, summary in sorted(summaries.items())},
        "shift_0.1_to_0.2": shift,
        "tolerance": SENSITIVITY_TOLERANCE,
        "passed": shift < SENSITIVITY_TOLERANCE,
    }
