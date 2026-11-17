"""The scale model's design arrays: the 2003+ sample and the window centered within year."""

from datetime import date
from functools import cache

import numpy as np
import polars as pl
import pytest
import vintage_data

from ces_revisions.collection_window import signed_revisions
from ces_revisions.scale_model.data import SERIES, scale_data


@cache
def revisions() -> pl.DataFrame:
    return signed_revisions(
        vintage_data.table_changes(), vintage_data.collection_days()
    )


def synthetic() -> pl.DataFrame:
    """Two reference years; 2013 holds a lapse month with a long window."""
    rows = [
        (date(2002, 12, 1), 14, False, "main"),
        (date(2013, 1, 1), 14, False, "main"),
        (date(2013, 2, 1), 16, False, "main"),
        (date(2013, 9, 1), 26, True, "main"),
        (date(2014, 3, 1), 13, False, "main"),
        (date(2014, 4, 1), 17, False, "main"),
        (date(2020, 4, 1), 15, False, "pandemic"),
    ]
    return pl.DataFrame(
        {
            "reference_month": [row[0] for row in rows],
            "seasonal_status": ["NSA"] * len(rows),
            "transition": ["F_S"] * len(rows),
            "revision": [10, -20, 30, -40, 50, -60, 70],
            "collection_days": [row[1] for row in rows],
            "lapse_delayed": [row[2] for row in rows],
            "period": [row[3] for row in rows],
            "five_week": [False, True, False, True, False, False, True],
        }
    ).with_columns(pl.col("collection_days").cast(pl.Int32))


def test_sample_keeps_main_period_months_from_2003():
    data = scale_data(synthetic(), ("NSA", "F_S"))
    assert data.reference_months == (
        date(2013, 1, 1),
        date(2013, 2, 1),
        date(2013, 9, 1),
        date(2014, 3, 1),
        date(2014, 4, 1),
    )
    np.testing.assert_array_equal(data.revision, [-20, 30, -40, 50, -60])
    assert data.revision.dtype == np.float64


def test_window_is_centered_on_the_non_lapse_months_of_each_year():
    data = scale_data(synthetic(), ("NSA", "F_S"))
    # 2013's non-lapse mean is 15, 2014's is 15. The lapse month sits 11 days above, but
    # a shutdown set that window, so it enters as 0 and the lapse indicator absorbs it.
    np.testing.assert_array_equal(data.window, [-1, 1, 0, -2, 2])
    np.testing.assert_array_equal(data.collection_days, [14, 16, 26, 13, 17])
    # The sample's non-lapse mean is 15, so neither year differs from it.
    np.testing.assert_array_equal(data.year_window, [0, 0])
    np.testing.assert_array_equal(data.lapse, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(data.five_week, [1, 0, 1, 0, 0])


def test_year_window_is_the_year_mean_less_the_sample_mean():
    frame = synthetic().with_columns(
        collection_days=pl.when(pl.col("reference_month") == date(2014, 4, 1))
        .then(21)
        .otherwise(pl.col("collection_days"))
        .cast(pl.Int32)
    )
    data = scale_data(frame, ("NSA", "F_S"))
    # Non-lapse days 14, 16, 13, 21: sample mean 16, year means 15 and 17.
    np.testing.assert_array_equal(data.year_window, [-1, 1])
    np.testing.assert_array_equal(data.window, [-1, 1, 0, -4, 4])


def test_indices_number_the_years_present_and_the_calendar_months():
    data = scale_data(synthetic(), ("NSA", "F_S"))
    assert data.years == (2013, 2014)
    np.testing.assert_array_equal(data.year, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(data.month, [0, 1, 8, 2, 3])


def test_unknown_series_is_refused():
    with pytest.raises(ValueError, match="series"):
        scale_data(synthetic(), ("NSA", "S_T"))


def test_a_year_of_lapse_months_only_is_refused():
    frame = synthetic().filter(pl.col("reference_month").dt.year() != 2014)
    frame = frame.with_columns(lapse_delayed=pl.lit(True))
    with pytest.raises(ValueError, match="no month without a lapse"):
        scale_data(frame, ("NSA", "F_S"))


@pytest.mark.parametrize(
    ("series", "count", "five_week"),
    [
        (("NSA", "F_S"), 258, 90),
        (("SA", "F_S"), 258, 90),
        (("NSA", "F_T"), 256, 89),
        (("SA", "F_T"), 256, 89),
    ],
)
def test_real_sample_matches_the_recommendation(series, count, five_week):
    data = scale_data(revisions(), series)
    assert series in SERIES
    assert len(data.reference_months) == count
    assert len(data.years) == 23
    assert 2021 not in data.years
    assert data.five_week.sum() == five_week
    assert data.lapse.sum() == 4
    assert data.reference_months[0] == date(2003, 1, 1)
    # The non-lapse months of each year center to zero; the lapse months carry none.
    keep = data.lapse == 0
    assert not data.window[~keep].any()
    for index in range(len(data.years)):
        in_year = keep & (data.year == index)
        assert abs(data.window[in_year].sum()) < 1e-9


def test_real_primary_series_has_the_recommendation_moments():
    data = scale_data(revisions(), ("NSA", "F_S"))
    assert round(data.revision.mean(), 1) == 9.0
    assert round(data.revision.std(), 1) == 56.8
