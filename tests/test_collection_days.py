"""The collection-window day count `D_t`: federal business days after the 12th, before release."""

from datetime import date

from ces_revisions.vintages.collection_days import collection_days


def test_window_runs_from_after_the_12th_to_before_the_release():
    assert collection_days(date(2026, 8, 1), date(2026, 9, 4)) == 16


def test_december_window_skips_observed_saturday_holidays():
    assert collection_days(date(2010, 12, 1), date(2011, 1, 7)) == 17
    assert collection_days(date(2021, 12, 1), date(2022, 1, 7)) == 17


def test_window_starts_after_a_columbus_day_on_the_13th():
    assert collection_days(date(2014, 10, 1), date(2014, 11, 7)) == 18


def test_holidays_count_from_their_first_observance():
    assert collection_days(date(1985, 1, 1), date(1985, 2, 1)) == 14
    assert collection_days(date(1986, 1, 1), date(1986, 2, 7)) == 18
    assert collection_days(date(2021, 6, 1), date(2021, 7, 2)) == 13
    assert collection_days(date(2026, 6, 1), date(2026, 7, 2)) == 12


def test_lapse_days_count_as_business_days():
    assert collection_days(date(2026, 1, 1), date(2026, 2, 11)) == 20
    assert collection_days(date(2013, 9, 1), date(2013, 10, 22)) == 26
