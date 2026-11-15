"""The legal public holidays of 5 U.S.C. 6103, observed by rule, and federal business days."""

from datetime import date

import pytest

from ces_revisions.vintages.federal_calendar import business_days, federal_holidays


def test_federal_holidays_2026_match_opm():
    assert federal_holidays(2026) == (
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 10, 12),
        date(2026, 11, 11),
        date(2026, 11, 26),
        date(2026, 12, 25),
    )


def test_saturday_new_years_day_is_observed_the_friday_before():
    assert date(2010, 12, 31) in federal_holidays(2011)
    assert date(2010, 12, 31) not in federal_holidays(2010)


def test_sunday_holiday_is_observed_the_monday_after():
    assert date(2021, 7, 5) in federal_holidays(2021)


def test_holidays_follow_the_statute_in_force():
    assert [len(federal_holidays(year)) for year in (1985, 1986, 2020, 2021)] == [
        9,
        10,
        10,
        11,
    ]
    assert date(1985, 1, 21) not in federal_holidays(1985)
    assert date(1986, 1, 20) in federal_holidays(1986)
    assert date(2021, 6, 18) in federal_holidays(2021)


def test_inauguration_and_executive_order_closures_are_not_holidays():
    assert date(2021, 1, 20) not in federal_holidays(2021)
    assert date(2019, 12, 24) not in federal_holidays(2019)


def test_years_before_1978_are_refused():
    with pytest.raises(ValueError):
        federal_holidays(1977)


def test_business_days_skip_weekends_and_holidays():
    assert business_days(date(2026, 8, 13), date(2026, 9, 4)) == 16
    assert business_days(date(2026, 11, 26), date(2026, 11, 27)) == 0
    assert business_days(date(2026, 9, 4), date(2026, 9, 4)) == 0
