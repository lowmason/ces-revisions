"""The legal public holidays of 5 U.S.C. 6103, as observed each year, and federal business days.

The holidays are those of subsection (a), computed by rule for every year from 1978, when
Veterans Day returned to November 11. The King holiday begins in 1986 and Juneteenth in 2021.
Under subsection (b) and Executive Order 11582, a holiday that falls on a Saturday is observed
the Friday before and one on a Sunday the Monday after, so New Year's Day can be observed on
December 31 of the year before. Inauguration Day, a holiday under subsection (c) only for
employees in the Washington area, and closures by executive order are not holidays here.
"""

from datetime import date, timedelta

FIRST_RULE_YEAR = 1978
KING_BIRTHDAY_FIRST_YEAR = 1986
JUNETEENTH_FIRST_YEAR = 2021
MONDAY, THURSDAY, SATURDAY, SUNDAY = 0, 3, 5, 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == SUNDAY:
        return day + timedelta(days=1)
    return day


def federal_holidays(year: int) -> tuple[date, ...]:
    """The observed dates of the year's legal public holidays, in date order."""
    if year < FIRST_RULE_YEAR:
        raise ValueError(f"the holiday rule holds from {FIRST_RULE_YEAR}, not {year}")
    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    if year >= JUNETEENTH_FIRST_YEAR:
        fixed.append(date(year, 6, 19))
    floating = [
        _nth_weekday(year, 2, MONDAY, 3),
        _last_weekday(year, 5, MONDAY),
        _nth_weekday(year, 9, MONDAY, 1),
        _nth_weekday(year, 10, MONDAY, 2),
        _nth_weekday(year, 11, THURSDAY, 4),
    ]
    if year >= KING_BIRTHDAY_FIRST_YEAR:
        floating.append(_nth_weekday(year, 1, MONDAY, 3))
    return tuple(sorted([_observed(day) for day in fixed] + floating))


def business_days(start: date, end: date) -> int:
    """Weekdays from start up to, but not including, end that are not holidays.

    The holidays of the year after end's are included, because that year's New Year's Day can
    be observed on December 31.
    """
    holidays = {
        day
        for year in range(start.year, end.year + 2)
        for day in federal_holidays(year)
    }
    days = (start + timedelta(days=offset) for offset in range((end - start).days))
    return sum(day.weekday() < SATURDAY and day not in holidays for day in days)
