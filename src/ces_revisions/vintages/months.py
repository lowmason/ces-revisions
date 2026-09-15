"""Month arithmetic on dates that stand for a month by its first day."""

from datetime import date

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def add_months(month: date, count: int) -> date:
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def month_range(first: date, last: date) -> list[date]:
    """Every month from `first` through `last`, both included."""
    months = []
    month = first
    while month <= last:
        months.append(month)
        month = add_months(month, 1)
    return months
