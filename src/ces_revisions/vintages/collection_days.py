"""The collection-window day count `D_t` of each Employment Situation release.

`D_t` counts the federal business days from the first business day after the 12th of the
reference month up to, but not including, the release date. BLS publishes no first-closing
dates (Req 3), so the release date stands in for the end of first-closing collection. Days of a
lapse in appropriations count as business days.
"""

from datetime import date

from ces_revisions.vintages.federal_calendar import business_days

# The CES reference period is the pay period that includes the 12th of the month.
REFERENCE_DAY = 12


def collection_days(reference_month: date, release_date: date) -> int:
    """`D_t` for the release of a reference month made on release_date."""
    return business_days(reference_month.replace(day=REFERENCE_DAY + 1), release_date)
