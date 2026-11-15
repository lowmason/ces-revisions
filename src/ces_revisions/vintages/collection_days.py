"""The collection-window day count `D_t` of each Employment Situation release.

`D_t` counts the federal business days from the first business day after the 12th of the
reference month up to, but not including, the release date. BLS publishes no first-closing
dates (Req 3), so the release date stands in for the end of first-closing collection. Days of a
lapse in appropriations count as business days.
"""

from datetime import date

import polars as pl

from ces_revisions.vintages.federal_calendar import business_days

# The CES reference period is the pay period that includes the 12th of the month.
REFERENCE_DAY = 12
# The five releases that a lapse in appropriations delayed. December 1995: footnote 1 of BLS's
# historical release dates (data/raw/bls/histreleasedates.txt, keyed in
# data/raw/manual/es-reschedules.csv). The others: the shutdown_release_delayed row of
# docs/ces-revisions-review.md. The January 2018 and 2018-19 lapses delayed no release, and the
# index's off_schedule flag is wider: it also marks October 1998 and the canceled October 2025.
LAPSE_DELAYED_MONTHS = (
    date(1995, 12, 1),
    date(2013, 9, 1),
    date(2013, 10, 1),
    date(2025, 11, 1),
    date(2026, 1, 1),
)
SCHEMA = {
    "reference_month": pl.Date,
    "release_date": pl.Date,
    "collection_days": pl.Int32,
    "lapse_delayed": pl.Boolean,
}


def collection_days(reference_month: date, release_date: date) -> int:
    """`D_t` for the release of a reference month made on release_date."""
    return business_days(reference_month.replace(day=REFERENCE_DAY + 1), release_date)


def build_collection_days(index: pl.DataFrame) -> pl.DataFrame:
    """One `D_t` for every release in the release-date index, with its lapse flag."""
    released = index.filter(pl.col("status") == "released").sort("reference_month")
    rows = [
        {
            "reference_month": month,
            "release_date": released_on,
            "collection_days": collection_days(month, released_on),
            "lapse_delayed": month in LAPSE_DELAYED_MONTHS,
        }
        for month, released_on in released.select(
            "reference_month", "release_date"
        ).iter_rows()
    ]
    return pl.DataFrame(rows, schema=SCHEMA)
