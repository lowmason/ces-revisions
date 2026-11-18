"""The collection-window day count `D_t` and the window to first closing `W_t` of each release.

`D_t` counts the federal business days from the first business day after the 12th of the
reference month up to, but not including, the release date, so `D_t` keeps the release date as
its end. `first_closing_date` and `first_closing_days` (`W_t`) follow the owner's rule (owner,
2026-11-17): first closing, when BLS stops collecting survey responses for the first estimate,
falls one week before the release. `W_t` counts the same business days through the closing day.
Days of a lapse in appropriations count as business days.
"""

from datetime import date, timedelta

import polars as pl

from ces_revisions.vintages.federal_calendar import business_days

# The CES reference period is the pay period that includes the 12th of the month.
REFERENCE_DAY = 12
# The owner's rule: first closing falls one week before the release.
FIRST_CLOSING_LEAD_DAYS = 7
# The six releases that a lapse in appropriations delayed. December 1995: footnote 1 of BLS's
# historical release dates (data/raw/bls/histreleasedates.txt, keyed in
# data/raw/manual/es-reschedules.csv). The others: the shutdown_release_delayed row of
# docs/ces-revisions-review.md. Stage 1's analysis drops September 2025, so five of them enter
# it. The January 2018 and 2018-19 lapses delayed no release, and the index's off_schedule flag
# is wider: it also marks October 1998 and the canceled October 2025.
LAPSE_DELAYED_MONTHS = (
    date(1995, 12, 1),
    date(2013, 9, 1),
    date(2013, 10, 1),
    date(2025, 9, 1),
    date(2025, 11, 1),
    date(2026, 1, 1),
)
SCHEMA = {
    "reference_month": pl.Date,
    "release_date": pl.Date,
    "collection_days": pl.Int32,
    "lapse_delayed": pl.Boolean,
    "first_closing_date": pl.Date,
    "first_closing_days": pl.Int32,
}


def collection_days(reference_month: date, release_date: date) -> int:
    """`D_t` for the release of a reference month made on release_date."""
    return business_days(reference_month.replace(day=REFERENCE_DAY + 1), release_date)


def first_closing_date(release_date: date) -> date:
    """The first closing of a release made on release_date, by the owner's rule."""
    return release_date - timedelta(days=FIRST_CLOSING_LEAD_DAYS)


def first_closing_days(reference_month: date, release_date: date) -> int:
    """`W_t`: the business days from the 13th of the reference month through first closing."""
    return business_days(
        reference_month.replace(day=REFERENCE_DAY + 1),
        first_closing_date(release_date) + timedelta(days=1),
    )


def build_collection_days(index: pl.DataFrame) -> pl.DataFrame:
    """`D_t` and `W_t` for every release in the release-date index, with its lapse flag."""
    released = index.filter(pl.col("status") == "released").sort("reference_month")
    rows = [
        {
            "reference_month": month,
            "release_date": released_on,
            "collection_days": collection_days(month, released_on),
            "lapse_delayed": month in LAPSE_DELAYED_MONTHS,
            "first_closing_date": first_closing_date(released_on),
            "first_closing_days": first_closing_days(month, released_on),
        }
        for month, released_on in released.select(
            "reference_month", "release_date"
        ).iter_rows()
    ]
    return pl.DataFrame(rows, schema=SCHEMA)
