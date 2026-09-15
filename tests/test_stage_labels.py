"""Req 2's stage labels for the vintage files and for BLS's revision table."""

from datetime import date

import polars as pl
import pytest
import vintage_data

from ces_revisions.vintages import raw, stages
from ces_revisions.vintages.months import month_range
from ces_revisions.vintages.stages import comment_release_month, stage_release_months

UNIT = ["source", "sector", "reference_month", "seasonal_status"]
LAPSE_GAPS = {(date(2025, 9, 1), "S"), (date(2025, 10, 1), "F")}
REDESIGN_GAPS = {
    (date(2003, 3, 1), "T"),
    (date(2003, 4, 1), "S"),
    (date(2003, 4, 1), "T"),
}
COMMENTED_RELEASES = [
    date(2003, 5, 1),
    date(2008, 1, 1),
    date(2011, 1, 1),
    date(2012, 1, 1),
    date(2014, 1, 1),
    date(2016, 1, 1),
    date(2018, 1, 1),
    date(2019, 1, 1),
    date(2020, 1, 1),
    date(2023, 1, 1),
    date(2025, 1, 1),
    date(2025, 10, 1),
    date(2026, 1, 1),
]


@pytest.mark.parametrize(
    ("month", "status", "expected"),
    [
        (
            date(2024, 4, 1),
            "NSA",
            ((2024, 4), (2024, 5), (2024, 6), (2025, 1), (2026, 1)),
        ),
        (
            date(2024, 11, 1),
            "NSA",
            ((2024, 11), (2024, 12), (2025, 1), (2026, 1), (2027, 1)),
        ),
        (
            date(2024, 12, 1),
            "SA",
            ((2024, 12), (2025, 1), (2025, 2), (2026, 1), (2030, 1)),
        ),
        (
            date(2024, 1, 1),
            "SA",
            ((2024, 1), (2024, 2), (2024, 3), (2025, 1), (2030, 1)),
        ),
    ],
)
def test_benchmark_stages_count_from_the_third_estimate(month, status, expected):
    releases = stage_release_months(month, status)
    assert list(releases) == list(stages.STAGES)
    assert tuple((value.year, value.month) for value in releases.values()) == expected


@pytest.mark.parametrize(
    ("text", "month"),
    [
        (
            "With the release of May 2003 data on June 6, 2003, the CES national nonfarm",
            date(2003, 5, 1),
        ),
        (
            "With the 2024 benchmark, CES reconstructed several series.",
            date(2025, 1, 1),
        ),
        (
            "Due to the 2025 lapse in appropriations, no estimates for October first",
            date(2025, 10, 1),
        ),
    ],
)
def test_comment_entries_name_their_release(text, month):
    assert comment_release_month(text) == month


def test_a_comment_without_a_release_rule_is_an_error():
    with pytest.raises(ValueError, match="no release rule"):
        comment_release_month("Historical revisions, reconstructions, or adjustments")


# --- The committed sources ------------------------------------------------------------------


def test_the_committed_comments_describe_thirteen_releases():
    comments = raw.read_vintage_comments()
    assert comments.height == 16
    months = {comment_release_month(text) for text in comments["adjustment"]}
    assert sorted(months) == COMMENTED_RELEASES


def test_every_unit_carries_each_of_its_stages_once_on_distinct_releases():
    labels = vintage_data.labels()
    assert labels.select(*UNIT, "release_stage").n_unique() == labels.height
    units = labels.group_by(UNIT).agg(
        stages=pl.col("release_stage").sort().str.join(""),
        releases=pl.col("release_month").n_unique(),
    )
    expected = {"cesvinall": ("BFMST", 5), "cesnaicsrev": ("FST", 3)}
    for source, (names, count) in expected.items():
        frame = units.filter(pl.col("source") == source)
        assert frame["stages"].unique().to_list() == [names]
        assert frame["releases"].unique().to_list() == [count]


def test_missing_vintages_are_the_2025_lapse_and_the_2003_redesign():
    missing = vintage_data.labels().filter(pl.col("status") == "missing_vintage")
    files = missing.filter(pl.col("source") == "cesvinall")
    assert (
        set(files.select("reference_month", "release_stage").iter_rows()) == LAPSE_GAPS
    )
    assert files.height == len(LAPSE_GAPS) * 12 * 2
    assert files["published_date"].null_count() == files.height
    table = missing.filter(pl.col("source") == "cesnaicsrev")
    gaps = LAPSE_GAPS | REDESIGN_GAPS
    assert set(table.select("reference_month", "release_stage").iter_rows()) == gaps
    assert table.height == len(gaps) * 2


def test_right_censoring_and_the_frontier_follow_release_timing():
    labels = vintage_data.labels()
    latest = vintage_data.index()["reference_month"].max()
    frontier = vintage_data.levels()["release_month"].max()
    # cesvinall.zip of 2026-03-06 ends with the January 2026 benchmark release.
    assert frontier == date(2026, 1, 1)
    for source, last_held in (("cesvinall", frontier), ("cesnaicsrev", latest)):
        frame = labels.filter(pl.col("source") == source)
        censored = frame.filter(pl.col("status") == "right_censored")
        assert censored.height == frame.filter(pl.col("release_month") > latest).height
        assert (censored["release_month"] > latest).all()
        beyond = frame.filter(pl.col("status") == "beyond_frontier")
        held_later = frame.filter(pl.col("release_month").is_between(last_held, latest))
        assert (
            beyond.height
            == held_later.height
            - held_later.filter(pl.col("release_month") == last_held).height
        )


def test_m_is_right_censored_exactly_where_its_benchmark_release_is_still_to_come():
    latest = vintage_data.index()["reference_month"].max()
    mature = vintage_data.labels().filter(pl.col("release_stage") == "M")
    censored = mature["status"] == "right_censored"
    assert (censored == (mature["release_month"] > latest)).all()


def test_every_stage_is_an_employment_situation_release_so_no_preliminary_benchmark_is():
    labels = vintage_data.labels()
    assert set(labels["release_stage"].unique()) == set(stages.STAGES)
    releases = vintage_data.index().select(
        "publication", release_month="reference_month"
    )
    held = labels.filter(pl.col("status") != "right_censored").join(
        releases, on="release_month", how="left"
    )
    assert held["publication"].unique().to_list() == ["employment_situation"]


def test_no_vintage_file_stage_precedes_the_may_2003_publication_vintage():
    files = vintage_data.labels().filter(pl.col("source") == "cesvinall")
    assert files["reference_month"].min() == date(2003, 5, 1)
    assert files["release_month"].min() == date(2003, 5, 1)


def test_benchmark_releases_carry_january_first_december_second_november_third():
    closing = vintage_data.labels().filter(
        (pl.col("source") == "cesvinall")
        & pl.col("release_stage").is_in(list(stages.CLOSING_STAGES))
        & pl.col("benchmark_release")
    )
    # The May 2003 release carried the March 2002 benchmark, but as the vintage files' first
    # row the only closing stage it holds is May 2003's first estimate.
    may_2003 = closing.filter(pl.col("release_month") == date(2003, 5, 1))
    assert set(
        may_2003.select("reference_month", "release_stage").unique().iter_rows()
    ) == {(date(2003, 5, 1), "F")}
    januaries = closing.filter(pl.col("release_month") != date(2003, 5, 1))
    pairs = januaries.select(pl.col("reference_month").dt.month(), "release_stage")
    assert set(pairs.unique().iter_rows()) == {(1, "F"), (12, "S"), (11, "T")}


def test_nonstandard_releases_are_the_releases_the_comments_describe():
    flagged = vintage_data.labels().filter(pl.col("nonstandard_release"))
    assert sorted(set(flagged["release_month"])) == COMMENTED_RELEASES


def test_revised_after_m_is_checked_where_a_later_release_exists():
    """No release in the vintage files follows the frontier, so an M in the frontier release
    leaves revised_after_m null rather than false."""
    labels = vintage_data.labels()
    checked = labels.filter(pl.col("revised_after_m").is_not_null())
    assert set(
        checked.select("source", "release_stage", "status").unique().iter_rows()
    ) == {("cesvinall", "M", "observed")}
    observed_m = labels.filter(
        (pl.col("release_stage") == "M") & (pl.col("status") == "observed")
    )
    unchecked = observed_m.filter(pl.col("revised_after_m").is_null())
    # cesvinall.zip of 2026-03-06: the January 2026 release is M for unadjusted November 2023 to
    # October 2024 and for adjusted 2020.
    expected = {
        ("NSA", month) for month in month_range(date(2023, 11, 1), date(2024, 10, 1))
    } | {("SA", month) for month in month_range(date(2020, 1, 1), date(2020, 12, 1))}
    assert (
        set(unchecked.select("seasonal_status", "reference_month").unique().iter_rows())
        == expected
    )
    assert unchecked.height == len(expected) * len(raw.SECTORS)
    assert checked.height == observed_m.height - unchecked.height


def test_panel_levels_carry_the_stage_their_vintage_serves():
    staged = vintage_data.long_panel().filter(
        (pl.col("source") == "cesvinall") & pl.col("release_stage").is_not_null()
    )
    labeled = vintage_data.labels().filter(
        (pl.col("source") == "cesvinall") & pl.col("vintage_id").is_not_null()
    )
    assert staged.height == labeled.height
