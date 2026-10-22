"""The pre-registered seasonal-layout rule, clause by clause, on hand-made comparisons."""

import numpy as np
import pytest

from ces_revisions.pilot.dimensions import stage_dimensions
from ces_revisions.pilot.seasonal import (
    COST_SAVING_MIN,
    LJUNG_BOX_Q_RATIO_MAX,
    PARETO_K_MAX,
    choose_layout,
    elpd_difference,
    grid_seconds,
    ljung_box_rejections,
    verdict,
)


def candidate(states, seconds, *, req18=True, april=True, q=(85.0, 50.0, 62.0)):
    return {
        "states": states,
        "meets_req18": req18,
        "april_2020_passed": april,
        "ljung_box_q": dict(zip(("F", "S", "T"), q, strict=True)),
        "seconds": seconds,
    }


DUMMY = candidate(247, 3.0)


def test_elpd_difference_sums_the_kept_blocks_with_az_compare_se():
    alternative = np.array([1.0, 2.0, 4.0, -1.0])
    reference = np.array([0.0, 1.0, 1.0, 0.0])
    keep = np.array([True, True, True, False])
    result = elpd_difference(alternative, reference, keep)
    # Kept differences 1, 1, 3: sum 5, SD (ddof 1) sqrt(4/3), se sqrt(3) * sqrt(4/3) = 2.
    assert result == {"diff": 5.0, "se": pytest.approx(2.0, rel=1e-15), "num_blocks": 3}


def test_ljung_box_rejections_count_columns_below_one_percent():
    tests = {"F": {"p_value": 0.001}, "S": {"p_value": 0.01}, "T": {"p_value": 0.2}}
    assert ljung_box_rejections(tests) == 1


def test_clause_a_takes_the_largest_difference_beyond_two_se():
    candidates = {
        "dummy": DUMMY,
        "trig": candidate(247, 3.0),
        "trig4": candidate(214, 2.0),
    }
    differences = {
        "trig": {"diff": 30.0, "se": 10.0},
        "trig4": {"diff": 25.0, "se": 5.0},
    }
    assert verdict(candidates, differences) == {"layout": "trig", "clause": "a"}


@pytest.mark.parametrize("failing", [{"req18": False}, {"april": False}])
def test_clause_a_needs_req18_and_the_april_check(failing):
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0, **failing)}
    assert verdict(candidates, {"trig": {"diff": 30.0, "se": 10.0}})["clause"] == "c"


def test_a_difference_of_exactly_two_se_does_not_trigger_clause_a():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    assert verdict(candidates, {"trig": {"diff": 20.0, "se": 10.0}})["clause"] == "c"


@pytest.mark.parametrize(
    ("trig4", "difference", "clause"),
    [
        (candidate(214, 2.25), {"diff": -15.0, "se": 10.0}, "b"),  # saving 0.25
        (candidate(214, 2.26), {"diff": -15.0, "se": 10.0}, "c"),  # saving below 0.25
        (candidate(214, 2.0), {"diff": -25.0, "se": 10.0}, "c"),  # beyond two SE
        (candidate(214, 2.0, req18=False), {"diff": 0.0, "se": 10.0}, "c"),
        (candidate(214, 2.0, april=False), {"diff": 0.0, "se": 10.0}, "c"),
        # Q at the ratio in every column passes; one column above it fails.
        (candidate(214, 2.0, q=(106.25, 62.5, 77.5)), {"diff": 0.0, "se": 10.0}, "b"),
        (candidate(214, 2.0, q=(85.0, 62.6, 62.0)), {"diff": 0.0, "se": 10.0}, "c"),
        (
            candidate(214, 2.0, q=(2286.0, 1514.0, 1507.0)),
            {"diff": 0.0, "se": 10.0},
            "c",
        ),
        (candidate(214, None), {"diff": 0.0, "se": 10.0}, "c"),  # cost unmeasured
        (candidate(247, 1.0), {"diff": 0.0, "se": 10.0}, "c"),  # no fewer states
    ],
)
def test_clause_b_needs_every_condition(trig4, difference, clause):
    assert 1 - 2.25 / DUMMY["seconds"] == pytest.approx(COST_SAVING_MIN)
    assert LJUNG_BOX_Q_RATIO_MAX * 85.0 == 106.25
    candidates = {"dummy": DUMMY, "trig4": trig4}
    result = verdict(candidates, {"trig4": difference})
    assert result["clause"] == clause
    assert result["layout"] == ("trig4" if clause == "b" else "dummy")


def test_choose_layout_drops_blocks_unreliable_in_any_layout():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    pointwise = {
        "dummy": np.zeros(4),
        "trig": np.array([3.0, 3.0, 3.0, 4.0]),
    }
    pareto_k = {
        "dummy": np.array([0.1, 0.2, 0.3, 1.9]),
        "trig": np.array([0.1, 0.2, PARETO_K_MAX, 0.4]),
    }
    result = choose_layout(candidates, pointwise, pareto_k)
    assert result["unreliable_blocks"] == 1
    assert result["all_blocks"]["differences"]["trig"]["num_blocks"] == 4
    assert result["reliable_blocks"]["differences"]["trig"]["num_blocks"] == 3
    # All four blocks: diff 13, se sqrt(4) * 0.5 = 1, so clause a picks trig. The three
    # reliable blocks differ by exactly 3 each: diff 9, se 0, so clause a again.
    assert result["all_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable"] is True
    assert result["layout"] == "trig"


def test_choose_layout_keeps_the_dummy_when_the_verdicts_disagree():
    candidates = {"dummy": DUMMY, "trig": candidate(247, 3.0)}
    pointwise = {
        "dummy": np.zeros(4),
        "trig": np.array([3.0, 3.0, 3.0, -40.0]),
    }
    pareto_k = {"dummy": np.array([0.1, 0.1, 0.1, 1.2]), "trig": np.full(4, 0.1)}
    result = choose_layout(candidates, pointwise, pareto_k)
    # The unreliable block's -40 sinks trig on all blocks; without it, clause a holds.
    assert result["all_blocks"]["verdict"]["layout"] == "dummy"
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig", "clause": "a"}
    assert result["reliable"] is False
    assert result["layout"] == "dummy"


def test_choose_layout_agrees_on_the_layout_whatever_the_clause():
    candidates = {"dummy": DUMMY, "trig4": candidate(214, 2.0)}
    pointwise = {"dummy": np.zeros(4), "trig4": np.array([3.0, 3.0, 3.1, -10.0])}
    pareto_k = {"dummy": np.array([0.1, 0.1, 0.1, 1.2]), "trig4": np.full(4, 0.1)}
    result = choose_layout(candidates, pointwise, pareto_k)
    # All blocks: diff -0.9 within 2 * 13.0, so clause b on cost. The three reliable
    # blocks: diff 9.1 beyond 2 * 0.1, so clause a. Both choose trig4.
    assert result["all_blocks"]["verdict"] == {"layout": "trig4", "clause": "b"}
    assert result["reliable_blocks"]["verdict"] == {"layout": "trig4", "clause": "a"}
    assert result["reliable"] is True
    assert result["layout"] == "trig4"


def grid_entry(layout, stage, wedge, steps, batches):
    return {**stage_dimensions(layout, stage, wedge, steps), "batches": batches}


def test_grid_seconds_reads_the_stage_9_low_full_span_batch_1_median():
    grid = {
        "entries": [
            grid_entry("dummy", 9, "low", 280, [{"batch": 1, "median_seconds": 1.5}]),
            grid_entry("dummy", 9, "high", 572, [{"batch": 1, "median_seconds": 5.6}]),
            grid_entry(
                "dummy",
                9,
                "low",
                572,
                [
                    {"batch": 4, "median_seconds": 10.7},
                    {"batch": 1, "median_seconds": 2.99},
                ],
            ),
            grid_entry("trig4", 9, "low", 572, [{"batch": 1, "error": "skipped"}]),
        ]
    }
    assert grid_seconds(grid, "dummy") == 2.99
    assert grid_seconds(grid, "trig") == 2.99  # trig has the dummy's dimensions
    assert grid_seconds(grid, "trig4") is None  # skipped cells carry no median
