"""The seasonal-state decision record states exactly what its committed evidence says."""

import json

from ces_revisions.pilot.decision_tables import (
    decision_lines,
    seasonal_rows,
    timing_rows,
)
from ces_revisions.vintages import raw as vintage_raw

DECISION = vintage_raw.ROOT / "docs" / "decisions" / "seasonal-state.md"
EVIDENCE = vintage_raw.ROOT / "docs" / "runs" / "stage-6"

COMPARISON = {
    "runs": [{"layout": "dummy"}, {"layout": "trig4"}],
    "candidates": {
        "dummy": {
            "states": 247,
            "meets_req18": True,
            "april_2020_passed": True,
            "ljung_box_q": {"F": 85.2, "S": 49.6, "T": 61.7},
            "seconds": 2.989,
        },
        "trig4": {
            "states": 214,
            "meets_req18": True,
            "april_2020_passed": False,
            "ljung_box_q": {"F": 2286.4, "S": 1514.0, "T": 1507.3},
            "seconds": None,
        },
    },
    "rule": {
        "layout": "dummy",
        "all_blocks": {"differences": {"trig4": {"diff": -466.36, "se": 70.4}}},
        "reliable_blocks": {"differences": {"trig4": {"diff": -520.68, "se": 26.64}}},
    },
}


def test_decision_lines_name_the_layout():
    assert decision_lines(COMPARISON) == [
        "The seasonal state for O6 and every later fit of this model is `dummy`."
    ]


def test_seasonal_rows_show_both_block_sets_the_checks_and_the_cost():
    assert seasonal_rows(COMPARISON) == [
        "| `dummy` | 247 | reference | reference | yes | passed | 85, 50, 62 | 2.989 |",
        (
            "| `trig4` | 214 | -466.4 ± 70.4 | -520.7 ± 26.6 | yes | failed "
            "| 2286, 1514, 1507 | — |"
        ),
    ]


def test_timing_rows_cover_every_batch_at_the_layouts_dimensions():
    memory = {"memory_analysis": {"total_bytes": 37_401_276_160}}
    record = {
        "label": "mac",
        "entries": [
            {
                "layout": "dummy",
                "stage": 9,
                "wedge": "high",
                "steps": 572,
                "states": 368,
                "cells": 277,
                "batches": [
                    {"batch": 1, "median_seconds": 5.6374, **memory},
                    {"batch": 4, "error": "skipped", **memory},
                    {"batch": 16, "error": "skipped", "memory_analysis": None},
                ],
            },
            {
                "layout": "trig4",
                "stage": 9,
                "wedge": "high",
                "steps": 572,
                "states": 335,
                "cells": 277,
                "batches": [{"batch": 1, "median_seconds": 4.9, **memory}],
            },
        ],
    }
    rows = [
        "| mac | 9 | high | 572 | 368 | 277 | 1 | 5.637 | 37.4 |",
        "| mac | 9 | high | 572 | 368 | 277 | 4 | skipped | 37.4 |",
        "| mac | 9 | high | 572 | 368 | 277 | 16 | skipped | — |",
    ]
    assert timing_rows([record], "dummy") == rows
    assert timing_rows([record], "trig") == rows  # trig has the dummy's dimensions


def test_the_decision_record_states_its_evidence():
    text = DECISION.read_text()
    comparison = json.loads((EVIDENCE / "seasonal-comparison.json").read_text())
    layout = comparison["rule"]["layout"]
    mac = json.loads((vintage_raw.ROOT / comparison["grid"]["path"]).read_text())
    assert "**Status:** Accepted" in text
    timings = timing_rows([mac], layout)
    # An empty table would pass the check below vacuously.
    assert timings
    expected = [*decision_lines(comparison), *seasonal_rows(comparison), *timings]
    missing = [line for line in expected if line not in text]
    assert missing == []


def test_decision_lines_name_the_layout_the_rule_chose():
    comparison = COMPARISON | {"rule": COMPARISON["rule"] | {"layout": "trig"}}
    assert decision_lines(comparison) == [
        "The seasonal state for O6 and every later fit of this model is `trig`."
    ]


def test_seasonal_rows_sign_a_gain_and_mark_a_layout_that_fails_req18():
    candidate = COMPARISON["candidates"]["trig4"] | {"meets_req18": False}
    comparison = {
        "runs": [{"layout": "trig4"}],
        "candidates": {"trig4": candidate},
        "rule": {
            "all_blocks": {"differences": {"trig4": {"diff": 12.83, "se": 8.648}}},
            "reliable_blocks": {"differences": {"trig4": {"diff": 10.5, "se": 8.49}}},
        },
    }
    assert seasonal_rows(comparison) == [
        (
            "| `trig4` | 214 | +12.8 ± 8.6 | +10.5 ± 8.5 | no | failed "
            "| 2286, 1514, 1507 | — |"
        )
    ]


def test_timing_rows_cover_every_record_in_order():
    def record(label, seconds):
        batch = {
            "batch": 1,
            "median_seconds": seconds,
            "memory_analysis": {"total_bytes": 1_000_000_000},
        }
        entry = {
            "stage": 7,
            "wedge": "low",
            "steps": 280,
            "states": 190,
            "cells": 37,
            "batches": [batch],
        }
        return {"label": label, "entries": [entry]}

    assert timing_rows([record("mac", 0.642), record("a10", 0.101)], "dummy") == [
        "| mac | 7 | low | 280 | 190 | 37 | 1 | 0.642 | 1.0 |",
        "| a10 | 7 | low | 280 | 190 | 37 | 1 | 0.101 | 1.0 |",
    ]
