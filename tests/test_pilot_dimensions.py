"""State and cell counts of the pilot and of the Stage 7-9 models, and the grid probe."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import gpu_probe_grid
import jax
import pytest
from gpu_probe_grid import DEFAULT_STEPS, measure_entry, measurement_entries

from ces_revisions.pilot.dimensions import (
    MEASUREMENT_GRID,
    PILOT_CELLS,
    SEASONAL_DIMS,
    matches_layout,
    pilot_dimensions,
    stage_dimensions,
)

# (layout, stage, wedge) -> (states, cells), counted by hand from the term list.
EXPECTED = {
    ("dummy", 7, "low"): (190, 37),
    ("dummy", 8, "low"): (203, 85),
    ("dummy", 8, "high"): (324, 217),
    ("dummy", 9, "low"): (247, 145),
    ("dummy", 9, "high"): (368, 277),
    ("trig", 7, "low"): (190, 37),
    ("trig4", 7, "low"): (157, 37),
    ("trig4", 8, "low"): (170, 85),
    ("trig4", 8, "high"): (291, 217),
    ("trig4", 9, "low"): (214, 145),
    ("trig4", 9, "high"): (335, 277),
}


def test_pilot_dimensions_count_trend_seasonal_and_news_states():
    assert pilot_dimensions("dummy") == {
        "steps": 572,
        "states": 17,
        "cells": PILOT_CELLS,
        "stage": 6,
        "layout": "dummy",
        "wedge": None,
    }
    assert pilot_dimensions("trig")["states"] == 17
    assert pilot_dimensions("trig4", num_steps=280)["states"] == 14
    assert pilot_dimensions("trig4", num_steps=280)["steps"] == 280
    assert SEASONAL_DIMS == {"dummy": 11, "trig": 11, "trig4": 8}


@pytest.mark.parametrize(("key", "expected"), EXPECTED.items(), ids=str)
def test_stage_dimensions_match_the_hand_count(key, expected):
    layout_name, stage, wedge = key
    dims = stage_dimensions(layout_name, stage, wedge, num_steps=280)

    assert (dims["states"], dims["cells"]) == expected
    assert dims == {
        "steps": 280,
        "states": expected[0],
        "cells": expected[1],
        "stage": stage,
        "layout": layout_name,
        "wedge": wedge,
    }


def test_stage_dimensions_default_to_the_full_span_and_the_low_wedge():
    assert stage_dimensions("dummy", 9) == stage_dimensions("dummy", 9, "low", 572)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("weekly", 7), "layout"),
        (("dummy", 6), "stage"),
        (("dummy", 10), "stage"),
        (("dummy", 9, "medium"), "wedge"),
        (("dummy", 7, "high"), "Stage 7"),
    ],
)
def test_stage_dimensions_reject_unknown_configurations(arguments, message):
    with pytest.raises(ValueError, match=message):
        stage_dimensions(*arguments)


def test_pilot_dimensions_reject_an_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        pilot_dimensions("weekly")


def test_measurement_grid_lists_both_layouts_every_stage_7_to_9_model_and_both_spans():
    configurations = [
        (entry["layout"], entry["stage"], entry["wedge"], entry["steps"])
        for entry in MEASUREMENT_GRID
    ]

    assert configurations == [
        (layout_name, stage, wedge, steps)
        for layout_name in ("dummy", "trig4")
        for stage, wedge in (
            (7, "low"),
            (8, "low"),
            (8, "high"),
            (9, "low"),
            (9, "high"),
        )
        for steps in (572, 280)
    ]
    assert len(MEASUREMENT_GRID) == 20
    for entry in MEASUREMENT_GRID:
        expected = EXPECTED[(entry["layout"], entry["stage"], entry["wedge"])]
        assert (entry["states"], entry["cells"]) == expected
    assert max(entry["states"] for entry in MEASUREMENT_GRID) == 368
    assert max(entry["cells"] for entry in MEASUREMENT_GRID) == 277


def test_grid_entries_default_to_the_measurement_grid():
    assert measurement_entries() == list(MEASUREMENT_GRID)
    assert measurement_entries(list(DEFAULT_STEPS)) == list(MEASUREMENT_GRID)


def test_grid_entries_take_other_steps_and_filter_by_configuration():
    filtered = measurement_entries([6], ["layout=trig4", "stage=7"])
    either_stage = measurement_entries([572], ["stage=7", "stage=9", "wedge=high"])

    assert filtered == [stage_dimensions("trig4", 7, "low", 6)]
    assert either_stage == [
        stage_dimensions(layout_name, 9, "high", 572)
        for layout_name in ("dummy", "trig4")
    ]


def test_grid_entries_reject_a_bad_filter_and_steps_beside_a_grid():
    grid = [{"steps": 6, "states": 3, "cells": 4}]

    assert measurement_entries(grid=grid) == [
        {
            "layout": None,
            "stage": None,
            "wedge": None,
            "steps": 6,
            "states": 3,
            "cells": 4,
        }
    ]
    with pytest.raises(ValueError, match="KEY=VALUE"):
        measurement_entries(filters=["sector=00"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        measurement_entries(filters=["layout"])
    with pytest.raises(ValueError, match="not both"):
        measurement_entries([6], grid=grid)
    with pytest.raises(ValueError, match="states"):
        measurement_entries(grid=[{"steps": 6, "states": 0, "cells": 4}])


def test_grid_probe_writes_one_record_with_timings_and_memory(tmp_path):
    grid = [
        {
            "layout": "tiny",
            "stage": 7,
            "wedge": "low",
            "steps": 6,
            "states": 3,
            "cells": 4,
        },
        {
            "layout": "tiny",
            "stage": 9,
            "wedge": "high",
            "steps": 5,
            "states": 4,
            "cells": 2,
        },
    ]
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid))
    out = tmp_path / "grid-probe.json"
    command = [sys.executable, str(Path("scripts") / "gpu_probe_grid.py")]
    command += ["--label", "test", "--out", str(out), "--grid-json", str(grid_path)]
    command += ["--batch", "1", "2", "--repeats", "2", "--grid-filter", "layout=tiny"]
    command += ["--hourly-price", "1.29", "--four-gpu-price", "5.16"]
    subprocess.run(command, check=True, cwd=Path(__file__).parents[1])
    record = json.loads(out.read_text())

    assert set(record) == {
        "label",
        "hourly_price_usd",
        "four_gpu_hourly_price_usd",
        "host",
        "versions",
        "entries",
        "timestamp_utc",
    }
    assert record["label"] == "test"
    assert record["hourly_price_usd"] == 1.29
    assert record["four_gpu_hourly_price_usd"] == 5.16
    assert record["host"]["backend"] == jax.default_backend()
    assert record["host"]["device_kind"] == jax.devices()[0].device_kind
    assert (record["host"]["nvidia_driver"] is None) == (
        shutil.which("nvidia-smi") is None
    )
    for name in ("jax", "jaxlib"):
        assert record["versions"][name] == importlib.metadata.version(name)
    assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset() == timedelta(0)
    assert [
        {
            key: entry[key]
            for key in ("layout", "stage", "wedge", "steps", "states", "cells")
        }
        for entry in record["entries"]
    ] == grid
    for entry in record["entries"]:
        assert [batch["batch"] for batch in entry["batches"]] == [1, 2]
        assert entry["repeats"] == 2
        assert (entry["peak_bytes_in_use"] is None) == (
            jax.devices()[0].memory_stats() is None
        )
        for batch in entry["batches"]:
            assert batch["compile_seconds"] > 0
            assert batch["median_seconds"] > 0
            assert batch["iqr_seconds"] >= 0
            assert batch["memory_analysis"]["total_bytes"] > 0
        smaller, larger = (batch["memory_analysis"] for batch in entry["batches"])
        assert larger["temp_size_in_bytes"] > smaller["temp_size_in_bytes"]


def test_grid_probe_refuses_a_filter_that_matches_nothing(tmp_path):
    command = [sys.executable, str(Path("scripts") / "gpu_probe_grid.py")]
    command += ["--label", "test", "--out", str(tmp_path / "none.json")]
    command += ["--grid-filter", "layout=weekly"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode == 2
    assert "no grid entry matches" in result.stderr
    assert not (tmp_path / "none.json").exists()


def test_grid_probe_skips_batches_beyond_the_memory_budget():
    entry = {"layout": "tiny", "stage": 7, "wedge": "low"}
    entry |= {"steps": 6, "states": 3, "cells": 4}
    unbounded = measure_entry(entry, [1, 2, 8], 1, 0.25, seed=0)
    totals = [batch["memory_analysis"]["total_bytes"] for batch in unbounded["batches"]]
    bounded = measure_entry(entry, [1, 2, 8], 1, 0.25, seed=0, max_bytes=totals[0])

    assert totals == sorted(totals)
    assert totals[0] < totals[1]
    assert all("error" not in batch for batch in unbounded["batches"])
    assert [batch.get("error") for batch in bounded["batches"]] == [
        None,
        f"skipped: compiled memory exceeds {totals[0]} bytes",
        "skipped: batch 2 did not fit in memory",
    ]
    assert bounded["batches"][0]["median_seconds"] > 0
    assert "median_seconds" not in bounded["batches"][1]
    assert bounded["batches"][1]["memory_analysis"]["total_bytes"] == totals[1]
    # Batch 8 follows a batch that did not fit, so it is skipped without compiling.
    assert bounded["batches"][2]["memory_analysis"] is None
    assert bounded["max_bytes"] == totals[0]


def test_trig_entries_are_the_dummys_and_trig4_entries_are_its_own():
    for entry in MEASUREMENT_GRID:
        own = entry["layout"]
        assert matches_layout(entry, own)
        assert matches_layout(entry, "trig") == (own == "dummy")
        assert matches_layout(entry, "dummy") == (own == "dummy")
        assert matches_layout(entry, "trig4") == (own == "trig4")


def test_an_entry_outside_stages_7_to_9_matches_no_layout():
    custom = {"layout": "tiny", "stage": None, "wedge": None}
    assert not matches_layout(custom | {"steps": 6, "states": 3, "cells": 4}, "dummy")
    stage_7_high = stage_dimensions("dummy", 7) | {"wedge": "high"}
    assert not matches_layout(stage_7_high, "dummy")


@pytest.mark.parametrize(
    "prices",
    [["--four-gpu-price", "1.99"], ["--four-gpu-price", "7.96"]],
    ids=["per-gpu-price", "no-hourly-price"],
)
def test_grid_probe_refuses_a_per_gpu_four_gpu_price(tmp_path, capsys, prices):
    out = tmp_path / "a100.json"
    command = ["--label", "a100", "--out", str(out), *prices]
    if prices[1] == "1.99":
        command += ["--hourly-price", "1.99"]
    with pytest.raises(SystemExit) as raised:
        gpu_probe_grid.main(command)
    assert raised.value.code == 2
    assert "whole 4-GPU instance" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="the GPU backend is present")
def test_grid_probe_refuses_a_cpu_backend_beside_an_nvidia_device(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(gpu_probe_grid, "has_nvidia_device", lambda: True)
    monkeypatch.setenv("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
    out = tmp_path / "cpu.json"
    with pytest.raises(SystemExit) as raised:
        gpu_probe_grid.main(["--label", "a10", "--out", str(out)])
    assert raised.value.code == 2
    assert "JAX's backend is 'cpu'" in capsys.readouterr().err
    assert not out.exists()
