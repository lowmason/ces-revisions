"""The GPU projection and per-stage choice rule, on hand-made probe-grid records."""

import math

import pytest

from ces_revisions.pilot.dimensions import STAGE_CONFIGURATIONS, stage_dimensions
from ces_revisions.pilot.gpu_choice import (
    BATCH_LOCKSTEP_FACTORS,
    batch_timings,
    batched_sbc,
    choose_gpu,
    choose_stage,
    configurations,
)

FIT_STEPS = 2_000 * 127  # iterations times leapfrog steps per iteration
SBC_STEPS = 1_490 * 127
FIT_TARGET = {"stage": 9, "wedge": "high", "steps": 572}


def entry(stage, wedge, steps, batches, layout_name="dummy"):
    """A grid entry with the layout's dimensions; None marks a batch that did not fit."""
    tried = [
        {"batch": b, "median_seconds": s}
        if s is not None
        else {"batch": b, "error": "x"}
        for b, s in batches.items()
    ]
    return {**stage_dimensions(layout_name, stage, wedge, steps), "batches": tried}


def record(label, price, fit, sbc=None, four_gpu_price=None, layout_name="dummy"):
    """A record with Stage 9 high's fit entry, optionally its SBC entry, and one entry of
    another stage, which the Stage 9 high rule must ignore."""
    entries = [entry(9, "high", 572, fit, layout_name)]
    if sbc is not None:
        entries.append(entry(9, "high", 280, sbc, layout_name))
    entries.append(entry(9, "low", 572, {1: 99.0}, layout_name))
    return {
        "label": label,
        "hourly_price_usd": price,
        "four_gpu_hourly_price_usd": four_gpu_price,
        "entries": entries,
    }


A100_SBC = {1: 0.05, 4: 0.15, 16: 0.4}
A10 = record("a10", 1.29, {1: 0.5, 4: None}, {1: 0.08, 4: 0.2})
A100 = record("a100", 1.99, {1: 0.2, 4: 0.3}, A100_SBC, four_gpu_price=7.96)
MAC = record("mac", None, {1: 5.6, 4: 20.0}, {1: 2.0})

# Relative cost of each Stage 7-9 model, for records that cover the whole grid.
SCALE = {(7, "low"): 1.0, (8, "low"): 1.2, (8, "high"): 2.5, (9, "low"): 1.5}
SCALE[(9, "high")] = 3.0


def full_record(label, price, fit4, fit1, sbc1, *, fits_from=None, skip=()):
    """A record over every Stage 7-9 model: batch 4 takes ``fit4`` times the model's
    scale, and does not fit from scale ``fits_from`` up; ``skip`` drops models."""
    entries = []
    for stage, wedge in STAGE_CONFIGURATIONS:
        if (stage, wedge) in skip:
            continue
        scale = SCALE[(stage, wedge)]
        batch4 = None if fits_from is not None and scale >= fits_from else fit4 * scale
        entries.append(entry(stage, wedge, 572, {1: fit1 * scale, 4: batch4}))
        entries.append(entry(stage, wedge, 280, {1: sbc1 * scale}))
    return {
        "label": label,
        "hourly_price_usd": price,
        "four_gpu_hourly_price_usd": None,
        "entries": entries,
    }


A10_FULL = full_record("a10", 1.29, 0.2, 0.15, 0.03, fits_from=2.5)
GH200_FULL = full_record("gh200", 2.29, 0.15, 0.12, 0.02)


def test_batch_timings_read_every_tried_batch_at_the_layouts_dimensions():
    assert batch_timings(A10, "dummy", FIT_TARGET) == {1: 0.5, 4: None}
    # trig has the dummy's dimensions, so it reads the dummy's entries; trig4 does not.
    assert batch_timings(A10, "trig", FIT_TARGET) == {1: 0.5, 4: None}
    assert batch_timings(A10, "trig4", FIT_TARGET) is None
    trig4 = record("a10", 1.29, {1: 0.4}, layout_name="trig4")
    assert batch_timings(trig4, "trig4", FIT_TARGET) == {1: 0.4}
    assert batch_timings(trig4, "dummy", FIT_TARGET) is None


def test_configurations_price_one_gpu_and_a_rented_four_gpu_instance():
    vectorized, parallel = configurations(A100, "dummy", 9, "high")
    hours = FIT_STEPS * 1.44 * 0.3 / 3_600
    assert vectorized == {
        "stage": 9,
        "wedge": "high",
        "host": "a100",
        "gpus": 1,
        "chain_method": "vectorized",
        "batch": 4,
        "hourly_price_usd": 1.99,
        "seconds": 0.3,
        "fit_hours": pytest.approx(hours, rel=1e-12),
        "fit_usd": pytest.approx(hours * 1.99, rel=1e-12),
        "sbc_seconds": 0.05,
        # The shipped runner: one replicate at a time on the one GPU, 100 rounds.
        "sbc_days": pytest.approx(100 * SBC_STEPS * 0.05 / 86_400, rel=1e-12),
        "sbc_batched": batched_sbc(A100_SBC, gpus=1),
        "measured": True,
        "feasible": True,
    }
    hours = FIT_STEPS * 0.2 / 3_600
    assert (parallel["gpus"], parallel["chain_method"], parallel["batch"]) == (
        4,
        "parallel",
        1,
    )
    assert parallel["fit_hours"] == pytest.approx(hours, rel=1e-12)
    # The 4-GPU instance's own hourly price, not four times the 1-GPU price.
    assert parallel["fit_usd"] == pytest.approx(hours * 7.96, rel=1e-12)
    # One replicate per GPU: ceil(100 / 4) = 25 rounds.
    assert parallel["sbc_days"] == pytest.approx(25 * SBC_STEPS * 0.05 / 86_400)


def test_a_gpu_type_without_a_four_gpu_instance_has_no_parallel_row():
    rows = configurations(A10, "dummy", 9, "high")
    assert [row["chain_method"] for row in rows] == ["vectorized"]


def test_a_batch_that_did_not_fit_is_measured_and_infeasible():
    (vectorized,) = configurations(A10, "dummy", 9, "high")
    assert vectorized["seconds"] is None
    assert vectorized["fit_hours"] is None
    assert vectorized["fit_usd"] is None
    assert vectorized["measured"] is True
    assert vectorized["feasible"] is False


def test_batched_sbc_takes_the_fastest_batch_with_whole_rounds():
    result = batched_sbc({**A100_SBC, 64: None}, gpus=1)
    rounds = {1: 100, 4: 25, 16: math.ceil(100 / 16)}
    days = {
        batch: rounds[batch]
        * SBC_STEPS
        * BATCH_LOCKSTEP_FACTORS[batch]
        * seconds
        / 86_400
        for batch, seconds in A100_SBC.items()
    }
    best = min(days, key=days.get)
    assert result == {"replicates_per_gpu": best, "days": pytest.approx(days[best])}
    assert batched_sbc(None, gpus=1) is None
    assert batched_sbc({1: None}, gpus=1) is None


def test_the_cheapest_feasible_configuration_wins_and_unpriced_hosts_are_ignored():
    result = choose_stage([MAC, A10, A100], "dummy", 9, "high")
    assert (result["stage"], result["wedge"]) == (9, "high")
    assert {row["host"] for row in result["configurations"]} == {"a10", "a100"}
    assert result["status"] == "chosen"
    choice = result["choice"]
    assert (choice["host"], choice["gpus"], choice["chain_method"]) == (
        "a100",
        1,
        "vectorized",
    )
    assert set(result["band_fit_hours"]) == {"a10", "a100"}
    assert result["band_fit_hours"]["a100"]["31"][0] == pytest.approx(
        2_000 * 31 * 1.44 * 0.3 / 3_600, rel=1e-12
    )


def test_each_stage_gets_the_cheapest_gpu_that_runs_it():
    result = choose_gpu([MAC, A10_FULL, GH200_FULL], "dummy")
    assert result["status"] == "chosen"
    assert [(r["stage"], r["wedge"]) for r in result["stages"]] == list(
        STAGE_CONFIGURATIONS
    )
    chosen = {(r["stage"], r["wedge"]): r["choice"]["host"] for r in result["stages"]}
    # The A10 is cheaper wherever batch 4 fits its memory; the larger models need the
    # GH200.
    assert chosen == {
        (7, "low"): "a10",
        (8, "low"): "a10",
        (8, "high"): "gh200",
        (9, "low"): "a10",
        (9, "high"): "gh200",
    }


def test_trig_reads_the_dummys_entries_and_gets_the_same_choices():
    records = [A10_FULL, GH200_FULL]
    assert choose_gpu(records, "trig") == choose_gpu(records, "dummy")
    assert choose_gpu(records, "trig4")["status"] == "unmeasured"


def test_a_configuration_whose_sbc_is_too_slow_is_not_chosen():
    # x is cheaper per fit, but one replicate at a time takes 17.5 days of SBC.
    cheap = record("x", 1.0, {1: 1.0, 4: 0.3}, {1: 0.08})
    dear = record("y", 2.0, {1: 1.0, 4: 0.3}, {1: 0.05})
    (row,) = configurations(cheap, "dummy", 9, "high")
    assert row["fit_hours"] <= 48
    assert row["sbc_days"] == pytest.approx(17.5, abs=0.1)
    assert row["feasible"] is False
    result = choose_stage([cheap, dear], "dummy", 9, "high")
    assert (result["status"], result["choice"]["host"]) == ("chosen", "y")


def test_one_infeasible_stage_makes_the_overall_status_infeasible():
    result = choose_gpu([A10_FULL], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert statuses == {
        (7, "low"): "chosen",
        (8, "low"): "chosen",
        (8, "high"): "infeasible",
        (9, "low"): "chosen",
        (9, "high"): "infeasible",
    }
    assert result["status"] == "infeasible"


@pytest.mark.parametrize(
    "records",
    [
        [],  # nothing measured
        [MAC],  # no priced host
        [record("a10", 1.29, {1: 0.2, 4: 0.3})],  # no SBC entry
        [record("a10", 1.29, {1: 0.2}, {1: 0.05})],  # batch 4 was never tried
        [A100, record("a10", 1.29, {1: 0.2, 4: 0.3})],  # one host incomplete
    ],
)
def test_an_incomplete_measurement_is_unmeasured_not_infeasible(records):
    result = choose_stage(records, "dummy", 9, "high")
    assert (result["status"], result["choice"]) == ("unmeasured", None)


def test_a_stage_missing_from_one_record_makes_the_overall_status_unmeasured():
    partial = full_record("gh200", 2.29, 0.15, 0.12, 0.02, skip={(8, "high")})
    result = choose_gpu([A10_FULL, partial], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert statuses[(8, "high")] == "unmeasured"
    assert result["status"] == "unmeasured"


def test_an_unmeasured_stage_outranks_an_infeasible_one():
    # Stage 7 is missing from the GH200 record, and Stage 9 high fits neither GPU.
    partial = full_record(
        "gh200", 2.29, 0.15, 0.12, 0.02, fits_from=3.0, skip={(7, "low")}
    )
    result = choose_gpu([A10_FULL, partial], "dummy")
    statuses = {(r["stage"], r["wedge"]): r["status"] for r in result["stages"]}
    assert (statuses[(7, "low")], statuses[(9, "high")]) == ("unmeasured", "infeasible")
    assert result["status"] == "unmeasured"


def test_measured_steps_per_iteration_replace_the_sizing_assumption():
    result = choose_gpu([A10_FULL, GH200_FULL], "dummy", steps_per_iteration=31)
    assert result["steps_per_iteration"] == 31
    stage_7 = result["stages"][0]["configurations"][0]
    assert stage_7["fit_hours"] == pytest.approx(2_000 * 31 * 1.44 * 0.2 / 3_600)
    assert stage_7["sbc_days"] == pytest.approx(100 * 1_490 * 31 * 0.03 / 86_400)
    assert choose_gpu([A10_FULL], "dummy")["steps_per_iteration"] == 127


def test_a_tie_in_dollars_goes_to_fewer_gpus():
    # Vectorized: 1.44 * 0.4 s at $2.00; parallel: 0.144 s at $8.00. Equal per fit.
    tied = record("a100", 2.0, {1: 0.144, 4: 0.4}, {1: 0.05}, four_gpu_price=8.0)
    vectorized, parallel = configurations(tied, "dummy", 9, "high")
    assert vectorized["fit_usd"] == pytest.approx(parallel["fit_usd"], rel=1e-12)
    assert choose_stage([tied], "dummy", 9, "high")["choice"]["gpus"] == 1
