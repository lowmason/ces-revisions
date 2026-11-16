"""Project Stage 7-9 fit and SBC cost from probe-grid records; choose a GPU for each stage.

Each rented host's record (``scripts/gpu_probe_grid.py --hourly-price``) times one value
and gradient of the engine at the Stage 7-9 dimensions, for batches of chains or SBC
replicates run at once. A grid entry belongs to a layout when its states and cells are
that layout's at the entry's stage, wedge, and steps, so ``trig``, whose dimensions equal
the dummy's, reads the dummy's entries.

The rule chooses a GPU type and count for each Stage 7-9 model of STAGE_CONFIGURATIONS
(decision D5), from that model's own dimensions: a fit over FIT_STEPS months and an SBC
replicate over SBC_STEPS. Stages 10-17 sample more parameters but keep Stage 9's Kalman
states and cells (Req 17). While they also keep Stage 9's engine and NUTS kernel, they
take Stage 9's choice at the wedge Stage 8 settles, rerun with their own measured steps
per iteration.

A fit runs NUM_CHAINS chains of ITERATIONS NUTS iterations, and every leapfrog step costs
one value and gradient. On one GPU the chains run as one batch (NumPyro's "vectorized"
chain method), so each iteration waits for the chain with the deepest tree. With one GPU
per chain they run side by side ("parallel") at batch 1; that configuration exists only
where Lambda rents the GPU type as a NUM_CHAINS-GPU instance, whose hourly price the
record carries (``--four-gpu-price``).

SBC runs SBC_REPLICATES single-chain replicates over SBC_STEPS, the reduced panel's span.
The projection assumes one replicate per GPU at a time, at batch 1, in
ceil(SBC_REPLICATES / GPUs) rounds. That is the shipped runner's throughput
(``ces_revisions.pilot.sbc.run_sbc``): it runs one replicate per process and one worker
on a GPU host, so a multi-GPU instance runs one shard per GPU under its own
``CUDA_VISIBLE_DEVICES``. The record also projects a batched runner, which Stage 7 would
have to build, for reference; that projection decides nothing.

The pre-registered rule, for each stage:

1. A configuration is measured when its record holds the stage's fit and SBC entries at
   the layout's dimensions with the batches it needs. A batch that did not fit device
   memory counts as measured and infeasible.
2. A configuration is feasible when its fit takes at most FIT_HOURS_MAX hours and its SBC
   at most SBC_DAYS_MAX days.
3. When every configuration is measured, the cheapest feasible one in dollars per fit, to
   the cent, wins ("chosen"); a tie goes to fewer GPUs. With none feasible the stage's
   status is "infeasible".
4. When no host is priced or any configuration is unmeasured, the status is
   "unmeasured" and nothing is chosen: the measurement is incomplete, not a finding.

The overall status is "unmeasured" if any stage's is, else "infeasible" if any stage's
is, and then the roadmap's resume re-validates Stages 7-17 before it routes Stage 7;
otherwise it is "chosen".
"""

import math

from ces_revisions.pilot.dimensions import (
    FULL_STEPS,
    SECTOR_STEPS,
    STAGE_CONFIGURATIONS,
    matches_layout,
)

NUM_CHAINS = 4
ITERATIONS = 2_000  # 1,000 warmup and 1,000 sampling iterations per chain (Req 18 runs)
# Leapfrog steps per iteration of the slowest chain. The pilot's 17 parameters took about
# 65 over warmup and sampling; Stages 7-9 sample far more parameters, so the rule sizes
# for depth-7 trees, the top of the 31-127 band the projections report.
STEPS_PER_ITERATION = 127
STEP_BAND = (31, 63, 127)
# A vectorized batch takes each iteration's deepest tree across chains; the pilot's
# lockstep total over its slowest chain's was 94.2 / 65.3 steps per iteration.
LOCKSTEP_FACTOR = 1.44
FIT_HOURS_MAX = 48.0
FIT_STEPS = FULL_STEPS  # a fit covers 1979-01 through 2026-08
SBC_STEPS = SECTOR_STEPS  # an SBC replicate covers the sector panel's span
SBC_REPLICATES = 100
SBC_ITERATIONS = 1_490  # 500 warmup and 990 sampling iterations, one chain each
SBC_DAYS_MAX = 14.0
# The batched reference projection's lockstep factor by replicates per batch: over the
# planning spike's 100 SBC replicates, the expected largest of b replicates' mean steps
# per iteration over their average. Trees also vary within a replicate, so these
# understate the lockstep cost.
BATCH_LOCKSTEP_FACTORS = {1: 1.0, 4: 1.331, 16: 1.637, 64: 1.991}
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0


def batch_timings(
    record: dict, layout_name: str, target: dict
) -> dict[int, float | None] | None:
    """Median seconds of every batch tried at a target, by batch size.

    ``target`` gives the entry's stage, wedge, and steps. None when the record holds no
    entry at the target and the layout's dimensions. A batch that did not fit device
    memory maps to None.
    """
    for entry in record["entries"]:
        if matches_layout(entry, layout_name) and all(
            entry[key] == value for key, value in target.items()
        ):
            return {
                batch["batch"]: batch.get("median_seconds")
                for batch in entry["batches"]
            }
    return None


def configurations(
    record: dict,
    layout_name: str,
    stage: int,
    wedge: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> list[dict]:
    """The host's configurations for one stage, each with its fit cost and SBC days."""
    fit = batch_timings(
        record, layout_name, {"stage": stage, "wedge": wedge, "steps": FIT_STEPS}
    )
    sbc = batch_timings(
        record, layout_name, {"stage": stage, "wedge": wedge, "steps": SBC_STEPS}
    )
    shapes = [
        (1, "vectorized", NUM_CHAINS, LOCKSTEP_FACTOR, record["hourly_price_usd"])
    ]
    if record.get("four_gpu_hourly_price_usd") is not None:
        shapes.append(
            (NUM_CHAINS, "parallel", 1, 1.0, record["four_gpu_hourly_price_usd"])
        )
    sbc_seconds = None if sbc is None else sbc.get(1)
    rows = []
    for gpus, chain_method, batch, factor, price in shapes:
        seconds = None if fit is None else fit.get(batch)
        hours = (
            None
            if seconds is None
            else ITERATIONS * steps_per_iteration * factor * seconds / SECONDS_PER_HOUR
        )
        days = (
            None
            if sbc_seconds is None
            else math.ceil(SBC_REPLICATES / gpus)
            * SBC_ITERATIONS
            * steps_per_iteration
            * sbc_seconds
            / SECONDS_PER_DAY
        )
        measured = fit is not None and batch in fit and sbc is not None and 1 in sbc
        rows.append(
            {
                "stage": stage,
                "wedge": wedge,
                "host": record["label"],
                "gpus": gpus,
                "chain_method": chain_method,
                "batch": batch,
                "hourly_price_usd": price,
                "seconds": seconds,
                "fit_hours": hours,
                "fit_usd": None if hours is None else hours * price,
                "sbc_seconds": sbc_seconds,
                "sbc_days": days,
                "sbc_batched": batched_sbc(sbc, gpus, steps_per_iteration),
                "measured": measured,
                "feasible": hours is not None
                and hours <= FIT_HOURS_MAX
                and days is not None
                and days <= SBC_DAYS_MAX,
            }
        )
    return rows


def batched_sbc(
    sbc: dict[int, float | None] | None,
    gpus: int,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict | None:
    """The fastest batched-runner SBC projection over the measured batches, if any."""
    options = [
        {
            "replicates_per_gpu": batch,
            "days": math.ceil(SBC_REPLICATES / (batch * gpus))
            * SBC_ITERATIONS
            * steps_per_iteration
            * BATCH_LOCKSTEP_FACTORS[batch]
            * seconds
            / SECONDS_PER_DAY,
        }
        for batch, seconds in (sbc or {}).items()
        if seconds is not None and batch in BATCH_LOCKSTEP_FACTORS
    ]
    return min(options, key=lambda option: option["days"], default=None)


def choose_stage(
    records: list[dict],
    layout_name: str,
    stage: int,
    wedge: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict:
    """One stage's configurations over every priced host, its band, status, and choice."""
    priced = [record for record in records if record["hourly_price_usd"] is not None]
    rows = [
        row
        for record in priced
        for row in configurations(
            record, layout_name, stage, wedge, steps_per_iteration
        )
    ]
    band = {
        record["label"]: {
            str(steps): [
                row["fit_hours"]
                for row in configurations(record, layout_name, stage, wedge, steps)
            ]
            for steps in STEP_BAND
        }
        for record in priced
    }
    feasible = [row for row in rows if row["feasible"]]
    if not rows or not all(row["measured"] for row in rows):
        status, choice = "unmeasured", None
    elif not feasible:
        status, choice = "infeasible", None
    else:
        status = "chosen"
        choice = min(feasible, key=lambda row: (round(row["fit_usd"], 2), row["gpus"]))
    return {
        "stage": stage,
        "wedge": wedge,
        "configurations": rows,
        "band_fit_hours": band,
        "status": status,
        "choice": choice,
    }


def choose_gpu(
    records: list[dict],
    layout_name: str,
    steps_per_iteration: int = STEPS_PER_ITERATION,
) -> dict:
    """The rule for every Stage 7-9 model, and the overall status.

    ``steps_per_iteration`` replaces the rule's sizing assumption when a later stage has
    measured its own leapfrog steps per iteration.
    """
    stages = [
        choose_stage(records, layout_name, stage, wedge, steps_per_iteration)
        for stage, wedge in STAGE_CONFIGURATIONS
    ]
    statuses = {result["status"] for result in stages}
    if "unmeasured" in statuses:
        status = "unmeasured"
    elif "infeasible" in statuses:
        status = "infeasible"
    else:
        status = "chosen"
    return {
        "steps_per_iteration": steps_per_iteration,
        "stages": stages,
        "status": status,
    }
