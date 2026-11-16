"""Time the Kalman engine at the Stage 7-9 model dimensions on this host.

For each configuration of ``MEASUREMENT_GRID`` (or of a ``--grid-json`` override) the
script builds the engine probe's synthetic model at that configuration's steps, states,
and cells, and times the vmapped value and gradient at each batch size, where the batch
stands for the chains or SBC replicates run at once. Each batch size is timed on its own,
so one that exhausts device memory is recorded, larger ones are skipped without
compiling, and the next configuration still runs. Each compiled batch also records XLA's
compiled memory analysis, which needs no device statistics and so sizes memory on the Mac
too. The JSON record is rewritten after every configuration, so an interrupted run on a
rented GPU keeps its finished entries. On a host with an NVIDIA device the script refuses
to run unless JAX's backend is the GPU, so a driver or plugin fault cannot produce a CPU
record under a GPU label.
"""

import argparse
import json
import os
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
import numpyro

from ces_revisions import engine_probe
from ces_revisions.devices import (
    DETERMINISTIC_GPU_FLAG,
    has_nvidia_device,
    jax_versions,
    nvidia_driver,
)
from ces_revisions.kalman import LinearGaussianSSM
from ces_revisions.pilot.dimensions import MEASUREMENT_GRID, stage_dimensions

DEFAULT_BATCHES = (1, 4, 16, 64)
DEFAULT_REPEATS = 5
DEFAULT_STEPS = (572, 280)
# The engine probe's default share of NaN cells; the filter's cost does not depend on it.
MISSING_SHARE = 0.2
SEED = sum(map(ord, "ces-revisions-stage-6-gpu-probe-grid"))
FILTER_KEYS = ("layout", "stage", "wedge")
DIMENSION_KEYS = ("steps", "states", "cells")


def measurement_entries(
    steps: list[int] | None = None,
    filters: list[str] | None = None,
    grid: list[dict] | None = None,
) -> list[dict]:
    """The configurations to measure, in grid order.

    Without ``grid``, every (layout, stage, wedge) configuration of MEASUREMENT_GRID at
    each of ``steps`` (default 572 and 280, which reproduces the grid). A ``grid`` lists
    complete entries and takes no ``steps``. A filter ``key=value`` keeps entries whose
    key has one of the given values: values of one key are alternatives, keys combine.
    """
    if grid is None:
        configurations = dict.fromkeys(
            (entry["layout"], entry["stage"], entry["wedge"])
            for entry in MEASUREMENT_GRID
        )
        entries = [
            stage_dimensions(layout_name, stage, wedge, num_steps)
            for layout_name, stage, wedge in configurations
            for num_steps in steps or DEFAULT_STEPS
        ]
    elif steps is not None:
        raise ValueError(
            "a grid lists its own steps: give --steps or --grid-json, not both"
        )
    else:
        entries = [_validated_grid_entry(entry) for entry in grid]
    wanted = _parse_filters(filters or [])
    return [
        entry
        for entry in entries
        if all(str(entry.get(key)) in values for key, values in wanted.items())
    ]


def measure_entry(
    entry: dict,
    batch_sizes: list[int],
    repeats: int,
    missing_share: float,
    seed: int,
    max_bytes: int | None = None,
) -> dict:
    """Timings and memory of the engine's value and gradient at one entry's dimensions.

    A batch runs only when its compiled memory fits the budget: the smaller of
    ``max_bytes`` and the device's ``bytes_limit``, whichever exist. A CPU reports no
    limit, so ``max_bytes`` keeps a large batch from swapping the host. Once a batch has
    not fit, larger batches are skipped without compiling, so their memory analysis is
    None.
    """
    ssm, panel = engine_probe.synthetic_problem(
        entry["steps"], entry["states"], entry["cells"], missing_share, seed
    )
    limits = [max_bytes, _device_memory_stats().get("bytes_limit")]
    budget = min((limit for limit in limits if limit is not None), default=None)
    batches, exhausted_at = [], None
    for batch in batch_sizes:
        if exhausted_at is not None and batch >= exhausted_at:
            batches.append(
                {
                    "batch": batch,
                    "memory_analysis": None,
                    "error": f"skipped: batch {exhausted_at} did not fit in memory",
                }
            )
            continue
        memory = _compiled_memory(ssm, panel, batch)
        result = {"batch": batch, "memory_analysis": memory}
        if budget is not None and memory["total_bytes"] > budget:
            result["error"] = f"skipped: compiled memory exceeds {budget} bytes"
            exhausted_at = batch
        else:
            try:
                (timing,) = engine_probe.time_batches(
                    ssm, panel, [batch], repeats, seed
                )
                result.update(timing)
            except jax.errors.JaxRuntimeError as error:
                if "RESOURCE_EXHAUSTED" not in str(error):
                    raise
                result["error"] = f"RESOURCE_EXHAUSTED: {str(error).splitlines()[0]}"
                exhausted_at = batch
        batches.append(result)
    stats = _device_memory_stats()
    return {
        **entry,
        "missing_share": missing_share,
        "seed": seed,
        "repeats": repeats,
        "max_bytes": max_bytes,
        "batches": batches,
        # The device's high-water mark for the whole process so far, read after this
        # entry's largest batch: an earlier, larger entry raises it for later ones.
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "load_average": list(os.getloadavg()),
    }


def host_record() -> dict:
    """The backend, devices, driver, and XLA flags the timings were taken under."""
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "device_count": jax.device_count(),
        "bytes_limit": _device_memory_stats().get("bytes_limit"),
        "nvidia_driver": nvidia_driver(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, measure every entry, and write the JSON record."""
    parser = _parser()
    args = parser.parse_args(argv)
    grid = json.loads(args.grid_json.read_text()) if args.grid_json else None
    try:
        entries = measurement_entries(args.steps, args.grid_filter, grid)
    except ValueError as error:
        parser.error(str(error))
    if not entries:
        parser.error("no grid entry matches the filters")
    if args.four_gpu_price is not None and (
        args.hourly_price is None or args.four_gpu_price < 2 * args.hourly_price
    ):
        parser.error(
            "--four-gpu-price is the whole 4-GPU instance's hourly price, 4 times the "
            "per-GPU price Lambda lists, and needs --hourly-price"
        )
    # The fits run with XLA's deterministic GPU operations, so the timings do too. Like
    # float64, the flag must be set before the first JAX operation.
    flags = os.environ.get("XLA_FLAGS", "").split()
    if has_nvidia_device() and DETERMINISTIC_GPU_FLAG not in flags:
        os.environ["XLA_FLAGS"] = " ".join([*flags, DETERMINISTIC_GPU_FLAG])
    numpyro.enable_x64()
    if has_nvidia_device() and jax.default_backend() != "gpu":
        parser.error(
            f"an NVIDIA device is present but JAX's backend is "
            f"{jax.default_backend()!r}: check nvidia-smi and uv sync --extra cuda"
        )
    started = datetime.now(UTC)
    record = {
        "label": args.label,
        "hourly_price_usd": args.hourly_price,
        "four_gpu_hourly_price_usd": args.four_gpu_price,
        "host": host_record(),
        "versions": jax_versions(),
        "entries": [],
        "timestamp_utc": started.isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(record, args.out)
    for entry in entries:
        measured = measure_entry(
            entry,
            args.batch,
            args.repeats,
            args.missing_share,
            args.seed,
            args.max_bytes,
        )
        record["entries"].append(measured)
        _write(record, args.out)
        print(_summary(measured), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/gpu_probe_grid.py",
        description="Time the Kalman engine at the Stage 7-9 model dimensions.",
    )
    parser.add_argument("--label", required=True, help="host label, e.g. mac or a100")
    parser.add_argument("--out", type=Path, required=True, help="JSON record to write")
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="batch sizes to time (chains or SBC replicates run at once)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="timed calls per batch after compiling",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help=f"time steps per grid configuration (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--grid-filter",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help=f"keep entries matching every key; keys {FILTER_KEYS}",
    )
    parser.add_argument(
        "--grid-json",
        type=Path,
        default=None,
        help="JSON list of entries with steps, states, and cells, replacing the grid",
    )
    parser.add_argument(
        "--missing-share",
        type=float,
        default=MISSING_SHARE,
        help="share of NaN panel cells",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="seed for model and draws"
    )
    parser.add_argument(
        "--max-bytes",
        type=_byte_count,
        default=None,
        help="skip a batch whose compiled memory exceeds this many bytes (e.g. 14e9)",
    )
    parser.add_argument(
        "--hourly-price",
        type=float,
        default=None,
        help="the rented instance's on-demand USD per hour, as billed; none on the Mac",
    )
    parser.add_argument(
        "--four-gpu-price",
        type=float,
        default=None,
        help=(
            "on-demand USD per hour of the whole 4-GPU instance of the same GPU type "
            "(4 times Lambda's per-GPU price), if Lambda rents one; none otherwise"
        ),
    )
    return parser


def _byte_count(text: str) -> int:
    return int(float(text))


def _parse_filters(filters: list[str]) -> dict[str, set[str]]:
    wanted = defaultdict(set)
    for item in filters:
        key, separator, value = item.partition("=")
        if not separator or key not in FILTER_KEYS:
            raise ValueError(
                f"a filter is KEY=VALUE with KEY in {FILTER_KEYS}: {item!r}"
            )
        wanted[key].add(value)
    return dict(wanted)


def _validated_grid_entry(entry: dict) -> dict:
    for key in DIMENSION_KEYS:
        value = entry.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"grid entry {entry} needs a positive integer {key!r}")
    return {key: entry.get(key) for key in FILTER_KEYS} | entry


def _device_memory_stats() -> dict:
    """The first device's memory statistics; empty where it reports none (a CPU)."""
    return jax.devices()[0].memory_stats() or {}


def _compiled_memory(
    ssm: LinearGaussianSSM, panel: np.ndarray, batch: int
) -> dict[str, int]:
    """XLA's memory analysis of the batched value and gradient, compiled from shapes."""
    value_and_grad = engine_probe.batched_value_and_grad()
    log_scales = jax.ShapeDtypeStruct((batch, 2), panel.dtype)
    ssm_shapes, panel_shape = jax.tree.map(
        lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), (ssm, panel)
    )
    lowered = value_and_grad.lower(log_scales, ssm_shapes, panel_shape)
    analysis = lowered.compile().memory_analysis()
    fields = {
        name: int(getattr(analysis, name))
        for name in dir(analysis)
        if name.endswith("_in_bytes") and not name.startswith("host_")
    }
    fields["total_bytes"] = (
        fields["argument_size_in_bytes"]
        + fields["output_size_in_bytes"]
        + fields["temp_size_in_bytes"]
        - fields["alias_size_in_bytes"]
    )
    return fields


def _write(record: dict, out: Path) -> None:
    out.write_text(json.dumps(record, indent=2) + "\n")


def _summary(entry: dict) -> str:
    timings = ", ".join(
        f"batch {batch['batch']}: "
        + (
            f"{batch['median_seconds']:.4g} s"
            if "median_seconds" in batch
            else "skipped"
        )
        for batch in entry["batches"]
    )
    name = f"{entry.get('layout')} stage {entry.get('stage')} {entry.get('wedge')}"
    dims = f"T={entry['steps']} n={entry['states']} p={entry['cells']}"
    return f"{name} {dims}: {timings}"


if __name__ == "__main__":
    main()
