"""The idle-stop guard shuts down only after a whole idle window, never on a busy sample."""

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "infra" / "vm" / "guards" / "idle_stop.py"
_spec = importlib.util.spec_from_file_location("idle_stop", GUARD)
idle_stop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idle_stop)

WINDOW_SECONDS = 45 * 60
LONG_AGO = 1.0  # one second after the epoch: far more than a window ago


def test_the_guard_parses_as_python_3_12():
    # The VM runs the guard with Ubuntu 24.04's system Python, not this project's 3.14.
    ast.parse(GUARD.read_text(), feature_version=(3, 12))


@pytest.mark.parametrize(
    ("load", "gpus", "expected"),
    [
        (0.1, [], True),
        (0.3, [], False),  # at the threshold is not below it
        (0.1, [0.0, 4.0], True),
        (0.1, [0.0, 5.0], False),
        (2.0, [0.0], False),
    ],
)
def test_a_sample_is_idle_only_below_both_thresholds(load, gpus, expected):
    assert (
        idle_stop.is_idle(load, gpus, load_threshold=0.3, gpu_threshold=5) is expected
    )


def test_an_idle_streak_shuts_down_once_it_spans_the_window():
    start, shut_down = idle_stop.next_state(1000.0, True, None, WINDOW_SECONDS)

    assert (start, shut_down) == (1000.0, False)
    almost = 1000.0 + WINDOW_SECONDS - 1
    assert idle_stop.next_state(almost, True, start, WINDOW_SECONDS) == (1000.0, False)
    whole = 1000.0 + WINDOW_SECONDS
    assert idle_stop.next_state(whole, True, start, WINDOW_SECONDS) == (1000.0, True)


def test_a_busy_sample_ends_the_streak():
    assert idle_stop.next_state(5000.0, False, 1000.0, WINDOW_SECONDS) == (None, False)


def _run_guard(tmp_path, *, load, idle_since, device, nvidia_smi="exit 9"):
    """Run the guard as its timer does, with a stub nvidia-smi whose body is nvidia_smi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\n{nvidia_smi}\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    nvidia0 = tmp_path / "nvidia0"
    if device:
        nvidia0.touch()
    loadavg = tmp_path / "loadavg"
    loadavg.write_text(f"{load} 0.10 0.20 1/100 1234\n")
    state = tmp_path / "run" / "idle-since"
    if idle_since is not None:
        state.parent.mkdir()
        state.write_text(f"{idle_since}\n")
    marker = tmp_path / "shutdown-ran"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "IDLE_LOADAVG_FILE": str(loadavg),
        "IDLE_NVIDIA_DEVICE": str(nvidia0),
        "IDLE_STATE_FILE": str(state),
        "IDLE_SHUTDOWN_COMMAND": f"touch {marker}",
    }
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return state, marker, result.stderr


def test_an_old_idle_streak_shuts_the_host_down(tmp_path):
    state, marker, _ = _run_guard(
        tmp_path,
        load=0.05,
        idle_since=LONG_AGO,
        device=True,
        nvidia_smi="printf '2\\n0\\n'",
    )

    assert marker.exists()
    assert float(state.read_text()) == LONG_AGO


def test_a_first_idle_sample_starts_a_streak_without_shutting_down(tmp_path):
    state, marker, _ = _run_guard(tmp_path, load=0.05, idle_since=None, device=False)

    assert not marker.exists()
    assert state.exists()


def test_a_busy_gpu_ends_the_streak(tmp_path):
    state, marker, _ = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=True, nvidia_smi="echo 50"
    )

    assert not marker.exists()
    assert not state.exists()


def test_an_unreadable_gpu_counts_as_busy(tmp_path):
    state, marker, stderr = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=True, nvidia_smi="exit 9"
    )

    assert not marker.exists()
    assert not state.exists()
    assert "unreadable sample" in stderr


def test_without_an_nvidia_device_nvidia_smi_is_not_asked(tmp_path):
    # A CPU size has the driver's nvidia-smi, which fails without a GPU.
    _, marker, _ = _run_guard(
        tmp_path, load=0.05, idle_since=LONG_AGO, device=False, nvidia_smi="exit 9"
    )

    assert marker.exists()
