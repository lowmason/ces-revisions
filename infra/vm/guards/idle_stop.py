#!/usr/bin/env python3
"""Stop the instance once it has been idle for a whole window.

ces-idle-stop.timer runs this as root every five minutes. A sample is idle when the
1-minute load average is below LOAD_THRESHOLD and, where an NVIDIA device exists, every
GPU's utilization is below GPU_UTILIZATION_THRESHOLD percent. The first idle sample of a
streak is recorded under /run, which a reboot clears, so a streak never spans a stop and
start. A busy sample, or one that cannot be read, ends the streak. Once a streak spans
IDLE_WINDOW_MINUTES, the host shuts down, and the instance's shutdown behavior turns that
into a stopped instance. Written for the VM's system Python (3.12 on Ubuntu 24.04).
"""

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

STATE_FILE = Path(os.environ.get("IDLE_STATE_FILE", "/run/ces-revisions/idle-since"))
LOADAVG_FILE = Path(os.environ.get("IDLE_LOADAVG_FILE", "/proc/loadavg"))
NVIDIA_DEVICE = Path(os.environ.get("IDLE_NVIDIA_DEVICE", "/dev/nvidia0"))
SHUTDOWN_COMMAND = os.environ.get("IDLE_SHUTDOWN_COMMAND", "shutdown -h now").split()


def is_idle(
    load: float,
    gpu_utilizations: list[float],
    load_threshold: float,
    gpu_threshold: float,
) -> bool:
    """Whether one sample is idle: low load, and every GPU (if any) below its threshold."""
    return load < load_threshold and all(u < gpu_threshold for u in gpu_utilizations)


def next_state(
    now: float, idle: bool, idle_since: float | None, window_seconds: float
) -> tuple[float | None, bool]:
    """The streak start to record (None ends the streak), and whether to shut down."""
    if not idle:
        return None, False
    start = now if idle_since is None else idle_since
    return start, now - start >= window_seconds


def main() -> None:
    window_seconds = 60 * float(os.environ.get("IDLE_WINDOW_MINUTES", "45"))
    load_threshold = float(os.environ.get("LOAD_THRESHOLD", "0.3"))
    gpu_threshold = float(os.environ.get("GPU_UTILIZATION_THRESHOLD", "5"))
    try:
        idle = is_idle(_load(), _gpu_utilizations(), load_threshold, gpu_threshold)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(
            f"idle-stop: treating an unreadable sample as busy: {error}",
            file=sys.stderr,
        )
        idle = False
    start, shut_down = next_state(time.time(), idle, _idle_since(), window_seconds)
    if start is None:
        STATE_FILE.unlink(missing_ok=True)
    else:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(f"{start}\n")
    if shut_down:
        print(f"idle-stop: idle since {start:.0f}; shutting down", file=sys.stderr)
        subprocess.run(SHUTDOWN_COMMAND, check=True)


def _load() -> float:
    return float(LOADAVG_FILE.read_text().split()[0])


def _gpu_utilizations() -> list[float]:
    # On a CPU size the driver's nvidia-smi is installed but fails, so ask it only
    # where the device exists.
    if not NVIDIA_DEVICE.exists():
        return []
    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    return [float(value) for value in output.split()]


def _idle_since() -> float | None:
    # No file, or an unreadable one, means no streak is under way.
    with contextlib.suppress(OSError, ValueError):
        return float(STATE_FILE.read_text())
    return None


if __name__ == "__main__":
    main()
