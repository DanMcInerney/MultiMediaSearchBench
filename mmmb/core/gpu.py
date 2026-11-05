"""Whether the GPU is free to run a system's model on, and whether it is still there (moved from bench.py of this repository at
bb498a6; `check_gpu` is keyed on `Description.needs_gpu` where it was keyed on the model)."""

from __future__ import annotations

import shutil
import subprocess

from mmmb.core.contract import Description


def gpu_busy() -> str | None:
    """What nvidia-smi lists as running on the GPU, or None when nothing does. The GPU runs alone (on Windows with a display
    the list can be empty while a window uses it: look at the machine too)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi is not on PATH, so whether the GPU is free for a real model cannot be told")
    done = subprocess.run([exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True,
                          text=True, timeout=30)
    if done.returncode:
        raise RuntimeError(f"nvidia-smi exited {done.returncode}: {done.stderr.strip()[-300:]}")
    return "; ".join(line.strip() for line in done.stdout.splitlines() if line.strip()) or None


def check_gpu(description: Description) -> None:
    """Raises when the system needs the GPU and something else is running on it."""
    if description.needs_gpu and (busy := gpu_busy()):
        raise RuntimeError(f"the GPU is busy, and it runs alone: {busy}")


def gpu_lost() -> str | None:
    """Why the GPU cannot be read any more, else None: after a driver crash nvidia-smi says "GPU is lost", a model worker cannot
    load, and an agent answers from the little else it has (2026-11-04: 18 results so before anything noticed)."""
    try:
        gpu_busy()
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return str(e)
    return None
