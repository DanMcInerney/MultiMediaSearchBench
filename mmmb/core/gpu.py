"""Whether the GPU is free to run a system's model on, and whether it is still there (moved from bench.py of this repository at
bb498a6; `check_gpu` is keyed on `Description.needs_gpu` where it was keyed on the model)."""

from __future__ import annotations

import os
import shutil
import subprocess

from mmmb.core.contract import Description

# Windows' own desktop, which nvidia-smi may list as a compute app (explorer.exe, 2026-11-06, with its memory "[N/A]", as every
# process's is under Windows' display driver): it loads no model, and no run would start beside it
DESKTOP = ("explorer.exe", "dwm.exe")
# the process ids, separated by commas, that the owner lets a run share the GPU with (2026-11-07: a test suite of another project and
# the Codex app's on-device model, 1.4 GB in all, that the owner chose to run beside): ids, not names, so nothing else slips by
BESIDE = "MMMB_GPU_BESIDE"


def gpu_busy() -> str | None:
    """What nvidia-smi lists as running on the GPU apart from Windows' desktop (`DESKTOP`) and the processes the owner named in
    `BESIDE`, or None when nothing else does. The GPU runs alone (on Windows with a display the list can be empty while a window uses
    it: look at the machine too)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi is not on PATH, so whether the GPU is free for a real model cannot be told")
    done = subprocess.run([exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True,
                          text=True, timeout=30)
    if done.returncode:
        raise RuntimeError(f"nvidia-smi exited {done.returncode}: {done.stderr.strip()[-300:]}")
    apps = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return "; ".join(a for a in apps if program(a) not in DESKTOP and pid(a) not in beside()) or None


def program(app: str) -> str:
    """The program's file name, lower case, of a line `pid, process_name` of nvidia-smi."""
    return app.rpartition(",")[2].strip().replace("/", "\\").rpartition("\\")[2].lower()


def pid(app: str) -> str:
    """The process id of a line `pid, process_name` of nvidia-smi."""
    return app.partition(",")[0].strip()


def beside() -> frozenset[str]:
    """The process ids the owner named in `BESIDE`."""
    return frozenset(p.strip() for p in os.environ.get(BESIDE, "").split(",") if p.strip())


def check_gpu(description: Description) -> None:
    """Raises when the system needs the GPU and something else is running on it; says so when it runs beside processes the owner
    named (`BESIDE`)."""
    if not description.needs_gpu:
        return
    if busy := gpu_busy():
        raise RuntimeError(f"the GPU is busy, and it runs alone: {busy}")
    if shared := sorted(beside()):
        print(f"the GPU is shared with the processes {', '.join(shared)} ({BESIDE}): timings may be slower than alone", flush=True)


def gpu_lost() -> str | None:
    """Why the GPU cannot be read any more, else None: after a driver crash nvidia-smi says "GPU is lost", a model worker cannot
    load, and an agent answers from the little else it has (2026-11-04: 18 results so before anything noticed)."""
    try:
        gpu_busy()
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return str(e)
    return None
