"""How many execs of one cell run at once (DESIGN 8.1): the one rule, which the runner, every dry run and estimate, `clock.check_queue` and
`clock-check` all ask. A cell runs `jobs` execs at once, unless its system needs the GPU: then it runs `gpu_jobs`, which is 1 unless the
owner passes `--gpu-jobs`, because three execs of one decision model at once crashed this laptop's driver (2 November 2026). `--jobs` and
`--clock` do not raise it. The system's own limit (`Session.max_jobs`: a slow model, an index cut to a horizon) comes on top, and only
lowers it. A command that cannot ask a session yet (`mmsb repeat`, before its stores exist) asks `asked`, the rule without that limit."""

from __future__ import annotations

from collections.abc import Sequence

from mmsb.core.contract import Case, Description, Session

JOBS = 3  # execs at once by default
GPU_JOBS = 1  # ... of a configuration whose system needs the GPU
JOBS_FLAG, GPU_JOBS_FLAG = "--jobs", "--gpu-jobs"


def on_gpu(description: Description | None) -> bool:
    """Whether a configuration's system needs the GPU (None: a configuration with no system)."""
    return description is not None and description.needs_gpu


def flag_of(description: Description | None) -> str:
    """The option that sets how many execs of a configuration run at once."""
    return GPU_JOBS_FLAG if on_gpu(description) else JOBS_FLAG


def asked(description: Description | None, jobs: int, gpu_jobs: int) -> int:
    """The execs at once asked of a configuration: `gpu_jobs` when its system needs the GPU, else `jobs`."""
    return gpu_jobs if on_gpu(description) else jobs


def execs_at_once(description: Description | None, session: Session, cases: Sequence[Case], jobs: int, gpu_jobs: int) -> int:
    """The execs of the cell of `cases` that run at once: what is `asked` of it, or the fewer its session allows (`Session.max_jobs`)."""
    return session.max_jobs(cases, asked(description, jobs, gpu_jobs))


def gpu_warning(name: str, description: Description | None, at_once: int) -> str | None:
    """The line a command prints when a cell that needs the GPU runs several execs at once (the owner asked for them with `--gpu-jobs`), else None."""
    if not on_gpu(description) or at_once < 2:
        return None
    return (f"warning: {name}: {at_once} execs at once on the GPU, because {GPU_JOBS_FLAG} asked for them. Several at once crashed this "
            "laptop's driver once: watch it")
