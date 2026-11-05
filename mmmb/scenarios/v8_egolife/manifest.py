"""The manifest of a written V8 dump: a `ManifestEntry` for each clip and for each day's transcript file, which is what a system, the
files view and the question writer see of the dump. A clip is a video of the source `glasses` that starts at its `t` and plays its `length`;
a day's transcript is a log of the source `transcript` whose `t` is the local midnight that starts the day and whose `length` is the day, so
that a moment of the day is its seconds after `t`. Sizes, mtimes and sha256 are measured (`measure_disk`) or given, so the entries of the
recorded run can be made from its recorded manifest with no download."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from mmmb.core import fs
from mmmb.core.contract import LogSource, ManifestEntry
from mmmb.scenarios.v8_egolife.plan import TEXT, VIDEO, Plan, local
from mmmb.scenarios.v8_egolife.source import DAY_S, plain

TRANSCRIPT = LogSource(name=TEXT, folder=TEXT, glob="*.jsonl", time_field="end")  # the log source, as the old build added it


def file_rows(plan: Plan) -> list[list]:
    """The plan's files as `[path, source, time, name, kind, spec]` rows: what `files_hash` covers."""
    return [[f.path, f.source, f.t, f.name, f.kind, f.spec] for f in plan.files]


def transcript_path(day: date) -> str:
    return f"{TEXT}/{day.isoformat()}.jsonl"


def entries_of(plan: Plan, measured: Callable[[str], tuple[int, int, str]]) -> tuple[ManifestEntry, ...]:
    """An entry for each clip of the plan, oldest first, then each day's transcript. `measured(path)` gives the `(size, mtime_ns,
    sha256)` of one."""
    entries = []
    for f in plan.files:
        size, mtime_ns, sha256 = measured(f.path)
        entries.append(ManifestEntry(f.path, VIDEO, f.t, "clip", "video", f.name, False, size, mtime_ns, sha256, f.spec["length"]))
    for n, day in enumerate(plan.days, 1):
        path = transcript_path(day)
        size, mtime_ns, sha256 = measured(path)
        entries.append(ManifestEntry(path, TEXT, local(plan.days, plain(n, 0)), "log", "records", None, False, size, mtime_ns, sha256,
                                     float(DAY_S)))
    return tuple(entries)


def measure_disk(root: Path, paths: Sequence[str], workers: int) -> Callable[[str], tuple[int, int, str]]:
    """`measured` for the `paths` under `root`, each hashed once, `workers` at a time (the clips are 40 GB for three days)."""
    def measure(path: str) -> tuple[int, int, str]:
        info = (root / path).stat()
        return info.st_size, info.st_mtime_ns, fs.sha256_of(root / path)

    with ThreadPoolExecutor(workers) as pool:
        found = dict(zip(paths, pool.map(measure, paths)))
    return found.__getitem__
