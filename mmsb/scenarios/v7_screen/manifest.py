"""The manifest of a written V7 week: a `ManifestEntry` for each file of the plan and each log, which is what a system and the question
writer see of the dump. Sizes, mtimes and sha256 are measured (`measure_disk`) or given, so the entries of the recorded pilot can be
made from its recorded manifest with no data folder."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mmsb.core import fs
from mmsb.core.contract import LogSource, ManifestEntry
from mmsb.scenarios.v7_screen.plan import LOG_SOURCE, PHONE, Plan

MEDIA = {"photo": "image", "clip": "video", "screen": "video", "porch": "video", "mic": "audio", "note": "text", "config": "text"}
LOGS = LogSource(name=LOG_SOURCE, folder="logs", glob="*.jsonl", time_field="ts")  # the log source, as `bench.build` added it


def file_rows(plan: Plan) -> list[list]:
    """The plan's files as `[path, source, time, name, kind, spec]` rows: what `files_hash` covers."""
    return [[f.path, f.source, f.t, f.name, f.kind, f.spec] for f in plan.files]


def log_path(service: str) -> str:
    return f"logs/{service}.jsonl"


def log_times(plan: Plan, service: str) -> tuple[float, float]:
    """The time of the first and of the last line of a service's log (a log has the time of its lines, and no other)."""
    lines = plan.logs[service]
    if not lines:
        raise ValueError(f"the {service} log has no line, so it has no time")
    return datetime.fromisoformat(lines[0]["ts"]).timestamp(), datetime.fromisoformat(lines[-1]["ts"]).timestamp()


def entries_of(plan: Plan, measured: Callable[[str], tuple[int, int, str]]) -> tuple[ManifestEntry, ...]:
    """An entry for each file of the plan, oldest first, then each log. `measured(path)` gives the `(size, mtime_ns, sha256)` of one.
    A file's `t` is when it was taken or starts; a log's is the time of its first line. The phone's files carry their own time."""
    entries = []
    for f in plan.files:
        size, mtime_ns, sha256 = measured(f.path)
        entries.append(ManifestEntry(f.path, f.source, f.t, f.kind, MEDIA[f.kind], f.name, f.source == PHONE, size, mtime_ns, sha256))
    for service in plan.logs:
        path = log_path(service)
        size, mtime_ns, sha256 = measured(path)
        entries.append(ManifestEntry(path, LOG_SOURCE, log_times(plan, service)[0], "log", "records", None, False, size, mtime_ns, sha256))
    return tuple(entries)


def measure_disk(root: Path) -> Callable[[str], tuple[int, int, str]]:
    """`measured` for the files under `root`."""
    def measured(path: str) -> tuple[int, int, str]:
        info = (root / path).stat()
        return info.st_size, info.st_mtime_ns, fs.sha256_of(root / path)

    return measured
