"""Loading a dump into an mms home (replaces `load` of haystack.py and the remember loop and `check_snapshot` of egolife.py of this repository at
bb498a6, which read the scenario's plan; this reads the `Dump` alone): each file remembered at its time, each log added as a source, and
the home checked against the dump when the daemon has taken them in.

A file is given its time `t`, except one whose time is its own (`own_time`: a camera roll's EXIF or `creation_time`), which must come back
stored at it, `time_from` metadata, within a second: one that does not is a `misdated` entry of the report, not an error. A file mms refuses
(the daemon's API says so) is a `refused` entry with its reason. A file mms leaves unchanged because an earlier file of its source has its
bytes is a `refused` entry too, naming that file: mms keeps one item for them (a camera's two identical photos of one moment, a clip a sound
library holds twice). Any other file not stored once (stored twice; unchanged with no such twin) is an error: the home would not hold what
the dump does."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from pathlib import Path

from mmsb.core.contract import Dump, LogSource, ManifestEntry, Refusal
from mmsb.systems.mm.daemon import Api, ApiFailure, wait_for

LOAD_WORKERS = 3  # sources remembered at once: the daemon serves `remember` calls together (2.6 times faster than one at a time)
LOG_WAIT_S = 600.0  # in all, for the daemon to take in the records of the logs
TIME_TOL_S = 1e-6  # the tolerance of a stored time against the dump's: the index keeps a float


def log_files(dump: Dump, log: LogSource) -> list[Path]:
    """The files of a log source, in name order."""
    return sorted((dump.root / log.folder).glob(log.glob))


def log_lines(dump: Dump, log: LogSource) -> list[tuple[float, dict]]:
    """The lines of a log source as `(time, line)`, the time from the line's `log.time_field` (an ISO time), oldest first."""
    rows = [json.loads(text) for file in log_files(dump, log) for text in file.read_text(encoding="utf-8").splitlines() if text.strip()]
    return sorted(((datetime.fromisoformat(row[log.time_field]).timestamp(), row) for row in rows), key=lambda found: found[0])


def remember(api: Api, dump: Dump, entries: Sequence[ManifestEntry]) -> tuple[list[Refusal], list[Refusal]]:
    """`entries` of one source remembered, oldest first (the versions of a file need the order): the refused and the misdated."""
    refused, misdated = [], []
    first: dict[str, ManifestEntry] = {}  # the first file of each content stored
    for e in entries:
        given = {} if e.own_time else {"t": e.t}
        try:
            out = api.call("remember", path=str(dump.root / e.path), source=e.source, **given, **({"name": e.name} if e.name else {}))
        except ApiFailure as error:
            refused.append(Refusal(e.path, str(error)))
            continue
        if not out["stored"] and len(out["unchanged"]) == 1 and (twin := first.get(e.sha256)) is not None:
            when = "at the same time" if twin.t == e.t else f"recorded {e.t - twin.t:+.0f} s from it"
            refused.append(Refusal(e.path, f"the bytes of {twin.path} ({when}): mms keeps one item for them"))
            continue
        if len(out["stored"]) != 1 or out["unchanged"]:
            raise RuntimeError(f"{e.path} was not stored once: {out}")
        first.setdefault(e.sha256, e)
        if e.own_time:
            stored = out["stored"][0]
            if stored.get("time_from") != "metadata" or abs(datetime.fromisoformat(stored["t0"]).timestamp() - e.t) > 1:
                misdated.append(Refusal(e.path, f"stored at {stored['t0']} from {stored.get('time_from')}, not at its own time {e.t} from its metadata"))
    return refused, misdated


def load(api: Api, dump: Dump) -> tuple[list[Refusal], list[Refusal]]:
    """Every file of the dump remembered (no caller, so every row is ambient), each log added as a source, and the daemon waited for until it
    holds every line. A source's files go oldest first in one thread, the sources LOAD_WORKERS at a time, the biggest first. Returns the
    refused and the misdated files."""
    by_source: dict[str, list[ManifestEntry]] = {}
    for e in sorted((e for e in dump.entries if e.media != "records"), key=lambda e: (e.t, e.path)):
        by_source.setdefault(e.source, []).append(e)
    refused, misdated = [], []
    with ThreadPoolExecutor(LOAD_WORKERS) as pool:
        for future in [pool.submit(remember, api, dump, entries) for entries in sorted(by_source.values(), key=len, reverse=True)]:
            more_refused, more_misdated = future.result()
            refused += more_refused
            misdated += more_misdated
    for log in dump.logs:
        api.call("source_set", name=log.name, kind="log", path=str(dump.root / log.folder), glob=log.glob, time_field=log.time_field, max_gb=1)
    for log in dump.logs:
        lines = len(log_lines(dump, log))

        def taken() -> bool:
            found = next((s for s in api.call("status")["sources"] if s["name"] == log.name), None)
            return found is not None and found["records"] >= lines

        wait_for(taken, timeout=LOG_WAIT_S, what=f"the {lines} records of the {log.name} log")
    return refused, misdated


def check_snapshot(home: Path, dump: Dump, refused: Collection[str] = ()) -> None:
    """The home holds what the dump says (the daemon must be stopped, so the index is whole): as many items in each source as the dump has
    files of it, a file with a `length` as an item at its `t` and `t + length` that is shown, and as many records in each log as it has lines,
    at their times. The files of `refused` (paths of the dump that mms refused: they are in the report, with their reasons) are not counted.
    Raises a RuntimeError naming the first difference."""
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30)) as db:
        items = db.execute("SELECT source, name, t0, t1, state FROM items").fetchall()
        records = db.execute("SELECT source, t FROM records").fetchall()
    kept = [e for e in dump.entries if e.path not in refused]
    for source in sorted({e.source for e in kept if e.media != "records"}):
        want, have = sum(1 for e in kept if e.source == source and e.media != "records"), sum(1 for i in items if i[0] == source)
        if want != have:
            raise RuntimeError(f"the home holds {have} items of {source}, the dump {want} files")
    by_name = {(source, name): (t0, t1, state) for source, name, t0, t1, state in items}
    for e in kept:
        if e.length is not None and e.name:
            t0, t1, state = by_name[e.source, e.name]
            if state != "ok" or abs(t0 - e.t) > TIME_TOL_S or abs(t1 - (e.t + e.length)) > TIME_TOL_S:
                raise RuntimeError(f"the home's {e.name} is {state} at {t0}..{t1}, not ok at {e.t}..{e.t + e.length}")
    for log in dump.logs:
        want, have = sorted(when for when, _ in log_lines(dump, log)), sorted(t for source, t in records if source == log.name)
        if len(have) != len(want) or any(abs(a - b) > TIME_TOL_S for a, b in zip(have, want)):
            raise RuntimeError(f"the home's {len(have)} records of {log.name} are not the dump's {len(want)} lines at their times")
