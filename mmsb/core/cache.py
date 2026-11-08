"""Cached dumps and cached stores (DESIGN 8.4, plan 6.3): what a recipe makes, and what a system ingests it into, is made once and found
again by the next command that wants it. Nothing here names a scenario or a system.

    <temp>/mmb-<key10>-data/            the dump, read-only (key10: the first 10 characters of its key); mmb-<key10>-data.part while it is written
    <cache>/dumps/<key>/                its record: manifest.jsonl (the entries), dump.json (key, scenario, logs), questions.jsonl, recipe.json,
                                        built.json (the seconds it took), data-manifest.json (what `fs.freeze` saw: `fs.changed` compares), and
                                        probe/ (the scenario's probe outputs and outcome, once a build has run it)
    <cache>/stores/<system>-<key10>/    what the system ingested the dump into, and ingest.json (its report, `store_hash`, the dump it read)
    <cache>/locks/                      one lock file for each dump and store being made

The key of a dump is the sha256 of its scenario's name, its recipe and its generator as canonical JSON (core/scenario.py `dump_key`): the
same recipe made by another generator is another dump. The key of a store is that of the dump with the system, its version and its
`store_hash`, which says what ingest depends on: a variant of a system whose ingest does not depend on it shares the store. Neither key
covers a library the generator uses: when a library change alters the bytes, bump the scenario's generator, or remove the cache.

Made whole or not at all. A dump is written into `<final>.part`, frozen there, and renamed; its record is renamed first, so that the one
way a command can die between the two renames, "a record and no data", is the state the temp folder leaves when it is cleaned. A `.part`
left behind by a command that died is removed (said aloud) and made again; a command that is still making it holds the lock, and the
second command stops with `Busy` and touches nothing. The lock is a byte of a file the operating system releases when its process ends,
a few milliseconds after however it ends (msvcrt on Windows, flock elsewhere), so a lock is never left by a crash and a `.part` with no
lock held is never a live build.

A dump that is there is checked against the stats frozen with it (`fs.changed`). A file changed or added is an error naming it, never a
rebuild, because a dump that changed may be an escape (an agent with its own tools wrote to it) or a failing disk, and what ran over it
ran over other recordings. A file gone, or all of it (the temp folder was cleaned), is written again from the recipe and compared with
the record: the same files, sizes and hashes, or an error that leaves the record as it was, so that every later command gives it too."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mmsb.core import fs
from mmsb.core.contract import Description, Dump, IngestReport, LogSource, ManifestEntry, Refusal, System
from mmsb.core.question import write_questions
from mmsb.core.scenario import Scenario, dump_key

KEY_CHARS = 10
DUMPS, STORES, LOCKS = "dumps", "stores", "locks"
MANIFEST, DUMP, QUESTIONS, RECIPE, BUILT, STATS, PROBE = ("manifest.jsonl", "dump.json", "questions.jsonl", "recipe.json", "built.json",
                                                         "data-manifest.json", "probe")
INGESTED = "ingest.json"  # in a store: the system's report of ingesting the dump, written when the ingest is done
OUTCOME = "outcome.json"  # in probe/: the probe's outcome, the seconds it took and the names of the files it wrote into the run folder
CACHED, LOST, NEW, CHANGED, ORPHAN = "cached", "lost", "new", "changed", "orphan"  # the states of a dump (`dump_state`)
BUILT_NOW, FOUND, REWRITTEN = "built", "cached", "rewritten"  # what `cached_dump` did


class Busy(RuntimeError):
    """Another command is making what this one asked for: it holds the lock. Nothing was touched."""


@dataclass(frozen=True)
class DumpState:
    """What the cache and the temp folder hold of one dump (`dump_state`): `detail` is the file that differs when the state is
    `CHANGED` (or `LOST` with a file gone), `seconds` how long writing it took when it was made (None: it never was)."""

    key: str
    final: Path
    meta: Path
    state: str
    detail: str | None
    seconds: float | None

    @property
    def problem(self) -> str | None:
        """Why a command stops at this dump, else None."""
        if self.state == CHANGED:
            return (f"the cached dump {self.final} is not as it was frozen ({self.detail}): a dump that changed may be an escape, or a disk that "
                    f"failed, and what ran over it ran over other recordings. Remove {self.final} and {self.meta} to write it again from "
                    "the recipe")
        if self.state == ORPHAN:
            return (f"{self.final} is there and the cache has no record of it ({self.meta} is missing), so nothing says it is the dump "
                    f"{self.key[:KEY_CHARS]}: remove {self.final} to write it again from the recipe")
        return None


@dataclass(frozen=True)
class CachedDump:
    dump: Dump  # its root is the frozen folder
    meta: Path  # the dump's record in the cache
    seconds: float  # what writing it took when it was made
    what: str  # BUILT_NOW, FOUND or REWRITTEN


@dataclass(frozen=True)
class StoreState:
    """What the cache holds of the store of one system over one dump."""

    final: Path
    key: str
    cached: bool
    seconds: float | None  # the ingest's seconds when it was made
    root: str | None  # the dump's folder it was ingested over


@dataclass(frozen=True)
class CachedStore:
    path: Path
    report: IngestReport
    hit: bool  # it was there: `ingest` was not called


# -- keys and places ----------------------------------------------------------------------------------------------------------------

def store_key(description: Description, dump: str) -> str:
    """The sha256 of the system, its version, its `store_hash` and the key of the dump as canonical JSON (sorted keys, no spaces). The
    variant and `config_hash` are not in it: a variant shares the store of the others when ingest does not depend on it."""
    body = {"system": description.name, "version": description.version, "store_hash": description.store_hash, "dump": dump}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def dump_folder(temp: Path, key: str) -> Path:
    return Path(temp) / f"mmb-{key[:KEY_CHARS]}-data"


def dump_meta(cache: Path, key: str) -> Path:
    return Path(cache) / DUMPS / key


def store_folder(cache: Path, description: Description, key: str) -> Path:
    return Path(cache) / STORES / f"{description.name}-{key[:KEY_CHARS]}"


def part_of(path: Path) -> Path:
    return path.with_name(path.name + ".part")


# -- locks and whole-or-not-at-all ----------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def locked(cache: Path, name: str, what: str) -> Iterator[None]:
    """`cache/locks/<name>.lock` held for the block, or `Busy` (naming `what`) when another process holds it. The lock is one byte of the
    file, which the operating system takes back when the process ends, however it ends."""
    file = Path(cache) / LOCKS / f"{name}.lock"
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Busy(f"another mmsb command is making {what} (it holds {file}): wait for it to finish, then run this again, which will "
                       "find it cached") from None
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def clear_part(part: Path) -> None:
    """A `.part` that is there is the work of a command that died (the lock is held, or this would not be called): removed, aloud."""
    if part.exists():
        print(f"{part} was left by a command that did not finish: removing it, and making it again", flush=True)
        fs.remove_tree(part)


# -- dumps --------------------------------------------------------------------------------------------------------------------------

def dump_state(scenario: Scenario, recipe: Mapping[str, object], temp: Path, cache: Path) -> DumpState:
    """Where the dump of `recipe` stands, reading and writing nothing. The record is the proof that the data is the dump: data with no
    record is `ORPHAN`; a record with the data gone (the folder or any file in it) is `LOST`; a file changed or added is `CHANGED`."""
    key = dump_key(scenario.name, recipe, scenario.generator)
    final, meta = dump_folder(temp, key), dump_meta(cache, key)
    if not meta.is_dir():
        return DumpState(key, final, meta, ORPHAN if final.exists() else NEW, None, None)
    seconds = fs.read_json(meta / BUILT)["seconds"]
    if not final.is_dir():
        return DumpState(key, final, meta, LOST, "the folder", seconds)
    found = fs.changed(final, meta / STATS)
    if found is None:
        return DumpState(key, final, meta, CACHED, None, seconds)
    return DumpState(key, final, meta, LOST if found.startswith("gone: ") else CHANGED, found, seconds)


def entries_of(meta: Path) -> tuple[ManifestEntry, ...]:
    return tuple(ManifestEntry(**json.loads(line)) for line in (meta / MANIFEST).read_text(encoding="utf-8").splitlines())


def read_dump(meta: Path, root: Path) -> Dump:
    """The dump the record `meta` holds, its data at `root`."""
    raw = fs.read_json(meta / DUMP)
    return Dump(raw["key"], Path(root), raw["scenario"], entries_of(meta), tuple(LogSource(**log) for log in raw["logs"]))


def write_record(meta: Path, dump: Dump, questions, recipe: Mapping[str, object]) -> None:
    """The files of a dump's record in the folder `meta` (made here): the entries, the dump's key, scenario and logs, the questions and
    the recipe."""
    meta.mkdir(parents=True, exist_ok=True)
    (meta / MANIFEST).write_text("".join(json.dumps(dataclasses.asdict(e)) + "\n" for e in dump.entries), encoding="utf-8")
    fs.write_json(meta / DUMP, {"key": dump.key, "scenario": dump.scenario, "logs": [dataclasses.asdict(log) for log in dump.logs]})
    write_questions(meta / QUESTIONS, questions)
    fs.write_json(meta / RECIPE, recipe)


def cached_dump(scenario: Scenario, recipe: Mapping[str, object], temp: Path, cache: Path, workers: int = 3) -> CachedDump:
    """The dump `scenario` makes of `recipe`, in `temp`, frozen: found when it is there and as it was frozen, written again from the recipe
    when the temp folder lost it (and compared with its record), else made. Made into `<folder>.part` with its record, which `scenario.build`
    does not freeze, and renamed when whole; a builder that raises leaves nothing but the error. `Busy` when another command is making it;
    a `RuntimeError` naming the file when the dump changed (`DumpState.problem`) or what was written again is not the dump."""
    state = dump_state(scenario, recipe, temp, cache)
    if state.state == CACHED:
        return CachedDump(read_dump(state.meta, state.final), state.meta, state.seconds, FOUND)
    if state.problem:
        raise RuntimeError(state.problem)
    with locked(cache, f"dump-{state.key[:KEY_CHARS]}", f"the dump {state.final.name}"):
        state = dump_state(scenario, recipe, temp, cache)  # (another command may have finished it while this one looked)
        if state.state == CACHED:
            return CachedDump(read_dump(state.meta, state.final), state.meta, state.seconds, FOUND)
        if state.problem:
            raise RuntimeError(state.problem)
        return write_again(scenario, recipe, state, workers) if state.state == LOST else write_new(scenario, recipe, state, workers)


def write_new(scenario: Scenario, recipe: Mapping[str, object], state: DumpState, workers: int) -> CachedDump:
    final, meta = state.final, state.meta
    part, meta_part = part_of(final), part_of(meta)
    final.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    clear_part(part)
    clear_part(meta_part)
    began = time.monotonic()
    try:
        dump, questions = scenario.build(recipe, part, workers)
        if dump.key != state.key:
            raise RuntimeError(f"{scenario.name} built the dump {dump.key}, and the cache asked for {state.key}: its key is not "
                               "dump_key(scenario, recipe, generator)")
        write_record(meta_part, dump, questions, recipe)
        fs.freeze(part, meta_part / STATS)
        seconds = round(time.monotonic() - began, 1)
        fs.write_json(meta_part / BUILT, {"seconds": seconds, "scenario": scenario.name, "generator": scenario.generator,
                                          "created": datetime.now().isoformat(timespec="seconds")})
        os.replace(meta_part, meta)  # the record first: a command that dies between the two leaves a record and no data, which is `LOST`
        os.replace(part, final)
    except BaseException:
        for leftover in (part, meta_part):
            if leftover.exists():
                fs.remove_tree(leftover)
        raise
    return CachedDump(read_dump(meta, final), meta, seconds, BUILT_NOW)


def write_again(scenario: Scenario, recipe: Mapping[str, object], state: DumpState, workers: int) -> CachedDump:
    """The data of a dump the temp folder lost, written again and compared with the record (the same files, sizes and hashes). It is
    frozen and its stats replace the old ones (the files have new times); when it is not the dump it is removed and the record is
    left as it was."""
    final, meta = state.final, state.meta
    part, stats = part_of(final), meta / (STATS + ".part")
    clear_part(part)
    print(f"the dump {final.name} is gone from {final.parent}: writing it again from the recipe", flush=True)
    if final.exists():
        fs.remove_tree(final)  # (what is left of it)
    try:
        again, _ = scenario.build(recipe, part, workers)
        was = {e.path: (e.size, e.sha256) for e in entries_of(meta)}
        now = {e.path: (e.size, e.sha256) for e in again.entries}
        if was != now:
            differs = sorted(path for path in was.keys() | now.keys() if was.get(path) != now.get(path))
            raise RuntimeError(f"the data written again is not the dump {final.name} was built with ({len(differs)} files differ, the first "
                               f"{differs[0]}): the generator or a library changed since it was cached, so a run over it would be over other "
                               f"recordings. Its record is left as it was, so every command gives this error; to use the new bytes, remove "
                               f"{meta} and the stores ingested from this dump (they hold the old ones), or give the scenario a new generator")
        fs.freeze(part, stats)
        os.replace(stats, meta / STATS)  # the new stats first: a command that dies before the rename leaves them with the data in `.part`
        os.replace(part, final)
    except BaseException:
        for leftover in (part, stats):
            if leftover.exists():
                fs.remove_tree(leftover)
        raise
    return CachedDump(read_dump(meta, final), meta, state.seconds, REWRITTEN)


# -- the shortcut probe, kept with the dump -------------------------------------------------------------------------------------------

def cached_probe(meta: Path, cache: Path, folder: Path, run: Callable[[], dict]) -> tuple[dict, float, bool]:
    """The probe's outcome over the dump whose record is `meta`, for the run in `folder`: the outcome and the files a probe made earlier
    put in a run folder, copied into `folder`, when there are any; else `run()` (which writes its files into `folder`) and what it wrote,
    kept for the next run. Returns the outcome, the seconds the probe took, and whether it was cached. To probe again, remove
    `meta/probe`: the key covers the dump and not the probe's code."""
    kept = meta / PROBE
    if kept.is_dir():
        noted = fs.read_json(kept / OUTCOME)
        for name in noted["wrote"]:
            (shutil.copytree if (kept / name).is_dir() else shutil.copy2)(kept / name, folder / name)
        return noted["outcome"], noted["seconds"], True
    before = {p.name for p in folder.iterdir()}
    began = time.monotonic()
    outcome = run()
    seconds = round(time.monotonic() - began, 1)
    wrote = sorted(p.name for p in folder.iterdir() if p.name not in before)
    with locked(cache, f"dump-{meta.name[:KEY_CHARS]}", f"the record of the dump {meta.name[:KEY_CHARS]}"):
        if not kept.is_dir():  # (another command may have kept its own meanwhile: the first stays)
            part = part_of(kept)
            clear_part(part)
            part.mkdir(parents=True)
            for name in wrote:
                (shutil.copytree if (folder / name).is_dir() else shutil.copy2)(folder / name, part / name)
            fs.write_json(part / OUTCOME, {"outcome": outcome, "seconds": seconds, "wrote": wrote})
            os.replace(part, kept)
    return outcome, seconds, False


# -- stores -------------------------------------------------------------------------------------------------------------------------

def report_of(store: Path) -> IngestReport:
    raw = fs.read_json(Path(store) / INGESTED)
    return IngestReport(raw["seconds"], raw["gpu_seconds"], raw["bytes_written"], tuple(Refusal(**r) for r in raw["refused"]),
                        tuple(Refusal(**r) for r in raw["misdated"]), raw["measured"])


def write_ingest(store: Path, report: IngestReport, description: Description, dump: Dump) -> None:
    """ingest.json: the report, the `store_hash` of the system that made the store (`check_stores` compares it), and the dump it read and
    where (a store may read its dump's files where it was ingested)."""
    fs.write_json(Path(store) / INGESTED, {**dataclasses.asdict(report), "store_hash": description.store_hash, "version": description.version,
                                           "dump": dump.key, "dump_root": str(dump.root)})


def store_state(description: Description, dump: str, cache: Path) -> StoreState:
    """Whether the store of `description`'s system over the dump with key `dump` is in the cache, reading and writing nothing."""
    key = store_key(description, dump)
    final = store_folder(cache, description, key)
    held = fs.read_json(final / INGESTED) if (final / INGESTED).exists() else None
    return StoreState(final, key, held is not None, held["seconds"] if held else None, held["dump_root"] if held else None)


def cached_store(system: System, variant: str | None, dump: Dump, cache: Path) -> CachedStore:
    """The store `system` (for `variant`) ingests `dump` into, found when it is in the cache and else made: `system.ingest` into
    `<store>.part`, `ingest.json` written there, and the folder renamed. A store that is there is not ingested again, and `Busy` stops a
    command while another is ingesting it. A store ingested over the dump in another place is an error: it may read its files there."""
    description = system.describe(variant)
    state = store_state(description, dump.key, cache)
    if state.cached:
        return found_store(state, dump)
    if state.final.exists():
        raise RuntimeError(f"{state.final} holds no {INGESTED}, and a store is renamed into place whole: it was changed by hand. Remove it")
    with locked(cache, f"store-{state.final.name}", f"the store {state.final.name}"):
        state = store_state(description, dump.key, cache)
        if state.cached:
            return found_store(state, dump)
        part = part_of(state.final)
        state.final.parent.mkdir(parents=True, exist_ok=True)
        clear_part(part)
        part.mkdir()
        try:
            report = system.ingest(dump, part)
            write_ingest(part, report, description, dump)
            os.replace(part, state.final)
        except BaseException:
            if part.exists():
                fs.remove_tree(part)
            raise
    return CachedStore(state.final, report, False)


def found_store(state: StoreState, dump: Dump) -> CachedStore:
    held = fs.read_json(state.final / INGESTED)
    if held["dump_root"] != str(dump.root):
        raise RuntimeError(f"the store {state.final} was ingested over the dump in {held['dump_root']}, and this dump is in {dump.root}: a store "
                           "may read its dump's files where it found them, so use the same --temp, or remove the store to ingest again")
    return CachedStore(state.final, report_of(state.final), True)


def clone_store(store: Path, into: Path) -> None:
    """A copy of the cached `store` at `into` that costs no space: every file a hard link (the store and `into` are on one volume), made
    whole or not at all. Nothing writes to a store (a session copies it into its own home), so the links are never written through."""
    part = part_of(into)
    clear_part(part)
    shutil.copytree(store, part, copy_function=os.link)
    os.replace(part, into)
