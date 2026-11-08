"""Making a run: the dump a scenario writes, cached and frozen in the temp folder; the questions; the stores the systems ingest the dump
into, cached and linked into the run; the probe; and run.json, last (moved from `build`, `ensure_data` and `ensure_bin` of bench.py and
`build` of egolife.py of this repository at bb498a6, now for any scenario and any system, and through core/cache.py: what an earlier run
of the same recipe and system made is found, not made again).

    <out>/<name>/questions.jsonl  data-manifest.json  dump.json  stores/<system>/ (the system's store, ingest.json)  run.json
    <temp>/mmb-<key10>-data/      the dump, read-only, named by its key; mmb-<tag>-bin/ holds the ffmpeg
    <cache>/dumps/<key>/, <cache>/stores/<system>-<key10>/    the dump's record and the stores (core/cache.py)

`dump.json` is the dump's entries and logs (run.json does not carry them): what `Scenario.view`, `System.open` and the audits read. A store
is made once for a dump and a system, by `System.ingest`, and its `ingest.json` says how: the files the system refused (`refused`, with their
reasons), a file it stored at a time other than its own (`misdated`: every question about it would be asked of a wrong index, so the run is
refused, here and by every command that would use the store, though the store was made for an earlier run), and the `store_hash` of the
system that made it (a store of another version of the system is refused too). A run's `stores/<system>` is a copy of the cached store that
costs no space (hard links), so a run is read by the same code whether its store was made for it or found."""

from __future__ import annotations

import dataclasses
import secrets
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from mmsb import systems
from mmsb.core import fs
from mmsb.core.cache import (BUILT_NOW, FOUND, INGESTED, PROBE, QUESTIONS, REWRITTEN, STATS, CachedDump, cached_dump, cached_probe, cached_store,
                             clone_store, report_of, store_state)
from mmsb.core.config import Config, describe_configs
from mmsb.core.contract import Description, Dump, IngestReport, LogSource, ManifestEntry, System
from mmsb.core.paths import check_apart, check_volume
from mmsb.core.probe import run_probe
from mmsb.core.runfolder import RunFolder, check_temp, commit_of
from mmsb.core.scenario import Scenario

WORKERS = 3  # files written, measured and hashed at once
SHOWN = 5  # refusals told by name in an error


class Misdated(RuntimeError):
    """A system stored files at a time other than their own: a run over its store is refused."""


def store_of(folder: Path, system: str) -> Path:
    """Where the store of `system` is in the run folder `folder`."""
    return Path(folder) / "stores" / system


def write_dump(folder: Path, dump: Dump) -> None:
    fs.write_json(folder / "dump.json", {"key": dump.key, "root": str(dump.root), "scenario": dump.scenario,
                                         "entries": [dataclasses.asdict(e) for e in dump.entries],
                                         "logs": [dataclasses.asdict(log) for log in dump.logs]})


def read_dump(run: RunFolder) -> Dump:
    """The dump the run was built from, as dump.json has it. Its root must be the run's data: a run's results are of that dump."""
    file = run.folder / "dump.json"
    if not file.exists():
        raise FileNotFoundError(f"{run.folder} has no dump.json, so what its data holds is not known: it was not built by mmsb build")
    raw = fs.read_json(file)
    if Path(raw["root"]) != run.data:
        raise ValueError(f"{file} is of the dump in {raw['root']}, not of the run's data in {run.data}")
    return Dump(raw["key"], Path(raw["root"]), raw["scenario"], tuple(ManifestEntry(**e) for e in raw["entries"]),
                tuple(LogSource(**log) for log in raw["logs"]))


def refuse_misdated(system: str, report: IngestReport) -> None:
    """Raises `Misdated`, naming the files, when `system` stored any of the dump at a time other than its own."""
    if report.misdated:
        named = "; ".join(f"{m.path} ({m.reason})" for m in report.misdated[:SHOWN])
        more = f"; and {len(report.misdated) - SHOWN} more" if len(report.misdated) > SHOWN else ""
        raise Misdated(f"{system} stored {len(report.misdated)} files at a time other than their own, so what it is asked of them would "
                       f"be answered from a wrong index and no number from this run could be trusted: {named}{more}. Remove its store "
                       "and ingest again once it stores them at their time")


def store_hashes(descriptions: Mapping[Config, Description | None]) -> dict[str, str]:
    """The `store_hash` of each system the configurations name. A system has one store, so its variants must agree on it."""
    out: dict[str, str] = {}
    for config, description in descriptions.items():
        if description is not None and out.setdefault(config.system, description.store_hash) != description.store_hash:
            raise ValueError(f"the variants of {config.system} named together need different stores: {config.name}'s is not the others'")
    return out


def check_stores(folder: Path, hashes: Mapping[str, str]) -> dict[str, object]:
    """The stores of the systems in `hashes` (each one's current `store_hash`) are there, finished, made by that version of the system
    and not misdated: returns what they measured. Raises otherwise."""
    measured: dict[str, object] = {}
    for name, want in hashes.items():
        store = store_of(folder, name)
        if not (store / INGESTED).exists():
            raise FileNotFoundError(f"{name} has no store in {folder}: mmsb ingest {folder.name} --configs {name} makes it")
        if (had := fs.read_json(store / INGESTED).get("store_hash")) != want:
            raise ValueError(f"the store of {name} in {folder.name} was made by {name} with store_hash {had}, and it is {want} now: {name} or "
                             f"what it stores changed since, so what is asked of the store is not what the run was built with. Build a "
                             f"new run (or, if nothing has run over this one, remove {store} and ingest again)")
        report = report_of(store)
        refuse_misdated(name, report)
        for key, value in report.measured.items():
            if measured.setdefault(key, value) != value:
                raise ValueError(f"two systems measured {key} as {measured[key]} and {value}")
    return measured


def ingest_stores(folder: Path, dump: Dump, descriptions: Mapping[Config, Description | None], load: Callable[[str], System] = systems.load,
                  *, cache: Path) -> dict[str, object]:
    """The stores of the systems of `descriptions` that the run lacks: each is the cached store of the system over this dump (made by
    `System.ingest` when the cache has none, found when it has: `cached_store`) copied into `stores/<system>` with hard links, and then
    checked (`check_stores`): the report of a store that was misdated refuses the run though the store was made for an earlier one. A store
    of the run whose copy did not finish is an error, not made again."""
    hashes = store_hashes(descriptions)
    variants = {c.system: c.variant for c in descriptions if c.system is not None}  # (variants of a system agree on its store: `store_hashes`)
    for name in hashes:
        store = store_of(folder, name)
        if (store / INGESTED).exists():
            continue
        if store.exists():
            raise RuntimeError(f"{store} holds an ingest that did not finish: remove it, and ingest again")
        system = load(name)  # a system whose extra is missing fails here, before the store is made
        description = next(d for c, d in descriptions.items() if c.system == name)
        if not store_state(description, dump.key, cache).cached:
            print(f"ingesting the dump into the store of {name} ...", flush=True)
        got = cached_store(system, variants[name], dump, cache)
        clone_store(got.path, store)
        report = got.report
        print(f"{name}: " + (f"cached, ingested in {report.seconds:g} s when it was made, so not ingested again" if got.hit
                             else f"ingested in {report.seconds:g} s, {report.bytes_written} bytes written, {len(report.refused)} refused, "
                                  f"{len(report.misdated)} misdated"), flush=True)
    return check_stores(folder, hashes)


def ensure_data(run: RunFolder, dump: Dump, workers: int = WORKERS) -> Dump:
    """The run's data, there to be read: its dump from the cache (`cached_dump`). As it was frozen, it is used; written again from the
    recipe when the temp folder lost it (Windows cleans it), it is checked to be the dump the stores were ingested from (the same files,
    sizes and hashes), else an error and the data written again is removed with the record left as it was, so that the next command finds
    the data gone and fails the same way. A file changed or added is an error naming it, not a rewrite. The run's manifest is made the
    one frozen with the data (a rewrite gives the files new times), and the ffmpeg of the agents with their own tools is put back."""
    if run.cache is None:
        raise ValueError(f"{run.name} was made before the cache (its run.json names none), so its data cannot be found or written again: "
                         "build a new run")
    got = cached_dump(run.scenario, run.recipe, run.temp, run.cache, workers)
    if (got.dump.key, got.dump.root) != (dump.key, run.data):
        raise ValueError(f"the dump of {run.name} is {got.dump.key} in {got.dump.root}, and its dump.json says {dump.key} in {run.data}")
    if not run.manifest.exists() or run.manifest.read_bytes() != (got.meta / STATS).read_bytes():
        shutil.copyfile(got.meta / STATS, run.manifest)
    run.ensure_bin()
    return got.dump


def name_of(scenario: Scenario, recipe: Mapping[str, object]) -> str:
    """The run folder's name: the minute it was made, the scenario, its scale and seed."""
    return f"{datetime.now():%Y%m%d-%H%M}-{scenario.code.lower()}-{recipe['scale']}-s{recipe['seed']}"


def say_dump(built: CachedDump) -> None:
    """What the cache did for the dump, with the time writing it takes (or took, for one that was there)."""
    seconds = f"{built.seconds:g} s"
    said = {BUILT_NOW: f"written in {seconds}", REWRITTEN: f"cached, its data written again (the first writing took {seconds})",
            FOUND: f"cached (writing it took {seconds}), not written again"}
    print(f"dump {built.dump.root.name}: {said[built.what]}", flush=True)


def build_run(scenario: Scenario, options: Mapping[str, object], out: Path, temp: Path, configs: Sequence[Config], *, timeout_s: float,
              cache: Path, probe: bool = True, workers: int = WORKERS, load: Callable[[str], System] = systems.load) -> RunFolder:
    """The run of the recipe `options` make: its dump (cached and frozen in `temp`, found when an earlier run of the recipe made it), the
    questions, the ffmpeg for the agents with their own tools, the store of each system of `configs` (found in the cache or ingested into
    it, and refused when misdated), the scenario's shortcut probe over the dump (found with it, or run; `probe=False` skips it), and last
    run.json with the probe's outcome and, as `build_s`, the seconds writing the dump took when it was made: a folder without run.json is
    a build that failed. A probe that fails leaves a complete run, flagged, for the caller to refuse."""
    loaded, descriptions = describe_configs(configs, load)  # a system whose extra is missing fails here, before anything is written
    check_temp(Path(temp))
    check_volume(Path(out), Path(temp), Path(cache))  # a store is copied into the run with hard links
    check_apart(Path(out), Path(temp))  # `RunFolder.deny` refuses the agents the folder a run is in: not the one their data is in
    commit_of()  # a build that cannot say what it was made by fails now, not after the dump is written
    recipe = scenario.recipe(**options)
    scenario.hashes(recipe)  # a recipe whose plan cannot be made (a wrong number of days, a download that is not there) fails here
    folder, temp, cache = Path(out) / name_of(scenario, recipe), Path(temp), Path(cache)
    if folder.exists():
        raise ValueError(f"{folder} exists: a run is one a minute for a scale and seed; wait, or remove it")
    tag = secrets.token_hex(5)  # the working folders are named by this, not by the dump: two runs of one dump share no slot
    built = cached_dump(scenario, recipe, temp, cache, workers)
    dump = built.dump
    say_dump(built)
    folder.mkdir(parents=True)
    shutil.copyfile(built.meta / STATS, folder / "data-manifest.json")
    shutil.copyfile(built.meta / QUESTIONS, folder / "questions.jsonl")
    write_dump(folder, dump)
    days = tuple(date.fromisoformat(d) for d in recipe["days"])
    started = RunFolder(folder, recipe["seed"], recipe["scale"], days, dump.root, temp, tag, timeout_s, scenario, tuple(configs), cache)
    started.ensure_bin()
    measured = ingest_stores(folder, dump, descriptions, loaded.__getitem__, cache=cache)
    if probe:
        outcome, seconds, kept = cached_probe(built.meta, cache, folder, lambda: run_probe(started, recipe, dump))
        if kept:
            print(f"shortcut probe: cached ({outcome['status']}; it took {seconds:g} s), not run again (remove {built.meta / PROBE} to run it again)",
                  flush=True)
    else:
        outcome = run_probe(started, recipe, dump, False)
    return RunFolder.write(folder, scenario, recipe, dump, temp, tag, timeout_s, configs, outcome, {**measured, "build_s": built.seconds},
                           cache=cache)
