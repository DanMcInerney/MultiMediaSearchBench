"""New tests for `mmmb build` and `mmmb ingest` (mmmb/cli.py, mmmb/core/build.py) and the run.json writer (`RunFolder.write`). Over the scenario
`toy` (V7's plan, a dump of three files) and the stand-in system `stub`, unless a test says it builds the real week. Ported from
tests/test_bench.py of Multimem at fe0ece5 (plan 7.4): `test_a_probe_that_fails_or_is_skipped_is_recorded_and_a_build_that_fails_it_exits_1`
and `test_the_probe_runs_over_the_written_and_loaded_week_with_the_files_arms_ffmpeg_and_is_recorded`; its `bench.build` is the command. The
other tests are new: the run.json this repository writes (plan 4.7), what a store's report does to a run (the finding of U7's notes: a file
stored at a time other than its own refuses the run), and what a system measured.

Every expected value is worked out here from independent sources: a hash is computed over the bytes written, the plan's hashes from V7's
own functions, the folders from the arguments of the command."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date

import cli_world
import pytest
from cli_world import CLEAN, DAYS, SCALE, SEED, build_args, built_run, world  # noqa: F401
from stub_system import FILES as FILES_CONFIG
from stub_system import STUB, Stub, StubSystem, tiny_questions

from mmmb import scenarios
from mmmb.core import fs, paths
from mmmb.core.build import ensure_data, read_dump, store_of
from mmmb.core.cache import INGESTED, report_of
from mmmb.core.config import Config
from mmmb.core.contract import IngestReport, Refusal
from mmmb.core.probe import probe_of
from mmmb.core.question import load_questions
from mmmb.core.runfolder import RunFolder, commit_of
from mmmb.core.scenario import ProbeOutcome, dump_key
from mmmb.scenarios import v7_screen as v7

pytestmark = pytest.mark.xdist_group("cli")
FIRST = tiny_questions()[0].qid
RECIPE = v7.recipe(scale=SCALE, seed=SEED, days=DAYS.split(","))
RUN_KEYS = {"scenario", "generator", "recipe", "dump_key", "seed", "scale", "days", "data", "temp", "cache", "tag", "timeout_s", "plan", "files",
            "question_hash", "configs", "probe", "created", "mmmb_commit", "build_s"}


def test_a_build_writes_the_dump_frozen_the_questions_the_stores_and_run_json_last(world, capsys):
    run = built_run(world)
    assert capsys.readouterr().out.splitlines()[-1] == str(run.folder), "the command prints the run's folder"
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    assert set(meta) == RUN_KEYS
    assert isinstance(meta["build_s"], float) and meta["build_s"] >= 0, "the seconds the build took, which `mmmb run --dry-run` quotes"
    assert re.fullmatch(r"\d{8}-\d{4}-v7-tiny-s7", run.name), "the minute, the scenario's code, the scale and the seed"
    assert (meta["scenario"], meta["generator"], meta["seed"], meta["scale"], meta["days"], meta["timeout_s"]) == (
        "toy", v7.generator, 7, "tiny", DAYS.split(","), 900.0)
    assert meta["recipe"] == RECIPE and re.fullmatch(r"[0-9a-f]{10}", meta["tag"])
    assert meta["data"] == str(world.temp / f"mmb-{meta['dump_key'][:10]}-data") and meta["temp"] == str(world.temp) and meta["cache"] == str(world.cache)
    assert (meta["plan"], meta["files"]) == v7.hashes(RECIPE), "the hashes of the plan the recipe makes: what `open` compares"
    assert meta["question_hash"] == hashlib.sha256((run.folder / "questions.jsonl").read_bytes()).hexdigest()
    assert meta["dump_key"] == dump_key("toy", RECIPE, v7.generator) and meta["mmmb_commit"] == commit_of()
    assert meta["configs"] == [{"name": "files", "system": "files", "variant": None, "own_tools": True},
                               {"name": "stub/all", "system": "stub", "variant": "all", "own_tools": False}]
    assert meta["probe"] == {"status": "passed", "failures": []} and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", meta["created"])
    # the data: the toy's three files, frozen, with a manifest of them; the ffmpeg for the agents' PATH
    assert sorted(p.relative_to(run.data).as_posix() for p in run.data.rglob("*") if p.is_file()) == sorted(cli_world.FILES)
    assert all(not os.access(p, os.W_OK) for p in run.data.rglob("*") if p.is_file()) and fs.changed(run.data, run.manifest) is None
    assert sorted(fs.read_json(run.manifest)) == sorted(cli_world.FILES) and run.bin_exe.read_bytes() == b"x"
    # the questions, and the dump the entries of which the commands after this one read
    assert [q.qid for q in load_questions(run.folder / "questions.jsonl")] == [q.qid for q in cli_world.build(RECIPE, run.temp / "again")[1]]
    dump = read_dump(run)
    assert (dump.key, dump.root, dump.scenario) == (meta["dump_key"], run.data, "toy")
    assert len(dump.entries) == len(v7.plan_of(RECIPE).files) + len(v7.plan_of(RECIPE).logs) and dump.logs[0].name == "logs"
    # each system ingested the dump once, into a store of its own, before the probe, and run.json came last
    assert [(e[0], e[2]) for e in world.system.events if e[0] == "ingest"] == [("ingest", world.system.events[0][2])]
    assert world.system.events[0][2] == dump.key and sorted(p.name for p in (run.folder / "stores").iterdir()) == ["files", "stub"]
    for name in ("files", "stub"):
        assert report_of(store_of(run.folder, name)) == CLEAN
    (call,) = cli_world.PROBED
    assert (call["out"], call["ffmpeg"], call["seed"], call["recipe"]) == (run.folder, run.bin_exe, 7, RECIPE)
    assert call["stores"] == ["files", "stub"] and call["data"] and not call["run_json"], "after the data and the stores, before run.json"


def test_a_probe_that_fails_or_is_skipped_is_recorded_and_a_build_that_fails_it_exits_1(world, capsys):
    failures = ("video-screen-error-1 (part, screen): 3 states in scope, 250 needed", "audio-mic-alarm-2 (week, mic): 2 sounds in scope, 30 needed")
    # passed (seed 1), failed (seed 2: the run is kept, complete and flagged), skipped (seed 3: the probe is not called)
    assert world.main(*build_args(world, 1)) == 0
    assert "skipped" not in capsys.readouterr().out and len(cli_world.PROBED) == 1
    cli_world.PROBE["outcome"] = ProbeOutcome("failed", failures)
    assert world.main(*build_args(world, 2)) == 1
    out = capsys.readouterr()
    kept = next(world.out.glob("*-s2"))
    assert str(kept) in out.out and (kept / "run.json").is_file() and probe_of(kept) == {"status": "failed", "failures": list(failures)}
    assert all(f"probe: {failure}" in out.err for failure in failures) and "the shortcut probe failed (2)" in out.err
    assert RunFolder.open(kept, scenarios.get).seed == 2, "the failed run is complete: the caller refuses it"
    assert world.main(*build_args(world, 3, "files,stub", "--no-probe")) == 0 and len(cli_world.PROBED) == 2
    assert "the shortcut probe was skipped (--no-probe)" in capsys.readouterr().out
    assert probe_of(next(world.out.glob("*-s3"))) == {"status": "skipped", "failures": []}
    assert probe_of(next(world.out.glob("*-s1"))) == {"status": "passed", "failures": []}


def test_a_build_that_cannot_be_made_is_an_error_before_anything_is_written_or_a_usage_error(world, capsys):
    assert world.main(*build_args(world, 1), "--scale", "huge") == 1  # (the later --scale wins)
    assert "a scale is one of" in capsys.readouterr().err and not world.out.exists()
    for bad, message in ((("--source", "x"), "toy takes no --source"), (("--per-age", "3"), "toy takes no --per-age")):
        with pytest.raises(SystemExit):
            world.main(*build_args(world), *bad)
        assert message in capsys.readouterr().err
    with pytest.raises(SystemExit):
        world.main("build", "--scenario", "toy", "--days", DAYS, "--out", str(world.out), "--temp", str(world.temp))
    assert "toy needs --scale" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        world.main("build", "--scenario", "nowhere", "--scale", SCALE)
    capsys.readouterr()
    assert world.main(*build_args(world, 1, "files,gold")) == 1 and "gold is not built yet" in capsys.readouterr().err
    assert world.main(*build_args(world, 1, "files,files")) == 1 and "share a results folder" in capsys.readouterr().err
    assert not world.out.exists() and not world.temp.exists(), "nothing was written"
    assert world.main(*build_args(world, 1)) == 0
    capsys.readouterr()
    world.patch.setattr("mmmb.core.build.name_of", lambda scenario, recipe: next(world.out.iterdir()).name)
    assert world.main(*build_args(world, 1)) == 1 and "exists: a run is one a minute" in capsys.readouterr().err


def test_a_system_whose_extra_is_missing_fails_the_build_before_it_writes_anything(world, capsys):
    (world.registry / "ghost").mkdir()
    (world.registry / "ghost" / "system.toml").write_text(
        '[system]\nname = "ghost"\nclass = "ghost_system:Ghost"\nextra = "ghost"\nlicense = "none"\nhorizon = "cut"\n', encoding="utf-8")
    assert world.main(*build_args(world, 1, "files,ghost")) == 1
    assert "ghost needs the ghost extra: uv sync --extra ghost" in capsys.readouterr().err
    assert not world.out.exists() and not world.temp.exists(), "no folder was made for a build that could not ingest"


def test_a_build_with_no_system_makes_the_stores_of_the_systems_it_names_only(world):
    run = built_run(world, 1, "files")
    assert [c.name for c in run.configs] == ["files"] and sorted(p.name for p in (run.folder / "stores").iterdir()) == ["files"]
    assert world.system.kinds("ingest") == [] and cli_world.PROBED[0]["stores"] == ["files"]


# -- what a store's report does to a run ----------------------------------------------------------------------------------------


MISDATED = (Refusal("phone/IMG_0001.jpg", "stored at 2026-10-30T09:00:00 from file, not at its own time 1.0 from its metadata"),
            Refusal("phone/IMG_0002.jpg", "stored at 2026-10-30T09:00:01 from file, not at its own time 2.0 from its metadata"))


def test_a_build_whose_store_holds_files_at_another_time_than_their_own_is_refused_and_names_them(world, capsys):
    world.system.report = IngestReport(1.0, 0.0, 10, (), MISDATED)
    assert world.main(*build_args(world, 1)) == 1
    err = capsys.readouterr().err
    assert "stub stored 2 files at a time other than their own" in err
    assert "phone/IMG_0001.jpg" in err and "phone/IMG_0002.jpg" in err and "stored at 2026-10-30T09:00:00 from file" in err
    (folder,) = world.out.iterdir()
    assert not (folder / "run.json").exists(), "a build that is refused is a build that failed: no run.json"
    assert cli_world.PROBED == [], "and nothing is probed over a store that cannot be trusted"
    assert report_of(store_of(folder, "stub")).misdated == MISDATED, "the report is kept: every command after this one refuses too"


def test_a_misdated_report_names_five_files_and_counts_the_rest(world, capsys):
    many = tuple(Refusal(f"phone/IMG_{n:04d}.jpg", "stored elsewhere") for n in range(8))
    world.system.report = IngestReport(1.0, 0.0, 10, (), many)
    assert world.main(*build_args(world, 1)) == 1
    err = capsys.readouterr().err
    assert "stored 8 files" in err and "IMG_0004.jpg" in err and "IMG_0005.jpg" not in err and "and 3 more" in err


def test_a_run_over_a_store_that_was_misdated_is_refused_whatever_made_it(world, capsys):
    run = built_run(world)
    stub = Stub()
    fs.write_json(store_of(run.folder, "stub") / INGESTED, {**fs.read_json(store_of(run.folder, "stub") / INGESTED),
                                                           "misdated": [{"path": r.path, "reason": r.reason} for r in MISDATED]})
    oracle = cli_world.use(world.patch, cli_world.OracleSystem(report=CLEAN))
    for argv in (["run", str(run.folder), "--agents", "claude"], ["run", str(run.folder), "--dry-run"],
                 ["retrieve", str(run.folder), "--dry-run", "--configs", "stub"], ["retrieve", str(run.folder), "--configs", "stub"]):
        assert world.main(*argv, agents=cli_world.agents_of(stub)) == 1, argv
        err = capsys.readouterr().err
        assert "stub stored 2 files at a time other than their own" in err and "phone/IMG_0002.jpg" in err, argv
    assert stub.execs == [] and not oracle.kinds("open") and not oracle.kinds("preview"), "refused before any session"
    # the configuration that does not use the store is not refused for it
    world.patch.setattr("mmmb.core.agents.claude.check_login", lambda: None)
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST,
                      agents=cli_world.agents_of(stub)) == 0 and stub.qids() == [FIRST]


def test_ingest_makes_the_store_of_a_configuration_added_later_and_refuses_it_when_misdated(world, capsys):
    run = built_run(world, 1, "files")
    assert world.main("ingest", str(run.folder), "--configs", "stub") == 0
    assert "stub: ingested in 0 s" in capsys.readouterr().out and len(world.system.kinds("ingest")) == 1
    assert [c.name for c in RunFolder.open(run.folder, scenarios.get).configs] == ["files", "stub/all"], "the run now has the configuration"
    assert world.main("ingest", str(run.folder), "--configs", "stub") == 0 and len(world.system.kinds("ingest")) == 1, "a store is made once"
    # a configuration ingested with a misdated store is recorded and refused
    world.system.report = IngestReport(1.0, 0.0, 10, (), MISDATED[:1])
    other = built_run(world, 2, "files")
    assert world.main("ingest", str(other.folder), "--configs", "stub") == 1
    assert "phone/IMG_0001.jpg" in capsys.readouterr().err and report_of(store_of(other.folder, "stub")).misdated == MISDATED[:1]
    assert world.main("ingest", str(other.folder), "--configs", "stub") == 1, "and refused again, not ingested again"
    assert len(world.system.kinds("ingest")) == 2


def test_a_store_whose_ingest_did_not_finish_is_an_error_and_a_missing_one_says_how_to_make_it(world, capsys):
    run = built_run(world, 1, "files")
    assert world.main("run", str(run.folder), "--configs", "files,stub", "--dry-run") == 1
    assert f"stub has no store in {run.folder}: mmmb ingest {run.name} --configs stub" in capsys.readouterr().err
    store_of(run.folder, "stub").mkdir(parents=True)
    assert world.main("ingest", str(run.folder), "--configs", "stub") == 1
    assert "holds an ingest that did not finish" in capsys.readouterr().err and world.system.kinds("ingest") == []


def test_what_a_system_measured_while_ingesting_is_kept_in_run_json_and_a_clash_is_refused(world, capsys):
    world.system.report = IngestReport(1.0, 0.0, 10, (), (), {"decode_s": 0.024})
    run = built_run(world)
    assert json.loads((run.folder / "run.json").read_text(encoding="utf-8"))["decode_s"] == 0.024
    later = built_run(world, 2, "files")
    assert "decode_s" not in json.loads((later.folder / "run.json").read_text(encoding="utf-8"))
    assert world.main("ingest", str(later.folder), "--configs", "stub") == 0
    assert json.loads((later.folder / "run.json").read_text(encoding="utf-8"))["decode_s"] == 0.024
    clash = built_run(world, 3, "files")
    fs.write_json(clash.folder / "run.json", {**fs.read_json(clash.folder / "run.json"), "decode_s": 0.5})
    assert world.main("ingest", str(clash.folder), "--configs", "stub") == 1
    assert "decode_s as other figures than the stores measured" in capsys.readouterr().err


# -- the data a run lost ------------------------------------------------------------------------------------------------------------


def test_a_lost_data_folder_is_written_again_from_the_recipe_and_frozen(world, capsys):
    run = built_run(world)
    dump, before = read_dump(run), fs.stats_of(run.data)
    assert ensure_data(run, dump) == dump and "writing it again" not in capsys.readouterr().out, "nothing is lost: nothing is written"
    fs.remove_tree(run.data)
    fs.remove_tree(run.bin)
    again = ensure_data(run, dump)
    assert "writing it again from the recipe" in capsys.readouterr().out and fs.changed(run.data, run.manifest) is None
    assert fs.stats_of(run.data).keys() == before.keys() and [(e.path, e.sha256) for e in again.entries] == [(e.path, e.sha256) for e in dump.entries]
    assert run.bin_exe.is_file() and all(not os.access(p, os.W_OK) for p in run.data.rglob("*") if p.is_file())
    fs.remove_tree(run.data / "screen")  # one file gone is the folder lost
    ensure_data(run, dump)
    assert "writing it again" in capsys.readouterr().out and (run.data / "screen" / "s.mkv").is_file()
    (run.data / "notes" / "a.md").chmod(stat.S_IWRITE)
    (run.data / "notes" / "a.md").write_text("vandal", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"is not as it was frozen \(changed: notes/a.md\)"):  # (a file that changed is not lost: the cache refuses it)
        ensure_data(run, dump)
    assert "writing it again" not in capsys.readouterr().out and fs.changed(run.data, run.manifest) == "changed: notes/a.md"
    assert (run.data / "notes" / "a.md").read_text(encoding="utf-8") == "vandal", "and it is not written over: what changed it may be looked at"


def test_data_written_again_that_is_not_the_dump_the_stores_were_made_from_is_an_error_every_command_gives_and_not_kept(world):
    run = built_run(world)
    dump = read_dump(run)
    manifest = run.manifest.read_bytes()
    cli_world.CONTENT[:] = ["v2"]  # the generator, or a library, changed since
    fs.remove_tree(run.data)
    for command in ("first", "second", "third"):  # (G1's F2: the second used to go on, with the other bytes frozen in)
        with pytest.raises(RuntimeError, match=r"is not the dump .* was built with \(\d+ files differ"):
            ensure_data(run, dump)
        assert run.manifest.read_bytes() == manifest and not run.data.exists(), f"the {command} command kept nothing of the other bytes"
    cli_world.CONTENT[:] = ["v1"]  # the generator is as it was: the data is written again, and is the dump
    assert ensure_data(run, dump) == dump and fs.changed(run.data, run.manifest) is None


def test_the_run_command_writes_a_lost_data_folder_again_before_it_runs(world, capsys):
    run = built_run(world)
    stub = Stub()
    world.patch.setattr("mmmb.core.agents.claude.check_login", lambda: None)
    fs.remove_tree(run.data)  # the temp folder was cleaned
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0
    assert "writing it again from the recipe" in capsys.readouterr().out and (run.data / "notes" / "a.md").is_file() and stub.qids() == [FIRST]
    assert fs.changed(run.data, run.manifest) is None
    assert json.loads((run.folder / "results" / "claude" / "files" / f"{FIRST}.json").read_text(encoding="utf-8"))["qid"] == FIRST


def test_the_writer_refuses_a_questions_file_that_is_not_there_and_open_a_hash_that_is_wrong(world):
    run = built_run(world)
    fs.write_json(run.folder / "run.json", {**fs.read_json(run.folder / "run.json"), "question_hash": "0" * 64})
    with pytest.raises(ValueError, match="questions are not those its run.json recorded"):
        RunFolder.open(run.folder, scenarios.get)
    (run.folder / "questions.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        RunFolder.write(run.folder, scenarios.get("toy"), RECIPE, read_dump(run), run.temp, run.tag, 900.0, run.configs, {"status": "none"},
                         cache=run.cache)


def test_a_run_keeps_a_new_timeout_and_the_configurations_it_is_given_in_its_run_json(world):
    run = built_run(world, 1, "files")
    assert run.with_timeout(1200.0).timeout_s == 1200.0 and RunFolder.open(run.folder, scenarios.get).timeout_s == 1200.0
    assert run.timeout_s == 900.0, "the run it was called on is unchanged"
    more = run.with_configs([FILES_CONFIG, STUB, Config("stub/all", "stub", "all", False)])
    assert [c.name for c in more.configs] == ["files", "stub/all"] and [c.name for c in RunFolder.open(run.folder, scenarios.get).configs] == [
        "files", "stub/all"]
    assert more.with_configs([STUB]) is more, "a configuration the run has is not added again"
    with pytest.raises(ValueError, match="as other configurations"):
        more.with_configs([Config("stub/all", "stub", "all", True)])
    assert date.fromisoformat(DAYS.split(",")[0]) == run.days[0]


# -- G1's repairs ---------------------------------------------------------------------------------------------------------------------


def test_a_store_keeps_the_hash_of_the_system_that_made_it_and_a_run_over_a_store_of_another_version_is_refused(world, capsys):
    run = built_run(world)
    assert fs.read_json(store_of(run.folder, "stub") / INGESTED)["store_hash"] == "s"
    assert fs.read_json(store_of(run.folder, "files") / INGESTED)["store_hash"] == "none"
    stub = Stub()
    oracle = cli_world.use(world.patch, cli_world.OracleSystem(report=CLEAN, store_hash="t"))  # the system moved on: its store is another
    for argv in (["run", str(run.folder), "--agents", "claude"], ["run", str(run.folder), "--dry-run"], ["ingest", str(run.folder), "--configs", "stub"],
                 ["retrieve", str(run.folder), "--dry-run", "--configs", "stub"], ["retrieve", str(run.folder), "--configs", "stub"]):
        assert world.main(*argv, agents=cli_world.agents_of(stub)) == 1, argv
        err = capsys.readouterr().err
        assert f"the store of stub in {run.name} was made by stub with store_hash s, and it is t now" in err, argv
        assert "Build a new run" in err and f"remove {store_of(run.folder, 'stub')} and ingest again" in err, argv
    assert stub.execs == [] and not oracle.kinds("open") and not oracle.kinds("preview") and not oracle.kinds("ingest"), "refused before any session"
    # the configuration that does not use the store is not refused for it, and the system as it was is not refused at all
    world.patch.setattr("mmmb.core.agents.claude.check_login", lambda: None)
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0
    cli_world.use(world.patch, StubSystem(report=CLEAN))
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "stub", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0


def test_a_variant_that_needs_another_store_than_the_others_of_its_system_is_refused():
    from mmmb.core.build import store_hashes

    @dataclass
    class Variants(StubSystem):
        def describe(self, variant):
            return dataclasses.replace(StubSystem.describe(self, variant), store_hash=f"s-{variant}")

    system = Variants()
    a, b = Config("stub/all", "stub", "all", False), Config("stub/narrow", "stub", "narrow", False)
    assert store_hashes({a: system.describe("all")}) == {"stub": "s-all"} and store_hashes({FILES_CONFIG: None}) == {}
    with pytest.raises(ValueError, match="the variants of stub named together need different stores: stub/narrow's"):
        store_hashes({a: system.describe("all"), b: system.describe("narrow")})


def test_ingest_writes_the_data_again_before_it_reads_it_when_the_temp_folder_lost_it(world, capsys):
    @dataclass
    class Reading(StubSystem):
        """A system that notes, when it ingests, whether the dump it is given is there to be read."""

        present: tuple = ()

        def ingest(self, dump, store):
            self.present = (dump.root.is_dir(), (dump.root / "notes" / "a.md").is_file())
            return super().ingest(dump, store)

    run = built_run(world, 1, "files")
    fs.remove_tree(run.data)  # Windows cleaned the temp folder
    system = cli_world.use(world.patch, Reading(report=CLEAN))
    assert world.main("ingest", str(run.folder), "--configs", "stub") == 0
    assert system.present == (True, True), "the dump was written again before the system read it"
    assert "writing it again from the recipe" in capsys.readouterr().out and fs.changed(run.data, run.manifest) is None


def test_a_build_into_a_temp_folder_that_holds_more_than_runs_is_refused_before_anything_is_written(world, capsys):
    world.temp.mkdir()
    (world.temp / "pytest-coord").mkdir()
    assert world.main(*build_args(world, 1)) == 1
    err = capsys.readouterr().err
    assert f"{world.temp} holds 'pytest-coord', which is not a folder of a run (mmb-*)" in err and "Move 'pytest-coord' out, or use --temp" in err
    assert not world.out.exists() and sorted(p.name for p in world.temp.iterdir()) == ["pytest-coord"]


def test_outside_a_git_checkout_a_build_records_the_installed_version_and_one_that_cannot_say_what_it_is_made_by_fails_first(world, capsys, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", world.root)  # no .git here
    run = built_run(world, 1, "files")
    assert fs.read_json(run.folder / "run.json")["mmmb_commit"] == importlib.metadata.version("mmmb") == commit_of()

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert world.main(*build_args(world, 2)) == 1 and "error: " in capsys.readouterr().err
    assert sorted(p.name for p in world.out.iterdir()) == [run.name], "no folder of the failed build, and no data written for it"
    assert sorted(p.name for p in world.temp.iterdir() if p.name.endswith("-data")) == [run.data.name]
