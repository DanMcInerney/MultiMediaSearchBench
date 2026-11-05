"""New tests for the cache as the commands use it (mmmb/core/build.py, mmmb/cli.py, mmmb/core/costing.py; plan 6.3): `mmmb build` twice says
`cached` and writes nothing, the stores are copied into each run, two runs of one dump keep their own slots, a lost or changed dump is dealt
with by every command that reads it, a misdated store still refuses a run, and `mmmb repeat --dry-run` says what is cached and makes nothing.
Over the scenario `toy` (V7's plan, three files) and the stand-in system `stub`, as tests/test_cli_build.py is. The cache itself is
tests/test_cache.py's, and V7 through mm tests/mm/test_cached_ingest.py's.

Every expected value is worked out here from independent sources: the times from the files' own stats, the folders from the arguments of the
command, the keys from `dump_key`, which tests/test_cache.py works out by hand."""

from __future__ import annotations

import os
import shutil
import types
from pathlib import Path

import cli_world
import pytest
from cache_world import stamps_of
from cli_world import DAYS, SCALE, SEED, build_args, built_run, world  # noqa: F401
from echo_agent import agents_of as echoing
from echo_agent import record_toy
from stub_system import Stub, tiny_questions

from mmmb import cli, scenarios
from mmmb.core import fs, paths
from mmmb.core.agents import claude
from mmmb.core.agents.common import reached
from mmmb.core.build import ensure_data, read_dump, store_of
from mmmb.core.cache import INGESTED, dump_folder, dump_meta, report_of
from mmmb.core.contract import IngestReport, Refusal
from mmmb.core.runfolder import RunFolder
from mmmb.core.scenario import dump_key
from mmmb.scenarios import v7_screen as v7

pytestmark = pytest.mark.xdist_group("cli")
FIRST = tiny_questions()[0].qid
RECIPE = v7.recipe(scale=SCALE, seed=SEED, days=DAYS.split(","))
KEY = dump_key("toy", RECIPE, v7.generator)


def again(world, seed: int = SEED, configs: str = "files,stub", *extra: str) -> RunFolder:
    """A second run of the same recipe: another runs folder, because a run's name is the minute it was made in."""
    other = world.out.with_name(f"runs-{len(list(world.out.parent.glob('runs-*')))}")
    assert world.main(*build_args(world, seed, configs, "--out", str(other), *extra)) == 0
    (folder,) = other.iterdir()
    return RunFolder.open(folder, scenarios.get)


def test_a_second_build_of_the_same_recipe_says_cached_ingests_and_probes_nothing_and_rewrites_nothing(world, capsys):
    first = built_run(world)
    out = capsys.readouterr().out
    assert f"dump mmb-{KEY[:10]}-data: written in " in out and "stub: ingested in 0 s" in out and "cached" not in out
    before = stamps_of(world.root / "temp") | {f"cache/{k}": v for k, v in stamps_of(world.cache).items()}
    second = again(world)
    out = capsys.readouterr().out
    assert f"dump mmb-{KEY[:10]}-data: cached (writing it took " in out and "), not written again" in out
    assert "stub: cached, ingested in 0 s when it was made, so not ingested again" in out and "files: cached" in out
    assert "shortcut probe: cached (passed; it took " in out
    assert len(world.system.kinds("ingest")) == 1 and len(cli_world.PROBED) == 1, "the system ingested once and the probe ran once, over two runs"
    assert before == stamps_of(world.root / "temp") | {f"cache/{k}": v for k, v in stamps_of(world.cache).items()}, "no file of the dump or the cache was rewritten"
    # what the second run holds is what the first made, with the times of the making
    assert (second.data, second.temp, second.cache) == (first.data, first.temp, first.cache) == (dump_folder(world.temp, KEY), world.temp, world.cache)
    assert second.folder != first.folder and fs.read_json(second.folder / "run.json")["dump_key"] == KEY
    for name in ("questions.jsonl", "data-manifest.json", "dump.json", "stores/stub/ingest.json", "stores/files/ingest.json", "shortcuts.json"):
        if (first.folder / name).exists():
            assert (second.folder / name).read_bytes() == (first.folder / name).read_bytes(), name
    assert fs.read_json(second.folder / "run.json")["build_s"] == fs.read_json(first.folder / "run.json")["build_s"] == fs.read_json(
        dump_meta(world.cache, KEY) / "built.json")["seconds"], "the seconds it took to write the dump, not to find it"
    assert fs.read_json(store_of(second.folder, "stub") / INGESTED)["seconds"] == 0.0


def test_two_runs_of_one_dump_share_its_files_and_the_stores_and_nothing_else(world):
    patch = world.patch
    patch.setattr(claude, "check_login", lambda: None)
    one, two = built_run(world), again(world)
    assert one.tag != two.tag and one.slot(0) != two.slot(0) and one.bin != two.bin and one.folder / "homes" != two.folder / "homes"
    assert one.slot(0).parent == two.slot(0).parent == world.temp, "the slots are named by the run's own random tag, not by the dump"
    for name in ("stub", "files"):
        assert os.stat(store_of(one.folder, name) / INGESTED).st_nlink == 3, "the cached store and the two runs' copies are one file"
    stubs = [Stub(), Stub()]
    for run, stub in zip((one, two), stubs):
        assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0
    works = [stub.execs[0].work for stub in stubs]
    assert works == [one.slot(0), two.slot(0)] and works[0] != works[1] and stubs[0].execs[0].data == stubs[1].execs[0].data == one.data, "one dump, two slots"


def test_a_dump_that_changed_refuses_a_build_and_a_run_naming_the_file_and_is_not_written_again(world, capsys):
    run = built_run(world)
    stub = Stub()
    (run.data / "notes" / "a.md").chmod(0o666)
    (run.data / "notes" / "a.md").write_text("vandal", encoding="utf-8")
    capsys.readouterr()
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 1
    assert "is not as it was frozen (changed: notes/a.md)" in capsys.readouterr().err and stub.execs == []
    assert world.main(*build_args(world, SEED, "files,stub", "--out", str(world.out.with_name("again")))) == 1
    assert "is not as it was frozen (changed: notes/a.md)" in capsys.readouterr().err and len(cli_world.PROBED) == 1
    assert not world.out.with_name("again").exists(), "no run folder of a build that was refused"
    assert (run.data / "notes" / "a.md").read_text(encoding="utf-8") == "vandal" and len(world.system.kinds("ingest")) == 1


def test_a_dump_the_temp_folder_lost_is_written_again_by_the_first_run_that_needs_it_and_the_others_manifests_follow(world, capsys):
    world.patch.setattr(claude, "check_login", lambda: None)
    one, two = built_run(world), again(world)
    fs.remove_tree(one.data)
    capsys.readouterr()
    for run in (one, two):
        stub = Stub()
        assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0
        assert stub.qids() == [FIRST] and fs.changed(run.data, run.manifest) is None, "the data is there, and as the run's manifest says"
    out = capsys.readouterr().out
    assert out.count("writing it again from the recipe") == 1, "the first command wrote it, the second found it"
    assert len(world.system.kinds("ingest")) == 1 and fs.read_json(one.manifest) == fs.read_json(two.manifest) == fs.stats_of(one.data)


def test_a_run_goes_on_over_a_cached_dump_and_asks_only_what_is_left(world):
    world.patch.setattr(claude, "check_login", lambda: None)
    run = built_run(world)
    first, rest = Stub(), Stub()
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(first)) == 0
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", agents=cli_world.agents_of(rest)) == 0
    assert first.qids() == [FIRST] and sorted(rest.qids()) == sorted(q.qid for q in tiny_questions() if q.qid != FIRST)


def test_the_store_of_a_cached_dump_that_was_misdated_refuses_a_second_run_though_nothing_is_ingested_for_it(world, capsys):
    misdated = (Refusal("phone/IMG_0001.jpg", "stored at 2026-10-30T09:00:00 from file, not at its own time 1.0 from its metadata"),)
    world.system.report = IngestReport(1.0, 0.0, 10, (), misdated)
    assert world.main(*build_args(world)) == 1
    assert "stub stored 1 files at a time other than their own" in capsys.readouterr().err
    other = world.out.with_name("again")
    assert world.main(*build_args(world, SEED, "files,stub", "--out", str(other))) == 1
    err = capsys.readouterr().err
    assert "stub stored 1 files at a time other than their own" in err and "phone/IMG_0001.jpg" in err
    assert len(world.system.kinds("ingest")) == 1, "the second build found the store, with its report, and ingested nothing"
    (folder,) = other.iterdir()
    assert report_of(store_of(folder, "stub")).misdated == misdated and not (folder / "run.json").exists()


def test_what_a_system_refused_and_measured_stays_in_the_cached_store_and_in_each_runs_copy_of_it(world):
    report = IngestReport(12.5, 0.0, 9, (Refusal("screen/s.mkv", "over the limit"),), (), {"decode_s": 0.024})
    world.system.report = report
    one, two = built_run(world), again(world)
    for run in (one, two):
        assert report_of(store_of(run.folder, "stub")) == report and fs.read_json(run.folder / "run.json")["decode_s"] == 0.024
    assert fs.read_json(store_of(two.folder, "stub") / INGESTED)["store_hash"] == "s" and len(world.system.kinds("ingest")) == 1


def test_a_store_a_run_lacks_is_found_in_the_cache_when_another_run_of_the_dump_ingested_it(world, capsys):
    one = built_run(world, SEED, "files,stub")
    two = again(world, SEED, "files")
    assert not store_of(two.folder, "stub").exists()
    capsys.readouterr()
    assert world.main("ingest", str(two.folder), "--configs", "stub") == 0
    assert "stub: cached, ingested in 0 s when it was made" in capsys.readouterr().out and len(world.system.kinds("ingest")) == 1
    assert (store_of(two.folder, "stub") / INGESTED).read_bytes() == (store_of(one.folder, "stub") / INGESTED).read_bytes()


def test_the_dumps_estimate_is_its_cached_build_time_and_a_dry_run_of_a_run_over_it_says_so(world, capsys):
    one = built_run(world)
    built = dump_meta(world.cache, KEY) / "built.json"
    fs.write_json(built, {**fs.read_json(built), "seconds": 600.0})  # (the toy's own time is too small to tell)
    two = again(world)
    assert fs.read_json(two.folder / "run.json")["build_s"] == 600.0 != fs.read_json(one.folder / "run.json")["build_s"], "the true time, not the time to find it"
    fs.remove_tree(two.data)
    capsys.readouterr()
    assert world.main("run", str(two.folder), "--dry-run", "--agents", "claude", "--configs", "files", agents=cli_world.agents_of(Stub())) == 0
    assert "gone from the temp folder; it is written again from the recipe, about 10 min" in capsys.readouterr().out


def test_the_cache_and_the_dumps_are_denied_to_an_agent_with_its_own_tools_in_whatever_folder_they_are_in(world):
    run = built_run(world)
    isolation = cli.Registry(world.registry).isolation()
    deny = run.deny(run.slot(0), isolation)
    for path in (dump_meta(world.cache, KEY) / "questions.jsonl", world.cache / "stores" / "stub-0123456789" / "ingest.json", world.cache / "locks" / "x.lock"):
        assert reached(str(path), deny) is not None, path
    assert reached(str(paths.CACHE / "dumps" / "k" / "questions.jsonl"), deny) is not None, "the default cache too"
    assert reached(str(run.temp / "mmb-0123456789-data" / "notes" / "a.md"), run.deny(run.slot(0), isolation, run.temp / "mmb-0123456789-d0")) is not None, \
        "another run's dump, when the agent reads a cut of its own"
    assert reached(f"cat {dump_folder(run.temp, KEY)}/notes/a.md", run.deny(run.slot(0), isolation, run.temp / "cut")) is not None


@pytest.mark.skipif(os.name != "nt", reason="drives are a Windows thing")
def test_a_cache_on_another_volume_than_the_runs_is_refused_before_anything_is_written(world, capsys):
    other = "C:/mmmb-cache-that-is-never-made"
    assert world.main(*build_args(world), "--cache", other) == 1
    assert "must share a volume" in capsys.readouterr().err and not Path(other).exists() and not world.out.exists() and not world.temp.exists()


def test_the_command_line_has_a_cache_for_build_and_repeat_that_is_the_default_folder_and_says_what_temp_is_now():
    parser = cli.parser_of()
    assert parser.parse_args(["build", "--scenario", "v7_screen", "--scale", "tiny"]).cache == paths.CACHE
    assert parser.parse_args(["repeat", "recorded"]).cache == paths.CACHE
    assert parser.parse_args(["build", "--scenario", "v7_screen", "--cache", "somewhere"]).cache == Path("somewhere")
    help_text = " ".join(parser._subparsers._group_actions[0].choices["build"].format_help().split())
    assert "each mmb-<key>-data, found again by the next build of the same recipe" in help_text and "(v8_egolife)" in help_text
    assert "on the volume of --out and --temp, because a store is copied into the run with hard links" in help_text


def test_a_run_made_before_the_cache_has_no_cache_and_cannot_be_asked_for_its_data(world):
    run = built_run(world)
    meta = {k: v for k, v in fs.read_json(run.folder / "run.json").items() if k != "cache"}
    fs.write_json(run.folder / "run.json", meta)
    old = RunFolder.open(run.folder, scenarios.get)
    assert old.cache is None
    with pytest.raises(ValueError, match="was made before the cache"):
        ensure_data(old, read_dump(old))


# -- mmmb repeat --------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    return record_toy(tmp_path_factory.mktemp("cache-repeat"))


@pytest.fixture
def w(world, recorded, tmp_path):  # noqa: F811
    shutil.copytree(recorded, tmp_path / "recorded")
    world.recorded = RunFolder.open(tmp_path / "recorded", scenarios.get)
    world.new = types.SimpleNamespace(out=tmp_path / "new-runs", temp=tmp_path / "new-temp", cache=tmp_path / "new-cache")
    world.patch.setattr(claude, "check_login", lambda: None)
    return world


def repeat_args(w, *extra: str) -> list[str]:
    return ["repeat", str(w.recorded.folder), "--out", str(w.new.out), "--temp", str(w.new.temp), "--cache", str(w.new.cache), "--answer-format", "line",
            *extra]


def test_a_repeat_dry_run_says_what_is_cached_and_what_would_be_built_and_makes_nothing_either_way(w, capsys):
    stub, key = Stub(), dump_key("toy", w.recorded.recipe, w.recorded.scenario.generator)
    assert w.main(*repeat_args(w, "--dry-run"), agents=echoing(stub)) == 0
    out = capsys.readouterr().out
    assert "  dump: written in the temp folder from the recipe (toy, tiny, seed 7, 2 days)" in out and "  store of stub: ingested from it, " in out
    assert not w.new.out.exists() and not w.new.temp.exists() and not w.new.cache.exists(), "nothing was made: not even the cache folder"
    # a repeat builds it; the dry run after it says cached, with the times of the making, and still makes nothing
    assert w.main(*repeat_args(w), agents=echoing(stub)) == 0
    built = capsys.readouterr().out
    assert f"dump mmb-{key[:10]}-data: written in " in built and "stub: ingested in " in built
    before = stamps_of(w.new.out.parent)
    assert w.main(*repeat_args(w, "--dry-run"), agents=echoing(stub)) == 0
    out = capsys.readouterr().out
    assert f"  dump mmb-{key[:10]}-data: cached (toy, tiny, seed 7, 2 days): writing it took " in out and "so it is not written again" in out
    assert "  store of stub: cached: ingesting it took 0 s, so it is not ingested again" in out
    assert stamps_of(w.new.out.parent) == before and len(w.system.kinds("ingest")) == 1


def test_a_second_repeat_finds_the_dump_and_the_stores_the_first_made_and_a_resume_goes_on_in_its_run(w, capsys):
    stub = Stub()
    assert w.main(*repeat_args(w), agents=echoing(stub)) == 0
    (first,) = w.new.out.iterdir()
    capsys.readouterr()
    w.patch.setattr("mmmb.core.build.name_of", lambda scenario, recipe: "second-run")
    assert w.main(*repeat_args(w), agents=echoing(Stub())) == 0
    out = capsys.readouterr().out
    assert "dump mmb-" in out and "cached (writing it took " in out and "stub: cached, ingested in 0 s when it was made" in out
    assert len(w.system.kinds("ingest")) == 1 and sorted(p.name for p in w.new.out.iterdir()) == sorted([first.name, "second-run"])
    lost = sorted((first / "results" / "codex" / "files").glob("*.json"))[:2]
    for file in lost:
        file.unlink()
    resumed = Stub()
    assert w.main(*repeat_args(w, "--into", str(first)), agents=echoing(resumed)) == 0
    assert sorted(resumed.qids()) == sorted(f.stem for f in lost), "a resume of a run on a cached dump asks only what is left"
