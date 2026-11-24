"""Tests for mmsb/core/runfolder.py and paths.py: where a run's files and working slots are, the words an agent with its own tools may not
reach (`RunFolder.deny`), and the refusals when a run is opened again. The deny words are checked against explicit expected values over made-up
folders at the top of a drive (`SENTINEL`), so they read the same on Windows and elsewhere; what belongs to a scenario's own words is fed in by
a stub here and checked against the real scenarios in their own tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from test_question import BASE, file_of

from mmsb.core import fs, paths
from mmsb.core.agents.common import reached
from mmsb.core.config import parse_config
from mmsb.core.contract import Isolation
from mmsb.core.question import load_questions, write_questions
from mmsb.core.runfolder import COMMIT_KEY, JOBS_MAX, OFF_GPU_DENY, RunFolder, check_temp, package_root

TOP = Path("D:/") if os.name == "nt" else Path("/")  # made-up folders at the top of a drive: no run's folder is beside another's
SENTINEL = {"folder": TOP / "__RUN__", "out": TOP / "__OUT__", "root": TOP / "__ROOT__", "temp": TOP / "__TEMP__",
            "data": TOP / "__TEMP__" / "mmsb-key0123456-data", "cache": TOP / "__CACHE__", "tag": "TAG", "seed": 1, "scale": "pilot",
            "days": ("2026-11-01", "2026-11-02", "2026-11-03")}
PLAN, FILES = "a" * 64, "b" * 64  # the hashes the stub scenario says its recipe makes
RECIPE = {"scale": "pilot", "seed": 1}  # what the stub scenario is built from
NO_ISOLATION = Isolation((), (), (), {})
MM_ISOLATION = Isolation(("MMS_",), (".mms",), (), {"MMS_HOME": "no-mm-home"})
DEFAULTS = {"files": None, "mm": "d1-3B"}
# how a folder's name is told in a path or a command: before a slash, a quote, a space or another quote
RUNS_WORDS = ("/__out__/", '/__out__"', "/__out__ ", "/__out__'")
CACHE_WORDS = ("/__cache__/", '/__cache__"', "/__cache__ ", "/__cache__'")
TESTS_WORDS = ("/mmsb-temp/", '/mmsb-temp"', "/mmsb-temp ", "/mmsb-temp'")


def posix(text) -> str:
    return str(text).lower().replace("\\", "/")


class StubScenario:
    """A scenario that says what it is told to about its deny words, by the slots that are others'."""

    name, code, generator = "stub", "S", 3

    def __init__(self, words: dict[tuple[int, ...], tuple[str, ...]] | None = None) -> None:
        self.words = words or {}
        self.asked: list[tuple[tuple[int, ...], Path, str]] = []
        self.hashed: list = []  # the recipes `hashes` was asked for
        self.hash_of = (PLAN, FILES)

    def deny_words(self, others, temp, key):
        self.asked.append((tuple(others), temp, key))
        return self.words.get(tuple(others), ())

    def hashes(self, recipe):
        self.hashed.append(recipe)
        if isinstance(self.hash_of, Exception):
            raise self.hash_of
        return self.hash_of


@pytest.fixture
def sentinels(monkeypatch):
    monkeypatch.setattr(paths, "RUNS", SENTINEL["out"])
    monkeypatch.setattr(paths, "CACHE", SENTINEL["cache"])
    monkeypatch.setattr(paths, "ROOT", SENTINEL["root"])


def run_of(scenario, configs=(), **fields) -> RunFolder:
    """A run in the sentinel folders (`fields`: more of its fields, such as `cache` or `parent`)."""
    return RunFolder(SENTINEL["folder"], SENTINEL["seed"], SENTINEL["scale"], tuple(date.fromisoformat(d) for d in SENTINEL["days"]),
                     SENTINEL["data"], SENTINEL["temp"], SENTINEL["tag"], 900.0, scenario, tuple(configs), **fields)


def slot_words(run: RunFolder, ks) -> list[str]:
    """What the deny words say of the working slots numbered `ks`: the path, the folder's name, and two globs that match the name."""
    return [word for k in ks for word in (f"in:{posix(run.slot(k))}", f"in:mmsb-tag-w{k}", f"in:*-w{k}", f"in:?-w{k}")]


GLOBS = ["in:mmsb-tag-w*", "in:mmsb-tag-w?", "in:mmsb-tag-w[", "in:mmsb-*", "in:mmsb-?", "in:mmsb-["]  # every slot, and every folder of a run


# -- the words an agent with its own tools may not reach -----------------------------------------------------------------------

def test_the_deny_words_are_the_runs_folder_the_other_runs_the_checkout_the_setup_the_isolation_the_scenario_and_the_slots(sentinels):
    scenario = StubScenario({(0, 1, 2, 3): ("in:truth", "answers.json")})
    run = run_of(scenario)
    assert run.deny(None, MM_ISOLATION) == (
        posix(SENTINEL["folder"]),  # this run's folder
        posix(SENTINEL["out"]), posix(SENTINEL["cache"]),  # the runs and cache folders
        posix(SENTINEL["root"]), ".claude", ".codex",  # the checkout; the setup of the agents
        *RUNS_WORDS, *CACHE_WORDS, *TESTS_WORDS,  # the folders by name, for a relative path
        posix(paths.TESTS), ".mms",  # the tests' folder; what the isolation adds
        "in:truth", "answers.json",  # the scenario's
        *slot_words(run, range(JOBS_MAX)), *GLOBS)
    assert scenario.asked == [((0, 1, 2, 3), run.temp, "TAG")]


def test_the_slot_a_run_works_in_is_not_denied_to_it_and_the_scenario_is_asked_for_the_slots_that_are_others(sentinels):
    scenario = StubScenario({(1, 2, 3): ("in:truth",), (0, 1, 2, 3): ("in:everything",)})
    run = run_of(scenario)
    words = run.deny(run.slot(0), NO_ISOLATION)
    assert words[-len(slot_words(run, (1, 2, 3))) - len(GLOBS):] == (*slot_words(run, (1, 2, 3)), *GLOBS)
    assert "in:truth" in words and "in:everything" not in words
    assert scenario.asked == [((1, 2, 3), run.temp, "TAG")]
    assert run.slot(0) == SENTINEL["temp"] / "mmsb-TAG-w0"


def test_the_isolation_given_adds_its_words_and_the_agents_setup_is_denied_whatever_it_says(sentinels):
    scenario = StubScenario()
    with_mm = run_of(scenario).deny(None, MM_ISOLATION)
    without = run_of(scenario).deny(None, NO_ISOLATION)
    assert set(with_mm) - set(without) == {".mms"} and set(without) <= set(with_mm)
    assert ".claude" in without and ".codex" in without and ".mms" not in without


def test_every_word_is_lower_case_with_forward_slashes_and_only_the_other_slots_are_denied_by_name(sentinels):
    run = run_of(StubScenario())
    for k in range(JOBS_MAX):
        words = run.deny(run.slot(k), NO_ISOLATION)
        assert all(w == w.lower() and "\\" not in w for w in words)
        assert not set(slot_words(run, [k])) & set(words), "the own slot is not denied by path, name or glob of its number"
        for j in range(JOBS_MAX):
            if j != k:
                assert set(slot_words(run, [j])) <= set(words), "the others are"
        assert set(GLOBS) <= set(words), "and every slot by glob"
    assert set(slot_words(run, range(JOBS_MAX))) <= set(run.deny(None, NO_ISOLATION)), "with no own slot, all of them"


def test_the_cache_the_run_was_built_with_and_the_run_it_was_branched_from_are_denied_too(sentinels):
    cache, parent = TOP / "__OTHER_CACHE__", TOP / "__PARENT__"
    plain = run_of(StubScenario()).deny(None, NO_ISOLATION)
    words = run_of(StubScenario(), cache=cache, parent=parent).deny(None, NO_ISOLATION)
    assert {posix(cache), posix(parent), "/__other_cache__/", '/__other_cache__"'} <= set(words)
    assert not {posix(cache), posix(parent)} & set(plain)


def test_the_dump_is_denied_by_path_name_and_glob_only_when_the_agent_reads_another_folder(sentinels):
    run = run_of(StubScenario())
    dump = [f"in:{posix(run.data)}", f"in:{run.data.name}", "in:*-data", "in:?-data", "in:-data/", 'in:-data"', "in:-data ", "in:-data'"]
    assert not set(dump) & set(run.deny(None, NO_ISOLATION)) and not set(dump) & set(run.deny(None, NO_ISOLATION, reads=run.data))
    assert set(dump) <= set(run.deny(None, NO_ISOLATION, reads=SENTINEL["temp"] / "mmsb-TAG-cut"))


def test_an_agent_kept_off_the_gpu_may_not_name_its_decoder_or_its_list_of_users(sentinels):
    run = run_of(StubScenario())
    assert OFF_GPU_DENY == ("in:-hwaccel", "in:nvidia-smi")
    assert run.deny(None, NO_ISOLATION, off_gpu=True)[-len(OFF_GPU_DENY):] == OFF_GPU_DENY
    assert not set(OFF_GPU_DENY) & set(run.deny(None, NO_ISOLATION))


def test_the_repository_of_an_installed_package_is_denied_and_a_missing_package_is_not(sentinels, tmp_path, monkeypatch):
    repo = tmp_path / "Other Repo"
    (repo / "src" / "fakeloaded").mkdir(parents=True)
    (repo / "src" / "fakeloaded" / "__init__.py").write_text("raise SystemExit('imported')\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'fakeloaded'\n")
    monkeypatch.syspath_prepend(str(repo / "src"))
    assert package_root("fakeloaded") == repo and package_root("no_such_package_installed") is None
    words = run_of(StubScenario()).deny(None, Isolation((), (), ("fakeloaded", "no_such_package_installed"), {}))
    assert posix(repo) in words, "the repository, not the package folder"
    assert "fakeloaded" not in sys.modules, "found without importing it"


def test_a_package_installed_in_site_packages_denies_its_own_folder_and_not_the_project_above_it(tmp_path, monkeypatch):
    site = tmp_path / "project" / ".venv" / "Lib" / "site-packages"
    (site / "fakeinstalled").mkdir(parents=True)
    (site / "fakeinstalled" / "__init__.py").write_text("")
    (tmp_path / "project" / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    monkeypatch.syspath_prepend(str(site))
    assert package_root("fakeinstalled") == site / "fakeinstalled"


# -- the answers elsewhere on the disk ----------------------------------------------------------------------------------------

REAL_RUN = paths.RUNS / "20261201-0000-v7-pilot-s1"
# where an agent with its own tools could read the questions of another run, each the way it might name them
ELSEWHERE = {
    "a run of the tests": str(paths.TESTS / "pytest-coord" / "popen-gw0" / "pilot0" / "20261104-0314-pilot-s1" / "questions.jsonl"),
    "a run in the runs folder, by a bash path": f"cat {paths.RUNS.as_posix()}/20261104-0314-pilot-s1/results/claude/files/text-config-asof-1.json",
    "a run's report": f"type {paths.RUNS / '20261104-0314-pilot-s1' / 'report.md'}",
    "the cache": str(paths.CACHE / "dumps" / ("k" * 64) / "store" / "notes" / "a.md"),
}


def real_run(folder: Path = REAL_RUN, temp: Path | None = None) -> RunFolder:
    temp = temp or paths.TEMP
    return RunFolder(folder, 1, "pilot", (date(2026, 11, 1),), temp / "mmsb-abc-data", temp, "abc", 900.0, StubScenario(), ())


@pytest.mark.parametrize("path", ELSEWHERE.values(), ids=ELSEWHERE.keys())
def test_what_a_run_of_the_tests_or_another_run_or_the_cache_holds_is_denied_to_a_run_that_is_not_in_them(path):
    run = real_run()
    assert reached(path, run.deny(run.slot(0), MM_ISOLATION)) is not None


def test_the_temp_folder_a_run_uses_by_default_is_not_the_folder_the_tests_use():
    assert paths.TEMP == paths.HOME / "mmsb-data" and not paths.TEMP.is_relative_to(paths.TESTS) and paths.TESTS == paths.HOME / "mmsb-temp"
    words = real_run().deny(None, NO_ISOLATION)
    assert {posix(paths.RUNS), posix(paths.CACHE), posix(paths.TESTS), *TESTS_WORDS} <= set(words)
    assert posix(paths.TEMP) not in words, "the temp folder is not denied whole: the run's own dump and slots are in it"


def paths_in(home: str | None, tmp_path: Path) -> dict[str, str]:
    """The folders `mmsb.core.paths` names in a process that has `MMSB_HOME` set to `home` (unset when None)."""
    env = {k: v for k, v in os.environ.items() if k != "MMSB_HOME"} | ({"MMSB_HOME": home} if home else {})
    code = "import json; from mmsb.core import paths; print(json.dumps({k: str(v) for k, v in vars(paths).items() if k.isupper() and k != 'ROOT'}, default=str))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path, check=True)
    return json.loads(done.stdout)


def test_every_folder_is_under_the_mmsb_home_which_is_the_users_home_unless_it_is_set(tmp_path):
    home = tmp_path / "mine"
    got = paths_in(str(home), tmp_path)
    assert {k: Path(v) for k, v in got.items()} == {"HOME": home, "RUNS": home / "mmsb-runs", "TEMP": home / "mmsb-data",
                                                    "CACHE": home / "mmsb-cache", "TESTS": home / "mmsb-temp"}
    assert paths_in(None, tmp_path)["HOME"] == str(Path.home()) and paths_in("", tmp_path)["HOME"] == str(Path.home())


def test_a_run_in_a_folder_of_its_own_denies_that_folder_so_the_runs_beside_it_are_out_of_reach(tmp_path):
    mine = tmp_path / "my-runs"
    beside = str(mine / "20261130-0000-v7-pilot-s1" / "questions.jsonl")
    run = real_run(mine / "20261201-0000-v7-pilot-s1", tmp_path / "data")
    assert reached(beside, run.deny(run.slot(0), MM_ISOLATION)) is not None, "a sibling of a run with a folder of its own"
    assert reached(beside, real_run().deny(None, MM_ISOLATION)) is None, "the default runs folder is denied by name; this one is not in it"
    assert reached(str(paths.RUNS / "20261130-0000-v7-pilot-s1" / "questions.jsonl"), real_run().deny(None, MM_ISOLATION)) is not None


def test_a_relative_out_is_denied_whole_and_not_as_a_bare_word_that_ordinary_output_holds(tmp_path):
    run = real_run(Path("relruns") / "20261201-0000-v7-pilot-s1", tmp_path / "data")
    words = run.deny(None, NO_ISOLATION)
    assert posix(Path("relruns").resolve()) in words and "relruns" not in words
    assert reached("the camera relruns all night", words, output=True) is None
    assert reached(str(Path("relruns").resolve() / "other-run" / "questions.jsonl"), words) is not None


def test_an_out_that_holds_the_temp_folder_is_refused_and_one_that_does_not_is_not(tmp_path):
    paths.check_apart(paths.RUNS, paths.TEMP)
    paths.check_apart(tmp_path / "runs", tmp_path / "data")
    for out in (tmp_path, paths.HOME, Path(tmp_path.anchor)):
        with pytest.raises(ValueError, match=r"--out .* holds --temp .*Give --out a folder of its own"):
            paths.check_apart(out, out / "mmsb-data" if out != tmp_path else tmp_path / "data")
    with pytest.raises(ValueError, match="holds --temp"):
        paths.check_apart(tmp_path / "x", tmp_path / "x")


def test_a_run_straight_in_a_drive_root_does_not_deny_the_whole_drive(tmp_path):
    root = Path(tmp_path.anchor)
    words = real_run(root / "run", tmp_path / "data").deny(None, NO_ISOLATION)
    assert posix(root) not in words and posix(root / "run") in words


def test_a_checkout_whose_path_begins_like_the_data_folders_is_refused_before_a_run_and_not_met_as_an_escape_in_it(monkeypatch):
    """With every folder under one home, a clone called `mmsb` beside `mmsb-data` holds the data folder's path as a prefix."""
    monkeypatch.setattr(paths, "ROOT", paths.HOME / "mmsb")
    with pytest.raises(ValueError, match=r"mmsb-data.*holds .*mmsb.*refuses to reach: build with --temp elsewhere"):
        real_run().check_paths(NO_ISOLATION)


def test_the_tests_folder_is_denied_unless_the_run_is_in_it_so_the_tests_still_run():
    sibling = str(paths.TESTS / "pytest-coord" / "popen-gw0" / "pilot0" / "questions.jsonl")
    assert reached(sibling, real_run().deny(None, NO_ISOLATION)) in (posix(paths.TESTS), *TESTS_WORDS)
    in_tests = paths.TESTS / "pytest-x" / "run"
    for run in (real_run(in_tests, paths.TESTS / "pytest-x" / "temp"), real_run(REAL_RUN, paths.TESTS / "pytest-x" / "temp"),
                real_run(in_tests, paths.TEMP)):
        words = run.deny(None, NO_ISOLATION)
        assert not {posix(paths.TESTS), *TESTS_WORDS} & set(words) and reached(sibling, words) is None, (run.folder, run.temp)
        assert reached(str(run.folder), words) is not None, "its own folder is still denied"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def test_a_run_started_from_a_worktree_cannot_read_the_main_checkout(tmp_path, monkeypatch):
    main, tree = tmp_path / "main", tmp_path / "tree"
    main.mkdir()
    git("init", "-q", cwd=main)
    git("commit", "-q", "--allow-empty", "-m", "x", cwd=main)
    git("worktree", "add", "-q", str(tree), cwd=main)
    answers = main / "runs" / "20261104-0314-pilot-s1" / "questions.jsonl"
    run = real_run(tmp_path / "runs" / "run", tmp_path / "temp")
    monkeypatch.setattr(paths, "ROOT", tree)
    words = run.deny(None, NO_ISOLATION)
    assert reached(str(answers), words) == posix(main) and reached(str(tree / "mmsb" / "x.py"), words) == posix(tree)
    assert paths.checkouts() == (tree, main)
    monkeypatch.setattr(paths, "ROOT", main)
    assert paths.checkouts() == (main,) and posix(main) in run.deny(None, NO_ISOLATION), "from the main checkout itself, once"
    monkeypatch.setattr(paths, "ROOT", tmp_path)  # a copy that is not a checkout (installed): there is no other
    assert paths.checkouts() == (tmp_path,)


def test_a_temp_folder_that_holds_anything_but_the_folders_of_runs_is_refused_naming_it(tmp_path):
    temp = tmp_path / "temp"
    (temp / "mmsb-abc-data").mkdir(parents=True)
    run = real_run(tmp_path / "runs" / "run", temp)
    run.check_paths(NO_ISOLATION)
    check_temp(tmp_path / "not-made-yet")
    (temp / "pytest-coord").mkdir()
    with pytest.raises(ValueError, match=r"holds 'pytest-coord', which is not a folder of a run \(mmsb-\*\).*Move 'pytest-coord' out, or use --temp"):
        run.check_paths(NO_ISOLATION)
    (temp / "pytest-coord").rmdir()
    (temp / "notes.txt").write_text("a file is no folder of a run either", encoding="utf-8")
    with pytest.raises(ValueError, match="holds 'notes.txt'"):
        check_temp(temp)


# -- the layout ---------------------------------------------------------------------------------------------------------

def test_the_files_of_a_run_are_by_agent_configuration_and_question_and_its_slots_by_the_runs_tag(sentinels):
    files, mm = parse_config("files", DEFAULTS), parse_config("mm", DEFAULTS)
    run = run_of(StubScenario(), (files, mm))
    folder = SENTINEL["folder"]
    assert run.result_path("claude", files, "video-porch-parcel-1") == folder / "results" / "claude" / "files" / "video-porch-parcel-1.json"
    assert run.result_path("codex", mm, "audio-mic-alarm-3") == folder / "results" / "codex" / "mm-d1-3B" / "audio-mic-alarm-3.json"
    assert run.exec_dir("claude", mm, "text-note-code-1") == folder / "execs" / "claude" / "mm-d1-3B" / "text-note-code-1"
    assert run.slot(0) == SENTINEL["temp"] / "mmsb-TAG-w0" and run.slot(3) == SENTINEL["temp"] / "mmsb-TAG-w3"
    assert run.bin == SENTINEL["temp"] / "mmsb-TAG-bin" and run.manifest == folder / "data-manifest.json" and run.name == "__RUN__"
    assert run.bin_exe.name == ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def test_the_ffmpeg_is_copied_once_and_left_alone_after(tmp_path, monkeypatch):
    import imageio_ffmpeg

    source = tmp_path / "source-ffmpeg.exe"
    source.write_bytes(b"first")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(source))
    run = RunFolder(tmp_path / "run", 1, "tiny", (), tmp_path / "data", tmp_path / "temp", "t", 900.0, StubScenario(), ())
    run.ensure_bin()
    assert run.bin_exe.read_bytes() == b"first"
    source.write_bytes(b"second")
    run.ensure_bin()
    assert run.bin_exe.read_bytes() == b"first"


def test_the_three_folders_share_a_volume_and_the_check_says_when_they_do_not():
    paths.check_volume(paths.RUNS, paths.TEMP, paths.CACHE)
    if os.name == "nt":
        paths.check_volume(Path("D:/a"), Path("d:/b/c"))
        with pytest.raises(ValueError, match="share a volume"):
            paths.check_volume(Path("D:/a"), Path("C:/b"))
    else:  # (a folder not made yet is on its parent's file system, and /proc is one of its own)
        paths.check_volume(Path.home() / "not-made" / "yet", Path.home())
        if Path("/proc").is_dir():
            with pytest.raises(ValueError, match="share a volume"):
                paths.check_volume(Path.home(), Path("/proc"))
    assert (paths.ROOT / "pyproject.toml").is_file() and (paths.ROOT / "mmsb" / "core" / "paths.py").is_file()


# -- opening a run again ------------------------------------------------------------------------------------------------

def folder_of(tmp_path: Path) -> tuple[Path, dict]:
    """A run folder as `RunFolder.write` leaves it: questions.jsonl and a run.json with every key `open` reads."""
    folder = tmp_path / "20261104-0314-pilot-s1"
    write_questions(folder / "questions.jsonl", load_questions(file_of(tmp_path, BASE)))
    meta = {"scenario": "stub", "generator": 3, "recipe": RECIPE, "dump_key": "k" * 64, "seed": 1, "scale": "pilot",
            "days": list(SENTINEL["days"]), "data": str(tmp_path / "temp" / "mmsb-de178c4d7f-data"), "temp": str(tmp_path / "temp"),
            "tag": "de178c4d7f", "timeout_s": 900.0, "plan": PLAN, "files": FILES, "question_hash": fs.sha256_of(folder / "questions.jsonl"),
            "configs": [{"name": "files", "system": "files", "variant": None, "own_tools": True},
                        {"name": "mm/d1-3B", "system": "mm", "variant": "d1-3B", "own_tools": False}],
            "created": "2026-11-04T03:14:00", COMMIT_KEY: "abc1234"}
    fs.write_json(folder / "run.json", meta)
    return folder, meta


def test_a_run_is_opened_again_as_its_run_json_says(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    opened = RunFolder.open(folder, lambda name: scenario if name == "stub" else pytest.fail(name))
    assert opened == RunFolder(folder, 1, "pilot", (date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)),
                               tmp_path / "temp" / "mmsb-de178c4d7f-data", tmp_path / "temp", "de178c4d7f", 900.0, scenario,
                               (parse_config("files", DEFAULTS), parse_config("mm", DEFAULTS)))
    assert [q.qid for q in opened.questions()] == ["video-porch-parcel-1"]


def test_a_run_whose_run_json_names_a_cache_has_it(tmp_path):
    folder, meta = folder_of(tmp_path)
    fs.write_json(folder / "run.json", {**meta, "cache": str(tmp_path / "cache")})
    assert RunFolder.open(folder, lambda name: StubScenario()).cache == tmp_path / "cache"


def test_a_folder_whose_run_json_names_no_scenario_is_not_a_run_mmsb_built(tmp_path):
    folder, meta = folder_of(tmp_path)
    fs.write_json(folder / "run.json", {k: v for k, v in meta.items() if k != "scenario"})
    with pytest.raises(ValueError, match=r"is not a run mmsb built \(its run.json names no scenario\)"):
        RunFolder.open(folder, lambda name: StubScenario())


def test_a_run_whose_generator_changed_is_refused(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    scenario.generator = 4
    with pytest.raises(ValueError, match="made by generator 3 of stub, this is 4"):
        RunFolder.open(folder, lambda name: scenario)


def test_a_run_whose_questions_were_changed_after_the_build_is_refused(tmp_path):
    folder, _ = folder_of(tmp_path)
    path = folder / "questions.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("parcels", "parcel"), encoding="utf-8")
    with pytest.raises(ValueError, match="questions are not those its run.json recorded"):
        RunFolder.open(folder, lambda name: StubScenario())


def test_the_plan_and_the_files_the_recipe_makes_now_are_those_the_run_recorded(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    RunFolder.open(folder, lambda name: scenario)
    assert scenario.hashed == [RECIPE], "asked once, for the recipe run.json recorded"


def test_a_run_whose_plan_changed_with_no_generator_bump_is_refused(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    scenario.hash_of = ("0" * 64, FILES)  # the questions the recipe makes now are other than those it was built with
    with pytest.raises(ValueError, match=r"plan is not the one its run.json recorded: stub changed without a generator bump"):
        RunFolder.open(folder, lambda name: scenario)


def test_a_run_whose_files_changed_with_no_generator_bump_is_refused(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    scenario.hash_of = (PLAN, "0" * 64)  # a distractor changed, and no question did
    with pytest.raises(ValueError, match=r"files are not those its run.json recorded: stub changed without a generator bump"):
        RunFolder.open(folder, lambda name: scenario)


def test_a_scenario_that_cannot_make_its_plan_stops_the_run_from_opening_and_says_why(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    scenario.hash_of = FileNotFoundError("the download of the stub is not at D:/bench-data/stub")
    with pytest.raises(FileNotFoundError, match="the download of the stub is not at"):
        RunFolder.open(folder, lambda name: scenario)


def test_a_run_json_missing_a_key_is_not_guessed_at(tmp_path):
    folder, meta = folder_of(tmp_path)
    fs.write_json(folder / "run.json", {k: v for k, v in meta.items() if k != "timeout_s"})
    with pytest.raises(KeyError, match="timeout_s"):
        RunFolder.open(folder, lambda name: StubScenario())
