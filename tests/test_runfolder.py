"""New tests for mmsb/core/runfolder.py and paths.py. The deny words are held to the goldens the old code wrote (U0:
tests/oracles/recorded/exec/deny-v7.json and deny-v8.json, from `Run.deny` and `EgoRun.deny` over sentinel folders); the intended
differences of plan 7.3 (difference 7) are asserted, not ignored. What belongs to a scenario's own words (V7's `haystack` words, V8's
folders of recordings) is fed in from the same goldens here and compared against the real scenarios in U2 and U6."""

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
from mmsb.core.runfolder import IN_ONLY, JOBS_MAX, RunFolder, check_temp, package_root

RECORDED = Path(__file__).parent / "oracles" / "recorded"
DENY = {name: json.loads((RECORDED / "exec" / f"deny-{name}.json").read_text(encoding="utf-8")) for name in ("v7", "v8")}
PILOT = json.loads((RECORDED / "pilot-s1" / "run.json").read_text(encoding="utf-8"))
RECIPE = {"scale": "pilot", "seed": 1}  # what the stub scenario is built from; its hashes are the recorded pilot's
SENTINEL = DENY["v7"]["inputs"]  # D:\__RUN__, D:\__TEMP__, D:\__OUT__, D:\__ROOT__, tag TAG
CACHE = r"D:\__CACHE__"  # the golden has no cache folder: the old code had none
NO_ISOLATION = Isolation((), (), (), {})
MM_ISOLATION = Isolation(("MULTIMEM_",), (".multimem",), (), {"MULTIMEM_HOME": "no-mm-home"})
DEFAULTS = {"files": None, "mm": "d1-3B"}
# the old DENY_WORDS' path part for the old out folder, which G1's F1 keeps, and the same rule for the sentinel runs and cache folders, for the
# results and the temp folder of the runs recorded before MMSB, and for the tests' folder
OLD_OUT_WORDS = ("/mm-bench/", '/mm-bench"', "/mm-bench ", "/mm-bench'")
NEW_RUNS_WORDS = ("/__out__/", '/__out__"', "/__out__ ", "/__out__'")
NEW_CACHE_WORDS = ("/__cache__/", '/__cache__"', "/__cache__ ", "/__cache__'")
EARLIER_TEMP_WORDS = ("/mm-bench-temp/", '/mm-bench-temp"', "/mm-bench-temp ", "/mm-bench-temp'")
TESTS_WORDS = ("d:/mmsb-temp", "/mmsb-temp/", '/mmsb-temp"', "/mmsb-temp ", "/mmsb-temp'")


class StubScenario:
    """A scenario that says what it is told to about its deny words, by the slots that are others'."""

    name, code, generator = "stub", "S", 3

    def __init__(self, words: dict[tuple[int, ...], tuple[str, ...]] | None = None) -> None:
        self.words = words or {}
        self.asked: list[tuple[tuple[int, ...], Path, str]] = []
        self.hashed: list = []  # the recipes `hashes` was asked for
        self.hash_of = (PILOT["plan"], PILOT["files"])

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
    monkeypatch.setattr(paths, "RUNS", Path(SENTINEL["out"]))
    monkeypatch.setattr(paths, "CACHE", Path(CACHE))
    monkeypatch.setattr(paths, "ROOT", Path(SENTINEL["root"]))


def run_of(scenario, configs=()) -> RunFolder:
    return RunFolder(Path(SENTINEL["folder"]), SENTINEL["seed"], SENTINEL["scale"], tuple(date.fromisoformat(d) for d in SENTINEL["days"]),
                     Path(SENTINEL["v7_data"]), Path(SENTINEL["temp"]), SENTINEL["tag"], 900.0, scenario, tuple(configs))


def between(words: list[str], after: str, before: str) -> tuple[str, ...]:
    return tuple(words[words.index(after) + 1:words.index(before)])


def posix(text) -> str:
    return str(text).lower().replace("\\", "/")


def check_against(golden: list[str], got: tuple[str, ...], scenario_words: tuple[str, ...]) -> None:
    """The deny words `got` are the golden's with the plan's differences, and nothing else."""
    assert len(set(got)) == len(got), "no word twice"
    assert set(golden) - set(got) == set(), "no word of the golden goes: the old out folder's stay (G1's F1)"
    assert set(got) - set(golden) == {posix(CACHE), *NEW_RUNS_WORDS, *NEW_CACHE_WORDS, *EARLIER_TEMP_WORDS, *TESTS_WORDS}, \
        "and only the cache folder's, the earlier runs' temp folder's, the tests' folder's and the new words come"
    assert set(OLD_OUT_WORDS) <= set(got)
    assert sorted(w for w in got if w in golden) == sorted(golden), "every word of the golden, once"
    assert got[:6] == (*golden[:2], posix(CACHE), *golden[2:5]), "the run, the runs folder, the cache (new), the repo, .claude, .codex first"
    slots = tuple(w for w in golden if w.startswith(IN_ONLY) and ("-w" in w or w in ("in:mmb-*", "in:mmb-?", "in:mmb-[")))
    assert got[-len(slots):] == slots, "the slot words last, in the old order"
    assert set(scenario_words) <= set(got)


def test_v7_deny_words_are_the_old_ones_but_for_the_new_folders_with_no_slot_and_with_slot_0(sentinels):
    v7 = DENY["v7"]["v7"]
    source = between(v7["none"], "/mm-bench'", "in:d:/__temp__/mmb-tag-w0")  # V7's `haystack` words: the scenario's, U2 checks them
    assert len(source) == 8 and all(w.startswith(IN_ONLY) for w in source)
    for own, key in ((None, "none"), (0, "slot0")):
        scenario = StubScenario({(0, 1, 2, 3): source, (1, 2, 3): source})
        run = run_of(scenario)
        got = run.deny(None if own is None else run.slot(own), MM_ISOLATION)
        check_against(v7[key], got, source)
        assert scenario.asked == [((0, 1, 2, 3) if own is None else (1, 2, 3), run.temp, "TAG")]
    assert run.slot(0) == Path(v7["slot0_path"])


def test_v8_deny_words_are_the_old_ones_but_for_the_new_folders_the_scenarios_words_come_from_it(sentinels):
    v8 = DENY["v8"]["v8"]
    for own, key, next_slot in ((None, "none", "in:d:/__temp__/mmb-tag-w0"), (0, "slot0", "in:d:/__temp__/mmb-tag-w1")):
        golden = v8[key]
        source = between(golden, "/mm-bench'", next_slot)  # bench.Run.deny's SOURCE_WORDS, which EgoRun.deny keeps
        extras = tuple(golden[golden.index("in:mmb-[") + 1:])  # V8's folders of recordings, after the last slot word
        assert extras[:3] == ("bench-data", "a1_jake", "in:egolife") and len(extras) > 20
        scenario = StubScenario({(0, 1, 2, 3): (*source, *extras), (1, 2, 3): (*source, *extras)})
        run = run_of(scenario)
        check_against(golden, run.deny(None if own is None else run.slot(own), MM_ISOLATION), (*source, *extras))


def test_the_isolation_given_adds_its_words_and_the_agents_setup_is_denied_whatever_it_says(sentinels):
    scenario = StubScenario()
    with_mm = run_of(scenario).deny(None, MM_ISOLATION)
    without = run_of(scenario).deny(None, NO_ISOLATION)
    assert set(with_mm) - set(without) == {".multimem"} and set(without) <= set(with_mm)
    assert ".claude" in without and ".codex" in without and ".multimem" not in without


def test_every_word_is_lower_case_with_forward_slashes_and_only_the_other_slots_are_denied_by_name(sentinels):
    run = run_of(StubScenario())
    for k in range(JOBS_MAX):
        words = run.deny(run.slot(k), NO_ISOLATION)
        assert all(w == w.lower() and "\\" not in w for w in words)
        own = [f"in:{posix(run.slot(k))}", f"in:mmb-tag-w{k}", f"in:*-w{k}", f"in:?-w{k}"]
        assert not set(own) & set(words), "the own slot is not denied by path, name or glob of its number"
        for j in range(JOBS_MAX):
            if j != k:
                assert {f"in:{posix(run.slot(j))}", f"in:mmb-tag-w{j}", f"in:*-w{j}", f"in:?-w{j}"} <= set(words), "the others are"
        assert {"in:mmb-tag-w*", "in:mmb-tag-w?", "in:mmb-tag-w[", "in:mmb-*", "in:mmb-?", "in:mmb-["} <= set(words), "and every slot by glob"
    assert all(f"in:mmb-tag-w{k}" in run.deny(None, NO_ISOLATION) for k in range(JOBS_MAX)), "with no own slot, all of them"


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


# -- the answers elsewhere on the disk (G1's F1) ---------------------------------------------------------------------------------

REAL_RUN = Path(r"D:\mmsb-runs\20261201-0000-v7-pilot-s1")
# where an agent with its own tools once could read a recorded or planted answer, each the way the review reached it
ELSEWHERE = {
    "a run of the tests": r"D:\mmsb-temp\pytest-coord\popen-gw0\pilot0\20261104-0314-pilot-s1\questions.jsonl",
    "a recorded run, by a bash path": "cat /d/mm-bench/20261104-0314-pilot-s1/results/claude/files/text-config-asof-1.json",
    "a recorded run's report": r"type D:\mm-bench\20261104-0314-pilot-s1\report.md",
    "the temp folder of a recorded run": r"D:\mm-bench-temp\mmb-de178c4d7f-data\notes\a.md",
}


def real_run(folder: Path = REAL_RUN, temp: Path | None = None) -> RunFolder:
    temp = temp or paths.TEMP
    return RunFolder(folder, 1, "pilot", (date(2026, 11, 1),), temp / "mmb-abc-data", temp, "abc", 900.0, StubScenario(), ())


@pytest.mark.parametrize("path", ELSEWHERE.values(), ids=ELSEWHERE.keys())
def test_what_a_run_of_the_tests_or_a_recorded_run_holds_is_denied_to_a_run_that_is_not_in_them(path):
    run = real_run()
    assert reached(path, run.deny(run.slot(0), MM_ISOLATION)) is not None


def test_the_temp_folder_a_run_uses_by_default_is_not_the_folder_the_tests_use_and_the_earlier_runs_words_keep_their_form():
    assert paths.TEMP == Path(r"D:\mmsb-data") and not paths.TEMP.is_relative_to(paths.TESTS) and paths.TESTS == Path(r"D:\mmsb-temp")
    assert [p.name for p in paths.EARLIER] == ["mm-bench", "mm-bench-temp"]
    words = real_run().deny(None, NO_ISOLATION)
    assert {*OLD_OUT_WORDS, *EARLIER_TEMP_WORDS, *TESTS_WORDS} <= set(words)


def test_the_tests_folder_is_denied_unless_the_run_is_in_it_so_the_tests_still_run():
    sibling = r"D:\mmsb-temp\pytest-coord\popen-gw0\pilot0\questions.jsonl"
    assert reached(sibling, real_run().deny(None, NO_ISOLATION)) in TESTS_WORDS
    in_tests = paths.TESTS / "pytest-x" / "run"
    for run in (real_run(in_tests, paths.TESTS / "pytest-x" / "temp"), real_run(REAL_RUN, paths.TESTS / "pytest-x" / "temp"),
                real_run(in_tests, paths.TEMP)):
        words = run.deny(None, NO_ISOLATION)
        assert not {*TESTS_WORDS} & set(words) and reached(sibling, words) is None, (run.folder, run.temp)
        assert reached(str(run.folder), words) is not None, "its own folder is still denied"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def test_a_run_started_from_a_worktree_cannot_read_the_recorded_answers_in_the_main_checkout(tmp_path, monkeypatch):
    main, tree = tmp_path / "main", tmp_path / "tree"
    main.mkdir()
    git("init", "-q", cwd=main)
    git("commit", "-q", "--allow-empty", "-m", "x", cwd=main)
    git("worktree", "add", "-q", str(tree), cwd=main)
    answers = main / "tests" / "oracles" / "recorded" / "pilot-s1" / "converted" / "questions.jsonl"
    run = real_run(tmp_path / "run", tmp_path / "temp")
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
    (temp / "mmb-abc-data").mkdir(parents=True)
    run = real_run(tmp_path / "run", temp)
    run.check_paths(NO_ISOLATION)
    check_temp(tmp_path / "not-made-yet")
    (temp / "pytest-coord").mkdir()
    with pytest.raises(ValueError, match=r"holds 'pytest-coord', which is not a folder of a run \(mmb-\*\).*Move 'pytest-coord' out, or use --temp"):
        run.check_paths(NO_ISOLATION)
    (temp / "pytest-coord").rmdir()
    (temp / "notes.txt").write_text("a file is no folder of a run either", encoding="utf-8")
    with pytest.raises(ValueError, match="holds 'notes.txt'"):
        check_temp(temp)


# -- the layout ---------------------------------------------------------------------------------------------------------

def test_the_folders_of_a_run_are_where_the_recorded_runs_and_the_old_code_put_them(sentinels):
    files, mm = parse_config("files", DEFAULTS), parse_config("mm", DEFAULTS)
    run = run_of(StubScenario(), (files, mm))
    execs = json.loads((RECORDED / "pilot-s1" / "execs.json").read_text(encoding="utf-8"))["execs"]
    by_dir = {"files": files, "mm-d1-3B": mm}
    assert len(execs) == 110
    for e in execs:
        assert run.result_path(e["agent"], by_dir[e["dir"]], e["qid"]) == run.folder / "results" / e["agent"] / e["dir"] / f"{e['qid']}.json"
        assert run.exec_dir(e["agent"], by_dir[e["dir"]], e["qid"]) == run.folder / "execs" / e["agent"] / e["dir"] / e["qid"]
    assert run.slot(0) == Path(DENY["v7"]["v7"]["slot0_path"]) and run.slot(3) == run.temp / "mmb-TAG-w3"
    assert run.bin == run.temp / "mmb-TAG-bin" and run.manifest == run.folder / "data-manifest.json" and run.name == "__RUN__"
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
    paths.check_volume(Path("D:/a"), Path("d:/b/c"))
    with pytest.raises(ValueError, match="share a volume"):
        paths.check_volume(Path("D:/a"), Path("C:/b"))
    assert (paths.ROOT / "pyproject.toml").is_file() and (paths.ROOT / "mmsb" / "core" / "paths.py").is_file()


# -- opening a run again ------------------------------------------------------------------------------------------------

def folder_of(tmp_path: Path) -> tuple[Path, dict]:
    """A run folder as the pilot's run.json says it, with the keys the new layout adds."""
    folder = tmp_path / "20261104-0314-pilot-s1"
    write_questions(folder / "questions.jsonl", load_questions(file_of(tmp_path, BASE)))
    meta = {**PILOT, "scenario": "stub", "recipe": RECIPE, "timeout_s": 900.0, "question_hash": fs.sha256_of(folder / "questions.jsonl"), "dump_key": "k" * 64,
            "generator": 3, "configs": [{"name": "files", "system": "files", "variant": None, "own_tools": True},
                                        {"name": "mm/d1-3B", "system": "mm", "variant": "d1-3B", "own_tools": False}]}
    fs.write_json(folder / "run.json", meta)
    return folder, meta


def test_a_run_is_opened_again_as_it_was_recorded(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    opened = RunFolder.open(folder, lambda name: scenario if name == "stub" else pytest.fail(name))
    assert opened == RunFolder(folder, 1, "pilot", (date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)),
                               Path(r"D:\mm-bench-temp\mmb-de178c4d7f-data"), Path(r"D:\mm-bench-temp"), "de178c4d7f", 900.0, scenario,
                               (parse_config("files", DEFAULTS), parse_config("mm", DEFAULTS)))
    assert [q.qid for q in opened.questions()] == ["video-porch-parcel-1"]


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
    scenario.hash_of = ("0" * 64, PILOT["files"])  # the questions the recipe makes now are other than those it was built with
    with pytest.raises(ValueError, match=r"plan is not the one its run.json recorded: stub changed without a generator bump"):
        RunFolder.open(folder, lambda name: scenario)


def test_a_run_whose_files_changed_with_no_generator_bump_is_refused(tmp_path):
    folder, _ = folder_of(tmp_path)
    scenario = StubScenario()
    scenario.hash_of = (PILOT["plan"], "0" * 64)  # a distractor changed, and no question did
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
