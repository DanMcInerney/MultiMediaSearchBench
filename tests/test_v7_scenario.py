"""New tests for the V7 scenario as the runner sees it (mmsb/scenarios/__init__.py and v7_screen/__init__.py, prompt.py): the recipe and
the time zone it records, the hashes, a week built and measured, the prompts as the pilot recorded them, the words an agent may not
reach, and the scenario run with Multimem and the old modules blocked.

Expected values are the recipe's definition, an independent read of the machine's time zone, the files a build left on disk, the pilot's
recorded prompts (tests/oracles/recorded/pilot-s1/execs.json), and the golden deny words of the old code (tests/oracles/recorded/exec)."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import imageio_ffmpeg
import pytest
from test_runfolder import DENY, MM_ISOLATION, SENTINEL, between, check_against, posix, run_of, sentinels  # noqa: F401

from mmsb import scenarios
from mmsb.core.contract import Dump, LogSource, McpServer, Served
from mmsb.core.question import load_questions, write_questions
from mmsb.core.runfolder import IN_ONLY
from mmsb.core.scenario import Access, ProbeOutcome, dump_key, files_hash, plan_hash
from mmsb.scenarios.v7_screen.manifest import entries_of
from mmsb.scenarios.v7_screen.questions import questions_of
from mmsb.scenarios.v7_screen.week import plan as make_plan

V7 = scenarios.get("v7_screen")
RECORDED = Path(__file__).parent / "oracles" / "recorded" / "pilot-s1"
DAYS = ["2026-10-30", "2026-10-31"]
PROTOCOL = ("recipe", "hashes", "build", "probe", "prompt", "view", "deny_words", "after_exec", "release")
PLAIN = Served(None, None, (), None, None, None, {})
MM = Served(McpServer("mm", "mm", (), {}, ()), "You answer the user's question from their memory, using the mm tools.", (), None, "mm", None, {})


def offset_of(day: date) -> str:
    """The machine's offset from UTC at local midnight on `day`, from the C library and not from datetime."""
    local = time.localtime(time.mktime((day.year, day.month, day.day, 0, 0, 0, 0, 0, -1)))
    minutes = local.tm_gmtoff // 60
    return f"{'+' if minutes >= 0 else '-'}{abs(minutes) // 60:02d}{abs(minutes) % 60:02d}"


def test_the_registry_has_the_scenario_by_its_name_and_the_scenario_has_the_names_of_a_scenario():
    assert scenarios.get("v7_screen") is sys.modules["mmsb.scenarios.v7_screen"]
    assert (V7.name, V7.code, V7.generator) == ("v7_screen", "V7", 6)
    assert all(callable(getattr(V7, name)) for name in PROTOCOL)
    with pytest.raises(ValueError, match="'nonesuch' is not a scenario: v7_screen"):
        scenarios.get("nonesuch")


def test_a_recipe_holds_the_scale_the_seed_the_days_and_the_utc_offsets_of_the_days_here():
    recipe = V7.recipe(scale="tiny", seed=3, days=[date(2026, 10, 30), "2026-10-31"])
    days = [date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 1)]  # (the offsets are at the midnights that start each day and end the last)
    assert recipe == {"scale": "tiny", "seed": 3, "days": ["2026-10-30", "2026-10-31"], "utc_offsets": [offset_of(d) for d in days]}
    assert json.loads(json.dumps(recipe)) == recipe
    ending = V7.recipe(scale="pilot", seed=1)  # by default the days of the scale that end yesterday
    assert ending["days"][-1] == (date.today() - timedelta(days=1)).isoformat() and len(ending["days"]) == 3
    with pytest.raises(ValueError, match="a scale is one of tiny, pilot, standard, not 'huge'"):
        V7.recipe(scale="huge", seed=1)


def test_a_recipe_made_in_another_time_zone_or_with_other_keys_is_refused():
    recipe = V7.recipe(scale="tiny", seed=3, days=DAYS)
    elsewhere = {**recipe, "utc_offsets": [("+0100" if o != "+0100" else "+0200") for o in recipe["utc_offsets"]]}
    with pytest.raises(ValueError, match=r"the recipe was made where the UTC offsets of its days are \['\+0[12]00'.*another week"):
        V7.plan_of(elsewhere)
    with pytest.raises(ValueError, match="has the keys days, scale, seed, utc_offsets, not days, scale, seed"):
        V7.plan_of({k: v for k, v in recipe.items() if k != "utc_offsets"})
    with pytest.raises(ValueError, match="a week has days"):
        V7.utc_offsets([])


def test_the_hashes_of_a_recipe_are_those_of_the_plan_it_makes_and_another_seed_has_others():
    recipe = V7.recipe(scale="tiny", seed=3, days=DAYS)
    plan = make_plan(3, "tiny", [date.fromisoformat(d) for d in DAYS])
    assert V7.hashes(recipe) == (plan_hash(plan.questions), files_hash([[f.path, f.source, f.t, f.name, f.kind, f.spec] for f in plan.files], plan.logs))
    assert V7.hashes(V7.recipe(scale="tiny", seed=4, days=DAYS))[0] != V7.hashes(recipe)[0]
    assert V7.plan_of(recipe) == plan


# -- the prompts ----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pilot_questions():
    plan = make_plan(1, "pilot", [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)])
    return {q.qid: q for q in questions_of(plan, entries_of(plan, lambda path: (1, 0, "0" * 64)))}


def stamp_masked(text: str) -> str:
    return re.sub(r"^It is now [A-Za-z]+ \d\d [A-Za-z]+ \d{4}, \d\d:\d\d\.", "It is now <the time>.", text)


def test_the_prompt_is_the_one_the_pilot_recorded_for_each_of_its_110_execs_but_for_the_time(pilot_questions):
    """Plan 7.3, difference 5: the pilot's prompts were stamped with the real time it asked; a question has its own `now`. Everything
    else, the week, the sources, how to reach them, the question and how to answer, is the recorded text, with the files and with mm."""
    execs = json.loads((RECORDED / "execs.json").read_text(encoding="utf-8"))["execs"]
    data = Path(json.loads((RECORDED / "run.json").read_text(encoding="utf-8"))["data"])
    asked = [e for e in execs if e["excluded"] is None]
    assert len(execs) == 110 and len(asked) == 108, "two execs of one question were overwritten by dry runs"
    for e in asked:
        access = Access(True, data, PLAIN, "line") if e["dir"] == "files" else Access(False, None, MM, "line")
        got = V7.prompt(pilot_questions[e["qid"]], access)
        assert stamp_masked(got) == stamp_masked(e["prompt"]), (e["agent"], e["dir"], e["qid"])
    assert V7.prompt(pilot_questions["text-note-code-1"], Access(False, None, MM, "line")).startswith("It is now Wednesday 04 November 2026, 12:00. mm has")


def test_a_system_and_the_files_together_get_both_sentences_and_one_way_to_answer(pilot_questions):
    question = pilot_questions["video-porch-parcel-1"]
    data = Path(r"D:\mmsb-temp\mmb-0123456789-data")
    got = V7.prompt(question, Access(True, data, MM, "line"))
    assert got == (
        "It is now Wednesday 04 November 2026, 12:00. mm has recorded these sources of mine from Sunday 1 November to Tuesday 3 November. "
        f"The files of these sources, from Sunday 1 November to Tuesday 3 November, are in {data}, one folder per source; each file is named by when "
        "it was recorded, and the logs are JSON lines with a ts field. Your working folder is yours to write in. ffmpeg is on PATH (there is no "
        "ffprobe). The sources are: screen (my screen recordings), phone (photos and short videos from my phone), porch (the front door camera's "
        "motion clips), mic (the hallway microphone), notes (my notes), config (the versions of app.toml), logs (the payments, web and auth "
        f"services' logs). {question.question} Answer from what mm holds or from these files. "
        "End with one line ANSWER: <answer>, or ANSWER: unknown when mm and these files do not hold it.")


def test_a_clock_note_ends_the_systems_sentence_and_a_prompt_with_no_way_to_the_recordings_or_an_unknown_format_is_refused(pilot_questions):
    question = pilot_questions["audio-mic-alarm-1"]
    late = replace(MM, clock_note="mm's clock is later than the date above.")
    assert "from Sunday 1 November to Tuesday 3 November. mm's clock is later than the date above. The sources are: screen" \
        in V7.prompt(question, Access(False, None, late, "line"))
    assert "clock" not in V7.prompt(question, Access(False, None, MM, "line"))
    with pytest.raises(ValueError, match="through a system's tools or the agent's own, not neither"):
        V7.prompt(question, Access(False, None, PLAIN, "line"))
    with pytest.raises(ValueError, match="the answer format is one of line, json, not 'yaml'"):
        V7.prompt(question, Access(False, None, MM, "yaml"))


# -- what an agent may not reach ------------------------------------------------------------------------------------------------

def test_the_deny_words_of_v7_are_the_old_eight_and_three_that_name_the_new_package_and_no_folder_of_a_run(sentinels):
    golden = DENY["v7"]["v7"]
    old = between(golden["none"], "/mm-bench'", "in:d:/__temp__/mmb-tag-w0")
    assert len(old) == 8
    new = tuple(IN_ONLY + word for word in ("v7_screen", "from mmsb", "import mmsb"))
    assert V7.deny_words([0, 1], Path("t"), "k") == (*old, *new) and V7.deny_words([], Path("t"), "k") == (*old, *new), "whatever the slots"
    for own, key in ((None, "none"), (0, "slot0")):
        run = run_of(V7)
        got = run.deny(None if own is None else run.slot(own), MM_ISOLATION)
        assert set(map(posix, new)) <= set(got) and not set(map(posix, new)) & set(golden[key])
        check_against(golden[key], tuple(w for w in got if w not in set(map(posix, new))), old)


def test_no_word_is_in_the_path_of_a_folder_a_run_makes_the_default_ones_included(sentinels):
    """`check_paths` refuses a run whose data, bin or slot a word is in: a bare `mmsb` would, with `D:\\mmsb-data`."""
    from mmsb.core import paths

    run = run_of(V7)
    words = [w.removeprefix(IN_ONLY) for w in V7.deny_words([], paths.TEMP, run.tag)]
    for folder in (run.data, run.bin, run.slot(0), paths.TEMP / "mmb-0123456789-data", paths.TEMP / "mmb-0123456789-bin", paths.TEMP / "mmb-0123456789-w0",
                   paths.RUNS / "20261104-1200-pilot-s1", paths.CACHE / "dumps" / ("k" * 64)):
        text = re.sub(r"[\\/]+", "/", json.dumps(str(folder)).lower())
        assert not [w for w in map(posix, words) if w in text], folder


def test_a_run_adds_nothing_to_a_result_makes_nothing_to_put_away_and_reads_the_dump_itself(tmp_path):
    dump = Dump("k", tmp_path / "data", "v7_screen", (), ())
    assert V7.after_exec(None, {"answer": "x"}, None, None) == {} and V7.release(tmp_path, "tag") is None
    assert V7.view(None, 2, dump, tmp_path, "tag", True) == tmp_path / "data" == V7.view(None, 0, dump, tmp_path, "tag", False)


# -- a week built -----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The tiny week built by the scenario, in a folder of its own (about 35 s)."""
    root = tmp_path_factory.mktemp("built")
    recipe = V7.recipe(scale="tiny", seed=7, days=DAYS)
    dump, questions = V7.build(recipe, root / "data", 3)
    return recipe, dump, questions, root


def test_a_week_is_built_and_its_dump_is_measured_from_the_files_left_on_disk(built):
    recipe, dump, questions, root = built
    plan = V7.plan_of(recipe)
    assert (dump.key, dump.root, dump.scenario) == (dump_key("v7_screen", recipe, 6), root / "data", "v7_screen")
    assert dump.logs == (LogSource("logs", "logs", "*.jsonl", "ts"),)
    assert [e.path for e in dump.entries] == [f.path for f in plan.files] + [f"logs/{s}.jsonl" for s in plan.logs]
    on_disk = {p.relative_to(dump.root).as_posix() for p in dump.root.rglob("*") if p.is_file()}
    assert on_disk == {e.path for e in dump.entries}, "the dump holds its files and no other"
    for e in dump.entries:
        path = dump.root / e.path
        assert (e.size, e.mtime_ns, e.sha256) == (path.stat().st_size, path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()), e.path
    by_kind = {f.kind: f for f in plan.files}
    first = {e.kind: e for e in dump.entries}
    assert {k: (first[k].source, first[k].media, first[k].own_time) for k in by_kind} == {
        "photo": ("phone", "image", True), "clip": ("phone", "video", True), "screen": ("screen", "video", False),
        "porch": ("porch", "video", False), "mic": ("mic", "audio", False), "note": ("notes", "text", False), "config": ("config", "text", False)}
    assert first["log"].media == "records" and first["config"].name == "app.toml" and first["note"].name is None


def test_the_questions_a_build_returns_are_the_plans_written_and_read_back(built, tmp_path):
    recipe, dump, questions, _ = built
    plan = V7.plan_of(recipe)
    assert questions == questions_of(plan, dump.entries) and len(questions) == 19
    write_questions(tmp_path / "questions.jsonl", questions)
    assert load_questions(tmp_path / "questions.jsonl") == questions


def test_a_week_is_built_in_a_folder_of_its_own(built, tmp_path):
    recipe, dump, _, _ = built
    with pytest.raises(ValueError, match="is not empty: a dump is written into a folder of its own"):
        V7.build(recipe, dump.root, 3)
    (tmp_path / "empty").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="is not empty"):
        V7.build(recipe, tmp_path / "other", 3)
    assert not (tmp_path / "other" / "screen").exists(), "and nothing was written there"


def test_the_probe_of_a_built_week_passes_and_reports_in_the_run_folder_and_never_in_the_data(built, tmp_path):
    recipe, dump, _, root = built
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    outcome = V7.probe(recipe, dump, root / "run", ffmpeg, 7)
    assert outcome == ProbeOutcome("passed", ())
    saved = json.loads((root / "run" / "shortcuts.json").read_text(encoding="utf-8"))
    assert (saved["failures"], saved["scale"], saved["seed"], saved["days"]) == ([], "tiny", 7, DAYS)
    assert (root / "run" / "shortcuts.md").read_text(encoding="utf-8").startswith("# Shortcut probe: tiny, seed 7")
    with pytest.raises(ValueError, match="never in the data folder"):
        V7.probe(recipe, dump, dump.root / "run", ffmpeg, 7)
    assert not (dump.root / "run").exists()
    broken = replace(dump, root=tmp_path / "gone")
    with pytest.raises(FileNotFoundError):  # a probe over a dump that is gone fails, it does not pass
        V7.probe(recipe, broken, tmp_path / "run", ffmpeg, 7)


# -- Multimem absent ---------------------------------------------------------------------------------------------------------------

OLD_MODULES = ("multimem", "conftest", "day", "fake_decider", "haystack", "haystack_media", "shortcuts", "grade", "bench", "claude_run",
               "codex_run", "egolife")
BLOCKED = f"""
import sys
class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {OLD_MODULES!r}:
            raise ImportError(f"{{name}} is blocked: V7 must not need it")
sys.meta_path.insert(0, Blocked())
import importlib, pkgutil
import mmsb.scenarios.v7_screen as package
names = [m.name for m in pkgutil.iter_modules(package.__path__)]
for name in names:
    importlib.import_module("mmsb.scenarios.v7_screen." + name)
import mmsb.core.probe
from mmsb import scenarios
v7 = scenarios.get("v7_screen")
print(len(names), *v7.hashes(v7.recipe(scale="tiny", seed=3, days={DAYS!r})))
"""


def test_v7_imports_and_plans_with_multimem_and_the_old_modules_blocked():
    """The blocker refuses `multimem` and the tests' own `conftest`, `day`, `fake_decider`, and the old root modules, which the editable
    install would otherwise let `import haystack` find without a word."""
    done = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    count, plan, files = done.stdout.split()
    assert int(count) >= 14 and (plan, files) == V7.hashes(V7.recipe(scale="tiny", seed=3, days=DAYS)), "and the hashes are those of this process"


def test_no_module_of_the_scenario_names_the_old_modules_or_multimem():
    root = Path(__file__).resolve().parents[1] / "mmsb"
    checked = [*(root / "scenarios").rglob("*.py"), root / "core" / "probe.py"]
    for path in checked:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not [n for n in names if n.split(".")[0] in OLD_MODULES], f"{path.name} imports {names}"
    assert len(checked) >= 20
