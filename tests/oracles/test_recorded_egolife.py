"""Oracle 2 for EgoLife: the report. The recorded EgoLife run's 150 agent results and 50 retrieval records, converted once to the new format
(convert_recorded.py, checked in under recorded/egolife-s1/converted), give through V8's `report` the figures and the Markdown that the old
code gave the recorded results: recorded/egolife-s1/report.json and report.md, written by the old code on 3 November 2026 and copied
unchanged. The Markdown is compared byte for byte; the figures through `legacy_view`, which renames what the new figures name otherwise and
nothing else. A difference is a fault of the port, not of the expectation. The recorded prompts (recorded/egolife-s1/prompts.json) are the
expected values of the V8 prompts, and the 150 recorded answers regraded through the questions are the recorded verdicts. About 3 s.

Unlike the pilot's, this oracle needs no download and holds in any time zone: the run is opened as a `RunFolder` directly, since
`RunFolder.open` makes the plan again to compare its hashes (test_v8_plan.py), and the report reads the checked-in questions."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import convert_recorded
import pytest
from v8_recorded import CONVERTED, RECORDED, REPORT, V8, download_skip, zone_skip

from mmmb import scenarios
from mmmb.core import fs, grade, report
from mmmb.core.config import Config
from mmmb.core.contract import McpServer, Served
from mmmb.core.runfolder import RunFolder
from mmmb.core.scenario import Access

NAME = "20261103-0257-egolife-s1"  # the run's folder name is in the report, so the converted run is copied to a folder of that name
RENAMES = {"control": "files", "tested": "mm", "tested_control": "mm_files", "control_blind": "files_blind", "tested_blind": "mm_blind",
           "mm/d1-3B": "d1-3B", "retrieval": "decider"}
CLOCK = "mm's clock reads a later date than now: give since and until as dates and times, not as ages or \"now\"."


def legacy_view(figures: dict) -> dict:
    """The new figures in the names of the recorded report.json: the sides are the arms (control `files`, tested `mm`) and their pairings,
    the configuration `mm/d1-3B` is the model `d1-3B`, the retrieval tier is the decider tier, and `frame`, what the Markdown says of the
    two sides, is not in the old figures."""
    def walk(x):
        if isinstance(x, dict):
            return {RENAMES.get(k, k): walk(v) for k, v in x.items() if k != "frame"}
        return [walk(v) for v in x] if isinstance(x, list) else x

    return walk(figures)


def differences(got, want, where: str = "") -> list[str]:
    """Where two JSON values differ, as paths, for a message that does not print 70 KB."""
    if isinstance(got, dict) and isinstance(want, dict):
        return [d for k in sorted(set(got) | set(want))
                for d in (differences(got[k], want[k], f"{where}/{k}") if k in got and k in want else [f"{where}/{k}: not in both"])]
    if isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        return [d for i, (a, b) in enumerate(zip(got, want)) for d in differences(a, b, f"{where}[{i}]")]
    return [] if got == want else [f"{where}: {got!r} != {want!r}"]


def open_run(folder: Path) -> RunFolder:
    meta = fs.read_json(folder / "run.json")
    return RunFolder(folder, meta["seed"], meta["scale"], tuple(date.fromisoformat(d) for d in meta["days"]), Path(meta["data"]),
                     Path(meta["temp"]), meta["tag"], meta["timeout_s"], scenarios.get(meta["scenario"]),
                     tuple(Config(**c) for c in meta["configs"]))


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> RunFolder:
    folder = tmp_path_factory.mktemp("egolife") / NAME
    shutil.copytree(CONVERTED, folder)
    return open_run(folder)


def test_the_converted_run_opens_as_the_run_it_was_with_its_three_configurations_and_50_questions(run):
    assert (run.name, run.seed, run.scale, [str(d) for d in run.days], run.timeout_s) == (
        NAME, 1, "egolife", ["2026-10-29", "2026-10-30", "2026-10-31"], 900.0)
    assert [(c.name, c.dir, c.own_tools) for c in run.configs] == [
        ("blind", "blind", False), ("files", "files", True), ("mm/d1-3B", "mm-d1-3B", False)]
    assert len(run.questions()) == 50 and run.scenario is V8


def test_the_figures_of_the_converted_run_are_the_recorded_report_json(run):
    got = legacy_view(V8.report(run)[0])
    assert not differences(got, REPORT), differences(got, REPORT)[:5]
    assert got == REPORT


def test_the_markdown_of_the_converted_run_is_the_recorded_report_md_byte_for_byte(run, tmp_path):
    """Written as a run writes it (`fs.write_text`, in text mode as the old code wrote report.md, so with Windows line ends)."""
    fs.write_text(tmp_path / "report.md", V8.report(run)[1])
    assert (tmp_path / "report.md").read_bytes() == (RECORDED / "report.md").read_bytes()


def test_regrading_the_150_recorded_answers_through_the_questions_gives_the_recorded_verdicts(run):
    questions = {q.qid: q for q in run.questions()}
    regraded = [(agent, folder, qid, r["problem"] is None and grade.grade(questions[qid].grade, r["answer"]), r["passed"])
                for agent, folders in report.results_of(run).items() for folder, results in folders.items() for qid, r in results.items()]
    assert len(regraded) == 150 and {x[0] for x in regraded} == {"codex"} and {x[1] for x in regraded} == {"blind", "files", "mm-d1-3B"}
    assert [x[:3] for x in regraded if x[3] != x[4]] == []


def test_the_recorded_prompts_are_the_ones_the_v8_prompt_makes_for_each_question_and_configuration(run):
    prompts = json.loads((RECORDED / "prompts.json").read_text(encoding="utf-8"))
    questions = {q.qid: q for q in run.questions()}
    assert sorted(prompts) == sorted(questions) and sum(len(arms) for arms in prompts.values()) == 150
    plain = Served(None, None, (), None, None, None, {})
    held = Served(McpServer("mm", "mm", (), {}, ()), "base", (), None, "mm", CLOCK, {})
    data = Path("<DATA>")  # the recorded prompts name the run's temp folder this way
    for qid, arms in prompts.items():
        assert V8.prompt(questions[qid], Access(False, None, plain, "line")) == arms["blind"], qid
        assert V8.prompt(questions[qid], Access(True, data, plain, "line")) == arms["files"], qid
        assert V8.prompt(questions[qid], Access(False, None, held, "line")) == arms["mm"], qid


def pairs_of(agent: str) -> dict[str, list[str]]:
    """The question ids of the recorded appendix by what the recorded verdicts say of the pair (control, tested): `a` both right, `b` the
    tested only, `c` the control only, `d` neither; of the video questions."""
    out: dict[str, list[str]] = {"a": [], "b": [], "c": [], "d": []}
    for row in REPORT["agents"][agent]["d1-3B"]["appendix"]:
        if row["type"] == "video":
            kind = {(True, True): "a", (False, True): "b", (True, False): "c", (False, False): "d"}[row["files"]["passed"], row["mm"]["passed"]]
            out[kind].append(row["qid"])
    return out


@pytest.mark.parametrize("pair, folder, passed, moves", [
    ("b", "mm-d1-3B", False, (-1, 0)),  # the tested side was right alone: now wrong with the control: b falls by one
    ("a", "mm-d1-3B", False, (0, +1)),  # both were right: the tested side is wrong now: c rises by one
    ("b", "files", True, (-1, 0))])  # the control was wrong: right now with the tested: b falls by one
def test_a_flipped_recorded_pass_fails_the_comparison_and_moves_b_or_c_by_one(run, tmp_path, pair, folder, passed, moves):
    qid = pairs_of("codex")[pair][0]
    copy = tmp_path / NAME
    shutil.copytree(CONVERTED, copy)
    file = copy / "results" / "codex" / folder / f"{qid}.json"
    record = fs.read_json(file)
    assert record["passed"] is not passed, "a flip changes the recorded pass"
    fs.write_json(file, {**record, "passed": passed})
    got = legacy_view(V8.report(replace(run, folder=copy))[0])

    assert got != REPORT and differences(got, REPORT), "the comparison with the recorded report fails"
    before, after = REPORT["agents"]["codex"]["d1-3B"]["video"]["mm_files"], got["agents"]["codex"]["d1-3B"]["video"]["mm_files"]
    assert (after["b"] - before["b"], after["c"] - before["c"]) == moves
    assert after["p"] != before["p"] or before["p"] == after["p"] == 1.0, "McNemar's p is told from the new b and c"


def test_the_converter_turns_a_retrieval_record_into_one_of_its_configuration_and_does_not_write_over_a_conversion(tmp_path):
    source = tmp_path / "run"
    folder = source / "results" / "decider" / "d1-3B"
    folder.mkdir(parents=True)
    fs.write_json(folder / "ego-004.json", {"qid": "ego-004", "model": "d1-3B", "found": True})
    fs.write_json(folder / "warm.json", {"model": "d1-3B", "seconds": 18.01, "at": "2026-11-03T10:33:53"})
    assert convert_recorded.convert_retrieval(source, tmp_path / "out") == 1
    done = tmp_path / "out" / "results" / "retrieval" / "mm-d1-3B"
    got = fs.read_json(done / "ego-004.json")
    assert list(got) == ["qid", "config", "system", "variant", "found"], "in the place of the model"
    assert (got["config"], got["system"], got["variant"]) == ("mm/d1-3B", "mm", "d1-3B")
    assert fs.read_json(done / "warm.json") == {"config": "mm/d1-3B", "seconds": 18.01, "at": "2026-11-03T10:33:53"}
    fs.write_json(folder / "ego-005.json", {"qid": "ego-005", "model": "d1-9B"})
    with pytest.raises(ValueError, match="holds the records of d1-9B"):
        convert_recorded.convert_retrieval(source, tmp_path / "again")
    with pytest.raises(FileExistsError, match="not regenerated"):
        convert_recorded.convert_egolife(convert_recorded.EGOLIFE, tmp_path)
    with pytest.raises(FileExistsError, match="not regenerated"):
        convert_recorded.capture_prompts(convert_recorded.EGOLIFE, RECORDED / "prompts.json")


def test_the_checked_in_conversion_is_what_the_converter_writes_from_the_recorded_run(tmp_path):
    if not convert_recorded.EGOLIFE.exists():
        pytest.skip(f"{convert_recorded.EGOLIFE} is not here, so the checked-in conversion is not compared with a fresh one")
    if reason := download_skip() or zone_skip():
        pytest.skip(reason)
    convert_recorded.convert_egolife(convert_recorded.EGOLIFE, tmp_path / "out")
    convert_recorded.capture_prompts(convert_recorded.EGOLIFE, tmp_path / "prompts.json")

    def files(root: Path) -> dict[str, bytes]:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    assert files(tmp_path / "out") == files(CONVERTED)
    assert (tmp_path / "prompts.json").read_bytes() == (RECORDED / "prompts.json").read_bytes()
    assert len(files(CONVERTED)) == 204  # 150 results, 50 retrieval records and the warm-up, the questions, systems.json and run.json


NEUTRAL = """
import sys
BLOCKED = {"bench", "haystack", "haystack_media", "shortcuts", "claude_run", "codex_run", "egolife", "grade", "multimem", "conftest", "day",
           "fake_decider"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is refused")


sys.meta_path.insert(0, Blocker())
import importlib, pkgutil
import mmmb.scenarios.v8_egolife as package
for module in pkgutil.iter_modules(package.__path__):
    importlib.import_module("mmmb.scenarios.v8_egolife." + module.name)
import json
from datetime import date
from pathlib import Path
from mmmb import scenarios
from mmmb.core import fs
from mmmb.core.config import Config
from mmmb.core.runfolder import RunFolder

folder = Path(sys.argv[1])
meta = fs.read_json(folder / "run.json")
run = RunFolder(folder, meta["seed"], meta["scale"], tuple(date.fromisoformat(d) for d in meta["days"]), Path(meta["data"]),
                Path(meta["temp"]), meta["tag"], meta["timeout_s"], scenarios.get(meta["scenario"]), tuple(Config(**c) for c in meta["configs"]))
fs.write_text(Path(sys.argv[2]), run.scenario.report(run)[1])
assert not BLOCKED & set(sys.modules), BLOCKED & set(sys.modules)
"""


def test_the_report_of_the_converted_run_is_the_same_with_the_old_modules_and_multimem_refused(run, tmp_path):
    done = subprocess.run([sys.executable, "-c", NEUTRAL, str(run.folder), str(tmp_path / "report.md")], capture_output=True, text=True,
                          cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "report.md").read_bytes() == (RECORDED / "report.md").read_bytes()
