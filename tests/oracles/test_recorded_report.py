"""Oracle 2: the report. The recorded pilot's 110 results, converted once to the new format (convert_recorded.py, checked in under
recorded/pilot-s1/converted), give through the new `report.numbers` and `render.markdown` the figures and the Markdown that the old
code gave the recorded results: recorded/pilot-s1/report.json and report.md, written by the old code on 4 November 2026 and copied
unchanged. The Markdown is compared byte for byte; the figures through `legacy_view`, which renames what the new figures name
otherwise and nothing else. A difference is a fault of the port, not of the expectation. (Not called test_report.py, like the plan's
other oracle files, because tests/test_report.py has that name and pytest keeps one test module of a name.)

The report opens the run through `RunFolder.open`, which makes the V7 plan again, and the plan is local time: elsewhere than the
pilot's time zone this skips, and says so (see v7_recorded.py). About 3 s."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import convert_recorded
import pytest
from v7_recorded import needs_the_recorded_zone

from mmsb import scenarios
from mmsb.core import fs, grade, render, report
from mmsb.core.runfolder import RunFolder

needs_the_recorded_zone()

RECORDED = Path(__file__).parent / "recorded" / "pilot-s1"
CONVERTED = RECORDED / "converted"
NAME = "20261104-0314-pilot-s1"  # the run's folder name is in the report ("Runs pooled: ..."), so the converted run is opened under it
RENAMES = {"control": "files", "tested": "mm", "n_control": "n_files", "mm/d1-3B": "d1-3B", "retrieval": "decider"}


def legacy_view(figures: dict) -> dict:
    """The new figures in the names of the recorded report.json: the sides are the arms (control `files`, tested `mm`), the
    configuration `mm/d1-3B` is the model `d1-3B`, the retrieval tier is the decider tier, and `frame`, what the Markdown quotes of
    the run, is not in the old figures."""
    def walk(x):
        if isinstance(x, dict):
            return {RENAMES.get(k, k): walk(v) for k, v in x.items() if k != "frame"}
        return [walk(v) for v in x] if isinstance(x, list) else x

    return walk(figures)


def differences(got, want, where: str = "") -> list[str]:
    """Where two JSON values differ, as paths, for a message that does not print 50 KB."""
    if isinstance(got, dict) and isinstance(want, dict):
        return [d for k in sorted(set(got) | set(want))
                for d in (differences(got[k], want[k], f"{where}/{k}") if k in got and k in want else [f"{where}/{k}: not in both"])]
    if isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        return [d for i, (a, b) in enumerate(zip(got, want)) for d in differences(a, b, f"{where}[{i}]")]
    return [] if got == want else [f"{where}: {got!r} != {want!r}"]


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> RunFolder:
    folder = tmp_path_factory.mktemp("pilot") / NAME
    shutil.copytree(CONVERTED, folder)
    return RunFolder.open(folder, scenarios.get)


@pytest.fixture(scope="module")
def want() -> dict:
    return fs.read_json(RECORDED / "report.json")


def test_the_converted_pilot_opens_as_the_run_it_was_with_its_two_configurations(run):
    assert (run.name, run.seed, run.scale, [str(d) for d in run.days], run.timeout_s) == (
        NAME, 1, "pilot", ["2026-11-01", "2026-11-02", "2026-11-03"], 900.0)
    assert [(c.name, c.dir, c.own_tools) for c in run.configs] == [("files", "files", True), ("mm/d1-3B", "mm-d1-3B", False)]
    assert len(run.questions()) == 31


def test_the_figures_of_the_converted_pilot_are_the_recorded_report_json(run, want):
    got = legacy_view(report.numbers([run]))
    assert not differences(got, want), differences(got, want)[:5]
    assert got == want


def test_the_markdown_of_the_converted_pilot_is_the_recorded_report_md_byte_for_byte(run, tmp_path):
    """Written as a run writes it (`fs.write_text`, in text mode as the old code wrote report.md, so with Windows line ends)."""
    fs.write_text(tmp_path / "report.md", render.markdown(report.numbers([run])))
    assert (tmp_path / "report.md").read_bytes() == (RECORDED / "report.md").read_bytes()


def test_regrading_the_110_recorded_answers_through_the_questions_gives_the_recorded_verdicts(run):
    questions = {q.qid: q for q in run.questions()}
    regraded = [(agent, folder, qid, r["problem"] is None and grade.grade(questions[qid].grade, r["answer"]), r["passed"])
                for agent, folders in report.results_of(run).items() for folder, results in folders.items() for qid, r in results.items()]
    assert len(regraded) == 110
    assert [x[:3] for x in regraded if x[3] != x[4]] == []


def pairs_of(want: dict, agent: str) -> dict[str, list[str]]:
    """The question ids of the recorded appendix by what the recorded verdicts say of the pair (control, tested): `a` both right,
    `b` the tested only, `c` the control only, `d` neither; of the media questions both sides ran."""
    out: dict[str, list[str]] = {"a": [], "b": [], "c": [], "d": []}
    for row in want["appendix"][agent]:
        control, tested = row["files"], row["mm"].get("d1-3B")
        if control and tested and row["type"] in ("image", "video", "audio"):
            kind = {(True, True): "a", (False, True): "b", (True, False): "c", (False, False): "d"}[control["passed"], tested["passed"]]
            out[kind].append(row["qid"])
    return out


@pytest.mark.parametrize("agent, pair, side, passed, moves", [
    ("claude", "b", "mm-d1-3B", False, (-1, 0)),  # the tested side was right alone: now wrong with the control: b falls by one
    ("claude", "a", "mm-d1-3B", False, (0, +1)),  # both were right: the tested side is wrong now: c rises by one
    ("claude", "b", "files", True, (-1, 0))])  # the control was wrong: right now with the tested: b falls by one
def test_a_flipped_recorded_pass_fails_the_comparison_and_moves_b_or_c_by_one(run, want, tmp_path, agent, pair, side, passed, moves):
    qid = pairs_of(want, agent)[pair][0]
    folder = tmp_path / NAME
    shutil.copytree(CONVERTED, folder)
    file = folder / "results" / agent / side / f"{qid}.json"
    record = fs.read_json(file)
    assert record["passed"] is not passed, "a flip changes the recorded pass"
    fs.write_json(file, {**record, "passed": passed})
    got = legacy_view(report.numbers([replace(run, folder=folder)]))

    assert got != want and differences(got, want), "the comparison with the recorded report fails"
    before, after = want["agents"][agent]["d1-3B"]["media"], got["agents"][agent]["d1-3B"]["media"]
    assert (after["b"] - before["b"], after["c"] - before["c"]) == moves
    assert after["p"] != before["p"] or before["p"] == after["p"] == 1.0, "McNemar's p is told from the new b and c"


def test_the_converter_refuses_a_result_with_no_work_and_turns_arm_and_model_into_the_configuration():
    with pytest.raises(ValueError, match="no `work`"):
        convert_recorded.convert_result({"qid": "x", "arm": "files", "model": None})
    got = convert_recorded.convert_result({"qid": "x", "arm": "mm", "model": "d1-3B", "answer": "1", "work": {}})
    assert list(got) == ["qid", "config", "system", "variant", "answer", "work"], "in the place of the arm, and the model is gone"
    assert (got["config"], got["system"], got["variant"]) == ("mm/d1-3B", "mm", "d1-3B")
    files = convert_recorded.convert_result({"qid": "x", "arm": "files", "model": None, "work": {}})
    assert (files["config"], files["system"], files["variant"]) == ("files", "files", None)


def test_the_converter_does_not_write_over_a_conversion(tmp_path):
    with pytest.raises(FileExistsError, match="not regenerated"):
        convert_recorded.convert_pilot(convert_recorded.PILOT, tmp_path)


def test_the_checked_in_conversion_is_what_the_converter_writes_from_the_recorded_run(tmp_path):
    if not convert_recorded.PILOT.exists():
        pytest.skip(f"{convert_recorded.PILOT} is not here, so the checked-in conversion is not compared with a fresh one")
    convert_recorded.convert_pilot(convert_recorded.PILOT, tmp_path / "out")

    def files(root: Path) -> dict[str, bytes]:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    assert files(tmp_path / "out") == files(CONVERTED)
    assert len(files(CONVERTED)) == 113  # 110 results, the questions, systems.json and run.json



NEUTRAL = """
import sys
BLOCKED = {"bench", "haystack", "haystack_media", "shortcuts", "claude_run", "codex_run", "egolife", "grade", "multimem", "conftest", "day",
           "fake_decider"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is refused")


sys.meta_path.insert(0, Blocker())
from pathlib import Path
from mmsb import scenarios
from mmsb.core import fs, render, report
from mmsb.core.runfolder import RunFolder

run = RunFolder.open(Path(sys.argv[1]), scenarios.get)
fs.write_text(Path(sys.argv[2]), render.markdown(report.numbers([run])))
assert not BLOCKED & set(sys.modules), BLOCKED & set(sys.modules)
"""


def test_the_report_of_the_converted_pilot_is_the_same_with_the_old_modules_and_multimem_refused(run, tmp_path):
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", NEUTRAL, str(run.folder), str(tmp_path / "report.md")], capture_output=True, text=True,
                          cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "report.md").read_bytes() == (RECORDED / "report.md").read_bytes()
