"""New tests for the answer format as a run keeps it (mmsb/core/runner.py `answer_format_of`, `mmsb run --answer-format`; DESIGN 6.2, plan 5.4):
a run not asked for a format is asked in json and keeps it in run.json; one asked for `line` gets the recorded prompt and no field but
the format in its results; a run that has results is refused the other format whatever the command; a run recorded before the format was kept
is a `line` run. Expected values are the plan's rules and the prompts' words, over the toy run `mmsb build` makes (tests/test_cli.py)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from cli_world import world  # noqa: F401
from stub_system import FILES, Stub
from test_cli import QUESTIONS, argv, built, result, w  # noqa: F401  (a copy of the toy run `mmsb build` makes, with no results)

from mmsb import scenarios
from mmsb.core import fs
from mmsb.core.question import write_questions
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent, answer_format_of


# -- the command line: a run keeps its format ----------------------------------------------------------------------------------------------

def run_args(w, *rest):
    return argv(w, "--agents", "claude", "--configs", "files", "--only", QUESTIONS[0].qid, *rest)


def held(run: RunFolder):
    return fs.read_json(run.folder / "run.json").get("answer_format")


def test_a_run_not_asked_for_a_format_is_asked_in_json_and_keeps_it(w, capsys):
    stub = Stub(by={("files", QUESTIONS[0].qid): {"final": json.dumps({"answer": "unknown", "evidence": []})}})
    assert held(w.run) is None and answer_format_of(w.run, None) == "json"
    assert w.main(*run_args(w), agents=lambda: {"claude": Agent(stub)}) == 0
    assert held(w.run) == "json", "kept in run.json"
    assert "End with one JSON object" in stub.execs[0].prompt
    result = json.loads((w.run.folder / "results" / "claude" / "files" / f"{QUESTIONS[0].qid}.json").read_text(encoding="utf-8"))
    assert result["answer_format"] == "json" and result["answer"] == "unknown" and result["grounded"] in ("grounded", "wrong")


def test_a_run_asked_for_line_has_the_recorded_prompt_and_no_new_field_but_the_format(w):
    stub = Stub()
    assert w.main(*run_args(w, "--answer-format", "line"), agents=lambda: {"claude": Agent(stub)}) == 0
    assert held(w.run) == "line" and "End with one line ANSWER:" in stub.execs[0].prompt and "End with one JSON" not in stub.execs[0].prompt
    result = json.loads((w.run.folder / "results" / "claude" / "files" / f"{QUESTIONS[0].qid}.json").read_text(encoding="utf-8"))
    assert result["answer_format"] == "line" and not {"evidence", "grounded", "evidence_unresolved", "answer_error"} & set(result)


def test_a_run_that_has_results_in_one_format_is_refused_the_other_and_a_second_command_keeps_the_format(w, capsys):
    stub = Stub()
    assert w.main(*run_args(w, "--answer-format", "line"), agents=lambda: {"claude": Agent(stub)}) == 0
    capsys.readouterr()
    other = Stub()
    assert w.main("run", str(w.run.folder), "--agents", "claude", "--configs", "files", "--answer-format", "json", agents=lambda: {"claude": Agent(other)}) == 1
    err = capsys.readouterr().err
    assert "has results in the line answer format, so it cannot be asked in the json format now" in err and "--answer-format line" in err
    assert other.execs == [] and held(w.run) == "line", "refused before anything ran, and run.json is as it was"
    assert w.main("run", str(w.run.folder), "--agents", "claude", "--configs", "files", "--only", QUESTIONS[1].qid, agents=lambda: {"claude": Agent(other)}) == 0
    assert "End with one line ANSWER:" in other.execs[0].prompt and held(w.run) == "line", "not asked for a format: the run's own"
    assert w.main("run", str(w.run.folder), "--agents", "claude", "--configs", "files", "--answer-format", "json", "--dry-run") == 1, "a dry run is refused too"
    capsys.readouterr()


def test_a_run_with_no_results_may_change_its_format_and_a_run_recorded_before_the_format_was_kept_is_line(w, tmp_path):
    fs.write_json(w.run.folder / "run.json", {**fs.read_json(w.run.folder / "run.json"), "answer_format": "line"})
    assert answer_format_of(w.run, "json") == "json" and answer_format_of(w.run, None) == "line"
    fs.write_json(w.run.folder / "run.json", {k: v for k, v in fs.read_json(w.run.folder / "run.json").items() if k != "answer_format"})
    fs.write_json(w.run.result_path("claude", FILES, QUESTIONS[0].qid), {"qid": QUESTIONS[0].qid, "problem": None})
    assert answer_format_of(w.run, None) == "line", "a run with results and no format in run.json is a recorded run: its answers are lines"
    assert answer_format_of(w.run, "line") == "line"
    with pytest.raises(ValueError, match="has results in the line answer format, so it cannot be asked in the json format now"):
        answer_format_of(w.run, "json")
    only_retrieval = tmp_path / "retrieval-only"
    shutil.copytree(w.run.folder, only_retrieval)
    shutil.rmtree(only_retrieval / "results")
    fs.write_json(only_retrieval / "results" / "retrieval" / "stub-all" / f"{QUESTIONS[0].qid}.json", {"qid": QUESTIONS[0].qid})
    assert answer_format_of(RunFolder.open(only_retrieval, scenarios.get), "json") == "json", "a retrieval record is no exec result"


def test_the_recorded_runs_are_lines_whatever_they_are_asked_by_default():
    recorded = Path(__file__).parent / "oracles" / "recorded"
    for name in ("pilot-s1", "egolife-s1"):
        meta = json.loads((recorded / name / "converted" / "run.json").read_text(encoding="utf-8"))
        assert "answer_format" not in meta, "(the converted run.json is the recorded one: it has no format)"
        results = list((recorded / name / "converted" / "results").rglob("*.json"))
        assert results and not any("answer_format" in r.read_text(encoding="utf-8") for r in results[:50]), "and no result has a field of the format"


def without_evidence(w):
    """The toy run with one question that is not absent left with no evidence (its question file and the hash run.json keeps of it written again):
    that question."""
    bare_q = next(q for q in QUESTIONS if q.kind != "absent")
    bare = dataclasses.replace(bare_q, truth=dataclasses.replace(bare_q.truth, evidence=()))
    write_questions(w.run.folder / "questions.jsonl", [bare if q.qid == bare_q.qid else q for q in QUESTIONS])
    fs.write_json(w.run.folder / "run.json", {**fs.read_json(w.run.folder / "run.json"), "question_hash": fs.sha256_of(w.run.folder / "questions.jsonl")})
    return bare_q


def test_a_json_run_with_a_question_no_citation_can_ground_is_refused_before_any_agent(w, capsys):
    """`ground` raised for a question that is not absent and has no evidence after the exec that answered it, which had spent the plan. The run
    refuses it at its start (`grounded.check_groundable`, in `run_agents` and in the dry run), naming the question."""
    bare_q = without_evidence(w)
    stub = Stub()
    ask = argv(w, "--agents", "claude", "--configs", "files", "--only", bare_q.qid)
    for flags in ((), ("--dry-run",), ("--answer-format", "json")):
        assert w.main(*ask, *flags, agents=lambda: {"claude": Agent(stub)}) == 1, flags
        assert f"error: {bare_q.qid} is not an absent question and has no evidence, so no citation can ground it" in capsys.readouterr().err
    assert stub.execs == [] and not (w.run.folder / "results").exists(), "refused before any agent ran"
    # only the questions the run asks count: another question of the run is not refused for this one
    other = next(q.qid for q in QUESTIONS if q.kind != "absent" and q.qid != bare_q.qid)
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--only", other), agents=lambda: {"claude": Agent(stub)}) == 0
    assert len(stub.execs) == 1


def test_a_line_run_grounds_nothing_so_a_question_without_evidence_is_asked_in_it(w):
    bare_q = without_evidence(w)
    stub = Stub()
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--only", bare_q.qid, "--answer-format", "line"),
                  agents=lambda: {"claude": Agent(stub)}) == 0
    assert len(stub.execs) == 1 and result(w.run, "claude", "files", bare_q.qid)["answer_format"] == "line"
