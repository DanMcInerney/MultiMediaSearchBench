"""`mmmb compare` and decision D1 (plan 9.3, core/compare.py): two runs of the same questions cell by cell, the flips by qid, the band of +-2 and
McNemar. Over the toy scenario's 19 questions, run once by the stand-in agents of tests/echo_agent.py, whose answers are known without
running anything: `files` (the agent's own tools only) is right for the 3 text and 3 records questions and the 3 absent ones about
pictures, video and sound, 9 of 19, and `stub/all` is right for all 19, for each agent. The recorded pilot's own counts (Claude files 23, Claude
mm 22, Codex files 27, Codex mm 23) and its flips are tests/oracles/test_recorded_repeat.py's."""

from __future__ import annotations

import dataclasses
import shutil
import types
from pathlib import Path

import pytest
from cli_world import world  # noqa: F401
from echo_agent import record_toy
from stub_system import tiny_questions

from mmmb import scenarios
from mmmb.core import compare, fs
from mmmb.core.compare import BAND, Cell, Comparison, compare_runs, render_comparison, settings_of
from mmmb.core.config import Config
from mmmb.core.runfolder import RunFolder

pytestmark = pytest.mark.xdist_group("cli")
N = len(tiny_questions())
FILES, STUB = Config("files", "files", None, True), Config("stub/all", "stub", "all", False)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """The toy run built and run by both agents, the Codex one at effort medium, once for the module."""
    return record_toy(tmp_path_factory.mktemp("compare"), "--codex-effort", "medium")


@pytest.fixture
def pair(world, recorded, tmp_path) -> tuple[RunFolder, RunFolder]:  # noqa: F811
    """The recorded run (A) and a copy of it (B) to change, both opened."""
    shutil.copytree(recorded, tmp_path / "again")
    return RunFolder.open(recorded, scenarios.get), RunFolder.open(tmp_path / "again", scenarios.get)


def set_passed(run: RunFolder, agent: str, config: Config, qids, passed: bool) -> None:
    for qid in qids:
        path = run.result_path(agent, config, qid)
        fs.write_json(path, {**fs.read_json(path), "passed": passed, "answer": "changed"})


def passed_in(run: RunFolder, agent: str, config: Config, passed: bool) -> list[str]:
    return [q.qid for q in run.questions() if fs.read_json(run.result_path(agent, config, q.qid))["passed"] is passed]


def cell(comparison: Comparison, agent: str, config: str) -> Cell:
    return next(c for c in comparison.cells if (c.agent, c.config) == (agent, config))


def test_a_run_compared_with_itself_has_no_flips_every_cell_inside_the_band_and_d1s_numbers_hold(pair):
    a, b = pair
    c = compare_runs(a, b)
    assert [(x.agent, x.config) for x in c.cells] == [("claude", "files"), ("claude", "stub/all"), ("codex", "files"), ("codex", "stub/all")]
    assert [(x.correct_a, x.correct_b, x.n_a, x.n_b, x.paired) for x in c.cells] == [(9, 9, N, N, N), (19, 19, N, N, N)] * 2
    assert all(x.flips == () and (x.b, x.c, x.p, x.unpaired) == (0, 0, 1.0, ()) and x.inside and not x.significant for x in c.cells)
    assert c.conditions == {"every cell inside the band": True, "no cell with McNemar p under 0.05": True, "the verdict lines read the same": True,
                            "no setup result in B": True}
    assert set(c.verdicts) == {("claude", "stub/all"), ("codex", "stub/all")} and all(x == y for x, y in c.verdicts.values())
    text = render_comparison(c)
    assert "Flips, to be read" in text and text.split("Flips, to be read")[1].split("Verdict lines")[0].strip().splitlines()[-1].strip() == "none"
    assert "claude files: 9/19 in A, 9/19 in B, +0, inside the band of +-2" in text
    assert "codex stub/all: 19/19 in A, 19/19 in B, +0, inside the band of +-2" in text


def test_one_planted_flip_is_the_one_flip_listed_by_its_qid_with_the_two_answers_and_what_the_question_wants(pair):
    a, b = pair
    target = passed_in(a, "codex", FILES, True)[0]
    set_passed(b, "codex", FILES, [target], False)
    c = compare_runs(a, b)
    (flip,) = [f for x in c.cells for f in x.flips]
    assert flip.qid == target and flip.a["passed"] and not flip.b["passed"]
    x = cell(c, "codex", "files")
    # by hand: one question right in A only, none right in B only: b 0, c 1; McNemar's p is 2 * (1/2) = 1; the count falls from 9 to 8
    assert (x.flips, x.b, x.c, x.p, x.correct_a, x.correct_b, x.difference, x.inside) == ((flip,), 0, 1, 1.0, 9, 8, -1, True)
    assert all(not y.flips for y in c.cells if y is not x)
    text = render_comparison(c)
    assert text.count(f" {target}: wants ") == 1, "named once, among the flips"
    line = next(line for line in text.splitlines() if f" {target}: wants " in line)
    assert line.startswith("  codex files ") and "A pass '" in line and "B FAIL 'changed'" in line
    assert "codex files: 9/19 in A, 8/19 in B, -1, inside the band" in text and "1 flipped (right in B only 0, in A only 1), McNemar p 1.000" in text


def test_a_cell_is_inside_the_band_at_exactly_plus_or_minus_2_and_outside_at_3_and_mcnemar_is_the_exact_p_of_the_flips(pair):
    a, b = pair
    # `claude files` is 9 of 19 in A: 9 right and 10 wrong to flip. -2 is inside and -3 is outside; +2 is inside and +3 is outside
    for flips, expected_p, direction, inside in ((2, 0.5, -1, True), (3, 0.25, -1, False), (2, 0.5, 1, True), (3, 0.25, 1, False)):
        shutil.rmtree(b.folder / "results")
        shutil.copytree(a.folder / "results", b.folder / "results")
        qids = passed_in(a, "claude", FILES, direction < 0)[:flips]
        set_passed(b, "claude", FILES, qids, direction > 0)
        x = cell(compare_runs(a, b), "claude", "files")
        assert (x.difference, x.inside) == (direction * flips, inside), (direction, flips)
        # by hand, all the flips on one side: p = 2 * C(n,0) / 2**n = 2 / 2**n, that is .5 for 2 and .25 for 3
        assert x.p == expected_p and not x.significant
    shutil.rmtree(b.folder / "results")
    shutil.copytree(a.folder / "results", b.folder / "results")
    set_passed(b, "claude", FILES, passed_in(a, "claude", FILES, True)[:6], False)
    x = cell(compare_runs(a, b), "claude", "files")
    assert (x.difference, x.p, x.significant) == (-6, 2 / 2 ** 6, True), "6 flips one way: p = .03125, under .05"
    assert (BAND, compare.ALPHA) == (2, 0.05), "D1's numbers"


def test_the_band_and_the_level_of_significance_are_d1s_in_one_place(pair, monkeypatch):
    a, b = pair
    set_passed(b, "claude", FILES, passed_in(a, "claude", FILES, True)[:2], False)
    assert cell(compare_runs(a, b), "claude", "files").inside
    monkeypatch.setattr(compare, "BAND", 1)
    assert not cell(compare_runs(a, b), "claude", "files").inside
    monkeypatch.setattr(compare, "ALPHA", 0.6)
    assert cell(compare_runs(a, b), "claude", "files").significant, "p of .5 is under .6"


def test_a_cell_made_at_another_effort_or_by_another_model_is_refused_and_said_so_unless_it_is_told_to_compare_anyway(pair):
    a, b = pair
    assert settings_of({"q": {"agent_model": "m", "agent_effort": "medium"}, "r": {"agent_model": "m", "agent_effort": "medium"}}) == ("m at medium effort",)
    assert settings_of({"q": {"agent_model": "m"}, "r": {"agent_model": None}}) == ("m", "unknown")
    for qid in (q.qid for q in b.questions()):
        path = b.result_path("codex", FILES, qid)
        fs.write_json(path, {**fs.read_json(path), "agent_effort": "high"})
    path = b.result_path("claude", STUB, tiny_questions()[0].qid)
    fs.write_json(path, {**fs.read_json(path), "agent_model": "claude-other"})
    c = compare_runs(a, b)
    refused = {(x.agent, x.config): x.refusal for x in c.refused}
    assert set(refused) == {("codex", "files"), ("claude", "stub/all")}
    assert refused["codex", "files"] == ("made by codex-default at medium effort in A and codex-default at high effort in B: another experiment; "
                                         "--across-settings compares it anyway")
    assert refused["claude", "stub/all"] == ("made by claude-default in A and claude-default + claude-other in B: another experiment; "
                                             "--across-settings compares it anyway")
    assert not cell(c, "codex", "files").figures and not c.conditions["every cell inside the band"]
    text = render_comparison(c)
    assert "codex files: REFUSED, made by codex-default at medium effort in A" in text and "2 cells were refused for their settings" in text
    # told to, it compares them and marks them
    again = compare_runs(a, b, across_settings=True)
    assert again.refused == [] and cell(again, "codex", "files").figures
    assert "SETTINGS DIFFER: A codex-default at medium effort, B codex-default at high effort" in render_comparison(again)


def test_a_cell_only_one_run_has_is_not_compared_and_stops_the_band_from_holding_and_a_setup_result_in_b_is_counted(pair):
    a, b = pair
    shutil.rmtree(b.folder / "results" / "codex" / "stub-all")
    first = tiny_questions()[0].qid
    path = b.result_path("claude", FILES, first)
    fs.write_json(path, {**fs.read_json(path), "passed": False, "problem": "setup: no login"})
    c = compare_runs(a, b)
    assert cell(c, "codex", "stub/all").only == "A" and not cell(c, "codex", "stub/all").inside
    x = cell(c, "claude", "files")
    assert (x.setup_b, x.unpaired, x.paired, x.correct_b, x.problems_b) == (1, (first,), N - 1, 8, {"setup": 1}), \
        "text-note-code-1 was right in A; B's setup result is no done result, so it is not a flip, and it is not right"
    assert c.conditions["no setup result in B"] is False and c.conditions["every cell inside the band"] is False
    text = render_comparison(c)
    assert "codex stub/all: only run A has it, so it is not compared" in text
    assert f"1 without a done pair, not compared: {first}" in text and "setup in B: 1" in text


def test_the_verdict_lines_are_told_a_then_b_and_a_difference_is_flagged_and_fails_that_condition():
    cells = (Cell("claude", "files", "files", (), ()),)
    same = Comparison("a", "b", cells, {("claude", "mm/x"): ("none: ...", "none: ...")}, 31)
    differ = Comparison("a", "b", cells, {("claude", "mm/x"): ("none: ...", "`day` by accuracy (accuracy +20 pts)")}, 31)
    assert same.conditions["the verdict lines read the same"] and not differ.conditions["the verdict lines read the same"]
    assert "claude mm/x: same: none: ..." in render_comparison(same)
    text = render_comparison(differ)
    assert "claude mm/x: DIFFER\n      A none: ...\n      B `day` by accuracy (accuracy +20 pts)" in text and "NO: the verdict lines read the same" in text


def test_the_text_says_what_d1_counts_and_what_a_pass_does_not_mean_and_that_the_flips_are_for_a_person(pair):
    a, b = pair
    text = render_comparison(compare_runs(a, b))
    assert ("D1 (plan 9.3): a cell is inside the band when its correct count is within +-2 of A's; McNemar's p of A against B is under 0.05 in none; "
            "the verdict lines read the same; B has no setup result; every flip is read by a person.") in text
    assert "run once in json; run `line` as well only if a cell falls outside the band." in text
    assert ("A pass of D1's numbers means no large loss, not that the runs agree: the agents are stochastic, a band of 2 is about 11 points of 19 "
            "questions, and McNemar's p over so few discordant pairs has little power. Read every flip above, one by one, before calling the runs the same.") in text
    assert "yes: every cell inside the band" in text and "yes: the verdict lines read the same" in text
    assert "codex records no effort" not in text and "claude records no effort in its results, so only its model is compared." in text


def test_the_command_compares_two_runs_prints_the_text_and_exits_1_only_when_a_cell_was_refused(world, recorded, tmp_path, capsys):
    shutil.copytree(recorded, tmp_path / "again")
    assert world.main("compare", str(recorded), str(tmp_path / "again")) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"compare {recorded.name} (A) with again (B): 19 questions")
    set_passed(RunFolder.open(tmp_path / "again", scenarios.get), "claude", FILES, [tiny_questions()[0].qid], False)
    assert world.main("compare", str(recorded), str(tmp_path / "again")) == 0 and "claude files" in capsys.readouterr().out
    path = RunFolder.open(tmp_path / "again", scenarios.get).result_path("codex", FILES, tiny_questions()[0].qid)
    fs.write_json(path, {**fs.read_json(path), "agent_effort": "low"})
    assert world.main("compare", str(recorded), str(tmp_path / "again")) == 1 and "REFUSED" in capsys.readouterr().out
    assert world.main("compare", str(recorded), str(tmp_path / "again"), "--across-settings") == 0 and "SETTINGS DIFFER" in capsys.readouterr().out
    assert world.main("compare", str(recorded), str(tmp_path / "nowhere")) == 1 and "error:" in capsys.readouterr().err


def test_runs_that_are_not_of_one_scenario_and_plan_are_not_compared(pair, monkeypatch):
    a, b = pair
    other = dataclasses.replace(b, scenario=types.SimpleNamespace(name="v7_screen"))
    with pytest.raises(ValueError, match="is of toy and .* of v7_screen: there is nothing to compare"):
        compare_runs(a, other)
    fs.write_json(b.folder / "run.json", {**fs.read_json(b.folder / "run.json"), "plan": "0" * 64})
    with pytest.raises(ValueError, match="do not ask the same questions"):
        compare_runs(a, b)
