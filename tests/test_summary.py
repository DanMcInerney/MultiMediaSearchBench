"""The summary report (core/summary.py): what each agent got right in each configuration, a leak filter's questions kept out of every
other figure, results to run again left out, each other configuration beside the control, and the scenarios pooled. Over runs written
by hand with results made by `record`; expected figures are counted from the answers the tests chose, not from the code."""

from __future__ import annotations

import types
from datetime import date
from pathlib import Path

import pytest

from mmmb.core import fs, stats, summary
from mmmb.core.config import parse_config
from mmmb.core.question import Span, write_questions
from mmmb.core.record import record
from mmmb.core.runfolder import RunFolder
from mmmb.scenarios import realdump

DEFAULTS = {"files": None, "listing": None, "stub": "all"}
FILES, LISTING, STUB = (parse_config(c, DEFAULTS) for c in ("files", "listing", "stub"))
CAMERA = types.SimpleNamespace(name="camera_scenario", code="C1")
SOUNDS = types.SimpleNamespace(name="sound_scenario", code="S1")


def questions():
    """Six questions: four with something to find (one a `set`), two absent."""
    def q(qid, kind, g, value=None):
        evidence = [] if kind == "absent" else [Span(f"{qid}.jpg", 0.0, 0.0)]
        return realdump.question(code="C1", qid=qid, now=1_700_000_000.0, text=f"{qid}?", kind=kind, template=kind, modality="image",
                                 shape="<x>", grade=g, evidence=evidence, sources=("camera",), since=0.0, until=1.0, value=value)
    return [q("count-1", "count", realdump.grade("count", truth="3"), "3"),
            q("count-2", "count", realdump.grade("count", truth="5"), "5"),
            q("days-1", "days", realdump.grade("set", items=["2013-12-16", "2013-12-18"]), "x"),
            q("count-3", "count", realdump.grade("count", truth="7"), "7"),
            q("absent-1", "absent", realdump.grade("absent")),
            q("absent-2", "absent", realdump.grade("absent"))]


RIGHT = {"count-1": "3", "count-2": "5", "days-1": "2013-12-16 and 2013-12-18", "count-3": "7", "absent-1": "unknown", "absent-2": "none"}


def run_of(root: Path, scenario, seed: int = 1, configs=(FILES, LISTING, STUB)) -> RunFolder:
    root.mkdir(parents=True)
    write_questions(root / "questions.jsonl", questions())
    fs.write_json(root / "run.json", {"mmmb_commit": "abc123", "probe": {"status": "passed", "failures": []}})
    fs.write_json(root / "systems.json", {"listing": {"system": "listing", "description": {"leak_filter": True}},
                                          "stub": {"system": "stub", "description": {"leak_filter": False}}})
    return RunFolder(root, seed, "tiny", (date(2013, 12, 16),), root / "data", root, "tag", 900.0, scenario, tuple(configs))


def answer(run: RunFolder, config, answers: dict, problems: dict | None = None, agent: str = "claude") -> None:
    """Results of `agent` in `config`: each question's answer from `answers` (a qid left out is not asked), with `problems` by qid."""
    for q in run.questions():
        if q.qid in answers:
            out = {"answer": answers[q.qid], "seconds": 10.0, "problem": (problems or {}).get(q.qid), "cost_usd": 0.02}
            fs.write_json(run.folder / "results" / agent / config.dir / f"{q.qid}.json",
                          record(q, agent, config, run.seed, "x", "2026-11-05T12:00:00", out))


@pytest.fixture
def run(tmp_path) -> RunFolder:
    """files: right on all but count-2 and the set (which names one extra day), count-3 to run again; listing: right on count-1 only
    (a leak) and on the absent ones; stub: right on all."""
    r = run_of(tmp_path / "a", CAMERA)
    answer(r, FILES, {**RIGHT, "count-2": "4", "days-1": "2013-12-16, 2013-12-18, 2013-12-20"}, {"count-3": "limit: out of quota"})
    answer(r, LISTING, {**{qid: "unknown" for qid in RIGHT}, "count-1": "3"})
    answer(r, STUB, RIGHT)
    return r


def test_a_leak_is_left_out_of_every_figure_but_the_filters_and_a_result_to_run_again_of_all(run):
    s = summary.numbers([run])["scenarios"][0]
    files, listing, stub = (s["cells"]["claude"][c.name] for c in (FILES, LISTING, STUB))
    assert s["leaks"] == [{"run": "a", "qid": "count-1", "agents": ["claude"]}] and s["filters"] == ["listing"]
    assert (files["correct"], files["n"]) == (2, 4), "count-1 leaked and count-3 is to run again: the absent two of four are right"
    assert (listing["correct"], listing["n"]) == (3, 6), "the filter keeps all its questions"
    assert (stub["correct"], stub["n"]) == (5, 5)
    assert s["to_rerun"] == {"claude files": 1}
    assert files["interval"] == pytest.approx(list(stats.wilson(2, 4)))
    assert files["items_f1"] == pytest.approx(0.8) and stub["items_f1"] == 1.0  # two found of two, one named of three wrong
    assert files["by_kind"] == {"absent": [2, 2], "count": [0, 1], "days": [0, 1]}
    assert files["usd_per_right"] == pytest.approx(0.04) and files["s_per_right"] == pytest.approx(20.0), "4 execs of 10 s, 2 right"


def test_a_control_no_run_has_is_refused(run):
    with pytest.raises(ValueError, match="no run has the configuration 'flies'"):
        summary.numbers([run], "flies")


def test_each_other_configuration_is_set_beside_the_control_on_the_questions_both_have(run):
    s = summary.numbers([run])["scenarios"][0]
    assert set(s["pairs"]["claude"]) == {STUB.name}, "the filter is no contestant"
    pair = s["pairs"]["claude"][STUB.name]
    assert (pair["n"], pair["b"], pair["c"]) == (4, 2, 0), "right only with stub: count-2 and days-1"
    assert pair["diff"] == pytest.approx(0.5)


def test_the_markdown_says_what_the_figures_hold(run):
    text = summary.markdown(summary.numbers([run]))
    assert text.startswith("# MMMB report\n")
    assert "| claude | files | 2/4 (50%) |" in text and "| claude | listing (leak check) | 3/6 (50%) |" in text
    assert "- count-1 (a): claude" in text and "To run again (left out): claude files 1." in text
    assert "| claude | stub/all vs files | 4 | +50 pts" in text and "not shown" in text
    assert "## Over all" not in text


def test_runs_of_several_scenarios_are_pooled_and_the_same_questions_twice_are_refused(run, tmp_path):
    other = run_of(tmp_path / "b", SOUNDS)
    answer(other, FILES, RIGHT)
    figures = summary.numbers([run, other])
    assert [s["scenario"] for s in figures["scenarios"]] == ["camera_scenario", "sound_scenario"]
    assert figures["overall"]["claude"]["files"] == [2 + 6, 4 + 6]
    assert "## Over all the scenarios" in summary.markdown(figures)
    assert "| claude | listing (leak check) | 3/6 (50%) |" in summary.markdown(figures).split("## Over all the scenarios")[1]
    with pytest.raises(ValueError, match="count each question twice"):
        summary.numbers([run, run_of(tmp_path / "c", CAMERA)])
    assert len(summary.numbers([run, run_of(tmp_path / "d", CAMERA, seed=2)])["scenarios"]) == 1, "another seed is pooled"
