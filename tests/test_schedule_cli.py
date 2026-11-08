"""`mmsb run --side-by-side` (plan 6.4, D5: off unless asked): the flag, what a dry run prints of it, and a run that has it, over the scenario
`toy` and the stand-in system of tests/cli_world.py, with the fixtures of tests/test_cli.py. Written for this repository (U13).

The expected values are the plan's words (the flag is off by default; the run says it is on; the same set of results as without it; a dry run
shows it, with a wall estimate through `simulate` or says it is not estimated; the agents' own tools are kept off the GPU beside other cells),
the estimate worked out by hand in the test that checks it, and what the test planted."""

from __future__ import annotations

import shutil

import cli_world
import pytest
from cli_world import world  # noqa: F401
from stub_system import Stub, tiny_questions
from test_cli import argv, built, no_agents_installed, w  # noqa: F401

from mmsb import cli
from mmsb.core import fs

pytestmark = pytest.mark.xdist_group("cli")
QUESTIONS = tiny_questions()
IN_TIME = sorted(QUESTIONS, key=lambda q: (q.scope.until, q.qid))


def kept(w) -> dict:
    """What each result of the run says of the exec's outcome (not its time): by agent, folder and question."""
    return {f.relative_to(w.run.folder / "results").as_posix(): (r["passed"], r["problem"], r["answer"], r["agent"], r["config"])
            for f in sorted((w.run.folder / "results").glob("*/*/*.json")) for r in [fs.read_json(f)]}


def test_the_flag_is_off_unless_it_is_given():
    parser = cli.parser_of()
    assert parser.parse_args(["run", "RUN"]).side_by_side is False
    assert parser.parse_args(["run", "RUN", "--side-by-side"]).side_by_side is True


def test_a_run_with_the_flag_says_it_is_on_and_leaves_the_results_a_run_without_it_leaves(w, capsys):
    configs = ("--agents", "claude,codex", "--configs", "files,stub/all", "--jobs", "2")
    assert w.main(*argv(w, *configs, "--side-by-side"), agents=cli_world.agents_of(Stub())) == 0
    out = capsys.readouterr().out
    assert out.count("side by side: on") == 1 and "side by side (--side-by-side)" in out
    beside = kept(w)
    assert len(beside) == 4 * len(QUESTIONS), "two agents, two configurations, every question"
    shutil.rmtree(w.run.folder / "results")
    assert w.main(*argv(w, *configs), agents=cli_world.agents_of(Stub())) == 0
    out = capsys.readouterr().out
    assert "side by side" not in out
    assert kept(w) == beside


def test_a_dry_run_with_the_flag_keeps_the_files_agent_off_the_gpu_and_one_without_it_does_not(w, capsys):
    no_agents_installed(w.patch)
    assert w.main(*argv(w, "--agents", "claude,codex", "--configs", "files", "--dry-run")) == 0
    plain = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES" not in plain and "kept off the GPU" not in plain and "side by side" not in plain
    assert w.main(*argv(w, "--agents", "claude,codex", "--configs", "files", "--dry-run", "--side-by-side")) == 0
    out = capsys.readouterr().out
    assert "set CUDA_VISIBLE_DEVICES=-1" in out, "Claude's process"
    assert "CUDA_VISIBLE_DEVICES='-1'" in out and "so is its shell's CUDA_VISIBLE_DEVICES" in out, "Codex's shell"
    assert out.count("kept off the GPU: -hwaccel, nvidia-smi are refused in its commands") == 2
    assert "side by side (--side-by-side): not estimated, no earlier exec of these cells to take their durations from" in out
    assert "nothing was run" in out


def test_a_dry_run_with_the_flag_estimates_the_wall_time_by_simulating_the_cells_over_what_earlier_execs_took(w, capsys):
    """Earlier runs took 100 s an exec in `claude files` (three results, kept in the run's own folder) and nothing is known of `claude stub/all`
    (counted at 2.5 min = 150 s). With one exec at once, the three questions still to run take, hand worked out:
      one after another  files 3 x 100 s = 300, then stub/all 3 x 150 s = 450: 750 s = 0.2 h
      side by side       the two cells run together, at one exec each, as long as the longer: 450 s = 0.1 h"""
    for q in IN_TIME[:3]:
        fs.write_json(w.run.folder / "results" / "claude" / "files" / f"{q.qid}.json", {"qid": q.qid, "problem": None, "seconds": 100.0})
    todo = [q.qid for q in IN_TIME[3:6]]
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files,stub/all", "--jobs", "1", "--only", ",".join(todo), "--dry-run",
                        "--side-by-side")) == 0
    out = capsys.readouterr().out
    assert ("side by side (--side-by-side): about 0.1 h at the medians, against 0.2 h one after another (the same cases simulated both ways: "
            "one cell that needs the GPU at a time, at most 4 execs at once); no earlier exec for claude stub/all: counted at 2.5 min an exec "
            "(1 blind)") in out
