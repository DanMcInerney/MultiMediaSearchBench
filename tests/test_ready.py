"""Readiness before spending (the contract's `System.check_ready`, the CLI's `run` and `repeat`): a run that names a system which cannot run now
(mm without torch, a model not downloaded) is refused before its first agent starts, where it used to run the `files` cells and fail at the
`open` of the next system. A dry run starts nothing and asks nothing. mm's own answer, through `Mm` with no torch, is tests/mm/test_ready_mm.py."""

from __future__ import annotations

import cli_world
import pytest
from cli_world import CLEAN, world  # noqa: F401
from stub_system import Stub, StubSystem, tiny_questions

from mmmb.core import fs
from mmmb.core.agents import claude
from mmmb.core.runner import Agent

pytestmark = pytest.mark.xdist_group("cli")
FIRST = tiny_questions()[0].qid


def test_a_run_of_a_cell_that_needs_no_system_and_one_that_cannot_run_is_refused_before_the_first_exec_and_changes_nothing(world, capsys):
    run = cli_world.built_run(world)
    world.patch.setattr(claude, "check_login", lambda: None)
    unready = cli_world.use(world.patch, StubSystem(report=CLEAN, unready="its environment has no torch"))
    stub, before = Stub(), (run.folder / "run.json").read_bytes()
    capsys.readouterr()
    assert world.main("run", str(run.folder), "--agents", "claude,codex", "--configs", "files,stub/all", "--only", FIRST,
                      agents=cli_world.agents_of(stub)) == 1
    assert "error: stub/all cannot run now: all: its environment has no torch" in capsys.readouterr().err
    assert stub.execs == [], "the files cell was not run: that is the point"
    assert unready.kinds("check_ready") and not unready.kinds("open") and not (run.folder / "results").exists()
    assert (run.folder / "run.json").read_bytes() == before and not (run.folder / "systems.json").exists()
    # the configuration that needs no system is not held back by one it does not name
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", FIRST, agents=cli_world.agents_of(stub)) == 0
    assert len(stub.execs) == 1


def test_every_configuration_is_asked_before_any_agent_starts_and_in_the_order_given(world):
    run = cli_world.built_run(world)
    world.patch.setattr(claude, "check_login", lambda: None)
    events: list = []
    system = cli_world.use(world.patch, StubSystem(report=CLEAN, shared=events))
    stub = Stub()

    def recording(spec):
        events.append(("exec", spec.run_dir.name))
        return stub(spec)

    agents = lambda: {"claude": Agent(recording)}  # noqa: E731
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files,stub/all,stub/narrow", "--only", FIRST, agents=agents) == 0
    kinds = [kind for kind, _ in events]
    assert kinds.index("exec") > max(i for i, kind in enumerate(kinds) if kind == "check_ready"), "all the checks come before the first exec"
    assert [e[2] for e in system.kinds("check_ready")] == ["all", "narrow"], "one for each configuration with a system, in the order given"


def test_a_dry_run_asks_nothing_so_a_system_that_cannot_run_still_has_its_exec_shown(world, capsys):
    run = cli_world.built_run(world)
    unready = cli_world.use(world.patch, StubSystem(report=CLEAN, unready="its environment has no torch"))
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files,stub/all", "--only", FIRST, "--dry-run") == 0
    assert "nothing was run" in capsys.readouterr().out and not unready.kinds("check_ready") and not unready.kinds("open")
    assert fs.read_json(run.folder / "run.json")["timeout_s"] == 900.0
