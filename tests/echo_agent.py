"""Stand-in agents that do what the real ones do with their settings (core/agents/claude.py, codex.py): the result says the model they were
given, as `agent_model` becomes in the record, and Codex's says its effort too (Claude's result holds none), each with the agent's default
where `mmmb` asked for none. Over `stub_system.Stub`, which answers; nothing here starts a process."""

from __future__ import annotations

import types
from pathlib import Path

import cli_world
import imageio_ffmpeg
import pytest
from cli_world import CLEAN, build_args
from stub_system import Stub, StubSystem

from mmmb import cli
from mmmb.core.agents import claude
from mmmb.core.agents.spec import SessionSpec
from mmmb.core.runner import Agent

DEFAULT_MODEL = {"claude": "claude-default", "codex": "codex-default"}
DEFAULT_EFFORT = "high"  # Codex's, as `ask_codex` returns it when none was asked for


def echoing(stub: Stub):
    def run(spec: SessionSpec) -> dict:
        agent = spec.run_dir.parent.parent.name  # execs/<agent>/<config dir>/<qid>
        out = stub(spec)
        out["model"] = spec.model or DEFAULT_MODEL[agent]
        if agent == "codex":
            out["effort"] = spec.effort or DEFAULT_EFFORT
        return out

    return run


def agents_of(stub: Stub):
    """The `agents` a command line is given: the stand-ins for both, which echo the model and effort (Codex's) they are given."""
    return lambda: {"claude": Agent(echoing(stub)), "codex": Agent(echoing(stub), must_report_use=True)}


def record_toy(root: Path, *extra: str) -> Path:
    """The toy run built in `root` (seed 7, tiny, the configurations files and stub/all) and run by both stand-in agents with `extra` arguments
    of `mmmb run`, in the `line` format the stub answers in: the folder of the run, as a recorded run to repeat or compare with."""
    with pytest.MonkeyPatch.context() as patch:
        ffmpeg = root / "ffmpeg-source.exe"
        ffmpeg.write_bytes(b"x")
        patch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
        patch.setattr(claude, "check_login", lambda: None)
        cli_world.use(patch, StubSystem(report=CLEAN))
        registry = cli_world.registry(root / "systems")
        assert cli.main(build_args(types.SimpleNamespace(out=root / "runs", temp=root / "temp", cache=root / "cache")), registry=registry) == 0
        (folder,) = (root / "runs").iterdir()
        assert cli.main(["run", str(folder), "--agents", "claude,codex", "--answer-format", "line", *extra], registry=registry,
                        agents=agents_of(Stub())) == 0
    return folder
