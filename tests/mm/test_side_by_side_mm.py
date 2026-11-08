"""Side-by-side scheduling over the real mm system and its real daemon with the fake decider (plan 6.4, U13): V7's tiny week through `mm/fake` and
`files` gives the same results beside each other as one after another; and the agent kept off the GPU does not take the GPU from the model, because
the model is in mm's daemon, which the run starts with the run's own environment (mmsb/systems/mm/daemon.py), and the server the agent starts does no
GPU work. Written for this repository (U13); no paid agent runs and no model is loaded.

Expected values are the plan's words (the same set of results), and what the test planted: a decider that writes the environment of the process it
runs in (tests/mm/env_decider.py) shows what the daemon has, and a stand-in agent run through the real `claude.ask_claude` or `codex.ask_codex`
(tests/stand_in_agent.py) starts the real server with the environment the real agent made, and writes the `CUDA_VISIBLE_DEVICES` it has."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
import seam  # noqa: E402
import stand_in_agent  # noqa: E402
from agent_spec import spec_of  # noqa: E402
from fakes import FILES, S, Opened, Probe, made_up_ffmpeg, run_of  # noqa: E402
from stub_system import right  # noqa: E402
from test_mm_arm import asks, by_qid, go, result  # noqa: E402
from test_seam import git  # noqa: E402, F401

from multimem import decider_d1  # noqa: E402

from mmsb.core.agents import claude, codex  # noqa: E402
from mmsb.core.config import Config  # noqa: E402
from mmsb.core.schedule import Pools, SideBySide  # noqa: E402
from mmsb.systems.mm.models import Model  # noqa: E402
from mmsb.systems.mm.system import Mm  # noqa: E402

pytestmark = pytest.mark.xdist_group("mm-arm")
ENVS = Model("envs", f"python:{Path(fakes.ROOT / 'env_decider.py')}", decider_d1.ABOUT)  # the fake decider that writes its process's environment
SYSTEM = Mm(models={"envs": ENVS})
ENVS_FILES = Config("mm/envs+files", "mm", "envs", True)
TIMING = ("seconds", "at", "held_s", "settle_s")  # what differs between two runs of the same stand-in


def kept(run) -> dict:
    """Every result of the run but its times, by agent, folder and question; and the labels that no exec owned."""
    out = {}
    for f in sorted((run.folder / "results").glob("*/*/*.json")):
        out[f.relative_to(run.folder / "results").as_posix()] = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if k not in TIMING}
    return out


class Timed:
    """The stand-in agent `probe`, with the (folder, source, qid, start, end) of each exec, the wait included. With `meet`, the first exec of
    a cell with no server waits until an exec with the server has started: the cells have to be running beside each other for it to end."""

    def __init__(self, probe: Probe, meet: bool) -> None:
        self.probe, self.meet, self.server_in, self.log, self.lock = probe, meet, threading.Event(), [], threading.Lock()

    def __call__(self, spec) -> dict:
        began = time.monotonic()
        if spec.server is not None:
            self.server_in.set()
        elif self.meet:
            assert self.server_in.wait(120), "no exec of the system started while a files exec waited: the cells do not run beside each other"
        out = self.probe(spec)
        with self.lock:
            self.log.append((spec.run_dir.parent.name, None, spec.run_dir.name, began, time.monotonic()))
        return out


def overlap(log) -> bool:
    return any(a[0] != b[0] and a[3] < b[4] and b[3] < a[4] for a in log for b in log)


def test_v7_tiny_through_mm_fake_and_files_gives_the_same_results_beside_each_other_as_one_after_another(week, store, tmp_path_factory):
    recipe, dump, questions = week
    got = {}
    for side in (None, SideBySide(Pools.of(["claude"]), {})):
        root = tmp_path_factory.mktemp("beside" if side else "alone")
        with pytest.MonkeyPatch.context() as patch:
            made_up_ffmpeg(patch, root)
            run = run_of(root, recipe, questions, dump, [FILES, S])
            probe = Timed(Probe(questions, tools=asks, answer=lambda spec, q: right(q), pause=0.02), meet=side is not None)
            go(run, dump, store, probe, jobs=2, side_by_side=side)
        got[side is not None] = (run, probe)
    (alone, once), (beside, both) = got[False], got[True]
    assert set(kept(alone)) == set(kept(beside)) and len(kept(alone)) >= 2 * len(questions), "every question of both configurations, and the labels no exec owned"
    assert kept(beside) == kept(alone)
    assert not overlap(once.log) and overlap(both.log), "the cells ran one after another, and then beside each other"
    assert all(spec.off_gpu for spec in both.probe.execs if spec.own_tools) and not any(spec.off_gpu for spec in once.probe.execs)
    assert result(beside, "mm-fake", "image-photo-sign-1")["passed"] is True


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_the_server_of_an_agent_kept_off_the_gpu_works_and_the_model_in_the_daemon_never_has_what_the_agent_hides(agent, week, store, tmp_path,
                                                                                                                monkeypatch, git):
    """A stand-in `claude` or `codex` run through the real agent code: Claude's own process has no CUDA device and so has the server it starts (the
    server does no GPU work: it asks the daemon); Codex hides it from its shell alone, so its server has what the run has. Either way the daemon,
    which the run started, has what the run has, and the model it holds is shown by the decider's own environment."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    recipe, dump, questions = week
    q = next(q for q in questions if q.tags["template"] == "photo-sign" and not q.tags["absent"])
    ask = {"question": by_qid(fakes.V7.plan_of(recipe))[q.qid]["fake"]["question"], "sources": ["phone"], "since": q.scope.since, "until": q.scope.until}
    run = run_of(tmp_path, recipe, questions, dump, [ENVS_FILES])
    seen = tmp_path / "server-cuda.txt"
    middle = f"open({str(seen)!r}, 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'))\n" + seam.MIDDLE
    with Opened(store, dump, run.folder / "homes", system=SYSTEM, variant="envs", agent=agent) as session:
        out, listed = stand_in_agent.run_agent(agent, tmp_path, monkeypatch, git, session, run, dump, ENVS_FILES, q, middle,
                                               {"ask": ask, "read": True, "answer": None}, off_gpu=True)
        home = session.home
    assert out["problem"] is None and [t["tool"] for t in out["tools"]][:2] == ["find", "ask"], "the server worked with what the agent gave it"
    assert any(t["tool"] == "ask" and "sightings" in json.dumps(t["result"]) for t in out["tools"]) and set(listed) == {"ask", "find", "look", "label"}
    assert seen.read_text(encoding="utf-8") == ("-1" if agent == "claude" else "unset")
    daemon_env = json.loads((home / "decider-env.json").read_text(encoding="utf-8"))
    assert "CUDA_VISIBLE_DEVICES" not in daemon_env and os.environ.get("CUDA_VISIBLE_DEVICES") is None


def test_building_an_agents_environment_leaves_the_environment_of_a_daemon_started_after_it_alone(week, store, tmp_path, monkeypatch):
    """What an agent kept off the GPU is given is a copy: the run's own `os.environ` is not changed, so no daemon the run starts later (the next cell's
    session, a restart) has the hidden GPU, and a model that runs on the GPU keeps it."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(claude, "git_root", lambda: tmp_path / "git")
    spec = spec_of("files", tmp_path, off_gpu=True)
    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    assert claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation, spec.off_gpu)["CUDA_VISIBLE_DEVICES"] == "-1"
    codex.prepare(spec, "gpt-test", "high", models, ["<codex>"])
    _, dump, _ = week
    with Opened(store, dump, tmp_path / "homes", system=SYSTEM, variant="envs") as session:
        daemon_env = json.loads((session.home / "decider-env.json").read_text(encoding="utf-8"))
    assert "CUDA_VISIBLE_DEVICES" not in daemon_env
