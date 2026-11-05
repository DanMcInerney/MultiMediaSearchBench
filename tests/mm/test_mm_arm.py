"""Ported from tests/test_bench.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order: the tests of the mm arm, which the neutral loop
(tests/test_runner.py) leaves to the system, over the tiny V7 week ingested by the system (conftest's `week` and `store`) and a daemon with the
fake decider. What moved, by the plan's 4.5 and 4.6, and what the expectations are:
- the arm is the configuration `mm/fake` (the system's tools) and `mm/fake+files` (and the agent's own); the `built` run is `fakes.run_of`;
  `bench.run_arm` is `run_config` over an opened session; `bench.Exec` is a `SessionSpec`; `arm` and `model` are `config`, `system`, `variant`;
- `copy_home`, `scrub_home`, `quiesce` and the rest are the system's (`mmmb.systems.mm.home`, `jobs`), `gpu_estimate` is `Session.estimate`;
- the ask-job half of `work_of` is `mmmb.systems.mm.audit.work_of`, which `Session.audit` merges over the core's `work`;
- the oracle's level rules (5 sightings for a `last` question, 50 for a count, `oldest_first` left out of a `last` one) are the question writer's and
  tests/test_v7_questions.py's; here the query they make is what the ask passes.
Not ported, with reasons: `test_remove_tree_removes_a_junction...` (the core's, tests/test_fs.py); the report, run-folder, stop, resume, files-arm
and CLI tests (the core's, U4-U6, U8); `test_a_lost_gpu_stops_the_mm_arm...` (a rule of the core keyed on `Description.needs_gpu`, tests/test_runner.py);
`test_the_real_agents_pass_the_timeout...` (the agents', U3); `test_agents_names_the_questions_whose_ask_may_need_more_gpu_seconds...` prints from a
dry run (U8): its thresholds are `test_the_flag_for_an_ask_that_cannot_finish...` below.
Every expected value is the source's or worked out from what the test planted; the changes of an expectation are named where they are."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sqlite3
import threading
import time
import types
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from agent_spec import clocked  # noqa: E402
from fakes import FAKE, FAKE_MODEL, FILES, MM, S, S_FILES, Probe, context, home_of, made_up_ffmpeg, run_of  # noqa: E402
from model_folders import model_folder  # noqa: E402
from stub_system import right  # noqa: E402

from multimem import config, decider  # noqa: E402

from mmmb import systems  # noqa: E402
from mmmb.core import fs, render, report, retrieval  # noqa: E402
from mmmb.core.agents.common import scrubbed  # noqa: E402
from mmmb.core.contract import Case, RetrievalQuery  # noqa: E402
from mmmb.core.record import Halt  # noqa: E402
from mmmb.core.runner import Agent, run_agents, run_config  # noqa: E402
from mmmb.systems.mm import estimate, models, system  # noqa: E402
from mmmb.systems.mm.audit import work_of  # noqa: E402
from mmmb.systems.mm.daemon import Api, Daemon  # noqa: E402
from mmmb.systems.mm.estimate import HEAVY_S, arm_jobs, beyond_timeout, window_s  # noqa: E402
from mmmb.systems.mm.home import COPY_SKIPS, MEDIA_FOLDERS, PROJECT_PREFIX, copy_home, exec_project, kept_items, scrub_home  # noqa: E402
from mmmb.systems.mm.jobs import cancel_open, open_jobs, quiesce, settle, wait_idle  # noqa: E402
from mmmb.systems.mm.models import Model, needs_gpu, resolve_models  # noqa: E402
from mmmb.systems.mm.oracle import ask, ask_params, moment_of, retrieve  # noqa: E402
from mmmb.systems.mm.system import MmSession, serve_record  # noqa: E402

AS_OF = ("notes", "config")  # the sources whose files are versions (the dump's text)
pytestmark = pytest.mark.xdist_group("mm-arm")  # (this xdist does not keep a group on one worker: each worker that runs a test builds its own week)


@pytest.fixture(scope="module")
def tiny(week):
    return fakes.V7.plan_of(week[0])


def by_qid(plan) -> dict[str, dict]:
    return {q["qid"]: q for q in plan.questions}


def scan(api, **params) -> dict:
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


def go(run, dump, store, agent, configs=None, jobs: int = 2, **kw) -> None:
    """`run_agents` over `run` with the mm system (the fake decider is its variant `fake`) and the registry for the rest."""
    run_agents(run, {"claude": Agent(agent)}, list(configs or run.configs), dump, lambda name: store, jobs,
               load=lambda name: MM if name == "mm" else systems.load(name), isolation=systems.isolation(), answer_format="line", **kw)


def result(run, folder: str, qid: str, agent: str = "claude") -> dict:
    return json.loads((run.folder / "results" / agent / folder / f"{qid}.json").read_text(encoding="utf-8"))


def mm_result(payload) -> dict:
    """A tool result as Codex's tools.json holds an mm tool's: the JSON in one text block."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "structured_content": None}


def call(tool: str, arguments: dict, result=None) -> dict:
    return {"tool": tool, "arguments": arguments, "result": result, "error": None, "status": "completed"}


def job_result(job: str, state: str, windows: int, seconds: float, stopped: str | None = None) -> dict:
    """What an ask gives back, as the smoke's tools.json shows: the job, its state, and its progress and stop."""
    return {"job": job, "state": state, "error": None, "sightings": [], "candidates": [], "searched": {"scope": "default"},
            **({"stopped": stopped} if stopped else {}), "warnings": [],
            "progress": {"windows": windows, "of_about": 121, "seconds": seconds, "decider": "ready"}}


# an mm exec of Codex as the smoke recorded them (find, ask, look: the pictures of `look` left out as a hash)...
CODEX_MM = [
    call("find", {"sources": ["screen"], "since": "2026-11-01T18:00:00", "limit": 20}, mm_result({"results": [], "next": None})),
    call("ask", {"question": "Is a red error screen visible?", "sources": ["screen"], "limit": 1}, mm_result(job_result("j-3c29", "running", 30, 2.0))),
    call("ask", {"job": "j-3c29"}, mm_result(job_result("j-3c29", "done", 121, 9.5, "max_s"))),
    call("ask", {"question": "Is a parcel visible?", "sources": ["porch"]}, mm_result(job_result("j-b7bb", "done", 4, 0.4, "limit"))),
    call("ask", {"question": "Is a sign visible?"}, mm_result(job_result("j-0001", "running", 10, 1.0))),
    call("ask", {"question": "?"}, "ask failed: bad_request: sources"),
    call("look", {"moments": [{"item": 36, "at": 80}]},
         {"content": [{"type": "image", "data": "<60296 characters, sha1 d0c4559cb9da>", "mimeType": "image/jpeg"},
                      {"type": "text", "text": "{\"kind\":\"image\",\"item\":36}"}], "structured_content": None}),
]


# ...Codex without mm (its `view_image` calls come from its rollout, with no result), and Claude's in both arms (its results are the
# blocks, or the text, of the tool)
CODEX_FILES = [call("shell", {"command": "ls"}, "a.mkv"), call("shell", {"command": "ffmpeg -i a.mkv -vf tile m.jpg"}, ""),
               call("view_image", {"path": "C:\\w\\m.jpg"}), call("shell", {"command": "ffmpeg ..."}, ""), call("view_image", {"path": "c.png"})]


CLAUDE_FILES = [call("Glob", {"pattern": "*"}, "a.mkv"), call("Bash", {"command": "ffmpeg -i a.mkv m.jpg"}, ""),
                call("Read", {"file_path": "C:\\w\\montage.jpg"}, [{"type": "text", "text": "[image]"}]),
                call("Read", {"file_path": "/data/notes/n.md"}, "text"), call("Read", {"file_path": "C:\\w\\FRAME.PNG"}),
                call("Grep", {"pattern": "ERR"}, "x"), call("Bash", {"command": "ffmpeg -y"}, "")]


CLAUDE_MM = [call("ask", {"question": "Is a sign visible?"}, [{"type": "text", "text": json.dumps(job_result("j-c1", "done", 50, 3.0, "range_end"))}]),
             call("ask", {"question": "again"}, json.dumps(job_result("j-c2", "done", 7, 0.5, "cancelled"))),
             call("find", {"sources": ["phone"]}, [{"type": "text", "text": "{}"}]), call("look", {"moments": []}, "ok")]


class JobApi:
    """The daemon's job API as a test scripts it: `jobs/cancel` returns at once with the job still running (a cancel that timed
    out), the job ends `ends_after` seconds after it (its late label lands then, noted in `events`), and `status` counts jobs
    from `counts` in turn, the last repeating. A job not in `known` is unknown to the daemon."""

    def __init__(self, events: list | None = None, ends_after: float = 0.4, counts=(0,), known=("j-late",)):
        self.events, self.ends_after, self.counts, self.known = events if events is not None else [], ends_after, list(counts), set(known)
        self.cancelled_at: dict[str, float] = {}
        self.calls: list[tuple] = []

    def call(self, op: str, **body) -> dict:
        self.calls.append((op, body))
        if op == "status":
            n = self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]
            return {"jobs": {"running": n, "queued": 0}}
        if body["job"] not in self.known:
            raise RuntimeError("not_found: no such job")
        if op == "jobs/cancel":
            self.cancelled_at.setdefault(body["job"], time.monotonic())
            return {"state": "running"}
        at = self.cancelled_at.get(body["job"])
        if at is None or time.monotonic() - at < self.ends_after:
            return {"state": "running"}
        if ("label lands", body["job"]) not in self.events:
            self.events.append(("label lands", body["job"]))
        return {"state": "cancelled"}


def ambient_ask(plan):
    """What an mm exec of the stand-in does during it: for a sign photo an ask with no project, whose labels are ambient."""
    words = by_qid(plan)

    def during(spec, q):
        if q.tags["template"] == "photo-sign":
            scan(Api(home_of(spec)), question=words[q.qid]["fake"]["question"], sources=["phone"], since=q.scope.since, until=q.scope.until)

    return during


def asks(spec, q, found):
    return [{"tool": "ask", "arguments": {"question": q.question}}, {"tool": "find", "arguments": {"scope": "all"}}] if spec.served.server else [
        {"tool": "shell", "arguments": {"command": "ls"}}]


@pytest.fixture(scope="module")
def ran(week, store, tiny, tmp_path_factory):
    """The stand-in over the tiny week: the files configuration, then `mm/fake` and `mm/fake+files` of the fake decider, two execs at once."""
    recipe, dump, questions = week
    root = tmp_path_factory.mktemp("ran")
    with pytest.MonkeyPatch.context() as patch:
        made_up_ffmpeg(patch, root)
        run = run_of(root, recipe, questions, dump, [FILES, S, S_FILES])
        probe = Probe(questions, during=ambient_ask(tiny), tools=asks, answer=lambda spec, q: right(q))
        go(run, dump, store, probe)
        yield types.SimpleNamespace(run=run, probe=probe, dump=dump, questions=questions, root=root, store=store)


# -- a store is copied into a home, and a run over it is continued ------------------------------------------------------------------

def test_a_copy_of_the_store_has_the_decider_and_every_item_and_record_and_one_with_a_daemon_is_refused(week, store, tiny, tmp_path):
    snapshot = store / "home"
    home = copy_home(snapshot, tmp_path / "copy", FAKE_MODEL)
    assert config.load(home).decider_model == FAKE and copy_home(snapshot, home, FAKE_MODEL) == home
    daemon = Daemon(home).start()
    try:
        held = {s["name"]: s for s in daemon.api.call("status")["sources"]}
        for source in {f.source for f in tiny.files}:
            assert held[source]["items"] == sum(1 for f in tiny.files if f.source == source), source
        assert held["logs"]["records"] == sum(len(v) for v in tiny.logs.values())
        assert daemon.api.call("status")["decider"]["model"] == FAKE
    finally:
        daemon.stop()
    busy = tmp_path / "busy"
    shutil.copytree(snapshot, busy)
    (busy / "daemon.json").write_text("{}")
    with pytest.raises(RuntimeError, match="daemon"):
        copy_home(busy, tmp_path / "never", FAKE_MODEL)
    assert not (tmp_path / "never").exists() and not (tmp_path / "never.part").exists()


def test_every_configuration_ran_every_question_the_system_ones_in_a_project_of_their_own_and_a_second_command_asks_nothing(ran, tiny, week, store):
    run, probe, n = ran.run, ran.probe, len(ran.questions)
    assert n >= 15 and len(probe.execs) == 3 * n
    folders = [spec.run_dir.parent.name for spec in probe.execs]
    assert folders == ["files"] * n + ["mm-fake"] * n + ["mm-fake+files"] * n, "the configuration with no variant first, then the variant's, as given"
    for spec in probe.execs:
        folder, qid = spec.run_dir.parent.name, spec.run_dir.name
        assert spec.work.parent == run.temp and spec.work.name.startswith(f"mmb-{run.tag}-w") and spec.bin_dir == run.bin
        assert spec.own_tools == (folder != "mm-fake") and (spec.data == run.data) == spec.own_tools and (spec.server is None) == (folder == "files")
        assert probe.marks[folder, qid] == (f"bench-{qid}" if folder != "files" else None), "the system's slots held a project of their own"
        assert ("mm has recorded" in spec.prompt) == (folder != "files") and (str(run.data) in spec.prompt) == spec.own_tools
    both = next(s.prompt for s in probe.execs if s.run_dir.parent.name == "mm-fake+files")
    assert "Answer from what mm holds or from these files." in both
    for folder, config_ in (("files", FILES), ("mm-fake", S), ("mm-fake+files", S_FILES)):
        got = {q.qid: result(run, folder, q.qid) for q in ran.questions}
        assert sorted(p.stem for p in (run.folder / "results" / "claude" / folder).glob("*.json") if p.stem != "ambient-labels") == sorted(got)
        assert all(r["problem"] is None and r["agent"] == "claude" and r["config"] == config_.name for r in got.values())
        assert all((r["system"], r["variant"]) == (config_.system, config_.variant) and "arm" not in r and "model" not in r for r in got.values())
    sign = result(run, "mm-fake", "image-photo-sign-1")
    assert (sign["passed"], sign["abstained"], sign["seed"], sign["denials"]) == (True, False, 7, 0)
    assert (sign["calls"], sign["tools"], sign["scope_all"]) == (2, {"ask": 1, "find": 1}, 1)
    assert sign["exec"] == "execs/claude/mm-fake/image-photo-sign-1" and sign["agent_model"] == "stub" and sign["turns"] == 2
    assert (sign["exclusive"], sign["beyond_timeout"], sign["busy_at_start"], sign["cancelled"], sign["unended"], sign["orphans"]) == (
        False, False, None, [], [], 0)
    assert "queued_s" not in sign and sign["held_s"] >= 0 and sign["gpu_estimate_s"] > 0
    assert "gpu_estimate_s" not in result(run, "files", "image-photo-sign-1")

    # the label check is live: the stand-in's ask had no project, and the pinned ones have
    for folder in ("mm-fake", "mm-fake+files"):
        ambient = json.loads((run.folder / "results" / "claude" / folder / "ambient-labels.json").read_text(encoding="utf-8"))
        assert ambient and all(r["by"] == "ask" and r["project"] is None for r in ambient), folder
    assert not (run.folder / "results" / "claude" / "files" / "ambient-labels.json").exists()

    # a second command asks nothing; a deleted result is asked again alone
    again = Probe(ran.questions)
    go(run, ran.dump, store, again)
    assert again.execs == []
    gone = run.folder / "results" / "claude" / "mm-fake" / "audio-mic-alarm-1.json"
    gone.unlink()
    once = Probe(ran.questions, answer=lambda spec, q: right(q))
    with pytest.MonkeyPatch.context() as patch:
        made_up_ffmpeg(patch, ran.root)
        go(run, ran.dump, store, once)
    assert [(s.run_dir.parent.name, s.run_dir.name) for s in once.execs] == [("mm-fake", "audio-mic-alarm-1")] and gone.exists()
    assert not (run.folder / "homes" / "claude-fake" / "daemon.json").exists(), "the daemon of the home was stopped when the cell ended"


def test_the_system_ones_see_the_horizon_free_index_and_the_homes_are_one_for_each_agent_and_variant(ran):
    assert sorted(p.name for p in (ran.run.folder / "homes").iterdir()) == ["claude-fake"], "mm/fake and mm/fake+files share a variant's home"


def test_the_decider_tier_scores_the_fake_questions(week, store, tiny, tmp_path, capsys):
    recipe, dump, questions = week
    words = by_qid(tiny)
    table = json.loads((store / "windows.json").read_text(encoding="utf-8"))
    # the long ranges (a screen, a porch clip or sound for a week) take the fake seconds each, and add nothing the shorter ones do not prove:
    # every kind of question stays (a part, a day and a `last` week one; a count; the absent ones). The screen's part question stays whatever
    # its windows (a frame an act: thousands), so the tier's screen scoring is covered.
    windows = lambda q: sum(estimate.windows_in(table, q["source"], [(q["since"], q["until"])]).values())  # noqa: E731
    asked = [q for q in tiny.questions if q["oracle"] is not None and (windows(q) <= 60 or (q["source"] == "screen" and q["level"] == "part" and not q["absent"]))]
    assert any(q["source"] == "screen" and windows(q) > 60 for q in asked)
    assert {"part", "day", "week"} <= {q["level"] for q in asked} and any(q.get("last") for q in asked) and any(q["absent"] for q in asked)
    assert any(q["fake"]["check"] == "count" for q in asked)
    run = run_of(tmp_path, recipe, questions, dump, [FILES, S])
    fs.write_json(run.folder / "run.json", {"mmmb_commit": "abc123", "probe": {"status": "passed", "failures": []}})  # (what the report reads of a run)
    fs.write_json(run.folder / "systems.json", {"mm/fake": {"system": "mm", "variant": "fake", "dir": "mm-fake", "readable": sorted(MM.describe("fake").reads),
                                                       "asked": sorted(MM.describe("fake").reads)}})
    mine = [q for q in questions if q.qid in {a["qid"] for a in asked}]

    def fake_query(q):  # the fake's own words, as the old `ask_with="fake"`
        return dataclasses.replace(retrieval.query_of(q), text=words[q.qid]["fake"]["question"])

    with fakes.Opened(store, dump, run.folder / "homes") as session:
        began = time.monotonic()
        assert retrieval.run_retrieval(run, S, session, mine, warm_s=time.monotonic() - began, query=fake_query) == len(asked)
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("mm/fake ")]
        assert len(lines) == len(asked) and all(" false=" in line and " windows in " in line for line in lines), lines
        folder = run.folder / "results" / "retrieval" / "mm-fake"
        assert (folder / "warm.json").exists() and sorted(p.stem for p in folder.glob("*.json")) == sorted(["warm", *(q["qid"] for q in asked)])
        assert {q["template"] for q in tiny.questions} - {q["template"] for q in tiny.questions if q["oracle"] is not None} == {"config-asof"}, \
            "every question but the config version has an oracle"
        for q in asked:
            r = json.loads((folder / f"{q['qid']}.json").read_text(encoding="utf-8"))
            assert (r["qid"], r["config"], r["system"], r["variant"], r["type"], r["absent"]) == (q["qid"], "mm/fake", "mm", "fake", q["type"], q["absent"])
            assert (r["level"], r["last"]) == (q["level"], bool(q.get("last")))
            assert r["limit"] == (5 if q.get("last") else 50 if q["fake"]["check"] == "count" else 20), q["qid"]
            if q["absent"]:
                assert (r["found"], r["top5"], r["false"], r["in_range"], r["p"], r["rank"]) == (None, None, 0, 0, None, None), q["qid"]
            else:
                assert r["found"] is True and r["top5"] is True and r["in_range"] >= 1 and r["rank"] >= 1, q["qid"]
                assert r["p"] is not None and r["moments_found"] and r["false"] == 0, q["qid"]
            assert r["windows"] > 0 and r["seconds"] >= 0 and r["wall_s"] >= 0 and r["sightings"] is not None
        tier = retrieval.decider_figures([run])
        assert tier["missed"] == [] and tier["asks"] == len(asked) and tier["models"]["mm/fake"]["first_s"] is not None
        assert all(t["found"] == t["needles"] and t["absent_clean"] == t["absent"] for t in tier["models"]["mm/fake"]["types"].values())
        text = render.markdown(report.numbers([run]))
        images = sum(1 for q in asked if q["type"] == "image" and not q["absent"])
        assert f"| fake | image | {images}/{images} (100%)" in text and "Every template was found" in text

        def no_session(*args, **kwargs):
            raise AssertionError("a configuration with every record is asked nothing")

        session.oracle = no_session
        assert retrieval.run_retrieval(run, S, session, mine, query=fake_query) == 0


# -- the models a variant can be -----------------------------------------------------------------------------------------------------

def test_models_are_resolved_and_the_ones_that_are_not_ready_say_why(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    model_folder(folder, "d1-3B")
    model_folder(folder, "h2o-lightning-4b")
    monkeypatch.setattr(models.modelenv, "problem", lambda backend, folder_, envs: None)
    monkeypatch.setattr(models, "serving", lambda url: False)
    ready, skipped = resolve_models(["d1-3b", "D3-EDGE", "h2o-lightning-4b", "nonsense", "d1-3B"], folder)
    assert [m.name for m in ready] == ["d1-3B"] and ready[0].spec == (folder / "d1-3B").resolve().as_posix()
    assert ready[0].url is None and ready[0].slug == "d1-3B" and ready[0].about is decider.known("d1-3B")[1]
    reasons = dict(line.split(": ", 1) for line in skipped)
    assert "mm model use d3-edge" in reasons["d3-edge"] and "not serving" in reasons["h2o-lightning-4b"]
    assert "not a model mm knows" in reasons["nonsense"] and len(skipped) == 3, "d1-3B named twice is one model"
    monkeypatch.setattr(models, "serving", lambda url: True)
    served, _ = resolve_models(["h2o-lightning-4b"], folder)
    assert served[0].url == "http://127.0.0.1:8741" and not needs_gpu(served[0])
    monkeypatch.setattr(models.modelenv, "problem", lambda backend, folder_, envs: "torch is missing")
    assert "env is not ready: torch is missing" in resolve_models(["d1-3B"], folder)[1][0]
    assert needs_gpu(ready[0]) and not needs_gpu(FAKE_MODEL)
    assert Model("pplx-decider-flash-v1.2", "x", ready[0].about).slug == "pplx-decider-flash-v1-2"
    # the system: a variant it was given is that model, any other has its facts whether or not it can run (what `describe` and `preview` need),
    # and a session is opened only over one that is ready, the reason it is not being the error (G1's F9)
    monkeypatch.setattr(models.modelenv, "problem", lambda backend, folder_, envs: None)
    mm =system.Mm(models={"fake": FAKE_MODEL}, models_dir=folder)
    assert mm.model_of("fake") is FAKE_MODEL and mm.model_of("d1-3B").spec == (folder / "d1-3B").resolve().as_posix()
    assert mm.ready_model("fake") is FAKE_MODEL and mm.ready_model("d1-3B") == mm.model_of("d1-3B")
    assert mm.model_of("d3-edge").name == "d3-edge", "its facts, though it is not downloaded"
    with pytest.raises(ValueError, match="d3-edge: not downloaded: mm model use d3-edge"):
        mm.ready_model("d3-edge")
    with pytest.raises(ValueError, match="mm has variants"):
        mm.model_of(None)
    # what the system says of a model: it needs the GPU unless it is a test decider or a server, and a store is not the model's
    assert (mm.describe("d1-3B").needs_gpu, mm.describe("fake").needs_gpu, mm.describe("fake").has_oracle) == (True, False, True)
    assert mm.describe("d1-3B").store_hash == mm.describe("fake").store_hash != mm.describe("d1-3B").config_hash != mm.describe("fake").config_hash
    assert mm.describe("d1-3B").reads == frozenset({"text", "records", "image", "video"}) and mm.describe("d1-3B").horizon == "cut"
    assert mm.describe("d1-3B").license == "see Multimem" and mm.describe("d1-3B").version == system.version_of()


# -- an oracle ask: what the query passes, and what it gets back -------------------------------------------------------------------

DAY = 86400.0
T0 = datetime(2026, 10, 30, 12, 0).timestamp()


class Scripted:
    """An API that answers each ask job at once with `result`, and keeps the parameters it was given."""

    def __init__(self, result: dict):
        self.result, self.params = result, []

    def call(self, op: str, **body) -> dict:
        assert op == "jobs/start" and body["script"] == "ask"
        self.params.append(body["params"])
        return {"state": "done", "job": "j-1", "error": None, "result": self.result}


def asked(sightings=(), candidates=(), windows=40, seconds=2.5, stopped="range_end") -> dict:
    """What an ask job's result holds: sightings `(t0, t1, p)`, candidates `(t, p)`, and its progress. A sighting of one instant
    has `t`, as the API gives it."""
    return {"sightings": [{"t": config.iso(a), "p": p, "source": "phone"} if a == b else {"t0": config.iso(a), "t1": config.iso(b), "p": p, "source": "phone"}
                          for a, b, p in sightings],
            "candidates": [{"t": config.iso(t), "p": p, "source": "phone"} for t, p in candidates], "stopped": stopped, "warnings": [],
            "progress": {"windows": windows, "seconds": seconds}}


def query(text: str = "yes?", ranges=((T0 - 5 * DAY, T0 + 6 * 3600),), k: int = 20, oldest: bool = False) -> RetrievalQuery:
    return RetrievalQuery("q", text, ("phone",), tuple(ranges), k, 5, oldest, 1800.0)


def test_an_oracle_ask_passes_the_query_as_the_level_rules_made_it_and_gives_back_the_moments_it_found():
    truth = T0 + 3600
    # a `last` question asks for 5 sightings, newest first, with the 1,800 s any oracle ask is allowed: `oldest_first` is left out of it
    api = Scripted(asked([(truth - 2 * DAY, truth - 2 * DAY, 0.91), (truth + 4, truth + 4, 0.83)], [(truth + 3, 0.7)], stopped="limit"))
    got = retrieve(api, query(k=5, oldest=False))
    assert api.params[0] == {"question": "yes?", "sources": ["phone"], "since": T0 - 5 * DAY, "until": T0 + 6 * 3600, "limit": 5, "candidates": 5,
                             "max_s": 1800}, "newest first: `oldest_first` is left out of a `last` ask"
    assert [(m.t0, m.t1, m.p, m.source) for m in got.sightings] == [(pytest.approx(truth - 2 * DAY, abs=0.001),) * 2 + (0.91, "phone"),
                                                                    (pytest.approx(truth + 4, abs=0.001),) * 2 + (0.83, "phone")]
    assert (got.stopped, got.windows, got.seconds, got.warnings) == ("limit", 40, 2.5, ()) and len(got.candidates) == 1
    # the others (a part, a day, a count): their limit, `oldest_first` kept
    api = Scripted(asked())
    retrieve(api, query(k=50, oldest=True))
    assert api.params[0]["limit"] == 50 and api.params[0]["oldest_first"] is True and api.params[0]["max_s"] == 1800
    # a query of several ranges (EgoLife's) is an ask for each, its windows and seconds added and its stops: `max_s` if any, else the first
    api = Scripted(asked([(truth, truth, 0.9)], windows=40, seconds=2.5, stopped="max_s"))
    got = retrieve(api, query(ranges=((T0, T0 + 600), (T0 + 3000, T0 + 3600))))
    assert [(p["since"], p["until"]) for p in api.params] == [(T0, T0 + 600), (T0 + 3000, T0 + 3600)] and len(got.sightings) == 2
    assert (got.windows, got.seconds, got.stopped) == (80, 5.0, "max_s")


def test_a_sighting_or_candidate_has_one_time_or_a_span_and_the_moments_read_both():
    at = T0 + 3600
    one = moment_of({"t": config.iso(at), "p": 0.7, "source": "phone"})
    assert (one.t0, one.t1) == (pytest.approx(at, abs=0.001),) * 2 and one.p == 0.7 and one.raw == {"t": config.iso(at), "p": 0.7, "source": "phone"}
    span_ = moment_of({"t0": config.iso(at), "t1": config.iso(at + 5), "p": 0.6, "source": "screen"})
    assert (span_.t0, span_.t1, span_.source) == (pytest.approx(at, abs=0.001), pytest.approx(at + 5, abs=0.001), "screen")
    moments = [[at, at]]
    assert retrieval.needle_p(moments, [moment_of({"t": config.iso(at + 3), "p": 0.7, "source": "phone"})], [], 10.0) == 0.7, "a sighting of one instant"
    assert retrieval.needle_p(moments, [], [moment_of({"t0": config.iso(at - 4), "t1": config.iso(at - 1), "p": 0.5, "source": "phone"})], 10.0) == 0.5, \
        "a candidate that spans"


# -- how many execs at once, and what a window costs --------------------------------------------------------------------------------

def model_of(name: str) -> Model:
    return Model(name, f"/models/{name}", decider.known(name)[1])


def test_a_model_slower_than_015_s_a_window_runs_one_exec_at_a_time_and_the_fake_one_as_many_as_asked():
    # a model above 0.15 s a window (mm's estimate from its About) runs one exec at a time
    jobs = {n: arm_jobs(model_of(n), 3) for n in ("d3-edge", "d1-3B", "h2o-lightning-4b", "d3-mini", "d3-flash", "pplx-decider-flash-v1.2")}
    assert jobs == {"d3-edge": 3, "d1-3B": 3, "h2o-lightning-4b": 3, "d3-mini": 1, "d3-flash": 1, "pplx-decider-flash-v1.2": 1}
    assert 0.15 < window_s(decider.known("d3-mini")[1]) < 0.2 and arm_jobs(FAKE_MODEL, 3) == 3


def test_a_session_runs_one_exec_at_a_time_for_a_slow_model_and_for_a_horizon_and_the_requested_number_else(week, store, tmp_path):
    _, dump, questions = week
    cases = [Case(q.qid, 1.0, None, q.scope.sources, 0.0, 1.0, q.group, 0.0, -1, Path(), Path()) for q in questions]
    with fakes.Opened(store, dump, tmp_path / "fast") as session:
        assert session.max_jobs(cases, 3) == 3 and session.max_jobs(cases, 1) == 1
        assert session.max_jobs([dataclasses.replace(cases[0], horizon=5.0), *cases[1:]], 3) == 1, "a horizon cuts the whole home"
        assert session.max_jobs([], 3) == 3
    slow = Mm_with(decider.known("d3-mini")[1])
    with fakes.Opened(store, dump, tmp_path / "slow", system=slow, variant="m") as session:
        assert session.max_jobs(cases, 3) == 1


def Mm_with(about):
    return system.Mm(models={"m": Model("m", FAKE, about)})


# -- how an exec worked ----------------------------------------------------------------------------------------------------------

def test_how_the_ask_jobs_worked_is_read_from_the_calls_whatever_the_agent(tmp_path):
    work = work_of(CODEX_MM)
    # a job counts once, by its last result (the second call collected the first's), and a job still running says so
    assert work == {"jobs": 3, "windows": 135, "seconds": 10.9, "stopped": {"max_s": 1, "limit": 1, "running": 1}}
    assert work_of(CODEX_FILES) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}}
    assert work_of(CLAUDE_FILES) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}}
    assert work_of(CLAUDE_MM) == {"jobs": 2, "windows": 57, "seconds": 3.5, "stopped": {"range_end": 1, "cancelled": 1}}
    assert work_of([]) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}}
    # a result that is not JSON, or not an object, or an ask that gave none, counts as no job
    for result_ in (None, "", "ask failed", "[1, 2]", [], [{"type": "image"}], {"content": None}, 7):
        assert work_of([call("ask", {"question": "?"}, result_)])["jobs"] == 0


def test_a_result_holds_what_the_audit_adds_to_the_core_s_counts(week, store, tmp_path):
    _, dump, _ = week
    with fakes.Opened(store, dump, tmp_path / "homes") as session:
        case = Case("video-screen-error-1", T0, None, ("screen",), T0, T0 + 1, "g", 0.0, 0, tmp_path / "w", tmp_path / "r")
        audit = session.audit(case, CODEX_MM)
        assert audit == {"work": {"jobs": 3, "windows": 135, "seconds": 10.9, "stopped": {"max_s": 1, "limit": 1, "running": 1}}}, \
            "no horizon fields for a case with no horizon: V7's results never had them"
        assert session.usage(case, CODEX_MM).gpu_s == 10.9 and session.queue_s(case) is None


# -- what an exec leaves in the home, and who sees it -------------------------------------------------------------------------------

class Probing:
    """An mm agent for the system's scheduler and isolation: it looks, with scope all, at the labels in its home over the whole
    week, asks the fake decider its question under its own project (which labels each sighting), once marks the phone's items
    kept with `label`, looks again, and answers right. It keeps what it saw and the span of its run."""

    def __init__(self, questions, plan, pause: float = 0.0, keep_in: str | None = None):
        self.questions, self.words = {q.qid: q for q in questions}, by_qid(plan)
        self.week = (min(q.scope.since for q in questions), max(q.scope.until for q in questions))
        self.pause, self.keep_in = pause, keep_in
        self.seen, self.after, self.spans, self.kept = {}, {}, {}, {}
        self.lock = threading.Lock()

    def labels(self, api) -> list[dict]:
        found = api.call("find", types=["label"], scope="all", since=self.week[0], until=self.week[1], limit=200)["results"]
        return [{"project": r.get("project"), "by": r.get("by"), "text": r["text"]} for r in found if r.get("by") in ("ask", "agent")]

    def __call__(self, spec) -> dict:
        qid = spec.run_dir.name
        q, home = self.questions[qid], home_of(spec)
        api, project = Api(home), exec_project(qid)
        began = time.monotonic()
        before = self.labels(api)
        ask(api, ask_params(dataclasses.replace(retrieval.query_of(q), text=self.words[qid]["fake"]["question"]), q.scope.since, q.scope.until), project)
        during = None
        if qid == self.keep_in:
            api.call("label", text="mine", since=q.scope.since, until=q.scope.until, source=q.scope.sources[0], keep=True, caller={"project": project})
            during = kept_items(home)
        time.sleep(self.pause)
        after = self.labels(api)
        with self.lock:
            self.seen[qid], self.after[qid], self.kept[qid] = before, after, during
            self.spans[qid] = (began, time.monotonic())
        return {"answer": right(q), "final": f"ANSWER: {right(q)}", "tools": [], **clocked(spec, 1.0), "problem": None, "turns": 1,
                "tokens": {"input": 10, "output": 1, "cache_read": 0, "cache_write": 0}, "cost_usd": None, "used_percent": 1.0,
                "model": "stub", "denials": 0}


def labels_left(home: Path, pattern: str = "id:bench-%") -> tuple[int, int, int]:
    """How many labels, project rows and session rows of the bench's execs an mm home holds."""
    with closing(sqlite3.connect(home / "index.sqlite")) as db:
        return (db.execute("SELECT COUNT(*) FROM labels WHERE project LIKE ?", (pattern,)).fetchone()[0],
                db.execute("SELECT COUNT(*) FROM projects WHERE id LIKE ?", (pattern,)).fetchone()[0],
                db.execute("SELECT COUNT(*) FROM sessions WHERE project LIKE ?", (pattern,)).fetchone()[0])


def test_no_exec_sees_a_label_an_earlier_exec_left_and_a_long_exec_holds_the_gpu_alone(week, store, tiny, tmp_path, monkeypatch):
    recipe, dump, questions = week
    run = run_of(tmp_path, recipe, questions, dump, [S])
    used: set[str] = set()

    def pick(*templates: str) -> list:
        out = []
        for template in templates:
            q = next(q for q in questions if q.tags["template"] == template and q.qid not in used)
            used.add(q.qid)
            out.append(q)
        return out

    first = pick("photo-sign", "screen-error")
    second = pick("note-code", "porch-parcel", "mic-doorbell", "photo-sign")
    third = pick("log-count", "absent-log", "log-last", "photo-count", "absent-note", "absent-sign", "absent-doorbell")
    description = MM.describe("fake")
    isolation = systems.isolation()
    with fakes.Opened(store, dump, tmp_path / "homes") as session:
        assert all(session.estimate(q.scope.sources, q.cost_ranges) <= HEAVY_S / 4 for q in [*first, *second, *third]), \
            "all cheap to ask (the screen's frames count: a part is 47 s): none holds the GPU alone"
        home = session.home
        baseline = kept_items(home)

        def arm(agent, questions_, jobs):
            run_config(run, "claude", Agent(agent), S, description, questions_, session, dump, isolation, jobs, Halt(), "line")

        # without the scrub, the later of two execs finds the earlier one's label (so the test below can fail)
        leaky = Probing(first, tiny)
        with monkeypatch.context() as patch:
            patch.setattr(system, "scrub_home", lambda home_, kept, **kw: {})
            arm(leaky, first, 1)
        early, late = sorted(first, key=lambda q: (q.scope.until, q.qid))
        assert leaky.seen[early.qid] == [] and {r["project"] for r in leaky.seen[late.qid]} == {exec_project(early.qid)}
        swept = scrub_home(home, baseline, prefix=PROJECT_PREFIX)
        assert swept["labels"] >= 2 and labels_left(home) == (0, 0, 0), "a sweep takes out what an interrupted run left"
        for q in first:
            (run.folder / "results" / "claude" / "mm-fake" / f"{q.qid}.json").unlink()

        # with it, an exec starts from the store's state: it sees no label of an exec that ended, and leaves none
        agent = Probing(second, tiny, keep_in=second[-1].qid)
        arm(agent, second, 1)
        for q in second:
            own = [r for r in agent.after[q.qid] if r["project"] == exec_project(q.qid)]
            rec = result(run, "mm-fake", q.qid)
            assert agent.seen[q.qid] == [], q.qid
            assert own and {r["project"] for r in agent.after[q.qid]} == {exec_project(q.qid)}, "it sees its own label, and no other"
            assert rec["left"] == {"labels": len(own), "unkept": len(agent.kept[q.qid] - baseline) if q.qid == agent.keep_in else 0}
        assert agent.kept[agent.keep_in] > baseline and result(run, "mm-fake", agent.keep_in)["left"]["unkept"] > 0, "it kept items"
        assert kept_items(home) == baseline and labels_left(home) == (0, 0, 0)
        assert next(r for r in agent.after[agent.keep_in] if r["by"] == "agent")["text"] == "mine"

        # the scheduler: two long execs (by an estimate of 400 s) run alone, two of one template never run together, and the others stay
        # parallel; each result holds its estimate, its wait and whether it ran alone
        long_ones = {"image-photo-count-1", "records-log-last-1"}
        long_ranges = {(q.scope.sources, q.cost_ranges) for q in third if q.qid in long_ones}
        assert len(long_ranges) == 2 and not any((q.scope.sources, q.cost_ranges) in long_ranges for q in third if q.qid not in long_ones)
        monkeypatch.setattr(session, "estimate", lambda sources, ranges: 400.0 if (tuple(sources), tuple(map(tuple, ranges))) in long_ranges else 20.0)
        slow = Probing(third, tiny, pause=0.15)
        arm(slow, third, 3)
        spans = slow.spans
        assert long_ones <= set(spans), "the long ones are in this part"

        def together(a: str, b: str) -> bool:
            return spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1]

        assert not any(together(h, x) for h in long_ones for x in spans if x != h), "an exec that holds the GPU runs alone"
        log_count, absent_log = (q.qid for q in third[:2])
        assert not together(log_count, absent_log), "a template's needle and the absent question in its words never run together"
        light = [q.qid for q in third if q.qid not in long_ones]
        assert any(together(a, b) for i, a in enumerate(light) for b in light[i + 1:]), "the others stay parallel"
        for q in third:
            rec = result(run, "mm-fake", q.qid)
            assert (rec["gpu_estimate_s"], rec["exclusive"]) == ((400.0, True) if q.qid in long_ones else (20.0, False))
            assert rec["held_s"] >= 0 and rec["left"]["labels"] >= 0 and rec["problem"] is None and "queued_s" not in rec
            assert (rec["cancelled"], rec["unended"], rec["orphans"], rec["beyond_timeout"]) == ([], [], 0, False)
        assert labels_left(home) == (0, 0, 0) and result(run, "mm-fake", third[0].qid)["held_s"] < 60
        assert all(result(run, "mm-fake", h)["busy_at_start"] == 0 for h in long_ones), "a long exec starts with the daemon idle"
        assert result(run, "mm-fake", third[0].qid)["busy_at_start"] is None


def test_the_baseline_of_the_keeps_is_the_stores_and_not_the_live_homes(week, store, tmp_path, monkeypatch):
    _, dump, _ = week
    bases = []
    monkeypatch.setattr(system, "kept_items", lambda home: bases.append(home) or set())
    with fakes.Opened(store, dump, tmp_path / "homes") as session:
        assert bases == [store / "home"] and session.kept == set()
        assert session.home == tmp_path / "homes" / "claude-fake" != store / "home"


def test_as_of_questions_run_one_at_a_time_in_time_order_and_the_rest_in_parallel(week, store, tiny, tmp_path):
    recipe, dump, questions = week
    run = run_of(tmp_path, recipe, questions, dump, [S])
    slow = Probe(questions, pause=0.15)
    with pytest.MonkeyPatch.context() as patch:
        made_up_ffmpeg(patch, tmp_path)
        go(run, dump, store, slow, jobs=3)
    mm = sorted((x for x in slow.log if x[0] == "mm-fake"), key=lambda x: x[3])
    as_of = [x for x in mm if x[1] in AS_OF]
    assert {x[1] for x in as_of} == set(AS_OF) and len(as_of) == sum(1 for q in tiny.questions if q["source"] in AS_OF)
    assert all(a[4] <= b[3] for a, b in zip(as_of, as_of[1:])), "two file-version questions were running at once"
    until = {s.run_dir.name: q.scope.until for s in slow.execs for q in questions if q.qid == s.run_dir.name}
    assert [until[x[2]] for x in as_of] == sorted(until[x[2]] for x in as_of), "as-of questions in `until` order"
    others = [x for x in mm if x[1] not in AS_OF]
    assert any(a[3] < b[4] and b[3] < a[4] for i, a in enumerate(others) for b in others[i + 1:]), "the others stay parallel"
    assert any(a[3] < b[4] and b[3] < a[4] for a in as_of for b in others), "and run beside a file-version question"


# -- the jobs an exec left behind ------------------------------------------------------------------------------------------------------

def test_a_check_in_that_leaves_out_what_did_not_change_still_counts_for_its_job():
    """A later reply holds `job`, `state` and `progress` (and `stopped` once set); the fields that are None or unchanged are gone."""
    first = job_result("j-a", "running", 10, 1.0)
    first.pop("error")
    check_in = {"job": "j-a", "state": "running", "progress": {"windows": 40, "sightings": 1, "of_about": 121, "seconds": 2.5, "decider": "ready"},
                "next": "Call ask"}
    last = {"job": "j-a", "state": "done", "stopped": "limit", "progress": {"windows": 61, "sightings": 1, "of_about": 121, "seconds": 3.5},
            "sightings": [{"source": "screen", "t": "2026-11-01T18:00:03.000-05:00", "p": 0.9, "kind": "video", "look": {"item": 1, "at": 3.0}}]}
    tools = [call("ask", {"question": "?"}, mm_result(first)), call("ask", {"job": "j-a"}, mm_result(check_in))]
    assert open_jobs(tools) == ["j-a"]
    assert work_of(tools) == {"jobs": 1, "windows": 40, "seconds": 2.5, "stopped": {"running": 1}}
    tools.append(call("ask", {"job": "j-a"}, mm_result(last)))
    assert open_jobs(tools) == []
    assert work_of(tools) == {"jobs": 1, "windows": 61, "seconds": 3.5, "stopped": {"limit": 1}}
    all_of_it = call("ask", {"job": "j-a", "all": True}, mm_result({**last, "searched": {}, "candidates": [], "warnings": []}))
    assert work_of([*tools[:2], all_of_it])["windows"] == 61, "a reply for `all` is a reply like another"


def test_ask_jobs_an_exec_left_running_are_cancelled_when_it_ends():
    left_running = call("ask", {"question": "?"}, mm_result(job_result("j-a", "running", 10, 1.0)))
    collected = [call("ask", {"question": "?"}, mm_result(job_result("j-b", "running", 5, 0.5))),
                 call("ask", {"job": "j-b"}, mm_result(job_result("j-b", "done", 50, 3.0, "range_end")))]
    queued = call("ask", {"question": "?"}, mm_result(job_result("j-c", "queued", 0, 0.0)))
    cancelled = call("ask", {"job": "j-d"}, mm_result(job_result("j-d", "cancelled", 3, 0.2, "cancelled")))
    not_asks = [call("ask", {"question": "?"}, "ask failed: bad_request"), call("find", {"sources": ["phone"]}, mm_result({"job": "j-z", "state": "running"}))]
    tools = [left_running, *collected, queued, cancelled, *not_asks]
    assert open_jobs(tools) == ["j-a", "j-c"] and open_jobs([]) == []

    class Daemon_:
        def __init__(self):
            self.calls = []

        def call(self, op, **body):
            self.calls.append((op, body))
            if body["job"] == "j-c":
                raise RuntimeError("not_found: the job ended")
            return {}

    api = Daemon_()
    assert cancel_open(api, tools) == ["j-a"] and api.calls == [("jobs/cancel", {"job": "j-a"}), ("jobs/cancel", {"job": "j-c"})]
    assert cancel_open(Daemon_(), [collected[1]]) == []


def test_an_exec_that_ended_waits_for_its_jobs_and_for_the_daemon_to_come_down_to_the_execs_still_running():
    open_ask = call("ask", {"question": "?"}, mm_result(job_result("j-late", "running", 10, 1.0)))
    # settle: a cancelled job that stops late is waited for, one that never stops is named, an unknown one is gone
    api = JobApi()
    api.call("jobs/cancel", job="j-late")
    began = time.monotonic()
    assert settle(api, ["j-late", "j-gone"], 5) == [] and time.monotonic() - began >= 0.35
    assert api.events == [("label lands", "j-late")]
    stuck = JobApi(ends_after=99)
    stuck.call("jobs/cancel", job="j-late")
    assert settle(stuck, ["j-late"], 0.3) == ["j-late"]
    # wait_idle: down to the allowed count, or how many are over it, or unreadable
    assert wait_idle(JobApi(counts=[2, 0]), 0, 5) == 0 and wait_idle(JobApi(counts=[1]), 1, 5) == 0
    assert wait_idle(JobApi(counts=[3]), 1, 0.3) == 2 and wait_idle(JobApi(counts=[4]), 0, 0) == 4

    class Unreadable:
        def __init__(self, answer):
            self.answer = answer

        def call(self, op, **body):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    assert wait_idle(Unreadable(RuntimeError("gone")), 0, 1) is None and wait_idle(Unreadable({"jobs": {"running": None}}), 0, 1) is None
    # quiesce: cancel, wait for the job, then for the count
    events = []
    got = quiesce(JobApi(events, counts=[1, 0]), [open_ask])
    assert got["cancelled"] == ["j-late"] and got["unended"] == [] and got["orphans"] == 0 and got["settle_s"] >= 0.35
    assert events == [("label lands", "j-late")], "it returned after the label had landed"
    # a job whose id was never logged (the exec timed out inside the ask that started it) is only in the count
    assert quiesce(JobApi(counts=[1]), [], others=0, limit_s=0.3)["orphans"] == 1
    assert quiesce(JobApi(counts=[1]), [], others=1, limit_s=0.3)["orphans"] == 0, "one for the exec still running"
    assert quiesce(JobApi(counts=[3]), [], others=1, limit_s=0.3)["orphans"] == 2
    unended = quiesce(JobApi(ends_after=99), [open_ask], limit_s=0.3)
    assert (unended["cancelled"], unended["unended"]) == (["j-late"], ["j-late"])


def test_the_next_exec_of_a_template_starts_after_the_late_label_of_the_last_one_has_landed_and_been_scrubbed(week, tmp_path, monkeypatch):
    """The session over a home that is not there and a daemon that is scripted (`JobApi`): the order of what the system does when an exec ends is
    the point, and the core's loop is the real one."""
    recipe, dump, questions = week
    run = run_of(tmp_path, recipe, questions, dump, [S])
    a, b = sorted((next(q for q in questions if q.tags["template"] == t) for t in ("log-count", "absent-log")), key=lambda q: (q.scope.until, q.qid))
    other = next(q for q in questions if q.tags["template"] == "mic-doorbell")
    events = []
    api = JobApi(events, counts=[0])
    monkeypatch.setattr(system, "Api", lambda home: api)
    monkeypatch.setattr(system, "scrub_home", lambda home, kept, **kw: events.append(("scrub", tuple(kw.get("projects", ())))) or {"labels": 0, "unkept": 0})
    session = scripted_session(tmp_path)
    monkeypatch.setattr(session, "estimate", lambda sources, ranges: 10.0)

    def agent(spec):
        qid = spec.run_dir.name
        events.append(("start", qid))
        time.sleep(0.1)
        events.append(("end", qid))
        tools = [call("ask", {"question": "?"}, mm_result(job_result("j-late", "running", 10, 1.0)))] if qid == a.qid else []
        return {"final": f"ANSWER: {right(next(q for q in questions if q.qid == qid))}", "tools": tools, **clocked(spec, 1.0), "problem": None, "turns": 1,
                "tokens": None, "cost_usd": None, "used_percent": 1.0, "model": "stub", "denials": 0}

    description, isolation = MM.describe("fake"), systems.isolation()
    run_config(run, "claude", Agent(agent), S, description, [b, a, other], session, week[1], isolation, 3, Halt(), "line")
    at = {event: i for i, event in enumerate(events)}
    mine = ("scrub", (exec_project(a.qid),))
    assert at[("end", a.qid)] < at[("label lands", "j-late")] < at[mine] < at[("start", b.qid)], events
    assert at[("start", other.qid)] < at[("label lands", "j-late")], "a question of another template is not held by it"
    assert [call_[0] for call_ in api.calls if call_[1].get("job") == "j-late"][:2] == ["jobs/cancel", "jobs/get"]
    rec_a, rec_b, rec_o = (result(run, "mm-fake", q.qid) for q in (a, b, other))
    assert (rec_a["cancelled"], rec_a["unended"], rec_a["orphans"]) == (["j-late"], [], 0) and rec_b["cancelled"] == []
    # the wait is how long the question was held back with a slot free
    assert rec_a["held_s"] == 0.0 and rec_b["held_s"] >= 0.3 and rec_o["held_s"] == 0.0
    assert (rec_a["beyond_timeout"], rec_a["busy_at_start"], rec_o["exclusive"]) == (False, None, False) and "queued_s" not in rec_a
    assert [c for c in api.calls if c[0] == "status"], "the daemon's jobs were counted"
    # a long exec, past the exec's timeout, starts when the daemon is idle: it waited for two jobs to end; and says how many were left
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    monkeypatch.setattr(session, "estimate", lambda sources, ranges: 1000.0)
    waited = JobApi(counts=[2, 0])
    monkeypatch.setattr(system, "Api", lambda home: waited)
    run_config(run, "claude", Agent(agent), S, description, [other], session, week[1], isolation, 1, Halt(), "line")
    rec_o = result(run, "mm-fake", other.qid)
    assert (rec_o["gpu_estimate_s"], rec_o["exclusive"], rec_o["beyond_timeout"], rec_o["busy_at_start"]) == (1000.0, True, True, 0)
    assert report.worked([rec_a, rec_o])["beyond"] == 1 and render.cell_text(report.cell_of(rec_o), "asks") == "pass 1 s a0 (est. past timeout)"
    # past the run's timeout, not the module's: a run that allows 1,500 s (an ask may have 1,125) does not flag 1,000
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    monkeypatch.setattr(system, "Api", lambda home: JobApi(counts=[0]))
    slower = scripted_session(tmp_path, timeout_s=1500.0)
    monkeypatch.setattr(slower, "estimate", lambda sources, ranges: 1000.0)
    run_config(dataclasses.replace(run, timeout_s=1500.0), "claude", Agent(agent), S, description, [other], slower, week[1], isolation, 1, Halt(), "line")
    assert result(run, "mm-fake", other.qid)["beyond_timeout"] is False
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    busy = JobApi(counts=[3])
    monkeypatch.setattr(system, "Api", lambda home: busy)
    monkeypatch.setattr(system, "JOB_WAIT_S", 0.3)
    run_config(run, "claude", Agent(agent), S, description, [other], session, week[1], isolation, 1, Halt(), "line")
    assert result(run, "mm-fake", other.qid)["busy_at_start"] == 3


class NoDaemon:
    """The daemon of a session that has none: it is never stopped."""

    def stop(self) -> None:
        pass


def scripted_session(root: Path, timeout_s: float = 900.0) -> MmSession:
    """A session over a home that is not there, for the tests whose daemon is `JobApi`."""
    return MmSession(FAKE_MODEL, root / "no-home", NoDaemon(), set(), {"sources": {}, "decode_s": {}}, frozenset(), ("skill",),
                     context(root / "homes", timeout_s=timeout_s))


def test_a_long_compact_ask_result_is_not_hashed_so_its_job_and_its_windows_are_still_read():
    said = "Isthereaparceltothefrontdoor?:yes(p0.97)," + "x" * 100
    text = json.dumps({"job": "j-big", "state": "running", "sightings": [{"said": said, "p": 0.97}] * 40,
                       "progress": {"windows": 40, "of_about": 100, "seconds": 2.5}}, separators=(",", ":"))
    assert len(text) > 3000 and " " not in text[:200], "a compact result, the first said with no space in it"
    result_ = {"content": [{"type": "text", "text": text}], "structured_content": None}
    assert scrubbed(result_) == result_ and scrubbed([{"type": "text", "text": text}]) == [{"type": "text", "text": text}]
    tools = [call("ask", {"question": "?"}, scrubbed(result_))]
    assert open_jobs(tools) == ["j-big"] and work_of(tools)["windows"] == 40 and work_of(tools)["seconds"] == 2.5
    # a picture's base64 and a data URL still are
    assert scrubbed("A" * 5000).startswith("<5000 characters, sha1 ") and scrubbed("data:image/png;base64," + "A" * 5000).startswith("<")
    assert scrubbed(["QUJD" * 2000])[0].startswith("<8000 characters") and scrubbed("word " * 2000) == "word " * 2000
    assert scrubbed("[" + "1," * 3000 + "1]") == "[" + "1," * 3000 + "1]"


def test_the_flag_for_an_ask_that_cannot_finish_is_set_past_the_share_of_the_execs_time_the_run_allows():
    # an ask that fits the exec's 900 s but not with the agent's own turns (the screen's day asks of d1-3B at pilot scale: 880 s) is flagged; 670 s is not
    assert [beyond_timeout(s, 900.0) for s in (988.0, 700.0, 675.0, 670.0, 5.0)] == [True, True, False, False, False], "past 675 s of 900"
    assert beyond_timeout(1000.0, 1500.0) is False and beyond_timeout(1200.0, 1500.0) is True, "past the run's timeout, not the module's"
    assert serve_record(988.0, 900.0, None) == {"gpu_estimate_s": 988.0, "exclusive": True, "beyond_timeout": True, "busy_at_start": None}
    assert serve_record(299.96, 900.0, 0)["exclusive"] is False and serve_record(300.0, 900.0, 0)["exclusive"] is True
    assert serve_record(12.345, 900.0, 0)["gpu_estimate_s"] == 12.3


def test_a_home_links_the_stores_media_and_copies_the_rest(store, tmp_path):
    snapshot = store / "home"
    home = copy_home(snapshot, tmp_path / "linked", FAKE_MODEL)
    media = [p for folder in MEDIA_FOLDERS for p in (snapshot / folder).rglob("*") if p.is_file()]
    assert media, "the store holds the media it was loaded with"
    for path in media:
        twin = home / path.relative_to(snapshot)
        assert os.path.samefile(path, twin), path
        assert not os.access(path, os.W_OK), "the store's media is read-only, and a link shares that"
        with pytest.raises(PermissionError):
            twin.open("ab")
    index = home / "index.sqlite"
    assert index.is_file() and not os.path.samefile(index, snapshot / "index.sqlite"), "the index is a copy"
    assert config.load(home).decider_model == FAKE
    others = [p for p in snapshot.rglob("*") if p.is_file() and p.relative_to(snapshot).parts[0] not in MEDIA_FOLDERS
              and (home / p.relative_to(snapshot)).exists()]
    assert others and not any(os.path.samefile(p, home / p.relative_to(snapshot)) for p in others), "all else is copied"
    assert not any((home / name).exists() for name in ("daemon.json", "fake-decider.jsonl")) and "looks" in COPY_SKIPS
    # the copy's index is its own: a write there leaves the store's
    before = kept_items(snapshot)
    with closing(sqlite3.connect(index)) as db:
        db.execute("UPDATE items SET keep = 1")
        db.commit()
    assert kept_items(snapshot) == before != kept_items(home)


def test_the_windows_table_is_kept_beside_the_store_s_home_and_not_in_a_copy_of_it(store, week, tmp_path):
    assert (store / "windows.json").is_file() and not (store / "home" / "windows.json").exists()
    home = copy_home(store / "home", tmp_path / "linked", FAKE_MODEL)
    assert not (home / "windows.json").exists()


def test_a_daemon_is_never_started_over_a_folder_outside_pytests_own_temp_folder(tmp_path):
    """The guard of tests/mm/conftest.py: any folder that is not under pytest's temp folder (the owner's `~/.multimem` is one) is refused before a
    process starts, so no test can start, stop or call the daemon that runs on this machine for live sessions."""
    for outside in (Path("D:/not-pytests-temp-folder/.multimem"), Path("C:/not-pytests-temp-folder/.multimem")):
        with pytest.raises(AssertionError, match="only over a folder of pytest's temp folder"):
            Daemon(outside).start()
        assert not outside.exists() and not outside.parent.exists(), "and nothing was made there"
    assert Daemon(tmp_path / "inside").process is None


KIND = {"mic": "audio", "logs": "records", "notes": "text", "config": "text"}  # the kind of window a source costs; else video (the old SOURCE_KINDS)


def test_the_decider_estimate_is_in_minutes_from_the_windows_a_range_holds_at_the_models_seconds_a_window(week, monkeypatch):
    questions = [q for q in week[2] if q.retrieval is not None]
    assert len(questions) == 18, "every question of the tiny week but the config version has an oracle"
    monkeypatch.setattr(estimate, "windows_in", lambda table, source, ranges: {k: 1000 if k == KIND.get(source, "video") else 0 for k in estimate.COST_KINDS})
    slow = model_of("d3-mini")
    minutes = {}
    for model in (FAKE_MODEL, slow):
        mine = [q for q in questions if q.tags["modality"] in {"text", "records", *(model.about.reads & {"image", "video", "audio"})}]
        minutes[model.name] = sum(1000 * window_s(model.about, KIND.get(q.retrieval.sources[0], "video")) for q in mine) / 60
        assert estimate.decider_minutes({"decode_s": {}}, model.about, mine) == pytest.approx(minutes[model.name])
    assert minutes["d3-mini"] > 2 * minutes["fake"] > 0, "the slow model takes longer on the same windows"
    assert estimate.minutes_text(0.4) == "under a minute" and estimate.minutes_text(42.4) == "about 42 min"
