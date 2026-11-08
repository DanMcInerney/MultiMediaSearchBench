"""Ported from tests/test_bench.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order: the benchmark's run loop with a stub
agent, over a stand-in system (tests/stub_system.py) instead of a tiny week loaded into a daemon. What moved, by the plan's 3.2 and 4.5:
- `arm` is the configuration (`files`, `stub/all`: a system with variants, standing for the old mm arm of a fake decider), `Exec` is the
  `SessionSpec` the agent gets, `Run` is `RunFolder` with a scenario, and the mm-only steps (home, daemon, scrub) are the stub session's;
- tests that need the daemon, the home, the models or the decider tier are not here but in tests/mm (U7): the home and its copy, the
  labels and the scrub, the quiesce of jobs, the GPU seconds of an ask, `work_of`'s counts of ask jobs, the window estimate;
- the report's, the command line's and the build's tests belong to the report, the CLI and the scenarios (U5, U8, U2) and are not here.
Every expected value is the source's, or written from the plan's table (4.5) and the recorded fields of the pilot's results; the only
changes of an expectation are `arm` to `config` and the names of the fields it replaced (`arm`, `model`: `config`, `system`, `variant`)."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import re
import stat
import subprocess
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pytest
from agent_spec import NO_ISOLATION, clocked
from stub_system import FILES, STUB, Stub, StubSystem, run_of, tiny_questions

from mmsb import systems
from mmsb.core import fs, gpu, paths
from mmsb.core.agents import claude, codex
from mmsb.core.agents.common import result_json, result_jsons
from mmsb.core.config import Config
from mmsb.core.contract import PlainSession
from mmsb.core.record import ERRORS_STOP, STOP_AT, Halt, record, stop_reason, work_of
from mmsb.core.runfolder import JOBS_MAX, RunFolder, commit_of, is_done, load_result, now_text
from mmsb.core.runner import AGENTS, Agent, real_agents, run_agents, run_config

pytestmark = pytest.mark.xdist_group("runner")
N = len(tiny_questions())
BLIND = Config("blind", None, None, False)
NARROW = Config("stub/narrow", "stub", "narrow", False)
MEDIA = Config("stub/media", "stub", "media", False)
VERSIONED = ("notes", "config")  # the old AS_OF: written here from the plan's words (4.6), not taken from any module


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A run folder of the tiny week's 19 questions with a made-up frozen data folder, and a stub system. The ffmpeg `ensure_bin` copies is
    a made-up file."""
    ffmpeg = tmp_path / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
    run, dump = run_of(tmp_path)
    return types.SimpleNamespace(run=run, dump=dump, system=StubSystem(), root=tmp_path)


def go(w, agents, configs=(FILES, STUB), jobs: int = 2, run: RunFolder | None = None, answer_format: str = "line", **kw) -> None:
    """`run_agents` over the world's run, with the stub system standing for `stub` and the real registry for the rest."""
    stub_system = w.system
    run = run or w.run
    run_agents(run, agents, list(configs), w.dump, lambda system: run.folder / "stores" / system, jobs,
               load=lambda name: stub_system if name == "stub" else systems.load(name), isolation=NO_ISOLATION, answer_format=answer_format, **kw)


def result(run: RunFolder, agent: str, folder: str, qid: str) -> dict:
    return json.loads((run.folder / "results" / agent / folder / f"{qid}.json").read_text(encoding="utf-8"))


def together(spans: dict, x: str, y: str) -> bool:
    return spans[x][0] < spans[y][1] and spans[y][0] < spans[x][1]


def test_every_configuration_is_run_over_its_questions_and_a_second_command_asks_nothing(world):
    w, stub = world, Stub()
    go(w, {"claude": Agent(stub)})
    run, qids = w.run, [q.qid for q in tiny_questions()]
    assert N >= 15 and len(stub.execs) == 2 * N
    assert [Stub.dir_of(s) for s in stub.execs] == ["files"] * N + ["stub-all"] * N, "the configuration with no variant before the system's"
    for s in stub.execs:
        files = Stub.dir_of(s) == "files"
        assert s.work.parent == run.temp and s.work.name.startswith("mmb-t1-w") and s.deny == run.deny(s.work, NO_ISOLATION)
        assert s.data == (run.data if files else None) and s.bin_dir == run.bin and s.timeout_s == 900.0 and s.own_tools == files
        assert (s.server is None) == files and s.isolation == NO_ISOLATION and s.model is None and s.effort is None
        assert ("stub has recorded" in s.prompt) != files and (str(run.data) in s.prompt) == files
    # the session served the case before its agent ran, audited it after, and was reset after that, each exec once
    for qid in qids:
        events = [e[0] for e in w.system.events if e[1] == qid]
        assert events == ["serve", "audit", "reset"], qid
    opens = w.system.kinds("open")
    assert [(o[2], o[3]) for o in opens] == [("claude", "all")] and len(w.system.kinds("close")) == 1
    for folder, config in (("files", FILES), ("stub-all", STUB)):
        got = {q: result(run, "claude", folder, q) for q in qids}
        assert sorted(p.stem for p in (run.folder / "results" / "claude" / folder).glob("*.json")) == sorted(qids)
        assert all(r["problem"] is None and r["agent"] == "claude" and r["config"] == config.name for r in got.values())
        assert all((r["system"], r["variant"]) == (config.system, config.variant) and "arm" not in r and "model" not in r for r in got.values())
    sign = result(run, "claude", "stub-all", "image-photo-sign-1")
    assert (sign["passed"], sign["abstained"], sign["seed"], sign["denials"], sign["timeout_s"]) == (True, False, 7, 0, 900.0)
    assert (sign["calls"], sign["tools"], sign["scope_all"]) == (2, {"lookup": 1, "fetch": 1}, 1)
    assert sign["exec"] == "execs/claude/stub-all/image-photo-sign-1" and sign["agent_model"] == "stub" and sign["turns"] == 2
    assert (sign["audit"], sign["left"], sign["exclusive"], sign["beyond_timeout"], sign["busy_at_start"]) == (
        "image-photo-sign-1:2", {"labels": 2, "unkept": 0}, False, False, None), "the session's fields, merged in"
    assert (sign["gpu_estimate_s"], sign["cancelled"], sign["unended"], sign["orphans"], sign["settle_s"]) == (0.0, [], [], 0, 0.0)
    assert sign["work"] == {"jobs": 2, "windows": 20, "seconds": 0, "stopped": {}, "views": 0, "shells": 0}, "the audit's counts over the core's"
    blind = result(run, "claude", "files", "image-photo-sign-1")
    assert (blind["passed"], blind["abstained"], blind["variant"], blind["scope_all"], "left" not in blind) == (False, True, None, 0, True)
    assert blind["work"] == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 0, "shells": 1}
    assert json.loads((run.folder / "systems.json").read_text(encoding="utf-8"))["stub/all"]["asked"] == sorted(
        {"image", "video", "audio", "text", "records"})
    # a second command asks nothing; a deleted result is asked again alone
    again = Stub()
    go(w, {"claude": Agent(again)})
    assert again.execs == []
    gone = run.folder / "results" / "claude" / "stub-all" / "audio-mic-alarm-1.json"
    gone.unlink()
    once = Stub()
    go(w, {"claude": Agent(once)})
    assert [(Stub.dir_of(s), Stub.qid_of(s)) for s in once.execs] == [("stub-all", "audio-mic-alarm-1")] and gone.exists()


def test_a_stopped_agent_starts_nothing_more(world, capsys):
    w, total = world, N
    results = w.run.folder / "results"
    limited = Stub({"problem": "limit: weekly", "final": ""})
    go(w, {"claude": Agent(limited)}, jobs=1)
    assert len(limited.execs) == 1 and len(list((results / "claude" / "files").glob("*.json"))) == 1
    assert "limit: weekly" in capsys.readouterr().out and not w.system.kinds("open"), "no session opens for a stopped agent"
    first = Stub.qid_of(limited.execs[0])
    assert not is_done(load_result(results / "claude" / "files" / f"{first}.json"))

    # the next command asks the limited question again, and every other
    resumed = Stub()
    go(w, {"claude": Agent(resumed)}, [FILES], jobs=1)
    assert len(resumed.execs) == total and Stub.qid_of(resumed.execs[0]) == first

    # Codex stops at STOP_AT of the week's use, and when the use cannot be read
    over = Stub({"used_percent": STOP_AT + 1})
    go(w, {"codex": Agent(over, must_report_use=True)}, [FILES], jobs=1)
    assert len(over.execs) == 1 and f"{STOP_AT + 1:g}" in capsys.readouterr().out
    assert is_done(load_result(results / "codex" / "files" / f"{Stub.qid_of(over.execs[0])}.json"))
    unread = Stub({"used_percent": None})
    go(w, {"codex": Agent(unread, must_report_use=True)}, [FILES], jobs=1)
    assert len(unread.execs) == 1 and "cannot be read" in capsys.readouterr().out
    steady = Stub()
    go(w, {"codex": Agent(steady, must_report_use=True)}, [FILES], jobs=1)
    assert len(steady.execs) == total - 2, "under the limit and readable, the rest runs"
    # an agent whose use need not be read goes on without it
    quiet = Stub({"used_percent": None})
    other, _ = run_of(w.root / "other", tag="t2")
    go(w, {"claude": Agent(quiet)}, [FILES], jobs=1, run=other)
    assert len(quiet.execs) == total

    # Claude's use of the week stops it at the same line, from the rate-limit event's number
    line = Stub({"used_percent": STOP_AT})
    third, _ = run_of(w.root / "third", tag="t3")
    go(w, {"claude": Agent(line)}, [FILES], jobs=1, run=third)
    assert len(line.execs) == 1 and f"{STOP_AT:g}%" in capsys.readouterr().out

    # a result that escaped is done, and is a failed answer; more than JOBS_MAX at once is refused
    escaped = Stub({"problem": "escaped: Read reached the answers", "final": ""})
    fourth, _ = run_of(w.root / "fourth", tag="t4")
    go(w, {"claude": Agent(escaped)}, [FILES], jobs=1, run=fourth)
    saved = load_result(fourth.folder / "results" / "claude" / "files" / f"{Stub.qid_of(escaped.execs[0])}.json")
    assert saved["passed"] is False and is_done(saved) and len(escaped.execs) == total
    with pytest.raises(ValueError, match="jobs"):
        go(w, {"claude": Agent(Stub())}, [FILES], jobs=JOBS_MAX + 1, run=fourth)


def test_a_changed_file_stops_the_configuration_with_its_own_tools_and_the_next_command_refuses(world):
    run = world.run
    victim = next(p for p in sorted(run.data.rglob("*")) if p.is_file())
    rel = victim.relative_to(run.data).as_posix()

    class Vandal(Stub):
        def __call__(self, spec):
            victim.chmod(stat.S_IWRITE)
            victim.write_bytes(victim.read_bytes() + b"x")
            return super().__call__(spec)

    vandal = Vandal()
    go(world, {"claude": Agent(vandal)}, [FILES], jobs=1)
    assert len(vandal.execs) == 1, "the configuration stops when the data changed"
    saved = load_result(run.folder / "results" / "claude" / "files" / f"{Stub.qid_of(vandal.execs[0])}.json")
    assert saved["problem"] == f"escaped: the data changed: changed: {rel}" and saved["passed"] is False
    with pytest.raises(RuntimeError, match="data changed before"):
        go(world, {"claude": Agent(Stub())}, [FILES], jobs=1)


def test_the_gpu_is_checked_before_a_session_opens_only_for_a_system_that_needs_it(world, monkeypatch):
    def smi(output, code=0):
        return types.SimpleNamespace(run=lambda *a, **k: subprocess.CompletedProcess(a, code, output, "no device"),
                                     SubprocessError=subprocess.SubprocessError)

    needs = dataclasses.replace(world.system.describe("all"), needs_gpu=True)
    free = world.system.describe("all")
    # nvidia-smi's compute apps: some are a busy GPU, none is free, and no nvidia-smi is an error
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "shutil", types.SimpleNamespace(which=lambda name: "nvidia-smi"))
        patch.setattr(gpu, "subprocess", smi("1234, python.exe\n"))
        assert gpu.gpu_busy() == "1234, python.exe"
        with pytest.raises(RuntimeError, match="GPU is busy.*python.exe"):
            gpu.check_gpu(needs)
        gpu.check_gpu(free)
        patch.setattr(gpu, "subprocess", smi("7112, C:\\Windows\\explorer.exe\n"))
        assert gpu.gpu_busy() is None, "Windows' desktop, as nvidia-smi listed it on 2026-11-06, loads no model"
        patch.setattr(gpu, "subprocess", smi("7112, C:\\Windows\\explorer.exe\n1234, C:\\x\\python.exe\n"))
        assert gpu.gpu_busy() == "1234, C:\\x\\python.exe", "and hides nothing beside it"
        patch.setattr(gpu, "subprocess", smi("1234, C:\\x\\python.exe\n5678, C:\\y\\ChatGPT.exe\n9012, C:\\x\\python.exe\n"))
        patch.setenv(gpu.BESIDE, "1234, 5678")
        assert gpu.gpu_busy() == "9012, C:\\x\\python.exe", "the owner's named processes are shared, by id: another python is not"
        patch.setenv(gpu.BESIDE, "1234,5678,9012")
        assert gpu.gpu_busy() is None
        gpu.check_gpu(needs)
        patch.delenv(gpu.BESIDE)
        patch.setattr(gpu, "subprocess", smi("\n"))
        assert gpu.gpu_busy() is None
        gpu.check_gpu(needs)
        patch.setattr(gpu, "subprocess", smi("", 9))
        with pytest.raises(RuntimeError, match="exited 9"):
            gpu.gpu_busy()
        assert "exited 9" in gpu.gpu_lost(), "a GPU that cannot be read is lost"
        patch.setattr(gpu, "shutil", types.SimpleNamespace(which=lambda name: None))
        with pytest.raises(RuntimeError, match="nvidia-smi is not on PATH"):
            gpu.gpu_busy()

    # a run of a system that needs the GPU while something else uses it opens no session and runs nothing
    monkeypatch.setattr(gpu, "gpu_busy", lambda: "4321, other.exe")
    busy = types.SimpleNamespace(**{**vars(world), "system": StubSystem(needs_gpu=True)})
    stub = Stub()
    with pytest.raises(RuntimeError, match="GPU is busy, and it runs alone: 4321, other.exe"):
        go(busy, {"claude": Agent(stub)}, [STUB])
    assert stub.execs == [] and not busy.system.kinds("open")


def test_the_folders_of_an_agent_with_its_own_tools_may_not_hold_a_word_it_is_refused(world, tmp_path):
    run = world.run
    run.check_paths(NO_ISOLATION)
    # the runs folder is matched as a path part: a folder named like its start is not it
    dataclasses.replace(run, data=tmp_path / "mmsb-runs-extra" / "data").check_paths(NO_ISOLATION)
    with pytest.raises(ValueError, match="mmsb-runs"):
        dataclasses.replace(run, data=tmp_path / "mmsb-runs" / "data").check_paths(NO_ISOLATION)
    with pytest.raises(ValueError, match=r"\.codex"):
        dataclasses.replace(run, data=tmp_path / ".codex" / "data").check_paths(NO_ISOLATION)
    held = dataclasses.replace(run, data=tmp_path / "x.multimem" / "data")
    held.check_paths(NO_ISOLATION)
    with pytest.raises(ValueError, match=r"\.multimem"):
        held.check_paths(dataclasses.replace(NO_ISOLATION, deny_words=(".multimem",)))


def test_a_timeout_a_cap_and_an_escape_are_wrong_answers_that_are_done(world):
    kinds = {"image-photo-sign-1": ("timeout", "timeout: no answer in 900 s"), "video-screen-error-1": ("capped", "capped: max turns"),
             "audio-mic-doorbell-1": ("escaped", "escaped: Read reached the answers")}
    stub = Stub(by={("stub-all", qid): {"problem": problem, "final": ""} for qid, (_, problem) in kinds.items()})
    go(world, {"claude": Agent(stub)})
    assert len(stub.execs) == 2 * N, "none of the three stops the agent"
    for qid, (kind, _) in kinds.items():
        saved = result(world.run, "claude", "stub-all", qid)
        assert saved["problem"].startswith(kind) and saved["passed"] is False and is_done(saved), qid
    again = Stub()
    go(world, {"claude": Agent(again)})
    assert again.execs == [], "an exec that timed out is done: it is not asked again"


def test_a_session_that_was_not_the_one_asked_for_is_left_out_and_run_again(world, capsys):
    stub = Stub(script=[None] * N + [{"problem": "setup: the session has tools the configuration does not", "final": ""}])
    go(world, {"claude": Agent(stub)}, jobs=1)
    assert [Stub.dir_of(s) for s in stub.execs] == ["files"] * N + ["stub-all"], "the agent stops at the first setup problem"
    assert "stopping: claude: setup: the session has tools" in capsys.readouterr().out
    first = Stub.qid_of(stub.execs[-1])
    saved = result(world.run, "claude", "stub-all", first)
    assert saved["problem"].startswith("setup") and saved["passed"] is False and not is_done(saved)

    # the next command asks it again with the rest, and not the files configuration
    resumed = Stub()
    go(world, {"claude": Agent(resumed)}, jobs=1)
    assert [Stub.dir_of(s) for s in resumed.execs] == ["stub-all"] * N and Stub.qid_of(resumed.execs[0]) == first
    untils = {q.qid: q.scope.until for q in tiny_questions()}
    assert [untils[Stub.qid_of(s)] for s in resumed.execs] == sorted(untils[Stub.qid_of(s)] for s in resumed.execs), "in `until` order"


def test_a_lost_gpu_stops_the_configuration_and_its_exec_is_asked_again(world, capsys, monkeypatch):
    """After a driver crash a system's model cannot read, and an agent still answers from the little else it has: the exec that ends
    on a lost GPU is a setup problem, nothing more starts, and the next command asks it again."""
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(needs_gpu=True)})
    go(w, {"claude": Agent(Stub())}, [FILES], jobs=1)  # the configuration with its own tools, before the GPU matters
    checks = iter([None, "nvidia-smi exited 6: GPU is lost.  Reboot the system to recover this GPU"])
    monkeypatch.setattr(gpu, "gpu_busy", lambda: None)
    monkeypatch.setattr(gpu, "gpu_lost", lambda: next(checks))
    stub = Stub()
    go(w, {"claude": Agent(stub)}, jobs=1)
    assert len(stub.execs) == 2 and "stopping: claude: setup: the GPU was lost" in capsys.readouterr().out
    first, second = (Stub.qid_of(s) for s in stub.execs)
    assert is_done(result(w.run, "claude", "stub-all", first))
    lost = result(w.run, "claude", "stub-all", second)
    assert lost["passed"] is False and not is_done(lost)
    assert lost["problem"].startswith("setup: the GPU was lost, so stub's model may not have read: ") and "GPU is lost" in lost["problem"]
    monkeypatch.setattr(gpu, "gpu_lost", lambda: None)
    resumed = Stub()
    go(w, {"claude": Agent(resumed)}, jobs=1)
    assert len(resumed.execs) == N - 1 and Stub.qid_of(resumed.execs[0]) == second


@pytest.mark.parametrize("problem", ["setup: the session is not the one asked for", "auth: not logged in", "limit: weekly"])
def test_a_setup_auth_or_limit_problem_stops_the_agent_at_once(world, capsys, problem):
    stub = Stub({"problem": problem, "final": ""})
    go(world, {"claude": Agent(stub)}, jobs=1)
    assert len(stub.execs) == 1 and f"stopping: claude: {problem}" in capsys.readouterr().out
    assert not is_done(result(world.run, "claude", "files", Stub.qid_of(stub.execs[0]))) and not world.system.kinds("open")
    resumed = Stub()
    go(world, {"claude": Agent(resumed)}, [FILES], jobs=1)
    assert len(resumed.execs) == N and Stub.qid_of(resumed.execs[0]) == Stub.qid_of(stub.execs[0])


def test_three_errors_in_a_row_stop_the_agent_and_a_good_result_between_them_resets_the_count(world, capsys):
    def error(n: int) -> dict:
        return {"problem": f"error: boom {n}", "final": ""}

    assert ERRORS_STOP == 3
    stub = Stub(script=[error(1), error(2), None, error(3), error(4), error(5)])
    go(world, {"claude": Agent(stub)}, [FILES], jobs=1)
    assert len(stub.execs) == 6, "two errors, a good result, then three errors in a row"
    assert "claude: 3 errors in a row, the last error: boom 5" in capsys.readouterr().out
    saved = [load_result(world.run.folder / "results" / "claude" / "files" / f"{Stub.qid_of(s)}.json") for s in stub.execs]
    assert [is_done(r) for r in saved] == [False, False, True, False, False, False]
    resumed = Stub()
    go(world, {"claude": Agent(resumed)}, [FILES], jobs=1)
    assert len(resumed.execs) == N - 1, "the five errors are asked again, the good result is not"


def test_versioned_questions_run_one_at_a_time_in_time_order_and_the_rest_in_parallel(world):
    # a queue is in ascending `until`
    one = Stub()
    go(world, {"claude": Agent(one)}, [FILES], jobs=1)
    untils = {q.qid: q.scope.until for q in tiny_questions()}
    ordered = [untils[Stub.qid_of(s)] for s in one.execs]
    assert len(ordered) == N and ordered == sorted(ordered)

    # in a session that holds file versions back, those never overlap, and run in ascending `until`; the others overlap
    other, _ = run_of(world.root / "parallel", tag="t5")
    slow = Stub(pause=0.15)
    go(world, {"claude": Agent(slow)}, [STUB], jobs=3, run=other)
    spans = sorted((x for x in slow.log if x[0] == "stub-all"), key=lambda x: x[3])
    versioned = [x for x in spans if x[1] in VERSIONED]
    assert {x[1] for x in versioned} == set(VERSIONED) and len(versioned) == sum(1 for q in tiny_questions() if q.scope.sources[0] in VERSIONED)
    assert all(a[4] <= b[3] for a, b in zip(versioned, versioned[1:])), "two file-version questions were running at once"
    assert [untils[x[2]] for x in versioned] == sorted(untils[x[2]] for x in versioned), "versioned questions in `until` order"
    others = [x for x in spans if x[1] not in VERSIONED]
    assert any(a[3] < b[4] and b[3] < a[4] for i, a in enumerate(others) for b in others[i + 1:]), "the others stay parallel"
    assert any(a[3] < b[4] and b[3] < a[4] for a in versioned for b in others), "and run beside a file-version question"


def test_a_long_exec_holds_the_system_alone_by_the_sessions_rule(world):
    """The heavy rule is the session's (`admits`); the stub holds it as the plan's 4.6 words it. A range's seconds times `rate` is a
    case's estimate, so the longest ranges are the long ones."""
    long_ones = sorted(tiny_questions(), key=lambda q: -(q.scope.until - q.scope.since))[:2]
    rate = 400.0 / max(q.scope.until - q.scope.since for q in long_ones)
    heavy = {q.qid for q in tiny_questions() if rate * (q.scope.until - q.scope.since) >= 300.0}
    assert heavy and len(heavy) < N
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(rate=rate)})
    stub = Stub(pause=0.1)
    go(w, {"claude": Agent(stub)}, [STUB], jobs=3)
    spans = {x[2]: (x[3], x[4]) for x in stub.log}
    assert not any(together(spans, h, x) for h in heavy for x in spans if x != h), "an exec that holds the system runs alone"
    for q in tiny_questions():
        rec = result(w.run, "claude", "stub-all", q.qid)
        assert (rec["exclusive"], rec["busy_at_start"]) == ((True, 0) if q.qid in heavy else (False, None))
        assert rec["gpu_estimate_s"] == round(rate * (q.scope.until - q.scope.since), 1) and rec["held_s"] >= 0


def test_a_result_holds_the_denials_and_abstains_by_the_questions_own_numbers():
    q = next(q for q in tiny_questions() if q.tags["template"] == "absent-log")
    code = q.grade.ignore[0]  # the reserved error code's digits: the question's own number

    def made(answer, **more):
        return record(q, "claude", STUB, 7, f"execs/claude/stub-all/{q.qid}", "2026-11-05T00:00:00",
                      {"answer": answer, "seconds": 2.0, "problem": None, "tools": [], **more})

    said = made(f"ERR-{code} was never logged", denials=3)
    assert (said["abstained"], said["passed"], said["denials"]) == (True, True, 3), "the code in the answer is the question's own"
    guess = made(f"ERR-{code}: 3 times", denials=0)
    assert (guess["abstained"], guess["passed"], guess["denials"]) == (False, False, 0)
    assert made("unknown")["denials"] is None, "Codex reports none"
    timed_out = made(None, problem="timeout: no answer in 900 s")
    assert (timed_out["passed"], timed_out["abstained"], timed_out["problem"][:7]) == (False, False, "timeout")
    # a problem makes the answer wrong whatever it says: an exec that escaped or was capped did not answer, though its last message did
    for problem in ("escaped: Read reached the answers", "capped: error_max_turns", "timeout: no answer in 900 s", "setup: x"):
        assert made("unknown", problem=problem)["passed"] is False and made("unknown")["passed"] is True, problem


def test_a_models_questions_come_from_what_its_system_reads_and_the_run_notes_them(world, capsys):
    """Each configuration is asked what its system reads (the text and records only the one `controls` names), the pilot has the media
    questions of what the systems read and 3 controls, and systems.json holds what each was asked. The expected sets are written from
    the questions' own modalities."""
    by_modality = {m: {q.qid for q in tiny_questions() if q.tags["modality"] == m} for m in ("image", "video", "audio", "text", "records")}
    stub = Stub()
    go(world, {"codex": Agent(stub)}, [FILES, NARROW, MEDIA], jobs=3, controls="stub/narrow")
    assert set(stub.qids("files")) == set().union(*by_modality.values()) and len(stub.qids("files")) == N
    assert set(stub.qids("stub-narrow")) == by_modality["image"] | by_modality["video"] | by_modality["text"] | by_modality["records"]
    assert set(stub.qids("stub-media")) == by_modality["image"] | by_modality["video"], "controls name another: media only"
    noted = json.loads((world.run.folder / "systems.json").read_text(encoding="utf-8"))
    assert noted["stub/media"]["asked"] == ["image", "video"] and noted["stub/media"]["readable"] == ["image", "video"]
    assert noted["stub/narrow"]["asked"] == ["image", "records", "text", "video"] == noted["stub/narrow"]["readable"]
    assert noted["stub/narrow"]["dir"] == "stub-narrow" and noted["stub/narrow"]["system"] == "stub"

    # no `controls`: every configuration is asked everything it reads
    other, _ = run_of(world.root / "free", tag="t6")
    free = Stub()
    go(world, {"codex": Agent(free)}, [NARROW, MEDIA], jobs=3, run=other)
    assert set(free.qids("stub-narrow")) == by_modality["image"] | by_modality["video"] | by_modality["text"] | by_modality["records"]
    assert set(free.qids("stub-media")) == by_modality["image"] | by_modality["video"], "it reads no text"

    # the pilot's media are what the configurations without tools of their own read, and a session may allow fewer jobs
    third, _ = run_of(world.root / "pilot", tag="t7")
    capsys.readouterr()
    limited = types.SimpleNamespace(**{**vars(world), "system": StubSystem(job_limit=1)})
    pilot = Stub()
    go(limited, {"codex": Agent(pilot)}, [FILES, MEDIA], jobs=3, run=third, subset="pilot")
    picked = {Stub.qid_of(s) for s in pilot.execs if Stub.dir_of(s) == "files"}
    media = {q for q in picked if q.split("-")[0] in ("image", "video", "audio")}
    assert media == by_modality["image"] | by_modality["video"] and picked - media and len(picked) < N
    assert set(pilot.qids("stub-media")) == picked & (by_modality["image"] | by_modality["video"]), "no controls for the media-only system"
    assert "codex stub-media: 1 execs at once, not 3" in capsys.readouterr().out


def test_a_missing_extra_a_shared_folder_and_too_many_jobs_fail_before_anything_runs(world):
    def missing(name):
        raise systems.ExtraMissing("stub needs the stub extra: uv sync --extra stub")

    stub = Stub()
    with pytest.raises(systems.ExtraMissing, match="uv sync --extra stub"):
        run_agents(world.run, {"claude": Agent(stub)}, [FILES, STUB], world.dump, lambda s: world.root, 2,
                   load=lambda name: missing(name) if name == "stub" else systems.load(name), isolation=NO_ISOLATION,
                   answer_format="line")
    assert stub.execs == [] and not (world.run.folder / "results").exists() and not (world.run.folder / "systems.json").exists()
    twin = [Config("stub/v1.2", "stub", "v1.2", False), Config("stub/v1-2", "stub", "v1-2", False)]
    assert twin[0].dir == twin[1].dir == "stub-v1-2"
    with pytest.raises(ValueError, match="share a results folder"):
        go(world, {"claude": Agent(stub)}, twin)
    with pytest.raises(ValueError, match="jobs is 1 to"):
        go(world, {"claude": Agent(stub)}, [FILES], jobs=0)
    assert stub.execs == []


def test_an_audit_that_raises_ends_the_command_after_the_session_is_reset_and_closed_and_the_question_runs_again(world):
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(leak="a recording past the horizon in the tool results")})
    stub = Stub()
    with pytest.raises(RuntimeError, match="past the horizon"):
        go(w, {"claude": Agent(stub)}, [STUB], jobs=1)
    qid = Stub.qid_of(stub.execs[0])
    assert [e[0] for e in w.system.events if e[1] == qid] == ["serve", "audit", "reset"], "reset runs whatever the audit does"
    assert len(w.system.kinds("close")) == 1 and not (w.run.folder / "results" / "claude" / "stub-all" / f"{qid}.json").exists()


def test_what_a_session_closes_with_is_written_beside_the_results_and_removed_when_empty(world, capsys):
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(closing={"stray": [{"project": None}]})})
    go(w, {"claude": Agent(Stub())}, [STUB], jobs=3)
    file = w.run.folder / "results" / "claude" / "stub-all" / "stray.json"
    assert json.loads(file.read_text(encoding="utf-8")) == [{"project": None}]
    assert f"claude stub/all: 1 stray: see {file}" in capsys.readouterr().out
    w.system.closing = {"stray": []}
    (w.run.folder / "results" / "claude" / "stub-all" / "audio-mic-alarm-1.json").unlink()
    go(w, {"claude": Agent(Stub())}, [STUB], jobs=3)
    assert not file.exists(), "a file of an earlier command that this one has none for is taken away"


def test_the_systems_clock_note_ends_its_sentence_in_the_prompt(world):
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(clock_note="stub's clock reads a later date than now.")})
    stub = Stub()
    go(w, {"claude": Agent(stub)}, [STUB], jobs=3)
    assert all("stub's clock reads a later date than now." in s.prompt for s in stub.execs)


def test_how_an_exec_worked_is_read_from_its_calls_whatever_the_agent():
    def call(tool, arguments, result=None):
        return {"tool": tool, "arguments": arguments, "result": result, "error": None, "status": "completed"}

    codex_files = [call("shell", {"command": "ls"}, "a.mkv"), call("shell", {"command": "ffmpeg -i a.mkv -vf tile m.jpg"}, ""),
                   call("view_image", {"path": "C:\\w\\m.jpg"}), call("shell", {"command": "ffmpeg ..."}, ""), call("view_image", {"path": "c.png"})]
    claude_files = [call("Glob", {"pattern": "*"}, "a.mkv"), call("Bash", {"command": "ffmpeg -i a.mkv m.jpg"}, ""),
                    call("Read", {"file_path": "C:\\w\\montage.jpg"}, [{"type": "text", "text": "[image]"}]),
                    call("Read", {"file_path": "/data/notes/n.md"}, "text"), call("Read", {"file_path": "C:\\w\\FRAME.PNG"}),
                    call("Grep", {"pattern": "ERR"}, "x"), call("Bash", {"command": "ffmpeg -y"}, "")]
    server = [call("ask", {"question": "Is a sign visible?"}, [{"type": "text", "text": json.dumps({"job": "j-c1", "state": "done"})}]),
              call("find", {"sources": ["phone"]}, [{"type": "text", "text": "{}"}])]
    assert work_of(codex_files) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 2, "shells": 3}
    assert work_of(claude_files) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 2, "shells": 2}, \
        "a `Read` of a picture is a view whatever the case of its suffix; of a note it is not"
    assert work_of(server) == work_of([]) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 0, "shells": 0}, \
        "what a system's tools did is the system's to count (its audit), not the core's"
    # a result that is not JSON, or not an object, or a call that gave none, has no JSON
    for result in (None, "", "ask failed", "[1, 2]", [], [{"type": "image"}], {"content": None}, 7):
        assert result_json(result) is None, result
    assert result_json({"content": [], "structured_content": {"job": "j-s"}}) == {"job": "j-s"}
    assert result_json(["{\"a\": 1}"]) == {"a": 1}
    # a look of several moments has a JSON block for each (and a picture between): result_jsons gives all, result_json the first
    blocks = {"content": [{"type": "image"}, {"type": "text", "text": "{\"item\": 1}"}, {"type": "text", "text": "not json"},
                          {"type": "text", "text": "{\"item\": 2}"}, {"type": "text", "text": "[3]"}]}
    assert result_jsons(blocks) == [{"item": 1}, {"item": 2}] and result_json(blocks) == {"item": 1}
    assert result_jsons(None) == [] and result_jsons({"structured_content": {"job": "j"}, "content": blocks["content"]}) == [{"job": "j"}]


def test_a_result_holds_its_level_and_what_the_core_counted_of_how_it_worked():
    q = next(q for q in tiny_questions() if q.qid == "video-screen-error-1")
    calls = [{"tool": "find", "arguments": {"scope": "all"}}, {"tool": "ask", "arguments": {}}, {"tool": "ask", "arguments": {}},
             {"tool": "look", "arguments": {}}, {"tool": "view_image", "arguments": {"path": "m.png"}},
             {"tool": "shell", "arguments": {"command": "ls"}}]
    got = record(q, "codex", STUB, 7, "execs/codex/stub-all/x", "2026-11-05T00:00:00", {"answer": "13:00", "seconds": 4.0, "problem": None, "tools": calls})
    assert (got["level"], got["last"], got["type"], got["template"], got["absent"]) == (q.hint, bool(q.tags["last"]), "video", "screen-error", False)
    assert got["tools"] == {"find": 1, "ask": 2, "look": 1, "view_image": 1, "shell": 1} and got["calls"] == 6 and got["scope_all"] == 1
    assert got["work"] == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 1, "shells": 1}
    assert (got["config"], got["system"], got["variant"], got["exec"], got["at"]) == ("stub/all", "stub", "all", "execs/codex/stub-all/x",
                                                                                     "2026-11-05T00:00:00")
    plain = dataclasses.replace(q, tags={k: v for k, v in q.tags.items() if k != "last"})
    got = record(plain, "codex", STUB, 7, "x", "t", {"answer": "13:00", "seconds": 4.0, "problem": None, "tools": []})
    assert (got["last"], got["work"]["views"]) == (False, 0)


def test_every_exec_of_a_configuration_with_its_own_tools_gets_the_deny_list_of_its_slot_and_its_slot_is_empty_when_it_ends(world):
    class Messy(Stub):
        def __call__(self, spec):
            (spec.work / "montage.jpg").write_bytes(b"x")
            (spec.work / "frames").mkdir()
            (spec.work / "frames" / "f1.png").write_bytes(b"y")
            return super().__call__(spec)

    def norm(path) -> str:
        return re.sub(r"[\\/]+", "/", str(path).lower())

    messy, run = Messy(), world.run
    go(world, {"claude": Agent(messy)}, [FILES], jobs=2)
    assert messy.execs and all(s.deny == run.deny(s.work, NO_ISOLATION) for s in messy.execs)
    assert all("in:" + norm(s.work) not in s.deny and all("in:" + norm(run.slot(k)) in s.deny for k in range(4) if run.slot(k) != s.work)
               for s in messy.execs)
    assert all(not any(run.slot(k).iterdir()) for k in range(2)), "what an exec wrote is gone when it ends"


def test_two_questions_of_one_group_never_run_together_in_any_configuration(world):
    by_group: dict[str, list[str]] = {}
    for q in tiny_questions():
        by_group.setdefault(q.group, []).append(q.qid)
    group, mine = next((g, qids) for g, qids in by_group.items() if len(qids) >= 3)
    other = next(qids[0] for g, qids in by_group.items() if g != group)
    # three slots for four questions that are first in line: the three of one group take turns, the fourth runs beside them
    stub = Stub(pause=0.1)
    go(world, {"claude": Agent(stub)}, [FILES], jobs=3, only=[*mine, other])
    spans = {x[2]: (x[3], x[4]) for x in stub.log if x[0] == "files"}
    assert sorted(spans) == sorted([*mine, other])
    pairs = [(x, y) for i, x in enumerate(mine) for y in mine[i + 1:]]
    assert pairs and not any(together(spans, x, y) for x, y in pairs)
    assert any(together(spans, x, other) for x in mine), "a question of another group runs beside them"
    # and all of them, as a run asks: no pair of a group overlaps
    full, _ = run_of(world.root / "full", tag="t8")
    every = Stub(pause=0.05)
    go(world, {"claude": Agent(every)}, [FILES], jobs=3, run=full)
    all_spans = {x[2]: (x[3], x[4]) for x in every.log if x[0] == "files"}
    assert not any(together(all_spans, x, y) for qids in by_group.values() for i, x in enumerate(qids) for y in qids[i + 1:])


# -- the seam: a scenario of another kind, a blind configuration, and what the scenario makes of each exec --------------------------

SEAM: list[tuple] = []  # what the scenario, the session and the stub agent were asked, in order, by the test that uses them


class Marked:
    """A scenario as another dataset's would be: it says what each exec reads, is asked and adds, and has no way to the recordings
    for the blind configuration."""

    name, code, generator = "marked", "M", 1

    def view(self, question, slot, dump, temp, key, build):
        if build:  # (U8: a call that only names the folder, as `run_agents` makes to check its path, makes nothing: it is not noted)
            SEAM.append(("view", question.qid, slot))
        return temp / f"view-{slot}"

    def prompt(self, question, access):
        SEAM.append(("prompt", question.qid))
        return f"{'own' if access.own_tools else 'none'}|{access.data}|{access.served.holder}|{question.qid}"

    def after_exec(self, question, out, dump, access):
        SEAM.append(("after_exec", question.qid))
        return {"audit_by_scenario": out["final"]}

    def deny_words(self, others, temp, key):
        return ("in:marked",)

    def release(self, temp, key):
        SEAM.append(("release",))


def test_each_configuration_is_given_what_the_scenarios_view_prompt_and_after_exec_make_and_blind_results_have_their_own_folder(world):
    scenario = Marked()
    run = dataclasses.replace(world.run, scenario=scenario)
    q = tiny_questions()[0]
    seen: list = []
    both = Config("stub/all+files", "stub", "all", True)

    def agent(spec):
        SEAM.append(("agent", spec.run_dir.name))
        seen.append(spec)
        return {"final": "x", "tools": [], **clocked(spec, 1.0), "problem": None, "turns": 1, "tokens": None,
                "cost_usd": None, "used_percent": 1.0, "model": "stub", "denials": 0}

    view = run.temp / "view-0"
    # the order of the plan's 4.5: the session serves (4), the view is made (5), the prompt (6), the agent runs (7), the scenario's
    # addition and the session's audit are read (8), and the session is reset (9)
    expected = {"blind": ["prompt", "agent", "after_exec"], "files": ["view", "prompt", "agent", "after_exec"],
                "stub/all": ["serve", "prompt", "agent", "after_exec", "audit", "reset"],
                "stub/all+files": ["serve", "view", "prompt", "agent", "after_exec", "audit", "reset"]}
    for config, data, holder in ((BLIND, None, None), (FILES, view, None), (STUB, None, "stub"), (both, view, "stub")):
        SEAM.clear()
        seen.clear()
        w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(shared=SEAM)})
        go(w, {"claude": Agent(agent)}, [config], jobs=1, run=run, only=[q.qid])
        spec = seen[-1]
        assert (spec.own_tools, spec.data) == (config.own_tools, data)
        assert spec.prompt == f"{'own' if config.own_tools else 'none'}|{data}|{holder}|{q.qid}" and spec.work == run.slot(0)
        assert spec.deny == run.deny(spec.work, NO_ISOLATION, spec.data) and "in:marked" in spec.deny
        hidden = "in:" + re.sub(r"[\\/]+", "/", str(run.data).lower())  # (U8: the folder cut from the dump is not the dump, so the dump is denied)
        assert (hidden in spec.deny) == config.own_tools, "the dump is hidden from the tools that read a folder cut from it, and from no other"
        assert [s[0] for s in SEAM if s[0] not in ("max_jobs", "open", "close", "release")] == expected[config.name],             "the folder is made only for tools of its own"
        assert SEAM[-1][0] == "release" or SEAM[-2][0] == "release", "the scenario puts away what it made when the run ends"
        rec = result(run, "claude", config.dir, q.qid)
        assert (rec["config"], rec["audit_by_scenario"], rec["exec"]) == (config.name, "x", f"execs/claude/{config.dir}/{q.qid}")
        assert rec["system"] == config.system and ("gpu_estimate_s" in rec) == (config.system == "stub")
    assert sorted(p.name for p in (run.folder / "results" / "claude").iterdir()) == ["blind", "files", "stub-all", "stub-all+files"]
    assert [c.dir for c in (BLIND, FILES, STUB, both)] == ["blind", "files", "stub-all", "stub-all+files"]
    assert run.result_path("codex", BLIND, "q-1") == run.folder / "results" / "codex" / "blind" / "q-1.json"
    # a done blind result is not asked again
    again: list = []
    go(world, {"claude": Agent(lambda spec: again.append(spec))}, [BLIND], jobs=1, run=run, only=[q.qid])
    assert again == []


def test_a_scenario_that_gives_the_tools_of_a_configuration_no_folder_is_refused(world):
    class Nowhere(Marked):
        def view(self, question, slot, dump, temp, key, build):
            return None

    run = dataclasses.replace(world.run, scenario=Nowhere())
    with pytest.raises(ValueError, match="no folder to read"):
        go(world, {"claude": Agent(Stub())}, [FILES], jobs=1, run=run, only=[tiny_questions()[0].qid])


def test_the_groups_of_a_scenarios_questions_decide_which_never_run_at_once(world):
    pair = [q for q in tiny_questions() if q.group == "photo-sign"][:2]
    assert len(pair) == 2
    session = PlainSession()
    run = dataclasses.replace(world.run, scenario=Marked())

    def overlap(questions) -> bool:
        stub = Stub(pause=0.2)
        run_config(run, "claude", Agent(stub), BLIND, None, questions, session, world.dump, NO_ISOLATION, 2, Halt(), "line")
        spans = {x[2]: (x[3], x[4]) for x in stub.log}
        return together(spans, pair[0].qid, pair[1].qid)

    assert not overlap(pair), "one group: one at a time"
    fs.remove_tree(run.folder / "results")
    assert overlap([dataclasses.replace(q, group=q.qid) for q in pair]), "a group of its own each: they run together, in the blind configuration too"


def test_the_real_agents_are_the_two_functions_of_a_spec_and_the_run_gives_each_its_timeout_folders_and_isolation(world):
    real = real_agents()
    assert set(real) == set(AGENTS) == {"claude", "codex"}
    assert real["claude"] == Agent(claude.ask_claude) and real["codex"] == Agent(codex.ask_codex, must_report_use=True)
    seen: list = []
    run = dataclasses.replace(world.run, timeout_s=123.0)
    isolation = dataclasses.replace(NO_ISOLATION, env_drop_prefixes=("X_",))
    stub = Stub()
    run_agents(run, {"claude": Agent(lambda spec: seen.append(spec) or stub(spec))}, [FILES, STUB], world.dump, lambda s: world.root, 1,
               load=lambda name: world.system if name == "stub" else systems.load(name), isolation=isolation, only=[tiny_questions()[0].qid], answer_format="line")
    files, stubbed = seen
    assert (files.timeout_s, files.data, files.bin_dir, files.deny, files.isolation) == (
        123.0, run.data, run.bin, run.deny(files.work, isolation), isolation)
    assert (stubbed.timeout_s, stubbed.data, stubbed.own_tools, stubbed.served.holder) == (123.0, None, False, "stub")
    assert stubbed.served.server.name == "stub" and stubbed.served.parts == ("THE STUB'S NOTES",)


def test_stop_reason_asks_for_the_use_of_the_week_only_of_an_agent_that_must_report_it():
    rec = {"problem": None, "used_percent": None}
    assert stop_reason("codex", rec, True) == "codex: the plan's weekly use cannot be read, so the quota cannot be watched"
    assert stop_reason("claude", rec) is None and stop_reason("claude", rec, False) is None
    assert stop_reason("claude", {"problem": "auth: not logged in", "used_percent": 1.0}) == "claude: auth: not logged in"
    assert stop_reason("claude", {"problem": None, "used_percent": 94.9}) is None
    assert stop_reason("claude", {"problem": None, "used_percent": 95.0}) == "claude: the plan is at 95% of its use; the run stops at 95%"
    halt = Halt(must_report_use=True)
    halt.after("codex", rec)
    assert halt.reason and "cannot be read" in halt.reason


def test_a_result_is_done_unless_its_problem_did_not_test_the_question_and_the_helpers_of_the_run_folder():
    assert not is_done(None) and is_done({"problem": None})
    for kind in ("escaped", "timeout", "capped"):
        assert is_done({"problem": f"{kind}: x"}), kind
    for kind in ("setup", "limit", "auth", "error"):
        assert not is_done({"problem": f"{kind}: x"}), kind
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", now_text())
    short = subprocess.run(["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
    assert commit_of() in (short, short + "+dirty")


def previewing(world):
    """What `previewer` gives `show_agents`: the preview session of each configuration's system (the real `files`, the world's stub)."""
    from mmsb.core.dryrun import previewer

    return previewer(world.run, {"files": systems.load("files"), "stub": world.system}, lambda system: world.root, world.dump)


def test_a_run_prints_what_it_will_run_and_about_how_long_that_takes(world, capsys):
    from mmsb.core.costing import BLIND_MIN, EXEC_MIN, estimate, show_agents

    descriptions = {FILES: systems.load("files").describe(None), STUB: world.system.describe("all"), BLIND: None}
    go(world, {"claude": Agent(Stub())}, [FILES], jobs=2)
    capsys.readouterr()
    show_agents(world.run, ["claude", "codex"], [FILES, STUB, BLIND], descriptions, previewing(world), 3, 1)
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    for expected in ("claude files 0 to run of 19", f"claude stub/all {N} to run of {N}", f"claude blind {N} to run of {N}",
                     f"codex files {N} to run of {N}", f"codex stub/all {N} to run of {N}", f"codex blind {N} to run of {N}"):
        assert expected in lines, expected
    # 5 N execs are left; 3 N of them at 2.5 minutes, the 2 N blind ones at 1, over 3 jobs
    hours = (3 * N * 2.5 + 2 * N * 1.0) / 3 / 60
    assert lines[-1] == f"{5 * N} execs to run, about {hours:.1f} h at 2.5 min an exec (1 for a blind one) over 3 jobs"
    assert (EXEC_MIN, BLIND_MIN) == (2.5, 1.0)
    assert estimate([(30, 3), (6, 3, BLIND_MIN)], 3) == f"about {(30 * 2.5 + 6 * 1.0) / 3 / 60:.1f} h at 2.5 min an exec (1 for a blind one) over 3 jobs"
    assert estimate([(30, 3)], 3) == "about 0.4 h at 2.5 min an exec over 3 jobs"
    assert not (world.run.folder / "results" / "codex").exists(), "it asks nothing"


# -- G1's repairs: what the runner records of the system, what it tells it, and what it does when an exec or a close goes wrong -----------------


def test_systems_json_notes_each_configurations_description_and_a_changed_configuration_with_results_is_refused(world):
    w, first = world, tiny_questions()[0].qid
    go(w, {"claude": Agent(Stub())}, [STUB], jobs=1, only=[first])
    file = w.run.folder / "systems.json"
    noted = json.loads(file.read_text(encoding="utf-8"))["stub/all"]
    described = w.system.describe("all")
    assert noted["description"] == {**dataclasses.asdict(described), "reads": sorted(described.reads)}, "all the system says of itself"
    assert (noted["system"], noted["variant"], noted["dir"], noted["readable"], noted["asked"]) == (
        "stub", "all", "stub-all", sorted(described.reads), sorted(described.reads)), "and the keys the report reads"
    # the system is another version now (whatever config_hash covers: its version, skill, notes or model settings)
    w.system.config_hash = "d"
    before, stub = file.read_bytes(), Stub()
    with pytest.raises(ValueError, match=r"stub/all has results in run from stub 1 \(config_hash c\), and it is 1 \(config_hash d\) now.*"
                                         r"Build a new run, or remove results/\*/stub-all"):
        go(w, {"claude": Agent(stub)}, [STUB], jobs=1)
    assert stub.execs == [] and file.read_bytes() == before, "refused before anything ran, and nothing was noted"
    # a configuration with no results yet is noted as the system is now, and the one with results is run again once its results are gone
    go(w, {"claude": Agent(Stub())}, [NARROW], jobs=1, only=[first])
    assert json.loads(file.read_text(encoding="utf-8"))["stub/narrow"]["description"]["config_hash"] == "d"
    fs.remove_tree(w.run.folder / "results" / "claude" / "stub-all")
    again = Stub()
    go(w, {"claude": Agent(again)}, [STUB], jobs=1, only=[first])
    assert again.qids() == [first] and json.loads(file.read_text(encoding="utf-8"))["stub/all"]["description"]["config_hash"] == "d"


@dataclass
class Gated(StubSystem):
    """A system whose session opens the gate when the exec of `first` is reset, so that a test can hold the others until then."""

    gate: threading.Event = None
    first: str = ""

    def note(self, kind: str, qid: str | None, *rest) -> None:
        super().note(kind, qid, *rest)
        if kind == "reset" and qid == self.first:
            self.gate.set()


def test_an_exec_is_reset_with_the_number_of_the_others_still_running_none_at_one_job_and_the_rest_at_three(world):
    go(world, {"claude": Agent(Stub())}, [STUB], jobs=1)
    resets = world.system.kinds("reset")
    assert len(resets) == N and {e[2] for e in resets} == {0}, "one job at a time: never another exec running"
    # three execs of three groups start together; the first ends while the other two are still in their agents, so it is told 2 others
    seen: dict = {}
    for q in tiny_questions():
        seen.setdefault(q.group, q.qid)
    quick, *held = list(seen.values())[:3]
    gate, in_agent = threading.Event(), [threading.Event(), threading.Event()]
    stub = Stub()

    def agent(spec):
        qid = spec.run_dir.name
        if qid == quick:
            assert all(e.wait(30) for e in in_agent), "the other two never started"
        else:
            in_agent[held.index(qid)].set()
            assert gate.wait(30), "the first never ended"
        return stub(spec)

    other, _ = run_of(world.root / "three", tag="t9")
    system = Gated(rules=False, gate=gate, first=quick)
    go(types.SimpleNamespace(**{**vars(world), "system": system}), {"claude": Agent(agent)}, [STUB], jobs=3, run=other, only=[quick, *held])
    told = {e[1]: e[2] for e in system.kinds("reset")}
    assert set(told) == {quick, *held} and told[quick] == 2 and all(0 <= told[qid] <= 2 for qid in held)


def test_an_exec_folder_of_an_earlier_command_is_removed_before_the_exec_runs_again(world):
    q = tiny_questions()[0]
    old = world.run.exec_dir("claude", FILES, q.qid)
    old.mkdir(parents=True)
    (old / "tools.json").write_text("[]", encoding="utf-8")
    (old / "final.txt").write_text("what an exec that was cut off left", encoding="utf-8")
    go(world, {"claude": Agent(Stub())}, [FILES], jobs=1, only=[q.qid])
    assert not (old / "tools.json").exists() and not (old / "final.txt").exists(), "nothing of the earlier exec is beside the new result"


def test_an_agent_that_raises_still_has_its_session_reset_and_closed_and_leaves_no_result(world):
    def gone(spec):
        raise RuntimeError("the agent's process is gone")

    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem()})
    with pytest.raises(RuntimeError, match="process is gone"):
        go(w, {"claude": Agent(gone)}, [STUB], jobs=1)
    (served,) = w.system.kinds("serve")
    assert [e[0] for e in w.system.events if e[1] == served[1]] == ["serve", "reset"], "its jobs are cancelled before the daemon stops"
    assert len(w.system.kinds("close")) == 1 and not list(w.run.folder.glob("results/**/*.json"))


def test_a_session_that_cannot_be_closed_after_an_error_does_not_take_the_errors_place(world, capsys):
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(leak="a recording past the horizon", close_raises="the daemon is gone")})
    with pytest.raises(RuntimeError, match="a recording past the horizon"):
        go(w, {"claude": Agent(Stub())}, [STUB], jobs=1)
    assert "claude stub/all: closing its session failed as well: the daemon is gone" in capsys.readouterr().out
    assert len(w.system.kinds("close")) == 1
    # a close that fails after a cell that went well is the error there is
    fine = types.SimpleNamespace(**{**vars(world), "system": StubSystem(close_raises="the daemon is gone")})
    with pytest.raises(RuntimeError, match="the daemon is gone"):
        go(fine, {"claude": Agent(Stub())}, [STUB], jobs=1, only=[tiny_questions()[0].qid])


def test_the_estimate_counts_a_configuration_at_the_execs_its_system_allows_at_once_and_says_so(world, capsys):
    from mmsb.core.costing import show_agents

    limited = types.SimpleNamespace(**{**vars(world), "system": StubSystem(job_limit=1)})
    descriptions = {FILES: systems.load("files").describe(None), STUB: limited.system.describe("all")}
    show_agents(limited.run, ["claude"], [FILES, STUB], descriptions, previewing(limited), 3, 1)
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert f"claude files {N} to run of {N}" in lines, "a configuration with no system has all its jobs"
    assert f"claude stub/all {N} to run of {N} (1 at once)" in lines
    hours = (N * 2.5 / 3 + N * 2.5 / 1) / 60  # the files at the 3 asked for, the stub one at a time
    assert lines[-1] == f"{2 * N} execs to run, about {hours:.1f} h at 2.5 min an exec over 3 jobs"


def test_outside_a_git_checkout_the_commit_a_run_is_made_by_is_the_installed_version_and_when_that_cannot_be_said_it_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)  # a copy with no .git: installed, or unpacked
    assert commit_of() == importlib.metadata.version("mmsb")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="is not a git checkout and mmsb is not installed"):
        commit_of()
