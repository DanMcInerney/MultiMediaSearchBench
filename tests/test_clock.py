"""The working-time clock (plan 6.5, DESIGN 8.4; written for this repository, U14): `core/clock.py` with a clock and a queue the test moves, the loop
that waits for a process, the two real agents killing a real process at a limit of a fifth of a second, a queue read live while a process runs,
the clock starting with the process and not before it (the files an agent makes ready, the gate of a side-by-side run), and the result fields a
run records. The command that tries the clock on a few cases is tests/test_clock_check.py.

The expected values are the plan's words (a case at 950 s wall with 100 s queued is not over; at 901 s of working time it is; the wall cap fires
whatever the queue; no queue figure means the wall and the recorded `timeout: no answer in 900 s`), the pilot's recorded timeouts (their text and
the 900.3 s they took), a law over generated inputs (over iff working time reaches the limit or the wall reaches the cap, and `left` is the first
moment it is over), a model of the waiting loop (never longer than a second, never past the time left), and what a test planted."""

from __future__ import annotations

import random
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from agent_spec import clocked, spec_of
from stub_system import FILES, STUB, Stub, StubSession, StubSystem, tiny_questions
from test_runner import go, result, world  # noqa: F401

from mmsb.core import clock as clocks
from mmsb.core import fs
from mmsb.core.agents import claude, codex
from mmsb.core.agents.common import wait_for
from mmsb.core.clock import CLOCKS, DEFAULT_CLOCK, FIELDS, WALL, WORKING, WorkingClock, budget_s, check_queue, unreported
from mmsb.core.contract import Case
from mmsb.core.record import record
from mmsb.core.runfolder import is_done
from mmsb.core.runner import Agent
from mmsb.core.schedule import Pools, SideBySide

pytestmark = pytest.mark.xdist_group("runner")
RECORDED = fs.read_json(Path(__file__).parent / "oracles" / "recorded" / "pilot-s1" / "converted" / "results" / "claude" / "files" / "audio-mic-alarm-3.json")
QUESTIONS = tiny_questions()


class Moving:
    """A clock a test moves (`t`), and the queue it sets (`None`: the system cannot say). It starts at 100, so a clock that took its start as 0 shows."""

    def __init__(self, queue: float | None = 0.0) -> None:
        self.t, self.queue = 100.0, queue

    def __call__(self) -> float:
        return self.t

    def queue_s(self) -> float | None:
        return self.queue

    def clock(self, limit: float = 900.0, cap: float = 2700.0) -> WorkingClock:
        return WorkingClock(limit, cap, self.queue_s, now=self)


# -- the clock, with a time and a queue that are moved by hand --------------------------------------------------------------------------

def test_a_case_at_950_s_wall_with_100_s_queued_is_not_over_and_at_901_s_of_working_time_it_is():
    moving = Moving(queue=100.0)
    clock = moving.clock()
    moving.t += 950
    assert clock.expired() is None
    assert clock.figures() == {"clock": WORKING, "working_s": 850.0, "queue_s": 100.0, "wall_s": 950.0}
    moving.t += 51  # 1,001 s on the wall, 901 s of working time
    assert clock.expired() == "timeout: no answer in 900 s of working time (1001 s on the wall clock, 100 s queued)"
    assert clock.figures() == {"clock": WORKING, "working_s": 901.0, "queue_s": 100.0, "wall_s": 1001.0}
    moving.queue = 101.0  # the queue is read live: a second more of it and the case is at exactly 900 s of working time
    assert clock.expired() is not None
    moving.queue = 102.0
    assert clock.expired() is None and clock.left() == pytest.approx(1.0)


def test_the_wall_cap_fires_whatever_the_queue():
    moving = Moving()
    clock = moving.clock()
    moving.t, moving.queue = clock.began + 2699, 2699.0  # everything queued: no working time at all
    assert clock.expired() is None and clock.figures()["working_s"] == 0.0
    moving.t, moving.queue = moving.t + 1, 2700.0
    assert clock.expired() == "timeout: no answer in 2700 s on the wall clock (2700 s of it queued)"
    moving.queue = 2000.0  # a queue of most of it: 700 s of working time, under the limit, and still over the cap
    assert clock.expired() == "timeout: no answer in 2700 s on the wall clock (2000 s of it queued)"
    moving.queue = 0.0  # nothing queued: the limit went first, at 900 s, long before the cap
    assert clock.expired() == "timeout: no answer in 900 s"
    moving.queue = None  # and with no figure at all, the wall is the clock
    assert clock.expired() == "timeout: no answer in 900 s"
    assert clocks.WALL_FACTOR == 3.0 and WorkingClock.of(300, moving.queue_s, now=moving).wall_cap_s == 900.0


def test_with_no_queue_figure_the_clock_is_the_wall_and_its_text_is_the_recorded_one():
    moving = Moving(queue=None)
    clock = moving.clock()
    assert RECORDED["problem"] == "timeout: no answer in 900 s" and RECORDED["seconds"] == 900.3  # (what the pilot's timeouts said and took)
    moving.t = clock.began + 899.5
    assert clock.expired() is None
    moving.t = clock.began + RECORDED["seconds"]
    assert clock.expired() == RECORDED["problem"]
    assert clock.figures() == {"clock": WALL, "working_s": 900.3, "queue_s": None, "wall_s": 900.3}
    moving.queue = 0.0  # a session with no queue says 0.0: the same text, and the clock is the working clock with nothing taken off
    assert clock.expired() == RECORDED["problem"]
    assert clock.figures() == {"clock": WORKING, "working_s": 900.3, "queue_s": 0.0, "wall_s": 900.3}


def test_a_queue_over_the_wall_leaves_no_working_time_and_stays_in_the_figures_for_anyone_to_see():
    moving = Moving(queue=120.0)
    clock = moving.clock()
    moving.t += 100.04
    assert clock.figures() == {"clock": WORKING, "working_s": 0.0, "queue_s": 120.0, "wall_s": 100.0}


def test_a_clock_is_made_of_a_positive_limit_under_a_cap_no_lower_than_it():
    for limit, cap in ((0, 10), (-1, 10), (11, 10)):
        with pytest.raises(ValueError, match="cap"):
            WorkingClock(limit, cap, unreported)
    WorkingClock(10, 10, unreported)


def test_a_clock_is_over_when_working_time_reaches_the_limit_or_the_wall_reaches_the_cap_and_left_is_the_first_moment_it_is():
    """The law over generated inputs in quarter seconds (exact in floating point): the model is the plan's sentence, and `left` is the time to the
    first moment the clock is over if the queue stays as it is."""
    rng = random.Random(14)
    for _ in range(3000):
        limit = rng.randrange(1, 40) / 4
        cap = limit * rng.choice((1, 2, 3))
        wall = rng.randrange(0, 200) / 4
        queue = rng.choice((None, 0.0, rng.randrange(0, 200) / 4))
        moving = Moving(queue)
        clock = moving.clock(limit, cap)
        moving.t += wall
        working = wall if queue is None else max(0.0, wall - queue)
        over = working >= limit or wall >= cap
        where = (limit, cap, wall, queue)
        assert (clock.expired() is not None) == over, where
        assert (clock.left() == 0.0) == over, where
        if not over and (queue is None or queue <= wall):  # (a queue over the wall is no state a system reports)
            left = clock.left()  # (a multiple of a quarter, at least a quarter)
            moving.t += left - 0.25
            assert clock.expired() is None, where
            moving.t += 0.25
            assert clock.expired() is not None, where


def test_the_text_says_the_wall_cap_only_when_the_cap_and_not_the_limit_is_what_ended_it():
    rng = random.Random(15)
    for _ in range(500):
        moving = Moving(rng.choice((None, 0.0, rng.randrange(0, 200) / 4)))
        clock = moving.clock(10.0, 30.0)
        moving.t += rng.randrange(0, 200) / 4
        working = moving.t - clock.began - (moving.queue or 0.0)
        text = clock.expired()
        if text is not None:
            assert text.startswith("timeout: no answer in 10 s") == (working >= 10.0), text
            assert text.startswith("timeout: no answer in 30 s on the wall clock") == (working < 10.0), text


# -- the loop that waits for a process ----------------------------------------------------------------------------------------------

class Passing:
    """A process stand-in that lets `moving`'s time pass by what `wait` is told to wait, ends at `ends` (None: never) and is killed when the loop
    gives up on it (`killed`)."""

    def __init__(self, moving: Moving, ends: float | None = None) -> None:
        self.moving, self.ends, self.waits, self.killed = moving, ends, [], False
        self.pid = 0

    def wait(self, timeout: float | None = None) -> int:
        if self.killed or (self.ends is not None and self.moving.t + (timeout or 0) >= self.ends):
            return 0
        self.waits.append(timeout)
        self.moving.t += timeout
        raise subprocess.TimeoutExpired("x", timeout)


def test_the_loop_waits_a_second_at_most_and_never_past_the_time_that_is_left_and_kills_a_process_that_outlives_the_clock(monkeypatch):
    killed = []
    monkeypatch.setattr("mmsb.core.agents.common.kill_tree", lambda process: killed.append(setattr(process, "killed", True)))
    moving = Moving(queue=None)
    process = Passing(moving)
    assert wait_for(process, moving.clock(2.5, 7.5)) == "timeout: no answer in 2.5 s"
    assert process.waits == [1.0, 1.0, 0.5] and killed == [None], "a second at a time, then what was left of 2.5 s; and the tree was killed"
    # a process that ends first is not killed and nothing is told
    moving = Moving(queue=None)
    process = Passing(moving, ends=moving.t + 1.5)
    assert wait_for(process, moving.clock(900.0, 2700.0)) is None and process.waits == [1.0] and len(killed) == 1
    # a queue that begins while it waits moves the end: 3 s of working time and 1 s of queue make 4 s of wall, and the loop sees it each time it wakes
    moving = Moving()
    began = moving.t
    process = Passing(moving)
    problem = wait_for(process, WorkingClock(3.0, 100.0, lambda: 1.0 if moving.t - began >= 2.0 else 0.0, now=moving))
    assert problem == "timeout: no answer in 3 s of working time (4 s on the wall clock, 1 s queued)" and process.waits == [1.0, 1.0, 1.0, 1.0]


# -- budgets and the refusal of a queue that cannot be told ----------------------------------------------------------------------------

def test_a_budget_is_300_seconds_up_to_an_hour_900_for_a_day_or_more_and_in_proportion_between():
    assert [budget_s(h) for h in (0, 0.25, 1)] == [300.0, 300.0, 300.0]
    assert [budget_s(h) for h in (24, 25, 24 * 7)] == [900.0, 900.0, 900.0]
    between = [budget_s(h) for h in (2, 6, 12, 18, 23)]
    assert between == sorted(between) and 300.0 < between[0] and between[-1] < 900.0
    assert budget_s(12.5) == pytest.approx(600.0)  # halfway in hours between 1 and 24 is halfway in seconds between 300 and 900
    assert WorkingClock.of(budget_s(1), unreported).wall_cap_s == 900.0 and WorkingClock.of(budget_s(24), unreported).wall_cap_s == 2700.0
    with pytest.raises(ValueError, match="hours"):
        budget_s(-1)


def test_more_than_one_exec_at_once_on_the_working_clock_is_refused_for_a_session_that_cannot_say_its_queue():
    case = Case("q", 0.0, None, (), 0.0, 1.0, "q", 0.0, -1, Path(), Path())
    says, silent = StubSession(StubSystem(), None), StubSession(StubSystem(reports_queue=False), None)
    says.at_once = silent.at_once = 4
    plain, on_gpu = StubSystem().describe("all"), StubSystem(needs_gpu=True).describe("all")
    for session in (says, silent):
        for clock in CLOCKS:
            check_queue("c", plain, session, case, 1, clock)  # one at a time has no queue
    check_queue("c", plain, says, case, 4, WORKING)
    check_queue("c", plain, silent, case, 4, WALL)
    # the remedy is one exec at a time, by the option that sets it for this system; it is never the wall clock, which would keep the several
    for description, option in ((plain, "--jobs"), (on_gpu, "--gpu-jobs")):
        with pytest.raises(ValueError, match=rf"c: 2 execs at once on the working clock.*no queue time.*one exec at a time \({option} 1\)") as refused:
            check_queue("c", description, silent, case, 2, WORKING)
        assert "--clock" not in str(refused.value) and "wall" not in str(refused.value), str(refused.value)
    with pytest.raises(ValueError, match="one of working, wall"):
        check_queue("c", plain, says, case, 1, "monotonic")
    assert DEFAULT_CLOCK == WORKING


# -- the real agents, with a real process -----------------------------------------------------------------------------------------------

SLEEPS = [sys.executable, "-c", "import time; time.sleep(60)"]
EXITS = [sys.executable, "-c", "pass"]


@pytest.fixture
def agents(tmp_path, monkeypatch):
    """`claude.ask_claude` and `codex.ask_codex` over a process that is a command of the test's: no real agent starts. `ask(agent, command, **spec)`."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(codex, "models_of", lambda command: {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]})
    command, made = [SLEEPS], iter(range(1000))
    monkeypatch.setattr(claude, "claude_command", lambda: command[0])
    monkeypatch.setattr(codex, "codex_command", lambda: command[0])

    def ask(agent: str, process: list[str] = SLEEPS, **fields) -> dict:
        command[0] = process
        spec = spec_of("blind", tmp_path, run_dir=tmp_path / f"run{next(made)}", **({"model": "gpt-test", "effort": "high"} if agent == "codex" else {}), **fields)
        return (claude.ask_claude if agent == "claude" else codex.ask_codex)(spec)

    return ask


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_a_real_process_is_killed_at_a_limit_of_a_fifth_of_a_second_and_the_result_holds_the_clocks_figures(agents, agent):
    out = agents(agent, timeout_s=0.2)
    assert out["problem"] == "timeout: no answer in 0.2 s"
    assert 0.2 <= out["seconds"] < 30, "killed at the limit, not left to sleep for its minute"
    assert (out["clock"], out["queue_s"], out["working_s"]) == (WORKING, 0.0, out["seconds"]), "a session with no queue: all the wall is working time"


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_with_no_queue_figure_the_result_says_wall_and_the_text_is_the_recorded_one(agents, agent):
    out = agents(agent, timeout_s=0.2, queue_s=unreported)
    assert (out["problem"], out["clock"], out["queue_s"], out["working_s"]) == ("timeout: no answer in 0.2 s", WALL, None, out["seconds"])


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_a_queue_read_live_while_the_process_runs_moves_the_end_of_the_limit_and_the_wall_cap_ends_what_waits_for_ever(agents, agent):
    began = time.monotonic()
    out = agents(agent, timeout_s=0.3, queue_s=lambda: time.monotonic() - began)  # the exec has waited since before it started
    assert out["problem"].startswith("timeout: no answer in 0.9 s on the wall clock (") and out["problem"].endswith(" s of it queued)")
    assert 0.9 <= out["seconds"] < 30 and out["working_s"] == 0.0 and out["queue_s"] >= out["seconds"]

    out = agents(agent, timeout_s=0.5, queue_s=lambda: 0.5)  # half a second of queue: the 0.5 s of working time are up at 1 s of wall, under the cap of 1.5
    assert out["problem"].startswith("timeout: no answer in 0.5 s of working time (") and out["problem"].endswith(" s on the wall clock, 0.5 s queued)")
    assert 1.0 <= out["seconds"] < 30 and out["queue_s"] == 0.5
    assert out["working_s"] == pytest.approx(out["seconds"] - 0.5, abs=0.11), "charged: the wall less the queue (to tenths of a second)"


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_the_clock_starts_with_the_process_and_not_before_the_files_the_agent_makes_ready_for_it(agents, monkeypatch, agent):
    """Time is moved by hand, and 100 s pass inside `prepare` (Codex's first exec asks `codex debug models` there): the seconds of the exec and the
    working time charged are the process's own, 0, and the limit of 30 s is not used up before it starts."""
    now = types.SimpleNamespace(t=1000.0)
    module = claude if agent == "claude" else codex
    monkeypatch.setattr(module, "WorkingClock", types.SimpleNamespace(of=lambda limit, queue: WorkingClock.of(limit, queue, now=lambda: now.t)))
    prepare = module.prepare
    monkeypatch.setattr(module, "prepare", lambda *a, **k: (setattr(now, "t", now.t + 100.0), prepare(*a, **k))[1])
    out = agents(agent, EXITS, timeout_s=30.0)
    assert out["seconds"] == 0.0 and out["working_s"] == 0.0 and not str(out["problem"]).startswith("timeout")


# -- the run: what a result records of the clock -----------------------------------------------------------------------------------------

def test_every_result_of_a_run_holds_the_clocks_figures_and_its_seconds_stay_the_wall_time(world):
    go(world, {"claude": Agent(Stub())}, [FILES, STUB], jobs=2)
    for folder in ("files", "stub-all"):
        for q in QUESTIONS:
            r = result(world.run, "claude", folder, q.qid)
            assert (r["clock"], r["queue_s"], r["seconds"], r["working_s"]) == (WORKING, 0.0, 1.5, 1.5), (folder, q.qid)


def test_a_session_that_reports_a_queue_is_charged_the_wall_less_it_and_one_that_cannot_runs_on_the_wall_at_one_exec_and_is_refused_more(world):
    first, second, third = (q.qid for q in QUESTIONS[:3])
    w = types.SimpleNamespace(**{**vars(world), "system": StubSystem(waits={first: 0.5})})
    go(w, {"claude": Agent(lambda spec: {**Stub()(spec), **clocked(spec, 2.0)})}, [STUB], jobs=2, only=[first])
    r = result(w.run, "claude", "stub-all", first)
    assert (r["clock"], r["queue_s"], r["working_s"], r["seconds"]) == (WORKING, 0.5, 1.5, 2.0), "waited 0.5 s of the 2 s: charged 1.5"

    silent, stub = types.SimpleNamespace(**{**vars(world), "system": StubSystem(reports_queue=False)}), Stub()
    with pytest.raises(ValueError, match="2 execs at once on the working clock"):
        go(silent, {"claude": Agent(stub)}, [STUB], jobs=2, only=[second])
    assert not stub.execs, "refused before an exec"
    go(silent, {"claude": Agent(stub)}, [STUB], jobs=1, only=[second])
    r = result(world.run, "claude", "stub-all", second)
    assert (r["clock"], r["queue_s"], r["working_s"], r["seconds"]) == (WALL, None, 1.5, 1.5), "one at a time has no queue: the wall is the working time"
    go(silent, {"claude": Agent(stub)}, [STUB], jobs=3, only=[third], clock=WALL)
    r = result(world.run, "claude", "stub-all", third)
    assert (r["clock"], r["queue_s"], r["working_s"]) == (WALL, None, 1.5), "asked for the wall clock, at any number of execs at once"


def test_the_spec_an_agent_is_given_reads_its_sessions_queue_live_and_with_the_wall_clock_nothing_is_asked_of_the_session(world):
    first, second = (q.qid for q in QUESTIONS[:2])
    system = StubSystem(waits={first: 0.0, second: 0.0})
    w = types.SimpleNamespace(**{**vars(world), "system": system})
    stub, readings = Stub(), []

    def agent(spec):
        readings.append(spec.queue_s())
        system.waits[spec.run_dir.name] = 7.0  # the session's figure for the exec moves while it runs
        readings.append(spec.queue_s())
        return stub(spec)

    go(w, {"claude": Agent(agent)}, [STUB], jobs=2, only=[first])
    assert readings == [0.0, 7.0]
    go(w, {"claude": Agent(agent)}, [STUB], jobs=2, only=[second], clock=WALL)
    assert readings[2:] == [None, None]


def test_a_result_made_before_the_clock_has_none_of_its_fields_and_is_still_done_and_not_asked_again(world):
    """The recorded runs and the runs of before: `record` makes the result as it was, with none of the clock's fields."""
    stub = Stub()
    for config in (FILES, STUB):
        for q in QUESTIONS:
            made = record(q, "claude", config, 7, f"execs/claude/{config.dir}/{q.qid}", "2026-11-05T00:00:00",
                          {"answer": "13:00", "seconds": 2.0, "problem": None, "tools": []})
            assert not set(made) & set(FIELDS)
            fs.write_json(world.run.result_path("claude", config, q.qid), made)
    go(world, {"claude": Agent(stub)}, [FILES, STUB], jobs=2)
    assert not stub.execs, "all of them are done"
    assert all(is_done(fs.read_json(world.run.result_path("claude", FILES, q.qid))) for q in QUESTIONS)


def test_side_by_side_the_seconds_and_the_working_time_leave_out_the_wait_for_the_agents_gate_and_the_clock_starts_after_it(world, monkeypatch):
    """An agent has one session at a time beside other cells, so the second exec waits at the gate while the first holds it. Time is moved by hand:
    1,000 s pass while the first exec holds the gate, and the real agent's clock, which starts when it starts its process, shows none of them in
    either exec."""
    now = types.SimpleNamespace(t=500.0)
    monkeypatch.setattr(claude, "WorkingClock", types.SimpleNamespace(of=lambda limit, queue: WorkingClock.of(limit, queue, now=lambda: now.t)))
    monkeypatch.setattr(claude, "claude_command", lambda: EXITS)
    ran, lock = [], threading.Lock()
    qids = [q.qid for q in QUESTIONS[:2]]

    def agent(spec):
        with lock:
            ran.append(spec.run_dir.name)
            first = len(ran) == 1
        if first:
            now.t += 1000.0  # (the other exec is at the gate, waiting for this one to end)
        out = claude.ask_claude(spec)
        return {**Stub()(spec), **{name: out[name] for name in ("seconds", *FIELDS)}}

    go(world, {"claude": Agent(agent)}, [STUB], jobs=2, only=qids, side_by_side=SideBySide(Pools(cpu=2, agents={"claude": 1}), {}))
    assert sorted(ran) == sorted(qids)
    for qid in qids:
        r = result(world.run, "claude", "stub-all", qid)
        assert (r["seconds"], r["working_s"], r["clock"], r["queue_s"]) == (0.0, 0.0, WORKING, 0.0), qid
