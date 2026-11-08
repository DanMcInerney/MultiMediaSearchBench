"""Tests of side-by-side scheduling (plan 6.4, DESIGN 8.3), written for this repository (U13): the pools on paper (`simulate`), the gate
their threads wait at, `run_beside`, and the order of a cell's cases. The run loop with them, the rule that keeps the agents off the GPU and the
command line are tests/test_schedule_run.py's and test_schedule_cli.py's.

The expected values are not the code's: a small timeline worked out by hand (in its comment), laws that every input obeys (checked over
seeded random runs by a checker that reads only the timeline, and shown to reject planted wrong ones), a simple model run in lockstep (the
makespan of cells one after another is a list scheduler of a few lines), and the plan's words (which case starts first, who waits)."""

from __future__ import annotations

import dataclasses
import heapq
import random
import threading
from collections import Counter
from pathlib import Path

import pytest
from stub_system import StubSystem

from mmsb.core.contract import Case
from mmsb.core.record import Halt
from mmsb.core.runfolder import JOBS_MAX
from mmsb.core.schedule import (Beside, Expected, Gate, Line, Pools, Span, Task, Timeline, agent_off_gpu, keeps_time_order, run_beside,
                                simulate, start_order)

WAIT = 10.0  # seconds a test waits for something that must happen (a failure, not a slow machine, when it does not)
EPS = 1e-9


# -- the checker of a timeline, and the model it is compared with ------------------------------------------------------------------

def sequential(lines, durations, pools) -> float:
    """The seconds the cells take one after another, each with as many execs at once as it may have: a list scheduler (the next case goes
    to the exec that is free first)."""
    total = 0.0
    for line in lines:
        free = [0.0] * min(line.jobs, pools.cpu, pools.agents[line.agent])
        end = 0.0
        for qid in line.cases:
            start = heapq.heappop(free)
            heapq.heappush(free, start + durations[line.name, qid])
            end = max(end, start + durations[line.name, qid])
        total += end
    return total


def problems(lines, durations, pools, timeline: Timeline) -> list[str]:
    """What a timeline gets wrong of the plan's laws: each case once for its duration and started in its cell's order; never more than a
    cell's `jobs`, an agent's sessions or the execs at once; never more GPU cells than the GPU's, each holding it for all of its cases, in
    order; and no slot free (or GPU free) while a case (or a cell) that could use it waits. Reads the timeline alone."""
    out, spans = [], timeline.spans
    if Counter((s.cell, s.qid) for s in spans) != Counter((line.name, q) for line in lines for q in line.cases):
        return ["the cases that ran are not the cases there were"]
    out += [f"{s.cell} {s.qid} takes {s.end - s.start}, not {durations[s.cell, s.qid]}" for s in spans
            if abs(s.end - s.start - durations[s.cell, s.qid]) > EPS]
    start = {(s.cell, s.qid): s.start for s in spans}
    for line in lines:
        starts = [start[line.name, q] for q in line.cases]
        out += [f"{line.name} starts its cases out of order"] * (starts != sorted(starts))
    holds = {h.cell: h for h in timeline.holds}
    gpu_lines = [line for line in lines if line.needs_gpu and line.cases]
    if set(holds) != {line.name for line in gpu_lines} or len(timeline.holds) != len(gpu_lines):
        return out + ["each cell with cases that needs the GPU holds it once, and no other does"]
    for line in gpu_lines:
        mine = [s for s in spans if s.cell == line.name]
        hold = holds[line.name]
        if hold.start > min(s.start for s in mine) + EPS or abs(hold.end - max(s.end for s in mine)) > EPS:
            out.append(f"{line.name} does not hold the GPU for all of its cases")
    out += [f"{a.name} takes the GPU before {b.name}, which comes first" for a, b in zip(gpu_lines[1:], gpu_lines, strict=False)
            if holds[a.name].start < holds[b.name].start - EPS]
    times = sorted({s.start for s in spans} | {s.end for s in spans} | {h.start for h in timeline.holds})
    for t in times:
        running = [s for s in spans if s.start <= t < s.end]
        by_cell, by_agent = Counter(s.cell for s in running), Counter(next(x.agent for x in lines if x.name == s.cell) for s in running)
        holding = [h for h in timeline.holds if h.start <= t < h.end]
        out += [f"at {t:g} {c} runs {n} execs" for c, n in by_cell.items() if n > next(x.jobs for x in lines if x.name == c)]
        out += [f"at {t:g} {a} has {n} sessions" for a, n in by_agent.items() if n > pools.agents[a]]
        out += [f"at {t:g} {len(running)} execs run"] * (len(running) > pools.cpu)
        out += [f"at {t:g} {len(holding)} cells hold the GPU"] * (len(holding) > pools.gpu)
        out += [f"at {t:g} the GPU is free while {line.name} waits for it" for line in gpu_lines
                if len(holding) < pools.gpu and holds[line.name].start > t + EPS]
        for line in lines:
            waiting = [s for s in spans if s.cell == line.name and s.start > t + EPS]
            open_ = not line.needs_gpu or (line.name in holds and holds[line.name].start <= t + EPS)
            if (waiting and open_ and by_cell[line.name] < line.jobs and by_agent[line.agent] < pools.agents[line.agent]
                    and len(running) < pools.cpu):
                out.append(f"at {t:g} a slot is free for {line.name}, whose next case waits until {min(s.start for s in waiting):g}")
    out += [f"the makespan {timeline.makespan} is more than the seconds of all the cases"] * (
        timeline.makespan > sum(durations.values()) + EPS)
    return out


def random_run(rng: random.Random, pools: Pools | None = None, binding: bool = True):
    """Cells, the seconds of each case (some zero, some equal), and pools. With `binding` False the cells' jobs fit the pools, so no pool
    ever makes a case wait: only the GPU does."""
    agents = ("claude", "codex")
    pools = pools or Pools(cpu=rng.randint(1, JOBS_MAX), agents={a: rng.randint(1, JOBS_MAX) for a in agents}, gpu=1)
    lines, durations = [], {}
    room = {"cpu": pools.cpu, **pools.agents}
    for i in range(rng.randint(1, 5)):
        agent = rng.choice(agents)
        jobs = rng.randint(1, JOBS_MAX)
        if not binding:
            jobs = min(jobs, room["cpu"], room[agent])
            if jobs < 1:
                continue
            room["cpu"] -= jobs
            room[agent] -= jobs
        line = Line(f"c{i}", agent, rng.random() < 0.4, jobs, tuple(f"q{k}" for k in range(rng.randint(0, 8))))
        lines.append(line)
        durations.update({(line.name, q): rng.choice([0.0, 1.0, rng.uniform(0.5, 30.0)]) for q in line.cases})
    return lines, durations, pools


# -- simulate ----------------------------------------------------------------------------------------------------------------------

def test_a_small_run_has_the_timeline_worked_out_by_hand():
    """claude's pool is 2 sessions. `claude mm` needs the GPU (one exec at a time, two cases of 10 s); `claude files` has four cases of 4 s,
    two at once; `codex mm` needs the GPU too (one case of 5 s) and has its own pool of 1.
      t=0   the GPU goes to claude mm, whose m1 takes a session (the cell that needs the GPU asks first); files f1 takes the other
      t=4   f1 ends; files f2 takes its session (m2 is not ready: claude mm has one exec at once)
      t=8   f2 ends; f3 starts
      t=10  m1 ends; m2 takes the session m1 left, before f4 could have it; f3 and m2 are the two sessions
      t=12  f3 ends; f4 starts
      t=16  f4 ends
      t=20  m2 ends; claude mm gives up the GPU, and codex mm takes it: c1 runs 20 to 25
    One after another: 20 (claude mm) + 8 (files: 4+4 in two) + 5 (codex mm) = 33; side by side 25."""
    lines = [Line("claude mm", "claude", True, 1, ("m1", "m2")), Line("claude files", "claude", False, 2, ("f1", "f2", "f3", "f4")),
             Line("codex mm", "codex", True, 1, ("c1",))]
    durations = {("claude mm", "m1"): 10.0, ("claude mm", "m2"): 10.0, **{("claude files", f"f{k}"): 4.0 for k in (1, 2, 3, 4)},
                 ("codex mm", "c1"): 5.0}
    pools = Pools(cpu=4, agents={"claude": 2, "codex": 1}, gpu=1)
    timeline = simulate(lines, durations, pools)
    assert timeline.spans == (
        Span("claude mm", "m1", 0.0, 10.0), Span("claude files", "f1", 0.0, 4.0), Span("claude files", "f2", 4.0, 8.0),
        Span("claude files", "f3", 8.0, 12.0), Span("claude mm", "m2", 10.0, 20.0), Span("claude files", "f4", 12.0, 16.0),
        Span("codex mm", "c1", 20.0, 25.0))
    assert timeline.holds == (Span("claude mm", None, 0.0, 20.0), Span("codex mm", None, 20.0, 25.0))
    assert timeline.makespan == 25.0 and sequential(lines, durations, pools) == 33.0
    assert problems(lines, durations, pools, timeline) == []


def test_when_a_pool_is_short_the_cell_that_needs_the_gpu_is_served_first_whatever_its_place():
    """One session of claude: `files` (first in the order) has two cases of 3 s, `mm` (needs the GPU) one of 5. The GPU cell asks first, so m1 runs
    0 to 5, then f1 5 to 8 and f2 8 to 11. (Served by place it would be f1 0 to 3, f2 3 to 6, m1 6 to 11, with the GPU idle for 6 s.)"""
    lines = [Line("files", "claude", False, 2, ("f1", "f2")), Line("mm", "claude", True, 1, ("m1",))]
    durations = {("files", "f1"): 3.0, ("files", "f2"): 3.0, ("mm", "m1"): 5.0}
    timeline = simulate(lines, durations, Pools(cpu=4, agents={"claude": 1}))
    assert timeline.spans == (Span("mm", "m1", 0.0, 5.0), Span("files", "f1", 5.0, 8.0), Span("files", "f2", 8.0, 11.0))
    assert timeline.holds == (Span("mm", None, 0.0, 5.0),)


def test_a_pool_of_execs_makes_the_cells_of_two_agents_wait_for_each_other():
    """cpu 2: claude files (two execs at once, two cases of 3 s) takes both slots, and codex files waits: 3 to 6. With cpu 4 they run
    together and the run takes 3."""
    lines = [Line("claude files", "claude", False, 2, ("a", "b")), Line("codex files", "codex", False, 2, ("a", "b"))]
    durations = dict.fromkeys([("claude files", "a"), ("claude files", "b"), ("codex files", "a"), ("codex files", "b")], 3.0)
    narrow = simulate(lines, durations, Pools(cpu=2, agents={"claude": 4, "codex": 4}))
    assert sorted((s.cell, s.start, s.end) for s in narrow.spans) == [("claude files", 0.0, 3.0)] * 2 + [("codex files", 3.0, 6.0)] * 2
    assert simulate(lines, durations, Pools.of(["claude", "codex"])).makespan == 3.0


def test_a_run_with_nothing_to_do_takes_no_time_and_an_unknown_duration_or_agent_is_an_error():
    pools = Pools.of(["claude"])
    assert simulate([], {}, pools) == Timeline((), ()) and simulate([Line("a", "claude", True, 2, ())], {}, pools).holds == ()
    with pytest.raises(KeyError):
        simulate([Line("a", "claude", False, 1, ("q",))], {}, pools)
    with pytest.raises(KeyError):
        simulate([Line("a", "codex", False, 1, ("q",))], {("a", "q"): 1.0}, pools)


@pytest.mark.parametrize("gpu", (1, 2))
def test_simulate_obeys_its_laws_on_random_runs(gpu):
    """Over 400 seeded runs of up to five cells: the cases once, in order; the limits of cells, agents, execs and the GPU; the GPU held for
    a cell's whole length, in cell order; and nothing idle that could work."""
    wrong = []
    for seed in range(400):
        rng = random.Random(seed)
        lines, durations, pools = random_run(rng)
        pools = dataclasses.replace(pools, gpu=gpu)
        wrong += [(seed, p) for p in problems(lines, durations, pools, simulate(lines, durations, pools))]
    assert not wrong, wrong[:5]


def test_side_by_side_is_never_slower_than_one_after_another_when_no_pool_makes_a_case_wait():
    """Then the cells are independent but for the GPU, and the run takes the longest of the GPU cells' total and the free cells: no more than
    the sum. (A theorem for this case only: see the next test.)"""
    for seed in range(400):
        lines, durations, pools = random_run(random.Random(seed), binding=False)
        assert simulate(lines, durations, pools).makespan <= sequential(lines, durations, pools) + EPS, seed


def test_with_pools_that_bind_it_is_hardly_ever_slower_and_never_by_much():
    """A greedy share of slots can lose to cells one after another (Graham's anomaly: a short case takes a slot just before the long one that
    needed it): in the seeded runs below that happens in a handful of 3,000, by a few percent. The test is the rate and the size."""
    slower = []
    for seed in range(3000):
        lines, durations, pools = random_run(random.Random(seed))
        one_after_another = sequential(lines, durations, pools)
        got = simulate(lines, durations, pools).makespan
        if got > one_after_another + EPS:
            slower.append((seed, round(got / one_after_another - 1, 3)))
    assert len(slower) <= 3 and all(excess < 0.15 for _, excess in slower), slower


def test_the_checker_rejects_a_timeline_that_breaks_a_law():
    lines = [Line("g", "claude", True, 1, ("a",)), Line("h", "codex", True, 1, ("a",)), Line("f", "claude", False, 2, ("a", "b"))]
    durations = dict.fromkeys([("g", "a"), ("h", "a"), ("f", "a"), ("f", "b")], 4.0)
    pools = Pools(cpu=4, agents={"claude": 2, "codex": 2})
    good = simulate(lines, durations, pools)
    assert problems(lines, durations, pools, good) == []

    def planted(spans, holds=good.holds) -> list[str]:
        return problems(lines, durations, pools, Timeline(tuple(spans), tuple(holds)))

    s = {(x.cell, x.qid): x for x in good.spans}
    both = [s["g", "a"], dataclasses.replace(s["h", "a"], start=0.0, end=4.0), s["f", "a"], s["f", "b"]]
    assert any("cells hold the GPU" in p for p in planted(both, [good.holds[0], Span("h", None, 0.0, 4.0)])), "two GPU cells at once"
    assert any("sessions" in p for p in planted([*both[:1], dataclasses.replace(s["f", "a"], start=0.0, end=4.0),
                                                  dataclasses.replace(s["f", "b"], start=0.0, end=4.0), s["h", "a"]])), "over an agent's limit"
    late = dataclasses.replace(s["f", "b"], start=s["f", "b"].start + 1, end=s["f", "b"].end + 1)
    assert any("slot is free" in p for p in planted([s["g", "a"], s["h", "a"], s["f", "a"], late])), "a slot idle while a case waits"
    short = dataclasses.replace(s["f", "a"], end=s["f", "a"].end - 1)
    assert any("takes" in p for p in planted([s["g", "a"], s["h", "a"], short, s["f", "b"]])), "a case that took less than it did"
    assert any("does not hold" in p for p in planted(good.spans, [Span("g", None, 0.0, 2.0), good.holds[1]])), "the GPU given up early"
    assert planted(good.spans[:-1]), "a case that never ran"
    assert any("out of order" in p for p in planted([s["g", "a"], s["h", "a"], dataclasses.replace(s["f", "a"], start=1.0, end=5.0),
                                                      dataclasses.replace(s["f", "b"], start=0.0, end=4.0)])), "a cell's cases out of order"


# -- the order of a cell's cases, and the rule for the agents ------------------------------------------------------------------------

def case(qid: str, until: float = 0.0) -> Case:
    return Case(qid, 0.0, None, ("s",), 0.0, until, qid, 0.0, -1, Path(), Path())


def test_the_cases_expected_to_take_longest_start_first_and_equal_ones_keep_their_order():
    pending = [case(q, i) for i, q in enumerate("abcdefg")]
    expected = Expected({"a": 5.0, "b": 50.0, "c": 5.0, "d": 20.0, "e": 50.0}, typical=7.0)  # f and g have none: the cell's typical, 7
    view = StubSystem().describe("media")
    got = [c.qid for c in start_order(pending, dataclasses.replace(view, horizon="view"), expected)]
    assert got == ["b", "e", "d", "f", "g", "a", "c"]
    assert [c.qid for c in start_order(pending, None, expected)] == got, "a cell with no system (blind) is ordered the same"


def test_a_cell_with_no_history_and_a_cell_whose_system_cuts_its_index_keep_the_order_of_their_time():
    pending = [case(q, i) for i, q in enumerate("abc")]
    expected = Expected({"a": 1.0, "b": 2.0, "c": 3.0}, typical=1.0)
    free = dataclasses.replace(StubSystem().describe("all"), horizon="view")
    cut = StubSystem(needs_gpu=True).describe("all")
    assert cut.horizon == "cut" and keeps_time_order(cut) and not keeps_time_order(free) and not keeps_time_order(None)
    assert [c.qid for c in start_order(pending, free, None)] == ["a", "b", "c"], "no history"
    assert [c.qid for c in start_order(pending, cut, expected)] == ["a", "b", "c"], "AS_OF: a GPU cell"
    gpu_free_cut = dataclasses.replace(cut, needs_gpu=False)
    assert [c.qid for c in start_order(pending, gpu_free_cut, expected)] == ["a", "b", "c"], "AS_OF does not depend on the GPU"
    assert [c.qid for c in start_order(pending, free, expected)] == ["c", "b", "a"]


def test_the_agents_own_tools_are_kept_off_the_gpu_beside_other_cells_or_when_their_system_needs_it():
    gpu = StubSystem(needs_gpu=True).describe("all")
    no_gpu = StubSystem().describe("all")
    for own in (False, True):
        for description in (None, no_gpu, gpu):
            for beside in (False, True):
                want = own and (beside or (description is gpu))  # the plan's words (6.4, C11)
                assert agent_off_gpu(own, description, beside) is want, (own, description and description.needs_gpu, beside)


def test_pools_are_the_laptops_limits():
    assert Pools.of(["claude", "codex"]) == Pools(cpu=JOBS_MAX, agents={"claude": JOBS_MAX, "codex": JOBS_MAX}, gpu=1) and JOBS_MAX == 4
    for bad in (dict(cpu=0, agents={"a": 1}), dict(cpu=JOBS_MAX + 1, agents={"a": 1}), dict(cpu=1, agents={}),
                dict(cpu=1, agents={"a": 0}), dict(cpu=1, agents={"a": JOBS_MAX + 1}), dict(cpu=1, agents={"a": 1}, gpu=0)):
        with pytest.raises(ValueError, match="pools are"):
            Pools(**bad)


# -- the gate and run_beside -------------------------------------------------------------------------------------------------------

def test_a_gate_gives_its_tokens_to_the_lowest_priority_then_the_first_to_ask_whatever_the_order_they_come_in():
    gate = Gate(range(1))
    holder = gate.reserve((9,))
    holder.__enter__()
    served, tickets = [], [gate.reserve(p) for p in ((2,), (1,), (2,), (0,))]
    order = ["(2,)#a", "(1,)#b", "(2,)#c", "(0,)#d"]

    def wait(name, ticket):
        with ticket:
            served.append(name)

    threads = [threading.Thread(target=wait, args=(n, t)) for n, t in zip(order, tickets, strict=True)]
    for thread in reversed(threads):  # the threads come in the opposite order to the reservations
        thread.start()
    holder.__exit__(None, None, None)
    for thread in threads:
        thread.join(WAIT)
    assert served == ["(0,)#d", "(1,)#b", "(2,)#a", "(2,)#c"]


def test_a_gate_gives_the_lowest_free_token_and_takes_it_back():
    gate = Gate(range(3))
    a, b, c = gate.reserve((0,)), gate.reserve((0,)), gate.reserve((0,))
    with a as x, b as y, c as z:
        assert (x, y, z) == (0, 1, 2)
    with gate.reserve((0,)) as one:
        assert one == 0
    with gate.reserve((0,)) as one, gate.reserve((0,)) as two:
        assert (one, two) == (0, 1)


def test_a_ticket_that_gives_up_its_turn_does_not_hold_the_others_back():
    gate = Gate(range(1))
    first, second, third = gate.reserve((0,)), gate.reserve((1,)), gate.reserve((2,))
    gate.drop(second.key)
    done = []

    def wait():
        with third:
            done.append(1)

    thread = threading.Thread(target=wait)
    with first:
        thread.start()
    thread.join(WAIT)
    assert done == [1] and not thread.is_alive()


def tasks_run(tasks, pools):
    """`run_beside` on its own thread, with a wait for it to end: a hang is a failure, not a stuck suite."""
    outcome = {}

    def go():
        try:
            run_beside(tasks, pools)
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = error

    thread = threading.Thread(target=go)
    thread.start()
    return thread, outcome


def test_a_free_cell_runs_while_a_cell_holds_the_gpu_and_the_second_gpu_cell_waits_for_the_first_to_end():
    in_a, free_done, release_a, b_in = (threading.Event() for _ in range(4))
    log, lock, priorities = [], threading.Lock(), []

    def note(what):
        with lock:
            log.append(what)

    def a(beside):
        note("a in")
        priorities.append(beside.priority)
        in_a.set()
        assert free_done.wait(WAIT), "the free cell did not run while the GPU cell was held open"
        assert release_a.wait(WAIT)
        note("a out")

    def free(beside):
        assert in_a.wait(WAIT)
        note("free")
        priorities.append(beside.priority)
        free_done.set()

    def b(beside):
        note("b in")
        priorities.append(beside.priority)
        b_in.set()

    thread, outcome = tasks_run([Task("a", True, None, a), Task("free", False, None, free), Task("b", True, None, b)], Pools.of(["claude"]))
    assert free_done.wait(WAIT) and in_a.is_set()
    assert not b_in.wait(0.3), "the second GPU cell started while the first held the GPU"  # (the one wait for something not to happen)
    release_a.set()
    thread.join(WAIT)
    assert not thread.is_alive() and not outcome
    assert log == ["a in", "free", "a out", "b in"]
    assert priorities == [(0, 0), (1, 1), (0, 2)], "the cell that needs the GPU asks first, then the cell's place"


def test_an_error_in_one_cell_stops_the_others_from_starting_more_the_first_error_is_raised_and_the_next_gpu_cell_finds_the_stop(capsys):
    seen, gpu_two = [], []

    def breaks(beside):
        raise ValueError("first")

    def next_gpu(beside):
        gpu_two.append(beside.stop.reason)

    def other(beside):
        seen.append("ran")

    pools = Pools.of(["claude"])
    thread, outcome = tasks_run([Task("one", True, None, breaks), Task("two", True, None, next_gpu), Task("free", False, None, other)], pools)
    thread.join(WAIT)
    assert not thread.is_alive()
    assert isinstance(outcome["error"], ValueError) and str(outcome["error"]) == "first"
    assert gpu_two == ["one ended on an error"], "the stop was set before the GPU was given to the next cell"
    assert "stopping: one ended on an error" in capsys.readouterr().out


def test_when_several_cells_fail_the_first_is_raised_and_the_others_are_told(capsys, monkeypatch):
    stopped = threading.Event()
    real = Halt.set

    def spy(self, reason):
        real(self, reason)
        stopped.set()

    monkeypatch.setattr(Halt, "set", spy)

    def first(beside: Beside):
        raise ValueError("first")

    def second(beside: Beside):
        assert stopped.wait(WAIT), "the first error did not stop the others"
        raise ValueError("second")

    thread, outcome = tasks_run([Task("a", False, None, second), Task("b", False, None, first)], Pools.of(["claude"]))
    thread.join(WAIT)
    assert not thread.is_alive() and str(outcome["error"]) == "first"
    assert "a: ended on an error as well: ValueError('second')" in capsys.readouterr().out
