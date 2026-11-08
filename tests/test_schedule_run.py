"""The run loop beside other cells (plan 6.4, U13): `run_agents` with `--side-by-side` over the stand-in system of tests/stub_system.py, with
real threads that wait on `threading.Event`s, not on time; and the same run without it, which is the run of before. Also the rule that keeps
the agents' own tools off the GPU (plan 6.4, C11), from the specs the run hands the agents to the environment and command line the real agents
make of them. Written for this repository (U13).

The expected values are the plan's words (a files case finishes while a GPU cell is held open; a second GPU cell waits for the first to end,
its `close` included; a Halt stops the cells of its agent and not another's; a lost GPU marks no free cell; `files` has no GPU rule unless it runs
beside others, `mm+files` always has it), the order the stub is known to run in, and what the test planted. A hang is a failure: each run is
on a thread the test gives 60 s."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from unittest import mock

import pytest
from stub_system import FILES, STUB, Stub, StubSession, StubSystem, tiny_questions
from test_runner import NARROW, go, result, world  # noqa: F401

from mmsb.core import fs, gpu
from mmsb.core.agents import claude, codex
from mmsb.core.config import Config
from mmsb.core.record import Halt
from mmsb.core.runner import Agent, cells_in_order
from mmsb.core.schedule import Expected, Pools, SideBySide

pytestmark = pytest.mark.xdist_group("runner")
WAIT = 10.0
QUESTIONS = tiny_questions()
N = len(QUESTIONS)
STUB_FILES = Config("stub/all+files", "stub", "all", True)  # the stand-in system's tools and the agent's own: `mm+files`
IN_TIME = sorted(QUESTIONS, key=lambda q: (q.scope.until, q.qid))
LIMIT = {"problem": "limit: weekly", "final": ""}
GPU_WORDS = {"in:-hwaccel", "in:nvidia-smi"}  # the plan's 6.4: what an agent kept off the GPU may not name (DESIGN 8.3)
NARROW_QIDS = {q.qid for q in QUESTIONS if q.tags["modality"] != "audio"}  # what the variant `narrow` of the stub system reads


def beside(*agents: str, expected: dict | None = None, pools: Pools | None = None) -> SideBySide:
    return SideBySide(pools or Pools.of(agents or ["claude"]), expected or {})


def finish(call) -> None:
    """`call` on a thread, which must end in 60 s; what it raised is raised here."""
    outcome = {}

    def target():
        try:
            call()
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = error

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the run did not end: a cell hangs"
    if "error" in outcome:
        raise outcome["error"]


def on_gpu(world, patch, **kw):
    """The world with the stand-in system needing the GPU, and a GPU that is free and not lost."""
    patch.setattr(gpu, "gpu_busy", lambda: None)
    patch.setattr(gpu, "gpu_lost", lambda: None)
    world.system = StubSystem(needs_gpu=True, **kw)
    return world


class Hooked(Stub):
    """A stub that calls `before(spec)` as its exec starts and `after(spec)` as it ends, so that a test can hold an exec open on an Event,
    and keeps (work folder, start, end) of each exec."""

    def __init__(self, before=lambda spec: None, after=lambda spec: None, **kw):
        super().__init__(**kw)
        self.before, self.after, self.ran = before, after, []

    def __call__(self, spec):
        self.before(spec)
        began = time.monotonic()
        out = super().__call__(spec)
        with self.lock:
            self.ran.append((spec.work, began, time.monotonic()))
        self.after(spec)
        return out


def overlap(log) -> bool:
    """Whether two execs of different cells (the folder of the results) were ever running at once."""
    return any(a[0] != b[0] and a[3] < b[4] and b[3] < a[4] for a in log for b in log)


def most_at_once(intervals) -> int:
    """The most intervals (start, end) that contain one moment: counted at the start of each."""
    return max((sum(1 for s, e in intervals if s <= t < e) for t, _ in intervals), default=0)


# -- cells beside each other -------------------------------------------------------------------------------------------------------

def test_a_files_case_finishes_while_a_gpu_cell_is_held_open(world, monkeypatch):
    w = on_gpu(world, monkeypatch)
    held, files_done = threading.Event(), threading.Event()

    def before(spec):
        if Stub.dir_of(spec) == "stub-all" and not held.is_set():
            held.set()
            assert files_done.wait(WAIT), "no files case finished while the GPU cell was held open"
        elif Stub.dir_of(spec) == "files":
            assert held.wait(WAIT)

    stub = Hooked(before, lambda spec: files_done.set() if Stub.dir_of(spec) == "files" else None)
    finish(lambda: go(w, {"claude": Agent(stub)}, [FILES, STUB], jobs=1, side_by_side=beside()))
    assert files_done.is_set() and len(stub.qids("files")) == len(stub.qids("stub-all")) == N
    assert all(result(w.run, "claude", folder, q.qid)["problem"] is None for folder in ("files", "stub-all") for q in QUESTIONS)


def test_a_second_gpu_cell_does_not_open_its_session_until_the_first_has_closed_its_own(world, monkeypatch):
    b_opened = threading.Event()

    class SlowClose(StubSession):
        def close(self):
            if self.context.variant == "all":
                b_opened.wait(0.3)  # a second GPU cell that was not held back opens now (the one wait for something not to happen)
            return super().close()

    class Watched(StubSystem):
        def open(self, store, dump, context):
            self.note("open", None, context.agent, context.variant, store)
            if context.variant == "narrow":
                b_opened.set()
            return SlowClose(self, context)

    w = on_gpu(world, monkeypatch)
    w.system = Watched(needs_gpu=True)
    stub = Stub()
    finish(lambda: go(w, {"claude": Agent(stub)}, [STUB, NARROW], jobs=2, side_by_side=beside()))
    assert [e[0] for e in w.system.events if e[0] in ("open", "close")] == ["open", "close", "open", "close"]
    assert [e[3] for e in w.system.kinds("open")] == ["all", "narrow"], "the cells took the GPU in the order they would have run in"
    assert len(stub.qids("stub-all")) == N and set(stub.qids("stub-narrow")) == NARROW_QIDS


def test_a_halt_stops_the_other_cells_of_its_agent_from_starting_more_and_not_the_cells_of_another_agent(world, monkeypatch):
    w = world
    files_in, halted = threading.Event(), threading.Event()
    real = Halt.set

    def spy(self, reason):
        real(self, reason)
        if "limit" in reason:
            halted.set()

    monkeypatch.setattr(Halt, "set", spy)

    def before(spec):
        if Stub.dir_of(spec) == "files":
            files_in.set()
            assert halted.wait(WAIT), "the limit of the other cell did not stop this one"
        else:
            assert files_in.wait(WAIT)  # (the files exec is in flight when the plan runs out)

    claude_stub = Hooked(before, by={("stub-all", q.qid): LIMIT for q in QUESTIONS})
    codex_stub = Stub()
    finish(lambda: go(w, {"claude": Agent(claude_stub), "codex": Agent(codex_stub)}, [FILES, STUB], jobs=1, side_by_side=beside("claude", "codex")))
    assert len(claude_stub.qids("files")) == 1 and len(claude_stub.qids("stub-all")) == 1, "the exec in flight ended, no other started"
    assert len(codex_stub.qids("files")) == len(codex_stub.qids("stub-all")) == N, "another agent's plan is its own"
    assert len(w.system.kinds("open")) == len(w.system.kinds("close")) == 2, "every session that opened was closed"


def test_an_error_in_one_cell_stops_every_cell_closes_its_session_and_the_first_error_is_the_one_raised(world, capsys, monkeypatch):
    w = world
    w.system = StubSystem(leak="boom", close_raises="close failed")
    files_in, stopping = threading.Event(), threading.Event()
    real = Halt.set

    def spy(self, reason):
        real(self, reason)
        if "ended on an error" in reason:
            stopping.set()

    monkeypatch.setattr(Halt, "set", spy)

    def before(spec):
        if Stub.dir_of(spec) == "files":
            files_in.set()
            assert stopping.wait(WAIT), "the error of the other cell did not stop this one"
        else:
            assert files_in.wait(WAIT)

    stub = Hooked(before)
    with pytest.raises(RuntimeError, match="boom"):
        finish(lambda: go(w, {"claude": Agent(stub)}, [FILES, STUB], jobs=1, side_by_side=beside()))
    assert len(stub.qids("files")) == 1, "the exec in flight ended, no other started"
    assert len(w.system.kinds("open")) == len(w.system.kinds("close")) == 1
    out = capsys.readouterr().out
    assert "closing its session failed as well: close failed" in out and "stopping: claude stub/all ended on an error" in out


def test_a_lost_gpu_marks_the_cell_that_needs_it_and_no_free_cell(world, monkeypatch):
    w = on_gpu(world, monkeypatch)
    monkeypatch.setattr(gpu, "gpu_lost", lambda: "GPU is lost")
    files_done = threading.Event()
    stub = Hooked(lambda spec: Stub.dir_of(spec) == "stub-all" and files_done.wait(WAIT),
                  lambda spec: files_done.set() if Stub.dir_of(spec) == "files" else None)
    finish(lambda: go(w, {"claude": Agent(stub)}, [FILES, STUB], jobs=1, side_by_side=beside()))
    files = [r for r in (w.run.folder / "results" / "claude" / "files").glob("*.json")]
    stubbed = [r for r in (w.run.folder / "results" / "claude" / "stub-all").glob("*.json")]
    assert files and stubbed
    assert all(result(w.run, "claude", "files", f.stem)["problem"] is None for f in files), "a file cell has no GPU to lose"
    assert all(result(w.run, "claude", "stub-all", f.stem)["problem"].startswith("setup: the GPU was lost") for f in stubbed)


# -- without the flag: the run of before ---------------------------------------------------------------------------------------------

def test_without_the_flag_no_two_cells_are_ever_running_at_once(world):
    stub = Stub(pause=0.01)
    go(world, {"claude": Agent(stub)}, [FILES, STUB], jobs=2)
    assert len(stub.log) == 2 * N and not overlap(stub.log)


def test_the_overlap_check_sees_cells_that_do_run_at_once(world):
    stub = Stub(pause=0.02)
    go(world, {"claude": Agent(stub)}, [FILES, STUB], jobs=2, side_by_side=beside())
    assert len(stub.log) == 2 * N and overlap(stub.log)


def result_lines() -> list[str]:
    """The line of each result, as the plan has it (`<agent> <folder> <qid>: pass|FAIL <seconds> s`), in the order of the cells and of time, for
    the stub: it answers right, but `unknown` to a picture, a video or a sound when it has only its own tools."""
    media = ("image", "video", "audio")

    def verdict(q, folder):
        return "pass" if folder == "stub-all" or q.tags["modality"] not in media or q.tags["absent"] else "FAIL"

    return [f"claude {folder} {q.qid}: {verdict(q, folder)} 1.5 s" for folder in ("files", "stub-all") for q in IN_TIME]


def test_the_printed_lines_of_a_run_without_the_flag_are_its_result_lines_in_the_order_of_the_cells_and_of_time(world, capsys):
    go(world, {"claude": Agent(Stub())}, [FILES, STUB], jobs=1)
    assert capsys.readouterr().out.splitlines() == result_lines()


def test_with_the_flag_the_run_says_it_is_on_and_then_prints_the_same_result_lines(world, capsys):
    side = beside(expected={("claude", "files"): Expected({}, 1.0)})
    go(world, {"claude": Agent(Stub())}, [FILES, STUB], jobs=1, side_by_side=side)
    first, *rest = capsys.readouterr().out.splitlines()
    assert first.startswith("side by side: on") and "at most 4 execs at once" in first and "4 of claude" in first
    assert Counter(rest) == Counter(result_lines()), "a cell that needs nothing of the others prints what it printed"


def test_the_cells_of_a_run_are_the_configurations_with_no_variant_of_each_agent_then_each_variant_for_each_agent():
    """The order the loops of `run_agents` had before the flag (bench.py at bb498a6: `for agent: for config with no variant`, then `for variant: for
    agent: for config of the variant`), which a side-by-side run takes as the order the GPU cells wait their turn in."""
    x, x_files, y = Config("mm/x", "mm", "x", False), Config("mm/x+files", "mm", "x", True), Config("mm/y", "mm", "y", False)
    configs = [x, FILES, y, x_files]
    got = [(agent, config.name) for agent, config in cells_in_order(["claude", "codex"], configs)]
    assert got == [("claude", "files"), ("codex", "files"), ("claude", "mm/x"), ("claude", "mm/x+files"), ("codex", "mm/x"),
                   ("codex", "mm/x+files"), ("claude", "mm/y"), ("codex", "mm/y")]


# -- the pools ---------------------------------------------------------------------------------------------------------------------

def test_the_execs_of_all_the_cells_take_working_folders_of_their_own_and_are_four_at_most(world):
    stub = Hooked(pause=0.03)
    go(world, {"claude": Agent(stub)}, [FILES, STUB, NARROW], jobs=3, side_by_side=beside())
    assert len(stub.ran) == 2 * N + len(NARROW_QIDS)
    assert most_at_once([(s, e) for _, s, e in stub.ran]) == 4, "three cells of three execs ask for 9: the laptop's 4 are used"
    for i, (work, start, end) in enumerate(stub.ran):
        assert not [w for w, s, e in stub.ran[i + 1:] if w == work and s < end and start < e], "two execs at once in one working folder"


@pytest.mark.parametrize("limit", (1, 2))
def test_one_agents_sessions_in_all_its_cells_are_at_most_its_limit(world, limit):
    stub = Hooked(pause=0.03)
    pools = Pools(cpu=4, agents={"claude": limit})
    go(world, {"claude": Agent(stub)}, [FILES, STUB, NARROW], jobs=3, side_by_side=beside(pools=pools))
    assert most_at_once([(s, e) for _, s, e in stub.ran]) == limit


def test_each_agent_has_its_own_limit(world):
    claude_stub, codex_stub = Hooked(pause=0.03), Hooked(pause=0.03)
    pools = Pools(cpu=4, agents={"claude": 1, "codex": 2})
    go(world, {"claude": Agent(claude_stub), "codex": Agent(codex_stub)}, [FILES, STUB], jobs=3, side_by_side=beside(pools=pools))
    assert most_at_once([(s, e) for _, s, e in claude_stub.ran]) == 1 and most_at_once([(s, e) for _, s, e in codex_stub.ran]) == 2


def test_a_run_whose_pools_name_no_limit_for_an_agent_is_refused_before_anything_runs(world):
    stub = Stub()
    with pytest.raises(ValueError, match="no limit for codex"):
        go(world, {"claude": Agent(stub), "codex": Agent(stub)}, [FILES], side_by_side=beside("claude"))
    assert not stub.execs


# -- the order of a cell's cases ---------------------------------------------------------------------------------------------------

def by_longest(order, seconds):
    """`order` with the longest of `seconds` first, the first of equals first: picked one at a time."""
    left, out = list(order), []
    while left:
        best = max(left, key=lambda q: seconds[q])  # (max keeps the first of equal ones)
        left.remove(best)
        out.append(best)
    return out


def test_a_cell_that_cuts_no_index_starts_the_cases_expected_to_take_longest_first_and_a_cell_that_does_keeps_the_time_order(world):
    seconds = {q.qid: float(i * 7 % 13) for i, q in enumerate(IN_TIME)}
    in_time = [q.qid for q in IN_TIME]
    stub = Stub()
    side = beside(expected={("claude", "files"): Expected(seconds, 1.0), ("claude", "stub-all"): Expected(seconds, 1.0)})
    go(world, {"claude": Agent(stub)}, [FILES, STUB], jobs=1, side_by_side=side)
    assert stub.qids("files") == by_longest(in_time, seconds) != in_time
    assert stub.qids("stub-all") == in_time, "the system cuts its index to a horizon: AS_OF needs the time order, GPU or not"


def test_a_gpu_cell_keeps_the_time_order_whatever_it_took_before(world, monkeypatch):
    w = on_gpu(world, monkeypatch)
    seconds = {q.qid: float(len(IN_TIME) - i) for i, q in enumerate(IN_TIME)}
    stub = Stub()
    go(w, {"claude": Agent(stub)}, [STUB], jobs=1, side_by_side=beside(expected={("claude", "stub-all"): Expected(seconds, 1.0)}))
    assert stub.qids("stub-all") == [q.qid for q in IN_TIME]


def test_a_cell_with_no_earlier_exec_keeps_its_order_and_says_so(world, capsys):
    stub = Stub()
    go(world, {"claude": Agent(stub)}, [FILES, STUB], jobs=1, side_by_side=beside())
    assert stub.qids("files") == [q.qid for q in IN_TIME]
    out = capsys.readouterr().out
    assert "claude files: no earlier exec of this cell to order its cases longest first: they keep their order" in out
    assert "claude stub-all: no earlier exec" not in out, "a cell that keeps the time order has nothing to say"


# -- the agents are kept off the GPU ------------------------------------------------------------------------------------------------

def specs_of(w, configs, side=None, jobs=1):
    """The specs the agent was given for `configs`, on a run with no results."""
    if (w.run.folder / "results").exists():
        fs.remove_tree(w.run.folder / "results")
    stub = Stub()
    go(w, {"claude": Agent(stub)}, configs, jobs=jobs, side_by_side=side)
    return stub.execs


@pytest.mark.parametrize(("configs", "gpu_system", "side", "kept"), [
    ([FILES], False, False, False),  # the recorded cell: no rule, so the recorded goldens stand
    ([FILES], False, True, True),  # beside other cells
    ([STUB_FILES], True, False, True),  # mm+files: a shell and a GPU job in one cell, with or without the flag
    ([STUB_FILES], True, True, True),
    ([STUB_FILES], False, False, False),  # the same cell with a system that needs no GPU
    ([STUB_FILES], False, True, True),
    ([STUB], True, True, False),  # a system's tools alone: no shell to decode on
])
def test_an_agent_with_its_own_tools_is_kept_off_the_gpu_beside_other_cells_or_when_its_system_needs_it(
        world, monkeypatch, configs, gpu_system, side, kept):
    w = on_gpu(world, monkeypatch) if gpu_system else world
    specs = specs_of(w, configs, beside() if side else None)
    assert len(specs) == N
    for spec in specs:
        assert spec.off_gpu is kept
        assert (set(spec.deny) & GPU_WORDS) == (GPU_WORDS if kept else set())


def test_a_kept_off_agents_environment_and_command_line_hide_the_gpu_and_the_others_do_not(world, tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "git_root", lambda: tmp_path / "git")
    off = specs_of(world, [FILES], beside())[0]
    on = specs_of(world, [FILES], None)[0]
    assert off.off_gpu and not on.off_gpu
    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    planted = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows")}
    for kept, spec in ((True, off), (False, on)):
        with mock.patch.dict(os.environ, planted, clear=True):
            env = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation, spec.off_gpu)
            argv = codex.prepare(spec, "gpt-test", "high", models, ["<codex>"])
            assert os.environ.get("CUDA_VISIBLE_DEVICES") is None, "building the exec leaves this process's own environment alone"
        shell = [argv[i + 1] for i, a in enumerate(argv) if a == "-c" and argv[i + 1].startswith("shell_environment_policy.set.")]
        assert env.get("CUDA_VISIBLE_DEVICES") == ("-1" if kept else None)
        assert ("shell_environment_policy.set.CUDA_VISIBLE_DEVICES='-1'" in shell) is kept


def test_a_kept_off_agent_may_not_name_the_gpus_decoder_or_its_list_of_users_and_another_may(world):
    off = specs_of(world, [FILES], beside())[0]
    on = specs_of(world, [FILES], None)[0]
    ffmpeg = {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": "Bash",
                                                            "input": {"command": "ffmpeg -hwaccel cuda -i clip.mp4 frame.png"}}]}}
    smi = {"type": "item.completed", "item": {"type": "command_execution", "command": "nvidia-smi --query-gpu=name", "aggregated_output": ""}}
    fine = {"type": "item.completed", "item": {"type": "command_execution", "command": "ffmpeg -i clip.mp4 frame.png", "aggregated_output": ""}}
    assert claude.check_event(ffmpeg, off) == "Bash reached -hwaccel" and claude.check_event(ffmpeg, on) is None
    assert codex.check_event(smi, off) == "shell reached nvidia-smi" and codex.check_event(smi, on) is None
    assert codex.check_event(fine, off) is None, "software decoding stays allowed"
