"""Which case starts next, and how the cells of a run run beside each other (DESIGN 8.3; `next_question` of bench.py of this repository at
bb498a6 is `next_case`, now for every configuration).

Without `--side-by-side` the cells of a run run one after another and nothing here but `next_case` is called. With it (off by default,
DESIGN 13.2), each cell gets a thread (`run_beside`), and what the threads share is `Pools`, made real by `Slots`:
- the GPU: one cell that needs it at a time holds its token for its whole length (its session's `open` and `close` included), the cells
  that need it taking it in the order they would have run one after another;
- the working folders: the laptop runs at most `JOBS_MAX` execs at once, and the working slots and the deny words are for that many, so
  the execs of all the cells together take a slot each from one pool (`Pools.cpu`);
- each agent's sessions: at most `Pools.agents[agent]` of one agent run at once, in any cells.
A shared slot goes to the cell that needs the GPU first, then to the earlier cell in the sequential order. `simulate` is the same policy on
paper, for the estimate a dry run prints, and `agent_off_gpu` is the rule that keeps the agents' own tools off the GPU while it is in use.
"""

from __future__ import annotations

import bisect
import itertools
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

from mmsb.core.contract import Case, Description, Session
from mmsb.core.record import Halt
from mmsb.core.runfolder import JOBS_MAX


def next_case(pending: Sequence[Case], busy: Sequence[Case], session: Session) -> Case | None:
    """The first of `pending` to start, or None when each of them has to wait. Two cases of one group never run at once, as two
    questions about one needle would show one the other's answer; the rest is the session's (`Session.admits`)."""
    groups = {c.group for c in busy}
    return next((c for c in pending if c.group not in groups and session.admits(c, busy)), None)


def agent_off_gpu(own_tools: bool, description: Description | None, side_by_side: bool) -> bool:
    """Whether the agent of a cell is kept off the GPU (`CUDA_VISIBLE_DEVICES=-1`, and `-hwaccel` and `nvidia-smi` in its deny words): an
    agent with its own tools that runs beside other cells, or whose own system needs the GPU for its questions (`mm+files` has a shell and
    a GPU job in one cell). An agent decoding on the GPU would be a second GPU user."""
    return own_tools and (side_by_side or (description is not None and description.needs_gpu))


# -- the order of a cell's cases --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Expected:
    """What the cases of a cell took in earlier runs, in seconds: the median of each case that has run, and the cell's typical exec for the
    others."""

    by_qid: Mapping[str, float]
    typical: float

    def of(self, qid: str) -> float:
        return self.by_qid.get(qid, self.typical)


def keeps_time_order(description: Description | None) -> bool:
    """Whether a cell's cases must start in ascending `until` (AS_OF): those of a system that cuts its whole index to a case's horizon,
    whether or not it needs the GPU."""
    return description is not None and description.horizon == "cut"


def start_order(pending: Sequence[Case], description: Description | None, expected: Expected | None) -> list[Case]:
    """`pending` (in ascending `until`) in the order a side-by-side run starts them: the cases expected to take longest first, so that the
    short ones fill the end of a cell and no long one starts last. Cases expected equally keep their order, and so do all of a cell with no
    history (`expected` None) and of one that must keep time order (`keeps_time_order`)."""
    if expected is None or keeps_time_order(description):
        return list(pending)
    return sorted(pending, key=lambda c: -expected.of(c.qid))


# -- the pools --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pools:
    """The most of each that a run uses at once. `cpu`: the laptop's rule of at most `JOBS_MAX` processes at once, which in a run are the
    execs, each in a working folder of its own. `agents`: the sessions of each agent (its quota and rate limits). `gpu`: the cells that
    hold the GPU."""

    cpu: int
    agents: Mapping[str, int]
    gpu: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.cpu <= JOBS_MAX or not self.agents or not all(1 <= n <= JOBS_MAX for n in self.agents.values()) or self.gpu < 1:
            raise ValueError(f"pools are 1 to {JOBS_MAX} execs and sessions per agent and 1 or more GPU cells, not {self}")

    @classmethod
    def of(cls, agents: Iterable[str]) -> Pools:
        """The pools of a run of `agents`: every limit at its most."""
        return cls(cpu=JOBS_MAX, agents=dict.fromkeys(agents, JOBS_MAX))


class Gate:
    """Tokens handed out to the waiters, the lowest priority first and, within one priority, the first to ask: a permit, or the number
    of a working slot (the lowest free). `reserve` asks; entering the ticket it gives waits for the turn. A ticket is entered at once
    (`with gate.reserve(p) as token:`) or, to fix the order of the waiters before any of them runs, long after it was reserved."""

    def __init__(self, tokens: Iterable[int]) -> None:
        self._free = sorted(tokens)
        self._waiting: list[tuple] = []  # (priority, number) of the tickets not yet entered, in turn order
        self._numbers = itertools.count()
        self._changed = threading.Condition()

    def reserve(self, priority: tuple) -> Ticket:
        with self._changed:
            key = (priority, next(self._numbers))
            bisect.insort(self._waiting, key)
        return Ticket(self, key)

    def enter(self, key: tuple) -> int:
        with self._changed:
            self._changed.wait_for(lambda: self._waiting[0] == key and self._free)
            self._waiting.pop(0)
            token = self._free.pop(0)
            self._changed.notify_all()  # (another waiter may be next, with tokens left)
        return token

    def leave(self, token: int) -> None:
        with self._changed:
            bisect.insort(self._free, token)
            self._changed.notify_all()

    def drop(self, key: tuple) -> None:
        """A ticket that will not be entered gives up its turn."""
        with self._changed:
            self._waiting.remove(key)
            self._changed.notify_all()


class Ticket:
    """A turn at a `Gate`: entering it waits for the turn and gives the token, leaving it gives the token back."""

    def __init__(self, gate: Gate, key: tuple) -> None:
        self.gate, self.key, self.token = gate, key, -1

    def __enter__(self) -> int:
        self.token = self.gate.enter(self.key)
        return self.token

    def __exit__(self, *exc: object) -> None:
        self.gate.leave(self.token)


class Slots:
    """The pools of a side-by-side run, as the semaphores its threads wait on."""

    def __init__(self, pools: Pools) -> None:
        self.pools = pools
        self.gpu = Gate(range(pools.gpu))
        self.execs = Gate(range(pools.cpu))  # the working slots: a token is a slot's number
        self.agents = {agent: Gate(range(n)) for agent, n in pools.agents.items()}


@dataclass(frozen=True)
class SideBySide:
    """What `--side-by-side` asks for: the `pools` the cells share, and what the cases of each cell, by (agent, `Config.dir`), took in earlier
    runs. A cell with no entry has no history, and keeps the order of its cases."""

    pools: Pools
    expected: Mapping[tuple[str, str], Expected]


@dataclass(frozen=True)
class Beside:
    """What one cell of a side-by-side run is given besides its own work: the shared `slots`, the `priority` it asks for them with (0 for a
    cell that needs the GPU, else 1, then its place in the sequential order), `stop` (set when any cell ends on an error: nothing more
    starts in any cell), and what its cases took before (`Expected`; None: nothing is known, and they keep their order)."""

    slots: Slots
    priority: tuple[int, int]
    stop: Halt
    expected: Expected | None


@dataclass(frozen=True)
class Task:
    """One cell for `run_beside`: its name for messages, whether it needs the GPU, what its cases took before, and what it does."""

    name: str
    needs_gpu: bool
    expected: Expected | None
    run: Callable[[Beside], None]


def run_beside(tasks: Sequence[Task], pools: Pools) -> None:
    """Every task in a thread of its own, with a cell that needs the GPU waiting for its token (the tasks that need it take it in the
    order given, one at a time, each holding it until its `run` has returned, whatever it did). When a task ends on an error, no task
    starts anything more (`Beside.stop`), the others end what they were doing, and the first error is raised once all have ended; later
    ones are told beside it."""
    slots, stop = Slots(pools), Halt()
    tickets = {i: slots.gpu.reserve((n,)) for n, i in enumerate(i for i, task in enumerate(tasks) if task.needs_gpu)}  # (the order is fixed here)
    failures: list[tuple[str, BaseException]] = []
    lock = threading.Lock()

    def body(i: int, task: Task) -> None:
        with tickets[i] if task.needs_gpu else nullcontext():
            try:
                task.run(Beside(slots, (0 if task.needs_gpu else 1, i), stop, task.expected))
            except BaseException as error:  # noqa: BLE001 - raised again below, from the thread that waits
                with lock:
                    failures.append((task.name, error))
                stop.set(f"{task.name} ended on an error")  # (before the GPU is given up: the next cell to take it must find the stop)

    threads = [threading.Thread(target=body, args=(i, task), name=task.name) for i, task in enumerate(tasks)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except BaseException:  # an error here, or Ctrl+C: nothing more starts, the running execs end
        stop.set("stopped by an error or an interrupt")
        for i, thread in enumerate(threads):
            if thread.ident is None and tasks[i].needs_gpu:
                tickets[i].gate.drop(tickets[i].key)
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        raise
    if failures:
        for name, error in failures[1:]:
            print(f"{name}: ended on an error as well: {error!r}", flush=True)
        raise failures[0][1]


# -- the same policy on paper -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    """A cell as `simulate` sees it: the cases it has left, in the order it starts them, `jobs` of them at once at most."""

    name: str
    agent: str
    needs_gpu: bool
    jobs: int
    cases: tuple[str, ...]


@dataclass(frozen=True)
class Span:
    cell: str
    qid: str | None  # None: the GPU token held by the cell, from when it took it to when its last case ended
    start: float
    end: float


@dataclass(frozen=True)
class Timeline:
    spans: tuple[Span, ...]  # each case, in the order they started
    holds: tuple[Span, ...]  # each GPU cell's hold of the GPU

    @property
    def makespan(self) -> float:
        return max((s.end for s in self.spans), default=0.0)


def simulate(cells: Sequence[Line], durations: Mapping[tuple[str, str], float], pools: Pools) -> Timeline:
    """When each case of `cells` starts and ends if the run takes `durations` ((cell name, qid) -> seconds) and runs under `pools` by the
    policy of `Slots`: the cells that need the GPU take it one at a time in the order given, and at each moment the free slots go to the cell
    that needs the GPU first, then the earlier one, while it has a case and fewer than `jobs` running. Event by event; nothing sleeps."""
    order = sorted(range(len(cells)), key=lambda i: (not cells[i].needs_gpu, i))
    waiting_gpu = [i for i, cell in enumerate(cells) if cell.needs_gpu and cell.cases]
    holding: dict[int, float] = {}  # GPU cell -> when it took the GPU
    next_at = [0] * len(cells)
    running = [0] * len(cells)
    used = {agent: 0 for agent in pools.agents}
    ends: list[tuple[float, int]] = []  # (end, cell)
    spans: list[Span] = []
    holds: list[Span] = []
    now = 0.0
    while True:
        while waiting_gpu and len(holding) < pools.gpu:
            holding[waiting_gpu.pop(0)] = now
        while (i := next((i for i in order if next_at[i] < len(cells[i].cases) and running[i] < cells[i].jobs
                          and (not cells[i].needs_gpu or i in holding) and used[cells[i].agent] < pools.agents[cells[i].agent]
                          and sum(running) < pools.cpu), None)) is not None:
            cell = cells[i]
            qid = cell.cases[next_at[i]]
            next_at[i] += 1
            running[i] += 1
            used[cell.agent] += 1
            end = now + durations[cell.name, qid]
            spans.append(Span(cell.name, qid, now, end))
            bisect.insort(ends, (end, i))
        if not ends:
            return Timeline(tuple(spans), tuple(holds))
        now = ends[0][0]
        while ends and ends[0][0] <= now:
            _, i = ends.pop(0)
            running[i] -= 1
            used[cells[i].agent] -= 1
            if i in holding and running[i] == 0 and next_at[i] == len(cells[i].cases):
                holds.append(Span(cells[i].name, None, holding.pop(i), now))
