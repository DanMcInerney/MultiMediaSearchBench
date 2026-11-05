"""The working-time clock (DESIGN 8.4, plan 6.5): what an exec is held to. When several cases share one resource (the GPU), a case's
wait for it is an artefact of the benchmark, so an exec is charged its own turns plus its own service time, never its time in the
system's queue; a wall-clock cap, a multiple of the limit, ends one that waits for ever. This is Inspect AI's `working_limit` beside a
`time_limit`.

    working = wall - queue        over when working >= limit, or when wall >= the wall cap, whatever the queue

The queue is whatever the exec's system reports (`Session.queue_s`, read live). A system that cannot say (`None`) leaves the clock to run on
the wall: `clock: wall` in the result, and the limit then applies to wall time, as it did before the working clock. That is exact only
when nothing else shares the system, so a session that cannot report it is refused more than one exec at once on the working clock
(`check_queue`). The agents start a clock when they start their process (`seconds` of a result is the same stopwatch), look at it while
they wait for the process (`agents.common.wait_for`) and put its `figures` in their result."""

from __future__ import annotations

import time
from collections.abc import Callable

from mmmb.core.contract import Case, Session

WORKING, WALL = "working", "wall"
CLOCKS = (WORKING, WALL)  # `--clock`: charge working time where the system reports its queue, or wall time throughout
DEFAULT_CLOCK = WORKING
WALL_FACTOR = 3.0  # the wall cap is this many times the working limit
STEP_S = 1.0  # a process is looked at again at least this often: a queue that grows moves the end of the limit
FIELDS = ("clock", "working_s", "queue_s")  # the figures of a clock that a result holds beside `seconds`, its wall time
HOUR_BUDGET_S, DAY_BUDGET_S = 300.0, 900.0
DAY_HOURS = 24.0


def unreported() -> float | None:
    """The queue of a system that is not asked (`--clock wall`, a dry run): nothing is known of it."""
    return None


class WorkingClock:
    """The stopwatch of one exec. It starts when it is made. `queue_s` gives the seconds so far that the exec has waited in its system's
    queue, or None when the system cannot say (the clock then runs on the wall). `now` is `time.monotonic`; a test injects it."""

    def __init__(self, limit_s: float, wall_cap_s: float, queue_s: Callable[[], float | None], now: Callable[[], float] = time.monotonic):
        if not 0 < limit_s <= wall_cap_s:
            raise ValueError(f"a limit of {limit_s:g} s under a wall cap of {wall_cap_s:g} s: the cap is no less than the limit, and both are positive")
        self.limit_s, self.wall_cap_s, self.queue_s, self.now = limit_s, wall_cap_s, queue_s, now
        self.began = now()

    @classmethod
    def of(cls, limit_s: float, queue_s: Callable[[], float | None], now: Callable[[], float] = time.monotonic) -> WorkingClock:
        """A clock for `limit_s` working seconds with the usual wall cap (`WALL_FACTOR` times the limit)."""
        return cls(limit_s, limit_s * WALL_FACTOR, queue_s, now)

    def look(self) -> tuple[float, float | None, float]:
        """(wall seconds so far, the queue the system reports, working seconds). Without a queue figure, working is wall; a queue over
        the wall (a system that counts from before the exec) leaves 0, and the figure itself stays in the result for anyone to see."""
        wall, queue = self.now() - self.began, self.queue_s()
        return wall, queue, wall if queue is None else max(0.0, wall - queue)

    def left(self) -> float:
        """The most seconds that can pass before the clock is over if no more of the exec's time is queued from now on."""
        wall, _, working = self.look()
        return max(0.0, min(self.limit_s - working, self.wall_cap_s - wall))

    def expired(self) -> str | None:
        """The `problem` of an exec that is out of time, else None. The limit's text is the recorded `timeout: no answer in 900 s` unless
        the exec waited in a queue, which it then says; the wall cap, which only a long queue reaches first, is told apart from it."""
        wall, queue, working = self.look()
        waited = f"{queue:g} s" if queue else None
        if working >= self.limit_s:
            return f"timeout: no answer in {self.limit_s:g} s" + (f" of working time ({wall:.0f} s on the wall clock, {waited} queued)" if waited else "")
        if wall >= self.wall_cap_s:
            return f"timeout: no answer in {self.wall_cap_s:g} s on the wall clock" + (f" ({waited} of it queued)" if waited else "")
        return None

    def figures(self) -> dict:
        """What a result holds of the clock, in tenths of a second: `clock` (`working` or `wall`), `working_s` (what the limit was
        measured on), `queue_s` (None when not reported) and `wall_s` (the exec's `seconds`)."""
        wall, queue, working = self.look()
        return {"clock": WALL if queue is None else WORKING, "working_s": round(working, 1),
                "queue_s": None if queue is None else round(queue, 1), "wall_s": round(wall, 1)}


def check_queue(label: str, session: Session, case: Case, at_once: int, clock: str) -> None:
    """Raises when `at_once` execs of `session` would share it and `clock` is `working`, yet the session cannot say how long a case waits
    in its queue (`Session.queue_s` is None): a case's wait for the shared resource would be charged to it, and the result would not be
    the working time it says. One exec at a time has no queue, so the wall is the working time there. `case` is any case of the cell."""
    if clock not in CLOCKS:
        raise ValueError(f"the clock is one of {', '.join(CLOCKS)}, not {clock!r}")
    if clock == WORKING and at_once > 1 and session.queue_s(case) is None:
        raise ValueError(f"{label}: {at_once} execs at once on the working clock, but its system reports no queue time (Session.queue_s is "
                         "None), so each exec would be charged its wait for the others. Run it at --jobs 1, or pass --clock wall "
                         "(each result then says clock: wall)")


def budget_s(media_hours: float) -> float:
    """The working seconds an exec is given for a haystack of `media_hours` of recordings (DESIGN 8.4): 300 for an hour or less, 900 for a
    day or more, in proportion to the hours between. Its wall cap is `WALL_FACTOR` times it (`WorkingClock.of`)."""
    if media_hours < 0:
        raise ValueError(f"a haystack of {media_hours:g} hours")
    if media_hours <= 1:
        return HOUR_BUDGET_S
    if media_hours >= DAY_HOURS:
        return DAY_BUDGET_S
    return HOUR_BUDGET_S + (DAY_BUDGET_S - HOUR_BUDGET_S) * (media_hours - 1) / (DAY_HOURS - 1)


def figures_of(out: dict) -> dict:
    """The clock's fields (`FIELDS`) of an agent's result: an agent that leaves one out is an error (a KeyError)."""
    return {name: out[name] for name in FIELDS}
