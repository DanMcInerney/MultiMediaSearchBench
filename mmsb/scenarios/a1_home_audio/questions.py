"""The questions of A1, every answer from the labels (source.py), each asked the morning after the day:

| kind     | templates                                     | question                                                    | rule                     |
| `start`  | `vacuum_on`, `tv_on`, `visit_on`, `leave`     | At what time yesterday, between 21:30 and 23:00, was the    | `time`, 60 s (TV: 120 s) |
|          |                                               | vacuum cleaner switched on? (did I start watching TV; did a |                          |
|          |                                               | visitor arrive; did I leave the house)                      |                          |
| `stop`   | `vacuum_off`, `tv_off`, `visit_off`, `return` | the same for the end (switched off; did I stop watching TV; | `time`, 60 s (TV: 120 s) |
|          |                                               | did a visitor leave; did I get back home)                   |                          |
| `count`  | `count_vacuum`, `count_tv`, `count_visit`     | How many times did the vacuum cleaner run yesterday? (did I | `count`                  |
|          |                                               | start watching TV; did someone come to visit)               |                          |
| `absent` | `absent_` and a `start` or `stop` one         | that question over a window in which no run of its label    | `absent`                 |
|          |                                               | comes near                                                  |                          |

A `start` or `stop` question is about one run of a label (`vacuumcleaner`, `watching tv` and `visit` of the living room's labels;
`absence` of the home's, which is nobody home) and the second it starts or stops, asked as HH:MM:SS. Its window holds the template's
tolerance either side of the moment, and no other run of the label comes within MARGIN_OUT of the window (more than any tolerance), so
that exactly one moment in it fits. A run is not asked about when it began before the day or ends after it (the day's edges), or is
shorter than the template's `shortest` (the home's absences of a minute, when the resident steps out to the door), and a window is kept
clear of every run of the label, the short ones too. An absent question is the same template over a window in which no run of the label
comes within MARGIN_OUT, and no step out to the door (whom a visitor may have been), so it reads no differently. A window's bounds are on
a grid of half-hours (tiny: five minutes) and drawn with the seed; every moment that has a window is asked, and the absent templates are
drawn.

The tolerance is 60 s: the sound that shows an edge of the vacuum cleaner, the visit or the door comes within about 20 s of the label's.
The TV's is 120 s: its label marks the watching, not the set, whose sound (measured in the audio at the four moments asked) began 30 s
and 55 s before the label and ended 15 s and 45 s after it, so that "21:58" for a stop labelled 21:56:56 is right. The evidence of a
moment is the tolerance either side of it.

Classes that run into one another (cooking, eating, dishwashing, working: the audio has no edge between them) and `other` are not asked."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.a1_home_audio.source import Run

CODE = "A1"
SOURCE_NAME = "recorder"  # the dump's source
TOL_S = 60.0  # a label's edge is told to the millisecond, but the sound that shows it comes a few seconds either side
TV_TOL_S = 120.0  # the TV's: its label marks the watching, and the set's sound is up to a minute before or after it
MARGIN_OUT = 600.0  # no other run of the label comes this near a window (more than any tolerance)
STEP_OUT = 1800.0  # an absence of the home shorter than this is the resident stepping out (to the door), not leaving the house
GRID = {"tiny": 300, "standard": 1800}  # the grid of a window's bounds, in seconds
LENGTH = {"tiny": (900, 2400), "standard": (3600, 14400)}  # the lengths a window has, in seconds
ABSENT = {"tiny": 2, "standard": 5}  # how many absent questions
WHEN = "<the time, as HH:MM:SS>"
NUMBER = "<a number>"
DAY = 86400


@dataclass(frozen=True)
class Template:
    key: str
    labels: int  # 0: the living room's labels, 1: the whole home's
    cls: str
    edge: str  # start or stop
    phrase: str  # what the question asks, after "At what time yesterday, between 21:30 and 23:00, "
    shortest: float  # a run shorter than this (seconds) is no moment to ask about
    tol: float = TOL_S  # an answer this many seconds from the moment is right


TEMPLATES = (
    Template("vacuum_on", 0, "vacuumcleaner", "start", "was the vacuum cleaner switched on", 120),
    Template("vacuum_off", 0, "vacuumcleaner", "stop", "was the vacuum cleaner switched off", 120),
    Template("tv_on", 0, "watching tv", "start", "did I start watching TV", 600, TV_TOL_S),
    Template("tv_off", 0, "watching tv", "stop", "did I stop watching TV", 600, TV_TOL_S),
    Template("visit_on", 0, "visit", "start", "did a visitor arrive", 120),
    Template("visit_off", 0, "visit", "stop", "did a visitor leave", 120),
    Template("leave", 1, "absence", "start", "did I leave the house", STEP_OUT),
    Template("return", 1, "absence", "stop", "did I get back home", STEP_OUT),
)
COUNTS = (("vacuum", "vacuumcleaner", "did the vacuum cleaner run"), ("tv", "watching tv", "did I start watching TV"),
          ("visit", "visit", "did someone come to visit"))  # (key, label, what the question asks)


def clock(seconds: float) -> str:
    return f"{int(seconds) // 3600:02d}:{int(seconds) % 3600 // 60:02d}"


def clock_s(seconds: float) -> str:
    """`22:04:43`, to the nearest second."""
    n = round(seconds)
    return f"{n // 3600:02d}:{n % 3600 // 60:02d}:{n % 60:02d}"


def questions_of(runs: Sequence[Sequence[Run]], day: date, first: int, seconds: int, path: str, scale: str, seed: int,
                 t_of: Callable[[datetime], float]) -> tuple[Question, ...]:
    """The questions of the recording `path` of `seconds` from `first` (seconds after midnight of `day`), over the label `runs` (the
    living room's and the home's) at `scale`, drawn with `seed`; `t_of(wall)` is a wall time's epoch seconds."""
    midnight = datetime.combine(day, time())
    last = first + seconds
    now = t_of(midnight + timedelta(days=1, hours=9))
    since = t_of(midnight + timedelta(seconds=first))
    step, length = GRID[scale], LENGTH[scale]
    grid = [s for s in range(first, last + 1, step) if s < DAY]  # (a window is not asked to end at midnight)
    placed = [[(r.cls, (r.start - midnight).total_seconds(), (r.stop - midnight).total_seconds()) for r in of] for of in runs]
    out: list[Question] = []

    def ask(qid: str, text: str, kind: str, template: str, shape: str, grade, spans: Sequence[tuple[float, float]], value: str | None):
        evidence = [Span(path, max(0.0, a - first), min(float(seconds), b - first)) for a, b in spans]
        return realdump.question(code=CODE, qid=qid, now=now, text=text, kind=kind, template=template, modality="audio", shape=shape,
                                 grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since, until=since + seconds, value=value)

    def windows(clear: Callable[[float, float], bool]) -> list[tuple[int, int]]:
        return [(a, b) for a in grid for b in grid if length[0] <= b - a <= length[1] and clear(a, b)]

    def when(t: Template, a: int, b: int) -> str:
        return f"At what time yesterday, between {clock(a)} and {clock(b)}, {t.phrase}?"

    # start and stop: one question for each run of the label that has a window, in the day
    for t in TEMPLATES:
        every = [(c, a, b) for c, a, b in placed[t.labels] if c == t.cls]
        for run in every:
            _, begin, end = run
            edge = begin if t.edge == "start" else end
            if end - begin < t.shortest or begin < 0 or end > DAY or not first + t.tol <= edge <= last - t.tol:
                continue
            others = [r for r in every if r is not run]
            fits = windows(lambda a, b: a + t.tol <= edge <= b - t.tol
                           and all(o[2] <= a - MARGIN_OUT or o[1] >= b + MARGIN_OUT for o in others))
            if not fits:
                continue
            a, b = random.Random(f"a1:{seed}:{t.key}:{edge}").choice(fits)
            qid = f"{t.key}-{clock(edge).replace(':', '')}"
            out.append(ask(qid, when(t, a, b), t.edge, t.key, WHEN,
                           realdump.grade("time", time=clock_s(edge), tol_s=t.tol), [(edge - t.tol, edge + t.tol)], clock_s(edge)))
    # count: how many times the label ran in the recording (every run starts and ends in it)
    for key, cls, what in COUNTS:
        found = [(a, b) for c, a, b in placed[0] if c == cls and first <= a and b <= last]
        if not found:
            continue
        text = f"How many times {what} yesterday?" if scale == "standard" else f"How many times {what} yesterday, between {clock(first)} and {clock(last)}?"
        out.append(ask(f"count-{key}", text, "count", f"count_{key}", NUMBER, realdump.grade("count", truth=str(len(found))), found,
                       str(len(found))))
    # absent: a template over a window in which no run of its label comes near
    rng = random.Random(f"a1:{seed}:absent")
    steps = [r for r in placed[1] if r[0] == "absence" and r[2] - r[1] < STEP_OUT]  # (a window is kept off the door, too)
    options = {}
    for t in TEMPLATES:
        every = [r for r in placed[t.labels] if r[0] == t.cls]
        if fits := windows(lambda a, b: all(o[2] <= a - MARGIN_OUT or o[1] >= b + MARGIN_OUT for o in (*every, *steps))):
            options[t.key] = (t, fits)
    for key in rng.sample(sorted(options), ABSENT[scale]):
        t, fits = options[key]
        a, b = rng.choice(fits)
        out.append(ask(f"absent-{key}-{clock(a).replace(':', '')}", when(t, a, b), "absent", f"absent_{key}", WHEN,
                       realdump.grade("absent"), [], None))
    return realdump.split(out, seed)
