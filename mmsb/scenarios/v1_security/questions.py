"""The questions of V1, every answer from Kitware's labels (source.py), each asked at noon on the day of the recordings:

| template       | example                                                            | rule                  |
| `first`        | When did someone first open a car trunk on cam1?                   | `time` and date, 30 s |
| `last`         | When did a car last make a U-turn on cam3?                         | `time` and date, 30 s |
| `count`        | How many times did someone close a car door on cam3 between 10:10  | `count`               |
|                | and 10:20? (2 to 5 times; at the tiny scale 1 to 5)                |                       |
| `absent_first` | `first` or `last` of an activity the labels show on no camera in   | `absent`              |
| `absent_last`  | these hours (a bicycle ridden, something stolen or bought)         | `absent`              |
| `absent_count` | `count` of an activity of the camera, in a window with none of it  | `absent`              |

`first` and `last` ask of an activity whose label the annotators are sure of (status `good`) and that lasts half a minute at most: the
time is the middle of the span, any time within 30 s of it is right, and the evidence is every span within those 30 s. A question is not
asked when a label that decides it is `unaudited` or `not_good`, or is within a second of the start or end of its clip (there a label may
go on in the next clip's, and one activity be two). A `count` window's edges are at least 30 s from every activity of that kind on the camera
(none is cut or nearly), and the activities in it are at least 10 s apart. The window of an `absent_count` is at least a minute clear of the
camera's activities of that kind and of its like (a trunk's opening and closing), so a label missed on one is not the answer; and the kind
does occur on the camera outside the window. A window lies inside one recording of its camera, not across the recorder's stop. Which ones
are asked is drawn with the seed, so that the activities and the cameras vary.

A car's manoeuvres (reverse, U-turn, left and right turn) are taken for one another, the more so on cam3, whose cars are small and far: a
`first` or `last` of one is asked only when none of the others is within 30 s of its moment (TOL_S), and a `count` only when none is in the
window or within 30 s (EDGE_S) of it. (A trunk's opening is not taken for its closing: only the manoeuvres have rivals.)

No question asks of the activities that are easily taken for another (`person_unloads_vehicle` for `person_loads_vehicle`,
`vehicle_drops_off_person` for a person getting out), or of a long or uncertain one (`person_talks_to_person`, `vehicle_moving`)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, time, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.v1_security.source import Event

CODE = "V1"
TOL_S = 30.0  # a time is told to the second, and the activities last seconds: half a minute either side of the middle is the same moment
MAX_SPAN_S = 30.0  # `first` and `last` ask of activities that last no longer
EDGE_S = 30.0  # a counted window's edges are this far from the camera's activities of the kind
ABSENT_EDGE_S = 60.0  # and an absent one's, which are as far from the activities of its like
MIN_GAP_S = 10.0  # counted activities are this far apart
SLACK_S = 5.0  # a window may begin or end this long outside a recording (cam1's recorder starts two seconds past the minute)
COUNT_MIN, COUNT_MAX = {"tiny": 1, "standard": 2}, 5  # the times a `count` asks of, at each scale (a 1 is easily guessed)
STEP_MIN = 5  # a window starts and ends on a multiple of this many minutes
WINDOW_MIN = 10  # and is at least this long
WHEN = "<the date and time, as YYYY-MM-DD HH:MM:SS>"
NUMBER = "<a number>"
QUOTAS = {"tiny": {"first": 3, "last": 1, "count": 2, "absent_first": 1, "absent_last": 1, "absent_count": 0},
          "standard": {"first": 6, "last": 5, "count": 7, "absent_first": 2, "absent_last": 1, "absent_count": 3}}
# the labels' activities a question asks of, as "When did <who> first <what> on cam1?"
ACTIVITIES = {"person_opens_trunk": ("someone", "open a car trunk"), "person_closes_trunk": ("someone", "close a car trunk"),
              "person_opens_vehicle_door": ("someone", "open a car door"), "person_closes_vehicle_door": ("someone", "close a car door"),
              "person_embraces_person": ("two people", "hug"), "vehicle_reverses": ("a car", "reverse"),
              "vehicle_makes_u_turn": ("a car", "make a U-turn"), "vehicle_turns_left": ("a car", "turn left"),
              "vehicle_turns_right": ("a car", "turn right")}
COUNTED = ("person_opens_trunk", "person_closes_trunk", "person_opens_vehicle_door", "person_closes_vehicle_door", "vehicle_reverses",
           "vehicle_makes_u_turn")  # the ones a `count` asks of
# activities of the dataset that no camera shows in these hours: crisp, and none is easily taken for one that is shown
NEVER = {"person_rides_bicycle": ("someone", "ride a bicycle"), "person_steals_object": ("someone", "steal something"),
         "person_purchases": ("someone", "buy something")}
# a car's manoeuvres, which are taken for one another (a reverse ends in a turn, a U-turn is two turns): none is asked about where another
# is near
MANOEUVRES = ("vehicle_reverses", "vehicle_makes_u_turn", "vehicle_turns_left", "vehicle_turns_right")
# activities that go together, or are taken for one another: an absent window is clear of one and all
LIKE = (("person_opens_trunk", "person_closes_trunk"), ("person_opens_vehicle_door", "person_closes_vehicle_door"), MANOEUVRES)


def like(kind: str) -> tuple[str, ...]:
    """`kind` and the activities that go with it or are taken for it."""
    return next((group for group in LIKE if kind in group), (kind,))


def rival_kinds(kind: str) -> tuple[str, ...]:
    """The activities that are taken for `kind`: the other manoeuvres, for a manoeuvre; none for the rest (a trunk's opening is not taken
    for its closing, though they go together)."""
    return tuple(k for k in MANOEUVRES if k != kind) if kind in MANOEUVRES else ()


def sure(e: Event) -> bool:
    return e.status == "good" and not e.edge


def touches(es: Sequence[Event], a: float, b: float, edge: float) -> bool:
    """Whether any of `es` is in the window `a` to `b` or within `edge` seconds of it."""
    return any(e.t1 >= a - edge and e.t0 <= b + edge for e in es)


def needle(es: Sequence[Event], end: int, rivals: Sequence[Event] = ()) -> tuple[Event, float, list[Event]] | None:
    """The first (`end` 0) or last (-1) of the activities `es` (in time order), the time asked for (the middle of its span, to the second)
    and every activity of `es` the rule accepts that time for; None when one of these is not sure (a label `unaudited` or `not_good`, or
    at the edge of its clip) or lasts too long, or when any of `rivals` (activities taken for these) comes within TOL_S of the time."""
    e = es[end]
    middle = float(round((e.t0 + e.t1) / 2))
    near = [x for x in es if x.t1 >= middle - TOL_S and x.t0 <= middle + TOL_S]
    if not all(sure(x) and x.o1 - x.o0 <= MAX_SPAN_S for x in near) or touches(rivals, middle, middle, TOL_S):
        return None
    return e, middle, near


def inside(es: Sequence[Event], a: float, b: float, edge: float) -> list[Event] | None:
    """The activities of `es` in the window `a` to `b`, when each is at least `edge` seconds from its edges and every other is at
    least that far outside it; None when one is nearer an edge than that."""
    found = []
    for e in es:
        if a + edge <= e.t0 and e.t1 <= b - edge:
            found.append(e)
        elif e.t1 >= a - edge and e.t0 <= b + edge:
            return None
    return found


def apart(found: Sequence[Event]) -> bool:
    """Whether each of `found` (in time order) ends MIN_GAP_S before any later one starts, and every one is sure."""
    reach = float("-inf")
    for e in found:
        if not sure(e) or e.t0 - reach < MIN_GAP_S:
            return False
        reach = max(reach, e.t1)
    return True


def pick(candidates: Sequence[tuple], n: int, rng: random.Random, used: Counter) -> list[tuple]:
    """`n` of `candidates` (`(camera, kind, what)`; one of a camera and kind at most), drawn so that the activities and then the cameras
    vary: each time the one whose kind, then whose camera, has been drawn least here, then whose camera and kind and whose camera
    `used` (which it updates) has counted least in the questions so far, then the first in a shuffled order."""
    pool = list(candidates)
    rng.shuffle(pool)
    here: Counter = Counter()
    out: list = []
    while pool and len(out) < n:
        best = min(pool, key=lambda c: (here["kind", c[1]], here["camera", c[0]], used["pair", c[0], c[1]], used["camera", c[0]]))
        pool = [c for c in pool if c[:2] != best[:2]]
        out.append(best)
        for counts in (here, used):
            counts["kind", best[1]] += 1
            counts["camera", best[0]] += 1
        used["pair", best[0], best[1]] += 1
    return out


def count_text(camera: str, kind: str, a: datetime, b: datetime) -> str:
    who, what = ACTIVITIES[kind]
    return f"How many times did {who} {what} on {camera} between {a:%H:%M} and {b:%H:%M}?"


def questions_of(events: Sequence[Event], files: Sequence[realdump.File], span: tuple[datetime, datetime], scale: str, seed: int,
                 t_of) -> tuple[Question, ...]:
    """The questions of the recordings `files` with the activities `events` placed in them, which run from the wall time `span[0]` to
    `span[1]`, at `scale`, drawn with `seed`; `t_of(wall)` is a wall time's epoch seconds."""
    rng, used, quota = random.Random(f"v1:{seed}"), Counter(), QUOTAS[scale]
    first, last = span
    now = t_of(datetime.combine(last.date(), time(12)))
    cameras = sorted({f.source for f in files})
    kinds = sorted({e.kind for e in events})
    groups = {(c, k): sorted((e for e in events if (e.camera, e.kind) == (c, k)), key=lambda e: (e.t0, e.t1)) for c in cameras for k in kinds}
    covered = {c: (min(f.t for f in files if f.source == c), max(f.t + f.length for f in files if f.source == c)) for c in cameras}
    minutes = range(0, int((last - first).total_seconds() // 60) + 1, STEP_MIN)
    grid = [(first + timedelta(minutes=m), first + timedelta(minutes=n)) for m in minutes for n in minutes if n - m >= WINDOW_MIN]
    recorded = {c: [(f.t - SLACK_S, f.t + f.length + SLACK_S) for f in files if f.source == c] for c in cameras}
    # a window lies inside one recording of its camera, not across the recorder's stop
    windows = {c: [(a, b) for a, b in grid if any(lo <= t_of(a) and t_of(b) <= hi for lo, hi in recorded[c])] for c in cameras}

    def of(camera: str, kinds: Sequence[str]) -> list[Event]:
        """The camera's activities of `kinds`."""
        return [e for k in kinds for e in groups.get((camera, k), ())]

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[Event], value: str | None, camera: str,
            since: float, until: float) -> Question:
        return realdump.question(code=CODE, qid=qid, now=now, text=text, kind="absent" if template.startswith("absent") else template,
                                 template=template, modality="video", shape=shape, grade=grade,
                                 evidence=[Span(e.file, round(e.o0, 3), round(e.o1, 3)) for e in found], sources=(camera,), since=since,
                                 until=until, value=value)

    def clear(camera: str, kind: str, a: datetime, b: datetime) -> bool:
        """Whether the camera has no activity of `kind`, or its like, within ABSENT_EDGE_S of the window."""
        return not touches(of(camera, like(kind)), t_of(a), t_of(b), ABSENT_EDGE_S)

    out: list[Question] = []
    # first and last: when an activity was first (last) seen on a camera
    for template, end in (("first", 0), ("last", -1)):
        sought = [(c, k, found) for (c, k), es in groups.items()
                  if k in ACTIVITIES and es and (found := needle(es, end, of(c, rival_kinds(k))))]
        for camera, kind, (_, middle, near) in pick(sought, quota[template], rng, used):
            when, (who, what) = datetime.fromtimestamp(middle), ACTIVITIES[kind]
            grade = realdump.grade("time", time=f"{when:%H:%M:%S}", tol_s=TOL_S, date=when.date().isoformat())
            out.append(ask(f"{template}-{camera}-{kind}", f"When did {who} {template} {what} on {camera}?", template, WHEN, grade, near,
                           f"{when:%Y-%m-%d %H:%M:%S}", camera, *covered[camera]))
    # count: how many times an activity happened on a camera in a window
    sought = []
    for (camera, kind), es in groups.items():
        for a, b in windows[camera] if kind in COUNTED else ():
            found = inside(es, t_of(a), t_of(b), EDGE_S)
            text = count_text(camera, kind, a, b)
            if (COUNT_MIN[scale] <= len(found or ()) <= COUNT_MAX and apart(found)
                    and not touches(of(camera, rival_kinds(kind)), t_of(a), t_of(b), EDGE_S)):
                sought.append((camera, kind, (a, b, found, text)))
    for camera, kind, (a, b, found, text) in pick(sought, quota["count"], rng, used):
        n = str(len(found))
        out.append(ask(f"count-{camera}-{kind}-{a:%H%M}-{b:%H%M}", text, "count", NUMBER, realdump.grade("count", truth=n), found, n, camera,
                       t_of(a), t_of(b)))
    # absent: an activity of the dataset no camera shows; an activity of a camera in a window with none of its kind
    shown = {e.kind for e in events}
    for template in ("first", "last"):
        sought = [(c, k, None) for c in cameras for k in sorted(NEVER) if k not in shown]
        for camera, kind, _ in pick(sought, quota["absent_" + template], rng, used):
            who, what = NEVER[kind]
            out.append(ask(f"absent-{template}-{camera}-{kind}", f"When did {who} {template} {what} on {camera}?", "absent_" + template, WHEN,
                           realdump.grade("absent"), [], None, camera, *covered[camera]))
    sought = [(c, k, (a, b)) for (c, k), es in groups.items() if es and k in COUNTED for a, b in windows[c] if clear(c, k, a, b)]
    for camera, kind, (a, b) in pick(sought, quota["absent_count"], rng, used):
        out.append(ask(f"absent-count-{camera}-{kind}-{a:%H%M}-{b:%H%M}", count_text(camera, kind, a, b), "absent_count", NUMBER,
                       realdump.grade("absent"), [], None, camera, t_of(a), t_of(b)))
    return realdump.split(out, seed)
