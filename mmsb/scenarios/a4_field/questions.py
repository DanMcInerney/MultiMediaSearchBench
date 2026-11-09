"""The questions of A4, every answer from the expert's marks of the flight calls (source.py: points in time, with a pitch, and no species),
each asked the morning after the night. Times are the recorder's clock, which begins the night at 03:31:02.

| template | question                                                                                 | rule                        |
| `first`  | When was the first flight call of the night?                                             | `time` with the date, 60 s  |
| `pitch`  | At about what pitch, in kilohertz, was the first (last) flight call of the night?        | `amount`, 0.8 kilohertz     |
| `did`    | Did the recorder pick up any flight call between 03:46 and 03:49? If so, when was the   | `time`, 60 s                |
|          | first one? (two or three minutes, with a call or none)                                   |                             |
| `count`  | About how many flight calls did the recorder pick up between 04:06 and 04:10?            | `amount`, 1 call            |
| `quiet`  | Which recording has the fewest flight calls in it? (a comparison: filed as who)          | `set` of one file name      |
| `absent` | `did` of a window with no flight call                                                    | `absent`                    |

What the labels may not be. The expert marked 35,402 calls in six units' nights and may have missed faint ones, so no answer rests on an exact
count or on a gap of exactly this length:
- a count is asked only of a window of two to six minutes with three to twelve calls in it, whose edges are at least 1.5 s from a call, and is
  graded within one call (`amount`, 1);
- the first call of a window is asked only where no call is marked in the 90 s after the window's start (so the window's start is no answer); a
  call is graded to the minute either side, as a person reads a clock. No question asks when the last call was: it is 0.3 s before the end of the
  recording, so the last file's start and length give it. No question asks how long the recorder ran before the first call: with the start time
  it gives the answer of `first`;
- the pitch is the one the expert marked a call at, a point of its sweep: a call is asked about only where it is the first or the last, and the
  answer is within 0.8 kHz of the mark;
- **an empty window** is asked (`absent`) only where it is empty in all six annotation files, the six units having run the same night within two
  seconds of each other (unit 10 from 2.7 hours in), with a margin of 10 s from the call on either side: no unit marked a call in it, so it is
  a quiet stretch of the night for the expert who marked 35,402 calls, not only a stretch where one unit's list has none. The night has only a few
  stretches so quiet (its first seven minutes, the next eight), so the empty windows are all in its first quarter hour. A window with a call and an
  empty one are alike in shape: two or three minutes at clock minutes, as a person writes them. One window with a call is always one that
  starts before the night's first call and holds it (03:46 to 03:49 at seed 1), the others are drawn; it has the answer of `first`, so it is kept
  in the split of `first` (a question of one split does not give the answer of one of the other). **The position tell that remains:** a window
  that ends before 03:48, the first call of the night, is an empty one, and a window that starts after it holds calls, which the data has;
- `quiet` is asked only where the fewest has at most half the calls of the next, and over the files the pile has.

Which windows and which calls are asked is drawn with the seed, no two windows with a call overlapping."""

from __future__ import annotations

import bisect
import dataclasses
import random
from collections.abc import Sequence
from datetime import datetime, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.a4_field.source import HOUR_S, Night

CODE = "A4"
SOURCE_NAME = "recorder"  # the dump's source
TOL_S = 60.0  # a call is told to the second and a person says its minute: the minute either side is the answer
CALLS_TOL = 1.0  # `count`: a call
PITCH_TOL_KHZ = 0.8
EDGE_S = 1.5  # no call within this many seconds of a window's edge
MARGIN_S = 10.0  # an empty window is this far from the call at either side
LEAD_S = 90.0  # the first call of a window with a call is at least this long after the window's start (more than TOL_S)
COUNT_CALLS = (3, 12)  # the calls in a `count` window
COUNT_MINUTES = (2, 6)
DID_MINUTES = (2, 3)  # a window with a call and one with none are of the same shape
EMPTY_MINUTES = (2, 3)
QUOTAS = {"tiny": {"first": 1, "pitch_first": 1, "pitch_last": 0, "quiet": 1, "did": 1, "count": 1, "absent_did": 1},
          "standard": {"first": 1, "pitch_first": 1, "pitch_last": 1, "quiet": 1, "did": 5, "count": 5, "absent_did": 4}}
WHEN = "<the date and time, as YYYY-MM-DD HH:MM>"
CLOCK = "<the time, as HH:MM>"
KILOHERTZ = "<a number of kilohertz>"
NUMBER = "<a number>"
FILE = "<the file name, as in REC_20150101_120000.flac>"


def questions_of(night: Night, files: Sequence[realdump.File], scale: str, seed: int, now: float) -> tuple[Question, ...]:
    """The questions of the recording `files` (hourly, the first of them at the night's start), over `night`, at `scale`, drawn with `seed`;
    `now` is epoch seconds."""
    quota = QUOTAS[scale]
    rng = random.Random(f"a4:{seed}:{scale}")
    covered = sum(f.length for f in files)  # seconds of the night the pile holds
    calls = [c for c in night.calls if c.t < covered]
    times = [c.t for c in calls]
    everyone = sorted([*times, *(t for t in night.others if t < covered)])
    since, until = files[0].t, files[-1].t + files[-1].length

    def wall(t: float) -> datetime:
        return night.start + timedelta(seconds=int(t))  # to the second below

    def span(t: float) -> Span:
        n = min(int(t // HOUR_S), len(files) - 1)
        low = t - n * HOUR_S
        return Span(files[n].path, max(0.0, low - 0.5), min(files[n].length, low + 0.5))

    def within(a: float, b: float) -> list[float]:
        return times[bisect.bisect_left(times, a):bisect.bisect_left(times, b)]

    def clear(a: float, b: float) -> bool:
        """No call within EDGE_S of either edge."""
        return not within(a - EDGE_S, a + EDGE_S) and not within(b - EDGE_S, b + EDGE_S)

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[Span], value: str | None, absent: bool = False) -> Question:
        return realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text, kind="absent" if absent else template,
                                 template="absent_" + template if absent else template, modality="audio", shape=shape, grade=grade, evidence=found,
                                 sources=(SOURCE_NAME,), since=since, until=until, value=value)

    def clock(t: float) -> str:
        return f"{wall(t):%H:%M}"

    def tag(a: float, b: float) -> str:
        return f"{wall(a):%H%M}-{wall(b):%H%M}"

    first_minute = (-night.start.second) % 60  # seconds into the recording at which its first clock minute begins
    minutes = [first_minute + 60 * i for i in range(int((covered - first_minute) // 60) + 1)]  # the clock minutes, as seconds into the recording
    taken: list[tuple[float, float]] = []

    def free(a: float, b: float) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def draw(candidates: Sequence[tuple[float, float]], n: int, apart: bool = True) -> list[tuple[float, float]]:
        """`n` of the windows, drawn with the seed, no two overlapping (those of an empty stretch may: it is one stretch asked about twice,
        and there are few) and none overlapping one drawn before."""
        order = list(candidates)
        rng.shuffle(order)
        out: list[tuple[float, float]] = []
        for a, b in order:
            if len(out) < n and (free(a, b) if apart else (a, b) not in out):
                out.append((a, b))
                taken.append((a, b))
        if len(out) < n:
            raise ValueError(f"only {len(out)} of the {n} windows a question asks about were found in the {covered:g} s the pile holds")
        return out

    def windows(low: int, high: int) -> list[tuple[float, float]]:
        return [(a, b) for i, a in enumerate(minutes) for b in minutes[i + low:i + high + 1] if b <= covered]

    out: list[Question] = []
    # the first call of the night
    c0, c1 = calls[0], calls[-1]
    if quota["first"]:
        when = wall(c0.t)
        out.append(ask("first", "When was the first flight call of the night?", "first", WHEN,
                       realdump.grade("time", time=when.strftime("%H:%M:%S"), tol_s=TOL_S, date=when.date().isoformat()), [span(c0.t)],
                       when.strftime("%Y-%m-%d %H:%M")))
    for template, call, role in (("first", c0, "pitch_first"), ("last", c1, "pitch_last")):
        if quota[role]:
            khz = f"{call.hz / 1000:g}"
            out.append(ask(f"pitch-{template}", f"At about what pitch, in kilohertz, was the {template} flight call of the night?", "pitch", KILOHERTZ,
                           realdump.grade("amount", truth=khz, tol_s=PITCH_TOL_KHZ), [span(call.t)], khz))
    # the recording with the fewest calls
    if quota["quiet"] and len(files) > 1:
        by_file = [len(within(k * HOUR_S, (k + 1) * HOUR_S)) for k in range(len(files))]
        order = sorted(range(len(files)), key=lambda k: by_file[k])
        if by_file[order[0]] * 2 > by_file[order[1]]:
            raise ValueError(f"no recording has at most half the calls of the next: {by_file}")
        name = files[order[0]].path.rsplit("/", 1)[-1]
        out.append(ask("quiet", "Which recording has the fewest flight calls in it?", "quiet", FILE, realdump.grade("set", items=[name]),
                       [Span(files[order[0]].path, 0.0, files[order[0]].length)], name))
    # a window with a call: when was the first
    positive = []
    for a, b in windows(*DID_MINUTES):
        inside = within(a, b)
        if inside and inside[0] - a >= LEAD_S and b - inside[0] >= EDGE_S and not within(a - EDGE_S, a) and not within(b - EDGE_S, b + EDGE_S):
            positive.append((a, b))
    early = [w for w in positive if within(*w)[0] == c0.t]  # the windows that start before the night's first call and hold it
    chosen = draw(early, 1) + draw(positive, quota["did"] - 1) if quota["did"] else []
    for a, b in chosen:
        first = within(a, b)[0]
        when = wall(first)
        out.append(ask(f"did-{tag(a, b)}", f"Did the recorder pick up any flight call between {clock(a)} and {clock(b)}? If so, when was the first one?",
                       "did", CLOCK, realdump.grade("time", time=when.strftime("%H:%M:%S"), tol_s=TOL_S), [span(first)], when.strftime("%H:%M")))
    # a window with none, in a stretch where no unit marked a call
    empty = []
    for a, b in windows(*EMPTY_MINUTES):
        near = everyone[bisect.bisect_left(everyone, a - MARGIN_S):bisect.bisect_left(everyone, b + MARGIN_S)]
        if not near:
            empty.append((a, b))
    for a, b in draw(empty, quota["absent_did"], apart=False):
        out.append(ask(f"did-{tag(a, b)}", f"Did the recorder pick up any flight call between {clock(a)} and {clock(b)}? If so, when was the first one?",
                       "did", CLOCK, realdump.grade("absent"), [], None, absent=True))
    # a count of the calls in a short window with few
    few = []
    for a, b in windows(*COUNT_MINUTES):
        if COUNT_CALLS[0] <= len(within(a, b)) <= COUNT_CALLS[1] and clear(a, b):
            few.append((a, b))
    for a, b in draw(few, quota["count"]):
        inside = within(a, b)
        out.append(ask(f"count-{tag(a, b)}", f"About how many flight calls did the recorder pick up between {clock(a)} and {clock(b)}?", "count",
                       NUMBER, realdump.grade("amount", truth=str(len(inside)), tol_s=CALLS_TOL), [span(t) for t in inside], str(len(inside))))
    split = list(realdump.split(out, seed))
    if quota["first"] and quota["did"]:  # the window that holds the night's first call has the answer of `first`: both are of one split
        mine = next(q.split for q in split if q.qid == "first")
        at = next(i for i, q in enumerate(split) if q.qid.startswith("did-") and q.kind != "absent" and q.truth.evidence[0] == span(c0.t))
        split[at] = dataclasses.replace(split[at], split=mine)
    return tuple(split)
