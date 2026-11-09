"""The questions of V3, every answer from the source's labels (source.py), each asked the morning after the last match. The pile is the
clips of three matches (a folder for each day, a clip for each 30 seconds the follow-cam recorded), and a question is about
goals and shots, the two classes whose labels are made by a person and complete in the clips that are kept (a goal has no other label at
its moment; a shot is any shot that did not score). The questions name a match by its day, and a moment by the clock, as the files do.

| template     | question                                                                          | rule                                 |
| `when`       | When was the second goal of the match on 29 March? (the first or last shot of a   | `time` and date, 10 s                |
|              | half, too)                                                                        |                                      |
| `when_clip`  | In the clip from 11:26:30 on 29 March, when was the shot?                          | `time`, 5 s                          |
| `goals`      | How many goals are there in the match on 8 March?                                  | `count`                              |
| `shots`      | How many shots that did not score were there in the first half of the match on     | `count`                              |
|              | 8 March?                                                                          |                                      |
| `find`       | Find the clip of the first goal of the match on 26 April.                          | `set` of one file name               |
| `every`      | Find every clip with a goal in the match on 8 March. (or a shot in a half; 2 to 13) | `set` of file names                 |
| `did`        | Did anyone score between 18:50 and 19:10 on 8 March? If so, when? (or a shot that  | `time` and date, 10 s                |
|              | did not score, in four minutes)                                                   |                                      |
| `window`     | How many goals were scored between 11:40 and 11:50 on 29 March?                    | `count`                              |
| `before`     | In the clip from 18:51:30 on 8 March, was the tackle before or after the shot?     | `text`: before or after              |
| `side`       | Did the first goal of the match on 8 March go into the left-hand goal or the       | `text`: left or right                |
|              | right-hand one, as the camera sees it?                                            |                                      |
| `absent_did`, `absent_window` | the same of a stretch in which the labels say no goal (no shot)  | `absent`                             |

What is asked, and what is not:
- Only the classes a person labelled: goal and shot (tackle, only to ask which came first in a clip that has one tackle and one shot, and
  never how many: the source labels the successful tackles only). `pass_received` and `recovery`, which a script derived, are never read.
- A match is the clips the source kept: a window with fewer than three events is missing (`gaps` in source.py), so "how many goals are in the
  match" is how many are in the clips, and the absent questions ask of a stretch of the day in which every 30-second window has its clip
  (a goal cannot hide in a missing one), between two whole minutes.
- A moment is the label's position in its clip, which is a person's to the frame (25 to the second): its time is the clip's time and that
  position. The tolerance is 10 s for a moment in a match or a half, which an agent finds by watching, and 5 s inside a clip it is given.
- `side`: the label's `team` is the team on the left or right of the picture in that half, and the team attacks the opposite goal (checked
  on frames of seven of the eight goals: a `left` goal went into the right-hand goal, a `right` goal into the left-hand one).
Which are asked is drawn with the seed. Dev and test are drawn a group of questions at a time: questions that rest on a clip in common (the
same goal, shot or stretch) are in one split, so that a dev answer gives away no test question. For that there is no `every` of every goal of the
three matches, which would join all of them."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import File
from mmsb.scenarios.v3_match.source import SECONDS, Clip

CODE = "V3"
SOURCE_NAME = "followcam"  # the dump's source
ORDINALS = ("first", "second", "third", "fourth")
TOL_S = {"when": 10.0, "did": 10.0, "when_clip": 5.0}
WHEN = "<the date and time, as YYYY-MM-DD HH:MM:SS>"
CLOCK = "<the time, as HH:MM:SS>"
FILES = "<the names of the files, as in FC_190030.mp4, separated by commas>"
NUMBER = "<a number>"
WORD = "<before or after>"
SIDE = "<left or right>"
QUOTAS = {"tiny": {"when": 1, "when_clip": 1, "goals": 0, "shots": 0, "find": 1, "every": 0, "did": 1, "window": 1, "before": 0, "side": 1,
                   "absent_did": 1, "absent_window": 1},
          "standard": {"when": 2, "when_clip": 2, "goals": 1, "shots": 1, "find": 1, "every": 2, "did": 3, "window": 1, "before": 2, "side": 2,
                       "absent_did": 4, "absent_window": 2}}
WINDOWS = {"tiny": {"goal": 3, "shot": 2}, "standard": {"goal": 10, "shot": 4}}  # minutes of a `did` or `window` stretch
SET_SIZE = (2, 13)  # the clips an `every` question holds
HALVES = {1: "first", 2: "second"}


@dataclass(frozen=True)
class Moment:
    """A labelled event placed in the dump."""

    kind: str  # goal, shot, tackle
    t: float  # epoch seconds
    file: str  # its path in the dump
    pos: float  # seconds into the file
    team: str
    match: str
    half: int
    window: int


def day_text(t: float) -> str:
    d = datetime.fromtimestamp(t)
    return f"{d.day} {realdump.MONTHS[d.month - 1]}"


def upper(text: str) -> str:
    """`text` with its first letter in capitals and nothing else changed (`str.capitalize` makes "8 March" "8 march")."""
    return text[:1].upper() + text[1:]


def clock_text(t: float) -> str:
    return f"{datetime.fromtimestamp(t):%H:%M}"


def span_of(m: Moment) -> Span:
    return Span(m.file, max(0.0, m.pos - 1.0), min(SECONDS, m.pos + 1.0))


def moments_of(clips: Sequence[Clip], files: Sequence[File]) -> list[Moment]:
    return [Moment(e.label, f.t + e.pos, f.path, e.pos, e.team, c.match, c.half, c.window)
            for c, f in zip(clips, files, strict=True) for e in c.events if e.label in ("goal", "shot", "tackle")]


@dataclass(frozen=True)
class Pool:
    """What a pile has to be asked about: its clips and moments, and for each match its day and each half's windows."""

    clips: tuple[Clip, ...]
    files: tuple[File, ...]
    moments: tuple[Moment, ...]
    matches: tuple[str, ...]  # in order
    start: dict[tuple[str, int, int], float]  # (match, half, window) -> the epoch seconds its clip begins
    first: dict[str, float]  # match -> its first clip's time (the day)

    def of(self, kind: str, match: str | None = None, half: int | None = None) -> list[Moment]:
        return sorted((m for m in self.moments if m.kind == kind and match in (None, m.match) and half in (None, m.half)), key=lambda m: m.t)


def pool_of(clips: Sequence[Clip], files: Sequence[File]) -> Pool:
    start = {(c.match, c.half, c.window): f.t for c, f in zip(clips, files, strict=True)}
    matches = tuple(sorted({c.match for c in clips}))
    first = {m: min(t for (mm, _, _), t in start.items() if mm == m) for m in matches}
    return Pool(tuple(clips), tuple(files), tuple(moments_of(clips, files)), matches, start, first)


def ask(q_kind: str, qid: str, text: str, grade, shape: str, evidence: Sequence[Span], now: float, since: float, until: float,
        value: str | None, template: str | None = None) -> Question:
    return realdump.question(code=CODE, qid=qid, now=now, text=text, kind=q_kind, template=template or q_kind, modality="video", shape=shape,
                             grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since, until=until, value=value)


def questions_of(pool: Pool, scale: str, seed: int, now: float) -> tuple[Question, ...]:
    """The questions of the pile `pool`, drawn with `seed` and asked at `now` (epoch seconds)."""
    since = min(f.t for f in pool.files)
    until = max(f.t + f.length for f in pool.files)
    library = pool_library(pool, scale, now, since, until)
    rng = random.Random(f"v3:{seed}:{scale}")
    out: list[Question] = []
    for template, n in QUOTAS[scale].items():
        items = sorted(library.get(template, []), key=lambda item: item[0].qid)
        rng.shuffle(items)
        taken: list[object] = []
        for q, about in items:
            if len(taken) == n:
                break
            if not any(overlaps(about, other) for other in taken):
                taken.append(about)
                out.append(q)
    return split_by_evidence(out, seed)


def split_by_evidence(questions: Sequence[Question], seed: int) -> tuple[Question, ...]:
    """The questions with about `realdump.DEV_SHARE` of them `dev` and the rest `test`, drawn by a seeded draw a group at a time: two questions that cite
    a clip in common (or are joined by a third that does) are in one group, and so in one split; a question that cites nothing (an absent one) is
    a group of its own. Both splits are held when there are two groups or more."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q in questions:
        files = sorted({s.file for s in q.truth.evidence})
        for f in files:
            parent.setdefault(f, f)
        for f in files[1:]:
            parent[find(f)] = find(files[0])
    groups: dict[str, list[str]] = {}
    for q in questions:
        groups.setdefault(find(q.truth.evidence[0].file) if q.truth.evidence else q.qid, []).append(q.qid)
    order = sorted(groups)
    random.Random(f"split:{seed}").shuffle(order)
    target = round(realdump.DEV_SHARE * len(questions))
    dev: set[str] = set()
    for key in order:
        if len(dev) + len(groups[key]) <= target + 1 and len(dev) < target:
            dev |= set(groups[key])
    if not dev and len(order) > 1:
        dev = set(groups[min(order, key=lambda k: len(groups[k]))])
    return tuple(dataclasses.replace(q, split="dev" if q.qid in dev else "test") for q in questions)


def overlaps(one: object, other: object) -> bool:
    """Whether two things a question is about are alike: the same clips, or two stretches of a half that share a moment."""
    if isinstance(one, tuple) and isinstance(other, tuple):
        return one[:2] == other[:2] and one[2] < other[3] and other[2] < one[3]
    return one == other


def stretches(pool: Pool, scale: str) -> dict[str, list[tuple[str, int, float, float]]]:
    """For each of `goal` and `shot`, the stretches of the pile in which every 30-second window has its clip, from a whole minute to a whole minute
    of the clock, as `(match, half, t0, t1)`: for each match and half, each start at a whole minute of the window slots, WINDOWS[scale] long."""
    out: dict[str, list[tuple[str, int, float, float]]] = {"goal": [], "shot": []}
    have = {k for k in pool.start}
    for kind, minutes in WINDOWS[scale].items():
        n = minutes * 2  # windows
        for match in pool.matches:
            for half in (1, 2):
                windows = sorted(w for (m, h, w) in have if m == match and h == half)
                if not windows:
                    continue
                for w in range(windows[0], windows[-1] - n + 2):
                    t0 = pool.start.get((match, half, w))
                    if t0 is not None and datetime.fromtimestamp(t0).second == 0 and all((match, half, w + i) in have for i in range(n)):
                        out[kind].append((match, half, t0, t0 + n * SECONDS))
    return out


def pool_library(pool: Pool, scale: str, now: float, since: float, until: float) -> dict[str, list[tuple[Question, object]]]:
    """Every question the pile can be asked, by template, each with what it is about: the clips of its evidence, or for a stretch of the day
    `(match, half, t0, t1)`; `questions_of` asks no two of one template about the same."""
    out: dict[str, list[tuple[Question, object]]] = {}
    stretch: list[object] = [None]  # the stretch the questions made next are about

    def put(template: str, q: Question) -> None:
        about = stretch[0] or frozenset(s.file for s in q.truth.evidence) or q.qid
        out.setdefault(template, []).append((q, about))

    def day(match: str) -> str:
        return day_text(pool.first[match])

    def when_of(m: Moment) -> tuple[str, str]:
        d = datetime.fromtimestamp(m.t)
        return d.strftime("%Y-%m-%d"), d.strftime("%H:%M:%S")

    whole = scale == "standard"  # the tiny pile is a slice of one half: a question about a half or a match is asked only where the slice answers it too
    for match in pool.matches:
        goals = pool.of("goal", match)
        shots = pool.of("shot", match)
        # when, find, side: the nth goal of a match; the first or last shot of a half
        for i, g in enumerate(goals[:len(ORDINALS) if whole else 1]):  # at tiny the first goal of the match, which the slice is built round
            date, clock = when_of(g)
            what = f"{ORDINALS[i]} goal of the match on {day(match)}"
            put("when", ask("when", f"when-goal-{match}-{i + 1}", f"When was the {what}?", realdump.grade("time", time=clock, tol_s=TOL_S["when"], date=date),
                            WHEN, [span_of(g)], now, since, until, f"{date} {clock}"))
            put("find", ask("find", f"find-goal-{match}-{i + 1}", f"Find the clip of the {what}.", realdump.grade("set", items=[g.file.rsplit("/", 1)[-1]]),
                            FILES, [span_of(g)], now, since, until, g.file.rsplit("/", 1)[-1]))
            right = g.team == "left"  # the team on the left attacks the right-hand goal
            put("side", ask("side", f"side-goal-{match}-{i + 1}", f"Did the {what} go into the left-hand goal or the right-hand goal, as the camera sees it?",
                            realdump.grade("text", truth="right|right hand" if right else "left|left hand", wrong=("left", "left hand") if right else ("right", "right hand")),
                            SIDE, [span_of(g)], now, since, until, "right" if right else "left"))
        for half in (1, 2) if whole else ():
            hs = pool.of("shot", match, half)
            for ordinal, shot in (("first", hs[:1]), ("last", hs[-1:])):
                if len(hs) >= 3 and shot:
                    date, clock = when_of(shot[0])
                    put("when", ask("when", f"when-shot-{match}-{half}-{ordinal}", f"When was the {ordinal} shot that did not score in the {HALVES[half]} half of the match on {day(match)}?",
                                    realdump.grade("time", time=clock, tol_s=TOL_S["when"], date=date), WHEN, [span_of(shot[0])], now, since, until, f"{date} {clock}"))
        if goals and whole:
            put("goals", ask("count", f"goals-{match}", f"How many goals are there in the match on {day(match)}?", realdump.grade("count", truth=str(len(goals))), NUMBER,
                             [span_of(g) for g in goals], now, since, until, str(len(goals)), "goals"))
            files = sorted({g.file for g in goals})
            if len(goals) >= SET_SIZE[0]:
                put("every", ask("every", f"every-goal-{match}", f"Find every clip with a goal in the match on {day(match)}.",
                                 realdump.grade("set", items=[f.rsplit("/", 1)[-1] for f in files]), FILES, [span_of(g) for g in goals], now, since, until,
                                 ", ".join(f.rsplit("/", 1)[-1] for f in files)))
        for half in (1, 2) if whole else ():
            hs = pool.of("shot", match, half)
            if hs:
                put("shots", ask("count", f"shots-{match}-{half}", f"How many shots that did not score were there in the {HALVES[half]} half of the match on {day(match)}?",
                                 realdump.grade("count", truth=str(len(hs))), NUMBER, [span_of(s) for s in hs], now, since, until, str(len(hs)), "shots"))
            files = sorted({s.file for s in hs})
            if SET_SIZE[0] <= len(files) <= SET_SIZE[1]:
                put("every", ask("every", f"every-shot-{match}-{half}", f"Find every clip with a shot that did not score in the {HALVES[half]} half of the match on {day(match)}.",
                                 realdump.grade("set", items=[f.rsplit("/", 1)[-1] for f in files]), FILES, [span_of(s) for s in hs], now, since, until,
                                 ", ".join(f.rsplit("/", 1)[-1] for f in files)))
    # when_clip, before: a clip named by its time
    by_file: dict[str, list[Moment]] = {}
    for m in pool.moments:
        by_file.setdefault(m.file, []).append(m)
    for file, ms in sorted(by_file.items()):
        t0 = ms[0].t - ms[0].pos
        where = f"in the clip from {datetime.fromtimestamp(t0):%H:%M:%S} on {day_text(t0)}"
        shots, tackles, goals = ([m for m in ms if m.kind == k] for k in ("shot", "tackle", "goal"))
        if len(shots) == 1 and not goals:
            clock = datetime.fromtimestamp(shots[0].t).strftime("%H:%M:%S")
            put("when_clip", ask("when_clip", f"when_clip-{file.rsplit('/', 1)[-1][:-4]}", f"{upper(where)}, when was the shot that did not score?",
                                 realdump.grade("time", time=clock, tol_s=TOL_S["when_clip"]), CLOCK, [span_of(shots[0])], now, since, until, clock))
            if len(tackles) == 1 and abs(tackles[0].pos - shots[0].pos) >= 3.0:
                first_is_tackle = tackles[0].pos < shots[0].pos
                put("before", ask("before", f"before-{file.rsplit('/', 1)[-1][:-4]}", f"{upper(where)}, was the tackle before or after the shot?",
                                  realdump.grade("text", truth="before|earlier" if first_is_tackle else "after|later",
                                                 wrong=("after", "later") if first_is_tackle else ("before", "earlier")),
                                  WORD, [span_of(tackles[0]), span_of(shots[0])], now, since, until, "before" if first_is_tackle else "after"))
    # did, window: a stretch of the day in which every window has its clip
    for kind, found in stretches(pool, scale).items():
        minutes = WINDOWS[scale][kind]
        for match, half, t0, t1 in found:
            inside = [m for m in pool.moments if m.match == match and m.half == half and t0 <= m.t < t1 and m.kind in ("goal", "shot")]
            goals = [m for m in inside if m.kind == "goal"]
            shots = [m for m in inside if m.kind == "shot"]
            between = f"between {clock_text(t0)} and {clock_text(t1)} on {day_text(t0)}"
            stretch[0] = (match, half, t0, t1)
            tag = f"{kind}-{match}-{half}-{datetime.fromtimestamp(t0):%H%M}"
            if kind == "goal":
                if len(goals) == 1 and 5.0 <= goals[0].t - t0 <= minutes * 60 - 5.0:
                    date, clock = when_of(goals[0])
                    put("did", ask("did", f"did-{tag}", f"Did anyone score {between}? If so, when?", realdump.grade("time", time=clock, tol_s=TOL_S["did"], date=date),
                                   WHEN, [span_of(goals[0])], now, since, until, f"{date} {clock}"))
                    put("window", ask("window", f"window-{tag}", f"How many goals were scored {between}?", realdump.grade("count", truth="1"), NUMBER,
                                      [span_of(goals[0])], now, since, until, "1"))
                elif len(goals) == 2:
                    put("window", ask("window", f"window-{tag}", f"How many goals were scored {between}?", realdump.grade("count", truth="2"), NUMBER,
                                      [span_of(g) for g in goals], now, since, until, "2"))
                elif not goals:
                    put("absent_did", ask("absent", f"absent-did-{tag}", f"Did anyone score {between}? If so, when?", realdump.grade("absent"), WHEN, [], now, since,
                                          until, None, "absent_did"))
                    put("absent_window", ask("absent", f"absent-window-{tag}", f"How many goals were scored {between}?", realdump.grade("absent"), NUMBER, [], now,
                                             since, until, None, "absent_window"))
            elif not goals:  # a shot that did not score: one in the stretch, or none (and no goal either)
                if len(shots) == 1:
                    date, clock = when_of(shots[0])
                    put("did", ask("did", f"did-{tag}", f"Was there a shot that did not score {between}? If so, when?",
                                   realdump.grade("time", time=clock, tol_s=TOL_S["did"], date=date), WHEN, [span_of(shots[0])], now, since, until, f"{date} {clock}"))
                elif not shots:
                    put("absent_did", ask("absent", f"absent-did-{tag}", f"Was there a shot that did not score {between}? If so, when?", realdump.grade("absent"), WHEN, [], now,
                                          since, until, None, "absent_did"))
    return out
