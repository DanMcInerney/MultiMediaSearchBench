"""The questions of V10, every answer from the labels of Charades (source.py) and the pile built to make them certain (pile.py), each asked the
morning after the last clip. The pile is about 3,000 phone clips of people at home (acted: see source.py), and a question names what a person
remembers filming and asks for the clip, the clips, how many, where, for how long, whether another action was before or after it, or whether it was
ever filmed.

| template  | question                                                                   | rule                                          |
| `find`    | Find the clip of me washing a window.                                       | `set` of one file name                        |
| `every`   | Show me every clip of me opening a window. (2 to 6)                         | `set` of file names                           |
| `count`   | How many clips are there of me sneezing? (12 to 30)                         | `count`                                       |
| `rooms`   | How many clips did I film in the garage? (12 to 30)                         | `count`                                       |
| `long`    | For how many seconds was I making a sandwich?                               | `amount`, the labelled span, 30% or 3 s       |
| `room`    | Which room was I in when I was washing a window?                            | `text`: one of five rooms                     |
| `before`  | In the clip of me fixing a light, was I turning on the light before or      | `text`: before or after                       |
|           | after fixing a light?                                                       |                                               |
| `did`     | Did I ever film myself fixing a doorknob? If so, when?                      | `time` and date, 60 s: the clip's time        |
| `absent_*`| `find`, `every`, `count` or `did` of an action the pile never has           | `absent`                                      |

A question asks only of a topic the pile was built for (pile.py): its clip is one for which the label and the words of the script agree, and every
clip where they do not is out of the pile, so what a question asks for is all of it. A `one` topic has one clip in the pile and serves `find`, `long`,
`room`, `before` and `did` (the others only where that clip supports them: a span of 8 seconds or more for `long`, a scene of the five rooms
that the words name for `room`, a second labelled action of the table NAMED with a gap of 2 seconds or more to the topic's for `before`: the
order of two labelled spans is certain); a `few` topic serves `every`, a `many` topic `count`, a `none`
topic the absent twins. The five rooms are thinned so that `rooms` is a count a person could make. A `time` answer is the clip's own time, which a person reads from the file;
`long` is the labelled span of the action in its clip, which annotators place to a second or two. Which of the topics are asked is drawn with the
seed; the pile does not change with it, so the same topics are never in it, and a topic that is never in it is asked of once only. A split is a
whole topic's (or room's) questions, so that a dev answer gives away no test question.
There is no `after` question ("what did I do right after"): an answer is a free phrase that the grader can only match against a few words per
action, and it graded "turned the light on" wrong and "turned on the TV" right."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping, Sequence
from datetime import datetime

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.v10_home_clips.pile import (
    NAMED,
    ROOMS,
    TOPICS,
    Pile,
    long_enough,
    order_of,
    scene_said,
    span_of,
)

CODE = "V10"
SOURCE_NAME = "phone"  # the dump's source
FILES = "<the names of the files, as in VID_20250612_181502.mp4, separated by commas>"
WHEN = "<the date and time, as YYYY-MM-DD HH:MM>"
NUMBER = "<a number>"
SECONDS = "<a number of seconds>"
ROOM = "<the room>"
WORD = "<before or after>"
TIME_TOL_S = 60.0  # a clip is at most 45 s long, and its time is that of its start
QUOTAS = {"tiny": {"find": 1, "every": 1, "count": 1, "rooms": 1, "long": 0, "room": 1, "before": 0, "did": 1, "absent_find": 1, "absent_every": 0,
                   "absent_count": 0, "absent_did": 1},
          "standard": {"find": 3, "every": 2, "count": 3, "rooms": 2, "long": 2, "room": 2, "before": 2, "did": 2, "absent_find": 2, "absent_every": 1,
                       "absent_count": 1, "absent_did": 1}}


def length_tol(a: float, b: float) -> float:
    """What an answer to `long` may be out by: 30% of the span, and at least 3 s (annotators place a span's ends to a second or two)."""
    return max(3.0, round(0.3 * (b - a), 1))


def questions_of(pile: Pile, path: Mapping[str, str], when: Mapping[str, float], scale: str, seed: int, now: float, since: float,
                 until: float) -> tuple[Question, ...]:
    """The questions of `pile` (each clip's dump path in `path`, its time in `when`, both by clip id), drawn with `seed`; `now` is epoch seconds. A
    topic that is never in the pile is asked of once at most. Dev and test are drawn a topic at a time (a room is one too)."""
    rng = random.Random(f"v10:{seed}:{scale}")
    library = library_of(pile, path, when, now, since, until)
    used: set[str] = set()  # the topics already asked of, so that the questions differ; a pool that runs short repeats one
    out: list[tuple[str, Question]] = []
    for template, n in QUOTAS[scale].items():
        pool = sorted(library.get(template, []), key=lambda item: item[0])
        rng.shuffle(pool)
        pool.sort(key=lambda item: item[0] in used)  # the unused first (a stable sort keeps the shuffle)
        taken = 0
        for key, q in pool:
            if taken == n:
                break
            if q.kind == "absent" and key in used:
                continue  # the pile has none of its topic whatever the seed: the answer to one question would be the answer to the next
            used.add(key)
            out.append((key, q))
            taken += 1
    return split_by_topic(out, seed)


def split_by_topic(questions: Sequence[tuple[str, Question]], seed: int) -> tuple[Question, ...]:
    """The questions with about `realdump.DEV_SHARE` of them `dev` and the rest `test`, a topic's questions (`(topic, question)`) all in one, by a draw
    seeded with `seed`: a dev answer then says nothing of a test question. Both splits are held when there are two topics or more."""
    groups: dict[str, list[str]] = {}
    for key, q in questions:
        groups.setdefault(key, []).append(q.qid)
    order = sorted(groups)
    random.Random(f"split:{seed}").shuffle(order)
    target = round(realdump.DEV_SHARE * len(questions))
    dev: set[str] = set()
    for key in order:
        if len(dev) + len(groups[key]) <= target + 1 and len(dev) < target:
            dev |= set(groups[key])
    if not dev and len(order) > 1:
        dev = set(groups[min(order, key=lambda k: len(groups[k]))])
    return tuple(dataclasses.replace(q, split="dev" if q.qid in dev else "test") for _, q in questions)


def library_of(pile: Pile, path: Mapping[str, str], when: Mapping[str, float], now: float, since: float,
               until: float) -> dict[str, list[tuple[str, Question]]]:
    """Every question the pile can be asked, by template, each with the topic it is of."""
    out: dict[str, list[tuple[str, Question]]] = {}

    def ask(template: str, key: str, text: str, kind: str, grade, shape: str, evidence: Sequence[Span], value: str | None) -> None:
        qid = ("absent-" if kind == "absent" else "") + f"{template.removeprefix('absent_')}-{key}"
        q = realdump.question(code=CODE, qid=qid, now=now, text=text, kind=kind, template=template, modality="video", shape=shape, grade=grade,
                              evidence=evidence, sources=(SOURCE_NAME,), since=since, until=until, value=value)
        out.setdefault(template, []).append((key, q))

    def name(clip_id: str) -> str:
        return path[clip_id].rsplit("/", 1)[-1]

    def clock(clip_id: str) -> tuple[str, str]:
        d = datetime.fromtimestamp(when[clip_id])
        return d.strftime("%Y-%m-%d"), d.strftime("%H:%M:%S")

    for key, clip in pile.one.items():
        t = TOPICS[key]
        label = t.labels[0]
        a, b = span_of(clip, label)
        evidence = [Span(path[clip.id], a, b)]
        ask("find", key, f"Find the clip of me {t.phrase}.", "find", realdump.grade("set", items=[name(clip.id)]), FILES, evidence, name(clip.id))
        date, hms = clock(clip.id)
        ask("did", key, f"Did I ever film myself {t.phrase}? If so, when?", "did", realdump.grade("time", time=hms, tol_s=TIME_TOL_S, date=date), WHEN,
            evidence, f"{date} {hms[:5]}")
        if long_enough(clip, (a, b)):
            ask("long", key, f"For how many seconds was I {t.phrase}?", "long", realdump.grade("amount", truth=f"{b - a:.1f}", tol_s=length_tol(a, b)), SECONDS,
                evidence, f"{b - a:.1f}")
        if scene_said(clip):
            ask("room", key, f"Which room was I in when I was {t.phrase}?", "room", realdump.grade("text", truth=ROOMS[clip.scene][1]), ROOM, evidence,
                clip.scene)
        if (two := order_of(clip, label)) is not None:
            cls, side, (c, d) = two
            first = side == "before"
            ask("before", key, f"In the clip of me {t.phrase}, was I {NAMED[cls]} before or after {t.phrase}?", "before",
                realdump.grade("text", truth="before|earlier" if first else "after|later", wrong=("after", "later") if first else ("before", "earlier")), WORD,
                [Span(path[clip.id], a, b), Span(path[clip.id], c, d)], side)
    for key, clips in pile.few.items():
        spans = [Span(path[c.id], *span_of(c, TOPICS[key].labels[0]) or (0.0, c.length)) for c in clips]
        names = [name(c.id) for c in clips]
        ask("every", key, f"Show me every clip of me {TOPICS[key].phrase}.", "every", realdump.grade("set", items=names), FILES, spans, ", ".join(names))
    for key, clips in pile.many.items():
        spans = [Span(path[c.id], *span_of(c, TOPICS[key].labels[0]) or (0.0, c.length)) for c in clips]
        ask("count", key, f"How many clips are there of me {TOPICS[key].phrase}?", "count", realdump.grade("count", truth=str(len(clips))), NUMBER, spans,
            str(len(clips)))
    for room in pile.rooms:
        clips = [c for c in pile.clips if c.scene == room]
        ask("rooms", room.lower().replace(" ", "_"), f"How many clips did I film {ROOMS[room][0]}?", "rooms",
            realdump.grade("count", truth=str(len(clips))), NUMBER, [Span(path[c.id], 0.0, c.length) for c in clips], str(len(clips)))
    for key in pile.none:
        t = TOPICS[key]
        ask("absent_find", key, f"Find the clip of me {t.phrase}.", "absent", realdump.grade("absent"), FILES, [], None)
        ask("absent_every", key, f"Show me every clip of me {t.phrase}.", "absent", realdump.grade("absent"), FILES, [], None)
        ask("absent_count", key, f"How many clips are there of me {t.phrase}?", "absent", realdump.grade("absent"), NUMBER, [], None)
        ask("absent_did", key, f"Did I ever film myself {t.phrase}? If so, when?", "absent", realdump.grade("absent"), WHEN, [], None)
    return out
