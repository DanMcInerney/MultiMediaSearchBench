"""Questions from what was said (DESIGN 5.8): the timed transcripts of the scenarios whose pile is talk, and the question templates over them. A
scenario reads its transcripts into `Talk`s and gives a `Say` (how the person calls things), its `Term`s, the terms that were never said and any
`Curated` questions, written by reading the transcripts. `questions_of` makes the questions, every answer from the transcripts:

| template | the person says                                                      | category | rule                     |
| `first`  | When did the lecturer first mention the knapsack problem?            | when     | `time` with the date     |
| `did`    | Did the lecturer ever mention bootstrapping? If so, when first?      | did      | `time` with the date     |
| `which`  | In which lecture did the lecturer talk about Monte Carlo simulation? | find     | `set` of one file        |
| `all`    | Which lectures mention greedy algorithms?                            | find_all | `set` of files           |
| `count`  | In how many lectures did the lecturer mention regression?            | how_many | `count`                  |
| `long`   | How long, in minutes, was the lecture on hypothesis testing?         | how_many | `amount`, a minute       |
| `order`  | In the lecture of Tuesday 3 May, was A before or after B?            | order    | `text`, before or after  |

An absent question is `first`, `did` or `which` of a term that is in no transcript: a transcript is every word, so its absence is certain, and it
reads no differently. A `Curated` question is checked at build: the phrase it rests on must be in the cues at its evidence, so a written answer
cannot drift from the transcript."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from mmsb.core.grade import base_name
from mmsb.core.question import Grade, Question, Span
from mmsb.core.taxonomy import with_absent
from mmsb.scenarios import realdump

TOL_S = 60.0  # the answer is asked for to the minute, and so may be up to 59 s from the cue
LONG_TOL_MIN = 1.0  # the minutes a `long` answer may be off by
MIN_APART_S = 120.0  # the two terms of an `order` question are at least this far apart
EARLIEST_S = 120.0  # a `first` or `did` moment is not in the first minutes of the pile: an intro, and the file list's first file
MAX_FILES = 6  # an `all` question lists no more files than this
WHEN = "<the date and time, as YYYY-MM-DD HH:MM>"
FILE = "<the file name>"
FILES = "<the file names, separated by commas>"
NUMBER = "<a number>"
MINUTES = "<a number of minutes>"
BEFORE_AFTER = "<before or after>"
FILING = with_absent({"first": ("when", "speech"), "did": ("did", "speech"), "which": ("find", "speech"), "all": ("find_all", "speech"),
                      "count": ("how_many", "speech"), "long": ("how_many", "speech"), "order": ("order", "speech"),
                      "what": ("what", "speech"), "who": ("who", "speech"), "before": ("order", "speech")})
TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # a caption's note of a sound: [SQUEAKING], (LAUGHTER)
SPEAKER = re.compile(r"^\s*([A-Z][A-Z .'-]{1,40}):\s*")  # a caption's name at a change of speaker: SARAH HANSEN:
SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d+)")


@dataclass(frozen=True)
class Cue:
    t0: float  # seconds into the file
    t1: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Talk:
    path: str  # in the dump
    start: datetime  # wall clock, no zone
    length: float  # seconds
    cues: tuple[Cue, ...]


@dataclass(frozen=True)
class Term:
    search: str  # the words looked for ("knapsack problem"); the last may have an s
    said: str  # as the person says it ("the knapsack problem")


@dataclass(frozen=True)
class Say:
    code: str  # "V9"
    source: str  # the dump's source
    modality: str  # video, audio
    person: str  # "the lecturer": said in the questions
    thing: str  # "lecture"
    things: str  # "lectures"


@dataclass(frozen=True)
class Curated:
    """A question written by reading the transcripts. `check` is a phrase that must be in the cues of `talk` that touch `t0` to `t1`."""

    key: str
    text: str
    template: str  # `what`, `who` or `before`, or another key of the scenario's filing
    shape: str
    grade: Grade
    talk: int  # the index of the talk the answer is in
    t0: float
    t1: float
    check: str
    value: str


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 10 ** len(ms)


def read_srt(text: str) -> tuple[Cue, ...]:
    """The cues of an SRT file in order, its notes of sounds taken out and empty cues left out, with the speaker of each where a caption names
    one (`NAME:`): a name holds until the next one."""
    cues, speaker = [], None
    for block in re.split(r"\r?\n\s*\r?\n", text.lstrip("﻿").strip()):
        lines = block.splitlines()
        at = next((i for i, line in enumerate(lines) if SRT_TIME.search(line)), None)
        if at is None:
            continue
        t = SRT_TIME.search(lines[at])
        body = " ".join(TAG.sub(" ", " ".join(lines[at + 1:])).split())
        if found := SPEAKER.match(body):
            speaker, body = found[1].strip().title(), body[found.end():]
        if body:
            cues.append(Cue(seconds(*t.groups()[:4]), seconds(*t.groups()[4:]), body, speaker))
    return tuple(cues)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def matches(words: Sequence[str], wanted: Sequence[str], at: int) -> bool:
    """Whether `wanted` is at `at` of `words`, as whole words, the last word perhaps with an s or es."""
    if at + len(wanted) > len(words) or list(words[at:at + len(wanted) - 1]) != list(wanted[:-1]):
        return False
    return words[at + len(wanted) - 1] in (wanted[-1], wanted[-1] + "s", wanted[-1] + "es")


def mentions(talk: Talk, search: str) -> list[Cue]:
    """The cue that each mention of the words `search` starts in, in order: a phrase may run over from one cue into the next."""
    wanted = tokens(search)
    stream = [(word, i) for i, cue in enumerate(talk.cues) for word in tokens(cue.text)]
    words = [w for w, _ in stream]
    return [talk.cues[stream[at][1]] for at in range(len(words)) if matches(words, wanted, at)]


def said_anywhere(talks: Sequence[Talk], search: str) -> bool:
    return any(mentions(t, search) for t in talks)


def date_of(talk: Talk) -> str:
    return realdump.date_text(talk.start.date())


def span_of(talk: Talk, cue: Cue) -> Span:
    return Span(talk.path, cue.t0, cue.t1)


def check_curated(talks: Sequence[Talk], curated: Sequence[Curated]) -> None:
    """Raises when the phrase a written question rests on is not in the cues at its evidence, within the 5 s a citation of a sound is held to."""
    for c in curated:
        talk = talks[c.talk]
        near = Talk(talk.path, talk.start, talk.length, tuple(q for q in talk.cues if q.t1 >= c.t0 - 5 and q.t0 <= c.t1 + 5))
        if not mentions(near, c.check):
            raise ValueError(f"the curated question {c.key}: {c.check!r} is not said in {talk.path} from {c.t0} to {c.t1}")


def unused(terms: Sequence[Term], used: set[str]) -> Iterator[Term]:
    return (t for t in terms if t.search not in used)


def questions_of(talks: Sequence[Talk], terms: Sequence[Term], never: Sequence[Term], say: Say, quota: dict[str, int], seed: int,
                 t_of: Callable[[datetime], float], curated: Sequence[Curated] = ()) -> tuple[Question, ...]:
    """The questions of `talks`, in the order of their starts, drawn with `seed`. `quota` has `first`, `did`, `which`, `all`, `count`, `long`,
    `order` and `absent_first`, `absent_did`, `absent_which`; a term is asked about once. `t_of` is a wall time's epoch seconds."""
    if bad := [t.search for t in never if said_anywhere(talks, t.search)]:
        raise ValueError(f"{say.code}: said after all, and so not absent: {', '.join(bad)}")
    check_curated(talks, curated)
    rng = random.Random(f"{say.code}:{seed}")
    last = talks[-1]
    now = t_of(datetime.combine(last.start.date() + timedelta(days=1), datetime.min.time().replace(hour=9)))
    since, until = t_of(talks[0].start), t_of(last.start) + last.length

    def ask(qid: str, text: str, template: str, shape: str, grade: Grade, evidence: Sequence[Span], value: str | None) -> Question:
        absent = template.startswith("absent_")
        return realdump.question(code=say.code, qid=qid, now=now, text=text, kind="absent" if absent else template, template=template,
                                 modality=say.modality, shape=shape, grade=grade, evidence=evidence, sources=(say.source,), since=since,
                                 until=until, value=value)

    def where(term: Term) -> list[tuple[Talk, list[Cue]]]:
        return [(t, found) for t in talks if (found := mentions(t, term.search))]

    def moment(talk: Talk, cue: Cue) -> tuple[str, str]:
        wall = talk.start + timedelta(seconds=cue.t0)
        return wall.strftime("%H:%M:%S"), wall.date().isoformat()

    def when_grade(talk: Talk, cue: Cue) -> Grade:
        clock, day = moment(talk, cue)
        return realdump.grade("time", time=clock, tol_s=TOL_S, date=day)

    def value_of(talk: Talk, cue: Cue) -> str:
        clock, day = moment(talk, cue)
        return f"{day} {clock}"

    def key(term: Term) -> str:
        return "-".join(tokens(term.search))

    out: list[Question] = []
    used: set[str] = set()
    pool = list(terms)
    rng.shuffle(pool)

    def take(n: int, fits: Callable[[Term], bool]) -> list[Term]:
        chosen = [t for t in unused(pool, used) if fits(t)][:n]
        used.update(t.search for t in chosen)
        return chosen

    def not_at_the_start(t: Term) -> bool:
        found = where(t)
        return bool(found) and not (found[0][0] is talks[0] and found[0][1][0].t0 < EARLIEST_S)

    for term in take(quota.get("first", 0), not_at_the_start):
        talk, found = where(term)[0]
        out.append(ask(f"first-{key(term)}", f"When did {say.person} first mention {term.said}?", "first", WHEN, when_grade(talk, found[0]),
                       [span_of(talk, found[0])], value_of(talk, found[0])))
    for term in take(quota.get("did", 0), not_at_the_start):
        talk, found = where(term)[0]
        out.append(ask(f"did-{key(term)}", f"Did {say.person} ever mention {term.said}? If so, when was the first time?", "did", WHEN,
                       when_grade(talk, found[0]), [span_of(talk, found[0])], value_of(talk, found[0])))
    for term in take(quota.get("which", 0), lambda t: len(where(t)) == 1):
        talk, found = where(term)[0]
        out.append(ask(f"which-{key(term)}", f"In which {say.thing} did {say.person} talk about {term.said}?", "which", FILE,
                       realdump.grade("set", items=[base_name(talk.path)]), [span_of(talk, found[0])], base_name(talk.path)))
    for term in take(quota.get("all", 0), lambda t: 2 <= len(where(t)) <= MAX_FILES):
        found = where(term)
        items = sorted(base_name(t.path) for t, _ in found)
        out.append(ask(f"all-{key(term)}", f"Which {say.things} mention {term.said}?", "all", FILES, realdump.grade("set", items=items),
                       [span_of(t, cues[0]) for t, cues in found], ", ".join(items)))
    for term in take(quota.get("count", 0), lambda t: 2 <= len(where(t)) < len(talks)):
        found = where(term)
        out.append(ask(f"count-{key(term)}", f"In how many {say.things} did {say.person} mention {term.said}?", "count", NUMBER,
                       realdump.grade("count", truth=str(len(found))), [span_of(t, cues[0]) for t, cues in found], str(len(found))))
    for term in take(quota.get("long", 0), lambda t: len(where(t)) == 1):
        talk, found = where(term)[0]
        minutes = round(talk.length / 60, 1)
        out.append(ask(f"long-{key(term)}", f"How long, in minutes, was the {say.thing} in which {say.person} talked about {term.said}?", "long",
                       MINUTES, realdump.grade("amount", truth=f"{minutes:g}", tol_s=LONG_TOL_MIN), [span_of(talk, found[0])], f"{minutes:g}"))
    def only_that_day(talk: Talk, term: Term) -> bool:
        """Whether `term` is said in no other file of `talk`'s day: an `order` question names the day, and a day may be several files."""
        return not any(t is not talk and t.start.date() == talk.start.date() and mentions(t, term.search) for t in talks)

    pairs = []
    for talk in talks:
        here = [(t, mentions(talk, t.search)[0]) for t in unused(pool, used) if len(mentions(talk, t.search)) == 1 and only_that_day(talk, t)]
        pairs += [(talk, a, b) for a in here for b in here if a[0].search < b[0].search and abs(a[1].t0 - b[1].t0) >= MIN_APART_S]
    rng.shuffle(pairs)
    for talk, (a, cue_a), (b, cue_b) in pairs:
        if quota.get("order", 0) <= sum(q.tags["template"] == "order" for q in out):
            break
        if a.search in used or b.search in used:
            continue
        used.update((a.search, b.search))
        first, second = "before", "after"
        truth = first if cue_a.t0 < cue_b.t0 else second
        wrong = second if truth == first else first
        out.append(ask(f"order-{key(a)}-{key(b)}", f"In the {say.thing} of {date_of(talk)}, did {say.person} bring up {a.said} before or after {b.said}?",
                       "order", BEFORE_AFTER, realdump.grade("text", truth=truth, wrong=(wrong,)), [span_of(talk, cue_a), span_of(talk, cue_b)], truth))
    missing = list(never)
    rng.shuffle(missing)
    words = {"absent_first": ("When did {p} first mention {s}?", WHEN), "absent_did": ("Did {p} ever mention {s}? If so, when was the first time?", WHEN),
             "absent_which": ("In which {t} did {p} talk about {s}?", FILE)}
    for template, (text, shape) in words.items():
        for term in missing[:quota.get(template, 0)]:
            out.append(ask(f"{template.replace('_', '-')}-{key(term)}", text.format(p=say.person, s=term.said, t=say.thing), template, shape,
                           realdump.grade("absent"), [], None))
        missing = missing[quota.get(template, 0):]
    for c in curated:
        talk = talks[c.talk]
        out.append(ask(f"{c.template}-{c.key}", c.text, c.template, c.shape, c.grade, [Span(talk.path, c.t0, c.t1)], c.value))
    return realdump.split(out, seed)
