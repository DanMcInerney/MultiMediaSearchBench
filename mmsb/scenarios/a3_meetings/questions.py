"""The questions of A3, the research group's meetings, every answer from the ICSI transcripts and topic annotation (source.py), each asked the
morning after the last meeting. The meetings are weekly and on different days, so a meeting is told by its date ("the meeting on Thursday 26 October
2000"); the group are real people and no question names one or asks who: a speaker is "someone" and a word is a word of the transcript.

The speech templates are `realdump/speech.py`'s (`first`, `did`, `which`, `all`, `count`, `long`, `order` and the absent twins of `first`, `did` and
`which`), over TERMS and NEVER; the rest are this scenario's:

| template      | the person says                                                          | category | cue    | rule                           |
| `what`        | (written) In the meeting on Thursday 26 October 2000, about how many     | what     | speech | `text`                         |
|               | overlapping stretches of speech had been counted in one meeting?          |          |        |                                |
| `before`      | (written) In the meeting on Wednesday 8 March 2000, what came up right   | order    | speech | `text`                         |
|               | after they finished with transcription?                                   |          |        |                                |
| `digits`      | In the meeting on Thursday 26 October 2000, when did they first read     | when     | speech | `time` with the date, 60 s     |
|               | out their digit strings?                                                  |          |        |                                |
| `digits_long` | ... how long, in minutes, did the reading of the digits take?             | how_many | speech | `amount`, a minute             |

The digits are the group's own ritual, each person reading out a string of digits, and the transcribers marked each one (`Digits`). `digits` asks for the
first marked segment, `digits_long` for the minutes from the first to the last; a meeting with two digit tasks (Bmr001 to Bmr003 read them at the start and
at the end) is not asked. A `before` question is written by reading the transcript and rests on the topic annotation: its two topics are consecutive in the
meeting's running order, the second beginning where the question says.

What is left out, and why:
- Who, by anyone: the speakers are real, named people (no `who` template; a speaker's code is read only to keep each one's words apart).
- Hot spots: the annotation is the preliminary one of 20 meetings and of the ten here covers seven; a question on one would be answerable only where it
  exists.
- A topic whose name holds a person's name (the coders wrote "<name>'s work on overlaps"): no question names it.

Every term is asked of only where the transcript read three ways (the cues in order, each speaker's own words, each with the transcribers' guesses
at unclear words) finds the same mentions; and a term never said is absent from the words and from the letters of them."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from mmsb.core.question import Grade, Question, Span
from mmsb.core.taxonomy import with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.a3_meetings.source import Meeting
from mmsb.scenarios.realdump import speech
from mmsb.scenarios.realdump.speech import Curated, Say, Talk, Term, matches, mentions, tokens

CODE = "A3"
SOURCE_NAME = "recorder"
SAY = Say(CODE, SOURCE_NAME, "audio", "someone", "meeting", "meetings")
TOL_S = speech.TOL_S
DIGITS_GAP_S = 150.0  # the digit strings of one task are no further apart than this
LONG_TOL_MIN = 1.0
TOPIC_SLACK_S = 20.0  # a topic begins this near a question's moment
FILING = {**speech.FILING, **with_absent({"digits": ("when", "speech"), "digits_long": ("how_many", "speech")})}
QUOTAS = {"tiny": {"first": 1, "which": 1, "absent_did": 1, "digits": 1, "what": 1},
          "standard": {"first": 3, "did": 2, "which": 2, "all": 2, "count": 1, "long": 1, "order": 1, "absent_first": 2, "absent_did": 2, "absent_which": 1,
                       "digits": 1, "digits_long": 1, "what": 3, "before": 2}}

TERMS = tuple(Term(search, said) for search, said in (
    ("cabinet", "a cabinet"), ("cables", "the cables"), ("pedals", "foot pedals"), ("gigabytes", "gigabytes"), ("laptop", "a laptop"),
    ("stereo", "stereo"), ("mixer", "a mixer"), ("radio", "the radio"), ("campus", "the campus"), ("chocolate", "chocolate"), ("snack", "snacks"),
    ("midland", "the Midland"), ("accent", "accents"), ("midwest", "the Midwest"), ("boston", "Boston"), ("french", "French"), ("praat", "Praat"),
    ("markov", "a Markov model"), ("gaussian", "Gaussians"), ("eurospeech", "Eurospeech"), ("blindfolds", "blindfolds"),
    ("reverberation", "reverberation"), ("echo cancellation", "echo cancellation"), ("wireless", "wireless"), ("lapel", "lapel mikes"),
    ("air conditioning", "the air conditioning"), ("forced alignment", "forced alignment")))
NEVER = tuple(Term(search, said) for search, said in (
    ("pizza", "pizza"), ("cookies", "cookies"), ("matlab", "Matlab"), ("unix", "Unix"), ("earthquake", "an earthquake"), ("weather", "the weather"),
    ("vacation", "a vacation"), ("refrigerator", "the refrigerator")))


@dataclass(frozen=True)
class Written:
    """A question written by reading the transcript: of `meeting`, from `t0` to `t1` seconds in, resting on the phrase `check` being said there."""

    key: str
    template: str
    meeting: str
    text: str  # with `{m}` for the meeting
    shape: str
    grade: Grade
    t0: float
    t1: float
    check: str
    value: str
    topics: tuple[str, str] = ()  # a `before` question's two consecutive topics of the running order (the first, and the one that came up)


WRITTEN = (
    Written("overlaps", "what", "Bmr005", "In {m}, about how many stretches of overlapping speech did they say had been counted in the one transcribed meeting?",
            "<a number>", realdump.grade("text", truth="300|3 hundred"), 1290, 1296, "three hundred", "300"),
    Written("spend", "what", "Bmr001", "In {m}, how much did they say they could spend without worrying about it?", "<an amount of money>",
            realdump.grade("text", truth="thousand|1000|1 000"), 376, 380, "thousand dollars", "a thousand dollars"),
    Written("tool", "what", "Bmr010", "In {m}, what was the speech tool called that someone said sounds like the Dutch word for speech?", "<a name>",
            realdump.grade("text", truth="praat"), 603, 608, "praat", "Praat"),
    Written("joke", "what", "Bmr007", "In {m}, what did someone joke they could put on everybody instead of building barriers between them?", "<a thing>",
            realdump.grade("text", truth="blindfold|blindfolds"), 1804, 1808, "blindfolds", "blindfolds"),
    Written("after-transcription", "before", "Bmr003", "In {m}, what new subject came up right after they had finished with transcription?", "<a subject>",
            realdump.grade("text", truth="web site|website|web page|web|internet site", wrong=("electronics", "cabinet")), 3171, 3176, "web site", "a web site", ("transcription", "website")),
    Written("after-segmentation", "before", "Bmr010", "In {m}, what did they turn to right after the talk about segmenting the speech?", "<a subject>",
            realdump.grade("text", truth="microphone|microphones|mikes|mike", wrong=("transcription", "transcribers")), 1184, 1188, "microphone issues", "the microphones",
            ("speech, non-speech segmentation", "microphone issues")),
)


def which_meeting(meeting: Meeting) -> str:
    return f"the meeting on {realdump.date_text(meeting.start.date())}"


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def own_counts(meeting: Meeting, search: str, loose: bool) -> int:
    """The mentions of `search` in each speaker's own words, with the transcribers' guesses or without."""
    wanted, by = tokens(search), {}
    for seg in meeting.segments:
        by.setdefault(seg.who, []).extend(tokens(seg.loose if loose else seg.text))
    return sum(matches(toks, wanted, i) for toks in by.values() for i in range(len(toks)))


def check_terms(meetings: Sequence[Meeting], talks: Sequence[Talk]) -> None:
    """Refuses a term whose mentions differ between the cues in order, each speaker's own words and those with the guesses; and a never-said term that
    is said, as a word, in a guess, or as letters in a word."""
    for m, t in zip(meetings, talks, strict=True):
        for term in TERMS:
            counts = {len(mentions(t, term.search)), own_counts(m, term.search, False), own_counts(m, term.search, True)}
            if len(counts) > 1:
                raise ValueError(f"{m.id}: {term.search!r} is read differently in the cues, the speakers' words and the guesses")
    text = [compact(" ".join(s.loose for s in m.segments)) for m in meetings]
    for term in NEVER:
        if any(own_counts(m, term.search, True) for m in meetings) or any(compact(term.search) in x for x in text):
            raise ValueError(f"{term.search!r} is said, so it is not a word never said")


def check_topics(meeting: Meeting, w: Written) -> None:
    """Refuses a written `before` question whose topics are not consecutive in the meeting's running order, or whose second does not begin where the question
    says (within TOPIC_SLACK_S)."""
    top = [t for t in meeting.topics if t.depth == 0]
    names = [t.name for t in top]
    if w.topics[0] not in names or names.index(w.topics[0]) + 1 >= len(top) or top[names.index(w.topics[0]) + 1].name != w.topics[1] or not (
            w.t0 - TOPIC_SLACK_S <= top[names.index(w.topics[0]) + 1].t0 <= w.t1 + TOPIC_SLACK_S):
        raise ValueError(f"{w.key}: {w.topics} are not consecutive topics of {meeting.id} beginning at {w.t0} to {w.t1}")


def curated_of(meetings: Sequence[Meeting], talks: Sequence[Talk]) -> list[Curated]:
    index = {m.id: i for i, m in enumerate(meetings)}
    for w in WRITTEN:
        if w.meeting in index and w.topics:
            check_topics(meetings[index[w.meeting]], w)
    return [Curated(w.key, w.text.format(m=which_meeting(meetings[index[w.meeting]])), w.template, w.shape, w.grade, index[w.meeting], w.t0, w.t1, w.check,
                    w.value) for w in WRITTEN if w.meeting in index]


def digit_strings(meeting: Meeting) -> list[tuple[float, float]]:
    """The (start, end) of each segment the transcribers marked `Digits`, in order, when they all lie in one stretch (no two further than
    DIGITS_GAP_S apart): `[]` for a meeting that reads digits at the start and at the end (Bmr001 to Bmr003), or not at all."""
    found = [(s.t0, s.t1) for s in meeting.segments if s.digits]
    if not found or any(b[0] - a[1] > DIGITS_GAP_S for a, b in zip(found, found[1:], strict=False)):
        return []
    return found


def questions_of(meetings: Sequence[Meeting], talks: Sequence[Talk], scale: str, seed: int, t_of: Callable[[datetime], float]) -> tuple[Question, ...]:
    """The questions of the meetings (in the order of their starts) with their transcripts as `talks`, at `scale`, drawn with `seed`."""
    check_terms(meetings, talks)
    quota, rng = QUOTAS[scale], random.Random(f"a3:{seed}")
    chosen = []
    for template in ("what", "before"):
        pool = [c for c in curated_of(meetings, talks) if c.template == template]
        chosen += rng.sample(pool, min(quota.get(template, 0), len(pool)))
    given = " ".join(" ".join((c.check, c.value, c.text)) for c in chosen).casefold()  # what the written questions say or answer: no other question asks of it
    terms = tuple(t for t in TERMS if t.search not in given)
    speech_quota = {k: v for k, v in quota.items() if k in ("first", "did", "which", "all", "count", "long", "order", "absent_first", "absent_did", "absent_which")}
    out = list(speech.questions_of(talks, terms, NEVER, SAY, speech_quota, seed, t_of, chosen))
    last = talks[-1]
    now = t_of(datetime.combine(last.start.date() + timedelta(days=1), datetime.min.time().replace(hour=9)))
    since, until = t_of(talks[0].start), t_of(last.start) + last.length

    def ask(qid: str, text: str, template: str, shape: str, grade: Grade, evidence: Sequence[Span], value: str) -> Question:
        return realdump.question(code=CODE, qid=qid, now=now, text=text, kind=template, template=template, modality="audio", shape=shape, grade=grade,
                                 evidence=evidence, sources=(SOURCE_NAME,), since=since, until=until, value=value)

    ready = [(m, t, digit_strings(m)) for m, t in zip(meetings, talks, strict=True) if digit_strings(m)]
    rng.shuffle(ready)
    for m, t, strings in ready[:quota.get("digits", 0)]:
        wall = t.start + timedelta(seconds=strings[0][0])
        out.append(ask(f"digits-{m.id}", f"In {which_meeting(m)}, when did they first read out their digit strings?", "digits", speech.WHEN,
                       realdump.grade("time", time=wall.strftime("%H:%M:%S"), tol_s=TOL_S, date=wall.date().isoformat()), [Span(t.path, *strings[0])],
                       f"{wall.date().isoformat()} {wall.strftime('%H:%M:%S')}"))
    for m, t, strings in ready[quota.get("digits", 0):][:quota.get("digits_long", 0)]:
        minutes = round((strings[-1][1] - strings[0][0]) / 60, 1)
        out.append(ask(f"digits_long-{m.id}", f"In {which_meeting(m)}, how long, in minutes, did the reading out of the digit strings take?", "digits_long",
                       speech.MINUTES, realdump.grade("amount", truth=f"{minutes:g}", tol_s=LONG_TOL_MIN), [Span(t.path, strings[0][0], strings[-1][1])],
                       f"{minutes:g}"))
    return realdump.split(out, seed)
