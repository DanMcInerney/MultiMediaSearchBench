"""The questions of V4, the meetings, every answer from the AMI corpus's words and annotations (source.py), each asked the morning after the last
meeting. A person asks of "anyone" in "the second meeting on Wednesday 16 March 2005": a team's four meetings are on one day, so a meeting is told by
its date and its place in the day (the order of the files' times), and the speakers by their role, "the project manager", which the
corpus gives for each of its speaker letters. All twelve meetings are about one remote control, so a question is asked of a thing that has one answer.

The speech templates are `realdump/speech.py`'s (`first`, `did`, `which`, `all`, `count`, `long`, and their absent twins), over TERMS and NEVER;
the rest are this scenario's:

| template      | the person says                                                         | category | cue    | rule                            |
| `whofirst`    | In the third meeting on Monday 13 December 2004, who was the first to  | who      | speech | `text`, a role                  |
|               | bring up a jog dial?                                                    |          |        |                                 |
| `firstof`     | In the second meeting on Wednesday 16 March 2005, which did anyone      | order    | speech | `text`, one of the two words    |
|               | bring up first: a banana or lunch?                                       |          |        |                                 |
| `next`        | In the second meeting on Wednesday 16 March 2005, who presented         | order    | speech | `text`, a role                  |
|               | straight after the marketing expert?                                     |          |        |                                 |
| `what`        | (written) In the third meeting on Monday 13 December 2004, what shape   | what     | speech | `text`                          |
|               | did they decide the remote should have?                                  |          |        |                                 |
| `who`         | (written) In the first meeting on Wednesday 16 March 2005, who first    | who      | speech | `text`, a role                  |
|               | suggested making the remote ball-shaped?                                 |          |        |                                 |
| `decided`     | (written) In the last meeting on Wednesday 9 March 2005, when did they  | when     | speech | `time` with the date, 60 s      |
|               | agree the remote would have two colours?                                 |          |        |                                 |
| `board_when`  | In the first meeting on Monday 13 December 2004, when did someone first | when     | sight  | `time` with the date, 60 s      |
|               | get up to the whiteboard?                                                |          |        |                                 |
| `board_who`   | ... who was the first to get up to the whiteboard?                       | who      | mixed  | `text`, a role                  |
| `board_count` | ... how many of them got up to the whiteboard?                           | how_many | sight  | `count`                         |
| `absent_board_when` | `board_when` of a meeting in which nobody went to the board       | when     | sight  | `absent`                        |

Every answer is the corpus's: a word is a word of a speaker's complete transcript (a term said by no speaker is certainly absent, and a term is
asked of only where the transcript read two ways, as one stream and as each speaker's own, finds the same mentions: a phrase an interruption
splits is left out); a role is `meetings.xml`'s; a decision's span is the annotators' (`decision`) and a written question rests on the phrase
being in its words; a presentation's place is the topic annotation's; and the whiteboard questions are the `movement` annotation's, in which every
speaker has a movement for every second of the meeting (source.py refuses a gap), so that a meeting in which nobody is marked at the whiteboard, and nobody
is marked walking, is one in which nobody went there.

What is left out, and why:
- How many decisions a meeting made. The decision annotation and the written summary disagree on the count in six of the eight meetings of ES2015
  and ES2016 (one counts the team's own decisions, the other what it was given too), so no number is the answer.
- ES2002's decisions (annotated for two of its four meetings), and the topics of ES2015b and ES2015c (not annotated).
- A decision whose annotated span is longer than a minute: it is a discussion, and no time is its moment.
- The decisions' own words as a kind of question ("what was decided"): the decision annotation gives a span, not a sentence, and the written summary's
  sentences are not always what was said (ES2015d's says the cost is 1.20 euros, the project manager says twelve twenty), so a `decided` or `what`
  question is written from the transcript and rests on a phrase in it.

The `sight` cue is the whiteboard questions' (a person is seen getting up and standing at the board; the Corner camera sees the board: frames at stand times of five of the meetings were looked at);
`board_who` names the person by role, which needs the words too, so it is `mixed`."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from mmsb.core.question import Grade, Question, Span
from mmsb.core.taxonomy import with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech
from mmsb.scenarios.realdump.speech import Cue, Curated, Say, Talk, Term, matches, mentions, tokens
from mmsb.scenarios.v4_meetings.source import ROLES, Meeting

CODE = "V4"
SOURCE_NAME = "meetings"  # the dump's source
SAY = Say(CODE, SOURCE_NAME, "video", "anyone", "meeting", "meetings")
ORDINALS = ("first", "second", "third", "fourth")
TOL_S = speech.TOL_S  # a moment is told to the second, and a person says when to the minute
DECISION_MAX_S = 60.0  # a decision of a longer span is a discussion
BOARD_FIRST_GAP_S = 2 * TOL_S  # the next to go to the board is this much later than the first: no answer to the first's time is right for the next
WALK_MAX_S = 20.0  # the first to go walks to the board no longer than this
STAND_MIN_S = 5.0  # a stand at the board no shorter than this counts
FIRST_OF_GAP_S = 60.0  # the two words of a `firstof` are first said this far apart
WHO_MIN_S = 60.0  # `whofirst`'s word is first said this late in the meeting (not in the introductions)
ROLE_SHAPE = "<a role: project manager, industrial designer, user interface designer or marketing expert>"
ACCEPT = {"PM": "project manager|pm", "ID": "industrial designer|industrial design",
          "UI": "user interface designer|user interface|interface designer|interface specialist|ui", "ME": "marketing expert|marketing|marketer"}
PRESENTERS = {"interface specialist presentation": "UI", "marketing expert presentation": "ME", "industrial designer presentation": "ID",
              "industrial design presentation": "ID"}
FILING = {**speech.FILING, **with_absent({"decided": ("when", "speech"), "whofirst": ("who", "speech"), "firstof": ("order", "speech"),
                                          "next": ("order", "speech"), "board_when": ("when", "sight"), "board_who": ("who", "mixed"),
                                          "board_count": ("how_many", "sight")})}
QUOTAS = {"tiny": {"first": 1, "which": 1, "absent_did": 1, "whofirst": 1, "board_when": 1, "who": 1},
          "standard": {"first": 2, "did": 1, "which": 1, "all": 1, "count": 1, "long": 1, "absent_first": 2, "absent_did": 1, "absent_which": 1,
                       "whofirst": 1, "firstof": 1, "next": 1, "board_when": 1, "board_who": 1, "board_count": 1, "absent_board_when": 1,
                       "what": 3, "who": 1, "decided": 2}}

# what a person would remember of these meetings, each said by the speakers (`search`) and as it is asked ("a banana")
TERMS = tuple(Term(search, said) for search, said in (
    ("teletext", "teletext"), ("titanium", "titanium"), ("banana", "a banana"), ("mango", "a mango"), ("strawberry", "a strawberry"),
    ("grapefruit", "a grapefruit"), ("lemon", "a lemon"), ("vanilla", "vanilla"), ("ninja", "a ninja"), ("snowman", "a snowman"),
    ("butterfly", "a butterfly"), ("beagle", "a beagle"), ("dynamo", "a dynamo"), ("hinge", "a hinge"), ("fluorescent", "something fluorescent"),
    ("ipod", "an iPod"), ("lunch", "lunch"), ("coffee", "coffee"), ("centimetres", "centimetres"), ("double curved", "a double-curved case"),
    ("watermelon", "watermelon"), ("interchangeable", "interchangeable covers"), ("leadership", "leadership"), ("dog", "a dog"), ("cat", "a cat")))
# said by no speaker of the twelve meetings, and plausible in them: animals for the drawing exercise, a colour, a fruit, a radio
NEVER = tuple(Term(search, said) for search, said in (
    ("elephant", "an elephant"), ("giraffe", "a giraffe"), ("penguin", "a penguin"), ("dolphin", "a dolphin"), ("kangaroo", "a kangaroo"),
    ("tiger", "a tiger"), ("pink", "pink"), ("silver", "silver"), ("bluetooth", "bluetooth"), ("pineapple", "a pineapple"), ("zebra", "a zebra")))


@dataclass(frozen=True)
class Written:
    """A question written by reading the transcript: of `meeting`, from `t0` to `t1` seconds in (or, for a `decision`, that decision's span),
    resting on the phrase `check` being said there. `text` has `{m}` for the meeting, told as a person would."""

    key: str
    template: str
    meeting: str
    text: str
    shape: str
    grade: Grade | None  # None for a `decision`: its time is the middle of the span
    t0: float
    t1: float
    check: str
    value: str
    decision: str | None = None


def role_grade(role: str, skip: str | None = None) -> Grade:
    """`text`: the role's words, and no other role's (but `skip`'s, which a question may itself name)."""
    wrong = [w for r, a in ACCEPT.items() if r not in (role, skip) for w in a.split("|")]
    return realdump.grade("text", truth=ACCEPT[role], wrong=wrong)


WRITTEN = (
    Written("age-group", "what", "ES2002b", "In {m}, what age group did they decide to aim the remote at?", "<an age range>",
            realdump.grade("text", truth="15 to 35|15 35|15 and 35"), 1252, 1262, "fifteen to thirty five", "15 to 35"),
    Written("shape", "what", "ES2002c", "In {m}, what shape did they decide the remote should have?", "<a shape>",
            realdump.grade("text", truth="snowman"), 1648, 1651, "snowman", "snowman"),
    Written("prototype-fruit", "what", "ES2015d", "In {m}, what fruit was the prototype they showed based on?", "<a fruit>",
            realdump.grade("text", truth="mango", wrong=("banana", "strawberry")), 133, 140, "mango", "mango"),
    Written("prototype-banana", "what", "ES2016d", "In {m}, what was the prototype they showed designed to look like?", "<a fruit>",
            realdump.grade("text", truth="banana", wrong=("mango", "strawberry")), 185, 196, "banana", "banana"),
    Written("profit", "what", "ES2015a", "In {m}, what profit did the project manager say they wanted to make?", "<an amount of money>",
            realdump.grade("text", truth="50 million"), 869, 873, "fifty million", "50 million"),
    Written("cost", "what", "ES2015d", "In {m}, what did the project manager say at the end that each remote would cost to operate, in euros?",
            "<an amount in euros>", realdump.grade("text", truth="12 20|12 2"), 1613, 1618, "twelve twenty", "12.20"),
    Written("scroll", "what", "ES2016c", "In {m}, which music player's scroll wheel did they want theirs to work like?", "<a product>",
            realdump.grade("text", truth="ipod|i pod"), 2124, 2128, "ipod", "iPod"),
    Written("ball", "who", "ES2016a", "In {m}, who first suggested making the remote ball-shaped?", ROLE_SHAPE, role_grade("UI"), 441, 444, "a ball",
            "user interface designer"),
    Written("glow", "who", "ES2015b", "In {m}, who first suggested that the remote could glow in the dark?", ROLE_SHAPE, role_grade("ID"), 200, 207,
            "glow in the dark", "industrial designer"),
    Written("colour", "decided", "ES2016b", "In {m}, when did the project manager settle on yellow for the colour of the remote?", speech.WHEN, None, 0, 0,
            "yellow", "", "6"),
    Written("two-colours", "decided", "ES2015d", "In {m}, when did they agree the remote would have two colours, one for the case and one for the buttons?",
            speech.WHEN, None, 0, 0, "two colours", "", "6"),
    Written("double-curved", "decided", "ES2016d", "In {m}, when did they agree the case would be double-curved?", speech.WHEN, None, 0, 0,
            "double curved", "", "4"),
    Written("plates", "decided", "ES2015c", "In {m}, when did they confirm they were still going for interchangeable plates?", speech.WHEN, None, 0, 0,
            "interchangeable plates", "", "11"),
)


def which_meeting(meeting: Meeting, meetings: Sequence[Meeting]) -> str:
    """`the second meeting on Wednesday 16 March 2005`: by its place among the meetings of its day in the dump, or `the meeting on ...` when it is
    alone that day."""
    same = [m for m in meetings if m.day == meeting.day]
    day = realdump.date_text(meeting.day)
    return f"the meeting on {day}" if len(same) == 1 else f"the {ORDINALS[same.index(meeting)]} meeting on {day}"


def clock_of(talk: Talk, seconds: float) -> tuple[str, str]:
    wall = talk.start + timedelta(seconds=seconds)
    return wall.strftime("%H:%M:%S"), wall.date().isoformat()


def time_grade(talk: Talk, seconds: float) -> Grade:
    clock, day = clock_of(talk, seconds)
    return realdump.grade("time", time=clock, tol_s=TOL_S, date=day)


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def streams(meeting: Meeting) -> list[list[str]]:
    """Each speaker's words as tokens, in the order spoken (an interruption by another speaker is not in it)."""
    return [[t for w in words if not w.punc for t in tokens(w.text)] for words in meeting.words.values()]


def own_count(meeting: Meeting, search: str) -> int:
    wanted = tokens(search)
    return sum(matches(toks, wanted, i) for toks in streams(meeting) for i in range(len(toks)))


def check_terms(meetings: Sequence[Meeting], talks: Sequence[Talk]) -> None:
    """Refuses a term whose mentions differ between the cues in order (what the speech templates read) and each speaker's own words: another
    speaker's cue inside the phrase would hide one, or join two. And refuses a never-said term that is said (as a word, or as letters in a word)."""
    for m, t in zip(meetings, talks, strict=True):
        for term in TERMS:
            if len(mentions(t, term.search)) != own_count(m, term.search):
                raise ValueError(f"{m.id}: {term.search!r} is read differently in the cues and in the speakers' own words")
    text = [compact(" ".join(t for toks in streams(m) for t in toks)) for m in meetings]
    for term in NEVER:
        if any(own_count(m, term.search) for m in meetings) or any(compact(term.search) in x for x in text):
            raise ValueError(f"{term.search!r} is said, so it is not a word never said")


def role_of(cue: Cue) -> str:
    return next(r for r, name in ROLES.items() if name == cue.speaker)


def curated_of(meetings: Sequence[Meeting], talks: Sequence[Talk]) -> list[tuple[Written, Curated]]:
    """The written questions of the meetings there are, each as `speech.Curated` (a decision's time is the middle of its span)."""
    index = {m.id: i for i, m in enumerate(meetings)}
    out = []
    for w in WRITTEN:
        if w.meeting not in index:
            continue
        i = index[w.meeting]
        t0, t1, grade, value = w.t0, w.t1, w.grade, w.value
        if w.decision is not None:
            d = next((d for d in meetings[i].decisions if d.key == w.decision), None)
            if d is None or d.external or d.t1 - d.t0 > DECISION_MAX_S:
                raise ValueError(f"{w.key}: decision {w.decision} of {w.meeting} is missing, was given to the team, or is a discussion")
            t0, t1 = d.t0, d.t1
            grade = time_grade(talks[i], (t0 + t1) / 2)
            value = " ".join(reversed(clock_of(talks[i], (t0 + t1) / 2)))
        out.append((w, Curated(w.key, w.text.format(m=which_meeting(meetings[i], meetings)), w.template, w.shape, grade, i, t0, t1, w.check, value)))
    return out


def asker(code: str, now: float, since: float, until: float) -> Callable[..., Question]:
    def ask(qid: str, text: str, template: str, shape: str, grade: Grade, evidence: Sequence[Span], value: str | None) -> Question:
        return realdump.question(code=code, qid=qid, now=now, text=text, kind="absent" if template.startswith("absent_") else template, template=template,
                                 modality="video", shape=shape, grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since, until=until, value=value)

    return ask


def whofirst(meetings, talks, terms: Sequence[Term], used: set[str]) -> list[tuple[Meeting, Talk, Term, Cue]]:
    """The (meeting, term) pairs whose first mention is by one role: no other speaker says it within 10 s, and not in the first minute."""
    out = []
    for m, t in zip(meetings, talks, strict=True):
        for term in terms:
            found = mentions(t, term.search)
            if not found or "-".join(tokens(term.search)) in used or found[0].t0 < WHO_MIN_S:
                continue
            if all(c.speaker == found[0].speaker or c.t0 - found[0].t1 > 10 for c in found[1:3]):
                out.append((m, t, term, found[0]))
    return out


def firstof(meetings, talks, terms: Sequence[Term], used: set[str]) -> list[tuple[Meeting, Talk, Term, Cue, Term, Cue]]:
    """Two single-word terms first said in one meeting, far apart."""
    out = []
    for m, t in zip(meetings, talks, strict=True):
        first = {term: mentions(t, term.search)[0] for term in terms
                 if len(tokens(term.search)) == 1 and "-".join(tokens(term.search)) not in used and mentions(t, term.search)}
        for a in first:
            for b in first:
                if a.search < b.search and abs(first[a].t0 - first[b].t0) >= FIRST_OF_GAP_S and min(first[a].t0, first[b].t0) >= WHO_MIN_S:
                    out.append((m, t, a, first[a], b, first[b]))
    return out


def presenters(meeting: Meeting) -> list[tuple[str, float, float]]:
    """The roles that presented in the meeting, in order, with the span of their presentation: the top-level topics named for one, each once."""
    found = sorted((t.t0, PRESENTERS[t.name], t.t1) for t in meeting.topics if t.top and t.name in PRESENTERS)
    roles = [r for _, r, _ in found]
    return [(r, t0, t1) for t0, r, t1 in found] if len(set(roles)) == len(roles) else []


def board(meeting: Meeting) -> dict:
    """What the movement annotation says of the whiteboard: the stands (`stand_whiteboard`, by start), and whether anyone walked (`move`)."""
    stands = sorted((m for m in meeting.moves if m.kind == "stand_whiteboard"), key=lambda m: m.t0)
    walked = [m for m in meeting.moves if m.kind in ("move", "stand_screen")]
    return {"stands": stands, "walked": walked}


def board_first(meeting: Meeting):
    """The first to go to the board, when it is clear: a stand, reached by a walk of no more than WALK_MAX_S, that no other speaker's stand starts
    within BOARD_FIRST_GAP_S of, and no stand of anyone's is shorter than STAND_MIN_S."""
    stands = board(meeting)["stands"]
    if not stands or any(s.t1 - s.t0 < STAND_MIN_S for s in stands):
        return None
    first = stands[0]
    walk = [m for m in meeting.moves if m.agent == first.agent and m.kind == "move" and abs(m.t1 - first.t0) < 0.01]
    if (walk and walk[0].t1 - walk[0].t0 > WALK_MAX_S) or any(s.agent != first.agent and s.t0 - first.t0 < BOARD_FIRST_GAP_S for s in stands):
        return None
    return first


def questions_of(meetings: Sequence[Meeting], talks: Sequence[Talk], scale: str, seed: int, t_of: Callable[[datetime], float]) -> tuple[Question, ...]:
    """The questions of the meetings (in the order of their starts) with their transcripts as `talks`, at `scale`, drawn with `seed`."""
    check_terms(meetings, talks)
    quota, rng = QUOTAS[scale], random.Random(f"v4:{seed}")
    written = curated_of(meetings, talks)
    chosen = []
    for template in ("what", "who", "decided"):
        pool = [c for w, c in written if w.template == template]
        chosen += rng.sample(pool, min(quota.get(template, 0), len(pool)))
    given = " ".join(" ".join((c.check, c.value, c.text)) for c in chosen).casefold()  # what the written questions say or answer: no other question asks of it
    terms = tuple(t for t in TERMS if t.search not in given)
    speech_quota = {k: v for k, v in quota.items() if k in ("first", "did", "which", "all", "count", "long", "absent_first", "absent_did", "absent_which")}
    out = list(speech.questions_of(talks, terms, NEVER, SAY, speech_quota, seed, t_of, chosen))
    used = {q.qid.split("-", 1)[1] for q in out}
    last = talks[-1]
    now = t_of(datetime.combine(last.start.date() + timedelta(days=1), datetime.min.time().replace(hour=9)))
    ask = asker(CODE, now, t_of(talks[0].start), t_of(last.start) + last.length)

    def span(talk: Talk, cue: Cue) -> Span:
        return speech.span_of(talk, cue)

    def draw(candidates: list, n: int) -> list:
        """`n` of `candidates` (the meeting first in each), one to a meeting where there are enough of them."""
        pool = list(candidates)
        rng.shuffle(pool)
        firsts = [c for i, c in enumerate(pool) if all(c[0].id != d[0].id for d in pool[:i])]
        return (firsts + [c for c in pool if c not in firsts])[:n]

    for m, t, term, cue in draw(whofirst(meetings, talks, terms, used), quota.get("whofirst", 0)):
        used.add("-".join(tokens(term.search)))
        out.append(ask(f"whofirst-{m.id}-{'-'.join(tokens(term.search))}", f"In {which_meeting(m, meetings)}, who was the first to bring up {term.said}?",
                       "whofirst", ROLE_SHAPE, role_grade(role_of(cue)), [span(t, cue)], ROLES[role_of(cue)]))
    for m, t, a, cue_a, b, cue_b in draw(firstof(meetings, talks, terms, used), quota.get("firstof", 0)):
        used.update(("-".join(tokens(a.search)), "-".join(tokens(b.search))))
        first, second = (a, b) if cue_a.t0 < cue_b.t0 else (b, a)
        shown = (a, b) if rng.random() < 0.5 else (b, a)
        out.append(ask(f"firstof-{m.id}-{tokens(a.search)[0]}-{tokens(b.search)[0]}",
                       f"In {which_meeting(m, meetings)}, which did anyone bring up first: {shown[0].said} or {shown[1].said}?", "firstof",
                       "<one of the two>", realdump.grade("text", truth=f"{first.search}|{first.search}s", wrong=(second.search, second.search + "s")),
                       [span(t, cue_a), span(t, cue_b)], first.search))
    nexts = []
    for m, t in zip(meetings, talks, strict=True):
        order = presenters(m)
        nexts += [(m, t, order[i], order[i + 1]) for i in range(len(order) - 1)]
    for m, t, (before, _, _), (after, t0, t1) in draw(nexts, quota.get("next", 0)):
        out.append(ask(f"next-{m.id}-{before}-{after}", f"In {which_meeting(m, meetings)}, who presented straight after the {ROLES[before]}?", "next",
                       ROLE_SHAPE, role_grade(after, skip=before), [Span(t.path, t0, min(t1, t0 + 30))], ROLES[after]))
    # the whiteboard: a first stand that is clear, how many went, and a meeting in which nobody did
    firsts = [(m, t, f) for m, t in zip(meetings, talks, strict=True) if (f := board_first(m))]
    counted = [(m, t, board(m)["stands"]) for m, t in zip(meetings, talks, strict=True)
               if board(m)["stands"] and all(s.t1 - s.t0 >= STAND_MIN_S for s in board(m)["stands"])]
    nobody = [(m, t) for m, t in zip(meetings, talks, strict=True) if not board(m)["stands"] and not board(m)["walked"]]
    for template, shape in (("board_when", speech.WHEN), ("board_who", ROLE_SHAPE)):
        for m, t, f in draw(firsts, quota.get(template, 0)):
            text = f"In {which_meeting(m, meetings)}, " + ("when did someone first get up to the whiteboard?" if template == "board_when"
                                                         else "who was the first to get up to the whiteboard?")
            clock, day = clock_of(t, f.t0)
            grade = time_grade(t, f.t0) if template == "board_when" else role_grade(m.roles[f.agent])
            value = f"{day} {clock}" if template == "board_when" else ROLES[m.roles[f.agent]]
            out.append(ask(f"{template}-{m.id}", text, template, shape, grade, [Span(t.path, f.t0, f.t1)], value))
    for m, t, stands in draw(counted, quota.get("board_count", 0)):
        n = len({s.agent for s in stands})
        out.append(ask(f"board_count-{m.id}", f"In {which_meeting(m, meetings)}, how many of them got up to the whiteboard?", "board_count", speech.NUMBER,
                       realdump.grade("count", truth=str(n)), [Span(t.path, s.t0, s.t1) for s in stands], str(n)))
    for m, _ in draw(nobody, quota.get("absent_board_when", 0)):
        out.append(ask(f"absent-board-when-{m.id}", f"In {which_meeting(m, meetings)}, when did someone first get up to the whiteboard?", "absent_board_when",
                       speech.WHEN, realdump.grade("absent"), [], None))
    return realdump.split(out, seed)
