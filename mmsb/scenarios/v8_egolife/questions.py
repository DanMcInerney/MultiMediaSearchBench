"""The questions of a V8 run as the question file has them (DESIGN 6.1, 5.2): a projection of the plan's question dicts, whose keys do not
change (`plan_hash` covers them), onto `Question`. What the plan says in absolute time is written as seconds into the file that holds it,
found in the manifest.

Evidence, `truth.evidence`, is one `Span` for each moment of the plan's `truth.moments` (the times of the question's targets, each a point):
for a video question the clip that holds the moment and its seconds after the clip's start (a moment on the cut between two clips is in the
later one); for a speech question, which is answered from the transcript, the day's transcript file and the moment's seconds after the start
of the day. A moment in no file is an error: a question the data cannot answer is not written. V8 has no `wrong` evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from mmsb.core.contract import ManifestEntry
from mmsb.core.question import Grade, Question, RetrievalSpec, Scope, Span, Truth
from mmsb.scenarios.v8_egolife.plan import TEXT, TOL_S, Plan, oracle_ranges

CODE = "V8"
ORACLE_LIMIT = 20  # sightings a retrieval run keeps (the old `bench.ORACLE_LIMIT`)
EPS = 1e-6  # seconds: a moment is in a file that it touches


def questions_of(plan: Plan, entries: Sequence[ManifestEntry]) -> tuple[Question, ...]:
    """A `Question` for each question of the plan, in its order, with its evidence found in `entries` (`manifest.entries_of`)."""
    return tuple(question_of(q, plan, entries) for q in plan.questions)


def span_of(q: dict, moment: Sequence[float], entries: Sequence[ManifestEntry]) -> Span:
    """The moment `[t0, t1]` (epoch seconds) of the plan's question `q` as a span of the file that holds it."""
    kind = "log" if q["source"] == TEXT else "clip"
    a, b = moment
    holders = [e for e in entries if e.kind == kind and e.t - EPS <= a and b <= e.t + e.length + EPS]
    if not holders:
        raise ValueError(f"{q['qid']}: the moment {a}..{b} is in no {kind} file of the {q['source']} source")
    entry = max(holders, key=lambda e: e.t)
    return Span(entry.path, a - entry.t, b - entry.t)


def question_of(q: dict, plan: Plan, entries: Sequence[ManifestEntry]) -> Question:
    truth = q["truth"]
    now = datetime.fromtimestamp(q["query"]).astimezone().isoformat(timespec="milliseconds")
    evidence = tuple(span_of(q, m, entries) for m in truth["moments"])
    tags = {"modality": q["type"], "template": q["template"], "level": q["level"], "last": q["last"], "absent": q["absent"],
            "age": q["level"], "category": q["template"], "lookback_h": lookback_h(truth["moments"], now), "rung": f"{len(plan.days)}d",
            "pilot": q["pilot"], "day": q["day"], "first_day": plan.days[0].isoformat()}
    return Question(
        qid=q["qid"], scenario=CODE, split="dev", now=now, question=q["question"], attachments=(), kind="choice", hint=q["level"],
        truth=Truth(q["grade"]["truth"], evidence, ()), grade=grade_for(q), tags=tags, options=dict(q["options"]),
        scope=Scope((q["source"],), q["since"], q["until"]), group=q["qid"], horizon=q["until"],
        cost_ranges=tuple(oracle_ranges(q)), retrieval=retrieval_of(q))


def grade_for(q: dict) -> Grade:
    """The plan question's `grade` dict as a `Grade`: a choice of a letter, with nothing else to say."""
    g = q["grade"]
    return Grade(g["kind"], g["truth"], None, None, (), tuple(g["ignore"]), None)


def lookback_h(moments: Sequence[Sequence[float]], now: str) -> float | None:
    """Hours from the end of the latest moment of the truth to `now`; None when there is none."""
    if not moments:
        return None
    return round((datetime.fromisoformat(now).timestamp() - max(b for _, b in moments)) / 3600, 1)


def retrieval_of(q: dict) -> RetrievalSpec:
    """The retrieval run's question (`oracle`): the yes/no of the question's subject over the source and the ranges around the targets,
    scored against the targets with TOL_S."""
    return RetrievalSpec(q["oracle"], (q["oracle_source"],), tuple(oracle_ranges(q)), ORACLE_LIMIT, False, TOL_S,
                         tuple((a, b) for a, b in q["truth"]["moments"]), ())
