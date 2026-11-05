"""The questions of a V7 week as the question file has them (DESIGN 6.1, 5.2): a projection of the plan's question dicts, whose keys do
not change (`plan_hash` covers them), onto `Question`. What the plan says in absolute time is written as seconds into the file that
holds it, found in the manifest.

Evidence, `truth.evidence` and `truth.wrong`, is one `Span` for each moment of the plan's `truth.moments` and `truth.others`: the file
of the question's source that holds the moment, and the moment as seconds after that file's `t`. A photo or a note is at its `t`
(offset 0); a log's `t` is its first line's, so a line is its seconds after the first; a version of app.toml is current until the next,
so the time it was asked about is its seconds after the version's `t`. A moment in no file is an error: a question the data cannot
answer is not written."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, time, timedelta

from mmmb.core.contract import ManifestEntry
from mmmb.core.question import Grade, Question, RetrievalSpec, Scope, Span, Truth
from mmmb.scenarios.v7_screen.manifest import log_path, log_times
from mmmb.scenarios.v7_screen.plan import BASE_OF, TOL_S, Plan

CODE = "V7"
DEV_SEEDS = range(1, 11)  # the questions of these seeds are the dev split (DESIGN 6.4), the others the test split
FILE_KIND = {"note-code": "note", "config-asof": "config", "log-count": "log", "log-last": "log", "photo-sign": "photo",
             "photo-count": "photo", "screen-error": "screen", "porch-parcel": "porch", "mic-doorbell": "mic", "mic-alarm": "mic"}
LOG_OF = {"log-count": "payments", "log-last": "web"}  # the service whose log holds a log template's lines
LAST_LIMIT, COUNT_LIMIT, ORACLE_LIMIT = 5, 50, 20  # sightings a retrieval run keeps: a `last` question's, a count's, the others'
EPS = 1e-6  # seconds: a moment is in a file that it touches


def questions_of(plan: Plan, entries: Sequence[ManifestEntry]) -> tuple[Question, ...]:
    """A `Question` for each question of the plan, in its order, with its evidence found in `entries` (`manifest.entries_of`)."""
    now = datetime.combine(plan.days[-1] + timedelta(days=1), time(12)).astimezone().isoformat(timespec="seconds")
    extents = extents_of(plan, entries)
    return tuple(question_of(q, plan, now, extents) for q in plan.questions)


def extents_of(plan: Plan, entries: Sequence[ManifestEntry]) -> list[tuple[ManifestEntry, float, float]]:
    """Each entry that can hold evidence with the time it shows, `(entry, from, until)` in epoch seconds: a recording its length, a
    note or a photo its instant, a version of app.toml until the next one, a log its first line to its last."""
    spec = {f.path: f.spec for f in plan.files}
    versions = sorted(e.t for e in entries if e.kind == "config")
    out = []
    for e in entries:
        if e.kind in ("screen", "porch", "mic"):
            out.append((e, e.t, e.t + spec[e.path]["length_s"]))
        elif e.kind in ("photo", "note"):
            out.append((e, e.t, e.t))
        elif e.kind == "config":
            later = [t for t in versions if t > e.t]
            out.append((e, e.t, later[0] if later else math.inf))
        elif e.kind == "log":
            service = next(s for s in plan.logs if log_path(s) == e.path)
            out.append((e, *log_times(plan, service)))
    return out


def span_of(q: dict, moment: Sequence[float], extents: list[tuple[ManifestEntry, float, float]]) -> Span:
    """The moment `[t0, t1]` (epoch seconds) of the plan's question `q` as a span of the file that holds it."""
    base = BASE_OF.get(q["template"], q["template"])
    kind, service = FILE_KIND[base], LOG_OF.get(base)
    a, b = moment
    holders = [(e, lo) for e, lo, hi in extents
               if e.kind == kind and e.source == q["source"] and (service is None or e.path == log_path(service))
               and lo - EPS <= a and b <= hi + EPS]
    if not holders:
        raise ValueError(f"{q['qid']}: the moment {a}..{b} is in no {kind} file of the {q['source']} source")
    entry = max(holders, key=lambda h: h[1])[0]  # a moment on the cut between two segments is in the later one
    return Span(entry.path, a - entry.t, b - entry.t)


def question_of(q: dict, plan: Plan, now: str, extents: list[tuple[ManifestEntry, float, float]]) -> Question:
    truth = q["truth"]
    evidence = tuple(span_of(q, m, extents) for m in truth["moments"])
    wrong = tuple(span_of(q, m, extents) for m in truth["others"])
    since, until = q["since"], q["until"]
    tags = {"modality": q["type"], "template": q["template"], "level": q["level"], "last": q["last"], "absent": q["absent"],
            "lookback_h": lookback_h(truth["moments"], now), "certificate_s": certificate_s(truth["moments"]), "rung": plan.scale,
            "first_day": plan.days[0].isoformat(), "last_day": plan.days[-1].isoformat()}
    return Question(
        qid=q["qid"], scenario=CODE, split="dev" if plan.seed in DEV_SEEDS else "test", now=now, question=q["question"], attachments=(),
        kind=kind_of(q), hint=q["level"], truth=Truth(truth["value"], evidence, wrong), grade=grade_for(q), tags=tags, options=None,
        scope=Scope((q["source"],), since, until), group=BASE_OF.get(q["template"], q["template"]), horizon=None,
        cost_ranges=((since, until),), retrieval=retrieval_of(q))


def grade_for(q: dict) -> Grade:
    """The plan question's `grade` dict as a `Grade`: its `kind` is the `rule`, its `tol_min` minutes are `tol_s` seconds."""
    g = q["grade"]
    return Grade(g["kind"], g["truth"], g["time"], None if g["tol_min"] is None else g["tol_min"] * 60.0, tuple(g["wrong"]),
                 tuple(g["ignore"]), g["date"])


def kind_of(q: dict) -> str:
    """`absent`, `last`, `count`, `when` (a time is asked for) or `find`."""
    if q["absent"]:
        return "absent"
    if q["last"]:
        return "last"
    if q["grade"]["kind"] == "count":
        return "count"
    return "when" if q["grade"]["kind"] in ("time", "code_time") else "find"


def lookback_h(moments: Sequence[Sequence[float]], now: str) -> float | None:
    """Hours from the end of the latest moment of the truth to `now`; None when there is none (an absent question)."""
    if not moments:
        return None
    return round((datetime.fromisoformat(now).timestamp() - max(b for _, b in moments)) / 3600, 1)


def certificate_s(moments: Sequence[Sequence[float]]) -> float | None:
    """Seconds of recording that show the answer: the lengths of the moments of the truth added up (a photo or a note is an instant)."""
    return round(sum(b - a for a, b in moments), 3) if moments else None


def retrieval_of(q: dict) -> RetrievalSpec | None:
    """The retrieval run's question (`oracle`): the needle's yes/no over its source and range, for as many sightings as the question
    can have (the newest few of a `last` question, up to 50 for a count) and oldest first when the fake's question is."""
    if q["oracle"] is None:
        return None
    counting = q["template"] == "photo-count" or q["fake"]["check"] == "count"
    k = LAST_LIMIT if q["last"] else COUNT_LIMIT if counting else ORACLE_LIMIT
    truth = q["truth"]
    return RetrievalSpec(q["oracle"], (q["source"],), ((q["since"], q["until"]),), k, q["fake"]["oldest_first"] and not q["last"], TOL_S,
                         tuple((a, b) for a, b in truth["moments"]), tuple((a, b) for a, b in truth["others"]))
