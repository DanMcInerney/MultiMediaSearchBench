"""Grounded scoring beside the old verdict (DESIGN 6.3): whether the evidence an agent cited is where the question's truth is.

`passed` is still the value rule and no problem. The grounded status is told apart from it:

| the answer                                                       | status         |
| the `line` format, or a result made before the status existed    | `not_recorded` |
| the value fails its rule                                         | `wrong`        |
| an absent question, the value passes, no evidence                | `grounded`     |
| an absent question, the value passes, evidence given             | `wrong`        |
| the value passes, and it cites, and every citation hits a truth  | `grounded`     |
| the value passes, and it cites nothing or a citation hits none   | `lucky`        |

A citation is placed in time (`resolve`): a file with seconds into it, a file with a clock time, a file or a source with ISO times are
one absolute time. It hits a truth moment by the truth file's kind (`hits`): a video within 10 s and a sound within 5 s on the same source; a
line of a log or a transcript within 5 s on the same source; a picture, a note or a version of a config only by being that file. A
citation the dump cannot place (a file it does not have) hits nothing and is counted as `unresolved`. For a count, `event_f1` also
scores the cited moments as events, by distinct moments, with a collar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import Literal

from mmmb.core.answer import AnswerFormat, Citation, FinalAnswer, clock_seconds, is_iso, iso_epoch
from mmmb.core.contract import Dump, ManifestEntry
from mmmb.core.question import Question

Status = Literal["grounded", "lucky", "wrong", "not_recorded"]
NOT_RECORDED: Status = "not_recorded"
TOLERANCE_S = {"video": 10.0, "audio": 5.0, "records": 5.0}  # DESIGN 6.3; the last is the correction for a log or transcript line (C8)
SAME_FILE = ("image", "text")  # a picture, a note, a version of a config: the file itself (DESIGN 6.3; C8)
POINT_S = 1.0  # a source's citation names the picture or note whose time it holds, told to the second
CLOCK_SLACK_S = 1.0  # a clock time is told to the second: one up to a second before a file starts is still that file's
EPS = 1e-6  # seconds: a gap that is the tolerance, as floats, is within it
COLLARS = (("f1_1s", 1.0), ("f1_5s", 5.0))  # the collars a count's event F1 is told with (DESIGN 6.3)


@dataclass(frozen=True)
class Placed:
    """A moment in absolute time: a citation after `resolve`, or a truth span."""

    source: str
    start: float  # epoch seconds
    end: float
    file: str | None  # the file it names, when it names one


@dataclass(frozen=True)
class Grounding:
    status: Status
    cited: int  # citations given
    matched: int  # of those, the ones that hit a truth moment
    unresolved: int  # of those, the ones the dump could not place (they hit nothing)


def status_of(result: dict) -> Status:
    """The grounded status a result holds, `not_recorded` when it holds none: a result of the `line` format, or made before the status."""
    return result.get("grounded", NOT_RECORDED)


# -- placing a moment in time ----------------------------------------------------------------------------------------------------

def entry_of(name: str, dump: Dump) -> ManifestEntry | None:
    """The file of the dump that `name` stands for: its path, or a path that ends with it after a folder (the agent read it in a
    folder that the manifest's paths are inside); None when it is none, or ends with the paths of more than one."""
    path = re.sub(r"[\\/]+", "/", name.strip())
    if exact := next((e for e in dump.entries if e.path == path), None):
        return exact
    ends = [e for e in dump.entries if path.endswith("/" + e.path)]
    return ends[0] if len(ends) == 1 else None


def at_clock(clock: str, not_before: float) -> float:
    """The first moment at or after `not_before` (epoch seconds) that the local wall clock reads `clock`: on the day of `not_before`,
    else the next day (a clock time before midnight and one after it are a night apart)."""
    day = datetime.fromtimestamp(not_before).date()
    wall = datetime.combine(day, time()) + timedelta(seconds=clock_seconds(clock))
    return wall.timestamp() if wall.timestamp() >= not_before - CLOCK_SLACK_S else (wall + timedelta(days=1)).timestamp()


def resolve(c: Citation, dump: Dump) -> Placed | None:
    """The moment `c` cites, in absolute time; None when the dump has no such file or source."""
    if c.source is not None:
        if not any(e.source == c.source for e in dump.entries):
            return None
        return Placed(c.source, iso_epoch(c.t0), iso_epoch(c.t1), None)
    entry = entry_of(c.file, dump)
    if entry is None:
        return None
    if isinstance(c.t0, float):
        return Placed(entry.source, entry.t + c.t0, entry.t + c.t1, entry.path)
    if is_iso(c.t0):
        return Placed(entry.source, iso_epoch(c.t0), iso_epoch(c.t1), entry.path)
    start = at_clock(c.t0, entry.t)
    return Placed(entry.source, start, max(start, at_clock(c.t1, start)), entry.path)


def moments_of(q: Question, dump: Dump) -> list[tuple[ManifestEntry, Placed]]:
    """The question's truth spans in absolute time, each with its file; a span in a file the dump lacks is an error."""
    out = []
    for span in q.truth.evidence:
        entry = next((e for e in dump.entries if e.path == span.file), None)
        if entry is None:
            raise ValueError(f"{q.qid}: its evidence is in {span.file}, which the dump does not hold")
        out.append((entry, Placed(entry.source, entry.t + span.t0, entry.t + span.t1, entry.path)))
    return out


def check_groundable(questions: Sequence[Question], dump: Dump) -> None:
    """Raises, naming the question, when a question that is not absent has no evidence to ground a citation in, or its evidence is in a file the
    dump lacks. `ground` raises the same after an exec, which has spent the plan, so a run in the `json` format refuses it before any agent starts."""
    for q in questions:
        if q.kind != "absent" and not moments_of(q, dump):
            raise ValueError(no_evidence(q))


def no_evidence(q: Question) -> str:
    return f"{q.qid} is not an absent question and has no evidence, so no citation can ground it"


# -- scoring ---------------------------------------------------------------------------------------------------------------------

def gap(a: Placed, b: Placed) -> float:
    """Seconds between two moments: 0 when they overlap."""
    return max(0.0, max(a.start, b.start) - min(a.end, b.end))


def hits(c: Placed, entry: ManifestEntry, truth: Placed) -> bool:
    """Whether the cited moment `c` hits the truth moment `truth`, which is in `entry` (see the module's text)."""
    if entry.media in SAME_FILE:
        if c.file is not None:
            return c.file == entry.path
        return c.source == entry.source and c.start - POINT_S - EPS <= entry.t <= c.end + POINT_S + EPS
    if entry.media not in TOLERANCE_S:
        raise ValueError(f"{entry.path} is {entry.media!r}, which has no tolerance for a citation of it (grounded.TOLERANCE_S)")
    return c.source == truth.source and gap(c, truth) <= TOLERANCE_S[entry.media] + EPS


def ground(q: Question, final: FinalAnswer, value_passed: bool, dump: Dump) -> Grounding:
    """Whether `final`'s evidence grounds its answer to `q`; `value_passed` is whether the answer passed its rule with no problem."""
    placed = [resolve(c, dump) for c in final.evidence]
    truth = moments_of(q, dump)
    matched = sum(1 for p in placed if p is not None and any(hits(p, entry, moment) for entry, moment in truth))
    cited, unresolved = len(placed), placed.count(None)
    if not value_passed:
        status: Status = "wrong"
    elif q.kind == "absent":
        status = "grounded" if cited == 0 else "wrong"
    elif not truth:
        raise ValueError(no_evidence(q))
    else:
        status = "grounded" if cited and matched == cited else "lucky"
    return Grounding(status, cited, matched, unresolved)


def near(c: Placed, truth: Placed, collar_s: float) -> bool:
    return c.source == truth.source and gap(c, truth) <= collar_s + EPS


def event_f1(cited: list[Placed | None], truth: list[Placed], collar_s: float) -> tuple[float, float, float]:
    """Precision, recall and F1 of the cited moments against the truth moments as events, by distinct moments (DCASE's event-based
    score with a collar of `collar_s` seconds in place of its 200 ms): a cited moment is within the collar of a truth moment of its
    source, a citation is matched to one truth moment and a truth moment to one citation (a maximum matching, so one wide citation
    finds one moment), a further citation of a moment already found counts once, not as a false alarm, and the ones that find nothing are
    false alarms, those within the collar of each other being one. A citation that is `None` (not placed) is a false alarm. With nothing
    to find and nothing cited all three are 1."""
    if not truth and not cited:
        return 1.0, 1.0, 1.0
    placed = [c for c in cited if c is not None]
    close = [[j for j, t in enumerate(truth) if near(c, t, collar_s)] for c in placed]  # the truth moments each citation is within the collar of
    owner: dict[int, int] = {}  # truth moment -> the citation matched to it

    def match(i: int, seen: set[int]) -> bool:
        for j in close[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or match(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    found = sum(match(i, set()) for i in range(len(placed)))
    stray = sorted((c for c, js in zip(placed, close) if not js), key=lambda c: (c.source, c.start))
    distinct, reach = 0, {}
    for c in stray:
        if c.source in reach and c.start - reach[c.source] <= collar_s + EPS:
            reach[c.source] = max(reach[c.source], c.end)
        else:
            distinct, reach[c.source] = distinct + 1, c.end
    false_alarms = distinct + len(cited) - len(placed)
    precision = found / (found + false_alarms) if found + false_alarms else 0.0
    recall = found / len(truth) if truth else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def result_fields(q: Question, final: FinalAnswer, fmt: AnswerFormat, value_passed: bool, dump: Dump) -> dict:
    """What a result holds of the answer's format and grounding: `answer_format` always; for `json` also the `evidence` as cited,
    `grounded`, `evidence_unresolved`, the reason when the message was unparsed (`answer_error`), and for a count the event F1 with a
    collar of 1 s and of 5 s. A `line` result holds nothing more, and reads as `not_recorded`."""
    if fmt == "line":
        return {"answer_format": "line"}
    got = ground(q, final, value_passed, dump)
    out = {"answer_format": "json", "evidence": [c.to_json() for c in final.evidence], "grounded": got.status,
           "evidence_unresolved": got.unresolved, **({"answer_error": final.error} if final.error else {})}
    if q.kind == "count":
        placed = [resolve(c, dump) for c in final.evidence]
        truth = [moment for _, moment in moments_of(q, dump)]
        out.update({name: event_f1(placed, truth, collar)[2] for name, collar in COLLARS})
    return out
