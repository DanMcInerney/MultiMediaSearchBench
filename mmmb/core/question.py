"""The question file of a scenario: one JSON object per line, one `Question` each (DESIGN 6.1). `load_questions` is the one place
the file is parsed. It refuses an unknown key, a missing key (but for a field with a default: a line written before the field was
added reads as it did), a value of the wrong type, an unknown rule and a repeated `qid`, so everything after it holds dataclasses and
trusts them."""

from __future__ import annotations

import dataclasses
import functools
import json
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from mmmb.core import fs

RULES = ("number", "count", "time", "code_time", "absent", "choice", "set")  # how an answer is graded (core/grade.py)
ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")  # a `set` of dates holds them so
QID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # a qid is a file name in the run folder
CLOCK = re.compile(r"\d{1,2}:\d{2}(:\d{2})?")


@dataclass(frozen=True)
class Span:
    file: str
    t0: float  # seconds into the file
    t1: float


@dataclass(frozen=True)
class Truth:
    value: str | None
    evidence: tuple[Span, ...]
    wrong: tuple[Span, ...]


@dataclass(frozen=True)
class Grade:
    rule: str  # one of RULES
    truth: str | None
    time: str | None
    tol_s: float | None
    wrong: tuple[str, ...]
    ignore: tuple[str, ...]
    date: str | None
    items: tuple[str, ...] = ()  # the `set` rule's: every one the answer must name, and no other (ISO dates, or file names)


@dataclass(frozen=True)
class Scope:
    sources: tuple[str, ...]
    since: float
    until: float


@dataclass(frozen=True)
class RetrievalSpec:
    query: str
    sources: tuple[str, ...]
    ranges: tuple[tuple[float, float], ...]
    k: int
    oldest_first: bool
    tol_s: float
    moments: tuple[tuple[float, float], ...]
    others: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Question:
    qid: str
    scenario: str
    split: str
    now: str  # ISO with an offset
    question: str
    attachments: tuple[str, ...]
    kind: str
    hint: str
    truth: Truth
    grade: Grade
    tags: Mapping[str, str | int | float | bool | None]
    options: Mapping[str, str] | None  # a multiple-choice question's options, by letter
    scope: Scope  # where the runner's order and a Case come from
    group: str  # two cases of one group never run at once
    horizon: float | None  # None when nothing need be cut
    cost_ranges: tuple[tuple[float, float], ...]  # what Session.estimate is asked about
    retrieval: RetrievalSpec | None


def line_of(q: Question) -> dict:
    """`q` as a line of the question file: its grade's `items` only when it has some, so that a file with no `set` question is the
    file it was before them."""
    out = dataclasses.asdict(q)
    if not out["grade"]["items"]:
        del out["grade"]["items"]
    return out


def write_questions(path: Path, questions: Sequence[Question]) -> None:
    """`questions` as the file `load_questions` reads, whole or not at all."""
    fs.write_text(path, "".join(json.dumps(line_of(q), ensure_ascii=False, allow_nan=False) + "\n" for q in questions))


def load_questions(path: Path) -> tuple[Question, ...]:
    """The questions in `path`, in file order. Raises a ValueError naming the file and line of the first fault."""
    questions: dict[str, Question] = {}
    first_line: dict[str, int] = {}
    lines = Path(path).read_text(encoding="utf-8").split("\n")  # not splitlines: U+2028 and the like are text in a JSON string
    for n, line in enumerate(lines[:-1] if lines[-1] == "" else lines, 1):
        try:
            try:
                record = json.loads(line, parse_constant=refuse_constant)
            except ValueError as error:
                raise ValueError(f"not JSON: {error}") from None
            q = read(Question, record, "")
            check(q)
            if q.qid in questions:
                raise ValueError(f"qid {q.qid!r} is repeated (first on line {first_line[q.qid]})")
        except ValueError as error:
            raise ValueError(f"{path}:{n}: {error}") from None
        questions[q.qid], first_line[q.qid] = q, n
    if not questions:
        raise ValueError(f"{path} holds no question")
    return tuple(questions.values())


def refuse_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def check(q: Question) -> None:
    """Raises a ValueError when `q` could not be asked or graded: what `read` cannot see from the types."""
    if not QID.fullmatch(q.qid):
        raise ValueError(f"qid {q.qid!r} is not a file name of letters, digits, `_`, `-` and `.`")
    try:
        if datetime.fromisoformat(q.now).utcoffset() is None:
            raise ValueError
    except ValueError:
        raise ValueError(f"now {q.now!r} is not an ISO time with an offset") from None
    g = q.grade
    if g.rule not in RULES:
        raise ValueError(f"grade.rule {g.rule!r} is not one of {', '.join(RULES)}")
    if g.date is not None:
        try:
            date.fromisoformat(g.date)
        except ValueError:
            raise ValueError(f"grade.date {g.date!r} is not an ISO date") from None
    if g.rule in ("number", "count", "code_time", "choice") and g.truth is None:
        raise ValueError(f"grade.truth is missing for the rule {g.rule}")
    if (g.rule == "set") != bool(g.items):
        raise ValueError("grade.items is for the rule set, which needs at least one" if g.items else "grade.items is missing for the rule set")
    if g.items and len(set(g.items)) != len(g.items):
        raise ValueError(f"grade.items {list(g.items)} repeats one")
    if g.items and any(ISO_DAY.fullmatch(x) for x in g.items) and not all(ISO_DAY.fullmatch(x) for x in g.items):
        raise ValueError(f"grade.items {list(g.items)} mixes dates and file names")
    if g.rule == "choice" and g.truth not in tuple("ABCD"):
        raise ValueError(f"grade.truth {g.truth!r} is not a letter from A to D")
    if g.rule in ("time", "code_time"):
        if g.time is None or g.tol_s is None:
            raise ValueError(f"grade.time and grade.tol_s are missing for the rule {g.rule}")
        if g.tol_s < 0:
            raise ValueError(f"grade.tol_s {g.tol_s} is negative")
    clocks = [("grade.ignore", x) for x in g.ignore if ":" in x]  # `grade.split` reads an ignored entry with a colon as a clock
    if g.rule == "time":
        clocks += [("grade.wrong", x) for x in g.wrong]
    if g.time is not None:
        clocks.append(("grade.time", g.time))
    for name, clock in clocks:
        if not CLOCK.fullmatch(clock):
            raise ValueError(f"{name} {clock!r} is not a clock time (H:MM or H:MM:SS)")


# -- reading a record into the dataclasses, strictly --------------------------------------------------------------------

NAMES = {str: "a string", int: "an integer", float: "a number", bool: "true or false", type(None): "null"}


@functools.cache
def hints(cls: type) -> dict[str, type]:
    return typing.get_type_hints(cls)


def read(tp, value, at: str):
    """`value` (parsed JSON) as the type `tp`: a dataclass from an object with exactly its fields, a tuple from a list, a Mapping
    from an object, and a primitive only from its own JSON type (a bool is no number, an integer is a number). `at` names where in
    the record `value` is, for the message."""
    def refuse(message: str):
        raise ValueError(f"{at}: {message}" if at else message)

    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            refuse(f"expected an object, got {value!r}")
        fields = hints(tp)
        if unknown := [k for k in value if k not in fields]:
            refuse("unknown key " + ", ".join(repr(k) for k in unknown))
        defaulted = {f.name for f in dataclasses.fields(tp) if f.default is not dataclasses.MISSING}
        if missing := [k for k in fields if k not in value and k not in defaulted]:
            refuse("missing key " + ", ".join(repr(k) for k in missing))
        return tp(**{name: read(ftype, value[name], f"{at}.{name}" if at else name) for name, ftype in fields.items() if name in value})
    if origin in (typing.Union, types.UnionType):
        arms = [a for a in args if a is not type(None)]
        if value is None and len(arms) < len(args):
            return None
        if len(arms) == 1:
            return read(arms[0], value, at)
        for arm in arms:
            try:
                return read(arm, value, at)
            except ValueError:
                pass
        names = [NAMES[a] for a in args]
        refuse(f"expected {', '.join(names[:-1])} or {names[-1]}, got {value!r}")
    if origin is tuple:
        if not isinstance(value, list):
            refuse(f"expected a list, got {value!r}")
        kinds = [args[0]] * len(value) if args[-1] is Ellipsis else list(args)
        if len(kinds) != len(value):
            refuse(f"expected {len(kinds)} items, got {len(value)}")
        return tuple(read(kind, item, f"{at}[{i}]") for i, (kind, item) in enumerate(zip(kinds, value)))
    if origin is Mapping:
        if not isinstance(value, dict):
            refuse(f"expected an object, got {value!r}")
        return {key: read(args[1], item, f"{at}.{key}") for key, item in value.items()}
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if tp in (str, int, bool) and type(value) is tp:
        return value
    refuse(f"expected {NAMES[tp]}, got {value!r}")
