"""The question file (mmsb/core/question.py; plan 5.1, DESIGN 6.1). New tests; the expected results are what each test plants: a
record written by hand that is valid, the same record with one fault, the rows a writer wrote and the reader must give back."""

from __future__ import annotations

import copy
import json
import random
import string
from pathlib import Path

import pytest

from mmsb.core import grade
from mmsb.core.question import Grade, Question, RetrievalSpec, Scope, Span, Truth, load_questions, write_questions

# a valid line, written by hand (a V7 `count` question): every key the schema has, nothing else
BASE = {
    "qid": "video-porch-parcel-1", "scenario": "V7", "split": "dev", "now": "2026-11-04T12:00:00-05:00",
    "question": "How many parcels were left at the door?", "attachments": [], "kind": "count", "hint": "day",
    "truth": {"value": "2", "evidence": [{"file": "porch/2026-11-02-09-14-05.mkv", "t0": 12.0, "t1": 18.5},
                                         {"file": "porch/2026-11-03-16-40-00.mkv", "t0": 3.0, "t1": 9.0}],
              "wrong": [{"file": "porch/2026-11-01-10-00-00.mkv", "t0": 0.0, "t1": 4.0}]},
    "grade": {"rule": "count", "truth": "2", "time": None, "tol_s": None, "wrong": [], "ignore": [], "date": None},
    "tags": {"modality": "video", "template": "parcel", "level": "day", "last": False, "absent": False, "lookback_h": 30.5,
             "rung": "pilot", "seed": 1, "note": None},
    "options": None, "scope": {"sources": ["porch"], "since": 1793700000.0, "until": 1793786400.0}, "group": "parcel",
    "horizon": None, "cost_ranges": [[1793700000.0, 1793786400.0]], "retrieval": None,
}
RETRIEVAL = {"query": "Is a parcel at the door?", "sources": ["porch"], "ranges": [[1793700000.0, 1793786400.0]], "k": 5,
             "oldest_first": False, "tol_s": 10.0, "moments": [[1793740445.0, 1793740451.5]], "others": []}


def with_(record: dict, **changes) -> dict:
    """`record` with the dotted keys of `changes` set (`grade.rule="x"`, written `**{"grade.rule": "x"}`); a value of KEY removes it."""
    record = copy.deepcopy(record)
    for dotted, value in changes.items():
        *path, last = dotted.split(".")
        node = record
        for step in path:
            node = node[int(step)] if isinstance(node, list) else node[step]
        if value is KEY:
            del node[last]
        else:
            node[int(last) if isinstance(node, list) else last] = copy.deepcopy(value)
    return record


KEY = object()


def file_of(tmp_path: Path, *records, raw: str | None = None) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(raw if raw is not None else "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_a_valid_line_is_read_into_the_dataclasses(tmp_path):
    (q,) = load_questions(file_of(tmp_path, BASE))
    assert q == Question(
        qid="video-porch-parcel-1", scenario="V7", split="dev", now="2026-11-04T12:00:00-05:00",
        question="How many parcels were left at the door?", attachments=(), kind="count", hint="day",
        truth=Truth("2", (Span("porch/2026-11-02-09-14-05.mkv", 12.0, 18.5), Span("porch/2026-11-03-16-40-00.mkv", 3.0, 9.0)),
                    (Span("porch/2026-11-01-10-00-00.mkv", 0.0, 4.0),)),
        grade=Grade("count", "2", None, None, (), (), None),
        tags={"modality": "video", "template": "parcel", "level": "day", "last": False, "absent": False, "lookback_h": 30.5,
              "rung": "pilot", "seed": 1, "note": None},
        options=None, scope=Scope(("porch",), 1793700000.0, 1793786400.0), group="parcel", horizon=None,
        cost_ranges=((1793700000.0, 1793786400.0),), retrieval=None)
    assert type(q.tags["seed"]) is int and type(q.tags["last"]) is bool and type(q.tags["lookback_h"]) is float


def test_a_retrieval_spec_options_and_a_horizon_are_read_and_an_integer_is_a_number(tmp_path):
    record = with_(BASE, retrieval=RETRIEVAL, options={"A": "red", "B": "blue"}, horizon=1793740000, **{"scope.since": 5})
    (q,) = load_questions(file_of(tmp_path, record))
    assert q.retrieval == RetrievalSpec("Is a parcel at the door?", ("porch",), ((1793700000.0, 1793786400.0),), 5, False, 10.0,
                                        ((1793740445.0, 1793740451.5),), ())
    assert q.options == {"A": "red", "B": "blue"} and q.horizon == 1793740000.0 and q.scope.since == 5.0
    assert type(q.horizon) is float and type(q.scope.since) is float and type(q.retrieval.k) is int


# every fault is one change to the valid line; the message names the line and the key
FAULTS = [
    ("an extra key", {"extra": 1}, "unknown key 'extra'"),
    ("an extra key in the grade", {"grade.extra": 1}, "grade: unknown key 'extra'"),
    ("an extra key in a span", {"truth.evidence.0.extra": 1}, "truth.evidence[0]: unknown key 'extra'"),
    ("a missing key", {"hint": KEY}, "missing key 'hint'"),
    ("a missing key in the grade", {"grade.date": KEY}, "grade: missing key 'date'"),
    ("a missing key in the scope", {"scope.until": KEY}, "scope: missing key 'until'"),
    ("a number for a qid", {"qid": 7}, "qid: expected a string"),
    ("a string for a list", {"scope.sources": "porch"}, "scope.sources: expected a list"),
    ("a list for an object", {"tags": ["modality"]}, "tags: expected an object"),
    ("an object for a tag value", {"tags.level": {"a": 1}}, "tags.level: expected a string, an integer, a number, true or false or null, got {'a': 1}"),
    ("a string for a time", {"truth.evidence.0.t0": "12"}, "truth.evidence[0].t0: expected a number"),
    ("a bool for a time", {"truth.evidence.0.t0": True}, "truth.evidence[0].t0: expected a number"),
    ("a float for an integer", {"retrieval": RETRIEVAL, "retrieval.k": 5.5}, "retrieval.k: expected an integer"),
    ("a bool for an integer", {"retrieval": RETRIEVAL, "retrieval.k": True}, "retrieval.k: expected an integer"),
    ("a number for a bool", {"retrieval": RETRIEVAL, "retrieval.oldest_first": 0}, "retrieval.oldest_first: expected true or false"),
    ("a range of three", {"cost_ranges": [[1.0, 2.0, 3.0]]}, "cost_ranges[0]: expected 2 items"),
    ("a string for a horizon", {"horizon": "noon"}, "horizon: expected a number, got 'noon'"),
    ("a list for the options", {"options": ["red"]}, "options: expected an object"),
    ("an option that is no string", {"options": {"A": 1}}, "options.A: expected a string"),
    ("an unknown rule", {"grade.rule": "fuzzy"}, "grade.rule 'fuzzy' is not one of number, count, time, code_time, absent, choice"),
    ("a qid with a slash", {"qid": "a/b"}, "qid 'a/b' is not a file name"),
    ("a qid that climbs", {"qid": "../x"}, "qid '../x' is not a file name"),
    ("an empty qid", {"qid": ""}, "qid '' is not a file name"),
    ("a hidden qid", {"qid": ".x"}, "qid '.x' is not a file name"),
    ("a now with no offset", {"now": "2026-11-04T12:00:00"}, "now '2026-11-04T12:00:00' is not an ISO time with an offset"),
    ("a now that is no time", {"now": "tomorrow"}, "now 'tomorrow' is not an ISO time with an offset"),
    ("a date that is not ISO", {"grade.date": "29 October"}, "grade.date '29 October' is not an ISO date"),
    ("an empty date", {"grade.date": ""}, "grade.date '' is not an ISO date"),
    ("a count with no truth", {"grade.truth": None}, "grade.truth is missing for the rule count"),
    ("a choice that is not a letter", {"grade.rule": "choice", "grade.truth": "E"}, "grade.truth 'E' is not a letter from A to D"),
    ("a time with no time", {"grade.rule": "time", "grade.tol_s": 60}, "grade.time and grade.tol_s are missing for the rule time"),
    ("a time with no tolerance", {"grade.rule": "time", "grade.time": "21:04:10"}, "grade.time and grade.tol_s are missing"),
    ("a negative tolerance", {"grade.rule": "time", "grade.time": "21:04:10", "grade.tol_s": -1}, "grade.tol_s -1.0 is negative"),
    ("a time that is no clock", {"grade.rule": "time", "grade.time": "9pm", "grade.tol_s": 60}, "grade.time '9pm' is not a clock time"),
    ("a wrong time that is no clock", {"grade.rule": "time", "grade.time": "21:04", "grade.tol_s": 60, "grade.wrong": ["late"]},
     "grade.wrong 'late' is not a clock time"),
    ("an ignored clock that is no clock", {"grade.ignore": ["18:xx"]}, "grade.ignore '18:xx' is not a clock time"),
]


@pytest.mark.parametrize("what, changes, message", FAULTS, ids=[f[0] for f in FAULTS])
def test_each_malformed_question_is_refused_with_its_file_line_and_key(tmp_path, what, changes, message):
    path = file_of(tmp_path, BASE, with_(BASE, qid="second"), with_(with_(BASE, qid="third"), **changes))
    with pytest.raises(ValueError) as refused:
        load_questions(path)
    assert str(refused.value).startswith(f"{path}:3: ") and message in str(refused.value)


RAW = [
    ("a line that is not JSON", '{"qid": \n', "not JSON"),
    ("a blank line", "\n", "not JSON"),
    ("an array", "[1, 2]\n", "expected an object"),
    ("a number", "7\n", "expected an object"),
    ("NaN", json.dumps(BASE).replace('"t0": 12.0', '"t0": NaN') + "\n", "NaN is not JSON"),
    ("Infinity", json.dumps(BASE).replace('"t0": 12.0', '"t0": Infinity') + "\n", "Infinity is not JSON"),
]


@pytest.mark.parametrize("what, line, message", RAW, ids=[r[0] for r in RAW])
def test_a_line_that_is_not_a_question_object_is_refused_at_its_line(tmp_path, what, line, message):
    path = file_of(tmp_path, raw=json.dumps(BASE) + "\n" + line)
    with pytest.raises(ValueError, match=rf"questions\.jsonl:2: .*{message}"):
        load_questions(path)


def test_a_repeated_qid_is_refused_and_names_the_first_line(tmp_path):
    path = file_of(tmp_path, BASE, with_(BASE, qid="other"), BASE)
    with pytest.raises(ValueError, match=r"questions\.jsonl:3: qid 'video-porch-parcel-1' is repeated \(first on line 1\)"):
        load_questions(path)


def test_an_empty_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no question"):
        load_questions(file_of(tmp_path, raw=""))


def test_a_qid_may_hold_dots_digits_and_hyphens_and_a_date_may_be_set_with_a_time_rule(tmp_path):
    record = with_(BASE, qid="ego-001.v2", **{"grade.rule": "time", "grade.time": "21:04:10", "grade.tol_s": 60, "grade.truth": "21:04:10",
                                              "grade.wrong": ["20:00"], "grade.ignore": ["18:00", "731"], "grade.date": "2026-10-29"})
    (q,) = load_questions(file_of(tmp_path, record))
    assert q.grade == Grade("time", "21:04:10", "21:04:10", 60.0, ("20:00",), ("18:00", "731"), "2026-10-29")


# -- write and read back ------------------------------------------------------------------------------------------------

def random_text(rng: random.Random) -> str:
    # the last five are line breaks to str.splitlines and text to JSON
    alphabet = string.ascii_letters + string.digits + " \n\t\"'\\/{}[]:,é世界🙂  \x85\x0b\x1c"
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))


def random_question(rng: random.Random, n: int) -> Question:
    def number() -> float:
        return rng.choice([rng.uniform(-1e9, 1e9), float(rng.randrange(0, 10 ** 10)), rng.random() * 1e-7, 0.1 + 0.2])

    def span() -> Span:
        t0 = number()
        return Span(random_text(rng), t0, t0 + rng.random())

    def pair() -> tuple[float, float]:
        return (number(), number())

    rule = rng.choice(["number", "count", "time", "code_time", "absent", "choice"])
    clock = f"{rng.randrange(0, 24)}:{rng.randrange(0, 60):02d}"
    g = Grade(rule, {"choice": "C", "absent": None}.get(rule, str(rng.randrange(1000))), clock if rule in ("time", "code_time") else None,
              float(rng.randrange(0, 600)) if rule in ("time", "code_time") else None,
              tuple(random_text(rng) for _ in range(rng.randrange(3))), tuple(rng.choice(["731", "18:00"]) for _ in range(rng.randrange(3))),
              rng.choice([None, "2026-10-29"]))
    if rule == "time":
        g = Grade(rule, g.truth, g.time, g.tol_s, tuple(clock for _ in g.wrong), g.ignore, g.date)
    tags = {random_text(rng) or "k": rng.choice([random_text(rng), rng.randrange(-5, 5), number(), rng.random() < 0.5, None])
            for _ in range(rng.randrange(5))}
    return Question(
        qid=f"q-{n}", scenario=rng.choice(["V7", "V8"]), split=rng.choice(["dev", "test"]),
        now=rng.choice(["2026-11-04T12:00:00-05:00", "2026-10-29T21:04:10+00:00", "2026-11-01T00:00:00.5+01:00"]),
        question=random_text(rng), attachments=tuple(random_text(rng) for _ in range(rng.randrange(3))), kind=random_text(rng),
        hint=random_text(rng), truth=Truth(rng.choice([None, random_text(rng)]), tuple(span() for _ in range(rng.randrange(4))),
                                           tuple(span() for _ in range(rng.randrange(3)))),
        grade=g, tags=tags, options=rng.choice([None, {"A": random_text(rng), "B": random_text(rng)}]),
        scope=Scope(tuple(random_text(rng) for _ in range(rng.randrange(1, 3))), number(), number()), group=random_text(rng),
        horizon=rng.choice([None, number()]), cost_ranges=tuple(pair() for _ in range(rng.randrange(3))),
        retrieval=rng.choice([None, RetrievalSpec(random_text(rng), (random_text(rng),), tuple(pair() for _ in range(rng.randrange(3))),
                                                  rng.randrange(1, 9), rng.random() < 0.5, number(), tuple(pair() for _ in range(2)),
                                                  tuple(pair() for _ in range(rng.randrange(2))))]))


def test_what_a_writer_wrote_is_what_the_reader_gives_back_for_any_question(tmp_path):
    rng = random.Random(20261105)
    questions = tuple(random_question(rng, n) for n in range(300))
    path = tmp_path / "out" / "questions.jsonl"
    write_questions(path, questions)
    assert load_questions(path) == questions
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == len(questions) and text.endswith("\n"), "a line each, whatever the text holds"
    assert sorted(p.name for p in path.parent.iterdir()) == ["questions.jsonl"], "whole or not at all"


def test_the_writer_refuses_a_number_json_cannot_hold(tmp_path):
    q = load_questions(file_of(tmp_path, BASE))[0]
    broken = Question(**{**q.__dict__, "horizon": float("nan")})
    with pytest.raises(ValueError):
        write_questions(tmp_path / "again.jsonl", [broken])
    assert not (tmp_path / "again.jsonl").exists()


# -- the grader reads what the loader gives -----------------------------------------------------------------------------

GRADED = [  # a grade as the file holds it, an answer it accepts and one it refuses
    ({"rule": "number", "truth": "4417", "time": None, "tol_s": None, "wrong": ["4471"], "ignore": [], "date": None}, "The code is 4417", "4471"),
    ({"rule": "count", "truth": "3", "time": None, "tol_s": None, "wrong": [], "ignore": ["731"], "date": None}, "three times, ERR-731", "4"),
    ({"rule": "time", "truth": "21:04:10", "time": "21:04:10", "tol_s": 60, "wrong": [], "ignore": [], "date": "2026-10-29"},
     "Thursday 29 October at 21:04", "21:04"),
    ({"rule": "code_time", "truth": "E-4417", "time": "21:04:20", "tol_s": 120, "wrong": [], "ignore": [], "date": None},
     "E-4417 at 21:05", "E-4417 at 21:08"),
    ({"rule": "absent", "truth": None, "time": None, "tol_s": None, "wrong": [], "ignore": [], "date": None}, "nothing, unknown", "at 14:05"),
    ({"rule": "choice", "truth": "B", "time": None, "tol_s": None, "wrong": [], "ignore": [], "date": None}, "B) red", "C"),
]


@pytest.mark.parametrize("record, right, wrong", GRADED, ids=[g[0]["rule"] for g in GRADED])
def test_a_grade_read_from_the_file_accepts_the_right_answer_and_refuses_the_wrong_one(tmp_path, record, right, wrong):
    (q,) = load_questions(file_of(tmp_path, with_(BASE, grade=record)))
    assert grade.grade(q.grade, right) is True and grade.grade(q.grade, wrong) is False
    assert grade.grade(q.grade, None) is False
