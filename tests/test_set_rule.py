"""The `set` rule (core/grade.py): a "find every one" answer is right with the exact set of dates or files, and its F1 is told beside it;
and the question file's `grade.items` (core/question.py), which a line written before it existed may leave out."""

from __future__ import annotations

import dataclasses
import json

import pytest

from mmmb.core import grade
from mmmb.core.question import Grade, Span, load_questions, write_questions
from mmmb.scenarios import realdump


def of(items, ignore=()) -> Grade:
    return realdump.grade("set", items=items, ignore=ignore)


DAYS = of(["2013-12-16", "2013-12-18"])
FILES = of(["DCIM/100RECNX/IMG_0003.JPG", "DCIM/100RECNX/IMG_0010.JPG"])


@pytest.mark.parametrize("answer, right", [
    ("2013-12-16, 2013-12-18", True),
    ("16 December and December 18", True),
    ("Dec 16 2013 and 18 Dec 2013", True),
    ("Monday 2013-12-16 and Wednesday 2013-12-18", True),  # a weekday beside a date names no other date
    ("2013-12-18, 2013-12-16, 2013-12-16", True),  # the same date twice is one
    ("2013-12-16", False),  # one missing
    ("2013-12-16, 2013-12-18, 2013-12-20", False),  # one more
    ("2013-12-16, 2014-12-18", False),  # a year that is not the date's
    ("2013-12-16, 2013-12-19", False),  # a wrong day
    ("unknown", False),
    (None, False),
])
def test_a_set_of_dates_is_right_only_when_exact(answer, right):
    assert grade.grade(DAYS, answer) is right


def test_the_bounds_the_question_gave_are_no_dates_of_the_answer():
    bounded = of(["2013-12-16", "2013-12-18"], ignore=["2013-12-01", "2013-12-31"])
    assert grade.grade(bounded, "Between 2013-12-01 and 2013-12-31: 2013-12-16 and 2013-12-18")
    assert not grade.grade(DAYS, "Between 2013-12-01 and 2013-12-31: 2013-12-16 and 2013-12-18")


@pytest.mark.parametrize("answer, right", [
    ("IMG_0003.JPG and IMG_0010.JPG", True),
    ("img_0010.jpg, img_0003.jpg", True),  # in any case and order
    (r"D:\temp\dump\DCIM\100RECNX\IMG_0003.JPG and DCIM/100RECNX/IMG_0010.JPG", True),  # wherever the agent saw them
    ("IMG_0003.JPG, IMG_0010.JPG and a 1.25 s clip", True),  # a number is no file
    ("IMG_0003.JPG", False),
    ("IMG_0003.JPG, IMG_0004.JPG, IMG_0010.JPG", False),
])
def test_a_set_of_files_is_matched_by_name(answer, right):
    assert grade.grade(FILES, answer) is right


def test_the_f1_of_the_items_is_told_beside_the_verdict():
    assert grade.items_f1(DAYS, "2013-12-16, 2013-12-18") == 1.0
    assert grade.items_f1(DAYS, "2013-12-16, 2013-12-20") == pytest.approx(0.5)  # one of two found, one of two named right
    assert grade.items_f1(DAYS, "2013-12-16") == pytest.approx(2 / 3)
    assert grade.items_f1(DAYS, "unknown") == 0.0 and grade.items_f1(DAYS, None) == 0.0
    assert grade.items_f1(realdump.grade("count", truth="3"), "3") is None


def a_question(g: Grade):
    return realdump.question(code="X", qid="q1", now=1_700_000_000.0, text="Which days?", kind="days", template="days", modality="image",
                             shape="<dates>", grade=g, evidence=[Span("a.jpg", 0.0, 0.0)], sources=("camera",), since=0.0, until=1.0,
                             value="2013-12-16")


def test_the_question_file_holds_the_items_and_reads_a_line_written_before_them(tmp_path):
    file = tmp_path / "questions.jsonl"
    write_questions(file, [a_question(DAYS)])
    (back,) = load_questions(file)
    assert back.grade.items == ("2013-12-16", "2013-12-18")
    old = dataclasses.asdict(a_question(realdump.grade("count", truth="2")))
    del old["grade"]["items"]
    file.write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert load_questions(file)[0].grade.items == ()


@pytest.mark.parametrize("g, said", [
    (realdump.grade("set"), "grade.items is missing for the rule set"),
    (realdump.grade("count", truth="2", items=["2013-12-16"]), "grade.items is for the rule set"),
    (of(["2013-12-16", "2013-12-16"]), "repeats one"),
    (of(["2013-12-16", "IMG_0003.JPG"]), "mixes dates and file names"),
])
def test_a_set_grade_that_could_not_be_graded_is_refused(tmp_path, g, said):
    file = tmp_path / "questions.jsonl"
    write_questions(file, [a_question(g)])
    with pytest.raises(ValueError, match=said):
        load_questions(file)


def test_an_answer_may_repeat_the_questions_own_numbers():
    """The builder puts the question's numbers and times in the grade's `ignore` (`grade.asked`), as V7's questions have them."""
    def asked(text, g):
        return realdump.question(code="X", qid="q", now=1.7e9, text=text, kind="absent" if g.rule == "absent" else "days",
                                 template="days", modality="image", shape="<dates>", grade=g,
                                 evidence=[] if g.rule == "absent" else [Span("a.jpg", 0, 0)], sources=("camera",), since=0, until=1,
                                 value=None)
    absent = asked("On which days in September 2013 did a bobcat come past my trail camera?", realdump.grade("absent"))
    assert grade.grade(absent.grade, "No, none came in September 2013.")  # (a month of a year is a date, no number)
    assert not grade.grade(absent.grade, "Not sure; maybe 3 times in September 2013.")
    found = asked("On which days in September 2013 did a rabbit come past my trail camera?", of(["2013-09-24", "2013-09-28"]))
    assert grade.grade(found.grade, "In September 2013: 2013-09-24 and 2013-09-28")
    timed = asked("When did the vacuum run between 21:00 and 23:00?", realdump.grade("time", time="22:04:10", tol_s=60))
    assert {"21:00", "23:00"} <= set(timed.grade.ignore) and grade.grade(timed.grade, "Between 21:00 and 23:00 it ran at 22:04.")


@pytest.mark.parametrize("answer", ["2013-12-16, 2013-12-18 and maybe the 20th", "2013-12-16, 2013-12-18 or Friday 20"])
def test_a_day_named_alone_that_is_none_of_the_set_is_another_date(answer):
    assert not grade.grade(DAYS, answer)
    assert grade.grade(DAYS, "2013-12-16 (the 16th) and 2013-12-18")


def test_a_file_with_no_set_question_is_the_file_it_was(tmp_path):
    file = tmp_path / "questions.jsonl"
    write_questions(file, [a_question(realdump.grade("count", truth="2"))])
    assert '"items"' not in file.read_text(encoding="utf-8")
    write_questions(file, [a_question(DAYS)])
    assert '"items": ["2013-12-16", "2013-12-18"]' in file.read_text(encoding="utf-8")


@pytest.mark.parametrize("g, said", [
    (of(["2013-02-30"]), "is not a date"),
    (of(["cam1/clip_0001.mp4", "cam2/clip_0001.mp4"]), "two files of one name"),
    (of([r"cam1\CLIP.mp4", "cam2/clip.mp4"]), "two files of one name"),
])
def test_a_set_that_no_answer_could_name_is_refused(tmp_path, g, said):
    file = tmp_path / "questions.jsonl"
    write_questions(file, [a_question(g)])
    with pytest.raises(ValueError, match=said):
        load_questions(file)
