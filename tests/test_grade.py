"""Ported from Multimem tests/test_bench_rules.py at fe0ece5 (its 13 grader tests; the 6 about the fake decider go to tests/mm), with
the literal specs now `Grade`s: `kind` is `rule`, `tol_min` is `tol_s`, lists are tuples. The one case dropped is a spec with no
`date` key, which a `Grade` cannot be (test_question refuses a question line without one)."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from mmmb.core import grade
from mmmb.core.question import Grade

# -- the grader's date --------------------------------------------------------------------------------------------------

DAY = "2026-10-29"
assert date.fromisoformat(DAY).strftime("%A") == "Thursday"


def spec(kind: str = "time", when=DAY, truth=None, time="21:04:10", ignore=()) -> Grade:
    return Grade(kind, truth, time, 60.0, (), tuple(ignore), when)


def test_a_time_with_a_date_set_is_right_on_that_day_in_each_form():
    one = spec()
    right = ["Thursday 29 October 2026 at 21:04", "Thursday 29 October at 21:04", "21:04 on 29 October", "On 29 October, 21:04",
             "29 Oct 2026, 21:04", "October 29 at 21:04", "Oct. 29th, 2026, 9:04 pm", "The 29th of October at 21:04",
             "2026-10-29T21:04:10-04:00", "2026-10-29 21:04", "21:04, 2026-10-29", "Thursday at 21:04", "21:04 on Thursday",
             "Thursday, October 29 at 21:04", "Thu 29 Oct 21:04", "Thu. 21:04", "Thursday the 29th at 21:04",
             "Thursday 29 at 21:04", "twenty-nine October at 21:04", "October 29 21:04", "Thursday 9:04 pm"]
    assert [a for a in right if not grade.grade(one, a)] == []


def test_a_time_with_a_date_set_is_wrong_with_no_date_or_with_another_one():
    one = spec()
    wrong = ["21:04", "at 21:04 yesterday", "the sun was low at 21:04", "unknown",
             "Thursday 29 October at 22:30", "Thursday 29 October",  # the right day and a wrong time, and no time
             "Tuesday 27 October at 21:04", "Tuesday at 21:04",  # another date in the week, by weekday too
             "Thursday 22 October at 21:04", "Thursday 22 at 21:04",  # the right weekday and another date
             "Wednesday 29 October at 21:04", "Wednesday 2026-10-29 at 21:04",  # a weekday that is not that date's
             "29 October 2025 at 21:04", "2026-10-28T21:04:10-04:00", "2026-10-30 21:04",
             "29 October or 30 October at 21:04", "Thursday 29 October at 21:04, or Friday",
             "29 October at 21:04, 30 October at 21:04",
             "Thursday 29 October and Tuesday 27 October at 21:04", "2026-10-29 21:04 and 2026-10-30 21:04"]
    assert [a for a in wrong if grade.grade(one, a)] == []
    assert grade.grade(one, None) is False


def test_code_time_with_a_date_set_wants_the_code_the_time_and_the_day():
    one = spec("code_time", truth="E-4417", time="21:04:20", ignore=["18:00", "24:00"])
    right = ["E-4417 at 21:04 on Thursday 29 October", "e4417, Oct 29, 21:04", "ERR 4417 at 21:04, 2026-10-29",
             "Thursday 29 October 2026: E-4417 at 21:04 (18:00 to 24:00)", "2026-10-29T21:04:20-04:00 E-4417"]
    wrong = ["E-4417 at 21:04", "E-4417 on 29 October", "E-4417 at 21:04 on 28 October", "E-4417 at 21:04 on Tuesday",
             "E-4471 at 21:04 on 29 October", "E-4417 at 21:04 on 29 October, E-5521 at 21:10"]
    assert [a for a in right if not grade.grade(one, a)] == []
    assert [a for a in wrong if grade.grade(one, a)] == []


def test_a_time_with_no_date_in_its_spec_grades_as_it_did():
    answers = {"21:04": True, "9:04 pm": True, "Tuesday 27 October at 21:04": True, "2026-10-28T21:04:10-04:00": True,
               "21:04 and 20:00": False, "22:30": False, "unknown": False}
    for kind, truth in (("time", None), ("code_time", "E-4417")):
        for dated in (None, ""):  # a date of None, and an empty one
            one = spec(kind, when=dated, truth=truth)
            for answer, expected in answers.items():
                text = f"E-4417, {answer}" if truth else answer
                assert grade.grade(one, text) is expected, (kind, dated, answer)
    for kind in ("number", "count", "absent"):  # a date means nothing to the other kinds
        one = Grade(kind, "3", None, None, (), (), DAY)
        assert grade.grade(one, "unknown" if kind == "absent" else "3") is True


def graded(one: Grade, answers: list[str], want: bool) -> list[str]:
    """The answers whose grade by `one` is not `want`."""
    return [a for a in answers if grade.grade(one, a) is not want]


def test_a_day_of_the_month_by_its_ordinal_is_a_date_when_it_is_one():
    one = spec()
    right = ["On the 29th at 21:04", "21:04 on the 29th", "On 29th at 21:04", "At 21:04, on the 29th.",
             "Thursday the 29th, 21:04", "On 29th October at 21:04", "On the 29th (21:04)",
             "The 2nd alarm was at 21:04 on Thursday 29 October"]  # "the 2nd alarm" is no date, so no other date
    wrong = ["On the 28th at 21:04", "On the 30th, 21:04", "21:04 on the 29th and the 30th", "On the 22nd at 21:04",
             "Thursday the 22nd, 21:04", "The 2nd alarm was at 21:04", "21:04 on the 29th ring"]  # no date, or not that one
    assert graded(one, right, True) == [] and graded(one, wrong, False) == []
    assert graded(spec("code_time", truth="E-4417", time="21:04:20"), ["E-4417 at 21:04 on the 29th"], True) == []
    # a date is no number: "the 29th" is not a second count
    assert grade.numbers("3 times on the 29th") == ["3"] and grade.numbers("the 2nd alarm") == ["2"]
    count = spec("count", truth="3", time=None, ignore=["731"])
    assert graded(count, ["ERR-731 three times on the 29th", "3 times, on the 29th"], True) == []
    assert graded(count, ["3 times on the 29th and 5 more", "3 times on the 29th, 5 times on the 30th"], False) == []


def test_a_year_in_brackets_is_no_number_unless_it_is_the_answer():
    code = spec("number", truth="4417", time=None)
    assert graded(code, ["4417 (2026)", "The code is 4417 (in 2026)", "4417, noted Thursday 29 October (2026)",
                         "4417 on 29 October 2026", "4417 ( 2026 )"], True) == []
    assert graded(code, ["4417 or 2026", "4417, 2026", "4417 (1234)", "4417 (2026 or 4471)", "2026 (4417)", "4417 (2026) 4471"],
                  False) == []
    # a code can look like a year: it is the answer, in brackets or not, and then a second number is a hedge as ever
    year = spec("number", truth="2026", time=None)
    assert graded(year, ["2026", "The code is (2026)", "2026 (29 October 2026)", "2026 (in 2026)"], True) == []
    assert graded(year, ["2026 or 4417", "4417 (2026)", "(4417)"], False) == []
    assert graded(dataclasses.replace(code, wrong=("2026",)), ["4417 (2026)"], False) == [], "a wrong answer in brackets is still wrong"
    assert graded(dataclasses.replace(code, wrong=("2026",)), ["4417 (2027)"], True) == []
    # the other kinds read years the same way
    count = spec("count", truth="3", time=None, ignore=["731"])
    assert graded(count, ["ERR-731 three times (2026)", "3 times (in 2026) on 29 October 2026"], True) == []
    assert graded(count, ["3 times (2026 or 2027)", "3 times (2026), 2026"], False) == []
    absent = spec("absent", truth=None, time=None, ignore=["845"])
    assert graded(absent, ["No doorbell rang (2026)", "No ring (in 2026)"], True) == []
    assert graded(absent, ["No, but the code is 2026", "No ring, code (4417)"], False) == []
    # a bare year in a code_time answer is the year of its date, and a second code is still a second code
    coded = spec("code_time", truth="E-4417", time="21:04:20")
    assert graded(coded, ["E-4417 at 21:04 on Thursday 29 October (2026)", "E-4417 at 21:04, 2026-10-29, 2026"], True) == []
    assert graded(coded, ["E-4417 at 21:04 on 29 October, E-2027", "E-4417 or E-4471 at 21:04 on 29 October"], False) == []
    yearly = spec("code_time", truth="E-2026", time="21:04:20")
    assert graded(yearly, ["E-2026 at 21:04 on 29 October", "E-2026 at 21:04, 29 October 2026"], True) == []
    assert graded(yearly, ["E-2026 or E-4417 at 21:04 on 29 October"], False) == []
    assert grade.numbers("Tuesday 28 October 2026 at 14:03, ERR-731 three times, 2026-10-28 (2026)") == ["731", "3"]


def test_an_absent_answer_may_give_the_range_it_searched_and_no_other_time():
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    clean = ["no, nothing (I checked 9:00-17:00)", "I searched 09:00 to 17:00 and found nothing",
             "Nothing found (searched 9:00–17:00)", "None. I looked between 8:00 and 20:00 and saw nothing",
             "I checked from 2:00 pm to 6:00 pm: nothing", "Nothing (I scanned 18:00 until 24:00)",
             "No doorbell rang between 12:00 and 18:00 (I checked 9:00-17:00)", "I checked 9:00-17:00 and found nothing, no ring",
             "None found from 9:00 to 17:00", "No parcel was left at the door (I checked 14:00-15:00)",
             "It did not ring at 13:00-14:00", "Nothing; I checked 9:00 to 5:00 pm", "I checked 9:30-10:30 and it never appeared"]
    assert graded(absent, clean, True) == []
    sighting = ["nothing, but there was one at 14:05",  # a time that is no range
                "No. I checked 9:00-17:00 but it rang at 14:05", "no, nothing (I checked 9:00-17:00) except 14:05",
                "I checked 9:00-17:00 and heard it once at 14:05, no ring after",  # the range is searched, the time is not
                "No, I checked 9:00-17:00, 14:05", "no ring; heard 14:05",
                "nothing, but I heard one between 14:05 and 14:10",  # a range that is short, in an answer of contrast
                "no, nothing: I checked 14:00-14:30", "No. I checked 14:10-14:05",  # under an hour, or backwards
                "No, I checked 12:75-17:00", "nothing found, I checked at 2026-10-29T21:04:10-04:00"]  # no range at all
    assert graded(absent, sighting, False) == []


def test_a_short_range_is_not_read_as_twelve_hours_more():
    # with no am or pm a time is two times of day, and two of them are read in the same half: a minute is a minute
    short = ["No; I checked 10:05-10:06.", "Could not confirm it; I listened 7:27-7:28 and it sounded like the alarm.",
             "Not certain; looked at 9:58-10:01, the dialog with the code showed then.", "No, I checked 10:05 to 10:06",
             "Nothing; I looked between 3:10 and 3:40", "No; I checked 11:30-12:20", "Nothing, I checked 10:00-10:00"]
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    assert graded(absent, short, False) == []
    assert [grade.span(*(grade.mentions(t)[0] for t in pair)) for pair in
            (("10:05", "10:06"), ("7:27", "7:28"), ("9:58", "10:01"), ("9:00", "17:00"), ("8:00", "20:00"), ("2:00", "6:00"),
             ("9:00", "5:00 pm"), ("11:30", "12:30"), ("10:05", "10:05"), ("9:00", "5:00"), ("14:10", "14:05"))] \
        == [1, 1, 3, 480, 720, 240, 480, 60, None, None, None]
    # an hour or more is a range, whatever the hour
    long = ["Nothing; I checked 7:00-9:30", "No; I checked 10:00-11:00", "No, I looked 22:00 to 23:30", "None found 1:00 to 3:00"]
    assert graded(absent, long, True) == []


def test_an_absent_answer_that_says_the_thing_was_there_keeps_its_times_whatever_it_negates():
    absent = spec("absent", truth=None, time=None, ignore=["12:00", "18:00"])
    claims = ["Yes, a parcel was left; I couldn't read the label. Checked 14:00-15:00.",
              "I could not pin the exact minute; scanning 14:00-16:00 shows the parcel was delivered.",
              "The error dialog never closed; I checked 13:00-15:00 and it was on screen the whole time.",
              "The alarm never stopped, I checked 14:00-16:00", "No, the alarm sounded; I checked 14:00-16:00",
              "No. It rang. I checked 13:00-15:00.", "A parcel was left at the door. Not sure when (I checked 14:00-15:00)",
              "No parcel? It rang between 14:00 and 16:00", "Not sure, but I checked 14:00-16:00 and it was there",
              "I did not see the label; the parcel was still there at 14:00-16:00",
              "Nothing in the log but yes, a ring between 14:00 and 16:00"]
    assert graded(absent, claims, False) == []
    # the same ranges in answers that negate throughout
    assert graded(absent, ["No parcel was left (I checked 14:00-15:00)", "No alarm sounded between 14:00 and 16:00",
                           "Not found; I checked 13:00-15:00 and nothing was on screen"], True) == []
    # `searched` leaves an answer it does not trust as it was, and blanks a range only where it trusts it
    assert grade.searched("no, nothing (i checked 9:00-17:00)").strip() == "no, nothing (i checked  )"
    assert grade.searched("no, nothing (i checked 9:00-17:00) but 14:05") == "no, nothing (i checked 9:00-17:00) but 14:05"
    assert grade.searched("it rang; i checked 9:00-17:00") == "it rang; i checked 9:00-17:00"


def test_a_spec_date_that_is_not_iso_is_an_error():
    with pytest.raises(ValueError):
        grade.grade(spec(when="29 October"), "Thursday 29 October at 21:04")


# -- the grader's choice (EgoLife's multiple-choice questions) -----------------------------------------------------------

def choice_spec(truth: str = "B") -> Grade:
    return Grade("choice", truth, None, None, (), (), None)


def test_a_choice_is_right_when_the_letter_the_answer_starts_with_is_the_truth():
    one = choice_spec()
    right = ["B", "b", "(B)", "B) Alice", "B.", "B - Alice", "  B  ", "(b) Alice", "B: Alice", "B, Alice", "b)", "B\nbecause"]
    wrong = ["Alice", "B or C", "The answer is B", "unknown", "A", "C) Tasha", "Bob", "E", "", "ANSWER: B"]
    assert graded(one, right, True) == [] and graded(one, wrong, False) == []
    assert grade.grade(one, None) is False and "choice" in grade.RULES
    assert [grade.choice(a) for a in ("a", "(D)", "d.", "C - Lucia", "(c) the cup")] == ["A", "D", "D", "C", "C"]
    assert grade.choice("a screwdriver") is None and grade.choice("b the cup") is None, "a lower-case word is no option"
    # one pair of brackets, quotes or backticks around the whole answer is taken off; a lower-case word inside is still no option
    wrapped = ["<B>", "[B]", '"B"', "'b'", "`B`", '"B) Alice"', "<B - Alice>", "[ b ]"]
    assert [grade.choice(a) for a in wrapped] == ["B"] * len(wrapped) and graded(choice_spec(), wrapped, True) == []
    assert [grade.choice(a) for a in ("'a screwdriver'", "<B or C>", '"Alice"', "<B", "B>", "<>", '""', "<<B>>")] == [None] * 8


def test_an_answer_that_names_a_second_letter_is_a_hedge_and_is_wrong_whichever_one_is_right():
    hedges = ["B or C", "B OR C", "B/C", "B, C", "B and C", "B & C", "B) Alice or C) Lucia", "(B) or (C)"]
    for truth in "BC":
        assert graded(choice_spec(truth), hedges, False) == [], truth
    assert [grade.choice(a) for a in hedges] == [None] * len(hedges)
    # a word that starts with a letter, or an option's own words, are no second letter
    for answer in ("B) Alice or Dan's cup", "B - Charlie, Dana", "B) a cup or a plate", "B, because Alice used it"):
        assert grade.choice(answer) == "B", answer


def test_an_unknown_choice_is_wrong_and_also_abstains_and_a_choice_has_no_other_rule():
    one = choice_spec("D")
    assert grade.grade(one, "unknown") is False and grade.abstained("unknown", one.ignore) is True
    assert grade.abstained("B", one.ignore) is False and grade.abstained(None, one.ignore) is False
    assert grade.grade(one, "D") is True and grade.grade(dataclasses.replace(one, truth="A"), "D") is False
    with pytest.raises(ValueError, match="one of number, count.*choice"):
        grade.grade(dataclasses.replace(one, rule="pick"), "D")


# -- new: what the move to `Grade` and to seconds could break, and a rule the ported tests do not reach ---------------------

def test_a_time_is_right_within_its_tolerance_in_seconds_and_not_a_second_further():
    """The model is plain arithmetic in seconds: an answer 21:MM is right when it is at most `tol_s` from the truth."""
    for tol_s in (30.0, 60.0, 120.0, 300.0):
        for rule, truth, time, text in (("time", None, "21:04:10", "{}"), ("code_time", "E-4417", "21:04:20", "E-4417 at {}")):
            one = Grade(rule, truth, time, tol_s, (), (), None)
            seconds = 4 * 60 + int(time[-2:])
            for minute in range(12):
                assert grade.grade(one, text.format(f"21:{minute:02d}")) is (abs(minute * 60 - seconds) <= tol_s), (rule, tol_s, minute)
    exact = Grade("time", None, "21:05:00", 60.0, (), (), None)  # 21:04 and 21:06 are exactly a tolerance away: right
    assert graded(exact, ["21:04", "21:05", "21:06"], True) == [] and graded(exact, ["21:03", "21:07"], False) == []


def test_a_count_reads_once_twice_and_thrice_as_numbers():
    two = Grade("count", "2", None, None, (), ("731",), None)
    assert graded(two, ["twice", "It happened twice", "2 times", "twice (ERR-731)"], True) == []
    assert graded(two, ["once", "thrice", "three times", "twice and once", "unknown"], False) == []
    assert graded(Grade("count", "1", None, None, (), (), None), ["once", "1"], True) == []
    assert graded(Grade("count", "3", None, None, (), (), None), ["thrice", "three times"], True) == []


def test_a_count_reads_no_number_in_a_name_and_takes_the_truth_even_when_the_question_holds_it():
    """A camera named `cam3` in the question or the answer is no count, and a count the question also holds is still right."""
    assert grade.asked("How many times did a car stop on cam3 between 10:10 and 10:40?") == ["10:10", "10:40"]
    on_cam3 = Grade("count", "2", None, None, (), ("10:10", "10:40"), None)
    assert graded(on_cam3, ["2", "2 (cam3)", "twice on cam3, in IMG_0003.JPG and IMG_0010.JPG"], True) == []
    assert graded(on_cam3, ["3", "2 or 3", "cam3: 4"], False) == []
    in_3_hours = Grade("count", "3", None, None, (), ("3",), None)  # "in the last 3 hours", and it was 3
    assert graded(in_3_hours, ["3", "3 times in 3 hours"], True) == []
    assert graded(in_3_hours, ["2 in 3 hours", "unknown"], False) == []
