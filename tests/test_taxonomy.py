"""The filing of the scenarios (core/taxonomy.py) and the two rules for values read or heard (`text`, `amount` of core/grade.py)."""

from __future__ import annotations

import pytest

from mmsb import scenarios
from mmsb.core.grade import grade
from mmsb.core.question import Grade, Scope, Question, Truth, check
from mmsb.core.taxonomy import CATEGORY_KEYS, UNFILED, Shelf, check as check_filing, filed, with_absent


def g(rule: str, truth: str, tol: float | None = None, wrong=(), ignore=()) -> Grade:
    return Grade(rule, truth, None, tol, tuple(wrong), tuple(ignore), None)


@pytest.mark.parametrize("name", scenarios.REGISTRY)
def test_every_scenario_is_on_a_shelf_and_files_its_templates(name):
    scenario = scenarios.get(name)
    check_filing(scenario.shelf, scenario.filing)
    assert scenario.filing


def test_absent_twins_are_filed_as_the_question_they_read_as():
    filing = with_absent({"first": ("when", "sight")})
    assert filing["absent_first"] == ("when", "sight")


def test_a_scenario_with_no_filing_is_unfiled():
    assert filed(object(), "first") == (UNFILED, UNFILED)


def test_a_filing_with_an_unknown_word_is_refused():
    with pytest.raises(ValueError, match="not a category"):
        check_filing(Shelf("x", "video", "home", "y"), {"t": ("guess", "sight")})
    with pytest.raises(ValueError, match="not one of"):
        check_filing(Shelf("x", "tape", "home", "y"), {})


def test_the_categories_are_the_eight_of_the_catalogue():
    assert CATEGORY_KEYS == ("find", "find_all", "when", "did", "how_many", "what", "who", "order")


@pytest.mark.parametrize("answer, right", [
    ("Project manager", True),
    ("It was the project manager.", True),
    ("the Project-Manager", True),
    ("the designer", False),
    ("the project manager or the designer", False),  # a wrong phrase beside the right one
    ("manager", False),  # part of the phrase only
    (" ".join(["word"] * 30) + " project manager", False),  # a list of guesses is no short value
])
def test_text_takes_a_short_answer_with_an_accepted_phrase(answer, right):
    assert grade(g("text", "project manager|pm", wrong=("designer",)), answer) is right


@pytest.mark.parametrize("truth, wrong, answer, right", [
    ("58", (), "It showed 58%.", True),
    ("58", (), "40, 50, 58, 60, 70, 76", False),
    ("thousand|1000", (), "ten thousand dollars", False),
    ("300", (), "somewhere between 200 and 300", False),
    ("paypal", ("apple", "amazon", "chase"), "PayPal, or maybe Apple", False),
    ("paypal", ("apple", "amazon", "chase"), "PayPal and Apple", False),
    ("chemistry", (), "chemistry, or maybe biology", False),
    ("before", ("after",), "Before: OPEC came first, and patents came up after it.", True),
    ("before", ("after",), "After, not before.", False),
    ("before", ("after",), "before or after, I can't tell", False),
    ("off", ("on",), "She asked for the lights to go off first, then on.", True),
])
def test_text_refuses_a_second_guess_and_lets_an_explanation_name_the_other_side(truth, wrong, answer, right):
    assert grade(g("text", truth, wrong=wrong), answer) is right


def test_text_takes_any_of_the_accepted_phrases():
    assert grade(g("text", "project manager|pm"), "The PM said it")


@pytest.mark.parametrize("answer, right", [
    ("12.50", True),
    ("$12.5", True),
    ("about 12.49 dollars", True),
    ("1,250.00", False),
    ("13", False),
    ("12.50 or 13", False),  # a hedge
    ("the total was twelve", False),
])
def test_amount_is_one_number_within_the_tolerance(answer, right):
    assert grade(g("amount", "12.50", 0.01), answer) is right


def test_amount_reads_thousands_and_leaves_out_the_questions_own_numbers():
    assert grade(g("amount", "1250", 0.01), "1,250.00")
    assert grade(g("amount", "7", 1.0, ignore=("30",)), "7 minutes in the 30 minute talk")
    assert not grade(g("amount", "7", 1.0), "7 minutes in the 30 minute talk")


def test_amount_reads_a_currency_stuck_to_the_number_and_a_dot_of_thousands():
    assert grade(g("amount", "91.50", 0.05), "RM91.50")
    assert grade(g("amount", "67000", 0.0), "Rp 67.000")
    assert grade(g("amount", "67", 0.5), "67.000")
    assert not grade(g("amount", "3", 0.0), "cam3")


def question(grade_: Grade) -> Question:
    return Question("q", "X", "dev", "2026-01-01T00:00:00+00:00", "?", (), "what", "none", Truth("v", (), ()), grade_, {}, None,
                    Scope((), 0.0, 1.0), "q", None, (), None)


@pytest.mark.parametrize("bad, message", [
    (g("text", None), "truth is missing"),
    (g("amount", None, 1.0), "truth is missing"),
    (g("amount", "7", None), "tol_s is missing"),
    (g("amount", "seven", 1.0), "not a number"),
])
def test_the_loader_refuses_a_bad_value_rule(bad, message):
    with pytest.raises(ValueError, match=message):
        check(question(bad))
