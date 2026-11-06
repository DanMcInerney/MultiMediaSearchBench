"""New tests for the final answer (mmmb/core/answer.py, DESIGN 6.2, plan 5.4): the `line` format is the recorded `answer_of`, the `json`
format is the last complete JSON object of the message with a string `answer` and a list `evidence` of citations, and the sentences the
prompts end with ask for them. Expected values are written by hand from the rules in the module's text and the plan, or are what a test
plants: a message built from known parts is read back as those parts (a law over many generated messages), a hostile message has the
result its rule gives, and the recorded prompts' last sentences are the pilot's and EgoLife's own words.

The command line's side (the format a run keeps, the refusal to change it) is tests/test_answer_run.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from stub_system import tiny_questions
from v8_fake import V8, built, oracle_file, plan, recipe, source, table  # noqa: F401  (a fake download, for the prompt of a question of EgoLife)

from mmmb import scenarios
from mmmb.core.answer import DEFAULT_FORMAT, FORMATS, Citation, FinalAnswer, answer_of, answer_request, parse_final
from mmmb.core.contract import McpServer, Served
from mmmb.core.scenario import Access

V7 = scenarios.get("v7_screen")


def json_of(text: str) -> FinalAnswer:
    return parse_final(text, "json")


# -- the line format: the recorded one -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, answer", [
    ("thinking\n**ANSWER: 4821**", "4821"),
    ("ANSWER: no\nmore\nANSWER: unknown", "unknown"),
    ("4821", None),
    ("  answer:   13:05 on Monday  \n", "13:05 on Monday"),
    ("The answer is 4. ANSWER: 4 is not at the start of a line", None),
    ("", None),
])
def test_the_line_format_is_the_text_after_the_last_answer_line_and_has_no_evidence(text, answer):
    assert answer_of(text) == answer
    assert parse_final(text, "line") == FinalAnswer(answer, (), None)


def test_the_line_format_does_not_read_a_json_message_and_the_json_format_does_not_read_an_answer_line():
    message = json.dumps({"answer": "7", "evidence": []})
    assert parse_final(message, "line") == FinalAnswer(None, (), None), "a json message has no ANSWER: line"
    got = json_of("ANSWER: 7")
    assert got.answer is None and got.evidence == () and got.error == "the final message holds no complete JSON object"
    mixed = json_of('ANSWER: 7\n{"answer": 7, "evidence": []}')
    assert mixed.answer is None and "must be a string" in mixed.error, "and a json message that fails does not fall back to the line"


def test_an_unknown_format_is_refused_and_the_formats_are_the_two_with_json_the_default():
    assert FORMATS == ("line", "json") and DEFAULT_FORMAT == "json"
    with pytest.raises(ValueError, match="the answer format is one of line, json, not 'xml'"):
        parse_final("x", "xml")
    with pytest.raises(ValueError, match="the answer format is one of line, json, not 'xml'"):
        answer_request("xml", "mm")


# -- the json format: what it takes -----------------------------------------------------------------------------------------------------

def test_a_message_that_is_one_object_gives_its_answer_and_its_citations_in_the_three_forms():
    got = json_of(json.dumps({"answer": "Monday 07:30", "evidence": [
        {"file": "cam1/2026-11-02_07.mp4", "t0": 1788, "t1": 1802.5},
        {"file": "cam1/2026-11-02_07.mp4", "t0": "07:29:50", "t1": "07:30:05"},
        {"source": "porch", "t0": "2026-11-02T07:29:50-05:00", "t1": "2026-11-02T07:30:05-05:00"}]}))
    assert got == FinalAnswer("Monday 07:30", (
        Citation("cam1/2026-11-02_07.mp4", None, 1788.0, 1802.5),
        Citation("cam1/2026-11-02_07.mp4", None, "07:29:50", "07:30:05"),
        Citation(None, "porch", "2026-11-02T07:29:50-05:00", "2026-11-02T07:30:05-05:00")), None)
    assert [c.to_json() for c in got.evidence] == [
        {"file": "cam1/2026-11-02_07.mp4", "t0": 1788.0, "t1": 1802.5},
        {"file": "cam1/2026-11-02_07.mp4", "t0": "07:29:50", "t1": "07:30:05"},
        {"source": "porch", "t0": "2026-11-02T07:29:50-05:00", "t1": "2026-11-02T07:30:05-05:00"}]


def test_an_answer_with_no_evidence_is_an_answer_and_an_empty_answer_is_a_string():
    assert json_of('{"answer": "unknown", "evidence": []}') == FinalAnswer("unknown", (), None)
    assert json_of('{"answer": "", "evidence": []}') == FinalAnswer("", (), None)
    assert json_of('{"evidence": [], "answer": "3", "note": "extra keys are the agent\'s"}') == FinalAnswer("3", (), None)


@pytest.mark.parametrize("message", [
    '```json\n{"answer": "4", "evidence": []}\n```',
    'I looked at the files.\n\n```\n{"answer": "4", "evidence": []}\n```\nHope that helps.',
    'Here is my answer: {"answer": "4", "evidence": []} -- done',
    '  \n\t{"answer": "4",\n "evidence": [\n]\n}\n\n',
])
def test_a_code_fence_and_the_prose_around_the_object_are_just_text(message):
    assert json_of(message) == FinalAnswer("4", (), None)


def test_braces_in_a_string_and_in_the_prose_do_not_break_the_object():
    nested = '{"answer": "set {a} and } then {", "evidence": []}'
    assert json_of(nested).answer == "set {a} and } then {"
    assert json_of('First {not json} then ' + nested + ' and {also not} after').answer == "set {a} and } then {"
    assert json_of('{"answer": "a \\" quote", "evidence": []}').answer == 'a " quote'


def test_the_last_complete_object_is_the_answer():
    one = '{"answer": "1", "evidence": []}'
    two = '{"answer": "2", "evidence": [{"source": "s", "t0": "2026-11-02T07:30:00+00:00", "t1": "2026-11-02T07:30:00+00:00"}]}'
    got = json_of(f"first {one} then {two}")
    assert got.answer == "2" and len(got.evidence) == 1
    assert json_of(f"{two} then {one}").answer == "1", "the last one, whatever is in the first"
    assert json_of(f'{one} then {{"answer": "3", "evidence": [').answer == "1", "an object that is not complete is text: the last complete one"
    assert json_of(f'{one} then {{"x": 1}}').answer is None, "the last complete object is the one that is read, even when it is not an answer"


# -- the json format: what it refuses ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("message, error", [
    ("", "the final message is empty"),
    ("  \n\t ", "the final message is empty"),
    ("the answer is 4", "the final message holds no complete JSON object"),
    ('{"answer": "4", "evidence": [', "the final message holds no complete JSON object"),
    ("[1, 2, 3]", "the final message holds no complete JSON object"),
    ('{"answer": 4, "evidence": []}', '"answer" must be a string, not 4'),
    ('{"answer": null, "evidence": []}', '"answer" must be a string, not None'),
    ('{"answer": ["4"], "evidence": []}', "\"answer\" must be a string, not ['4']"),
    ('{"evidence": []}', '"answer" must be a string, not None'),
    ('{"answer": "4"}', '"evidence" must be a list, not None'),
    ('{"answer": "4", "evidence": "none"}', '"evidence" must be a list, not \'none\''),
    ('{"answer": "4", "evidence": {"file": "a", "t0": 1, "t1": 2}}', '"evidence" must be a list, not {'),
])
def test_a_message_without_a_string_answer_and_a_list_of_evidence_is_a_failed_answer(message, error):
    got = json_of(message)
    assert got.answer is None and got.evidence == () and got.error.startswith(error), got.error


@pytest.mark.parametrize("citation, error", [
    ('"just a string"', "evidence[0] must be an object"),
    ('{"url": "x", "t0": 1, "t1": 2}', 'evidence[0] must name a "file" or a "source", and not both'),
    ('{"file": "a", "source": "b", "t0": 1, "t1": 2}', 'evidence[0] must name a "file" or a "source", and not both'),
    ('{"file": "", "t0": 1, "t1": 2}', 'evidence[0]: "file" must be a non-empty string'),
    ('{"file": 7, "t0": 1, "t1": 2}', 'evidence[0]: "file" must be a non-empty string, not 7'),
    ('{"source": "  ", "t0": "2026-11-02T07:30:00+00:00", "t1": "2026-11-02T07:30:00+00:00"}', 'evidence[0]: "source" must be a non-empty string'),
    ('{"file": "a"}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": 1}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": 1, "t1": "00:00:02"}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": true, "t1": 2}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": NaN, "t1": 2}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": 1, "t1": Infinity}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": -1, "t1": 2}', 'evidence[0]: "t0" and "t1" of a file must both be seconds'),
    ('{"file": "a", "t0": 5, "t1": 4}', 'evidence[0]: "t1" (4) is before "t0" (5)'),
    ('{"file": "a", "t0": "25:00", "t1": "26:00"}', 'evidence[0]: "t0" must be a clock time'),
    ('{"file": "a", "t0": "07:61", "t1": "07:62"}', 'evidence[0]: "t0" must be a clock time'),
    ('{"file": "a", "t0": "7pm", "t1": "8pm"}', 'evidence[0]: "t0" must be a clock time'),
    ('{"file": "a", "t0": "2026-11-02T07:30:00", "t1": "2026-11-02 07:30:61"}', 'evidence[0]: "t1" must be an ISO date and time'),
    ('{"file": "a", "t0": "07:30", "t1": "2026-11-02T07:30:05+00:00"}', 'evidence[0]: "t0" and "t1" of a file must both be clock times or both ISO'),
    ('{"file": "a", "t0": "2026-11-02T07:30:05+00:00", "t1": "2026-11-02T07:30:00+00:00"}', 'evidence[0]: "t1"'),
    ('{"source": "s", "t0": 1, "t1": 2}', 'evidence[0]: "t0" and "t1" of a source must be ISO times with an offset'),
    ('{"source": "s", "t0": "2026-11-02T07:30:00", "t1": "2026-11-02T07:30:05"}', 'evidence[0]: "t0" must be an ISO date and time with an offset'),
    ('{"source": "s", "t0": "2026-11-02", "t1": "2026-11-02"}', 'evidence[0]: "t0" must be an ISO date and time with an offset'),
    ('{"source": "s", "t0": "yesterday", "t1": "today"}', 'evidence[0]: "t0" must be an ISO date and time with an offset'),
    ('{"source": "s", "t0": "2026-11-02T07:30:05+00:00", "t1": "2026-11-02T07:30:00+00:00"}', 'evidence[0]: "t1"'),
])
def test_a_citation_of_an_unknown_shape_or_with_a_bad_time_makes_the_message_a_failed_answer(citation, error):
    """NaN, Infinity, a negative time, a time that ends before it starts, a clock time that is none, an ISO time with no offset, a file
    with a clock time and an ISO time and a source with seconds: each is refused whole, not skipped, so that a citation is never read as something it is not."""
    got = json_of('{"answer": "4", "evidence": [' + citation + "]}")
    assert got.answer is None and got.evidence == () and got.error.startswith(error), got.error


def test_the_second_citation_is_named_by_its_place_in_the_list():
    got = json_of('{"answer": "4", "evidence": [{"file": "a", "t0": 1, "t1": 2}, {"file": "a", "t0": -3, "t1": 2}]}')
    assert got.answer is None and got.error.startswith("evidence[1]: ")


def test_an_object_nested_too_deep_for_the_parser_is_no_object_and_not_a_crash():
    deep = "{" * 5000 + '"a": 1' + "}" * 5000
    got = json_of(deep + ' {"answer": "4", "evidence": []}')
    assert got.answer == "4", "the deep one is text, the last complete object is the answer"
    assert json_of(deep).error == '"answer" must be a string, not None', "(the object inside the deep ones is the one complete object)"


def test_a_clock_time_is_read_with_or_without_seconds_and_a_zero_before_the_hour():
    got = json_of('{"answer": "x", "evidence": [{"file": "a", "t0": "7:05", "t1": "07:05:30.5"}, {"file": "a", "t0": " 23:59:59 ", "t1": "0:00"}]}')
    assert got.evidence == (Citation("a", None, "7:05", "07:05:30.5"), Citation("a", None, "23:59:59", "0:00"))


# -- a law: a message made from known parts is read back as those parts ------------------------------------------------------------------

def test_a_message_made_of_an_object_between_any_prose_without_braces_is_read_back_as_the_object():
    """100 generated answers, each with 0 to 4 citations of the three forms, written with json.dumps (so with any characters, braces and
    quotes included in the answer), put between prose that holds no brace, in a fence or not: the answer and the citations come back."""
    rng = random.Random(20261105)
    letters = "abc {}[]\",:\\ \n\u00e9\u4e2d"
    for _ in range(100):
        cites, expected = [], []
        for _ in range(rng.randint(0, 4)):
            kind = rng.choice("fcs")
            a = round(rng.uniform(0, 5000), 2)
            b = round(a + rng.uniform(0, 60), 2)
            if kind == "f":
                cites.append({"file": f"cam{rng.randint(1, 3)}/f{rng.randint(1, 99)}.mp4", "t0": a, "t1": b})
            elif kind == "c":
                cites.append({"file": "cam/x.mp4", "t0": f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}", "t1": f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"})
            else:
                cites.append({"source": rng.choice(["porch", "mic", "screen"]), "t0": "2026-11-03T21:00:00-05:00", "t1": "2026-11-03T21:00:01.5-05:00"})
            expected.append(Citation(cites[-1].get("file"), cites[-1].get("source"), cites[-1]["t0"], cites[-1]["t1"]))
        answer = "".join(rng.choice(letters) for _ in range(rng.randint(0, 12)))
        body = json.dumps({"answer": answer, "evidence": cites}, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 1]))
        fence = rng.choice(["", "```json\n", "```\n"])
        message = f"{rng.choice(['', 'Done. ', 'Looked: [x] (y) '])}{fence}{body}{'```' if fence else ''}{rng.choice(['', ' Thanks.'])}"
        assert json_of(message) == FinalAnswer(answer, tuple(expected), None), message


# -- the sentence a prompt ends with -----------------------------------------------------------------------------------------------------

LINE_ENDS = [
    ("mm", False, "End with one line ANSWER: <answer>, or ANSWER: unknown when mm does not hold it."),  # the pilot's, for mm
    ("the files", True, "End with one line ANSWER: <answer>, or ANSWER: unknown when the files do not hold it."),  # the pilot's, for the files
    ("mm and these files", True, "End with one line ANSWER: <answer>, or ANSWER: unknown when mm and these files do not hold it."),
]
CITE = ('"evidence" lists each moment that shows the answer, as {"source": "<source>", "t0": "<start as an ISO time with its offset>", "t1": '
        '"<end as an ISO time with its offset>"}, or, for a file you read, as {"file": "<its path in the folder>", "t0": <seconds into the file, '
        'or a clock time>, "t1": <the same for the end>}')


@pytest.mark.parametrize("holder, plural, text", LINE_ENDS)
def test_the_line_request_is_the_words_the_pilot_recorded(holder, plural, text):
    assert answer_request("line", holder, plural=plural) == text


def test_the_line_request_of_a_choice_is_egolifes_recorded_last_sentence():
    assert answer_request("line", None, "<A, B, C or D>") == "End with one line ANSWER: <A, B, C or D>."


@pytest.mark.parametrize("holder, plural, none", [
    ("mm", False, 'Answer "unknown" with "evidence": [] when mm does not hold it.'),
    ("the files", True, 'Answer "unknown" with "evidence": [] when the files do not hold it.'),
    ("mm and these files", True, 'Answer "unknown" with "evidence": [] when mm and these files do not hold it.'),
])
def test_the_json_request_asks_for_one_object_with_the_evidence_and_says_when_the_answer_is_unknown(holder, plural, none):
    assert answer_request("json", holder, plural=plural) == (
        'End with one JSON object and nothing after it: {"answer": "<answer>", "evidence": [...]}. ' + CITE + ". " + none)


def test_the_json_request_of_a_choice_has_no_unknown_and_says_when_the_evidence_is_empty():
    assert answer_request("json", None, "<A, B, C or D>") == (
        'End with one JSON object and nothing after it: {"answer": "<A, B, C or D>", "evidence": [...]}. ' + CITE
        + '. "evidence" is [] when there is no moment to cite.')


# -- the prompts: each configuration is told what it has -----------------------------------------------------------------------------------

PLAIN = Served(None, None, (), None, None, None, {})
MM = Served(McpServer("mm", "mm", (), {}, ()), "You answer the user's question from their memory, using the mm tools.", (), None, "mm", None, {})
DATA = Path(r"D:\mmmb-temp\mmb-0123456789-data")


@pytest.fixture(scope="module")
def question():
    return tiny_questions()[0]


@pytest.mark.parametrize("access, fmt, ends", [
    (Access(False, None, MM, "line"), "line", LINE_ENDS[0][2]),
    (Access(True, DATA, PLAIN, "line"), "line", LINE_ENDS[1][2]),
    (Access(True, DATA, MM, "line"), "line", LINE_ENDS[2][2]),
    (Access(False, None, MM, "json"), "json", 'Answer "unknown" with "evidence": [] when mm does not hold it.'),
    (Access(True, DATA, PLAIN, "json"), "json", 'Answer "unknown" with "evidence": [] when the files do not hold it.'),
    (Access(True, DATA, MM, "json"), "json", 'Answer "unknown" with "evidence": [] when mm and these files do not hold it.'),
])
def test_a_v7_prompt_ends_with_the_request_for_its_format_that_names_what_the_agent_has(question, access, fmt, ends):
    """The mm and files configuration is told `unknown` when neither holds it: the old line said mm alone (the files may hold the answer)."""
    got = V7.prompt(question, access)
    assert got.endswith(ends) and (got.count("End with one JSON object") == 1) == (fmt == "json")
    assert f"{question.question} Answer from " in got, "the sentence before it is as it was"
    assert "ANSWER:" in got or fmt == "json"


@pytest.mark.parametrize("access, last", [
    (Access(False, None, PLAIN, "line"), "give your best guess. End with one line ANSWER: <A, B, C or D>."),
    (Access(False, None, PLAIN, "json"), "give your best guess. " + answer_request("json", None, "<A, B, C or D>")),
    (Access(False, None, MM, "json"), "give your best guess. " + answer_request("json", None, "<A, B, C or D>")),
])
def test_an_egolife_prompt_ends_with_the_guess_and_then_the_request_for_its_format(built, access, last):
    q = next(q for q in built[1] if q.qid == "ego-001")
    got = V8.prompt(q, access)
    assert got.endswith("\n\nIf what you can find does not settle it, " + last)
    assert ("evidence" in got) == (access.answer_format == "json")


def test_a_file_may_be_cited_with_iso_times_as_a_photo_is_and_they_are_kept_as_written():
    """A real agent cited a trail-camera photo by its file and the date and time of its EXIF, with an offset: that is an instant, as a
    source's times are, and it is read, not refused with the right answer (a run of 5 November 2026)."""
    message = ('A roadrunner came by.\n\n{"answer": "2013-12-22", "evidence": [{"file": "DCIM/100RECNX/IMG_0190.JPG", '
               '"t0": "2013-12-22T16:05:20-08:00", "t1": "2013-12-22T16:05:20-08:00"}]}')
    got = json_of(message)
    assert got == FinalAnswer("2013-12-22", (Citation("DCIM/100RECNX/IMG_0190.JPG", None, "2013-12-22T16:05:20-08:00",
                                                       "2013-12-22T16:05:20-08:00"),), None)


@pytest.mark.parametrize("t", ["2013-09-17T01:09:35", "2013-09-17 01:09:35", "2013-09-17T01:09:35-05:00"])
def test_a_file_cited_with_the_date_and_time_of_its_exif_with_or_without_an_offset_is_read(t):
    got = json_of(json.dumps({"answer": "2013-09-17", "evidence": [{"file": "DCIM/100RECNX/IMG_0003.JPG", "t0": t, "t1": t}]}))
    assert got.answer == "2013-09-17" and got.evidence == (Citation("DCIM/100RECNX/IMG_0003.JPG", None, t, t),), got.error
