"""New tests for grounded scoring (mmmb/core/grounded.py, DESIGN 6.3, plan 5.5): where a citation is in time, whether it hits the truth by the
kind of file the truth is in, the status beside the verdict, the event F1 of a count, the fields a result holds, and the report's figures.

Expected values are written by hand from DESIGN 6.3 and the plan's table, over a small dump made here by hand (a video, a sound, two
pictures, notes, versions of a config, two logs, a file that crosses midnight), not from the code: a video gap of exactly 10 s is accepted
and 10.01 s is not, a sound's 5 s and 5.01 s, a log line within 5 s of the same source, a picture only by being that file; the same moment
cited as seconds, as a clock time and as a source with ISO times is one absolute time, across midnight too; the event F1 tables are worked
out on paper. The last tests run the real run loop and the real report over the tiny week with stand-in agents whose final messages are
planted: right, right with evidence elsewhere, a wrong value, and a message that is no JSON."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timedelta

import pytest
from stub_system import FILES, STUB, Stub, right, tiny_questions
from test_report import FAKE as REPORT_FAKE
from test_report import FILES as REPORT_FILES
from test_report import run_of as report_run_of
from test_report import stub_out, tiny  # noqa: F401
from test_runner import go, world  # noqa: F401

from mmmb.core import fs, grade, render, report
from mmmb.core.agents.common import result_json  # noqa: F401
from mmmb.core.answer import Citation, FinalAnswer, parse_final
from mmmb.core.contract import Dump, ManifestEntry
from mmmb.core.grounded import Grounding, Placed, event_f1, ground, resolve, result_fields, status_of
from mmmb.core.question import Grade, Question, Scope, Span, Truth
from mmmb.core.record import record
from mmmb.core.runner import Agent
from mmmb.scenarios.v7_screen.manifest import entries_of
from mmmb.scenarios.v7_screen.week import plan as make_plan


def local(*parts) -> float:
    """Epoch seconds of a wall-clock time in this machine's zone, from the standard library (the zone the dump is written in)."""
    return datetime(*parts).timestamp()


T0 = local(2026, 11, 3, 12, 0, 0)  # (a day with no change of the clock in the zones the benchmark runs in)
NIGHT = local(2026, 11, 3, 23, 59, 0)


def entry(path, source, t, media, length=None) -> ManifestEntry:
    return ManifestEntry(path, source, t, "k", media, None, False, 1, 0, "0" * 64, length)


DUMP = Dump("k" * 64, "data", "v7_screen", (
    entry("cam/long.mp4", "cam", T0, "video", 3600.0),
    entry("other/long.mp4", "other", T0, "video", 3600.0),
    entry("cam/night.mp4", "cam", NIGHT, "video", 120.0),
    entry("mic/a1.wav", "mic", T0 + 3600, "audio", 600.0),
    entry("phone/p1.jpg", "phone", T0 + 7200, "image"),
    entry("phone/p2.jpg", "phone", T0 + 7300, "image"),
    entry("notes/n1.md", "notes", T0 + 10800, "text"),
    entry("notes/n2.md", "notes", T0 + 10900, "text"),
    entry("config/app-1.toml", "config", T0 + 11000, "text"),
    entry("config/app-2.toml", "config", T0 + 12000, "text"),
    entry("logs/web.jsonl", "logs", T0 + 14400, "records"),
    entry("logs/auth.jsonl", "logs", T0 + 14400, "records"),
), ())


def question(kind: str, *spans: Span) -> Question:
    """A question of this kind whose truth is in `spans`. Its value is graded `number` 5."""
    return Question("q", "T", "dev", "2026-11-04T12:00:00-05:00", "?", (), kind, "day", Truth("5", tuple(spans), ()),
                    Grade("absent" if kind == "absent" else "number", None if kind == "absent" else "5", None, None, (), (), None), {}, None,
                    Scope(("cam",), 0.0, 1.0), "g", None, (), None)


def said(*cites, answer: str = "5") -> FinalAnswer:
    return FinalAnswer(answer, tuple(cites), None)


def cf(file: str, t0, t1) -> Citation:
    return Citation(file, None, t0, t1)


def at(t: float) -> str:
    """The ISO time with the machine's offset of epoch second `t`."""
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def cs(source: str, t0: float, t1: float) -> Citation:
    """A source's citation of the epoch seconds t0..t1."""
    return Citation(None, source, at(t0), at(t1))


def status(q: Question, final: FinalAnswer, passed: bool = True) -> str:
    return ground(q, final, passed, DUMP).status


# -- the tolerances of DESIGN 6.3 ----------------------------------------------------------------------------------------------------------

VIDEO = question("find", Span("cam/long.mp4", 100.0, 110.0))
SOUND = question("find", Span("mic/a1.wav", 100.0, 110.0))
LINE = question("find", Span("logs/web.jsonl", 60.0, 60.0))


@pytest.mark.parametrize("a, b, expected", [
    (120.0, 130.0, "grounded"),  # 10 s after the span's end: the video's tolerance, accepted
    (120.01, 130.0, "lucky"),  # 10.01 s
    (60.0, 90.0, "grounded"),  # 10 s before its start
    (60.0, 89.99, "lucky"),  # 10.01 s
    (105.0, 106.0, "grounded"),  # inside
    (95.0, 125.0, "grounded"),  # around it
    (0.0, 3600.0, "grounded"),  # the whole file overlaps it (DESIGN says overlap; a broad citation is not penalised)
    (2000.0, 2010.0, "lucky"),
])
def test_a_citation_of_a_video_hits_within_ten_seconds_by_file_seconds(a, b, expected):
    assert status(VIDEO, said(cf("cam/long.mp4", a, b))) == expected


@pytest.mark.parametrize("a, b, expected", [
    (115.0, 120.0, "grounded"), (115.01, 120.0, "lucky"), (90.0, 95.0, "grounded"), (90.0, 94.99, "lucky"),
    (108.0, 125.0, "grounded"), (118.0, 125.0, "lucky"),
])
def test_a_citation_of_a_sound_hits_within_five_seconds(a, b, expected):
    assert status(SOUND, said(cf("mic/a1.wav", a, b))) == expected


@pytest.mark.parametrize("a, b, expected", [(65.0, 65.0, "grounded"), (65.01, 66.0, "lucky"), (55.0, 55.0, "grounded"), (54.99, 54.99, "lucky")])
def test_a_citation_of_a_line_hits_within_five_seconds_of_it_in_any_file_of_its_source(a, b, expected):
    """The truth is a line of web.jsonl; a log source holds several files, and a line of auth.jsonl at that time is the same source (DESIGN 6.3 corrected)."""
    assert status(LINE, said(cf("logs/auth.jsonl", a, b))) == expected
    assert status(LINE, said(cf("logs/web.jsonl", a, b))) == expected


def test_a_citation_of_the_same_time_in_another_source_hits_nothing_whatever_its_file():
    assert status(VIDEO, said(cf("other/long.mp4", 100.0, 110.0))) == "lucky"
    assert status(VIDEO, said(cs("other", T0 + 100, T0 + 110))) == "lucky"
    assert status(VIDEO, said(cs("cam", T0 + 100, T0 + 110))) == "grounded"
    assert status(VIDEO, said(cs("cam", T0 + 120, T0 + 130))) == "grounded"
    assert status(VIDEO, said(cs("cam", T0 + 120.01, T0 + 130))) == "lucky"


def test_a_picture_a_note_and_a_version_of_a_config_are_hit_only_by_being_that_file_or_by_its_time():
    picture = question("find", Span("phone/p1.jpg", 0.0, 0.0))
    assert status(picture, said(cf("phone/p1.jpg", 0.0, 0.0))) == "grounded"
    assert status(picture, said(cf("phone/p1.jpg", 30.0, 31.0))) == "grounded", "the file is the thing, not the seconds into it"
    assert status(picture, said(cf("phone/p2.jpg", 0.0, 0.0))) == "lucky", "the other picture, 100 s later"
    assert status(picture, said(cs("phone", T0 + 7200, T0 + 7200))) == "grounded"
    assert status(picture, said(cs("phone", T0 + 7199.5, T0 + 7200.5))) == "grounded"
    assert status(picture, said(cs("phone", T0 + 7230, T0 + 7231))) == "lucky"
    assert status(picture, said(cs("phone", T0 + 7300, T0 + 7300))) == "lucky", "the other picture's time"
    assert status(picture, said(cs("notes", T0 + 7200, T0 + 7200))) == "lucky", "the right time in another source"
    note = question("find", Span("notes/n1.md", 0.0, 0.0))
    assert status(note, said(cf("notes/n1.md", 0.0, 0.0))) == "grounded"
    assert status(note, said(cf("notes/n2.md", 0.0, 0.0))) == "lucky"
    assert status(note, said(cs("notes", T0 + 10800, T0 + 10800))) == "grounded"
    version = question("when", Span("config/app-1.toml", 500.0, 500.0))  # the value asked about as of 500 s after this version was written
    assert status(version, said(cf("config/app-1.toml", 0.0, 0.0))) == "grounded"
    assert status(version, said(cf("config/app-2.toml", 0.0, 0.0))) == "lucky"
    assert status(version, said(cs("config", T0 + 11000, T0 + 11000))) == "grounded", "cited by the version's own time"


# -- the table of statuses -----------------------------------------------------------------------------------------------------------------

RIGHT = cf("cam/long.mp4", 100.0, 110.0)
ELSEWHERE = cf("cam/long.mp4", 2000.0, 2010.0)
ABSENT = question("absent")


@pytest.mark.parametrize("final, passed, expected, cited, matched, unresolved", [
    (said(RIGHT), True, "grounded", 1, 1, 0),
    (said(RIGHT, RIGHT), True, "grounded", 2, 2, 0),
    (said(RIGHT, cf("cam/long.mp4", 105.0, 106.0)), True, "grounded", 2, 2, 0),
    (said(), True, "lucky", 0, 0, 0),  # right, and nothing cited
    (said(ELSEWHERE), True, "lucky", 1, 0, 0),  # right value, wrong place
    (said(RIGHT, ELSEWHERE), True, "lucky", 2, 1, 0),  # one citation out of place is enough
    (said(cf("cam/nothing.mp4", 0.0, 1.0)), True, "lucky", 1, 0, 1),  # a file the dump has not
    (said(Citation(None, "nosuch", at(T0), at(T0))), True, "lucky", 1, 0, 1),  # a source the dump has not
    (said(RIGHT, cf("cam/nothing.mp4", 0.0, 1.0)), True, "lucky", 2, 1, 1),
    (said(RIGHT), False, "wrong", 1, 1, 0),  # right evidence with a wrong value
    (said(), False, "wrong", 0, 0, 0),
    (said(ELSEWHERE), False, "wrong", 1, 0, 0),
])
def test_the_status_of_a_needle_question_is_told_by_the_value_and_the_evidence_as_6_3_says(final, passed, expected, cited, matched, unresolved):
    assert ground(VIDEO, final, passed, DUMP) == Grounding(expected, cited, matched, unresolved)


@pytest.mark.parametrize("final, passed, expected", [
    (said(answer="unknown"), True, "grounded"),  # right only with "not there" and no evidence
    (said(RIGHT, answer="unknown"), True, "wrong"),  # a citation for what is not there
    (said(ELSEWHERE, answer="unknown"), True, "wrong"),
    (said(answer="5"), False, "wrong"),
    (said(RIGHT, answer="5"), False, "wrong"),
])
def test_an_absent_question_is_grounded_only_by_the_right_value_with_no_evidence(final, passed, expected):
    assert ground(ABSENT, final, passed, DUMP).status == expected


def test_a_question_with_evidence_in_a_file_the_dump_lacks_or_with_none_at_all_is_an_error_not_a_status():
    with pytest.raises(ValueError, match="q: its evidence is in cam/gone.mp4, which the dump does not hold"):
        ground(question("find", Span("cam/gone.mp4", 0.0, 1.0)), said(), True, DUMP)
    with pytest.raises(ValueError, match="q is not an absent question and has no evidence"):
        ground(question("find"), said(RIGHT), True, DUMP)
    assert ground(question("find"), said(RIGHT), False, DUMP).status == "wrong", "(a value that fails is wrong without looking)"


def test_the_status_a_result_holds_is_read_and_a_result_that_holds_none_is_not_recorded():
    assert status_of({"grounded": "lucky"}) == "lucky" and status_of({"grounded": "grounded"}) == "grounded"
    assert status_of({"passed": True, "answer": "5"}) == "not_recorded", "a recorded result, or one of the line format"
    assert status_of({"answer_format": "line"}) == "not_recorded"


# -- one moment, three ways to cite it ----------------------------------------------------------------------------------------------------------

def test_an_offset_a_clock_time_and_a_source_with_iso_times_are_one_absolute_moment():
    by_offset = resolve(cf("cam/long.mp4", 1500.0, 1510.5), DUMP)
    by_clock = resolve(cf("cam/long.mp4", "12:25:00", "12:25:10.5"), DUMP)
    by_iso = resolve(cs("cam", T0 + 1500.0, T0 + 1510.5), DUMP)
    assert by_offset == Placed("cam", T0 + 1500.0, T0 + 1510.5, "cam/long.mp4")
    assert by_clock == by_offset
    assert (by_iso.source, by_iso.start, by_iso.end, by_iso.file) == ("cam", pytest.approx(T0 + 1500.0, abs=1e-3), pytest.approx(T0 + 1510.5, abs=1e-3), None)
    assert resolve(cf("cam/long.mp4", "12:25", "12:25:10"), DUMP) == Placed("cam", T0 + 1500.0, T0 + 1510.0, "cam/long.mp4")
    by_file_iso = resolve(cf("cam/long.mp4", at(T0 + 1500.0), at(T0 + 1510.5)), DUMP)
    assert (by_file_iso.start, by_file_iso.end, by_file_iso.file) == (pytest.approx(T0 + 1500.0, abs=1e-3), pytest.approx(T0 + 1510.5, abs=1e-3),
                                                                     "cam/long.mp4")


def test_a_photo_cited_by_its_file_with_an_iso_time_is_grounded_whatever_offset_the_agent_guessed():
    """As a real agent cited a trail-camera photo: the file with the time from its EXIF and an offset of its own guessing."""
    picture = question("find", Span("phone/p1.jpg", 0.0, 0.0))
    assert status(picture, said(cf("phone/p1.jpg", "2026-11-03T14:00:00-08:00", "2026-11-03T14:00:00-08:00"))) == "grounded"
    assert status(VIDEO, said(cf("cam/long.mp4", at(T0 + 2000.0), at(T0 + 2010.0)))) == "lucky", "a video's instant must still be near"
    naive = datetime.fromtimestamp(T0 + 1500.0).isoformat()  # with no offset: the dump's wall clock, as a clock time is
    assert resolve(cf("cam/long.mp4", naive, naive), DUMP) == Placed("cam", T0 + 1500.0, T0 + 1500.0, "cam/long.mp4")


def test_a_clock_time_across_midnight_is_on_the_next_day_and_one_before_the_file_starts_is_too():
    """cam/night.mp4 starts at 23:59:00 and plays 120 s: its second 30 is 23:59:30, its second 90 is 00:00:30 the next day."""
    assert resolve(cf("cam/night.mp4", "23:59:30", "23:59:50"), DUMP) == Placed("cam", NIGHT + 30, NIGHT + 50, "cam/night.mp4")
    after = resolve(cf("cam/night.mp4", "00:00:30", "00:00:50"), DUMP)
    assert after == Placed("cam", NIGHT + 90, NIGHT + 110, "cam/night.mp4"), "the same moments as 90 and 110 seconds into the file"
    assert after == resolve(cf("cam/night.mp4", 90.0, 110.0), DUMP)
    assert resolve(cs("cam", NIGHT + 90, NIGHT + 110), DUMP).start == pytest.approx(NIGHT + 90, abs=1e-3)
    across = resolve(cf("cam/night.mp4", "23:59:50", "00:00:10"), DUMP)
    assert across == Placed("cam", NIGHT + 50, NIGHT + 70, "cam/night.mp4"), "it ends after midnight: the first 00:00:10 after it starts"
    assert resolve(cf("cam/night.mp4", "23:58:00", "23:58:30"), DUMP).start == NIGHT + 86400 - 60, "a time before the file starts is a day later"
    assert resolve(cf("cam/night.mp4", "23:58:59.5", "23:59:10"), DUMP).start == NIGHT - 0.5, "(a clock time is told to the second: up to a second before)"
    q = question("find", Span("cam/night.mp4", 90.0, 110.0))
    assert status(q, said(cf("cam/night.mp4", "00:00:30", "00:00:50"))) == "grounded"
    assert status(q, said(cs("cam", NIGHT + 90, NIGHT + 110))) == "grounded"
    assert status(q, said(cf("cam/night.mp4", "23:59:30", "23:59:50"))) == "lucky", "60 s before it: not the moment"


def test_a_file_is_found_by_its_path_or_by_a_longer_path_that_ends_with_it_and_not_by_a_name_that_is_not_one():
    want = Placed("cam", T0 + 5, T0 + 6, "cam/long.mp4")
    assert resolve(cf("cam/long.mp4", 5.0, 6.0), DUMP) == want
    assert resolve(cf("cam\\long.mp4", 5.0, 6.0), DUMP) == want, "backslashes are slashes"
    assert resolve(cf(r"D:\mmmb-data\mmb-1-data\cam\long.mp4", 5.0, 6.0), DUMP) == want, "the path the agent read it by"
    assert resolve(cf("/work/view/cam/long.mp4", 5.0, 6.0), DUMP) == want
    assert resolve(cf("long.mp4", 5.0, 6.0), DUMP) is None, "a bare name is no path"
    assert resolve(cf("xcam/long.mp4", 5.0, 6.0), DUMP) is None, "a path that ends with the name but not after a folder"
    assert resolve(cf("cam/nothing.mp4", 5.0, 6.0), DUMP) is None
    assert resolve(Citation(None, "nosuch", at(T0), at(T0)), DUMP) is None


# -- event F1, worked out on paper ---------------------------------------------------------------------------------------------------------------

def p(t0: float, t1: float | None = None, source: str = "mic") -> Placed:
    return Placed(source, t0, t0 if t1 is None else t1, None)


TRUTH = [p(100), p(200), p(300)]


def close(got, expected) -> None:
    assert got == pytest.approx(expected)


def test_three_truths_two_hits_and_a_false_alarm_give_two_thirds_for_precision_and_recall():
    got = event_f1([p(100), p(200.5), p(500)], TRUTH, 1.0)
    close(got, (2 / 3, 2 / 3, 2 / 3))


def test_two_citations_of_one_moment_count_once_and_so_do_two_citations_that_reach_one_truth():
    close(event_f1([p(100), p(100.4), p(200)], TRUTH, 1.0), (1.0, 2 / 3, 0.8))  # P 2/2, R 2/3, F1 = 2*1*(2/3)/(5/3)
    close(event_f1([p(100), p(100.5), p(200), p(300)], TRUTH, 1.0), (1.0, 1.0, 1.0))
    close(event_f1([p(100), p(101.5)], [p(100)], 2.0), (1.0, 1.0, 1.0))  # (1.5 s from the truth, the collar is 2 s: both reach it)


def test_one_wide_citation_finds_one_moment_and_a_matching_gives_each_citation_its_own_where_it_can():
    close(event_f1([p(90, 310)], TRUTH, 5.0), (1.0, 1 / 3, 0.5))
    # A reaches both of the first two truths (collar 5), B only the first: B takes the first and A the second
    close(event_f1([p(102), p(98)], [p(100), p(104)], 5.0), (1.0, 1.0, 1.0))


def test_the_collar_decides_what_is_a_hit_and_the_source_must_be_the_same():
    cited = [p(103), p(204), p(299)]  # 3 s, 4 s and 1 s from the truths
    close(event_f1(cited, TRUTH, 1.0), (1 / 3, 1 / 3, 1 / 3))
    close(event_f1(cited, TRUTH, 5.0), (1.0, 1.0, 1.0))
    close(event_f1([p(100, source="cam")], TRUTH, 5.0), (0.0, 0.0, 0.0))
    close(event_f1([p(100, 104)], [p(110, 111)], 5.0), (0.0, 0.0, 0.0))  # 6 s apart
    close(event_f1([p(100, 105)], [p(110, 111)], 5.0), (1.0, 1.0, 1.0))  # exactly 5 s apart


def test_false_alarms_within_the_collar_of_each_other_are_one_and_a_citation_not_placed_is_a_false_alarm():
    close(event_f1([p(100), p(500), p(502)], TRUTH, 5.0), (0.5, 1 / 3, 0.4))  # P 1/2: one hit, one false alarm (500 and 502 are one)
    close(event_f1([p(100), p(500), p(502)], TRUTH, 1.0), (1 / 3, 1 / 3, 1 / 3))  # two false alarms at a 1 s collar
    close(event_f1([p(100), None], [p(100)], 1.0), (0.5, 1.0, 2 / 3))
    close(event_f1([None, None], [p(100)], 1.0), (0.0, 0.0, 0.0))


def test_nothing_to_find_and_nothing_cited_is_perfect_and_one_without_the_other_is_not():
    assert event_f1([], [], 1.0) == (1.0, 1.0, 1.0)
    assert event_f1([], TRUTH, 1.0) == (0.0, 0.0, 0.0)
    assert event_f1([p(100)], [], 1.0) == (0.0, 0.0, 0.0)
    assert event_f1([None], [], 1.0) == (0.0, 0.0, 0.0)


# -- the fields of a result ------------------------------------------------------------------------------------------------------------------------

COUNT = question("count", Span("mic/a1.wav", 100.0, 101.0), Span("mic/a1.wav", 200.0, 201.0), Span("mic/a1.wav", 300.0, 301.0))


def test_a_line_result_holds_only_its_format_and_a_json_result_holds_what_it_cited_and_how_it_grounds():
    assert result_fields(VIDEO, said(RIGHT), "line", True, DUMP) == {"answer_format": "line"}
    got = result_fields(VIDEO, said(RIGHT, cf("cam/nothing.mp4", 0.0, 1.0)), "json", True, DUMP)
    assert got == {"answer_format": "json", "evidence": [{"file": "cam/long.mp4", "t0": 100.0, "t1": 110.0},
                                                         {"file": "cam/nothing.mp4", "t0": 0.0, "t1": 1.0}],
                   "grounded": "lucky", "evidence_unresolved": 1}, "(no F1 for a question that is not a count)"
    assert result_fields(VIDEO, said(RIGHT), "json", False, DUMP)["grounded"] == "wrong"


def test_a_count_also_holds_its_event_f1_with_a_collar_of_one_second_and_one_of_five():
    near = [cf("mic/a1.wav", 100.0, 101.0), cf("mic/a1.wav", 203.0, 204.0), cf("mic/a1.wav", 500.0, 501.0)]
    got = result_fields(COUNT, said(*near, answer="3"), "json", True, DUMP)
    # 1 s collar: one hit, two false alarms, two misses: P 1/3, R 1/3. 5 s: two hits, one false alarm, one miss: P 2/3, R 2/3.
    assert got["grounded"] == "lucky"
    close((got["f1_1s"], got["f1_5s"]), (1 / 3, 2 / 3))


def test_a_message_that_could_not_be_read_is_a_failed_answer_whose_reason_the_result_keeps():
    unread = parse_final("ANSWER: 3", "json")
    got = result_fields(COUNT, unread, "json", False, DUMP)
    assert got == {"answer_format": "json", "evidence": [], "grounded": "wrong", "evidence_unresolved": 0,
                   "answer_error": "the final message holds no complete JSON object", "f1_1s": 0.0, "f1_5s": 0.0}
    assert "answer_error" not in result_fields(COUNT, said(answer="3"), "json", False, DUMP)


# -- the report: grounded accuracy beside the verdict ---------------------------------------------------------------------------------------------

def tiny_dump() -> Dump:
    """The dump of the tiny week as its manifest has it (made-up sizes: nothing reads the files)."""
    plan = make_plan(7, "tiny", [datetime(2026, 10, 30).date(), datetime(2026, 10, 31).date()])
    return Dump("k" * 64, "data", "v7_screen", entries_of(plan, lambda path: (1, 0, "0" * 64)), ())


def cited(q: Question, answer: str | None = None, *, extra: list | None = None) -> str:
    """The final message of an agent that answers `q` right (or with `answer`) and cites the truth's own spans."""
    evidence = [{"file": s.file, "t0": s.t0, "t1": s.t1} for s in q.truth.evidence]
    return json.dumps({"answer": answer or right(q), "evidence": evidence + (extra or [])})


def json_result(config, q: Question, message: str, dump: Dump) -> dict:
    """The result `record` and `result_fields` make of an exec of `config` that ended with `message`, as the run loop makes them."""
    final = parse_final(message, "json")
    rec = record(q, "claude", config, 7, f"execs/claude/{config.dir}/{q.qid}", "2026-11-02T12:00:00",
                 {**stub_out(config, q), "final": message, "answer": final.answer})
    return {**rec, **result_fields(q, final, "json", rec["passed"], dump)}


def line_result(config, q: Question) -> dict:
    """A result of the line format, as a recorded run has it: the verdict and no status."""
    return record(q, "claude", config, 7, f"execs/claude/{config.dir}/{q.qid}", "2026-11-02T12:00:00", stub_out(config, q))


def planted_for(tiny, dump: Dump) -> tuple[dict, tuple]:
    """The four finals that are not right and cited right: a right value cited elsewhere (`lucky`), a wrong value with the right
    evidence, a message that is no JSON, and an absent question answered `unknown` with a citation; and the questions they are for."""
    needles, absent = [q for q in tiny if q.kind != "absent"], [q for q in tiny if q.kind == "absent"]
    lucky, wrong, unread = needles[0], needles[1], needles[2]
    elsewhere = next(e.path for e in dump.entries if e.media == "image" and e.path not in {s.file for s in lucky.truth.evidence})
    planted = {lucky.qid: cited(lucky, extra=[{"file": elsewhere, "t0": 0.0, "t1": 0.0}]), wrong.qid: cited(wrong, "-1"),
               unread.qid: f"ANSWER: {right(unread)}",
               absent[0].qid: json.dumps({"answer": "unknown", "evidence": [{"file": dump.entries[0].path, "t0": 0.0, "t1": 1.0}]})}
    return planted, (lucky, wrong, unread, absent[0])


def test_the_report_tells_grounded_accuracy_lucky_and_unparsed_beside_the_verdict(tiny, tmp_path):
    """The tested side's results are all right and cite the truth, but four (planted_for). The files side is of the `line` format."""
    dump = tiny_dump()
    planted, (lucky, wrong, unread, absent) = planted_for(tiny, dump)
    run = report_run_of(tmp_path / "run", tiny)
    for q in tiny:
        fs.write_json(run.result_path("claude", REPORT_FILES, q.qid), line_result(REPORT_FILES, q))
        fs.write_json(run.result_path("claude", REPORT_FAKE, q.qid), json_result(REPORT_FAKE, q, planted.get(q.qid) or cited(q), dump))
    n = len(tiny)
    results = {q.qid: fs.read_json(run.result_path("claude", REPORT_FAKE, q.qid)) for q in tiny}
    assert {qid: r["grounded"] for qid, r in results.items() if r["grounded"] != "grounded"} == {
        lucky.qid: "lucky", wrong.qid: "wrong", unread.qid: "wrong", absent.qid: "wrong"}, "the other 15 are grounded"
    assert results[unread.qid]["answer"] is None and results[unread.qid]["answer_error"] == "the final message holds no complete JSON object"
    assert [results[q.qid]["passed"] for q in (lucky, wrong, unread, absent)] == [True, False, False, True], "the verdict is the value rule alone"

    figures = report.numbers([run])
    all_ = figures["agents"]["claude"]["mm/fake"]["all"]
    assert all_["tested"]["grounding"] == {"grounded": n - 4, "lucky": 1, "wrong": 3, "not_recorded": 0, "unparsed": 1,
                                           "accuracy": pytest.approx((n - 4) / n)}
    assert all_["tested"]["correct"] == n - 2 and "grounding" not in all_["control"], "the verdict is as it was, and the line side tells nothing of it"
    text = render.markdown(figures)
    assert "| grounded (files / mm) | lucky (files / mm) | unparsed (files / mm) |" in text
    assert f"| - / {n - 4}/{n} ({100 * (n - 4) / n:.0f}%) | - / 1 | - / 1 |" in text
    assert "Grounded: the answer is right and the evidence it cites is at the moment" in text


def test_results_with_no_status_among_results_with_one_are_not_recorded_and_left_out_of_the_accuracy(tiny, tmp_path):
    dump = tiny_dump()
    run = report_run_of(tmp_path / "run", tiny)
    older = {q.qid for q in tiny[:5]}  # five results of the tested side are of the line format (as if recorded before)
    for q in tiny:
        fs.write_json(run.result_path("claude", REPORT_FILES, q.qid), line_result(REPORT_FILES, q))
        fs.write_json(run.result_path("claude", REPORT_FAKE, q.qid),
                      line_result(REPORT_FAKE, q) if q.qid in older else json_result(REPORT_FAKE, q, cited(q), dump))
    got = report.numbers([run])["agents"]["claude"]["mm/fake"]["all"]["tested"]["grounding"]
    assert got == {"grounded": len(tiny) - 5, "lucky": 0, "wrong": 0, "not_recorded": 5, "unparsed": 0, "accuracy": 1.0}


def test_a_report_of_results_that_hold_no_status_tells_nothing_of_grounding(tiny, tmp_path):
    """Recorded results, and results of the line format, have no status: the figures hold no `grounding` and the Markdown no word of it."""
    run = report_run_of(tmp_path / "run", tiny)
    for config in (REPORT_FILES, REPORT_FAKE):
        for q in tiny:
            fs.write_json(run.result_path("claude", config, q.qid), {
                **line_result(config, q), **result_fields(q, parse_final(f"ANSWER: {right(q)}", "line"), "line", True, tiny_dump())})
    figures = report.numbers([run])
    assert "grounding" not in json.dumps(figures)
    text = render.markdown(figures)
    assert "grounded" not in text.lower() and "unparsed" not in text and "lucky" not in text


# -- through the run loop -------------------------------------------------------------------------------------------------------------------------

def test_the_evidence_a_stand_in_agent_cites_reaches_the_result_and_the_report_through_the_run_loop(world, tiny):
    """The real `run_agents` over the tiny week, with the files configuration and a stand-in system's, in the json format: a stand-in agent
    ends with a message that cites the truth's own spans, but for four planted ones. Each result holds the status the plant makes."""
    w, dump = world, tiny_dump()
    w.dump = dataclasses.replace(w.dump, entries=dump.entries)
    planted, (lucky, wrong, unread, absent) = planted_for(tiny, dump)
    stub = Stub(by={(folder, q.qid): {"final": planted.get(q.qid) or cited(q)} for folder in ("files", "stub-all") for q in tiny})
    go(w, {"claude": Agent(stub)}, answer_format="json")
    assert "End with one JSON object" in stub.execs[0].prompt and len(stub.execs) == 2 * len(tiny)
    counts = [q for q in tiny if q.kind == "count"]
    assert counts, "the tiny week has counts"
    for folder in ("files", "stub-all"):
        got = {q.qid: fs.read_json(w.run.folder / "results" / "claude" / folder / f"{q.qid}.json") for q in tiny}
        want = {q.qid: (right(q), True, "grounded", 0) for q in tiny}
        want.update({lucky.qid: (right(lucky), True, "lucky", 0), wrong.qid: ("-1", False, "wrong", 0), unread.qid: (None, False, "wrong", 0),
                     absent.qid: ("unknown", True, "wrong", 0)})
        assert {qid: (r["answer"], r["passed"], r["grounded"], r["evidence_unresolved"]) for qid, r in got.items()} == want, folder
        assert all(r["answer_format"] == "json" for r in got.values())
        assert got[unread.qid]["answer_error"] == "the final message holds no complete JSON object"
        assert not any("answer_error" in got[q.qid] for q in tiny if q != unread)
        assert got[lucky.qid]["evidence"][-1]["t0"] == 0.0 and len(got[lucky.qid]["evidence"]) == len(lucky.truth.evidence) + 1
        assert all(got[q.qid]["f1_1s"] == got[q.qid]["f1_5s"] == 1.0 for q in counts if q.qid not in planted), "the truth's own moments: perfect"
        assert {q.qid for q in tiny if "f1_1s" in got[q.qid]} == {q.qid for q in counts}, "a count only"
    fs.write_json(w.run.folder / "run.json", {"mmmb_commit": "abc123"})
    both = report.numbers([w.run])["agents"]["claude"]["stub/all"]["all"]
    n = len(tiny)
    assert both["tested"]["grounding"] == both["control"]["grounding"] == {
        "grounded": n - 4, "lucky": 1, "wrong": 3, "not_recorded": 0, "unparsed": 1, "accuracy": pytest.approx((n - 4) / n)}
