"""The transcripts and the question templates of the speech scenarios (scenarios/realdump/speech.py)."""

from __future__ import annotations

from datetime import datetime

import pytest

from mmsb.core.grade import grade
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump.speech import Cue, Curated, Say, Talk, Term, mentions, questions_of, read_srt

SRT = """1
00:00:01,000 --> 00:00:03,000
[SQUEAKING]

2
00:00:04,500 --> 00:00:06,250
SARAH HANSEN: Today on
the podcast,

3
00:00:06,250 --> 00:00:09,000
we talk about ionizing
radiation.

4
00:00:10,000 --> 00:00:12,000
MIKE SHORT: Thanks for having me.
"""

SAY = Say("X9", "lectures", "video", "the lecturer", "lecture", "lectures")


def cue(t: float, text: str) -> Cue:
    return Cue(t, t + 3, text)


def talks() -> list[Talk]:
    a = Talk("L/lecture_a.mp4", datetime(2024, 9, 3, 10), 3000.0, (cue(10, "welcome"), cue(300, "the knapsack"), cue(303, "problem is hard"),
                                                                    cue(900, "greedy algorithms and regression"), cue(1500, "Monte Carlo simulation")))
    b = Talk("L/lecture_b.mp4", datetime(2024, 9, 5, 10), 3600.0, (cue(200, "regression again"), cue(1000, "greedy algorithm once more"),
                                                                    cue(2000, "bootstrapping"), cue(2400, "hypothesis testing")))
    c = Talk("L/lecture_c.mp4", datetime(2024, 9, 10, 10), 2700.0, (cue(100, "linear programming"),))
    return [a, b, c]


TERMS = [Term("knapsack problem", "the knapsack problem"), Term("greedy algorithm", "greedy algorithms"), Term("regression", "regression"),
         Term("Monte Carlo", "Monte Carlo simulation"), Term("bootstrapping", "bootstrapping"), Term("hypothesis testing", "hypothesis testing")]
NEVER = [Term("Higgs boson", "the Higgs boson")]
T_OF = realdump.local_t


def test_read_srt_drops_notes_of_sounds_and_carries_the_speaker():
    cues = read_srt(SRT)
    assert [c.text for c in cues] == ["Today on the podcast,", "we talk about ionizing radiation.", "Thanks for having me."]
    assert [c.speaker for c in cues] == ["Sarah Hansen", "Sarah Hansen", "Mike Short"]
    assert cues[0].t0 == 4.5 and cues[0].t1 == 6.25


def test_a_phrase_may_run_over_two_cues_and_a_last_word_may_be_plural():
    a = talks()[0]
    assert [c.t0 for c in mentions(a, "knapsack problem")] == [300]
    assert [c.t0 for c in mentions(a, "greedy algorithm")] == [900]
    assert not mentions(a, "knapsack hard")


def test_the_templates_answer_from_the_transcripts():
    qs = {q.tags["template"]: q for q in questions_of(talks(), TERMS, NEVER, SAY, {"first": 1, "which": 1, "count": 1, "absent_did": 1}, 1, T_OF)}
    assert grade(qs["absent_did"].grade, "No, never mentioned.")
    assert qs["absent_did"].kind == "absent" and not qs["absent_did"].truth.evidence
    assert qs["count"].grade.truth == "2"
    assert qs["which"].grade.items[0].endswith(".mp4")
    assert qs["first"].grade.rule == "time" and qs["first"].grade.date


def test_a_term_is_asked_about_once():
    qs = questions_of(talks(), TERMS, NEVER, SAY, {"first": 2, "did": 2, "which": 2, "count": 2, "all": 2}, 3, T_OF)
    words = [q.qid.split("-", 1)[1] for q in qs]
    assert len(words) == len(set(words))


def test_an_order_question_takes_terms_said_once_and_far_apart():
    qs = [q for q in questions_of(talks(), TERMS, NEVER, SAY, {"order": 1}, 2, T_OF) if q.tags["template"] == "order"]
    assert len(qs) == 1
    q = qs[0]
    assert q.grade.rule == "text" and q.grade.truth in ("before", "after") and len(q.truth.evidence) == 2
    assert grade(q.grade, q.grade.truth) and not grade(q.grade, q.grade.wrong[0])


def test_an_order_question_leaves_out_a_term_said_in_another_file_of_that_day():
    day = datetime(2024, 9, 3, 10)
    one = Talk("L/a.mp4", day, 3000.0, (cue(300, "regression"), cue(900, "bootstrapping"), cue(1500, "hypothesis testing")))
    two = Talk("L/b.mp4", day.replace(hour=11), 3000.0, (cue(300, "bootstrapping"),))
    qs = [q for q in questions_of([one, two], TERMS, NEVER, SAY, {"order": 3}, 1, T_OF) if q.tags["template"] == "order"]
    assert len(qs) == 1 and "bootstrapping" not in qs[0].question


def test_long_gives_the_length_of_the_talk_in_minutes():
    q = next(q for q in questions_of(talks(), TERMS, NEVER, SAY, {"long": 1}, 2, T_OF) if q.tags["template"] == "long")
    assert q.grade.rule == "amount" and q.grade.tol_s == 1.0 and float(q.grade.truth) in (50.0, 60.0)


def test_a_term_that_was_said_is_refused_as_absent():
    with pytest.raises(ValueError, match="said after all"):
        questions_of(talks(), TERMS, [Term("regression", "regression")], SAY, {"absent_did": 1}, 1, T_OF)


def test_a_curated_question_must_rest_on_what_was_said():
    ok = Curated("c", "What did he call it?", "what", "<a name>", realdump.grade("text", truth="knapsack"), 0, 300, 305, "knapsack problem", "knapsack")
    assert any(q.tags["template"] == "what" for q in questions_of(talks(), TERMS, NEVER, SAY, {}, 1, T_OF, [ok]))
    bad = Curated("c", "?", "what", "<a>", realdump.grade("text", truth="x"), 0, 1000, 1005, "knapsack problem", "x")
    with pytest.raises(ValueError, match="curated question c"):
        questions_of(talks(), TERMS, NEVER, SAY, {}, 1, T_OF, [bad])
