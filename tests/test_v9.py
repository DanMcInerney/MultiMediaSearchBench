"""V9, the lecture recordings, from its real source on disk (MIT 14.01): the plan at both scales with every answer derived again by a plain reader of the
SRTs (tests/speech_check.py, never the scenario's), the curated questions checked against the captions, the lectures' days read again from the MP4s'
own tags, the grader taking each truth and refusing planted wrong answers, the tiny pile built (video copied with none of the source's tags, as long as the
manifest says) and its leak probe. Skips, and says so, without the source."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import av
import imageio_ffmpeg
import pytest
from speech_check import check_notes, check_plan, check_questions, check_terms

from mmsb.core.grade import grade
from mmsb.scenarios import v9_lectures as V9
from mmsb.scenarios.v9_lectures.questions import NEVER, NOTES, TERMS
from mmsb.scenarios.v9_lectures.source import CLASS_DAYS, LECTURES, SOURCE, stem_of

if not all((SOURCE / f"{stem_of(n)}{ext}").is_file() for n in LECTURES for ext in (".mp4", ".srt")):
    pytest.skip(f"the 26 lectures of MIT 14.01 (MP4 and SRT) are not at {SOURCE}, so V9 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("v9")
NAMES = r"Lectures/lecture_2023-\d\d-\d\d\.mp4"


def srt_of(f) -> Path:
    return SOURCE / str(f.spec["src"]).replace(".mp4", ".srt")


@pytest.fixture(scope="module")
def plans():
    return {seed: V9.plan_of(V9.recipe(scale="standard", seed=seed)) for seed in (1, 2)}


def test_the_standard_plan_is_a_term_of_lectures_and_about_two_dozen_questions(plans):
    for files, questions in plans.values():
        check_plan(V9, files, questions, 20, 26, NAMES)
        assert len(files) == 26 and 20 < sum(f.length for f in files) / 3600 < 21
        assert {q.tags["template"] for q in questions} >= {"first", "did", "which", "count", "long", "order", "what", "before",
                                                           "absent_first", "absent_did", "absent_which"}


def test_every_answer_is_what_a_plain_reader_of_the_captions_finds(plans):
    for files, questions in plans.values():
        check_questions(questions, files, srt_of, TERMS, NEVER)
        check_notes(NOTES, files, srt_of, [str(f.spec["src"]) for f in files])


def test_the_terms_are_said_in_the_captions_and_the_terms_never_said_are_said_in_none():
    check_terms(TERMS, NEVER, [SOURCE / f"{stem_of(n)}.srt" for n in LECTURES])


def test_a_seed_draws_other_questions_from_the_same_pile(plans):
    (files_1, q_1), (files_2, q_2) = plans[1], plans[2]
    assert files_1 == files_2 and {q.qid for q in q_1} != {q.qid for q in q_2}


def test_the_lectures_are_timed_to_their_own_days_and_the_hour_of_a_laptop_beginning_to_record(plans):
    """The days are read again from the MP4s' date tags (each a Monday or a Wednesday): a lecture's day is its tag's."""
    files, _ = plans[1]
    assert len(CLASS_DAYS) == 26 and len(set(CLASS_DAYS)) == 26
    for f in files:
        with av.open(str(SOURCE / str(f.spec["src"]))) as c:
            tag = datetime.fromisoformat(c.metadata["date"].removesuffix("Z")).date()
        when = datetime.fromtimestamp(f.t)
        assert when.date() == tag and tag.weekday() in (0, 2), (f.path, tag)
        assert (11, 5) <= (when.hour, when.minute) <= (11, 8), f.path
        assert f.path == f"Lectures/lecture_{when:%Y-%m-%d}.mp4"
    assert [datetime.fromtimestamp(f.t).date().isoformat() for f in files[:3]] == ["2023-09-06", "2023-09-11", "2023-09-13"]


def test_no_time_answer_is_the_time_of_a_file_alone(plans):
    """The listing of files and times gives no `first` or `did` answer: none is within a minute of a file's start."""
    for files, questions in plans.values():
        for q in questions:
            if q.grade.rule == "time":
                t = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}").timestamp()
                assert min(abs(t - f.t) for f in files) > 60, q.qid


def test_a_curated_answer_is_graded_and_a_planted_wrong_one_is_not():
    by_key = {n.key: n for n in NOTES}
    bid = by_key["kidney-bid"].grade
    assert grade(bid, "$25,000") and grade(bid, "25000") and not grade(bid, "$5 million") and not grade(bid, "$25,000 to $5,000,000")
    assert grade(by_key["cereal"].grade, "Apple Cinnamon Cheerios") and not grade(by_key["cereal"].grade, "Frosted Flakes")
    assert grade(by_key["cereal-types"].grade, "three") and grade(by_key["cereal-types"].grade, "3") and not grade(by_key["cereal-types"].grade, "150")
    assert grade(by_key["twenty-years"].grade, "20 years") and not grade(by_key["twenty-years"].grade, "10 years")
    after = by_key["after-cheerios"].grade
    assert grade(after, "Cornflakes") and grade(after, "Corn flakes") and not grade(after, "Quaker Oats")
    assert grade(after, "Cornflakes, not Quaker Oats") and not grade(after, "Quaker Oats, not Cornflakes") and not grade(after, "Cornflakes or Quaker Oats")
    assert grade(by_key["positive-first"].grade, "The positive one") and not grade(by_key["positive-first"].grade, "the normative one")


def test_the_tiny_pile_builds_as_video_with_none_of_the_sources_tags_that_plays_as_long_as_it_says_and_passes_the_probe(tmp_path):
    recipe = V9.recipe(scale="tiny", seed=1)
    files, questions = V9.plan_of(recipe)
    assert len(files) == 3 and 2 <= len(questions) <= 8
    dump, built = V9.build(recipe, tmp_path / "dump", 3)
    assert built == questions and len(dump.entries) == 3
    for e in dump.entries:
        path = tmp_path / "dump" / e.path
        assert os.stat(path).st_mtime == e.t and e.size == path.stat().st_size and e.media == "video"
        with av.open(str(path)) as c:
            assert c.streams.video and c.streams.audio and abs(c.duration / av.time_base - e.length) < 0.5
            assert {k: v for k, v in c.metadata.items() if k not in ("major_brand", "minor_version", "compatible_brands")} == {}
    out = tmp_path / "probe"
    out.mkdir()
    assert V9.probe(recipe, dump, out, Path(imageio_ffmpeg.get_ffmpeg_exe()), 1).status == "passed"


def test_a_recipe_of_another_time_zone_or_with_other_keys_is_refused():
    recipe = V9.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="keys"):
        V9.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="UTC offsets"):
        V9.plan_of({**recipe, "utc_offsets": ["+1234", "+1234"]})
    with pytest.raises(ValueError, match="scale"):
        V9.recipe(scale="huge", seed=1)
