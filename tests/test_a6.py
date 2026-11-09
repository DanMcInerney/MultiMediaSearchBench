"""A6, the lecture audio, from its real source on disk (MIT 6.0002): the plan at both scales with every answer derived again by a plain reader of the
SRTs (tests/speech_check.py, never the scenario's), the curated questions checked against the captions, the lectures' days read again from the MP4s' own
tags, the grader taking each truth and refusing planted wrong answers, the tiny pile built (sound only, no tag of the source's, as long as the manifest
says) and its leak probe. Skips, and says so, without the source."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import av
import imageio_ffmpeg
import pytest
from speech_check import check_notes, check_plan, check_questions, check_terms

from mmsb.core.grade import grade
from mmsb.scenarios import a6_lecture_audio as A6
from mmsb.scenarios.a6_lecture_audio.questions import NEVER, NOTES, TERMS
from mmsb.scenarios.a6_lecture_audio.source import SOURCE, stem_of

LECTURES = range(1, 16)
if not all((SOURCE / f"{stem_of(n)}{ext}").is_file() for n in LECTURES for ext in (".mp4", ".srt")):
    pytest.skip(f"the 15 lectures of MIT 6.0002 (MP4 and SRT) are not at {SOURCE}, so A6 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("a6")
NAMES = r"Recordings/Recording_\d{8}_\d{6}\.m4a"


def srt_of(f) -> Path:
    return SOURCE / str(f.spec["src"]).replace(".mp4", ".srt")


@pytest.fixture(scope="module")
def plans():
    return {seed: A6.plan_of(A6.recipe(scale="standard", seed=seed)) for seed in (1, 2)}


def test_the_standard_plan_is_a_term_of_recordings_and_about_two_dozen_questions(plans):
    for files, questions in plans.values():
        check_plan(A6, files, questions, 20, 26, NAMES)
        assert len(files) == 15 and 11 < sum(f.length for f in files) / 3600 < 13
        assert {q.tags["template"] for q in questions} >= {"first", "did", "which", "count", "order", "what", "before", "absent_first", "absent_did", "absent_which"}


def test_every_answer_is_what_a_plain_reader_of_the_captions_finds(plans):
    for files, questions in plans.values():
        check_questions(questions, files, srt_of, TERMS, NEVER)
        check_notes(NOTES, files, srt_of, [str(f.spec["src"]) for f in files])


def test_the_terms_are_said_in_the_captions_and_the_terms_never_said_are_said_in_none():
    check_terms(TERMS, NEVER, [SOURCE / f"{stem_of(n)}.srt" for n in LECTURES])


def test_a_seed_draws_other_questions_from_the_same_pile(plans):
    (files_1, q_1), (files_2, q_2) = plans[1], plans[2]
    assert files_1 == files_2
    assert {q.qid for q in q_1} != {q.qid for q in q_2} and {q.qid for q in q_1} & {q.qid for q in q_2}  # the written ones stay


def test_the_recordings_are_timed_to_the_lectures_own_days_and_the_hour_of_a_phone_pressing_record(plans):
    """The days are read again from the MP4s' date tags: a lecture's day is its tag's day, or the Monday or Wednesday after."""
    files, _ = plans[1]
    for f in files:
        with av.open(str(SOURCE / str(f.spec["src"]))) as c:
            tag = datetime.fromisoformat(c.metadata["date"].removesuffix("Z")).date()
        when = datetime.fromtimestamp(f.t)
        assert 0 <= (when.date() - tag).days <= 1, (f.path, tag)
        assert when.weekday() in (0, 2) and (14, 5) <= (when.hour, when.minute) <= (14, 8), f.path
        assert f.path == f"Recordings/Recording_{when:%Y%m%d_%H%M%S}.m4a"
    assert [datetime.fromtimestamp(f.t).date().weekday() for f in files[:4]] == [0, 2, 0, 2]


def test_no_time_answer_is_the_time_of_a_file_alone(plans):
    """The listing of files and times gives no `first` or `did` answer: none is within a minute of a file's start."""
    for files, questions in plans.values():
        for q in questions:
            if q.grade.rule == "time":
                t = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}").timestamp()
                assert min(abs(t - f.t) for f in files) > 60, q.qid


def test_a_curated_answer_is_graded_and_a_planted_wrong_one_is_not():
    by_key = {n.key: n for n in NOTES}
    assert grade(by_key["finger-exercises"].grade, "15 minutes") and grade(by_key["finger-exercises"].grade, "15")
    assert not grade(by_key["finger-exercises"].grade, "10 minutes") and not grade(by_key["finger-exercises"].grade, "15 or 20 minutes")
    assert grade(by_key["first-reptile"].grade, "A cobra") and not grade(by_key["first-reptile"].grade, "a boa constrictor")
    assert grade(by_key["titanic-died"].grade, "812") and not grade(by_key["titanic-died"].grade, "1,300") and not grade(by_key["titanic-died"].grade, "813")
    assert grade(by_key["drunk-steps"].grade, "1.1 steps") and not grade(by_key["drunk-steps"].grade, "0.9")
    assert grade(by_key["birthday-367"].grade, "367") and not grade(by_key["birthday-367"].grade, "366")
    assert grade(by_key["models-first"].grade, "Optimization models") and not grade(by_key["models-first"].grade, "simulation")


def test_the_tiny_pile_builds_as_sound_alone_that_plays_as_long_as_it_says_and_passes_the_probe(tmp_path):
    recipe = A6.recipe(scale="tiny", seed=1)
    files, questions = A6.plan_of(recipe)
    assert len(files) == 3 and 2 <= len(questions) <= 8
    dump, built = A6.build(recipe, tmp_path / "dump", 3)
    assert built == questions
    for e in dump.entries:
        path = tmp_path / "dump" / e.path
        assert os.stat(path).st_mtime == e.t and e.size == path.stat().st_size and e.media == "audio"
        with av.open(str(path)) as c:
            assert not c.streams.video and c.streams.audio[0].codec_context.name == "aac"
            assert abs(c.streams.audio[0].duration * c.streams.audio[0].time_base - e.length) < 0.5
            assert {k: v for k, v in c.metadata.items() if k not in ("major_brand", "minor_version", "compatible_brands")} == {}
    out = tmp_path / "probe"
    out.mkdir()
    assert A6.probe(recipe, dump, out, Path(imageio_ffmpeg.get_ffmpeg_exe()), 1).status == "passed"


def test_a_recipe_of_another_time_zone_or_with_other_keys_is_refused():
    recipe = A6.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="keys"):
        A6.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="UTC offsets"):
        A6.plan_of({**recipe, "utc_offsets": ["+1234", "+1234"]})
    with pytest.raises(ValueError, match="scale"):
        A6.recipe(scale="huge", seed=1)
