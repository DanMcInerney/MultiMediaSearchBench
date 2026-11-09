"""A7, the podcast backlog, from its real source on disk (Chalk Radio): the plan at both scales with every answer derived again by a plain reader of the
SRTs (tests/speech_check.py, never the scenario's), the curated questions checked against the captions, the `who` answers checked against the speakers
the captions name and the artists the MP4s' own tags give, the episodes' days read again from the tags, the grader taking each truth and refusing planted
wrong answers, the tiny pile built (MP3 sound alone, no tag of the source's, as long as the manifest says) and its leak probe. Skips, and says so,
without the source."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import av
import imageio_ffmpeg
import pytest
from speech_check import blocks, check_notes, check_plan, check_questions, check_terms, hits

from mmsb.core.grade import grade
from mmsb.scenarios import a7_podcast as A7
from mmsb.scenarios.a7_podcast.questions import GUESTS, HOST, NEVER, NOTES, TERMS
from mmsb.scenarios.a7_podcast.source import EPISODE, EPISODES, SOURCE

EPISODE_FILES = sorted(p for p in SOURCE.glob("*.mp4") if EPISODE.match(p.name)) if SOURCE.is_dir() else []
if len(EPISODE_FILES) != EPISODES or not all(p.with_suffix(".srt").is_file() for p in EPISODE_FILES):
    pytest.skip(f"the {EPISODES} numbered episodes of Chalk Radio (MP4 and SRT) are not at {SOURCE}, so A7 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("a7")
NAMES = r"Podcasts/episode_\d\d\.mp3"
NOT_PEOPLE = {HOST.casefold(), "amazon alexa", "audio bumper", "brett the producer", "child", "crew", "instagram", "little girl", "parent", "producer", "quote",
              "capozzola"}


def srt_of(f) -> Path:
    return SOURCE / str(f.spec["src"]).replace(".mp4", ".srt")


@pytest.fixture(scope="module")
def plans():
    return {seed: A7.plan_of(A7.recipe(scale="standard", seed=seed)) for seed in (1, 2)}


def test_the_standard_plan_is_the_backlog_in_the_order_the_episodes_came_out_and_about_two_dozen_questions(plans):
    for files, questions in plans.values():
        check_plan(A7, files, questions, 20, 26, NAMES)
        assert len(files) == 48 and 17 < sum(f.length for f in files) / 3600 < 19
        templates = {q.tags["template"] for q in questions}
        assert templates >= {"first", "did", "which", "count", "order", "who", "what", "before", "absent_first", "absent_did", "absent_which"}
        assert sum(q.tags["template"] == "who" for q in questions) == 4


def test_every_answer_is_what_a_plain_reader_of_the_captions_finds(plans):
    for files, questions in plans.values():
        check_questions(questions, files, srt_of, TERMS, NEVER)
        check_notes(NOTES, files, srt_of, [str(f.spec["src"]) for f in files])


def test_the_terms_are_said_in_the_captions_and_the_terms_never_said_are_said_in_none():
    check_terms(TERMS, NEVER, [p.with_suffix(".srt") for p in EPISODE_FILES])


def test_the_days_are_the_episodes_own_and_the_files_are_numbered_in_that_order(plans):
    files, _ = plans[1]
    for n, f in enumerate(files, 1):
        with av.open(str(SOURCE / str(f.spec["src"]))) as c:
            tag = datetime.fromisoformat(c.metadata["date"].removesuffix("Z"))
        assert datetime.fromtimestamp(f.t) == tag and f.path == f"Podcasts/episode_{n:02d}.mp3", f.path
    assert [str(f.spec["src"])[:6] for f in files[:3]] == ["S01E01", "S01E02", "S01E03"]


def test_each_who_answer_is_the_guest_the_episode_names_twice_and_the_clue_is_said_in_that_episode_alone(plans):
    files, _ = plans[1]
    clues = {"oldest-stars": "oldest stars", "four-decades": "four decades", "cloud-chamber": "cloud chamber", "astronaut": "former NASA astronaut"}
    for note in (n for n in NOTES if n.template == "who"):
        (episode,) = [f for f in files if note.file in str(f.spec["src"])]
        raw = srt_of(episode).read_text(encoding="utf-8-sig")
        assert re.search(rf"^{note.value.upper()}:", raw, re.MULTILINE), f"{note.value} is no speaker the captions name in {episode.path}"
        with av.open(str(SOURCE / str(episode.spec["src"]))) as c:
            assert note.value.split()[-1] in c.metadata["artist"], c.metadata["artist"]  # the tags of the MP4 itself
        saying = [f for f in files if hits(blocks(srt_of(f)), clues[note.key])]
        assert saying == [episode], (note.key, [f.path for f in saying])
        assert grade(note.grade, note.value) and grade(note.grade, f"It was {note.value}.") and not grade(note.grade, HOST)
        assert grade(note.grade, f"{note.value}, interviewed by {HOST}") and grade(note.grade, f"{HOST} interviewed {note.value}")
        assert not grade(note.grade, "Mike Short") and not grade(note.grade, f"{note.value} or Mike Short") and not grade(note.grade, f"Mike Short, not {note.value}")
        assert HOST not in note.grade.wrong


def test_the_names_of_the_guests_cover_every_speaker_the_captions_name():
    """No guest's name is missing from the `wrong` answers' list: every speaker label of the 48 captions is a guest or a voice that is no one."""
    labels = set()
    for p in EPISODE_FILES:
        raw = p.with_suffix(".srt").read_text(encoding="utf-8-sig")
        labels |= {m.group(1).strip().casefold() for m in re.finditer(r"^([A-Z][A-Za-z .'\-]{1,40}):", raw, re.MULTILINE)}
    assert labels - {g.casefold() for g in GUESTS} <= NOT_PEOPLE, sorted(labels - {g.casefold() for g in GUESTS} - NOT_PEOPLE)


def test_no_time_answer_is_the_time_of_a_file_alone(plans):
    """The listing of files and times gives no `first` or `did` answer: none is within a minute of a file's start."""
    for files, questions in plans.values():
        for q in questions:
            if q.grade.rule == "time":
                t = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}").timestamp()
                assert min(abs(t - f.t) for f in files) > 60, q.qid


def test_the_pile_is_the_numbered_episodes_alone():
    files, _ = A7.plan_of(A7.recipe(scale="standard", seed=1))
    stems = {str(f.spec["src"]) for f in files}
    assert not {s for s in stems if re.search(r"S\d\dSE|S06|E00|BE_|promo", s)} and len(stems) == 48


def test_a_curated_answer_is_graded_and_a_planted_wrong_one_is_not():
    by_key = {n.key: n for n in NOTES}
    assert grade(by_key["toenails"].grade, "Their toenails") and not grade(by_key["toenails"].grade, "their fingernails")
    since = by_key["teaching-since"].grade
    assert grade(since, "1972") and not grade(since, "1969") and not grade(since, "1972 and 1969")
    assert grade(by_key["alexa-lights"].grade, "Off") and grade(by_key["alexa-lights"].grade, "She asked for them to be turned off.")
    assert grade(by_key["alexa-lights"].grade, "off, and then on") and not grade(by_key["alexa-lights"].grade, "On, then off")
    assert not grade(by_key["alexa-lights"].grade, "On") and not grade(by_key["alexa-lights"].grade, "off or on")


def test_the_tiny_pile_builds_as_mp3_sound_alone_that_plays_as_long_as_it_says_and_passes_the_probe(tmp_path):
    recipe = A7.recipe(scale="tiny", seed=1)
    files, questions = A7.plan_of(recipe)
    assert len(files) == 3 and 2 <= len(questions) <= 8
    dump, built = A7.build(recipe, tmp_path / "dump", 3)
    assert built == questions
    for e in dump.entries:
        path = tmp_path / "dump" / e.path
        assert os.stat(path).st_mtime == e.t and e.size == path.stat().st_size and e.media == "audio"
        with av.open(str(path)) as c:
            assert not c.streams.video and c.format.name == "mp3" and dict(c.metadata) == {}
            assert abs(c.streams.audio[0].duration * c.streams.audio[0].time_base - e.length) < 0.5
    out = tmp_path / "probe"
    out.mkdir()
    assert A7.probe(recipe, dump, out, Path(imageio_ffmpeg.get_ffmpeg_exe()), 1).status == "passed"


def test_a_recipe_of_another_time_zone_or_with_other_keys_is_refused():
    recipe = A7.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="keys"):
        A7.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="UTC offsets"):
        A7.plan_of({**recipe, "utc_offsets": ["+1234", "+1234"]})
