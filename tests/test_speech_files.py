"""What the speech scenarios share (scenarios/realdump/speech_files.py): a source of MP4s with their SRTs read, a video copied and a sound
extracted with none of the source's tags, what the probe reads of the files written, a term screened out when its question would be less than
certain, the days of a term, and the curated questions. Over small media made here with ffmpeg."""

from __future__ import annotations

import subprocess
from datetime import date, datetime
from pathlib import Path

import av
import imageio_ffmpeg
import pytest

from mmsb.core.question import Grade
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech, speech_files
from mmsb.scenarios.realdump.speech import Cue, Talk, Term
from mmsb.scenarios.realdump.speech_files import Note

SECRET = "Knapsack Lecture"  # the title planted in the source's tags


def make_mp4(path: Path, seconds: int = 6, date_tag: str = "2016-10-24T04:00:00Z") -> Path:
    """An MP4 of `seconds` of blue video and a sine tone, tagged with a title, a keyword and a date, as OCW's are."""
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                          "-f", "lavfi", "-i", f"color=c=blue:s=160x120:d={seconds}", "-c:v", "libx264", "-c:a", "aac", "-shortest", "-metadata",
                          f"title={SECRET}", "-metadata", "keywords=greedy algorithm", "-metadata", f"date={date_tag}", "-y", str(path)],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    return path


def text_of(path: Path, read) -> str:
    return realdump.text_of(read(path)).casefold()


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("media")
    make_mp4(folder / "a.mp4")
    return folder


def test_the_source_has_a_title_in_its_tags_and_a_written_file_has_none(media, tmp_path):
    """The planted title is in the source's header, where the probe would find it, and in none of what is written from it."""
    source = media / "a.mp4"
    assert "knapsack lecture" in text_of(source, speech_files.mp4_header) and "greedy algorithm" in text_of(source, speech_files.mp4_header)
    video, m4a, mp3 = tmp_path / "v.mp4", tmp_path / "s.m4a", tmp_path / "s.mp3"
    speech_files.copy_video(source, video)
    speech_files.extract_sound(source, m4a)
    speech_files.extract_sound(source, mp3)
    for path in (video, m4a, mp3):
        assert "knapsack" not in text_of(path, speech_files.HEADERS[path.suffix]) and "greedy" not in text_of(path, speech_files.HEADERS[path.suffix])
        with av.open(str(path)) as c:
            assert not {k for k in c.metadata if k not in ("major_brand", "minor_version", "compatible_brands")}, f"{path}: {dict(c.metadata)}"
    assert text_of(video, speech_files.mp4_header).count("ftyp") == 1 and "ftypm4a" in text_of(m4a, speech_files.mp4_header).replace(" ", "")


def test_a_written_file_plays_as_long_as_planned_and_the_sound_is_what_the_source_has(media, tmp_path):
    video, m4a, mp3 = tmp_path / "v.mp4", tmp_path / "s.m4a", tmp_path / "s.mp3"
    speech_files.copy_video(media / "a.mp4", video)
    speech_files.extract_sound(media / "a.mp4", m4a)
    speech_files.extract_sound(media / "a.mp4", mp3)
    planned = speech_files.audio_seconds(media / "a.mp4")
    assert abs(speech_files.video_seconds(video) - 6) < 0.2
    speech_files.check_plays(m4a, planned, speech_files.audio_seconds)
    speech_files.check_plays(mp3, planned, speech_files.audio_seconds)
    with pytest.raises(ValueError, match="plays"):
        speech_files.check_plays(mp3, planned + 2, speech_files.audio_seconds)
    with av.open(str(mp3)) as c:
        assert (c.streams.audio[0].channels, c.streams.audio[0].rate) == (1, 44100) and not c.streams.video
    with av.open(str(m4a)) as c:
        assert not c.streams.video and c.streams.audio[0].codec_context.name == "aac"


def test_the_probe_reads_a_title_in_an_mp4_or_an_id3_tag_and_passes_the_files_written(media, tmp_path):
    """The probe finds a planted word in the source's own header (MP4 box or ID3 tag), and passes what `copy_video` and `extract_sound` write."""
    tagged_mp3 = tmp_path / "tagged.mp3"
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(media / "a.mp4"), "-vn", "-c:a", "libmp3lame",
                          "-metadata", "title=greedy algorithm", "-y", str(tagged_mp3)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "greedy algorithm" in text_of(tagged_mp3, speech_files.mp3_header)
    out = tmp_path / "dump"
    files = [realdump.File("a.mp4", "s", 1.0e9, "lecture", "video", {}), realdump.File("b.m4a", "s", 1.0e9 + 60, "recording", "audio", {}),
             realdump.File("c.mp3", "s", 1.0e9 + 120, "episode", "audio", {})]

    def write(f: realdump.File, path: Path) -> None:
        (speech_files.copy_video if path.suffix == ".mp4" else speech_files.extract_sound)(media / "a.mp4", path)

    dump = realdump.dump_of("k", out, "x", realdump.write(files, out, 2, write))
    (tmp_path / "p1").mkdir()
    assert realdump.probe(dump, tmp_path / "p1", ["greedy algorithm", "knapsack"], speech_files.HEADERS).status == "passed"
    leaky = realdump.dump_of("k", tmp_path / "leaky", "x", realdump.write(files[:1], tmp_path / "leaky", 1,
                                                                        lambda f, path: path.write_bytes((media / "a.mp4").read_bytes())))
    (tmp_path / "p2").mkdir()
    outcome = realdump.probe(leaky, tmp_path / "p2", ["greedy algorithm"], speech_files.HEADERS)
    assert outcome.status == "failed" and "greedy" in outcome.failures[0]


def test_an_mp3_that_starts_with_neither_a_tag_nor_a_frame_is_refused(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"RIFF" + bytes(300))
    with pytest.raises(ValueError, match="ID3 tag or an MP3 frame"):
        speech_files.mp3_header(bad)


def test_the_files_of_a_source_are_the_mp4s_and_srts_that_match_and_a_lone_one_is_refused(tmp_path):
    for stem in ("lec01", "lec02", "other"):
        (tmp_path / f"{stem}.mp4").write_bytes(b"")
    for stem in ("lec01", "lec02"):
        (tmp_path / f"{stem}.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nhello there\n", encoding="utf-8")
    recs = speech_files.read_recordings(tmp_path, lambda stem: stem.startswith("lec"))
    assert [r.stem for r in recs] == ["lec01", "lec02"] and recs[0].cues[0].text == "hello there"
    with pytest.raises(FileNotFoundError, match="same recordings"):
        speech_files.read_recordings(tmp_path, lambda stem: True)  # `other` has no SRT
    with pytest.raises(FileNotFoundError, match="same recordings"):
        speech_files.read_recordings(tmp_path, lambda stem: stem == "none")


def test_the_date_tag_is_a_wall_clock_time(media):
    assert speech_files.tagged_time(media / "a.mp4") == datetime(2016, 10, 24, 4, 0, 0)


def test_the_days_of_a_term_skip_what_is_not_a_class_day():
    days = speech_files.class_days(date(2016, 10, 5), (0, 2), [date(2016, 10, 10)], 5)
    assert days == [date(2016, 10, 5), date(2016, 10, 12), date(2016, 10, 17), date(2016, 10, 19), date(2016, 10, 24)]
    assert all(d.weekday() in (0, 2) for d in days)


def cue(t: float, text: str, speaker: str | None = "Prof") -> Cue:
    return Cue(t, t + 3, text, speaker)


def talks() -> list[Talk]:
    a = Talk("a/one.mp3", datetime(2024, 9, 3, 10), 3000.0, (cue(300, "the knapsack problem"), cue(900, "greedy algorithms"), cue(1200, "bit coin")))
    b = Talk("a/two.mp3", datetime(2024, 9, 3, 11), 3000.0, (cue(100, "greedy algorithms again"), cue(500, "regression", "Student")))
    c = Talk("a/three.mp3", datetime(2024, 9, 5, 10), 3000.0, (cue(100, "greedy"), cue(200, "algorithms"), cue(700, "regression"), cue(800, "knapsack problem")))
    return [a, b, c]


def test_a_term_a_student_says_or_two_files_of_a_day_say_is_screened_out_where_asked():
    terms = [Term("knapsack problem", "the knapsack problem"), Term("greedy algorithm", "greedy algorithms"), Term("regression", "regression"),
             Term("Higgs boson", "the Higgs boson")]
    assert [t.search for t in speech_files.screened(talks(), terms)] == ["knapsack problem", "greedy algorithm"]  # a student says regression once; the Higgs is nowhere
    assert [t.search for t in speech_files.screened(talks(), terms, others=())] == ["knapsack problem", "greedy algorithm", "regression"]
    assert [t.search for t in speech_files.screened(talks(), terms, one_a_day=True, others=())] == ["knapsack problem", "regression"]  # greedy: two files of 3 September


def test_a_term_run_on_in_a_caption_is_not_absent_but_a_term_nowhere_is():
    speech_files.check_never(talks(), [Term("Higgs boson", "the Higgs boson")])
    with pytest.raises(ValueError, match="run-on letters"):
        speech_files.check_never(talks(), [Term("bitcoin", "Bitcoin")])  # "bit coin" in a caption
    with pytest.raises(ValueError, match="run-on letters"):
        speech_files.check_never(talks(), [Term("greedy algorithm", "greedy algorithms")])


def note(file: str, key: str = "k") -> Note:
    return Note(file, key, "What?", "what", "<a>", Grade("text", "x", None, None, (), (), None), 300, 303, "knapsack problem", "x")


def test_a_note_is_asked_of_its_recording_and_left_out_of_a_pile_that_has_not_got_it():
    assert [c.talk for c in speech_files.curated([note("lec02_")], ["lec01_300k", "lec02_300k"])] == [1]
    assert speech_files.curated([note("lec03_")], ["lec01_300k", "lec02_300k"]) == []
    with pytest.raises(ValueError, match="2 recordings"):
        speech_files.curated([note("lec0")], ["lec01_300k", "lec02_300k"])
    kept = speech_files.curated([note("one")], ["one", "two"])
    assert speech.questions_of(talks()[:1], [Term("greedy algorithm", "greedy algorithms")], [], speech.Say("X1", "m", "audio", "they", "episode", "episodes"),
                               {}, 1, realdump.local_t, kept)[0].tags["template"] == "what"


def test_a_note_whose_phrase_is_not_in_its_cues_is_refused_by_the_build():
    bad = Note("one", "k", "What?", "what", "<a>", Grade("text", "x", None, None, (), (), None), 300, 303, "greedy algorithm", "x")
    with pytest.raises(ValueError, match="curated question"):
        speech.questions_of(talks()[:1], [], [], speech.Say("X1", "m", "audio", "they", "episode", "episodes"), {}, 1, realdump.local_t,
                            speech_files.curated([bad], ["one"]))


def test_a_cue_is_told_to_the_millisecond_and_a_note_of_a_sound_is_no_cue(tmp_path):
    (tmp_path / "x.srt").write_text("1\n00:00:01,500 --> 00:00:03,250\nHello\nthere\n\n2\n00:01:00,000 --> 00:01:02,000\n[MUSIC]\n", encoding="utf-8")
    (tmp_path / "x.mp4").write_bytes(b"")
    (rec,) = speech_files.read_recordings(tmp_path, lambda s: True)
    assert [(c.t0, c.t1, c.text) for c in rec.cues] == [(1.5, 3.25, "Hello there")]
