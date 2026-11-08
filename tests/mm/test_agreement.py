"""What is copied from MultimediaSearch, still agrees with it (plan 3.5, 7.4; U7's acceptance). The benchmark's core and V7 run without MultimediaSearch, so they hold
small copies of what MultimediaSearch decides: the reader's frames, the fake decider's and the day's constants, the ask script's words, ffmpeg's path, the
packet duration of a clip, and the share of an exec's time an ask may have. Each is run here beside the original, on the same files and inputs,
and the expected value is the original's. A change in MultimediaSearch that this finds is a change to take over or to refuse.

Expected values are MultimediaSearch's, from the installed package (fe0ece5 on this machine) and from the copies of its tests (tests/mm/day.py,
tests/mm/fake_decider.py)."""

from __future__ import annotations

import importlib
import itertools
import math
import os

import numpy as np
import pytest

pytest.importorskip("mms")

import day
import fake_decider
from test_v7_media import packet_duration as v7_packet_duration

from mms import media, reader
from mms.config import ffmpeg_exe as mmss_ffmpeg
from mms.scripts import ask as ask_script

from mmsb.core import render
from mmsb.scenarios.v7_screen import constants, frames, media_common
from mmsb.scenarios.v8_egolife import plan as v8_plan
from mmsb.scenarios.v8_egolife import source as v8_source
from mmsb.systems.mm import estimate

v8_report = importlib.import_module("mmsb.scenarios.v8_egolife.report")  # (the package's own `report` is a function)


def first_of(dump, kind: str, count: int = 1):
    return [dump.root / e.path for e in dump.entries if e.kind == kind][:count]


# -- the reader's frames and sound -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("max_px", (0, 230_400, 1_638_400, 5_000))
def test_the_size_a_picture_is_scaled_to_is_the_readers(max_px):
    sizes = [(w, h) for w, h in itertools.product((1, 7, 160, 540, 640, 720, 1280, 1408, 1920, 3840), (1, 9, 90, 360, 720, 960, 1080, 1408, 2400))]
    assert [frames.fit(w, h, max_px) for w, h in sizes] == [reader.fit(w, h, max_px) for w, h in sizes]


@pytest.mark.parametrize("kind", ("screen", "porch", "clip"))
def test_the_frames_of_a_written_file_are_the_readers_in_time_and_in_pixels(kind, week):
    _, dump, _ = week
    for path in first_of(dump, kind, 2):
        times = reader._times(path)
        assert frames.frame_times(path) == times and len(times) > 0, path
        wanted = times[:: max(1, len(times) // 7)]
        for max_px in (0, reader.PROFILE["max_px"]):
            mine, theirs = list(frames.decode_frames(path, wanted, max_px)), list(reader._decode(path, wanted, max_px))
            assert [sec for sec, _ in mine] == [sec for sec, _ in theirs] == [sec for _, sec in wanted], (path, max_px)
            assert all(a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b)) for (_, a), (_, b) in zip(mine, theirs)), (path, max_px)


def test_the_sound_of_a_written_file_is_the_readers_from_one_seek(week):
    _, dump, _ = week
    for path in first_of(dump, "mic", 2):
        for t0, t1 in ((0.0, 8.0), (30.0, 41.5), (0.0, 1e9)):
            (first_a, a), (first_b, b) = frames.audio_span(path, t0, t1, constants.RATE), reader._span(path, t0, t1, constants.RATE)
            assert first_a == first_b and a.dtype == b.dtype and np.array_equal(a, b), (path, t0, t1)


# -- the constants the week is made to be told by --------------------------------------------------------------------------------

def test_the_constants_of_the_writer_are_the_days_the_fake_deciders_and_the_readers():
    assert constants.RATE == day.RATE == reader.RATE
    assert (constants.SHADE_TOL, constants.PITCH_RMS, constants.PITCH_SHARE, constants.PITCH_BAND) == (
        fake_decider.SHADE_TOL, fake_decider.PITCH_RMS, fake_decider.PITCH_SHARE, fake_decider.PITCH_BAND)
    assert {key: reader.PROFILE[key] for key in constants.PROFILE} == constants.PROFILE
    assert [constants.font(n).size for n in (10, 40)] == [day.font(n).size for n in (10, 40)]
    assert estimate.reader.PROFILE["records_chars"] == 3000, "the records window the estimate counts by"


def test_the_shade_and_the_pitch_of_the_writers_checks_are_the_fake_deciders_on_planted_pictures_and_sounds():
    rng = np.random.default_rng(3)
    from PIL import Image

    for shade in ((200, 40, 40), (30, 160, 60), (125, 80, 45)):
        for jitter in (0, 20, 29, 31, 60):
            pixels = np.clip(np.array(shade) + rng.integers(-jitter, jitter + 1, (36, 64, 3)), 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            assert constants.shade_share(image, shade) == fake_decider.shade_share(image, shade), (shade, jitter)
    for hz in (440.0, 659.0, 900.0):
        for rate in (16000, 8000):
            tone = np.sin(2 * np.pi * hz * np.arange(rate) / rate) + rng.normal(0, 0.3, rate)
            for probe in (hz, hz * 1.02, hz * 1.5):
                assert constants.pitch_share(tone, rate, probe) == fake_decider.pitch_share(tone, rate, probe), (hz, probe)
    assert constants.pitch_share(np.zeros(100), 16000, 440.0) == fake_decider.pitch_share(np.zeros(100), 16000, 440.0) == 0.0


def test_the_ffmpeg_of_the_writer_is_the_one_mms_falls_back_to_and_not_one_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # no ffmpeg on it
    assert media_common.ffmpeg_exe() == mmss_ffmpeg(), "with none on PATH they are the same binary"
    fake = tmp_path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    fake.write_bytes(b"x")
    assert mmss_ffmpeg().lower() == str(fake).lower(), "MultimediaSearch prefers one on PATH"
    assert media_common.ffmpeg_exe() != mmss_ffmpeg(), "the writer's bytes are those of the shipped binary alone"


def test_the_length_of_a_clip_is_the_packet_duration_mm_stores(week):
    _, dump, _ = week
    files = [p for kind in ("screen", "porch", "clip") for p in first_of(dump, kind, 3)]
    assert len(files) >= 7, "three of the screen and the porch, and the week's phone clip"
    for path in files:
        want = media.duration(path)
        assert v8_source.packet_duration(path) == want == v7_packet_duration(path), path


# -- the words and the figures other modules quote -------------------------------------------------------------------------------

def test_the_words_ask_refuses_in_a_question_are_the_ones_the_oracles_are_written_against():
    assert (v8_plan.TIME_WORDS.pattern, v8_plan.TIME_WORDS.flags) == (ask_script.TIME_WORDS.pattern, ask_script.TIME_WORDS.flags)
    assert (v8_plan.JOINED.pattern, v8_plan.JOINED.flags) == (ask_script.JOINED.pattern, ask_script.JOINED.flags)


def test_the_decode_second_the_report_quotes_is_the_ask_scripts():
    assert v8_report.ASSUMED_DECODE_S == ask_script.DECODE_S == 0.010


def test_the_share_of_an_execs_time_an_ask_may_have_is_the_same_in_the_report_and_in_the_system():
    assert render.ASK_SHARE == estimate.ASK_SHARE == 0.75


def test_a_window_is_priced_as_mm_prices_it_and_the_records_window_is_the_readers():
    from mms import decider

    about = decider.known("d1-3B")[1]
    for kind in ("video", "audio", "text", "records"):
        assert estimate.window_s(about, kind) == ask_script.window_seconds(kind, reader.profile_of(None, about), about)
    assert estimate.window_s(about, "video") == pytest.approx(0.061) and math.isclose(estimate.window_s(about, "records"), 0.075)
    assert estimate.SLOW_S < estimate.window_s(decider.known("d3-mini")[1]) < 0.2, "d3-mini is the slow one the pilot ran alone"
