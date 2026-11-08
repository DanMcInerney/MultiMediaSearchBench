"""New tests for mmsb/scenarios/v7_screen/frames.py and constants.py: the small copies of what MultimediaSearch's reader and its fake decider do
to a file or a picture, which V7 uses so that it needs neither. Expected values are laws of the functions (a picture is never scaled
up, a tone has its own pitch, a plain colour is all of its own shade) and files written here with known frames and sound. That the copies
agree with MultimediaSearch's, byte for byte, is tests/mm's: it needs MultimediaSearch."""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from mmsb.scenarios.v7_screen.constants import PITCH_BAND, PROFILE, RATE, SHADE_TOL, font, pitch_share, shade_share
from mmsb.scenarios.v7_screen.frames import audio_span, decode_frames, fit, frame_times
from mmsb.scenarios.v7_screen.media_common import _write_video
from mmsb.scenarios.v7_screen.media_sound import write_mic


def test_a_picture_is_scaled_to_the_budget_with_its_aspect_never_up_and_not_at_all_with_none():
    assert fit(1280, 720, 230_400) == (640, 360), "the reader's 640x360 for a 1280x720 screen"
    assert fit(640, 360, 230_400) == (640, 360) and fit(100, 50, 230_400) == (100, 50), "a picture within the budget is as it is"
    assert fit(3840, 2400, 0) == (3840, 2400), "a budget of 0 is as stored"
    r = random.Random(3)
    for _ in range(300):
        w, h, budget = r.randint(1, 5000), r.randint(1, 5000), r.randint(100, 3_000_000)
        got = fit(w, h, budget)
        assert got[0] * got[1] <= budget or got == (w, h), (w, h, budget, got)
        assert got[0] <= w and got[1] <= h and min(got) >= 1, (w, h, budget, got)
        if w * h > budget and min(w, h) > 40:
            assert got[0] * got[1] > 0.9 * budget and abs(got[0] / got[1] - w / h) < 0.05 * w / h + 0.05, (w, h, budget, got)


def test_a_variable_frame_rate_file_gives_back_the_seconds_it_was_written_at_and_frames_at_the_budget(tmp_path):
    seconds = [0, 1.5, 2, 30, 61.5]
    images = [Image.new("RGB", (1280, 720), (k * 40, 20, 200 - 30 * k)) for k in range(len(seconds))]
    path = tmp_path / "v.mkv"
    _write_video(path, list(zip(seconds, images)), 90, (1280, 720), 30, tmp_path / "work")
    times = frame_times(path)
    assert [round(sec, 3) for _, sec in times] == seconds and [pts for pts, _ in times] == [round(s * 1000) for s in seconds]
    got = list(decode_frames(path, times, PROFILE["max_px"]))
    assert [round(sec, 3) for sec, _ in got] == seconds and all(image.size == (640, 360) for _, image in got)
    for (_, back), k in zip(got, range(len(seconds))):  # the colour of each frame survives the encoder to within a few levels
        assert np.abs(np.asarray(back, dtype=int).mean(axis=(0, 1)) - (k * 40, 20, 200 - 30 * k)).max() < 20
    only = list(decode_frames(path, times[3:4], 0))
    assert [round(sec, 3) for sec, _ in only] == [30.0] and only[0][1].size == (1280, 720), "one frame asked for, at the size it was stored"


def test_the_sound_of_a_written_file_is_mono_float32_at_the_rate_from_the_first_sample_and_a_video_has_none(tmp_path):
    spec = {"length_s": 20, "seed": 1, "bed_dbfs": -50, "tv": [], "sounds": [{"kind": "doorbell", "at": 5.0, "dur": 1.8, "notes": [659.26, 523.25], "seed": 1}]}
    path = tmp_path / "m.ogg"
    write_mic(path, spec, tmp_path / "work")
    first, samples = audio_span(path, 0, 1e9)
    assert samples.dtype == np.float32 and samples.ndim == 1 and abs(len(samples) - 20 * RATE) < 0.1 * RATE and abs(first) < 0.1
    loud = np.abs(samples).reshape(-1)[: len(samples) // 100 * 100].reshape(-1, 100).max(axis=1)  # the chime is the loudest stretch
    assert 5.0 <= np.argmax(loud) * 100 / RATE < 6.8
    _, part = audio_span(path, 10, 12)
    assert len(part) < len(samples) and len(part) > 0, "a span is read from a seek, a little past its end"
    video = tmp_path / "v.mkv"
    _write_video(video, [(0, Image.new("RGB", (64, 64)))], 2, (64, 64), 30, tmp_path / "work")
    assert audio_span(video, 0, 1e9)[0] == 0.0 and len(audio_span(video, 0, 1e9)[1]) == 0, "no sound in a video file"


def test_a_plain_picture_is_all_of_its_own_shade_and_none_of_another_and_a_tone_is_its_pitch():
    red = Image.new("RGB", (40, 30), (200, 20, 20))
    assert shade_share(red, (200, 20, 20)) == 1.0 and shade_share(red, (200 + SHADE_TOL, 20, 20)) == 1.0, "on the edge of the tolerance"
    assert shade_share(red, (200 + SHADE_TOL + 1, 20, 20)) == 0.0 and shade_share(red, (20, 200, 20)) == 0.0
    half = Image.new("RGB", (40, 30), (20, 200, 20))
    half.paste(Image.new("RGB", (20, 30), (200, 20, 20)), (0, 0))
    assert shade_share(half, (200, 20, 20)) == pytest.approx(0.5)
    t = np.arange(RATE) / RATE
    tone = np.sin(2 * np.pi * 659.0 * t).astype(np.float32)
    assert pitch_share(tone, RATE, 659.0) > 0.9 and pitch_share(tone, RATE, 659.0 * (1 + 2 * PITCH_BAND)) < 0.05
    assert pitch_share(np.zeros(RATE, np.float32), RATE, 659.0) == 0.0 and pitch_share(np.ones(1), RATE, 659.0) == 0.0
    noise = np.random.default_rng(1).standard_normal(RATE)
    assert pitch_share(noise, RATE, 659.0) < 0.1


def test_the_values_of_the_readers_profile_the_checks_use_and_the_font_are_the_documented_ones():
    assert PROFILE == {"every_s": 1.0, "audio_s": 5.0, "audio_hop_s": 2.5, "quiet_rms": 0.004, "max_px": 640 * 360} and RATE == 16000
    assert font(24).size == 24 and font(48).getbbox("E-4417")[2] > font(24).getbbox("E-4417")[2]
