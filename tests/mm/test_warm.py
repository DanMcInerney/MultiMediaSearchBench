"""The ask that loads mm's model before its execs (`oracle.warm_params`): one window of the earliest file the model reads. The first mm run
of A1 (a day of sound) was refused before any exec, because only a picture or a video could load a model, though d1-Omni-2B hears."""

from __future__ import annotations

import pytest

pytest.importorskip("multimem")

from mmmb.core.contract import Dump, ManifestEntry  # noqa: E402
from mmmb.systems.mm.oracle import warm_params  # noqa: E402


def entry(path: str, media: str, t: float) -> ManifestEntry:
    return ManifestEntry(path, f"{media}-source", t, media, media, None, False, 1, 0, "0" * 64)


def test_the_model_is_loaded_on_the_earliest_picture_else_video_else_sound_that_it_reads(tmp_path):
    mixed = Dump("k", tmp_path, "x", (entry("b.wav", "audio", 5.0), entry("c.mp4", "video", 9.0), entry("a.jpg", "image", 7.0)), ())
    assert warm_params(mixed)["sources"] == ["image-source"] and warm_params(mixed)["since"] == 7.0
    sound = Dump("k", tmp_path, "a1_home_audio", (entry("late.wav", "audio", 9.0), entry("early.wav", "audio", 5.0)), ())
    heard = warm_params(sound, frozenset({"audio", "image", "video"}))
    assert heard == {"question": "Is anything heard?", "sources": ["audio-source"], "since": 5.0, "until": 6.0, "limit": 1, "candidates": 0}
    with pytest.raises(ValueError, match="a1_home_audio's dump has nothing the model reads \\(image, video\\)"):
        warm_params(sound, frozenset({"image", "video"}))
