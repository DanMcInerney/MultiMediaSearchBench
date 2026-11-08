"""Constants and small functions that V7 shares with MultimediaSearch's tests and reader, copied so that V7 builds without MultimediaSearch:
`RATE` and `font` (tests/day.py), the fake decider's `SHADE_TOL`, `PITCH_*`, `shade_share` and `pitch_share` (tests/fake_decider.py),
and the values of the reader's `PROFILE` that the writer's checks and the tests use (src/mms/reader.py). Copied at fe0ece5,
when MultimediaSearch was called Multimem (src/multimem/). The needles are made to be told by these, so if the originals change the week is no longer what its decider sees: tests/mm
asserts that they are equal."""

from __future__ import annotations

import numpy as np
from PIL import ImageFont

RATE = 16000


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


PITCH_RMS = 0.01  # the RMS a sound needs for `pitch:<hz>`, so that room noise alone is never a pitch
PITCH_SHARE = 0.3  # the share of a sound's spectral energy that must lie within PITCH_BAND of the pitch
PITCH_BAND = 0.03  # of the pitch, either side

SHADE_TOL = 30  # the RGB distance within which a pixel is a `shade:R,G,B`


def shade_share(image, rgb) -> float:
    """The share of a picture's pixels within SHADE_TOL (Euclidean RGB distance) of the colour `rgb`."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.int32).reshape(-1, 3)
    if not len(pixels):
        return 0.0
    return float((np.square(pixels - np.asarray(rgb, dtype=np.int32)).sum(axis=1) <= SHADE_TOL ** 2).mean())


def pitch_share(samples: np.ndarray, rate: int, hz: float) -> float:
    """The share of a sound's spectral energy (|rfft|², the DC bin left out) within 3% of `hz`; 0.0 for silence."""
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < 2:
        return 0.0
    energy = np.abs(np.fft.rfft(samples)) ** 2
    energy[0] = 0.0
    total = energy.sum()
    if total <= 0:
        return 0.0
    near = np.abs(np.fft.rfftfreq(len(samples), 1 / rate) - hz) <= PITCH_BAND * hz
    return float(energy[near].sum() / total)


# the values of the reader's PROFILE (src/mms/reader.py, src/multimem/reader.py at fe0ece5) that the writer's checks and the tests use
PROFILE = {
    "every_s": 1.0,      # video: one frame per this many seconds
    "audio_s": 5.0,      # audio: the length of a window, and ...
    "audio_hop_s": 2.5,  # ... the step between windows
    "quiet_rms": 0.004,  # audio: a window whose RMS (samples in -1..1) is below this is skipped, about -48 dBFS
    "max_px": 230_400,   # pictures are cut to at most this many pixels (640x360)
}
