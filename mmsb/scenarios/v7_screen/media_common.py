"""What the drawings and sounds of the V7 week share (the other modules of `media_*` draw them):

    NEEDLE_RGB, DISTRACTOR_RGB   the exact shade of each needle, and the shades of the distractors that share its named colour
    _write_video(path, frames, length_s, size, crf, work)   frames as x264 through PyAV, put in place by `_finish`
    ffmpeg_exe()                 the ffmpeg the files are written with

Everything is a function of its spec (`seed` and the rest): the same spec is the same pixels and samples, and nothing reads a clock
or a global random stream. The specs are plain JSON-able dicts, the shapes `plan.py` makes (times are seconds from the file's start);
each `media_*` module describes its own.

A needle is one exact shade (`NEEDLE_RGB`): the fake decider's `shade:R,G,B` says yes when 2% of a picture's pixels are within
`SHADE_TOL` of it. Every distractor of the needle's source has the needle's named colour and a shade at least `SHADE_GAP` from it
(`DISTRACTOR_RGB`), so a colour filter gives a shortlist and not the answer; `_clear_of_needles` refuses a colour that is closer, and
`media_check` checks the constants of all the modules against each other when the package is imported.

The files are written as the capture writes them (TESTBED.md): the screen at crf 30 and the camera at crf 28 with x264's
`zerolatency` tune, the microphone as opus at 24 kbit/s; in this process through PyAV (video) and a pipe to ffmpeg (sound, never a
WAV: 42 h of it is 4.8 GB), one thread each, so that three files written at once use three cores and the bytes do not depend on
the machine's core count. Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

import math
import os
import random
import shutil
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from mmsb.scenarios.v7_screen.constants import SHADE_TOL, font

SHADE_GAP = 2 * SHADE_TOL  # a distractor's colour is at least this far from its source's needle shades (SHADE_TOL: the fake decider's)

NEEDLE_RGB = {"error": (176, 28, 36), "sign": (20, 170, 80), "car": (240, 20, 20), "parcel": (190, 142, 88)}
DISTRACTOR_RGB = {  # by the needle they share a named colour or a look with: what a colour filter shortlists
    "error": {"warning": (240, 130, 20), "info": (40, 100, 200), "save": (110, 115, 125), "confirm": (232, 80, 72),
              "banner": (225, 85, 85), "cells": (240, 120, 120), "badge": (255, 59, 48), "toast": (225, 75, 60)},
    "sign": {"yellow": (250, 215, 0), "white": (245, 245, 245), "blue": (20, 60, 200), "grey": (130, 130, 135),
             "plant": (80, 140, 60), "door": (50, 110, 60), "shirt": (110, 190, 120), "car": (90, 150, 40)},
    "car": {"blue": (30, 60, 200), "white": (245, 245, 245), "grey": (130, 130, 135), "yellow": (240, 210, 20),
            "door": (150, 30, 30), "umbrella": (200, 60, 70), "tomato": (205, 70, 50)},
    "parcel": {"carried": (120, 88, 52), "dog": (110, 75, 40), "door": (55, 75, 115), "mat": (100, 90, 70),
               "planter": (160, 80, 55), "cat": (225, 125, 35), "courier": (110, 75, 40)},
}
SOURCE_NEEDLES = {"phone": ("sign", "car"), "screen": ("error",), "porch": ("parcel",)}  # the needles each source's colours are cleared of


def distance(a, b) -> float:
    return math.dist(a, b)


def gap(a: float, b: float) -> float:
    """How far apart two pitches are, as a share of the lower."""
    return abs(a - b) / min(a, b)


def _rng(seed, salt: str = "") -> random.Random:
    return random.Random(f"{seed}-{salt}")


def _rgb(value) -> tuple[int, int, int]:
    return tuple(int(c) for c in value)  # type: ignore[return-value]


def _clear_of_needles(rgb, what: str, source: str, allow: str | None = None) -> None:
    """`rgb` is no shade of a needle of `source` (nor within SHADE_GAP of one), except the one of `allow`, which it may be exactly."""
    for kind in SOURCE_NEEDLES[source]:
        shade = NEEDLE_RGB[kind]
        if distance(rgb, shade) < SHADE_GAP and not (kind == allow and tuple(rgb) == shade):
            raise ValueError(f"{what} {tuple(rgb)} is within {SHADE_GAP} of the {kind} needle's shade {shade}")


def _ink(fill) -> tuple[int, int, int]:
    """Letters for a sign or button of `fill`: white on a dark one, near black on a light one."""
    return (20, 20, 24) if 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2] > 150 else (255, 255, 255)


def _gradient(size: tuple[int, int], top, bottom) -> Image.Image:
    w, h = size
    rows = np.linspace(top, bottom, h, dtype=np.float32)[:, None, :]
    return Image.fromarray(np.broadcast_to(rows, (h, w, 3)).astype(np.uint8), "RGB")


def _fit(draw: ImageDraw.ImageDraw, xy, text: str, fill, size: int, width: float, anchor: str = "mm") -> None:
    """`text` at `xy`, in `size` pixels or the largest below it that is `width` wide at most."""
    while size > 12 and draw.textlength(text, font=font(size)) > width:
        size -= 2
    draw.text(xy, text, fill=fill, font=font(size), anchor=anchor)


def _px(size: tuple[int, int], *fractions: float) -> tuple[int, ...]:
    """Pixel corners from fractions of the picture's width and height: (x0, y0, x1, y1, ...)."""
    return tuple(int(round(f * size[i % 2])) for i, f in enumerate(fractions))


SAFE = ((70, 90, 140), (150, 120, 90), (110, 110, 115), (235, 228, 210), (222, 228, 238), (238, 232, 240), (240, 215, 150), (90, 70, 60),
        (120, 90, 70), (180, 150, 110), (60, 60, 70), (200, 180, 120), (130, 100, 150), (70, 130, 150))  # no needle's shade, nor near one


def ffmpeg_exe() -> str:
    """The ffmpeg that imageio-ffmpeg ships, whatever is on PATH: the files' bytes are those of this binary (the recorded pilot's
    microphone files are reproduced by it)."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def _finish(part: Path, path: Path) -> None:
    """Put a finished file from the working folder in its place, so a data folder never holds half a file. It is copied, not
    moved: on Windows a moved file keeps the working folder's owner-only permissions (Python makes temporary folders so), and
    Codex's sandbox then cannot open it, while a new file takes the permissions of the folder it is made in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    shutil.copyfile(part, temp)
    os.replace(temp, path)
    part.unlink()


def _write_video(path: Path, frames, length_s: float, size: tuple[int, int], crf: int, work: Path) -> None:
    """`frames`, `(second, image)` in order, as x264 at `crf` (preset veryfast, tune zerolatency, one thread), each at its own
    time. The container is the path's (`.mkv` or `.mp4`). A frame says it lasts 40 ms (as capture's own files do: ffmpeg's demuxer
    reads a frame's decode time from the frame rate and the duration, and a long duration on a 25 fps stream shifts the stream's
    start, and with it every time the reader computes), and the last lasts until `length_s`, which is the file's duration."""
    Path(work).mkdir(parents=True, exist_ok=True)
    part = Path(work) / Path(path).name
    packets = []
    with av.open(str(part), "w", options={"fflags": "+bitexact"}) as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.options = {"crf": str(crf), "preset": "veryfast", "tune": "zerolatency", "threads": "1"}
        for second, image in frames:
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = round(second * 1000), Fraction(1, 1000)
            packets += stream.encode(frame)
        packets += stream.encode(None)
        for i, packet in enumerate(packets):
            packet.duration = 40 if i + 1 < len(packets) else max(round(length_s * 1000) - packet.pts, 40)
            container.mux(packet)
    _finish(part, Path(path))
