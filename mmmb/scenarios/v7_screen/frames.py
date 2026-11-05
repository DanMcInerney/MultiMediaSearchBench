"""The frames and the sound of a written file as Multimem's reader hands them to a decider: small PyAV copies of `fit`, `_times`,
`_decode` and `_span` of src/multimem/reader.py at fe0ece5 (`frame_times`, `decode_frames` and `audio_span` here, so that no caller's
local `times` hides one), so that the writer checks a file as d1 would read it without importing Multimem. tests/mm runs both on the
same files and asserts that they agree."""

from __future__ import annotations

import math
import os
from pathlib import Path

import av
import numpy as np

from mmmb.scenarios.v7_screen.constants import RATE

DECODE_THREADS = min(os.cpu_count() or 1, 8)  # frame threads per decode: 4 to 16 were alike on a 1408x1408 clip, and more crowd capture


def fit(width: int, height: int, max_px: int) -> tuple[int, int]:
    """The size a picture is scaled to for a budget of `max_px` pixels: aspect kept, never up, 0 for as stored."""
    if not max_px or width * height <= max_px:
        return width, height
    scale = math.sqrt(max_px / (width * height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    while w * h > max_px:
        if w >= h:
            w -= 1
        else:
            h -= 1
    return w, h


def _seconds(stream, pts: int) -> float:
    return float((pts - (stream.start_time or 0)) * stream.time_base)


def _video_stream(container):
    """The first video stream, set to decode with frame threads. PyAV's default is slice threads, which help only a stream
    encoded in slices (a 1408x1408 camera clip decodes 3 to 7 times faster with frame threads). The frames are the same
    ones with the same pts, each coming out a few frames later, and the end of the file drains the last."""
    if not container.streams.video:
        raise ValueError("no video stream")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.thread_count = DECODE_THREADS
    return stream


def frame_times(path: Path) -> list[tuple[int, float]]:
    """Every video frame's `(pts, seconds into the file)`, in order."""
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        pts = sorted({packet.pts for packet in container.demux(stream) if packet.pts is not None})
        return [(p, _seconds(stream, p)) for p in pts]


def decode_frames(path: Path, wanted: list[tuple[int, float]], max_px: int = 0):
    """`(seconds, PIL image)` for the frames with these pts, from one seek and one pass. With `max_px` a frame is
    scaled down to the budget in the decoder's own scaler (bicubic), not decoded whole and resized.
    A frame is turned by its display matrix (a phone's portrait video is coded sideways), the box taken from the
    frame's own size."""
    want = dict(wanted)
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        first = min(want)
        if first > (stream.start_time or 0):
            container.seek(first, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts in want:
                box = fit(frame.width, frame.height, max_px)
                image = (frame.to_image(width=box[0], height=box[1], interpolation="BICUBIC")
                         if box != (frame.width, frame.height) else frame.to_image())
                yield want.pop(frame.pts), image.rotate(frame.rotation, expand=True) if frame.rotation else image
                if not want:
                    return


def audio_span(path: Path, t0: float, t1: float, rate: int = RATE) -> tuple[float, np.ndarray]:
    """The first audio stream from `t0` to a little past `t1` (seconds), as mono float32 in -1..1 at `rate`, from one
    seek, with the time of its first sample; empty when the file has no sound. The resampler is made again whenever a
    frame's format, layout or rate differs from the one before (joined files change rate)."""
    out, first, setup, resampler = [], None, None, None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return 0.0, np.zeros(0, np.float32)
        stream = container.streams.audio[0]
        start = stream.start_time or 0
        if t0 > 0:
            container.seek(int(t0 / stream.time_base) + start, stream=stream, backward=True)
        for frame in container.decode(stream):
            at = float((frame.pts - start) * stream.time_base) if frame.pts is not None else None
            if first is None:
                first = at or 0.0
            if setup != (frame.format.name, frame.layout.name, frame.sample_rate):
                if resampler is not None:
                    out += [part.to_ndarray().reshape(-1) for part in resampler.resample(None)]
                setup = frame.format.name, frame.layout.name, frame.sample_rate
                resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            out += [part.to_ndarray().reshape(-1) for part in resampler.resample(frame)]
            if at is not None and at > t1:
                break
        if resampler is not None:
            out += [part.to_ndarray().reshape(-1) for part in resampler.resample(None)]
    return first or 0.0, np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
