"""The source of A1: the SINS database on disk (a week of a person living in a vacation home, recorded by microphone arrays in the rooms;
here node 7, the living room) and its labels. The node's audio is one FLAC of about a minute at a time (`node07/<day>/Node7_YYYYMMDD_
HHMMSS_FFF_audio.flac`: four channels, 16 kHz, the name the start time to the millisecond, as the recorder's clock read it), and the
labels are CSV files of `Class;Start time;Stop time` with wall-clock times to the millisecond, no zone: `living_labels.csv` for what is
done in the living room, `labels.csv` for the whole home (where `absence` is nobody home).

A1's recording is the minute files joined into one WAV, one channel (the first) as recorded, 16-bit and 16 kHz, with the DC offset it carries
(about half the range) kept. Each minute file goes where its name puts it, so that a moment's seconds into the WAV are its wall time
since the WAV began: where one file ends before the next begins (each plays a tenth of a second short of its minute, and the recorder's
clock jumps a second now and then) the line is held at its last sample (zeros would click, as the signal rides on that offset), and
where one overlaps the next the start of the next is trimmed."""

from __future__ import annotations

import csv
import functools
import re
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import av
import numpy as np

from mmmb.scenarios.realdump import sources_root

SOURCE = sources_root() / "sins"
NODE = "node07"
LIVING, HOME = "labels/living_labels.csv", "labels/labels.csv"
STAMP = re.compile(r"Node7_(\d{8})_(\d{6})_(\d{3})_audio\.flac")
RATE = 16000
LONGEST = timedelta(seconds=61)  # no minute file plays longer; one that began this long before the recording may still reach into it


@dataclass(frozen=True)
class Run:
    cls: str  # the label
    start: datetime  # wall clock, no zone
    stop: datetime


@functools.lru_cache(maxsize=2)
def read_runs(source: str) -> tuple[tuple[Run, ...], tuple[Run, ...]]:
    """The runs of the living room's labels and of the whole home's, each in the order of the file."""
    def read(name: str) -> tuple[Run, ...]:
        with open(Path(source) / name, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f, delimiter=";"))
        if rows[0] != ["Class", "Start time", "Stop time"]:
            raise ValueError(f"{name} does not start with the header Class;Start time;Stop time")
        return tuple(Run(c, *(datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in (a, b))) for c, a, b in rows[1:])
    return read(LIVING), read(HOME)


@dataclass(frozen=True)
class Minute:
    path: Path
    start: datetime  # from its name


def minute_files(source: str, first: datetime, last: datetime) -> list[Minute]:
    """The minute files that play in `first` to `last` (wall clock, no zone), in the order of their start times."""
    found = []
    for day in dict.fromkeys((first.date() - timedelta(days=1), first.date(), last.date())):
        for path in sorted((Path(source) / NODE / day.isoformat()).glob("Node7_*_audio.flac")):
            m = STAMP.fullmatch(path.name)
            if m is None:
                raise ValueError(f"{path} is no minute file of node 7")
            start = datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S") + timedelta(milliseconds=int(m[3]))
            if first - LONGEST < start < last:
                found.append(Minute(path, start))
    if not found:
        raise FileNotFoundError(f"no minute file of {NODE} plays between {first} and {last} in {source}")
    return sorted(found, key=lambda m: m.start)


def first_channel(path: Path) -> np.ndarray:
    """The first channel of a FLAC, 16-bit."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if stream.rate != RATE or stream.format.name != "s16":
            raise ValueError(f"{path} is {stream.rate} Hz, {stream.format.name}: A1 reads 16 kHz, 16-bit minute files")
        frames = [f.to_ndarray()[0] if f.format.is_planar else f.to_ndarray().reshape(-1, stream.channels)[:, 0]
                  for f in container.decode(audio=0)]
    return np.concatenate(frames)


def write_wav(source: str, first: datetime, seconds: int, path: Path) -> None:
    """Writes the recording of `seconds` from `first` (wall clock, no zone) into `path` as a WAV (see the module's text)."""
    total, at, last = seconds * RATE, 0, None
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        for minute in minute_files(source, first, first + timedelta(seconds=seconds)):
            x = first_channel(minute.path)
            begins = round((minute.start - first).total_seconds() * RATE)
            if begins > at and at < total:  # a gap: the line, held
                held = np.full(min(begins, total) - at, x[0] if last is None else last, dtype="<i2")
                out.writeframes(held.tobytes())
                at += len(held)
            skip = max(0, at - begins)  # an overlap: the start of this file is dropped
            x = x[skip:skip + total - at]  # (and its end where the recording ends)
            if len(x):
                out.writeframes(x.astype("<i2").tobytes())
                at, last = at + len(x), x[-1]
        if at < total:
            out.writeframes(np.full(total - at, last, dtype="<i2").tobytes())

