"""The source of A4: BirdVox-full-night on disk (`birdvox-full-night`): six autonomous recording units left out near Ithaca, New York, on the
night of 23 September 2015, each with one FLAC of about eleven hours (24 kHz, one channel), and for each a CSV of the flight
calls of migrating birds that an expert marked, `Time (s),Freq (Hz)`: the seconds after the unit's recording began, and the pitch he marked the
call at, to the Hz. The labels are points, with no species, and are those of an expert who may have missed faint calls. The units' start times
are in `utc-start-times.csv` (epoch seconds: the six units began within two seconds of each other but for unit 10, which began 2.7 hours later).

A4 is unit 3's night: its FLAC (the audio in this source is only unit 3's) and its 9,113 calls. The other units' calls (26,289) are read only to
tell where no call was marked anywhere: a stretch in which none of the six annotation files has a call is one in which the sky was quiet, and an
empty window is trusted only there (questions.py).

The recording goes into a dump as the recorder would have left it: hourly FLAC files, each of whole seconds but the last, cut from the source
sample for sample (decoded and encoded again by ffmpeg, so the samples are the source's) and with no tags (the container's encoder name
apart)."""

from __future__ import annotations

import csv
import functools
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "birdvox-full-night"
UNIT = "unit03"
UNITS = ("unit01", "unit02", "unit03", "unit05", "unit07", "unit10")
AUDIO = f"BirdVox-full-night_flac-audio_{UNIT}.flac"
ANNOTATIONS = "BirdVox-full-night_csv-annotations_{}.csv"
STARTS = "BirdVox-full-night_utc-start-times.csv"
RATE = 24000
HOUR_S = 3600


@dataclass(frozen=True)
class Call:
    t: float  # seconds after the unit's recording began
    hz: int  # the pitch the expert marked it at


@dataclass(frozen=True)
class Night:
    calls: tuple[Call, ...]  # unit 3's, in time order
    others: tuple[float, ...]  # the other units' calls, in seconds after unit 3's recording began, in time order
    start: datetime  # when unit 3's recording began by its clock (UTC, no zone)
    samples: int  # in unit 3's FLAC


def streaminfo(path: Path) -> tuple[int, int, int, int]:
    """`(rate, channels, bits per sample, samples)` from a FLAC's STREAMINFO block."""
    with open(path, "rb") as f:
        head = f.read(42)
    if head[:4] != b"fLaC" or head[4] & 0x7F != 0 or head[5:8] != b"\x00\x00\x22":
        raise ValueError(f"{path} is no FLAC that starts with its STREAMINFO block")
    packed = int.from_bytes(head[18:26], "big")  # 20 bits of rate, 3 of channels - 1, 5 of bits - 1, 36 of samples
    return packed >> 44, (packed >> 41 & 7) + 1, (packed >> 36 & 31) + 1, packed & (1 << 36) - 1


def flac_header(path: Path) -> bytes:
    """A FLAC's metadata blocks (its marker and every block before the first frame): the bytes a tag would be in."""
    with open(path, "rb") as f:
        out = bytearray(f.read(4))
        while True:
            head = f.read(4)
            out += head
            size = int.from_bytes(head[1:], "big")
            out += f.read(size)
            if not head or head[0] & 0x80:
                return bytes(out)


def read_times(path: Path) -> list[Call]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["Time (s)", "Freq (Hz)"]:
        raise ValueError(f"{path} does not start with the header Time (s),Freq (Hz)")
    return [Call(float(t), int(hz)) for t, hz in rows[1:]]


@functools.lru_cache(maxsize=2)
def read_night(source: str) -> Night:
    """Unit 3's calls and when its recording began, and the other units' calls on unit 3's clock. A call outside the recording, or calls out of
    time order, are refused."""
    root = Path(source)
    with open(root / STARTS, newline="") as f:
        starts = {unit: int(epoch) for unit, epoch in csv.reader(f) if unit.startswith("unit")}
    rate, channels, bits, samples = streaminfo(root / AUDIO)
    if (rate, channels, bits) != (RATE, 1, 16):
        raise ValueError(f"{AUDIO} is {rate} Hz, {channels} channels, {bits} bits: A4 reads 24 kHz, one channel, 16-bit")
    calls = read_times(root / ANNOTATIONS.format(UNIT))
    if [c.t for c in calls] != sorted(c.t for c in calls) or calls[0].t < 0 or calls[-1].t > samples / RATE:
        raise ValueError(f"the calls of {UNIT} are out of time order or outside its recording")
    others = sorted(starts[u] - starts[UNIT] + c.t for u in UNITS if u != UNIT for c in read_times(root / ANNOTATIONS.format(u)))
    return Night(tuple(calls), tuple(others), datetime.fromtimestamp(starts[UNIT], timezone.utc).replace(tzinfo=None), samples)


def cut(source: Path, first: int, seconds: int | None, out: Path) -> None:
    """The recording from second `first` for `seconds` (to its end when None) as a FLAC at `out`, sample for sample, with no tags."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(first), *(["-t", str(seconds)] if seconds else []),
               "-i", str(source / AUDIO), "-map_metadata", "-1", "-map", "0:a", "-c:a", "flac", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(out)]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"ffmpeg could not cut {AUDIO} from {first} s into {out}: {run.stderr.strip()[-500:]}")
