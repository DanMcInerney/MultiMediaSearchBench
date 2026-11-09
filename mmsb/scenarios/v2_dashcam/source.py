"""The source of V2: the Australian dashcam clips on disk (128 short clips of real dashcam footage from Australian roads, cut from two long
compilations; `<class>/<clip>.mp4`, 1280x720, 60 fps, no sound) and their one CSV (`video_metadata.csv`): for each clip its class (`Crash`,
`Near Crash`, `Normal`, `Outliers`, which the source's folders also say), and `ego` (whether the camera's own car is in it), `time` (`Day`
or `Night`), `weather` (`Clear`, `Cloudy` or `Rain`), `location` and `hazard` (what the other road user is: `Vehicles`, `Pedestrian`,
`Other` or `None`). Every clip has a row, and every row a clip, or the source is refused; a value the scenario does not know is refused too.

A clip goes into the dump as a dashcam would have saved it: remuxed by stream copy (the video as it is) with no metadata, so that
the class folders, the source's "produced by" handler name and the file's own name are gone."""

from __future__ import annotations

import csv
import functools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import imageio_ffmpeg

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "aus-dashcam"
TABLE = "video_metadata.csv"
HEADER = ["video_name", "classification", "ego", "time", "weather", "location", "hazard"]
KINDS = ("Crash", "Near Crash", "Normal", "Outliers")  # a clip is in the folder of its class
VALUES = {"ego": ("Ego-Centric", "Non-Ego-Centric"), "time": ("Day", "Night"), "weather": ("Clear", "Cloudy", "Rain"),
          "location": ("City", "Highway", "Intersections", "Parking Lot", "Residential", "Roundabouts", "Rural", "Tunnel"),
          "hazard": ("None", "Other", "Pedestrian", "Vehicles")}


@dataclass(frozen=True)
class Clip:
    name: str  # `20260106_tUVzDyp65K4_001.mp4`
    rel: str  # in the source: `Near Crash/<name>`
    kind: str  # its class: one of KINDS
    ego: str
    time: str
    weather: str
    location: str
    hazard: str
    seconds: float  # how long it plays


@functools.lru_cache(maxsize=2)
def read_clips(source: str) -> tuple[Clip, ...]:
    """The 128 clips of the CSV in its order, each with how long it plays (from its container). A CSV with another header, a value the
    scenario does not know, a row with no clip, a clip with no row, or a repeated name is refused."""
    root = Path(source)
    with open(root / TABLE, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != HEADER:
        raise ValueError(f"{root / TABLE} has the header {rows[0]}, not {HEADER}")
    clips, seen = [], set()
    for n, row in enumerate(rows[1:], 2):
        if len(row) != len(HEADER):
            raise ValueError(f"{TABLE}:{n}: {len(row)} fields, not {len(HEADER)}")
        name, kind, *rest = row
        values = dict(zip(VALUES, rest, strict=True))
        if kind not in KINDS or name in seen or any(values[k] not in VALUES[k] for k in VALUES):
            raise ValueError(f"{TABLE}:{n}: {row} has an unknown class or value, or repeats a clip")
        seen.add(name)
        rel = f"{kind}/{name}"
        if not (root / rel).is_file():
            raise FileNotFoundError(f"{TABLE}:{n}: the clip {root / rel} is not on disk")
        clips.append(Clip(name, rel, kind, values["ego"], values["time"], values["weather"], values["location"], values["hazard"],
                          seconds_of(root / rel)))
    on_disk = {f"{k}/{x}" for k in KINDS for x in os.listdir(root / k)}
    if extra := on_disk - {c.rel for c in clips}:
        raise ValueError(f"{len(extra)} clips on disk have no row in {TABLE} (the first is {sorted(extra)[0]})")
    return tuple(clips)


def seconds_of(path: Path) -> float:
    """How long the video of `path` plays, to the millisecond, from its container."""
    with av.open(str(path)) as c:
        return round(float(c.duration) / av.time_base, 3)


def remux(src: Path, out: Path) -> None:
    """The video of `src` as an MP4 at `out` by stream copy (no re-encode): without the metadata, the source's handler name, or ffmpeg's
    name and the time it was made."""
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", "0:v", "-c", "copy",
                          "-an", "-map_metadata", "-1", "-map_metadata:s:v", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                          "-movflags", "+faststart", "-y", str(out)], capture_output=True, text=True, check=False)
    if run.returncode:
        raise RuntimeError(f"ffmpeg could not remux {src} into {out}: {run.stderr.strip()[-500:]}")


def mp4_header(path: Path) -> bytes:
    """An MP4's top-level boxes but the payload of `mdat` (the video): what the file says about itself, for the leak probe."""
    out = bytearray()
    with open(path, "rb") as f:
        while head := f.read(8):
            size, kind = int.from_bytes(head[:4], "big"), head[4:]
            if size == 1:  # a 64-bit size follows
                head += f.read(8)
                size = int.from_bytes(head[8:], "big")
            if len(head) < 8 or size and size < len(head):
                raise ValueError(f"{path} is not an MP4: a box of {size} bytes")
            out += head
            rest = size - len(head) if size else None  # None: the box goes to the end of the file
            if kind == b"mdat" and rest is not None:
                f.seek(rest, os.SEEK_CUR)
            elif kind != b"mdat":
                out += f.read(rest)
            if rest is None:
                break
    return bytes(out)
