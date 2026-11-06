"""The source of V1: the MEVA security cameras on disk. Of its three visible cameras (the infrared ones and the night clip have no official
labels, so a question about them could be wrong), 10:00 to 11:15 on 12 March 2018: five-minute H.264 clips (`video/<date>/<hour>/<clip>.avi`,
30 fps, no sound, named `<date>.<start>.<end>.<site>.<camera>.r13`, the times a wall clock with no zone) and Kitware's labels of each
(`<set>/<date>/<hour>/<clip>.activities.yml`, KPF: for each activity its type, a status and its span in frames of the clip). Only the two
official Kitware sets are read (`contrib/` holds other groups'); each clip is in exactly one, or is refused.

A KPF line is a Python literal, so `ast.literal_eval` reads it, a line at a time (PyYAML is not a dependency, and a file whose line is not
one is refused). A clip with no activity says so (`marked empty_37`) in a `meta` line; a file with neither is refused.

A camera's unbroken run of clips (each begins where the last ended) goes into the dump as one MP4 by stream copy: the clips' video as it
is, and no title, source path or encoder name. The cameras' names in the dump are neutral (`cam1` to `cam3`)."""

from __future__ import annotations

import ast
import functools
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import imageio_ffmpeg

SOURCE = Path(r"D:\bench-data\sources\meva")
DAY = "2018-03-12"
VIDEO = f"video/{DAY}"
LABELS = "annotations/meva-data-repo-master/annotation/DIVA-phase-2/MEVA"
SETS = ("kitware", "kitware-meva-training")  # the two official sets of Kitware's labels
CAMERAS = {"G336": "cam1", "G341": "cam2", "G436": "cam3"}  # the source's camera -> the dump's name for it
FIRST, LAST = datetime(2018, 3, 12, 10, 0), datetime(2018, 3, 12, 11, 15)  # the clips that begin from FIRST until LAST are used
FPS = 30
EDGE_FRAMES = FPS  # a label this close to the start or end of its clip may go on in the next clip (or come from the last)
CLIP = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d\d)-(\d\d)-(\d\d)\.(\d\d)-(\d\d)-(\d\d)\.(\w+)\.(G\d+)\.r13")
EMPTY = "marked empty_37"  # what a `meta` line says of a clip with no activity


@dataclass(frozen=True)
class Act:
    kind: str  # an activity of the labels: `person_opens_trunk`
    status: str  # the label's own: good, unaudited, not_good
    f0: int  # its span, in frames of its clip
    f1: int


@dataclass(frozen=True)
class Clip:
    name: str  # `2018-03-12.10-00-00.10-05-00.hospital.G341`
    rel: str  # the video, in the source
    camera: str  # the dump's name for its camera
    start: datetime  # as its name says (a wall clock, no zone)
    end: datetime
    frames: int
    acts: tuple[Act, ...]


@dataclass(frozen=True)
class Event:
    """A labelled activity placed in the dump: its file, and its span as seconds into the file and as epoch seconds."""

    camera: str
    file: str
    kind: str
    status: str
    o0: float
    o1: float
    t0: float
    t1: float
    edge: bool  # within a second of the start or end of its clip


def read_acts(path: Path, frames: int) -> tuple[Act, ...]:
    """The activities of a `.activities.yml` of a clip of `frames` frames. A line that is not a KPF activity or meta line, a span outside
    the clip, a file with no activity that is not marked empty, and a marked-empty one with activities are refused."""
    acts, empty = [], False
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        try:
            if not line.startswith("- "):
                raise ValueError
            item = ast.literal_eval(line[2:])
            if set(item) == {"meta"}:
                empty = empty or EMPTY in item["meta"]
                continue
            act = item["act"]
            ((kind, _),) = act["act2"].items()  # one activity
            (span,) = act["timespan"]  # in one span
            ((key, (f0, f1)),) = span.items()
            status = act["src_status"]
            if key != "tsr0" or not isinstance(kind, str) or not isinstance(status, str) or not 0 <= f0 <= f1 < frames:
                raise ValueError
        except (ValueError, SyntaxError, KeyError, TypeError):
            raise ValueError(f"{where}: not a KPF activity or meta line of a clip of {frames} frames") from None
        acts.append(Act(kind, status, f0, f1))
    if bool(acts) == empty:
        raise ValueError(f"{path} has {'activities and says the clip is empty' if acts else 'no activity and does not say the clip is empty'}")
    return tuple(acts)


def frames_of(path: Path) -> int:
    """The frames of a clip, from its container's index; a clip that is not 30 fps video alone is refused."""
    with av.open(str(path)) as c:
        video = c.streams.video[0]
        if c.streams.audio or video.average_rate != FPS or not video.frames:
            raise ValueError(f"{path} is not 30 fps video with no sound")
        return video.frames


@functools.lru_cache(maxsize=2)
def read_clips(source: str) -> tuple[Clip, ...]:
    """The clips of the three cameras that begin from 10:00 until 11:15, by camera (cam1 to cam3) and time. A clip with no label file, or
    in both sets, is refused."""
    root = Path(source)
    found = []
    for hour in sorted(os.listdir(root / VIDEO)):
        for name in sorted(os.listdir(root / VIDEO / hour)):
            m = CLIP.fullmatch(name.removesuffix(".avi"))
            if name.endswith(".avi") and m and m[9] in CAMERAS:
                a = datetime.fromisoformat(f"{m[1]}T{m[2]}:{m[3]}:{m[4]}")
                if FIRST <= a < LAST:
                    found.append((CAMERAS[m[9]], a, datetime.fromisoformat(f"{m[1]}T{m[5]}:{m[6]}:{m[7]}"), name.removesuffix(".r13.avi"),
                                  f"{VIDEO}/{hour}/{name}", hour))
    if not found:
        raise FileNotFoundError(f"no clip of {', '.join(CAMERAS)} between {FIRST} and {LAST} is in {root / VIDEO}")
    clips = []
    for camera, start, end, name, rel, hour in sorted(found):
        labels = [p for s in SETS if (p := root / LABELS / s / DAY / hour / f"{name}.activities.yml").is_file()]
        if len(labels) != 1:
            raise ValueError(f"{name} has {'no label file' if not labels else 'a label file in both sets'}: a clip is in one of {', '.join(SETS)}")
        frames = frames_of(root / rel)
        clips.append(Clip(name, rel, camera, start, end, frames, read_acts(labels[0], frames)))
    return tuple(clips)


def runs_of(clips: tuple[Clip, ...]) -> list[tuple[Clip, ...]]:
    """`clips` (by camera and time) split into each camera's unbroken runs: a clip that does not begin where the one before it ended
    starts another."""
    out: list[list[Clip]] = []
    for c in clips:
        if out and out[-1][-1].camera == c.camera and out[-1][-1].end == c.start:
            out[-1].append(c)
        else:
            out.append([c])
    return [tuple(r) for r in out]


def events_in(path: str, camera: str, t: float, run: tuple[Clip, ...]) -> list[Event]:
    """The activities of the clips `run`, which are the file `path` that starts at `t` (epoch seconds), placed in it."""
    out, done = [], 0
    for c in run:
        for a in c.acts:
            o0, o1 = (done + a.f0) / FPS, (done + a.f1) / FPS
            out.append(Event(camera, path, a.kind, a.status, o0, o1, t + o0, t + o1,
                             a.f0 < EDGE_FRAMES or a.f1 >= c.frames - EDGE_FRAMES))
        done += c.frames
    return out


@functools.lru_cache(maxsize=2)
def activity_names(source: str) -> tuple[str, ...]:
    """The activities of the dataset's labels: those its `status.csv` lists and those the clips have."""
    listed = (Path(source) / LABELS / SETS[0] / "status.csv").read_text(encoding="utf-8")
    names = set(re.findall(r"^([a-z]+(?:_[a-z0-9]+)+),\s*\d+", listed, re.MULTILINE))
    return tuple(sorted(names | {a.kind for c in read_clips(source) for a in c.acts}))


def join(clips: list[Path], out: Path) -> None:
    """The video of `clips`, in order, as one MP4 at `out` by stream copy (ffmpeg's concat; no re-encode): without the clips' metadata, or
    ffmpeg's name and the time it was made."""
    with tempfile.TemporaryDirectory() as folder:
        listing = Path(folder) / "clips.txt"
        listing.write_text("".join(f"file '{Path(c).resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
        run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                              str(listing), "-map", "0:v", "-c", "copy", "-an", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v",
                              "+bitexact", "-y", str(out)], capture_output=True, text=True, check=False)
    if run.returncode:
        raise RuntimeError(f"ffmpeg could not join {len(clips)} clips into {out}: {run.stderr.strip()[-500:]}")


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
