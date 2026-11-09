"""The source of V3: the SoccerTrack v2 follow-cam clips on disk (`soccertrack/clips/st<match>_h<half>_c<index>.mp4`: 30 seconds each, 1280x720,
25 fps, no sound; a virtual camera aimed with lag at the play and cropped out of the fixed panoramic picture of an amateur university match)
and their labels (`soccertrack/labels/<clip>.Labels-ball.json`): for each clip its match, its half, `pano_start_frame` (the frame of the half
the clip begins at, 25 to the second) and the ball events in it, each with its class, `position` (milliseconds into the clip), `team` and
`source`. Only the matches that have clips on disk are read; the labels of other matches are left alone.

The clips of a half are its consecutive 30-second windows, but for the windows the source dropped (a clip with fewer than three events is not
kept): `read_clips` refuses a start that is not a whole window, and `gaps` says which windows are missing. A class is read only when the source
marks it as made by a person (`soccertrack:` in `source`: pass, ball_out_of_play, tackle, interception, shot, block, aerial_duel and goal); the
other two, `pass_received` and `recovery`, are derived by a script from the next touch and are dropped here, so that no question can rest on
them. `team` is `left` or `right`: the team on that side of the pitch in the camera's picture in that half (they change ends at half time),
which is the team attacking the other side's goal."""

from __future__ import annotations

import functools
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "st2-followcam"
LABELS = "soccertrack/labels"
CLIPS = "soccertrack/clips"
FPS, FRAMES = 25, 750  # of every clip: 30 seconds
SECONDS = FRAMES / FPS
HUMAN = ("pass", "ball_out_of_play", "tackle", "interception", "shot", "block", "aerial_duel", "goal")  # made by a person (`soccertrack:...`)
DERIVED = ("pass_received", "recovery")  # made by a script from the next touch: not read
TEAMS = ("left", "right")
LABEL_FILE = re.compile(r"st(\d+)_h([12])_c(\d+)\.Labels-ball\.json")


@dataclass(frozen=True)
class Event:
    label: str  # one of HUMAN
    pos: float  # seconds into the clip
    team: str  # `left` or `right`


@dataclass(frozen=True)
class Clip:
    name: str  # `st117092_h1_c056.mp4`
    match: str  # `117092`
    half: int
    window: int  # which 30-second window of the half it is: 0 is the first
    events: tuple[Event, ...]  # in time order


def matches_on_disk(source: str) -> tuple[str, ...]:
    """The match ids that have at least one clip in the source's clips folder, in order."""
    found = {m[1] for name in os.listdir(Path(source) / CLIPS) if (m := re.fullmatch(r"st(\d+)_h[12]_c\d+\.mp4", name))}
    return tuple(sorted(found))


@functools.lru_cache(maxsize=2)
def read_clips(source: str) -> tuple[Clip, ...]:
    """The clips of every match that has clips on disk, by match, half and window. A label file whose clip is not on disk (one match is
    all or nothing), a file that does not say what its name does, a class or team the scenario does not know, a class marked as made by a
    person that the source says a script made (or the other way), and a start that is no whole window are refused."""
    root = Path(source)
    wanted = set(matches_on_disk(source))
    if not wanted:
        raise FileNotFoundError(f"no clip of a match is in {root / CLIPS}")
    clips = []
    for name in sorted(os.listdir(root / LABELS)):
        m = LABEL_FILE.fullmatch(name)
        if m is None or m[1] not in wanted:
            continue
        clip = f"st{m[1]}_h{m[2]}_c{m[3]}.mp4"
        if not (root / CLIPS / clip).is_file():
            raise FileNotFoundError(f"{name}: the clip {root / CLIPS / clip} is not on disk, though other clips of its match are")
        data = json.loads((root / LABELS / name).read_text(encoding="utf-8"))
        if (data["match"], data["half"], data["clip"], data["fps"], data["num_frames"]) != (m[1], m[2], clip, FPS, FRAMES):
            raise ValueError(f"{name} says it is another clip or another kind of clip than its name does")
        if data["pano_start_frame"] % FRAMES:
            raise ValueError(f"{name}: the clip starts at frame {data['pano_start_frame']}, which is no whole window of {FRAMES} frames")
        events = []
        for a in data["annotations"]:
            label, made = a["label"], a["source"].split(":")[0]
            if label not in (*HUMAN, *DERIVED) or a["team"] not in TEAMS or made != ("soccertrack" if label in HUMAN else "derived"):
                raise ValueError(f"{name}: an event of class {label!r}, team {a['team']!r}, source {a['source']!r} that the scenario does not know")
            pos = int(a["position"]) / 1000
            if not 0 <= pos < SECONDS:
                raise ValueError(f"{name}: an event at {pos} s in a clip of {SECONDS} s")
            if label in HUMAN:
                events.append(Event(label, pos, a["team"]))
        clips.append(Clip(clip, m[1], int(m[2]), data["pano_start_frame"] // FRAMES, tuple(sorted(events, key=lambda e: e.pos))))
    return tuple(sorted(clips, key=lambda c: (c.match, c.half, c.window)))


def gaps(clips: tuple[Clip, ...]) -> set[tuple[str, int, int]]:
    """The `(match, half, window)` of every window missing between a half's first clip and its last."""
    out: set[tuple[str, int, int]] = set()
    by: dict[tuple[str, int], list[int]] = {}
    for c in clips:
        by.setdefault((c.match, c.half), []).append(c.window)
    for (match, half), windows in by.items():
        out |= {(match, half, w) for w in range(min(windows), max(windows) + 1)} - {(match, half, w) for w in windows}
    return out


def mp4_header(path: Path) -> bytes:
    """An MP4's top-level boxes but the payload of `mdat` (the video): what the file says about itself, for the leak probe."""
    out = bytearray()
    with open(path, "rb") as f:
        while head := f.read(8):
            size, kind = struct.unpack(">I4s", head)
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
