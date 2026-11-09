"""The source of V10: Charades on disk (AI2's 9,848 short videos of people acting out scripted household activities in their own homes, filmed
by crowd workers; the videos are `Charades_v1_480.zip`, `Charades_v1_480/<id>.mp4`, read member by member and never unpacked) and its labels
(`Charades.zip`, `Charades/Charades_v1_train.csv` and `_test.csv`): for each video its `scene` (one of 15 rooms, always given), `quality` and
`relevance` (an annotator's 1 to 7), `verified` (an annotator saw that the video matches its script), `script` and `descriptions` (what was acted,
in words, and what annotators saw), `objects` (the script's), `actions` (`class start end` for each of 157 classes, in seconds) and `length`.

THE VIDEOS ARE ACTED. People were given a script ("a person opens a refrigerator, then sneezes") and filmed themselves doing it. What a pile of
them shows is staged home life, and the labels are the dataset's own: noisy, and not exhaustive (an action that happens may have no label,
and the script and descriptions do not name every action either). Nothing here is certain without that said: questions.py asks of an action only
where a clip's label and its words agree, and keeps out of the pile every clip where they do not (`pile.py`).

A video's bytes go into the dump as they are (the source has no metadata that names a label): each member of the zip is copied out whole."""

from __future__ import annotations

import csv
import functools
import io
import os
import shutil
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "charades"
LABELS, VIDEOS = "Charades.zip", "Charades_v1_480.zip"
TABLES = ("Charades/Charades_v1_train.csv", "Charades/Charades_v1_test.csv")
CLASSES, OBJECT_CLASSES = "Charades/Charades_v1_classes.txt", "Charades/Charades_v1_objectclasses.txt"
MEMBER = "Charades_v1_480/{}.mp4"
HEADER = ["id", "subject", "scene", "quality", "relevance", "verified", "script", "objects", "descriptions", "actions", "length"]
SCENES = ("Basement", "Bathroom", "Bedroom", "Closet", "Dining room", "Entryway", "Garage", "Hallway", "Home Office", "Kitchen", "Laundry room",
          "Living room", "Other", "Pantry", "Recreation room", "Stairs")  # a scene in the CSV starts with one of these ("Basement (A room below ...)")


@dataclass(frozen=True)
class Clip:
    id: str
    scene: str  # one of SCENES
    quality: int  # 0 when the CSV gives none
    relevance: int
    verified: bool
    text: str  # the script and the descriptions, in lower case: what the words say happens
    objects: frozenset[str]  # the script's, in lower case
    spans: tuple[tuple[str, float, float], ...]  # `(class, start, end)` in seconds into the clip, in the order of the CSV; ends cut to the length
    length: float  # seconds
    flawed: bool = False  # an action of it ends before it starts, or starts at or after the video's end: the labels of the clip are not to be trusted

    @property
    def member(self) -> str:
        return MEMBER.format(self.id)

    def classes(self) -> set[str]:
        return {c for c, _, _ in self.spans}


def scene_of(text: str) -> str:
    for scene in SCENES:
        if text.startswith(scene):
            return scene
    raise ValueError(f"the scene {text!r} is not one of {', '.join(SCENES)}")


@functools.lru_cache(maxsize=2)
def class_names(source: str) -> dict[str, str]:
    """The 157 action classes of the source, `c008` to `Opening a door`."""
    with zipfile.ZipFile(Path(source) / LABELS) as z:
        lines = z.read(CLASSES).decode("utf-8").splitlines()
    names = dict(line.strip().partition(" ")[::2] for line in lines if line.strip())
    if len(names) != 157:
        raise ValueError(f"{CLASSES} has {len(names)} classes, not 157")
    return names


@functools.lru_cache(maxsize=2)
def object_names(source: str) -> tuple[str, ...]:
    with zipfile.ZipFile(Path(source) / LABELS) as z:
        lines = z.read(OBJECT_CLASSES).decode("utf-8").splitlines()
    return tuple(line.strip().partition(" ")[2] for line in lines if line.strip())


@functools.lru_cache(maxsize=2)
def read_clips(source: str) -> tuple[Clip, ...]:
    """Every video of the CSVs, train and test, in file order. A header that is not HEADER, a repeated id, a scene or class the source does
    not list, and an action that does not read `class start end` are refused. An action that ends before it starts, or starts at or after the video's
    end, is dropped and marks its clip `flawed` (the CSV has a few); one that ends after the video's end is cut to it."""
    names = class_names(source)
    out: list[Clip] = []
    seen: set[str] = set()
    with zipfile.ZipFile(Path(source) / LABELS) as z:
        for table in TABLES:
            rows = list(csv.reader(io.TextIOWrapper(z.open(table), encoding="utf-8", newline="")))
            if rows[0] != HEADER:
                raise ValueError(f"{table} has the header {rows[0]}, not {HEADER}")
            for n, row in enumerate(rows[1:], 2):
                if len(row) != len(HEADER):
                    raise ValueError(f"{table}:{n}: {len(row)} fields, not {len(HEADER)}")
                id_, _, scene, quality, relevance, verified, script, objects, descriptions, actions, length = row
                if id_ in seen or verified not in ("Yes", "No"):
                    raise ValueError(f"{table}:{n}: {id_!r} is repeated or says verified {verified!r}")
                seen.add(id_)
                seconds = float(length)
                spans, flawed = [], False
                for token in filter(None, actions.split(";")):
                    parts = token.split()
                    if len(parts) != 3 or parts[0] not in names:
                        raise ValueError(f"{table}:{n}: the action {token!r} is not `class start end` of a class of the source")
                    start, end = float(parts[1]), float(parts[2])
                    if 0 <= start <= end and start < seconds:
                        spans.append((parts[0], start, min(end, seconds)))
                    else:
                        flawed = True
                out.append(Clip(id_, scene_of(scene), int(quality or 0), int(relevance or 0), verified == "Yes", f"{script} . {descriptions}".casefold(),
                                frozenset(o for o in objects.casefold().split(";") if o), tuple(spans), seconds, flawed))
    return tuple(out)


@functools.lru_cache(maxsize=2)
def members(source: str) -> frozenset[str]:
    """The names of the files in the videos zip (its central directory is read; the videos are not)."""
    with zipfile.ZipFile(Path(source) / VIDEOS) as z:
        return frozenset(z.namelist())


def check_members(source: str, clips: list[Clip]) -> None:
    """Refuses `clips` when any has no video in the videos zip: a pile of fewer videos than its labels need is another pile."""
    have = members(source)
    if lacking := [c.member for c in clips if c.member not in have]:
        raise FileNotFoundError(f"{len(lacking)} of the {len(clips)} videos this pile needs are not in {Path(source) / VIDEOS} (the first is "
                                f"{lacking[0]}): `mmsb fetch --scenario v10_home_clips` gets it")


_OPEN = threading.local()


def copy_member(source: str, member: str, out: Path) -> None:
    """The member of the videos zip, whole and as it is, written at `out` (each thread keeps its own handle on the zip)."""
    path = str(Path(source) / VIDEOS)
    if getattr(_OPEN, "path", None) != path:
        _OPEN.zip, _OPEN.path = zipfile.ZipFile(path), path
    with _OPEN.zip.open(member) as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


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
