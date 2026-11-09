"""The source of I2: the Android in the Wild screenshots on disk. `screenshots/<episode>_<step>.png` are phone screens (PNG, 270 to 720 px wide, by
device) of people doing a task on a phone, one episode of them a task, and `screenshots_labels.jsonl` gives each one's episode, step, the task
(`goal`, in the words it was given to the person), the device, the app on top (`activity`) and the text OCR read off the screen (`ocr_text`, a line
for each text box, imperfect). Episodes are on disk as far as their prefix: an episode whose `episode_length` is more than the steps on disk is
cut, and a question is asked only of an episode that is whole.

The screens of an episode are what a person on a phone sees: its first screens are often the last page of the task before it (the same phone), and
the history, the suggestions and the recent searches of the browser show the tasks of other episodes. A task's goal is therefore a label of its
screens, not a promise that every screen of the episode shows it."""

from __future__ import annotations

import functools
import json
import os
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "android-in-the-wild"
LABELS = "screenshots_labels.jsonl"
PNG = b"\x89PNG\r\n\x1a\n"
PLAIN_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}  # a screenshot of the source has these and no other


@dataclass(frozen=True)
class Shot:
    file: str  # in the source, `screenshots/<episode>_<step>.png`
    episode: str
    step: int
    activity: str  # the app on top: `com.android.chrome/...`
    ocr: tuple[str, ...]  # the text lines OCR read, in the order of its boxes; empty lines left out


@dataclass(frozen=True)
class Episode:
    id: str
    goal: str
    device: str
    length: int  # the steps the episode has; its shots are those of them that are on disk (or, in a folder, that were kept: tasks.py `trim`)
    shots: tuple[Shot, ...]  # in the order of the steps

    @property
    def whole(self) -> bool:
        return len(self.shots) == self.length


def fold(text: str) -> str:
    """`text` casefolded, quotes straightened, every run of other than letters and digits one space: what a phrase is found in."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'").replace("'", "")))


@functools.cache
def seen(shot: Shot) -> str:
    """The screen's text, folded, as one string with a space at each end: a folded phrase between spaces is found in it as whole words."""
    return " " + fold(" ".join(shot.ocr)) + " "


@functools.lru_cache(maxsize=2)
def read_labels(source: str) -> tuple[Episode, ...]:
    """The episodes of `source`, by episode id, each with its shots in the order of their steps. A row with other keys, an episode with two goals
    or devices, steps that do not run from 0 without a gap, and a screenshot that is not on disk are refused."""
    root = Path(source)
    rows = defaultdict(list)
    with open(root / LABELS, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            row = json.loads(line)
            if missing := {"file", "episode_id", "step_id", "episode_length", "goal", "device", "activity", "ocr_text"} - set(row):
                raise ValueError(f"{LABELS}:{n} lacks {', '.join(sorted(missing))}")
            rows[row["episode_id"]].append(row)
    have = set(os.listdir(root / "screenshots"))
    episodes = []
    for episode, found in sorted(rows.items()):
        found.sort(key=lambda r: r["step_id"])
        if [r["step_id"] for r in found] != list(range(len(found))):
            raise ValueError(f"episode {episode}: its steps are not 0 to {len(found) - 1} with no gap")
        if len({(r["goal"], r["device"], r["episode_length"]) for r in found}) != 1:
            raise ValueError(f"episode {episode}: its rows give more than one goal, device or length")
        if lacking := [r["file"] for r in found if r["file"].removeprefix("screenshots/") not in have]:
            raise FileNotFoundError(f"{lacking[0]} is not in {root / 'screenshots'}: `mmsb fetch --scenario i2_screenshots` gets it")
        first = found[0]
        episodes.append(Episode(episode, first["goal"], first["device"], first["episode_length"],
                                tuple(Shot(r["file"], episode, r["step_id"], r["activity"], tuple(t for t in r["ocr_text"] if t.strip())) for r in found)))
    return tuple(episodes)


def png_chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    """The chunks of a PNG, `(type, data)` in order; refused when it is no PNG or its chunks run past its end."""
    if raw[:8] != PNG:
        raise ValueError("not a PNG")
    out, at = [], 8
    while at < len(raw):
        if at + 12 > len(raw):
            raise ValueError("a PNG chunk is cut short")
        (size,) = struct.unpack(">I", raw[at:at + 4])
        if at + 12 + size > len(raw):
            raise ValueError("a PNG chunk runs past the end of the file")
        out.append((raw[at + 4:at + 8], raw[at + 8:at + 8 + size]))
        at += 12 + size
    return out


def png_plain(raw: bytes) -> bytes:
    """`raw` as it goes into the dump, a screenshot as the phone wrote it: a PNG of its header, image data and end alone (the source's all are);
    refused when it has any other chunk, text, a time or a colour profile, which would be metadata to carry or to strip."""
    if extra := {t for t, _ in png_chunks(raw)} - PLAIN_CHUNKS:
        raise ValueError(f"the PNG has the chunks {', '.join(sorted(t.decode() for t in extra))}")
    return raw


def png_header(path: Path) -> bytes:
    """A PNG's chunks but its image data: its header and any metadata, for the leak probe."""
    return b"".join(t + d for t, d in png_chunks(Path(path).read_bytes()) if t != b"IDAT")
