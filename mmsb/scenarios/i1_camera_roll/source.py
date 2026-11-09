"""The source of I1: ATM-Bench's photos on disk (`data/raw_memory/image/<YYYYMMDD_HHMMSS>.jpg`: one person's phone photos over about four
years, resized to about 1000 by 750, the name their capture time to the second, EXIF with GPS) and its questions (`data/atm-bench/atm-bench.json`:
1,013 questions in the owner's own words, each with the ids of the photos, videos and emails its answer rests on). A question whose evidence is
all photos has a stated answer: for `list_recall` the photos' ids, for `number` mostly a date, for `open_end` a sentence.

A photo goes into the dump as a phone would have left it: numbered in the order taken, with EXIF holding the time it was taken and nothing else
(no GPS, no device, no software), and the same time as its modified time. 28 files have a suffix after the time (`(0)`, `_001`, `_resized`);
they are photos of the pile like the rest, and a question rests on a photo only when its id is exactly one file's name and no other file has the
same time."""

from __future__ import annotations

import functools
import io
import json
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from PIL import Image

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "atm-bench"
QUESTIONS = "data/atm-bench/atm-bench.json"
IMAGES = "data/raw_memory/image"
STEM = re.compile(r"(\d{8}_\d{6})")
ORIENTATION = 0x0112


@dataclass(frozen=True)
class Photo:
    file: str  # its name in the source's images folder
    wall: datetime  # when it was taken, from its name (no zone: the clock of the place it was taken in)
    alone: bool  # its time is the name's whole stem, and no other photo has that time: a question may rest on it


@dataclass(frozen=True)
class Real:
    id: str  # the dataset's id of the question (a uuid)
    question: str
    answer: str
    qtype: str  # list_recall, number or open_end
    evidence: tuple[str, ...]  # the ids (the times) of the photos the answer rests on, in the order given


@functools.lru_cache(maxsize=2)
def read_photos(source: str) -> tuple[Photo, ...]:
    """Every photo of the source, in the order taken (then by name)."""
    files = sorted(f for f in os.listdir(Path(source) / IMAGES) if f.lower().endswith(".jpg"))
    stems: dict[str, int] = {}
    for f in files:
        stems[f[:15]] = stems.get(f[:15], 0) + 1
    out = []
    for f in files:
        if STEM.fullmatch(f[:15]) is None:
            raise ValueError(f"{f} is not named by its time")
        out.append(Photo(f, datetime.strptime(f[:15], "%Y%m%d_%H%M%S"), f == f"{f[:15]}.jpg" and stems[f[:15]] == 1))
    return tuple(sorted(out, key=lambda p: (p.wall, p.file)))


@functools.lru_cache(maxsize=2)
def read_real(source: str) -> tuple[Real, ...]:
    """The questions whose evidence is all photos that a question may rest on (`Photo.alone`), in the dataset's order."""
    with open(Path(source) / QUESTIONS, encoding="utf-8") as f:
        data = json.load(f)
    alone = {p.file[:15] for p in read_photos(source) if p.alone}
    return tuple(Real(q["id"], q["question"], q["answer"], q["qtype"], tuple(q["evidence_ids"])) for q in data
                 if q["evidence_ids"] and all(e in alone for e in q["evidence_ids"]))


def check_photos(source: str, files: Sequence[str]) -> None:
    """Refuses `files` when any is not in `source`'s images folder."""
    have = set(os.listdir(Path(source) / IMAGES))
    if lacking := [f for f in files if f not in have]:
        raise FileNotFoundError(f"{len(lacking)} of the {len(files)} photos this pile needs are not in {Path(source) / IMAGES} (the first is "
                                f"{lacking[0]}): the ATM-Bench download holds them")


# -- the stated date of an answer -----------------------------------------------------------------------------------------------------------

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
NAME = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?"
        r"|dec(?:ember)?)")
MONTH = rf"({NAME})\.?"
DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}\b,?\s+(\d{{4}})\b", re.IGNORECASE)
MONTH_FIRST = re.compile(rf"\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?[,.]?\s+(\d{{4}})\b", re.IGNORECASE)
COMPACT = re.compile(r"\s*(\d{4})(\d{2})(\d{2})\s*")
MONTH_WORD = re.compile(rf"\b{NAME}\b", re.IGNORECASE)
CLOCK = re.compile(r"\d:\d{2}|\b\d{1,2}\s*[ap]\.?m\b", re.IGNORECASE)  # a time of day: an answer that has one is no date alone


def stated_date(answer: str) -> date | None:
    """The one date `answer` says in full (`July 31st, 2024.`, `24 December 2022.`, `Oct 5, 2022`, `20230205`), or None when it says no date,
    several, a month named twice, or one that is no day."""
    found = {(int(y), MONTHS[m.lower()[:3]], int(d)) for d, m, y in DAY_FIRST.findall(answer)}
    found |= {(int(y), MONTHS[m.lower()[:3]], int(d)) for m, d, y in MONTH_FIRST.findall(answer)}
    if (c := COMPACT.fullmatch(answer)) is not None:
        found.add((int(c[1]), int(c[2]), int(c[3])))
    if len(found) != 1 or len(MONTH_WORD.findall(answer)) > 1:
        return None
    try:
        return date(*next(iter(found)))
    except ValueError:
        return None


# -- the file a phone would have left -------------------------------------------------------------------------------------------------------

DROPPED = (0xE1, 0xED, 0xFE)  # the segments taken out: EXIF and XMP, IPTC, comments


def exif(wall: datetime, orientation: int | None) -> bytes:
    """An APP1 EXIF segment (little-endian TIFF) holding `wall` as DateTime, DateTimeOriginal and DateTimeDigitized, and `orientation` when
    it is not None: nothing else."""
    stamp = wall.strftime("%Y:%m:%d %H:%M:%S").encode("ascii") + b"\0"
    first = [(0x0112, 3, struct.pack("<HH", orientation, 0), 1)] if orientation is not None else []
    first.append((0x0132, 2, stamp, len(stamp)))
    second = [(0x9003, 2, stamp, len(stamp)), (0x9004, 2, stamp, len(stamp))]
    first_at = 8
    first_end = first_at + 2 + 12 * (len(first) + 1) + 4  # the Exif IFD's pointer is one more entry
    first_data = first_end
    second_at = first_data + sum(len(v) for _, _, v, _ in first if len(v) > 4)
    second_data = second_at + 2 + 12 * len(second) + 4
    out = bytearray(b"II*\0" + struct.pack("<I", first_at) + struct.pack("<H", len(first) + 1))
    at, tail = first_data, b""
    for tag, kind, value, count in first:
        if len(value) <= 4:
            out += struct.pack("<HHI", tag, kind, count) + value.ljust(4, b"\0")
        else:
            out += struct.pack("<HHII", tag, kind, count, at)
            at += len(value)
            tail += value
    out += struct.pack("<HHII", 0x8769, 4, 1, second_at) + struct.pack("<I", 0) + tail
    out += struct.pack("<H", len(second))
    at = second_data
    for tag, kind, value, count in second:
        out += struct.pack("<HHII", tag, kind, count, at)
        at += len(value)
    out += struct.pack("<I", 0) + b"".join(v for _, _, v, _ in second)
    body = b"Exif\0\0" + bytes(out)
    return b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body


def orientation_of(jpeg: bytes) -> int | None:
    with Image.open(io.BytesIO(jpeg)) as img:
        return img.getexif().get(ORIENTATION)


def stamped(jpeg: bytes, wall: datetime) -> bytes:
    """`jpeg` as a phone would leave it: its EXIF, XMP, IPTC and comment segments (and the GPS in them) taken out, and one EXIF segment put in
    with `wall` as its time (and its orientation, when it had one); the image data is not touched."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    at, kept = 2, []
    while at + 4 <= len(jpeg) and jpeg[at] == 0xFF and jpeg[at + 1] != 0xDA:
        end = at + 2 + int.from_bytes(jpeg[at + 2:at + 4], "big")
        if jpeg[at + 1] not in DROPPED:
            kept.append(jpeg[at:end])
        at = end
    head = kept[:1] if kept and kept[0][1] == 0xE0 else []  # a JFIF segment stays first
    return b"".join([jpeg[:2], *head, exif(wall, orientation_of(jpeg)), *kept[len(head):], jpeg[at:]])


def app_segments(path: Path) -> bytes:
    """The metadata of a JPEG: its application segments (JFIF, EXIF, ...) and comments, the markers' bytes and no more. Not its quantisation
    and Huffman tables, whose binary runs hold the odd four letters that are no one's name."""
    with open(path, "rb") as f:
        data = f.read(1 << 18)
    at, out = 2, bytearray()
    while at + 4 <= len(data) and data[at] == 0xFF and data[at + 1] != 0xDA:
        end = at + 2 + int.from_bytes(data[at + 2:at + 4], "big")
        if 0xE0 <= data[at + 1] <= 0xEF or data[at + 1] == 0xFE:
            out += data[at + 4:end]
        at = end
    return bytes(out)
