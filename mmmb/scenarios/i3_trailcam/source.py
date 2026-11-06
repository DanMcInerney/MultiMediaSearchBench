"""The source of I3: the Caltech Camera Traps photos of location 23 on disk (`images/<uuid>.jpg`, as LILA serves them: no EXIF) and the
dataset's labels (`labels/caltech_camera_traps.json.zip`): for each photo, its location, when the camera took it (`date_captured`, a
wall-clock time with no zone, to the second) and one label, a species or `empty`. The camera writes the hour after midnight as 12 (there
is no hour 00), which is kept as it is; a question never rests on a photo whose hour is 12.

A photo goes into the dump as the camera would have left it on its card: `DCIM/100RECNX/IMG_0001.JPG` onwards in the order taken, with
an EXIF block (the camera's make and model, and the time it was taken) put before its image data, which is not re-encoded."""

from __future__ import annotations

import functools
import json
import os
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SOURCE = Path(r"D:\bench-data\sources\caltech-camera-traps")
LABELS = "labels/caltech_camera_traps.json.zip"
MEMBER = "caltech_images_20210113.json"
IMAGES = "images"
LOCATION = "23"
EMPTY = "empty"
MIDNIGHT_HOUR = 12  # the hour this camera writes for the hour after midnight (and for noon)
MAKE, MODEL = b"RECONYX\0", b"HC500 HYPERFIRE\0"


@dataclass(frozen=True)
class Photo:
    file: str  # its name in the source's images folder
    wall: datetime  # when the camera took it, as it wrote it (no zone)
    species: str  # its label: a species, or `empty`


@functools.lru_cache(maxsize=2)
def read_labels(source: str) -> tuple[tuple[Photo, ...], tuple[str, ...]]:
    """The photos of location 23 that are in `source`'s images folder, in the order taken (then by name), and every label name of
    the dataset. A photo with other than one label is refused."""
    with zipfile.ZipFile(Path(source) / LABELS) as z:
        data = json.loads(z.read(MEMBER))
    names = {c["id"]: c["name"] for c in data["categories"]}
    labels: dict[str, list[str]] = {}
    for a in data["annotations"]:
        labels.setdefault(a["image_id"], []).append(names[a["category_id"]])
    have = set(os.listdir(Path(source) / IMAGES))
    photos = []
    for image in data["images"]:
        if str(image["location"]) == LOCATION and image["file_name"] in have:
            found = labels.get(image["id"], [])
            if len(found) != 1:
                raise ValueError(f"{image['file_name']} has {len(found)} labels: a photo of I3 has one")
            photos.append(Photo(image["file_name"], datetime.fromisoformat(image["date_captured"]), found[0]))
    if not photos:
        raise FileNotFoundError(f"no photo of location {LOCATION} is in {Path(source) / IMAGES}")
    return tuple(sorted(photos, key=lambda p: (p.wall, p.file))), tuple(sorted(set(names.values())))


def exif(wall: datetime) -> bytes:
    """An APP1 EXIF segment (little-endian TIFF) with the camera's make and model, and `wall` as DateTime, DateTimeOriginal and
    DateTimeDigitized."""
    stamp = wall.strftime("%Y:%m:%d %H:%M:%S").encode("ascii") + b"\0"
    first = [(0x010F, MAKE), (0x0110, MODEL), (0x0132, stamp)]  # IFD0, then its pointer to the Exif IFD
    second = [(0x9003, stamp), (0x9004, stamp)]
    first_at = 8
    first_data = first_at + 2 + 12 * (len(first) + 1) + 4
    second_at = first_data + sum(len(v) for _, v in first)
    second_data = second_at + 2 + 12 * len(second) + 4
    out = bytearray(b"II*\0" + struct.pack("<I", first_at) + struct.pack("<H", len(first) + 1))
    at = first_data
    for tag, value in first:
        out += struct.pack("<HHII", tag, 2, len(value), at)
        at += len(value)
    out += struct.pack("<HHII", 0x8769, 4, 1, second_at) + struct.pack("<I", 0) + b"".join(v for _, v in first)
    out += struct.pack("<H", len(second))
    at = second_data
    for tag, value in second:
        out += struct.pack("<HHII", tag, 2, len(value), at)
        at += len(value)
    out += struct.pack("<I", 0) + b"".join(v for _, v in second)
    body = b"Exif\0\0" + bytes(out)
    return b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body


def with_exif(jpeg: bytes, wall: datetime) -> bytes:
    """`jpeg` with `exif(wall)` put right after its start marker; refused when it is no JPEG or has an EXIF block already."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    if b"Exif\0\0" in jpeg[:1 << 16]:
        raise ValueError("the JPEG has an EXIF block already")
    return jpeg[:2] + exif(wall) + jpeg[2:]
