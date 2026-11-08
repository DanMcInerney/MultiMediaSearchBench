"""The source of I3: the Caltech Camera Traps photos of location 23 on disk (`images/<uuid>.jpg`, as LILA serves them: no EXIF) and the
dataset's labels (`labels/caltech_camera_traps.json.zip`): for each photo, its location, when the camera took it (`date_captured`, a
wall-clock time with no zone, to the second) and one label, a species or `empty`. The camera's clock has hours 1 to 12 and AM or PM, and
the dataset dropped the AM and PM of the hour 12: the hour after midnight is written 12 as noon is (the photos' burned-in strip says
"2013-11-12 12:34:40 AM" of one the labels give as 12:34:40). `told` reads that strip's AM or PM and moves the photos of the hour after
midnight to 00:MM:SS; a question still never rests on a photo whose hour the dataset wrote as 12.

A photo goes into the dump as the camera would have left it on its card: `DCIM/100RECNX/IMG_0001.JPG` onwards in the order taken, with
an EXIF block (the camera's make and model, and the time it was taken) put before its image data, which is not re-encoded."""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageStat

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "caltech-camera-traps"
LABELS = "labels/caltech_camera_traps.json.zip"
MEMBER = "caltech_images_20210113.json"
IMAGES = "images"
LOCATION = "23"
CARD_DAYS = ("2013-09-10", "2014-02-28")  # the card: all of the location's photos taken from the first of these days to the last (4,256)
EMPTY = "empty"
TWELVE = 12  # the hour the dataset writes for the hour after midnight (and for noon)
FOOT = (686, 16, 700, 28)  # in a photo (2,048 px wide), the box of its strip holding the right foot of the A of an hour 12's "AM" (not a P's)
FOOT_AM, FOOT_PM = 133.0, 0.0  # the box's mean brightness (0 to 255) when it holds that foot, and when it does not
MAKE, MODEL = b"RECONYX\0", b"HC500 HYPERFIRE\0"


@dataclass(frozen=True)
class Photo:
    file: str  # its name in the source's images folder
    wall: datetime  # when the camera took it, as it wrote it (no zone)
    species: str  # its label: a species, or `empty`
    twelve: bool = False  # the dataset wrote its hour as 12: noon, or the hour after midnight (`wall` is that hour told, by `told`)


def card_images(data: dict) -> list[dict]:
    """The images of the dataset's labels (`data`, the JSON) that are on the card: location 23's, taken from the first to the last of
    CARD_DAYS."""
    return [i for i in data["images"] if str(i["location"]) == LOCATION and CARD_DAYS[0] <= i["date_captured"][:10] <= CARD_DAYS[1]]


@functools.lru_cache(maxsize=2)
def read_labels(source: str) -> tuple[tuple[Photo, ...], tuple[str, ...]]:
    """The photos of the card (`card_images`), in the order the dataset's clock gives (then by name), and every label name of the
    dataset, whether the photos are on disk or not (`check_photos` asks that of those a scenario uses). A photo with other than one
    label is refused."""
    with zipfile.ZipFile(Path(source) / LABELS) as z:
        data = json.loads(z.read(MEMBER))
    names = {c["id"]: c["name"] for c in data["categories"]}
    labels: dict[str, list[str]] = {}
    for a in data["annotations"]:
        labels.setdefault(a["image_id"], []).append(names[a["category_id"]])
    photos = []
    for image in card_images(data):
        found = labels.get(image["id"], [])
        if len(found) != 1:
            raise ValueError(f"{image['file_name']} has {len(found)} labels: a photo of I3 has one")
        wall = datetime.fromisoformat(image["date_captured"])
        photos.append(Photo(image["file_name"], wall, found[0], wall.hour == TWELVE))
    return tuple(sorted(photos, key=lambda p: (p.wall, p.file))), tuple(sorted(set(names.values())))


def check_photos(source: str, photos: Sequence[Photo]) -> None:
    """Refuses `photos` when any is not in `source`'s images folder: a card of fewer photos than the labels give is another card, and
    its questions would be drawn from what happens to be there."""
    have = set(os.listdir(Path(source) / IMAGES))
    if lacking := [p.file for p in photos if p.file not in have]:
        raise FileNotFoundError(f"{len(lacking)} of the {len(photos)} photos this card needs are not in {Path(source) / IMAGES} (the first is "
                                f"{lacking[0]}): `mmsb fetch --scenario i3_trailcam` gets them")


@functools.cache
def is_am(image: str) -> bool:
    """Whether the strip burned into the top of `image` (a photo of the hour 12, whose strip reads `2013-11-12 12:34:40 AM`) says AM, the
    hour after midnight, and not PM, noon: read from the right foot of the A (a P has none), in a box (`FOOT`) that is bright for an A and
    black for a P. Read once per photo; a box that is neither is refused."""
    with Image.open(image) as img:
        img.draft("L", img.size)  # greyscale: no colour to decode
        foot = ImageStat.Stat(img.crop(FOOT)).mean[0]
    if min(abs(foot - FOOT_AM), abs(foot - FOOT_PM)) > 10:
        raise ValueError(f"{image}: the strip's AM or PM is not read (its mean brightness at {FOOT} is {foot:.1f}, not {FOOT_AM} or {FOOT_PM})")
    return foot > (FOOT_AM + FOOT_PM) / 2


def told(source: str, photos: Sequence[Photo]) -> tuple[Photo, ...]:
    """`photos` with the hour after midnight told from noon: each photo the dataset wrote as hour 12 whose strip says AM is moved to
    00:MM:SS of its date (it stays `twelve`). In the order taken (then by name)."""
    out = [dataclasses.replace(p, wall=p.wall.replace(hour=0)) if p.twelve and is_am(str(Path(source) / IMAGES / p.file)) else p
           for p in photos]
    return tuple(sorted(out, key=lambda p: (p.wall, p.file)))


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
