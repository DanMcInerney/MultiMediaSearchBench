"""The source of I9: one iNaturalist observer's photos on disk (`photos/<photo id>.<ext>`, the first photo of each observation, `large` size) and
their observations (`observations.jsonl`: for each, its taxon and common name, the taxon's rank, the day and local time it was observed, and its
photo), as the open data lists them, research grade only (the community agrees on the species) and with no place or user name in them.

A photo goes into the dump as a phone or a camera would have left it: numbered in the order taken, with EXIF holding the local time it was taken
and nothing else (no position, no device), which is also its modified time. An observation of a subspecies, variety or form is of its species;
an observation to the genus or only a hybrid, section or complex is not in the pile (the species is not told), nor is one whose photo is not a
JPEG."""

from __future__ import annotations

import functools
import io
import json
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "inat-observer"
OBSERVATIONS = "observations.jsonl"
PHOTOS = "photos"
SPECIES_RANKS = ("species", "subspecies", "variety", "form")
JPEG_EXTS = ("jpg", "jpeg")
ORIENTATION = 0x0112
DROPPED = (*range(0xE1, 0xEE), 0xFE)  # the segments taken out of a JPEG: EXIF and XMP, colour profiles (an iPhone's is Apple's, and its binary
# tables hold the odd run of letters the probe takes for a name), IPTC and the other application segments, comments; JFIF and Adobe's stay


@dataclass(frozen=True)
class Obs:
    id: int
    species: str  # its binomial (`Juglans cinerea`), the species of a subspecies
    wall: datetime  # when it was observed: the local time, no zone
    photo: str  # its photo's name in the source's photos folder

    @property
    def day(self) -> str:
        return self.wall.date().isoformat()


@functools.lru_cache(maxsize=2)
def read_rows(source: str) -> tuple[dict, ...]:
    with open(Path(source) / OBSERVATIONS, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def binomial(taxon_name: str) -> str:
    """`Juglans cinerea subsp. x` as `Juglans cinerea`."""
    return " ".join(taxon_name.split()[:2])


@functools.lru_cache(maxsize=2)
def read_pile(source: str) -> tuple[Obs, ...]:
    """The observations of the pile, in the order observed (then by id): those of a species or below whose photo is a JPEG. Refused when a
    day is not the date of its local time."""
    out = []
    for r in read_rows(source):
        if r["taxon_rank"] not in SPECIES_RANKS or r["photo_ext"] not in JPEG_EXTS:
            continue
        wall = datetime.fromisoformat(r["time_observed_at"][:19])
        if wall.date().isoformat() != r["observed_on"]:
            raise ValueError(f"observation {r['id']} was observed on {r['observed_on']} at {r['time_observed_at']}")
        out.append(Obs(r["id"], binomial(r["taxon_name"]), wall, f"{r['photo_id']}.{r['photo_ext']}"))
    return tuple(sorted(out, key=lambda o: (o.wall, o.id)))


@functools.lru_cache(maxsize=2)
def read_names(source: str) -> dict[str, str]:
    """The common name of each species the observations name at species rank. Refused when two species have one common name."""
    names = {binomial(r["taxon_name"]): r["taxon_common_name"] for r in read_rows(source)
             if r["taxon_rank"] == "species" and r["taxon_common_name"]}
    if len({n.casefold() for n in names.values()}) != len(names):
        raise ValueError("two species have one common name")
    return names


@functools.lru_cache(maxsize=2)
def read_genera(source: str) -> frozenset[str]:
    """Every genus the observations name, to whatever rank (an observation to the genus alone is of some species of it)."""
    return frozenset(r["taxon_name"].split()[0] for r in read_rows(source))


def check_photos(source: str, obs: Sequence[Obs]) -> None:
    """Refuses `obs` when any photo is not in `source`'s photos folder."""
    have = set(os.listdir(Path(source) / PHOTOS))
    if lacking := [o.photo for o in obs if o.photo not in have]:
        raise FileNotFoundError(f"{len(lacking)} of the {len(obs)} photos this pile needs are not in {Path(source) / PHOTOS} (the first is "
                                f"{lacking[0]}): `mmsb fetch --scenario i9_nature` gets them")


def exif(wall: datetime, orientation: int | None) -> bytes:
    """An APP1 EXIF segment (little-endian TIFF) holding `wall` as DateTime, DateTimeOriginal and DateTimeDigitized, and `orientation` when
    it is not None: nothing else."""
    stamp = wall.strftime("%Y:%m:%d %H:%M:%S").encode("ascii") + b"\0"
    first = [(ORIENTATION, 3, struct.pack("<HH", orientation, 0), 1)] if orientation is not None else []
    first.append((0x0132, 2, stamp, len(stamp)))
    second = [(0x9003, 2, stamp, len(stamp)), (0x9004, 2, stamp, len(stamp))]
    first_at = 8
    first_data = first_at + 2 + 12 * (len(first) + 1) + 4  # the Exif IFD's pointer is one more entry
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


def stamped(jpeg: bytes, wall: datetime) -> bytes:
    """`jpeg` as a phone would leave it: its EXIF, XMP, colour profile, IPTC and comment segments (and any position in them) taken out, and
    one EXIF segment put in with `wall` as its time (and its orientation, when it had one); the image data is not touched."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    with Image.open(io.BytesIO(jpeg)) as img:
        orientation = img.getexif().get(ORIENTATION)
    at, kept = 2, []
    while at + 4 <= len(jpeg) and jpeg[at] == 0xFF and jpeg[at + 1] != 0xDA:
        end = at + 2 + int.from_bytes(jpeg[at + 2:at + 4], "big")
        if jpeg[at + 1] not in DROPPED:
            kept.append(jpeg[at:end])
        at = end
    head = kept[:1] if kept and kept[0][1] == 0xE0 else []  # a JFIF segment stays first
    return b"".join([jpeg[:2], *head, exif(wall, orientation), *kept[len(head):], jpeg[at:]])


def app_segments(path: Path) -> bytes:
    """The metadata of a JPEG: its application segments (JFIF, EXIF, ...) and comments, the markers' bytes and no more. Not its quantisation
    and Huffman tables, whose binary runs hold the odd four letters (`CRab`) that are no one's name."""
    with open(path, "rb") as f:
        data = f.read(1 << 18)
    at, out = 2, bytearray()
    while at + 4 <= len(data) and data[at] == 0xFF and data[at + 1] != 0xDA:
        end = at + 2 + int.from_bytes(data[at + 2:at + 4], "big")
        if 0xE0 <= data[at + 1] <= 0xEF or data[at + 1] == 0xFE:
            out += data[at + 4:end]
        at = end
    return bytes(out)
