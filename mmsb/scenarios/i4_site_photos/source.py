"""The source of I4: BD3's test photos on disk (`test_images/<n>_<n>.jpg`: 793 close-ups of building surfaces, 512 by 512 pixels, PNGs under a
`.jpg` name) and their labels (`test_images_labels.jsonl`): for each photo, one class, `algae`, `major_crack`, `minor_crack`, `peeling`,
`plain`, `spalling` or `stain`. The class is the photo's own (one per photo): a photo of one class may show a little of another, and some
labels are loose (a photo of a hairline crack under `stain`, a patch of paint under `spalling`); questions.py says how a question lives with that.

A photo goes into the dump as an inspector's phone or camera would have left it: a JPEG with EXIF holding the time it was taken and nothing
else, under a number of the order taken. The dataset has no recording times; questions.py draws them."""

from __future__ import annotations

import functools
import io
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "bd3-building-defects"
LABELS = "test_images_labels.jsonl"
CLASSES = ("algae", "major_crack", "minor_crack", "peeling", "plain", "spalling", "stain")
COUNT = 793  # the photos of the source
QUALITY = 92  # the JPEG each photo is written as


@dataclass(frozen=True)
class Shot:
    file: str  # its path in the source (`test_images/00000_00000.jpg`)
    label: str  # its class


@functools.lru_cache(maxsize=2)
def read_labels(source: str) -> tuple[Shot, ...]:
    """The photos of the source with their classes, in the order of their names. Refused when there are not COUNT of them, a class is
    not one of CLASSES, or a photo is not there."""
    shots = []
    with open(Path(source) / LABELS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["answer"] not in CLASSES:
                raise ValueError(f"{row['file']} is of the class {row['answer']!r}, which is none of {', '.join(CLASSES)}")
            shots.append(Shot(row["file"], row["answer"]))
    if len(shots) != COUNT or len({s.file for s in shots}) != COUNT:
        raise ValueError(f"{Path(source) / LABELS} labels {len(shots)} photos, not the {COUNT} of the BD3 test set")
    return tuple(sorted(shots, key=lambda s: s.file))


def check_photos(source: str, shots: Sequence[Shot]) -> None:
    """Refuses `shots` when any photo is not in `source`."""
    have = set(os.listdir(Path(source) / "test_images"))
    if lacking := [s.file for s in shots if s.file.rsplit("/", 1)[-1] not in have]:
        raise FileNotFoundError(f"{len(lacking)} of the {len(shots)} photos this pile needs are not in {Path(source) / 'test_images'} (the first "
                                f"is {lacking[0]}): the BD3 download holds them")


def jpeg_of(png: bytes, wall: datetime) -> bytes:
    """`png` (any image the source holds) as a JPEG of QUALITY, with `wall` as the time of its EXIF (DateTime, DateTimeOriginal and
    DateTimeDigitized) and nothing else."""
    stamp = wall.strftime("%Y:%m:%d %H:%M:%S")
    exif = Image.Exif()
    exif[0x0132] = stamp
    inner = exif.get_ifd(0x8769)
    inner[0x9003] = stamp
    inner[0x9004] = stamp
    out = io.BytesIO()
    with Image.open(io.BytesIO(png)) as img:
        img.convert("RGB").save(out, "JPEG", quality=QUALITY, exif=exif)
    return out.getvalue()
