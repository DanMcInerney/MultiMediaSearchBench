"""The source of M4: the evaluation split of FSD50K on disk. `clips/eval/<freesound id>.wav` are 10,231 clips (44.1 kHz mono 16-bit, 0.3 to
30 s), `labels/eval.csv` gives each clip's labels (AudioSet classes: a leaf with its ancestors; the eval labels are exhaustive and checked by
people), and `metadata/pp_pnp_ratings_FSD50K.json` the raters' judgement of each label: present and predominant (PP, 1) or present but not
predominant (PNP, 0.5). `metadata/eval_clips_info_FSD50K.json` (the uploader's title, tags and name) is never read: it names the sound.

Every clip of the source is a `RIFF` with the chunks `fmt `, `LIST` (the encoder's name) and `data`: 78 bytes before the samples. A clip goes
into the dump with the `fmt ` and `data` chunks alone, so nothing of the source's holds a name; its length is known from its size."""

from __future__ import annotations

import csv
import functools
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from mmmb.scenarios.realdump import sources_root

SOURCE = sources_root() / "fsd50k"
CLIPS = "clips/eval"
LABELS = "labels/eval.csv"
VOCABULARY = "labels/vocabulary.csv"  # `index,label,mid`, with no header
RATINGS = "metadata/pp_pnp_ratings_FSD50K.json"
PP, PNP = 1.0, 0.5
HEADER = 78  # bytes before the samples in each clip of the source
FORMAT = struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)  # PCM, mono, 44.1 kHz, 16 bits: the `fmt ` chunk of each
BYTE_RATE = 88200


@dataclass(frozen=True)
class Clip:
    id: str  # its Freesound id, the name of its file in the source
    labels: tuple[str, ...]  # the classes it holds
    strong: frozenset[str]  # of those, the ones two raters found present and predominant and none found only present
    seconds: float  # how long it plays


@functools.lru_cache(maxsize=2)
def read_labels(source: str) -> tuple[tuple[Clip, ...], tuple[str, ...]]:
    """The clips of `source`'s eval split, by Freesound id, and the 200 class names of its vocabulary. A labelled clip that is not on disk,
    or not a clip of the shape above, is refused."""
    root = Path(source)
    sizes = {e.name: e.stat().st_size for e in os.scandir(root / CLIPS)}
    ratings = json.loads((root / RATINGS).read_text(encoding="utf-8"))
    with open(root / LABELS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(root / VOCABULARY, newline="", encoding="utf-8") as f:
        names = tuple(sorted(row[1] for row in csv.reader(f)))
    clips = []
    for row in rows:
        labels, mids = row["labels"].split(","), row["mids"].split(",")
        size = sizes.get(row["fname"] + ".wav")
        if size is None or size <= HEADER or (size - HEADER) % 2:
            raise ValueError(f"{row['fname']}.wav is not in {root / CLIPS}, or is not a clip of 16-bit samples after {HEADER} bytes")
        rated = ratings.get(row["fname"], {})
        strong = frozenset(label for label, mid in zip(labels, mids, strict=True)
                           if (r := rated.get(mid, [])).count(PP) >= 2 and PNP not in r)
        clips.append(Clip(row["fname"], tuple(labels), strong, (size - HEADER) / BYTE_RATE))
    return tuple(sorted(clips, key=lambda c: int(c.id))), names


def chunks(raw: bytes) -> dict[bytes, bytes]:
    """The chunks of a RIFF/WAVE file by their ids (the first of each id)."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    out: dict[bytes, bytes] = {}
    at = 12
    while at + 8 <= len(raw):
        size = int.from_bytes(raw[at + 4:at + 8], "little")
        out.setdefault(raw[at:at + 4], raw[at + 8:at + 8 + size])
        at += 8 + size + (size & 1)
    return out


def with_samples_only(raw: bytes) -> bytes:
    """The WAV `raw` as a WAV of its `fmt ` and `data` chunks alone; refused when it is not 16-bit mono PCM at 44.1 kHz."""
    found = chunks(raw)
    if found.get(b"fmt ") != FORMAT or len(found.get(b"data", b"")) % 2:
        raise ValueError("not 16-bit mono PCM at 44.1 kHz")
    data = found[b"data"]
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(FORMAT)) + FORMAT + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body
