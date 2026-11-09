"""The source of M3: the Robocall Audio Dataset on disk (`robocall-ftc`): 1,427 recordings of automated phone calls that the FTC linked to
in its letters to the carriers that routed them (`audio-wav-16khz/*.wav`, 16 kHz or 8 kHz mono 16-bit, a few seconds to some minutes), and
`metadata.csv`: for each recording the transcript (made by Whisper's medium model, so by a machine: the dataset says so), the language
Whisper detected, and the case (`case_details`: the cease-and-desist or warning letter, named for the carrier the FTC wrote to).

Three things about the labels are told apart here. A row may be there twice (26 file names are: 17 with the same text, which is one row,
and 9 with different texts, which is no label to trust, so those recordings are left out). The recordings of a stereo call that were not
transcribed (`*_right.wav`, the side of the honeypot) have no row and are left out. A recording is put in a dump as a WAV of its `fmt ` and
`data` chunks alone (the source's has a `LIST` chunk with the converter's name), under the name of a voicemail box: nothing of the source's
file name is kept."""

from __future__ import annotations

import csv
import functools
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "robocall-ftc"
METADATA = "metadata.csv"
AUDIO = "audio-wav-16khz"
FORMATS = {16000: struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16), 8000: struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)}  # PCM mono, `fmt `


@dataclass(frozen=True)
class Call:
    file: str  # in the source: `audio-wav-16khz/1112259_normalized.wav`
    language: str  # as Whisper detected it: `en` or `zh`
    transcript: str  # Whisper's, with its white space made single spaces
    case: str  # the letter it came with: `cease-desist-letter-every1-telecom`
    seconds: float  # how long it plays


def chunks(raw: bytes) -> dict[bytes, tuple[int, int]]:
    """The chunks of a RIFF/WAVE file by id: `(start, size)` of the data of the first of each id."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    out: dict[bytes, tuple[int, int]] = {}
    at = 12
    while at + 8 <= len(raw):
        size = int.from_bytes(raw[at + 4:at + 8], "little")
        out.setdefault(raw[at:at + 4], (at + 8, size))
        at += 8 + size + (size & 1)
    return out


def with_samples_only(raw: bytes) -> bytes:
    """The WAV `raw` as a WAV of its `fmt ` and `data` chunks alone; refused when it is not 16-bit mono PCM at 16 or 8 kHz."""
    found = chunks(raw)
    fmt = raw[found[b"fmt "][0]:found[b"fmt "][0] + found[b"fmt "][1]][:16] if b"fmt " in found else b""
    if fmt not in FORMATS.values() or b"data" not in found:
        raise ValueError("not 16-bit mono PCM at 16 or 8 kHz")
    start, size = found[b"data"]
    data = raw[start:start + size]
    if len(data) != size or size % 2:
        raise ValueError("the data chunk is cut short or is not whole 16-bit samples")
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def seconds_of(path: Path) -> float:
    """How long the WAV at `path` plays, from its header and size (the first 4 KB hold the `fmt ` and `LIST` chunks and the start of `data`)."""
    with open(path, "rb") as f:
        head = f.read(4096)
    found = chunks(head)
    if b"fmt " not in found or b"data" not in found or head[found[b"fmt "][0]:found[b"fmt "][0] + 16] not in FORMATS.values():
        raise ValueError(f"{path} is not 16-bit mono PCM at 16 or 8 kHz")
    start, size = found[b"data"]
    rate = int.from_bytes(head[found[b"fmt "][0] + 4:found[b"fmt "][0] + 8], "little")
    return min(size, os.stat(path).st_size - start) / (2 * rate)


@functools.lru_cache(maxsize=2)
def read_calls(source: str) -> tuple[tuple[Call, ...], tuple[str, ...]]:
    """The recordings of `source` that have one row of labels (or several with one text), in the order of the file, and every case name of
    the file (its letters), for the words that no label holds. A row whose recording is not on disk is refused."""
    root = Path(source)
    with open(root / METADATA, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row["file_name"], []).append(row)
    calls = []
    for name, same in by_file.items():
        if len({(r["transcript"], r["language"]) for r in same}) != 1:
            continue  # two transcripts of one recording: no label to trust
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is labelled in {METADATA} and is not on disk")
        row = same[0]
        calls.append(Call(name, row["language"], " ".join(row["transcript"].split()), row["case_details"], seconds_of(path)))
    return tuple(calls), tuple(sorted({r["case_details"] for r in rows}))
