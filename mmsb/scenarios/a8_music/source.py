"""The source of A8: the Free Music Archive on disk (`fma`): `fma_metadata.zip`, whose `tracks.csv` (106,574 rows, a header of three lines) gives
each track its album, its uploader-assigned genre and the time it was uploaded, and `genres.csv` the tree of the genres; and `fma_small.zip`,
8,000 thirty-second MP3 excerpts (`fma_small/000/000002.mp3`, mostly 44.1 kHz stereo at 256 kb/s, some 128 or 320 kb/s, 48 kHz or mono) of the tracks of `tracks.csv`'s `small` subset, 1,000 of
each of eight genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop and Rock. The zips are read member by member and
never unpacked.

The genre of a track is the one its uploader gave it (`genre_top`, the top of the tree of the genres), a coarse label that a listener may not
share: no more than that is claimed for it. In `small` every track has exactly one top genre (every genre it is tagged with is under it), which is
checked here.

A clip goes into a dump as its MP3 frames alone: the ID3v2 tag at its start (the title, the artist, the album, the genre and a link) and any ID3v1
tag at its end are taken off, which is all a tag is made of in the source. The clip's first frame is ffmpeg's `Info` frame, which holds the length
and the encoder's name and no more."""

from __future__ import annotations

import ast
import csv
import functools
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "fma"
METADATA = "fma_metadata.zip"
TRACKS = "fma_metadata/tracks.csv"
GENRES = "fma_metadata/genres.csv"
CLIPS = "fma_small.zip"
SUBSET = "small"
CLIP_S = 29.98  # what a plan says every clip plays: the least any does (29.9886 s) to the hundredth; the build measures each


@dataclass(frozen=True)
class Track:
    id: int
    genre: str  # the uploader's: `Folk`
    album: int
    number: int  # its place in its album
    created: datetime  # when it was uploaded (wall clock, no zone)


def member(track: int) -> str:
    """Where the clip of track `track` is in `fma_small.zip`."""
    return f"fma_small/{track // 1000:03d}/{track:06d}.mp3"


@functools.lru_cache(maxsize=2)
def read_tracks(source: str) -> tuple[Track, ...]:
    """The 8,000 tracks of `fma_small` in the order of `tracks.csv`, with the size of each clip in the zip. A track whose genres are not all under
    its top genre, or that has no clip, or whose subset is not `small` where its clip is, is refused."""
    root = Path(source)
    with zipfile.ZipFile(root / METADATA) as z:
        with z.open(GENRES) as f:
            genres = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")))
        top = {int(g["genre_id"]): g["title"] for g in genres if g["top_level"] == g["genre_id"]}
        family = {int(g["genre_id"]): top[int(g["top_level"])] for g in genres}
        with z.open(TRACKS) as f:
            rows = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
            group, name, _ = next(rows), next(rows), next(rows)
            at = {pair: i for i, pair in enumerate(zip(group, name))}
            keys = {k: at[k] for k in (("set", "subset"), ("track", "genre_top"), ("track", "genres_all"), ("album", "id"), ("track", "number"),
                                       ("track", "date_created"))}
            small = [row for row in rows if row[keys["set", "subset"]] == SUBSET]
    out = []
    with zipfile.ZipFile(root / CLIPS) as z:
        infos = {i.filename: i for i in z.infolist()}
        for row in small:
            track = int(row[0])
            genre = row[keys["track", "genre_top"]]
            if {family[g] for g in ast.literal_eval(row[keys["track", "genres_all"]])} != {genre}:
                raise ValueError(f"track {track} is tagged with genres that are not all under its top genre {genre}")
            if member(track) not in infos:
                raise ValueError(f"track {track} of the {SUBSET} subset has no clip {member(track)} in {CLIPS}")
            out.append(Track(track, genre, int(row[keys["album", "id"]]), int(row[keys["track", "number"]] or 0),
                             datetime.strptime(row[keys["track", "date_created"]], "%Y-%m-%d %H:%M:%S")))
    return tuple(out)


def leading_tag(first: bytes) -> int:
    """The bytes of the ID3v2 tag at the start of an MP3 whose first ten bytes are `first` (0 when it has none)."""
    if first[:3] != b"ID3":
        return 0
    return 10 + ((first[6] << 21) | (first[7] << 14) | (first[8] << 7) | first[9]) + (10 if first[5] & 0x10 else 0)


def tag_bytes(raw: bytes) -> tuple[int, int]:
    """`(bytes before the first frame, bytes after the last)` of an MP3 `raw`: an ID3v2 tag at its start and an ID3v1 tag at its end. Refused
    when what follows the tag is no frame, or the file ends in a tag of another kind."""
    head = leading_tag(raw[:10])
    if raw[head] != 0xFF or raw[head + 1] & 0xE0 != 0xE0:
        raise ValueError(f"the MP3 has no frame after its first {head} bytes")
    if re.search(rb"APETAGEX|LYRICSBEGIN", raw[-300:]):
        raise ValueError("the MP3 ends in an APE or Lyrics3 tag")
    return head, 128 if raw[-128:-125] == b"TAG" else 0


def frames_only(raw: bytes) -> bytes:
    """The MP3 `raw` without its ID3 tags."""
    head, tail = tag_bytes(raw)
    return raw[head:len(raw) - tail]


def mp3_header(path: Path) -> bytes:
    """The start and the end of an MP3 where a tag or text would be: its first 256 bytes (the `Info` frame's header, with the encoder's name)
    and its last 128 (where an ID3v1 tag is). The frames between are audio, which has runs of printable bytes by chance."""
    with open(path, "rb") as f:
        head = f.read(256)
        f.seek(-128, 2)
        return head + f.read()
