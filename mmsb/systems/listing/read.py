"""What a file says of itself, read as any tool on the files could (`Listing.ingest`): its size and mtime from the file system, a
picture's EXIF tags by Pillow and its format and size, a video's or a sound's container fields by PyAV (`ffprobe` shows the same). Nothing
is read that the manifest knows and the file does not (its `t`, `kind` and `sha256`), and no pixel or sample is decoded. The one use of the
manifest's `t` is `misdated_reason`: it is compared with the file's mtime, the listing's only time, and never listed or served.

EXIF is listed by the tag names Pillow knows, from the main table, the Exif table and the GPS table. A value is a number, a text of at
most TEXT_MAX characters (longer is cut, and ends with `…`), or a short list of numbers; a binary value is listed only when it is printable
text (a `UserComment` is; a `MakerNote` or a thumbnail is not), so no blob reaches an agent."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import av
from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from mmsb.core.contract import ManifestEntry, Refusal

TEXT_MAX = 256
LIST_MAX = 16
POINTERS = {"ExifOffset", "GPSInfo", "InteropOffset", "MakerNote"}  # tags that point to a table, or hold the maker's private bytes
CHARSETS = (b"ASCII\x00\x00\x00", b"UNICODE\x00", b"JIS\x00\x00\x00\x00\x00", b"\x00\x00\x00\x00\x00\x00\x00\x00")  # a UserComment's first 8 bytes
TIME_TOL_S = 1.0  # a file's mtime is its time when it is within this of the manifest's `t`: see `misdated_reason`
UNREADABLE = (OSError, ValueError, SyntaxError, av.error.FFmpegError)  # what Pillow and PyAV raise for a file that is not what its name says


def text_of(name: str, raw: bytes) -> str | None:
    """A binary EXIF value as the text it is, or None when it is not printable text (the `XP...` tags are UTF-16)."""
    body = next((raw[len(c):] for c in CHARSETS if raw.startswith(c)), raw)
    try:
        text = body.decode("utf-16-le" if name.startswith("XP") else "utf-8").strip("\x00 ")
    except UnicodeDecodeError:
        return None
    return text if text.isprintable() else None


def shown(name: str, value: object) -> object | None:
    """An EXIF value as JSON can say it, or None when it is not listed (see the module)."""
    if isinstance(value, bytes):
        value = text_of(name, value)
    if isinstance(value, str):
        value = value.strip("\x00 ")
        return value if len(value) <= TEXT_MAX else value[:TEXT_MAX] + "…"
    if isinstance(value, IFDRational):
        value = float(value) if value.denominator else math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)) and len(value) <= LIST_MAX:
        items = [shown(name, v) for v in value]
        return items if all(isinstance(i, (int, float)) for i in items) else None
    return None


def exif_of(path: Path) -> dict:
    """The EXIF tags of the picture at `path` by name: the main table, the Exif table and the GPS table."""
    with Image.open(path) as image:
        exif = image.getexif()
        tables = ((exif, ExifTags.TAGS), (exif.get_ifd(ExifTags.IFD.Exif), ExifTags.TAGS), (exif.get_ifd(ExifTags.IFD.GPSInfo), ExifTags.GPSTAGS))
        found = {}
        for table, names in tables:
            for tag, value in table.items():
                name = names.get(tag)
                if name is not None and name not in POINTERS and (listed := shown(name, value)) is not None:
                    found[name] = listed
        return found


def picture_of(path: Path) -> dict:
    with Image.open(path) as image:
        return {"format": image.format, "width": image.width, "height": image.height}


def container_of(path: Path) -> dict:
    """The container fields of the video or sound at `path` that it has: its format, duration, the first video stream's size, frame
    rate and codec (else the first audio stream's codec), the first audio stream's sample rate, and the container's `creation_time`."""
    with av.open(str(path)) as container:
        video, audio = next(iter(container.streams.video), None), next(iter(container.streams.audio), None)
        found: dict = {"format": container.format.name}
        if container.duration is not None:
            found["duration"] = round(container.duration / av.time_base, 3)
        if video is not None:
            found.update(width=video.codec_context.width, height=video.codec_context.height)
            if video.average_rate is not None:
                found["fps"] = round(float(video.average_rate), 3)
        if (stream := video or audio) is not None:
            found["codec"] = stream.codec_context.name
        if audio is not None:
            found["sample_rate"] = audio.codec_context.sample_rate
        if "creation_time" in container.metadata:
            found["creation_time"] = container.metadata["creation_time"]
        return found


def clock_text(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def misdated_reason(entry: ManifestEntry, mtime: float) -> str | None:
    """Why the listing would put `entry` at a time other than its own, else None. The listing's only time is the file's mtime (the order, `since`
    and `until`, and the horizon `--until` all cut by it), so it must be the time the file holds: the manifest's `t`. V7's generator sets it so
    (`os.utime(path, (t, t))`), and its recorded pilot's 1,572 files are all exactly at `t`, so the tolerance is not the generator's but
    the file systems': one second, as mm's own check of the time it stored a file at (systems/mm/load.py), which covers a volume that keeps whole
    seconds. A download's mtimes are the download's (V8's are days after the recordings), a copy may keep the server's dates: the listing would cut a
    question's horizon wrongly, at nothing or at everything. A log (media `records`) is the exception, because its `t` is its first line's time and
    its mtime its last (V7's pilot: 237,409 to 237,435 s later): it must only not be earlier than `t`."""
    if entry.media == "records":
        ok = mtime >= entry.t - TIME_TOL_S
        wanted = f"not before its first line, at {clock_text(entry.t)}"
    else:
        ok = abs(mtime - entry.t) <= TIME_TOL_S
        wanted = f"at {clock_text(entry.t)}"
    if ok:
        return None
    return (f"modified {clock_text(mtime)}, but the dump says it is from {clock_text(entry.t)} and its modification time should be {wanted} "
            f"(within {TIME_TOL_S:g} s): the listing's only time is the modification time, so a horizon would cut it at a wrong time")


READERS = {"image": (("exif", exif_of), ("container", picture_of)), "video": (("container", container_of),),
           "audio": (("container", container_of),)}  # by the entry's media: what the listing reads of it, and under which key


def row_of(root: Path, entry: ManifestEntry) -> tuple[dict, list[Refusal]]:
    """The listing's row for `entry`, read from its file under `root`, and a refusal for each part of it the file would not give (a
    file that is not the picture, video or sound its entry says). The file not being there is an error."""
    info = (root / entry.path).stat()
    row: dict = {"path": entry.path, "source": entry.source, "size": info.st_size, "mtime": info.st_mtime}
    refused = []
    for key, read in READERS.get(entry.media, ()):
        try:
            if found := read(root / entry.path):
                row[key] = found
        except UNREADABLE as error:
            refused.append(Refusal(entry.path, f"{key} not read: {type(error).__name__}: {error}"))
    return row, refused
