"""What the speech scenarios share besides their questions (speech.py): a source of talks, each an MP4 with the SRT beside it; the files written
into the dump from them (the video as it is, or its sound alone) with nothing of the source's in their metadata; what the leak probe reads of
those files; a term-screening step that keeps a question from being less than certain or from being given away by the file list; and the
days of a term's classes. A scenario of talk (V9, A6, A7) is a reader of its source, a schedule, its terms and its curated questions; it
imports this, speech.py and realdump, and no other scenario."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import av
import imageio_ffmpeg

from mmsb.core.question import Grade
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech
from mmsb.scenarios.realdump.speech import Cue, Curated, Talk, Term

# a caption's voices that are no speaker of the talk
OTHERS = frozenset({"Audience", "Student", "Speaker", "Child", "Little Girl", "Crew", "Producer", "Audio Bumper", "Amazon Alexa"})
NEAR_START_S = 60.0  # a question of a term first said this soon after its file begins is not asked (see `screened`)
PLAY_TOL_S = 0.5  # a written file plays this close to the length it was planned with (an encoder pads a frame, and a container rounds)
AUDIO = {"m4a": ("-c:a", "copy"),  # a sound file's suffix -> how its sound is written: the source's AAC as it is, or MP3 at 64 kbit/s, mono
         "mp3": ("-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "64k", "-write_xing", "0", "-id3v2_version", "0")}


@dataclass(frozen=True)
class Recording:
    stem: str  # the source's name for it, without its suffix
    video: Path
    cues: tuple[Cue, ...]


def read_recordings(root: Path, wanted: Callable[[str], bool]) -> tuple[Recording, ...]:
    """The recordings of `root` (a flat folder of `<stem>.mp4` and `<stem>.srt`) whose stem `wanted` takes, by stem. An MP4 with no SRT, or an
    SRT with no MP4, is refused: a question is drawn from a transcript, and a transcript of what is not in the dump is not."""
    names = os.listdir(root)
    videos = {n.removesuffix(".mp4") for n in names if n.endswith(".mp4") and wanted(n.removesuffix(".mp4"))}
    captions = {n.removesuffix(".srt") for n in names if n.endswith(".srt") and wanted(n.removesuffix(".srt"))}
    if videos != captions or not videos:
        raise FileNotFoundError(f"{root}: {len(videos)} videos and {len(captions)} captions are wanted, and they must be the same recordings "
                                f"({sorted(videos ^ captions)[:3]} are in one only); `mmsb fetch` gets them")
    return tuple(Recording(stem, root / f"{stem}.mp4", speech.read_srt((root / f"{stem}.srt").read_text(encoding="utf-8-sig")))
                 for stem in sorted(videos))


def video_seconds(path: Path) -> float:
    """What an MP4 plays: its container's length."""
    with av.open(str(path)) as c:
        if c.duration is None:
            raise ValueError(f"{path} says no length")
        return c.duration / av.time_base


def audio_seconds(path: Path) -> float:
    """What the first sound track of `path` plays."""
    with av.open(str(path)) as c:
        track = c.streams.audio[0]
        if track.duration is None or track.time_base is None:
            raise ValueError(f"{path}'s sound says no length")
        return float(track.duration * track.time_base)


def tagged_time(path: Path) -> datetime:
    """The `date` an MP4 carries in its own tags (ISO, `2020-03-18T09:00:13Z`) as a wall-clock time with no zone: when its lecture was given,
    or its episode came out."""
    with av.open(str(path)) as c:
        tag = c.metadata.get("date")
    if tag is None:
        raise ValueError(f"{path} has no date tag")
    return datetime.fromisoformat(tag.removesuffix("Z"))


def ffmpeg(args: Sequence[str | Path], out: Path) -> None:
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *map(str, args), "-y", str(out)],
                         capture_output=True, text=True, check=False)
    if run.returncode:
        raise RuntimeError(f"ffmpeg could not write {out}: {run.stderr.strip()[-500:]}")


NO_TITLE = ("-map_metadata", "-1", "-map_chapters", "-1", "-metadata:s:v:0", "handler_name=", "-metadata:s:a:0", "handler_name=",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact")  # what no written file keeps of the source: title, encoder, handler names


def copy_video(src: Path, out: Path) -> None:
    """The video and first sound track of `src` as an MP4 by stream copy (no re-encode), with none of its metadata."""
    ffmpeg(["-i", src, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", *NO_TITLE], out)


def extract_sound(src: Path, out: Path) -> None:
    """The first sound track of `src` alone, as the file type of `out`'s suffix (`AUDIO`), with none of its metadata."""
    ffmpeg(["-i", src, "-vn", "-map", "0:a:0", *AUDIO[out.suffix.removeprefix(".")], *NO_TITLE], out)


def check_plays(path: Path, planned: float, seconds: Callable[[Path], float]) -> None:
    """Raises when the written file `path` does not play the `planned` seconds."""
    got = seconds(path)
    if abs(got - planned) > PLAY_TOL_S:
        raise ValueError(f"{path} plays {got:.2f} s, not the {planned:.2f} s it was planned with")


# -- what the probe reads -----------------------------------------------------------------------------------------------------------

def mp4_header(path: Path) -> bytes:
    """An MP4's (an M4A's) top-level boxes but the payload of `mdat`: what the file says about itself."""
    out = bytearray()
    with open(path, "rb") as f:
        while head := f.read(8):
            size, kind = int.from_bytes(head[:4], "big"), head[4:]
            if size == 1:  # a 64-bit size follows
                head += f.read(8)
                size = int.from_bytes(head[8:], "big")
            if len(head) < 8 or size and size < len(head):
                raise ValueError(f"{path} is not an MP4: a box of {size} bytes")
            out += head
            rest = size - len(head) if size else None  # None: the box goes to the end of the file
            if kind == b"mdat" and rest is not None:
                f.seek(rest, os.SEEK_CUR)
            elif kind != b"mdat":
                out += f.read(rest)
            if rest is None:
                break
    return bytes(out)


def mp3_header(path: Path) -> bytes:
    """An MP3's tags: the ID3v2 block that starts it, the ID3v1 block that ends it, and its first frame when that is a VBR tool's (Xing or Info:
    it names the tool). A file that does not start with a tag or a frame is refused."""
    with open(path, "rb") as f:
        head = f.read(10)
        out = bytearray()
        if head[:3] == b"ID3":
            size = sum((b & 0x7F) << (7 * i) for i, b in enumerate(reversed(head[6:10])))
            out += head + f.read(size)
            head = f.read(10)
        if len(head) < 4 or head[0] != 0xFF or head[1] & 0xE0 != 0xE0:
            raise ValueError(f"{path} does not start with an ID3 tag or an MP3 frame")
        frame = head + f.read(200)
        if b"Xing" in frame or b"Info" in frame:
            out += frame
        f.seek(-128, os.SEEK_END)
        if (tail := f.read(128))[:3] == b"TAG":
            out += tail
    return bytes(out)


HEADERS = {**realdump.METADATA, ".mp4": mp4_header, ".m4a": mp4_header, ".mp3": mp3_header}  # by suffix: what a talk dump's probe reads


# -- the days of a term -------------------------------------------------------------------------------------------------------------

def class_days(first: date, weekdays: Sequence[int], skip: Sequence[date], n: int) -> list[date]:
    """The first `n` days from `first` on that fall on one of `weekdays` (Monday is 0) and are not in `skip`: the days a course meets."""
    out, day = [], first
    while len(out) < n:
        if day.weekday() in weekdays and day not in skip:
            out.append(day)
        day += timedelta(days=1)
    return out


# -- the terms and the curated questions --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    """A `speech.Curated` question about one recording, which it names by a part of its source's stem (`lec04_`, `S01E01`) so that it holds
    whatever the scale's recordings are."""

    file: str
    key: str
    text: str
    template: str
    shape: str
    grade: Grade
    t0: float
    t1: float
    check: str
    value: str


def curated(notes: Sequence[Note], stems: Sequence[str]) -> list[Curated]:
    """The notes of the recordings in `stems` (the source stems of the talks, in the talks' order) as `Curated`; one whose recording is not
    there is left out (a tiny pile has a few of them), and one that names more than one is refused."""
    out = []
    for n in notes:
        at = [i for i, s in enumerate(stems) if n.file in s]
        if len(at) > 1:
            raise ValueError(f"the note {n.key}: {n.file!r} is in {len(at)} recordings")
        if at:
            out.append(Curated(n.key, n.text, n.template, n.shape, n.grade, at[0], n.t0, n.t1, n.check, n.value))
    return out


def screened(talks: Sequence[Talk], terms: Sequence[Term], one_a_day: bool = False, others: Collection[str] = OTHERS) -> list[Term]:
    """`terms` but those a question of them would be less than certain, or whose answer the list of files gives away:
    - one whose first mention is in the first `NEAR_START_S` seconds of its file (a cold open names its topic): the answer to "when" would then
      be the file's own time, and "which" and "when" one question;
    - one that someone of `others` (a student in the room, a voice that is no speaker of the talk) says, when the question asks what the
      speaker said;
    - with `one_a_day`, one that is said in two files of one day (a lecture the recorder was stopped in is two files, and "in which
      lecture" and "how many lectures" then have two answers).
    A term said nowhere in the talks is left out too (a tiny pile has a few of the talks)."""
    out = []
    for term in terms:
        found = [(t, mentions) for t in talks if (mentions := speech.mentions(t, term.search))]
        if not found or found[0][1][0].t0 < NEAR_START_S or any(cue.speaker in others for _, mentions in found for cue in mentions):
            continue
        if one_a_day and len({t.start.date() for t, _ in found}) != len(found):
            continue
        out.append(term)
    return out


def glued(talks: Sequence[Talk], search: str) -> bool:
    """Whether the letters and digits of `search` run on, whatever the spacing or the break between cues, in the talks' words: the way a
    caption writer may have set a name that was said ("bit coin" for "bitcoin")."""
    flat = "".join("".join(re.findall(r"[a-z0-9]", cue.text.casefold())) for t in talks for cue in t.cues)
    return re.sub(r"[^a-z0-9]", "", search.casefold()) in flat


def check_never(talks: Sequence[Talk], never: Sequence[Term]) -> None:
    """Raises when a term of `never` is in the talks at all, in any spacing: an absent question is certain only of what was not said."""
    if bad := [t.search for t in never if speech.said_anywhere(talks, t.search) or glued(talks, t.search)]:
        raise ValueError(f"said after all, or set by a caption writer as run-on letters, and so not absent: {', '.join(bad)}")
