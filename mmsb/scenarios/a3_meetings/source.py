"""The source of A3: ten meetings of the ICSI Meeting Recorder group (the `Bmr` series: the group's own weekly meeting, recorded from February 2000 to
March 2001) on disk. A meeting is one WAV (`signals/<m>.interaction.wav`, 16 kHz mono, the close-talking microphones mixed) and, from
`annotations/ICSI_original_transcripts.zip`, the ICSI transcript `transcripts/<m>.mrt`: its date and time (`DateTimeStamp`, to the minute), and
every speaker's every utterance as a segment with its start and end second and its words (the words of the digit-reading task as well), with the
transcribers' marks for laughter, breath and so on between them. From `annotations/ICSI_plus_NXT.zip` come the topic segmentations (`Contributions/
TopicSegmentation/<m>.topic.xml`, the coder's names for what was talked about, nested) over the NXT segments (`Segments/<m>.<speaker>.segs.xml`).
Nothing is read of who the speakers are: they are real, named people, and no question is about one.

A segment the transcribers marked `Digits` (a comment in it) is a string of digits one of the group read out, the weekly ritual of the corpus.

A segment is a cue (a long one, over 12 seconds, is cut into cues of 25 words with times by word count). The text of a cue leaves out what a
transcriber marked `Uncertain` (a guess at an unclear word), and the text of a segment as the transcript has it, the guesses too, is kept for the
checks: a term is used only where the two read the same.

The transcript's own spelling is kept: letters said one by one are `P_Z_M_`, which the speech templates read as p, z and m (so no term is an
acronym, and an absent term is checked in the letters too)."""

from __future__ import annotations

import functools
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import av

from mmsb.scenarios.realdump import sources_root
from mmsb.scenarios.realdump.speech import Cue

SOURCE = sources_root() / "icsi"
TRANSCRIPTS = "annotations/ICSI_original_transcripts.zip"
NXT = "annotations/ICSI_plus_NXT.zip"
MEETINGS = ("Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007", "Bmr008", "Bmr009", "Bmr010", "Bmr019")
NS = "{http://nite.sourceforge.net/}"
SKIP = {"Comment", "VocalSound", "NonVocalSound", "Pause", "Gap"}  # a transcriber's marks between the words: no text
LONG_S = 12.0  # a segment longer than this is cut into cues
CUE_WORDS = 25
MAX_BROKEN = 0.01  # the share of a topic file's references that may point at nothing (Bmr019's has two)


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    who: str  # the transcript's speaker code, kept to read each speaker's words alone and to nothing else
    text: str  # without what is marked Uncertain
    loose: str  # with it
    digits: bool = False  # the transcribers marked it `Digits`: a string of digits read out


@dataclass(frozen=True)
class Topic:
    name: str  # the coder's
    t0: float
    t1: float
    depth: int


@dataclass(frozen=True)
class Meeting:
    id: str
    start: datetime  # wall clock, no zone: the transcript's DateTimeStamp
    segments: tuple[Segment, ...]
    topics: tuple[Topic, ...]

    def cues(self) -> tuple[Cue, ...]:
        out: list[Cue] = []
        for s in self.segments:
            words = s.text.split()
            if not words:
                continue
            if s.t1 - s.t0 <= LONG_S or len(words) <= CUE_WORDS:
                out.append(Cue(s.t0, s.t1, s.text))
                continue
            step = s.t1 - s.t0
            for i in range(0, len(words), CUE_WORDS):
                chunk = words[i:i + CUE_WORDS]
                out.append(Cue(s.t0 + step * i / len(words), s.t0 + step * (i + len(chunk)) / len(words), " ".join(chunk)))
        return tuple(sorted(out, key=lambda c: (c.t0, c.t1)))


def text_of(e: ET.Element, uncertain: bool) -> str:
    """The words of a segment: its text and its marks' text (an emphasis), but for the marks that are no words, and for a guess when `uncertain`
    is false."""
    out = [e.text or ""]
    for c in e:
        if c.tag not in SKIP and (uncertain or c.tag != "Uncertain"):
            out.append(text_of(c, uncertain))
        out.append(c.tail or "")
    return " ".join(" ".join(out).split())


def read_transcript(zf: zipfile.ZipFile, meeting: str) -> tuple[datetime, tuple[Segment, ...]]:
    root = ET.fromstring(zf.read(f"transcripts/{meeting}.mrt"))
    if root.get("Session") != meeting:
        raise ValueError(f"transcripts/{meeting}.mrt is the transcript of {root.get('Session')}")
    start = datetime.strptime(root.get("DateTimeStamp"), "%Y-%m-%d-%H%M")
    segments = []
    for s in root.iter("Segment"):
        t0, t1 = float(s.get("StartTime")), float(s.get("EndTime"))
        if t1 < t0:
            raise ValueError(f"{meeting}: a segment from {t0} to {t1}")
        digits = any(c.get("Description") == "Digits" for c in s.iter("Comment"))
        segments.append(Segment(t0, t1, s.get("Participant"), text_of(s, False), text_of(s, True), digits))
    return start, tuple(sorted(segments, key=lambda s: (s.t0, s.t1)))


def read_topics(zf: zipfile.ZipFile, meeting: str) -> tuple[Topic, ...]:
    """The topics of the main topic file (the coder's names, nested: `depth` 0 is the running order), with the span of each (all that is in it,
    nested topics too) from the segments they point at. A reference that points at no segment (the corpus's own slip) is left out when there
    are few, and refused when there are many."""
    name = f"ICSIplus/Contributions/TopicSegmentation/{meeting}.topic.xml"
    spans: dict[str, list[tuple[float, float]]] = {}
    for member in zf.namelist():
        if re.fullmatch(rf"ICSIplus/Segments/{meeting}\.\w+\.segs\.xml", member):
            spans[member.split("/")[-1]] = [(float(s.get("starttime")), float(s.get("endtime")), s.get(NS + "id"))
                                            for s in ET.fromstring(zf.read(member))]
    broken = total = 0
    out: list[Topic] = []

    def ends(topic: ET.Element) -> list[tuple[float, float]]:
        nonlocal broken, total
        found = []
        for c in topic.iter(NS + "child"):
            total += 1
            file, frag = c.get("href").split("#")
            m = re.fullmatch(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", frag)
            ids = [x[2] for x in spans[file]]
            if m[1] not in ids or (m[2] or m[1]) not in ids:
                broken += 1
                continue
            found += [(s[0], s[1]) for s in spans[file][ids.index(m[1]):ids.index(m[2] or m[1]) + 1]]
        return found

    def walk(parent: ET.Element, depth: int) -> None:
        for t in parent:
            if t.tag == "topic":
                found = ends(t)
                if found:
                    out.append(Topic(t.get("description", "").strip(), min(a for a, _ in found), max(b for _, b in found), depth))
                walk(t, depth + 1)

    walk(ET.fromstring(zf.read(name)), 0)
    if broken > MAX_BROKEN * max(total, 1):
        raise ValueError(f"{name}: {broken} of {total} references point at no segment")
    return tuple(sorted(out, key=lambda t: (t.t0, -t.t1)))


@functools.lru_cache(maxsize=2)
def read_meetings(source: str, ids: tuple[str, ...] = MEETINGS) -> tuple[Meeting, ...]:
    with zipfile.ZipFile(Path(source) / TRANSCRIPTS) as tz, zipfile.ZipFile(Path(source) / NXT) as nz:
        out = []
        for m in ids:
            start, segments = read_transcript(tz, m)
            out.append(Meeting(m, start, segments, read_topics(nz, m)))
    return tuple(out)


def seconds_of(source: str, meeting: str) -> float:
    with av.open(str(Path(source) / "signals" / f"{meeting}.interaction.wav")) as c:
        return c.duration / 1_000_000
