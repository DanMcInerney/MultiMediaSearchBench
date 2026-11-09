"""The source of V4: twelve meetings of the AMI corpus on disk, three teams' four meetings each (ES2002, ES2015 and ES2016 a to d: a team's
design project for a remote control, a meeting a stage of it, all four on one day). Each meeting is a room's Corner video (`signals/<m>/video/
<m>.Corner.avi`, 352x288, 25 fps) and the headset mix of its four microphones (`signals/<m>/audio/<m>.Mix-Headset.wav`, 16 kHz mono), and
the manual annotations of `annotations/ami_public_manual_1.6.2.zip` as NXT files: `words` (every word of each speaker with its time),
`topics`, `decision`, `abstractive` (a written summary), `movement` (what each person does: sit, take notes, stand at the whiteboard, ...)
and `corpusResources/meetings.xml` (the date of a meeting and the role each speaker letter plays in it).

The speakers are letters, and their letters change from a team's meeting to the next team's (ES2002 has the project manager as B, ES2015 as A), so a
role is read from `meetings.xml` for each meeting, and checked twice here (the four speakers of a meeting have the four roles, and a person keeps
one role over a team's four meetings) and once more in the tests, against what the speakers say of themselves ("I'm the project manager"). The
corpus's `participantRoles` files are no help: they are the social roles of the argumentation work (gatekeeper, supporter). A meeting's date is real; its time of day is real for some meetings (the AMI Hub meetings have a `startTime`) and these have none,
so `start_times` draws it.

An NXT range, `file#id(a)..id(b)`, is the words from a to b in the order of that speaker's words file; a decision or a topic is some such ranges of
several speakers, and its span is from the first word's start to the last word's end. The XML is of the corpus's own files, and it is parsed
into the types below once, here: a word with no time, a range whose end is before its start and a movement that does not follow the one before
it are refused."""

from __future__ import annotations

import functools
import os
import random
import re
import subprocess
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import av
import imageio_ffmpeg

from mmsb.scenarios.realdump import sources_root
from mmsb.scenarios.realdump.speech import Cue

SOURCE = sources_root() / "ami"
ANNOTATIONS = "annotations/ami_public_manual_1.6.2.zip"
TEAMS = ("ES2002", "ES2015", "ES2016")  # the teams: their meetings are the letters a to d
LETTERS = "abcd"
MEETINGS = tuple(f"{t}{x}" for t in TEAMS for x in LETTERS)
DECIDED = ("ES2015", "ES2016")  # the teams whose decisions are annotated in all four meetings (ES2002 has them for a and d only)
ROLES = {"PM": "project manager", "ID": "industrial designer", "UI": "user interface designer", "ME": "marketing expert"}
NS = "{http://nite.sourceforge.net/}"
CUE_GAP_S = 1.0  # a pause this long ends a cue
CUE_WORDS = 40  # and so does this many words
SIGNAL_SLACK_S = 2.0  # the annotation's last word may be this far past the end of the shorter of a meeting's two signals
FIRST_START = (9 * 60, 11 * 60)  # a meeting's day starts, by a draw, between 09:00 and 11:00,
GAP_MIN = (20, 75)  # and each next meeting begins this many minutes after the last ended
MOVES = ("sit", "take_notes", "move", "stand_whiteboard", "stand_screen", "other", "off_camera")  # the ontology's `movement` types


@dataclass(frozen=True)
class Word:
    t0: float
    t1: float
    text: str
    punc: bool  # a full stop, a comma, a question mark: it has a time but is no spoken word


@dataclass(frozen=True)
class Span:
    t0: float
    t1: float


@dataclass(frozen=True)
class Topic:
    name: str  # the annotator's: from the ontology or written in
    t0: float
    t1: float
    top: bool  # a topic of the meeting's running order, not one inside another


@dataclass(frozen=True)
class Decision:
    key: str  # `ES2016a.decision.melissa.3`'s last part: `3`
    external: bool  # a decision the team was given, not one it made (the price, the budget): the annotation's `external`
    t0: float
    t1: float
    words: int


@dataclass(frozen=True)
class Move:
    agent: str
    kind: str
    t0: float
    t1: float


@dataclass(frozen=True)
class Meeting:
    id: str  # `ES2016b`
    day: date
    start: time | None  # the meeting's real start of day, when the corpus gives it
    roles: dict[str, str]  # a speaker letter's role key (PM, ID, UI, ME)
    words: dict[str, tuple[Word, ...]]  # by speaker letter
    topics: tuple[Topic, ...]
    decisions: tuple[Decision, ...]
    summary: dict[str, tuple[str, ...]]  # the written summary's parts (abstract, actions, decisions, problems), by name
    moves: tuple[Move, ...]

    def role_of(self, agent: str) -> str:
        return ROLES[self.roles[agent]]

    def cues(self) -> tuple[Cue, ...]:
        """The meeting as the speech templates read it: each speaker's words cut into cues at a pause or a sentence's end, all speakers' cues in
        the order of their starts, a cue with the role of its speaker."""
        out: list[Cue] = []
        for agent, words in self.words.items():
            run: list[Word] = []
            for w in words:
                if run and (w.t0 - run[-1].t1 > CUE_GAP_S or len([x for x in run if not x.punc]) >= CUE_WORDS or run[-1].text in ".?!"):
                    out.append(cue_of(run, self.role_of(agent)))
                    run = []
                run.append(w)
            if run:
                out.append(cue_of(run, self.role_of(agent)))
        return tuple(sorted(out, key=lambda c: (c.t0, c.t1)))


def cue_of(run: Sequence[Word], role: str) -> Cue:
    said = [w for w in run if not w.punc]
    text = ""
    for w in run:
        text += w.text if w.punc else " " + w.text
    return Cue(said[0].t0 if said else run[0].t0, max(w.t1 for w in run), text.strip(), role)


def parse_day(text: str) -> date:
    """`13-12-04` or `30-11-2004` (day, month, year) as a date."""
    d, m, y = text.split("-")
    return date(int(y) + 2000 if len(y) == 2 else int(y), int(m), int(d))


class Archive:
    """The annotation zip, read member by member into XML."""

    def __init__(self, source: Path | str):
        self.zip = zipfile.ZipFile(Path(source) / ANNOTATIONS)

    def names(self) -> set[str]:
        return set(self.zip.namelist())

    def xml(self, name: str) -> ET.Element:
        return ET.fromstring(self.zip.read(name))


def ranged(href: str) -> tuple[str, str, str]:
    """`ES2016a.A.words.xml#id(ES2016a.A.words646)..id(ES2016a.A.words706)` as the speaker letter and the two ends (the same when one word)."""
    file, frag = href.split("#")
    m = re.fullmatch(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?", frag)
    if m is None or len(file.split(".")) < 3:
        raise ValueError(f"not an NXT word range: {href}")
    return file.split(".")[1], m[1], m[2] or m[1]


def read_words(archive: Archive, meeting: str) -> tuple[dict[str, tuple[Word, ...]], dict[str, dict[str, int]], dict[str, list[tuple[float, float] | None]]]:
    """The words of each speaker of `meeting`, the position of each element's id in its file, and each element's time (None for one with none: a
    pause, a gap). Every word has a time."""
    words, where, times = {}, {}, {}
    for agent in "ABCD":
        out, ids, at = [], {}, []
        for i, e in enumerate(archive.xml(f"words/{meeting}.{agent}.words.xml")):
            ids[e.get(NS + "id")] = i
            t0, t1 = e.get("starttime"), e.get("endtime")
            at.append((float(t0), float(t1)) if t0 and t1 else None)
            if e.tag == "w":
                if at[-1] is None:
                    raise ValueError(f"{meeting} {agent}: the word {e.text!r} ({e.get(NS + 'id')}) has no time")
                out.append(Word(at[-1][0], at[-1][1], e.text or "", e.get("punc") == "true"))
        words[agent], where[agent], times[agent] = tuple(out), ids, at
    return words, where, times


def span_of(children: Sequence[ET.Element], where, times, what: str) -> tuple[float, float, int]:
    """The span of the word ranges `children` point at, and how many elements they hold; refused when there is none with a time or a range runs
    backwards."""
    t0, t1, n = float("inf"), 0.0, 0
    for c in children:
        agent, a, b = ranged(c.get("href"))
        i, j = where[agent][a], where[agent][b]
        if j < i:
            raise ValueError(f"{what}: the range {a}..{b} runs backwards")
        for t in times[agent][i:j + 1]:
            if t is not None:
                t0, t1, n = min(t0, t[0]), max(t1, t[1]), n + 1
    if not n:
        raise ValueError(f"{what}: no word with a time")
    return t0, t1, n


def read_roles(archive: Archive) -> dict[str, tuple[date, time | None, dict[str, tuple[str, str]]]]:
    """From `meetings.xml`, each meeting's date, start of day when it has one, and its speakers: the letter's (role, person)."""
    out = {}
    for m in archive.xml("corpusResources/meetings.xml"):
        if m.get("observation") in MEETINGS:
            start = m.get("startTime")
            hh = re.fullmatch(r"(\d\d)h(\d\d)", start or "")
            speakers = {s.get("nxt_agent"): (s.get("role"), s.get("global_name")) for s in m.iter("speaker")}
            out[m.get("observation")] = (parse_day(m.get("dateOnly")), time(int(hh[1]), int(hh[2])) if hh else None, speakers)
    return out


def check_roles(found: dict[str, tuple[date, time | None, dict[str, tuple[str, str]]]], archive: Archive) -> None:
    """The checks of the roles: four speakers, four roles; one person one role over a team's four meetings."""
    for m, (_, _, speakers) in found.items():
        if sorted(speakers) != list("ABCD") or sorted(r for r, _ in speakers.values()) != sorted(ROLES):
            raise ValueError(f"{m}: the speakers are not the four roles once each: {speakers}")
    for team in TEAMS:
        people = {}
        for x in LETTERS:
            for role, person in found[f"{team}{x}"][2].values():
                if people.setdefault(person, role) != role:
                    raise ValueError(f"{team}: {person} is {people[person]} in one meeting and {role} in {team}{x}")


def read_topics(archive: Archive, meeting: str, where, times) -> tuple[Topic, ...]:
    """The meeting's topics by start, with the annotator's name (written in, or the ontology's by the pointer). `()` when the meeting has no topic
    file (ES2015b and c)."""
    name = f"topics/{meeting}.topic.xml"
    if name not in archive.names():
        return ()
    onto = {e.get(NS + "id"): e.get("name") for e in archive.xml("ontologies/default-topics.xml").iter()}
    out: list[Topic] = []

    def walk(parent: ET.Element, top: bool) -> None:
        for t in parent:
            if t.tag != "topic":
                continue
            label = t.get("other_description")
            for p in t:
                if p.tag == NS + "pointer" and not label:
                    label = onto[re.search(r"#id\(([^)]+)\)", p.get("href"))[1]]
            if not label or label == "other":
                raise ValueError(f"{meeting}: a topic with no name")
            kids = [c for c in t.iter(NS + "child")]  # the topic's own words and those of the topics inside it
            if kids:
                t0, t1, _ = span_of(kids, where, times, f"{meeting} topic")
                out.append(Topic(label.strip(), t0, t1, top))
            walk(t, False)

    walk(archive.xml(name), True)
    return tuple(sorted(out, key=lambda t: (t.t0, t.t1)))


def read_decisions(archive: Archive, meeting: str, where, times) -> tuple[Decision, ...]:
    name = f"decision/manual/{meeting}.decision.xml"
    if name not in archive.names():
        return ()
    out = []
    for d in archive.xml(name):
        kids = [c for c in d if c.tag == NS + "child"]
        t0, t1, n = span_of(kids, where, times, f"{meeting} decision")
        out.append(Decision(d.get(NS + "id").split(".")[-1], d.get("external") == "true", t0, t1, n))
    return tuple(sorted(out, key=lambda d: d.t0))


def read_summary(archive: Archive, meeting: str) -> dict[str, tuple[str, ...]]:
    return {part.tag: tuple((s.text or "").strip() for s in part if s.tag == "sentence") for part in archive.xml(f"abstractive/{meeting}.abssumm.xml")}


def read_moves(archive: Archive, meeting: str) -> tuple[Move, ...]:
    """What each speaker does, from the start of the meeting to its end: each speaker's movements follow one another with no gap and none
    overlaps (refused when they do), so that no stretch is unlabelled. The last of a file may be an `off_camera` that begins after the
    meeting ends (the later meetings of ES2016 have one, begun a moment after the audio ends and ended at its end): that end marker is left out."""
    out = []
    for agent in "ABCD":
        found = [Move(agent, e.get("type"), float(e.get("starttime")), float(e.get("endtime")))
                 for e in archive.xml(f"movement/{meeting}.{agent}.movement.xml") if e.tag == "movement"]
        if found and found[-1].kind == "off_camera" and found[-1].t1 < found[-1].t0:
            found.pop()
        for before, m in zip([None, *found[:-1]], found, strict=True):
            if m.kind not in MOVES or m.t1 < m.t0 or (before is None and m.t0 > 5) or (before is not None and abs(m.t0 - before.t1) > 0.01):
                raise ValueError(f"{meeting} {agent}: the movement {m} does not follow the one before it, or is not of the ontology")
        out += found
    return tuple(out)


@functools.lru_cache(maxsize=2)
def read_meetings(source: str, ids: tuple[str, ...] = MEETINGS) -> tuple[Meeting, ...]:
    """The meetings `ids` (in the order given) with every layer the questions use."""
    archive = Archive(source)
    found = read_roles(archive)
    check_roles(found, archive)
    out = []
    for m in ids:
        words, where, times = read_words(archive, m)
        day, start, speakers = found[m]
        out.append(Meeting(m, day, start, {a: r for a, (r, _) in speakers.items()}, words, read_topics(archive, m, where, times),
                           read_decisions(archive, m, where, times), read_summary(archive, m), read_moves(archive, m)))
    return tuple(out)


def seconds_of(source: str, meeting: str) -> float:
    """How long the muxed file plays: the longer of the room's video and the headset mix (they differ by seconds; both start at 0, and the file
    ends with the longer one)."""
    base = Path(source) / "signals" / meeting
    lengths = []
    for rel in (f"video/{meeting}.Corner.avi", f"audio/{meeting}.Mix-Headset.wav"):
        with av.open(str(base / rel)) as c:
            lengths.append(c.duration / 1_000_000)
    return max(lengths)


def start_times(meetings: Sequence[Meeting], seconds: dict[str, float]) -> dict[str, datetime]:
    """When each meeting began (wall clock, no zone): the corpus's own time of day where it has one; else a draw, the same for every seed (it
    is the pile's), that starts a day between 09:00 and 11:00 and puts each next meeting of the day 20 to 75 minutes after the last ended."""
    out: dict[str, datetime] = {}
    for day in sorted({m.day for m in meetings}):
        rng = random.Random(f"v4:{day.isoformat()}")
        free = datetime.combine(day, time.min) + timedelta(minutes=rng.randint(*FIRST_START), seconds=rng.randint(0, 59))
        for m in (m for m in meetings if m.day == day):
            began = datetime.combine(day, m.start) if m.start else free
            out[m.id] = began
            free = began + timedelta(seconds=round(seconds[m.id]), minutes=rng.randint(*GAP_MIN))
    return out


def mux(source: str, meeting: str, out: Path) -> None:
    """The meeting as one MP4: the room's video re-encoded to H.264 and the headset mix to AAC, both from time 0 so that the annotations' seconds
    are the file's, with no title or tool name."""
    base = Path(source) / "signals" / meeting
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(base / "video" / f"{meeting}.Corner.avi"),
                          "-i", str(base / "audio" / f"{meeting}.Mix-Headset.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset",
                          "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-ar", "16000", "-map_metadata", "-1",
                          "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-movflags", "+faststart", "-y", str(out)],
                         capture_output=True, text=True, check=False)
    if run.returncode:
        raise RuntimeError(f"ffmpeg could not make {out} from {meeting}: {run.stderr.strip()[-500:]}")


def mp4_header(path: Path) -> bytes:
    """An MP4's top-level boxes but the payload of `mdat` (the media): what the file says about itself, for the leak probe."""
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
