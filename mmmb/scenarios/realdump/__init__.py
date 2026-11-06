"""What the scenarios made from real recordings share (PLAN.md): a dump that is copies of a source's files under the names a real device
or folder would give them, each file's time set to when it was recorded; its manifest; the questions' common fields and their split; the
prompt; the words no command may hold; a leak probe; and the summary report. Each such scenario is then a reader of its source and
labels, a writer of its files and its question templates; it imports this and no other scenario.

Times. A source's labels give a recording's wall-clock time with no zone. A dump keeps it as wall-clock time on this machine: the file's
time is that wall time in this machine's zone, so `ls` and the file's own metadata (a photo's EXIF) read the same clock, and so do the
question's `now` and the grader's clock times. A recipe records this machine's UTC offsets at its first and last day (`utc_offsets`) and
is refused where they differ, as another zone would make other files."""

from __future__ import annotations

import dataclasses
import os
import random
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmmb.core import fs
from mmmb.core.answer import answer_request
from mmmb.core.contract import Dump, ManifestEntry
from mmmb.core.grade import asked
from mmmb.core.question import Grade, Question, Scope, Span, Truth, line_of
from mmmb.core.runfolder import IN_ONLY
from mmmb.core.scenario import Access, ProbeOutcome, files_hash, plan_hash
from mmmb.core.summary import report  # noqa: F401  (the scenarios' `report`)

DEV_SHARE = 0.3  # of a scenario's questions, the share marked `dev` by the seeded draw; the rest are `test`
CODE_WORDS = ("realdump", "from mmmb", "import mmmb")  # code that reaches a scenario: no command may hold it
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


@dataclass(frozen=True)
class File:
    """A file of a dump: where it goes, what it is, when it was recorded (`t`, epoch seconds, which becomes its modified time) and what
    it is made from (`spec`, plain JSON: its scenario's writer reads it, and `files_hash` covers it)."""

    path: str  # in the dump, with `/`
    source: str  # the dump's name for the device or folder it is from
    t: float
    kind: str  # photo, clip, recording, ...
    media: str  # image, video, audio
    spec: Mapping[str, object]
    length: float | None = None  # seconds it plays, for a video or a sound
    own_time: bool = False  # its metadata carries its time (a photo's EXIF)


# -- times --------------------------------------------------------------------------------------------------------------------------

def local_t(wall: datetime) -> float:
    """Epoch seconds of the wall-clock time `wall` (no zone) on this machine."""
    if wall.tzinfo is not None:
        raise ValueError(f"{wall} has a zone: a source's wall-clock time has none")
    return wall.timestamp()


def iso_local(t: float) -> str:
    """`t` as an ISO time with this machine's offset then (to the second)."""
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def span_days(first: date, last: date) -> list[str]:
    """Every day from `first` to `last`, ISO: a recipe's `days`."""
    return [(first + timedelta(days=n)).isoformat() for n in range((last - first).days + 1)]


def utc_offsets(days: Sequence[str]) -> list[str]:
    """This machine's UTC offsets (`-0500`) at the local midnights that start the first and the last of `days`."""
    return [datetime.combine(date.fromisoformat(d), time.min).astimezone().strftime("%z") for d in (days[0], days[-1])]


def check_offsets(recipe: Mapping[str, object]) -> None:
    """Refuses a recipe made in another time zone: its files' times would be other instants."""
    here = utc_offsets(recipe["days"])
    if list(recipe["utc_offsets"]) != here:
        raise ValueError(f"the recipe was made where the UTC offsets of its days are {', '.join(recipe['utc_offsets'])}, here they are "
                         f"{', '.join(here)}: its files' times are wall-clock times of this machine, so it is another dump; make a new recipe")


def date_text(day: date) -> str:
    """`Tuesday 17 December 2013`, from fixed English names (not `strftime`, which follows the locale)."""
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]} {day.year}"


# -- the dump -----------------------------------------------------------------------------------------------------------------------

def file_rows(files: Sequence[File]) -> list[list]:
    """The files as `files_hash` rows, `[path, source, time, name, kind, spec]`."""
    return [[f.path, f.source, f.t, None, f.kind, dict(f.spec)] for f in files]


def hashes(questions: Sequence[Question], files: Sequence[File]) -> tuple[str, str]:
    """`Scenario.hashes` of a plan's questions and files (no file is written)."""
    return plan_hash([line_of(q) for q in questions]), files_hash(file_rows(files), {})


def write(files: Sequence[File], out: Path, workers: int, writer: Callable[[File, Path], None]) -> tuple[ManifestEntry, ...]:
    """Writes each of `files` into `out` (new or empty) with `writer(file, path)`, `workers` at once, sets its modified time to its
    `t`, and measures it: the manifest, in the order of `files`."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: a dump is written into a folder of its own")
    if len({f.path for f in files}) != len(files):
        raise ValueError("two files of the dump have one path")

    def one(f: File) -> ManifestEntry:
        path = out / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(f, path)
        os.utime(path, (f.t, f.t))
        info = path.stat()
        return ManifestEntry(f.path, f.source, f.t, f.kind, f.media, None, f.own_time, info.st_size, info.st_mtime_ns, fs.sha256_of(path),
                             f.length)

    with ThreadPoolExecutor(workers) as pool:
        return tuple(pool.map(one, files))


def dump_of(key: str, out: Path, scenario: str, entries: Sequence[ManifestEntry]) -> Dump:
    return Dump(key, Path(out), scenario, tuple(entries), ())


# -- the questions ------------------------------------------------------------------------------------------------------------------

def grade(rule: str, *, truth: str | None = None, time: str | None = None, tol_s: float | None = None, date: str | None = None,
          items: Sequence[str] = (), ignore: Sequence[str] = ()) -> Grade:
    return Grade(rule, truth, time, tol_s, (), tuple(ignore), date, tuple(items))


def question(*, code: str, qid: str, now: float, text: str, kind: str, template: str, modality: str, shape: str, grade: Grade,
             evidence: Sequence[Span], sources: Sequence[str], since: float, until: float, value: str | None) -> Question:
    """A question of a real dump, `dev` until `split` draws: asked as of `now` over the whole dump (nothing is cut), with the answer's
    `shape` for the prompt. An absent question (`kind` absent) has no evidence and no value. The numbers and clock times of `text` join
    the grade's `ignore` (`grade.asked`): an answer may repeat them ("none in September 2013")."""
    absent = kind == "absent"
    if absent != (grade.rule == "absent") or absent == bool(evidence):
        raise ValueError(f"{qid}: an absent question has the rule absent and no evidence, and only it")
    grade = dataclasses.replace(grade, ignore=tuple(dict.fromkeys((*grade.ignore, *asked(text)))))
    return Question(qid=qid, scenario=code, split="dev", now=iso_local(now), question=text, attachments=(), kind=kind, hint="none",
                    truth=Truth(value, tuple(evidence), ()), grade=grade,
                    tags={"modality": modality, "template": template, "level": "none", "absent": absent, "shape": shape},
                    options=None, scope=Scope(tuple(sources), since, until), group=qid, horizon=None, cost_ranges=((since, until),),
                    retrieval=None)


def split(questions: Sequence[Question], seed: int) -> tuple[Question, ...]:
    """`questions` with DEV_SHARE of them (rounded) marked `dev` and the rest `test`, by a draw seeded with `seed`."""
    order = sorted(q.qid for q in questions)
    random.Random(f"split:{seed}").shuffle(order)
    dev = set(order[:round(DEV_SHARE * len(order))])
    return tuple(dataclasses.replace(q, split="dev" if q.qid in dev else "test") for q in questions)


# -- what an agent is asked ---------------------------------------------------------------------------------------------------------

def prompt(question: Question, access: Access, what: str, layout: str) -> str:
    """The prompt for `question` through `access`: the time, how the files are reached (`what` they are; their `layout`, for an agent
    with its own tools), the question, and how to answer in the shape the question's `shape` tag gives."""
    t = datetime.fromisoformat(question.now)
    head = f"It is now {date_text(t.date())}, {t:%H:%M}."
    served, how = access.served, []
    if served.server is not None:
        if served.holder is None:
            raise ValueError(f"the server {served.server.name!r} has no word for itself for the prompt")
        how.append(f"{served.holder} holds {what}." + (f" {served.clock_note}" if served.clock_note else ""))
    if access.own_tools:
        how.append(f"The files of {what} are in {access.data}: {layout} Your working folder is yours to write in. ffmpeg is on PATH (there is no ffprobe).")
    if served.server is not None:
        holder, plural = (f"{served.holder} and these files", True) if access.own_tools else (served.holder, False)
    elif access.own_tools:
        holder, plural = "the files", True
    else:
        how.append(f"You cannot see {what}.")
        holder, plural = "what you know", False
    end = answer_request(access.answer_format, holder, str(question.tags["shape"]), plural)
    return f"{head} {' '.join(how)}\n\n{question.question}\n\n{end}"


def deny_words(scenario: str, command_words: Sequence[str], words: Sequence[str]) -> tuple[str, ...]:
    """What no command of an agent with its own tools may hold: the code that reaches the scenario and `command_words` (the source's
    names); and `words`, which neither a command nor what it prints may hold (the folder the source is in)."""
    return (*(IN_ONLY + w for w in (scenario, *CODE_WORDS, *command_words)), *words)


# -- the probe ----------------------------------------------------------------------------------------------------------------------

def jpeg_header(path: Path) -> bytes:
    """A JPEG's segments before its image data (the start of scan): its markers and metadata."""
    with open(path, "rb") as f:
        data = f.read(1 << 18)
    at = 2
    while at + 4 <= len(data) and data[at] == 0xFF and data[at + 1] != 0xDA:
        at += 2 + int.from_bytes(data[at + 2:at + 4], "big")
    return data[:at]


def riff_header(path: Path) -> bytes:
    """A RIFF file's (a WAV's) chunks but its `data` chunk's samples: its metadata."""
    out = bytearray()
    with open(path, "rb") as f:
        out += f.read(12)
        while len(head := f.read(8)) == 8:
            size = int.from_bytes(head[4:], "little")
            out += head
            if head[:4] == b"data":
                f.seek(size + (size & 1), os.SEEK_CUR)
            else:
                out += f.read(size + (size & 1))
    return bytes(out)


METADATA = {".jpg": jpeg_header, ".jpeg": jpeg_header, ".wav": riff_header}  # by suffix: what of a file the probe reads


STOP_WORDS = {"and", "the", "of", "in", "on", "at", "with", "by", "for", "other", "person", "vehicle", "object", "human", "sound"}


def label_words(labels: Sequence[str]) -> list[str]:
    """What the probe looks for: each label as a phrase (`_` read as a space) and each of its words of three letters or more but the
    commonest (`STOP_WORDS`), so that "microwave beep" is caught for the label `Microwave_oven`. A label written as one word
    (`vacuumcleaner`) is one word: a scenario passes the words it is said in as labels too."""
    out = set()
    for label in labels:
        if phrase := label.casefold().replace("_", " ").strip():
            out.add(phrase)
            out |= {w for w in re.split(r"[^a-z]+", phrase) if len(w) >= 3 and w not in STOP_WORDS}
    return sorted(out, key=lambda w: (-len(w), w))  # the longest first: a phrase is told before a word of it


def text_of(header: bytes) -> str:
    """The text of a file's metadata: its runs of four printable characters or more (the binary tables between them hold no words)."""
    return " ".join(m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{4,}", header))


def probe(dump: Dump, out: Path, labels: Sequence[str], metadata: Mapping[str, Callable[[Path], bytes]] = METADATA) -> ProbeOutcome:
    """The leak probe of a real dump: no file's path, nor the text of its metadata (read by its suffix with `metadata`: a file of a
    suffix with no reader fails, so a scenario of other files passes its own readers), names one of `labels` (the source's labels, each
    as a phrase and by its words, `label_words`; whole words, in any case, `_` read as a space). Writes `shortcuts.md` into `out`. The
    files' sizes are not checked: they are the recordings' own (in I3 an animal's photo tends to be smaller than an empty one, as on any
    camera's card)."""
    words = label_words([w for w in labels if w])
    pattern = re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")(?![a-z])")
    failures = []
    for e in dump.entries:
        if found := pattern.search(e.path.casefold().replace("_", " ")):
            failures.append(f"{e.path}: its name says {found[1]!r}")
            continue
        read = metadata.get(Path(e.path).suffix.casefold())
        if read is None:
            failures.append(f"{e.path}: its metadata is not read (no reader for {Path(e.path).suffix or 'a file with no suffix'})")
        elif found := pattern.search(text_of(read(dump.root / e.path)).casefold().replace("_", " ")):
            failures.append(f"{e.path}: its metadata says {found[1]!r}")
    lines = ["# Leak probe", "", f"{len(dump.entries)} files; label words: {', '.join(words)}.", ""]
    lines += [f"- {f}" for f in failures] or ["No file's name or metadata names a label."]
    fs.write_text(Path(out) / "shortcuts.md", "\n".join(lines) + "\n")
    return ProbeOutcome("failed" if failures else "passed", tuple(failures))
