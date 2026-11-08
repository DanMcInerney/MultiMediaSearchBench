"""The EgoLife download as plain data (moved from egolife.py of this repository at bb498a6): the clips of Jake's glasses with their lengths,
the questions of EgoLifeQA and the transcripts, read as the data gives them. The download is `A1_JAKE/DAYn/DAYn_A1_JAKE_HHMMSSFF.mp4`,
`EgoLifeQA/EgoLifeQA_A1_JAKE.json` and `EgoLifeCap/Transcript/A1_JAKE/DAYn/*.srt`: third-party data, read here and never run.

Times: a `plain` time is `day * 86400 + seconds into the day`, with no zone, which is what the data names; the plan turns them into local
epochs (plan.py). So the questions chosen for a seed do not depend on the day the run is built on.

A clip's length is read from its packets, as MultimediaSearch's `media.duration` does (it is the end mms stores for the clip): `packet_duration` is a
copy of it, so that no module of the scenario imports MultimediaSearch. tests/mm runs both on the same clips and asserts that they agree."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import av

PERSON, DAYS = "A1_JAKE", 3
DAY_S = 86400
QA_FILE, CLIP_FOLDER, SRT_FOLDER = f"EgoLifeQA/EgoLifeQA_{PERSON}.json", PERSON, f"EgoLifeCap/Transcript/{PERSON}"

CATEGORIES = ("EventRecall", "EntityLog", "RelationMap", "TaskMaster", "HabitInsight")

CLIP_NAME = re.compile(rf"DAY(\d+)_{PERSON}_(\d{{8}})\.mp4")
SRT_NAME = re.compile(rf"{PERSON}_DAY(\d+)_(\d{{8}})\.srt")
SRT_TIME = re.compile(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})")
SPEAKER = re.compile(r"\s*([^:：]{1,20})[:：]\s*(.*)", re.S)


def plain(day: int, seconds: float) -> float:
    """A plain time: `seconds` into day number `day` (1 is the first), as seconds from the start of day 0."""
    return day * DAY_S + seconds


def clock_of(text: str) -> float:
    """Seconds into the day of EgoLife's `HHMMSSFF`, where FF is hundredths of a second: 11210217 is 11:21:02.17."""
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"a time is HHMMSSFF (8 digits), not {text!r}")
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:6]) + int(text[6:]) / 100


def day_of(text: str) -> int:
    """The day number of `DAY1` (the data also writes `Day1`)."""
    found = re.fullmatch(r"day(\d+)", text.strip().lower())
    if found is None:
        raise ValueError(f"a day is DAY and a number, not {text!r}")
    return int(found.group(1))


@dataclass(frozen=True)
class Clip:
    rel: str  # in the download: A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4
    day: int
    t0: float  # plain, from the name
    length: float  # seconds, as `packet_duration` gives it (the end mms stores)

    @property
    def t1(self) -> float:
        return self.t0 + self.length


def packet_duration(path: Path) -> float | None:
    """How long a media file plays, from its packets; None when PyAV finds nothing in it (a copy of `media.duration` of MultimediaSearch at
    fe0ece5, without the `sound` option)."""
    first = last = None
    try:
        with av.open(str(path)) as container:
            for packet in container.demux():
                if packet.pts is None or packet.stream.time_base is None:
                    continue
                begin = float(packet.pts * packet.stream.time_base)
                end = begin + float((packet.duration or 0) * packet.stream.time_base)
                first = begin if first is None else min(first, begin)
                last = end if last is None else max(last, end)
    except (av.FFmpegError, OSError, ValueError):
        pass  # a file cut short ends in an error: what was read counts
    return None if first is None else max(last - first, 0.0)


def clip_of(rel: str, length: float) -> Clip:
    """The clip at `rel` in the download (`A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4`), which plays `length` seconds."""
    parts = rel.split("/")
    found = CLIP_NAME.fullmatch(parts[-1])
    if len(parts) != 3 or parts[0] != CLIP_FOLDER or found is None or parts[1] != f"DAY{found.group(1)}":
        raise ValueError(f"{rel} is not a clip of the download: {CLIP_FOLDER}/DAYn/DAYn_{PERSON}_HHMMSSFF.mp4")
    day = int(found.group(1))
    return Clip(rel, day, plain(day, clock_of(found.group(2))), length)


def clip_table(source: Path, days: int = DAYS, workers: int = 3) -> list[Clip]:
    """The clips of days 1..`days` in time order, with their durations (read from the packets, as `remember` does), `workers` files
    at a time."""
    names = [(day, name) for day in range(1, days + 1) for name in sorted(os.listdir(Path(source) / CLIP_FOLDER / f"DAY{day}"))]
    paths = [Path(source) / CLIP_FOLDER / f"DAY{day}" / name for day, name in names]
    with ThreadPoolExecutor(workers) as pool:
        lengths = list(pool.map(packet_duration, paths))
    out = []
    for path, length in zip(paths, lengths):
        if length is None:
            raise ValueError(f"{path} is not media this machine can read")
        out.append(clip_of(path.relative_to(source).as_posix(), length))
    out.sort(key=lambda c: c.t0)
    return out


def read_items(source: Path) -> list[dict]:
    """The questions of the download, one plain dict each: `id`, `category`, `audio` (needs sound), `need_name`, `last`, `query` (a
    plain time), `targets` (plain times, in order: the evidence), `question`, `options` (A to D), `answer`, `keywords`, `reason`.
    `Day1` is DAY1; a target is `time` or `time_list`. `targets` is empty for the few questions whose time holds a range or
    several moments in one string (`11153417DAY1_11181201`): they have no one moment to look at."""
    out, seen = [], set()
    for raw in json.loads((Path(source) / QA_FILE).read_text(encoding="utf-8")):
        id_ = str(raw["ID"])
        if id_ in seen:
            raise ValueError(f"the question ID {id_} is twice in {QA_FILE}")
        seen.add(id_)
        if raw["type"] not in CATEGORIES:
            raise ValueError(f"question {id_} is of the type {raw['type']!r}, not one of {', '.join(CATEGORIES)}")
        if raw["answer"] not in ("A", "B", "C", "D"):
            raise ValueError(f"question {id_}'s answer is {raw['answer']!r}, not A, B, C or D")
        target = raw["target_time"]
        times, day = target["time_list"] if "time_list" in target else [target["time"]], day_of(target["date"])
        out.append({"id": id_, "category": raw["type"], "audio": bool(raw["need_audio"]), "need_name": bool(raw["need_name"]),
                    "last": bool(raw["last_time"]),
                    "query": plain(day_of(raw["query_time"]["date"]), clock_of(raw["query_time"]["time"])),
                    "targets": sorted(plain(day, clock_of(t)) for t in times) if all(re.fullmatch(r"\d{8}", t) for t in times) else [],
                    "question": raw["question"], "options": {k: raw[f"choice_{k.lower()}"] for k in "ABCD"},
                    "answer": raw["answer"], "keywords": raw["keywords"], "reason": raw["reason"]})
    return out


def spoken(row: str) -> tuple[str, str]:
    """A transcript line as `(speaker, words)`: `Jake: Okay.` is `("Jake", "Okay.")`; a line with no speaker has the speaker ""."""
    found = SPEAKER.fullmatch(row)
    return (found.group(1).strip(), found.group(2).strip()) if found else ("", row.strip())


def read_transcripts(source: Path, days: int = DAYS) -> list[dict]:
    """The spoken lines of days 1..`days`, by their end: `start`, `end` (plain), `speaker`, `zh` (the first line of an entry,
    as written) and `en` (its second line, the English: "" for an entry of one line). A SRT file is an hour, its times offsets from
    the hour in its name; a line with no `Speaker:` has the speaker ""."""
    out = []
    for day in range(1, days + 1):
        folder = Path(source) / SRT_FOLDER / f"DAY{day}"
        for name in sorted(os.listdir(folder)):
            found = SRT_NAME.fullmatch(name)
            if found is None or int(found.group(1)) != day:
                raise ValueError(f"{folder / name} is not a transcript of DAY{day}: {PERSON}_DAY{day}_HHMMSSFF.srt")
            base = plain(day, clock_of(found.group(2)))
            text = (folder / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
            for block in re.split(r"\n\s*\n", text.strip()):
                rows = block.split("\n")
                when = SRT_TIME.fullmatch(rows[1].strip()) if len(rows) in (3, 4) and rows[0].strip().isdigit() else None
                if when is None:
                    raise ValueError(f"{folder / name}: an entry that is not a number, a time range and one or two lines: {block[:80]!r}")
                h0, m0, s0, ms0, h1, m1, s1, ms1 = (int(v) for v in when.groups())
                speaker, zh = spoken(rows[2])
                en = spoken(rows[3])[1] if len(rows) == 4 else ""
                out.append({"start": base + h0 * 3600 + m0 * 60 + s0 + ms0 / 1000, "end": base + h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                            "speaker": speaker, "zh": zh, "en": en})
    out.sort(key=lambda line: (line["end"], line["start"]))
    return out
