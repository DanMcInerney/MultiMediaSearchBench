"""The plan of a V8 run (moved from egolife.py of this repository at bb498a6): a pure function of the download, the seed, the days, the clip
table and the oracles file. One `File` for each clip (kind `clip`, spec `{"src", "length"}`), the transcript as logs (a date to its lines),
and a question for each chosen item. The plan's dicts are what `plan_hash` and `files_hash` cover, so their keys do not change; questions.py
projects them onto the question file.

Times: a plain time is turned into a local epoch with `local`, which needs the days; the offset of the local zone must not change during
them (`check_days`), else elapsed time across that night would be an hour other in a system that keeps offsets than in the files, which
keep naive local times.

`oracles.json` holds the retrieval question of each of the 50 questions the recorded run chose, written by hand by ORACLE_RULES. Its text is part
of the plan, so a change to it changes the plan hash."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from mmmb.core.agents.common import reached
from mmmb.scenarios.v8_egolife.choose import PER_AGE, SPEECH, horizon, select
from mmmb.scenarios.v8_egolife.source import DAY_S, DAYS, Clip, read_items, read_transcripts

SCALE = "egolife"  # what the plan and a run call the size of this scenario: there is one
TOL_S, PAD_S = 30.0, 300.0  # a sighting within this of a target counts; an oracle's range is a target +- this, cut at H

VIDEO, TEXT = "glasses", "transcript"  # the sources of the dump: the clips, and the transcript's day files
TEXT_ASK = "Does the text talk about "  # how a speech question's oracle starts (rule 6): its words are no borrowing from an option

PEOPLE = ("Alice", "Choiszt", "Jake", "Katrina", "Lucia", "Nicous", "Shure", "Tasha", "Violet")  # who the transcript names
ORACLES = Path(__file__).with_name("oracles.json")
ORACLE_VERSION = 1
# the words mm's ask refuses in a question (src/multimem/scripts/ask.py at fe0ece5: TIME_WORDS, JOINED): rules 3 and 5 of ORACLE_RULES.
# tests/mm asserts that they are still Multimem's.
TIME_WORDS = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|this (morning|afternoon|evening|week|month)|last (night|week|month|year)|"
    r"(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|at \d{1,2}(:\d\d)?\s?(am|pm)?|\d{1,2}\s?(am|pm)|"
    r"\d{4}-\d\d-\d\d|(january|february|march|april|may|june|july|august|september|october|november|december) \d)\b",
    re.IGNORECASE)
JOINED = re.compile(r"\b(and|or)\b", re.IGNORECASE)

ORACLE_RULES = """\
1. Write it from the item's `question` and `keywords` only. Read `reason` to learn what the evidence looks like (an action, an
   object), never who, where or which.
2. Never use the words of the right option, or of any option that alone would settle the answer.
3. Ask about one thing that one window can show, in the present tense: "Is someone using a screwdriver?", "Is a black marker lying
   on a table?".
4. For `glasses`, name no person: d1 cannot tell people by name, and the agent tells who from a look and the transcript.
5. No time words: the range carries the time.
6. A `need_audio` item uses `transcript`: "Does the text talk about X?", with X the question's subject, not its answer.
7. Never the trigger: it is at the query time, after H."""


@dataclass(frozen=True)
class File:
    path: str  # relative to the dump, forward slashes; its first part is the source
    source: str
    t: float  # local epoch: when it starts
    name: str  # the name a system that wants one is given: the path without its source, its slashes spaces
    kind: str  # "clip"
    spec: dict  # {"src": the clip in the download, "length": seconds}


@dataclass(frozen=True)
class Plan:
    seed: int
    scale: str
    days: tuple[date, ...]
    files: tuple[File, ...]  # oldest first
    logs: dict[str, tuple[dict, ...]]  # date -> its transcript lines ({"start", "end", "speaker", "en", "zh"}), in time order
    questions: tuple[dict, ...]


def day_date(days: Sequence[date], day: int) -> date:
    """The date of day number `day` (1 is the first of `days`)."""
    if not 1 <= day <= len(days):
        raise ValueError(f"day {day} is not one of the run's {len(days)} days")
    return days[day - 1]


def local(days: Sequence[date], t: float) -> float:
    """The local epoch of plain time `t`, day number `d` being `days[d - 1]` (hundredths kept: `datetime` takes microseconds)."""
    cs = round(t * 100)
    day, cs = divmod(cs, DAY_S * 100)
    seconds, hundredths = divmod(cs, 100)
    d = day_date(days, day)
    return datetime(d.year, d.month, d.day, seconds // 3600, seconds // 60 % 60, seconds % 60, hundredths * 10000).timestamp()


def iso_of(days: Sequence[date], t: float) -> str:
    """Plain time `t` as a local ISO time to the millisecond, with no zone (mm reads it as local)."""
    day, ms = divmod(round(t * 1000), DAY_S * 1000)
    return (datetime.combine(day_date(days, day), datetime.min.time()) + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds")


def clip_path(days: Sequence[date], c: Clip) -> str:
    """Where the run's data holds a clip, and the name `remember` gives it: neutral (no person, dataset or day number)."""
    s = round(c.t0 * 100) % (DAY_S * 100) // 100
    return f"{VIDEO}/{day_date(days, c.day).isoformat()}/{s // 3600:02d}.{s // 60 % 60:02d}.{s % 60:02d}.mp4"


def clip_end(days: Sequence[date], c: Clip) -> float:
    """A clip's end as mm stores it: the local epoch it is remembered at, plus its length (the same sum, so H compares exactly)."""
    return local(days, c.t0) + c.length


def log_lines(days: Sequence[date], lines: Sequence[dict]) -> dict[str, tuple[dict, ...]]:
    """The transcript as the run holds it: a date to its lines, each `start`, `end` (local ISO), `speaker`, `en` and `zh`."""
    out: dict[str, list[dict]] = {d.isoformat(): [] for d in days}
    for line in lines:
        day = int(line["end"] // DAY_S)
        out[day_date(days, day).isoformat()].append({"start": iso_of(days, line["start"]), "end": iso_of(days, line["end"]),
                                               "speaker": line["speaker"], "en": line["en"], "zh": line["zh"]})
    return {d: tuple(v) for d, v in out.items()}


def load_oracles(path: Path | None = None) -> dict[str, dict]:
    """The oracles file: question ID to `{"source", "question"}`."""
    data = json.loads(Path(path or ORACLES).read_text(encoding="utf-8"))
    if data.get("version") != ORACLE_VERSION:
        raise ValueError(f"{path or ORACLES} is version {data.get('version')}, this is {ORACLE_VERSION}")
    return data["items"]


def check_oracle(item: dict, oracle: dict) -> None:
    """The rules of ORACLE_RULES that a machine can read, applied to the oracle of `item`: its source is the question's, it asks
    one thing, with no time word, no 'and' or 'or', no option's words (a word of 4 letters or more of an option is allowed only
    when the question has it too) and (for the glasses) no person. A `ValueError` names the ID."""
    id_, text = item["id"], oracle.get("question", "")
    wanted = TEXT if item["audio"] else VIDEO
    if oracle.get("source") != wanted:
        raise ValueError(f"the oracle of question {id_} has the source {oracle.get('source')!r}, not {wanted!r}")
    if not isinstance(text, str) or not text.endswith("?"):
        raise ValueError(f"the oracle of question {id_} is not a question: {text!r}")
    if TIME_WORDS.search(text):
        raise ValueError(f"the oracle of question {id_} has a time word (the range carries the time): {text!r}")
    if JOINED.search(text):
        raise ValueError(f"the oracle of question {id_} joins two things with 'and' or 'or': {text!r}")
    if not item["audio"] and (named := [p for p in PEOPLE if re.search(rf"\b{p}\b", text, re.I)]):
        raise ValueError(f"the oracle of question {id_} names {named[0]} (the glasses cannot tell people by name): {text!r}")
    if leaked := [o for o in item["options"].values() if o.strip() and re.search(rf"\b{re.escape(o.strip())}\b", text, re.I)]:
        raise ValueError(f"the oracle of question {id_} holds an option's words ({leaked[0]!r}): {text!r}")
    asked = set(re.findall(r"[a-z]+", item["question"].lower()))
    if borrowed := sorted({w for o in item["options"].values() for w in re.findall(r"[a-z]{4,}", o.lower())}
                          & (set(re.findall(r"[a-z]+", text.removeprefix(TEXT_ASK).lower())) - asked)):
        raise ValueError(f"the oracle of question {id_} uses {borrowed[0]!r}, a word of an option that the question does not use: "
                         f"{text!r}")


def plan(source: Path, seed: int, days: Sequence[date], clips: Sequence[Clip], oracles: dict[str, dict], per_age: int = PER_AGE,
         speech: int = SPEECH) -> Plan:
    """The run's plan, a pure function of the download, the seed, the days, the clip table and the oracles: one `File` per clip
    (kind `clip`, spec `{"src", "length"}`), the transcript as logs (a date to its lines), and a question for each chosen item
    (the keys questions.py reads, and `options`, `age`, `oracle`, `oracle_source`, `pilot`). A chosen question with no oracle is
    refused by its ID."""
    days = tuple(days)
    items, lines = read_items(source), read_transcripts(source, len(days))
    chosen = select(items, clips, lines, seed, per_age, speech, len(days))
    if missing := [i["id"] for i in chosen if i["id"] not in oracles]:
        raise ValueError(f"no oracle for question {', '.join(missing)}: write it in {ORACLES.name} by ORACLE_RULES")
    for item in chosen:
        check_oracle(item, oracles[item["id"]])
    paths = [clip_path(days, c) for c in clips]
    if dup := [p for p, n in Counter(paths).items() if n > 1]:
        raise ValueError(f"two clips would be named {dup[0]}")
    files = tuple(File(p, VIDEO, local(days, c.t0), p.split("/", 1)[1].replace("/", " "), "clip",
                       {"src": c.rel, "length": c.length}) for p, c in zip(paths, clips))
    questions = []
    for item in chosen:
        top = horizon(clips, item["query"])
        oracle = oracles[item["id"]]
        questions.append({
            "qid": f"ego-{int(item['id']):03d}", "id": item["id"], "type": "records" if item["audio"] else "video",
            "template": item["category"], "level": item["age"], "last": item["last"], "absent": False, "need_name": item["need_name"],
            "source": TEXT if item["audio"] else VIDEO, "day": int(item["query"] // DAY_S), "query": local(days, item["query"]),
            "since": local(days, clips[0].t0), "until": clip_end(days, top), "question": item["question"], "options": item["options"],
            "grade": {"kind": "choice", "truth": item["answer"], "ignore": []},
            "truth": {"moments": [[local(days, t)] * 2 for t in item["targets"]], "others": []},
            "oracle": oracle["question"], "oracle_source": oracle["source"], "pilot": item["pilot"]})
    return Plan(seed, SCALE, days, files, log_lines(days, lines), tuple(questions))


def oracle_ranges(q: dict) -> list[tuple[float, float]]:
    """What the decider tier asks over: each target +- PAD_S, cut at the horizon, overlapping ranges merged."""
    spans = sorted((m - PAD_S, min(m + PAD_S, q["until"])) for m, _ in q["truth"]["moments"])
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def utc_offset(day: date) -> timedelta | None:
    """The local UTC offset at the start of `day`."""
    return datetime(day.year, day.month, day.day).astimezone().utcoffset()


def crosses_a_change(days: Sequence[date], offset=None) -> bool:
    """Whether the local UTC offset changes between the start of the first of `days` and the end of the last (the clocks change)."""
    return len({(offset or utc_offset)(d) for d in (*days, days[-1] + timedelta(days=1))}) > 1


def default_days(today: date, n: int = DAYS, offset=None) -> tuple[date, ...]:
    """The latest `n` consecutive days before `today` that cross no change of the local UTC offset: elapsed time across such a
    night is an hour different in mm (which keeps offsets) and in the files arm (which keeps naive local times)."""
    last = today - timedelta(days=1)
    while crosses_a_change(days := tuple(last - timedelta(days=n - 1 - i) for i in range(n)), offset):
        last -= timedelta(days=1)
    return days


def check_days(days: Sequence[date], offset=None) -> None:
    """`days` are consecutive, oldest first, and cross no change of the local UTC offset."""
    if any(b - a != timedelta(days=1) for a, b in zip(days, days[1:])):
        raise ValueError(f"the days are consecutive, oldest first: {', '.join(d.isoformat() for d in days)}")
    if crosses_a_change(days, offset):
        raise ValueError(f"{days[0]} to {days[-1]} cross a change of the local UTC offset (the clocks change): elapsed time across "
                         "that night would differ by an hour between mm and the files arm; use days that do not")


def check_transcript(logs: dict[str, Sequence[dict]], deny: Sequence[str]) -> None:
    """No transcript line holds a word that an agent with its own tools is ended for printing (`deny`, lower case with forward
    slashes; the words that are not command-only count): a line the agent reads and prints would end its exec as `escaped`."""
    for day, rows in logs.items():
        for line in rows:
            if hit := reached(line, tuple(deny), output=True):
                raise ValueError(f"the transcript line ending {line['end']} holds {hit!r}, which the files arm ends an exec for "
                                 f"printing: {line['en'] or line['zh']!r}")
