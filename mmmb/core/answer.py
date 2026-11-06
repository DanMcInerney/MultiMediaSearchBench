"""The final answer of an exec, in the two formats a run can ask for (DESIGN 6.2), and the sentence a prompt ends with that asks for it.

- `line`: the format of the recorded runs. The prompt asks for one line `ANSWER: <answer>`; the answer is what follows the last such
  line of the final message, and there is no evidence.
- `json`: the prompt asks for one JSON object `{"answer": "...", "evidence": [...]}`. The answer is that of the last complete JSON object
  of the final message (a code fence around it is just text), which needs a string `answer` and a list `evidence` of citations. Anything
  else is a failed answer: `answer` is None and `error` says why. It is never read as a `line` answer.

A citation names the moment that shows the answer in one of two ways, because an agent that has only a system's tools never sees a file
path (DESIGN 6.2 corrected): `{"file": <path in the dump>, "t0", "t1"}` with seconds into the file, a clock time (`H:MM[:SS[.f]]`) or
an ISO time (the instant; with no offset, the dump's wall clock: a photo's time is written so) for the two times, or `{"source": <name>, "t0", "t1"}` with
ISO times that carry an offset. `parse_final` is the one place a final message is
read; `core/grounded.py` places a citation in time and scores it."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

AnswerFormat = Literal["line", "json"]
FORMATS: tuple[AnswerFormat, ...] = ("line", "json")
DEFAULT_FORMAT: AnswerFormat = "json"
CLOCK = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?")
ISO_START = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Citation:
    file: str | None  # a path in the dump, as its manifest has it (or the one the agent read it by, ending in it)
    source: str | None  # a source's name; exactly one of file and source is set
    t0: str | float  # `file`: seconds into the file (float) or a clock time (str); `source`: an ISO time with an offset (str)
    t1: str | float

    def to_json(self) -> dict:
        """The citation as the agent wrote it, which is what a result keeps."""
        return {**({"file": self.file} if self.file is not None else {"source": self.source}), "t0": self.t0, "t1": self.t1}


@dataclass(frozen=True)
class FinalAnswer:
    answer: str | None  # None: no answer was given, or the message could not be read (`error` then says why)
    evidence: tuple[Citation, ...]
    error: str | None  # why a `json` message is a failed answer; always None for `line`


def answer_of(text: str) -> str | None:
    """The text after the last `ANSWER:` line of a message (the `line` format)."""
    found = re.findall(r"^\W*ANSWER:\s*(.+?)\s*$", text, re.I | re.M)
    return found[-1].strip(" *`") if found else None


def parse_final(text: str, fmt: AnswerFormat) -> FinalAnswer:
    """The answer and the evidence in the final message `text` of an exec, read in the format the prompt asked for."""
    if fmt == "line":
        return FinalAnswer(answer_of(text), (), None)
    if fmt != "json":
        raise ValueError(f"the answer format is one of {', '.join(FORMATS)}, not {fmt!r}")
    if not text.strip():
        return failed("the final message is empty")
    found = last_object(text)
    if found is None:
        return failed("the final message holds no complete JSON object")
    answer, evidence = found.get("answer"), found.get("evidence")
    if not isinstance(answer, str):
        return failed(f'"answer" must be a string, not {shown(answer)}')
    if not isinstance(evidence, list):
        return failed(f'"evidence" must be a list, not {shown(evidence)}')
    try:
        return FinalAnswer(answer, tuple(citation(item, n) for n, item in enumerate(evidence)), None)
    except ValueError as error:
        return failed(str(error))


def failed(error: str) -> FinalAnswer:
    return FinalAnswer(None, (), error)


def shown(value: object) -> str:
    """`value` for a message, cut short (None is a key the object lacks, or null)."""
    return text if len(text := repr(value)) <= 40 else text[:37] + "..."


def last_object(text: str) -> dict | None:
    """The last complete JSON object in `text`, at any depth of prose: each `{` is tried as the start of one, and a tried object is
    skipped over whole (an object inside another is no object of its own). A `{` that starts no complete object is text."""
    decoder, found, at = json.JSONDecoder(), None, 0
    while (at := text.find("{", at)) != -1:
        try:
            found, at = decoder.raw_decode(text, at)
        except (ValueError, RecursionError):
            at += 1
    return found


def citation(item: object, n: int) -> Citation:
    """The citation `evidence[n]`; a ValueError names what is wrong with it."""
    where = f"evidence[{n}]"
    if not isinstance(item, dict):
        raise ValueError(f"{where} must be an object, not {shown(item)}")
    file, source = item.get("file"), item.get("source")
    if (file is None) == (source is None):
        raise ValueError(f'{where} must name a "file" or a "source", and not both')
    key, name = ("file", file) if file is not None else ("source", source)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'{where}: "{key}" must be a non-empty string, not {shown(name)}')
    t0, t1 = item.get("t0"), item.get("t1")
    if file is not None:
        if isinstance(t0, str) and isinstance(t1, str):
            if (iso := is_iso(t0)) != is_iso(t1):
                raise ValueError(f'{where}: "t0" and "t1" of a file must both be clock times or both ISO times, not {shown(t0)} and {shown(t1)}')
            if iso:  # an instant, as a source's, or with no offset the dump's wall clock (a photo's time is a date and a time)
                start = iso_epoch(t0, f'{where}: "t0"', local=True)
                if iso_epoch(t1, f'{where}: "t1"', local=True) < start:
                    raise ValueError(f'{where}: "t1" ({t1}) is before "t0" ({t0})')
            else:
                clock_seconds(t0, f'{where}: "t0"'), clock_seconds(t1, f'{where}: "t1"')
        elif is_offset(t0) and is_offset(t1):
            if t1 < t0:
                raise ValueError(f'{where}: "t1" ({t1}) is before "t0" ({t0})')
            t0, t1 = float(t0), float(t1)
        else:
            raise ValueError(f'{where}: "t0" and "t1" of a file must both be seconds into it (numbers, 0 or more) or both clock times '
                             f"(H:MM:SS), not {shown(t0)} and {shown(t1)}")
        return Citation(name.strip(), None, t0.strip() if isinstance(t0, str) else t0, t1.strip() if isinstance(t1, str) else t1)
    if not (isinstance(t0, str) and isinstance(t1, str)):
        raise ValueError(f'{where}: "t0" and "t1" of a source must be ISO times with an offset (strings), not {shown(t0)} and {shown(t1)}')
    start, end = iso_epoch(t0, f'{where}: "t0"'), iso_epoch(t1, f'{where}: "t1"')
    if end < start:
        raise ValueError(f'{where}: "t1" ({t1}) is before "t0" ({t0})')
    return Citation(None, name.strip(), t0.strip(), t1.strip())


def is_iso(text: str) -> bool:
    """Whether a time written as a string is an ISO date and time (it starts with a date), and not a clock time."""
    return ISO_START.match(text.strip()) is not None


def is_offset(value: object) -> bool:
    """Whether `value` is seconds into a file: a number (not a bool) that is finite and not negative."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def clock_seconds(text: str, where: str = "a clock time") -> float:
    """Seconds after midnight of the clock time `H:MM[:SS[.f]]`; a ValueError saying `where` when it is no clock time."""
    m = CLOCK.fullmatch(text.strip())
    hours, minutes, seconds = (int(m[1]), int(m[2]), float(m[3] or 0)) if m else (99, 0, 0.0)
    if hours > 23 or minutes > 59 or seconds >= 60:
        raise ValueError(f"{where} must be a clock time (H:MM or H:MM:SS), not {text!r}")
    return hours * 3600 + minutes * 60 + seconds


def iso_epoch(text: str, where: str = "an ISO time", local: bool = False) -> float:
    """Epoch seconds of the ISO time `text`, which must carry an offset unless `local` (a file's time, which with no offset is the dump's
    wall clock, as a clock time is); a ValueError saying `where` when it is no such time."""
    try:
        moment = datetime.fromisoformat(text.strip())
        if moment.utcoffset() is None and not local:
            raise ValueError
        return moment.timestamp()
    except (ValueError, OverflowError):
        raise ValueError(f"{where} must be an ISO date and time with an offset (2026-11-03T21:00:00-05:00), not {text!r}") from None


# -- what a prompt ends with ------------------------------------------------------------------------------------------------------

CITATIONS = ('"evidence" lists each moment that shows the answer, as {"source": "<source>", "t0": "<start as an ISO time with its offset>", '
             '"t1": "<end as an ISO time with its offset>"}, or, for a file you read, as {"file": "<its path in the folder>", "t0": '
             '<seconds into the file, or a clock time>, "t1": <the same for the end>}')


def answer_request(fmt: AnswerFormat, holder: str | None, shape: str = "<answer>", plural: bool = False) -> str:
    """The sentence a prompt ends with, that asks for the answer in `fmt`: `shape` is what the answer looks like. `holder` names what
    the agent can find the answer in (`mm`, `the files`, `mm and these files`; `plural` when it takes `do` and not `does`), and the
    sentence says to answer `unknown` when it holds nothing; None for a question that always has an answer (a choice)."""
    does = "do" if plural else "does"
    if fmt == "line":
        return f"End with one line ANSWER: {shape}" + (f", or ANSWER: unknown when {holder} {does} not hold it" if holder else "") + "."
    if fmt != "json":
        raise ValueError(f"the answer format is one of {', '.join(FORMATS)}, not {fmt!r}")
    none = f'Answer "unknown" with "evidence": [] when {holder} {does} not hold it.' if holder else '"evidence" is [] when there is no moment to cite.'
    return f'End with one JSON object and nothing after it: {{"answer": "{shape}", "evidence": [...]}}. {CITATIONS}. {none}'
