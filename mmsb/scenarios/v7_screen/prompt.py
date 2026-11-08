"""What an agent is asked in the V7 week (moved from `Run.prompt` of bench.py of the repository at bb498a6): the time, how the sources
are reached (the only difference between the configurations), the question and how to answer. A system's word for itself is
`served.holder`, so this knows no system.

- a system's tools alone: `{holder} has recorded these sources of mine {week}.`, ending `Answer from what {holder} holds.`;
- the agent's own tools alone: the folder the files are in, ending `Answer from these files.`;
- both: the system's sentence, the files' sentence, and `Answer from what {holder} holds or from these files.`;
- the system's clock note, when it has one, ends its sentence (V8's recorded text has it where this has);
- the last sentence asks for the answer in the access's format (`answer_request`), and says "unknown" is the answer when what the agent
  has (the system, the files, or both) does not hold it.

The time is the question's `now`, not the clock; the days of the week are the question's `first_day` and `last_day`."""

from __future__ import annotations

from datetime import date, datetime

from mmsb.core.answer import answer_request
from mmsb.core.question import Question
from mmsb.core.scenario import Access
from mmsb.scenarios.v7_screen.plan import MONTHS, SOURCES, WEEKDAYS, date_text


def prompt(question: Question, access: Access) -> str:
    """The prompt for `question` asked through `access`."""
    t = datetime.fromisoformat(question.now)
    stamp = f"{WEEKDAYS[t.weekday()]} {t.day:02d} {MONTHS[t.month - 1]} {t.year}, {t:%H:%M}"
    first, last = (date.fromisoformat(str(question.tags[key])) for key in ("first_day", "last_day"))
    week = f"from {date_text(first)} to {date_text(last)}"
    sources = ", ".join(f"{name} ({what})" for name, what in SOURCES.items())
    served = access.served
    parts = []
    if served.server is not None:
        holder = served.holder
        parts.append(f"{holder} has recorded these sources of mine {week}." + (f" {served.clock_note}" if served.clock_note else ""))
        answer = f"Answer from what {holder} holds or from these files." if access.own_tools else f"Answer from what {holder} holds."
        end = answer_request(access.answer_format, f"{holder} and these files", plural=True) if access.own_tools else answer_request(
            access.answer_format, holder)
    elif access.own_tools:
        answer = "Answer from these files."
        end = answer_request(access.answer_format, "the files", plural=True)
    else:
        raise ValueError("a V7 question is asked through a system's tools or the agent's own, not neither")
    if access.own_tools:
        parts.append(f"The files of these sources, {week}, are in {access.data}, one folder per source; each file is "
                     "named by when it was recorded, and the logs are JSON lines with a ts field. Your working folder is yours "
                     "to write in. ffmpeg is on PATH (there is no ffprobe).")
    return f"It is now {stamp}. {' '.join(parts)} The sources are: {sources}. {question.question} {answer} {end}"
