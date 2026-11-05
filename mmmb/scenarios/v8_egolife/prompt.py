"""What an agent is asked in a V8 run (moved from `EgoRun.prompt` of egolife.py of this repository at bb498a6): the scenario's time (the
question's own), how the recordings are reached (the only difference between the configurations), the question with its options, and how to
answer. A system's word for itself is `served.holder`, so this knows no system.

- no tools: `You cannot see my recordings.`;
- the agent's own tools alone: the folder the recordings are in, how they are named, and that ffmpeg is on PATH;
- a system's tools alone: `{holder} has recorded these sources of mine from day 1 to now: ...`, and the system's clock note when it has one
  (V8's recorded text has it: `mm's clock reads a later date than now ...`);
- both: the system's sentence, then the files'. (No recorded run has this; tests/test_v8.py pins the text.)"""

from __future__ import annotations

from datetime import date, datetime

from mmmb.core.answer import answer_request
from mmmb.core.question import Question
from mmmb.core.scenario import Access
from mmmb.scenarios.v8_egolife.plan import TEXT, VIDEO

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def date_text(day: date) -> str:
    """`Tuesday 28 October`, from fixed English names (not `strftime`, which follows the locale)."""
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def prompt(question: Question, access: Access) -> str:
    """The prompt for `question` asked through `access`; it ends by asking for the answer in the access's format (`answer_request`)."""
    t = datetime.fromisoformat(question.now)
    first = date.fromisoformat(str(question.tags["first_day"]))
    head = (f"It is now {date_text(t.date())} {t.year}, {t:%H:%M} (day {question.tags['day']} of these recordings; day 1 was "
            f"{date_text(first)}). I am Jake.")
    served = access.served
    how = []
    if served.server is not None:
        if served.holder is None:
            raise ValueError(f"the server {served.server.name!r} has no word for itself for the prompt")
        how.append(f"{served.holder} has recorded these sources of mine from day 1 to now: {VIDEO} (the video of the camera on my glasses, "
                   f"30 s clips at 20 fps) and {TEXT} (what was said around me: a record per spoken line, with its speaker and the "
                   "words in English and Chinese)." + (f" {served.clock_note}" if served.clock_note else ""))
    if access.own_tools:
        how.append(f"The recordings of my days so far are in {access.data}: glasses/<date>/ holds the video of the camera on my glasses as "
                   "30 s clips (1408x1408, 20 fps), each named by the time it starts; transcript/<date>.jsonl holds what was said "
                   "around me, one JSON line per spoken line with its start, end, speaker and the words in English and Chinese. "
                   "Your working folder is yours to write in. ffmpeg is on PATH (there is no ffprobe).")
    elif served.server is None:
        how.append("You cannot see my recordings.")
    if question.options is None:
        raise ValueError(f"{question.qid} has no options: a V8 question is a choice")
    options = "\n".join(f"{k}) {v}" for k, v in question.options.items())
    end = f"If what you can find does not settle it, give your best guess. {answer_request(access.answer_format, None, '<A, B, C or D>')}"
    return f"{head} {' '.join(how)}\n\n{question.question}\n{options}\n\n{end}"
