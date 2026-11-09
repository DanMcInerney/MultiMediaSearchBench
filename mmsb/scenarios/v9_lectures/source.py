"""The source of V9: MIT 14.01, Principles of Microeconomics (Fall 2023), on the Internet Archive (`mit14_01s23`): the 26 lectures of the course as MP4s
of 360p video with sound (20 hours, one lecturer), and the SRT of each lecture's words beside it (cue-level times, the captions of MIT
OpenCourseWare, with the speaker's name where the caption gives one).

Each MP4 carries the day of its lecture in its own `date` tag (every one a Monday or a Wednesday, 6 September to 13 December 2023). The time of
day is not in the source: V9 puts every lecture at 11:05 and up to a few minutes later, as a student's laptop would have begun a recording. The
schedule is the same for every seed (only the questions are drawn with it)."""

from __future__ import annotations

import functools
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.scenarios.realdump import sources_root, speech_files
from mmsb.scenarios.realdump.speech import Talk
from mmsb.scenarios.realdump.speech_files import Recording

SOURCE = sources_root() / "ocw-14-01"
LECTURES = tuple(range(1, 27))
CLASS_DAYS = (date(2023, 9, 6), date(2023, 9, 11), date(2023, 9, 13), date(2023, 9, 18), date(2023, 9, 20), date(2023, 9, 25), date(2023, 9, 27),
              date(2023, 10, 2), date(2023, 10, 4), date(2023, 10, 11), date(2023, 10, 16), date(2023, 10, 18), date(2023, 10, 23), date(2023, 10, 25),
              date(2023, 10, 30), date(2023, 11, 1), date(2023, 11, 6), date(2023, 11, 8), date(2023, 11, 13), date(2023, 11, 15), date(2023, 11, 27),
              date(2023, 11, 29), date(2023, 12, 4), date(2023, 12, 6), date(2023, 12, 11), date(2023, 12, 13))
# lecture n is on CLASS_DAYS[n - 1]: the day of its MP4's own date tag (`read_term` checks it)
CLASS_STARTS = time(11, 5)
FOLDER = "Lectures"


def stem_of(number: int) -> str:
    return f"mit14_01s23_lec{number:02d}_360p"


def class_day(number: int) -> date:
    return CLASS_DAYS[number - 1]


def start_of(number: int) -> datetime:
    """When the recording of lecture `number` began: its class day at CLASS_STARTS and up to a few minutes more."""
    return datetime.combine(class_day(number), CLASS_STARTS) + timedelta(seconds=(number * 53) % 170)


def path_of(start: datetime) -> str:
    """`Lectures/lecture_2023-09-06.mp4`: the day the recording began."""
    return f"{FOLDER}/lecture_{start:%Y-%m-%d}.mp4"


@functools.lru_cache(maxsize=2)
def read_term(source: str, numbers: tuple[int, ...]) -> tuple[tuple[Recording, ...], tuple[float, ...]]:
    """The lectures `numbers` of the source in the order of the term, with what was said in each and the seconds each plays. The source must
    have exactly these lectures' MP4s and SRTs, each MP4 dated to the day CLASS_DAYS gives it."""
    stems = {stem_of(n) for n in numbers}
    by_stem = {r.stem: r for r in speech_files.read_recordings(Path(source), stems.__contains__)}
    if set(by_stem) != stems:
        raise FileNotFoundError(f"{source} does not hold the {len(stems)} lectures V9 is made of ({sorted(stems - set(by_stem))[:3]} are missing)")
    recs = tuple(by_stem[stem_of(n)] for n in numbers)
    for n, rec in zip(numbers, recs, strict=True):
        if (day := speech_files.tagged_time(rec.video).date()) != class_day(n):
            raise ValueError(f"{rec.video} is dated to {day}, not the {class_day(n)} V9 gives lecture {n}")
    return recs, tuple(speech_files.video_seconds(r.video) for r in recs)


def talks_of(numbers: tuple[int, ...], recs: tuple[Recording, ...], lengths: tuple[float, ...]) -> tuple[Talk, ...]:
    return tuple(Talk(path_of(start_of(n)), start_of(n), length, rec.cues) for n, rec, length in zip(numbers, recs, lengths, strict=True))
