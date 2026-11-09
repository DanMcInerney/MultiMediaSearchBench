"""The source of A6: MIT 6.0002, Introduction to Computational Thinking and Data Science (Fall 2016), on the Internet Archive (`MIT6.0002F16`):
the 15 lectures of the course as MP4s of 360p video with sound, and the SRT of each lecture's words beside it (cue-level times, the captions
of MIT OpenCourseWare, with the speaker's name where the caption gives one). Each lecture opens with the same OCW credit, read over a blank
screen, which no question asks about.

Each MP4 carries the day of its lecture in its own `date` tag (Mondays and Wednesdays, 24 October to 14 December 2016; lecture 9's tag is
Sunday 20 November, and a class that falls on no Monday or Wednesday is taken as the day after). The time of day is not in the source: A6
puts every lecture at 14:05 and up to a few minutes later, as a student's phone would have begun a recording. The schedule is the same for
every seed (only the questions are drawn with it). Only the sound of a lecture goes into the dump (`speech_files`)."""

from __future__ import annotations

import functools
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.scenarios.realdump import sources_root, speech_files
from mmsb.scenarios.realdump.speech import Talk
from mmsb.scenarios.realdump.speech_files import Recording

SOURCE = sources_root() / "ocw-6-0002"
LECTURES = tuple(range(1, 16))
CLASS_DAYS = (date(2016, 10, 24), date(2016, 10, 26), date(2016, 10, 31), date(2016, 11, 2), date(2016, 11, 7), date(2016, 11, 9),
              date(2016, 11, 14), date(2016, 11, 16), date(2016, 11, 21), date(2016, 11, 28), date(2016, 11, 30), date(2016, 12, 5),
              date(2016, 12, 7), date(2016, 12, 12), date(2016, 12, 14))  # lecture n on CLASS_DAYS[n - 1]: the day of its MP4's own date tag
CLASS_STARTS = time(14, 5)
FOLDER = "Recordings"


def stem_of(number: int) -> str:
    return f"MIT6_0002F16_lec{number:02d}_300k"


def class_day(number: int) -> date:
    return CLASS_DAYS[number - 1]


def day_of_tag(tag: datetime) -> date:
    """The day of the class that an MP4's date tag says: its day, or the next that is a Monday or a Wednesday (the course's days)."""
    day = tag.date()
    while day.weekday() not in (0, 2):
        day += timedelta(days=1)
    return day


def start_of(number: int) -> datetime:
    """When the phone began recording lecture `number`: its class day at CLASS_STARTS and up to a few minutes more."""
    return datetime.combine(class_day(number), CLASS_STARTS) + timedelta(seconds=(number * 71) % 170)


def path_of(start: datetime) -> str:
    """`Recordings/Recording_20161024_140611.m4a`: the moment the recording began, as a phone's recorder names it."""
    return f"{FOLDER}/Recording_{start:%Y%m%d_%H%M%S}.m4a"


@functools.lru_cache(maxsize=2)
def read_term(source: str, numbers: tuple[int, ...]) -> tuple[tuple[Recording, ...], tuple[float, ...]]:
    """The lectures `numbers` of the source in the order of the term, with what was said in each and the seconds its sound plays. The source
    must have exactly these lectures' MP4s and SRTs, each MP4 dated to the day CLASS_DAYS gives it."""
    stems = {stem_of(n) for n in numbers}
    by_stem = {r.stem: r for r in speech_files.read_recordings(Path(source), stems.__contains__)}
    if set(by_stem) != stems:
        raise FileNotFoundError(f"{source} does not hold the {len(stems)} lectures A6 is made of ({sorted(stems - set(by_stem))[:3]} are missing)")
    recs = tuple(by_stem[stem_of(n)] for n in numbers)
    for n, rec in zip(numbers, recs, strict=True):
        if (day := day_of_tag(speech_files.tagged_time(rec.video))) != class_day(n):
            raise ValueError(f"{rec.video} is dated to {day}, not the {class_day(n)} A6 gives lecture {n}")
    return recs, tuple(speech_files.audio_seconds(r.video) for r in recs)


def talks_of(numbers: tuple[int, ...], recs: tuple[Recording, ...], lengths: tuple[float, ...]) -> tuple[Talk, ...]:
    return tuple(Talk(path_of(start_of(n)), start_of(n), length, rec.cues) for n, rec, length in zip(numbers, recs, lengths, strict=True))
