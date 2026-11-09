"""The source of A7: the Chalk Radio podcast of MIT OpenCourseWare (the Internet Archive's `chalk-radio`): an interview with an MIT teacher in each
episode, the host Sarah Hansen (mostly) asking. The item has 60 MP4s of 360p video with sound, and the SRT of each, whose captions name the
speaker at each change of speaker (`SARAH HANSEN:`). Each MP4 carries the day the episode came out in its own `date` tag.

A7's pile is the numbered episodes of seasons 1 to 5 and 7 (48): not the trailer, the three special editions (a replay of other
interviews, a talk on teaching remotely, and another podcast), the promo and season 6, whose guests are learners and not teachers. Each
episode is one MP3 of its sound alone, numbered in the order the episodes came out, and its time is the moment it came out."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from mmsb.scenarios.realdump import sources_root, speech_files
from mmsb.scenarios.realdump.speech import Talk
from mmsb.scenarios.realdump.speech_files import Recording

SOURCE = sources_root() / "chalk-radio"
EPISODE = re.compile(r"S0[1-57]E(?:0[1-9]|1\d)_")  # a numbered episode of seasons 1 to 5 and 7
EPISODES = 48
FIRST_DAY, LAST_DAY = date(2020, 2, 19), date(2026, 3, 5)  # the days the first and the last episode came out (their own date tags)
FOLDER = "Podcasts"


@dataclass(frozen=True)
class Episode:
    rec: Recording
    came_out: datetime  # wall clock, no zone
    seconds: float  # its sound plays


def path_of(n: int) -> str:
    """`Podcasts/episode_07.mp3`: the episode's number in the order they came out."""
    return f"{FOLDER}/episode_{n:02d}.mp3"


@functools.lru_cache(maxsize=2)
def read_pile(source: str) -> tuple[Episode, ...]:
    """The pile's episodes in the order they came out. The source must hold exactly EPISODES of them, MP4 and SRT each, the first and the last
    out on FIRST_DAY and LAST_DAY."""
    recs = speech_files.read_recordings(Path(source), lambda stem: EPISODE.match(stem) is not None)
    if len(recs) != EPISODES:
        raise FileNotFoundError(f"{source} holds {len(recs)} numbered episodes with their captions, and A7 is made of {EPISODES}")
    episodes = sorted((Episode(r, speech_files.tagged_time(r.video), speech_files.audio_seconds(r.video)) for r in recs), key=lambda e: e.came_out)
    if (episodes[0].came_out.date(), episodes[-1].came_out.date()) != (FIRST_DAY, LAST_DAY):
        raise ValueError(f"the pile's first and last episodes came out on {episodes[0].came_out.date()} and {episodes[-1].came_out.date()}, not "
                         f"{FIRST_DAY} and {LAST_DAY}: A7's recipe says its days")
    return tuple(episodes)


def talks_of(episodes: tuple[Episode, ...]) -> tuple[Talk, ...]:
    return tuple(Talk(path_of(n), e.came_out, e.seconds, e.rec.cues) for n, e in enumerate(episodes, 1))
