"""A7, the podcast backlog: the episodes of an interview podcast that one listener has kept, one MP3 each, in the order they came out (Chalk Radio
of MIT OpenCourseWare: a teacher at MIT talks about teaching, seasons 1 to 5 and 7, 48 episodes, 18 hours). The files are numbered, and each is
timed to the moment its episode came out. The questions ask when they first mentioned something, in which episode, in how many, how long an
episode was, who the guest was, what came before what, and what was said, and of things never said (questions.py). Every answer is from the
transcripts the episodes come with, which name the speaker at each change of speaker; the scenario is the names of a `Scenario`, as functions
(DESIGN 5), over the shared builder (realdump, speech).

    recipe = a7_podcast.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = a7_podcast.build(recipe, folder, 3)  # the episodes' sound written as MP3 with their times, and the questions

Scales: `tiny`, the first three episodes (0.03 GB, built in 11 s, for the tests and a smoke run); `standard`, all 48
(0.53 GB, 3 minutes, 19 February 2020 to 5 March 2026)."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf
from mmsb.scenarios import realdump
from mmsb.scenarios.a7_podcast.questions import CODE, GUESTS, HOST, NEVER, SOURCE_NAME, TERMS, questions_of
from mmsb.scenarios.a7_podcast.source import EPISODES, FIRST_DAY, LAST_DAY, SOURCE, read_pile, talks_of
from mmsb.scenarios.realdump import File, report, speech, speech_files  # noqa: F401  (`report`: the Scenario's)

name = "a7_podcast"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Podcast backlog", "audio", "fun", "the episodes of one interview podcast, kept in the order they came out")
filing = speech.FILING

SCALES = {"tiny": 3, "standard": EPISODES}  # the episodes at each scale: the first so many
TINY_LAST_DAY = date(2020, 3, 18)  # the day the third episode came out
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
WHAT = "the episodes of a podcast I follow"
LAYOUT = ("Podcasts holds one MP3 for each episode, episode_01.mp3 onwards in the order the episodes came out, and a file's modified time is the "
          "moment its episode came out.")
SOURCE_WORDS = ("ocw", "opencourseware", "chalk", "chalkradio", "hansen", "archive", "mit")  # the source's names, which no file of the dump may hold
COMMAND_WORDS = ("chalk-radio", "chalk_radio", "chalkradio", "archive.org", "opencourseware")  # and which no command may


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days from the
    first episode's to the last and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(FIRST_DAY, LAST_DAY if scale == "standard" else TINY_LAST_DAY)
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


@functools.lru_cache(maxsize=4)
def planned(source: str, scale: str, seed: int) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    episodes = read_pile(source)[:SCALES[scale]]
    talks = talks_of(episodes)
    files = tuple(File(t.path, SOURCE_NAME, realdump.local_t(t.start), "episode", "audio", {"src": f"{e.rec.stem}.mp4"}, length=t.length)
                  for t, e in zip(talks, episodes, strict=True))
    return files, questions_of(talks, [e.rec.stem for e in episodes], scale, seed)


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    return planned(str(recipe["source"]), str(recipe["scale"]), int(recipe["seed"]))


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the episodes into `out` (new or empty): the sound of each as mono MP3 with none of the source's metadata, and its time; `workers`
    at once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        speech_files.extract_sound(source / str(f.spec["src"]), path)
        speech_files.check_plays(path, float(f.length), speech_files.audio_seconds)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no episode's name or metadata names a term of the questions, a guest or the podcast."""
    guests = [w for g in (*GUESTS, HOST) for w in g.split()]
    return realdump.probe(dump, out, [*(t.search for t in (*TERMS, *NEVER)), *guests, *SOURCE_WORDS], speech_files.HEADERS)


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The episodes themselves: nothing is cut (every question is asked after the last episode)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
