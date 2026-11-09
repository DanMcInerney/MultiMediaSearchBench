"""V9, the lecture recordings: one course's term of recorded lectures, as a student has them (MIT 14.01, Principles of Microeconomics, Fall 2023:
26 MP4 files of 360p video, 20 hours, one lecturer). The files are named for the day of the class, and the course meets on Mondays and Wednesdays
through the term. The questions ask when the professor first mentioned something, in which lecture, in how many, how long a lecture was, what came
before what, and what was said, and of things never said (questions.py). Every answer is from the transcripts the lectures come with; the scenario is
the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump, speech).

    recipe = v9_lectures.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = v9_lectures.build(recipe, folder, 3)  # the lectures copied in with their times, and the questions

Scales: `tiny`, the first three lectures (about 0.1 GB, for the tests and a smoke run); `standard`, all 26 (2.9 GB, 6 September to 13 December 2023)."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import File, report, speech, speech_files  # noqa: F401  (`report`: the Scenario's)
from mmsb.scenarios.v9_lectures.questions import CODE, NEVER, SOURCE_NAME, TERMS, questions_of
from mmsb.scenarios.v9_lectures.source import LECTURES, SOURCE, class_day, read_term, talks_of

name = "v9_lectures"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Lecture recordings", "video", "school", "one course's term of recorded lectures, two a week")
filing = speech.FILING

SCALES = {"tiny": LECTURES[:3], "standard": LECTURES}  # the lectures at each scale
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
WHAT = "the recordings of my lectures this term"
LAYOUT = ("Lectures holds one MP4 for each class I recorded, lecture_<date>.mp4, named for the day it began; a file's modified time is when its "
          "recording began.")
SOURCE_WORDS = ("ocw", "opencourseware", "mit14", "1401", "14.01", "mit", "archive", "gruber", "microeconomics")  # the source's names, which no file may hold
COMMAND_WORDS = ("ocw-14-01", "mit14_01", "mit1401", "archive.org", "opencourseware")  # and which no command may


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days from the
    first lecture's to the last and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    numbers = SCALES[scale]
    days = realdump.span_days(class_day(numbers[0]), class_day(numbers[-1]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


@functools.lru_cache(maxsize=4)
def planned(source: str, scale: str, seed: int) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    numbers = SCALES[scale]
    recs, lengths = read_term(source, numbers)
    talks = talks_of(numbers, recs, lengths)
    files = tuple(File(t.path, SOURCE_NAME, realdump.local_t(t.start), "lecture", "video", {"src": f"{r.stem}.mp4"}, length=t.length)
                  for t, r in zip(talks, recs, strict=True))
    return files, questions_of(talks, [r.stem for r in recs], scale, seed)


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    return planned(str(recipe["source"]), str(recipe["scale"]), int(recipe["seed"]))


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the lectures into `out` (new or empty): each MP4 copied by stream copy with none of its metadata, and its time; `workers` at
    once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        speech_files.copy_video(source / str(f.spec["src"]), path)
        speech_files.check_plays(path, float(f.length), speech_files.video_seconds)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no lecture's name or metadata names a term of the questions, or the course."""
    return realdump.probe(dump, out, [*(t.search for t in (*TERMS, *NEVER)), *SOURCE_WORDS], speech_files.HEADERS)


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The lectures themselves: nothing is cut (every question is asked after the last lecture)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
