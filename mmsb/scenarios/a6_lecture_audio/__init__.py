"""A6, the lecture audio: someone's phone recordings of a course, one a class through a term (MIT 6.0002, Introduction to Computational Thinking
and Data Science, Fall 2016: 15 lectures, 12 hours, sound only). The files are named for the moment the recording began, and the course meets
twice a week through the term. The questions ask when the professor first mentioned something, in which lecture, in how many, how long a lecture
was, what came before what, and what was said, and of things never said (questions.py). Every answer is from the transcripts the lectures come
with; the scenario is the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump, speech).

    recipe = a6_lecture_audio.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = a6_lecture_audio.build(recipe, folder, 3)  # the lectures' sound written as M4A with their times, and the questions

Scales: `tiny`, the first three lectures (0.08 GB, built in 2 s, for the tests and a smoke run); `standard`, all fifteen
(0.43 GB, 20 s, 24 October to 14 December 2016)."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf
from mmsb.scenarios import realdump
from mmsb.scenarios.a6_lecture_audio.questions import CODE, NEVER, SOURCE_NAME, TERMS, questions_of
from mmsb.scenarios.a6_lecture_audio.source import LECTURES, SOURCE, class_day, read_term, talks_of
from mmsb.scenarios.realdump import File, report, speech, speech_files  # noqa: F401  (`report`: the Scenario's)

name = "a6_lecture_audio"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Recorded lectures", "audio", "school", "a phone's recordings of one course, a class at a time through a term")
filing = speech.FILING

SCALES = {"tiny": LECTURES[:3], "standard": LECTURES}  # the lectures at each scale
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
WHAT = "the voice recordings I made of my lectures this term"
LAYOUT = ("Recordings holds one M4A for each class, Recording_<date>_<time>.m4a, named for the moment I pressed record; a file's modified time "
          "is when its recording began.")
SOURCE_WORDS = ("ocw", "opencourseware", "mit6", "60002", "6.0002", "mit", "archive", "guttag", "grimson")  # the source's names, which no file may hold
COMMAND_WORDS = ("ocw-6-0002", "mit6_0002", "mit6.0002", "archive.org", "opencourseware")  # and which no command may


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
    files = tuple(File(t.path, SOURCE_NAME, realdump.local_t(t.start), "recording", "audio", {"src": f"{r.stem}.mp4"}, length=t.length)
                  for t, r in zip(talks, recs, strict=True))
    return files, questions_of(talks, [r.stem for r in recs], scale, seed)


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
    """Writes the recordings into `out` (new or empty): the sound of each lecture as mono M4A with none of the source's metadata, and its
    time; `workers` at once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        speech_files.extract_sound(source / str(f.spec["src"]), path)
        speech_files.check_plays(path, float(f.length), speech_files.audio_seconds)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no recording's name or metadata names a term of the questions, or the course."""
    return realdump.probe(dump, out, [*(t.search for t in (*TERMS, *NEVER)), *SOURCE_WORDS], speech_files.HEADERS)


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The recordings themselves: nothing is cut (every question is asked after the last lecture)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
