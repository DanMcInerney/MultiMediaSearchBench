"""V1, the security cameras: three fixed outdoor cameras over roads and car parks (MEVA, Kitware's labels), 75 minutes each, one MP4 of
each unbroken run of the recorder's five-minute clips (the recorder stopped for five minutes at 10:35). The questions ask when an activity
(a trunk opened, a car reversing) was first or last seen on a camera, how often it happened in a window, and of activities that
never happened (questions.py). Every answer is from the labels; the scenario is the names of a `Scenario`, as functions (DESIGN 5), over the
shared builder (realdump).

    recipe = v1_security.recipe(scale="tiny", seed=1)      # the source, the scale's day, this machine's UTC offsets
    dump, questions = v1_security.build(recipe, folder, 3)  # the clips joined by stream copy, with their times, and the questions

Scales: `tiny`, the first ten minutes of each camera (3 files, about 0.8 GB, for the tests and a smoke run); `standard`, 10:00 to 11:15 on
12 March 2018 (6 files, about 5.9 GB)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.core.scenario import Access, ProbeOutcome, dump_key
from mmmb.scenarios import realdump
from mmmb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)
from mmmb.scenarios.v1_security.questions import CODE, questions_of
from mmmb.scenarios.v1_security.source import (
    DAY,
    FIRST,
    FPS,
    SOURCE,
    activity_names,
    events_in,
    join,
    mp4_header,
    read_clips,
    runs_of,
)

name = "v1_security"
code = CODE
generator = 1  # bumped when the scenario makes something other than before

SCALES = {"tiny": (FIRST, datetime(2018, 3, 12, 10, 10)), "standard": (FIRST, datetime(2018, 3, 12, 11, 15))}  # the wall times each covers
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
WHAT = "the recordings of my three outdoor security cameras"
LAYOUT = ("cam1, cam2 and cam3 are three fixed cameras over roads and car parks, each with a folder of its own; a recording's name gives when "
          "it started and ended (YYYYMMDD_HHMMSS-HHMMSS) and the file's modified time is its start. A camera has more than one recording "
          "when it stopped for a while.")
SOURCE_NAMES = ("meva", "kitware", "g336", "g341", "g436")  # the source's names
# which no command may hold: "meva" as a path, since it is in `timevalue` and `somevar`
COMMAND_WORDS = ("/meva", "meva/", "kitware", "g336", "g341", "g436")
SOURCE_WORDS = (*SOURCE_NAMES, "school", "hospital")  # and the words no file of the dump may hold


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the day and
    this machine's UTC offsets at its start and end."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = [DAY]
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale = str(recipe["scale"])
    first, last = SCALES[scale]
    clips = [c for c in read_clips(str(recipe["source"])) if first <= c.start < last]
    runs = runs_of(tuple(clips))
    files, events = [], []
    for run in runs:
        camera, a, b = run[0].camera, run[0].start, run[-1].end
        path = f"{camera}/{camera}_{a:%Y%m%d_%H%M%S}-{b:%H%M%S}.mp4"
        frames = [c.frames for c in run]
        files.append(File(path, camera, realdump.local_t(a), "recording", "video", {"clips": [c.rel for c in run], "frames": frames},
                          length=sum(frames) / FPS))
        events += events_in(path, camera, files[-1].t, run)
    return tuple(files), questions_of(events, files, SCALES[scale], scale, int(recipe["seed"]), realdump.local_t)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the cameras' recordings into `out` (new or empty): each run of clips joined by stream copy, with its time; `workers` at once.
    Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        join([source / rel for rel in f.spec["clips"]], path)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no recording's name or metadata names an activity of the labels, or the dataset's sites and cameras."""
    return realdump.probe(dump, out, [*activity_names(str(recipe["source"])), *SOURCE_WORDS], {**realdump.METADATA, ".mp4": mp4_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The recordings themselves: nothing is cut (every question is asked after the last recording)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, ("bench-data",))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
