"""A1, the home recorder: a voice recorder left running in a living room for a day, one long audio file (a day of the SINS database, node 7:
a person living alone in a vacation home). The questions ask when the vacuum cleaner went on and off, when the TV did, when a visitor
came and left, when the resident left the house and got back, how many times, and of things that did not happen in a stretch of the day
(questions.py). Every answer is from the database's labels; the scenario is the names of a `Scenario`, as functions (DESIGN 5), over
the shared builder (realdump).

    recipe = a1_home_audio.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = a1_home_audio.build(recipe, folder, 3)  # the recording written in with its time, and the questions

Scales: `tiny`, one hour of the evening of 2 February 2017, 21:30 to 22:30 (115 MB, for the tests and a smoke run); `standard`, the whole
day, midnight to midnight (2.8 GB, one WAV of 24 hours)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.scenarios import realdump
from mmsb.scenarios.a1_home_audio.questions import CODE, SOURCE_NAME, questions_of
from mmsb.scenarios.a1_home_audio.source import SOURCE, read_runs, write_wav
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "a1_home_audio"
code = CODE
generator = 1  # bumped when the scenario makes something other than before

DAY = "2017-02-02"
SCALES = {"tiny": (21 * 3600 + 30 * 60, 3600), "standard": (0, 86400)}  # the recording at each scale: seconds after midnight it begins, how long
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "RECORD"
WHAT = "the voice recorder I left running in my living room"
LAYOUT = (f"{FOLDER} holds its recording, REC_<date>_<time>.wav, named for the moment it began (which is also the file's modified time); "
          "it plays on from there without a break.")
SOURCE_NAMES = ("sins", "node07", "node7_", "kuleuven", "dcase")  # the source's names, which no file of the dump may hold
# and which no command may: as paths, since "sins" is in `isinstance` and "dcase" in "mixedcase"
COMMAND_WORDS = ("/sins", "sins/", "node07", "node7_", "kuleuven", "dcase20")
# the plain words the labels are said in (`vacuumcleaner`, `watching tv`, `visit`, `absence`), which the probe looks for as it does the labels
LABEL_WORDS = ("vacuum", "vacuuming", "cleaner", "tv", "television", "visitor", "guest", "away", "absent")


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the day and this
    machine's UTC offsets at its midnight."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(date.fromisoformat(DAY), date.fromisoformat(DAY))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The file and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    day = date.fromisoformat(recipe["days"][0])
    first, seconds = SCALES[str(recipe["scale"])]
    begins = datetime.combine(day, datetime.min.time()) + timedelta(seconds=first)
    path = f"{FOLDER}/REC_{begins:%Y%m%d_%H%M%S}.wav"
    wav = File(path, SOURCE_NAME, realdump.local_t(begins), "recording", "audio", {"from": begins.isoformat(), "seconds": seconds},
               length=float(seconds))
    runs = read_runs(str(recipe["source"]))
    return (wav,), questions_of(runs, day, first, seconds, path, str(recipe["scale"]), int(recipe["seed"]), realdump.local_t)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the recording into `out` (new or empty): the minute files joined into one WAV, with its time. Returns the dump and the
    questions."""
    files, questions = plan_of(recipe)
    source = str(recipe["source"])

    def write(f: File, path: Path) -> None:
        write_wav(source, datetime.fromisoformat(str(f.spec["from"])), int(f.spec["seconds"]), path)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no file's name or metadata names a class of the labels (or a plain word for one), or the database."""
    living, home = read_runs(str(recipe["source"]))
    return realdump.probe(dump, out, [*sorted({r.cls for r in (*living, *home)}), *LABEL_WORDS, *SOURCE_NAMES])


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The recorder itself: nothing is cut (every question is asked after the recording's end)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
