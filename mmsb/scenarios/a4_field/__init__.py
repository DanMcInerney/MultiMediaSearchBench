"""A4, the field recorder: one autonomous recorder's night in a field near Ithaca, New York (BirdVox-full-night, unit 3, 23 September 2015),
as the recorder wrote it, hourly FLAC files of one microphone, nearly eleven hours of the sky in the autumn migration. The calls of the
migrating birds are faint, short and many (9,113 of them marked), and the labels are points in time, with a pitch and no species. The
questions ask when the first call came, how many calls there were in a short stretch with few, when the first call in a window came,
which hour was quietest, what pitch a call had, and of stretches with no call at all
(questions.py). Every answer is from the expert's marks, with the care that faint calls may be missing; the scenario is the names of a
`Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = a4_field.recipe(scale="tiny", seed=1)      # the source, the scale's day, this machine's UTC offsets
    dump, questions = a4_field.build(recipe, folder, 3)  # the hourly files cut from the night, with their times, and the questions

Scales: `tiny`, the first two hours (2 files, 0.16 GB, for the tests and a smoke run); `standard`, the whole night, 03:31 to 14:27 by the
recorder's clock (11 files, 0.81 GB). The recorder's clock is the dataset's UTC start time of the unit (epoch 1443065462), kept as the dump's
wall-clock time (realdump). The calls come thickest in the last hour, where the labels run on to the last second of the file, as at a dawn
chorus, which a start at 03:31 UTC (23:31 in Ithaca) would not give: the start time is probably four hours out. It does not matter to the
questions (each is from the offsets into the recording), only to the hour of the day they are said in."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.a4_field.questions import CODE, SOURCE_NAME, questions_of
from mmsb.scenarios.a4_field.source import HOUR_S, RATE, SOURCE, cut, flac_header, read_night, streaminfo
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "a4_field"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Field recorder", "audio", "fun", "one night of a recorder in a field, hourly files, mostly quiet sky with faint calls")
filing = with_absent({"first": ("when", "sound"), "count": ("how_many", "sound"), "pitch": ("what", "sound"), "did": ("did", "sound"),
                      "quiet": ("who", "sound")})

DAY = "2015-09-24"  # the day of the night's start by the recorder's clock
SCALES = {"tiny": 2, "standard": None}  # the hourly files of the night each holds (None: all of them)
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "AUDIO"
WHAT = "the field recorder I left out for the night"
LAYOUT = (f"{FOLDER} holds the night's recording of its one microphone as hourly files, REC_<date>_<time>.flac, each named for the moment it began "
          "by the recorder's clock (which is also the file's modified time) and playing one hour, the last the rest of the night (FLAC, mono, 24 kHz).")
SOURCE_NAMES = ("birdvox", "unit03", "lostanlen", "zenodo", "ithaca", "cornell")  # the source's names, which no file of the dump holds
COMMAND_WORDS = ("birdvox", "unit03", "lostanlen")  # and which no command may
# the plain words the labels are said in, which the probe looks for as it does the labels
LABEL_WORDS = ("flight call", "flight", "call", "migrant", "migration", "thrush", "warbler", "sparrow", "species")


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the night's day and
    this machine's UTC offsets at its midnight."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(date.fromisoformat(DAY), date.fromisoformat(DAY))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    night = read_night(str(recipe["source"]))
    if night.start.date().isoformat() != DAY:
        raise ValueError(f"the recording begins on {night.start.date()}, not {DAY}: the recipe's day")
    count = SCALES[scale] or -(-night.samples // (HOUR_S * RATE))  # whole hours and the part of one
    files = []
    for k in range(count):
        begins = night.start + timedelta(seconds=k * HOUR_S)
        samples = min(HOUR_S * RATE, night.samples - k * HOUR_S * RATE)
        files.append(File(f"{FOLDER}/REC_{begins:%Y%m%d_%H%M%S}.flac", SOURCE_NAME, realdump.local_t(begins), "recording", "audio",
                          {"from": k * HOUR_S, "seconds": None if samples < HOUR_S * RATE or k == count - 1 and SCALES[scale] is None else HOUR_S},
                          length=samples / RATE))
    now = realdump.local_t(datetime.combine(date.fromisoformat(recipe["days"][0]) + timedelta(days=1), time(9)))
    return tuple(files), questions_of(night, files, scale, seed, now)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the night into `out` (new or empty): each hourly file cut from the source's FLAC sample for sample, with its time; `workers` at
    once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        cut(source, int(f.spec["from"]), f.spec["seconds"], path)
        rate, channels, bits, samples = streaminfo(path)
        if (rate, channels, bits) != (RATE, 1, 16) or samples != round(f.length * RATE):
            raise ValueError(f"{path.name} is {samples} samples at {rate} Hz, not the {round(f.length * RATE)} its place in the night gives")

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no file's name or FLAC metadata names a label of the questions (or a plain word for one), or the dataset."""
    return realdump.probe(dump, out, [*LABEL_WORDS, *SOURCE_NAMES], {".flac": flac_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The card itself: nothing is cut (every question is asked after the night's end)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
