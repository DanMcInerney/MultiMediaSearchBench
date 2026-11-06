"""I3, the trail camera: the memory card of one camera in the hills of the south-western United States (Caltech Camera Traps, location
23), months of photos of whatever passed, day and night (at night in infrared), most of them of nothing. The questions ask which days an
animal came, how often, when it was first or last seen, and of animals that never came (questions.py). Every answer is from the
dataset's labels; the scenario is the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = i3_trailcam.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = i3_trailcam.build(recipe, folder, 3)  # the photos copied in with EXIF and their times, and the questions

Scales: `tiny`, one week of December 2013 (199 photos, for the tests and a smoke run); `standard`, the whole card, 10 September 2013 to
28 February 2014 (4,256 photos, 1.9 GB)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.core.scenario import Access, ProbeOutcome, dump_key
from mmmb.scenarios import realdump
from mmmb.scenarios.i3_trailcam.questions import CODE, SOURCE_NAME, questions_of
from mmmb.scenarios.i3_trailcam.source import IMAGES, SOURCE, read_labels, with_exif
from mmmb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i3_trailcam"
code = CODE
generator = 1  # bumped when the scenario makes something other than before

SCALES = {"tiny": ("2013-12-16", "2013-12-22"), "standard": ("2013-09-10", "2014-02-28")}  # the days of the card at each scale
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "DCIM/100RECNX"
WHAT = "the memory card of my trail camera"
LAYOUT = (f"{FOLDER} holds its photos, IMG_0001.JPG onwards in the order they were taken, day and night (at night in infrared); a "
          "photo's time is in its EXIF (DateTimeOriginal) and is the file's modified time.")
COMMAND_WORDS = ("caltech", "cct_images", "camera-traps", "camera_traps")  # the source's names


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the card's
    days and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    photos, names = read_labels(str(recipe["source"]))
    first, last = recipe["days"][0], recipe["days"][-1]
    card = [p for p in photos if first <= p.wall.date().isoformat() <= last]
    files = tuple(File(f"{FOLDER}/IMG_{n:04d}.JPG", SOURCE_NAME, realdump.local_t(p.wall), "photo", "image",
                       {"src": f"{IMAGES}/{p.file}", "taken": p.wall.isoformat()}, own_time=True) for n, p in enumerate(card, 1))
    return files, questions_of(card, [f.path for f in files], names, str(recipe["scale"]), int(recipe["seed"]), realdump.local_t)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the card into `out` (new or empty): each photo copied with its EXIF and its time; `workers` at once. Returns the dump and
    the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        path.write_bytes(with_exif((source / str(f.spec["src"])).read_bytes(), datetime.fromisoformat(str(f.spec["taken"]))))

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no photo's name or EXIF names a label of the dataset, or the dataset."""
    _, names = read_labels(str(recipe["source"]))
    return realdump.probe(dump, out, [*names, *COMMAND_WORDS])


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The card itself: nothing is cut (every question is asked after its last photo)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, ("bench-data",))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""

