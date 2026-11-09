"""I1, the phone camera roll: one person's 3,759 phone photos over about three years (ATM-Bench: April 2022 to June 2025, Cambridge and a
dozen trips), of food, posters and slides at conferences, boarding passes, sights, the cat. The questions are the owner's own, in their own
words: find the photo of this, every photo of that, when was that, how much was it, what did it say, which one, how many, did I ever,
and ones whose answer is not in the pile (questions.py). Every answer is from the dataset's labels. The scenario is the names of a `Scenario`,
as functions (DESIGN 5), over the shared builder (realdump).

    recipe = i1_camera_roll.recipe(scale="tiny", seed=1)      # the source, the pile's days, this machine's UTC offsets
    dump, questions = i1_camera_roll.build(recipe, folder, 3)  # the photos copied in with EXIF and their times, and the questions

Absent questions are held-out needles: a real question whose evidence photos, and every photo within a minute of them, are left out of the
pile, so the dump depends on the seed. Scales: `tiny`, a few days of the pile (for the tests and a smoke run); `standard`, all of it but the
held-out photos (about 3,700 photos, 1.3 GB). A photo is named as a phone names it, `DCIM/Camera/IMG_0001.jpg` onwards in the order taken, with
EXIF that holds its time and nothing else (the dataset's GPS is stripped), which is also its modified time. The time is the clock where it was
taken (the pile has no zone); a photo of an hour that this machine's clock skips or repeats (daylight saving) is left out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.i1_camera_roll.questions import (
    CODE,
    READ,
    SOURCE_NAME,
    THINGS,
    WHICH,
    Pile,
    candidates,
    draw,
    questions_of,
)
from mmsb.scenarios.i1_camera_roll.source import (
    IMAGES,
    SOURCE,
    Photo,
    app_segments,
    check_photos,
    read_photos,
    read_real,
    stamped,
)
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i1_camera_roll"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Phone camera roll", "image", "home", "one person's phone photos over three years: food, posters, passes, sights, the cat")
filing = with_absent({"find": ("find", "sight"), "photos": ("find_all", "sight"), "date": ("when", "sight"), "price": ("what", "text"),
                      "read": ("what", "text"), "which": ("who", "mixed"), "days": ("how_many", "sight"), "first": ("order", "sight"),
                      "ever": ("did", "sight")})

PILE_DAYS = ("2022-04-30", "2025-06-25")  # the photos of the source: first and last day
SCALES = {"tiny": ("2022-06-19", "2022-06-21"), "standard": PILE_DAYS}  # the days of the pile at each scale
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "DCIM/Camera"
WHAT = "my phone's camera roll"
LAYOUT = (f"{FOLDER} holds its photos, IMG_0001.jpg onwards in the order they were taken (JPEG, mostly 1000 by 750 pixels); a photo's time "
          "is in its EXIF (DateTimeOriginal) and is the file's modified time, on the clock of the place where I took it.")
COMMAND_WORDS = ("atm-bench", "atm_bench", "atmbench", "jingbiao", "raw_memory")  # the source's names
MIN_WORD = 4  # the probe looks for no word shorter than this (a seat, "1k")


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions and held-out photos are drawn
    with, the first and last day of the pile and this machine's UTC offsets at them."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = list(SCALES[scale])
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def clock_ok(wall: datetime) -> bool:
    """Whether this machine's clock has the wall time `wall` once: a time it skips or repeats (daylight saving) would have a file whose
    modified time reads as another hour than its EXIF."""
    t = wall.timestamp()
    return datetime.fromtimestamp(t) == wall and wall.replace(fold=0).timestamp() == wall.replace(fold=1).timestamp()


def pile_of(source: str, first: str, last: str) -> Pile:
    """The photos of `source` taken from the day `first` to the day `last` (ISO), but those of an hour the clock skips or repeats."""
    photos = [p for p in read_photos(source) if first <= p.wall.date().isoformat() <= last and clock_ok(p.wall)]
    if not photos:
        raise ValueError(f"the source has no photo from {first} to {last}")
    return Pile(photos)


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    source, scale, seed = str(recipe["source"]), str(recipe["scale"]), int(recipe["seed"])
    first, last = recipe["days"][0], recipe["days"][-1]
    whole = pile_of(source, first, last)
    chosen, held = draw(candidates(read_real(source), whole), whole, scale, seed)
    kept: Sequence[Photo] = [p for p in whole.photos if p.file not in held]
    check_photos(source, [p.file for p in kept])
    files = tuple(File(f"{FOLDER}/IMG_{n:04d}.jpg", SOURCE_NAME, realdump.local_t(p.wall), "photo", "image",
                       {"src": f"{IMAGES}/{p.file}", "taken": p.wall.isoformat()}, own_time=True) for n, p in enumerate(kept, 1))
    paths = {p.file: f.path for p, f in zip(kept, files, strict=True)}
    now = realdump.local_t(datetime.combine(kept[-1].wall.date() + timedelta(days=1), time(9)))
    questions = questions_of(chosen, paths, files[0].t, files[-1].t + 1, now, seed)
    return files, questions


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the camera roll into `out` (new or empty): each photo copied with its EXIF cleared to its time, and that time; `workers` at
    once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        path.write_bytes(stamped((source / str(f.spec["src"])).read_bytes(), datetime.fromisoformat(str(f.spec["taken"]))))

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no photo's name or metadata (its application segments and comments) says one of the things, titles, names or the
    dataset the questions are about."""
    words = [w for text in (*(t for t, _ in THINGS.values()), *(k for _, k in THINGS.values()), *(a for a, _ in (*READ.values(), *WHICH.values())))
             for w in text.split("|") if len(w) >= MIN_WORD]
    return realdump.probe(dump, out, [*words, *COMMAND_WORDS], {".jpg": app_segments})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The camera roll itself: nothing is cut (every question is asked after its last photo)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
