"""I4, the building inspector's photos: an inspector's close-ups of walls, a few hundred of them, in a folder for each job ("Maple Street",
"Harbour Road"), of cracks, peeling paint, spalling, algae, staining and plain wall. The questions are the inspector's: which photos on this
job show cracks, how many show algae, did I photograph any spalling there, find the photo of the staining; and some are about a defect a job
has none of (questions.py). BD3's labels are the truth: one class per photo, which may show a little of another, so a question is which
photos are labelled so (questions.py says how it is kept certain). The scenario is the names of a `Scenario`, as functions (DESIGN 5), over
the shared builder (realdump).

    recipe = i4_site_photos.recipe(scale="tiny", seed=1)      # the source, the days of the visits, this machine's UTC offsets
    dump, questions = i4_site_photos.build(recipe, folder, 3)  # the photos written as JPEGs with EXIF and their times, and the questions

Scales: `tiny`, 3 folders, 45 photos (for the tests and a smoke run); `standard`, 24 folders, the 793 photos of the source but those
of two classes held out (stain and spalling) and a few left out of folders (about 540 photos, 20 MB: the source is all there is of it). A folder is
`Site photos/<job>/`, its photos `IMG_0001.jpg` onwards in the order taken across folders, 512 by 512 pixels, with EXIF that holds their time
and nothing else, which is also their modified time. The source has no times: a seeded weekday of 2025 for each folder, its photos a half to
two minutes apart."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.i4_site_photos.questions import (
    CODE,
    FOLDER,
    SOURCE_NAME,
    VISITS,
    draw,
    layout,
    questions_of,
    times_of,
)
from mmsb.scenarios.i4_site_photos.source import CLASSES, SOURCE, check_photos, jpeg_of, read_labels
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i4_site_photos"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Building inspection photos", "image", "work", "an inspector's close-ups of walls in a folder for each job")
filing = with_absent({"photos": ("find_all", "sight"), "count": ("how_many", "sight"), "any": ("did", "sight"), "one": ("find", "sight")})

SCALES = {"tiny": ("2025-03-10", "2025-03-14"), "standard": VISITS}  # the days the visits are drawn from at each scale
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
WHAT = "my site photos"
LAYOUT = (f"{FOLDER} holds a folder for each job, named for the site, and in each its photos, IMG_0001.jpg onwards in the order I took them "
          "(JPEG, 512 by 512 pixels, a close-up of a wall each); a photo's time is in its EXIF (DateTimeOriginal) and is the file's modified time.")
COMMAND_WORDS = ("bd3", "building_defect", "building-defect", "building defect", "kottari", "chandrabhuma")  # the source's names
WORDS = ("algae", "crack", "cracks", "cracking", "peeling", "spalling", "spalled", "staining", "stained", "mould", "mold")  # a label's other forms


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the held-out classes, folders, questions and
    times are drawn with, the first and last day the visits are drawn from and this machine's UTC offsets at them."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = list(SCALES[scale])
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    source, scale, seed = str(recipe["source"]), str(recipe["scale"]), int(recipe["seed"])
    sites, absent = layout(read_labels(source), scale, seed, recipe["days"])
    asked = draw(sites, absent, scale, seed)
    walls = times_of(sites, seed, realdump.local_t)
    files, paths = [], {}
    for site, row in zip(sites, walls, strict=True):
        for shot, t in zip(site.shots, row, strict=True):
            path = f"{FOLDER}/{site.name}/IMG_{len(files) + 1:04d}.jpg"
            paths[shot.file] = path
            files.append(File(path, SOURCE_NAME, t, "photo", "image", {"src": shot.file, "taken": datetime.fromtimestamp(t).isoformat()},
                              own_time=True))
    check_photos(source, [s for site in sites for s in site.shots])
    now = realdump.local_t(datetime.combine(sites[-1].day + timedelta(days=1), time(9)))
    return tuple(files), questions_of(sites, asked, paths, files[0].t, files[-1].t + 1, now, seed)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the photos into `out` (new or empty): each source photo written as a JPEG with its time as EXIF and as its modified time;
    `workers` at once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        png = (source / "test_images" / str(f.spec["src"]).rsplit("/", 1)[-1]).read_bytes()
        path.write_bytes(jpeg_of(png, datetime.fromisoformat(str(f.spec["taken"]))))

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no photo's name or EXIF says a class of the dataset, a word of one, or the dataset."""
    return realdump.probe(dump, out, [*CLASSES, *WORDS, *COMMAND_WORDS])


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The photos themselves: nothing is cut (every question is asked after the last photo)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
