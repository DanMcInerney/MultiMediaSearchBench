"""I9, the nature photos: one person's walks over four years, about 3,200 photos of the plants, birds, insects and animals they met in the north-
eastern United States (an iNaturalist observer: Ruffed Grouse, red trillium, pink lady's slipper, Butternut), each the first photo of an
observation the community has named to the species. The questions are the owner's, by the names a person says: on which days did I photograph a
trillium, on how many days a Wild Turkey, when did I first or last photograph a hummingbird, find my photo of that woodpecker, did I ever
photograph a moose, which did I see first; and some about a species the pile has none of (questions.py). Every answer is from the observations
(source.py). The scenario is the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = i9_nature.recipe(scale="tiny", seed=1)      # the source, the pile's days, this machine's UTC offsets
    dump, questions = i9_nature.build(recipe, folder, 3)  # the photos copied in with EXIF and their times, and the questions

A held-out species (an absent question's, one of the seed's) has every photo of it left out of the pile, so the dump depends on the seed. Scales:
`tiny`, a few weeks of the pile (for the tests and a smoke run); `standard`, all of it but the held-out species (about 3,200 photos, 1.2 GB). A photo
is named as a phone names it, `DCIM/Camera/IMG_0001.jpg` onwards in the order taken, with EXIF that holds the local time it was taken and nothing
else, which is also its modified time. A photo of an hour that this machine's clock skips or repeats (daylight saving) is left out."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.i9_nature.questions import (
    CODE,
    NEVER,
    SOURCE_NAME,
    candidates,
    draw,
    held_out,
    questions_of,
)
from mmsb.scenarios.i9_nature.source import (
    PHOTOS,
    SOURCE,
    Obs,
    app_segments,
    check_photos,
    read_genera,
    read_names,
    read_pile,
    read_rows,
    stamped,
)
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i9_nature"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Nature photos", "image", "fun", "four years of one person's walks: plants, birds, insects and animals, each named")
filing = with_absent({"days": ("find_all", "sight"), "count": ("how_many", "sight"), "first": ("when", "sight"), "last": ("when", "sight"),
                      "find": ("find", "sight"), "ever": ("did", "sight"), "which": ("order", "sight")})

PILE_DAYS = ("2022-01-01", "2025-12-31")
SCALES = {"tiny": ("2023-05-01", "2023-06-14"), "standard": PILE_DAYS}  # the days of the pile at each scale
HELD = {"tiny": 1, "standard": 3}  # the species held out of the pile, each asked of as absent
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "DCIM/Camera"
WHAT = "my nature photos"
LAYOUT = (f"{FOLDER} holds its photos, IMG_0001.jpg onwards in the order they were taken (JPEG, about 1000 pixels across); a photo's time "
          "is in its EXIF (DateTimeOriginal) and is the file's modified time, on the clock where I took it.")
COMMAND_WORDS = ("inaturalist", "inat-observer", "inat_observer")  # the source's names


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the held-out species and questions are drawn
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


def genera_of(source: str) -> dict[str, set[str]]:
    """The species of each genus the observations name, a genus named alone (to whatever rank) counting as a species of its own, so that a
    genus that has such a name has more than one."""
    out: dict[str, set[str]] = defaultdict(set)
    for r in read_rows(source):
        out[r["taxon_name"].split()[0]].add(" ".join(r["taxon_name"].split()[:2]) if r["taxon_rank"] != "genus" else r["taxon_name"] + " (genus)")
    return out


def pile_of(source: str, first: str, last: str) -> list[Obs]:
    """The observations of the pile taken from the day `first` to the day `last` (ISO), but those of an hour the clock skips or repeats."""
    pile = [o for o in read_pile(source) if first <= o.day <= last and clock_ok(o.wall)]
    if not pile:
        raise ValueError(f"the source has no observation from {first} to {last}")
    return pile


def never_of(source: str) -> list[str]:
    """The common names of NEVER that no observation has: neither the species' genus (to whatever rank), nor the name itself."""
    genera, names = read_genera(source), {n.casefold() for n in read_names(source).values()}
    return [c for c, genus in NEVER.items() if genus not in genera and c.casefold() not in names]


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    source, scale, seed = str(recipe["source"]), str(recipe["scale"]), int(recipe["seed"])
    whole = pile_of(source, recipe["days"][0], recipe["days"][-1])
    names = read_names(source)
    gone = held_out(whole, names, genera_of(source), random.Random(f"i9:{scale}:{seed}:held"), HELD[scale])
    pile: Sequence[Obs] = [o for o in whole if o.species not in gone]
    asked = draw(candidates(pile, names, scale), never_of(source), gone, scale, seed, names)
    check_photos(source, pile)
    files = tuple(File(f"{FOLDER}/IMG_{n:04d}.jpg", SOURCE_NAME, realdump.local_t(o.wall), "photo", "image",
                       {"src": f"{PHOTOS}/{o.photo}", "taken": o.wall.isoformat()}, own_time=True) for n, o in enumerate(pile, 1))
    paths = {o.photo: f.path for o, f in zip(pile, files, strict=True)}
    now = realdump.local_t(datetime.combine(pile[-1].wall.date() + timedelta(days=1), time(9)))
    return files, questions_of(asked, names, paths, files[0].t, files[-1].t + 1, now, seed)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the photos into `out` (new or empty): each photo copied with its EXIF cleared to its time, and that time; `workers` at once.
    Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        path.write_bytes(stamped((source / str(f.spec["src"])).read_bytes(), datetime.fromisoformat(str(f.spec["taken"]))))

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no photo's name or metadata (its application segments and comments) says a species of the observations, its genus, or
    the source."""
    source = str(recipe["source"])
    labels = [*read_names(source).values(), *(b for b in read_names(source)), *read_genera(source), *NEVER, *COMMAND_WORDS]
    return realdump.probe(dump, out, labels, {".jpg": app_segments, ".jpeg": app_segments})


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
