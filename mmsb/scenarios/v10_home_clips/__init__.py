"""V10, phone clips at home: the camera roll of a phone's videos of life at home, about 3,000 short clips of a person in the kitchen, the bedroom, on
the stairs, doing something. THE CLIPS ARE ACTED: they are Charades (AI2), where crowd workers filmed themselves in their own homes acting out a
script ("a person opens a refrigerator, then sneezes"); the people are 267 different people in as many homes, here one camera roll, and the
dates are made up. The questions ask for the clip of one action, every clip of another, how many clips there are of one, or were filmed in a room,
how long something lasted, which room, whether another action was before or after it, and whether an action was ever filmed (questions.py). Every answer is from the dataset's
labels (a clip's actions with their start and end, its scene), but those are not exhaustive, so the pile is built to make each question certain: of
every action asked about it holds only the clips where the label and the words of the script agree, and none where they do not (pile.py). The
scenario is the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = v10_home_clips.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = v10_home_clips.build(recipe, folder, 3)  # the clips copied out of the zip as they are, with their times, and the questions

Scales: `tiny`, 72 clips (about 0.1 GB, for the tests and a smoke run); `standard`, 3,000 clips (about 24 hours, 5.0 GB). The source has no
recording time: the clips are named as a phone names them (`DCIM/100MEDIA/VID_20250612_181502.mp4`) and their times are a seeded spread over February
2025 to October 2026 (March 2026 at tiny) that has no link to what a clip holds. The pile does not depend on the seed that draws the questions.
Charades' licence allows use for research and evaluation but not passing the data to a third party: the dump is built on the user's machine from
the user's own download, and is not to be published."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)
from mmsb.scenarios.v10_home_clips.pile import ROOMS, TOPICS, choose
from mmsb.scenarios.v10_home_clips.questions import CODE, SOURCE_NAME, questions_of
from mmsb.scenarios.v10_home_clips.source import (
    SCENES,
    SOURCE,
    check_members,
    class_names,
    copy_member,
    mp4_header,
    object_names,
    read_clips,
)

name = "v10_home_clips"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Phone clips at home", "video", "home", "about 3,000 short phone videos of people at home, acted out from scripts, named as a phone names them")
filing = with_absent({"find": ("find", "sight"), "every": ("find_all", "sight"), "count": ("how_many", "sight"), "rooms": ("how_many", "sight"),
                      "long": ("how_many", "sight"), "room": ("who", "sight"), "before": ("order", "sight"), "did": ("did", "sight")})

SCALES = {"tiny": ("2026-03-01", "2026-03-14"), "standard": ("2025-02-01", "2026-10-31")}  # the days the clips are made over
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "DCIM/100MEDIA"
GAP_S = 60  # at least this long between the end of a clip and the next one's time
WHAT = "the videos on my phone"
LAYOUT = (f"{FOLDER} holds its videos, VID_<date>_<time>.mp4 onwards in the order they were filmed (each is the moment it began, which is also the "
          "file's modified time), one clip of a person at home each, with sound, 8 to 45 seconds long.")
SOURCE_NAMES = ("charades", "allenai", "ai2", "hollywood in homes")  # the source's names, which no file of the dump may hold
COMMAND_WORDS = ("charades", "allenai", "ai2-public")  # and which no command may
LABEL_WORDS = ("washing", "fixing", "opening", "closing", "sneezing", "vacuum", "fridge", "refrigerator", "sandwich", "medicine")  # plain words of the labels


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days the clips
    are made over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def spread(lengths: Sequence[float], days: Sequence[str], rng: random.Random) -> list[float]:
    """The times (epoch seconds, in order) of clips that play `lengths`: seeded draws of a day and a time between 7:00 and 22:30, each moved later
    when it would be within GAP_S of the end of the clip before."""
    draws = sorted(datetime.combine(date.fromisoformat(rng.choice(days)), time(7)).timestamp() + rng.uniform(0, 15.5 * 3600) for _ in lengths)
    out, free = [], 0.0
    for draw, length in zip(draws, lengths, strict=True):
        out.append(float(max(round(draw), free)))
        free = out[-1] + length + GAP_S
    return out


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone, or the source's
    videos lack one the pile needs."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed, days, source = str(recipe["scale"]), int(recipe["seed"]), recipe["days"], str(recipe["source"])
    pile = choose(read_clips(source), scale)
    check_members(source, list(pile.clips))
    order = list(pile.clips)
    random.Random(f"v10:{scale}:order").shuffle(order)  # the order the clips were filmed in: no link to what they hold or to their ids
    times = spread([c.length for c in order], days, random.Random(f"v10:{scale}:times"))
    files = tuple(File(f"{FOLDER}/VID_{datetime.fromtimestamp(t):%Y%m%d_%H%M%S}.mp4", SOURCE_NAME, t, "clip", "video", {"member": c.member},
                       length=c.length) for c, t in zip(order, times, strict=True))
    path = {c.id: f.path for c, f in zip(order, files, strict=True)}
    when = {c.id: f.t for c, f in zip(order, files, strict=True)}
    now = realdump.local_t(datetime.combine(datetime.fromtimestamp(times[-1]).date() + timedelta(days=1), time(9)))
    return files, questions_of(pile, path, when, scale, seed, now, files[0].t, files[-1].t + files[-1].length)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the clips into `out` (new or empty): each member of the videos zip copied out whole, with its time; `workers` at once. Returns
    the dump and the questions."""
    files, questions = plan_of(recipe)
    source = str(recipe["source"])

    def write(f: File, path: Path) -> None:
        copy_member(source, str(f.spec["member"]), path)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no clip's name or metadata names an action, object or room of the source, or the source. (The folder is not `Camera`, a
    word of the class `Holding a phone/camera`.) A clip's length in the manifest is the CSV's, which is the container's to a third of a second."""
    source = str(recipe["source"])
    labels = [*class_names(source).values(), *object_names(source), *SCENES, *(t.phrase for t in TOPICS.values()), *(v[0] for v in ROOMS.values()),
              *SOURCE_NAMES, *LABEL_WORDS]
    return realdump.probe(dump, out, labels, {**realdump.METADATA, ".mp4": mp4_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The videos themselves: nothing is cut (every question is asked after the last clip)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
