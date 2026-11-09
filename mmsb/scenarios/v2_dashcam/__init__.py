"""V2, the dashcam card, STAGED: it is not one person's dashcam card. It is about a hundred short clips of real driving on Australian roads,
each a crash, a near miss or an ordinary drive, cut from two YouTube compilations of footage from many different people's dashcams (the source has
128 clips), and presented here as one card. Their dates and times are invented, and the clocks and the sun do not agree (a night clip may be dated
in the middle of the day). It is small, about half an hour of video, and the questions are about the vehicles and the road, never about the people in a crash: which
clips have a crash or a near miss, at night, in the rain, on a highway, at a roundabout or in a car park, how many, which one clip, and
what the time of day was (questions.py). Every answer is from the source's CSV; the scenario is the names of a `Scenario`, as functions
(DESIGN 5), over the shared builder (realdump).

    recipe = v2_dashcam.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = v2_dashcam.build(recipe, folder, 3)  # the clips remuxed without metadata, with their times, and the questions

Scales: `tiny`, 30 clips (about 0.1 GB, for the tests and a smoke run); `standard`, every clip but the outliers and the crashes and near
misses the absent questions ask about (107 clips, about 0.5 GB). The source has no recording time (the dates in its names are the days its
clips were downloaded, and the dashcams' own clocks differ): the clips are named as a Viofo dashcam names them
(`2025_1201_081530_001.MP4`, under `DCIM/Movie`) and their times are a seeded spread over the six weeks from 1 December 2025, drawn without
regard to the clip, so that a clip's time says nothing of its class or of whether it is day. No question rests on a time."""

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
from mmsb.scenarios.v2_dashcam.questions import CODE, SOURCE_NAME, held_out, questions_of
from mmsb.scenarios.v2_dashcam.source import SOURCE, VALUES, Clip, mp4_header, read_clips, remux, seconds_of

name = "v2_dashcam"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Dashcam event clips", "video", "home",
              "staged: about a hundred short clips of crashes, near misses and ordinary drives, cut from two YouTube compilations of many people's dashcams "
              "and presented as one card, with invented dates and times")
filing = with_absent({"crashes": ("find_all", "sight"), "near": ("find_all", "sight"), "count": ("how_many", "sight"),
                      "which": ("find", "sight"), "did": ("did", "sight"), "what": ("what", "sight")})

SCALES = {"tiny": ("2025-12-01", "2025-12-14"), "standard": ("2025-12-01", "2026-01-11")}  # the days the clips are made over
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "DCIM/Movie"
KINDS_TINY = {"Near Crash": 10, "Normal": 3}  # of what is left, the clips of each class the tiny card holds (it holds every crash)
GAP_S = 90  # at least this long between the end of a clip and the next one's time
WHAT = "the memory card of my dashcam"
LAYOUT = (f"{FOLDER} holds its clips, one video each, named for the moment it was saved (YYYY_MMDD_HHMMSS_NNN.MP4); a clip's modified time is "
          "that moment. A clip is a few seconds to a minute of the road ahead, with no sound.")
SOURCE_NAMES = ("qutegocentric", "australian", "dashcam_driving", "youtube", "ego", "centric")  # the source's names, which no file of the dump may hold
COMMAND_WORDS = ("qutegocentric", "australian_roads", "video_metadata", "aus-dashcam")  # and which no command may
LABEL_WORDS = ("crash", "crashes", "near miss", "accident", "collision", "outlier", "hazard")  # the plain words the labels are said in


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days the
    clips are made over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def card_of(clips: Sequence[Clip], scale: str) -> list[Clip]:
    """The clips of the card at `scale`, by name: the source but for what `held_out` says; at `tiny`, every crash and a seeded 10 near misses
    and 3 ordinary drives of what is left. It does not depend on the seed that draws the questions."""
    left = sorted((c for c in clips if not held_out(c)), key=lambda c: c.name)
    if scale == "standard":
        return left
    rng = random.Random("v2:tiny:card")
    return sorted([c for c in left if c.kind == "Crash"] + rng.sample([c for c in left if c.kind == "Near Crash"], KINDS_TINY["Near Crash"])
                  + rng.sample([c for c in left if c.kind == "Normal"], KINDS_TINY["Normal"]), key=lambda c: c.name)


def spread(lengths: Sequence[float], days: Sequence[str], rng: random.Random) -> list[float]:
    """The times (epoch seconds, in order) of clips that play `lengths`: seeded draws of a day and a time between 6:00 and 22:00, each moved
    later when it would be within GAP_S of the end of the clip before."""
    draws = sorted(datetime.combine(date.fromisoformat(rng.choice(days)), time(6)).timestamp() + rng.uniform(0, 16 * 3600) for _ in lengths)
    out, free = [], 0.0
    for draw, length in zip(draws, lengths, strict=True):
        out.append(float(max(round(draw), free)))
        free = out[-1] + length + GAP_S
    return out


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed, days = str(recipe["scale"]), int(recipe["seed"]), recipe["days"]
    card = card_of(read_clips(str(recipe["source"])), scale)
    random.Random(f"v2:{scale}:order").shuffle(card)  # the order the clips were saved in: no link to class or to the source's own order
    times = spread([c.seconds for c in card], days, random.Random(f"v2:{scale}:times"))
    files = tuple(File(f"{FOLDER}/{datetime.fromtimestamp(t):%Y_%m%d_%H%M%S}_{n % 1000:03d}.MP4", SOURCE_NAME, t, "clip", "video",
                       {"src": c.rel}, length=c.seconds) for n, (c, t) in enumerate(zip(card, times, strict=True), 1))
    now = realdump.local_t(datetime.combine(datetime.fromtimestamp(times[-1]).date() + timedelta(days=1), time(9)))
    return files, questions_of(card, [f.path for f in files], scale, seed, now, files[0].t, files[-1].t + files[-1].length)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the card into `out` (new or empty): each clip remuxed by stream copy, without metadata, with its time; `workers` at once.
    Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        remux(source / str(f.spec["src"]), path)
        if abs(seconds_of(path) - f.length) > 0.05:
            raise ValueError(f"{f.spec['src']} plays {seconds_of(path)} s once remuxed, not the {f.length} s it did")

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no clip's name or metadata names a class or value of the CSV, or the source."""
    labels = [*(v for values in VALUES.values() for v in values), "Crash", "Near Crash", "Normal", "Outliers", *SOURCE_NAMES, *LABEL_WORDS]
    return realdump.probe(dump, out, labels, {**realdump.METADATA, ".mp4": mp4_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The card itself: nothing is cut (every question is asked after its last clip)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
