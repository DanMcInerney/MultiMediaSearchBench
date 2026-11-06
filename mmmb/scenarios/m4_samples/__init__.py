"""M4, the sound samples: one person's folder of short sound clips, thousands of them, each a few seconds of something (a door, a dog, a
tune), in a few numbered folders under names that say nothing. The questions ask to find every clip of a sound, how many there are of
one, and of sounds that are not in the folder (questions.py). The clips are FSD50K's evaluation split; every answer is from its labels. The
scenario is the names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = m4_samples.recipe(scale="tiny", seed=1)       # the source, the scale's days, this machine's UTC offsets
    dump, questions = m4_samples.build(recipe, folder, 3)  # the clips copied in (samples only) with their times, and the questions

Scales: `tiny`, 240 clips (0.2 GB, for the tests and a smoke run); `standard`, the split but for the clips of the sounds that are asked
about, of which a few stay, and of the sounds easily taken for them (about 6,100 clips, 5.2 GB). The source gives no recording time: the
clips are numbered in the order of a seeded draw, and their times, a seeded spread over April (`tiny`) or April to September 2026
(`standard`), rise with the numbers, a minute at least after the end of the clip before."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.core.scenario import Access, ProbeOutcome, dump_key
from mmmb.scenarios import realdump
from mmmb.scenarios.m4_samples.questions import CODE, SOURCE_NAME, choose, questions_of
from mmmb.scenarios.m4_samples.source import (
    BYTE_RATE,
    CLIPS,
    SOURCE,
    read_labels,
    with_samples_only,
)
from mmmb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "m4_samples"
code = CODE
generator = 2  # bumped when the scenario makes something other than before

SCALES = {"tiny": ("2026-04-01", "2026-04-30"), "standard": ("2026-04-01", "2026-09-30")}  # the days the clips are made over
FOLDERS = {"tiny": 3, "standard": 10}  # the folders the clips are in
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
GAP_S = 60  # at least this long between the end of a clip and the next one's time: a citation of one never reaches another
WHAT = "my folder of sound clips"
LAYOUT = ("folder_01 onwards hold its clips, rec_00001.wav onwards in the order I saved them, one short sound each (WAV, 44.1 kHz mono, "
          "0.3 to 30 seconds); a clip's modified time is when I saved it.")
COMMAND_WORDS = ("fsd50k", "freesound", "audioset")  # the source's names


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the sounds, clip numbers, times and questions
    are drawn with, the days the clips are made over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def spread(lengths: Sequence[float], days: Sequence[str], rng: random.Random) -> list[float]:
    """The times (epoch seconds, in order) of clips that play `lengths`: seeded draws over the days, each moved later when it would be
    within GAP_S of the end of the clip before."""
    start = realdump.local_t(datetime.combine(date.fromisoformat(days[0]), time.min))
    end = realdump.local_t(datetime.combine(date.fromisoformat(days[-1]) + timedelta(days=1), time.min)) - 3600
    free, out = start, []
    for draw, length in zip(sorted(round(rng.uniform(start, end)) for _ in lengths), lengths, strict=True):
        out.append(float(max(draw, free)))
        free = out[-1] + length + GAP_S
    return out


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed, days = str(recipe["scale"]), int(recipe["seed"]), recipe["days"]
    roles, kept = choose(read_labels(str(recipe["source"]))[0], scale, seed)
    random.Random(f"m4:{seed}:{scale}:numbers").shuffle(kept)
    times = spread([c.seconds for c in kept], days, random.Random(f"m4:{seed}:{scale}:times"))
    files = tuple(File(f"folder_{n * FOLDERS[scale] // len(kept) + 1:02d}/rec_{n + 1:05d}.wav", SOURCE_NAME, t, "clip", "audio",
                       {"src": f"{CLIPS}/{c.id}.wav"}, length=c.seconds) for n, (c, t) in enumerate(zip(kept, times, strict=True)))
    now = realdump.local_t(datetime.combine(date.fromisoformat(days[-1]) + timedelta(days=1), time(9)))
    return files, questions_of(kept, files, roles, seed, now)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the folder into `out` (new or empty): each clip copied with its samples alone, and its time; `workers` at once. Returns the
    dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        wav = with_samples_only((source / str(f.spec["src"])).read_bytes())
        if (len(wav) - 44) / BYTE_RATE != f.length:
            raise ValueError(f"{f.spec['src']} plays {(len(wav) - 44) / BYTE_RATE} s, not the {f.length} s its size said")
        path.write_bytes(wav)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no clip's name or WAV header names a class of the vocabulary, or the dataset."""
    _, classes = read_labels(str(recipe["source"]))
    return realdump.probe(dump, out, [*classes, *COMMAND_WORDS])


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder itself: nothing is cut (every question is asked after its last clip)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, ("bench-data",))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
