"""A8, the music library: one person's music, three thousand tracks of independent artists in about a hundred album folders, as 30-second MP3
excerpts with their tags taken off (the Free Music Archive's `fma_small`). The questions ask to find every track of a genre in a folder, how
many there are, whether a folder has one and which is the first, which folder has the most of a genre, and of a genre or a folder that has
none (questions.py). **The genre of a track is the one its uploader gave it**, a coarse label that a listener may not share; no title or artist
is asked, as the tags that held them are gone. Every answer is from the archive's labels; the scenario is the names of a `Scenario`, as functions
(DESIGN 5), over the shared builder (realdump).

    recipe = a8_music.recipe(scale="tiny", seed=1)      # the source, the span of the upload times, this machine's UTC offsets
    dump, questions = a8_music.build(recipe, folder, 3)  # the tracks copied in without their tags, with their times, and the questions

Scales: `tiny`, about 150 tracks in ten folders or so (0.15 GB, for the tests and a smoke run); `standard`, about 3,000 tracks in about 100 folders
(2.9 GB, 25 hours). Each folder is whole albums of the archive's dataset (the albums of `tracks.csv` that `fma_small` has three tracks or more
of, of the genres the library has), in a seeded order; each file's modified time is the real time its track was uploaded to the archive, from 2008
to 2017."""

from __future__ import annotations

import dataclasses
import io
import threading
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from pathlib import Path

import av

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.a8_music.questions import ASKED, CODE, NEAR, SOURCE_NAME, choose, questions_of
from mmsb.scenarios.a8_music.source import CLIP_S, CLIPS, SOURCE, Track, frames_only, member, mp3_header, read_tracks
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "a8_music"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Music library", "audio", "fun", "thousands of short tracks of independent artists, in album folders, with their tags taken off")
filing = with_absent({"genre": ("find_all", "sound"), "count": ("how_many", "sound"), "first": ("did", "sound"), "most": ("who", "sound")})

DAYS = ("2008-11-25", "2017-03-24")  # the first and last day any track of `fma_small` was uploaded: the span of the files' times
SCALES = ("tiny", "standard")
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Music"
WHAT = "my music library"
LAYOUT = (f"{FOLDER} holds my albums, one folder each (Album 001 onwards), with the tracks in them as track_0001.mp3 onwards, numbered through the whole "
          "library in the order of the folders (MP3, a 30-second excerpt of each track, with no tags); a file's modified time is when I added it.")
SOURCE_NAMES = ("fma", "freemusicarchive", "mdeff")  # the source's names, which no file of the dump holds (not the words of "free music archive": `Music` is the folder)
COMMAND_WORDS = ("fma_", "/fma", "fma/", "freemusicarchive", "free music archive", "mdeff")  # and no command


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the genres, albums, folders and questions are
    drawn with, the first and last day of the upload times and this machine's UTC offsets on them."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = list(DAYS)
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    roles, folders = choose(read_tracks(str(recipe["source"])), scale, seed)
    made: list[list[tuple[Track, File]]] = []
    n = 0
    for k, folder in enumerate(folders, 1):
        made.append([])
        for t in folder:
            n += 1
            made[-1].append((t, File(f"{FOLDER}/Album {k:03d}/track_{n:04d}.mp3", SOURCE_NAME, realdump.local_t(t.created), "track", "audio",
                                     {"src": member(t.id)}, length=CLIP_S)))
    newest = max(t.created for folder in folders for t in folder)
    now = realdump.local_t(datetime.combine(newest.date() + timedelta(days=1), time(9)))
    return tuple(f for folder in made for _, f in folder), questions_of(made, roles, scale, seed, now)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the library into `out` (new or empty): each track's clip copied without its tags, and its time; `workers` at once. Returns the
    dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))
    local = threading.local()  # a zip file is read by one thread at a time: one handle for each
    plays: dict[str, float] = {}

    def write(f: File, path: Path) -> None:
        if not hasattr(local, "zip"):
            local.zip = zipfile.ZipFile(source / CLIPS)
        data = frames_only(local.zip.read(str(f.spec["src"])))
        with av.open(io.BytesIO(data)) as clip:
            plays[f.path] = clip.duration / 1e6
        if plays[f.path] < CLIP_S:
            raise ValueError(f"{f.spec['src']} plays {plays[f.path]:.4f} s, less than the {CLIP_S} s of a clip")
        path.write_bytes(data)

    entries = tuple(dataclasses.replace(e, length=plays[e.path]) for e in realdump.write(files, out, workers, write))
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def label_words() -> list[str]:
    """The words the probe looks for in a track's name and metadata: the archive's genres, as the questions say them and as the dataset does,
    and the source's names."""
    return [*(g for g in (*ASKED, "Pop", "Experimental", "International", "Instrumental")), "hip hop", *SOURCE_NAMES,
            *(n for near in NEAR.values() for n in near)]


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no track's name or the start and end of its MP3 names a genre of the archive or the archive."""
    return realdump.probe(dump, out, label_words(), {".mp3": mp3_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The library itself: nothing is cut (every question is asked after the newest track was added)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
