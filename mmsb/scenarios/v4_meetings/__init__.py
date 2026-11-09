"""V4, the work meetings: the recordings of a small company's design meetings (the AMI corpus's scenario meetings): three teams, each of four
people, each meeting four times in a day over a remote control's design, so twelve meetings of 20 to 45 minutes, a Corner camera's video with the
sound of the four of them in one MP4 each, named for when it began. The questions ask of what was said (when something was first mentioned, in
which meeting, who was the first to bring it up, what they decided and when), of how a meeting ran (who presented after whom, who went up to the
whiteboard and when) and of things never said (questions.py). Every answer is from the corpus's words and annotations; the scenario is the
names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = v4_meetings.recipe(scale="tiny", seed=1)      # the source, the meetings' days, this machine's UTC offsets
    dump, questions = v4_meetings.build(recipe, folder, 3)  # each meeting's video and sound muxed into an MP4, and the questions

Scales: `tiny`, the first meeting of two teams (ES2015a and ES2016a, 2 files, about 40 minutes); `standard`, all twelve (ES2002, ES2015 and
ES2016, a to d: about 7 hours, a few hundred MB). The meetings are on their real days (13 December 2004, 9 and 16 March 2005); their times of
day are drawn (source.py `start_times`: the corpus has none for these), the same for every seed, so a meeting's modified time is when it began
and the meetings of a day follow one another."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)
from mmsb.scenarios.realdump.speech import Talk
from mmsb.scenarios.v4_meetings.questions import CODE, FILING, NEVER, ROLES, SOURCE_NAME, TERMS, questions_of
from mmsb.scenarios.v4_meetings.source import (
    MEETINGS,
    SOURCE,
    mp4_header,
    mux,
    read_meetings,
    seconds_of,
    start_times,
)

name = "v4_meetings"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Work meetings", "video", "work", "twelve recorded design meetings of three teams, a corner camera and the sound of the room")
filing = FILING

SCALES = {"tiny": ("ES2015a", "ES2016a"), "standard": MEETINGS}  # the meetings each scale holds
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Meetings"
WHAT = "the recordings of my work meetings"
LAYOUT = (f"{FOLDER} holds one MP4 for each meeting, meeting_<date>_<time>.mp4, named for when it began (which is also the file's modified "
          "time): a camera in a corner of the room, and the sound of the people at the table. A team meets several times in a day, one meeting "
          "after another, so the second meeting of a day is the second file of that day.")
SOURCE_NAMES = ("ami", "amicorpus", "edinburgh", "scenario", "corner", "headset", "es2002", "es2015", "es2016")  # the corpus's names
COMMAND_WORDS = ("/ami", "ami/", "amicorpus", "es2002", "es2015", "es2016", "mix-headset", "corner.avi")


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days of the
    meetings and this machine's UTC offsets at the first and the last."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = sorted({m.day.isoformat() for m in read_meetings(str(source), SCALES[scale])})
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    source = str(recipe["source"])
    meetings = read_meetings(source, SCALES[str(recipe["scale"])])
    seconds = {m.id: seconds_of(source, m.id) for m in meetings}
    began = start_times(meetings, seconds)
    meetings = tuple(sorted(meetings, key=lambda m: began[m.id]))
    paths = {m.id: f"{FOLDER}/meeting_{began[m.id]:%Y%m%d_%H%M%S}.mp4" for m in meetings}
    files = tuple(File(paths[m.id], SOURCE_NAME, realdump.local_t(began[m.id]), "recording", "video", {"meeting": m.id}, length=seconds[m.id])
                  for m in meetings)
    talks = tuple(Talk(paths[m.id], began[m.id], seconds[m.id], m.cues()) for m in meetings)
    return files, questions_of(meetings, talks, str(recipe["scale"]), int(recipe["seed"]), realdump.local_t)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the meetings into `out` (new or empty): each one's video and sound muxed into an MP4 (H.264 and AAC), with its time; `workers` at
    once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)

    def write(f: File, path: Path) -> None:
        mux(str(recipe["source"]), str(f.spec["meeting"]), path)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no recording's name or metadata names the corpus, a team's meeting, a role or a word the questions ask of."""
    meetings = read_meetings(str(recipe["source"]), SCALES[str(recipe["scale"])])
    labels = [*SOURCE_NAMES, *ROLES.values(), *(t.search for t in (*TERMS, *NEVER)), *(t.name for m in meetings for t in m.topics)]
    return realdump.probe(dump, out, labels, {**realdump.METADATA, ".mp4": mp4_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The recordings themselves: nothing is cut (every question is asked after the last meeting)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
