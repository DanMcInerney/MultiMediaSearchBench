"""A3, the research group's meetings: the weekly meetings of a speech-research group at a lab in Berkeley (the ICSI Meeting Recorder group), ten of
them from February 2000 to March 2001, sound only: one WAV of the room (everyone's headset mixed) for each, named for when it began. The questions
ask of what was said (when something was first mentioned, in which meetings, how many, what came up after what, what they said a number or a tool was)
and of what the group does every week (when they read out their digit strings), and of things never said (questions.py). Every answer is from the
corpus's transcripts and topic annotation; no question names or asks for a person, since the speakers are real people. The scenario is the names of a
`Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = a3_meetings.recipe(scale="tiny", seed=1)      # the source, the meetings' days, this machine's UTC offsets
    dump, questions = a3_meetings.build(recipe, folder, 3)  # each meeting's WAV copied in with its time, and the questions

Scales: `tiny`, two meetings (Bmr005 and Bmr007, about 2.2 hours, 250 MB); `standard`, the ten (about 10 hours, 1.1 GB). The meetings are on their
real days and times (the transcripts' own `DateTimeStamp`), so the modified time of a file is when its meeting began."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf
from mmsb.scenarios import realdump
from mmsb.scenarios.a3_meetings.questions import CODE, FILING, NEVER, SOURCE_NAME, TERMS, questions_of
from mmsb.scenarios.a3_meetings.source import MEETINGS, SOURCE, read_meetings, seconds_of
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)
from mmsb.scenarios.realdump.speech import Talk

name = "a3_meetings"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Research group meetings", "audio", "work", "ten weekly meetings of one research group over thirteen months, sound only")
filing = FILING

SCALES = {"tiny": ("Bmr005", "Bmr007"), "standard": MEETINGS}  # the meetings each scale holds
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Weekly"
WHAT = "the recordings of my research group's weekly meetings"
LAYOUT = (f"{FOLDER} holds one WAV for each meeting, rec_<date>_<time>.wav, named for when it began (which is also the file's modified time): "
          "the sound of the room as one channel (16 kHz), with everyone's voices in it. The group meets once a week, so no two files are of one day.")
SOURCE_NAMES = ("icsi", "bmr", "berkeley", "interaction", "meeting recorder", "mrt", "nxt")  # the corpus's names
COMMAND_WORDS = ("/icsi", "icsi/", "bmr0", "interaction.wav", "icsi_original", "icsi_plus")
WAV_WORDS = {"data", "list", "wave", "riff", "fmt"}  # what a WAV's header says of itself (its chunks' names): no label, though a topic is named "data collection"


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days of the
    meetings and this machine's UTC offsets at the first and the last."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = sorted({m.start.date().isoformat() for m in read_meetings(str(source), SCALES[scale])})
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    source = str(recipe["source"])
    meetings = tuple(sorted(read_meetings(source, SCALES[str(recipe["scale"])]), key=lambda m: m.start))
    seconds = {m.id: seconds_of(source, m.id) for m in meetings}
    paths = {m.id: f"{FOLDER}/rec_{m.start:%Y%m%d_%H%M}.wav" for m in meetings}
    files = tuple(File(paths[m.id], SOURCE_NAME, realdump.local_t(m.start), "recording", "audio", {"meeting": m.id}, length=seconds[m.id]) for m in meetings)
    talks = tuple(Talk(paths[m.id], m.start, seconds[m.id], m.cues()) for m in meetings)
    return files, questions_of(meetings, talks, str(recipe["scale"]), int(recipe["seed"]), realdump.local_t)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the meetings into `out` (new or empty): each WAV copied as it is, with its time; `workers` at once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        path.write_bytes((source / "signals" / f"{f.spec['meeting']}.interaction.wav").read_bytes())

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no recording's name or WAV header names the corpus or a word the questions ask of."""
    meetings = read_meetings(str(recipe["source"]), SCALES[str(recipe["scale"])])
    topics = (" ".join(w for w in t.name.split() if w.casefold() not in WAV_WORDS) for m in meetings for t in m.topics)
    return realdump.probe(dump, out, [*SOURCE_NAMES, *(t.search for t in (*TERMS, *NEVER)), *topics])


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
