"""V3, the football match clips: three matches of an amateur team, filmed by an automatic follow camera and kept as the thirty-second clips
it saved, a folder for each match day. The camera is a virtual one: the source (SoccerTrack v2) films each match with a fixed panoramic
camera, and a script crops a moving, zooming 16:9 picture out of it that follows the play, which is what these clips are. The three
matches are real university matches (two by day, one under floodlights); their match days and kick-off times here are made up. The questions
ask when a goal or a shot happened, how many there were, which clips have them, whether anyone scored in a stretch of the match, which goal
it went into, and whether a tackle was before a shot (questions.py). Every answer is from the source's labels; the scenario is the names of
a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = v3_match.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = v3_match.build(recipe, folder, 3)  # the clips copied as they are, with their times, and the questions

Scales: `tiny`, 25 clips of the first match (about 0.3 GB: the stretch round its first goal, for the tests and a smoke run); `standard`, every
clip of three matches (525 clips, 8.3 GB, which is over the 6 GB a standard dump is held to: a match is 2.8 GB of clips, so two matches would
be 5.5 GB). Each clip's time is that of the stretch of the match it holds, from kick-off (made up, in the folder of the match day):
`Football/2025-03-08/FC_183000.mp4` is the first 30 seconds of the first half, the second half's clips begin 65 minutes after kick-off,
and a clip the source dropped (it keeps a clip only with three events or more) is a hole in the day's files."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)
from mmsb.scenarios.v3_match.questions import CODE, SOURCE_NAME, pool_of, questions_of
from mmsb.scenarios.v3_match.source import CLIPS, DERIVED, HUMAN, SECONDS, SOURCE, Clip, mp4_header, read_clips

name = "v3_match"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Football match clips", "video", "fun", "the clips of three amateur matches in date folders, thirty seconds each, from a virtual follow camera")
filing = with_absent({"when": ("when", "sight"), "when_clip": ("when", "sight"), "goals": ("how_many", "sight"), "shots": ("how_many", "sight"),
                      "window": ("how_many", "sight"), "find": ("find", "sight"), "every": ("find_all", "sight"), "did": ("did", "sight"),
                      "before": ("order", "sight"), "side": ("who", "sight")})

MATCH_DAYS = ("2025-03-08", "2025-03-29", "2025-04-26")  # the three Saturdays of the matches, in the order of their ids
KICKOFFS = (time(18, 30), time(11, 0), time(14, 0))  # the first match is under floodlights, the others by day
SECOND_HALF_S = 65 * 60  # the second half's clips begin this long after kick-off
TINY_WINDOWS = 25  # the clips of the tiny pile: this many windows of the first match's first half, round its first goal
SCALES = {"tiny": MATCH_DAYS[:1], "standard": MATCH_DAYS}
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Football"
WHAT = "my season's football match clips"
LAYOUT = (f"{FOLDER} has a folder for each match day (YYYY-MM-DD); in it a clip is 30 seconds of the match from a follow camera, named for the "
          "time it begins (FC_HHMMSS.mp4), which is also the file's modified time. The clips of a half run on one after the other, but for "
          "a few that were not kept; the second half begins about an hour after kick-off.")
SOURCE_NAMES = ("soccertrack", "followcam", "algo-dev99", "pano_start")  # the source's names, which no file of the dump may hold
COMMAND_WORDS = ("soccertrack", "followcam", "algo-dev99", "labels-ball")  # and which no command may
LABEL_WORDS = ("scored", "scoring", "tackled", "penalty")  # the plain words the labels are said in


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the match days
    and this machine's UTC offsets at its first and last."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(date.fromisoformat(SCALES[scale][0]), date.fromisoformat(SCALES[scale][-1]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def card_of(clips: Sequence[Clip], scale: str) -> list[Clip]:
    """The clips of the pile at `scale`: every clip of the three matches, or at `tiny` the TINY_WINDOWS windows of the first match's first half
    that begin a third of the way before its first goal (the half's first windows when it has no goal)."""
    matches = sorted({c.match for c in clips})
    if len(matches) != len(MATCH_DAYS):
        raise ValueError(f"{name} is made of {len(MATCH_DAYS)} matches, and {len(matches)} have clips on disk: {', '.join(matches)}")
    if scale == "standard":
        return list(clips)
    half = [c for c in clips if c.match == matches[0] and c.half == 1]
    goals = [c.window for c in half if any(e.label == "goal" for e in c.events)]
    first = max(0, (goals[0] if goals else TINY_WINDOWS // 2) - TINY_WINDOWS // 3)
    return [c for c in half if first <= c.window < first + TINY_WINDOWS]


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    card = card_of(read_clips(str(recipe["source"])), scale)
    day = {m: i for i, m in enumerate(sorted({c.match for c in read_clips(str(recipe["source"]))}))}
    files = []
    for c in card:
        start = datetime.combine(date.fromisoformat(MATCH_DAYS[day[c.match]]), KICKOFFS[day[c.match]]) + timedelta(
            seconds=(SECOND_HALF_S if c.half == 2 else 0) + c.window * SECONDS)
        files.append(File(f"{FOLDER}/{start:%Y-%m-%d}/FC_{start:%H%M%S}.mp4", SOURCE_NAME, realdump.local_t(start), "clip", "video",
                          {"src": f"{CLIPS}/{c.name}"}, length=SECONDS))
    last = max(datetime.fromtimestamp(f.t) for f in files).date()
    now = realdump.local_t(datetime.combine(last + timedelta(days=1), time(9)))
    return tuple(files), questions_of(pool_of(card, files), scale, seed, now)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the clips into `out` (new or empty): each copied as it is, with its time; `workers` at once. Returns the dump and the
    questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        shutil.copyfile(source / str(f.spec["src"]), path)

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no clip's name or metadata names a class of the labels, or the source."""
    return realdump.probe(dump, out, [*HUMAN, *DERIVED, *SOURCE_NAMES, *LABEL_WORDS], {**realdump.METADATA, ".mp4": mp4_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The clips themselves: nothing is cut (every question is asked after the last match)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
