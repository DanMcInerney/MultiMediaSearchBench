"""I2, the phone's screenshots: a Screenshots folder, thousands of screens of a phone in use, mostly the browser and the apps people open: a search
for flights, the weather somewhere, an app's page in the Play Store. The questions ask for the screens of what the person was doing, how many there
were, the screen that says a few words, what a screen read, and of tasks that are not in the folder (questions.py). The screens are Android in the
Wild's; every answer is from its labels, the goal of each episode and the text OCR read off each screen. The scenario is the names of a `Scenario`,
as functions (DESIGN 5), over the shared builder (realdump).

The folder is not one phone's: the source's episodes were done on six kinds of device (Pixel 3, 4, 4 XL, 5 and 6, and a custom one; the screens are 270
to 720 px wide), and they are all in this one folder, laid over the months as one person's, in their own sizes. An episode is one device's.

    recipe = i2_screenshots.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = i2_screenshots.build(recipe, folder, 3)  # the screenshots copied in, named by when they were taken, and the questions

Scales: `tiny`, about 15 episodes (some 120 screenshots, for the tests and a smoke run); `standard`, the source's 419 episodes but for the tasks
held out (about 3,100 screenshots, 0.45 GB). Every episode of a task a question can rest on is cut to the screens that show the task's words, the
others being left out (tasks.py `trim`), so that all the screens a `tasks` or `size` question counts show the task. The source has no time: the
episodes are laid over days at seeded times, 20 minutes or more apart, and the screens of each a few seconds apart, in the order of their steps.
The same recipe gives the same folder at every seed: the seed draws other questions from it."""

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
from mmsb.scenarios.i2_screenshots.questions import CODE, READS, SOURCE_NAME, questions_of
from mmsb.scenarios.i2_screenshots.source import SOURCE, Episode, png_header, png_plain, read_labels
from mmsb.scenarios.i2_screenshots.tasks import OTHERS, Task, asked_tasks, trim
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i2_screenshots"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Phone screenshots", "image", "home", "one Screenshots folder that mixes the screens of six kinds of phone (Pixel 3, 4, 4 XL, 5 and 6 and a custom "
                                                     "device): thousands of screens of the browser and the apps in use")
filing = with_absent({"tasks": ("find_all", "text"), "size": ("how_many", "text"), "did": ("did", "text"), "words": ("find", "text"),
                      "read": ("what", "text")})

SCALES = {"tiny": ("2026-10-05", "2026-10-11"), "standard": ("2026-05-04", "2026-10-30")}  # the days the screenshots are taken over
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Pictures/Screenshots"
WHAT = "my phone's Screenshots folder"
LAYOUT = (f"{FOLDER} holds its screenshots, Screenshot_YYYYMMDD-HHMMSS.png, named by the day and time I took them, in the order I took them; "
          "a file's modified time is when I took it.")
COMMAND_WORDS = ("android_in_the_wild", "android-in-the-wild", "aitw", "androidinthewild")  # the source's names
DAY = (time(7), time(23))  # the hours of a day a phone is used in
EPISODE_GAP_S = 20 * 60  # at least this long between the last screenshot of an episode and the first of the next
STEP_S = (3, 15)  # the seconds between two screenshots of an episode
TINY_TASKS = 3  # the tiny folder's tasks that its questions are asked of (one episode each)
TINY_HELD = 2  # the tasks held out of it, for its absent questions
TINY_FILLERS = 9  # the other episodes it holds, for a haystack


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days the
    screenshots are taken over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def choose(episodes: Sequence[Episode], scale: str) -> tuple[list[Episode], list[Task]]:
    """The episodes the folder holds at `scale` and the tasks held out of it. The tasks that have no screen anywhere else to be taken for them
    are held out of the standard folder, all of them: the absent questions ask of them. The tiny folder is a seeded few episodes: some tasks to ask
    of, those of the `read` question it asks, a few held-out tasks' absence and others for a haystack."""
    read_in = {r.episode for r in READS}
    held = [t for t, _ in asked_tasks(episodes, 0, held=True) if not read_in & {e.id for e in t.episodes}]
    gone = {e.id for t in held for e in t.episodes}
    rest = trim([e for e in episodes if e.id not in gone], read_in)  # the episodes the written questions are of are left whole
    if scale == "standard":
        return rest, held
    rng = random.Random("i2:tiny")
    asked = [t for t, _ in asked_tasks(rest, OTHERS) if len(t.episodes) == 1]
    taken = {e.id for t in rng.sample(asked, TINY_TASKS) for e in t.episodes} | {r.episode for r in READS if r.tiny}
    fillers = rng.sample([e for e in rest if e.id not in taken], TINY_FILLERS)
    keep = taken | {e.id for e in fillers}
    return [e for e in rest if e.id in keep], rng.sample(held, TINY_HELD)


def layout(pile: Sequence[Episode], days: Sequence[str]) -> dict[str, float]:
    """The time (epoch seconds) of each screenshot of `pile` (by `Shot.file`): the episodes at seeded times of the days, in a seeded order, EPISODE_GAP_S
    apart at least, and each one's screenshots a few seconds apart in the order of their steps."""
    rng = random.Random("i2:layout")
    order = sorted(pile, key=lambda e: e.id)
    steps = {e.id: [rng.randint(*STEP_S) for _ in e.shots[1:]] for e in order}
    rng.shuffle(order)
    first = date.fromisoformat(days[0])
    window = (DAY[1].hour - DAY[0].hour) * 3600
    draws = sorted((rng.randrange(len(days)), rng.randrange(window)) for _ in order)  # a day of `days` and seconds into the hours of DAY
    out: dict[str, float] = {}
    free = realdump.local_t(datetime.combine(first, DAY[0]))
    for episode, (day, within) in zip(order, draws, strict=True):
        wall = datetime.combine(first + timedelta(days=day), DAY[0]) + timedelta(seconds=within)
        t = max(float(round(realdump.local_t(wall))), free)
        for shot, gap in zip(episode.shots, [0, *steps[episode.id]], strict=True):
            t += gap
            out[shot.file] = t
        free = t + EPISODE_GAP_S
    return out


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    episodes = read_labels(str(recipe["source"]))
    pile, held = choose(episodes, scale)
    times = layout(pile, recipe["days"])
    shots = sorted((s for e in pile for s in e.shots), key=lambda s: times[s.file])
    paths = {s.file: f"{FOLDER}/Screenshot_{datetime.fromtimestamp(times[s.file]):%Y%m%d-%H%M%S}.png" for s in shots}
    if len(set(paths.values())) != len(paths):
        raise ValueError("two screenshots have one time to the second")
    files = tuple(File(paths[s.file], SOURCE_NAME, times[s.file], "screenshot", "image", {"src": s.file}) for s in shots)
    last = datetime.fromtimestamp(times[shots[-1].file]).date()
    if last.isoformat() > recipe["days"][-1]:
        raise ValueError(f"a screenshot is taken on {last}, after the recipe's last day {recipe['days'][-1]}")
    now = realdump.local_t(datetime.combine(last + timedelta(days=1), time(9)))
    return files, questions_of(pile, held, episodes, paths, now, times, scale, seed)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the folder into `out` (new or empty): each screenshot copied with its time, which no other metadata of it holds; `workers` at
    once. Returns the dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        path.write_bytes(png_plain((source / str(f.spec["src"])).read_bytes()))

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no screenshot's name or PNG header names a task of the source, a device of it, or the dataset."""
    episodes = read_labels(str(recipe["source"]))
    labels = {e.goal for e in episodes} | {e.device for e in episodes}
    return realdump.probe(dump, out, [*sorted(labels), *COMMAND_WORDS], {**realdump.METADATA, ".png": png_header})


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder itself: nothing is cut (every question is asked after its last screenshot)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
