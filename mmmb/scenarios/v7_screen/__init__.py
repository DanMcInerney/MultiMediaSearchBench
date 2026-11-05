"""V7, the screen scenario: a synthetic week of an owner's phone, screen, front door camera, hallway microphone, notes, config and logs,
with needles hidden among distractors (plan.py has the week and its questions, media_*.py draw and play it, write.py writes it). This
module is the scenario (DESIGN 5): the names a `Scenario` has, as functions.

    recipe = v7_screen.recipe(scale="tiny", seed=1)            # what the week is made from: the same recipe is the same bytes
    dump, questions = v7_screen.build(recipe, folder, workers)  # the files, their manifest, and the questions with their evidence
    plan_hash, files_hash = v7_screen.hashes(recipe)            # what a run records, to refuse a changed generator

The bytes of the files depend on the libraries (pyproject pins PyAV, numpy, Pillow and imageio-ffmpeg) and on the machine's time zone:
the plan's times are local. The recipe records the UTC offsets of its days, and a recipe is refused in a time zone that gives others."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.core.render import markdown
from mmmb.core.report import numbers
from mmmb.core.runfolder import IN_ONLY, RunFolder
from mmmb.core.scenario import Access, ProbeOutcome, dump_key, files_hash, plan_hash
from mmmb.scenarios.v7_screen import media_check  # noqa: F401  (importing it checks the constants of the media modules against each other)
from mmmb.scenarios.v7_screen import week
from mmmb.scenarios.v7_screen.manifest import LOGS, entries_of, file_rows, measure_disk
from mmmb.scenarios.v7_screen.plan import SCALES, VERSION, Plan, days_before
from mmmb.scenarios.v7_screen.prompt import prompt  # noqa: F401  (the Scenario's `prompt`)
from mmmb.scenarios.v7_screen.questions import CODE, questions_of
from mmmb.scenarios.v7_screen.shortcuts import run as run_shortcuts
from mmmb.scenarios.v7_screen.write import WORKERS, write

name = "v7_screen"
code = CODE
generator = VERSION  # bumped when the plan makes something other than before

# What no command of an agent with its own tools may hold, beyond the folders: the generator of the week and its truth, as code reaches
# them (a mention in prose is no escape). An agent that runs `haystack.plan(seed, scale)` from the repository has every answer. The
# quotes are as `reached` reads a command: a `"` after a backslash is `/"`. The first eight are the words of the old generator, which
# copies of it still have; the rest are the new one's. A bare `mmmb` is not one: the folders of the run are in `D:\mmmb-data` and `D:\mmmb-runs`.
OLD_WORDS = ("haystack.py", "import haystack", "from haystack", "haystack.plan", "['truth']", '[/"truth/"]', "get('truth'", 'get(/"truth/"')
NEW_WORDS = ("v7_screen", "from mmmb", "import mmmb")
RECIPE_KEYS = {"scale", "seed", "days", "utc_offsets"}


def utc_offsets(days: Sequence[date]) -> list[str]:
    """This machine's offset from UTC (`-0500`) at the local midnight that starts each day and at the one that ends the last."""
    if not days:
        raise ValueError("a week has days")
    return [datetime.combine(day, time.min).astimezone().strftime("%z") for day in (*days, days[-1] + timedelta(days=1))]


def recipe(*, scale: str, seed: int, days: Sequence[date | str] | None = None) -> dict:
    """What the week is made from: its `scale`, `seed` and `days` (dates or ISO dates, oldest first; by default the days of the scale
    ending yesterday) and the UTC offsets of the days in this time zone."""
    if scale not in SCALES:
        raise ValueError(f"a scale is one of {', '.join(SCALES)}, not {scale!r}")
    chosen = days_before(date.today(), SCALES[scale].days) if days is None else [date.fromisoformat(d) if isinstance(d, str) else d
                                                                                for d in days]
    return {"scale": scale, "seed": seed, "days": [d.isoformat() for d in chosen], "utc_offsets": utc_offsets(chosen)}


def plan_of(recipe: Mapping[str, object]) -> Plan:
    """The plan a recipe makes. Refused when the recipe has other keys than `recipe` makes, or was made in a time zone whose offsets
    for its days are not this machine's: the plan, and the bytes, would be other than the recipe's."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a v7_screen recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    days = [date.fromisoformat(d) for d in recipe["days"]]
    if list(recipe["utc_offsets"]) != utc_offsets(days):
        raise ValueError(f"the recipe was made where the UTC offsets of its days are {recipe['utc_offsets']}, here they are "
                         f"{utc_offsets(days)}: the times of the week are local, so it is another week; make a new recipe")
    return week.plan(recipe["seed"], recipe["scale"], days)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    """The `plan_hash` of the questions the recipe makes and the `files_hash` of its files and logs (about a second; no file is written)."""
    made = plan_of(recipe)
    return plan_hash(made.questions), files_hash(file_rows(made), made.logs)


def run_fields(recipe: Mapping[str, object]) -> dict:
    """V7 adds nothing to run.json beyond what every scenario's recipe gives."""
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = WORKERS) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the week the recipe makes into `out`, which is new or empty, `workers` files at once, and returns its dump (the manifest
    measured from the files written) and its questions. The files are neither frozen nor made read-only: that is the cache's."""
    made = plan_of(recipe)  # a wrong number of days fails here, before anything is written
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: a dump is written into a folder of its own")
    write(made, out, workers)
    entries = entries_of(made, measure_disk(out))
    return Dump(dump_key(name, recipe, generator), out, name, entries, (LOGS,)), questions_of(made, entries)


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The shortcut probe over the written week (shortcuts.py), reporting `shortcuts.json` and `shortcuts.md` in `out`."""
    failures = run_shortcuts(plan_of(recipe), dump.root, out, ffmpeg=ffmpeg, seed=seed)
    return ProbeOutcome("failed" if failures else "passed", tuple(failures))


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder an agent with its own tools reads the recordings in: the dump itself, the same for every question and slot."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    """What no command of the agent in a slot may hold, but for the run's folders: see OLD_WORDS and NEW_WORDS."""
    return tuple(IN_ONLY + word for word in (*OLD_WORDS, *NEW_WORDS))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    """Nothing is added to a result: V7 cuts the data for no question."""
    return {}


def release(temp: Path, key: str) -> None:
    """V7 makes nothing in the temp folder that a run puts away: the dump is read-only and stays for the next run."""


def report(run: RunFolder) -> tuple[dict, str]:
    """The figures of the run and the Markdown that renders them (report.json and report.md): the core's report, for one run."""
    figures = numbers([run])
    return figures, markdown(figures)

