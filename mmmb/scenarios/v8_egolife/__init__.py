"""V8, the EgoLife scenario: EgoLifeQA (the public benchmark of the first-person days of EgoLife: Jake's glasses, 30 s clips, and the
transcript of what was said around him) asked three ways: without the recordings (`blind`, the floor), with the agent's own tools (`files`)
and with a system's tools. The download is read here and never run; this module is the scenario (DESIGN 5): the names a `Scenario` has, as
functions.

    recipe = v8_egolife.recipe(source=DOWNLOAD, seed=1)           # what the run is made from: the download, the days, the clip table
    dump, questions = v8_egolife.build(recipe, folder, workers)    # the clips linked, the transcript written, the manifest, the questions
    plan_hash, files_hash = v8_egolife.hashes(recipe)               # what a run records, to refuse a changed plan or download

Which questions: for each of four ages, 10 of the questions that need no sound, and 10 that need sound, asked from the transcript alone
(choose.py). Every question is asked as of its horizon H, the end of the last whole clip that ended by the query: no configuration may
read a clip or a transcript line past H. A system is cut to H by the system (`Question.horizon`); a configuration with its own tools reads a
folder of the clips and lines up to H (view.py).

The times of the plan are local, so the recipe records the UTC offset of its days and is refused in a zone with another; the days must not
cross a change of the offset (plan.check_days). `run.json` also holds `pools` (`pool_counts`) and `decode_s`, which the report quotes.
There is no shortcut probe: a public dataset, so the blind configuration is the floor; `probe` says `none`, as the recorded run did."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path

from mmmb.core.agents.common import command_outputs
from mmmb.core.config import CONTROL
from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.core.runfolder import IN_ONLY, RunFolder
from mmmb.core.scenario import Access, ProbeOutcome, dump_key, files_hash, plan_hash
from mmmb.scenarios.v8_egolife import plan as plans  # (the package's own `plan` is this module: the scenario has no `plan` function)
from mmmb.scenarios.v8_egolife.choose import PER_AGE, SPEECH, pools
from mmmb.scenarios.v8_egolife.manifest import TRANSCRIPT, entries_of, file_rows, measure_disk, transcript_path
from mmmb.scenarios.v8_egolife.plan import SCALE, TEXT, VIDEO, Plan, check_days, check_transcript, default_days, load_oracles
from mmmb.scenarios.v8_egolife.prompt import prompt  # noqa: F401  (the Scenario's `prompt`)
from mmmb.scenarios.v8_egolife.questions import CODE, questions_of
from mmmb.scenarios.v8_egolife.report import markdown, numbers
from mmmb.scenarios.v8_egolife.source import PERSON, clip_of, clip_table, read_items, read_transcripts
from mmmb.scenarios.v8_egolife.view import files_view, release_views, seen_past_horizon, view_dir

name = "v8_egolife"
code = CODE
generator = 1  # bumped when the plan makes something other than before

DOWNLOAD = Path(r"D:\bench-data\egolife")  # where the download is, read only: the folders A1_JAKE, EgoLifeQA and EgoLifeCap
WORKERS = 3  # clips measured, and files hashed, at once

# What no command of an agent with its own tools may hold, beyond the folders: the old generator's words, which the old `EgoRun.deny`
# inherited from V7's (a copy of V7's OLD_WORDS: a scenario does not import another), the code that reaches this scenario, the download
# and the folders of the other slots' recordings. The quotes are as `reached` reads a command: a `"` after a backslash is `/"`.
OLD_WORDS = ("haystack.py", "import haystack", "from haystack", "haystack.plan", "['truth']", '[/"truth/"]', "get('truth'", 'get(/"truth/"')
NEW_WORDS = ("v8_egolife", "from mmmb", "import mmmb")
DOWNLOAD_WORDS = ("bench-data", PERSON.lower(), IN_ONLY + "egolife")  # the download's: `egolife` only in a command (the transcript says it once)
RECIPE_KEYS = {"source", "scale", "seed", "days", "per_age", "speech", "utc_offset", "clips"}


def utc_offset_text(day: date) -> str:
    """This machine's offset from UTC (`-0400`) at the local midnight that starts `day`."""
    return datetime.combine(day, time.min).astimezone().strftime("%z")


def recipe(*, source: Path | str = DOWNLOAD, seed: int, days: Sequence[date | str] | None = None, per_age: int = PER_AGE,
           speech: int = SPEECH, workers: int = WORKERS) -> dict:
    """What the run is made from: the download at `source`, the `seed`, the `days` (dates or ISO dates, oldest first, which cross no
    change of the UTC offset; by default the latest three that do not, ending yesterday), the questions of each age and for speech, the
    UTC offset of the days in this time zone and the clip table of the download (each clip's name and length, which reading takes
    seconds a clip, `workers` at once: the plan then needs no clip file)."""
    chosen = default_days(date.today()) if days is None else tuple(date.fromisoformat(d) if isinstance(d, str) else d for d in days)
    if not chosen:
        raise ValueError("a run has days")
    check_days(chosen)
    clips = clip_table(Path(source), len(chosen), workers)
    return {"source": str(source), "scale": SCALE, "seed": seed, "days": [d.isoformat() for d in chosen], "per_age": per_age, "speech": speech,
            "utc_offset": utc_offset_text(chosen[0]), "clips": [[c.rel, c.length] for c in clips]}


def plan_of(recipe: Mapping[str, object]) -> Plan:
    """The plan a recipe makes. Refused when the recipe has other keys than `recipe` makes, when its days are not consecutive or cross
    a change of the UTC offset, or when it was made where the offset of its days is not this machine's (the plan's times are local, so
    it would be another run); the download must be at the recipe's source."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"a v8_egolife recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    days = tuple(date.fromisoformat(d) for d in recipe["days"])
    check_days(days)
    if recipe["utc_offset"] != utc_offset_text(days[0]):
        raise ValueError(f"the recipe was made where the UTC offset of its days is {recipe['utc_offset']}, here it is "
                         f"{utc_offset_text(days[0])}: the times of the plan are local, so it is another run; make a new recipe")
    source = Path(recipe["source"])
    if not source.is_dir():
        raise FileNotFoundError(f"the EgoLife download is not at {source}: the plan is made from its questions and transcripts")
    clips = [clip_of(rel, length) for rel, length in recipe["clips"]]
    if any(b.t0 < a.t0 for a, b in zip(clips, clips[1:])):
        raise ValueError("the recipe's clips are not in time order")
    return plans.plan(source, recipe["seed"], days, clips, load_oracles(), recipe["per_age"], recipe["speech"])


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    """The `plan_hash` of the questions the recipe makes and the `files_hash` of its files and transcript (a second; no file is written)."""
    made = plan_of(recipe)
    return plan_hash(made.questions), files_hash(file_rows(made), made.logs)


def pool_counts(recipe: Mapping[str, object]) -> dict:
    """The questions the bench could ask before choosing (`run.json`'s `pools`): `video` and `speech` counts, and `dropped`, how many of
    the others for each reason."""
    source, days = Path(recipe["source"]), len(recipe["days"])
    video, heard, dropped = pools(read_items(source), [clip_of(rel, length) for rel, length in recipe["clips"]],
                                  read_transcripts(source, days), days)
    return {"video": len(video), "speech": len(heard), "dropped": dict(dropped)}


def run_fields(recipe: Mapping[str, object]) -> dict:
    """What run.json holds beside the recipe's own: the pools the questions were chosen from (`pool_counts`), which the report quotes."""
    return {"pools": pool_counts(recipe)}


def build(recipe: Mapping[str, object], out: Path, workers: int = WORKERS) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the dump the recipe makes into `out`, which is new or empty: the clips as hardlinks to the download under neutral names
    (`glasses/<date>/HH.MM.SS.mp4`) and the transcript as JSON lines (`transcript/<date>.jsonl`); `workers` files are hashed at once.
    Returns the dump (the manifest measured from the files written) and the questions. The files are neither frozen nor made
    read-only: that is the cache's (core/cache.py writes it into `<temp>/mmb-<key>-data.part`, so the temp folder must be on the download's
    volume: `mmmb build --temp`). `out` must be on the download's volume."""
    made = plan_of(recipe)  # a wrong number of days, or a transcript an exec could not print, fails here, before anything is written
    check_transcript(made.logs, DOWNLOAD_WORDS)
    out, source = Path(out), Path(recipe["source"])
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: a dump is written into a folder of its own")
    for f in made.files:
        (out / f.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source / f.spec["src"], out / f.path)
        except OSError as error:
            raise ValueError(f"cannot link {source / f.spec['src']} into {out}: {error}; the dump goes in --temp, which must be on the download's volume") from None
    (out / TEXT).mkdir(parents=True, exist_ok=True)
    for day in made.days:
        (out / transcript_path(day)).write_bytes(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in made.logs[day.isoformat()]).encode("utf-8"))
    entries = entries_of(made, measure_disk(out, [f.path for f in made.files] + [transcript_path(d) for d in made.days], workers))
    return Dump(dump_key(name, recipe, generator), out, name, entries, (TRANSCRIPT,)), questions_of(made, entries)


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """V8 has no shortcut probe (a public dataset): the outcome is `none`, as the recorded run's run.json has it."""
    return ProbeOutcome("none", ())


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder an agent with its own tools reads the recordings in for `question`: slot `slot`'s, cut to the question's horizon
    (made new when `build`; else only named, for a dry run)."""
    folder = view_dir(temp, key, slot)
    return files_view(folder, dump, question.horizon) if build else folder


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    """What no command of the agent in a slot may hold, but for the run's folders: the old generator's words and the code that reaches
    this scenario; the download (`egolife` only in a command: the transcript says it once); and the other slots' folders of recordings
    (by path, by name, by glob, and the `glasses` and `transcript` folders in them, which a command's output may name too)."""
    base = f"mmb-{key}-d"
    return (*(IN_ONLY + word for word in (*OLD_WORDS, *NEW_WORDS)), *DOWNLOAD_WORDS,
            *(f"{base}{k}/{sub}" for k in others for sub in (VIDEO, TEXT)),
            *(IN_ONLY + w for k in others for w in (str(view_dir(temp, key, k)), f"{base}{k}", f"*-d{k}", f"?-d{k}")))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    """For an exec with tools of its own, the recordings past the horizon its outputs show (`seen_past_horizon`), which end it as
    `escaped` like the other tripwires. Nothing is added for an exec with no such tools: the system's own audit (a recording past the
    horizon in its results is a leak of the harness) is its session's."""
    if not access.own_tools:
        return {}
    seen = seen_past_horizon(command_outputs(out.get("events") or []), dump, question.horizon)
    late = {"problem": f"escaped: its output showed recordings past the horizon: {', '.join(seen[:5])}"
                       + (f" and {len(seen) - 5} more" if len(seen) > 5 else ""), "passed": False}
    return {"seen_past_horizon": seen, **(late if seen and out.get("problem") is None else {})}


def release(temp: Path, key: str) -> None:
    """The slots' folders of recordings removed: they hold links only."""
    release_views(temp, key)


def report(runs: Sequence[RunFolder], control: str = CONTROL) -> tuple[dict, str]:
    """The figures of the one run and the Markdown that renders them (report.json and report.md), against `control`. V8's report is of
    one run: its runs are not pooled (the core's pooled report, of V7's runs, cannot read V8's results)."""
    if len(runs) != 1:
        raise ValueError("a v8_egolife report is of one run: report " + ", ".join(r.name for r in runs) + " one at a time")
    figures = numbers(runs[0], control)
    return figures, markdown(figures)
