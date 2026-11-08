"""One-off conversion of the recorded pilot (D:\\mm-bench\\20261104-0314-pilot-s1) and of the recorded EgoLife run
(D:\\mm-bench\\20261103-0257-egolife-s1) to the format the new code reads (plan 7.2): no production code knows the old result format, so
the old results are rewritten once and the rewritten files are checked in, in tests/oracles/recorded/pilot-s1/converted/ and
tests/oracles/recorded/egolife-s1/converted/. The recorded runs are read only. Run it from the repo root with any Python 3.12, in the time
zone they were made in (US Eastern: the times of the plans are local):

    uv run python tests\\oracles\\convert_recorded.py [pilot|egolife|egolife-prompts] [--run RUN] [--out DIR]

It writes, in DIR (refused if it exists: a golden is not regenerated to make a test pass):
    results/<agent>/<config dir>/<qid>.json   the 110 results; `arm` and `model` become `config`, `system` and `variant`, nothing else
                                              changes (a result with no `work` is refused: the report no longer makes it up)
    questions.jsonl                           the 31 questions, from the V7 writer over the recorded manifest (sizes, mtimes, sha256)
    systems.json                              models.json as the runner writes it now (`note_configs`): for the tested configuration, its
                                              system, variant and folder, what its system reads (`readable`) and what it was `asked`
    run.json                                  what `RunFolder.open` needs (the recipe, the hashes, the configurations) and the run's
                                              probe and creation time; `mm_commit` is `COMMIT_KEY` (the benchmark then lived in
                                              Multimem's repository, so it was the same commit); `timeout_s` is the 900 s the old code
                                              had for every run, which the results' `timeout: no answer in 900 s` show; `agents`
                                              has each agent's model and effort, from its results, for `mmsb repeat`
The expected report (report.json, report.md) is the recorded pair one folder up.

EgoLife's folder has the same files, from the V8 writer, and these differences. The recorded run's `clips.json` is its clip table, which
the recipe holds now (reading the 2,592 clips for it takes minutes); the download at D:\\bench-data\\egolife is needed, for the plan the
recipe makes. The retrieval tier's records, `results/decider/<model>/<qid>.json` and `warm.json`, become
`results/retrieval/<config dir>/...`: `model` becomes `config`, `system` and `variant` in a record and `config` in warm.json, nothing
else changes. `run.json` also has `decode_s` and `pools`, which the report quotes. Its questions' entries have the recorded sizes and
mtimes and no hash (a question needs the path, time and length of a file).

`egolife-prompts` copies the 150 prompts the recorded EgoLife run was asked (execs/codex/<blind|files|mm-d1-3B>/<qid>/prompt.txt) into one
file, recorded/egolife-s1/prompts.json, `{qid: {"blind", "files", "mm"}}`: the expected values of the V8 prompts. The folder of the files
configuration, which is the recorded run's own temp folder, is `<DATA>`.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import v7_recorded

from mmsb import scenarios
from mmsb.core import fs
from mmsb.core.config import Config, parse_config
from mmsb.core.question import write_questions
from mmsb.core.runfolder import COMMIT_KEY
from mmsb.core.scenario import dump_key
from mmsb.scenarios.v7_screen.manifest import entries_of
from mmsb.scenarios.v7_screen.questions import questions_of
from mmsb.scenarios.v8_egolife import manifest as v8_manifest
from mmsb.scenarios.v8_egolife import questions as v8_questions

RUNS = Path(r"D:\mm-bench")
PILOT = RUNS / "20261104-0314-pilot-s1"
EGOLIFE = RUNS / "20261103-0257-egolife-s1"
RECORDED = Path(__file__).with_name("recorded") / "pilot-s1"
CONVERTED = RECORDED / "converted"
CONVERTED_EGOLIFE = Path(__file__).with_name("recorded") / "egolife-s1" / "converted"
PROMPTS_EGOLIFE = Path(__file__).with_name("recorded") / "egolife-s1" / "prompts.json"
EGOLIFE_ARMS = {"blind": "blind", "files": "files", "mm": "mm-d1-3B"}  # the recorded prompts' arms and their folders
DEFAULTS = {"files": None, "mm": "d1-3B"}  # the recorded runs' systems and their default variant
TIMEOUT_S = 900.0
OLD_RETRIEVAL = "decider"  # the old name of the results folder of the retrieval tier
EGOLIFE_OFFSET = "-0400"  # the UTC offset of the recorded EgoLife days (29 to 31 October 2026, US Eastern, still daylight time)


def config_of(arm: str, model: str | None) -> Config:
    """The configuration an old result's `arm` and `model` stand for: `blind`, `files`, `mm` with its model."""
    return parse_config(arm if model is None else f"{arm}/{model}", DEFAULTS)


def convert_result(record: dict) -> dict:
    """`record` with `arm` and `model` replaced, in their place, by `config`, `system` and `variant`."""
    if "work" not in record:
        raise ValueError(f"{record['qid']} has no `work`: it was made before results held it")
    config = config_of(record["arm"], record["model"])
    out: dict = {}
    for key, value in record.items():
        if key == "arm":
            out.update(config=config.name, system=config.system, variant=config.variant)
        elif key != "model":
            out[key] = value
    return out


def convert_results(source: Path, out: Path) -> list[tuple[str, dict]]:
    """Every result under `source`/results converted into `out`/results (the same agent and configuration folders, which are
    `Config.dir`), as `(agent, converted record)` in order."""
    converted = []
    for file in sorted((source / "results").rglob("*.json")):
        agent, folder = file.relative_to(source / "results").parts[:2]
        if agent == OLD_RETRIEVAL:
            continue  # the retrieval tier's records: convert_retrieval
        raw = fs.read_json(file)
        record, config = convert_result(raw), config_of(raw["arm"], raw["model"])
        if (record["agent"], config.dir) != (agent, folder):
            raise ValueError(f"{file} is a result of {record['agent']} {config.dir}")
        fs.write_json(out / "results" / agent / folder / file.name, record)
        converted.append((agent, record))
    return converted


def convert_retrieval(source: Path, out: Path) -> int:
    """The records of the retrieval tier under `source`/results/decider converted into `out`/results/retrieval, in the folder of their
    configuration (`Config.dir`): in a record `model` becomes `config`, `system` and `variant`, in warm.json `config`; nothing else
    changes. The tier is mm's. Returns the number of records."""
    count = 0
    for file in sorted((source / "results" / OLD_RETRIEVAL).rglob("*.json")):
        raw = fs.read_json(file)
        config, record = config_of("mm", raw["model"]), {}
        for key, value in raw.items():
            if key == "model":
                record.update(config=config.name)
                if file.stem != "warm":
                    record.update(system=config.system, variant=config.variant)
            else:
                record[key] = value
        if fs.slug(raw["model"]) != file.parent.name:
            raise ValueError(f"{file} holds the records of {raw['model']}")
        fs.write_json(out / "results" / "retrieval" / config.dir / file.name, record)
        count += file.stem != "warm"
    return count


def agents_of(results: list[tuple[str, dict]]) -> dict[str, dict]:
    """Each agent's model and effort (`agent_model`, `agent_effort` of its results; three Claude results from before the key have
    none), which must be one pair."""
    found: dict[str, set] = {}
    for agent, r in results:
        found.setdefault(agent, set()).add((r["agent_model"], r.get("agent_effort")))
    if bad := {a: pairs for a, pairs in found.items() if len(pairs) != 1}:
        raise ValueError(f"results of two settings of one agent: {bad}")
    return {agent: {"model": model, "effort": effort} for agent, ((model, effort),) in sorted(found.items())}


def convert_pilot(source: Path, out: Path) -> None:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: a golden is not regenerated to make a test pass")
    if reason := v7_recorded.zone_skip():
        raise SystemExit(reason)
    meta, models = fs.read_json(source / "run.json"), fs.read_json(source / "models.json")
    scenario = scenarios.get("v7_screen")
    recipe = scenario.recipe(scale=meta["scale"], seed=meta["seed"], days=meta["days"])
    if scenario.hashes(recipe) != (meta["plan"], meta["files"]) or meta["generator"] != scenario.generator:
        raise ValueError("the recorded pilot's recipe does not make the plan and files its run.json recorded")
    results = convert_results(source, out)

    plan = scenario.plan_of(recipe)
    sizes, hashes = v7_recorded.MANIFEST, v7_recorded.SHA256
    questions = questions_of(plan, entries_of(plan, lambda path: (sizes[path][0], sizes[path][1], hashes[path])))
    write_questions(out / "questions.jsonl", questions)

    configs = [config_of("files", None), config_of("mm", "d1-3B")]
    (info,) = models.values()
    tested = config_of("mm", "d1-3B")
    fs.write_json(out / "systems.json", {tested.name: {"system": tested.system, "variant": tested.variant, "dir": tested.dir,
                                                       "readable": info["readable"], "asked": info["asked"]}})
    fs.write_json(out / "run.json", {
        "scenario": scenario.name, "generator": scenario.generator, "recipe": recipe, "dump_key": dump_key(scenario.name, recipe, scenario.generator),
        "seed": meta["seed"], "scale": meta["scale"], "days": meta["days"], "data": meta["data"], "temp": meta["temp"], "tag": meta["tag"],
        "timeout_s": TIMEOUT_S, "plan": meta["plan"], "files": meta["files"], "question_hash": fs.sha256_of(out / "questions.jsonl"),
        "configs": [dataclasses.asdict(c) for c in configs], "agents": agents_of(results), "probe": meta["probe"],
        "created": meta["created"], COMMIT_KEY: meta["mm_commit"]})
    print(f"{len(results)} results, {len(questions)} questions -> {out}")


def convert_egolife(source: Path, out: Path) -> None:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: a golden is not regenerated to make a test pass")
    meta, models = fs.read_json(source / "run.json"), fs.read_json(source / "models.json")
    scenario = scenarios.get("v8_egolife")
    recipe = {"source": meta["source"], "scale": meta["scale"], "seed": meta["seed"], "days": meta["days"], "per_age": meta["per_age"],
              "speech": meta["speech"], "utc_offset": EGOLIFE_OFFSET,
              "clips": [[c["rel"], c["length"]] for c in fs.read_json(source / "clips.json")]}
    if scenario.hashes(recipe) != (meta["plan"], meta["files"]) or meta["generator"] != scenario.generator:
        raise ValueError("the recorded run's recipe does not make the plan and files its run.json recorded")
    if scenario.pool_counts(recipe) != meta["pools"]:
        raise ValueError("the recorded run's pools are not those its recipe makes")
    results = convert_results(source, out)
    asked = convert_retrieval(source, out)

    plan = scenario.plan_of(recipe)
    stats = fs.read_json(source / "data-manifest.json")
    questions = v8_questions.questions_of(plan, v8_manifest.entries_of(plan, lambda path: (*stats[path], "0" * 64)))
    write_questions(out / "questions.jsonl", questions)

    configs = [config_of("blind", None), config_of("files", None), config_of("mm", "d1-3B")]
    (info,) = models.values()
    tested = configs[-1]
    fs.write_json(out / "systems.json", {tested.name: {"system": tested.system, "variant": tested.variant, "dir": tested.dir,
                                                       "readable": info["readable"], "asked": info["asked"]}})
    fs.write_json(out / "run.json", {
        "scenario": scenario.name, "generator": scenario.generator, "recipe": recipe, "dump_key": dump_key(scenario.name, recipe, scenario.generator),
        "seed": meta["seed"], "scale": meta["scale"], "days": meta["days"], "data": meta["data"], "temp": meta["temp"], "tag": meta["tag"],
        "timeout_s": meta["timeout_s"], "plan": meta["plan"], "files": meta["files"], "question_hash": fs.sha256_of(out / "questions.jsonl"),
        "configs": [dataclasses.asdict(c) for c in configs], "agents": agents_of(results), "probe": meta["probe"],
        "created": meta["created"], COMMIT_KEY: meta["mm_commit"], "decode_s": meta["decode_s"], "pools": meta["pools"]})
    print(f"{len(results)} results, {asked} retrieval records, {len(questions)} questions -> {out}")


def capture_prompts(source: Path, out: Path) -> None:
    """The prompts of the recorded EgoLife execs of Codex (Claude's one exec is a smoke), written whole to `out`."""
    if out.exists():
        raise FileExistsError(f"{out} exists: a golden is not regenerated to make a test pass")
    prompts: dict[str, dict[str, str]] = {}
    for arm, folder in EGOLIFE_ARMS.items():
        for file in sorted((source / "execs" / "codex" / folder).glob("*/prompt.txt")):
            text = file.read_text(encoding="utf-8")
            if arm == "files":
                text, found = re.subn(r"are in D:\\mmb-temp\\mmb-[0-9a-f]{10}-d\d:", "are in <DATA>:", text)
                if found != 1:
                    raise ValueError(f"{file} does not say where the recordings are, once")
            prompts.setdefault(file.parent.name, {})[arm] = text
    if bad := {qid: sorted(arms) for qid, arms in prompts.items() if sorted(arms) != sorted(EGOLIFE_ARMS)}:
        raise ValueError(f"questions not asked in all three arms: {bad}")
    fs.write_json(out, prompts)
    print(f"{sum(len(arms) for arms in prompts.values())} prompts of {len(prompts)} questions -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("which", nargs="?", choices=("pilot", "egolife", "egolife-prompts"), default="pilot")
    parser.add_argument("--run", type=Path, help="the recorded run (default: the one in D:\\mm-bench)")
    parser.add_argument("--out", type=Path, help="where the conversion goes (default: converted/ beside the recorded report)")
    args = parser.parse_args()
    if args.which == "pilot":
        convert_pilot(args.run or PILOT, args.out or CONVERTED)
    elif args.which == "egolife":
        convert_egolife(args.run or EGOLIFE, args.out or CONVERTED_EGOLIFE)
    else:
        capture_prompts(args.run or EGOLIFE, args.out or PROMPTS_EGOLIFE)


if __name__ == "__main__":
    main()
