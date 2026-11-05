"""`mmmb repeat RECORDED`: a run done again as it was done (plan 8.1): the same recipe, and the same agents at the same models and efforts, over
the same configurations and the same questions of each cell. This module reads what the recorded run says of itself and says what the
repeat will do; the building and the running are `mmmb build` and `mmmb run`, which the command calls, and the comparison with the
recorded run is `mmmb compare`.

What a run knows of its agents is in its results (`agent_model`, `agent_effort` of each) and in run.json's `agents`, which `mmmb run` writes
from them when it ends (`note_agents`) and the converter wrote for the recorded runs: `{agent: {"model", "effort"}}`. A recorded run whose
results disagree with its `agents` is refused. An agent that records no effort (Claude's results hold none) has effort None, which is its
own default; the models are the ones the agents reported, not the aliases they were asked for."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mmmb.core import costing, fs, report
from mmmb.core.config import Config
from mmmb.core.contract import Description
from mmmb.core.runfolder import RunFolder
from mmmb.core.scenario import Scenario
from mmmb.core.select import config_questions, selected


def agents_of(results: Mapping[str, Mapping[str, Mapping[str, dict]]]) -> dict[str, dict]:
    """Each agent's model and effort as its results (`report.results_of`) say, which must be one pair for an agent: results of two
    settings are not one experiment."""
    out: dict[str, dict] = {}
    for agent, cells in sorted(results.items()):
        pairs = {(r.get("agent_model"), r.get("agent_effort")) for rs in cells.values() for r in rs.values()}
        if len(pairs) > 1:
            raise ValueError(f"the results of {agent} were made at {len(pairs)} settings, " + " and ".join(
                f"{model or 'an unknown model'}{f' at {effort}' if effort else ''}" for model, effort in sorted(pairs, key=str))
                             + ": remove the results of one, or build another run")
        if pairs:
            ((model, effort),) = pairs
            out[agent] = {"model": model, "effort": effort}
    return out


def note_agents(run: RunFolder) -> None:
    """run.json's `agents` made what the results of `run` say, so that a later `mmmb repeat` can do it again."""
    found = agents_of(report.results_of(run))
    if fs.read_json(run.folder / "run.json").get("agents") != found:
        run.rewritten(agents=found)


@dataclass(frozen=True)
class Repeat:
    """What a recorded run did: its agents (name -> model and effort), its configurations (those that have results), the qids each cell
    was asked, and all the qids in the questions' order."""

    recorded: RunFolder
    agents: Mapping[str, Mapping[str, str | None]]
    configs: tuple[Config, ...]
    cells: Mapping[tuple[str, str], tuple[str, ...]]  # (agent, Config.name) -> qids
    qids: tuple[str, ...]


def repeat_of(recorded: RunFolder) -> Repeat:
    """What to do to repeat `recorded`. Refused when it has no results, no `agents` (it was not run to the end by `mmmb run`) or
    results that say other agents or settings than its `agents` do."""
    results = report.results_of(recorded)
    held = fs.read_json(recorded.folder / "run.json").get("agents")
    if not any(rs for cells in results.values() for rs in cells.values()):
        raise ValueError(f"{recorded.name} has no results: there is nothing to repeat")
    if held is None:
        raise ValueError(f"{recorded.name} records no agents in its run.json: `mmmb run` writes them when it ends, so it has not been run to the end")
    if held != (found := agents_of(results)):
        raise ValueError(f"{recorded.name}'s run.json says its agents are {held}, and its results say {found}: it was run again after that; "
                         "`mmmb run` it to the end to note them")
    by_dir = {c.dir: c for c in recorded.configs}
    cells = {}
    for agent, folders in results.items():
        for folder, rs in folders.items():
            if rs and folder not in by_dir:
                raise ValueError(f"results/{agent}/{folder} of {recorded.name} is not one of its configurations ({', '.join(by_dir)})")
            if rs:
                cells[agent, by_dir[folder].name] = tuple(sorted(rs))
    asked = {qid for qids in cells.values() for qid in qids}
    return Repeat(recorded, held, tuple(c for c in recorded.configs if any(name == c.name for _, name in cells)), cells,
                  tuple(q.qid for q in recorded.questions() if q.qid in asked))


def check_same_cases(repeat: Repeat, descriptions: Mapping[Config, Description | None]) -> None:
    """Raises when a run of the recorded configurations over the recorded qids would ask a cell other questions than the recorded run
    asked it (a recorded run that gave only some configurations the text and records, say): the repeat would not be of the same cases."""
    questions = selected(repeat.recorded.questions(), "all", descriptions, repeat.qids)
    for config in repeat.configs:
        wanted = {q.qid for q in config_questions(questions, config, descriptions[config])}
        for agent in repeat.agents:
            if (had := set(repeat.cells.get((agent, config.name), ()))) != wanted:
                raise ValueError(f"{repeat.recorded.name} asked {agent} {config.name} {len(had)} questions, and a run now asks it {len(wanted)} "
                                 f"(a difference of {len(had ^ wanted)}, the first {sorted(had ^ wanted)[0]}): use mmmb run with --only to do that")


def options_of(scenario: Scenario, recipe: Mapping[str, object]) -> dict[str, object]:
    """The options that make `recipe` again: those of the recipe that `scenario.recipe` takes. Refused when they do not (the time zone is
    another, a download changed)."""
    given = {name: recipe[name] for name in inspect.signature(scenario.recipe).parameters if name in recipe}
    if dict(scenario.recipe(**given)) != dict(recipe):
        raise ValueError(f"{scenario.name} makes a recipe other than the recorded one from {given}: the recorded run was made where the time "
                         "zone, a library or a download was other than here, so this would not be the same week")
    return given


def run_argv(folder: Path, repeat: Repeat, jobs: int) -> list[str]:
    """The `mmmb run` command line that does the repeat in `folder`: the recorded agents at their models and efforts, configurations,
    questions and time limit."""
    argv = ["run", str(folder), "--agents", ",".join(repeat.agents), "--configs", ",".join(c.name for c in repeat.configs),
            "--only", ",".join(repeat.qids), "--jobs", str(jobs), "--timeout", f"{repeat.recorded.timeout_s:g}"]
    for name, setting in repeat.agents.items():
        for what, value in setting.items():
            if value is not None:
                argv += [f"--{name}-{what}", str(value)]
    return argv


def setting_text(setting: Mapping[str, str | None]) -> str:
    return f"{setting['model']}, " + (f"at {setting['effort']} effort" if setting["effort"] else "at the agent's own effort (its results hold none)")


def show_repeat(repeat: Repeat, descriptions: Mapping[Config, Description | None], jobs: int, folders: Sequence[Path], temp: Path,
                cache: Path) -> None:
    """What the repeat will do, for `--dry-run` and before it starts: the agents, models and efforts, the cells and their execs, what each
    took in the earlier runs `folders` (the recorded run among them), and what has to be made first: the dump and the stores that the cache
    in `cache` holds for the recipe (the dump in `temp`) are said to be cached, the rest to be made. A configuration whose system needs the
    GPU is counted one exec at a time (DESIGN 8.1), as its store, which tells the system's own limit, does not exist yet."""
    run = repeat.recorded
    history = costing.history_of(folders)
    print(f"repeat of {run.name}: {run.scenario.name}, {run.scale}, seed {run.seed}, {len(run.days)} days; {len(repeat.qids)} questions, "
          f"{sum(len(qids) for qids in repeat.cells.values())} execs")
    for name, setting in repeat.agents.items():
        print(f"  {name}: {setting_text(setting)}")
    print(f"  configurations: {', '.join(c.name for c in repeat.configs)}")
    cells = []
    for name in repeat.agents:
        for config in repeat.configs:
            if qids := repeat.cells.get((name, config.name)):
                description = descriptions[config]
                at_once = 1 if description is not None and description.needs_gpu else jobs
                cells.append(costing.Cell(name, config, len(qids), len(qids), at_once, history.get((name, config.dir))))
    costing.show_cells(cells, jobs, True)
    costing.show_new_builds(run, folders, {c.system: descriptions[c] for c in repeat.configs if c.system is not None}, temp, cache)
