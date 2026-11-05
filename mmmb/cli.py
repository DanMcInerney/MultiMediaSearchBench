"""The `mmmb` command (moved from `parser_of` and `main` of bench.py and egolife.py of this repository at bb498a6, now one command over
any scenario and any system). Argument parsing and the order of the steps; what each step does is below it (core/build.py, core/dryrun.py,
core/runner.py, core/retrieval.py, the scenario's `report`).

    mmmb build    --scenario NAME [--scale S --seed N --days D,D --source DIR ...] [--out DIR --temp DIR --cache DIR --configs A,B --no-probe]
    mmmb ingest   RUN [--configs A,B]                       the stores of the systems (build makes those of its --configs)
    mmmb run      RUN [--agents claude,codex --configs A,B --jobs N --side-by-side --subset all|pilot --controls CONFIG --only QID,.. --timeout S
                       --clock working|wall --anyway --dry-run --claude-effort L --codex-effort L --claude-model M --codex-model M --answer-format json|line
                       --scenarios S,.. --kinds K,.. --modalities M,.. --splits S,.. --tag NAME=VALUE --rung R --limit N --discriminating RUN,..]
    mmmb retrieve RUN [--configs A,B --dry-run]             each system asked for the needles, with no agent (the GPU, alone)
    mmmb report   RUN [RUN ...] [--control CONFIG]
    mmmb compare  RUN_A RUN_B [--across-settings]            two runs cell by cell: flips, counts, D1's band, McNemar, verdict lines
    mmmb repeat   RECORDED [--into RUN --out DIR --temp DIR --cache DIR --jobs N --answer-format json|line --clock working|wall --dry-run]
                                                            a recorded run again at its agents' models and efforts
    mmmb clock-check RUN --config CONFIG [--agents claude,codex --cases N --jobs N]
                                                            the working clock tried: N cases asked alone and then queued, side by side

(The options are in arguments.py, which holds `parser_of`.)

`RUN` is a run folder, or its name under the runs folder. `build` and `repeat` go through the cache (core/cache.py): a dump and the stores
of the systems that an earlier run of the same recipe made are found and not made again, and the command says `cached`. A later command
adds a subparser in `parser_of`, a case in `main` and a `*_command` function; `main` is the one entry. A configuration is `blind`, or
`<system>[/<variant>][+files]`: the systems are those with a system.toml in mmmb/systems, and a system whose extra is not installed is
refused with the extra's name."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mmmb import scenarios, systems
from mmmb.arguments import DEFAULT_SEED, OPTIONS, csv, parser_of
from mmmb.core import fs, gpu, paths, render, report
from mmmb.core.agents import claude
from mmmb.core.build import build_run, check_stores, ensure_data, ingest_stores, read_dump, store_hashes, store_of
from mmmb.core.clockcheck import check_possible, light_cases, pick, run_halves, show_sides
from mmmb.core.compare import compare_runs, render_comparison
from mmmb.core.config import Config, describe_configs, parse_config
from mmmb.core.contract import Description, Isolation, Session, SessionContext, System
from mmmb.core.costing import earlier_runs, expected_of, history_of, show_agents, show_builds
from mmmb.core.dryrun import RETRIEVAL_AGENT, check_clocks, dry_run, previewer, show_limits, show_retrieval, warn_of_probe
from mmmb.core.probe import probe_of
from mmmb.core.repeat import check_same_cases, note_agents, options_of, repeat_of, run_argv, show_repeat
from mmmb.core.retrieval import FOLDER, run_retrieval
from mmmb.core.runfolder import JOBS_MAX, RunFolder
from mmmb.core.runner import AGENTS, Agent, answer_format_of, real_agents, run_agents
from mmmb.core.schedule import Pools, SideBySide
from mmmb.core.select import Selection, config_questions, select, selected


def resolve_run(text: str) -> RunFolder:
    """A run from its folder, or its name under the runs folder."""
    path = Path(text)
    return RunFolder.open(path if path.is_dir() else paths.RUNS / text, scenarios.get)


@dataclass(frozen=True)
class Registry:
    """The systems a command may name: the folders with a system.toml in `folder` (a test gives another). Their configurations, isolation
    and systems all come from it."""

    folder: Path = systems.HERE

    def load(self, name: str) -> System:
        return systems.load(name, self.folder)

    def isolation(self) -> Isolation:
        return systems.isolation(self.folder)

    def parse(self, text: str) -> list[Config]:
        """The configurations written in `text`, comma separated, in the order given."""
        known = systems.defaults(self.folder)
        configs = [parse_config(part, known) for part in csv(text)]
        if len({c.dir for c in configs}) != len(configs):
            raise ValueError(f"these configurations share a results folder: {text}")
        return configs

    def configs(self, text: str | None, run: RunFolder) -> list[Config]:
        """The configurations written in `text`, or the run's own when there is none."""
        return list(run.configs) if text is None else self.parse(text)

    def setup(self, configs: Sequence[Config]) -> tuple[dict[str, System], dict[Config, Description | None]]:
        """The systems the configurations name, loaded (one whose extra is missing fails here), and the description of each configuration."""
        return describe_configs(configs, self.load)

    @staticmethod
    def check_ready(configs: Sequence[Config], loaded: Mapping[str, System]) -> None:
        """Raises, naming the configuration, when a system cannot run a configuration's variant now (`System.check_ready`): asked of every
        configuration before any agent starts, so that a run does not spend its first cells and then fail at the next system's."""
        for config in configs:
            if config.system is not None:
                try:
                    loaded[config.system].check_ready(config.variant)
                except ValueError as error:
                    raise ValueError(f"{config.name} cannot run now: {error}") from None


def with_effort(agent: Agent, level: str | None) -> Agent:
    """`agent` asked to work at effort `level` (None: its own default)."""
    if level is None:
        return agent
    return dataclasses.replace(agent, run=lambda spec: agent.run(dataclasses.replace(spec, effort=level)))


def with_model(agent: Agent, name: str | None) -> Agent:
    """`agent` asked to use the model `name` (None: its own default)."""
    if name is None:
        return agent
    return dataclasses.replace(agent, run=lambda spec: agent.run(dataclasses.replace(spec, model=name)))


def agents_named(text: str, parser: argparse.ArgumentParser) -> list[str]:
    """The agents `--agents` names, refused when none or one that is not an agent."""
    names = csv(text)
    if not names or (bad := [n for n in names if n not in AGENTS]):
        parser.error(f"--agents is some of {', '.join(AGENTS)}, not {', '.join(bad) if names else 'none'}")
    return names


def check_jobs(jobs: int, parser: argparse.ArgumentParser) -> None:
    if not 1 <= jobs <= JOBS_MAX:
        parser.error(f"--jobs is 1 to {JOBS_MAX}: more crashed the laptop once")


def selection_of(args: argparse.Namespace, parser: argparse.ArgumentParser, names: Sequence[str]) -> Selection:
    """What the selection flags of `run` ask for (`--rung X` is `--tag rung=X`); the agents `names` are those whose results count in
    the `--discriminating` runs."""
    tags: dict[str, str] = {}
    for text in args.tag or ():
        key, equals, value = text.partition("=")
        if not (key and equals and value):
            parser.error(f"--tag is NAME=VALUE, not {text!r}")
        if tags.setdefault(key, value) != value:
            parser.error(f"--tag {key} is given two values")
    if args.rung is not None and tags.setdefault("rung", args.rung) != args.rung:
        parser.error(f"--rung {args.rung} and --tag rung={tags['rung']} name two rungs")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit is 1 or more, not {args.limit}")
    return Selection(tuple(csv(args.scenarios)), tuple(csv(args.kinds)), tuple(csv(args.modalities)), tuple(csv(args.splits)), tags, args.limit,
                     discriminating=tuple(resolve_run(r) for r in csv(args.discriminating)), agents=tuple(names))


def picked(run: RunFolder, descriptions: Mapping[Config, Description | None], subset: str, only: Sequence[str],
           selection: Selection) -> list[str]:
    """`only` (the qids asked for), or with a selection the qids it picks among the subset's questions and `only`, said step by step.
    A selection that picks nothing is refused: an empty `only` would mean every question."""
    if selection.empty:
        return list(only)
    trail: list[tuple[str, int]] = []
    chosen = select(selected(run.questions(), subset, descriptions, only), selection, trail)
    print("selection: " + ", then ".join(f"{step}: {n}" for step, n in trail))
    if not chosen:
        raise ValueError("the selection picks no question")
    return [q.qid for q in chosen]


def earlier(run: RunFolder) -> list[Path]:
    """The run's own folder and the other runs of its scenario and scale in the runs folder: what the estimate takes its medians from."""
    return [run.folder, *(f for f in earlier_runs(paths.RUNS, run.scenario.name, run.scale) if f.resolve() != run.folder.resolve())]


# -- the commands -----------------------------------------------------------------------------------------------------------------

def build_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry) -> int:
    scenario = scenarios.get(args.scenario)
    takes = inspect.signature(scenario.recipe).parameters
    given = {name: getattr(args, name) for name in OPTIONS if getattr(args, name) is not None}
    if "seed" in takes:
        given.setdefault("seed", DEFAULT_SEED)
    if "workers" in takes:
        given["workers"] = args.workers
    if wrong := [name for name in given if name not in takes]:
        parser.error(f"{scenario.name} takes no " + ", ".join(f"--{name.replace('_', '-')}" for name in wrong))
    if missing := [name for name, p in takes.items() if p.default is inspect.Parameter.empty and name not in given]:
        parser.error(f"{scenario.name} needs " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    run = build_run(scenario, given, args.out, args.temp, registry.parse(args.configs), timeout_s=args.timeout, cache=args.cache,
                    probe=not args.no_probe, workers=args.workers, load=registry.load)
    print(run.folder)
    outcome = probe_of(run.folder)
    if outcome["status"] == "failed":
        for failure in outcome["failures"]:
            print(f"probe: {failure}", file=sys.stderr)
        print(f"error: the shortcut probe failed ({len(outcome['failures'])}): the run is kept in {run.folder} (see shortcuts.md); change "
              "the scenario and build another", file=sys.stderr)
        return 1
    if outcome["status"] == "skipped":
        print("the shortcut probe was skipped (--no-probe): the report will say so")
    return 0


def ingest_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry) -> int:
    run = resolve_run(args.run)
    configs = registry.configs(args.configs, run)
    loaded, descriptions = registry.setup(configs)
    measured = ingest_stores(run.folder, ensure_data(run, read_dump(run)), descriptions, loaded.__getitem__,  # (the dump is on disk to be read)
                             cache=run.cache)
    held = fs.read_json(run.folder / "run.json")
    if clash := [k for k, v in measured.items() if held.get(k, v) != v]:
        raise ValueError(f"{run.name} holds {', '.join(clash)} as other figures than the stores measured: build a new run")
    run.with_configs(configs).rewritten(**measured)
    return 0


def announce(run: RunFolder, names: Sequence[str], configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
             preview: Callable[[str, Config], Session], jobs: int, controls: str | None, subset: str, only: Sequence[str],
             side_by_side: bool) -> None:
    """What a run will do before it does it (and all a dry run prints of cost): what has to be built, the execs of each cell with the
    medians of earlier runs, the limits of the systems."""
    print("the data and the stores it needs:")
    show_builds(run, sorted({c.system for c in configs if c.system is not None}))
    show_agents(run, names, configs, descriptions, preview, jobs, controls, subset, only, history_of(earlier(run)), side_by_side)
    show_limits(run, names, configs, descriptions, preview, jobs, controls, subset, only)


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry,
                real: Callable[[], Mapping[str, Agent]]) -> int:
    names = agents_named(args.agents, parser)
    check_jobs(args.jobs, parser)
    only = csv(args.only)
    run = resolve_run(args.run)
    configs = registry.configs(args.configs, run)
    if args.controls is not None and args.controls not in {c.name for c in configs}:
        parser.error(f"--controls names {args.controls}, which is not one of the configurations: {', '.join(c.name for c in configs)}")
    efforts = {"claude": args.claude_effort, "codex": args.codex_effort}
    models = {"claude": args.claude_model, "codex": args.codex_model}
    selection = selection_of(args, parser, names)
    answer_format = answer_format_of(run, args.answer_format)  # refused when the run has results in the other format
    if args.timeout is not None:
        run = dataclasses.replace(run, timeout_s=args.timeout)  # kept in run.json when nothing has refused the command (below)
    loaded, descriptions = registry.setup(configs)
    check_stores(run.folder, store_hashes(descriptions))  # missing, made by another version, or misdated: refused
    dump = read_dump(run)
    only = picked(run, descriptions, args.subset, only, selection)
    preview = previewer(run, loaded, lambda system: store_of(run.folder, system), dump)
    check_clocks(run, names, configs, descriptions, preview, args.jobs, args.controls, args.subset, only, args.clock)  # refused before it spends anything
    if args.dry_run:
        announce(run, names, configs, descriptions, preview, args.jobs, args.controls, args.subset, only, args.side_by_side)
        dry_run(run, names, configs, descriptions, preview, dump, registry.isolation(), args.controls, args.subset, only, efforts, models,
                answer_format=answer_format, side_by_side=args.side_by_side)
        return 0
    if (outcome := probe_of(run.folder)) and outcome["status"] == "failed" and not args.anyway:
        print(f"error: {run.name} failed the shortcut probe ({len(outcome['failures'])} failures, in its shortcuts.md): a shortcut may "
              "find its needles, so its numbers would not say what a system is worth. Build another run, or pass --anyway", file=sys.stderr)
        return 1
    registry.check_ready(configs, loaded)  # before any agent starts: not in a dry run, which starts nothing
    warn_of_probe(run)
    announce(run, names, configs, descriptions, preview, args.jobs, args.controls, args.subset, only, args.side_by_side)
    dump = ensure_data(run, dump)
    if "claude" in names:
        claude.check_login()
    agents = {name: with_model(with_effort(agent, efforts[name]), models[name]) for name, agent in real().items() if name in names}
    run = run.with_configs(configs)
    if args.timeout is not None:
        run = run.with_timeout(args.timeout)
    run.rewritten(answer_format=answer_format)  # kept: the next command asks in the same format
    beside = SideBySide(Pools.of(names), expected_of(history_of(earlier(run)))) if args.side_by_side else None
    run_agents(run, agents, configs, dump, lambda system: store_of(run.folder, system), args.jobs, controls=args.controls,
               subset=args.subset, only=only, load=registry.load, isolation=registry.isolation(), answer_format=answer_format, side_by_side=beside,
               clock=args.clock)
    note_agents(run)  # what the agents were (the results say), so that `mmmb repeat` can do this run again
    return 0


def retrieve_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry) -> int:
    run = resolve_run(args.run)
    chosen = registry.configs(args.configs, run)
    loaded, found = registry.setup(chosen)
    configs = [c for c in chosen if found[c] is not None and found[c].has_oracle]
    if not configs:
        raise ValueError(f"none of {', '.join(c.name for c in chosen)} is a system that can be asked for needles with no agent")
    descriptions = {c: found[c] for c in configs}
    check_stores(run.folder, store_hashes(descriptions))
    dump = read_dump(run)
    questions = selected(run.questions(), "all", descriptions)
    show_retrieval(run, configs, descriptions, previewer(run, loaded, lambda system: store_of(run.folder, system), dump), questions)
    if args.dry_run:
        return 0
    for config in configs:
        mine = [q for q in config_questions(questions, config, descriptions[config]) if q.retrieval is not None]
        done = run.folder / "results" / FOLDER / config.dir
        if all((done / f"{q.qid}.json").exists() for q in mine):
            continue  # nothing is left to ask: opening a system to find that out would load its model
        gpu.check_gpu(descriptions[config])
        context = SessionContext(RETRIEVAL_AGENT, config.variant, run.timeout_s, run.folder / "homes", time.time)
        began = time.monotonic()
        session = loaded[config.system].open(store_of(run.folder, config.system), dump, context)
        try:
            run_retrieval(run, config, session, mine, warm_s=round(time.monotonic() - began, 1))
        finally:
            for what, items in session.close().items():
                if items:
                    fs.write_json(run.folder / "results" / FOLDER / config.dir / f"{what}.json", items)
                    print(f"{config.name}: {len(items)} {what} left by the retrieval", flush=True)
    return 0


def report_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    runs = [resolve_run(r) for r in args.runs]
    if never := [r.name for r in runs if not (r.folder / "systems.json").exists()]:
        raise ValueError(f"{', '.join(never)} has not been run: it has no systems.json and no results to report")
    if len({r.scenario.name for r in runs}) > 1:
        raise ValueError("these runs are of different scenarios: " + ", ".join(f"{r.name} ({r.scenario.name})" for r in runs))
    if len(runs) == 1:
        figures, text = runs[0].scenario.report(runs[0])
    else:  # pooled: the core's report of the control against the tested configurations over the runs
        figures = report.numbers(runs, args.control)
        text = render.markdown(figures)
    folder = runs[0].folder
    fs.write_text(folder / "report.md", text)
    fs.write_json(folder / "report.json", figures)
    print(text)
    print(f"written to {folder / 'report.md'} and report.json")
    return 0


def repeat_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry,
                   real: Callable[[], Mapping[str, Agent]]) -> int:
    """A recorded run again (core/repeat.py): its recipe built into a new run, then `mmmb run` of that run with the recorded agents at their
    models and efforts, over the recorded configurations and questions. With `--into` the run is one an earlier repeat built: it goes on."""
    check_jobs(args.jobs, parser)
    recorded = resolve_run(args.recorded)
    repeat = repeat_of(recorded)
    loaded, descriptions = registry.setup(repeat.configs)
    check_same_cases(repeat, descriptions)
    options = options_of(recorded.scenario, recorded.recipe)
    show_repeat(repeat, descriptions, args.jobs, earlier(recorded), args.temp, args.cache)
    if args.into is not None:
        run = resolve_run(args.into)
        if dict(run.recipe) != dict(recorded.recipe):
            raise ValueError(f"{run.name} was not made from the recipe of {recorded.name}, so its results would not be of the same week")
    elif args.dry_run:
        print("\nnothing was built or started")
        return 0
    else:
        registry.check_ready(repeat.configs, loaded)  # before a build that takes minutes
        run = build_run(recorded.scenario, options, args.out, args.temp, list(repeat.configs), timeout_s=recorded.timeout_s,
                        cache=args.cache, workers=args.workers, load=registry.load)
        print(run.folder)
    argv = [*run_argv(run.folder, repeat, args.jobs), *(["--answer-format", args.answer_format] if args.answer_format else []),
            *(["--clock", args.clock] if args.clock else []), *(["--dry-run"] if args.dry_run else [])]
    status = run_command(parser.parse_args(argv), parser, registry, real)
    if status == 0 and not args.dry_run:
        print(f"repeated {recorded.name} as {run.name}; compare them, and read every flip:\n  mmmb compare {recorded.folder} {run.folder}")
    return status


def clock_check_command(args: argparse.Namespace, parser: argparse.ArgumentParser, registry: Registry,
                        real: Callable[[], Mapping[str, Agent]]) -> int:
    """The working clock tried on a few cases of one system (core/clockcheck.py): refused, before any agent starts, when the system cannot
    report its queue or would not run the cases beside each other; else each of the cases asked alone and then queued, and the two printed
    side by side."""
    names = agents_named(args.agents, parser)
    check_jobs(args.jobs, parser)
    if args.cases < 1:
        parser.error(f"--cases is 1 or more, not {args.cases}")
    run = resolve_run(args.run)
    if len(configs := registry.parse(args.config)) != 1:
        parser.error(f"--config is one configuration, not {args.config!r}")
    config = configs[0]
    if config.system is None:
        raise ValueError(f"{config.name} has no system, so there is no queue to check")
    loaded, descriptions = registry.setup([config])
    check_stores(run.folder, store_hashes(descriptions))
    dump = read_dump(run)
    preview = previewer(run, loaded, lambda system: store_of(run.folder, system), dump)
    session = preview(names[0], config)
    cases = light_cases(run, names[0], config, descriptions[config], session)
    chosen = pick(run, config, cases, args.cases)
    qids = [q.qid for q in chosen]
    check_possible(config, session, [c for c in cases if c.qid in qids], args.jobs)
    registry.check_ready([config], loaded)  # before any agent starts
    answer_format = answer_format_of(run, None)
    dump = ensure_data(run, dump)
    if "claude" in names:
        claude.check_login()
    agents = {name: agent for name, agent in real().items() if name in names}

    def ask(half: RunFolder, at_once: int) -> None:
        run_agents(half, agents, [config], dump, lambda system: store_of(run.folder, system), at_once, only=qids, load=registry.load,
                   isolation=registry.isolation(), answer_format=answer_format)

    show_sides(run_halves(run, args.jobs, ask), names, config, qids)
    return 0


def compare_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    comparison = compare_runs(resolve_run(args.run_a), resolve_run(args.run_b), across_settings=args.across_settings)
    print(render_comparison(comparison))
    return 1 if comparison.refused else 0


def main(argv: list[str] | None = None, *, agents: Callable[[], Mapping[str, Agent]] = real_agents,
         registry: Path = systems.HERE) -> int:
    """Runs one command and returns its exit status. `agents` gives the agents a run uses (real ones spend the owner's plan; a test gives
    stand-ins) and `registry` is the folder of the systems (a test gives one with a stand-in system)."""
    parser, registry = parser_of(), Registry(registry)
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "build":
                return build_command(args, parser, registry)
            case "ingest":
                return ingest_command(args, parser, registry)
            case "run":
                return run_command(args, parser, registry, agents)
            case "retrieve":
                return retrieve_command(args, parser, registry)
            case "repeat":
                return repeat_command(args, parser, registry, agents)
            case "compare":
                return compare_command(args, parser)
            case "clock-check":
                return clock_check_command(args, parser, registry, agents)
            case _:
                return report_command(args, parser)
    except KeyboardInterrupt:
        print("interrupted: run the same command to continue", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, AssertionError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
