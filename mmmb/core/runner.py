"""The run loop (moved from `run_arm`, `run_agents` and `real_agents` of bench.py of this repository at bb498a6): the pending cases of
one configuration of one agent, in their order, each in a working slot; and the configurations of a run in order, each with the
session of its system. Only what a system does moved behind `Session`: the loop, its order, its stops and its record are the
core's, and it names no system."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mmmb import systems
from mmmb.core import fs, gpu, grounded
from mmmb.core.agents import claude, codex
from mmmb.core.agents.spec import SessionSpec
from mmmb.core.answer import DEFAULT_FORMAT, AnswerFormat, parse_final
from mmmb.core.clock import DEFAULT_CLOCK, WORKING, check_queue, figures_of, unreported
from mmmb.core.config import Config, describe_configs
from mmmb.core.contract import Case, Description, Dump, Isolation, PlainSession, Session, SessionContext, System
from mmmb.core.question import Question
from mmmb.core.record import Halt, record
from mmmb.core.retrieval import FOLDER as RETRIEVAL_FOLDER
from mmmb.core.runfolder import JOBS_MAX, RunFolder, is_done, load_result, now_text
from mmmb.core.scenario import Access
from mmmb.core.schedule import Beside, SideBySide, Task, agent_off_gpu, keeps_time_order, next_case, run_beside, start_order
from mmmb.core.select import asked_modalities, config_questions, selected


JOBS = 3  # execs at once by default
AGENTS = ("claude", "codex")
TIMEOUT_S = 900.0  # the seconds an exec may take, unless the run says other (`RunFolder.timeout_s`)
NO_SLOT = -1  # the slot of a case that has not started


@dataclass(frozen=True)
class Agent:
    run: Callable[[SessionSpec], dict]  # one exec: the common result of the agents (agents/claude.py, codex.py)
    must_report_use: bool = False  # its results must hold its use of the plan's week, else the quota cannot be watched (Codex's)


def reads_of(view: Path | None, dump: Dump) -> Path | None:
    """What `RunFolder.deny` and `check_paths` are told an agent with its own tools reads: the folder `Scenario.view` gave it when that is
    not the dump (then the dump is hidden from it), else None, the dump itself."""
    return None if view is None or view == dump.root else view


def answer_format_of(run: RunFolder, asked: AnswerFormat | None) -> AnswerFormat:
    """The format the final answers of a command are asked in: `asked` (the command's `--answer-format`), else the one run.json keeps,
    else DEFAULT_FORMAT; the command then keeps it in run.json (`RunFolder.rewritten`). A run's format does not change once it has
    results (they would be of two formats): a run with results and no format in its run.json was made before the format was kept, and
    its results are of the `line` format."""
    held = fs.read_json(run.folder / "run.json").get("answer_format")
    qids = {q.qid for q in run.questions()}
    started = any(f.stem in qids for f in (run.folder / "results").glob("*/*/*.json") if f.parent.parent.name != RETRIEVAL_FOLDER)
    held = held or ("line" if started else None)
    if asked is not None and started and asked != held:
        raise ValueError(f"{run.name} has results in the {held} answer format, so it cannot be asked in the {asked} format now: its results "
                         f"would be of two formats. Run it with --answer-format {held}, or build a new run")
    return asked or held or DEFAULT_FORMAT


def case_of(run: RunFolder, name: str, config: Config, q: Question, session: Session) -> Case:
    """Question `q` as the session sees it, not yet in a working slot (`NO_SLOT`, no working folder)."""
    return Case(q.qid, datetime.fromisoformat(q.now).timestamp(), q.horizon, q.scope.sources, q.scope.since, q.scope.until, q.group,
                session.estimate(q.scope.sources, q.cost_ranges), NO_SLOT, Path(), run.exec_dir(name, config, q.qid))


def run_config(run: RunFolder, name: str, agent: Agent, config: Config, description: Description | None,
               questions: Sequence[Question], session: Session, dump: Dump, isolation: Isolation, jobs: int, halt: Halt,
               answer_format: AnswerFormat, *, beside: Beside | None = None, clock: str = DEFAULT_CLOCK) -> None:
    """The pending questions of one configuration of agent `name`, in ascending `until`, `jobs` at a time (fewer when the session
    says so), each in a working slot. Each exec is given what `session.serve` serves for its case, the folder `scenario.view`
    makes ready when the configuration has tools of its own, the prompt of `scenario.prompt`, and its result holds what
    `scenario.after_exec` and `session.audit` add. The rules of the order:
    - two cases of one group (two questions about one needle) never run at once, in any configuration;
    - the session may hold more back (`Session.admits`: file versions that must be read in time order, an exec that holds the
      GPU alone) and takes its state out when an exec ends (`Session.reset`, before the exec's working folder is emptied), so
      no exec can see another's.
    Each result records the seconds the question was held back by a rule while first in line with a slot free (`held_s`), what
    the session served (`Served.record`) and what it did when the exec ended. The final message of an exec is read in
    `answer_format` (core/answer.py), and its result holds the format's fields (core/grounded.py).
    Beside other cells (`beside`, `--side-by-side`, core/schedule.py) an exec waits for a working folder of the run's shared slots when it
    starts (so a started case of `busy` has no slot until then) and for its agent's turn before the agent runs; none starts once any cell
    has ended on an error. The cases of a cell whose system does not cut its index to a horizon start longest first when earlier runs say
    how long they took (the cell says so when they do not). The agent of an exec with its own tools is kept off the GPU when beside other
    cells or when the system needs it (`schedule.agent_off_gpu`).
    An exec is held to `run.timeout_s` on the working clock (`clock`, core/clock.py): its seconds on the wall less the ones its session says
    it waited in the session's queue (`Session.queue_s`), or on the wall where the session cannot say (or with `clock="wall"`), and to a
    wall cap of three times that. Each result says which (`clock`, `working_s`, `queue_s`; its `seconds` stay the wall time). A session that
    cannot say how long a case waits is refused more than one exec at once on the working clock (`clock.check_queue`)."""
    by_qid = {q.qid: q for q in questions}
    cases = [case_of(run, name, config, q, session) for q in questions]
    pending = sorted((c for c in cases if not is_done(load_result(run.result_path(name, config, c.qid)))),
                     key=lambda c: (c.until, c.qid))
    if not pending:
        return
    off_gpu = agent_off_gpu(config.own_tools, description, beside is not None)
    if beside is not None:
        if beside.expected is None and not keeps_time_order(description):
            print(f"{name} {config.dir}: no earlier exec of this cell to order its cases longest first: they keep their order", flush=True)
        pending = start_order(pending, description, beside.expected)
    if (at_once := session.max_jobs(cases, jobs)) < jobs:
        print(f"{name} {config.dir}: {at_once} execs at once, not {jobs}", flush=True)
    jobs = at_once
    check_queue(f"{name} {config.name}", session, pending[0], jobs, clock)
    data_halt = Halt()  # a configuration with tools of its own: it alone stops when the data changed
    free = list(range(jobs))  # the working slots not in use
    running: dict = {}  # future -> (case, slot)
    held: dict[str, float] = {}  # qid -> when it was first in line with a slot free and a rule kept it from starting

    def stopped() -> bool:
        return bool(halt.reason or data_halt.reason or (beside is not None and beside.stop.reason))

    def one(case: Case) -> None:
        """An exec, in the working slot its case was given, or in the one it waits for beside other cells."""
        if beside is None:
            return execute(case)
        with beside.slots.execs.reserve(beside.priority) as slot:
            execute(dataclasses.replace(case, slot=slot, work=run.slot(slot)))

    def execute(case: Case) -> None:
        q = by_qid[case.qid]
        since = held.pop(case.qid, None)
        waited = 0.0 if since is None else time.monotonic() - since
        fs.empty(case.work)
        if case.run_dir.exists():
            fs.remove_tree(case.run_dir)
        served = session.serve(case)
        data = run.scenario.view(q, case.slot, dump, run.temp, run.tag, True) if config.own_tools else None
        if config.own_tools and data is None:
            raise ValueError(f"{run.scenario.name} gave the tools of {config.name} no folder to read")
        access = Access(config.own_tools, data, served, answer_format)
        prompt = run.scenario.prompt(q, access)
        queue = (lambda: session.queue_s(case)) if clock == WORKING else unreported
        spec = SessionSpec(prompt, case.run_dir, case.work, config.own_tools, data, run.bin, served,
                           run.deny(case.work, isolation, reads_of(data, dump), off_gpu), isolation, run.timeout_s, queue, None, None, off_gpu)
        out = None
        try:
            with beside.slots.agents[name].reserve(beside.priority) if beside is not None else nullcontext():
                if beside is not None and stopped():  # a cell ended on an error, or this agent's plan ran out, while it waited for its turn
                    return
                out = agent.run(spec)
            final = parse_final(out.get("final") or "", answer_format)
            calls = out.get("tools") or []
            rec = {**record(q, name, config, run.seed, case.run_dir.relative_to(run.folder).as_posix(), now_text(),
                            {**out, "answer": final.answer}),
                   "timeout_s": run.timeout_s, **figures_of(out), **run.scenario.after_exec(q, out, dump, access)}
            audit = dict(session.audit(case, calls))
            rec["work"] = {**rec["work"], **audit.pop("work", {})}
            rec.update(audit)
        finally:
            reset = session.reset(case, (out or {}).get("tools") or [], max(0, len(running) - 1))
            try:
                fs.empty(case.work)  # what the exec wrote is gone before another exec can reach it
            except OSError:
                pass  # the next exec in this slot empties it again before it starts
        rec.update(held_s=round(waited, 1), **served.record, **reset)
        if config.own_tools and (bad := fs.changed(run.data, run.manifest)):
            rec.update(problem=f"escaped: the data changed: {bad}", passed=False)
            data_halt.set(f"{name}'s {config.name} arm changed the data ({bad})")
        if description is not None and description.needs_gpu and (lost := gpu.gpu_lost()):  # a setup problem: asked again
            rec.update(problem=f"setup: the GPU was lost, so {config.system}'s model may not have read: {lost}", passed=False)
        rec.update(grounded.result_fields(q, final, answer_format, rec["passed"], dump))
        fs.write_json(run.result_path(name, config, case.qid), rec)
        print(f"{name} {config.dir} {case.qid}: {'pass' if rec['passed'] else 'FAIL'} {rec['seconds']:g} s"
              f"{'  ' + rec['problem'] if rec['problem'] else ''}", flush=True)
        halt.after(name, rec)

    with ThreadPoolExecutor(jobs) as pool:
        try:
            while True:
                while free and not stopped() and (c := next_case(pending, [b for b, _ in running.values()], session)) is not None:
                    pending.remove(c)
                    k = free.pop(0)
                    started = c if beside is not None else dataclasses.replace(c, slot=k, work=run.slot(k))
                    running[pool.submit(one, started)] = (started, k)
                if free and pending and not stopped():
                    held.setdefault(pending[0].qid, time.monotonic())  # first in line, a slot free, and a rule says wait
                if not running:
                    return
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    free.append(running.pop(future)[1])
                    future.result()
        except BaseException:  # an error, or Ctrl+C: nothing more starts, the running execs end
            data_halt.set("stopped by an error or an interrupt")
            raise


def note_configs(run: RunFolder, descriptions: Mapping[Config, Description | None], controls: str | None) -> None:
    """systems.json: for each configuration of this run that has a system, the modalities its system reads (`readable`) and the ones
    the configuration is asked about (`asked`: `asked_modalities`), so that the report counts only those as asked, and its
    `description` (the system's version and the hash of its configuration, which say what produced the results). Kept for the
    configurations of earlier commands. A configuration that has results in the run is refused when its `config_hash` is not the one
    noted: its results would come from two configurations of the system, and no file would say which is which."""
    file = run.folder / "systems.json"
    known = fs.read_json(file) if file.exists() else {}
    for config, description in descriptions.items():
        noted = known.get(config.name, {}).get("description")
        if description is not None and noted is not None and noted["config_hash"] != description.config_hash and any(
                (run.folder / "results").glob(f"*/{config.dir}/*.json")):
            raise ValueError(f"{config.name} has results in {run.name} from {description.name} {noted['version']} (config_hash "
                             f"{noted['config_hash'][:12]}), and it is {description.version} (config_hash {description.config_hash[:12]}) now: "
                             "its version, skill, notes or model settings changed, so its results would come from two configurations. "
                             f"Build a new run, or remove results/*/{config.dir} to run it again")
    for config, description in descriptions.items():
        if description is not None:
            known[config.name] = {"system": config.system, "variant": config.variant, "dir": config.dir,
                                  "readable": sorted(description.reads), "asked": asked_modalities(config, description, controls),
                                  "description": {**dataclasses.asdict(description), "reads": sorted(description.reads)}}
    fs.write_json(file, known)


def run_agents(run: RunFolder, agents: Mapping[str, Agent], configs: Sequence[Config], dump: Dump, store_of: Callable[[str], Path],
               jobs: int = JOBS, *, controls: str | None = None, subset: str = "all", only: Sequence[str] = (),
               load: Callable[[str], System] = systems.load, isolation: Isolation | None = None,
               answer_format: AnswerFormat, side_by_side: SideBySide | None = None, clock: str = DEFAULT_CLOCK) -> None:
    """The configurations with no variant (`blind`, `files`: nothing for the session to hold) of every agent over their questions
    (`config_questions`: all of them for a configuration with tools of its own or no system, else those of the modalities its
    system reads, the text and records only for the configuration `controls` names, or every one when it is None), then for each
    variant, for each agent, that variant's configurations, each with a session of its system opened and closed around it.
    `jobs` at a time (`Session.max_jobs`). `store_of(system)` is where the system's ingested store is, `dump` what was ingested,
    `load` gives a system by name and `isolation` is what the systems declare (by default all that `systems` has). The final answers
    are asked in `answer_format`, and an exec is held to its time on `clock` (see `run_config`). A system whose extra is missing fails here, before anything runs. An agent that stops (its plan is out, or at STOP_AT) leaves its remaining
    execs for the next command.
    Without `side_by_side` the cells run one after another (`cells_in_order`), and nothing else about a run changes. With it (core/schedule.py)
    they run beside each other: `jobs` still bounds the execs of one cell, and the pools bound all of them (one cell that needs the GPU at a
    time, at most `Pools.cpu` execs at once in all, at most `Pools.agents[a]` sessions of agent `a`). An agent that stops, or a lost GPU,
    stops every cell of that agent; an error in any cell stops all of them, each cell closes its session, and the first error is raised."""
    if not 1 <= jobs <= JOBS_MAX:
        raise ValueError(f"jobs is 1 to {JOBS_MAX}, not {jobs}: more crashed the laptop once")
    if len({c.dir for c in configs}) != len(configs):
        raise ValueError("these configurations share a results folder (two variants that differ only in characters a folder name "
                         f"cannot hold): {', '.join(c.name for c in configs)}")
    if side_by_side is not None and (missing := [n for n in agents if n not in side_by_side.pools.agents]):
        raise ValueError(f"the pools of a side-by-side run name no limit for {', '.join(missing)}")
    isolation = isolation or systems.isolation()
    loaded, descriptions = describe_configs(configs, load)  # a missing extra fails here
    questions = selected(run.questions(), subset, descriptions, only)
    view = run.scenario.view(questions[0], 0, dump, run.temp, run.tag, False) if questions and any(c.own_tools for c in configs) else None
    run.check_paths(isolation, reads_of(view, dump))
    run.ensure_bin()
    note_configs(run, descriptions, controls)
    halts = {name: Halt(agent.must_report_use) for name, agent in agents.items()}
    if agents and (bad := fs.changed(run.data, run.manifest)):
        raise RuntimeError(f"the data changed before the run ({bad}): build a new run")

    def cell(name: str, config: Config, beside: Beside | None = None) -> None:
        mine = config_questions(questions, config, descriptions[config], controls)
        if halts[name].reason or all(is_done(load_result(run.result_path(name, config, q.qid))) for q in mine):
            return
        if (description := descriptions[config]) is not None:
            gpu.check_gpu(description)
        context = SessionContext(agent=name, variant=config.variant, timeout_s=run.timeout_s, homes=run.folder / "homes", wall=time.time)
        session = (loaded[config.system].open(store_of(config.system), dump, context) if config.system is not None
                   else PlainSession())
        try:
            run_config(run, name, agents[name], config, description, mine, session, dump, isolation, jobs, halts[name], answer_format,
                       beside=beside, clock=clock)
        except BaseException:
            close_after_error(name, config, session)
            raise
        named = session.close()
        for what, items in named.items():
            file = run.folder / "results" / name / config.dir / f"{what}.json"
            if items:
                fs.write_json(file, items)
                print(f"{name} {config.name}: {len(items)} {what}: see {file}", flush=True)
            elif file.exists():
                file.unlink()

    try:
        if side_by_side is None:
            for name, config in cells_in_order(agents, configs):
                cell(name, config)
        else:
            pools = side_by_side.pools
            print(f"side by side: on. The cells run beside each other: one that needs the GPU at a time, at most {pools.cpu} execs at once "
                  f"and {', '.join(f'{n} of {a}' for a, n in pools.agents.items())}; an agent with its own tools sees no CUDA device",
                  flush=True)
            run_beside([Task(f"{name} {config.name}", bool(descriptions[config] and descriptions[config].needs_gpu),
                             side_by_side.expected.get((name, config.dir)), lambda beside, n=name, c=config: cell(n, c, beside))
                        for name, config in cells_in_order(agents, configs)], pools)
    finally:
        run.scenario.release(run.temp, run.tag)


def cells_in_order(agents: Iterable[str], configs: Sequence[Config]) -> list[tuple[str, Config]]:
    """The cells of a run, an agent's configuration each, in the order they run one after another: the configurations with no variant
    (`blind`, `files`), for each agent; then for each system and variant in turn, for each agent, that variant's configurations."""
    names = list(agents)
    return [*((name, config) for name in names for config in configs if config.variant is None),
            *((name, config) for system, variant in dict.fromkeys((c.system, c.variant) for c in configs if c.variant is not None)
              for name in names for config in configs if (config.system, config.variant) == (system, variant))]


def close_after_error(name: str, config: Config, session: Session) -> None:
    """Closes the session of a cell that ended on an error, which is the one the command tells: a close that fails as well (its system
    is gone, which may be why the cell failed) is told beside it and does not take its place."""
    try:
        session.close()
    except Exception as error:  # noqa: BLE001
        print(f"{name} {config.name}: closing its session failed as well: {error}", flush=True)


def real_agents() -> dict[str, Agent]:
    """Claude Code and Codex, each as its function of a `SessionSpec`. Running one spends the owner's plan."""
    return {"claude": Agent(claude.ask_claude), "codex": Agent(codex.ask_codex, must_report_use=True)}
