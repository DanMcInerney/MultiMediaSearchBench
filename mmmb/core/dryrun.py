"""What a command says before it spends anything (moved from `dry_run`, `argv_lines`, `show_decider` and the system-specific lines of
`show_agents` of bench.py of this repository at bb498a6): the first exec of each agent and configuration prepared in a folder of its own
and its command line printed with nothing run, the limits only a system can state (execs at once, work that cannot finish inside an exec),
and the retrieval asks still to run with about how long they take.

Nothing here opens a session: a system's `preview` answers from its store's files, so a dry run starts no daemon and loads no model, and
this module names no system."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mmmb.core import grounded, render
from mmmb.core.agents import claude, codex
from mmmb.core.agents.spec import SessionSpec
from mmmb.core.answer import AnswerFormat
from mmmb.core.atonce import asked, execs_at_once, gpu_warning
from mmmb.core.clock import check_queue, unreported
from mmmb.core.config import Config
from mmmb.core.contract import BEYOND_TIMEOUT, GPU_ESTIMATE_S, Description, Dump, Isolation, PlainSession, Session, SessionContext, System
from mmmb.core.probe import probe_line, probe_of
from mmmb.core.question import Question
from mmmb.core.retrieval import FOLDER
from mmmb.core.runfolder import IN_ONLY, OFF_GPU_DENY, OFF_GPU_ENV, RunFolder
from mmmb.core.runner import case_of, reads_of
from mmmb.core.scenario import Access
from mmmb.core.schedule import agent_off_gpu
from mmmb.core.select import config_questions, selected

RETRIEVAL_AGENT = "retrieval"  # the agent of a retrieval run's sessions: no agent execs, and its home is `homes/retrieval-<variant>`


def previewer(run: RunFolder, loaded: Mapping[str, System], store_of: Callable[[str], Path], dump: Dump) -> Callable[[str, Config], Session]:
    """A function of (agent, configuration) giving the preview session of the configuration's system for that agent, or a plain session
    for a configuration with no system (`blind`)."""
    def preview(agent: str, config: Config) -> Session:
        if config.system is None:
            return PlainSession()
        context = SessionContext(agent=agent, variant=config.variant, timeout_s=run.timeout_s, homes=run.folder / "homes", wall=time.time)
        return loaded[config.system].preview(store_of(config.system), dump, context)

    return preview


def warn_of_probe(run: RunFolder) -> None:
    """Says so, first, when the run's shortcut probe did not pass (see `probe_line`)."""
    if line := probe_line(run.name, probe_of(run.folder)):
        print(f"warning: {line}")


def show_limits(run: RunFolder, names: Sequence[str], configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
                preview: Callable[[str, Config], Session], jobs: int, gpu_jobs: int, controls: str | None = None, subset: str = "all",
                only: Sequence[str] = ()) -> None:
    """What a system says of the configurations it serves before a run, from its preview session: the execs at once when it lowers them
    (`Session.max_jobs`: a slow model, a cut of the whole index to a horizon) and, when `--gpu-jobs` asked for several on a system that
    needs the GPU, the warning that they run at once (`atonce.gpu_warning`); and the questions whose work may need more GPU seconds
    than an exec leaves it (`contract.GPU_ESTIMATE_S` and `BEYOND_TIMEOUT` of `Served.record`), where a timeout is the system's speed
    and not a miss."""
    questions = selected(run.questions(), subset, descriptions, only)
    for config in configs:
        if config.system is None:
            continue
        session = preview(names[0], config)
        cases = [case_of(run, names[0], config, q, session) for q in config_questions(questions, config, descriptions[config], controls)]
        if (at_once := execs_at_once(descriptions[config], session, cases, jobs, gpu_jobs)) < (want := asked(descriptions[config], jobs, gpu_jobs)):
            print(f"  {config.name}: {at_once} execs at once, not {want}")
        if warning := gpu_warning(config.name, descriptions[config], at_once):
            print(f"  {warning}")
        records = [(c.qid, session.serve(dataclasses.replace(c, slot=0, work=run.slot(0))).record) for c in cases]
        if past := [(qid, record[GPU_ESTIMATE_S]) for qid, record in records if record.get(BEYOND_TIMEOUT)]:
            print(f"  {config.name}: its work over the whole range of {len(past)} questions may need more GPU seconds than the "
                  f"{render.ASK_SHARE * run.timeout_s:g} s an exec leaves it ({render.ASK_SHARE:.0%} of its {run.timeout_s:g} s: the "
                  "agent's own turns take the rest), so a timeout there is its speed, not a miss: "
                  + ", ".join(f"{qid} ({seconds:.0f} s)" for qid, seconds in past))


def check_clocks(run: RunFolder, names: Sequence[str], configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
                 preview: Callable[[str, Config], Session], jobs: int, gpu_jobs: int, controls: str | None, subset: str, only: Sequence[str],
                 clock: str) -> None:
    """Raises (`clock.check_queue`) for the first configuration whose system would run more than one exec at once on the working
    clock and cannot report its queue, from the preview sessions: a run is refused before it spends anything."""
    questions = selected(run.questions(), subset, descriptions, only)
    for config in configs:
        if config.system is None:
            continue
        session = preview(names[0], config)
        cases = [case_of(run, names[0], config, q, session) for q in config_questions(questions, config, descriptions[config], controls)]
        if cases:
            check_queue(config.name, descriptions[config], session, cases[0], execs_at_once(descriptions[config], session, cases, jobs, gpu_jobs),
                        clock)


def minutes_text(minutes: float) -> str:
    return "under a minute" if minutes < 1 else f"about {minutes:.0f} min"


def show_retrieval(run: RunFolder, configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
                   preview: Callable[[str, Config], Session], questions: Sequence[Question]) -> None:
    """The retrieval asks to run (`mmmb retrieve`), each configuration's: how many of the questions it reads have a retrieval run and no
    record yet, and about how long they take, from the preview session's estimate of each ask's sources and ranges."""
    total, minutes = 0, 0.0
    print(f"run {run.name}: {run.scale}, seed {run.seed}, {len(run.days)} days")
    warn_of_probe(run)
    for config in configs:
        mine = [q for q in config_questions(questions, config, descriptions[config]) if q.retrieval is not None]
        folder = run.folder / "results" / FOLDER / config.dir
        todo = [q for q in mine if not (folder / f"{q.qid}.json").exists()]
        session = preview(RETRIEVAL_AGENT, config)
        spent = sum(session.estimate(q.retrieval.sources, q.retrieval.ranges) for q in todo) / 60
        total += len(todo)
        minutes += spent
        print(f"  {config.name:26} {len(todo)} retrieval asks to run of {len(mine)}, {minutes_text(spent)}")
    print(f"{total} asks to run, {minutes_text(minutes)} in all (each range's windows at the system's seconds a window; opening a "
          "system is extra)")


def argv_lines(argv: list[str]) -> list[str]:
    """A command line as lines: each option with its values, quoted as a shell would need."""
    lines: list[str] = []
    for arg in argv:
        text = subprocess.list2cmdline([arg])
        if not lines or (arg.startswith("-") and len(arg) > 1):
            lines.append(text)
        else:
            lines[-1] += " " + text
    return lines


def environment_lines(name: str, spec: SessionSpec, isolation: Isolation) -> list[str]:
    """What the agent's process is given and kept from, for a dry run to print."""
    off = [f"kept off the GPU: {', '.join(w.removeprefix(IN_ONLY) for w in OFF_GPU_DENY)} are refused in its commands"] if spec.off_gpu else []
    if name == "claude":
        child = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, isolation, spec.off_gpu)
        dropped = [*claude.ENV_DROP, *([*claude.ENV_DROP_FILES, *(f"{p}*" for p in isolation.env_drop_prefixes)] if spec.own_tools else [])]
        return [f"drop {', '.join(dropped)}", *(f"set {k}={v}" for k, v in claude.ENV_SET.items()),
                *(f"set {k}={child[k]}" for k in ("PATH", "CLAUDE_CODE_GIT_BASH_PATH", *isolation.env_nowhere, *(OFF_GPU_ENV if spec.off_gpu else ()))
                  if spec.own_tools and k in child), *off]
    return ["Codex reads its own login (~/.codex/auth.json is never read or copied)",
            *([f"its shell's PATH and {', '.join(isolation.env_nowhere)} are the shell_environment_policy settings above"]
              if spec.own_tools else []),
            *([f"so is its shell's {', '.join(OFF_GPU_ENV)}"] if spec.off_gpu else []), *off]


def dry_run(run: RunFolder, names: Sequence[str], configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
            preview: Callable[[str, Config], Session], dump: Dump, isolation: Isolation, controls: str | None = None, subset: str = "all",
            only: Sequence[str] = (), efforts: Mapping[str, str | None] | None = None, models: Mapping[str, str | None] | None = None, *,
            answer_format: AnswerFormat, side_by_side: bool = False) -> None:
    """The first question of each agent and configuration prepared in a folder of its own under `dry-run` (an exec folder may hold a
    run's record), its command line and environment printed, nothing run and no session opened. `efforts` and `models` are those asked
    of each agent (None: its own). Its prompt asks for the final answer in `answer_format`. With `side_by_side` an agent with its own tools is
    kept off the GPU, as in the run (`schedule.agent_off_gpu`). Codex's catalog comes from `codex debug models`, which only reads."""
    questions = selected(run.questions(), subset, descriptions, only)
    if answer_format == "json":
        grounded.check_groundable(questions, dump)  # (as the run refuses it)
    run.check_paths(isolation, reads_of(run.scenario.view(questions[0], 0, dump, run.temp, run.tag, False), dump) if questions else None)
    for name in names:
        found = shutil.which(name)
        command = claude.claude_command() if name == "claude" and found else [found or name]
        for config in configs:
            mine = config_questions(questions, config, descriptions[config], controls)
            if not mine:
                print(f"\n== {name} {config.dir}: no question to ask it")
                continue
            q, session = mine[0], preview(name, config)
            run_dir, work = run.folder / "dry-run" / name / config.dir / q.qid, run.slot(0)
            served = session.serve(dataclasses.replace(case_of(run, name, config, q, session), slot=0, work=work))
            data = run.scenario.view(q, 0, dump, run.temp, run.tag, False) if config.own_tools else None
            text = run.scenario.prompt(q, Access(config.own_tools, data, served, answer_format, run.timeout_s))
            effort, model = (efforts or {}).get(name), (models or {}).get(name)
            off_gpu = agent_off_gpu(config.own_tools, descriptions[config], side_by_side)
            spec = SessionSpec(text, run_dir, work, config.own_tools, data, run.bin, served, run.deny(work, isolation, reads_of(data, dump), off_gpu),
                               isolation, run.timeout_s, unreported, model, effort, off_gpu)
            if name == "claude":
                argv = claude.prepare(spec, command)
            else:
                argv = codex.prepare(spec, model or codex.model(), effort or codex.effort(), command=command)
            (run_dir / "prompt.txt").write_text(text, encoding="utf-8")
            print(f"\n== {name} {config.dir}, question {q.qid}\n   exec folder {run_dir}\n   working folder {work}")
            print("   command:")
            for line in argv_lines(argv):
                print("     " + line)
            has = [*([f"{run.bin} first on PATH, the data is read-only"] if config.own_tools else []),
                   *([f"{served.server.name}'s tools"] if served.server else [])]
            print("   environment: " + ("; ".join(has) or "no tools"))
            for line in environment_lines(name, spec, isolation):
                print(f"     {line}")
            print(f"   prompt ({run_dir / 'prompt.txt'}): {text}")
    print("\nnothing was run")
