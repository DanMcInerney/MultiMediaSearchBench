"""`mmmb clock-check` (DESIGN 8.4, plan 8.2): the working clock tried before it is trusted. A few cases are asked of each agent twice, each
one alone and then all of them queued at once for the same system, and the answers, tokens and charged seconds of the two halves are
printed side by side. If the clock does what it says, a case is charged about the same working seconds and gives about the same answer
either way, while its wall seconds grow with the queue.

The cases are those that a system serves beside another (the ones its `serve` does not mark `exclusive`: a case that holds the system alone
queues for nothing), drawn with `select` and a fixed seed. The halves are two run folders under `<run>/clock-check/` that carry copies of
the run's own files and the run's dump and stores (the halves run in them, one after the other, resumable like any run). Nothing is asked
of an agent before the refusals: a system that cannot report its queue cannot be checked, since the queued half would have no charged time
to compare, and a system that would run the cases one at a time has no queue."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mmmb.core import fs, stats
from mmmb.core.config import Config
from mmmb.core.contract import Case, Description, Session
from mmmb.core.question import Question
from mmmb.core.runfolder import RunFolder, load_result
from mmmb.core.runner import case_of
from mmmb.core.select import Selection, config_questions, select, selected

CASES = 10  # how many cases the check asks of each agent, twice
FOLDER = "clock-check"
HALVES = ("alone", "queued")
COPIED = ("run.json", "questions.jsonl", "data-manifest.json")  # what a half's folder carries of the run's: its dump and stores are the run's own
EXCLUSIVE = "exclusive"  # the key of a case's `Served.record` that says the system serves it alone (the report counts it too)


def light_cases(run: RunFolder, agent: str, config: Config, description: Description | None, preview: Session) -> list[Case]:
    """The cases of `config` whose system serves them beside another: those a preview `serve` does not record as `exclusive`."""
    cases = [case_of(run, agent, config, q, preview) for q in config_questions(selected(run.questions(), "all", {config: description}), config, description)]
    return [c for c in cases if not preview.serve(dataclasses.replace(c, slot=0, work=run.slot(0))).record.get(EXCLUSIVE)]


def pick(run: RunFolder, config: Config, light: Sequence[Case], cases: int) -> list[Question]:
    """`cases` of the questions of `light` (all of them when there are fewer), the stratified sample of `select`, with the trail of the
    step printed as `run` does."""
    qids = {c.qid for c in light}
    trail: list[tuple[str, int]] = []
    chosen = select([q for q in run.questions() if q.qid in qids], Selection(limit=cases), trail)
    print("selection: " + ", then ".join(f"{step}: {n}" for step, n in trail))
    return list(chosen)


def check_possible(config: Config, preview: Session, cases: Sequence[Case], jobs: int) -> None:
    """Raises why the queued half would show nothing: the system cannot report its queue (more than one exec at once, and
    `Session.queue_s` is None: decision D3, a change in the system itself), or it would run the cases one at a time."""
    if not cases:
        raise ValueError(f"every case of {config.name} holds its system alone, so none of them queues: there is nothing to check")
    if jobs > 1 and preview.queue_s(cases[0]) is None:
        raise ValueError(f"{config.name}'s system reports no queue time (Session.queue_s is None; decision D3: the system itself must report the "
                         f"seconds each of its jobs waits), so the {jobs} queued execs would have no charged time to compare with the ones asked "
                         "alone. The clock-check waits for that change")
    if (at_once := preview.max_jobs(list(cases), jobs)) < 2:
        raise ValueError(f"{config.name}'s system would run these cases {at_once} at a time, not {jobs}, so nothing would queue: there is "
                         "nothing to check")


def branch(run: RunFolder, half: str) -> RunFolder:
    """The run for one half of the check: a folder `<run>/clock-check/<half>` holding copies of the run's own small files (run.json,
    the questions, the data manifest), opened as `run` is but there. Results and execs are its own; what is left over from an earlier
    command stays, and the half goes on from it."""
    if half not in HALVES:
        raise ValueError(f"a half is one of {', '.join(HALVES)}, not {half!r}")
    folder = run.folder / FOLDER / half
    folder.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        shutil.copyfile(run.folder / name, folder / name)
    return dataclasses.replace(run, folder=folder)


def run_halves(run: RunFolder, jobs: int, ask: Callable[[RunFolder, int], None]) -> dict[str, RunFolder]:
    """Each half asked in turn, `ask(half's run, execs at once)`: alone at one exec at a time, then queued at `jobs`."""
    halves = {}
    for half, at_once in zip(HALVES, (1, jobs)):
        halves[half] = branch(run, half)
        print(f"\nclock-check, {half}: {at_once} exec{'s' * (at_once != 1)} at once ({halves[half].folder})", flush=True)
        ask(halves[half], at_once)
    return halves


def cell(text: object, width: int) -> str:
    return f"{'-' if text is None else text}"[:width].rjust(width)


def side(result: dict | None) -> tuple[str, str, str]:
    """The answer, tokens and charged seconds of a result, as the table prints them ('-' where there is none)."""
    if result is None:
        return "-", "-", "-"
    tokens = stats.token_total(result)
    return str(result["answer"]), "-" if tokens is None else f"{tokens:,}", f"{result['working_s']:g}"


def show_sides(halves: Mapping[str, RunFolder], names: Sequence[str], config: Config, qids: Sequence[str]) -> None:
    """The two halves side by side, for each agent: a case in a line with its answer, tokens and charged (working) seconds alone and
    queued, and the seconds it waited and its wall seconds queued; then what the cases add up to."""
    alone, queued = halves["alone"], halves["queued"]
    for name in names:
        print(f"\n{name} {config.name}: {len(qids)} cases, each asked alone and then queued (answer, tokens, charged seconds)")
        print(f"  {'case':32} {'alone':>40}   {'queued':>40} {'queue s':>8} {'wall s':>7}")
        rows = [(qid, load_result(alone.result_path(name, config, qid)), load_result(queued.result_path(name, config, qid))) for qid in qids]
        for qid, one, many in rows:
            (a, t, w), (a2, t2, w2) = side(one), side(many)
            print(f"  {qid:32.32} {cell(a, 16)} {cell(t, 10)} {cell(w, 8)}   {cell(a2, 16)} {cell(t2, 10)} {cell(w2, 8)} "
                  f"{cell(None if many is None else many['queue_s'], 8)} {cell(None if many is None else many['seconds'], 7)}")
        both = [(one, many) for _, one, many in rows if one is not None and many is not None]
        if not both:
            print("  nothing was answered in both halves")
            continue
        same = sum(1 for one, many in both if one["answer"] == many["answer"])
        total = lambda key, results: sum(r[key] for r in results if r[key] is not None)  # noqa: E731
        tokens = [(stats.token_total(one), stats.token_total(many)) for one, many in both]
        tokens = [(a, b) for a, b in tokens if a is not None and b is not None]
        print(f"  the same answer in {same} of {len(both)} cases; tokens {sum(a for a, _ in tokens):,} alone, {sum(b for _, b in tokens):,} queued; "
              f"charged seconds {total('working_s', [o for o, _ in both]):.1f} alone, {total('working_s', [m for _, m in both]):.1f} queued; "
              f"wall seconds {total('seconds', [o for o, _ in both]):.1f} alone, {total('seconds', [m for _, m in both]):.1f} queued; "
              f"seconds waited in the queue {total('queue_s', [m for _, m in both]):.1f}")
        clocks = sorted({r["clock"] for pair in both for r in pair})
        print(f"  clock: {', '.join(clocks)}")
