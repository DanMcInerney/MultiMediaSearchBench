"""What a run will ask of the machine, said before it starts (moved from bench.py of this repository at bb498a6): how many execs
each agent and configuration has left, and about how long that takes. It asks nothing and opens no session (a system's `preview` says how
many execs it allows at once). With a `history` (DESIGN 8.2, plan 6.1) the estimate is from the medians of what the same cell took in
earlier runs: seconds, GPU seconds, tokens and dollars an exec. A cell with no earlier exec is counted at EXEC_MIN and said so; a median
is the typical exec, and a few long ones (a timeout is 900 s) make a run take longer, so the mean is told beside it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mmmb.core import fs
from mmmb.core.atonce import GPU_JOBS_FLAG, JOBS_FLAG, execs_at_once, on_gpu
from mmmb.core.build import store_of
from mmmb.core.cache import CACHED, INGESTED, LOST, dump_state, store_state
from mmmb.core.config import Config
from mmmb.core.contract import Case, Description, Session
from mmmb.core.question import Question
from mmmb.core.retrieval import FOLDER
from mmmb.core.runfolder import RunFolder, is_done, load_result
from mmmb.core.runner import case_of, cells_in_order
from mmmb.core.schedule import Expected, Line, Pools, simulate, start_order
from mmmb.core.select import config_questions, selected
from mmmb.core.stats import mean, median, token_total

EXEC_MIN = 2.5  # minutes an exec takes, for the estimate printed before a run, where no earlier run says
BLIND_MIN = 1.0  # ... and a blind one, which runs no tool
ONLY_NAMED = 8  # the qids of a selection that the header names; more are counted


def pending_count(run: RunFolder, agent: str, config: Config, questions: Sequence[Question]) -> tuple[int, int]:
    todo = sum(1 for q in questions if not is_done(load_result(run.result_path(agent, config, q.qid))))
    return todo, len(questions)


def estimate(cells: Sequence[tuple], jobs: int) -> str:
    """About how long the execs take: `cells` is (execs, execs at once) for each configuration, at EXEC_MIN minutes an exec, or
    (execs, execs at once, minutes an exec) for one with its own figure (`blind`: BLIND_MIN)."""
    hours = sum(n * (minutes[0] if minutes else EXEC_MIN) / at_once / 60 for n, at_once, *minutes in cells)
    blind = f" ({BLIND_MIN:g} for a blind one)" if any(a[2:] == (BLIND_MIN,) for a in cells) else ""
    return f"about {hours:.1f} h at {EXEC_MIN:g} min an exec{blind} over {jobs} jobs"


# -- earlier runs -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Figures:
    """What an exec took by one statistic: seconds, and the GPU seconds, tokens and dollars where results say (None: none does; Codex
    reports no dollars)."""

    seconds: float
    gpu_s: float | None
    tokens: float | None
    usd: float | None


@dataclass(frozen=True)
class Typical:
    """What one exec of a cell (an agent and a configuration) took in earlier runs, over its done results: the medians (the typical
    exec) and the means (which a few long execs raise, and which add up to what those runs took in all)."""

    n: int
    median: Figures
    mean: Figures
    by_qid: Mapping[str, float] = field(default_factory=dict, compare=False)  # the median seconds of each question that has a done result


def typical_of(results: Sequence[dict]) -> Typical | None:
    """The typical exec of the done results among `results` (a `setup` or `error` one did not run to an answer); None if none."""
    done = [r for r in results if is_done(r)]
    if not done:
        return None
    gpu = [r["work"]["seconds"] for r in done if r.get("work")]
    tokens = [t for r in done if (t := token_total(r)) is not None]
    usd = [r["cost_usd"] for r in done if r.get("cost_usd") is not None]

    def by(statistic) -> Figures:
        return Figures(statistic([r["seconds"] for r in done]), statistic(gpu), statistic(tokens), statistic(usd))

    seconds: dict[str, list[float]] = defaultdict(list)
    for r in done:
        seconds[r["qid"]].append(r["seconds"])
    return Typical(len(done), by(median), by(mean), {qid: median(each) for qid, each in seconds.items()})


def history_of(folders: Sequence[Path]) -> dict[tuple[str, str], Typical]:
    """The typical exec of each cell, `(agent, Config.dir)`, over the results of the run `folders` (read as files: nothing is opened,
    so a run of a scenario that is gone is still history). A file is a result when it is named after the qid it holds."""
    found: dict[tuple[str, str], list[dict]] = {}
    for folder in folders:
        root = Path(folder) / "results"
        for file in sorted(root.glob("*/*/*.json")):
            agent, cell = file.relative_to(root).parts[:2]
            if agent != FOLDER and isinstance(result := fs.read_json(file), dict) and result.get("qid") == file.stem:
                found.setdefault((agent, cell), []).append(result)
    return {key: typical for key, results in found.items() if (typical := typical_of(results)) is not None}


def expected_of(history: Mapping[tuple[str, str], Typical]) -> dict[tuple[str, str], Expected]:
    """What the cases of each cell took in earlier runs, for the order a side-by-side run starts them in (core/schedule.py)."""
    return {key: Expected(t.by_qid, t.median.seconds) for key, t in history.items()}


def earlier_runs(folder: Path, scenario: str, scale: str) -> list[Path]:
    """The runs in `folder` (a runs folder) of this scenario at this scale, as their run.json say: what an exec took there is what it
    takes here, which another scale's does not say."""
    found = []
    for run_json in sorted(Path(folder).glob("*/run.json")):
        meta = fs.read_json(run_json)
        if meta.get("scenario") == scenario and meta.get("scale") == scale:
            found.append(run_json.parent)
    return found


def median_figure(folders: Sequence[Path], file: str, key: str) -> float | None:
    """The median of `key` in the JSON `file` (relative to a run folder) over the folders that have it; None if none does."""
    held = [fs.read_json(Path(f) / file) for f in folders if (Path(f) / file).exists()]
    return median([h[key] for h in held if key in h])


# -- the estimate -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """The execs an agent has left in one configuration, `at_once` of them at a time, and what one took in earlier runs."""

    agent: str
    config: Config
    todo: int
    of: int
    at_once: int
    typical: Typical | None = None
    waiting: tuple[Case, ...] = ()  # the cases still to run, in ascending `until`
    description: Description | None = None

    @property
    def name(self) -> str:
        return f"{self.agent} {self.config.name}"

    @property
    def guess_s(self) -> float:
        return 60 * (BLIND_MIN if self.config.system is None else EXEC_MIN)


@dataclass(frozen=True)
class Totals:
    """The execs still to run, added up by one statistic: session seconds, the wall time they take (execs at once), and the GPU
    seconds, tokens and dollars of the cells that have an earlier exec to say."""

    session_s: float = 0.0
    wall_s: float = 0.0
    gpu_s: float = 0.0
    tokens: float = 0.0
    usd: float = 0.0


@dataclass(frozen=True)
class Forecast:
    execs: int
    median: Totals
    mean: Totals
    guessed: tuple[str, ...]  # the cells with execs to run and no earlier exec: counted at EXEC_MIN, with nothing else


def forecast(cells: Sequence[Cell]) -> Forecast:
    totals = {"median": Totals(), "mean": Totals()}
    execs, guessed = 0, []
    for c in cells:
        if not c.todo:
            continue
        execs += c.todo
        if c.typical is None:
            guessed.append(c.name)
        for statistic, sofar in totals.items():
            f = getattr(c.typical, statistic) if c.typical is not None else Figures(c.guess_s, None, None, None)
            totals[statistic] = Totals(sofar.session_s + c.todo * f.seconds, sofar.wall_s + c.todo * f.seconds / c.at_once,
                                       sofar.gpu_s + c.todo * (f.gpu_s or 0.0), sofar.tokens + c.todo * (f.tokens or 0.0),
                                       sofar.usd + c.todo * (f.usd or 0.0))
    return Forecast(execs, totals["median"], totals["mean"], tuple(guessed))


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def cell_line(c: Cell, jobs: int, history: bool) -> str:
    line = f"  {c.agent:6} {c.config.name:24} {c.todo} to run of {c.of}" + (f" ({c.at_once} at once)" if c.at_once != jobs else "")
    if not history:
        return line
    t = c.typical
    if t is None:
        return line + f"  no earlier exec: {c.guess_s:.0f} s assumed"
    m = t.median
    return (line + f"  median {m.seconds:.0f} s" + (f", {m.gpu_s:.0f} GPU s" if m.gpu_s else "") + (f", {m.tokens / 1000:.0f}k tokens" if m.tokens else "")
            + (f", ${m.usd:.2f}" if m.usd else "") + f" ({t.n} earlier)")


def totals_text(t: Totals) -> str:
    return (f"about {hours(t.wall_s)}, {t.session_s:,.0f} session seconds, {t.gpu_s:,.0f} GPU seconds, {t.tokens / 1e6:.1f}M tokens, "
            f"${t.usd:.2f} at list")


def show_cells(cells: Sequence[Cell], jobs: int, gpu_jobs: int, history: bool) -> None:
    """A line for each cell and the estimate. A cell whose execs at once are not `jobs` says so, and the line after the cells says which
    configurations need the GPU and so run `gpu_jobs` (`atonce`). Without `history` (the earlier runs were not looked at) every exec is at
    EXEC_MIN; with it, the totals at the medians of earlier runs (DESIGN 8.2) and at their means, which are what those runs took in all
    (the medians leave out the few long execs, a timeout is 900 s), and the cells that have no earlier exec."""
    for c in cells:
        print(cell_line(c, jobs, history))
    if gpu := sorted({c.config.name for c in cells if c.todo and on_gpu(c.description)}):
        print(f"  execs at once: {jobs} ({JOBS_FLAG}); {gpu_jobs} for those that need the GPU, {', '.join(gpu)} ({GPU_JOBS_FLAG}); "
              "a system may allow fewer")
    f = forecast(cells)
    old = estimate([(c.todo, c.at_once, *((BLIND_MIN,) if c.config.system is None else ())) for c in cells], jobs)
    if not history or len(f.guessed) == sum(1 for c in cells if c.todo):  # no cell with execs to run has an earlier exec to say
        print(f"{f.execs} execs to run, {old}" + (" (no earlier exec of these cells to take a median from)" if history else ""))
        return
    print(f"{f.execs} execs to run over {jobs} jobs (a system may allow fewer), from the earlier runs of these cells:")
    print(f"  at the medians, the typical exec: {totals_text(f.median)}")
    print(f"  at the means, what those runs took in all: {totals_text(f.mean)}")
    print("  (dollars are what an agent reports: Codex's use is on the plan, not in dollars)")
    if f.guessed:
        print(f"  no earlier exec for {', '.join(f.guessed)}: counted at {EXEC_MIN:g} min an exec ({BLIND_MIN:g} blind), with no GPU seconds, "
              "tokens or dollars")


def show_agents(run: RunFolder, names: Sequence[str], configs: Sequence[Config], descriptions: Mapping[Config, Description | None],
                preview: Callable[[str, Config], Session], jobs: int, gpu_jobs: int, controls: str | None = None, subset: str = "all",
                only: Sequence[str] = (), history: Mapping[tuple[str, str], Typical] | None = None, side_by_side: bool = False) -> None:
    """What `run_agents` will run, and how long that takes: it asks nothing. `descriptions` are those of `configs` (None for one
    with no system). Each configuration is counted at the execs at once `run_config` will run (`atonce.execs_at_once`: `jobs`, or
    `gpu_jobs` for a system that needs the GPU, or the fewer its session allows, from the `preview` session of its system), and said so. With a `history` (`history_of`, maybe empty) the
    estimate is from the medians of what each cell took there. With `side_by_side` it also says how long the same cases take run beside each
    other (`show_beside`)."""
    questions = selected(run.questions(), subset, descriptions, only)
    cells = []
    named = f", only {', '.join(only)}" if 0 < len(only) <= ONLY_NAMED else f", only {len(only)} chosen questions" if only else ""
    print(f"run {run.name}: {run.scale}, seed {run.seed}, {len(run.days)} days, {'all the' if subset == 'all' else f'the {subset}'} questions{named}")
    for name in names:
        for config in configs:
            mine = config_questions(questions, config, descriptions[config], controls)
            todo, of = pending_count(run, name, config, mine)
            session = preview(name, config)
            cases = [case_of(run, name, config, q, session) for q in mine]
            at_once = execs_at_once(descriptions[config], session, cases, jobs, gpu_jobs)
            waiting = tuple(sorted((c for c in cases if not is_done(load_result(run.result_path(name, config, c.qid)))),
                                   key=lambda c: (c.until, c.qid)))
            cells.append(Cell(name, config, todo, of, at_once, (history or {}).get((name, config.dir)), waiting, descriptions[config]))
    show_cells(cells, jobs, gpu_jobs, history is not None)
    if side_by_side:
        show_beside(cells, names, configs, history)


def show_beside(cells: Sequence[Cell], names: Sequence[str], configs: Sequence[Config], history: Mapping[tuple[str, str], Typical] | None) -> None:
    """What `--side-by-side` is expected to take, simulated (core/schedule.py) over the cases still to run, each at what it took in earlier
    runs (its own median, else its cell's) and a cell with none at its guess, beside the same cases one cell after another. Said not to be
    estimated when no cell with execs to run has an earlier exec to take durations from."""
    by = {(c.agent, c.config): c for c in cells}
    known = expected_of(history or {})
    lines, durations, guessed = [], {}, []
    for agent, config in cells_in_order(names, configs):
        c = by[agent, config]
        if c.waiting:
            expected = known.get((agent, config.dir))
            order = start_order(c.waiting, c.description, expected)
            lines.append(Line(c.name, agent, bool(c.description and c.description.needs_gpu), c.at_once, tuple(x.qid for x in order)))
            durations.update({(c.name, x.qid): expected.of(x.qid) if expected else c.guess_s for x in order})
            guessed += [] if expected else [c.name]
    if history is None or len(guessed) == len(lines):
        print("side by side (--side-by-side): not estimated, no earlier exec of these cells to take their durations from")
        return
    pools = Pools.of(names)
    beside = simulate(lines, durations, pools).makespan
    alone = sum(simulate([line], durations, pools).makespan for line in lines)
    print(f"side by side (--side-by-side): about {hours(beside)} at the medians, against {hours(alone)} one after another (the same cases "
          f"simulated both ways: one cell that needs the GPU at a time, at most {pools.cpu} execs at once)"
          + (f"; no earlier exec for {', '.join(guessed)}: counted at {EXEC_MIN:g} min an exec ({BLIND_MIN:g} blind)" if guessed else ""))


# -- the dump and the stores ------------------------------------------------------------------------------------------------------

def took(seconds: float | None, what: str) -> str:
    return f"about {seconds / 60:.0f} min, from earlier runs' {what}" if seconds is not None else f"no earlier {what} was recorded to say how long"


def show_builds(run: RunFolder, systems: Sequence[str]) -> None:
    """What has to be built before the run can start: its data, when the temp folder lost it (`run` writes it again from the recipe),
    and the stores of `systems`, which `run` never makes (they are there or it refuses)."""
    meta = fs.read_json(run.folder / "run.json")
    print(f"  data {run.data}: " + ("there" if run.data.is_dir() else "gone from the temp folder; it is written again from the recipe, "
                                    + took(meta.get("build_s"), "build time")))
    for name in systems:
        ingest = fs.read_json(store_of(run.folder, name) / INGESTED)
        print(f"  store of {name}: made ({ingest['seconds']:g} s to ingest)")


def show_new_builds(run: RunFolder, folders: Sequence[Path], descriptions: Mapping[str, Description], temp: Path, cache: Path) -> None:
    """What `mmmb repeat` makes first, for the recipe of `run` with its dump in `temp` and its stores (of the systems of `descriptions`,
    each one's description) in `cache`: what is cached is said so, with the seconds it took to make, and is not made again; what is not is
    made, and takes about the median of what it took in the earlier runs `folders` (a run records its `build_s`, and each store its
    ingest seconds). A dump the cache would refuse (a file changed) is said so. Nothing is read but names and times."""
    meta = fs.read_json(run.folder / "run.json")
    what = f"{meta['scenario']}, {run.scale}, seed {run.seed}, {len(run.days)} days"
    dump = dump_state(run.scenario, run.recipe, temp, cache)
    if dump.state == CACHED:
        print(f"  dump {dump.final.name}: cached ({what}): writing it took {dump.seconds:g} s, so it is not written again")
    elif dump.state == LOST:
        print(f"  dump {dump.final.name}: cached ({what}), and its data is gone from {temp} ({dump.detail}): it is written again from the "
              f"recipe, about {dump.seconds / 60:.0f} min, as long as it took before")
    elif dump.problem:
        print(f"  dump {dump.final.name}: the cache refuses it ({what}): {dump.problem}")
    else:
        print(f"  dump: written in the temp folder from the recipe ({what}), " + took(median_figure(folders, "run.json", "build_s"), "build time"))
    for name, description in descriptions.items():
        store = store_state(description, dump.key, cache)
        if store.cached and store.root != str(dump.final):
            print(f"  store of {name}: cached over the dump in {store.root}, and this dump is in {dump.final}: a build refuses it (use that --temp)")
        elif store.cached:
            print(f"  store of {name}: cached: ingesting it took {store.seconds:g} s, so it is not ingested again")
        else:
            print(f"  store of {name}: ingested from it, " + took(median_figure(folders, f"stores/{name}/{INGESTED}", "seconds"), "ingest time"))
