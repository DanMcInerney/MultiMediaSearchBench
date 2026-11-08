"""The mm system (DESIGN 3): Multimem's memory, read by a local decision model through its daemon. `Mm` is the `System` (`describe`, `ingest`,
`open`); `MmSession` is what `run_config` calls for a run of one (agent, variant): it owns a copy of the ingested store and the daemon
over it, serves each case its server, skill and notes, holds cases back by the rules of the old `run_arm` (`admits`, `max_jobs`) and takes
an exec's state out of the home when it ends (`reset`).

    store/home/            the ingested home (`ingest`): the daemon is never run on it again, only on copies
    store/windows.json     what the dump costs the model in windows (estimate.py), counted from the dump when it was ingested
    <homes>/<agent>-<variant>/   one copy per (agent, variant), made by `open`, its daemon started and the model loaded

What the old `run_arm`, `run_agents` and `copy_home` did to the home, in their order, is split over the calls (plan 4.5, 4.6):
`open`: copy, start the daemon, load the model with one window, sweep the labels of an earlier run (`id:bench-*`); `preview` is `open`
for a dry run: the estimate, the jobs and the server from the store's files, with no copy, daemon or model;
`serve`: wait for the daemon to be idle when the case holds the GPU, write the project marker into the working folder, take out the state
of the execs that ended, cut the index to the case's horizon, and hand the agent the home's `looks` folder (`Served.opens`: where `look` writes);
`reset`: cancel and wait for the exec's jobs, take out the exec's state, mark it ended; `close`: list the labels no project owns, stop the daemon
(the state of the execs that ended is taken out by the next `serve`, or by the next `open` of the home)."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from multimem.adapter import INSTRUCTIONS
from multimem.project import MARKER

from mmsb.core import fs
from mmsb.core.contract import (BEYOND_TIMEOUT, EXCLUSIVE, GPU_ESTIMATE_S, Case, Description, Dump, IngestReport, McpServer, PlainSession,
                                RetrievalQuery, RetrievalResult, Served, SessionContext, Usage)
from mmsb.core.runfolder import package_root
from mmsb.systems import read as read_system_toml
from mmsb.systems.mm.audit import HorizonLeak, ambient_labels, work_of
from mmsb.systems.mm.audit import audit as horizon_audit
from mmsb.systems.mm.daemon import Api, Daemon, wait_for, write_config
from mmsb.systems.mm.estimate import ASK_SHARE, HEAVY_S, arm_jobs, beyond_timeout, estimate_s, heavy, table_of
from mmsb.systems.mm.home import PROJECT_PREFIX, SNAPSHOT_CONFIG, copy_home, exec_project, kept_items, scrub_home, set_horizon
from mmsb.systems.mm.jobs import JOB_WAIT_S, quiesce, wait_idle
from mmsb.systems.mm.load import check_snapshot, load
from mmsb.systems.mm.models import MODELS_DIR, Model, known_model, needs_gpu, not_ready
from mmsb.systems.mm.oracle import retrieve, warm, warm_params

HERE = Path(__file__).parent
BASE = "You answer the user's question from their memory, using the mm tools."
SERVER = "mm"  # the MCP server's name in the agents: `mcp__mm__ask`
LOOKS = "looks"  # the folder of a home where `look` writes the pictures and sounds it shows: the agent with its own tools may open them
TOOLS = ("ask", "find", "look", "label")  # the only mm tools the agents are offered: none that reads the disk
CLOCK_NOTE = "mm's clock reads a later date than now: give since and until as dates and times, not as ages or \"now\"."
CLOCK_LATE_S = 3600.0  # mm's clock is the machine's: a case whose `now` is more than this earlier needs the note
INGEST_VERSION = 1  # bumped when ingest stores something other than before
SKILL = resources.files("multimem") / "skills" / "multimem-questions" / "SKILL.md"
NOTES_DIR = resources.files("multimem") / "skills" / "multimem-questions" / "models"


def notes_text(notes: str) -> str:
    """The text of a notes file of the skill: `About.notes`, a plain file name in its models folder."""
    if notes in (".", "..") or "/" in notes or "\\" in notes:
        raise ValueError(f"notes is a plain file name in the skill's models folder, not {notes!r}")
    return (NOTES_DIR / notes).read_text(encoding="utf-8")


@functools.cache
def version_of() -> str:
    """Multimem's short commit, `+dirty` when tracked files differ from it, when it is a checkout (read only: git is told not to touch
    the index); else the version it was installed as."""
    root = package_root("multimem")
    if root is None or not (root / ".git").exists():
        return importlib.metadata.version("multimem")

    def git(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def plain(value: object) -> object:
    """What JSON cannot say of a model's facts (`About`): a set as a sorted list, a mapping as a dict, a dataclass as its fields."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    raise TypeError(f"{value!r} is not part of a hash")


def sha_of(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, default=plain, sort_keys=True).encode("utf-8")).hexdigest()


def serve_record(estimate: float, timeout_s: float, busy_at_start: int | None) -> dict:
    """The fields `serve` adds to a result: the GPU seconds the case may need, whether it ran alone, whether the ask cannot finish inside the
    exec's time (the model's speed, not a miss), and the jobs the daemon still had when a case that holds the GPU started (0: idle)."""
    return {GPU_ESTIMATE_S: round(estimate, 1), EXCLUSIVE: heavy(estimate), BEYOND_TIMEOUT: beyond_timeout(estimate, timeout_s),
            "busy_at_start": busy_at_start}


def size_of(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def measured_of(table: Mapping) -> dict[str, float]:
    """What ingesting measured that a scenario's report quotes (`IngestReport.measured`): the seconds the reader decodes a window of the one
    source whose pictures are large (V8's glasses) as `decode_s`; nothing when no source was measured, or several (which one is meant?)."""
    return {"decode_s": next(iter(table["decode_s"].values()))} if len(table["decode_s"]) == 1 else {}


def parts_of(model: Model) -> tuple[str, ...]:
    """The skill and the model's notes: what the agents are told beside the server's own instructions."""
    return (SKILL.read_text(encoding="utf-8"), *([notes_text(model.about.notes)] if model.about.notes else []))


def jobs_of(model: Model, cases: Sequence[Case], requested: int) -> int:
    """One exec at a time when a case has a horizon (the cut is the whole home's) or the model is slow, else `requested`."""
    return 1 if any(c.horizon is not None for c in cases) else arm_jobs(model, requested)


def served_of(home: Path, parts: tuple[str, ...], case: Case, context: SessionContext, busy_at_start: int | None) -> Served:
    """The server over `home`, its instructions and the fields added to the result of the exec of `case`: all that `serve` gives that
    needs no daemon."""
    args = ("-m", "multimem.cli", "--home", str(home), "mcp", "--tools", ",".join(TOOLS))
    server = McpServer(SERVER, sys.executable, args, {"CLAUDE_PROJECT_DIR": str(case.work)}, TOOLS)
    clock = CLOCK_NOTE if context.wall() - case.now > CLOCK_LATE_S else None
    return Served(server, BASE, parts, INSTRUCTIONS, SERVER, clock, serve_record(case.estimate_s, context.timeout_s, busy_at_start),
                  opens=(home / LOOKS,))


class Mm:
    """Multimem as a system. `models` are variants this instance knows besides the ones in the models folder (a test's fake decider);
    a variant it does not know is looked for in `models_dir`, where it must be ready to run."""

    def __init__(self, models: Mapping[str, Model] | None = None, models_dir: Path = MODELS_DIR) -> None:
        self.models, self.models_dir = dict(models or {}), Path(models_dir)

    def model_of(self, variant: str | None) -> Model:
        """The model of `variant`: its facts, whether or not it can run now (`ready_model`)."""
        if variant is None:
            raise ValueError("mm has variants: name the decision model, as mm/d1-3B")
        return self.models[variant] if variant in self.models else known_model(variant, self.models_dir)

    def ready_model(self, variant: str | None) -> Model:
        """The model of `variant`, which must be able to run now: a model given to this instance is, any other is checked (`not_ready`)."""
        model = self.model_of(variant)
        if variant not in self.models and (why := not_ready(model)):
            raise ValueError(why)
        return model

    def describe(self, variant: str | None) -> Description:
        model = self.model_of(variant)
        kinds = {"text", "records"} | (set(model.about.reads) & {"image", "video", "audio"})  # (the questions about text and records are asked of every model)
        store = sha_of("mm", version_of(), SNAPSHOT_CONFIG, INGEST_VERSION)
        config = sha_of(store, model.name, model.spec, model.url, model.about, sha_of(SKILL.read_text(encoding="utf-8")), sha_of(INSTRUCTIONS),
                        sha_of(notes_text(model.about.notes) if model.about.notes else None), BASE, TOOLS, CLOCK_NOTE, CLOCK_LATE_S, HEAVY_S,
                        ASK_SHARE)
        return Description(name="mm", version=version_of(), variant=variant, reads=frozenset(kinds), local=True, needs_gpu=needs_gpu(model),
                           gpu_memory_gb=None, network=False, license=read_system_toml(HERE / "system.toml").license, config_hash=config,
                           store_hash=store, horizon="cut", has_oracle=True)

    def check_ready(self, variant: str | None) -> None:
        """Raises why the model of `variant` cannot run now (`not_ready`): `open` says the same, but only when its cell starts."""
        self.ready_model(variant)

    def ingest(self, dump: Dump, store: Path) -> IngestReport:
        """The dump remembered in a home of its own, `store/home`, and counted for the estimate in `store/windows.json`. `store` is new."""
        began = time.monotonic()
        home = Path(store) / "home"
        if home.exists():
            raise ValueError(f"{home} exists: an mm store is made once")
        home.mkdir(parents=True)
        write_config(home, SNAPSHOT_CONFIG)
        daemon = Daemon(home).start()
        try:
            refused, misdated = load(daemon.api, dump)
        finally:
            daemon.stop()
        wait_for(lambda: not (home / "daemon.json").exists(), what="the snapshot's daemon to stop")
        check_snapshot(home, dump, {r.path for r in refused})
        table = table_of(dump)
        fs.write_json(Path(store) / "windows.json", table)
        return IngestReport(seconds=round(time.monotonic() - began, 1), gpu_seconds=0.0, bytes_written=size_of(Path(store)),
                            refused=tuple(refused), misdated=tuple(misdated), measured=measured_of(table))

    def preview(self, store: Path, dump: Dump, context: SessionContext) -> MmPreview:
        """What `open` would answer without a daemon: the jobs, the estimate from `store/windows.json` and the server over the home that
        `open` would make. Nothing is copied or started, and the model need not be ready to run (`open` says when it is not)."""
        model = self.model_of(context.variant)
        return MmPreview(model, context.homes / f"{context.agent}-{model.slug}", fs.read_json(Path(store) / "windows.json"), parts_of(model),
                         context)

    def open(self, store: Path, dump: Dump, context: SessionContext) -> MmSession:
        """A copy of the store for (agent, variant), its daemon started, the model loaded by one window of the dump's first picture,
        video or sound it reads, and what an earlier run left of the execs' labels swept out. The daemon is stopped by `MmSession.close`."""
        model = self.ready_model(context.variant)
        snapshot = Path(store) / "home"
        table = fs.read_json(Path(store) / "windows.json")
        home = copy_home(snapshot, context.homes / f"{context.agent}-{model.slug}", model)
        daemon = Daemon(home).start()
        try:
            warm(daemon.api, warm_params(dump, model.about.reads))
            kept = kept_items(snapshot)  # the store's, not the live home's: an interrupted exec's keeps are not the baseline
            scrub_home(home, kept, prefix=PROJECT_PREFIX)  # the warm ask's label, and an interrupted exec's
        except BaseException:
            daemon.stop()
            raise
        versioned = frozenset(e.source for e in dump.entries if e.media == "text")
        return MmSession(model, home, daemon, kept, table, versioned, parts_of(model), context)


class NoQueue(PlainSession):
    """What both of mm's sessions share: Multimem keeps no per-job wait, so they cannot say how long a case queued for the GPU (decision
    D3, a change in Multimem). The working clock then runs on the wall, and more than one exec at once on it is refused (core/clock.py)."""

    def queue_s(self, case: Case) -> float | None:
        return None


class MmPreview(NoQueue):
    """The session of a dry run: what `MmSession` answers from the model's facts and the store's table, with no home, daemon or marker."""

    def __init__(self, model: Model, home: Path, table: dict, parts: tuple[str, ...], context: SessionContext) -> None:
        self.model, self.home, self.table, self.parts, self.context = model, home, table, parts, context

    def max_jobs(self, cases: Sequence[Case], requested: int) -> int:
        return jobs_of(self.model, cases, requested)

    def estimate(self, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
        return estimate_s(self.table, self.model.about, sources, ranges)

    def serve(self, case: Case) -> Served:
        return served_of(self.home, self.parts, case, self.context, None)


class MmSession(NoQueue):
    """One (agent, variant) over a home and its daemon. `serve`, `audit` and `reset` run on the runner's worker threads at once: the set
    of the projects of the execs that ended is the only state they share, and has a lock."""

    def __init__(self, model: Model, home: Path, daemon: Daemon, kept: set[int], table: dict, versioned: frozenset[str],
                 parts: tuple[str, ...], context: SessionContext) -> None:
        self.model, self.home, self.daemon, self.kept, self.table = model, home, daemon, kept, table
        self.versioned, self.parts, self.context = versioned, parts, context
        self.ended: set[str] = set()  # the projects of the execs that ended
        self.lock = threading.Lock()

    def versioned_case(self, case: Case) -> bool:
        return any(s in self.versioned for s in case.sources)

    def max_jobs(self, cases: Sequence[Case], requested: int) -> int:
        return jobs_of(self.model, cases, requested)

    def estimate(self, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
        return estimate_s(self.table, self.model.about, sources, ranges)

    def admits(self, case: Case, busy: Sequence[Case]) -> bool:
        """No case starts beside one that holds the GPU; one that may need over HEAVY_S seconds starts alone; and the cases of the
        sources of file versions run one at a time (in the core's order, which is ascending `until`), so a later case's ask has
        labelled no earlier note or version before that one's own case ran."""
        if any(heavy(b.estimate_s) for b in busy) or (heavy(case.estimate_s) and busy):
            return False
        return not (self.versioned_case(case) and any(self.versioned_case(b) for b in busy))

    def serve(self, case: Case) -> Served:
        busy_at_start = wait_idle(Api(self.home), 0, JOB_WAIT_S) if heavy(case.estimate_s) else None  # a case that holds the GPU starts idle
        (case.work / MARKER).write_text(exec_project(case.qid).removeprefix("id:"), encoding="utf-8")
        (self.home / LOOKS).mkdir(parents=True, exist_ok=True)  # (the daemon makes it at its first look; the agent is handed it now)
        with self.lock:
            gone = sorted(self.ended)
        scrub_home(self.home, self.kept, projects=gone)  # a job of an exec that ended may have labelled since
        if case.horizon is not None:
            set_horizon(self.home, case.horizon)
        return served_of(self.home, self.parts, case, self.context, busy_at_start)

    def audit(self, case: Case, calls: Sequence[dict]) -> dict:
        """The ask jobs of the exec (`work`); for a case with a horizon also the recordings past it in the exec's results, which are a
        `HorizonLeak`: the home was not cut to it, and the question stays to run again."""
        out: dict = {"work": work_of(calls)}
        if case.horizon is not None:
            found = horizon_audit(calls, case.horizon, self.home)
            if found["after_horizon"]:
                raise HorizonLeak(f"{case.qid}: {found['after_horizon']} recording times after the horizon in {self.context.agent}'s "
                                  "tool results: the index was not cut to it")
            out.update(found)
        return out

    def reset(self, case: Case, calls: Sequence[dict], others: int) -> dict:
        """What an mm exec leaves running when it ends, dealt with before its labels are scrubbed and its project released to the next exec:
        its jobs cancelled and waited for, and the daemon waited down to the `others` execs still running (see `quiesce`); then its
        state taken out of the home. Returns what was left, and what the wait found."""
        quiet = quiesce(Api(self.home), calls, others)
        left = scrub_home(self.home, self.kept, projects=[exec_project(case.qid)])
        with self.lock:
            self.ended.add(exec_project(case.qid))
        return {"left": left, **quiet}

    def usage(self, case: Case, calls: Sequence[dict]) -> Usage:
        return Usage(gpu_s=work_of(calls)["seconds"], queue_s=None, api_usd=0.0)

    def oracle(self, query: RetrievalQuery) -> RetrievalResult:
        return retrieve(Api(self.home), query)

    def close(self) -> dict[str, list]:
        """The labels that no project owns listed, and the daemon stopped."""
        try:
            labels = ambient_labels(Api(self.home), *self.table["span"])
        finally:
            self.daemon.stop()
        return {"ambient-labels": labels}
