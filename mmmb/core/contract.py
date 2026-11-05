"""What a system, a scenario and the runner say to each other (DESIGN 3.1, corrected: a daemon and a model load span many cases, so a
`System` builds and clones stores and `open` returns a `Session` that lives across cases). Types only, plus `PlainSession`, the
session of a system with nothing to run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class Description:
    name: str
    version: str
    variant: str | None
    reads: frozenset[str]  # of image, video, audio, text, records
    local: bool
    needs_gpu: bool
    gpu_memory_gb: float | None
    network: bool
    license: str
    config_hash: str
    store_hash: str  # only what ingest depends on, so variants can share a store
    horizon: Literal["cut", "view"]  # "view": the scenario hands it a folder cut to the horizon
    has_oracle: bool
    leak_filter: bool = False  # a floor, not a contestant: a question its tools alone answer is flagged by the report and kept out of the paired counts


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    source: str
    t: float
    kind: str
    media: str  # image, video, audio, text, records
    name: str | None
    own_time: bool  # the file's metadata carries its time (a camera roll)
    size: int
    mtime_ns: int
    sha256: str
    length: float | None = None  # seconds the file plays or covers, when its scenario needs the end of a recording (V8's clips); else None


@dataclass(frozen=True)
class LogSource:
    name: str
    folder: str
    glob: str
    time_field: str


@dataclass(frozen=True)
class Dump:
    key: str
    root: Path
    scenario: str
    entries: tuple[ManifestEntry, ...]
    logs: tuple[LogSource, ...]


@dataclass(frozen=True)
class Refusal:
    path: str
    reason: str


@dataclass(frozen=True)
class IngestReport:
    seconds: float
    gpu_seconds: float
    bytes_written: int
    refused: tuple[Refusal, ...]
    misdated: tuple[Refusal, ...]  # a file stored at a time other than its own: every question about it would be asked of a wrong index
    measured: Mapping[str, float] = field(default_factory=dict)  # figures the system measured while ingesting, that a scenario's report quotes


@dataclass(frozen=True)
class Case:
    """One question as a system sees it. It never holds the question's text or truth."""

    qid: str
    now: float
    horizon: float | None
    sources: tuple[str, ...]
    since: float
    until: float
    group: str
    estimate_s: float
    slot: int
    work: Path
    run_dir: Path


@dataclass(frozen=True)
class McpServer:
    name: str
    command: str
    args: tuple[str, ...]
    env: Mapping[str, str]
    tools: tuple[str, ...]


# The keys of `Served.record` that the core acts on, for any system. A system sets the ones that mean something for it; a case whose record
# lacks a key is read as a case without it (not exclusive, no GPU estimate, not beyond its timeout).
EXCLUSIVE = "exclusive"  # the system serves the case alone: `clock-check` does not draw it (it queues for nothing), and the report counts it
GPU_ESTIMATE_S = "gpu_estimate_s"  # the seconds of its own the system thinks the case needs: the report's mean, and the dry run's list of those beyond the timeout
BEYOND_TIMEOUT = "beyond_timeout"  # the case's work cannot finish inside the exec's time, so a timeout there is the system's speed, not a miss: the dry run and report say so


@dataclass(frozen=True)
class Served:
    server: McpServer | None
    base_line: str | None  # "You answer the user's question from their memory, using the mm tools."
    parts: tuple[str, ...]  # skill, model notes
    server_instructions: str | None  # Codex only, as recorded
    holder: str | None  # "mm": the word the prompt's sentences use
    clock_note: str | None  # the sentence about a clock later than `now`
    record: Mapping[str, object]  # fields merged into the result; those the core acts on are EXCLUSIVE, GPU_ESTIMATE_S and BEYOND_TIMEOUT, the rest are the system's own
    opens: tuple[Path, ...] = ()  # folders the system hands the agent and its own tools may open though the run's folder is denied (mm's `looks`)


@dataclass(frozen=True)
class Usage:
    gpu_s: float | None
    queue_s: float | None  # None: the system cannot say
    api_usd: float


@dataclass(frozen=True)
class RetrievalQuery:
    qid: str
    text: str
    sources: tuple[str, ...]
    ranges: tuple[tuple[float, float], ...]
    k: int
    candidates: int
    oldest_first: bool
    max_s: float


@dataclass(frozen=True)
class Moment:
    t0: float
    t1: float
    p: float
    source: str
    raw: Mapping[str, object]


@dataclass(frozen=True)
class RetrievalResult:
    sightings: tuple[Moment, ...]
    candidates: tuple[Moment, ...]
    windows: int
    seconds: float
    wall_s: float
    stopped: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Isolation:
    """What the systems' system.toml files declare, for every agent shell that has its own tools, whichever system is being run."""

    env_drop_prefixes: tuple[str, ...]  # environment variables with these prefixes are dropped
    deny_words: tuple[str, ...]  # words no tool input may hold
    deny_package_roots: tuple[str, ...]  # packages whose repository no tool input may reach, when they are installed
    env_nowhere: Mapping[str, str]  # variable -> a folder in the working folder that is never made


@dataclass(frozen=True)
class SessionContext:
    agent: str
    variant: str | None
    timeout_s: float
    homes: Path  # where a per-agent clone of a store goes
    wall: Callable[[], float]  # time.time in production; a test injects it


class System(Protocol):
    def describe(self, variant: str | None) -> Description: ...
    def ingest(self, dump: Dump, store: Path) -> IngestReport: ...  # sees the dump only, never questions
    def open(self, store: Path, dump: Dump, context: SessionContext) -> Session: ...

    def check_ready(self, variant: str | None) -> None:
        """Raises a ValueError saying why `open` would fail for `variant` on this machine (weights not downloaded, an environment not
        installed), nothing when it would not. `describe` and `preview` need no readiness, so a run asks this of every configuration
        before its first agent starts: a run of two configurations must not spend its first cell and then fail at the second.
        Starts nothing."""
        ...

    def preview(self, store: Path, dump: Dump, context: SessionContext) -> Session:
        """The session of a dry run (`--dry-run`, and the estimates printed before a run): it starts no process, loads no model and
        changes no file, and answers `max_jobs`, `estimate`, `serve` (the server and instructions an exec would be given, and
        the `record` it would add to its result) and `queue_s` (whether the system can say, which the working clock needs before a run:
        core/clock.py) as `open`'s would, from the store's files. Nothing else of it is called."""
        ...


class Session(Protocol):
    def max_jobs(self, cases: Sequence[Case], requested: int) -> int: ...
    def estimate(self, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float: ...  # pure cost function
    def admits(self, case: Case, busy: Sequence[Case]) -> bool: ...
    def serve(self, case: Case) -> Served: ...
    def audit(self, case: Case, calls: Sequence[dict]) -> dict: ...  # fields for the result; may raise to end the command
    def reset(self, case: Case, calls: Sequence[dict], others: int) -> dict: ...
    def usage(self, case: Case, calls: Sequence[dict]) -> Usage: ...  # per case, not "since the last call": cases overlap

    def queue_s(self, case: Case) -> float | None:
        """The seconds `case`'s exec has waited in the system's queue so far, read live by the working clock during the exec (0.0 before
        it has queued); None, for every case, when the system cannot say. A `preview` session answers it too."""
        ...

    def oracle(self, query: RetrievalQuery) -> RetrievalResult: ...
    def close(self) -> dict[str, list]: ...  # named lists, each written as results/<agent>/<config>/<name>.json when not empty


class PlainSession:
    """A session with nothing to do: no server, no cost, no queue. `files` and `blind` use it."""

    def max_jobs(self, cases: Sequence[Case], requested: int) -> int:
        return requested

    def estimate(self, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
        return 0.0

    def admits(self, case: Case, busy: Sequence[Case]) -> bool:
        return True

    def serve(self, case: Case) -> Served:
        return Served(server=None, base_line=None, parts=(), server_instructions=None, holder=None, clock_note=None, record={})

    def audit(self, case: Case, calls: Sequence[dict]) -> dict:
        return {}

    def reset(self, case: Case, calls: Sequence[dict], others: int) -> dict:
        return {}

    def usage(self, case: Case, calls: Sequence[dict]) -> Usage:
        return Usage(gpu_s=0.0, queue_s=0.0, api_usd=0.0)

    def queue_s(self, case: Case) -> float | None:
        return 0.0

    def oracle(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError(f"{type(self).__name__} has no oracle (Description.has_oracle is False)")

    def close(self) -> dict[str, list]:
        return {}
