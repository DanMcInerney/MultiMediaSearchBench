"""A stand-in system, session and agent for the tests of the runner (plan 7.4: neutral loop tests use a `StubSystem` built from recorded
fields), and the folders of a run they run in. Nothing here starts a process or imports multimem.

What the stub system copies from the recorded runs is the shape of what a real session adds to a result (the fields `gpu_estimate_s`,
`exclusive`, `beyond_timeout`, `busy_at_start`, `left`, `cancelled`, `unended`, `orphans`, `settle_s` of the pilot's results), and from the
plan's 4.6 the two rules it holds a case back by: file versions (the sources `notes` and `config`) one at a time, and a case that may
need more than `HEAVY_S` of the system's seconds alone. Those rules are written here from the plan's words, not from any module."""

from __future__ import annotations

import functools
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from agent_spec import clocked

from mmmb.core import fs
from mmmb.core.agents.spec import SessionSpec
from mmmb.core.config import Config
from mmmb.core.contract import (Case, Description, Dump, IngestReport, McpServer, PlainSession, Served, SessionContext, Usage)
from mmmb.core.question import Question, write_questions
from mmmb.core.runfolder import RunFolder
from mmmb.scenarios.v7_screen.manifest import entries_of
from mmmb.scenarios.v7_screen.questions import questions_of
from mmmb.scenarios.v7_screen.week import plan as make_plan

MODALITIES = frozenset({"image", "video", "audio", "text", "records"})
READS = {"all": MODALITIES, "narrow": frozenset({"image", "video", "text", "records"}), "media": frozenset({"image", "video"})}
VERSIONED = ("notes", "config")  # the sources whose files are versions: one question at a time, in time order
HEAVY_S = 300.0  # a case that may need more than this alone holds the system
SERVER = McpServer("stub", sys.executable, ("-m", "stub", "serve"), {}, ("lookup", "fetch"))
DAYS = [date(2026, 10, 30), date(2026, 10, 31)]
STUB = Config("stub/all", "stub", "all", False)
FILES = Config("files", "files", None, True)


@functools.cache
def tiny_questions() -> tuple[Question, ...]:
    """The 19 questions of the tiny V7 week of seed 7 (the plan only: no file is written), as the question file has them."""
    plan = make_plan(7, "tiny", DAYS)
    return questions_of(plan, entries_of(plan, lambda path: (1, 0, "0" * 64)))


def right(q: Question) -> str:
    """The answer that is right for a question (an absent one is `unknown`; a day's question names its date)."""
    g = q.grade
    if g.rule == "absent":
        return "unknown"
    if g.rule in ("number", "count"):
        return g.truth
    answer = g.time[:5] if g.rule == "time" else f"{g.truth} at {g.time[:5]}"
    return f"{answer} on {g.date}" if g.date else answer


class StubSession(PlainSession):
    """A session that serves a server of no use, records what the runner calls in `system.events` (kind, qid, what else) and adds to a
    result what the pilot's results from a real session held. `rate` turns a range's seconds into the system's seconds."""

    def __init__(self, system: StubSystem, context: SessionContext) -> None:
        self.system, self.context = system, context

    def max_jobs(self, cases, requested):
        self.system.note("max_jobs", None, len(cases), requested)
        return min(requested, self.system.job_limit or requested)

    def estimate(self, sources, ranges):
        return self.system.rate * sum(b - a for a, b in ranges)

    def admits(self, case: Case, busy) -> bool:
        if not self.system.rules:
            return True
        heavy = lambda c: c.estimate_s >= HEAVY_S  # noqa: E731
        if any(heavy(b) for b in busy) or (heavy(case) and busy):
            return False
        return not (any(b.sources[0] in VERSIONED for b in busy) and case.sources[0] in VERSIONED)

    def serve(self, case: Case) -> Served:
        self.system.note("serve", case.qid, case.work, case.slot)
        heavy = case.estimate_s >= HEAVY_S
        record = {"gpu_estimate_s": round(case.estimate_s, 1), "exclusive": heavy,
                  "beyond_timeout": case.estimate_s > 0.75 * self.context.timeout_s, "busy_at_start": 0 if heavy else None}
        return Served(SERVER, "You answer from the stub.", ("THE STUB'S NOTES",), None, "stub", self.system.clock_note, record)

    def audit(self, case: Case, calls) -> dict:
        self.system.note("audit", case.qid, len(calls))
        if self.system.leak:
            raise RuntimeError(f"{case.qid}: {self.system.leak}")
        return {"audit": f"{case.qid}:{len(calls)}", "work": {"jobs": len(calls), "windows": 10 * len(calls)}}

    def reset(self, case: Case, calls, others: int) -> dict:
        self.system.note("reset", case.qid, others, len(calls))
        return {"left": {"labels": len(calls), "unkept": 0}, "cancelled": [], "unended": [], "orphans": 0, "settle_s": 0.0}

    def usage(self, case: Case, calls) -> Usage:
        return Usage(gpu_s=case.estimate_s, queue_s=None, api_usd=0.0)

    def close(self) -> dict[str, list]:
        self.system.note("close", None, self.context.agent)
        if self.system.close_raises:
            raise RuntimeError(self.system.close_raises)
        return self.system.closing


@dataclass
class StubSystem:
    """A system named `stub` with the variants `all` (reads every modality), `narrow` (no sound) and `media` (pictures and video).
    `rules`: its sessions hold cases back as the plan's 4.6 says. `rate`: the system's seconds for a second of a range."""

    needs_gpu: bool = False
    rules: bool = True
    rate: float = 0.0
    job_limit: int | None = None
    clock_note: str | None = None
    leak: str | None = None  # an audit that raises, as a leak into the index does
    close_raises: str | None = None  # a close that raises, as a daemon that is gone does
    config_hash: str = "c"  # what `describe` says of the configuration and of the store: a test changes them to make another version of the system
    store_hash: str = "s"
    closing: dict = None
    shared: list | None = None  # a log that other stand-ins write to as well, to tell the order of the calls among them
    report: IngestReport | None = None  # what `ingest` reports (the runner never ingests; the CLI's build does); None: ingest is an error
    unready: str | None = None  # why `check_ready` raises (a model not downloaded, an environment not installed); None: it is ready

    def __post_init__(self) -> None:
        self.events: list[tuple] = []
        self.lock = threading.Lock()
        self.closing = self.closing or {}

    def note(self, kind: str, qid: str | None, *rest) -> None:
        with self.lock:
            self.events.append((kind, qid, *rest, time.monotonic()))
            if self.shared is not None:
                self.shared.append((kind, qid))

    def kinds(self, kind: str) -> list[tuple]:
        return [e for e in self.events if e[0] == kind]

    def describe(self, variant: str | None) -> Description:
        return Description(name="stub", version="1", variant=variant, reads=READS[variant], local=True, needs_gpu=self.needs_gpu,
                           gpu_memory_gb=None, network=False, license="none", config_hash=self.config_hash, store_hash=self.store_hash,
                           horizon="cut", has_oracle=False)

    def check_ready(self, variant: str | None) -> None:
        self.note("check_ready", None, variant)
        if self.unready:
            raise ValueError(f"{variant}: {self.unready}")

    def ingest(self, dump: Dump, store: Path) -> IngestReport:
        if self.report is None:
            raise AssertionError("the runner never ingests")
        self.note("ingest", None, dump.key, store)
        return self.report

    def open(self, store: Path, dump: Dump, context: SessionContext) -> StubSession:
        self.note("open", None, context.agent, context.variant, store)
        return StubSession(self, context)

    def preview(self, store: Path, dump: Dump, context: SessionContext) -> StubSession:
        self.note("preview", None, context.agent, context.variant, store)
        return StubSession(self, context)


class Stub:
    """An agent that answers right for text and records in both kinds of configuration and for every modality with a system's tools,
    `unknown` for pictures, video and sound with the agent's own tools only. `first` changes the fields of its first result,
    `script[n]` those of its n-th (None: none), and `by[(config dir, qid)]` those of that exec's; `pause` is how long each exec takes.
    `execs` holds the specs it was given, `log` (config dir, source, qid, start, end)."""

    def __init__(self, first: dict | None = None, *, script=(), by=None, pause: float = 0.0, questions=None, tools=None):
        self.questions = {q.qid: q for q in (questions or tiny_questions())}
        self.execs: list[SessionSpec] = []
        self.first, self.script, self.by, self.pause, self.tools = first or {}, list(script), by or {}, pause, tools
        self.log: list[tuple] = []
        self.lock = threading.Lock()

    @staticmethod
    def qid_of(spec: SessionSpec) -> str:
        return spec.run_dir.name

    @staticmethod
    def dir_of(spec: SessionSpec) -> str:
        return spec.run_dir.parent.name

    def __call__(self, spec: SessionSpec) -> dict:
        q, folder = self.questions[self.qid_of(spec)], self.dir_of(spec)
        began = time.monotonic()
        with self.lock:
            self.execs.append(spec)
            n = len(self.execs) - 1
        time.sleep(self.pause)
        own_only = spec.own_tools and spec.server is None
        answer = "unknown" if own_only and q.tags["modality"] in ("image", "video", "audio") else right(q)
        tools = self.tools if self.tools is not None else (
            [{"tool": "shell", "arguments": {"command": "ls"}}] if spec.own_tools else
            [{"tool": "lookup", "arguments": {"question": q.question}}, {"tool": "fetch", "arguments": {"scope": "all"}}])
        out = {"final": f"ANSWER: {answer}", "tools": tools, **clocked(spec, 1.5), "problem": None, "turns": 2,
               "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1,
               "used_percent": 10.0, "model": "stub", "denials": 0}
        out.update(self.first if n == 0 else {})
        out.update((self.script[n] if n < len(self.script) else None) or {})
        out.update(self.by.get((folder, q.qid), {}))
        with self.lock:
            self.log.append((folder, q.scope.sources[0], q.qid, began, time.monotonic()))
        return out

    def qids(self, folder: str | None = None) -> list[str]:
        return [self.qid_of(s) for s in self.execs if folder is None or self.dir_of(s) == folder]


def run_of(root: Path, configs=(FILES, STUB), scenario=None, questions=None, tag: str = "t1") -> tuple[RunFolder, Dump]:
    """A run folder in `root` for the tiny week's questions: a few made-up files as the frozen data, its manifest, the question file, and
    the dump that says where the data is. The ffmpeg of `ensure_bin` is a made-up file too (the caller patches it)."""
    from mmmb import scenarios

    questions = tuple(questions or tiny_questions())
    folder, temp = root / "run", root / "temp"
    data = temp / f"mmb-{tag}-data"
    for rel, text in (("notes/a.md", "a note"), ("screen/s.mkv", "a screen"), ("logs/web.jsonl", "{}")):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text(text, encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    fs.write_json(folder / "data-manifest.json", fs.stats_of(data))
    write_questions(folder / "questions.jsonl", questions)
    run = RunFolder(folder, 7, "tiny", tuple(DAYS), data, temp, tag, 900.0, scenario or scenarios.get("v7_screen"), tuple(configs))
    return run, Dump("k" * 64, data, "v7_screen", (), ())
