"""What the tests of the mm system share (tests/mm): the fake decider and its model, the system with that model as a variant, the tiny V7 week
built once for a session of tests and ingested once, a session opened over a copy of that store, the run folder `run_agents` needs, and a
stand-in agent. Nothing here starts a real agent or loads a real model; every daemon is over a folder of the test's own and has the fake
decider (`python:` models need no GPU).

Moved from tests/conftest.py of MultimediaSearch at fe0ece5: `FAKE`, `DECIDER`, `span`. Written for this repository: the rest."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import imageio_ffmpeg

from agent_spec import clocked
from mms import config as configuration
from mms import decider, decider_d1

from mmsb import scenarios
from mmsb.core import fs
from mmsb.core.agents.spec import SessionSpec
from mmsb.core.config import Config
from mmsb.core.contract import Dump, IngestReport, SessionContext
from mmsb.core.question import Question, write_questions
from mmsb.core.runfolder import RunFolder
from mmsb.systems.mm.models import Model
from mmsb.systems.mm.system import Mm, MmSession

ROOT = Path(__file__).resolve().parent
FAKE = f"python:{ROOT / 'fake_decider.py'}"
DECIDER = f'[decider]\nmodel = "{FAKE.replace(chr(92), "/")}"\n'
FAKE_MODEL = Model("fake", FAKE, decider_d1.ABOUT)  # the fake has d1-Omni-2B's facts: it is asked every question
D13_MODEL = Model("d1-3B", FAKE, decider.known("d1-3B")[1])  # d1-3B's facts and notes, the fake's answers: what the recorded pilot's skill and notes were for
MM = Mm(models={"fake": FAKE_MODEL, "d1-3B": D13_MODEL})
S = Config("mm/fake", "mm", "fake", False)  # the system's tools alone
S_FILES = Config("mm/fake+files", "mm", "fake", True)  # and with the agent's own tools
FILES = Config("files", "files", None, True)
DAYS = ["2026-10-30", "2026-10-31"]  # the tiny week of the old tests: MultimediaSearch's tests/test_haystack.py
SEED = 7
V7 = scenarios.get("v7_screen")


def span(found: dict) -> tuple[float, float]:
    """A sighting's (or a candidate's) start and end as epoch seconds: `t` is both, else `t0` and `t1`."""
    return (configuration.parse_when(found.get("t") or found["t0"]), configuration.parse_when(found.get("t") or found["t1"]))


def context(homes: Path, agent: str = "claude", variant: str | None = "fake", timeout_s: float = 900.0, wall=time.time) -> SessionContext:
    return SessionContext(agent=agent, variant=variant, timeout_s=timeout_s, homes=homes, wall=wall)


def tiny_week(root: Path, workers: int = 3) -> tuple[dict, Dump, tuple[Question, ...]]:
    """The tiny week of seed 7 built into `root / "data"` (about 35 s), with its recipe, dump and questions."""
    recipe = V7.recipe(scale="tiny", seed=SEED, days=DAYS)
    dump, questions = V7.build(recipe, root / "data", workers)
    return recipe, dump, questions


REPORTS: dict[str, IngestReport] = {}  # the report of each store `ingested` made, by its path


def ingested(dump: Dump, store: Path, system: Mm = MM) -> Path:
    """`dump` ingested into `store` by `system`, whose report must have nothing refused or misdated."""
    report = REPORTS[str(store)] = system.ingest(dump, store)
    assert report.refused == () and report.misdated == (), report
    return store


def home_of(spec: SessionSpec) -> Path:
    """The mms home a spec's server is over (its `--home` argument): where an agent of the test looks for the daemon."""
    args = spec.served.server.args
    return Path(args[args.index("--home") + 1])


class Opened:
    """A session opened over a copy of `store`, closed at the end of the `with`. `homes` is where the copy goes."""

    def __init__(self, store: Path, dump: Dump, homes: Path, system: Mm = MM, **kw) -> None:
        self.session = system.open(store, dump, context(homes, **kw))

    def __enter__(self) -> MmSession:
        return self.session

    def __exit__(self, *exc) -> None:
        self.session.close()


def run_of(root: Path, recipe: dict, questions: Sequence[Question], dump: Dump, configs: Sequence[Config], tag: str = "t7",
           timeout_s: float = 900.0) -> RunFolder:
    """A run folder in `root` for the questions of the tiny week: the question file, the manifest of the dump (the data must not change),
    and a folder for the temp files of the run."""
    folder, temp = root / "run", root / "temp"
    folder.mkdir(parents=True, exist_ok=True)
    write_questions(folder / "questions.jsonl", questions)
    fs.write_json(folder / "data-manifest.json", fs.stats_of(dump.root))
    return RunFolder(folder, SEED, "tiny", tuple(date.fromisoformat(d) for d in recipe["days"]), dump.root, temp, tag,
                     timeout_s, V7, tuple(configs))


def made_up_ffmpeg(patch, root: Path) -> None:
    """`ensure_bin` copies a made-up file, not 80 MB of ffmpeg."""
    ffmpeg = root / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    patch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))


class Probe:
    """A stand-in agent that records what each exec was given and answers right for the questions in `RIGHT`, unknown for the rest. An exec of
    a system with tools (`spec.served.server`) reads its home: `during(spec, question)` may call the daemon, and what it returns is kept in
    `seen[qid]` and given to `tools(spec, question, found)`, which makes the calls the exec reports. `log` holds (config dir, source, qid, start, end), `execs` the specs."""

    def __init__(self, questions: Sequence[Question], *, pause: float = 0.0, during=None, tools=None, answer=None) -> None:
        self.questions = {q.qid: q for q in questions}
        self.pause, self.during, self.tools, self.answer = pause, during, tools, answer
        self.execs: list[SessionSpec] = []
        self.marks: dict[tuple[str, str], str | None] = {}
        self.seen: dict[str, object] = {}
        self.log: list[tuple] = []
        self.lock = threading.Lock()

    def __call__(self, spec: SessionSpec) -> dict:
        qid, folder = spec.run_dir.name, spec.run_dir.parent.name
        q, began = self.questions[qid], time.monotonic()
        marker = spec.work / ".mms-project"
        with self.lock:
            self.execs.append(spec)
            self.marks[folder, qid] = marker.read_text(encoding="utf-8") if marker.exists() else None
        found = self.during(spec, q) if self.during and spec.served.server is not None else None
        time.sleep(self.pause)
        answer = self.answer(spec, q) if self.answer else "unknown"
        tools = self.tools(spec, q, found) if self.tools else []
        with self.lock:
            self.seen[qid] = found
            self.log.append((folder, q.scope.sources[0], qid, began, time.monotonic()))
        return {"final": f"ANSWER: {answer}", "tools": tools, **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}
