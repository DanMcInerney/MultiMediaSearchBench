"""The replay of recorded execs through the real run loop (plan 7.5, oracle 5): a stand-in process, launched as the agent's command, prints a
recorded `events.jsonl` line by line (and, for a timeout or a planted fault, then blocks until it is killed), so that the real
`claude.ask_claude` or `codex.ask_codex` runs its watcher, reader and classifier over what the real agent once printed, inside the real
`run_agents`. The session is a stand-in with the recorded `Served` text and no daemon.

Expected values are the recorded results: nothing is computed from the code under test. The fields compared are the neutral ones of
`SAME` and the neutral half of `work`; and, where the mm extra is installed, the rest of an mm cell: the whole `work` (what the system's audit adds:
`jobs`, `windows`, `seconds`, `stopped`, from systems/mm/audit.py, in `mm_audit`) and the fields its `serve` adds (`gpu_estimate_s`, `exclusive`,
`beyond_timeout`, from the pilot's table of windows and the system's estimate, in `mm_estimate` and `mm_record`). The fields that come of the daemon
(`left`, `cancelled`, `busy_at_start`...) are not replayed: there is none. Without the extra the mm cells are compared on the neutral fields only."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import imageio_ffmpeg
from agent_spec import server_of

from mmsb import scenarios
from mmsb.core import fs, paths
from mmsb.core.agents import claude, codex
from mmsb.core.contract import Case, Description, Dump, PlainSession, Served
from mmsb.core.question import Question, write_questions
from mmsb.core.runfolder import RunFolder
from mmsb.scenarios.v7_screen.manifest import entries_of
from mmsb.scenarios.v7_screen.questions import questions_of
from mmsb.scenarios.v7_screen.week import plan as make_plan

RECORDED = Path(__file__).parent / "oracles" / "recorded"
PILOT = RECORDED / "pilot-s1"
RECORDED_RUN = paths.EARLIER[0] / "20261104-0314-pilot-s1"  # read only
GPU_LOST = RECORDED_RUN / "set-aside-gpu-lost"
DAYS = [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]
CELLS = [("claude", "files"), ("claude", "mm-d1-3B"), ("codex", "files"), ("codex", "mm-d1-3B")]
# what a replay is compared on: every field of a result that is not the time it took, when it was made, where its folder is, how the
# run was configured (`arm` and `model` became `config`, `system`, `variant`) or the session's (`held_s` is a wait, not a record)
SAME = ("qid", "type", "template", "level", "last", "absent", "seed", "agent", "answer", "passed", "abstained", "problem", "turns",
        "denials", "calls", "tools", "scope_all", "tokens", "cost_usd", "used_percent", "agent_model", "agent_effort")
NEUTRAL_WORK = ("views", "shells")
MM_FIELDS = ("gpu_estimate_s", "exclusive", "beyond_timeout")  # what the system's `serve` adds to a result, that a replay can compute
PILOT_WINDOWS = Path(__file__).parent / "mm" / "tables" / "pilot-windows.json"  # the pilot's table of windows (tests/mm/capture_windows.py)
RECORDED_TIMEOUT_S = 900.0  # the exec time of the recorded run: `beyond_timeout` is a fact of the cell, not of the replay's shorter limit
OPTIONAL = ("agent_effort",)  # a result made before the field existed has none, which is None
# stdin is not read (the agent writes the prompt and closes it); a hold keeps the process alive after its last line, as a timeout
STAND_IN = """\
import sys, time
for line in open(sys.argv[1], encoding="utf-8"):
    sys.stdout.write(line)
    sys.stdout.flush()
time.sleep(float(sys.argv[2]))
"""


@dataclass(frozen=True)
class Source:
    """One recorded exec: its stream and its result."""

    agent: str
    directory: str  # the results folder: `files` or `mm-d1-3B`
    qid: str
    events: Path
    result: Path
    tools: Path | None  # the exec's tools.json, when the folder is the recorded run's

    @property
    def recorded(self) -> dict:
        return json.loads(self.result.read_text(encoding="utf-8"))

    @property
    def klass(self) -> str:
        problem = self.recorded["problem"]
        return "normal" if problem is None else problem.split(":", 1)[0]


def checked_in() -> list[Source]:
    """The eight small normal execs in tests/oracles/recorded/pilot-s1/replay."""
    return [Source(f.parts[-4], f.parts[-3], f.parts[-2], f.parent / "events.jsonl", f, None) for f in sorted((PILOT / "replay").rglob("result.json"))]


def recorded_run() -> list[Source]:
    """All 110 execs of the recorded pilot, read in place from the earlier runs folder (`paths.EARLIER`); [] when it is not there."""
    out = []
    for agent, directory in CELLS:
        folder = RECORDED_RUN / "results" / agent / directory
        for f in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            exec_dir = RECORDED_RUN / "execs" / agent / directory / f.stem
            out.append(Source(agent, directory, f.stem, exec_dir / "events.jsonl", f, exec_dir / "tools.json"))
    return out


def gpu_lost_run() -> list[Source]:
    """The 20 Codex execs that ended on a lost GPU (set-aside-gpu-lost), which were recorded as answered; [] when not there."""
    folder = GPU_LOST / "results" / "codex" / "mm-d1-3B"
    return [Source("codex", "mm-d1-3B", f.stem, GPU_LOST / "execs" / "codex" / "mm-d1-3B" / f.stem / "events.jsonl", f,
                   GPU_LOST / "execs" / "codex" / "mm-d1-3B" / f.stem / "tools.json") for f in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def pilot_questions() -> tuple[Question, ...]:
    """The pilot's 31 questions (the plan only: the recorded files' sizes do not matter to a replay)."""
    plan = make_plan(1, "pilot", DAYS)
    return questions_of(plan, entries_of(plan, lambda path: (1, 0, "0" * 64)))


def mm_installed() -> bool:
    return importlib.util.find_spec("mms") is not None


def mm_audit(case: Case, calls: Sequence[dict]) -> dict:
    """What the mm system's audit (systems/mm/audit.py) adds to a result of an exec: `{"work": {"jobs", "windows", "seconds", "stopped"}}`, the
    ask jobs of its calls. Without the mm extra the replay of an mm cell compares the neutral fields only, and this adds nothing."""
    if not mm_installed():
        return {}
    from mmsb.systems.mm.audit import work_of

    return {"work": work_of(calls)}


def mm_estimate(sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
    """The mm system's estimate of the GPU seconds of a case for d1-3B, from the pilot's table of windows (0.0 without the mm extra)."""
    if not mm_installed():
        return 0.0
    from mms import decider

    from mmsb.systems.mm.estimate import estimate_s

    return estimate_s(json.loads(PILOT_WINDOWS.read_text(encoding="utf-8")), decider.known("d1-3B")[1], sources, ranges)


def mm_record(case: Case) -> dict:
    """What the mm system's `serve` adds to a result that a replay can compute (MM_FIELDS), for the case's estimate."""
    if not mm_installed():
        return {}
    from mmsb.systems.mm.system import serve_record

    return {k: v for k, v in serve_record(case.estimate_s, RECORDED_TIMEOUT_S, None).items() if k in MM_FIELDS}


class ReplaySession(PlainSession):
    """A session that serves the recorded text of its agent and runs nothing; the audit, the estimate and the record of `serve` are mm's, where the
    extra is installed."""

    def __init__(self, served: Served, home: Path, audit: Callable[[Case, Sequence[dict]], dict]) -> None:
        self.served, self.home, self.audit_with = served, home, audit

    def estimate(self, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
        return mm_estimate(sources, ranges)

    def serve(self, case: Case) -> Served:
        return Served(server_of(self.home, case.work), self.served.base_line, self.served.parts, self.served.server_instructions, "mm", None,
                      mm_record(case))

    def audit(self, case: Case, calls: Sequence[dict]) -> dict:
        return self.audit_with(case, calls)


class ReplaySystem:
    """The system `mm` with a fixed `Served` for each agent, as the pilot recorded it."""

    def __init__(self, served: dict[str, Served], home: Path, needs_gpu: bool = False, audit=mm_audit) -> None:
        self.served, self.home, self.needs_gpu, self.audit = served, home, needs_gpu, audit

    def describe(self, variant):
        return Description(name="mm", version="recorded", variant=variant, reads=frozenset({"image", "video", "audio", "text", "records"}),
                           local=True, needs_gpu=self.needs_gpu, gpu_memory_gb=None, network=False, license="none", config_hash="c",
                           store_hash="s", horizon="cut", has_oracle=False)

    def open(self, store, dump, context):
        return ReplaySession(self.served[context.agent], self.home, self.audit)


COMMAND = threading.local()  # the command the patched `claude_command` and `codex_command` give, for the exec this thread is running
# The name of the pilot's temp folder, where it read the data and worked, in a recorded stream (in every way a path is written there: escaped in the
# JSON text, with slashes, quoted for PowerShell). A run's deny words name the temp folder of the runs recorded before MMSB (G1's F1: their answers
# are on that disk), which a recorded stream of them cannot help naming: the replay's folder has another name.
EARLIER_TEMP = re.compile(r"mm-bench-temp", re.IGNORECASE)
REPLAY_TEMP = "replay-temp"


def relocated(events: Path, into: Path) -> Path:
    """The stream of `events` with the pilot's temp folder named `REPLAY_TEMP`, in `into`."""
    text = EARLIER_TEMP.sub(REPLAY_TEMP, events.read_text(encoding="utf-8"))
    into.parent.mkdir(parents=True, exist_ok=True)
    into.write_text(text, encoding="utf-8")
    return into


def rollout_lines(source: Source) -> list[str]:
    """What Codex's own rollout file held that the stream does not: the week's use at the end of the exec, and a `view_image` call for
    each the exec made (the stream has no item for it)."""
    recorded = source.recorded
    used = {"type": "event_msg", "payload": {"type": "token_count", "rate_limits": {
        "primary": {"window_minutes": 10080, "used_percent": recorded["used_percent"]}}}}
    calls = [t for t in json.loads(source.tools.read_text(encoding="utf-8")) if t["tool"] == "view_image"] if source.tools else [
        {"arguments": {"path": "C:\\recorded\\picture.png"}} for _ in range(recorded["tools"].get("view_image", 0))]
    return [json.dumps({"type": "session_meta", "payload": {"id": "x"}}), json.dumps(used),
            *(json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "view_image",
                                                              "arguments": EARLIER_TEMP.sub(REPLAY_TEMP, json.dumps(c["arguments"]))}})
              for c in calls)]


def thread_of(events: Path) -> str:
    first = json.loads(events.read_text(encoding="utf-8").split("\n", 1)[0])
    return first["thread_id"]


def install(monkeypatch, tmp_path: Path) -> Path:
    """The patches that let the real agents run a stand-in: the command, Codex's saved model table, a Git for Windows tree and the
    ffmpeg of the run. Returns the stand-in script."""
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN, encoding="utf-8")
    monkeypatch.setattr(claude, "claude_command", lambda: COMMAND.value)
    monkeypatch.setattr(codex, "codex_command", lambda: COMMAND.value)
    monkeypatch.setattr(codex, "models_of", lambda command: json.loads((Path(__file__).parent / "oracles" / "codex-models.json").read_text(
        encoding="utf-8")))
    git = tmp_path / "Git"
    for sub in ("cmd", "bin", "usr/bin"):
        (git / sub).mkdir(parents=True, exist_ok=True)
    (git / "bin" / "bash.exe").write_text("")
    monkeypatch.setattr(claude, "git_root", lambda: git)
    monkeypatch.setenv("MMSB_CODEX_EFFORT", "medium")  # the pilot ran Codex at medium
    ffmpeg = tmp_path / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))  # `ensure_bin` copies a made-up file, not 80 MB
    return script


class Replayer:
    """An agent for `run_agents`: it finds the recorded exec of the spec's question, points the stand-in at its stream (with the pilot's temp
    folder renamed, `relocated`), and runs the real agent. `edited` maps (directory, qid) to a stream made for the test; `hold` keeps the
    stand-in alive after its last line, for the execs in `holding` (all of them when that is None)."""

    def __init__(self, agent: str, sources: Sequence[Source], script: Path, tmp_path: Path, hold: float = 0.0, edited=None,
                 holding=None) -> None:
        self.agent, self.script, self.hold, self.tmp_path, self.edited = agent, script, hold, tmp_path, edited or {}
        self.holding = holding
        self.by_key = {(s.directory, s.qid): s for s in sources}
        self.ran: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        key = (spec.run_dir.parent.name, spec.run_dir.name)
        source = self.by_key[key]
        events = relocated(self.edited.get(key, source.events), self.tmp_path / "streams" / f"{self.agent}-{key[0]}-{key[1]}.jsonl")
        hold = self.hold if self.holding is None or key in self.holding else 0.0
        COMMAND.value = [sys.executable, str(self.script), str(events), str(hold)]
        with self.lock:
            self.ran.append(key)
        if self.agent == "codex":
            day = codex.codex_home() / "sessions" / "2026" / "11" / "04"
            day.mkdir(parents=True, exist_ok=True)
            (day / f"rollout-2026-11-04T00-00-00-{thread_of(events)}.jsonl").write_text("\n".join(rollout_lines(source)), encoding="utf-8")
        return (claude.ask_claude if self.agent == "claude" else codex.ask_codex)(spec)


def run_of(root: Path, timeout_s: float = 120.0) -> tuple[RunFolder, Dump]:
    """A run folder of the pilot's questions with made-up frozen data, as the replay's world."""
    questions = pilot_questions()
    folder, temp = root / "runs" / "run", root / "temp"
    data = temp / "mmb-r1-data"
    (data / "notes").mkdir(parents=True)
    (data / "notes" / "a.md").write_text("a note", encoding="utf-8")
    folder.mkdir(parents=True)
    fs.write_json(folder / "data-manifest.json", fs.stats_of(data))
    write_questions(folder / "questions.jsonl", questions)
    run = RunFolder(folder, 1, "pilot", tuple(DAYS), data, temp, "r1", timeout_s, scenarios.get("v7_screen"), ())
    return run, dump_of(run)


def dump_of(run: RunFolder) -> Dump:
    return Dump("k" * 64, run.data, "v7_screen", (), ())


def whole(cell_directory: str) -> bool:
    """Whether a replay of the results folder `cell_directory` is compared on the whole `work`: the files cell's always, an mm cell's where the mm
    extra is installed."""
    return cell_directory == "files" or mm_installed()


def differences(new: dict, recorded: dict, *, work_extra: Callable[[dict], dict] | None = None, skip: Sequence[str] = (),
                extra: Sequence[str] = ()) -> list[str]:
    """The fields in which a replayed result is not the recorded one, as strings (empty: the same). The `work` of a result is compared
    on its neutral fields, and on all of them where `work_extra` says what the system adds; `extra` are further fields of the record."""
    out = [f"{k}: {new.get(k)!r} != recorded {recorded.get(k)!r}" for k in (*SAME, *extra)
           if k not in skip and new.get(k) != (recorded.get(k) if k in OPTIONAL else recorded[k]) and not said_now(k, new, recorded)]
    if work_extra is None:
        out += [f"work.{k}: {new['work'].get(k)!r} != recorded {recorded['work'][k]!r}" for k in NEUTRAL_WORK
                if new["work"].get(k) != recorded["work"][k]]
    else:
        out += [f"work: {new['work']!r} != recorded {recorded['work']!r}"] if new["work"] != recorded["work"] else []
    return out


def said_now(key: str, new: dict, recorded: dict) -> bool:
    """The intended difference of 5449e06: a Claude result says the effort it ran at, the default when none was asked
    (`claude.effort()`), where the pilot's results, recorded before, say none."""
    return key == "agent_effort" and new.get("agent") == "claude" and recorded.get(key) is None and new.get(key) == claude.effort()


def class_counts(sources: Sequence[Source]) -> Counter:
    return Counter(s.klass for s in sources)
