"""Oracle 5 (plan 7.5): the recorded execs replayed through the real run loop. A stand-in process prints a recorded `events.jsonl`, and the real
`ask_claude` / `ask_codex` (watcher, stream reader, classifier) inside the real `run_agents` (order, record, stops) make a result, which is
compared with the recorded result of the same exec. See tests/replay.py for the stand-in and what is compared.

| class            | source                                                 | expected                                                         |
| normal           | 8 checked in; all 104 of the recorded run when there   | the recorded result, field for field                             |
| timeout          | 3 recorded, held alive until killed                    | `timeout:`, a wrong answer, done (not asked again)               |
| capped           | 3 recorded (`error_max_turns`)                         | `capped:`, a wrong answer, done                                  |
| escaped          | planted: a recorded files exec, one command edited     | `escaped:`, process killed, the next exec runs                   |
| setup            | planted: an extra tool in the init event, or Codex's   | `setup:`, not done, the agent stops, the next exec does not run  |
| setup, GPU lost  | 20 recorded in set-aside-gpu-lost, `gpu_lost` injected | `setup: the GPU was lost, ...`, not done, the agent stops        |

The recorded run, `D:\\mm-bench`, is read in place and never changed; without it the cases that need it skip, saying so. The
checked-in cases need the time zone the pilot's week was made in (its questions are built from the plan, whose times are local).
Where the mm extra is installed an mm cell is compared on the whole `work` (the ask jobs, from systems/mm/audit.py) and on the three fields its `serve`
adds (`gpu_estimate_s`, `exclusive`, `beyond_timeout`, from the system's estimate over the pilot's table of windows, tests/mm/tables), for every mm exec
replayed, the timeouts and the lost-GPU ones included; without it, on the neutral fields (tests/replay.py)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from replay import (CELLS, MM_FIELDS, RECORDED_RUN, ReplaySystem, Replayer, Source, checked_in, class_counts, differences, dump_of, gpu_lost_run,
                    install, mm_audit, mm_installed, pilot_questions, recorded_run, run_of, whole)
from test_exec import pilot_execs, recorded_text, served_from
from v7_recorded import zone_skip

from mmmb import systems
from mmmb.core import gpu
from mmmb.core.config import Config
from mmmb.core.contract import Served
from mmmb.core.runfolder import is_done
from mmmb.core.runner import Agent, run_agents

FILES = Config("files", "files", None, True)
MM = Config("mm/d1-3B", "mm", "d1-3B", False)
CONFIG = {"files": FILES, "mm-d1-3B": MM}
LOST = "nvidia-smi exited 6: GPU is lost.  Reboot the system to recover this GPU"


@pytest.fixture(autouse=True)
def pilot_zone():
    """The checked-in cases grade against the pilot's questions, whose times are local to the pilot's time zone."""
    if reason := zone_skip():
        pytest.skip(reason)


def needs_record() -> None:
    if not RECORDED_RUN.is_dir():
        pytest.skip(f"{RECORDED_RUN} is not here: the 110 recorded streams, the timeouts, the capped ones and the GPU-lost ones are read from it")


def served() -> Served:
    """The pilot's recorded instructions, taken apart as oracle 3 does: the base line, the server's own words and the rest."""
    sha = {a: next(e["instructions_sha256"] for e in pilot_execs() if e["agent"] == a and e["dir"] != "files") for a in ("claude", "codex")}
    return served_from(recorded_text(sha["claude"]), recorded_text(sha["codex"]), None)


def run_cell(tmp_path, monkeypatch, agent: str, sources: list[Source], *, timeout_s: float = 120.0, hold: float = 0.0, needs_gpu: bool = False,
             edited=None, holding=None, jobs: int = 3, only=None, replayer=None, run=None):
    """The real `run_agents` over the real agent on a stand-in, for the execs of `sources` (all of one results folder), in `run` or a
    new one."""
    directory = {s.directory for s in sources}
    assert len(directory) == 1
    config = CONFIG[directory.pop()]
    script = install(monkeypatch, tmp_path)
    run, dump = (run, dump_of(run)) if run else run_of(tmp_path, timeout_s)
    system = ReplaySystem({agent: served()}, tmp_path / "home", needs_gpu, mm_audit)
    replayer = replayer or Replayer(agent, sources, script, tmp_path, hold, edited, holding)
    run_agents(run, {agent: Agent(replayer, must_report_use=agent == "codex")}, [config], dump, lambda s: tmp_path / "stores" / s, jobs,
               load=lambda name: system if name == "mm" else systems.load(name), isolation=systems.isolation(),
               only=only or [s.qid for s in sources], answer_format="line")
    return run, replayer


def result_of(run, agent: str, source: Source) -> dict:
    return json.loads((run.folder / "results" / agent / source.directory / f"{source.qid}.json").read_text(encoding="utf-8"))


def whole_of(s: Source) -> dict:
    """How a replay of `s` is compared beyond the neutral fields (see tests/replay.py): the whole `work`, and for an mm exec its `serve`'s fields."""
    return {"work_extra": (lambda w: w) if whole(s.directory) else None, "extra": MM_FIELDS if s.directory != "files" and mm_installed() else ()}


def check_normal(run, agent: str, sources: list[Source]) -> None:
    wrong = {}
    for s in sources:
        found = differences(result_of(run, agent, s), s.recorded, **whole_of(s))
        if found:
            wrong[s.qid] = found
    assert not wrong, wrong


def by_until(sources: list[Source]) -> list[Source]:
    """In the order the loop runs them with one job."""
    keys = {q.qid: (q.scope.until, q.qid) for q in pilot_questions()}
    return sorted(sources, key=lambda s: keys[s.qid])


# -- normal: eight checked in ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("agent", "directory"), CELLS, ids=[f"{a}-{d}" for a, d in CELLS])
def test_the_checked_in_execs_replay_as_they_were_recorded(agent, directory, tmp_path, monkeypatch):
    sources = [s for s in checked_in() if (s.agent, s.directory) == (agent, directory)]
    assert len(sources) == 2
    run, replayer = run_cell(tmp_path, monkeypatch, agent, sources)
    assert sorted(replayer.ran) == sorted((directory, s.qid) for s in sources)
    check_normal(run, agent, sources)
    for s in sources:
        assert is_done(result_of(run, agent, s)) and result_of(run, agent, s)["timeout_s"] == 120.0


def test_the_checked_in_execs_are_eight_and_two_for_each_cell_of_the_pilot():
    assert class_counts(checked_in()) == {"normal": 8}
    assert sorted((s.agent, s.directory) for s in checked_in()) == sorted(CELLS * 2)


# -- the recorded run: normal and capped, timeouts, and the GPU lost ----------------------------------------------------------------

def test_the_recorded_run_has_the_classes_the_oracle_covers():
    needs_record()
    sources = recorded_run()
    assert len(sources) == 110 and class_counts(sources) == {"normal": 104, "timeout": 3, "capped": 3}
    assert sorted((s.agent, s.directory, s.qid) for s in sources if s.klass == "timeout") == [
        ("claude", "files", "audio-mic-alarm-3"), ("claude", "mm-d1-3B", "video-absent-error-2"), ("claude", "mm-d1-3B", "video-screen-error-3")]
    assert sorted(s.qid for s in sources if s.klass == "capped") == ["video-absent-error-1", "video-absent-error-2", "video-screen-error-3"]
    assert all(s.agent == "claude" for s in sources if s.klass != "normal")
    assert class_counts(gpu_lost_run()) == {"normal": 20}, "recorded as answered: the guard did not exist yet"


@pytest.mark.parametrize(("agent", "directory"), CELLS, ids=[f"{a}-{d}" for a, d in CELLS])
def test_every_recorded_normal_and_capped_exec_of_the_cell_replays_as_it_was_recorded(agent, directory, tmp_path, monkeypatch):
    needs_record()
    sources = [s for s in recorded_run() if (s.agent, s.directory) == (agent, directory) and s.klass in ("normal", "capped")]
    assert sources
    run, replayer = run_cell(tmp_path, monkeypatch, agent, sources, timeout_s=120.0)
    assert len(replayer.ran) == len(sources)
    check_normal(run, agent, sources)
    for s in sources:
        saved = result_of(run, agent, s)
        assert (saved["problem"] or "").split(":")[0] == (s.recorded["problem"] or "").split(":")[0]
        assert is_done(saved) and saved["passed"] == s.recorded["passed"]
        if s.klass == "capped":
            assert saved["problem"].startswith("capped:") and saved["passed"] is False


def test_the_three_recorded_timeouts_are_timeouts_wrong_answers_and_not_asked_again(tmp_path, monkeypatch):
    needs_record()
    sources = [s for s in recorded_run() if s.klass == "timeout"]
    assert len(sources) == 3
    for directory in sorted({s.directory for s in sources}):
        mine = [s for s in sources if s.directory == directory]
        folder = tmp_path / directory
        folder.mkdir()
        run, replayer = run_cell(folder, monkeypatch, "claude", mine, timeout_s=3.0, hold=3600.0)
        for s in mine:
            saved = result_of(run, "claude", s)
            assert s.recorded["problem"] == "timeout: no answer in 900 s", "the recorded text, with the limit of the recorded run"
            assert saved["problem"] == "timeout: no answer in 3 s" and saved["passed"] is False and is_done(saved)
            assert saved["seconds"] < 30, "the stand-in was killed at the limit, not left to its hold"
            assert not differences(saved, s.recorded, skip=("problem",), **whole_of(s)), differences(saved, s.recorded, skip=("problem",), **whole_of(s))
        again = Replayer("claude", mine, replayer.script, folder, 3600.0)
        run_cell(folder, monkeypatch, "claude", mine, timeout_s=3.0, hold=3600.0, replayer=again, run=run)
        assert again.ran == [], "an exec that timed out is not asked again"


def test_the_twenty_recorded_execs_that_ended_on_a_lost_gpu_are_setup_problems_left_to_run_again(tmp_path, monkeypatch, capsys):
    needs_record()
    sources = gpu_lost_run()
    assert len(sources) == 20
    monkeypatch.setattr(gpu, "gpu_busy", lambda: None)  # the GPU is free to start with, and lost by the time an exec ends
    monkeypatch.setattr(gpu, "gpu_lost", lambda: LOST)
    run, _ = run_of(tmp_path)
    replayer = Replayer("codex", sources, install(monkeypatch, tmp_path), tmp_path)
    for s in sources:
        run_cell(tmp_path, monkeypatch, "codex", [s], needs_gpu=True, replayer=replayer, run=run, jobs=1)
        saved = result_of(run, "codex", s)
        assert saved["problem"] == f"setup: the GPU was lost, so mm's model may not have read: {LOST}"
        assert saved["passed"] is False and not is_done(saved)
        assert not differences(saved, s.recorded, skip=("problem", "passed"), **whole_of(s)), differences(saved, s.recorded, skip=("problem", "passed"), **whole_of(s))
    assert capsys.readouterr().out.count("stopping: codex: setup: the GPU was lost") == 20


# -- planted: an escape and a setup problem -----------------------------------------------------------------------------------------

def events_of(source: Source) -> list[dict]:
    return [json.loads(line) for line in source.events.read_text(encoding="utf-8").splitlines()]


def write_events(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


def plant_escape(agent: str, source: Source, tmp_path: Path) -> Path:
    """The recorded stream with its first command (Codex) or tool call (Claude) naming the owner's agent setup."""
    events = copy.deepcopy(events_of(source))
    if agent == "codex":
        shell = next(e for e in events if e.get("item", {}).get("type") == "command_execution")
        shell["item"]["command"] += " ; type C:\\Users\\dan\\.codex\\auth.json"
    else:
        first = next(e for e in events if e["type"] == "assistant" and any(b.get("type") == "tool_use" for b in e["message"]["content"]))
        block = next(b for b in first["message"]["content"] if b.get("type") == "tool_use")
        block["input"] = {"file_path": "C:\\Users\\dan\\.claude\\settings.json"} if block["name"] == "Read" else {
            "pattern": "x", "path": "C:\\Users\\dan\\.claude"}
        block["name"] = "Read" if block["name"] == "Read" else "Grep"
    return write_events(tmp_path / f"escape-{source.qid}.jsonl", events)


def plant_setup(agent: str, source: Source, tmp_path: Path) -> Path:
    """The recorded init event with an extra tool (Claude), or a stream that is only the failure to start the server (Codex)."""
    events = copy.deepcopy(events_of(source))
    if agent == "claude":
        init = next(e for e in events if e.get("type") == "system" and e.get("subtype") == "init")
        init["tools"] = [*init["tools"], "Bash"]
    else:
        events = [events[0], {"type": "error", "message": "required MCP servers failed to initialize: mm"}]
    return write_events(tmp_path / f"setup-{source.qid}.jsonl", events)


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_a_planted_escape_ends_the_exec_killed_and_the_configuration_goes_on(agent, tmp_path, monkeypatch):
    sources = by_until([s for s in checked_in() if (s.agent, s.directory) == (agent, "files")])
    planted, after = sources
    edited = {("files", planted.qid): plant_escape(agent, planted, tmp_path)}
    run, replayer = run_cell(tmp_path, monkeypatch, agent, sources, timeout_s=60.0, hold=3600.0, edited=edited, jobs=1,
                             holding={("files", planted.qid)})
    saved = result_of(run, agent, planted)
    assert saved["problem"].startswith("escaped: ") and (".claude" if agent == "claude" else ".codex") in saved["problem"]
    assert saved["passed"] is False and is_done(saved) and saved["seconds"] < 30, "killed, not left to its hold"
    if agent == "codex":
        assert saved["problem"] == "escaped: shell reached .codex"
    # the configuration went on to the next exec, which is as recorded
    assert replayer.ran == [("files", planted.qid), ("files", after.qid)]
    assert result_of(run, agent, after)["problem"] == after.recorded["problem"]


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_a_planted_setup_problem_stops_the_agent_and_is_left_to_run_again(agent, tmp_path, monkeypatch, capsys):
    sources = by_until([s for s in checked_in() if (s.agent, s.directory) == (agent, "mm-d1-3B")])
    planted, after = sources
    edited = {("mm-d1-3B", planted.qid): plant_setup(agent, planted, tmp_path)}
    run, replayer = run_cell(tmp_path, monkeypatch, agent, sources, timeout_s=60.0, hold=3600.0 if agent == "claude" else 0.0, edited=edited,
                             holding={("mm-d1-3B", planted.qid)}, jobs=1)
    saved = result_of(run, agent, planted)
    assert saved["problem"].startswith("setup: ") and saved["passed"] is False and not is_done(saved) and saved["seconds"] < 30
    assert f"stopping: {agent}: setup: " in capsys.readouterr().out
    assert replayer.ran == [("mm-d1-3B", planted.qid)], "the agent stopped: the next exec did not run"
    assert not (run.folder / "results" / agent / "mm-d1-3B" / f"{after.qid}.json").exists()
