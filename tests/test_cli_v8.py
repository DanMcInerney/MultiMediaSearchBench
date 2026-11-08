"""The command line over EgoLife (V8): ported from tests/test_egolife.py of MultimediaSearch at fe0ece5 (plan 7.4, oracle 4) the tests of what its command
line did (`test_a_dry_run_names_the_files_folder_and_makes_nothing`, `test_the_commands_say_what_they_will_run_and_report_writes_its_files`,
`test_the_estimate_prices_a_blind_exec_at_its_own_figure`), and new: `mmsb build` of a download, and the finding of U6's notes, that the
agents with their own tools must not read the whole dump. Over the fake download of tests/v8_fake.py and the stand-in system of
tests/stub_system.py; nothing starts a real agent.

The old data sat in the run's folder, which the deny list covers; the new dump is a folder of its own in the temp folder. The agents read
the folder `Scenario.view` cuts from it, for the question's horizon, so the dump (which holds the clips past it) is hidden: `RunFolder.deny`
names it when what the agent reads is not the dump itself. The tests of that rule are the two after the build.

Intended differences of an expectation: the arms are configurations, a run's estimate counts every configuration at `--jobs` (the old mm arm
went one at a time, which the system's own line says now), and the Markdown names `stub/all` where it named `mm`."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import threading
import types

import cli_world
import imageio_ffmpeg
import pytest
from agent_spec import ISOLATION, clocked
from cli_world import world  # noqa: F401
from stub_system import StubSession, StubSystem
from v8_fake import DAYS, V8, oracle_file, recipe, source  # noqa: F401

from mmsb import cli, scenarios
from mmsb.core import dryrun, fs
from mmsb.core.agents import claude, codex
from mmsb.core.agents.common import reached
from mmsb.core.build import read_dump
from mmsb.core.contract import IngestReport
from mmsb.core.costing import BLIND_MIN, EXEC_MIN
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios.v8_egolife.view import view_dir

pytestmark = pytest.mark.xdist_group("v8")
MEASURED = IngestReport(0.0, 0.0, 0, (), (), {"decode_s": 0.024})
CLIP = "glasses/2026-10-29/11.09.42.mp4"
N = 10  # the questions of the fake download: 8 of video and 2 of speech


class Cut(StubSession):
    """The stand-in system's session with the audit fields of a system that is cut to the horizon: no recording past it, one ask."""

    def audit(self, case, calls) -> dict:
        return {"after_horizon": 0, "asks": 1, "asks_past_horizon": 0}


class CutSystem(StubSystem):
    def open(self, store, dump, context):
        self.note("open", None, context.agent, context.variant, store)
        return Cut(self, context)


class Spy:
    """An agent that keeps the spec of each exec and answers right with its tools, wrong without (a blind exec says the next letter)."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.specs: dict[tuple[str, str], object] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.specs[folder, q.qid] = spec
        answer = q.grade.truth if folder != "blind" else next(x for x in "ABCD" if x != q.grade.truth)
        return {"final": f"ANSWER: {answer}", "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


@pytest.fixture(scope="module")
def built(tmp_path_factory, source, recipe):  # noqa: F811
    """`mmsb build` of the fake download with the blind, files and stand-in configurations: the run folder and the registry."""
    root = tmp_path_factory.mktemp("cli-v8")
    ffmpeg = root / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
        cli_world.use(patch, CutSystem(report=MEASURED))
        registry = cli_world.registry(root / "systems")
        argv = ["build", "--scenario", "v8_egolife", "--source", str(source), "--days", ",".join(d.isoformat() for d in DAYS), "--per-age", "2",
                "--speech", "2", "--out", str(root / "runs"), "--temp", str(root / "temp"), "--cache", str(root / "cache"), "--configs", "blind,files,stub"]
        assert cli.main(argv, registry=registry) == 0
        (folder,) = (root / "runs").iterdir()
    return types.SimpleNamespace(folder=folder, registry=registry, recipe=recipe)


@pytest.fixture
def w(world, built, tmp_path):  # noqa: F811
    """The world with `run`, a copy of the built run with no results (it shares the dump, which is only read), and the system `stub`
    that is cut to the horizon."""
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "dry-run", "homes", "systems.json") or n.startswith("report.")]

    shutil.copytree(built.folder, tmp_path / "runs" / "run", ignore=skip)
    world.patch.setattr(claude, "check_login", lambda: None)
    cli_world.use(world.patch, CutSystem(report=MEASURED, rules=False))
    world.run = RunFolder.open(tmp_path / "runs" / "run", scenarios.get)
    world.registry_folder = built.registry
    world.spy = Spy(world.run.questions())
    return world


def run_command(w, *rest: str) -> int:
    return w.main("run", str(w.run.folder), *rest, agents=lambda: {"claude": Agent(w.spy), "codex": Agent(w.spy, must_report_use=True)})


# -- the build ---------------------------------------------------------------------------------------------------------------------


def test_a_build_of_the_download_links_the_clips_and_records_what_the_report_quotes(built):
    run = RunFolder.open(built.folder, scenarios.get)
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d{8}-\d{4}-v8-egolife-s1", run.name)
    assert (meta["scenario"], meta["scale"], meta["seed"], meta["days"], meta["recipe"]) == (
        "v8_egolife", "egolife", 1, [d.isoformat() for d in DAYS], built.recipe)
    assert meta["pools"] == V8.pool_counts(built.recipe) and meta["decode_s"] == 0.024, "the scenario's pools and the system's decode"
    assert meta["probe"] == {"status": "none", "failures": []}, "a public dataset has no shortcut probe"
    assert [c.name for c in run.configs] == ["blind", "files", "stub/all"]
    clips = [e for e in read_dump(run).entries if e.kind == "clip"]
    assert len(clips) == 24 and all(os.stat(run.data / e.path).st_nlink >= 2 for e in clips), "the clips are links to the download's"
    assert fs.changed(run.data, run.manifest) is None and sorted(p.name for p in (run.folder / "stores").iterdir()) == ["files", "stub"]


# -- the dump is hidden from the agents with their own tools ----------------------------------------------------------------------


def test_the_deny_list_names_the_dump_when_the_agent_reads_a_folder_cut_from_it_and_not_when_it_reads_the_dump(w):
    run = w.run
    own = view_dir(run.temp, run.tag, 0)
    hidden = run.deny(run.slot(0), ISOLATION, own)
    seen = run.deny(run.slot(0), ISOLATION, run.data)
    default = run.deny(run.slot(0), ISOLATION)

    def stopped(deny, text: str, output: bool = False) -> bool:
        return reached(text, deny, output) is not None

    other = run.temp / "mmb-0123456789-data"  # another run's dump, as a listing of the temp folder shows it
    for text in (f"ffmpeg -i {run.data}/{CLIP} frame.png", f"cat {run.data}\\transcript\\2026-10-29.jsonl", f"copy ..\\{run.data.name}\\glasses x",
                 f"dir {run.temp}\\*-data", f"ffmpeg -i {other}/{CLIP} frame.png", f"dir {other}", json.dumps({"path": str(other)})):
        assert stopped(hidden, text), text
        assert not stopped(seen, text) and not stopped(default, text), f"{text}: the dump is the agent's own when it reads it"
    assert stopped(hidden, json.dumps({"file_path": f"{run.data}\\{CLIP.replace('/', chr(92))}"})), "a tool's input, as the watcher reads it"
    # the agent's own view and a listing of the temp folder are not refused
    assert not stopped(hidden, f"ffmpeg -i {own}/{CLIP} frame.png") and not stopped(hidden, f"Input #0, from '{own}\\glasses\\x.mp4'", output=True)
    assert not stopped(hidden, f"{run.data.name}\n{own.name}", output=True), "a listing of the temp folder may name the dump"


def test_the_folders_checked_are_those_the_agent_reads_and_a_dump_that_is_hidden_may_hold_a_word(w, tmp_path):
    run = w.run
    own = view_dir(run.temp, run.tag, 0)
    run.check_paths(ISOLATION, own)
    run.check_paths(ISOLATION)
    # a dump's name as a run gives it: a bare `data`, denied by name when hidden, is in any path through AppData (pytest's own temp)
    refused = dataclasses.replace(run, data=tmp_path / ".codex" / "mmb-0123456789-data")
    refused.check_paths(ISOLATION, own)  # the dump is not read: the word in its path is only more reason to keep the agent off it
    with pytest.raises(ValueError, match=r"holds '.*\.codex"):
        refused.check_paths(ISOLATION)
    with pytest.raises(ValueError, match="holds 'mmb-"):
        run.check_paths(ISOLATION, run.temp / f"mmb-{run.tag}-d0" / f"mmb-{run.tag}-w1")


def test_a_temp_folder_an_agent_may_not_name_is_refused_before_the_run(w, tmp_path, monkeypatch):
    """The first baseline: TEMP was in the tests' folder, a compiler's warning printed its path, and the exec ended as an escape."""
    monkeypatch.setenv("TMP", str(tmp_path / ".codex" / "tmp"))
    with pytest.raises(ValueError, match=r"TMP is .*holds '.*\.codex"):
        w.run.check_paths(ISOLATION)
    monkeypatch.setenv("TMP", str(tmp_path / "plain"))
    w.run.check_paths(ISOLATION)


def test_an_exec_with_its_own_tools_is_given_a_view_and_a_deny_list_that_names_the_whole_dump(w):
    run = w.run
    assert run_command(w, "--agents", "claude", "--configs", "blind,files,stub", "--only", "ego-001,ego-018") == 0
    files = w.spy.specs["files", "ego-001"]
    assert files.data == view_dir(run.temp, run.tag, next(k for k in range(4) if run.slot(k) == files.work)) and files.data != run.data
    assert reached(f"ffmpeg -i {run.data}/{CLIP} out.png", files.deny) is not None, "the whole dump holds clips past this question's horizon"
    assert reached(f"ffmpeg -i {files.data}/{CLIP} out.png", files.deny) is None, "the view is the agent's"
    assert reached(json.dumps({"command": f"ls {run.temp}"}), files.deny) is None, "a listing of the temp folder is not refused"
    assert reached(json.dumps({"command": f"ls {run.temp}/{run.data.name}"}), files.deny) is not None
    assert files.deny == run.deny(files.work, cli.Registry(w.registry_folder).isolation(), files.data)
    # the other configurations read no folder: no view, and nothing to hide
    assert w.spy.specs["blind", "ego-001"].data is None and w.spy.specs["stub-all", "ego-001"].data is None


# -- the commands -------------------------------------------------------------------------------------------------------------------


def test_a_dry_run_names_the_files_folder_and_makes_nothing(w, capsys):
    run = w.run
    w.patch.setattr(dryrun, "shutil", types.SimpleNamespace(which=lambda name: None))
    w.patch.setattr(codex, "models_of", lambda command: {"models": [{"slug": codex.model(), "supported_reasoning_levels": [{"effort": "high"}]}]})
    assert w.main("run", str(run.folder), "--agents", "claude,codex", "--configs", "blind,files", "--dry-run", "--timeout", "1200") == 0
    out = capsys.readouterr().out
    assert "nothing was run" in out and "You cannot see my recordings." in out and f"are in {view_dir(run.temp, run.tag, 0)}:" in out
    assert not view_dir(run.temp, run.tag, 0).exists() and not (run.folder / "results").exists()
    for agent in ("claude", "codex"):
        for folder in ("blind", "files"):
            assert (run.folder / "dry-run" / agent / folder / "ego-001" / "prompt.txt").read_text(encoding="utf-8").startswith("It is now")
    assert fs.read_json(run.folder / "run.json")["timeout_s"] == 900.0, "a dry run keeps no timeout"


def test_the_commands_say_what_they_will_run_and_report_writes_its_files(w, capsys):
    assert run_command(w, "--agents", "claude,codex", "--subset", "pilot", "--only", "ego-001") == 0
    out = capsys.readouterr().out
    assert "egolife, seed 1, 2 days, the pilot questions, only ego-001" in out and out.count("1 to run of 1") == 6
    assert re.search(r"claude\s+blind\s+1 to run of 1", out) and re.search(r"codex\s+stub/all\s+1 to run of 1", out) and "6 execs to run" in out
    assert run_command(w, "--agents", "claude") == 0
    capsys.readouterr()
    assert run_command(w, "--agents", "claude") == 0
    assert capsys.readouterr().out.count("0 to run of 10") == 3, "a second command has nothing left to run"
    assert w.main("report", str(w.run.folder)) == 0 and "# EgoLifeQA" in capsys.readouterr().out
    figures = fs.read_json(w.run.folder / "report.json")
    assert figures["agents"]["claude"]["stub/all"]["paired"] == N and figures["run"]["decode_s"] == 0.024
    assert figures["run"]["pools"] == V8.pool_counts(fs.read_json(w.run.folder / "run.json")["recipe"])
    assert (w.run.folder / "report.md").is_file()


def test_the_estimate_prices_a_blind_exec_at_its_own_figure(w, capsys):
    assert run_command(w, "--agents", "claude,codex") == 0
    # 10 questions in each of 6 cells; every configuration is counted at the 3 jobs asked (the old mm arm went one at a time, and says so)
    hours = (2 * BLIND_MIN / 3 + 2 * EXEC_MIN / 3 + 2 * EXEC_MIN / 3) * N / 60
    assert f"60 execs to run, about {hours:.1f} h at {EXEC_MIN:g} min an exec ({BLIND_MIN:g} for a blind one) over 3 jobs" in capsys.readouterr().out
