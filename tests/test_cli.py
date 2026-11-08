"""Ported from tests/test_bench.py and tests/test_egolife.py of MultimediaSearch at fe0ece5 (plan 7.4, oracle 4): what the command line did, now
`mmsb` over any scenario and system, over the scenario `toy` (V7's plan, a dump of three files) and the stand-in system `stub` of
tests/cli_world.py. What moved, by the plan's 3.2 and 8:
- `bench.main` is `cli.main`, `agents` is `run`, `decider` is `retrieve`, `--models` is `--configs`, `show_decider` and the system's lines of
  `show_agents` are told from a system's `preview` (mmsb/core/dryrun.py), `set_timeout` is `RunFolder.with_timeout`, and the build and the
  ingest have their own tests (tests/test_cli_build.py);
- the intended differences of an expectation: the arms are configurations (`files`, `stub/all`), a run's estimate counts every configuration
  at `--jobs` (the old "mm at 1 job" is the system's to lower, and a line of its own says so), the words "ask" and "mm" of the limits line are
  "its work" and the configuration's name, and a run that was never run has no report (the old one said "No agent results yet");
- not here: the tests of the agents' own arguments (tests/test_agents.py, test_runner.py), the report's figures (tests/test_report.py), the
  probe's lines in the report (U5), and what needs mm's daemon (tests/mm).
Every expected value is the source's, or worked out from what the test gave the command."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import cli_world
import imageio_ffmpeg
import pytest
from agent_spec import spec_of
from cli_world import CLEAN, build_args, world  # noqa: F401
from stub_system import Stub, StubSystem, tiny_questions

from mmsb import cli, scenarios
from mmsb.core import dryrun, fs, paths, report
from mmsb.core.agents import claude, codex
from mmsb.core.atonce import JOBS
from mmsb.core.costing import EXEC_MIN
from mmsb.core.runfolder import JOBS_MAX, RunFolder
from mmsb.core.runner import AGENTS, TIMEOUT_S, Agent
from mmsb.core.select import pilot_questions

pytestmark = pytest.mark.xdist_group("cli")
QUESTIONS = tiny_questions()
N = len(QUESTIONS)
FIRST = QUESTIONS[0].qid
HEAVY = next(q for q in QUESTIONS if q.tags["template"] == "screen-error" and q.tags["level"] == "week")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """The toy run `mmsb build` makes with the files and the stand-in system's configurations, built once for the module."""
    root = tmp_path_factory.mktemp("cli")
    with pytest.MonkeyPatch.context() as patch:
        ffmpeg = root / "ffmpeg-source.exe"
        ffmpeg.write_bytes(b"x")
        patch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
        cli_world.use(patch, StubSystem(report=CLEAN))
        where = types.SimpleNamespace(out=root / "runs", temp=root / "temp", cache=root / "cache")
        assert cli.main(build_args(where), registry=cli_world.registry(root / "systems")) == 0
        (folder,) = (root / "runs").iterdir()
    return folder


@pytest.fixture
def w(world, built, tmp_path):  # noqa: F811
    """The world with `run`: a copy of the built run with no results (it shares the data, which is only read)."""
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "dry-run", "homes", "systems.json") or n.startswith("report.")]

    shutil.copytree(built, tmp_path / "runs" / "run", ignore=skip)
    world.run = RunFolder.open(tmp_path / "runs" / "run", scenarios.get)
    world.patch.setattr(claude, "check_login", lambda: None)
    return world


def result(run: RunFolder, agent: str, folder: str, qid: str) -> dict:
    return json.loads((run.folder / "results" / agent / folder / f"{qid}.json").read_text(encoding="utf-8"))


def argv(w, *rest: str) -> list[str]:
    return ["run", str(w.run.folder), *rest]


def no_agents_installed(patch) -> None:
    """The commands a dry run prints are `claude` and `codex` as written, and Codex lists the model it is pinned to."""
    patch.setattr(dryrun, "shutil", types.SimpleNamespace(which=lambda name: None))
    patch.setattr(codex, "models_of", lambda command: {"models": [
        {"slug": codex.model(), "supported_reasoning_levels": [{"effort": codex.effort()}, {"effort": "medium"}]}]})


# -- dry runs ----------------------------------------------------------------------------------------------------------------------


def test_a_dry_run_writes_the_first_exec_of_each_agent_and_configuration_prints_their_command_lines_and_runs_nothing(w, capsys):
    run = w.run
    no_agents_installed(w.patch)
    told, prompt = [], run.scenario.prompt
    w.patch.setattr(run.scenario, "prompt", lambda q, access: told.append(access.timeout_s) or prompt(q, access))
    assert w.main(*argv(w, "--agents", "claude,codex", "--dry-run", "--timeout", "1200")) == 0
    assert told and set(told) == {1200.0}, "the prompt is told the limit the command asks for"
    out = capsys.readouterr().out
    for text in ("--restricted", "--strict-mcp-config", "sandbox_mode=workspace-write", "nothing was run", "ANTHROPIC_API_KEY",
                 f"question {FIRST}", "environment:", "== claude files", "== claude stub-all", "== codex stub-all", "MMS_*"):
        assert text in out, text
    for agent in ("claude", "codex"):
        for folder in ("files", "stub-all"):
            exec_dir = run.folder / "dry-run" / agent / folder / FIRST
            assert (exec_dir / "instructions.md").exists()
            assert (exec_dir / "prompt.txt").read_text(encoding="utf-8").startswith("It is now")
    # with a system, the exec is prepared with the server it would be given
    mcp = json.loads((run.folder / "dry-run" / "claude" / "stub-all" / FIRST / "mcp.json").read_text(encoding="utf-8"))
    assert mcp == {"mcpServers": {"stub": {"command": sys.executable, "args": ["-m", "stub", "serve"], "env": {}}}} and "--mcp-config" in out
    assert "THE STUB'S NOTES" in (run.folder / "dry-run" / "claude" / "stub-all" / FIRST / "instructions.md").read_text(encoding="utf-8")
    assert "stub has recorded these sources" in (run.folder / "dry-run" / "codex" / "stub-all" / FIRST / "prompt.txt").read_text(encoding="utf-8")
    # nothing else was made, no session was opened, and the time given was not kept
    assert not (run.folder / "results").exists() and not (run.folder / "homes").exists() and not (run.folder / "execs").exists(), \
        "an exec folder may hold a run's record: a dry run writes beside it"
    assert not w.system.kinds("open") and w.system.kinds("preview") and not w.system.kinds("close")
    assert fs.read_json(run.folder / "run.json")["timeout_s"] == 900.0, "a dry run keeps no timeout"


def test_a_dry_run_is_not_refused_for_a_probe_that_failed_and_a_configuration_with_no_question_is_said_so(w, capsys):
    fs.write_json(w.run.folder / "run.json", {**fs.read_json(w.run.folder / "run.json"), "probe": {"status": "failed", "failures": ["a: x"]}})
    no_agents_installed(w.patch)
    assert w.main(*argv(w, "--agents", "claude", "--dry-run")) == 0 and "nothing was run" in capsys.readouterr().out
    text = next(q.qid for q in QUESTIONS if q.tags["modality"] == "text" and not q.tags["absent"])
    # stub/media reads pictures and video: a text question is not asked of it
    assert w.main(*argv(w, "--agents", "claude", "--configs", "stub/media", "--dry-run", "--only", text)) == 0
    assert "== claude stub-media: no question to ask it" in capsys.readouterr().out


# -- running -----------------------------------------------------------------------------------------------------------------------


def test_run_says_what_it_will_run_and_an_estimate_and_runs_it(w, capsys):
    stub = Stub()
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files"), agents=cli_world.agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert "claude files" in out and f"{N} to run of {N}" in out
    assert f"{N} execs to run, about {N * EXEC_MIN / 3 / 60:.1f} h at 2.5 min an exec over 3 jobs" in out
    assert len(stub.execs) == N and set(stub.qids("files")) == {q.qid for q in QUESTIONS}


def test_claudes_login_is_checked_once_before_its_first_exec_and_codexs_is_not_the_commands_to_check(w):
    logins, stub = [], Stub()
    w.patch.setattr(claude, "check_login", lambda: logins.append(1))
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--only", FIRST), agents=cli_world.agents_of(stub)) == 0
    assert logins == [1]
    assert w.main(*argv(w, "--agents", "codex", "--configs", "files", "--only", FIRST), agents=cli_world.agents_of(stub)) == 0
    assert logins == [1] and stub.qids() == [FIRST, FIRST]


def test_the_command_line_has_limits_on_its_arguments(w, capsys):
    for bad in (["--jobs", str(JOBS_MAX + 1)], ["--jobs", "0"], ["--agents", "gemini"], ["--subset", "half"], ["--controls", "nowhere/x"]):
        with pytest.raises(SystemExit):
            w.main(*argv(w, *bad))
        capsys.readouterr()
    assert w.main(*argv(w, "--configs", "gold")) == 1 and "gold is not built yet" in capsys.readouterr().err
    assert w.main(*argv(w, "--configs", "nothing")) == 1 and "is not a configuration" in capsys.readouterr().err
    assert w.main(*argv(w, "--only", "no-such-question", "--agents", "claude")) == 1 and "--only names questions" in capsys.readouterr().err
    assert w.main("run", str(w.run.folder.parent / "nowhere")) == 1 and "error:" in capsys.readouterr().err
    assert w.system.kinds("open") == []


def test_run_gives_each_configuration_its_questions_and_a_pilot_is_a_subset(w):
    stub = Stub()
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--subset", "pilot"), agents=cli_world.agents_of(stub)) == 0
    assert sorted(stub.qids()) == sorted(q.qid for q in pilot_questions(QUESTIONS, [])) and 0 < len(stub.execs) < N
    assert w.main(*argv(w, "--agents", "claude", "--configs", "stub/narrow", "--controls", "stub/narrow"), agents=cli_world.agents_of(Stub())) == 0
    noted = json.loads((w.run.folder / "systems.json").read_text(encoding="utf-8"))["stub/narrow"]
    assert noted["asked"] == noted["readable"] == ["image", "records", "text", "video"]
    assert [c.name for c in RunFolder.open(w.run.folder, scenarios.get).configs] == ["files", "stub/all", "stub/narrow"], \
        "a configuration run is kept in the run, so the report compares it"


def test_a_codex_effort_and_a_timeout_given_reach_the_execs_and_the_timeout_is_kept_in_the_run(w):
    stub = Stub()
    assert w.main(*argv(w, "--agents", "claude,codex", "--configs", "files", "--only", FIRST, "--codex-effort", "medium", "--timeout", "1200"),
                  agents=cli_world.agents_of(stub)) == 0
    assert {s.run_dir.parts[-3]: (s.effort, s.timeout_s) for s in stub.execs} == {"claude": (None, 1200.0), "codex": ("medium", 1200.0)}
    assert fs.read_json(w.run.folder / "run.json")["timeout_s"] == 1200.0 and RunFolder.open(w.run.folder, scenarios.get).timeout_s == 1200.0
    assert result(w.run, "claude", "files", FIRST)["timeout_s"] == 1200.0 and w.run.timeout_s == 900.0
    again = Stub()
    shutil.rmtree(w.run.folder / "results")
    assert w.main(*argv(w, "--agents", "codex", "--configs", "files", "--only", FIRST), agents=cli_world.agents_of(again)) == 0
    assert [(s.effort, s.timeout_s) for s in again.execs] == [(None, 1200.0)], "the timeout stays with the run; an effort does not"


def test_an_effort_wraps_an_agent_and_none_leaves_it_alone():
    seen = []
    agent = Agent(lambda spec: seen.append(spec.effort) or {}, must_report_use=True)
    assert cli.with_effort(agent, None) is agent
    wrapped = cli.with_effort(agent, "medium")
    assert wrapped.must_report_use
    wrapped.run(spec_of("blind"))
    assert seen == ["medium"]


def test_a_system_that_allows_fewer_execs_at_once_is_told_before_and_while_it_runs(w, capsys):
    cli_world.use(w.patch, StubSystem(report=CLEAN, job_limit=1))
    assert w.main(*argv(w, "--agents", "claude", "--configs", "stub/all", "--only", FIRST), agents=cli_world.agents_of(Stub())) == 0
    out = capsys.readouterr().out
    assert "  stub/all: 1 execs at once, not 3" in out
    assert re.search(r"claude\s+stub/all\s+1 to run of 1 \(1 at once\)", out)
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--only", FIRST), agents=cli_world.agents_of(Stub())) == 0
    assert "execs at once" not in capsys.readouterr().out, "a configuration with no system has all its jobs"


def test_run_names_the_questions_whose_work_may_need_more_gpu_seconds_than_the_exec_has(w, capsys):
    span = sum(b - a for a, b in HEAVY.cost_ranges)
    # the old test's numbers: 988 s is named, 700 s is (over 675 s, the exec's 900 s less the agent's own turns), 670 s is not
    for seconds, named in ((988.0, True), (700.0, True), (670.0, False)):
        cli_world.use(w.patch, StubSystem(report=CLEAN, rules=False, rate=seconds / span))
        shutil.rmtree(w.run.folder / "results", ignore_errors=True)
        assert w.main(*argv(w, "--agents", "codex", "--configs", "stub/all", "--only", HEAVY.qid), agents=cli_world.agents_of(Stub())) == 0
        out = capsys.readouterr().out
        assert (f"{HEAVY.qid} ({seconds:.0f} s)" in out) is named, seconds
        if named:
            assert "stub/all: its work over the whole range of 1 questions may need more GPU seconds than the 675 s an exec leaves it" in out
            assert "(75% of its 900 s: the agent's own turns take the rest)" in out and "its speed, not a miss" in out
        else:
            assert "its speed, not a miss" not in out


def test_agents_that_name_none_are_refused_where_they_are_checked(w, capsys):
    for text in ("", " , "):
        with pytest.raises(SystemExit):
            w.main(*argv(w, "--agents", text, "--dry-run"))
        assert "--agents is some of claude, codex, not none" in capsys.readouterr().err
    assert w.system.kinds("preview") == [] and w.system.kinds("open") == []


def test_a_command_that_is_refused_leaves_the_run_json_as_it_was_and_one_that_runs_keeps_the_time_and_the_configurations(w, capsys):
    stub = Stub()
    cli_world.use(w.patch, StubSystem(report=CLEAN, store_hash="t"))  # the system moved on: its store is refused
    before = (w.run.folder / "run.json").read_bytes()
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files,stub/narrow", "--timeout", "1200"), agents=cli_world.agents_of(stub)) == 1
    assert "was made by stub with store_hash s" in capsys.readouterr().err and stub.execs == []
    assert (w.run.folder / "run.json").read_bytes() == before, "G1's F12: the time given used to be kept before the refusal"
    probe_was(w, {"status": "failed", "failures": ["a: x"]})
    cli_world.use(w.patch, StubSystem(report=CLEAN))
    before = (w.run.folder / "run.json").read_bytes()
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files,stub/narrow", "--timeout", "1200"), agents=cli_world.agents_of(stub)) == 1
    assert "failed the shortcut probe" in capsys.readouterr().err and (w.run.folder / "run.json").read_bytes() == before
    probe_was(w, {"status": "passed", "failures": []})
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files,stub/narrow", "--timeout", "1200", "--only", FIRST),
                  agents=cli_world.agents_of(stub)) == 0
    kept = RunFolder.open(w.run.folder, scenarios.get)
    assert kept.timeout_s == 1200.0 and "stub/narrow" in [c.name for c in kept.configs]


def test_a_report_of_a_run_the_old_bench_made_says_what_it_is(w, tmp_path, capsys):
    old = tmp_path / "20261104-0314-pilot-s1"
    old.mkdir()
    shutil.copy2(Path(__file__).parent / "oracles" / "recorded" / "pilot-s1" / "run.json", old / "run.json")  # as bench.py wrote it: no scenario
    assert w.main("report", str(old)) == 1
    err = capsys.readouterr().err
    assert f"error: {old} was made before mmsb (its run.json names no scenario): mmsb reads only the runs it built" in err and "Traceback" not in err


# -- probes ------------------------------------------------------------------------------------------------------------------------


def probe_was(w, outcome) -> None:
    meta = fs.read_json(w.run.folder / "run.json")
    meta.pop("probe", None)
    fs.write_json(w.run.folder / "run.json", {**meta, **({"probe": outcome} if outcome else {})})


def test_run_refuses_a_run_whose_probe_failed_unless_anyway_and_only_warns_when_it_was_skipped(w, capsys):
    stub = Stub()
    probe_was(w, {"status": "failed", "failures": ["a: x", "b: y"]})
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files"), agents=cli_world.agents_of(stub)) == 1 and stub.execs == []
    err = capsys.readouterr().err
    assert "failed the shortcut probe (2 failures, in its shortcuts.md)" in err and "--anyway" in err
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--anyway"), agents=cli_world.agents_of(stub)) == 0 and len(stub.execs) == N
    out = capsys.readouterr().out
    assert "warning: " in out and "failed the shortcut probe" in out, "run anyway, and said so"
    # a skipped probe runs, with the warning
    stub.execs.clear()
    shutil.rmtree(w.run.folder / "results")
    probe_was(w, {"status": "skipped", "failures": []})
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files"), agents=cli_world.agents_of(stub)) == 0 and len(stub.execs) == N
    assert "built with --no-probe" in capsys.readouterr().out
    parse = cli.parser_of().parse_args
    assert parse(["run", "r"]).anyway is False and parse(["run", "r", "--anyway"]).anyway


def test_the_commands_say_when_a_runs_probe_failed_or_was_skipped_or_never_ran(w, capsys):
    outcomes = {"failed": {"status": "failed", "failures": ["a: x", "b: y"]}, "skipped": {"status": "skipped", "failures": []},
                "none": None, "passed": {"status": "passed", "failures": []}}
    cli_world.use(w.patch, cli_world.OracleSystem(report=CLEAN))
    lines = {}
    for label, outcome in outcomes.items():
        probe_was(w, outcome)
        shutil.rmtree(w.run.folder / "results", ignore_errors=True)
        capsys.readouterr()
        # `run` and `retrieve` warn before they spend anything
        assert w.main(*argv(w, "--agents", "claude", "--configs", "files", "--only", FIRST, "--anyway"), agents=cli_world.agents_of(Stub())) == 0
        warned = [line for line in capsys.readouterr().out.splitlines() if line.startswith("warning: ")]
        assert w.main("retrieve", str(w.run.folder), "--configs", "stub", "--dry-run") == 0
        warned += [line for line in capsys.readouterr().out.splitlines() if line.startswith("warning: ")]
        assert len(warned) == (0 if label == "passed" else 2), label
        lines[label] = warned
    assert "failed the shortcut probe (2 failures, in its shortcuts.md)" in lines["failed"][0] and lines["failed"][0] == lines["failed"][1]
    assert "was built with --no-probe" in lines["skipped"][0]
    assert "no shortcut probe record (it was built before the probe)" in lines["none"][0]


# -- retrieval ---------------------------------------------------------------------------------------------------------------------


def test_retrieve_says_what_it_will_ask_and_asks_it_once(w, capsys):
    oracle = cli_world.use(w.patch, cli_world.OracleSystem(report=CLEAN))
    asking = [q for q in QUESTIONS if q.retrieval is not None]
    assert 0 < len(asking) < N
    assert w.main("retrieve", str(w.run.folder), "--configs", "stub", "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"{len(asking)} retrieval asks to run of {len(asking)}" in out and f"{len(asking)} asks to run, " in out
    assert not oracle.kinds("open") and oracle.kinds("preview") and not (w.run.folder / "results" / "retrieval").exists()
    assert w.main("retrieve", str(w.run.folder), "--configs", "stub") == 0
    (opened,) = oracle.kinds("open")
    assert opened[2:4] == ("retrieval", "all") and [len(s.queries) for s in oracle.sessions] == [len(asking)] and len(oracle.kinds("close")) == 1
    folder = w.run.folder / "results" / "retrieval" / "stub-all"
    assert sorted(p.stem for p in folder.glob("*.json")) == sorted([q.qid for q in asking] + ["warm"])
    assert fs.read_json(folder / "warm.json")["config"] == "stub/all"
    assert w.main("retrieve", str(w.run.folder), "--configs", "stub") == 0
    assert len(oracle.kinds("open")) == 1, "nothing is left to ask: no session is opened for it"
    assert "0 retrieval asks to run" in capsys.readouterr().out
    assert w.main("retrieve", str(w.run.folder), "--configs", "files") == 1 and "can be asked for needles" in capsys.readouterr().err


def test_retrieve_tells_how_long_the_asks_take_from_the_systems_own_estimate_of_each_range(w, capsys):
    asking = [q for q in QUESTIONS if q.retrieval is not None]
    seconds = sum(sum(b - a for a, b in q.retrieval.ranges) for q in asking)
    for rate, text in ((0.0, "under a minute"), (3 * 60.0 / seconds, "about 3 min")):
        cli_world.use(w.patch, cli_world.OracleSystem(report=CLEAN, rate=rate))
        assert w.main("retrieve", str(w.run.folder), "--configs", "stub", "--dry-run") == 0
        out = capsys.readouterr().out
        assert f"{len(asking)} retrieval asks to run of {len(asking)}, {text}" in out and f"{len(asking)} asks to run, {text} in all" in out


# -- the report --------------------------------------------------------------------------------------------------------------------


def test_report_writes_its_files_for_a_run_and_pools_runs_of_other_seeds(w, tmp_path, capsys):
    assert w.main(*argv(w, "--agents", "claude", "--configs", "files,stub/all"), agents=cli_world.agents_of(Stub())) == 0
    capsys.readouterr()
    assert w.main("report", str(w.run.folder)) == 0
    out = capsys.readouterr().out
    text = (w.run.folder / "report.md").read_text(encoding="utf-8")
    assert text in out and f"written to {w.run.folder / 'report.md'} and report.json" in out and "Runs pooled: 1; seeds 7" in text
    figures = fs.read_json(w.run.folder / "report.json")
    assert figures["runs"][0]["seed"] == 7 and set(figures["agents"]["claude"]) == {"stub/all"}
    want, markdown = scenarios.get("v7_screen").report([w.run])  # (what the scenario gives for one run: the core's report of it)
    assert figures == json.loads(json.dumps(want)) == json.loads(json.dumps(report.numbers([w.run]))) and text == markdown.replace("\r\n", "\n")
    # a second run of another seed (the same results under it): pooled, and written in the first run's folder
    other = tmp_path / "eight"
    shutil.copytree(w.run.folder, other, ignore=lambda where, names: [n for n in names if n.startswith("report.")])
    fs.write_json(other / "run.json", {**fs.read_json(other / "run.json"), "seed": 8})
    assert w.main("report", str(w.run.folder), str(other)) == 0
    pooled = fs.read_json(w.run.folder / "report.json")
    assert [r["seed"] for r in pooled["runs"]] == [7, 8] and not (other / "report.md").exists()
    assert "Runs pooled: run (seed 7, tiny, 2 days," in (w.run.folder / "report.md").read_text(encoding="utf-8")
    assert w.main("report", str(w.run.folder), str(w.run.folder)) == 1, "the same seed and scale are the same questions"
    assert "same seed (7)" in capsys.readouterr().err
    assert w.main("report", str(tmp_path / "nowhere")) == 1 and "error:" in capsys.readouterr().err


def test_a_report_of_a_run_that_was_never_run_is_an_error_that_says_so(w, capsys):
    assert w.main("report", str(w.run.folder)) == 1
    err = capsys.readouterr().err
    assert f"{w.run.name} has not been run" in err and "systems.json" in err and not (w.run.folder / "report.md").exists()


# -- the rest of the command line ---------------------------------------------------------------------------------------------------


def test_the_command_line_names_the_scenario_and_has_the_defaults_of_the_old_ones():
    parser = cli.parser_of()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--scale", "tiny"])
    args = parser.parse_args(["build", "--scenario", "v7_screen", "--scale", "tiny"])
    assert (args.seed, args.out, args.temp, args.configs, args.no_probe, args.timeout) == (None, paths.RUNS, paths.TEMP, "files", False, TIMEOUT_S)
    assert parser.parse_args(["build", "--scenario", "v7_screen", "--no-probe", "--days", "2026-11-01, 2026-11-02"]).days == ["2026-11-01", "2026-11-02"]
    args = parser.parse_args(["run", "run", "--subset", "pilot", "--timeout", "1200"])
    assert (args.agents, args.jobs, args.subset, args.timeout, args.configs, args.controls, args.only, args.dry_run) == (
        ",".join(AGENTS), JOBS, "pilot", 1200.0, None, None, "", False)
    for bad in (["build", "--scenario", "v9_nothing"], []):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_the_command_line_tells_a_failed_subprocess_as_an_error_and_an_interrupt_as_a_status(w, monkeypatch, capsys):
    def failing(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["icacls"], stderr="denied")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_run", failing)
    assert w.main(*build_args(w)) == 1
    assert "error: Command '['icacls']' returned non-zero exit status 1" in capsys.readouterr().err
    monkeypatch.setattr(cli, "build_run", interrupted)
    assert w.main(*build_args(w)) == 130 and "interrupted: run the same command to continue" in capsys.readouterr().err
