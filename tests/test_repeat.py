"""`mmsb repeat` (core/repeat.py, plan 8.1): a recorded run again, its recipe built into a new run and the same agents run at the same models and
efforts over the same configurations and questions, then the command to compare the two. Over the toy scenario and the stand-in agents of
tests/echo_agent.py, which return the model and the effort (Codex's) they were given as the real agents do, so that a repeat that loses a
setting shows in the results as it would in a real run. The recorded pilot's own repeat (110 execs) is tests/oracles/test_recorded_repeat.py's."""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import cli_world
import pytest
from cli_world import CLEAN, world  # noqa: F401
from echo_agent import agents_of, record_toy
from stub_system import Stub, StubSystem, tiny_questions

from mmsb import cli, scenarios
from mmsb.core import fs, gpu, repeat
from mmsb.core.agents import claude
from mmsb.core.compare import compare_runs
from mmsb.core.report import results_of
from mmsb.core.runfolder import RunFolder

pytestmark = pytest.mark.xdist_group("cli")
N = len(tiny_questions())
EXECS = 4 * N  # two agents, two configurations


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """The toy run that was run by both agents, Claude with the model claude-test-1 and Codex at effort medium."""
    return record_toy(tmp_path_factory.mktemp("repeat"), "--codex-effort", "medium", "--claude-model", "claude-test-1")


@pytest.fixture
def w(world, recorded, tmp_path):  # noqa: F811
    """The world with the recorded run copied to `tmp_path/recorded`, and the folders a repeat builds into (not made)."""
    shutil.copytree(recorded, tmp_path / "recorded")
    world.recorded = RunFolder.open(tmp_path / "recorded", scenarios.get)
    world.new = types.SimpleNamespace(out=tmp_path / "new-runs", temp=tmp_path / "new-temp", cache=tmp_path / "new-cache")
    world.patch.setattr(claude, "check_login", lambda: None)
    return world


def args_of(w, *extra: str) -> list[str]:
    return ["repeat", str(w.recorded.folder), "--out", str(w.new.out), "--temp", str(w.new.temp), "--cache", str(w.new.cache), "--answer-format", "line", *extra]


def repeated(w) -> RunFolder:
    (folder,) = w.new.out.iterdir()
    return RunFolder.open(folder, scenarios.get)


def tree(root: Path) -> set[Path]:
    return {p for p in root.rglob("*")}


def test_a_recorded_run_notes_its_agents_models_and_efforts_from_its_results(w):
    meta = fs.read_json(w.recorded.folder / "run.json")
    # Claude's results hold no effort (the agent does not say), Codex's the medium it was asked for
    assert meta["agents"] == {"claude": {"model": "claude-test-1", "effort": None}, "codex": {"model": "codex-default", "effort": "medium"}}
    assert repeat.agents_of(results_of(w.recorded)) == meta["agents"]
    assert isinstance(meta["build_s"], float)


def test_a_dry_run_says_what_it_would_do_builds_nothing_calls_no_agent_and_makes_no_folder(w, capsys):
    stub, before = Stub(), tree(w.root)
    assert w.main(*args_of(w, "--dry-run"), agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert stub.execs == [] and not w.system.kinds("open") and not w.system.kinds("check_ready"), "no agent, no session, no readiness check"
    assert tree(w.root) == before, "nothing was made or changed anywhere: not the new run, not the temp folder, not a file in the recorded run"
    assert not w.new.out.exists() and not w.new.temp.exists()
    lines = [" ".join(line.split()) for line in out.splitlines()]
    assert f"repeat of recorded: toy, tiny, seed 7, 2 days; {N} questions, {EXECS} execs" in lines
    assert "claude: claude-test-1, at the agent's own effort (its results hold none)" in lines
    assert "codex: codex-default, at medium effort" in lines and "configurations: files, stub/all" in lines
    for agent in ("claude", "codex"):
        for config in ("files", "stub/all"):
            assert any(line.startswith(f"{agent} {config} {N} to run of {N} median ") for line in lines), (agent, config)
    assert f"{EXECS} execs to run over 3 jobs (a system may allow fewer), from the earlier runs of these cells:" in lines
    assert any("dump: written in the temp folder from the recipe (toy, tiny, seed 7, 2 days)" in line for line in lines)
    assert out.rstrip().endswith("nothing was built or started")


def test_a_repeat_builds_the_recipe_runs_the_same_agents_at_their_models_and_efforts_over_the_same_cells_and_says_what_to_compare(w, capsys):
    stub = Stub()
    assert w.main(*args_of(w), agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    new = repeated(w)
    assert dict(new.recipe) == dict(w.recorded.recipe) and new.folder != w.recorded.folder
    # the same cells: the same qids for each agent and configuration
    assert {a: {c: sorted(r) for c, r in cells.items()} for a, cells in results_of(new).items()} == \
           {a: {c: sorted(r) for c, r in cells.items()} for a, cells in results_of(w.recorded).items()}
    assert len(stub.execs) == EXECS
    by_agent = {agent: {(s.model, s.effort) for s in stub.execs if s.run_dir.parts[-3] == agent} for agent in ("claude", "codex")}
    assert by_agent == {"claude": {("claude-test-1", None)}, "codex": {("codex-default", "medium")}}, "Codex at medium, the recorded models"
    assert {s.timeout_s for s in stub.execs} == {w.recorded.timeout_s}
    assert fs.read_json(new.folder / "run.json")["agents"] == fs.read_json(w.recorded.folder / "run.json")["agents"]
    assert fs.read_json(new.folder / "run.json")["answer_format"] == "line"
    assert [c.name for c in new.configs] == ["files", "stub/all"]
    assert out.rstrip().endswith(f"repeated recorded as {new.name}; compare them, and read every flip:\n"
                                 f"  mmsb compare {w.recorded.folder} {new.folder}")
    c = compare_runs(w.recorded, new)
    assert c.refused == [] and all(not x.flips and x.inside for x in c.cells), "the stub answers the same, and the settings are the same"


def test_a_repeat_that_stopped_goes_on_in_its_run_with_into_and_asks_only_what_is_left(w, capsys):
    assert w.main(*args_of(w), agents=agents_of(Stub())) == 0
    new = repeated(w)
    lost, files = sorted(results_of(new)["codex"]["files"])[:3], next(c for c in new.configs if c.name == "files")
    for qid in lost:
        new.result_path("codex", files, qid).unlink()
    stub = Stub()
    assert w.main(*args_of(w, "--into", str(new.folder)), agents=agents_of(stub)) == 0
    assert sorted(stub.qids()) == lost and {s.effort for s in stub.execs} == {"medium"} and {s.model for s in stub.execs} == {"codex-default"}
    assert len(list(w.new.out.iterdir())) == 1, "no second run was built"
    # a run of another week is not this repeat's run
    other = cli_world.built_run(w, 9)
    capsys.readouterr()
    assert w.main(*args_of(w, "--into", str(other.folder)), agents=agents_of(Stub())) == 1
    assert "was not made from the recipe of recorded" in capsys.readouterr().err


def test_a_recorded_run_that_a_run_now_would_ask_other_questions_of_is_refused_and_so_is_one_that_has_no_agents_or_no_results(w, capsys):
    folder = w.recorded.folder
    # a cell with a result missing: a run now asks 19 questions of it, the recorded one was asked 18
    gone = sorted(results_of(w.recorded)["claude"]["files"])[0]
    w.recorded.result_path("claude", next(c for c in w.recorded.configs if c.name == "files"), gone).unlink()
    assert w.main(*args_of(w, "--dry-run"), agents=agents_of(Stub())) == 1
    assert f"recorded asked claude files {N - 1} questions, and a run now asks it {N} (a difference of 1, the first {gone})" in capsys.readouterr().err
    # no agents noted: it was not run to the end by `mmsb run`
    meta = fs.read_json(folder / "run.json")
    fs.write_json(folder / "run.json", {k: v for k, v in meta.items() if k != "agents"})
    assert w.main(*args_of(w, "--dry-run")) == 1 and "records no agents in its run.json" in capsys.readouterr().err
    # agents noted that its results do not say
    fs.write_json(folder / "run.json", {**meta, "agents": {**meta["agents"], "codex": {"model": "codex-default", "effort": "high"}}})
    assert w.main(*args_of(w, "--dry-run")) == 1 and "its results say" in capsys.readouterr().err
    # no results at all
    shutil.rmtree(folder / "results")
    assert w.main(*args_of(w, "--dry-run")) == 1 and "has no results: there is nothing to repeat" in capsys.readouterr().err


def test_results_of_two_settings_of_one_agent_are_not_one_experiment_and_run_says_so_when_it_ends(w, capsys):
    new = cli_world.built_run(w, 9)
    capsys.readouterr()
    first, rest = tiny_questions()[0].qid, [q.qid for q in tiny_questions()[1:]]
    assert w.main("run", str(new.folder), "--agents", "claude", "--configs", "files", "--only", first, "--claude-model", "m1", "--answer-format", "line",
                  agents=agents_of(Stub())) == 0
    assert fs.read_json(new.folder / "run.json")["agents"] == {"claude": {"model": "m1", "effort": None}}
    assert w.main("run", str(new.folder), "--agents", "claude", "--configs", "files", "--only", rest[0], "--claude-model", "m2", "--answer-format", "line",
                  agents=agents_of(Stub())) == 1
    assert "the results of claude were made at 2 settings, m1 and m2" in capsys.readouterr().err
    with pytest.raises(ValueError, match="were made at 2 settings"):
        repeat.agents_of(results_of(new))


def test_a_repeat_asks_the_systems_whether_they_are_ready_before_it_builds_and_a_dry_run_does_not(w, capsys):
    stub, unready = Stub(), cli_world.use(w.patch, StubSystem(report=CLEAN, unready="weights not downloaded"))
    assert w.main(*args_of(w, "--dry-run"), agents=agents_of(stub)) == 0
    assert not unready.kinds("check_ready")
    capsys.readouterr()
    assert w.main(*args_of(w), agents=agents_of(stub)) == 1
    assert "stub/all cannot run now: all: weights not downloaded" in capsys.readouterr().err
    assert unready.kinds("check_ready")
    assert stub.execs == [] and not w.new.out.exists() and not w.new.temp.exists(), "refused before the build, which takes minutes"


def test_a_recipe_the_scenario_cannot_make_again_is_refused(w):
    recipe = dict(w.recorded.recipe)
    assert repeat.options_of(w.recorded.scenario, recipe) == {"scale": "tiny", "seed": 7, "days": recipe["days"]}
    with pytest.raises(ValueError, match="makes a recipe other than the recorded one"):
        repeat.options_of(w.recorded.scenario, {**recipe, "utc_offsets": ["-0300", "-0300", "-0300"]})


def test_the_run_command_line_a_repeat_makes_has_each_agents_recorded_model_and_effort_and_none_it_did_not_record(w):
    r = repeat.repeat_of(w.recorded)
    argv = repeat.run_argv(Path("somewhere"), r, 2, 1)
    assert argv[:2] == ["run", "somewhere"]
    assert argv[argv.index("--agents") + 1] == "claude,codex" and argv[argv.index("--configs") + 1] == "files,stub/all"
    assert argv[argv.index("--jobs") + 1] == "2" and argv[argv.index("--gpu-jobs") + 1] == "1" and argv[argv.index("--timeout") + 1] == "900"
    assert argv[argv.index("--only") + 1] == ",".join(q.qid for q in tiny_questions())
    assert argv[-6:] == ["--claude-model", "claude-test-1", "--codex-model", "codex-default", "--codex-effort", "medium"], "(no --claude-effort: none recorded)"
    assert cli.parser_of().parse_args(argv).codex_effort == "medium"


def test_a_repeat_on_a_system_that_cannot_say_its_queue_is_refused_at_three_execs_once_the_run_is_built_and_goes_on_with_the_wall_clock(w, capsys):
    """The refusal of the working clock (core/clock.py) needs the system's store, so it comes after the build, which stays; `--clock wall` is passed on to the run."""
    cli_world.use(w.patch, StubSystem(report=CLEAN, reports_queue=False))
    stub = Stub()
    assert w.main(*args_of(w, "--jobs", "3"), agents=agents_of(stub)) == 1
    err = capsys.readouterr().err
    assert "stub/all: 3 execs at once on the working clock" in err and "one exec at a time (--jobs 1)" in err and "--clock" not in err
    assert stub.execs == []
    new = repeated(w)
    assert w.main(*args_of(w, "--into", str(new.folder), "--jobs", "3", "--clock", "wall"), agents=agents_of(stub)) == 0
    assert len(stub.execs) == EXECS
    clocks = {config: {r["clock"] for cells in results_of(new).values() for r in cells[config].values()} for config in ("files", "stub-all")}
    assert clocks == {"files": {"wall"}, "stub-all": {"wall"}}, "asked for the wall clock: every result says so, a system with a queue figure or not"
    assert cli.parser_of().parse_args(["repeat", "recorded"]).clock is None and cli.parser_of().parse_args(["run", "x"]).clock == "working"


def on_gpu(w, **kw) -> StubSystem:
    """The world's `stub` as a system that needs the GPU, which is free and not lost (nvidia-smi is not asked)."""
    w.patch.setattr(gpu, "gpu_busy", lambda: None)
    w.patch.setattr(gpu, "gpu_lost", lambda: None)
    return cli_world.use(w.patch, StubSystem(report=CLEAN, needs_gpu=True, **kw))


def test_a_repeat_runs_the_files_cells_at_jobs_and_a_system_that_needs_the_gpu_one_at_a_time_and_says_so_in_its_dry_run_before_it_builds(w, capsys):
    """The pilot's way (plan 8.1): `files` three at once and `mm` one. With a system that cannot report its queue the working clock refuses several
    execs, so the repeat is not refused only because the GPU system is held to one: that is the rule, not the refusal, and it comes before the build."""
    system = on_gpu(w, reports_queue=False)
    stub = Stub(pause=0.05)
    assert w.main(*args_of(w, "--dry-run"), agents=agents_of(stub)) == 0
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    for agent in ("claude", "codex"):
        assert any(line.startswith(f"{agent} stub/all {N} to run of {N} (1 at once) median ") for line in lines), agent
        assert any(line.startswith(f"{agent} files {N} to run of {N} median ") for line in lines), "three at once is --jobs, and says nothing"
    assert "execs at once: 3 (--jobs); 1 for those that need the GPU, stub/all (--gpu-jobs); a system may allow fewer" in lines
    assert not any(line.startswith("warning") for line in lines)
    assert w.main(*args_of(w), agents=agents_of(stub)) == 0, "not refused on the working clock: nothing runs beside another on the system that cannot say its queue"
    capsys.readouterr()
    assert (stub.at_once("stub-all"), stub.at_once("files")) == (1, 3), "what the run did: the GPU system one at a time, `files` three"
    assert {e[3] for e in system.kinds("max_jobs") if e[1] is None} == {1}, "the system was asked for one at a time"


def test_a_repeat_may_ask_for_more_gpu_execs_with_gpu_jobs_which_the_working_clock_refuses_after_the_build_without_offering_the_wall_clock(w, capsys):
    system = on_gpu(w, reports_queue=False)
    stub = Stub(pause=0.05)
    assert w.main(*args_of(w, "--gpu-jobs", "2"), agents=agents_of(stub)) == 1
    err = capsys.readouterr().err
    assert "stub/all: 2 execs at once on the working clock" in err and "one exec at a time (--gpu-jobs 1)" in err and "--clock" not in err
    assert stub.execs == []
    new = repeated(w)
    # the owner asks for the wall clock as well: it goes through, with the warning, at two at once
    assert w.main(*args_of(w, "--into", str(new.folder), "--gpu-jobs", "2", "--clock", "wall"), agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert "warning: stub/all: 2 execs at once on the GPU" in out
    assert (stub.at_once("stub-all"), stub.at_once("files")) == (2, 3)
    assert {e[3] for e in system.kinds("max_jobs") if e[1] is None} >= {2}
