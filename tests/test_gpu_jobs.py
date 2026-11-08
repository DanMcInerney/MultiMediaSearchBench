"""How many execs of a cell run at once (core/atonce.py; the owner's rule of 5 November 2026, after three execs of one decision model at once crashed
the laptop's driver on 2 November): a configuration whose system needs the GPU runs one exec at a time unless `--gpu-jobs` asks for more, and `--jobs`
and `--clock` do not; the others run `--jobs`. Through `main`, over the stand-in system of tests/stub_system.py (which can be made to need the GPU, with
a GPU that is free and not lost, so nvidia-smi is not asked) and stand-in agents. The rule is the same for the runner, every dry run and its estimate,
the refusal of the working clock, `--side-by-side` and `clock-check`; the repeat's own tests are in test_repeat.py, the check's in test_clock_check.py.

The expected values are not the code's: the table of the rule is written here from the owner's words, what a run did is read from the times the
stand-in agent was called at (`Stub.at_once`), and the dry run's estimate is worked out from those counts by hand (a cell's execs at 2.5 minutes each over
the execs at once it was seen to run)."""

from __future__ import annotations

import itertools
import re
import shutil

import pytest
from cli_world import agents_of, built_run, world  # noqa: F401
from stub_system import Stub, StubSession, StubSystem, tiny_questions

from mmsb import cli
from mmsb.core import fs, gpu
from mmsb.core.agents import claude
from mmsb.core.atonce import GPU_JOBS, JOBS, asked, execs_at_once, flag_of, gpu_warning
from mmsb.core.contract import Case

pytestmark = pytest.mark.xdist_group("cli")
QUESTIONS = tiny_questions()
N = len(QUESTIONS)
RUN = ("--agents", "claude", "--configs", "files,stub/all")
EXEC_MIN = 2.5  # the minutes an exec is counted at in an estimate with no earlier exec to take a median from (costing.EXEC_MIN, here written out)


@pytest.fixture
def run(world):  # noqa: F811
    """The built run of the toy week, asked in the `line` format the stand-in answers in; Claude's login is not asked of the real CLI."""
    world.patch.setattr(claude, "check_login", lambda: None)
    return built_run(world).rewritten(answer_format="line")


def on_gpu(world, **kw) -> StubSystem:  # noqa: F811
    """The world's `stub` as a system that needs the GPU, which is free and not lost."""
    world.patch.setattr(gpu, "gpu_busy", lambda: None)
    world.patch.setattr(gpu, "gpu_lost", lambda: None)
    world.system.needs_gpu = True
    for name, value in kw.items():
        setattr(world.system, name, value)
    return world.system


def fresh(run) -> None:
    """The run as it was before any exec: nothing asked, so that the same command is a whole run again."""
    shutil.rmtree(run.folder / "results", ignore_errors=True)
    shutil.rmtree(run.folder / "execs", ignore_errors=True)


# -- the rule --------------------------------------------------------------------------------------------------------------------------

def test_the_rule_is_jobs_for_a_system_with_no_gpu_gpu_jobs_for_one_that_needs_it_and_the_system_s_own_limit_comes_on_top():
    case = Case("q", 0.0, None, (), 0.0, 1.0, "q", 0.0, -1, None, None)
    for needs, jobs, gpu_jobs, limit in itertools.product((False, True), range(1, 5), range(1, 5), (None, 1, 2)):
        description = StubSystem(needs_gpu=needs).describe("all")
        session = StubSession(StubSystem(job_limit=limit), None)
        want = gpu_jobs if needs else jobs  # the owner's rule, written out
        assert asked(description, jobs, gpu_jobs) == want
        assert execs_at_once(description, session, [case], jobs, gpu_jobs) == min(want, limit or 4), (needs, jobs, gpu_jobs, limit)
        assert flag_of(description) == ("--gpu-jobs" if needs else "--jobs")
    assert asked(None, 3, 1) == 3 and flag_of(None) == "--jobs", "a configuration with no system (blind) has all its jobs"
    assert (JOBS, GPU_JOBS) == (3, 1), "by default: three execs at once, and one on the GPU"


def test_the_warning_is_for_several_execs_on_the_gpu_and_nothing_else():
    gpu_system, plain = StubSystem(needs_gpu=True).describe("all"), StubSystem().describe("all")
    assert gpu_warning("mm/x", gpu_system, 1) is None and gpu_warning("files", plain, 3) is None and gpu_warning("blind", None, 3) is None
    assert gpu_warning("mm/x", gpu_system, 2).startswith("warning: mm/x: 2 execs at once on the GPU")


def test_both_options_are_one_to_four_on_run_repeat_and_clock_check_and_default_to_three_and_one(run, world, capsys):  # noqa: F811
    parser = cli.parser_of()
    for command in (["run", "x"], ["repeat", "x"], ["clock-check", "x", "--config", "stub/all"]):
        args = parser.parse_args(command)
        assert (args.jobs, args.gpu_jobs) == (3, 1), command
        for flag in ("--jobs", "--gpu-jobs"):
            for bad in ("0", "5"):
                with pytest.raises(SystemExit):
                    world.main(command[0], str(run.folder), *command[2:], flag, bad)
                assert f"{flag} is 1 to 4" in capsys.readouterr().err


# -- a run -----------------------------------------------------------------------------------------------------------------------

def test_a_run_gives_a_system_that_needs_the_gpu_one_exec_at_a_time_and_the_others_jobs_and_gpu_jobs_is_the_only_way_to_more(world, run, capsys):  # noqa: F811
    system = on_gpu(world)
    # (--jobs, --gpu-jobs) -> what ran at once in the GPU cell and in the `files` cell
    for flags, seen in ((("--jobs", "3"), (1, 3)), (("--jobs", "2"), (1, 2)), (("--jobs", "4", "--gpu-jobs", "1"), (1, 4)), (("--gpu-jobs", "2"), (2, 3)),
                        (("--jobs", "1", "--gpu-jobs", "2"), (2, 1)), (("--jobs", "2", "--gpu-jobs", "3"), (3, 2)), ((), (1, 3))):
        fresh(run)
        system.events.clear()
        stub = Stub(pause=0.03)
        assert world.main("run", str(run.folder), *RUN, *flags, agents=agents_of(stub)) == 0, flags
        capsys.readouterr()
        assert (stub.at_once("stub-all"), stub.at_once("files")) == seen, flags
        assert {e[3] for e in system.kinds("max_jobs") if e[1] is None} == {seen[0]}, "the system is asked for what the rule gives, once for each agent"
    # (`--clock wall` does not raise it either)
    fresh(run)
    stub = Stub(pause=0.03)
    assert world.main("run", str(run.folder), *RUN, "--clock", "wall", agents=agents_of(stub)) == 0
    assert (stub.at_once("stub-all"), stub.at_once("files")) == (1, 3)


def test_a_dry_run_prints_the_execs_at_once_and_the_wall_estimate_that_the_run_then_does(world, run, capsys):  # noqa: F811
    on_gpu(world)
    for jobs, gpu_jobs in ((3, 1), (2, 1), (3, 2), (1, 2), (4, 3)):
        flags = ("--jobs", str(jobs), "--gpu-jobs", str(gpu_jobs))
        fresh(run)
        stub = Stub(pause=0.03)
        assert world.main("run", str(run.folder), *RUN, *flags, agents=agents_of(stub)) == 0
        seen_gpu, seen_files = stub.at_once("stub-all"), stub.at_once("files")
        assert (seen_gpu, seen_files) == (gpu_jobs, jobs), flags
        capsys.readouterr()
        fresh(run)
        assert world.main("run", str(run.folder), *RUN, *flags, "--dry-run", agents=agents_of(Stub())) == 0
        out = capsys.readouterr().out
        # each cell says how many it runs at once when that is not --jobs; the estimate counts the cells at the counts the run was seen to run
        files_line = next(line for line in out.splitlines() if re.match(r"\s+claude\s+files\s", line))
        gpu_line = next(line for line in out.splitlines() if re.match(r"\s+claude\s+stub/all\s", line))
        assert "at once)" not in files_line and ", all the questions" in out and "the all questions" not in out
        assert (f"({gpu_jobs} at once)" in gpu_line) == (gpu_jobs != jobs), (flags, gpu_line)
        hours = N * EXEC_MIN / seen_files / 60 + N * EXEC_MIN / seen_gpu / 60
        assert f"{2 * N} execs to run, about {hours:.1f} h at 2.5 min an exec over {jobs} jobs" in out, (flags, hours)
        assert f"execs at once: {jobs} (--jobs); {gpu_jobs} for those that need the GPU, stub/all (--gpu-jobs); a system may allow fewer" in out
        assert ("warning: stub/all: " in out) == (gpu_jobs > 1)


def test_several_execs_on_the_gpu_are_warned_of_in_a_dry_run_and_in_the_run_and_one_exec_at_a_time_is_not(world, run, capsys):  # noqa: F811
    on_gpu(world)
    for flags in ((), ("--gpu-jobs", "1"), ("--jobs", "4")):
        fresh(run)
        assert world.main("run", str(run.folder), *RUN, *flags, agents=agents_of(Stub())) == 0
        assert "warning" not in capsys.readouterr().out, flags
    for flags in (("--dry-run",), ()):
        fresh(run)
        assert world.main("run", str(run.folder), *RUN, "--gpu-jobs", "2", *flags, agents=agents_of(Stub())) == 0
        out = capsys.readouterr().out
        assert out.count("warning: stub/all: 2 execs at once on the GPU, because --gpu-jobs asked for them") == 1, flags
    # a system that needs no GPU is never warned of, at any --jobs
    world.system.needs_gpu = False
    fresh(run)
    assert world.main("run", str(run.folder), *RUN, "--jobs", "4", agents=agents_of(Stub())) == 0
    assert "warning" not in capsys.readouterr().out


def test_the_wall_clock_lets_several_gpu_execs_through_when_the_owner_asked_for_them_and_the_results_say_it_was_the_wall(world, run, capsys):  # noqa: F811
    on_gpu(world, reports_queue=False)
    for flags in (("--gpu-jobs", "2"), ("--gpu-jobs", "2", "--dry-run")):  # the working clock refuses them, in a dry run too
        assert world.main("run", str(run.folder), *RUN, *flags, agents=agents_of(Stub())) == 1
        assert "stub/all: 2 execs at once on the working clock" in capsys.readouterr().err
    assert not (run.folder / "results").exists()
    stub = Stub(pause=0.03)
    assert world.main("run", str(run.folder), *RUN, "--gpu-jobs", "2", "--clock", "wall", agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert "warning: stub/all: 2 execs at once on the GPU" in out and stub.at_once("stub-all") == 2
    assert {fs.read_json(f)["clock"] for f in (run.folder / "results" / "claude" / "stub-all").glob("*.json")} == {"wall"}
    # and at its default the GPU system, one at a time, needs no queue figure and no wall clock
    fresh(run)
    stub = Stub(pause=0.03)
    assert world.main("run", str(run.folder), *RUN, agents=agents_of(stub)) == 0
    assert stub.at_once("stub-all") == 1
    assert {fs.read_json(f)["clock"] for f in (run.folder / "results" / "claude" / "stub-all").glob("*.json")} == {"wall"}, "(it cannot say its queue)"


def test_no_refusal_offers_the_wall_clock_and_each_names_one_exec_at_a_time_by_the_option_of_its_system(world, run, capsys):  # noqa: F811
    refusals = []
    for needs_gpu, flags, option in ((False, ("--jobs", "3"), "--jobs"), (True, ("--gpu-jobs", "2"), "--gpu-jobs")):
        world.system.needs_gpu = needs_gpu
        world.system.reports_queue = False
        world.patch.setattr(gpu, "gpu_busy", lambda: None)
        world.patch.setattr(gpu, "gpu_lost", lambda: None)
        for more in ((), ("--dry-run",)):
            assert world.main("run", str(run.folder), *RUN, *flags, *more, agents=agents_of(Stub())) == 1
            err = capsys.readouterr().err
            assert f"one exec at a time ({option} 1)" in err, err
            refusals.append(err)
        for check in (("--gpu-jobs", "2"), ("--gpu-jobs", "1")) if needs_gpu else (("--jobs", "3"), ("--jobs", "1")):
            assert world.main("clock-check", str(run.folder), "--config", "stub/all", *check, agents=agents_of(Stub())) == 1
            refusals.append(capsys.readouterr().err)
    assert len(refusals) == 8 and all(r.startswith("error: ") for r in refusals)
    assert not [r for r in refusals if "--clock" in r or "wall" in r], "a refusal that offers the wall clock offers to keep the several execs"


# -- side by side ----------------------------------------------------------------------------------------------------------------------

def test_side_by_side_keeps_its_one_gpu_cell_and_inside_it_the_same_rule_holds(world, run, capsys):  # noqa: F811
    on_gpu(world)
    for flags, seen in (((), (1, 3)), (("--gpu-jobs", "2"), (2, 3))):
        fresh(run)
        stub = Stub(pause=0.03)
        assert world.main("run", str(run.folder), *RUN, "--side-by-side", *flags, agents=agents_of(stub)) == 0, flags
        capsys.readouterr()
        assert (stub.at_once("stub-all"), stub.at_once("files")) == seen, flags
