"""`mmsb clock-check` and the `--clock` option of `mmsb run` (plan 8.2, 6.5, DESIGN 8.4; written for this repository, U14), through `main`, over the stand-in system of tests/stub_system.py and stand-in
agents: the cases are drawn from those the system serves beside another, each is asked alone and then queued, in two folders under the run, and the answers,
tokens and charged seconds are printed side by side. It refuses before any agent starts when the system cannot report its queue (decision D3) or would
not run the cases beside each other, and it asks the system whether it is ready first. Over mm itself: tests/mm/test_clock_mm.py.

The expected values are what the test planted: the queue the stand-in system reports for each case (`StubSystem.waits`), and an agent that takes 2 s of its
own and the queue besides, so that its charged seconds are 2 whether alone or queued and its wall seconds are not; and the cases that hold the system
alone, which are those the plan's `HEAVY_S` names (300 of the system's seconds), worked out here from the questions' own ranges."""

from __future__ import annotations

import pytest
from agent_spec import clocked
from cli_world import agents_of, built_run, world  # noqa: F401
from stub_system import Stub, peak, tiny_questions

from mmsb.core import fs, gpu
from mmsb.core.agents import claude
from mmsb.core.agents.common import reached
from mmsb.core.runfolder import is_done, load_result

QUESTIONS = tiny_questions()
BASE_S = 2.0
CHECK = ("clock-check", "--config", "stub/all")


class Takes(Stub):
    """A stub that answers as `Stub` does and takes BASE_S of its own and the queue besides, as an exec that waits for the GPU does."""

    def __call__(self, spec):
        return {**super().__call__(spec), **clocked(spec, BASE_S + (spec.queue_s() or 0.0))}


@pytest.fixture
def run(world):  # noqa: F811
    """The built run of the toy week, asked in the `line` format that the stand-in answers in (as `mmsb run --answer-format line` would have kept
    in its run.json); Claude's login is not asked of the real CLI, and is counted (`world.logins`)."""
    world.logins = []
    world.patch.setattr(claude, "check_login", lambda: world.logins.append(1))
    return built_run(world).rewritten(answer_format="line")


def result_of(run, half: str, agent: str, qid: str) -> dict:
    return fs.read_json(run.folder / "clock-check" / half / "results" / agent / "stub-all" / f"{qid}.json")


def asked(run, half: str, agent: str) -> list[str]:
    return sorted(p.stem for p in (run.folder / "clock-check" / half / "results" / agent / "stub-all").glob("*.json"))


def table(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("  ", "selection"))]  # (the progress lines of the execs are not the table)


def test_each_case_is_asked_alone_and_then_queued_in_two_folders_and_the_answers_tokens_and_charged_seconds_are_printed_side_by_side(world, run, capsys):
    waits = {q.qid: 0.5 * (n % 3 + 1) for n, q in enumerate(QUESTIONS)}
    world.system.waits = waits
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), "--cases", "10", "--jobs", "3", agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert not (run.folder / "results").exists(), "the run's own folder is not asked: the halves are"
    for agent in ("claude", "codex"):
        qids = asked(run, "alone", agent)
        assert len(qids) == 10 and qids == asked(run, "queued", agent), "the same ten cases in both halves"
        for qid in qids:
            alone, queued = result_of(run, "alone", agent, qid), result_of(run, "queued", agent, qid)
            assert (alone["queue_s"], alone["working_s"], alone["seconds"]) == (0.0, BASE_S, BASE_S), "alone nothing waits"
            assert (queued["queue_s"], queued["working_s"], queued["seconds"]) == (waits[qid], BASE_S, BASE_S + waits[qid]), "queued, charged the same"
            assert alone["answer"] == queued["answer"] and alone["clock"] == queued["clock"] == "working"
        waited = sum(waits[qid] for qid in qids)
        assert f"{agent} stub/all: 10 cases, each asked alone and then queued" in out
        for want in ("the same answer in 10 of 10 cases", "tokens: 11,000 alone, 11,000 queued",
                     "charged seconds: 20.0 alone, 20.0 queued (queued / alone 1.00)",
                     f"wall seconds: 20.0 alone, {20 + waited:.1f} queued (queued / alone {(20 + waited) / 20:.2f}); {waited:.1f} s waited in the queue",
                     "clock: working"):
            assert f"  {want}" in out, want
        for qid in qids:  # a line for the case: its answer and tokens alone and queued, the seconds it waited and its wall seconds
            line = next(line for line in out.splitlines() if line.strip().startswith(qid))
            assert line.count("1,100") == 2 and line.split()[-2:] == [f"{waits[qid]:g}", f"{BASE_S + waits[qid]:g}"], line
    assert len(stub.execs) == 2 * 2 * 10 and world.logins == [1], "Claude's login was asked once, before any agent"
    assert "clock-check, alone: 1 exec at once" in out and "clock-check, queued: 3 execs at once" in out
    assert sorted({e[3] for e in world.system.kinds("max_jobs") if e[1] is None}) == [1, 3], "one half alone, one queued"


def test_the_cases_are_drawn_from_those_that_do_not_hold_the_system_alone(world, run):
    ranges = {q.qid: sum(b - a for a, b in q.cost_ranges) for q in QUESTIONS}
    world.system.rate = 400.0 / max(ranges.values())
    heavy = {qid for qid, seconds in ranges.items() if world.system.rate * seconds >= 300.0}  # HEAVY_S: the system serves these alone
    light = {q.qid for q in QUESTIONS} - heavy
    assert heavy and 10 < len(light) < len(QUESTIONS)
    assert world.main(*CHECK, str(run.folder), "--cases", "10", "--agents", "claude", agents=agents_of(Takes())) == 0
    got = asked(run, "alone", "claude")
    assert len(got) == 10 and set(got) <= light and not set(got) & heavy


def test_fewer_cases_than_asked_for_are_all_there_is(world, run, capsys):
    world.system.rate = 400.0 / max(sum(b - a for a, b in q.cost_ranges) for q in QUESTIONS)
    assert world.main(*CHECK, str(run.folder), "--cases", "50", "--agents", "claude", agents=agents_of(Takes())) == 0
    got = asked(run, "alone", "claude")
    assert 10 < len(got) < 50 and f"claude stub/all: {len(got)} cases" in capsys.readouterr().out


def test_a_system_that_holds_every_case_alone_has_nothing_to_queue(world, run, capsys):
    world.system.rate = 1000.0  # every range costs it more than HEAVY_S seconds
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), agents=agents_of(stub)) == 1
    assert "every case of stub/all holds its system alone" in capsys.readouterr().err and not stub.execs


def test_a_system_that_cannot_report_its_queue_is_refused_naming_decision_d3_before_any_agent_and_nothing_is_written(world, run, capsys):
    world.system.reports_queue = False
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), "--jobs", "3", agents=agents_of(stub)) == 1
    err = capsys.readouterr().err
    assert "decision D3" in err and "reports no queue time" in err and "stub/all" in err
    assert not stub.execs and not (run.folder / "clock-check").exists()
    assert world.system.kinds("open") == [], "no session was opened for it"


def test_a_system_that_would_run_the_cases_one_at_a_time_has_no_queue_and_is_refused_and_the_message_names_what_to_change(world, run, capsys):
    stub = Takes()
    world.system.job_limit = 1  # the system's own limit: no option raises it
    assert world.main(*CHECK, str(run.folder), "--jobs", "3", agents=agents_of(stub)) == 1
    err = capsys.readouterr().err
    assert "would run these cases 1 at a time though 3 were asked for" in err and "so nothing would queue" in err
    world.system.job_limit = None
    assert world.main(*CHECK, str(run.folder), "--jobs", "1", agents=agents_of(stub)) == 1
    err = capsys.readouterr().err
    assert "would run these cases one at a time (--jobs 1), so nothing would queue" in err and "Ask for several with --jobs 2 or more" in err
    assert "1 at a time, not 1" not in err
    assert not stub.execs and not (run.folder / "clock-check").exists()


def test_a_system_that_needs_the_gpu_is_checked_only_when_gpu_jobs_asks_for_several_and_the_queued_half_runs_that_many_with_a_warning(
        world, run, capsys):
    """The one command that puts several execs on the GPU on purpose, and only by the explicit option: `--jobs` does not raise it."""
    world.system.needs_gpu = True
    world.patch.setattr(gpu, "gpu_busy", lambda: None)  # (a GPU that is free and not lost: nvidia-smi is not asked)
    world.patch.setattr(gpu, "gpu_lost", lambda: None)
    world.system.waits = {q.qid: 0.5 for q in QUESTIONS}
    stub = Takes(pause=0.05)
    for flags in ((), ("--jobs", "3"), ("--gpu-jobs", "1")):
        assert world.main(*CHECK, str(run.folder), *flags, agents=agents_of(stub)) == 1, flags
        err = capsys.readouterr().err
        assert "would run these cases one at a time (--gpu-jobs 1), so nothing would queue" in err and "Ask for several with --gpu-jobs 2 or more" in err
    assert not stub.execs and not (run.folder / "clock-check").exists() and world.logins == [], "refused before any agent"
    assert world.main(*CHECK, str(run.folder), "--agents", "claude", "--jobs", "1", "--gpu-jobs", "2", agents=agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert "warning: stub/all: 2 execs at once on the GPU" in out
    assert "clock-check, alone: 1 exec at once" in out and "clock-check, queued: 2 execs at once" in out
    assert sorted({e[3] for e in world.system.kinds("max_jobs") if e[1] is None}) == [1, 2], "one half alone, one queued: --jobs did not count"
    spans = [(began, ended) for _, _, _, began, ended in stub.log]  # (in the order the execs ended: the alone half's ten, then the queued half's)
    assert len(spans) == 20 and (peak(spans[:10]), peak(spans[10:])) == (1, 2), "the alone half one at a time, the queued half two at once"


def test_a_clock_check_half_denies_the_folder_of_the_run_it_branched_from_and_every_folder_in_it(world, run):
    """The run's `questions.jsonl` holds the answers. A half's own folder is a folder of the run, so a rule that names only that folder leaves the run's
    own files, and the other half's results, open to an agent with its own tools when the run is not in a folder that is denied for other reasons."""
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), "--agents", "claude", agents=agents_of(stub)) == 0
    planted = [run.folder / "questions.jsonl", run.folder / "run.json", run.folder / "data-manifest.json",
               run.folder / "clock-check" / "alone" / "results" / "claude" / "stub-all" / f"{asked(run, 'alone', 'claude')[0]}.json"]
    assert all(path.is_file() for path in planted) and len(stub.execs) == 20
    for spec in stub.execs:
        for path in planted:
            assert reached(str(path), spec.deny) is not None, (spec.run_dir, path)
            assert reached(f"cat {path}", spec.deny) is not None, "and in a command"
        assert reached(str(run.temp / "elsewhere.txt"), spec.deny) is None, "the rule is the run's folders, not the whole machine"


def test_a_system_that_is_not_ready_is_refused_before_any_agent_starts(world, run, capsys):
    world.system.unready = "the weights are not downloaded"
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), agents=agents_of(stub)) == 1
    assert "stub/all cannot run now: all: the weights are not downloaded" in capsys.readouterr().err
    assert [e[0] for e in world.system.events if e[0] in ("check_ready", "open")] == ["check_ready"] and not stub.execs


def test_a_configuration_that_is_not_one_system_is_refused(world, run, capsys):
    assert world.main("clock-check", str(run.folder), "--config", "blind", agents=agents_of(Takes())) == 1
    assert "blind has no system" in capsys.readouterr().err
    for flags in (["--config", "stub/all,stub/narrow"], ["--config", "stub/all", "--cases", "0"], ["--config", "stub/all", "--jobs", "5"],
                  ["--config", "stub/all", "--agents", "gemini"]):
        with pytest.raises(SystemExit):
            world.main("clock-check", str(run.folder), *flags, agents=agents_of(Takes()))
        capsys.readouterr()


def test_a_second_command_asks_nothing_and_prints_the_same_table_and_a_cut_off_half_goes_on_from_where_it_was(world, run, capsys):
    stub = Takes()
    assert world.main(*CHECK, str(run.folder), "--agents", "claude", agents=agents_of(stub)) == 0
    first = capsys.readouterr().out
    done = len(stub.execs)
    assert world.main(*CHECK, str(run.folder), "--agents", "claude", agents=agents_of(stub)) == 0
    assert len(stub.execs) == done, "every case of both halves is done"
    assert table(first) == table(capsys.readouterr().out)
    gone = run.folder / "clock-check" / "queued" / "results" / "claude" / "stub-all" / f"{asked(run, 'queued', 'claude')[0]}.json"
    gone.unlink()
    assert world.main(*CHECK, str(run.folder), "--agents", "claude", agents=agents_of(stub)) == 0
    assert len(stub.execs) == done + 1 and is_done(load_result(gone))


def test_a_session_with_no_queue_at_all_is_a_session_whose_queue_is_nothing(world, run, capsys):
    """`PlainSession`'s answer, 0.0, which the stand-in system gives when nothing is planted: the working clock with nothing taken off, in both halves."""
    assert world.main(*CHECK, str(run.folder), "--agents", "codex", agents=agents_of(Takes())) == 0
    out = capsys.readouterr().out
    for qid in asked(run, "queued", "codex"):
        r = result_of(run, "queued", "codex", qid)
        assert (r["clock"], r["queue_s"], r["working_s"], r["seconds"]) == ("working", 0.0, BASE_S, BASE_S)
    assert "0.0 s waited in the queue" in out and "claude stub/all" not in out and world.logins == [], "no Claude: no login asked"


# -- the --clock option of `mmsb run` ----------------------------------------------------------------------------------------------------

def test_the_run_command_refuses_several_execs_on_the_working_clock_for_a_system_that_cannot_say_its_queue_before_any_agent_and_a_dry_run_says_so_too(
        world, run, capsys):
    world.system.reports_queue = False
    stub = Takes()
    first, second, third = (q.qid for q in QUESTIONS[:3])
    base = ["run", str(run.folder), "--configs", "stub/all", "--agents", "claude"]
    for flags in ((), ("--dry-run",)):
        assert world.main(*base, "--jobs", "3", *flags, agents=agents_of(stub)) == 1
        assert "stub/all: 3 execs at once on the working clock" in capsys.readouterr().err
    assert not stub.execs and not (run.folder / "results").exists() and world.logins == [], "refused before the login, the build of a store or an agent"
    # one exec at a time has no queue, and on the wall clock the number does not matter: each result says which clock it ran on
    assert world.main(*base, "--jobs", "1", "--only", first, agents=agents_of(stub)) == 0
    assert world.main(*base, "--jobs", "3", "--clock", "wall", "--only", second, agents=agents_of(stub)) == 0
    for qid in (first, second):
        r = fs.read_json(run.folder / "results" / "claude" / "stub-all" / f"{qid}.json")
        assert (r["clock"], r["queue_s"], r["working_s"], r["seconds"]) == ("wall", None, BASE_S, BASE_S), qid
    # a system that can say its queue is charged the wall less it
    world.system.reports_queue, world.system.waits = True, {third: 0.5}
    assert world.main(*base, "--jobs", "3", "--only", third, agents=agents_of(stub)) == 0
    r = fs.read_json(run.folder / "results" / "claude" / "stub-all" / f"{third}.json")
    assert (r["clock"], r["queue_s"], r["working_s"], r["seconds"]) == ("working", 0.5, BASE_S, BASE_S + 0.5)
    with pytest.raises(SystemExit):
        world.main(*base, "--clock", "monotonic", agents=agents_of(stub))
