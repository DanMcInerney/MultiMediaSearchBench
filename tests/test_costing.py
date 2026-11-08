"""What a run says it will cost before it starts (DESIGN 8.2, plan 6.1): the medians (and means) of what each cell took in earlier runs, the
cells with no history at EXEC_MIN said so, and `--dry-run` printing all that and starting nothing. The expected numbers are worked out by hand
from the results the tests write; `tests/test_runner.py` and `tests/test_cli.py` keep the estimate with no history as it was.

The standard of the run: `world` has the toy scenario (19 questions, a run of seed 7 at scale tiny) and the stand-in system `stub`, whose
variant `all` reads everything and allows as many execs at once as are asked."""

from __future__ import annotations

import re
import types

import cli_world
import pytest
from cli_world import build_args, world  # noqa: F401
from stub_system import Stub, StubSystem, tiny_questions

from mmsb import scenarios
from mmsb.core import costing, dryrun, fs, paths
from mmsb.core.agents import claude, codex
from mmsb.core.config import Config
from mmsb.core.costing import BLIND_MIN, EXEC_MIN, Cell, Figures, Typical, earlier_runs, forecast, history_of, typical_of
from mmsb.core.runfolder import RunFolder

pytestmark = pytest.mark.xdist_group("cli")
N = len(tiny_questions())
FILES, STUB, BLIND = Config("files", "files", None, True), Config("stub/all", "stub", "all", False), Config("blind", None, None, False)


def result(qid: str, seconds: float, *, agent: str = "claude", problem: str | None = None, gpu: float = 0.0, tokens: int | None = None,
           usd: float | None = None) -> dict:
    """The fields of a result that the estimate reads."""
    return {"qid": qid, "agent": agent, "passed": problem is None, "problem": problem, "seconds": seconds, "work": {"seconds": gpu},
            "tokens": None if tokens is None else {"input": tokens, "output": 0, "cache_read": 0, "cache_write": 0}, "cost_usd": usd}


def earlier(folder, cells: dict[tuple[str, str], list[dict]], scenario: str = "toy", scale: str = "tiny"):
    """A run folder of earlier results, `cells` by (agent, config dir); run.json says only what `earlier_runs` reads."""
    fs.write_json(folder / "run.json", {"scenario": scenario, "scale": scale})
    for (agent, cell), results in cells.items():
        for r in results:
            fs.write_json(folder / "results" / agent / cell / f"{r['qid']}.json", r)
    return folder


def test_the_typical_exec_of_a_cell_is_the_median_and_the_mean_of_its_done_results_and_a_setup_one_is_not_done(tmp_path):
    seconds = [10.0, 20.0, 30.0, 100.0]
    done = [result(f"q{i}", s, gpu=g, tokens=t, usd=u) for i, (s, g, t, u) in enumerate(zip(seconds, (0, 0, 4, 12), (1000, 2000, 3000, 6000),
                                                                                              (0.1, 0.3, 0.5, 0.9)))]
    done.append(result("q4", 900.0, problem="timeout: no answer in 900 s"))  # a timeout is a wrong answer: it ran
    t = typical_of([*done[:4], result("q9", 1.0, problem="setup: no login")])
    # by hand: the seconds 10, 20, 30, 100 have the median 25 and the mean 40; the GPU seconds 0, 0, 4, 12 the median 2 and the mean 4; the
    # tokens 1000, 2000, 3000, 6000 the median 2500 and the mean 3000; the dollars .1 .3 .5 .9 the median .4 and the mean .45
    assert t == Typical(4, Figures(25.0, 2.0, 2500.0, 0.4), Figures(40.0, 4.0, 3000.0, 0.45))
    with_timeout = typical_of(done)
    assert (with_timeout.n, with_timeout.median.seconds, with_timeout.mean.seconds) == (5, 30.0, 212.0), "a timeout counts: 900 s was spent"
    assert typical_of([result("q9", 1.0, problem="setup: x"), result("q8", 1.0, problem="error: x")]) is None, "none ran to an answer"
    # no figure where no result gives one (Codex reports no dollars)
    bare = typical_of([result("q0", 5.0)])
    assert bare == Typical(1, Figures(5.0, 0.0, None, None), Figures(5.0, 0.0, None, None))


def test_history_is_read_by_agent_and_configuration_from_result_files_and_the_other_lists_in_the_folder_are_not_results(tmp_path):
    folder = earlier(tmp_path / "old", {("claude", "files"): [result("a", 10.0), result("b", 30.0)], ("codex", "files"): [result("a", 7.0, agent="codex")],
                                        ("claude", "stub-all"): [result("a", 5.0)]})
    fs.write_json(folder / "results" / "claude" / "stub-all" / "ambient-labels.json", [{"label": "x"}])  # a session's named list
    fs.write_json(folder / "results" / "claude" / "files" / "stray.json", result("other", 99.0))  # (a file not named for the qid it holds)
    fs.write_json(folder / "results" / "retrieval" / "stub-all" / "q.json", result("q", 400.0))
    second = earlier(tmp_path / "old2", {("claude", "files"): [result("c", 50.0)]})
    found = history_of([folder, second])
    assert set(found) == {("claude", "files"), ("codex", "files"), ("claude", "stub-all")}
    assert (found["claude", "files"].n, found["claude", "files"].median.seconds) == (3, 30.0), "the results of both runs: 10, 30, 50"
    assert found["codex", "files"].median.seconds == 7.0 and found["claude", "stub-all"].n == 1
    assert history_of([tmp_path / "nowhere"]) == {} and history_of([]) == {}


def test_earlier_runs_are_those_of_the_same_scenario_and_scale_in_the_runs_folder(tmp_path):
    earlier(tmp_path / "a", {}, "toy", "tiny")
    earlier(tmp_path / "b", {}, "toy", "pilot")
    earlier(tmp_path / "c", {}, "v7_screen", "tiny")
    earlier(tmp_path / "d", {}, "toy", "tiny")
    (tmp_path / "e").mkdir()  # not a run
    fs.write_json(tmp_path / "f.json", {"scenario": "toy", "scale": "tiny"})
    fs.write_json((tmp_path / "g").joinpath("run.json"), {"seed": 1, "scale": "tiny"})  # made before mmsb: names no scenario
    assert earlier_runs(tmp_path, "toy", "tiny") == [tmp_path / "a", tmp_path / "d"]
    assert earlier_runs(tmp_path / "nowhere", "toy", "tiny") == []


def test_a_history_where_every_exec_took_100_s_gives_100_times_the_cases_in_session_seconds_and_the_wall_time_over_the_execs_at_once():
    typical = typical_of([result(f"q{i}", 100.0, gpu=2.0, tokens=500, usd=0.25) for i in range(5)])
    cells = [Cell("claude", FILES, N, N, 3, typical), Cell("claude", STUB, N, N, 1, typical)]
    f = forecast(cells)
    assert f.execs == 2 * N and f.guessed == ()
    for totals in (f.median, f.mean):  # (when every exec is the same, the median is the mean)
        assert totals.session_s == 100 * 2 * N, "100 s an exec"
        assert totals.wall_s == 100 * N / 3 + 100 * N / 1, "the files cell three at once, the stub one at a time"
        assert (totals.gpu_s, totals.tokens, totals.usd) == (2 * 2 * N, 500 * 2 * N, 0.25 * 2 * N)
    # a cell with nothing left to run adds nothing
    assert forecast([Cell("claude", FILES, 0, N, 3, typical)]).median == costing.Totals()


def test_a_few_long_execs_make_the_mean_larger_than_the_median_and_both_are_told():
    typical = typical_of([result("a", 1.0, gpu=0.0), result("b", 1.0, gpu=0.0), result("c", 1.0, gpu=0.0), result("d", 97.0, gpu=100.0)])
    f = forecast([Cell("claude", STUB, 8, 8, 2, typical)])
    # by hand: the median second is 1 and the mean 25; the GPU median 0 and the mean 25; 8 execs, two at once
    assert (f.median.session_s, f.median.wall_s, f.median.gpu_s) == (8.0, 4.0, 0.0)
    assert (f.mean.session_s, f.mean.wall_s, f.mean.gpu_s) == (200.0, 100.0, 200.0)


def test_a_cell_with_no_history_is_counted_at_exec_min_and_blind_at_its_own_figure_and_is_named():
    typical = typical_of([result("a", 100.0, tokens=1000)])
    cells = [Cell("claude", FILES, 6, 6, 3, typical), Cell("claude", STUB, 6, 6, 3, None), Cell("codex", BLIND, 6, 6, 3, None)]
    f = forecast(cells)
    assert f.guessed == ("claude stub/all", "codex blind")
    # by hand: 6 * 100 + 6 * 150 (EXEC_MIN 2.5 min) + 6 * 60 (BLIND_MIN 1 min) session seconds, and nothing else of the cells with no history
    assert (EXEC_MIN, BLIND_MIN) == (2.5, 1.0)
    assert f.median.session_s == f.mean.session_s == 600 + 900 + 360 and f.median.wall_s == (600 + 900 + 360) / 3
    assert f.median.tokens == 6000


def run_history(w, tmp_path, *, seconds: float = 100.0):
    """`paths.RUNS` is a folder with one earlier run of the toy scenario at scale tiny, where claude's files took `seconds` an exec and its
    stub/all took half of that with 5 GPU seconds; it is where `mmsb run` looks for history."""
    folder = tmp_path / "hist"
    qids = [q.qid for q in tiny_questions()]
    earlier(folder / "20261101-0000-v7-tiny-s7", {("claude", "files"): [result(q, seconds, usd=0.5) for q in qids],
                                                  ("claude", "stub-all"): [result(q, seconds / 2, gpu=5.0) for q in qids]})
    w.patch.setattr(paths, "RUNS", folder)
    return folder


def no_agents_installed(patch) -> None:
    patch.setattr(dryrun, "shutil", types.SimpleNamespace(which=lambda name: None))
    patch.setattr(codex, "models_of", lambda command: {"models": [
        {"slug": codex.model(), "supported_reasoning_levels": [{"effort": codex.effort()}, {"effort": "medium"}]}]})


def test_a_dry_run_says_what_the_run_costs_from_the_medians_of_earlier_runs_calls_no_agent_and_leaves_only_its_own_exec_folder(world, tmp_path, capsys):
    run = cli_world.built_run(world)
    run_history(world, tmp_path)
    no_agents_installed(world.patch)
    stub = Stub()
    before = {p for p in world.root.rglob("*")}
    assert world.main("run", str(run.folder), "--agents", "claude", "--dry-run", agents=cli_world.agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert stub.execs == [] and not world.system.kinds("open"), "no agent is called, no session is opened"
    made = {p for p in world.root.rglob("*")} - before
    own = run.folder / "dry-run"
    assert all(p == own or own in p.parents or p == run.slot(0) for p in made),         f"only the first execs' folders in the run's own dry-run folder, and the working folder they are prepared for: {sorted(made)}"
    assert "the data and the stores it needs:" in out and f"data {run.data}: there" in out and "store of files: made" in out
    lines = [" ".join(line.split()) for line in out.splitlines()]
    assert f"claude files {N} to run of {N} median 100 s, $0.50 ({N} earlier)" in lines
    assert f"claude stub/all {N} to run of {N} median 50 s, 5 GPU s ({N} earlier)" in lines
    # by hand: 19 execs at 100 s and 19 at 50 s: 2,850 session seconds; over 3 jobs 633 s and 317 s, 950 s: 0.3 h; 19 * 5 = 95 GPU seconds; 19 * 0.5 dollars
    assert f"{2 * N} execs to run over 3 jobs" in out
    assert "at the medians, the typical exec: about 0.3 h, 2,850 session seconds, 95 GPU seconds, 0.0M tokens, $9.50 at list" in out
    assert "at the means, what those runs took in all: about 0.3 h, 2,850 session seconds, 95 GPU seconds, 0.0M tokens, $9.50 at list" in out
    assert "== claude files, question" in out and "nothing was run" in out


def test_a_cell_with_no_history_is_said_so_by_the_dry_run_and_a_run_with_none_at_all_keeps_the_old_estimate(world, tmp_path, capsys):
    run = cli_world.built_run(world)
    run_history(world, tmp_path)
    no_agents_installed(world.patch)
    assert world.main("run", str(run.folder), "--agents", "claude,codex", "--configs", "files", "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"codex files {N} to run of {N} no earlier exec: 150 s assumed" in [" ".join(line.split()) for line in out.splitlines()]
    assert "no earlier exec for codex files: counted at 2.5 min an exec (1 blind), with no GPU seconds, tokens or dollars" in out
    # by hand: claude's 19 at 100 s and codex's 19 at 150 s over 3 jobs: 1,900 + 2,850 s / 3 = 1,583 s = 0.4 h
    assert "at the medians, the typical exec: about 0.4 h, 4,750 session seconds" in out
    # nothing earlier at all: the line the tests of the estimate know, and why it is that
    world.patch.setattr(paths, "RUNS", tmp_path / "empty")
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--dry-run") == 0
    assert f"{N} execs to run, about {N * EXEC_MIN / 3 / 60:.1f} h at 2.5 min an exec over 3 jobs (no earlier exec of these cells to take a median from)" in \
        capsys.readouterr().out


def test_a_run_that_is_resumed_takes_its_medians_from_its_own_results_too(world, tmp_path, capsys):
    run = cli_world.built_run(world)
    world.patch.setattr(paths, "RUNS", tmp_path / "empty")
    world.patch.setattr(claude, "check_login", lambda: None)
    no_agents_installed(world.patch)
    first = tiny_questions()[0].qid
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--only", first, agents=cli_world.agents_of(Stub())) == 0
    capsys.readouterr()
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"{N - 1} to run of {N}  median 2 s" in out and "(1 earlier)" in out, "the one done exec took the stub's 1.5 s (told as 2 s)"
    assert (run.folder / "results" / "claude" / "files" / f"{first}.json").exists()


def test_the_dumps_data_and_stores_a_run_needs_are_said_and_a_lost_data_folder_is_to_be_written_again(world, capsys):
    run = cli_world.built_run(world)
    meta = fs.read_json(run.folder / "run.json")
    capsys.readouterr()
    costing.show_builds(run, ["stub"])
    out = capsys.readouterr().out
    assert out.splitlines() == [f"  data {run.data}: there", "  store of stub: made (0 s to ingest)"]
    fs.remove_tree(run.data)
    costing.show_builds(run, [])
    assert capsys.readouterr().out == (f"  data {run.data}: gone from the temp folder; it is written again from the recipe, about "
                                       f"{meta['build_s'] / 60:.0f} min, from earlier runs' build time\n")
    fs.write_json(run.folder / "run.json", {k: v for k, v in meta.items() if k != "build_s"})
    costing.show_builds(run, [])
    assert "no earlier build time was recorded to say how long" in capsys.readouterr().out


def test_the_dump_and_stores_of_a_run_to_be_built_are_told_with_the_medians_of_earlier_builds(world, tmp_path, capsys):
    run = cli_world.built_run(world)
    olds = []
    for name, build, ingest in (("one", 600.0, 80.0), ("two", 800.0, 100.0), ("three", 1000.0, 300.0)):
        folder = earlier(tmp_path / name, {})
        fs.write_json(folder / "run.json", {"scenario": "toy", "scale": "tiny", "build_s": build})
        fs.write_json(folder / "stores" / "stub" / "ingest.json", {"seconds": ingest})
        olds.append(folder)
    stub, empty = {"stub": world.system.describe("all")}, (tmp_path / "temp-to-fill", tmp_path / "cache-to-fill")  # (a temp folder and a cache with nothing in them)
    costing.show_new_builds(run, olds, stub, *empty)
    out = capsys.readouterr().out
    assert "dump: written in the temp folder from the recipe (toy, tiny, seed 7, 2 days), about 13 min, from earlier runs' build time" in out
    assert "store of stub: ingested from it, about 2 min, from earlier runs' ingest time" in out
    costing.show_new_builds(run, [], stub, *empty)
    assert out.count("about") == 2 and capsys.readouterr().out.count("no earlier") == 2
    assert not empty[0].exists() and not empty[1].exists(), "it only looks"
    costing.show_new_builds(run, olds, stub, run.temp, world.cache)  # the run's own dump and store are in the cache: nothing is built
    out = capsys.readouterr().out
    assert f"  dump {run.data.name}: cached (toy, tiny, seed 7, 2 days): writing it took " in out and "so it is not written again" in out
    assert "  store of stub: cached: ingesting it took 0 s, so it is not ingested again" in out and "about" not in out


def test_the_selection_flags_are_given_to_the_run_said_step_by_step_and_a_selection_that_picks_nothing_is_refused(world, tmp_path, capsys):
    run = cli_world.built_run(world)
    stub = Stub()
    world.patch.setattr(claude, "check_login", lambda: None)
    absent = [q.qid for q in tiny_questions() if q.kind == "absent"]
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--kinds", "absent", "--tag", "absent=true",
                      agents=cli_world.agents_of(stub)) == 0
    out = capsys.readouterr().out
    assert "selection: the questions: 19, then --kinds absent: 5, then --tag absent=true: 5" in out
    assert sorted(stub.qids()) == sorted(absent) and f"{len(absent)} to run of {len(absent)}" in out
    no_agents_installed(world.patch)
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", "--rung", "tiny", "--limit", "3", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "selection: the questions: 19, then --tag rung=tiny: 19, then --limit 3: 3" in out
    assert re.search(r"claude\s+files\s+\d to run of 3\b", out)
    assert world.main("run", str(run.folder), "--agents", "claude", "--kinds", "count", "--modalities", "video") == 1
    assert "--modalities names video, which no question" in capsys.readouterr().err
    for bad in (["--tag", "absent"], ["--tag", "rung=tiny", "--rung", "huge"], ["--limit", "0"], ["--tag", "a=1", "--tag", "a=2"]):
        capsys.readouterr()
        with pytest.raises(SystemExit):
            world.main("run", str(run.folder), "--agents", "claude", *bad)
        assert capsys.readouterr().err.strip().splitlines()[-1].startswith("mmsb: error: ")
    # the selection is `--only` of the qids it picks, so a later command without it asks the rest
    stub.execs.clear()
    assert world.main("run", str(run.folder), "--agents", "claude", "--configs", "files", agents=cli_world.agents_of(stub)) == 0
    assert len(stub.execs) == N - len(absent)
