"""`mmsb compare` and `discriminating` on the recorded pilot (plan 6.1, 8.1, decision D1). The recorded pilot's 110 results, converted once
(recorded/pilot-s1/converted), are the run to compare with itself, with a copy that has flips planted, and to pick the discriminating cases
of; the expected values are the recorded files' own and D1's: recorded/pilot-s1/report.json (written by the old code on 4 November 2026, its
`b` and `c` and its per-question appendix) and D1's counts of correct answers (Claude files 23, Claude mm 22, Codex files 27, Codex mm 23).

Like oracle 2, it opens the run through `RunFolder.open`, which makes the V7 plan, and the plan is local time: elsewhere than the pilot's time
zone this skips, and says so (v7_recorded.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from v7_recorded import needs_the_recorded_zone

from mmsb import scenarios
from mmsb.core import fs
from mmsb.core.compare import BAND, compare_cells, compare_runs, render_comparison
from mmsb.core.config import Config
from mmsb.core.runfolder import RunFolder
from mmsb.core.select import discriminating, separate

needs_the_recorded_zone()

RECORDED = Path(__file__).parent / "recorded" / "pilot-s1"
CONVERTED = RECORDED / "converted"
REPORT = json.loads((RECORDED / "report.json").read_text(encoding="utf-8"))  # the old code's, on the recorded results
FILES, MM = Config("files", "files", None, True), Config("mm/d1-3B", "mm", "d1-3B", False)
# D1 (plan 9.3): the correct answers of each cell in the recorded run, and how many questions it was asked
D1_COUNTS = {("claude", "files"): (23, 31), ("claude", "mm/d1-3B"): (22, 24), ("codex", "files"): (27, 31), ("codex", "mm/d1-3B"): (23, 24)}
CONFIGS = {"files": FILES, "mm/d1-3B": MM}


@pytest.fixture(scope="module")
def pilot(tmp_path_factory) -> RunFolder:
    folder = tmp_path_factory.mktemp("pilot") / "20261104-0314-pilot-s1"
    shutil.copytree(CONVERTED, folder)
    return RunFolder.open(folder, scenarios.get)


@pytest.fixture
def again(pilot, tmp_path) -> RunFolder:
    """A copy of the recorded run to plant flips in."""
    shutil.copytree(pilot.folder, tmp_path / "again")
    return RunFolder.open(tmp_path / "again", scenarios.get)


def flips_of(agent: str) -> list[str]:
    """The questions the recorded report's own appendix shows right in one arm and wrong in the other, for the paired ones (both arms done)."""
    return [row["qid"] for row in REPORT["appendix"][agent] if row["mm"] and row["files"]["passed"] != row["mm"]["d1-3B"]["passed"]]


def test_the_discriminating_cases_before_canaries_are_the_recorded_reports_b_plus_c_of_each_agent_and_by_the_same_questions(pilot):
    for agent, count in (("claude", 4), ("codex", 1)):
        report = REPORT["agents"][agent]["d1-3B"]["all"]
        assert report["b"] + report["c"] == count, "the recorded numbers: Claude 4, Codex 1"
        chosen = discriminating(pilot.questions(), [pilot], canaries=0, agents=(agent,))
        assert [q.qid for q in chosen] == flips_of(agent) and len(chosen) == count
        # with the canaries, 4 + 2 and 1 + 2: two cases that did not separate the configurations are kept
        with_canaries = discriminating(pilot.questions(), [pilot], agents=(agent,))
        assert len(with_canaries) == count + 2 and {q.qid for q in chosen} <= {q.qid for q in with_canaries}
    both = {*flips_of("claude"), *flips_of("codex")}
    assert {q.qid for q in discriminating(pilot.questions(), [pilot], canaries=0)} == both, "the agents together: a case that separated for either"


def test_the_canaries_are_one_that_every_configuration_got_right_and_one_that_every_configuration_got_wrong(pilot):
    sep = separate(pilot.questions(), [pilot], ("claude",))
    assert len(sep.separating) == 4 and sep.right and sep.wrong, "there are cases of each kind among the 24 that both configurations were asked"
    chosen = [q.qid for q in discriminating(pilot.questions(), [pilot], agents=("claude",))]
    assert {q for q in chosen if q not in sep.separating} == {sep.right[0], sep.wrong[0]}, "in the questions' order, whichever comes first"
    # (a question asked of `files` only, the sound ones, has one done result: it is right or wrong, never separating)
    audio = [q.qid for q in pilot.questions() if q.tags["modality"] == "audio"]
    assert not set(audio) & set(sep.separating)


def test_a_run_compared_with_itself_has_d1s_counts_and_no_flips(pilot):
    c = compare_runs(pilot, pilot)
    got = {(x.agent, x.config): (x.correct_a, x.n_a) for x in c.cells}
    assert got == D1_COUNTS
    assert all(x.correct_a == x.correct_b and not x.flips and x.p == 1.0 and x.inside and not x.significant for x in c.cells)
    assert c.conditions == {"every cell inside the band": True, "no cell with McNemar p under 0.05": True, "the verdict lines read the same": True,
                            "no setup result in B": True}
    assert set(c.verdicts) == {("claude", "mm/d1-3B"), ("codex", "mm/d1-3B")}
    assert all(a == b and a.startswith("none: at no level with 10 or more paired questions") for a, b in c.verdicts.values()), "the recorded report's own"
    text = render_comparison(c)
    assert "yes: every cell inside the band" in text and "  none" in text.split("Flips, to be read")[1].split("Verdict lines")[0]
    assert "claude records no effort in its results, so only its model is compared." in text
    # the recorded run's problems, as its report counts them (3 timeouts, 3 capped); none of them a setup
    assert sum(x.problems_a.get("timeout", 0) for x in c.cells) == 3 and sum(x.problems_a.get("capped", 0) for x in c.cells) == 3


def test_one_planted_flip_in_a_copy_is_the_one_flip_and_it_is_named(pilot, again):
    target = next(q.qid for q in again.questions() if fs.read_json(again.result_path("codex", MM, q.qid))["passed"])
    path = again.result_path("codex", MM, target)
    fs.write_json(path, {**fs.read_json(path), "passed": False, "answer": "unknown"})
    c = compare_runs(pilot, again)
    flips = [(x.agent, x.config, f.qid) for x in c.cells for f in x.flips]
    assert flips == [("codex", "mm/d1-3B", target)]
    x = next(x for x in c.cells if (x.agent, x.config) == ("codex", "mm/d1-3B"))
    assert (x.correct_a, x.correct_b, x.b, x.c, x.p, x.inside) == (23, 22, 0, 1, 1.0, True)
    assert f"codex mm/d1-3B {target}: wants " in render_comparison(c) and render_comparison(c).count(" wants ") == 1


def test_a_cell_is_inside_d1s_band_at_plus_or_minus_2_of_the_recorded_count_and_outside_at_3_in_each_of_the_four_cells(pilot, again):
    assert BAND == 2
    for (agent, config), (correct, asked) in D1_COUNTS.items():
        cfg = CONFIGS[config]
        results = {q.qid: fs.read_json(again.result_path(agent, cfg, q.qid)) for q in again.questions() if again.result_path(agent, cfg, q.qid).exists()}
        right = [qid for qid, r in results.items() if r["passed"]]
        wrong = [qid for qid, r in results.items() if not r["passed"]]
        assert (len(right), len(results)) == (correct, asked)
        for direction, pool, room in ((-1, right, correct), (1, wrong, asked - correct)):
            for k in (2, 3):
                if k > room:
                    continue  # (the cell has no more wrong answers to turn right)
                for qid in pool[:k]:
                    fs.write_json(again.result_path(agent, cfg, qid), {**results[qid], "passed": direction > 0})
                x = next(x for x in compare_cells(pilot, again) if (x.agent, x.config) == (agent, config))
                assert (x.correct_b, x.difference, x.inside) == (correct + direction * k, direction * k, k == 2), (agent, config, direction, k)
                for qid in pool[:k]:  # put it back
                    fs.write_json(again.result_path(agent, cfg, qid), results[qid])
    assert all(x.inside and not x.flips for x in compare_cells(pilot, again)), "put back, the copy is the recorded run"
