"""The estimate of what a case costs the model, held to the recorded runs (plan 4.6, U7's acceptance): `gpu_estimate_s`, `exclusive` and
`beyond_timeout` of all 48 recorded pilot mm results and all 50 recorded EgoLife mm results are what the system's table of windows and its
`serve_record` give for the same questions. The expected values are the recorded results'; nothing is computed from the code under test.

The pilot's table is counted from its data by the reader's own demux (the data is at D:/mm-bench-temp, frozen); tests/mm/tables/pilot-windows.json
is that table, made once, so the first test needs no data, and the second asserts that the data still gives it. EgoLife's clips have a length (a
camera's clip plays all of it: a window a second, rounded up, as MultimediaSearch's index estimate counts a video) so its table needs no clip but the
first, for its size; the transcript's lines are the recorded run's (read in place)."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("mms")

from recorded import EGOLIFE, EGOLIFE_RUN, PILOT_DATA, PILOT_WINDOWS, pilot_dump, pilot_results
from v7_recorded import zone_skip as pilot_zone_skip
from v8_recorded import CONVERTED, RECIPE, V8, download_skip
from v8_recorded import zone_skip as ego_zone_skip

from mms import decider

from mmsb.core.contract import Dump
from mmsb.core.question import load_questions
from mmsb.scenarios.v8_egolife.manifest import TRANSCRIPT, entries_of
from mmsb.systems.mm.estimate import ASK_SHARE, HEAVY_S, estimate_s, table_of, windows_in
from mmsb.systems.mm.system import serve_record

D13 = decider.known("d1-3B")[1]
TIMEOUT_S = 900.0  # both recorded runs


@pytest.fixture
def pilot_zone():
    """The pilot's questions are the plan's, whose times are local to the zone the pilot was made in."""
    if reason := pilot_zone_skip():
        pytest.skip(reason)


def test_the_estimate_gives_the_recorded_gpu_seconds_exclusive_and_beyond_timeout_of_every_pilot_mm_result(pilot_zone):
    results = pilot_results()
    assert len(results) == 48 and sum(r["exclusive"] for r in results) == 10 and sum(r["beyond_timeout"] for r in results) == 6, \
        "the recorded results are the pilot's: 48, of which 10 held the GPU alone and 6 could not finish inside the exec"
    table = json.loads(PILOT_WINDOWS.read_text(encoding="utf-8"))
    _, questions = pilot_dump(PILOT_DATA)
    by_qid = {q.qid: q for q in questions}
    wrong = []
    for r in results:
        q = by_qid[r["qid"]]
        got = serve_record(estimate_s(table, D13, q.scope.sources, q.cost_ranges), TIMEOUT_S, None)
        if (got["gpu_estimate_s"], got["exclusive"], got["beyond_timeout"]) != (r["gpu_estimate_s"], r["exclusive"], r["beyond_timeout"]):
            wrong.append((r["agent"], r["qid"], got, (r["gpu_estimate_s"], r["exclusive"], r["beyond_timeout"])))
    assert not wrong, wrong[:5]


def test_the_table_of_the_recorded_pilot_data_is_the_one_checked_in(pilot_zone):
    if not PILOT_DATA.is_dir():
        pytest.skip(f"{PILOT_DATA} is not here: the table is counted from the pilot's 1,572 files and 3 logs, read in place")
    began = time.monotonic()
    dump, _ = pilot_dump(PILOT_DATA)
    table = table_of(dump)
    print(f"\nthe pilot's table counted from its data in {time.monotonic() - began:.0f} s")
    assert table == json.loads(PILOT_WINDOWS.read_text(encoding="utf-8"))
    assert table["decode_s"] == {}, "no source of the pilot has pictures larger than 1280x720"


def test_the_thresholds_are_the_ones_the_old_bench_had_and_the_recorded_results_show_them():
    assert (HEAVY_S, ASK_SHARE) == (300.0, 0.75)
    results = pilot_results()
    assert min(r["gpu_estimate_s"] for r in results if r["exclusive"]) >= HEAVY_S - 0.05 and max(r["gpu_estimate_s"] for r in results if not r["exclusive"]) < HEAVY_S
    assert min(r["gpu_estimate_s"] for r in results if r["beyond_timeout"]) > ASK_SHARE * TIMEOUT_S
    assert max(r["gpu_estimate_s"] for r in results if not r["beyond_timeout"]) <= ASK_SHARE * TIMEOUT_S


def test_a_source_the_dump_does_not_have_is_an_error():
    table = json.loads(PILOT_WINDOWS.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="a source is one of .*, not 'tv'"):
        windows_in(table, "tv", [(0.0, 1.0)])


# -- EgoLife ----------------------------------------------------------------------------------------------------------------------

def test_the_estimate_gives_the_recorded_gpu_seconds_exclusive_and_beyond_timeout_of_every_egolife_mm_result():
    if reason := download_skip() or ego_zone_skip():
        pytest.skip(reason)
    if not (EGOLIFE_RUN / "data" / "transcript").is_dir():
        pytest.skip(f"{EGOLIFE_RUN} is not here: the recorded run's transcript and its first clip are read in place")
    plan = V8.plan_of(RECIPE)
    entries = entries_of(plan, lambda path: (1, 0, "0" * 64))
    dump = Dump("k" * 64, EGOLIFE_RUN / "data", "v8_egolife", entries, (TRANSCRIPT,))
    decode_s = json.loads((EGOLIFE / "run.json").read_text(encoding="utf-8"))["decode_s"]
    table = table_of(dump, decode=lambda root, clips: decode_s)  # (measuring it needs the reader on three clips: the recorded figure is given)
    assert table["decode_s"] == {"glasses": decode_s} == {"glasses": 0.024}, "glasses clips are 1408x1408, larger than 1280x720"
    questions = {q.qid: q for q in load_questions(CONVERTED / "questions.jsonl")}
    results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((CONVERTED / "results" / "codex" / "mm-d1-3B").glob("*.json"))]
    assert len(results) == 50 and not any(r["exclusive"] or r["beyond_timeout"] for r in results)
    wrong = []
    for r in results:
        q = questions[r["qid"]]
        got = serve_record(estimate_s(table, D13, q.scope.sources, q.cost_ranges), TIMEOUT_S, None)
        if (got["gpu_estimate_s"], got["exclusive"], got["beyond_timeout"]) != (r["gpu_estimate_s"], r["exclusive"], r["beyond_timeout"]):
            wrong.append((r["qid"], got, (r["gpu_estimate_s"], r["exclusive"], r["beyond_timeout"])))
    assert not wrong, wrong[:5]
