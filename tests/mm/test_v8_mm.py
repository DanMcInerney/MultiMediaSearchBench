"""Ported from tests/test_egolife.py of MultimediaSearch at fe0ece5 (plan 7.4, oracle 4), in that file's order, part 3: the tests of V8 that need the mm system,
over the fake EgoLife download of tests/v8_fake.py (a two-day download with tiny clips; the rest of V8's tests are tests/test_v8*.py) and a daemon with the
fake decider. What moved, by the plan's 3.4, 4.5, 4.6 and 5.2: the arm `mm` is the configuration `mm/fake`; `egolife.set_horizon`, `audit`, `check_snapshot`
and `HorizonLeak` are the system's (`mmsb.systems.mm`); the decider tier is `Session.oracle` through the core's `run_retrieval`; `bench.run_agents` is the
core's over a stand-in agent.
The changes of an expectation, each from the plan or from a listed difference:
- the decider tier's record is the union record of the core (plan 5.2: `config`, `system` and `variant` for `model`, and V7's fields besides V8's);
- the estimate of a glasses window adds the reader's measured decode only for pictures larger than 1280x720 (plan 4.6): these clips are 64x64, so it adds
  none, and `decode_of` is tested where it is measured;
- the index counts of `test_each_arm_is_given...` are what the stand-in reads of its home, `homes/claude-fake`.
New here, from the handoff: the prompt the system's `Served` makes for each of the 50 recorded EgoLife mm execs is the recorded prompt, clock sentence included
(it is `Served.clock_note`, given when mm's clock is more than an hour past the question's `now`)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("mms")

import fakes  # noqa: E402
from fakes import MM, S, Probe, home_of, made_up_ffmpeg  # noqa: E402
from v8_fake import (BLIND, FILES, SHADE, V8, WORD, built, clip_ends, ep, lines_by, oracle_file, plan, recipe, source, table, write_run)  # noqa: E402,F401

from mms import config  # noqa: E402

from mmsb import systems  # noqa: E402
from mmsb.core import fs, retrieval  # noqa: E402
from mmsb.core.contract import Case  # noqa: E402
from mmsb.core.question import load_questions  # noqa: E402
from mmsb.core.runner import Agent, run_agents  # noqa: E402
from mmsb.core.scenario import Access  # noqa: E402
from mmsb.scenarios.v8_egolife.report import decider_figures, decider_lines  # noqa: E402
from mmsb.systems.mm.audit import HorizonLeak, audit  # noqa: E402
from mmsb.systems.mm.daemon import Api  # noqa: E402
from mmsb.systems.mm.estimate import decode_of, window_s  # noqa: E402
from mmsb.systems.mm.home import exec_project, set_horizon  # noqa: E402
from mmsb.systems.mm.oracle import ask  # noqa: E402

pytestmark = pytest.mark.xdist_group("v8")

RECORDED = Path(__file__).parent.parent / "oracles" / "recorded" / "egolife-s1"
VIDEO_QIDS = ("ego-001", "ego-003", "ego-004", "ego-005", "ego-006", "ego-018", "ego-020", "ego-021")
SPEECH_QIDS = ("ego-008", "ego-009")
RIGHT = {*VIDEO_QIDS, *SPEECH_QIDS} - {"ego-021"}  # who answers right with the system (the evidence clip of ego-021 is a day away)


@pytest.fixture(scope="module")
def v8store(built, tmp_path_factory):
    """The fake download's dump ingested by the system: the clips as video items at their times, the transcript as a log source."""
    return fakes.ingested(built[0], tmp_path_factory.mktemp("v8store") / "store")


@pytest.fixture
def opened(built, v8store, tmp_path, request):
    session = MM.open(v8store, built[0], fakes.context(tmp_path / "homes"))
    request.addfinalizer(session.close)
    return session


def listed(api, types_: list[str], **extra) -> list[dict]:
    return api.call("find", types=types_, since=ep(1, "00:00:00.00"), until=ep(2, "23:59:00.00"), limit=200, scope="all", **extra)["results"]


def scan(api, **params) -> dict:
    return ask(api, {"question": SHADE, "sources": ["glasses"], "since": ep(1, "11:00:00.00"), "until": ep(2, "23:00:00.00"),
                     "limit": 20, "candidates": 5, **params}, "id:bench-test")[0]


# -- the store -----------------------------------------------------------------------------------------------------------------------------

def test_the_store_holds_each_clip_and_each_line_at_its_time(plan, v8store):
    with closing(sqlite3.connect(v8store / "home" / "index.sqlite")) as db:
        items = db.execute("SELECT name, t0, t1, state, kind FROM items WHERE source = 'glasses' ORDER BY t0").fetchall()
        records = [t for (t,) in db.execute("SELECT t FROM records WHERE source = 'transcript' ORDER BY t")]
    assert [(n, t0, t1) for n, t0, t1, *_ in items] == [(f.name, f.t, f.t + f.spec["length"]) for f in plan.files]
    assert {(state, kind) for *_, state, kind in items} == {("ok", "video")}
    assert records == sorted(datetime.fromisoformat(r["end"]).timestamp() for rows in plan.logs.values() for r in rows)
    assert not (v8store / "home" / "daemon.json").exists()


# -- the horizon ------------------------------------------------------------------------------------------------------------------------------

def test_set_horizon_hides_what_ended_after_it(table, opened):
    api = opened.daemon.api
    h = clip_ends(table)[1]  # the end of the second clip: the evidence clip starts there
    reference = scan(api, until=ep(1, "11:10:30.00"))  # the clips overlap by their last 0.06 s: a window of the next one is before H
    assert scan(api)["sightings"], "the red frame of the evidence clip is found while nothing is cut"
    evidence = next(r["id"] for r in listed(api, ["item"]) if r["name"] == "2026-10-29 11.10.30.mp4")
    assert len(listed(api, ["record"])) == 13
    assert set_horizon(opened.home, h) == {"clips": 2, "clips_hidden": 22, "records": 2, "records_hidden": 11}
    # find shows no item and no record after H
    items, records = listed(api, ["item"]), listed(api, ["record"])
    assert len(items) == 2 and all(config.parse_when(r["t1"]) <= h + 0.001 for r in items)
    assert len(records) == 2 and all(config.parse_when(r["t0"]) <= h for r in records)
    # an ask over all time reads what an ask that ended at H read before the cut, and finds nothing of the evidence clip
    cut = scan(api)
    assert reference["sightings"] == cut["sightings"] == [] and cut["progress"]["windows"] > 0
    assert (cut["progress"]["windows"], cut["candidates"]) == (reference["progress"]["windows"], reference["candidates"])
    # a later item is not found by look, nor a later record by its source and time
    assert api.failure("look", item=evidence).code == "not_found"
    assert api.failure("look", source="transcript", t=ep(1, "11:12:36.00")).code == "not_found"
    shown = api.call("look", source="transcript", t=ep(1, "11:09:55.00"))
    assert "Okay, then we need a stopwatch" in shown["text"] and "screwdriver" not in shown["text"]


def test_moving_the_horizon_back_and_forward_restores_everything(opened):
    api, home = opened.daemon.api, opened.home
    before = listed(api, ["record"])
    assert api.call("find", text="screwdriver", types=["record"])["results"]
    set_horizon(home, ep(1, "11:09:00.00"))
    assert listed(api, ["record"]) == [] and api.call("find", text="screwdriver", types=["record"])["results"] == []
    set_horizon(home, ep(1, "11:10:49.00"))
    assert len(listed(api, ["record"])) == 3, "the lines that ended by then: the screwdriver one ends at 11:10:47, the next at 11:10:49.5"
    assert api.call("find", text="screwdriver", types=["record"])["results"]
    set_horizon(home, ep(2, "23:59:00.00"))
    assert listed(api, ["record"]) == before, "the same rows, with the same ids"
    assert api.call("find", text="screwdriver", types=["record"])["results"], "and the search index finds a shown word again"
    with closing(sqlite3.connect(home / "index.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM bench_hidden_records").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM items WHERE state = 'ok'").fetchone() == (24,)


# -- the decider tier ------------------------------------------------------------------------------------------------------------------------------

RECORD_KEYS = {"qid", "type", "template", "level", "last", "absent", "config", "system", "variant", "question", "source", "since", "until", "ranges",
               "limit", "found", "top5", "p", "rank", "moments", "moments_found", "others", "others_found", "in_range", "false", "away", "windows",
               "seconds", "wall_s", "stopped", "warnings", "sightings", "candidates"}  # the union record of the core (plan 5.2)


@pytest.fixture(scope="module")
def decided(built, recipe, v8store, tmp_path_factory):
    """Every question asked of the fake decider by the retrieval tier: the run folder and the records by qid."""
    dump, questions = built
    root = tmp_path_factory.mktemp("decided")
    (root / "run").mkdir()
    run = write_run(root / "run", recipe, questions, dump, root / "temp", configs=(BLIND, FILES, S))
    with fakes.Opened(v8store, dump, run.folder / "homes") as session:
        retrieval.run_retrieval(run, S, session, questions, warm_s=0.5)
    return run, {q.qid: fs.read_json(run.folder / "results" / "retrieval" / "mm-fake" / f"{q.qid}.json") for q in questions}


def test_the_decider_finds_the_evidence_with_the_fake(built, decided):
    questions = {q.qid: q for q in built[1]}
    _, records = decided
    one = records["ego-001"]
    assert set(one) == RECORD_KEYS
    assert (one["found"], one["top5"], one["moments_found"], one["away"], one["moments"]) == (True, True, [0], 0, 1)
    assert one["p"] >= 0.9 and one["windows"] > 0 and (one["config"], one["system"], one["variant"]) == ("mm/fake", "mm", "fake")
    assert one["question"] == SHADE and one["source"] == "glasses"
    spec = questions["ego-001"].retrieval
    assert one["ranges"] == [list(r) for r in spec.ranges] and one["ranges"][0][1] == questions["ego-001"].horizon
    for qid in ("ego-003", "ego-004", "ego-005", "ego-020"):
        assert records[qid]["found"] is True and records[qid]["away"] == 0, qid
    # a time_list asks two ranges and finds the first of its two targets only; the second range is cut at H
    six = records["ego-006"]
    assert (six["found"], six["moments_found"], len(six["ranges"])) == (True, [0], 2)
    assert six["ranges"][1][1] == questions["ego-006"].horizon and six["ranges"][0][1] < six["ranges"][1][0]
    # nothing to find: the evidence clip has not ended by H, or is hours from the target
    for qid in ("ego-018", "ego-021"):
        assert (records[qid]["found"], records[qid]["moments_found"], records[qid]["away"]) == (False, [], 0), qid
    assert all(r["stopped"] == "range_end" for r in records.values()), "every range was scanned to its end"


def test_the_decider_reads_the_transcript_for_a_speech_question(decided):
    _, records = decided
    eight, nine = records["ego-008"], records["ego-009"]
    assert (eight["source"], eight["question"], eight["type"]) == ("transcript", WORD, "records")
    assert (eight["found"], eight["moments_found"]) == (True, [0]) and eight["p"] >= 0.9
    assert (nine["found"], nine["away"]) == (False, 0), "no line near its target says screwdriver"


def test_the_decider_tier_is_asked_once_and_reported(built, decided, v8store, tmp_path):
    run, records = decided
    folder = run.folder / "results" / "retrieval" / "mm-fake"
    assert sorted(p.stem for p in folder.glob("*.json")) == sorted(["warm", *records]) and fs.read_json(folder / "warm.json") == {"config": "mm/fake", "seconds": 0.5}

    class Refuses:
        def oracle(self, query):
            raise AssertionError("a configuration with every record is asked nothing")

    assert retrieval.run_retrieval(run, S, Refuses(), built[1]) == 0
    tier = decider_figures(run)["mm/fake"]
    assert (tier["all"]["asked"], tier["all"]["found"]) == (10, 7) and tier["all"]["top5"] >= 7 and tier["all"]["p50"] >= 7
    assert [(tier["types"][t]["asked"], tier["types"][t]["found"]) for t in ("video", "speech")] == [(8, 6), (2, 1)]
    assert tier["categories"]["EventRecall"]["asked"] == 3 and tier["first_s"] is not None
    text = "\n".join(decider_lines({"mm/fake": tier}))
    assert "| fake | all | 10 |" in text and "| fake | speech | 2 |" in text and "no fault" in text


# -- the system in the run loop of V8 ---------------------------------------------------------------------------------------------------------------

def mm_ask(leak: str | None = None):
    """What a stand-in does in the system's configuration: it reads its home (the clips and records not cut away), asks over all time (over its horizon
    for ego-001), and reports the ask as a tool call; with `leak` the result of that question also holds a recording past its horizon."""
    def during(spec, q):
        home = home_of(spec)
        with closing(sqlite3.connect(home / "index.sqlite")) as db:
            seen = {"home": home, "clips": db.execute("SELECT COUNT(*) FROM items WHERE source = 'glasses' AND state = 'ok'").fetchone()[0],
                    "records": db.execute("SELECT COUNT(*) FROM records WHERE source = 'transcript'").fetchone()[0]}
        until = q.horizon if q.qid == "ego-001" else ep(2, "23:00:00.00")
        params = {"question": SHADE, "sources": ["glasses"], "since": q.scope.since, "until": until, "limit": 20, "candidates": 5}
        result, _ = ask(Api(home), params, exec_project(q.qid))
        seen["ask"] = result
        if q.qid == leak:
            late = {"source": "glasses", "t": config.iso(q.horizon + 60), "p": 0.9}  # one recording past the horizon (G1's a5: a tolerance of one let it by)
            result = {**result, "sightings": [*result["sightings"], late]}
        seen["tools"] = [{"tool": "ask", "arguments": {**params, "until": config.iso(until)},
                          "result": {"content": [{"type": "text", "text": json.dumps({**result, "job": "j1"})}]}}]
        return seen

    return during, lambda spec, q, found: found["tools"] if found else []


def go(run, dump, store, agent, configs, jobs: int = 3, **kw) -> None:
    run_agents(run, {"claude": Agent(agent)}, list(configs), dump, lambda name: store, jobs, load=lambda name: MM if name == "mm" else systems.load(name),
               isolation=systems.isolation(), answer_format="line", **kw)


@pytest.fixture(scope="module")
def ran(built, recipe, v8store, tmp_path_factory):
    """Every question of the fake download asked of the system's configuration by a stand-in, three execs at once asked for, one at a time given (the
    horizon cuts the whole home)."""
    dump, questions = built
    root = tmp_path_factory.mktemp("ran")
    (root / "run").mkdir()
    during, tools = mm_ask()
    with pytest.MonkeyPatch.context() as patch:
        made_up_ffmpeg(patch, root)
        run = write_run(root / "run", recipe, questions, dump, root / "temp", configs=(S,))
        fs.write_json(run.manifest, fs.stats_of(dump.root))
        probe = Probe(questions, pause=0.3, during=during, tools=tools, answer=lambda spec, q: q.grade.truth if q.qid in RIGHT else "A" if q.grade.truth != "A" else "B")
        go(run, dump, v8store, probe, (S,))
        yield type("Ran", (), {"run": run, "probe": probe, "dump": dump, "questions": questions, "root": root})


def result(run, folder: str, qid: str) -> dict:
    return fs.read_json(run.folder / "results" / "claude" / folder / f"{qid}.json")


def test_the_system_is_given_the_home_cut_to_the_horizon_and_an_ask_over_all_time_reads_no_window_past_it(ran, plan, table):
    ends = clip_ends(table)
    assert sorted(ran.probe.seen) == sorted(q.qid for q in ran.questions)
    for q in ran.questions:
        seen = ran.probe.seen[q.qid]
        assert (seen["clips"], seen["records"]) == (sum(1 for end in ends if end <= q.horizon), len(lines_by(plan, q.horizon))), "the home was cut before the exec"
        assert seen["home"] == ran.run.folder / "homes" / "claude-fake"
    # an exec that asks over all time reads no window past H: ego-018's evidence clip is past it
    assert ran.probe.seen["ego-018"]["ask"]["sightings"] == [] and ran.probe.seen["ego-018"]["ask"]["progress"]["windows"] == 2, \
        "a window for each of the two clips before H (a still clip is one run of frames)"
    assert ran.probe.seen["ego-001"]["ask"]["sightings"], "and the same ask finds it for the question that has it by H"


def test_results_are_audited_for_recordings_past_the_horizon(ran):
    for q in ran.questions:
        r = result(ran.run, "mm-fake", q.qid)
        assert r["problem"] is None and r["timeout_s"] == 900.0 and r["config"] == "mm/fake", q.qid
        assert (r["after_horizon"], r["asks"]) == (0, 1)
        assert r["asks_past_horizon"] == int(q.qid != "ego-001"), "only ego-001 ended its ask at H"


def test_the_system_runs_one_at_a_time(ran):
    spans = sorted((a, b) for _, _, _, a, b in ran.probe.log)
    assert len(spans) == len(ran.questions) and not any(b > c for (_, b), (c, _) in zip(spans, spans[1:])), "jobs=3 asked for, one given"


def test_a_leak_into_a_result_ends_the_command_and_leaves_the_question_to_run_again(built, recipe, v8store, tmp_path):
    dump, questions = built
    (tmp_path / "run").mkdir()
    run = write_run(tmp_path / "run", recipe, questions, dump, tmp_path / "temp", configs=(BLIND, FILES, S))
    fs.write_json(run.manifest, fs.stats_of(dump.root))
    during, tools = mm_ask(leak="ego-018")
    with pytest.MonkeyPatch.context() as patch:
        made_up_ffmpeg(patch, tmp_path)
        with pytest.raises(HorizonLeak, match="ego-018: 1 recording times after the horizon"):
            go(run, dump, v8store, Probe(questions, during=during, tools=tools, answer=lambda spec, q: "A"), (BLIND, FILES, S), jobs=1, only=["ego-018"])
    assert (run.folder / "results" / "claude" / "blind" / "ego-018.json").exists() and (run.folder / "results" / "claude" / "files" / "ego-018.json").exists()
    assert not (run.folder / "results" / "claude" / "mm-fake" / "ego-018.json").exists(), "the question stays to run again"
    assert not (run.folder / "homes" / "claude-fake" / "daemon.json").exists(), "and the daemon of the home is stopped"


# -- what the audit counts ---------------------------------------------------------------------------------------------------------------------------

def call(tool: str, result_, **args) -> dict:
    return {"tool": tool, "arguments": args, "result": {"content": [{"type": "text", "text": json.dumps(result_)}]}}


def index_ids(v8store, h: float) -> tuple[int, int]:
    """The ids of an item that ends after `h` and one that ends by it, in the store's index."""
    with closing(sqlite3.connect(v8store / "home" / "index.sqlite")) as db:
        later = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 > ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
        earlier = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 <= ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
    return later, earlier


def test_the_audit_counts_recordings_past_the_horizon_in_asks_finds_and_looks(built, v8store):
    h = built[1][0].horizon
    iso, home = config.iso, v8store / "home"
    ask_ = call("ask", {"job": "j1", "sightings": [{"t0": iso(h - 5), "t1": iso(h)}], "candidates": [{"t": iso(h - 9)}]}, question="q", until=iso(h))
    assert audit([ask_], h, None) == {"after_horizon": 0, "asks": 1, "asks_past_horizon": 0}, "a time at H is not past it"
    late = call("ask", {"job": "j2", "sightings": [{"t0": iso(h + 5), "t1": iso(h + 6)}], "candidates": [{"t": iso(h + 9)}]}, question="q")
    assert audit([late, call("ask", {"job": "j2"}, job="j2")], h, None) == {"after_horizon": 3, "asks": 1, "asks_past_horizon": 1}
    # an ask with `questions` counts; one mms refused (no job in its result) does not, whatever its until says
    several = call("ask", {"job": "j3"}, questions={"a": {}, "b": {}}, until="2026-10-30T21:30:00")
    refused = {"tool": "ask", "arguments": {"question": "q", "until": "yesterday morning"}, "result": "bad_request: until is a time"}
    assert audit([refused, several], h, None) == {"after_horizon": 0, "asks": 1, "asks_past_horizon": 1}
    # a find's rows, but not the labels the agent made itself (mms dates them by its clock, days later than the question)
    found = call("find", {"results": [{"type": "item", "t0": iso(h - 1), "t1": iso(h + 1)}, {"type": "record", "t0": iso(h + 30), "t1": None},
                                      {"type": "label", "t0": iso(h + 40), "t1": iso(h + 41)}]})
    assert audit([found], h, None)["after_horizon"] == 2
    later, earlier = index_ids(v8store, h)
    assert audit([call("look", {"item": later, "kind": "image"})], h, home)["after_horizon"] == 1
    assert audit([call("look", {"moments": [{"item": earlier}, {"error": {}}]})], h, home)["after_horizon"] == 0
    # a look of several moments gives a JSON block for each, with a picture before it: all are audited, not the first alone
    blocks = {"content": [{"type": "image"}, {"type": "text", "text": json.dumps({"item": earlier})}, {"type": "image"},
                          {"type": "text", "text": json.dumps({"item": later})}]}
    assert audit([{"tool": "look", "arguments": {"moments": []}, "result": blocks}], h, home)["after_horizon"] == 1
    assert audit([{"tool": "ask", "arguments": {"question": "q"}, "result": "not json"}], h, None) == {"after_horizon": 0, "asks": 0, "asks_past_horizon": 0}


def test_the_audit_catches_a_sheet_cut_from_an_item_past_the_horizon_and_not_the_times_the_agent_asked_for(built, v8store):
    h = built[1][0].horizon
    iso, home = config.iso, v8store / "home"
    later, earlier = index_ids(v8store, h)
    tile = lambda n, item, **when: {"n": n, **when, "look": {"item": item, "at": 0}}  # noqa: E731
    sheet = lambda since, until, *tiles: {"kind": "sheet", "source": "glasses", "since": iso(since), "until": iso(until), "labels": [],
                                          "image": {}, "tiles": list(tiles)}  # noqa: E731
    leak = lambda result_: audit([call("look", result_, source="glasses")], h, home)["after_horizon"]  # noqa: E731
    before = sheet(h - 20, h, tile(1, earlier, t=iso(h - 20)), tile(2, earlier, t0=iso(h - 10), t1=iso(h)))
    assert leak(before) == 0, "tiles cut from items that end by the horizon are no leak"
    asked = sheet(h - 20, h + 5000, tile(1, earlier, t=iso(h + 3)), tile(2, earlier, t0=iso(h + 4), t1=iso(h + 6)))
    assert leak(asked) == 0, "a range and tile times past it are what the agent asked for, as an ask's until"
    assert leak(sheet(h - 20, h, tile(1, later, t=iso(h - 20)))) == 1, "a tile cut from an item that ended past it"
    later_moments = {"moments": [before, {"error": {}}, sheet(h - 1, h + 2, tile(1, later, t=iso(h)))]}
    assert leak(later_moments) == 1, "a sheet among moments, and a failed moment"
    assert leak({"kind": "sheet", "tiles": [None, {"n": 1}]}) == 0, "a sheet's fields may be missing"


def test_the_audit_catches_a_time_given_as_t_in_a_sighting_or_a_candidate_and_in_a_check_in_that_leaves_out_the_rest(built):
    h = built[1][0].horizon
    iso = config.iso
    before = {"source": "glasses", "t": iso(h - 5), "p": 0.9, "kind": "video", "look": {"item": 1, "at": 0}}
    spanning = {"source": "glasses", "t0": iso(h + 8), "t1": iso(h + 9), "p": 0.5, "kind": "audio", "look": {"item": 2, "at": 0}}
    # a sighting of one instant has only `t`, a candidate that spans has `t0` and `t1`: each is a time the agent saw
    late = call("ask", {"job": "j1", "sightings": [{**before, "t": iso(h + 7)}], "candidates": [spanning]}, question="q", until=iso(h))
    assert audit([late], h, None) == {"after_horizon": 3, "asks": 1, "asks_past_horizon": 0}
    assert audit([call("ask", {"job": "j1", "sightings": [before], "candidates": [{**before, "t": iso(h)}]}, question="q", until=iso(h))],
                 h, None)["after_horizon"] == 0, "a `t` at or before the horizon is no leak"
    # a later reply of the same job holds only what is new, and is audited as it is: its own sightings, none when it has none
    check_in = call("ask", {"job": "j1", "state": "running", "progress": {"windows": 3, "sightings": 1}, "sightings": [{**before, "t": iso(h + 11)}]}, job="j1")
    assert audit([check_in], h, None)["after_horizon"] == 1
    assert audit([call("ask", {"job": "j1", "state": "running", "progress": {"windows": 3}}, job="j1")], h, None)["after_horizon"] == 0


# -- the estimate ------------------------------------------------------------------------------------------------------------------------------------------

def test_the_estimate_is_the_decider_ask_at_the_cost_of_a_window_here(built, decided, opened):
    _, questions = built
    by = {q.qid: q for q in questions}
    about = opened.model.about
    video = by["ego-001"]
    windows = opened.estimate(video.scope.sources, video.cost_ranges) / window_s(about)
    api = opened.daemon.api
    read = sum(ask(api, {"question": video.retrieval.query, "sources": [video.retrieval.sources[0]], "since": a, "until": b, "limit": 20, "candidates": 5,
                         "max_s": 1800, "still_s": 0}, "id:bench-oracle")[0]["progress"]["windows"] for a, b in video.cost_ranges)
    assert read * 0.9 <= windows <= read * 1.1, "close to the windows an ask of a frame a second reads"
    assert 0 < decided[1]["ego-001"]["windows"] <= windows, "and the ask with the gate reads no more: these still clips join their frames"
    speech = by["ego-008"]
    assert opened.estimate(speech.scope.sources, speech.cost_ranges) == pytest.approx(window_s(about, "records")), "a record window"
    assert opened.table["decode_s"] == {}, "these clips are 64x64: the reader's decode is what mms assumes, and none is added"


def test_the_decode_of_a_window_is_measured_over_the_first_a_middle_and_the_last_clip(built):
    dump, _ = built
    clips = sorted((e for e in dump.entries if e.media == "video"), key=lambda e: e.t)
    assert 0 < decode_of(dump.root, clips) < 1


# -- the prompt, as the recorded run made it -----------------------------------------------------------------------------------------------------------------

def test_the_prompt_the_system_makes_for_each_recorded_egolife_mm_exec_is_the_recorded_one(opened, tmp_path):
    prompts = json.loads((RECORDED / "prompts.json").read_text(encoding="utf-8"))
    questions = load_questions(RECORDED / "converted" / "questions.jsonl")
    assert sorted(prompts) == sorted(q.qid for q in questions) and len(questions) == 50
    (tmp_path / "w").mkdir()
    for q in questions:
        case = Case(q.qid, datetime.fromisoformat(q.now).timestamp(), None, q.scope.sources, q.scope.since, q.scope.until, q.group, 0.0, 0, tmp_path / "w", tmp_path / "r")
        served = opened.serve(case)
        assert served.clock_note is not None and served.holder == "mm", "mm's clock is days past the recorded questions'"
        assert V8.prompt(q, Access(False, None, served, "line")) == prompts[q.qid]["mm"], q.qid
