"""Ported from tests/test_egolife.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order, part 2: the run loop of V8 with a
stub agent and the report, over the fake download of tests/v8_fake.py. The tests of the scenario's data, build, views and deny words are
tests/test_v8.py. What moved, by the plan's 3.4, 4.5 and 5.2:
- the mm arm is the configuration `stub/all` of a stand-in system (tests/stub_system.py), as in tests/test_runner.py; its words in the
  report ("mm", the model "fake") are the stand-in's ("stub", the variant "all"). The stand-in's session adds the audit fields a system
  that is cut to the horizon adds; what the real audit makes of an ask's results, the home's index and the decider tier are tests/mm's (U7);
- the retrieval tier's records are planted by hand, with the numbers the expected figures are worked out from.
Every expected value is the source's or worked out from what the test planted; the only changes are `arm` to `config` and the names above."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import threading
import time
import types

import imageio_ffmpeg
import pytest
from agent_spec import NO_ISOLATION, clocked
from stub_system import STUB, StubSession, StubSystem
from v8_fake import (BLIND, FILES, OPTIONS, SHADE, V8, built, clip_ends, line_at, lines_by, oracle_file, plan, recipe, source, table,  # noqa: F401
                     view_files, write_run)

from mmmb import systems
from mmmb.core import fs
from mmmb.core.contract import Dump
from mmmb.core.runfolder import RunFolder
from mmmb.core.runner import Agent, run_agents
from mmmb.core.select import selected
from mmmb.core.stats import mcnemar
from mmmb.scenarios.v8_egolife import choose
from mmmb.scenarios.v8_egolife.report import ROLES
from mmmb.scenarios.v8_egolife.view import view_dir

pytestmark = pytest.mark.xdist_group("v8")


CLOCK = "stub's clock reads a later date than now: give since and until as dates and times, not as ages or \"now\"."


VIDEO_QIDS = ("ego-001", "ego-003", "ego-004", "ego-005", "ego-006", "ego-018", "ego-020", "ego-021")


SPEECH_QIDS = ("ego-008", "ego-009")


RIGHT = {"blind": {"ego-001", "ego-003"}, "files": {"ego-001", "ego-003", "ego-004", "ego-005", "ego-008"},
         "stub-all": {*VIDEO_QIDS, *SPEECH_QIDS} - {"ego-021"}}  # who answers which question right, by the configuration's folder


class EgoSession(StubSession):
    """The stand-in system's session, with the audit fields a system that is cut to the horizon adds to a result (what its real audit
    makes of an ask's results is tests/mm's): no recording past H, one ask, started past H for every question but ego-001."""

    def audit(self, case, calls) -> dict:
        return {"after_horizon": 0, "asks": 1, "asks_past_horizon": int(case.qid != "ego-001")}


class EgoSystem(StubSystem):
    def open(self, store, dump, context):
        self.note("open", None, context.agent, context.variant, store)
        return EgoSession(self, context)


class Ego:
    """An agent that answers right for the questions in `RIGHT[folder]` (as `B` or `B) Alice`) and wrong for the rest (blind says an
    option's words for ego-018: no letter). It records what each exec was given: with tools of its own, the files and lines it can read."""

    def __init__(self, questions, *, pause: float = 0.0, events=None):
        self.questions = {q.qid: q for q in questions}
        self.pause, self.events = pause, events
        self.seen: dict[tuple[str, str], dict] = {}
        self.log: list[tuple] = []
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        qid, folder = spec.run_dir.name, spec.run_dir.parent.name
        q, began = self.questions[qid], time.monotonic()
        seen = {"data": spec.data, "prompt": spec.prompt, "work": spec.work}
        if spec.own_tools:
            seen["files"] = view_files(spec.data)
            seen["lines"] = [line for f in sorted((spec.data / "transcript").glob("*.jsonl")) for line in f.read_text(encoding="utf-8").splitlines()]
        time.sleep(self.pause)
        truth = q.grade.truth
        if qid in RIGHT[folder]:
            answer = truth if qid < "ego-005" else f"{truth}) {q.options[truth]}"
        else:
            answer = OPTIONS["A"] if (folder, qid) == ("blind", "ego-018") else next(x for x in "ABCD" if x != truth)
        tools = [{"tool": "ask", "arguments": {"question": SHADE}}] if spec.served.server is not None else []
        with self.lock:
            self.seen[folder, qid] = seen
            self.log.append((folder, qid, began, time.monotonic()))
        return {"final": f"ANSWER: {answer}", "tools": tools, **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0, **({"events": self.events} if self.events else {})}


@pytest.fixture
def made_up_ffmpeg(tmp_path, monkeypatch):
    """`ensure_bin` copies a made-up file, not 80 MB of ffmpeg."""
    ffmpeg = tmp_path / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))


def run_over(run: RunFolder, dump: Dump, agent, configs=(BLIND, FILES, STUB), system=None, jobs: int = 3, **kw) -> StubSystem:
    """`run_agents` over `run` with the stand-in system for `stub` and the real registry for the rest."""
    from mmmb import systems

    system = system or EgoSystem(rules=False)
    run_agents(run, {"claude": Agent(agent)}, list(configs), dump, lambda name: run.folder / "stores" / name, jobs,
               load=lambda name: system if name == "stub" else systems.load(name), isolation=NO_ISOLATION, answer_format="line", **kw)
    return system


@pytest.fixture(scope="module")
def ran(tmp_path_factory, built, recipe):
    """Every question of the fake download asked of the three configurations by a stub agent, three execs at once; the stand-in system
    serves one at a time. The retrieval tier's records are planted for it (see RECORDS)."""
    dump, questions = built
    root = tmp_path_factory.mktemp("run")
    ffmpeg = root / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
        folder = root / "run"
        folder.mkdir()
        run = write_run(folder, recipe, questions, dump, root / "temp")
        fs.write_json(run.manifest, fs.stats_of(dump.root))
        stub = Ego(questions, pause=0.3)
        system = run_over(run, dump, stub, system=EgoSystem(rules=False, job_limit=1, clock_note=CLOCK))
        for q in questions:
            fs.write_json(run.folder / "results" / "retrieval" / "stub-all" / f"{q.qid}.json", RECORDS[q.qid])
        fs.write_json(run.folder / "results" / "retrieval" / "stub-all" / "warm.json", {"config": "stub/all", "seconds": 4.5})
        yield types.SimpleNamespace(run=run, dump=dump, questions=questions, stub=stub, system=system, root=root)


def result(run: RunFolder, folder: str, qid: str) -> dict:
    return fs.read_json(run.folder / "results" / "claude" / folder / f"{qid}.json")


def record_of(qid: str, type_: str, template: str, level: str, found: bool, p: float | None, windows: int, seconds: float, away: int = 0) -> dict:
    """A record of the retrieval tier with the fields the V8 report reads (the others are core's: tests/test_retrieval.py)."""
    return {"qid": qid, "type": type_, "template": template, "level": level, "config": "stub/all", "found": found, "top5": found, "p": p,
            "away": away, "windows": windows, "seconds": seconds, "stopped": "range_end"}


# planted: 7 of the 10 found (6 of the 8 video, 1 of the 2 speech), p of 0.95 for a find, 0.4 for a near miss (ego-021) and none for the rest
RECORDS = {r["qid"]: r for r in (
    record_of("ego-001", "video", "EntityLog", "0-2h", True, 0.95, 100, 10.0),
    record_of("ego-018", "video", "EventRecall", "0-2h", False, None, 50, 5.0),
    record_of("ego-003", "video", "EntityLog", "2-6h", True, 0.95, 100, 10.0, away=1),
    record_of("ego-006", "video", "HabitInsight", "2-6h", True, 0.95, 200, 40.0),
    record_of("ego-004", "video", "RelationMap", "6-24h", True, 0.95, 100, 20.0),
    record_of("ego-020", "video", "TaskMaster", "6-24h", True, 0.95, 100, 10.0),
    record_of("ego-021", "video", "EventRecall", "24h+", False, 0.4, 100, 10.0),
    record_of("ego-005", "video", "HabitInsight", "24h+", True, 0.95, 100, 30.0),
    record_of("ego-008", "records", "TaskMaster", "0-2h", True, 0.95, 2, 1.0),
    record_of("ego-009", "records", "EventRecall", "24h+", False, None, 0, 0.0))}


def test_each_configuration_is_given_what_it_reads_cut_to_the_horizon(ran, plan, table):
    ends = clip_ends(table)
    assert sorted(ran.stub.seen) == sorted((folder, q.qid) for folder in ("blind", "files", "stub-all") for q in ran.questions)
    for q in ran.questions:
        h = q.horizon
        clips, lines = sum(1 for end in ends if end <= h), lines_by(plan, h)
        blind, files, tested = (ran.stub.seen[folder, q.qid] for folder in ("blind", "files", "stub-all"))
        assert blind["data"] is None and tested["data"] is None
        slot = next(k for k in range(4) if ran.run.slot(k) == files["work"])
        assert files["data"] == view_dir(ran.run.temp, "t8", slot)
        assert len([n for n in files["files"] if n.startswith("glasses/")]) == clips
        assert [json.loads(line) for line in files["lines"]] == lines
    assert not view_dir(ran.run.temp, "t8", 0).exists(), "the slots' folders of recordings are removed at the end"
    # a question whose evidence clip has not ended by its horizon is asked with the files of before it: nothing of that clip
    assert "glasses/2026-10-29/11.10.30.mp4" not in ran.stub.seen["files", "ego-018"]["files"]
    assert "glasses/2026-10-29/11.10.30.mp4" in ran.stub.seen["files", "ego-001"]["files"]


def test_the_configurations_differ_only_in_how_the_recordings_are_reached(ran):
    seen = ran.stub.seen
    blind, files, tested = (seen[folder, "ego-001"]["prompt"] for folder in ("blind", "files", "stub-all"))
    head = "It is now Thursday 29 October 2026, 11:12 (day 1 of these recordings; day 1 was Thursday 29 October). I am Jake."
    tail = ("Question 1?\nA) Tasha\nB) Alice\nC) Shure\nD) Lucia\n\nIf what you can find does not settle it, give your best guess. "
            "End with one line ANSWER: <A, B, C or D>.")
    for text in (blind, files, tested):
        assert text.startswith(head) and text.endswith(tail), text
    assert "You cannot see my recordings." in blind and str(seen["files", "ego-001"]["data"]) in files
    assert "glasses/<date>/" in files and "ffmpeg is on PATH (there is no ffprobe)" in files
    assert "stub has recorded these sources of mine from day 1 to now: glasses" in tested and CLOCK in tested
    assert not any(re.search(r"A1_JAKE|egolife|EgoLife|DAY\d", t) for t in (blind, files, tested))
    assert seen["blind", "ego-005"]["prompt"].startswith(
        "It is now Friday 30 October 2026, 15:21 (day 2 of these recordings; day 1 was Thursday 29 October)")


def test_results_are_graded_by_letter_and_audited(ran):
    for folder in ("blind", "files", "stub-all"):
        for q in ran.questions:
            r = result(ran.run, folder, q.qid)
            assert r["passed"] is (q.qid in RIGHT[folder]) and r["problem"] is None and r["timeout_s"] == 900.0, (folder, q.qid)
            assert r["level"] == q.hint and r["template"] == q.tags["template"] and r["absent"] is False and r["last"] is False
            assert ("after_horizon" in r) == (folder == "stub-all") and ("seen_past_horizon" in r) == (folder == "files")
            assert r["config"] == {"blind": "blind", "files": "files", "stub-all": "stub/all"}[folder]
    tested = {q.qid: result(ran.run, "stub-all", q.qid) for q in ran.questions}
    assert all((r["after_horizon"], r["asks"]) == (0, 1) for r in tested.values())
    assert {qid: r["asks_past_horizon"] for qid, r in tested.items()} == {qid: int(qid != "ego-001") for qid in tested}
    assert result(ran.run, "blind", "ego-018")["abstained"] is False and result(ran.run, "blind", "ego-018")["answer"] == "Tasha"
    assert all(result(ran.run, "files", q.qid)["seen_past_horizon"] == [] for q in ran.questions)


def test_the_system_runs_one_at_a_time_and_the_other_configurations_in_parallel(ran):
    def overlap(folder: str) -> bool:
        spans = sorted((a, b) for x, _, a, b in ran.stub.log if x == folder)
        return any(b > c for (_, b), (c, _) in zip(spans, spans[1:]))

    assert overlap("blind") and overlap("files") and not overlap("stub-all"), "jobs=3 for blind and files, one at a time for the system"


def test_a_second_command_asks_nothing(ran):
    again = Ego(ran.questions)
    run_over(ran.run, ran.dump, again)
    assert again.log == []


def test_the_subset_and_only_choose_questions(ran):
    pilot = [q.qid for q in ran.questions if q.tags["pilot"]]
    assert pilot == [q.qid for q in ran.questions] and len(pilot) == 10, "every question of the two-a-bin fake download is a pilot one"
    qs = [dataclasses.replace(q, tags={**q.tags, "pilot": n < 3}) for n, q in enumerate(ran.questions[:6])]
    chosen = lambda *args: [q.qid for q in selected(qs, *args)]  # noqa: E731
    assert chosen("all", {}) == [q.qid for q in qs] and chosen("pilot", {}) == [q.qid for q in qs[:3]]
    assert chosen("pilot", {}, [qs[1].qid]) == [qs[1].qid]
    with pytest.raises(ValueError, match=f"--only names questions that are not in the pilot questions: {qs[4].qid}"):
        chosen("pilot", {}, [qs[4].qid])
    with pytest.raises(ValueError, match="a subset is one of all, pilot"):
        chosen("some", {})


# -- 9. the report ----------------------------------------------------------------------------------------------------------------


def test_the_report_counts_the_three_configurations(ran):
    figures, text = V8.report(ran.run)
    m = figures["agents"]["claude"]["stub/all"]
    assert (m["questions"], m["paired"]) == (10, 10) and m["video"]["n"] == 8 and m["speech"]["n"] == 2
    video = m["video"]
    assert [video["arms"][a]["correct"] for a in ROLES] == [2, 4, 7]
    assert video["arms"]["blind"]["unparsed"] == 1 and video["arms"]["control"]["unparsed"] == 0
    assert [(video[k]["b"], video[k]["c"]) for k in ("tested_control", "control_blind", "tested_blind")] == [(3, 0), (2, 0), (5, 0)]
    assert video["tested_control"]["p"] == mcnemar(3, 0) and video["tested_control"]["diff"] == pytest.approx(3 / 8)
    assert video["tested_control"]["ci"] is None
    assert video["tested_control"]["base"]["correct"] == 4 and video["tested_control"]["other"]["correct"] == 7
    speech = m["speech"]
    assert [speech["arms"][a]["correct"] for a in ROLES] == [0, 1, 2]
    assert (speech["tested_control"]["b"], speech["tested_control"]["c"]) == (1, 0)
    assert [(a, g["n"]) for a, g in m["ages"].items()] == [(a, 2) for a in choose.AGE_NAMES]
    assert m["categories"]["TaskMaster"] == {"n": 1, "blind": 0, "control": 0, "tested": 1}
    assert m["crossover"] is None and m["thin"] == {a: 2 for a in choose.AGE_NAMES}
    assert (m["work"]["tested"]["ask_execs"], m["work"]["tested"]["n"]) == (8, 8) and m["cost"]["blind"]["correct"] == 2
    h = m["horizon"]
    assert (h["audited"], h["after_horizon"], h["asks"], h["asks_past_horizon"]) == (10, 0, 10, 9) and h["max_s"] > 0
    assert figures["run"]["video"] == [2, 2, 2, 2] and figures["run"]["speech"] == 2 and figures["totals"]["to_rerun"] == 0
    assert figures["frame"] == {"control": "files", "tested": "stub"} and figures["run"]["commit"] == "abc1234"
    assert json.loads(json.dumps(figures)) == figures


def test_the_retrieval_tier_is_told_from_the_records_with_the_numbers_they_were_planted_with(ran):
    tier = V8.report(ran.run)[0]["retrieval"]["stub/all"]
    assert (tier["all"]["asked"], tier["all"]["found"], tier["all"]["top5"], tier["all"]["p50"], tier["all"]["p30"]) == (10, 7, 7, 7, 8), \
        "the near miss (p of 0.4) counts at p of 0.3 and not at 0.5"
    assert tier["all"]["p"] == 0.95 and tier["all"]["away"] == 1 and tier["all"]["windows"] == 852 and tier["all"]["stopped"] == 0
    assert [(tier["types"][t]["asked"], tier["types"][t]["found"]) for t in ("video", "speech")] == [(8, 6), (2, 1)]
    assert tier["categories"]["EventRecall"]["asked"] == 3 and tier["first_s"] == 4.5
    assert tier["all"]["s_per_window"] == pytest.approx(median_of([r["seconds"] / r["windows"] for r in RECORDS.values() if r["windows"]]))


def median_of(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


def test_the_report_text_has_its_tables(ran):
    text = V8.report(ran.run)[1]
    for line in ("# EgoLifeQA through the stub bench", "8 video questions (2 aged 0-2h, 2 aged 2-6h, 2 aged 6-24h, 2 aged 24h+;",
                 "No shortcut probe: a public dataset, so the blind arm is the floor",
                 "## claude with all: 8 video questions of 10 (paired: done in all three arms)",
                 "| blind | 2/8 (25%, 7-59) |", "| files | 4/8 (50%,", "| stub | 7/8 (88%,",
                 "| stub vs files | 8 | +38 pts | 3 / 0 | 0.250 |",
                 "| files vs blind | 8 | +25 pts | 2 / 0 | 0.500 |",
                 "| stub vs blind | 8 | +62 pts | 5 / 0 | 0.062 |",
                 "A difference is the second arm less the first", "by age (stub vs files; blind is the floor)", "| 0-2h | 2 |",
                 "The youngest age at which stub leads files: none: at no level with 10 or more paired questions",
                 "too few paired questions at 0-2h (2), 2-6h (2), 6-24h (2), 24h+ (2)", "by category (right / n)",
                 "TaskMaster has few questions in each age", "the 2 speech questions, apart", "Answered from the transcripts alone",
                 "how each arm worked, and what a correct answer cost", "## The decider tier", "| all | all | 10 |", "| all | speech | 2 |",
                 "no fault",
                 "Every arm saw the clips that ended by each question's horizon H (median 225 s before the question",
                 "Recording times after H in stub tool results: 0 in 10 audited execs; asks started past H (the agent's confusion about "
                 "stub's clock, harmless): 9/10.", "## Read with these",
                 "EgoLife has been public since 2025: the blind arm measures", "## Appendix: claude with all, one row per question",
                 "| ego-001 | 0-2h | EntityLog | video | pass 2 s a0 | pass 2 s a0 | pass 2 s a1 |"):
        assert line in text, line
    assert text.count("| ego-0") == 10


def test_a_run_with_no_results_reports_none(ran, tmp_path):
    folder = tmp_path / "run"
    shutil.copytree(ran.run.folder, folder, ignore=lambda where, names: [n for n in names if n in ("results", "execs", "data-manifest.json")])
    figures, text = V8.report(dataclasses.replace(ran.run, folder=folder))
    assert figures["agents"] == {} and figures["retrieval"] == {} and figures["totals"]["results"] == 0
    assert "No agent results yet." in text and "No decider results yet." in text


def test_the_report_reads_a_run_with_no_video_question_paired(ran, tmp_path, made_up_ffmpeg):
    folder = tmp_path / "run"
    shutil.copytree(ran.run.folder, folder, ignore=lambda where, names: [n for n in names if n in ("results", "execs", "data-manifest.json")])
    run = dataclasses.replace(ran.run, folder=folder, data=ran.dump.root, temp=tmp_path / "temp")
    fs.write_json(run.manifest, fs.stats_of(ran.dump.root))
    run_over(run, ran.dump, Ego(ran.questions), jobs=1, only=["ego-008"])  # a speech question alone
    figures, text = V8.report(run)
    m = figures["agents"]["claude"]["stub/all"]
    assert m["video"] is None and m["speech"]["n"] == 1 and m["ages"] == {} and m["paired"] == 1
    assert "## claude with all: 0 video questions of 10" in text and "None paired yet." in text
    assert "the 1 speech questions, apart" in text and "| ego-008 | 0-2h | TaskMaster | speech |" in text


# -- the review's cases -----------------------------------------------------------------------------------------------------------


def test_the_result_of_a_files_exec_whose_output_showed_a_recording_past_the_horizon_says_escaped_and_the_blind_one_does_not(
        built, plan, recipe, tmp_path, made_up_ffmpeg):
    dump, questions = built
    codex = [{"type": "item.completed", "item": {"type": "command_execution", "command": "Get-Content x",
                                                 "aggregated_output": json.dumps(line_at(plan, True))}}]
    folder = tmp_path / "run"
    folder.mkdir()
    run = write_run(folder, recipe, questions, dump, tmp_path / "temp")
    fs.write_json(run.manifest, fs.stats_of(dump.root))
    run_over(run, dump, Ego(questions, events=codex), [BLIND, FILES], jobs=1, only=["ego-001"])
    r = result(run, "files", "ego-001")
    assert r["problem"].startswith("escaped: its output showed") and r["passed"] is False and r["seen_past_horizon"]
    assert result(run, "blind", "ego-001")["problem"] is None and "seen_past_horizon" not in result(run, "blind", "ego-001")
