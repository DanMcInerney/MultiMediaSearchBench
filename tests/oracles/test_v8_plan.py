"""Oracle 1 for V8: the EgoLife plan without MultimediaSearch. The plan the recorded run's recipe makes now has the plan hash and files hash the
run's run.json recorded; its pool before choosing, its 50 questions with their ages, categories and horizons, and its 2,592 clips and 3
transcript files are the recorded run's; the questions the writer makes of it are the checked-in ones, with their evidence inside the files
they cite; and the recipe that reading the download makes, with its clip table, is the recorded one. Takes a few seconds (the clip table, a
minute and a half, is marked slow). (The file is test_v8_plan.py in tests/oracles, so no module of tests/ may have that name.)

Expected values are the recorded run's own files (tests/oracles/recorded/egolife-s1), captured before the code was moved, and the recorded
run's clips.json and data-manifest.json in D:\\mm-bench where they are. The oracle needs the download at D:\\bench-data\\egolife and holds
only in the time zone the run was made in: elsewhere it skips, and says so."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from v8_recorded import CONVERTED, META, RECIPE, REPORT, RUN, V8, download_skip, zone_skip

from mmsb.core.question import load_questions
from mmsb.scenarios.v8_egolife.choose import AGE_NAMES
from mmsb.scenarios.v8_egolife.manifest import entries_of, transcript_path
from mmsb.scenarios.v8_egolife.questions import questions_of

if reason := download_skip() or zone_skip():
    pytest.skip(reason, allow_module_level=True)


@pytest.fixture(scope="module")
def plan():
    return V8.plan_of(RECIPE)


@pytest.fixture(scope="module")
def questions():
    return load_questions(CONVERTED / "questions.jsonl")


def test_the_recipe_of_the_recorded_run_makes_the_plan_and_files_it_recorded():
    assert (META["seed"], META["scale"], META["days"], META["per_age"], META["speech"], META["generator"]) == (
        RECIPE["seed"], RECIPE["scale"], RECIPE["days"], RECIPE["per_age"], RECIPE["speech"], V8.generator)
    assert Path(META["source"]) == Path(RECIPE["source"])
    assert V8.hashes(RECIPE) == (META["plan"], META["files"])


def test_the_pool_before_choosing_is_the_recorded_one():
    assert V8.pool_counts(RECIPE) == META["pools"] == {
        "video": 143, "speech": 95, "dropped": {"its time is a range or a list in one string": 7,
                                                "a target is outside the clips (a gap) or after the horizon": 1,
                                                "its query is after the last day": 254}}


def test_the_50_questions_are_the_recorded_ones_by_qid_age_category_and_the_seconds_the_horizon_cut_off(plan):
    rows = REPORT["agents"]["codex"]["d1-3B"]["appendix"]
    assert [q["qid"] for q in plan.questions] == [r["qid"] for r in rows] and len(rows) == 50
    assert [(q["level"], q["template"]) for q in plan.questions] == [(r["level"], r["template"]) for r in rows]
    run = REPORT["run"]
    assert [sum(1 for q in plan.questions if q["type"] == "video" and q["level"] == a) for a in AGE_NAMES] == run["video"] == [10, 10, 10, 10]
    assert sum(1 for q in plan.questions if q["type"] == "records") == run["speech"] == 10
    assert {c: sum(1 for q in plan.questions if q["template"] == c) for c in run["categories"]} == run["categories"]
    assert sum(1 for q in plan.questions if q["pilot"]) == 10, "two of each age and two of the speech ones"
    gaps = [q["query"] - q["until"] for q in plan.questions]
    assert (sorted(gaps)[24] + sorted(gaps)[25]) / 2 == run["horizon"]["median_s"] and max(gaps) == run["horizon"]["max_s"]
    assert sum(1 for g in gaps if g > 30) == run["horizon"]["past_30s"] == 0


def test_the_planned_clips_and_transcript_are_the_2595_files_of_the_recorded_data_folder(plan):
    if not (RUN / "data-manifest.json").exists():
        pytest.skip(f"{RUN} is not here, so the recorded data folder is not read")
    recorded = set(json.loads((RUN / "data-manifest.json").read_text(encoding="utf-8")))
    planned = {f.path for f in plan.files} | {transcript_path(d) for d in plan.days}
    assert len(plan.files) == 2592 and len(recorded) == 2595 and planned == recorded


def test_the_recorded_clip_table_in_the_recipe_is_the_recorded_clips_json():
    if not (RUN / "clips.json").exists():
        pytest.skip(f"{RUN} is not here, so the recorded clips.json is not read")
    recorded = json.loads((RUN / "clips.json").read_text(encoding="utf-8"))
    assert RECIPE["clips"] == [[c["rel"], c["length"]] for c in recorded] and len(recorded) == 2592


def test_the_questions_the_writer_makes_of_the_plan_are_the_checked_in_ones_with_their_evidence_inside_the_files_they_cite(plan, questions):
    entries = entries_of(plan, lambda path: (0, 0, ""))  # a question needs the path, time and length of a file, not its size or hash
    assert questions_of(plan, entries) == questions and len(questions) == 50
    by_path = {e.path: e for e in entries}
    spans = 0
    for q, p in zip(questions, plan.questions):
        assert len(q.truth.evidence) == len(p["truth"]["moments"]) >= 1
        for span, (moment, _) in zip(q.truth.evidence, p["truth"]["moments"]):
            e = by_path[span.file]
            assert abs(e.t + span.t0 - moment) < 1e-3 and 0 <= span.t0 <= span.t1 <= e.length, (q.qid, span)
            assert e.kind == ("log" if q.scope.sources == ("transcript",) else "clip")
            spans += 1
        assert q.horizon == p["until"] <= datetime.fromisoformat(q.now).timestamp() and q.group == q.qid
    assert spans >= 50
    assert [q.qid for q in questions if q.tags["pilot"]] == [p["qid"] for p in plan.questions if p["pilot"]]


@pytest.mark.slow
def test_the_recipe_made_from_the_download_is_the_recorded_one_with_the_clip_table_read_from_its_packets():
    """Reads the packets of the 2,592 clips (41 GB), 4 at a time: about a minute and a half."""
    assert V8.recipe(source=RECIPE["source"], seed=1, days=RECIPE["days"], workers=4) == RECIPE
