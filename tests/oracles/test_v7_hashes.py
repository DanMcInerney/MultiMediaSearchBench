"""Oracle 1a: the V7 plan without Multimem. The plan the recorded pilot's recipe makes now has the plan hash and files hash the pilot's
run.json recorded, its 1,572 files and 3 logs are the 1,575 the recorded manifest holds, each with the time the pilot's file was set to,
and the manifest and questions made from the recorded sizes and hashes are the whole week's. Takes about 2 s. (The file is not called
test_v7_plan.py, like the plan's other oracle files, because tests/test_v7_plan.py has that name and pytest keeps one test module of a name.)

Expected values are the pilot's own files (tests/oracles/recorded/pilot-s1), captured before the code was moved. The oracle holds only in
the time zone the pilot was made in (see v7_recorded.py): elsewhere it skips, and says so."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import pytest
from v7_recorded import MANIFEST, META, SHA256, V7, needs_the_recorded_zone, recipe

from mmmb.core.question import load_questions, write_questions
from mmmb.scenarios.v7_screen.manifest import entries_of, log_path
from mmmb.scenarios.v7_screen.questions import questions_of

needs_the_recorded_zone()


@pytest.fixture(scope="module")
def plan():
    return V7.plan_of(recipe())


def test_the_recipe_of_the_recorded_pilot_makes_the_plan_and_files_it_recorded():
    assert (META["seed"], META["scale"], META["days"], META["generator"]) == (1, "pilot", ["2026-11-01", "2026-11-02", "2026-11-03"], V7.generator)
    assert V7.hashes(recipe()) == (META["plan"], META["files"])


def test_the_planned_files_and_logs_are_the_1575_of_the_recorded_manifest_each_with_the_time_it_was_set_to(plan):
    assert len(plan.files) == 1572 and len(MANIFEST) == 1575 == len(SHA256)
    assert {f.path for f in plan.files} | {log_path(s) for s in plan.logs} == set(MANIFEST)
    wrong = [(f.path, MANIFEST[f.path][1], int(f.t * 1e9)) for f in plan.files if MANIFEST[f.path][1] != int(f.t * 1e9)]
    assert not wrong, f"{len(wrong)} files have another mtime than the plan's time, the first {wrong[:3]}"
    for service, lines in plan.logs.items():  # a log's mtime is its last line's
        assert MANIFEST[log_path(service)][1] == int(datetime.fromisoformat(lines[-1]["ts"]).timestamp() * 1e9), service


def test_the_manifest_made_from_the_recorded_sizes_and_hashes_has_each_source_as_many_files_as_the_recorded_folder(plan):
    entries = entries_of(plan, lambda path: (MANIFEST[path][0], MANIFEST[path][1], SHA256[path]))
    assert [e.path for e in entries][:-3] == [f.path for f in plan.files] and len(entries) == 1575
    assert Counter(e.source for e in entries) == Counter(path.split("/")[0] for path in MANIFEST), "a source is a folder of the data"
    assert all((e.size, e.mtime_ns, e.sha256) == (MANIFEST[e.path][0], MANIFEST[e.path][1], SHA256[e.path]) for e in entries)
    assert {e.source for e in entries if e.own_time} == {"phone"} and {e.kind for e in entries if e.source == "logs"} == {"log"}


def test_the_31_questions_of_the_pilot_are_written_with_their_evidence_in_the_recorded_manifest(plan, tmp_path):
    entries = entries_of(plan, lambda path: (MANIFEST[path][0], MANIFEST[path][1], SHA256[path]))
    questions = questions_of(plan, entries)
    assert len(questions) == 31 and [q.qid for q in questions] == [q["qid"] for q in plan.questions]
    write_questions(tmp_path / "questions.jsonl", questions)
    assert load_questions(tmp_path / "questions.jsonl") == questions
    assert all(span.file in MANIFEST for q in questions for span in (*q.truth.evidence, *q.truth.wrong)), "every file cited is in the data"
