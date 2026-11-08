"""New tests for mmsb/scenarios/v7_screen/questions.py and manifest.py: the plan's question dicts written as `Question`s, with the evidence
of each found in the manifest. Nothing is written to disk but the question file: the manifest is made from the plan with made-up sizes.

Expected values are the plan's own specs (a file's length, a note's text, a state's overlay, a sound's time), a recomputation of the
definitions in DESIGN 6.1 and plan 5.2, and the recorded pilot's replayed results (tests/oracles/recorded/pilot-s1/replay), which have the
verdict the pilot gave them."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from mmsb.core import grade
from mmsb.core.question import load_questions, write_questions
from mmsb.scenarios.v7_screen.manifest import entries_of, log_path
from mmsb.scenarios.v7_screen.media_common import NEEDLE_RGB
from mmsb.scenarios.v7_screen.plan import days_before
from mmsb.scenarios.v7_screen.questions import questions_of
from mmsb.scenarios.v7_screen.week import plan as make_plan

REPLAYS = Path(__file__).parent / "oracles" / "recorded" / "pilot-s1" / "replay"
PILOT_DAYS = [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]
# the kind of file that holds a template's evidence, and the service whose log holds a log template's lines
FILE_OF = {"note-code": "note", "config-asof": "config", "log-count": "log", "log-last": "log", "photo-sign": "photo", "photo-count": "photo",
           "screen-error": "screen", "porch-parcel": "porch", "mic-doorbell": "mic", "mic-alarm": "mic"}
SERVICE_OF = {"log-count": "payments", "log-last": "web"}
BASE_OF = {"absent-note": "note-code", "absent-log": "log-count", "absent-sign": "photo-sign", "absent-error": "screen-error",
           "absent-parcel": "porch-parcel", "absent-doorbell": "mic-doorbell"}


def made_up(path: str) -> tuple[int, int, str]:
    return 1, 0, "0" * 64


def stamp(line: dict) -> float:
    return datetime.fromisoformat(line["ts"]).timestamp()


@pytest.fixture(scope="module")
def plans():
    return {"tiny": make_plan(7, "tiny", [date(2026, 10, 30), date(2026, 10, 31)]), "pilot": make_plan(1, "pilot", PILOT_DAYS),
            "standard": make_plan(2, "standard", days_before(date(2026, 11, 2), 7))}


@pytest.fixture(scope="module")
def written(plans):
    return {scale: (plan, entries_of(plan, made_up), questions_of(plan, entries_of(plan, made_up))) for scale, plan in plans.items()}


def test_every_question_of_every_scale_is_written_and_read_back_equal(written, tmp_path):
    for scale, (plan, entries, questions) in written.items():
        assert [q.qid for q in questions] == [q["qid"] for q in plan.questions], scale
        write_questions(tmp_path / f"{scale}.jsonl", questions)
        assert load_questions(tmp_path / f"{scale}.jsonl") == questions, scale
    assert [len(questions) for _, _, questions in written.values()] == [19, 31, 31]


def inside_its_file(plan, entry, files, span, template, where):
    """The span is inside the file it names, as the plan's own spec says the file is: a recording its length, a photo or a note its
    instant, a version of app.toml until the next version, a log between its first line and its last."""
    kind = entry.kind
    if kind in ("screen", "porch", "mic"):
        assert 0 <= span.t0 <= span.t1 <= files[span.file].spec["length_s"], where
    elif kind in ("photo", "note"):
        assert span.t0 == span.t1 == 0, where
    elif kind == "config":
        later = [f.t for f in plan.files if f.kind == "config" and f.t > entry.t]
        assert span.t0 >= 0 and (not later or entry.t + span.t1 <= min(later)), where
    else:
        first, last = (stamp(plan.logs[SERVICE_OF[template]][i]) for i in (0, -1))
        assert entry.t == first and 0 <= span.t0 <= span.t1 <= last - first, where


def shows_the_answer(plan, files, q, planned):
    """What the first span's file shows at the span is the answer, read from the spec of the file and not from the question."""
    template, value, first = planned["template"], q.truth.value, q.truth.evidence[0]
    spec = files[first.file].spec if first.file in files else None
    if template == "note-code":
        assert value in spec["text"]
    elif template == "config-asof":
        assert f"retention_days = {value}\n" in spec["text"]
    elif template == "log-count":
        code = planned["fake"]["question"].split(":", 1)[1]
        lines = {stamp(line): line for line in plan.logs["payments"]}
        assert len(planned["truth"]["moments"]) == int(value) and all(lines[a]["code"] == code for a, _ in planned["truth"]["moments"])
    elif template == "log-last":
        lines = {stamp(line): line for line in plan.logs["web"]}
        assert all(lines[a]["msg"].startswith("deploy finished") for a, _ in planned["truth"]["moments"])
    elif template == "photo-sign":
        assert spec["scene"] == "sign" and tuple(spec["rgb"]) == NEEDLE_RGB["sign"] and str(spec["number"]) == value
    elif template == "photo-count":
        cars = [files[s.file].spec for s in q.truth.evidence]
        assert len(cars) == int(value) and all(c["scene"] == "car" and tuple(c["rgb"]) == NEEDLE_RGB["car"] for c in cars)
    elif template == "screen-error":
        (state,) = [s for s in spec["states"] if abs(s["from"] - first.t0) < 1e-3 and abs(s["to"] - first.t1) < 1e-3]
        assert state["overlay"]["kind"] == "error" and state["overlay"]["code"] in value
    elif template == "porch-parcel":
        assert all(list(files[s.file].spec["parcel"]) == pytest.approx([s.t0, s.t1], abs=1e-3) for s in q.truth.evidence)
    else:
        sound = template.split("-")[1]  # doorbell or alarm
        assert any(s["kind"] == sound and abs(s["at"] - first.t0) < 1e-3 and abs(s["at"] + s["dur"] - first.t1) < 1e-3 for s in spec["sounds"])


def test_a_questions_evidence_is_the_file_that_shows_the_moment_at_the_seconds_it_is_there(written):
    """At every scale, for every question with a needle: each moment of the plan is a span of a file of the kind that shows it, found
    in the manifest, which gives the moment back within a millisecond; the span is inside the file; and the file shows the answer."""
    for scale, (plan, entries, questions) in written.items():
        by_path, files = {e.path: e for e in entries}, {f.path: f for f in plan.files}
        for q, planned in zip(questions, plan.questions):
            if planned["absent"]:
                assert q.truth.evidence == () and q.truth.wrong == () and q.truth.value is None, q.qid
                continue
            for spans, moments in ((q.truth.evidence, planned["truth"]["moments"]), (q.truth.wrong, planned["truth"]["others"])):
                assert len(spans) == len(moments), q.qid
                for span, (a, b) in zip(spans, moments):
                    where, entry = f"{scale} {q.qid} {span}", by_path[span.file]
                    assert abs(entry.t + span.t0 - a) < 1e-3 and abs(entry.t + span.t1 - b) < 1e-3, where
                    assert entry.kind == FILE_OF[planned["template"]] and entry.source == planned["source"], where
                    if planned["template"] in SERVICE_OF:
                        assert span.file == log_path(SERVICE_OF[planned["template"]]), where
                    inside_its_file(plan, entry, files, span, planned["template"], where)
            shows_the_answer(plan, files, q, planned)


def moved(plan, template: str, to: list[list[float]]):
    """The pilot's plan with the one question of `template` that has a needle asked about the moments `to`."""
    q = next(q for q in plan.questions if q["template"] == template and not q["absent"])
    return dataclasses.replace(plan, questions=({**q, "truth": {**q["truth"], "moments": to}},))


def test_a_moment_in_no_file_makes_the_writer_fail(written):
    plan, entries, _ = written["pilot"]
    last = max(f.t + f.spec["length_s"] for f in plan.files if f.kind == "screen")
    first = min(f.t for f in plan.files if f.kind == "screen")
    for to in ([[last + 10.0, last + 40.0]], [[first - 5 * 86400.0, first - 5 * 86400.0 + 30]]):  # after the last segment, before the first
        with pytest.raises(ValueError, match=r"video-screen-error-1: the moment .* is in no screen file of the screen source"):
            questions_of(moved(plan, "screen-error", to), entries)
    segments = sorted((f for f in plan.files if f.kind == "screen"), key=lambda f: f.t)
    away = next(a.t + a.spec["length_s"] for a, b in zip(segments, segments[1:]) if b.t > a.t + a.spec["length_s"] + 60)  # between two blocks
    with pytest.raises(ValueError, match="is in no screen file"):
        questions_of(moved(plan, "screen-error", [[away + 10, away + 20]]), entries)
    note = next(q for q in plan.questions if q["template"] == "note-code" and not q["absent"])["truth"]["moments"][0][0]
    with pytest.raises(ValueError, match="is in no note file"):  # a note is an instant: a second later is no note
        questions_of(moved(plan, "note-code", [[note + 1.0, note + 1.0]]), entries)
    questions_of(moved(plan, "note-code", [[note, note]]), entries)


def test_the_kind_the_hint_the_group_and_the_scope_are_what_the_plan_says(written):
    plan, _, questions = written["pilot"]
    kinds = {"text-note-code-1": "find", "text-config-asof-1": "find", "text-absent-note-1": "absent", "records-log-count-1": "count",
             "records-log-last-1": "when", "image-photo-count-2": "count", "image-photo-sign-3": "last", "video-screen-error-1": "when",
             "video-screen-error-3": "last", "audio-mic-alarm-2": "when", "audio-absent-doorbell-1": "absent"}
    by_qid = {q.qid: q for q in questions}
    assert {qid: by_qid[qid].kind for qid in kinds} == kinds
    for q, planned in zip(questions, plan.questions):
        assert q.hint == planned["level"] and q.scenario == "V7" and q.attachments == () and q.options is None and q.horizon is None
        assert q.scope.sources == (planned["source"],) and (q.scope.since, q.scope.until) == (planned["since"], planned["until"]), q.qid
        assert q.cost_ranges == ((planned["since"], planned["until"]),), q.qid
        assert q.group == BASE_OF.get(planned["template"], planned["template"]), "an absent question is grouped with its needle's template"
        assert (q.kind == "absent") is planned["absent"] and (q.kind == "last") is (planned["last"] and not planned["absent"]), q.qid


def test_the_grade_is_the_plans_with_the_tolerance_in_seconds(written):
    plan, _, questions = written["pilot"]
    for q, planned in zip(questions, plan.questions):
        g = planned["grade"]
        assert (q.grade.rule, q.grade.truth, q.grade.time, q.grade.date) == (g["kind"], g["truth"], g["time"], g["date"]), q.qid
        assert q.grade.tol_s == (None if g["tol_min"] is None else 60 * g["tol_min"]), q.qid
        assert (q.grade.wrong, q.grade.ignore) == (tuple(g["wrong"]), tuple(g["ignore"])), q.qid
    by_qid = {q.qid: q for q in questions}
    assert by_qid["video-screen-error-1"].grade.tol_s == 60.0 and by_qid["video-porch-parcel-1"].grade.tol_s == 120.0, "1 and 2 minutes"


def test_the_tags_are_the_definitions_of_the_plan_recomputed(written):
    """`lookback_h`: hours from the end of the latest moment of the truth to `now`; `certificate_s`: the moments added up; both None
    without a needle. `now` is 12:00 the day after the last day."""
    for scale, (plan, _, questions) in written.items():
        now = datetime.combine(plan.days[-1] + timedelta(days=1), time(12)).timestamp()  # noon, local
        for q, planned in zip(questions, plan.questions):
            moments = planned["truth"]["moments"]
            assert datetime.fromisoformat(q.now).timestamp() == now and datetime.fromisoformat(q.now).utcoffset() is not None
            assert q.tags == {"modality": planned["type"], "template": planned["template"], "level": planned["level"], "last": planned["last"],
                              "absent": planned["absent"], "rung": scale, "first_day": plan.days[0].isoformat(), "last_day": plan.days[-1].isoformat(),
                              "lookback_h": round((now - max(b for _, b in moments)) / 3600, 1) if moments else None,
                              "certificate_s": round(sum(b - a for a, b in moments), 3) if moments else None}, (scale, q.qid)


def test_the_split_is_dev_for_the_seeds_1_to_10_and_test_for_the_others():
    days = [date(2026, 10, 30), date(2026, 10, 31)]
    got = {}
    for seed in (0, 1, 10, 11):
        plan = make_plan(seed, "tiny", days)
        got[seed] = {q.split for q in questions_of(plan, entries_of(plan, made_up))}
    assert got == {0: {"test"}, 1: {"dev"}, 10: {"dev"}, 11: {"test"}}


def test_the_retrieval_run_asks_the_oracle_question_of_the_plan_over_its_source_and_range(written):
    plan, _, questions = written["pilot"]
    counts = {"image-photo-count-1", "image-photo-count-2"}
    for q, planned in zip(questions, plan.questions):
        if planned["oracle"] is None:
            assert q.retrieval is None, q.qid
            continue
        spec = q.retrieval
        want = 5 if planned["last"] else 50 if q.qid in counts else 20  # the newest few of a `last` question, a count's, the others'
        assert (spec.query, spec.sources, spec.ranges, spec.k, spec.tol_s) == (planned["oracle"], (planned["source"],),
                                                                                  ((planned["since"], planned["until"]),), want, 10.0), q.qid
        assert spec.oldest_first is (planned["fake"]["oldest_first"] and not planned["last"]), q.qid
        assert spec.moments == tuple(map(tuple, planned["truth"]["moments"])) and spec.others == tuple(map(tuple, planned["truth"]["others"]))
    assert [q.qid for q in questions if q.retrieval is None] == ["text-config-asof-1"], "only the version asked at a time has no oracle"


@pytest.mark.parametrize("path", sorted(REPLAYS.rglob("result.json")), ids=lambda p: "-".join(p.relative_to(REPLAYS).parts[:3]))
def test_the_recorded_answers_are_graded_as_the_pilot_graded_them(path, written):
    """The pilot's verdict on an answer is `grade(...) and no problem`: the question written from the plan, graded by its `Grade`, gives it
    back for all 8 results kept (a count, a number, a time and an absent question, with the files and with mm)."""
    result = json.loads(path.read_text(encoding="utf-8"))
    question = next(q for q in written["pilot"][2] if q.qid == result["qid"])
    assert (grade.grade(question.grade, result["answer"]) and result["problem"] is None) is result["passed"], (result["answer"], result["passed"])
