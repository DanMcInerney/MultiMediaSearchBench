"""Ported from MultimediaSearch tests/test_bench.py at fe0ece5: the report's, the verdicts' and the retrieval scoring's tests, in that file's order
(the statistics are in test_stats.py, the run loop's are in test_runner.py, and what needs mm's daemon is in tests/mm). The mechanical
changes: `bench.` is the module that has the name now; an arm is a configuration, the two sides of a comparison are `control` and
`tested` in the figures, and the model `fake` is the configuration `mm/fake`; a question is a `Question`; and a run is a `RunFolder`
that the tests write by hand, with results made by `record` where the old tests ran a stub through the runner. The changes that are
more than that are marked `# changed:` where they are made. Expected values are the figures the old tests worked out by hand from the
plan's questions and the stub's answers, not the code's output."""

from __future__ import annotations

import json
import types
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest

from mmsb import scenarios
from mmsb.core import fs, grade, render, report, retrieval, select, stats
from mmsb.core.config import Config, parse_config
from mmsb.core.contract import Moment, RetrievalQuery, RetrievalResult
from mmsb.core.question import Grade, Question, RetrievalSpec, Scope, Truth, write_questions
from mmsb.core.record import record
from mmsb.core.runfolder import COMMIT_KEY, RunFolder, is_done
from mmsb.scenarios.v7_screen.manifest import entries_of
from mmsb.scenarios.v7_screen.questions import questions_of
from mmsb.scenarios.v7_screen.week import plan as make_plan

DAYS = [date(2026, 10, 30), date(2026, 10, 31)]
DEFAULTS = {"files": None, "mm": "d1-3B"}
FILES, FAKE = parse_config("files", DEFAULTS), parse_config("mm/fake", DEFAULTS)
D13, EDGE = parse_config("mm/d1-3B", DEFAULTS), parse_config("mm/d3-edge", DEFAULTS)
ALL = ["audio", "image", "records", "text", "video"]
PASSED = {"status": "passed", "failures": []}


@pytest.fixture(scope="module")
def tiny() -> tuple[Question, ...]:
    """The 19 questions of a tiny week (seed 7, two days), as the V7 scenario writes them, over a manifest of made-up sizes: the report
    reads questions and results and never the data."""
    plan = make_plan(7, "tiny", DAYS)
    return questions_of(plan, entries_of(plan, lambda path: (1, 0, "0" * 64)))


def run_of(root: Path, questions, *, seed: int = 7, configs=(FILES, FAKE), asked=None, probe=PASSED, timeout_s: float = 900.0) -> RunFolder:
    """A run folder with its questions and the run.json and systems.json that `report.numbers` reads, and no results. `asked` has, for
    each tested configuration, the modalities it was asked about (all of them by default); `probe` None leaves the probe out of
    run.json."""
    root.mkdir(parents=True)
    write_questions(root / "questions.jsonl", questions)
    fs.write_json(root / "run.json", {COMMIT_KEY: "abc123", **({"probe": probe} if probe else {})})
    fs.write_json(root / "systems.json", {c.name: {"system": c.system, "variant": c.variant, "dir": c.dir, "readable": ALL,
                                                   "asked": (asked or {}).get(c.name, ALL)}
                                          for c in configs if c.system not in (None, "files")})
    return RunFolder(root, seed, "tiny", tuple(DAYS), root / "data", root, "tag", timeout_s, scenarios.get("v7_screen"), tuple(configs))


def reseeded(run: RunFolder, tmp_path: Path, seed: int) -> RunFolder:
    """Another seed's run of the same questions, with no results yet."""
    return run_of(tmp_path / f"seed-{seed}", run.questions(), seed=seed)


def right(q: Question) -> str:
    """The answer that is right for a question (an absent one is `unknown`; a week question names its day)."""
    g = q.grade
    if g.rule == "absent":
        return "unknown"
    if g.rule in ("number", "count"):
        return g.truth
    answer = g.time[:5] if g.rule == "time" else f"{g.truth} at {g.time[:5]}"
    return f"{answer} on {g.date}" if g.date else answer


def shape(questions) -> types.SimpleNamespace:
    """How many questions of each kind a plan has, so that a test does not hold the number of its week: `n` all of them, `texts`
    those of text and records (the controls), `media` the rest, `absent_media` and `needles` (media with something to find),
    `by_type` the count of each type and `absent_of` the absent ones of each."""
    media = [q for q in questions if q.tags["modality"] in select.MEDIA]
    return types.SimpleNamespace(
        n=len(questions), texts=sum(1 for q in questions if q.tags["modality"] in ("text", "records")), media=len(media),
        absent_media=sum(1 for q in media if q.tags["absent"]), needles=sum(1 for q in media if not q.tags["absent"]),
        by_type={t: sum(1 for q in questions if q.tags["modality"] == t) for t in retrieval.TYPES},
        absent_of={t: sum(1 for q in questions if q.tags["modality"] == t and q.tags["absent"]) for t in retrieval.TYPES})


def stub_out(config: Config, q: Question) -> dict:
    """What the old test's stub agent returned: the right answer for text and records in both configurations and for every type with
    mm, `unknown` for pictures, video and sound without it."""
    mm = config.system == "mm"
    answer = "unknown" if not mm and q.tags["modality"] in ("image", "video", "audio") else right(q)
    tools = ([{"tool": "ask", "arguments": {"question": q.question}}, {"tool": "find", "arguments": {"scope": "all"}}]
             if mm else [{"tool": "shell", "arguments": {"command": "ls"}}])
    return {"answer": answer, "final": f"ANSWER: {answer}", "tools": tools, "seconds": 1.5, "problem": None, "turns": 2,
            "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
            "model": "stub", "denials": 0}


def stub_results(run: RunFolder, questions, *, by=None, only=None) -> None:
    """The stub's results of one agent in the files and mm/fake configurations, made by `record` as the runner makes them.
    `by[(config name, qid)]` changes the fields of that exec's output, and `only[config name]` says which questions that
    configuration is run over (all of them by default)."""
    for config in (FILES, FAKE):
        for q in (only or {}).get(config.name, questions):
            out = {**stub_out(config, q), **(by or {}).get((config.name, q.qid), {})}
            fs.write_json(run.result_path("claude", config, q.qid),
                          record(q, "claude", config, 7, f"execs/claude/{config.dir}/{q.qid}", "2026-11-02T12:00:00", out))


@pytest.fixture(scope="module")
def ran(tiny, tmp_path_factory) -> RunFolder:
    """The stub over the tiny week: the results of one agent in the files and mm/fake configurations, made by `record` as the runner
    makes them, and the one ask label with no project that the old test's mm session left."""
    run = run_of(tmp_path_factory.mktemp("ran") / "run", tiny)
    stub_results(run, tiny)
    fs.write_json(run.folder / "results" / "claude" / FAKE.dir / "ambient-labels.json", [{"id": "id:bench-0", "project": None}])
    return run


def test_the_report_counts_as_it_should(tiny, ran, tmp_path):
    k = shape(tiny)
    assert all(grade.grade(q.grade, right(q)) for q in tiny), "the stub's right answers pass the grader"
    assert not any(grade.grade(q.grade, "unknown") for q in tiny if not q.tags["absent"])
    figures = report.numbers([ran])
    model = figures["agents"]["claude"]["mm/fake"]
    m = model["all"]
    assert (m["n"], model["questions"], model["left_out"], model["pending"]) == (k.n, k.n, 0, 0)
    right_without = k.texts + k.absent_media  # the text and records, and the absent media questions
    assert (m["control"]["correct"], m["tested"]["correct"]) == (right_without, k.n), "files: the controls, and `unknown` for what is absent"
    assert (m["b"], m["c"], m["p"]) == (k.needles, 0, stats.mcnemar(k.needles, 0)) and m["diff"] == pytest.approx(k.needles / k.n)
    assert m["ci"][0] > 0
    absent_all = sum(k.absent_of.values())
    for arm in ("control", "tested"):
        assert (m[arm]["absent"], m[arm]["absent_answered"], m[arm]["needles"]) == (absent_all, 0, k.n - absent_all)
    assert (m["control"]["abstained"], m["tested"]["abstained"]) == (k.needles, 0), "without mm the media needles got `unknown`"
    for type_ in ("image", "video", "audio"):
        assert model["types"][type_] == {"n": k.by_type[type_], "control": k.absent_of[type_], "tested": k.by_type[type_]}
    for type_ in ("text", "records"):
        assert model["types"][type_] == {"n": k.by_type[type_], "control": k.by_type[type_], "tested": k.by_type[type_]}
    assert (model["scope_all"], m["tested"]["median_calls"], m["control"]["median_calls"]) == (k.n, 2, 1)
    assert m["control"]["tokens"]["input"] == 1000 * k.n and m["control"]["median_s"] == 1.5
    assert m["tested"]["cost_usd"] == pytest.approx(0.1 * k.n) and model["kinds"] == {"control": {}, "tested": {}}

    # the media row: pictures, video and sound pooled, the part that depends on the model (the absent ones are right without)
    media = model["media"]
    assert (media["n"], media["control"]["correct"], media["tested"]["correct"], media["b"], media["c"]) ==         (k.media, k.absent_media, k.media, k.needles, 0)
    assert media["diff"] == pytest.approx(k.needles / k.media) and media["p"] == stats.mcnemar(k.needles, 0) < 0.05
    totals = figures["totals"]
    assert totals["results"] == 2 * k.n and totals["problems"] == {} and totals["to_rerun"] == 0
    assert totals["ambient_labels"] >= 1, "the stub's ask with no project"
    assert figures["agent_models"] == {"claude": {"stub": 2 * k.n}}
    text = render.markdown(figures)
    pct = lambda right_, n: f"{right_}/{n} ({100 * right_ / n:.0f}%)"  # noqa: E731
    by_type = [f"{k.absent_of[t]}/{k.by_type[t]} -> {k.by_type[t]}/{k.by_type[t]}" for t in ("image", "video", "audio")]
    for line in ("## claude", f"| fake | all | {k.n}/{k.n} |", f"| fake | media | {k.media} |", pct(right_without, k.n), pct(k.n, k.n),
                 f"{100 * (k.n - right_without) / k.n:+.0f} pts", f"{100 * k.needles / k.media:+.0f} pts", f"{k.needles} / 0",
                 render.chance(media["p"]), "## By data type", *by_type, "## The decider tier", "problems (files / mm)",
                 "would not be recorded at all", "mcnemar(6, 0) = 0.031", "mcnemar(5, 0) = 0.0625", "Runs pooled: 1; seeds 7",
                 f"Agents: claude stub ({2 * k.n} results).",
                 "Ask labels with no project", f"leak into another's scope): {totals['ambient_labels']}.",
                 "Problems by kind: none", "Multiple comparisons: this report has k = 1 agent-model rows",
                 "(2 p values). At p < 0.05 about 0.05 of k rows would fall below it by chance alone; 2 of the 2 p values here are."):
        assert line in text, line

    # runs are pooled by question: another seed adds new questions (here with no results yet)
    other = reseeded(ran, tmp_path, 8)
    pooled = report.numbers([ran, other])
    assert [r["seed"] for r in pooled["runs"]] == [7, 8]
    assert (pooled["agents"]["claude"]["mm/fake"]["all"]["n"], pooled["agents"]["claude"]["mm/fake"]["pending"]) == (k.n, k.n)
    assert "Runs pooled: 2; seeds 7, 8" in render.markdown(pooled)


def test_the_report_counts_each_agents_model_and_effort():
    """Results of two settings pooled by mistake show in the report's Agents line."""
    medium, high = {"agent_model": "terra", "agent_effort": "medium"}, {"agent_model": "terra", "agent_effort": "high"}
    loaded = [(None, {"codex": {"files": {"a": medium, "b": high}, "mm-d1-3b": {"a": medium}}, "claude": {"files": {"a": {}}}})]
    assert report.agent_models(loaded) == {"claude": {"unknown": 1}, "codex": {"terra at high effort": 1, "terra at medium effort": 2}}


def test_a_timeout_a_cap_and_an_escape_are_wrong_answers_that_are_paired(tiny, tmp_path):
    """The report's half of the old test (that none of the three stops the agent, and that they are not asked again, is the
    runner's: test_runner.py)."""
    kinds = {"image-photo-sign-1": ("timeout", "timeout: no answer in 900 s"), "video-screen-error-1": ("capped", "capped: max turns"),
             "audio-mic-doorbell-1": ("escaped", "escaped: Read reached the answers")}
    run = run_of(tmp_path / "run", tiny)
    stub_results(run, tiny, by={(FAKE.name, qid): {"problem": problem, "answer": None} for qid, (_, problem) in kinds.items()})
    k = shape(tiny)
    for qid, (kind, _) in kinds.items():
        saved = fs.read_json(run.result_path("claude", FAKE, qid))
        assert saved["problem"].startswith(kind) and saved["passed"] is False and is_done(saved), qid

    figures = report.numbers([run])
    model = figures["agents"]["claude"]["mm/fake"]
    assert (model["all"]["n"], model["left_out"], model["pending"]) == (k.n, 0, 0), "the three are paired, not dropped"
    assert (model["all"]["control"]["correct"], model["all"]["tested"]["correct"]) == (k.texts + k.absent_media, k.n - 3)
    assert (model["all"]["b"], model["all"]["c"]) == (k.needles - 3, 0)
    assert (model["media"]["n"], model["media"]["tested"]["correct"], model["media"]["b"]) == (k.media, k.media - 3, k.needles - 3)
    assert model["kinds"] == {"control": {}, "tested": {"timeout": 1, "capped": 1, "escaped": 1}}
    assert figures["totals"]["problems"] == {"timeout": 1, "capped": 1, "escaped": 1} and figures["totals"]["to_rerun"] == 0
    text = render.markdown(figures)
    assert "none / capped 1, escaped 1, timeout 1" in text and "Problems by kind: capped 1, escaped 1, timeout 1." in text


def test_a_session_that_was_not_the_one_asked_for_is_left_out_of_the_pairs_until_it_is_run_again(tiny, tmp_path):
    """The report's half of the old test (the agent stopping at the first setup problem, and the next command asking it again with
    the rest, in `until` order, is the runner's)."""
    run = run_of(tmp_path / "run", tiny)
    n, first = shape(tiny).n, tiny[0].qid
    stub_results(run, tiny, only={FILES.name: tiny, FAKE.name: tiny[:1]},
                 by={(FAKE.name, first): {"problem": "setup: the session has tools the arm does not", "answer": None}})
    saved = fs.read_json(run.result_path("claude", FAKE, first))
    assert saved["problem"].startswith("setup") and saved["passed"] is False and not is_done(saved)

    figures = report.numbers([run])
    model = figures["agents"]["claude"]["mm/fake"]
    assert (model["all"]["n"], model["left_out"], model["pending"]) == (0, 1, n - 1), "left out of the pairs, not a wrong answer"
    assert model["kinds"]["tested"] == {"setup": 1} and model["all"]["diff"] is None and model["all"]["p"] is None
    assert f"| fake | all | 0/{n} | -" in render.markdown(figures) and figures["totals"]["to_rerun"] == 1

    # run again, with the rest: every question is paired
    stub_results(run, tiny, only={FILES.name: []})
    assert report.numbers([run])["agents"]["claude"]["mm/fake"]["all"]["n"] == n


def test_a_list_another_systems_session_closed_with_does_not_stop_the_report_or_change_it(tiny, tmp_path):
    run = run_of(tmp_path / "run", tiny)
    stub_results(run, tiny)
    figures = report.numbers([run])
    fs.write_json(run.folder / "results" / "claude" / FAKE.dir / "leftovers.json", [{"x": 1}])  # (a list, where a result is an object)
    assert report.numbers([run]) == figures


def test_a_control_result_to_run_again_is_left_out_of_the_pairs_as_a_tested_one_is(tiny, tmp_path):
    run = run_of(tmp_path / "run", tiny)
    first = tiny[0].qid
    stub_results(run, tiny, by={(FILES.name, first): {"problem": "setup: the session is not the one asked for", "answer": None}})
    assert not is_done(fs.read_json(run.result_path("claude", FILES, first)))
    model = report.numbers([run])["agents"]["claude"]["mm/fake"]
    assert (model["all"]["n"], model["left_out"], model["pending"]) == (shape(tiny).n - 1, 1, 0), "left out, not pending: it ran and must run again"
    assert model["kinds"]["control"] == {"setup": 1}


T0 = datetime(2026, 10, 30, 12, 0).timestamp()
DAY = 86400.0


def sq(qid: str, level: str = "part", *, last: bool = False, absent: bool = False, moments=None, others=(), oracle: str | None = "yes?",
       oldest: bool = False, since: float = T0, k: int = 20) -> Question:
    """A stub question with the fields the report reads; `qid` is `type-template-n`. Its retrieval run asks for `k` sightings."""
    type_, *rest = qid.split("-")
    template = "-".join(rest[:-1])
    moments = [] if absent else (moments or [[T0 + 3600, T0 + 3600]])
    until = T0 + 6 * 3600
    spec = None if oracle is None else RetrievalSpec(oracle, ("phone",), ((since, until),), k, oldest, 10.0,
                                                    tuple((a, b) for a, b in moments), tuple((a, b) for a, b in others))
    return Question(qid=qid, scenario="V7", split="dev", now="2026-10-31T12:00:00-04:00", question=f"Question {qid}?", attachments=(),
                    kind="absent" if absent else "when", hint=level, truth=Truth(None, (), ()),
                    grade=Grade("absent" if absent else "time", None, "13:00", 60.0, (), (), None),
                    tags={"modality": type_, "template": template, "level": level, "last": last, "absent": absent}, options=None,
                    scope=Scope(("phone",), since, until), group=template, horizon=None, cost_ranges=((since, until),), retrieval=spec)


def asked(sightings=(), candidates=(), windows=40, seconds=2.5, stopped="range_end") -> RetrievalResult:
    """What a system gives back for a retrieval query: sightings `(t0, t1, p)` and candidates `(t, p)`, and how far it got."""
    return RetrievalResult(tuple(Moment(a, b, p, "phone", {"p": p}) for a, b, p in sightings),
                           tuple(Moment(t, t, p, "phone", {"p": p}) for t, p in candidates), windows, seconds, 1.25, stopped, ())


def scored(q: Question, result: RetrievalResult, config: Config = FAKE) -> dict:
    """The record of the retrieval run of `q` that the system gave `result` for."""
    return retrieval.retrieval_record(q, config, retrieval.query_of(q), result)


def test_an_oracle_ask_follows_the_level_rules_and_records_the_needles_p_and_moments(tiny):
    truth, other = T0 + 3600, T0 + 3600 - 2 * DAY

    # a `last` question asks for LAST_LIMIT sightings (5), newest first, with the 1,800 s any oracle ask is allowed; a sighting of
    # another needle of the template is no false sighting, and the needle's p is the best near the truth over sightings and candidates
    last = sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]], others=[[other, other]], since=T0 - 5 * DAY, k=5)
    got = scored(last, asked([(other, other, 0.91), (truth + 4, truth + 4, 0.83), (truth + 5000, truth + 5000, 0.9)],
                             [(truth + 3, 0.7), (truth + 500, 0.95), (T0 + 10, 0.99)], stopped="limit"))
    # changed: the ask's parameters are the query (the translation to a system's own parameters is the system's, tested in tests/mm)
    assert retrieval.query_of(last) == RetrievalQuery("image-photo-sign-3", "yes?", ("phone",), ((T0 - 5 * DAY, T0 + 6 * 3600),), 5, 5,
                                                      False, 1800.0)
    assert (got["level"], got["last"], got["limit"]) == ("week", True, 5)
    assert (got["found"], got["top5"], got["p"]) == (True, True, 0.83), "the best p within 10 s of the truth, not 0.95 or 0.99"
    assert (got["false"], got["in_range"], got["others"], got["others_found"]) == (1, 3, 1, 1)
    assert (got["moments"], got["moments_found"], got["stopped"], got["windows"], got["seconds"]) == (1, [0], "limit", 40, 2.5)
    assert got["rank"] == 2, "all three are in range; by end time truth+5000 (false) is first, truth+4 (the needle) second, `other` third"

    # the rank counts in-range sightings only: a false one newer than the needle makes it 2, one past `until` does not count
    late = T0 + 6 * 3600 + 100
    result = asked([(late, late, 0.9), (truth + 2000, truth + 2000, 0.8), (truth + 2, truth + 2, 0.7)], stopped="limit")
    got = scored(sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]]), result)
    assert (got["found"], got["rank"], got["in_range"], got["false"]) == (True, 2, 2, 1)
    # the newest sighting is the needle: 1; sightings given oldest first are ranked by their end time all the same
    got = scored(sq("image-photo-sign-3", "week", last=True), asked([(truth - 900, truth - 900, 0.8), (truth + 1, truth + 1, 0.9)]))
    assert (got["found"], got["rank"], got["false"]) == (True, 1, 1)
    # no needle among the sightings (false ones, a sighting of an earlier needle): no rank, and the false ones are told
    result = asked([(truth + 3000, truth + 3000, 0.9), (other, other, 0.8), (truth + 2000, truth + 2000, 0.7)], stopped="limit")
    got = scored(sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]], others=[[other, other]], since=T0 - 5 * DAY), result)
    assert (got["found"], got["rank"], got["false"], got["in_range"], got["stopped"]) == (False, None, 2, 3, "limit")
    assert retrieval.outcome(got) == "stopped"

    # the others (part, day, a count): `oldest_first` kept; a needle only a candidate names
    part = sq("video-screen-error-1", oldest=True, moments=[[truth, truth]])
    got = scored(part, asked([], [(truth - 6, 0.68), (T0 + 99, 0.4)]))
    assert (retrieval.query_of(part).oldest_first, retrieval.query_of(part).max_s, retrieval.query_of(part).candidates) == (True, 1800, 5)
    assert (got["found"], got["top5"], got["p"], got["moments_found"], got["false"]) == (False, True, 0.68, [], 0)
    count = sq("image-photo-count-1", "day", moments=[[truth, truth], [truth + 600, truth + 600], [truth + 900] * 2], k=50)
    got = scored(count, asked([(truth, truth, 0.9), (truth + 900, truth + 900, 0.9)]))
    assert retrieval.query_of(count).k == 50 and not retrieval.query_of(count).oldest_first
    assert (got["found"], got["moments"], got["moments_found"], got["p"]) == (True, 3, [0, 2], 0.9)

    # a needle the ask never names has no p; an absent question has none, and any sighting in range is false
    got = scored(sq("video-porch-parcel-1"), asked([], [(T0 + 99, 0.4)]))
    assert (got["found"], got["top5"], got["p"], got["rank"]) == (False, False, None, None)
    got = scored(sq("video-absent-error-1", absent=True), asked([(T0 + 50, T0 + 60, 0.9)]))
    assert (got["found"], got["top5"], got["p"], got["false"], got["moments_found"], got["rank"]) == (None, None, None, 1, [], None)

    # changed: the level rules (5 sightings for a `last` question, 50 for a count, 20 for the rest) are the V7 writer's now, and are
    # read here from the questions it writes for a real tiny week (the last ones are those the plan marks `last`)
    k_of = {q.qid: retrieval.query_of(q).k for q in tiny if q.retrieval}
    assert {qid for qid, k in k_of.items() if k == 5} == {q.qid for q in tiny if q.tags["last"] and q.retrieval}
    assert {qid for qid, k in k_of.items() if k == 50} == {"image-photo-count-1"}
    assert {k for qid, k in k_of.items() if qid not in {q.qid for q in tiny if q.tags["last"]} | {"image-photo-count-1"}} == {20}
    assert len(k_of) == 18, "every question but the config version has a retrieval run"


def level_of_pairs(files_correct, mm_correct, n, files_s=100.0, mm_s=100.0, files_k=1000.0, mm_k=1000.0, b=None, c=None) -> dict:
    """A level as `by_level` gives it, from the counts: n pairs, each side's correct answers, the seconds and the tokens an exec
    spends (so a side's cost per correct answer is what its correct ones divide)."""
    def arm(correct, s, k):
        return {"correct": correct, "n": n, "accuracy": correct / n, "seconds": s * n if s is not None else 0, "median_s": s,
                "tokens_all": k * n if k is not None else None, "tokens_n": n if k is not None else 0, "tokens_mean": k}

    b = max(mm_correct - files_correct, 0) if b is None else b
    c = max(files_correct - mm_correct, 0) if c is None else c
    both_right = files_correct - c
    return {"n": n, "n_control": n, "diff": (b - c) / n, "ci": stats.interval(both_right, b, c, n - both_right - b - c), "b": b, "c": c,
            "control": arm(files_correct, files_s, files_k), "tested": arm(mm_correct, mm_s, mm_k)}


def test_the_crossover_is_the_first_level_with_enough_pairs_where_mm_leads_by_20_points_or_costs_half_without_being_worse():
    level = level_of_pairs
    # nothing: 10 more right out of 20 is no lead, a level of 10 pairs or more is needed, and an exact time is no level of the hint
    assert report.crossover({"part": level(10, 11, 20), "day": level(10, 11, 20), "week": level(10, 11, 20)}) is None
    assert report.crossover({"part": level(0, 5, 9)}) is None and report.thin_levels({"part": level(0, 5, 9), "day": level(0, 5, 20)}) == {"part": 9}
    assert report.crossover({"time": level(0, 20, 20, 10, 1)}) is None and report.crossover({}) is None
    assert report.crossover({"part": {**level(1, 1, 10), "n": 0}}) is None and report.thin_levels({"part": {**level(1, 1, 10), "n": 0}}) == {}
    # a lead of 20 points is a lead (0.8 - 0.6 is 0.20000000000000007, and 0.6 - 0.4 is 0.1999...96)
    assert report.crossover({"part": level(12, 16, 20)})["by"] == ["accuracy"]
    assert report.crossover({"part": level(10, 11, 20), "day": level(8, 12, 20)})["level"] == "day"
    got = report.crossover({"part": level(10, 14, 20), "day": level(5, 15, 20)})
    assert (got["level"], got["by"], got["lead"]) == ("part", ["accuracy"], pytest.approx(0.2))
    assert got["seconds"] == pytest.approx(10 / 14) == pytest.approx(got["tokens"]), "per correct answer: 14 right for what 10 cost"
    # the cost branch: half the seconds, or the tokens, per correct answer, with mm no less accurate
    assert report.crossover({"part": level(10, 10, 20, 100, 90), "day": level(10, 10, 20, 100, 40)})["level"] == "day"
    got = report.crossover({"week": level(10, 10, 20, 100, 100, 1000, 500)})
    assert (got["level"], got["by"], got["lead"], got["seconds"], got["tokens"]) == ("week", ["cost"], 0.0, 1.0, 0.5)
    both = report.crossover({"day": level(8, 16, 20, 100, 40)})
    assert both["by"] == ["accuracy", "cost"] and both["seconds"] == pytest.approx(0.2)
    # per correct answer, not per exec: mm twice as fast with half the answers right costs the same per right answer
    assert report.crossover({"day": level(16, 16, 20, 100, 50), "week": level(16, 8, 20, 100, 50, b=0, c=8)}) ["level"] == "day"
    assert report.crossover({"week": level(16, 8, 20, 100, 50, b=0, c=8)}) is None
    # less accurate and cheaper is no crossover: a difference of -0.40 at a quarter of the cost, and -0.05 with an interval below it
    worse = level(18, 10, 20, 100, 25, 1000, 250, b=0, c=8)
    assert worse["diff"] == pytest.approx(-0.4) and report.crossover({"day": worse}) is None
    assert report.crossover({"day": level(10, 10, 20, 100, 25, b=1, c=2)}) is None, "-0.05, and too few discordant pairs for an interval"
    close = level(160, 158, 200, 100, 25, b=8, c=10)
    assert close["diff"] < 0 and close["ci"][0] > -report.NOT_WORSE and report.crossover({"day": close})["by"] == ["cost"]
    assert report.crossover({"day": level(160, 140, 200, 100, 25, b=0, c=20)}) is None, "-10 points, interval below -0.10"
    # a cost per correct answer needs MIN_CORRECT right answers in each arm, and tokens that every exec gave
    assert report.crossover({"day": level(4, 4, 10, 100, 10)}) is None
    nothing = level(10, 10, 20, None, None, None, None)
    assert report.crossover({"part": nothing}) is None
    # the text says which branch fired, and why there is none
    cost_only = report.crossover({"week": level(10, 10, 20, 100, 100, 1000, 500)})
    assert render.crossover_text(cost_only, "mm") == "`week` by cost (accuracy +0 pts, 1.00x the seconds per correct answer, 0.50x the tokens per correct answer)"
    assert render.crossover_text(report.crossover({"day": level(8, 12, 20)}), "mm").startswith("`day` by accuracy (accuracy +20 pts")
    assert "too few paired questions at part (9)" in render.crossover_text(None, "mm", {"part": 9})
    assert render.crossover_text(None, "mm").startswith("none: at no level with 10 or more paired questions does mm lead")


def test_the_difference_gets_newcombes_interval_only_from_ten_pairs_right_in_one_arm_only_and_mcnemar_speaks_for_the_rest():
    def rows(a, b, c, d):
        out = []
        for mm, files, count in ((True, True, a), (True, False, b), (False, True, c), (False, False, d)):
            def result(passed):
                return {"passed": passed, "absent": False, "answer": "x", "abstained": False, "seconds": 10.0, "calls": 1, "problem": None,
                        "tokens": None, "cost_usd": None, "agent": "codex", "work": None, "tools": {}}
            out += [({"type": "image", "level": "part"}, result(files), result(mm)) for _ in range(count)]
        return out

    # the case that a percentile bootstrap got wrong: 4 right with mm only, 0 the other way, 16 pairs: [.06, .50] with p = .125
    four = report.paired(rows(6, 4, 0, 6))
    assert (four["b"], four["c"], four["p"], four["ci"], four["diff"]) == (4, 0, 0.125, None, 0.25)
    assert render.diff_cell(four) == "+25 pts (too few discordant pairs: b 4, c 0)"
    assert report.lean(rows(6, 4, 0, 6))["ci"] is None and report.paired(rows(6, 5, 4, 5))["ci"] is None, "9 discordant pairs"
    # from 10: the interval, in points, beside b, c and the p
    ten = report.paired(rows(5, 10, 0, 5))
    assert ten["ci"] == pytest.approx([0.223, 0.675], abs=1e-3) and ten["p"] == pytest.approx(0.00195, abs=1e-4) and ten["diff"] == 0.5
    assert render.diff_cell(ten) == "+50 pts [+22, +67]"
    even = report.paired(rows(10, 5, 5, 10))
    assert even["ci"][0] < 0 < even["ci"][1] and even["p"] == 1.0 and render.diff_cell(even).startswith("+0 pts [-")
    assert report.paired([])["ci"] is None and report.paired([])["diff"] is None and render.diff_cell(report.paired([])) == "-"


def test_a_pooled_difference_is_the_mean_of_the_paired_differences_whatever_the_files_arm_is_counted_once():
    def result(passed):
        return {"passed": passed, "absent": False, "answer": "x", "abstained": False, "seconds": 10.0, "calls": 1, "problem": None,
                "tokens": None, "cost_usd": None, "agent": "codex", "work": None}

    lost, won = result(False), result(True)  # one files result that was wrong, paired with three models; one that was right, with one
    q = {"type": "image", "level": "day"}
    pairs = [(q, lost, result(True)), (q, lost, result(True)), (q, lost, result(False)), (q, won, result(True))]
    got = report.lean(pairs)
    assert (got["n"], got["n_control"]) == (4, 2) and (got["b"], got["c"]) == (2, 0)
    assert got["diff"] == pytest.approx(0.5), "the mean of (+1, +1, 0, 0), not 0.75 - 0.5 from a files side counted once"
    assert got["control"]["accuracy"] == pytest.approx(0.5) and got["tested"]["accuracy"] == pytest.approx(0.75)


def test_claudes_tokens_count_the_cache_it_read_and_wrote_as_codexs_input_already_does():
    claude = {"agent": "claude", "tokens": {"input": 100, "output": 50, "cache_read": 1000, "cache_write": 200}}
    got = {**claude, "passed": True, "seconds": 2.0, "absent": False, "answer": "x", "abstained": False, "calls": 1, "problem": None,
           "cost_usd": None, "work": None}
    side = report.side([got, {**got, "passed": False, "tokens": None}])
    assert (side["tokens_mean"], side["tokens_all"], side["tokens_n"], side["correct"], side["seconds"]) == (1350, 1350, 1, 1, 4.0)
    assert report.per_correct([got])["tokens_k"] == pytest.approx(1.35) and report.per_correct([got, {**got, "tokens": None}])["tokens_k"] is None
    assert render.kilo(side) == "1" and render.kilo(report.side([])) == "-"


def rec(model: str, qid: str, found: bool, top5: bool, p: float | None, *, false: int = 0, level: str = "part",
        stopped: str = "range_end", absent: bool = False, candidates=()) -> dict:
    """A stub retrieval record, of the sort `retrieval_record` writes, of the variant `model` of the system mm. changed: the old
    test's records could leave out `level` and `p` (a record from before them); the format has them now."""
    type_, *rest = qid.split("-")
    return {"qid": qid, "type": type_, "template": "-".join(rest[:-1]), "absent": absent, "config": f"mm/{model}", "system": "mm",
            "variant": model, "question": "?", "source": "phone", "since": T0, "until": T0 + 6 * 3600, "found": found, "top5": top5,
            "in_range": 0, "false": false, "windows": 100, "seconds": 6.0, "wall_s": 6.0, "stopped": stopped, "warnings": [],
            "sightings": [], "candidates": list(candidates), "level": level, "p": p}


def test_a_miss_is_the_haystacks_only_when_nobody_found_the_needle_and_two_families_scanned_it_without_it_in_their_top_5():
    def found(m):
        return rec(m, "video-screen-error-1", True, True, 0.97)

    def near(m):
        return rec(m, "video-screen-error-1", False, True, 0.68)

    def far(m):
        return rec(m, "video-screen-error-1", False, False, None)

    def stopped(m):  # a `last` ask that met its LAST_LIMIT sightings, false ones newer than the needle, before it reached it
        return rec(m, "video-screen-error-1", False, False, None, false=1, stopped="limit")

    def cut(m):
        return rec(m, "video-screen-error-1", False, False, None, stopped="max_s")

    # one model: this tier cannot tell a miss from a haystack fault; a near miss is below its model's hit
    assert retrieval.verdicts([far("d1-3B")]) == {"mm/d1-3B": "one model ran: the agent arm's look, or a second family, tells a miss from a haystack fault"}
    assert retrieval.verdicts([near("d1-3B")]) == {"mm/d1-3B": "below its model's hit (calibration)"}
    assert retrieval.verdicts([found("d1-3B")]) == {}
    # two models of one family are one family
    assert retrieval.verdicts([far("d1-3B"), far("d1-Omni-2B")]) == {n: retrieval.ONE_FAMILY for n in ("mm/d1-3B", "mm/d1-Omni-2B")}
    assert retrieval.ONE_FAMILY.startswith("one family ran:")
    # nobody found it, two families scanned it, neither had it in its top 5: the haystack is suspect
    assert retrieval.verdicts([far("d1-3B"), far("d3-edge")]) == {"mm/d1-3B": "haystack suspect", "mm/d3-edge": "haystack suspect"}
    # a model found it: every miss is that model's miss, of whichever family (a model of the same family included)
    assert retrieval.verdicts([found("d1-3B"), far("d3-edge")]) == {"mm/d3-edge": "this model's miss"}
    assert retrieval.verdicts([found("d1-Omni-2B"), far("d1-3B"), far("d3-edge")]) == {"mm/d1-3B": retrieval.MISS, "mm/d3-edge": retrieval.MISS}
    assert retrieval.verdicts([far("d1-3B"), far("d1-Omni-2B"), found("d3-edge")]) == {"mm/d1-3B": retrieval.MISS, "mm/d1-Omni-2B": retrieval.MISS}
    assert retrieval.verdicts([found("d1-Omni-2B"), far("d1-3B")]) == {"mm/d1-3B": retrieval.MISS}
    # another family had it in its top 5: this model's miss; a model in the top 5 is below its hit
    assert retrieval.verdicts([near("d1-3B"), far("d3-edge")]) == {"mm/d1-3B": "below its model's hit (calibration)", "mm/d3-edge": "this model's miss"}
    # a scan that ended before the needle is told as it is, and it does not make the haystack suspect or count as a family that ran
    assert retrieval.verdicts([stopped("d1-3B"), far("d3-edge")]) == {"mm/d1-3B": retrieval.STOPPED, "mm/d3-edge": retrieval.ONE_MODEL}
    assert retrieval.verdicts([cut("pplx-decider-flash-v1.2"), far("d3-edge"), far("d1-3B")]) == {
        "mm/pplx-decider-flash-v1.2": retrieval.CUT, "mm/d3-edge": retrieval.SUSPECT, "mm/d1-3B": retrieval.SUSPECT}
    assert retrieval.verdicts([stopped("d1-3B"), found("d3-edge")]) == {"mm/d1-3B": retrieval.STOPPED}
    assert retrieval.STOPPED == "stopped on a false sighting before the needle" and retrieval.CUT == "cut at max_s before the needle"
    # the outcomes: a needle with p of 0.5 or more is a near miss even outside the top 5; below 0.5 it is a miss
    assert retrieval.outcome(rec("d1-3B", "video-screen-error-1", False, False, 0.5)) == "near"
    assert retrieval.outcome(rec("d1-3B", "video-screen-error-1", False, False, 0.49)) == "missed"
    assert retrieval.outcome(rec("d1-3B", "video-screen-error-1", False, False, None)) == "missed"
    assert retrieval.outcome(found("d1-3B")) == "found" and retrieval.outcome(stopped("d1-3B")) == "stopped" and retrieval.outcome(cut("d1-3B")) == "cut"
    # the limit met with a sighting that was no false one, or the needle's own candidate in the top 5, are not a scan that ended early
    assert retrieval.outcome(rec("d1-3B", "video-screen-error-1", False, False, None, stopped="limit")) == "missed"
    assert retrieval.outcome(rec("d1-3B", "video-screen-error-1", False, True, 0.3, false=1, stopped="limit")) == "near"


NO_TOKENS = {"passed": True, "seconds": 1.0, "tokens": None, "work": None}


def wk(jobs: int = 0, windows: int = 0, seconds: float = 0.0, stopped: dict | None = None, views: int = 0, shells: int = 0) -> dict:
    return {"jobs": jobs, "windows": windows, "seconds": seconds, "stopped": stopped or {}, "views": views, "shells": shells}


FILES_ARM = {
    "image-photo-sign-1": (True, 40, {"shell": 6, "view_image": 5}, wk(views=5, shells=6), (30000, 2000), None),
    "image-photo-sign-2": (False, 90, {"shell": 10, "view_image": 9}, wk(views=9, shells=10), (60000, 3000), None),
    "image-photo-sign-3": (False, 900, {"shell": 20, "view_image": 14}, wk(views=14, shells=20), (90000, 1000), "timeout: no answer in 900 s"),
    "video-screen-error-1": (True, 30, {"shell": 4, "view_image": 3}, wk(views=3, shells=4), (20000, 1000), None),
    "video-screen-error-2": (True, 80, {"shell": 9, "view_image": 8}, wk(views=8, shells=9), (50000, 2000), None),
    "video-porch-parcel-1": (False, 120, {"shell": 15, "view_image": 12}, wk(views=12, shells=15), (70000, 1500), None),
    "text-note-code-1": (True, 10, {"shell": 2}, wk(shells=2), (8000, 500), None)}


MM_D13 = {
    "image-photo-sign-1": (True, 30, {"ask": 1, "find": 1, "look": 2}, wk(1, 50, 3.0, {"range_end": 1}), (20000, 1000), None),
    "image-photo-sign-2": (True, 20, {"ask": 2, "look": 1}, wk(1, 400, 25.0, {"max_s": 1}), (25000, 1500), None),
    "image-photo-sign-3": (True, 25, {"ask": 1, "look": 3}, wk(1, 90, 6.0, {"limit": 1}), (22000, 1000), None),
    "video-screen-error-1": (True, 35, {"find": 2, "look": 4}, wk(), (40000, 1200), None),
    "video-screen-error-2": (True, 50, {"ask": 1, "look": 2}, wk(1, 200, 12.0, {"range_end": 1}), (30000, 1000), None),
    "video-porch-parcel-1": (False, 60, {"ask": 1}, wk(1, 300, 20.0, {"max_s": 1}), (28000, 500), None),
    "text-note-code-1": (True, 8, {"find": 1}, wk(), (6000, 300), None)}


MM_EDGE = {  # the third has no `work` (a result that does not say); the fourth is the one the old test read from its exec's tools.json
    "image-photo-sign-1": (True, 38, {"ask": 1, "look": 1}, wk(1, 60, 5.0, {"range_end": 1}), (21000, 900), None),
    "image-photo-sign-2": (False, 30, {"ask": 1}, wk(1, 450, 80.0, {"max_s": 1}), (15000, 400), None),
    "video-screen-error-1": (True, 33, {"ask": 1, "look": 2}, None, (20000, 800), None),
    # changed: the old test left `work` out here and the report read it from EDGE_TOOLS in the exec's tools.json (two jobs, 380 windows,
    # 70 GPU s, stopped at max_s and at the limit), which it no longer does; this is what it read
    "video-screen-error-2": (True, 70, {"ask": 3, "look": 1}, wk(2, 380, 70.0, {"max_s": 1, "limit": 1}), (50000, 2000), None)}


DECIDERS = {
    "d1-3B": [rec("d1-3B", "image-photo-sign-1", True, True, 0.96), rec("d1-3B", "image-photo-sign-3", False, False, None, false=1, level="week"),
              rec("d1-3B", "video-screen-error-1", True, True, 0.97), rec("d1-3B", "video-porch-parcel-1", False, False, None, level="week"),
              rec("d1-3B", "video-absent-error-1", None, None, None, absent=True),
              # changed: the old test left out this record's level and p and the report took them from the plan (day, and the p of the candidate
              # at the truth); the record has them now
              rec("d1-3B", "image-photo-sign-2", False, True, 0.68, level="day", candidates=[{"t": T0 + 3602, "p": 0.68}])],
    "d3-edge": [rec("d3-edge", "image-photo-sign-1", True, True, 0.94), rec("d3-edge", "image-photo-sign-2", True, True, 0.90, level="day"),
                rec("d3-edge", "image-photo-sign-3", False, False, None, level="week"),
                rec("d3-edge", "video-screen-error-1", False, False, 0.3)]}


def stub_run(tmp_path) -> RunFolder:
    """A run folder of hand-written results over stub questions that carry the new fields, with a plan that is theirs: two models
    asked the media questions of one agent, and d1-3B also its text control."""
    questions = [sq("image-photo-sign-1", "part"), sq("image-photo-sign-2", "day"),
                 sq("image-photo-sign-3", "week", last=True, others=[[T0 - 2 * DAY] * 2]), sq("video-screen-error-1", "part"),
                 sq("video-screen-error-2", "day"), sq("video-porch-parcel-1", "week", last=True), sq("text-note-code-1", "part")]
    by_qid = {q.qid: q for q in questions}
    run = run_of(tmp_path / "stub-run", questions, seed=1, configs=(FILES, D13, EDGE),
                 asked={D13.name: ["image", "records", "text", "video"], EDGE.name: ["image", "video"]})
    for config, results in ((FILES, FILES_ARM), (D13, MM_D13), (EDGE, MM_EDGE)):
        for qid, (passed, seconds, tools, work, tokens, problem) in results.items():
            q = by_qid[qid]
            out = {"qid": qid, "type": q.tags["modality"], "template": q.tags["template"], "level": q.hint, "last": q.tags["last"],
                   "absent": False, "seed": 1, "agent": "codex", "config": config.name, "system": config.system,
                   "variant": config.variant, "answer": "13:00" if passed else "14:00",
                   "passed": passed, "abstained": False, "problem": problem, "seconds": seconds, "turns": 1, "denials": None,
                   "calls": sum(tools.values()), "tools": tools, "scope_all": 0, "cost_usd": None, "used_percent": 10.0,
                   "tokens": {"input": tokens[0], "output": tokens[1], "cache_read": 0, "cache_write": 0},
                   "exec": f"execs/codex/{config.dir}/{qid}", "at": "2026-11-02T12:00:00", **({"work": work} if work is not None else {}),
                   **({"gpu_estimate_s": (1200.0 if qid == "video-porch-parcel-1" else 400.0) if q.hint == "week" else 20.0,
                       "held_s": 4.0, "exclusive": q.hint == "week", "beyond_timeout": qid == "video-porch-parcel-1"}
                      if config == D13 else {})}
            fs.write_json(run.result_path("codex", config, qid), out)
    for config, records in ((D13, DECIDERS["d1-3B"]), (EDGE, DECIDERS["d3-edge"])):
        folder = run.folder / "results" / retrieval.FOLDER / config.dir
        fs.write_json(folder / "warm.json", {"config": config.name, "seconds": 20.0})
        for r in records:
            fs.write_json(folder / f"{r['qid']}.json", r)
    return run


def test_the_report_gives_the_levels_how_each_arm_worked_and_what_a_correct_answer_cost(tmp_path, monkeypatch):
    run = stub_run(tmp_path)
    thin = report.numbers([run])  # with its two pairs a level, nothing is told of a crossover
    assert thin["agents"]["codex"]["mm/d1-3B"]["crossover"] is None and thin["agents"]["codex"]["mm/d1-3B"]["thin"] == {"part": 2, "day": 2, "week": 2}
    assert "too few paired questions at part (2), day (2), week (2)" in render.markdown(thin)
    monkeypatch.setattr(report, "MIN_LEVEL_N", 2)  # the stub's levels have two pairs each
    figures = report.numbers([run])
    d13, edge = figures["agents"]["codex"]["mm/d1-3B"], figures["agents"]["codex"]["mm/d3-edge"]
    assert (d13["questions"], d13["pending"], d13["all"]["n"], d13["media"]["n"]) == (7, 0, 7, 6)
    assert (edge["questions"], edge["pending"], edge["all"]["n"], edge["media"]["n"]) == (6, 2, 4, 4), "d3-edge is asked media alone"
    assert (d13["media"]["control"]["correct"], d13["media"]["tested"]["correct"], d13["media"]["b"], d13["media"]["c"]) == (3, 5, 2, 0)

    # accuracy, seconds, tokens and timeouts at each level of the hint
    assert list(d13["levels"]) == ["part", "day", "week"]
    part, day, week = d13["levels"]["part"], d13["levels"]["day"], d13["levels"]["week"]
    assert (part["n"], part["control"]["correct"], part["tested"]["correct"], part["control"]["median_s"], part["tested"]["median_s"]) == (2, 2, 2, 35.0, 32.5)
    assert (part["control"]["tokens_mean"], part["tested"]["tokens_mean"]) == (26500, 31100)
    assert (day["control"]["correct"], day["tested"]["correct"], day["diff"], day["control"]["median_s"], day["tested"]["median_s"]) == (1, 2, 0.5, 85.0, 35.0)
    assert (day["control"]["tokens_mean"], day["tested"]["tokens_mean"]) == (57500, 28750)
    assert (week["control"]["correct"], week["tested"]["correct"], week["control"]["timeouts"], week["tested"]["timeouts"]) == (0, 1, 1, 0)
    assert d13["crossover"] == {"level": "day", "by": ["accuracy"], "lead": 0.5, "seconds": None, "tokens": None}, \
        "two right answers in an arm make no cost per correct answer"
    assert (part["b"], part["c"], part["ci"], day["b"], day["c"], day["ci"]) == (0, 0, None, 1, 0, None)
    assert [lv["control"]["correct"] for lv in edge["levels"].values()] == [2, 1] and edge["crossover"] is None
    # pooled over the models: the files results count once (4 pairs at `day`, 2 files results), the mm results of both models
    pooled = figures["pooled"]["codex"]
    assert (pooled["levels"]["day"]["n"], pooled["levels"]["day"]["n_control"], pooled["levels"]["day"]["control"]["correct"],
            pooled["levels"]["day"]["tested"]["correct"], pooled["levels"]["day"]["diff"]) == (4, 2, 1, 3, 0.25)
    assert pooled["levels"]["day"]["tested"]["median_s"] == 40.0 and pooled["levels"]["week"]["n"] == 2
    assert pooled["crossover"]["level"] == "day" and pooled["crossover"]["lead"] == 0.25 and pooled["crossover"]["by"] == ["accuracy"]

    # how each arm worked, per exec on the paired media questions
    mm, files = d13["work"]["tested"], d13["work"]["control"]
    assert (mm["n"], mm["ask_execs"], mm["asks"], mm["finds"], mm["looks"], mm["known"]) == (6, 5, 1.0, 0.5, 2.0, 6)
    assert (mm["windows"], mm["gpu_s"], mm["max_s"], mm["views"]) == (pytest.approx(1040 / 6), 11.0, 2, 0.0)
    assert (files["n"], files["ask_execs"], files["views"], files["shells"]) == (6, 0, 8.5, pytest.approx(64 / 6))
    # a result with no `work` is unknown (changed: the report no longer makes it up from the exec's tools.json): 3 of the 4 say it
    slim = edge["work"]["tested"]
    assert (slim["n"], slim["known"], slim["ask_execs"], slim["asks"], slim["looks"]) == (4, 3, 4, 1.5, 1.0)
    assert (slim["windows"], slim["gpu_s"], slim["max_s"]) == (pytest.approx(890 / 3), pytest.approx(155 / 3), 2)

    # seconds, tokens (k) and GPU seconds per correct answer: the GPU seconds only when every result says
    assert d13["cost"]["control"]["s"] == 420.0 and d13["cost"]["control"]["tokens_k"] == pytest.approx(330.5 / 3)
    assert (d13["cost"]["tested"]["s"], d13["cost"]["tested"]["tokens_k"], d13["cost"]["tested"]["gpu_s"]) == (44.0, pytest.approx(34.24), 13.2)
    assert (edge["cost"]["tested"]["correct"], edge["cost"]["tested"]["s"], edge["cost"]["tested"]["gpu_s"]) == (3, 57.0, None)
    assert report.per_correct([])["s"] is None and report.per_correct([{**NO_TOKENS, "passed": False}])["tokens_k"] is None

    # one row per question, in the plan's order: the files arm's cell and each model's (none where it was not asked)
    rows = figures["appendix"]["codex"]
    assert [r["qid"] for r in rows][:3] == ["image-photo-sign-1", "image-photo-sign-2", "image-photo-sign-3"] and len(rows) == 7
    assert rows[0] == {"seed": 1, "qid": "image-photo-sign-1", "level": "part", "type": "image",
                       "control": {"passed": True, "seconds": 40, "problem": None, "beyond": False, "views": 5, "asks": 0},
                       "tested": {"mm/d1-3B": {"passed": True, "seconds": 30, "problem": None, "beyond": False, "views": 0, "asks": 1},
                              "mm/d3-edge": {"passed": True, "seconds": 38, "problem": None, "beyond": False, "views": 0, "asks": 1}}}
    assert rows[2]["control"]["problem"] == "timeout" and list(rows[2]["tested"]) == ["mm/d1-3B"] and list(rows[6]["tested"]) == ["mm/d1-3B"]
    assert rows[5]["tested"]["mm/d1-3B"]["passed"] is False and rows[5]["tested"]["mm/d1-3B"]["beyond"] is True and mm["beyond"] == 1
    json.dumps(figures)

    # the markdown holds each table
    text = render.markdown(figures)
    for line in (
            "## codex: by how vague the time hint is (the media questions, paired)",
            "| model | level | paired | without mm | with mm | difference (95% CI) | b / c | McNemar p | median s (files / mm) | "
            "tokens k per exec (files / mm) | timed out (files / mm) |",
            "| d1-3B | part | 2 | 2/2 (100%) | 2/2 (100%) | +0 pts (too few discordant pairs: b 0, c 0) | 0 / 0 | 1.000 | 35.0 / 32.5 | 26 / 31 | 0 / 0 |",
            "| d1-3B | day | 2 | 1/2 (50%) | 2/2 (100%) | +50 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 85.0 / 35.0 | 58 / 29 | 0 / 0 |",
            "| d1-3B | week | 2 | 0/2 (0%) | 1/2 (50%) | +50 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 510.0 / 42.5 | 81 / 26 | 1 / 0 |",
            "| all models | day | 4 | 1/2 (50%) | 3/4 (75%) | +25 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 85.0 / 40.0 | 58 / 31 | 0 / 0 |",
            "- d1-3B: `day` by accuracy (accuracy +50 pts)",
            "- d3-edge: none: at no level with 2 or more paired questions does mm lead by 20 points, or cost half as much per correct answer",
            "- all models: `day` by accuracy (accuracy +25 pts)",
            "## How each arm worked (the media questions, per exec)",
            "| agent | model | arm | execs | called ask | asks / finds / looks | ask windows | ask GPU s | asks stopped at max_s | "
            "GPU s estimated | held s (run alone) | estimated past the timeout | pictures viewed | shell commands |",
            "| codex | d1-3B | files | 6 | - | - | - | - | - | - | - | - | 8.5 | 10.7 |",
            "| codex | d1-3B | mm | 6 | 5/6 (83%) | 1.0 / 0.5 / 2.0 | 173.3 | 11.0 | 2 | 280.0 | 4.0 (2 alone) | 1 | - | - |",
            "| codex | d3-edge | mm | 4 | 4/4 (100%) | 1.5 / 0.0 / 1.0 | 296.7 (3/4) | 51.7 (3/4) | 2 | - | - | 0 | - | - |",
            "## Cost per correct answer (the media questions)",
            "| agent | model | arm | correct (k) | seconds | tokens k | GPU seconds |",
            "| codex | d1-3B | files | 3/6 | - | - | - |", "| codex | d1-3B | mm | 5/6 | 44.0 | 34.2 | 13.2 |",
            "| codex | d3-edge | mm | 3/4 | - | - | - |", "| codex | d3-edge | files | 3/4 | - | - | - |",
            "A cost per correct answer is told from 5 correct answers (k) in the arm; below that it is `-`.",
            "## Appendix: codex, one row per question",
            "| qid | level | type | files | mm d1-3B | mm d3-edge |",
            "| image-photo-sign-1 | part | image | pass 40 s v5 | pass 30 s a1 | pass 38 s a1 |",
            "| image-photo-sign-3 | week | image | FAIL timeout 900 s v14 | pass 25 s a1 | - |",
            "| text-note-code-1 | part | text | pass 10 s v0 | pass 8 s a0 | - |",
            "| video-porch-parcel-1 | week | video | FAIL 120 s v12 | FAIL 60 s a1 (est. past timeout) | - |",
            "- An mm exec whose ask may need more GPU seconds than 75% of the exec's time (675 s of 900:",
            "- The screen changes like a real one while the owner is active",
            "- `look` costs about 25 times more per frame than an ffmpeg montage"):
        assert line in text, line
    assert text.index("## Read with these") < text.index("## Appendix: codex")


def test_the_decider_tier_reports_each_templates_needle_p_and_verdict(tmp_path):
    figures = report.numbers([stub_run(tmp_path)])
    tier = figures["retrieval"]
    d13, edge = tier["models"]["mm/d1-3B"], tier["models"]["mm/d3-edge"]
    # the record from before the levels and p got both from the plan: its level, and the p of the candidate at the truth
    assert d13["types"]["image"] == {"asked": 3, "needles": 3, "found": 1, "p50": 2, "top5": 2, "false": 1, "s_per_window": 0.06,
                                     "absent": 0, "absent_clean": 0}
    assert d13["types"]["video"]["absent"] == 1 and d13["types"]["video"]["absent_clean"] == 1 and d13["family"] == "d1"
    assert d13["media"] == {"asked": 6, "needles": 5, "found": 2, "p50": 3, "top5": 3, "false": 1, "s_per_window": 0.06}
    assert (edge["media"]["needles"], edge["media"]["found"], edge["media"]["p50"], edge["media"]["top5"]) == (4, 2, 2, 2)
    assert tier["missed"] == ["porch-parcel"] and tier["asks"] == 10
    sign = tier["templates"]["photo-sign"]
    assert sign["questions"] == 3
    assert sign["models"]["mm/d1-3B"] == {"asked": 3, "found": 1, "near": 1, "stopped": 0, "cut": 0, "missed": 1, "p": pytest.approx(0.82)}
    assert sign["models"]["mm/d3-edge"] == {"asked": 3, "found": 2, "near": 0, "stopped": 0, "cut": 0, "missed": 1, "p": pytest.approx(0.92)}
    assert sign["verdicts"] == {"below its model's hit (calibration)": {"models": ["mm/d1-3B"], "levels": ["day"]},
                                "haystack suspect": {"models": ["mm/d1-3B", "mm/d3-edge"], "levels": ["week"]}}
    assert tier["templates"]["screen-error"]["verdicts"] == {"this model's miss": {"models": ["mm/d3-edge"], "levels": ["part"]}}
    assert tier["templates"]["porch-parcel"]["verdicts"] == {retrieval.ONE_MODEL: {"models": ["mm/d1-3B"], "levels": ["week"]}}

    text = render.markdown(figures)
    for line in (
            "| model | type | recall (at its hit) | recall at p >= 0.5 | top 5 | absent with no false sighting | false sightings per ask |",
            "| d1-3B | image | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) | - |",
            "Templates no model found: porch-parcel. The verdicts below say whether that is the haystack or the models.",
            "| template | needle questions | found by | nearly missed by (top 5 or p >= 0.5) | missed by | "
            "scan ended before the needle by (false sighting, max_s) | needle p (median) | verdict |",
            "| photo-sign | 3 | d1-3B 1/3, d3-edge 2/3 | d1-3B 1 | d1-3B 1, d3-edge 1 | - | d1-3B 0.82, d3-edge 0.92 | "
            "below its model's hit (calibration): d1-3B (day); haystack suspect: d1-3B, d3-edge (week) |",
            "| screen-error | 1 | d1-3B 1/1 | - | d3-edge 1 | - | d1-3B 0.97, d3-edge 0.30 | this model's miss: d3-edge (part) |",
            "| porch-parcel | 1 | - | - | d1-3B 1 | - | - | one model ran: the agent arm's look, or a second family, tells a miss from a "
            "haystack fault: d1-3B (week) |",
            "## Summary per model (the media questions)",
            "| model | family | recall (at its hit) | recall at p >= 0.5 | top 5 | false sightings per ask | median s per window | "
            "codex: mm accuracy | codex: s / k tokens / GPU s per correct |",
            "| d1-3B | d1 | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | 0.17 | 0.060 | 5/6 (83%) | 44.0 / 34.2 / 13.2 (k=5) |",
            "| d3-edge | d3 | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) | 0.00 | 0.060 | 3/4 (75%) | - / - / - (k=3) |"):
        assert line in text, line


def test_the_report_says_when_a_runs_probe_failed_or_was_skipped_or_never_ran(tiny, tmp_path):
    # changed: only the report's half of the old test (the lines of report.md); what `agents` and `decider` warn before they spend
    # anything is the commands' (test_cli.py)
    outcomes = {"failed": {"status": "failed", "failures": ["a: x", "b: y"]}, "skipped": {"status": "skipped", "failures": []},
                "none": None, "passed": {"status": "passed", "failures": []}}
    notes = {}
    for label, outcome in outcomes.items():
        run = run_of(tmp_path / label, tiny[:2], probe=outcome)
        lines = render.markdown(report.numbers([run])).splitlines()
        notes[label] = [line for line in lines if line.startswith("- Shortcut probe:")]
        if notes[label]:
            assert lines.index(notes[label][0]) < next(i for i, line in enumerate(lines) if line.startswith("- Runs pooled"))
    assert notes["passed"] == []
    assert "failed the shortcut probe (2 failures, in its shortcuts.md)" in notes["failed"][0]
    assert "was built with --no-probe" in notes["skipped"][0]
    assert "no shortcut probe record (it was built before the probe)" in notes["none"][0]
    # pooled runs: one line for each run that did not pass
    a = run_of(tmp_path / "a", tiny[:2], seed=8, probe=outcomes["failed"])
    b = run_of(tmp_path / "b", tiny[:2], seed=9, probe=outcomes["skipped"])
    lines = render.markdown(report.numbers([run_of(tmp_path / "c", tiny[:2], seed=7, probe=outcomes["passed"]), a, b])).splitlines()
    assert [line.split(" ")[3] for line in lines if line.startswith("- Shortcut probe:")] == [a.name, b.name]


def test_the_needles_p_takes_the_tolerance_it_is_given():
    truth = T0 + 3600
    moments = [[truth, truth]]
    sightings = [Moment(truth + 20, truth + 22, 0.8, "phone", {})]
    candidates = [Moment(truth + 25, truth + 25, 0.9, "phone", {})]
    # changed: there is no default tolerance (the old default was V7's, 10 s; a question's retrieval spec has its own)
    assert retrieval.needle_p(moments, sightings, candidates, 10.0) is None, "V7's tolerance"
    assert retrieval.needle_p(moments, sightings, candidates, tol=21) == 0.8
    assert retrieval.needle_p(moments, sightings, candidates, tol=30.0) == 0.9
    assert retrieval.needle_p(moments, [], [], tol=30.0) is None


def test_a_sighting_or_candidate_has_one_time_or_a_span_and_the_oracle_reads_both():
    # changed: reading a system's own answer (`t`, or `t0` and `t1`, as ISO text) into a Moment is the system's, tested in tests/mm;
    # here a Moment has two times, equal for one instant
    at = T0 + 3600
    assert retrieval.span_of(Moment(at, at, 0.7, "phone", {})) == (at, at)
    assert retrieval.span_of(Moment(at, at + 5, 0.7, "phone", {})) == (at, at + 5)
    moments = [[at, at]]
    assert retrieval.needle_p(moments, [Moment(at + 3, at + 3, 0.7, "phone", {})], [], 10.0) == 0.7, "a sighting of one instant"
    assert retrieval.needle_p(moments, [Moment(at + 3, at + 6, 0.6, "phone", {})], [], 10.0) == 0.6
    assert retrieval.needle_p(moments, [], [Moment(at - 4, at - 1, 0.5, "phone", {})], 10.0) == 0.5, "a candidate that spans"
    assert retrieval.needle_p(moments, [Moment(at + 50, at + 50, 0.9, "phone", {})], [Moment(at + 80, at + 80, 0.9, "phone", {})], 10.0) is None


def test_the_crossover_looks_at_the_levels_it_is_given_in_their_order():
    level = level_of_pairs
    ages = ("0-2h", "2-6h", "6-24h", "24h+")
    levels = {"24h+": level(2, 12, 20), "2-6h": level(10, 15, 20), "0-2h": level(10, 10, 20)}
    got = report.crossover(levels, order=ages)
    assert (got["level"], got["by"], got["lead"]) == ("2-6h", ["accuracy"], pytest.approx(0.25))
    assert report.crossover(levels) is None, "by default it reads part, day and week"
    thin = {"0-2h": level(0, 5, 9), "6-24h": level(0, 5, 20)}
    assert report.thin_levels(thin, order=ages) == {"0-2h": 9} and report.thin_levels(thin) == {}


# -- new: the tests of what the move changed --------------------------------------------------------------------------------------
# Expected values are hand-counted from the plan's 19 questions of a tiny week, or from the definitions in the docstrings.

def test_a_models_family_comes_from_its_name():
    """The family assertion of test_a_models_family_its_jobs_and_the_pilot_come_from_names_abouts_and_question_fields; the rest of that
    test (jobs, the questions a model is asked, the pilot) is the runner's and mm's. changed: a name is a configuration's."""
    names = ("d1-3B", "d1-2B", "d1-Omni-2B", "d3-edge", "pplx-decider-flash-v1.2", "h2o-lightning-4b", "fake")
    assert [retrieval.family(f"mm/{n}") for n in names] == ["d1", "d1", "d1", "d3", "pplx", "h2o", "fake"]
    assert [retrieval.variant_of(c) for c in ("mm/d1-3B", "mm/d1-3B+files", "listing")] == ["d1-3B", "d1-3B+files", "listing"]
    assert retrieval.family("mm/d1-3B+files") == "d1" and retrieval.family("listing") == "listing"


def test_runs_are_pooled_by_seed_once_and_with_one_time_limit(tiny, tmp_path):
    """The pooling half of test_a_run_whose_plan_changed_is_refused_and_a_seed_is_pooled_once (opening a run whose plan changed is
    test_runfolder's). The time limit is new: the ask-estimate sentence in the Markdown quotes one."""
    one = run_of(tmp_path / "one", tiny[:3])
    with pytest.raises(ValueError, match=r"same seed \(7\) and scale \(tiny\)"):
        report.numbers([one, run_of(tmp_path / "twin", tiny[:3])])
    assert [r["seed"] for r in report.numbers([one, run_of(tmp_path / "eight", tiny[:3], seed=8)])["runs"]] == [7, 8]
    with pytest.raises(ValueError, match="other time limits"):
        report.numbers([one, run_of(tmp_path / "slow", tiny[:3], seed=9, timeout_s=1800.0)])


def test_a_run_needs_the_control_and_says_which_questions_each_tested_configuration_was_asked(tiny, tmp_path):
    with pytest.raises(ValueError, match="needs the one configuration 'files'"):
        report.numbers([run_of(tmp_path / "no-control", tiny[:3], configs=(FAKE,))])
    run = run_of(tmp_path / "no-systems", tiny[:3])
    fs.write_json(run.folder / "systems.json", {})
    with pytest.raises(ValueError, match="does not say which questions mm/fake was asked"):
        report.numbers([run])


def test_blind_is_not_compared_and_the_frame_has_the_words_for_the_sides_from_the_configurations(tiny, tmp_path):
    blind, listing = parse_config("blind", DEFAULTS), parse_config("listing", {**DEFAULTS, "listing": None})
    run = run_of(tmp_path / "run", tiny[:3], configs=(blind, FILES, D13, EDGE), asked={D13.name: ["text", "image"], EDGE.name: ["image"]})
    for q in tiny[:3]:
        for config in (blind, FILES, D13):
            fs.write_json(run.result_path("claude", config, q.qid),
                          record(q, "claude", config, 7, f"execs/claude/{config.dir}/{q.qid}", "2026-11-02T12:00:00", stub_out(config, q)))
    figures = report.numbers([run])
    assert figures["frame"] == {"control": "files", "tested": "mm", "timeout_s": 900.0, "text_asked_of": ["mm/d1-3B"]}
    assert list(figures["agents"]["claude"]) == ["mm/d1-3B"], "blind has no system to compare, and mm/d3-edge has no result"
    assert figures["totals"]["results"] == 9, "the blind results are told in the totals"
    mixed = run_of(tmp_path / "mixed", tiny[:3], seed=8, configs=(FILES, FAKE, listing))
    assert report.numbers([mixed])["frame"]["tested"] == "tested", "two systems are no one system's word"


def test_the_results_of_a_run_are_those_of_its_agents_and_configurations_and_not_the_retrieval_tiers_or_the_lists_a_session_left(tiny, tmp_path):
    run = run_of(tmp_path / "run", tiny[:3])
    a = record(tiny[0], "claude", FAKE, 7, "execs/claude/mm-fake/x", "2026-11-02T12:00:00", stub_out(FAKE, tiny[0]))
    fs.write_json(run.result_path("claude", FAKE, tiny[0].qid), a)
    fs.write_json(run.folder / "results" / "claude" / FAKE.dir / "ambient-labels.json", [{"id": "id:bench-0"}])
    fs.write_json(run.folder / "results" / "claude" / FAKE.dir / "warm.json", {"seconds": 1.0})
    fs.write_json(run.folder / "results" / retrieval.FOLDER / FAKE.dir / "text-note-code-1.json", {"qid": "text-note-code-1"})
    assert report.results_of(run) == {"claude": {"mm-fake": {tiny[0].qid: a}}}
    # a list some other system's session closes with is no result, whatever it is called (G1's F10): only a file named after a question is
    fs.write_json(run.folder / "results" / "claude" / FAKE.dir / "leftovers.json", [{"x": 1}])
    fs.write_json(run.folder / "results" / "claude" / FILES.dir / "stray.json", {"x": 1})
    assert report.results_of(run) == {"claude": {"mm-fake": {tiny[0].qid: a}, "files": {}}}


def test_a_record_holds_the_fields_of_both_scenarios_old_records(tiny):
    """V7's record had `since`, `until`, `in_range`, `false` and `rank`; V8's, asked over several ranges, had `ranges` and `away`
    (sightings farther from every moment than the tolerance, in a range or not). One record has them all."""
    ranges, moments = ((100.0, 200.0), (1000.0, 1100.0)), ((150.0, 150.0), (1050.0, 1050.0))
    spec = RetrievalSpec("Is a shelf being assembled?", ("glasses",), ranges, 20, False, 30.0, moments, ())
    q = replace(sq("video-screen-error-1"), retrieval=spec)
    got = scored(q, asked([(140.0, 145.0, 0.9), (600.0, 610.0, 0.8), (1090.0, 1095.0, 0.7)]))
    assert (got["source"], got["since"], got["until"], got["ranges"], got["limit"]) == (
        "glasses", 100.0, 1100.0, [[100.0, 200.0], [1000.0, 1100.0]], 20)
    assert (got["found"], got["moments_found"], got["p"], got["rank"]) == (True, [0], 0.9, 2), "newest by end: 1095 (far) first, 145 second"
    assert (got["in_range"], got["false"], got["away"]) == (2, 1, 2), "600 to 610 is out of both ranges and far; 1090 to 1095 is in one, 40 s off"
    assert (got["config"], got["system"], got["variant"]) == ("mm/fake", "mm", "fake") and "model" not in got
    assert got["sightings"] == [{"p": 0.9}, {"p": 0.8}, {"p": 0.7}], "what the system said, as it said it"
    json.dumps(got)


class Oracle:
    """A session whose oracle finds each needle where the question's truth says it is, with p 0.9, and sees nothing for an absent
    question: the offline stand-in for a system. It keeps the queries it was asked."""

    def __init__(self, questions):
        self.by_qid, self.queries = {q.qid: q for q in questions}, []

    def oracle(self, query: RetrievalQuery) -> RetrievalResult:
        self.queries.append(query)
        return asked([(a, b, 0.9) for a, b in self.by_qid[query.qid].retrieval.moments], windows=10, seconds=1.5)


def test_a_retrieval_run_asks_each_question_with_a_spec_once_and_the_report_tells_what_it_wrote(tiny, tmp_path, capsys):
    run = run_of(tmp_path / "run", tiny)
    session = Oracle(tiny)
    with_spec = [q for q in tiny if q.retrieval]
    assert retrieval.run_retrieval(run, FAKE, session, tiny, warm_s=12.5) == len(with_spec) == 18
    assert [x.qid for x in session.queries] == [q.qid for q in with_spec], "in the questions' order, the config version (no spec) left out"
    folder = run.folder / "results" / retrieval.FOLDER / "mm-fake"
    assert sorted(p.stem for p in folder.glob("*.json")) == sorted(["warm", *(q.qid for q in with_spec)])
    assert fs.read_json(folder / "warm.json") == {"config": "mm/fake", "seconds": 12.5}
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("mm/fake ")]
    assert len(lines) == 18 and lines[0] == "mm/fake text-note-code-1: found=True false=0 10 windows in 1.5 s"
    # it goes on where it stopped: nothing is asked again, and the first warm-up seconds stay
    assert retrieval.run_retrieval(run, FAKE, session, tiny, warm_s=99.0) == 0 and len(session.queries) == 18
    assert fs.read_json(folder / "warm.json")["seconds"] == 12.5
    # a question whose record is gone is asked again, alone
    (folder / "video-screen-error-1.json").unlink()
    assert retrieval.run_retrieval(run, FAKE, session, tiny) == 1 and session.queries[-1].qid == "video-screen-error-1"

    # the report: the oracle finds every needle (13 of them with a spec) and sees nothing for the 5 absent questions
    figures = report.numbers([run])
    tier = figures["retrieval"]
    assert tier["asks"] == 18 and tier["missed"] == [] and tier["models"]["mm/fake"]["first_s"] == 12.5
    needles = {"text": 1, "records": 2, "image": 3, "video": 4, "audio": 3}  # (the plan's 19 questions, less the config version)
    absent = {"text": 1, "records": 1, "image": 1, "video": 1, "audio": 1}
    assert {t: (v["needles"], v["found"], v["absent"], v["absent_clean"]) for t, v in tier["models"]["mm/fake"]["types"].items()} == {
        t: (needles[t], needles[t], absent[t], absent[t]) for t in needles}
    text = render.markdown(figures)
    assert text.count("No agent results yet.") == 1 and "Every template was found by some model." in text


def test_five_correct_answers_in_each_side_are_enough_for_a_cost_per_correct_answer_and_four_are_not():
    """The boundary of MIN_CORRECT, which the ported crossover test crosses far from (4 of 10, and 10 of 20)."""
    five, four = level_of_pairs(5, 5, 10, 100, 40), level_of_pairs(4, 4, 10, 100, 40)
    assert report.cost_ratios(five) == pytest.approx((0.4, 1.0)) and report.cost_ratios(four) == (None, None)
    assert report.crossover({"day": five})["by"] == ["cost"] and report.crossover({"day": four}) is None
