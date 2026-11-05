"""Ported from tests/test_bench.py of Multimem at fe0ece5 (plan 7.4), `test_a_models_family_its_jobs_and_the_pilot_come_from_names_abouts_and_question_fields`
(test_bench.py:933-971): which questions a configuration is asked, the pilot and the controls. `family` and `arm_jobs` are the mm
system's and are tests/mm's (U7). `model` is the configuration and `About.reads` is `Description.reads`; the expected sets are written
from the questions' own modalities, and the controls' rule is the source's: a needle of text, one of records and an absent one, each with
a retrieval run (the old oracle) and the least vague hint, the qid breaking ties. The tests from "choosing cases" down are new (plan 6.1):
the expected values are worked by hand, from the strata and the results the test writes, not from the code under test."""

from __future__ import annotations

import dataclasses
import math

import pytest
from stub_system import FILES, STUB, StubSystem, run_of, tiny_questions

from mmmb.core import fs
from mmmb.core.config import Config
from mmmb.core.select import (CONTROL_TYPES, MEDIA, MODALITIES, SUBSETS, Selection, Separation, asked_modalities, canaries_of, config_questions,
                              discriminating, pick_controls, pilot_questions, select, selected, separate, stratified)

SYSTEM = StubSystem()
ALL, NARROW, MEDIA_ONLY = (SYSTEM.describe(v) for v in ("all", "narrow", "media"))
BLIND = Config("blind", None, None, False)
NARROW_CONFIG, ALL_CONFIG = Config("stub/narrow", "stub", "narrow", False), STUB
QUESTIONS = list(tiny_questions())


def modalities(questions) -> set[str]:
    return {q.tags["modality"] for q in questions}


def test_the_modalities_are_the_five_the_contract_names_and_the_subsets_are_all_and_pilot():
    assert set(MODALITIES) == {"image", "video", "audio", "text", "records"} and set(MEDIA) | set(CONTROL_TYPES) == set(MODALITIES)
    assert SUBSETS == ("all", "pilot")


def test_a_configuration_is_asked_what_its_system_reads_and_one_with_tools_of_its_own_or_no_system_is_asked_everything():
    # the text and records are asked of one configuration's system (or of every one's, with no `controls`)
    assert config_questions(QUESTIONS, ALL_CONFIG, ALL, "stub/all") == QUESTIONS
    assert config_questions(QUESTIONS, NARROW_CONFIG, NARROW) == [q for q in QUESTIONS if q.tags["modality"] != "audio"]
    assert modalities(config_questions(QUESTIONS, NARROW_CONFIG, NARROW, "stub/all")) == {"image", "video"}
    assert modalities(config_questions(QUESTIONS, Config("stub/media", "stub", "media", False), MEDIA_ONLY)) == {"image", "video"}
    assert asked_modalities(NARROW_CONFIG, NARROW, "stub/all") == ["image", "video"]
    assert asked_modalities(NARROW_CONFIG, NARROW, "stub/narrow") == ["image", "records", "text", "video"]
    assert asked_modalities(NARROW_CONFIG, NARROW) == ["image", "records", "text", "video"]
    # tools of its own read every modality (ffmpeg hears), and a configuration with no system has none to read through
    both = Config("stub/all+files", "stub", "all", True)
    assert config_questions(QUESTIONS, both, NARROW, "stub/narrow") == QUESTIONS and asked_modalities(both, NARROW, "x") == sorted(MODALITIES)
    assert config_questions(QUESTIONS, FILES, None, "stub/narrow") == QUESTIONS
    assert config_questions(QUESTIONS, BLIND, None, "stub/narrow") == QUESTIONS and asked_modalities(BLIND, None) == sorted(MODALITIES)


def stub_question(qid: str, hint: str = "part", *, modality: str, absent: bool = False, retrieval=True):
    """A question with the fields the controls' rule reads, made from a real one."""
    base = next(q for q in QUESTIONS if q.tags["modality"] == modality)
    return dataclasses.replace(base, qid=qid, hint=hint, tags={**base.tags, "absent": absent},
                               retrieval=base.retrieval or next(q.retrieval for q in QUESTIONS if q.retrieval) if retrieval else None)


def test_the_controls_are_a_needle_of_text_one_of_records_and_an_absent_one_each_with_a_retrieval_run_and_the_least_vague_hint():
    controls = pick_controls(QUESTIONS)
    assert [(q.tags["modality"], q.tags["absent"]) for q in controls][:2] == [("text", False), ("records", False)]
    assert controls[2].tags["absent"] and controls[2].tags["modality"] in CONTROL_TYPES and all(q.retrieval for q in controls)
    # on stub questions, whatever the scale: a needle with a retrieval run and the least vague hint, the qid breaking ties
    stub = [stub_question("text-config-asof-1", "time", modality="text", retrieval=False), stub_question("text-note-code-1", "day", modality="text"),
            stub_question("text-note-code-2", "part", modality="text"), stub_question("records-log-count-1", "part", modality="records"),
            stub_question("records-absent-log-1", "day", modality="records", absent=True), stub_question("image-photo-sign-1", modality="image")]
    assert [q.qid for q in pick_controls(stub)] == ["text-note-code-2", "records-log-count-1", "records-absent-log-1"]


def test_the_pilot_is_the_media_questions_of_what_some_system_reads_and_three_controls_and_a_scenarios_own_marks_are_its_pilot():
    controls = {q.qid for q in pick_controls(QUESTIONS)}
    pilot = pilot_questions(QUESTIONS, [NARROW.reads, MEDIA_ONLY.reads])
    assert [q.qid for q in pilot] == [q.qid for q in QUESTIONS if q.tags["modality"] in ("image", "video") or q.qid in controls]
    assert "audio" in modalities(pilot_questions(QUESTIONS, [NARROW.reads, ALL.reads]))
    assert modalities(pilot_questions(QUESTIONS, [])) == {"image", "video", "text", "records"}
    # a scenario that marks its pilot (a `pilot` tag) has the marked ones, whatever they read
    marked = [dataclasses.replace(q, tags={**q.tags, "pilot": q.qid in ("image-photo-sign-1", "audio-mic-alarm-1")}) for q in QUESTIONS]
    assert [q.qid for q in pilot_questions(marked, [NARROW.reads])] == ["image-photo-sign-1", "audio-mic-alarm-1"]


def test_selected_gives_all_the_pilot_or_the_named_ones_of_them_and_refuses_what_is_not_in_the_subset():
    d = {ALL_CONFIG: ALL, FILES: None, NARROW_CONFIG: NARROW}
    pilot = pilot_questions(QUESTIONS, [NARROW.reads])
    assert selected(QUESTIONS, "all", d) == QUESTIONS and selected(QUESTIONS, "pilot", {NARROW_CONFIG: NARROW}) == pilot
    # the reads of a configuration with tools of its own do not widen the pilot's media
    own = {Config("stub/all+files", "stub", "all", True): ALL, NARROW_CONFIG: NARROW}
    assert selected(QUESTIONS, "pilot", own) == pilot
    with pytest.raises(ValueError, match="subset"):
        selected(QUESTIONS, "half", d)
    two = [pilot[0].qid, pilot[-1].qid]
    assert [q.qid for q in selected(QUESTIONS, "pilot", {NARROW_CONFIG: NARROW}, two)] == two
    with pytest.raises(ValueError, match="not in the pilot questions: audio-mic-alarm-1"):
        selected(QUESTIONS, "pilot", {NARROW_CONFIG: NARROW}, ["audio-mic-alarm-1"])
    assert [q.qid for q in selected(QUESTIONS, "all", d, ["audio-mic-alarm-1"])] == ["audio-mic-alarm-1"]


# -- choosing cases (DESIGN 8.2, plan 6.1) --------------------------------------------------------------------------------------------

def strata_of(scenario_sizes: dict[tuple[str, str], int]) -> list:
    """Questions made from a real one, `size` of each (scenario, kind) stratum, qids numbered within it."""
    base = QUESTIONS[0]
    return [dataclasses.replace(base, qid=f"{s}-{k}-{i}", scenario=s, kind=k)
            for (s, k), size in scenario_sizes.items() for i in range(size)]


def counts(questions) -> dict[tuple[str, str], int]:
    out: dict = {}
    for q in questions:
        out[q.scenario, q.kind] = out.get((q.scenario, q.kind), 0) + 1
    return out


def test_a_stratified_draw_of_6_from_20_and_10_is_4_and_2_and_the_remainders_go_to_the_largest_fractions_then_the_first_stratum():
    # worked by hand: the quotas of 6 are 6 * 20/30 = 4 and 6 * 10/30 = 2, whole numbers
    assert counts(stratified(strata_of({("V", "a"): 20, ("V", "b"): 10}), 6, 1)) == {("V", "a"): 4, ("V", "b"): 2}
    # 7, 5 and 3 of 15, n = 5: the quotas are 2.33, 1.67 and 1; the floors 2, 1, 1 leave one place, to the largest fraction (.67)
    assert counts(stratified(strata_of({("V", "a"): 7, ("V", "b"): 5, ("V", "c"): 3}), 5, 1)) == {("V", "a"): 2, ("V", "b"): 2, ("V", "c"): 1}
    # 2, 2 and 2, n = 4: the quotas are 1.33 each, one place is left, and the fractions tie: the stratum named first has it
    assert counts(stratified(strata_of({("V", "c"): 2, ("V", "b"): 2, ("V", "a"): 2}), 4, 1)) == {("V", "a"): 2, ("V", "b"): 1, ("V", "c"): 1}
    # a stratum is a scenario and a kind: the same kind of two scenarios are two strata
    assert counts(stratified(strata_of({("X", "a"): 4, ("Y", "a"): 2}), 3, 1)) == {("X", "a"): 2, ("Y", "a"): 1}


def test_a_stratified_draw_has_n_questions_each_stratum_within_one_of_its_share_and_is_the_same_set_for_a_seed():
    questions = strata_of({("V", "a"): 7, ("V", "b"): 5, ("V", "c"): 3, ("W", "a"): 4})
    total = len(questions)
    order = [q.qid for q in questions]
    for n in range(total + 3):
        drawn = stratified(questions, n, 3)
        assert len(drawn) == min(n, total)
        assert [q.qid for q in drawn] == [qid for qid in order if qid in {q.qid for q in drawn}], "in the questions' order"
        assert drawn == stratified(questions, n, 3), "the same seed, the same set"
        for stratum, size in counts(questions).items():
            share = n * size / total
            assert math.floor(share) <= counts(drawn).get(stratum, 0) <= min(math.ceil(share), size), (n, stratum)
    assert stratified(questions, total + 5, 1) == tuple(questions) and stratified(questions, total, 1) == tuple(questions), "n over the total gives all"
    assert stratified(questions, 0, 1) == ()
    with pytest.raises(ValueError, match="a sample of -1"):
        stratified(questions, -1, 1)


def test_a_stratified_draw_depends_on_its_seed_and_a_stratum_is_drawn_by_its_own_generator():
    questions = strata_of({("V", "a"): 12})
    drawn = {seed: [q.qid for q in stratified(questions, 4, seed)] for seed in range(1, 7)}
    assert len({tuple(qids) for qids in drawn.values()}) > 1, "another seed draws other questions"
    # questions in another stratum do not change which of this stratum's are drawn, if it keeps its number of places
    other = strata_of({("V", "a"): 12, ("V", "b"): 12})
    assert counts(stratified(other, 8, 4)) == {("V", "a"): 4, ("V", "b"): 4}
    assert [q.qid for q in stratified(other, 8, 4) if q.kind == "a"] == drawn[4]


def test_select_narrows_by_scenario_kind_modality_split_and_tag_and_refuses_a_value_no_question_has():
    absent = [q for q in QUESTIONS if q.kind == "absent"]
    assert select(QUESTIONS, Selection(kinds=("absent",))) == tuple(absent) and len(absent) == 5
    assert select(QUESTIONS, Selection(kinds=("ABSENT", "count"))) == tuple(q for q in QUESTIONS if q.kind in ("absent", "count")), "any case"
    assert select(QUESTIONS, Selection(modalities=("video", "image"), kinds=("when", "last"))) == tuple(
        q for q in QUESTIONS if q.kind in ("when", "last") and q.tags["modality"] in ("video", "image"))
    assert select(QUESTIONS, Selection(scenarios=("V7",), splits=("dev",))) == tuple(QUESTIONS)
    assert select(QUESTIONS, Selection(tags={"template": "photo-sign"})) == tuple(q for q in QUESTIONS if q.tags["template"] == "photo-sign")
    assert select(QUESTIONS, Selection(tags={"absent": "true"})) == tuple(absent), "a true or false tag is matched in any case"
    assert select(QUESTIONS, Selection(tags={"rung": "tiny", "last": "True"})) == tuple(q for q in QUESTIONS if q.tags["last"])
    assert select(QUESTIONS, Selection()) == tuple(QUESTIONS)
    for selection, message in ((Selection(kinds=("lastt",)), r"--kinds names lastt, which no question has; they have absent, count, find, last, when"),
                               (Selection(scenarios=("V1",)), "--scenarios names V1"), (Selection(splits=("test",)), "--splits names test"),
                               (Selection(modalities=("smell",)), "--modalities names smell"),
                               (Selection(tags={"colour": "red"}), "no question has the tag colour; they have "),
                               (Selection(tags={"rung": "1h"}), "--tag rung names 1h, which no question has; they have tiny"),
                               (Selection(kinds=("count",), modalities=("video",)), "--modalities names video, which no question")):
        with pytest.raises(ValueError, match=message):
            select(QUESTIONS, selection)


def test_limit_draws_a_stratified_sample_in_the_questions_order_and_the_trail_says_what_each_step_left():
    trail: list = []
    drawn = select(QUESTIONS, Selection(limit=6), trail)
    assert len(drawn) == 6 and list(drawn) == [q for q in QUESTIONS if q in drawn] and drawn == select(QUESTIONS, Selection(limit=6))
    for kind, size in counts(QUESTIONS).items():
        assert math.floor(6 * size / 19) <= counts(drawn).get(kind, 0) <= math.ceil(6 * size / 19), kind
    assert trail == [("the questions", 19), ("--limit 6", 6)]
    trail.clear()
    assert len(select(QUESTIONS, Selection(kinds=("absent", "when"), limit=4), trail)) == 4
    assert trail == [("the questions", 19), ("--kinds absent,when", 10), ("--limit 4", 4)]
    assert select(QUESTIONS, Selection(limit=100)) == tuple(QUESTIONS)
    for bad in (0, -2):
        with pytest.raises(ValueError, match=r"--limit is 1 or more"):
            select(QUESTIONS, Selection(limit=bad))


# discriminating: results written by hand, one run of two agents and two configurations

FILES_CONFIG, MM_CONFIG = Config("files", "files", None, True), Config("stub/all", "stub", "all", False)


def put(run, agent: str, config: Config, verdicts: dict[str, bool | str]) -> None:
    """A result for each qid: True or False for a pass or a fail, a word (`setup`) for a result that did not test the question."""
    for qid, verdict in verdicts.items():
        fs.write_json(run.result_path(agent, config, qid),
                      {"qid": qid, "passed": verdict is True, "problem": None if isinstance(verdict, bool) else f"{verdict}: x"})


def history(tmp_path, name: str = "run") -> tuple:
    """A run where, by the number of the question in `q`: 0 is right in every cell, 1 wrong in every cell, 2 right with `files` only (claude),
    3 right with `stub` only (claude), 4 right in claude's both and wrong in codex's `stub`, 5 right with `files` and a `setup` with `stub`, 6 asked
    of `files` only and wrong, 7 and 8 never asked."""
    run, _ = run_of(tmp_path / name)
    q = [x.qid for x in QUESTIONS[:9]]
    put(run, "claude", FILES_CONFIG, {q[0]: True, q[1]: False, q[2]: True, q[3]: False, q[4]: True, q[5]: True, q[6]: False})
    put(run, "claude", MM_CONFIG, {q[0]: True, q[1]: False, q[2]: False, q[3]: True, q[4]: True, q[5]: "setup"})
    put(run, "codex", FILES_CONFIG, {q[0]: True, q[1]: False, q[4]: True})
    put(run, "codex", MM_CONFIG, {q[0]: True, q[1]: False, q[4]: False})
    return run, q


def test_a_question_is_discriminating_when_an_agents_done_results_differ_between_configurations_of_one_run(tmp_path):
    run, q = history(tmp_path)
    sep = separate(QUESTIONS[:9], [run])
    assert sep == Separation(separating=(q[2], q[3], q[4]), right=(q[0], q[5]), wrong=(q[1], q[6]), mixed=(), unrun=(q[7], q[8]))
    assert separate(QUESTIONS[:9], [run], ("claude",)).separating == (q[2], q[3]), "q4 separates only for codex"
    assert separate(QUESTIONS[:9], [run], ("codex",)).separating == (q[4],)
    assert [x.qid for x in discriminating(QUESTIONS[:9], [run], canaries=0)] == [q[2], q[3], q[4]]
    # a canary of each kind in turn, the first of each in the questions' order: the first right one and the first wrong one
    assert [x.qid for x in discriminating(QUESTIONS[:9], [run])] == [q[0], q[1], q[2], q[3], q[4]]
    assert canaries_of(sep, 1) == (q[0],) and canaries_of(sep, 3) == (q[0], q[1], q[5]) and canaries_of(sep, 9) == (q[0], q[1], q[5], q[6])
    assert [x.qid for x in discriminating(QUESTIONS[:9], [run], 2, ("claude",))] == [q[0], q[1], q[2], q[3]]
    assert discriminating(QUESTIONS[:9], [], 2) == (), "no earlier run, nothing separated or answered"


def test_results_of_different_runs_that_disagree_are_mixed_and_one_cell_answered_in_two_runs_is_not_separation(tmp_path):
    one, q = history(tmp_path, "one")
    two, _ = run_of(tmp_path / "two")
    put(two, "claude", FILES_CONFIG, {q[0]: False, q[1]: False, q[2]: True})
    sep = separate(QUESTIONS[:9], [one, two])
    assert q[0] in sep.mixed and q[0] not in sep.separating, "right in every cell of one run and wrong in the other's"
    assert q[2] in sep.separating and q[1] in sep.wrong and q[0] not in canaries_of(sep)


def test_select_with_discriminating_keeps_those_that_separated_and_two_canaries_and_a_limit_counts_the_canaries(tmp_path):
    run, q = history(tmp_path)
    trail: list = []
    kept = select(QUESTIONS[:9], Selection(discriminating=(run,)), trail)
    assert [x.qid for x in kept] == [q[0], q[1], q[2], q[3], q[4]]
    assert trail[-1] == ("--discriminating run (3 that separated them, 2 canaries)", 5)
    assert [x.qid for x in select(QUESTIONS[:9], Selection(discriminating=(run,), agents=("claude",)))] == [q[0], q[1], q[2], q[3]]
    two = select(QUESTIONS[:9], Selection(discriminating=(run,), limit=4))
    assert len(two) == 4 and {q[0], q[1]} <= {x.qid for x in two}, "the limit is the whole selection, canaries among it, kept"
    assert [x.qid for x in select(QUESTIONS[:9], Selection(discriminating=(run,), limit=1))] == [q[0]]
    assert [x.qid for x in select(QUESTIONS[:9], Selection(discriminating=(run,), limit=2))] == [q[0], q[1]]
    with pytest.raises(ValueError, match=r"none of the 3 questions separated the configurations in run \(2 have no done result there\)"):
        select(QUESTIONS[6:9], Selection(discriminating=(run,)))
