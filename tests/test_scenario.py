"""New tests for mmmb/core/scenario.py (the plan and files hashes) and the session with nothing to run (mmmb/core/contract.py). The
expected hashes are those of bytes written out here by hand; the V7 plan's recorded hashes are checked in U2 (oracle 1a)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mmmb.core.contract import Case, PlainSession, RetrievalQuery
from mmmb.core.scenario import dump_key, files_hash, plan_hash


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_plan_hash_is_the_sha256_of_the_questions_as_json_with_sorted_keys():
    assert plan_hash([{"b": 1, "a": [2, None]}, {"q": "é"}]) == sha('[{"a": [2, null], "b": 1}, {"q": "\\u00e9"}]')
    assert plan_hash([{"a": 1, "b": 2}]) == plan_hash([{"b": 2, "a": 1}]), "the order of keys is no change"
    assert plan_hash([{"a": 1}, {"b": 2}]) != plan_hash([{"b": 2}, {"a": 1}]), "the order of questions is"
    assert plan_hash(({"a": 1},)) == plan_hash([{"a": 1}])


def test_the_files_hash_is_the_sha256_of_the_files_and_the_log_lines_as_json_with_sorted_keys():
    rows = [("porch/a.mkv", "porch", 1.5, None, "clip", {"z": 1, "a": [True]}), ["notes/n.md", "notes", 2.0, "n", "note", {}]]
    logs = {"logs": [{"ts": 3, "msg": "x"}]}
    expected = ('{"files": [["porch/a.mkv", "porch", 1.5, null, "clip", {"a": [true], "z": 1}], '
                '["notes/n.md", "notes", 2.0, "n", "note", {}]], "logs": {"logs": [{"msg": "x", "ts": 3}]}}')
    assert files_hash(rows, logs) == sha(expected)
    assert files_hash(rows, logs) != files_hash(rows[::-1], logs) and files_hash(rows, logs) != files_hash(rows, {})


def test_a_value_json_cannot_hold_is_hashed_as_its_text_in_a_files_hash():
    assert files_hash([("p", "s", 1.0, None, "k", {"where": Path("a")})], {}) == sha('{"files": [["p", "s", 1.0, null, "k", {"where": "a"}]], "logs": {}}')


def test_a_dump_is_named_by_the_sha256_of_its_scenario_recipe_and_generator_as_canonical_json():
    recipe = {"seed": 1, "scale": "tiny", "days": ["2026-11-01", "2026-11-02"]}
    expected = sha('{"generator":6,"recipe":{"days":["2026-11-01","2026-11-02"],"scale":"tiny","seed":1},"scenario":"v7_screen"}')
    assert dump_key("v7_screen", recipe, 6) == expected
    assert dump_key("v7_screen", dict(reversed(recipe.items())), 6) == expected, "the order of keys is no change"
    assert len({expected, dump_key("v7_screen", recipe, 7), dump_key("v8_egolife", recipe, 6), dump_key("v7_screen", {**recipe, "seed": 2}, 6)}) == 4


def test_a_plain_session_has_no_server_costs_nothing_waits_for_nothing_and_has_no_oracle(tmp_path):
    case = Case("q", 1.0, None, ("s",), 0.0, 1.0, "g", 0.0, 0, tmp_path / "work", tmp_path / "run")
    session = PlainSession()
    assert session.max_jobs([case], 3) == 3 and session.estimate(["s"], [(0.0, 1.0)]) == 0.0 and session.admits(case, [case]) is True
    served = session.serve(case)
    assert (served.server, served.base_line, served.parts, served.server_instructions, served.holder, served.clock_note, served.record) \
        == (None, None, (), None, None, None, {})
    assert session.audit(case, []) == {} and session.reset(case, [], 2) == {}
    usage = session.usage(case, [])
    assert (usage.gpu_s, usage.queue_s, usage.api_usd) == (0.0, 0.0, 0.0) and session.queue_s(case) == 0.0
    with pytest.raises(NotImplementedError, match="no oracle"):
        session.oracle(RetrievalQuery("q", "text", ("s",), ((0.0, 1.0),), 5, 5, False, 10.0))
    assert session.close() == {}
