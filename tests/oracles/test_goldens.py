"""The recorded goldens (tests/oracles/recorded), checked as a set before anything is held to them: they are what the capture scripts
wrote and nobody edited, they are as many as the recorded runs have, and the two sources of them agree where they overlap (the old
code's output for an exec and the exec folders the old code wrote in the pilot). Expected values are the plan's own figures
(sections 1 and 7) and the recorded files; nothing here needs D:\\mm-bench."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

RECORDED = Path(__file__).with_name("recorded")
PILOT, EXEC = RECORDED / "pilot-s1", RECORDED / "exec"
CELLS = {("claude", "files"): 31, ("claude", "mm-d1-3B"): 24, ("codex", "files"): 31, ("codex", "mm-d1-3B"): 24}
# plan 7.3, difference 3: five lines of mm's skill changed in MultimediaSearch since the pilot ("this machine" became "the developer's machine";
# one of the four lines wraps after "this")
SKILL_DRIFT = {"nervelet simulator": "robot simulator", "on this": "on the developer's"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def execs() -> list[dict]:
    return load(PILOT / "execs.json")["execs"]


def test_every_captured_file_is_as_it_was_written():
    for index in (RECORDED / "recorded.json", EXEC / "capture.json"):
        for rel, sha in load(index)["files"].items():
            assert digest(index.parent / rel) == sha, f"{index.parent / rel} was edited after it was captured"


def test_the_exec_goldens_hold_nothing_but_what_the_capture_wrote():
    on_disk = {p.relative_to(EXEC).as_posix() for p in EXEC.rglob("*") if p.is_file()}
    assert on_disk == set(load(EXEC / "capture.json")["files"]) | {"capture.json"}


def test_the_exec_goldens_come_from_the_old_code_and_the_current_skill():
    meta = load(EXEC / "capture.json")
    assert meta["repo_commit"] == "bb498a6fd0dfc5d5d839357a8cb57c0358f1aea6"
    assert meta["multimem_commit"].startswith("fe0ece5")
    assert sorted(p.name for p in EXEC.iterdir() if p.is_dir()) == [f"{a}-{m}" for a in ("claude", "codex") for m in ("blind", "files", "mm")]


def test_there_is_one_entry_for_each_of_the_110_pilot_execs(execs):
    assert Counter((e["agent"], e["dir"]) for e in execs) == Counter(CELLS)
    assert len(execs) == 110 == len({(e["agent"], e["dir"], e["qid"]) for e in execs})
    assert [e["qid"] for e in execs if e["excluded"]] == ["text-note-code-1", "text-note-code-1"]


def test_every_file_of_the_data_folder_is_hashed_once():
    hashes, manifest = load(PILOT / "sha256.json"), load(PILOT / "data-manifest.json")
    assert len(hashes) == 1575 and set(hashes) == set(manifest)  # 1,572 files and 3 logs
    assert sum(size for size, _ in manifest.values()) == 465_254_840  # plan section 1
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes.values())


def test_run_json_holds_the_hashes_of_the_recorded_plan():
    meta = load(PILOT / "run.json")
    assert meta["plan"].startswith("cfdb321d22e4") and meta["files"].startswith("b1406b488821")  # plan 7.1
    assert (meta["generator"], meta["seed"], meta["scale"], meta["days"]) == (6, 1, "pilot", ["2026-11-01", "2026-11-02", "2026-11-03"])
    assert meta["probe"] == {"status": "passed", "failures": []} and load(PILOT / "shortcuts.json")["failures"] == []


def test_the_old_code_wrote_the_settings_the_pilot_recorded(execs):
    seen = {e["settings_sha256"] for e in execs if e["agent"] == "claude" and not e["excluded"]}
    assert seen == {digest(EXEC / f"claude-{arm}" / "settings.json") for arm in ("files", "mm")} and len(seen) == 1


def test_the_old_code_wrote_the_instructions_the_pilot_recorded_apart_from_the_skills_five_lines(execs):
    texts = {}
    for e in execs:
        if e["instructions_sha256"]:
            text = (PILOT / "instructions" / f"{e['instructions_sha256'][:12]}.md").read_text(encoding="utf-8")
            texts.setdefault((e["agent"], e["dir"][:2]), set()).add(text)
    assert {k: len(v) for k, v in texts.items()} == {("claude", "fi"): 1, ("claude", "mm"): 1, ("codex", "fi"): 1, ("codex", "mm"): 1}
    for agent in ("claude", "codex"):
        (files,), (mm,) = texts[agent, "fi"], texts[agent, "mm"]
        assert files == (EXEC / f"{agent}-files" / "instructions.md").read_text(encoding="utf-8")
        captured = (EXEC / f"{agent}-mm" / "instructions.md").read_text(encoding="utf-8")
        assert len(mm.splitlines()) == len(captured.splitlines())
        changed = [(a, b) for a, b in zip(mm.splitlines(), captured.splitlines()) if a != b]
        assert len(changed) == 5
        for old, new in changed:
            for was, now in SKILL_DRIFT.items():
                old = old.replace(was, now)
            assert old == new
        assert len(captured) - len(mm) == 41  # plan section 1: 25,821 recorded, 25,862 now


def test_the_old_code_wrote_the_mcp_config_the_pilot_recorded(execs):
    recorded = {json.dumps(e["mcp"], sort_keys=True) for e in execs if e["mcp"]}
    assert recorded == {json.dumps(load(EXEC / "claude-mm" / "mcp.json"), sort_keys=True)}


def test_the_old_code_asked_claude_for_the_tools_the_pilot_listed(execs):
    """argv holds the tools the exec is allowed; the init event lists the tools the session has. They must be the same set."""
    argv = {arm: load(EXEC / f"claude-{arm}" / "argv.json") for arm in ("blind", "files", "mm")}
    files = set(argv["files"][argv["files"].index("--tools") + 1].split(","))
    mm = set(argv["mm"][argv["mm"].index("--allowedTools") + 1:])
    assert argv["blind"][argv["blind"].index("--tools") + 1] == ""
    got = {(e["dir"][:2], tuple(e["init"]["tools"])) for e in execs if e["agent"] == "claude"}
    assert {(arm, frozenset(tools)) for arm, tools in got} == {("fi", frozenset(files)), ("mm", frozenset(mm))}


def test_the_slot_of_an_exec_is_left_out_of_its_deny_list():
    """The exec in slot 0 may name its own working folder (and for V8 its own folder of recordings), and no other slot's."""
    for name, own, others in (("v7", ("-w0",), ("-w1", "-w3")), ("v8", ("-w0", "-d0"), ("-w1", "-d1", "-w3", "-d3"))):
        words = load(EXEC / f"deny-{name}.json")[name]
        assert not [w for w in words["slot0"] if any(slot in w for slot in own)]
        assert all(any(slot in w for w in words["slot0"]) for slot in others)
        assert set(words["slot0"]) < set(words["none"]) and any(own[0] in w for w in words["none"])


def test_the_replay_execs_are_two_for_each_cell_and_ended_without_a_problem():
    found = sorted(p.parent.relative_to(PILOT / "replay").parts[:2] for p in (PILOT / "replay").rglob("result.json"))
    assert Counter(map(tuple, found)) == Counter({cell: 2 for cell in CELLS})
    for result in (PILOT / "replay").rglob("result.json"):
        record = load(result)
        assert record["problem"] is None and record["passed"] is True
        events = (result.with_name("events.jsonl")).read_text(encoding="utf-8").splitlines()
        assert events and all(isinstance(json.loads(line), dict) for line in events)
