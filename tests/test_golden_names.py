"""New tests for tests/golden_names.py, the two lists of how the search tool differs from the goldens: the names it changed (RENAMES) and the
lines its skill gained in facd716 (SKILL_CHANGES). Every entry is used, nothing the renames leave is an old name, the lists touch only the
recordings that name or hold the tool's text (the rest are read as they are). That the text the tool serves now equals the golden with both
applied is tests/mm/test_exec_mm.py's (it imports `mms`, which a test outside tests/mm may not: tests/test_neutral.py), so the lists are all
the difference."""

from __future__ import annotations

from pathlib import Path

from golden_names import RENAMES, SKILL_CHANGES, SKILL_CHANGES_WHY, golden_bytes, golden_json, golden_text, renamed

RECORDED = Path(__file__).with_name("oracles") / "recorded"
# the goldens the tests read through the lists: the exec goldens (but not capture.json, the capture's own record), the pilot's
# instruction texts and its execs.json
READ = [*(p for p in sorted((RECORDED / "exec").rglob("*")) if p.is_file() and p.name != "capture.json"),
        *sorted((RECORDED / "pilot-s1" / "instructions").glob("*.md")), RECORDED / "pilot-s1" / "execs.json"]
OLD_NAMES = ("multimem", "MULTIMEM")
# the four texts that hold the skill: what the agents are told, as the exec goldens and the pilot recorded it (the pilot's other texts are
# the files cells', which hold none of it)
INSTRUCTIONS = [RECORDED / "exec" / "claude-mm" / "instructions.md", RECORDED / "exec" / "codex-mm" / "instructions.md",
                *(RECORDED / "pilot-s1" / "instructions" / f"{sha}.md" for sha in ("88d94d46706f", "0eaf4d4f20f7"))]


def test_every_rename_has_a_reason_and_is_used_by_a_golden():
    raw = {p: p.read_bytes().decode("utf-8") for p in READ}
    for old, new, why in RENAMES:
        assert why.strip() and old != new
        assert any(old in text for text in raw.values()), f"{old!r} is in no golden: take it off the list"


def test_no_old_name_is_left_in_a_golden_read_through_the_lists():
    left = {p.relative_to(RECORDED).as_posix(): word for p in READ for word in OLD_NAMES if word.lower() in golden_text(p).lower()}
    assert left == {}, "a name the tool changed that RENAMES does not map"


def test_every_skill_change_is_in_each_of_the_four_instruction_texts_once_and_in_no_other_golden():
    assert len(INSTRUCTIONS) == 4 and "facd716" in SKILL_CHANGES_WHY
    for old, new in SKILL_CHANGES:
        assert old != new
        for path in INSTRUCTIONS:
            assert renamed(path.read_bytes().decode("utf-8")).count(old) == 1, f"{old[:60]!r} is not once in {path.name}"
        for path in set(READ) - set(INSTRUCTIONS):
            assert old not in renamed(path.read_bytes().decode("utf-8")), f"{old[:60]!r} is in {path.name}"


def test_the_lists_change_only_the_goldens_that_name_the_tool_and_the_rest_are_read_as_they_are():
    changed = sorted(p.relative_to(RECORDED).as_posix() for p in READ if golden_bytes(p) != p.read_bytes())
    assert changed == sorted([
        "exec/claude-blind/env.json", "exec/claude-files/env.json", "exec/claude-mm/env.json", "exec/claude-mm/instructions.md",
        "exec/claude-mm/mcp.json", "exec/codex-blind/env.json", "exec/codex-files/argv.json", "exec/codex-files/env.json",
        "exec/codex-mm/argv.json", "exec/codex-mm/env.json", "exec/codex-mm/instructions.md", "exec/deny-v7.json", "exec/deny-v8.json",
        "pilot-s1/execs.json", "pilot-s1/instructions/0eaf4d4f20f7.md", "pilot-s1/instructions/88d94d46706f.md"])
    for path in INSTRUCTIONS:
        assert len(golden_text(path)) - len(renamed(path.read_bytes().decode("utf-8"))) == 1243, "the skill's changes are 1,243 characters"


def test_the_lists_are_applied_to_text_and_through_json_to_keys_and_values():
    assert renamed("MULTIMEM_HOME and multimem.cli, `mm approve <name>`") == "MMS_HOME and mms.cli, `mms approve <name>`"
    assert renamed("the mm tools, mcp__mm__ask, no-mm-home") == "the mm tools, mcp__mm__ask, no-mm-home", "the benchmark's own `mm` stays"
    assert golden_json(RECORDED / "exec" / "claude-files" / "env.json")["MMS_HOME"].endswith("no-mm-home")
    assert golden_json(RECORDED / "exec" / "claude-mm" / "mcp.json")["mcpServers"]["mm"]["args"][:2] == ["-m", "mms.cli"]
