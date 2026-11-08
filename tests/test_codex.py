"""Ported from tests/test_codex.py of MultimediaSearch at fe0ece5 (plan 7.4, oracle 4): the fast tests of the harness of the old Codex recall
run, which are the table of its graders and the reading of Codex's answers, events and plan use. Not ported:
`test_codex_recalls_the_day`, the paid run (19 questions to a real `codex exec` over a daemon of the old recall tier, which
spends the plan's quota); its fixtures went with it. The graders are the ones of `mmsb.core.grade`."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import NamedTuple

import pytest
from agent_spec import served_of, server_of, spec_of

from mmsb.core.agents import codex, common
from mmsb.core.answer import answer_of
from mmsb.core.grade import code, near, norm, numbers, unknown

STOP_AT = 60.0  # percent of the plan's use at which the remaining questions are skipped
QUOTA: dict = {"used": None, "ran": False}  # the last weekly plan use read, in percent; whether an exec has run


# -- graders ----------------------------------------------------------------------------------------------------------

def invoice_total(a: str) -> bool:
    """342.50 as the day's invoice says it, or as a reader writes it ("342.5", "342,50 EUR"), or "unknown": an agent that
    cannot read a PDF may say so, and must not invent a total."""
    return unknown(a) or re.search(r"\b342[.,]50?\b", norm(a)) is not None


def rule_3(a: str) -> bool:
    counts = re.sub(r"(?:err[- ]?)?\b(?:503|429)\b", " ", norm(a))
    return code(a, "503") and not code(a, "429") and "5" in numbers(counts)


class Question(NamedTuple):
    n: int
    tier: str  # C1 (find and look, no decider) or C2 (real d1)
    text: str
    truth: str
    grade: Callable[[str], bool]


QUESTIONS = [
    Question(1, "C1", "What's the gate code from the email this morning?", "4821", lambda a: "4821" in numbers(a)),
    Question(2, "C1", "How many days do we keep logs for now?", "30 (45 was the morning's, 45 is wrong)",
             lambda a: "30" in numbers(a) and "45" not in numbers(a)),
    Question(3, "C1", "Which error showed up most in the server log, and how many times?", "ERR-503, 5", rule_3),
    Question(4, "C1", "When did the deploy finish?", "12:07", lambda a: near(a, "12:07", 1)),
    Question(5, "C1", "What did I note about the dentist?", "moved to Thursday 9:30",
             lambda a: "thursday" in norm(a) and near(a, "9:30", 0.5)),
    Question(6, "C1", "What's the Wi-Fi password?", "unknown", unknown),
    Question(7, "C2", "When did the doorbell ring last?", "16:40", lambda a: near(a, "16:40", 2)),
    Question(8, "C2", "How many times did the doorbell ring?", "3", lambda a: "3" in numbers(a)),
    Question(9, "C2", "What error code was on my screen in the morning?", "E-4417",
             lambda a: re.search(r"\be-?4417\b", norm(a)) is not None),
    Question(10, "C2", "When did the disk-full error first appear?", "10:00", lambda a: near(a, "10:00", 1)),
    Question(11, "C2", "What did the whiteboard say?", "SHIP 2.1 FRI",
             lambda a: all(w in norm(a) for w in ("ship", "2.1", "fri"))),
    Question(12, "C2", "Did a phone ring during the 2 pm call, and when?", "yes, 14:03",
             lambda a: re.search(r"\byes\b", norm(a)) is not None and near(a, "14:03", 1)),
    Question(13, "C2", "Was the garage light on or off when the garage camera last saw it?", "off",
             lambda a: re.search(r"\boff\b", norm(a)) is not None and re.search(r"\bon\b", norm(a)) is None),
    Question(14, "C2", "Which app was on my screen at 3 pm?", "VS Code", lambda a: "code" in norm(a)),
    Question(15, "C2", "What level did I park on?", "3", lambda a: "3" in numbers(a)),
    Question(16, "C2", "What did Sam's voice note say?", "bring the blue folder; unknown accepted",
             lambda a: unknown(a) or all(w in norm(a) for w in ("blue", "folder"))),
    Question(17, "C2", "When was the screen showing the error, and what did it say?", "10:00:20-10:00:50, E-4417 disk full",
             lambda a: near(a, "10:00:35", 1) and re.search(r"\be-?4417\b", norm(a)) is not None and "disk" in norm(a)),
    Question(18, "C1", "What's the total on the invoice PDF?", "342.50 (unknown accepted)", invoice_total),
    Question(19, "C2", "What's the total on the invoice PDF?", "342.50 (unknown accepted; d1's ask names the PDF as not read)",
             invoice_total),
]

# per question: answers that must pass, and answers that must not
ANSWERS = {
    1: (["4821", "The gate code is 4821.", "four eight two one"], ["1234", "4281", "unknown"]),
    2: (["30", "30 days", "thirty days"], ["45", "45 days", "30 or 45", "unknown"]),
    3: (["ERR-503, 5 times", "ERR-503 (five times)", "503, 5"], ["ERR-429, 2", "ERR-503, 2", "ERR-503", "unknown"]),
    4: (["12:07", "12:07:31", "12:08"], ["12:30", "12:05", "unknown"]),
    5: (["moved to Thursday 9:30", "Thursday at 9:30 am", "thursday 09:30"], ["Friday 9:30", "Thursday 10:30", "unknown"]),
    6: (["unknown", "Unknown."], ["hunter2", "the password is 12345678", "not known"]),
    7: (["16:40", "4:40 pm", "4:41 PM"], ["11:20", "13:05", "16:50", "unknown"]),
    8: (["3", "three times"], ["2", "4", "one", "unknown"]),
    9: (["E-4417", "e-4417 disk full"], ["E-4471", "E-4416", "unknown"]),
    10: (["10:00", "10:00:20", "10:01"], ["10:05", "09:00", "unknown"]),
    11: (["SHIP 2.1 FRI", "Ship 2.1 Friday"], ["SHIP 2.0 FRI", "SHIP 2.1", "unknown"]),
    12: (["yes, 14:03", "Yes at 2:03 pm", "yes 14:04"], ["no", "yes, 14:30", "yes", "14:03"]),
    13: (["off", "Off."], ["on", "it was on", "unknown"]),
    14: (["VS Code", "Visual Studio Code"], ["Chrome", "unknown"]),
    15: (["3", "level 3", "Level three"], ["2", "level 2", "unknown"]),
    16: (["bring the blue folder", "unknown"], ["bring the red folder", "call me back"]),
    17: (["10:00:20-10:00:50, E-4417 disk full", "10:00 E-4417: disk full"],
         ["E-4417 disk full", "10:00:20, disk full", "11:00, E-4417 disk full"]),
    18: (["342.50", "342.50 EUR", "342.5", "342,50 \u20ac", "Total 342.50 EUR", "unknown"],
         ["324.50", "34.25", "342", "1342.50", "3425", "the total is 34.5"]),
}
ANSWERS[19] = ANSWERS[18]  # the same question, put to Codex with real d1 behind `ask`


def test_the_graders_grade_as_the_table_says():
    assert [q.n for q in QUESTIONS] == list(range(1, 20)) and set(ANSWERS) == {q.n for q in QUESTIONS}
    for q in QUESTIONS:
        right, wrong = ANSWERS[q.n]
        assert all(q.grade(a) for a in right), f"question {q.n} must accept {[a for a in right if not q.grade(a)]}"
        assert not any(q.grade(a) for a in wrong), f"question {q.n} must refuse {[a for a in wrong if q.grade(a)]}"
    assert [q.tier for q in QUESTIONS].count("C1") == 7 and [q.tier for q in QUESTIONS].count("C2") == 12
    abstaining = {q.n for q in QUESTIONS if q.grade("unknown")}
    assert abstaining == {6, 16, 18, 19}, "only question 6 expects unknown; 16, 18 and 19 (the invoice PDF no tool reads) accept it"


def stop_reason() -> str | None:
    """Why the remaining questions are skipped (the week's use is high or unreadable), else None."""
    if QUOTA["ran"] and QUOTA["used"] is None:
        return "weekly use unreadable"
    if QUOTA["used"] is not None and QUOTA["used"] >= STOP_AT:
        return f"the plan is at {QUOTA['used']:g}% of its use; the run stops at {STOP_AT:g}%"
    return None


def test_the_harness_reads_answers_events_and_the_quota_and_refuses_what_it_must(tmp_path, monkeypatch):
    assert answer_of("thinking\n**ANSWER: 4821**") == "4821"
    assert answer_of("ANSWER: no\nmore\nANSWER: unknown") == "unknown" and answer_of("4821") is None
    week, hours = {"used_percent": 14, "window_minutes": 10080, "resets_at": 1}, {"used_percent": 90, "window_minutes": 300}
    lines = [json.dumps({"payload": {"rate_limits": {"primary": {**week, "used_percent": 12.5}, "secondary": None}}}),
             "not json", json.dumps({"payload": {"info": 1}}), json.dumps({"payload": {"rate_limits": None}}),
             json.dumps({"payload": {"rate_limits": {"primary": week, "secondary": None}}})]
    assert codex.percent_in(lines) == 14.0 and codex.percent_in(["{}"]) is None
    # the week is read from either window, and never the 5-hour one
    assert codex.percent_in([json.dumps({"rate_limits": {"primary": hours, "secondary": week}})]) == 14.0
    for limits in ({"primary": hours, "secondary": None}, {"primary": None, "secondary": None}, {"primary": {"used_percent": 14}},
                   {"primary": {**week, "used_percent": None}}, {"primary": {**week, "used_percent": "14"}}):
        assert codex.percent_in([json.dumps({"rate_limits": limits})]) is None, limits
    monkeypatch.setitem(QUOTA, "ran", False)
    monkeypatch.setitem(QUOTA, "used", None)
    assert stop_reason() is None  # nothing has run yet
    monkeypatch.setitem(QUOTA, "ran", True)
    assert stop_reason() == "weekly use unreadable"
    monkeypatch.setitem(QUOTA, "used", 59.0)
    assert stop_reason() is None
    monkeypatch.setitem(QUOTA, "used", 60.0)
    assert "60%" in stop_reason()

    # the exec is offered mm's read tools only: a command line without --tools is refused
    argv = codex.argv_of(["codex"], "m", "high", spec_of("mm", tmp_path))
    assert "'--tools','ask,find,look,label'" in next(a for a in argv if a.startswith("mcp_servers.mm.args="))
    with pytest.raises(codex.CodexMissing, match="lists no tools"):
        no_tools = served_of(server_of(tmp_path / "home", tmp_path / "w0", tools=()))
        codex.argv_of(["codex"], "m", "high", spec_of("mm", tmp_path, served=no_tools))

    call = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mm", "tool": "find"}}
    assert codex.check_event(call, spec_of("mm")) is None and codex.check_event({"type": "turn.started"}, spec_of("mm")) is None
    assert codex.check_event({"type": "item.started", "item": {"type": "command_execution"}}, spec_of("mm"))
    for plan in ("item.started", "item.updated", "item.completed"):  # Codex's own plan list stays inside the sandbox
        assert codex.check_event({"type": plan, "item": {"type": "todo_list", "items": []}}, spec_of("mm")) is None
    assert codex.check_event({"type": "item.started", "item": {"type": "file_change"}}, spec_of("mm"))
    assert codex.check_event({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "x", "tool": "t"}}, spec_of("mm"))
    assert codex.check_event({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "codex",
                                                                    "tool": "list_mcp_resources"}}, spec_of("mm")) is None

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    codex.check_personal()
    (tmp_path / "AGENTS.md").write_text("")
    codex.check_personal()  # an empty file adds no message
    (tmp_path / "AGENTS.md").write_text("always reply in French")
    with pytest.raises(codex.CodexMissing, match="personal Codex instructions"):
        codex.check_personal()
