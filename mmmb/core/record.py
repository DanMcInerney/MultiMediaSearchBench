"""The result record of one exec and why an agent stops (moved from bench.py of this repository at bb498a6): how an exec ended
(`kind_of`), what it did that every configuration can count (`work_of`: views and shell commands), the record written for it
(`record`) and the reasons an agent stops (`Halt`, `stop_reason`). What a system adds to a record is its own: `Session.audit`
and `Session.reset` give fields the runner merges in."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Sequence

from mmmb.core import grade
from mmmb.core.config import Config
from mmmb.core.question import Question


STOP_AT = 95.0  # percent of the week's plan use at which an agent stops
DONE_KINDS = ("escaped", "timeout", "capped")  # problems that are a wrong answer: the exec ran and did not answer
STOP_KINDS = ("setup", "limit", "auth")  # problems after which the agent stops at once: nothing would count
ERRORS_STOP = 3  # `error:` results in a row after which the agent stops
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")  # a file Claude's `Read` shows as a picture


def kind_of(problem: str | None) -> str | None:
    """`timeout` for `timeout: no answer in 900 s`; None for no problem."""
    return problem.split(":", 1)[0] if problem else None


def is_view(tool: dict) -> bool:
    """Whether a call shows the agent a picture: Codex's `view_image`, or Claude's `Read` of a picture file."""
    args = tool.get("arguments")
    if tool.get("tool") == "view_image":
        return True
    path = (args.get("file_path") or args.get("path")) if isinstance(args, dict) else None
    return tool.get("tool") == "Read" and isinstance(path, str) and path.lower().endswith(IMAGE_SUFFIXES)


def work_of(tools: Sequence[dict]) -> dict:
    """How an exec worked that every configuration can count, from its calls (tools.json): the pictures it viewed and the shell
    (ffmpeg) commands it ran. The system's own counts (`jobs`, `windows`, `seconds`, `stopped`) are zero here and come with its
    `audit`, which the runner merges over them."""
    views = shells = 0
    for t in tools:
        if is_view(t):
            views += 1
        elif t.get("tool") in ("shell", "Bash"):
            shells += 1
    return {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": views, "shells": shells}


def record(q: Question, agent: str, config: Config, seed: int, exec_dir: str, at: str, out: dict) -> dict:
    """The result record of one exec of question `q` (no truth is stored: the report grades from the questions). `out` is the agent's
    result with the `answer` that the run read in its final message (`answer.parse_final`; the format's other fields are
    `grounded.result_fields`'). `exec_dir` is the exec's folder from the run's, `at` when it ended. A `set` question's result also holds
    `items_f1`, the F1 of the items its answer named (0 with a problem), told beside `passed`, which needs the exact set."""
    problem, answer = out.get("problem"), out.get("answer")
    tools = out.get("tools") or []
    tags = q.tags
    return {"qid": q.qid, "type": tags["modality"], "template": tags["template"], "level": tags["level"], "last": bool(tags.get("last")),
            "absent": tags["absent"], "seed": seed,
            "agent": agent, "config": config.name, "system": config.system, "variant": config.variant, "answer": answer,
            "passed": problem is None and grade.grade(q.grade, answer),
            "abstained": grade.abstained(answer, q.grade.ignore), "problem": problem, "seconds": out["seconds"],
            "turns": out.get("turns"), "denials": out.get("denials"), "calls": len(tools),
            "tools": dict(Counter(t["tool"] for t in tools)), "work": work_of(tools),
            "scope_all": sum(1 for t in tools if isinstance(t.get("arguments"), dict) and t["arguments"].get("scope") == "all"),
            "tokens": out.get("tokens"), "cost_usd": out.get("cost_usd"), "used_percent": out.get("used_percent"),
            "agent_model": out.get("model"), "agent_effort": out.get("effort"), "exec": exec_dir, "at": at,
            **({"items_f1": grade.items_f1(q.grade, answer if problem is None else None)} if q.grade.rule == "set" else {})}


class Halt:
    """Why an agent (or one of its configurations) stops: the first reason is kept and printed. `must_report_use`: the agent's
    results must hold its use of the plan's week, else the quota cannot be watched (`stop_reason`)."""

    def __init__(self, must_report_use: bool = False) -> None:
        self.must_report_use = must_report_use
        self.reason: str | None = None
        self.errors = 0  # `error:` results in a row
        self._lock = threading.Lock()

    def set(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
                print(f"stopping: {reason}", flush=True)

    def after(self, agent: str, rec: dict) -> None:
        """Count a result: the agent stops on its reason, or on ERRORS_STOP errors in a row."""
        with self._lock:
            self.errors = self.errors + 1 if kind_of(rec["problem"]) == "error" else 0
            streak = self.errors
        reason = stop_reason(agent, rec, self.must_report_use)
        if reason is None and streak >= ERRORS_STOP:
            reason = f"{agent}: {streak} errors in a row, the last {rec['problem']}"
        if reason:
            self.set(reason)


def stop_reason(agent: str, rec: dict, must_report_use: bool = False) -> str | None:
    """Why an agent stops after an exec: its session was not the one asked for, its plan is out or its login is, or its use of
    the week is at STOP_AT (for an agent that `must_report_use` also when it cannot be read: Codex's is read from the exec's own
    rollout)."""
    problem, used = rec["problem"] or "", rec["used_percent"]
    if kind_of(problem) in STOP_KINDS:
        return f"{agent}: {problem}"
    if must_report_use and used is None:
        return f"{agent}: the plan's weekly use cannot be read, so the quota cannot be watched"
    if used is not None and used >= STOP_AT:
        return f"{agent}: the plan is at {used:g}% of its use; the run stops at {STOP_AT:g}%"
    return None
