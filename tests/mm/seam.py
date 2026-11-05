"""The seam between the real session and the real agent, which no other test crosses: tests/mm has the real `MmSession` with stand-in agents that call
the daemon's `Api` themselves, and tests/replay.py has the real agents with a stand-in session. Here a stand-in `claude` or `codex` process is run
through the real `claude.ask_claude` or `codex.ask_codex`, with the real `SessionSpec` the runner makes (the deny words of `RunFolder.deny`, the server
and instructions of `MmSession.serve`). It finds the server as the agent does (Claude: the `--mcp-config` file; Codex: the `-c mcp_servers.*`
settings), starts the real server command `serve` returned, speaks JSON-RPC to it over a throwaway home with the fake decider, and writes the stream as
the agent does: Claude's init event lists the tools the server listed, Codex's items are `mcp_tool_call`s. What it asks is `find`, `ask` and `look`; with
the agent's own tools too it then opens the file `look` named (Claude: a `Read`; Codex: a shell command, as it has no read tool). The stand-in's
plumbing is tests/stand_in_agent.py's, which the listing system's seam test uses too; what is mm's is what it asks.

Written for this repository (G1's repair of F4 and shared cause 3)."""

from __future__ import annotations

from pathlib import Path

import stand_in_agent

from mmmb.core.config import Config
from mmmb.core.contract import Dump, Session
from mmmb.core.question import Question
from mmmb.core.runfolder import RunFolder

# what the stand-in does of mm's server (tests/stand_in_agent.py has the rest): find, ask until it ends, look at the first sighting, and with its own tools open the picture
MIDDLE = r'''
call("find", {"scope": "all", "limit": 2})
asked = jsons_of(call("ask", plan["ask"]))[0]
while asked.get("state") in ("queued", "running"):
    asked = jsons_of(call("ask", {"job": asked["job"]}))[0]
handle = next((s["look"] for s in asked.get("sightings", []) if s.get("look")), None)
if handle is None:
    sys.exit("the ask found nothing to look at: " + json.dumps(asked)[:500])
seen = next(m for m in jsons_of(call("look", {"moments": [handle]})) if m.get("image"))
if plan["read"]:  # the agent's own tools: it opens the picture `look` named, at full size
    use, path = f"toolu_{next(counter)}", seen["image"]["path"]
    if plan["agent"] == "claude":
        emit({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": use, "name": "Read", "input": {"file_path": path}}]}})
        emit({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": use, "content": [{"type": "text", "text": "[image]"}]}]}})
    else:
        shell = {"id": use, "type": "command_execution", "command": f'ffmpeg -i "{path}" -vf scale=640:-1 seen.png', "aggregated_output": "", "exit_code": 0}
        emit({"type": "item.started", "item": {**shell, "status": "in_progress"}})
        emit({"type": "item.completed", "item": {**shell, "status": "completed"}})
final = "ANSWER: unknown"
if plan["answer"] is not None:  # the json format: the answer given, citing each sighting of the ask by its source and its ISO times
    cites = [{"source": s["source"], "t0": s.get("t0", s.get("t")), "t1": s.get("t1", s.get("t"))} for s in asked.get("sightings", [])]
    final = json.dumps({"answer": plan["answer"], "evidence": cites})
'''


def run_agent(agent: str, root: Path, monkeypatch, git: Path, session: Session, run: RunFolder, dump: Dump, config: Config, question: Question,
              ask: dict, answer: str | None = None) -> tuple[dict, list[str]]:
    """The result of `ask_claude` or `ask_codex` (`agent`) for `question` in `config`, with a stand-in for the agent's command (see the module) over the
    real session, and the names of the tools the server listed to the stand-in. `ask` is the arguments of the `ask` it makes. `git` is a Git for
    Windows folder tree for Claude's shell with its own tools. It ends with `ANSWER: unknown`, or with `answer` as a JSON answer that cites each
    sighting the ask returned (`source` and ISO times, which is all an agent with the system's tools alone has)."""
    plan = {"ask": ask, "read": config.own_tools, "answer": answer}
    return stand_in_agent.run_agent(agent, root, monkeypatch, git, session, run, dump, config, question, MIDDLE, plan)
