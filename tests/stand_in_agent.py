"""A stand-in `claude` or `codex` process, run through the real `claude.ask_claude` or `codex.ask_codex`, with the real `SessionSpec` the runner
makes (the deny words of `RunFolder.deny`, the server and instructions a session's `serve` gave). It finds the server as the agent does (Claude:
the `--mcp-config` file; Codex: the `-c mcp_servers.*` settings), starts the real server command, speaks JSON-RPC to it, and writes the stream as the
agent does: Claude's init event lists the tools the server listed (and `plan["extra_tools"]`, so a test can plant a tool the session should not
have), Codex's items are `mcp_tool_call`s. What it asks of the server is the test's: the middle of the script, a piece of Python with `call(tool,
arguments)` (the blocks of the server's answer), `jsons_of(blocks)`, `emit`, `plan`, `name` and `counter` in reach, that sets `final`.

Generalized from tests/mm/seam.py (which the mm system's seam tests use, with its own middle) for the listing system's, so that the plumbing is
written once. Written for this repository (G1's repair of F4 and shared cause 3)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from agent_spec import nothing_queued

from mmsb import systems
from mmsb.core.agents import claude, codex
from mmsb.core.agents.spec import SessionSpec
from mmsb.core.config import Config
from mmsb.core.contract import Case, Dump, Session
from mmsb.core.question import Question
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import case_of, reads_of

# stdlib only: it runs as the agent's command. argv: this script, the plan (a JSON file), then the arguments the agent is given.
HEAD = r'''
import json, os, queue, subprocess, sys, threading, tomllib

plan = json.load(open(sys.argv[1], encoding="utf-8"))
argv = sys.argv[2:]
sys.stdin.read()  # the prompt
if plan["agent"] == "claude":
    servers = json.load(open(argv[argv.index("--mcp-config") + 1], encoding="utf-8"))["mcpServers"]
    own = [t for t in argv[argv.index("--tools") + 1].split(",") if t]
else:  # Codex has the server in settings: `-c mcp_servers.mm.command='...'`, `.args=[...]`, `.env.X='...'`, `.required=true`, `.x=approve`
    servers = {}
    for setting in (argv[i + 1] for i, a in enumerate(argv) if a == "-c"):
        key, _, raw = setting.partition("=")
        if key.startswith("mcp_servers."):
            try:
                value = tomllib.loads("v = " + raw)["v"]
            except tomllib.TOMLDecodeError:
                value = raw  # (a bare word, which Codex reads as a string)
            node = servers
            *path, last = key.split(".")[1:]
            for part in path:
                node = node.setdefault(part, {})
            node[last] = value
(name, server), = servers.items()


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


# the server, started as the agent starts it: its command, arguments and environment, in this process's working folder and environment
process = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           env={**os.environ, **server.get("env", {})}, cwd=os.getcwd())
replies, ids = queue.Queue(), iter(range(1, 1000))


def pump():
    for raw in process.stdout:
        try:
            replies.put(json.loads(raw))
        except ValueError:
            continue


threading.Thread(target=pump, daemon=True).start()


def send(message):
    process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
    process.stdin.flush()


def rpc(method, params=None):
    n = next(ids)
    send({"jsonrpc": "2.0", "id": n, "method": method, "params": params or {}})
    while True:
        message = replies.get(timeout=180)
        if message.get("id") == n:
            return message


def jsons_of(blocks):
    out = []
    for block in blocks:
        try:
            got = json.loads(block["text"]) if block.get("type") == "text" else None
        except ValueError:
            got = None
        if isinstance(got, dict):
            out.append(got)
    return out


rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": plan["agent"], "version": "stand-in"}})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
listed = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
json.dump(listed, open(plan["listed"], "w"))
if plan["agent"] == "claude":
    emit({"type": "system", "subtype": "init", "session_id": "seam", "model": "claude-stand-in", "plugins": [], "apiKeySource": "none",
          "tools": [*own, *(f"mcp__{name}__{t}" for t in listed), *plan.get("extra_tools", [])],
          "mcp_servers": [{"name": name, "status": "connected"}]})
else:
    emit({"type": "turn.started"})

counter = iter(range(1, 1000))


def call(tool, arguments):
    """One call of a tool of the server: the call, the server's answer to it, and the answer's blocks."""
    use = f"toolu_{next(counter)}"
    if plan["agent"] == "claude":
        emit({"type": "assistant", "message": {"model": "claude-stand-in", "content": [
            {"type": "tool_use", "id": use, "name": f"mcp__{name}__{tool}", "input": arguments}]}})
    else:
        emit({"type": "item.started", "item": {"id": use, "type": "mcp_tool_call", "server": name, "tool": tool, "arguments": arguments, "status": "in_progress"}})
    reply = rpc("tools/call", {"name": tool, "arguments": arguments})
    blocks = reply["result"]["content"]
    if plan["agent"] == "claude":
        emit({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": use, "content": blocks,
                                                       "is_error": bool(reply["result"].get("isError"))}]}})
    else:
        emit({"type": "item.completed", "item": {"id": use, "type": "mcp_tool_call", "server": name, "tool": tool, "arguments": arguments,
                                                 "result": {"content": blocks, "structured_content": None}, "error": None, "status": "completed"}})
    return blocks

'''

END = r'''
if plan["agent"] == "claude":
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "result": final, "total_cost_usd": 0.0,
          "permission_denials": [], "usage": {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}})
else:
    emit({"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": final}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}})
process.stdin.close()
process.kill()
'''


def serve_case(run: RunFolder, session: Session, config: Config, question: Question, agent: str = "claude") -> Case:
    """The case of `question` in the first working slot, with the folder made, as `run_config` hands it to `serve`."""
    run.slot(0).mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(case_of(run, agent, config, question, session), slot=0, work=run.slot(0))


def run_agent(agent: str, root: Path, monkeypatch, git: Path, session: Session, run: RunFolder, dump: Dump, config: Config, question: Question,
              middle: str, plan: dict, off_gpu: bool = False) -> tuple[dict, list[str]]:
    """The result of `ask_claude` or `ask_codex` (`agent`) for `question` in `config` with a stand-in for the agent's command that does `middle`
    (see the module) over the real session, and the names of the tools the server listed to the stand-in. `plan` is what the middle reads, beside
    what the script keeps there. `git` is a Git for Windows folder tree for Claude's shell with its own tools. `off_gpu`: the agent is kept off the
    GPU, as the run keeps one with its own tools whose system needs it."""
    script, plan_file, listed = root / f"{agent}_stand_in.py", root / "plan.json", root / "listed.json"
    script.write_text(HEAD + middle + END, encoding="utf-8")
    plan_file.write_text(json.dumps({**plan, "agent": agent, "listed": str(listed)}), encoding="utf-8")
    command = lambda: [sys.executable, str(script), str(plan_file)]  # noqa: E731
    monkeypatch.setattr(claude, "claude_command", command)
    monkeypatch.setattr(codex, "codex_command", command)
    monkeypatch.setattr(claude, "git_root", lambda: git)
    monkeypatch.setattr(codex, "models_of", lambda _: {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]})
    case = serve_case(run, session, config, question, agent)
    served = session.serve(case)
    data = run.scenario.view(question, 0, dump, run.temp, run.tag, True) if config.own_tools else None
    isolation = systems.isolation()
    spec = SessionSpec("seam", case.run_dir, case.work, config.own_tools, data, run.bin, served,
                       run.deny(case.work, isolation, reads_of(data, dump), off_gpu), isolation, 180.0, nothing_queued, "gpt-test" if agent == "codex" else None,
                       "high" if agent == "codex" else None, off_gpu)
    out = (claude.ask_claude if agent == "claude" else codex.ask_codex)(spec)
    assert listed.exists(), f"the stand-in never reached the server's tool list: {out['problem']}: {out['stderr']}"
    return out, json.loads(listed.read_text(encoding="utf-8"))
