"""Ported from tests/test_agents.py of Multimem at fe0ece5 (plan 7.4, oracle 4): the benchmark's two agents, Claude Code and Codex, in
all four modes of an exec (blind, files, a server, a server and files) (TESTBED.md, Benchmark): the command lines, the checks of what
a session may use, and the reading of a stream. No real `claude` or `codex` runs: a command line is built and read, an event is made
up, and the few tests that start a process start a stand-in script that prints made-up events.

The calls go through a `SessionSpec` (tests/agent_spec.py); the expectations of argv, environment, messages and results are the
source's. What changed with the move, by the plan's 3.3 and 4.4:
- the agents get their server and instructions from `Served`, so no test here reads an installed skill or its notes (those assertions
  are in tests/mm), and the tests that built the home, `notes`, `guidance` or `arm` of the old calls build a spec of the mode instead;
- a home or a working folder that was missing is not a case any more (the spec has them), and the two refusals about it are gone."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from agent_spec import BASE_LINE, ISOLATION, NO_ISOLATION, served_of, server_of, spec_of

from mmsb.core.agents import claude, codex, common
from mmsb.core.agents.spec import SessionSpec
from mmsb.core.answer import answer_of
from mmsb.core.contract import McpServer, Served

CONTEXT_FULL = ("Codex ran out of room in the model's context window. Start a new thread or clear earlier history before "
                "retrying.")  # as codex.exe words it
MM_TOOLS = ("mcp__mm__ask", "mcp__mm__find", "mcp__mm__look", "mcp__mm__label")  # written by hand: what a session of the server lists
RECORDED = Path(__file__).parent / "oracles" / "recorded" / "exec"
STAND_IN = """\
import json, os, sys, time
for line in open(sys.argv[1], encoding="utf-8"):
    sys.stdout.write(line)
    sys.stdout.flush()
json.dump({"env": dict(os.environ), "cwd": os.getcwd()}, open(sys.argv[3], "w"))
time.sleep(float(sys.argv[2]))
"""


def after(argv: list[str], flag: str) -> list[str]:
    """The values that follow `flag`, up to the next flag."""
    start = argv.index(flag) + 1
    return argv[start:next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))]


def settings_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


def disabled_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]


def lower(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


# -- Claude: the command lines and the environment ----------------------------------------------------------------------

def test_claudes_command_lines_give_each_arm_its_tools_and_nothing_of_the_owners(tmp_path, monkeypatch):
    spec = spec_of("mm", tmp_path)
    run, work, home, data = spec.run_dir, spec.work, tmp_path / "home", tmp_path / "data"
    argv = claude.prepare(spec, command=["claude"])
    assert argv[:3] == ["claude", "-p", "--verbose"] and after(argv, "--output-format") == ["stream-json"]
    for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert after(argv, "--permission-mode") == ["dontAsk"] and "--add-dir" not in argv
    assert after(argv, "--tools") == [""], "no built-in tool: only the MCP server's"
    assert after(argv, "--allowedTools") == ["mcp__mm__ask", "mcp__mm__find", "mcp__mm__look", "mcp__mm__label"]
    assert after(argv, "--mcp-config") == [str(run / "mcp.json")]
    assert after(argv, "--settings") == [str(run / "settings.json")]
    assert json.loads((run / "settings.json").read_text()) == {
        "disableAllHooks": True, "enabledPlugins": {"cc-plugin-agents-md@builtin": False,
                                                    "cc-plugin-plugin-authoring@builtin": False}}
    servers = json.loads((run / "mcp.json").read_text())["mcpServers"]
    assert list(servers) == ["mm"], "one server"
    assert servers["mm"] == {"command": sys.executable, "env": {"CLAUDE_PROJECT_DIR": str(work)},
                             "args": ["-m", "multimem.cli", "--home", str(home), "mcp", "--tools", "ask,find,look,label"]}
    assert after(argv, "--append-system-prompt-file") == [str(run / "instructions.md")]
    assert (run / "instructions.md").read_text() == f"{BASE_LINE}\n\nTHE SKILL\n\nTHE NOTES"
    assert work.is_dir()

    files = spec_of("files", tmp_path, run_dir=tmp_path / "files-run")
    argv = claude.prepare(files, command=["claude"])
    assert after(argv, "--tools") == ["Read,Glob,Grep,Bash"]
    assert after(argv, "--allowedTools") == ["Read", "Glob", "Grep", "Bash(ffmpeg *)"]
    assert after(argv, "--add-dir") == [str(data)]
    assert "--mcp-config" not in argv and "--strict-mcp-config" in argv, "no MCP server at all"
    assert not (tmp_path / "files-run" / "mcp.json").exists()
    assert (tmp_path / "files-run" / "instructions.md").read_text() == claude.BASE_FILES
    with pytest.raises(ValueError, match="data"):
        claude.argv_of(["claude"], spec_of("files", tmp_path, data=None), name="sonnet", level="high")

    # the two flags `claude --help` leaves out each have one switch
    assert after(argv, "--max-turns") == ["60"] and "--max-budget-usd" in argv
    monkeypatch.setattr(claude, "PROMPT_FILE", False)
    monkeypatch.setattr(claude, "MAX_TURNS", None)
    argv = claude.argv_of(["claude"], files, name="sonnet", level="high")
    assert after(argv, "--append-system-prompt") == [claude.BASE_FILES]
    assert "--append-system-prompt-file" not in argv and "--max-turns" not in argv
    # the longest instructions on record (the skill and the notes: 25,862 characters) still fit the command line
    long = (RECORDED / "claude-mm" / "instructions.md").read_text(encoding="utf-8")
    assert len(long) > 25_000
    on_the_line = served_of(server_of(home, work), parts=(long.split("\n\n", 1)[1],))
    claude.prepare(spec_of("mm", tmp_path, run_dir=tmp_path / "in-the-line", served=on_the_line), command=["claude"])
    if os.name == "nt":
        too_long = served_of(server_of(home, work), parts=(long, long))
        with pytest.raises(claude.ClaudeMissing, match="command line"):
            claude.prepare(spec_of("mm", tmp_path, run_dir=tmp_path / "too-long", served=too_long), command=["claude"])


def test_claudes_environment_drops_the_api_key_and_the_instructions_are_the_modes_line_and_what_is_served(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-test")
    bin_dir = tmp_path / "bin"
    env = claude.env_of(bin_dir)
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert all(env[k] == v for k, v in claude.ENV_SET.items())
    assert env["PATH"].startswith(str(bin_dir) + os.pathsep) and os.environ["ANTHROPIC_API_KEY"] == "sk-test"
    assert claude.env_of(None)["PATH"] == os.environ["PATH"]

    def written(spec: SessionSpec) -> str:
        claude.prepare(spec, command=["claude"])
        return (spec.run_dir / "instructions.md").read_text(encoding="utf-8")

    server = server_of(tmp_path / "home", tmp_path / "w0")
    assert written(spec_of("mm", tmp_path / "a")) == f"{BASE_LINE}\n\nTHE SKILL\n\nTHE NOTES"
    assert written(spec_of("mm", tmp_path / "b", served=served_of(server, parts=("THE SKILL",)))) == f"{BASE_LINE}\n\nTHE SKILL"
    told = served_of(server, server_instructions="THE SERVER'S OWN WORDS")
    assert "THE SERVER'S OWN WORDS" not in written(spec_of("mm", tmp_path / "c", served=told)), "Claude gets them through the server"
    # a mode without a server never holds what a system serves: the skill and the notes stay out of the files and blind lines
    held = Served(server=None, base_line=BASE_LINE, parts=("THE SKILL", "THE NOTES"), server_instructions="X", holder="mm",
                  clock_note=None, record={})
    assert written(spec_of("files", tmp_path / "d", served=held)) == claude.BASE_FILES
    assert written(spec_of("blind", tmp_path / "e", served=held)) == common.BASE_BLIND
    # and with a server and its own tools, the files line comes first
    both = written(spec_of("mm+files", tmp_path / "f"))
    assert both == f"{claude.BASE_FILES}\n\n{BASE_LINE}\n\nTHE SKILL\n\nTHE NOTES"
    with pytest.raises(ValueError, match="base line"):
        written(spec_of("mm", tmp_path / "g", served=served_of(server, base_line=None)))


@pytest.fixture
def fake_git(tmp_path, monkeypatch):
    """A Git for Windows folder tree, with `claude.git_root` pointed at it: the real one is the owner's."""
    root = tmp_path / "Git"
    for sub in ("cmd", "bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("")
    (root / "usr" / "bin" / "bash.exe").write_text("")
    monkeypatch.setattr(claude, "git_root", lambda: root)
    return root


def test_claudes_files_arm_shell_cannot_reach_the_owners_mm(tmp_path, monkeypatch, fake_git):
    """Under `uv run` PATH starts with the repo's venv folder (mm.exe), and MULTIMEM_HOME may be the owner's."""
    venv, bin_dir, work = tmp_path / ".venv" / "Scripts", tmp_path / "bin", tmp_path / "w0"
    venv.mkdir(parents=True)
    (venv / "mm.exe").write_text("")
    (venv / "mm.exe").chmod(0o755)
    monkeypatch.setenv("PATH", str(venv) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    monkeypatch.setenv("MULTIMEM_HOME", str(tmp_path / "owners-home"))
    monkeypatch.setenv("MULTIMEM_MODELS", "D:/models")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert shutil.which("mm.exe", path=os.environ["PATH"]), "the owner's PATH does reach mm"

    env = claude.env_of(bin_dir, True, work, ISOLATION)
    folders = env["PATH"].split(os.pathsep)
    assert folders[0] == str(bin_dir)
    assert folders[1:4] == [str(fake_git / "cmd"), str(fake_git / "bin"), str(fake_git / "usr" / "bin")]
    assert folders[4:] == common.system_path(), "then the system's folders and nothing else"
    assert not any(".venv" in f for f in folders) and shutil.which("mm.exe", path=env["PATH"]) is None
    assert env["MULTIMEM_HOME"] == str(work / "no-mm-home") and not Path(env["MULTIMEM_HOME"]).exists()
    assert not {"VIRTUAL_ENV", "MULTIMEM_MODELS", "ANTHROPIC_API_KEY"} & set(env)
    assert env["CLAUDE_CODE_GIT_BASH_PATH"] == str(fake_git / "bin" / "bash.exe")
    assert all(env[k] == v for k, v in claude.ENV_SET.items())

    env = claude.env_of(bin_dir)
    assert str(venv) in env["PATH"] and env["MULTIMEM_HOME"] == str(tmp_path / "owners-home"), "the mm arm is as it was"
    with pytest.raises(ValueError, match="working folder"):
        claude.env_of(bin_dir, True)
    with pytest.raises(ValueError, match="isolation"):
        claude.env_of(bin_dir, True, work)
    if os.name == "nt":
        monkeypatch.setattr(claude, "git_root", lambda: None)
        with pytest.raises(claude.ClaudeMissing, match="Git for Windows"):
            claude.env_of(bin_dir, True, work, ISOLATION)


def test_what_the_shell_of_own_tools_keeps_out_is_what_the_isolation_says_and_nothing_else(tmp_path, monkeypatch, fake_git):
    """The words are the systems' (system.toml): the same environment gives a different shell when the isolation differs."""
    work = tmp_path / "w0"
    monkeypatch.setenv("MULTIMEM_HOME", str(tmp_path / "owners-home"))
    monkeypatch.setenv("MULTIMEM_MODELS", "D:/models")
    monkeypatch.setenv("OTHER_THING_HOME", "kept")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    none = claude.env_of(tmp_path / "bin", True, work, NO_ISOLATION)
    assert none["MULTIMEM_HOME"] == str(tmp_path / "owners-home") and none["MULTIMEM_MODELS"] == "D:/models"
    assert "VIRTUAL_ENV" not in none, "the virtual environment is the agent's own business, not a system's"
    other = claude.env_of(tmp_path / "bin", True, work, dataclasses.replace(
        NO_ISOLATION, env_drop_prefixes=("OTHER_",), env_nowhere={"ANOTHER": "second"}))
    assert "OTHER_THING_HOME" not in other and other["ANOTHER"] == str(work / "second")
    assert other["MULTIMEM_HOME"] == str(tmp_path / "owners-home"), "only what the isolation names"


@pytest.mark.skipif(os.name != "nt", reason="Git for Windows")
def test_git_for_windows_is_found_from_git_or_bash_and_not_from_wsls_bash(tmp_path):
    root, system = tmp_path / "Program Files" / "Git", tmp_path / "Windows" / "System32"
    for sub in ("cmd", "mingw64/bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    system.mkdir(parents=True)
    for exe in ("cmd/git.exe", "mingw64/bin/git.exe", "usr/bin/bash.exe"):
        (root / exe).write_text("")
    (system / "bash.exe").write_text("")
    assert claude.git_root(str(root / "cmd")) == root and claude.git_root(str(root / "mingw64" / "bin")) == root
    assert claude.git_root(str(system) + os.pathsep + str(root / "usr" / "bin")) == root, "from bash when no git"
    assert claude.git_root(str(system)) is None, "the bash of System32 is WSL's, not Git's"


# -- Claude: the stream ------------------------------------------------------------------------------------------------

def init(tools=MM_TOOLS, servers=({"name": "mm", "status": "connected"},), **extra) -> dict:
    return {"type": "system", "subtype": "init", "tools": list(tools), "mcp_servers": list(servers), "plugins": [],
            "model": "claude-sonnet-x", "apiKeySource": "none", **extra}


def use(name: str, arguments: dict, id: str = "t1") -> dict:
    return {"type": "assistant", "message": {"model": "claude-sonnet-x", "content": [
        {"type": "tool_use", "id": id, "name": name, "input": arguments}]}}


def says(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def finished(**fields) -> dict:
    return {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": "It was 47.\nANSWER: 47",
            "total_cost_usd": 0.0421, "permission_denials": [],
            "usage": {"input_tokens": 120, "output_tokens": 45, "cache_read_input_tokens": 3000,
                      "cache_creation_input_tokens": 800}, **fields}


def test_a_session_that_is_not_the_arms_is_a_setup_problem_and_a_tool_out_of_bounds_an_escape(tmp_path):
    files = {"tools": claude.FILES_TOOLS, "servers": ()}
    assert claude.check_init(init(), spec_of("mm")) is None and claude.check_init(init(**files), spec_of("files")) is None
    assert "Bash" in claude.check_init(init(tools=[*MM_TOOLS, "Bash"]), spec_of("mm"))
    assert "mcp__mm__ask" in claude.check_init(init(tools=MM_TOOLS[1:]), spec_of("mm"))
    extra = [{"name": "mm", "status": "connected"}, {"name": "claude.ai Gmail", "status": "connected"}]
    assert "Gmail" in claude.check_init(init(servers=extra), spec_of("mm"))
    assert "not connected" in claude.check_init(init(servers=[{"name": "mm", "status": "failed"}]), spec_of("mm"))
    assert "not just 'mm'" in claude.check_init(init(servers=[{"name": "other", "status": "connected"}]), spec_of("mm"))
    assert claude.check_init(init(servers=[]), spec_of("mm")), "no server is not the mm arm"
    assert "MCP servers" in claude.check_init(init(tools=claude.FILES_TOOLS), spec_of("files")), "init() has the mm server"
    assert claude.check_init(init(tools=MM_TOOLS), spec_of("files")), "the mm arm's tools are not the files arm's"
    assert claude.check_init(init(plugins=[{"name": "codex", "path": "x"}]), spec_of("mm"))
    assert "API key" in claude.check_init(init(apiKeySource="ANTHROPIC_API_KEY"), spec_of("mm"))
    assert claude.check_init({"type": "system", "subtype": "init"}, spec_of("mm")), "an init that says nothing fails closed"

    deny = (lower(tmp_path / "answers"), ".claude")
    own = tmp_path / "data" / "notes" / "note-2026-10-28_09-12.md"
    assert claude.check_event(use("mcp__mm__find", {"scope": "all"}), spec_of("mm")) is None
    assert "'Write'" in claude.check_event(use("Write", {"file_path": "x"}), spec_of("files", deny=deny))
    assert "'Write'" in claude.check_event(use("Write", {"file_path": "x"}), spec_of("mm"))
    assert claude.check_event(use("mcp__mm__remember", {"path": "x"}), spec_of("mm")), "remember is not offered"
    assert claude.check_event(use("Read", {"file_path": str(own)}), spec_of("mm")), "the mm arm has no Read"
    assert claude.check_event(use("Read", {"file_path": str(own)}), spec_of("files", deny=deny)) is None
    assert claude.check_event(use("Bash", {"command": f"ffmpeg -i {own} -vf fps=1 f%d.png"}), spec_of("files", deny=deny)) is None
    hit = claude.check_event(use("Read", {"file_path": str(tmp_path / "Answers" / "x.json")}), spec_of("files", deny=deny))
    assert hit == f"Read reached {deny[0]}", "a path is read in lower case with forward slashes"
    assert claude.check_event(use("Grep", {"path": "C:\\Users\\dan\\.Claude\\settings.json"}), spec_of("files", deny=deny))
    assert claude.check_event(use("Glob", {"pattern": "**/*", "path": "~/.claude"}), spec_of("files", deny=deny))
    assert claude.check_event(use("Read", {"file_path": ".claude"}), spec_of("files")) is None, "no deny list, nothing denied"
    assert claude.check_event({"type": "assistant", "message": {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {}}]}}, spec_of("files", deny=deny))
    assert claude.check_event(says("ANSWER: 47"), spec_of("files", deny=deny)) is None
    assert claude.check_event(init(), spec_of("mm")) is None

    watch = claude.watcher(spec_of("mm"))
    missing = watch(use("mcp__mm__find", {}))
    assert missing.startswith("setup: ") and "init" in missing, "a message before the init event: not the session asked for"
    watch = claude.watcher(spec_of("mm"))
    wrong = watch(init(tools=[*MM_TOOLS, "Bash"]))
    assert wrong.startswith("setup: ") and "Bash" in wrong and watch(init()) is None
    assert watch(use("mcp__mm__ask", {})) is None and watch({"type": "user"}) is None
    assert watch(use("Bash", {})).startswith("escaped: claude used 'Bash'"), "what Claude does is its own"
    watch = claude.watcher(spec_of("files", deny=deny))
    assert watch(init(tools=claude.FILES_TOOLS, servers=())) is None
    assert watch(use("Read", {"file_path": ".claude/x"})) == "escaped: Read reached .claude"


def test_the_files_arm_may_read_back_its_own_saved_output_and_nothing_else_under_claude():
    """A Bash output over Claude Code's limit is saved under ~/.claude/projects/<folder>/<session>/tool-results, and its result
    names the file (2.1.302): the files arm may read that file back, not another session's or anything else under ~/.claude."""
    sid = "eb90273e-5f48-4dad-b3d2-2370a8a15ac1"
    saved = f"C:\\Users\\dan\\.claude\\projects\\D--mm-bench-temp-mmb-w2\\{sid}\\tool-results\\bbelb0265.txt"
    deny = (".claude",)
    big = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content":
           f"<persisted-output>\nOutput too large (2.4MB). Full output saved to: {saved}\n\nPreview (first 2KB):\nInput #0\n"}]}}

    def session(sid: str):
        watch = claude.watcher(spec_of("files", deny=deny))
        assert watch(init(tools=claude.FILES_TOOLS, servers=(), session_id=sid)) is None
        return watch

    watch = session(sid)
    assert watch(use("Bash", {"command": "ffmpeg -i a.ogg -af astats -f null -"})) is None and watch(big) is None
    assert watch(use("Read", {"file_path": saved, "offset": 100}, "t2")) is None
    assert watch(use("Read", {"file_path": saved.replace("\\", "/")}, "t3")) is None
    assert watch(use("Grep", {"pattern": "RMS", "path": saved.rsplit("\\", 1)[0]}, "t4")) is None, "the folder itself"
    for path in (saved.replace(sid, "0b7c-another-session"), saved.replace("bbelb0265.txt", "..\\..\\..\\..\\settings.json"),
                 "C:\\Users\\dan\\.claude\\settings.json", saved.replace("tool-results", "subagents")):
        assert session(sid)(use("Read", {"file_path": path})) == "escaped: Read reached .claude", path
    assert session("0b7c-another-session")(big) == "escaped: a tool output reached .claude"
    assert claude.watcher(spec_of("files", deny=deny))(big) == "escaped: a tool output reached .claude", "no session known: none is own"


def test_a_made_up_run_gives_its_answer_tools_turns_tokens_and_cost():
    find = use("mcp__mm__find", {"since": "2026-10-28", "scope": "all"})
    got = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                                                    "content": [{"type": "text", "text": '{"items":[]}'}]}]}}
    events = [init(), find, got, says("It was 47.\nANSWER: 47"), finished()]
    done = claude.result_of(events, 0, "")
    assert done["problem"] is None and answer_of(done["final"]) == "47"
    assert (done["turns"], done["cost_usd"], done["model"], done["denials"]) == (2, 0.0421, "claude-sonnet-x", 0)
    assert done["tokens"] == {"input": 120, "output": 45, "cache_read": 3000, "cache_write": 800}
    tools = claude.tools_of(events)
    assert [t["tool"] for t in tools] == ["find"] and tools[0]["arguments"]["scope"] == "all"
    assert tools[0]["result"] == [{"type": "text", "text": '{"items":[]}'}] and tools[0]["error"] is False
    twice = [init(), find, find, use("Read", {}, "t2"), finished()]
    assert [t["tool"] for t in claude.tools_of(twice)] == ["find", "Read"]
    # the prefix of any server comes off, the name of a tool of its own does not
    other = [use("mcp__listing__list_files", {}, "a"), use("mcp__a_b__x__y", {}, "b"), use("Bash", {}, "c"), use("mcp_x", {}, "d")]
    assert [t["tool"] for t in claude.tools_of(other)] == ["list_files", "x__y", "Bash", "mcp_x"]

    assert claude.result_of([init(), says("thinking"), finished(result="")], 0, "")["final"] == "thinking"
    assert claude.result_of([init(), finished(permission_denials=[{"tool_name": "Bash"}] * 2)], 0, "")["denials"] == 2
    cut = claude.result_of([init(), says("partial"), {"type": "result", "subtype": "error_max_turns", "is_error": True,
                                                      "num_turns": 60}], 1, "")
    assert cut["problem"] == "capped: error_max_turns" and cut["final"] == "partial" and cut["tokens"] is None
    assert claude.result_of([init(), finished(subtype="error_max_budget_usd", is_error=True, result="")], 1,
                            "")["problem"] == "capped: error_max_budget_usd"
    assert claude.result_of([init(), finished(subtype="error_during_execution", is_error=True, result="")], 1,
                            "")["problem"] == "error: error_during_execution"
    assert claude.result_of([init(), finished(is_error=True, result="API Error: 500")], 1, "")["problem"] == \
        "error: API Error: 500"


def test_a_limit_a_login_and_a_missing_result_are_told_apart():
    def problem(result: dict | None, code: int = 1, stderr: str = "") -> str | None:
        return claude.result_of([init(), *([result] if result else [])], code, stderr)["problem"]

    assert problem(finished(is_error=True, result="You've hit your weekly limit \u00b7 resets Nov 5")).startswith("limit: ")
    assert problem(finished(is_error=True, result="Claude usage limit reached")).startswith("limit: ")
    assert problem(finished(is_error=True, result="API Error: 429 rate_limit_error")).startswith("limit: ")
    assert problem(finished(is_error=True, result="Invalid API key \u00b7 Please run /login")).startswith("auth: ")
    assert problem(finished(is_error=True, result="Not logged in")).startswith("auth: ")
    expired = "Failed to authenticate: OAuth session expired and could not be refreshed"  # seen in the 2026-11-02 smoke
    assert problem(finished(is_error=True, result=expired)) == f"auth: {expired}"
    assert problem(finished(is_error=True, result="API Error: 500"), stderr="rate limit exceeded").startswith("limit: ")
    assert problem(None, 3) == "error: claude exited 3 without a result"
    refused = "error: unknown option '--append-system-prompt-file'\n(Did you mean --append-system-prompt?)\n"
    assert problem(None, 1, refused) == "setup: error: unknown option '--append-system-prompt-file'"
    assert problem(None, 1, "error: option '--effort <level>' argument 'x' is invalid. Allowed choices are low.") \
        .startswith("setup: error: option '--effort <level>' argument 'x' is invalid")
    assert problem(None, 1, "error: required option '--model <m>' not specified").startswith("setup: ")
    assert problem(None, 1, "error: too many arguments. Expected 0 arguments but got 1.").startswith("setup: ")
    own = problem(finished(is_error=True, result="API Error: 500"), 1, refused)
    assert own == "error: API Error: 500", "a result is the session's own word"
    assert problem(finished(subtype="error_max_turns", is_error=True, result="")) == "capped: error_max_turns"
    spent = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True}
    assert problem(spent) == "capped: error_max_budget_usd"
    assert problem(None, 1, "Claude AI usage limit reached").startswith("limit: ")
    assert problem(finished(result="the answer mentions a quota")) is None, "a success is never a problem"

    def week(**info) -> dict:
        return {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", **info}}

    assert claude.used_percent_of([init(), finished()]) is None
    assert claude.used_percent_of([week(rateLimitType="five_hour", utilization=0.9)]) is None
    assert claude.used_percent_of([week(rateLimitType="seven_day", utilization=0.25)]) == pytest.approx(25.0)
    windows = {"five_hour": {"utilization": 0.9, "resetsAt": 1}, "seven_day": {"utilization": 0.61, "resetsAt": 2}}
    both = week(rateLimitType="five_hour", utilization=0.9, unifiedWindows=windows)
    assert claude.used_percent_of([both]) == pytest.approx(61.0)
    later = week(rateLimitType="seven_day", utilization=0.3)
    assert claude.used_percent_of([week(unifiedWindows=windows), later]) == pytest.approx(30.0), "the last one counts"
    assert claude.used_percent_of([week(rateLimitType="seven_day", utilization="x")]) is None


def test_output_that_holds_a_deny_string_ends_the_files_arm_for_both_agents(tmp_path):
    deny = (lower(tmp_path / "mm-bench"), ".multimem")
    leak = f"{tmp_path}/mm-bench/run1/results/claude/files/x.json:3: 47"

    def returned(content, id: str = "g1") -> dict:
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "content": content}]}}

    watch = claude.watcher(spec_of("files", deny=deny))
    assert watch(init(tools=claude.FILES_TOOLS, servers=())) is None
    assert watch(use("Grep", {"pattern": "ERR-731", "path": str(tmp_path / "data")}, "g1")) is None
    assert watch(returned("data/logs/payments.jsonl:12: ERR-731")) is None
    assert watch(returned(leak)) == f"escaped: Grep output reached {deny[0]}"
    assert watch(returned([{"type": "text", "text": f"see {tmp_path / 'MM-Bench' / 'run1'}"}])) == \
        f"escaped: Grep output reached {deny[0]}", "a path in a list of blocks, in any case"
    assert watch(returned("/home/dan/.MultiMem/index.db", "unknown-id")) == "escaped: a tool output reached .multimem"
    picture = {"type": "image", "source": {"type": "base64", "data": ".multimem" * 20}}
    assert watch(returned([picture, {"type": "text", "text": "ok"}])) is None, "a picture's data is not text"
    watch = claude.watcher(spec_of("mm", deny=deny))
    assert watch(init()) is None and watch(use("mcp__mm__find", {}, "g1")) is None
    assert watch(returned(leak)) is None, "only the files arm has answers to keep from"
    assert claude.check_result(returned(leak), spec_of("files"), {}) is None

    def output(text: str, kind: str = "item.completed") -> dict:
        return {"type": kind, "item": {"type": "command_execution", "command": "rg ERR-731 .", "aggregated_output": text,
                                        "exit_code": 0}}

    assert codex.check_event(output("data/logs/payments.jsonl:12: ERR-731"), spec_of("files", deny=deny)) is None
    assert codex.check_event(output(leak), spec_of("files", deny=deny)) == f"shell output reached {deny[0]}"
    assert codex.check_event(output(f"{tmp_path}\\MM-BENCH\\run1\\x"), spec_of("files", deny=deny)) == \
        f"shell output reached {deny[0]}", "backslashes and case do not matter"
    home_file = output("C:/Users/dan/.multimem/index.db")
    assert codex.check_event(home_file, spec_of("files", deny=deny)) == "shell output reached .multimem"
    assert codex.check_event(output(leak), spec_of("files")) is None and codex.check_event(output(""), spec_of("files", deny=deny)) is None


# -- the server and the own tools together: a server's tool is checked by name, the own tools by the deny words ------------

def test_with_a_server_and_own_tools_only_the_own_tools_are_held_to_the_deny_words(tmp_path):
    deny = (lower(tmp_path / "answers"), ".multimem")
    both = spec_of("mm+files", deny=deny)
    assert claude.session_tools(both) == (*claude.FILES_TOOLS, *MM_TOOLS)
    both_init = init(tools=[*claude.FILES_TOOLS, *MM_TOOLS])
    assert claude.check_init(both_init, both) is None
    assert "Bash" in claude.check_init(init(), both), "the server alone is not the session asked for"
    assert "mcp__mm__ask" in claude.check_init(init(tools=claude.FILES_TOOLS, servers=()), both)
    assert claude.check_init(init(tools=[*claude.FILES_TOOLS, *MM_TOOLS], servers=[]), both), "no server is not the session asked for"
    assert claude.check_init(init(tools=MM_TOOLS), spec_of("mm")) is None, "the server alone is the server mode"

    def returned(content, id: str) -> dict:
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "content": content}]}}

    watch = claude.watcher(both)
    assert watch(both_init) is None
    # a server's call and its output may hold the system's own words, as its results hold paths under its own home
    assert watch(use("mcp__mm__find", {"since": "2026-11-01", "note": ".multimem"}, "s1")) is None
    assert watch(returned('{"home": "C:/Users/dan/.multimem/index.db"}', "s1")) is None
    assert watch(returned([{"type": "text", "text": f"{lower(tmp_path / 'answers')}/x"}], "s1")) is None
    # the own tools are held to them, in the call and in the output, and a result of an id nobody knows is one of them
    assert watch(use("Read", {"file_path": str(tmp_path / "Answers" / "x.json")}, "r1")) == f"escaped: Read reached {deny[0]}"
    watch = claude.watcher(both)
    assert watch(both_init) is None and watch(use("Grep", {"pattern": "x", "path": str(tmp_path / "data")}, "r1")) is None
    assert watch(returned("/home/dan/.MultiMem/index.db", "r1")) == "escaped: Grep output reached .multimem"
    assert watch(returned("/home/dan/.MultiMem/index.db", "who-knows")) == "escaped: a tool output reached .multimem"
    # a tool of another server is not the mode's
    assert claude.watcher(both)(both_init) is None
    assert "mcp__other__ask" in claude.check_event(use("mcp__other__ask", {}), both)
    assert claude.check_event(use("mcp__mm__ask", {}), spec_of("files")), "a server's tool is not an own-tools tool"

    def item(kind: str, event: str = "item.started", **fields) -> dict:
        return {"type": event, "item": {"type": kind, **fields}}

    for kind in ("agent_message", "reasoning", "error", "todo_list", "command_execution"):
        assert codex.check_event(item(kind), both) is None, kind
    assert codex.check_event(item("mcp_tool_call", server="mm", tool="find"), both) is None
    for kind in ("file_change", "web_search", "collab_tool_call"):
        assert "mm's tools or the files arm's" in codex.check_event(item(kind), both), kind
    assert "server 'other'" in codex.check_event(item("mcp_tool_call", server="other", tool="t"), both)
    hit = codex.check_event(item("command_execution", command=f"type {tmp_path}\\Answers\\truth.json"), both)
    assert hit == f"shell reached {deny[0]}"
    assert codex.check_event(item("command_execution", "item.completed", command="dir", aggregated_output="C:/.multimem/x"),
                             both) == "shell output reached .multimem"
    # the server's own call, and what it returned, say whatever they say
    served = item("mcp_tool_call", "item.completed", server="mm", tool="find", arguments={"note": ".multimem"},
                  result={"content": [{"type": "text", "text": "C:/Users/dan/.multimem/index.db"}]})
    assert codex.check_event(served, both) is None
    assert "held for approval" in codex.check_event(item("mcp_tool_call", "item.completed", server="mm", tool="find",
                                                          error={"message": "requires approval"}), both)


# -- Claude: a stand-in process, for the reading of a stream, the escape, the timeout and the environment ----------

@pytest.fixture
def stand_in(tmp_path):
    """A script that prints the events of a file, writes its environment and cwd to a file, then sleeps, and the command
    that runs it in place of `claude` or `codex`: no real agent starts."""
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN, encoding="utf-8")
    probe = tmp_path / "probe.json"

    def command(events: list[dict], sleep: float = 0) -> list[str]:
        file = tmp_path / "events.jsonl"
        file.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return [sys.executable, str(script), str(file), str(sleep), str(probe)]

    command.probe = lambda: json.loads(probe.read_text(encoding="utf-8"))
    return command


def test_claudes_stream_is_read_from_a_process_and_a_setup_problem_an_escape_or_a_timeout_ends_it(
        tmp_path, monkeypatch, stand_in, fake_git):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    run, work, bin_dir = (tmp_path / n for n in ("run", "w0", "bin"))
    find = use("mcp__mm__find", {"scope": "all"})
    got = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}}

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([init(), find, got, says("ANSWER: 47"), finished()]))
    out = claude.ask_claude(spec_of("mm", tmp_path, prompt="What was the sign?"))
    assert out["problem"] is None and answer_of(out["final"]) == "47" and out["turns"] == 2 and out["cost_usd"] == 0.0421
    assert [t["tool"] for t in out["tools"]] == ["find"] and out["tokens"]["cache_write"] == 800
    assert out["model"] == "claude-sonnet-x" and out["used_percent"] is None and out["argv"][0] == sys.executable
    assert out["effort"] == out["argv"][out["argv"].index("--effort") + 1] == claude.effort(), "the result says the effort it ran at"
    assert (run / "prompt.txt").read_text() == "What was the sign?" and (run / "init.json").is_file()
    assert json.loads((run / "tools.json").read_text())[0]["tool"] == "find"
    assert (run / "final.txt").read_text().endswith("ANSWER: 47")
    assert len((run / "events.jsonl").read_text().splitlines()) == 5
    probe = stand_in.probe()
    assert Path(probe["cwd"]).samefile(work) and "ANTHROPIC_API_KEY" not in probe["env"]
    assert probe["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and probe["env"]["PATH"].startswith(str(bin_dir))

    deny = (lower(tmp_path / "answers"),)
    bad = use("Read", {"file_path": str(tmp_path / "answers" / "truth.json")})
    in_files = init(tools=claude.FILES_TOOLS, mcp_servers=[])
    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([in_files, bad, finished()], 60))
    out = claude.ask_claude(spec_of("files", tmp_path, prompt="q", run_dir=tmp_path / "run2", deny=deny))
    assert out["problem"] == f"escaped: Read reached {deny[0]}" and out["seconds"] < 30 and answer_of(out["final"]) is None

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([in_files, says("ANSWER: 47"), finished()]))
    out = claude.ask_claude(spec_of("files", tmp_path, prompt="q", run_dir=tmp_path / "run2b", deny=deny))
    probe = stand_in.probe()
    assert out["problem"] is None and probe["env"]["MULTIMEM_HOME"] == str(work / "no-mm-home")
    assert probe["env"]["PATH"].split(os.pathsep)[0] == str(bin_dir) and "ANTHROPIC_API_KEY" not in probe["env"]
    git_folders = tuple(str(fake_git / sub) for sub in ("cmd", "bin", "usr/bin"))
    assert probe["env"]["PATH"] == common.shell_path(bin_dir, git_folders)

    grep = use("Grep", {"pattern": "ERR-731", "path": str(tmp_path / "data")})
    leaked = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                                       "content": f"{tmp_path}/answers/truth.json:1: ERR-731"}]}}
    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([in_files, grep, leaked, finished()], 60))
    out = claude.ask_claude(spec_of("files", tmp_path, prompt="q", run_dir=tmp_path / "run2c", deny=deny))
    assert out["problem"] == f"escaped: Grep output reached {deny[0]}" and out["seconds"] < 30 and answer_of(out["final"]) is None

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([init(tools=[*MM_TOOLS, "Bash"])], 60))
    out = claude.ask_claude(spec_of("mm", tmp_path, prompt="q", run_dir=tmp_path / "run3"))
    assert out["problem"].startswith("setup: ") and "Bash" in out["problem"] and out["seconds"] < 30

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([init()], 60))
    out = claude.ask_claude(spec_of("mm", tmp_path, prompt="q", run_dir=tmp_path / "run4", timeout_s=2))
    assert out["problem"] == "timeout: no answer in 2 s" and answer_of(out["final"]) is None

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([says("ANSWER: 47")]))
    out = claude.ask_claude(spec_of("mm", tmp_path, prompt="q", run_dir=tmp_path / "run5"))
    assert out["problem"].startswith("setup: ") and "init" in out["problem"], "no init event, no answer"

    # a server and own tools together: the shell is the own tools' (no mm, a home that leads nowhere), the server is connected
    both = init(tools=[*claude.FILES_TOOLS, *MM_TOOLS])
    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([both, find, got, says("ANSWER: 47"), finished()]))
    out = claude.ask_claude(spec_of("mm+files", tmp_path, prompt="q", run_dir=tmp_path / "run6", deny=deny))
    probe = stand_in.probe()
    assert out["problem"] is None and answer_of(out["final"]) == "47" and [t["tool"] for t in out["tools"]] == ["find"]
    assert probe["env"]["MULTIMEM_HOME"] == str(work / "no-mm-home") and probe["env"]["PATH"] == common.shell_path(bin_dir, git_folders)
    assert (tmp_path / "run6" / "mcp.json").is_file() and "--mcp-config" in out["argv"] and "--add-dir" in out["argv"]


# -- Codex: the files arm -----------------------------------------------------------------------------------------------

def test_codexs_files_arm_has_its_shell_and_no_mm_and_the_mm_arm_is_pinned_to_its_project(tmp_path):
    run, work, home = tmp_path / "run", tmp_path / "w0", tmp_path / "home"
    argv = codex.argv_of(["codex"], "m", "high", spec_of("files", tmp_path))
    settings, disabled = settings_of(argv), disabled_of(argv)
    assert not any(s.startswith("mcp_servers.") for s in settings) and "--tools" not in argv
    assert not {"shell_tool", "unified_exec", "view_image"} & set(disabled)
    assert {"browser_use", "computer_use", "apps", "plugins", "multi_agent"} <= set(disabled), "the rest stay off"
    assert set(codex.DISABLED) - set(disabled) == {"shell_tool", "unified_exec", "view_image"}
    for setting in ("sandbox_mode=workspace-write", "include_environment_context=true",
                    "include_permissions_instructions=true", "sandbox_workspace_write.network_access=false",
                    "shell_environment_policy.inherit=core", "shell_environment_policy.exclude=['PATH']",
                    "windows.sandbox=unelevated", "approval_policy=never", "web_search=disabled"):
        assert setting in settings
    assert "shell_environment_policy.inherit=all" not in settings
    assert "sandbox_mode=read-only" not in settings and "include_environment_context=false" not in settings
    assert argv[argv.index("-C") + 1] == str(work) and argv[-2:] == ["--", "-"]

    # the shell sees ffmpeg's folder and the system's, and a MULTIMEM_HOME that does not exist: no mm, no .venv
    bin_dir = tmp_path / "bin"
    argv = codex.argv_of(["codex.CMD"], "m", "high", spec_of("files", tmp_path))
    codex.check_argv(argv, ["codex.CMD"])  # a .cmd shim reads none of it as syntax
    policy = dict(s.split("=", 1) for s in settings_of(argv) if s.startswith("shell_environment_policy.set."))
    assert list(policy) == ["shell_environment_policy.set.PATH", "shell_environment_policy.set.MULTIMEM_HOME"]
    shell_path = policy["shell_environment_policy.set.PATH"]
    assert shell_path.startswith("'") and shell_path.endswith("'")
    assert os.name != "nt" or ("\\" in shell_path and "/" not in shell_path), "a Windows PATH keeps its backslashes"
    assert shell_path.strip("'").split(os.pathsep) == [str(bin_dir), *common.system_path()]
    assert not any(".venv" in folder for folder in shell_path.split(os.pathsep))
    assert policy["shell_environment_policy.set.MULTIMEM_HOME"] == f"'{str(work / 'no-mm-home').replace(chr(92), '/')}'"
    assert not (work / "no-mm-home").exists()
    assert common.shell_path(None) == os.pathsep.join(common.system_path()), "no bin_dir: the system's folders alone"

    argv = codex.argv_of(["codex"], "m", "high", spec_of("mm", tmp_path))
    assert not any(s.startswith("shell_environment_policy.") for s in settings_of(argv)), "the mm arm has no shell"
    pin = next(s for s in settings_of(argv) if s.startswith("mcp_servers.mm.env."))
    assert pin == f"mcp_servers.mm.env.CLAUDE_PROJECT_DIR='{str(work).replace(chr(92), '/')}'"
    assert "'--tools','ask,find,look,label'" in next(a for a in settings_of(argv) if a.startswith("mcp_servers.mm.args="))
    assert "sandbox_mode=read-only" in settings_of(argv) and set(disabled_of(argv)) == set(codex.DISABLED)
    assert argv[argv.index("-C") + 1] == str(work)
    unpinned = served_of(McpServer("mm", sys.executable, ("-m", "multimem.cli", "--home", str(home)), {}, ("ask",)))
    argv = codex.argv_of(["codex"], "m", "high", spec_of("mm", tmp_path, served=unpinned))
    assert not any(".env." in s for s in settings_of(argv)), "a server given no environment is pinned to nothing"
    with pytest.raises(codex.CodexMissing, match="tools"):
        codex.argv_of(["codex"], "m", "high", spec_of("mm", tmp_path, served=served_of(server_of(home, work, tools=()))))

    models = {"models": [{"slug": "m", "supported_reasoning_levels": [{"effort": "high"}]}]}
    argv = codex.prepare(spec_of("files", tmp_path), "m", "high", models, ["codex"])
    assert (run / "instructions.md").read_text() == codex.BASE_FILES and work.is_dir()
    assert argv[argv.index("-C") + 1] == str(work)
    assert json.loads((run / "catalog.json").read_text())["models"][0]["tool_mode"] == "direct"


def test_codexs_instructions_are_the_modes_line_and_what_is_served_in_order(tmp_path):
    def written(spec: SessionSpec) -> str:
        models = {"models": [{"slug": "m", "supported_reasoning_levels": [{"effort": "high"}]}]}
        codex.prepare(spec, "m", "high", models, ["codex"])
        return (spec.run_dir / "instructions.md").read_text(encoding="utf-8")

    server = server_of(tmp_path / "home", tmp_path / "w0")
    assert written(spec_of("mm", tmp_path / "a")) == f"{BASE_LINE}\n\nTHE SERVER'S OWN WORDS\n\nTHE SKILL\n\nTHE NOTES"
    only_skill = served_of(server, parts=("THE SKILL",))
    assert written(spec_of("mm", tmp_path / "b", served=only_skill)) == f"{BASE_LINE}\n\nTHE SERVER'S OWN WORDS\n\nTHE SKILL"
    bare = served_of(server, parts=(), server_instructions=None)
    assert written(spec_of("mm", tmp_path / "c", served=bare)) == BASE_LINE, "a system with neither: its line alone"
    # a mode without a server never holds what a system serves
    held = Served(server=None, base_line=BASE_LINE, parts=("THE SKILL",), server_instructions="X", holder="mm",
                  clock_note=None, record={})
    assert written(spec_of("files", tmp_path / "d", served=held)) == codex.BASE_FILES
    assert written(spec_of("blind", tmp_path / "e", served=held)) == common.BASE_BLIND
    assert written(spec_of("mm+files", tmp_path / "f")) == \
        f"{codex.BASE_FILES}\n\n{BASE_LINE}\n\nTHE SERVER'S OWN WORDS\n\nTHE SKILL\n\nTHE NOTES"


def test_codexs_problems_are_told_apart_by_what_the_benchmark_does_with_them():
    def turn_failed(message: str) -> dict:
        return {"type": "turn.failed", "error": {"message": message}}

    def error(message: str) -> dict:
        return {"type": "error", "message": message}

    def problem(events: list[dict], stderr: str = "", code: int = 1) -> str | None:
        return codex.problem_of(code, events, stderr)

    full = CONTEXT_FULL
    assert problem([turn_failed(full)]) == f"capped: {full}"
    assert problem([error(full), turn_failed("turn aborted")]).startswith("capped: Codex ran out of room")
    required = "required MCP servers failed to initialize: mm"
    assert problem([error(required), turn_failed(required)]) == f"setup: {required}"
    assert problem([error(required), turn_failed("turn aborted")]) == f"setup: {required}", "an earlier message counts"
    assert problem([turn_failed("required MCP server `mm` was not initialized")]).startswith("setup: ")
    assert problem([error("MCP client for `mm` failed to start: no such file")]).startswith("setup: ")
    assert problem([error("MCP client for `mm` timed out after 60 seconds.")]).startswith("setup: ")
    assert problem([], "Error loading configuration: unknown field `mcp_servers.mm.foo`").startswith("setup: Error loading")
    assert problem([turn_failed("You've hit your usage limit. Try again later.")]).startswith("limit: ")
    assert problem([turn_failed("401 Unauthorized: not logged in")]).startswith("auth: ")
    assert problem([turn_failed("stream disconnected before completion")]) == "error: stream disconnected before completion"
    assert problem([]) == "error: codex exec exited 1 without finishing the turn"
    assert problem([{"type": "turn.completed"}], code=0) is None
    assert problem([{"type": "turn.completed"}, turn_failed(full)], code=0).startswith("capped: ")


def test_codexs_files_arm_may_run_commands_and_nothing_else_and_none_may_reach_the_answers(tmp_path):
    deny = (lower(tmp_path / "answers"), ".codex")

    def item(kind: str, event: str = "item.started", **fields) -> dict:
        return {"type": event, "item": {"type": kind, **fields}}

    ffmpeg = item("command_execution", command=f'powershell -Command "ffmpeg -i {tmp_path}\\data\\phone\\a.jpg o.png"')
    assert codex.check_event(ffmpeg, spec_of("files", deny=deny)) is None
    for kind in ("agent_message", "reasoning", "error", "todo_list"):
        assert codex.check_event(item(kind), spec_of("files", deny=deny)) is None
    assert codex.check_event({"type": "turn.completed"}, spec_of("files", deny=deny)) is None
    hit = codex.check_event(item("command_execution", command=f"type {tmp_path}\\Answers\\truth.json"), spec_of("files", deny=deny))
    assert hit == f"shell reached {deny[0]}"
    assert "shell reached .codex" in codex.check_event(
        item("command_execution", "item.completed", command="cat ~/.CODEX/auth.json"), spec_of("files", deny=deny))
    assert codex.check_event(item("command_execution", command="cat ~/.codex/auth.json"), spec_of("files")) is None
    for kind in ("mcp_tool_call", "file_change", "web_search", "collab_tool_call"):
        assert "files arm" in codex.check_event(item(kind, server="mm", tool="find"), spec_of("files", deny=deny))

    # the mm arm is as it was: a command or a file change ends it, and so does a call of another server
    assert codex.check_event(ffmpeg, spec_of("mm"))
    assert codex.check_event(item("file_change"), spec_of("mm", deny=deny))
    assert codex.check_event(item("mcp_tool_call", server="mm", tool="find"), spec_of("mm")) is None

    done = [{"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}},
            {"type": "turn.started"}, {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 4}}]
    assert codex.usage_of(done) == (2, {"input": 150, "output": 13, "cache_read": 60, "cache_write": 0})
    assert codex.usage_of([{"type": "turn.completed"}]) == (1, None) and codex.usage_of([]) == (0, None)
    shell = item("command_execution", "item.completed", command="ffmpeg -i a.mp4", aggregated_output="frame=1", exit_code=0,
                 status="completed")
    call = item("mcp_tool_call", "item.completed", server="mm", tool="ask", arguments={"scope": "all"}, status="completed")
    tools = codex.tools_of([item("command_execution"), shell, call])
    assert [t["tool"] for t in tools] == ["shell", "ask"] and tools[0]["arguments"] == {"command": "ffmpeg -i a.mp4"}
    assert tools[0]["result"] == "frame=1" and tools[0]["error"] is None and tools[1]["arguments"] == {"scope": "all"}
    failed = {**shell["item"], "exit_code": 1}
    assert codex.tools_of([{"type": "item.completed", "item": failed}])[0]["error"] == "exit code 1"

    env = common.path_with(tmp_path / "bin")
    assert env["PATH"].startswith(str(tmp_path / "bin") + os.pathsep) and common.path_with(None) == dict(os.environ)


def test_codexs_files_exec_is_read_from_a_process_and_a_command_at_the_answers_ends_it(tmp_path, monkeypatch, stand_in):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    work, bin_dir = tmp_path / "w0", tmp_path / "bin"
    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    monkeypatch.setattr(codex, "models_of", lambda command: models)
    deny = (lower(tmp_path / "answers"),)

    def ask(name: str, mode: str = "files", **fields) -> dict:
        fields = {"prompt": "q", "run_dir": tmp_path / name, "model": "gpt-test", "effort": "high", **fields}
        return codex.ask_codex(spec_of(mode, tmp_path, **fields))

    def command(name: str, kind: str = "command_execution", **fields) -> dict:
        return {"type": name, "item": {"type": kind, "command": "ffmpeg -i x.mp4 f.png", **fields}}

    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: 47"}}
    usage = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}}
    events = [{"type": "turn.started"}, command("item.started"), command("item.completed", exit_code=0), said, usage]
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in(events))
    out = ask("run", prompt="What was the sign?", deny=deny)
    assert out["problem"] is None and answer_of(out["final"]) == "47" and out["model"] == "gpt-test" and out["cost_usd"] is None
    assert out["turns"] == 1 and out["tokens"] == {"input": 100, "output": 9, "cache_read": 60, "cache_write": 0}
    assert [t["tool"] for t in out["tools"]] == ["shell"] and out["used_percent"] is None
    assert Path(stand_in.probe()["cwd"]).samefile(work) and stand_in.probe()["env"]["PATH"].startswith(str(bin_dir))
    assert (tmp_path / "run" / "instructions.md").read_text() == codex.BASE_FILES

    bad = command("item.started", command=f"type {tmp_path}\\answers\\truth.json")
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "turn.started"}, bad, said, usage], 60))
    out = ask("run2", deny=deny)
    assert out["problem"] == f"escaped: shell reached {deny[0]}" and out["seconds"] < 30 and answer_of(out["final"]) is None

    monkeypatch.setattr(codex, "codex_command",
                        lambda: stand_in([{"type": "turn.started"}, command("item.started", "mcp_tool_call")], 60))
    out = ask("run3")
    assert out["problem"].startswith("escaped: ") and "mcp_tool_call" in out["problem"]

    leak = command("item.completed", exit_code=0, aggregated_output=f"{tmp_path}/answers/truth.json:1: ERR-731")
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "turn.started"}, leak, said, usage], 60))
    out = ask("run4", deny=deny)
    assert out["problem"] == f"escaped: shell output reached {deny[0]}" and out["seconds"] < 30 and answer_of(out["final"]) is None

    full = CONTEXT_FULL
    monkeypatch.setattr(codex, "codex_command",
                        lambda: stand_in([{"type": "turn.started"}, {"type": "turn.failed", "error": {"message": full}}]))
    out = ask("run5")
    assert out["problem"] == f"capped: {full}" and out["turns"] == 0 and answer_of(out["final"]) is None

    required = "required MCP servers failed to initialize: mm"
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "error", "message": required}]))
    out = ask("run6")
    assert out["problem"] == f"setup: {required}"

    # a server and own tools together: the server's call is the server's, the shell is checked
    call = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mm", "tool": "find", "arguments": {},
                                               "result": {"content": []}, "status": "completed"}}
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "turn.started"}, call, command("item.completed", exit_code=0),
                                                                    said, usage]))
    out = ask("run7", "mm+files", deny=deny)
    assert out["problem"] is None and [t["tool"] for t in out["tools"]] == ["find", "shell"]
    settings = settings_of(out["argv"])
    assert any(s.startswith("mcp_servers.mm.command=") for s in settings) and "sandbox_mode=workspace-write" in settings


def test_codex_view_image_calls_are_read_from_the_execs_own_rollout(tmp_path, monkeypatch):
    """`codex exec --json` has no item for view_image, so the files arm counts them from the rollout file."""
    picture = r"C:\data\phone\IMG_1.jpg"
    lines = [json.dumps({"type": "session_meta", "payload": {"id": "t1"}}),
             json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                                                               "arguments": json.dumps({"cmd": "dir"})}}),
             json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "view_image",
                                                               "arguments": json.dumps({"path": picture})}}),
             "not json"]
    assert codex.images_in(lines) == [{"tool": "view_image", "arguments": {"path": picture}, "result": None,
                                       "error": None, "status": "completed"}]
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    day = tmp_path / "sessions" / "2026" / "11" / "02"
    day.mkdir(parents=True)
    (day / "rollout-2026-11-02T13-01-33-t1.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert [c["arguments"]["path"] for c in codex.rollout_images("t1")] == [picture]
    assert codex.rollout_images("t2") == [] and codex.rollout_images("") == []
    assert common.reached({"path": picture}, ("c:/data",)) == "c:/data", "a picture's path is checked like a command"


def test_claude_login_is_checked_with_auth_status_before_a_run(tmp_path):
    def fake(stdout: str) -> list[str]:
        script = tmp_path / f"claude{len(list(tmp_path.iterdir()))}.py"
        script.write_text(f"import sys\nassert sys.argv[1:] == ['auth', 'status']\nprint({stdout!r})\n", encoding="utf-8")
        return [sys.executable, str(script)]

    claude.check_login(fake('{"loggedIn": true, "authMethod": "claude.ai"}'))
    with pytest.raises(claude.ClaudeMissing, match="claude auth login"):
        claude.check_login(fake('{"loggedIn": false, "authMethod": "none"}'))
    with pytest.raises(claude.ClaudeMissing, match="gave no status"):
        claude.check_login(fake("Usage: claude [options]"))


# -- the blind arm: no tools, no server, no recordings ------------------------------------------------------------------

BLIND = "You answer the user's question from what you know; you have no tools."


def test_the_blind_arm_has_one_line_of_instructions_and_no_tools_for_both_agents(tmp_path):
    assert common.BASE_BLIND == BLIND
    run, work = tmp_path / "run", tmp_path / "w0"

    # Claude: `--tools ""`, the strict MCP flag with no config, and nothing allowed
    argv = claude.prepare(spec_of("blind", tmp_path), command=["claude"])
    assert after(argv, "--tools") == [""] and "--strict-mcp-config" in argv and "--restricted" in argv
    for flag in ("--mcp-config", "--allowedTools", "--add-dir"):
        assert flag not in argv, flag
    assert after(argv, "--permission-mode") == ["dontAsk"] and after(argv, "--settings") == [str(run / "settings.json")]
    assert (run / "instructions.md").read_text() == BLIND and not (run / "mcp.json").exists() and work.is_dir()
    held = Served(server=None, base_line=BASE_LINE, parts=("THE SKILL", "THE NOTES"), server_instructions="X", holder="mm",
                  clock_note=None, record={})
    claude.prepare(spec_of("blind", tmp_path, served=held), command=["claude"])
    assert (run / "instructions.md").read_text() == BLIND, "no skill, no notes"
    given = claude.argv_of(["claude"], spec_of("blind", tmp_path, data=tmp_path / "data"), name="sonnet", level="high")
    assert "--add-dir" not in given and after(given, "--tools") == [""], "no data folder either"
    assert claude.env_of(tmp_path / "bin")["PATH"].startswith(str(tmp_path / "bin"))

    # Codex: the mm arm's read-only settings with every feature off and no server
    argv = codex.argv_of(["codex"], "m", "high", spec_of("blind", tmp_path))
    settings, disabled = settings_of(argv), disabled_of(argv)
    assert not any(s.startswith(("mcp_servers.", "shell_environment_policy.")) for s in settings) and "--tools" not in argv
    assert set(disabled) == set(codex.DISABLED) and {"shell_tool", "unified_exec", "view_image"} <= set(disabled)
    for setting in codex.CONFIG:
        assert setting in settings, setting
    assert "sandbox_workspace_write.network_access=false" not in settings and "windows.sandbox=unelevated" not in settings
    assert argv[argv.index("-C") + 1] == str(work) and argv[-2:] == ["--", "-"]
    models = {"models": [{"slug": "m", "supported_reasoning_levels": [{"effort": "high"}]}]}
    codex.prepare(spec_of("blind", tmp_path, run_dir=tmp_path / "codex-run", served=held), "m", "high", models, ["codex"])
    assert (tmp_path / "codex-run" / "instructions.md").read_text() == BLIND
    assert after(codex.argv_of(["codex"], "m", "high", spec_of("blind", tmp_path, served=held)), "-C")[0] == str(work)


def test_a_session_with_a_tool_or_a_server_is_not_the_blind_arms_and_any_tool_use_is_an_escape(tmp_path):
    assert claude.check_init(init(tools=(), servers=()), spec_of("blind")) is None
    assert "Bash" in claude.check_init(init(tools=["Bash"], servers=()), spec_of("blind"))
    assert "mcp__mm__ask" in claude.check_init(init(tools=MM_TOOLS, servers=()), spec_of("blind"))
    assert "MCP servers" in claude.check_init(init(tools=(), servers=[{"name": "mm", "status": "connected"}]), spec_of("blind"))
    assert claude.check_init(init(tools=(), servers=(), plugins=[{"name": "x"}]), spec_of("blind"))

    for name in ("Read", "Bash", "Glob", "mcp__mm__find", "WebSearch"):
        hit = claude.check_event(use(name, {"command": "ffmpeg -i a.mp4", "file_path": "x"}), spec_of("blind"))
        assert hit and repr(name) in hit and "blind arm" in hit, name
    assert claude.check_event({"type": "assistant", "message": {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {}}]}}, spec_of("blind"))
    assert claude.check_event(says("ANSWER: B"), spec_of("blind")) is None
    watch = claude.watcher(spec_of("blind"))
    assert watch(init(tools=(), servers=())) is None and watch(says("Let me think.")) is None
    assert watch(use("Read", {"file_path": "x"})).startswith("escaped: claude used 'Read'")

    # Codex: a message, a thought, an error and the plan list are the only items; a shell, a server call or a search is not
    def item(kind: str, event: str = "item.started", **fields) -> dict:
        return {"type": event, "item": {"type": kind, **fields}}

    for kind in ("agent_message", "reasoning", "error", "todo_list"):
        assert codex.check_event(item(kind), spec_of("blind")) is None, kind
    assert codex.check_event({"type": "turn.completed"}, spec_of("blind")) is None
    for kind in ("command_execution", "mcp_tool_call", "file_change", "web_search", "collab_tool_call"):
        for event in ("item.started", "item.completed"):
            hit = codex.check_event(item(kind, event, server="mm", tool="ask"), spec_of("blind"))
            assert hit and repr(kind) in hit and "blind arm" in hit, (kind, event)


def test_a_blind_exec_is_read_from_a_process_and_a_tool_event_ends_it_as_escaped(tmp_path, monkeypatch, stand_in):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    blind_init = init(tools=(), servers=())
    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([blind_init, says("B."), finished(result="B.\nANSWER: B")]))
    out = claude.ask_claude(spec_of("blind", tmp_path, prompt="Which?", run_dir=tmp_path / "claude-ok"))
    assert out["problem"] is None and answer_of(out["final"]) == "B" and out["tools"] == []
    assert (tmp_path / "claude-ok" / "prompt.txt").read_text() == "Which?" and "--tools" in out["argv"]
    assert "--mcp-config" not in out["argv"] and "--allowedTools" not in out["argv"]

    monkeypatch.setattr(claude, "claude_command",
                        lambda: stand_in([blind_init, use("Bash", {"command": "ffmpeg -i a.mp4"}), says("ANSWER: B")], 60))
    out = claude.ask_claude(spec_of("blind", tmp_path, prompt="Which?", run_dir=tmp_path / "claude-tool"))
    assert out["problem"].startswith("escaped: claude used 'Bash'") and out["seconds"] < 30 and answer_of(out["final"]) is None

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([init(tools=["Bash"], servers=())], 60))
    out = claude.ask_claude(spec_of("blind", tmp_path, prompt="Which?", run_dir=tmp_path / "claude-setup"))
    assert out["problem"].startswith("setup: ") and "Bash" in out["problem"] and out["seconds"] < 30

    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    monkeypatch.setattr(codex, "models_of", lambda command: models)
    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: C"}}
    usage = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}}
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "turn.started"}, said, usage]))
    out = codex.ask_codex(spec_of("blind", tmp_path, prompt="Which?", run_dir=tmp_path / "codex-ok", model="gpt-test", effort="high"))
    assert out["problem"] is None and answer_of(out["final"]) == "C" and out["tools"] == []
    assert (tmp_path / "codex-ok" / "instructions.md").read_text() == BLIND

    shell = {"type": "item.started", "item": {"type": "command_execution", "command": "ffmpeg -i a.mp4 f.png"}}
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in([{"type": "turn.started"}, shell, said, usage], 60))
    out = codex.ask_codex(spec_of("blind", tmp_path, prompt="Which?", run_dir=tmp_path / "codex-tool", model="gpt-test", effort="high"))
    assert out["problem"].startswith("escaped: ") and "command_execution" in out["problem"] and "blind arm" in out["problem"]
    assert out["seconds"] < 30 and answer_of(out["final"]) is None


# -- new with the move: an agent is a function of a SessionSpec that knows no system ------------------------------------------

def test_an_agent_is_a_function_of_a_session_spec_and_the_model_and_effort_default_to_the_agents_own(monkeypatch):
    for ask in (claude.ask_claude, codex.ask_codex):
        assert list(inspect.signature(ask).parameters) == ["spec"]
    for var in ("MMSB_CLAUDE_MODEL", "MMSB_CLAUDE_EFFORT", "MMSB_CODEX_MODEL", "MMSB_CODEX_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    assert (claude.model(), claude.effort(), codex.model(), codex.effort()) == ("sonnet", "high", "gpt-6.1-terra", "high")
    monkeypatch.setenv("MMSB_CLAUDE_MODEL", "opus")
    monkeypatch.setenv("MMSB_CODEX_EFFORT", "medium")
    assert claude.model() == "opus" and codex.effort() == "medium"
    monkeypatch.setenv("MM_CLAUDE_MODEL", "old-name")
    assert claude.model() == "opus", "the old names mean nothing"
    argv = claude.argv_of(["claude"], spec_of("blind"), name=claude.model(), level=claude.effort())
    assert after(argv, "--model") == ["opus"]


def test_the_arm_names_in_the_messages_are_blind_files_the_servers_name_and_both():
    assert [spec_of(m).arm for m in ("blind", "files", "mm", "mm+files")] == ["blind", "files", "mm", "mm+files"]
    other = served_of(server_of(Path("home"), Path("w0"), name="listing", tools=("list_files", "file_info")))
    assert spec_of("mm", served=other).arm == "listing" and spec_of("mm+files", served=other).arm == "listing+files"
    assert claude.session_tools(spec_of("mm", served=other)) == ("mcp__listing__list_files", "mcp__listing__file_info")
    assert "listing's tools" in codex.check_event({"type": "item.started", "item": {"type": "file_change"}}, spec_of("mm", served=other))


NEUTRAL = re.compile(r"\bmm\b|multimem", re.I)
AGENTS = Path(__file__).parents[1] / "mmsb" / "core" / "agents"


def test_no_agent_module_names_mm_or_imports_multimem_or_an_old_root_module():
    """Tools and instructions come from `Served`, so the agents name no system. (The old root modules stay until U8; nothing
    new may import them.)"""
    for file in sorted(AGENTS.glob("*.py")):
        found = [m.group(0) for line in file.read_text(encoding="utf-8").splitlines() if (m := NEUTRAL.search(line))]
        assert not found, f"{file.name} names {found}"
        tree = ast.parse(file.read_text(encoding="utf-8"))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not imported & {"multimem", "claude_run", "codex_run", "bench", "egolife", "haystack", "shortcuts"}, file.name


def test_the_agents_import_without_multimem_or_the_old_root_modules_in_a_process_that_refuses_them():
    """A `sys.meta_path` blocker, because the editable install would make `import claude_run` work: a slip would go unnoticed."""
    code = textwrap.dedent("""
        import importlib.abc, sys

        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"multimem", "claude_run", "codex_run", "bench", "egolife", "haystack", "haystack_media",
                                          "shortcuts", "grade", "conftest", "day", "fake_decider"}:
                    raise ImportError(f"refused: {name}")

        sys.meta_path.insert(0, Refuse())
        import mmsb.core.agents.spec, mmsb.core.agents.common, mmsb.core.agents.claude, mmsb.core.agents.codex
        assert "multimem" not in sys.modules and "claude_run" not in sys.modules
        print("imported")
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert done.returncode == 0 and done.stdout.strip() == "imported", done.stderr[-800:]


# -- a folder the system handed the agent (G1's F4) ----------------------------------------------------------------------------

def opens_world(tmp_path: Path) -> tuple[Path, Path, Path, tuple[str, ...]]:
    """A run folder (denied), the home of a system inside it and the `looks` folder of the home, which the system hands the agent."""
    run = tmp_path / "run"
    home = run / "homes" / "claude-d1-3B"
    return run, home, home / "looks", (lower(run), ".multimem")


def test_the_own_tools_may_open_a_folder_the_system_handed_them_and_nothing_else_of_the_run_folder_for_claude(tmp_path):
    run, home, looks, deny = opens_world(tmp_path)
    picture = str(looks / "c3f662536fc8.png")
    both = spec_of("mm+files", tmp_path, deny=deny, served=served_of(server_of(home, tmp_path / "w0"), opens=(looks,)))
    bare = spec_of("mm+files", tmp_path, deny=deny)  # a system that handed nothing: the same reads are an escape
    escaped = f"Read reached {deny[0]}"

    def read(path: str) -> dict:
        return use("Read", {"file_path": path})

    assert claude.check_event(read(picture), both) is None
    assert claude.check_event(read(picture.upper().replace("\\", "/")), both) is None, "in either case and style of slash"
    assert claude.check_event(use("Bash", {"command": f'ffmpeg -i "{picture}" -vf scale=640:-1 out.png'}), both) is None
    assert claude.check_event(use("Glob", {"pattern": f"{looks}\\*.png"}), both) is None
    assert claude.check_event(use("Grep", {"pattern": "x", "path": str(looks)}), both) is None, "the folder itself"
    assert claude.check_event(read(picture), bare) == escaped, "the reach was an escape before the system handed the folder"
    for path in (home / "index.sqlite", f"{looks}\\..\\index.sqlite", f"{looks}/../index.sqlite", home / "files" / "a.mkv", home / "looks-old" / "a.png",
                 home, run / "results" / "claude" / "files" / "x.json", f"{picture}\\..\\..\\index.sqlite"):
        assert claude.check_event(read(str(path)), both) == escaped, path
    assert claude.check_event(use("Bash", {"command": f'ffmpeg -i "{picture}" -y {home}\\index.sqlite'}), both) == "Bash reached " + deny[0]

    # what the own tools print: the folder's files may be listed, the rest of the home may not
    def returned(content, id: str = "g1") -> dict:
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "content": content}]}}

    names = {"g1": "Glob"}
    assert claude.check_result(returned(f"{looks}\\a.png\n{looks}\\b.png"), both, names) is None
    assert claude.check_result(returned(f"{home}\\index.sqlite"), both, names) == f"Glob output reached {deny[0]}"
    assert claude.check_result(returned(f"{looks}\\..\\index.sqlite"), both, names) == f"Glob output reached {deny[0]}"
    watch = claude.watcher(both)
    assert watch(init(tools=[*claude.FILES_TOOLS, *MM_TOOLS])) is None and watch(read(picture)) is None
    assert watch(read(str(home / "index.sqlite"))) == f"escaped: {escaped}"
    # and the permission is the agent's own as well: its file tools are told the folder too, after the data's
    adds = lambda spec: [claude.argv_of(["claude"], spec, name="m", level="high")[i + 1] for i, a in enumerate(claude.argv_of(["claude"], spec, name="m", level="high"))
                         if a == "--add-dir"]  # noqa: E731
    assert adds(both) == [str(tmp_path / "data"), str(looks)] and adds(bare) == [str(tmp_path / "data")]
    assert "--add-dir" not in claude.argv_of(["claude"], spec_of("mm", tmp_path, served=both.served), name="m", level="high"), "no file tools, no folder"


def test_the_own_tools_may_open_a_folder_the_system_handed_them_and_nothing_else_of_the_run_folder_for_codex(tmp_path, monkeypatch, stand_in):
    run, home, looks, deny = opens_world(tmp_path)
    picture = str(looks / "c3f662536fc8.png")
    both = spec_of("mm+files", tmp_path, deny=deny, served=served_of(server_of(home, tmp_path / "w0"), opens=(looks,)))
    bare = spec_of("mm+files", tmp_path, deny=deny)

    def shell(command: str, output: str = "") -> dict:
        return {"type": "item.completed", "item": {"type": "command_execution", "command": command, "aggregated_output": output, "exit_code": 0}}

    assert codex.check_event(shell(f'ffmpeg -i "{picture}" -vf scale=640:-1 out.png'), both) is None
    assert codex.check_event(shell(f"dir {looks}", f"{looks}\\c3f662536fc8.png"), both) is None
    assert codex.check_event(shell(f'ffmpeg -i "{picture}" out.png'), bare) == f"shell reached {deny[0]}"
    for command in (f"type {home}\\index.sqlite", f"type {looks}\\..\\index.sqlite", f"type {home}\\files\\a.mkv", f"copy {picture} {home}\\x"):
        assert codex.check_event(shell(command), both) == f"shell reached {deny[0]}", command
    assert codex.check_event(shell("dir", f"{home}\\files\\a.mkv"), both) == f"shell output reached {deny[0]}"
    assert codex.check_event(shell("dir", f"{looks}\\..\\index.sqlite"), both) == f"shell output reached {deny[0]}"

    # `view_image` has no item in the stream: the exec's own rollout names it, and is read after the exec
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(codex, "models_of", lambda command: {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]})
    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: 47"}}
    events = [{"type": "thread.started", "thread_id": "t9"}, said, {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}]
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in(events))

    def rollout(path: str) -> None:
        day = tmp_path / "codex-home" / "sessions" / "2026" / "11" / "04"
        day.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"type": "event_msg", "payload": {"type": "token_count", "rate_limits": {"primary": {"window_minutes": 10080, "used_percent": 3.0}}}}),
                 json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "view_image", "arguments": json.dumps({"path": path})}})]
        (day / "rollout-2026-11-04T00-00-00-t9.jsonl").write_text("\n".join(lines), encoding="utf-8")

    def ask(name: str) -> dict:
        return codex.ask_codex(dataclasses.replace(both, run_dir=tmp_path / name, model="gpt-test", effort="high"))

    rollout(picture)
    out = ask("viewed")
    assert out["problem"] is None and [t["tool"] for t in out["tools"]] == ["view_image"]
    rollout(str(home / "index.sqlite"))
    assert ask("peeked")["problem"] == f"escaped: view_image reached {deny[0]}"
    rollout(f"{looks}\\..\\index.sqlite")
    assert ask("climbed")["problem"] == f"escaped: view_image reached {deny[0]}"


# -- the effort asked for reaches the command line of each agent (G1's F6) --------------------------------------------------------

def test_the_effort_asked_for_is_on_the_command_line_of_both_agents_and_without_one_each_has_its_own_default(tmp_path, monkeypatch, stand_in, fake_git):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    for variable in ("MMSB_CODEX_EFFORT", "MMSB_CLAUDE_EFFORT"):
        monkeypatch.delenv(variable, raising=False)
    levels = [{"effort": e} for e in ("high", "medium", "low")]
    monkeypatch.setattr(codex, "models_of", lambda command: {"models": [{"slug": "gpt-test", "supported_reasoning_levels": levels}]})
    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: 47"}}
    done = [{"type": "turn.started"}, said, {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}]
    monkeypatch.setattr(codex, "codex_command", lambda: stand_in(done))
    for n, (asked, got) in enumerate((("medium", "medium"), ("low", "low"), (None, "high"))):
        out = codex.ask_codex(spec_of("blind", tmp_path, run_dir=tmp_path / f"codex{n}", model="gpt-test", effort=asked))
        assert out["problem"] is None and out["effort"] == got
        assert f"model_reasoning_effort={got}" in settings_of(out["argv"]) and sum(s.startswith("model_reasoning_effort=") for s in settings_of(out["argv"])) == 1
    monkeypatch.setenv("MMSB_CODEX_EFFORT", "low")
    out = codex.ask_codex(spec_of("blind", tmp_path, run_dir=tmp_path / "codex-env", model="gpt-test"))
    assert "model_reasoning_effort=low" in settings_of(out["argv"]), "the agent's own default is the environment's"

    monkeypatch.setattr(claude, "claude_command", lambda: stand_in([init(), says("ANSWER: 47"), finished()]))
    for n, (asked, got) in enumerate((("medium", "medium"), ("low", "low"), (None, "high"))):
        out = claude.ask_claude(spec_of("mm", tmp_path, run_dir=tmp_path / f"claude{n}", effort=asked))
        assert out["problem"] is None and after(out["argv"], "--effort") == [got]
    monkeypatch.setenv("MMSB_CLAUDE_EFFORT", "low")
    out = claude.ask_claude(spec_of("mm", tmp_path, run_dir=tmp_path / "claude-env"))
    assert after(out["argv"], "--effort") == ["low"], "the agent's own default is the environment's"
