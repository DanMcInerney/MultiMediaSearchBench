"""Claude Code asked a question (TESTBED.md, Benchmark; moved from claude_run.py of this repository at bb498a6): one `claude -p`
for the `SessionSpec` it is given, locked down to the tools of its mode, in the way `codex.py` locks down Codex. A dry run
(`prepare`) runs nothing. It never reads, copies or moves a file of `~/.claude`: Claude reads its own login.

`ask_claude` returns `{final, tools, events, seconds, clock, working_s, queue_s, turns, tokens, cost_usd, used_percent, model, denials,
problem, argv, stderr}`: the last message (the run reads the answer in it, core/answer.py), what Claude called and got back (a tool's bare
name, without its `mcp__<server>__` prefix, or `Read`, `Glob`, `Grep`, `Bash`), how long it took on the wall (`seconds`) and by the working
clock it was held to (`clock`, `working_s`, `queue_s`: core/clock.py; `timeout` is that clock running out), its turns, tokens and cost, the
weekly plan use a `rate_limit_event` reports (None when none does), and `problem` (None, or a kind and the reason). The kinds,
by what the benchmark does with them: `escaped` (a tool outside the mode, or a `deny` string in the input or output of one of
its own tools), `timeout` and `capped` (`error_max_turns`, `error_max_budget_usd`) count as a wrong answer; `setup` (the session
is not the one asked for: a wrong init event, or an option `claude` refused), `limit` and `auth` stop the agent; `error` is run
again. It raises `ClaudeMissing` before any run when nothing can run.

The four modes, from `spec.own_tools` and `spec.served.server`:
- blind: `--tools ""`, no MCP server and no `--allowedTools`: what the model knows, with no recordings. A tool use is `escaped`.
- a server: the one MCP server the system serves (its command, arguments and environment are the system's) and no built-in tool.
  The server's instructions reach Claude through the server; the system's base line and parts are appended to the system prompt,
  because skills are off in the run.
- own tools: Read, Glob, Grep and Bash that runs ffmpeg and the read-only commands only, over the working folder and the data
  folder (and the folders a system handed it, `Served.opens`), with no MCP server. No tool input or output may hold a `deny` string
  (a path or word that leads to the answers), but for a path inside a folder the system handed it. Its
  PATH is `bin_dir`, the Windows folders and Git for Windows' (the Bash tool runs in it), so none of the owner's programs and no
  `.venv`. The variables the systems' `Isolation` names are dropped or lead to a folder that does not exist. Beside other cells, or with
  a system that needs the GPU, it also sees no CUDA device (`spec.off_gpu`).
- a server and own tools: both. Only the own tools' inputs and outputs are held to `deny`: a server's tool is checked by name, and
  its results may hold paths of its own.

How the exec is made:
- `claude -p --verbose --output-format stream-json --restricted --strict-mcp-config --permission-mode dontAsk ...`, the
  prompt on stdin. `--restricted` loads managed settings and `--settings` only (no user hooks, plugins or permission
  rules) and keeps the file tools in the working folder and `--add-dir`; `--strict-mcp-config` drops the owner's own MCP
  registrations, so only `--mcp-config` connects; `--tools` sets the built-in tools (MCP tools are not affected);
  `dontAsk` refuses whatever `--allowedTools` does not allow; `--disable-slash-commands` keeps installed skills out;
- the environment: the owner's API key out (`-p` would use it: the benchmark uses the plan), and `ENV_SET` in (no
  CLAUDE.md or auto-memory, no claude.ai connectors, no background tasks, no updates or telemetry);
- every event is checked as it arrives: an `init` whose tools, MCP servers or plugins are not the mode's, or that says the
  session is billed to an API key, or a missing `init`, stops the exec as `setup`; a tool the mode does not have, or
  (own tools) an input or an output that holds a `deny` string, as `escaped`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from mmmb.core.agents.common import (AUTH as AUTH_BASE, blocks_of, instructions_of, kill_tree, path_with, reached,
                                     scrubbed, shell_path, wait_for)
from mmmb.core.agents.spec import SessionSpec
from mmmb.core.clock import WorkingClock
from mmmb.core.contract import Isolation, McpServer
from mmmb.core.runfolder import OFF_GPU_ENV

# The files arm's Bash runs ffmpeg alone (FILES_ALLOWED), and Claude is told so: in the pilot it was denied a `cd ... && find`,
# took Bash for gone and answered "unknown" without trying ffmpeg. Codex needs no such line: its shell is the sandbox's, whole.
# Claude Code (2.1.x) parses no Bash command longer than LONGEST_COMMAND characters, so a longer one matches no allow rule and
# dontAsk refuses it: in the baseline of 2026-11-06 an 11,000-character ffmpeg montage was refused twice, and Claude took ffmpeg
# itself for denied and gave up. The line says so.
LONGEST_COMMAND = 10_000
BASE_FILES = ("You answer the user's question from their files, using your tools. Bash runs ffmpeg only: start each Bash "
              f"command with ffmpeg (no cd, pipes, && or other programs), keep it under {LONGEST_COMMAND:,} characters (a longer "
              "one is refused), and write what it makes in your working folder. "
              "Find and read files with Glob, Grep and Read; Read shows you the pictures ffmpeg writes.")

# `claude --help` (2.1.302) does not list --max-turns or --append-system-prompt-file, but both are taken (checked on
# 2026-11-02). If a later version refuses one, set PROMPT_FILE to False (the text then goes in --append-system-prompt,
# about 26k characters, under Windows' 32k command line) or MAX_TURNS to None (no cap).
PROMPT_FILE = True
MAX_TURNS: int | None = 60
BUDGET_USD = 5.0  # a list-price estimate per exec; the plan's own limit is separate
CMD_LIMIT = 32_000  # characters of a Windows command line (CreateProcess takes 32,767)
FILES_TOOLS = ("Read", "Glob", "Grep", "Bash")
FILES_ALLOWED = ("Read", "Glob", "Grep", "Bash(ffmpeg *)")
# Claude's built-in plugins load under --restricted too (2.1.302: these two, each adding a skill); off, the init lists none
BUILTIN_PLUGINS = ("cc-plugin-agents-md@builtin", "cc-plugin-plugin-authoring@builtin")
SETTINGS = {"disableAllHooks": True, "enabledPlugins": dict.fromkeys(BUILTIN_PLUGINS, False)}
ENV_SET = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",  # no user, project or auto-memory CLAUDE.md
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "ENABLE_CLAUDEAI_MCP_SERVERS": "false",  # no claude.ai connectors
           "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
           "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",  # a backgrounded call's result is lost in -p
           "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}
ENV_DROP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")  # -p uses a key when one is set; the bench uses the plan
ENV_DROP_FILES = ("VIRTUAL_ENV",)  # and, from an agent with its own tools, these and the variables `Isolation` names (the owner's)

MCP_PREFIX = re.compile(r"^mcp__.+?__")  # of a tool the session names `mcp__<server>__<tool>`
TOKEN_KEYS = {"input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_input_tokens",
              "cache_write": "cache_creation_input_tokens"}  # the result's `usage`
LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|rate limit|rate_limit|\b429\b|quota", re.I)
# also how Claude tells a login is needed: "Failed to authenticate: OAuth session expired and could not be refreshed"
AUTH = re.compile(AUTH_BASE.pattern + r"|/login|invalid api key|failed to authenticate|oauth", re.I)
CAPPED = ("error_max_turns", "error_max_budget_usd")  # result subtypes: the turn or the dollar cap was reached
# what `claude` prints, from the text of claude.exe (commander's), when it refuses an option
SETUP = re.compile(r"error: (unknown option|unknown command|too many arguments|missing required argument|required option|"
                   r"option '[^']*' argument)", re.I)


class ClaudeMissing(RuntimeError):
    """Nothing can run: no usable `claude` command, or a command line Windows would cut."""


def model() -> str:
    return os.environ.get("MMMB_CLAUDE_MODEL", "sonnet")


def effort() -> str:
    return os.environ.get("MMMB_CLAUDE_EFFORT", "high")


def claude_command() -> list[str]:
    found = shutil.which("claude")
    if not found:
        raise ClaudeMissing("the claude command is not on PATH: install Claude Code and log in")
    if found.lower().endswith((".cmd", ".bat")):
        raise ClaudeMissing(f"{found} is a .cmd shim, which cmd.exe would misread the arguments of (an empty --tools, "
                            "parentheses): use the native claude.exe")
    return [found]


def check_login(command: list[str] | None = None) -> None:
    """Raise ClaudeMissing unless `claude auth status`, in an exec's environment (no API key), says the CLI is logged in.
    The CLI has a login of its own, which can be out of date while the desktop app's is fine."""
    done = subprocess.run([*(command or claude_command()), "auth", "status"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=60, env=env_of(None))
    try:
        status = json.loads(done.stdout)
    except ValueError:
        raise ClaudeMissing(f"`claude auth status` gave no status (exit {done.returncode}): "
                            f"{(done.stdout + done.stderr).strip()[-300:]}") from None
    if not isinstance(status, dict) or status.get("loggedIn") is not True:
        raise ClaudeMissing("the claude CLI is not logged in (`claude auth status`): run `claude auth login` in a "
                            "terminal, then the same command")


def server_tools(spec: SessionSpec) -> tuple[str, ...]:
    """The names Claude gives the tools of the server of `spec`, if it has one."""
    return tuple(f"mcp__{spec.server.name}__{tool}" for tool in spec.server.tools) if spec.server else ()


def session_tools(spec: SessionSpec) -> tuple[str, ...]:
    """The tools a session of `spec` lists in its init event: its own, then its server's."""
    return (*(FILES_TOOLS if spec.own_tools else ()), *server_tools(spec))


def mcp_config(server: McpServer) -> dict:
    """The `--mcp-config` of a session with a server: the one server, as the system gave it."""
    return {"mcpServers": {server.name: {"command": server.command, "args": list(server.args), "env": dict(server.env)}}}


def argv_of(command: list[str], spec: SessionSpec, *, name: str, level: str) -> list[str]:
    """The command line of an exec (the prompt is on stdin); `spec.run_dir` holds the files `prepare` wrote."""
    run_dir = spec.run_dir
    args = [*command, "-p", "--verbose", "--output-format", "stream-json", "--restricted", "--strict-mcp-config",
            "--permission-mode", "dontAsk", "--no-session-persistence", "--disable-slash-commands",
            "--model", name, "--effort", level, "--max-budget-usd", f"{BUDGET_USD:g}",
            "--settings", str(run_dir / "settings.json")]
    if MAX_TURNS:
        args += ["--max-turns", str(MAX_TURNS)]
    if PROMPT_FILE:
        args += ["--append-system-prompt-file", str(run_dir / "instructions.md")]
    else:
        args += ["--append-system-prompt", (run_dir / "instructions.md").read_text(encoding="utf-8")]
    if spec.server:
        args += ["--mcp-config", str(run_dir / "mcp.json")]
    if not spec.own_tools:
        return [*args, "--tools", "", *(["--allowedTools", *server_tools(spec)] if spec.server else [])]
    if spec.data is None:
        raise ValueError("an agent with its own tools needs the data folder")
    allowed = [*FILES_ALLOWED, *server_tools(spec)]
    handed = [arg for folder in spec.served.opens for arg in ("--add-dir", str(folder))]  # the file tools may open what the system handed it
    return [*args, "--tools", ",".join(FILES_TOOLS), "--allowedTools", *allowed, "--add-dir", str(spec.data), *handed]


def git_root(search: str | None = None) -> Path | None:
    """The root of Git for Windows, which Claude's Bash tool runs in. It is found from the folders of `search` (default:
    this process's PATH) that hold a `git.exe`, and failing that a `bash.exe`: going up from it to the first folder that
    holds `usr/bin/bash.exe`. (A `bash.exe` of System32 is WSL's: nothing above it matches.) None off Windows or when
    it is not found."""
    if os.name != "nt":
        return None
    folders = [Path(f) for f in (os.environ.get("PATH", "") if search is None else search).split(os.pathsep) if f]
    for exe in ("git.exe", "bash.exe"):
        for found in (folder / exe for folder in folders if (folder / exe).is_file()):
            for parent in found.parents:
                if (parent / "usr" / "bin" / "bash.exe").is_file():
                    return parent
    return None


def env_of(bin_dir: Path | None, own_tools: bool = False, work: Path | None = None, isolation: Isolation | None = None,
           off_gpu: bool = False) -> dict:
    """This process's environment less the API key, plus `ENV_SET`, with `bin_dir` first on PATH. An agent with its own tools
    (with its working folder `work` and the `isolation` of the systems) also gets a PATH of `bin_dir`, Git for Windows' folders
    and the Windows folders only, with Claude told where its bash is, the variables of `isolation.env_nowhere` leading to
    folders that do not exist and none of the owner's others that start as `isolation.env_drop_prefixes` say, or VIRTUAL_ENV:
    its Bash tool cannot reach the owner's programs or `.venv`. Kept `off_gpu`, it sees no CUDA device (`OFF_GPU_ENV`): in its process,
    so in its Bash tool and in the MCP server it starts, neither of which uses the GPU (the model runs in the system's daemon, which the
    run starts with the run's own environment and not this one)."""
    if off_gpu and not own_tools:
        raise ValueError("only an agent with its own tools is kept off the GPU")
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    if not own_tools:
        return path_with(bin_dir, {**env, **ENV_SET})
    if work is None or isolation is None:
        raise ValueError("an agent with its own tools needs its working folder and the isolation")
    env = {k: v for k, v in env.items() if k not in ENV_DROP_FILES and not k.startswith(isolation.env_drop_prefixes)}
    root = git_root()
    if os.name == "nt" and root is None:
        raise ClaudeMissing("Git for Windows was not found from `git` or `bash` on PATH: Claude's Bash tool needs it")
    folders, bash = (), None
    if root:
        folders = tuple(str(root / sub) for sub in ("cmd", "bin", "usr/bin") if (root / sub).is_dir())
        bash = next((root / sub / "bash.exe" for sub in ("bin", "usr/bin") if (root / sub / "bash.exe").is_file()), None)
    env.update(ENV_SET)
    env["PATH"] = shell_path(bin_dir, folders)
    env.update({var: str(work / folder) for var, folder in isolation.env_nowhere.items()})
    if bash:
        env["CLAUDE_CODE_GIT_BASH_PATH"] = str(bash)
    if off_gpu:
        env.update(OFF_GPU_ENV)
    return env


def prepare(spec: SessionSpec, command: list[str] | None = None) -> list[str]:
    """Write `instructions.md`, `settings.json` and, for a session with a server, `mcp.json` into `spec.run_dir`, make
    `spec.work`, and return the command line."""
    command = command or claude_command()
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    spec.work.mkdir(parents=True, exist_ok=True)
    (run_dir / "instructions.md").write_text(instructions_of(spec, BASE_FILES), encoding="utf-8", newline="\n")
    (run_dir / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8", newline="\n")
    if spec.server:
        (run_dir / "mcp.json").write_text(json.dumps(mcp_config(spec.server), indent=1), encoding="utf-8", newline="\n")
    argv = argv_of(command, spec, name=spec.model or model(), level=spec.effort or effort())
    if os.name == "nt" and (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
        raise ClaudeMissing(f"the command line is {size} characters, over the {CMD_LIMIT} Windows passes")
    return argv


def is_init(event: dict) -> bool:
    return event.get("type") == "system" and event.get("subtype") == "init"


def check_init(event: dict, spec: SessionSpec) -> str | None:
    """Why the session the init event describes is not the one asked for (its tools, MCP servers or plugins differ from
    the mode's, or it does not say, or it is billed to an API key), else None."""
    tools = event.get("tools")
    if not isinstance(tools, list):
        return "the init event lists no tools, so what the session can use is unknown"
    want, have = set(session_tools(spec)), set(map(str, tools))
    if have != want:
        return f"the session's tools are not the {spec.arm} arm's: extra {sorted(have - want)}, missing {sorted(want - have)}"
    servers = event.get("mcp_servers") or []
    if not isinstance(servers, list):
        return f"the init event's mcp_servers is not a list: {servers!r}"
    if spec.server:
        if len(servers) != 1 or not isinstance(servers[0], dict) or servers[0].get("name") != spec.server.name:
            return f"the session's MCP servers are not just {spec.server.name!r}: {servers!r}"
        if servers[0].get("status") != "connected":
            return f"the {spec.server.name!r} server is {servers[0].get('status')!r}, not connected"
    elif servers:
        return f"the {spec.arm} arm's session has MCP servers: {servers!r}"
    if event.get("plugins"):
        return f"the session has plugins: {event['plugins']!r}"
    if event.get("apiKeySource") not in (None, "none"):
        return f"the session is billed to an API key ({event['apiKeySource']}), not the plan's login"
    return None


def own_outputs(value, session: str | None):
    """`value` with the folder where Claude Code keeps this session's long tool outputs named `/agent-outputs` instead: a Bash
    output over Claude Code's limit is saved there and its result names the file (`<persisted-output>`), so an agent with its
    own tools may read its own output back, though the folder is under ~/.claude. Another session's folder, and a path that
    climbs out of this one (`..`), are left as they are."""
    if not session:
        return value
    folder = re.compile(r"[\\/]\.claude[\\/]+projects[\\/]+[^\\/\s\"']+[\\/]+" + re.escape(session)
                        + r"[\\/]+tool-results(?=[\\/\s\"']|$)(?![^\s\"']*\.\.)", re.IGNORECASE)

    def put(v):
        if isinstance(v, str):
            return folder.sub("/agent-outputs", v)
        if isinstance(v, list):
            return [put(x) for x in v]
        if isinstance(v, dict):
            return {k: put(x) for k, x in v.items()}
        return v

    return put(value)


def check_event(event: dict, spec: SessionSpec, session: str | None = None) -> str | None:
    """Why an assistant message ends the exec: a tool the mode does not have (the blind mode has none), or an input of one of its
    own tools holding a `deny` string, outside the `session`'s own outputs (`own_outputs`). A server's tool is checked by name
    alone. Else None."""
    if event.get("type") != "assistant":
        return None
    allowed = session_tools(spec)
    for block in blocks_of(event):
        if not str(block.get("type")).endswith("tool_use"):
            continue
        name = str(block.get("name"))
        if name not in allowed:
            return f"claude used {name!r}, which is not one of the {spec.arm} arm's tools: the restriction failed"
        if name in FILES_TOOLS and (hit := reached(own_outputs(ran_of(block), session), spec.deny, opens=spec.served.opens)):
            return f"{name} reached {hit}"
    return None


def ran_of(block: dict):
    """What a tool use asks for, to be checked against the deny list: a shell command is its `command` alone (its `description`
    is the agent's prose about it, which may say anything), any other tool's input without its `description`."""
    value = block.get("input")
    if not isinstance(value, dict):
        return value
    if block.get("name") == "Bash" and "command" in value:
        return value["command"]
    return {k: v for k, v in value.items() if k != "description"}


def check_result(event: dict, spec: SessionSpec, names: dict, session: str | None = None) -> str | None:
    """Why the results of tools in a `user` event end the exec: output of an own tool that holds a `deny` string, a picture's
    data and the `session`'s own outputs folder left out (`own_outputs`). `names` maps a tool use's id to its tool; a result of
    a server's tool is not read (it may name paths of its own). Else None."""
    if not spec.own_tools or event.get("type") != "user":
        return None
    theirs = server_tools(spec)
    for block in blocks_of(event):
        if block.get("type") != "tool_result" or names.get(block.get("tool_use_id")) in theirs:
            continue
        content = block.get("content")
        if isinstance(content, list):
            content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "image")]
        if hit := reached(own_outputs(content, session), spec.deny, output=True, opens=spec.served.opens):
            return f"{names.get(block.get('tool_use_id'), 'a tool')} output reached {hit}"
    return None


def watcher(spec: SessionSpec) -> Callable[[dict], str | None]:
    """A check for the events of one exec, in order: the init event, which must come before the first assistant message,
    then each assistant message and each tool's result. It returns the problem that must end the exec (`setup: ...` for a
    session that is not the one asked for, `escaped: ...` for what Claude did), else None."""
    inited = False
    session: str | None = None  # the init event's session_id: its own outputs folder may be read
    names: dict[str, str] = {}

    def watch(event: dict) -> str | None:
        nonlocal inited, session
        if is_init(event):
            inited = True
            session = str(event.get("session_id") or "") or None
            return f"setup: {why}" if (why := check_init(event, spec)) else None
        if event.get("type") == "assistant":
            if not inited:
                return "setup: an assistant message came before the init event, so what the session can use is unknown"
            names.update({str(b.get("id")): str(b.get("name")) for b in blocks_of(event) if b.get("type") == "tool_use"})
            why = check_event(event, spec, session)
        else:
            why = check_result(event, spec, names, session)
        return f"escaped: {why}" if why else None

    return watch


def tools_of(events: list[dict]) -> list[dict]:
    """Each call of a tool: its bare name (a server's without its `mcp__<server>__`), input, result (without picture data) and
    whether it failed. A call is once, by its id, however many events carry it."""
    results = {b.get("tool_use_id"): b for e in events if e.get("type") == "user" for b in blocks_of(e)
               if b.get("type") == "tool_result"}
    out, seen = [], set()
    for e in events:
        if e.get("type") != "assistant":
            continue
        for block in blocks_of(e):
            if block.get("type") != "tool_use" or block.get("id") in seen:
                continue
            seen.add(block.get("id"))
            got = results.get(block.get("id"), {})
            out.append({"tool": MCP_PREFIX.sub("", str(block.get("name"))), "arguments": block.get("input"),
                        "result": scrubbed(got.get("content")), "error": bool(got.get("is_error"))})
    return out


def used_percent_of(events: list[dict]) -> float | None:
    """The week's plan use in percent from the last `rate_limit_event` that gives it, else None: the seven-day window of its
    `unifiedWindows`, or its own `utilization` when `rateLimitType` is `seven_day`. Both are fractions of the window."""
    used = None
    for e in events:
        info = e.get("rate_limit_info") if e.get("type") == "rate_limit_event" else None
        if not isinstance(info, dict):
            continue
        windows = info.get("unifiedWindows")
        week = windows.get("seven_day") if isinstance(windows, dict) else None
        if isinstance(week, dict):
            fraction = week.get("utilization")
        elif info.get("rateLimitType") == "seven_day":
            fraction = info.get("utilization")
        else:
            continue
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            used = float(fraction) * 100
    return used


def problem_of(result: dict | None, code: int | None, stderr: str) -> str | None:
    """`kind: reason` when the exec did not succeed, else None. `capped`: the turn or the dollar cap was reached. `setup`:
    `claude` exited without a result after refusing an option. A usage limit or a login problem is told from the result's
    text or stderr; no result at all is an error."""
    if result is not None and result.get("subtype") == "success" and not result.get("is_error"):
        return None
    if result is not None and result.get("subtype") in CAPPED:
        return f"capped: {result['subtype']}"
    if result is None and (refused := next((x for x in stderr.splitlines() if SETUP.search(x)), None)):
        return f"setup: {refused.strip()[:500]}"
    text = str((result or {}).get("result") or "").strip()
    tail = stderr.strip()[-500:]
    for kind, pattern in (("limit", LIMIT), ("auth", AUTH)):
        if pattern.search(text) or pattern.search(tail):
            return f"{kind}: {(text or tail)[:500]}"
    if result is None:
        return f"error: claude exited {code} without a result"
    subtype = result.get("subtype")
    why = ": ".join(part for part in (subtype if subtype != "success" else "", text[:500]) if part)
    return f"error: {why or 'the result is an error with no text'}"


def result_of(events: list[dict], code: int | None, stderr: str) -> dict:
    """What the stream says of the exec: `final` (the result's text, else the last assistant text), `problem`, `turns`,
    `tokens`, `cost_usd`, `model`, `denials` (permission denials) and `used_percent`."""
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    texts = [str(b.get("text") or "") for e in events if e.get("type") == "assistant" for b in blocks_of(e)
             if b.get("type") == "text"]
    final = str((result or {}).get("result") or "") or (texts[-1] if texts else "")
    usage = (result or {}).get("usage")
    tokens = {name: int(usage.get(key) or 0) for name, key in TOKEN_KEYS.items()} if isinstance(usage, dict) else None
    init = next((e for e in events if is_init(e)), None)
    first = next((e["message"].get("model") for e in events if e.get("type") == "assistant"
                  and isinstance(e.get("message"), dict) and e["message"].get("model")), None)
    cost = (result or {}).get("total_cost_usd")
    denials = (result or {}).get("permission_denials")
    return {"final": final, "problem": problem_of(result, code, stderr), "turns": (result or {}).get("num_turns"),
            "tokens": tokens, "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
            "model": (init or {}).get("model") or first, "denials": len(denials) if isinstance(denials, list) else 0,
            "used_percent": used_percent_of(events)}


def ask_claude(spec: SessionSpec) -> dict:
    """Run one exec for `spec.prompt` in the working folder `spec.work`, with its files and logs in `spec.run_dir`. A server and
    own tools are as `spec` says; `spec.bin_dir` goes first on PATH."""
    run_dir = spec.run_dir
    argv = prepare(spec)
    (run_dir / "prompt.txt").write_text(spec.prompt, encoding="utf-8")
    events: list[dict] = []
    stopped: list[str] = []
    watch = watcher(spec)
    stderr_file = (run_dir / "stderr.txt").open("wb")
    clock = WorkingClock.of(spec.timeout_s, spec.queue_s)  # (`seconds` is its wall time: it starts with the process, not the gate or the files)
    process = subprocess.Popen(argv, cwd=spec.work, env=env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation, spec.off_gpu),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file,
                               start_new_session=os.name != "nt")

    def read() -> None:
        for raw in process.stdout:
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if problem := watch(event):
                stopped.append(problem)
                kill_tree(process)
                break

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(spec.prompt.encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass  # an exec that died before reading its prompt is told by its exit
    timed_out = wait_for(process, clock)
    reader.join(30)
    stderr_file.close()
    figures = clock.figures()
    seconds = figures.pop("wall_s")  # (what `seconds` has always been: the wall time of the process)
    stderr = (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
    done = result_of(events, process.returncode, stderr)
    problem = stopped[0] if stopped else timed_out or done["problem"]
    out = {"final": done["final"], "tools": tools_of(events),
           "events": [scrubbed(e) for e in events], "seconds": seconds, **figures, "turns": done["turns"],
           "tokens": done["tokens"], "cost_usd": done["cost_usd"], "used_percent": done["used_percent"],
           "model": done["model"], "effort": spec.effort or effort(), "denials": done["denials"], "problem": problem, "argv": argv,
           "stderr": stderr}
    if init := next((e for e in events if is_init(e)), None):
        (run_dir / "init.json").write_text(json.dumps(scrubbed(init), indent=1), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in out["events"]), encoding="utf-8")
    (run_dir / "tools.json").write_text(json.dumps(out["tools"], indent=1), encoding="utf-8")
    (run_dir / "final.txt").write_text(done["final"], encoding="utf-8")
    return out
