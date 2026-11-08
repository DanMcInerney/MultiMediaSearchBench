"""Codex asked a question (TESTBED.md, Codex recall; moved from codex_run.py of this repository at bb498a6): one `codex exec` for
the `SessionSpec` it is given, locked down to the tools of its mode. It copies the pattern of another benchmark's
`stbench.codex_exec` (read, never imported).

`ask_codex` returns `{final, tools, events, seconds, clock, working_s, queue_s, turns, tokens, cost_usd, used_percent, model, effort,
thread, problem, argv, stderr}`: the last message (the run reads the answer in it, core/answer.py), what Codex called and got back, how long it
took on the wall (`seconds`) and by the working clock it was held to (`clock`, `working_s`, `queue_s`: core/clock.py; `timeout` is that
clock running out), the turns and tokens it used, the plan usage that exec's own rollout file reports, and `problem` (None, or `escaped`,
`timeout`, `limit`, `auth` or `error` with the reason). It raises `CodexMissing` before any run when nothing can run: no `codex` command, a
model or effort it does not list, or personal instructions (`AGENTS.md`) in the Codex home, whose existence and size are all that
is looked at.

How the exec is made (the plan's section 1, tier C):
- `codex exec --json --skip-git-repo-check --ignore-user-config --ignore-rules --strict-config -m MODEL -C <working folder>
  -- -`, the prompt on stdin (a `.cmd` shim cuts a command line at 8,000 characters and reads `[&|<>^%"\r\n]` as syntax,
  so an argument holding one is refused); every setting is a `-c`, nothing is written to `~/.codex/config.toml`;
- `--disable` for the browser, computer use, image generation, apps, plugins, multi-agent and the other features that reach files
  or the web (and, without own tools, the shell and `view_image`); the instruction messages Codex adds are switched off;
- `model_catalog_json`: `codex debug models` with the model's entry patched to `tool_mode` `direct` (the gpt-6 models
  are code-mode only, which reaches MCP tools only through a script tool), no apply_patch, no extra tools;
- `model_instructions_file`: the text `common.instructions_of` makes from what the system serves (its base line, its server's own
  instructions, the skill and notes it names); skills are off in the run so an installed copy is never read.
The four modes, from `spec.own_tools` and `spec.served.server`:
- a server: the one MCP server the system serves (command, arguments and environment are the system's), required, its calls
  pre-approved, and a server that lists no tools is refused: it would offer more than the system means to. Every event is checked
  as it arrives: an item other than a message, a thought, Codex's plan list or a call of that server stops the exec as `escaped`.
- own tools: no server; the shell and `view_image` stay on, in a `workspace-write` sandbox with no network, whose cwd is the
  exec's working folder; the base instructions are one line; and the only items allowed are messages, thoughts, the plan list and
  shell commands, none of which may hold a `deny` string (a path or word that leads to the answers), in the command or in its
  output. Its shell gets `shell_environment_policy` settings: only Codex's core variables, a PATH of `bin_dir` and the Windows
  folders (so none of the owner's programs and no `.venv`), and the variables of `isolation.env_nowhere` leading to folders that do
  not exist. A path inside a folder the system handed it (`Served.opens`) is no reach. `cost_usd` is None: Codex reports none.
- blind (what the model knows with no recordings): the read-only settings with every feature disabled and no server; its base
  instructions are one line, and any item but a message, a thought, an error or the plan list is `escaped`.
- both: own tools and a server. Only the commands and `view_image` calls are held to `deny`; the server's calls are checked by
  name.
Problems, by what the benchmark does with them: `escaped`, `timeout` and `capped` (the context window ran out) count as
a wrong answer; `setup` (the exec was not the one asked for: a required MCP server or the configuration failed), `limit`
and `auth` stop the agent; `error` is run again.
`~/.codex/auth.json` is never read, copied or moved: Codex reads its own login.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from mmsb.core.agents.common import AUTH, instructions_of, kill_tree, path_with, reached, scrubbed, shell_path, wait_for
from mmsb.core.agents.spec import SessionSpec
from mmsb.core.clock import WorkingClock
from mmsb.core.runfolder import OFF_GPU_ENV


BASE_FILES = "You answer the user's question from their files, using your shell and view_image."

DISABLED = ("shell_tool", "unified_exec", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
            "computer_use", "image_generation", "apps", "plugins", "multi_agent", "multi_agent_v2", "view_image",
            "skill_search", "tool_suggest", "goals", "hooks", "workspace_dependencies", "remote_plugin", "sleep_tool")
CONFIG = ("sandbox_mode=read-only", "approval_policy=never", "web_search=disabled",
          "skills.include_instructions=false", "include_environment_context=false",
          "include_permissions_instructions=false", "include_collaboration_mode_instructions=false",
          "include_apps_instructions=false")
# Own tools: a `workspace-write` shell with no network, which is told what its shell is and where it may write
FILES_SWAP = {"sandbox_mode=read-only": "sandbox_mode=workspace-write",
              "include_environment_context=false": "include_environment_context=true",
              "include_permissions_instructions=false": "include_permissions_instructions=true"}
# On Windows the shell runs only in Codex's own sandbox: without `windows.sandbox` every command is "blocked by policy".
# "unelevated" needs no admin setup; it refuses writes to the home folder but lets the shell write TEMP, where the data
# is, so the data files are read-only and checked after each exec (`fs.freeze`, `fs.changed`).
CONFIG_FILES = (*(FILES_SWAP.get(c, c) for c in CONFIG), "sandbox_workspace_write.network_access=false",
                "windows.sandbox=unelevated", "shell_environment_policy.inherit=core",
                "shell_environment_policy.exclude=['PATH']")
DISABLED_FILES = tuple(f for f in DISABLED if f not in ("shell_tool", "unified_exec", "view_image"))
SERVER_CONFIG = ("required=true", "default_tools_approval_mode=approve", "startup_timeout_sec=60", "tool_timeout_sec=900")
CATALOG_PATCH = {"apply_patch_tool_type": None, "experimental_supported_tools": [], "multi_agent_version": None,
                 "tool_mode": "direct", "supports_search_tool": False}

WEEK = 10080  # window_minutes of the weekly limit
ITEMS = ("agent_message", "reasoning", "error", "todo_list", "mcp_tool_call")  # what the events of an exec may hold
ITEMS_FILES = ("agent_message", "reasoning", "error", "todo_list", "command_execution")  # and of the files arm's
ITEMS_BLIND = ("agent_message", "reasoning", "error", "todo_list")  # and of the blind arm's: no tool at all
HELPERS = {"codex": ("list_mcp_resources", "list_mcp_resource_templates")}  # Codex's own listers: no resources exist
PERSONAL = ("AGENTS.md", "AGENTS.override.md")  # what Codex reads from its home into every exec's messages
UNSAFE = re.compile(r'[&|<>^%"\r\n]')  # what cmd.exe reads as syntax, in an argument given to a .cmd shim
CMD_LIMIT = 8000
SHELL_OUTPUT = 3000  # characters of a shell command's output kept in tools.json
LIMIT = re.compile(r"usage limit|usage_limit|rate limit|rate_limit|quota|usage not included|\b429\b", re.I)

# Codex's own wordings, read from codex.exe's text: a required MCP server that did not start, a configuration it refused,
# and the context window running out ("Codex ran out of room in the model's context window...")
SETUP = re.compile(r"required MCP server|MCP client for .{0,80}(failed to start|timed out)|Error loading config", re.I)
CAPPED = re.compile(r"context window|context_length", re.I)
_MODELS: dict[tuple, dict] = {}  # `codex debug models`, read once a process


class CodexMissing(RuntimeError):
    """Nothing can run: no usable `codex`, an unlisted model or effort, or personal instructions in the Codex home."""


def model() -> str:
    return os.environ.get("MMSB_CODEX_MODEL", "gpt-6.1-terra")


def effort() -> str:
    return os.environ.get("MMSB_CODEX_EFFORT", "high")


def literal(text: str) -> str:
    """`text` as a TOML literal string: no escapes to get wrong in an argument."""
    if "'" in text or "\n" in text:
        raise CodexMissing(f"cannot pass {text!r} to codex: it holds a quote or a newline")
    return f"'{text}'"


def toml(value) -> str:
    """`value` as a TOML literal string with forward slashes."""
    return literal(str(value).replace("\\", "/"))


def codex_command() -> list[str]:
    found = shutil.which("codex")
    if not found:
        raise CodexMissing("the codex command is not on PATH: install the Codex CLI and run `codex login`")
    return [found]


def codex_home() -> Path:
    return Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else Path.home() / ".codex"


def personal_files() -> list[Path]:
    """The non-empty personal instruction files of the Codex home; only their existence and size are looked at."""
    return [p for p in (codex_home() / name for name in PERSONAL) if p.is_file() and p.stat().st_size > 0]


def check_personal() -> None:
    if found := personal_files():
        raise CodexMissing(f"{found[0]} holds your personal Codex instructions, which Codex adds to every exec as a "
                           "message: move it aside for the run (and put it back after)")


def models_of(command: list[str]) -> dict:
    """`codex debug models`, parsed (once a process). It only reads, so a dry run runs it too; CodexMissing when there is no Codex to ask."""
    key = tuple(command)
    if key not in _MODELS:
        try:
            done = subprocess.run([*command, "debug", "models"], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", stdin=subprocess.DEVNULL, timeout=60)
        except FileNotFoundError:
            raise CodexMissing(f"{command[0]} cannot be run: the Codex CLI is not installed or not on PATH (install it and run `codex login`; "
                               "even a dry run asks it for the model's settings)") from None
        if done.returncode:
            raise CodexMissing(f"`codex debug models` exited {done.returncode}: {done.stderr.strip()[-300:]}")
        _MODELS[key] = json.loads(done.stdout)
    return _MODELS[key]


def catalog(models: dict, name: str, level: str) -> dict:
    """The model catalog with `name`'s entry patched for direct tool mode; CodexMissing for a model or effort it lacks."""
    entries = [dict(m) for m in models.get("models", [])]
    slugs = [m.get("slug") for m in entries]
    if name not in slugs:
        raise CodexMissing(f"codex does not list the model {name!r}; it lists {', '.join(map(str, slugs))}")
    for entry in entries:
        if entry["slug"] == name:
            listed = [x.get("effort") if isinstance(x, dict) else x for x in entry.get("supported_reasoning_levels") or ()]
            if listed and level not in listed:
                raise CodexMissing(f"codex lists {name!r} at the efforts {', '.join(map(str, listed))}, not {level!r}")
            entry.update(CATALOG_PATCH)
    return {"models": entries}


def argv_of(command: list[str], name: str, level: str, spec: SessionSpec) -> list[str]:
    """The command line of an exec (the prompt is on stdin), in the working folder `spec.work`. A server is refused without its
    `tools`: it would offer more than the system means to. Its environment is the system's (for example its project, which the
    working folder pins, so the project does not hang on the cwd Codex gives it). An agent with its own tools has its shell:
    Codex's core variables, a PATH of `bin_dir` and the system folders only, and the variables of `isolation.env_nowhere` leading
    to folders that do not exist, and no CUDA device when it is kept off the GPU (`spec.off_gpu`; the codex process itself keeps its own
    environment, so its MCP server has the run's). Without a server or own tools it has
    the read-only settings and every feature disabled."""
    run_dir, work = spec.run_dir, spec.work
    settings = [f"model_reasoning_effort={level}", f"model_catalog_json={toml(run_dir / 'catalog.json')}",
                f"model_instructions_file={toml(run_dir / 'instructions.md')}"]
    if spec.own_tools:
        nowhere = spec.isolation.env_nowhere
        settings = [*CONFIG_FILES, *settings, f"shell_environment_policy.set.PATH={literal(shell_path(spec.bin_dir))}",
                    *(f"shell_environment_policy.set.{var}={toml(work / folder)}" for var, folder in nowhere.items()),
                    *(f"shell_environment_policy.set.{var}={toml(value)}" for var, value in OFF_GPU_ENV.items() if spec.off_gpu)]
        disabled = DISABLED_FILES
    else:
        settings = [*CONFIG, *settings]
        disabled = DISABLED
    if server := spec.server:
        if not server.tools:
            raise CodexMissing(f"the {server.name} server lists no tools: without them Codex is offered all it has")
        settings += [f"mcp_servers.{server.name}.command={toml(server.command)}",
                     f"mcp_servers.{server.name}.args=[{','.join(toml(a) for a in server.args)}]",
                     *(f"mcp_servers.{server.name}.{c}" for c in SERVER_CONFIG),
                     *(f"mcp_servers.{server.name}.env.{key}={toml(value)}" for key, value in server.env.items())]
    args = [*command, "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
            "--strict-config", "-m", name]
    for setting in settings:
        args += ["-c", setting]
    for feature in disabled:
        args += ["--disable", feature]
    return [*args, "-C", str(work), "--", "-"]


def check_argv(argv: list[str], command: list[str]) -> None:
    """A `.cmd` shim is run by cmd.exe: refuse an argument it would misread and a command line it would cut."""
    if os.name == "nt" and argv[0].lower().endswith((".cmd", ".bat")):
        for arg in argv[len(command):]:
            if UNSAFE.search(arg):
                raise CodexMissing(f"an argument holds a character that cmd.exe reads as syntax: {arg[:80]!r}")
        if (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
            raise CodexMissing(f"the command line is {size} characters, over the {CMD_LIMIT} a .cmd shim passes")


def prepare(spec: SessionSpec, name: str, level: str, models: dict | None = None,
            command: list[str] | None = None) -> list[str]:
    """Write `catalog.json`, `instructions.md` and the working folder into `spec.run_dir`, and return the command line.
    `models` is a saved `codex debug models` output (a dry run); else the installed Codex is asked."""
    command = command or codex_command()
    models = models if models is not None else models_of(command)
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    spec.work.mkdir(parents=True, exist_ok=True)
    (run_dir / "catalog.json").write_text(json.dumps(catalog(models, name, level)), encoding="utf-8", newline="\n")
    (run_dir / "instructions.md").write_text(instructions_of(spec, BASE_FILES, server_instructions=True), encoding="utf-8",
                                             newline="\n")
    argv = argv_of(command, name, level, spec)
    check_argv(argv, command)
    return argv


def items_of(spec: SessionSpec) -> tuple[str, ...]:
    """The item types the events of an exec of this mode may hold."""
    if spec.server is None:
        return ITEMS_FILES if spec.own_tools else ITEMS_BLIND
    return tuple(dict.fromkeys((*ITEMS, *(ITEMS_FILES if spec.own_tools else ()))))


def refusal_of(spec: SessionSpec) -> str:
    """How `check_event` ends the sentence about an item the mode does not allow."""
    if spec.server is None:
        return ("the files arm does not allow: the restriction failed" if spec.own_tools
                else "the blind arm does not allow: it has no tools")
    plus = " or the files arm's" if spec.own_tools else ""
    return f"is not one of {spec.server.name}'s tools{plus}: the restriction failed"


def check_event(event: dict, spec: SessionSpec) -> str | None:
    """Why an event ends the exec, else None. An item that is not allowed in the mode: a message, a thought and a plan list are,
    a call of the server with a server, a shell command with own tools, nothing else (so any tool item at all without either).
    A command or its output that holds a `deny` string. A server's calls are checked by name."""
    if event.get("type") not in ("item.started", "item.completed", "item.updated"):
        return None
    item = event.get("item") or {}
    what = item.get("type")
    if what not in items_of(spec):
        return f"codex used {what!r}, which {refusal_of(spec)}"
    if what == "command_execution" and (hit := reached(item.get("command"), spec.deny, opens=spec.served.opens)):
        return f"shell reached {hit}"
    if what == "command_execution" and (hit := reached(item.get("aggregated_output"), spec.deny, output=True, opens=spec.served.opens)):
        return f"shell output reached {hit}"
    if what != "mcp_tool_call" or item.get("tool") in HELPERS.get(item.get("server"), ()):
        return None
    if item.get("server") != spec.server.name:
        return f"codex called {item.get('tool')!r} of the server {item.get('server')!r}, not {spec.server.name!r}"
    if event["type"] == "item.completed" and "requires approval" in str((item.get("error") or {}).get("message", "")):
        return "a tool call was held for approval: the pre-approval did not apply"
    return None


def tools_of(events: list[dict]) -> list[dict]:
    """Each completed call of a tool: its name, arguments, result (without picture data) and error. A shell command is
    `shell`, its arguments the command and its result the start of its output."""
    out = []
    for e in events:
        item = e.get("item") or {}
        if e.get("type") != "item.completed":
            continue
        if item.get("type") == "mcp_tool_call":
            out.append({"tool": item.get("tool"), "arguments": item.get("arguments"),
                        "result": scrubbed(item.get("result")), "error": item.get("error"),
                        "status": item.get("status")})
        elif item.get("type") == "command_execution":
            code = item.get("exit_code")
            out.append({"tool": "shell", "arguments": {"command": item.get("command")},
                        "result": str(item.get("aggregated_output") or "")[:SHELL_OUTPUT],
                        "error": None if code in (0, None) else f"exit code {code}", "status": item.get("status")})
    return out


def usage_of(events: list[dict]) -> tuple[int, dict | None]:
    """How many turns completed, and the tokens of all of them (None when none reports usage). Codex's `input_tokens`
    includes the cached ones, which are also `cache_read`; it has no cache writes."""
    done = [e for e in events if e.get("type") == "turn.completed"]
    usages = [e["usage"] for e in done if isinstance(e.get("usage"), dict)]
    if not usages:
        return len(done), None

    def total(key: str) -> int:
        return sum(v for u in usages if isinstance(v := u.get(key), int))

    return len(done), {"input": total("input_tokens"), "output": total("output_tokens"),
                       "cache_read": total("cached_input_tokens"), "cache_write": 0}


def weekly(limits: dict) -> float | None:
    """`used_percent` of the window of `limits` that is the week (`window_minutes` 10080, `primary` or `secondary`
    by plan); None when no window is the week or its percent is not a number."""
    for window in (limits.get("primary"), limits.get("secondary")):
        if isinstance(window, dict) and window.get("window_minutes") == WEEK:
            used = window.get("used_percent")
            return float(used) if isinstance(used, (int, float)) and not isinstance(used, bool) else None
    return None


def percent_in(lines) -> float | None:
    """The weekly use of the last `rate_limits` among the JSON lines of a rollout; None when it cannot be read."""
    last = None

    def walk(value):
        nonlocal last
        if isinstance(value, dict):
            if isinstance(value.get("rate_limits"), dict):
                last = weekly(value["rate_limits"])
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for line in lines:
        try:
            walk(json.loads(line))
        except ValueError:
            continue
    return last


def rollout_percent(thread: str, wait_s: float = 10.0) -> float | None:
    """The plan usage this exec's own rollout file reports: `~/.codex/sessions/**/rollout-*-<thread>.jsonl`, and no other
    session's file."""
    if not thread:
        return None
    end = time.monotonic() + wait_s
    while True:
        for file in (codex_home() / "sessions").glob(f"**/rollout-*-{thread}.jsonl"):
            with file.open(encoding="utf-8", errors="replace") as lines:
                if (found := percent_in(lines)) is not None:
                    return found
        if time.monotonic() > end:
            return None
        time.sleep(0.5)


def images_in(lines) -> list[dict]:
    """The `view_image` calls among the JSON lines of a rollout, as `tools_of` gives a call. `codex exec --json` has no
    item for them (seen 2026-11-02), so the rollout is the only place they are told."""
    out = []
    for line in lines:
        try:
            payload = json.loads(line).get("payload") or {}
        except (ValueError, AttributeError):
            continue
        if isinstance(payload, dict) and payload.get("type") == "function_call" and payload.get("name") == "view_image":
            try:
                arguments = json.loads(payload.get("arguments") or "{}")
            except ValueError:
                arguments = {"raw": payload.get("arguments")}
            out.append({"tool": "view_image", "arguments": arguments, "result": None, "error": None,
                        "status": "completed"})
    return out


def rollout_images(thread: str) -> list[dict]:
    """The `view_image` calls of this exec's own rollout file (read after `rollout_percent` waited for it)."""
    for file in (codex_home() / "sessions").glob(f"**/rollout-*-{thread}.jsonl") if thread else ():
        with file.open(encoding="utf-8", errors="replace") as lines:
            return images_in(lines)
    return []


def problem_of(code: int | None, events: list[dict], stderr: str) -> str | None:
    """`kind: reason` when the exec did not complete its turn, else None. The exit code alone is not enough. `setup`: a
    required MCP server or the configuration failed, so this was not the exec asked for. `capped`: the context window ran
    out. Then `limit` and `auth` by the last message (or stderr), else `error`."""
    types = {e.get("type") for e in events}
    if code == 0 and "turn.completed" in types and "turn.failed" not in types:
        return None
    messages = []
    for e in events:
        if e.get("type") in ("turn.failed", "error"):
            found = e["error"].get("message") if isinstance(e.get("error"), dict) else e.get("message")
            messages.append(str(found or ""))
    text = next((m for m in reversed(messages) if m), "") or stderr.strip()[-500:]
    text = text or f"codex exec exited {code} without finishing the turn"
    everything = "\n".join([*messages, stderr.strip()[-500:]])
    for kind, found in (("setup", SETUP), ("capped", CAPPED)):
        if found.search(everything):
            return f"{kind}: {next((m for m in [*messages, stderr] if found.search(m)), text)[:500]}"
    return f"{'limit' if LIMIT.search(text) else 'auth' if AUTH.search(text) else 'error'}: {text[:500]}"


def ask_codex(spec: SessionSpec) -> dict:
    """Run one exec for `spec.prompt` in the working folder `spec.work`, with its files and logs in `spec.run_dir`. A server and
    own tools are as `spec` says, `spec.deny` is what the commands of own tools may not hold, and `spec.bin_dir` goes first on its
    PATH."""
    check_personal()
    command = codex_command()
    name, level = spec.model or model(), spec.effort or effort()
    run_dir = spec.run_dir
    argv = prepare(spec, name, level, command=command)
    (run_dir / "prompt.txt").write_text(spec.prompt, encoding="utf-8")
    events: list[dict] = []
    escaped: list[str] = []
    stderr_file = (run_dir / "stderr.txt").open("wb")
    clock = WorkingClock.of(spec.timeout_s, spec.queue_s)  # (`seconds` is its wall time: it starts with the process, not the gate or the files)
    process = subprocess.Popen(argv, cwd=spec.work, env=path_with(spec.bin_dir), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=stderr_file, start_new_session=os.name != "nt")

    def read() -> None:
        for raw in process.stdout:
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if reason := check_event(event, spec):
                escaped.append(reason)
                kill_tree(process)
                break

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(spec.prompt.encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass  # an exec that died before reading its prompt is told by its exit
    problem = wait_for(process, clock)
    reader.join(30)
    stderr_file.close()
    figures = clock.figures()
    seconds = figures.pop("wall_s")  # (what `seconds` has always been: the wall time of the process)
    stderr = (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
    if escaped:
        problem = f"escaped: {escaped[0]}"
    elif problem is None:
        problem = problem_of(process.returncode, events, stderr)
    thread = next((str(e.get("thread_id") or "") for e in events if e.get("type") == "thread.started"), "")
    messages = [str((e.get("item") or {}).get("text") or "") for e in events
                if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message"]
    final = messages[-1] if messages else ""
    turns, tokens = usage_of(events)
    used = rollout_percent(thread)
    tools = tools_of(events)
    if spec.own_tools:
        images = rollout_images(thread)
        tools += images
        hit = next((h for c in images if (h := reached(c["arguments"], spec.deny, opens=spec.served.opens))), None)
        if hit and not str(problem).startswith("escaped:"):
            problem = f"escaped: view_image reached {hit}"
    out = {"final": final, "tools": tools, "events": [scrubbed(e) for e in events],
           "seconds": seconds, **figures, "turns": turns, "tokens": tokens, "cost_usd": None,
           "used_percent": used, "model": name, "effort": level, "thread": thread, "problem": problem, "argv": argv,
           "stderr": stderr}
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in out["events"]), encoding="utf-8")
    (run_dir / "tools.json").write_text(json.dumps(out["tools"], indent=1), encoding="utf-8")
    (run_dir / "final.txt").write_text(final, encoding="utf-8")
    return out
