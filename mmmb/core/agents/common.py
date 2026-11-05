"""What the two agents share (moved from claude_run.py, codex_run.py, bench.py and egolife.py of this repository at bb498a6): a
tool's input or output read against the deny words, long data kept out of a log, a process tree ended, the PATH an exec gets, the text
of the instructions file, and the JSON a tool's result holds. (The final answer in the message is read by core/answer.py, in the run.)"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from mmmb.core.agents.spec import SessionSpec
from mmmb.core.clock import STEP_S, WorkingClock
from mmmb.core.runfolder import IN_ONLY


BASE_BLIND = "You answer the user's question from what you know; you have no tools."

AUTH = re.compile(r"not logged in|not signed in|codex login|log in again|unauthori[sz]ed|\b401\b|credentials|"
                  r"refresh token|sign in", re.I)


def system_path() -> list[str]:
    """The folders of the operating system itself: on Windows System32, the Windows folder, Wbem and Windows PowerShell."""
    if os.name != "nt":
        return ["/usr/bin", "/bin"]
    root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or "C:\\Windows"
    return [f"{root}\\System32", root, f"{root}\\System32\\Wbem", f"{root}\\System32\\WindowsPowerShell\\v1.0"]


def shell_path(bin_dir: Path | None, extra: tuple[str, ...] = ()) -> str:
    """The PATH of the shell of an agent with its own tools: `bin_dir` (ffmpeg), `extra` and the system folders, and nothing of
    the owner's (so none of their programs and no `.venv`)."""
    return os.pathsep.join([*([str(bin_dir)] if bin_dir is not None else []), *extra, *system_path()])


def opened(value, opens: Sequence[Path]):
    """`value` (a command, a tool's input, with strings in lists and dicts) with the path of each folder of `opens` named `/opened`
    instead, where what follows is a name inside it: a path that climbs out of the folder (`..`) is left as it is."""
    if not opens:
        return value
    folders = [re.compile(r"[\\/]+".join(re.escape(part) for part in re.split(r"[\\/]+", str(folder).rstrip("\\/")))
                          + r"(?=[\\/\s\"']|$)(?![^\s\"']*\.\.)", re.IGNORECASE) for folder in opens]

    def put(v):
        if isinstance(v, str):
            for folder in folders:
                v = folder.sub("/opened", v)
            return v
        if isinstance(v, list):
            return [put(x) for x in v]
        if isinstance(v, dict):
            return {k: put(x) for k, x in v.items()}
        return v

    return put(value)


def reached(value, deny: tuple[str, ...], output: bool = False, opens: Sequence[Path] = ()) -> str | None:
    """The first of `deny` that `value` (a command, a tool's input; with `output`, what a tool printed) holds, read as lower-case
    JSON with forward slashes (each run of slashes and backslashes is one `/`); else None. `deny` is lower case with forward
    slashes; an `IN_ONLY` entry is left out of an output (a listing may name a folder that a command may not open). A path inside a
    folder of `opens` (`Served.opens`: what the system handed the agent) is not a reach."""
    text = re.sub(r"[\\/]+", "/", json.dumps(opened(value, opens), ensure_ascii=False).lower())
    for d in deny:
        if d.startswith(IN_ONLY):
            if output:
                continue
            d = d[len(IN_ONLY):]
        if d and d in text:
            return d
    return None


BLOB = re.compile(r"[A-Za-z0-9+/=_-]{200}")  # the start of base64 (or base64url): 200 characters with no space or punctuation


def scrubbed(value):
    """`value` with long data (a `data:` URL, an image block's base64) as its hash, so a log holds no pictures. Text that starts
    like JSON is never hashed: a server tool's result is a long compact JSON string, which the run's counts and the system's
    audit read."""
    if isinstance(value, dict):
        return {k: scrubbed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrubbed(v) for v in value]
    if isinstance(value, str) and len(value) > 3000 and (value.startswith("data:") or BLOB.match(value)):
        return f"<{len(value)} characters, sha1 {hashlib.sha1(value.encode()).hexdigest()[:12]}>"
    return value


def kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()


def wait_for(process: subprocess.Popen, clock: WorkingClock) -> str | None:
    """Waits for `process` to end or for `clock` to run out, looking at the clock at least every `STEP_S`: the problem text, after
    the process tree is killed, when the clock ran out; None when the process ended first."""
    while True:
        try:
            process.wait(timeout=min(clock.left(), STEP_S))
            return None
        except subprocess.TimeoutExpired:
            if (problem := clock.expired()) is not None:
                kill_tree(process)
                process.wait()
                return problem


def path_with(bin_dir: Path | None, env: dict | None = None) -> dict:
    """A copy of `env` (default: this process's environment) with `bin_dir` first on PATH."""
    out = dict(os.environ if env is None else env)
    if bin_dir is not None:
        out["PATH"] = os.pathsep.join([str(bin_dir), out.get("PATH", "")])
    return out


def instructions_of(spec: SessionSpec, base_files: str, server_instructions: bool = False) -> str:
    """The text of the instructions file, from the mode of `spec`. No server: the agent's one line, `base_files` with its own tools
    (it is told how they work), else `BASE_BLIND`. A server: the system's base line, with `server_instructions` (Codex reads the
    server's own, Claude gets them through the server) and the other parts after it, and `base_files` before all of it when the
    agent has its own tools too. `Served.parts` are the system's, so a blind exec never holds them."""
    served = spec.served
    if served.server is None:
        return base_files if spec.own_tools else BASE_BLIND
    if served.base_line is None:
        raise ValueError(f"the server {served.server.name!r} has no base line for the instructions")
    own = [base_files] if spec.own_tools else []
    told = [served.server_instructions] if server_instructions and served.server_instructions else []
    return "\n\n".join([*own, served.base_line, *told, *served.parts])


def blocks_of(event: dict) -> list[dict]:
    """The content blocks of an `assistant` or `user` event's message."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def result_jsons(result) -> list[dict]:
    """Every JSON object a tool's result holds: a server's tools answer with one in a text block, and a `look` of several moments
    with one for each (Codex's `result` is `{"content": [...]}`, Claude's the list of blocks or the text). [] when there is none."""
    if isinstance(result, dict):
        if isinstance(result.get("structured_content"), dict):
            return [result["structured_content"]]
        result = result.get("content")
    if isinstance(result, str):
        texts = [result]
    elif isinstance(result, (list, tuple)):
        texts = [b.get("text") if isinstance(b, dict) else b for b in result]
    else:
        return []
    out = []
    for text in texts:
        if isinstance(text, str):
            try:
                got = json.loads(text)
            except ValueError:
                continue
            if isinstance(got, dict):
                out.append(got)
    return out


def result_json(result) -> dict | None:
    """The first JSON object a tool's result holds (`result_jsons`), else None."""
    found = result_jsons(result)
    return found[0] if found else None


def command_outputs(events: Sequence[dict]) -> list[str]:
    """What the commands and tools of an agent with its own tools printed, from the exec's full events (tools.json cuts a shell's
    output): a Codex command's output, and the content of each tool result of Claude."""
    out = []
    for e in events:
        item = e.get("item") or {}
        if item.get("type") == "command_execution":
            out.append(str(item.get("aggregated_output") or ""))
        elif e.get("type") == "user":
            for block in blocks_of(e):
                content = block.get("content") if block.get("type") == "tool_result" else None
                out.append(content if isinstance(content, str) else
                           "\n".join(str(x.get("text") or "") for x in content if isinstance(x, dict)) if isinstance(content, list) else "")
    return out
