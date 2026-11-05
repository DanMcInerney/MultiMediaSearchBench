"""SessionSpecs for the tests of the agents: the four modes of an exec (blind, files, a server, a server and files) with a server and
instructions shaped like the ones recorded, built by hand. Nothing here imports multimem or starts a process."""

from __future__ import annotations

import sys
from pathlib import Path

from mmmb.core.agents.spec import SessionSpec
from mmmb.core.contract import Isolation, McpServer, Served

# what the system.toml of the system with a private home declare (plan 4.2); the recorded env.json files are the check of these
ISOLATION = Isolation(env_drop_prefixes=("MULTIMEM_",), deny_words=(".multimem",), deny_package_roots=("multimem",),
                      env_nowhere={"MULTIMEM_HOME": "no-mm-home"})
NO_ISOLATION = Isolation(env_drop_prefixes=(), deny_words=(), deny_package_roots=(), env_nowhere={})
TOOLS = ("ask", "find", "look", "label")
BASE_LINE = "You answer the user's question from their memory, using the mm tools."
MODES = ("blind", "files", "mm", "mm+files")  # the arm names the checks' messages use


def nothing_queued() -> float:
    """The queue of an exec of a session that has none (`PlainSession.queue_s`): what the agents' specs are given unless a test says other."""
    return 0.0


def clocked(spec: SessionSpec, seconds: float) -> dict:
    """The clock fields a stand-in agent adds to its result for an exec of `spec` that took `seconds` on the wall: the arithmetic of the plan's 6.5
    (working is wall less the queue; without a queue figure the clock is the wall), written out here and not taken from core/clock.py."""
    queue = spec.queue_s()
    return {"seconds": seconds, "clock": "wall" if queue is None else "working", "queue_s": queue,
            "working_s": seconds if queue is None else max(0.0, seconds - queue)}


def server_of(home: Path, work: Path, name: str = "mm", tools: tuple[str, ...] = TOOLS) -> McpServer:
    """The server as recorded in the pilot's mcp.json: this interpreter running a module with a home, in the project of `work`."""
    args = ("-m", "multimem.cli", "--home", str(home), "mcp", "--tools", ",".join(tools))
    return McpServer(name=name, command=sys.executable, args=args, env={"CLAUDE_PROJECT_DIR": str(work)}, tools=tools)


def served_of(server: McpServer | None, parts: tuple[str, ...] = ("THE SKILL", "THE NOTES"),
              server_instructions: str | None = "THE SERVER'S OWN WORDS", base_line: str | None = BASE_LINE, opens: tuple[Path, ...] = ()) -> Served:
    return Served(server=server, base_line=base_line if server else None, parts=parts if server else (),
                  server_instructions=server_instructions if server else None, holder="mm" if server else None,
                  clock_note=None, record={}, opens=opens)


def spec_of(mode: str = "blind", root: Path = Path("root"), *, deny: tuple[str, ...] = (), served: Served | None = None,
            isolation: Isolation = ISOLATION, **fields) -> SessionSpec:
    """A spec for `mode` (one of `MODES`) with its folders under `root`: run, w0, home, data and bin."""
    work, home = root / "w0", root / "home"
    own = mode.endswith("files")
    if served is None:
        served = served_of(server_of(home, work) if mode.startswith("mm") else None)
    values = dict(prompt="What was the sign?", run_dir=root / "run", work=work, own_tools=own, data=root / "data" if own else None,
                  bin_dir=root / "bin", served=served, deny=deny, isolation=isolation, timeout_s=900.0, queue_s=nothing_queued, model=None, effort=None)
    return SessionSpec(**{**values, **fields})
