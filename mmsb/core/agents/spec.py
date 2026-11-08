"""What an agent is given for one exec (plan 4.4), in place of the old `bench.Exec`: everything both agents read, and nothing of a
system or a scenario. An agent is a `Callable[[SessionSpec], dict]`. What a system offers it comes only from `served`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mmsb.core.contract import Isolation, McpServer, Served


@dataclass(frozen=True)
class SessionSpec:
    prompt: str
    run_dir: Path  # where the exec's files and logs go
    work: Path  # its working folder
    own_tools: bool  # the agent has its shell and file tools (`Config.own_tools`)
    data: Path | None  # the folder those tools read; None when it has no own tools
    bin_dir: Path  # goes first on the PATH
    served: Served  # the system's server and instructions; `PlainSession.serve` for none
    deny: tuple[str, ...]  # what no input or output of the agent's own tools may hold (`RunFolder.deny`)
    isolation: Isolation  # what the systems' system.toml keep out of the shell of an agent with its own tools
    timeout_s: float  # the working seconds the exec may take (`WorkingClock.limit_s`)
    queue_s: Callable[[], float | None]  # the seconds so far it has waited in its system's queue; None: the system cannot say (core/clock.py)
    model: str | None  # None: the agent's default
    effort: str | None
    off_gpu: bool = False  # its own tools see no CUDA device and may not name the GPU's decoder (`schedule.agent_off_gpu`)

    def __post_init__(self) -> None:
        if self.off_gpu and not self.own_tools:
            raise ValueError("only an agent with its own tools is kept off the GPU: the others have no shell to decode on")

    @property
    def server(self) -> McpServer | None:
        return self.served.server

    @property
    def arm(self) -> str:
        """What the checks' messages call this exec: `blind`, `files`, the server's name, or the server's name and `+files`."""
        if self.server is None:
            return "files" if self.own_tools else "blind"
        return self.server.name + ("+files" if self.own_tools else "")
