"""A configuration: what an agent is given for one run of a question (DESIGN 4). It is a system's tools, the agent's own tools, both,
or nothing (`blind`). Results and exec folders are named by `Config.dir`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from mmmb.core import fs
from mmmb.core.contract import Description, System

CONTROL = "files"  # the configuration the others are set beside unless a report is told another: the agent's own tools alone


@dataclass(frozen=True)
class Config:
    name: str  # as written, with the default variant filled in: `mm/d1-3B+files`
    system: str | None  # None: no system
    variant: str | None
    own_tools: bool  # the agent has its shell and file tools

    @property
    def dir(self) -> str:
        """The folder name in `results/<agent>/` and `execs/<agent>/`: `blind`, `files`, `mm-d1-3B`, `mm-d1-3B+files`."""
        base = "blind" if self.system is None else self.system if self.variant is None else f"{self.system}-{fs.slug(self.variant)}"
        return base + "+files" if self.own_tools and self.system not in (None, "files") else base  # `files` is its own tools


def parse_config(text: str, defaults: Mapping[str, str | None]) -> Config:
    """The configuration written `text`: `blind`, `<system>[/<variant>][+files]`. `defaults` has each installed system's default
    variant, None for a system without variants (the systems' system.toml say so). Raises a ValueError for anything else."""
    text = text.strip()
    plus = text.endswith("+files")
    base = text.removesuffix("+files")
    system, slash, variant = base.partition("/")
    if "+" in base:
        raise ValueError(f"{text!r} is not a configuration: the only + is a final +files")
    if system == "gold":
        raise ValueError("gold is not built yet (DESIGN section 4; not P0)")
    if system == "blind":
        if slash or plus:
            raise ValueError(f"blind has no tools to add to and no variants: {text!r}")
        return Config("blind", None, None, False)
    if system not in defaults:
        raise ValueError(f"{text!r} is not a configuration: blind, or one of the systems {', '.join(sorted(defaults))}, "
                         "as <system>[/<variant>][+files]")
    if system == "files":
        if slash or plus:
            raise ValueError(f"files is the agent's own tools already and has no variants: {text!r}")
        return Config("files", "files", None, True)
    if defaults[system] is None:
        if slash:
            raise ValueError(f"{system} has no variants: {text!r}")
    elif slash and not variant:
        raise ValueError(f"{text!r} names no variant after the slash")
    else:
        variant = variant or defaults[system]
    return Config(system + (f"/{variant}" if variant else "") + ("+files" if plus else ""), system, variant or None, plus)


def describe_configs(configs: Sequence[Config], load: Callable[[str], System]) -> tuple[dict[str, System], dict[Config, Description | None]]:
    """The systems the configurations name, loaded by `load` (one whose extra is missing fails here), and the description of each
    configuration (None for one with no system)."""
    loaded = {c.system: load(c.system) for c in configs if c.system is not None}
    return loaded, {c: loaded[c.system].describe(c.variant) if c.system is not None else None for c in configs}
