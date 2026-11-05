"""The systems by name (DESIGN 3). A system is a folder here with a `system.toml`, which the registry reads as TOML and never by
importing the system: that is how the core knows the isolation every agent shell must have (the variables, the words and the
repositories of each system that keeps state on the machine) with the system itself absent. A system is imported only when `load`
is asked for it, and a system whose extra is not installed is refused then, with no fallback.

    [system]
    name = "mm"                        # the folder's name
    class = "mmmb.systems.mm.system:Mm"
    extra = "mm"                       # the pyproject extra it needs; omit for none
    license = "..."
    default_variant = "d1-3B"          # omit for a system without variants
    horizon = "cut"                    # or "view" (Description.horizon)

    [isolation]                        # applied to every agent shell that has its own tools, whichever system is being run
    env_drop_prefixes = ["MULTIMEM_"]
    deny_words = [".multimem"]
    deny_package_roots = ["multimem"]

    [isolation.env_nowhere]            # variable -> a folder in the working folder that is never made
    MULTIMEM_HOME = "no-mm-home"
"""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from mmmb.core.contract import Isolation, System

HERE = Path(__file__).parent
SYSTEM_KEYS = {"name", "class", "extra", "license", "default_variant", "horizon"}
ISOLATION_KEYS = {"env_drop_prefixes", "deny_words", "deny_package_roots", "env_nowhere"}
HORIZONS = ("cut", "view")


class ExtraMissing(RuntimeError):
    """A system is named whose pyproject extra is not installed."""


@dataclass(frozen=True)
class SystemInfo:
    name: str
    cls: str  # "module:Class"
    extra: str | None
    license: str
    default_variant: str | None
    horizon: str
    isolation: Isolation


def infos(folder: Path = HERE) -> dict[str, SystemInfo]:
    """Every system in `folder` (the subfolders that hold a system.toml), by name, read as TOML. A folder without a system.toml is
    not a system. A file with a key it does not know, a key missing, or a name other than its folder's is refused."""
    found = {}
    for toml_file in sorted(Path(folder).glob("*/system.toml")):
        info = read(toml_file)
        found[info.name] = info
    return found


def read(toml_file: Path) -> SystemInfo:
    try:
        raw = tomllib.loads(toml_file.read_text(encoding="utf-8"))
        system, isolation = dict(raw["system"]), dict(raw.get("isolation", {}))
        if unknown := set(raw) - {"system", "isolation"}:
            raise ValueError(f"unknown table {', '.join(sorted(unknown))}")
        if unknown := set(system) - SYSTEM_KEYS:
            raise ValueError(f"unknown key {', '.join(sorted(unknown))} in [system]")
        if unknown := set(isolation) - ISOLATION_KEYS:
            raise ValueError(f"unknown key {', '.join(sorted(unknown))} in [isolation]")
        if system["name"] != toml_file.parent.name:
            raise ValueError(f"name {system['name']!r} is not the folder's, {toml_file.parent.name!r}")
        if system["horizon"] not in HORIZONS:
            raise ValueError(f"horizon is one of {', '.join(HORIZONS)}, not {system['horizon']!r}")
        words = {key: tuple(isolation.get(key, ())) for key in ("env_drop_prefixes", "deny_words", "deny_package_roots")}
        nowhere = dict(isolation.get("env_nowhere", {}))
        if not all(isinstance(w, str) for ws in words.values() for w in ws) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in nowhere.items()):
            raise ValueError("[isolation] holds lists of strings, and env_nowhere strings by string")
        return SystemInfo(system["name"], system["class"], system.get("extra"), system["license"], system.get("default_variant"),
                          system["horizon"], Isolation(env_nowhere=nowhere, **words))
    except KeyError as error:
        raise ValueError(f"{toml_file}: {error.args[0]} is missing") from None
    except ValueError as error:  # TOMLDecodeError is one
        raise ValueError(f"{toml_file}: {error}") from None


def defaults(folder: Path = HERE) -> dict[str, str | None]:
    """Each system's default variant, None for one without variants: what `config.parse_config` needs."""
    return {name: info.default_variant for name, info in infos(folder).items()}


def isolation(folder: Path = HERE) -> Isolation:
    """What every agent shell with its own tools keeps out, whichever system is being run: the union over the systems."""
    found = list(infos(folder).values())
    nowhere: dict[str, str] = {}
    for info in found:
        for variable, place in info.isolation.env_nowhere.items():
            if nowhere.setdefault(variable, place) != place:
                raise ValueError(f"{variable} leads to {nowhere[variable]!r} in one system.toml and {place!r} in {info.name}'s")
    return Isolation(
        env_drop_prefixes=tuple(dict.fromkeys(p for i in found for p in i.isolation.env_drop_prefixes)),
        deny_words=tuple(dict.fromkeys(w for i in found for w in i.isolation.deny_words)),
        deny_package_roots=tuple(dict.fromkeys(r for i in found for r in i.isolation.deny_package_roots)),
        env_nowhere=nowhere)


def load(name: str, folder: Path = HERE) -> System:
    """The system `name`, imported now. A ValueError names the systems there are for any other name; an ExtraMissing says which
    extra to install when the system's extra is not (a missing package that is not the system's own)."""
    known = infos(folder)
    if name not in known:
        raise ValueError(f"{name!r} is not a system: {', '.join(known)}")
    info = known[name]
    module, _, cls = info.cls.partition(":")
    try:
        return getattr(importlib.import_module(module), cls)()
    except ModuleNotFoundError as error:
        if info.extra and error.name and error.name.split(".")[0] != module.split(".")[0]:
            raise ExtraMissing(f"{name} needs the {info.extra} extra: uv sync --extra {info.extra}") from error
        raise
