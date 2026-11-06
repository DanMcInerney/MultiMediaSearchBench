"""The scenarios by name (DESIGN 5): a name stands for the module that is the scenario, which has the names of a `Scenario` as
functions and is imported when it is asked for. A scenario is added by its module and a line here."""

from __future__ import annotations

import importlib

from mmmb.core.scenario import Scenario

REGISTRY = {"v7_screen": "mmmb.scenarios.v7_screen", "v8_egolife": "mmmb.scenarios.v8_egolife", "i3_trailcam": "mmmb.scenarios.i3_trailcam"}


def get(name: str) -> Scenario:
    """The scenario `name` stands for, imported now. A ValueError names the scenarios there are for any other name."""
    if name not in REGISTRY:
        raise ValueError(f"{name!r} is not a scenario: {', '.join(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])  # type: ignore[return-value]  (a module with the Scenario's names)
