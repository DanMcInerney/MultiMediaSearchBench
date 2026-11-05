"""A decider for tests: the fake decider (tests/mm/fake_decider.py), which also writes the environment of the process it runs in (a daemon's, the
model's) to `decider-env.json` in the home when the model loads. It shows what a model would have been given of the agent's environment.

Written for this repository (U13). It is run as `python:<this file>`, so it loads the fake by its path and holds no import of the tests."""

import importlib.util
import json
import os
from pathlib import Path

from multimem.decider_d1 import ABOUT  # noqa: F401 - the loader reads ABOUT from this module: d1's, like the fake's

_found = importlib.util.spec_from_file_location("fake_decider_of_env_decider", Path(__file__).with_name("fake_decider.py"))
_fake = importlib.util.module_from_spec(_found)
_found.loader.exec_module(_fake)
home = None  # set by the loader


def load() -> None:
    Path(home, "decider-env.json").write_text(json.dumps(dict(os.environ)), encoding="utf-8")


def decide(state, questions: dict, media: list) -> dict:
    _fake.home = home
    return _fake.decide(state, questions, media)
