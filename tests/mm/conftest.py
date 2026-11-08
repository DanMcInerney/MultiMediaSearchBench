"""Setup of the tests of the mm system (tests/mm): copied from tests/conftest.py of Multimem at fe0ece5 are the two autouse fixtures that keep a test
off the network and off a real `uv` (`no_hugging_face`, `no_real_uv`), and the names the tests use for the daemon (`Daemon`, `Api`, `write_config`,
`wait_for` are the system's own, mmsb/systems/mm/daemon.py); `FAKE`, the fake decider, is tests/mm/fakes.py's.

Every test module here starts with `pytest.importorskip("multimem")`, which tests/conftest.py counts and says so in its summary when the mm extra
is not installed (and `--require-mm` makes a failure). This file is read before they are, so it needs nothing of the extra at import: what it
re-exports is only there when `multimem` is.

A daemon is only ever started over a folder of the test's own (`tmp_path`), with the throwaway HOME that tests/conftest.py sets: the guard below
refuses any other home, whatever the test (and tests/conftest.py refuses a write into the owner's three folders), so that no test can start, stop or call the daemon that runs on this machine for live sessions."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "oracles"))  # the helpers of oracle 3 (test_exec.py, capture_exec.py)

if importlib.util.find_spec("multimem") is not None:
    from fakes import FAKE  # noqa: F401
    from mmsb.systems.mm.daemon import Api, Daemon, wait_for, write_config  # noqa: F401


@pytest.fixture(autouse=True)
def no_hugging_face(monkeypatch):
    """A test never reaches the hub: importing huggingface_hub in the test's process fails (a test that fakes the hub puts its
    own module in its place), and a process a test starts gets HF_HUB_OFFLINE (`env`)."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)


@pytest.fixture(autouse=True)
def no_real_uv(monkeypatch):
    """`mm model use` and `prepare` build an env with uv, which downloads gigabytes: a test that gets that far runs a script
    that fails instead. A test that wants to see uv's calls puts a recording one in its place (test_model_cli's `fake_uv`)."""
    from multimem import prepare

    monkeypatch.setattr(prepare, "uv_command", lambda: [sys.executable, "-c", "import sys; sys.exit('no real uv in tests')"])


@pytest.fixture(autouse=True)
def daemons_only_over_the_tests_own_folders(monkeypatch, tmp_path_factory):
    """Refuses to start a daemon over any folder that is not under pytest's base temp folder (the owner's `~/.multimem` never is)."""
    from mmsb.systems.mm.daemon import Daemon

    base = os.path.normcase(os.path.abspath(tmp_path_factory.getbasetemp().parent))
    start = Daemon.start

    def guarded(self, ready: bool = True):
        home = os.path.normcase(os.path.abspath(self.home))
        if not home.startswith(base + os.sep):
            raise AssertionError(f"a test may start a daemon only over a folder of pytest's temp folder {base}, not {self.home}")
        return start(self, ready)

    monkeypatch.setattr(Daemon, "start", guarded)


@pytest.fixture(scope="session")
def week(tmp_path_factory):
    """The tiny V7 week of seed 7 built once for a worker's session (about 35 s): `(recipe, dump, questions)`. It is only read."""
    import fakes

    return fakes.tiny_week(tmp_path_factory.mktemp("week"))


@pytest.fixture(scope="session")
def store(week, tmp_path_factory):
    """That week ingested once by the system into a store (mm's snapshot home and its windows table): copied by every session opened over it."""
    import fakes

    _, dump, _ = week
    return fakes.ingested(dump, tmp_path_factory.mktemp("store") / "store")
