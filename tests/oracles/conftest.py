"""Options of the oracles that run something long. `--v7-full=FOLDER` is oracle 1c (test_v7_full.py): FOLDER is where the pilot's week is
built, or has been built (a folder that holds `data/` is read, not built again). Without it that oracle skips, and says so."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--v7-full", default=None, metavar="FOLDER",
                     help="oracle 1c: build the recorded pilot's week in FOLDER (or read the one built there) and compare all of it")
    parser.addoption("--v7-workers", type=int, default=3, help="oracle 1c: files written at once (the recorded build used 3)")
