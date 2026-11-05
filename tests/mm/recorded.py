"""What the mm tests compare with, from the recorded runs (read only): the pilot's 48 mm results and the dump they were asked of, rebuilt from
its recorded manifest (tests/oracles/recorded/pilot-s1) and read in place where its data is (D:/mm-bench-temp), and the recorded EgoLife run's 50.
`pilot-windows.json` is the pilot's table of windows (mmmb/systems/mm/estimate.py) that `capture_windows.py` made once from that data; the test that
reads the data asserts the system still makes it (test_recorded_estimate.py)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from mmmb.core.contract import Dump
from mmmb.core.question import Question
from mmmb.scenarios.v7_screen.manifest import LOGS, entries_of
from mmmb.scenarios.v7_screen.questions import questions_of
from mmmb.scenarios.v7_screen.week import plan as make_plan

ORACLES = Path(__file__).parent.parent / "oracles" / "recorded"
PILOT, EGOLIFE = ORACLES / "pilot-s1", ORACLES / "egolife-s1"
CHECKED_IN = Path(__file__).parent / "tables"
PILOT_WINDOWS = CHECKED_IN / "pilot-windows.json"
PILOT_DATA = Path(r"D:\mm-bench-temp\mmb-de178c4d7f-data")  # the pilot's week as it was written, frozen
EGOLIFE_RUN = Path(r"D:\mm-bench\20261103-0257-egolife-s1")  # the recorded EgoLife run: its data folder holds the transcript
DAYS = [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]


def pilot_dump(root: Path) -> tuple[Dump, tuple[Question, ...]]:
    """The pilot's dump: its plan's entries with the sizes and hashes the pilot recorded, over the data at `root`; and its questions."""
    manifest = json.loads((PILOT / "data-manifest.json").read_text(encoding="utf-8"))
    sha256 = json.loads((PILOT / "sha256.json").read_text(encoding="utf-8"))
    plan = make_plan(1, "pilot", DAYS)
    entries = entries_of(plan, lambda path: (manifest[path][0], manifest[path][1], sha256[path]))
    return Dump("k" * 64, root, "v7_screen", entries, (LOGS,)), questions_of(plan, entries)


def pilot_results() -> list[dict]:
    """The pilot's mm results (24 of each agent), as converted (tests/oracles/convert_recorded.py)."""
    found = sorted((PILOT / "converted" / "results").glob("*/mm-d1-3B/*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in found]
