"""Writes tests/mm/tables/pilot-windows.json, once: the table of windows (`table_of`) of the recorded pilot's dump, counted from the data it was
run on (D:/mm-bench-temp/mmb-de178c4d7f-data, read only; about 30 s). The recorded gpu_estimate_s of the pilot's results are what the table is
held to (test_recorded_estimate.py), not this file's output. Run from the repository root: `.venv/Scripts/python.exe tests/mm/capture_windows.py`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recorded import PILOT_DATA, PILOT_WINDOWS, pilot_dump  # noqa: E402

from mmsb.systems.mm.estimate import table_of  # noqa: E402

if __name__ == "__main__":
    dump, _ = pilot_dump(PILOT_DATA)
    PILOT_WINDOWS.parent.mkdir(exist_ok=True)
    PILOT_WINDOWS.write_text(json.dumps(table_of(dump), separators=(",", ":")), encoding="utf-8")
    print(f"wrote {PILOT_WINDOWS} ({PILOT_WINDOWS.stat().st_size} bytes)")
