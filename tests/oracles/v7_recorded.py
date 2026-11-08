"""What the V7 oracles (1a, 1b, 1c) compare with: the recorded pilot's own files, captured in U0 (tests/oracles/recorded/pilot-s1). Nothing
here is computed by the code under test. The pilot's week is the plan's times, which are local, so these oracles hold only in the time
zone it was made in: elsewhere they skip, and the reason says which offsets they need."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmsb import scenarios

RECORDED = Path(__file__).parent / "recorded" / "pilot-s1"
META = json.loads((RECORDED / "run.json").read_text(encoding="utf-8"))  # plan, files, probe, seed, scale, days
MANIFEST = json.loads((RECORDED / "data-manifest.json").read_text(encoding="utf-8"))  # path -> [size, mtime_ns], 1,575 of them
SHA256 = json.loads((RECORDED / "sha256.json").read_text(encoding="utf-8"))  # path -> sha256 of the file the pilot was run on
# the offsets from UTC at local midnight on each of the pilot's days and after the last: daylight saving ended on 1 November 2026 at 02:00
OFFSETS = ["-0400", "-0500", "-0500", "-0500"]
V7 = scenarios.get("v7_screen")


def recipe() -> dict:
    return V7.recipe(scale=META["scale"], seed=META["seed"], days=META["days"])


def zone_skip() -> str | None:
    """Why these oracles cannot run on this machine, or None."""
    here = recipe()["utc_offsets"]
    if here != OFFSETS:
        return (f"the pilot's week was made where the UTC offsets of its days were {OFFSETS}; here they are {here}. Its times are local, so a "
                "build here is another week and its hashes and bytes would differ: run this oracle in US Eastern time")
    return None


def needs_the_recorded_zone():
    reason = zone_skip()
    if reason:
        pytest.skip(reason, allow_module_level=True)
