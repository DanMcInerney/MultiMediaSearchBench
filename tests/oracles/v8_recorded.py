"""What the V8 oracles (1: test_v8_plan.py, 2: test_recorded_egolife.py) compare with: the recorded EgoLife run's own files, captured before the
code was moved (tests/oracles/recorded/egolife-s1: run.json, report.json and report.md by tests/oracles/capture_recorded.py; the converted
run, with its recipe, questions and results, and the prompts by tests/oracles/convert_recorded.py). Nothing here is computed by the code
under test. The plan of the run is made from the download at D:\\bench-data\\egolife, read only, and its times are local: the oracles of the
plan hold only where the download is and in the time zone the run was made in, and elsewhere they skip, and say why."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from mmsb import scenarios

RECORDED = Path(__file__).parent / "recorded" / "egolife-s1"
CONVERTED = RECORDED / "converted"
RUN = Path(r"D:\mm-bench\20261103-0257-egolife-s1")  # the recorded run, where it is: only some tests read it
META = json.loads((RECORDED / "run.json").read_text(encoding="utf-8"))  # plan, files, pools, generator, seed, days, per_age, speech
REPORT = json.loads((RECORDED / "report.json").read_text(encoding="utf-8"))
RECIPE = json.loads((CONVERTED / "run.json").read_text(encoding="utf-8"))["recipe"]  # the recorded clips.json is its clip table
V8 = scenarios.get("v8_egolife")


def download_skip() -> str | None:
    """Why the plan cannot be made on this machine, or None."""
    source = Path(RECIPE["source"])
    if not source.is_dir():
        return (f"the EgoLife download is not at {source}: the plan of the recorded run is made from its questions and transcripts, so "
                "its hashes cannot be checked without it")
    return None


def zone_skip() -> str | None:
    """Why the plan would not be the recorded run's here, or None."""
    here = V8.utc_offset_text(date.fromisoformat(RECIPE["days"][0]))
    if here != RECIPE["utc_offset"]:
        return (f"the recorded run's days were made where the UTC offset was {RECIPE['utc_offset']}; here it is {here}. Its times are local, so "
                "a plan here is another run and its hashes would differ: run this oracle in US Eastern time")
    return None
