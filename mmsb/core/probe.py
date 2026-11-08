"""The shortcut probe in a run (DESIGN 7.3): whether a cheap shortcut, a colour filter or the size of a file, can single out a needle, so
that an agent with its own tools could skip the work. The probe itself is the scenario's (`Scenario.probe`: V7's reads the specs of its
plan); the core runs it, records its outcome in run.json and says what to say of a run whose probe did not pass (moved from `run_probe`,
`probe_of` and `probe_line` of bench.py of the repository at bb498a6)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from mmsb.core import fs
from mmsb.core.contract import Dump
from mmsb.core.runfolder import RunFolder


def run_probe(run: RunFolder, recipe: Mapping[str, object], dump: Dump, probe: bool = True) -> dict:
    """The scenario's probe over the run's dump, reporting in the run folder, with the ffmpeg the agents with their own tools have
    (`RunFolder.ensure_bin` has put it there): `{"status": "passed" | "failed" | "skipped", "failures": [...]}`, as run.json records
    it. `probe=False` skips it."""
    if not probe:
        return {"status": "skipped", "failures": []}
    outcome = run.scenario.probe(recipe, dump, run.folder, run.bin_exe, run.seed)
    return {"status": outcome.status, "failures": list(outcome.failures)}


def probe_of(folder: Path) -> dict | None:
    """The probe's outcome that a run's run.json records, None for a run built before the probe."""
    return fs.read_json(Path(folder) / "run.json").get("probe")


def probe_line(name: str, outcome: dict | None) -> str | None:
    """What to say of a run whose probe did not pass, else None."""
    if outcome is None:
        return f"{name} has no shortcut probe record (it was built before the probe), so nothing says a shortcut cannot find its needles"
    if outcome["status"] == "skipped":
        return (f"{name} was built with --no-probe: nothing says a colour filter, frame difference, band filter or file size cannot find "
                "its needles")
    if outcome["status"] == "failed":
        return (f"{name} failed the shortcut probe ({len(outcome['failures'])} failures, in its shortcuts.md): a shortcut may find "
                "its needles, so its files arm may be doing less work than an ask")
    return None
