"""New tests for mmsb/core/probe.py: the shortcut probe in a run. The probe itself is the scenario's (tests/test_v7_probe.py and
tests/test_v7_scenario.py); here a stand-in scenario says what it is asked and what it found. Expected values are the sentences of the
replaced `probe_line` of bench.py (the wording a run's report and commands print) and what run.json records."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mmsb.core import fs
from mmsb.core.contract import Dump
from mmsb.core.probe import probe_line, probe_of, run_probe
from mmsb.core.runfolder import RunFolder
from mmsb.core.scenario import ProbeOutcome


class Probing:
    """A scenario whose probe finds what it is told and remembers what it was asked."""

    name, code, generator = "stub", "S", 1

    def __init__(self, failures: tuple[str, ...] = ()) -> None:
        self.failures, self.asked = failures, []

    def probe(self, recipe, dump, out, ffmpeg, seed):
        self.asked.append((recipe, dump, out, ffmpeg, seed))
        return ProbeOutcome("failed" if self.failures else "passed", self.failures)


def run_of(tmp_path: Path, scenario) -> RunFolder:
    return RunFolder(tmp_path / "run", 4, "tiny", (date(2026, 10, 30),), tmp_path / "data", tmp_path, "tag", 900.0, scenario, ())


def test_the_scenarios_probe_is_asked_over_the_dump_with_the_files_arms_ffmpeg_and_its_outcome_is_what_run_json_records(tmp_path):
    scenario = Probing()
    run, dump = run_of(tmp_path, scenario), Dump("k", tmp_path / "data", "stub", (), ())
    assert run_probe(run, {"seed": 4}, dump) == {"status": "passed", "failures": []}
    assert scenario.asked == [({"seed": 4}, dump, run.folder, run.bin_exe, 4)]
    failing = Probing(("a: a filter finds it", "b: its file is the largest"))
    assert run_probe(run_of(tmp_path, failing), {}, dump) == {"status": "failed", "failures": ["a: a filter finds it", "b: its file is the largest"]}
    json.dumps(run_probe(run, {}, dump))  # (and it is JSON)


def test_a_probe_asked_to_be_skipped_is_not_asked_and_is_recorded_as_skipped(tmp_path):
    scenario = Probing(("never",))
    assert run_probe(run_of(tmp_path, scenario), {}, Dump("k", tmp_path, "stub", (), ()), probe=False) == {"status": "skipped", "failures": []}
    assert scenario.asked == []


def test_the_outcome_a_run_recorded_is_read_back_and_a_run_from_before_the_probe_has_none(tmp_path):
    fs.write_json(tmp_path / "with" / "run.json", {"seed": 1, "probe": {"status": "failed", "failures": ["x"]}})
    fs.write_json(tmp_path / "without" / "run.json", {"seed": 1})
    assert probe_of(tmp_path / "with") == {"status": "failed", "failures": ["x"]} and probe_of(tmp_path / "without") is None
    with pytest.raises(FileNotFoundError):
        probe_of(tmp_path / "none")


def test_a_run_whose_probe_did_not_pass_is_told_apart_from_one_that_did_in_the_words_the_report_and_the_commands_print():
    assert probe_line("r1", {"status": "passed", "failures": []}) is None
    assert probe_line("r1", None) == ("r1 has no shortcut probe record (it was built before the probe), so nothing says a shortcut cannot "
                                      "find its needles")
    assert probe_line("r1", {"status": "skipped", "failures": []}) == ("r1 was built with --no-probe: nothing says a colour filter, frame "
                                                                        "difference, band filter or file size cannot find its needles")
    assert probe_line("r1", {"status": "failed", "failures": ["a", "b"]}) == (
        "r1 failed the shortcut probe (2 failures, in its shortcuts.md): a shortcut may find its needles, so its files arm may be doing less "
        "work than an ask")
