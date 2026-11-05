"""`mmmb repeat` of the recorded pilot, through the real registry and the real `Mm` (plan 8.1): the dry run that Dan reads before the 3-hour re-run
prints the 110 execs with the recorded agents, models, efforts and configurations, and starts and makes nothing; and a repeat on a machine where
mm cannot run (no torch) is refused before it builds, not after. The expected values are the recorded results' own, read here from their
files. Nothing is built, started or loaded.

The recorded pilot is the converted run, recorded/pilot-s1/converted: the folder recorded/pilot-s1 holds the file as the old code wrote it,
which names no scenario and which `mmmb` does not read. Its plan is local time, so this skips outside the pilot's time zone (v7_recorded.py)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("multimem")

from v7_recorded import needs_the_recorded_zone  # noqa: E402

from mmmb import cli, scenarios  # noqa: E402
from mmmb.core import paths  # noqa: E402
from mmmb.core.repeat import repeat_of, run_argv  # noqa: E402
from mmmb.core.runfolder import RunFolder  # noqa: E402

needs_the_recorded_zone()

RECORDED = Path(__file__).parent.parent / "oracles" / "recorded" / "pilot-s1"
CONVERTED = RECORDED / "converted"
RAW = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((CONVERTED / "results").rglob("*.json"))]


def no_agents():
    pytest.fail("an agent was asked for: a dry run, and a repeat that is refused, ask for none")


@pytest.fixture
def pilot(tmp_path, monkeypatch) -> Path:
    """The converted pilot copied under its recorded name, and a runs folder with no other run in it, so that the only history is the pilot's."""
    monkeypatch.setattr(paths, "RUNS", tmp_path / "no-runs")
    shutil.copytree(CONVERTED, tmp_path / "20261104-0314-pilot-s1")
    return tmp_path / "20261104-0314-pilot-s1"


def tree(root: Path) -> set[Path]:
    return {p for p in root.rglob("*")}


def test_the_dry_run_of_the_recorded_pilot_prints_110_execs_with_the_recorded_agents_models_efforts_and_configs_and_starts_nothing(pilot, tmp_path, capsys):
    before = tree(tmp_path)
    out_dir, temp_dir = tmp_path / "runs", tmp_path / "temp"
    assert cli.main(["repeat", str(pilot), "--out", str(out_dir), "--temp", str(temp_dir), "--cache", str(tmp_path / "cache"), "--dry-run"],
                    agents=no_agents) == 0
    out = capsys.readouterr().out
    assert tree(tmp_path) == before and not out_dir.exists() and not temp_dir.exists(), "no folder, no file, no change in the recorded run"
    lines = [" ".join(line.split()) for line in out.splitlines()]
    assert "repeat of 20261104-0314-pilot-s1: v7_screen, pilot, seed 1, 3 days; 31 questions, 110 execs" in lines
    assert "claude: claude-sonnet-5-5, at the agent's own effort (its results hold none)" in lines
    assert "codex: gpt-6.1-terra, at medium effort" in lines
    assert "configurations: files, mm/d1-3B" in lines
    for agent in ("claude", "codex"):
        assert any(line.startswith(f"{agent} files 31 to run of 31 median ") for line in lines), agent
        assert any(line.startswith(f"{agent} mm/d1-3B 24 to run of 24 (1 at once) median ") for line in lines), "d1-3B needs the GPU: one exec at a time"
    assert "110 execs to run over 3 jobs (a system may allow fewer), from the earlier runs of these cells:" in lines
    # the means are what the recorded run took in all: the sums of its results' own seconds and GPU seconds, which are plan 8.1's 13,044 and 3,655
    seconds, gpu = sum(r["seconds"] for r in RAW), sum(r["work"]["seconds"] for r in RAW)
    assert (round(seconds), round(gpu)) == (13044, 3655)
    mean = next(line for line in lines if line.startswith("at the means"))
    assert f"{seconds:,.0f} session seconds, {gpu:,.0f} GPU seconds" in mean
    # (the means count the three timeouts that report no tokens or dollars at their cell's mean, so these are a little over the plan's $19.00 and 50.3M)
    assert "$19.76 at list" in mean and "51.8M tokens" in mean
    assert any(line.startswith("at the medians") for line in lines) and "dump: written in the temp folder from the recipe (v7_screen, pilot" in out


def test_the_folder_as_the_old_code_wrote_it_is_not_a_run_mmmb_reads_and_the_converted_one_beside_it_is(tmp_path, capsys):
    assert cli.main(["repeat", str(RECORDED), "--dry-run"], agents=no_agents) == 1
    assert "was made before mmmb (its run.json names no scenario)" in capsys.readouterr().err
    assert cli.main(["compare", str(RECORDED), str(CONVERTED)], agents=no_agents) == 1 and "was made before mmmb" in capsys.readouterr().err


def test_the_repeat_runs_the_recorded_agents_at_their_settings_over_the_recorded_questions_of_each_cell(pilot):
    repeat = repeat_of(RunFolder.open(pilot, scenarios.get))
    assert {a: dict(s) for a, s in repeat.agents.items()} == {"claude": {"model": "claude-sonnet-5-5", "effort": None},
                                                              "codex": {"model": "gpt-6.1-terra", "effort": "medium"}}
    assert [c.name for c in repeat.configs] == ["files", "mm/d1-3B"]
    assert {key: len(qids) for key, qids in repeat.cells.items()} == {("claude", "files"): 31, ("claude", "mm/d1-3B"): 24, ("codex", "files"): 31,
                                                                       ("codex", "mm/d1-3B"): 24}
    argv = run_argv(Path("new-run"), repeat, 3)
    assert argv[:2] == ["run", "new-run"] and len(argv[argv.index("--only") + 1].split(",")) == 31
    assert argv[-6:] == ["--claude-model", "claude-sonnet-5-5", "--codex-model", "gpt-6.1-terra", "--codex-effort", "medium"]


def test_a_repeat_on_a_machine_where_mm_cannot_run_is_refused_before_it_builds_anything(pilot, tmp_path, capsys, monkeypatch):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed here: mm may be able to run, and this repeat would then build")

    def building(*args, **kwargs):
        pytest.fail("the dump of the pilot was built (11 minutes of the laptop): the repeat should have been refused first")

    monkeypatch.setattr(cli, "build_run", building)
    before = tree(tmp_path)
    assert cli.main(["repeat", str(pilot), "--out", str(tmp_path / "runs"), "--temp", str(tmp_path / "temp"), "--cache", str(tmp_path / "cache")],
                    agents=no_agents) == 1
    err = capsys.readouterr().err
    assert "error: mm/d1-3B cannot run now: d1-3B: " in err and ("not downloaded" in err or "its env is not ready: torch is not installed" in err)
    assert tree(tmp_path) == before, "nothing was built: not the dump, not the run, not the temp folder"
