"""The slow half of the neutrality check (plan 7.6; tests/test_neutral.py runs it): run by the python of a fresh environment made with no extra
and no group (`uv sync --frozen --no-default-groups`), so MultimediaSearch is not installed (and pytest is not either). It builds the tiny week of V7
with the command, runs the shortcut probe, runs a `files` cell with a stand-in agent, writes a report, and sees that a configuration of the
system that needs the extra is refused with the extra's name.

    <env>/Scripts/python.exe -I tests/oracles/neutral_smoke.py <folder for the run and the temp files>

It prints a line of JSON for the test to check, and `neutral smoke ok` last. Nothing here starts a real agent: the stand-in is
tests/stub_system.py's, which answers the questions of the tiny week of seed 7 over 30 and 31 October 2026 (the days built here)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

assert importlib.util.find_spec("mms") is None, "mms is installed in the environment this smoke test is for"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/: the stand-in agent

from stub_system import Stub, tiny_questions  # noqa: E402

from mmsb import cli, systems  # noqa: E402
from mmsb.core.agents import claude  # noqa: E402
from mmsb.core.runner import Agent  # noqa: E402

DAYS = "2026-10-30,2026-10-31"
SEED = 7


def main(folder: Path) -> dict:
    claude.check_login = lambda: None  # (no login, and no claude, is needed: the agent is a stand-in)
    runs, temp, cache = folder / "runs", folder / "temp", folder / "cache"
    # build: the week, its questions, the probe over it, and run.json last
    built = io.StringIO()
    with contextlib.redirect_stdout(built):
        status = cli.main(["build", "--scenario", "v7_screen", "--scale", "tiny", "--seed", str(SEED), "--days", DAYS, "--out", str(runs),
                           "--temp", str(temp), "--cache", str(cache), "--configs", "files"])
    assert status == 0, built.getvalue()
    (run,) = runs.iterdir()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["probe"]["status"] == "passed" and meta["probe"]["failures"] == [], meta["probe"]
    assert (run / "shortcuts.json").is_file() and (run / "questions.jsonl").is_file() and (run / "dump.json").is_file()
    # a files cell with a stand-in agent over every question
    assert cli.main(["run", str(run), "--agents", "claude", "--configs", "files", "--answer-format", "line"], agents=lambda: {"claude": Agent(Stub())}) == 0
    results = sorted((run / "results" / "claude" / "files").glob("*.json"))
    assert len(results) == len(tiny_questions()) > 0
    # a report
    assert cli.main(["report", str(run)]) == 0
    figures = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert (run / "report.md").read_text(encoding="utf-8").startswith("# ") and figures["runs"][0]["probe"]["status"] == "passed"
    # the system that needs the extra is refused with the extra's name, and nothing runs for it
    refused = io.StringIO()
    with contextlib.redirect_stderr(refused):
        assert cli.main(["run", str(run), "--agents", "claude", "--configs", "mm", "--dry-run"]) == 1
    assert "mm needs the mm extra: uv sync --extra mm --group mm-gpu" in refused.getvalue(), refused.getvalue()
    assert set(systems.infos()) >= {"files", "mm"} and systems.isolation().deny_package_roots == ("mms",)
    return {"mms": None, "probe": meta["probe"]["status"], "questions": len(tiny_questions()), "results": len(results),
            "report_runs": len(figures["runs"]), "refused": refused.getvalue().strip().splitlines()[-1]}


if __name__ == "__main__":
    summary = main(Path(sys.argv[1]))
    print(json.dumps(summary))
    print("neutral smoke ok")
