"""mm and the working clock (plan 6.5, 8.2, decision D3; written for this repository, U14): Multimem keeps no per-job wait, so neither of mm's sessions
can say how long a case queued for the GPU, and the clock runs on the wall. That is exact at one exec at a time and wrong at several, so the core
refuses several on the working clock (`mmmb run`, before any agent starts) and `mmmb clock-check` refuses mm altogether until Multimem reports the
queue. Over the real system, the tiny week's real store and the fake decider; no agent, no model and no daemon of the owner's.

The expected values are the plan's words: a session that cannot say its queue says None, the preview as the opened one."""

from __future__ import annotations

import dataclasses
import shutil
from datetime import datetime

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from fakes import MM, Opened, context  # noqa: E402

from mmmb import cli, scenarios  # noqa: E402
from mmmb.core import dryrun, fs  # noqa: E402
from mmmb.core.agents import claude  # noqa: E402
from mmmb.core.build import INGESTED, write_dump  # noqa: E402
from mmmb.core.clock import WALL, WORKING, check_queue  # noqa: E402
from mmmb.core.clockcheck import check_possible  # noqa: E402
from mmmb.core.config import Config  # noqa: E402
from mmmb.core.contract import Case  # noqa: E402
from mmmb.core.question import write_questions  # noqa: E402
from mmmb.core.runfolder import RunFolder  # noqa: E402
from mmmb.systems.mm import home as mm_home  # noqa: E402
from mmmb.systems.mm import models  # noqa: E402
from mmmb.systems.mm.daemon import Daemon  # noqa: E402

pytestmark = pytest.mark.xdist_group("mm-arm")
D1 = Config("mm/d1-3B", "mm", "d1-3B", False)


def case_of(q, work, run_dir) -> Case:
    return Case(q.qid, datetime.fromisoformat(q.now).timestamp(), q.horizon, q.scope.sources, q.scope.since, q.scope.until, q.group, 3.0, 0, work, run_dir)


def test_neither_session_of_mm_can_say_its_queue_and_more_than_one_exec_at_once_on_the_working_clock_is_refused_for_both(week, store, tmp_path):
    _, dump, questions = week
    cases = [case_of(q, tmp_path / "work", tmp_path / "run" / q.qid) for q in questions]
    preview = MM.preview(store, dump, context(tmp_path / "homes"))
    with Opened(store, dump, tmp_path / "homes") as opened:
        for session in (preview, opened):
            assert session.queue_s(cases[0]) is None, type(session).__name__
            assert session.max_jobs(cases, 3) == 3, "the fake decider is fast: three at once would run"
            check_queue("mm/fake", session, cases[0], 1, WORKING)
            check_queue("mm/fake", session, cases[0], 3, WALL)
            with pytest.raises(ValueError, match=r"mm/fake: 3 execs at once on the working clock.*no queue time.*--clock wall"):
                check_queue("mm/fake", session, cases[0], 3, WORKING)
            with pytest.raises(ValueError, match="decision D3"):
                check_possible(D1, session, cases, 3)
        assert opened.usage(cases[0], []).queue_s is None, "and it has none to report as the cost either"


def built(week, store, tmp_path) -> RunFolder:
    """A run folder over the tiny week and its real store, with `mm/d1-3B` among its configurations (its model need not be installed to be previewed)."""
    recipe, dump, questions = week
    folder = tmp_path / "run"
    folder.mkdir()
    write_questions(folder / "questions.jsonl", questions)
    fs.write_json(folder / "data-manifest.json", fs.stats_of(dump.root))
    write_dump(folder, dump)
    shutil.copytree(store, folder / "stores" / "mm")
    fs.write_json(folder / "stores" / "mm" / INGESTED, {**dataclasses.asdict(fakes.REPORTS[str(store)]), "store_hash": MM.describe("fake").store_hash})
    return RunFolder.write(folder, scenarios.get("v7_screen"), recipe, dump, tmp_path / "temp", "t7", 900.0, [D1], {"status": "skipped", "failures": []},
                           cache=tmp_path / "cache")


def test_the_commands_refuse_mm_at_several_execs_on_the_working_clock_before_anything_starts_and_clock_check_refuses_it_naming_d3(
        week, store, tmp_path, monkeypatch, capsys):
    run = built(week, store, tmp_path)
    monkeypatch.setattr(models.modelenv, "installed", lambda: {})  # no torch: a model that is not ready is not the refusal looked for
    monkeypatch.setattr(dryrun, "shutil", type("NoPath", (), {"which": staticmethod(lambda name: None)}))
    monkeypatch.setattr(Daemon, "start", lambda self, ready=True: pytest.fail("a refusal starts no daemon"))
    monkeypatch.setattr(mm_home, "copy_home", lambda *args: pytest.fail("a refusal copies no store"))
    monkeypatch.setattr(claude, "check_login", lambda: pytest.fail("a refusal logs nothing in"))

    def command(*argv: str) -> tuple[int, str]:
        status = cli.main([*argv, str(run.folder)] if argv[0] == "clock-check" else [argv[0], str(run.folder), *argv[1:]])
        return status, capsys.readouterr().err

    for flags in ((), ("--dry-run",)):  # a dry run says so too
        status, err = command("run", "--agents", "claude", "--configs", "mm/d1-3B", "--jobs", "3", *flags)
        assert status == 1 and "mm/d1-3B: 3 execs at once on the working clock" in err and "--clock wall" in err
    status, err = command("clock-check", "--config", "mm/d1-3B", "--agents", "claude", "--jobs", "3")
    assert status == 1 and "mm/d1-3B's system reports no queue time" in err and "decision D3" in err
    assert not (run.folder / "clock-check").exists() and not (run.folder / "results").exists()
    # at one exec at a time the wall is the working time, and on the wall clock the number does not matter: the dry runs go through
    for flags in (("--jobs", "1"), ("--jobs", "3", "--clock", "wall")):
        status, err = command("run", "--dry-run", "--agents", "claude", "--configs", "mm/d1-3B", *flags)
        assert (status, err) == (0, ""), flags
    status, err = command("clock-check", "--config", "mm/d1-3B", "--agents", "claude", "--jobs", "1")
    assert status == 1 and "would run these cases 1 at a time, not 1, so nothing would queue" in err, "one at a time queues for nothing: no check"
