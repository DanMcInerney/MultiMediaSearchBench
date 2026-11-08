"""mm and the working clock (plan 6.5, 8.2, decision D3; written for this repository, U14): Multimem keeps no per-job wait, so neither of mm's sessions
can say how long a case queued for the GPU, and the clock runs on the wall. That is exact at one exec at a time and wrong at several, so the core
refuses several on the working clock (`mmsb run`, before any agent starts) and `mmsb clock-check` refuses mm altogether until Multimem reports the
queue. Over the real system, the tiny week's real store and the fake decider; no agent, no model and no daemon of the owner's.

mm needs the GPU, so it runs one exec at a time unless `--gpu-jobs` asks for more (core/atonce.py): the defaults are never refused, `--gpu-jobs 2`
is refused on the working clock, and nothing a refusal says offers the wall clock as the way to keep the several.

The expected values are the plan's words: a session that cannot say its queue says None, the preview as the opened one."""

from __future__ import annotations

import dataclasses
import re
import shutil
from datetime import datetime

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from fakes import MM, Opened, context  # noqa: E402

from mmsb import cli, scenarios  # noqa: E402
from mmsb.core import dryrun, fs  # noqa: E402
from mmsb.core.agents import claude  # noqa: E402
from mmsb.core.build import INGESTED, write_dump  # noqa: E402
from mmsb.core.clock import WALL, WORKING, check_queue  # noqa: E402
from mmsb.core.clockcheck import queued_at_once  # noqa: E402
from mmsb.core.config import Config  # noqa: E402
from mmsb.core.contract import Case  # noqa: E402
from mmsb.core.question import write_questions  # noqa: E402
from mmsb.core.runfolder import RunFolder  # noqa: E402
from mmsb.systems.mm import home as mm_home  # noqa: E402
from mmsb.systems.mm import models  # noqa: E402
from mmsb.systems.mm.daemon import Daemon  # noqa: E402
from mmsb.systems.mm.system import Mm  # noqa: E402

pytestmark = pytest.mark.xdist_group("mm-arm")
D1 = Config("mm/d1-3B", "mm", "d1-3B", False)


def case_of(q, work, run_dir) -> Case:
    return Case(q.qid, datetime.fromisoformat(q.now).timestamp(), q.horizon, q.scope.sources, q.scope.since, q.scope.until, q.group, 3.0, 0, work, run_dir)


def test_neither_session_of_mm_can_say_its_queue_and_more_than_one_exec_at_once_on_the_working_clock_is_refused_for_both(week, store, tmp_path):
    _, dump, questions = week
    cases = [case_of(q, tmp_path / "work", tmp_path / "run" / q.qid) for q in questions]
    preview = MM.preview(store, dump, context(tmp_path / "homes"))
    fake, d1 = MM.describe("fake"), Mm().describe("d1-3B")  # (the real d1-3B's facts: the tests' MM stands the fake decider in its place)
    assert not fake.needs_gpu and d1.needs_gpu
    with Opened(store, dump, tmp_path / "homes") as opened:
        for session in (preview, opened):
            assert session.queue_s(cases[0]) is None, type(session).__name__
            assert session.max_jobs(cases, 3) == 3, "the fake decider is fast: three at once would run"
            check_queue("mm/fake", fake, session, cases[0], 1, WORKING)
            check_queue("mm/fake", fake, session, cases[0], 3, WALL)
            with pytest.raises(ValueError, match=r"mm/fake: 3 execs at once on the working clock.*no queue time.*one exec at a time \(--jobs 1\)") as refused:
                check_queue("mm/fake", fake, session, cases[0], 3, WORKING)
            with pytest.raises(ValueError, match=r"mm/d1-3B: 2 execs at once on the working clock.*one exec at a time \(--gpu-jobs 1\)") as gpu_refused:
                check_queue("mm/d1-3B", d1, session, cases[0], 2, WORKING)
            assert "--clock" not in str(refused.value) + str(gpu_refused.value), "no refusal offers the wall clock"
            with pytest.raises(ValueError, match="decision D3"):
                queued_at_once(D1, d1, session, cases, 3, 2)
            with pytest.raises(ValueError, match=r"one at a time \(--gpu-jobs 1\).*--gpu-jobs 2 or more"):
                queued_at_once(D1, d1, session, cases, 3, 1)
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


def test_mm_runs_one_exec_at_a_time_at_the_defaults_is_refused_at_several_on_the_working_clock_before_anything_starts_and_clock_check_needs_gpu_jobs(
        week, store, tmp_path, monkeypatch, capsys):
    run = built(week, store, tmp_path)
    monkeypatch.setattr(models.modelenv, "installed", lambda: {})  # no torch: a model that is not ready is not the refusal looked for
    monkeypatch.setattr(dryrun, "shutil", type("NoPath", (), {"which": staticmethod(lambda name: None)}))
    monkeypatch.setattr(Daemon, "start", lambda self, ready=True: pytest.fail("a refusal starts no daemon"))
    monkeypatch.setattr(mm_home, "copy_home", lambda *args: pytest.fail("a refusal copies no store"))
    monkeypatch.setattr(claude, "check_login", lambda: pytest.fail("a refusal logs nothing in"))

    def command(*argv: str) -> tuple[int, str, str]:
        status = cli.main([*argv, str(run.folder)] if argv[0] == "clock-check" else [argv[0], str(run.folder), *argv[1:]])
        out, err = capsys.readouterr()
        return status, out, err

    # several execs of mm are only what --gpu-jobs asks for, and the working clock refuses them: in a dry run too, and the remedy is one at a time
    for flags in ((), ("--dry-run",)):
        status, out, err = command("run", "--agents", "claude", "--configs", "mm/d1-3B", "--gpu-jobs", "2", *flags)
        assert status == 1 and "mm/d1-3B: 2 execs at once on the working clock" in err and "one exec at a time (--gpu-jobs 1)" in err
        assert "--clock" not in err and "warning" not in out, "no offer of the wall clock, and nothing to warn of: nothing runs"
    status, out, err = command("clock-check", "--config", "mm/d1-3B", "--agents", "claude", "--gpu-jobs", "2")
    assert status == 1 and "mm/d1-3B's system reports no queue time" in err and "decision D3" in err
    assert not (run.folder / "clock-check").exists() and not (run.folder / "results").exists()
    # at the defaults mm is one exec at a time whatever --jobs says: nothing to refuse, and the dry run says so
    for flags in ((), ("--jobs", "3"), ("--jobs", "4", "--gpu-jobs", "1")):
        status, out, err = command("run", "--dry-run", "--agents", "claude", "--configs", "mm/d1-3B", *flags)
        assert (status, err) == (0, ""), flags
        assert re.search(r"claude\s+mm/d1-3B\s+\d+ to run of \d+ \(1 at once\)", out) and "warning" not in out, flags
    # the owner may ask for the wall clock with several: it goes through, with a warning that several GPU execs run at once
    status, out, err = command("run", "--dry-run", "--agents", "claude", "--configs", "mm/d1-3B", "--gpu-jobs", "2", "--clock", "wall")
    assert (status, err) == (0, "") and "warning: mm/d1-3B: 2 execs at once on the GPU" in out and "(2 at once)" in out
    # the check needs the explicit GPU option: at the defaults the cases would run one at a time and nothing would queue
    status, out, err = command("clock-check", "--config", "mm/d1-3B", "--agents", "claude", "--jobs", "3")
    assert status == 1 and "one at a time (--gpu-jobs 1)" in err and "--gpu-jobs 2 or more" in err and "--clock" not in err
