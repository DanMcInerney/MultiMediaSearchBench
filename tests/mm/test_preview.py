"""New tests for what U8 added to the mm system (mmsb/systems/mm/system.py): `Mm.preview`, the session of a dry run, and the figures `ingest` measures.

`preview` is `open` without the daemon, the copy of the store and the model: the tests set it beside an opened session over the same store (the
tiny week ingested once for the worker, tests/mm/conftest.py), for the jobs, the estimate and the server and instructions an exec would get, and
look that it started nothing and wrote nothing. `measured_of` is worked out from hand-made tables."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

pytest.importorskip("mms")

import fakes  # noqa: E402
from fakes import MM, Opened, context  # noqa: E402

from mmsb.core import fs  # noqa: E402

from mmsb.core.contract import Case  # noqa: E402
from mmsb.systems.mm.system import MmPreview, measured_of  # noqa: E402


def case_of(q, work, run_dir, slot: int = 0, estimate: float = 3.0) -> Case:
    return Case(q.qid, datetime.fromisoformat(q.now).timestamp(), q.horizon, q.scope.sources, q.scope.since, q.scope.until, q.group, estimate,
                slot, work, run_dir)


def test_a_preview_answers_the_jobs_the_estimate_and_the_server_as_an_open_session_does_and_starts_nothing(week, store, tmp_path):
    _, dump, questions = week
    homes, work = tmp_path / "homes", tmp_path / "work"
    work.mkdir()
    preview = MM.preview(store, dump, context(homes))
    assert isinstance(preview, MmPreview) and not homes.exists(), "a preview copies no store"
    cases = [case_of(q, work, tmp_path / "run" / q.qid, estimate=preview.estimate(q.scope.sources, q.cost_ranges)) for q in questions]
    served = [preview.serve(c) for c in cases]
    assert not list(work.iterdir()) and not homes.exists(), "no marker is written into the working folder and no home is made"
    with Opened(store, dump, homes) as opened:
        for q, c, got in zip(questions, cases, served):
            assert preview.estimate(q.scope.sources, q.cost_ranges) == opened.estimate(q.scope.sources, q.cost_ranges)
            assert c.estimate_s < 300.0, "the cases of the tiny week are light: serving them waits for no idle daemon"
            assert opened.serve(c) == got, q.qid
        assert preview.max_jobs(cases, 3) == opened.max_jobs(cases, 3) == 3
        horizon = dataclasses.replace(cases[0], horizon=cases[0].now - 100.0)
        assert preview.max_jobs([horizon], 3) == opened.max_jobs([horizon], 3) == 1, "a cut to a horizon is the whole home's: one exec at a time"
    first = served[0]
    assert first.server.name == "mm" and first.server.args[first.server.args.index("--home") + 1] == str(homes / "claude-fake")
    assert first.holder == "mm" and first.record == {"gpu_estimate_s": round(cases[0].estimate_s, 1), "exclusive": False,
                                                     "beyond_timeout": False, "busy_at_start": None}


def test_a_preview_of_a_case_that_holds_the_gpu_says_so_without_waiting_for_an_idle_daemon(week, store, tmp_path):
    _, dump, questions = week
    preview = MM.preview(store, dump, context(tmp_path / "homes", timeout_s=900.0))
    served = preview.serve(case_of(questions[0], tmp_path / "work", tmp_path / "run", estimate=988.0))
    assert served.record == {"gpu_estimate_s": 988.0, "exclusive": True, "beyond_timeout": True, "busy_at_start": None}
    assert not (tmp_path / "homes").exists() and not (tmp_path / "work").exists(), "no daemon was asked and no file was written"


def test_a_preview_of_a_variant_that_is_not_known_or_a_store_that_is_not_there_is_an_error(week, tmp_path):
    _, dump, _ = week
    with pytest.raises(ValueError, match="mm has variants"):
        MM.preview(tmp_path, dump, context(tmp_path / "homes", variant=None))
    with pytest.raises(FileNotFoundError):
        MM.preview(tmp_path / "no-store", dump, context(tmp_path / "homes"))


def test_ingest_measures_the_decode_of_the_one_source_whose_pictures_are_large_and_nothing_else(store):
    assert measured_of({"decode_s": {}}) == {} and measured_of({"decode_s": {"glasses": 0.024}}) == {"decode_s": 0.024}
    assert measured_of({"decode_s": {"glasses": 0.024, "screen": 0.05}}) == {}, "which one is meant?"
    assert fakes.REPORTS[str(store)].measured == measured_of(fs.read_json(store / "windows.json")) == {}, "the tiny week has no large picture"
