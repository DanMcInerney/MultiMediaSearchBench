"""New tests (G1's repair of F5, F11's a15 and F3's provenance of mm): what keeps one exec's state out of the next one's, through the real session and a
real daemon with the fake decider and nothing patched, and the rules by which a session holds a case back, called directly.

A fake daemon answers in milliseconds, so the real run cannot show a late label or an interrupted exec's label that the system forgot to take out; here
the label is put where a slow job or an interrupted run would put it, and the call that must take it out is the one asked.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from fakes import FAKE_MODEL, Opened, context  # noqa: E402
from test_mm_arm import NoDaemon, labels_left  # noqa: E402

from mmsb.core.contract import Case  # noqa: E402
from mmsb.systems.mm.daemon import Api  # noqa: E402
from mmsb.systems.mm.estimate import HEAVY_S  # noqa: E402
from mmsb.systems.mm.home import exec_project  # noqa: E402
from mmsb.systems.mm.system import MmSession  # noqa: E402

pytestmark = pytest.mark.xdist_group("mm-arm")


def label_as(home: Path, qid: str, q) -> None:
    """What a job of the exec `qid` does when it ends: labels what it found, in the exec's project."""
    Api(home).call("label", text="late", since=q.scope.since, until=q.scope.until, source=q.scope.sources[0], caller={"project": exec_project(qid)})


def case_in(q, tmp_path: Path, slot: int = 0) -> Case:
    work = tmp_path / f"w{slot}-{q.qid}"
    work.mkdir(parents=True, exist_ok=True)
    return Case(q.qid, 1.0, None, q.scope.sources, q.scope.since, q.scope.until, q.group, 3.0, slot, work, tmp_path / "run" / q.qid)


def test_a_home_a_run_that_stopped_left_labels_in_is_swept_when_the_next_session_opens_it_with_nothing_patched(week, store, tmp_path):
    _, dump, questions = week
    q = questions[0]
    homes = tmp_path / "homes"
    with Opened(store, dump, homes) as first:
        home = first.home
        label_as(home, "under-way", q)  # the exec that was running when the run stopped on quota
        label_as(home, "other", q)
        assert labels_left(home)[0] >= 2
    assert labels_left(home)[0] >= 2, "closing a session does not take it out: the next open does"
    with Opened(store, dump, homes) as second:
        assert second.home == home and labels_left(home) == (0, 0, 0)


def test_a_label_a_job_makes_after_its_exec_was_reset_is_gone_when_the_next_exec_is_served_and_one_of_an_exec_still_running_is_not(week, store, tmp_path):
    _, dump, questions = week
    a, b, c = questions[:3]
    with Opened(store, dump, tmp_path / "homes") as session:
        home = session.home
        served = case_in(a, tmp_path)
        session.serve(served)
        session.reset(served, [], 0)  # a's exec ended, and its jobs were cancelled: a late one labels after the reset's scrub
        label_as(home, a.qid, a)
        assert labels_left(home)[0] == 1
        session.serve(case_in(b, tmp_path, 1))  # the next exec starts
        assert labels_left(home) == (0, 0, 0), "a's late label did not reach b"
        label_as(home, b.qid, b)  # b is running: its own label is its own
        session.serve(case_in(c, tmp_path, 2))
        assert labels_left(home)[0] == 1, "the project of an exec that has not ended is left alone"


def test_a_session_closed_after_a_late_label_lists_only_the_labels_no_project_owns_and_stops_its_daemon(week, store, tmp_path):
    _, dump, questions = week
    q = questions[0]
    session = fakes.MM.open(store, dump, context(tmp_path / "homes"))
    served = case_in(q, tmp_path)
    session.serve(served)
    session.reset(served, [], 0)
    label_as(session.home, q.qid, q)
    assert session.close() == {"ambient-labels": []}
    assert not (session.home / "daemon.json").exists()


# -- the rules by which a session holds a case back --------------------------------------------------------------------------------

def session_of(tmp_path: Path) -> MmSession:
    """A session that has no home and no daemon (the rules ask neither), with the two sources of file versions."""
    return MmSession(FAKE_MODEL, tmp_path / "no-home", NoDaemon(), set(), {"sources": {}, "decode_s": {}}, frozenset({"notes", "config"}), (),
                     context(tmp_path / "homes"))


def test_a_case_is_held_back_beside_one_that_holds_the_gpu_and_a_case_that_holds_it_starts_alone_and_file_versions_run_one_at_a_time(tmp_path):
    session = session_of(tmp_path)

    def case(source: str, estimate: float, qid: str) -> Case:
        return Case(qid, 1.0, None, (source,), 0.0, 1.0, qid, estimate, 0, tmp_path, tmp_path)

    light, other_light = case("phone", 20.0, "a"), case("porch", HEAVY_S - 0.1, "b")
    heavy, other_heavy = case("screen", HEAVY_S, "c"), case("mic", 4 * HEAVY_S, "d")
    note, other_note, version = case("notes", 1.0, "e"), case("notes", 1.0, "f"), case("config", 1.0, "g")
    assert session.admits(light, []) and session.admits(heavy, []) and session.admits(note, []), "the first case always starts"
    assert session.admits(light, [other_light]) and session.admits(light, [other_light, note]), "light cases run together"
    assert not session.admits(heavy, [light]), "a case that may need over HEAVY_S seconds starts alone: not beside a light one"
    assert not session.admits(heavy, [light, other_light]) and not session.admits(heavy, [heavy]) and not session.admits(other_heavy, [heavy])
    assert not session.admits(light, [heavy]) and not session.admits(note, [heavy]), "and nothing starts beside it"
    assert not session.admits(note, [other_note]) and not session.admits(version, [note]) and not session.admits(note, [light, other_note]), \
        "the sources of file versions are read one case at a time"
    assert session.admits(note, [light]) and session.admits(light, [note]), "but beside others"
    assert session.admits(dataclasses.replace(heavy, estimate_s=HEAVY_S - 0.1), [light]), "just under the limit is light"
