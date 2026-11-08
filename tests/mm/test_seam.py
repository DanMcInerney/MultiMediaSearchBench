"""New tests (G1's shared cause 3 and the F4 repair): the real `MmSession`, the real server command it serves and the real `claude.ask_claude` or
`codex.ask_codex` together, through a stand-in agent process (tests/mm/seam.py). What a stand-in agent that calls the daemon's `Api` itself cannot show:
that the server the session serves starts as an agent starts it, lists the four tools, works in the exec's project, and that what `look` returns to the
agent with its own tools too (a path in the run folder, which is denied) is not an escape.

Expected values come from what the test plants and from the protocol: the four tools mm offers, the project the exec's working folder names, the folder
`look` writes in. No paid agent runs and no model is loaded; every daemon is over a folder of the test's own with the fake decider."""

from __future__ import annotations

import dataclasses
import json
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("mms")

import fakes  # noqa: E402
import seam  # noqa: E402
from fakes import S, S_FILES, Opened, run_of  # noqa: E402
from test_mm_arm import by_qid  # noqa: E402

from mmsb.core.agents import claude  # noqa: E402
from mmsb.core.agents.common import result_jsons  # noqa: E402
from mmsb.core import grade  # noqa: E402
from mmsb.core.answer import answer_of, parse_final  # noqa: E402
from mmsb.core.grounded import ground, resolve  # noqa: E402
from mmsb.systems.mm.home import exec_project  # noqa: E402

TOOLS = {"ask", "find", "look", "label"}
pytestmark = pytest.mark.xdist_group("mm-arm")
agents = pytest.mark.parametrize("agent", ("claude", "codex"))


@pytest.fixture
def git(tmp_path):
    """A Git for Windows folder tree: Claude's shell with its own tools needs one, and the real one is the owner's."""
    root = tmp_path / "Git"
    for sub in ("cmd", "bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("")
    return root


def exec_of(agent, week, store, tmp_path, monkeypatch, git, config, answer=None) -> types.SimpleNamespace:
    """One exec of a photo question of the tiny week, run by the stand-in agent over a session of the fake decider: the question, its result, the tools
    the server listed, the run, the exec's init event (Claude's) and the session's home."""
    recipe, dump, questions = week
    q = next(q for q in questions if q.tags["template"] == "photo-sign" and not q.tags["absent"])
    ask = {"question": by_qid(fakes.V7.plan_of(recipe))[q.qid]["fake"]["question"], "sources": ["phone"], "since": q.scope.since, "until": q.scope.until}
    run = run_of(tmp_path, recipe, questions, dump, [config])
    with Opened(store, dump, run.folder / "homes") as session:
        out, listed = seam.run_agent(agent, tmp_path, monkeypatch, git, session, run, dump, config, q, ask, answer)
        init = run.exec_dir(agent, config, q.qid) / "init.json"
        return types.SimpleNamespace(q=q, out=out, listed=listed, run=run, home=session.home,
                                     init=json.loads(init.read_text(encoding="utf-8")) if init.exists() else None)


def check_server_exchange(done: types.SimpleNamespace) -> dict:
    """What every exec's stream shows of the server: it ended well (for Claude `check_init` passed: its init event lists the four tools the server
    listed), the asks ran in the exec's project, and the `look` of a sighting gave a picture in the home's `looks` folder. Returns that look."""
    out = done.out
    assert out["problem"] is None and answer_of(out["final"]) == "unknown", out["problem"]
    assert set(done.listed) == TOOLS and len(done.listed) == 4, "the server lists exactly its four tools"
    if done.init is not None:
        assert {t.removeprefix("mcp__mm__") for t in done.init["tools"] if t.startswith("mcp__mm__")} == TOOLS
        assert done.init["mcp_servers"] == [{"name": "mm", "status": "connected"}] and done.init["plugins"] == []
    calls = [t["tool"] for t in out["tools"]]
    assert calls[:2] == ["find", "ask"] and "look" in calls
    asked = [got for t in out["tools"] if t["tool"] == "ask" for got in result_jsons(t["result"])]
    assert asked and all(got["searched"]["project"]["id"] == exec_project(done.q.qid) for got in asked), "the asks ran in the exec's project"
    assert asked[-1]["state"] == "done" and asked[-1]["sightings"], "and found the photo"
    look = next(got for t in out["tools"] if t["tool"] == "look" for got in result_jsons(t["result"]) if got.get("image"))
    assert Path(look["image"]["path"]).parent == done.home / "looks" and Path(look["image"]["path"]).is_file()
    return look


@agents
def test_the_server_a_session_serves_starts_under_the_real_agent_path_lists_its_four_tools_and_works_in_the_execs_project(agent, week, store, tmp_path,
                                                                                                                      monkeypatch, git):
    done = exec_of(agent, week, store, tmp_path, monkeypatch, git, S)
    check_server_exchange(done)
    assert done.out["tools"][-1]["tool"] == "look"
    if agent == "claude":
        assert sorted(done.init["tools"]) == sorted(f"mcp__mm__{t}" for t in TOOLS), "no tool of its own: the server alone"


@agents
def test_an_agent_with_its_own_tools_too_may_open_the_picture_look_named_though_it_is_in_the_denied_run_folder(agent, week, store, tmp_path, monkeypatch, git):
    """G1's F4 through the real path: before, opening the file `look` named ended the exec as `escaped`, which counts as a wrong answer."""
    done = exec_of(agent, week, store, tmp_path, monkeypatch, git, S_FILES)
    look = check_server_exchange(done)
    path, last = look["image"]["path"], done.out["tools"][-1]
    assert str(done.run.folder) in path, "a path in the run folder, which is denied to the own tools"
    if agent == "claude":
        argv = done.out["argv"]
        assert sorted(done.init["tools"]) == sorted([*claude.FILES_TOOLS, *(f"mcp__mm__{t}" for t in TOOLS)]), "its own tools, then the server's"
        assert last["tool"] == "Read" and last["arguments"] == {"file_path": path}
        assert [argv[i + 1] for i, a in enumerate(argv) if a == "--add-dir"] == [str(done.run.data), str(done.home / "looks")]
    else:
        assert last["tool"] == "shell" and path in last["arguments"]["command"]


@agents
def test_an_agent_that_cites_the_sightings_of_its_ask_by_source_and_iso_time_is_grounded_by_the_real_servers_own_times(agent, week, store, tmp_path,
                                                                                                                       monkeypatch, git):
    """DESIGN 6.2 corrected (C7): an agent with the system's tools alone has no file path, only a source and times, which are what the real
    server's `ask` gives. The final message of the stand-in here is the right answer citing each sighting its ask returned; the sightings of the
    photo question are the photo, so the citation is where the truth is: `source` phone and the photo's own time, from the manifest."""
    _, dump, questions = week
    q = next(q for q in questions if q.tags["template"] == "photo-sign" and not q.tags["absent"])
    done = exec_of(agent, week, store, tmp_path, monkeypatch, git, S, answer=q.grade.truth)
    assert done.out["problem"] is None
    final = parse_final(done.out["final"], "json")
    assert final.error is None and final.answer == q.grade.truth and final.evidence
    assert all(c.source == "phone" and c.file is None for c in final.evidence), "the system's tools give no file"
    photo = next(e for e in dump.entries if e.path == q.truth.evidence[0].file)
    placed = [resolve(c, dump) for c in final.evidence]
    assert all(p is not None and p.source == "phone" for p in placed)
    assert any(abs(p.start - photo.t) < 0.01 and abs(p.end - photo.t) < 0.01 for p in placed), "one of them is the photo, to the millisecond"
    got = ground(q, final, grade.grade(q.grade, final.answer), dump)
    assert got.status == "grounded" and got.cited == got.matched == len(final.evidence) and got.unresolved == 0, got
    shifted = [dataclasses.replace(c, t0=at_hours_later(c.t0, 1), t1=at_hours_later(c.t1, 1)) for c in final.evidence]
    assert ground(q, dataclasses.replace(final, evidence=tuple(shifted)), True, dump).status == "lucky", "an hour later is not the photo"


def at_hours_later(iso: str, hours: int) -> str:
    return (datetime.fromisoformat(iso) + timedelta(hours=hours)).isoformat()
