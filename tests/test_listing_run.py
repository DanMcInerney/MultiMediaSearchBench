"""New tests for the listing system (plan 6.2) beside test_listing.py's: the configuration `listing` run end to end through the real runner with
a stand-in agent that reads the real server; the real `claude.ask_claude` and `codex.ask_codex` over a stand-in process, with the real session and
server (tests/stand_in_agent.py); and the report, which flags a question the listing answers (`answerable_from_metadata`) and keeps it out of the
paired counts.

Expected values are hand-counted from the plan's 19 questions of the tiny week, from what a test plants (a result that passed, an absent question,
a tool the session should not have), and from the arguments the agents are given by their own flags: no paid agent runs."""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import imageio_ffmpeg
import pytest
from agent_spec import NO_ISOLATION
from listing_world import Reader, ingested, planted_dump
from stand_in_agent import run_agent
from stub_system import run_of as stub_run_of
from stub_system import tiny_questions
from test_report import FAKE, FILES, right, run_of, shape, stub_out, stub_results

from mmmb.core import fs, render, report
from mmmb.core.agents.common import result_jsons
from mmmb.core.config import Config, parse_config
from mmmb.core.contract import SessionContext
from mmmb.core.record import record
from mmmb.core.runner import Agent, note_configs, run_agents
from mmmb.systems.listing.system import Listing

pytestmark = pytest.mark.xdist_group("listing-run")
LISTING = Config("listing", "listing", None, False)
LISTING_FILES = Config("listing+files", "listing", None, True)
QUESTIONS = tiny_questions()
ALL = ["audio", "image", "records", "text", "video"]


# -- through the runner ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    """A run folder of the tiny week's questions with a made-up frozen data folder (listing never opens it), a planted dump and its listing."""
    ffmpeg = tmp_path / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
    run, _ = stub_run_of(tmp_path, configs=(FILES, LISTING))
    (tmp_path / "planted").mkdir()
    dump = planted_dump(tmp_path / "planted")
    store = ingested(dump, tmp_path / "store").parent
    return types.SimpleNamespace(run=run, dump=dump, store=store, root=tmp_path)


def test_the_listing_configuration_runs_every_question_through_the_runner_and_the_report_names_the_one_it_answered(world):
    run, absent = world.run, [q for q in QUESTIONS if q.tags["absent"]]
    needle = next(q for q in QUESTIONS if not q.tags["absent"])
    reader = Reader({needle.qid: right(needle)})
    run_agents(run, {"claude": Agent(reader)}, [LISTING], world.dump, lambda system: world.store, 2, isolation=NO_ISOLATION, answer_format="line")
    results = {q.qid: json.loads((run.result_path("claude", LISTING, q.qid)).read_text(encoding="utf-8")) for q in QUESTIONS}
    assert len(results) == len(reader.execs) == 19, "every question: it reads all five modalities"
    assert all(r["problem"] is None and (r["config"], r["system"], r["variant"]) == ("listing", "listing", None) for r in results.values())
    assert all(r["tools"] == {"list_files": 1, "file_info": 1} and r["calls"] == 2 for r in results.values())
    assert {qid for qid, r in results.items() if r["passed"]} == {needle.qid, *(q.qid for q in absent)}, "unknown is right for an absent question"
    for spec in reader.execs:
        assert (spec.own_tools, spec.data) == (False, None) and spec.server.name == "listing" and spec.served.holder == "listing"
        assert spec.server.tools == ("list_files", "file_info") and "--until" not in spec.server.args, "V7 has no horizon"
        assert "listing has recorded these sources of mine" in spec.prompt and spec.isolation == NO_ISOLATION
    noted = json.loads((run.folder / "systems.json").read_text(encoding="utf-8"))["listing"]
    assert noted["asked"] == ALL and noted["readable"] == ALL and noted["description"]["leak_filter"] is True and noted["description"]["horizon"] == "cut"
    assert noted["description"] == {**dataclasses.asdict(Listing().describe(None)), "reads": ALL}
    fs.write_json(run.folder / "run.json", {"mmmb_commit": "abc"})
    figures = report.numbers([run])
    assert figures["agents"] == {} and figures["frame"]["tested"] == "tested", "the listing is not compared with the control"
    found = figures["answerable_from_metadata"]
    assert (found["filters"], found["asked"]) == (["listing"], len(QUESTIONS) - len(absent))
    assert [(q["qid"], q["agents"]) for q in found["questions"]] == [(needle.qid, ["claude"])]
    text = render.markdown(figures)
    assert "## Answerable from metadata" in text and needle.qid in text and "an absent question is exempt" in text


# -- through the real agents -------------------------------------------------------------------------------------------------------

AGENTS = pytest.mark.parametrize("agent", ("claude", "codex"))
LOOK = r'''
listed = jsons_of(call("list_files", {"limit": 2}))[0]
info = jsons_of(call("file_info", {"path": listed["files"][0]["path"]}))[0]
if plan.get("shell"):  # a tool the session does not have
    if plan["agent"] == "claude":
        use = f"toolu_{next(counter)}"
        emit({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": use, "name": "Read", "input": {"file_path": "x"}}]}})
    else:
        emit({"type": "item.started", "item": {"id": "s", "type": "command_execution", "command": "type listing.jsonl", "aggregated_output": "", "status": "in_progress"}})
final = "ANSWER: unknown"
'''


@pytest.fixture
def git(tmp_path):
    root = tmp_path / "Git"
    for sub in ("cmd", "bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("")
    return root


def exec_of(agent, world, monkeypatch, git, **plan):
    """One exec of the first question by the stand-in `agent`, in a folder of its own and with the exec's folder as the runner leaves it: gone."""
    folder = world.run.exec_dir(agent, LISTING, QUESTIONS[0].qid)
    if folder.exists():
        fs.remove_tree(folder)
    session = Listing().open(world.store, world.dump, SessionContext(agent, None, 900.0, world.root / "homes", lambda: 0.0))
    where = world.root / f"{agent}-{len(list(world.root.glob(f'{agent}-*')))}"
    where.mkdir()
    return run_agent(agent, where, monkeypatch, git, session, world.run, world.dump, LISTING, QUESTIONS[0], LOOK, plan)


@AGENTS
def test_the_server_a_session_serves_starts_under_the_real_agent_path_with_no_tool_but_its_two(agent, world, monkeypatch, git):
    out, listed = exec_of(agent, world, monkeypatch, git)
    assert out["problem"] is None, out["problem"]  # (for Claude: `check_init` found exactly the two tools and the one server, connected)
    assert listed == ["list_files", "file_info"] and [t["tool"] for t in out["tools"]] == ["list_files", "file_info"]
    run_dir = world.run.exec_dir(agent, LISTING, QUESTIONS[0].qid)
    if agent == "claude":
        argv = out["argv"]
        assert argv[argv.index("--tools") + 1] == "", "no built-in tool: not Read, not Bash"
        assert argv[argv.index("--allowedTools") + 1:argv.index("--allowedTools") + 3] == ["mcp__listing__list_files", "mcp__listing__file_info"]
        assert "--add-dir" not in argv and "--mcp-config" in argv
        init = json.loads((run_dir / "init.json").read_text(encoding="utf-8"))
        assert sorted(init["tools"]) == ["mcp__listing__file_info", "mcp__listing__list_files"]
        assert init["mcp_servers"] == [{"name": "listing", "status": "connected"}]
        servers = json.loads((run_dir / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        assert list(servers) == ["listing"] and servers["listing"]["args"][1:3] == ["--listing", str(world.store / "listing.jsonl")]
    else:
        assert "mcp_servers.listing.command" in " ".join(out["argv"])
    assert len(result_jsons(out["tools"][0]["result"])[0]["files"]) == 2


def test_a_session_that_has_a_third_tool_is_not_the_listing_arm_and_a_call_outside_its_two_ends_the_exec(world, monkeypatch, git):
    extra, _ = exec_of("claude", world, monkeypatch, git, extra_tools=["Read"])
    assert extra["problem"] and "extra ['Read']" in extra["problem"], extra["problem"]
    for agent in ("claude", "codex"):
        used, _ = exec_of(agent, world, monkeypatch, git, shell=True)
        assert used["problem"] and used["problem"].startswith("escaped"), (agent, used["problem"])


# -- the report --------------------------------------------------------------------------------------------------------------------

def run_with_listing(tmp_path: Path, name: str, *, seed: int = 7, agents=("claude",)):
    """A run of the tiny week's questions in the configurations files, mm/fake and listing, with the results of the stub for files and mm/fake
    of each agent (mm/fake is right on all 19, files only where the stub is) and no result of the listing: systems.json is what `note_configs`
    writes."""
    run = run_of(tmp_path / name, QUESTIONS, seed=seed, configs=(FILES, FAKE, LISTING))
    note_configs(run, {LISTING: Listing().describe(None)}, None)
    for agent in agents:
        for config in (FILES, FAKE):
            for q in QUESTIONS:
                fs.write_json(run.result_path(agent, config, q.qid),
                              record(q, agent, config, seed, f"execs/{agent}/{config.dir}/{q.qid}", "2026-11-02T12:00:00", stub_out(config, q)))
    return run


def listing_result(run, agent: str, q, *, passed: bool) -> None:
    """A result of the listing for `q`, passed or not, as `record` writes it."""
    out = {**stub_out(FAKE, q), "final": "x", "tools": [{"tool": "list_files", "arguments": {}}], **({} if passed else {"answer": "wrong 99999"})}
    fs.write_json(run.result_path(agent, LISTING, q.qid), record(q, agent, LISTING, run.seed, f"execs/{agent}/listing/{q.qid}", "2026-11-02T12:00:00", out))


def paired(figures: dict, agent: str = "claude") -> tuple[int, int, int]:
    """The questions asked, the paired questions and the paired media questions of `mm/fake` against files for `agent`."""
    row = figures["agents"][agent]["mm/fake"]
    return row["questions"], row["all"]["n"], row["media"]["n"]


def test_a_passing_listing_result_for_a_question_with_a_needle_leaves_the_paired_count_by_one_and_is_named(tmp_path):
    k = shape(QUESTIONS)
    run = run_with_listing(tmp_path, "run")
    before = report.numbers([run])
    assert paired(before) == (k.n, k.n, k.media) and before["answerable_from_metadata"]["questions"] == [], "the listing is there and answered nothing"
    assert list(before["agents"]["claude"]) == ["mm/fake"] and before["frame"]["tested"] == "mm", "it is not a tested configuration"
    q = next(q for q in QUESTIONS if q.tags["modality"] == "image" and not q.tags["absent"])
    listing_result(run, "claude", q, passed=True)
    after = report.numbers([run])
    assert paired(after) == (k.n - 1, k.n - 1, k.media - 1), "one question fewer, everywhere it was counted"
    assert after["agents"]["claude"]["mm/fake"]["all"]["control"]["n"] == k.n - 1 and after["agents"]["claude"]["mm/fake"]["pending"] == 0
    assert after["answerable_from_metadata"] == {"filters": ["listing"], "asked": 1, "questions": [
        {"run": run.name, "seed": 7, "qid": q.qid, "type": "image", "level": q.hint, "agents": ["claude"]}]}
    text = render.markdown(after)
    assert "## Answerable from metadata" in text and f"| {run.name} | {q.qid} | image | {q.hint} | claude |" in text
    assert after["totals"]["results"] == before["totals"]["results"] + 1, "its result is still a result"


def test_a_passing_absent_question_and_a_failed_one_with_a_needle_are_not_flagged(tmp_path):
    k = shape(QUESTIONS)
    run = run_with_listing(tmp_path, "run")
    absent = next(q for q in QUESTIONS if q.tags["absent"] and q.tags["modality"] in ("image", "video", "audio"))
    needle = next(q for q in QUESTIONS if not q.tags["absent"] and q.tags["modality"] == "video")
    listing_result(run, "claude", absent, passed=True)
    listing_result(run, "claude", needle, passed=False)
    figures = report.numbers([run])
    assert paired(figures) == (k.n, k.n, k.media), "an absent question is exempt (C9): it passes by saying unknown"
    assert figures["answerable_from_metadata"] == {"filters": ["listing"], "asked": 1, "questions": []}, "asked: the one with a needle that had a result"


def test_a_question_one_agents_listing_passed_is_out_of_every_agents_count_and_it_names_who_passed_it(tmp_path):
    k = shape(QUESTIONS)
    run = run_with_listing(tmp_path, "run", agents=("claude", "codex"))
    q = next(q for q in QUESTIONS if q.tags["modality"] == "video" and not q.tags["absent"])
    listing_result(run, "codex", q, passed=True)
    figures = report.numbers([run])
    assert paired(figures, "claude") == paired(figures, "codex") == (k.n - 1, k.n - 1, k.media - 1), "what the dump leaks, not what an agent knows"
    assert [x["agents"] for x in figures["answerable_from_metadata"]["questions"]] == [["codex"]]
    listing_result(run, "claude", q, passed=True)
    assert [x["agents"] for x in report.numbers([run])["answerable_from_metadata"]["questions"]] == [["claude", "codex"]]


def test_a_flagged_question_is_flagged_in_its_own_run_and_not_in_a_run_of_another_seed_that_pools_with_it(tmp_path):
    k = shape(QUESTIONS)
    one, other = run_with_listing(tmp_path, "one"), run_with_listing(tmp_path, "other", seed=8)
    q = next(q for q in QUESTIONS if q.tags["modality"] == "image" and not q.tags["absent"])
    listing_result(one, "claude", q, passed=True)
    figures = report.numbers([one, other])
    assert paired(figures) == (2 * k.n - 1, 2 * k.n - 1, 2 * k.media - 1), "the same qid in the other seed's run is another question"
    assert [(x["seed"], x["qid"]) for x in figures["answerable_from_metadata"]["questions"]] == [(7, q.qid)]
    assert figures["answerable_from_metadata"]["asked"] == 1


def test_the_listing_with_the_agents_own_tools_too_is_a_tested_configuration_and_a_run_without_a_filter_has_no_such_section(tmp_path):
    both = parse_config("listing+files", {"files": None, "listing": None, "mm": "d1-3B"})
    assert both == LISTING_FILES
    run = run_of(tmp_path / "both", QUESTIONS, configs=(FILES, FAKE, both))
    note_configs(run, {both: Listing().describe(None)}, None)  # (its system says leak_filter, and it has tools of its own)
    stub_results(run, QUESTIONS)
    for q in QUESTIONS:
        fs.write_json(run.result_path("claude", both, q.qid), record(q, "claude", both, 7, "x", "2026-11-02T12:00:00", stub_out(both, q)))
    figures = report.numbers([run])
    assert sorted(figures["agents"]["claude"]) == ["listing+files", "mm/fake"] and "answerable_from_metadata" not in figures
    assert "Answerable from metadata" not in render.markdown(figures)
    plain = run_of(tmp_path / "plain", QUESTIONS)
    stub_results(plain, QUESTIONS)
    assert "answerable_from_metadata" not in report.numbers([plain])
