"""Ported from tests/test_egolife.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order, part 2: what an agent with its own tools
reads in V8 and what it may not reach: the folder of the clips and lines up to a question's horizon, the deny words of the other slots and
of the download, the prompts, and the tripwire for a recording past the horizon in an exec's output. The scenario's data, plan and build are
tests/test_v8.py, the run loop and the report tests/test_v8_run.py, over the fake download of tests/v8_fake.py. The messages of the moved
tests changed with the code they call (see test_v8.py's first lines); a frozen dump is the cache's, so the view tests check that linking
and unlinking leave the dump as it was."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
from agent_spec import server_of
from test_runfolder import DENY, MM_ISOLATION, between, check_against, posix, run_of, sentinels  # noqa: F401
from v8_fake import DAYS, V8, built, clip_ends, line_at, lines_by, oracle_file, plan, recipe, source, table, view_files  # noqa: F401

from mmmb.core import fs
from mmmb.core.agents.common import command_outputs, reached
from mmmb.core.contract import Served
from mmmb.core.runfolder import IN_ONLY, RunFolder
from mmmb.core.scenario import Access
from mmmb.scenarios.v8_egolife import plan as plans
from mmmb.scenarios.v8_egolife import source as src
from mmmb.scenarios.v8_egolife.view import files_view, release_views, seen_past_horizon, view_dir

pytestmark = pytest.mark.xdist_group("v8")

# -- 6. the files arm's view and its deny list -----------------------------------------------------------------------------------


def test_the_files_view_holds_the_clips_and_lines_that_ended_by_h(built, plan, table, tmp_path):
    dump, questions = built
    ends, before = clip_ends(table), fs.stats_of(dump.root)
    h = ends[4]
    view = files_view(tmp_path / "d0", dump, h)
    assert view == tmp_path / "d0"
    expect = sorted(plans.clip_path(DAYS, c) for c, end in zip(table, ends) if end <= h)
    got = [n for n in view_files(view) if n.startswith("glasses/")]
    assert got == expect and len(got) == 5
    assert all(os.path.samefile(view / n, dump.root / n) for n in got), "the clips are the dump's, linked"
    kept = (view / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in kept] == [r for r in lines_by(plan, h) if r["end"].startswith("2026-10-29")]
    assert len(kept) == 5 and not (view / "transcript" / "2026-10-30.jsonl").exists(), "a later day has no file, not an empty one"
    assert not os.access(view / "transcript" / "2026-10-29.jsonl", os.W_OK)
    # a second, earlier H removes the later files; another slot's view is its own
    files_view(tmp_path / "d1", dump, ends[8])
    files_view(tmp_path / "d0", dump, ends[1])
    assert [n for n in view_files(view) if n.startswith("glasses/")] == expect[:2]
    assert len((view / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len([n for n in view_files(tmp_path / "d1") if n.startswith("glasses/")]) == 9
    # deleting the links left the dump as it was: the same files, sizes and times
    files_view(tmp_path / "d0", dump, ends[0] - 3600)
    assert view_files(tmp_path / "d0") == [] and fs.stats_of(dump.root) == before


def test_the_scenarios_view_is_the_slots_folder_cut_to_the_questions_horizon_and_only_named_without_build(built, table, tmp_path):
    dump, questions = built
    q = dataclasses.replace(questions[0], horizon=clip_ends(table)[2])
    folder = V8.view(q, 1, dump, tmp_path, "abc", True)
    assert folder == tmp_path / "mmb-abc-d1" and len([n for n in view_files(folder) if n.startswith("glasses/")]) == 3
    assert V8.view(q, 3, dump, tmp_path, "abc", False) == tmp_path / "mmb-abc-d3" and not (tmp_path / "mmb-abc-d3").exists(), "a dry run"
    V8.release(tmp_path, "abc")
    assert not folder.exists()
    assert V8.view(q, 1, dump, tmp_path, "abc", True) == folder and folder.exists(), "made again"


def test_a_view_that_cannot_be_removed_is_told_and_does_not_hide_the_error(tmp_path, monkeypatch, capsys):
    for k in (0, 1):
        view_dir(tmp_path, "abc", k).mkdir(parents=True, exist_ok=True)
    real = fs.remove_tree

    def busy(path):
        if path == view_dir(tmp_path, "abc", 0):
            raise PermissionError("a child still holds a file")
        real(path)

    monkeypatch.setattr(fs, "remove_tree", busy)
    release_views(tmp_path, "abc")  # does not raise
    out = capsys.readouterr().out
    assert "could not remove" in out and "a child still holds a file" in out
    assert view_dir(tmp_path, "abc", 0).exists() and not view_dir(tmp_path, "abc", 1).exists(), "the others are still removed"
    monkeypatch.undo()
    release_views(tmp_path, "abc")
    assert not view_dir(tmp_path, "abc", 0).exists()


def test_the_deny_words_of_v8_are_the_old_ones_but_for_the_new_folders_and_three_that_name_the_new_package(sentinels):
    golden = DENY["v8"]["v8"]
    new = tuple(IN_ONLY + word for word in ("v8_egolife", "from mmmb", "import mmmb"))
    assert V8.OLD_WORDS == tuple(w.removeprefix(IN_ONLY) for w in between(golden["none"], "/mm-bench'", "in:d:/__temp__/mmb-tag-w0"))
    for own, key, next_slot in ((None, "none", "in:d:/__temp__/mmb-tag-w0"), (0, "slot0", "in:d:/__temp__/mmb-tag-w1")):
        old = between(golden[key], "/mm-bench'", next_slot)  # V7's eight words, which the old EgoRun.deny inherited
        extras = tuple(golden[key][golden[key].index("in:mmb-[") + 1:])  # V8's own, after the last slot word
        assert extras[:3] == ("bench-data", "a1_jake", "in:egolife") and len(old) == 8 and len(extras) > 20
        run = run_of(V8)
        got = run.deny(None if own is None else run.slot(own), MM_ISOLATION)
        assert set(map(posix, new)) <= set(got) and not set(map(posix, new)) & set(golden[key])
        check_against(golden[key], tuple(w for w in got if w not in set(map(posix, new))), (*old, *extras))
        at = got.index("bench-data")
        assert got[at:at + len(extras)] == extras, "V8's own words, in the old order"
    assert V8.deny_words([], Path("t"), "k") == (*(IN_ONLY + w for w in (*V8.OLD_WORDS, "v8_egolife", "from mmmb", "import mmmb")),
                                                 "bench-data", "a1_jake", "in:egolife"), "with no other slot, no folder of recordings"


def test_the_deny_list_keeps_an_agent_off_the_other_slots_and_the_download_and_not_off_its_own_view(built, tmp_path):
    dump, _ = built
    run = RunFolder(tmp_path / "run", 1, "egolife", DAYS, dump.root, tmp_path / "temp", "t8", 900.0, V8, ())
    deny = run.deny(run.slot(0), MM_ISOLATION)
    own, other = view_dir(run.temp, "t8", 0), view_dir(run.temp, "t8", 1)

    def stopped(text: str, output: bool = False) -> bool:
        return reached(text, deny, output) is not None

    clip = "glasses/2026-10-29/11.09.42.mp4"
    assert not stopped(f"ffmpeg -i {own}/{clip} frame.png") and not stopped(f"Input #0, from '{own}\\glasses\\x.mp4'", output=True)
    assert stopped(f"ffmpeg -i {other}/{clip} frame.png")
    assert stopped(r"copy D:\bench-data\egolife\A1_JAKE\DAY2\DAY2_A1_JAKE_10050125.mp4 .")
    assert stopped(f"cd ..\\{other.name}") and stopped("dir D:\\mmb-temp\\*-d1") and stopped(f"dir {run.temp}\\mmb-?-d2")
    assert stopped(f"Input #0, from '{other}\\glasses\\x.mp4'", output=True), "ffmpeg echoes the path it reads"
    assert stopped("Directory of D:\\x\\A1_JAKE\\DAY1", output=True) and stopped(f"{other.name}/transcript/2026-10-29.jsonl", output=True)
    assert not stopped(f"{other.name}\n{run.slot(1).name}", output=True), "a listing of the temp folder may name a view"
    assert stopped(str(run.folder)) and stopped(str(run.slot(1)))
    # the dataset's name is refused in a command and not in an output: a transcript line says "EGOlife" and an exec may print it
    assert stopped("cd D:\\egolife-folder") and not stopped("EGOlife都是过日子的人: Everyone here is living their lives.", output=True)
    assert deny == run.deny(run.slot(0), MM_ISOLATION) and run.deny(run.slot(1), MM_ISOLATION) != deny
    run.check_paths(MM_ISOLATION)
    with pytest.raises(ValueError, match="holds 'egolife'"):
        dataclasses.replace(run, temp=tmp_path / "egolife-temp").check_paths(MM_ISOLATION)


PLAIN = Served(None, None, (), None, None, None, {})


def test_a_prompt_for_a_system_and_the_files_together_has_both_sentences_and_a_prompt_for_another_format_or_a_holderless_server_is_refused(built):
    q = next(q for q in built[1] if q.qid == "ego-001")
    data = Path(r"D:\mmmb-temp\mmb-0123456789-d0")
    held = Served(server_of(Path("home"), Path("work")), "base", (), None, "mm", None, {})
    got = V8.prompt(q, Access(True, data, held, "line"))
    assert got == (
        "It is now Thursday 29 October 2026, 11:12 (day 1 of these recordings; day 1 was Thursday 29 October). I am Jake. "
        "mm has recorded these sources of mine from day 1 to now: glasses (the video of the camera on my glasses, 30 s clips at 20 fps) and "
        "transcript (what was said around me: a record per spoken line, with its speaker and the words in English and Chinese). "
        f"The recordings of my days so far are in {data}: glasses/<date>/ holds the video of the camera on my glasses as 30 s clips "
        "(1408x1408, 20 fps), each named by the time it starts; transcript/<date>.jsonl holds what was said around me, one JSON line per "
        "spoken line with its start, end, speaker and the words in English and Chinese. Your working folder is yours to write in. ffmpeg is "
        "on PATH (there is no ffprobe).\n\nQuestion 1?\nA) Tasha\nB) Alice\nC) Shure\nD) Lucia\n\nIf what you can find does not settle it, "
        "give your best guess. End with one line ANSWER: <A, B, C or D>.")
    with pytest.raises(ValueError, match="the answer format is one of line, json, not 'yaml'"):
        V8.prompt(q, Access(False, None, PLAIN, "yaml"))
    with pytest.raises(ValueError, match="has no word for itself"):
        V8.prompt(q, Access(False, None, dataclasses.replace(held, holder=None), "line"))
    with pytest.raises(ValueError, match="has no options"):
        V8.prompt(dataclasses.replace(q, options=None), Access(False, None, PLAIN, "line"))


def test_a_transcript_line_that_a_files_exec_could_not_print_is_refused_at_build(plan, recipe, tmp_path, monkeypatch):
    run = RunFolder(tmp_path / "run", 1, "egolife", DAYS, tmp_path / "data", tmp_path / "temp", "t8", 900.0, V8, ())
    deny = run.deny(run.slot(0), MM_ISOLATION)
    plans.check_transcript(plan.logs, deny)
    line = {"start": "2026-10-29T11:09:47.100", "end": "2026-10-29T11:09:50.766", "speaker": "Jake", "zh": "", "en": "x"}
    said = {"2026-10-29": ({**line, "zh": "EGOlife都是过日子的人", "en": "Everyone here is living their lives."},)}
    plans.check_transcript(said, deny)  # "egolife" is refused in a command only
    for word in ("bench-data", "A1_JAKE", ".claude", str(run.folder)):
        with pytest.raises(ValueError, match="2026-10-29T11:09:50.766 holds"):
            plans.check_transcript({"2026-10-29": ({**line, "en": f"look in {word} now"},)}, deny)
    # and a build refuses the download whose transcript holds one of the scenario's own words, before it writes anything
    real = src.read_transcripts

    def with_word(source, days):
        lines = real(source, days)
        return [{**lines[0], "en": "look in bench-data now"}, *lines[1:]]

    monkeypatch.setattr(plans, "read_transcripts", with_word)
    with pytest.raises(ValueError, match="holds 'bench-data'"):
        V8.build(recipe, tmp_path / "out", 2)
    assert not (tmp_path / "out").exists()


def test_a_files_exec_whose_output_shows_a_time_or_a_clip_past_the_horizon_is_escaped(built, plan, table):
    dump, questions = built
    q = next(q for q in questions if q.qid == "ego-001")
    h = q.horizon
    late, early = line_at(plan, True), line_at(plan, False)
    seen = seen_past_horizon([json.dumps(late), json.dumps(early)], dump, h)
    assert seen == sorted({late["start"], late["end"]}), "the times of the late line, none of the early one"
    texts = ["glasses/2026-10-29/12.30.30.mp4  2026-11-03T01:20:11.123  2026-10-29T99:99:99.123", "x glasses\\2026-10-29\\12.31.00.mp4 y",
             "glasses/2026-10-29/11.09.42.mp4 glasses/2026-10-29/11.11.30.mp4 glasses/2026-10-29/01.01.01.mp4"]
    assert seen_past_horizon(texts, dump, h) == ["glasses/2026-10-29/12.30.30.mp4", "glasses/2026-10-29/12.31.00.mp4"], \
        "a clip that ended after H, either slash; not the clips in the view, an unknown name, the time now or a time that is none"
    # a path with its backslashes escaped, as a JSON or a Python repr prints it, is a path too
    escaped = "path: 'glasses\\\\2026-10-29\\\\12.31.00.mp4' and glasses\\\\\\\\2026-10-29\\\\\\\\12.30.30.mp4"
    assert seen_past_horizon([escaped], dump, h) == ["glasses/2026-10-29/12.30.30.mp4", "glasses/2026-10-29/12.31.00.mp4"]
    # a time that is no transcript line's is the agent's own (a file's mtime, a computed time), after H or not
    own = f"mtime 2026-10-29T12:59:59.999 now {late['end'][:-3]}999 and {early['end']}"
    assert seen_past_horizon([own], dump, h) == []
    # a clip name in a line that reports a missing file is a guess at the clip of "now": no escape, in any of the usual wordings
    guess = "glasses/2026-10-29/12.30.30.mp4"
    for said in (f"{guess}: No such file or directory", f"FileNotFoundError: [Errno 2] No such file or directory: '{guess}'",
                 f"Get-Content : Cannot find path 'D:\\x\\{guess}' because it does not exist.", f"bash: {guess}: not found",
                 f"Traceback ... FileNotFoundError: {guess}"):
        assert seen_past_horizon([f"{said}\nexit 1"], dump, h) == [], said
        assert seen_past_horizon([said, guess], dump, h) == [guess], "the same name on a line of its own is still read"
    codex = [{"type": "item.completed", "item": {"type": "command_execution", "command": "Get-Content x", "aggregated_output": json.dumps(late)}}]
    claude = [{"type": "user", "message": {"content": [
                  {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": json.dumps(late)}]},
                  {"type": "tool_result", "tool_use_id": "u", "content": json.dumps(early)}]}},
              {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(late)}]}}]
    assert command_outputs(codex) == [json.dumps(late)] and command_outputs(claude) == [json.dumps(late), json.dumps(early)]
    def on(own_tools: bool, **out) -> dict:
        return V8.after_exec(q, {"problem": None, **out}, dump, Access(own_tools, dump.root if own_tools else None, PLAIN, "line"))
    for events in (codex, claude):
        got = on(True, events=events)
        assert got["seen_past_horizon"] == sorted({late["start"], late["end"]}) and got["passed"] is False
        assert got["problem"].startswith("escaped: its output showed recordings past the horizon: ")
    assert on(True, events=[]) == {"seen_past_horizon": []} and on(True) == {"seen_past_horizon": []}
    only_early = [{"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": json.dumps(early)}}]
    assert on(True, events=only_early) == {"seen_past_horizon": []}
    assert "problem" not in on(True, events=codex, problem="timeout: no answer in 900 s"), "an exec that already failed keeps its problem"
    assert on(False, events=codex) == {}, "an exec with no tools of its own has no such tripwire"
    names = sorted(plans.clip_path(DAYS, c) for c, end in zip(table, clip_ends(table)) if end > h)[:7]
    many = [{"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "\n".join(names)}}]
    got = on(True, events=many)
    assert got["seen_past_horizon"] == names and got["problem"].endswith(f"{', '.join(names[:5])} and 2 more"), "five are told, the rest counted"
