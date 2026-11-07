"""The fake EgoLife download of the V8 tests and the fixtures they share (tests/test_v8.py, tests/test_v8_run.py): a two-day download (tiny
clips made with ffmpeg, SRT files with the real file's oddities, a QA file of about twenty items; copied from tests/test_egolife.py of
Multimem at fe0ece5), the recipe, plan and dump made from it, and the run folder a build writes. A test module imports the fixtures it
uses; `oracle_file` is autouse, so a module that imports it plans with this file's oracles instead of the 50 the recorded run chose."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import imageio_ffmpeg
import pytest
from stub_system import STUB

from mmmb import scenarios
from mmmb.core import fs
from mmmb.core.config import Config
from mmmb.core.contract import Dump
from mmmb.core.question import write_questions
from mmmb.core.runfolder import RunFolder
from mmmb.scenarios.v8_egolife import plan as plans
from mmmb.scenarios.v8_egolife import source as src

V8 = scenarios.get("v8_egolife")
DAYS = (date(2026, 10, 29), date(2026, 10, 30))
BLIND = Config("blind", None, None, False)
FILES = Config("files", "files", None, True)
OPTIONS = {"A": "Tasha", "B": "Alice", "C": "Shure", "D": "Lucia"}
SHADE = "Is shade:200,30,30 in view?"  # the fake decider's own words for the red box of the evidence clip
WORD = "Does the text have contains:screwdriver ?"


DAYS = (date(2026, 10, 29), date(2026, 10, 30))


BLIND = Config("blind", None, None, False)


FILES = Config("files", "files", None, True)


OPTIONS = {"A": "Tasha", "B": "Alice", "C": "Shure", "D": "Lucia"}


SHADE = "Is shade:200,30,30 in view?"  # the fake decider's own words for the red box of the evidence clip


WORD = "Does the text have contains:screwdriver ?"


# Clips: a block starts with a short clip (18 s) at an unaligned time, then 30 s clips on :00 and :30; each day has two blocks and
# a gap between them. The evidence clip of day 1 (11:10:30 to 11:11:00) shows a red frame from 10 s to 20 s into it.
CLIPS = {1: [("11094208", 18), ("11100000", 30), ("11103000", 30), ("11110000", 30), ("11113000", 30), ("11120000", 30),
             ("12301225", 18), ("12303000", 30), ("12310000", 30), ("12313000", 30), ("12320000", 30), ("12323000", 30)],
         2: [("10050125", 18), ("10053000", 30), ("10060000", 30), ("10063000", 30), ("10070000", 30), ("10073000", 30),
             ("15195000", 18), ("15200000", 30), ("15203000", 30), ("15210000", 30), ("15213000", 30), ("15220000", 30)]}


EVIDENCE = "11103000"


# (offset from the hour, text lines): two-line entries with the real file's spacing, a one-line entry and one with no speaker
SRT = {(1, 11): [("09:47,100", "09:50,766", ["Jake: 好然后一个秒表", "Jake: Okay, then we need a stopwatch."]),
                 ("09:54,700", "09:55,700", ["Shure:  时间戳呗", "Shure: Mark a timestamp, right?"]),
                 ("10:44,000", "10:47,000", ["Alice:我用螺丝刀拧螺丝", "Alice:I use the screwdriver to tighten screws."]),
                 ("10:46,000", "10:49,500", ["Tasha: 好的", "Tasha: OK."]),
                 ("10:50,000", "10:51,000", ["Lucia: 要稍微使点力气"]),
                 ("12:25,000", "12:35,000", ["Jake: 再来一个", "Jake: One more."]),
                 ("12:40,000", "12:41,000", ["Jake: 快好了", "Jake: Almost done."])],
       (1, 12): [("30:40,000", "30:42,000", ["Katrina: 看这个", "Katrina: Look at this."]),
                 ("31:14,000", "31:16,000", ["Shure: 我们晚上做烧烤", "Shure: We will grill tonight."]),
                 ("34:00,000", "34:02,000", ["Jake: 收工", "Jake: Done for now."])],
       (2, 10): [("05:40,000", "05:42,000", ["Jake: 今天开始", "Jake: Let's start today."])],
       (2, 15): [("20:10,000", "20:12,000", ["Alice: 看", "Alice: Look."]), ("21:00,000", "21:01,000", ["小薇;哪个"])]}


def make_clip(ffmpeg: str, path: Path, seconds: int, n: int, evidence: bool) -> None:
    """A 64x64 clip of 2 frames a second with a silent AAC track; the colour is its own (remember keeps one copy of a file)."""
    video = f"color=c=0x{80:02x}{80:02x}{40 + 5 * n:02x}:s=64x64:r=2:d={seconds}"
    if evidence:
        video += ",drawbox=x=0:y=0:w=iw:h=ih:color=0xC81E1E:t=fill:enable='between(t,10,20)'"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "4", "-c:a", "aac", "-shortest",
                    str(path)], check=True)


def pl(day: int, text: str) -> float:
    """The plain time of `HH:MM:SS.ff` on a day."""
    h, m, s = text.split(":")
    return src.plain(day, int(h) * 3600 + int(m) * 60 + float(s))


def ep(day: int, text: str) -> float:
    return plans.local(DAYS, pl(day, text))


def raw(id_: str, category: str, query: tuple, targets: list[tuple], *, audio: bool = False, listed: bool = False,
        date_text: str | None = None, time_text: str | None = None) -> dict:
    """One item of the QA file. `query` and each target are (day, `HH:MM:SS.ff`); a list of targets is a `time_list`."""
    def code(text: str) -> str:
        return text.replace(":", "").replace(".", "")

    day = targets[0][0]
    when = {"time_list": [code(t) for _, t in targets]} if listed else {"time": time_text or code(targets[0][1])}
    out = {"ID": id_, "query_time": {"date": f"DAY{query[0]}", "time": code(query[1])}, "type": category, "need_audio": audio,
           "need_name": False, "last_time": False, "trigger": "the table is full", "question": f"Question {id_}?",
           "answer": "B", "target_time": {"date": date_text or f"DAY{day}", **when}, "keywords": "use tool",
           "reason": "Saw someone with a tool"}
    out.update({f"choice_{k.lower()}": v for k, v in OPTIONS.items()})
    return out


ITEMS = [
    raw("1", "EntityLog", (1, "11:12:10.17"), [(1, "11:10:45.00")]),  # 0-2h; H 11:12:00
    raw("18", "EventRecall", (1, "11:10:50.00"), [(1, "11:10:05.00")]),  # 0-2h; H 11:10:30, before the evidence clip ends
    raw("2", "RelationMap", (1, "11:11:40.00"), [(1, "11:09:55.00")], date_text="Day1"),  # H 11:11:30; the date as "Day1"
    raw("12", "RelationMap", (1, "12:30:20.00"), [(1, "11:11:10.00")]),  # in the short clip after the gap: H 11:12:30
    raw("15", "RelationMap", (1, "11:12:20.00"), [(1, "11:09:50.00"), (1, "11:11:20.00")], listed=True),  # two targets, one range
    raw("17", "RelationMap", (1, "11:11:50.00"), [(1, "11:10:05.00")]),
    raw("3", "EntityLog", (1, "14:00:00.00"), [(1, "11:10:20.00")]),  # 2-6h; H 12:33:00
    raw("6", "HabitInsight", (1, "15:30:00.00"), [(1, "11:10:40.00"), (1, "12:30:45.00")], listed=True),  # two ranges
    raw("4", "RelationMap", (2, "09:00:00.00"), [(1, "11:10:50.00")]),  # 6-24h
    raw("20", "TaskMaster", (2, "09:00:00.00"), [(1, "11:11:15.00")]),
    raw("5", "HabitInsight", (2, "15:21:05.00"), [(1, "11:11:10.00")]),  # 24h+; H 15:21:00, the end of a clip
    raw("21", "EventRecall", (2, "15:21:00.00"), [(1, "12:30:50.00")]),
    raw("8", "TaskMaster", (1, "12:40:00.00"), [(1, "11:10:46.00")], audio=True),  # speech, 0-2h
    raw("9", "EventRecall", (2, "15:21:00.00"), [(1, "12:31:15.00")], audio=True),  # speech, 24h+
    raw("10", "EntityLog", (1, "13:00:00.00"), [(1, "11:30:00.00")]),  # dropped: the target is in the gap
    raw("11", "RelationMap", (4, "10:00:00.00"), [(4, "09:00:00.00")]),  # dropped: after the last day
    raw("13", "HabitInsight", (1, "13:00:00.00"), [(1, "11:15:34.17")], time_text="11153417DAY1_11181201"),  # dropped: a range
    raw("14", "TaskMaster", (2, "10:07:55.00"), [(2, "10:06:40.00")], audio=True)]  # dropped: no line within 30 s of the target


ORACLES = {i: {"source": plans.VIDEO, "question": SHADE} for i in ("1", "18", "2", "12", "15", "17", "3", "6", "4", "20", "5", "21")}


ORACLES |= {i: {"source": plans.TEXT, "question": WORD} for i in ("8", "9")}


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    """The fake download, made once: `A1_JAKE/DAYn/*.mp4`, `EgoLifeQA/*.json` and `EgoLifeCap/Transcript/A1_JAKE/DAYn/*.srt`."""
    root = tmp_path_factory.mktemp("download")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    jobs = []
    for day, clips in CLIPS.items():
        (root / "A1_JAKE" / f"DAY{day}").mkdir(parents=True)
        jobs += [(root / "A1_JAKE" / f"DAY{day}" / f"DAY{day}_A1_JAKE_{name}.mp4", seconds, name == EVIDENCE and day == 1)
                 for name, seconds in clips]
    with ThreadPoolExecutor(3) as pool:
        for future in [pool.submit(make_clip, ffmpeg, path, seconds, n, evidence) for n, (path, seconds, evidence) in enumerate(jobs)]:
            future.result()
    (root / "EgoLifeQA").mkdir()
    (root / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    for (day, hour), entries in SRT.items():
        folder = root / "EgoLifeCap" / "Transcript" / "A1_JAKE" / f"DAY{day}"
        folder.mkdir(parents=True, exist_ok=True)
        blocks = [f"{n}\n00:{a} --> 00:{b}\n" + "\n".join(lines) for n, (a, b, lines) in enumerate(entries, 1)]
        (folder / f"A1_JAKE_DAY{day}_{hour:02d}000000.srt").write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module", autouse=True)
def oracle_file(tmp_path_factory):
    """The oracles file the module's plans read: `plans.ORACLES` points at it."""
    path = tmp_path_factory.mktemp("oracles") / "oracles.json"
    path.write_text(json.dumps({"version": plans.ORACLE_VERSION, "items": ORACLES}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plans, "ORACLES", path)
        yield path


@pytest.fixture(scope="module")
def table(source) -> list[src.Clip]:
    return src.clip_table(source, 2)


@pytest.fixture(scope="module")
def recipe(source) -> dict:
    """The recipe of the fake download: 2 questions of each age and 2 of speech (deterministic: see ITEMS)."""
    return V8.recipe(source=source, seed=1, days=DAYS, per_age=2, speech=2)


@pytest.fixture(scope="module")
def plan(recipe) -> plans.Plan:
    return V8.plan_of(recipe)


@pytest.fixture(scope="module")
def built(tmp_path_factory, recipe) -> tuple[Dump, tuple]:
    """The dump the recipe makes, in a folder on the download's volume, and its questions. No daemon, no snapshot: that is mm's. The
    folder is named as a run names a dump: a bare `data`, denied by name when hidden, is in any path through AppData (pytest's own temp)."""
    return V8.build(recipe, tmp_path_factory.mktemp("dump") / "mmb-0123456789-data", 2)


def clip_ends(table: list[src.Clip]) -> list[float]:
    return [plans.clip_end(DAYS, c) for c in table]


def view_files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def offset_of(day: date) -> str:
    """The machine's offset from UTC at local midnight on `day`, from the C library and not from datetime."""
    local = time.localtime(time.mktime((day.year, day.month, day.day, 0, 0, 0, 0, 0, -1)))
    minutes = local.tm_gmtoff // 60
    return f"{'+' if minutes >= 0 else '-'}{abs(minutes) // 60:02d}{abs(minutes) % 60:02d}"


# -- the scenario ------------------------------------------------------------------------------------------------------------------


def write_run(folder: Path, recipe: dict, questions, dump: Dump, temp: Path, tag: str = "t8", configs=(BLIND, FILES, STUB)) -> RunFolder:
    """What U8's build will write: the question file, and a run.json that holds what a RunFolder needs and V8's report reads."""
    write_questions(folder / "questions.jsonl", questions)
    plan_h, files_h = V8.hashes(recipe)
    fs.write_json(folder / "run.json", {
        "scenario": "v8_egolife", "generator": V8.generator, "recipe": recipe, "seed": 1, "scale": "egolife", "days": recipe["days"],
        "data": str(dump.root), "temp": str(temp), "tag": tag, "timeout_s": 900.0, "plan": plan_h, "files": files_h,
        "question_hash": fs.sha256_of(folder / "questions.jsonl"), "configs": [dataclasses.asdict(c) for c in configs],
        "probe": {"status": "none"}, "mmmb_commit": "abc1234", "decode_s": 0.024, "pools": V8.pool_counts(recipe)})
    return RunFolder.open(folder, scenarios.get)


def lines_by(plan: plans.Plan, h: float) -> list[dict]:
    return [r for rows in plan.logs.values() for r in rows if datetime.fromisoformat(r["end"]).timestamp() <= h]


def line_at(plan: plans.Plan, after: bool) -> dict:
    """A transcript line of the plan that ended after question ego-001's horizon H, or by it."""
    h = next(q for q in plan.questions if q["qid"] == "ego-001")["until"]
    return next(r for rows in plan.logs.values() for r in rows
                if (datetime.fromisoformat(r["start"]).timestamp() > h) == after and (datetime.fromisoformat(r["end"]).timestamp() > h) == after)
