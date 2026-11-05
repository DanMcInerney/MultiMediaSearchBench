"""Ported from tests/test_egolife.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order: V8, EgoLifeQA without Multimem, over a
two-day fake download (tests/v8_fake.py) that is read, selected from, planned and built into a dump with its questions. What an agent with its
own tools reads is tests/test_v8_view.py, the run loop and the report tests/test_v8_run.py. No real agent or model runs, and no daemon: the
old `built` fixture started one to load a snapshot, and is the dump (`built`) and the run over it (`ran` in test_v8_run.py) here.
What moved, by the plan's 3.4, 4.5 and 5.2:
- `EgoRun` is `RunFolder` with the scenario `v8_egolife`; a plan question is a `Question` (questions.py); the old `built.plan()` is
  `V8.plan_of(recipe)` and the old `bench.build` is `V8.build(recipe, out)`, which links the clips and writes the transcript into the dump
  but does not freeze it (that is the cache's) or load a snapshot (that is mm's);
- the tests that need the daemon, the index or the home are not here but in tests/mm (U7), and the command line's and the timeout's belong
  to the CLI (U8). Both lists are in the unit's report.
Every expected value is the source's or the plan's; the only changes are `arm` to `config`, the names above, the messages of
`RunFolder.open` (U1) for those of `EgoRun.open`, the `timeout_s` that the runner writes now and not `extra`, and the assertions on
a frozen dump, which are the cache's."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from v8_fake import (DAYS, ITEMS, ORACLES, OPTIONS, SHADE, V8, built, clip_ends, ep, offset_of, oracle_file,  # noqa: F401
                     pl, plan, recipe, source, table, view_files, write_run)

from mmmb import scenarios
from mmmb.core import fs
from mmmb.core.contract import LogSource
from mmmb.core.question import load_questions, write_questions
from mmmb.core.runfolder import RunFolder
from mmmb.core.scenario import ProbeOutcome, dump_key, files_hash, plan_hash
from mmmb.scenarios.v8_egolife import choose, manifest, questions as writer, source as src
from mmmb.scenarios.v8_egolife import plan as plans


pytestmark = pytest.mark.xdist_group("v8")


V8 = scenarios.get("v8_egolife")


pytestmark = pytest.mark.xdist_group("v8")


def test_the_registry_has_the_scenario_by_its_name_and_it_has_the_names_of_a_scenario():
    assert scenarios.get("v8_egolife") is sys.modules["mmmb.scenarios.v8_egolife"]
    assert (V8.name, V8.code, V8.generator) == ("v8_egolife", "V8", 1)
    assert all(callable(getattr(V8, name)) for name in ("recipe", "hashes", "build", "probe", "prompt", "view", "deny_words", "after_exec",
                                                        "release", "report"))
    assert V8.probe(None, None, None, None, 1) == ProbeOutcome("none", ()), "the recorded run's run.json says `none`: no probe"


# -- 1. parsing -------------------------------------------------------------------------------------------------------------------


def test_times_are_hundredths_and_days_are_numbers():
    assert src.clock_of("11210217") == pytest.approx(11 * 3600 + 21 * 60 + 2.17)
    assert src.day_of("DAY3") == 3 and src.day_of("Day1") == 1
    assert src.plain(2, 10.5) == 2 * 86400 + 10.5
    for bad in ("1121021", "11153417DAY1_11181201"):
        with pytest.raises(ValueError, match="HHMMSSFF"):
            src.clock_of(bad)
    with pytest.raises(ValueError, match="DAY and a number"):
        src.day_of("Tuesday")
    # a plain time is a local epoch of the day it names, kept to the hundredth, and a local ISO time to the millisecond
    t = ep(2, "10:05:01.25")
    assert datetime.fromtimestamp(t) == datetime(2026, 10, 30, 10, 5, 1, 250000)
    assert plans.iso_of(DAYS, pl(2, "10:05:01.25")) == "2026-10-30T10:05:01.250"
    assert plans.iso_of(DAYS, pl(1, "11:09:47.10")) == "2026-10-29T11:09:47.100"


def test_the_items_are_read_as_the_data_gives_them(source):
    items = {i["id"]: i for i in src.read_items(source)}
    assert len(items) == len(ITEMS)
    one = items["1"]
    assert one["query"] == pytest.approx(pl(1, "11:12:10.17")) and one["targets"] == [pytest.approx(pl(1, "11:10:45.00"))]
    assert (one["category"], one["audio"], one["answer"], one["options"]) == ("EntityLog", False, "B", OPTIONS)
    assert items["2"]["targets"] == [pytest.approx(pl(1, "11:09:55.00"))], "`Day1` is DAY1"
    assert items["6"]["targets"] == [pytest.approx(pl(1, "11:10:40.00")), pytest.approx(pl(1, "12:30:45.00"))], "a time_list, in order"
    assert items["8"]["audio"] is True and items["11"]["query"] > src.plain(4, 0)
    assert items["13"]["targets"] == [], "a range in one string is no moment"


def test_the_transcript_is_read_from_srt_entries(source):
    lines = src.read_transcripts(source, 2)
    assert len(lines) == 13 and [x["end"] for x in lines] == sorted(x["end"] for x in lines)
    by_en = {x["en"]: x for x in lines}
    first = by_en["Okay, then we need a stopwatch."]
    assert (first["speaker"], first["zh"]) == ("Jake", "好然后一个秒表")
    assert first["start"] == pytest.approx(pl(1, "11:09:47.10")) and first["end"] == pytest.approx(pl(1, "11:09:50.766")), \
        "times are offsets from the hour of the file's name"
    assert (by_en["Mark a timestamp, right?"]["speaker"], by_en["Mark a timestamp, right?"]["zh"]) == ("Shure", "时间戳呗"), "two spaces"
    glued = by_en["I use the screwdriver to tighten screws."]
    assert (glued["speaker"], glued["zh"]) == ("Alice", "我用螺丝刀拧螺丝"), "no space after the colon"
    assert pl(1, "11:10:44.00") <= by_en["OK."]["start"] <= glued["end"], "entries overlap"
    one = next(x for x in lines if x["zh"] == "要稍微使点力气")
    assert (one["speaker"], one["en"]) == ("Lucia", ""), "an entry of one line has no English"
    bare = next(x for x in lines if x["zh"] == "小薇;哪个")
    assert (bare["speaker"], bare["en"]) == ("", "")
    assert [x["end"] for x in src.read_transcripts(source, 1)] == [x["end"] for x in lines if x["end"] < src.plain(2, 0)]
    assert plans.log_lines(DAYS, lines)["2026-10-29"][0]["start"] == "2026-10-29T11:09:47.100"


def test_the_clip_table_has_the_clips_in_time_order(table):
    assert len(table) == 24 and [c.t0 for c in table] == sorted(c.t0 for c in table)
    first = table[0]
    assert first.rel == "A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4" and first.day == 1
    assert first.t0 == pytest.approx(pl(1, "11:09:42.08")) and 17.9 < first.length < 18.2, "an 18 s clip at a name in hundredths"
    assert 29.9 < table[1].length < 30.2 and table[1].t0 == pytest.approx(pl(1, "11:10:00.00"))
    assert table[-1].day == 2 and table[-1].t0 == pytest.approx(pl(2, "15:22:00.00")) and table[-1].t1 == table[-1].t0 + table[-1].length


def test_a_clip_is_named_by_its_path_in_the_download_and_a_file_that_is_no_clip_or_no_media_is_refused(source, table, tmp_path):
    for c in table:
        assert src.clip_of(c.rel, c.length) == c, "the table is the clips of its own paths and lengths"
    for bad in ("A1_JAKE/DAY2/DAY1_A1_JAKE_11094208.mp4", "A1_JAKE/DAY1/DAY1_A1_JAKE_1109420.mp4", "A1_JAKE/DAY1/x/DAY1_A1_JAKE_11094208.mp4",
                "OTHER/DAY1/DAY1_A1_JAKE_11094208.mp4", "DAY1_A1_JAKE_11094208.mp4"):
        with pytest.raises(ValueError, match="is not a clip of the download"):
            src.clip_of(bad, 30.0)
    assert src.packet_duration(source / table[1].rel) == table[1].length and src.packet_duration(tmp_path / "gone.mp4") is None
    (tmp_path / "text.mp4").write_text("not a video", encoding="utf-8")
    assert src.packet_duration(tmp_path / "text.mp4") is None
    copy = tmp_path / "copy"
    for day in (1, 2):
        (copy / "A1_JAKE" / f"DAY{day}").mkdir(parents=True)
    shutil.copy2(source / table[0].rel, copy / table[0].rel)
    shutil.copy2(tmp_path / "text.mp4", copy / "A1_JAKE" / "DAY2" / "DAY2_A1_JAKE_10050125.mp4")
    with pytest.raises(ValueError, match="DAY2_A1_JAKE_10050125.mp4 is not media this machine can read"):
        src.clip_table(copy, 2)


# -- 2. the horizon ---------------------------------------------------------------------------------------------------------------


def test_the_horizon_is_the_end_of_the_last_whole_clip(table):
    end = lambda day, text: choose.horizon(table, pl(day, text))  # noqa: E731
    assert end(1, "11:12:10.17").t1 == pytest.approx(pl(1, "11:12:00.00"), abs=0.2)
    assert end(1, "11:12:10.17").t0 == pytest.approx(pl(1, "11:11:30.00"))
    assert end(1, "12:40:00.00").t1 == pytest.approx(pl(1, "12:33:00.00"), abs=0.2), "after a block, its last clip"
    # a query in the short first clip after a gap gets the clip before the gap
    assert end(1, "12:30:20.00").t1 == pytest.approx(pl(1, "11:12:30.00"), abs=0.2)
    # the clip that ended exactly at the query counts
    clip = choose.horizon(table, table[3].t1)
    assert clip == table[3]
    assert choose.horizon(table, table[3].t1 - 0.001) == table[2]
    assert choose.horizon(table, pl(1, "11:00:00.00")) is None, "no clip had ended"
    assert choose.horizon(table, pl(2, "23:00:00.00")) == table[-1]


def test_every_question_is_asked_as_of_a_clip_end_after_its_targets(plan, table, built):
    ends = clip_ends(table)
    assert len(plan.questions) == 10 and {q["qid"] for q in plan.questions} == {f"ego-{i:03d}" for i in (1, 18, 3, 6, 4, 20, 5, 21, 8, 9)}
    for q in plan.questions:
        assert q["until"] in ends, "H is exactly the end mm stores for a clip"
        assert q["until"] <= q["query"] and all(m[0] <= q["until"] for m in q["truth"]["moments"])
        assert q["since"] == plan.files[0].t
    by = {q["qid"]: q for q in plan.questions}
    assert by["ego-001"]["until"] == pytest.approx(ep(1, "11:12:00.00"), abs=0.2)
    assert by["ego-018"]["until"] == pytest.approx(ep(1, "11:10:30.00"), abs=0.2), "the evidence clip has not ended by this question"
    assert by["ego-005"]["until"] == pytest.approx(ep(2, "15:21:00.00"), abs=0.2) and by["ego-005"]["until"] < by["ego-005"]["query"]
    dump, questions = built
    assert {e.t + e.length for e in dump.entries if e.kind == "clip"} == set(ends), "and the dump's clips end exactly there"
    assert {q.horizon for q in questions} <= set(ends) and all(q.scope.until == q.horizon <= datetime.fromisoformat(q.now).timestamp()
                                                                for q in questions)


# -- 3. the selection -------------------------------------------------------------------------------------------------------------


IDS = itertools.count(1000)


def synthetic(bin_: str, category: str, n: int) -> list[dict]:
    """`n` items of an age bin and category, with targets inside the first block of day 1."""
    query = {"0-2h": pl(1, "11:11:50.00"), "2-6h": pl(1, "14:00:00.00"), "6-24h": pl(2, "09:00:00.00"), "24h+": pl(2, "15:21:00.00")}[bin_]
    return [{"id": str(next(IDS)), "category": category, "audio": False, "need_name": False, "last": False, "query": query,
             "targets": [pl(1, "11:10:05.00")], "question": "?", "options": OPTIONS, "answer": "B", "keywords": "", "reason": ""}
            for k in range(n)]


def test_the_pools_count_what_is_dropped_and_why(source, table):
    items, lines = src.read_items(source), src.read_transcripts(source, 2)
    video, speech, dropped = choose.pools(items, table, lines, 2)
    assert {i["id"] for i in video} == {"1", "18", "2", "12", "15", "17", "3", "6", "4", "20", "5", "21"}
    assert {i["id"] for i in speech} == {"8", "9"}
    assert dict(dropped) == {"its query is after the last day": 1, "its time is a range or a list in one string": 1,
                             "a target is outside the clips (a gap) or after the horizon": 1,
                             "speech: no transcript line near a target": 1}
    assert [choose.age_of(i) for i in video if i["id"] in ("1", "3", "4", "5")] == ["0-2h", "2-6h", "6-24h", "24h+"]


def test_select_takes_the_bins_in_order_and_the_speech_apart(source, table):
    items, lines = src.read_items(source), src.read_transcripts(source, 2)
    one = choose.select(items, table, lines, 1, per_age=1, speech=1, days=2)
    assert [(i["id"], i["age"]) for i in one] == [("1", "0-2h"), ("3", "2-6h"), ("4", "6-24h"), ("21", "24h+"), ("8", "0-2h")], \
        "the rarest category of each bin (ties by name), then the speech question"
    assert [i["audio"] for i in one] == [False] * 4 + [True]
    two = choose.select(items, table, lines, 1, per_age=2, speech=2, days=2)
    assert [i["id"] for i in two] == ["1", "18", "3", "6", "4", "20", "21", "5", "8", "9"] and all(i["pilot"] for i in two)
    assert choose.select(items, table, lines, 1, per_age=2, speech=2, days=2) == two
    with pytest.raises(ValueError, match="2-6h age bin holds 2 questions .* fewer than 3"):
        choose.select(items, table, lines, 1, per_age=3, speech=1, days=2)
    with pytest.raises(ValueError, match="speech pool holds 2 questions, fewer than 3"):
        choose.select(items, table, lines, 1, per_age=1, speech=3, days=2)


def test_a_bin_is_taken_from_its_categories_in_turn_by_seed(table):
    queues = choose.by_category([*synthetic("0-2h", "RelationMap", 5), *synthetic("0-2h", "EventRecall", 2),
                                 *synthetic("0-2h", "EntityLog", 3)])
    assert [q[0]["category"] for q in queues] == ["EventRecall", "EntityLog", "RelationMap"], "the rarest first"
    assert [i["category"] for i in choose.rotate(queues, 7)] == ["EventRecall", "EntityLog", "RelationMap", "EventRecall", "EntityLog",
                                                                 "RelationMap", "EntityLog"], "in turn, a short queue skipped"
    pool = [i for b in choose.AGE_NAMES for c, n in (("RelationMap", 12), ("EventRecall", 3), ("EntityLog", 8)) for i in synthetic(b, c, n)]
    chosen = {seed: choose.select(pool, table, [], seed, per_age=6, speech=0, days=2) for seed in (1, 1, 2, 3)}
    assert chosen[1] == choose.select(pool, table, [], 1, per_age=6, speech=0, days=2), "the same seed, the same questions"
    assert any([i["id"] for i in chosen[s]] != [i["id"] for i in chosen[1]] for s in (2, 3)), "another seed, others"
    for items in chosen.values():
        for name in choose.AGE_NAMES:
            of = [i for i in items if i["age"] == name]
            assert [i["category"] for i in of] == ["EventRecall", "EntityLog", "RelationMap"] * 2 and len(of) == 6
            assert [i["pilot"] for i in of] == [True, True, False, False, False, False], "the first two of a bin are the pilot's"


def test_a_chosen_question_with_no_oracle_or_a_leaking_one_is_refused_by_its_id(source, table):
    days = DAYS
    without = {k: v for k, v in ORACLES.items() if k != "5"}
    with pytest.raises(ValueError, match="no oracle for question 5"):
        plans.plan(source, 1, days, table, without, 2, 2)
    base = {"source": plans.VIDEO, "question": SHADE}
    for change, message in (({"source": plans.TEXT}, "source 'transcript'"), ({"question": "Is someone there"}, "not a question"),
                            ({"question": "Is someone working tonight?"}, "time word"),
                            ({"question": "Is someone cooking or cleaning?"}, "'and' or 'or'"),
                            ({"question": "Is Alice using a tool?"}, "names Alice"),
                            ({"question": "Is a tool in view?", "source": "screen"}, "source 'screen'")):
        with pytest.raises(ValueError, match=rf"oracle of question 4 .*{message}"):
            plans.plan(source, 1, days, table, {**ORACLES, "4": {**base, **change}}, 2, 2)
    with pytest.raises(ValueError, match="oracle of question 9 .*source 'glasses'"):
        plans.plan(source, 1, days, table, {**ORACLES, "9": base}, 2, 2)
    with pytest.raises(ValueError, match=r"oracle of question 9 holds an option's words \('Alice'\)"):
        plans.plan(source, 1, days, table, {**ORACLES, "9": {"source": plans.TEXT, "question": "Does the text mention Alice?"}}, 2, 2)
    made = plans.plan(source, 1, days, table, ORACLES, 2, 2)
    assert made == plans.plan(source, 1, days, table, ORACLES, 2, 2) and made.scale == "egolife"
    assert json.loads(json.dumps(made.questions)) == list(made.questions)


def test_the_plan_names_the_clips_neutrally_and_holds_the_transcript(source, table):
    made = plans.plan(source, 1, DAYS, table, ORACLES, 2, 2)
    assert len(made.files) == 24 and made.files[0].path == "glasses/2026-10-29/11.09.42.mp4"
    assert made.files[0].name == "2026-10-29 11.09.42.mp4" and made.files[0].kind == "clip" and made.files[0].source == "glasses"
    assert made.files[0].t == ep(1, "11:09:42.08") and made.files[0].spec["src"] == table[0].rel
    assert not any("A1_JAKE" in f.path or "DAY" in f.path or "A1_JAKE" in f.name for f in made.files)
    assert sorted(made.logs) == ["2026-10-29", "2026-10-30"] and sum(len(v) for v in made.logs.values()) == 13
    q = next(q for q in made.questions if q["qid"] == "ego-001")
    assert (q["type"], q["template"], q["level"], q["source"], q["day"], q["absent"]) == ("video", "EntityLog", "0-2h", "glasses", 1, False)
    assert q["grade"] == {"kind": "choice", "truth": "B", "ignore": []} and q["options"] == OPTIONS
    assert q["truth"] == {"moments": [[ep(1, "11:10:45.00")] * 2], "others": []} and (q["oracle"], q["oracle_source"]) == (SHADE, "glasses")
    assert next(q for q in made.questions if q["qid"] == "ego-008")["source"] == "transcript"


# -- 4. the recipe and the build --------------------------------------------------------------------------------------------------


def test_a_recipe_holds_the_download_the_seed_the_days_the_utc_offset_and_the_clip_table_and_the_hashes_are_those_of_its_plan(
        recipe, plan, source, table):
    assert recipe == {"source": str(source), "scale": "egolife", "seed": 1, "days": ["2026-10-29", "2026-10-30"], "per_age": 2, "speech": 2,
                      "utc_offset": offset_of(DAYS[0]), "clips": [[c.rel, c.length] for c in table]}
    assert json.loads(json.dumps(recipe)) == recipe
    assert V8.hashes(recipe) == (plan_hash(plan.questions), files_hash(manifest.file_rows(plan), plan.logs))
    assert V8.hashes({**recipe, "per_age": 1})[0] != V8.hashes(recipe)[0] and V8.hashes({**recipe, "seed": 1}) == V8.hashes(recipe)
    assert V8.pool_counts(recipe) == {"video": 12, "speech": 2, "dropped": {
        "its query is after the last day": 1, "its time is a range or a list in one string": 1,
        "a target is outside the clips (a gap) or after the horizon": 1, "speech: no transcript line near a target": 1}}
    assert V8.plan_of(recipe) == plan


def test_a_recipe_with_other_keys_or_days_or_another_zone_or_no_download_is_refused(recipe, tmp_path):
    with pytest.raises(ValueError, match="has the keys clips, days, per_age, scale, seed, source, speech, utc_offset, not clips, days, per_age"):
        V8.plan_of({k: v for k, v in recipe.items() if k != "scale"})
    elsewhere = "+0100" if recipe["utc_offset"] != "+0100" else "+0200"
    with pytest.raises(ValueError, match=rf"the UTC offset of its days is {re.escape(elsewhere)}, here it is {recipe['utc_offset']}: .*another run"):
        V8.plan_of({**recipe, "utc_offset": elsewhere})
    with pytest.raises(ValueError, match="consecutive, oldest first"):
        V8.plan_of({**recipe, "days": ["2026-10-30", "2026-10-29"]})
    with pytest.raises(ValueError, match="clips are not in time order"):
        V8.plan_of({**recipe, "clips": recipe["clips"][::-1]})
    with pytest.raises(FileNotFoundError, match="the EgoLife download is not at .*nowhere"):
        V8.hashes({**recipe, "source": str(tmp_path / "nowhere")})
    with pytest.raises(ValueError, match="a run has days"):
        V8.recipe(source=tmp_path, seed=1, days=[])


def test_a_dump_is_built_with_neutral_names_the_downloads_clips_linked_and_a_manifest_measured_from_the_files(built, source, table):
    dump, questions = built
    names = view_files(dump.root)
    assert len(names) == 26 and not any(re.search(r"A1_JAKE|DAY\d|egolife", n, re.I) for n in names)
    assert {n.split("/")[0] for n in names} == {"glasses", "transcript"}
    first = dump.root / "glasses" / "2026-10-29" / "11.09.42.mp4"
    assert os.path.samefile(first, source / table[0].rel), "the data is the download's clips, linked"
    rows = [json.loads(line) for line in (dump.root / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and all(set(r) == {"start", "end", "speaker", "en", "zh"} for r in rows)
    assert rows[0]["start"] == "2026-10-29T11:09:47.100" and rows[0]["speaker"] == "Jake" and rows[0]["zh"] == "好然后一个秒表"
    assert [r["end"] for r in rows] == sorted(r["end"] for r in rows)
    assert (dump.scenario, dump.logs) == ("v8_egolife", (LogSource("transcript", "transcript", "*.jsonl", "end"),))
    for e in dump.entries:
        path = dump.root / e.path
        assert (e.size, e.mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns) and e.sha256 == fs.sha256_of(path), e.path
    assert sorted(e.path for e in dump.entries) == names, "the dump holds its files and no other"
    clips = [e for e in dump.entries if e.kind == "clip"]
    assert [(e.source, e.media, e.own_time, e.t, e.length) for e in clips] == [
        ("glasses", "video", False, plans.local(DAYS, c.t0), c.length) for c in table]
    assert (clips[0].path, clips[0].name) == ("glasses/2026-10-29/11.09.42.mp4", "2026-10-29 11.09.42.mp4")
    assert (clips[-1].path, clips[-1].name) == ("glasses/2026-10-30/15.22.00.mp4", "2026-10-30 15.22.00.mp4")
    assert all(e.name == e.path.split("/", 1)[1].replace("/", " ") for e in clips), "the name mm gets is the path less its source"
    days = [e for e in dump.entries if e.kind == "log"]
    assert [(e.path, e.source, e.media, e.name, e.t, e.length) for e in days] == [
        (f"transcript/{d.isoformat()}.jsonl", "transcript", "records", None, datetime(d.year, d.month, d.day).timestamp(), 86400.0) for d in DAYS]


def test_a_dump_is_built_in_a_folder_of_its_own_with_the_key_of_its_recipe(built, recipe, tmp_path):
    dump, questions = built
    assert dump.key == dump_key("v8_egolife", recipe, 1) and len(questions) == 10
    with pytest.raises(ValueError, match="is not empty: a dump is written into a folder of its own"):
        V8.build(recipe, dump.root, 2)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="is not empty"):
        V8.build(recipe, tmp_path / "other", 2)
    assert not (tmp_path / "other" / "glasses").exists(), "and nothing was written there"


def test_a_dump_whose_files_cannot_be_linked_to_the_download_is_refused_by_what_it_asks_for(recipe, tmp_path, monkeypatch):
    def refuse(a, b):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(ValueError, match="cannot link .* cross-device link; the dump goes in --temp, which must be on the download's volume"):
        V8.build(recipe, tmp_path / "data", 2)


def test_the_questions_a_build_returns_are_the_plans_written_and_read_back_with_evidence_in_the_dump(built, plan, tmp_path):
    dump, questions = built
    assert questions == writer.questions_of(plan, dump.entries) and [q.qid for q in questions] == [q["qid"] for q in plan.questions]
    write_questions(tmp_path / "questions.jsonl", questions)
    assert load_questions(tmp_path / "questions.jsonl") == questions
    by_path = {e.path: e for e in dump.entries}
    for q, p in zip(questions, plan.questions):
        assert (q.scenario, q.split, q.kind, q.hint, q.group, q.horizon) == ("V8", "dev", "choice", p["level"], p["qid"], p["until"])
        assert len(q.truth.evidence) == len(p["truth"]["moments"]) and q.truth.wrong == ()
        for span, (moment, _) in zip(q.truth.evidence, p["truth"]["moments"]):
            e = by_path[span.file]
            assert abs(e.t + span.t0 - moment) < 1e-3 and 0 <= span.t0 <= span.t1 <= e.length, "the moment, inside its file"
            assert e.kind == ("log" if p["source"] == "transcript" else "clip")
        assert q.cost_ranges == tuple(plans.oracle_ranges(p)) and q.retrieval.ranges == q.cost_ranges
        assert (q.retrieval.query, q.retrieval.sources, q.retrieval.tol_s) == (p["oracle"], (p["oracle_source"],), 30.0)
        assert q.grade.rule == "choice" and q.grade.truth == p["grade"]["truth"] == q.truth.value and q.options == p["options"]
    ego1 = next(q for q in questions if q.qid == "ego-001")
    assert ego1.truth.evidence[0].file == "glasses/2026-10-29/11.10.30.mp4" and ego1.truth.evidence[0].t0 == pytest.approx(15.0, abs=1e-3)
    assert ego1.now == "2026-10-29T11:12:10.170" + offset_of(DAYS[0])[:3] + ":" + offset_of(DAYS[0])[3:]
    assert dict(ego1.tags) == {"modality": "video", "template": "EntityLog", "level": "0-2h", "last": False, "absent": False, "age": "0-2h",
                               "category": "EntityLog", "lookback_h": 0.0, "rung": "2d", "pilot": True, "day": 1, "first_day": "2026-10-29"}
    speech = next(q for q in questions if q.qid == "ego-008")
    assert speech.truth.evidence[0].file == "transcript/2026-10-29.jsonl" and speech.scope.sources == ("transcript",)
    assert speech.tags["modality"] == "records" and speech.retrieval.sources == ("transcript",)


def test_a_moment_in_no_file_makes_the_writer_fail(plan, built):
    dump, _ = built
    bad = dict(plan.questions[0])
    bad["truth"] = {"moments": [[plan.files[0].t - 3600] * 2], "others": []}
    with pytest.raises(ValueError, match="ego-001: the moment .* is in no clip file of the glasses source"):
        writer.questions_of(dataclasses.replace(plan, questions=(bad,)), dump.entries)
    speech = dict(next(q for q in plan.questions if q["qid"] == "ego-008"))
    speech["truth"] = {"moments": [[ep(1, "11:10:46.00") + 5 * 86400] * 2], "others": []}
    with pytest.raises(ValueError, match="ego-008: the moment .* is in no log file of the transcript source"):
        writer.questions_of(dataclasses.replace(plan, questions=(speech,)), dump.entries)


def test_a_run_is_opened_and_refused_when_its_plan_or_generator_changed_or_the_download_is_gone(built, recipe, tmp_path, monkeypatch):
    dump, questions = built
    run = write_run(tmp_path / "run", recipe, questions, dump, tmp_path / "temp")
    assert (run.name, run.seed, run.scale, run.data, run.scenario, run.days) == ("run", 1, "egolife", dump.root, V8, DAYS)
    assert run.questions() == questions and [c.name for c in run.configs] == ["blind", "files", "stub/all"]
    changed = tmp_path / "oracles.json"
    changed.write_text(json.dumps({"version": plans.ORACLE_VERSION, "items": {
        **ORACLES, "1": {"source": "glasses", "question": "Is a red frame in view?"}}}), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(plans, "ORACLES", changed)
        with pytest.raises(ValueError, match="plan is not the one its run.json recorded: v8_egolife changed without a generator bump"):
            RunFolder.open(run.folder, scenarios.get)
    with monkeypatch.context() as patch:
        patch.setattr(V8, "generator", V8.generator + 1)
        with pytest.raises(ValueError, match="was made by generator 1 of v8_egolife, this is 2"):
            RunFolder.open(run.folder, scenarios.get)
    meta = fs.read_json(run.folder / "run.json")
    fs.write_json(run.folder / "run.json", {**meta, "recipe": {**recipe, "source": str(tmp_path / "gone")}})
    with pytest.raises(FileNotFoundError, match="the EgoLife download is not at"):
        RunFolder.open(run.folder, scenarios.get)


# -- 5. the retrieval tier's ranges ----------------------------------------------------------------------------------------------


def test_oracle_ranges_are_capped_at_h_and_merged():
    ranges = lambda until, *moments: plans.oracle_ranges({"until": until, "truth": {"moments": [[m, m] for m in moments]}})  # noqa: E731
    assert ranges(1000.0, 900.0) == [(600.0, 1000.0)]
    assert ranges(1000.0, 500.0, 700.0) == [(200.0, 1000.0)], "overlapping ranges are one"
    assert ranges(3000.0, 400.0, 2000.0) == [(100.0, 700.0), (1700.0, 2300.0)], "far apart ranges are two"


def test_the_oracles_file_follows_the_rules_a_machine_can_check():
    data = json.loads(Path(plans.__file__).with_name("oracles.json").read_text(encoding="utf-8"))
    assert data["version"] == plans.ORACLE_VERSION and len(data["items"]) == 50
    assert Counter(o["source"] for o in data["items"].values()) == {"glasses": 40, "transcript": 10}
    for id_, o in data["items"].items():
        assert set(o) == {"source", "question"} and o["question"].endswith("?") and o["question"].count("?") == 1, id_
        assert not plans.TIME_WORDS.search(o["question"]) and not plans.JOINED.search(o["question"]), id_
        assert not re.search(r"\b(now|just|last|first|usually|always|often|ago)\b", o["question"], re.I), f"{id_}: a time word"
        if o["source"] == "glasses":
            assert not any(re.search(rf"\b{p}\b", o["question"], re.I) for p in plans.PEOPLE), f"{id_}: names a person"
            assert re.match(r"(Is|Are) ", o["question"]), id_
        else:
            assert o["question"].startswith("Does the text talk about "), id_


# -- the review's cases ----------------------------------------------------------------------------------------------------------


def test_the_default_days_cross_no_change_of_the_clocks_and_a_recipe_refuses_days_that_do(source, tmp_path, monkeypatch):
    def offset(day: date) -> timedelta:  # the clocks go back during 1 November: the offset at its start is still summer's
        return timedelta(hours=-4) if day <= date(2026, 11, 1) else timedelta(hours=-5)

    days = lambda today: plans.default_days(today, offset=offset)  # noqa: E731
    assert days(date(2026, 11, 20)) == (date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)), "the latest three before today"
    assert days(date(2026, 11, 5)) == (date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4))
    assert days(date(2026, 11, 3)) == (date(2026, 10, 29), date(2026, 10, 30), date(2026, 10, 31)), "the nights of 31 Oct to 2 Nov cross it"
    assert days(date(2026, 10, 20)) == (date(2026, 10, 17), date(2026, 10, 18), date(2026, 10, 19))
    assert plans.default_days(date(2026, 11, 20)) and plans.default_days(date(2026, 11, 20), n=1) == (date(2026, 11, 19),)
    for ok in ((date(2026, 10, 29), date(2026, 10, 30)), (date(2026, 11, 2), date(2026, 11, 3))):
        plans.check_days(ok, offset)
    for crossing in ((date(2026, 10, 31), date(2026, 11, 1)), (date(2026, 11, 1),), (date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 1))):
        with pytest.raises(ValueError, match="cross a change of the local UTC offset"):
            plans.check_days(crossing, offset)
    with pytest.raises(ValueError, match="consecutive, oldest first"):
        plans.check_days((date(2026, 10, 30), date(2026, 10, 29)), offset)
    with pytest.raises(ValueError, match="consecutive, oldest first"):
        plans.check_days((date(2026, 10, 29), date(2026, 10, 31)), offset)
    monkeypatch.setattr(plans, "utc_offset", offset)
    with pytest.raises(ValueError, match="cross a change of the local UTC offset"):
        V8.recipe(source=source, seed=1, days=(date(2026, 10, 31), date(2026, 11, 1)), per_age=2, speech=2)
    monkeypatch.undo()
    assert plans.utc_offset(date(2026, 7, 1)) is not None and plans.crosses_a_change((date(2026, 7, 1), date(2026, 7, 2))) is False


def test_an_oracle_may_not_borrow_a_word_of_an_option_that_its_question_lacks(source, table):
    item = {"id": "1", "audio": False, "question": "Who washed the dishes?",
            "options": {"A": "Wash dishes", "B": "Dry cups", "C": "Sweep", "D": "Me"}}
    ok = {"source": "glasses", "question": "Are dishes being washed?"}
    plans.check_oracle(item, ok)  # `dishes` is in the question too
    with pytest.raises(ValueError, match="uses 'cups', a word of an option that the question does not use"):
        plans.check_oracle(item, {**ok, "question": "Are cups in view?"})
    plans.check_oracle(item, {**ok, "question": "Is someone drying a plate?"})  # words are compared whole, and `dry` is under 4 letters
    spoken_item = {**item, "audio": True, "question": "When was AI mentioned?",
                   "options": {"A": "About two hours ago", "B": "About a day ago", "C": "x", "D": "y"}}
    plans.check_oracle(spoken_item, {"source": "transcript", "question": plans.TEXT_ASK + "AI?"})  # the template's `about` is no borrowing
    with pytest.raises(ValueError, match="uses 'hours'"):
        plans.check_oracle(spoken_item, {"source": "transcript", "question": plans.TEXT_ASK + "hours?"})
    # the whole plan still passes with the rule on
    assert plans.plan(source, 1, DAYS, table, ORACLES, 2, 2)
