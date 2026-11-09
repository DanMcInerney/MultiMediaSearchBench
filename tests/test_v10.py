"""V10, phone clips at home, from the real labels of Charades on disk (`Charades.zip`; the videos zip is faked here with empty members, so that the
plan is made without 16 GB): both scales planned, the pile and every answer checked again by a plain reader of the CSV (never the scenario's): that
the clips of the pile with the label of a topic are exactly the clips the question asks for, that no clip of an absent topic has its label, its words
or its objects, that a room's count is the scene's, and that a `long` answer is a labelled span. The grader takes each truth and refuses planted wrong
answers. With the real videos zip complete, the tiny pile is built too (clips copied out whole, their lengths the CSV's to a fraction of a second) and
its probe passes. Skips, and says so, without the source."""

from __future__ import annotations

import csv
import functools
import io
import os
import re
import zipfile
from datetime import datetime

import av
import pytest

from mmsb.core import grade, taxonomy
from mmsb.core.question import load_questions, write_questions
from mmsb.scenarios import realdump
from mmsb.scenarios import v10_home_clips as V10
from mmsb.scenarios.v10_home_clips import pile as P
from mmsb.scenarios.v10_home_clips import source as S

pytestmark = pytest.mark.xdist_group("v10")
HAVE = (S.SOURCE / S.LABELS).is_file()
needs_labels = pytest.mark.skipif(not HAVE, reason=f"the Charades labels are not at {S.SOURCE}, so V10 is not planned")
SCENE_WORDS = {"Stairs": r"\bstair", "Garage": r"\bgarage", "Bathroom": r"\bbath ?room|\btoilet|\brestroom", "Basement": r"\bbasement",
               "Laundry room": r"\blaundry"}  # how a script names each room, written here
ABSENT_WORDS = {"fridge_open": ("c143", "c142", r"fridge|refrigerator"), "medicine_take": ("c129", "c128", r"medicine|medication|\bpills?\b"),
                "hair_fix": ("c144", None, r"\bhair\b"), "vacuum_use": ("c136", "c137", r"vacuum"), "tv_laugh": ("c131", "c132", r"\btv\b|television")}
OBJECTS = {"fridge_open": {"refrigerator"}, "medicine_take": {"medicine"}, "hair_fix": {"hair"}, "vacuum_use": {"vacuum"}, "tv_laugh": {"television", "tv"}}


@functools.cache
def raw() -> dict[str, dict]:
    """The rows of the CSVs of the source, as a second reader: by id."""
    out = {}
    with zipfile.ZipFile(S.SOURCE / S.LABELS) as z:
        for name in S.TABLES:
            for row in csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8", newline="")):
                out[row["id"]] = row
    return out


def labelled(row: dict) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for token in filter(None, row["actions"].split(";")):
        c, a, b = token.split()
        out.setdefault(c, []).append((float(a), float(b)))
    return out


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A source folder with the real labels and a videos zip that has an empty member for every video."""
    folder = tmp_path_factory.mktemp("charades")
    (folder / S.LABELS).write_bytes((S.SOURCE / S.LABELS).read_bytes())
    with zipfile.ZipFile(folder / S.VIDEOS, "w") as z:
        for vid in raw():
            z.writestr(S.MEMBER.format(vid), b"")
    return str(folder)


@pytest.fixture(scope="module")
def planned(source):
    return {(scale, seed): V10.plan_of(V10.recipe(scale=scale, seed=seed, source=source)) for scale in ("tiny", "standard") for seed in (1, 2, 3)}


def by_name(files, rows: dict[str, dict] | None = None):
    """The pile's clips as `(file, row)`, by file name, from the member each file is."""
    rows = rows or raw()
    return {f.path.rsplit("/", 1)[-1]: (f, rows[re.search(r"/(\w+)\.mp4", "/" + f.spec["member"])[1]]) for f in files}


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "absent": lambda: "unknown", "amount": lambda: g.truth,
            "time": lambda: f"{g.date} {g.time[:5]}", "text": lambda: g.truth.split("|")[0]}[g.rule]()


def wrong(q) -> list[str]:
    g = q.grade
    if g.rule == "set":
        return [", ".join([*g.items, "VID_20300101_000000.mp4"]), ", ".join([*g.items[:-1], "VID_20300101_000000.mp4"]), "unknown"]
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "amount":
        return [str(float(g.truth) + g.tol_s + 2), "unknown"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date}T{g.time}")
        return [f"{g.date} {t.replace(minute=(t.minute + 5) % 60):%H:%M}", "unknown"]
    if g.rule == "text":
        return ["unknown"] + ([g.wrong[0]] if g.wrong else [])
    return ["2025-03-03 10:15", "2025-03-03"]


@needs_labels
def test_the_reader_has_every_video_of_the_source_and_its_labels():
    clips = S.read_clips(str(S.SOURCE))
    assert len(clips) == len(raw()) == 9848 and {c.id for c in clips} == set(raw())
    by_id = {c.id: c for c in clips}
    for vid in ("N11GT", "46GP8"):
        row, c = raw()[vid], by_id[vid]
        assert c.scene == S.scene_of(row["scene"]) and c.length == float(row["length"]) and not c.flawed
        assert {k: [(a, b) for x, a, b in c.spans if x == k] for k in {x for x, _, _ in c.spans}} == {k: [(a, min(b, c.length)) for a, b in v] for k, v in labelled(row).items()}
    assert sum(c.flawed for c in clips) == 4, "four rows have an action that ends before it starts"
    assert len(S.class_names(str(S.SOURCE))) == 157 and "refrigerator" in S.object_names(str(S.SOURCE))


@needs_labels
@pytest.mark.parametrize("scale, n, quota", [("tiny", 72, {"one": 3, "few": 1, "many": 1, "none": 2}), ("standard", 3000, {"one": 7, "few": 5, "many": 5, "none": 5})])
def test_the_pile_is_the_pile_of_its_scale_whatever_the_seed(planned, scale, n, quota):
    files = planned[scale, 1][0]
    assert len(files) == n and all(planned[scale, s][0] == files for s in (2, 3))
    pile = P.choose(S.read_clips(str(S.SOURCE)), scale)
    assert (len(pile.one), len(pile.few), len(pile.many), len(pile.none)) == (quota["one"], quota["few"], quota["many"], quota["none"])
    assert len({f.spec["member"] for f in files}) == n and len({f.path for f in files}) == n


@needs_labels
@pytest.mark.parametrize("scale", ["tiny", "standard"])
def test_the_pile_holds_of_each_topic_exactly_the_clips_a_question_asks_for(planned, scale):
    files = planned[scale, 1][0]
    pile = P.choose(S.read_clips(str(S.SOURCE)), scale)
    members = {f.spec["member"] for f in files}
    rows = [raw()[m.split("/")[1][:-4]] for m in members]
    for key, clip in pile.one.items():
        label = P.TOPICS[key].labels[0]
        assert [r["id"] for r in rows if label in labelled(r)] == [clip.id], f"{key}: the one clip with the label"
        assert len(labelled(raw()[clip.id])[label]) == 1
    for key, clips in {**pile.few, **pile.many}.items():
        label = P.TOPICS[key].labels[0]
        assert {r["id"] for r in rows if label in labelled(r)} == {c.id for c in clips}, f"{key}: every clip with the label, and no other"
        assert all(re.search(P.TOPICS[key].text, raw()[c.id]["script"] + " . " + raw()[c.id]["descriptions"], re.I) for c in clips), "the words say it too"
    for key in pile.none:
        label, near, words = ABSENT_WORDS[key]
        for r in rows:
            assert label not in labelled(r) and (near is None or near not in labelled(r)) and not re.search(words, (r["script"] + " " + r["descriptions"]).lower()), (key, r["id"])
            assert not set(r["objects"].lower().split(";")) & OBJECTS[key], (key, r["id"])
    for room in pile.rooms:
        assert sum(r["scene"].startswith(room) for r in rows) == sum(c.scene == room for c in pile.clips)
        for r in rows:
            text = (r["script"] + " " + r["descriptions"]).lower()
            assert bool(re.search(SCENE_WORDS[room], text)) <= r["scene"].startswith(room), f"{room}: a clip that names it in another scene ({r['id']})"


@needs_labels
def test_every_clip_of_the_pile_is_verified_watchable_and_has_an_action_that_is_not_flawed(planned):
    for scale in ("tiny", "standard"):
        for f, row in by_name(planned[scale, 1][0]).values():
            assert row["verified"] == "Yes" and int(row["quality"]) >= 3 and 8 <= float(row["length"]) <= 45 and row["actions"] and not row["scene"].startswith("Other")
            assert f.length == float(row["length"])


@needs_labels
@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3)), ("standard", (1, 2, 3))])
def test_every_question_is_the_labels_and_the_grader_takes_each_truth(planned, scale, seeds):
    files = planned[scale, 1][0]
    named = by_name(files)
    for seed in seeds:
        questions = planned[scale, seed][1]
        if scale == "standard":
            assert len(questions) == 23 and sum(q.kind == "absent" for q in questions) == 5, seed
            assert {V10.filing[q.tags["template"]][0] for q in questions} >= {"find", "find_all", "how_many", "who", "order", "did"}, seed
            absent = [q.qid.split("-", 2)[2] for q in questions if q.kind == "absent"]
            assert len(set(absent)) == len(absent) == 5, "a topic that is never in the pile is asked of once"
        else:
            assert 2 <= len(questions) <= 8 and sum(q.kind == "absent" for q in questions) == 2, seed
        assert len({q.qid for q in questions}) == len(questions) and {q.split for q in questions} == {"dev", "test"}
        for q in questions:
            assert grade.grade(q.grade, right(q)), (q.qid, right(q))
            for answer in wrong(q):
                assert not grade.grade(q.grade, answer), (q.qid, answer)
            for s in q.truth.evidence:
                f, row = named[s.file.rsplit("/", 1)[-1]]
                assert 0 <= s.t0 <= s.t1 <= float(row["length"]) + 1e-6, (q.qid, s)
            template = q.qid.removeprefix("absent-").split("-")[0]
            if template in ("find", "did", "long", "room", "before"):
                if q.kind == "absent":
                    continue
                (s,) = q.truth.evidence[:1]
                f, row = named[s.file.rsplit("/", 1)[-1]]
                key = q.qid.split("-", 1)[1]
                label = P.TOPICS[key].labels[0]
                a, b = labelled(row)[label][0]
                assert s.t0 == a and s.t1 == min(b, float(row["length"])), (q.qid, s)
                if template == "before":
                    two = q.truth.evidence[1]
                    classes = [x for x, v in labelled(row).items() if x in P.NAMED and x != label and len(v) == 1
                               and (v[0][0], min(v[0][1], float(row["length"]))) == (two.t0, two.t1)]
                    assert len(classes) == 1, q.qid
                    cls = classes[0]
                    assert P.NAMED[cls] in q.question and q.question.count(P.TOPICS[key].phrase) == 2, q.qid
                    first = (two.t1 <= a - 2.0)
                    assert first or two.t0 >= b + 2.0, f"{q.qid}: the two actions are not a clear 2 s apart"
                    assert q.grade.truth.startswith("before" if first else "after") and q.truth.value == ("before" if first else "after"), q.qid
                if template == "long":
                    a, b = a, min(b, float(row["length"]))
                if template == "did":
                    assert (q.grade.date, q.grade.time) == (f"{datetime.fromtimestamp(f.t):%Y-%m-%d}", f"{datetime.fromtimestamp(f.t):%H:%M:%S}")
                if template == "long":
                    b = min(b, float(row["length"]))
                    assert q.grade.truth == f"{b - a:.1f}" and q.grade.tol_s == max(3.0, round(0.3 * (b - a), 1)) and b - a >= 8
                if template == "room":
                    assert row["scene"].startswith(next(r for r in SCENE_WORDS if r.split()[0].lower()[:5] in q.grade.truth)), q.qid
            if template == "rooms":
                room = next(r for r in SCENE_WORDS if r.lower().replace(" ", "_") == q.qid.split("-", 1)[1])
                assert q.grade.truth == str(sum(r["scene"].startswith(room) for _, r in named.values())), q.qid
            if template == "count" and q.kind != "absent":
                key = q.qid.split("-", 1)[1]
                assert q.grade.truth == str(sum(P.TOPICS[key].labels[0] in labelled(r) for _, r in named.values())), q.qid
            if template == "every" and q.kind != "absent":
                key = q.qid.split("-", 1)[1]
                assert set(q.grade.items) == {n for n, (_, r) in named.items() if P.TOPICS[key].labels[0] in labelled(r)}, q.qid


@needs_labels
def test_the_shelf_and_the_filing_are_the_taxonomys_and_cover_every_template_asked(planned):
    taxonomy.check(V10.shelf, V10.filing)
    assert V10.shelf.media == "video" and V10.shelf.pile
    asked = {q.tags["template"] for plan in planned.values() for q in plan[1]}
    assert asked <= set(V10.filing), asked - set(V10.filing)
    assert {V10.filing[t][0] for t in asked} >= {"find_all", "how_many", "did"}
    assert V10.filing["room"][0] == "who" and "after" not in V10.filing and V10.filing["before"][0] == "order"


@needs_labels
def test_the_second_action_of_a_before_question_is_one_labelled_span_a_clear_gap_from_the_topics():
    pile = P.choose(S.read_clips(str(S.SOURCE)), "standard")
    found = 0
    for key, clip in pile.one.items():
        got = P.order_of(clip, P.TOPICS[key].labels[0])
        if got is None:
            continue
        found += 1
        cls, side, two = got
        spans = labelled(raw()[clip.id])
        a, b = spans[P.TOPICS[key].labels[0]][0]
        assert cls in P.NAMED and len(spans[cls]) == 1 and (spans[cls][0][0], min(spans[cls][0][1], float(raw()[clip.id]["length"]))) == two
        assert (two[1] <= a - P.GAP_S) if side == "before" else (two[0] >= b + P.GAP_S)
    assert found >= 2


@needs_labels
def test_a_topic_that_is_never_in_the_pile_is_asked_of_once_and_a_dev_answer_gives_away_no_test_question(planned):
    for seed in (1, 2, 3):
        questions = planned["standard", seed][1]
        absent = [q.qid.split("-", 2)[2] for q in questions if q.kind == "absent"]
        assert len(absent) == len(set(absent)) == 5, seed
        side = {}
        for q in questions:
            key = q.qid.removeprefix("absent-").split("-", 1)[1]  # the topic, or the room
            side.setdefault(key, set()).add(q.split)
        assert all(len(v) == 1 for v in side.values()), {k: v for k, v in side.items() if len(v) > 1}
        assert {"dev", "test"} == {q.split for q in questions}


@needs_labels
def test_the_names_and_times_are_a_phones_and_say_nothing_of_the_clip(planned):
    files, _ = planned["standard", 1]
    assert all(re.fullmatch(r"DCIM/100MEDIA/VID_\d{8}_\d{6}\.mp4", f.path) for f in files)
    assert [f.path for f in files] == sorted(f.path for f in files) and all(b.t - a.t >= a.length + 59 for a, b in zip(files, files[1:], strict=False))
    assert datetime.fromtimestamp(files[0].t).year == 2025 and datetime.fromtimestamp(files[-1].t) < datetime(2026, 11, 1)
    # nothing in the list gives a clip away: the order of the files is not the order of the ids, or of the scenes
    ids = [f.spec["member"] for f in files]
    assert ids != sorted(ids) and len({datetime.fromtimestamp(f.t).hour for f in files}) >= 15
    scenes = [raw()[m.split("/")[1][:-4]]["scene"][:6] for m in ids]
    runs = sum(a == b for a, b in zip(scenes, scenes[1:], strict=False))
    assert runs < len(scenes) * 0.2, "neighbouring clips are not of one scene"


@needs_labels
def test_the_questions_make_a_file_that_loads(planned, tmp_path):
    questions = planned["standard", 1][1]
    write_questions(tmp_path / "q.jsonl", questions)
    assert load_questions(tmp_path / "q.jsonl") == questions
    assert {q.qid for q in planned["standard", 1][1]} != {q.qid for q in planned["standard", 2][1]}


@needs_labels
def test_a_pile_with_a_video_missing_is_refused(source, tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / S.LABELS).write_bytes((S.SOURCE / S.LABELS).read_bytes())
    have = {f.spec["member"] for f in V10.plan_of(V10.recipe(scale="tiny", seed=1, source=source))[0]}
    with zipfile.ZipFile(folder / S.VIDEOS, "w") as z:
        for vid in raw():
            if S.MEMBER.format(vid) not in list(have)[:1]:
                z.writestr(S.MEMBER.format(vid), b"")
    with pytest.raises(FileNotFoundError, match="are not in"):
        V10.plan_of(V10.recipe(scale="tiny", seed=1, source=str(folder)))


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = V10.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        V10.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        V10.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        V10.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = V10.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:v10_home_clips" in deny and "in:charades" in deny and "in:from mmsb" in deny


def test_a_before_or_after_answer_is_told_apart_and_so_is_a_room():
    """What a right answer to `before` and `room` says is taken, and a wrong one is not, with the explanation that a person gives."""
    first = realdump.grade("text", truth="before|earlier", wrong=("after", "later"))
    assert grade.grade(first, "Before.") and grade.grade(first, "I turned the light on before fixing it.") and not grade.grade(first, "After fixing it.")
    assert grade.grade(first, "Before, not after.") and not grade.grade(first, "After, not before.") and not grade.grade(first, "Before or after, I cannot tell.")
    second = realdump.grade("text", truth="after|later", wrong=("before", "earlier"))
    assert grade.grade(second, "It was after.") and not grade.grade(second, "Before.")
    rooms = realdump.grade("text", truth=P.ROOMS["Stairs"][1])
    assert grade.grade(rooms, "On the staircase") and not grade.grade(rooms, "In the kitchen")


def test_a_clip_with_an_action_that_ends_before_it_starts_is_flawed_and_the_rest_are_read(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    header = ",".join(S.HEADER)
    classes = "\n".join(f"c{n:03d} Action {n}" for n in range(157))
    rows = ['A1,S,Kitchen,5,6,Yes,"a person opens a door.",door,"opens a door",c008 1.00 4.00;c001 5.00 2.00,10.0',
            'A2,S,Garage (a room),5,6,Yes,"a person",door,"x",c008 9.00 12.00,10.0',
            'A3,S,Garage (a room),5,6,Yes,"a person",door,"x",c008 11.00 12.00,10.0']
    with zipfile.ZipFile(folder / S.LABELS, "w") as z:
        z.writestr(S.TABLES[0], header + "\n" + "\n".join(rows) + "\n")
        z.writestr(S.TABLES[1], header + "\n")
        z.writestr(S.CLASSES, classes)
    S.read_clips.cache_clear()
    S.class_names.cache_clear()
    a1, a2, a3 = S.read_clips(str(folder))
    assert a1.flawed and a1.spans == (("c008", 1.0, 4.0),), "an action that ends before it starts is dropped and the clip is flawed"
    assert not a2.flawed and a2.spans == (("c008", 9.0, 10.0),) and a2.scene == "Garage", "an action that ends after the video is cut to it"
    assert a3.flawed and a3.spans == (), "an action that starts after the video ends is dropped and the clip is flawed"
    S.read_clips.cache_clear()
    S.class_names.cache_clear()


# -- the real videos -----------------------------------------------------------------------------------------------------------------

def videos_complete() -> bool:
    try:
        with zipfile.ZipFile(S.SOURCE / S.VIDEOS) as z:
            return len(z.namelist()) >= 9848
    except (FileNotFoundError, zipfile.BadZipFile):
        return False


@pytest.mark.skipif(not HAVE or not videos_complete(), reason="the Charades videos zip is not complete on disk, so the tiny pile is not built")
def test_the_tiny_pile_is_copied_out_whole_with_its_times_and_passes_the_probe(tmp_path):
    recipe = V10.recipe(scale="tiny", seed=1)
    dump, _ = V10.build(recipe, tmp_path / "dump", 3)
    assert len(dump.entries) == 72
    for e in dump.entries[:12]:
        path = tmp_path / "dump" / e.path
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == datetime.fromtimestamp(e.t)
        with av.open(str(path)) as c:
            assert abs(float(c.duration) / 1e6 - e.length) < 0.5, "the CSV's length is the container's to a third of a second"
    with zipfile.ZipFile(S.SOURCE / S.VIDEOS) as z:
        files, _ = V10.plan_of(recipe)
        assert all(e.size == z.getinfo(f.spec["member"]).file_size for e, f in zip(dump.entries, files, strict=True))
    assert V10.probe(recipe, dump, tmp_path, None, 1).status == "passed"
