"""V2, the dashcam card: the questions of a made-up card (every answer derived again here by a plain reader of the CSV rows, never the scenario's),
the grader taking each truth and refusing planted wrong answers; and, from the real source on disk, both scales planned, the tiny card built
(remuxed clips without metadata, their times), its leak probe, and that the times of the clips say nothing of the class or of day and night.
The real-source tests skip, and say so, without the source."""

from __future__ import annotations

import csv
import functools
import itertools
import os
import random
import re
from datetime import datetime

import av
import pytest

from mmsb.core import grade, taxonomy
from mmsb.core.question import load_questions, write_questions
from mmsb.scenarios import realdump
from mmsb.scenarios import v2_dashcam as V2
from mmsb.scenarios.v2_dashcam import questions as Q
from mmsb.scenarios.v2_dashcam.source import SOURCE, TABLE, Clip, mp4_header

pytestmark = pytest.mark.xdist_group("v2")
HAVE = (SOURCE / TABLE).is_file()
needs_source = pytest.mark.skipif(not HAVE, reason=f"the dashcam source is not at {SOURCE}, so the real V2 is not built")
KEYS = {"crash": "Crash", "near": "Near Crash", "normal": "Normal"}
PLACES = {"highway": ("location", "Highway"), "roundabout": ("location", "Roundabouts"), "carpark": ("location", "Parking Lot"),
          "tunnel": ("location", "Tunnel"), "rain": ("weather", "Rain"), "night": ("time", "Night")}  # written here, not read from the scenario


def made_up(n: int = 160, seed: int = 4) -> list[Clip]:
    """A made-up CSV's clips: every class, place, weather and time in some combination, some of the crash-and-near-miss combinations that
    the absent questions ask about among them (which the card must hold out)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = rng.choice(["Crash", "Near Crash", "Near Crash", "Normal", "Outliers"])
        out.append(Clip(f"20260106_x_{i:03d}.mp4", f"{kind}/20260106_x_{i:03d}.mp4", kind, rng.choice(["Ego-Centric", "Non-Ego-Centric"]),
                        rng.choice(["Day", "Day", "Night"]), rng.choice(["Clear", "Clear", "Cloudy", "Rain"]),
                        rng.choice(["Highway", "Roundabouts", "Residential", "Rural", "Parking Lot", "Tunnel", "Intersections"]),
                        rng.choice(["Vehicles", "None", "Pedestrian"]), float(rng.randint(3, 40))))
    return out


def plain(clip: Clip, atoms) -> bool:
    return all(getattr(clip, PLACES[a][0]) == PLACES[a][1] for a in atoms)


def paths_of(card: list[Clip]) -> list[str]:
    return [f"DCIM/Movie/2025_1201_08{n // 60:02d}{n % 60:02d}_{n:03d}.MP4" for n in range(len(card))]


def check_against_rows(questions, card: list[Clip], paths: list[str]) -> None:
    """Each question's truth as a plain reader of the rows finds it, by the question's id (`template-class-atoms`)."""
    name = {c.name: p.rsplit("/", 1)[-1] for c, p in zip(card, paths, strict=True)}
    for q in questions:
        absent = q.qid.startswith("absent-")
        template, key, atoms = (q.qid.removeprefix("absent-").split("-") + [""])[:3]
        atoms = [] if atoms == "any" else atoms.split("_")
        kind = KEYS.get(key)
        found = [c for c in card if (kind is None or c.kind == kind) and plain(c, atoms)]
        assert kind != "Outliers" and all(c.kind != "Outliers" for c in card)
        if absent:
            assert not found and q.grade.rule == "absent" and not q.truth.evidence and (kind is None or kind in ("Crash", "Near Crash")), q.qid
            assert not any(c.kind in ("Crash", "Near Crash") and plain(c, atoms) for c in card), f"{q.qid}: a crash or near miss is there"
            continue
        assert found, q.qid
        if template == "count":
            assert q.grade.truth == str(len(found)), q.qid
        elif template == "what":
            assert len(found) == 1 and q.truth.value == found[0].time, q.qid
            assert q.grade.rule == "text"
        else:
            assert q.grade.items == tuple(name[c.name] for c in found), q.qid
            assert (template != "which") or len(found) == 1, q.qid
        assert [s.file for s in q.truth.evidence] == [dict(zip((c.name for c in card), paths, strict=True))[c.name] for c in found], q.qid


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "absent": lambda: "unknown",
            "text": lambda: g.truth.split("|")[0]}[g.rule]()


def wrong(q) -> list[str]:
    g = q.grade
    if g.rule == "set":
        extra = "2030_0101_000000_999.MP4"
        return [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), "unknown"] + ([", ".join(g.items[1:])] if len(g.items) > 1 else [])
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "text":
        return [g.wrong[0], "unknown"]
    return ["2025-12-03 10:15", "2025-12-03"]  # absent: a time for it


def test_a_made_up_card_is_asked_only_what_its_rows_say_and_holds_out_what_an_absent_question_asks_about():
    clips = made_up()
    card = V2.card_of(clips, "standard")
    assert all(c.kind != "Outliers" for c in card) and len(card) < len(clips)
    for atoms in Q.ABSENT:
        assert not any(c.kind in Q.EVENTS and Q.matches(c, atoms) for c in card), atoms
    paths = paths_of(card)
    for seed in range(1, 6):
        questions = Q.questions_of(card, paths, "standard", seed, 1.8e9, 1.7e9, 1.7e9 + 1e5)
        check_against_rows(questions, card, paths)
        assert len({q.qid for q in questions}) == len(questions)
        assert {q.split for q in questions} == {"dev", "test"}


def test_the_hold_out_takes_a_near_miss_in_a_car_park_at_night_out_of_a_card_that_is_asked_for_a_crash_there():
    near = Clip("a.mp4", "Near Crash/a.mp4", "Near Crash", "Ego-Centric", "Night", "Clear", "Parking Lot", "Vehicles", 9.0)
    other = Clip("b.mp4", "Near Crash/b.mp4", "Near Crash", "Ego-Centric", "Day", "Clear", "Highway", "Vehicles", 9.0)
    crash = Clip("c.mp4", "Crash/c.mp4", "Crash", "Ego-Centric", "Night", "Clear", "Highway", "Vehicles", 9.0)
    day = Clip("d.mp4", "Normal/d.mp4", "Normal", "Ego-Centric", "Night", "Clear", "Parking Lot", "None", 9.0)  # an ordinary drive is no event
    assert Q.held_out(near) and not Q.held_out(other) and not Q.held_out(crash) and not Q.held_out(day), "a crash on a highway at night is not in a car park"
    assert V2.card_of([near, other, crash, day], "standard") == [other, crash, day]
    pools = Q.library(V2.card_of([near, other, crash, day], "standard"))
    assert {(i.kind, i.atoms) for i in pools["absent_did"]} >= {("Crash", ("carpark", "night")), ("Near Crash", ("carpark", "night"))}


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    card = V2.card_of(made_up(), "standard")
    for seed in (1, 2):
        for q in Q.questions_of(card, paths_of(card), "standard", seed, 1.8e9, 1.7e9, 1.7e9 + 1e5):
            assert grade.grade(q.grade, right(q)), q.qid
            for answer in wrong(q):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_the_time_of_day_is_asked_in_words_that_a_wrong_answer_does_not_hold_and_a_right_one_may_use_in_passing():
    cases = {"Night": [("It was night", True), ("at night, after dark", True), ("it was daytime", False), ("night and day", False),
                       ("Night, in bright headlights", True), ("Night, with the sunny afternoon far off", False)],
             "Day": [("It was daytime", True), ("Daytime, inside a dark tunnel", True), ("daylight, sunny", True), ("night", False),
                     ("It was the evening", False), ("day or night, I cannot tell", False)]}
    for time_of_day, answers in cases.items():
        accepted, other = Q.SAYS[time_of_day]
        q = realdump.grade("text", truth=accepted, wrong=other.split("|"))
        for answer, ok in answers:
            assert grade.grade(q, answer) is ok, (time_of_day, answer)
    assert not {"dark", "bright", "sunny"} & set(Q.SAYS["Day"][1].split("|")) and not {"dark", "bright", "sunny"} & set(Q.SAYS["Night"][1].split("|"))


def test_the_pile_says_that_the_card_is_staged():
    assert "staged" in V2.shelf.pile and "YouTube" in V2.shelf.pile and "invented" in V2.shelf.pile
    assert "STAGED" in V2.__doc__ and "YouTube" in V2.__doc__ and "invented" in V2.__doc__


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = V2.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        V2.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        V2.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        V2.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = V2.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:v2_dashcam" in deny and "in:from mmsb" in deny and "in:qutegocentric" in deny


def test_times_are_a_spread_with_a_gap_and_no_link_to_the_order_of_the_names():
    times = V2.spread([5.0] * 400, realdump.span_days(datetime(2025, 12, 1).date(), datetime(2025, 12, 14).date()), random.Random(1))
    assert times == sorted(times) and all(b - a >= 95 for a, b in itertools.pairwise(times))
    assert {datetime.fromtimestamp(t).hour for t in times} >= set(range(7, 21))


# -- the real source ---------------------------------------------------------------------------------------------------------------

@functools.cache
def rows() -> list[dict]:
    with open(SOURCE / TABLE, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@needs_source
@pytest.mark.parametrize("scale, n, absent", [("tiny", 30, 2), ("standard", 107, 5)])
def test_the_plan_at_each_scale(scale, n, absent):
    files, questions = V2.plan_of(V2.recipe(scale=scale, seed=1))
    assert len(files) == n and sum(q.kind == "absent" for q in questions) == absent
    assert 2 <= len(questions) <= (8 if scale == "tiny" else 24)
    assert len({f.t for f in files}) == n and all(f.path.startswith("DCIM/Movie/") and f.path.endswith(".MP4") for f in files)
    by_src = {f.spec["src"]: f for f in files}
    table = {f"{r['classification']}/{r['video_name']}": r for r in rows()}
    assert not any(table[s]["classification"] == "Outliers" for s in by_src), "the outliers are held out"
    card = [Clip(r["video_name"], s, r["classification"], r["ego"], r["time"], r["weather"], r["location"], r["hazard"], by_src[s].length)
            for s, r in ((s, table[s]) for s in by_src)]
    ordered = sorted(card, key=lambda c: by_src[c.rel].t)
    check_against_rows(questions, ordered, [by_src[c.rel].path for c in ordered])


@needs_source
def test_the_standard_card_holds_the_rows_but_the_outliers_and_the_crashes_and_near_misses_that_an_absent_question_asks_about():
    files, questions = V2.plan_of(V2.recipe(scale="standard", seed=1))
    kept = {f.spec["src"] for f in files}
    left_out = [r for r in rows() if f"{r['classification']}/{r['video_name']}" not in kept]
    assert len(rows()) == 128 and len(left_out) == 21
    assert sum(r["classification"] == "Outliers" for r in left_out) == 20
    assert {(r["classification"], r["location"], r["weather"], r["time"]) for r in left_out if r["classification"] != "Outliers"} == {
        ("Near Crash", "Intersections", "Rain", "Night")}
    assert [q.qid for q in questions] == [q.qid for q in V2.plan_of(V2.recipe(scale="standard", seed=1))[1]]


@needs_source
def test_the_shelf_and_the_filing_are_the_taxonomys_and_cover_every_template_asked():
    taxonomy.check(V2.shelf, V2.filing)
    assert V2.shelf.media == "video" and V2.shelf.pile
    asked = {q.tags["template"] for scale in ("tiny", "standard") for seed in (1, 2, 3) for q in V2.plan_of(V2.recipe(scale=scale, seed=seed))[1]}
    assert asked <= set(V2.filing), asked - set(V2.filing)
    assert {V2.filing[t][0] for t in asked} >= {"find_all", "how_many", "did"}


@needs_source
def test_another_seed_asks_other_questions_of_the_same_card():
    files, one = V2.plan_of(V2.recipe(scale="standard", seed=1))
    for seed in (2, 3):
        other_files, other = V2.plan_of(V2.recipe(scale="standard", seed=seed))
        assert files == other_files and {q.qid for q in one} != {q.qid for q in other}


@needs_source
def test_the_questions_make_a_file_that_loads(tmp_path):
    _, questions = V2.plan_of(V2.recipe(scale="standard", seed=1))
    write_questions(tmp_path / "q.jsonl", questions)
    assert load_questions(tmp_path / "q.jsonl") == questions


@needs_source
def test_a_clip_s_time_says_nothing_of_its_class_or_of_day_and_night():
    """The list of the files alone must not answer: no class and no day or night is the same hours."""
    files, _ = V2.plan_of(V2.recipe(scale="standard", seed=1))
    table = {f"{r['classification']}/{r['video_name']}": r for r in rows()}
    night = [datetime.fromtimestamp(f.t).hour for f in files if table[f.spec["src"]]["time"] == "Night"]
    day = [datetime.fromtimestamp(f.t).hour for f in files if table[f.spec["src"]]["time"] == "Day"]
    assert any(6 <= h < 18 for h in night) and any(h >= 20 for h in day), "a night clip at noon and a day clip at nine in the evening: the clocks are not the sun's"
    order = [table[f.spec["src"]]["classification"] for f in files]
    assert order != sorted(order) and order != sorted(order, reverse=True)


@needs_source
def test_the_tiny_card_is_remuxed_without_metadata_and_with_its_times(tmp_path):
    recipe = V2.recipe(scale="tiny", seed=1)
    dump, _ = V2.build(recipe, tmp_path / "dump", 3)
    assert len(dump.entries) == 30
    for e in dump.entries[:6]:
        path = tmp_path / "dump" / e.path
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == datetime.fromtimestamp(e.t)
        with av.open(str(path)) as c:
            assert abs(float(c.duration) / 1e6 - e.length) < 0.05 and not c.streams.audio
        text = " ".join(m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{4,}", mp4_header(path)))
        assert "Google" not in text and "produced by" not in text and "VideoHandler" in text
    assert V2.probe(recipe, dump, tmp_path, None, 1).status == "passed"
