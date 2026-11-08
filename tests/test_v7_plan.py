"""Ported from Multimem tests/test_haystack.py at fe0ece5 (24 of its 31 tests; the other 7 are tests/mm's or the select tests).

The 7 left out start a daemon, use `load`, `windows_in` or the real reader, or ask which model reads what. Its graders are the core's now.

The V7 plan (plan.py, week.py): it is pure, the same for a seed, and checks itself; the levels of its questions are balanced and each
question's scope holds what its level says (one needle, the last needle, none), with distractors of the needle's colour or pitch in
every scope; the week is written in a few seconds, and what is written is what the plan says; and the graders (`mmsb/core/grade.py`)
grade as their table says. The files are made once for the module on each worker that runs it, and only read.

Two of the tests changed more than their names: `test_the_levels_are_balanced_as_the_plan_says` takes the 24 questions of a model that
does not hear as the ones whose type is not audio (it asked `questions_for`, which `Description.reads` and core/select.py replaced),
and `test_tiny_is_written_in_30_seconds_and_loaded_in_8` is the first half (the load is tests/mm's)."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from collections import Counter
from datetime import date, datetime

import av
import numpy as np
import pytest
from PIL import ExifTags, Image, ImageChops

from mmsb.core import grade
from mmsb.core.question import RULES, Grade
from mmsb.scenarios.v7_screen import week as v7_week
from mmsb.scenarios.v7_screen.constants import RATE
from mmsb.scenarios.v7_screen.media_common import DISTRACTOR_RGB, distance
from mmsb.scenarios.v7_screen.media_porch import BOX_X0, BOX_Y1
from mmsb.scenarios.v7_screen.media_screen import (
    APPS, CURSOR_HOME, DIALOG_AT, FLOOR_S, SCREEN, screen_frame, screen_frames, screen_layers, typing_cap,
)
from mmsb.scenarios.v7_screen.media_sound import BEEPER_HZ, mic_samples, sound
from mmsb.scenarios.v7_screen.pictures import photo_image, photo_needle
from mmsb.scenarios.v7_screen.plan import (
    ABSENT, ACT_KINDS, ACTIVE_SHARE, ACTIVITY, LEVELS, MOVE_PX, PARTS, SCALES, SERVICES, SOURCES, TEMPLATES, TYPES, VERSION, WEEKDAYS,
    File, Plan, arrange, at, clock, date_text, days_before, last_days, needle_moments, part_of, segments, share_out, verify,
)
from mmsb.scenarios.v7_screen.questions import grade_for
from mmsb.scenarios.v7_screen.week import _Week
from mmsb.scenarios.v7_screen.write import check_shades, check_sounds, write

DAYS = [date(2026, 10, 30), date(2026, 10, 31)]
KEYS = {"qid", "type", "template", "absent", "level", "last", "source", "since", "until", "hint", "question", "truth", "grade",
        "oracle", "fake", "probe"}
SOURCE_OF = {"note-code": "notes", "config-asof": "config", "log-count": "logs", "log-last": "logs", "photo-sign": "phone",
             "photo-count": "phone", "screen-error": "screen", "porch-parcel": "porch", "mic-doorbell": "mic", "mic-alarm": "mic",
             "absent-note": "notes", "absent-log": "logs", "absent-sign": "phone", "absent-error": "screen", "absent-parcel": "porch",
             "absent-doorbell": "mic"}

# the questions of each scale: (template, level), in the plan's order (absent templates last in their type)
TINY = [("note-code", "part"), ("config-asof", "time"), ("absent-note", "part"), ("log-count", "part"), ("log-last", "part"),
        ("absent-log", "day"), ("photo-sign", "part"), ("photo-sign", "week"), ("photo-count", "day"), ("absent-sign", "part"),
        ("screen-error", "part"), ("screen-error", "week"), ("porch-parcel", "part"), ("porch-parcel", "week"), ("absent-error", "part"),
        ("mic-doorbell", "part"), ("mic-alarm", "part"), ("mic-alarm", "week"), ("absent-doorbell", "part")]
FULL = [("note-code", "part"), ("note-code", "day"), ("config-asof", "time"), ("absent-note", "part"), ("log-count", "part"),
        ("log-count", "day"), ("log-last", "part"), ("absent-log", "day"), ("photo-sign", "part"), ("photo-sign", "day"),
        ("photo-sign", "week"), ("photo-count", "day"), ("photo-count", "week"), ("absent-sign", "part"), ("absent-sign", "day"),
        ("screen-error", "part"), ("screen-error", "day"), ("screen-error", "week"), ("porch-parcel", "part"), ("porch-parcel", "day"),
        ("porch-parcel", "week"), ("absent-error", "part"), ("absent-error", "day"), ("absent-parcel", "day"), ("mic-doorbell", "part"),
        ("mic-doorbell", "day"), ("mic-alarm", "part"), ("mic-alarm", "day"), ("mic-alarm", "week"), ("absent-doorbell", "part"),
        ("absent-doorbell", "day")]
TIMES: dict[str, float] = {}  # seconds: how long the module's tiny week took to write, and its last load
WRITE_S = 30.0  # a tiny week is written in this many seconds: its screen is about 3,700 frames (2 a second while the owner types and

# a single process (`-n 0`) is held to these; under xdist (`-n 4`, with another suite or a GPU run possibly sharing the machine) the bound is
# SLACK times as loose, so that the test fails for a writer or a daemon that got several times slower and never for a busy machine
SLACK = 4.0 if os.environ.get("PYTEST_XDIST_WORKER") else 1.0
pytestmark = pytest.mark.xdist_group("haystack")  # (this xdist does not keep a group on one worker: each worker that runs a test builds its own week)


@pytest.fixture(scope="module")
def tiny():
    return v7_week.plan(7, "tiny", DAYS)


@pytest.fixture(scope="module")
def folder(tiny, tmp_path_factory):
    where = tmp_path_factory.mktemp("haystack")
    began = time.perf_counter()
    write(tiny, where)
    TIMES["write"] = time.perf_counter() - began
    return where


@pytest.fixture(scope="module")
def weeks():
    """The pilot and the standard week's plans (pure: nothing is written), for a few seeds."""
    return {scale: [v7_week.plan(seed, scale, days_before(date(2026, 11, 2), SCALES[scale].days)) for seed in (1, 2, 3)]
            for scale in ("pilot", "standard")}


def levels_of(plan) -> list[tuple[str, str]]:
    return [(q["template"], q["level"]) for q in plan.questions]


# -- the plan ---------------------------------------------------------------------------------------------------------------

def test_the_plan_is_the_same_for_a_seed_and_differs_for_another(tiny):
    assert v7_week.plan(7, "tiny", DAYS) == tiny
    other = v7_week.plan(8, "tiny", DAYS)
    assert other != tiny and [q["truth"]["value"] for q in other.questions] != [q["truth"]["value"] for q in tiny.questions]
    assert days_before(date(2026, 11, 1), 2) == DAYS, "the days ending yesterday"
    assert VERSION == 6
    with pytest.raises(ValueError, match="7 days"):
        v7_week.plan(1, "standard", DAYS)
    with pytest.raises(ValueError, match="oldest first"):
        v7_week.plan(1, "tiny", DAYS[::-1])
    with pytest.raises(ValueError, match="a scale is one of"):
        v7_week.plan(1, "huge", DAYS)


def test_the_plan_is_pure(tiny, monkeypatch):
    """No file is written, no clock read, no global random stream used: the same inputs in another process state give the same plan."""
    import random

    def no(*args, **kwargs):
        raise AssertionError("the plan read the clock or the global random stream")

    monkeypatch.setattr(time, "time", no)
    monkeypatch.setattr(random, "random", no)
    monkeypatch.setattr(random, "randint", no)
    monkeypatch.setattr(random, "choice", no)
    assert v7_week.plan(7, "tiny", DAYS) == tiny
    assert json.loads(json.dumps(tiny.questions[0])) == tiny.questions[0]


def test_the_plan_does_not_depend_on_the_hash_seed():
    """Python orders a set of strings by a hash that differs in every process: no plan may follow it."""
    import os
    import subprocess
    import sys

    code = ("import hashlib, json; from datetime import date; from mmsb.scenarios.v7_screen.plan import SCALES, days_before; "
            "from mmsb.scenarios.v7_screen.week import plan; "
            "out = [hashlib.sha256(json.dumps([list(p.questions), [(f.path, f.t, f.spec) for f in p.files], p.logs], sort_keys=True, default=str).encode()).hexdigest() "
            "for p in (plan(5, s, days_before(date(2026, 11, 2), SCALES[s].days)) for s in ('tiny', 'pilot'))]; print(out)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {seed: subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, check=True,
                                 env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2")}
    assert outs["1"] == outs["2"] and outs["1"].startswith("['"), outs


def check_question(q: dict, plan) -> None:
    who = q["qid"]
    needle = q["template"] in sum(TEMPLATES.values(), ())
    assert set(q) == KEYS and json.loads(json.dumps(q)) == q, who
    assert needle or q["template"] in sum(ABSENT.values(), ()), who
    assert q["absent"] is (not needle) and q["source"] == SOURCE_OF[q["template"]] and q["source"] in SOURCES, who
    assert re.fullmatch(rf"{q['type']}-{q['template']}-\d+", q["qid"]) and q["level"] in LEVELS, who
    assert q["since"] < q["until"] and q["hint"] in q["question"], who
    assert q["grade"]["kind"] in RULES and set(q["grade"]) == {"kind", "truth", "time", "tol_min", "wrong", "ignore", "date"}, who
    assert q["fake"]["check"] in ("overlap", "none", "count", "look", "last") and set(q["fake"]) == {"question", "oldest_first", "limit", "check"}, who
    assert set(grade.asked(q["question"])) <= set(q["grade"]["ignore"]), f"{who} ignores the numbers and times it asks"
    assert set(re.findall(r"\d\d:00", q["hint"])) <= set(q["grade"]["ignore"]), f"{who} ignores its hint's hours"
    assert set(q["truth"]) == {"value", "moments", "others"}
    day = q["until"] - q["since"]
    if q["level"] == "part":
        assert day <= 12 * 3600 + 1 and re.search(r"in the (morning|afternoon|evening) \(\d\d:00 to \d\d:00\)$", q["hint"]), who
    elif q["level"] == "day":
        assert 23 * 3600 <= day <= 25 * 3600 and re.search(r"^on \w+ \d+ \w+$", q["hint"]), who
    elif q["level"] == "week":
        assert q["hint"] == "" and day >= 23 * 3600 * (len(plan.days)) and q["since"] == at(plan.days[0], 0), who
        assert q["until"] == at(plan.days[-1], 86400), who
    if q["absent"]:
        assert q["truth"] == {"value": None, "moments": [], "others": []} and q["grade"]["kind"] == "absent" and q["fake"]["check"] == "none", who
        assert not q["last"] and q["level"] in ("part", "day"), who
    else:
        assert q["truth"]["value"] and q["truth"]["moments"], who
        assert all(q["since"] <= a <= b <= q["until"] for a, b in q["truth"]["moments"]), f"a moment outside the range of {who}"
        assert all(q["since"] <= a <= b <= q["until"] for a, b in q["truth"]["others"]), who
        assert (q["level"] == "week" and q["template"] != "photo-count") is q["last"], who
        assert bool(q["truth"]["others"]) <= (q["level"] == "week"), f"{who}: only a week question has other needles in range"
    media = q["type"] in ("image", "video", "audio")
    assert (q["probe"] is not None) is media, who
    if media:
        kind = "pitch" if q["type"] == "audio" else "shade"
        assert q["probe"]["kind"] == kind and (q["probe"]["hue"] in ("red", "green", "brown") if kind == "shade" else q["probe"]["band"][0] < q["probe"]["hz"] < q["probe"]["band"][1]), who
        assert q["fake"]["question"].startswith(kind + ":"), who
    needs_date = q["level"] == "week" and q["grade"]["kind"] in ("time", "code_time")
    assert (q["grade"]["date"] is not None) is needs_date, who
    if needs_date:
        assert grade.on_date(f"{WEEKDAYS[date.fromisoformat(q['grade']['date']).weekday()]}", q["grade"]["date"]), who
        assert "on what day" in q["question"].lower(), f"{who} asks for the day its answer must name"


def test_each_question_is_well_formed_at_every_scale(tiny, weeks):
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        assert len({q["qid"] for q in plan.questions}) == len(plan.questions)
        for q in plan.questions:
            check_question(q, plan)
        by = {q["template"]: q for q in plan.questions if q["level"] != "week"}
        words = lambda q: q["question"].replace(q["hint"], "")  # noqa: E731
        for needle, absent in (("photo-sign", "absent-sign"), ("screen-error", "absent-error"), ("mic-doorbell", "absent-doorbell"),
                               ("porch-parcel", "absent-parcel")):
            if absent in by:
                assert words(by[needle]) == words(by[absent]), f"an absent question has its template's words ({absent})"
        shape = lambda q: re.sub(r"ERR-\d+", "ERR-", q["question"]).split(" on ")[0].split(" in the ")[0]  # noqa: E731
        assert shape(by["log-count"]) == shape(by["absent-log"]), "an absent question has its template's words (absent-log)"
        assert "Wi-Fi password" in by["absent-note"]["question"] and "Wi-Fi" not in by["note-code"]["question"]
        assert [q["type"] for q in plan.questions] == sorted((q["type"] for q in plan.questions), key=TYPES.index), "in TYPES order"


def test_the_levels_are_balanced_as_the_plan_says(tiny, weeks):
    assert levels_of(tiny) == TINY and len(tiny.questions) == 19
    for plan in (*weeks["pilot"], *weeks["standard"]):
        assert sorted(levels_of(plan)) == sorted(FULL) and len(plan.questions) == 31
        assert sum(q["absent"] for q in plan.questions) == 9 and sum(q["last"] for q in plan.questions) == 4
        asked = [q for q in plan.questions if q["type"] != "audio"]  # a model that does not hear (core/select.py filters by Description.reads)
        assert len(asked) == 24
        media = [q for q in asked if q["type"] in ("image", "video")]
        assert Counter(q["type"] for q in media) == {"image": 7, "video": 9}
        assert Counter(q["level"] for q in media) == {"part": 5, "day": 7, "week": 4}, "media questions per deaf model: part 5, day 7, week 4"
        assert Counter(q["level"] for q in plan.questions) == {"part": 11, "day": 12, "week": 6, "time": 1} or \
            Counter(q["level"] for q in plan.questions) == Counter(level for _, level in FULL)


# -- what each scope holds ---------------------------------------------------------------------------------------------------

def test_every_scope_holds_what_its_level_says_at_every_scale(tiny, weeks):
    """`plan` runs `verify` itself; run it again on every plan, and count by hand what it counts for the pilot's and the
    standard's week questions: the last needle is on a day with a full day or more after it, and the earlier ones are `others`."""
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        verify(plan)
        n = len(plan.days)
        for q in plan.questions:
            if q["level"] != "week" or q["template"] == "photo-count":
                continue
            last_end = max(b for _, b in q["truth"]["moments"])
            index = next(i for i, d in enumerate(plan.days) if at(d, 0) <= last_end < at(d, 86400))
            assert index in last_days(n), f"{q['qid']}: the last needle is on day {index + 1} of {n}"
            assert q["until"] - last_end > 86400, f"{q['qid']}: a full day follows the last needle"
            template = q["template"]
            others = {m for m in map(tuple, q["truth"]["others"])}
            if plan.scale == "standard":
                assert len({round(a) for a, _ in others}) >= 2, f"{q['qid']}: the standard week has earlier needles to be wrong with"
            if plan.scale != "tiny":
                assert others, f"{q['qid']}: {template} has an earlier needle"
                assert all(b < min(a for a, _ in q["truth"]["moments"]) for _, b in others), f"{q['qid']}: the others are all earlier"


def tamper(plan, change, qid: str) -> Plan:
    """A copy of `plan` whose files `change(files)` has changed, with the question `qid` only."""
    return dataclasses.replace(plan, files=tuple(change(list(plan.files))), questions=tuple(q for q in plan.questions if q["qid"] == qid))


def test_a_scope_that_holds_the_wrong_needles_is_refused(tiny):
    sign = next(q for q in tiny.questions if q["template"] == "photo-sign" and q["level"] == "part")
    absent = next(q for q in tiny.questions if q["template"] == "absent-sign")
    needle = next(f for f in tiny.files if f.kind == "photo" and photo_needle(f.spec) == "sign")
    plain = lambda lo, hi: next(f for f in tiny.files if f.kind == "photo" and lo <= f.t < hi and f.spec["scene"] != "sign")  # noqa: E731

    def swap(file, spec):
        return lambda files: [dataclasses.replace(f, spec=spec) if f is file else f for f in files]

    verify(tamper(tiny, lambda files: files, "image-photo-sign-1"))
    with pytest.raises(AssertionError, match=r"image-photo-sign-1 \(part\): the scope holds 2 needles"):  # a second needle in the part
        verify(tamper(tiny, swap(plain(sign["since"], sign["until"]), needle.spec), "image-photo-sign-1"))
    with pytest.raises(AssertionError, match=r"image-photo-sign-2 \(week\): the answer is not the template's last needle"):  # one after the last
        verify(tamper(tiny, swap(plain(sign["until"], tiny.questions[0]["until"] + 9e9), needle.spec), "image-photo-sign-2"))
    with pytest.raises(AssertionError, match=r"image-absent-sign-1 \(part\): an absent question's scope holds 1 needles"):  # one where none should be
        verify(tamper(tiny, swap(plain(absent["since"], absent["until"]), needle.spec), "image-absent-sign-1"))
    with pytest.raises(AssertionError, match=r"image-photo-sign-1 \(part\): the scope holds 0 needles"):  # the needle gone
        verify(tamper(tiny, lambda files: [f for f in files if f is not needle], "image-photo-sign-1"))
    count = next(q for q in tiny.questions if q["template"] == "photo-count")
    car = next(f for f in tiny.files if f.kind == "photo" and photo_needle(f.spec) == "car" and count["since"] <= f.t < count["until"])
    with pytest.raises(AssertionError, match=r"image-photo-count-1 \(day\): the scope holds 1 red cars, the answer is 2"):
        verify(tamper(tiny, lambda files: [f for f in files if f is not car], "image-photo-count-1"))


def test_the_solver_places_needles_or_says_what_it_cannot_meet():
    import random

    slots = {(d, p) for d in range(7) for p in PARTS}
    for seed in range(60):
        spot = arrange(random.Random(seed), "t", 7, ("part", "day", "week"), ("part", "day"), slots, slots, extra=1)
        needles = spot.needles
        assert len(needles) == 4 and spot.last == needles[-1] and spot.last[0] in last_days(7)
        assert sum(s[0] == spot.day[0] for s in needles) == 1, "the day question's day holds one needle"
        assert sum(s == spot.part for s in needles) == 1 and spot.part != spot.last and spot.day[0] < spot.last[0]
        assert spot.absent_part not in needles and all(s[0] != spot.absent_day for s in needles)
    with pytest.raises(ValueError, match=r"no way to place needles for \('part', 'day', 'week'\).*in 2 days"):
        arrange(random.Random(1), "the template", 2, ("part", "day", "week"), (), {(0, "morning"), (1, "morning")}, {(0, "morning")})
    with pytest.raises(ValueError, match="the template: no way to place needles"):  # nothing is big enough to name
        arrange(random.Random(1), "the template", 3, ("part",), ("part",), {(0, "morning")}, set())
    assert [list(last_days(n)) for n in (2, 3, 4, 7)] == [[0], [1], [2], [3]]
    assert share_out(30, (5, 6, 5)) == [10, 11, 9] and share_out(12, (5, 6, 5)) == [4, 4, 4]


def test_the_error_dialog_and_the_parcel_move_within_their_areas_and_so_do_their_distractors(tiny, weeks):
    """Dialogs open in different places of the middle of the screen (the error's too), and a parcel is left in different places of the
    step, the same in every clip that shows it: a crop of one place finds neither."""
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        dialogs, boxes = [], {}
        parcels = needle_moments(plan, "porch-parcel")
        for f in plan.files:
            if f.kind == "screen":
                for s in f.spec["states"]:
                    if s["overlay"]:
                        cx, cy = s["overlay"]["at"]
                        assert DIALOG_AT[0][0] <= cx <= DIALOG_AT[0][1] and DIALOG_AT[1][0] <= cy <= DIALOG_AT[1][1], f.path
                        dialogs.append((s["overlay"]["kind"], cx, cy))
            elif f.kind == "porch" and f.spec["parcel"]:
                x0, y0, x1, y1 = f.spec["box"]
                assert BOX_X0[0] <= x0 <= BOX_X0[1] and BOX_Y1[0] <= y1 <= BOX_Y1[1], f.path
                span = (f.t + f.spec["parcel"][0], f.t + f.spec["parcel"][1])
                which = next(i for i, m in enumerate(parcels) if any(abs(span[0] - a) < 1e-6 and abs(span[1] - b) < 1e-6 for a, b in m))
                boxes.setdefault(which, set()).add(tuple(f.spec["box"]))
        errors = [d for d in dialogs if d[0] == "error"]
        assert len(errors) == len(needle_moments(plan, "screen-error"))
        assert all(len(places) == 1 for places in boxes.values()), "a parcel stays where it was left: one place in every clip that shows it"
        if plan.scale == "standard":
            assert len({(round(cx, 2), round(cy, 2)) for _, cx, cy in dialogs}) > 50, "dialogs open in many places"
            assert len({(round(d[1], 2), round(d[2], 2)) for d in errors}) == len(errors) and max(d[1] for d in errors) - min(d[1] for d in errors) > 0.05
            assert len({next(iter(b)) for b in boxes.values()}) == len(boxes) >= 4, "each parcel is left in a place of its own"


def test_only_the_courier_who_leaves_the_parcel_drops_a_box(tiny, weeks):
    """A courier of the plan either drops the box in the second its clip shows the parcel from, or keeps it (`drop_at` None):
    no other clip shows a hand-over, so no absent or `last` question has a competing delivery."""
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        kept = left = 0
        for f in plan.files:
            for e in f.spec.get("events", ()) if f.kind == "porch" else ():
                if e["kind"] != "courier":
                    continue
                if e["drop_at"] is None:
                    kept += 1
                else:
                    left += 1
                    assert f.spec["parcel"] and f.spec["parcel"][0] == e["drop_at"], f"{f.path}: a courier drops a box and no parcel is on the step"
        assert left == len(needle_moments(plan, "porch-parcel")) and (kept >= 1 or plan.scale == "tiny"), plan.scale


def test_segments_are_cut_at_the_first_start_after_each_multiple_of_the_segment_time():
    """The recorder's segment muxer ends the n-th segment on the first key frame at or after n times the segment time from the stream's
    start, and a still screen has one at its next change: `segments` cuts a block's state starts so."""
    assert segments([0, 25, 50, 70, 100, 130], 0, 150, 60) == [(0, 70), (70, 130), (130, 150)]
    assert segments([100, 125, 160, 190], 100, 250, 60) == [(100, 160), (160, 250)], "multiples are counted from the block's start"
    assert segments([0, 130, 140, 200], 0, 260, 60) == [(0, 130), (130, 140), (140, 200), (200, 260)], "a long state passes 2 multiples"
    assert segments([0, 30, 55, 121, 140, 190], 0, 250, 60) == [(0, 121), (121, 140), (140, 190), (190, 250)], "a state over 60 s"
    assert segments([0], 0, 45, 60) == [(0, 45)] and segments([0, 30], 0, 45, 60) == [(0, 45)], "a block shorter than a segment is one"


def test_screen_segments_start_on_a_state_change_and_tile_their_blocks(tiny, weeks):
    """Each screen block is planned whole and cut as the recorder cuts it (60 s for every scale): every segment starts on a state's start
    (its first state is at 0) and lasts until the next segment's start (or the block's end); each segment starts at the first state
    start at or after a multiple of 60 s from the block's start (so a block makes a segment a minute, about 60 s long on average and as
    short as a state when the cut before it came late); the states of a block follow each other without a gap or an overlap, none is in two
    segments; every segment is in one part of the day; the error is one state, wholly inside one segment; and the names are unique and in time order."""
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        sc = SCALES[plan.scale]
        assert sc.screen_s == 60
        screen = [f for f in plan.files if f.kind == "screen"]
        names = [f.path for f in sorted(screen, key=lambda f: f.t)]
        assert len(set(names)) == len(names) and names == sorted(names), "the names are unique and sort by time"
        seen = 0
        for d, day in enumerate(plan.days):
            for lo, hi in sc.screen[d % len(sc.screen)]:
                t0, t1 = at(day, lo), at(day, hi)
                segs = sorted((f for f in screen if t0 <= f.t < t1), key=lambda f: f.t)
                seen += len(segs)
                who = f"{plan.scale} seed {plan.seed} {day} {clock(lo)}"
                assert abs(len(segs) - (hi - lo) / sc.screen_s) <= 1, f"{who}: {len(segs)} segments in {hi - lo} s"
                spans = []  # the block's states, in time
                for k, f in enumerate(segs):
                    states, length = f.spec["states"], f.spec["length_s"]
                    end = segs[k + 1].t if k + 1 < len(segs) else t1
                    assert f.t + length == end, f"{who}: {f.path} does not last until the next segment (or the block's end)"
                    assert states[0]["from"] == 0 and states[-1]["to"] == length, f"{who}: {f.path} does not start and end on a state"
                    assert all(10 <= s["to"] - s["from"] <= 68 for s in states) and length >= 10, f"{who}: {f.path} has a state of another length"
                    spans += [(f.t + s["from"], f.t + s["to"]) for s in states]
                    first = f.t - t0
                    assert first >= sc.screen_s * k, f"{who}: {f.path} starts {first} s into the block, before multiple {k}"
                    # the rule itself: the segment's later states all start before the next multiple (else one of them was the cut)
                    assert all(first + s["from"] < sc.screen_s * (k + 1) for s in states[1:]), f"{who}: {f.path} holds a state that was the cut after multiple {k + 1}"
                    a, b = datetime.fromtimestamp(f.t), datetime.fromtimestamp(f.t + length - 1)
                    assert part_of(a.hour * 60 + a.minute) == part_of(b.hour * 60 + b.minute), f"{who}: {f.path} is in two parts of the day"
                assert spans[0][0] == t0 and spans[-1][1] == t1, f"{who}: the states do not span the block"
                assert all(x[1] == y[0] for x, y in zip(spans, spans[1:])), f"{who}: a gap or an overlap between states"
        assert seen == len(screen), "every segment is in a block"
        errors = needle_moments(plan, "screen-error")
        assert len(errors) == sum(1 for f in screen for s in f.spec["states"] if (s["overlay"] or {}).get("kind") == "error")
        for (span,) in errors:
            holding = [f for f in screen if f.t <= span[0] and span[1] <= f.t + f.spec["length_s"]]
            assert len(holding) == 1, f"{plan.scale} seed {plan.seed}: the error at {span} is wholly inside {len(holding)} segments"
            assert 20 <= span[1] - span[0] <= 40


def test_a_part_question_names_a_part_with_enough_to_look_through(weeks):
    """The floors of the shortcut probe, by the files' specs: a part named in a question holds 250 screen states, 8 porch clips, 15
    photos or 5 sounds (the evening's 20:00 block and the microphone's one hour of the afternoon are too small to name)."""
    floor = {"screen": 250, "porch": 8, "phone": 15, "mic": 5}
    for plan in (*weeks["pilot"], *weeks["standard"]):
        for q in plan.questions:
            if q["level"] != "part" or q["source"] not in floor:
                continue
            have = 0
            for f in plan.files:
                if f.source != q["source"] or not q["since"] <= f.t < q["until"]:
                    continue
                if f.kind == "screen":
                    have += len(f.spec["states"])
                elif f.kind in ("porch", "photo"):
                    have += 1
                elif f.kind == "mic":
                    have += len(f.spec["sounds"])
            assert have >= floor[q["source"]], f"{plan.scale} {q['qid']} names a part that holds {have} {q['source']} candidates, the floor is {floor[q['source']]}"
            part = re.search(r"in the (\w+)", q["hint"]).group(1)
            assert not (q["source"] == "screen" and part == "evening") and not (q["source"] == "mic" and part == "afternoon"), q["qid"]


def classes_in(plan, source: str, since: float, until: float) -> Counter:
    """The distractors of each class the files' specs hold in a range, by what a shortcut for the needle's colour or pitch would also
    pick: red dialogs and accents, a green or red thing that is no sign or car, a sign of another colour, a car of another colour, a brown
    dog or the courier carrying a box, a light change, a beeper, a chime-like ding, ring or bark."""
    out: Counter = Counter()
    green = {tuple(c) for c in DISTRACTOR_RGB["sign"].values()}
    red = {tuple(c) for c in DISTRACTOR_RGB["car"].values()}
    for f in plan.files:
        if f.source != source:
            continue
        if f.kind == "screen":
            for s in f.spec["states"]:
                if since <= f.t + s["from"] < until:
                    if s["overlay"] and s["overlay"]["kind"] != "error":  # an `alert` is a warning with a red header
                        out["alert" if s["overlay"]["kind"] == "warning" and s["overlay"].get("rgb") else s["overlay"]["kind"]] += 1
                    out.update(s["accent"]["kind"] for s in [s] if s["accent"])
        elif f.kind == "photo" and since <= f.t < until:
            rgb = tuple(f.spec["rgb"] or ())
            if f.spec["scene"] == "sign" and not photo_needle(f.spec):
                out["sign"] += 1
            elif f.spec["scene"] == "car" and rgb in red:
                out["car"] += 1
            elif rgb in green or (f.spec["scene"] == "car" and rgb in green):
                out["green"] += 1
            elif rgb in red:
                out["red"] += 1
            out["wall"] += f.spec["scene"] == "wall"
        elif f.kind == "porch" and since <= f.t < until:
            for e in f.spec["events"]:
                out["courier"] += e["kind"] == "courier" and "drop_at" in e and e["drop_at"] is None
                out.update([e["kind"]] if e["kind"] in ("dog", "light", "car", "visitor") else [])
        elif f.kind == "mic":
            for s in f.spec["sounds"]:
                if since <= f.t + s["at"] < until:
                    out["beeper"] += s["kind"] in BEEPER_HZ
                    out["band"] += s["kind"] in ("ding", "ring", "bark", "timer")
                    out["loud"] += s.get("gain", 1.0) > 1.0
    return out


def test_every_scope_has_distractors_of_the_needles_colour_or_pitch(tiny, weeks):
    """Every asked scope (a part, a day, and for a `last` question the days after the last needle) holds the scale's planted distractors
    of each class, so a colour filter or a band filter shortlists and never gives the needle alone."""
    for plan in (tiny, *weeks["pilot"][:1], *weeks["standard"][:1]):
        mix = SCALES[plan.scale].mix
        for q in plan.questions:
            if q["probe"] is None or q["level"] not in ("part", "day", "week"):
                continue
            since = q["since"]
            if q["last"]:  # a scan from the newest reads what comes after the last needle before it hits one
                since = max(b for _, b in q["truth"]["moments"])
            got = classes_in(plan, q["source"], since, q["until"])
            who = f"{plan.scale} {q['qid']} ({q['level']}): {dict(got)}"
            source = {"phone": "phone", "screen": "screen", "porch": "porch", "mic": "mic"}[q["source"]]
            for kind, need in mix[source]["slot"].items():
                cls = {"accent": "toast", "greenwall": "wall", "redwall": "wall", "green": "green", "red": "red"}.get(kind, kind)
                if q["level"] == "part" and plan.scale != "tiny":
                    assert got[cls] >= (need if kind not in ("accent", "banner") else 1), f"{who}: fewer than {need} {kind}"
                else:
                    assert got[cls] >= 1, f"{who}: no {kind}"
            if q["source"] == "screen":
                assert got["confirm"] >= 1 and got["alert"] >= 1 and got["banner"] >= 1 and got["warning"] >= 1, who
            if q["source"] == "phone":  # (tiny plants its cars and red walls a day and not a part: one part may hold none, whatever the seed)
                reds = got["car"] + got["red"] >= 1 or (plan.scale == "tiny" and q["level"] == "part")
                assert got["sign"] >= 1 and reds and got["green"] + got["wall"] >= 1, who
            if q["source"] == "mic":
                assert got["beeper"] >= 1 and got["band"] >= 1 and got["loud"] >= 2, who


def level_of_sound(spec: dict) -> tuple[float, float]:
    """(the highest RMS over half seconds, the dominant pitch in Hz) of one sound alone: what the shortcut probe sorts a band by."""
    x = sound(spec).astype(np.float64)
    size = int(0.5 * RATE)
    if len(x) <= size:
        rms = float(np.sqrt(np.mean(x * x)))
    else:
        total = np.concatenate([[0.0], np.cumsum(x * x)])
        starts = np.arange(0, len(x) - size + 1, size // 2)
        rms = float(np.sqrt(((total[starts + size] - total[starts]) / size).max()))
    spectrum = np.abs(np.fft.rfft(x))
    spectrum[0] = 0.0
    return rms, float(spectrum.argmax() * RATE / len(x))


def test_a_sort_by_loudness_in_the_needles_band_does_not_put_the_needle_first(tiny, weeks):
    """For every sound question, in its scope (for a `last` question, the days after the last needle, which a scan from the newest reads
    first), the sounds with a dominant pitch in the needle's band that are louder than the needle: one or more at every level, so
    the needle is not first by loudness; and the band's shortlist (the others at a quarter of the needle's level or more) meets the
    probe's floors, 3 in a part, 8 in a day and 15 in the week."""
    floors = {"part": 3, "day": 8, "week": 15}
    for plan in (tiny, *weeks["pilot"], *weeks["standard"]):
        sounds = [(f.t + s["at"], f.t + s["at"] + s["dur"], *level_of_sound(s)) for f in plan.files if f.kind == "mic" for s in f.spec["sounds"]]
        for q in plan.questions:
            if q["source"] != "mic" or q["absent"]:
                continue
            lo, hi = q["since"], q["until"]
            if q["last"]:
                lo = max(lo, max(b for _, b in q["truth"]["moments"]))
            (moment,) = q["truth"]["moments"]
            band_lo, band_hi = q["probe"]["band"]
            needle = next(rms for t0, t1, rms, hz in sounds if t0 <= moment[1] and t1 >= moment[0])
            others = [(rms, hz) for t0, t1, rms, hz in sounds if lo <= t0 < hi and not (t0 <= moment[1] and t1 >= moment[0]) and band_lo <= hz <= band_hi]
            louder, shortlist = sum(rms > needle for rms, _ in others), sum(rms >= 0.25 * needle for rms, _ in others)
            who = f"{plan.scale} seed {plan.seed} {q['qid']} ({q['level']}): the needle's level {needle:.2f}"
            assert louder >= 1, f"{who}: no sound in its band is louder, so the band sorted by loudness puts it first"
            if plan.scale != "tiny":
                assert shortlist >= floors[q["level"]], f"{who}: {shortlist} others in its band at a quarter of its level, {floors[q['level']]} needed"
            else:
                assert shortlist >= 1, who


def test_a_porch_event_is_never_made_to_end_before_it_starts_and_a_window_too_short_for_one_is_refused():
    import random

    week = _Week(1, "tiny", DAYS)
    for window in ((5, 9), (0, 4), (3, 14)):
        event = week.event(random.Random(window[0]), "walker", 14, window)
        assert window[0] <= event["from"] < event["to"] <= window[1] - 1, (window, event)
    with pytest.raises(ValueError, match="needs 2 seconds"):
        week.event(random.Random(1), "walker", 12, (5, 6))


def test_the_plans_screen_has_acts_that_carry_the_cursor_over_and_frames_as_the_recorder_makes_them(tiny, weeks):
    """Each state names its level and where the cursor starts (where the last state left it) and its acts, in half seconds after its
    start, in order: a move to a new place near the last, a type of 1 to 3 characters on the typing line (a full one is followed by a scroll
    of 1 line, and any scroll empties it) or a scroll; an idle state has none. A segment's frames are the states' starts, the acts, the
    clock's minutes, at most one every half second and one at least every 60 s."""
    for plan in (tiny, *weeks["pilot"][:1], *weeks["standard"][:1]):
        screen = sorted((f for f in plan.files if f.kind == "screen"), key=lambda f: f.t)
        where, frames, moves, far = CURSOR_HOME, 0, 0, 0
        for f in screen:
            for state in f.spec["states"]:
                who = f"{plan.scale} {f.path} state at {state['from']} s"
                assert tuple(state["cursor"]) == where, f"{who}: the cursor does not go on from the last state"
                assert state["level"] in ACTIVITY and (state["level"] != "idle" or state["acts"] == []), who
                ticks = [act[0] for act in state["acts"]]
                assert ticks == sorted(set(ticks)) and all(1 <= tick < 2 * (state["to"] - state["from"]) for tick in ticks), who
                typed, lines = 0, 0
                cap = typing_cap(state, 0)
                for k, act in enumerate(state["acts"]):
                    kind = act[1]
                    assert kind in ACT_KINDS, who
                    if kind == "move":
                        x, y = act[2:]
                        assert 0 <= x < SCREEN[0] and 0 <= y < SCREEN[1] and distance(where, (x, y)) >= MOVE_PX[0], who
                        far += distance(where, (x, y)) > MOVE_PX[1]
                        moves += 1
                        where = (x, y)
                    elif kind == "type":
                        assert typed < act[2] <= min(cap, typed + 3), f"{who}: a type adds 1 to 3 visible characters to a line that shows at most {cap}"
                        typed = act[2]
                    else:
                        assert lines < act[2] <= lines + (1 if typed >= cap > 0 else 3), f"{who}: a scroll of 1 to 3 lines, of 1 after a full line"
                        lines, typed = act[2], 0
                        cap = typing_cap(state, lines)
                    assert not typed >= cap > 0 or k + 1 == len(state["acts"]) or state["acts"][k + 1][1] == "scroll", f"{who}: a full line is followed by a scroll"
            listed = screen_frames(f.spec, f.t)
            seconds = [fr.second for fr in listed]
            assert seconds == sorted(set(seconds)) and seconds[0] == 0 and seconds[-1] < f.spec["length_s"], f.path
            assert all(sec * 2 == int(sec * 2) for sec in seconds) and max(Counter(int(sec) for sec in seconds).values()) <= 2, f"{f.path}: at most 2 frames a second, on the ticks"
            assert all(b - a <= FLOOR_S for a, b in zip(seconds, seconds[1:])) and f.spec["length_s"] - seconds[-1] <= FLOOR_S, f.path
            assert {s["from"] for s in f.spec["states"]} <= set(seconds), f"{f.path}: a state's start is a frame"
            assert all(any(fr.state == k for fr in listed) for k in range(len(f.spec["states"])))
            minutes = {fr.second for fr in listed if fr.second and (int(f.t) + fr.second) % 60 == 0}
            assert minutes == {sec for sec in range(1, int(f.spec["length_s"])) if (int(f.t) + sec) % 60 == 0}, f"{f.path}: a minute of the clock is a frame"
            frames += len(listed)
        states = sum(len(f.spec["states"]) for f in screen)
        assert far <= 0.01 * moves, f"{plan.scale}: {far} of {moves} moves are not near the last place"
        assert 20 * states < frames, f"{plan.scale}: {frames} frames for {states} states: the owner is rarely seen at work"


def test_the_plans_acts_follow_the_apps_and_the_levels_are_a_chain_with_idle_stretches_and_the_active_shares_of_their_levels(weeks):
    """Over the pilot and standard weeks: an editor, a chat, slides and a sheet mostly type, a terminal types and scrolls, a browser mostly
    scrolls and moves; a state's level depends on the last (idle comes in stretches of a minute or more); the share of a state's seconds
    with an act is each level's own; and the error state is no different from any other."""
    kinds: dict[str, Counter] = {app: Counter() for app in APPS}
    shares: dict[str, list[float]] = {level: [] for level in ACTIVITY}
    follows, error_levels, longest = Counter(), Counter(), 0
    for plan in (*weeks["pilot"], *weeks["standard"][:1]):
        previous, end, idle = None, None, 0
        for f in sorted((f for f in plan.files if f.kind == "screen"), key=lambda f: f.t):
            for state in f.spec["states"]:
                seconds, level = state["to"] - state["from"], state["level"]
                kinds[state["app"]].update(act[1] for act in state["acts"])
                shares[level].append(len({tick // 2 for tick, *_ in state["acts"]}) / seconds)
                error_levels[level] += (state["overlay"] or {}).get("kind") == "error"
                if end is not None and f.t + state["from"] != end:  # (a new block: its chain starts afresh)
                    previous = None
                if previous is not None:
                    follows[previous, level] += 1
                idle = (idle if previous == "idle" else 0) + seconds if level == "idle" else 0
                longest = max(longest, idle)
                previous, end = level, f.t + state["to"]
    for app in ("editor", "chat", "slides", "sheet"):
        assert kinds[app]["type"] > 0.5 * sum(kinds[app].values()), f"{app} mostly types: {dict(kinds[app])}"
    assert kinds["terminal"]["type"] > 0.25 * sum(kinds["terminal"].values()) and kinds["terminal"]["scroll"] > 0.25 * sum(kinds["terminal"].values()), dict(kinds["terminal"])
    assert kinds["browser"]["scroll"] + kinds["browser"]["move"] > 0.7 * sum(kinds["browser"].values()) and kinds["browser"]["type"] < 0.15 * sum(kinds["browser"].values())
    assert all(sum(kinds[app].values()) > 500 for app in APPS), "every app is used"
    total = Counter()
    for (a, b), n in follows.items():
        total[a] += n
    stay = {level: follows[level, level] / total[level] for level in ACTIVITY}
    overall = {level: sum(follows[a, level] for a in ACTIVITY) / sum(total.values()) for level in ACTIVITY}
    assert all(stay[level] > overall[level] + 0.05 for level in ACTIVITY), f"a level depends on the last: {stay} against {overall}"
    assert stay["idle"] >= 0.6 and longest >= 120, f"idle comes in stretches: stays {stay['idle']:.0%}, the longest {longest} s"
    assert all(share == 0 for share in shares["idle"])
    mean = {level: sum(values) / len(values) for level, values in shares.items()}
    for level in ("light", "busy"):
        lo, hi = ACTIVE_SHARE[level]
        assert lo - 0.1 <= mean[level] <= hi + 0.02, f"{level}: {mean[level]:.0%} of a state's seconds have an act, drawn from {lo:.0%} to {hi:.0%}"
    assert mean["busy"] > mean["light"] + 0.2
    assert sum(error_levels.values()) >= 8 and sum(n > 0 for n in error_levels.values()) >= 2, f"the error state's level is the chain's: {dict(error_levels)}"


def test_no_frame_of_a_state_is_the_picture_of_the_one_before_it(tiny, weeks):
    """The recorder's `mpdecimate` never keeps a frame that is the last one again, so every frame the plan lists differs from the one
    before it within a state: a type adds a visible character that a decoration does not cover, a scroll moves the card, a move moves the
    cursor. Drawn from the layers (no encoder): all the tiny week, and an hour of a pilot day."""
    pilot = weeks["pilot"][0]
    days = days_before(date(2026, 11, 2), SCALES["pilot"].days)
    lo = at(days[0], 13 * 3600 + 30 * 60)
    for who, files in (("tiny", [f for f in tiny.files if f.kind == "screen"]),
                       ("a pilot hour", [f for f in pilot.files if f.kind == "screen" and lo <= f.t < lo + 3600])):
        pairs = 0
        for f in files:
            previous, k, layers = None, -1, None
            for frame in screen_frames(f.spec, f.t):
                if frame.state != k:
                    k, layers, previous = frame.state, screen_layers(f.spec["states"][frame.state]), None
                image = screen_frame(layers, frame)
                if previous is not None:
                    pairs += 1
                    assert ImageChops.difference(image, previous).getbbox() is not None, f"{who}: {f.path} frame at {frame.second} s is the one before it again"
                previous = image
        assert pairs > 1000, who


def test_tiny_is_written_in_30_seconds(folder, tiny):
    print(f"\ntiny: {len(tiny.files)} files written in {TIMES['write']:.1f} s "
          f"(the target: {WRITE_S:g} s{f', held to {SLACK:g} times that under xdist' if SLACK > 1 else ''})")
    assert TIMES["write"] <= WRITE_S * SLACK, f"the tiny week took {TIMES['write']:.1f} s to write, the target is {WRITE_S:g} (x{SLACK:g})"


def test_the_folder_holds_the_plans_files_and_logs_with_their_times(tiny, folder):
    found = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert found == {f.path for f in tiny.files} | {f"logs/{s}.jsonl" for s in SERVICES}, "no work files are left"
    for f in tiny.files:
        assert (folder / f.path).stat().st_mtime == pytest.approx(f.t, abs=1), f.path
    for service, lines in tiny.logs.items():
        text = (folder / "logs" / f"{service}.jsonl").read_text(encoding="utf-8")
        assert [json.loads(line) for line in text.splitlines()] == list(lines)
    # a photo and a clip carry their own time, and names follow capture: a screen segment and a mic segment are regular, a clip or a photo
    # is named by its start (and the name says the same time)
    for f in tiny.files:
        if f.kind in ("photo", "clip"):
            assert re.fullmatch(r"phone/(IMG|VID)_\d{8}_\d{6}\.(jpg|mp4)", f.path)
            assert datetime.strptime(f.path.split("_", 1)[1].rsplit(".", 1)[0], "%Y%m%d_%H%M%S").timestamp() == f.t
            if f.kind == "clip":
                with av.open(str(folder / f.path)) as container:
                    made = datetime.fromisoformat(container.metadata["creation_time"].replace("Z", "+00:00"))
                assert made.timestamp() == pytest.approx(f.t, abs=0.5), f.path
        if f.kind == "photo":
            with Image.open(folder / f.path) as image:
                exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
            assert exif[ExifTags.Base.DateTimeOriginal] == datetime.fromtimestamp(f.t).strftime("%Y:%m:%d %H:%M:%S"), f.path
            assert re.fullmatch(r"[+-]\d\d:\d\d", exif[ExifTags.Base.OffsetTimeOriginal]), f.path
        if f.kind == "mic":
            assert f.t % 900 == 0, f"{f.path} is not a regular segment"
        if f.kind == "screen":  # (it starts on a state change, at any second)
            assert re.fullmatch(r"screen/screen-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.mkv", f.path), f.path
        if f.kind == "porch":
            assert re.fullmatch(r"porch/porch-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.mp4", f.path)


def test_write_refuses_a_fixture_that_is_not_what_its_plan_says(tiny, tmp_path):
    sign = next(f for f in tiny.files if f.kind == "photo" and photo_needle(f.spec) == "sign")
    wall = next(f for f in tiny.files if f.kind == "photo" and f.spec["scene"] == "wall")
    image = photo_image(sign.spec)
    check_shades(image, "phone", "sign", sign.path)
    with pytest.raises(AssertionError, match="of the sign shade, which only a needle has"):
        check_shades(image, "phone", None, sign.path)
    with pytest.raises(AssertionError, match="of the sign shade, which only a needle has"):  # a sign is no red car
        check_shades(image, "phone", "car", sign.path)
    with pytest.raises(AssertionError, match="of the sign shade, not 3% to 50%"):
        check_shades(photo_image(wall.spec), "phone", "sign", wall.path)
    check_shades(photo_image(wall.spec), "phone", None, wall.path)

    ring = next(f for f in tiny.files if f.kind == "mic" and any(s["kind"] == "doorbell" for s in f.spec["sounds"]))
    alarm = next(f for f in tiny.files if f.kind == "mic" and any(s["kind"] == "alarm" for s in f.spec["sounds"]))
    alarm = dataclasses.replace(alarm, spec={**alarm.spec, "sounds": [s for s in alarm.spec["sounds"] if s["kind"] != "doorbell"]})  # (the same segment may ring too)
    samples = mic_samples(ring.spec)
    check_sounds(ring, samples)
    with pytest.raises(AssertionError, match=r"the doorbell at .* is heard in no window"):
        check_sounds(ring, np.zeros(len(samples), dtype=np.float32))
    shift = lambda s: {**s, "at": s["at"] + (150.0 if s["at"] < 450 else -150.0)} if s["kind"] == "doorbell" else s  # noqa: E731
    with pytest.raises(AssertionError, match=r"sounds like .* Hz and overlaps no such doorbell"):  # a chime where the plan has none
        check_sounds(ring, mic_samples({**ring.spec, "sounds": [shift(s) for s in ring.spec["sounds"]]}))
    check_sounds(alarm, mic_samples(alarm.spec))
    with pytest.raises(AssertionError, match=r"the alarm at .* is heard in no window"):
        check_sounds(alarm, np.zeros(int(alarm.spec["length_s"] * RATE), dtype=np.float32))

    clip = next(f for f in tiny.files if f.kind == "clip")  # the same clip twice: `remember` would keep one
    twin = File(clip.path.replace("VID_", "VID_twin_"), clip.source, clip.t, None, "clip", clip.spec)
    twins = Plan(tiny.seed, tiny.scale, tiny.days, (clip, twin), {s: () for s in SERVICES}, ())
    with pytest.raises(AssertionError, match="has the bytes of"):
        write(twins, tmp_path)


# -- the graders ----------------------------------------------------------------------------------------------------------------

def test_the_graders_grade_as_the_table_says():
    def spec(kind: str, truth=None, time=None, tol_min=None, wrong=(), ignore=(), date_=None) -> Grade:
        return Grade(kind, truth, time, None if tol_min is None else tol_min * 60.0, tuple(wrong), tuple(ignore), date_)

    hours = ["12:00", "18:00"]  # what a question's hint holds: "in the afternoon (12:00 to 18:00)"
    table = [(spec("number", "47", wrong=["23"], ignore=hours),
              ["47", "GATE 47", "forty-seven", "GATE 47, in the afternoon (12:00 to 18:00)", "The sign said 47 on 28 October"],
              ["23", "47 or 23", "unknown", "74", "47 or 1234", "I think 47, maybe 51"]),
             (spec("number", "4417"), ["4417", "The code is four four one seven", "Once I checked, it was 4417"],
              ["4417 or 1234", "1234", "4417, 4471"]),
             (spec("count", "3", ignore=["731", "18:00", "24:00"]),
              ["3", "ERR-731 three times", "3 times on 28 October", "ERR-731 thrice", "3 times in the evening (18:00 to 24:00)"],
              ["2", "3 or 4", "unknown", "I counted 3, and 5 more on another day"]),
             (spec("count", "2", ignore=["731"]), ["twice", "ERR-731 was logged twice", "2"], ["thrice", "2 or 3", "once"]),
             (spec("count", "1"), ["once", "1 time"], ["twice", "1 or 2"]),
             (spec("time", "14:03:10", "14:03:10", 1, ["13:05"], hours),
              ["14:03", "2:04 pm", "2026-10-28T14:03:10-04:00", "14:03:30", "14:03, in the afternoon (12:00 to 18:00)"],
              ["14:10", "13:05 and 14:03", "unknown", "I heard sounds at 14:01, 14:03:30 and 14:07", "14:03 and 17:00"]),
             (spec("time", "14:03:10", "14:03:10", 1, ["13:05"], date_="2026-10-29"),  # a week question: the day it was on, and no other
              ["Thursday 29 October at 14:03", "2026-10-29 14:03", "14:03 on October 29", "Thu 29 Oct, 14:03:30"],
              ["14:03", "Wednesday 28 October at 14:03", "Thursday 29 October at 14:03 and Friday 30 October", "29 October at 13:05", "unknown"]),
             (spec("code_time", "E-4417", "14:03:20", 1, ignore=hours),
              ["E-4417 at 14:03", "e4417, 2:03 PM", "E-4417 at 14:03 (12:00 to 18:00)"],
              ["E-4417", "14:03", "E-4471 at 14:03", "E-4417 at 14:03, then W-5521 at 14:10", "E-4417 at 14:03 and E-5521"]),
             (spec("code_time", "E-4417", "14:03:20", 1, date_="2026-10-29"),
              ["E-4417 at 14:03 on Thursday 29 October", "On 29 October, e4417 appeared at 14:03:40"],
              ["E-4417 at 14:03", "E-4417 at 14:03 on Friday 30 October"]),
             (spec("absent", ignore=["845", *hours]),
              ["unknown", "None.", "Never", "No ring was recorded", "0", "There was no green sign", "The doorbell did not ring",
               "Nobody rang the doorbell", "I found no green sign", "ERR-845 was never logged", "There were zero ERR-845 lines",
               "I could not find it", "No doorbell rang between 12:00 and 18:00", "No one rang the doorbell",
               "I didn't see one", "No, not a single one", "There was not one"],
              ["47", "14:03", "the password is hunter2", "Not sure, maybe 47", "It rang, not at 14:03", "The code is ERR-845"])]
    assert {s.rule for s, _, _ in table} == set(RULES) - {"choice", "set"}  # EgoLife's kind: test_grade.py; the real scenarios' "find every one": test_set_rule.py
    for one, right, wrong in table:
        what = f"{one.rule} {one.truth}"
        assert all(grade.grade(one, a) for a in right), f"{what} must accept {[a for a in right if not grade.grade(one, a)]}"
        assert not any(grade.grade(one, a) for a in wrong), f"{what} must refuse {[a for a in wrong if grade.grade(one, a)]}"
        assert grade.grade(one, None) is False, "no answer is wrong"
    with pytest.raises(ValueError, match="one of number, count"):
        grade.grade(spec("poem"), "47")
    # an answer that abstains has no number or time but the question's own, and none at all is `abstained`'s default
    assert grade.abstained("Not sure, maybe 47") is False and grade.abstained("Not sure, maybe 47", ["47"]) is True
    assert grade.abstained(None) is False and grade.abstained("it is 47") is False
    # a date or an ISO time is no number, and an ISO time is read as written
    assert grade.numbers("Tuesday 28 October 2026 at 14:03, ERR-731 three times, 2026-10-28") == ["731", "3"]
    assert grade.numbers("October 28, Tuesday 28, the 28th of October") == []
    assert grade.minutes("2026-10-28T14:03:10-04:00") == [14 * 60 + 3 + 10 / 60]
    assert grade.asked("How many times did the payments service log ERR-731 on Friday 30 October, in the evening (18:00 to 24:00)?") \
        == ["731", "18:00", "24:00"]


def test_a_week_questions_right_answer_passes_its_grader_and_the_earlier_needles_do_not(weeks):
    """The truth of each question of the pilot's and the standard's weeks is accepted by its own grade, with its day; the same time
    on another day, and an earlier needle's answer, are refused."""
    for plan in (weeks["pilot"][0], weeks["standard"][0]):
        for q in plan.questions:
            g = q["grade"]
            graded = grade_for(q)
            if g["kind"] in ("time", "code_time") and g["date"]:
                day = date.fromisoformat(g["date"])
                right = f"{date_text(day)}, " + (f"{g['truth']} at {g['time']}" if g["kind"] == "code_time" else g["time"])
                assert grade.grade(graded, right), f"{q['qid']}: {right!r}"
                other = date_text(plan.days[0] if plan.days[0] != day else plan.days[1])
                assert not grade.grade(graded, right.replace(date_text(day), other)), q["qid"]
                assert not grade.grade(graded, right.split(", ", 1)[1]), f"{q['qid']}: a time with no day is wrong"
            elif g["kind"] == "number" and q["level"] == "week":
                assert grade.grade(graded, g["truth"]) and not any(grade.grade(graded, w) for w in g["wrong"]), q["qid"]
                assert g["wrong"], f"{q['qid']}: the earlier needles are its wrong answers"
