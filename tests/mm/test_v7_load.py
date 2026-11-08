"""Ported from tests/test_haystack.py and tests/test_haystack_media.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in those files' orders: the tests of
the haystack that need Multimem's reader or a daemon (the rest are tests/test_v7_plan.py and tests/test_v7_media.py). The tiny week is built
once by the scenario (V7.build, about 35 s) and ingested once by the system (`Mm.ingest`, conftest's `week` and `store`), and each test that
asks a daemon asks one over a copy of that store. What changed, by plan 4.6 and 7.4:
- `haystack.load` is `Mm.ingest`; a phone file that is not at its own time is a `misdated` entry of the report, no longer an AssertionError
  (test_ingest_reports_a_phone_file...);
- `haystack.windows_in` is the table of windows (mmsb/systems/mm/estimate.py), counted from the files by the reader's own demux; the mic's count is
  every window of the sound, so an upper bound (the old count used the plan's loud spans): its equality with what the reader reads is `<=` now;
- the pilot week cut down for the `last` test is written by `write` and ingested by a dump of its files alone.
Every other expectation is the source's."""

from __future__ import annotations

import dataclasses
import os
import time
from datetime import date, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from fakes import span  # noqa: E402
from PIL import Image  # noqa: E402

from multimem import decider, media, reader  # noqa: E402
from multimem.store import Store  # noqa: E402

from mmsb.core.config import Config  # noqa: E402
from mmsb.core.contract import Dump  # noqa: E402
from mmsb.core.select import config_questions  # noqa: E402
from mmsb.scenarios.v7_screen.manifest import entries_of, measure_disk  # noqa: E402
from mmsb.scenarios.v7_screen.media_screen import APPS, screen_frames, write_screen  # noqa: E402
from mmsb.scenarios.v7_screen.plan import TOL_S, at, days_before  # noqa: E402
from mmsb.scenarios.v7_screen.week import plan as make_plan  # noqa: E402
from mmsb.scenarios.v7_screen.write import write  # noqa: E402
from mmsb.systems.mm.estimate import audio_windows  # noqa: E402
from mmsb.systems.mm.estimate import windows_in as table_windows  # noqa: E402
from mmsb.systems.mm.models import Model  # noqa: E402
from mmsb.systems.mm.system import Mm  # noqa: E402

DAYS_OF = [date(2026, 10, 30), date(2026, 10, 31)]
LOAD_S = 8.0  # a tiny week is ingested in this many seconds (the old `load` took 8: the daemon starts, the files are remembered, the daemon stops)
# a single process (`-n 0`) is held to this; under xdist (`-n 4`, with another suite or a GPU run possibly sharing the machine) the bound is
# SLACK times as loose, so that the test fails for a daemon that got several times slower and never for a busy machine
SLACK = 4.0 if os.environ.get("PYTEST_XDIST_WORKER") else 1.0
pytestmark = pytest.mark.xdist_group("v7-mm")


@pytest.fixture(scope="module")
def tiny(week):
    """The plan of the tiny week (the questions as dicts, with the fake decider's words): what the files were written from."""
    return fakes.V7.plan_of(week[0])


def opened_over(dump, store, homes, request):
    """A session over a copy of `store`, closed at the end of the test."""
    session = fakes.MM.open(store, dump, fakes.context(homes))
    request.addfinalizer(session.close)
    return session


@pytest.fixture
def opened(week, store, tmp_path, request):
    return opened_over(week[1], store, tmp_path / "homes", request)


@pytest.fixture
def api(opened):
    return opened.daemon.api


def test_each_model_is_asked_only_about_what_it_reads(tiny, week):
    abouts = decider.known_abouts()
    assert len(abouts) == 8
    questions = week[2]
    for about in abouts:
        assert about.files == () and about.speech is False, f"{about.name} reads a document or speech: a template is needed"
        system = Mm(models={about.name: Model(about.name, "/models/" + about.name, about)})
        described = system.describe(about.name)
        deaf = {"text", "records", "image", "video"}
        assert described.reads == (deaf | {"audio"} if about.name == "d1-Omni-2B" else deaf), about.name
        asked = config_questions(questions, Config(f"mm/{about.name}", "mm", about.name, False), described)
        assert len(asked) == (19 if about.name == "d1-Omni-2B" else 15), about.name
        assert {q.tags["modality"] for q in asked} <= described.reads


# -- through the reader ----------------------------------------------------------------------------------------------------------

def scan(api, q: dict, **more) -> dict:
    """The shipped `ask` as a job over the question's source and range with its fake question, as `test_day`'s `scan`."""
    params = {"question": q["fake"]["question"], "sources": [q["source"]], "since": q["since"], "until": q["until"],
              "limit": q["fake"]["limit"], "oldest_first": q["fake"]["oldest_first"], **more}
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


SLICE_S = 100  # a screen question is read over this many seconds either side of its needle


def screen_slice(plan, q: dict) -> dict:
    """The question over the screen round its needle (round the first segment of its range for an absent one): a range of the screen is a
    frame for each act and each minute of the clock, over three thousand in a tiny week and 25 ms each through the reader, and a slice shows the same words finding the same needle."""
    if q["truth"]["moments"]:
        centre = q["truth"]["moments"][0][0]
    else:
        centre = min(f.t for f in plan.files if f.kind == "screen" and q["since"] <= f.t < q["until"]) + 60
    return {**q, "since": max(q["since"], centre - SLICE_S), "until": min(q["until"], centre + SLICE_S)}


def overlaps(sighting: dict, moment: list[float]) -> bool:
    start, end = span(sighting)
    return start <= moment[1] + TOL_S and end >= moment[0] - TOL_S


def test_the_fake_decider_finds_every_needle_at_its_time(api, tiny, opened):
    for q in tiny.questions:
        q = screen_slice(tiny, q) if q["source"] == "screen" else q
        result = scan(api, q)
        sightings = result["sightings"]
        in_range = [s for s in sightings if span(s)[0] < q["until"] and span(s)[1] >= q["since"]]
        moments, others, check = q["truth"]["moments"], q["truth"]["others"], q["fake"]["check"]
        where = f"{q['qid']} ({q['fake']['question']}): {[span(s) for s in sightings]}"
        if check == "none":
            assert in_range == [], where
        elif check == "count":
            assert len(in_range) == int(q["truth"]["value"]), where
            assert all(any(overlaps(s, m) for m in moments) for s in in_range), f"a sighting of no needle: {where}"
        elif check == "look":  # the version current at the time was written before it: one sighting, whatever its time
            (one,) = sightings
            assert f"retention_days = {q['truth']['value']}\n" in api.call("look", **one["look"])["text"], where
        elif check == "last":  # the newest sighting in the range is the last needle's, and every one is some needle's
            assert in_range, where
            newest = max(in_range, key=lambda s: span(s)[1])
            assert any(overlaps(newest, m) for m in moments), f"the newest sighting is not the last needle: {where}"
            assert all(any(overlaps(s, m) for m in (*moments, *others)) for s in in_range), f"a sighting of no needle: {where}"
            assert all(any(overlaps(s, m) for s in in_range) for m in moments), f"the last needle is not found in all its moments: {where}"
        else:
            wanted = moments[-1:] if q["template"] == "log-last" else moments
            assert in_range and all(any(overlaps(s, m) for s in in_range) for m in wanted), where
            if q["template"] != "log-last":  # (a deploy's start line has its build too, so it is a hit)
                assert all(any(overlaps(s, m) for m in moments) for s in in_range), f"a sighting of no needle: {where}"
        if q["template"] in ("log-count", "absent-log"):
            code = q["fake"]["question"].split(":", 1)[1]
            everywhere = api.call("find", text=code, types=["record"])["results"]
            in_slot = api.call("find", text=code, types=["record"], since=q["since"], until=q["until"])["results"]
            if q["absent"]:
                assert everywhere == [], f"{q['qid']}: {code} is never logged"
            else:
                assert len(in_slot) == int(q["truth"]["value"]), f"{q['qid']}: the lines of {code} in the slot"
                assert 1 <= len(everywhere) - len(in_slot) <= 3, f"{q['qid']}: and one to three on other days"
        if q["source"] in ("screen", "porch", "phone", "mic") and len(sightings) < q["fake"]["limit"]:  # the whole range was read
            planned = sum(table_windows(opened.table, q["source"], [(q["since"], q["until"])]).values())
            assert result["progress"]["windows"] <= planned, f"{q['qid']}: the gate only ever reads fewer windows than the plan counts"
            if q["source"] != "screen" or (q["level"] == "part" and not q["absent"]):  # (the screen's frames are many: its part question stands for the rest)
                read = scan(api, q, still_s=0)["progress"]["windows"]  # a frame the step picks, as the plan counts them
                if q["source"] == "mic":  # (a window of a quiet room is skipped by the reader, and counted: the table is an upper bound)
                    assert read <= planned, f"{q['qid']}: the reader read {read} windows of {q['source']}, the table counts {planned}"
                else:
                    assert read == planned, f"{q['qid']}: the reader read {read} windows of {q['source']}, the table counts {planned}"


def cut_down(plan, extra: dict[str, int]):
    """`plan` with only the files that hold the needles of the `last` question of each template of `extra` (the last one and the earlier
    ones) and, as many as it says, the files of the source nearest to them in time, no logs, and those questions: a week that is
    written and loaded in seconds and still holds, in each question's range, several needles."""
    keep, questions = {}, []
    for template, nearest in extra.items():
        q = next(q for q in plan.questions if q["template"] == template and q["last"])
        questions.append(q)
        spans = [tuple(m) for m in (*q["truth"]["moments"], *q["truth"]["others"])]
        files = [f for f in plan.files if f.source == q["source"] and f.kind in ("photo", "screen", "porch", "mic")]
        holds = lambda f: any(f.t <= b + 0.5 and f.t + f.spec.get("length_s", 0) >= a - 0.5 for a, b in spans)  # noqa: E731
        mine = [f for f in files if holds(f)]
        near = sorted((f for f in files if f not in mine), key=lambda f: min(abs(f.t - a) for a, _ in spans))[:nearest]
        keep.update({f.path: f for f in (*mine, *near)})
    return dataclasses.replace(plan, files=tuple(sorted(keep.values(), key=lambda f: (f.t, f.path))), questions=tuple(questions),
                               logs={})


def test_a_last_question_whose_range_holds_several_needles_gets_the_newest_through_the_reader(tmp_path, request):
    """The tiny week has one needle to a template, so its `last` questions never choose. Here the pilot week cut down to the sign
    photos, the error screens, the parcel clips and the smoke alarms of the four `last` questions (an earlier needle or more in each
    question's range, among the distractors nearest to them) is written, loaded and asked with the fake decider: the newest sighting is
    the last needle, every earlier needle is found and nothing else, and an ask for the one newest sighting (limit 1, as an agent asks for
    "the last") returns the last needle alone."""
    plan = make_plan(1, "pilot", days_before(date(2026, 11, 2), 3))
    cut = cut_down(plan, {"photo-sign": 6, "screen-error": 2, "porch-parcel": 2, "mic-alarm": 0})
    began = time.perf_counter()
    write(cut, tmp_path / "data")
    dump = Dump("cut", tmp_path / "data", "v7_screen", entries_of(cut, measure_disk(tmp_path / "data")), ())
    fakes.ingested(dump, tmp_path / "store")
    api = opened_over(dump, tmp_path / "store", tmp_path / "homes", request).daemon.api
    print(f"\nthe cut-down pilot week: {len(cut.files)} files written and loaded in {time.perf_counter() - began:.1f} s")
    assert {q["template"] for q in cut.questions} == {"photo-sign", "screen-error", "porch-parcel", "mic-alarm"}
    for q in cut.questions:
        earlier = q["truth"]["others"]
        assert q["last"] and q["level"] == "week" and len(earlier) >= 1, f"{q['qid']}: {len(earlier)} earlier needle spans in the range"
        asked = time.perf_counter()
        result = scan(api, q)
        print(f"{q['qid']}: {len(result['sightings'])} sightings, {result['progress']['windows']} windows, {time.perf_counter() - asked:.1f} s")
        in_range = [s for s in result["sightings"] if span(s)[0] < q["until"] and span(s)[1] >= q["since"]]
        where = f"{q['qid']} ({q['fake']['question']}): {[span(s) for s in result['sightings']]}"
        newest = max(in_range, key=lambda s: span(s)[1])
        assert any(overlaps(newest, m) for m in q["truth"]["moments"]), f"the newest sighting is not the last needle's: {where}"
        assert all(any(overlaps(s, m) for s in in_range) for m in (*q["truth"]["moments"], *earlier)), f"a needle is not found: {where}"
        assert all(any(overlaps(s, m) for m in (*q["truth"]["moments"], *earlier)) for s in in_range), f"a sighting of no needle: {where}"
        assert any(not any(overlaps(s, m) for m in q["truth"]["moments"]) for s in in_range), f"only the last needle was found: {where}"
        one = api.call("jobs/start", script="ask", params={"question": q["fake"]["question"], "sources": [q["source"]], "since": q["since"],
                                                           "until": q["until"], "limit": 1}, wait_s=50)
        while one["state"] in ("queued", "running"):
            one = api.call("jobs/get", job=one["job"], wait_s=50)
        (only,) = one["result"]["sightings"]
        assert any(overlaps(only, m) for m in q["truth"]["moments"]), f"limit 1 does not return the last needle of {q['qid']}: {span(only)}"


def test_the_phones_files_are_remembered_with_no_time_at_the_time_they_carry(opened, tiny):
    """A photo's EXIF and a clip's `creation_time` are the time `remember` stores it at (`time_from` metadata), through `write` and `ingest`:
    `ingest` gave the phone's files no `t`, and reports one that is not at `f.t`; every other source is given its `t`."""
    store = Store(opened.home)
    try:
        items = store.items_of("phone")
        phone = [f for f in tiny.files if f.source == "phone"]
        assert len(items) == len(phone) and {f.kind for f in phone} == {"photo", "clip"}
        for f, item in zip(sorted(phone, key=lambda f: f.t), items):
            assert item.t0 == pytest.approx(f.t, abs=1) and item.meta["time_from"] == "metadata", f.path
            assert item.kind == ("image" if f.kind == "photo" else "video"), f.path
        for source in ("screen", "porch", "mic"):
            assert {i.meta.get("time_from") for i in store.items_of(source)} == {"given"}, source
    finally:
        store.close()


def test_ingest_reports_a_phone_file_that_carries_no_time_of_its_own_as_misdated_and_stores_the_rest(week, tmp_path):
    _, dump, _ = week
    photo = next(e for e in dump.entries if e.kind == "photo")
    bare = tmp_path / "bare" / photo.path  # the same picture with no EXIF, as a messenger strips it: remember falls back to its mtime
    bare.parent.mkdir(parents=True)
    Image.open(dump.root / photo.path).convert("RGB").save(bare, quality=90)
    os.utime(bare, (photo.t, photo.t))
    only = Dump("bare", tmp_path / "bare", "v7_screen", (photo,), ())
    report = fakes.MM.ingest(only, tmp_path / "store")
    assert [(r.path, "not at its own time" in r.reason) for r in report.misdated] == [(photo.path, True)] and report.refused == ()


def test_the_table_counts_a_ranges_windows_by_source(tiny, opened):
    """What the fake decider test checks against the reader for the media (every question's range), by hand for the rest: a second of the
    screen that has a frame (`screen_frames`: the reader takes the first frame of a second, and there are up to 2), a second of a porch
    clip, the photos in the range, every note and the config's versions, the records' windows."""
    week = (at(DAYS_OF[0], 0), at(DAYS_OF[-1], 86400))

    def windows_in(tiny_, source, since, until):  # the table's count, all kinds together (each source of the week has one kind of window)
        return sum(table_windows(opened.table, source, [(since, until)]).values())

    states = sum(len(f.spec["states"]) for f in tiny.files if f.kind == "screen")
    frames = sum(len(screen_frames(f.spec, f.t)) for f in tiny.files if f.kind == "screen")
    seconds = sum(len({int(fr.second) for fr in screen_frames(f.spec, f.t)}) for f in tiny.files if f.kind == "screen")
    assert states < seconds < frames and windows_in(tiny, "screen", *week) == seconds and windows_in(tiny, "screen", week[0], week[0]) == 0
    one = next(f for f in tiny.files if f.kind == "screen")
    later = next(fr for fr in screen_frames(one.spec, one.t) if fr.second >= 1)  # (the second whole second with a frame)
    assert windows_in(tiny, "screen", one.t, one.t + later.second) == 1
    assert windows_in(tiny, "screen", one.t, one.t + int(later.second) + 1) == 2
    # two frames in one second are one window; a range that holds the second frame of a second only still holds the second
    t = datetime(2026, 10, 30, 10, 0, 0).timestamp()
    frames = [0, 1, 1.5, 2, 2.5]  # a state's start, an act at 1 s and 1.5 s, and 2 and 2.5 s: the segment's frames
    by_hand = {"sources": {"screen": [{"k": "frames", "t": t, "frames": frames}]}, "decode_s": {}}
    assert table_windows(by_hand, "screen", [(t, t + 30)])["video"] == 3 and table_windows(by_hand, "screen", [(t + 1.5, t + 2.5)])["video"] == 2
    assert table_windows(by_hand, "screen", [(t + 1.5, t + 1.6)])["video"] == 1
    assert windows_in(tiny, "porch", *week) == sum(int(f.spec["length_s"]) for f in tiny.files if f.kind == "porch")
    photos = [f.t for f in tiny.files if f.kind == "photo"]
    first = min(photos)
    clip_frames = sum(week[0] <= f.t + s < first + 1 for f in tiny.files if f.kind == "clip" for s in range(3))
    assert windows_in(tiny, "phone", week[0], first + 1) == 1 + clip_frames, "a photo is an instant"
    assert windows_in(tiny, "phone", week[1], week[1] + 1) == 0, "and is read by the range that holds it only"
    assert windows_in(tiny, "phone", *week) == len(photos) + 3 * sum(f.kind == "clip" for f in tiny.files)
    assert windows_in(tiny, "notes", *week) == sum(f.kind == "note" for f in tiny.files)
    assert windows_in(tiny, "config", *week) == sum(f.kind == "config" for f in tiny.files)
    assert windows_in(tiny, "config", week[1], week[1] + 1) == 1, "the version current at a time is read"
    assert windows_in(tiny, "logs", *week) >= 1 and windows_in(tiny, "logs", week[0], week[0] + 1) == 0
    mics = [f for f in tiny.files if f.kind == "mic"]
    planned = sum(len(audio_windows(f.spec["length_s"])) for f in mics)  # every window of a sound, the quiet ones the reader skips too: an upper bound
    assert planned <= windows_in(tiny, "mic", *week) <= planned + len(mics), "the length mm indexes (the packets') may run past the segment's by a window"
    with pytest.raises(ValueError, match="a source is one of"):
        windows_in(tiny, "tv", *week)


def test_each_written_screen_file_has_the_frames_of_its_list_and_the_reader_sees_more_windows_than_states(week, tiny):
    """The file's frames are `screen_frames`' (`check_screen` pins their seconds too): the reader's windows of it (the real profile, with the
    change gate) are more than its states, since a moved cursor, a typed character and a scroll are changes, and at most its seconds with a
    frame, which is what `windows_in` counts."""
    screen = [f for f in tiny.files if f.kind == "screen"][:6]
    more = 0
    for f in screen:
        path = week[1].root / f.path
        assert len(reader._times(path)) == len(screen_frames(f.spec, f.t)), f.path
        item = SimpleNamespace(kind="video", t0=f.t, t1=f.t + f.spec["length_s"], source="screen", id=1)
        windows = list(reader._video(SimpleNamespace(path_of=lambda _, path=path: path), item, dict(reader.PROFILE), None, None, False))
        states, seconds = len(f.spec["states"]), len({int(fr.second) for fr in screen_frames(f.spec, f.t)})
        assert states <= len(windows) <= seconds, f"{f.path}: {len(windows)} windows, {states} states, {seconds} seconds with a frame"
        more += len(windows) > states
    assert more >= len(screen) - 1, "the cursor and the clock add windows to nearly every segment"


def test_tiny_is_ingested_in_8_seconds(store):
    report = fakes.REPORTS[str(store)]
    print(f"\ntiny: ingested in {report.seconds:g} s (the target: {LOAD_S:g} s{f', held to {SLACK:g} times that under xdist' if SLACK > 1 else ''})")
    assert report.seconds <= LOAD_S * SLACK, f"the tiny week took {report.seconds:g} s to ingest, the target is {LOAD_S:g} (x{SLACK:g})"
    assert (report.refused, report.misdated, report.gpu_seconds) == ((), (), 0.0) and report.bytes_written > 0


def test_a_phones_photo_and_clip_carry_the_time_mm_reads_from_them(week):
    """`media.recorded`, the time Multimem takes a file at when it is given none, is the plan's time for a photo (EXIF) and a clip (creation_time)."""
    _, dump, _ = week
    phone = [e for e in dump.entries if e.kind in ("photo", "clip")]
    assert {e.kind for e in phone} == {"photo", "clip"}
    for e in phone:
        assert media.recorded(dump.root / e.path, "image/jpeg" if e.kind == "photo" else "video/mp4") == pytest.approx(e.t, abs=0.5), e.path



# -- the screen's runs of still frames, through the reader ----------------------------------------------------------------------

# a single process (`-n 0`) is held to the targets; under xdist (`-n 4` runs four workers, and another suite or a GPU run may share the machine) the
# bounds are SLACK times as loose, so that the test fails for a writer that got several times slower and never for a busy machine
START = datetime(2026, 10, 28, 10, 0, 0).timestamp()  # a screen segment starts on a minute: a 30 s state's start is the only frame its clock adds


def states_of(*spans: tuple[int, int, list]) -> list[dict]:
    return [{"from": a, "to": b, "app": APPS[k % len(APPS)], "overlay": None, "accent": None, "seed": k, "cursor": [50 * k, 60],
             "acts": acts} for k, (a, b, acts) in enumerate(spans)]


def windows_of(path: Path, start: float, length: int) -> list:
    """The reader's windows of a written segment at its real profile (a frame a second, a run of frames that look the same one window)."""
    item = SimpleNamespace(kind="video", t0=start, t1=start + length, source="screen", id=1)
    return list(reader._video(SimpleNamespace(path_of=lambda _: path), item, dict(reader.PROFILE), None, None, False))


def test_the_cursor_typing_and_scrolling_split_the_readers_runs_of_still_frames(tmp_path):
    """A segment of 3 still states: with no acts it is a window a state (the clock's minute at a state's start joins none); with the
    owner's cursor moving, typing and scrolling it is more, and never more than the seconds that have a frame (the reader takes the first
    frame of a second, and a segment has up to 2 a second)."""
    still = states_of((0, 30, []), (30, 60, []), (60, 90, []))
    moving = states_of((0, 30, [[2 * s, "move", 100 + 60 * s, 200 + 20 * s] for s in (2, 3, 4, 10, 11)]),
                       (30, 60, [[2 * s, "type", 3 * s] for s in (1, 2, 3, 8)] + [[40, "scroll", 2]]),
                       (60, 90, sorted([[2 * s, "scroll", s - 3] for s in (4, 5, 6, 7)] + [[9, "type", 2]])))
    for name, states in (("still", still), ("moving", moving)):
        spec = {"length_s": 90, "states": states}
        path = tmp_path / f"{name}.mkv"
        write_screen(path, spec, START, tmp_path / "w")
        windows, frames = windows_of(path, START, 90), screen_frames(spec, START)
        if name == "still":
            assert len(windows) == len(states)
        else:
            seconds = {int(f.second) for f in frames}
            assert len(frames) > len(seconds), "a second can have 2 frames"
            assert len(states) < len(windows) <= len(seconds), (len(windows), len(seconds))
            assert len(windows) >= len(states) + 8, "each move, line typed and scroll is a window of its own"


def test_a_minute_of_the_clock_alone_does_not_split_a_run_but_the_gates_30_s_does(tmp_path):
    """The clock's digits are a few pixels of a cell: its minute joins the run of a still state (15 s in), and `still_s` ends the run at
    30 s, so a still state costs a window at its start and one at the first frame 30 s or more on, the clock's next minute (75 s in), in
    every app."""
    for k, app in enumerate(APPS):
        spec = {"length_s": 90, "states": [{"from": 0, "to": 90, "app": app, "overlay": None, "accent": None, "seed": 3}]}
        path = tmp_path / f"c{k}.mkv"
        write_screen(path, spec, START + 45, tmp_path / "w")  # the clock changes at 15 s and 75 s
        assert [f.second for f in screen_frames(spec, START + 45)] == [0, 15, 75]
        assert [round(w.at) for w in windows_of(path, START + 45, 90)] == [0, 75], app
