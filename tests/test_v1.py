"""V1, the security cameras, from its real source on disk (MEVA): the tiny dump built (the first ten minutes of three cameras, each as one MP4
joined from its clips by stream copy, with names, times, lengths, frames and metadata checked), every answer derived again here by a plain
reader of the label files (never the scenario's), the grader taking each truth and refusing planted wrong answers, the leak probe, and
`mmmb build`, `run` and `report` over it with a stand-in agent. Skips, and says so, without the source."""

from __future__ import annotations

import functools
import json
import os
import re
import threading
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmmb import scenarios
from mmmb.core import fs, grade
from mmmb.core.agents import claude
from mmmb.core.contract import ManifestEntry
from mmmb.core.runfolder import RunFolder
from mmmb.core.runner import Agent
from mmmb.scenarios import realdump
from mmmb.scenarios.v1_security import source as v1_source
from mmmb.scenarios.v1_security.questions import questions_of
from mmmb.scenarios.v1_security.source import LABELS, SOURCE, Event, read_acts

if not (SOURCE / LABELS / "kitware" / "status.csv").is_file() or not (SOURCE / "video" / "2018-03-12" / "10").is_dir():
    pytest.skip(f"the MEVA source is not at {SOURCE}, so V1 is not built", allow_module_level=True)

V1 = scenarios.get("v1_security")
pytestmark = pytest.mark.xdist_group("v1")
REAL_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # (the `world` fixture puts a made-up file in its place)
MIDNIGHT = datetime(2018, 3, 12).timestamp()
CAMS = {"cam1": "G336", "cam2": "G341", "cam3": "G436"}
LIMITS = {"tiny": 10 * 60 + 10, "standard": 11 * 60 + 15}  # the minute of the day each scale's clips begin before
GAP = 10 * 3600 + 37 * 60  # the recorder's gap is from 10:35 to 10:40: a clip that begins after this is in a camera's second file
ACT = re.compile(r"'act2': \{'(\w+)': [\d.]+\}.*?'src_status': '(\w+)', 'timespan': \[\{'tsr0': \[(\d+), (\d+)\]\}\]")
NAME = re.compile(r"2018-03-12\.(\d\d)-(\d\d)-(\d\d)\.(\d\d)-(\d\d)-(\d\d)\.\w+\.(G\d+)")
PHRASES = {"person_opens_trunk": "open a car trunk", "person_closes_trunk": "close a car trunk", "person_opens_vehicle_door": "open a car door",
           "person_closes_vehicle_door": "close a car door", "person_embraces_person": "hug", "vehicle_reverses": "reverse",
           "vehicle_makes_u_turn": "make a U-turn", "vehicle_turns_left": "turn left", "vehicle_turns_right": "turn right",
           "person_rides_bicycle": "ride a bicycle", "person_steals_object": "steal something", "person_purchases": "buy something"}
NEVER = ("person_rides_bicycle", "person_steals_object", "person_purchases")
LIKE = [{"person_opens_trunk", "person_closes_trunk"}, {"person_opens_vehicle_door", "person_closes_vehicle_door"},
        {"vehicle_reverses", "vehicle_makes_u_turn", "vehicle_turns_left", "vehicle_turns_right"}]


@dataclass(frozen=True)
class Act:
    start: float  # seconds after local midnight
    end: float
    status: str
    edge: bool  # within a second of its clip's start or end
    file: str
    o0: float  # seconds into its file
    o1: float


@functools.lru_cache(maxsize=2)
def plain_world(limit: int) -> tuple[dict, dict]:
    """`(acts, files)` of the three cameras' clips that begin before minute `limit` of the day, as read here from the label files and the
    clips' frame counts: each camera's acts by kind; and each camera's files, `(path, start, length)`."""
    clips: dict[str, list] = defaultdict(list)
    for path in sorted((SOURCE / LABELS).glob("kitware*/2018-03-12/*/*.activities.yml")):
        m = NAME.match(path.name)
        start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        if m[7] in CAMS.values() and start < limit * 60:
            hour = path.parent.name  # (a folder per hour the clip ends in)
            avi = SOURCE / "video" / "2018-03-12" / hour / path.name.replace(".activities.yml", ".r13.avi")
            with av.open(str(avi)) as c:
                frames = c.streams.video[0].frames
            acts = [(k, s, int(a), int(b)) for k, s, a, b in ACT.findall(path.read_text(encoding="utf-8"))]
            clips[m[7]].append((start, m[1] + m[2] + m[3], m[4] + m[5] + m[6], frames, acts))
    found: dict[tuple[str, str], list[Act]] = defaultdict(list)
    files: dict[str, list] = defaultdict(list)
    for cam, camera in CAMS.items():
        for late in (False, True):
            run = [c for c in sorted(clips[camera]) if (c[0] > GAP) == late]
            if not run:
                continue
            path, done = f"{cam}/{cam}_20180312_{run[0][1]}-{run[-1][2]}.mp4", 0
            for _, _, _, frames, acts in run:
                for kind, status, f0, f1 in acts:
                    found[cam, kind].append(Act(run[0][0] + (done + f0) / 30, run[0][0] + (done + f1) / 30, status,
                                                f0 < 30 or f1 >= frames - 30, path, (done + f0) / 30, (done + f1) / 30))
                done += frames
            files[cam].append((path, run[0][0], done / 30))
    return {k: sorted(v, key=lambda a: (a.start, a.end)) for k, v in found.items()}, dict(files)


def spans(acts) -> list[tuple[str, float, float]]:
    return sorted((a.file, round(a.o0, 2), round(a.o1, 2)) for a in acts)


def evidence(q) -> list[tuple[str, float, float]]:
    return sorted((s.file, round(s.t0, 2), round(s.t1, 2)) for s in q.truth.evidence)


def check_against_labels(questions, limit: int) -> None:
    """Each question's truth and evidence as the plain reader finds them."""
    acts, files = plain_world(limit)
    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template = "absent_" + parts[1] if absent else parts[0]
        rest = parts[2:] if absent else parts[1:]
        (cam, kind), window = rest[:2], rest[2:]
        mine = acts.get((cam, kind), [])
        assert q.tags["template"] == template and q.kind == ("absent" if absent else template) and q.tags["absent"] is absent, q.qid
        assert f" on {cam}" in q.question and PHRASES[kind] in q.question, f"{q.qid} asks of another camera or activity: {q.question}"
        assert q.scope.sources == (cam,) and q.now.startswith("2018-03-12T12:00:00"), q.qid
        if window:
            a, b = (int(w[:2]) * 3600 + int(w[2:]) * 60 for w in window)
            assert f" between {window[0][:2]}:{window[0][2:]} and {window[1][:2]}:{window[1][2:]}?" in q.question, q.qid
            assert q.scope.since == pytest.approx(MIDNIGHT + a) and q.scope.until == pytest.approx(MIDNIGHT + b), q.qid
        else:
            assert q.scope.since == pytest.approx(MIDNIGHT + files[cam][0][1]), q.qid
            assert q.scope.until == pytest.approx(MIDNIGHT + files[cam][-1][1] + files[cam][-1][2]), q.qid
        if absent:
            assert q.grade.rule == "absent" and not q.truth.evidence and q.truth.value is None, q.qid
            if template == "absent_count":  # the camera has the activity, but neither it nor its like near the window
                like = [x for k in next(g for g in LIKE if kind in g) for x in acts.get((cam, k), [])]
                assert mine and all(x.end < a - 60 or x.start > b + 60 for x in like), f"{q.qid}: {cam} has {kind} or its like near the window"
            else:  # no camera has it in these hours
                assert kind in NEVER and not any(k == kind for _, k in acts), f"{q.qid}: the labels have {kind}"
        elif template == "count":
            inside = [x for x in mine if a <= x.start and x.end <= b]
            assert all((a + 30 <= x.start and x.end <= b - 30) or x.end < a - 30 or x.start > b + 30 for x in mine), f"{q.qid}: an edge cuts one"
            assert q.grade.rule == "count" and q.grade.truth == q.truth.value == str(len(inside)) and 1 <= len(inside) <= 5, q.qid
            assert evidence(q) == spans(inside) and all(x.status == "good" and not x.edge for x in inside), q.qid
            assert all(y.start - x.end >= 10 for x, y in pairwise(inside)), f"{q.qid}: two run into each other"
        else:
            e = mine[0] if template == "first" else mine[-1]
            middle = round((e.start + e.end) / 2)
            clock = f"{middle // 3600:02d}:{middle % 3600 // 60:02d}:{middle % 60:02d}"
            near = [x for x in mine if x.end >= middle - 30 and x.start <= middle + 30]
            assert (q.grade.rule, q.grade.date, q.grade.tol_s) == ("time", "2018-03-12", 30.0), q.qid
            assert abs(grade.at(q.grade.time) - grade.at(clock)) <= 1 / 60 and q.truth.value == f"2018-03-12 {q.grade.time}", q.qid
            assert evidence(q) == spans(near), q.qid
            assert all(x.status == "good" and not x.edge and x.end - x.start <= 30 for x in near), f"{q.qid} rests on a label that is not sure"


def right(q) -> str:
    g = q.grade
    return {"count": lambda: g.truth, "time": lambda: f"{g.date} {g.time}", "absent": lambda: "unknown"}[g.rule]()


def right_too(q) -> list[str]:
    """Other natural answers in the shape asked: the date in words, a time 20 s out; a count with its unit; an absent one said otherwise."""
    g = q.grade
    if g.rule == "count":
        return [f"{g.truth} times"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date}T{g.time}")
        return [f"{t:%d %B %Y}, {t:%H:%M:%S}", f"{t + timedelta(seconds=20):%Y-%m-%d %H:%M:%S}", f"{t - timedelta(seconds=20):%Y-%m-%d %H:%M:%S}"]
    return ["none", "No, never.", "0" if q.tags["template"] == "absent_count" else "nothing at all"]


def wrong(q) -> list[str]:
    """Planted wrong answers: a time 45 s, two minutes or a day out; one more or fewer; `unknown` on a needle; a time or a number for an
    absent one."""
    g = q.grade
    if g.rule == "count":
        return [str(int(g.truth) + 1), str(int(g.truth) - 1), "unknown", f"{g.truth} or {int(g.truth) + 7}"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date}T{g.time}")
        return [f"{t + timedelta(seconds=45):%Y-%m-%d %H:%M:%S}", f"{t - timedelta(seconds=45):%Y-%m-%d %H:%M:%S}",
                f"{t + timedelta(minutes=2):%Y-%m-%d %H:%M}", f"{t + timedelta(days=1):%Y-%m-%d %H:%M:%S}", "unknown"]
    return ["2018-03-12 10:42:11", "2018-03-12", "3", "once"]


def boxes(path: Path) -> list[tuple[str, int, int]]:
    """An MP4's top-level boxes, `(type, start, size)`, read here by hopping from one to the next."""
    out, at, size = [], 0, path.stat().st_size
    with open(path, "rb") as f:
        while at < size:
            f.seek(at)
            head = f.read(8)
            out.append((head[4:].decode("latin-1"), at, int.from_bytes(head[:4], "big")))
            at += out[-1][2]
    return out


@pytest.fixture(scope="module")
def tiny():
    return V1.plan_of(V1.recipe(scale="tiny", seed=1))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("v1")
    recipe = V1.recipe(scale="tiny", seed=1)
    dump, questions = V1.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_dump_is_ten_minutes_of_three_cameras_with_a_quarter_of_its_questions_absent(tiny):
    files, questions = tiny
    _, plain = plain_world(LIMITS["tiny"])
    assert [f.path for f in files] == ["cam1/cam1_20180312_100002-101002.mp4", "cam2/cam2_20180312_100000-101000.mp4",
                                       "cam3/cam3_20180312_100000-101000.mp4"] == [p for cam in plain for p, _, _ in plain[cam]]
    assert [f.source for f in files] == ["cam1", "cam2", "cam3"] and all(f.length == 600.0 for f in files)
    assert [datetime.fromtimestamp(f.t) for f in files] == [datetime(2018, 3, 12, 10, 0, 2), datetime(2018, 3, 12, 10), datetime(2018, 3, 12, 10)]
    assert [len(f.spec["clips"]) for f in files] == [2, 2, 2] and files[0].spec["clips"][0].endswith("school.G336.r13.avi")
    assert len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}


@pytest.mark.parametrize("scale", ["tiny", "standard"])
def test_every_answer_is_the_labels(scale):
    files, questions = V1.plan_of(V1.recipe(scale=scale, seed=1))
    acts, plain = plain_world(LIMITS[scale])
    assert [f.path for f in files] == [p for cam in plain for p, _, _ in plain[cam]]
    assert [f.length for f in files] == [length for cam in plain for _, _, length in plain[cam]]
    if scale == "standard":
        assert len(files) == 6 and len(questions) == 24 and sum(q.kind == "absent" for q in questions) == 6
        assert {a.status for v in acts.values() for a in v} == {"good", "unaudited", "not_good"}, "labels that are not sure are to be kept out"
    check_against_labels(questions, LIMITS[scale])
    length = {f.path: f.length for f in files}
    assert all(0 <= s.t0 <= s.t1 <= length[s.file] for q in questions for s in q.truth.evidence), "every span is inside its file"
    assert len({q.qid for q in questions}) == len(questions)


def test_another_seed_asks_other_questions_of_the_same_recordings():
    files, one = V1.plan_of(V1.recipe(scale="standard", seed=1))
    files_two, two = V1.plan_of(V1.recipe(scale="standard", seed=2))
    assert files == files_two and {q.qid for q in one} != {q.qid for q in two}
    check_against_labels(two, LIMITS["standard"])


def test_no_question_rests_on_a_label_that_is_not_sure_or_a_window_that_cuts_an_activity():
    """A made-up recording of 10:00 to 10:30: a first activity that is not good, one at the edge of its clip, one with an unaudited label
    beside it, and windows that cut an activity are never asked; the others are."""
    base = realdump.local_t(datetime(2018, 3, 12, 10))

    def act(kind, at, status="good", edge=False, lasts=2.0):
        return Event("cam1", "cam1/a.mp4", kind, status, at, at + lasts, base + at, base + at + lasts, edge)

    events = [act("person_opens_trunk", 300), act("person_opens_trunk", 720), act("person_opens_trunk", 1315),
              act("person_closes_trunk", 100, edge=True), act("person_closes_trunk", 1200),
              act("vehicle_reverses", 200), act("vehicle_reverses", 215, status="unaudited"),
              act("vehicle_makes_u_turn", 400, status="not_good"), act("vehicle_makes_u_turn", 900)]
    files = [realdump.File("cam1/a.mp4", "cam1", base, "recording", "video", {}, length=1800.0)]
    span = (datetime(2018, 3, 12, 10), datetime(2018, 3, 12, 10, 30))
    qids = set()
    for seed in range(5):
        qids |= {q.qid for q in questions_of(events, files, span, "standard", seed, realdump.local_t)}
    asked = {"first-cam1-person_opens_trunk", "last-cam1-person_opens_trunk", "last-cam1-person_closes_trunk", "last-cam1-vehicle_makes_u_turn"}
    assert asked <= qids
    assert not {"first-cam1-person_closes_trunk", "first-cam1-vehicle_makes_u_turn"} & qids, "an edge label and a not_good one decide them"
    assert not {q for q in qids if "vehicle_reverses" in q and not q.startswith("absent")}, "the unaudited label is within 30 s of the other"
    counts = [q for q in qids if q.startswith("count-cam1-person_opens_trunk-")]
    assert counts, "a window that keeps clear of them all is asked"
    for qid in counts:
        a, b = (int(w[:2]) * 3600 + int(w[2:]) * 60 - 36000 for w in qid.split("-")[-2:])
        assert all((a + 30 <= e.o0 and e.o1 <= b - 30) or e.o1 < a - 30 or e.o0 > b + 30 for e in events if e.kind == "person_opens_trunk"), qid


def test_a_label_file_is_read_as_kpf_and_a_clip_with_none_or_two_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "a.activities.yml"
    act = ("- {'act': {'act2': {'person_opens_trunk': 1.0}, 'actors': [{'id1': 1, 'timespan': [{'tsr0': [10, 70]}]}], 'id2': 3, "
           "'src_status': 'good', 'timespan': [{'tsr0': [10, 70]}]}}\n")
    path.write_text(act, encoding="utf-8")
    assert read_acts(path, 9000) == (v1_source.Act("person_opens_trunk", "good", 10, 70),)
    path.write_text("- {'meta': 'x was marked empty_37'}\n", encoding="utf-8")
    assert read_acts(path, 9000) == ()
    empty = "- {'meta': 'x was marked empty_37'}\n"
    for text, why in (("", "does not say"), (act + empty, "says the clip is empty"), ("- {'act': \n", "not a KPF"),
                      (act.replace("[10, 70]", "[10, 9000]"), "not a KPF"), ("hello\n", "not a KPF")):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            read_acts(path, 9000)
    root = tmp_path / "meva"
    name = "2018-03-12.10-00-00.10-05-00.hospital.G341"
    (root / "video" / "2018-03-12" / "10").mkdir(parents=True)
    (root / "video" / "2018-03-12" / "10" / f"{name}.r13.avi").write_bytes(b"")
    monkeypatch.setattr(v1_source, "frames_of", lambda p: 9000)
    with pytest.raises(ValueError, match="no label file"):
        v1_source.read_clips(str(root))
    for s in v1_source.SETS:
        (root / LABELS / s / "2018-03-12" / "10").mkdir(parents=True)
        (root / LABELS / s / "2018-03-12" / "10" / f"{name}.activities.yml").write_text(act, encoding="utf-8")
    with pytest.raises(ValueError, match="both sets"):
        v1_source.read_clips(str(root))
    (root / LABELS / v1_source.SETS[1] / "2018-03-12" / "10" / f"{name}.activities.yml").unlink()
    (clip,) = v1_source.read_clips(str(root) + "/")
    assert (clip.camera, clip.frames, clip.acts[0].kind) == ("cam2", 9000, "person_opens_trunk")


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    questions = V1.plan_of(V1.recipe(scale="standard", seed=1))[1] + V1.plan_of(V1.recipe(scale="standard", seed=2))[1] + tiny[1]
    for q in questions:
        assert grade.grade(q.grade, right(q)), (q.qid, right(q))
        for answer in right_too(q):
            assert grade.grade(q.grade, answer), (q.qid, answer)
        for answer in wrong(q):
            assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_recording_is_the_clips_joined_with_its_time_its_length_and_no_name_of_the_source(built):
    out, recipe, dump, _ = built
    _, plain = plain_world(LIMITS["tiny"])
    assert [e.path for e in dump.entries] == [p for cam in plain for p, _, _ in plain[cam]]
    for e, cam in zip(dump.entries, plain, strict=True):
        (_, start, length), file = plain[cam][0], out / "dump" / e.path
        assert datetime.fromtimestamp(os.stat(file).st_mtime) == datetime.fromtimestamp(e.t) == datetime(2018, 3, 12) + timedelta(seconds=start)
        assert (e.source, e.kind, e.media, e.length, e.own_time) == (cam, "recording", "video", length, False)
        with av.open(str(file)) as c:
            video = c.streams.video[0]
            packets = [p for p in c.demux(video) if p.size]
            assert [s.type for s in c.streams] == ["video"] and (video.codec_context.name, video.average_rate, video.width) == ("h264", 30, 1920)
            assert len(packets) == 18000 and set(np.diff([p.pts for p in packets])) == {512} and abs(float(c.duration) / 1e6 - length) < 0.01
            assert set(c.metadata) == {"major_brand", "minor_version", "compatible_brands"} and set(video.metadata) == {"language", "handler_name"}
        clips = sorted((SOURCE / "video" / "2018-03-12" / "10").glob(f"*.{CAMS[cam]}.r13.avi"))[:2]
        for n, clip in enumerate(clips):  # each clip's first frame is the file's at its offset, bit for bit
            with av.open(str(clip)) as c:
                first = next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24")
            with av.open(str(file)) as c:
                video = c.streams.video[0]
                c.seek(int(300 * n / video.time_base), stream=video, backward=True)
                frame = next(f for f in c.decode(video) if f.pts * video.time_base >= 300 * n)
                assert float(frame.pts * video.time_base) == 300 * n and np.array_equal(first, frame.to_ndarray(format="rgb24")), (e.path, n)
        # what the file says of itself (all but its video): no title, source path, site, camera or encoder
        head = [b for b in boxes(file) if b[0] != "mdat"]
        assert [b[0] for b in head] == ["ftyp", "free", "moov"]
        with open(file, "rb") as f:
            said = b"".join(f.read(size) for _, at, size in head if f.seek(at) >= 0)
        for word in (b"Lavf", b"Lavc", b"x264", b"title", b"comment", b".avi", b"meva", b"kitware", b"school", b"hospital", b"G336", b"G341",
                     b"G436"):
            assert word not in said, (e.path, word)
    assert V1.probe(recipe, dump, out, None, 1).status == "passed"


def test_the_probe_fails_a_recording_whose_name_or_metadata_names_an_activity(built, tmp_path):
    _, recipe, _, _ = built

    def box(kind: bytes, payload: bytes = b"") -> bytes:
        return (8 + len(payload)).to_bytes(4, "big") + kind + payload

    named = tmp_path / "dump"
    named.mkdir()
    (named / "cam1_person_opens_trunk.mp4").write_bytes(box(b"free"))
    (named / "cam2.mp4").write_bytes(box(b"free", b"\0person opens trunk\0"))  # in what the file says of itself
    (named / "cam3.mp4").write_bytes(box(b"free") + box(b"mdat", b"person opens trunk"))  # in its video: not read
    entries = tuple(ManifestEntry(p.name, "cam", 0.0, "recording", "video", None, False, 0, 0, "", 1.0) for p in sorted(named.iterdir()))
    result = V1.probe(recipe, realdump.dump_of("k", named, "v1_security", entries), tmp_path, None, 1)
    assert result.status == "failed" and len(result.failures) == 2
    assert "its name says" in result.failures[0] and "its metadata says" in result.failures[1]


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = V1.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        V1.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        V1.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        V1.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = V1.deny_words([], None, "k")
    assert "bench-data" in deny and "in:meva" in deny and "in:kitware" in deny and "in:v1_security" in deny and "in:from mmmb" in deny


class Spy:
    """An agent that answers right in the `files` configuration, citing the first span of the truth, and `unknown` in any other."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": s.t0, "t1": s.t1} for s in q.truth.evidence[:1]]
        answer = right(q) if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


def test_build_run_and_report_over_the_tiny_dump(world, capsys):  # noqa: F811
    world.patch.setattr(v1_source, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: REAL_FFMPEG))  # (the world's is made up)
    assert world.main("build", "--scenario", "v1_security", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    first = next(q for q in run.questions() if q.qid.startswith("first-"))
    prompt = spy.prompts["files", first.qid]
    assert prompt.startswith("It is now Monday 12 March 2018, 12:00. The files of the recordings of my three outdoor security cameras are in ")
    assert first.question in prompt and "<the date and time, as YYYY-MM-DD HH:MM:SS>" in prompt
    assert "school" not in prompt and "hospital" not in prompt and "meva" not in prompt.casefold()
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMMB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (8, 8, 8)
    assert (listing["correct"], listing["n"]) == (2, 8) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 8}
