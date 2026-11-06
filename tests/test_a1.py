"""A1, the home recorder, from its real source on disk: the tiny hour built (one WAV of the minute files where their names put them, with
its time), every answer derived again here by a plain reader of the label files (never the scenario's), the grader taking each truth and
refusing planted wrong answers, the leak probe, and `mmmb build`, `run` and `report` over it with a stand-in agent. Skips, and says so,
without the source."""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import wave
from collections import defaultdict
from datetime import datetime, timedelta

import av
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
from mmmb.scenarios.a1_home_audio.source import SOURCE, write_wav

if not (SOURCE / "labels" / "living_labels.csv").is_file():
    pytest.skip(f"the SINS source is not at {SOURCE}, so A1 is not built", allow_module_level=True)

A1 = scenarios.get("a1_home_audio")
pytestmark = pytest.mark.xdist_group("a1")
MIDNIGHT = datetime(2017, 2, 2)
RATE = 16000
PRESENT = ["vacuum_on-0001", "vacuum_on-2204", "vacuum_off-0006", "vacuum_off-2212", "tv_on-0229", "tv_on-1843", "tv_off-0319",
           "tv_off-2156", "visit_on-1459", "visit_off-1508", "leave-0829", "return-1730", "count-vacuum", "count-tv", "count-visit"]
# template -> (label file, class, which edge), and what its question asks
TEMPLATES = {"vacuum_on": ("living_labels.csv", "vacuumcleaner", "start", "was the vacuum cleaner switched on"),
             "vacuum_off": ("living_labels.csv", "vacuumcleaner", "stop", "was the vacuum cleaner switched off"),
             "tv_on": ("living_labels.csv", "watching tv", "start", "did I start watching TV"),
             "tv_off": ("living_labels.csv", "watching tv", "stop", "did I stop watching TV"),
             "visit_on": ("living_labels.csv", "visit", "start", "did a visitor arrive"),
             "visit_off": ("living_labels.csv", "visit", "stop", "did a visitor leave"),
             "leave": ("labels.csv", "absence", "start", "did I leave the house"),
             "return": ("labels.csv", "absence", "stop", "did I get back home")}
COUNTS = {"count_vacuum": "vacuumcleaner", "count_tv": "watching tv", "count_visit": "visit"}
TOL = {"tv_on": 120.0, "tv_off": 120.0}  # an answer this many seconds out is right: the TV's, whose label marks the watching, not the set


def seconds_of(text: str) -> float:
    return (datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f") - MIDNIGHT).total_seconds()


def plain_labels(name: str) -> list[tuple[str, float, float]]:
    """`(class, start, stop)` of each label in the file, in seconds after the midnight that starts 2 February 2017: read here, as a second
    reader."""
    with open(SOURCE / "labels" / name, encoding="utf-8") as f:
        rows = [line.split(";") for line in f.read().splitlines()[1:]]
    return [(c, seconds_of(a), seconds_of(b)) for c, a, b in rows]


LABELS = {name: plain_labels(name) for name in ("living_labels.csv", "labels.csv")}


def spans(q) -> list[float]:
    return [x for s in q.truth.evidence for x in (s.t0, s.t1)]


def hms(n: float) -> str:
    n = round(n)
    return f"{n // 3600:02d}:{n % 3600 // 60:02d}:{n % 60:02d}"


def check_against_labels(questions, first: int, seconds: int) -> None:
    """Each question's truth, evidence and window as the plain reader finds them, for the recording of `seconds` from `first`."""
    last = first + seconds
    path = f"RECORD/REC_20170202_{first // 3600:02d}{first % 3600 // 60:02d}00.wav"
    for q in questions:
        key, absent = q.tags["template"], q.kind == "absent"
        assert (q.tags["modality"], q.tags["absent"], q.scope.sources) == ("audio", absent, ("recorder",)) and not q.truth.wrong
        if key in COUNTS:
            found = [(a, b) for c, a, b in LABELS["living_labels.csv"] if c == COUNTS[key] and first <= a and b <= last]
            assert found and (q.kind, q.qid, q.grade.rule, q.grade.truth) == ("count", "count-" + key[6:], "count", str(len(found))), q.qid
            assert spans(q) == pytest.approx([x - first for a, b in found for x in (a, b)]), q.qid
            assert {s.file for s in q.truth.evidence} == {path}
            continue
        which, cls, edge, phrase = TEMPLATES[key.removeprefix("absent_")]
        runs = [(a, b) for c, a, b in LABELS[which] if c == cls]
        m = re.fullmatch(r"At what time yesterday, between (\d\d):(\d\d) and (\d\d):(\d\d), (.+)\?", q.question)
        assert m and m[5] == phrase, f"{q.qid} asks of another thing: {q.question}"
        lo, hi = int(m[1]) * 3600 + int(m[2]) * 60, int(m[3]) * 3600 + int(m[4]) * 60
        assert first <= lo < hi <= last and hi < 86400 and lo % 300 == hi % 300 == 0, q.qid
        assert (900 if seconds == 3600 else 3600) <= hi - lo <= (2400 if seconds == 3600 else 14400), f"{q.qid}'s window is the wrong length"
        assert set(q.grade.ignore) == {f"{lo // 3600:02d}:{lo % 3600 // 60:02d}", f"{hi // 3600:02d}:{hi % 3600 // 60:02d}"}, q.qid
        if absent:
            assert (q.kind, q.grade.rule, q.truth.value, q.truth.evidence) == ("absent", "absent", None, ()), q.qid
            assert q.qid == f"absent-{key[7:]}-{lo // 3600:02d}{lo % 3600 // 60:02d}", q.qid
            steps = [(a, b) for c, a, b in LABELS["labels.csv"] if c == "absence" and b - a < 1800]
            assert all(b <= lo - 600 or a >= hi + 600 for a, b in (*runs, *steps)), f"{q.qid}: a {cls} run or a step out comes near its window"
            continue
        moments = [(a, b) for a, b in runs if hms(a if edge == "start" else b) == q.grade.time]
        assert len(moments) == 1, f"{q.qid}: {len(moments)} runs of {cls} have {q.grade.time} as their {edge}"
        a, b = moments[0]
        at = a if edge == "start" else b
        tol = TOL.get(key, 60.0)
        assert (q.kind, q.grade.rule, q.grade.tol_s, q.grade.date, q.truth.value) == (edge, "time", tol, None, q.grade.time), q.qid
        assert q.qid == f"{key}-{int(at) // 3600:02d}{int(at) % 3600 // 60:02d}", q.qid
        assert 0 <= a and b <= 86400 and b - a >= 120, f"{q.qid} rests on a run that crosses the day's edge, or a blip"
        assert lo + tol <= at <= hi - tol, f"{q.qid}: its window does not hold the tolerance of the moment"
        assert all(y <= lo - 600 or x >= hi + 600 for x, y in runs if (x, y) != (a, b)), f"{q.qid}: another {cls} run is near its window"
        assert all(abs(x - at) > 600 and abs(y - at) > 600 for x, y in runs if (x, y) != (a, b)), f"{q.qid}: another {cls} run is near the moment"
        assert spans(q) == pytest.approx([max(0.0, at - tol - first), min(float(seconds), at + tol - first)]), "the tolerance either side of the moment"
        assert {s.file for s in q.truth.evidence} == {path}
    if seconds == 86400:  # the whole day: every moment that is alone in its stretch is asked
        assert sorted(q.qid for q in questions if q.kind != "absent") == sorted(PRESENT)


def right(q) -> str:
    g = q.grade
    return {"count": lambda: g.truth, "time": lambda: g.time, "absent": lambda: "unknown"}[g.rule]()


def wrong(q) -> list[str]:
    """Planted wrong answers: one more or fewer; a time 10 s past the tolerance either side, three minutes either side, or that and the truth
    together; `unknown` on a needle; a time for an absent one."""
    g = q.grade
    if g.rule == "count":
        return [str(int(g.truth) + 1), str(int(g.truth) - 1), "unknown"]
    if g.rule == "time":
        t, out = datetime.strptime(g.time, "%H:%M:%S"), timedelta(seconds=g.tol_s + 10)
        late, early = (t + timedelta(minutes=3)).strftime("%H:%M"), (t - timedelta(minutes=3)).strftime("%H:%M")
        return [late, early, f"{g.time[:5]} or {late}", "unknown", (t + out).strftime("%H:%M:%S"), (t - out).strftime("%H:%M:%S")]
    return ["04:12", "yes, at 14:20"]


def natural(q) -> list[str]:
    """Other ways to answer right: a time to the second, to the minute, in the 12-hour clock, in a sentence, and 20 s inside the tolerance
    either side; an abstention with the question's own times."""
    g = q.grade
    if g.rule == "count":
        return [f"{g.truth}", "twice" if g.truth == "2" else "once" if g.truth == "1" else g.truth]
    if g.rule == "time":
        t, inside = datetime.strptime(g.time, "%H:%M:%S"), timedelta(seconds=g.tol_s - 20)
        return [g.time, (t + timedelta(seconds=30)).strftime("%H:%M"), t.strftime("%I:%M %p"), f"It was at {g.time}, I think.",
                (t + inside).strftime("%H:%M:%S"), (t - inside).strftime("%H:%M:%S")]
    low, high = sorted(g.ignore)
    return ["no", f"Nothing between {low} and {high}.", f"None. I listened from {low} to {high} and heard nothing of the kind."]


def plain_wav(first: datetime, seconds: int) -> np.ndarray:
    """The recording of `seconds` from `first` as a plain reader makes it: each minute file's first channel painted where its name says,
    into the places no earlier file has, and a place no file reaches holding the sample before it."""
    total = seconds * RATE
    out, painted = np.zeros(total, dtype=np.int16), np.zeros(total, dtype=bool)
    for path in sorted(glob.glob(str(SOURCE / "node07" / "*" / "Node7_*_audio.flac"))):
        m = re.search(r"Node7_(\d{8})_(\d{6})_(\d{3})_audio", path)
        begins = round((datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S") + timedelta(milliseconds=int(m[3])) - first).total_seconds() * RATE)
        if not -61 * RATE < begins < total:
            continue
        with av.open(path) as c:
            y = np.concatenate([f.to_ndarray().reshape(-1, 4)[:, 0] for f in c.decode(audio=0)])
        lo, hi = max(begins, 0), min(begins + len(y), total)
        free = ~painted[lo:hi]
        out[lo:hi][free] = y[lo - begins:hi - begins][free]
        painted[lo:hi] = True
    reach = np.maximum.accumulate(np.where(painted, np.arange(total), -1))  # the last painted place at or before each
    reach[reach < 0] = np.argmax(painted)
    return out[reach]


@pytest.fixture(scope="module")
def tiny():
    return A1.plan_of(A1.recipe(scale="tiny", seed=1))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("a1")
    recipe = A1.recipe(scale="tiny", seed=1)
    dump, questions = A1.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_dump_is_an_hour_of_the_evening_with_a_third_of_its_questions_absent(tiny):
    files, questions = tiny
    (wav,) = files
    assert (wav.path, wav.media, wav.length, wav.t) == ("RECORD/REC_20170202_213000.wav", "audio", 3600.0, datetime(2017, 2, 2, 21, 30).timestamp())
    assert len(questions) == 6 and sum(q.kind == "absent" for q in questions) == 2
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}
    assert all(q.now.startswith("2017-02-03T09:00:00") for q in questions), "asked the morning after"
    assert {q.qid for q in questions if q.kind != "absent"} == {"vacuum_on-2204", "vacuum_off-2212", "tv_off-2156", "count-vacuum"}


@pytest.mark.parametrize("scale, first, seconds", [("tiny", 21 * 3600 + 1800, 3600), ("standard", 0, 86400)])
def test_every_answer_is_the_labels(scale, first, seconds):
    files, questions = A1.plan_of(A1.recipe(scale=scale, seed=1))
    if scale == "standard":
        assert files[0].path == "RECORD/REC_20170202_000000.wav" and files[0].length == 86400.0
        assert len(questions) == 20 and sum(q.kind == "absent" for q in questions) == 5 and sum(q.split == "dev" for q in questions) == 6
    check_against_labels(questions, first, seconds)


def test_another_seed_asks_other_questions_of_the_same_recording():
    for seed in (2, 3, 4):
        files_one, one = A1.plan_of(A1.recipe(scale="standard", seed=1))
        files, two = A1.plan_of(A1.recipe(scale="standard", seed=seed))
        assert files == files_one and {q.question for q in one} != {q.question for q in two}
        check_against_labels(two, 0, 86400)
        assert len(two) == 20 and sum(q.kind == "absent" for q in two) == 5
        check_against_labels(A1.plan_of(A1.recipe(scale="tiny", seed=seed))[1], 21 * 3600 + 1800, 3600)


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    for q in A1.plan_of(A1.recipe(scale="standard", seed=1))[1] + tiny[1]:
        for answer in [right(q), *natural(q)]:
            assert grade.grade(q.grade, answer), (q.qid, answer)
        for answer in wrong(q):
            assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_an_absent_question_reads_as_a_present_one_of_its_template():
    shape = re.compile(r"At what time yesterday, between \d\d:\d\d and \d\d:\d\d, .+\?")
    for seed in (1, 2, 3):
        questions = A1.plan_of(A1.recipe(scale="standard", seed=seed))[1]
        timed = [q for q in questions if q.kind != "count"]
        assert all(shape.fullmatch(q.question) and q.tags["shape"] == "<the time, as HH:MM:SS>" for q in timed)
        present = {q.tags["template"] for q in timed if q.kind != "absent"}
        absent = [q.tags["template"] for q in timed if q.kind == "absent"]
        assert len(set(absent)) == len(absent) and {t.removeprefix("absent_") for t in absent} <= present, "each of a template the dump asks"


def test_the_recording_is_the_minute_files_where_their_names_put_them_with_the_time_of_its_start(built):
    out, recipe, dump, _ = built
    (entry,) = dump.entries
    path = out / "dump" / entry.path
    first = datetime(2017, 2, 2, 21, 30)
    assert entry.path == "RECORD/REC_20170202_213000.wav" and entry.length == 3600.0 and entry.media == "audio" and not entry.own_time
    assert datetime.fromtimestamp(os.stat(path).st_mtime) == first == datetime.fromtimestamp(entry.t)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, RATE, 3600 * RATE)
    assert entry.size == 44 + 3600 * RATE * 2, "a header and the samples, nothing else"
    samples = np.memmap(path, dtype="<i2", mode="r", offset=44)
    for at in (0, 2060, 3300):  # the start, the vacuum cleaner's start, near the end
        want = plain_wav(first + timedelta(seconds=at), 200)
        assert np.array_equal(samples[at * RATE:(at + 200) * RATE], want), f"the recording differs from the minute files at {at} s"
    level = {s: float(np.std(samples[s * RATE:(s + 10) * RATE].astype(float))) for s in (2020, 2100, 2500, 2590)}
    assert level[2100] > 10 * level[2020] and level[2500] > 10 * level[2590], "the vacuum cleaner is on from second 2083 to 2563"
    assert A1.probe(recipe, dump, out, None, 1).status == "passed"


def test_where_minute_files_overlap_or_leave_a_gap_the_line_is_trimmed_or_held(tmp_path):
    """The real minutes 09:02 to 09:07, which hold an overlap of 1.9 s and gaps of 1 s as well as the usual tenth of a second."""
    first = datetime(2017, 2, 2, 9, 2)
    write_wav(str(SOURCE), first, 300, tmp_path / "part.wav")
    want = plain_wav(first, 300)
    got = np.memmap(tmp_path / "part.wav", dtype="<i2", mode="r", offset=44)
    assert len(got) == len(want) and np.array_equal(got, want)
    names = sorted(glob.glob(str(SOURCE / "node07" / "2017-02-02" / "Node7_20170202_090[2-6]*")))
    starts = [datetime.strptime(re.search(r"_(\d{8})_(\d{6})_(\d{3})_", n).expand(r"\1\2\3"), "%Y%m%d%H%M%S%f") for n in names]
    gaps = [(b - a).total_seconds() - 59.9 for a, b in zip(starts, starts[1:])]
    assert min(gaps) < -1.5 and max(gaps) > 0.9, "the stretch is one with an overlap and a gap"


def test_a_tv_answer_a_minute_and_a_bit_out_is_right_where_the_vacuum_cleaners_is_not():
    """The TV's label marks the watching, not the set: its sound ended at 21:57:42, 46 s after the label's 21:56:56, and "21:58" is 64 s
    out. The tolerance is 120 s for the TV and 60 s for the rest."""
    questions = {q.qid: q for q in A1.plan_of(A1.recipe(scale="standard", seed=1))[1]}
    tv, vacuum = questions["tv_off-2156"], questions["vacuum_off-2212"]
    assert (tv.grade.time, tv.grade.tol_s, vacuum.grade.time, vacuum.grade.tol_s) == ("21:56:56", 120.0, "22:12:43", 60.0)
    assert [grade.grade(tv.grade, a) for a in ("21:58", "21:57:42", "21:58:50", "21:55", "21:54:57")] == [True] * 5
    assert [grade.grade(tv.grade, a) for a in ("21:59", "21:59:10", "21:54", "22:00")] == [False] * 4
    assert [grade.grade(vacuum.grade, a) for a in ("22:13", "22:13:30", "22:12")] == [True] * 3
    assert [grade.grade(vacuum.grade, a) for a in ("22:14", "22:13:50", "22:11")] == [False] * 3
    assert all(q.grade.tol_s == (120.0 if q.tags["template"] in TOL else 60.0) for q in questions.values() if q.grade.rule == "time")
    assert all(q.tags["shape"] == "<the time, as HH:MM:SS>" for q in questions.values() if q.grade.rule != "count")


def level(first: float, seconds: int) -> float:
    """The level in dB of `seconds` of the recording from `first` (seconds after midnight), as the plain reader makes it: the spread of the
    samples, whatever their offset. The room's quiet is about 31 dB, the TV's sound 45 to 55."""
    x = plain_wav(MIDNIGHT + timedelta(seconds=first), seconds).astype(float)
    return float(20 * np.log10(x.std() + 1))


@pytest.mark.parametrize("key", ["tv_on", "tv_off"])
def test_the_tvs_sound_starts_and_stops_inside_the_tolerance_either_side_of_its_label(key):
    """The room is quiet just outside the tolerance on one side of the label of a TV moment, and the TV is on just outside it on the other:
    the sound's edge (30 s and 55 s before the starts, 15 s and 45 s after the stops) is inside the tolerance, so that an answer that heard it
    right is right."""
    questions = [q for q in A1.plan_of(A1.recipe(scale="standard", seed=1))[1] if q.tags["template"] == key]
    assert len(questions) == 2
    for q in questions:
        at = (datetime.strptime(q.grade.time, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds()
        before, after = (level(at + off, 10) for off in (-q.grade.tol_s - 10, q.grade.tol_s))
        quiet, on = (before, after) if key == "tv_on" else (after, before)
        assert quiet < 35 and on > 38, f"{q.qid}: {quiet:.0f} dB where it is quiet and {on:.0f} dB where the TV is on"


def test_the_probe_fails_a_recording_whose_name_or_metadata_names_a_label_or_a_plain_word_for_one(built, tmp_path):
    _, recipe, _, _ = built

    def riff(*chunks: tuple[bytes, bytes]) -> bytes:
        body = b"WAVE" + b"".join(kind + len(data).to_bytes(4, "little") + data + b"\x00" * (len(data) & 1) for kind, data in chunks)
        return b"RIFF" + len(body).to_bytes(4, "little") + body

    words = ["vacuumcleaner", "vacuum", "cleaner", "watching", "tv", "television", "visit", "visitor", "absence", "away", "node07"]
    named = tmp_path / "dump"
    named.mkdir()
    (named / "REC_20170202_213000.wav").write_bytes(riff((b"fmt ", bytes(16)), (b"data", b"the vacuum cleaner and a visitor in front of the tv")))
    for n, word in enumerate(words):
        (named / f"REC_{word}.wav").write_bytes(riff((b"fmt ", bytes(16)), (b"data", bytes(8))))
        (named / f"REC_{n:02d}.wav").write_bytes(riff((b"fmt ", bytes(16)), (b"LIST", f"a note of {word} here".encode()), (b"data", bytes(8))))
    entries = tuple(ManifestEntry(p.name, "recorder", 0.0, "recording", "audio", None, False, 0, 0, "", 1.0) for p in sorted(named.iterdir()))
    result = A1.probe(recipe, realdump.dump_of("k", named, "a1_home_audio", entries), tmp_path, None, 1)
    assert result.status == "failed" and len(result.failures) == 2 * len(words), result.failures
    assert not any(f.startswith("REC_20170202_213000") for f in result.failures), "the samples are not read"
    assert sum("its name says" in f for f in result.failures) == len(words) and sum("its metadata says" in f for f in result.failures) == len(words)


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = A1.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        A1.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        A1.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        A1.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = A1.deny_words([], None, "k")
    assert "bench-data" in deny and "in:/sins" in deny and "in:a1_home_audio" in deny and "in:from mmmb" in deny


class Spy:
    """An agent that answers right in the `files` configuration, citing the first moment of the truth, and `unknown` in any other."""

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


def test_build_run_and_report_over_the_tiny_hour(world, capsys):  # noqa: F811
    assert world.main("build", "--scenario", "a1_home_audio", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    asked = next(q for q in run.questions() if q.qid == "vacuum_on-2204")
    prompt = spy.prompts["files", asked.qid]
    assert prompt.startswith("It is now Friday 3 February 2017, 09:00. The files of the voice recorder I left running in my living room are in ")
    assert asked.question in prompt and "<the time, as HH:MM:SS>" in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMMB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (6, 6, 6)
    assert (listing["correct"], listing["n"]) == (2, 6) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 6}
