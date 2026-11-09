"""A4, the field recorder: the offline parts (the FLAC's header read, the cut of a night into hourly files sample for sample, and the questions
of a made-up night, every answer derived again here by a plain reader of its calls), and, where the real source is on disk, the same over
BirdVox-full-night's unit 3: the tiny night built (two hourly files cut sample for sample, with their times), every answer of the standard
night derived again here from the annotation CSVs (never the scenario's reader), the grader taking each truth and refusing planted wrong
answers, the leak probe, and `mmsb build`, `run` and `report` over it with a stand-in agent. The real-source tests skip, and say so, without
the source."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import threading
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import a4_field as A4
from mmsb.scenarios import realdump
from mmsb.scenarios.a4_field import questions as Q
from mmsb.scenarios.a4_field import source as a4_source
from mmsb.scenarios.a4_field.source import AUDIO, SOURCE, Call, Night, cut, flac_header, streaminfo

REAL = pytest.mark.skipif(not (SOURCE / AUDIO).is_file(), reason=f"BirdVox-full-night is not at {SOURCE}, so A4 is not built")
GROUP = pytest.mark.xdist_group("a4")
START = datetime(2015, 9, 24, 3, 31, 2)
REAL_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # (the `world` fixture puts a made-up file in its place)
WINDOW = re.compile(r"(\d{2})(\d{2})-(\d{2})(\d{2})")


def sine(path: Path, seconds: int, rate: int = 24000) -> Path:
    """A FLAC of a sine tone of `seconds`, mono 16-bit, made by ffmpeg."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate={rate}:duration={seconds}", "-sample_fmt", "s16", "-c:a", "flac", str(path)], check=True)
    return path


def samples(path: Path) -> np.ndarray:
    parts = []
    with av.open(str(path)) as c:
        for f in c.decode(audio=0):
            parts.append(f.to_ndarray().reshape(-1))
    return np.concatenate(parts)


# -- the offline parts --------------------------------------------------------------------------------------------------------------

def test_a_flacs_stream_info_and_metadata_blocks_are_read(tmp_path):
    path = sine(tmp_path / "a.flac", 3)
    assert streaminfo(path) == (24000, 1, 16, 72000)
    header = flac_header(path)
    assert header[:4] == b"fLaC" and path.read_bytes().startswith(header) and len(header) < 20000 and not path.read_bytes()[len(header):len(header) + 2] == b"fL"
    (tmp_path / "x.flac").write_bytes(b"RIFF" + b"\0" * 60)
    with pytest.raises(ValueError, match="no FLAC"):
        streaminfo(tmp_path / "x.flac")


def test_the_cut_of_a_flac_is_the_source_sample_for_sample_and_has_no_tags(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sine(source / AUDIO, 10)
    whole = samples(source / AUDIO)
    cut(source, 3, 4, tmp_path / "mid.flac")
    cut(source, 8, None, tmp_path / "end.flac")
    assert np.array_equal(samples(tmp_path / "mid.flac"), whole[3 * 24000:7 * 24000])
    assert np.array_equal(samples(tmp_path / "end.flac"), whole[8 * 24000:]) and streaminfo(tmp_path / "end.flac") == (24000, 1, 16, 48000)
    raw = flac_header(tmp_path / "mid.flac")
    assert b"Lavf" not in raw and b"title" not in raw.lower() and b"artist" not in raw.lower()
    with pytest.raises(RuntimeError, match="ffmpeg could not cut"):
        cut(tmp_path, 0, 1, tmp_path / "none.flac")


def made_up_night(seed: int = 1) -> Night:
    """A night of two hours: no call for 700 s in any unit, then bursts of calls (six to ten, 10 to 25 s apart) with two to five minutes between
    them in the first hour, and a busy second hour (a call every 3 to 10 s)."""
    rng = random.Random(seed)
    calls, t = [], 700.0
    while t < 3600:
        for _ in range(rng.randint(6, 10)):
            calls.append(Call(round(t, 3), rng.choice((2900, 6632))))
            t += rng.uniform(10, 25)
        t += rng.uniform(120, 300)
    t = max(t, 3650.0)
    while t < 7190:
        calls.append(Call(round(t, 3), rng.choice((2900, 6632))))
        t += rng.uniform(3, 10)
    others = sorted(round(rng.uniform(650, 7190), 3) for _ in range(300))
    return Night(tuple(calls), tuple(others), START, 7200 * 24000)


def files_of(night: Night, count: int) -> list[realdump.File]:
    return [realdump.File(f"AUDIO/REC_{(night.start + timedelta(hours=k)):%Y%m%d_%H%M%S}.flac", "recorder", realdump.local_t(night.start + timedelta(hours=k)),
                          "recording", "audio", {"from": 3600 * k, "seconds": 3600}, length=3600.0) for k in range(count)]


def clock_t(night: Night, hhmm: tuple[str, str]) -> float:
    """Seconds into the recording of the clock minute `HH`, `MM` (the day of the night's start)."""
    return (datetime.combine(night.start.date(), datetime.strptime(hhmm[0] + hhmm[1], "%H%M").time()) - night.start).total_seconds()


def plain_check(night: Night, files, questions, scale: str) -> None:
    """Each question's truth and evidence as a plain reader of the calls finds them."""
    covered = sum(f.length for f in files)
    calls = [c for c in night.calls if c.t < covered]
    everyone = sorted([c.t for c in calls] + [t for t in night.others if t < covered])

    def wall(t: float) -> datetime:
        return night.start + timedelta(seconds=int(t))

    def where(span) -> float:
        return next(k for k, f in enumerate(files) if f.path == span.file) * 3600 + (span.t0 + span.t1) / 2

    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template = parts[1] if absent else parts[0]
        assert (q.kind == "absent") is absent and q.tags["modality"] == "audio" and q.scenario == "A4" and (q.tags["template"] == (
            "absent_did" if absent else template)), q.qid
        if template == "first":
            when = wall(calls[0].t)
            assert (q.grade.rule, q.grade.date, q.grade.time, q.grade.tol_s) == ("time", when.date().isoformat(), when.strftime("%H:%M:%S"), 60.0), q.qid
            assert abs(where(q.truth.evidence[0]) - calls[0].t) <= 0.5 and " first flight call" in q.question
        elif template == "pitch":
            call = calls[0] if parts[1] == "first" else calls[-1]
            assert q.grade.rule == "amount" and float(q.grade.truth) == call.hz / 1000 and q.grade.tol_s == 0.8 and f"the {parts[1]} flight call" in q.question
        elif template == "quiet":
            by_file = [sum(k * 3600 <= c.t < (k + 1) * 3600 for c in calls) for k in range(len(files))]
            names = [f.path.rsplit("/", 1)[-1] for f in files]
            assert q.grade.rule == "set" and q.grade.items == (names[by_file.index(min(by_file))],) and sorted(by_file)[0] * 2 <= sorted(by_file)[1]
        else:
            a, b = (clock_t(night, (m[0], m[1])) for m in (WINDOW.search(q.qid).groups()[:2], WINDOW.search(q.qid).groups()[2:]))
            assert f"between {wall(a):%H:%M} and {wall(b):%H:%M}" in q.question and a % 60 == (-night.start.second) % 60 and b <= covered
            inside = [c.t for c in calls if a <= c.t < b]
            if template == "did" and absent:
                assert not [t for t in everyone if a - 10 <= t < b + 10], f"{q.qid}: a call is marked in or near it, in some unit"
                assert q.grade.rule == "absent" and not q.truth.evidence and q.truth.value is None and 2 <= (b - a) / 60 <= 3
            elif template == "did":
                first = inside[0]
                assert first - a >= 90 and 2 <= (b - a) / 60 <= 3 and abs(where(q.truth.evidence[0]) - first) <= 0.5, q.qid
                assert (q.grade.rule, q.grade.time, q.grade.tol_s, q.grade.date) == ("time", wall(first).strftime("%H:%M:%S"), 60.0, None)
            else:
                assert 3 <= len(inside) <= 12 and 2 <= (b - a) / 60 <= 6 and q.grade.rule == "amount" and (q.grade.truth, q.grade.tol_s) == (str(len(inside)), 1.0)
                assert not any(abs(c.t - e) <= 1.5 for c in calls for e in (a, b)), f"{q.qid}: a call at an edge"
                assert sorted(round(where(s), 1) for s in q.truth.evidence) == sorted(round(t, 1) for t in inside)
    early = [q for q in questions if q.qid.startswith("did-") and abs(where(q.truth.evidence[0]) - calls[0].t) <= 0.5]
    assert len(early) == 1, "one window with a call holds the night's first call, which starts before it"
    assert early[0].split == next(q.split for q in questions if q.qid == "first"), "and is of the split of `first`, which has its answer"
    windows = [WINDOW.search(q.qid).groups() for q in questions if WINDOW.search(q.qid) and not q.qid.startswith("absent")]
    spans = sorted((clock_t(night, w[:2]), clock_t(night, w[2:])) for w in windows)
    assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:])), "no two windows with a call overlap"


def right(q) -> list[str]:
    """Natural answers in the shape asked, all right."""
    g = q.grade
    if g.rule == "time" and g.date:
        return [f"{g.date} {g.time[:5]}", f"It was at {g.time[:5]} on {date.fromisoformat(g.date):%d %B %Y}."]
    if g.rule == "time":
        return [g.time[:5], f"The first was at {g.time[:5]}."]
    if g.rule == "amount":
        return [g.truth, f"About {g.truth}", f"It was {g.truth}."]
    if g.rule == "set":
        return [", ".join(g.items), f"The recording is {g.items[0]}."]
    return ["unknown", "None", "No flight call.", "I found none, so zero."]


def wrong(q, others: list[str]) -> list[str]:
    """Planted wrong answers: a time two minutes out and a day out; a number three out, a hedge; another file; `unknown` on a needle; a time
    for an absent one."""
    g = q.grade
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date or '2015-09-24'}T{g.time}")
        later = t + timedelta(minutes=3)
        return [f"{g.date} {later:%H:%M}" if g.date else f"{later:%H:%M}", f"{later:%H:%M}", "unknown"] + (
            [f"{t + timedelta(days=1):%Y-%m-%d %H:%M}"] if g.date else [])
    if g.rule == "amount":
        v = float(g.truth)
        return [f"{v + 3:g}", f"{v:g} or {v + 3:g}", "unknown"]
    if g.rule == "set":
        return [next(o for o in others if o not in g.items), "unknown"]
    return ["03:48", "Yes, at 04:12.", "5 calls"]


def test_the_questions_of_a_made_up_night_are_what_its_calls_say():
    for seed in (1, 2, 3):
        night = made_up_night(seed)
        files = files_of(night, 2)
        questions = Q.questions_of(night, files, "tiny", seed, files[-1].t + 86400)
        assert len(questions) == 6 and sum(q.kind == "absent" for q in questions) == 1
        assert {q.tags["template"] for q in questions} == {"first", "pitch", "quiet", "did", "count", "absent_did"}
        plain_check(night, files, questions, "tiny")
        others = [f.path.rsplit("/", 1)[-1] for f in files]
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_night_with_no_quiet_stretch_asks_no_empty_window_and_no_hour_with_half_the_calls_of_the_next_is_no_quiet_question():
    night = made_up_night()
    busy = Night(tuple(Call(float(t), 2900) for t in range(100, 7100, 5)), (), START, 7200 * 24000)
    files = files_of(busy, 2)
    with pytest.raises(ValueError, match="no recording has at most half the calls of the next"):
        Q.questions_of(busy, files, "tiny", 1, files[-1].t + 86400)
    quiet_first = Night(tuple(Call(float(t), 2900) for t in (*range(100, 3600, 100), *range(3600, 7100, 5))), (), START, 7200 * 24000)
    with pytest.raises(ValueError, match="windows a question asks about"):
        Q.questions_of(quiet_first, files, "tiny", 1, files[-1].t + 86400)
    assert night.calls[0].t > 600


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys_and_scales():
    recipe = A4.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="recipe has the keys"):
        A4.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        A4.recipe(scale="huge", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        A4.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})


def test_no_command_may_reach_the_source_or_the_code():
    deny = A4.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:birdvox" in deny and "in:unit03" in deny and "in:a4_field" in deny and "in:from mmsb" in deny


def test_every_template_is_filed_and_the_shelf_is_a_shelf():
    from mmsb.core.taxonomy import check

    check(A4.shelf, A4.filing)
    assert {c for c, _ in A4.filing.values()} == {"when", "how_many", "what", "did", "who"}
    assert A4.filing["absent_did"] == A4.filing["did"]


# -- the real source ----------------------------------------------------------------------------------------------------------------

def plain_night() -> Night:
    """From the CSVs, as a second reader: unit 3's calls, the other units' on unit 3's clock, and the start by unit 3's clock."""
    root = SOURCE
    starts = {r[0]: int(r[1]) for r in csv.reader(open(root / "BirdVox-full-night_utc-start-times.csv")) if r[0].startswith("unit")}

    def read(unit: str) -> list[tuple[float, int]]:
        rows = list(csv.reader(open(root / f"BirdVox-full-night_csv-annotations_{unit}.csv")))[1:]
        return [(float(t), int(h)) for t, h in rows]

    mine = read("unit03")
    others = sorted(starts[u] - starts["unit03"] + t for u in starts if u != "unit03" for t, _ in read(u))
    return Night(tuple(Call(t, h) for t, h in mine), tuple(others), datetime.fromtimestamp(starts["unit03"], tz=timezone.utc).replace(tzinfo=None), 0)


@GROUP
@REAL
def test_the_scenarios_reader_agrees_with_a_plain_one_of_the_labels():
    plain, mine = plain_night(), scenario_night()
    assert [(c.t, c.hz) for c in mine.calls] == [(c.t, c.hz) for c in plain.calls] and len(mine.calls) == 9113
    assert list(mine.others) == list(plain.others) and len(mine.others) == 26289 and mine.start == plain.start == datetime(2015, 9, 24, 3, 31, 2)
    assert mine.samples == 944534424 and mine.samples / 24000 == 39355.601


def scenario_night() -> Night:
    from mmsb.scenarios.a4_field.source import read_night

    return read_night(str(SOURCE))


@GROUP
@REAL
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_answer_is_the_labels_at_the_standard_scale(seed):
    files, questions = A4.plan_of(A4.recipe(scale="standard", seed=seed))
    night = plain_night()
    night = Night(night.calls, night.others, night.start, 944534424)
    assert len(files) == 11 and [f.length for f in files] == [3600.0] * 10 + [3355.601]
    assert [f.path for f in files][:2] == ["AUDIO/REC_20150924_033102.flac", "AUDIO/REC_20150924_043102.flac"]
    plain_check(night, files, questions, "standard")
    assert len(questions) == 18 and sum(q.kind == "absent" for q in questions) == 4
    assert {"first", "pitch", "quiet", "did", "count", "absent_did"} == {q.tags["template"] for q in questions}
    assert not [q for q in questions if q.qid in ("last", "wait")] and A4.filing["quiet"][0] == "who"
    assert 4 <= sum(q.split == "dev" for q in questions) <= 7 and all(q.now.startswith("2015-09-25T09:00:00") for q in questions)


@GROUP
@REAL
def test_the_standard_night_is_eleven_hourly_files_whose_times_run_on_from_each_other():
    files, _ = A4.plan_of(A4.recipe(scale="standard", seed=1))
    assert all(b.t == a.t + a.length for a, b in zip(files, files[1:]) if a.length == 3600.0)
    assert sum(f.length for f in files) == 39355.601 and files[0].t == datetime(2015, 9, 24, 3, 31, 2).timestamp()
    assert all(f.spec["seconds"] == 3600 for f in files[:-1]) and files[-1].spec["seconds"] is None


@GROUP
@REAL
def test_another_seed_asks_other_windows_of_the_same_night_and_one_seed_makes_one_dump():
    one = A4.plan_of(A4.recipe(scale="standard", seed=1))
    assert one == A4.plan_of(A4.recipe(scale="standard", seed=1)) and A4.hashes(A4.recipe(scale="standard", seed=1)) == A4.hashes(
        A4.recipe(scale="standard", seed=1))
    files, other = A4.plan_of(A4.recipe(scale="standard", seed=2))
    assert files == one[0] and {q.qid for q in one[1]} != {q.qid for q in other}


@GROUP
@REAL
def test_the_empty_windows_are_empty_in_all_six_units_and_the_first_calls_of_the_night_are_the_labels():
    plain = plain_night()
    everyone = sorted([c.t for c in plain.calls] + list(plain.others))
    assert everyone[0] == pytest.approx(417.728, abs=0.001) and plain.calls[0].t == 1021.186 and plain.calls[-1].t == 39355.293
    _, questions = A4.plan_of(A4.recipe(scale="standard", seed=1))
    empty = [q for q in questions if q.kind == "absent"]
    assert len(empty) == 4
    for q in empty:
        a, b = (clock_t(plain, m) for m in (WINDOW.search(q.qid).groups()[:2], WINDOW.search(q.qid).groups()[2:]))
        assert not [t for t in everyone if a - 10 <= t < b + 10] and b < 912 + 1, q.qid


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("a4")
    recipe = A4.recipe(scale="tiny", seed=1)
    dump, questions = A4.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@GROUP
@REAL
def test_the_tiny_night_is_two_hourly_files_cut_from_the_source_sample_for_sample(built):
    out, recipe, dump, questions = built
    files, plan = A4.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] == ["AUDIO/REC_20150924_033102.flac", "AUDIO/REC_20150924_043102.flac"]
    assert len(plan) == 6 and sum(q.kind == "absent" for q in plan) == 1 and [q.qid for q in plan] == [q.qid for q in questions]
    for e, f in zip(dump.entries, files):
        path = out / "dump" / e.path
        assert os.stat(path).st_mtime == e.t == f.t and not e.own_time and e.length == f.length == 3600.0 and e.media == "audio"
        assert streaminfo(path) == (24000, 1, 16, 86400000)
        assert b"Lavf" not in flac_header(path)
    source = []
    with av.open(str(SOURCE / AUDIO)) as c:
        n = 0
        for frame in c.decode(audio=0):
            source.append(frame.to_ndarray().reshape(-1))
            n += len(source[-1])
            if n >= 7200 * 24000:
                break
    mine = np.concatenate([samples(out / "dump" / e.path) for e in dump.entries])
    assert np.array_equal(mine, np.concatenate(source)[:7200 * 24000]), "the samples are the source's"
    assert A4.probe(recipe, dump, out, None, 1).status == "passed"


@GROUP
@REAL
def test_every_answer_is_the_labels_at_the_tiny_scale(built):
    _, recipe, _, _ = built
    files, questions = A4.plan_of(recipe)
    night = plain_night()
    plain_check(Night(night.calls, night.others, night.start, 0), files, questions, "tiny")
    assert all(q.now.startswith("2015-09-25T09:00:00") for q in questions) and 1 <= sum(q.split == "dev" for q in questions) <= 3


@GROUP
@REAL
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        files, questions = A4.plan_of(A4.recipe(scale=scale, seed=1))
        others = [f.path.rsplit("/", 1)[-1] for f in files]
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


class Spy:
    """An agent that answers right in the `files` configuration, citing a moment of the truth, and `unknown` in any other."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": s.t0, "t1": s.t1} for s in q.truth.evidence[:1]]
        answer = right(q)[0] if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


@GROUP
@REAL
def test_build_run_and_report_over_the_tiny_night(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "a4_field", "mmsb.scenarios.a4_field")
    world.patch.setattr(a4_source, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: REAL_FFMPEG))  # (the world's is made up)
    assert world.main("build", "--scenario", "a4_field", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    questions = run.questions()
    spy = Spy(questions)
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    first = next(q for q in questions if q.qid == "first")
    prompt = spy.prompts["files", "first"]
    assert prompt.startswith("It is now Friday 25 September 2015, 09:00. The files of the field recorder I left out for the night are in ")
    assert first.question in prompt and "<the date and time, as YYYY-MM-DD HH:MM>" in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    (section,) = fs.read_json(folder / "report.json")["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"]) == (6, 6) and files["grounding"]["grounded"] == 6
    assert (listing["correct"], listing["n"]) == (1, 6) and section["leaks"] == [], "listing alone answered only the absent one"
