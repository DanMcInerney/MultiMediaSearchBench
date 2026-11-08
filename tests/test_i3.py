"""I3, the trail camera, from its real source on disk: the tiny card built (photos in order, with EXIF and their times), every answer
derived again here by a plain reader of the label file (never the scenario's), the hour after midnight read again from the photos'
own strips, the grader taking each truth and refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and `report`
over it with a stand-in agent. Skips, and says so, without the source."""

from __future__ import annotations

import functools
import json
import os
import threading
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401
from PIL import Image

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import realdump
from mmsb.scenarios.i3_trailcam.questions import questions_of
from mmsb.scenarios.i3_trailcam.source import SOURCE, Photo

if not (SOURCE / "labels" / "caltech_camera_traps.json.zip").is_file():
    pytest.skip(f"the Caltech Camera Traps source is not at {SOURCE}, so I3 is not built", allow_module_level=True)

I3 = scenarios.get("i3_trailcam")
pytestmark = pytest.mark.xdist_group("i3")
TINY = ("2013-12-16", "2013-12-22")


def plain_card(first: str, last: str) -> list[tuple[str, str, str]]:
    """`(date_captured, file_name, label)` of each photo of location 23 on disk from `first` to `last`, in time order: read here
    from the label file, as a second reader."""
    with zipfile.ZipFile(SOURCE / "labels" / "caltech_camera_traps.json.zip") as z:
        data = json.loads(z.read("caltech_images_20210113.json"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    label = {a["image_id"]: names[a["category_id"]] for a in data["annotations"]}
    on_disk = set(os.listdir(SOURCE / "images"))
    return sorted((i["date_captured"], i["file_name"], label[i["id"]]) for i in data["images"]
                  if i["location"] == "23" and i["file_name"] in on_disk and first <= i["date_captured"][:10] <= last)


LETTER = (640, 0, 700, 34)  # of a photo whose strip says `2013-11-12 12:34:40 AM`: the box of the first letter of the AM or PM


def letter_of(file: str) -> np.ndarray:
    with Image.open(SOURCE / "images" / file) as img:
        return np.asarray(img.convert("L").crop(LETTER)) > 128


@functools.cache
def letters() -> tuple[np.ndarray, np.ndarray]:
    """The mean first letter of the strips of photos that are AM (hours 10 and 11) and PM (22 and 23, which are 10 and 11 PM): read from
    the photos, as a second reader (the scenario reads the right foot of the A in a box of its own)."""
    card = plain_card("2013-09-10", "2014-02-28")
    am = [letter_of(f) for t, f, _ in card if t[11:13] in ("10", "11")][:30]
    pm = [letter_of(f) for t, f, _ in card if t[11:13] in ("22", "23")][:30]
    return np.mean(am, 0), np.mean(pm, 0)


@functools.cache
def strip_says_am(file: str) -> bool:
    am, pm = letters()
    letter = letter_of(file)
    one, other = np.abs(letter - am).sum(), np.abs(letter - pm).sum()
    assert abs(one - other) > 250, f"{file}: the strip's letter is neither an A nor a P ({one}, {other})"  # a template apart: ~400
    return one < other


def told(card: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """`card` as the camera's clock has it: a photo whose hour is 12 and whose strip says AM is at 00:MM:SS; in the order of the times."""
    return sorted((f"{t[:11]}00{t[13:]}" if t[11:13] == "12" and strip_says_am(f) else t, f, s) for t, f, s in card)


def is_timelapse(taken: str) -> bool:
    return taken[14:19] == "00:00"


def check_against_labels(questions, card) -> None:
    """Each question's truth and evidence as the plain reader finds them."""
    path = {f: f"DCIM/100RECNX/IMG_{n:04d}.JPG" for n, (_, f, _) in enumerate(told(card), 1)}
    when = {f: t for t, f, _ in told(card)}
    written = {f: t for t, f, _ in card}  # as the dataset gives it
    motion = [t for t, _, _ in told(card) if not is_timelapse(t)]
    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template, species = (parts[1], parts[2]) if absent else (parts[0], parts[1])
        month = parts[-2] + "-" + parts[-1] if template == "days" and len(parts) > (4 if absent else 2) else None
        of = sorted((when[f], f) for t, f, s in card if s == species and (month is None or t[:7] == month))
        days = sorted({t[:10] for t, _ in of})
        assert species != "bird", f"{q.qid}: a bird, which a labeller may have missed in an empty frame"
        assert not (absent and species == "fox"), f"{q.qid}: a fox, which a night photo of a coyote is taken for"
        assert species.replace("_", " ") in q.question, f"{q.qid} asks of another animal: {q.question}"
        if month:
            named = datetime.strptime(month, "%Y-%m").strftime("%B %Y")  # (an English locale: the tests run in one)
            assert f" in {named} " in q.question, f"{q.qid} asks of another month: {q.question}"
        assert (q.kind == "absent") is absent and not (absent and of), q.qid
        if absent:
            assert q.grade.rule == "absent" and not q.truth.evidence
            continue
        assert sorted(s.file for s in q.truth.evidence) == sorted(path[f] for _, f in of) or template in ("first", "last"), q.qid
        if template == "days":
            assert q.grade.items == tuple(days), q.qid
        elif template == "count":
            assert q.grade.truth == str(len(days)), q.qid
        else:
            t = (of[0] if template == "first" else of[-1])[0]
            visit = [path[g] for x, g in of if abs((datetime.fromisoformat(x) - datetime.fromisoformat(t)).total_seconds()) <= 60]
            assert (q.grade.date, q.grade.time) == (t[:10], t[11:19]) and [s.file for s in q.truth.evidence] == visit, q.qid
            assert all(written[g][11:13] != "12" for x, g in of if x[:10] == t[:10]), f"{q.qid} rests on a photo whose hour is in doubt"
            edge = motion[0] if template == "first" else motion[-1]
            assert abs((datetime.fromisoformat(t) - datetime.fromisoformat(edge)).total_seconds()) > 60, f"{q.qid}: the listing answers it"


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "time": lambda: f"{g.date} {g.time[:5]}",
            "absent": lambda: "unknown"}[g.rule]()


def wrong(q) -> list[str]:
    """Planted wrong answers: a wrong, missing or extra day; one more; a time two minutes out or a day out; `unknown` on a needle; a
    time for an absent one."""
    g = q.grade
    if g.rule == "set":
        days = [date.fromisoformat(d) for d in g.items]
        extra = next(d for n in range(1, 40) if (d := days[-1] + timedelta(days=n)).isoformat() not in g.items).isoformat()
        out = [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra])]
        return out + ([", ".join(g.items[1:])] if len(g.items) > 1 else []) + ["unknown"]
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date}T{g.time}")
        return [f"{t + timedelta(minutes=2):%Y-%m-%d %H:%M}", f"{t + timedelta(days=1):%Y-%m-%d %H:%M}", "unknown"]
    return ["2013-12-18 06:12", "2013-12-18"]


@pytest.fixture(scope="module")
def tiny():
    return I3.plan_of(I3.recipe(scale="tiny", seed=1))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i3")
    recipe = I3.recipe(scale="tiny", seed=1)
    dump, questions = I3.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_card_is_a_week_of_photos_with_a_quarter_of_its_questions_absent(tiny):
    files, questions = tiny
    card = told(plain_card(*TINY))
    assert len(files) == len(card) == 199 and [f.path for f in files][:2] == ["DCIM/100RECNX/IMG_0001.JPG", "DCIM/100RECNX/IMG_0002.JPG"]
    assert [f.spec["src"] for f in files] == [f"images/{f}" for _, f, _ in card], "named in the order taken, the hour after midnight told"
    assert len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}
    assert all(q.now.startswith("2013-12-23T09:00:00") for q in questions), "asked the morning after the last photo"


@pytest.mark.parametrize("scale, days", [("tiny", TINY), ("standard", ("2013-09-10", "2014-02-28"))])
def test_every_answer_is_the_labels(scale, days):
    _, questions = I3.plan_of(I3.recipe(scale=scale, seed=1))
    card = plain_card(*days)
    if scale == "standard":
        assert len(card) == 4256 and len(questions) == 23 and sum(q.kind == "absent" for q in questions) == 6
    check_against_labels(questions, card)


def test_another_seed_asks_other_questions_of_the_same_card():
    files, one = I3.plan_of(I3.recipe(scale="standard", seed=1))
    for seed in (2, 3):
        files_other, other = I3.plan_of(I3.recipe(scale="standard", seed=seed))
        assert files == files_other and {q.qid for q in one} != {q.qid for q in other}
        check_against_labels(other, plain_card("2013-09-10", "2014-02-28"))


def asked_of(card, names=("bobcat", "fox")) -> set[str]:
    """The ids of the questions of a made-up `card` of `Photo`s, over five seeds, at the tiny scale."""
    out = set()
    for seed in range(5):
        out |= {q.qid for q in questions_of(card, [f"IMG_{n}.JPG" for n in range(1, len(card) + 1)], list(names), "tiny", seed,
                                            lambda w: w.timestamp())}
    return out


def photo(n, when, species, twelve=False):
    return Photo(f"{n}.jpg", datetime.fromisoformat(when), species, twelve)


def test_the_hour_after_midnight_is_told_from_noon_by_the_strip_of_each_photo():
    """The dataset writes the hour after midnight as 12, as it does noon. Each photo of the hour 12 says AM or PM in its burned-in strip
    (read here by its own letter, as a second reader): the night ones are at 00:MM:SS in the plan, in time order, with EXIF to match."""
    card = plain_card("2013-09-10", "2014-02-28")
    files, _ = I3.plan_of(I3.recipe(scale="standard", seed=1))
    twelve = [(t, f) for t, f, _ in card if t[11:13] == "12"]
    night = {f for _, f in twelve if strip_says_am(f)}
    assert len(twelve) == 579 and len(night) == 375, "the card's photos of the hour 12, and those the strip says AM of"
    taken = {f.spec["src"]: f.spec["taken"] for f in files}
    for t, f in twelve:
        assert taken[f"images/{f}"] == (f"{t[:10]}T00:{t[14:]}" if f in night else t.replace(" ", "T")), f
    assert [f.spec["taken"] for f in files] == sorted(f.spec["taken"] for f in files), "in the order of the corrected times"
    assert all(f.t == datetime.fromisoformat(str(f.spec["taken"])).timestamp() for f in files)
    assert taken["images/59e5e1bc-23d2-11e8-a6a3-ec086b02610b.jpg"] == "2013-11-12T00:34:40", "the photo whose strip says 12:34:40 AM"
    assert [f.spec["src"] for f in files] == [f"images/{f}" for _, f, _ in told(card)]


@pytest.mark.parametrize("scale, seeds", [("tiny", range(1, 6)), ("standard", range(1, 9))])
def test_no_question_is_about_a_bird_or_asks_for_an_absent_fox(scale, seeds):
    """A small bird in an empty frame is a label a careful annotator may have missed; a fox is what a night photo of a coyote is taken
    for. The standard card asks 23 questions, six of them absent, at every seed."""
    for seed in seeds:
        _, questions = I3.plan_of(I3.recipe(scale=scale, seed=seed))
        assert not [q.qid for q in questions if "bird" in q.qid or q.qid.startswith("absent-first-fox")]
        if scale == "standard":
            assert len(questions) == 23 and sum(q.kind == "absent" for q in questions) == 6, seed


def test_the_last_motion_photo_of_the_card_is_a_coyote_and_no_question_asks_for_it():
    """The card's last photo that is not a time-lapse one (all that come after are on the hour) is a coyote's at 13:32:07 on
    28 February 2014: the list of the files gave the answer to `last-coyote`, which was asked at seed 3 (and the first one is told
    the same way)."""
    card = told(plain_card("2013-09-10", "2014-02-28"))
    motion = [(t, s) for t, _, s in card if not is_timelapse(t)]
    assert motion[-1] == ("2014-02-28 13:32:07", "coyote") and all(is_timelapse(t) for t, _, _ in card if t > motion[-1][0])
    for seed in range(1, 9):
        _, questions = I3.plan_of(I3.recipe(scale="standard", seed=seed))
        assert "last-coyote" not in {q.qid for q in questions}, seed


def test_no_first_or_last_question_rests_on_a_day_whose_hour_12_is_in_doubt():
    """A made-up card: the fox's first day has a photo the dataset wrote as 12:xx (noon or the hour after midnight, however it is told
    now), the bobcat's last day too; a deer's photos are the card's first and last motion photos, so that only the hour decides."""
    card = [photo(0, "2013-12-16 05:00:00", "empty"), photo(1, "2013-12-16 05:20:10", "deer"), photo(2, "2013-12-16 08:00:00", "fox"),
            photo(3, "2013-12-16 12:10:00", "fox", True), photo(4, "2013-12-17 09:00:00", "fox"), photo(5, "2013-12-17 10:00:00", "bobcat"),
            photo(6, "2013-12-18 06:00:00", "bobcat"), photo(7, "2013-12-18 12:30:00", "bobcat", True), photo(8, "2013-12-18 13:00:00", "empty"),
            photo(9, "2013-12-18 21:15:30", "deer")]
    qids = asked_of(card)
    assert "first-fox" not in qids and "last-bobcat" not in qids
    assert {"last-fox", "first-bobcat"} <= qids, "the others are asked"


def test_no_first_or_last_question_is_answered_by_the_list_of_the_files():
    """A made-up card whose first photo that is not a time-lapse one (on the hour on the dot) is a fox's and whose last is another's:
    the list of the files gives their times, so they are not asked, nor is a bobcat whose last photo is within 60 s of the last one."""
    card = [photo(0, "2013-12-16 06:00:00", "empty"), photo(1, "2013-12-16 06:30:10", "fox"), photo(2, "2013-12-16 07:00:00", "empty"),
            photo(3, "2013-12-16 12:20:00", "deer"), photo(4, "2013-12-17 08:00:20", "bobcat"), photo(5, "2013-12-17 09:00:30", "bobcat"),
            photo(6, "2013-12-17 09:00:40", "fox"), photo(7, "2013-12-17 10:00:00", "empty"), photo(8, "2013-12-17 11:00:00", "empty")]
    qids = asked_of(card, ("badger", "opossum"))
    assert "first-fox" not in qids and "last-fox" not in qids, "the card's first and last photo that is not a time-lapse one"
    assert "last-bobcat" not in qids, "a minute from the last"
    assert {"first-deer", "last-deer", "first-bobcat"} <= qids, "the others are asked"


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    for q in I3.plan_of(I3.recipe(scale="standard", seed=1))[1] + tiny[1]:
        assert grade.grade(q.grade, right(q)), q.qid
        for answer in wrong(q):
            assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_photo_is_copied_with_its_exif_and_its_time_and_its_image_data_untouched(built):
    out, recipe, dump, _ = built
    card = told(plain_card(*TINY))
    assert [e.path for e in dump.entries] == [f"DCIM/100RECNX/IMG_{n:04d}.JPG" for n in range(1, 200)]
    night = next(n for n, (t, f, _) in enumerate(card) if t[11:13] == "00")  # a photo of the hour after midnight, written 12:xx by the dataset
    for n in (0, 77, 198, night):
        e, (taken, name, _) = dump.entries[n], card[n]
        path = out / "dump" / e.path
        wall = datetime.fromisoformat(taken)
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == wall == datetime.fromtimestamp(e.t) and e.own_time
        exif = Image.open(path).getexif()
        assert exif[0x0132] == exif.get_ifd(0x8769)[0x9003] == wall.strftime("%Y:%m:%d %H:%M:%S")
        source, copy = (SOURCE / "images" / name).read_bytes(), path.read_bytes()
        assert copy.endswith(source[2:]) and len(copy) - len(source) < 300
    assert I3.probe(recipe, dump, out, None, 1).status == "passed"


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = I3.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I3.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I3.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I3.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = I3.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:caltech" in deny and "in:i3_trailcam" in deny and "in:from mmsb" in deny


class Spy:
    """An agent that answers right in the `files` configuration, citing a photo of the truth, and `unknown` in any other."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": 0, "t1": 0} for s in q.truth.evidence[:1]]
        answer = right(q) if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


def test_build_run_and_report_over_the_tiny_card(world, capsys):  # noqa: F811
    assert world.main("build", "--scenario", "i3_trailcam", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    prompt = spy.prompts["files", "first-skunk"]
    assert prompt.startswith("It is now Monday 23 December 2013, 09:00. The files of the memory card of my trail camera are in ")
    assert "When did my trail camera first photograph a skunk?" in prompt and "<the date and time, as YYYY-MM-DD HH:MM>" in prompt
    assert "You have 900 seconds: " in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (8, 8, 8)
    assert (listing["correct"], listing["n"]) == (2, 8) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 8}
