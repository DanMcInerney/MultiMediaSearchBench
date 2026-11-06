"""I3, the trail camera, from its real source on disk: the tiny card built (photos in order, with EXIF and their times), every answer
derived again here by a plain reader of the label file (never the scenario's), the grader taking each truth and refusing planted wrong
answers, the leak probe, and `mmmb build`, `run` and `report` over it with a stand-in agent. Skips, and says so, without the source."""

from __future__ import annotations

import json
import os
import threading
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401
from PIL import Image

from mmmb import scenarios
from mmmb.core import fs, grade
from mmmb.core.agents import claude
from mmmb.core.runfolder import RunFolder
from mmmb.core.runner import Agent
from mmmb.scenarios.i3_trailcam.questions import questions_of
from mmmb.scenarios.i3_trailcam.source import SOURCE, Photo

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


def check_against_labels(questions, card) -> None:
    """Each question's truth and evidence as the plain reader finds them."""
    path = {f: f"DCIM/100RECNX/IMG_{n:04d}.JPG" for n, (_, f, _) in enumerate(card, 1)}
    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template, species = (parts[1], parts[2]) if absent else (parts[0], parts[1])
        month = parts[-2] + "-" + parts[-1] if template == "days" and len(parts) > (4 if absent else 2) else None
        of = [(t, f) for t, f, s in card if s == species and (month is None or t[:7] == month)]
        days = sorted({t[:10] for t, _ in of})
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
            t, f = of[0] if template == "first" else of[-1]
            visit = [path[g] for x, g in of if abs((datetime.fromisoformat(x) - datetime.fromisoformat(t)).total_seconds()) <= 60]
            assert (q.grade.date, q.grade.time) == (t[:10], t[11:19]) and [s.file for s in q.truth.evidence] == visit, q.qid
            assert all(x[11:13] != "12" for x, _ in of if x[:10] == t[:10]), f"{q.qid} rests on a photo whose hour is in doubt"


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
    card = plain_card(*TINY)
    assert len(files) == len(card) == 199 and [f.path for f in files][:2] == ["DCIM/100RECNX/IMG_0001.JPG", "DCIM/100RECNX/IMG_0002.JPG"]
    assert [f.spec["src"] for f in files] == [f"images/{f}" for _, f, _ in card], "named in the order taken"
    assert len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}
    assert all(q.now.startswith("2013-12-23T09:00:00") for q in questions), "asked the morning after the last photo"


@pytest.mark.parametrize("scale, days", [("tiny", TINY), ("standard", ("2013-09-10", "2014-02-28"))])
def test_every_answer_is_the_labels(scale, days):
    _, questions = I3.plan_of(I3.recipe(scale=scale, seed=1))
    card = plain_card(*days)
    if scale == "standard":
        assert len(card) == 4256 and len(questions) == 24 and sum(q.kind == "absent" for q in questions) == 7
    check_against_labels(questions, card)


def test_another_seed_asks_other_questions_of_the_same_card():
    files, one = I3.plan_of(I3.recipe(scale="standard", seed=1))
    files_two, two = I3.plan_of(I3.recipe(scale="standard", seed=2))
    assert files == files_two and {q.qid for q in one} != {q.qid for q in two}
    check_against_labels(two, plain_card("2013-09-10", "2014-02-28"))


def test_no_first_or_last_question_rests_on_a_day_whose_hour_12_is_in_doubt():
    """A made-up card: the fox's first day has a photo at 12:xx (noon or the hour after midnight), the bobcat's last day too."""
    def photo(n, when, species):
        return Photo(f"{n}.jpg", datetime.fromisoformat(when), species)
    card = [photo(1, "2013-12-16 08:00:00", "fox"), photo(2, "2013-12-16 12:10:00", "fox"), photo(3, "2013-12-17 09:00:00", "fox"),
            photo(4, "2013-12-17 10:00:00", "bobcat"), photo(5, "2013-12-18 06:00:00", "bobcat"), photo(6, "2013-12-18 12:30:00", "bobcat"),
            photo(7, "2013-12-18 13:00:00", "empty")]
    for seed in range(5):
        qids = {q.qid for q in questions_of(card, [f"IMG_{n}.JPG" for n in range(1, 8)], ["bobcat", "fox"], "tiny", seed, lambda w: w.timestamp())}
        assert "first-fox" not in qids and "last-bobcat" not in qids
        assert {"last-fox", "first-bobcat"} & qids, "the others are asked"


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    for q in I3.plan_of(I3.recipe(scale="standard", seed=1))[1] + tiny[1]:
        assert grade.grade(q.grade, right(q)), q.qid
        for answer in wrong(q):
            assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_photo_is_copied_with_its_exif_and_its_time_and_its_image_data_untouched(built):
    out, recipe, dump, _ = built
    card = plain_card(*TINY)
    assert [e.path for e in dump.entries] == [f"DCIM/100RECNX/IMG_{n:04d}.JPG" for n in range(1, 200)]
    for n in (0, 77, 198):
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
    assert "bench-data" in deny and "in:caltech" in deny and "in:i3_trailcam" in deny and "in:from mmmb" in deny


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
    prompt = spy.prompts["files", "first-deer"]
    assert prompt.startswith("It is now Monday 23 December 2013, 09:00. The files of the memory card of my trail camera are in ")
    assert "When did my trail camera first photograph a deer?" in prompt and "<the date and time, as YYYY-MM-DD HH:MM>" in prompt
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
