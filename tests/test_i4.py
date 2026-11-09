"""I4, the building inspector's photos, from its real source on disk: the tiny pile built (JPEGs under numbers in job folders, EXIF holding the
time and nothing else), every answer derived again here by a plain reader of the label file (never the scenario's), the held-out classes and
the classes taken for them checked absent where an absent question asks, the grader taking each truth and refusing planted wrong answers, the
leak probe, and `mmsb build`, `run` and `report` over it with a stand-in agent. Skips, and says so, without the source."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401
from PIL import Image

from mmsb import scenarios
from mmsb.core import fs, grade, taxonomy
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import i4_site_photos as I4
from mmsb.scenarios import realdump
from mmsb.scenarios.i4_site_photos import questions as Q
from mmsb.scenarios.i4_site_photos import source as S

pytestmark = pytest.mark.xdist_group("i4")
REAL = (S.SOURCE / S.LABELS).is_file() and (S.SOURCE / "test_images").is_dir()
needs_source = pytest.mark.skipif(not REAL, reason=f"the BD3 source is not at {S.SOURCE}, so I4 is not built")
HELD = {"stain", "spalling"}  # the classes the pile has no photo of
CLASSES_OF = {"cracks": {"major_crack", "minor_crack"}, "algae": {"algae"}, "peeling": {"peeling"}}


def plain_labels() -> dict[str, str]:
    """The class of each photo (by its name in the source), read here from the JSONL."""
    with open(S.SOURCE / S.LABELS, encoding="utf-8") as f:
        return {json.loads(line)["file"].rsplit("/", 1)[-1]: json.loads(line)["answer"] for line in f}


def plan(scale: str, seed: int):
    return I4.plan_of(I4.recipe(scale=scale, seed=seed))


# -- what needs no source -----------------------------------------------------------------------------------------------------------------

def test_a_png_is_written_as_a_jpeg_with_its_time_alone_and_its_picture_kept():
    import io
    img = Image.new("RGB", (64, 64), (120, 100, 80))
    img.paste((250, 20, 20), (0, 0, 32, 64))
    png = io.BytesIO()
    img.save(png, "PNG")
    out = S.jpeg_of(png.getvalue(), datetime(2025, 3, 10, 9, 5, 7))
    assert out[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(out)) as back:
        exif = back.getexif()
        assert back.format == "JPEG" and back.size == (64, 64)
        assert dict(exif) == {0x0132: "2025:03:10 09:05:07", 0x8769: exif[0x8769]} and not exif.get_ifd(0x8825)
        assert dict(exif.get_ifd(0x8769)) == {0x9003: "2025:03:10 09:05:07", 0x9004: "2025:03:10 09:05:07"}
        assert abs(back.convert("RGB").getpixel((10, 30))[0] - 250) < 12 and abs(back.convert("RGB").getpixel((55, 40))[0] - 120) < 12


def test_a_recipe_is_refused_in_another_time_zone_with_other_keys_and_at_another_scale():
    recipe = I4.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I4.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I4.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I4.recipe(scale="huge", seed=1)


def test_every_template_is_filed_and_four_categories_are_used():
    taxonomy.check(I4.shelf, I4.filing)
    assert I4.shelf.media == "image" and I4.shelf.area == "work"
    assert {I4.filing[t][0] for t in ("photos", "count", "any", "one")} == {"find_all", "how_many", "did", "find"}
    assert all(I4.filing["absent_" + t] == I4.filing[t] for t in ("photos", "count", "any", "one"))


def test_what_is_asked_about_and_what_is_held_out_are_told_by_the_tables():
    assert set(Q.THINGS) == {"cracks", "algae", "peeling"} and Q.HELD == {"stain": "algae", "spalling": "peeling"}
    assert Q.classes_of("cracks") == ("major_crack", "minor_crack"), "the two cracks are asked together: a width is no thing to tell"
    assert not {c for thing in Q.THINGS for c in Q.classes_of(thing)} & set(Q.HELD), "no class is asked about that the pile does not hold"


# -- what reads the source ----------------------------------------------------------------------------------------------------------------

def folders_of(files) -> dict[str, list[str]]:
    """The dump's photos by job folder: the source name of each, in the order taken."""
    out: dict[str, list[str]] = defaultdict(list)
    for f in files:
        out[f.path.split("/")[1]].append(f.spec["src"].rsplit("/", 1)[-1])
    return out


def check_against_labels(files, questions, scale: str, seed: int) -> None:
    """Each question's template, folder, class and truth as the plain reader of the label file finds them."""
    label = plain_labels()
    folders = folders_of(files)
    by_path = {f.path: f.spec["src"].rsplit("/", 1)[-1] for f in files}
    dump_name = {src: path.rsplit("/", 1)[-1] for path, src in by_path.items()}  # the source's name of a photo -> its name in the dump
    in_pile = {label[n] for names in folders.values() for n in names}
    assert in_pile >= {"plain", "major_crack", "minor_crack", "algae", "peeling"} and not in_pile & HELD
    for q in questions:
        template, absent = str(q.tags["template"]).removeprefix("absent_"), q.kind == "absent"
        m = re.fullmatch(r"(?:absent-)?(photos|count|any|one)-(.+)-(cracks|algae|peeling)", q.qid)
        assert m and m[1] == template, q.qid
        site = next(name for name in folders if re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") == m[2])
        assert site in q.question, q.qid
        thing = m[3]
        words = {"cracks": "cracks", "algae": "algae", "peeling": "peeling paint"}[thing]
        assert words in q.question and q.question.endswith(("?", ".")), q.qid
        names = folders[site]
        wanted = [n for n in names if label[n] in CLASSES_OF[thing]]
        if absent:
            assert q.grade.rule == "absent" and not q.truth.evidence and not wanted, q.qid
            others = [n for j, ns in folders.items() if j != site for n in ns]
            assert [n for n in others if label[n] in CLASSES_OF[thing]], f"{q.qid}: no other folder holds the class: it is asked as present elsewhere"
            continue
        assert wanted, q.qid
        if template in ("photos", "one"):
            assert sorted(q.grade.items) == sorted(dump_name[n] for n in wanted) and q.grade.rule == "set" and len(wanted) <= 20, q.qid
            assert (len(wanted) == 1) == (template == "one"), q.qid
            assert 2 <= len(wanted) or template == "one"
        elif template == "count":
            assert q.grade.rule == "count" and q.grade.truth == str(len(wanted)), q.qid
        else:
            assert q.grade.items == (dump_name[wanted[0]],), "the first of them in the order taken"
        evidence = {by_path[s.file] for s in q.truth.evidence}
        assert evidence <= set(wanted) and (template != "any" or evidence == {wanted[0]}), q.qid
    assert len(questions) == (6 if scale == "tiny" else 24) and sum(q.kind == "absent" for q in questions) == (2 if scale == "tiny" else 6), seed


@needs_source
@pytest.mark.parametrize("scale, seeds", [("tiny", range(1, 6)), ("standard", range(1, 9))])
def test_every_answer_is_the_labels(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        check_against_labels(files, questions, scale, seed)


@needs_source
def test_the_standard_pile_is_the_photos_but_two_classes_in_24_job_folders():
    label = plain_labels()
    assert len(label) == 793
    files, questions = plan("standard", 1)
    folders = folders_of(files)
    assert not {label[n] for names in folders.values() for n in names} & HELD, "stain and spalling have no photo in the pile"
    assert len(folders) == 24 and 450 < len(files) <= 793 - sum(1 for v in label.values() if v in HELD)
    assert max(len(v) for v in folders.values()) < 40
    assert [f.path for f in files][:1] == ["Site photos/" + next(iter(folders)) + "/IMG_0001.jpg"]
    assert [f.path.rsplit("/", 1)[-1] for f in files] == [f"IMG_{n:04d}.jpg" for n in range(1, len(files) + 1)], "numbered across folders"
    assert [f.spec["taken"] for f in files] == sorted(f.spec["taken"] for f in files)
    assert all(f.t == datetime.fromisoformat(str(f.spec["taken"])).timestamp() and f.own_time for f in files)
    day_after = (datetime.fromisoformat(files[-1].spec["taken"]) + timedelta(days=1)).date().isoformat()
    assert all(q.now.startswith(f"{day_after}T09:00") for q in questions), "asked the morning after the last photo"
    assert sum(q.split == "dev" for q in questions) == 7 and {q.split for q in questions} == {"dev", "test"}


@needs_source
def test_another_seed_asks_other_questions_and_the_same_three_things_are_asked_both_ways():
    """What a question asks of does not tell it absent: each of cracks, algae and peeling is asked as present and as absent, and no question is
    of stain or spalling."""
    files, one = plan("standard", 1)
    asked: dict[bool, set[str]] = {True: set(), False: set()}
    for seed in range(1, 9):
        other_files, other = plan("standard", seed)
        if seed > 1:
            assert {q.qid for q in one} != {q.qid for q in other} and [f.spec["src"] for f in files] != [f.spec["src"] for f in other_files]
        for q in other:
            asked[q.kind == "absent"].add(q.qid.rsplit("-", 1)[-1])
    assert asked == {True: {"cracks", "algae", "peeling"}, False: {"cracks", "algae", "peeling"}}


@needs_source
def test_a_folder_that_lacks_a_class_is_no_smaller_than_the_others():
    """Its photos of the class go to other folders, so the size of a folder does not tell an absent question: across seeds the folders absent
    questions are about are as big, on average, as the rest."""
    absent, rest = [], []
    for seed in range(1, 9):
        files, questions = plan("standard", seed)
        folders = folders_of(files)
        asked = {re.fullmatch(r"absent-\w+-(.+)-(?:cracks|algae|peeling)", q.qid)[1] for q in questions if q.kind == "absent"}
        for name, names in folders.items():
            (absent if re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") in asked else rest).append(len(names))
    assert len(absent) == 8 * 6 and abs(sum(absent) / len(absent) - sum(rest) / len(rest)) < 0.12 * sum(rest) / len(rest)


@needs_source
def test_no_name_says_a_class_or_when_and_no_file_holds_gps():
    files, _ = plan("standard", 1)
    assert not any(re.search(r"\d{8}_\d{6}", f.path) or re.search(r"(?i)crack|algae|stain|peel|spall|plain", f.path) for f in files)


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "absent": lambda: "none"}[g.rule]()


def wrong(q) -> list[str]:
    g = q.grade
    if g.rule == "set":
        return [", ".join([*g.items, "IMG_9999.jpg"]), "IMG_9999.jpg", "unknown", *([", ".join(g.items[1:])] if len(g.items) > 1 else [])]
    if g.rule == "count":
        return [str(int(g.truth) + 1), str(int(g.truth) - 1), "unknown"]
    return ["IMG_0001.jpg", "3"]


@needs_source
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        for seed in (1, 2, 3):
            for q in plan(scale, seed)[1]:
                assert grade.grade(q.grade, right(q)), (q.qid, right(q))
                for answer in wrong(q):
                    assert not grade.grade(q.grade, answer), (q.qid, answer)


# -- what builds ---------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i4")
    recipe = I4.recipe(scale="tiny", seed=1)
    dump, questions = I4.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@needs_source
def test_a_photo_is_written_as_a_jpeg_with_its_time_alone_and_its_picture_kept(built):
    out, recipe, dump, _ = built
    files, _ = I4.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] and len(files) == 45
    for n in (0, len(files) // 2, len(files) - 1):
        e, f = dump.entries[n], files[n]
        path = out / "dump" / e.path
        wall = datetime.fromisoformat(str(f.spec["taken"]))
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == wall == datetime.fromtimestamp(e.t) and e.own_time
        with Image.open(path) as img, Image.open(S.SOURCE / "test_images" / str(f.spec["src"]).rsplit("/", 1)[-1]) as src:
            exif = img.getexif()
            assert img.format == "JPEG" and img.size == src.size == (512, 512)
            assert set(exif) == {0x0132, 0x8769} and exif[0x0132] == exif.get_ifd(0x8769)[0x9003] == wall.strftime("%Y:%m:%d %H:%M:%S")
            assert not exif.get_ifd(0x8825), "no GPS"
            small, small_src = (i.convert("RGB").resize((8, 8)).tobytes() for i in (img, src))
            diff = sum(abs(a - b) for a, b in zip(small, small_src, strict=True))
            assert diff / (8 * 8 * 3) < 6, "the picture is the source's"
    assert I4.probe(recipe, dump, out, None, 1).status == "passed"


@needs_source
def test_the_probe_fails_a_photo_whose_name_says_a_class(built, tmp_path):
    out, recipe, dump, _ = built
    entry = dump.entries[0]
    renamed = type(dump)(dump.key, dump.root, dump.scenario, (entry.__class__(**{**entry.__dict__, "path": "Site photos/Maple Street/spalling.jpg"}),
                                                              *dump.entries[1:]), ())
    result = I4.probe(recipe, renamed, tmp_path, None, 1)
    assert result.status == "failed" and any("spalling" in f for f in result.failures)


@needs_source
def test_no_command_may_reach_the_source_or_the_code():
    deny = I4.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:bd3" in deny and "in:i4_site_photos" in deny and "in:from mmsb" in deny


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


@needs_source
def test_build_run_and_report_over_the_tiny_pile(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "i4_site_photos", "mmsb.scenarios.i4_site_photos")
    assert world.main("build", "--scenario", "i4_site_photos", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    qid = next(q.qid for q in run.questions() if q.qid.startswith("photos-"))
    prompt = spy.prompts["files", qid]
    assert prompt.startswith("It is now ") and "The files of my site photos are in " in prompt and "Site photos holds a folder for each job" in prompt
    assert "the names of the files, as in IMG_0042.jpg" in prompt and "You have 900 seconds: " in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (6, 6, 6)
    assert (listing["correct"], listing["n"]) == (2, 6) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 6}
