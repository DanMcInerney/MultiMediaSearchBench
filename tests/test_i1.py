"""I1, the phone camera roll, from its real source on disk: the tiny pile built (photos numbered in the order taken, EXIF holding the time and
nothing else, the GPS gone), every answer derived again here by a plain reader of the dataset's JSON and the folder's listing (never the
scenario's), the held-out needles checked absent from the pile and everything within a minute of them, the grader taking each truth and
refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and `report` over it with a stand-in agent. Skips, and says so,
without the source. Tests of the parts that need no source (dates, EXIF, the draw) come first."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401
from PIL import Image

from mmsb import scenarios
from mmsb.core import fs, grade, taxonomy
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import i1_camera_roll as I1
from mmsb.scenarios import realdump
from mmsb.scenarios.i1_camera_roll import questions as Q
from mmsb.scenarios.i1_camera_roll import source as S

pytestmark = pytest.mark.xdist_group("i1")
REAL = (S.SOURCE / S.QUESTIONS).is_file() and (S.SOURCE / S.IMAGES).is_dir()
needs_source = pytest.mark.skipif(not REAL, reason=f"the ATM-Bench source is not at {S.SOURCE}, so I1 is not built")
TINY = ("2022-06-19", "2022-06-21")
STANDARD = ("2022-04-30", "2025-06-25")
STAMP = re.compile(r"\d{8}_\d{6}")


# -- what needs no source -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    ("July 31st, 2024.", "2024-07-31"), ("24 December 2022.", "2022-12-24"), ("Oct 5, 2022", "2022-10-05"), ("20230205", "2023-02-05"),
    ("Sept 3, 2024", "2024-09-03"), (" June 26, 2024.", "2024-06-26"), ("On December 10 2023.", "2023-12-10"), ("March 02, 2023", "2023-03-02"),
    ("May 25. 2024.", "2024-05-25"), ("4 September 2022.", "2022-09-04"),
    ("10 days", None), ("Around 25.", None), ("July 5 and July 6, 2024", None), ("30 February 2023", None), ("", None)])
def test_a_stated_date_is_the_one_date_an_answer_says_in_full(answer, expected):
    found = S.stated_date(answer)
    assert (found.isoformat() if found else None) == expected


def jpeg(extra: bytes = b"", size=(8, 6)) -> bytes:
    """A real JPEG with an EXIF block holding a GPS position and a make, an XMP segment and a comment, as the dataset's are."""
    import io
    img = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x010F] = "Maker"
    exif[0x0112] = 6
    exif.get_ifd(0x8825)[1] = "N"
    exif.get_ifd(0x8825)[2] = (52.0, 12.0, 5.0)
    out = io.BytesIO()
    img.save(out, "JPEG", exif=exif, comment=b"a comment", quality=90)
    return out.getvalue()


def test_a_jpeg_is_stamped_with_its_time_alone_and_its_image_data_untouched():
    import io
    source = jpeg()
    assert b"Maker" in source and b"comment" in source
    wall = datetime(2023, 3, 2, 16, 58, 53)
    out = S.stamped(source, wall)
    with Image.open(io.BytesIO(out)) as img:
        exif = img.getexif()
        assert dict(exif) == {0x0112: 6, 0x0132: "2023:03:02 16:58:53", 0x8769: exif[0x8769]}, "the time and the orientation, no make"
        assert dict(exif.get_ifd(0x8769)) == {0x9003: "2023:03:02 16:58:53", 0x9004: "2023:03:02 16:58:53"} and not exif.get_ifd(0x8825)
        assert img.size == (8, 6)
    assert b"Maker" not in out and b"comment" not in out and b"Exif\0\0" in out
    assert out.endswith(source[source.rindex(b"\xff\xda"):]), "from the start of scan on, the bytes are the source's"
    with pytest.raises(ValueError, match="not a JPEG"):
        S.stamped(b"GIF89a", wall)


def test_an_hour_the_clock_skips_is_no_time_of_the_pile():
    skipped, repeated, plain = datetime(2023, 3, 12, 2, 30), datetime(2023, 11, 5, 1, 30), datetime(2023, 6, 1, 12, 0)
    zone_has_dst = datetime.fromtimestamp(skipped.timestamp()) != skipped
    assert I1.clock_ok(plain)
    assert I1.clock_ok(skipped) is (not zone_has_dst) and I1.clock_ok(repeated) is (not zone_has_dst)


def test_a_recipe_is_refused_in_another_time_zone_with_other_keys_and_at_another_scale():
    recipe = I1.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I1.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I1.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I1.recipe(scale="huge", seed=1)


def test_every_template_is_filed_and_the_eight_categories_are_all_used():
    taxonomy.check(I1.shelf, I1.filing)
    assert I1.shelf.media == "image" and I1.shelf.area == "home"
    templates = {t for t in I1.filing if not t.startswith("absent_")}
    assert templates == {"find", "photos", "date", "price", "read", "which", "days", "first", "ever"}
    assert {I1.filing[t][0] for t in templates} == set(taxonomy.CATEGORY_KEYS)
    assert all(I1.filing["absent_" + t] == I1.filing[t] for t in templates)


def test_the_tables_hold_no_word_of_one_thing_in_the_phrase_of_another():
    """`first` asks of two things and takes the answer by a word: no word of one is in the other's phrase."""
    for a, (phrase, word) in Q.THINGS.items():
        assert word in phrase.casefold(), a
        for b, (other, _) in Q.THINGS.items():
            assert a == b or word not in re.findall(r"[a-z]+", other.casefold()) and word not in other.casefold().replace("'s", ""), (a, b)
    assert set(Q.HOLD["read"]) == set(Q.READ) and set(Q.HOLD["ever"]) == set(Q.THINGS)
    assert not Q.BLOCK & (set(Q.READ) | set(Q.WHICH) | set(Q.PRICE) | set(Q.THINGS) | set(Q.DAYS))


# -- what reads the source ----------------------------------------------------------------------------------------------------------------

def plain_real() -> dict[str, dict]:
    """The dataset's questions by the first six characters of their id, read here from the JSON."""
    with open(S.SOURCE / S.QUESTIONS, encoding="utf-8") as f:
        return {q["id"][:6]: q for q in json.load(f)}


def plain_when(name: str) -> datetime:
    return datetime.strptime(name[:15], "%Y%m%d_%H%M%S")


def plain_files() -> dict[str, list[str]]:
    """The photos of the folder by the time in their name: a time with more than one photo is no question's evidence."""
    out: dict[str, list[str]] = defaultdict(list)
    for name in os.listdir(S.SOURCE / S.IMAGES):
        out[name[:15]].append(name)
    return out


def words(text: str) -> str:
    return " " + " ".join(re.findall(r"[^\W_]+", text.casefold())) + " "


def plain_day(answer: str) -> str:
    """The date of a stated answer, read here by `strptime` over the few shapes the answers have."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", re.sub(r"^On ", "", answer.strip()).strip(".,").replace(". ", ", ").replace("Sept", "Sep"))
    for shape in ("%B %d, %Y", "%B %d %Y", "%d %B %Y", "%b %d, %Y", "%b %d %Y", "%d %b %Y", "%Y%m%d"):
        try:
            return datetime.strptime(text.strip(), shape).date().isoformat()
        except ValueError:
            continue
    raise AssertionError(f"no date in {answer!r}")


def plan(scale: str, seed: int):
    return I1.plan_of(I1.recipe(scale=scale, seed=seed))


def source_of(files, q) -> list[str]:
    """The names of the source's photos behind the evidence of a question (or its dump's names given)."""
    spec = {f.path: f.spec["src"].rsplit("/", 1)[-1] for f in files}
    return [spec[s.file] for s in q.truth.evidence]


def check_against_labels(files, questions) -> None:
    """Each question's template, text, truth and evidence as the plain reader of the dataset finds them."""
    real, stems = plain_real(), plain_files()
    spec = {f.path.rsplit("/", 1)[-1]: f.spec["src"].rsplit("/", 1)[-1] for f in files}  # the dump's base name -> the source's
    have = {name[:15] for name in spec.values()}
    walls = sorted(plain_when(n) for n in spec.values())
    for q in questions:
        template, absent = str(q.tags["template"]), q.kind == "absent"
        base = template.removeprefix("absent_")
        ids = q.qid.removeprefix("absent-").split("-")[1:]  # one id, or two for `first`
        r = real[ids[0]]
        found = [stems[e] for e in r["evidence_ids"]]
        assert all(len(f) == 1 for f in found), f"{q.qid} rests on a photo another shares its second with"
        evidence = [f[0] for f in found]
        needles = [plain_when(e) for e in evidence]
        assert (q.grade.rule == "absent") is absent and absent == (not q.truth.evidence), q.qid
        if absent:
            assert not {n[:15] for n in evidence} & have, f"{q.qid}: a photo of the held-out question is in the pile"
            for w in needles:
                assert not [x for x in walls if abs((x - w).total_seconds()) <= Q.HOLD_S], f"{q.qid}: a photo within a minute of it is in the pile"
            assert base in Q.HOLD and ids[0] in Q.HOLD[base], q.qid
            continue
        assert source_of(files, q) == sorted(evidence, key=lambda n: (plain_when(n), n)) or base == "first", q.qid
        assert all(not near_midnight(w) for w in needles), q.qid
        assert ids[0] not in Q.BLOCK and "�" not in q.question
        if base in ("find", "photos"):
            assert q.question == " ".join(r["question"].split()) and r["qtype"] == "list_recall", q.qid
            assert set(re.findall(r"\d{8}_\d{6}", r["answer"])) == set(r["evidence_ids"]), "the answer names the evidence"
            assert sorted(q.grade.items) == sorted(p.rsplit("/", 1)[-1] for p in (s.file for s in q.truth.evidence)) and q.grade.rule == "set"
            assert (len(evidence) == 1) == (base == "find") and len({w.date() for w in needles}) == 1
            assert all(not [x for x in walls if 0 < abs((x - w).total_seconds()) <= Q.NEIGHBOUR_S and x not in needles] for w in needles), q.qid
        elif base == "date":
            assert q.question == " ".join(r["question"].split()) and r["qtype"] == "number"
            assert q.grade.items == (plain_day(r["answer"]),) == tuple({w.date().isoformat() for w in needles}), q.qid
            assert not re.search(r"(?i)today is|last time|yesterday", q.question)
        elif base == "price":
            number = re.search(r"\d[\d.]*", r["answer"].replace(",", ""))[0].rstrip(".")
            assert q.grade.rule == "amount" and float(q.grade.truth) == float(number), q.qid
        elif base in ("read", "which"):
            assert q.question == " ".join(r["question"].split()) and q.grade.rule == "text" and not q.grade.wrong
            assert any(words(x) in words(r["answer"]) for x in q.grade.truth.split("|")), f"{q.qid}: {r['answer']!r}"
        elif base == "days":
            expected = {"093253": "2", "90ba46": "10"}[ids[0]]
            assert q.grade.truth == expected and re.search(rf"\b{expected}\b|twice|ten", r["answer"], re.I)
            dates = sorted({w.date() for w in needles})
            assert str(len(dates) if ids[0] == "093253" else (dates[-1] - dates[0]).days) == expected
        elif base == "ever":
            assert q.grade.items == tuple({w.date().isoformat() for w in needles}) and Q.THINGS[ids[0]][0] in q.question
        elif base == "first":
            a, b = (real[i] for i in ids)
            first, second = min(plain_when(e) for e in a["evidence_ids"]), min(plain_when(e) for e in b["evidence_ids"])
            assert first < second - timedelta(seconds=Q.GAP_S), q.qid
            assert q.grade.truth == Q.THINGS[ids[0]][1] and q.grade.wrong == (Q.THINGS[ids[1]][1],)
            assert q.question == f"Which did I photograph first: {Q.THINGS[ids[0]][0]} or {Q.THINGS[ids[1]][0]}?"
            assert (second - first) <= timedelta(days=Q.TRIP_DAYS)
        else:
            raise AssertionError(base)


def near_midnight(wall: datetime) -> bool:
    day = datetime.combine(wall.date(), datetime.min.time())
    return min(wall - day, day + timedelta(days=1) - wall) < timedelta(seconds=60)


@needs_source
@pytest.mark.parametrize("scale, seeds", [("tiny", range(1, 6)), ("standard", range(1, 9))])
def test_every_answer_is_the_labels(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        if scale == "tiny":
            assert len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2, seed
        else:
            assert len(questions) == 24 and sum(q.kind == "absent" for q in questions) == 6, seed
        check_against_labels(files, questions)


@needs_source
def test_the_standard_pile_is_the_photos_but_the_held_out_ones_and_the_tiny_pile_is_three_days():
    photos = plain_files()
    assert sum(len(v) for v in photos.values()) == 3759
    files, questions = plan("standard", 1)
    held = sum(len(v) for v in photos.values()) - len(files)
    assert 0 < held < 30 and len(files) > 3700, "the held-out photos and those within a minute of them: a few"
    tiny, _ = plan("tiny", 1)
    assert 150 < len(tiny) < 250
    assert {f.spec["taken"][:10] for f in tiny} == {"2022-06-19", "2022-06-20", "2022-06-21"}
    assert sum(q.split == "dev" for q in plan("tiny", 1)[1]) == 2
    assert all(q.now.startswith("2025-06-26T09:00") for q in questions), "asked the morning after the last photo"
    assert sum(q.split == "dev" for q in questions) == 7 and {q.split for q in questions} == {"dev", "test"}


@needs_source
def test_another_seed_asks_other_questions_and_holds_out_other_photos():
    files, one = plan("standard", 1)
    for seed in (2, 3):
        other_files, other = plan("standard", seed)
        assert {q.qid for q in one} != {q.qid for q in other}
        assert {f.spec["src"] for f in files} != {f.spec["src"] for f in other_files}, "the held-out needles are the seed's"
    seen = {q.qid.removeprefix("absent-").split("-")[0] for seed in range(1, 9) for q in plan("standard", seed)[1]}
    assert seen >= {"find", "photos", "date", "price", "read", "which", "days", "first", "ever"}, "every template is asked at some seed"


@needs_source
def test_the_files_are_numbered_in_the_order_taken_and_no_name_says_when():
    files, _ = plan("standard", 1)
    assert [f.path for f in files][:2] == ["DCIM/Camera/IMG_0001.jpg", "DCIM/Camera/IMG_0002.jpg"]
    assert [f.path for f in files] == [f"DCIM/Camera/IMG_{n:04d}.jpg" for n in range(1, len(files) + 1)]
    assert [f.spec["taken"] for f in files] == sorted(f.spec["taken"] for f in files)
    assert not any(STAMP.search(f.path) for f in files), "a name must not say the time as YYYYMMDD_HHMMSS"
    assert all(f.t == datetime.fromisoformat(str(f.spec["taken"])).timestamp() and f.own_time for f in files)
    assert {f.spec["src"][:len(S.IMAGES)] for f in files} == {S.IMAGES}


@needs_source
def test_a_question_that_is_asked_is_in_the_real_question_file_unchanged_but_for_its_spaces():
    real = plain_real()
    for seed in range(1, 6):
        for q in plan("standard", seed)[1]:
            if q.tags["template"].removeprefix("absent_") in ("find", "photos", "date", "read", "which", "price", "days"):
                assert q.question == " ".join(real[q.qid.removeprefix("absent-").split("-")[1]]["question"].split())


@needs_source
def test_no_question_names_a_person_a_password_or_a_child():
    banned = re.compile(r"(?i)password|wifi|wi-fi|footage|girlfriend|boyfriend|\bson\b|daughter|baby|child|kid|Andrew|Grace|Rafael|Angela|Hao Sun")
    for seed in range(1, 9):
        for q in plan("standard", seed)[1]:
            assert not banned.search(q.question), q.question
    for template in (Q.READ, Q.WHICH, Q.THINGS):
        assert not banned.search(json.dumps(template))


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "amount": lambda: g.truth, "text": lambda: g.truth.split("|")[0],
            "absent": lambda: "unknown"}[g.rule]()


def wrong(q) -> list[str]:
    """Planted wrong answers: a missing, extra or other file or day; one more; a number out; the other thing; `unknown` on a needle; a
    file or a date for an absent one."""
    g = q.grade
    if g.rule == "set" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", g.items[0]):
        other = (date.fromisoformat(g.items[0]) + timedelta(days=1)).isoformat()
        return [other, f"{g.items[0]}, {other}", "unknown"]
    if g.rule == "set":
        out = [", ".join([*g.items, "IMG_9999.jpg"]), "IMG_9999.jpg", "unknown"]
        return out + ([", ".join(g.items[1:])] if len(g.items) > 1 else [])
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "amount":
        return [str(float(g.truth) * 2 + 1), "unknown"]
    if g.rule == "text":
        return ["unknown", *([f"{w} and {g.truth.split('|')[0]}" for w in g.wrong][:1]), *([f"{g.truth.split('|')[0]} or {w}" for w in g.wrong][:1])]
    return ["2022-06-20", "IMG_0001.jpg"]


@needs_source
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        for seed in (1, 2, 3):
            for q in plan(scale, seed)[1]:
                assert grade.grade(q.grade, right(q)), (q.qid, right(q))
                for answer in wrong(q):
                    assert not grade.grade(q.grade, answer), (q.qid, answer)


@needs_source
def test_an_answer_may_say_a_date_or_a_file_as_a_person_does():
    qs = {str(q.tags["template"]): q for q in plan("standard", 1)[1]}
    date_q = qs["date"]
    day = date.fromisoformat(date_q.grade.items[0])
    month = day.strftime("%B")
    assert grade.grade(date_q.grade, f"{day.day} {month} {day.year}") and grade.grade(date_q.grade, f"{month} {day.day}, {day.year}")
    photos = qs["photos"]
    assert grade.grade(photos.grade, "I found them: " + " and ".join(f"DCIM/Camera/{n}" for n in photos.grade.items))


# -- what builds ---------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i1")
    recipe = I1.recipe(scale="tiny", seed=1)
    dump, questions = I1.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@needs_source
def test_a_photo_is_copied_with_its_time_alone_its_gps_gone_and_its_image_data_untouched(built):
    out, recipe, dump, _ = built
    files, _ = I1.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] and len(dump.entries) == len(files)
    for n in (0, 77, len(files) - 1):
        e, f = dump.entries[n], files[n]
        path = out / "dump" / e.path
        wall = plain_when(str(f.spec["src"]).rsplit("/", 1)[-1])
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == wall == datetime.fromtimestamp(e.t) and e.own_time
        exif = Image.open(path).getexif()
        assert set(exif) <= {0x0112, 0x0132, 0x8769} and exif[0x0132] == exif.get_ifd(0x8769)[0x9003] == wall.strftime("%Y:%m:%d %H:%M:%S")
        assert not exif.get_ifd(0x8825), "no GPS"
        source = (S.SOURCE / str(f.spec["src"])).read_bytes()
        copy = path.read_bytes()
        assert copy.endswith(source[source.rindex(b"\xff\xda"):]) and len(copy) < len(source) + 200
    for e in dump.entries:
        header = realdump.jpeg_header(out / "dump" / e.path)
        assert b"GPS" not in header and b"Make" not in header and b"http://ns.adobe.com" not in header
    assert I1.probe(recipe, dump, out, None, 1).status == "passed"


@needs_source
def test_the_probe_fails_a_photo_whose_name_says_a_thing(built, tmp_path):
    out, recipe, dump, _ = built
    first = dump.entries[0]
    entries = (first.__class__(**{**first.__dict__, "path": "DCIM/Camera/concorde.jpg"}), *dump.entries[1:])
    renamed = type(dump)(dump.key, dump.root, dump.scenario, entries, ())
    result = I1.probe(recipe, renamed, tmp_path, None, 1)
    assert result.status == "failed" and any("concorde" in f for f in result.failures)


@needs_source
def test_no_command_may_reach_the_source_or_the_code():
    deny = I1.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:atm-bench" in deny and "in:i1_camera_roll" in deny and "in:from mmsb" in deny


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
    world.patch.setitem(scenarios.REGISTRY, "i1_camera_roll", "mmsb.scenarios.i1_camera_roll")
    assert world.main("build", "--scenario", "i1_camera_roll", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    prompt = spy.prompts["files", "find-3020b8"]
    assert prompt.startswith("It is now Wednesday 22 June 2022, 09:00.")
    assert "The files of my phone's camera roll are in " in prompt and "DCIM/Camera holds its photos, IMG_0001.jpg onwards" in prompt
    assert "the names of the files, as in IMG_0042.jpg" in prompt and "You have 900 seconds: " in prompt
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
