"""I9, the nature photos, from its real source on disk: the tiny pile built (photos numbered in the order taken, EXIF holding the local time and
nothing else), every answer derived again here by a plain reader of the observations file (never the scenario's), the absent species checked
in no photo of the pile and, for the ones never seen, in no genus of the observations, the grader taking each truth and refusing planted wrong
answers, the leak probe, and `mmsb build`, `run` and `report` over it with a stand-in agent. Skips, and says so, without the source."""

from __future__ import annotations

import io
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
from mmsb.scenarios import i9_nature as I9
from mmsb.scenarios import realdump
from mmsb.scenarios.i9_nature import questions as Q
from mmsb.scenarios.i9_nature import source as S

pytestmark = pytest.mark.xdist_group("i9")
REAL = (S.SOURCE / S.OBSERVATIONS).is_file() and (S.SOURCE / S.PHOTOS).is_dir()
needs_source = pytest.mark.skipif(not REAL, reason=f"the iNaturalist source is not at {S.SOURCE}, so I9 is not built")


# -- what needs no source -----------------------------------------------------------------------------------------------------------------

def test_a_species_is_said_with_the_article_a_person_gives_it():
    assert [Q.a_(n) for n in ("Wild Turkey", "American elm", "European Starling", "Eastern Chipmunk", "ostrich fern", "unicorn root")] == [
        "a Wild Turkey", "an American elm", "a European Starling", "an Eastern Chipmunk", "an ostrich fern", "a unicorn root"]


def test_a_name_is_accepted_whole_or_without_its_first_word_unless_the_other_name_shares_it():
    assert Q.accepted("American elm", "American chestnut") == "american elm|elm"
    assert Q.accepted("American Royal Fern", "American white waterlily") == "american royal fern|royal fern"
    assert Q.accepted("Butternut", "red trillium") == "butternut", "one word"
    assert Q.accepted("red trillium", "large white trillium") == "red trillium", "its last word is the other's"
    assert Q.accepted("Wild Turkey", "Monarch") == "wild turkey|turkey"


def test_an_answer_with_the_wrong_name_or_both_is_wrong_and_the_tail_of_a_name_is_right():
    g = realdump.grade("text", truth=Q.accepted("American elm", "American chestnut"),
                       wrong=Q.accepted("American chestnut", "American elm").split("|"))
    assert grade.grade(g, "American elm") and grade.grade(g, "the elm, first") and grade.grade(g, "american Elm.")
    assert not grade.grade(g, "American chestnut") and not grade.grade(g, "unknown")
    assert grade.grade(g, "American elm and American chestnut"), "an explanation that names the answer first"
    assert not grade.grade(g, "American chestnut and American elm") and not grade.grade(g, "American elm or American chestnut")


def test_a_recipe_is_refused_in_another_time_zone_with_other_keys_and_at_another_scale():
    recipe = I9.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I9.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I9.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I9.recipe(scale="huge", seed=1)


def test_every_template_is_filed_and_six_categories_are_used():
    taxonomy.check(I9.shelf, I9.filing)
    assert I9.shelf.media == "image" and I9.shelf.area == "fun" and I9.shelf.name == "Nature photos"
    templates = {t for t in I9.filing if not t.startswith("absent_")}
    assert templates == set(Q.WORDING) and {I9.filing[t][0] for t in templates} == {"find_all", "how_many", "when", "find", "did", "order"}
    assert all(I9.filing["absent_" + t] == I9.filing[t] for t in templates)


def test_a_jpeg_is_stamped_with_its_local_time_alone_and_its_image_data_untouched():
    img = Image.new("RGB", (8, 6), (30, 160, 40))
    exif = Image.Exif()
    exif[0x010F] = "Maker"
    exif[0x0112] = 6
    exif.get_ifd(0x8825)[2] = (44.0, 3.0, 2.0)
    out = io.BytesIO()
    img.save(out, "JPEG", exif=exif, comment=b"a comment")
    source = out.getvalue()
    stamped = S.stamped(source, datetime(2023, 5, 17, 13, 5, 9))
    with Image.open(io.BytesIO(stamped)) as back:
        e = back.getexif()
        assert dict(e) == {0x0112: 6, 0x0132: "2023:05:17 13:05:09", 0x8769: e[0x8769]} and not e.get_ifd(0x8825)
        assert dict(e.get_ifd(0x8769)) == {0x9003: "2023:05:17 13:05:09", 0x9004: "2023:05:17 13:05:09"}
    assert b"Maker" not in stamped and b"comment" not in stamped and stamped.endswith(source[source.rindex(b"\xff\xda"):])
    with pytest.raises(ValueError, match="not a JPEG"):
        S.stamped(b"\x89PNG", datetime(2023, 5, 17))


def test_the_metadata_the_probe_reads_is_the_application_segments_alone(tmp_path):
    img = Image.new("RGB", (16, 16), (30, 160, 40))
    out = io.BytesIO()
    img.save(out, "JPEG", comment=b"a comment")
    path = tmp_path / "a.jpg"
    path.write_bytes(S.stamped(out.getvalue(), datetime(2023, 5, 17, 13, 5, 9)))
    text = S.app_segments(path)
    assert b"JFIF" in text and b"Exif" in text and b"2023:05:17 13:05:09" in text and b"comment" not in text
    assert len(text) < 300, "no quantisation or Huffman table"


# -- what reads the source ----------------------------------------------------------------------------------------------------------------

def plain_rows() -> list[dict]:
    with open(S.SOURCE / S.OBSERVATIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def species_of(row: dict) -> str:
    return " ".join(row["taxon_name"].split()[:2])


def plain_names() -> dict[str, str]:
    return {species_of(r): r["taxon_common_name"] for r in plain_rows() if r["taxon_rank"] == "species" and r["taxon_common_name"]}


def plan(scale: str, seed: int):
    return I9.plan_of(I9.recipe(scale=scale, seed=seed))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def check_against_labels(files, questions, scale: str, seed: int) -> None:
    """Each question's template, species and truth as the plain reader of the observations finds them."""
    rows = {f"{r['photo_id']}.{r['photo_ext']}": r for r in plain_rows()}
    names = plain_names()
    by_name = {n.casefold(): s for s, n in names.items()}
    in_dump = defaultdict(list)  # species -> [(day, dump file name)]
    for f in files:
        row = rows[f.spec["src"].rsplit("/", 1)[-1]]
        assert row["time_observed_at"][:19] == f.spec["taken"], "the local time of the observation"
        in_dump[species_of(row)].append((row["observed_on"], f.path.rsplit("/", 1)[-1]))
    days_all = sorted({d for v in in_dump.values() for d, _ in v})
    genera = {r["taxon_name"].split()[0] for r in plain_rows()}
    first_day, last_day = days_all[0], days_all[-1]
    asked = set()
    for q in questions:
        template, absent = str(q.tags["template"]).removeprefix("absent_"), q.kind == "absent"
        m = re.fullmatch(r"(?:absent-)?(days|count|first|last|find|ever|which)-.+", q.qid)
        assert m and m[1] == template, q.qid
        if template == "which":
            shown = re.fullmatch(r"Which did I photograph first, an? (.+) or an? (.+)\?", q.question)
            assert shown, q.question
            species = [by_name[n.casefold()] for n in shown.groups()]
        else:
            shown = re.fullmatch(r"(?:.*?photograph|Find my photo of) an? (.+?)(?:\?|\.)(?: If I did, when was the first time\?)?", q.question)
            assert shown, q.question
            species = [by_name.get(shown[1].casefold(), shown[1])]
        asked |= set(species)
        if absent:
            assert q.grade.rule == "absent" and not q.truth.evidence
            assert not [s for s in species if s in in_dump], f"{q.qid}: a photo of it is in the pile"
            assert template != "which"
            only = species[0]
            if only in by_name.values():  # a held-out species: in the observations, the only one of its genus, and not in the pile
                assert sum(1 for s in names if s.split()[0] == only.split()[0]) == 1 and not any(
                    r["taxon_name"].split()[0] == only.split()[0] and species_of(r) != only for r in plain_rows())
            else:  # never seen: no taxon of the observations is of the species' genus
                genus = Q.NEVER[only]
                assert genus not in genera and only.casefold() not in {n.casefold() for n in names.values()}, q.qid
            continue
        a = species[0]
        v = sorted(in_dump[a])
        days = sorted({d for d, _ in v})
        assert names[a] in q.question and Q.MIN_OBS[scale] <= len(v) <= Q.MAX_OBS or template == "find"
        assert not re.search(r"sedge|grass|rush|canker", names[a], re.I)
        if template == "days":
            assert q.grade.rule == "set" and q.grade.items == tuple(days) and 3 <= len(days) <= 12, q.qid
            assert sorted(s.file.rsplit("/", 1)[-1] for s in q.truth.evidence) == sorted(n for _, n in v)
        elif template == "count":
            assert q.grade.rule == "count" and q.grade.truth == str(len(days)), q.qid
        elif template in ("first", "ever"):
            assert q.grade.items == (days[0],) and days[0] not in (first_day, last_day), q.qid
            assert sorted(s.file.rsplit("/", 1)[-1] for s in q.truth.evidence) == sorted(n for d, n in v if d == days[0])
        elif template == "last":
            assert q.grade.items == (days[-1],) and days[-1] not in (first_day, last_day), q.qid
        elif template == "find":
            assert len(v) == 1 and q.grade.items == (v[0][1],) and names[a] in Q.FAMILIAR, q.qid
        else:
            b = species[1]
            da, db = min(d for d, _ in in_dump[a]), min(d for d, _ in in_dump[b])
            assert (date.fromisoformat(db) - date.fromisoformat(da)).days >= Q.FORTNIGHT, q.qid
            assert q.grade.truth.split("|")[0] == names[a].casefold() and names[b].casefold() in q.grade.wrong
            assert names[a].casefold() not in names[b].casefold() and names[b].casefold() not in names[a].casefold()
    assert len(asked) >= len(questions), "no species is asked of twice"
    assert len(questions) == (6 if scale == "tiny" else 24) and sum(q.kind == "absent" for q in questions) == (2 if scale == "tiny" else 6), seed


@needs_source
@pytest.mark.parametrize("scale, seeds", [("tiny", range(1, 6)), ("standard", range(1, 6))])
def test_every_answer_is_the_labels(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        check_against_labels(files, questions, scale, seed)


@needs_source
def test_the_standard_pile_is_the_photos_of_the_species_and_the_tiny_pile_is_six_weeks():
    rows = plain_rows()
    files, questions = plan("standard", 1)
    jpeg = [r for r in rows if r["taxon_rank"] in S.SPECIES_RANKS and r["photo_ext"] in S.JPEG_EXTS]
    assert len(rows) == 3240 and len(jpeg) == 3206 and 3100 < len(files) <= len(jpeg)
    assert [f.path for f in files] == [f"DCIM/Camera/IMG_{n:04d}.jpg" for n in range(1, len(files) + 1)]
    assert [f.spec["taken"] for f in files] == sorted(f.spec["taken"] for f in files)
    assert all(f.t == datetime.fromisoformat(str(f.spec["taken"])).timestamp() and f.own_time for f in files)
    assert not any(re.search(r"\d{8}_\d{6}", f.path) for f in files)
    tiny, tq = plan("tiny", 1)
    assert 150 < len(tiny) < 300 and {f.spec["taken"][:7] for f in tiny} == {"2023-05", "2023-06"}
    assert sum(q.split == "dev" for q in questions) == 7 and sum(q.split == "dev" for q in tq) == 2
    day_after = (datetime.fromisoformat(files[-1].spec["taken"]) + timedelta(days=1)).date().isoformat()
    assert all(q.now.startswith(f"{day_after}T09:00") for q in questions), "asked the morning after the last photo"


@needs_source
def test_another_seed_holds_out_other_species_and_asks_other_questions():
    files, one = plan("standard", 1)
    for seed in (2, 3):
        other_files, other = plan("standard", seed)
        assert {q.qid for q in one} != {q.qid for q in other} and {f.spec["src"] for f in files} != {f.spec["src"] for f in other_files}
    held = {n for seed in range(1, 5) for n in {f.spec["src"] for f in plan("standard", 1)[0]} - {f.spec["src"] for f in plan("standard", seed)[0]}}
    assert held, "the held-out species are the seed's"


@needs_source
def test_the_never_seen_species_are_in_no_genus_of_the_observations_and_no_asked_species_is_in_a_genus_held_out_with_others():
    never = I9.never_of(str(S.SOURCE))
    assert len(never) >= 15 and all(c in Q.NEVER for c in never)
    genera = {r["taxon_name"].split()[0] for r in plain_rows()}
    assert all(Q.NEVER[c] not in genera for c in never)


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "text": lambda: g.truth.split("|")[0], "absent": lambda: "unknown"}[g.rule]()


def wrong(q) -> list[str]:
    g = q.grade
    if g.rule == "set" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", g.items[0]):
        other = (date.fromisoformat(g.items[0]) + timedelta(days=1)).isoformat()
        return [other, f"{g.items[0]}, {other}", "unknown", *([", ".join(g.items[1:])] if len(g.items) > 1 else [])]
    if g.rule == "set":
        return [", ".join([*g.items, "IMG_9999.jpg"]), "IMG_9999.jpg", "unknown"]
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "text":
        return ["unknown", f"{g.wrong[0]} and {g.truth.split('|')[0]}", f"{g.truth.split('|')[0]} or {g.wrong[0]}"]
    return ["2023-05-20", "IMG_0001.jpg"]


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
    out = tmp_path_factory.mktemp("i9")
    recipe = I9.recipe(scale="tiny", seed=1)
    dump, questions = I9.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@needs_source
def test_a_photo_is_copied_with_its_local_time_alone_and_its_image_data_untouched(built):
    out, recipe, dump, _ = built
    files, _ = I9.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files]
    for n in (0, len(files) // 2, len(files) - 1):
        e, f = dump.entries[n], files[n]
        path = out / "dump" / e.path
        wall = datetime.fromisoformat(str(f.spec["taken"]))
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == wall == datetime.fromtimestamp(e.t) and e.own_time
        exif = Image.open(path).getexif()
        assert set(exif) <= {0x0112, 0x0132, 0x8769} and exif[0x0132] == exif.get_ifd(0x8769)[0x9003] == wall.strftime("%Y:%m:%d %H:%M:%S")
        assert not exif.get_ifd(0x8825), "no GPS"
        source = (S.SOURCE / str(f.spec["src"])).read_bytes()
        copy = path.read_bytes()
        assert copy.endswith(source[source.rindex(b"\xff\xda"):]) and len(copy) < len(source) + 200
    for e in dump.entries:
        header = realdump.jpeg_header(out / "dump" / e.path)
        assert b"GPS" not in header and b"http://ns.adobe.com" not in header and b"Make" not in header
    assert I9.probe(recipe, dump, out, None, 1).status == "passed"


@needs_source
def test_the_probe_fails_a_photo_whose_name_says_a_species(built, tmp_path):
    out, recipe, dump, _ = built
    entry = dump.entries[0]
    renamed = type(dump)(dump.key, dump.root, dump.scenario, (entry.__class__(**{**entry.__dict__, "path": "DCIM/Camera/red trillium.jpg"}),
                                                              *dump.entries[1:]), ())
    result = I9.probe(recipe, renamed, tmp_path, None, 1)
    assert result.status == "failed" and any("trillium" in f for f in result.failures)


@needs_source
def test_no_command_may_reach_the_source_or_the_code():
    deny = I9.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:inaturalist" in deny and "in:i9_nature" in deny and "in:from mmsb" in deny


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
    world.patch.setitem(scenarios.REGISTRY, "i9_nature", "mmsb.scenarios.i9_nature")
    assert world.main("build", "--scenario", "i9_nature", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    qid = next(q.qid for q in run.questions() if q.qid.startswith("days-"))
    prompt = spy.prompts["files", qid]
    assert prompt.startswith("It is now ") and "The files of my nature photos are in " in prompt and "DCIM/Camera holds its photos" in prompt
    assert "<the dates, as YYYY-MM-DD, separated by commas>" in prompt and "You have 900 seconds: " in prompt
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
