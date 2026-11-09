"""I8, the course pages, from its real source on disk: the tiny folder built (pages rendered, named and timed), every answer derived again here by a
plain reader of the PDFs (never the scenario's: a second extraction of each page's text, its own folding and its own search), the held-out terms on no
page that is left, the grader taking each truth and refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and `report` over it with a
stand-in agent; and the choice of pages, terms and held-out pages over made-up ones. Skips, and says so, without the source or `pypdfium2`."""

from __future__ import annotations

import functools
import io
import json
import re
import threading
import unicodedata
from collections import defaultdict
from datetime import datetime

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

pdfium = pytest.importorskip("pypdfium2", reason="pypdfium2 reads the PDFs of I8: `uv run --with pypdfium2`")
from mmsb.scenarios import i8_course_pages as I8  # noqa: E402
from mmsb.scenarios.i8_course_pages import pile as piling  # noqa: E402
from mmsb.scenarios.i8_course_pages import questions as asking  # noqa: E402
from mmsb.scenarios.i8_course_pages import source, terms  # noqa: E402

if not (source.SOURCE / "5-111sc-principles-of-chemical-science-fall-2014").is_dir():
    pytest.skip(f"the OpenCourseWare pages are not at {source.SOURCE}, so I8 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("i8")
ASKS = {"find": re.compile(r"Find the page that mentions (.+)\.$"), "all": re.compile(r"Show me every page that mentions (.+)\.$"),
        "count": re.compile(r"How many pages mention (.+)\?$"), "which": re.compile(r"In which of my courses do the notes mention (.+)\?$"),
        "order": re.compile(r"In my (\w+) notes, is (.+) before or after (.+)\?$"),
        "did": re.compile(r"Did my (\w+) notes ever get to (.+)\? If so, on which page\?$")}


def fold(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())) + " "


@functools.cache
def second_text(path: str, number: int) -> tuple[str, str]:
    """A page's text read here, with the folding and the hyphen reading of its own (the scenario's are in terms.py): a word hyphenated at the end of
    a line joined and not."""
    doc = pdfium.PdfDocument(path)
    raw = doc[number].get_textpage().get_text_range()
    doc.close()
    return fold(re.sub(r"-\s*\n\s*", "", raw.replace("\r\n", "\n"))), fold(re.sub(r"-\s*\n\s*", " ", raw.replace("\r\n", "\n")))


def has(texts: tuple[str, str], phrase: str) -> bool:
    """Whether the folded text holds the phrase as whole words or those with a plural s."""
    wanted = fold(phrase)
    return any(wanted in t or wanted[:-1] + "s " in t or wanted[:-1] + "es " in t for t in texts)


def pdf_of(f) -> str:
    return str(source.pdf_path(source.SOURCE, f.spec["course"], f.spec["pdf"]))


def plan(scale: str, seed: int):
    return _plan(scale, seed)


@functools.cache
def _plan(scale: str, seed: int):
    return I8.plan_of(I8.recipe(scale=scale, seed=seed))


def term_of(said: str, course: str | None = None) -> terms.Term:
    (found,) = {t for t in (*terms.TERMS, *terms.NEVER) if t.said == said and (course is None or t.course == course)}
    return found


def check_against_pdfs(files, questions) -> None:
    """Each question's truth and evidence as the plain reader finds them in the PDFs."""
    name = {f.path: f.path.rsplit("/", 1)[-1] for f in files}
    texts = {f.path: second_text(pdf_of(f), f.spec["page"]) for f in files}
    here = {(f.spec["course"], f.spec["pdf"], f.spec["page"]) for f in files}
    for q in questions:
        template = q.tags["template"].removeprefix("absent_")
        absent = q.kind == "absent"
        assert (q.tags["template"] != template) is absent, q.qid
        if template == "read":
            read = next(r for r in asking.READS if f"read-{r.key}" == q.qid)
            found = sorted(p for p, t in texts.items() if has(t, read.check))
            assert (read.course, read.pdf, read.number) in here and found and sorted(s.file for s in q.truth.evidence) == found, q.qid
            assert q.grade.rule == "text" and q.grade.truth == read.accepted, q.qid
            continue
        match = ASKS[template].search(q.question)
        assert match, f"{q.qid}: {q.question}"
        if template == "order":
            a, b = term_of(match[2], match[1]), term_of(match[3], match[1])
            on_a, on_b = [p for p, t in texts.items() if has(t, a.search)], [p for p, t in texts.items() if has(t, b.search)]
            assert len(on_a) == len(on_b) == 1 and on_a != on_b and not absent, q.qid
            first = sorted(name[p] for p in (on_a[0], on_b[0]))[0] == name[on_a[0]]  # the files are named in the order they were saved
            assert q.grade.truth == ("before" if first else "after") and q.grade.wrong == (("after",) if first else ("before",)), q.qid
            continue
        t = term_of(match[2] if template == "did" else match[1], match[1] if template == "did" else None)
        found = sorted(p for p, x in texts.items() if has(x, t.search))
        if absent:
            assert not found and q.grade.rule == "absent" and not q.truth.evidence, f"{q.qid}: still on {found[:2]}"
            continue
        assert sorted(s.file for s in q.truth.evidence) == found, q.qid
        if template in ("find", "did"):
            assert len(found) == 1 and q.grade.items == (name[found[0]],), q.qid
        elif template == "all":
            assert 2 <= len(found) <= 40 and set(q.grade.items) == {name[p] for p in found}, q.qid
        elif template == "count":
            assert 3 <= len(found) <= 60 and q.grade.truth == str(len(found)), q.qid
        else:
            courses = {next(f.spec["course"] for f in files if f.path == p) for p in found}
            assert len(courses) == 1 and q.truth.value in courses and q.grade.truth == dict((c.key, c.accepted) for c in source.COURSES)[q.truth.value], q.qid
            assert t.search in terms.ANYWHERE, f"{q.qid}: the word names its course"
            assert set(q.grade.wrong) == {p for c in source.COURSES if c.key != q.truth.value for p in c.accepted.split("|")}, q.qid


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "text": lambda: g.truth.split("|")[0], "absent": lambda: "unknown"}[g.rule]()


def wrong(q, other_names) -> list[str]:
    g = q.grade
    if g.rule == "set":
        extra = next(n for n in other_names if n not in g.items)
        out = [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), "unknown", "none"]
        return out + ([", ".join(g.items[1:])] if len(g.items) > 1 else [])
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", "unknown"]
    if g.rule == "text":
        return ["unknown", "I cannot tell"] + (["after" if g.truth == "before" else "before"] if g.truth in ("before", "after") else ["maybe pottery"])
    return [", ".join(other_names[:2]), other_names[0], "3", "Yes, there are some."]


@pytest.fixture(scope="module")
def tiny():
    return plan("tiny", 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i8")
    recipe = I8.recipe(scale="tiny", seed=1)
    dump, questions = I8.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_folder_is_four_lectures_with_a_quarter_of_its_questions_absent(tiny):
    files, questions = tiny
    assert 35 <= len(files) <= 70 and len(questions) == 9 and sum(q.kind == "absent" for q in questions) == 2
    assert {q.tags["template"] for q in questions} == {"find", "all", "count", "which", "read", "absent_find", "absent_count"}
    assert {(f.spec["course"], f.spec["pdf"]) for f in files} == {(c, p) for c, p in piling.TINY_PDFS}
    assert {q.split for q in questions} == {"dev", "test"}


@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3)), ("standard", (1, 2, 3))])
def test_every_answer_is_the_pdfs(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        check_against_pdfs(files, questions)
        if scale == "standard":
            assert 400 < len(files) < 500 and len(questions) == 23 and sum(q.kind == "absent" for q in questions) == 5, seed
            assert {q.tags["template"] for q in questions} == {"find", "all", "count", "which", "order", "did", "read", "absent_find", "absent_all",
                                                              "absent_count", "absent_did"}
            assert 6 <= sum(q.split == "dev" for q in questions) <= 8


def test_the_folder_is_the_same_at_every_seed_and_the_questions_differ():
    files_1, one = plan("standard", 1)
    files_2, two = plan("standard", 2)
    assert files_1 == files_2, "a seed draws questions from one folder"
    assert {q.qid for q in one} != {q.qid for q in two}
    assert {q.qid for q in one if q.tags["template"] == "read"} == {q.qid for q in two if q.tags["template"] == "read"}, "written, not drawn"


def test_the_pages_are_saved_a_lecture_at_a_time_in_order_at_the_hour_of_its_course():
    files, _ = plan("standard", 1)
    assert [f.path for f in files] == [f"Notes/page_{n:04d}.jpg" for n in range(1, len(files) + 1)] and [f.t for f in files] == sorted(f.t for f in files)
    runs = []
    for f in files:
        if not runs or runs[-1][0] != (f.spec["course"], f.spec["pdf"]):
            runs.append([(f.spec["course"], f.spec["pdf"]), []])
        runs[-1][1].append(f)
    assert len(runs) == len({r[0] for r in runs}), "one run of files for each PDF: no two lectures overlap"
    hour = {"chemistry": 10, "algorithms": (11, 16), "biology": 13, "psychology": 14}
    for (course, pdf), run in runs:
        pages = [f.spec["page"] for f in run]
        assert pages == sorted(pages) and all(3 <= b.t - a.t <= 60 for a, b in zip(run, run[1:])), (course, pdf)
        start = datetime.fromtimestamp(run[0].t)
        assert start.hour in (hour[course] if isinstance(hour[course], tuple) else (hour[course],)) and start.date().isoformat() < "2026-04-17", (course, pdf, start)
    first = {c: min(datetime.fromtimestamp(f.t) for f in files if f.spec["course"] == c) for c in hour}
    assert first["chemistry"].weekday() == 0 and first["algorithms"].weekday() == 1 and first["biology"].weekday() == 2


def test_no_page_has_a_figure_over_a_tenth_of_it_or_next_to_no_text_and_no_held_out_term_is_left():
    files, _ = plan("standard", 1)
    for f in list(files)[::9]:
        doc = pdfium.PdfDocument(pdf_of(f))
        page = doc[f.spec["page"]]
        area = page.get_size()[0] * page.get_size()[1]
        for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
            left, bottom, right, top = obj.get_bounds()
            assert (right - left) * (top - bottom) <= 0.10 * area, f.path
        assert len(page.get_textpage().get_text_range()) >= 40, f.path
        doc.close()
    held = piling.choose(source.read_pages(str(source.SOURCE)), "standard", [(r.course, r.pdf, r.number) for r in asking.READS]).held
    assert len(held) == 6 and len({p.pdf + str(p.number) for _, p in held}) == 6
    texts = {f.path: second_text(pdf_of(f), f.spec["page"]) for f in files}
    for t, p in held:
        assert not [path for path, x in texts.items() if has(x, t.search)], f"{t.search} is still on a page"
        assert (p.course, p.pdf, p.number) not in {(f.spec["course"], f.spec["pdf"], f.spec["page"]) for f in files}


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale, seed in (("tiny", 1), ("standard", 1), ("standard", 2)):
        files, questions = plan(scale, seed)
        names = [f.path.rsplit("/", 1)[-1] for f in files]
        for q in questions:
            assert grade.grade(q.grade, right(q)), (q.qid, right(q))
            for answer in wrong(q, names):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


@pytest.mark.parametrize("key, right_answers, wrong_answers", [
    ("chemistry-date", ["September 5", "September 5th", "Sept. 5", "Sep 5th", "5 September", "the 5th of September", "Friday, September 5, 2014", "5th Sept 2014"],
     ["September 3", "September 6", "September 5 or 6", "September 5, 2013", "maybe September 5", "unknown"]),
    ("biology-lecture", ["9/12", "September 12", "Sept. 12th", "12 September 2018", "Sep 12, 2018"], ["9/11", "Lecture 4 of 9/12 or so", "September 13", "September 12 or 13", "perhaps 9/12"]),
    ("psychology-colour", ["0.09 seconds", ".09 seconds", "0.09 s", "0.09", "90 ms", "90 milliseconds"], ["0.29 seconds", "0.20", "0.09 or 0.2", "9 seconds", "unknown"]),
    ("psychology-applications", ["2200", "2,200", "2200 applications", "about 2,200"], ["2000", "150", "2200 or 2300", "possibly 2200"]),
    ("algorithms-array", ["[8, 2, 4, 9, 3]", "8 2 4 9 3", "The array [8, 2, 4, 9, 3]"], ["[2, 3, 4, 8, 9]", "[8, 2, 4, 9]", "unknown"])])
def test_a_value_read_off_a_page_takes_the_forms_a_person_writes_and_refuses_a_second_guess(key, right_answers, wrong_answers):
    (q,) = [q for q in plan("standard", 1)[1] if q.qid == f"read-{key}"]
    for answer in right_answers:
        assert grade.grade(q.grade, answer), answer
    for answer in wrong_answers:
        assert not grade.grade(q.grade, answer), answer


def test_which_course_takes_the_course_by_its_name_or_number_and_refuses_any_answer_that_names_another():
    _, questions = plan("standard", 1)
    which = [q for q in questions if q.tags["template"] == "which"]
    assert len(which) == 2
    names = {c.key: c for c in source.COURSES}
    for q in which:
        right_course = q.truth.value
        number = names[right_course].accepted.split("|")[-1].replace(" ", ".")
        assert grade.grade(q.grade, right_course) and grade.grade(q.grade, f"It is {right_course.title()} ({number})."), q.qid
        others = [c for c in names if c != right_course]
        for answer in ("unknown", f"{others[0]}", f"{right_course} or {others[1]}", f"{right_course} and {others[2]}", f"maybe {right_course}"):
            assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_page_is_rendered_as_a_jpeg_of_its_pdf_pages_size_with_no_metadata(built):
    out, recipe, dump, _ = built
    files, _ = plan("tiny", 1)
    assert [e.path for e in dump.entries] == [f.path for f in files]
    for e, f in list(zip(dump.entries, files))[::6]:
        data = (out / "dump" / e.path).read_bytes()
        image = Image.open(io.BytesIO(data))
        assert image.format == "JPEG" and not image.getexif() and data[:2] == b"\xff\xd8"
        doc = pdfium.PdfDocument(pdf_of(f))
        width, height = doc[f.spec["page"]].get_size()
        doc.close()
        assert abs(image.size[0] - width * source.SCALE) <= 1 and abs(image.size[1] - height * source.SCALE) <= 1, e.path
        assert abs((out / "dump" / e.path).stat().st_mtime - f.t) < 1e-3 and e.t == f.t
    assert I8.probe(recipe, dump, out, None, 1).status == "passed"


def test_a_name_that_says_a_term_fails_the_probe(built, tmp_path):
    out, recipe, dump, _ = built
    raw = (out / "dump" / dump.entries[0].path).read_bytes()
    folder = tmp_path / "dump"
    entries = realdump.write([realdump.File("Notes/helicase page.jpg", "notes", 1.0, "page", "image", {})], folder, 1, lambda f, p: p.write_bytes(raw))
    outcome = I8.probe(recipe, realdump.dump_of("k", folder, "i8_course_pages", entries), tmp_path, None, 1)
    assert outcome.status == "failed" and "helicase" in outcome.failures[0]


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = I8.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I8.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I8.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I8.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = I8.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:ocw" in deny and "in:i8_course_pages" in deny and "in:from mmsb" in deny


# -- the choice of pages, terms and held-out pages, over made-up ones --------------------------------------------------------------------------

def page(course: str, pdf: str, number: int, text: str, big: bool = False) -> source.Page:
    return source.Page(course, pdf, number, text, big)


def test_a_page_is_usable_only_with_text_and_no_big_picture_and_no_citation_page():
    body = "Mitosis is the division of a cell into two daughter cells with the same chromosomes."
    assert page("biology", "a.pdf", 0, body).usable
    assert not page("biology", "a.pdf", 0, "Figure 3").usable, "next to no text"
    assert not page("biology", "a.pdf", 0, body, big=True).usable, "a picture over a tenth of it"
    assert not page("biology", "a.pdf", 1, "MIT OpenCourseWare http://ocw.mit.edu 7.016 Introductory Biology Fall 2018 For information about citing").usable
    assert page("biology", "a.pdf", 1, "MIT OpenCourseWare. " + body * 6).usable, "a long page that names OCW is notes"


@pytest.mark.parametrize("text, search, found", [
    ("The photo-\nelectric effect was explained by Einstein.", "photoelectric effect", True),
    ("long-\r\nterm memory", "long-term memory", True),
    ("A hash table is used; hash tables are fast.", "hash table", True),
    ("The Hash-Table class", "hash table", True),
    ("hash tabletop", "hash table", False),
    ("a hash\ntable", "hash table", True),
    ("The ﬁrst ﬂow", "first flow", True),
    ("A catalyst", "cat", False),
    ("heaps and heapsort", "heap", True),
    ("heapsort", "heap", False)])
def test_a_term_is_found_as_whole_words_with_a_hyphen_at_a_line_end_read_either_way(text, search, found):
    assert terms.mentions(page("algorithms", "a.pdf", 0, unicodedata.normalize("NFKC", text)), search) is found


def test_the_held_out_page_of_a_term_is_its_only_page_and_none_a_written_question_reads():
    body = lambda s: f"{s} " + "plain words of a lecture page " * 3
    pages = {"chemistry": [page("chemistry", "MIT5_111F14_Lec2.pdf", n, body(t)) for n, t in enumerate(
        ["the Bohr model", "Bohr again and Planck", "the photoelectric effect", "orbital shapes", "nothing special"])]}
    pile = piling.choose(pages, "tiny", [("chemistry", "MIT5_111F14_Lec2.pdf", 0)])
    held = {t.search for t, _ in pile.held}
    assert held <= {"photoelectric effect", "orbital", "Planck"}, "a term on one page, and not the one a written question reads (Bohr is on two as well)"
    assert len(pile.pages) == 4 and all(not terms.mentions(p, t.search) for t, _ in pile.held for p in pile.pages)


# -- the whole, run -----------------------------------------------------------------------------------------------------------------------

class Spy:
    """An agent that answers right in the `files` configuration, citing a page of the truth, and `unknown` in any other."""

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


def test_build_run_and_report_over_the_tiny_folder(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "i8_course_pages", "mmsb.scenarios.i8_course_pages")
    assert world.main("build", "--scenario", "i8_course_pages", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    prompt = next(p for (config, qid), p in spy.prompts.items() if config == "files" and qid.startswith("find-"))
    assert prompt.startswith("It is now ") and " The files of my folder of saved pages of lecture notes and slides are in " in prompt and "Notes holds my saved pages" in prompt
    assert "<the name of the file, as in page_0042.jpg>" in prompt and "You have 900 seconds: " in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    (section,) = fs.read_json(folder / "report.json")["scenarios"]
    files_cell, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files_cell["correct"], files_cell["n"], files_cell["grounding"]["grounded"]) == (9, 9, 9)
    assert (listing["correct"], listing["n"]) == (2, 9) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 9}
