"""I8, the course pages: a student's folder of saved pages of their notes and slides, from four courses (chemistry, algorithms, biology, psychology),
a few hundred pages, named page_0001.jpg onwards in the order they were saved. The questions ask for the page that mentions a term, every page that
does, how many, which course's notes mention a term that more than one subject has, which of two terms comes first in a course's notes, what a page said, and of terms the notes never use
(questions.py). The pages are MIT OpenCourseWare's lecture PDFs, rendered with pypdfium2; every answer is from their text layers. The scenario is the
names of a `Scenario`, as functions (DESIGN 5), over the shared builder (realdump).

    recipe = i8_course_pages.recipe(scale="tiny", seed=1)      # the source, the semester's days, this machine's UTC offsets
    dump, questions = i8_course_pages.build(recipe, folder, 3)  # the pages rendered as JPEGs and saved as `Notes/page_0001.jpg`, and the questions

Scales: `tiny`, four lectures (about 60 pages, for the tests and a smoke run); `standard`, the notes of the four courses but for the pages held out
(about 440 pages, 0.1 GB; pile.py). The source gives no time: each PDF is one lecture on a day of the 2026 spring semester at the hour its course meets,
and the pages of it are saved a few seconds apart in order; the courses meet at other hours, so that no two lectures overlap. The same recipe gives the
same folder at every seed: the seed draws other questions from it."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from itertools import count
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.i8_course_pages.pile import Pile, choose, key
from mmsb.scenarios.i8_course_pages.questions import CODE, READS, SOURCE_NAME, questions_of
from mmsb.scenarios.i8_course_pages.source import COURSES, SOURCE, Page, notes_of, read_pages, render
from mmsb.scenarios.i8_course_pages.terms import NEVER, TERMS
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i8_course_pages"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Course notes and slides", "image", "school", "a student's folder of saved pages of lecture notes and slides, page_0001.jpg onwards")
filing = with_absent({"find": ("find", "text"), "all": ("find_all", "text"), "count": ("how_many", "text"), "which": ("who", "text"),
                      "order": ("order", "text"), "did": ("did", "text"), "read": ("what", "text")})

SEMESTER = date(2026, 2, 2)  # a Monday: the first day of the semester the lectures are held on
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "Notes"
WHAT = "my folder of saved pages of lecture notes and slides"
LAYOUT = (f"{FOLDER} holds my saved pages, page_0001.jpg onwards, in the order I saved them (JPEG, one page each, from four of my courses); a file's "
          "modified time is when I saved it.")
COMMAND_WORDS = ("ocw", "opencourseware", "pypdfium2")  # the source's names
STEP_S = (6, 40)  # the seconds between two saved pages of a lecture
DAYS = {"tiny": 28, "standard": 80}  # the days the recipe covers from the start of the semester: every lecture is within them


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with, the days the lectures are
    on and this machine's UTC offsets at its first and last day."""
    if scale not in DAYS:
        raise ValueError(f"{name}'s scale is one of {', '.join(DAYS)}, not {scale!r}")
    days = realdump.span_days(SEMESTER, SEMESTER + timedelta(days=DAYS[scale] - 1))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def meeting_days(weekdays: Sequence[int]):
    """The days from the semester's start on that fall on one of `weekdays` (0 for Monday), one after another."""
    return (d for d in (SEMESTER + timedelta(days=x) for x in count()) if d.weekday() in weekdays)


def lectures(source: Path | str) -> dict[tuple[str, str], datetime]:
    """When each PDF of the notes is a lecture, by `(course, pdf)`: the k-th of its session on the k-th day from the semester's start that its session
    meets on, at its hour."""
    out = {}
    for course in COURSES:
        dates = {id(session): meeting_days(session.days) for session in course.sessions}
        for session, path in notes_of(course, Path(source)):
            out[course.key, path.name.split("_", 1)[-1]] = datetime.combine(next(dates[id(session)]), session.at)
    return out


def layout(pile: Pile, source: Path | str) -> dict[tuple[str, str, int], tuple[float, str]]:
    """The time and the path of each page of the folder, by `(course, pdf, number)`: a lecture's pages a few seconds apart from its start in the
    order of their numbers, the files named in the order of their times."""
    start = lectures(source)
    t: dict[tuple[str, str, int], float] = {}
    for (course, pdf), at in sorted(start.items(), key=lambda kv: kv[1]):
        rng = random.Random(f"i8:layout:{course}:{pdf}")
        now = realdump.local_t(at)
        for page in (p for p in pile.pages if (p.course, p.pdf) == (course, pdf)):
            now += rng.randint(*STEP_S)
            t[key(page)] = float(now)
    if len(t) != len(pile.pages):
        raise ValueError("a page of the folder is of no lecture")
    return {k: (time_, f"{FOLDER}/page_{n:04d}.jpg") for n, (k, time_) in enumerate(sorted(t.items(), key=lambda kv: kv[1]), 1)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    pile = choose(read_pages(str(recipe["source"])), scale, [(r.course, r.pdf, r.number) for r in READS])
    placed = layout(pile, str(recipe["source"]))
    files = tuple(File(placed[key(p)][1], SOURCE_NAME, placed[key(p)][0], "page", "image", {"course": p.course, "pdf": p.pdf, "page": p.number})
                  for p in sorted(pile.pages, key=lambda p: placed[key(p)][0]))
    last = datetime.fromtimestamp(files[-1].t).date()
    if last.isoformat() > recipe["days"][-1]:
        raise ValueError(f"a page is saved on {last}, after the recipe's last day {recipe['days'][-1]}")
    now = realdump.local_t(datetime.combine(last + timedelta(days=1), time(9)))
    return files, questions_of(pile, {k: p for k, (_, p) in placed.items()}, now, {k: t for k, (t, _) in placed.items()}, scale, seed)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the folder into `out` (new or empty): each page rendered as a JPEG, with its time; `workers` at once for the writing. Returns the dump and
    the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))
    pages = [Page(str(f.spec["course"]), str(f.spec["pdf"]), int(f.spec["page"]), "", False) for f in files]
    jpeg = render(source, pages)

    def write(f: File, path: Path) -> None:
        path.write_bytes(jpeg[str(f.spec["course"]), str(f.spec["pdf"]), int(f.spec["page"])])

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no page's name or JPEG header names a term of the notes, a course or the source."""
    labels = [*(t.search for t in (*TERMS, *NEVER)), *(c.key for c in COURSES), *COMMAND_WORDS]
    return realdump.probe(dump, out, labels)


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder itself: nothing is cut (every question is asked after its last page)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
