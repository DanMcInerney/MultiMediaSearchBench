"""The questions of I8, every answer from the pages' text layers (source.py), each asked the morning after the last page was saved. A student asks of a
folder of saved pages of their lecture notes by what the notes were about:

| template | question                                                                 | rule                  |
| `find`   | Find the page that mentions mitosis.   (a term on one page only)          | `set` of one file     |
| `all`    | Show me every page that mentions hash tables. (2 to 40 pages)             | `set` of file names   |
| `count`  | How many pages mention long-term memory? (3 to 60 pages)                  | `count`               |
| `which`  | In which of my courses do the notes mention neurons? (a term of ANYWHERE)  | `text`, the subject   |
| `order`  | In my chemistry notes, is Bohr before or after Planck? (a page each)      | `text`, before/after  |
| `did`    | Did my biology notes ever get to helicase? If so, on which page?          | `set` of one file     |
| `read`   | What was the example array the sorting notes sorted? (written, READS)     | `text`                |
| `absent_*` | `find`, `all` and `count` of a term held out, or of one no page has; `did` of either | `absent`   |

A page mentions a term when its text layer has the term's words as whole words (terms.py). A term is asked about by the count of pages that have it in
the folder, so that every answer is certain by the text layers. The pages a question may rest on are the usable ones (source.py); the folder holds no
other. The absent questions are held out (a term on one page, which was taken out of the folder: pile.py) or of a term of the course's subject that
the notes never use (`NEVER`). A written question (`READS`) is checked at the plan: its phrase is on the page it was written from. Which questions are
asked is drawn with the seed, a course at a time so that the courses vary."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i8_course_pages.pile import Pile, key
from mmsb.scenarios.i8_course_pages.source import COURSES, Page
from mmsb.scenarios.i8_course_pages.terms import ANYWHERE, NEVER, TERMS, Term, fold, mentions, texts_of

CODE = "I8"
SOURCE_NAME = "notes"  # the dump's source
QUOTAS = {"tiny": {"find": 1, "all": 1, "count": 1, "which": 1, "absent_find": 1, "absent_count": 1},
          "standard": {"find": 3, "all": 2, "count": 2, "which": 2, "order": 2, "did": 2, "absent_find": 1, "absent_all": 1, "absent_count": 1,
                       "absent_did": 2}}
SEVERAL = (2, 40)  # the pages an `all` question lists
COUNTED = (3, 60)  # the pages a `count` question counts
FILE = "<the name of the file, as in page_0042.jpg>"
FILES = "<the names of the files, as in page_0042.jpg, separated by commas>"
NUMBER = "<a number>"
SHORT = "<a short answer>"
BEFORE_AFTER = "<before or after>"
COURSE = "<the course: chemistry, algorithms, biology or psychology>"
SAYING = {c.key: c.accepted for c in COURSES}


@dataclass(frozen=True)
class Read:
    """A question written by reading a page: its text layer has `check` (folded) on the page `number` of the PDF `pdf`. `accepted` are the phrases of
    a right answer, `|` between; it is asked where the page is in the folder."""

    key: str
    text: str
    course: str
    pdf: str
    number: int
    check: str
    accepted: str


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def date_forms(day: int, year: int) -> str:
    """The ways a person writes a day of September (`September 5th`, `5 Sept`, `the 5th of September 2014`, `Sep. 5, 2014`), as the phrases a right
    answer holds, `|` between."""
    out = []
    for month in ("september", "sept", "sep"):
        for d in (str(day), ordinal(day)):
            out += [f"{month} {d}", f"{d} {month}", f"{d} of {month}", f"{month} {d} {year}", f"{d} {month} {year}", f"{d} of {month} {year}"]
    return "|".join(dict.fromkeys(out))


READS = (
    Read("chemistry-date", "What date is on the second lecture summary in my chemistry notes?", "chemistry", "MIT5_111F14_Lec2.pdf", 0,
         "lecture summary 2 friday september 5 2014", date_forms(5, 2014)),
    Read("algorithms-array", "What was the example array the sorting notes sorted?", "algorithms", "MIT6_006S20_lec3.pdf", 1,
         "example 8 2 4 9 3 2 3 4 8 9", "8 2 4 9 3"),
    Read("biology-lecture", "Which lecture date does the third biology recitation summarise?", "biology", "MIT7_016F18rec3.pdf", 0,
         "summary of lecture 4 9 12", f"9 12|9 12 2018|{date_forms(12, 2018)}"),
    Read("psychology-colour", "How long did the psychology notes say it takes to judge a colour?", "psychology", "MIT9_00SCF11_lec02_scires.pdf", 8,
         "time to judge color 0 09 sec", "0 09|09|90 ms|90 msec|90 millisecond|90 milliseconds"),
    Read("psychology-applications", "How many applications did the medical school in my psychology slides look at?", "psychology",
         "MIT9_00SCF11_lec02_scires.pdf", 56, "selected by looking at 2200 applications", "2200|2 200"),
)


def pick(candidates: Sequence, n: int, rng: random.Random, course=lambda c: c.course) -> list:
    """`n` of `candidates`, a course at a time in a shuffled order so that the courses vary; refused when there are fewer."""
    by_course: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_course[course(c)].append(c)
    for of in by_course.values():
        rng.shuffle(of)
    order = sorted(by_course)
    rng.shuffle(order)
    out: list = []
    while len(out) < n and any(by_course.values()):
        for c in order:
            if by_course[c] and len(out) < n:
                out.append(by_course[c].pop())
    if len(out) < n:
        raise ValueError(f"only {len(out)} of the {len(candidates)} candidates can be asked about, and {n} are wanted")
    return out


def questions_of(pile: Pile, paths: Mapping[tuple[str, str, int], str], now: float, times: Mapping[tuple[str, str, int], float], scale: str,
                 seed: int) -> tuple[Question, ...]:
    """The questions of `pile` at `scale`, drawn with `seed`. `paths` and `times` are the dump path and epoch seconds of each page, by `(course, pdf,
    number)`; `now` is epoch seconds."""
    rng = random.Random(f"i8:{seed}:{scale}")
    quota = QUOTAS[scale]
    since, until = min(times.values()), max(times.values()) + 1
    where: dict[Term, list[Page]] = {t: [p for p in pile.pages if mentions(p, t.search)] for t in TERMS}
    file_of = {key(p): paths[key(p)].rsplit("/", 1)[-1] for p in pile.pages}
    out: list[Question] = []
    taken: set[Term] = set()

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[Page], value: str | None) -> None:
        absent = grade.rule == "absent"
        out.append(realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text, kind="absent" if absent else template,
                                     template="absent_" + template if absent else template, modality="image", shape=shape, grade=grade,
                                     evidence=[Span(paths[key(p)], 0.0, 0.0) for p in found], sources=(SOURCE_NAME,), since=since, until=until,
                                     value=value))

    def slug(t: Term) -> str:
        return "-".join(fold(t.search).split())

    def draw(fits, n: int, among: Sequence[Term] = TERMS) -> list[Term]:
        """`n` terms of `among` that `fits` and no question has asked of."""
        chosen = pick([t for t in among if t not in taken and fits(where[t])], n, rng, lambda t: t.course)
        taken.update(chosen)
        return chosen

    for t in draw(lambda w: len(w) == 1, quota.get("find", 0)):
        (page,) = where[t]
        ask(f"find-{slug(t)}", f"Find the page that mentions {t.said}.", "find", FILE, realdump.grade("set", items=[file_of[key(page)]]), [page], file_of[key(page)])
    for t in draw(lambda w: SEVERAL[0] <= len(w) <= SEVERAL[1], quota.get("all", 0)):
        found = [file_of[key(p)] for p in where[t]]
        ask(f"all-{slug(t)}", f"Show me every page that mentions {t.said}.", "all", FILES, realdump.grade("set", items=found), where[t], ", ".join(found))
    for t in draw(lambda w: COUNTED[0] <= len(w) <= COUNTED[1], quota.get("count", 0)):
        ask(f"count-{slug(t)}", f"How many pages mention {t.said}?", "count", NUMBER, realdump.grade("count", truth=str(len(where[t]))), where[t], str(len(where[t])))
    for t in draw(lambda w: bool(w) and len({p.course for p in w}) == 1, quota.get("which", 0), [t for t in TERMS if t.search in ANYWHERE]):
        course = where[t][0].course
        others = tuple(phrase for c in COURSES if c.key != course for phrase in SAYING[c.key].split("|"))  # naming another course, whatever else, is wrong
        ask(f"which-{slug(t)}", f"In which of my courses do the notes mention {t.said}?", "which", COURSE,
            realdump.grade("text", truth=SAYING[course], wrong=others), where[t], course)
    # order: two terms with one page each, in the notes of one course, on different pages
    sole = [t for t in TERMS if len(where[t]) == 1 and t not in taken]
    position = {key(p): n for n, p in enumerate(sorted(pile.pages, key=lambda p: times[key(p)]))}
    pairs = [(a, b) for a in sole for b in sole if a.search < b.search and a.course == b.course and where[a][0].course == a.course
             and key(where[a][0]) != key(where[b][0])]
    for a, b in pick(pairs, quota.get("order", 0), rng, lambda ab: ab[0].course):
        first = position[key(where[a][0])] < position[key(where[b][0])]
        taken.update((a, b))
        ask(f"order-{slug(a)}-{slug(b)}", f"In my {a.course} notes, is {a.said} before or after {b.said}?", "order", BEFORE_AFTER,
            realdump.grade("text", truth="before" if first else "after", wrong=("after" if first else "before",)), [where[a][0], where[b][0]],
            "before" if first else "after")
    for t in draw(lambda w: len(w) == 1, quota.get("did", 0)):
        (page,) = where[t]
        ask(f"did-{slug(t)}", f"Did my {t.course} notes ever get to {t.said}? If so, on which page?", "did", FILE,
            realdump.grade("set", items=[file_of[key(page)]]), [page], file_of[key(page)])
    for read in READS:
        page = next((p for p in pile.pages if (p.course, p.pdf, p.number) == (read.course, read.pdf, read.number)), None)
        if page is None:
            continue
        found = [p for p in pile.pages if any(f" {read.check} " in text for text in texts_of(p))]
        if not found or page not in found:
            raise ValueError(f"{read.key}: {read.check!r} is not on {read.pdf} page {read.number}")
        ask(f"read-{read.key}", read.text, "read", SHORT, realdump.grade("text", truth=read.accepted), found, read.accepted.split("|")[0])
    # absent: a term whose one page was held out, a term of the subject no page has
    never = [t for t in NEVER if not any(mentions(p, t.search) for p in pile.pages)]
    held = [t for t, _ in pile.held if not where[t]]
    if bad := [t.search for t, _ in pile.held if where[t]]:
        raise ValueError(f"{bad[0]} is held out and still on a page")
    quota_absent = {k: quota.get(k, 0) for k in ("absent_find", "absent_all", "absent_count", "absent_did")}
    gone = pick(never, quota_absent["absent_all"] + quota_absent["absent_count"] + quota_absent["absent_did"], rng, lambda t: t.course)
    for t in pick(held, quota_absent["absent_find"], rng, lambda t: t.course):
        ask(f"find-{slug(t)}", f"Find the page that mentions {t.said}.", "find", FILE, realdump.grade("absent"), [], None)
    for t in gone[:quota_absent["absent_all"]]:
        ask(f"all-{slug(t)}", f"Show me every page that mentions {t.said}.", "all", FILES, realdump.grade("absent"), [], None)
    for t in gone[quota_absent["absent_all"]:quota_absent["absent_all"] + quota_absent["absent_count"]]:
        ask(f"count-{slug(t)}", f"How many pages mention {t.said}?", "count", NUMBER, realdump.grade("absent"), [], None)
    for t in gone[quota_absent["absent_all"] + quota_absent["absent_count"]:]:
        ask(f"did-{slug(t)}", f"Did my {t.course} notes ever get to {t.said}? If so, on which page?", "did", FILE, realdump.grade("absent"), [], None)
    if len({q.qid for q in out}) != len(out):
        raise ValueError("two questions have one id")
    return realdump.split(out, seed)
