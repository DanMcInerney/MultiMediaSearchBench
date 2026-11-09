"""The catalogue of the benchmark (DESIGN 5.8): its media by shelf and its questions by category, in Markdown, made from the scenarios
themselves (`mmsb catalogue`). Nothing is written to a dump and no agent or model is run: each scenario's plan is read for its files and
questions. The catalogue shows examples from the `dev` split only, as the `test` split's questions are not for publishing (DESIGN 6.4); the
`--full` file holds every question with its answer, and stays out of the repository."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mmsb import scenarios
from mmsb.core.question import Question, load_questions
from mmsb.core.taxonomy import AREAS, CATEGORIES, CUES, MEDIA, Shelf, filed

SCALE_OF = {"v7_screen": "pilot"}  # a scenario whose largest usual scale has another name than `standard`
BUILT = {"v8_egolife": Path(__file__).parents[1] / "tests" / "oracles" / "recorded" / "egolife-s1" / "converted" / "questions.jsonl"}  # V8's recorded questions: its plan needs the 41 GB download
ORIGIN = {  # each scenario's code and where its media is from; `media/sources.jsonl` has the links and licences
    "V1": "MEVA (scripted activity at a real site)", "V2": "Australian Roads Dashcam Driving", "V3": "SoccerTrack v2 follow-cam clips (a virtual camera)",
    "V4": "AMI Meeting Corpus (role-played)", "V7": "Made by the benchmark", "V8": "EgoLife", "V9": "MIT OpenCourseWare 14.01",
    "V10": "Charades (acted)", "I1": "ATM-Bench (one person's phone)", "I2": "Android in the Wild", "I3": "Caltech Camera Traps",
    "I4": "BD3 building defects", "I7": "ICDAR 2019 SROIE, CORD v2", "I8": "MIT OpenCourseWare course PDFs", "I9": "iNaturalist (one observer)",
    "A1": "SINS database", "A3": "ICSI Meeting Corpus", "A4": "BirdVox-full-night", "A6": "MIT OpenCourseWare 6.0002", "A7": "Chalk Radio (MIT OpenCourseWare)",
    "A8": "Free Music Archive (fma_small)", "M3": "FTC robocall recordings (machine transcripts)", "M4": "FSD50K"}
NOT_THERE = "not there"
NOTE_V8 = ("V8 is filed by the question types of its dataset, not question by question, so some of the questions it counts under "
           "\"What did it say or show\" ask who or when.")
EXAMPLES = 3  # of each category and media, that have something to find
ORDER = {media: n for n, media in enumerate(MEDIA)}


@dataclass(frozen=True)
class Row:
    qid: str
    text: str
    answer: str
    split: str
    absent: bool
    category: str
    cue: str
    rule: str


@dataclass(frozen=True)
class Entry:
    name: str
    code: str
    shelf: Shelf
    rows: tuple[Row, ...]
    files: int | None  # None when only the questions are known
    hours: float | None  # None for a pile with no durations
    note: str  # why it has no questions listed, when it has none


def answer_of(q: Question) -> str:
    """The answer in words: a multiple-choice question's option rather than its letter."""
    if q.kind == "absent":
        return NOT_THERE
    if q.options and q.truth.value in q.options:
        return q.options[q.truth.value]
    return ", ".join(q.grade.items) if q.grade.items else (q.truth.value or "")


def row_of_question(scenario, q: Question) -> Row:
    category, cue = filed(scenario, q.tags.get("template"))
    return Row(q.qid, q.question, answer_of(q), q.split, q.kind == "absent", category, cue, q.grade.rule)


def row_of_plan(scenario, q: Mapping) -> Row:
    """A row of a plan's question dict, as V7's and V8's plans hold them."""
    category, cue = filed(scenario, q["template"])
    return Row(q["qid"], q["question"], NOT_THERE if q["absent"] else str(q["truth"]["value"]), "dev", bool(q["absent"]), category, cue, q["grade"]["kind"])


def entry_of(name: str, scale: str, seed: int, built: Path | None = None) -> Entry:
    """The catalogue's entry for the scenario `name`: its plan at `scale` and `seed`, or, from `built` (a `questions.jsonl`), its questions alone.
    A scenario whose source is not on this machine has an entry with no questions, and says why."""
    scenario = scenarios.get(name)
    if built is not None:
        rows = tuple(row_of_question(scenario, q) for q in load_questions(built))
        return Entry(name, scenario.code, scenario.shelf, rows, None, None, "")
    try:
        made = scenario.plan_of(scenario.recipe(scale=SCALE_OF.get(name, scale), seed=seed))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return Entry(name, scenario.code, scenario.shelf, (), None, None, f"{type(error).__name__}: {error}")
    if isinstance(made, tuple):
        files, questions = made
        lengths = [f.length for f in files if f.length is not None]
        return Entry(name, scenario.code, scenario.shelf, tuple(row_of_question(scenario, q) for q in questions), len(files),
                     sum(lengths) / 3600 if lengths else None, "")
    return Entry(name, scenario.code, scenario.shelf, tuple(row_of_plan(scenario, q) for q in made.questions), len(made.files), None, "")


def entries_of(scale: str, seed: int) -> list[Entry]:
    return [entry_of(name, scale, seed, BUILT[name] if name in BUILT and BUILT[name].exists() else None) for name in scenarios.REGISTRY]


def code_order(entry: Entry) -> tuple[str, int]:
    """V2 before V10."""
    return entry.code[0], int(entry.code[1:])


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)]


def size_of(entry: Entry) -> str:
    parts = []
    if entry.files is not None:
        parts.append(f"{entry.files:,} file{'' if entry.files == 1 else 's'}")
    if entry.hours is not None:
        parts.append(f"{entry.hours:,.1f} h")
    return ", ".join(parts) or "-"


def categories_lines() -> list[str]:
    return table(["Category", "A person says", "The answer"], [[c.name, f"\"{c.says}\"", c.answer] for c in CATEGORIES])


def shelves_lines(entries: Sequence[Entry], origin: Mapping[str, str]) -> list[str]:
    out = []
    for media in MEDIA:
        here = sorted((e for e in entries if e.shelf.media == media), key=code_order)
        if not here:
            continue
        out += ["", f"### {media.capitalize() if media != 'mixed' else 'Mixed'}", ""]
        out += table(["ID", "Shelf", "Life area", "The pile", "Where it is from", "Size (standard)", "Questions (absent)"],
                     [[e.code, e.shelf.name, AREAS[e.shelf.area], e.shelf.pile, origin.get(e.code, ""), size_of(e),
                       f"{len(e.rows)} ({sum(r.absent for r in e.rows)})" if e.rows else "not built here"] for e in here])
    return out


def balance_lines(entries: Sequence[Entry]) -> list[str]:
    rows = [(e.shelf, r) for e in entries for r in e.rows]
    keys = [c.key for c in CATEGORIES]
    medias = [m for m in MEDIA if any(s.media == m for s, _ in rows)]
    counts = Counter((r.category, s.media) for s, r in rows)
    out = ["### Shelves, by media and life area", ""]
    out += table(["", *AREAS.values(), "All"],
                 [[m.capitalize(), *(sum(e.shelf.media == m and e.shelf.area == a for e in entries) for a in AREAS),
                   sum(e.shelf.media == m for e in entries)] for m in MEDIA if any(e.shelf.media == m for e in entries)])
    out += ["", "### Questions, by category and media", ""]
    out += table(["Category", *(m.capitalize() for m in medias), "All", "Absent"],
                 [[c.name, *(counts[c.key, m] for m in medias), sum(counts[c.key, m] for m in medias),
                   sum(r.absent for _, r in rows if r.category == c.key)] for c in CATEGORIES])
    out += ["", "### Questions, by what must be perceived", ""]
    cues = Counter(r.cue for _, r in rows)
    out += table(["Cue", "Questions"], [[f"{cue} ({said})", cues[cue]] for cue, said in CUES.items() if cue in cues])
    total, absent = len(rows), sum(r.absent for _, r in rows)
    out += ["", f"{total} questions in all, {absent} of them about things that are not there ({100 * absent / total:.0f}%)." if total else ""]
    if unfiled := sum(r.category not in keys for _, r in rows):
        out += ["", f"**{unfiled} questions are not filed under a category.**"]
    return out


def spread(rows: Sequence[tuple[Entry, Row]], n: int) -> list[tuple[Entry, Row]]:
    """`n` of `rows`, a shelf at a time, so that the examples come from as many shelves as there are."""
    by: dict[str, list] = {}
    for pair in rows:
        by.setdefault(pair[0].code, []).append(pair)
    out: list = []
    while len(out) < n and any(by.values()):
        for code in sorted(by):
            if by[code] and len(out) < n:
                out.append(by[code].pop(0))
    return out


def examples_lines(entries: Sequence[Entry], per: int) -> list[str]:
    """Examples of each category and media from the `dev` split: those with something to find, and one that is not there."""
    out = []
    pairs = [(e, r) for e in entries for r in e.rows if r.split == "dev"]
    for c in CATEGORIES:
        here = [(e, r) for e, r in pairs if r.category == c.key]
        out += ["", f"### {c.name}", ""]
        if not here:
            out += ["None on the dev split yet."]
            continue
        for media in MEDIA:
            of = [(e, r) for e, r in here if e.shelf.media == media]
            shown = spread([x for x in of if not x[1].absent], per) + spread([x for x in of if x[1].absent], 1)
            out += [f"- *{media}, {e.code}:* \"{r.text}\" Answer: {r.answer}" for e, r in shown]
    return out


def markdown(entries: Sequence[Entry], scale: str, seed: int, origin: Mapping[str, str] | None = None, per: int = EXAMPLES) -> str:
    """CATALOGUE.md: the categories, the shelves, the balance and examples of the questions."""
    out = ["# The MMSB catalogue", "", (f"Made by `mmsb catalogue` from the scenarios themselves, at scale `{scale}` and seed {seed}. Examples are from the "
                                         "`dev` split. Every answer is computed from the source's own labels and graded by a fixed rule."), ""]
    out += ["## What people ask", "", ("Eight categories, by what the person wants, whatever the media. \"It isn't there\" is no category: about a quarter "
                                        "of the questions are about things that never happened, and read exactly like the others."), ""]
    out += categories_lines()
    out += ["", "## The media, by shelf"]
    out += shelves_lines(entries, origin or {})
    out += ["", "## Balance", ""]
    out += balance_lines(entries)
    if any(e.code == "V8" and e.rows for e in entries):
        out += ["", NOTE_V8]
    out += ["", "## Questions, by category"]
    out += examples_lines(entries, per)
    if missing := [e for e in entries if not e.rows]:
        out += ["", "## Not built here", ""]
        out += [f"- {e.code} ({e.name}): {e.note}" for e in missing]
    return "\n".join(out) + "\n"


def full(entries: Sequence[Entry]) -> str:
    """Every question, with its answer, by category and shelf: not for the repository."""
    out = ["# Every question", ""]
    for c in CATEGORIES:
        out += [f"## {c.name}", ""]
        rows = [[e.code, e.shelf.media, r.cue, r.split, "yes" if r.absent else "", r.text.replace("|", "/"), r.answer.replace("|", "/")]
                for e in sorted(entries, key=lambda e: (ORDER[e.shelf.media], code_order(e))) for r in e.rows if r.category == c.key]
        out += table(["ID", "Media", "Cue", "Split", "Absent", "Question", "Answer"], rows) + [""]
    return "\n".join(out)


def write(out: Path | None, full_out: Path | None, scale: str, seed: int) -> str:
    """Writes the catalogue to `out` and every question to `full_out`, each when given; returns the catalogue."""
    entries = entries_of(scale, seed)
    text = markdown(entries, scale, seed, ORIGIN)
    if out is not None:
        Path(out).write_bytes(text.encode("utf-8"))
    if full_out is not None:
        Path(full_out).write_bytes(full(entries).encode("utf-8"))
    return text
