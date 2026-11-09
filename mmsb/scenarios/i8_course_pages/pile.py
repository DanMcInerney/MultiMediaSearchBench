"""The pile of I8: which pages of the notes are in the folder, and which were taken out of it for the questions about what is not there.

The folder holds every usable page (source.py `Page.usable`) of the notes of the four courses, but the pages held out: for each of a few terms that
are on one page only, that page, so that the term is on no page of the folder. They are drawn with a fixed seed, and are the same at every question
seed. The tiny folder is four of the PDFs."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mmsb.scenarios.i8_course_pages.source import Page
from mmsb.scenarios.i8_course_pages.terms import TERMS, Term, mentions

HELD = {"standard": 6, "tiny": 1}  # how many terms have their one page held out
TINY_PDFS = (("chemistry", "MIT5_111F14_Lec2.pdf"), ("algorithms", "MIT6_006S20_lec3.pdf"), ("biology", "MIT7_016F18rec3.pdf"),
             ("psychology", "MIT9_00SCF11_lec07_attn.pdf"))  # the lectures of the tiny folder


@dataclass(frozen=True)
class Pile:
    pages: tuple[Page, ...]  # in the folder, in the order of the courses and their PDFs (the folder's own order is the time they were saved at)
    held: tuple[tuple[Term, Page], ...]  # a term and its one page, which was taken out


def key(page: Page) -> tuple[str, str, int]:
    return page.course, page.pdf, page.number


def choose(pages: Mapping[str, Sequence[Page]], scale: str, keep: Sequence[tuple[str, str, int]]) -> Pile:
    """The pile of `scale` from the pages of the source: all the usable pages but the held out (`standard`), or those of TINY_PDFS (`tiny`). The pages
    in `keep` are never held out (the ones a written question reads)."""
    usable = [p for course in pages.values() for p in course if p.usable]
    if scale == "tiny":
        usable = [p for p in usable if (p.course, p.pdf) in TINY_PDFS]
    on = {t: [p for p in usable if mentions(p, t.search)] for t in TERMS}
    sole = sorted((t for t, w in on.items() if len(w) == 1 and key(w[0]) not in keep), key=lambda t: (t.course, t.search))
    rng = random.Random(f"i8:{scale}")
    held: list[tuple[Term, Page]] = []
    for t in rng.sample(sole, len(sole)):
        if len(held) < HELD[scale] and on[t][0] not in [p for _, p in held]:
            held.append((t, on[t][0]))
    gone = {key(p) for _, p in held}
    return Pile(tuple(p for p in usable if key(p) not in gone), tuple(held))
