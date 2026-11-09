"""The source of I8: the PDFs of four MIT OpenCourseWare courses on disk, one folder each (`ocw-pages/<course>/<hash>_<name>.pdf`, read with `pypdfium2`).

Each course's folder holds its lecture material and its problem sets, solutions, exams, clicker questions, equation sheets and readings. Only the lecture
material is notes (`Course.notes`, from the file names and the first pages): chemistry's six lecture summaries (`Lec1` to `Lec6`), algorithms' ten
lectures and three recitations, biology's ten recitation summaries (the course has no lecture notes), psychology's ten slide decks. A page's truth is
the text layer of its PDF.

Not every page is one a question can rest on (`usable`): a page whose text layer has next to nothing is a figure or a blank, one with a picture over a
tenth of its area may carry words in the picture that the text layer does not have, and each PDF's last page of the OCW citation says only what course
it is. The rest are the folder."""

from __future__ import annotations

import functools
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from mmsb.scenarios.realdump import sources_root

SOURCE = sources_root() / "ocw-pages"
MIN_CHARS = 40  # a page whose text layer is shorter than this is no page of notes
BIG_PICTURE = 0.10  # the share of a page's area a picture on it may cover
SCALE = 1.6  # the pixels of a rendered page for each point of the PDF's page (72 to the inch)
JPEG_QUALITY = 82


@dataclass(frozen=True)
class Session:
    """The PDFs of a course that meet when: a lecture on some weekdays at a time. `match` picks them by their file names, in the order of their numbers."""

    match: str
    days: tuple[int, ...]  # weekdays, 0 for Monday
    at: time


@dataclass(frozen=True)
class Course:
    key: str  # what a person calls it
    folder: str
    sessions: tuple[Session, ...]  # the notes: no other PDF of the folder is one
    accepted: str  # the phrases of a right answer to which course, `|` between


COURSES = (
    Course("chemistry", "5-111sc-principles-of-chemical-science-fall-2014", (Session(r"_Lec\d\.pdf$", (0, 2, 4), time(10, 0)),),
           "chemistry|chemical science|5 111"),
    Course("algorithms", "6-006-introduction-to-algorithms-spring-2020",
           (Session(r"_lec\d+\.pdf$", (1, 3), time(11, 30)), Session(r"_r\d+\.pdf$", (4,), time(16, 30))), "algorithms|6 006"),
    Course("biology", "7-016-introductory-biology-fall-2018", (Session(r"rec\d+\.pdf$", (2,), time(13, 0)),), "biology|7 016"),
    Course("psychology", "9-00sc-introduction-to-psychology-fall-2011", (Session(r"lec\d+(_\w+)?\.pdf$", (0, 2, 4), time(14, 30)),), "psychology|9 00"),
)


@dataclass(frozen=True)
class Page:
    course: str  # Course.key
    pdf: str  # the PDF's file name in the source
    number: int  # the page in it, from 0
    text: str  # its text layer, as the PDF gives it (NFKC)
    big_picture: bool  # a picture covers over BIG_PICTURE of it

    @property
    def usable(self) -> bool:
        """Whether a question may rest on this page's text (see the module's docstring)."""
        return len(self.text) >= MIN_CHARS and not self.big_picture and not ("opencourseware" in self.text.casefold() and len(self.text) < 400)


def numbered(name: str) -> int:
    """The lecture's or recitation's number in a PDF's name: `MIT6_006S20_lec10.pdf` is 10, `MIT9_00SCF11_lec01.pdf` 1; the first number after
    `lec`, `rec` or `r`."""
    found = re.search(r"(?:lec|rec|_r)(\d+)", name.split("_", 1)[-1], re.IGNORECASE)
    if not found:
        raise ValueError(f"{name} has no lecture number")
    return int(found[1])


def notes_of(course: Course, root: Path) -> list[tuple[Session, Path]]:
    """The PDFs of a course that are its notes, each with the session it meets in, in the order of session and number."""
    out = []
    for n, session in enumerate(course.sessions):
        found = sorted((p for p in (root / course.folder).glob("*.pdf") if re.search(session.match, p.name, re.IGNORECASE)), key=lambda p: numbered(p.name))
        if not found:
            raise FileNotFoundError(f"no PDF of {course.folder} matches {session.match}: `mmsb fetch --scenario i8_course_pages` gets them")
        out += [(session, p) for p in found]
    return out


def page_of(course: Course, pdf: Path, number: int, page) -> Page:
    area = page.get_size()[0] * page.get_size()[1]
    big = False
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = obj.get_bounds()
        big = big or (right - left) * (top - bottom) > BIG_PICTURE * area
    text = unicodedata.normalize("NFKC", page.get_textpage().get_text_range())
    return Page(course.key, pdf.name.split("_", 1)[-1], number, text, big)


@functools.lru_cache(maxsize=2)
def read_pages(source: str) -> dict[str, tuple[Page, ...]]:
    """Every page of every PDF that is notes, by course key, in the order of the course's sessions and the PDFs' numbers."""
    root = Path(source)
    out = {}
    for course in COURSES:
        pages = []
        for _, pdf in notes_of(course, root):
            doc = pdfium.PdfDocument(pdf)
            pages += [page_of(course, pdf, n, doc[n]) for n in range(len(doc))]
            doc.close()
        out[course.key] = tuple(pages)
    return out


def pdf_path(source: Path | str, course: str, name: str) -> Path:
    """The file of the PDF `name` (as `Page.pdf` has it) of `course` in `source`."""
    (folder,) = [c.folder for c in COURSES if c.key == course]
    (found,) = [p for p in (Path(source) / folder).glob("*.pdf") if p.name.split("_", 1)[-1] == name]
    return found


def render(source: Path | str, pages: list[Page], workers: int = 1) -> dict[tuple[str, str, int], bytes]:
    """The pages as JPEGs of SCALE and JPEG_QUALITY, by `(course, pdf, number)`: each PDF opened once."""
    out: dict[tuple[str, str, int], bytes] = {}
    by_pdf: dict[tuple[str, str], list[int]] = {}
    for p in pages:
        by_pdf.setdefault((p.course, p.pdf), []).append(p.number)
    for (course, name), numbers in by_pdf.items():
        doc = pdfium.PdfDocument(pdf_path(source, course, name))
        for n in numbers:
            image = doc[n].render(scale=SCALE).to_pil().convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=JPEG_QUALITY)
            out[course, name, n] = buffer.getvalue()
        doc.close()
    return out
