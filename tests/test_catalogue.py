"""The catalogue of the media and questions (mmsb/catalogue.py), from made-up entries: no scenario's source is read."""

from __future__ import annotations

from mmsb.catalogue import Entry, Row, full, markdown
from mmsb.core.taxonomy import Shelf


def row(qid: str, category: str, *, split: str = "dev", absent: bool = False, answer: str = "yes") -> Row:
    return Row(qid, f"Question {qid}?", answer, split, absent, category, "sight", "time")


def entries() -> list[Entry]:
    video = Entry("v", "V1", Shelf("Security", "video", "home", "cameras"), (row("a", "when"), row("b", "when", split="test", answer="hidden"),
                                                                              row("c", "find_all", absent=True, answer="not there")), 6, 3.5, "")
    image = Entry("i", "I3", Shelf("Trail", "image", "fun", "a card"), (row("d", "when"),), 4256, None, "")
    empty = Entry("a", "A9", Shelf("Field", "audio", "fun", "a night"), (), None, None, "OSError: no source")
    return [video, image, empty]


def test_the_catalogue_files_shelves_by_media_and_counts_the_questions():
    text = markdown(entries(), "standard", 1, {"V1": "MEVA"})
    assert "### Video" in text and "### Image" in text and "### Audio" in text
    assert "| V1 | Security | Home and family | cameras | MEVA | 6 files, 3.5 h | 3 (1) |" in text
    assert "| I3 | Trail | Fun and hobbies | a card |  | 4,256 files | 1 (0) |" in text
    assert "4 questions in all, 1 of them about things that are not there (25%)." in text


def test_a_scenario_with_no_questions_says_why():
    text = markdown(entries(), "standard", 1)
    assert "- A9 (a): OSError: no source" in text
    assert "not built here" in text


def test_examples_are_from_the_dev_split_only_and_full_has_every_question():
    text = markdown(entries(), "standard", 1)
    assert "Question a?" in text and "Question b?" not in text and "hidden" not in text
    everything = full(entries())
    assert "Question b?" in everything and "hidden" in everything


def test_an_unfiled_question_is_called_out():
    odd = Entry("v", "V1", Shelf("S", "video", "home", "p"), (row("x", "unfiled"),), 1, None, "")
    assert "1 questions are not filed under a category" in markdown([odd], "standard", 1)
