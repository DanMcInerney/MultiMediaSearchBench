"""The general categories the benchmark's questions and media are filed under (DESIGN 5.8). A question is in one category, by what the person
wants, whatever the media. A scenario is one `Shelf`, by the kind of pile a real person has and the part of their life it is from. A cue says
what has to be perceived to answer: a system with no speech can then be told apart from one that misses the needle.

A scenario files its own questions: besides the names of a `Scenario` it has a `shelf` and a `filing`, `{template: (category, cue)}`. Nothing of
this is in a question file, so filing a scenario changes no question, plan hash or recorded run. "It isn't there" is no category: about a quarter
of the questions are about what never happened, and an absent question reads, and is filed, as the question it looks like (`with_absent`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Category:
    key: str
    name: str
    says: str  # what a person says
    answer: str  # what comes back


CATEGORIES = (
    Category("find", "Find it", "Find the photo of the parking-level sign.", "one file or moment"),
    Category("find_all", "Find every one", "Show me every clip with the cat.", "a set of files, moments or dates"),
    Category("when", "When", "What time did the van come? When did I last see it?", "a time or date"),
    Category("did", "Did it happen", "Did the smoke alarm go off? Was there ever a bear?", "yes and when, or not there"),
    Category("how_many", "How many, how long", "How many times did the dog bark? How long did it run?", "a number"),
    Category("what", "What did it say or show", "What was the total? What deadline did they give?", "a short value, read or heard"),
    Category("who", "Who or which", "Who suggested the price? Which camera saw it first?", "a person, file or place"),
    Category("order", "Before and after", "What happened right after the alarm? Was the parcel before the rain?", "an order, or a neighbouring event"),
)
CATEGORY_KEYS = tuple(c.key for c in CATEGORIES)
AREAS = {"home": "Home and family", "work": "Work", "school": "School", "fun": "Fun and hobbies"}
CUES = {"sight": "seen", "sound": "heard, not speech", "speech": "heard, words", "text": "read on screen or paper", "mixed": "several of these"}
MEDIA = ("video", "image", "audio", "mixed")
UNFILED = "unfiled"  # a scenario with no filing (a test's stand-in), or a template it does not list


@dataclass(frozen=True)
class Shelf:
    name: str  # "Trail camera"
    media: str  # one of MEDIA
    area: str  # a key of AREAS
    pile: str  # the pile, in a line


Filing = Mapping[str, tuple[str, str]]


def with_absent(filing: Filing, prefix: str = "absent_") -> dict[str, tuple[str, str]]:
    """`filing` and the absent twin of each of its templates (`absent_first` reads as `first`, and is filed with it)."""
    return {**filing, **{prefix + template: filed for template, filed in filing.items()}}


def check(shelf: Shelf, filing: Filing) -> None:
    """Raises a ValueError when a shelf or a filing uses a word that is not one of this module's."""
    if shelf.media not in MEDIA or shelf.area not in AREAS:
        raise ValueError(f"{shelf.name}: media {shelf.media!r} or area {shelf.area!r} is not one of {MEDIA} and {tuple(AREAS)}")
    for template, (category, cue) in filing.items():
        if category not in CATEGORY_KEYS or cue not in CUES:
            raise ValueError(f"{shelf.name}: template {template!r} is filed under {category!r}, {cue!r}: not a category and a cue")


def filed(scenario: Any, template: object) -> tuple[str, str]:
    """`(category, cue)` of a template of `scenario`; `(UNFILED, UNFILED)` when it has no filing or does not list the template."""
    return getattr(scenario, "filing", {}).get(template, (UNFILED, UNFILED))
