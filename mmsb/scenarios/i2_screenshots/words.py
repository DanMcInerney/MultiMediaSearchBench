"""The words of I2: the phrases a person remembers from a screen, and finds it by. A phrase is one line of the text OCR read off a screen
(source.py), folded; it is asked about only where the screen it is on is certain, so that a phrase is:
- ordinary words: each in the text of VOCABULARY episodes of the source (OCR's garbled words and the names of private people are in one
  episode), the first and last of them whole and not a filler (a line OCR cut short begins or ends in a stub or an "and"), and no digits (a phone
  number, a price, a code);
- on one screen of the pile, and no other screen has most of its words (NEAR): a screen where OCR missed or garbled the phrase would be an
  answer too, and the list of OCR's lines cannot tell it;
- on a screen of an app that shows no private person's name (BANNED): mail, contacts, photos, messages."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from mmsb.scenarios.i2_screenshots.source import Episode, Shot, fold, seen

VOCABULARY = 7  # the episodes of the source whose text holds each word of a phrase
WORDS = (5, 10)  # the words of a phrase, at least and at most
LONGEST_STUB = 2  # the first and last word of a phrase are longer than this: a stub is a word OCR cut
FILLER = frozenset("and or of in on at to for is are with from by as it this that than then but not you your".split())  # nor are they one of these
ENDINGS = 5  # a first word of this many letters or fewer that ends this many other words (`tion`) is a word OCR cut
NEAR = 0.75  # the share of a phrase's words that another screen must not show
BANNED = ("com.google.android.gm", "com.android.contacts", "com.google.android.apps.photos", "com.google.android.apps.messaging",
          "com.android.dialer", "com.google.android.dialer")
SHORT = frozenset("a i an as at by do go he if in is it me my no of on or so to up us we".split())  # the words of two letters or fewer that are words
NOT_SAID = frozenset("gmail password account http https htps www com org".split())  # a screen about a person's own account, and a web address


@dataclass(frozen=True)
class Phrase:
    text: str  # folded
    shot: Shot  # the screen it is on


def spread(episodes: Sequence[Episode]) -> dict[str, int]:
    """For each word of the text of `episodes`, in how many of them it is."""
    found: dict[str, set[str]] = defaultdict(set)
    for e in episodes:
        for s in e.shots:
            for w in seen(s).split():
                found[w].add(e.id)
    return {w: len(es) for w, es in found.items()}


def phrases(targets: Sequence[Shot], rest: Sequence[Shot], spoken: dict[str, int], once: bool) -> list[Phrase]:
    """The phrases of the screens `targets` a question may be asked of: on one screen of `targets` (`once`) or any number of them, on none
    of `rest`, and with no screen of either having NEAR of its words but the screen it is on."""
    words = [set(seen(s).split()) for s in (*targets, *rest)]
    index: dict[str, set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for x in w:
            index[x].add(i)
    ends = Counter(w[-k:] for w in spoken for k in range(2, 5) if len(w) > k)
    where: dict[str, set[int]] = {}
    for i, shot in enumerate(targets):
        if any(shot.activity.startswith(b) for b in BANNED):
            continue
        for line in shot.ocr:
            text = fold(line)
            parts = text.split()
            if (WORDS[0] <= len(parts) <= WORDS[1] and not any(c.isdigit() for c in text) and not NOT_SAID & set(parts)
                    and len(parts[0]) > LONGEST_STUB and len(parts[-1]) > LONGEST_STUB and not {parts[0], parts[-1]} & FILLER and ends[parts[0]] < ENDINGS
                    and all(spoken.get(w, 0) >= VOCABULARY and (w in SHORT or (len(w) > 2 and re.search("[aeiouy]", w))) for w in parts)):
                where.setdefault(text, set()).add(i)
    out = []
    for text, found in where.items():
        if once and len(found) != 1:
            continue
        needle = " " + text + " "
        if any(needle in seen(s) for s in rest):
            continue
        parts = set(text.split())
        near = Counter(j for w in parts for j in index[w])
        if any(c >= math.ceil(NEAR * len(parts)) for j, c in near.items() if j not in found and (once or j >= len(targets))):
            continue
        if once and any(needle in seen(s) for k, s in enumerate(targets) if k not in found):
            continue
        out.append(Phrase(text, targets[min(found)]))
    return sorted(out, key=lambda p: (p.shot.file, p.text))
