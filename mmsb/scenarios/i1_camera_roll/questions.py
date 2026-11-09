"""The questions of I1, every answer from ATM-Bench's own labels (source.py): its owner's real questions about their phone photos, in their
own words, whose evidence is all photos; each asked the morning after the pile's last photo. Which are asked, and which are held out of the
pile, is drawn with the seed.

| template | question                                                                        | rule                    |
| `find`   | I remember a very cute photo of a penguin lying down. Can you help me recall it? | `set` of one file name |
| `photos` | I took some iconic photos of Edinburgh's clock tower. Can you help me recall them?| `set` of file names    |
| `date`   | I saw a skywriting smiley face. When was that?                                  | `set` of one date       |
| `price`  | How much did my dinner cost when I first arrived in Thailand?                   | `amount`                |
| `read`   | What is my seat number on the flight to Amman? / What was that paper called?    | `text`                  |
| `which`  | Which airline did I take to Santiago?                                           | `text`                  |
| `days`   | I had an issue with a leaking hot water system at home. How long did the repair | `count`                 |
|          | take?                                                                           |                         |
| `first`  | Which did I photograph first: the giraffes at Edinburgh Zoo or a panda?         | `text`                  |
| `ever`   | Did I ever take a photo of a Concorde? If I did, when?                          | `set` of one date       |
| `absent` | any of the above, but a real question whose evidence photos, and every photo   | `absent`                |
|          | within a minute of them, are left out of the pile (a held-out needle)           |                         |

What a question avoids:
- one the labels may have wrong or loose: the stated answer is checked against the evidence (a `list_recall` answer must name exactly the
  evidence photos, a date answer must be the evidence photos' date, none of them within a minute of midnight: the capture time is read from
  the file name, EXIF is a second or more later), and a question about a photo that another file shares the second with is left out;
- one an agent could not answer from the photos with their GPS stripped: `where` questions, whose answers are addresses from the position, are
  left out, and so is a name of a restaurant or a hotel unless the photo shows it (the `read` and `which` tables are the ones whose answers were
  seen in the photos);
- a name of a person, a child, a password, a bill for the owner's health, a video ("footage"), a question dated by its own "today is";
- a set that may be longer than the evidence: a `find` or `photos` question whose evidence has another photo within two minutes is left out,
  and a `photos` set is of one day, of at most six;
- an absent question whose answer is elsewhere in the pile: only distinctive things are held out (HOLD), and the held-out photos go with
  every photo within a minute of them."""

from __future__ import annotations

import bisect
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i1_camera_roll.source import CLOCK, Photo, Real, stated_date

CODE = "I1"
SOURCE_NAME = "phone"  # the dump's source
NEIGHBOUR_S = 120  # a `find` or `photos` question has no other photo this close to its evidence
HOLD_S = 60  # a held-out needle is left out with every photo this close to it
GAP_S = 600  # `first` asks of two things at least this far apart
TRIP_DAYS = 30  # and no further apart than this
MAX_SET = 6
MIDNIGHT_S = 60
FILES = "<the names of the files, as in IMG_0042.jpg, separated by commas>"
DATE = "<the date, as YYYY-MM-DD>"
NUMBER = "<a number>"
SHORT = "<a short answer, as the photo gives it>"
THING = "<the one that came first>"
QUOTAS = {"tiny": {"find": 1, "photos": 1, "date": 2, "which": 1, "ever": 1, "absent_find": 1, "absent_date": 1},
          "standard": {"find": 3, "photos": 2, "date": 4, "price": 1, "read": 2, "which": 2, "days": 1, "first": 2, "ever": 1,
                       "absent_find": 1, "absent_photos": 1, "absent_date": 2, "absent_read": 1, "absent_ever": 1}}

# The first six characters of the dataset's question ids (they are unique among the questions this reads).
BLOCK = {  # asked of no pile
    "2f75e1",  # names a person (a speaker)
    "d1002e", "64be36",  # a wifi password and a lounge's
    "dd6f1b",  # footage: videos
    "5e2253",  # a discussion with a colleague, not a photo to find
    "31fd11",  # a person described: her clothes
    "25012e", "6f1910",  # the owner's dental care
    "731bab",  # at a school: children
    "c57b0d"}  # asks for "image IDs", the dataset's word
BROAD = {"0fba0d", "98d80f", "772a6e", "bc0420", "0f1e19", "500f10", "2312cc"}  # sets of "the photos of X" that may be longer than the evidence
PRICE = {  # how much: id -> the number the photo shows (a receipt, a bill, a tag, a screen, a slide), and what an answer may be out by
    "f5338b": (23, 0), "d1583d": (2194, 0), "3a73ae": (892.85, 0.01), "ebfdee": (900, 0), "cd3567": (70, 0), "4bd421": (3218, 0),
    "7df35b": (2.1, 0), "2715a5": (159, 0)}
READ = {  # a short value read off a photo: id -> accepted phrases (`|`), phrases that make it wrong (`|`)
    "109937": ("1k", ""), "f7e986": ("1a", ""), "3a4ee2": ("16a", ""),  # a seat on a boarding pass
    "3c6f24": ("safe clip", ""), "f478cb": ("single image hdr reconstruction", ""), "be03ff": ("inverse scaling law for clip training", ""),
    "bbec34": ("retrieval guided contrastive learning", ""), "f56d14": ("latently perform multi hop reasoning", ""),
    "19c8fa": ("crossscore", ""), "aa0942": ("search adaptor", ""), "93703c": ("discom kd", ""), "f3e5cf": ("mm safetybench", ""),
    "aaef20": ("hard positive truth", ""), "afb25c": ("autoeval video", ""), "f98cc2": ("label poisoning is all you need", ""),  # a poster's title
    "c742b9": ("chatgpt|chat gpt", "")}  # a slide's credit
WHICH = {  # which one: id -> accepted phrases (`|`), phrases that make it wrong; the photo shows the name, or a sight a person names
    "230a80": ("ryanair", ""), "cc0ab9": ("vueling", ""), "917f59": ("winter palace", ""), "74455c": ("bytedance|byte dance", ""),
    "0c5bd9": ("harbour bridge|harbor bridge", ""), "a496f2": ("anthony", ""), "0b9c42": ("minack", ""), "cfd96f": ("giles", ""),
    "a51c5f": ("st mary|saint mary", "")}
DAYS = {  # how many: id -> what the evidence's dates give: `dates` (distinct days) or `span` (days from the first to the last)
    "093253": "dates", "90ba46": "span"}
THINGS = {  # a thing a person may say they photographed: id -> (how they say it, the word an answer names it by), of a real question whose
    # evidence is of one day (so that `ever` has a date) and which holds no other thing's word
    "34537e": ("the meerkats at Edinburgh Zoo", "meerkats"), "e4ae5e": ("the black bear at Edinburgh Zoo", "bear"),
    "371f6d": ("a panda lying down", "panda"), "2f890b": ("a penguin lying down", "penguin"),
    "73242c": ("the giraffes at Edinburgh Zoo", "giraffes"),
    "5a8b3f": ("the City Observatory in Edinburgh", "observatory"), "6d5dde": ("Edinburgh's clock tower from Calton Hill", "clock tower"),
    "6e54d9": ("the front gate of Edinburgh Castle", "castle"), "55093c": ("a street performer playing the bagpipes", "bagpipes"),
    "15d8d0": ("a koala in Australia", "koala"), "84573c": ("the Sydney Opera House at sunset", "opera house"),
    "3e2eb9": ("a departure board with a flight to the North Pole", "north pole"), "2af75f": ("a Concorde", "concorde"),
    "57ee48": ("my NeurIPS 2023 badge", "badge"), "d9983f": ("a pair of ladybugs", "ladybugs"),
    "9ea37b": ("a small dog figurine on a brick wall", "figurine"), "cfa294": ("a cat sleeping on a green bin", "bin"),
    "0b9c42": ("the Minack Theatre in Cornwall", "minack"), "72a6b8": ("the sunrise from the plane to Cornwall", "sunrise"),
    "d8aa22": ("the gold mask of Tutankhamun", "mask"), "4be897": ("black honey at a hotel breakfast in Egypt", "honey")}
HOLD = {  # real questions whose subject is one distinctive thing, so that with their photos out of the pile it is not elsewhere in it, but
    # where the question's own words limit it: 3175b0 asks for the Guangzhou Tower photo "in 2023", and the tower is in a 2024 photo of the pile too
    "find": {"e86634", "2af75f", "d9983f", "2f890b", "371f6d", "bec883", "3175b0", "01d308", "09467c", "2a7af0"},
    "photos": {"6d5dde", "3e2eb9", "84573c", "2cffed", "8b731d", "387e24"},
    "date": {"a23f04", "e2ffb7", "020811", "a6afce", "f37d8c", "8e937f", "739a6e", "de2b02", "896fa6", "b52e45", "d8aa22", "67feec",
             "42debb", "73242c", "e4ae5e"},
    "read": set(READ),
    "ever": set(THINGS)}
NOT_TODAY = re.compile(r"\btoday is\b|\bit['’]?s (?:now|[a-z]+ \d)|\bit is (?:now|\d)|\blast time\b|\bmost recent|\byesterday\b", re.IGNORECASE)


@dataclass(frozen=True)
class Cand:
    """A real question that can be asked of the pile: `template` (find, photos, date, price, read, which, days, ever) with its truth. `files`
    are the evidence photos' file names, in the order taken."""
    template: str
    id: str
    text: str
    files: tuple[str, ...]
    truth: tuple[str, ...] = ()  # date: the ISO day; price: the number and tolerance; read, which: accepted and wrong phrases; days: the number
    value: str | None = None  # ever: how the thing is said; the thing's word


def clean(text: str) -> str:
    return " ".join(text.split())


def near_midnight(wall: datetime) -> bool:
    midnight = datetime.combine(wall.date(), time.min)
    return min((wall - midnight).total_seconds(), (midnight + timedelta(days=1) - wall).total_seconds()) < MIDNIGHT_S


class Pile:
    """The photos of a pile in the order taken, and what is near each."""

    def __init__(self, photos: Sequence[Photo]) -> None:
        self.photos = tuple(photos)
        self.walls = [p.wall for p in self.photos]
        self.by_stem = {p.file[:15]: p for p in self.photos if p.alone}

    def near(self, files: Sequence[str], seconds: float) -> set[str]:
        """The photos (file names) within `seconds` of any of `files`, themselves not among them."""
        out: set[str] = set()
        for f in files:
            wall = self.by_stem[f[:15]].wall
            lo = bisect.bisect_left(self.walls, wall - timedelta(seconds=seconds))
            hi = bisect.bisect_right(self.walls, wall + timedelta(seconds=seconds))
            out |= {p.file for p in self.photos[lo:hi]}
        return out - set(files)


def candidates(real: Sequence[Real], pile: Pile) -> list[Cand]:
    """Every real question that can be asked of `pile` (whose photos are the ones the questions rest on), with its truth."""
    out: list[Cand] = []
    for r in real:
        key = r.id[:6]
        stems = list(r.evidence)
        if key in BLOCK or "�" in r.question or not all(s in pile.by_stem for s in stems):
            continue
        photos = sorted((pile.by_stem[s] for s in stems), key=lambda p: (p.wall, p.file))
        files, days = tuple(p.file for p in photos), {p.wall.date() for p in photos}
        if any(near_midnight(p.wall) for p in photos):
            continue
        text = clean(r.question)
        shared = bool(pile.near(files, NEIGHBOUR_S))
        if r.qtype == "list_recall":
            if set(re.findall(r"\d{8}_\d{6}", r.answer)) != set(stems) or shared or key in BROAD:
                continue
            if len(files) == 1:
                out.append(Cand("find", key, text, files))
            elif len(files) <= MAX_SET and len(days) == 1:
                out.append(Cand("photos", key, text, files))
        if (r.qtype == "number" and (stated := stated_date(r.answer)) is not None and days == {stated} and not NOT_TODAY.search(text)
                and not CLOCK.search(r.answer)):
            out.append(Cand("date", key, text, files, (stated.isoformat(),)))
        if key in PRICE:
            out.append(Cand("price", key, text, files, (str(PRICE[key][0]), str(PRICE[key][1]))))
        if key in READ and not NOT_TODAY.search(text):
            out.append(Cand("read", key, text, files, READ[key]))
        if key in WHICH:
            out.append(Cand("which", key, text, files, WHICH[key]))
        if key in DAYS:
            gone = {p.wall.date() for p in photos}
            n = len(gone) if DAYS[key] == "dates" else (max(gone) - min(gone)).days
            out.append(Cand("days", key, text, files, (str(n),)))
        if key in THINGS and len(days) == 1:
            out.append(Cand("ever", key, f"Did I ever take a photo of {THINGS[key][0]}? If I did, when?", files, (next(iter(days)).isoformat(),),
                            THINGS[key][1]))
    return out


def pairs(cands: Sequence[Cand], pile: Pile) -> list[tuple[Cand, Cand]]:
    """The `first` questions the things allow: two things whose photos are at least GAP_S apart (the last of the earlier to the first of the
    later) and no more than TRIP_DAYS days, the earlier one first, with neither's word in the other's phrase."""
    things = sorted((c for c in cands if c.template == "ever"), key=lambda c: pile.by_stem[c.files[0][:15]].wall)
    out = []
    for i, a in enumerate(things):
        for b in things[i + 1:]:
            end, start = pile.by_stem[a.files[-1][:15]].wall, pile.by_stem[b.files[0][:15]].wall
            ok_gap = GAP_S <= (start - end).total_seconds() <= TRIP_DAYS * 86400
            ok_words = a.value not in THINGS[b.id][0].casefold().split() and b.value not in THINGS[a.id][0].casefold().split()
            if ok_gap and ok_words and a.id != b.id:
                out.append((a, b))
    return out


def draw(cands: Sequence[Cand], pile: Pile, scale: str, seed: int) -> tuple[list[tuple[str, Cand, Cand | None]], set[str]]:
    """The questions of `scale` drawn with `seed`, as `(template, candidate, other)` (`other` the second thing of a `first`), the held-out
    ones with the template `absent_<template>`, and the photos (file names) held out of the pile: the evidence of each held-out question and
    every photo within HOLD_S of it. No question rests on a held-out photo, and none shares a photo with another."""
    rng = random.Random(f"i1:{scale}:{seed}")
    quota = QUOTAS[scale]
    held: set[str] = set()
    used: set[str] = set()
    chosen: list[tuple[str, Cand, Cand | None]] = []
    by = {t: [c for c in cands if c.template == t] for t in {c.template for c in cands}}
    for t in by:
        by[t].sort(key=lambda c: c.id)
    firsts = pairs(cands, pile)

    def take(template: str, pool: list, n: int, absent: bool) -> None:
        pool = list(pool)
        rng.shuffle(pool)
        for item in pool:
            if n == 0:
                break
            pair = item if isinstance(item, tuple) else (item, None)
            files = {f for c in pair if c is not None for f in c.files}
            if absent:
                gone = files | pile.near(sorted(files), HOLD_S)
                if gone & used:
                    continue
                held.update(gone)
                used.update(files)
                chosen.append(("absent_" + template, item, None))
            else:
                if files & (held | used):
                    continue
                used.update(files)
                chosen.append((template, pair[0], pair[1]))
            n -= 1

    for template in ("find", "photos", "date", "read", "ever"):  # the held-out needles are drawn first, so that nothing else rests on them
        if quota.get("absent_" + template):
            take(template, [c for c in by.get(template, []) if c.id in HOLD[template]], quota["absent_" + template], True)
    for template in ("find", "photos", "date", "price", "read", "which", "days", "ever"):
        take(template, by.get(template, []), quota.get(template, 0), False)
    take("first", [p for p in firsts if not (p[0].files[0] in held or p[1].files[0] in held)], quota.get("first", 0), False)
    return chosen, held


def questions_of(chosen: Sequence[tuple[str, Cand, Cand | None]], paths: dict[str, str], since: float, until: float, now: float,
                 seed: int) -> tuple[Question, ...]:
    """The questions of `chosen` (`draw`'s), over the pile whose photos' dump paths are `paths` (by file name), asked at `now` (epoch
    seconds) of the span `since` to `until`."""
    out: list[Question] = []
    for template, c, other in chosen:
        absent = template.startswith("absent_")
        base = template.removeprefix("absent_")
        evidence = [Span(paths[f], 0.0, 0.0) for f in c.files] if not absent else []
        names = [paths[f].rsplit("/", 1)[-1] for f in c.files] if not absent else []
        text, shape, value, grade = c.text, FILES, None, realdump.grade("absent")
        if base == "find" or base == "photos":
            value = ", ".join(names)
            grade = realdump.grade("set", items=names)
        elif base in ("date", "ever"):
            shape, value = DATE, c.truth[0]
            grade = realdump.grade("set", items=c.truth)
        elif base == "price":
            shape, value = NUMBER, c.truth[0]
            grade = realdump.grade("amount", truth=c.truth[0], tol_s=float(c.truth[1]))
        elif base in ("read", "which"):
            shape, value = SHORT, c.truth[0].split("|")[0]
            grade = realdump.grade("text", truth=c.truth[0], wrong=[w for w in c.truth[1].split("|") if w])
        elif base == "days":
            shape, value = NUMBER, c.truth[0]
            grade = realdump.grade("count", truth=c.truth[0])
        elif base == "first":
            assert other is not None
            text = f"Which did I photograph first: {THINGS[c.id][0]} or {THINGS[other.id][0]}?"
            shape, value = THING, c.value
            evidence = [Span(paths[f], 0.0, 0.0) for f in (*c.files, *other.files)]
            grade = realdump.grade("text", truth=c.value, wrong=[other.value])
        if absent:
            shape, value, grade = {"find": FILES, "photos": FILES}.get(base, shape), None, realdump.grade("absent")
        qid = ("absent-" if absent else "") + f"{base}-{c.id}" + (f"-{other.id}" if other is not None else "")
        out.append(realdump.question(code=CODE, qid=qid, now=now, text=text, kind="absent" if absent else base, template=template,
                                     modality="image", shape=shape, grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since,
                                     until=until, value=value))
    return realdump.split(out, seed)
