"""The questions of I9, every answer from the observer's own observations (source.py), each asked the morning after the pile's last photo. A species
is asked of by the common name a person says; a date is a day (the pile's photos carry their time, but a day is what a person asks by):

| template | question                                                                         | rule                     |
| `days`   | On which days did I photograph a pink lady's slipper?                            | `set` of dates           |
| `count`  | On how many different days did I photograph a Wild Turkey?                       | `count`                  |
| `first`  | When did I first photograph a red trillium?                                      | `set` of one date        |
| `last`   | When did I last photograph a Ruby-throated Hummingbird?                          | `set` of one date        |
| `find`   | Find my photo of a Hairy Woodpecker. (a species photographed once)               | `set` of one file name   |
| `ever`   | Did I ever photograph a Monarch? If I did, when was the first time?              | `set` of one date        |
| `which`  | Which did I photograph first, a Wild Turkey or a Monarch?                        | `text`                   |
| `absent` | `days`, `count`, `first`, `last`, `find` or `ever` of a species with no photo     | `absent`                 |

A species the questions are asked of is one with a common name no other species has that a person says (no sedge or grass), photographed 5 to 30
times (the observer's few species with hundreds of photos are in the pile, never asked of: their days are the days with most photos, which a list
of the files gives away); a `find` is asked of a species photographed once that is on a list of ones a person knows (FAMILIAR),
and a `first`, `last` or `ever` question is not asked when its day is the first or the last day of the pile, which the list of the files gives away as
well. A `which` pair is of two species whose first photographs are at least a fortnight apart, neither's name in the other's.

An absent question is of a species that is in no photo at species or genus level: either a species of the region that the observer's
observations never name, nor any other species of its genus (NEVER, checked against the observations' every taxon), or a held-out species: a
species that is the only one of its genus in the pile, with every photo of it left out (so that the dump depends on the seed). Both read as the
real questions do."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i9_nature.source import Obs

CODE = "I9"
SOURCE_NAME = "phone"  # the dump's source
MAX_OBS = 30  # the most photos of a species the questions are asked of
MIN_OBS = {"tiny": 3, "standard": 5}  # the fewest photos of a species asked of: one the owner came back to, one a person asks about
DAYS = (3, 12)  # the days of a species a `days` or `count` question is asked of
# names no person says, and the canker the observer finds on the butternuts they photograph by the hundred (a photo of a butternut may show it)
OBSCURE = re.compile(r"sedge|grass|rush|moss\b|lichen|liverwort|pondweed|cottongrass|bryozoan|canker|\?", re.IGNORECASE)
FAMILIAR = {  # a species photographed once that a person remembers by name: a `find` question is asked of these
    "American Woodcock", "Belted Kingfisher", "Black-throated Green Warbler", "Brown Pelican", "Canada Goose", "Common Snapping Turtle",
    "Eastern Bluebird", "Eastern Cottontail", "European Starling", "Gray Fox", "Gray Treefrog", "Groundhog", "Humpback Whale", "Merlin", "Moose",
    "Northern Flicker", "Purple Finch", "Red-eyed Vireo", "Red-shouldered Hawk", "Red-tailed Hawk", "Scarlet Tanager", "Snow Goose",
    "Snowy Egret", "Spotted Sandpiper", "Spotted Lanternfly", "Wood Thrush", "White-throated Sparrow", "Atlantic Horseshoe Crab",
    "Harbour Porpoise", "Northern Gannet", "Upland Sandpiper", "Yellow-crowned Night Heron", "Nashville Warbler", "Tennessee Warbler",
    "Northern Shrike", "Silvery Blue", "Silver-spotted Skipper", "Ebony Jewelwing", "chicken of the woods", "garlic mustard", "great mullein",
    "black locust", "box elder", "quaking aspen", "red spruce", "white ash", "silver maple", "turkey-tail", "ostrich fern", "walking fern",
    "trailing arbutus", "Catnip", "dame's rocket", "Bull Thistle", "Smallmouth Bass", "Eastern Carpenter Bee", "Small White",
    "Blue Grosbeak", "Great Shearwater", "Northern Short-tailed Shrew"}
FORTNIGHT = 14
HELD_OBS = (5, 40)  # the photos of a species that may be held out
FILES = "<the name of the file, as in IMG_0042.jpg>"
DATES = "<the dates, as YYYY-MM-DD, separated by commas>"
DATE = "<the date, as YYYY-MM-DD>"
NUMBER = "<a number>"
NAME = "<the name of the one that came first>"
NEVER = {  # a species of the region as a person says it -> its genus: neither is in the observations (checked at the build)
    "Snowshoe Hare": "Lepus", "Great Horned Owl": "Bubo", "Coyote": "Canis", "Common Dandelion": "Taraxacum", "Fly Agaric": "Amanita",
    "Luna Moth": "Actias", "Common Nighthawk": "Chordeiles", "Barn Swallow": "Hirundo", "Hermit Thrush": "Catharus",
    "Cooper's Hawk": "Accipiter", "Common Grackle": "Quiscalus", "Queen Anne's Lace": "Daucus", "Morel": "Morchella", "Paper Wasp": "Polistes",
    "Mourning Cloak": "Nymphalis", "Snowy Owl": "Bubo", "Common Green Darner": "Anax", "Giant Puffball": "Calvatia",
    "Northern Water Snake": "Nerodia", "Brook Trout": "Salvelinus"}
QUOTAS = {"tiny": {"days": 1, "first": 1, "find": 1, "ever": 1, "absent_days": 1, "absent_ever": 1},
          "standard": {"days": 5, "count": 3, "first": 2, "last": 2, "find": 2, "ever": 2, "which": 2, "absent_days": 2, "absent_count": 1,
                       "absent_first": 1, "absent_ever": 2}}
WORDING = {"days": "On which days did I photograph {a}?", "count": "On how many different days did I photograph {a}?",
           "first": "When did I first photograph {a}?", "last": "When did I last photograph {a}?", "find": "Find my photo of {a}.",
           "ever": "Did I ever photograph {a}? If I did, when was the first time?",
           "which": "Which did I photograph first, {a} or {b}?"}
SHAPES = {"days": DATES, "count": NUMBER, "first": DATE, "last": DATE, "find": FILES, "ever": DATE, "which": NAME}


@dataclass(frozen=True)
class Taxon:
    species: str  # its binomial
    name: str  # the common name


def a_(name: str) -> str:
    """`a Wild Turkey`, `an American elm`, `a European starling`."""
    return ("an " if name[0].casefold() in "aeiou" and not name.casefold().startswith(("eu", "uni", "one")) else "a ") + name


def tail_of(name: str) -> str | None:
    """The name without its first word (`American elm` -> `elm`), when it leaves three letters or more."""
    words = name.split()
    return " ".join(words[1:]) if len(words) > 1 and len(" ".join(words[1:])) >= 3 else None


def held_out(pile: Sequence[Obs], names: dict[str, str], genera_of: dict[str, set[str]], rng: random.Random, n: int) -> list[str]:
    """`n` species (binomials) to hold out of `pile`, drawn: each with a common name, HELD_OBS photos, and the only species of its genus in
    the observations."""
    count: dict[str, int] = defaultdict(int)
    for o in pile:
        count[o.species] += 1
    pool = sorted(s for s, c in count.items() if s in names and not OBSCURE.search(names[s]) and HELD_OBS[0] <= c <= HELD_OBS[1]
                  and len(genera_of[s.split()[0]]) == 1)
    rng.shuffle(pool)
    return pool[:n]


@dataclass(frozen=True)
class Cand:
    template: str
    species: str  # the binomial (`which`: the first of the two)
    obs: tuple[Obs, ...]  # the observations of the evidence, in the order observed
    truth: tuple[str, ...] = ()  # the dates (`days`, `first`, `last`, `ever`), the number (`count`), the name asked (`which`)
    other: str | None = None  # `which`: the second species


def candidates(pile: Sequence[Obs], names: dict[str, str], scale: str) -> list[Cand]:
    """Every question the pile can be asked, with its truth: of the species with a common name and MIN_OBS (of `scale`) to MAX_OBS photos."""
    by: dict[str, list[Obs]] = defaultdict(list)
    for o in pile:
        by[o.species].append(o)
    first_day, last_day = pile[0].day, pile[-1].day
    out: list[Cand] = []
    for species in sorted(by):
        obs = by[species]
        if species not in names or OBSCURE.search(names[species]):
            continue
        if len(obs) == 1 and names[species] in FAMILIAR:
            out.append(Cand("find", species, tuple(obs)))
        if not MIN_OBS[scale] <= len(obs) <= MAX_OBS:
            continue
        days = sorted({o.day for o in obs})
        if DAYS[0] <= len(days) <= DAYS[1]:
            out.append(Cand("days", species, tuple(obs), tuple(days)))
            out.append(Cand("count", species, tuple(obs), (str(len(days)),)))
        if days[0] not in (first_day, last_day):
            at_first = tuple(o for o in obs if o.day == days[0])
            out.append(Cand("first", species, at_first, (days[0],)))
            out.append(Cand("ever", species, at_first, (days[0],)))
        if days[-1] not in (first_day, last_day):
            out.append(Cand("last", species, tuple(o for o in obs if o.day == days[-1]), (days[-1],)))
    return out


def draw(cands: Sequence[Cand], never: Sequence[str], gone: Sequence[str], scale: str, seed: int,
         names: dict[str, str]) -> list[tuple[str, Cand | str]]:
    """The questions of `scale` drawn with `seed`: `(template, candidate)`, and for an absent one `(absent_<template>, name)` of a species of
    `gone` (held-out binomials, all asked) or of `never` (common names, the rest). No two questions are of one species. A `which` is drawn
    from the species of the `ever` questions: two whose first photographs are FORTNIGHT apart or more, neither's name in the other's."""
    rng = random.Random(f"i9:{scale}:{seed}:questions")
    quota = QUOTAS[scale]
    used: set[str] = set()
    out: list[tuple[str, Cand | str]] = []
    by: dict[str, list[Cand]] = defaultdict(list)
    for c in cands:
        by[c.template].append(c)
    absent_total = sum(n for t, n in quota.items() if t.startswith("absent_"))
    if len(gone) > absent_total or len(never) < absent_total - len(gone):
        raise ValueError(f"{len(gone)} held-out species and {len(never)} never-seen ones do not fill {absent_total} absent questions")
    absent_pool = [names[s] for s in gone] + rng.sample(sorted(never), absent_total - len(gone))
    rng.shuffle(absent_pool)
    for template, n in quota.items():
        if template.startswith("absent_"):
            out += [(template, absent_pool.pop()) for _ in range(n)]
            continue
        if template == "which":
            pool = list(by["ever"])
            for _ in range(20000):
                if n == 0:
                    break
                a, b = rng.sample(pool, 2)
                low, high = sorted((a, b), key=lambda c: c.truth[0])
                gap = (date.fromisoformat(high.truth[0]) - date.fromisoformat(low.truth[0])).days
                n1, n2 = names[low.species].casefold(), names[high.species].casefold()
                if gap >= FORTNIGHT and n1 not in n2 and n2 not in n1 and not {low.species, high.species} & used:
                    used |= {low.species, high.species}
                    out.append(("which", Cand("which", low.species, low.obs + high.obs, (names[low.species],), high.species)))
                    n -= 1
        else:
            pool = list(by[template])
            rng.shuffle(pool)
            for c in pool:
                if n == 0:
                    break
                if c.species not in used:
                    used.add(c.species)
                    out.append((template, c))
                    n -= 1
        if n:
            raise ValueError(f"the pile has no more species for a {template} question (seed {seed})")
    return out


def accepted(name: str, other: str) -> str:
    """The phrases an answer naming `name` may hold (`|`): the name, and the name without its first word when `other`'s name has none of
    its words."""
    tail = tail_of(name)
    ok = tail is not None and not set(re.findall(r"[^\W_]+", tail.casefold())) & set(re.findall(r"[^\W_]+", other.casefold()))
    return f"{name.casefold()}|{tail.casefold()}" if ok and tail else name.casefold()


def questions_of(asked: Sequence[tuple[str, Cand | str]], names: dict[str, str], paths: dict[str, str], since: float, until: float, now: float,
                 seed: int) -> tuple[Question, ...]:
    """The questions of `asked` (`draw`'s), over a pile whose observations' dump paths are `paths` (by photo name), asked at `now` (epoch
    seconds) of the span `since` to `until`."""
    out = []
    for template, what in asked:
        base = template.removeprefix("absent_")
        absent = template.startswith("absent_")
        if absent:
            assert isinstance(what, str)
            text, slug = WORDING[base].format(a=a_(what)), re.sub(r"[^a-z0-9]+", "-", what.casefold()).strip("-")
            grade, evidence, value = realdump.grade("absent"), [], None
        else:
            assert isinstance(what, Cand)
            name = names[what.species]
            text = WORDING[base].format(a=a_(name), b=a_(names[what.other]) if what.other else "")
            slug = "-".join(re.sub(r"[^a-z0-9]+", "-", n.casefold()).strip("-") for n in ((name, names[what.other]) if what.other else (name,)))
            evidence = [Span(paths[o.photo], 0.0, 0.0) for o in what.obs]
            if base == "days" or base in ("first", "last", "ever"):
                grade, value = realdump.grade("set", items=what.truth), ", ".join(what.truth)
            elif base == "count":
                grade, value = realdump.grade("count", truth=what.truth[0]), what.truth[0]
            elif base == "find":
                file = paths[what.obs[0].photo].rsplit("/", 1)[-1]
                grade, value = realdump.grade("set", items=[file]), file
            else:
                other = names[what.other]
                grade = realdump.grade("text", truth=accepted(name, other), wrong=accepted(other, name).split("|"))
                value = name
        out.append(realdump.question(
            code=CODE, qid=("absent-" if absent else "") + f"{base}-{slug}", now=now, text=text, kind="absent" if absent else base,
            template=template, modality="image", shape=SHAPES[base], grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since,
            until=until, value=value))
    return realdump.split(out, seed)
