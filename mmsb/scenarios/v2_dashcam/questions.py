"""The questions of V2, every answer from the CSV (source.py), each asked the morning after the card's last clip. The card is small: about
a hundred clips, half an hour of video, so a question can ask for the clips themselves, and every answer is a set of them, a count, or a
word.

| template  | question                                                                  | rule                                  |
| `crashes` | Show me every clip with a crash at night.                                  | `set` of file names (2 to 12)         |
| `near`    | Show me every near miss at a roundabout.                                   | `set` of file names (2 to 16)         |
| `count`   | How many crashes did the dashcam catch on a highway? (2 to 40)             | `count`                               |
| `which`   | Which clip shows a crash in a car park?                                    | `set` of one file name                |
| `did`     | Did the dashcam ever catch a near miss in the rain? If so, which clip?     | `set` of file names (1 to 3)          |
| `what`    | What time of day was the crash at the roundabout?                          | `text`: day or night                  |
| `absent_*`| `crashes`, `near`, `count` or `did` of a place and weather the card never  | `absent`                              |
|           | has a crash or near miss in                                                |                                       |

What a question rests on is the clip's class, `time`, `weather` and `location`, and only the values that can be told at a glance: night,
rain, a highway, a roundabout, a tunnel and a car park. Clear and cloudy, and a residential street, a city street, a rural road and an
intersection, run into one another, so none is asked. `ego` and `hazard` are not asked either: what they mean is not told. Nothing is asked
about a person: the clips are of vehicles and the road.

What is left out of the card, as M4 leaves out the sounds easily taken for the ones it asks about:
- every clip of the class `Outliers`, the source's catch-all for what is neither a crash, a near miss nor an ordinary drive, which a
  viewer could as well take for one of the two;
- of each place and weather that an absent question asks about (ABSENT), every crash and near miss there (a near miss at a roundabout in the
  rain is taken for a crash there). The CSV is complete clip by clip, so what is left holds none.
A crash and a near miss are the source's own judgement of what happened (a clip is one or the other); the questions never ask for both
at once. Which questions are drawn is by the seed; the card does not change with it."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.v2_dashcam.source import Clip

CODE = "V2"
SOURCE_NAME = "dashcam"  # the dump's source
OUTLIERS = "Outliers"
EVENTS = {"Crash": ("crash", "a crash", "crashes"), "Near Crash": ("near", "a near miss", "near misses")}  # class -> key, one, many
KINDS = {**EVENTS, "Normal": ("normal", "an ordinary drive", "ordinary drives")}  # the classes a question names; Normal only in `count` and `which`
ATOMS = {  # what can be told at a glance: key -> (the clip's field, its value, how it is said)
    "highway": ("location", "Highway", "on a highway"), "roundabout": ("location", "Roundabouts", "at a roundabout"),
    "carpark": ("location", "Parking Lot", "in a car park"), "tunnel": ("location", "Tunnel", "in a tunnel"),
    "rain": ("weather", "Rain", "in the rain"), "night": ("time", "Night", "at night")}
ORDER = tuple(ATOMS)  # a phrase puts the place first, then the weather, then the time: "on a highway in the rain at night"
ABSENT = (("rain", "night"), ("carpark", "night"), ("highway", "rain"), ("roundabout", "rain"), ("carpark", "rain"))
QUOTAS = {"tiny": {"crashes": 1, "near": 0, "count": 1, "which": 1, "did": 1, "what": 1, "absent_crashes": 1, "absent_near": 0,
                   "absent_count": 0, "absent_did": 1},
          "standard": {"crashes": 3, "near": 3, "count": 3, "which": 3, "did": 3, "what": 2, "absent_crashes": 1, "absent_near": 1,
                       "absent_count": 1, "absent_did": 2}}
SET_SIZE = {"crashes": (2, 12), "near": (2, 16), "did": (1, 3)}  # the clips a set answer holds
COUNT_SIZE = (2, 40)
FILES = "<the names of the files, as in 2025_1201_081530_001.MP4, separated by commas>"
NUMBER = "<a number>"
WORD = "<day or night>"
# the words that say the time of day of a clip, as `text` accepts them: the clip's own, and the other's, which no right answer holds. A word a right
# answer may use in passing ("daytime, inside a dark tunnel", "night, in bright headlights") is in the first list only.
SAYS = {"Day": ("day|daytime|daylight|morning|afternoon|midday|noon|sunny|bright", "night|nighttime|evening|dusk|dawn"),
        "Night": ("night|nighttime|dark|after dark|evening|dusk", "day|daytime|daylight|morning|afternoon|midday|noon")}


@dataclass(frozen=True)
class Item:
    """A question that can be asked of the card: its template, the class it asks about (a key of EVENTS, or None for any clip), its
    qualifier (keys of ATOMS), and the clips it is about."""

    template: str
    kind: str | None
    atoms: tuple[str, ...]
    clips: tuple[Clip, ...]


def matches(clip: Clip, atoms: Sequence[str]) -> bool:
    return all(getattr(clip, ATOMS[a][0]) == ATOMS[a][1] for a in atoms)


def said(atoms: Sequence[str]) -> str:
    """`on a highway in the rain at night`: the qualifier as a person says it."""
    return " ".join(ATOMS[a][2] for a in ORDER if a in atoms)


def held_out(clip: Clip) -> bool:
    """A clip the card does not hold: of the class Outliers, or a crash or near miss in a place and weather an absent question asks about."""
    return clip.kind == OUTLIERS or clip.kind in EVENTS and any(matches(clip, atoms) for atoms in ABSENT)


def combos() -> list[tuple[str, ...]]:
    """The empty qualifier, every qualifier of one atom, and every of two atoms of different fields (a place, a weather, a time), in ORDER."""
    out: list[tuple[str, ...]] = [()]
    for i, a in enumerate(ORDER):
        out.append((a,))
        out += [(a, b) for b in ORDER[i + 1:] if ATOMS[a][0] != ATOMS[b][0]]
    return out


def library(card: Sequence[Clip]) -> dict[str, list[Item]]:
    """Every question the card `card` (what is left of the source) can be asked, by template: a set, count, which, did and what item for
    each class and qualifier whose clips are as many as the template takes, and an absent item for each class and place and weather of
    ABSENT that the card has all the parts of, and no crash or near miss in."""
    out: dict[str, list[Item]] = {}
    seen = {a for clip in card for a in ATOMS if matches(clip, (a,))}
    for kind, (key, _, _) in KINDS.items():
        for atoms in combos():
            found = tuple(c for c in card if c.kind == kind and matches(c, atoms))
            for template, (low, high) in SET_SIZE.items():
                if kind in EVENTS and template in ("crashes", "near") and template != ("crashes" if key == "crash" else "near") or kind not in EVENTS:
                    continue
                if low <= len(found) <= high and (atoms or template != "did"):
                    out.setdefault(template, []).append(Item(template, kind, atoms, found))
            if COUNT_SIZE[0] <= len(found) <= COUNT_SIZE[1] and (atoms or kind != "Near Crash"):
                out.setdefault("count", []).append(Item("count", kind, atoms, found))
            if len(found) == 1 and atoms and not any(len(atoms) > 1 and sum(c.kind == kind and matches(c, (a,)) for c in card) == 1 for a in atoms):
                out.setdefault("which", []).append(Item("which", kind, atoms, found))  # asked by as little as it takes: a single atom, or a pair
                if "night" not in atoms and kind in EVENTS:
                    out.setdefault("what", []).append(Item("what", kind, atoms, found))
        for atoms in ABSENT if kind in EVENTS else ():
            if set(atoms) <= seen and not any(c.kind == kind and matches(c, atoms) for c in card):
                for template in ("absent_crashes" if key == "crash" else "absent_near", "absent_count", "absent_did"):
                    out.setdefault(template, []).append(Item(template, kind, atoms, ()))
    for atoms in combos()[1:]:  # any clip: how many were filmed so, and which clip is the only one
        found = tuple(c for c in card if matches(c, atoms))
        if COUNT_SIZE[0] <= len(found) <= COUNT_SIZE[1]:
            out.setdefault("count", []).append(Item("count", None, atoms, found))
        if len(found) == 1 and not any(len(atoms) > 1 and sum(matches(c, (a,)) for c in card) == 1 for a in atoms):
            out.setdefault("which", []).append(Item("which", None, atoms, found))
    for template in ("which",):  # `Which clip was filmed in a car park?` and `Which clip shows a crash in a car park?`: one question of the clip
        anyone = {(i.atoms, i.clips) for i in out.get(template, []) if i.kind is None}
        out[template] = [i for i in out.get(template, []) if i.kind is None or (i.atoms, i.clips) not in anyone]
    return out


def text_of(item: Item) -> str:
    """The question of `item`, as a person asks it of their own dashcam card."""
    one, many = (KINDS[item.kind][1], KINDS[item.kind][2]) if item.kind else ("", "")
    where = (" " + said(item.atoms)) if item.atoms else ""
    template = item.template.removeprefix("absent_")
    if template in ("crashes", "near"):
        return f"Show me every clip with {one}{where}." if item.kind == "Crash" else f"Show me every near miss{where}."
    if template == "count":
        return f"How many clips were filmed{where}?" if item.kind is None else f"How many {many} did the dashcam catch{where}?"
    if template == "which":
        return f"Which clip was filmed{where}?" if item.kind is None else f"Which clip shows {one}{where}?"
    if template == "did":
        return f"Did the dashcam ever catch {one}{where}? If so, which clip?"
    return f"What time of day was the {'crash' if item.kind == 'Crash' else 'near miss'}{where}?"


def questions_of(card: Sequence[Clip], paths: Sequence[str], scale: str, seed: int, now: float, since: float, until: float) -> tuple[Question, ...]:
    """The questions of the card `card` (each clip's dump path in `paths`), drawn with `seed`; `now` is epoch seconds."""
    path = {c.name: p for c, p in zip(card, paths, strict=True)}
    pools = library(card)
    rng = random.Random(f"v2:{seed}:{scale}")
    used: set[object] = set()  # what is already asked of (the clips; or class and qualifier, when there are none), so that the questions differ
    out: list[Question] = []
    for template, n in QUOTAS[scale].items():
        pool = sorted(pools.get(template, []), key=lambda i: (i.kind or "", i.atoms))
        rng.shuffle(pool)
        pool.sort(key=lambda i: about(i) in used)  # the unused first (a stable sort keeps the shuffle)
        for item in pool[:n]:
            if about(item) in used and template not in ("did", "what"):
                continue  # a set already asked of: no other question that names clips repeats it (`did` and `what` ask of one again)
            used.add(about(item))
            out.append(ask(item, path, now, since, until))
    return realdump.split(out, seed)


def about(item: Item) -> object:
    """What `item` is about: the clips it names, or, for an absent one, its class and qualifier."""
    return frozenset(c.name for c in item.clips) or (item.kind, item.atoms)


def ask(item: Item, path: dict[str, str], now: float, since: float, until: float) -> Question:
    absent = item.template.startswith("absent_")
    template = item.template.removeprefix("absent_")
    names = [path[c.name].rsplit("/", 1)[-1] for c in item.clips]
    spans = [Span(path[c.name], 0.0, c.seconds) for c in item.clips]
    key = KINDS[item.kind][0] if item.kind else "clip"
    qid = f"{'absent-' if absent else ''}{template}-{key}-{'_'.join(item.atoms) or 'any'}"
    if absent:
        grade, shape, value = realdump.grade("absent"), {"count": NUMBER}.get(template, FILES), None
    elif template == "count":
        grade, shape, value = realdump.grade("count", truth=str(len(item.clips))), NUMBER, str(len(item.clips))
    elif template == "what":
        accepted, other = SAYS[item.clips[0].time]
        grade, shape, value = realdump.grade("text", truth=accepted, wrong=other.split("|")), WORD, item.clips[0].time
    else:
        grade, shape, value = realdump.grade("set", items=names), FILES, ", ".join(names)
    return realdump.question(code=CODE, qid=qid, now=now, text=text_of(item), kind="absent" if absent else template, template=item.template,
                             modality="video", shape=shape, grade=grade, evidence=spans, sources=(SOURCE_NAME,), since=since, until=until,
                             value=value)
