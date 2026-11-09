"""The pile of V10: which of the source's videos go into the dump (about 3,000 clips at the standard scale), and what the questions can be about.

The labels are not exhaustive: an action may happen in a clip that has no label for it (and no word of it in the script or the descriptions
either). So a question is asked only of what the pile is built to make certain, as M4's is, by holding out of the dump every clip that is
unsure for a topic that is asked about:

- A TOPIC is an action of the kind a person films ("washing a window"): the classes that are it, a regex of how the script and descriptions say
  it, the classes taken for it (`near`) and, for a topic that is never in the pile, the script objects it is done with.
- For a clip and a topic: it is a POSITIVE when it has the label and the words say it too; AMBIGUOUS when it has only one of the two, or a class taken
  for it, or (of a topic that is never in the pile) its object in the script; else NONE.
- The pile holds, of a topic, only its positives (one clip of a `one` topic, a few of a `few` topic, a dozen or more of a `many` topic: what
  `choose` draws, with a seed of its own, so that the pile does not change with the seed that draws the questions) and clips that are NONE for
  it. A clip that is positive for one topic is NONE for every other. A `none` topic has no clip.
- A clip is left out whose words and scene say different rooms (a scene asked about as a room, `ROOMS`): the scene is the one complete label, but
  a script that puts the clip in another room makes it unsure, so a clip of one of the five rooms asked about, and no other, names it.
- Only a clip an annotator verified, of watchable quality, between 8 and 45 seconds, and with at least one action, is a candidate at all.

The five rooms (Stairs, Garage, Bathroom, Basement, Laundry room) are those whose scene and script agree in nine clips of ten or more (the scene
of a kitchen, a bedroom or a living room often disagrees with a script that names a pantry, a closet or a hall): each is thinned to a seeded 12 to
30 clips (4 to 7 at tiny), so that how many clips were filmed in one can be counted."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from mmsb.scenarios.v10_home_clips.source import Clip

STANDARD, TINY = "standard", "tiny"
TOTAL = {STANDARD: 3000, TINY: 72}  # the clips of the pile
LENGTH = (8.0, 45.0)  # seconds a candidate clip plays
QUALITY = 3  # the least a candidate's quality (an annotator's 1 to 7)
KEEP = {STANDARD: {"few": (2, 6), "many": (12, 30), "room": (12, 30)}, TINY: {"few": (2, 3), "many": (5, 8), "room": (4, 7)}}
ROLES = {STANDARD: {"one": 7, "few": 5, "many": 5, "none": 5, "rooms": 5}, TINY: {"one": 3, "few": 1, "many": 1, "none": 2, "rooms": 2}}  # topics at each role
LONG_S = 8.0  # the shortest span a `long` question asks the length of
WHOLE = 0.85  # and the longest, as a share of its clip (a span that is the clip's whole length is told by the file's duration)
GAP_S = 2.0  # the clear gap between the two actions of a `before` question: the one ends at least this long before the other starts


@dataclass(frozen=True)
class Topic:
    key: str  # a qid's part
    role: str  # one, few, many or none: how many clips of it the pile holds
    phrase: str  # "washing a window": after "me", "I was", "before or after"
    labels: tuple[str, ...]  # the classes that are it
    text: str  # a regex of how the script and descriptions say it
    near: tuple[str, ...] = ()  # classes taken for it
    objects: tuple[str, ...] = ()  # of a `none` topic: the script objects it is done with, which keep a clip out of the pile


TOPICS = {t.key: t for t in (
    # one clip each
    Topic("window_wash", "one", "washing a window", ("c091",), r"\b(wash|clean|wip|scrub|polish|squeegee)\w*\b[^.;]{0,40}\bwindows?\b"),
    Topic("mirror_wash", "one", "cleaning a mirror", ("c095",), r"\b(wash|clean|wip|scrub|polish)\w*\b[^.;]{0,40}\bmirror"),
    Topic("light_fix", "one", "fixing a light", ("c103",), r"\bfix\w*\b[^.;]{0,30}\blight|\b(chang|replac)\w*\b[^.;]{0,30}\b(light ?bulb|bulb)"),
    Topic("sandwich_make", "one", "making a sandwich", ("c066",), r"\b(mak|prepar|assembl)\w*\b[^.;]{0,30}\bsandwich"),
    Topic("chair_stand", "one", "standing on a chair", ("c060",), r"\bstand\w*\b[^.;]{0,20}\bon\b[^.;]{0,10}\bchair"),
    Topic("clothes_wash", "one", "washing some clothes", ("c005",), r"\b(wash|launder)\w*\b[^.;]{0,30}\bclothes|\bdo(es|ing)? (the |some |a )?laundry|\blaunder(s|ed|ing)?\b"),
    Topic("doorknob_fix", "one", "fixing a doorknob", ("c140",), r"\b(fix|repair|tighten)\w*\b[^.;]{0,30}\bdoor ?knob"),
    # a few clips
    Topic("window_open", "few", "opening a window", ("c090",), r"\bopen\w*\b[^.;]{0,30}\bwindow"),
    Topic("window_close", "few", "closing a window", ("c089",), r"\bclos\w*\b[^.;]{0,30}\bwindow"),
    Topic("shoes_on", "few", "putting on my shoes", ("c055",),
          r"\b(put|putting|puts|slip|slips|slipping)\w*\b[^.;]{0,15}\bon\b[^.;]{0,20}\bshoes?|\bshoes?\b[^.;]{0,20}\bon\b"),
    Topic("box_open", "few", "opening a box", ("c041",), r"\bopen\w*\b[^.;]{0,30}\bbox"),
    Topic("floor_lie", "few", "lying on the floor", ("c124",), r"\b(lay|lie|lies|lying|laying)\b[^.;]{0,30}\bfloor"),
    Topic("phone_talk", "few", "talking on the phone", ("c019",), r"\b(talk\w*|speak\w*|call\w*)\b[^.;]{0,20}\b(phone|cellphone|camera)"),
    # a dozen or more
    Topic("sneeze", "many", "sneezing", ("c153",), r"\bsneez\w*"),
    Topic("hands_wash", "many", "washing my hands", ("c139",), r"\b(wash|rinse|scrub)\w*\b[^.;]{0,30}\bhands?\b"),
    Topic("cooking", "many", "cooking", ("c147",), r"\bcook\w*"),
    Topic("dishes_wash", "many", "washing the dishes", ("c121",), r"\b(wash|clean|rinse|scrub)\w*\b[^.;]{0,30}\b(dish|dishes|plate|plates)\b", ("c111",)),
    Topic("dressing", "many", "getting dressed", ("c148",), r"\bdress(es|ing|ed)?\b|\bgets? dressed"),
    # none
    Topic("fridge_open", "none", "opening the fridge", ("c143",), r"\b(refrigerator|fridge)\b", ("c142",), ("refrigerator",)),
    Topic("medicine_take", "none", "taking some medicine", ("c129",), r"\b(medicine|medication|pills?)\b", ("c128",), ("medicine",)),
    Topic("hair_fix", "none", "fixing my hair", ("c144",), r"\bhair\b", (), ("hair",)),
    Topic("vacuum_use", "none", "using the vacuum cleaner", ("c136",), r"\bvacuum\w*", ("c137", "c138"), ("vacuum",)),
    Topic("tv_laugh", "none", "laughing at the TV", ("c131",), r"\b(tv|television)\b", ("c132",), ("television", "tv")),
)}

# the five rooms asked about: scene -> how a person says where, the words that name it in an answer, and a regex of how a script names it
ROOMS = {
    "Stairs": ("on the stairs", "stairs|staircase|stairway|steps", r"\bstair"),
    "Garage": ("in the garage", "garage", r"\bgarage"),
    "Bathroom": ("in the bathroom", "bathroom|bath room|toilet|restroom|washroom", r"\bbath ?room|\brestroom|\btoilet"),
    "Basement": ("in the basement", "basement|cellar", r"\bbasement"),
    "Laundry room": ("in the laundry room", "laundry|utility room", r"\blaundry"),
}
OTHER_ROOMS = {  # how a script names the other rooms: only to tell that a clip's words and its scene disagree
    "Kitchen": r"\bkitchen", "Bedroom": r"\bbed ?room", "Living room": r"\bliving ?room", "Home Office": r"\boffice|\bstudy\b",
    "Entryway": r"\bentry ?way|\bentrance|\bfoyer", "Dining room": r"\bdining", "Hallway": r"\bhall\b|\bhallway", "Closet": r"\bcloset|\bwardrobe",
    "Recreation room": r"\brecreation|\brec room|\bman cave", "Pantry": r"\bpantry"}

# the actions a `before` question may name beside a topic's: a class that is plain to see and said in one phrase, as "was I {phrase} before or after"
NAMED = {
    "c153": "sneezing", "c150": "running", "c148": "getting dressed", "c133": "waking up", "c106": "drinking", "c132": "watching TV",
    "c019": "talking on the phone", "c156": "eating", "c154": "standing up", "c151": "sitting down", "c104": "turning on the light",
    "c105": "turning off the light", "c139": "washing my hands", "c097": "walking through a doorway",
}


def eligible(c: Clip) -> bool:
    """A clip that can be in the pile at all: verified, with no action the CSV gets wrong, watchable, of a length a phone clip has, with an action, and
    not of the scene `Other`."""
    return c.verified and not c.flawed and c.quality >= QUALITY and LENGTH[0] <= c.length <= LENGTH[1] and bool(c.spans) and c.scene != "Other"


def named_rooms(c: Clip) -> set[str]:
    """The rooms the clip's script and descriptions name."""
    return {room for room, rx in ((r, v[2]) for r, v in ROOMS.items()) if re.search(rx, c.text)} | {
        room for room, rx in OTHER_ROOMS.items() if re.search(rx, c.text)}


def scene_clean(c: Clip) -> bool:
    """Whether the clip's words and its scene agree about the five rooms: a clip of one of them names no other room, and a clip of another
    scene names none of the five."""
    named = named_rooms(c)
    return all((named <= {c.scene}) if c.scene == room else room not in named for room in ROOMS)


def scene_said(c: Clip) -> bool:
    """Whether the words confirm the scene of a clip of one of the five rooms: they name it, and no other."""
    return c.scene in ROOMS and named_rooms(c) == {c.scene}


def status(c: Clip, t: Topic, regex: dict[str, re.Pattern]) -> str | None:
    """`pos`, `amb` or None: what the clip is for the topic."""
    label = any(x in c.classes() for x in t.labels)
    words = bool(regex[t.key].search(c.text))
    if t.role == "none":
        return "amb" if label or words or any(x in c.classes() for x in t.near) or any(o in c.objects for o in t.objects) else None
    if label and words and not any(x in c.classes() for x in t.near):
        return "pos"
    return "amb" if label or words or any(x in c.classes() for x in t.near) else None


def long_enough(c: Clip, span: tuple[float, float]) -> bool:
    """Whether a `long` question may ask the length of `span` in the clip: at least LONG_S, and not (nearly) the whole clip."""
    return span[1] - span[0] >= LONG_S and span[1] - span[0] <= WHOLE * c.length


def span_of(c: Clip, label: str) -> tuple[float, float] | None:
    """The one span of the class `label` in the clip, or None when it has none or several."""
    found = [(a, b) for x, a, b in c.spans if x == label]
    return found[0] if len(found) == 1 else None


def order_of(c: Clip, label: str) -> tuple[str, str, tuple[float, float]] | None:
    """Of the clip's one action of `label`, another labelled action of NAMED that is plain to order with it, as `(class, "before" or "after", its
    span)`: it has one span in the clip, which ends GAP_S or more before the first starts (`before`) or starts GAP_S or more after it ends (`after`).
    The widest such gap wins, the lowest class number among equals; None when there is none. Both spans are labels, so the order is certain."""
    one = span_of(c, label)
    if one is None:
        return None
    found = []
    for cls in NAMED:
        two = span_of(c, cls)
        if cls == label or two is None:
            continue
        if one[0] - two[1] >= GAP_S:
            found.append((one[0] - two[1], cls, "before", two))
        elif two[0] - one[1] >= GAP_S:
            found.append((two[0] - one[1], cls, "after", two))
    if not found:
        return None
    _, cls, side, two = min(found, key=lambda f: (-f[0], f[1]))
    return cls, side, two


@dataclass(frozen=True)
class Pile:
    clips: tuple[Clip, ...]  # the pile, by id
    one: dict[str, Clip]  # a `one` topic -> its clip
    few: dict[str, tuple[Clip, ...]]
    many: dict[str, tuple[Clip, ...]]
    none: tuple[str, ...]  # the `none` topics
    rooms: tuple[str, ...]  # the scenes asked about
    held: dict[str, int] = field(default_factory=dict)  # how many candidates each hold-out took out, for the report


def choose(clips: Sequence[Clip], scale: str) -> Pile:
    """The pile of `scale`: its topics drawn by role, their clips, the rooms, and the rest, all by a seed of the scale's own."""
    rng = random.Random(f"v10:{scale}")
    regex = {k: re.compile(t.text, re.IGNORECASE) for k, t in TOPICS.items()}
    quota = ROLES[scale]
    by_role = {role: sorted(k for k, t in TOPICS.items() if t.role == role) for role in ("one", "few", "many", "none")}
    drawn = {role: sorted(rng.sample(keys, quota[role])) for role, keys in by_role.items()}
    asked = [k for keys in drawn.values() for k in keys]
    pool = [c for c in clips if eligible(c) and scene_clean(c)]
    held = {"not a candidate": len(clips) - len(pool)}
    # only the topics of the pile are asked about, so only they hold clips out
    state = {c.id: {k: status(c, TOPICS[k], regex) for k in asked} for c in pool}

    def clean(c: Clip, besides: str | None = None) -> bool:
        return all(v is None for k, v in state[c.id].items() if k != besides)

    chosen: dict[str, tuple[Clip, ...]] = {}
    used: set[str] = set()
    scenes_used: dict[str, int] = {}
    for key in drawn["one"]:
        t = TOPICS[key]
        found = [c for c in pool if state[c.id][key] == "pos" and clean(c, key) and c.id not in used and span_of(c, t.labels[0]) is not None]
        if not found:
            raise ValueError(f"no clip is a plain case of {t.phrase}")

        def fit(c: Clip, t: Topic = t) -> float:
            """What a `one` clip can be asked besides `find` and `did`: its span's length, its room, whether another action was before or after it;
            less for a room that an earlier topic's clip is in already (so that the rooms vary)."""
            return (2 * long_enough(c, span_of(c, t.labels[0])) + 3 * scene_said(c) + 3 * (order_of(c, t.labels[0]) is not None)
                    - 2 * scenes_used.get(c.scene, 0) + rng.random())
        best = max(found, key=fit)
        chosen[key] = (best,)
        used.add(best.id)
        scenes_used[best.scene] = scenes_used.get(best.scene, 0) + 1
    for role in ("few", "many"):
        for key in drawn[role]:
            found = sorted((c for c in pool if state[c.id][key] == "pos" and clean(c, key) and c.id not in used), key=lambda c: c.id)
            low, high = KEEP[scale][role]
            n = min(rng.randint(low, high), len(found))
            if n < low:
                raise ValueError(f"only {len(found)} clips are plain cases of {TOPICS[key].phrase}: a {role} topic needs {low}")
            chosen[key] = tuple(rng.sample(found, n))
            used |= {c.id for c in chosen[key]}
    needles = [c for cs in chosen.values() for c in cs]
    rooms = tuple(sorted(rng.sample(sorted(ROOMS), quota["rooms"])))
    taken = {c.id for c in needles}
    # the rooms: each thinned to a seeded number of clips, the topics' own among them
    left = [c for c in pool if clean(c) and c.id not in taken]
    for room in rooms:
        have = sum(c.scene == room for c in needles)
        low, high = KEEP[scale]["room"]
        want = max(rng.randint(low, high), have + 3)
        more = rng.sample(sorted((c for c in left if c.scene == room), key=lambda c: c.id), max(0, want - have))
        taken |= {c.id for c in more}
        needles += more
    rest = sorted((c for c in left if c.id not in taken and c.scene not in ROOMS), key=lambda c: c.id)
    if len(rest) < TOTAL[scale] - len(needles):
        raise ValueError(f"only {len(rest)} clips are left to fill a pile of {TOTAL[scale]}, of which {len(needles)} are chosen")
    kept = needles + rng.sample(rest, TOTAL[scale] - len(needles))
    held["left out by a topic"] = sum(not clean(c) for c in pool)
    return Pile(tuple(sorted(kept, key=lambda c: c.id)), {k: chosen[k][0] for k in drawn["one"]}, {k: chosen[k] for k in drawn["few"]},
                {k: chosen[k] for k in drawn["many"]}, tuple(drawn["none"]), rooms, held)
