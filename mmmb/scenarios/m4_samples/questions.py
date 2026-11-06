"""The questions of M4, every answer from the labels (source.py), each asked the morning after the folder's last clip:

| template | question                                          | rule                                              |
| `find`   | Find every clip of a toilet flushing. (2 to 8)    | `set` of file names                               |
| `count`  | How many clips of a gong are there? (9 to 40)     | `count`                                           |
| `absent` | `find` of a class whose clips are all left out    | `absent`                                          |

Which sounds are asked about is a short table (CLASSES): leaf classes a person names in plain words, whose sound no other class of the
vocabulary is easily mistaken for (no bells and drums, no vehicles, no speech, no bird or insect: each has a sibling a careful listener
confuses with it), drawn with the seed. The source holds 40 to 130 clips of each, too many to list or count, so a class that is asked about is
thinned to a seeded number of its clips (`find` 2 to 8, `count` 9 to 40): only clips where the raters agreed the sound is predominant stay,
so that a clip is a plain case of its class, and none that holds two of the asked-about classes. An `absent` class loses every clip. The
other clips are the dump's rest: all of them at the standard scale, a seeded draw at `tiny`."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mmmb.core.question import Question, Span
from mmmb.scenarios import realdump
from mmmb.scenarios.m4_samples.source import Clip

CODE = "M4"
SOURCE_NAME = "sounds"  # the dump's source
CLASSES = {  # the class of the vocabulary: how a person says it
    "Toilet_flush": "a toilet flushing", "Microwave_oven": "a microwave oven", "Meow": "a cat meowing", "Purr": "a cat purring",
    "Zipper_(clothing)": "a zipper", "Scissors": "scissors cutting", "Printer": "a printer", "Tambourine": "a tambourine", "Gong": "a gong",
    "Thunder": "thunder", "Hammer": "hammering", "Skateboard": "a skateboard", "Coin_(dropping)": "a coin dropping", "Fart": "a fart",
    "Burping_and_eructation": "a burp", "Packing_tape_and_duct_tape": "packing tape", "Camera": "a camera shutter", "Harp": "a harp",
    "Finger_snapping": "fingers snapping", "Sneeze": "a sneeze", "Chewing_and_mastication": "someone chewing", "Drill": "a power drill",
    "Cough": "a cough", "Writing": "someone writing"}
QUOTAS = {"tiny": {"find": 4, "count": 2, "absent": 2}, "standard": {"find": 11, "count": 6, "absent": 6}}  # the classes of each template
KEEP = {"find": (2, 8), "count": (9, 40)}  # the clips of a class that stay, by the template that asks about it
TINY_CLIPS = 240  # the clips of the `tiny` dump, the classes' clips among them
FILES = "<the names of the files, as in rec_00042.wav, separated by commas>"
NUMBER = "<a number>"


@dataclass(frozen=True)
class Roles:
    find: tuple[str, ...]  # classes (of CLASSES) a `find` question asks about
    count: tuple[str, ...]
    absent: tuple[str, ...]  # classes with no clip in the dump


def slug(label: str) -> str:
    """`Zipper_(clothing)` as `zipper_clothing`: a qid's part."""
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def choose(clips: Sequence[Clip], scale: str, seed: int) -> tuple[Roles, list[Clip]]:
    """The classes the questions of `scale` ask about, drawn with `seed`, and the clips of the dump (by Freesound id): every clip but those
    of the classes asked about, and of each class asked about, its seeded few."""
    quota = QUOTAS[scale]
    drawn = random.Random(f"m4:{seed}:{scale}").sample(sorted(CLASSES), sum(quota.values()))
    roles = Roles(tuple(drawn[:quota["find"]]), tuple(drawn[quota["find"]:quota["find"] + quota["count"]]),
                  tuple(drawn[quota["find"] + quota["count"]:]))
    asked = set(drawn)
    needles: list[Clip] = []
    for template, labels in (("find", roles.find), ("count", roles.count)):
        low, high = KEEP[template]
        for label in labels:
            pool = [c for c in clips if label in c.strong and not (asked - {label}) & set(c.labels)]
            if len(pool) < low:
                raise ValueError(f"only {len(pool)} clips of {label} are plain cases of it with no other asked-about class: {template} needs {low}")
            n = random.Random(f"m4:{seed}:keep:{label}").randint(low, min(high, len(pool)))
            needles += random.Random(f"m4:{seed}:pick:{label}").sample(pool, n)
    others = [c for c in clips if not asked & set(c.labels)]
    if scale == "tiny":
        others = random.Random(f"m4:{seed}:others").sample(others, TINY_CLIPS - len(needles))
    kept = others + needles
    return roles, sorted(kept, key=lambda c: int(c.id))


def questions_of(clips: Sequence[Clip], files: Sequence[realdump.File], roles: Roles, seed: int, now: float) -> tuple[Question, ...]:
    """The questions of the dump `files` (each made from the clip of `clips` at its place), over `roles`; `now` is epoch seconds."""
    since, until = min(f.t for f in files), max(f.t + f.length for f in files)
    with_label: dict[str, list[int]] = {}
    for n, clip in enumerate(clips):
        for label in clip.labels:
            with_label.setdefault(label, []).append(n)

    def ask(template: str, label: str, shape: str, grade, found: Sequence[int], value: str | None) -> Question:
        absent = grade.rule == "absent"
        text = f"Find every clip of {CLASSES[label]}." if template == "find" else f"How many clips of {CLASSES[label]} are there?"
        return realdump.question(code=CODE, qid=("absent-" if absent else "") + f"{template}-{slug(label)}", now=now, text=text,
                                 kind="absent" if absent else template, template="absent_" + template if absent else template,
                                 modality="audio", shape=shape, grade=grade,
                                 evidence=[Span(files[n].path, 0.0, files[n].length) for n in found], sources=(SOURCE_NAME,),
                                 since=since, until=until, value=value)

    out: list[Question] = []
    for label in roles.find:
        found = with_label[label]
        names = [files[n].path.rsplit("/", 1)[-1] for n in found]
        out.append(ask("find", label, FILES, realdump.grade("set", items=names), found, ", ".join(names)))
    for label in roles.count:
        found = with_label[label]
        out.append(ask("count", label, NUMBER, realdump.grade("count", truth=str(len(found))), found, str(len(found))))
    for label in roles.absent:
        if label in with_label:
            raise ValueError(f"{label} is asked about as absent, and the dump holds a clip of it")
        out.append(ask("find", label, FILES, realdump.grade("absent"), [], None))
    return realdump.split(out, seed)
