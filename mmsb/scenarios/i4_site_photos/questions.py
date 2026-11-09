"""The questions of I4, every answer from BD3's labels (source.py), each asked the morning after the pile's last photo. The photos are split into
site folders by a seeded draw ("the Maple Street job"), and every question is about one folder, so that a set is a few photos, not hundreds:

| template | question                                                                           | rule                        |
| `photos` | Which photos on the Maple Street job show cracks?                                  | `set` of file names         |
| `count`  | How many photos on the Maple Street job show algae?                                | `count`                     |
| `any`    | Did I photograph any peeling paint on the Maple Street job? If so, which photo     | `set` of one file name      |
|          | was the first?                                                                     |                             |
| `one`    | Find the photo of staining on the Maple Street job.                                | `set` of one file name      |
| `absent` | `photos`, `count` or `any` of a class, about a job folder that holds none of it       | `absent`                    |

What a class is. The dataset labels each photo with one class, which is the photo's own, not the surface's: a photo of one class may show a
little of another (a crack with a stain beside it, spalled concrete with paint peeling at its edge), and some labels are loose. So a question
is "which photos are labelled as this", and an answer found by looking is right only where the label and the eye agree. Three things keep that
from making an answer wrong for no reason of the agent's:
- `minor_crack` and `major_crack` are told apart by a width a person does not measure, so they are asked together, as "cracks";
- a class easily taken for another (algae for stain, peeling for spalling) is never asked beside it: `stain` and `spalling` have no photo in the
  pile at all (HELD) and are asked about in no question, so that what is asked about, cracks, algae and peeling, is never taken for a class in
  the pile;
- an absent question is about a class the pile does hold, in a folder that holds none of it. The folder is chosen first and filled from the pile
  without photos of that class (the photos go to other folders, so no folder is smaller for it, nor is the pile), so that absent and present
  questions ask about the same three things in the same words, and a question cannot be told absent by what it asks. A `one` question is about a
  folder that holds exactly one photo of the class (the others of it are left out of the folder).
`plain` photos (no defect) are the rest of the pile, never asked about.

No question's answer is a time: the dataset has none. A folder is one visit on a seeded day (a weekday of 2025), its photos taken a half to
two minutes apart in the order of their numbers, which are as a camera numbers them, across folders."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i4_site_photos.source import Shot

CODE = "I4"
SOURCE_NAME = "phone"  # the dump's source
FOLDER = "Site photos"
THINGS = {  # what is asked about -> how an inspector says it, and the classes of the dataset it is
    "cracks": ("cracks", ("major_crack", "minor_crack")), "algae": ("algae", ("algae",)), "peeling": ("peeling paint", ("peeling",))}
HELD = {"stain": "algae", "spalling": "peeling"}  # classes with no photo in the pile, and the asked class each is easily taken for
SITES = ("Maple Street", "Harbour Road", "Elm Court", "Kingsley Avenue", "Birch Lane", "Mill Road", "Station Road", "Orchard Close",
         "Willow Way", "Beech Terrace", "Hillcrest Drive", "Church Street", "Albert Square", "Victoria Road", "Meadow Lane", "Cedar Gardens",
         "Park Row", "Fairfield Crescent", "Granary Wharf", "Bridge Street", "Ashford Place", "Linden Road", "Tannery Yard", "Chapel Lane")
SHAPES = {"tiny": (3, 45), "standard": (24, None)}  # the folders and the photos of a scale (None: the whole pile)
QUOTAS = {"tiny": {"photos": 2, "count": 1, "any": 1, "absent_photos": 1, "absent_any": 1},
          "standard": {"photos": 6, "count": 5, "any": 4, "one": 3, "absent_photos": 2, "absent_count": 2, "absent_any": 2}}
MAX_SET = 14  # a `photos` or `count` question is about a folder with no more photos than this of the class
MAX_ONE = 4  # a `one` question is about a folder with no more photos than this of the class (the others are left out)
FILES = "<the names of the files, as in IMG_0042.jpg, separated by commas>"
FILE = "<the name of the file, as in IMG_0042.jpg>"
NUMBER = "<a number>"
VISITS = ("2025-02-03", "2025-11-28")
WORDING = {
    "photos": ("Which photos on the {site} job show {thing}?", "Give me every photo of {thing} on the {site} job."),
    "count": ("How many photos on the {site} job show {thing}?", "How many photos of {thing} did I take on the {site} job?"),
    "any": ("Did I photograph any {thing} on the {site} job? If so, which photo was the first?",),
    "one": ("Find the photo of {thing} on the {site} job.", "Which photo on the {site} job shows {thing}?")}


@dataclass
class Site:
    name: str
    day: date
    shots: list[Shot]  # in the order taken


def classes_of(thing: str) -> tuple[str, ...]:
    return THINGS[thing][1]


def of(site: Site, thing: str) -> list[Shot]:
    return [s for s in site.shots if s.label in classes_of(thing)]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def layout(shots: Sequence[Shot], scale: str, seed: int, days: Sequence[str]) -> tuple[list[Site], list[tuple[str, int, str]]]:
    """The folders of `scale`, drawn with `seed`: the pile (every photo but those of HELD; `tiny` is a seeded few of them) split into folders
    of seeded sizes, each a visit on a seeded weekday of `days`, in time order. And the absent questions, `(template, folder, thing)`: each is
    about a folder of its own and a thing drawn, and that folder is filled from the pile without photos of the thing, so that it holds none of
    it and is no smaller than another."""
    rng = random.Random(f"i4:{scale}:{seed}")
    n, size = SHAPES[scale]
    pile = [s for s in shots if s.label not in HELD]
    if size is not None:
        pile = rng.sample(pile, size)
    rng.shuffle(pile)
    templates = [t.removeprefix("absent_") for t, count in QUOTAS[scale].items() if t.startswith("absent_") for _ in range(count)]
    rng.shuffle(templates)
    absent = [(t, k, rng.choice(sorted(THINGS))) for t, k in zip(templates, rng.sample(range(n), len(templates)), strict=True)]
    weights = [rng.uniform(0.7, 1.3) for _ in range(n)]
    cuts = [round(len(pile) * sum(weights[:k]) / sum(weights)) for k in range(n + 1)]
    forbidden = {k: set(classes_of(thing)) for _, k, thing in absent}
    pool, folders = list(pile), {}
    for k in sorted(range(n), key=lambda k: k not in forbidden):  # the folders that lack a class first, from the whole pool
        take: list[Shot] = []
        rest: list[Shot] = []
        for shot in pool:
            (take if len(take) < cuts[k + 1] - cuts[k] and shot.label not in forbidden.get(k, ()) else rest).append(shot)
        if len(take) != cuts[k + 1] - cuts[k]:
            raise ValueError(f"the pile has too few photos for folder {k} to lack its class (seed {seed})")
        folders[k], pool = take, rest
    for _, k, thing in absent:
        if not any(s.label in classes_of(thing) for j, f in folders.items() if j != k for s in f):
            raise ValueError(f"no other folder holds {thing}, which folder {k} lacks (seed {seed})")
    names = rng.sample(SITES, n)
    first, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    weekdays = [first + timedelta(d) for d in range((last - first).days + 1) if (first + timedelta(d)).weekday() < 5]
    visits = sorted(rng.sample(weekdays, n))
    return [Site(names[k], visits[k], folders[k]) for k in range(n)], absent


def draw(sites: list[Site], absent: Sequence[tuple[str, int, str]], scale: str, seed: int) -> list[tuple[str, int, str, bool]]:
    """The questions of `scale` as `(template, folder, thing, absent)`, drawn with `seed` over `sites`: `layout`'s absent ones and the present
    ones, which this draws, and which change `sites`: a folder a `one` question is about keeps one photo of the class. No folder has more
    than two questions, nor two of one thing."""
    rng = random.Random(f"i4:{scale}:{seed}:questions")
    quota = {t: n for t, n in QUOTAS[scale].items() if not t.startswith("absent_")}
    claimed: dict[int, list[str]] = defaultdict(list)
    for _, k, thing in absent:
        claimed[k].append(thing)
    out: list[tuple[str, int, str, bool]] = [(t, k, thing, True) for t, k, thing in absent]
    jobs = [t for t, n in quota.items() for _ in range(n)]
    rng.shuffle(jobs)
    jobs.sort(key=lambda t: t != "one")  # the `one` questions first: they thin a folder most
    for template in jobs:
        options = []
        for k, site in enumerate(sites):
            for thing in THINGS:
                n = len(of(site, thing))
                fits = {"photos": 2 <= n <= MAX_SET, "count": 2 <= n <= MAX_SET, "any": n >= 1, "one": 1 <= n <= MAX_ONE}[template]
                if fits and len(claimed[k]) < 2 and thing not in claimed[k]:
                    options.append((k, thing))
        if not options:
            raise ValueError(f"no folder can be asked a {template} question (seed {seed})")
        k, thing = rng.choice(options)
        claimed[k].append(thing)
        if template == "one":
            keep = rng.choice(of(sites[k], thing))
            sites[k].shots = [s for s in sites[k].shots if s is keep or s.label not in classes_of(thing)]
        out.append((template, k, thing, False))
    return sorted(out, key=lambda q: (q[1], q[2], q[0]))


def questions_of(sites: Sequence[Site], asked: Sequence[tuple[str, int, str, bool]], paths: dict[str, str], since: float, until: float,
                 now: float, seed: int) -> tuple[Question, ...]:
    """The questions of `asked` (`draw`'s) over `sites` (as `draw` left them), whose photos' dump paths are `paths` (by source path), asked
    at `now` (epoch seconds) of the span `since` to `until`."""
    rng = random.Random(f"i4:{seed}:wording")
    out = []
    for template, k, thing, is_absent in asked:
        site = sites[k]
        found = of(site, thing)
        names = [paths[s.file].rsplit("/", 1)[-1] for s in found]
        text = rng.choice(WORDING[template]).format(site=site.name, thing=THINGS[thing][0])
        if is_absent:
            if found:
                raise ValueError(f"{site.name} has a photo of {thing}, which is asked as absent")
            shape = {"photos": FILES, "count": NUMBER, "any": FILE}[template]
            grade, evidence, value = realdump.grade("absent"), [], None
        elif template in ("photos", "one"):
            shape, grade, value = FILES if template == "photos" else FILE, realdump.grade("set", items=names), ", ".join(names)
            evidence = [Span(paths[s.file], 0.0, 0.0) for s in found]
        elif template == "count":
            shape, grade, value = NUMBER, realdump.grade("count", truth=str(len(found))), str(len(found))
            evidence = [Span(paths[s.file], 0.0, 0.0) for s in found]
        else:
            shape, grade, value = FILE, realdump.grade("set", items=names[:1]), names[0]
            evidence = [Span(paths[found[0].file], 0.0, 0.0)]
        out.append(realdump.question(
            code=CODE, qid=("absent-" if is_absent else "") + f"{template}-{slug(site.name)}-{thing}", now=now, text=text,
            kind="absent" if is_absent else template, template=("absent_" if is_absent else "") + template, modality="image", shape=shape,
            grade=grade, evidence=evidence, sources=(SOURCE_NAME,), since=since, until=until, value=value))
    return realdump.split(out, seed)


def times_of(sites: Sequence[Site], seed: int, t_of) -> list[list[float]]:
    """The time (epoch seconds) of each photo of each folder: the folder's day, from a seeded hour of the morning, a half to two minutes
    between photos; `t_of(wall)` is a wall time's epoch seconds."""
    rng = random.Random(f"i4:{seed}:times")
    out = []
    for site in sites:
        wall = datetime.combine(site.day, time(8)) + timedelta(minutes=rng.randrange(0, 240), seconds=rng.randrange(60))
        row = []
        for _ in site.shots:
            row.append(t_of(wall))
            wall += timedelta(seconds=rng.randrange(30, 121))
        out.append(row)
    return out
