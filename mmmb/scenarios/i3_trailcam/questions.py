"""The questions of I3, every answer from the labels (source.py), each asked the morning after the card's last photo:

| template      | question                                                                  | rule                    |
| `days`        | On which days (in a month, at the standard scale) did a deer come past?    | `set` of dates          |
| `count`       | On how many different days did a bobcat come past?                         | `count`                 |
| `first`       | When did the camera first photograph a skunk?                              | `time` and date, 60 s   |
| `last`        | When did the camera last photograph a coyote?                              | `time` and date, 60 s   |
| `absent`      | `first` of a large animal the dataset has and the card never shows, or (at | `absent`                |
|               | the standard scale) `days` in a month the species never came               |                         |

A `first` or `last` question is not asked of a species with a photo on its first (or last) day whose hour is 12: the camera writes the
hour after midnight as 12, so which of that day's photos came first is not known. A date is never in doubt. Which ones are asked is drawn
with the seed, a species at a time so that the species vary."""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime, time, timedelta

from mmmb.core.question import Question, Span
from mmmb.scenarios import realdump
from mmmb.scenarios.i3_trailcam.source import EMPTY, MIDNIGHT_HOUR, Photo

CODE = "I3"
SOURCE_NAME = "camera"  # the dump's source
TOL_S = 60.0  # a photo's time is told to the second; a minute either side is the same visit
MAX_DAYS = 8  # a `days` question lists no more dates than this
COUNT_DAYS = (2, 40)  # the days a `count` question's answer is within
QUOTAS = {"tiny": {"days": 2, "count": 1, "first": 2, "last": 1, "absent_first": 2, "absent_days": 0},
          "standard": {"days": 6, "count": 4, "first": 4, "last": 3, "absent_first": 4, "absent_days": 3}}
NOT_ANIMALS = (EMPTY, "car", "insect")  # labels no question asks about
NEVER = ("badger", "cow", "fox", "mountain_lion", "opossum")  # what an absent `first` asks of: large and unlike the animals that came (not
# `cat`, which a bobcat is, nor a bat or a rodent, which a labeller may miss in a frame called empty)
DATES = "<the dates, as YYYY-MM-DD, separated by commas>"
WHEN = "<the date and time, as YYYY-MM-DD HH:MM>"
NUMBER = "<a number>"


def spoken(species: str) -> str:
    """`a deer`, `an opossum`, `a mountain lion`."""
    words = species.replace("_", " ")
    return ("an " if words[0] in "aeiou" else "a ") + words


def month_text(month: str) -> str:
    year, number = month.split("-")
    return f"{realdump.MONTHS[int(number) - 1]} {year}"


def pick(candidates: Sequence[tuple[str, object]], n: int, rng: random.Random) -> list[tuple[str, object]]:
    """`n` of `candidates` (`(species, what)` pairs), drawn a species at a time in a shuffled order so that the species vary."""
    by_species: dict[str, list] = {}
    for c in candidates:
        by_species.setdefault(c[0], []).append(c)
    for of in by_species.values():
        rng.shuffle(of)
    order = sorted(by_species)
    rng.shuffle(order)
    out: list = []
    while len(out) < n and any(by_species.values()):
        for species in order:
            if by_species[species] and len(out) < n:
                out.append(by_species[species].pop())
    return out


def questions_of(photos: Sequence[Photo], paths: Sequence[str], names: Sequence[str], scale: str, seed: int,
                 t_of) -> tuple[Question, ...]:
    """The questions of the card `photos` (in order, with their dump `paths`), over the dataset's label `names`, at `scale`, drawn with
    `seed`; `t_of(wall)` is a wall time's epoch seconds."""
    rng = random.Random(f"i3:{seed}")
    quota = QUOTAS[scale]
    path = {p.file: x for p, x in zip(photos, paths, strict=True)}
    animals = [p for p in photos if p.species not in NOT_ANIMALS]
    seen = sorted({p.species for p in animals})
    last_day = photos[-1].wall.date()
    now = t_of(datetime.combine(last_day + timedelta(days=1), time(9)))
    since, until = t_of(photos[0].wall), t_of(photos[-1].wall) + 1
    months = sorted({p.wall.strftime("%Y-%m") for p in photos}) if scale != "tiny" else []

    def of(species: str, month: str | None = None) -> list[Photo]:
        return [p for p in animals if p.species == species and (month is None or p.wall.strftime("%Y-%m") == month)]

    def days_of(found: Sequence[Photo]) -> list[str]:
        return sorted({p.wall.date().isoformat() for p in found})

    def ask(qid: str, text: str, kind: str, shape: str, grade, found: Sequence[Photo], value: str | None) -> Question:
        template = qid.split("-")[0] if kind != "absent" else "absent_" + qid.split("-")[1]
        return realdump.question(code=CODE, qid=qid, now=now, text=text, kind=kind, template=template, modality="image",
                                 shape=shape, grade=grade, evidence=[Span(path[p.file], 0.0, 0.0) for p in found],
                                 sources=(SOURCE_NAME,), since=since, until=until, value=value)

    out: list[Question] = []
    # days: the dates a species came, in a month (standard) or on the card (tiny)
    scopes = [(s, m) for s in seen for m in (months or [None])]
    days = [(s, m) for s, m in scopes if 1 <= len(days_of(of(s, m))) <= MAX_DAYS]
    for species, month in pick(days, quota["days"], rng):
        found = of(species, month)
        items = days_of(found)
        where = f" in {month_text(month)}" if month else ""
        out.append(ask(f"days-{species}" + (f"-{month}" if month else ""),
                       f"On which days{where} did {spoken(species)} come past my trail camera?", "days", DATES,
                       realdump.grade("set", items=items), found, ", ".join(items)))
    # count: on how many days a species came, over the card
    counts = [(s, None) for s in seen if COUNT_DAYS[0] <= len(days_of(of(s))) <= COUNT_DAYS[1]]
    for species, _ in pick(counts, quota["count"], rng):
        found = of(species)
        n = str(len(days_of(found)))
        out.append(ask(f"count-{species}", f"On how many different days did {spoken(species)} come past my trail camera?", "count",
                       NUMBER, realdump.grade("count", truth=n), found, n))
    # first and last: when a species was first (last) photographed, when the hour of no photo of it on that day is in doubt
    for template, end in (("first", 0), ("last", -1)):
        sure = []
        for s in seen:
            day = of(s)[end].wall.date()
            if all(p.wall.hour != MIDNIGHT_HOUR for p in of(s) if p.wall.date() == day):
                sure.append((s, None))
        for species, _ in pick(sure, quota[template], rng):
            photo = of(species)[end]
            clock = photo.wall.strftime("%H:%M:%S")
            out.append(ask(f"{template}-{species}", f"When did my trail camera {template} photograph {spoken(species)}?", template,
                           WHEN, realdump.grade("time", time=clock, tol_s=TOL_S, date=photo.wall.date().isoformat()), [photo],
                           f"{photo.wall.date().isoformat()} {clock}"))
    # absent: a species of the dataset the card never shows; a month a species of the card never came in
    never = [(s, None) for s in NEVER if s in names and s not in seen]
    for species, _ in pick(never, quota["absent_first"], rng):
        out.append(ask(f"absent-first-{species}", f"When did my trail camera first photograph {spoken(species)}?", "absent", WHEN,
                       realdump.grade("absent"), [], None))
    missed = [(s, m) for s in seen for m in months if not of(s, m)]
    for species, month in pick(missed, quota["absent_days"], rng):
        out.append(ask(f"absent-days-{species}-{month}", f"On which days in {month_text(month)} did {spoken(species)} come past my trail "
                       "camera?", "absent", DATES, realdump.grade("absent"), [], None))
    return realdump.split(out, seed)

