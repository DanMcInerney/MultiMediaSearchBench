"""The week being planned: `_Week` makes the files, log lines and questions of a seed over its days, one source after the other with a
random stream of its own for each, and `plan(seed, scale, days)` runs it and checks the result (`plan.verify`). What a plan holds is in
plan.py. Moved from haystack.py of the repository at bb498a6."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from datetime import date, datetime

from mmsb.core import grade
from mmsb.scenarios.v7_screen.media_common import DISTRACTOR_RGB, NEEDLE_RGB, _rng
from mmsb.scenarios.v7_screen.media_photo import _backdrop
from mmsb.scenarios.v7_screen.media_porch import parcel_box
from mmsb.scenarios.v7_screen.media_screen import APPS, CURSOR_HOME, dialog_at
from mmsb.scenarios.v7_screen.media_sound import ALARM_DUR, BED_DBFS, CHIME_DUR, DOORBELL_PAIRS, LOUD_GAIN
from mmsb.scenarios.v7_screen.plan import (
    ABSENT, ABSENT_OF, BANDS, DAY_END, DAY_START, FENCE_RGB, HOUR, INFO, MEAN_STATE_S, MIC_GAP_S, PART_WEIGHTS, PARTS, SCALES, SERVICES,
    TEMPLATES, TYPES, VERSION, File, Plan, Spot, arrange, at, clip_length, clip_range, clock, cut, date_text, next_level, part_of,
    segments, share_out, slot_key, spaced, state_acts, verify,
)
from mmsb.scenarios.v7_screen.wording import (
    ERRORS, ORACLES, OTHER_CARS, OTHER_SIGNS, REASONS, SIGN_WORDS, TOPICS, WARNINGS, WEEK_WORDS, WORDS, apart, background_line,
    distractor_note, grade_of, shade_of,
)


class _Week:
    """The plan being made: a random stream of its own for each source (so a change to one leaves the rest as they were),
    the files, log lines and questions so far, and every text of the week, which no needle's code may be found in."""

    def __init__(self, seed: int, scale: str, days: Sequence[date]):
        self.seed, self.scale, self.sc, self.days = seed, scale, SCALES[scale], tuple(days)
        self.files: list[File] = []
        self.lines: dict[str, list[dict]] = {service: [] for service in SERVICES}
        self.questions: list[dict] = []
        self.texts: list[str] = []  # every text of the week so far
        self.taken: set[int] = set()  # the numbers that are a code, an error's or a sign's
        r = self.rng("house")  # the house's smoke alarm and doorbell: one pitch and one chime for the whole week
        self.alarm_hz = float(r.randrange(3050, 3351, 10))
        self.chime = [float(n) for n in r.choice(DOORBELL_PAIRS)]

    def rng(self, name: str) -> random.Random:
        return random.Random(f"haystack-{VERSION}-{self.seed}-{self.scale}-{name}")

    @property
    def slots(self) -> list[tuple[int, str]]:
        return [(i, part) for i in range(len(self.days)) for part in PARTS]

    def span(self, slot: tuple[int, str]) -> tuple[float, float]:
        i, part = slot
        return at(self.days[i], PARTS[part][0] * HOUR), at(self.days[i], PARTS[part][1] * HOUR)

    def hint(self, slot: tuple[int, str]) -> str:
        i, part = slot
        lo, hi = PARTS[part]
        return f"on {date_text(self.days[i])}, in the {part} ({lo:02d}:00 to {hi:02d}:00)"

    def scope(self, level: str, where) -> tuple[float, float, str]:
        """The range and the hint of a question: `where` is its slot (`part`), its day (`day`), nothing (`week`) or the
        (since, until, hint) itself (`time`)."""
        if level == "part":
            return (*self.span(where), self.hint(where))
        if level == "day":
            return at(self.days[where], 0), at(self.days[where], 86400), f"on {date_text(self.days[where])}"
        if level == "week":
            return at(self.days[0], 0), at(self.days[-1], 86400), ""
        return where

    def pick(self, r: random.Random, population: Sequence, n: int, what: str) -> list:
        if n > len(population):
            raise ValueError(f"{what} needs {n} places and the week has {len(population)}")
        return r.sample(list(population), n)

    def fresh(self, r: random.Random, lo: int, hi: int, in_text: bool = True) -> int:
        """A number in [lo, hi) that is no other code or sign's number, and (`in_text`) is no number of any text so far."""
        blob = "\n".join(self.texts) if in_text else ""
        for _ in range(20_000):
            x = r.randrange(lo, hi)
            if x in self.taken or (in_text and (2020 <= x <= 2030 or re.search(rf"(?<!\d){x}(?!\d)", blob))):
                continue
            self.taken.add(x)
            return x
        raise ValueError(f"no number is left in [{lo}, {hi})")

    def eligible(self, source: str, avail: set) -> set:
        """The slots of `avail` with enough candidates (states, clips, photos, sounds) for a question to name them: at least the
        source's floor, with a tenth to spare for what is random (a screen state lasts 30 s on average, a segment of the
        microphone holds a sound or so)."""
        sc = self.sc
        floor = sc.part_floor[source]
        out = set()
        for slot in avail:
            d, part = slot
            k = list(PARTS).index(part)
            if source == "phone":
                have = share_out(sc.photos, PART_WEIGHTS)[k]
            elif source == "porch":
                have = share_out(sc.porch, PART_WEIGHTS)[k]
            elif source == "screen":
                lo, hi = PARTS[part][0] * HOUR, PARTS[part][1] * HOUR
                have = sum(max(0, min(b, hi) - max(a, lo)) for a, b in sc.screen[d % len(sc.screen)]) / MEAN_STATE_S
            else:  # about a sound to a segment
                lo, hi = PARTS[part][0] * HOUR, PARTS[part][1] * HOUR
                have = sum(max(0, min(b, hi) - max(a, lo)) for a, b in sc.mic[d % len(sc.mic)]) / 900
            if have >= floor * (1.1 if floor > 1 else 1):
                out.add(slot)
        return out

    def spot(self, template: str, avail: set, eligible: set | None = None) -> Spot:
        levels, absent = self.sc.asks[template]
        return arrange(self.rng(f"arrange-{template}"), template, len(self.days), levels, absent, avail,
                       avail if eligible is None else eligible, self.sc.extra, self.sc.shared, self.sc.early)

    def ask(self, template: str, level: str, where, *, thing: dict, value: str | None, moments: Sequence[Sequence[float]],
            grade_spec: dict, fake: tuple[str, bool, int, str], probe: dict | None = None, others: Sequence[Sequence[float]] = (),
            absent: bool = False, last: bool = False) -> None:
        """One question record (see the module's docstring). `thing` fills the template's words (`thing`, `topic`, `code`); an
        absent question is the template's own words about something that is not there."""
        type_, source = INFO[template]
        name = ABSENT_OF[template] if absent else template
        n = 1 + sum(1 for q in self.questions if q["template"] == name)
        since, until, hint = self.scope(level, where)
        words = {"hint": hint, **thing}
        question = (WEEK_WORDS[template] if level == "week" else WORDS[template]).format(**words)
        oracle = ORACLES[template]
        # an answer may repeat the numbers and times of the question (a code, the hours of the part of the day), and no others
        ignore = [*grade_spec["ignore"], *(x for x in grade.asked(question) if x not in grade_spec["ignore"])]
        self.questions.append({
            "qid": f"{type_}-{name}-{n}", "type": type_, "template": name, "absent": absent, "level": level, "last": last,
            "source": source, "since": since, "until": until, "hint": hint, "question": question,
            "truth": {"value": value, "moments": [[float(a), float(b)] for a, b in moments],
                      "others": [[float(a), float(b)] for a, b in others]},
            "grade": {**grade_spec, "ignore": ignore}, "oracle": oracle.format(**words) if oracle else None,
            "fake": {"question": fake[0], "oldest_first": fake[1], "limit": fake[2], "check": fake[3]}, "probe": probe})

    def emit(self, template: str, spot: Spot, placed: dict, args, absent_args) -> None:
        """The template's questions at the levels its scale asks. `args(level, needle, earlier)` is a question's keywords for
        the needle `placed` holds at the level's slot (`earlier`: the other needles, for a `week` question: all before it),
        `absent_args()` an absent one's."""
        levels, absent = self.sc.asks[template]
        for level in levels:
            if level == "time":
                continue
            slot = {"part": spot.part, "day": spot.day, "week": spot.last}[level]
            where = {"part": slot, "day": slot[0], "week": None}[level]
            earlier = [placed[s] for s in sorted(placed, key=slot_key) if s != slot] if level == "week" else []
            self.ask(template, level, where, last=level == "week" and template != "photo-count", **args(level, placed[slot], earlier))
        for level in absent:
            self.ask(template, level, spot.absent_part if level == "part" else spot.absent_day, absent=True, **absent_args())

    # -- text: config, notes

    def config(self) -> None:
        r, sc = self.rng("config"), self.sc
        n = len(self.days)
        times = [(i, m) for i in range(n) for m in spaced(r, 8 * 60, 20 * 60, sc.config, 120, what="app.toml versions")]
        values = self.pick(r, range(10, 100), len(times), "app.toml versions")
        for (i, minute), value in zip(times, values):
            text = f"[storage]\nretention_days = {value}\nmax_upload_mb = 25\n\n[server]\nport = 8080\n"
            self.texts.append(text)
            day = self.days[i]
            self.files.append(File(f"config/app.toml.{day:%Y-%m-%d}_{minute // 60:02d}-{minute % 60:02d}", "config",
                                   at(day, minute * 60), "app.toml", "config", {"text": text, "retention_days": value}))
        when = [at(self.days[i], m * 60) for i, m in times]
        used: set[tuple[int, str]] = set()  # the slots of the times asked about: one question to a slot
        for k in self.pick(r, range(len(times) - 1), 1, "config-asof (versions with a next one)"):
            candidates = [(i, minute) for i in range(n) for minute in range(6 * 60, 23 * 60 + 50)
                          if when[k] + 600 <= at(self.days[i], minute * 60) <= when[k + 1] - 600
                          and (i, part_of(minute)) not in used]
            if not candidates:
                raise ValueError("config-asof: no time is 10 minutes after a version and 10 minutes before the next")
            i, minute = r.choice(candidates)
            used.add((i, part_of(minute)))
            t = at(self.days[i], minute * 60)
            neighbours = [str(values[j]) for j in (k - 1, k + 1) if 0 <= j < len(values)]
            self.ask("config-asof", "time", (t, t + 1, f"at {minute // 60:02d}:{minute % 60:02d} on {date_text(self.days[i])}"),
                     thing={}, value=str(values[k]), moments=[[t, t + 1]], grade_spec=grade_of("number", str(values[k]), wrong=neighbours),
                     fake=("contains:retention_days", False, 50, "look"))

    def notes(self) -> None:
        r, sc = self.rng("notes"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("note-code", everywhere)
        topics = dict(zip(spot.needles, self.pick(r, TOPICS, len(spot.needles), "note-code (topics)")))
        minutes: dict[int, list[int]] = {i: [] for i in range(n)}
        entries: dict[tuple[int, str], int] = {}
        for slot in spot.needles:
            lo, hi = clip_range(slot[1])
            (minute,) = spaced(r, lo // 60, hi // 60, 1, 1, minutes[slot[0]], "note-code")
            minutes[slot[0]].append(minute)
            entries[slot] = minute
        for i in range(n):  # the distractors first: a needle's code must be in none of them
            need = sum(1 for slot in entries if slot[0] == i)
            for minute in spaced(r, DAY_START // 60, DAY_END // 60, max(0, sc.notes - need), 1, minutes[i], "notes"):
                self.add_note(i, minute, distractor_note(r))
        placed = {}
        for slot, minute in entries.items():
            thing, topic, sentence, digits = topics[slot]
            code = str(self.fresh(r, 10 ** (digits - 1), 10 ** digits))
            self.add_note(slot[0], minute, sentence.format(code=code), needle=True)
            placed[slot] = {"thing": thing, "topic": topic, "code": code, "t": at(self.days[slot[0]], minute * 60)}

        def args(level, needle, earlier):
            return {"thing": {"thing": needle["thing"], "topic": needle["topic"]}, "value": needle["code"],
                    "moments": [[needle["t"]] * 2], "grade_spec": grade_of("number", needle["code"]),
                    "fake": (f"contains:{needle['code']}", False, 50, "overlap")}

        def absent_args():
            return {"thing": {"thing": "Wi-Fi password", "topic": "a Wi-Fi password"}, "value": None, "moments": [],
                    "grade_spec": grade_of("absent"), "fake": ("contains:password", False, 50, "none")}

        self.emit("note-code", spot, placed, args, absent_args)

    def add_note(self, i: int, minute: int, text: str, needle: bool = False) -> None:
        day = self.days[i]
        self.texts.append(text)
        self.files.append(File(f"notes/note-{day:%Y-%m-%d}_{minute // 60:02d}-{minute % 60:02d}.md", "notes", at(day, minute * 60),
                               None, "note", {"text": text + "\n", "needle": needle}))

    # -- records: logs

    def logs_background(self) -> None:
        """Every service's ordinary lines: info, warnings and errors with codes ERR-100 to ERR-699. No number in them is a
        needle's code (ERR-700 to ERR-999, which is why a line has no bare number from 700 up)."""
        r, sc = self.rng("logs"), self.sc
        for i in range(len(self.days)):
            for service, share in zip(SERVICES, (0.4, 0.4, 0.2)):
                for secs in sorted(r.randrange(6 * HOUR, 86400) for _ in range(round(sc.log_lines * share))):
                    level, msg, code = background_line(r, service)
                    self.add_line(service, i, secs, level, msg, code)

    def add_line(self, service: str, i: int, secs: int, level: str, msg: str, code: str | None = None) -> None:
        day = self.days[i]
        line = {"ts": datetime(day.year, day.month, day.day, secs // 3600, secs // 60 % 60, secs % 60).isoformat(timespec="seconds"),
                "service": service, "level": level, "msg": msg, **({"code": code} if code else {})}
        self.lines[service].append(line)
        self.texts.append(json.dumps(line))

    def logs_needles(self) -> None:
        r, sc = self.rng("log-needles"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("log-count", everywhere)
        placed = {}
        for slot in spot.needles:
            i, part = slot
            k = r.randint(2, 6)
            code = f"ERR-{self.fresh(r, 700, 1000)}"
            lo, hi = (PARTS[part][0] * HOUR, PARTS[part][1] * HOUR) if slot == spot.part else (6 * HOUR, 86400)
            times = spaced(r, lo, hi, k, 600, what="log-count")
            others = [j for j in range(n) if j != i]
            if not others:
                raise ValueError("log-count: the week has no other day for the code to be logged on")
            for secs in times:
                self.add_line("payments", i, secs, "error", f"{r.choice(REASONS)} order={r.randrange(10000, 99999)}", code)
            for _ in range(r.randint(1, 3)):
                self.add_line("payments", r.choice(others), r.randrange(6 * HOUR, 86400), "error",
                              f"{r.choice(REASONS)} order={r.randrange(10000, 99999)}", code)
            placed[slot] = {"k": k, "code": code, "moments": [[at(self.days[i], s)] * 2 for s in times]}

        def args(level, needle, earlier):
            return {"thing": {"code": needle["code"]}, "value": str(needle["k"]), "moments": needle["moments"],
                    "grade_spec": grade_of("count", str(needle["k"]), ignore=[needle["code"][4:]]),
                    "fake": (f"contains:{needle['code']}", False, 50, "overlap")}

        def absent_args():
            code = f"ERR-{self.fresh(r, 700, 1000)}"
            return {"thing": {"code": code}, "value": None, "moments": [], "grade_spec": grade_of("absent"),
                    "fake": (f"contains:{code}", False, 50, "none")}

        self.emit("log-count", spot, placed, args, absent_args)
        last = self.spot("log-last", everywhere)
        builds: set[str] = set()
        for slot in last.needles:
            i, part = slot
            finishes = spaced(r, PARTS[part][0] * HOUR + 8 * 60, PARTS[part][1] * HOUR, r.randint(2, 4), 1200, what="log-last")
            ids = []
            for finish in finishes:
                build = self.new_build(r, builds)
                ids.append(build)
                self.add_line("web", i, finish - r.randint(180, 480), "info", f"deploy started build {build}")
                self.add_line("web", i, finish, "info", f"deploy finished build {build}")
            self.ask("log-last", "part", slot, thing={}, value=clock(finishes[-1]),
                     moments=[[at(self.days[i], s)] * 2 for s in finishes],
                     grade_spec=grade_of("time", clock(finishes[-1]), clock(finishes[-1]), 1, [clock(s) for s in finishes[:-1]]),
                     fake=(f"contains:{ids[-1]}", False, 50, "overlap"))
        free = [s for s in self.slots if s not in last.needles]  # the web service's other deploys, in no needle's slot
        for i in range(n):
            here = [s for s in free if s[0] == i]
            for slot in r.sample(here, min(len(here), r.randint(1, 2))):
                (finish,) = spaced(r, PARTS[slot[1]][0] * HOUR + 8 * 60, PARTS[slot[1]][1] * HOUR, 1, 1, what="web deploys")
                build = self.new_build(r, builds)
                self.add_line("web", i, finish - r.randint(180, 480), "info", f"deploy started build {build}")
                self.add_line("web", i, finish, "info", f"deploy finished build {build}")

    def new_build(self, r: random.Random, used: set[str]) -> str:
        while (build := f"{r.randrange(1 << 24):06x}") in used or build.isdigit():
            pass
        used.add(build)
        return build

    # -- image: the phone

    def photo_spec(self, r: random.Random, kind: str) -> dict:
        """A photo of a kind: a `sign` of another colour, `green` or `red` things that are no sign or car and a good part of the
        photo (a green car, a red bag or umbrella), `greensmall` or `redsmall` ones that are not (a door, a tomato), a `greenwall` or
        `redwall` (a hedge or a facade, most of the photo: more of the needle's colour than the needle has), a `car` of another
        colour, or a `plain` scene."""
        seed = r.randrange(1 << 30)
        green, red = DISTRACTOR_RGB["sign"], DISTRACTOR_RGB["car"]
        if kind == "sign":
            colour = r.choice(OTHER_SIGNS)
            return {"scene": "sign", "rgb": list(green[colour]), "word": r.choice(SIGN_WORDS),
                    "number": str(self.fresh(r, 10, 1000, False)), "box": None, "seed": seed}
        if kind == "car":
            return {"scene": "car", "rgb": list(red[r.choice(OTHER_CARS)]), "seed": seed}
        if kind == "greenwall":
            return {"scene": "wall", "rgb": list(r.choice((green["plant"], green["door"]))), "seed": seed}
        if kind == "redwall":
            return {"scene": "wall", "rgb": list(r.choice((red["umbrella"], red["tomato"]))), "seed": seed}
        if kind in ("green", "greensmall"):  # (a green car is a third of the photo, a green door a tenth)
            scene = "car" if kind == "green" else r.choices(("room", "plant", "street", "object"), (2, 1, 1, 1))[0]
            rgb = {"car": green["car"], "object": r.choice((green["shirt"], green["plant"], green["door"])), "room": green["door"],
                   "plant": green["plant"], "street": green["shirt"]}[scene]
            return {"scene": scene, "rgb": list(rgb), "seed": seed}
        if kind in ("red", "redsmall"):
            scene = r.choices(("object",) if kind == "red" else ("room", "food", "street"), (1,) if kind == "red" else (2, 1, 1))[0]
            rgb = {"object": r.choice((red["umbrella"], red["door"], red["tomato"])), "room": red["door"], "food": red["tomato"],
                   "street": red["umbrella"]}[scene]
            return {"scene": scene, "rgb": list(rgb), "seed": seed}
        return {"scene": r.choice(("object", "room", "food", "plant", "street")), "rgb": None, "seed": seed}

    def needle_seed(self, r: random.Random, scene: str) -> int:
        """A seed for a needle's photo (`sign` or `car`) whose backdrop is not a fence. A fence's vertical lines make a JPEG 15 KB
        bigger than a wall's or a sky's, so a third of the signs and cars would be the biggest files of their day: the needle must not
        be one by luck (the probe's size check), and it is no smaller than the rest either."""
        while True:
            seed = r.randrange(1 << 30)
            if tuple(_backdrop(_rng(seed, scene)).getpixel((3, 0))) != FENCE_RGB:
                return seed

    def phone(self) -> None:
        r, sc = self.rng("phone"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("photo-sign", everywhere, self.eligible("phone", everywhere))
        car_levels = sc.asks["photo-count"][0]
        car_day = r.randrange(n)  # the day a count question asks about; the other days hold red cars too
        cars = {d: r.randint(*sc.other_cars) for d in range(n)}
        cars[car_day] = r.randint(*sc.cars)
        others = [d for d in range(n) if d != car_day]
        while sum(cars[d] for d in others) < sc.other_min:
            cars[r.choice(others)] += 1
        shots: dict[tuple[int, str], list[dict]] = {}  # slot -> its photos: {"t": second of the day, "spec": ..}
        busy: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for part, count in zip(PARTS, share_out(sc.photos, PART_WEIGHTS)):
                times = spaced(r, *clip_range(part), count, 120, busy[i], f"photos of {part}")
                busy[i] += times
                shots[i, part] = [{"t": t, "spec": None} for t in times]

        def free(slot: tuple[int, str]) -> list[dict]:
            return [p for p in shots[slot] if p["spec"] is None]

        def take(pool: list[dict], what: str) -> dict:
            if not pool:
                raise ValueError(f"phone: no free photo for {what}: the scale's photos a day ({sc.photos}) hold less than its needles and distractors")
            return r.choice(pool)

        placed = {}
        for slot in spot.needles:
            p = take(free(slot), "a green sign")
            number = str(self.fresh(r, 10, 1000, False))
            p["spec"] = {"scene": "sign", "rgb": list(NEEDLE_RGB["sign"]), "word": r.choice(SIGN_WORDS), "number": number,
                         "box": None, "seed": self.needle_seed(r, "sign")}
            placed[slot] = {"slot": slot, "number": number, "t": at(self.days[slot[0]], p["t"])}
        for slot in self.slots:
            for kind, count in sc.mix["phone"]["slot"].items():
                for _ in range(count):
                    take(free(slot), kind)["spec"] = self.photo_spec(r, kind)
        for i in range(n):
            for kind, count in sc.mix["phone"]["day"].items():
                for _ in range(count):
                    take([p for part in PARTS for p in free((i, part))], kind)["spec"] = self.photo_spec(r, kind)
        for i in range(n):  # the day's red cars, 5 minutes apart
            pool = [p for part in PARTS for p in free((i, part))]
            for _ in range(500):
                chosen = r.sample(pool, cars[i]) if len(pool) >= cars[i] else None
                if chosen is not None and all(abs(a["t"] - b["t"]) >= 300 for k, a in enumerate(chosen) for b in chosen[:k]):
                    break
            else:
                raise ValueError(f"phone: no room for {cars[i]} red cars 5 minutes apart on day {i + 1}")
            for p in chosen:
                p["spec"] = {"scene": "car", "rgb": list(NEEDLE_RGB["car"]), "seed": self.needle_seed(r, "car")}
        for ps in shots.values():  # the rest: ordinary photos, some of them of the needles' colours
            for p in ps:
                if p["spec"] is None:
                    p["spec"] = self.photo_spec(r, r.choices(("plain", "sign", "greensmall", "redsmall", "car"), (55, 10, 10, 10, 15))[0])
        for (i, part), ps in sorted(shots.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[i]
            for p in ps:
                self.files.append(File(f"phone/IMG_{day:%Y%m%d}_{clock(p['t']).replace(':', '')}.jpg", "phone", at(day, p["t"]), None,
                                       "photo", p["spec"]))
        for i in range(n):
            day = self.days[i]
            for secs in spaced(r, DAY_START, DAY_END, sc.clips, 120, busy[i], "phone clips"):
                busy[i].append(secs)
                self.files.append(File(f"phone/VID_{day:%Y%m%d}_{clock(secs).replace(':', '')}.mp4", "phone", at(day, secs), None,
                                       "clip", {"scene": r.choice(("street", "dog", "shop")), "seed": r.randrange(1 << 30)}))

        def wrong_signs(since: float, until: float) -> list[str]:
            """The numbers of the signs in other colours in a range: what a wrong answer of a model would be."""
            return [f.spec["number"] for f in self.files if f.kind == "photo" and f.spec["scene"] == "sign"
                    and tuple(f.spec["rgb"]) != NEEDLE_RGB["sign"] and since <= f.t < until]

        def args(level, needle, earlier):
            since, until, _ = self.scope(level, {"part": needle["slot"], "day": needle["slot"][0], "week": None}[level])
            wrong = [e["number"] for e in earlier] if level == "week" else wrong_signs(since, until)
            return {"thing": {}, "value": needle["number"], "moments": [[needle["t"]] * 2],
                    "others": [[e["t"]] * 2 for e in earlier], "grade_spec": grade_of("number", needle["number"], wrong=wrong),
                    "fake": (shade_of("sign"), False, 50, "last" if level == "week" else "overlap"),
                    "probe": {"kind": "shade", "hue": "green", "rgb": list(NEEDLE_RGB["sign"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("sign"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "green", "rgb": list(NEEDLE_RGB["sign"])}}

        self.emit("photo-sign", spot, placed, args, absent_args)
        for level in car_levels:  # the red cars: a day's, or the week's
            days = [car_day] if level == "day" else list(range(n))
            moments = [[at(self.days[i], p["t"])] * 2 for i in days for part in PARTS for p in shots[i, part]
                       if p["spec"]["scene"] == "car" and tuple(p["spec"]["rgb"]) == NEEDLE_RGB["car"]]
            self.ask("photo-count", level, car_day if level == "day" else None, thing={}, value=str(len(moments)),
                     moments=moments, grade_spec=grade_of("count", str(len(moments))), fake=(shade_of("car"), False, 50, "count"),
                     probe={"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["car"])})

    # -- video: the screen and the porch

    def screen(self) -> None:
        """Each day's screen blocks as the recorder writes them (DESIGN.md §4.1, §6): a block is planned whole as states of 10 to 60 s,
        each with an activity `level` (`next_level`: a chain over the block, from a stream of its own) and the owner's acts at that level
        (`state_acts`, from the state's own seed: the cursor's place at the start, which is where the last state left it, and its moves,
        typing and scrolling), and then cut into segments of about `screen_s` (`segments`), each starting on a state's start and holding
        each state once. Its frames are `screen_frames`': a state's start, an act, a minute of the clock, at most 2 a second. The error
        is a state of 20 to 40 s at least 20 s from its block's edges, with the activity of any other state."""
        r, sc = self.rng("screen"), self.sc
        blocks: dict[tuple[int, str], list[dict]] = {}  # slot -> its blocks: {"start", "end", "error", "states", "segments"} (seconds of the day)
        for d in range(len(self.days)):
            for lo, hi in sc.screen[d % len(sc.screen)]:
                part = part_of(lo // 60)
                if part_of((hi - 1) // 60) != part:  # (so is every segment: none leaves its block)
                    raise ValueError(f"screen: the block at {clock(lo)} is in two parts of the day")
                blocks.setdefault((d, part), []).append({"start": lo, "end": hi, "error": None, "states": [], "segments": []})
        avail = set(blocks)
        spot = self.spot("screen-error", avail, self.eligible("screen", avail))
        for slot in spot.needles:
            block = r.choice(blocks[slot])
            span = r.randint(20, 40)
            a = r.randint(block["start"] + 20, block["end"] - span - 20)
            block["error"] = {"from": a, "to": a + span, "code": f"E-{self.fresh(r, 1000, 10000)}", "msg": r.choice(ERRORS)}
        levels, cursor = self.rng("screen-activity"), CURSOR_HOME  # the owner's cursor goes on from state to state
        for here in blocks.values():
            for block in here:
                err = block["error"]
                if err:
                    spans = [*((a, b, None) for a, b in cut(r, block["start"], err["from"])), (err["from"], err["to"], err),
                             *((a, b, None) for a, b in cut(r, err["to"], block["end"]))]
                else:
                    spans = [(a, b, None) for a, b in cut(r, block["start"], block["end"])]
                prev = None
                for a, b, e in spans:
                    app = r.choice([x for x in APPS if x != prev])
                    prev = app
                    overlay = {"kind": "error", "rgb": list(NEEDLE_RGB["error"]), "code": e["code"], "msg": e["msg"], "at": dialog_at(r)} if e else None
                    block["states"].append({"from": a, "to": b, "app": app, "overlay": overlay, "accent": None, "seed": r.randrange(1 << 30)})
            self.decorate(r, [s for block in here for s in block["states"]])
            for block in here:  # then the owner's acts (the decorations can make a state a browser's), a chain of levels over the block
                level = None
                for s in block["states"]:
                    level = next_level(levels, level)
                    s["level"], s["cursor"] = level, list(cursor)
                    s["acts"], cursor = state_acts(s, cursor, level)
            for block in here:  # then the segments, whose states are timed from their own start
                for lo, hi in segments([s["from"] for s in block["states"]], block["start"], block["end"], sc.screen_s):
                    block["segments"].append({"start": lo, "length": hi - lo, "states": [{**s, "from": s["from"] - lo, "to": s["to"] - lo}
                                                                                        for s in block["states"] if lo <= s["from"] < hi]})
        for slot, here in sorted(blocks.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for block in here:
                for seg in block["segments"]:
                    self.files.append(File(f"screen/screen-{day:%Y-%m-%d}_{clock(seg['start']).replace(':', '-')}.mkv", "screen",
                                           at(day, seg["start"]), None, "screen", {"length_s": seg["length"], "states": seg["states"]}))
        placed = {}
        for slot in spot.needles:  # (the error is one state, so it is whole in one segment: `verify` reads it back from the files)
            err = next(b["error"] for b in blocks[slot] if b["error"])
            day = self.days[slot[0]]
            placed[slot] = {"day": slot[0], "code": err["code"], "clock": clock(err["from"]), "moments": [[at(day, err["from"]), at(day, err["to"])]]}

        def args(level, needle, earlier):
            week = level == "week"
            value = f"{needle['code']} at {needle['clock']}" + (f" on {date_text(self.days[needle['day']])}" if week else "")
            return {"thing": {}, "value": value, "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                    "grade_spec": grade_of("code_time", needle["code"], needle["clock"], 1, date_=self.days[needle["day"]].isoformat() if week else None),
                    "fake": (shade_of("error"), False, 50, "last" if week else "overlap"),
                    "probe": {"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["error"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("error"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["error"])}}

        self.emit("screen-error", spot, placed, args, absent_args)

    def dialog(self, r: random.Random, kind: str) -> dict:
        """An overlay that is no error, opened somewhere in the middle of the screen as an error is: a warning (orange, with its
        code), an `alert` (a warning with a red header, which a red filter takes for an error), a save prompt, an info box or a red
        "Delete?" confirm."""
        at = dialog_at(r)
        if kind in ("warning", "alert"):
            red = {"rgb": list(DISTRACTOR_RGB["error"]["confirm"])} if kind == "alert" else {}
            return {"kind": "warning", **red, "code": f"W-{self.fresh(r, 1000, 10000, False)}", "msg": r.choice(WARNINGS), "at": at}
        return {"kind": kind, "at": at}

    def decorate(self, r: random.Random, states: list[dict]) -> None:
        """The distractors of a part of a day (its states): the scale's planted ones (dialogs of each kind, red ones among them, red banners and
        other red accents), then about 15% of the states with a red accent and 2% with a dialog."""
        mix = self.sc.mix["screen"]["slot"]
        free = [s for s in states if s["overlay"] is None]
        r.shuffle(free)
        for kind in ("confirm", "alert", "warning", "info", "save"):
            for _ in range(mix.get(kind, 0)):
                if not free:
                    raise ValueError("screen: a part of a day has fewer states than the scale's dialogs")
                free.pop()["overlay"] = self.dialog(r, kind)
        bare = [s for s in states if s["accent"] is None]
        r.shuffle(bare)
        for _ in range(mix.get("banner", 0)):  # a banner is a web page's: a red band bigger than the error's header
            s = bare.pop()
            s["app"], s["accent"] = "browser", {"kind": "banner"}
        for _ in range(mix.get("accent", 0)):
            s = bare.pop()
            s["accent"] = {"kind": {"browser": "banner", "sheet": "cells"}.get(s["app"], "toast")}
        for s in states:
            if s["accent"] is None and r.random() < 0.15:
                s["accent"] = {"kind": r.choice({"browser": ("banner", "toast", "badge"), "sheet": ("cells", "badge", "toast")}.get(s["app"], ("toast", "badge")))}
            if s["overlay"] is None and r.random() < 0.02:
                s["overlay"] = self.dialog(r, r.choice(("confirm", "warning", "info", "save", "save")))

    def event(self, r: random.Random, kind: str, length: int, window: tuple[int, int] | None = None) -> dict:
        """A distractor of the porch camera over some seconds of a clip of `length` (within the seconds `window` when there is one): a
        `walker`, `cat`, `dog`, `car`, a `visitor` close to the camera in a brown coat (more brown, in the place a parcel is left, than a
        parcel has), a `light` change, or the `courier` bringing a box up to the door and taking it away again (`drop_at` None: he never
        puts it down, and the step stays empty: nothing is delivered)."""
        lo, hi = window or (0, length)
        a = r.randint(lo, max(lo, hi - 8))
        b = min(hi - 1, a + r.randint(4, 10))
        if b <= a:
            a, b = max(lo, hi - 6), hi - 1
        if b <= a:
            raise ValueError(f"a {kind} needs 2 seconds or more of a clip, and the window {lo} to {hi} has fewer")
        if kind == "courier":
            return {"kind": "courier", "from": a, "to": b, "drop_at": None}
        if kind == "visitor":  # a brown coat close to the camera, at the door (x), with a carrier bag or without
            return {"kind": "visitor", "from": a, "to": b, "x": r.randint(110, 200), "scale": round(r.uniform(2.9, 3.2), 2), "bag": r.random() < 0.5}
        return {"kind": kind, "from": a, "to": b}

    @staticmethod
    def parcel_free(clip: dict) -> tuple[int, int]:
        """The seconds `(from, to)` of a clip that show no parcel: all of it, the seconds before the drop, or those after the pickup."""
        parcel = clip["parcel"]
        return (0, clip["length"]) if not parcel else (0, parcel[0]) if parcel[0] > 0 else (parcel[1], clip["length"])

    def has_person(self, clip: dict) -> bool:
        return any(e["kind"] in ("courier", "owner_pickup") for e in clip["events"])

    def porch(self) -> None:
        """Clips of 10 to 30 s all day, one frame a second. On the days of the needles a courier leaves a parcel in a clip, the
        clips of the next 20 to 120 minutes show it, and the owner takes it in in the last: all in one part of the day (so that
        no part holds a parcel that was left in another)."""
        r, sc = self.rng("porch"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("porch-parcel", everywhere, self.eligible("porch", everywhere))
        clips: dict[tuple[int, str], list[dict]] = {}
        fixed: dict[int, list[int]] = {i: [] for i in range(n)}
        placed = {}
        for slot in spot.needles:  # the clip a parcel is left in, and the clip it is taken in
            i, part = slot
            lo, hi = clip_range(part)
            for _ in range(2000):
                stay = r.randint(20, 120) * 60
                first, second = r.randint(14, 20), r.randint(12, 20)  # (a middling length: a needle's file is neither the biggest nor the smallest)
                drop_at, pick = r.randint(4, 8), r.randint(4, 10)
                top = hi - 30 - second - drop_at - stay + pick
                if top > lo:
                    break
            else:
                raise ValueError(f"porch-parcel: no room for a parcel to stay up to 2 hours in the {part}")
            t = r.randrange(lo, top)
            back = t + drop_at + stay - pick
            box = parcel_box(r)  # where it is left on the step: it stays there from the drop to the pickup
            clips.setdefault(slot, []).extend([
                {"t": t, "length": first, "events": [{"kind": "courier", "from": 0, "to": min(first - 1, drop_at + 6), "drop_at": drop_at}],
                 "parcel": [drop_at, first], "box": box},
                {"t": back, "length": second, "events": [{"kind": "owner_pickup", "from": max(0, pick - 3), "to": min(second - 1, pick + 3)}],
                 "parcel": [0, pick], "box": box}])
            fixed[i] += [t, back]
            placed[slot] = {"day": i, "clock": clock(t + drop_at), "t": t, "back": back, "drop_at": drop_at, "box": box}
        for i in range(n):
            for part, count in zip(PARTS, share_out(sc.porch, PART_WEIGHTS)):
                made = clips.setdefault((i, part), [])
                if len(made) > count:
                    raise ValueError(f"porch: the {part} of day {i + 1} has {count} clips, a parcel needs {len(made)} of them")
                lo, hi = clip_range(part)
                for t in spaced(r, lo, hi - 30, count - len(made), 120, fixed[i], f"porch clips of the {part}"):
                    made.append({"t": t, "length": clip_length(r), "events": [], "parcel": None})
                    fixed[i].append(t)
            day_clips = sorted((c for part in PARTS for c in clips[i, part]), key=lambda c: c["t"])
            for slot, needle in placed.items():  # the parcel is on the step in the clips between the two
                if slot[0] == i:
                    for c in day_clips:
                        if needle["t"] < c["t"] < needle["back"] and c["parcel"] is None:
                            c["parcel"], c["box"] = [0, c["length"]], needle["box"]
        for slot, cs in clips.items():
            plant = [kind for kind, count in sc.mix["porch"]["slot"].items() for _ in range(count)]
            for kind in plant:  # (in a clip that shows the parcel, on the seconds that do not: a scope of only the needle's own clips must still
                #                   hold the distractors' frames, and not have them hidden among the parcel's)
                usable = [c for c in cs if (free := self.parcel_free(c))[1] - free[0] >= 4]
                if not usable:
                    raise ValueError(f"porch: no clip of {slot} has 4 seconds with no parcel for a {kind}")
                calm = [c for c in usable if not self.has_person(c)]
                c = r.choice([c for c in calm if c["parcel"] is None] or calm or usable)
                c["events"].append(self.event(r, kind, c["length"], self.parcel_free(c)))
        for i in range(n):
            plant = [kind for kind, count in sc.mix["porch"]["day"].items() for _ in range(count)]
            day_clips = [c for part in PARTS for c in clips[i, part]]
            for kind in plant:
                c = r.choice(day_clips)
                c["events"].append(self.event(r, kind, c["length"]))
        for slot, cs in sorted(clips.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for c in sorted(cs, key=lambda c: c["t"]):
                if not c["events"] or (r.random() < 0.15 and not self.has_person(c)):
                    kinds, weights = ("walker", "car", "cat", "dog", "light", "courier"), (45, 15, 12, 8, 10, 5)
                    if c["parcel"]:  # a clip that shows the parcel has nothing that makes its file bigger than the others': its size gives it away
                        kinds, weights = kinds[:3:2], weights[:3:2]
                    c["events"].append(self.event(r, r.choices(kinds, weights)[0], c["length"]))
                spec = {"length_s": c["length"], "events": c["events"], "parcel": c["parcel"], "box": c.get("box"), "seed": r.randrange(1 << 30)}
                self.files.append(File(f"porch/porch-{day:%Y-%m-%d}_{clock(c['t']).replace(':', '-')}.mp4", "porch", at(day, c["t"]),
                                       None, "porch", spec))
        for slot, needle in placed.items():  # where the parcel is on the step: its clips, in the files as they were made
            day = self.days[needle["day"]]
            needle["moments"] = [[at(day, c["t"]) + c["parcel"][0], at(day, c["t"]) + c["parcel"][1]]
                                 for part in PARTS for c in sorted(clips[needle["day"], part], key=lambda c: c["t"])
                                 if c["parcel"] and needle["t"] <= c["t"] <= needle["back"]]

        def args(level, needle, earlier):
            week = level == "week"
            return {"thing": {}, "value": needle["clock"], "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                    "grade_spec": grade_of("time", needle["clock"], needle["clock"], 2, apart([e["clock"] for e in earlier], needle["clock"]) if week else (),
                                           date_=self.days[needle["day"]].isoformat() if week else None),
                    "fake": (shade_of("parcel"), False, 50, "last" if week else "overlap"),
                    "probe": {"kind": "shade", "hue": "brown", "rgb": list(NEEDLE_RGB["parcel"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("parcel"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "brown", "rgb": list(NEEDLE_RGB["parcel"])}}

        self.emit("porch-parcel", spot, placed, args, absent_args)

    # -- audio: the hallway microphone

    def mic(self) -> None:
        r, sc = self.rng("mic"), self.sc
        n = len(self.days)
        segs: dict[tuple[int, str], list[dict]] = {}
        for d in range(n):
            for lo, hi in sc.mic[d % len(sc.mic)]:
                for start in range(lo, hi, 900):
                    segs.setdefault((d, part_of(start // 60)), []).append(
                        {"day": d, "start": start, "length": 900, "sounds": [], "tv": [], "seed": r.randrange(1 << 30)})
        avail = set(segs)
        eligible = self.eligible("mic", avail)
        spots = {"mic-doorbell": self.spot("mic-doorbell", avail, eligible), "mic-alarm": self.spot("mic-alarm", avail, eligible)}
        needles: dict[str, dict] = {}
        for template, kind, dur in (("mic-doorbell", "doorbell", CHIME_DUR), ("mic-alarm", "alarm", ALARM_DUR)):
            needles[template] = {}
            for slot in spots[template].needles:
                extra = {"notes": list(self.chime)} if kind == "doorbell" else {"hz": self.alarm_hz}
                seg, sound = self.place_sound(r, segs[slot], kind, dur, extra, margin=100)
                needles[template][slot] = {"day": slot[0], "start": seg["start"], "sound": sound}
        mix = sc.mix["mic"]["slot"]
        beepers, band = ("microwave", "oven", "watch", "truck"), ("ding", "ding", "ding", "ring", "bark")
        for slot, ss in segs.items():
            for kind in [r.choice(beepers) for _ in range(mix.get("beeper", 0))] + [r.choice(band) for _ in range(mix.get("band", 0))]:
                extra = {"hz": 987.77} if kind == "ding" else {"hz": 440.0} if kind == "ring" else {}
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), extra)
            for _ in range(mix.get("loud", 0)):  # louder than the needles, each in its band: a beeper in the alarm's, a timer in the chime's
                kind = r.choice(("oven", "truck"))
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), {"gain": LOUD_GAIN})
                self.place_sound(r, ss, "timer", self.sound_dur(r, "timer"), {"gain": LOUD_GAIN})
            for _ in range(sum(r.random() < sc.mic_extra for _ in ss)):
                kind = r.choice(("kettle", "slam", "steps", "vacuum", "ding", "ring", "bark", "microwave", "watch"))
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), {})
        for d in range(n):  # TV: stretches of 2 to 8 minutes, up to the day's minutes, never within a minute of a needle
            today = [seg for slot, ss in segs.items() if slot[0] == d for seg in ss]
            total = 0
            for _ in range(2000):
                if total >= sc.tv_s:
                    break
                seg = r.choice(today)
                length = r.randint(120, 480)
                a = r.randint(0, seg["length"] - length)
                needles_here = [s for s in seg["sounds"] if s["kind"] in ("doorbell", "alarm")]
                if all(a + length + MIC_GAP_S <= s["at"] or a >= s["at"] + s["dur"] + MIC_GAP_S for s in needles_here) and \
                        all(a + length <= t0 or a >= t1 for t0, t1 in seg["tv"]):
                    seg["tv"].append([a, a + length])
                    total += length
            else:
                raise ValueError(f"mic: no room for {sc.tv_s} s of TV on day {d + 1}")
        for slot, ss in sorted(segs.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for seg in sorted(ss, key=lambda s: s["start"]):
                spec = {"length_s": seg["length"], "seed": seg["seed"], "bed_dbfs": BED_DBFS, "tv": sorted(seg["tv"]),
                        "sounds": sorted(seg["sounds"], key=lambda s: s["at"])}
                self.files.append(File(f"mic/mic-{day:%Y-%m-%d}_{clock(seg['start']).replace(':', '-')}.ogg", "mic", at(day, seg["start"]),
                                       None, "mic", spec))
        for template, kind in (("mic-doorbell", "doorbell"), ("mic-alarm", "alarm")):
            hz = self.chime[0] if kind == "doorbell" else self.alarm_hz
            probe = {"kind": "pitch", "hz": hz, "band": BANDS[kind]}
            placed = {}
            for slot, needle in needles[template].items():
                s, day = needle["sound"], self.days[needle["day"]]
                heard = int(s["at"])
                placed[slot] = {"day": needle["day"], "clock": clock(needle["start"] + heard),
                                "moments": [[at(day, needle["start"]) + s["at"], at(day, needle["start"]) + s["at"] + s["dur"]]]}

            def args(level, needle, earlier, hz=hz, probe=probe):
                week = level == "week"
                return {"thing": {}, "value": needle["clock"], "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                        "grade_spec": grade_of("time", needle["clock"], needle["clock"], 1, apart([e["clock"] for e in earlier], needle["clock"]) if week else (),
                                               date_=self.days[needle["day"]].isoformat() if week else None),
                        "fake": (f"pitch:{hz:g}", False, 50, "last" if week else "overlap"), "probe": probe}

            def absent_args(hz=hz, probe=probe):
                return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (f"pitch:{hz:g}", False, 50, "none"),
                        "probe": probe}

            self.emit(template, spots[template], placed, args, absent_args)

    def sound_dur(self, r: random.Random, kind: str) -> float:
        return {"microwave": 2.0, "oven": 2.5, "watch": 1.0, "truck": 3.0, "timer": 2.8, "ding": 0.8, "ring": 2.4, "slam": 0.15, "bark": 1.2,
                "kettle": round(r.uniform(3, 5), 1), "steps": 0.5 * r.randint(4, 8), "vacuum": float(r.randint(5, 10))}[kind]

    def place_sound(self, r: random.Random, segments: list[dict], kind: str, dur: float, extra: dict, margin: int = 2) -> tuple[dict, dict]:
        """A sound of `kind` somewhere in one of `segments`: 2 s from the others' and a needle's MIC_GAP_S, and from the segment's
        ends by `margin`. Returns the segment and the sound's spec."""
        for _ in range(500):
            seg = r.choice(segments)
            at_ = round(r.uniform(margin, seg["length"] - dur - margin), 1)
            ok = True
            for s in seg["sounds"]:
                gap = MIC_GAP_S if "doorbell" in (kind, s["kind"]) or "alarm" in (kind, s["kind"]) else 2
                if not (at_ + dur + gap <= s["at"] or at_ >= s["at"] + s["dur"] + gap):
                    ok = False
                    break
            if ok:
                sound = {"kind": kind, "at": at_, "dur": dur, "seed": r.randrange(1 << 30), **extra}
                seg["sounds"].append(sound)
                return seg, sound
        raise ValueError(f"mic: no room for a {kind} in {len(segments)} segments")


def plan(seed: int, scale: str, days: Sequence[date]) -> Plan:
    """The week for `seed` over `days` (oldest first, one per day of the scale): its files, its log lines and its questions.
    Pure: nothing is written and no clock is read. A `ValueError` names a template whose needs the week cannot meet; the
    plan checks itself (`verify`)."""
    if scale not in SCALES:
        raise ValueError(f"a scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = tuple(days)
    if len(days) != SCALES[scale].days:
        raise ValueError(f"the {scale} scale is {SCALES[scale].days} days, not {len(days)}")
    if any(later <= earlier for earlier, later in zip(days, days[1:])):
        raise ValueError("the days are in order, oldest first, none twice")
    week = _Week(seed, scale, days)
    week.logs_background()
    week.config()
    week.notes()
    week.logs_needles()
    week.phone()
    week.screen()
    week.porch()
    week.mic()
    order = {template: (TYPES.index(type_), k) for type_ in TYPES for k, template in enumerate((*TEMPLATES[type_], *ABSENT[type_]))}
    made = Plan(seed, scale, days, tuple(sorted(week.files, key=lambda f: (f.t, f.path))),
                {service: tuple(sorted(lines, key=lambda line: line["ts"])) for service, lines in week.lines.items()},
                tuple(sorted(week.questions, key=lambda q: order[q["template"]])))
    verify(made)
    return made
