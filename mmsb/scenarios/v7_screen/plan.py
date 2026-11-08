"""A synthetic week of an owner's phone, screen, front door camera, hallway microphone, notes, config and logs (TESTBED.md,
benchmark): the ground truth of the V7 scenario. Needles are things a question asks for (a code in a note, a sign's number,
an error on screen, a doorbell), hidden among distractors that must never match a needle's question. A needle is told apart by
meaning (an error dialog among red banners and a red "Delete?" dialog, a parcel among a brown dog and the courier's own box, a
smoke alarm among beepers and chimes) and, for the fake decider of the offline tests, by an exact shade or pitch that is its alone
(`media_*.py` draw and play everything).

    plan = week.plan(seed, "tiny", days_before(date.today(), 2))   # pure: no files, no clock
    write.write(plan, folder)            # the files (PIL, numpy, PyAV, ffmpeg), each checked as it is made; never the truth

This module holds what a plan is made of (`Scale`, `SCALES`, `File`, `Plan`, the vocabulary of templates, levels and sources), the helpers
that place needles in a week (`arrange`, `segments`, `state_acts`, ...) and the plan's check of itself (`verify`). `week.plan` makes it.

`plan` is a function of (seed, scale, days): the same inputs give the same files, lines and questions, and nothing is
written until `write`. It checks itself (`verify`): the scope of every question holds what its level says, counted from the
files' specs and not from the questions. The truth stays in the plan's questions and is never on disk. A question is a plain dict:
- `qid` (`image-photo-sign-1`: type, template, n), `type`, `template`, `absent` and `source` (the source name, and the folder);
- `level`: how vague its time hint is, `part` (a part of a day: `PARTS`), `day`, `week` (no hint: the prompt gives the span, and the
  question asks for the last one, or, for a count, all of them) or `time` (an exact time: `config-asof`); `last`: a `week` question
  that asks for the last needle (a count asks for all of them);
- `since` and `until`, the range it names (local epoch), `hint` (the day and part of the day it words, with hours; "" for `week`)
  and `question`; an absent question has its template's exact words, so an agent cannot tell it from a needle question;
- `truth`: `{"value", "moments", "others"}`, the answer, the spans `[t0, t1]` it is on record in (None and none when absent) and,
  for a `week` question, the spans of the template's other needles (all earlier: nothing of the template comes after the last);
- `grade`: what `mmsb/core/grade.py` grades by (`kind`, `truth`, `time`, `tol_min`, `wrong`, `ignore`, `date`: the day a `week` question's
  time must be on); the earlier needles are a `week` question's `wrong` answers. The keys are those of the recorded plan, which
  `plan_hash` covers; `questions.py` writes them as a `Grade` (`kind` is its `rule`, `tol_min` is `tol_s` in seconds);
- `oracle`: a yes/no for a real model over the source and range, or None; `fake`: the question the fake decider is asked
  (`question`, `oldest_first`, `limit`) and what its sightings must show (`check`: `overlap` each moment is found and every sighting
  is at one; `none` no sighting in the range; `count` the answer's number of them; `look` one version whose text holds the value;
  `last` the newest sighting is at the last needle and every one is at some needle);
- `probe`: what the shortcut probe (`shortcuts.py`) measures: `{"kind": "shade", "hue", "rgb"}`, `{"kind": "pitch", "hz",
  "band"}` or None (text and records).

| Template      | Source (type)   | The needle                                          | The question asks                         |
| note-code     | notes (text)    | a note with a 4- or 6-digit code                    | the code, for a thing noted in a slot     |
| config-asof   | config (text)   | the version of app.toml current at a time           | retention_days at that time               |
| log-count     | logs (records)  | ERR-7xx-9xx k times in a slot, and on other days    | how many times payments logged it         |
| log-last      | logs (records)  | web deploys in a slot, started and finished         | when the last one finished                |
| photo-sign    | phone (image)   | a green sign (22-34% of the photo): word, number    | the sign's number                         |
| photo-count   | phone (image)   | k photos of a red car (30% or more) on one day      | how many (on a day, or all week)          |
| screen-error  | screen (video)  | a dialog, red header "ERROR E-xxxx", 20-40 s        | its code, and when it first appeared      |
| porch-parcel  | porch (video)   | a cardboard box left on the step by a courier       | when it was left                          |
| mic-doorbell  | mic (audio)     | a two-note chime (the house's pair)                 | when it rang                              |
| mic-alarm     | mic (audio)     | 3 beeps of 0.5 s at the house's alarm pitch         | when the smoke alarm beeped               |
| absent-*      | each type       | none: the same words, about what is not there       | the answer is "not there"                 |

Three scales (`SCALES`; a day is what `Scale` says, 6.5 h of screen and 6 h of microphone at the harder ones):
- `tiny`: 2 days, one 20-minute screen block a day, 6 porch clips, 12 photos and 30 minutes of microphone a day, 19 questions, one
  needle for a template (its part question and its week question ask about it; day 2 holds none). The offline test and the smoke.
- `pilot`: 3 days at the density below, 31 questions. For the four templates with a week question (sign, error, parcel, alarm) the
  last needle is on day 2, in the first part of it that has data, and the part and day questions ask about one needle on day 1: a
  week of 3 days has no room for two before the last one, and a late last needle would leave less than 2 days to scan.
- `standard`: 7 days, 31 questions (24 for a model that does not hear). Per day: about 790 screen states and 30,000 frames (VFR segments cut as
  the recorder cuts them, each ending at the first state at or after each minute from its block's start: 60 s on average, 10 to about 120 s,
  390 files; 09:00-11:30, 13:30-16:30, 20:00-21:00, a new state every 10 to 60 s, and a frame, at most one every half second, for each state,
  each act of the owner (a cursor move, typing, a scroll) and each minute of the taskbar's clock: `screen_frames`; the reader reads about 16,000
  of those seconds), 30 porch clips of 10 to 30 s (about 480 frames), 60 photos
  and 3 clips (each carries its own time, EXIF or `creation_time`, and is remembered without one), 6 h of microphone (07:00-09:00, 17:00-21:00) as 15 minute segments with about 25 sounds and 30 minutes of TV, and
  the controls unchanged (15 notes, 2 versions of app.toml, 2,000 log lines).
Distractors are planted in every part of a day before the random ones (`Scale.mix`), so each scope a question names holds some of every
class: red dialogs, banners and toasts among the screen's states (a banner is bigger than the error's header), green and red things,
signs and cars of other colours and facades and hedges bigger than a sign or a car among the photos, a brown dog, a light change and
the courier's own box and a visitor in a brown coat close to the camera among the porch's clips, beepers, timers, dings, rings and barks among the microphone's sounds (a beeper and a timer in each part are louder than the alarm and the chime). Dialogs (the error's and
the others', red ones among them) open in different places of the middle of the screen, a parcel is left in different places of the step (the same
in every clip that shows it), and the distractors of the error's red and of the parcel's brown are in those places too (`DIALOG_AREA`, `BOX_AREA`
of `media_screen` and `media_porch`): a crop of one place keeping red or brown does not find the needle first. A needle's photo has no
fence behind it, and a clip that shows the parcel has no light or car, so that no needle's file is the biggest of its scope by luck.
A part question names a part with enough to look through (`Scale.part_floor`: 250 screen states, 8 porch clips, 15 photos, 5 sounds), so
no question names the evening's screen block (an hour) or the microphone's one hour of the afternoon.
Times are local, made with `datetime(y, m, d, h, mi, s)` and never by adding seconds across days, and no event is before 06:00
(daylight saving changes happen at night). `write` asserts its own checks (a needle's shade or pitch is the fake decider's after
encoding, no distractor has it, a needle's code is in its own file only), so a wrong fixture fails here and never in the reader.
Moved from haystack.py of the repository at bb498a6."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from mmsb.scenarios.v7_screen.media_common import NEEDLE_RGB, _rng
from mmsb.scenarios.v7_screen.media_screen import SCREEN, typing_cap

VERSION = 6  # bump when plan() changes what it makes (6: the screen's acts, TESTBED.md)
TYPES = ("text", "records", "image", "video", "audio")
TEMPLATES = {"text": ("note-code", "config-asof"), "records": ("log-count", "log-last"),
             "image": ("photo-sign", "photo-count"), "video": ("screen-error", "porch-parcel"),
             "audio": ("mic-doorbell", "mic-alarm")}
ABSENT = {"text": ("absent-note",), "records": ("absent-log",), "image": ("absent-sign",),
          "video": ("absent-error", "absent-parcel"), "audio": ("absent-doorbell",)}
ABSENT_OF = {"note-code": "absent-note", "log-count": "absent-log", "photo-sign": "absent-sign", "screen-error": "absent-error",
             "porch-parcel": "absent-parcel", "mic-doorbell": "absent-doorbell"}
BASE_OF = {absent: template for template, absent in ABSENT_OF.items()}  # the template an absent one asks in the words of
INFO = {"note-code": ("text", "notes"), "config-asof": ("text", "config"), "log-count": ("records", "logs"),
        "log-last": ("records", "logs"), "photo-sign": ("image", "phone"), "photo-count": ("image", "phone"),
        "screen-error": ("video", "screen"), "porch-parcel": ("video", "porch"), "mic-doorbell": ("audio", "mic"),
        "mic-alarm": ("audio", "mic")}
LEVELS = ("part", "day", "week", "time")
PARTS = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24)}
SOURCES = {"screen": "my screen recordings", "phone": "photos and short videos from my phone",
           "porch": "the front door camera's motion clips", "mic": "the hallway microphone", "notes": "my notes",
           "config": "the versions of app.toml", "logs": "the payments, web and auth services' logs"}
LOG_SOURCE = "logs"
PHONE = "phone"  # the source whose files carry their own time
SERVICES = ("payments", "web", "auth")
TOL_S = 10.0  # a sighting overlaps a moment when it comes within this many seconds of it
MIC_GAP_S = 75  # a room's sound is this far from a needle's: the answer's tolerance is a minute, so it must not be mistaken for it

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
DAY_START, DAY_END = 7 * 3600, 23 * 3600  # photos, clips, notes and porch clips are at these times of day

MEAN_STATE_S = 30.0  # a screen state lasts 10 to 60 s, 30 on average
BANDS = {"alarm": [2000, 5000], "doorbell": [400, 1000]}  # what a band filter for the sound's class keeps (the probe's `band`)
HOUR = 3600
FENCE_RGB = (150, 142, 130)  # the colour of a fence's posts in a photo's backdrop (media_photo draws them at the top row of the photo)


@dataclass(frozen=True)
class Scale:
    """What a week holds: per day unless said. `asks` is each template's levels, `(needle levels, absent levels)`; `extra` the
    earlier needles of a template beyond the ones its questions need; `shared`: None, `last` (the part question and the week
    question ask about one needle: tiny has one needle for a template) or `day` (the part and day questions of a template that
    has a week question ask about one, so that its last needle can be early on a short week); `early`: the last needle is in the
    first part of its day that has data. `screen` and `mic` are the blocks of each day (`(start_s, end_s)`; the day after the
    last has the first's again), recorded as segments: the screen's of about `screen_s` (each ends at the first state change at or
    after a multiple of `screen_s` from its block's start: the recorder cuts at the first key frame at or after each `screen_s`, and
    here key frames are at a file's start and at scene cuts, which are the state changes; the recorder also forces one every 10 s, so
    its cuts land mid-state: not modelled), the microphone's of 900 s. `mix` is what every part of a day (`slot`) or day
    (`day`) holds of each source's distractors (planted before the random ones), `part_floor` the least candidates (states, clips,
    photos, sounds) a part of a day needs for a question to name it."""

    days: int
    asks: dict
    extra: int
    shared: str | None
    early: bool
    photos: int
    clips: int
    cars: tuple[int, int]
    other_cars: tuple[int, int]
    other_min: int
    screen: tuple
    screen_s: int
    porch: int
    mic: tuple
    mic_extra: float
    tv_s: int
    notes: int
    config: int
    log_lines: int
    mix: dict
    part_floor: dict


TINY_ASKS = {"note-code": (("part",), ("part",)), "config-asof": (("time",), ()), "log-count": (("part",), ("day",)),
             "log-last": (("part",), ()), "photo-sign": (("part", "week"), ("part",)), "photo-count": (("day",), ()),
             "screen-error": (("part", "week"), ("part",)), "porch-parcel": (("part", "week"), ()),
             "mic-doorbell": (("part",), ("part",)), "mic-alarm": (("part", "week"), ())}
FULL_ASKS = {"note-code": (("part", "day"), ("part",)), "config-asof": (("time",), ()),
             "log-count": (("part", "day"), ("day",)), "log-last": (("part",), ()),
             "photo-sign": (("part", "day", "week"), ("part", "day")), "photo-count": (("day", "week"), ()),
             "screen-error": (("part", "day", "week"), ("part", "day")), "porch-parcel": (("part", "day", "week"), ("day",)),
             "mic-doorbell": (("part", "day"), ("part", "day")), "mic-alarm": (("part", "day", "week"), ())}
TINY_MIX = {"screen": {"slot": {"banner": 1, "accent": 2, "confirm": 1, "alert": 1, "warning": 1, "info": 1, "save": 1}, "day": {}},
            "phone": {"slot": {"greenwall": 1, "sign": 1}, "day": {"redwall": 1, "car": 1}},
            "porch": {"slot": {"dog": 1, "light": 1, "visitor": 1}, "day": {"car": 1}},
            "mic": {"slot": {"beeper": 1, "band": 1, "loud": 1}, "day": {}}}
FULL_MIX = {"screen": {"slot": {"banner": 1, "accent": 4, "confirm": 3, "alert": 3, "warning": 1, "info": 1, "save": 1}, "day": {}},
            "phone": {"slot": {"greenwall": 3, "green": 3, "redwall": 5, "red": 2, "sign": 2, "car": 2}, "day": {}},
            "porch": {"slot": {"dog": 2, "light": 2, "car": 2, "courier": 1, "visitor": 2}, "day": {}},
            "mic": {"slot": {"beeper": 3, "band": 3, "loud": 1}, "day": {}}}
BLOCKS = ((9 * HOUR, 9 * HOUR + 9000), (13 * HOUR + 1800, 16 * HOUR + 1800), (20 * HOUR, 21 * HOUR))  # 09:00-11:30, 13:30-16:30, 20:00-21:00
MIC_BLOCKS = ((7 * HOUR, 9 * HOUR), (17 * HOUR, 21 * HOUR))
FULL = dict(asks=FULL_ASKS, photos=60, clips=3, screen=(BLOCKS,), screen_s=60, porch=30, mic=(MIC_BLOCKS,), mic_extra=0.25,
            tv_s=1800, notes=15, config=2, log_lines=2000, mix=FULL_MIX, cars=(2, 4), other_cars=(0, 2), other_min=2,
            part_floor={"screen": 250, "porch": 8, "phone": 15, "mic": 5})
# The owner's activity on the screen, calibrated against two real hours (TESTBED.md: per hour, over 150 plan hours, 2,360 seconds with a frame
# and 1.86 frames a second with a frame; over three encoded hours, 2,329 gated windows, 10% of the minutes with 2 or fewer gated windows, 34% of
# the frame pairs changing under 0.05%, median 0.084%, 90th percentile 7.6%). A state has a level, drawn as a chain over its block
# (the next state's level depends on the last's, so idle comes in stretches of a minute or more): `idle` has no acts (reading, thinking,
# away), `light` and `busy` have runs of active seconds, each with 1 or 2 acts (on the second's two ticks).
ACTIVITY = ("idle", "light", "busy")
ACTIVITY_START = (1, 2, 6)  # the odds of a block's first level
ACTIVITY_CHAIN = {"idle": (89, 4, 7), "light": (4, 50, 46), "busy": (18, 130, 852)}  # the odds of the next level, by the level before it
ACTIVE_SHARE = {"light": (0.2, 0.55), "busy": (0.72, 0.92)}  # the share of a state's seconds that are active, drawn for each state
ACTIVE_RUN_S = {"light": (2, 6), "busy": (6, 20)}  # ... in runs of this many seconds
TWO_ACTS = {"light": 0.7, "busy": 0.92}  # the odds that an active second has 2 acts and not 1
ACT_KINDS = ("move", "type", "scroll")
KIND_ODDS = {"editor": (1, 8, 1.5), "chat": (1, 9, 0.3), "slides": (2, 8, 1), "sheet": (2, 8, 1), "terminal": (1, 6, 3), "browser": (5, 1, 3)}  # by app
TYPE_CHARS = ((1, 2, 3), (3, 4, 3))  # a `type` adds this many characters, with these odds
SCROLL_LINES = ((1, 2, 3), (5, 3, 2))  # a `scroll` that is not an enter moves this many lines, with these odds
MOVE_PX = (15, 140, 90)  # a cursor move: at least this far, at most this far, and a spread of this much (px)
SCALES = {"tiny": Scale(days=2, asks=TINY_ASKS, extra=0, shared="last", early=False, photos=12, clips=1,
                        cars=(2, 2), other_cars=(1, 1), other_min=1,
                        screen=(((14 * HOUR, 14 * HOUR + 1200),), ((10 * HOUR, 10 * HOUR + 1200),)), screen_s=60, porch=6,
                        mic=(((7 * HOUR, 7 * HOUR + 900), (17 * HOUR, 17 * HOUR + 900)),), mic_extra=0.5, tv_s=240,
                        notes=4, config=1, log_lines=150, mix=TINY_MIX,
                        part_floor={"screen": 1, "porch": 1, "phone": 1, "mic": 1}),
          "pilot": Scale(days=3, extra=0, shared="day", early=True, **FULL),
          "standard": Scale(days=7, extra=1, shared=None, early=False, **FULL)}


@dataclass(frozen=True)
class File:
    path: str  # relative to the data folder, forward slashes; its first part is the source
    source: str
    t: float  # local epoch: when it was taken or starts
    name: str | None  # remember's name: "app.toml" for config versions, else None (the file name)
    kind: str  # "photo" | "clip" | "screen" | "porch" | "mic" | "note" | "config"
    spec: dict  # what to draw or play (kind-specific, JSON-able; the writer's business)


@dataclass(frozen=True)
class Plan:
    seed: int
    scale: str
    days: tuple[date, ...]
    files: tuple[File, ...]  # oldest first
    logs: dict[str, tuple[dict, ...]]  # service -> its lines ({"ts", "service", "level", "msg", "code"?}), in time order
    questions: tuple[dict, ...]  # in TYPES then template order


def days_before(today: date, n: int) -> list[date]:
    """The `n` days ending yesterday, oldest first."""
    return [today - timedelta(days=n - i) for i in range(n)]


def at(day: date, seconds: int) -> float:
    """The local epoch of `seconds` into `day` (86,400 is the next midnight), made from the clock and not by adding seconds."""
    if seconds >= 86400:
        day, seconds = day + timedelta(days=1), seconds - 86400
    return datetime(day.year, day.month, day.day, seconds // 3600, seconds // 60 % 60, seconds % 60).timestamp()


def clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def date_text(day: date) -> str:
    """`Tuesday 28 October`, from fixed English names (not `strftime`, which follows the locale)."""
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def part_of(minute: int) -> str:
    return next(part for part, (lo, hi) in PARTS.items() if lo * 60 <= minute < hi * 60)


def slot_key(slot: tuple[int, str]) -> tuple[int, int]:
    """Where a (day, part) slot is in time."""
    return slot[0], list(PARTS).index(slot[1])


def clip_range(part: str) -> tuple[int, int]:
    """The seconds of the day a photo, clip or note of `part` can be at: the part, inside DAY_START and DAY_END."""
    lo, hi = PARTS[part]
    return max(lo * HOUR, DAY_START), min(hi * HOUR, DAY_END)


PART_WEIGHTS = tuple(clip_range(part)[1] - clip_range(part)[0] for part in PARTS)


def share_out(total: int, weights: Sequence[float]) -> list[int]:
    """`total` split in proportion to `weights` in whole numbers (the largest remainders get the last ones)."""
    raw = [total * w / sum(weights) for w in weights]
    out = [int(x) for x in raw]
    for k in sorted(range(len(raw)), key=lambda k: (-(raw[k] - out[k]), k))[:total - sum(out)]:
        out[k] += 1
    return out


def spaced(r: random.Random, lo: int, hi: int, n: int, gap: int, avoid: Sequence[int] = (), what: str = "") -> list[int]:
    """`n` whole numbers in [lo, hi), each at least `gap` from the others and from `avoid`, in order."""
    out: list[int] = []
    for _ in range(n):
        for _ in range(2000):
            x = r.randrange(lo, hi)
            if all(abs(x - y) >= gap for y in (*avoid, *out)):
                out.append(x)
                break
        else:
            raise ValueError(f"{what}: no room for {n} times {gap} apart from {lo} to {hi}")
    return sorted(out)


# -- where a template's needles go --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spot:
    """Where a template's needles and absences are, as (day, part) slots (`absent_day` is a day): `last` the template's last
    needle (for a `week` question: nothing of the template comes after it, and a day or more of haystack does), `part` the one
    its part-level questions ask about (its part holds no other), `day` the one its day-level questions ask about (its day holds
    no other), `extras` earlier needles, `absent_part` a part and `absent_day` a day that hold none."""

    last: tuple[int, str] | None = None
    part: tuple[int, str] | None = None
    day: tuple[int, str] | None = None
    extras: tuple[tuple[int, str], ...] = ()
    absent_part: tuple[int, str] | None = None
    absent_day: int | None = None

    @property
    def needles(self) -> list[tuple[int, str]]:
        """The needles' slots, oldest first (there is one needle to a slot)."""
        return sorted({s for s in (self.last, self.part, self.day, *self.extras) if s is not None}, key=slot_key)


def last_days(n: int) -> range:
    """The day (0-based) a week's last needle is on: three full days of haystack follow it in a week of 5 or more (day 4 of 7: the
    probe's floors for a scan from the newest, 150 photos among them, want that), two in the 4 days of a short week and one in a
    shorter (day 2 of 3, day 1 of 2)."""
    return range(n - 4, n - 3) if n >= 5 else range(max(0, n - 2), max(1, n - 1))


def arrange(r: random.Random, what: str, n_days: int, levels: Sequence[str], absent: Sequence[str], avail: set, eligible: set,
            extra: int = 0, shared: str | None = None, early: bool = False) -> Spot:
    """Sample where a template's needles go so that each of its questions has the scope its level says (a `ValueError` names
    what cannot be met): a `part` question's part and a `day` question's day hold exactly one needle; a `week` question's answer
    is the last needle, on one of `last_days`, with nothing of the template after it and `extra` earlier needles besides; an
    absent question names a part (of `eligible`) or a day that holds none. `avail` are the slots the source has data in and
    `eligible` those that have enough to name in a question. `shared` and `early`: see `Scale`."""
    want_last, want_part, want_day = "week" in levels, "part" in levels, "day" in levels
    extra = extra if want_last else 0  # earlier needles only make a week question's answer the last of several
    for _ in range(5000):
        last = day_slot = part = None
        if want_last:
            day = r.choice(list(last_days(n_days)))
            options = sorted((s for s in avail if s[0] == day), key=slot_key)
            if not options:
                continue
            last = options[0] if early else r.choice(options)
        before = [s for s in avail if last is None or slot_key(s) < slot_key(last)]
        if want_day:
            days = sorted({s[0] for s in before if last is None or s[0] < last[0]})
            if not days:
                continue
            day = r.choice(days)
            options = sorted(s for s in (eligible if shared == "day" and want_last else avail) if s[0] == day)
            if not options:
                continue
            day_slot = r.choice(options)
        if want_part:
            if shared == "last" and last is not None:
                part = last if last in eligible else None
            elif shared == "day" and want_last and day_slot is not None:
                part = day_slot
            else:
                options = sorted(s for s in eligible if (last is None or slot_key(s) < slot_key(last))
                                 and (day_slot is None or s[0] != day_slot[0]))
                part = r.choice(options) if options else None
            if part is None:
                continue
        used = {s for s in (last, part, day_slot) if s is not None}
        pool = sorted(s for s in before if s not in used and (day_slot is None or s[0] != day_slot[0]))
        if len(pool) < extra:
            continue
        extras = tuple(sorted(r.sample(pool, extra), key=slot_key))
        taken = used | set(extras)
        a_part = a_day = None
        if "part" in absent:
            free = sorted(s for s in eligible if s not in taken)
            if not free:
                continue
            a_part = r.choice(free)
        if "day" in absent:
            free_days = [d for d in range(n_days) if all(s[0] != d for s in taken)]
            if not free_days:
                continue
            a_day = r.choice(free_days)
        return Spot(last, part, day_slot, extras, a_part, a_day)
    raise ValueError(f"{what}: no way to place needles for {tuple(levels)} (absent {tuple(absent)}, {extra} extra, shared {shared}, "
                     f"early {early}) in {n_days} days with data in {len(avail)} slots, {len(eligible)} of them big enough to name")


def cut(r: random.Random, lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) cut into states of 10 to 60 s, 30 s on average (a tail under 10 s joins the last state)."""
    out: list[tuple[int, int]] = []
    t = lo
    while t < hi:
        d = int(r.triangular(10, 60, 20))
        if hi - (t + d) < 10:
            d = hi - t
        out.append((t, t + d))
        t += d
    return out


def segments(starts: Sequence[int], lo: int, hi: int, step: int) -> list[tuple[int, int]]:
    """The segments `(start, end)` the recorder's segment muxer makes of a block [lo, hi) whose states start at `starts` (in order, the
    first at `lo`): a segment ends only on a key frame, which here is a state's start (x264's scene cut puts one at each state change; the
    real recorder also forces one every 10 s, so its cuts land mid-state: not modelled), so the n-th segment ends where the first state that starts at or after `lo + n * step`
    begins (a long state can make one segment end where the next would). The last ends at `hi`. A segment starts on a state's start,
    so none repeats a still frame at its start."""
    cuts = [starts[0]]
    for t in starts[1:]:
        if t >= lo + step * len(cuts):
            cuts.append(t)
    return list(zip(cuts, [*cuts[1:], hi]))


def next_level(r: random.Random, last: str | None) -> str:
    """The activity level of the state after one of level `last` (None: the first of a block)."""
    return r.choices(ACTIVITY, ACTIVITY_START if last is None else ACTIVITY_CHAIN[last])[0]


def state_acts(state: dict, at: tuple[int, int], level: str) -> tuple[list[list], tuple[int, int]]:
    """The owner's acts over `state` (its `app`, `seed`, `from`, `to`, and the accent and overlay that can hide its typing line), which
    starts with the cursor at `at`, at activity `level`, from the state's own seed: none when idle; else an active share of its seconds
    (ACTIVE_SHARE, drawn here) in runs (ACTIVE_RUN_S), each active second with 1 or 2 acts (TWO_ACTS) on its two ticks. An act is a cursor
    `move` to a place near the last (MOVE_PX), a `type` of 1 to 3 visible characters or a `scroll`, in the odds of the app (KIND_ODDS); a
    full typing line (`typing_cap`: the app's, or less where a decoration hides its end) is followed by a scroll of 1 line (enter), and
    any scroll starts the typing line afresh. A line that is hidden from its start is not typed. Returns
    `[[tick, "move", x, y] | [tick, "type", n] | [tick, "scroll", lines], ..]` (ticks are half seconds from the state's start, from 1, in
    order; `n` the visible characters on the typing line, `lines` the lines scrolled since the state's start) and where the cursor ends."""
    if level == "idle":
        return [], at
    seconds, app = state["to"] - state["from"], state["app"]
    r = _rng(state["seed"], "acts")
    share = r.uniform(*ACTIVE_SHARE[level])
    gap = lambda run: round(r.uniform(0.5, 1.5) * run * (1 - share) / share)  # noqa: E731  (idle seconds that give the share)
    acts: list[list] = []
    (x, y), typed, lines, sec = at, 0, 0, gap(3)
    cap = typing_cap(state, 0)
    while sec < seconds:
        run = r.randint(*ACTIVE_RUN_S[level])
        for _ in range(min(run, seconds - sec)):
            ticks = (2 * sec, 2 * sec + 1) if r.random() < TWO_ACTS[level] else (2 * sec + r.randint(0, 1),)
            for tick in (t for t in ticks if t >= 1):
                odds = KIND_ODDS[app] if cap else (KIND_ODDS[app][0], 0, KIND_ODDS[app][2])
                kind = "scroll" if typed >= cap > 0 else r.choices(ACT_KINDS, odds)[0]
                if kind == "move":
                    for _ in range(20):
                        nx = min(max(round(x + r.gauss(0, MOVE_PX[2])), 0), SCREEN[0] - 1)
                        ny = min(max(round(y + r.gauss(0, MOVE_PX[2])), 0), SCREEN[1] - 1)
                        if MOVE_PX[0] <= math.dist((x, y), (nx, ny)) <= MOVE_PX[1]:
                            break
                    else:  # (cornered: any other place)
                        nx, ny = r.randrange(SCREEN[0]), r.randrange(SCREEN[1])
                    x, y = nx, ny
                    acts.append([tick, "move", x, y])
                elif kind == "type":
                    typed = min(cap, typed + r.choices(*TYPE_CHARS)[0])
                    acts.append([tick, "type", typed])
                else:
                    lines += 1 if typed >= cap > 0 else r.choices(*SCROLL_LINES)[0]
                    typed = 0
                    cap = typing_cap(state, lines)
                    acts.append([tick, "scroll", lines])
            sec += 1
        sec += gap(run)
    return acts, (x, y)


def clip_length(r: random.Random, least: int = 10) -> int:
    """A porch clip's seconds: 10 to 30, 16 on average, at least `least`."""
    while (length := int(r.triangular(10, 30, 10))) < least:
        pass
    return length


# -- what the files hold ------------------------------------------------------------------------------------------------------

def needle_moments(plan: Plan, template: str) -> list[list[tuple[float, float]]]:
    """Each needle of `template` as the spans of time its files show it in, read from the files' specs and not from the questions,
    oldest first: a sign or a car one photo, an error its state, a parcel its clips (the one it is left in, the ones between and the
    one it is taken in), a chime or an alarm its sound, a note with a code the note."""
    out: list[list[tuple[float, float]]] = []
    for f in plan.files:
        spec = f.spec
        if template == "note-code" and f.kind == "note" and spec.get("needle"):
            out.append([(f.t, f.t)])
        elif template == "photo-sign" and f.kind == "photo" and spec["scene"] == "sign" and tuple(spec["rgb"]) == NEEDLE_RGB["sign"]:
            out.append([(f.t, f.t)])
        elif template == "photo-count" and f.kind == "photo" and spec["scene"] == "car" and tuple(spec["rgb"]) == NEEDLE_RGB["car"]:
            out.append([(f.t, f.t)])
        elif template == "screen-error" and f.kind == "screen":
            out += [[(f.t + s["from"], f.t + s["to"])] for s in spec["states"] if (s["overlay"] or {}).get("kind") == "error"]
        elif template == "porch-parcel" and f.kind == "porch" and spec["parcel"]:
            a, b = spec["parcel"]
            if a > 0:  # the clip the parcel is left in
                out.append([(f.t + a, f.t + b)])
            elif out:  # the clips it is on the step in, and the one it is taken in
                out[-1].append((f.t + a, f.t + b))
            else:
                raise AssertionError(f"{f.path}: a parcel on the step that no clip left there")
        elif template in ("mic-doorbell", "mic-alarm") and f.kind == "mic":
            kind = template.split("-")[1]
            out += [[(f.t + s["at"], f.t + s["at"] + s["dur"])] for s in spec["sounds"] if s["kind"] == kind]
    return sorted(out)


def same_spans(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=1e-6) for u, v in zip(a, b) for x, y in zip(u, v))


def verify(plan: Plan) -> None:
    """Each media (and note) question's scope holds what its level says, counted from the files' specs: a `part` or `day` question's
    scope holds exactly its template's one needle (its moments are that needle's and it has no others); a `week` question's
    answer is the template's last needle, its `others` are all the earlier ones, and a full day of haystack or more follows it; an
    absent question's scope holds none; a count is the number of needles in its scope. An `AssertionError` names the question."""
    found: dict[str, list] = {}
    for q in plan.questions:
        base = BASE_OF.get(q["template"], q["template"])
        if q["level"] == "time" or base not in ("note-code", "photo-sign", "photo-count", "screen-error", "porch-parcel", "mic-doorbell", "mic-alarm"):
            continue
        needles = found.setdefault(base, needle_moments(plan, base))
        inside = [m for m in needles if q["since"] <= m[0][0] < q["until"]]
        who, truth = f"{q['qid']} ({q['level']})", q["truth"]
        if q["absent"]:
            if inside:
                raise AssertionError(f"{who}: an absent question's scope holds {len(inside)} needles")
        elif base == "photo-count":
            if len(inside) != int(truth["value"]) or not same_spans([s for m in inside for s in m], truth["moments"]):
                raise AssertionError(f"{who}: the scope holds {len(inside)} red cars, the answer is {truth['value']}")
        elif q["level"] in ("part", "day"):
            if len(inside) != 1 or not same_spans(inside[0], truth["moments"]) or truth["others"]:
                raise AssertionError(f"{who}: the scope holds {len(inside)} needles, not exactly the question's own")
        else:
            last = needles[-1]
            if not q["last"] or not same_spans(last, truth["moments"]) or not same_spans([s for m in needles[:-1] for s in m], truth["others"]):
                raise AssertionError(f"{who}: the answer is not the template's last needle, or its others are not all the earlier ones")
            if len(needles) < 2 and plan.scale != "tiny":
                raise AssertionError(f"{who}: the template has no earlier needle for a wrong answer")
            if q["until"] - max(b for _, b in last) < 86400:
                raise AssertionError(f"{who}: less than a day of haystack follows the last needle")
            if any(not q["since"] <= m[0][0] < q["until"] for m in needles):
                raise AssertionError(f"{who}: a needle is outside the week")
