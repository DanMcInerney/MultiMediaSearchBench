"""Which questions of the download are asked (moved from egolife.py of this repository at bb498a6): for each of four ages (the query time
less the latest target: under 2 h, 2-6 h, 6-24 h, over 24 h), 10 of the questions that need no sound, chosen by the rarest categories
first, and 10 that need sound, asked from the transcript alone (`select`). Every question is asked as of its horizon H, the end of the last
whole clip that ended by the query: no configuration may read a clip or a transcript line past H. What is lost is the seconds between H and
the question: under 30, except after a gap in the recording."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence

from mmmb.scenarios.v8_egolife.source import DAY_S, DAYS, Clip, plain


AGES = (("0-2h", 7200), ("2-6h", 21600), ("6-24h", DAY_S), ("24h+", math.inf))  # the bins of a question's age, with their ends
AGE_NAMES = tuple(name for name, _ in AGES)
PER_AGE, SPEECH, PILOT_PER_AGE, PILOT_SPEECH = 10, 10, 2, 2  # questions per age bin and for speech; the pilot's share of each

NEAR_S = 30.0  # a speech question is asked when a transcript line is within this of a target


def horizon(clips: Sequence[Clip], t: float) -> Clip | None:
    """The last whole clip at or before plain time `t` (the one whose end is the latest that is not after it): the question's
    horizon H is its end. None when no clip had ended by `t`."""
    ended = [c for c in clips if c.t1 <= t]
    return max(ended, key=lambda c: c.t1) if ended else None


def age_of(item: dict) -> str:
    """The age bin of a question: its query time less its latest target."""
    age = item["query"] - max(item["targets"])
    return next(name for name, end in AGES if age < end)


def pools(items: Sequence[dict], clips: Sequence[Clip], lines: Sequence[dict], days: int = DAYS) -> tuple[list[dict], list[dict], Counter]:
    """The questions the bench can ask, `(video, speech, dropped)`. A question is asked when its query is on days 1..`days`, a clip
    had ended by then, and each of its targets is inside a clip that ended by the horizon; a speech question (needs sound) also
    when a transcript line that ended by the horizon is within NEAR_S of a target. `dropped` counts the others by why."""
    video, speech, dropped = [], [], Counter()
    for item in sorted(items, key=lambda i: int(i["id"])):
        if item["query"] >= plain(days + 1, 0):
            dropped["its query is after the last day"] += 1
            continue
        if not item["targets"]:
            dropped["its time is a range or a list in one string"] += 1
            continue
        top = horizon(clips, item["query"])
        if top is None:
            dropped["no clip had ended by its query"] += 1
            continue
        held = [c for c in clips if c.t1 <= top.t1]
        if not all(any(c.t0 <= t <= c.t1 for c in held) for t in item["targets"]):
            dropped["a target is outside the clips (a gap) or after the horizon"] += 1
            continue
        if item["audio"] and not any(line["start"] <= t + NEAR_S and line["end"] >= t - NEAR_S and line["end"] <= top.t1
                                     for t in item["targets"] for line in lines):
            dropped["speech: no transcript line near a target"] += 1
            continue
        (speech if item["audio"] else video).append(item)
    return video, speech, dropped


def rotate(queues: Sequence[Sequence], n: int) -> list:
    """`n` items taken one from each queue in turn, the first queue first; a queue that runs out is skipped."""
    queues, out = [list(q) for q in queues], []
    while len(out) < n and any(queues):
        for queue in queues:
            if queue and len(out) < n:
                out.append(queue.pop(0))
    return out


def by_category(items: Sequence[dict]) -> list[list[dict]]:
    """The items split by category, the rarest category first (ties by name), each in the order given."""
    groups = {c: [i for i in items if i["category"] == c] for c in sorted({i["category"] for i in items})}
    return [groups[c] for c in sorted(groups, key=lambda c: (len(groups[c]), c))]


def binned(items: Sequence[dict]) -> dict[str, list[dict]]:
    return {name: [i for i in items if age_of(i) == name] for name in AGE_NAMES}


def select(items: Sequence[dict], clips: Sequence[Clip], lines: Sequence[dict], seed: int, per_age: int = PER_AGE,
           speech: int = SPEECH, days: int = DAYS) -> list[dict]:
    """The questions of a seed: `per_age` video questions for each age bin, then `speech` speech ones, each item with its `age`
    and `pilot` (the first PILOT_PER_AGE of its bin, or PILOT_SPEECH of the speech ones). A bin is shuffled by the seed and its
    questions taken from the categories in turn, the rarest first; the speech ones from the bins in turn, the rarest first, and
    within a bin from the categories in turn. A pool too short for its share is a `ValueError` naming it."""
    video, heard, _ = pools(items, clips, lines, days)
    chosen = []
    for name, of in binned(video).items():
        if len(of) < per_age:
            raise ValueError(f"the {name} age bin holds {len(of)} questions the bench can ask, fewer than {per_age}")
        random.Random(f"{seed}-{name}").shuffle(of)
        chosen += [{**i, "age": name, "pilot": rank < PILOT_PER_AGE} for rank, i in enumerate(rotate(by_category(of), per_age))]
    queues = {}
    for name, of in binned(heard).items():
        random.Random(f"{seed}-speech-{name}").shuffle(of)
        queues[name] = rotate(by_category(of), len(of))
    if sum(len(q) for q in queues.values()) < speech:
        raise ValueError(f"the speech pool holds {sum(len(q) for q in queues.values())} questions, fewer than {speech}")
    order = sorted(queues, key=lambda n: (len(queues[n]), AGE_NAMES.index(n)))
    chosen += [{**i, "age": age_of(i), "pilot": rank < PILOT_SPEECH} for rank, i in enumerate(rotate([queues[n] for n in order], speech))]
    return chosen
