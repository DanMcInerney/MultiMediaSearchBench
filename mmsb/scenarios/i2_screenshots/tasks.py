"""The tasks of I2: the episodes of the source grouped by what the person was doing (their goal), which of them a question can rest on, and how
a person says them.

A task is the episodes with one goal. A question about a task is certain only where its screens are that task's, so a task is asked about when:
- every episode of it is whole (source.py);
- its goal is one of a family of goals that differ by a name alone, such as the flights between two cities or the weather of one city (`TOPICS`),
  and no other goal of the source asks the same of the same names in other words (`price of the Sony TV` and `how much does the Sony TV cost`), nor
  holds its words in its own (`GameSpot` in `GameSpot eSports`): a person's word for the task is then not another task's;
- its goal is on a screen of the episode (`on_screen`): in some episodes of the source the screens are another task's, which the goal then
  mislabels;
- no more than OTHERS screens of other episodes show all its words (the browser's recent searches and suggestions do, of the tasks done on that
  phone before).
A task asked about as absent has none of those screens (`OTHERS` is 0) and its episodes are all taken out of the pile.

The screens of a task are those that show its words (`showing`): the first screens of an episode are often the last page of the task before it (a home
screen, another search's results), which no one can tie to the task, so the episodes of every task a question can rest on are cut to the screens that
show its words (`trim`) and the rest are left out of the folder. A task's set is then every screen of it that is there, and each shows it."""

from __future__ import annotations

import dataclasses
import functools
import re
from collections import defaultdict
from difflib import SequenceMatcher
from collections.abc import Sequence
from dataclasses import dataclass

from mmsb.scenarios.i2_screenshots.source import Episode, Shot, fold

OTHERS = 8  # the screens of other episodes that may show all a real task's words
ON_SCREEN = 0.75  # the share of the goal's words some screen of its episode shows
GARBLE = 0.7  # how alike a word OCR read and a name of the goal must be (difflib's ratio) for the screen to be taken as showing the name
MIN_SHOTS, MAX_SHOTS = 2, 20  # a task's screens (those that show its words): below, one is no set; above, too long a list to ask for whole
MAX_EPISODES = 3
STOP = frozenset("a an the of in on at for to from is it are what whats how do does i me my show some get nearest search find check open play "
                 "install set new google latest".split())
IMPERATIVE = ("open", "check", "find", "play", "install", "search", "set", "toggle", "turn", "google")
TOPICS = {  # what is asked, and its goals' wordings: the groups of a wording are the names (a city, a brand, an app) that tell one goal from another
    "flights": (r"search for flights from (.+) to (.+)",),
    "hotels": (r"search for hotels in (.+)",),
    "weather": (r"what's the weather like in (.+)\?",),
    "news": (r"what's the news in (.+)\?",),
    "rent": (r"how much does a \d bedroom apartment rent for in (.+)\?",),
    "menu": (r"what's on the menu at (.+)\?",),
    "alarm": (r"set an alarm for (\d+(?:am|pm))",),
    "settings": (r"check the settings for the (.+) app",),
    "install": (r"install the (.+) app",),
    "new video": (r"play the new (.+) video on youtube",),
    "video from": (r"play the latest video from (.+)", r"what's the latest video from (.+)\?"),
    "tv price": (r"what's the price of the (.+) tv\?", r"how much does the (.+) tv cost\?"),
    "capital": (r"what is the capital of (.+)\?", r"google the capital of (.+)"),
    "nearest": (r"how do i get to the nearest (.+)\?",),
    "beach": (r"where can i buy a nice beach (.+)\?",),
    "dollar": (r"what's (?:the )?us dollar exchange rate against the (.+)\?",),
    "time": (r"what time is it in (.+)\?", r"what's the time in (.+)\?"),
    "restaurants": (r"search for good (.+) restaurants",),
    "restaurant": (r"what's a good restaurant in (.+)\?",),
    "big": (r"how big is the (.+)\?",),
    "speed": (r"what is the speed of a (.+)\?",),
}
WORDINGS = [(topic, re.compile(p, re.IGNORECASE)) for topic, ps in TOPICS.items() for p in ps]


@dataclass(frozen=True)
class Task:
    goal: str
    episodes: tuple[Episode, ...]

    @property
    def shots(self) -> tuple[Shot, ...]:
        return tuple(s for e in self.episodes for s in e.shots)

    def says(self) -> tuple[str, str]:
        """`(past, plain)`: how a person puts the task after "when I" and after "Did I ever": `was trying to open the calendar app` and
        `try to open the calendar app`; a goal that is a question is what they `googled`."""
        goal = self.goal.rstrip("?. ")
        if goal.split()[0].casefold() in IMPERATIVE:
            act = goal[0].lower() + goal[1:]
            return f"was trying to {act}", f"try to {act}"
        return f'googled "{goal}"', f'google "{goal}"'


def key_of(goal: str) -> tuple[str, ...] | None:
    """What a goal asks and of what, whatever its words: `("tv price", "sony")`; None for a goal of no family of TOPICS."""
    for topic, pattern in WORDINGS:
        if found := pattern.fullmatch(goal):
            return (topic, *(fold(g) for g in found.groups()))
    return None


def content(goal: str) -> frozenset[str]:
    """The words of `goal` that tell it from another, with a trailing s taken off."""
    return frozenset(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in fold(goal).split() if w not in STOP)


def stems(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in fold(text).split())


@functools.cache
def words_of(shot: Shot) -> frozenset[str]:
    return stems(" ".join(shot.ocr))


@functools.lru_cache(maxsize=None)
def like(name: str, vocabulary: frozenset[str]) -> frozenset[str]:
    """The words of `vocabulary` that are `name` or what OCR may have made of it, a few letters wrong (GARBLE)."""
    matcher = SequenceMatcher(None)
    matcher.set_seq2(name)
    out = set()
    for w in vocabulary:
        if abs(len(w) - len(name)) <= 2:
            matcher.set_seq1(w)
            if w == name or (matcher.real_quick_ratio() >= GARBLE and matcher.quick_ratio() >= GARBLE and matcher.ratio() >= GARBLE):
                out.add(w)
    return frozenset(out)


def showing(shot: Shot, wanted: frozenset[str]) -> bool:
    """Whether the screen shows ON_SCREEN of the words `wanted`."""
    return len(wanted & words_of(shot)) >= ON_SCREEN * len(wanted)


def on_screen(episode: Episode, wanted: frozenset[str]) -> bool:
    """Whether some screen of `episode` shows ON_SCREEN of the words `wanted`."""
    return any(showing(s, wanted) for s in episode.shots)


def groups(episodes: Sequence[Episode]) -> list[Task]:
    by_goal = defaultdict(list)
    for e in episodes:
        by_goal[e.goal].append(e)
    return [Task(goal, tuple(es)) for goal, es in sorted(by_goal.items())]


def alone(goal: str, others: Sequence[str]) -> bool:
    """Whether no other goal of `others` asks the same of the same names (`key_of`), and none has `goal`'s words in its own or its own in
    `goal`'s, in order and next to one another: `GameSpot` and `GameSpot eSports`."""
    mine, key = " " + fold(goal) + " ", key_of(goal)
    for other in others:
        if other == goal:
            continue
        theirs = " " + fold(other) + " "
        if mine in theirs or theirs in mine or (key is not None and key == key_of(other)):
            return False
    return True


def asked_tasks(pile: Sequence[Episode], others: int | None, held: bool = False, cut: bool = True) -> list[tuple[Task, int]]:
    """The tasks of `pile` a question can rest on, each with the number of screens of other episodes that show all its words, that being at most
    `others` (None: no limit). A task to hold out (`held`) need not be whole or on screen: all its episodes go. Every screen of a task a question rests
    on shows its words (`cut`; `trim` asks for the tasks before they are cut)."""
    every = groups(pile)
    goals = [t.goal for t in every]
    screens = [(s.episode, words_of(s)) for e in pile for s in e.shots]
    vocabulary = frozenset().union(*(w for _, w in screens))
    out = []
    for task in every:
        wanted = content(task.goal)
        mine = {e.id for e in task.episodes}
        if not (len(wanted) >= 2 and key_of(task.goal) is not None and alone(task.goal, goals)):
            continue
        if not held and not (all(e.whole for e in task.episodes) and len(task.episodes) <= MAX_EPISODES and MIN_SHOTS <= len(task.shots) <= MAX_SHOTS
                             and all(on_screen(e, wanted) for e in task.episodes) and (not cut or all(showing(s, wanted) for s in task.shots))):
            continue
        names = stems(" ".join(key_of(task.goal)[1:])) & wanted
        plain, near = wanted - names, [like(n, vocabulary) for n in names]  # a screen shows the goal when it has the plain words and each name, as OCR may read it
        shown = sum(1 for episode, words in screens if episode not in mine and plain <= words and all(n & words for n in near))
        if others is None or shown <= others:
            out.append((task, shown))
    return out


def trim(pile: Sequence[Episode], whole: set[str]) -> list[Episode]:
    """`pile` with each episode of a task a question can rest on (`asked_tasks` with no limit on the other screens), but those of ids in `whole`, cut to
    the screens that show the task's words; its `length` is then the screens it has left, so that it is whole."""
    cut: dict[str, Episode] = {}
    for task, _ in asked_tasks(pile, None, cut=False):
        wanted = content(task.goal)
        for e in task.episodes:
            if e.id not in whole:
                kept = tuple(s for s in e.shots if showing(s, wanted))
                cut[e.id] = dataclasses.replace(e, shots=kept, length=len(kept))
    return [cut.get(e.id, e) for e in pile]
