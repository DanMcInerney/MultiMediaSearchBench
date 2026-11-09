"""The questions of I2, every answer from the labels (source.py), each asked the morning after the folder's last screenshot. A person asks of a
folder of their phone's screenshots by what they remember of it: what they were doing, or a few words they read:

| template  | question                                                                                    | rule                  |
| `tasks`   | Show me every screenshot from when I was trying to check the settings for the Instagram app. | `set` of file names   |
| `size`    | How many screenshots did I take when I googled "What's the weather like in Chicago"?         | `count`               |
| `did`     | Did I ever try to install the Lyft app? If so, which was the first screenshot of it?         | `set` of one file     |
| `words`   | Find the screenshot that says "consent to our use of cookies".                               | `set` of one file     |
| `read`    | What was the humidity when I looked up the weather in Chicago?  (written by reading, READS)  | `text`                |
| `absent_*`| `tasks`, `size`, `did` of a task the pile has no screenshot of, or `words` of a phrase it has none of | `absent`    |

A task is what the source gives as an episode's goal (tasks.py says which a question can rest on); its screenshots are the episode's that show the task's words
(the others were left out of the folder, tasks.py `trim`), taken a few seconds apart in one sitting, and a person finds them by the sitting. An absent question is a held-out one: the task's episodes (or the phrase's)
were taken out of the pile, with every other screen that shows them (tasks.py, words.py), so that the file list and the screens that remain give it
no answer. Which questions are asked is drawn with the seed, a family of tasks (flights, weather, apps) at a time so that the families vary; the
`read` questions are written by reading the screens, and are the same at every seed."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mmsb.core.question import Grade, Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i2_screenshots.source import Episode, Shot, seen
from mmsb.scenarios.i2_screenshots.tasks import OTHERS, Task, asked_tasks, key_of
from mmsb.scenarios.i2_screenshots.words import Phrase, phrases, spread

CODE = "I2"
SOURCE_NAME = "phone"  # the dump's source
QUOTAS = {"tiny": {"tasks": 1, "size": 1, "did": 1, "words": 1, "absent_tasks": 1, "absent_did": 1},
          "standard": {"tasks": 5, "size": 3, "did": 2, "words": 4, "absent_tasks": 2, "absent_size": 1, "absent_did": 1, "absent_words": 1}}
FILES = "<the names of the files, as in Screenshot_20260101-120000.png, separated by commas>"
FILE = "<the name of the file, as in Screenshot_20260101-120000.png>"
NUMBER = "<a number>"
SHORT = "<a short answer>"


@dataclass(frozen=True)
class Read:
    """A question written by reading the screens: every screen of the pile that shows the `where` words reads the same, `check`, which holds `accepted`."""

    key: str
    text: str
    episode: str  # the episode to read it in, which the tiny pile holds when `tiny`
    where: tuple[str, ...]  # folded words that a screen shows for it to be the one asked about
    check: str  # a folded phrase that each of them shows
    accepted: str  # the answer's phrases, `|` between: a short answer holding one of them is right
    tiny: bool = False


READS = (
    Read("humidity-chicago", "What was the humidity in the weather I looked up for Chicago?", "1275373898822310304", ("chicago", "humidity"),
         "humidity 58", "58"),
    Read("humidity-los-angeles", "What humidity did the weather I looked up for Los Angeles show?", "14783885567009844024", ("angeles", "humidity"),
         "humidity 76", "76", tiny=True),
    Read("downloads-alexa", "How many downloads did the Play Store say the Amazon Alexa app has?", "16085273296455767348", ("alexa", "downloads"),
         "100m", "100m|100 million|100 m|100 000 000"),
    Read("downloads-lyft", "How many downloads did the Play Store show for the Lyft app?", "4512649956411865155", ("lyft", "downloads"), "50m",
         "50m|50 million|50 m|50 000 000"),
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:64].strip("-")


def pick(candidates: Sequence[Task], n: int, rng: random.Random) -> list[Task]:
    """`n` of `candidates`, a family of tasks (their goals' topic) at a time in a shuffled order so that the families vary."""
    by_topic: dict[str, list[Task]] = {}
    for task in candidates:
        by_topic.setdefault(key_of(task.goal)[0], []).append(task)
    for of in by_topic.values():
        rng.shuffle(of)
    order = sorted(by_topic)
    rng.shuffle(order)
    out: list[Task] = []
    while len(out) < n and any(by_topic.values()):
        for topic in order:
            if by_topic[topic] and len(out) < n:
                out.append(by_topic[topic].pop())
    if len(out) < n:
        raise ValueError(f"only {len(out)} tasks of the {len(candidates)} there are can be asked about, and {n} are wanted")
    return out


def questions_of(pile: Sequence[Episode], held: Sequence[Task], everything: Sequence[Episode], paths: Mapping[str, str], now: float,
                 times: Mapping[str, float], scale: str, seed: int) -> tuple[Question, ...]:
    """The questions of `pile` (the episodes the folder holds), at `scale`, drawn with `seed`. `held` are the tasks taken out of it, which the
    absent questions ask about; `everything` are all the source's episodes (for the words that are ordinary); `paths` are the dump path of each
    screenshot file of the source (by `Shot.file`) and `times` its epoch seconds; `now` is epoch seconds."""
    rng = random.Random(f"i2:{seed}:{scale}")
    quota = QUOTAS[scale]
    shots = [s for e in pile for s in e.shots]
    since, until = min(times[s.file] for s in shots), max(times[s.file] for s in shots) + 1
    out: list[Question] = []

    def ask(qid: str, text: str, template: str, shape: str, grade: Grade, found: Sequence[Shot], value: str | None) -> None:
        absent = grade.rule == "absent"
        out.append(realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text, kind="absent" if absent else template,
                                     template="absent_" + template if absent else template, modality="image", shape=shape, grade=grade,
                                     evidence=[Span(paths[s.file], 0.0, 0.0) for s in found], sources=(SOURCE_NAME,), since=since, until=until,
                                     value=value))

    def names(found: Sequence[Shot]) -> list[str]:
        return [paths[s.file].rsplit("/", 1)[-1] for s in found]

    read_in = {r.episode for r in READS}
    real = [(t, n) for t, n in asked_tasks(pile, OTHERS) if not read_in & {e.id for e in t.episodes}]
    chosen = pick([t for t, _ in real], quota["tasks"], rng)
    rest = [t for t, _ in real if t not in chosen and len(t.episodes) == 1]
    sized = pick(rest, quota["size"], rng)
    rest = [t for t in rest if t not in sized]
    asked_first = pick(rest, quota["did"], rng)
    for task in chosen:
        past, _ = task.says()
        found = names(task.shots)
        ask(f"tasks-{slug(task.goal)}", f"Show me every screenshot from when I {past}.", "tasks", FILES, realdump.grade("set", items=found), task.shots,
            ", ".join(found))
    for task in sized:
        past, _ = task.says()
        ask(f"size-{slug(task.goal)}", f"How many screenshots did I take when I {past}?", "size", NUMBER,
            realdump.grade("count", truth=str(len(task.shots))), task.shots, str(len(task.shots)))
    for task in asked_first:
        _, plain = task.says()
        first = task.shots[:1]
        ask(f"did-{slug(task.goal)}", f"Did I ever {plain}? If so, which was the first screenshot of it?", "did", FILE,
            realdump.grade("set", items=names(first)), first, names(first)[0])
    spoken = spread(everything)
    for phrase in pick_phrases(phrases(shots, [], spoken, True), quota["words"], rng):
        ask(f"words-{slug(phrase.text)}", f'Find the screenshot that says "{phrase.text}".', "words", FILE,
            realdump.grade("set", items=names([phrase.shot])), [phrase.shot], names([phrase.shot])[0])
    for read in READS:
        if scale == "tiny" and not read.tiny:
            continue
        found = [s for s in shots if set(read.where) <= set(seen(s).split())]
        if bad := [s.file for s in found if f" {read.check} " not in seen(s)]:
            raise ValueError(f"{read.key}: {bad[0]} shows {' '.join(read.where)} and not {read.check!r}")
        if not any(s.episode == read.episode for s in found):
            raise ValueError(f"{read.key}: episode {read.episode} is not in the pile or does not show {read.check!r}")
        ask(f"read-{read.key}", read.text, "read", SHORT, realdump.grade("text", truth=read.accepted), found, read.accepted.split("|")[0])
    # absent: a task whose episodes were taken out of the pile, a phrase that only such screens had
    gone = iter(pick(held, sum(quota.get(f"absent_{t}", 0) for t in ("tasks", "size", "did")), rng))
    for template in ("tasks", "size", "did"):
        for task in [next(gone) for _ in range(quota.get(f"absent_{template}", 0))]:
            past, plain = task.says()
            text = {"tasks": f"Show me every screenshot from when I {past}.", "size": f"How many screenshots did I take when I {past}?",
                    "did": f"Did I ever {plain}? If so, which was the first screenshot of it?"}[template]
            ask(f"{template}-{slug(task.goal)}", text, template, {"tasks": FILES, "size": NUMBER, "did": FILE}[template], realdump.grade("absent"), [],
                None)
    if quota.get("absent_words"):
        gone_shots = [s for t in held for s in t.shots]
        for phrase in pick_phrases(phrases(gone_shots, shots, spoken, False), quota["absent_words"], rng):
            ask(f"words-{slug(phrase.text)}", f'Find the screenshot that says "{phrase.text}".', "words", FILE, realdump.grade("absent"), [], None)
    if len({q.qid for q in out}) != len(out):
        raise ValueError("two questions have one id")
    return realdump.split(out, seed)


def pick_phrases(candidates: Sequence[Phrase], n: int, rng: random.Random) -> list[Phrase]:
    """`n` of `candidates` from `n` different episodes."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    out: list[Phrase] = []
    for phrase in shuffled:
        if len(out) < n and phrase.shot.episode not in {p.shot.episode for p in out}:
            out.append(phrase)
    if len(out) < n:
        raise ValueError(f"only {len(out)} phrases of the {len(candidates)} there are can be asked about, and {n} are wanted")
    return out
