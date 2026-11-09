"""M3, the voicemail box: over a thousand robocalls in one folder under numbers, each with the time it came in. **It is not one person's box:**
the messages are the Robocall Audio Dataset's recordings, caught by honeypots and left as voicemails by phone users, that the FTC linked to in
its letters to the carriers in 31 cases (Social Security, Amazon and Apple orders, student loans, a car's warranty, a cut-off electricity bill),
and their times are invented (a seeded spread over nine weeks; the source has none). The questions ask to find every message that mentions a
scheme, how many do, whether and when the first did, which one a name is in, what amount or percentage one named by its minute said, who it
said it was from, and of schemes that no message has (questions.py). The truth is from the dataset's machine-made transcripts (Whisper's
medium model), so what a message "says" is what its transcript has; the scenario is the names of a `Scenario`, as functions (DESIGN 5), over
the shared builder (realdump).

    recipe = m3_voicemail.recipe(scale="tiny", seed=1)      # the source, the scale's days, this machine's UTC offsets
    dump, questions = m3_voicemail.build(recipe, folder, 3)  # the messages copied in (samples only) with their times, and the questions

Scales: `tiny`, 90 messages of a fortnight (2 to 9 MB); `standard`, the dataset's English messages but those a scheme that is asked about
leaves out (questions.py), about 1,200 messages of nine weeks, 7 September to 8 November 2026 (about 1 GB). The source gives no time: the
messages are in a seeded order, numbered in the order they came in, and their times a seeded spread over the days, in the hours of a day, two
minutes at least from the end of the message before."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.m3_voicemail.questions import ABSENT, CALLERS, CODE, SCHEMES, SOURCE_NAME, choose, questions_of
from mmsb.scenarios.m3_voicemail.source import SOURCE, read_calls, seconds_of, with_samples_only
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "m3_voicemail"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Voicemails", "audio", "home", "over a thousand robocalls from honeypots and voicemails of 31 FTC cases, in one folder, with invented times over nine weeks")
filing = with_absent({"scheme": ("find_all", "speech"), "count": ("how_many", "speech"), "first": ("did", "speech"), "single": ("find", "speech"),
                      "price": ("what", "speech"), "percent": ("what", "speech"), "caller": ("who", "speech")})

SCALES = {"tiny": ("2026-10-26", "2026-11-08"), "standard": ("2026-09-07", "2026-11-08")}  # the days the messages come in over (Monday to Sunday)
RECIPE_KEYS = {"source", "scale", "seed", "days", "utc_offsets"}
FOLDER = "voicemail"
HOURS = (8, 21)  # the hours of a day a message comes in
GAP_S = 120  # at least this long between the end of a message and the next one's time: one a minute of its own, and no citation reaches another
WHAT = "my voicemail box"
LAYOUT = (f"{FOLDER} holds the messages, msg_0001.wav onwards in the order they came in, one call each (WAV, mono, 16 or 8 kHz, a few "
          "seconds to some minutes); a message's modified time is when it came in.")
SOURCE_NAMES = ("robocall", "pointofnoentry", "wspr", "ncsu", "normalized")  # the source's names, which no file of the dump holds
COMMAND_WORDS = ("robocall", "pointofnoentry", "wspr-ncsu", "audio-wav-16khz", "_normalized")  # and which no command may


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SOURCE) -> dict:
    """What the dump is made from: the source, the `scale` (`tiny` or `standard`), the `seed` the schemes, messages, times and questions
    are drawn with, the days the messages come in over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def spread(lengths: Sequence[float], days: Sequence[str], rng: random.Random) -> list[float]:
    """The times (epoch seconds, in order) of messages that play `lengths`: seeded minutes of the days' hours, each moved later when it would be
    within GAP_S of the end of the message before."""
    per_day = (HOURS[1] - HOURS[0]) * 60
    out: list[float] = []
    free = 0.0
    for slot, length in zip(sorted(rng.sample(range(len(days) * per_day), len(lengths))), lengths, strict=True):
        day, minute = divmod(slot, per_day)
        start = realdump.local_t(datetime.combine(date.fromisoformat(days[day]), time(HOURS[0])))
        out.append(max(start + minute * 60 + rng.randrange(60), free))
        free = out[-1] + length + GAP_S
    return out


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed, days = str(recipe["scale"]), int(recipe["seed"]), recipe["days"]
    calls, cases = read_calls(str(recipe["source"]))
    roles, kept = choose(calls, cases, scale, seed)
    times = spread([c.seconds for c in kept], days, random.Random(f"m3:{seed}:{scale}:times"))
    files = tuple(File(f"{FOLDER}/msg_{n:04d}.wav", SOURCE_NAME, t, "message", "audio", {"src": c.file}, length=c.seconds)
                  for n, (c, t) in enumerate(zip(kept, times, strict=True), 1))
    now = realdump.local_t(datetime.combine(date.fromisoformat(days[-1]) + timedelta(days=1), time(9)))
    return files, questions_of(calls, kept, files, roles, scale, seed, now)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the box into `out` (new or empty): each message copied with its samples alone, and its time; `workers` at once. Returns the
    dump and the questions."""
    files, questions = plan_of(recipe)
    source = Path(str(recipe["source"]))

    def write(f: File, path: Path) -> None:
        wav = with_samples_only((source / str(f.spec["src"])).read_bytes())
        path.write_bytes(wav)
        if abs(seconds_of(path) - f.length) > 1e-6:
            raise ValueError(f"{f.spec['src']} plays {seconds_of(path)} s, not the {f.length} s the labels were read with")

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def label_words() -> list[str]:
    """The words the probe looks for in a message's name and metadata: every scheme and caller the questions name, as a phrase and by its
    words, and the source's names."""
    said = [s.said for s in (*SCHEMES.values(), *ABSENT.values())]
    called = [w for names, _ in CALLERS.values() for w in names.split("|")]
    return [*said, *called, *SOURCE_NAMES]


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no message's name or WAV header names a scheme or caller of the questions, or the dataset."""
    return realdump.probe(dump, out, label_words())


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The box itself: nothing is cut (every question is asked after its last message)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
