"""I7, the receipts: a folder of scanned receipts and bills, scanned in a few sittings. It merges two countries' receipts: SROIE's Malaysian shop
receipts (ringgit, a shop's name and the day on each) and CORD's Indonesian cafe and shop receipts (rupiah, the menu lines, no shop and no day), so
the folder is one person's only as a pile of what a person who travels between the two might keep. The questions ask for the receipt of a shop on a day, what was paid, every receipt of a shop, how many, which shop gave the biggest bill in a
month, the receipt with an item and its price, and of shops and items that are not there (questions.py). The receipts are SROIE's scans (company, date
and total of each) and CORD's photographs (the menu lines and totals); every answer is from their labels. The scenario is the names of a `Scenario`,
as functions (DESIGN 5), over the shared builder (realdump).

    recipe = i7_receipts.recipe(scale="tiny", seed=1)       # the sources, the scale's days, this machine's UTC offsets
    dump, questions = i7_receipts.build(recipe, folder, 3)  # the receipts copied in as `scan_0001.jpg` onwards, and the questions

Scales: `tiny`, about 30 receipts (for the tests and a smoke run); `standard`, every receipt of both sources whose labels are certain but the ones held
out (about 1,400 receipts, 0.7 GB; pile.py). The sources give no scan time: the receipts are scanned in sittings at seeded times of 2026, in a seeded
order that has nothing to do with the shop, the day or the source, a few seconds apart, and named in the order of their times. The same recipe gives the
same folder at every seed: the seed draws other questions from it. The SROIE scans are copied as they are; the CORD photographs, PNGs, are saved as
JPEGs, so that the folder has one kind of file."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mmsb.core.contract import Dump
from mmsb.core.question import Question
from mmsb.core.scenario import Access, ProbeOutcome, dump_key
from mmsb.core.taxonomy import Shelf, with_absent
from mmsb.scenarios import realdump
from mmsb.scenarios.i7_receipts import names
from mmsb.scenarios.i7_receipts.pile import Pile, choose
from mmsb.scenarios.i7_receipts.questions import CODE, SOURCE_NAME, questions_of
from mmsb.scenarios.i7_receipts.source import CORD, SROIE, as_jpeg, images, jpeg_plain, read_bills, read_slips
from mmsb.scenarios.realdump import File, report  # noqa: F401  (`report`: the Scenario's)

name = "i7_receipts"
code = CODE
generator = 1  # bumped when the scenario makes something other than before
shelf = Shelf("Receipts and bills", "image", "work", "a folder of scanned receipts and bills that merges Malaysian shop receipts (ringgit) and Indonesian cafe "
                                                     "receipts (rupiah), renamed scan_0001.jpg onwards")
filing = with_absent({"total": ("what", "text"), "bill": ("find", "text"), "shop": ("find_all", "text"), "howmany": ("how_many", "text"),
                      "biggest": ("who", "text"), "day": ("when", "text"), "did": ("did", "text"), "price": ("what", "text"),
                      "item": ("find", "text"), "items": ("find_all", "text"), "paid": ("what", "text")})

SCALES = {"tiny": ("2026-10-05", "2026-10-06"), "standard": ("2026-09-01", "2026-10-31")}  # the days the receipts are scanned over
SITTINGS = {"tiny": 2, "standard": 9}  # how many sittings of scanning
RECIPE_KEYS = {"source", "cord", "scale", "seed", "days", "utc_offsets"}
WHAT = "my folder of scanned receipts and bills"
LAYOUT = ("the folder holds my scans, scan_0001.jpg onwards, in the order I scanned them (JPEG); a file's modified time is when I scanned it, not "
          "the date on the receipt.")
COMMAND_WORDS = ("sroie", "icdar", "cord-v2", "cord_v2", "naver")  # the sources' names
DAY = (time(9), time(20))  # the hours a sitting may start in
STEP_S = (4, 14)  # the seconds between two scans of a sitting


def recipe(*, scale: str = "tiny", seed: int, source: Path | str = SROIE, cord: Path | str = CORD) -> dict:
    """What the dump is made from: the two sources (`source` is SROIE), the `scale` (`tiny` or `standard`), the `seed` the questions are drawn with,
    the days the receipts are scanned over and this machine's UTC offsets at its first and last day."""
    if scale not in SCALES:
        raise ValueError(f"{name}'s scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = realdump.span_days(*(date.fromisoformat(d) for d in SCALES[scale]))
    return {"source": str(source), "cord": str(cord), "scale": scale, "seed": seed, "days": days, "utc_offsets": realdump.utc_offsets(days)}


def keys_of(pile: Pile) -> list[str]:
    return sorted([b.key for b in pile.bills] + [s.key for s in pile.slips])


def layout(keys: Sequence[str], scale: str, days: Sequence[str]) -> dict[str, tuple[float, str]]:
    """The time (epoch seconds) and the name of each receipt of the folder, by its key: the receipts dealt in a seeded order into sittings at seeded
    times of the days, a few seconds apart, named in the order of their times."""
    rng = random.Random(f"i7:layout:{scale}")
    order = list(keys)
    rng.shuffle(order)
    sittings = SITTINGS[scale]
    first = date.fromisoformat(days[0])
    window = (DAY[1].hour - DAY[0].hour) * 3600
    starts = sorted((rng.randrange(len(days)), rng.randrange(window)) for _ in range(sittings))
    out: dict[str, float] = {}
    free = realdump.local_t(datetime.combine(first, time.min))
    for n, (day, within) in enumerate(starts):
        t = max(float(round(realdump.local_t(datetime.combine(first + timedelta(days=day), DAY[0]) + timedelta(seconds=within)))), free)
        for key in order[n * len(order) // sittings:(n + 1) * len(order) // sittings]:
            t += rng.randint(*STEP_S)
            out[key] = t
        free = t + 3600
    return {key: (t, f"scan_{n:04d}.jpg") for n, (key, t) in enumerate(sorted(out.items(), key=lambda kv: kv[1]), 1)}


def plan_of(recipe: Mapping[str, object]) -> tuple[tuple[File, ...], tuple[Question, ...]]:
    """The files and questions a recipe makes; refused when its keys are not `recipe`'s, or it was made in another time zone."""
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(f"an {name} recipe has the keys {', '.join(sorted(RECIPE_KEYS))}, not {', '.join(sorted(recipe))}")
    realdump.check_offsets(recipe)
    scale, seed = str(recipe["scale"]), int(recipe["seed"])
    bills, slips = read_bills(str(recipe["source"])), read_slips(str(recipe["cord"]))
    pile = choose(bills, slips, scale)
    placed = layout(keys_of(pile), scale, recipe["days"])
    spec = {**{b.key: {"set": "sroie", "parquet": b.parquet, "row": b.row} for b in pile.bills},
            **{s.key: {"set": "cord", "parquet": s.parquet, "row": s.row} for s in pile.slips}}
    files = tuple(File(placed[k][1], SOURCE_NAME, placed[k][0], "scan", "image", {"key": k, **spec[k]}) for k in sorted(placed, key=lambda k: placed[k][0]))
    paths = {k: p for k, (_, p) in placed.items()}
    last = datetime.fromtimestamp(files[-1].t).date()
    if last.isoformat() > recipe["days"][-1]:
        raise ValueError(f"a receipt is scanned on {last}, after the recipe's last day {recipe['days'][-1]}")
    now = realdump.local_t(datetime.combine(last + timedelta(days=1), time(9)))
    text = " " + " ".join(" ".join(names.words_of(" ".join((b.company, *b.words)))) for b in bills) + " "
    return files, questions_of(pile, paths, now, {k: t for k, (t, _) in placed.items()}, scale, seed, text)


def hashes(recipe: Mapping[str, object]) -> tuple[str, str]:
    files, questions = plan_of(recipe)
    return realdump.hashes(questions, files)


def run_fields(recipe: Mapping[str, object]) -> dict:
    return {}


def build(recipe: Mapping[str, object], out: Path, workers: int = 3) -> tuple[Dump, tuple[Question, ...]]:
    """Writes the folder into `out` (new or empty): each scan with its time, as a JPEG with no metadata; `workers` at once. Returns the dump and
    the questions."""
    files, questions = plan_of(recipe)
    roots = {"sroie": Path(str(recipe["source"])), "cord": Path(str(recipe["cord"]))}
    wanted: dict[tuple[str, str], dict[int, str]] = {}
    for f in files:
        wanted.setdefault((str(f.spec["set"]), str(f.spec["parquet"])), {})[int(f.spec["row"])] = str(f.spec["key"])
    jpeg: dict[str, bytes] = {}
    with ThreadPoolExecutor(workers) as pool:
        for (kind, parquet), rows in wanted.items():
            raw = images(roots[kind], parquet, set(rows))  # one file of images at a time: the CORD files are large
            if kind == "sroie":
                jpeg.update({rows[n]: jpeg_plain(b) for n, b in raw.items()})
            else:
                jpeg.update(zip((rows[n] for n in raw), pool.map(as_jpeg, raw.values()), strict=True))

    def write(f: File, path: Path) -> None:
        path.write_bytes(jpeg[str(f.spec["key"])])

    entries = realdump.write(files, out, workers, write)
    return realdump.dump_of(dump_key(name, recipe, generator), out, name, entries), questions


def probe(recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    """The leak probe: no scan's name or JPEG header names a shop, an item or a source of the folder."""
    labels = {b.company for b in read_bills(str(recipe["source"]))} | {i.name for s in read_slips(str(recipe["cord"])) if s.items for i in s.items}
    return realdump.probe(dump, out, [*sorted(l for l in labels if any(c.isalpha() for c in l)), *COMMAND_WORDS])  # (a label of figures only is no name)


def prompt(question: Question, access: Access) -> str:
    return realdump.prompt(question, access, WHAT, LAYOUT)


def view(question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None:
    """The folder itself: nothing is cut (every question is asked after its last scan)."""
    return dump.root


def deny_words(others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]:
    return realdump.deny_words(name, COMMAND_WORDS, (realdump.source_word(),))


def after_exec(question: Question, out: dict, dump: Dump, access: Access) -> dict:
    return {}


def release(temp: Path, key: str) -> None:
    """Nothing to remove: every slot reads the dump."""
