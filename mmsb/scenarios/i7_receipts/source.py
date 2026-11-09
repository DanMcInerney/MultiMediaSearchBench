"""The sources of I7: two sets of receipts on disk, read from their Hugging Face parquet files (`pyarrow`).

SROIE (`sroie/data/{train,test}-*.parquet`, ICDAR 2019): scanned Malaysian shop receipts, each with a `key`, the text lines OCR read (`words`) and
the entities the competition's annotators gave: `company`, `date` (as printed, in any of a dozen forms), `address` and `total` (the amount due, with
the currency as printed or not). Malaysian receipts print the day before the month (`24/12/2017`) but three of the set's shops print the month first
(`12/28/2017`), so a numeric date is read only for a shop that shows which it is: some receipt of it with a first number over 12 (day first) or
with a second one, and not both (`Bill.date`).

CORD (`cord-v2/data/*.parquet`): photographs of Indonesian receipts, PNG, with the menu lines as annotated (`ground_truth`, JSON: each line's name,
count and price, and the totals). The shop is blurred out and no date is given. A price is read only as a whole number of rupiah, printed with the
thousands marked (`60.000`, `13,500`) or not (`Slip.items`).

Neither image has metadata that names anything: SROIE's scans are JPEGs with no segments but their tables (and one Adobe marker), and CORD's PNGs
are decoded and saved again as JPEGs, as a scanning app would."""

from __future__ import annotations

import functools
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from mmsb.scenarios.realdump import sources_root

SROIE = sources_root() / "sroie"
CORD = sources_root() / "cord-v2"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
FIRST_YEAR, LAST_YEAR = 2015, 2019  # the years a receipt of the set is dated in
JPEG_QUALITY = 88  # of a CORD photograph saved again


@dataclass(frozen=True)
class Bill:
    key: str  # the source's, `X51005200931`
    parquet: str  # the file it is in
    row: int  # and its row there
    company: str
    date: str | None  # ISO, when the printed date reads one way only (see the docstring above); else None
    total: float | None  # the amount due, in ringgit: None when the label is no plain positive amount
    words: tuple[str, ...]  # the lines OCR read


@dataclass(frozen=True)
class Item:
    name: str  # as printed
    count: int  # 1 where the line has no count
    price: int  # the line's, in rupiah


@dataclass(frozen=True)
class Slip:
    key: str  # `train2-012`: the file's split and part, and the row
    parquet: str
    row: int
    items: tuple[Item, ...] | None  # every menu line, an extra of one (`sub`) too; None when a line is not read whole (no name, a price that is no whole rupiah)
    total: int | None  # in rupiah
    names: tuple[str, ...] = ()  # the name of every menu line (extras and voided lines too), whether the line is read whole or not: the labels a name must not be near


def amount(text: str) -> float | None:
    """A printed amount of ringgit as a number: `RM 8.60`, `$9.00` and `193.00`; None for anything else (no digits, a minus, a comma that is no
    thousands mark)."""
    found = re.fullmatch(r"(?:RM|\$)?\s*(\d{1,3}(?:,\d{3})*|\d+)\.(\d{1,2})", text.strip(), re.IGNORECASE)
    return float(found[1].replace(",", "") + "." + found[2]) if found else None


def rupiah(text: object) -> int | None:
    """A printed amount of rupiah as a whole number: `60.000`, `13,500` and `20000`; None for anything else."""
    if not isinstance(text, str):
        return None
    found = re.fullmatch(r"(?:Rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+)", text.strip())
    return int(re.sub(r"[.,]", "", found[1])) if found else None


def read_date(text: str, day_first: bool | None) -> str | None:
    """The ISO date a printed one reads as, or None. A month named (`12 MAR 2018`, `27/MAR/2018`) or a year first (`2018/03/27`) is one date; a
    number of three parts with a year of four digits is read day first when `day_first`, month first when not, and not at all when None; a year
    of two digits is not read."""
    text = text.strip().upper()
    out: tuple[int, int, int] | None = None
    if found := re.fullmatch(r"(\d{1,2})[ /-]([A-Z]{3})[ /-](\d{4})", text):
        out = (int(found[3]), MONTHS.index(found[2]) + 1, int(found[1])) if found[2] in MONTHS else None
    elif found := re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text):
        out = (int(found[1]), int(found[2]), int(found[3]))
    elif (found := re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)) and day_first is not None:
        a, b = int(found[1]), int(found[2])
        out = (int(found[3]), b, a) if day_first else (int(found[3]), a, b)
    if out is None or not FIRST_YEAR <= out[0] <= LAST_YEAR:
        return None
    try:
        return date(*out).isoformat()
    except ValueError:
        return None


def numeric_order(text: str) -> bool | None:
    """For a printed date of three numbers with a year of four digits: True when its first number is over 12 (the day), False when its second
    is (the day), None when neither tells; None too for any other date."""
    found = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text.strip())
    if not found:
        return None
    a, b = int(found[1]), int(found[2])
    return True if a > 12 >= b else False if b > 12 >= a else None


@functools.lru_cache(maxsize=2)
def read_bills(source: str) -> tuple[Bill, ...]:
    """The receipts of SROIE in `source`, by key. A shop whose receipts show a first number over 12 and a second over 12 both is refused: it does
    not say which comes first."""
    rows = []
    for path in sorted((Path(source) / "data").glob("*.parquet")):
        for n, r in enumerate(pq.read_table(path, columns=["key", "entities", "words"]).to_pylist()):
            rows.append((path.name, n, r))
    order: dict[str, set[bool]] = defaultdict(set)
    for _, _, r in rows:
        if (day := numeric_order(r["entities"]["date"])) is not None:
            order[r["entities"]["company"]].add(day)
    bills = []
    for name, n, r in rows:
        e = r["entities"]
        shown = order.get(e["company"], set())  # a shop that shows both orders shows neither
        total = amount(e["total"]) if e["total"] else None
        bills.append(Bill(r["key"], name, n, e["company"], read_date(e["date"], next(iter(shown)) if len(shown) == 1 else None),
                          total if total and total > 0 else None, tuple(r["words"])))
    if len({b.key for b in bills}) != len(bills):
        raise ValueError("two receipts have one key")
    return tuple(sorted(bills, key=lambda b: b.key))


def read_item(line: object) -> list[Item] | None:
    """The items of one menu line (the line and its extras), or None when it is not read whole."""
    if not isinstance(line, dict) or not isinstance(line.get("nm"), str) or not line["nm"].strip():
        return None
    count, price = line.get("cnt"), rupiah(line.get("price"))
    number = re.match(r"\s*(\d+)\s*(?:x|X)?\s*$", count) if isinstance(count, str) else None
    if price is None or (count is not None and not number):
        return None
    out = [Item(line["nm"].strip(), int(number[1]) if number else 1, price)]
    subs = line.get("sub")
    for sub in ([] if subs is None else subs if isinstance(subs, list) else [subs]):
        if (more := read_item(sub)) is None:
            return None
        out += more
    return out


def menu_names(node: object) -> list[str]:
    """The name (`nm`) of every line of a menu in the annotation, a line's extras (`sub`) and a voided line (`void_menu`) among them."""
    if isinstance(node, list):
        return [n for x in node for n in menu_names(x)]
    if isinstance(node, dict):
        return ([node["nm"]] if isinstance(node.get("nm"), str) else []) + menu_names(node.get("sub"))
    return []


@functools.lru_cache(maxsize=2)
def read_slips(source: str) -> tuple[Slip, ...]:
    """The receipts of CORD in `source`, by key."""
    slips = []
    for path in sorted((Path(source) / "data").glob("*.parquet")):
        split, part, parts = re.match(r"(\w+?)-(\d+)-of-(\d+)", path.name).groups()
        stem = split + (str(int(part)) if int(parts) > 1 else "")  # `train2`, `test`
        for n, r in enumerate(pq.read_table(path, columns=["ground_truth"]).to_pylist()):
            parse = json.loads(r["ground_truth"])["gt_parse"]
            menu = parse.get("menu")
            lines = [read_item(m) for m in (menu if isinstance(menu, list) else [menu] if isinstance(menu, dict) else [])]
            items = None if not lines or any(x is None for x in lines) or "void_menu" in parse else tuple(i for x in lines for i in x)
            total = parse.get("total")
            slips.append(Slip(f"{stem}-{n:03d}", path.name, n, items, rupiah(total.get("total_price")) if isinstance(total, dict) else None,
                              tuple(menu_names([menu, parse.get("void_menu")]))))
    return tuple(sorted(slips, key=lambda s: s.key))


def images(source: Path | str, parquet: str, rows: set[int]) -> dict[int, bytes]:
    """The image bytes of `rows` of one parquet file of `source`, by row, read in order so that a file of many large images is not held whole."""
    out: dict[int, bytes] = {}
    at = 0
    for batch in pq.ParquetFile(Path(source) / "data" / parquet).iter_batches(batch_size=8, columns=["image"]):
        for n, image in enumerate(batch.column("image").to_pylist(), at):
            if n in rows:
                out[n] = image["bytes"]
        at += batch.num_rows
        if len(out) == len(rows):
            break
    if len(out) != len(rows):
        raise ValueError(f"{parquet} has no row {sorted(rows - set(out))[0]}")
    return out


def as_jpeg(png: bytes) -> bytes:
    """A photograph saved as a JPEG of the folder's quality, with no metadata."""
    with Image.open(io.BytesIO(png)) as im:
        out = io.BytesIO()
        im.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def jpeg_plain(raw: bytes) -> bytes:
    """A scan as it goes into the dump, a JPEG with no marker segments other than its tables and the Adobe one (SROIE's all have so); refused
    when it has any other, an EXIF block, a comment or a thumbnail: metadata to carry or to strip."""
    if raw[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    at = 2
    while at + 4 <= len(raw) and raw[at] == 0xFF and raw[at + 1] != 0xDA:
        if raw[at + 1] not in (0xDB, 0xC0, 0xC2, 0xC4, 0xEE):
            raise ValueError(f"the JPEG has a segment of marker {raw[at + 1]:#x}")
        at += 2 + int.from_bytes(raw[at + 2:at + 4], "big")
    return raw
