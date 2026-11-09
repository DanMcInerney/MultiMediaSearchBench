"""The pile of I7: which receipts of the two sources are in the folder, and which were taken out of it for the questions about what is not there.

A receipt is in the pile only where what the questions read off its label is certain:
- an SROIE receipt has a date that reads one way (source.py) and a total that is a plain amount printed on it (one of its lines holds it), and is the
  first of those with its shop, date and total: the set holds some receipts twice, scanned twice, which would make a shop's receipts too many;
- a CORD receipt has every menu line read whole and a total.

A receipt may be held out, as a real question's needle: a bill of a shop that has others, so that the shop is there and the day is not; the only
bill of a shop, so that it is not there at all; a receipt with an item that is on no other. Each is of a name that no other name of the source is near
(names.py), so that nothing left is easily taken for it. The held-out are drawn with a fixed seed and are the same at every question seed."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from mmsb.scenarios.i7_receipts import names
from mmsb.scenarios.i7_receipts.source import Bill, Item, Slip

HELD = {"standard": {"bills": 8, "shops": 6, "slips": 8}, "tiny": {"bills": 1, "shops": 1, "slips": 1}}  # how many of each are taken out
TINY = {"shops": 3, "singles": 3, "slips": 6}  # the tiny folder: shops of a few receipts to ask about, shops of one, and CORD receipts with an item of their own


@dataclass(frozen=True)
class Pile:
    bills: tuple[Bill, ...]  # in the folder
    slips: tuple[Slip, ...]
    held_bills: tuple[Bill, ...]  # a bill of a shop that has others in the folder, taken out
    held_shops: tuple[Bill, ...]  # the only bill of a shop, taken out
    held_slips: tuple[tuple[Slip, tuple[str, ...]], ...]  # a receipt taken out, with the key of an item that is on no other
    shops: frozenset[tuple[str, ...]]  # the keys of the shops no other shop of the source is near
    shop_keys: frozenset[tuple[str, ...]]  # the keys of every shop of the source
    items: frozenset[tuple[str, ...]]  # the keys of the items no other item of the source is near


def printed(bill: Bill) -> bool:
    """Whether a line of the receipt holds its total."""
    total = bill.total or 0.0
    forms = {f"{total:.2f}", f"{total:.2f}".rstrip("0").rstrip(".")}
    return any(form in line for line in bill.words for form in forms)


def usable_bills(bills: Sequence[Bill]) -> list[Bill]:
    seen, out = set(), []
    for b in bills:
        if b.date and b.total and printed(b) and (b.company, b.date, b.total) not in seen:
            seen.add((b.company, b.date, b.total))
            out.append(b)
    return out


def usable_slips(slips: Sequence[Slip]) -> list[Slip]:
    return [s for s in slips if s.items is not None and s.total is not None]


def item_lines(slips: Sequence[Slip]) -> tuple[dict[tuple[str, ...], list[Slip]], dict[tuple[str, ...], list[Item]]]:
    """For each item key, the slips it is on and its lines (an item twice on one slip is two lines on one slip)."""
    present: dict[tuple[str, ...], list[Slip]] = defaultdict(list)
    lines: dict[tuple[str, ...], list[Item]] = defaultdict(list)
    for s in slips:
        for i in s.items:
            key = names.item_key(i.name)
            lines[key].append(i)
            if s not in present[key]:
                present[key].append(s)
    return present, lines


def askable(key: tuple[str, ...]) -> bool:
    """Whether a person could ask for an item of this name: words of letters, one of four or more (no code, no number)."""
    return bool(key) and all(w.isalpha() and len(w) >= 2 for w in key) and max(len(w) for w in key) >= 4


def sole_items(slips: Sequence[Slip], keys: frozenset[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """The item keys of `slips` that are on one slip, on one line there and bought singly (so the price of the line is what was paid for it), that
    a person could ask for and no other name of the source is near (`keys`)."""
    present, lines = item_lines(slips)
    return sorted(k for k in present if k in keys and askable(k) and len(present[k]) == 1 and len(lines[k]) == 1 and lines[k][0].count == 1)


def choose(bills: Sequence[Bill], slips: Sequence[Slip], scale: str) -> Pile:
    """The pile of `scale`: all the usable receipts but those held out (`standard`), or a seeded few (`tiny`)."""
    every_bill, every_slip = usable_bills(bills), usable_slips(slips)
    shops = frozenset(names.clean({names.shop_key(b.company) for b in bills}))
    keys = frozenset(names.clean({names.item_key(n) for s in slips for n in s.names}))  # against every menu line of the source, read whole or not
    rng = random.Random(f"i7:{scale}")
    by_shop: dict[tuple[str, ...], list[Bill]] = defaultdict(list)
    for b in every_bill:
        by_shop[names.shop_key(b.company)].append(b)
    many = sorted(k for k, v in by_shop.items() if k in shops and len(v) >= 2 and len({b.date for b in v}) == len(v))
    single = sorted(k for k, v in by_shop.items() if k in shops and len(v) == 1)
    held = HELD[scale]
    held_bills = [rng.choice(by_shop[k]) for k in rng.sample(many, held["bills"])]
    held_shops = [by_shop[k][0] for k in rng.sample([k for k in single], held["shops"])]
    present, _ = item_lines(every_slip)
    sole = sole_items(every_slip, keys)
    held_slips: list[tuple[Slip, tuple[str, ...]]] = []
    for k in rng.sample(sole, len(sole)):
        if len(held_slips) < held["slips"] and present[k][0].key not in {s.key for s, _ in held_slips}:
            held_slips.append((present[k][0], k))
    gone = {b.key for b in (*held_bills, *held_shops)}
    gone_slips = {s.key for s, _ in held_slips}
    kept_bills = [b for b in every_bill if b.key not in gone]
    kept_slips = [s for s in every_slip if s.key not in gone_slips]
    if scale == "tiny":
        kept_bills, kept_slips = tiny(kept_bills, kept_slips, shops, keys, held_bills, rng)
    return Pile(tuple(kept_bills), tuple(kept_slips), tuple(held_bills), tuple(held_shops), tuple(held_slips), shops,
                frozenset(names.shop_key(b.company) for b in bills), keys)


def tiny(bills: Sequence[Bill], slips: Sequence[Slip], shops: frozenset, keys: frozenset, held: Sequence[Bill], rng: random.Random) -> tuple[list[Bill], list[Slip]]:
    """A few shops of the pile with all their receipts, the shops the held-out bills are of among them, and a few CORD receipts with an item of
    their own."""
    by_shop: dict[tuple[str, ...], list[Bill]] = defaultdict(list)
    for b in bills:
        by_shop[names.shop_key(b.company)].append(b)
    need = {names.shop_key(b.company) for b in held}
    few = sorted(k for k, v in by_shop.items() if k in shops and 2 <= len(v) <= 4 and k not in need)
    singles = sorted(k for k, v in by_shop.items() if k in shops and len(v) == 1)
    chosen = need | set(rng.sample(few, TINY["shops"])) | set(rng.sample(singles, TINY["singles"]))
    present, _ = item_lines(slips)
    sole = sorted({present[k][0].key for k in sole_items(slips, keys)})
    pick = set(rng.sample(sole, TINY["slips"]))
    return [b for b in bills if names.shop_key(b.company) in chosen], [s for s in slips if s.key in pick]
