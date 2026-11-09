"""The questions of I7, every answer from the labels (source.py), each asked the morning after the last scan. A person asks of a folder of scanned
receipts by the shop, the day, the item and the sum:

| template  | question                                                                     | rule                         |
| `total`   | How much did I pay at Sanyu Stationery Shop on 16 December 2017?              | `amount`, to 5 sen (ringgit) |
| `bill`    | Find the receipt from Moonlight Cake House on 3 March 2018.                   | `set` of one file name       |
| `shop`    | Show me every receipt from Bemed (SP). (2 to 12)                              | `set` of file names          |
| `howmany` | How many receipts do I have from Aeon (M)? (2 to 40)                          | `count`                      |
| `biggest` | Which shop gave me my biggest bill in March 2018?                            | `text`, the shop's name      |
| `day`     | On which day did I shop at Yam Fresh? (a shop with one receipt)               | `set` of one date            |
| `did`     | Did I ever shop at Pinghwai Trading? If so, on which days? (2 to 4 receipts)  | `set` of dates               |
| `price`   | How much did I pay for the Dadar Pisang, in rupiah?                           | `amount`, the rupiah         |
| `item`    | Find the receipt with the Gulai Tunjang on it.                                | `set` of one file name       |
| `items`   | Find every receipt that has Milk Pastry Roll on it. (2 to 8)                  | `set` of file names          |
| `paid`    | What was the total on the receipt with the Bubur Ungu, in rupiah?             | `amount`, the rupiah         |
| `absent_*`| `did` of a shop with no receipt; `bill` of a day the shop has none; `item` of an item no receipt has | `absent` |

The shop of a question is a shop of SROIE whose label has no other near it (names.py), asked as a person says it (`spoken`), and a shop is every
receipt whose label has its words: so a count and a list are of the pile, certain. A date is a printed one that reads one way. A total is the amount
due that the label gives and a line of the receipt holds; it is asked to the 5 sen of Malaysian cash rounding, which is the label's or the receipt's
rounded line. An item of CORD is asked only where it is on one receipt of the pile (`price`, `item`, `paid`) or on a few (`items`), and no other name
of the source is near it. The absent questions are held out: a shop with a real receipt on other days, a shop with its one receipt, an item with its
one receipt, were taken out of the pile (pile.py), and the shops of `did` that never were are Malaysian chains no label or line of SROIE names. Which
questions are asked is drawn with the seed."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.i7_receipts import names
from mmsb.scenarios.i7_receipts.pile import Pile, askable, item_lines, sole_items
from mmsb.scenarios.i7_receipts.source import Bill

CODE = "I7"
SOURCE_NAME = "scans"  # the dump's source
QUOTAS = {"tiny": {"total": 1, "shop": 1, "howmany": 1, "price": 1, "item": 1, "absent_did": 1, "absent_bill": 1, "absent_item": 1},
          "standard": {"total": 2, "bill": 2, "shop": 2, "howmany": 2, "biggest": 1, "day": 2, "did": 1, "price": 2, "item": 1, "items": 1, "paid": 1,
                       "absent_did": 2, "absent_bill": 1, "absent_item": 2}}
NEVER = ("Tesco", "Mydin", "Econsave", "Jaya Grocer", "KK Mart", "Padini", "Big Pharmacy", "Village Grocer")  # Malaysian chains no label or line of SROIE names
SEVERAL = (2, 12)  # the receipts of a shop a `shop` question lists
COUNTED = (2, 40)  # the receipts of a shop a `howmany` question counts
FEW = (2, 4)  # the receipts of a shop a `did` question asks the days of
ITEMS = (2, 8)  # the receipts an `items` question lists
MONTH_BILLS = 8  # the receipts a month needs for `biggest` to be asked of it
LEAD = 1.2  # how much more than the next the biggest bill of the month is
SEN = 0.05  # the ringgit an amount may be off by: the cash rounding to 5 sen
RM = "<the amount in ringgit as a plain number, such as 12.50, with no RM before it>"
RP = "<the amount in rupiah as a plain number, such as 25000, with no Rp before it>"
FILE = "<the name of the file, as in scan_0042.jpg>"
FILES = "<the names of the files, as in scan_0042.jpg, separated by commas>"
DAYS = "<the days, as YYYY-MM-DD, separated by commas>"
DAY = "<the day, as YYYY-MM-DD>"
NUMBER = "<a number>"
SHORT = "<the name of the shop>"
MONTHS = realdump.MONTHS


def day_text(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def month_text(iso: str) -> str:
    return f"{MONTHS[int(iso[5:7]) - 1]} {iso[:4]}"


def pick(candidates: Sequence, n: int, rng: random.Random) -> list:
    """`n` of `candidates`, in a seeded order; refused when there are fewer."""
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} can be asked about, and {n} are wanted")
    return rng.sample(list(candidates), n)


def questions_of(pile: Pile, paths: Mapping[str, str], now: float, times: Mapping[str, float], scale: str, seed: int,
                 words: Sequence[str]) -> tuple[Question, ...]:
    """The questions of `pile` at `scale`, drawn with `seed`. `paths` and `times` are the dump path and epoch seconds of each receipt by its key
    (a bill's or a slip's); `words` is all the text of the source's bills, folded, for the shops that never were; `now` is epoch seconds."""
    rng = random.Random(f"i7:{seed}:{scale}")
    quota = QUOTAS[scale]
    since, until = min(times.values()), max(times.values()) + 1
    out: list[Question] = []

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[str], value: str | None) -> None:
        absent = grade.rule == "absent"
        out.append(realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text, kind="absent" if absent else template,
                                     template="absent_" + template if absent else template, modality="image", shape=shape, grade=grade,
                                     evidence=[Span(paths[k], 0.0, 0.0) for k in found], sources=(SOURCE_NAME,), since=since, until=until, value=value))

    def file(key: str) -> str:
        return paths[key].rsplit("/", 1)[-1]

    # -- the shops
    by_shop: dict[tuple[str, ...], list[Bill]] = defaultdict(list)
    for b in pile.bills:
        by_shop[names.shop_key(b.company)].append(b)
    shops = {k: sorted(v, key=lambda b: b.date) for k, v in by_shop.items() if k in pile.shops}
    day_unique = {k: v for k, v in shops.items() if len({b.date for b in v}) == len(v)}
    taken: set[tuple[str, ...]] = set()

    def draw(candidates: Mapping[tuple[str, ...], list[Bill]], n: int) -> list[tuple[str, ...]]:
        """`n` shops of `candidates` no question has asked of, a shop at most once in all."""
        chosen = pick([k for k in sorted(candidates) if k not in taken], n, rng)
        taken.update(chosen)
        return chosen

    for key in draw(day_unique, quota.get("total", 0)):
        b = rng.choice(shops[key])
        ask(f"total-{'-'.join(key)}", f"How much did I pay at {names.spoken(b.company)} on {day_text(b.date)}?", "total", RM,
            realdump.grade("amount", truth=f"{b.total:.2f}", tol_s=SEN), [b.key], f"{b.total:.2f}")
    for key in draw(day_unique, quota.get("bill", 0)):
        b = rng.choice(shops[key])
        ask(f"bill-{'-'.join(key)}", f"Find the receipt from {names.spoken(b.company)} on {day_text(b.date)}.", "bill", FILE,
            realdump.grade("set", items=[file(b.key)]), [b.key], file(b.key))
    several = {k: v for k, v in shops.items() if SEVERAL[0] <= len(v) <= SEVERAL[1]}
    for key in draw(several, quota.get("shop", 0)):
        found = [file(b.key) for b in shops[key]]
        ask(f"shop-{'-'.join(key)}", f"Show me every receipt from {names.spoken(shops[key][0].company)}.", "shop", FILES, realdump.grade("set", items=found),
            [b.key for b in shops[key]], ", ".join(found))
    counted = {k: v for k, v in shops.items() if COUNTED[0] <= len(v) <= COUNTED[1]}
    for key in draw(counted, quota.get("howmany", 0)):
        ask(f"howmany-{'-'.join(key)}", f"How many receipts do I have from {names.spoken(shops[key][0].company)}?", "howmany", NUMBER,
            realdump.grade("count", truth=str(len(shops[key]))), [b.key for b in shops[key]], str(len(shops[key])))
    for key in draw({k: v for k, v in day_unique.items() if len(v) == 1}, quota.get("day", 0)):
        (b,) = shops[key]
        ask(f"day-{'-'.join(key)}", f"On which day did I shop at {names.spoken(b.company)}?", "day", DAY, realdump.grade("set", items=[b.date]), [b.key], b.date)
    for key in draw({k: v for k, v in day_unique.items() if FEW[0] <= len(v) <= FEW[1]}, quota.get("did", 0)):
        days = sorted({b.date for b in shops[key]})
        ask(f"did-{'-'.join(key)}", f"Did I ever shop at {names.spoken(shops[key][0].company)}? If so, on which days?", "did", DAYS,
            realdump.grade("set", items=days), [b.key for b in shops[key]], ", ".join(days))
    # -- the biggest bill of a month
    months: dict[str, list[Bill]] = defaultdict(list)
    for b in pile.bills:
        months[b.date[:7]].append(b)
    winners = []
    for month, bills in sorted(months.items()):
        top = sorted(bills, key=lambda b: -b.total)
        if len(bills) >= MONTH_BILLS and names.shop_key(top[0].company) in pile.shops and top[0].total >= LEAD * top[1].total:
            winners.append((month, top))
    for month, top in pick(winners, quota.get("biggest", 0), rng):
        key = names.shop_key(top[0].company)
        alone = [w for w in key if len(w) >= 3 and sum(w in other for other in pile.shop_keys) == 1]  # the words of its name that no other shop's has
        mine = f" {' '.join(names.words_of(names.spoken(top[0].company)))} "
        others = {' '.join(names.words_of(names.spoken(b.company))) for b in top[1:] if names.shop_key(b.company) != key}
        others = {o for o in others if f" {o} " not in mine}  # a name that is part of the winner's would refuse the right answer
        ask(f"biggest-{month}", f"Which shop gave me my biggest bill in {month_text(month)}?", "biggest", SHORT,
            realdump.grade("text", truth="|".join([" ".join(key), *alone]), wrong=sorted(others)), [top[0].key], names.spoken(top[0].company))
    # -- the items
    present, lines = item_lines(pile.slips)
    askables = [k for k in sorted(present) if k in pile.items and askable(k)]
    sole = sole_items(pile.slips, pile.items)
    sole_taken: set[tuple[str, ...]] = set()

    def sole_item(n: int) -> list[tuple[str, ...]]:
        chosen = pick([k for k in sole if k not in sole_taken], n, rng)
        sole_taken.update(chosen)
        return chosen

    for key in sole_item(quota.get("price", 0)):
        (i,) = lines[key]
        ask(f"price-{'-'.join(key)}", f"How much did I pay for the {names.item_said(i.name)}, in rupiah?", "price", RP,
            realdump.grade("amount", truth=str(i.price), tol_s=0.0), [present[key][0].key], str(i.price))
    for key in sole_item(quota.get("item", 0)):
        (i,) = lines[key]
        ask(f"item-{'-'.join(key)}", f"Find the receipt with the {names.item_said(i.name)} on it.", "item", FILE,
            realdump.grade("set", items=[file(present[key][0].key)]), [present[key][0].key], file(present[key][0].key))
    for key in pick([k for k in askables if ITEMS[0] <= len(present[k]) <= ITEMS[1]], quota.get("items", 0), rng):
        found = [file(s.key) for s in present[key]]
        ask(f"items-{'-'.join(key)}", f"Find every receipt that has {names.item_said(lines[key][0].name)} on it.", "items", FILES,
            realdump.grade("set", items=found), [s.key for s in present[key]], ", ".join(found))
    for key in sole_item(quota.get("paid", 0)):
        (i,) = lines[key]
        slip = present[key][0]
        ask(f"paid-{'-'.join(key)}", f"What was the total on the receipt with the {names.item_said(i.name)}, in rupiah?", "paid", RP,
            realdump.grade("amount", truth=str(slip.total), tol_s=0.0), [slip.key], str(slip.total))
    # -- what is not there
    never = [n for n in NEVER if f" {' '.join(names.words_of(n))} " not in words]
    gone = [(names.spoken(b.company), "shop") for b in pile.held_shops] + [(n, "never") for n in never]
    for name, kind in pick(gone, quota.get("absent_did", 0), rng):
        ask(f"did-{'-'.join(names.words_of(name))}", f"Did I ever shop at {name}? If so, on which days?", "did", DAYS, realdump.grade("absent"), [], None)
    for b in pick(pile.held_bills, quota.get("absent_bill", 0), rng):
        ask(f"bill-{'-'.join(names.shop_key(b.company))}-{b.date}", f"Find the receipt from {names.spoken(b.company)} on {day_text(b.date)}.", "bill", FILE,
            realdump.grade("absent"), [], None)
    for slip, key in pick(pile.held_slips, quota.get("absent_item", 0), rng):
        name = next(i.name for i in slip.items if names.item_key(i.name) == key)
        ask(f"item-{'-'.join(key)}", f"Find the receipt with the {names.item_said(name)} on it.", "item", FILE, realdump.grade("absent"), [], None)
    if len({q.qid for q in out}) != len(out):
        raise ValueError("two questions have one id")
    return realdump.split(out, seed)
