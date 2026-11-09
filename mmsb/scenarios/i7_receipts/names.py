"""The names of I7: how a shop and an item are told apart, and said as a person says them.

The labels spell one shop several ways (`SDN BHD` and `SDN. BHD.`, a misprint, a branch in brackets) and one item several ways (`Thai Iced Tea`,
`THAI ICED TEA (L)`), so a name is asked about only where it is certain what is meant: its key (the words of the label, with the legal ending of a
company left off) has no other key of the source near it, by holding its words or all but a letter or two (`near`). A shop is then every receipt
whose label has that key; a person's name for it (`spoken`) is the label in the case a person writes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher

LEGAL = frozenset("sdn bhd berhad sb co ltd pte plt".split())  # the words that end a company's name: SDN. BHD., S/B
LEGAL_WORD = re.compile(r"\b(?:sdn|bhd|berhad|s/b|co|ltd|pte|plt)\b\.?", re.IGNORECASE)
NEAR_RATIO = 0.8  # how alike two keys must be (difflib's ratio of their words run together) to be taken for one name misprinted
SUBSTRING = 5  # the letters a name must have for another that holds it, spaces left out, to be taken for it (`Gohiong` in `Ngohiong`)
SHARED = 0.5  # the share of a key's letter triples another must have to be looked at for a misprint


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold().replace("'", ""))


def shop_key(company: str) -> tuple[str, ...]:
    """The words of a company's label but its legal ending: `("mr", "d", "i", "y", "m")` for `MR. D.I.Y. (M) SDN BHD`."""
    found = [w for w in words_of(company.replace("/", "")) if w not in LEGAL]
    if not found:
        raise ValueError(f"the label {company!r} is nothing but a legal ending")
    return tuple(found)


def item_key(name: str) -> tuple[str, ...]:
    """The words of an item's name: `("thai", "iced", "tea", "l")` for `THAI ICED TEA (L).`"""
    return tuple(words_of(name))


def near(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether two different keys may be one name: one holds all the words of the other (a branch, an extra word), they begin with the same words
    (two branches: `MR. D.I.Y. (M)` and `MR. D.I.Y. (KUCHAI)`; two words, or half of the shorter name), they are the same with the spaces left out or
    one holds the other so (`Go Hiong`, `Ngohiong`), or the words run together are NEAR_RATIO alike (a letter or two misprinted)."""
    if a == b:
        return False
    if set(a) <= set(b) or set(b) <= set(a):
        return True
    shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    if shared >= 2 or (shared and 2 * shared >= min(len(a), len(b))):
        return True
    ja, jb = "".join(a), "".join(b)  # spaces and case do not tell a name from another: `Go Hiong` and `Ngohiong`
    if ja == jb or (min(len(ja), len(jb)) >= SUBSTRING and (ja in jb or jb in ja)):
        return True
    return abs(len(ja) - len(jb)) <= (1 - NEAR_RATIO) * max(len(ja), len(jb)) * 2 and SequenceMatcher(None, ja, jb).ratio() >= NEAR_RATIO


def triples(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def clean(keys: Iterable[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """The keys of `keys` that no other key of it is near."""
    every = sorted({k for k in keys if k})
    grams = [triples("".join(k)) for k in every]
    by_word: dict[str, list[int]] = defaultdict(list)
    by_gram: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(every):
        for w in set(k):
            by_word[w].append(i)
        for g in grams[i]:
            by_gram[g].append(i)
    out = set()
    for i, k in enumerate(every):
        rivals = {j for w in set(k) for j in by_word[w]}
        shared = Counter(j for g in grams[i] for j in by_gram[g])
        rivals |= {j for j, c in shared.items() if c >= SHARED * len(grams[i])}
        if not any(near(k, every[j]) for j in rivals if j != i):
            out.add(k)
    return out


def proper(word: str) -> str:
    """A word of capitals in the case a person writes it: `BAKERIES` as `Bakeries`; a short one (`KL`, `MR.`), one with dots in it (`D.I.Y.`) and
    any with lower case in it as they are."""
    letters = re.sub(r"[^A-Za-z]", "", word)
    if len(letters) < 3 or not letters.isupper() or "." in word.strip(".,"):
        return word
    at = next(i for i, c in enumerate(word) if c.isalpha())
    return word[:at + 1] + word[at + 1:].lower()


def spoken(label: str) -> str:
    """A shop's label as a person writes the name: the legal ending left off and the capitals of its words (`GARDENIA BAKERIES (KL) SDN BHD`) as
    `Gardenia Bakeries (KL)`."""
    return " ".join(proper(w) for w in LEGAL_WORD.sub(" ", label.replace("’", "'")).split()).strip(" ,.-")


def item_said(name: str) -> str:
    """An item's name as a person writes it: capitals as `proper` gives, the stray marks at its ends left off."""
    return " ".join(proper(w) for w in name.split()).strip(" ,.-*")
