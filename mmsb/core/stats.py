"""Statistics of the paired comparison and of a result's cost (moved from bench.py of the Multimem repo, fe0ece5). A row of
`pair_counts` is `(question, control result, tested result)`; `b` counts the pairs right with the tested configuration only,
`c` those right with the control only."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

MIN_DISCORDANT = 10  # pairs right in one configuration only (b + c) from which a difference gets an interval; below it only McNemar speaks


def mcnemar(b: int, c: int) -> float:
    """McNemar's exact two-sided p: b pairs right only with the tested configuration and c right only with the control, under a fair
    coin."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval (95%) of k out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def newcombe(a: int, b: int, c: int, d: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% hybrid score interval of Newcombe (1998, method 10) for the difference of two proportions measured on the same
    pairs: `a` pairs right in both configurations, `b` right with the tested one only, `c` right with the control only, `d` right in
    neither. The difference is (b - c) / n, and the correlation of the two narrows the Wilson intervals' sum (with the paper's
    correction toward 0)."""
    n = a + b + c + d
    if n == 0:
        raise ValueError("no pairs")
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = wilson(a + b, n, z), wilson(a + c, n, z)
    spread = (a + b) * (c + d) * (a + c) * (b + d)
    cross = a * d - b * c
    cross = cross - n / 2 if cross > n / 2 else 0.0 if cross >= 0 else cross
    phi = cross / math.sqrt(spread) if spread else 0.0
    low = (p1 - p2) - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    high = (p1 - p2) + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return low, high


def pair_counts(rows: list[tuple]) -> tuple[int, int, int, int]:
    """(a, b, c, d) of paired rows `(question, control result, tested result)`: right in both, right with the tested only, right
    with the control only, right in neither."""
    pairs = [(bool(tested["passed"]), bool(control["passed"])) for _, control, tested in rows]
    return (sum(1 for x, y in pairs if x and y), sum(1 for x, y in pairs if x and not y),
            sum(1 for x, y in pairs if y and not x), sum(1 for x, y in pairs if not x and not y))


def interval(a: int, b: int, c: int, d: int) -> list[float] | None:
    """The interval of the difference when there are MIN_DISCORDANT pairs or more that are right in one configuration only, else None: with
    fewer, an interval that excludes 0 is no sign (McNemar's p says as much)."""
    return list(newcombe(a, b, c, d)) if b + c >= MIN_DISCORDANT else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def token_total(r: dict) -> int | None:
    """What a result spent in tokens: input and output, and for Claude the cache it read and wrote too (its `input` leaves the
    cache out; Codex's already holds its cached tokens). None for a result that gives none."""
    t = r.get("tokens")
    if not t:
        return None
    cache = (t.get("cache_read") or 0) + (t.get("cache_write") or 0) if r.get("agent") == "claude" else 0
    return t["input"] + t["output"] + cache
