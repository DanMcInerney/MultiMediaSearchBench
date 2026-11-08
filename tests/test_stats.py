"""Ported from MultimediaSearch tests/test_bench.py at fe0ece5 (the statistics parts of test_the_difference_gets_newcombes_interval_...,
test_the_report_counts_as_it_should and test_claudes_tokens_count_the_cache_...), with `bench.` now `stats.`. The tests marked new
are not from there: the cells of the two recorded reports, which the old code computed, and laws every input obeys."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from mmsb.core import stats

RECORDED = Path(__file__).parent / "oracles" / "recorded"


def cells(report: dict):
    """Every cell of a recorded report that has a paired comparison and the correct counts of both sides, as
    `(where, (a, b, c, d), cell)`: a is the pairs right in both, which the counts of the sides give back."""
    found = []

    def walk(node, where):
        if isinstance(node, dict):
            sides = [node.get(x) for x in ("files", "mm")] if "files" in node else [node.get(x) for x in ("base", "other")]
            if {"n", "b", "c", "p", "ci"} <= node.keys() and all(isinstance(s, dict) and s.get("n") == node["n"] for s in sides):
                control, tested = sides
                a = tested["correct"] - node["b"]
                assert control["correct"] == a + node["c"], where  # the two sides and b, c agree on a
                found.append((where, (a, node["b"], node["c"], node["n"] - a - node["b"] - node["c"]), node))
            for key, child in node.items():
                walk(child, f"{where}/{key}")

    walk(report, "")
    return found


@pytest.mark.parametrize("name, how_many, with_interval", [("pilot-s1", 16, 0), ("egolife-s1", 18, 3)])
def test_every_recorded_cell_gets_the_p_and_the_interval_the_old_code_gave_it(name, how_many, with_interval):
    report = json.loads((RECORDED / name / "report.json").read_text(encoding="utf-8"))
    found = cells(report)
    assert len(found) == how_many and sum(c["ci"] is not None for _, _, c in found) == with_interval, "the walk found the cells"
    for where, counts, cell in found:
        a, b, c, d = counts
        assert stats.mcnemar(b, c) == pytest.approx(cell["p"], rel=1e-12, abs=0), where
        if cell["ci"] is None:
            assert stats.interval(a, b, c, d) is None and b + c < stats.MIN_DISCORDANT, where
        else:
            assert stats.interval(a, b, c, d) == pytest.approx(cell["ci"], rel=1e-12, abs=0), where


def test_the_pilots_four_against_nothing_is_p_an_eighth_and_one_against_nothing_is_p_one():
    assert stats.mcnemar(4, 0) == 0.125 and stats.mcnemar(1, 0) == 1.0


def test_the_difference_gets_newcombes_interval_only_from_ten_pairs_right_in_one_arm_only_and_mcnemar_speaks_for_the_rest():
    # the case that a percentile bootstrap got wrong: 4 right with mm only, 0 the other way, 16 pairs: [.06, .50] with p = .125
    assert stats.newcombe(6, 4, 0, 6)[0] > 0, "the interval would exclude 0 where McNemar does not"
    assert stats.interval(6, 4, 0, 6) is None and stats.interval(6, 5, 4, 5) is None, "9 discordant pairs"
    # from 10: the interval, beside the p
    assert stats.interval(5, 10, 0, 5) == pytest.approx([0.223, 0.675], abs=1e-3) and stats.mcnemar(10, 0) == pytest.approx(0.00195, abs=1e-4)
    low, high = stats.interval(10, 5, 5, 10)
    assert low < 0 < high and stats.mcnemar(5, 5) == 1.0
    # it agrees with McNemar where one speaks plainly: p of 0.2 or more has 0 inside, p of 0.01 or less has it outside
    for n in (12, 20, 30):
        for b in range(n + 1):
            for c in range(n + 1 - b):
                if b + c < stats.MIN_DISCORDANT:
                    assert stats.interval(n - b - c, b, c, 0) is None
                    continue
                for a in range(0, n - b - c + 1, max(1, (n - b - c) // 3)):
                    low, high = stats.newcombe(a, b, c, n - a - b - c)
                    p = stats.mcnemar(b, c)
                    assert not (p >= 0.2 and (low > 0 or high < 0)) and not (p <= 0.01 and low <= 0 <= high), (a, b, c, p, low, high)
    with pytest.raises(ValueError):
        stats.newcombe(0, 0, 0, 0)
    assert stats.wilson(0, 0) == (0.0, 1.0) and stats.wilson(5, 10)[0] < 0.5 < stats.wilson(5, 10)[1]


def test_mcnemar_is_the_statistics_ported_values():
    assert stats.mcnemar(6, 0) == 0.03125 and stats.mcnemar(5, 0) == 0.0625 and stats.mcnemar(0, 0) == 1.0
    assert stats.mcnemar(3, 3) == 1.0 and stats.mcnemar(10, 2) == pytest.approx(0.0386, abs=1e-4)
    assert stats.mcnemar(2, 6) == stats.mcnemar(6, 2)


def test_claudes_tokens_count_the_cache_it_read_and_wrote_as_codexs_input_already_does():
    claude = {"agent": "claude", "tokens": {"input": 100, "output": 50, "cache_read": 1000, "cache_write": 200}}
    codex = {"agent": "codex", "tokens": {"input": 1300, "output": 50, "cache_read": 1000, "cache_write": 0}}
    assert stats.token_total(claude) == 1350 == stats.token_total(codex) and stats.token_total({"agent": "claude", "tokens": None}) is None
    assert stats.token_total({"agent": "claude", "tokens": {"input": 5, "output": 1, "cache_read": None, "cache_write": None}}) == 6


# -- new ----------------------------------------------------------------------------------------------------------------

def test_mcnemar_is_the_exact_two_sided_p_of_a_fair_coin_by_counting_every_outcome():
    """A simple model run in lockstep: every one of the 2^n ways the n discordant pairs can fall, counted."""
    for n in range(13):
        ways = [x.bit_count() for x in range(2 ** n)]
        for b in range(n + 1):
            c = n - b
            extreme = sum(1 for k in ways if k <= min(b, c) or k >= max(b, c))
            assert stats.mcnemar(b, c) == pytest.approx(extreme / 2 ** n, abs=1e-12), (b, c)


def test_newcombes_interval_holds_the_difference_and_turns_over_when_the_two_sides_swap():
    for a, b, c, d in itertools.product((0, 3, 9), (0, 2, 7, 12), (0, 1, 8), (0, 4, 10)):
        if a + b + c + d == 0:
            continue
        low, high = stats.newcombe(a, b, c, d)
        n = a + b + c + d
        assert low <= (b - c) / n <= high and -1 <= low and high <= 1, (a, b, c, d)
        swapped = stats.newcombe(a, c, b, d)
        assert (swapped[0], swapped[1]) == pytest.approx((-high, -low), abs=1e-12), (a, b, c, d)


def test_wilson_holds_the_proportion_and_is_the_textbook_value_for_five_of_ten():
    for n in (1, 2, 10, 57):
        for k in range(n + 1):
            low, high = stats.wilson(k, n)
            assert -1e-12 <= low <= k / n + 1e-12 and k / n - 1e-12 <= high <= 1 + 1e-12, (k, n)  # to float rounding at 0 and n
    assert stats.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)  # the 95% interval of 5 in 10, from the tables


def test_paired_rows_count_a_b_c_d_from_the_control_and_the_tested_result():
    def row(control: bool, tested: bool):
        return ({"qid": "q"}, {"passed": control}, {"passed": tested})

    rows = [row(True, True)] * 3 + [row(False, True)] * 2 + [row(True, False)] + [row(False, False)] * 4
    assert stats.pair_counts(rows) == (3, 2, 1, 4) and stats.pair_counts([]) == (0, 0, 0, 0)


def test_the_median_the_mean_and_the_ratio_say_none_where_there_is_nothing_to_say():
    assert stats.median([]) is None and stats.median([3.0, 1.0, 2.0]) == 2.0 and stats.median([1.0, 3.0]) == 2.0
    assert stats.mean([]) is None and stats.mean([1, 2, 6]) == 3.0
    assert stats.ratio(3, 2) == 1.5 and stats.ratio(None, 2) is None and stats.ratio(3, None) is None and stats.ratio(3, 0) is None
    assert math.isclose(stats.ratio(1, 3), 1 / 3)
