"""I7, the receipts, from its real sources on disk: the tiny folder built (scans named and timed, SROIE's copied and CORD's saved as JPEGs), every answer
derived again here by a plain reader of the parquet files (never the scenario's: its own date and amount reading, its own shop and item matching), the
held-out receipts on no scan that is left, the grader taking each truth and refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and
`report` over it with a stand-in agent; and the readings of dates, amounts and names over planted ones. Skips, and says so, without the sources or
`pyarrow`."""

from __future__ import annotations

import difflib
import functools
import io
import json
import re
import threading
from collections import defaultdict
from datetime import date, datetime

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401
from PIL import Image

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import realdump

pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow reads the receipts of I7: `uv run --with pyarrow`")
from mmsb.scenarios import i7_receipts as I7  # noqa: E402
from mmsb.scenarios.i7_receipts import names, pile as piling, source  # noqa: E402

if not (source.SROIE / "data").is_dir() or not (source.CORD / "data").is_dir():
    pytest.skip(f"the SROIE and CORD receipts are not at {source.SROIE} and {source.CORD}, so I7 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("i7")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
LEGAL = re.compile(r"\b(?:sdn|bhd|berhad|s/b|co|ltd)\b\.?", re.IGNORECASE)
DAY = r"(\d{1,2} \w+ \d{4})"
ASKS = {"total": re.compile(rf"How much did I pay at (.+) on {DAY}\?$"), "bill": re.compile(rf"Find the receipt from (.+) on {DAY}\.$"),
        "shop": re.compile(r"Show me every receipt from (.+)\.$"), "howmany": re.compile(r"How many receipts do I have from (.+)\?$"),
        "biggest": re.compile(r"Which shop gave me my biggest bill in (\w+ \d{4})\?$"), "day": re.compile(r"On which day did I shop at (.+)\?$"),
        "did": re.compile(r"Did I ever shop at (.+)\? If so, on which days\?$"), "price": re.compile(r"How much did I pay for the (.+), in rupiah\?$"),
        "item": re.compile(r"Find the receipt with the (.+) on it\.$"), "items": re.compile(r"Find every receipt that has (.+) on it\.$"),
        "paid": re.compile(r"What was the total on the receipt with the (.+), in rupiah\?$")}


def fold(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("'", "")))


def shop_fold(company: str) -> str:
    return fold(LEGAL.sub(" ", company))


@functools.cache
def plain_bills() -> dict[str, dict]:
    """From the parquet files, as a second reader: each SROIE receipt's company, printed date, total and lines by its key."""
    out = {}
    for f in sorted((source.SROIE / "data").glob("*.parquet")):
        for r in pq.read_table(f, columns=["key", "entities", "words"]).to_pylist():
            out[r["key"]] = {**r["entities"], "words": r["words"], "key": r["key"]}
    return out


@functools.cache
def day_first() -> dict[str, bool | None]:
    """For each company, whether it prints the day first, from its receipts' numeric dates: a first number over 12, or a second."""
    seen = defaultdict(set)
    for r in plain_bills().values():
        m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", r["date"].strip())
        if m and (int(m[1]) > 12 or int(m[2]) > 12):
            seen[r["company"]].add(int(m[1]) > 12)
    return {c: (next(iter(s)) if len(s) == 1 else None) for c, s in seen.items()}


def iso(r: dict) -> str | None:
    """A receipt's printed date as ISO by the formats `strptime` knows, or None: written here, not the scenario's."""
    text = r["date"].strip()
    for fmt in ("%d %b %Y", "%d/%b/%Y", "%d-%b-%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            d = datetime.strptime(text.title(), fmt).date()
            return d.isoformat() if 2015 <= d.year <= 2019 else None
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    order = day_first().get(r["company"])
    if not m or order is None:
        return None
    day, month = (int(m[1]), int(m[2])) if order else (int(m[2]), int(m[1]))
    try:
        return date(int(m[3]), month, day).isoformat() if 2015 <= int(m[3]) <= 2019 else None
    except ValueError:
        return None


def ringgit(text: str) -> float | None:
    m = re.fullmatch(r"(?:RM|\$)?\s*([\d,]*\d\.\d{1,2})", text.strip(), re.IGNORECASE)
    return float(m[1].replace(",", "")) if m and float(m[1].replace(",", "")) > 0 else None


@functools.cache
def plain_slips() -> dict[str, dict]:
    """Each CORD receipt by the key the scenario gives (`train2-012`): its menu lines (name, count, price) and its total, read here."""
    out = {}
    for f in sorted((source.CORD / "data").glob("*.parquet")):
        split, part, parts = re.match(r"(\w+?)-(\d+)-of-(\d+)", f.name).groups()
        stem = split + (str(int(part)) if int(parts) > 1 else "")
        for n, r in enumerate(pq.read_table(f, columns=["ground_truth"]).to_pylist()):
            out[f"{stem}-{n:03d}"] = json.loads(r["ground_truth"])["gt_parse"]
    return out


def lines_of(parse: dict) -> list[tuple[str, int, int]] | None:
    """The `(name, count, rupiah)` of each menu line (extras included), or None when one is not read whole."""
    menu = parse.get("menu")
    out = []

    def one(line) -> bool:
        if not isinstance(line, dict) or not isinstance(line.get("nm"), str) or not line["nm"].strip():
            return False
        price = re.fullmatch(r"(?:Rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+)", str(line.get("price", "")).strip())
        count = line.get("cnt")
        n = re.fullmatch(r"\s*(\d+)\s*[xX]?\s*", count) if isinstance(count, str) else None
        if not price or (count is not None and not n):
            return False
        out.append((line["nm"].strip(), int(n[1]) if n else 1, int(re.sub(r"[.,]", "", price[1]))))
        subs = line.get("sub")
        return all(one(s) for s in ([] if subs is None else subs if isinstance(subs, list) else [subs]))

    lines = menu if isinstance(menu, list) else [menu] if isinstance(menu, dict) else []
    return out if lines and all(one(x) for x in lines) and "void_menu" not in parse else None


def all_names(node) -> list[str]:
    """The name of every line of a menu in the annotation, extras and voided lines among them, as a second reader."""
    if isinstance(node, dict):
        return [v for k, v in node.items() if k == "nm" and isinstance(v, str)] + [n for k, v in node.items() if k in ("sub", "void_menu", "menu") for n in all_names(v)]
    return [n for x in node for n in all_names(x)] if isinstance(node, list) else []


def total_of(parse: dict) -> int | None:
    t = parse.get("total")
    m = re.fullmatch(r"(?:Rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+)", str(t.get("total_price", "")).strip()) if isinstance(t, dict) else None
    return int(re.sub(r"[.,]", "", m[1])) if m else None


def plan(scale: str, seed: int):
    return _plan(scale, seed)


@functools.cache
def _plan(scale: str, seed: int):
    return I7.plan_of(I7.recipe(scale=scale, seed=seed))


def iso_of_text(text: str) -> str:
    d, month, y = text.split()
    return f"{y}-{MONTHS.index(month) + 1:02d}-{int(d):02d}"


def near_in_source(label: str, others: list[str]) -> list[str]:
    """The other labels near `label`: one holds all the words of the other, or they are 85% alike (difflib's own matcher)."""
    mine = shop_fold(label)
    return [o for o in others if shop_fold(o) != mine and (set(mine.split()) <= set(shop_fold(o).split()) or set(shop_fold(o).split()) <= set(mine.split())
                                                           or difflib.SequenceMatcher(None, mine, shop_fold(o)).ratio() >= 0.85)]


def check_against_parquet(files, questions) -> None:
    """Each question's truth and evidence as the plain reader finds them in the parquet files."""
    name = {f.spec["key"]: f.path for f in files}
    bills = {f.spec["key"]: plain_bills()[f.spec["key"]] for f in files if f.spec["set"] == "sroie"}
    slips = {f.spec["key"]: plain_slips()[f.spec["key"]] for f in files if f.spec["set"] == "cord"}
    shops = defaultdict(list)
    for k, r in bills.items():
        shops[shop_fold(r["company"])].append(k)
    every_labels = [r["company"] for r in plain_bills().values()]
    on_item = defaultdict(list)  # item name folded -> [(slip key, count)] over every line of the folder
    for k, parse in slips.items():
        for nm, n, price in lines_of(parse):
            on_item[fold(nm)].append((k, n, price))
    gone_slip = {k for k in plain_slips() if k not in slips}
    for q in questions:
        template = q.tags["template"].removeprefix("absent_")
        absent = q.kind == "absent"
        assert (q.tags["template"] != template) is absent, q.qid
        match = ASKS[template].search(q.question)
        assert match, f"{q.qid}: {q.question}"
        if template in ("total", "bill", "shop", "howmany", "day", "did"):
            shop = fold(match[1])
            found = shops.get(shop, [])
            ours = [k for k in found if template not in ("total", "bill") or iso(bills[k]) == iso_of_text(match[2])]
            if absent:
                assert not q.truth.evidence and q.grade.rule == "absent", q.qid
                assert not ours, f"{q.qid}: the folder holds {ours[:2]}"
                if template == "bill":
                    assert found, f"{q.qid}: the shop is not there at all, and a held-out bill is of a shop that is"
                else:
                    assert not found and (shop in {shop_fold(r["company"]) for r in plain_bills().values()} or shop in
                                          {fold(n) for n in I7.questions.NEVER}), q.qid
                continue
            label = bills[found[0]]["company"]
            assert not near_in_source(label, every_labels), f"{q.qid}: {label} has a near name {near_in_source(label, every_labels)[:2]}"
            assert sorted(s.file for s in q.truth.evidence) == sorted(name[k] for k in (ours if template in ("total", "bill") else found)), q.qid
            if template in ("total", "bill"):
                assert len(ours) == 1 and all(iso(bills[k]) for k in found), q.qid
                if template == "bill":
                    assert q.grade.items == (name[ours[0]].rsplit("/", 1)[-1],), q.qid
                else:
                    t = ringgit(bills[ours[0]]["total"])
                    assert q.grade.truth == f"{t:.2f}" and q.grade.tol_s == 0.05 and q.grade.rule == "amount", q.qid
                    assert any(f"{t:.2f}" in w or f"{t:.2f}".rstrip("0").rstrip(".") in w for w in bills[ours[0]]["words"]), f"{q.qid}: the total is not printed"
            elif template == "shop":
                assert 2 <= len(found) <= 12 and set(q.grade.items) == {name[k].rsplit("/", 1)[-1] for k in found}, q.qid
            elif template == "howmany":
                assert 2 <= len(found) <= 40 and q.grade.truth == str(len(found)), q.qid
            elif template == "day":
                assert len(found) == 1 and q.grade.items == (iso(bills[found[0]]),), q.qid
            else:
                assert 2 <= len(found) <= 4 and q.grade.items == tuple(sorted({iso(bills[k]) for k in found})), q.qid
            continue
        if template == "biggest":
            month = f"{MONTHS.index(match[1].split()[0]) + 1:02d}"
            ranked = sorted(((ringgit(r["total"]), r["company"]) for r in bills.values() if iso(r) and iso(r)[:7] == f"{match[1].split()[1]}-{month}"), reverse=True)
            assert len(ranked) >= 8 and ranked[0][0] >= 1.2 * ranked[1][0], q.qid
            assert shop_fold(ranked[0][1]) == fold(q.truth.value) or fold(q.truth.value) in shop_fold(ranked[0][1]), q.qid
            continue
        item = fold(match[1])
        if absent:
            assert not q.truth.evidence and not on_item.get(item) and q.grade.rule == "absent", q.qid
            assert any(fold(nm) == item for k in gone_slip for nm, _, _ in (lines_of(plain_slips()[k]) or [])), f"{q.qid}: no receipt held out has it"
            squashed = item.replace(" ", "")  # and no line of any receipt of the source, read whole or not, is the item with its spaces off, or near it
            for k, parse in plain_slips().items():
                for nm in all_names(parse):
                    other = fold(nm).replace(" ", "")
                    if fold(nm) == item:
                        assert k in gone_slip, f"{q.qid}: {nm} is on {k}"
                        continue
                    assert not (other == squashed or (min(len(other), len(squashed)) >= 5 and (squashed in other or other in squashed))
                                or difflib.SequenceMatcher(None, squashed, other).ratio() >= 0.75), f"{q.qid}: {nm} on {k} is near it"
            continue
        lines = on_item[item]
        if template == "items":
            assert 2 <= len({k for k, _, _ in lines}) <= 8 and set(q.grade.items) == {name[k].rsplit("/", 1)[-1] for k, _, _ in lines}, q.qid
        else:
            assert len(lines) == 1 and lines[0][1] == 1 and all(w.isalpha() for w in item.split()), q.qid
            k, _, price = lines[0]
            assert [s.file for s in q.truth.evidence] == [name[k]], q.qid
            if template == "item":
                assert q.grade.items == (name[k].rsplit("/", 1)[-1],), q.qid
            else:
                want = price if template == "price" else total_of(slips[k])
                assert q.grade.truth == str(want) and q.grade.tol_s == 0.0 and q.grade.rule == "amount", q.qid


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "amount": lambda: g.truth, "text": lambda: g.truth.split("|")[0],
            "absent": lambda: "unknown"}[g.rule]()


def wrong(q, other_names) -> list[str]:
    g = q.grade
    if g.rule == "set":
        dates = re.fullmatch(r"\d{4}-\d\d-\d\d", g.items[0]) is not None
        extra = "2013-05-17" if dates else next(n for n in other_names if n not in g.items)
        out = [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), "unknown", "none"]
        return out + ([", ".join(g.items[1:])] if len(g.items) > 1 else [])
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", "unknown"]
    if g.rule == "amount":
        t = float(g.truth)
        return [f"{t + 1:g}", f"{t * 2 + 1:g}", "unknown", f"{t:g} or {t + 3:g}"]
    if g.rule == "text":
        return ["unknown", "I cannot tell", "maybe a bakery"] + [w for w in g.wrong]
    return [", ".join(other_names[:2]), other_names[0], "3", "Yes, there are some."]


@pytest.fixture(scope="module")
def tiny():
    return plan("tiny", 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i7")
    recipe = I7.recipe(scale="tiny", seed=1)
    dump, questions = I7.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_folder_is_a_few_scans_with_a_third_of_its_questions_absent(tiny):
    files, questions = tiny
    assert 12 <= len(files) <= 30 and len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 3
    assert {q.tags["template"] for q in questions} == {"total", "shop", "howmany", "price", "item", "absent_did", "absent_bill", "absent_item"}
    assert {f.spec["set"] for f in files} == {"sroie", "cord"} and {q.split for q in questions} == {"dev", "test"}


@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3)), ("standard", (1, 2, 3))])
def test_every_answer_is_the_parquet_labels(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        check_against_parquet(files, questions)
        if scale == "standard":
            assert 1300 < len(files) < 1500 and len(questions) == 22 and sum(q.kind == "absent" for q in questions) == 5, seed
            assert {q.tags["template"] for q in questions} == {f"{a}{t}" for a in ("", ) for t in ("total", "bill", "shop", "howmany", "biggest", "day", "did",
                                                                                                  "price", "item", "items", "paid")} | {"absent_did", "absent_bill", "absent_item"}
            assert 5 <= sum(q.split == "dev" for q in questions) <= 8


def test_the_folder_is_the_same_at_every_seed_and_the_questions_differ():
    files_1, one = plan("standard", 1)
    files_2, two = plan("standard", 2)
    assert files_1 == files_2, "a seed draws questions from one folder"
    assert {q.qid for q in one} != {q.qid for q in two}


def test_every_receipt_of_the_folder_has_a_date_that_reads_one_way_a_printed_total_and_is_the_only_one_of_its_shop_day_and_total():
    files, _ = plan("standard", 1)
    seen = set()
    for f in files:
        if f.spec["set"] == "sroie":
            r = plain_bills()[f.spec["key"]]
            assert iso(r) and ringgit(r["total"]), f.spec["key"]
            t = ringgit(r["total"])
            assert any(f"{t:.2f}" in w or f"{t:.2f}".rstrip("0").rstrip(".") in w for w in r["words"]), f"{f.spec['key']}: the total is not printed"
            assert (r["company"], iso(r), t) not in seen, f"{f.spec['key']} is a second scan of a receipt"
            seen.add((r["company"], iso(r), t))
        else:
            parse = plain_slips()[f.spec["key"]]
            assert lines_of(parse) and total_of(parse), f.spec["key"]


def test_a_shop_that_prints_the_month_first_has_no_numeric_date_read():
    """Three shops of SROIE print month first (`12/28/2017`): a receipt of one with both numbers 12 or under is dated month first by the scenario's reading
    only through its shop's others, and the plain reader agrees on each receipt it dates."""
    bills = source.read_bills(str(source.SROIE))
    month_first = {r["company"] for r in plain_bills().values() if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", r["date"].strip())
                   and int(re.split(r"[/-]", r["date"].strip())[1]) > 12}
    assert len(month_first) == 3
    for b in bills:
        assert b.date == iso(plain_bills()[b.key]), b.key
    assert any(b.company in month_first and b.date for b in bills)


def test_the_scans_are_named_in_the_order_of_their_times_in_sittings_that_have_nothing_to_do_with_the_shop():
    files, _ = plan("standard", 1)
    assert [f.path for f in files] == [f"scan_{n:04d}.jpg" for n in range(1, len(files) + 1)] and [f.t for f in files] == sorted(f.t for f in files)
    gaps = [b.t - a.t for a, b in zip(files, files[1:])]
    sittings = 1 + sum(g > 600 for g in gaps)
    assert sittings == 9 and all(4 <= g <= 14 for g in gaps if g <= 600)
    keys = [f.spec["key"] for f in files]
    assert keys != sorted(keys) and [f.spec["set"] for f in files] != sorted(f.spec["set"] for f in files)
    shops = [shop_fold(plain_bills()[f.spec["key"]]["company"]) for f in files if f.spec["set"] == "sroie"]
    same = sum(a == b for a, b in zip(shops, shops[1:])) / len(shops)
    share = max(shops.count(s) for s in set(shops)) / len(shops)
    assert same < 3 * share * share + 0.02, "a shop's receipts are not next to one another"


def test_no_size_or_count_answer_is_given_by_the_list_of_the_files():
    """The list of the files shows the folder's size, and a count asked of the whole folder would be given by it: every `howmany` and `items` answer is under
    a third of the receipts."""
    files, questions = plan("standard", 1)
    for q in questions:
        if q.tags["template"] == "howmany":
            assert int(q.grade.truth) < len(files) / 3


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale, seed in (("tiny", 1), ("standard", 1), ("standard", 2)):
        files, questions = plan(scale, seed)
        names_ = [f.path for f in files]
        for q in questions:
            assert grade.grade(q.grade, right(q)), (q.qid, right(q))
            for answer in wrong(q, names_):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_an_amount_is_taken_to_the_cent_of_a_decimal_the_sen_of_a_cash_rounding_and_a_rupiah_by_its_thousands():
    standard = plan("standard", 1)[1]
    total = next(q for q in standard if q.tags["template"] == "total")
    t = float(total.grade.truth)
    for ok in (f"{t:.2f}", f"RM {t:,.2f}", f"It was {t:.2f} ringgit", f"{t + 0.04:.2f}"):
        assert grade.grade(total.grade, ok), ok
    for bad in (f"{t + 0.06:.2f}", f"{t + 1:.2f}", f"RM {t:.2f} or RM {t + 5:.2f}"):
        assert not grade.grade(total.grade, bad), bad
    assert grade.grade(total.grade, f"RM{t:.2f}"), "a currency stuck to the number is no name, as `cam3` is"
    price = next(q for q in standard if q.tags["template"] == "price")
    n = int(price.grade.truth)
    for ok in (str(n), f"Rp {n:,}", f"{n:,} rupiah", f"IDR {n}"):
        assert grade.grade(price.grade, ok), ok
    for bad in (str(n + 1), str(n + 1000), f"{n} or {n + 1000}"):
        assert not grade.grade(price.grade, bad), bad


def test_the_shop_of_the_biggest_bill_is_told_by_its_name_or_a_word_no_other_shop_has_and_not_by_another_shop():
    for seed in (1, 2, 3):
        _, questions = plan("standard", seed)
        (q,) = [q for q in questions if q.tags["template"] == "biggest"]
        said = q.truth.value
        assert grade.grade(q.grade, said) and grade.grade(q.grade, f"It was {said}."), q.qid
        assert all(grade.grade(q.grade, w.title()) for w in q.grade.truth.split("|")[1:]), "a word of the name that no other shop has"
        assert all(not grade.grade(q.grade, other) for other in q.grade.wrong) and not grade.grade(q.grade, "unknown"), q.qid


def test_a_scan_is_copied_as_it_is_and_a_photograph_is_saved_as_a_jpeg_of_its_size_and_the_probe_passes(built):
    out, recipe, dump, _ = built
    files, _ = plan("tiny", 1)
    assert [e.path for e in dump.entries] == [f.path for f in files]
    for e, f in zip(dump.entries, files):
        data = (out / "dump" / e.path).read_bytes()
        image = Image.open(io.BytesIO(data))
        assert image.format == "JPEG" and not image.getexif() and abs((out / "dump" / e.path).stat().st_mtime - f.t) < 1e-3 and e.t == f.t
        if f.spec["set"] == "sroie":
            blobs = pq.ParquetFile(source.SROIE / "data" / f.spec["parquet"]).read_row_group(f.spec["row"] // 100, columns=["image"]).to_pylist()
            assert data == blobs[f.spec["row"] % 100]["image"]["bytes"], e.path
        else:
            batch = next(pq.ParquetFile(source.CORD / "data" / f.spec["parquet"]).iter_batches(batch_size=1 + f.spec["row"], columns=["image"]))
            original = Image.open(io.BytesIO(batch.column("image").to_pylist()[f.spec["row"]]["bytes"]))
            assert image.size == original.size and original.format == "PNG", e.path
    assert I7.probe(recipe, dump, out, None, 1).status == "passed"


def test_a_name_that_says_a_shop_or_an_item_fails_the_probe_and_a_jpeg_with_a_comment_fails_the_copy(built, tmp_path):
    out, recipe, dump, _ = built
    raw = (out / "dump" / dump.entries[0].path).read_bytes()
    folder = tmp_path / "dump"
    entries = realdump.write([realdump.File("scan of Gardenia Bakeries.jpg", "scans", 1.0, "scan", "image", {})], folder, 1, lambda f, p: p.write_bytes(raw))
    outcome = I7.probe(recipe, realdump.dump_of("k", folder, "i7_receipts", entries), tmp_path, None, 1)
    assert outcome.status == "failed" and "gardenia" in outcome.failures[0].casefold()
    commented = raw[:2] + b"\xff\xfe\x00\x08hello!" + raw[2:]
    with pytest.raises(ValueError, match="segment of marker 0xfe"):
        source.jpeg_plain(commented)
    scan = pq.ParquetFile(source.SROIE / "data" / "test-00000-of-00001.parquet").read_row_group(0, columns=["image"]).to_pylist()[0]["image"]["bytes"]
    assert source.jpeg_plain(scan) == scan, "a scan of SROIE has no segment but its tables"


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = I7.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I7.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I7.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I7.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_sources_or_the_code():
    deny = I7.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:sroie" in deny and "in:i7_receipts" in deny and "in:from mmsb" in deny


# -- the readings, over planted ones -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, order, iso_date", [
    ("24/12/2017", True, "2017-12-24"), ("06/01/2018", True, "2018-01-06"), ("06/01/2018", False, "2018-06-01"), ("06/01/2018", None, None),
    ("12 MAR 2018", None, "2018-03-12"), ("27/MAR/2018", None, "2018-03-27"), ("02-JAN-2017", None, "2017-01-02"), ("2018/03/27", None, "2018-03-27"),
    ("2016-07-31", None, "2016-07-31"), ("07-02-17", True, None), ("31/02/2018", True, None), ("15/01/2030", True, None), ("20171224", True, None),
    ("12/28/2017", False, "2017-12-28")])
def test_a_printed_date_is_read_one_way_or_not_at_all(text, order, iso_date):
    assert source.read_date(text, order) == iso_date


@pytest.mark.parametrize("text, value", [("193.00", 193.0), ("RM 8.60", 8.6), ("$9.5", 9.5), ("RM1,250.00", 1250.0), ("-9.99", None), ("", None),
                                         ("8,60", None), ("RM", None), ("12", None)])
def test_a_printed_amount_of_ringgit_is_a_plain_one_or_none(text, value):
    assert source.amount(text) == value


@pytest.mark.parametrize("text, value", [("60.000", 60000), ("13,500", 13500), ("20000", 20000), ("Rp 5.455", 5455), ("2.50", None), ("1.2", None),
                                         ("-60.000", None), (None, None), ("", None), ("60.00.0", None)])
def test_a_price_is_a_whole_number_of_rupiah_or_none(text, value):
    assert source.rupiah(text) == value


@pytest.mark.parametrize("line, items", [
    ({"nm": "Thai Tea", "cnt": "1", "price": "20.000"}, [("Thai Tea", 1, 20000)]),
    ({"nm": "Thai Tea", "price": "20,000"}, [("Thai Tea", 1, 20000)]),
    ({"nm": "Set", "cnt": "2 x", "price": "40.000", "sub": [{"nm": "Ice", "cnt": "2", "price": "0"}]}, [("Set", 2, 40000), ("Ice", 2, 0)]),
    ({"nm": "", "price": "1.000"}, None), ({"price": "1.000"}, None), ({"nm": "X", "price": "1.5"}, None), ({"nm": "X", "cnt": "some", "price": "1.000"}, None),
    ({"nm": "Set", "price": "40.000", "sub": [{"price": "5.000"}]}, None), ("Thai Tea", None)])
def test_a_menu_line_is_read_whole_or_not_at_all(line, items):
    got = source.read_item(line)
    assert (None if got is None else [(i.name, i.count, i.price) for i in got]) == items


@pytest.mark.parametrize("label, key, said", [
    ("GARDENIA BAKERIES (KL) SDN BHD", ("gardenia", "bakeries", "kl"), "Gardenia Bakeries (KL)"), ("99 SPEED MART S/B", ("99", "speed", "mart"), "99 Speed Mart"),
    ("MR. D.I.Y. (M) SDN BHD", ("mr", "d", "i", "y", "m"), "MR. D.I.Y. (M)"), ("AEON CO. (M) BHD", ("aeon", "m"), "Aeon (M)"),
    ("KING'S CONFECTIONERY S/B", ("kings", "confectionery"), "King's Confectionery"), ("BEMED (SP) SDN. BHD.", ("bemed", "sp"), "Bemed (SP)"),
    ("DOMINO'S PIZZA", ("dominos", "pizza"), "Domino's Pizza")])
def test_a_shop_is_told_by_the_words_of_its_label_but_the_legal_ending_and_said_in_the_case_a_person_writes_it(label, key, said):
    assert names.shop_key(label) == key and names.spoken(label) == said


def test_a_name_is_clean_only_where_no_other_is_near_it_by_its_words_or_a_misprint():
    keys = [("gardenia", "bakeries", "kl"), ("gardenia", "bakeries", "ki"), ("restoran", "wan", "sheng"), ("restoran", "wan", "she"),
            ("unihakka", "international"), ("unihakka", "interantional"), ("mr", "d", "i", "y", "m"), ("mr", "d", "i", "y", "kuchai"),
            ("sanyu", "stationery", "shop"), ("popular", "book"), ("aeon", "m"), ("yhm", "aeon", "tebrau", "city"),
            ("tan", "wah"), ("lee", "tan", "wah")]
    clean = names.clean(keys)
    assert clean == {("sanyu", "stationery", "shop"), ("popular", "book"), ("aeon", "m"), ("yhm", "aeon", "tebrau", "city")}, "a misprint, a branch, a letter off"
    assert names.near(("a", "b"), ("a", "b", "c")) and names.near(("b", "c"), ("a", "b", "c")) and not names.near(("a", "b"), ("a", "b"))
    assert names.near(("go", "hiong"), ("ngohiong",)) and names.near(("thai", "tea"), ("thaitea",)) and names.near(("ice", "cream", "cone"), ("icecream", "cone"))
    assert not names.near(("tea",), ("steak",)), "a name of under five letters inside another is none"
    assert not names.near(("restoran", "wan", "sheng"), ("restoran", "xyz", "abc")), "one generic word in common is no branch"


def test_a_receipt_held_out_is_of_a_shop_the_folder_still_has_or_has_not_and_an_item_on_no_other_receipt():
    pile = piling.choose(source.read_bills(str(source.SROIE)), source.read_slips(str(source.CORD)), "standard")
    kept = {b.company for b in pile.bills}
    assert len(pile.held_bills) == 8 and all(b.company in kept and b not in pile.bills and names.shop_key(b.company) in pile.shops for b in pile.held_bills)
    assert len(pile.held_shops) == 6 and all(b.company not in kept and names.shop_key(b.company) in pile.shops for b in pile.held_shops)
    keys_on = {names.item_key(i.name) for s in pile.slips for i in s.items}
    assert len(pile.held_slips) == 8 and all(k not in keys_on and k in pile.items and s not in pile.slips for s, k in pile.held_slips)


# -- the whole, run ------------------------------------------------------------------------------------------------------------------------

class Spy:
    """An agent that answers right in the `files` configuration, citing a scan of the truth, and `unknown` in any other."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": 0, "t1": 0} for s in q.truth.evidence[:1]]
        answer = right(q) if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


def test_build_run_and_report_over_the_tiny_folder(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "i7_receipts", "mmsb.scenarios.i7_receipts")
    assert world.main("build", "--scenario", "i7_receipts", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    prompt = next(p for (config, qid), p in spy.prompts.items() if config == "files" and qid.startswith("total-"))
    assert prompt.startswith("It is now ") and " The files of my folder of scanned receipts and bills are in " in prompt and "scan_0001.jpg onwards" in prompt
    assert "<the amount in ringgit as a plain number, such as 12.50, with no RM before it>" in prompt and "You have 900 seconds: " in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    (section,) = fs.read_json(folder / "report.json")["scenarios"]
    files_cell, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files_cell["correct"], files_cell["n"], files_cell["grounding"]["grounded"]) == (8, 8, 8)
    assert (listing["correct"], listing["n"]) == (3, 8) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 8}
