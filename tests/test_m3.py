"""M3, the voicemail box: the offline parts (the WAV rewritten with its samples alone, the spread of the times, the amounts, percentages and
callers a transcript has, and the questions of a made-up box, every answer derived again here by a plain reader of its transcripts), and,
where the real source is on disk, the same over the Robocall Audio Dataset: the tiny box built (messages numbered in the order they came in,
copied with their samples alone and their times), every answer of the standard box derived again here from `metadata.csv` (never the
scenario's reader), the grader taking each truth and refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and `report`
over it with a stand-in agent. The real-source tests skip, and say so, without the source."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import struct
import threading
import wave
from collections import defaultdict
from datetime import date, datetime, timedelta

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import m3_voicemail as M3
from mmsb.scenarios import realdump
from mmsb.scenarios.m3_voicemail import questions as Q
from mmsb.scenarios.m3_voicemail.source import SOURCE, Call, chunks, seconds_of, with_samples_only

REAL = pytest.mark.skipif(not (SOURCE / "metadata.csv").is_file(), reason=f"the Robocall Audio Dataset is not at {SOURCE}, so M3 is not built")
GROUP = pytest.mark.xdist_group("m3")
FMT = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
MONTHS = [m.casefold() for m in realdump.MONTHS]


def wav(samples: bytes, fmt: bytes = FMT, extra: bytes = b"LIST\x10\x00\x00\x00INFOISFT\x04\x00\x00\x00Lavf") -> bytes:
    """A WAV as the source's are: `fmt `, a `LIST` chunk with the converter's name, `data`."""
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + extra + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


def call(n: int, text: str, language: str = "en", seconds: float = 20.0) -> Call:
    return Call(f"audio-wav-16khz/{n}_normalized.wav", language, text, "cease-desist-letter-some-telecom", seconds)


# -- the offline parts --------------------------------------------------------------------------------------------------------------

def test_a_wav_is_rewritten_as_its_format_and_samples_alone_and_a_wrong_one_is_refused():
    samples = bytes(range(200))
    raw = wav(samples)
    assert [k for k in chunks(raw)] == [b"fmt ", b"LIST", b"data"]
    clean = with_samples_only(raw)
    assert list(chunks(clean)) == [b"fmt ", b"data"] and b"Lavf" not in clean and clean.endswith(samples) and len(clean) == 44 + len(samples)
    with pytest.raises(ValueError, match="16-bit mono"):
        with_samples_only(wav(samples, struct.pack("<HHIIHH", 1, 2, 16000, 64000, 4, 16)))
    with pytest.raises(ValueError, match="not a WAV"):
        with_samples_only(b"x" * 100)
    with pytest.raises(ValueError, match="cut short"):
        with_samples_only(raw[:-5])


@pytest.mark.parametrize("rate", [16000, 8000])
def test_how_long_a_wav_plays_is_read_from_its_header_at_either_rate(tmp_path, rate):
    fmt = struct.pack("<HHIIHH", 1, 1, rate, 2 * rate, 2, 16)
    (tmp_path / "a.wav").write_bytes(wav(b"\0\0" * (rate * 3), fmt))
    assert seconds_of(tmp_path / "a.wav") == 3.0


def test_the_times_are_in_the_hours_of_a_day_and_each_message_is_two_minutes_clear_of_the_one_before():
    days = realdump.span_days(date(2026, 10, 26), date(2026, 11, 8))
    lengths = [random.Random(n).uniform(5, 300) for n in range(300)]
    times = M3.spread(lengths, days, random.Random("t"))
    assert times == sorted(times) and len({datetime.fromtimestamp(t).strftime("%Y%m%d%H%M") for t in times}) == 300, "each in a minute of its own"
    assert all(b >= a + length + M3.GAP_S for a, b, length in zip(times, times[1:], lengths))
    assert all(8 <= datetime.fromtimestamp(t).hour <= 21 for t in times)
    assert times == M3.spread(lengths, days, random.Random("t")) != M3.spread(lengths, days, random.Random("u"))


def test_a_dollar_amount_a_percentage_and_a_caller_are_read_only_where_the_transcript_is_plain():
    assert Q.money_of(call(1, "an order of $1,499.50 on your card")) == 1499.5
    assert Q.money_of(call(2, "an order of $20 and a charge of $30")) is None, "two amounts"
    assert Q.money_of(call(3, "a charge of $20 and fifty dollars more")) is None, "an amount in words as well"
    assert Q.money_of(call(4, "no amount here")) is None
    assert Q.percent_of(call(5, "a 50% discount, again 50 percent")) == 50.0
    assert Q.percent_of(call(6, "a rate of 0 to 5 percent")) is None, "a range"
    assert Q.percent_of(call(7, "a 30% discount and 5% interest")) is None
    assert Q.caller_of(call(8, "Hello, this is Daniel from Amazon customer service. Your Amazon account")) == "amazon"
    assert Q.caller_of(call(9, "Your Amazon account has a charge. This is the fraud team")) is None, "the target of a warning, not the caller"
    assert Q.caller_of(call(10, "Hi, I'm calling you from AT&T Direct TV about your Amazon account")) == "att"
    assert Q.caller_of(call(11, "Hello. " + "word " * 40 + "this is Daniel from Amazon")) is None, "named late"


def synthetic() -> list[Call]:
    """A made-up set of messages: five of each scheme of SCHEMES (one of those that a single message has) but the electric company, which is
    left out so that the messages of a caller with no scheme (Jersey Central Power and Light) are never a scheme's near words; and messages
    with an amount and a caller of their own."""
    out, n = [], 0
    for key, scheme in Q.SCHEMES.items():
        if key == "electric":
            continue
        word = re.sub(r"^(?:my|a|an|the) ", "", scheme.said)
        for _ in range(1 if key in Q.SINGLES else 5):
            n += 1
            out.append(call(n, f"Hello, calling about {word}. Press 1 to talk. Bye"))
    for amount, kind in ((999, "a"), (999, "b"), (999, "c"), (1537.35, "d"), (1537.35, "e"), (1537.35, "f"), (50, "g")):
        n += 1
        out.append(call(n, f"Hello, this is Daniel from Acme Corp. A payment of ${amount} is due. Press 1 to cancel {kind}"))
    for _ in range(6):
        n += 1
        out.append(call(n, "Hello, this is the Notifications Department with Jersey Central Power and Light. You owe 30% more. Press 1"))
    out += [call(n + k, f"Thank you for listening, this is message {k}. Press 1 to stop") for k in range(1, 200)]
    out.append(call(n + 999, "ni hao", language="zh"))
    return out


def plain_check(calls, kept, files, questions, scale) -> set[str]:
    """Each question's truth and evidence as a plain reader of the transcripts of the box `kept` finds them; returns the schemes asked about."""
    schemes = {**Q.SCHEMES, **Q.ABSENT}
    english = [c for c in calls if c.language == "en"]
    asked = set()
    seen = defaultdict(set)  # the messages that have each dollar amount and percentage, over the dataset

    def has(key, c, near=False):
        return re.search(schemes[key].near if near else schemes[key].strict, c.transcript, re.I) is not None

    for c in english:
        for m in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", c.transcript):
            seen["$" + str(float(m.replace(",", "")))].add(c.file)
        for m in re.findall(r"(\d+(?:\.\d+)?)\s?(?:%|percent)", c.transcript, re.I):
            seen["%" + str(float(m))].add(c.file)
    assert [f.path for f in files] == [f"voicemail/msg_{n:04d}.wav" for n in range(1, len(kept) + 1)]
    assert all(b.t >= a.t + a.length + M3.GAP_S for a, b in zip(files, files[1:])), "two minutes at least between messages"
    assert all(f.length == c.seconds and f.spec["src"] == c.file for f, c in zip(files, kept, strict=True))
    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template, key = (parts[1], parts[2]) if absent else (parts[0], parts[1])
        assert q.tags["template"] == ("absent_" + template if absent else template) and (q.kind == "absent") is absent, q.qid
        assert q.tags["modality"] == "audio" and q.scenario == "M3"
        if template in ("scheme", "count", "first", "single"):
            asked.add(key)
            said = schemes[key].said
            assert said in q.question, f"{q.qid} asks of another scheme: {q.question}"
            have = [n for n, c in enumerate(kept) if has(key, c)]
            if absent:
                assert not have and not [c for c in kept if key in Q.SCHEMES and has(key, c, near=True)], f"{q.qid}: the box holds some"
                assert q.grade.rule == "absent" and not q.truth.evidence and q.truth.value is None
                if key in Q.ABSENT:
                    assert not [c for c in calls if has(key, c)], f"{q.qid}: a name the dataset has"
                continue
            assert key in Q.SCHEMES and have and not [c for c in kept if not has(key, c) and has(key, c, near=True)], q.qid
            assert len(have) == len([c for c in english if has(key, c)]), f"{q.qid}: the box has not every message of it"
            if parts[-1] == "week":  # the week the question names, read from its words: a Monday to the Sunday after
                (m1, d1), (m2, d2) = [(MONTHS.index(m.casefold()) + 1, int(d)) for _, d, m in re.findall(r"(Monday|Sunday) (\d+) (\w+)", q.question)]
                year = datetime.fromtimestamp(files[0].t).year
                start, end = date(year, m1, d1), date(year, m2, d2)
                assert start.weekday() == 0 and end - start == timedelta(days=6), q.qid
                have = [n for n in have if start <= datetime.fromtimestamp(files[n].t).date() <= end]
                assert Q.WEEK_FILES[0] <= len(have) <= Q.WEEK_FILES[1]
            names = [files[n].path.rsplit("/", 1)[-1] for n in have]
            assert sorted(s.file for s in q.truth.evidence) == sorted(files[n].path for n in have) or template == "first", q.qid
            if template == "scheme":
                assert q.grade.rule == "set" and sorted(q.grade.items) == sorted(names) and Q.PLAIN_COUNT[0] <= len(names) <= Q.MAX_FILES or key in Q.SINGLES
            elif template == "single":
                assert key in Q.SINGLES and q.grade.items == tuple(names) and len(names) == 1
            elif template == "count":
                assert (q.grade.rule, q.grade.truth) == ("count", str(len(have))) and 3 <= len(have) <= 40, q.qid
            else:
                n = have[0]
                assert n >= Q.EARLIEST and [s.file for s in q.truth.evidence] == [files[n].path], q.qid
                when = datetime.fromtimestamp(files[n].t)
                assert (q.grade.rule, q.grade.date, q.grade.time, q.grade.tol_s) == ("time", when.date().isoformat(), when.strftime("%H:%M:%S"), 60.0)
            continue
        n = int(parts[-1]) - 1
        c, f = kept[n], files[n]
        assert [(s.file, s.t0, s.t1) for s in q.truth.evidence] == [(f.path, 0.0, f.length)], q.qid
        text = f"{realdump.WEEKDAYS[datetime.fromtimestamp(f.t).weekday()]} {datetime.fromtimestamp(f.t).day} " \
               f"{realdump.MONTHS[datetime.fromtimestamp(f.t).month - 1]} at {datetime.fromtimestamp(f.t):%H:%M}"
        assert text in q.question, f"{q.qid} names another message: {q.question}"
        assert len({datetime.fromtimestamp(g.t).strftime("%Y%m%d%H%M") for g in files}) == len(files), "a minute names one message"
        if template in ("price", "percent"):
            sign, pattern = ("$", r"\$\s?(\d[\d,]*(?:\.\d+)?)") if template == "price" else ("%", r"(\d+(?:\.\d+)?)\s?(?:%|percent)")
            found = {float(m.replace(",", "")) for m in re.findall(pattern, c.transcript, re.I)}
            assert len(found) == 1 and not re.search(r"\b(?:dollars?|bucks?|cents?|thousand|hundred|million|billion)\b", c.transcript, re.I) or template == "percent", q.qid
            (value,) = found
            assert q.grade.rule == "amount" and float(q.grade.truth) == value and len(seen[sign + str(value)]) >= Q.SEEN_BY, q.qid
        else:
            assert template == "caller" and q.grade.rule == "text"
            accepted = q.grade.truth.split("|")
            assert any(re.search(re.escape(a.replace(" t", "&t")) if a == "at t" else re.escape(a), c.transcript, re.I) for a in accepted), q.qid
            assert not [w for w in q.grade.wrong if w in accepted], q.qid
    return asked


def right(q) -> list[str]:
    """Natural answers in the shape asked, all right."""
    g = q.grade
    if g.rule == "set":
        return [", ".join(g.items), "\n".join(f"- {n}" for n in g.items), json.dumps(list(g.items))]
    if g.rule == "count":
        return [g.truth, f"{g.truth} voicemails", f"There are {g.truth} of them."]
    if g.rule == "time":
        return [f"{g.date} {g.time[:5]}", f"The first was on {g.date} at {g.time[:5]}."]
    if g.rule == "amount" and q.tags["template"] == "price":
        return [g.truth, f"${float(g.truth):,.2f}".replace(".00", ""), f"It mentioned {float(g.truth):g} dollars."]
    if g.rule == "amount":
        return [g.truth, f"{float(g.truth):g}%", f"It mentioned {float(g.truth):g} percent."]
    if g.rule == "text":
        return [g.truth.split("|")[0], f"It said it was from {g.truth.split('|')[-1]}."]
    return ["unknown", "None", "No voicemail mentions it.", "I found none, so zero.", "0"]


def wrong(q, other_names) -> list[str]:
    """Planted wrong answers: a missing, an extra and a swapped message, a made-up one; one more, one fewer and a hedge for a count; a time two
    minutes out, a day out; `unknown` on a needle; messages, a number and a yes for an absent one."""
    g = q.grade
    if g.rule == "set":
        extra = next(n for n in other_names if n not in g.items)
        return [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), ", ".join([*g.items, "msg_9999.wav"]), "unknown", "none"] + (
            [", ".join(g.items[:-1])] if len(g.items) > 1 else [])
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", "unknown", "none"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date}T{g.time}")
        return [f"{t.replace(minute=(t.minute + 2) % 60, hour=t.hour + (t.minute + 2) // 60):%Y-%m-%d %H:%M}", f"{g.time[:5]}", "unknown"]
    if g.rule == "amount":
        v = float(g.truth)
        return [f"{v + 5:g}", f"{v:g} or {v + 5:g}", "unknown"]
    if g.rule == "text":
        return ["Nobody in particular", "unknown", g.wrong[0], f"{g.truth.split('|')[0]} or {g.wrong[0]}"]
    return [", ".join(other_names[:2]), other_names[0], "3", "Yes, there are some.", "There are 5 of them."]


def test_a_made_up_box_asks_every_template_and_each_answer_is_what_its_transcripts_have():
    calls = synthetic()
    for seed in (1, 2, 3):
        roles, kept = Q.choose(calls, ["cease-desist-letter-some-telecom"], "tiny", seed)
        assert len(kept) == Q.TINY_CALLS and all(c.language == "en" for c in kept)
        days = realdump.span_days(date(2026, 10, 26), date(2026, 11, 8))
        times = M3.spread([c.seconds for c in kept], days, random.Random(seed))
        files = [realdump.File(f"voicemail/msg_{n:04d}.wav", "voicemail", t, "message", "audio", {"src": c.file}, length=c.seconds)
                 for n, (c, t) in enumerate(zip(kept, times, strict=True), 1)]
        questions = Q.questions_of(calls, kept, files, roles, "tiny", seed, files[-1].t + 3600)
        assert len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2, seed
        assert {q.tags["template"] for q in questions} == {"scheme", "count", "first", "single", "absent_first", "absent_single", "price", "caller"}
        plain_check(calls, kept, files, questions, "tiny")
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, ["msg_0001.wav", "msg_0002.wav", "msg_0003.wav"]):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_name_the_dataset_has_is_refused_as_absent_and_a_set_of_one_is_a_single():
    calls = synthetic()
    with pytest.raises(ValueError, match="netflix is asked about as a name no message has"):
        Q.choose([*calls, call(9000, "Your Netflix account")], [], "tiny", 1)
    with pytest.raises(ValueError, match="venmo is asked about as a name no message has"):
        Q.choose(calls, ["cease-desist-letter-venmo-payments"], "tiny", 1)
    assert [k for k in Q.SINGLES if k in Q.SCHEMES] == list(Q.SINGLES) and set(Q.SINGLES) <= set(Q.SCHEMES)


def test_a_scheme_is_drawn_only_where_it_leaves_out_no_message_another_needs():
    """Apple's near words include a MacBook: where Apple is asked the MacBook messages are out of the box, so no draw asks both."""
    calls = synthetic()
    for seed in range(1, 12):
        roles, kept = Q.choose(calls, [], "tiny", seed)
        sets = Q.sets_of([c for c in calls if c.language == "en"])
        asked = [k for k in roles.asked() if k in Q.SCHEMES]
        absent = {k for r in (roles.absent_scheme, roles.absent_count, roles.absent_first, roles.absent_single) for k in r}
        for a in asked:
            for b in asked:
                if a != b and a not in absent:
                    assert not (sets[a][1] & sets[b][0]), f"{a} leaves out the messages {b} needs"
        assert not (set(roles.scheme) | set(roles.count) | set(roles.first) | set(roles.single)) & absent


# -- the real source ----------------------------------------------------------------------------------------------------------------

def plain_calls() -> tuple[list[Call], list[str]]:
    """From `metadata.csv`, as a second reader: each recording that has one row (or several of one text), with its transcript."""
    with open(SOURCE / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["file_name"]].append(r)
    out = [call_of(name, rs[0]) for name, rs in by.items() if len({r["transcript"] for r in rs}) == 1]
    return out, sorted({r["case_details"] for r in rows})


def call_of(name: str, row: dict) -> Call:
    size = os.path.getsize(SOURCE / name)
    with open(SOURCE / name, "rb") as f:
        head = f.read(4096)
    found = chunks(head)
    rate = int.from_bytes(head[found[b"fmt "][0] + 4:found[b"fmt "][0] + 8], "little")
    return Call(name, row["language"], " ".join(row["transcript"].split()), row["case_details"], (size - found[b"data"][0]) / (2 * rate))


@GROUP
@REAL
def test_the_scenarios_reader_agrees_with_a_plain_one_of_the_labels():
    plain, cases = plain_calls()
    mine, mine_cases = M3.read_calls(str(SOURCE))
    assert sorted((c.file, c.language, c.transcript) for c in mine) == sorted((c.file, c.language, c.transcript) for c in plain)
    assert len(mine) == 1397 and sum(c.language == "en" for c in mine) == 1344 and list(mine_cases) == cases
    assert max(abs(a.seconds - b.seconds) for a, b in zip(sorted(mine, key=lambda c: c.file), sorted(plain, key=lambda c: c.file))) < 1e-9


@GROUP
@REAL
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_answer_is_the_transcripts_at_the_standard_scale(seed):
    calls, cases = plain_calls()
    files, questions = M3.plan_of(M3.recipe(scale="standard", seed=seed))
    roles, kept = Q.choose(calls, cases, "standard", seed)
    assert [c.file for c in kept] == [f.spec["src"] for f in files]
    asked = plain_check(calls, kept, files, questions, "standard")
    assert 22 <= len(questions) <= 24 and 4 <= sum(q.kind == "absent" for q in questions) <= 6, [q.qid for q in questions]
    assert 6 <= sum(q.split == "dev" for q in questions) <= 8
    assert {q.tags["template"] for q in questions} >= {"scheme", "count", "first", "single", "price", "percent", "caller", "absent_first"}
    assert len(asked) >= 12 and 1100 < len(files) < 1345 and all(c.language == "en" for c in kept)
    assert 8 * 3600 < sum(f.length for f in files) < 9.6 * 3600 and sum(os.path.getsize(SOURCE / f.spec["src"]) for f in files) < 1.2e9
    assert all(q.now.startswith("2026-11-09T09:00:00") for q in questions), "asked the morning after the last day"


@GROUP
@REAL
def test_the_standard_box_has_nine_weeks_of_messages_in_order_and_no_name_or_time_that_tells_a_scheme():
    files, questions = M3.plan_of(M3.recipe(scale="standard", seed=1))
    times = [f.t for f in files]
    assert times == sorted(times) and datetime.fromtimestamp(times[0]).date() >= date(2026, 9, 7) and datetime.fromtimestamp(times[-1]).date() <= date(2026, 11, 9)
    assert {datetime.fromtimestamp(t).weekday() for t in times} == set(range(7))
    sources = [int(re.match(r"audio-wav-16khz/(\d+)", str(f.spec["src"]))[1]) for f in files if re.match(r"audio-wav-16khz/\d+", str(f.spec["src"]))]
    assert sources != sorted(sources), "not in the source's order"
    scheme = next(q for q in questions if q.tags["template"] == "scheme" and q.qid.endswith("week"))
    days = {datetime.fromtimestamp(next(f.t for f in files if f.path.endswith(n))).date() for n in scheme.grade.items}
    assert max(days) - min(days) <= date(2026, 1, 7) - date(2026, 1, 1), "within a week"


@GROUP
@REAL
def test_another_seed_asks_other_schemes_of_another_box_and_one_seed_makes_one_dump():
    one = M3.plan_of(M3.recipe(scale="standard", seed=1))
    assert one == M3.plan_of(M3.recipe(scale="standard", seed=1)) and M3.hashes(M3.recipe(scale="standard", seed=1)) == M3.hashes(
        M3.recipe(scale="standard", seed=1))
    two = M3.plan_of(M3.recipe(scale="standard", seed=2))
    assert {q.qid for q in one[1]} != {q.qid for q in two[1]} and M3.hashes(M3.recipe(scale="standard", seed=1)) != M3.hashes(
        M3.recipe(scale="standard", seed=2))


@GROUP
@REAL
def test_the_tiny_box_is_a_few_dozen_messages_and_every_answer_is_the_transcripts():
    calls, cases = plain_calls()
    files, questions = M3.plan_of(M3.recipe(scale="tiny", seed=1))
    roles, kept = Q.choose(calls, cases, "tiny", 1)
    assert len(files) == 90 and len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}
    plain_check(calls, kept, files, questions, "tiny")
    assert all(q.now.startswith("2026-11-09T09:00:00") for q in questions)


@GROUP
@REAL
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        files, questions = M3.plan_of(M3.recipe(scale=scale, seed=1))
        others = [f.path.rsplit("/", 1)[-1] for f in files[:5]]
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("m3")
    recipe = M3.recipe(scale="tiny", seed=1)
    dump, questions = M3.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@GROUP
@REAL
def test_a_message_is_copied_with_its_samples_alone_and_its_time(built):
    out, recipe, dump, _ = built
    files, _ = M3.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] and {e.media for e in dump.entries} == {"audio"}
    rates = set()
    for e, f in zip(dump.entries, files):
        path = out / "dump" / e.path
        assert os.stat(path).st_mtime == e.t == f.t and not e.own_time
        raw = path.read_bytes()
        assert list(chunks(raw)) == [b"fmt ", b"data"] and len(raw) == e.size and b"Lavf" not in raw and b"ISFT" not in raw
        with wave.open(str(path)) as w:
            assert (w.getnchannels(), w.getsampwidth()) == (1, 2)
            rates.add(w.getframerate())
            assert w.getnframes() / w.getframerate() == e.length == f.length
        source = (SOURCE / f.spec["src"]).read_bytes()
        start, size = chunks(source)[b"data"]
        assert raw[44:] == source[start:start + size], "the samples are the source's"
    assert rates <= {8000, 16000} and not set(os.listdir(out / "dump")) - {"voicemail"}
    assert M3.probe(recipe, dump, out, None, 1).status == "passed"


@GROUP
@REAL
def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = M3.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        M3.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        M3.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        M3.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = M3.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:robocall" in deny and "in:m3_voicemail" in deny and "in:from mmsb" in deny


def test_every_template_is_filed_and_the_shelf_is_a_shelf():
    from mmsb.core.taxonomy import check

    check(M3.shelf, M3.filing)
    assert M3.shelf.area == "home" and "31 FTC cases" in M3.shelf.pile and "invented times" in M3.shelf.pile
    assert {c for c, _ in M3.filing.values()} == {"find_all", "how_many", "did", "find", "what", "who"}
    assert all(M3.filing["absent_" + t] == M3.filing[t] for t in ("scheme", "count", "first", "single"))


class Spy:
    """An agent that answers right in the `files` configuration, citing a message of the truth, and `unknown` in any other."""

    def __init__(self, questions) -> None:
        self.questions = {q.qid: q for q in questions}
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": 0, "t1": 0} for s in q.truth.evidence[:1]]
        answer = right(q)[0] if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


@GROUP
@REAL
def test_build_run_and_report_over_the_tiny_box(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "m3_voicemail", "mmsb.scenarios.m3_voicemail")
    assert world.main("build", "--scenario", "m3_voicemail", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    questions = run.questions()
    spy = Spy(questions)
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    scheme = next(q for q in questions if q.tags["template"] == "scheme")
    prompt = spy.prompts["files", scheme.qid]
    assert prompt.startswith("It is now Monday 9 November 2026, 09:00. The files of my voicemail box are in ")
    assert scheme.question in prompt and "<the file names, as in msg_0042.wav, separated by commas>" in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (8, 8, 8)
    assert (listing["correct"], listing["n"]) == (2, 8) and section["leaks"] == [], "listing alone answered only the absent ones"
