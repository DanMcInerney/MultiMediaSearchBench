"""A3, the research group's meetings: the readers and rules on made-up transcripts, and, from the real ICSI files on disk, every answer derived again here
by a plain reader (regular expressions over the raw transcript files, never the scenario's reader), the grader taking each truth and refusing planted
wrong answers, the leak checks, and the tiny dump built. The tests that need the source skip, and say so, without it."""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import av
import pytest

from mmsb.core.grade import grade
from mmsb.scenarios import a3_meetings as A3
from mmsb.scenarios import realdump
from mmsb.scenarios.a3_meetings import questions as Q
from mmsb.scenarios.a3_meetings import source as S
from mmsb.scenarios.realdump.speech import Talk, Term

ZIP = S.SOURCE / S.TRANSCRIPTS
needs_source = pytest.mark.skipif(not (ZIP.is_file() and (S.SOURCE / S.NXT).is_file()), reason=f"the ICSI annotations are not at {S.SOURCE}, so A3 is not built")
pytestmark = pytest.mark.xdist_group("a3")


# -- made-up transcripts -----------------------------------------------------------------------------------------------------------

def seg(t0, t1, text, who="x", loose=None):
    return S.Segment(t0, t1, who, text, loose if loose is not None else text)


def meeting(segments, topics=(), id_="Bmr900", start=datetime(2000, 10, 26, 14, 40)):
    return S.Meeting(id_, start, tuple(segments), tuple(topics))


def test_a_long_segment_is_cut_into_cues_with_times_by_word_count_and_a_short_one_is_kept():
    words = [f"w{i}" for i in range(60)]
    m = meeting([seg(100.0, 130.0, " ".join(words)), seg(140.0, 142.0, "short one")])
    cues = m.cues()
    assert [c.text.split()[0] for c in cues] == ["w0", "w25", "w50", "short"]
    assert [round(c.t0, 1) for c in cues[:3]] == [100.0, 112.5, 125.0] and cues[2].t1 == 130.0 and cues[3].t1 == 142.0


def test_the_digit_strings_are_the_segments_marked_digits_and_only_when_they_lie_in_one_stretch():
    def digits(t0, t1):
        return S.Segment(t0, t1, "x", "one two", "one two", True)

    one = meeting([digits(205, 210), seg(220, 225, "chat"), digits(240, 245), digits(300, 305)])
    assert Q.digit_strings(one) == [(205, 210), (240, 245), (300, 305)]
    assert Q.digit_strings(meeting([digits(205, 210), digits(900, 905)])) == [], "digits at the start and at the end: two tasks"
    assert Q.digit_strings(meeting([seg(10, 14, "seven four zero one nine")])) == [], "a string of numbers no transcriber marked is no digit task"


def test_a_term_read_differently_by_the_cues_and_by_the_guesses_is_refused(monkeypatch):
    plain = meeting([seg(0, 5, "the scroll wheel is nice", who="a"), seg(6, 9, "the scroll wheel", who="a", loose="the scroll wheel")])
    guess = meeting([seg(0, 5, "the scroll is nice", who="a", loose="the scroll wheel is nice")])
    for m, ok in ((plain, True), (guess, False)):
        t = Talk("x.wav", m.start, 10.0, m.cues())
        monkeypatch.setattr(Q, "TERMS", (Term("scroll wheel", "a scroll wheel"),))
        monkeypatch.setattr(Q, "NEVER", ())
        if ok:
            Q.check_terms([m], [t])
        else:
            with pytest.raises(ValueError, match="read differently"):
                Q.check_terms([m], [t])


def test_a_never_said_term_said_as_letters_or_in_a_guess_is_refused(monkeypatch):
    monkeypatch.setattr(Q, "TERMS", ())
    for said in ("we use P_Z_M_ mikes", "a pizza"):
        m = meeting([seg(0, 5, "nothing here", loose=said)])
        t = Talk("x.wav", m.start, 10.0, m.cues())
        monkeypatch.setattr(Q, "NEVER", (Term("pzm" if "P_Z" in said else "pizza", "x"),))
        with pytest.raises(ValueError, match="is said"):
            Q.check_terms([m], [t])
    monkeypatch.setattr(Q, "NEVER", (Term("zebra", "a zebra"),))
    Q.check_terms([m], [t])


def test_a_topic_file_with_few_broken_references_is_read_and_with_many_is_refused():
    def archive(broken: int) -> zipfile.ZipFile:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            segs = "".join(f'<segment nite:id="Bmr900.segment.{i}" starttime="{i}.0" endtime="{i}.5" participant="p"/>' for i in range(1, 301))
            z.writestr("ICSIplus/Segments/Bmr900.A.segs.xml", f'<nite:root xmlns:nite="http://nite.sourceforge.net/">{segs}</nite:root>')
            kids = "".join(f'<nite:child href="Bmr900.A.segs.xml#id(Bmr900.segment.{i if i > broken else 9999 + i})"/>' for i in range(1, 301))
            z.writestr("ICSIplus/Contributions/TopicSegmentation/Bmr900.topic.xml",
                       f'<nite:root xmlns:nite="http://nite.sourceforge.net/"><topic nite:id="t" description="opening">{kids}</topic></nite:root>')
        return zipfile.ZipFile(io.BytesIO(buf.getvalue()))

    (topic,) = S.read_topics(archive(2), "Bmr900")
    assert (topic.name, topic.t0, topic.t1) == ("opening", 3.0, 300.5), "the two broken references left out"
    with pytest.raises(ValueError, match="point at no segment"):
        S.read_topics(archive(50), "Bmr900")


def test_a_written_text_question_takes_its_natural_answer_and_refuses_a_second_guess():
    for w in Q.WRITTEN:
        if w.grade is not None and w.grade.rule == "text":
            assert grade(w.grade, w.value) and grade(w.grade, f"I think it was {w.value}"), w.key
            assert not grade(w.grade, f"{w.value} or maybe something else") and not grade(w.grade, f"perhaps {w.value}"), w.key


# -- the real transcripts, read again ---------------------------------------------------------------------------------------------

def mrt(meeting_id: str) -> str:
    with zipfile.ZipFile(ZIP) as z:
        return z.read(f"transcripts/{meeting_id}.mrt").decode("latin-1")


def plain_segments(meeting_id: str) -> list[tuple[float, float, str, list[str]]]:
    """(start, end, speaker, tokens) of each segment, the transcribers' marks that are no words taken out and the guesses kept, by regular expression."""
    out = []
    for a, b, who, body in re.findall(r'<Segment StartTime="([\d.]+)" EndTime="([\d.]+)" Participant="(\w+)"[^>]*>(.*?)</Segment>', mrt(meeting_id), re.S):
        body = re.sub(r"<(?:Comment|VocalSound|NonVocalSound|Pause|Gap)\b[^>]*/>|<(?:Comment|VocalSound|NonVocalSound)\b[^>]*>.*?</(?:Comment|VocalSound|NonVocalSound)>", " ", body, flags=re.S)
        body = re.sub(r"<[^>]+>", " ", body)
        out.append((float(a), float(b), who, re.findall(r"[a-z0-9']+", body.casefold())))
    return out


def plain_digits(meeting_id: str) -> list[tuple[float, float]]:
    """(start, end) of each segment with the transcribers' `Digits` mark."""
    return [(float(a), float(b)) for a, b, body in re.findall(r'<Segment StartTime="([\d.]+)" EndTime="([\d.]+)" Participant="\w+"[^>]*>(.*?)</Segment>', mrt(meeting_id), re.S)
            if 'Description="Digits"' in body]


def plain_mentions(meeting_id: str, search: str) -> list[float]:
    """The start of the segment of each mention of `search` (the last word with an s or es), in each speaker's own words."""
    want = re.findall(r"[a-z0-9']+", search.casefold())
    by: dict[str, list[tuple[float, str]]] = {}
    for a, _, who, toks in plain_segments(meeting_id):
        by.setdefault(who, []).extend((a, t) for t in toks)
    out = []
    for toks in by.values():
        for i in range(len(toks) - len(want) + 1):
            window = [t for _, t in toks[i:i + len(want)]]
            if window[:-1] == want[:-1] and window[-1] in (want[-1], want[-1] + "s", want[-1] + "es"):
                out.append(toks[i][0])
    return sorted(out)


def plain_stamp(meeting_id: str) -> datetime:
    return datetime.strptime(re.search(r'DateTimeStamp="([\d-]+)"', mrt(meeting_id))[1], "%Y-%m-%d-%H%M")


LENGTHS: dict[str, float] = {}


@pytest.fixture(scope="module")
def plan(request):
    mp = pytest.MonkeyPatch()
    for m in S.MEETINGS:
        LENGTHS[m] = max(b for _, b, *_ in plain_segments(m)) + 0.5
    mp.setattr(A3, "seconds_of", lambda source, m: LENGTHS[m])
    request.addfinalizer(mp.undo)
    return {seed: A3.plan_of(A3.recipe(scale="standard", seed=seed)) for seed in (1, 2, 3)}


def meeting_of(files, path: str) -> str:
    return next(f.spec["meeting"] for f in files if f.path == path)


def key_term(qid: str) -> Term:
    return next(t for t in (*Q.TERMS, *Q.NEVER) if qid.endswith("-" + "-".join(re.findall(r"[a-z0-9']+", t.search.casefold()))))


@needs_source
def test_the_standard_plan_is_ten_meetings_on_their_real_days_with_varied_questions(plan):
    files, qs = plan[1]
    assert len(files) == 10 and 20 <= len(qs) <= 26
    assert [f.path for f in files][:3] == ["Weekly/rec_20000202_1700.wav", "Weekly/rec_20000216_1600.wav", "Weekly/rec_20000308_1530.wav"]
    for f in files:
        assert datetime.fromtimestamp(f.t) == plain_stamp(f.spec["meeting"]) and f.path.endswith(f"{plain_stamp(f.spec['meeting']):%Y%m%d_%H%M}.wav")
    cats = Counter(A3.filing[q.tags["template"]][0] for q in qs)
    assert len(cats) >= 6 and {"what", "order", "did", "find", "when", "how_many", "find_all"} <= set(cats) and "who" not in cats
    assert 4 <= sum(q.kind == "absent" for q in qs) <= 7
    assert round(0.3 * len(qs)) == sum(q.split == "dev" for q in qs)
    assert all(q.now.startswith("2001-03-30T09:00:00") for q in qs)
    for seed in (2, 3):
        assert plan[seed][0] == files and {q.qid for q in plan[seed][1]} != {q.qid for q in qs}


@needs_source
def test_no_question_names_a_person_or_asks_who(plan):
    names = re.compile(r"\b(?:jose|jane|adam|javier|thilo|dan|morgan|nelson|eric|chuck|brian|steve|susan|jerry|susanne)\b", re.I)
    for seed in (1, 2, 3):
        for q in plan[seed][1]:
            assert not names.search(q.question) and not q.question.lower().startswith("who"), q.question
            assert not any(names.search(w) for w in (q.grade.truth or "", *q.grade.wrong)), q.qid


@needs_source
def test_every_speech_answer_is_the_transcripts_read_again(plan):
    for seed in (1, 2, 3):
        files, qs = plan[seed]
        order = [f.spec["meeting"] for f in files]
        for q in (q for q in qs if q.tags["template"] in ("first", "did", "which", "all", "count", "long", "absent_first", "absent_did", "absent_which", "order")):
            t = q.tags["template"]
            if t == "order":
                first, second = re.search(r"bring up (.+?) before or after (.+?)\?", q.question).groups()
                a, b = (next(x for x in Q.TERMS if x.said == w) for w in (first, second))
                day = next(m for m in order if realdump.date_text(plain_stamp(m).date()) in q.question)
                ta, tb = plain_mentions(day, a.search), plain_mentions(day, b.search)
                assert len(ta) == len(tb) == 1 and q.grade.truth == ("before" if ta[0] < tb[0] else "after"), q.qid
                continue
            term = key_term(q.qid)
            found = {m: plain_mentions(m, term.search) for m in order}
            if q.kind == "absent":
                text = re.sub(r"[^a-z0-9]", "", " ".join(w for m in order for *_, toks in plain_segments(m) for w in toks))
                assert not any(found.values()) and re.sub(r"[^a-z0-9]", "", term.search) not in text, f"{q.qid}: said after all"
            elif t in ("first", "did"):
                m = next(m for m in order if found[m])
                wall = plain_stamp(m) + timedelta(seconds=found[m][0])
                at = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}")
                assert abs((wall - at).total_seconds()) <= 28 and q.grade.date == wall.date().isoformat(), q.qid
            else:
                have = sorted(Path(f.path).name for f in files if found[f.spec["meeting"]])
                if t in ("which", "all"):
                    assert list(q.grade.items) == have, q.qid
                elif t == "count":
                    assert q.grade.truth == str(len(have)), q.qid
                else:
                    (f,) = [f for f in files if found[f.spec["meeting"]]]
                    assert len(have) == 1 and float(q.grade.truth) == round(f.length / 60, 1), q.qid


@needs_source
def test_the_written_and_digit_answers_are_the_transcripts_read_again(plan):
    for seed in (1, 2, 3):
        files, qs = plan[seed]
        for q in qs:
            t = q.tags["template"]
            if t in ("what", "before"):
                w = next(w for w in Q.WRITTEN if q.qid == f"{t}-{w.key}")
                near = [x for x in plain_segments(w.meeting) if w.t0 - 5 <= x[0] <= w.t1 + 5]
                assert re.search(r"\b" + r"\W+".join(re.findall(r"[a-z]+", w.check)) + r"\b", " ".join(" ".join(x[3]) for x in near)), f"{w.key}: {w.check!r} is not said there"
            elif t in ("digits", "digits_long"):
                m = q.qid.split("-")[1]
                digits = plain_digits(m)
                assert len(digits) >= 20 and all(b[0] - a[1] <= 150 for a, b in zip(digits, digits[1:], strict=False)), "read in one stretch"
                if t == "digits":
                    wall = plain_stamp(m) + timedelta(seconds=digits[0][0])
                    assert (q.grade.date, q.grade.time) == (wall.date().isoformat(), wall.strftime("%H:%M:%S")), q.qid
                else:
                    assert abs(float(q.grade.truth) - (digits[-1][1] - digits[0][0]) / 60) < 0.06, q.qid


@needs_source
def test_no_question_asks_of_what_a_written_question_says_or_answers(plan):
    for seed in (1, 2, 3):
        qs = plan[seed][1]
        for w in Q.WRITTEN:
            mine = {q.qid for q in qs if q.qid == f"{w.template}-{w.key}"}
            for q in qs:
                assert q.qid in mine or w.check.casefold() not in q.question.casefold() or not mine, f"{q.qid} says what {w.key} asks"


@needs_source
def test_the_grader_takes_each_truth_and_refuses_planted_wrong_answers(plan):
    for q in plan[1][1]:
        g = q.grade
        if g.rule == "absent":
            assert grade(g, "No, never.") and not grade(g, "Yes, at 10:15.")
        elif g.rule == "time":
            t = datetime.fromisoformat(f"{g.date}T{g.time}")
            assert grade(g, f"{t:%Y-%m-%d %H:%M:%S}") and not grade(g, f"{t + timedelta(minutes=3):%Y-%m-%d %H:%M}")
        elif g.rule == "set":
            assert grade(g, ", ".join(g.items)) and not grade(g, "rec_19990101_0000.wav")
        elif g.rule == "count":
            assert grade(g, g.truth) and not grade(g, str(int(g.truth) + 1))
        elif g.rule == "amount":
            assert grade(g, g.truth) and not grade(g, str(float(g.truth) + 5))
        else:
            assert g.rule == "text" and grade(g, g.truth.split("|")[0])
            if g.rule == "text" and g.truth in ("before", "after"):
                assert not grade(g, g.wrong[0])


@needs_source
def test_no_time_answer_is_a_files_start_or_end_so_the_list_of_the_files_cannot_give_it(plan):
    files, qs = plan[1]
    marks = [f.t for f in files] + [f.t + f.length for f in files]
    for q in qs:
        if q.grade.rule == "time":
            at = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}").timestamp()
            assert min(abs(at - t) for t in marks) > 60, f"{q.qid}: the list of the files answers it"


@needs_source
def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = A3.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        A3.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        A3.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        A3.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code(monkeypatch):
    monkeypatch.setenv(realdump.SOURCES_ENV, str(Path("D:/bench-data") / "sources"))
    deny = A3.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:bmr0" in deny and "in:a3_meetings" in deny and "in:from mmsb" in deny


@needs_source
def test_the_tiny_dump_builds_with_the_wavs_as_they_are_and_passes_the_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(A3, "seconds_of", S.seconds_of)
    recipe = A3.recipe(scale="tiny", seed=1)
    dump, qs = A3.build(recipe, tmp_path / "dump", 2)
    assert len(dump.entries) == 2 and 2 <= len(qs) <= 8 and len({q.qid for q in qs}) == len(qs)
    for e in dump.entries:
        with av.open(str(dump.root / e.path)) as c:
            assert [s.type for s in c.streams] == ["audio"] and abs(c.duration / 1e6 - e.length) < 0.1
        assert abs((dump.root / e.path).stat().st_mtime - e.t) < 1
    assert A3.probe(recipe, dump, tmp_path / "probe", Path(), 1).status == "passed"
