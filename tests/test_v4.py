"""V4, the work meetings: the readers and rules on made-up meetings, and, from the real AMI annotations on disk, every answer derived again here by a
plain reader (regular expressions over the raw files, never the scenario's reader), the roles checked against what the speakers say of
themselves, the leak checks, the grader taking each truth and refusing planted wrong answers, and the tiny dump built with ffmpeg. The tests that
need the annotations skip, and say so, without the source; the build skips without the videos. The questions' truths do not depend on how long a
meeting's media plays but for `long`, so the plan is made with the annotations' own lengths (`LENGTHS`), and the media's are checked on the build."""

from __future__ import annotations

import html
import re
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import av
import pytest

from mmsb.core.grade import grade
from mmsb.scenarios import realdump
from mmsb.scenarios import v4_meetings as V4
from mmsb.scenarios.v4_meetings import questions as Q
from mmsb.scenarios.v4_meetings import source as S

ZIP = S.SOURCE / S.ANNOTATIONS
needs_source = pytest.mark.skipif(not ZIP.is_file(), reason=f"the AMI annotations are not at {ZIP}, so V4 is not built")
pytestmark = pytest.mark.xdist_group("v4")


# -- made-up meetings ---------------------------------------------------------------------------------------------------------------

def meeting(id_: str, day: str, *, moves=(), topics=(), roles=None) -> S.Meeting:
    d = datetime.fromisoformat(day).date()
    return S.Meeting(id_, d, None, roles or {"A": "PM", "B": "ID", "C": "UI", "D": "ME"}, {}, tuple(topics), (), {}, tuple(moves))


def test_a_meeting_is_told_by_its_day_and_its_place_in_it():
    a, b, c = meeting("x1", "2005-03-09"), meeting("x2", "2005-03-09"), meeting("x3", "2005-03-16")
    assert Q.which_meeting(b, [a, b, c]) == "the second meeting on Wednesday 9 March 2005"
    assert Q.which_meeting(c, [a, b, c]) == "the meeting on Wednesday 16 March 2005", "alone that day"
    assert Q.which_meeting(b, [a, b]) == "the second meeting on Wednesday 9 March 2005"
    assert Q.which_meeting(b, [b, c]) == "the meeting on Wednesday 9 March 2005", "alone that day in the dump"
    assert Q.which_meeting(a, [b, a]) == "the second meeting on Wednesday 9 March 2005", "by place in the list given (the dump's order of start)"


def test_the_first_to_go_to_the_whiteboard_is_asked_only_when_it_is_clear():
    def m(*moves):
        return meeting("x", "2005-03-09", moves=[S.Move(*x) for x in moves])

    clear = m(("A", "move", 90, 100), ("A", "stand_whiteboard", 100, 160), ("B", "move", 250, 255), ("B", "stand_whiteboard", 255, 300))
    assert Q.board_first(clear).agent == "A"
    assert Q.board_first(m(("A", "stand_whiteboard", 100, 160), ("B", "stand_whiteboard", 130, 200))) is None, "two start within 45 s"
    assert Q.board_first(m(("A", "stand_whiteboard", 100, 160), ("B", "stand_whiteboard", 190, 250))) is None, "the next 90 s later: within a minute either side of the two"
    assert Q.board_first(m(("A", "move", 50, 100), ("A", "stand_whiteboard", 100, 160))) is None, "a walk of 50 s: when did they get up?"
    assert Q.board_first(m(("A", "stand_whiteboard", 100, 103))) is None, "a stand of 3 s"
    assert Q.board_first(m(("A", "sit", 0, 500))) is None


def test_presenters_come_from_the_top_level_topics_and_only_when_each_role_presents_once():
    t = lambda name, t0, top=True: S.Topic(name, t0, t0 + 50, top)  # noqa: E731
    ok = meeting("x", "2005-03-09", topics=[t("opening", 0), t("marketing expert presentation", 100), t("interface specialist presentation", 300),
                                          t("industrial designer presentation", 600), t("industrial design presentation", 650, False)])
    assert [r for r, _, _ in Q.presenters(ok)] == ["ME", "UI", "ID"]
    twice = meeting("x", "2005-03-09", topics=[t("marketing expert presentation", 100), t("marketing expert presentation", 300)])
    assert Q.presenters(twice) == []


def test_a_role_grade_takes_the_role_and_refuses_the_others_and_a_hedge():
    g = Q.role_grade("UI")
    assert grade(g, "The user interface designer.") and grade(g, "the interface designer") and not grade(g, "the industrial designer")
    assert not grade(g, "the project manager or the user interface designer") and not grade(g, "unknown")
    nxt = Q.role_grade("ME", skip="UI")
    assert grade(nxt, "After the user interface designer it was the marketing expert.") and not grade(nxt, "the industrial designer")


def test_times_of_day_that_the_corpus_lacks_are_drawn_in_order_and_the_same_every_time():
    ms = [meeting(f"x{i}", d) for i, d in enumerate(["2005-03-09"] * 4 + ["2005-03-16"] * 4)]
    seconds = {m.id: 1500.0 for m in ms}
    first = S.start_times(ms, seconds)
    assert first == S.start_times(ms, seconds)
    for day in ("2005-03-09", "2005-03-16"):
        starts = [first[m.id] for m in ms if m.day.isoformat() == day]
        assert starts == sorted(starts) and starts[0].hour in (9, 10, 11) or starts[0].hour == 11
        assert all(b - a >= timedelta(seconds=1500 + 20 * 60) for a, b in zip(starts, starts[1:], strict=False)), "after the last one ended"
    real = S.Meeting("x", ms[0].day, datetime.strptime("09:29", "%H:%M").time(), {}, {}, (), (), {}, ())
    assert S.start_times([real], {"x": 800.0})["x"] == datetime(2005, 3, 9, 9, 29)


def test_a_movement_that_leaves_a_gap_is_refused_and_a_trailing_marker_past_the_end_is_left_out():
    class Stub:
        def __init__(self, rows):
            self.rows = rows

        def xml(self, name):
            from xml.etree import ElementTree as ET
            body = "".join(f'<movement starttime="{a}" endtime="{b}" type="{t}"/>' for a, b, t in self.rows)
            return ET.fromstring(f'<root xmlns:nite="http://nite.sourceforge.net/">{body}</root>')

    ok = Stub([(0, 10, "sit"), (10, 20, "take_notes"), (20, 30, "off_camera"), (31, 29, "off_camera")])
    assert [m.kind for m in S.read_moves(ok, "x")[:3]] == ["sit", "take_notes", "off_camera"]
    assert len(S.read_moves(ok, "x")) == 12 and all(m.t1 >= m.t0 for m in S.read_moves(ok, "x"))
    with pytest.raises(ValueError, match="does not follow"):
        S.read_moves(Stub([(0, 10, "sit"), (12, 20, "take_notes")]), "x")
    with pytest.raises(ValueError, match="does not follow"):
        S.read_moves(Stub([(0, 10, "levitate")]), "x")


def test_a_never_said_word_that_is_said_as_letters_or_in_another_word_is_refused(monkeypatch):
    from mmsb.scenarios.realdump.speech import Talk, Term
    words = {"A": tuple(S.Word(float(i), i + 0.5, w, False) for i, w in enumerate(["no", "L_C_D_", "display", "wi", "fi"]))}
    m = S.Meeting("x", datetime(2005, 3, 9).date(), None, {"A": "PM"}, words, (), (), {}, ())
    t = Talk("x.mp4", datetime(2005, 3, 9), 100.0, m.cues())
    monkeypatch.setattr(Q, "TERMS", ())
    for word in ("lcd", "wifi"):
        monkeypatch.setattr(Q, "NEVER", (Term(word, word),))
        with pytest.raises(ValueError, match="is said"):
            Q.check_terms([m], [t])
    monkeypatch.setattr(Q, "NEVER", (Term("zebra", "a zebra"),))
    Q.check_terms([m], [t])


def test_a_written_text_question_takes_its_natural_answer_and_refuses_a_second_guess():
    for w in Q.WRITTEN:
        if w.grade is not None and w.grade.rule == "text":
            assert grade(w.grade, w.value) and grade(w.grade, f"I think it was {w.value}"), w.key
            assert not grade(w.grade, f"{w.value} or maybe something else") and not grade(w.grade, f"perhaps {w.value}"), w.key


# -- the real annotations, read again ---------------------------------------------------------------------------------------------

def zread(name: str) -> str:
    with zipfile.ZipFile(ZIP) as z:
        return z.read(name).decode("latin-1")


def plain_words(meeting_id: str) -> dict[str, list[tuple[float, str]]]:
    """Each speaker's spoken words with their start times, read with a regular expression."""
    out = {}
    for agent in "ABCD":
        raw = zread(f"words/{meeting_id}.{agent}.words.xml")
        out[agent] = [(float(a), html.unescape(w).casefold()) for a, w in re.findall(r'<w nite:id="[^"]*" starttime="([\d.]+)" endtime="[\d.]+"(?: trunc="true")?(?: mispronounced="true")?>([^<]*)</w>', raw)]
    return out


def plain_roles(meeting_id: str) -> dict[str, str]:
    raw = zread("corpusResources/meetings.xml")
    block = re.search(rf'observation="{meeting_id}".*?</meeting>', raw, re.S)[0]
    return {a: r for a, r in re.findall(r'nxt_agent="(\w)".*?role="(\w+)"', block)}


def plain_mentions(meeting_id: str, search: str) -> list[tuple[float, str]]:
    """(start, speaker) of each mention of the words `search`, the last with an s or es, in each speaker's own words."""
    want = re.findall(r"[a-z0-9']+", search.casefold())
    out = []
    for agent, words in plain_words(meeting_id).items():
        toks = [(t, w2) for t, w in words for w2 in re.findall(r"[a-z0-9']+", w)]
        for i in range(len(toks) - len(want) + 1):
            window = [w for _, w in toks[i:i + len(want)]]
            if window[:-1] == want[:-1] and window[-1] in (want[-1], want[-1] + "s", want[-1] + "es"):
                out.append((toks[i][0], agent))
    return sorted(out)


def plain_stands(meeting_id: str) -> list[tuple[float, float, str]]:
    stands = []
    for agent in "ABCD":
        raw = zread(f"movement/{meeting_id}.{agent}.movement.xml")
        stands += [(float(a), float(b), agent) for b, a in re.findall(r'<movement endtime="([\d.]+)" starttime="([\d.]+)" type="stand_whiteboard"', raw)]
    return sorted(stands)


LENGTHS = {m: None for m in S.MEETINGS}


@pytest.fixture(scope="module")
def plan(request):
    """The standard plan at seeds 1 to 3, made with the annotations' own lengths for the media."""
    mp = pytest.MonkeyPatch()
    for m in S.read_meetings(str(S.SOURCE)):
        LENGTHS[m.id] = max(w.t1 for words in m.words.values() for w in words) + 5.0
    mp.setattr(V4, "seconds_of", lambda source, m: LENGTHS[m])
    request.addfinalizer(mp.undo)
    return {seed: V4.plan_of(V4.recipe(scale="standard", seed=seed)) for seed in (1, 2, 3)}


def meeting_of(files, path: str) -> str:
    return next(f.spec["meeting"] for f in files if f.path == path)


def start_of(files, meeting_id: str) -> datetime:
    return datetime.fromtimestamp(next(f.t for f in files if f.spec["meeting"] == meeting_id))


def key_term(qid: str) -> Q.Term:
    """The term a speech question of this id asks of (the id's last words are its key)."""
    return next(t for t in (*Q.TERMS, *Q.NEVER) if qid.endswith("-" + "-".join(re.findall(r"[a-z0-9']+", t.search.casefold()))))


@needs_source
def test_each_speakers_role_is_the_one_they_give_when_they_introduce_themselves():
    said = {"PM": r"project manager", "ID": r"industrial designer", "UI": r"user interface", "ME": r"marketing"}
    for m in ("ES2002a", "ES2015a", "ES2016a"):
        roles, found = plain_roles(m), {}
        for agent, words in plain_words(m).items():
            head = " ".join(w for t, w in words if t < 420)
            for role, phrase in said.items():
                if re.search(rf"i(?:'m| am| will be| 'm) (?:the |your |supposed to be an? |our |uh our )?{phrase}", head):
                    found[agent] = role
        assert len(found) >= 3 and all(roles[a] == r for a, r in found.items()), f"{m}: {found} against {roles}"


@needs_source
def test_the_roles_of_a_teams_meetings_are_those_of_the_corpus_and_change_with_the_team():
    ms = {m.id: m for m in S.read_meetings(str(S.SOURCE))}
    assert ms["ES2002a"].roles["B"] == "PM" and ms["ES2002d"].roles["D"] == "UI" and ms["ES2002d"].roles["C"] == "ME"
    assert ms["ES2015a"].roles["A"] == "PM" and ms["ES2016c"].roles["B"] == "ID"
    assert {m.day.isoformat() for m in ms.values()} == {"2004-12-13", "2005-03-09", "2005-03-16"}
    assert all(sorted(m.roles.values()) == sorted(Q.ROLES) for m in ms.values())


@needs_source
def test_the_standard_plan_is_twelve_meetings_with_varied_questions_and_a_quarter_absent(plan):
    files, qs = plan[1]
    assert len(files) == 12 and 20 <= len(qs) <= 26
    cats = Counter(V4.filing[q.tags["template"]][0] for q in qs)
    assert len(cats) >= 6 and {"what", "who", "order", "did", "find", "when", "how_many"} <= set(cats)
    assert 4 <= sum(q.kind == "absent" for q in qs) <= 7
    dev = sum(q.split == "dev" for q in qs)
    assert round(0.3 * len(qs)) == dev and {q.split for q in qs} == {"dev", "test"}
    assert all(q.now.startswith("2005-03-17T09:00:00") for q in qs)
    assert list(files) == sorted(files, key=lambda f: f.t) and all(f.path.startswith("Meetings/meeting_2") for f in files)
    for seed in (2, 3):
        assert plan[seed][0] == files and {q.qid for q in plan[seed][1]} != {q.qid for q in qs}


@needs_source
def test_every_speech_answer_is_the_words_read_again(plan):
    for seed in (1, 2, 3):
        files, qs = plan[seed]
        by_day = sorted(((meeting_of(files, f.path), f.t) for f in files), key=lambda x: x[1])
        for q in (q for q in qs if q.tags["template"] in ("first", "did", "which", "all", "count", "long", "absent_first", "absent_did", "absent_which")):
            term = key_term(q.qid)
            found = {m: plain_mentions(m, term.search) for m, _ in by_day}
            t = q.tags["template"]
            if q.kind == "absent":
                assert not any(found.values()), f"{q.qid}: said after all"
                assert not any(re.sub(r"[^a-z0-9]", "", term.search) in re.sub(r"[^a-z0-9]", "", " ".join(w for _, w in ws))
                               for m, _ in by_day for ws in plain_words(m).values()), f"{q.qid}: inside a word"
            elif t in ("first", "did"):
                m = next(m for m, _ in by_day if found[m])
                wall = start_of(files, m) + timedelta(seconds=found[m][0][0])
                at = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}")
                assert abs((wall - at).total_seconds()) <= 15 and q.grade.date == wall.date().isoformat(), q.qid
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
def test_every_written_and_derived_answer_is_the_annotations_read_again(plan):
    for seed in (1, 2, 3):
        files, qs = plan[seed]
        for q in qs:
            t = q.tags["template"]
            if t in ("whofirst", "firstof"):
                _, m, *words = q.qid.split("-")
                if t == "whofirst":
                    first = plain_mentions(m, key_term(q.qid).search)[0]
                    assert q.grade.truth == Q.ACCEPT[plain_roles(m)[first[1]]], q.qid
                else:
                    a, b = (plain_mentions(m, w)[0][0] for w in words)
                    assert q.grade.truth.startswith(words[0] if a < b else words[1]), q.qid
            elif t in ("board_when", "board_who", "board_count"):
                m = q.qid.split("-")[1]
                stands = plain_stands(m)
                assert stands, q.qid
                if t == "board_count":
                    assert q.grade.truth == str(len({a for _, _, a in stands}))
                elif t == "board_who":
                    assert q.grade.truth == Q.ACCEPT[plain_roles(m)[stands[0][2]]]
                else:
                    wall = start_of(files, m) + timedelta(seconds=stands[0][0])
                    assert (q.grade.date, q.grade.time) == (wall.date().isoformat(), wall.strftime("%H:%M:%S")), q.qid
            elif t == "absent_board_when":
                m = q.qid.split("-", 3)[3]
                assert not plain_stands(m) and not any(re.search(r'type="(?:move|stand_screen)"', zread(f"movement/{m}.{a}.movement.xml")) for a in "ABCD")
            elif t in ("what", "who", "decided"):
                w = next(w for w in Q.WRITTEN if q.qid == f"{t}-{w.key}")
                found = plain_mentions(w.meeting, w.check)
                assert found, f"{w.key}: {w.check!r} is not said in {w.meeting}"
                if t == "decided":
                    wall = start_of(files, w.meeting)
                    at = datetime.fromisoformat(f"{q.grade.date}T{q.grade.time}")
                    assert any(abs((wall + timedelta(seconds=s) - at).total_seconds()) <= 30 for s, _ in found), f"{w.key}: no mention near its time"
            elif t == "next":
                _, m, before, after = q.qid.split("-")
                raw = zread(f"topics/{m}.topic.xml")
                assert re.search(r"scenario_topic_type|other_description", raw)


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
            day = datetime.fromisoformat(g.date)
            t = datetime.fromisoformat(f"{g.date}T{g.time}")
            assert grade(g, f"{t:%Y-%m-%d %H:%M:%S}") and not grade(g, f"{t + timedelta(minutes=3):%Y-%m-%d %H:%M}")
            assert not grade(g, f"{day + timedelta(days=1):%Y-%m-%d} {t:%H:%M}"), "another day"
        elif g.rule == "set":
            assert grade(g, ", ".join(g.items)) and not grade(g, "meeting_19990101_000000.mp4")
        elif g.rule == "count":
            assert grade(g, g.truth) and not grade(g, str(int(g.truth) + 1))
        elif g.rule == "amount":
            assert grade(g, g.truth) and not grade(g, str(float(g.truth) + 5))
        else:
            assert g.rule == "text"
            assert grade(g, g.truth.split("|")[0]) and not any(grade(g, w) for w in g.wrong[:1])


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
    recipe = V4.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        V4.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        V4.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        V4.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code(monkeypatch):
    monkeypatch.setenv(realdump.SOURCES_ENV, str(Path("D:/bench-data") / "sources"))
    deny = V4.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:es2015" in deny and "in:v4_meetings" in deny and "in:from mmsb" in deny


# -- the build ---------------------------------------------------------------------------------------------------------------------

def signals(ids) -> bool:
    return all((S.SOURCE / "signals" / m / "video" / f"{m}.Corner.avi").is_file() and (S.SOURCE / "signals" / m / "audio" / f"{m}.Mix-Headset.wav").is_file()
               for m in ids)


@needs_source
def test_the_tiny_dump_builds_plays_for_as_long_as_the_manifest_says_and_passes_the_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(V4, "seconds_of", S.seconds_of)
    if not signals(V4.SCALES["tiny"]):
        pytest.skip(f"the media of {', '.join(V4.SCALES['tiny'])} are not on disk yet")
    recipe = V4.recipe(scale="tiny", seed=1)
    dump, qs = V4.build(recipe, tmp_path / "dump", 2)
    assert len(dump.entries) == 2 and 2 <= len(qs) <= 8 and len({q.qid for q in qs}) == len(qs)
    for e in dump.entries:
        with av.open(str(dump.root / e.path)) as c:
            assert [s.type for s in c.streams] == ["video", "audio"] and abs(c.duration / 1e6 - e.length) < 0.5
        assert abs((dump.root / e.path).stat().st_mtime - e.t) < 1
    assert V4.probe(recipe, dump, tmp_path / "probe", Path(), 1).status == "passed"
