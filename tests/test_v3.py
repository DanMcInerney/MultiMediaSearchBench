"""V3, the football match clips, from its real source on disk: both scales planned, every answer derived again here by a plain reader of the label
files (never the scenario's), the grader taking each truth and refusing planted wrong answers, the tiny pile built (clips copied as they are, with
their times) and its leak probe. A made-up pile checks the stretches an absent question asks about (every window has its clip) and that a derived
class is never read. The real-source tests skip, and say so, without the source."""

from __future__ import annotations

import functools
import json
import os
import re
from datetime import date, datetime, time, timedelta

import av
import pytest

from mmsb.core import grade, taxonomy
from mmsb.core.question import load_questions, write_questions
from mmsb.scenarios import realdump
from mmsb.scenarios import v3_match as V3
from mmsb.scenarios.v3_match import questions as Q
from mmsb.scenarios.v3_match.source import CLIPS, LABELS, SOURCE, Clip, Event, gaps, matches_on_disk

pytestmark = pytest.mark.xdist_group("v3")
HAVE = (SOURCE / LABELS).is_dir() and (SOURCE / CLIPS).is_dir() and len(matches_on_disk(str(SOURCE))) == 3
needs_source = pytest.mark.skipif(not HAVE, reason=f"the follow-cam source with three matches is not at {SOURCE}, so the real V3 is not built")
DAYS = {"117092": ("2025-03-08", time(18, 30)), "117093": ("2025-03-29", time(11, 0)), "118575": ("2025-04-26", time(14, 0))}  # written here
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


@functools.cache
def plain() -> list[dict]:
    """The events of the labels of the matches on disk, as a second reader: the ones a person labelled (their `source` says `soccertrack`), each with
    its wall time under the made-up match days and kick-offs."""
    out = []
    for name in sorted(os.listdir(SOURCE / LABELS)):
        m = re.fullmatch(r"st(\d+)_h([12])_c(\d+)\.Labels-ball\.json", name)
        if not m or m[1] not in DAYS or not (SOURCE / CLIPS / name.replace(".Labels-ball.json", ".mp4")).is_file():
            continue
        data = json.loads((SOURCE / LABELS / name).read_text(encoding="utf-8"))
        day, kickoff = DAYS[m[1]]
        start = datetime.combine(date.fromisoformat(day), kickoff) + timedelta(seconds=(65 * 60 if m[2] == "2" else 0) + data["pano_start_frame"] / 25)
        out.append({"match": m[1], "half": int(m[2]), "start": start, "name": f"FC_{start:%H%M%S}.mp4", "day": day, "window": data["pano_start_frame"] // 750,
                    "events": [(a["label"], int(a["position"]) / 1000, a["team"]) for a in data["annotations"] if a["source"].startswith("soccertrack:")]})
    return out


def events(label: str, match: str | None = None, half: int | None = None) -> list[tuple[datetime, dict, tuple]]:
    return sorted(((c["start"] + timedelta(seconds=e[1]), c, e) for c in plain() for e in c["events"]
                   if e[0] == label and match in (None, c["match"]) and half in (None, c["half"])), key=lambda x: x[0])


def day_of(text: str) -> str:
    """`8 March` of a question's text, as the date of the first match day that has it."""
    d, month = text.split()
    return next(day for day, _ in DAYS.values() if date.fromisoformat(day).day == int(d) and MONTHS[date.fromisoformat(day).month - 1] == month)


def match_of(text: str) -> str:
    day = day_of(re.search(r"(\d+ [A-Z][a-z]+)", text)[1])
    return next(m for m, (d, _) in DAYS.items() if d == day)


def check_against_labels(questions) -> None:
    """Each question's truth as the plain reader finds it."""
    by_name = {(c["day"], c["name"]): c for c in plain()}
    for q in questions:
        g, parts = q.grade, q.qid.removeprefix("absent-").split("-")
        template = parts[0]
        if template == "when" and parts[1] == "goal":
            _, _, match, n = parts
            t, c, e = events("goal", match)[int(n) - 1]
            assert (g.date, g.time) == (t.date().isoformat(), f"{t:%H:%M:%S}") and g.tol_s == 10.0, q.qid
        elif template == "find":
            _, _, match, n = parts
            t, c, e = events("goal", match)[int(n) - 1]
            assert g.items == (c["name"],), q.qid
        elif template == "side":
            _, _, match, n = parts
            t, c, e = events("goal", match)[int(n) - 1]
            assert g.truth == ("right|right hand" if e[2] == "left" else "left|left hand"), q.qid
        elif template == "when" and parts[1] == "shot":
            _, _, match, half, ordinal = parts
            shots = events("shot", match, int(half))
            t = shots[0][0] if ordinal == "first" else shots[-1][0]
            assert (g.date, g.time) == (t.date().isoformat(), f"{t:%H:%M:%S}"), q.qid
        elif template == "goals":
            assert g.truth == str(len(events("goal", parts[1]))), q.qid
        elif template == "shots":
            assert g.truth == str(len(events("shot", parts[1], int(parts[2])))), q.qid
        elif template == "every":
            if parts[1] == "goal":
                found = events("goal", parts[2] if len(parts) > 2 else None)
            else:
                found = events("shot", parts[2], int(parts[3]))
            assert sorted(g.items) == sorted({c["name"] for _, c, _ in found}), q.qid
        elif template in ("when_clip", "before"):
            clock = re.search(r"from (\d\d:\d\d:\d\d) on (\d+ [A-Z][a-z]+)", q.question)
            c = by_name[(day_of(clock[2]), f"FC_{clock[1].replace(':', '')}.mp4")]
            shots = [e for e in c["events"] if e[0] == "shot"]
            assert len(shots) == 1 and not any(e[0] == "goal" for e in c["events"]), q.qid
            if template == "when_clip":
                assert g.time == f"{(c['start'] + timedelta(seconds=shots[0][1])):%H:%M:%S}" and g.tol_s == 5.0 and g.date is None
            else:
                tackles = [e for e in c["events"] if e[0] == "tackle"]
                assert len(tackles) == 1 and abs(tackles[0][1] - shots[0][1]) >= 3
                assert g.truth.startswith("before") == (tackles[0][1] < shots[0][1]), q.qid
        elif template in ("did", "window"):
            w = re.search(r"between (\d\d):(\d\d) and (\d\d):(\d\d) on (\d+ [A-Z][a-z]+)", q.question)
            day = day_of(w[5])
            t0 = datetime.combine(date.fromisoformat(day), time(int(w[1]), int(w[2])))
            t1 = datetime.combine(date.fromisoformat(day), time(int(w[3]), int(w[4])))
            match = next(m for m, (d, _) in DAYS.items() if d == day)
            inside = [c for c in plain() if c["match"] == match and t0 <= c["start"] < t1]
            assert len(inside) == (t1 - t0).total_seconds() / 30, f"{q.qid}: a clip is missing in the stretch"  # every window has its clip
            goals = [e for c in inside for e in c["events"] if e[0] == "goal"]
            shots = [e for c in inside for e in c["events"] if e[0] == "shot"]
            if "goal" in "-".join(parts):
                found = goals
            else:
                found = shots
                assert not goals, q.qid
            if q.grade.rule == "absent":
                assert not found, q.qid
            elif template == "window":
                assert g.truth == str(len(found)), q.qid
            else:
                assert len(found) == 1, q.qid
        else:
            raise AssertionError(f"no reader for {q.qid}")
        assert (q.kind == "absent") == (g.rule == "absent") and (g.rule == "absent") == (not q.truth.evidence)


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "absent": lambda: "unknown",
            "time": lambda: f"{g.date or ''} {g.time}".strip(), "text": lambda: g.truth.split("|")[0]}[g.rule]()


def wrong(q) -> list[str]:
    g = q.grade
    if g.rule == "set":
        return [", ".join([*g.items, "FC_000000.mp4"]), ", ".join([*g.items[:-1], "FC_000000.mp4"]), "unknown"]
    if g.rule == "count":
        return [str(int(g.truth) + 1), "unknown"]
    if g.rule == "time":
        t = datetime.fromisoformat(f"{g.date or '2025-03-08'}T{g.time}")
        return [f"{g.date or '2025-03-08'} {t + timedelta(seconds=40):%H:%M:%S}", "unknown"]
    if g.rule == "text":
        return [g.wrong[0], "unknown"]
    return ["2025-03-08 18:50:00"]


# -- a made-up pile ------------------------------------------------------------------------------------------------------------------

def made_up():
    """Two halves of a made-up match: windows 0 to 19 of the first half but 9 (missing), a goal in window 3 and a shot in window 12, a tackle before a
    shot in window 15; a `pass_received` is never in a clip's events."""
    def clip(w: int, *events: Event) -> Clip:
        return Clip(f"st1_h1_c{w:03d}.mp4", "1", 1, w, tuple(events))
    clips = [clip(w, *{3: (Event("goal", 4.0, "right"),), 12: (Event("shot", 9.0, "left"),),
                       15: (Event("tackle", 3.0, "left"), Event("shot", 12.0, "right"))}.get(w, (Event("pass", 1.0, "left"),))) for w in range(20) if w != 9]
    files = [realdump.File(f"Football/2025-03-08/FC_{18 * 3600 + 1800 + 30 * w:06d}.mp4", "followcam", datetime(2025, 3, 8, 18, 30).timestamp() + 30 * w,
                           "clip", "video", {"src": c.name}, length=30.0) for w, c in zip((c.window for c in clips), clips, strict=True)]
    return clips, files


def test_a_stretch_of_the_day_is_asked_of_only_where_every_window_has_its_clip():
    clips, files = made_up()
    assert gaps(tuple(clips)) == {("1", 1, 9)}
    pool = Q.pool_of(clips, files)
    saved = Q.WINDOWS
    Q.WINDOWS = {"tiny": {"goal": 3, "shot": 2}, "standard": {"goal": 3, "shot": 2}}
    try:
        stretches = Q.stretches(pool, "tiny")
    finally:
        Q.WINDOWS = saved
    t0 = files[0].t
    for found in stretches.values():
        for _, _, a, b in found:
            assert not (a <= t0 + 9 * 30 < b), "a stretch that spans the missing window"
            assert datetime.fromtimestamp(a).second == 0
    assert stretches["shot"] and all(b - a == 120 for _, _, a, b in stretches["shot"]) and all(b - a == 180 for _, _, a, b in stretches["goal"])


def test_a_goal_is_asked_for_by_its_place_in_the_match_and_the_side_it_went_into():
    clips, files = made_up()
    pool = Q.pool_of(clips, files)
    library = Q.pool_library(pool, "tiny", datetime(2025, 3, 9, 9).timestamp(), files[0].t, files[-1].t + 30)
    (when,) = [q for q, _ in library["when"] if q.qid == "when-goal-1-1"]
    assert when.grade.time == "18:31:34" and when.grade.date == "2025-03-08" and when.question == "When was the first goal of the match on 8 March?"
    (side,) = [q for q, _ in library["side"]]
    assert side.grade.truth == "left|left hand", "the right team's goal is in the left-hand goal"
    (before,) = [q for q, _ in library["before"]]
    assert before.grade.truth == "before|earlier" and "18:37:30" in before.question


def test_only_the_classes_a_person_labelled_are_read(tmp_path):
    root = tmp_path / "src"
    (root / LABELS).mkdir(parents=True)
    (root / CLIPS).mkdir()
    (root / CLIPS / "st9_h1_c000.mp4").write_bytes(b"")
    body = {"match": "9", "half": "1", "clip": "st9_h1_c000.mp4", "fps": 25, "num_frames": 750, "pano_start_frame": 0, "annotations": [
        {"gameTime": "1 - 00:01", "label": "pass_received", "position": "1000", "team": "left", "visibility": "visible", "score": 0.8, "source": "derived:x"},
        {"gameTime": "1 - 00:02", "label": "shot", "position": "2000", "team": "right", "visibility": "visible", "score": 0.9, "source": "soccertrack:SHOT"}]}
    (root / LABELS / "st9_h1_c000.Labels-ball.json").write_text(json.dumps(body))
    (clip,) = V3.read_clips(str(root))
    assert [e.label for e in clip.events] == ["shot"]
    from mmsb.scenarios.v3_match import source
    source.read_clips.cache_clear()
    body["annotations"][0]["source"] = "soccertrack:oops"
    (root / LABELS / "st9_h1_c000.Labels-ball.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="does not know"):
        V3.read_clips(str(root))
    source.read_clips.cache_clear()


def test_a_label_file_without_its_clip_is_refused_and_so_is_a_start_that_is_no_whole_window(tmp_path):
    from mmsb.scenarios.v3_match import source
    root = tmp_path / "src"
    (root / LABELS).mkdir(parents=True)
    (root / CLIPS).mkdir()
    (root / CLIPS / "st9_h1_c000.mp4").write_bytes(b"")
    body = {"match": "9", "half": "1", "clip": "st9_h1_c001.mp4", "fps": 25, "num_frames": 750, "pano_start_frame": 750, "annotations": []}
    (root / LABELS / "st9_h1_c001.Labels-ball.json").write_text(json.dumps(body))
    with pytest.raises(FileNotFoundError, match="not on disk"):
        source.read_clips.__wrapped__(str(root))
    (root / CLIPS / "st9_h1_c001.mp4").write_bytes(b"")
    body["pano_start_frame"] = 700
    (root / LABELS / "st9_h1_c001.Labels-ball.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="no whole window"):
        source.read_clips.__wrapped__(str(root))


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = V3.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        V3.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        V3.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        V3.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = V3.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:v3_match" in deny and "in:soccertrack" in deny and "in:from mmsb" in deny


# -- the real source -----------------------------------------------------------------------------------------------------------------

@needs_source
def test_every_clip_of_the_three_matches_is_a_file_in_a_folder_of_its_day():
    files, _ = V3.plan_of(V3.recipe(scale="standard", seed=1))
    assert len(files) == len(plain()) == 525 and len({f.path for f in files}) == 525
    assert {f.path.split("/")[1] for f in files} == {d for d, _ in DAYS.values()}
    by_name = {(c["day"], c["name"]): c for c in plain()}
    for f in files:
        day, name = f.path.split("/")[1:]
        c = by_name[day, name]
        assert datetime.fromtimestamp(f.t) == c["start"] and f.spec["src"].endswith(f"st{c['match']}_h{c['half']}_c{f.spec['src'][-7:-4]}.mp4")


@needs_source
@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3)), ("standard", (1, 2, 3, 4))])
def test_every_answer_is_the_labels(scale, seeds):
    for seed in seeds:
        _, questions = V3.plan_of(V3.recipe(scale=scale, seed=seed))
        if scale == "standard":
            assert 21 <= len(questions) <= 26 and 4 <= sum(q.kind == "absent" for q in questions) <= 7, seed
            assert {"when", "when_clip", "find", "every", "did", "before", "side"} <= {q.tags["template"] for q in questions}, seed
            assert {"find", "find_all", "when", "did", "how_many", "order", "who"} == {V3.filing[q.tags["template"]][0] for q in questions}, seed
        else:
            assert 2 <= len(questions) <= 9
        check_against_labels(questions)
        assert {q.split for q in questions} == {"dev", "test"}


@needs_source
def test_the_shelf_and_the_filing_are_the_taxonomys_and_cover_every_template_asked():
    taxonomy.check(V3.shelf, V3.filing)
    assert V3.shelf.media == "video" and V3.shelf.pile
    asked = {q.tags["template"] for scale in ("tiny", "standard") for seed in (1, 2, 3) for q in V3.plan_of(V3.recipe(scale=scale, seed=seed))[1]}
    assert asked <= set(V3.filing), asked - set(V3.filing)
    assert {V3.filing[t][0] for t in asked} >= {"find_all", "how_many", "did"}


@needs_source
def test_no_dev_answer_shares_its_evidence_with_a_test_question():
    """A published dev answer must not give a test one: no clip a dev question cites is cited by a test question, for every seed, and every goal and
    shot (a clip holds them) is in one split. The question of every goal of the three matches, which would join them all, is not asked."""
    for scale, seeds in (("tiny", (1, 2, 3)), ("standard", range(1, 9))):
        for seed in seeds:
            _, questions = V3.plan_of(V3.recipe(scale=scale, seed=seed))
            dev = [q for q in questions if q.split == "dev"]
            test = [q for q in questions if q.split == "test"]
            assert dev and test and len(dev) + len(test) == len(questions), (scale, seed)
            shared = {s.file for q in dev for s in q.truth.evidence} & {s.file for q in test for s in q.truth.evidence}
            assert not shared, (scale, seed, shared)
            assert "every-goal" not in {q.qid for q in questions}
            if scale == "standard":
                assert 5 <= len(dev) <= 9, (seed, len(dev))


@needs_source
def test_the_derived_classes_are_never_in_a_question_and_a_pass_received_has_no_place_in_the_pile():
    _, questions = V3.plan_of(V3.recipe(scale="standard", seed=1))
    assert not [q.qid for q in questions if "pass" in q.qid or "recovery" in q.question or "pass" in q.question]
    assert all(e.label not in ("pass_received", "recovery") for c in V3.read_clips(str(SOURCE)) for e in c.events)


@needs_source
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        for q in V3.plan_of(V3.recipe(scale=scale, seed=1))[1]:
            assert grade.grade(q.grade, right(q)), (q.qid, right(q))
            for answer in wrong(q):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


@needs_source
def test_the_questions_make_a_file_that_loads_and_another_seed_asks_other_things_of_the_same_pile(tmp_path):
    files, questions = V3.plan_of(V3.recipe(scale="standard", seed=1))
    write_questions(tmp_path / "q.jsonl", questions)
    assert load_questions(tmp_path / "q.jsonl") == questions
    other_files, other = V3.plan_of(V3.recipe(scale="standard", seed=2))
    assert files == other_files and {q.qid for q in questions} != {q.qid for q in other}


@needs_source
def test_the_tiny_pile_is_copied_as_it_is_with_its_times_and_passes_the_probe(tmp_path):
    recipe = V3.recipe(scale="tiny", seed=1)
    dump, _ = V3.build(recipe, tmp_path / "dump", 3)
    assert len(dump.entries) == 25
    for e in dump.entries[:4]:
        path = tmp_path / "dump" / e.path
        assert datetime.fromtimestamp(os.stat(path).st_mtime) == datetime.fromtimestamp(e.t)
        with av.open(str(path)) as c:
            assert abs(float(c.duration) / 1e6 - e.length) < 0.05 and c.streams.video[0].frames == 750
    files, _ = V3.plan_of(recipe)
    assert [e.size for e in dump.entries] == [os.path.getsize(SOURCE / f.spec["src"]) for f in files]
    assert V3.probe(recipe, dump, tmp_path, None, 1).status == "passed"
