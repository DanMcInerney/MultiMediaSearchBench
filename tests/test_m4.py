"""M4, the sound samples, from its real source on disk: the tiny folder built (clips numbered, copied with their samples alone and their
times), every answer derived again here by a plain reader of the label files (never the scenario's), the grader taking each truth and
refusing planted wrong answers, the leak probe, and `mmmb build`, `run` and `report` over it with a stand-in agent. Skips, and says so,
without the source."""

from __future__ import annotations

import csv
import functools
import itertools
import json
import os
import re
import threading
import wave
from collections import defaultdict
from pathlib import Path

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmmb import scenarios
from mmmb.core import fs, grade
from mmmb.core.agents import claude
from mmmb.core.runfolder import RunFolder
from mmmb.core.runner import Agent
from mmmb.scenarios.m4_samples.questions import CLASSES, NEAR
from mmmb.scenarios.m4_samples.source import SOURCE

if not (SOURCE / "labels" / "eval.csv").is_file():
    pytest.skip(f"the FSD50K source is not at {SOURCE}, so M4 is not built", allow_module_level=True)

M4 = scenarios.get("m4_samples")
pytestmark = pytest.mark.xdist_group("m4")
WORDS = {"toilet_flush": "toilet flushing", "microwave_oven": "microwave oven", "meow": "cat meowing",
         "zipper_clothing": "zipper", "scissors": "scissors cutting", "printer": "printer", "tambourine": "tambourine", "gong": "gong",
         "thunder": "thunder", "hammer": "hammering", "skateboard": "skateboard", "coin_dropping": "coin dropping", "fart": "fart",
         "packing_tape_and_duct_tape": "packing tape", "camera": "camera shutter", "harp": "harp", "finger_snapping": "fingers snapping",
         "chewing_and_mastication": "someone chewing", "drill": "power drill", "cough": "cough",
         "writing": "someone writing"}  # what a question about a class says, written here
REVIEWED = {"Gong": {"Bell", "Chime", "Crash_cymbal", "Church_bell"}, "Printer": {"Typing", "Typewriter"},
            "Thunder": {"Explosion", "Fireworks", "Boom"}, "Hammer": {"Thump_and_thud", "Knock", "Tap"},
            "Coin_(dropping)": {"Chink_and_clink", "Cutlery_and_silverware", "Keys_jangling"},
            "Tambourine": {"Bell", "Rattle_(instrument)"}}  # the classes a reviewer found in the dumps for a sound asked about


@functools.cache
def plain_labels() -> tuple[dict[str, list[str]], dict[str, set[str]], list[str]]:
    """From the label files, as a second reader: each eval clip's classes, those of them two raters found present and predominant (and
    none found only present), and the vocabulary's classes."""
    with open(SOURCE / "labels" / "vocabulary.csv", newline="") as f:
        vocabulary = [row[1] for row in csv.reader(f)]
    ratings = json.loads((SOURCE / "metadata" / "pp_pnp_ratings_FSD50K.json").read_text())
    labels, strong = {}, {}
    with open(SOURCE / "labels" / "eval.csv", newline="") as f:
        for row in csv.DictReader(f):
            classes, mids = row["labels"].split(","), row["mids"].split(",")
            labels[row["fname"]] = classes
            strong[row["fname"]] = {c for c, m in zip(classes, mids) if ratings.get(row["fname"], {}).get(m, []).count(1.0) >= 2
                                    and 0.5 not in ratings[row["fname"]][m]}
    return labels, strong, vocabulary


def slug_of(vocabulary: list[str]) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"): name for name in vocabulary}


def clip_of(f) -> str:
    """The Freesound id of the source clip a file of the plan is made from."""
    return Path(f.spec["src"]).stem


def riff_ids(raw: bytes) -> list[bytes]:
    """The ids of a RIFF file's chunks, in order, read here."""
    ids, at = [], 12
    while at < len(raw):
        ids.append(raw[at:at + 4])
        at += 8 + int.from_bytes(raw[at + 4:at + 8], "little") + (int.from_bytes(raw[at + 4:at + 8], "little") & 1)
    return ids


def parse(qid: str) -> tuple[bool, str, str]:
    """`absent-find-harp` as (True, `find`, `harp`): whether it is absent, its template and the sound's slug."""
    parts = qid.split("-")
    return (True, parts[1], parts[2]) if parts[0] == "absent" else (False, parts[0], parts[1])


def check_against_labels(files, questions) -> set[str]:
    """Each question's truth and evidence as the plain reader finds them; returns the classes asked about."""
    labels, strong, vocabulary = plain_labels()
    label_of = slug_of(vocabulary)
    by_path = {f.path: f for f in files}
    asked = {label_of[parse(q.qid)[2]] for q in questions}
    near = {n for a in asked for n in NEAR[a]}
    assert not asked & near and near <= set(vocabulary)
    taken = sorted(p for p, f in by_path.items() if near & set(labels[clip_of(f)]))
    assert not taken, f"the dump holds clips of a class taken for a sound asked about: {taken[:3]}"
    for q in questions:
        absent, template, key = parse(q.qid)
        label = label_of[key]
        of = sorted(p for p, f in by_path.items() if label in labels[clip_of(f)])
        assert WORDS[key] in q.question, f"{q.qid} asks of another sound: {q.question}"
        assert (q.kind == "absent") is absent and q.tags["absent"] is absent and q.tags["modality"] == "audio", q.qid
        if absent:
            assert q.question.startswith("Find every clip of ") and not of, f"{q.qid}: the dump holds {of}"
            assert any(label in ls for ls in labels.values()), f"{q.qid}: a sound the source never had"
            assert q.grade.rule == "absent" and not q.truth.evidence and q.truth.value is None
            continue
        assert of and all(label in strong[clip_of(by_path[p])] for p in of), f"{q.qid}: a clip that is not a plain case"
        assert all(asked & set(labels[clip_of(by_path[p])]) == {label} for p in of), f"{q.qid}: a clip of two asked-about sounds"
        assert sorted(s.file for s in q.truth.evidence) == of and all((s.t0, s.t1) == (0.0, by_path[s.file].length) for s in q.truth.evidence)
        if template == "find":
            assert q.question.startswith("Find every clip of ") and 2 <= len(of) <= 8, q.qid
            assert sorted(q.grade.items) == sorted(p.rsplit("/", 1)[-1] for p in of) and q.grade.rule == "set", q.qid
        else:
            assert q.question.startswith("How many clips of ") and 9 <= len(of) <= 40, q.qid
            assert (q.grade.rule, q.grade.truth) == ("count", str(len(of))), q.qid
    return asked


def right(q, path_of) -> list[str]:
    """Natural answers in the shape asked, all right."""
    g = q.grade
    if g.rule == "set":
        return [", ".join(g.items), "The clips are " + " and ".join(path_of[n] for n in reversed(g.items)) + ".",
                "\n".join(f"- {n}" for n in g.items), json.dumps(list(g.items))]
    if g.rule == "count":
        return [g.truth, f"{g.truth} clips", f"There are {g.truth} clips of it."]
    return ["unknown", "None", "There are no such clips.", "I found no clip of it, so none.", "0"]


def wrong(q, other_names) -> list[str]:
    """Planted wrong answers: a missing, an extra and a swapped clip, a made-up one, a number for a set; one more, one fewer and a hedge
    for a count; `unknown` on a needle; clips, a number and a yes for an absent one."""
    g = q.grade
    if g.rule == "set":
        extra = next(n for n in other_names if n not in g.items)
        return [", ".join(g.items[:-1]), ", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), ", ".join([*g.items, "rec_99999.wav"]),
                "unknown", "none", str(len(g.items))]
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", f"about {n + 3}", "unknown", "none"]
    return [", ".join(other_names[:2]), other_names[0], "3", "Yes, there are some.", "There are 5 clips of it."]


@pytest.fixture(scope="module")
def tiny():
    return M4.plan_of(M4.recipe(scale="tiny", seed=1))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("m4")
    recipe = M4.recipe(scale="tiny", seed=1)
    dump, questions = M4.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_folder_is_a_few_hundred_clips_with_a_quarter_of_its_questions_absent(tiny):
    files, questions = tiny
    assert len(files) == 240 and len(questions) == 8 and sum(q.kind == "absent" for q in questions) == 2
    assert {q.kind for q in questions} == {"find", "count", "absent"}
    assert sum(q.split == "dev" for q in questions) == 2 and {q.split for q in questions} == {"dev", "test"}
    assert all(q.now.startswith("2026-05-01T09:00:00") for q in questions), "asked the morning after the last day"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_answer_is_the_labels_at_the_standard_scale(seed):
    files, questions = M4.plan_of(M4.recipe(scale="standard", seed=seed))
    asked = check_against_labels(files, questions)
    labels = plain_labels()[0]
    assert len(questions) == 21 and sum(q.kind == "absent" for q in questions) == 5 and len(asked) == 21
    assert [sum(q.kind == k for q in questions) for k in ("find", "count")] == [10, 6]
    assert 5 <= sum(q.split == "dev" for q in questions) <= 7
    # the dump is the split but for the asked-about sounds and those taken for them: all the clips holding none of them, and of each of
    # the asked-about its few
    here = {clip_of(f) for f in files}
    near = {n for a in asked for n in NEAR[a]}
    plain = {c for c, ls in labels.items() if not (asked | near) & set(ls)}
    kept = here - plain
    assert plain <= here and all(len(asked & set(labels[c])) == 1 and not near & set(labels[c]) for c in kept)
    needles = sum(len(q.grade.items) if q.kind == "find" else int(q.grade.truth) for q in questions if q.kind != "absent")
    assert len(kept) == needles and len(files) == len(plain) + needles and 5500 < len(files) < 6600
    for label in asked & set(REVIEWED):  # what a reviewer found: none of those classes is in the dump, whichever way the sound is asked
        assert not [c for c in here if REVIEWED[label] & set(labels[c])], f"a class taken for {label} is in the dump"
    seconds = {f.path: (os.path.getsize(SOURCE / f.spec["src"]) - 78) / 88200 for f in files}
    assert all(f.length == seconds[f.path] for f in files)
    assert 4.8e9 < sum(s * 88200 + 44 for s in seconds.values()) < 5.6e9


def test_every_answer_is_the_labels_at_the_tiny_scale(tiny):
    files, questions = tiny
    asked = check_against_labels(files, questions)
    labels, _, _ = plain_labels()
    here = {clip_of(f) for f in files}
    assert len(asked) == 8 and all(len(asked & set(labels[c])) <= 1 for c in here), "no clip holds two asked-about sounds"
    assert sum(1 for c in here if asked & set(labels[c])) == sum(len(q.truth.evidence) for q in questions)


def test_another_seed_asks_other_sounds_of_another_dump_and_one_seed_makes_one_dump():
    one = M4.plan_of(M4.recipe(scale="standard", seed=1))
    assert one == M4.plan_of(M4.recipe(scale="standard", seed=1)) and M4.hashes(M4.recipe(scale="standard", seed=1)) == M4.hashes(
        M4.recipe(scale="standard", seed=1))
    two = M4.plan_of(M4.recipe(scale="standard", seed=2))
    assert {q.qid for q in one[1]} != {q.qid for q in two[1]} and M4.hashes(M4.recipe(scale="standard", seed=1)) != M4.hashes(
        M4.recipe(scale="standard", seed=2))


def test_every_sound_asked_about_is_a_leaf_class_of_the_vocabulary():
    """No class of the vocabulary has all its clips among a table class's (a child would make `find a harp` also ask for its children)."""
    labels, _, vocabulary = plain_labels()
    clips = defaultdict(set)
    for c, ls in labels.items():
        for name in ls:
            clips[name].add(c)
    assert set(CLASSES) <= set(vocabulary) and set(WORDS) == {re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_") for k in CLASSES}
    for name in CLASSES:
        assert not [other for other in clips if other != name and clips[other] <= clips[name]], f"{name} has a child class"


def test_each_sound_asked_about_names_the_classes_taken_for_it():
    """Every class of the table has its list (empty for a sound nothing is taken for); each class listed is of the vocabulary, none is a
    sound of the table (two asked about would be taken for each other) nor a parent of the sound's own clips (they would all go); and the
    lists include what a reviewer found in the dumps."""
    labels, _, vocabulary = plain_labels()
    clips = defaultdict(set)
    for c, ls in labels.items():
        for name in ls:
            clips[name].add(c)
    assert set(NEAR) == set(CLASSES) and all(isinstance(n, tuple) for n in NEAR.values())
    for name, near in NEAR.items():
        assert set(near) <= set(vocabulary) and not set(near) & set(CLASSES), name
        assert not [n for n in near if clips[name] <= clips[n]], f"{name}: a parent of its own clips is no class taken for it"
        assert len(near) == len(set(near)) and name not in near
    assert all(set(REVIEWED[name]) <= set(NEAR[name]) for name in REVIEWED)


@pytest.mark.parametrize("scale, folders", [("tiny", 3), ("standard", 10)])
def test_the_clips_are_numbered_in_time_order_in_folders_that_do_not_group_by_sound(scale, folders):
    files, questions = M4.plan_of(M4.recipe(scale=scale, seed=1))
    n = len(files)
    assert [f.path.rsplit("/", 1)[-1] for f in files] == [f"rec_{k:05d}.wav" for k in range(1, n + 1)], "distinct, in order"
    assert [f.path.split("/")[0] for f in files] == [f"folder_{k * folders // n + 1:02d}" for k in range(n)]
    assert all(b.t >= a.t + a.length + 60 for a, b in itertools.pairwise(files)), "a minute at least between one clip's end and the next"
    ids = [int(clip_of(f)) for f in files]
    assert ids != sorted(ids), "not in the source's order"
    count = next(q for q in questions if q.kind == "count")
    assert len({s.file.split("/")[0] for s in count.truth.evidence}) >= 2, "one sound is in more than one folder"
    assert all(re.fullmatch(r"folder_\d\d/rec_\d{5}\.wav", f.path) for f in files)


def test_a_clip_is_copied_with_its_samples_alone_and_its_time(built):
    out, recipe, dump, _ = built
    files, _ = M4.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] and {e.media for e in dump.entries} == {"audio"}
    frames = 0
    for e, f in zip(dump.entries, files):
        path = out / "dump" / e.path
        assert os.stat(path).st_mtime == e.t == f.t and not e.own_time
        raw = path.read_bytes()
        assert riff_ids(raw) == [b"fmt ", b"data"] and len(raw) == e.size and b"Lavf" not in raw and b"ISFT" not in raw
        with wave.open(str(path)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 44100)
            frames += w.getnframes()
            assert w.getnframes() / 44100 == e.length == f.length
    assert frames / 44100 == sum(e.length for e in dump.entries)
    for n in (0, 119, 239):
        source = (SOURCE / files[n].spec["src"]).read_bytes()
        assert (out / "dump" / files[n].path).read_bytes()[44:] == source[78:], "the samples are the source's"
    assert M4.probe(recipe, dump, out, None, 1).status == "passed"


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = M4.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        M4.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        M4.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        M4.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = M4.deny_words([], None, "k")
    assert "bench-data" in deny and "in:fsd50k" in deny and "in:freesound" in deny and "in:m4_samples" in deny and "in:from mmmb" in deny


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    for files, questions in (tiny, M4.plan_of(M4.recipe(scale="standard", seed=1))):
        path_of = {f.path.rsplit("/", 1)[-1]: f.path for f in files}
        others = list(path_of)[:5]
        for q in questions:
            for answer in right(q, path_of):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


class Spy:
    """An agent that answers right in the `files` configuration, citing a clip of the truth, and `unknown` in any other."""

    def __init__(self, questions, path_of) -> None:
        self.questions = {q.qid: q for q in questions}
        self.path_of = path_of
        self.prompts: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec) -> dict:
        q, folder = self.questions[spec.run_dir.name], spec.run_dir.parent.name
        with self.lock:
            self.prompts[folder, q.qid] = spec.prompt
        cited = [{"file": s.file, "t0": 0, "t1": 0} for s in q.truth.evidence[:1]]
        answer = right(q, self.path_of)[0] if folder == "files" else "unknown"
        final = json.dumps({"answer": answer, "evidence": cited if answer != "unknown" else []})
        return {"final": final, "tools": [], **clocked(spec, 1.5), "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


def test_build_run_and_report_over_the_tiny_folder(world, capsys):  # noqa: F811
    assert world.main("build", "--scenario", "m4_samples", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    questions = run.questions()
    spy = Spy(questions, {f.path.rsplit("/", 1)[-1]: f.path for f in M4.plan_of(M4.recipe(scale="tiny", seed=1))[0]})
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    find = next(q for q in questions if q.kind == "find")
    prompt = spy.prompts["files", find.qid]
    assert prompt.startswith("It is now Friday 1 May 2026, 09:00. The files of my folder of sound clips are in ")
    assert find.question in prompt and "<the names of the files, as in rec_00042.wav, separated by commas>" in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMMB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (8, 8, 8)
    assert (listing["correct"], listing["n"]) == (2, 8) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 8}
