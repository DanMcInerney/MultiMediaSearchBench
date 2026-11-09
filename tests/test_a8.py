"""A8, the music library: the offline parts (the tags taken off an MP3, how long a clip plays from its size, and the questions of a made-up
library, every answer derived again here by a plain reader of its tracks' genres), and, where the real source is on disk, the same over the
Free Music Archive's `fma_small`: the tiny library built (tracks copied without tags into album folders, with their upload times), every
answer of the standard library derived again here from `tracks.csv` (never the scenario's reader), the grader taking each truth and refusing
planted wrong answers, the leak probe, and `mmsb build`, `run` and `report` over it with a stand-in agent. The real-source tests skip, and
say so, without the source."""

from __future__ import annotations

import ast
import csv
import io
import json
import os
import random
import re

import threading
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import av
import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import a8_music as A8
from mmsb.scenarios import realdump
from mmsb.scenarios.a8_music import questions as Q
from mmsb.scenarios.a8_music.source import CLIP_S, CLIPS, METADATA, SOURCE, Track, frames_only, leading_tag, member, tag_bytes

REAL = pytest.mark.skipif(not ((SOURCE / CLIPS).is_file() and (SOURCE / METADATA).is_file()), reason=f"the Free Music Archive is not at {SOURCE}, so A8 is not built")
GROUP = pytest.mark.xdist_group("a8")
GENRES = ("Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock")
FRAME = b"\xff\xfb\xd0\x00" + b"\x01" * 831  # a frame of 835 bytes, as 256 kb/s at 44.1 kHz has


def id3v2(payload: int, flags: int = 0) -> bytes:
    size = payload
    return b"ID3\x04\x00" + bytes([flags]) + bytes([(size >> 21) & 127, (size >> 14) & 127, (size >> 7) & 127, size & 127]) + b"x" * payload


def mp3(frames: int = 4, tag: int = 259, v1: bool = False) -> bytes:
    return (id3v2(tag) if tag else b"") + FRAME * frames + (b"TAG" + b"t" * 125 if v1 else b"")


# -- the offline parts --------------------------------------------------------------------------------------------------------------

def test_the_tags_of_an_mp3_are_taken_off_and_what_has_no_frame_or_another_tag_is_refused():
    raw = mp3(5, 259, v1=True)
    assert leading_tag(raw[:10]) == 269 and tag_bytes(raw) == (269, 128)
    assert frames_only(raw) == FRAME * 5 and frames_only(mp3(3, 0)) == FRAME * 3 and tag_bytes(mp3(3, 0)) == (0, 0)
    assert leading_tag(id3v2(10, flags=0x10)[:10]) == 30, "an ID3v2 footer is 10 bytes more"
    with pytest.raises(ValueError, match="no frame after"):
        tag_bytes(id3v2(40) + b"junk" * 50)
    with pytest.raises(ValueError, match="APE or Lyrics3"):
        tag_bytes(mp3(3, 10) + b"APETAGEX" + b"\0" * 24)


def made_up_tracks(albums: int = 16, seed: int = 1) -> list[Track]:
    """Eight genres, `albums` albums of each, four to nine tracks an album: made-up, with real shapes (an album is of one genre)."""
    rng = random.Random(seed)
    out, album, n = [], 0, 0
    for genre in GENRES:
        for _ in range(albums):
            album += 1
            for number in range(1, rng.randint(4, 9) + 1):
                n += 1
                out.append(Track(n * 3, genre, album, number, datetime(2010, 1, 1) + timedelta(hours=rng.randint(0, 50000))))
    return out


def library(tracks, scale: str, seed: int):
    roles, folders = Q.choose(tracks, scale, seed)
    n, made = 0, []
    for k, folder in enumerate(folders, 1):
        made.append([])
        for t in folder:
            n += 1
            made[-1].append((t, realdump.File(f"Music/Album {k:03d}/track_{n:04d}.mp3", "library", realdump.local_t(t.created), "track", "audio",
                                              {"src": member(t.id)}, length=CLIP_S)))
    return roles, folders, made


def plain_check(made, questions, roles) -> None:
    """Each question's truth and evidence as a plain reader of the tracks' genres finds them."""
    names = [folder[0][1].path.split("/")[1] for folder in made]
    genre_of = {f.path: t.genre for folder in made for t, f in folder}
    said = {v: k for k, v in Q.SAID.items()}
    asked_twice = set()
    for q in questions:
        parts = q.qid.split("-")
        absent = parts[0] == "absent"
        template = parts[1] if absent else parts[0]
        genre = said[next(w for w in said if w in q.question)]
        assert (q.kind == "absent") is absent and q.tags["modality"] == "audio" and q.scenario == "A8", q.qid
        assert q.tags["template"] == ("absent_" + template if absent else template) and genre in (*roles.asked, roles.held), q.qid
        if template == "most":
            if absent:
                assert genre == roles.held and not [g for g in genre_of.values() if g == genre] and q.grade.rule == "absent" and not q.truth.evidence
                continue
            counts = [sum(genre_of[f.path] == genre for _, f in folder) for folder in made]
            top = counts.index(max(counts))
            assert sorted(counts)[-1] - sorted(counts)[-2] >= Q.MARGIN and names[top] in q.truth.value and genre in roles.asked
            assert q.grade.rule == "text" and names[top].casefold() in q.grade.truth.split("|") and f"album {int(names[top][-3:])}" in q.grade.truth.split("|")
            assert sorted(s.file for s in q.truth.evidence) == sorted(f.path for t, f in made[top] if t.genre == genre)
            continue
        folder = next(n for n in names if n in q.question)
        k = names.index(folder)
        assert (k, genre) not in asked_twice, f"{q.qid}: a folder and a genre asked about twice"
        asked_twice.add((k, genre))
        have = [f for t, f in made[k] if t.genre == genre]
        assert (len(made[k]) >= Q.FOLDER_MIN["tiny"]), q.qid
        if absent:
            assert not have and q.grade.rule == "absent" and not q.truth.evidence and q.truth.value is None, q.qid
            assert genre in roles.asked and genre != roles.held, f"{q.qid}: an absent folder question is of a genre asked about as present"
            continue
        assert genre in roles.asked and have and sorted(s.file for s in q.truth.evidence) == sorted(f.path for f in (have[:1] if template == "first" else have)), q.qid
        base = [f.path.rsplit("/", 1)[-1] for f in have]
        if template == "genre":
            assert q.grade.rule == "set" and sorted(q.grade.items) == sorted(base) and Q.GENRE_FILES[0] <= len(base) <= Q.GENRE_FILES[1], q.qid
        elif template == "count":
            assert (q.grade.rule, q.grade.truth) == ("count", str(len(base))) and Q.COUNT_FILES[0] <= len(base) <= Q.COUNT_FILES[1], q.qid
        else:
            assert q.grade.rule == "set" and q.grade.items == (base[0],) and len(base) >= Q.FIRST_FILES and "track is the first" in q.question, q.qid


def right(q) -> list[str]:
    """Natural answers in the shape asked, all right."""
    g = q.grade
    if g.rule == "set":
        return [", ".join(g.items), "\n".join(f"- {n}" for n in g.items), json.dumps(list(g.items))]
    if g.rule == "count":
        return [g.truth, f"{g.truth} tracks", f"There are {g.truth} of them."]
    if g.rule == "text":
        first, second = g.truth.split("|")
        return [first.title(), f"It is {second.title()}.", second.title()]
    return ["unknown", "None", "No track of it.", "I found none, so zero.", "0"]


def wrong(q, other_names) -> list[str]:
    """Planted wrong answers: a missing, an extra and a swapped track, a made-up one; one more, one fewer and a hedge for a count; another folder
    or two for `most`; `unknown` on a needle; tracks, a number, a folder and a yes for an absent one."""
    g = q.grade
    if g.rule == "set":
        extra = next(n for n in other_names if n not in g.items)
        return [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), ", ".join([*g.items, "track_9999.mp3"]), "unknown", "none"] + (
            [", ".join(g.items[:-1])] if len(g.items) > 1 else [])
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", "unknown", "none"]
    if g.rule == "text":
        return [g.wrong[0], f"{g.truth.split('|')[0].title()} or {g.wrong[0]}", "unknown", "Album 999"]
    return [", ".join(other_names[:2]), other_names[0], "3", "Album 012", "Yes, there are some.", "There are 5 of them."]


@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3, 4)), ("standard", (1, 2))])
def test_the_questions_of_a_made_up_library_are_what_its_genres_say(scale, seeds):
    for seed in seeds:
        tracks = made_up_tracks(albums=30 if scale == "standard" else 16, seed=seed)
        roles, folders, made = library(tracks, scale, seed)
        files = [f for folder in made for _, f in folder]
        questions = Q.questions_of(made, roles, scale, seed, max(f.t for f in files) + 86400)
        assert len(questions) == (6 if scale == "tiny" else 20) and sum(q.kind == "absent" for q in questions) == (2 if scale == "tiny" else 5), seed
        plain_check(made, questions, roles)
        others = [f.path.rsplit("/", 1)[-1] for f in files[:5]]
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_the_library_leaves_out_the_genres_taken_for_one_asked_about_and_the_held_out_one_and_keeps_albums_whole():
    tracks = made_up_tracks(albums=30)
    for seed in range(1, 9):
        roles, folders = Q.choose(tracks, "standard", seed)
        here = Counter(t.genre for folder in folders for t in folder)
        out = {"Instrumental", roles.held, *(n for g in (roles.held, *roles.asked) for n in Q.NEAR[g])}
        assert not set(here) & out and {*roles.asked, roles.other} <= set(here), (seed, here)
        assert sorted({roles.held, *roles.asked, roles.other}) == sorted(Q.ASKED)
        assert all(len(f) >= Q.FOLDER_MIN["standard"] for f in folders)
        albums = [t.album for folder in folders for t in folder]
        assert all(len({t.album for t in f} & {t.album for g in folders if g is not f for t in g}) == 0 for f in folders), "an album is in one folder"
        assert [t.number for t in folders[0][:3]] == sorted(t.number for t in folders[0][:3]) and len(albums) >= 500


def test_a_library_that_cannot_be_asked_everything_of_is_refused():
    """A few small albums make folders with no margin for `most` and too few of a genre for every question: no order of them is viable."""
    tracks = [t for t in made_up_tracks(albums=3) if t.genre in ("Hip-Hop", "Folk", "Rock", "Electronic")]
    with pytest.raises(ValueError, match="no order of the albums"):
        Q.choose(tracks, "standard", 1)
    roles, folders, _ = library(made_up_tracks(albums=16), "tiny", 1)
    assert Q.viable(folders, roles, "tiny") and not Q.viable(folders[:1], roles, "tiny")


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys_and_scales():
    recipe = A8.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="recipe has the keys"):
        A8.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        A8.recipe(scale="huge", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        A8.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})


def test_no_command_may_reach_the_source_or_the_code():
    deny = A8.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:fma_" in deny and "in:freemusicarchive" in deny and "in:a8_music" in deny and "in:from mmsb" in deny


def test_every_template_is_filed_and_the_shelf_is_a_shelf():
    from mmsb.core.taxonomy import check

    check(A8.shelf, A8.filing)
    assert {c for c, _ in A8.filing.values()} == {"find_all", "how_many", "did", "who"}
    assert all(A8.filing["absent_" + t] == A8.filing[t] for t in ("genre", "count", "first", "most"))


# -- the real source ----------------------------------------------------------------------------------------------------------------

def plain_tracks() -> dict[int, tuple[str, int, int, datetime]]:
    """From `tracks.csv` and `genres.csv`, as a second reader: for each track of the small subset, `(genre, album, number, uploaded)`."""
    with zipfile.ZipFile(SOURCE / METADATA) as z:
        genres = list(csv.DictReader(io.TextIOWrapper(z.open("fma_metadata/genres.csv"), encoding="utf-8", newline="")))
        top = {int(g["genre_id"]): g["title"] for g in genres if g["top_level"] == g["genre_id"]}
        family = {int(g["genre_id"]): top[int(g["top_level"])] for g in genres}
        rows = csv.reader(io.TextIOWrapper(z.open("fma_metadata/tracks.csv"), encoding="utf-8", newline=""))
        group, name, _ = next(rows), next(rows), next(rows)
        col = {c: i for i, c in enumerate(zip(group, name))}
        out = {}
        for r in rows:
            if r[col["set", "subset"]] == "small":
                assert {family[g] for g in ast.literal_eval(r[col["track", "genres_all"]])} == {r[col["track", "genre_top"]]}
                out[int(r[0])] = (r[col["track", "genre_top"]], int(r[col["album", "id"]]), int(r[col["track", "number"]] or 0),
                                  datetime.strptime(r[col["track", "date_created"]], "%Y-%m-%d %H:%M:%S"))
    return out


@pytest.fixture(scope="module")
def plain():
    return plain_tracks()


def as_made(files, plain) -> list[list[tuple[Track, realdump.File]]]:
    """The plan's files as folders of `(Track, File)` rebuilt from the plain reader's labels."""
    folders: dict[str, list] = defaultdict(list)
    for f in files:
        i = int(re.search(r"(\d{6})\.mp3", str(f.spec["src"]))[1])
        genre, album, number, created = plain[i]
        folders[f.path.split("/")[1]].append((Track(i, genre, album, number, created), f))
    return list(folders.values())


@GROUP
@REAL
def test_the_scenarios_reader_agrees_with_a_plain_one_of_the_labels(plain):
    from mmsb.scenarios.a8_music.source import read_tracks

    mine = read_tracks(str(SOURCE))
    assert len(mine) == len(plain) == 8000 and Counter(t.genre for t in mine) == {g: 1000 for g in GENRES}
    assert all(plain[t.id] == (t.genre, t.album, t.number, t.created) for t in mine)



@GROUP
@REAL
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_answer_is_the_labels_at_the_standard_scale(plain, seed):
    files, questions = A8.plan_of(A8.recipe(scale="standard", seed=seed))
    made = as_made(files, plain)
    roles = Q.roles_of(seed, "standard")
    plain_check(made, questions, roles)
    here = Counter(t.genre for folder in made for t, _ in folder)
    out = {"Instrumental", roles.held, *(n for g in (roles.held, *roles.asked) for n in Q.NEAR[g])}
    assert not set(here) & out and {*roles.asked, roles.other} <= set(here)
    assert len(questions) == 20 and sum(q.kind == "absent" for q in questions) == 5 and 2400 < len(files) < 3300 and 60 < len(made) < 140
    assert {q.tags["template"] for q in questions} == {"genre", "count", "first", "most", "absent_genre", "absent_count", "absent_first", "absent_most"}
    assert 5 <= sum(q.split == "dev" for q in questions) <= 7 and all(f.length == CLIP_S for f in files) and sum(f.length for f in files) / 3600 > 20
    assert all(len(f) >= 24 for f in made) and [f.path for f in files][0] == "Music/Album 001/track_0001.mp3"
    # albums are whole and in one folder
    folder_of = {t.album: k for k, folder in enumerate(made) for t, _ in folder}
    assert all(folder_of[t.album] == k for k, folder in enumerate(made) for t, _ in folder)
    inside = [[t.number for t, _ in folder] for folder in made]
    assert any(n != sorted(n) for n in inside), "folders hold more than one album"


@GROUP
@REAL
def test_another_seed_asks_of_other_genres_and_albums_and_one_seed_makes_one_dump():
    one = A8.plan_of(A8.recipe(scale="standard", seed=1))
    assert one == A8.plan_of(A8.recipe(scale="standard", seed=1)) and A8.hashes(A8.recipe(scale="standard", seed=1)) == A8.hashes(
        A8.recipe(scale="standard", seed=1))
    two = A8.plan_of(A8.recipe(scale="standard", seed=2))
    assert {q.qid for q in one[1]} != {q.qid for q in two[1]} and one[0] != two[0]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("a8")
    recipe = A8.recipe(scale="tiny", seed=1)
    dump, questions = A8.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


@GROUP
@REAL
def test_the_tiny_library_has_its_tracks_without_tags_in_album_folders_with_their_upload_times(built, plain):
    out, recipe, dump, questions = built
    files, plan = A8.plan_of(recipe)
    assert [e.path for e in dump.entries] == [f.path for f in files] and {e.media for e in dump.entries} == {"audio"} and 140 < len(files) < 200
    assert [q.qid for q in plan] == [q.qid for q in questions] and len(plan) == 6 and sum(q.kind == "absent" for q in plan) == 2
    plain_check(as_made(files, plain), plan, Q.roles_of(1, "tiny"))
    with zipfile.ZipFile(SOURCE / CLIPS) as z:
        for e, f in zip(dump.entries[::7], files[::7]):
            path = out / "dump" / e.path
            raw = path.read_bytes()
            assert os.stat(path).st_mtime == e.t == f.t == f.t and not e.own_time and len(raw) == e.size
            assert raw[:3] != b"ID3" and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0 and raw[-128:-125] != b"TAG"
            source = z.read(str(f.spec["src"]))
            assert raw == source[leading_tag(source[:10]):], "the frames are the source's"
            with av.open(str(path)) as clip:
                assert abs(clip.duration / 1e6 - e.length) < 1e-3 and e.length >= CLIP_S and not clip.metadata
    assert datetime.fromtimestamp(dump.entries[0].t) == plain[int(re.search(r"(\d{6})\.mp3", str(files[0].spec["src"]))[1])][3]
    assert A8.probe(recipe, dump, out, None, 1).status == "passed"


@GROUP
@REAL
def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers():
    for scale in ("tiny", "standard"):
        files, questions = A8.plan_of(A8.recipe(scale=scale, seed=1))
        others = [f.path.rsplit("/", 1)[-1] for f in files[:5]]
        for q in questions:
            for answer in right(q):
                assert grade.grade(q.grade, answer), (q.qid, answer)
            for answer in wrong(q, others):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


class Spy:
    """An agent that answers right in the `files` configuration, citing a track of the truth, and `unknown` in any other."""

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
def test_build_run_and_report_over_the_tiny_library(world, capsys):  # noqa: F811
    world.patch.setitem(scenarios.REGISTRY, "a8_music", "mmsb.scenarios.a8_music")
    assert world.main("build", "--scenario", "a8_music", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    questions = run.questions()
    spy = Spy(questions)
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    genre = next(q for q in questions if q.tags["template"] == "genre")
    prompt = spy.prompts["files", genre.qid]
    assert prompt.startswith("It is now ") and "The files of my music library are in " in prompt and genre.question in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    (section,) = fs.read_json(folder / "report.json")["scenarios"]
    files, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files["correct"], files["n"], files["grounding"]["grounded"]) == (6, 6, 6)
    assert (listing["correct"], listing["n"]) == (2, 6) and section["leaks"] == [], "listing alone answered only the absent ones"
