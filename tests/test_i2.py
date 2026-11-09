"""I2, the phone's screenshots, from its real source on disk: the tiny folder built (screenshots named by when they were taken, copied as they
are, with their times), every answer derived again here by a plain reader of the label file (never the scenario's), the held-out tasks and
phrases absent from what is left, the grader taking each truth and refusing planted wrong answers, the leak probe, and `mmsb build`, `run` and
`report` over it with a stand-in agent; and the choice of tasks and phrases over made-up screens. Skips, and says so, without the source."""

from __future__ import annotations

import functools
import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime

import pytest
from agent_spec import clocked
from cli_world import world  # noqa: F401

from mmsb import scenarios
from mmsb.core import fs, grade
from mmsb.core.agents import claude
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.scenarios import i2_screenshots as I2
from mmsb.scenarios import realdump
from mmsb.scenarios.i2_screenshots import questions as asking
from mmsb.scenarios.i2_screenshots import source, tasks, words

if not (source.SOURCE / source.LABELS).is_file():
    pytest.skip(f"the Android in the Wild source is not at {source.SOURCE}, so I2 is not built", allow_module_level=True)

pytestmark = pytest.mark.xdist_group("i2")
IMPERATIVE = {"open", "check", "find", "play", "install", "search", "set", "toggle", "turn", "google"}
ASKS = {"tasks": re.compile(r"Show me every screenshot from when I (.+)\.$"), "size": re.compile(r"How many screenshots did I take when I (.+)\?$"),
        "did": re.compile(r"Did I ever (.+)\? If so, which was the first screenshot of it\?$"),
        "words": re.compile(r'Find the screenshot that says "(.+)"\.$')}


def fold(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'", "")))


@functools.cache
def plain() -> dict[str, dict]:
    """From the label file, as a second reader: each screenshot's row by its file."""
    with open(source.SOURCE / source.LABELS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {r["file"]: r for r in rows}


@functools.cache
def episodes_of() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in plain().values():
        out[r["episode_id"]].append(r)
    return {e: sorted(v, key=lambda r: r["step_id"]) for e, v in out.items()}


def said(goal: str) -> tuple[str, str]:
    """How a person says a task, written out here: after `when I` and after `Did I ever`."""
    g = goal.rstrip("?. ")
    if g.split()[0].lower() in IMPERATIVE:
        return f"was trying to {g[0].lower()}{g[1:]}", f"try to {g[0].lower()}{g[1:]}"
    return f'googled "{g}"', f'google "{g}"'


@functools.cache
def goals_by_wording() -> dict[tuple[str, str], str]:
    """`(template, what the question says)` -> the goal it asks about."""
    out = {}
    for goal in {r["goal"] for r in plain().values()}:
        past, plain_ = said(goal)
        out["tasks", past] = out["size", past] = goal
        out["did", plain_] = goal
    return out


def plan(scale: str, seed: int):
    return _plan(scale, seed)


@functools.cache
def _plan(scale: str, seed: int):
    return I2.plan_of(I2.recipe(scale=scale, seed=seed))


def texts_of(files) -> dict[str, str]:
    """The folded text of each file of a dump, by its path: the OCR lines the source gives, as a second reader."""
    return {f.path: " " + fold(" ".join(t for t in plain()[f.spec["src"]]["ocr_text"] if t.strip())) + " " for f in files}


def in_order(words: list[str], text: str) -> bool:
    """Whether the words of `words` are among those of `text`, as whole words, in that order."""
    found = iter(text.split())
    return all(w in found for w in words)


COMMON = {"the", "of", "a", "an", "in", "is", "to", "for", "at", "on", "and", "from", "what", "whats", "how", "do", "does", "i", "me", "my", "it", "are",
          "nearest", "search", "open", "check", "play", "install", "set", "find", "google", "show", "some", "get", "new", "latest"}


def stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def goal_words(goal: str) -> set[str]:
    """The words of a goal that tell it from another, read here."""
    return {stem(w) for w in fold(goal).split() if w not in COMMON}


def share(row: dict, words: set[str]) -> float:
    """The share of `words` that a row's OCR text has, as whole words."""
    return len(words & {stem(w) for w in fold(" ".join(t for t in row["ocr_text"] if t.strip())).split()}) / len(words)


def scale_of(files) -> str:
    return "tiny" if len(files) < 500 else "standard"


def check_against_labels(files, questions) -> None:
    """Each question's truth and evidence as the plain reader finds them in the label file."""
    src = {f.path: f.spec["src"] for f in files}
    name = {s: p.rsplit("/", 1)[-1] for p, s in src.items()}
    in_dump = set(src.values())
    texts = texts_of(files)
    episodes = {r["episode_id"] for s in in_dump for r in [plain()[s]]}
    goals_here = {plain()[s]["goal"] for s in in_dump}
    outside = [r for f, r in plain().items() if f not in in_dump]
    for q in questions:
        absent = q.kind == "absent"
        template = q.tags["template"].removeprefix("absent_")
        assert (q.tags["template"] != template) is absent, q.qid
        if template == "read":
            read = next(r for r in asking.READS if f"read-{r.key}" == q.qid)
            shown = sorted(p for p, t in texts.items() if all(f" {w} " in t or f"{w} " in t for w in read.where) and set(read.where) <= set(t.split()))
            assert shown and all(f" {read.check} " in texts[p] for p in shown), q.qid
            assert sorted(s.file for s in q.truth.evidence) == shown and q.grade.rule == "text" and q.grade.truth == read.accepted, q.qid
            continue
        match = ASKS[template].search(q.question)
        assert match, f"{q.qid}: {q.question}"
        if template == "words":
            phrase = match[1]
            assert phrase == fold(phrase) and all(c.isalpha() or c == " " for c in phrase), q.qid
            holders = [p for p, t in texts.items() if f" {phrase} " in t]
            if absent:
                assert not holders and q.grade.rule == "absent" and not q.truth.evidence, q.qid
                assert any(f" {phrase} " in " " + fold(" ".join(r["ocr_text"])) + " " for r in outside), f"{q.qid}: the phrase was never on a held-out screen"
            else:
                assert len(holders) == 1 and [s.file for s in q.truth.evidence] == holders and q.grade.items == (holders[0].rsplit("/", 1)[-1],), q.qid
            continue
        goal = goals_by_wording()[template if template != "did" else "did", match[1]]
        asked_eps = {e for e, rows in episodes_of().items() if rows[0]["goal"] == goal}
        if absent:
            assert goal not in goals_here and not asked_eps & episodes and q.grade.rule == "absent" and not q.truth.evidence, q.qid
            # what is left shows nothing of the goal: no screen has its words but the commonest, as words, in its order
            longer = [w for w in fold(goal).split() if w not in {"the", "of", "a", "an", "in", "is", "to", "for", "at", "on", "and", "from"}]
            assert not any(in_order(longer, t) for t in texts.values()), f"{q.qid}: a screen shows {longer}"
            continue
        mine = [s for s in in_dump if plain()[s]["episode_id"] in asked_eps]
        kept = {plain()[s]["episode_id"] for s in mine}  # the folder holds all of the source's episodes of a task, but at the tiny scale
        assert kept and all(len(episodes_of()[e]) == episodes_of()[e][0]["episode_length"] for e in kept), "a task of whole episodes"
        assert len(kept) == len(asked_eps) or scale_of(files) == "tiny"
        words = goal_words(goal)
        for e in kept:  # each episode is cut to the screens that show the goal's words: all that show every one are there, and each there shows most
            there = {s for s in mine if plain()[s]["episode_id"] == e}
            assert {r["file"] for r in episodes_of()[e] if share(r, words) == 1} <= there <= {r["file"] for r in episodes_of()[e] if share(r, words) >= 0.6}, q.qid
        assert sorted(s.file for s in q.truth.evidence) == sorted(p for p, s in src.items() if s in mine) or template == "did", q.qid
        if template == "tasks":
            assert set(q.grade.items) == {name[s] for s in mine}, q.qid
        elif template == "size":
            assert q.grade.truth == str(len(mine)) and q.grade.rule == "count", q.qid
        else:
            (first,) = q.grade.items
            assert len(kept) == 1 and first == name[min(mine, key=lambda s: plain()[s]["step_id"])], q.qid


def right(q) -> str:
    g = q.grade
    return {"set": lambda: ", ".join(g.items), "count": lambda: g.truth, "text": lambda: g.truth.split("|")[0], "absent": lambda: "unknown"}[g.rule]()


def wrong(q, other_names) -> list[str]:
    """Planted wrong answers: a file missing, extra or swapped; one more or fewer, and a hedge; another value; `unknown` on a needle; files,
    a number and a yes for an absent one."""
    g = q.grade
    if g.rule == "set":
        extra = next(n for n in other_names if n not in g.items)
        out = [", ".join([*g.items, extra]), ", ".join([*g.items[:-1], extra]), "unknown", "none"]
        return out + ([", ".join(g.items[1:])] if len(g.items) > 1 else [])
    if g.rule == "count":
        n = int(g.truth)
        return [str(n + 1), str(n - 1), f"{n} or {n + 1}", "unknown"]
    if g.rule == "text":
        return ["unknown", "I cannot tell", "7"]
    return [", ".join(other_names[:2]), other_names[0], "3", "Yes, there are some."]


@pytest.fixture(scope="module")
def tiny():
    return plan("tiny", 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("i2")
    recipe = I2.recipe(scale="tiny", seed=1)
    dump, questions = I2.build(recipe, out / "dump", 3)
    return out, recipe, dump, questions


def test_the_tiny_folder_is_a_hundred_screenshots_with_a_quarter_of_its_questions_absent(tiny):
    files, questions = tiny
    assert 80 <= len(files) <= 140 and len(questions) == 7 and sum(q.kind == "absent" for q in questions) == 2
    assert {q.tags["template"] for q in questions} == {"tasks", "size", "did", "words", "read", "absent_tasks", "absent_did"}
    assert {q.split for q in questions} == {"dev", "test"}
    assert all(q.now.startswith("2026-10-12T09:00:00") or q.now.startswith("2026-10-11T09:00:00") for q in questions), "the morning after the last"
    assert all(re.fullmatch(r"Pictures/Screenshots/Screenshot_2026101\d-\d{6}\.png|Pictures/Screenshots/Screenshot_202610(05|06|07|08|09)-\d{6}\.png", f.path)
               for f in files)


@pytest.mark.parametrize("scale, seeds", [("tiny", (1, 2, 3)), ("standard", (1, 2, 3))])
def test_every_answer_is_the_labels(scale, seeds):
    for seed in seeds:
        files, questions = plan(scale, seed)
        check_against_labels(files, questions)
        absent = sum(q.kind == "absent" for q in questions)
        if scale == "standard":
            assert 2900 < len(files) < 3300 and len(questions) == 23 and absent == 5, seed
            assert {q.tags["template"] for q in questions} == {f"{a}{t}" for a in ("", "absent_") for t in ("tasks", "size", "did", "words")} | {"read"}
            assert 6 <= sum(q.split == "dev" for q in questions) <= 8


def test_the_folder_is_the_same_at_every_seed_and_the_questions_differ():
    files_1, one = plan("standard", 1)
    files_2, two = plan("standard", 2)
    assert files_1 == files_2, "a seed draws questions from one folder"
    assert {q.qid for q in one} != {q.qid for q in two}
    assert {q.qid for q in one if q.tags["template"] == "read"} == {q.qid for q in two if q.tags["template"] == "read"}, "written, not drawn"


def test_the_screenshots_are_in_sittings_of_an_episode_a_few_seconds_apart_and_twenty_minutes_from_the_next():
    files, _ = plan("standard", 1)
    by_episode = defaultdict(list)
    for f in files:
        by_episode[plain()[f.spec["src"]]["episode_id"]].append(f)
    spans = []
    for fs_ in by_episode.values():
        steps = [plain()[f.spec["src"]]["step_id"] for f in fs_]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        gaps = [b.t - a.t for a, b in zip(fs_, fs_[1:])]
        assert all(3 <= g <= 15 for g in gaps)
        spans.append((fs_[0].t, fs_[-1].t))
    spans.sort()
    assert all(b[0] - a[1] >= 20 * 60 for a, b in zip(spans, spans[1:]))
    assert [f.t for f in files] == sorted(f.t for f in files) and len({f.path for f in files}) == len(files)
    assert all(f.path == f"Pictures/Screenshots/Screenshot_{datetime.fromtimestamp(f.t):%Y%m%d-%H%M%S}.png" for f in files)
    assert datetime.fromtimestamp(files[0].t).date().isoformat() >= "2026-05-04" and datetime.fromtimestamp(files[-1].t).date().isoformat() <= "2026-10-31"


def test_no_size_answer_is_the_size_of_one_sitting_alone():
    """The list of the files shows each sitting's size, and a `size` question whose answer is the size of one sitting only would be answered
    by the list: every asked size is also that of at least two other sittings."""
    files, questions = plan("standard", 1)
    sizes = defaultdict(int)
    for f in files:
        sizes[plain()[f.spec["src"]]["episode_id"]] += 1
    sittings = list(sizes.values())
    for q in questions:
        if q.tags["template"] == "size":
            assert sittings.count(int(q.grade.truth)) >= 3, q.qid


def test_no_held_out_task_has_a_screen_left_and_no_read_question_rests_on_one_that_disagrees():
    files, _ = plan("standard", 1)
    here = {plain()[f.spec["src"]]["episode_id"] for f in files}
    gone = [rows for e, rows in episodes_of().items() if e not in here]
    assert 20 <= len({rows[0]["goal"] for rows in gone}) <= 30 and len(gone) < 60
    assert {r.episode for r in asking.READS} <= here, "the episodes the `read` questions are written from are in the folder"
    for g in {rows[0]["goal"] for rows in gone}:
        assert not [e for e in here if episodes_of()[e][0]["goal"] == g], f"{g} has an episode left"


@pytest.mark.parametrize("key, right_answers, wrong_answers", [
    ("humidity-chicago", ["58", "58%", "It showed 58% humidity", "Humidity: 58 percent", "about 58%"], ["76%", "58 or 60", "maybe 58%", "58% and 60%", "unknown"]),
    ("humidity-los-angeles", ["76", "76%", "The humidity was 76%"], ["58%", "76 or 77", "perhaps 76"]),
    ("downloads-alexa", ["100M+", "100M", "100 million", "over 100 million downloads", "100,000,000+", "100 M"], ["10M+", "50M+", "100M or 50M", "maybe 100M"]),
    ("downloads-lyft", ["50M+", "50 million", "50,000,000", "It has 50M+ downloads"], ["100M+", "5M+", "50M+ or 100M+", "possibly 50M"])])
def test_a_value_read_off_a_screen_takes_the_forms_a_person_writes_and_refuses_a_second_guess(key, right_answers, wrong_answers):
    (q,) = [q for q in plan("standard", 1)[1] if q.qid == f"read-{key}"]
    for answer in right_answers:
        assert grade.grade(q.grade, answer), answer
    for answer in wrong_answers:
        assert not grade.grade(q.grade, answer), answer


def test_the_grader_takes_each_truth_and_refuses_the_planted_wrong_answers(tiny):
    for scale, seed in (("tiny", 1), ("standard", 1), ("standard", 2)):
        files, questions = plan(scale, seed)
        names = [f.path.rsplit("/", 1)[-1] for f in files]
        for q in questions:
            assert grade.grade(q.grade, right(q)), (q.qid, right(q))
            for answer in wrong(q, names):
                assert not grade.grade(q.grade, answer), (q.qid, answer)


def test_a_screenshot_is_copied_as_it_is_with_its_time_and_the_probe_passes(built):
    out, recipe, dump, _ = built
    files, _ = plan("tiny", 1)
    assert [e.path for e in dump.entries] == [f.path for f in files]
    for e, f in list(zip(dump.entries, files))[::7]:
        path = out / "dump" / e.path
        assert path.read_bytes() == (source.SOURCE / f.spec["src"]).read_bytes()
        assert os.stat(path).st_mtime == f.t == e.t and not e.own_time
    assert I2.probe(recipe, dump, out, None, 1).status == "passed"


def test_a_name_or_a_text_chunk_that_says_a_task_fails_the_probe_and_a_chunk_fails_the_copy(built, tmp_path):
    out, recipe, dump, _ = built
    raw = (out / "dump" / dump.entries[0].path).read_bytes()
    text = b"tEXt" + b"Comment" + bytes(1) + b"Search for hotels in Seattle"
    marked = raw[:8] + (len(text) - 4).to_bytes(4, "big") + text + bytes(4) + raw[8:]  # (its crc is not read)
    with pytest.raises(ValueError, match="tEXt"):
        source.png_plain(marked)
    for path, content in (("Pictures/Screenshots/Search for hotels in Seattle.png", raw), ("Pictures/Screenshots/Screenshot_20261005-101010.png", marked)):
        folder = tmp_path / path.replace("/", "_")
        entries = realdump.write([realdump.File(path, "phone", 1.0, "screenshot", "image", {})], folder, 1, lambda f, p, content=content: p.write_bytes(content))
        outcome = I2.probe(recipe, realdump.dump_of("k", folder, "i2_screenshots", entries), tmp_path, None, 1)
        assert outcome.status == "failed" and ("name says" if "Seattle" in path else "metadata says") in outcome.failures[0], path


def test_a_recipe_is_refused_in_another_time_zone_and_with_other_keys():
    recipe = I2.recipe(scale="tiny", seed=1)
    with pytest.raises(ValueError, match="another dump"):
        I2.plan_of({**recipe, "utc_offsets": ["+1400", "+1400"]})
    with pytest.raises(ValueError, match="recipe has the keys"):
        I2.plan_of({**recipe, "extra": 1})
    with pytest.raises(ValueError, match="scale is one of"):
        I2.recipe(scale="huge", seed=1)


def test_no_command_may_reach_the_source_or_the_code():
    deny = I2.deny_words([], None, "k")
    assert realdump.source_word() in deny and "in:android-in-the-wild" in deny and "in:i2_screenshots" in deny and "in:from mmsb" in deny


# -- the choice of tasks and phrases, over made-up screens -----------------------------------------------------------------------------

def shot(episode: str, step: int, *lines: str, activity: str = "com.android.chrome/Main") -> source.Shot:
    return source.Shot(f"screenshots/{episode}_{step:02d}.png", episode, step, activity, tuple(lines))


def episode(id_: str, goal: str, screens: list[list[str]], length: int | None = None, activity: str = "com.android.chrome/Main") -> source.Episode:
    return source.Episode(id_, goal, "PIXEL_5", length or len(screens), tuple(shot(id_, n, *lines, activity=activity) for n, lines in enumerate(screens)))


def queries(city: str) -> list[list[str]]:
    return [["Mon, Oct 10"], ["Search or type web address"], [f"hotels in {city}", "Results", "Choose area"], [f"hotels in {city}", "Hotels"]]


def test_a_task_is_asked_about_only_where_its_goal_says_one_thing_that_is_on_its_screens():
    goal = "Search for hotels in Denver"
    pile = [episode("a", goal, queries("denver")),
            episode("cut", "Search for hotels in Paris", queries("paris"), length=9),
            episode("wrong", "Search for hotels in Miami", queries("orlando")),
            episode("twin", "Search for hotels in Seattle", queries("seattle")),
            episode("g1", "What's the latest video from GameSpot?", [["gamespot video latest"]] * 4),
            episode("g2", "What's the latest video from GameSpot eSports?", [["gamespot esports video latest"]] * 4),
            episode("t1", "What's the price of the Sony TV?", [["sony tv price"]] * 4),
            episode("t2", "How much does the Sony TV cost?", [["sony tv cost much"]] * 4),
            episode("free", "Open the calendar and tell me", [["calendar open tell"]] * 4)]
    asked = {t.goal for t, _ in tasks.asked_tasks(tasks.trim(pile, set()), None)}
    assert asked == {goal, "Search for hotels in Seattle"}, "not the cut episode, nor the one whose screens are Orlando's, nor a goal nested in another, nor one asked twice"
    assert {t.goal for t, _ in tasks.asked_tasks(pile, None, held=True)} == {"Search for hotels in Denver", "Search for hotels in Paris",
                                                                             "Search for hotels in Miami", "Search for hotels in Seattle"}


def test_a_task_with_more_than_a_few_screens_of_other_episodes_that_show_all_its_words_is_not_asked_about():
    pile = [episode("a", "Search for hotels in Denver", queries("denver"))]
    pile += [episode(f"h{n}", f"What's the weather like in City{n}", [["recent searches", "hotels in denver"]] * 2) for n in range(5)]
    pile = tasks.trim(pile, set())
    (task, shown), = tasks.asked_tasks(pile, None)
    assert shown == 10
    assert tasks.asked_tasks(pile, 9) == [] and len(tasks.asked_tasks(pile, 10)) == 1
    assert tasks.asked_tasks(pile, 0, held=True) == [], "a task that is held out has none"


def test_the_episodes_of_a_task_are_cut_to_the_screens_that_show_its_words_and_an_uncut_task_is_not_asked_about():
    """The first screens of an episode are the last page of the task before it: home screen, another search. Each episode of a task a question can rest on is
    cut to the screens that show its words (not the ones a written question is of), and a task with a screen that does not show them is asked of no one."""
    pile = [episode("a", "Search for hotels in Denver", queries("denver")), episode("b", "Search for hotels in Seattle", queries("seattle"))]
    assert tasks.asked_tasks(pile, None) == [], "a home screen and a search page are not the task's"
    cut = tasks.trim(pile, {"b"})
    a, b = cut
    assert [s.step for s in a.shots] == [2, 3] and a.length == 2 and a.whole, "the screens that show it, in the order of their steps"
    assert b == pile[1], "an episode left whole is left as it is"
    assert [(t.goal, [s.step for s in t.shots]) for t, _ in tasks.asked_tasks(cut, None)] == [("Search for hotels in Denver", [2, 3])]
    assert all(tasks.showing(s, tasks.content("Search for hotels in Denver")) for s in a.shots)


def test_a_name_a_few_letters_wrong_on_another_screen_counts_as_the_task_being_there():
    pile = [episode("a", "How much does the Vizio TV cost?", [["vizio tv cost much"]] * 4), episode("b", "Open the clock app", [["vizlo tv cost much"]] * 4)]
    assert tasks.asked_tasks(pile, 0, held=True) == [] and [s for _, s in tasks.asked_tasks(pile, None, held=True)] == [4]


def test_a_phrase_is_on_one_screen_of_the_folder_and_no_other_has_most_of_its_words():
    ordinary = {w: 9 for w in "give feedback on these results about your search".split()}
    shots = [shot("a", 0, "give feedback on these results"), shot("a", 1, "please give feedback on these resutls"),
             shot("b", 0, "your search results about this"), shot("b", 1, "send your search feedback"), shot("b", 2, "give feedback on these results")]
    assert [p.text for p in words.phrases(shots[:2], [], ordinary, True)] == []  # not in a screen that has most of its words
    one = [shot("a", 0, "give feedback on these results"), shot("b", 0, "your search about this")]
    assert [p.text for p in words.phrases(one, [], ordinary, True)] == ["give feedback on these results"]
    two = [shot("a", 0, "give feedback on these results"), shot("b", 0, "give feedback on these results")]
    assert words.phrases(two, [], ordinary, True) == [], "on two screens"
    assert [p.text for p in words.phrases(two, [], ordinary, False)] == ["give feedback on these results"], "any number among the held out"
    assert words.phrases(two, [shot("c", 0, "please give feedback on these results now")], ordinary, False) == [], "on a screen that is left"


def test_a_phrase_is_ordinary_whole_words_of_a_screen_with_no_name_in_it():
    spoken = {w: 9 for w in "send feedback about your search results".split()} | {"zzxq": 1, "tion": 9, "and": 9, "htps": 9}
    spoken |= {w: 9 for w in "the best hotels".split()} | {f"sub{n}tion": 1 for n in range(6)} | {f"x{n}tion": 1 for n in range(6)}
    for line in ("send feedback about your zzxq", "send feedback about 555 results", "send feedback about your search and",
                 "tion send feedback about your search", "send feedback about your search results htps", "send feedback", "the best hotels the best hotels in rome again now"):
        assert not words.phrases([shot("a", 0, line)], [], spoken, True), line
    assert words.phrases([shot("a", 0, "send feedback about your search results")], [], spoken, True)
    assert not words.phrases([shot("a", 0, "send feedback about your search results", activity="com.google.android.gm/Inbox")], [], spoken, True), "mail"


# -- the whole, run -----------------------------------------------------------------------------------------------------------------------

class Spy:
    """An agent that answers right in the `files` configuration, citing a screenshot of the truth, and `unknown` in any other."""

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
    world.patch.setitem(scenarios.REGISTRY, "i2_screenshots", "mmsb.scenarios.i2_screenshots")
    assert world.main("build", "--scenario", "i2_screenshots", "--scale", "tiny", "--seed", "1", "--out", str(world.out), "--temp",
                      str(world.temp), "--cache", str(world.cache), "--configs", "files,listing") == 0
    (folder,) = world.out.iterdir()
    run = RunFolder.open(folder, scenarios.get)
    assert fs.read_json(folder / "run.json")["probe"]["status"] == "passed"
    world.patch.setattr(claude, "check_login", lambda: None)
    spy = Spy(run.questions())
    assert world.main("run", str(folder), "--agents", "claude", agents=lambda: {"claude": Agent(spy)}) == 0
    prompt = next(p for (config, qid), p in spy.prompts.items() if config == "files" and qid.startswith("words-"))
    assert prompt.startswith("It is now ") and " The files of my phone's Screenshots folder are in " in prompt and "Pictures/Screenshots holds its screenshots" in prompt
    assert "<the name of the file, as in Screenshot_20260101-120000.png>" in prompt and "You have 900 seconds: " in prompt
    capsys.readouterr()
    assert world.main("report", str(folder)) == 0 and capsys.readouterr().out.startswith("# MMSB report")
    figures = fs.read_json(folder / "report.json")
    (section,) = figures["scenarios"]
    files_cell, listing = section["cells"]["claude"]["files"], section["cells"]["claude"]["listing"]
    assert (files_cell["correct"], files_cell["n"], files_cell["grounding"]["grounded"]) == (7, 7, 7)
    assert (listing["correct"], listing["n"]) == (2, 7) and section["leaks"] == [], "listing alone answered only the absent ones"
    by = defaultdict(int)
    for row in section["rows"]:
        by[row["verdicts"]["claude files"]] += 1
    assert by == {"right": 7}
