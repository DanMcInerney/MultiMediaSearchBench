"""New tests for mmmb/core/cache.py (plan 6.3, DESIGN 8.4): the keys of dumps and stores, a dump or a store made once and found again, made
whole or not at all, a dump that changed refused and one that was lost written again and compared, and two commands after one key at once.
Over `cache_world.Toy`, a scenario of three files that counts its builds, and `Ingester`, a system that counts its ingests. The command
line over the cache (a build twice, a repeat, the stores of a run) is tests/test_cache_cli.py's, and V7 through mm tests/mm/test_cached_ingest.py's.

Every expected value is worked out here from independent sources: the keys from the canonical JSON written out by hand (and, for two, the
sha256 that coreutils' sha256sum gave of the same text), the hashes from the bytes on disk, the times from the files' own stats."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cache_world import FILES, RECIPE, Ingester, Toy, stamps_of
from stub_system import tiny_questions

import mmmb
from mmmb.core import fs
from mmmb.core.cache import (BUILT, BUILT_NOW, FOUND, INGESTED, MANIFEST, OUTCOME, PROBE, QUESTIONS, REWRITTEN, STATS, Busy, cached_dump,
                             cached_probe, cached_store, clone_store, dump_folder, dump_meta, locked, part_of, store_folder, store_key)
from mmmb.core.question import load_questions
from mmmb.core.runfolder import check_temp
from mmmb.core.scenario import dump_key
from mmmb.scenarios import v7_screen

KEY = dump_key(Toy.name, RECIPE, Toy.generator)
SHORT = KEY[:10]


@pytest.fixture
def where(tmp_path):
    """The folders a dump and the cache are in, none made yet."""
    return tmp_path / "temp", tmp_path / "cache"


def bytes_of(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


# -- the keys ------------------------------------------------------------------------------------------------------------------------


def test_a_dump_is_named_by_the_sha256_of_its_canonical_json_worked_out_by_hand():
    recipe = {"seed": 1, "scale": "tiny", "days": ["2026-11-01", "2026-11-02"]}
    text = '{"generator":6,"recipe":{"days":["2026-11-01","2026-11-02"],"scale":"tiny","seed":1},"scenario":"v7_screen"}'
    assert dump_key("v7_screen", recipe, 6) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert dump_key("v7_screen", recipe, 6) == "4e3ac830a40816b195b81be1da6f61d062e07f1affe73abdaab259b3c9b3e1b0", "(and sha256sum says the same)"


def test_the_recipe_the_generator_and_the_scenario_change_the_key_and_the_order_of_a_dict_does_not():
    recipe = {"seed": 1, "scale": "tiny", "days": ["2026-11-01", "2026-11-02"], "nested": {"b": [1, 2], "a": {"y": 1, "x": 2}}}
    key = dump_key("v7_screen", recipe, 6)
    shuffled = {"nested": {"a": {"x": 2, "y": 1}, "b": [1, 2]}, "days": ["2026-11-01", "2026-11-02"], "scale": "tiny", "seed": 1}
    assert dump_key("v7_screen", shuffled, 6) == key, "the order of keys, nested ones too, is no change"
    others = [dump_key("v7_screen", {**recipe, "seed": 2}, 6), dump_key("v7_screen", {**recipe, "days": recipe["days"][::-1]}, 6),
              dump_key("v7_screen", recipe, 7), dump_key("v8_egolife", recipe, 6), dump_key("v7_screen", {**recipe, "scale": "pilot"}, 6)]
    assert len({key, *others}) == 6, "a seed, the order of the days, the generator, the scenario and the scale each make another dump"


def test_v7s_recipe_holds_the_utc_offsets_of_its_days_and_they_make_another_dump():
    recipe = v7_screen.recipe(scale="tiny", seed=7, days=["2026-10-30", "2026-10-31"])
    assert set(recipe) == {"scale", "seed", "days", "utc_offsets"} and len(recipe["utc_offsets"]) == 3, "a day's offset and the one that ends the last"
    other = {**recipe, "utc_offsets": ["-0500", *recipe["utc_offsets"][1:]]}
    assert dump_key("v7_screen", other, 6) != dump_key("v7_screen", recipe, 6), "the hash of the week depends on the zone, so the dump does"


def test_a_store_is_named_by_the_sha256_of_the_system_its_version_its_store_hash_and_the_dump_worked_out_by_hand():
    system = Ingester(version="fe0ece5", store_hash="h1")
    description = dataclasses.replace(system.describe("a"), name="mm")
    text = '{"dump":"abc","store_hash":"h1","system":"mm","version":"fe0ece5"}'
    assert store_key(description, "abc") == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert store_key(description, "abc") == "3a95c9e951892d9d03bcab9a7a170b708f511e44f83b7a3463fe2ab6fa363fde", "(and sha256sum says the same)"


def test_the_version_the_store_hash_the_system_and_the_dump_each_make_another_store_and_a_variant_does_not():
    base = Ingester().describe("a")
    key = store_key(base, "abc")
    assert store_key(Ingester().describe("b"), "abc") == key, "a variant (and its config_hash) shares the store: ingest does not depend on it"
    assert store_key(dataclasses.replace(base, config_hash="another"), "abc") == key
    others = [store_key(Ingester(version="2").describe("a"), "abc"), store_key(Ingester(store_hash="h2").describe("a"), "abc"),
              store_key(dataclasses.replace(base, name="other"), "abc"), store_key(base, "abd")]
    assert len({key, *others}) == 5, "the version, the store hash, the system and the dump each make another store"


# -- a dump made once ----------------------------------------------------------------------------------------------------------------


def test_a_dump_is_built_once_and_found_again_with_nothing_rewritten(where):
    temp, cache = where
    toy = Toy()
    first = cached_dump(toy, RECIPE, temp, cache)
    before = stamps_of(temp.parent)
    second = cached_dump(toy, RECIPE, temp, cache)
    assert len(toy.builds) == 1 and (first.what, second.what) == (BUILT_NOW, FOUND), "a counting builder runs once over two calls"
    assert second.dump == first.dump and second.meta == first.meta and second.seconds == first.seconds
    assert stamps_of(temp.parent) == before, "nothing in the temp folder or the cache was written by the second call"


def test_what_a_build_leaves_is_a_frozen_dump_named_by_its_key_and_a_record_that_says_what_it_is(where):
    temp, cache = where
    made = cached_dump(Toy(), RECIPE, temp, cache)
    final, meta = temp / f"mmb-{SHORT}-data", cache / "dumps" / KEY
    assert (dump_folder(temp, KEY), dump_meta(cache, KEY)) == (final, meta) and made.dump.root == final and made.meta == meta
    assert sorted(p.name for p in temp.iterdir()) == [final.name] and sorted(p.name for p in (cache / "dumps").iterdir()) == [KEY], "no .part is left"
    assert sorted(p.relative_to(final).as_posix() for p in final.rglob("*") if p.is_file()) == sorted(FILES)
    assert all(not os.access(p, os.W_OK) for p in final.rglob("*") if p.is_file()) and fs.changed(final, meta / STATS) is None
    # the entries are the bytes on disk, measured here by hand
    for e in made.dump.entries:
        data = (final / e.path).read_bytes()
        assert (e.size, e.sha256) == (len(data), hashlib.sha256(data).hexdigest())
    assert made.dump.key == KEY and made.dump.scenario == Toy.name and [log.name for log in made.dump.logs] == ["logs"]
    # the record: entries, questions, recipe, the stats frozen with the files, the seconds it took
    assert [json.loads(line)["path"] for line in (meta / MANIFEST).read_text(encoding="utf-8").splitlines()] == list(FILES)
    assert load_questions(meta / QUESTIONS) == tiny_questions() and fs.read_json(meta / "recipe.json") == RECIPE
    assert fs.read_json(meta / STATS) == fs.stats_of(final) and fs.read_json(meta / BUILT)["seconds"] == made.seconds >= 0


def test_a_builder_that_raises_leaves_no_folder_no_part_and_no_record_and_the_next_call_builds(where):
    temp, cache = where
    for failure in (RuntimeError("disk full"), KeyboardInterrupt()):
        with pytest.raises(type(failure)):
            cached_dump(Toy(fails=failure), RECIPE, temp, cache)
        assert not (temp / f"mmb-{SHORT}-data").exists() and not (temp / f"mmb-{SHORT}-data.part").exists()
        assert not (cache / "dumps" / KEY).exists() and not (cache / "dumps" / f"{KEY}.part").exists()
    assert cached_dump(Toy(), RECIPE, temp, cache).what == BUILT_NOW, "the lock was let go"


def test_one_changed_file_refuses_naming_it_and_an_added_one_too_and_nothing_is_built_again(where):
    temp, cache = where
    toy = Toy()
    made = cached_dump(toy, RECIPE, temp, cache)
    (made.dump.root / "notes" / "a.md").chmod(stat.S_IWRITE)
    (made.dump.root / "notes" / "a.md").write_text("vandal", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"is not as it was frozen \(changed: notes/a\.md\).*escape"):
        cached_dump(toy, RECIPE, temp, cache)
    assert len(toy.builds) == 1 and (made.dump.root / "notes" / "a.md").read_text(encoding="utf-8") == "vandal", "no quiet rebuild: what changed is left to look at"
    other = {**RECIPE, "seed": 2}  # another dump, in which a file is added
    planted = cached_dump(toy, other, temp, cache).dump.root
    (planted / "extra.txt").write_text("planted", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"\(added: extra\.txt\)"):
        cached_dump(toy, other, temp, cache)
    assert len(toy.builds) == 2 and (planted / "extra.txt").read_text(encoding="utf-8") == "planted"


def test_a_dump_the_temp_folder_lost_is_written_again_compared_and_frozen_again(where, capsys):
    temp, cache = where
    toy = Toy()
    made = cached_dump(toy, RECIPE, temp, cache)
    record = bytes_of(made.meta)
    fs.remove_tree(made.dump.root)  # Windows cleaned the temp folder
    again = cached_dump(toy, RECIPE, temp, cache)
    assert f"the dump mmb-{SHORT}-data is gone from {temp}: writing it again from the recipe" in capsys.readouterr().out
    assert (again.what, len(toy.builds), again.dump) == (REWRITTEN, 2, made.dump), "the dump is what the record says, not what the rewrite measured"
    assert fs.changed(again.dump.root, made.meta / STATS) is None and fs.read_json(made.meta / STATS) == fs.stats_of(again.dump.root)
    assert {k: v for k, v in bytes_of(made.meta).items() if k != STATS} == {k: v for k, v in record.items() if k != STATS}
    assert cached_dump(toy, RECIPE, temp, cache).what == FOUND and len(toy.builds) == 2, "and it is found from then on"
    fs.remove_tree(made.dump.root / "screen")  # one file gone is the folder lost
    assert cached_dump(toy, RECIPE, temp, cache).what == REWRITTEN and (made.dump.root / "screen" / "s.mkv").is_file() and len(toy.builds) == 3


def test_data_written_again_that_is_not_the_dump_is_removed_the_record_is_left_and_every_command_gives_the_error(where):
    temp, cache = where
    made = cached_dump(Toy(), RECIPE, temp, cache)
    record = bytes_of(made.meta)
    fs.remove_tree(made.dump.root)
    for command in ("first", "second", "third"):
        with pytest.raises(RuntimeError, match=r"is not the dump mmb-\w+-data was built with \(3 files differ, the first logs/web\.jsonl\)"):
            cached_dump(Toy(content="v2"), RECIPE, temp, cache)
        assert not made.dump.root.exists() and not part_of(made.dump.root).exists(), f"the {command} command kept nothing of the other bytes"
        assert bytes_of(made.meta) == record, "the record, the stats too, is as it was"
    assert cached_dump(Toy(), RECIPE, temp, cache).what == REWRITTEN, "the generator is as it was: the data is written again, and is the dump"


def test_a_part_left_by_a_command_that_died_is_removed_aloud_and_never_taken_for_the_dump(where, capsys):
    temp, cache = where
    final, meta = dump_folder(temp, KEY), dump_meta(cache, KEY)
    for part in (part_of(final), part_of(meta)):
        part.mkdir(parents=True)
        (part / "junk.txt").write_text("a half written file", encoding="utf-8")
    made = cached_dump(Toy(), RECIPE, temp, cache)
    out = capsys.readouterr().out
    assert f"{part_of(final)} was left by a command that did not finish" in out and f"{part_of(meta)} was left by a command" in out
    assert not (final / "junk.txt").exists() and not (meta / "junk.txt").exists() and made.what == BUILT_NOW
    assert not part_of(final).exists() and not part_of(meta).exists() and sorted(p.relative_to(final).as_posix() for p in final.rglob("*") if p.is_file()) == sorted(FILES)


def test_a_command_that_died_between_the_two_renames_leaves_a_record_and_a_part_which_is_a_lost_dump(where, capsys):
    temp, cache = where
    toy = Toy()
    made = cached_dump(toy, RECIPE, temp, cache)
    os.replace(made.dump.root, part_of(made.dump.root))  # the record was renamed, and the data had not been
    again = cached_dump(toy, RECIPE, temp, cache)
    assert f"{part_of(made.dump.root)} was left by a command that did not finish" in capsys.readouterr().out
    assert again.what == REWRITTEN and len(toy.builds) == 2 and not part_of(made.dump.root).exists(), "written again, and compared with the record"


def test_data_with_no_record_is_not_taken_for_the_dump_and_is_not_removed(where):
    temp, cache = where
    toy = Toy()
    final = dump_folder(temp, KEY)
    final.mkdir(parents=True)
    (final / "mystery.txt").write_text("whose?", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"the cache has no record of it .*remove .*to write it again"):
        cached_dump(toy, RECIPE, temp, cache)
    assert toy.builds == [] and (final / "mystery.txt").is_file()


def test_the_temp_folder_that_holds_a_part_passes_the_check_that_it_holds_only_runs(where):
    temp, _ = where
    part_of(dump_folder(temp, KEY)).mkdir(parents=True)
    check_temp(temp)  # (a dump being written is `mmb-*`, like the folders of a run)


# -- two commands after one dump -----------------------------------------------------------------------------------------------------


def test_a_command_that_finds_the_lock_held_stops_with_busy_and_touches_nothing(where):
    temp, cache = where
    toy = Toy()
    with locked(cache, f"dump-{SHORT}", "the dump"), pytest.raises(Busy, match=r"another mmmb command is making the dump mmb-\w+-data .*wait for it to finish, then run this again"):
        cached_dump(toy, RECIPE, temp, cache)
    assert toy.builds == [] and not temp.exists(), "no part, no folder"
    assert cached_dump(toy, RECIPE, temp, cache).what == BUILT_NOW, "the lock is let go at the end of the block"


SCRIPT = """
import sys
from pathlib import Path
sys.path[:0] = [{tests!r}, {source!r}]
import cache_world
import mmmb
assert Path(mmmb.__file__).resolve().parents[1] == Path({source!r}).resolve(), mmmb.__file__
from mmmb.core.cache import cached_dump
toy = cache_world.Toy(started=Path({started!r}), wait_for=Path({go!r}))
cached_dump(toy, cache_world.RECIPE, Path({temp!r}), Path({cache!r}))
"""


def second_process(tmp_path: Path, temp: Path, cache: Path) -> tuple[subprocess.Popen, Path, Path]:
    """A python that builds the toy dump into `temp` and blocks half way, after its first file, until `go` exists; its `started` file says
    when it is there. Its stderr is kept in a file."""
    started, go = tmp_path / "started", tmp_path / "go"
    script = SCRIPT.format(tests=str(Path(__file__).parent), source=str(Path(mmmb.__file__).resolve().parents[1]), started=str(started), go=str(go),
                           temp=str(temp), cache=str(cache))
    process = subprocess.Popen([sys.executable, "-c", script], stderr=(tmp_path / "stderr.txt").open("w"))
    deadline = time.monotonic() + 60
    while not started.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert started.exists(), (tmp_path / "stderr.txt").read_text()
    return process, started, go


def test_a_second_process_after_the_same_dump_stops_and_leaves_the_first_ones_part_whole(where, tmp_path):
    temp, cache = where
    process, _, go = second_process(tmp_path, temp, cache)
    try:
        toy = Toy()
        with pytest.raises(Busy, match="another mmmb command is making the dump"):
            cached_dump(toy, RECIPE, temp, cache)
        part = part_of(dump_folder(temp, KEY))
        assert toy.builds == [] and (part / "notes" / "a.md").read_text(encoding="utf-8") == "a notev1", "the first command's work is where it left it"
        assert not dump_folder(temp, KEY).exists() and not dump_meta(cache, KEY).exists()
    finally:
        go.write_text("go", encoding="utf-8")
        process.wait(timeout=60)
    assert process.returncode == 0, (tmp_path / "stderr.txt").read_text()
    found = cached_dump(toy, RECIPE, temp, cache)
    assert found.what == FOUND and toy.builds == [] and fs.changed(found.dump.root, found.meta / STATS) is None
    assert {e.path: e.sha256 for e in found.dump.entries} == {rel: hashlib.sha256((text + "v1").encode()).hexdigest() for rel, text in FILES.items()}


def test_a_command_that_was_killed_leaves_a_part_and_no_lock_and_the_next_one_removes_the_part_aloud_and_builds(where, tmp_path, capsys):
    temp, cache = where
    process, _, _ = second_process(tmp_path, temp, cache)
    process.kill()
    process.wait(timeout=60)
    part = part_of(dump_folder(temp, KEY))
    assert part.is_dir(), "the killed command left its work"
    deadline = time.monotonic() + 10
    while True:  # (the operating system lets go of a dead process's lock, within a few milliseconds)
        try:
            made = cached_dump(Toy(), RECIPE, temp, cache)
            break
        except Busy:
            assert time.monotonic() < deadline, "the lock of a killed process is never let go"
            time.sleep(0.02)
    assert f"{part} was left by a command that did not finish" in capsys.readouterr().out and made.what == BUILT_NOW and not part.exists()


# -- the probe, kept with the dump ----------------------------------------------------------------------------------------------------


def test_the_probe_runs_once_over_a_dump_and_its_files_and_outcome_are_kept_for_the_next_run(where, tmp_path):
    temp, cache = where
    meta = cached_dump(Toy(), RECIPE, temp, cache).meta
    ran = []

    def probe_in(folder: Path):
        def run() -> dict:
            ran.append(folder)
            (folder / "shortcuts.json").write_text('{"seconds": 1.5}', encoding="utf-8")
            (folder / "shortcuts.md").write_text("# shortcuts", encoding="utf-8")
            return {"status": "failed", "failures": ["video-1: 3 states in scope, 250 needed"]}
        return run

    first, second = tmp_path / "run1", tmp_path / "run2"
    for folder in (first, second):
        folder.mkdir()
        (folder / "questions.jsonl").write_text("q", encoding="utf-8")
    outcome, seconds, kept = cached_probe(meta, cache, first, probe_in(first))
    assert (outcome, kept, ran) == ({"status": "failed", "failures": ["video-1: 3 states in scope, 250 needed"]}, False, [first])
    assert sorted(p.name for p in (meta / PROBE).iterdir()) == [OUTCOME, "shortcuts.json", "shortcuts.md"], "what the probe wrote, and not the run's other files"
    again, again_seconds, kept = cached_probe(meta, cache, second, probe_in(second))
    assert (again, again_seconds, kept, ran) == (outcome, seconds, True, [first]), "the probe was not run again"
    assert (second / "shortcuts.json").read_bytes() == (first / "shortcuts.json").read_bytes() and (second / "shortcuts.md").read_text(encoding="utf-8") == "# shortcuts"
    assert (second / "questions.jsonl").read_text(encoding="utf-8") == "q"


# -- a store made once ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def made(where):
    """A cached dump of the toy week, in the folders of `where`."""
    temp, cache = where
    return cached_dump(Toy(), RECIPE, temp, cache).dump, cache


def test_a_store_is_ingested_once_and_a_hit_calls_no_ingest_and_gives_the_report_it_was_made_with(made):
    dump, cache = made
    system = Ingester()
    first = cached_store(system, "a", dump, cache)
    second = cached_store(system, "a", dump, cache)
    assert len(system.ingests) == 1 and (first.hit, second.hit) == (False, True), "a store hit calls no ingest"
    assert second.report == first.report == system.report and second.path == first.path == cache / "stores" / f"sys-{store_key(system.describe('a'), dump.key)[:10]}"
    assert (store_folder(cache, system.describe("a"), store_key(system.describe("a"), dump.key))) == first.path
    assert (first.path / "index.bin").read_bytes() == b"index of " + dump.key.encode() and not part_of(first.path).exists()
    held = fs.read_json(first.path / INGESTED)
    assert held["seconds"] == 12.5 and held["refused"] == [{"path": "screen/s.mkv", "reason": "too big"}] and held["measured"] == {"decode_s": 0.024}
    assert (held["store_hash"], held["version"], held["dump"], held["dump_root"]) == ("h", "1", dump.key, str(dump.root)), "what check_stores compares"


def test_a_variant_shares_the_store_and_another_version_store_hash_or_dump_makes_another(made):
    dump, cache = made
    system = Ingester()
    a = cached_store(system, "a", dump, cache)
    assert cached_store(system, "b", dump, cache).path == a.path and len(system.ingests) == 1, "mm's ingest does not depend on the variant"
    newer, changed, other = Ingester(version="2"), Ingester(store_hash="h2"), dataclasses.replace(dump, key="f" * 64)
    paths = {cached_store(newer, "a", dump, cache).path, cached_store(changed, "a", dump, cache).path, cached_store(system, "a", other, cache).path}
    assert len(paths | {a.path}) == 4 and all(len(s.ingests) == 1 for s in (newer, changed)), "each was ingested into a store of its own"


def test_an_ingest_that_raises_leaves_no_store_and_a_part_left_by_a_dead_command_is_removed_aloud(made, capsys):
    dump, cache = made
    key = store_key(Ingester().describe("a"), dump.key)
    final = store_folder(cache, Ingester().describe("a"), key)
    with pytest.raises(RuntimeError, match="the model is gone"):
        cached_store(Ingester(fails=RuntimeError("the model is gone")), "a", dump, cache)
    assert not final.exists() and not part_of(final).exists()
    part_of(final).mkdir(parents=True)
    (part_of(final) / "junk").write_text("half", encoding="utf-8")
    system = Ingester()
    got = cached_store(system, "a", dump, cache)
    assert f"{part_of(final)} was left by a command that did not finish" in capsys.readouterr().out
    assert not (got.path / "junk").exists() and not got.hit and len(system.ingests) == 1


def test_a_store_ingested_over_the_dump_in_another_place_is_refused_and_one_being_ingested_stops_a_second_command(made, tmp_path):
    dump, cache = made
    system = Ingester()
    cached_store(system, "a", dump, cache)
    elsewhere = dataclasses.replace(dump, root=tmp_path / "other" / dump.root.name)
    with pytest.raises(RuntimeError, match=r"was ingested over the dump in .* and this dump is in .*use the same --temp"):
        cached_store(system, "a", elsewhere, cache)
    fresh = Ingester(store_hash="h2")
    name = store_folder(cache, fresh.describe("a"), store_key(fresh.describe("a"), dump.key)).name
    with locked(cache, f"store-{name}", "the store"), pytest.raises(Busy, match=f"another mmmb command is making the store {name}"):
        cached_store(fresh, "a", dump, cache)
    assert fresh.ingests == []


def test_a_store_is_copied_with_hard_links_whole_or_not_at_all(made, tmp_path):
    dump, cache = made
    got = cached_store(Ingester(), "a", dump, cache)
    into = tmp_path / "run" / "stores" / "sys"
    part_of(into).mkdir(parents=True)
    (part_of(into) / "junk").write_text("half", encoding="utf-8")
    clone_store(got.path, into)
    assert sorted(p.name for p in into.iterdir()) == sorted([INGESTED, "index.bin"]) and not part_of(into).exists()
    for name in (INGESTED, "index.bin"):
        assert (into / name).read_bytes() == (got.path / name).read_bytes() and os.stat(into / name).st_nlink == 2 == os.stat(got.path / name).st_nlink
    assert os.path.samefile(into / "index.bin", got.path / "index.bin"), "one file on disk, two names"
