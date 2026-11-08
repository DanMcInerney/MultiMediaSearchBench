"""New tests for the cache over the real mm system (plan 6.3, DESIGN 8.4; core/cache.py and core/build.py are tests/test_cache.py's and
tests/test_cache_cli.py's, over stand-ins): the tiny week of V7 built into two runs through `mm/fake`, with the fake decider (no model is loaded,
and ingest loads none). The dump is written once and the store ingested once, by a counter, the second run finds both; the dump's bytes are the
bytes of the plain build the other tests use; the run's copy of the store is the cached store's files under other names, so a session over it
(which copies the home and starts a daemon on the copy) writes nothing through them; and two runs over one dump keep their own homes."""

from __future__ import annotations

import hashlib
import os
import types

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from cache_world import stamps_of  # noqa: E402

from mmsb.core import fs  # noqa: E402
from mmsb.core.build import build_run, read_dump, store_of  # noqa: E402
from mmsb.core.cache import INGESTED, dump_folder  # noqa: E402
from mmsb.core.runfolder import RunFolder  # noqa: E402
from mmsb.core.scenario import dump_key  # noqa: E402
from mmsb.systems.mm.daemon import Api  # noqa: E402
from mmsb.systems.mm.system import Mm  # noqa: E402

pytestmark = pytest.mark.xdist_group("v7-mm")


class Counting(Mm):
    """The system `fakes.MM` is (mm with the fake decider), counting its ingests."""

    def __init__(self) -> None:
        super().__init__(models=fakes.MM.models)
        self.ingests = 0

    def ingest(self, dump, store):
        self.ingests += 1
        return super().ingest(dump, store)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs of the tiny week of seed 7 through `mm/fake`, built one after the other into one cache: `(first, second, system, root)`."""
    root = tmp_path_factory.mktemp("cached-ingest")
    system, options = Counting(), {"scale": "tiny", "seed": 7, "days": fakes.DAYS}
    def made_up_bin(self) -> None:  # (the writers need the real ffmpeg: only the copy for the agents is made up)
        self.bin.mkdir(parents=True, exist_ok=True)
        self.bin_exe.write_bytes(b"x")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RunFolder, "ensure_bin", made_up_bin)
        built = [build_run(fakes.V7, options, root / f"runs-{n}", root / "temp", [fakes.S], timeout_s=900.0, cache=root / "cache", probe=False,
                           load=lambda name: system) for n in (1, 2)]
    return types.SimpleNamespace(first=built[0], second=built[1], system=system, root=root)


def test_the_tiny_week_through_mm_is_written_once_and_ingested_once_over_two_runs(runs):
    first, second = runs.first, runs.second
    assert runs.system.ingests == 1, "the second run found the store the first made"
    key = dump_key("v7_screen", first.recipe, fakes.V7.generator)
    assert first.data == second.data == dump_folder(runs.root / "temp", key) and fs.read_json(second.folder / "run.json")["dump_key"] == key
    assert sorted(p.name for p in (runs.root / "temp").iterdir() if p.name.endswith("-data")) == [first.data.name], "one dump in the temp folder"
    assert first.tag != second.tag and first.slot(0) != second.slot(0)
    for name in (INGESTED, "windows.json", "home/index.sqlite"):
        assert (store_of(second.folder, "mm") / name).read_bytes() == (store_of(first.folder, "mm") / name).read_bytes(), name
        assert os.stat(store_of(first.folder, "mm") / name).st_nlink == 3, f"{name}: the cached store and the two runs hold one file"
    assert fs.read_json(second.folder / "run.json")["build_s"] == fs.read_json(first.folder / "run.json")["build_s"] > 0, "the true time of writing the week"
    assert fs.read_json(store_of(second.folder, "mm") / INGESTED)["seconds"] > 0, "and of ingesting it"


def test_the_dump_a_cached_build_froze_is_the_plain_builds_bytes_and_frozen(runs, week):
    _, plain, _ = week  # (V7.build of the same recipe, without the cache: the other tests' week)
    first = runs.first
    dump = read_dump(first)
    assert {e.path: (e.size, e.sha256) for e in dump.entries} == {e.path: (e.size, e.sha256) for e in plain.entries}, "the entries"
    for e in dump.entries:  # the bytes on disk, hashed here
        assert hashlib.sha256((first.data / e.path).read_bytes()).hexdigest() == e.sha256, e.path
    assert fs.changed(first.data, first.manifest) is None and not os.access(first.data / dump.entries[0].path, os.W_OK)


def test_sessions_over_the_runs_of_one_dump_copy_the_store_into_homes_of_their_own_and_write_nothing_through_the_links(runs):
    first, second, system = runs.first, runs.second, runs.system
    cache = runs.root / "cache"
    stores_before = stamps_of(cache)
    dump = read_dump(first)
    homes = []
    for run in (first, second):
        with fakes.Opened(store_of(run.folder, "mm"), dump, run.folder / "homes", system=system) as session:
            homes.append(session.home)
            sources = {s["name"]: s for s in Api(session.home).call("status")["sources"]}
            assert "logs" in sources and sources["logs"]["records"] > 0, "the daemon over the copy of the cached store reads the week's logs"
    assert homes[0] != homes[1] and all(h.is_dir() for h in homes) and homes[0].parent == first.folder / "homes"
    assert not os.path.samefile(homes[0] / "index.sqlite", store_of(first.folder, "mm") / "home" / "index.sqlite"), "a home's index is its own copy"
    assert stamps_of(cache) == stores_before, "the cached store was not written through"
    assert runs.system.ingests == 1
