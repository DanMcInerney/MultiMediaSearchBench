"""New tests for the finding of U7's notes, over the real mm system: `Mm.ingest` reports a file it stored at a time other than its own as
`misdated`, and a run is refused for it, by name, through the core (mmsb/core/build.py, which names no system); a file it refuses is in the report with
its reason (G1's F7). The neutral half, with a stand-in
system, is tests/test_cli_build.py; this is the same route with the real `ingest` of the system that can make the report (the fake decider: no
model is loaded, and ingest loads none)."""

from __future__ import annotations

import dataclasses
import os
import shutil

import pytest

pytest.importorskip("mms")

import fakes  # noqa: E402
from PIL import Image  # noqa: E402

from mmsb.core import fs  # noqa: E402
from mmsb.core.build import Misdated, check_stores, ingest_stores, store_of  # noqa: E402
from mmsb.core.cache import INGESTED, report_of  # noqa: E402
from mmsb.core.config import Config  # noqa: E402
from mmsb.core.contract import Dump  # noqa: E402
from mmsb.systems.mm.load import remember  # noqa: E402

CONFIG = Config("mm/fake", "mm", "fake", False)


def test_a_run_over_a_real_mm_store_that_holds_a_photo_at_another_time_than_its_own_is_refused_naming_the_photo(week, tmp_path):
    _, dump, _ = week
    photo = next(e for e in dump.entries if e.kind == "photo")
    bare = tmp_path / "bare" / photo.path  # the same picture with no EXIF, as a messenger strips it: remember falls back to its mtime
    bare.parent.mkdir(parents=True)
    Image.open(dump.root / photo.path).convert("RGB").save(bare, quality=90)
    os.utime(bare, (photo.t, photo.t))
    only = Dump("bare", tmp_path / "bare", "v7_screen", (photo,), ())
    folder = tmp_path / "run"
    described = {CONFIG: fakes.MM.describe("fake")}
    with pytest.raises(Misdated) as refused:
        ingest_stores(folder, only, described, load=lambda name: fakes.MM, cache=tmp_path / "cache")
    assert "mm stored 1 files at a time other than their own" in str(refused.value) and photo.path in str(refused.value)
    assert "not at its own time" in str(refused.value)
    # the store was made and its report kept, so the next command refuses it without ingesting again
    store = store_of(folder, "mm")
    assert (store / INGESTED).is_file() and (store / "windows.json").is_file() and (store / "home").is_dir()
    assert [r.path for r in report_of(store).misdated] == [photo.path]
    with pytest.raises(Misdated, match=photo.path.replace("\\", "/")):
        check_stores(folder, {"mm": described[CONFIG].store_hash})
    with pytest.raises(Misdated):
        ingest_stores(folder, only, described, load=lambda name: pytest.fail("a store that was made is not ingested again"), cache=tmp_path / "cache")


def test_a_file_mm_refuses_is_in_the_report_with_its_reason_and_the_store_is_made_and_kept(week, tmp_path):
    """G1's F7: the daemon refuses a file (here one that is not there), `remember` collects it, and the check that the home holds the dump used to
    count it missing and raise, so the reason was lost and the next try found a store that 'did not finish'."""
    _, dump, _ = week
    note, other = [e for e in dump.entries if e.kind == "note"][:2]
    root = tmp_path / "data"
    (root / note.path).parent.mkdir(parents=True)
    shutil.copy2(dump.root / note.path, root / note.path)
    only = Dump("x", root, "v7_screen", (note, other), ())  # the second file is not there
    folder, described = tmp_path / "run", {CONFIG: fakes.MM.describe("fake")}
    ingest_stores(folder, only, described, load=lambda name: fakes.MM, cache=tmp_path / "cache")
    store = store_of(folder, "mm")
    report = report_of(store)
    assert [r.path for r in report.refused] == [other.path] and report.refused[0].reason and report.misdated == ()
    kept = fs.read_json(store / INGESTED)
    assert kept["refused"] == [{"path": other.path, "reason": report.refused[0].reason}] and kept["store_hash"] == described[CONFIG].store_hash
    # the next command finds a finished store, and does not ingest again
    ingest_stores(folder, only, described, load=lambda name: pytest.fail("a store that was made is not ingested again"), cache=tmp_path / "cache")
    check_stores(folder, {"mm": described[CONFIG].store_hash})


def test_a_file_with_the_bytes_of_an_earlier_one_is_one_item_with_it_and_in_the_report_naming_it(week, tmp_path):
    """The first mm baseline: I3's camera wrote five pairs of identical photos of one moment, and FSD50K holds one clip twice; mms left the
    second of each unchanged and the ingest stopped. Each is a refused file now, naming the first, at the same time or not. (mms keeps one
    item for a picture, sound or video of its source's bytes, whatever its time; a file of text it keeps each time.)"""
    _, dump, _ = week
    photo = next(e for e in dump.entries if e.kind == "photo")
    root = tmp_path / "data"
    (root / photo.path).parent.mkdir(parents=True)
    shutil.copy2(dump.root / photo.path, root / photo.path)
    twins = []
    for n, later in enumerate((0.0, 3600.0)):  # the same moment, and an hour on
        path = f"{photo.path.rsplit('/', 1)[0]}/twin{n}-{photo.path.rsplit('/', 1)[-1]}"
        shutil.copy2(dump.root / photo.path, root / path)
        twins.append(dataclasses.replace(photo, path=path, t=photo.t + later, name=None if photo.name is None else f"twin{n}-{photo.name}"))
    only = Dump("x", root, "v7_screen", (photo, *twins), ())
    folder, described = tmp_path / "run", {CONFIG: fakes.MM.describe("fake")}
    ingest_stores(folder, only, described, load=lambda name: fakes.MM, cache=tmp_path / "cache")
    refused = report_of(store_of(folder, "mm")).refused
    assert [(r.path, r.reason) for r in refused] == [
        (twins[0].path, f"the bytes of {photo.path} (at the same time): mms keeps one item for them"),
        (twins[1].path, f"the bytes of {photo.path} (recorded +3600 s from it): mms keeps one item for them")]
    check_stores(folder, {"mm": described[CONFIG].store_hash})


class Unchanged:
    """A daemon that leaves every file unchanged, as it does a file whose bytes it holds."""

    def call(self, op: str, **given) -> dict:
        return {"stored": [], "unchanged": [{"type": "item", "id": 1}], "source": given["source"], "expires": None}


def test_a_file_left_unchanged_with_no_earlier_file_of_its_bytes_is_still_an_error(week):
    _, dump, _ = week
    note = next(e for e in dump.entries if e.kind == "note")
    with pytest.raises(RuntimeError, match="was not stored once"):
        remember(Unchanged(), dump, [note])
