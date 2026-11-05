"""New tests for the finding of U7's notes, over the real mm system: `Mm.ingest` reports a file it stored at a time other than its own as
`misdated`, and a run is refused for it, by name, through the core (mmmb/core/build.py, which names no system); a file it refuses is in the report with
its reason (G1's F7). The neutral half, with a stand-in
system, is tests/test_cli_build.py; this is the same route with the real `ingest` of the system that can make the report (the fake decider: no
model is loaded, and ingest loads none)."""

from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from PIL import Image  # noqa: E402

from mmmb.core import fs  # noqa: E402
from mmmb.core.build import Misdated, check_stores, ingest_stores, store_of  # noqa: E402
from mmmb.core.cache import INGESTED, report_of  # noqa: E402
from mmmb.core.config import Config  # noqa: E402
from mmmb.core.contract import Dump  # noqa: E402

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
