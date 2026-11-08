"""Ported from MultimediaSearch tests/test_bench.py at fe0ece5 (the junction, frozen-folder and permissions tests), with `bench.` now `fs.`;
the tests marked new are not from there. Files and folders: JSON written whole, deletion of read-only files and links, the frozen
data folder."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from mmsb.core import fs


@pytest.mark.skipif(os.name != "nt", reason="a directory junction is a Windows link")
def test_remove_tree_removes_a_junction_and_never_what_it_points_to(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    work = tmp_path / "work"
    work.mkdir()
    (work / "mine.txt").write_text("mine")

    def junction(link: Path) -> Path:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
        assert link.is_junction() and not link.is_symlink()
        return link

    fs.remove_tree(junction(work / "link"))  # as `empty(work)` meets one an agent made: it is the child itself
    assert not (work / "link").exists() and (target / "keep.txt").read_text() == "keep" and (work / "mine.txt").exists()
    junction(work / "link")
    fs.remove_tree(work)  # and one met inside a tree
    assert not work.exists() and (target / "keep.txt").read_text() == "keep"
    work.mkdir()
    junction(work / "again")
    fs.empty(work)
    assert list(work.iterdir()) == [] and (target / "keep.txt").exists()


def test_a_frozen_folder_names_a_change_an_addition_and_a_removal_and_a_link_is_removed_as_a_name(tmp_path):
    def frozen(name: str):
        folder = tmp_path / name
        (folder / "a").mkdir(parents=True)
        (folder / "a" / "one.txt").write_text("one")
        (folder / "two.txt").write_text("two")
        fs.freeze(folder, tmp_path / f"{name}.json")
        assert fs.changed(folder, tmp_path / f"{name}.json") is None and not os.access(folder / "two.txt", os.W_OK)
        return folder, tmp_path / f"{name}.json"

    folder, manifest = frozen("edited")
    (folder / "two.txt").chmod(stat.S_IWRITE)
    (folder / "two.txt").write_text("twenty")
    assert fs.changed(folder, manifest) == "changed: two.txt"
    folder, manifest = frozen("added")
    (folder / "a" / "three.txt").write_text("3")
    assert fs.changed(folder, manifest) == "added: a/three.txt"
    folder, manifest = frozen("removed")
    fs.remove_tree(folder / "a")
    assert fs.changed(folder, manifest) == "gone: a/one.txt"

    # removing a tree that holds a link to a read-only file (a home's media, a clip an agent linked) deletes the name and leaves
    # the file and its other links read-only; a read-only file with no other link is still removed
    folder, manifest = frozen("linked")
    link = tmp_path / "copy" / "one.txt"
    link.parent.mkdir()
    os.link(folder / "a" / "one.txt", link)
    fs.remove_tree(link.parent)
    assert not link.parent.exists() and not os.access(folder / "a" / "one.txt", os.W_OK) and fs.changed(folder, manifest) is None
    os.link(folder / "two.txt", tmp_path / "two-link.txt")
    fs.remove_tree(tmp_path / "two-link.txt")
    assert not (tmp_path / "two-link.txt").exists() and not os.access(folder / "two.txt", os.W_OK)
    fs.unlink(folder / "two.txt")
    assert not (folder / "two.txt").exists(), "unlink deletes a read-only file too"
    fs.remove_tree(folder / "a")
    assert not (folder / "a").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows permissions")
def test_every_data_file_is_left_with_the_permissions_its_folder_gives_so_codexs_sandbox_reads_it(tmp_path):
    # the pilot's screen, porch and mic files were moved out of a Python temporary folder, kept its owner-only permissions,
    # and Codex's sandbox could not open one: its files arm answered "unknown" to every video question
    import tempfile

    # the data folder sits right in TEMP, as a run's does: pytest's tmp_path is owner-only itself, which hides the fault
    data, work = Path(tempfile.gettempdir()) / f"mmb-acl-test-{os.getpid()}-{time.time_ns()}", Path(tempfile.mkdtemp(prefix="acl-"))
    try:
        (data / "screen").mkdir(parents=True)
        (data / "notes").mkdir()
        (work / "moved.mkv").write_bytes(b"x")
        shutil.move(str(work / "moved.mkv"), str(data / "screen" / "moved.mkv"))
        (work / "made.mkv").write_bytes(b"y")
        shutil.copyfile(work / "made.mkv", data / "screen" / "made.mkv")  # what the generator's `_finish` does: copy in, then drop
        (work / "made.mkv").unlink()
        (data / "notes" / "note.md").write_text("n", encoding="utf-8")
        assert fs.own_permissions(data) == ["screen/moved.mkv"], "a moved file keeps its folder's; a copied or new one does not"
        assert not (work / "made.mkv").exists() and (data / "screen" / "made.mkv").read_bytes() == b"y"
        fs.freeze(data, tmp_path / "manifest.json")
        assert fs.own_permissions(data) == [] and fs.changed(data, tmp_path / "manifest.json") is None
        assert not os.access(data / "screen" / "moved.mkv", os.W_OK), "still read-only"
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if data.exists():
            fs.remove_tree(data)



# -- new ----------------------------------------------------------------------------------------------------------------

def test_json_is_written_whole_or_not_at_all(tmp_path):
    path = tmp_path / "deep" / "folder" / "data.json"
    fs.write_json(path, {"a": [1, 2.5, None], "b": "é"})
    assert fs.read_json(path) == {"a": [1, 2.5, None], "b": "é"}
    with pytest.raises(TypeError):
        fs.write_json(path, {"a": object()})
    assert fs.read_json(path) == {"a": [1, 2.5, None], "b": "é"}, "a write that failed leaves the file as it was"
    assert sorted(p.name for p in path.parent.iterdir()) == ["data.json"], "and no half-written part"
    fs.write_text(path, "plain")
    assert path.read_text(encoding="utf-8") == "plain"


def test_a_write_that_fails_half_way_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    fs.write_text(path, "old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fs.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        fs.write_text(path, "new and longer")
    assert path.read_text(encoding="utf-8") == "old", "the new text is in a part until it is whole"


def test_sha256_of_is_hashlibs_whatever_the_size_including_the_block_boundary(tmp_path):
    block = 1 << 20
    rng = random.Random(7)
    for size in (0, 1, block - 1, block, block + 1, 3 * block + 5):
        path = tmp_path / f"{size}.bin"
        path.write_bytes(rng.randbytes(size))
        assert fs.sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest(), size


def test_a_slug_keeps_letters_digits_underscore_and_hyphen_and_turns_the_rest_into_hyphens():
    assert fs.slug("pplx-decider-flash-v1.2") == "pplx-decider-flash-v1-2"
    assert fs.slug("d1_3B-ok") == "d1_3B-ok" and fs.slug("mm/d1-3B+files é") == "mm-d1-3B-files--"


def test_stats_of_names_each_file_by_its_posix_path_with_its_size_and_mtime(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.txt").write_bytes(b"abc")
    (tmp_path / "top.txt").write_bytes(b"")
    assert fs.stats_of(tmp_path) == {"a/b/c.txt": [3, os.stat(tmp_path / "a" / "b" / "c.txt").st_mtime_ns],
                                     "top.txt": [0, os.stat(tmp_path / "top.txt").st_mtime_ns]}
    fresh = tmp_path / "new" / "folder"
    fs.empty(fresh)
    assert fresh.is_dir() and list(fresh.iterdir()) == []
