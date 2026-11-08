"""The listing refuses a dump whose files are not at the times they hold (G2 review, finding 2). Its only time is a file's modification time, which
the question's horizon (`--until`) cuts by; V8's clips are hard links to the download, whose modification times are the download's, days after the
recordings, so the horizon hid every clip from every question and the agent was given an empty listing (and a download that kept the server's dates
would show clips past the horizon). `Listing.ingest` compares each file's mtime with the manifest's `t` and reports a file that is not at its time as
`misdated`, which refuses the run in every command (core/build.py). The recorded V7 pilot is held to the same rule in tests/oracles/test_listing_pilot_times.py.

Expected values are what a test planted (a file moved in time by an amount, a download made today of recordings from last week) and the tolerance the
rule states (one second, whole seconds of a file system), not the code's."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from cli_world import agents_of, world  # noqa: F401
from listing_world import planted_dump
from stub_system import Stub
from v8_fake import built, oracle_file, recipe, source  # noqa: F401

from mmsb.core.agents import claude
from mmsb.core.build import Misdated, refuse_misdated
from mmsb.core.contract import ManifestEntry
from mmsb.systems.listing import read as reading
from mmsb.systems.listing.system import Listing

pytestmark = pytest.mark.xdist_group("v8")
DAY = 86400.0
DAYS = "2026-10-29,2026-10-30"


def entry(media: str, t: float) -> ManifestEntry:
    return ManifestEntry("x/y", "x", t, "k", media, None, False, 1, 1, "00")


def test_a_file_is_at_its_time_within_a_second_either_way_and_a_log_only_must_not_be_before_its_first_line():
    t = 1_700_000_000.0
    for media in ("image", "video", "audio", "text"):
        for moved in (0.0, 0.9, -0.9, 1.0, -1.0):
            assert reading.misdated_reason(entry(media, t), t + moved) is None, (media, moved)
        for moved in (1.1, -1.1, 60.0, -60.0, 3 * DAY, -3 * DAY):
            assert reading.misdated_reason(entry(media, t), t + moved), (media, moved)
    # a log's `t` is the time of its first line and its mtime the time of its last (V7 sets it so): later is right, earlier than the first is not
    for moved in (0.0, 0.5, -0.9, 60.0, 3 * DAY):
        assert reading.misdated_reason(entry("records", t), t + moved) is None, moved
    for moved in (-1.1, -60.0, -3 * DAY):
        assert reading.misdated_reason(entry("records", t), t + moved), moved


def test_a_dump_with_one_file_moved_in_time_either_way_is_refused_for_that_file_and_the_reason_names_both_times(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    dump = planted_dump(root)
    notes = next(e for e in dump.entries if e.media == "text" and e.path.startswith("notes/"))
    for moved in (-3 * DAY, 3 * DAY, 2.0):
        mtime = notes.t - 0.4375 + moved  # (the planted entries are 0.4375 s after their file)
        os.utime(root / notes.path, (mtime, mtime))
        report = Listing().ingest(dump, tmp_path / f"store{moved}")
        assert [m.path for m in report.misdated] == [notes.path], moved
        reason = report.misdated[0].reason
        assert f"modified {datetime.fromtimestamp(mtime).astimezone().isoformat(timespec='seconds')}" in reason, reason
        assert f"the dump says it is from {datetime.fromtimestamp(notes.t).astimezone().isoformat(timespec='seconds')}" in reason, reason
        assert "the listing's only time is the modification time" in reason
        with pytest.raises(Misdated, match=f"listing stored 1 files at a time other than their own.*{notes.path}"):
            refuse_misdated("listing", report)
    os.utime(root / notes.path, (notes.t - 0.4375 + 0.5, notes.t - 0.4375 + 0.5))  # half a second is a file system's rounding: in
    assert Listing().ingest(dump, tmp_path / "store-in").misdated == ()


def test_a_log_whose_mtime_is_before_its_first_line_is_refused_and_one_after_it_is_not(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    dump = planted_dump(root)
    log = next(e for e in dump.entries if e.media == "records")
    for moved, refused in ((5 * DAY, False), (-5 * DAY, True)):
        mtime = log.t - 0.4375 + moved
        os.utime(root / log.path, (mtime, mtime))
        shutil.rmtree(tmp_path / "store", ignore_errors=True)
        assert [m.path for m in Listing().ingest(dump, tmp_path / "store").misdated] == ([log.path] if refused else [])


# -- V8: the download's times ----------------------------------------------------------------------------------------------------

def test_a_v8_dump_is_refused_for_its_clips_which_carry_the_download_s_times_and_not_for_its_transcripts(built, tmp_path):  # noqa: F811
    dump, _ = built
    clips = [e for e in dump.entries if e.kind == "clip"]
    assert len(clips) == 24 and min(os.stat(dump.root / e.path).st_mtime for e in clips) > max(e.t for e in clips) + DAY, \
        "the fake download was made after the days it holds, as the real one was (3 November, against 29 to 31 October)"
    report = Listing().ingest(dump, tmp_path / "store")
    assert sorted(m.path for m in report.misdated) == sorted(e.path for e in clips), "every clip, and no transcript: its mtime is after its first line"
    with pytest.raises(Misdated, match="listing stored 24 files at a time other than their own") as refused:
        refuse_misdated("listing", report)
    assert "but the dump says it is from 2026-10-29" in str(refused.value) and "so a horizon would cut it at a wrong time" in str(refused.value)


def build_args(world, source, configs: str, out: Path):  # noqa: F811
    return ["build", "--scenario", "v8_egolife", "--source", str(source), "--days", DAYS, "--per-age", "2", "--speech", "2", "--out", str(out),
            "--temp", str(world.temp), "--cache", str(world.cache), "--configs", configs]


def test_building_v8_with_the_listing_is_refused_by_name_and_so_is_ingesting_it_later_and_running_over_that_store(world, source, oracle_file, capsys):  # noqa: F811
    world.patch.setattr(claude, "check_login", lambda: pytest.fail("refused before an agent's login"))
    # at the build (the run is not made, so there is nothing to run)
    assert world.main(*build_args(world, source, "files,listing", world.out)) == 1
    err = capsys.readouterr().err
    assert "listing stored 24 files at a time other than their own" in err and "the listing's only time is the modification time" in err
    assert not list(world.out.glob("*/run.json")), "no run was made"
    # a run built without it, then ingested: refused the same, and the store that was made is refused by every command that would use it
    assert world.main(*build_args(world, source, "files", world.out.parent / "runs-files")) == 0
    capsys.readouterr()
    (folder,) = (world.out.parent / "runs-files").iterdir()
    assert world.main("ingest", str(folder), "--configs", "listing") == 1
    assert "listing stored 24 files at a time other than their own" in capsys.readouterr().err
    stub = Stub()
    for flags in ((), ("--dry-run",)):
        assert world.main("run", str(folder), "--agents", "claude", "--configs", "files,listing", *flags, agents=agents_of(stub)) == 1, flags
        assert "listing stored 24 files at a time other than their own" in capsys.readouterr().err
    assert stub.execs == [] and not (folder / "results").exists()
