"""The listing's rule for a file's time (mmmb/systems/listing/read.py `misdated_reason`) held to the recorded V7 pilot, whose files the generator set to
the times they hold: all 1,575 entries of its recorded manifest are at their time, and the tolerance of one second is what separates them from a dump
whose times are the download's. The entries are the plan's `t`, and the times are the mtimes the pilot's own data folder had (`data-manifest.json`),
both from the files in tests/oracles/recorded/pilot-s1, with no data folder and no ingest. Its plan is local time, so this skips outside the pilot's zone."""

from __future__ import annotations

from v7_recorded import MANIFEST, V7, needs_the_recorded_zone, recipe

from mmmb.scenarios.v7_screen.manifest import entries_of
from mmmb.systems.listing.read import TIME_TOL_S, misdated_reason

needs_the_recorded_zone()
ENTRIES = entries_of(V7.plan_of(recipe()), lambda path: (MANIFEST[path][0], MANIFEST[path][1], "0" * 64))
MTIME = {e.path: e.mtime_ns / 1e9 for e in ENTRIES}
FILES = [e for e in ENTRIES if e.media != "records"]
LOGS = [e for e in ENTRIES if e.media == "records"]


def test_the_recorded_pilot_has_1572_files_exactly_at_their_time_and_3_logs_whose_mtime_is_their_last_line_s():
    assert (len(ENTRIES), len(FILES), len(LOGS)) == (1575, 1572, 3)
    assert max(abs(MTIME[e.path] - e.t) for e in FILES) < 1e-6, "the generator's os.utime(path, (t, t)), as the recorded manifest holds it"
    assert sorted(round(MTIME[e.path] - e.t) for e in LOGS) == [237409, 237415, 237435], "a log's t is its first line's time, its mtime its last's"
    assert TIME_TOL_S == 1.0


def test_every_recorded_pilot_entry_is_at_its_time_and_a_dump_whose_files_are_a_little_off_or_the_downloads_is_not():
    assert [e.path for e in ENTRIES if misdated_reason(e, MTIME[e.path])] == [], "the V7 pilot still ingests"
    # every file moved by so much: the files refused, and the logs refused (they are only refused for a mtime before their first line, which is
    # 237,409 s or more before their last: three days back puts all three there)
    for moved, files, logs in ((0.9, 0, 0), (-0.9, 0, 0), (1.1, 1572, 0), (-1.1, 1572, 0), (86400.0, 1572, 0), (-259200.0, 1572, 3)):
        refused = [e for e in ENTRIES if misdated_reason(e, MTIME[e.path] + moved)]
        assert (len([e for e in refused if e.media != "records"]), len([e for e in refused if e.media == "records"])) == (files, logs), moved
