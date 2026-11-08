"""Oracle 1c: the V7 pilot's whole week built with no Multimem, and compared with the recorded pilot's. All 1,575 files (1,572 and 3 logs)
have the size, mtime and sha256 the pilot's had; the shortcut probe over the build passes, as the pilot's did, and writes the
`shortcuts.json` and `shortcuts.md` the pilot's did, but for the seconds it took.

    uv run pytest -m slow tests/oracles/test_v7_full.py --v7-full=D:/mmsb-temp/oracle1c -n 0      # about 8 minutes, at 3 workers

builds the week in `FOLDER/data` (a build that is already there is read, not made again), runs the probe into `FOLDER/run` and compares.
`--v7-workers=N` is the files written at once (the pilot's build wrote 3). The bytes hold only with the libraries pyproject pins and in the
time zone the pilot was made in (see v7_recorded.py); without `--v7-full` the oracle skips and says what to run.

Expected values are the pilot's own files (tests/oracles/recorded/pilot-s1), captured before the code was moved."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import imageio_ffmpeg
import pytest
from v7_recorded import MANIFEST, META, RECORDED, SHA256, V7, needs_the_recorded_zone, recipe

from mmsb.core.contract import Dump
from mmsb.core.scenario import ProbeOutcome, dump_key
from mmsb.scenarios.v7_screen.manifest import LOGS, entries_of, measure_disk

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("v7-full")]  # one worker builds it
needs_the_recorded_zone()


@pytest.fixture(scope="module")
def folder(request) -> Path:
    given = request.config.getoption("--v7-full")
    if not given:
        pytest.skip("oracle 1c builds the pilot's whole week (about 8 minutes) and compares it: pytest -m slow tests/oracles/test_v7_full.py "
                    "--v7-full=D:/mmsb-temp/oracle1c -n 0")
    return Path(given)


@pytest.fixture(scope="module")
def built(folder, request) -> Dump:
    data, r = folder / "data", recipe()
    if data.exists():
        return Dump(dump_key(V7.name, r, V7.generator), data, V7.name, entries_of(V7.plan_of(r), measure_disk(data)), (LOGS,))
    began = time.perf_counter()
    dump, _ = V7.build(r, data, request.config.getoption("--v7-workers"))
    print(f"\nbuilt {len(dump.entries)} files in {time.perf_counter() - began:.0f} s at {request.config.getoption('--v7-workers')} workers")
    return dump


def test_all_1575_files_are_the_recorded_ones_by_size_mtime_and_sha256(built):
    assert {e.path for e in built.entries} == set(MANIFEST) == set(SHA256) and len(built.entries) == 1575
    on_disk = {p.relative_to(built.root).as_posix() for p in built.root.rglob("*") if p.is_file()}
    assert on_disk == set(MANIFEST), f"files that are not the pilot's: {sorted(on_disk - set(MANIFEST))[:5]}, missing: {sorted(set(MANIFEST) - on_disk)[:5]}"
    wrong = [(e.path, (e.size, e.mtime_ns, e.sha256), (*MANIFEST[e.path], SHA256[e.path])) for e in built.entries
             if (e.size, e.mtime_ns, e.sha256) != (MANIFEST[e.path][0], MANIFEST[e.path][1], SHA256[e.path])]
    print(f"\n{len(built.entries)} files compared with the pilot's, {len(wrong)} differ")
    assert not wrong, f"{len(wrong)} of 1575 files differ from the pilot's (path, got, recorded): {wrong[:3]}"


def test_the_probe_over_the_build_passes_and_reports_what_the_pilots_did_but_for_the_seconds(built, folder):
    ffmpeg = folder / "bin" / "ffmpeg.exe"  # the files arm's ffmpeg is a copy of this, named so
    if not ffmpeg.exists():
        ffmpeg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    outcome = V7.probe(recipe(), built, folder / "run", ffmpeg, META["seed"])
    assert META["probe"] == {"status": "passed", "failures": []} and outcome == ProbeOutcome("passed", ())
    got = json.loads((folder / "run" / "shortcuts.json").read_text(encoding="utf-8"))
    want = json.loads((RECORDED / "shortcuts.json").read_text(encoding="utf-8"))
    seconds = got.pop("seconds"), want.pop("seconds")
    print(f"\nthe probe took {seconds[0]} s (the pilot's: {seconds[1]} s)")
    assert got == want
    by_seconds = lambda text: re.sub(r"questions in \d+(\.\d+)? s;", "questions in <seconds> s;", text)  # noqa: E731
    assert by_seconds((folder / "run" / "shortcuts.md").read_text(encoding="utf-8")) == by_seconds((RECORDED / "shortcuts.md").read_text(encoding="utf-8"))
