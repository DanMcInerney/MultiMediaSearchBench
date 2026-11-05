"""Oracle 1b: a sample of the V7 pilot's files, written with no Multimem and compared with the files the recorded pilot wrote. 43 files:
the rule is `random.Random(7)`, then in this order `sample(files of the kind, n)` for screen 12, photo 12, porch 6, mic 4, note 4, clip 3
and config 2. Each is written by `write_file` (which checks it as it makes it) and must have the recorded size, mtime and sha256: the
same bytes, which holds only with the libraries pyproject pins. About 10 s. Run with `-m slow`.

Expected values are the pilot's own files (tests/oracles/recorded/pilot-s1), captured before the code was moved. The oracle holds only in
the time zone the pilot was made in (see v7_recorded.py): elsewhere it skips, and says so."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest
from v7_recorded import MANIFEST, SHA256, V7, needs_the_recorded_zone, recipe

from mmmb.core import fs
from mmmb.scenarios.v7_screen.write import write_file

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("v7-sample")]
needs_the_recorded_zone()
SAMPLE = (("screen", 12), ("photo", 12), ("porch", 6), ("mic", 4), ("note", 4), ("clip", 3), ("config", 2))


def test_43_of_the_pilots_files_are_written_with_the_bytes_the_pilot_wrote(tmp_path):
    plan = V7.plan_of(recipe())
    r = random.Random(7)
    sample = [f for kind, n in SAMPLE for f in r.sample([f for f in plan.files if f.kind == kind], n)]
    assert len(sample) == 43 == len({f.path for f in sample})
    with ThreadPoolExecutor(3) as pool:
        for future in [pool.submit(write_file, f, tmp_path) for f in sample]:
            future.result()
    wrong = []
    for f in sample:
        info = (tmp_path / f.path).stat()
        got = (info.st_size, info.st_mtime_ns, fs.sha256_of(tmp_path / f.path))
        if got != (MANIFEST[f.path][0], MANIFEST[f.path][1], SHA256[f.path]):
            wrong.append((f.path, got, (*MANIFEST[f.path], SHA256[f.path])))
    assert not wrong, f"{len(wrong)} of 43 files differ from the pilot's (path, got, recorded): {wrong[:3]}"
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == {f.path for f in sample}, "and no work file is left"
