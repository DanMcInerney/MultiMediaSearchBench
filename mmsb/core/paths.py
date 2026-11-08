"""Where MMSB keeps things on this machine, all under one home (`MMSB_HOME`, else the user's home folder). The three working folders share
a volume, because a store copy hard-links the dump's files and a hard link cannot cross volumes. Also the other places on this disk that
hold the answers of runs (the tests' runs, the runs recorded before MMSB, the main checkout of this repository), which `RunFolder.deny`
names so that an agent with its own tools cannot read them."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

HOME = Path(os.environ.get("MMSB_HOME") or Path.home())  # the folder the others are in
RUNS = HOME / "mmsb-runs"      # a run's folder: results, execs, report
TEMP = HOME / "mmsb-data"      # the frozen dumps, the agents' working slots, the ffmpeg they have on PATH: for runs only (`check_temp`)
CACHE = HOME / "mmsb-cache"    # dump metadata and ingested stores
TESTS = HOME / "mmsb-temp"     # where the tests put their runs and fresh environments (pytest --basetemp): their questions hold answers
EARLIER = (HOME / "mm-bench", HOME / "mm-bench-temp")  # the results and the temp folder of the runs recorded before MMSB
ROOT = Path(__file__).resolve().parents[2]  # this repository: the agents may not read it


@functools.cache
def main_checkout(root: Path) -> Path | None:
    """The main checkout of the repository that `root` is a checkout of (the parent of its git common folder): `root` itself, or the
    checkout a linked worktree was made from. None when `root` is not a checkout (an installed copy)."""
    if not (root / ".git").exists():
        return None
    common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True).stdout.strip()
    return (root / common).resolve().parent


def checkouts() -> tuple[Path, ...]:
    """This repository's folders that hold recorded answers (`tests/oracles/recorded`): this checkout and, when it is a linked
    worktree, the main checkout as well."""
    main = main_checkout(ROOT)
    return (ROOT,) if main is None or main == ROOT else (ROOT, main)


def check_apart(out: Path, temp: Path) -> None:
    """Raises when the runs folder `out` is, or holds, the temp folder `temp`. An agent with its own tools is refused the folder its run is
    in (the questions of the runs beside it are there), and that would be the folder of its own dump, ffmpeg and working folders. A
    drive's root (`--out D:/`) holds every folder of its drive, so it is refused as well."""
    out, temp = Path(out).resolve(), Path(temp).resolve()
    if temp == out or temp.is_relative_to(out):
        raise ValueError(f"--out {out} holds --temp {temp}: an agent is refused the folder a run is in, so it would be refused its own data, "
                         "ffmpeg and working folders. Give --out a folder of its own (not your home folder or a drive's root), or --temp one outside it")


def check_volume(*folders: Path) -> None:
    """Raises when `folders` are not all on one drive."""
    drives = {f.drive.casefold() for f in folders}
    if len(drives) > 1:
        raise ValueError("these folders must share a volume, so that a copy of a store can hard-link the dump: "
                         + ", ".join(str(f) for f in folders))
