"""Where MMSB keeps things on this machine. The three folders share a volume, because a store copy hard-links the dump's files
and a hard link cannot cross volumes. Also the other places on this disk that hold the answers of runs (the tests' runs, the runs
recorded before MMSB, the main checkout of this repository), which `RunFolder.deny` names so that an agent with its own tools
cannot read them."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

RUNS = Path(r"D:\mmsb-runs")    # a run's folder: results, execs, report
TEMP = Path(r"D:\mmsb-data")    # the frozen dumps, the agents' working slots, the ffmpeg they have on PATH: for runs only (`check_temp`)
CACHE = Path(r"D:\mmsb-cache")  # dump metadata and ingested stores
TESTS = Path(r"D:\mmsb-temp")   # where the tests put their runs and fresh environments (pytest --basetemp): their questions hold answers
EARLIER = (Path(r"D:\mm-bench"), Path(r"D:\mm-bench-temp"))  # the results and the temp folder of the runs recorded before MMSB
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


def check_volume(*folders: Path) -> None:
    """Raises when `folders` are not all on one drive."""
    drives = {f.drive.casefold() for f in folders}
    if len(drives) > 1:
        raise ValueError("these folders must share a volume, so that a copy of a store can hard-link the dump: "
                         + ", ".join(str(f) for f in folders))
