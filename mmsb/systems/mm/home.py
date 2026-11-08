"""An mms home and what a run does to it (moved from `copy_home`, `kept_items` and `scrub_home` of bench.py and `set_horizon` of
egolife.py of this repository at bb498a6): a home per (agent, variant) copied from the ingested store, the baseline of what the
execs may keep, the sweep that takes an exec's state out of it, and the cut of its index to a horizon. Nothing here starts a daemon
or reads the owner's ~/.mms: a home is the folder it is given."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from mms import config

from mmsb.core.fs import remove_tree
from mmsb.systems.mm.models import Model

# remember_max_mb: the largest file a home takes. The real piles' largest are A1's day of sound (2,765 MB) and V1's videos (up to 1,133
# MB); at the 500 the pilot's small files allowed, mms refused every one of them and held nothing of A1 or V1.
SNAPSHOT_CONFIG = "[memory]\nfree_gb = 0\nremember_max_mb = 4000\n\n[decider]\nidle_s = 3600\n"
COPY_SKIPS = ("daemon.json", "daemon.log*", "daemon.test.log", "daemon.stop", "looks", "fake-decider.jsonl")
MEDIA_FOLDERS = ("files", "streams")  # what a home keeps of the recordings: a copy of a home links them instead of copying
PROJECT_PREFIX = "id:bench-"  # the project of every exec's asks and labels: `id:bench-<hash>`, so that they are in no default scope


def exec_project(qid: str) -> str:
    """The project an exec's asks and labels carry: `id:bench-` and 12 hex digits of the hash of its question's id. Its working
    folder's `.mms-project` holds it, less `id:`. Not the id itself: every result of mms's tools shows the agent its project,
    and an id says what its question is after (2026-11-07: A1's `leave-0829` and `return-1730` held their answers, and every
    `absent-` one said there is none, in every mm exec until then)."""
    return f"{PROJECT_PREFIX}{hashlib.sha256(qid.encode()).hexdigest()[:12]}"


def copy_home(snapshot: Path, target: Path, model: Model) -> Path:
    """A home for one (agent, model): the snapshot's files, with `model` as its decider. A `target` that exists (a run that
    is continued) is used as it is. The snapshot must have no daemon, so its index is whole. The media (MEDIA_FOLDERS) are
    hardlinked, not copied, so a home costs its index alone: the snapshot's are made read-only first, and a link shares that,
    so no home can write one. The daemon never rewrites a media file, and the janitor deletes nothing here (the snapshot's
    config sets no cap or age). The snapshot and the target must be on one volume."""
    if (snapshot / "daemon.json").exists():
        raise RuntimeError(f"the snapshot {snapshot} has a daemon running: stop it (mms stop) before copying")
    if target.exists():
        return target
    part = target.with_name(target.name + ".part")
    if part.exists():
        remove_tree(part)
    for folder in MEDIA_FOLDERS:
        for path in (snapshot / folder).rglob("*"):
            if path.is_file():
                os.chmod(path, stat.S_IREAD)

    def link_media(src: str, dst: str) -> None:
        (os.link if Path(src).relative_to(snapshot).parts[0] in MEDIA_FOLDERS else shutil.copy2)(src, dst)

    shutil.copytree(snapshot, part, ignore=shutil.ignore_patterns(*COPY_SKIPS), copy_function=link_media)
    config.set_decider(part, model=model.spec, url=model.url)
    os.replace(part, target)
    return target


def kept_items(home: Path) -> set[int]:
    """The items of an mms home that are kept (a label with `keep` marks them)."""
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30)) as db:
        return {row[0] for row in db.execute("SELECT id FROM items WHERE keep = 1")}


def scrub_home(home: Path, kept: set[int], *, projects: Sequence[str] = (), prefix: str | None = None) -> dict:
    """Take out of an mms home what the execs of `projects` (or of every project that starts with `prefix`) left in it, so that
    the next exec starts from the snapshot's state. That is every piece of state an exec can write there: the labels of its asks
    (`ask` labels each sighting with the caller's project, when the job ends) and of the `label` tool, the items that `label`
    kept (put back to `kept`, the snapshot's), and the project and session rows its calls made. What stays is no state of the
    exec's: the `looks/` pictures (named by what they show), the daemon's job records (random ids, in memory) and its log.
    Nothing else keeps an answer: ask has no cache, and no agent has `remember`. Done in the index, which the daemon reads at
    each call. It is idempotent, and run again at the start of each exec for the execs that ended, so that the label of a job
    that outlived its exec is taken out too."""
    where, arg = (("{} LIKE ?", prefix + "%") if prefix is not None else ("{} IN (SELECT value FROM json_each(?))", json.dumps(list(projects))))
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        labels = db.execute(f"DELETE FROM labels WHERE {where.format('project')}", (arg,)).rowcount
        db.execute(f"DELETE FROM sessions WHERE {where.format('project')}", (arg,))
        db.execute(f"DELETE FROM projects WHERE {where.format('id')}", (arg,))
        unkept = db.execute("UPDATE items SET keep = 0 WHERE keep = 1 AND id NOT IN (SELECT value FROM json_each(?))",
                            (json.dumps(sorted(kept)),)).rowcount
        db.execute("COMMIT")
    return {"labels": labels, "unkept": unkept}


def set_horizon(home: Path, h: float) -> dict:
    """Cut the home's index to the horizon `h` (a local epoch), in one transaction on the live index (the daemon reads it at each
    call, as `scrub_home` relies on): a clip that ends after `h` is `gone`, one that ends by it is `ok`, and the records after `h`
    move to a side table (`bench_hidden_records`) and come back, with their ids, when `h` moves later. Every source is cut: a
    horizon is the whole home's. Returns how many clips and records are shown and hidden."""
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS bench_hidden_records AS SELECT * FROM records WHERE 0")
        db.execute("UPDATE items SET state = CASE WHEN t1 <= ? THEN 'ok' ELSE 'gone' END WHERE state IN ('ok', 'gone')", (h,))
        db.execute("INSERT INTO bench_hidden_records SELECT * FROM records WHERE t > ?", (h,))
        db.execute("DELETE FROM records WHERE t > ?", (h,))  # records_ad takes them out of the search index
        db.execute("INSERT INTO records SELECT * FROM bench_hidden_records WHERE t <= ?", (h,))  # records_ai puts them back
        db.execute("DELETE FROM bench_hidden_records WHERE t <= ?", (h,))
        shown = db.execute("SELECT COUNT(*) FROM items WHERE state = 'ok'").fetchone()[0]
        hidden = db.execute("SELECT COUNT(*) FROM items WHERE state = 'gone'").fetchone()[0]
        records = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        hidden_records = db.execute("SELECT COUNT(*) FROM bench_hidden_records").fetchone()[0]
        db.execute("COMMIT")
    return {"clips": shown, "clips_hidden": hidden, "records": records, "records_hidden": hidden_records}
