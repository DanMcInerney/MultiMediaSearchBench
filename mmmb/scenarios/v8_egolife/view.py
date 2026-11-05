"""What an agent with its own tools reads in V8, and the recordings past the horizon it must not have seen (moved from egolife.py of this
repository at bb498a6). The configurations with their own tools get, for each question, a folder of the recordings up to the question's
horizon H: the clips that ended by H as hardlinks to the dump (the agent cannot write a copy) and the transcript lines that ended by H.
Nothing later is there to read, not even the name of a later day's file. A command's output that shows a recording past H read it somewhere
else, and ends the exec as `escaped` like the other tripwires.

The folder of slot `k` is `<temp>/mmb-<key>-d<k>`, `key` being the run's tag, and the deny words of the other slots name it (see
`__init__.deny_words`)."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from mmmb.core import fs
from mmmb.core.contract import Dump, ManifestEntry
from mmmb.core.runfolder import JOBS_MAX

SEEN_TIME = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}")  # a transcript line's start or end, as the files arm's lines write it
SEEN_CLIP = re.compile(r"glasses[/\\]+\d{4}-\d\d-\d\d[/\\]+\d\d\.\d\d\.\d\d\.mp4")  # a clip's path in the files arm (`\\` escaped too)
MISSING = re.compile(r"no such file|does not exist|cannot find|not found|filenotfound", re.I)  # a line that reports a missing file


def view_dir(temp: Path, key: str, k: int) -> Path:
    """Slot `k`'s folder of recordings, for the run whose tag is `key`."""
    return temp / f"mmb-{key}-d{k}"


def clips_of(dump: Dump) -> list[ManifestEntry]:
    """The dump's clips, oldest first: each with its start (`t`) and its length."""
    return [e for e in dump.entries if e.kind == "clip"]


def transcripts_of(dump: Dump) -> list[ManifestEntry]:
    """The dump's transcript files, one for each day."""
    return [e for e in dump.entries if e.kind == "log"]


def files_view(view: Path, dump: Dump, h: float) -> Path:
    """The folder `view` made new with the recordings of `dump` up to the horizon `h`: `glasses/<date>/HH.MM.SS.mp4` for each clip
    that ended by `h` (hardlinks to the dump, so the agent cannot write a copy) and `transcript/<date>.jsonl` with the lines that
    ended by `h` (read-only), for a date that has any. Nothing later is there to read, not even the name of a later day's file."""
    fs.empty(view)
    for c in clips_of(dump):
        if c.t + c.length <= h:
            (view / c.path).parent.mkdir(parents=True, exist_ok=True)
            os.link(dump.root / c.path, view / c.path)
    for file in transcripts_of(dump):
        kept = [line for line in (dump.root / file.path).read_text(encoding="utf-8").splitlines()
                if datetime.fromisoformat(json.loads(line)["end"]).timestamp() <= h]
        if kept:
            target = view / file.path
            target.parent.mkdir(exist_ok=True)
            target.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            os.chmod(target, stat.S_IREAD)
    return view


def release_views(temp: Path, key: str) -> None:
    """The slots' folders of recordings removed (they hold links only). One that cannot be (a child of a killed exec still holds
    a file) is told and left, so it does not hide the error that ended the run."""
    for k in range(JOBS_MAX):
        if view_dir(temp, key, k).exists():
            try:
                fs.remove_tree(view_dir(temp, key, k))
            except OSError as error:
                print(f"could not remove {view_dir(temp, key, k)}: {error}", flush=True)


def seen_past_horizon(texts: Sequence[str], dump: Dump, h: float) -> list[str]:
    """The recordings past the horizon `h` that `texts` show: a start or an end of a transcript line that is after it (a time of
    the transcript, not any time: a file's mtime or a computed one is the agent's own), or the name of a clip that ended after
    it. The files arm's view holds neither, so an output that shows one read it somewhere else, without printing a path that
    `deny` would have stopped. A line that reports a missing file shows a clip name the agent made up (a guess at the clip of
    "now"), and its names are not counted."""
    ends = {c.path: c.t + c.length for c in clips_of(dump)}
    late = set()
    for file in transcripts_of(dump):
        for line in (dump.root / file.path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            late |= {t for t in (row["start"], row["end"]) if datetime.fromisoformat(t).timestamp() > h}
    seen = set()
    for text in texts:
        for line in text.splitlines():
            seen.update(t for t in SEEN_TIME.findall(line) if t in late)
            if not MISSING.search(line):
                names = (re.sub(r"[/\\]+", "/", found) for found in SEEN_CLIP.findall(line))
                seen.update(name for name in names if ends.get(name, 0.0) > h)
    return sorted(seen)
