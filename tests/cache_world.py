"""What the tests of the cache (tests/test_cache.py) build over: a scenario of three small files that counts its builds, can be told to block
half way (for a second process to meet it) or to fail, and a system that counts its ingests. Nothing here starts an agent or imports
multimem; the scenario is a module-like object with the names core/cache.py reads (`name`, `generator`, `build`)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from stub_system import tiny_questions

from mmmb.core.contract import Description, Dump, IngestReport, LogSource, ManifestEntry, Refusal
from mmmb.core.scenario import dump_key

RECIPE = {"scale": "tiny", "seed": 1, "days": ["2026-10-30", "2026-10-31"], "utc_offsets": ["-0400", "-0400", "-0400"]}
FILES = {"notes/a.md": "a note", "screen/s.mkv": "a screen", "logs/web.jsonl": "{}"}
LOGS = LogSource("logs", "logs", "*.jsonl", "ts")
WAITED_S = 60.0  # the longest a blocked build waits for the test to let it go


def stamps_of(root: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime in ns) of every file under `root` but the lock files (which taking a lock touches) and the folders of the ffmpeg a run
    copies for its agents: what a command that only finds what was made leaves as it was."""
    skipped = ("locks",)
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))
            if p.is_file() and not set(skipped) & set(p.relative_to(root).parts) and not any(part.endswith("-bin") for part in p.relative_to(root).parts)}


def entry(root: Path, rel: str) -> ManifestEntry:
    """The entry of a file as written: its size, time and hash measured from the bytes on disk."""
    path = root / rel
    info = path.stat()
    return ManifestEntry(rel, rel.split("/")[0], 1.0, "note", "text", None, False, info.st_size, info.st_mtime_ns,
                         hashlib.sha256(path.read_bytes()).hexdigest())


class Toy:
    """A scenario that writes `FILES` (their text followed by `content`) into the folder it is given and says what it did: `builds` holds
    the folder of each call. `started` is touched after the first file is written and `wait_for` waited for before the others (a build
    that a second process can meet half way); `fails` is raised after the first file."""

    name = "toy-cache"
    generator = 1

    def __init__(self, content: str = "v1", *, started: Path | None = None, wait_for: Path | None = None, fails: Exception | None = None) -> None:
        self.content, self.started, self.wait_for, self.fails = content, started, wait_for, fails
        self.builds: list[Path] = []

    def build(self, recipe, out: Path, workers: int = 3):
        out = Path(out)
        self.builds.append(out)
        for n, (rel, text) in enumerate(FILES.items()):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(text + self.content, encoding="utf-8")
            if n == 0:
                if self.started is not None:
                    self.started.write_text("started", encoding="utf-8")
                if self.wait_for is not None:
                    deadline = time.monotonic() + WAITED_S
                    while not self.wait_for.exists() and time.monotonic() < deadline:
                        time.sleep(0.05)
                if self.fails is not None:
                    raise self.fails
        return Dump(dump_key(self.name, recipe, self.generator), out, self.name, tuple(entry(out, rel) for rel in FILES), (LOGS,)), tiny_questions()


class Ingester:
    """A system called `sys` that counts its ingests, writes a file into the store it is given, and reports `report`; its variants differ in
    `config_hash` only, and `version` and `store_hash` are what a test changes to make another version of it."""

    def __init__(self, *, version: str = "1", store_hash: str = "h", report: IngestReport | None = None, fails: Exception | None = None) -> None:
        self.version, self.store_hash, self.fails = version, store_hash, fails
        self.report = report or IngestReport(seconds=12.5, gpu_seconds=0.0, bytes_written=9, refused=(Refusal("screen/s.mkv", "too big"),),
                                             misdated=(), measured={"decode_s": 0.024})
        self.ingests: list[tuple[str, Path]] = []

    def describe(self, variant: str | None) -> Description:
        return Description(name="sys", version=self.version, variant=variant, reads=frozenset({"text"}), local=True, needs_gpu=False,
                           gpu_memory_gb=None, network=False, license="none", config_hash=f"c-{variant}", store_hash=self.store_hash,
                           horizon="cut", has_oracle=False)

    def ingest(self, dump: Dump, store: Path) -> IngestReport:
        self.ingests.append((dump.key, Path(store)))
        (Path(store) / "index.bin").write_bytes(b"index of " + dump.key.encode())
        if self.fails is not None:
            raise self.fails
        return self.report
