"""The listing system (DESIGN 3.2, plan 6.2): the floor and the leak filter. The agent gets a listing of the dump's files, their paths, sizes,
times, EXIF tags and container fields, through two tools, and no way to open a file (`--tools ""`; the server is given no path to the dump).
A question it answers right is answerable from metadata, or from what the model knows, and the report flags it (`Description.leak_filter`).

    store/listing.jsonl     one row per file of the dump (`ingest`), oldest first; `server.py` documents the row
    server.py               the MCP server `serve` hands the agent, started as a script over that file

`ingest` reads each file as any tool on it could (`read.py`) and never copies the manifest's `t`, `kind` or `sha256`. `serve` gives the
server the case's horizon as `--until`, so a row whose mtime is past it is not there; a case with no horizon (V7's) is not cut. The server
holds nothing that belongs to a case, so execs run at once."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import av
import PIL

from mmmb.core import fs
from mmmb.core.contract import Case, Description, Dump, IngestReport, McpServer, PlainSession, Served, SessionContext
from mmmb.systems import read as read_system_toml
from mmmb.systems.listing import read as reading
from mmmb.systems.listing import server

HERE = Path(__file__).parent
NAME = "listing"
LISTING = "listing.jsonl"  # in the store
BASE = "You answer the user's question from a listing of their files: paths, sizes, times and the metadata the files carry. You cannot open the files."
HOLDER = NAME  # the word the prompt's sentences use for it, as "mm"
MODALITIES = frozenset({"image", "video", "audio", "text", "records"})  # every question is asked of it: it is a filter, not a reader
INGEST_VERSION = 1  # bumped when ingest writes something other than before


def sha_of(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


class Listing:
    def describe(self, variant: str | None) -> Description:
        if variant is not None:
            raise ValueError(f"listing has no variants, not {variant!r}")
        store = sha_of(NAME, INGEST_VERSION, server.FILE_KEYS, server.OPTIONAL_KEYS, server.CONTAINER_KEYS, reading.TEXT_MAX, reading.LIST_MAX,
                       PIL.__version__, av.__version__)
        config = sha_of(store, {tool: [server.DESCRIPTIONS[tool], server.SCHEMAS[tool]] for tool in server.TOOLS}, server.INSTRUCTIONS, BASE, HOLDER)
        return Description(name=NAME, version="1", variant=None, reads=MODALITIES, local=True, needs_gpu=False, gpu_memory_gb=None, network=False,
                           license=read_system_toml(HERE / "system.toml").license, config_hash=config, store_hash=store, horizon="cut",
                           has_oracle=False, leak_filter=True)

    def check_ready(self, variant: str | None) -> None:
        """Nothing to check: Pillow and PyAV are the project's own dependencies."""

    def ingest(self, dump: Dump, store: Path) -> IngestReport:
        """`store/listing.jsonl`, a row per entry of the dump read from its file, oldest mtime first, and read back as the server reads it.
        A picture, video or sound whose header will not read still has its row, with a refusal for what was not read."""
        began = time.monotonic()
        file = Path(store) / LISTING
        if file.exists():
            raise ValueError(f"{file} exists: a listing store is made once")
        rows, refused = [], []
        for entry in dump.entries:
            row, refusals = reading.row_of(dump.root, entry)
            rows.append(row)
            refused += refusals
        rows.sort(key=server.order)
        fs.write_text(file, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        server.read_rows(file)
        return IngestReport(seconds=round(time.monotonic() - began, 1), gpu_seconds=0.0, bytes_written=file.stat().st_size,
                            refused=tuple(refused), misdated=())

    def open(self, store: Path, dump: Dump, context: SessionContext) -> ListingSession:
        file = Path(store) / LISTING
        if not file.is_file():
            raise FileNotFoundError(f"{file} is not there: the dump was not ingested for listing")
        return ListingSession(file)

    def preview(self, store: Path, dump: Dump, context: SessionContext) -> ListingSession:
        """The session of a dry run is the session: it starts nothing."""
        return self.open(store, dump, context)


class ListingSession(PlainSession):
    def __init__(self, listing: Path) -> None:
        self.listing = listing

    def serve(self, case: Case) -> Served:
        """The server over the listing, cut to the case's horizon when it has one. Nothing else is the case's."""
        until = () if case.horizon is None else ("--until", repr(case.horizon))
        args = (str(Path(server.__file__)), "--listing", str(self.listing), *until)
        return Served(McpServer(server.NAME, sys.executable, args, {}, server.TOOLS), BASE, (), server.INSTRUCTIONS, HOLDER, None, {})
