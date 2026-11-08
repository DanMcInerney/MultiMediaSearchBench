"""What the tests of the listing system share (tests/test_listing.py, test_listing_run.py): a small dump of the files a leak would come out of,
with its secrets planted; a real MCP client over stdio to the server; the case a session serves; and the stand-in agent that reads the listing
through the server it is given. Nothing here starts a real agent or imports multimem.

The planted dump holds, beside a note, a config and a log whose bodies say `SECRET`, a PNG whose pixels are the bytes of `SECRET` (stored, not
compressed, so the word is in the file's bytes), a WAV whose samples are the same bytes, and a JPEG that says a comment in its EXIF. The
manifest's entries carry values no file holds (`KIND`, `SHA`, and a `t` that is the file's mtime plus `T_OFFSET`: within the listing's tolerance
of a file's time, which ingest checks, yet a value no file holds), so that a copy of one into the listing shows."""

from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import sys
import wave
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

from agent_spec import clocked
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from PIL import ExifTags, Image

from mmsb.core.agents.spec import SessionSpec
from mmsb.core.contract import Case, Dump, ManifestEntry
from mmsb.systems.listing.system import Listing, ListingSession

SECRET = "ZEBRAQUARTZ"
T, KIND, SHA = 424242.5, "KINDWORD", "ab" * 32  # what the manifest says and no file does (T: a time that no row says, for a test that plants it)
T_OFFSET = 0.4375  # the manifest's `t` of a file is its mtime plus this
COMMENT = "planted exif comment"
BLOB = re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")  # a run of base64: a file's bytes carried in text
ROW_KEYS = {"path", "source", "size", "mtime", "exif", "container"}  # the spec's rows
CONTAINER_KEYS = {"format", "duration", "width", "height", "fps", "codec", "creation_time", "sample_rate"}  # and its container fields
ANSWER_KEYS = {"files", "matching", "next", "error", "code", "message"}
EXIF_NAMES = set(ExifTags.TAGS.values()) | set(ExifTags.GPSTAGS.values())
ALLOWED_KEYS = ROW_KEYS | CONTAINER_KEYS | ANSWER_KEYS | EXIF_NAMES


def png_of(pixels: bytes) -> bytes:
    """A grey PNG one pixel high whose pixels are `pixels`, written by hand with the filter none and stored (not compressed) data, so that they are
    in the file's bytes as they are (Pillow's writer filters them)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", len(pixels), 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + pixels, 0)) + chunk(b"IEND", b""))


def planted_dump(root: Path) -> Dump:
    """The files above under `root`, with their entries, and mtimes of 2023; an entry's `t` is its file's mtime plus `T_OFFSET`."""
    files = {
        "notes/n.md": f"the code word is {SECRET}\n".encode(),
        "config/app.toml": f'password = "{SECRET}"\n'.encode(),
        "logs/web.jsonl": (json.dumps({"ts": "2026-01-02T03:04:05-05:00", "msg": SECRET}) + "\n").encode(),
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "phone").mkdir()
    (root / "phone" / "IMG_planted.png").write_bytes(png_of(SECRET.encode()))
    exif = Image.Exif()
    exif[ExifTags.Base.ImageDescription] = COMMENT
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.UserComment] = b"ASCII\x00\x00\x00user comment text"
    Image.new("RGB", (32, 24), (200, 30, 30)).save(root / "phone" / "IMG_birthday.jpg", exif=exif)
    (root / "mic").mkdir()
    with wave.open(str(root / "mic" / "mic-1.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit samples are bytes: the word is in the file
        w.setframerate(8000)
        w.writeframes(SECRET.encode() * 100)
    for rel in ("phone/IMG_planted.png", "mic/mic-1.wav"):
        assert SECRET.encode() in (root / rel).read_bytes(), f"{rel} must hold the secret as bytes, or a leak of its content would not show"
    media = {"png": "image", "jpg": "image", "wav": "audio", "md": "text", "toml": "text", "jsonl": "records"}
    entries = []
    for n, rel in enumerate(sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())):
        mtime = 1_700_000_000 + n * 60
        os.utime(root / rel, (mtime, mtime))
        entries.append(ManifestEntry(rel, rel.split("/")[0], mtime + T_OFFSET, KIND, media[rel.rsplit(".", 1)[1]], None, False, 1, 1, SHA))
    return Dump("k" * 64, root, "planted", tuple(entries), ())


def ingested(dump: Dump, store: Path) -> Path:
    """`dump` ingested by the real system into `store` (made), whose report must have nothing refused: the listing's file."""
    store.mkdir(parents=True)
    report = Listing().ingest(dump, store)
    assert report.refused == () and report.misdated == () and report.gpu_seconds == 0.0, report
    return store / "listing.jsonl"


def case(horizon: float | None = None, qid: str = "q") -> Case:
    return Case(qid, 0.0, horizon, (), 0.0, 0.0, qid, 0.0, 0, Path(), Path())


def server_of(listing: Path, horizon: float | None = None):
    """The server command a session over `listing` serves a case with `horizon`: what the agent starts."""
    return ListingSession(listing).serve(case(horizon)).server


def params_of(server, cwd: Path | None = None) -> StdioServerParameters:
    return StdioServerParameters(command=server.command, args=list(server.args), env=dict(server.env) or None, cwd=cwd)


def talk(script: Callable, listing: Path, horizon: float | None = None, program: Sequence[str] | None = None):
    """The value of `await script(client)` over a real MCP client (the SDK's) on stdio to the server `serve` gives for `horizon`; or, with
    `program`, to that command line instead (its arguments after the python of this process)."""
    server = server_of(listing, horizon)
    params = params_of(server) if program is None else StdioServerParameters(command=server.command, args=[*program, *server.args])

    async def go():
        async with Client(params) as client:
            return await script(client)

    return asyncio.run(go())


async def called(client, tool: str, **arguments) -> tuple[bool, dict]:
    """(whether the call is an error, the JSON object of its answer) of one call of a tool."""
    result = await client.call_tool(tool, arguments)
    (block,) = result.content
    return result.is_error, json.loads(block.text)


async def pages(client, **arguments) -> list[dict]:
    """Every page of `list_files` over `arguments`, following `next`."""
    out, after = [], None
    while True:
        error, got = await called(client, "list_files", **arguments, **({"after": after} if after else {}))
        assert not error, got
        out.append(got)
        if "next" not in got:
            return out
        after = got["next"]


def keys_of(value) -> set[str]:
    """Every key of every object in a JSON value."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_of(v)}
    return {k for v in value for k in keys_of(v)} if isinstance(value, list) else set()


class Reader:
    """A stand-in agent for a session with the listing's server: it starts the server its spec gives, as the agent does, asks `list_files` and
    then `file_info` of the first file, and reports those calls as the agents do. It answers right for the questions in `knows` (a qid to its
    answer) and `unknown` for the rest. `execs` holds the specs it was given."""

    def __init__(self, knows: dict[str, str] | None = None) -> None:
        self.knows, self.execs = knows or {}, []

    def __call__(self, spec: SessionSpec) -> dict:
        self.execs.append(spec)

        async def go() -> list[dict]:
            async with Client(params_of(spec.server, spec.work)) as client:
                listed = await client.call_tool("list_files", {"limit": 3})
                first = json.loads(listed.content[0].text)["files"][0]["path"]
                info = await client.call_tool("file_info", {"path": first})
                return [{"tool": tool, "arguments": arguments, "result": result.content[0].text}
                        for tool, arguments, result in (("list_files", {"limit": 3}, listed), ("file_info", {"path": first}, info))]

        return {"final": f"ANSWER: {self.knows.get(spec.run_dir.name, 'unknown')}", "tools": asyncio.run(go()), **clocked(spec, 1.5), "problem": None,
                "turns": 2, "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}
