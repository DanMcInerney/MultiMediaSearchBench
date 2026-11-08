"""The listing's MCP server over stdio (DESIGN 3.2): a dump's files as their metadata and nothing of their content. It reads one file,
`listing.jsonl`, that `Listing.ingest` wrote, and is given no path to the dump, so it cannot open a recording; it imports the SDK and the
standard library only, and starts as a script (`python server.py --listing FILE [--until EPOCH]`), so the file that runs is the file you
read. Written on the pattern of Multimem's `adapter.py`.

A row of the listing is `{path, source, size, mtime}` and, when the file says so, `exif` (a picture's tags, by their names) and `container`
(`format, duration, width, height, fps, codec, creation_time, sample_rate`). `mtime` is epoch seconds in the file and an ISO 8601 time in the
local zone in what the tools say. It is the one time the listing has: `since`, `until` and the horizon (`--until`) are all about it, and
the files are listed oldest first by it. A row past `--until` is not in the listing at all: not in `list_files`, and `file_info` says it is
not there, as it says of a path that never was.

Tools: `list_files(source, since, until, limit, after)` and `file_info(path)`. An answer is one JSON object in a text block; a refusal is
`{"error": {"code", "message"}}` with the result marked as an error."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from mcp import types
from mcp.server import Server

NAME = "listing"
TOOLS = ("list_files", "file_info")
FILE_KEYS = ("path", "source", "size", "mtime")  # every row has these
OPTIONAL_KEYS = ("exif", "container")
CONTAINER_KEYS = ("format", "duration", "width", "height", "fps", "codec", "creation_time", "sample_rate")
LIMIT_DEFAULT, LIMIT_MAX = 50, 200

INSTRUCTIONS = """\
The listing is the owner's files seen from outside: each file's path, source, size and modification time (local, ISO 8601), and for \
pictures, video and sound the EXIF tags and container fields the file carries itself (duration, width, height, frame rate, codec, \
creation time, sample rate). It never shows what a file contains: no picture, no sound, no text. Use list_files to find files by \
source and time, oldest first, a page at a time (call it again with after set to its next), and file_info for one file."""

DESCRIPTIONS = {
    "list_files": "Files of the listing, oldest modification time first: path, source, size and mtime. Narrow by source and by "
                  "mtime between since and until (both included). When the result's next is set, call again with after set to it "
                  "for the next page; matching is how many files match.",
    "file_info": "One file of the listing by its path, as list_files gives it: its size and mtime, and its EXIF tags and "
                 "container fields when it has any. Not its content.",
}

_TIME = {"type": ["string", "number"], "description": "ISO 8601 (no zone is local) or epoch seconds"}
SCHEMAS = {
    "list_files": {"type": "object", "additionalProperties": False, "properties": {
        "source": {"type": "string", "description": "only this source"},
        "since": _TIME, "until": _TIME,
        "limit": {"type": "integer", "minimum": 1, "maximum": LIMIT_MAX, "default": LIMIT_DEFAULT},
        "after": {"type": "string", "description": "the next of an earlier result: the page after it"}}},
    "file_info": {"type": "object", "additionalProperties": False, "required": ["path"],
                  "properties": {"path": {"type": "string", "description": "a path as list_files gives it, with forward slashes"}}},
}


class ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_row(row: object) -> None:
    """Raises a ValueError unless `row` is a row of the listing: the keys above and no other, the right types. The one place that says what
    a row may hold. `read_rows` checks every row it reads with it, when the server starts and when `Listing.ingest` reads back what it wrote."""
    if not isinstance(row, dict):
        raise ValueError("a row is a JSON object")
    if missing := [k for k in FILE_KEYS if k not in row]:
        raise ValueError(f"missing {', '.join(missing)}")
    if extra := sorted(set(row) - {*FILE_KEYS, *OPTIONAL_KEYS}):
        raise ValueError(f"unknown key {', '.join(extra)}")
    if not (isinstance(row["path"], str) and isinstance(row["source"], str) and isinstance(row["size"], int)
            and not isinstance(row["size"], bool) and is_number(row["mtime"])):
        raise ValueError("path and source are strings, size an integer and mtime a number")
    if "container" in row:
        container = row["container"]
        if not isinstance(container, dict) or (extra := sorted(set(container) - set(CONTAINER_KEYS))):
            raise ValueError(f"container is an object of {', '.join(CONTAINER_KEYS)}" + (f", not {extra[0]}" if isinstance(container, dict) else ""))
    if "exif" in row and not (isinstance(row["exif"], dict) and all(isinstance(k, str) for k in row["exif"])):
        raise ValueError("exif is an object by tag name")


def read_rows(file: Path) -> list[dict]:
    """The rows of `file`, checked, in file order. A path may be in it once."""
    rows: dict[str, dict] = {}
    lines = Path(file).read_text(encoding="utf-8").split("\n")  # not splitlines: U+2028 and the like are text in a JSON string
    for n, line in enumerate(lines[:-1] if lines[-1] == "" else lines, 1):
        try:
            row = json.loads(line)
            check_row(row)
            if row["path"] in rows:
                raise ValueError(f"{row['path']} is listed twice")
        except ValueError as error:
            raise ValueError(f"{file}:{n}: {error}") from None
        rows[row["path"]] = row
    return list(rows.values())


def iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def order(row: dict) -> tuple[float, str]:
    return row["mtime"], row["path"]


class View:
    """The listing as the agent has it: the rows no later than `until` (all when it is None), oldest first."""

    def __init__(self, rows: list[dict], until: float | None) -> None:
        self.rows = sorted((r for r in rows if until is None or r["mtime"] <= until), key=order)
        self.by_path = {r["path"]: r for r in self.rows}
        self.sources = sorted({r["source"] for r in self.rows})


def public(row: dict, whole: bool = False) -> dict:
    out = {"path": row["path"], "source": row["source"], "size": row["size"], "mtime": iso(row["mtime"])}
    return {**out, **{k: row[k] for k in OPTIONAL_KEYS if k in row}} if whole else out


def time_arg(arguments: dict, field: str) -> float | None:
    value = arguments.get(field)
    if value is None:
        return None
    try:
        t = float(value) if is_number(value) else datetime.fromisoformat(value).timestamp() if isinstance(value, str) else math.nan
    except (ValueError, OverflowError, OSError):  # (a year the platform has no epoch second for)
        t = math.nan
    if not math.isfinite(t):
        raise ToolError("bad_request", f"{field} is an ISO 8601 time or epoch seconds, not {value!r}")
    return t


def unknown_arguments(arguments: dict, tool: str) -> None:
    if extra := sorted(set(arguments) - set(SCHEMAS[tool]["properties"])):
        raise ToolError("bad_request", f"{tool} has no argument {extra[0]!r}; its arguments are {', '.join(SCHEMAS[tool]['properties'])}")


def list_files(view: View, arguments: dict) -> dict:
    unknown_arguments(arguments, "list_files")
    source, after = arguments.get("source"), arguments.get("after")
    limit = arguments.get("limit", LIMIT_DEFAULT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= LIMIT_MAX:
        raise ToolError("bad_request", f"limit is a whole number from 1 to {LIMIT_MAX}, not {limit!r}")
    if source is not None and source not in view.sources:
        raise ToolError("bad_request", f"there is no source {source!r}; the sources are {', '.join(view.sources)}")
    if after is not None and after not in view.by_path:
        raise ToolError("not_found", f"after is not a path of the listing: {after!r}")
    since, until = time_arg(arguments, "since"), time_arg(arguments, "until")
    matching = [r for r in view.rows if (source is None or r["source"] == source) and (since is None or r["mtime"] >= since)
                and (until is None or r["mtime"] <= until)]
    rest = matching if after is None else [r for r in matching if order(r) > order(view.by_path[after])]
    page = rest[:limit]
    return {"files": [public(r) for r in page], "matching": len(matching), **({"next": page[-1]["path"]} if len(rest) > limit else {})}


def file_info(view: View, arguments: dict) -> dict:
    unknown_arguments(arguments, "file_info")
    path = arguments.get("path")
    if not isinstance(path, str):
        raise ToolError("bad_request", "file_info needs path, a string")
    if path not in view.by_path:
        raise ToolError("not_found", f"{path!r} is not in the listing: give a path as list_files gives it, with forward slashes")
    return public(view.by_path[path], whole=True)


HANDLERS = {"list_files": list_files, "file_info": file_info}


def text_block(value: Any) -> types.TextContent:
    return types.TextContent(text=json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def build_server(view: View) -> Server:
    """The SDK server over `view`, not yet running."""
    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name=name, description=DESCRIPTIONS[name], input_schema=SCHEMAS[name])
                                            for name in TOOLS])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            if params.name not in HANDLERS:
                raise ToolError("not_found", f"there is no tool {params.name!r}")
            return types.CallToolResult(content=[text_block(HANDLERS[params.name](view, params.arguments or {}))], is_error=False)
        except ToolError as error:
            return types.CallToolResult(content=[text_block({"error": {"code": error.code, "message": error.message}})], is_error=True)

    return Server(NAME, instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call_tool)


async def serve_stdio(server: Server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def epoch(text: str) -> float:
    t = float(text)
    if not math.isfinite(t):
        raise argparse.ArgumentTypeError(f"{text!r} is not epoch seconds")
    return t


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The MCP server of the listing of a dump (stdio).")
    parser.add_argument("--listing", type=Path, required=True, help="the listing.jsonl an ingest wrote")
    parser.add_argument("--until", type=epoch, help="the horizon, epoch seconds: a row with a later mtime is not in the listing")
    args = parser.parse_args(argv)
    anyio.run(serve_stdio, build_server(View(read_rows(args.listing), args.until)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
