"""Copies and measures, once, what the recorded runs hold that the new code must match (plan 7.1 to 7.5). The recorded runs are
read only: nothing under D:\\mm-bench or D:\\mm-bench-temp is written. Run it from the repo root with any Python 3.12:

    uv run python tests\\oracles\\capture_recorded.py [--pilot RUN] [--egolife RUN] [--out recorded]

It writes, in recorded/ (refused if one exists: a golden is not regenerated to make a test pass):
    pilot-s1/       run.json data-manifest.json shortcuts.json shortcuts.md report.json report.md: byte copies
                    sha256.json            the sha256 of every file of the run's data folder (1,575), hashed in this one process
                    execs.json             what each of the 110 execs wrote before it ran, for oracle 3 (plan 7.3)
                    instructions/<sha>.md  the distinct texts of the execs' instructions.md
                    replay/<agent>/<dir>/<qid>/events.jsonl, result.json: 8 small execs that ended without a problem, for oracle 5
    egolife-s1/     run.json report.json report.md: byte copies
    recorded.json   where it came from, and the sha256 of every file above

The converted results, questions and run.json of the new format are U5's and U6's (tests/oracles/convert_recorded.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RUNS = Path(r"D:\mm-bench")
PILOT = RUNS / "20261104-0314-pilot-s1"
EGOLIFE = RUNS / "20261103-0257-egolife-s1"
OUT = Path(__file__).with_name("recorded")
PILOT_FILES = ("run.json", "data-manifest.json", "shortcuts.json", "shortcuts.md", "report.json", "report.md")
EGOLIFE_FILES = ("run.json", "report.json", "report.md")
# execs/{claude,codex}/files/text-note-code-1: NOTES.txt says two dry runs overwrote their prompt, instructions, settings and catalog
OVERWRITTEN = {("claude", "files", "text-note-code-1"), ("codex", "files", "text-note-code-1")}
OVERWRITTEN_WHY = "two dry runs (11:13 and 12:41) overwrote prompt.txt, instructions.md and settings.json or catalog.json (NOTES.txt)"
# the small execs oracle 5 replays when D:\mm-bench is not there: two a cell, none with a problem
REPLAY = (("codex", "mm-d1-3B", "records-log-count-1"), ("codex", "mm-d1-3B", "video-porch-parcel-1"),
          ("codex", "files", "records-log-count-1"), ("codex", "files", "records-absent-log-1"),
          ("claude", "mm-d1-3B", "text-config-asof-1"), ("claude", "mm-d1-3B", "records-log-count-1"),
          ("claude", "files", "records-log-count-1"), ("claude", "files", "text-config-asof-1"))
INIT_KEYS = ("tools", "mcp_servers", "plugins", "apiKeySource", "permissionMode", "slash_commands", "skills")
MCP_HEAD, MCP_TAIL = ["-m", "multimem.cli", "--home"], ["mcp", "--tools", "ask,find,look,label"]  # mm's home is between them; as the pilot recorded it


def dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def data_files(data: Path) -> dict[str, list[int]]:
    """Every file of the data folder as bench.stats_of() lists it: relative path with forward slashes -> [size, mtime_ns]."""
    found = {}
    for path in sorted(data.rglob("*")):
        if path.is_file():
            info = path.stat()
            found[path.relative_to(data).as_posix()] = [info.st_size, info.st_mtime_ns]
    return found


def hashes(data: Path, manifest: dict) -> dict[str, str]:
    """The sha256 of each file of `data`, one file at a time, after checking that the folder is the manifest's."""
    if (now := data_files(data)) != manifest:
        sys.exit(f"{data} is not its manifest: {sorted(set(now) ^ set(manifest))[:3] or 'sizes or times differ'}")
    out = {}
    for rel in manifest:
        digest = hashlib.sha256()
        with (data / rel).open("rb") as file:
            while chunk := file.read(1 << 20):
                digest.update(chunk)
        out[rel] = digest.hexdigest()
    return out


def mcp_of(raw: bytes) -> dict:
    """mcp.json with the three paths named: the interpreter <PYTHON>, mm's home <HOME> and the project folder <WORK>."""
    config = json.loads(raw)
    [(name, server)] = config["mcpServers"].items()
    args = server["args"]
    if (name != "mm" or set(server) != {"command", "args", "env"} or args[:3] != MCP_HEAD or args[4:] != MCP_TAIL
            or list(server["env"]) != ["CLAUDE_PROJECT_DIR"]):
        sys.exit(f"mcp.json is not the shape the old code wrote: {config}")
    return {"mcpServers": {name: {"command": "<PYTHON>", "args": [*MCP_HEAD, "<HOME>", *MCP_TAIL], "env": {"CLAUDE_PROJECT_DIR": "<WORK>"}}}}


def exec_entry(run: Path, agent: str, dirname: str, qid: str, texts: dict[str, bytes]) -> dict:
    folder = run / "execs" / agent / dirname / qid
    entry = {"agent": agent, "dir": dirname, "qid": qid, "files": sorted(p.name for p in folder.iterdir()),
             "excluded": OVERWRITTEN_WHY if (agent, dirname, qid) in OVERWRITTEN else None}

    def read(name: str) -> bytes | None:
        return None if entry["excluded"] or not (folder / name).is_file() else (folder / name).read_bytes()

    prompt, settings, instructions, mcp, catalog = (read(n) for n in ("prompt.txt", "settings.json", "instructions.md", "mcp.json",
                                                                       "catalog.json"))
    if instructions is not None:
        texts[sha256(instructions)] = instructions
    init = json.loads((folder / "init.json").read_text(encoding="utf-8")) if (folder / "init.json").is_file() else None
    entry.update(prompt=None if prompt is None else prompt.decode("utf-8"),
                 settings_sha256=None if settings is None else sha256(settings),
                 instructions_sha256=None if instructions is None else sha256(instructions),
                 mcp=None if mcp is None else mcp_of(mcp),
                 init=None if init is None else {k: init.get(k) for k in INIT_KEYS},
                 catalog_sha256=None if catalog is None else sha256(catalog))
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--egolife", type=Path, default=EGOLIFE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if existing := [n for n in ("pilot-s1", "egolife-s1", "recorded.json") if (args.out / n).exists()]:
        sys.exit(f"{existing} exist in {args.out}: a golden is not regenerated to make a test pass; remove it on purpose to capture again")
    written: dict[str, str] = {}

    def write(rel: str, data: bytes) -> None:
        (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
        (args.out / rel).write_bytes(data)
        written[rel] = sha256(data)

    for name in PILOT_FILES:
        write(f"pilot-s1/{name}", (args.pilot / name).read_bytes())
    for name in EGOLIFE_FILES:
        write(f"egolife-s1/{name}", (args.egolife / name).read_bytes())

    meta = json.loads((args.pilot / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pilot / "data-manifest.json").read_text(encoding="utf-8"))
    write("pilot-s1/sha256.json", dump(hashes(Path(meta["data"]), manifest)).encode())

    texts: dict[str, bytes] = {}
    execs = [exec_entry(args.pilot, agent.name, cell.name, qid.name, texts)
             for agent in sorted((args.pilot / "execs").iterdir()) for cell in sorted(agent.iterdir()) for qid in sorted(cell.iterdir())]
    write("pilot-s1/execs.json", dump({"source": str(args.pilot), "execs": execs}).encode())
    for digest, data in texts.items():
        write(f"pilot-s1/instructions/{digest[:12]}.md", data)

    for agent, dirname, qid in REPLAY:
        result = (args.pilot / "results" / agent / dirname / f"{qid}.json").read_bytes()
        if (found := json.loads(result))["problem"] is not None or found["passed"] is not True:
            sys.exit(f"{agent}/{dirname}/{qid} is not a normal exec")
        write(f"pilot-s1/replay/{agent}/{dirname}/{qid}/result.json", result)
        write(f"pilot-s1/replay/{agent}/{dirname}/{qid}/events.jsonl", (args.pilot / "execs" / agent / dirname / qid / "events.jsonl").read_bytes())

    (args.out / "recorded.json").write_text(dump({"pilot": str(args.pilot), "egolife": str(args.egolife), "files": dict(sorted(written.items()))}),
                                            encoding="utf-8", newline="\n")
    if len(execs) != 110:
        sys.exit(f"{len(execs)} execs, not the pilot's 110")
    print(f"{len(written)} files in {args.out}: {len(execs)} execs, {len(manifest)} hashes, {len(texts)} instruction texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
