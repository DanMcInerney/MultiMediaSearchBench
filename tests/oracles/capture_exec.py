"""Records what the old benchmark code builds for an exec, once, for the new code to match (plan 7.3, oracle 3).

The argv of an exec is not on disk in any recorded run, so it is captured from the code that made them: bench.py, claude_run.py,
codex_run.py and egolife.py as commit bb498a6 holds them (U8 deleted them from the tree: they are read with `git show bb498a6:<file>`
into a temporary folder). No CLI is started and nothing is asked. `claude_run.prepare` and
`codex_run.prepare` are called with a stand-in command, and the files they write are copied. Run it with the search tool's own venv,
which has what the old modules import, from the repo root:

    set PYTHONDONTWRITEBYTECODE=1
    set MMS_CHECKOUT=<the search tool's checkout>
    %MMS_CHECKOUT%\\.venv\\Scripts\\python.exe tests\\oracles\\capture_exec.py

It was run once, when the tool was called Multimem (package `multimem`, command `mm`, commit fe0ece5), and recorded/exec
holds what it wrote. The tool is now MultimediaSearch (`mms`) and this script names it so, as the tests that import its helpers (`placer`,
`plant_git`, `planted_environment`) need; the tool's checkout may still be in a folder called `Multimem`. Run again it would capture the renamed tool, with
the old modules' old names, so it is not run again: tests/golden_names.py maps the recording's names to the tool's present ones.

The old modules were written to live in the tool's benchmarks/ folder and find the skill relative to their own file, so here they
are pointed at its skill: the file the pilot used, at the commit the tool was at then (its text drifted by 5 lines since the
pilot: plan section 1). The folders of the exec are real but temporary, and are replaced in the output by <RUN> <WORK> <HOME> <DATA>
<BIN>, with <PYTHON> for the interpreter that starts mm's server and <GIT> for the planted Git for Windows. The process environment
is planted, not the owner's. The run is Windows-only, as the code is.

Output, in recorded/exec/ (refused if it exists: a golden is not regenerated to make a test pass):
    capture.json                  where it came from: commits, the sha256 of every old module and of the text it read, the inputs
                                  and the planted environment, and the sha256 of every file below
    <agent>-<arm>/argv.json       the command line, for claude and codex and the arms blind, files and mm
    <agent>-<arm>/env.json        claude: env_of(); codex: the environment of the process (its shell's is in the argv)
    <agent>-<arm>/instructions.md, settings.json (claude), mcp.json (claude mm), catalog.json (codex): as prepare wrote them
    deny-v7.json, deny-v8.json    Run.deny() and EgoRun.deny() for the paths in the file, for no slot and for slot 0

codex-models.json is the saved `codex debug models` table of the capture: the entries gpt-6.1-terra and gpt-6-sol of the catalog of
the pilot's execs/codex/files/records-log-count-1 (sha256 ac5d2260...), with their model_messages and base_instructions cut to
"<cut>" (86 KB each, never read by the code).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any import of the repo or of MultimediaSearch: neither is written to

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
OLD_COMMIT = "bb498a6"  # the commit that holds the old modules
OLD_MODULES = ("bench.py", "claude_run.py", "codex_run.py", "egolife.py", "haystack.py", "grade.py")
OLD_FILES = (*OLD_MODULES, "haystack_media.py", "shortcuts.py", "egolife_oracles.json")  # (what they import of each other, and their data)
MODELS = Path(__file__).with_name("codex-models.json")
DEFAULT_OUT = Path(__file__).with_name("recorded") / "exec"
AGENTS = ("claude", "codex")
ARMS = ("blind", "files", "mm")
CLAUDE = {"name": "sonnet", "level": "high"}  # claude_run.model() and effort() with no MM_CLAUDE_* set, as the pilot ran
CODEX = {"name": "gpt-6.1-terra", "level": "medium"}  # MM_CODEX_EFFORT=medium, as RESUME.md says the pilot ran
MODEL = "d1-3B"
PLACEHOLDERS = ("RUN", "WORK", "HOME", "DATA", "BIN")
SENTINEL = {"folder": r"D:\__RUN__", "temp": r"D:\__TEMP__", "out": r"D:\__OUT__", "root": r"D:\__ROOT__", "tag": "TAG"}
DAYS = (date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3))  # the pilot's


def planted_environment(git: Path) -> dict[str, str]:
    """What an owner's shell holds that the exec must not pass on, and what it keeps."""
    return {"PATH": os.pathsep.join([str(git / "cmd"), r"C:\owner\bin", r"C:\owner\venv\Scripts"]),
            "SYSTEMROOT": r"C:\Windows", "USERPROFILE": r"C:\owner", "KEEP_ME": "kept",
            "ANTHROPIC_API_KEY": "sk-planted", "ANTHROPIC_AUTH_TOKEN": "planted-token",
            "MMS_HOME": r"C:\owner\mm-home", "MMS_MODELS": r"D:\models", "MMS_DEBUG": "1",
            "VIRTUAL_ENV": r"C:\owner\venv"}


def plant_git(git: Path) -> None:
    """Git for Windows as claude_run.git_root() finds it: git.exe in cmd, and usr/bin/bash.exe above it."""
    for sub in ("cmd/git.exe", "bin/bash.exe", "usr/bin/bash.exe"):
        (git / sub).parent.mkdir(parents=True, exist_ok=True)
        (git / sub).write_bytes(b"")


def placer(places: dict[str, Path | str]):
    """A function that puts each name for the real path in text, lists and dicts, in both slash styles, longest path first."""
    table = sorted(((form, f"<{name}>") for name, real in places.items()
                    for form in {str(real), str(real).replace("\\", "/")}), key=lambda p: -len(p[0]))

    def place(value):
        if isinstance(value, str):
            for form, name in table:
                value = value.replace(form, name)
            return value
        if isinstance(value, (list, tuple)):
            return [place(v) for v in value]
        if isinstance(value, dict):
            return {place(k): place(v) for k, v in value.items()}
        return value

    return place


def dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def mms_checkout() -> Path:
    """The checkout of the search tool (its folder may keep the old name), which `MMS_CHECKOUT` names."""
    if not (named := os.environ.get("MMS_CHECKOUT")):
        raise SystemExit("set MMS_CHECKOUT to the checkout of the search tool, which the old modules' tests folder and skill are read from")
    return Path(named)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head(repo: Path) -> str:
    return git_head_of("HEAD", repo)


def git_head_of(rev: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", rev], capture_output=True, text=True, check=True).stdout.strip()


def old_source(name: str) -> bytes:
    """The bytes of old file `name` as commit OLD_COMMIT holds it."""
    return subprocess.run(["git", "-C", str(REPO), "show", f"{OLD_COMMIT}:{name}"], capture_output=True, check=True).stdout


def old_code(folder: Path) -> None:
    """The old modules and the data they read written into `folder`, from the commit that holds them."""
    for name in OLD_FILES:
        (folder / name).write_bytes(old_source(name))


def import_old_modules(old: Path):
    """The old modules (in `old`), with MultimediaSearch's tests folder on the path (haystack and bench import its day, fake_decider and
    conftest) and the skill's place pointed at MultimediaSearch's."""
    sys.path[:0] = [str(old), str(mms_checkout() / "tests")]
    import bench
    import claude_run
    import codex_run
    import egolife

    skill = mms_checkout() / "src" / "mms" / "skills" / "mms-questions" / "SKILL.md"
    codex_run.SKILL, codex_run.NOTES_DIR = skill, skill.parent / "models"
    codex_run.NOTES = codex_run.NOTES_DIR / "d1.md"
    claude_run.SKILL = skill
    return bench, claude_run, codex_run, egolife


def cell(agent: str, arm: str, base: Path, modules, notes: str, models: dict):
    """prepare() for one agent and arm in folders under `base`, as bench.real_agents() calls the two agents. Returns the argv, the
    environment and the places to put in place of the folders."""
    _, claude_run, codex_run, _ = modules
    paths = {name: base / name for name in PLACEHOLDERS}
    run_dir, work, home, data, bin_dir = (paths[n] for n in PLACEHOLDERS)
    if agent == "claude":
        argv = claude_run.prepare(run_dir, arm, work, home=home if arm == "mm" else None, data=data if arm == "files" else None,
                                  notes=notes if arm == "mm" else None, command=["<claude>"], **CLAUDE)
        env = claude_run.env_of(bin_dir, arm, work)
    else:
        argv = codex_run.prepare(run_dir, CODEX["name"], CODEX["level"], "default", home if arm == "mm" else None, models,
                                 ["<codex>"], arm=arm, work=work, notes=notes if arm == "mm" else "", bin_dir=bin_dir)
        env = codex_run.path_with(bin_dir)
    return argv, env, paths


def deny_words(modules) -> dict:
    """Run.deny() and EgoRun.deny() over the sentinel folders. The paths are in the output as they are: the module constants are
    set to sentinels too, so the file says which input made each word."""
    bench, _, codex_run, egolife = modules
    folder, temp = Path(SENTINEL["folder"]), Path(SENTINEL["temp"])
    out = {"inputs": {**SENTINEL, "days": [d.isoformat() for d in DAYS], "seed": 1, "scale": "pilot",
                      "v7_data": str(temp / f"mmb-{SENTINEL['tag']}-data"), "v8_data": str(folder / "data")}}
    with mock.patch.object(bench, "OUT", Path(SENTINEL["out"])), mock.patch.object(codex_run, "ROOT", Path(SENTINEL["root"])):
        v7 = bench.Run(folder, 1, "pilot", DAYS, temp / f"mmb-{SENTINEL['tag']}-data", temp, SENTINEL["tag"])
        v8 = egolife.EgoRun(folder, 1, "egolife", DAYS, folder / "data", temp, SENTINEL["tag"], source=Path(r"D:\bench-data\egolife"))
        for name, run in (("v7", v7), ("v8", v8)):
            out[name] = {"none": list(run.deny()), "slot0": list(run.deny(run.slot(0))), "slot0_path": str(run.slot(0))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = parser.parse_args().out
    if os.name != "nt":
        sys.exit("the goldens are what the old code builds on Windows")
    if out.exists():
        sys.exit(f"{out} exists: a golden is not regenerated to make a test pass; remove it on purpose to capture again")
    old = Path(tempfile.mkdtemp(prefix="mmsb-old-"))
    old_code(old)
    modules = import_old_modules(old)
    _, claude_run, codex_run, _ = modules
    from mms import decider

    notes = next(a.notes for a in decider.known_abouts() if a.name == MODEL)
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="mmsb-capture-") as tmp:
        git = Path(tmp) / "git"
        plant_git(git)
        planted = planted_environment(git)
        out.mkdir(parents=True)

        def write(rel: str, data: bytes) -> None:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(data)
            files[rel] = sha256(data)

        for agent in AGENTS:
            for arm in ARMS:
                with mock.patch.dict(os.environ, planted, clear=True):
                    argv, env, paths = cell(agent, arm, Path(tmp) / f"{agent}-{arm}", modules, notes, models)
                place = placer({**paths, "PYTHON": sys.executable, "GIT": git})
                write(f"{agent}-{arm}/argv.json", dump(place(argv)).encode())
                write(f"{agent}-{arm}/env.json", dump(dict(sorted(place(env).items()))).encode())
                for name in ("instructions.md", "settings.json", "mcp.json", "catalog.json"):
                    if (made := paths["RUN"] / name).is_file():
                        raw = made.read_bytes()
                        write(f"{agent}-{arm}/{name}", dump(place(json.loads(raw))).encode() if name == "mcp.json" else raw)
        deny = deny_words(modules)
        for name in ("v7", "v8"):
            write(f"deny-{name}.json", dump({"inputs": deny["inputs"], name: deny[name]}).encode())
        skill = codex_run.SKILL
        from mms import adapter

        meta = {"repo_commit": git_head(REPO), "old_commit": git_head_of(OLD_COMMIT), "multimem_commit": git_head(mms_checkout()),
                "python": sys.version.split()[0], "model": MODEL, "notes": notes, "claude": CLAUDE, "codex": CODEX,
                "old_modules": {n: sha256(old_source(n)) for n in OLD_MODULES},
                "read": {"SKILL.md": sha256(skill.read_bytes()), notes: sha256((codex_run.NOTES_DIR / notes).read_bytes()),
                         "adapter.INSTRUCTIONS": sha256(adapter.INSTRUCTIONS.encode("utf-8")),
                         "codex-models.json": sha256(MODELS.read_bytes())},
                "planted_environment": placer({"GIT": git})(planted), "files": dict(sorted(files.items()))}
        (out / "capture.json").write_text(dump(meta), encoding="utf-8", newline="\n")
    shutil.rmtree(old, ignore_errors=True)
    print(f"{len(files)} files in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
