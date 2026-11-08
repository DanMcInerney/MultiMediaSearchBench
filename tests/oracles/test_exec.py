"""Oracle 3 (plan 7.3) for the blind, files and server cells of both agents: the exec the new agents build is the exec the old code
built.

The expected values are not the new code's. They are the goldens that `capture_exec.py` wrote from the old claude_run.py and
codex_run.py (argv, environment, instructions, settings, mcp.json, catalog), the 110 recorded pilot execs (`pilot-s1/execs.json`
and its instruction texts), the recorded catalogs of `D:\\mm-bench` (read, not copied: 54 files of 648 KB), and the recorded
event streams of eight execs (`pilot-s1/replay`). The server cell is built from a `Served` made of the recorded instruction text.
A server with the agent's own tools has no golden: its expected argv, settings and instructions are written here by hand from plan
section 4.4, and its environment is the files cell's by the rule that the own tools pick it.

Intended differences (plan 7.3, each asserted somewhere, none ignored): the skill's five drifted lines (test_goldens.py), the
`MM_*` variables renamed `MMSB_*` (test_agents.py), the prompt (not an agent's), the deny words (RunFolder's), and the redirects
and longest command Claude's files line names (here, `files_now`)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest
from agent_spec import BASE_LINE, server_of, spec_of
from capture_exec import PLACEHOLDERS, placer, plant_git, planted_environment

from mmsb.core.agents import claude, codex, common
from mmsb.core.answer import answer_of
from mmsb.core.contract import Served

RECORDED = Path(__file__).with_name("recorded")
EXEC, PILOT = RECORDED / "exec", RECORDED / "pilot-s1"
MODELS = json.loads(Path(__file__).with_name("codex-models.json").read_text(encoding="utf-8"))
RECORDED_RUN = Path(r"D:\mm-bench\20261104-0314-pilot-s1")
GOLDEN_CELLS = [(agent, mode) for agent in ("claude", "codex") for mode in ("blind", "files", "mm")]
windows_only = pytest.mark.skipif(os.name != "nt", reason="the goldens are what the old code builds on Windows (backslashes, Git)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Claude's files line names redirects and the longest command Claude Code runs (claude.LONGEST_COMMAND), which the old code's did not
FILES_WAS = "(no cd, pipes, && or other programs), and write"
FILES_NOW = "(no cd, pipes, redirects, && or other programs), keep it under 10,000 characters (a longer one is refused), and write"


def files_now(text: str) -> str:
    """A recorded instructions text of Claude's with the files line as the agent writes it now: the words on redirects and the
    command's length put in, once, and nothing else changed."""
    assert text.count(FILES_WAS) == 1, "Claude's files line, as recorded"
    return text.replace(FILES_WAS, FILES_NOW)


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def served_from(claude_text: str, codex_text: str, server) -> Served:
    """The `Served` whose pieces join into the recorded instructions of `agent`: the base line, the server's own words (what
    Codex reads between the line and the skill) and the rest (the skill and the notes). The pieces are taken apart from the two
    recorded files, which share the skill and the notes."""
    base, rest = claude_text.split("\n\n", 1)
    codex_base, codex_rest = codex_text.split("\n\n", 1)
    assert base == codex_base == BASE_LINE and codex_rest.endswith("\n\n" + rest)
    told = codex_rest.removesuffix("\n\n" + rest)
    assert told and told not in rest, "the server's own words are a part of their own"
    return Served(server=server, base_line=base, parts=(rest,), server_instructions=told, holder="mm", clock_note=None, record={})


def cell(base: Path, mode: str, claude_text: str, codex_text: str, deny: tuple[str, ...] = ()):
    """The spec of a cell, in the folders the goldens call RUN, WORK, HOME, DATA and BIN, and those folders. A server's `Served`
    joins into the recorded instructions of `claude_text` and `codex_text`."""
    paths = {name: base / name for name in PLACEHOLDERS}
    served = served_from(claude_text, codex_text, server_of(paths["HOME"], paths["WORK"])) if mode.startswith("mm") else None
    return spec_of(mode, base, run_dir=paths["RUN"], work=paths["WORK"], bin_dir=paths["BIN"], deny=deny,
                   data=paths["DATA"] if mode.endswith("files") else None, **({"served": served} if served else {})), paths


# -- the goldens: what the old code built --------------------------------------------------------------------------------

@windows_only
@pytest.mark.parametrize(("agent", "mode"), GOLDEN_CELLS, ids=[f"{a}-{m}" for a, m in GOLDEN_CELLS])
def test_the_exec_the_old_code_built_is_the_one_the_new_code_builds(agent, mode, tmp_path):
    golden = EXEC / f"{agent}-{mode}"
    git = tmp_path / "git"
    plant_git(git)
    planted = planted_environment(git)
    spec, paths = cell(tmp_path / "cell", mode, read(EXEC / "claude-mm" / "instructions.md"), read(EXEC / "codex-mm" / "instructions.md"))
    with mock.patch.dict(os.environ, planted, clear=True):
        if agent == "claude":
            argv = claude.prepare(spec, command=["<claude>"])
            env = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation)
        else:
            argv = codex.prepare(spec, "gpt-6.1-terra", "medium", MODELS, ["<codex>"])
            env = common.path_with(spec.bin_dir)
    place = placer({**paths, "PYTHON": sys.executable, "GIT": git})
    assert place(argv) == json.loads(read(golden / "argv.json"))
    assert dict(sorted(place(env).items())) == json.loads(read(golden / "env.json"))
    made = {p.name for p in paths["RUN"].iterdir()}
    assert made == {p.name for p in golden.iterdir()} - {"argv.json", "env.json"}, "the same files, and no other"
    for name in sorted(made):
        if name == "mcp.json":
            assert place(json.loads(read(paths["RUN"] / name))) == json.loads(read(golden / name))
        elif (agent, mode, name) == ("claude", "files", "instructions.md"):
            assert read(paths["RUN"] / name) == files_now(read(golden / name))
        else:
            assert (paths["RUN"] / name).read_bytes() == (golden / name).read_bytes(), name
    assert paths["WORK"].is_dir()


# -- a server with the agent's own tools: no golden, so the plan's table written out by hand -----------------------------------

CLAUDE_BOTH_ARGV = [
    "<claude>", "-p", "--verbose", "--output-format", "stream-json", "--restricted", "--strict-mcp-config", "--permission-mode",
    "dontAsk", "--no-session-persistence", "--disable-slash-commands", "--model", "sonnet", "--effort", "high", "--max-budget-usd", "5",
    "--settings", "<RUN>\\settings.json", "--max-turns", "60", "--append-system-prompt-file", "<RUN>\\instructions.md",
    "--mcp-config", "<RUN>\\mcp.json",
    "--tools", "Read,Glob,Grep,Bash",
    "--allowedTools", "Read", "Glob", "Grep", "Bash(ffmpeg *)", "mcp__mm__ask", "mcp__mm__find", "mcp__mm__look", "mcp__mm__label",
    "--add-dir", "<DATA>",
]
CLAUDE_SETTINGS = ('{"disableAllHooks": true, "enabledPlugins": {"cc-plugin-agents-md@builtin": false, '
                   '"cc-plugin-plugin-authoring@builtin": false}}')
# the files cell's settings (sandbox, no network, the shell's own variables), then the server's, then the features that stay off
CODEX_BOTH_SETTINGS = [
    "sandbox_mode=workspace-write", "approval_policy=never", "web_search=disabled", "skills.include_instructions=false",
    "include_environment_context=true", "include_permissions_instructions=true", "include_collaboration_mode_instructions=false",
    "include_apps_instructions=false", "sandbox_workspace_write.network_access=false", "windows.sandbox=unelevated",
    "shell_environment_policy.inherit=core", "shell_environment_policy.exclude=['PATH']",
    "model_reasoning_effort=medium", "model_catalog_json='<RUN>/catalog.json'", "model_instructions_file='<RUN>/instructions.md'",
    "shell_environment_policy.set.PATH='<BIN>;C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\Wbem;"
    "C:\\Windows\\System32\\WindowsPowerShell\\v1.0'",
    "shell_environment_policy.set.MULTIMEM_HOME='<WORK>/no-mm-home'",
    "mcp_servers.mm.command='<PYTHON>'",
    "mcp_servers.mm.args=['-m','multimem.cli','--home','<HOME>','mcp','--tools','ask,find,look,label']",
    "mcp_servers.mm.required=true", "mcp_servers.mm.default_tools_approval_mode=approve", "mcp_servers.mm.startup_timeout_sec=60",
    "mcp_servers.mm.tool_timeout_sec=900", "mcp_servers.mm.env.CLAUDE_PROJECT_DIR='<WORK>'",
]
CODEX_BOTH_OFF = ["browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use", "image_generation", "apps",
                  "plugins", "multi_agent", "multi_agent_v2", "skill_search", "tool_suggest", "goals", "hooks",
                  "workspace_dependencies", "remote_plugin", "sleep_tool"]  # the shell, the unified exec and view_image are on


@windows_only
@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_a_server_with_own_tools_gets_the_files_exec_plus_the_server_by_the_plans_table(agent, tmp_path):
    git = tmp_path / "git"
    plant_git(git)
    planted = planted_environment(git)
    claude_text, codex_text = read(EXEC / "claude-mm" / "instructions.md"), read(EXEC / "codex-mm" / "instructions.md")
    spec, paths = cell(tmp_path / "cell", "mm+files", claude_text, codex_text)
    place = placer({**paths, "PYTHON": sys.executable, "GIT": git})
    with mock.patch.dict(os.environ, planted, clear=True):
        if agent == "claude":
            argv = claude.prepare(spec, command=["<claude>"])
            env = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation)
        else:
            argv = codex.prepare(spec, "gpt-6.1-terra", "medium", MODELS, ["<codex>"])
            env = common.path_with(spec.bin_dir)
    run = paths["RUN"]
    if agent == "claude":
        assert place(argv) == CLAUDE_BOTH_ARGV
        assert read(run / "settings.json") == CLAUDE_SETTINGS
        assert place(json.loads(read(run / "mcp.json"))) == json.loads(read(EXEC / "claude-mm" / "mcp.json"))
        # the files line first, then the server's: each as it was recorded
        assert read(run / "instructions.md") == files_now(read(EXEC / "claude-files" / "instructions.md")) + "\n\n" + claude_text
        # the own tools pick the environment: the files cell's, whatever else the exec has
        assert dict(sorted(place(env).items())) == json.loads(read(EXEC / "claude-files" / "env.json"))
        assert {p.name for p in run.iterdir()} == {"instructions.md", "settings.json", "mcp.json"}
    else:
        expected = ["<codex>", "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "--strict-config",
                    "-m", "gpt-6.1-terra"]
        for setting in CODEX_BOTH_SETTINGS:
            expected += ["-c", setting]
        for feature in CODEX_BOTH_OFF:
            expected += ["--disable", feature]
        assert place(argv) == [*expected, "-C", "<WORK>", "--", "-"]
        assert read(run / "instructions.md") == read(EXEC / "codex-files" / "instructions.md") + "\n\n" + codex_text
        assert (run / "catalog.json").read_bytes() == (EXEC / "codex-mm" / "catalog.json").read_bytes()
        assert dict(sorted(place(env).items())) == json.loads(read(EXEC / "codex-mm" / "env.json"))
        assert {p.name for p in run.iterdir()} == {"instructions.md", "catalog.json"}


# -- the 110 recorded pilot execs ----------------------------------------------------------------------------------------------

def pilot_execs() -> list[dict]:
    return [e for e in json.loads(read(PILOT / "execs.json"))["execs"] if not e["excluded"]]


def recorded_text(sha: str) -> str:
    return read(PILOT / "instructions" / f"{sha[:12]}.md")


def cell_mode(e: dict) -> str:
    return "files" if e["dir"] == "files" else "mm"


def pilot_cell(base: Path, mode: str, deny: tuple[str, ...] = ()):
    """The spec of a pilot cell, its server's text taken apart from the instruction texts of the pilot itself."""
    sha = {a: next(e["instructions_sha256"] for e in pilot_execs() if e["agent"] == a and e["dir"] != "files") for a in ("claude", "codex")}
    return cell(base, mode, recorded_text(sha["claude"]), recorded_text(sha["codex"]), deny)


def test_the_pilot_has_the_cells_the_oracle_covers_and_leaves_out_the_overwritten_exec():
    assert Counter((e["agent"], e["dir"]) for e in pilot_execs()) == Counter(
        {("claude", "files"): 30, ("codex", "files"): 30, ("claude", "mm-d1-3B"): 24, ("codex", "mm-d1-3B"): 24})


def test_the_settings_and_instructions_every_pilot_exec_had_are_what_the_agents_write(tmp_path):
    wrong = []
    for agent, writer in (("claude", lambda s: claude.prepare(s, command=["claude"])),
                          ("codex", lambda s: codex.prepare(s, "gpt-6.1-terra", "medium", MODELS, ["codex"]))):
        written = {}
        for mode in ("files", "mm"):
            spec, paths = pilot_cell(tmp_path / f"{agent}-{mode}", mode)
            writer(spec)
            written[mode] = {name: sha256((paths["RUN"] / name).read_bytes()) for name in ("instructions.md", "settings.json")
                             if (paths["RUN"] / name).exists()}
        for e in pilot_execs():
            if e["agent"] != agent:
                continue
            mine = written[cell_mode(e)]
            now = e["instructions_sha256"]
            if agent == "claude" and cell_mode(e) == "files":
                now = sha256(files_now(recorded_text(now)))
            if mine["instructions.md"] != now:
                wrong.append((agent, e["dir"], e["qid"], "instructions"))
            if agent == "claude" and mine["settings.json"] != e["settings_sha256"]:
                wrong.append((agent, e["dir"], e["qid"], "settings"))
    assert not wrong, wrong[:5]


def test_the_mcp_config_of_every_claude_server_exec_is_what_the_agent_writes(tmp_path):
    spec, paths = pilot_cell(tmp_path / "cell", "mm")
    placed = json.loads(json.dumps(placer({**paths, "PYTHON": sys.executable})(claude.mcp_config(spec.server))))
    recorded = [e["mcp"] for e in pilot_execs() if e["agent"] == "claude" and e["dir"] != "files"]
    assert len(recorded) == 24 and all(placed == mcp for mcp in recorded)
    assert [e["mcp"] for e in pilot_execs() if e["agent"] == "claude" and e["dir"] == "files"] == [None] * 30


def test_every_recorded_init_event_is_the_session_the_agent_asks_for(tmp_path):
    wrong = []
    for e in pilot_execs():
        if e["agent"] != "claude":
            continue
        spec, _ = pilot_cell(tmp_path / e["dir"], cell_mode(e))
        event = {"type": "system", "subtype": "init", **e["init"]}
        if set(e["init"]["tools"]) != set(claude.session_tools(spec)) or claude.check_init(event, spec):
            wrong.append((e["dir"], e["qid"]))
    assert not wrong, wrong[:5]
    spec, _ = pilot_cell(tmp_path / "other", "mm")
    files = next(e for e in pilot_execs() if e["agent"] == "claude" and e["dir"] == "files")
    assert claude.check_init({"type": "system", "subtype": "init", **files["init"]}, spec), "the files session is not the server's"


def test_the_catalog_of_every_recorded_codex_exec_is_what_the_agent_writes_from_its_table():
    """The recorded catalog is the real `codex debug models` table with the model patched, so patching it again must give it back
    byte for byte (the patch is the same and the writing is the same). It is read from the recorded run, not copied."""
    if not RECORDED_RUN.is_dir():
        pytest.skip(f"{RECORDED_RUN} is not here: the recorded Codex catalogs (648 KB each) are read from it")
    codex_execs = [e for e in pilot_execs() if e["agent"] == "codex"]
    assert Counter(e["catalog_sha256"][:8] for e in codex_execs) == {"ac5d2260": 30, "e67870b7": 24}
    wrong = []
    for e in codex_execs:
        file = RECORDED_RUN / "execs" / "codex" / e["dir"] / e["qid"] / "catalog.json"
        raw = file.read_bytes()
        if sha256(raw) != e["catalog_sha256"]:
            wrong.append((e["dir"], e["qid"], "the recorded file is not the one captured"))
            continue
        mine = json.dumps(codex.catalog(json.loads(raw), "gpt-6.1-terra", "medium")).encode("utf-8")
        if sha256(mine) != e["catalog_sha256"]:
            wrong.append((e["dir"], e["qid"], "patched again, it is not itself"))
    assert not wrong, wrong[:5]


# -- the recorded streams of eight execs, through the watchers and the readers ------------------------------------------------

REPLAYS = sorted(p.parent for p in (PILOT / "replay").rglob("result.json"))


def events_of(folder: Path) -> list[dict]:
    return [json.loads(line) for line in read(folder / "events.jsonl").splitlines()]


@pytest.mark.parametrize("folder", REPLAYS, ids=[f"{p.parts[-3]}-{p.parts[-2]}-{p.parts[-1]}" for p in REPLAYS])
def test_a_recorded_stream_passes_the_watcher_and_reads_as_recorded(folder, tmp_path):
    agent, directory = folder.parts[-3], folder.parts[-2]
    mode = "files" if directory == "files" else "mm"
    spec, _ = pilot_cell(tmp_path, mode, deny=(".claude", ".codex", "answers"))
    result = json.loads(read(folder / "result.json"))
    events = events_of(folder)
    if agent == "claude":
        watch = claude.watcher(spec)
        assert [p for p in map(watch, events) if p] == [], "a recorded exec that ended well raises nothing"
        done = claude.result_of(events, 0, "")
        assert done["problem"] is None and answer_of(done["final"]) == result["answer"]
        assert (done["turns"], done["tokens"], done["cost_usd"], done["denials"], done["used_percent"], done["model"]) == (
            result["turns"], result["tokens"], result["cost_usd"], result["denials"], result["used_percent"], result["agent_model"])
        assert Counter(t["tool"] for t in claude.tools_of(events)) == Counter(result["tools"])
    else:
        assert [p for p in (codex.check_event(e, spec) for e in events) if p] == []
        assert codex.problem_of(0, events, "") is None
        turns, tokens = codex.usage_of(events)
        assert (turns, tokens) == (result["turns"], result["tokens"])
        assert Counter(t["tool"] for t in codex.tools_of(events)) == Counter(result["tools"])
        said = [e["item"]["text"] for e in events if e["type"] == "item.completed" and e["item"]["type"] == "agent_message"]
        assert answer_of(said[-1]) == result["answer"]


def test_a_recorded_files_stream_with_one_command_at_a_denied_path_is_an_escape(tmp_path):
    """The recorded streams have no escape (the pilot had none), so one is planted: a recorded command edited to name a denied folder."""
    deny = (".claude", ".codex", "answers")
    claude_folder = PILOT / "replay" / "claude" / "files" / "records-log-count-1"
    spec, _ = pilot_cell(tmp_path / "c", "files", deny)
    events = events_of(claude_folder)
    first = next(i for i, e in enumerate(events) if e["type"] == "assistant"
                 and any(b.get("type") == "tool_use" for b in claude.blocks_of(e)))
    planted = json.loads(json.dumps(events))
    block = next(b for b in claude.blocks_of(planted[first]) if b.get("type") == "tool_use")
    block["input"] = {"file_path": "C:\\Users\\dan\\.claude\\settings.json"} if block["name"] == "Read" else {
        "pattern": "x", "path": "C:\\Users\\dan\\.claude"}
    block["name"] = "Read" if block["name"] == "Read" else "Grep"
    watch = claude.watcher(spec)
    found = [p for p in map(watch, planted) if p]
    assert found and found[0].startswith("escaped: ") and ".claude" in found[0]

    codex_folder = PILOT / "replay" / "codex" / "files" / "records-log-count-1"
    spec, _ = pilot_cell(tmp_path / "x", "files", deny)
    events = events_of(codex_folder)
    planted = json.loads(json.dumps(events))
    shell = next(e for e in planted if e.get("item", {}).get("type") == "command_execution")
    shell["item"]["command"] += " ; type C:\\Users\\dan\\.codex\\auth.json"
    found = [p for p in (codex.check_event(e, spec) for e in planted) if p]
    assert found and found[0] == "shell reached .codex"
