"""What an agent is given for an mm exec: the instructions the system serves (`MmSession.serve`: the base line, the skill and the model's notes) and the
exec each agent builds from them (command line, environment, `mcp.json`, settings) for Claude and for Codex, in `mm` (the system's tools alone) and
`mm+files` (and the agent's own), against expected values written out here. The skill and the notes are read from the installed MultimediaSearch, so
a skill that changes does not fail a test; what the instructions are made of does.

Also the tests of what reads the installed skill:
- the notes of every model fit Claude's command line;
- the instructions hold the skill and the notes of the model, or the skill alone when the model has none;
- a notes file is a plain file name in the skill's models folder."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mms")

import fakes
from agent_spec import spec_of
from model_folders import model_folder

from mms import decider
from mms.adapter import INSTRUCTIONS

from mmsb import systems
from mmsb.core.agents import claude, codex, common
from mmsb.core.agents.spec import SessionSpec
from mmsb.core.contract import Case, Served
from mmsb.core.schedule import agent_off_gpu
from mmsb.systems.mm.models import Model
from mmsb.systems.mm.system import BASE, CLOCK_NOTE, NOTES_DIR, SKILL, Mm, notes_text

D3 = decider.known("d3-edge")[1]  # notes d3.md
SKILL_TEXT = SKILL.read_text(encoding="utf-8")
MM_TOOLS = ["mcp__mm__ask", "mcp__mm__find", "mcp__mm__look", "mcp__mm__label"]  # what Claude calls the server's four tools
MODELS = {"models": [{"slug": "gpt-6.1-terra", "supported_reasoning_levels": [{"effort": "medium"}]}]}  # what `codex debug models` lists

# what Codex is given, `-c` by `-c`: the read-only settings of an exec with no shell, and the settings of an exec with one (a sandbox with no
# network and a shell that inherits only Codex's core variables), then the server's
CODEX_READ_ONLY = ["sandbox_mode=read-only", "approval_policy=never", "web_search=disabled", "skills.include_instructions=false",
                   "include_environment_context=false", "include_permissions_instructions=false",
                   "include_collaboration_mode_instructions=false", "include_apps_instructions=false"]
CODEX_OWN_TOOLS = ["sandbox_mode=workspace-write", "approval_policy=never", "web_search=disabled", "skills.include_instructions=false",
                   "include_environment_context=true", "include_permissions_instructions=true",
                   "include_collaboration_mode_instructions=false", "include_apps_instructions=false",
                   "sandbox_workspace_write.network_access=false", "windows.sandbox=unelevated", "shell_environment_policy.inherit=core",
                   "shell_environment_policy.exclude=['PATH']"]
# the features `--disable`d: with the agent's own tools these, and without them also the shell, the unified exec and `view_image`
CODEX_OFF = ["browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use", "image_generation", "apps", "plugins",
             "multi_agent", "multi_agent_v2", "skill_search", "tool_suggest", "goals", "hooks", "workspace_dependencies", "remote_plugin",
             "sleep_tool"]
CODEX_OFF_NO_SHELL = ["shell_tool", "unified_exec", "browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use",
                      "image_generation", "apps", "plugins", "multi_agent", "multi_agent_v2", "view_image", "skill_search", "tool_suggest",
                      "goals", "hooks", "workspace_dependencies", "remote_plugin", "sleep_tool"]


@pytest.fixture(scope="module")
def opened(week, store, tmp_path_factory):
    """A session of the system with d1-3B's facts (its skill notes) and the fake decider, over a copy of the tiny week's store; closed at the end."""
    _, dump, _ = week
    with fakes.Opened(store, dump, tmp_path_factory.mktemp("homes"), system=fakes.MM, variant="d1-3B") as session:
        yield session


@pytest.fixture
def git(tmp_path, monkeypatch) -> Path:
    """A Git for Windows folder tree, with `claude.git_root` pointed at it: the real one is the owner's."""
    root = tmp_path / "git"
    for sub in ("cmd", "bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("")
    (root / "usr" / "bin" / "bash.exe").write_text("")
    monkeypatch.setattr(claude, "git_root", lambda: root)
    return root


def case_in(base: Path) -> Case:
    work = base / "WORK"
    work.mkdir(parents=True, exist_ok=True)
    return Case("image-photo-sign-1", 1.0, None, ("phone",), 0.0, 1.0, "g", 0.0, 0, work, base / "RUN")


def cell(base: Path, mode: str, opened, **fields) -> SessionSpec:
    """The spec of an exec of `mode` (`mm` or `mm+files`) with what the session serves for a case, its folders under `base`."""
    case = case_in(base)
    return spec_of(mode, base, work=case.work, served=opened.serve(case), isolation=systems.isolation(), model="sonnet", effort="high", **fields)


def prepare(agent: str, spec: SessionSpec) -> list[str]:
    """What the agent writes into the exec's folder and the command line it builds."""
    if agent == "claude":
        return claude.prepare(spec, command=["<claude>"])
    return codex.prepare(spec, "gpt-6.1-terra", "medium", MODELS, ["<codex>"])


def written(agent: str, spec: SessionSpec) -> str:
    prepare(agent, spec)
    return (spec.run_dir / "instructions.md").read_text(encoding="utf-8")


def toml(value) -> str:
    """A TOML literal string with forward slashes, as Codex is given a path."""
    return "'" + str(value).replace("\\", "/") + "'"


def claude_argv(spec: SessionSpec, *tools: str) -> list[str]:
    run = spec.run_dir
    return ["<claude>", "-p", "--verbose", "--output-format", "stream-json", "--restricted", "--strict-mcp-config", "--permission-mode",
            "dontAsk", "--no-session-persistence", "--disable-slash-commands", "--model", "sonnet", "--effort", "high", "--max-budget-usd", "5",
            "--settings", str(run / "settings.json"), "--max-turns", "60", "--append-system-prompt-file", str(run / "instructions.md"),
            "--mcp-config", str(run / "mcp.json"), *tools]


def codex_argv(spec: SessionSpec, home: Path, settings: list[str], off: list[str], shell: list[str] = ()) -> list[str]:
    run = spec.run_dir
    server = ["mcp_servers.mm.command=" + toml(sys.executable),
              f"mcp_servers.mm.args=['-m','mms.cli','--home',{toml(home)},'mcp','--tools','ask,find,look,label']",
              "mcp_servers.mm.required=true", "mcp_servers.mm.default_tools_approval_mode=approve", "mcp_servers.mm.startup_timeout_sec=60",
              "mcp_servers.mm.tool_timeout_sec=900", "mcp_servers.mm.env.CLAUDE_PROJECT_DIR=" + toml(spec.work)]
    every = [*settings, "model_reasoning_effort=medium", "model_catalog_json=" + toml(run / "catalog.json"),
             "model_instructions_file=" + toml(run / "instructions.md"), *shell, *server]
    return ["<codex>", "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "--strict-config", "-m", "gpt-6.1-terra",
            *(arg for setting in every for arg in ("-c", setting)), *(arg for feature in off for arg in ("--disable", feature)),
            "-C", str(spec.work), "--", "-"]


# -- the text the system serves ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_the_instructions_are_the_base_line_the_skill_and_the_notes_of_the_model_and_codex_is_told_the_servers_own_words_too(agent, opened, tmp_path):
    spec = cell(tmp_path, "mm", opened)
    served = spec.served
    assert (served.base_line, served.holder, served.server.name, served.server.tools) == (BASE, "mm", "mm", ("ask", "find", "look", "label"))
    assert served.parts == (SKILL_TEXT, notes_text("d1_3b.md")) and served.server_instructions == INSTRUCTIONS
    told = [INSTRUCTIONS] if agent == "codex" else []  # Claude gets the server's own words through the server
    assert written(agent, spec) == "\n\n".join([BASE, *told, SKILL_TEXT, notes_text("d1_3b.md")])


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_with_the_agents_own_tools_the_files_line_comes_first_and_the_rest_is_the_mm_cells(agent, opened, tmp_path):
    files_line = claude.BASE_FILES if agent == "claude" else codex.BASE_FILES
    mm = written(agent, cell(tmp_path / "mm", "mm", opened))
    assert written(agent, cell(tmp_path / "both", "mm+files", opened)) == files_line + "\n\n" + mm


def test_a_clock_later_than_the_question_by_over_an_hour_is_noted_and_an_earlier_one_is_not(opened, tmp_path):
    case = case_in(tmp_path)
    at = lambda now: dataclasses.replace(case, now=now)  # noqa: E731
    wall = opened.context.wall()
    assert opened.serve(at(wall)).clock_note is None and opened.serve(at(wall - 3599.0)).clock_note is None
    assert opened.serve(at(wall - 3601.0)).clock_note == CLOCK_NOTE == "mm's clock reads a later date than now: give since and until as dates and times, not as ages or \"now\"."
    assert opened.serve(at(1.0)).clock_note == CLOCK_NOTE, "a question of long ago"


# -- the exec the agents build from it --------------------------------------------------------------------------------------------

def test_claude_is_given_the_servers_four_tools_and_no_built_in_one_with_the_mcp_config_the_system_serves(opened, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-owner")
    monkeypatch.setenv("KEEP_ME", "kept")
    spec = cell(tmp_path, "mm", opened)
    assert prepare("claude", spec) == claude_argv(spec, "--tools", "", "--allowedTools", *MM_TOOLS)
    run = spec.run_dir
    assert {p.name for p in run.iterdir()} == {"instructions.md", "settings.json", "mcp.json"} and spec.work.is_dir()
    assert json.loads((run / "settings.json").read_text(encoding="utf-8")) == {
        "disableAllHooks": True, "enabledPlugins": {"cc-plugin-agents-md@builtin": False, "cc-plugin-plugin-authoring@builtin": False}}
    assert json.loads((run / "mcp.json").read_text(encoding="utf-8")) == {"mcpServers": {"mm": {
        "command": sys.executable, "args": ["-m", "mms.cli", "--home", str(opened.home), "mcp", "--tools", "ask,find,look,label"],
        "env": {"CLAUDE_PROJECT_DIR": str(spec.work)}}}}
    env = claude.env_of(spec.bin_dir)
    assert env["PATH"].split(os.pathsep)[0] == str(spec.bin_dir) and "ANTHROPIC_API_KEY" not in env and env["KEEP_ME"] == "kept"
    assert all(env[key] == value for key, value in claude.ENV_SET.items())


def test_claude_with_own_tools_is_given_those_and_the_servers_and_the_folder_of_the_pictures_look_makes_and_no_cuda_device(opened, tmp_path, monkeypatch, git):
    models = tmp_path / "models"
    model_folder(models, "d1-3B")
    description = Mm(models_dir=models).describe("d1-3B")
    assert description.needs_gpu and agent_off_gpu(True, description, False), "d1-3B reads on the GPU, so the agent is kept off it with no flag"
    for name, value in (("MMS_HOME", str(tmp_path / "owners-home")), ("MMS_MODELS", "D:/models"), ("VIRTUAL_ENV", str(tmp_path / ".venv")),
                        ("ANTHROPIC_API_KEY", "sk-owner"), ("CUDA_VISIBLE_DEVICES", "0")):
        monkeypatch.setenv(name, value)
    spec = cell(tmp_path / "cell", "mm+files", opened, off_gpu=True)
    assert prepare("claude", spec) == claude_argv(
        spec, "--tools", "Read,Glob,Grep,Bash", "--allowedTools", "Read", "Glob", "Grep", "Bash(ffmpeg *)", *MM_TOOLS,
        "--add-dir", str(spec.data), "--add-dir", str(opened.home / "looks"))
    assert {p.name for p in spec.run_dir.iterdir()} == {"instructions.md", "settings.json", "mcp.json"}
    env = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation, spec.off_gpu)
    assert env["PATH"].split(os.pathsep) == [str(spec.bin_dir), str(git / "cmd"), str(git / "bin"), str(git / "usr" / "bin"), *common.system_path()]
    assert env["MMS_HOME"] == str(spec.work / "no-mm-home") and "MMS_MODELS" not in env, "the owner's mms is out of reach"
    assert not {"VIRTUAL_ENV", "ANTHROPIC_API_KEY"} & set(env) and env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["CLAUDE_CODE_GIT_BASH_PATH"] == str(git / "bin" / "bash.exe")


def test_codex_is_given_the_server_and_the_read_only_settings_with_no_shell(opened, tmp_path, monkeypatch):
    spec = cell(tmp_path, "mm", opened)
    assert prepare("codex", spec) == codex_argv(spec, opened.home, CODEX_READ_ONLY, CODEX_OFF_NO_SHELL)
    assert {p.name for p in spec.run_dir.iterdir()} == {"instructions.md", "catalog.json"} and spec.work.is_dir()
    catalog = json.loads((spec.run_dir / "catalog.json").read_text(encoding="utf-8"))["models"]
    assert [(m["slug"], m["tool_mode"]) for m in catalog] == [("gpt-6.1-terra", "direct")]
    assert common.path_with(spec.bin_dir)["PATH"].split(os.pathsep)[0] == str(spec.bin_dir)


def test_codex_with_own_tools_has_a_shell_that_sees_no_mms_and_no_cuda_device_beside_the_server(opened, tmp_path):
    spec = cell(tmp_path, "mm+files", opened, off_gpu=True)
    shell = ["shell_environment_policy.set.PATH='" + os.pathsep.join([str(spec.bin_dir), *common.system_path()]) + "'",
             "shell_environment_policy.set.MMS_HOME=" + toml(spec.work / "no-mm-home"), "shell_environment_policy.set.CUDA_VISIBLE_DEVICES='-1'"]
    assert prepare("codex", spec) == codex_argv(spec, opened.home, CODEX_OWN_TOOLS, CODEX_OFF, shell)
    assert {p.name for p in spec.run_dir.iterdir()} == {"instructions.md", "catalog.json"}


# -- what reads the installed skill -----------------------------------------------------------------------------------------------

def served_with(about, week, store, tmp_path) -> Served:
    """What a session of a model with these facts serves (the fake decider answers for it)."""
    _, dump, _ = week
    system = Mm(models={"m": Model("m", fakes.FAKE, about)})
    with fakes.Opened(store, dump, tmp_path / "homes", system=system, variant="m") as session:
        return session.serve(case_in(tmp_path))


def test_the_notes_of_every_model_fit_claudes_command_line(opened, tmp_path, monkeypatch):
    monkeypatch.setattr(claude, "PROMPT_FILE", False)  # the text then goes in the command line, about 26k characters, under Windows' 32k
    notes = sorted(p.name for p in NOTES_DIR.iterdir() if p.name.endswith(".md"))
    assert {"d1.md", "d1_3b.md", "d3.md"} <= set(notes)
    spec = cell(tmp_path, "mm", opened)
    for name in notes:
        served = Served(spec.served.server, spec.served.base_line, (SKILL_TEXT, notes_text(name)), None, "mm", None, {})
        claude.prepare(dataclasses.replace(spec, served=served), command=["claude"])


def test_the_instructions_hold_the_skill_and_the_notes_of_the_model_and_no_notes_leave_the_skill_alone(week, store, opened, tmp_path):
    text = written("claude", cell(tmp_path / "d1-3b", "mm", opened))
    assert SKILL_TEXT in text and notes_text("d1_3b.md") in text and notes_text("d1.md") not in text

    def with_model(about, folder: str, agent: str) -> tuple[Served, str]:
        served = served_with(about, week, store, tmp_path / folder)
        return served, written(agent, spec_of("mm", tmp_path / folder, served=served, model="sonnet", effort="high"))

    bare, bare_text = with_model(dataclasses.replace(D3, notes=""), "bare", "claude")
    assert bare_text == "\n\n".join([BASE, SKILL_TEXT]) and bare.parts == (SKILL_TEXT,)
    _, codex_text = with_model(D3, "d3", "codex")
    assert notes_text("d3.md") in codex_text and notes_text("d1.md") not in codex_text
    assert SKILL_TEXT in codex_text and codex_text.startswith(BASE + "\n\n")
    _, omni_text = with_model(decider.known("d1-Omni-2B")[1], "d1", "codex")
    assert notes_text("d1.md") in omni_text, "d1's notes are those of d1-Omni-2B, which the skill points to"


def test_notes_are_a_plain_file_name_in_the_skills_models_folder():
    for name in ("../SKILL.md", "..\\SKILL.md", "..", ".", "a/b.md"):
        with pytest.raises(ValueError, match="plain file name"):
            notes_text(name)
    assert notes_text("d1_3b.md") != notes_text("d3.md") != notes_text("d1.md")
