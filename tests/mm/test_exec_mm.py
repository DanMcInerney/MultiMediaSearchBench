"""Oracle 3 (plan 7.3) for the mm cells of both agents, and the skill tests of Multimem's test_agents.py (and test_codex.py) that read the
installed skill: what `Mm.serve` gives the agents, and the exec they build from it, is the exec the old code built.

The expected values are not the new code's: they are the goldens that `capture_exec.py` wrote from the old claude_run.py and codex_run.py
(argv, environment, instructions, settings, mcp.json, catalog; tests/oracles/recorded/exec) and the 48 recorded pilot mm execs
(tests/oracles/recorded/pilot-s1/execs.json and its instruction texts). The one intended difference, plan 7.3 difference 3, is the skill:
five of its lines changed in Multimem between the pilot and the commit the goldens were made at (41 characters). It is an exact line map, so
that a skill that changes again fails here, saying which lines.

Ported from tests/test_agents.py of Multimem at fe0ece5 (plan 7.4, oracle 4), in that file's order, for what reads the installed skill:
- test_claudes_command_lines_give_each_arm_its_tools_and_nothing_of_the_owners: the last lines, that the notes of every model fit the line;
- test_claudes_environment_drops_the_api_key_and_the_instructions_hold_the_skill_and_the_notes: the instructions' half;
- test_codexs_instructions_hold_the_models_notes_and_the_files_arms_line_alone: the mm half.
What changed: the instructions are the agents' join of `Served` (plan 4.4), `notes_text` is the system's. `guidance="none"` went with the old recall CLI.
`test_codex_recalls_the_day` of test_codex.py is not ported: it starts a real Codex."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("multimem")

import fakes
from agent_spec import spec_of
from capture_exec import PLACEHOLDERS, placer, plant_git, planted_environment
from model_folders import model_folder
from test_exec import CLAUDE_BOTH_ARGV, CLAUDE_SETTINGS, MODELS, files_now, pilot_execs, read, recorded_text

from multimem import decider

from mmmb.core.agents import claude, codex, common
from mmmb.core.contract import Case, Served
from mmmb.core.schedule import agent_off_gpu
from mmmb.systems.mm.models import Model
from mmmb.systems.mm.system import BASE, CLOCK_NOTE, NOTES_DIR, SKILL, Mm, notes_text

RECORDED = Path(__file__).parent.parent / "oracles" / "recorded"
EXEC, PILOT = RECORDED / "exec", RECORDED / "pilot-s1"
D3 = decider.known("d3-edge")[1]  # notes d3.md
windows_only = pytest.mark.skipif(os.name != "nt", reason="the goldens are what the old code builds on Windows (backslashes, Git)")
SKILL_TEXT = SKILL.read_text(encoding="utf-8")
# plan 7.3, difference 3: the five lines of the skill and the notes of d1-3B that Multimem changed since the pilot ("nervelet simulator" became
# "robot simulator", and "this machine" the developer's), as the pilot recorded them and as they are now. Each is in the recorded instructions once.
DRIFT = {
    'they are: "Lines of the nervelet simulator\'s log; height_m is in metres." With a picture or a sound it is the only text':
        'they are: "Lines of the robot simulator\'s log; height_m is in metres." With a picture or a sound it is the only text',
    "The notes give the model's cost for a frame, a sound and a page of text, and say whether it was measured on this":
        "The notes give the model's cost for a frame, a sound and a page of text, and say whether it was measured on the developer's",
    "key, and it reads any number of questions in one pass. Its accuracy has not been measured on this machine. The advice":
        "key, and it reads any number of questions in one pass. Its accuracy has not been measured on the developer's machine. The advice",
    "  probabilities on this machine. The card calls its answers calibrated but gives no figures. Treat the first sightings":
        "  probabilities on the developer's machine. The card calls its answers calibrated but gives no figures. Treat the first sightings",
    "Measured on this machine with one yes/no question; four questions cost about the same as one.":
        "Measured on the developer's machine with one yes/no question; four questions cost about the same as one.",
}


@pytest.fixture(scope="module")
def opened(week, store, tmp_path_factory):
    """A session of the system with d1-3B's facts (its skill notes) and the fake decider, over a copy of the tiny week's store; closed at the end."""
    _, dump, _ = week
    with fakes.Opened(store, dump, tmp_path_factory.mktemp("homes"), system=fakes.MM, variant="d1-3B") as session:
        yield session


def case_in(base: Path) -> Case:
    work = base / "WORK"
    work.mkdir(parents=True, exist_ok=True)
    return Case("image-photo-sign-1", 1.0, None, ("phone",), 0.0, 1.0, "g", 0.0, 0, work, base / "RUN")


def instructions_of(served: Served, agent: str) -> str:
    """The text the agents write to instructions.md for a served server (plan 4.4)."""
    parts = [served.base_line, *([served.server_instructions] if agent == "codex" else []), *served.parts]
    return "\n\n".join(parts)


# -- the text the system serves ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_the_instructions_the_system_serves_are_the_goldens_the_old_code_wrote(agent, opened, tmp_path):
    served = opened.serve(case_in(tmp_path))
    assert instructions_of(served, agent) == read(EXEC / f"{agent}-mm" / "instructions.md")
    assert (served.base_line, served.holder, served.server.name, served.server.tools) == (BASE, "mm", "mm", ("ask", "find", "look", "label"))
    assert served.parts[0] == SKILL_TEXT and served.parts[1] == notes_text("d1_3b.md") and len(served.parts) == 2


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_the_instructions_the_pilot_recorded_are_the_ones_the_system_serves_but_for_the_five_lines_of_the_skill_that_changed(agent, opened, tmp_path):
    shas = {e["instructions_sha256"] for e in pilot_execs() if e["agent"] == agent and e["dir"] != "files"}
    assert len(shas) == 1, "all 24 mm execs of an agent had the same instructions"
    recorded = recorded_text(shas.pop()).splitlines()
    served = instructions_of(opened.serve(case_in(tmp_path)), agent).splitlines()
    assert len(recorded) == len(served)
    changed = [line for line in recorded if line in DRIFT]
    assert sorted(changed) == sorted(DRIFT), "each of the five recorded lines is in the pilot's instructions once"
    assert [DRIFT.get(line, line) for line in recorded] == served, "and nothing else differs: if the skill changed again, a line is missing from the map"
    assert sum(len(new) - len(old) for old, new in DRIFT.items()) == 41, "plan section 1: 25,821 characters recorded, 25,862 now"


def test_a_clock_later_than_the_question_by_over_an_hour_is_noted_and_an_earlier_one_is_not(opened, tmp_path):
    case = case_in(tmp_path)
    at = lambda now: dataclasses.replace(case, now=now)  # noqa: E731
    wall = opened.context.wall()
    assert opened.serve(at(wall)).clock_note is None and opened.serve(at(wall - 3599.0)).clock_note is None
    assert opened.serve(at(wall - 3601.0)).clock_note == CLOCK_NOTE == "mm's clock reads a later date than now: give since and until as dates and times, not as ages or \"now\"."
    assert opened.serve(at(1.0)).clock_note == CLOCK_NOTE, "a question of long ago, as the recorded EgoLife run's"


# -- the exec the agents build from it --------------------------------------------------------------------------------------------

def cell(base: Path, mode: str, opened, agent_text: bool = True):
    """The spec of a cell of the pilot: the system's `Served`, in the folders the goldens call RUN, WORK, HOME, DATA and BIN (HOME is the
    session's own)."""
    paths = {name: base / name for name in PLACEHOLDERS}
    paths["HOME"] = opened.home
    served = opened.serve(case_in(base))
    return spec_of(mode, base, run_dir=paths["RUN"], work=paths["WORK"], bin_dir=paths["BIN"], served=served,
                   data=paths["DATA"] if mode.endswith("files") else None), paths


@windows_only
@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_the_exec_the_old_code_built_for_the_mm_cell_is_the_one_built_from_what_the_system_serves(agent, opened, tmp_path):
    golden = EXEC / f"{agent}-mm"
    git = tmp_path / "git"
    plant_git(git)
    planted = planted_environment(git)
    spec, paths = cell(tmp_path / "cell", "mm", opened)
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
        else:
            assert (paths["RUN"] / name).read_bytes() == (golden / name).read_bytes(), name


@windows_only
@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_with_the_agents_own_tools_the_files_line_comes_first_and_the_server_is_the_same(agent, opened, tmp_path):
    """mm+files has no recorded counterpart (plan 4.4, a new cell): its instructions are the files line and then the server's, and its server is
    the mm cell's; the rest of its exec is test_exec.py's, from the plan's table. U13 (plan 6.4, C11): a shell and a model that needs the GPU in one
    cell, so the agent is kept off the GPU whether or not the cells run side by side: its environment and its shell gain `CUDA_VISIBLE_DEVICES=-1`
    and nothing else changes."""
    git = tmp_path / "git"
    plant_git(git)
    models = tmp_path / "models"
    model_folder(models, "d1-3B")
    description = Mm(models_dir=models).describe("d1-3B")
    assert description.needs_gpu and agent_off_gpu(True, description, False), "d1-3B reads on the GPU, so the rule holds with no flag"
    spec, paths = cell(tmp_path / "cell", "mm+files", opened)
    spec = dataclasses.replace(spec, off_gpu=True)
    with mock.patch.dict(os.environ, planted_environment(git), clear=True):
        if agent == "claude":
            argv = claude.prepare(spec, command=["<claude>"])
            env = claude.env_of(spec.bin_dir, spec.own_tools, spec.work, spec.isolation, spec.off_gpu)
        else:
            argv = codex.prepare(spec, "gpt-6.1-terra", "medium", MODELS, ["<codex>"])
    place = placer({**paths, "PYTHON": sys.executable, "GIT": git})
    plain = (EXEC / f"{agent}-files" / "instructions.md").read_text(encoding="utf-8")
    plain = files_now(plain) if agent == "claude" else plain
    assert read(paths["RUN"] / "instructions.md") == plain + "\n\n" + read(EXEC / f"{agent}-mm" / "instructions.md")
    if agent == "claude":
        assert place(argv) == [*CLAUDE_BOTH_ARGV, "--add-dir", "<HOME>\\looks"], "and the folder of the pictures `look` makes, which the system handed it (G1's F4)"
        assert read(paths["RUN"] / "settings.json") == CLAUDE_SETTINGS
        assert place(json.loads(read(paths["RUN"] / "mcp.json"))) == json.loads(read(EXEC / "claude-mm" / "mcp.json"))
        expected_env = {**json.loads(read(EXEC / "claude-files" / "env.json")), "CUDA_VISIBLE_DEVICES": "-1"}
        assert dict(sorted(place(env).items())) == dict(sorted(expected_env.items())), "the files cell's environment, which the own tools pick, and no CUDA device"
    else:
        assert "mcp_servers.mm.command='<PYTHON>'" in place(argv) and "mcp_servers.mm.required=true" in place(argv)
        settings = [place(argv)[i + 1] for i, a in enumerate(argv) if a == "-c"]
        nowhere = settings.index("shell_environment_policy.set.MULTIMEM_HOME='<WORK>/no-mm-home'")
        assert settings[nowhere + 1] == "shell_environment_policy.set.CUDA_VISIBLE_DEVICES='-1'", "the shell's, after the others"


def test_the_mcp_config_of_every_pilot_claude_mm_exec_is_what_the_agent_writes_for_the_server_the_system_serves(opened, tmp_path):
    spec, paths = cell(tmp_path / "cell", "mm", opened)
    placed = json.loads(json.dumps(placer({**paths, "PYTHON": sys.executable})(claude.mcp_config(spec.server))))
    recorded = [e["mcp"] for e in pilot_execs() if e["agent"] == "claude" and e["dir"] != "files"]
    assert len(recorded) == 24 and all(placed == mcp for mcp in recorded)


def test_every_recorded_init_event_of_a_pilot_claude_mm_exec_is_the_session_the_served_server_asks_for(opened, tmp_path):
    spec, _ = cell(tmp_path / "cell", "mm", opened)
    mm = [e for e in pilot_execs() if e["agent"] == "claude" and e["dir"] != "files"]
    assert len(mm) == 24
    wrong = [e["qid"] for e in mm if set(e["init"]["tools"]) != set(claude.session_tools(spec))
             or claude.check_init({"type": "system", "subtype": "init", **e["init"]}, spec)]
    assert not wrong, wrong
    files = next(e for e in pilot_execs() if e["agent"] == "claude" and e["dir"] == "files")
    assert claude.check_init({"type": "system", "subtype": "init", **files["init"]}, spec), "the files session is not the server's"
    assert Counter(e["dir"] for e in mm) == {"mm-d1-3B": 24}


# -- ported: what reads the installed skill ---------------------------------------------------------------------------------------

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
    spec, _ = cell(tmp_path / "cell", "mm", opened)
    for name in notes:
        served = Served(spec.served.server, spec.served.base_line, (SKILL_TEXT, notes_text(name)), None, "mm", None, {})
        claude.prepare(dataclasses.replace(spec, served=served), command=["claude"])


def test_the_instructions_hold_the_skill_and_the_notes_of_the_model_and_no_notes_leave_the_skill_alone(week, store, opened, tmp_path):
    text = instructions_of(opened.serve(case_in(tmp_path)), "claude")
    assert SKILL_TEXT in text and notes_text("d1_3b.md") in text and notes_text("d1.md") not in text
    bare = served_with(dataclasses.replace(D3, notes=""), week, store, tmp_path / "bare")
    assert instructions_of(bare, "claude") == "\n\n".join([BASE, SKILL_TEXT]) and bare.parts == (SKILL_TEXT,)
    with_notes = served_with(D3, week, store, tmp_path / "d3")
    codex_text = instructions_of(with_notes, "codex")
    assert notes_text("d3.md") in codex_text and notes_text("d1.md") not in codex_text
    assert SKILL_TEXT in codex_text and codex_text.startswith(BASE + "\n\n")
    omni = served_with(decider.known("d1-Omni-2B")[1], week, store, tmp_path / "d1")
    assert notes_text("d1.md") in instructions_of(omni, "codex"), "d1's notes are those of d1-Omni-2B, which the skill points to"


def test_notes_are_a_plain_file_name_in_the_skills_models_folder():
    for name in ("../SKILL.md", "..\\SKILL.md", "..", ".", "a/b.md"):
        with pytest.raises(ValueError, match="plain file name"):
            notes_text(name)
    assert notes_text("d1_3b.md") != notes_text("d3.md") != notes_text("d1.md")
