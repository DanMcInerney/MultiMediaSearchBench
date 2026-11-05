"""New tests (G1's repair of F3, F4 and F9 for the mm system): what `config_hash` and `store_hash` cover, what the system hands the agent, and what
`describe` and `preview` need of a decision model. The core's half (what is written in systems.json and ingest.json, and what is refused) is
tests/test_runner.py and tests/test_cli_build.py with a stand-in system; here is the system that has to make the hashes mean something.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import date, datetime

import pytest

pytest.importorskip("multimem")

import fakes  # noqa: E402
from fakes import D13_MODEL, MM, Opened, context  # noqa: E402
from model_folders import model_folder  # noqa: E402

from mmmb import cli, scenarios  # noqa: E402
from mmmb.core import fs  # noqa: E402
from mmmb.core.build import INGESTED, write_dump  # noqa: E402
from mmmb.core.config import Config  # noqa: E402
from mmmb.core.contract import Case, Description  # noqa: E402
from mmmb.core.question import write_questions  # noqa: E402
from mmmb.core.runfolder import RunFolder  # noqa: E402
from mmmb.systems.mm import estimate, models, system  # noqa: E402
from mmmb.systems.mm.system import Mm  # noqa: E402

pytestmark = pytest.mark.xdist_group("mm-arm")


# -- what the hashes cover -----------------------------------------------------------------------------------------------------------

def described(mm: Mm = MM) -> Description:
    return mm.describe("d1-3B")  # d1-3B's facts and notes, the fake decider


def test_the_hashes_are_the_same_for_the_same_system_and_config_hash_covers_everything_a_result_depends_on_that_store_hash_does_not(tmp_path, monkeypatch):
    base = described()
    assert described() == base and base.version == system.version_of()
    skill = tmp_path / "SKILL.md"
    skill.write_text(system.SKILL.read_text(encoding="utf-8") + "\nA line the skill did not have.\n", encoding="utf-8")
    notes = tmp_path / "models"
    notes.mkdir()
    for name in ("d1_3b.md", "d1.md"):
        (notes / name).write_text((system.NOTES_DIR / name).read_text(encoding="utf-8") + "\nA line the notes did not have.\n", encoding="utf-8")
    about = D13_MODEL.about
    settings = {
        "the skill's text": lambda patch: patch.setattr(system, "SKILL", skill),
        "the model's notes": lambda patch: patch.setattr(system, "NOTES_DIR", notes),
        "the server's instructions": lambda patch: patch.setattr(system, "INSTRUCTIONS", system.INSTRUCTIONS + " More."),
        "the base line": lambda patch: patch.setattr(system, "BASE", system.BASE + " More."),
        "the tools offered": lambda patch: patch.setattr(system, "TOOLS", system.TOOLS[:-1]),
        "the clock note": lambda patch: patch.setattr(system, "CLOCK_NOTE", system.CLOCK_NOTE + " More."),
        "the lateness that earns it": lambda patch: patch.setattr(system, "CLOCK_LATE_S", 7200.0),
        "the share of the exec an ask may take": lambda patch: patch.setattr(system, "ASK_SHARE", 0.5),
        "the size that holds the GPU alone": lambda patch: patch.setattr(system, "HEAVY_S", 100.0),
    }
    for what, plant in settings.items():
        with monkeypatch.context() as patch:
            plant(patch)
            changed = described()
        assert changed.config_hash != base.config_hash, what
        assert changed.store_hash == base.store_hash and changed.version == base.version, f"{what} is no part of what a store holds"
    # the model's own settings: where it is, its server, its name and what it reads
    for what, model in {"the model's folder": dataclasses.replace(D13_MODEL, spec="python:elsewhere.py"),
                        "its server": dataclasses.replace(D13_MODEL, url="http://127.0.0.1:1"),
                        "its name": dataclasses.replace(D13_MODEL, name="d1-3B-b"),
                        "what it reads": dataclasses.replace(D13_MODEL, about=dataclasses.replace(about, reads=frozenset({"image"}))),
                        "its picture cap": dataclasses.replace(D13_MODEL, about=dataclasses.replace(about, max_px=640)),
                        "its notes file": dataclasses.replace(D13_MODEL, about=dataclasses.replace(about, notes="d1.md"))}.items():
        changed = Mm(models={"d1-3B": model}).describe("d1-3B")
        assert changed.config_hash != base.config_hash and changed.store_hash == base.store_hash, what
    # the version of Multimem is both: a store is of the version that ingested it
    with monkeypatch.context() as patch:
        patch.setattr(system, "version_of", lambda: "abc1234+dirty")
        moved = described()
    assert moved.version == "abc1234+dirty" and moved.config_hash != base.config_hash and moved.store_hash != base.store_hash
    with monkeypatch.context() as patch:
        patch.setattr(system, "INGEST_VERSION", system.INGEST_VERSION + 1)
        assert described().store_hash != base.store_hash, "and so is what ingest stores"
    assert system.sha_of("a") != system.sha_of("b") and estimate.HEAVY_S == system.HEAVY_S


def test_the_variants_of_one_store_share_its_hash_and_each_has_its_own_configuration():
    fake, d13 = MM.describe("fake"), MM.describe("d1-3B")
    assert fake.store_hash == d13.store_hash and fake.config_hash != d13.config_hash


# -- the folder the system hands the agent ----------------------------------------------------------------------------------------------

def test_serve_hands_the_agent_the_folder_look_writes_in_and_nothing_else_and_makes_it(week, store, tmp_path):
    _, dump, questions = week
    q = questions[0]
    work = tmp_path / "work"
    work.mkdir()
    case = Case(q.qid, datetime.fromisoformat(q.now).timestamp(), q.horizon, q.scope.sources, q.scope.since, q.scope.until, q.group, 3.0, 0, work, tmp_path / "run")
    with Opened(store, dump, tmp_path / "homes") as session:
        assert not (session.home / "looks").exists(), "a copy of a store has none: the daemon makes it at its first look"
        served = session.serve(case)
        assert served.opens == (session.home / "looks",) and (session.home / "looks").is_dir()
    assert MM.preview(store, dump, context(tmp_path / "other")).serve(case).opens == (tmp_path / "other" / "claude-fake" / "looks",)


# -- a model that cannot run now still has its facts ---------------------------------------------------------------------------------------

def test_describe_and_preview_need_the_facts_of_a_model_and_not_that_it_can_run_and_open_says_why_it_cannot(week, store, tmp_path, monkeypatch):
    _, dump, questions = week
    nowhere = Mm(models_dir=tmp_path / "no-models")  # d1-3B is not downloaded here
    facts = nowhere.describe("d1-3B")
    assert facts.needs_gpu and facts.reads == frozenset({"text", "records", "image", "video"}) and facts.store_hash == described().store_hash
    homes = tmp_path / "homes"
    preview = nowhere.preview(store, dump, context(homes, variant="d1-3B"))
    q = questions[0]
    assert preview.max_jobs([], 3) == 3 and preview.estimate(q.scope.sources, q.cost_ranges) >= 0.0
    with pytest.raises(ValueError, match="d1-3B: not downloaded: mm model use d1-3B"):
        nowhere.open(store, dump, context(homes, variant="d1-3B"))
    assert not homes.exists(), "nothing was copied or started for a model that cannot run"
    # downloaded, and this environment lacks what a model that mm runs itself needs: the way to get it is this project's
    models_dir = tmp_path / "models"
    model_folder(models_dir, "d1-3B")
    monkeypatch.setattr(models.modelenv, "installed", lambda: {})
    here = Mm(models_dir=models_dir)
    assert here.describe("d1-3B").needs_gpu
    with pytest.raises(ValueError) as lacking:
        here.open(store, dump, context(homes, variant="d1-3B"))
    message = str(lacking.value)
    assert message.startswith("d1-3B: its env is not ready: torch is not installed") and message.endswith("run `uv sync --group mm-gpu`"), message
    assert "group d1" not in message and "mm's project" not in message and not homes.exists()


def test_a_dry_run_of_the_command_prints_the_exec_of_a_model_this_environment_cannot_run_and_starts_nothing(week, store, tmp_path, monkeypatch, capsys):
    """`mmmb run RUN --dry-run --configs mm/d1-3B` with no torch installed (G1's F9), over the real system and the tiny week's real store."""
    from mmmb.core import dryrun
    from mmmb.core.agents import claude
    from mmmb.systems.mm import home as mm_home
    from mmmb.systems.mm.daemon import Daemon

    recipe, dump, questions = week
    config = Config("mm/d1-3B", "mm", "d1-3B", False)
    folder, temp = tmp_path / "run", tmp_path / "temp"
    folder.mkdir()
    write_questions(folder / "questions.jsonl", questions)
    fs.write_json(folder / "data-manifest.json", fs.stats_of(dump.root))
    write_dump(folder, dump)
    shutil.copytree(store, folder / "stores" / "mm")
    fs.write_json(folder / "stores" / "mm" / INGESTED, {**dataclasses.asdict(fakes.REPORTS[str(store)]), "store_hash": MM.describe("fake").store_hash})
    RunFolder.write(folder, scenarios.get("v7_screen"), recipe, dump, temp, "t7", 900.0, [config], {"status": "skipped", "failures": []},
                    cache=tmp_path / "cache")
    monkeypatch.setattr(models.modelenv, "installed", lambda: {})  # no torch
    monkeypatch.setattr(dryrun, "shutil", type("NoPath", (), {"which": staticmethod(lambda name: None)}))
    monkeypatch.setattr(Daemon, "start", lambda self, ready=True: pytest.fail("a dry run starts no daemon"))
    monkeypatch.setattr(mm_home, "copy_home", lambda *args: pytest.fail("a dry run copies no store"))
    monkeypatch.setattr(claude, "check_login", lambda: pytest.fail("a dry run logs nothing in"))
    assert cli.main(["run", str(folder), "--dry-run", "--agents", "claude", "--configs", "mm/d1-3B", "--clock", "wall"]) == 0
    out = capsys.readouterr().out
    assert "== claude mm-d1-3B" in out and "--mcp-config" in out and out.rstrip().endswith("nothing was run")
    assert not (folder / "homes").exists() and not (folder / "results").exists()
    assert json.loads((folder / "dry-run" / "claude" / "mm-d1-3B" / questions[0].qid / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["mm"]["args"][0] == "-m"
