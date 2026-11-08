"""New tests for mmsb/systems (the registry and the `files` system): the registry reads every system.toml as TOML and imports a system only
when it is asked for, so the core knows the isolation of mm with mm absent; a system whose extra is missing is refused with the message
the plan gives (4.2) and no fallback.

Expected values are plan 4.2's TOML (mm's isolation, copied by hand into tests/agent_spec.py by U3 before any registry existed), the
contract's `Description` for a system with nothing to read through, and synthetic systems made here."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from agent_spec import ISOLATION

from mmsb import systems
from mmsb.core.config import parse_config
from mmsb.core.contract import Description, Dump, PlainSession, SessionContext

ROOT = Path(__file__).parents[1]
GOOD = """\
[system]
name = "{name}"
class = "{cls}"
license = "none"
horizon = "cut"
"""


def make(folder: Path, name: str, text: str | None = None, cls: str = "nothing:Nothing") -> Path:
    (folder / name).mkdir(parents=True)
    (folder / name / "system.toml").write_text(text if text is not None else GOOD.format(name=name, cls=cls), encoding="utf-8")
    return folder


def test_the_registry_reads_the_toml_files_there_are_and_a_folder_with_none_is_not_a_system():
    found = systems.infos()
    assert set(found) >= {"files", "listing", "mm"}
    mm, files, listing = found["mm"], found["files"], found["listing"]
    assert (listing.cls, listing.extra, listing.default_variant, listing.horizon, listing.isolation.deny_words) == (
        "mmsb.systems.listing.system:Listing", None, None, "cut", ())  # a system of its own that keeps nothing on the machine: no isolation to declare
    assert (mm.cls, mm.extra, mm.group, mm.default_variant, mm.horizon) == ("mmsb.systems.mm.system:Mm", "mm", "mm-gpu", "d1-3B", "cut")
    assert (files.cls, files.extra, files.default_variant, files.horizon) == ("mmsb.systems.files.system:Files", None, None, "view")
    assert systems.defaults() == {"files": None, "mm": "d1-3B"} | {k: v for k, v in systems.defaults().items() if k not in ("files", "mm")}
    # a configuration parsed with them: `mm` is `mm/d1-3B`, and `files` has no variants
    assert parse_config("mm", systems.defaults()).name == "mm/d1-3B" and parse_config("mm+files", systems.defaults()).dir == "mm-d1-3B+files"
    with pytest.raises(ValueError, match="no variants"):
        parse_config("files/x", systems.defaults())


def test_mms_isolation_is_read_from_its_toml_and_is_what_the_agents_were_tested_with():
    assert systems.infos()["mm"].isolation == ISOLATION
    assert systems.isolation() == ISOLATION, "the union over the systems: files adds none"


def test_the_isolation_is_the_union_over_the_systems_and_two_that_disagree_on_a_folder_are_refused(tmp_path):
    one = GOOD.format(name="one", cls="x:X") + textwrap.dedent("""\
        [isolation]
        env_drop_prefixes = ["A_", "B_"]
        deny_words = [".one"]
        deny_package_roots = ["one"]
        [isolation.env_nowhere]
        ONE_HOME = "no-one"
        SHARED = "nowhere"
        """)
    two = GOOD.format(name="two", cls="x:X") + textwrap.dedent("""\
        [isolation]
        env_drop_prefixes = ["B_", "C_"]
        deny_words = [".two"]
        [isolation.env_nowhere]
        SHARED = "nowhere"
        """)
    make(tmp_path, "one", one)
    make(tmp_path, "two", two)
    got = systems.isolation(tmp_path)
    assert got.env_drop_prefixes == ("A_", "B_", "C_") and got.deny_words == (".one", ".two") and got.deny_package_roots == ("one",)
    assert dict(got.env_nowhere) == {"ONE_HOME": "no-one", "SHARED": "nowhere"}
    make(tmp_path, "three", GOOD.format(name="three", cls="x:X") + '[isolation.env_nowhere]\nSHARED = "elsewhere"\n')
    with pytest.raises(ValueError, match="SHARED leads to 'nowhere' in one system.toml and 'elsewhere' in three"):
        systems.isolation(tmp_path)


@pytest.mark.parametrize("text, message", [
    ("[system]\nname = 'bad'\nclass = 'x:X'\nlicense = 'n'\nhorizon = 'cut'\nextra_key = 1\n", "unknown key extra_key in .system."),
    ("[system]\nname = 'other'\nclass = 'x:X'\nlicense = 'n'\nhorizon = 'cut'\n", "name 'other' is not the folder's, 'bad'"),
    ("[system]\nname = 'bad'\nclass = 'x:X'\nlicense = 'n'\nhorizon = 'sideways'\n", "horizon is one of cut, view, not 'sideways'"),
    ("[system]\nname = 'bad'\nlicense = 'n'\nhorizon = 'cut'\n", "class is missing"),
    ("[other]\nx = 1\n", "system is missing"),
    ("[system]\nname = 'bad'\nclass = 'x:X'\nlicense = 'n'\nhorizon = 'cut'\n[isolation]\nwhy = []\n", "unknown key why in .isolation."),
    ("[system]\nname = 'bad'\nclass = 'x:X'\nlicense = 'n'\nhorizon = 'cut'\n[isolation]\ndeny_words = [1]\n", "lists of strings"),
    ("[system]\nname = 'bad'\nclass = ", "Invalid value"),
], ids=["unknown key", "other name", "bad horizon", "missing key", "missing table", "unknown isolation key", "not strings", "not toml"])
def test_a_system_toml_with_a_fault_is_refused_naming_the_file(tmp_path, text, message):
    make(tmp_path, "bad", text)
    with pytest.raises(ValueError, match=message) as error:
        systems.infos(tmp_path)
    assert "bad" in str(error.value) and "system.toml" in str(error.value)


def test_a_system_is_imported_only_when_it_is_asked_for_and_an_unknown_name_names_the_ones_there_are(tmp_path, monkeypatch):
    package = tmp_path / "pkg_for_test"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text("IMPORTED = []\nclass Nothing:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    make(tmp_path / "s", "fine", cls="pkg_for_test.mod:Nothing")
    assert "pkg_for_test.mod" not in sys.modules, "reading the toml imports nothing"
    assert systems.infos(tmp_path / "s")["fine"].cls == "pkg_for_test.mod:Nothing" and "pkg_for_test.mod" not in sys.modules
    assert type(systems.load("fine", tmp_path / "s")).__name__ == "Nothing" and "pkg_for_test.mod" in sys.modules
    with pytest.raises(ValueError, match="'nope' is not a system: fine"):
        systems.load("nope", tmp_path / "s")


def test_a_system_whose_extra_is_not_installed_is_refused_with_the_extra_to_install_and_any_other_import_error_is_not_that(tmp_path, monkeypatch):
    package = tmp_path / "pkg_extra_test"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "needs.py").write_text("import a_package_nobody_installed_xyz\nclass X:\n    pass\n", encoding="utf-8")
    (package / "own.py").write_text("from pkg_extra_test import missing_sibling\nclass X:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    folder = tmp_path / "s"
    make(folder, "needy", GOOD.format(name="needy", cls="pkg_extra_test.needs:X") + 'extra = "thing"\n')
    make(folder, "plain", GOOD.format(name="plain", cls="pkg_extra_test.needs:X"))
    make(folder, "mine", GOOD.format(name="mine", cls="pkg_extra_test.own:X") + 'extra = "thing"\n')
    with pytest.raises(systems.ExtraMissing) as error:
        systems.load("needy", folder)
    assert str(error.value) == "needy needs the thing extra: uv sync --extra thing" and isinstance(error.value, RuntimeError)
    with pytest.raises(ModuleNotFoundError, match="a_package_nobody_installed_xyz"):
        systems.load("plain", folder)  # it names no extra: the missing package is a fault, not an extra to install
    with pytest.raises(ImportError, match="missing_sibling"):
        systems.load("mine", folder)  # a missing part of the system's own package is a fault of the system, not of the environment


def test_the_mm_system_is_refused_with_the_extra_to_install_when_mms_is_absent():
    import importlib.util

    if importlib.util.find_spec("mms") is not None:
        pytest.skip("mms is installed here: the mm extra is not missing")
    if not (ROOT / "mmsb" / "systems" / "mm" / "system.py").exists():
        pytest.skip("mmsb/systems/mm/system.py is U7's and is not here yet, so importing mm fails for want of the module, not of the extra")
    with pytest.raises(systems.ExtraMissing) as error:
        systems.load("mm")
    assert str(error.value) == "mm needs the mm extra: uv sync --extra mm --group mm-gpu"


def test_the_files_system_has_nothing_to_ingest_or_serve_and_reads_every_modality(tmp_path):
    files = systems.load("files")
    d = files.describe(None)
    assert isinstance(d, Description) and (d.name, d.variant, d.needs_gpu, d.local, d.network, d.horizon, d.has_oracle) == (
        "files", None, False, True, False, "view", False)
    assert d.reads == frozenset({"image", "video", "audio", "text", "records"})
    with pytest.raises(ValueError, match="no variants"):
        files.describe("x")
    report = files.ingest(Dump("k", tmp_path, "s", (), ()), tmp_path / "store")
    assert (report.seconds, report.gpu_seconds, report.bytes_written, report.refused, report.misdated) == (0.0, 0.0, 0, (), ())
    assert not (tmp_path / "store").exists(), "it writes no store"
    session = files.open(tmp_path / "store", Dump("k", tmp_path, "s", (), ()), SessionContext("claude", None, 900.0, tmp_path, lambda: 0.0))
    assert isinstance(session, PlainSession)


def test_the_runner_and_the_registry_import_with_mms_and_the_old_root_modules_refused():
    """A process whose import system refuses the old root modules (which the editable install would let through unnoticed) and
    mms, the tests' helpers and the fake decider: the new code must not reach for any of them."""
    code = textwrap.dedent("""\
        import importlib, sys
        BLOCKED = {"bench", "claude_run", "codex_run", "egolife", "grade", "haystack", "haystack_media", "shortcuts",
                   "mms", "conftest", "day", "fake_decider"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked: {name}")

        sys.meta_path.insert(0, Blocker())
        for name in ("mmsb.systems", "mmsb.systems.files.system", "mmsb.core.gpu", "mmsb.core.record", "mmsb.core.schedule",
                     "mmsb.core.select", "mmsb.core.costing", "mmsb.core.runfolder", "mmsb.core.runner"):
            importlib.import_module(name)
        from mmsb import systems
        assert systems.isolation().deny_package_roots == ("mms",)
        assert type(systems.load("files")).__name__ == "Files"
        from mmsb.core.runner import real_agents
        assert set(real_agents()) == {"claude", "codex"}
        assert not BLOCKED & {m.split(".")[0] for m in sys.modules}, BLOCKED & set(sys.modules)
        print("ok")
        """)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=ROOT,
                          env={"PYTHONPATH": str(ROOT), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
                              "MMSB_HOME": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})  # (a home to put the folders in: there is none in this environment)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-1500:]


def test_no_file_of_the_runner_imports_an_old_root_module_or_mms_or_names_a_system():
    import ast

    old = {"bench", "claude_run", "codex_run", "egolife", "grade", "haystack", "haystack_media", "shortcuts", "mms"}
    core = ["runner", "record", "schedule", "select", "costing", "gpu"]
    for file in [*(ROOT / "mmsb" / "core" / f"{n}.py" for n in core), ROOT / "mmsb" / "systems" / "__init__.py",
                 ROOT / "mmsb" / "systems" / "files" / "system.py"]:
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not [n for n in names if n.split(".")[0] in old], (file.name, names)
