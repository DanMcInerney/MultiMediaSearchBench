"""New tests (plan 7.6, oracle neutrality): the core, the scenarios and the systems but mm run with no MultimediaSearch. Three checks.

- A process whose import system refuses `mms`, the tests' helpers `conftest`, `day` and `fake_decider` (and the old root modules, which
  the editable install would let through unnoticed) imports every module under `mmsb.core`, `mmsb.scenarios` and each system but mm (found
  by walking the packages, so `mmsb.systems.listing` is in it once it exists), reads mm's system.toml, and sees opening `mm` give "needs
  the mm extra". Fast: every run.
- A scan of the source for an import of mms outside mmsb/systems/mm and tests/mm. The scan is of the syntax, not the text: a docstring
  that speaks of MultimediaSearch is not an import.
- A scan of mmsb/core and mmsb/scenarios for a system by name: an import of mmsb.systems.<name> (the registry, `mmsb.systems` itself, stays) and
  a string equal to the name of a system that has a system.toml (`config.system == "mm"`). The few strings that are a system's name and not the
  system are listed with their reason (EXEMPT); the scan finds each of them there and nowhere else.
- Slow (`-m slow`): a fresh environment made with `uv sync --frozen --no-default-groups`, in which tests/oracles/neutral_smoke.py builds the
  tiny week of V7, runs its probe, runs a `files` cell with a stand-in agent, writes a report and sees `mm` refused.

The first two fail when a module of the core imports mms, even in a function; the third when anything the command needs is not in the
dependencies the project lists."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from mmsb import systems
from mmsb.core import fs, paths

ROOT = Path(__file__).parents[1]
MM_PLACES = ("mmsb/systems/mm", "tests/mm")  # where an import of mms is the point
# the files outside them that import it, and why: a guarded import for the mm cells of a replay when the extra is installed, and a capture
# tool that records what MultimediaSearch's own skill and instructions held, run in MultimediaSearch's venv
ALLOWED = {"tests/replay.py": "the replay of an mm exec, behind `mm_installed()`", "tests/oracles/capture_exec.py": "a capture tool, run once"}

# (file, string) of mmsb/core and mmsb/scenarios that is the name of a system and not the system, and why. Words the recorded reports fix are here.
EXEMPT = {
    ("mmsb/core/agents/spec.py", "files"): "an exec with the agent's own tools alone is called the files arm in the checks' messages",
    ("mmsb/core/config.py", "files"): "the configuration `files` is the agent's own tools: part of the vocabulary of configurations (DESIGN 4)",
    ("mmsb/core/report.py", "files"): "the control the recorded report compares with is called files",
    ("mmsb/core/runfolder.py", "files"): "run.json's `files` is the hash of the scenario's files, not a system",
    ("mmsb/core/scenario.py", "files"): "the body of the dump key names the scenario's files",
    ("mmsb/scenarios/v8_egolife/report.py", "files"): "the control the recorded EgoLife report compares with is called files",
    ("mmsb/scenarios/v7_screen/media_common.py", "mm"): "PIL's text anchor (middle, middle)",
    ("mmsb/scenarios/v7_screen/media_screen.py", "mm"): "PIL's text anchor (middle, middle)",
    ("mmsb/scenarios/v7_screen/pictures.py", "mm"): "PIL's text anchor (middle, middle)",
}
PLACES = ("mmsb/core", "mmsb/scenarios")  # where no system is named

WALK = textwrap.dedent("""\
    import importlib, pkgutil, sys
    from pathlib import Path

    BLOCKED = {"mms", "conftest", "day", "fake_decider", "bench", "claude_run", "codex_run", "egolife", "grade", "haystack",
               "haystack_media", "shortcuts"}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, Blocker())
    import mmsb
    from mmsb import systems

    folder = Path(systems.__file__).parent
    packages = ["mmsb.core", "mmsb.scenarios", *(f"mmsb.systems.{p.name}" for p in sorted(folder.iterdir())
                                                  if (p / "__init__.py").exists() and p.name != "mm")]
    seen = ["mmsb", "mmsb.cli", "mmsb.systems"]
    for name in seen:
        importlib.import_module(name)
    for name in packages:
        package = importlib.import_module(name)
        seen.append(name)
        for info in pkgutil.walk_packages(package.__path__, name + "."):
            importlib.import_module(info.name)
            seen.append(info.name)
    assert not BLOCKED & {m.split(".")[0] for m in sys.modules}, BLOCKED & {m.split(".")[0] for m in sys.modules}

    mm = systems.infos()["mm"]
    assert (mm.extra, mm.default_variant) == ("mm", "d1-3B") and mm.isolation.deny_words == (".mms/", '.mms"', ".mms ", ".mms'")
    assert systems.isolation().deny_package_roots == ("mms",) and systems.isolation().env_nowhere == {"MMS_HOME": "no-mm-home"}
    try:
        systems.load("mm")
    except systems.ExtraMissing as error:
        assert str(error) == "mm needs the mm extra: uv sync --extra mm --group mm-gpu", str(error)
    else:
        raise AssertionError("mm opened with mms refused")
    assert type(systems.load("files")).__name__ == "Files"
    print("\\n".join(seen))
    """)


def modules_walked() -> list[str]:
    done = subprocess.run([sys.executable, "-I", "-c", WALK], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.split()


def test_every_module_of_the_core_the_scenarios_and_the_systems_but_mm_imports_with_mms_refused_and_mm_says_what_it_needs():
    seen = modules_walked()
    for name in ("mmsb.cli", "mmsb.core.runner", "mmsb.core.build", "mmsb.core.dryrun", "mmsb.core.agents.claude", "mmsb.core.agents.codex",
                 "mmsb.scenarios.v7_screen.week", "mmsb.scenarios.v7_screen.shortcuts", "mmsb.scenarios.v8_egolife.report",
                 "mmsb.systems.files.system"):
        assert name in seen, name
    assert not [m for m in seen if m.startswith("mmsb.systems.mm")], "mm is not walked: it is the one place mms is imported"
    systems_dir = ROOT / "mmsb" / "systems"
    for package in (p for p in systems_dir.iterdir() if (p / "__init__.py").exists() and p.name != "mm"):
        assert f"mmsb.systems.{package.name}" in seen, f"{package.name} is a system, so it is walked"
    assert len(seen) > 60


def mms_imports(root: Path, places=MM_PLACES, allowed=ALLOWED) -> list[str]:
    """The imports of mms in the python files under `root`'s mmsb and tests folders, except in `places` and `allowed`: `file:line`."""
    found = []
    for file in sorted(p for folder in ("mmsb", "tests") if (root / folder).is_dir() for p in (root / folder).rglob("*.py")):
        rel = file.relative_to(root).as_posix()
        if any(rel.startswith(place + "/") for place in places) or rel in allowed:
            continue
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"), filename=rel)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            found += [f"{rel}:{node.lineno}" for name in names if name.split(".")[0] == "mms"]
    return found


def test_nothing_imports_mms_outside_mm_and_its_tests():
    assert mms_imports(ROOT) == []
    everywhere = mms_imports(ROOT, allowed={})
    for rel in ALLOWED:
        assert any(found.startswith(rel + ":") for found in everywhere), f"{rel} no longer imports mms: take it off ALLOWED"


def test_the_scan_finds_an_import_in_a_function_a_from_import_and_a_dotted_one_and_not_a_word_in_a_docstring(tmp_path):
    (tmp_path / "mmsb" / "core").mkdir(parents=True)
    (tmp_path / "mmsb" / "systems" / "mm").mkdir(parents=True)
    (tmp_path / "tests" / "mm").mkdir(parents=True)
    (tmp_path / "mmsb" / "core" / "plain.py").write_text('"""import mms is how mm is reached; from mms import x"""\nx = "import mms"\n')
    (tmp_path / "mmsb" / "core" / "lazy.py").write_text("def f():\n    import mms\n")
    (tmp_path / "mmsb" / "core" / "from_.py").write_text("from mms.adapter import INSTRUCTIONS\n")
    (tmp_path / "mmsb" / "core" / "dotted.py").write_text("import mms.config as c\n")
    (tmp_path / "tests" / "test_x.py").write_text("from mms import decider\n")
    (tmp_path / "mmsb" / "systems" / "mm" / "system.py").write_text("import mms\n")
    (tmp_path / "tests" / "mm" / "test_y.py").write_text("import mms\n")
    assert mms_imports(tmp_path) == ["mmsb/core/dotted.py:1", "mmsb/core/from_.py:1", "mmsb/core/lazy.py:2", "tests/test_x.py:1"]
    assert mms_imports(tmp_path, allowed={"tests/test_x.py": "x"}) == ["mmsb/core/dotted.py:1", "mmsb/core/from_.py:1", "mmsb/core/lazy.py:2"]


def systems_named(root: Path, names, exempt=EXEMPT, places=PLACES) -> tuple[list[str], set]:
    """`file:line: what` for each import of a system (`mmsb.systems.<name>`, `from mmsb.systems import <name>`) and each string equal to the
    name of a system, or naming a module of one, in the python files under `places` of `root` (docstrings and comments are not code), except the
    (file, string) of `exempt`; and the exemptions it met."""
    found, met = [], set()
    for file in sorted(p for place in places if (root / place).is_dir() for p in (root / place).rglob("*.py")):
        rel = file.relative_to(root).as_posix()
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=rel)
        docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            what = None
            if isinstance(node, ast.Import):
                what = next((f"imports {a.name}" for a in node.names for n in names if a.name.split(".")[:3] == ["mmsb", "systems", n]), None)
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")
                what = (f"imports from {node.module}" if module[:2] == ["mmsb", "systems"] and len(module) > 2 and module[2] in names else
                        next((f"imports {a.name} from mmsb.systems" for a in node.names if module == ["mmsb", "systems"] and a.name in names), None))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value in names:
                    if (rel, node.value) in exempt:
                        met.add((rel, node.value))
                    else:
                        what = f"names the system {node.value!r}"
                elif any(node.value == f"mmsb.systems.{n}" or node.value.startswith(f"mmsb.systems.{n}.") for n in names):
                    what = f"names the module {node.value!r}"
            if what:
                found.append((rel, node.lineno, what))
    return [f"{rel}:{line}: {what}" for rel, line, what in sorted(found)], met


def test_the_core_and_the_scenarios_name_no_system_and_import_none_of_them():
    names = set(systems.infos())
    assert {"files", "mm"} <= names, "the systems there are, read from their system.toml"
    found, met = systems_named(ROOT, names)
    assert found == [], "a helper of the core that knows a system by name: P1's second system would be treated otherwise"
    assert met == set(EXEMPT), "an exemption nobody uses any more is taken off the list"


def test_the_scan_finds_a_system_by_import_and_by_name_in_any_function_and_not_in_a_docstring_a_comment_or_the_registry(tmp_path):
    core, scenario, other = tmp_path / "mmsb" / "core", tmp_path / "mmsb" / "scenarios" / "s", tmp_path / "mmsb" / "other"
    for folder in (core, scenario, other):
        folder.mkdir(parents=True)
    (core / "fine.py").write_text('"""alpha is a system; import mmsb.systems.alpha"""\n'
                                  "from mmsb import systems\nfrom mmsb.systems import load, isolation, defaults\n"
                                  "x = 'alphabet'  # alpha\nimport mmsb.systemsx\ndef f():\n    'beta'\n    return systems.load(name)\n")
    (core / "equal.py").write_text("def f(config):\n    return 1 if config.system == 'alpha' else 3\n")
    (core / "lazy.py").write_text("def f():\n    from mmsb.systems.beta.jobs import WAIT\n    return WAIT\n")
    (core / "dotted.py").write_text("import mmsb.systems.alpha.system as s\n")
    (core / "package.py").write_text("from mmsb.systems import alpha\n")
    (core / "by_string.py").write_text("import importlib\nm = importlib.import_module('mmsb.systems.beta.system')\nn = 'mmsb.systems.alpha'\n")
    (core / "twice.py").write_text("a = 'beta'\nb = ['beta']\n")
    (scenario / "deep.py").write_text("OK = {'alpha': 1}\nfrom mmsb.systems.beta import Beta\n")
    (other / "outside.py").write_text("import mmsb.systems.alpha\nx = 'alpha'\n")
    found, met = systems_named(tmp_path, {"alpha", "beta"}, exempt={})
    assert found == ["mmsb/core/by_string.py:2: names the module 'mmsb.systems.beta.system'", "mmsb/core/by_string.py:3: names the module 'mmsb.systems.alpha'",
                     "mmsb/core/dotted.py:1: imports mmsb.systems.alpha.system", "mmsb/core/equal.py:2: names the system 'alpha'",
                     "mmsb/core/lazy.py:2: imports from mmsb.systems.beta.jobs", "mmsb/core/package.py:1: imports alpha from mmsb.systems",
                     "mmsb/core/twice.py:1: names the system 'beta'", "mmsb/core/twice.py:2: names the system 'beta'",
                     "mmsb/scenarios/s/deep.py:1: names the system 'alpha'", "mmsb/scenarios/s/deep.py:2: imports from mmsb.systems.beta"]
    assert met == set()
    found, met = systems_named(tmp_path, {"alpha", "beta"}, exempt={("mmsb/core/equal.py", "alpha"): "a reason"})
    assert "mmsb/core/equal.py:2: names the system 'alpha'" not in found and met == {("mmsb/core/equal.py", "alpha")}


@pytest.mark.slow
def test_the_core_runs_in_a_fresh_environment_that_has_no_mms(tmp_path_factory):
    """`uv sync --frozen --no-default-groups` into a new folder under the temp folder (on the volume of uv's cache, so uv links and does not
    copy), then the smoke test in that environment's python. The environment is removed at the end."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH: a fresh environment cannot be made")
    paths.TESTS.mkdir(parents=True, exist_ok=True)  # (the temp folder of runs holds runs only, and this is the tests')
    environment = Path(tempfile.mkdtemp(prefix="neutral-env-", dir=paths.TESTS))
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment), "PYTHONDONTWRITEBYTECODE": "1"}
    env.setdefault("UV_CACHE_DIR", str(paths.TESTS / "uv-cache"))
    try:
        synced = subprocess.run([uv, "sync", "--frozen", "--no-default-groups"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
        assert synced.returncode == 0, synced.stderr[-2000:]
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        listed = subprocess.run([python, "-m", "pip", "list"], capture_output=True, text=True, env=env)
        installed = subprocess.run([python, "-I", "-c", "import importlib.util as u; print(u.find_spec('mms'), u.find_spec('pytest'))"],
                                   capture_output=True, text=True, env=env)
        assert installed.stdout.strip() == "None None", (installed.stdout, installed.stderr, listed.stdout)
        smoke = ROOT / "tests" / "oracles" / "neutral_smoke.py"
        work = tmp_path_factory.mktemp("neutral")
        done = subprocess.run([python, "-I", str(smoke), str(work)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
        assert done.returncode == 0, (done.stdout[-1500:], done.stderr[-2500:])
        *lines, last = done.stdout.strip().splitlines()
        summary = json.loads(lines[-1])
        assert last == "neutral smoke ok" and summary["mms"] is None and summary["probe"] == "passed"
        assert summary["results"] == summary["questions"] == 19 and summary["report_runs"] == 1
        assert summary["refused"].endswith("mm needs the mm extra: uv sync --extra mm --group mm-gpu")
        assert fs.read_json(next((work / "runs").iterdir()) / "run.json")["configs"] == [
            {"name": "files", "system": "files", "variant": None, "own_tools": True}]
    finally:
        fs.remove_tree(environment)
