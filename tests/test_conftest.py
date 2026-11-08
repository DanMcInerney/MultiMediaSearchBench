"""The suite's own setup (tests/conftest.py), run as pytest in a subprocess over a small project that holds a copy of it. What
each project plants is what the expected results are written from: three tests in a module that skips for the missing extra,
a write into a stand-in for the owner's folders."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")
# one inner pytest at a time per outer worker (and with -n 2 inside, at most three processes of its own)
pytestmark = pytest.mark.xdist_group("inner-pytest")

MM_SKIPS = """
    import sys
    import pytest

    sys.modules["multimem"] = None  # what a missing extra looks like to importorskip
    pytest.importorskip("multimem")


    def test_one(): pass
    def test_two(): pass


    class TestGroup:
        def test_three(self): pass
"""

GUARD = """
    import os
    from pathlib import Path

    OWNER = Path(os.environ["OWNER_HOME"])
    CODEX = Path(os.environ["OWNER_CODEX"])


    def test_a_write_to_the_owners_folder_is_refused():
        (OWNER / ".claude" / "settings.json").write_text("{}")


    def test_a_refusal_the_code_swallows_still_fails_the_test():
        try:
            (OWNER / ".codex" / "made").mkdir()
        except Exception:
            pass


    def test_a_rename_out_of_the_owners_folder_is_refused():
        os.rename(OWNER / ".multimem" / "keep.txt", OWNER / "elsewhere.txt")


    def test_where_codex_home_pointed_at_start_is_the_owners_too():
        (CODEX / "sessions.json").write_text("{}")


    def test_reading_the_owners_folder_is_allowed():
        assert (OWNER / ".claude" / "keep.txt").read_text() == "owner"


    def test_a_folder_that_only_starts_like_the_owners_is_not_it():
        (OWNER / ".claude-other").mkdir()


    def test_the_session_has_throwaway_homes():
        home = Path.home()
        assert home != OWNER and OWNER not in home.parents
        for var, folder in (("CLAUDE_CONFIG_DIR", ".claude"), ("CODEX_HOME", ".codex"), ("MULTIMEM_HOME", ".multimem")):
            assert Path(os.environ[var]) == home / folder
        (Path(os.environ["CLAUDE_CONFIG_DIR"])).mkdir()
"""


MMSB_GUARD = """
    from mmsb.core import paths


    def test_a_write_to_the_cache_is_refused():
        (paths.CACHE / "dumps").mkdir(parents=True)


    def test_a_write_to_the_temp_folder_is_refused():
        (paths.TEMP / "mmb-0123456789-data.part").write_text("x")


    def test_a_swallowed_write_to_the_runs_folder_still_fails_the_test():
        try:
            (paths.RUNS / "run").mkdir(parents=True)
        except Exception:
            pass


    def test_reading_them_is_allowed():
        assert isinstance(paths.CACHE.is_dir(), bool) and isinstance(list(paths.TEMP.glob("*")), list)
"""


PROGRAM_GUARD = """
    import os
    import subprocess
    import sys
    from pathlib import Path

    FOLDER = Path(os.environ["STAND_INS"])
    OWNER = Path(os.environ["OWNER_HOME"])
    EXT = ".cmd" if os.name == "nt" else ""


    def test_starting_a_program_named_claude_is_refused():
        subprocess.run([str(FOLDER / ("claude" + EXT))])


    def test_starting_a_program_named_codex_is_refused_by_a_command_string_too():
        subprocess.run(f'"{FOLDER / ("codex" + EXT)}" exec', shell=True)


    def test_os_system_is_watched_as_well():
        os.system(f'"{FOLDER / ("claude" + EXT)}" auth status')


    def test_a_refusal_the_code_swallows_still_fails_the_test():
        try:
            subprocess.run([str(FOLDER / ("claude" + EXT))])
        except Exception:
            pass


    def test_a_program_inside_the_owners_folders_is_refused_whatever_it_is_called():
        subprocess.run([str(OWNER / ".codex" / ("helper" + EXT))])


    def test_python_is_allowed():
        subprocess.run([sys.executable, "-c", "pass"], check=True)


    def test_a_stand_in_of_another_name_is_allowed():
        subprocess.run([str(FOLDER / ("stand_in" + EXT))], check=True)
"""


def project(root: Path, files: dict[str, str]) -> Path:
    root.mkdir()
    shutil.copy(CONFTEST, root / "conftest.py")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")  # a rootdir of its own, no addopts
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def pytest_in(root: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items()
            if k not in ("PYTEST_ADDOPTS", "PYTHONPATH", "HOME", "USERPROFILE", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "MULTIMEM_HOME")}
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--color=no", "-v", *args], cwd=root,
                          env={**base, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", **env}, capture_output=True, text=True,
                          encoding="utf-8", timeout=180)


def outcomes(done: subprocess.CompletedProcess) -> dict[str, set[str]]:
    """Each test's outcomes in the -v output: its call's, and an ERROR when its teardown failed."""
    found: dict[str, set[str]] = {}
    for m in re.finditer(r"^\S+::(?P<test>\S+) (?P<outcome>PASSED|FAILED|SKIPPED|ERROR)", done.stdout, re.M):
        found.setdefault(m["test"], set()).add(m["outcome"])
    return found


BANNER = "mm tests SKIPPED: the mm extra is not installed; run uv sync --extra mm"


@pytest.mark.parametrize("workers", [(), ("-n", "2")], ids=["serial", "xdist"])
def test_the_skipped_mm_tests_are_counted_in_a_banner(tmp_path, workers):
    root = project(tmp_path / "p", {"mm/test_x.py": MM_SKIPS, "test_core.py": "def test_ok(): pass\n"})
    done = pytest_in(root, *workers)
    assert done.returncode == 0, done.stdout
    assert f"3 {BANNER}" in done.stdout
    assert "test_ok" in done.stdout


def test_require_mm_turns_the_skip_into_a_failure(tmp_path):
    root = project(tmp_path / "p", {"mm/test_x.py": MM_SKIPS, "test_core.py": "def test_ok(): pass\n"})
    done = pytest_in(root, "--require-mm")
    assert done.returncode == 1, done.stdout
    assert "3 mm tests NOT RUN, which --require-mm fails: the mm extra is not installed" in done.stdout


def test_an_mm_test_that_is_here_or_that_skips_for_another_reason_makes_no_banner(tmp_path):
    here = "import pytest\npytest.importorskip('json')\n\ndef test_a(): pass\n"
    other = "import pytest\n\ndef test_b():\n    pytest.skip('no download')\n"
    root = project(tmp_path / "p", {"mm/test_here.py": here, "mm/test_other.py": other})
    done = pytest_in(root, "--require-mm")
    assert done.returncode == 0, done.stdout
    assert "mm tests" not in done.stdout
    assert outcomes(done) == {"test_a": {"PASSED"}, "test_b": {"SKIPPED"}}


def test_a_test_may_read_the_owners_folders_and_may_not_change_them(tmp_path):
    owner = tmp_path / "owner"
    codex = tmp_path / "owner-codex"
    for folder in (".claude", ".codex", ".multimem"):
        (owner / folder).mkdir(parents=True)
        (owner / folder / "keep.txt").write_text("owner", encoding="utf-8")
    codex.mkdir()
    root = project(tmp_path / "p", {"test_guard.py": GUARD})
    done = pytest_in(root, HOME=str(owner), USERPROFILE=str(owner), CODEX_HOME=str(codex), OWNER_HOME=str(owner), OWNER_CODEX=str(codex))
    found = outcomes(done)
    refused = {"test_a_write_to_the_owners_folder_is_refused", "test_a_refusal_the_code_swallows_still_fails_the_test",
               "test_a_rename_out_of_the_owners_folder_is_refused", "test_where_codex_home_pointed_at_start_is_the_owners_too"}
    allowed = {"test_reading_the_owners_folder_is_allowed", "test_a_folder_that_only_starts_like_the_owners_is_not_it",
               "test_the_session_has_throwaway_homes"}
    assert set(found) == refused | allowed, done.stdout
    # a refused test never passes clean: it fails, or (when the code swallowed the refusal) its teardown does
    assert all({"FAILED", "ERROR"} & found[t] for t in refused), done.stdout
    assert all(found[t] == {"PASSED"} for t in allowed), done.stdout
    # refused before anything changed
    assert not (owner / ".claude" / "settings.json").exists()
    assert not (owner / ".codex" / "made").exists()
    assert (owner / ".multimem" / "keep.txt").read_text(encoding="utf-8") == "owner" and not (owner / "elsewhere.txt").exists()
    assert not (codex / "sessions.json").exists()
    assert (owner / ".claude-other").is_dir()
    assert sorted(p.name for p in (owner / ".claude").iterdir()) == ["keep.txt"]


def test_a_test_may_read_the_folders_of_runs_dumps_and_cache_and_may_not_change_them(tmp_path):
    from mmsb.core import paths

    before = [folder.exists() for folder in (paths.CACHE, paths.TEMP, paths.RUNS)]
    done = pytest_in(project(tmp_path / "p", {"test_guard.py": MMSB_GUARD}))
    found = outcomes(done)
    refused = {"test_a_write_to_the_cache_is_refused", "test_a_write_to_the_temp_folder_is_refused",
               "test_a_swallowed_write_to_the_runs_folder_still_fails_the_test"}
    allowed = {"test_reading_them_is_allowed"}
    assert set(found) == refused | allowed, done.stdout
    assert all({"FAILED", "ERROR"} & found[t] for t in refused), done.stdout
    assert all(found[t] == {"PASSED"} for t in allowed), done.stdout
    for folder in (paths.CACHE, paths.TEMP, paths.RUNS):  # (each was refused by the guard, not by a folder that was not there)
        assert re.search(rf"must not.*: (os\.mkdir|open) {re.escape(str(folder))}\\", done.stdout), (folder, done.stdout)
    assert [folder.exists() for folder in (paths.CACHE, paths.TEMP, paths.RUNS)] == before, "refused before anything was made"


def program(path: Path) -> Path:
    """A program at `path` (a script) that leaves `<path>.started` beside it when it runs, so that a refused one shows as never having run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"{path}.started"
    if os.name == "nt":
        path.write_text(f'@echo off\r\necho x> "{marker}"\r\n', encoding="utf-8")
    else:
        path.write_text(f'#!/bin/sh\ntouch "{marker}"\n', encoding="utf-8")
        path.chmod(0o755)
    return path


def test_a_test_may_not_start_the_owners_agents_or_a_program_in_the_owners_folders_and_may_start_anything_else(tmp_path):
    """Two test runs once reached the real `claude auth status` before a fixture patched `check_login`. Programs named as the agents are stand-ins
    here that leave a marker when they run, so the test cannot start a real one even if the guard were gone."""
    ext = ".cmd" if os.name == "nt" else ""
    owner, stand_ins = tmp_path / "owner", tmp_path / "stand-ins"
    for name in ("claude", "codex", "stand_in"):
        program(stand_ins / f"{name}{ext}")
    helper = program(owner / ".codex" / f"helper{ext}")
    for folder in (".claude", ".multimem"):
        (owner / folder).mkdir(parents=True)
    done = pytest_in(project(tmp_path / "p", {"test_guard.py": PROGRAM_GUARD}), HOME=str(owner), USERPROFILE=str(owner), CODEX_HOME=str(owner / ".codex"),
                     OWNER_HOME=str(owner), STAND_INS=str(stand_ins))
    found = outcomes(done)
    refused = {"test_starting_a_program_named_claude_is_refused", "test_starting_a_program_named_codex_is_refused_by_a_command_string_too",
               "test_os_system_is_watched_as_well", "test_a_refusal_the_code_swallows_still_fails_the_test",
               "test_a_program_inside_the_owners_folders_is_refused_whatever_it_is_called"}
    allowed = {"test_python_is_allowed", "test_a_stand_in_of_another_name_is_allowed"}
    assert set(found) == refused | allowed, done.stdout
    assert all({"FAILED", "ERROR"} & found[t] for t in refused), done.stdout
    assert all(found[t] == {"PASSED"} for t in allowed), done.stdout
    assert "must not start the owner's agent (it spends the plan and reads the login)" in done.stdout
    assert "must not start a program in the owner's agent setup" in done.stdout
    # refused before the program ran; the allowed stand-in did run
    for path in (stand_ins / f"claude{ext}", stand_ins / f"codex{ext}", helper):
        assert not Path(f"{path}.started").exists(), path
    assert Path(f"{stand_ins / f'stand_in{ext}'}.started").exists()
