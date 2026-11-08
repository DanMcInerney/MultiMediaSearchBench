"""Setup every test shares.

- Throwaway homes: for the whole session HOME, USERPROFILE, CODEX_HOME, CLAUDE_CONFIG_DIR and MMS_HOME lead into a
  temporary folder, so code that finds the owner's agent setup through the environment finds an empty one.
- The guard: the real `~/.claude`, `~/.codex`, `~/.mms` (and wherever the three variables pointed when pytest
  started) may be read but not changed, and so may the folders MMSB keeps its runs, dumps and cache in (`paths.RUNS`,
  `paths.TEMP`, `paths.CACHE`: a test gives its commands a --out, --temp and --cache under its own tmp_path). An audit hook
  catches the change where it happens, and the test that caused it fails even when the code under test swallowed the error.
  A snapshot of those folders would not do: a live Claude Code or Codex session or the mms daemon changes them while the
  suite runs. A process a test starts is not watched: it gets the throwaway environment instead.
- The same hook refuses to start a real `claude` or `codex` (any program of that name, or any program inside the owner's three
  folders), which spends the plan or reads the owner's login. A test that needs an agent puts a stand-in in its place (a program of
  another name, as tests/stand_in_agent.py does), and patches `claude.check_login`; two test runs once reached the real
  `claude auth status` before a fixture had patched it.
- Markers `slow`, `paid` and `gpu`, which pyproject's addopts leave out of a default run.
- The mm tests (tests/mm) skip themselves when the mm extra is not installed. The summary says so in a banner, and
  `--require-mm` turns the skip into a failure.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

from mmsb.core import paths

HOME_FOLDERS = (".claude", ".codex", ".mms")
HOME_VARS = {"CLAUDE_CONFIG_DIR": ".claude", "CODEX_HOME": ".codex", "MMS_HOME": ".mms"}  # variable -> folder it overrides
MM_MISSING = "could not import 'mms'"  # the start of the reason `pytest.importorskip("mms")` gives
BANNER = "the mm extra is not installed; run uv sync --extra mm"
AGENTS = frozenset({"claude", "codex"})  # programs a test must not start: the owner's agents
SHELLS = frozenset({"cmd", "sh", "bash", "zsh", "powershell", "pwsh"})  # a program that runs the command it is given: the program is that command's
SHELL_COMMAND = re.compile(r"(?:^|\s)(?:/c|-c|-command)\s+(.*)", re.I | re.S)
WRITING = frozenset("wax+")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
# audit event -> positions among its arguments of the paths it changes
CHANGES = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,), "os.rename": (0, 1),
           "os.link": (1,), "os.symlink": (1,), "shutil.copyfile": (1,), "shutil.copytree": (1,),
           "shutil.move": (0, 1), "shutil.rmtree": (0,)}


def normal(path) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def real_homes() -> tuple[str, ...]:
    """The owner's three folders, as the environment says at start, before the session fixture changes it."""
    folders = [Path.home() / name for name in HOME_FOLDERS]
    folders += [Path(os.environ[var]) for var in HOME_VARS if os.environ.get(var)]
    return tuple(dict.fromkeys(normal(f) for f in folders))


REAL_HOMES = real_homes()
MMSB_FOLDERS = tuple(normal(folder) for folder in (paths.RUNS, paths.TEMP, paths.CACHE))  # where a real run, dump and cache are kept
REFUSED: list[str] = []  # what the guard refused, in order, for the autouse fixture to tell


class Refused(Exception):
    """Not an OSError: code that skips what it cannot write or start must not skip this."""


def within(path, folders: tuple[str, ...]) -> bool:
    here = normal(path)
    return any(here == folder or here.startswith(folder + os.sep) for folder in folders)


def guarded(path) -> str | None:
    """What a test must not change `path` of: the owner's agent setup, or the folders MMSB's real runs, dumps and cache are in; else None."""
    if within(path, REAL_HOMES):
        return "the owner's agent setup"
    if within(path, MMSB_FOLDERS):
        return "the folders MMSB keeps its runs, dumps and cache in (give the command a --out, --temp and --cache of the test's own)"
    return None


def first_word(text: str) -> str | None:
    """The program of a command line: its first word, or the quoted path it starts with (which may hold spaces)."""
    if not (text := text.strip()):
        return None
    if text[0] in "\"'":
        end = text.find(text[0], 1)
        return text[1:end] if end > 0 else text[1:]
    return text.split(None, 1)[0]


def text_of(value) -> str | None:
    """A path or command as text: a string, bytes or a path."""
    return os.fsdecode(value) if isinstance(value, (str, bytes, os.PathLike)) else None


def program_of(event: str, args: tuple) -> str | None:
    """The program a process-starting audit event runs, else None; that of the command a shell is given to run (`cmd /c`, `sh -c`).
    `subprocess.Popen` gives (executable, args, cwd, env): the executable when it is named, else the first of `args`, or the first word of
    it when it is one string; `os.system` a command line; `os.exec*`, `os.spawn*` and `os.posix_spawn` a path."""
    line = None  # the command line, when there is one to look into
    if event == "subprocess.Popen":
        named, command = args[0], args[1]
        line = " ".join(map(str, command)) if isinstance(command, (list, tuple)) else text_of(command)
        program = text_of(named) if named is not None else (text_of(command[0]) if isinstance(command, (list, tuple)) and command else first_word(line or ""))
    elif event == "os.system":
        line = text_of(args[0])
        program = first_word(line or "")
    elif event in ("os.exec", "os.posix_spawn", "os.spawn"):
        program = text_of(args[1] if event == "os.spawn" else args[0])
    else:
        return None
    if program and Path(first_word(program) or "").stem.lower() in SHELLS and line and (inner := SHELL_COMMAND.search(line)):
        return first_word(inner[1].strip().removeprefix('"') if inner[1].lstrip().startswith('""') else inner[1]) or program
    return program or None


def started(program: str) -> str | None:
    """What a test must not start `program` of: the owner's agents, or anything in the owner's three folders; else None."""
    if Path(program).stem.lower() in AGENTS:
        return "the owner's agent (it spends the plan and reads the login): use a stand-in of another name"
    if within(program, REAL_HOMES):
        return "a program in the owner's agent setup"
    return None


def audit(event: str, args: tuple) -> None:
    if (program := program_of(event, args)) is not None and (what := started(program)):
        REFUSED.append(f"{event} {program}")
        raise Refused(f"a test must not start {what}: {event} {program}")
    if event == "open":
        path, mode, flags = args
        changed = (path,) if (WRITING & set(mode) if isinstance(mode, str) else flags & WRITE_FLAGS) else ()
    elif event in CHANGES:
        changed = tuple(args[i] for i in CHANGES[event])
    else:
        return
    for path in changed:
        if isinstance(path, (str, bytes, os.PathLike)) and (what := guarded(path)):
            REFUSED.append(f"{event} {os.fsdecode(path)}")
            raise Refused(f"a test must not change {what}: {event} {os.fsdecode(path)}")


sys.addaudithook(audit)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--require-mm", action="store_true",
                     help="fail, instead of skipping, the tests in tests/mm when the mm extra is not installed")


def pytest_configure(config: pytest.Config) -> None:
    for line in ("slow: seconds to minutes (a byte-exact build, a fresh environment); run with -m slow",
                 "paid: starts a real claude or codex agent and spends the plan's quota; run with -m paid",
                 "gpu: loads a model on the GPU, whose driver here is fragile; run with -m gpu"):
        config.addinivalue_line("markers", line)
    config.pluginmanager.register(MmSkips(config), "mm-skips")


@pytest.fixture(scope="session", autouse=True)
def throwaway_homes(tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("home")
    with pytest.MonkeyPatch.context() as patch:
        for var in ("HOME", "USERPROFILE"):
            patch.setenv(var, str(home))
        for var, folder in HOME_VARS.items():
            patch.setenv(var, str(home / folder))
        yield home


@pytest.fixture(autouse=True)
def real_homes_untouched():
    before = len(REFUSED)
    yield
    if refused := REFUSED[before:]:
        pytest.fail("the test did what it must not, a change to the owner's folders or the start of the owner's agents (refused, but the test must not "
                    "try): " + "; ".join(refused))


class MmSkips:
    """The tests under a folder named `mm` that were skipped because mms is not installed. A module that skips itself
    at import is one skip in pytest's report: its tests are counted from its source (a parametrized test counts once)."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.counts: dict[str, int] = {}  # node id -> tests skipped

    def note(self, report) -> None:
        if not report.skipped or "mm" not in Path(report.nodeid.split("::")[0]).parts[:-1]:
            return
        if not str(report.longrepr[2]).removeprefix("Skipped: ").startswith(MM_MISSING):
            return
        file = self.config.rootpath / report.nodeid
        self.counts[report.nodeid] = self.tests_in(file) if file.is_file() else 1

    def pytest_collectreport(self, report) -> None:
        self.note(report)

    def pytest_runtest_logreport(self, report) -> None:
        if report.when in ("setup", "call"):
            self.note(report)

    @staticmethod
    def tests_in(file: Path) -> int:
        return sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
                   for n in ast.walk(ast.parse(file.read_text(encoding="utf-8"))))

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if not self.counts:
            return
        total = sum(self.counts.values())
        if self.config.getoption("--require-mm"):
            terminalreporter.write_sep("=", f"{total} mm tests NOT RUN, which --require-mm fails: {BANNER}", red=True, bold=True)
        else:
            terminalreporter.write_sep("=", f"{total} mm tests SKIPPED: {BANNER}", yellow=True, bold=True)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.counts and self.config.getoption("--require-mm") and not hasattr(self.config, "workerinput"):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
