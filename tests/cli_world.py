"""The world the tests of the command line run in (tests/test_cli.py, test_cli_build.py, test_cli_v8.py): a registry folder of systems with
a stand-in system `stub` beside the real `files` (and the real system.toml of the others), and a scenario `toy` that is V7's plan, prompts
and views over a dump of three small files, written in a second instead of a week's media in half a minute. Nothing here starts a process
or imports mms.

`toy` is a module with the names of a `Scenario`, as the real ones are; `use` puts it in the scenario registry for a test. Its probe
returns what a test sets (`PROBE`) and notes how it was called (`PROBED`), as the old tests' stand-in for `shortcuts.run` did."""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import types
from pathlib import Path

import imageio_ffmpeg
import pytest
from stub_system import Stub, StubSession, StubSystem

from mmsb import cli, scenarios, systems
from mmsb.core.contract import Description, Dump, IngestReport, RetrievalResult
from mmsb.core.runfolder import RunFolder
from mmsb.core.runner import Agent
from mmsb.core.scenario import ProbeOutcome, dump_key
from mmsb.scenarios import v7_screen as v7
from mmsb.scenarios.v7_screen.manifest import LOGS, entries_of
from mmsb.scenarios.v7_screen.questions import questions_of

DAYS = "2026-10-30,2026-10-31"
SEED = 7
SCALE = "tiny"
FILES = {"notes/a.md": "a note", "screen/s.mkv": "a screen", "logs/web.jsonl": "{}"}  # what the toy dump holds, in all
CONTENT = ["v1"]  # what the files and the entries' hashes are made from: a test changes it to make the dump another
CLEAN = IngestReport(seconds=0.0, gpu_seconds=0.0, bytes_written=0, refused=(), misdated=())
PROBED: list[dict] = []  # how the probe was called, in order
PROBE = {"outcome": ProbeOutcome("passed", ())}  # what it returns
CURRENT: list[StubSystem] = []  # the system `stub` is: the test puts its own here

# -- the scenario `toy`: V7's plan over a dump of three files ----------------------------------------------------------------------

name = "toy"
code = v7.code
generator = v7.generator
recipe = v7.recipe
hashes = v7.hashes
prompt = v7.prompt
view = v7.view
deny_words = v7.deny_words
after_exec = v7.after_exec
release = v7.release
report = v7.report


def run_fields(recipe) -> dict:
    return {}


def build(recipe, out: Path, workers: int = 3):
    """The V7 plan of the recipe with its entries as if measured (from `CONTENT`) and three real files in `out`."""
    plan = v7.plan_of(recipe)
    entries = entries_of(plan, lambda path: (len(CONTENT[0]), 0, hashlib.sha256((CONTENT[0] + path).encode()).hexdigest()))
    for rel, text in FILES.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text + CONTENT[0], encoding="utf-8")
    return Dump(dump_key(name, recipe, generator), out, name, entries, (LOGS,)), questions_of(plan, entries)


def probe(recipe, dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome:
    PROBED.append({"recipe": recipe, "dump": dump, "out": Path(out), "ffmpeg": ffmpeg, "seed": seed,
                   "stores": sorted(p.name for p in (Path(out) / "stores").iterdir()) if (Path(out) / "stores").exists() else [], "run_json": (Path(out) / "run.json").exists(),
                   "data": all((dump.root / rel).is_file() for rel in FILES)})
    return PROBE["outcome"]


class OracleSession(StubSession):
    """A stand-in session that can be asked for needles with no agent: it keeps the queries and finds nothing."""

    def __init__(self, system, context) -> None:
        super().__init__(system, context)
        self.queries: list = []

    def oracle(self, query):
        self.queries.append(query)
        return RetrievalResult(sightings=(), candidates=(), windows=3, seconds=0.5, wall_s=0.1, stopped=None, warnings=())


class OracleSystem(StubSystem):
    """`stub` as a system with a retrieval run (`Description.has_oracle`); its sessions are kept in `sessions`."""

    def describe(self, variant: str | None) -> Description:
        return dataclasses.replace(super().describe(variant), has_oracle=True)

    def open(self, store, dump, context):
        self.note("open", None, context.agent, context.variant, store)
        self.sessions = [*getattr(self, "sessions", []), OracleSession(self, context)]
        return self.sessions[-1]


# -- the registry -----------------------------------------------------------------------------------------------------------------

def current() -> StubSystem:
    """What `systems.load("stub")` gives: the system a test put in `CURRENT`."""
    return CURRENT[-1]


def registry(folder: Path) -> Path:
    """A folder of systems: the real ones' system.toml copied, and `stub`, a system with the variants all, narrow and media."""
    for toml in systems.HERE.glob("*/system.toml"):
        (folder / toml.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(toml, folder / toml.parent.name / "system.toml")
    (folder / "stub").mkdir()
    (folder / "stub" / "system.toml").write_text(
        '[system]\nname = "stub"\nclass = "cli_world:current"\nlicense = "none"\ndefault_variant = "all"\nhorizon = "cut"\n', encoding="utf-8")
    return folder


def use(patch, system: StubSystem | None = None) -> StubSystem:
    """For a test: `toy` is a scenario, `system` (a new StubSystem by default) is `stub`, and the probe has not been called."""
    system = system or StubSystem()
    CURRENT.append(system)
    patch.setitem(scenarios.REGISTRY, "toy", "cli_world")
    PROBED.clear()
    PROBE["outcome"] = ProbeOutcome("passed", ())
    return system



# -- fixtures and helpers ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    """`toy` is a scenario, `stub` a system (a new one, whose ingest reports nothing wrong), the ffmpeg `ensure_bin` copies a made-up file,
    and the folders of a run, of its dump and of the cache are under `tmp_path`. `main` is the command line over this world."""
    ffmpeg = tmp_path / "ffmpeg-source.exe"
    ffmpeg.write_bytes(b"x")
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(ffmpeg))
    CONTENT[:] = ["v1"]
    system = use(monkeypatch, StubSystem(report=CLEAN))
    w = types.SimpleNamespace(system=system, registry=registry(tmp_path / "systems"), out=tmp_path / "runs", temp=tmp_path / "temp",
                              cache=tmp_path / "cache", root=tmp_path, patch=monkeypatch)
    w.main = lambda *argv, agents=None: cli.main(list(argv), registry=w.registry, **({"agents": agents} if agents else {}))
    return w


def build_args(w, seed: int = SEED, configs: str = "files,stub", *extra: str) -> list[str]:
    return ["build", "--scenario", "toy", "--scale", SCALE, "--seed", str(seed), "--days", DAYS, "--out", str(w.out), "--temp", str(w.temp), "--cache", str(w.cache),
            "--configs", configs, *extra]


def built_run(w, seed: int = SEED, configs: str = "files,stub", *extra: str) -> RunFolder:
    """The run `mmsb build` makes in the world, opened."""
    assert w.main(*build_args(w, seed, configs, *extra)) == 0
    (folder,) = [p for p in w.out.iterdir() if p.name.endswith(f"-s{seed}")]
    return RunFolder.open(folder, scenarios.get)


def agents_of(stub: Stub):
    """The `agents` a command line is given for a test: the stand-in for both."""
    return lambda: {"claude": Agent(stub), "codex": Agent(stub, must_report_use=True)}
