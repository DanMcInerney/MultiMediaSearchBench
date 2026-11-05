"""A run's folder, the neutral half of what the old `bench.Run` was: where its files and the agents' working slots are, what no tool
input of an agent with its own tools may hold, and the refusals when it is opened again. What is specific to a scenario
(prompts, views, extra deny words) is asked of `RunFolder.scenario`; what is specific to a system never enters.

    <folder>/run.json, questions.jsonl, data-manifest.json, report.json, report.md
             results/<agent>/<config.dir>/<qid>.json     execs/<agent>/<config.dir>/<qid>/
    <temp>/mmb-<key10>-data/ (the dump: named by its key and shared by the runs of one recipe, core/cache.py)
    <temp>/mmb-<tag>-bin/ (the ffmpeg on the agents' PATH)  mmb-<tag>-w<k>/ (working slot k): named by the run's own random `tag`

`run.json` holds seed, scale, days (ISO dates), data, temp, cache (where the dump's record and the stores are kept), tag, timeout_s,
scenario (its registered name), generator, recipe (what
the scenario was built from), dump_key, question_hash (the sha256 of questions.jsonl), configs (a list of `Config` fields), plan and files
(the scenario's `plan_hash` and `files_hash` of the plan its recipe makes, which `open` makes again and compares), probe, created,
mmmb_commit, build_s (the seconds the build took) and what the scenario adds (`Scenario.run_fields`). `write` makes it, last, so a folder
without it is a build that failed. `mmmb run` keeps the answer format it asked in and, when it ends, `agents`: each agent's model and
effort as its results say (`core/repeat.py`).

The agents with their own tools read what `Scenario.view` gives them: the dump itself (V7), or a folder cut for each question (V8). When
that is not the dump, the dump is hidden: it holds what no question may be answered from, so `deny` names it and `check_paths` looks at
the folder that is read instead."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from mmmb.core import fs, paths
from mmmb.core.config import Config
from mmmb.core.contract import Dump, Isolation
from mmmb.core.question import Question, load_questions
from mmmb.core.record import DONE_KINDS, kind_of
from mmmb.core.scenario import Scenario

JOBS_MAX = 4  # execs at once, so the working slots; more crashed the laptop once
ENDS = ('/', '"', " ", "'")  # what ends a folder's name in a path or a command
HIDDEN_DATA = tuple("-data" + after for after in ENDS)  # the end of a dump's folder name (`mmb-<tag>-data`) in a path or a command
IN_ONLY = "in:"  # a deny word that starts with this is checked in what the agent runs (a command, a path), not in what it reads
# What keeps an agent with its own tools off the GPU (`schedule.agent_off_gpu` says when): it sees no CUDA device, and may not ask for the
# GPU's decoder or for the GPU's list of users. Its shell gets the variable (Codex: its shell's; Claude: its process, so its MCP server too).
OFF_GPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1"}
OFF_GPU_DENY = (IN_ONLY + "-hwaccel", IN_ONLY + "nvidia-smi")


@dataclass(frozen=True)
class RunFolder:
    folder: Path
    seed: int
    scale: str
    days: tuple[date, ...]
    data: Path
    temp: Path
    tag: str  # an opaque name for the run's folders in the temp folder, apart from the dump's (the run's name says its seed and scale)
    timeout_s: float  # the seconds an exec may take
    scenario: Scenario
    configs: tuple[Config, ...]
    cache: Path | None = None  # where the dump's record and the stores are cached; None for a run recorded before the cache, which builds nothing

    @classmethod
    def open(cls, folder: Path, scenarios: Callable[[str], Scenario]) -> RunFolder:
        """The run in `folder`, from its run.json, with `scenarios` giving the scenario a name stands for. Refused when the
        scenario's generator has changed since, when questions.jsonl is not the file run.json recorded, or when the plan or the
        files that the recipe makes now are not those the run was built with (the scenario changed with no generator bump). A scenario
        that cannot make its plan (its source is missing) raises, and the run is not opened."""
        folder = Path(folder)
        meta = fs.read_json(folder / "run.json")
        if "scenario" not in meta:
            raise ValueError(f"{folder} was made before mmmb (its run.json names no scenario): mmmb reads only the runs it built")
        scenario = scenarios(meta["scenario"])
        if meta["generator"] != scenario.generator:
            raise ValueError(f"{folder} was made by generator {meta['generator']} of {scenario.name}, this is {scenario.generator}: "
                             "its questions are not the plan's now; build a new run")
        if meta["question_hash"] != fs.sha256_of(folder / "questions.jsonl"):
            raise ValueError(f"{folder}'s questions are not those its run.json recorded, so its results would be graded against "
                             "other questions; build a new run")
        plan, files = scenario.hashes(meta["recipe"])
        if meta["plan"] != plan:
            raise ValueError(f"{folder}'s plan is not the one its run.json recorded: {scenario.name} changed without a generator bump, so "
                             "its results would be graded against other questions; build a new run")
        if meta["files"] != files:
            raise ValueError(f"{folder}'s files are not those its run.json recorded: {scenario.name} changed without a generator bump, so "
                             "the data on disk is not the plan's; build a new run")
        return cls(folder, meta["seed"], meta["scale"], tuple(date.fromisoformat(d) for d in meta["days"]), Path(meta["data"]),
                   Path(meta["temp"]), meta["tag"], meta["timeout_s"], scenario, tuple(Config(**c) for c in meta["configs"]),
                   Path(meta["cache"]) if "cache" in meta else None)

    @classmethod
    def write(cls, folder: Path, scenario: Scenario, recipe: Mapping[str, object], dump: Dump, temp: Path, tag: str, timeout_s: float,
              configs: Sequence[Config], probe: dict, measured: Mapping[str, object] | None = None, *, cache: Path) -> RunFolder:
        """run.json written in `folder`, which holds questions.jsonl, and the run opened from it. `recipe` is what `dump` was built from
        (its seed, scale and days are the run's) and the plan and files hashes are those of the recipe. `measured` is what the systems
        measured while ingesting (`IngestReport.measured`) and the seconds the dump took to write (`build_s`); the scenario's own fields
        come from `Scenario.run_fields`. `cache` is where the dump's record and the stores are."""
        folder = Path(folder)
        plan, files = scenario.hashes(recipe)
        meta = {"scenario": scenario.name, "generator": scenario.generator, "recipe": recipe, "dump_key": dump.key, "seed": recipe["seed"],
                "scale": recipe["scale"], "days": list(recipe["days"]), "data": str(dump.root), "temp": str(temp), "cache": str(cache), "tag": tag,
                "timeout_s": timeout_s, "plan": plan, "files": files, "question_hash": fs.sha256_of(folder / "questions.jsonl"),
                "configs": [dataclasses.asdict(c) for c in configs], "probe": probe, "created": now_text(), "mmmb_commit": commit_of(),
                **scenario.run_fields(recipe), **(measured or {})}
        fs.write_json(folder / "run.json", meta)
        return cls.open(folder, lambda name: scenario)

    def rewritten(self, **fields: object) -> RunFolder:
        """run.json with `fields` set, and this run with the ones it holds as fields (`timeout_s`, and `configs` as `Config`s)."""
        fs.write_json(self.folder / "run.json", {**fs.read_json(self.folder / "run.json"), **fields})
        own = {k: v for k, v in fields.items() if k in ("timeout_s", "configs")}
        if "configs" in own:
            own["configs"] = tuple(Config(**c) for c in own["configs"])
        return dataclasses.replace(self, **own)

    def with_timeout(self, seconds: float) -> RunFolder:
        """The run with another time an exec may take, kept in its run.json (each result records the time it had)."""
        return self.rewritten(timeout_s=seconds)

    def with_configs(self, configs: Sequence[Config]) -> RunFolder:
        """The run with the `configs` it lacks added to its own, kept in its run.json (the report compares the configurations of the run)."""
        known = {c.name: c for c in self.configs}
        if clash := [c.name for c in configs if known.get(c.name, c) != c]:
            raise ValueError(f"{self.name} has the configurations {', '.join(clash)} as other configurations")
        merged = (*self.configs, *(c for c in dict.fromkeys(configs) if c.name not in known))
        return self if merged == self.configs else self.rewritten(configs=[dataclasses.asdict(c) for c in merged])

    @property
    def recipe(self) -> Mapping[str, object]:
        """What the dump was built from, as run.json recorded it."""
        return fs.read_json(self.folder / "run.json")["recipe"]

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def bin(self) -> Path:
        return self.temp / f"mmb-{self.tag}-bin"

    @property
    def bin_exe(self) -> Path:
        """The ffmpeg the agents with their own tools have first on their PATH (`ensure_bin` copies it)."""
        return self.bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    @property
    def manifest(self) -> Path:
        return self.folder / "data-manifest.json"

    def slot(self, k: int) -> Path:
        return self.temp / f"mmb-{self.tag}-w{k}"

    def questions(self) -> tuple[Question, ...]:
        return load_questions(self.folder / "questions.jsonl")

    def result_path(self, agent: str, config: Config, qid: str) -> Path:
        return self.folder / "results" / agent / config.dir / f"{qid}.json"

    def exec_dir(self, agent: str, config: Config, qid: str) -> Path:
        return self.folder / "execs" / agent / config.dir / qid

    def ensure_bin(self) -> None:
        """imageio-ffmpeg's binary copied to the folder that goes first on the agents' PATH, named `ffmpeg` (no ffprobe)."""
        import imageio_ffmpeg

        if not self.bin_exe.exists():
            self.bin.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), self.bin_exe)

    def check_paths(self, isolation: Isolation, reads: Path | None = None) -> None:
        """Raises when the temp folder holds anything but runs (`check_temp`), or when a folder of an agent with its own tools (the one
        it reads the recordings in, the ffmpeg, the first working slot) holds one of its deny words, else every command naming it would
        be refused. `reads` is the folder `Scenario.view` gives it; None: the dump itself."""
        from mmmb.core.agents.common import reached  # agents.common takes IN_ONLY from this module

        check_temp(self.temp)
        deny = self.deny(self.slot(0), isolation, reads)
        for folder in (reads or self.data, self.bin, self.slot(0)):
            if hit := reached(str(folder), deny):
                raise ValueError(f"{folder} holds {hit!r}, which an agent with its own tools refuses to reach: build with --temp elsewhere")

    def deny(self, own: Path | None, isolation: Isolation, reads: Path | None = None, off_gpu: bool = False) -> tuple[str, ...]:
        """What no tool input of an agent with its own tools may hold, lower case with forward slashes: this run's folder, the folders
        of other runs (the runs and cache folders, the results and temp folder of the runs recorded before MMMB, and the tests' folder
        unless this run is in it), the checkouts of this repository (a worktree's main checkout too), the owner's agent setup, the
        `isolation` of the systems, the scenario's words, the working slots other than `own` (all of them when None), and the dump
        when the agent reads another folder (`reads`, from `Scenario.view`; None: the dump itself), by absolute path, by name (a
        relative path) and by glob (an `IN_ONLY` word: a listing of the temp folder may name them, a command may not open them). An agent
        kept `off_gpu` also may not name the GPU's decoder or its list of users (`OFF_GPU_DENY`)."""
        others = [k for k in range(JOBS_MAX) if own is None or self.slot(k) != own]
        base = f"mmb-{self.tag}-w"
        slots = [IN_ONLY + word for k in others for word in (str(self.slot(k)), f"{base}{k}", f"*-w{k}", f"?-w{k}")]
        slots += [IN_ONLY + word for word in (f"{base}*", f"{base}?", f"{base}[", "mmb-*", "mmb-?", "mmb-[")]
        if reads is not None and reads != self.data:  # the dump is hidden: what the agent reads is cut from it
            slots += [IN_ONLY + word for word in (str(self.data), self.data.name, "*-data", "?-data", *HIDDEN_DATA)]  # (any run's dump)
        # a folder name is matched as a path part, before a slash, a quote or a space: a worktree named like it is not one
        cached = [self.cache] if self.cache is not None else []  # (the cache this run was built with, when it is not the default)
        elsewhere = [paths.RUNS, paths.CACHE, *cached, *paths.EARLIER]
        tests = not any(folder.is_relative_to(paths.TESTS) for folder in (self.folder, self.temp))  # the tests' own runs are in it
        named = [f"/{folder.name}{after}" for folder in (*elsewhere, *([paths.TESTS] if tests else [])) for after in ENDS]
        checkouts = [str(c) for c in paths.checkouts()]
        roots = [str(root) for name in isolation.deny_package_roots if (root := package_root(name)) is not None]
        words = (str(self.folder), str(paths.RUNS), str(paths.CACHE), *map(str, cached), checkouts[0], ".claude", ".codex", *named, *checkouts[1:],
                 *([str(paths.TESTS)] if tests else []), *isolation.deny_words, *roots,
                 *self.scenario.deny_words(others, self.temp, self.tag), *slots, *(OFF_GPU_DENY if off_gpu else ()))
        return tuple(dict.fromkeys(re.sub(r"[\\/]+", "/", w.lower()) for w in words))


def check_temp(temp: Path) -> None:
    """Raises when the temp folder holds anything but the folders of runs (`mmb-*`). An agent with its own tools can list the folder
    it works in the neighbourhood of, so a temp folder that also holds other things (the tests' runs, whose questions hold answers)
    would lead it to them, and no list of words can name what it will hold tomorrow."""
    if temp.is_dir() and (stray := next((p for p in sorted(temp.iterdir()) if not p.name.startswith("mmb-")), None)) is not None:
        raise ValueError(f"{temp} holds {stray.name!r}, which is not a folder of a run (mmb-*): the temp folder is for runs only, as an agent "
                         f"with its own tools can list it and read what is there. Move {stray.name!r} out, or use --temp with a folder "
                         f"that nothing else uses (the default is {paths.TEMP})")


def package_root(name: str) -> Path | None:
    """The repository of the installed top-level package `name`, found without importing it: the nearest folder above the package
    that holds a pyproject.toml or .git, else the package's own folder when it sits in site-packages; None when it is not
    installed."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    package = Path(next(iter(spec.submodule_search_locations)))
    for parent in package.parents:
        if parent.name == "site-packages":
            break
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return package


def commit_of() -> str:
    """The repo's short commit, `+dirty` when tracked files differ from it; the installed version outside a checkout (an installed
    copy has no git)."""
    if not (paths.ROOT / ".git").exists():
        try:
            return importlib.metadata.version("mmmb")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"{paths.ROOT} is not a git checkout and mmmb is not installed, so what version made a run cannot be told: "
                               "install it (uv sync)") from None

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_result(path: Path) -> dict | None:
    return fs.read_json(path) if path.exists() else None


def is_done(result: dict | None) -> bool:
    """A result that is not asked again: it finished, or it escaped, timed out or was capped (a wrong answer). A `setup`,
    `limit`, `auth` or `error` problem did not test the question: the next command asks it again."""
    return result is not None and (result["problem"] is None or kind_of(result["problem"]) in DONE_KINDS)
