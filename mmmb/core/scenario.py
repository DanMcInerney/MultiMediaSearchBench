"""What a scenario gives the runner (DESIGN 5): a recipe, a dump with its questions, the prompt, the view of the data an agent reads
and the words it may not reach. The scenario knows its media and questions; it never knows a system."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mmmb.core.answer import AnswerFormat
from mmmb.core.contract import Dump, Served
from mmmb.core.question import Question

if TYPE_CHECKING:
    from mmmb.core.runfolder import RunFolder


@dataclass(frozen=True)
class Access:
    """How an agent reaches the recordings in one configuration; the prompt's sentences differ by it and by nothing else."""

    own_tools: bool
    data: Path | None  # the folder the agent reads the recordings in (`Scenario.view`)
    served: Served
    answer_format: AnswerFormat  # how the final message is asked to end (core/answer.py)


@dataclass(frozen=True)
class ProbeOutcome:
    status: str  # passed, failed, skipped; as run.json's `probe`
    failures: tuple[str, ...]


class Scenario(Protocol):
    name: str  # "v7_screen"
    code: str  # "V7": what Question.scenario holds
    generator: int  # bumped when the scenario makes something other than before

    def recipe(self, **options) -> Mapping[str, object]: ...
    def hashes(self, recipe: Mapping[str, object]) -> tuple[str, str]: ...  # (plan_hash, files_hash): what run.json records
    def run_fields(self, recipe: Mapping[str, object]) -> Mapping[str, object]: ...  # what the scenario adds to run.json (V8: `pools`); V7: {}
    def build(self, recipe: Mapping[str, object], out: Path, workers: int) -> tuple[Dump, tuple[Question, ...]]: ...
    def probe(self, recipe: Mapping[str, object], dump: Dump, out: Path, ffmpeg: Path, seed: int) -> ProbeOutcome: ...
    def prompt(self, question: Question, access: Access) -> str: ...
    def view(self, question: Question, slot: int, dump: Dump, temp: Path, key: str, build: bool) -> Path | None: ...  # key: the run's tag
    def deny_words(self, others: Sequence[int], temp: Path, key: str) -> tuple[str, ...]: ...
    def after_exec(self, question: Question, out: dict, dump: Dump, access: Access) -> dict: ...  # V8's seen_past_horizon; V7 returns {}
    def release(self, temp: Path, key: str) -> None: ...
    def report(self, run: RunFolder) -> tuple[dict, str]: ...  # V7: core's; V8: its own


def dump_key(scenario: str, recipe: Mapping[str, object], generator: int) -> str:
    """The sha256 of the scenario's name, its recipe and its generator as canonical JSON (sorted keys, no spaces): what names a dump.
    The same recipe made by another generator is another dump."""
    body = {"scenario": scenario, "recipe": recipe, "generator": generator}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def plan_hash(questions: Sequence[Mapping]) -> str:
    """The sha256 of a plan's questions (JSON, sorted keys): a run whose plan is not the one it was built with is refused."""
    return hashlib.sha256(json.dumps(list(questions), sort_keys=True).encode("utf-8")).hexdigest()


def files_hash(files: Sequence[Sequence], logs: Mapping) -> str:
    """The sha256 of a plan's files, each `[path, source, time, name, kind, spec]`, and log lines: a distractor that changed with no
    change to a question, and no generator bump, would otherwise go unseen. A run's `run.json` holds it as `files`."""
    body = {"files": [list(row) for row in files], "logs": logs}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
