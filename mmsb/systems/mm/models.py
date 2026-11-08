"""The decision model behind a variant of mm (moved from `Model`, `serving`, `needs_gpu` and `MODELS_DIR` of bench.py of
this repository at bb498a6; `resolve_models`, which listed the ready ones among several names, was not moved: `Mm.ready_model` asks for one): a variant is the name of a model Multimem knows (`mm model list`). A variant that is ready to run has its
folder of weights in MODELS_DIR, its env prepared and, for a served model, its server answering."""

from __future__ import annotations

import http.client
import os
from dataclasses import dataclass
from pathlib import Path

from multimem import config, decider, fetch, modelenv
from multimem.decider import About

from mmsb.core import fs


MODELS_DIR = Path(os.environ.get("MULTIMEM_MODELS", "D:/models"))
MM_GPU = "uv sync --group mm-gpu"  # what this environment needs to run a model that mm runs itself (torch, transformers)


@dataclass(frozen=True)
class Model:
    name: str
    spec: str  # the `[decider] model`: a model folder, or python:<file>
    about: About
    url: str | None = None  # where a served model (H2O's) is served

    @property
    def slug(self) -> str:
        return fs.slug(self.name)


def serving(url: str) -> bool:
    """Whether a model server answers 200 at /health."""
    try:
        connection = http.client.HTTPConnection(*config.loopback(url), timeout=5)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status == 200
        finally:
            connection.close()
    except (OSError, http.client.HTTPException):
        return False


def known_model(name: str, models_dir: Path) -> Model:
    """The facts of the model `name` (what its description and an estimate need), whether or not it can run now (`not_ready`). A
    ValueError for a name mm does not know."""
    found = decider.known(name)
    if found is None:
        raise ValueError(f"{name}: not a model mm knows ({', '.join(a.name for a in decider.known_abouts())})")
    backend, about = found
    return Model(about.name, (Path(models_dir).expanduser().resolve() / about.name).as_posix(), about,
                 backend.default_url if backend.served else None)


def not_ready(model: Model) -> str | None:
    """Why `model` (from `known_model`) cannot run now, else None: its weights are not downloaded, its env is not installed in this
    environment (torch, for the models mm runs itself), its server is not answering."""
    backend, about = decider.known(model.name)
    folder = Path(model.spec)
    claimed = decider.claim(folder)
    if claimed is None or claimed[1] != about:
        return f"{about.name}: not downloaded: mm model use {about.name}"
    if backend.served:
        return None if serving(backend.default_url) else (
            f"{about.name}: not serving at {backend.default_url}: start serve.sh in WSL (DESIGN.md, 13.3)")
    if problem := modelenv.problem(backend, folder, None):
        # Multimem says to run `backend.install` in mm's project: here the model runs in this environment, which has it as a group
        return f"{about.name}: its env is not ready: " + problem.replace(f"run `{backend.install}` in mm's project", f"run `{MM_GPU}`")
    if (why := fetch.unfinished(folder)) is not None:
        return f"{about.name}: its weights are not all there: {why}"
    return None


def needs_gpu(model: Model) -> bool:
    """Whether running `model` loads weights on this machine's GPU: not a test decider, not a server the owner runs."""
    return not model.spec.startswith("python:") and model.url is None
