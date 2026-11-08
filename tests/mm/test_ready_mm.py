"""`Mm.check_ready` (mmsb/systems/mm/system.py, the contract's `System.check_ready`): a run of `files,mm/d1-3B` on a machine without torch used to run
the `files` cells and then fail at `open`; the CLI now asks every configuration's system before any agent starts (tests/test_ready.py), and this
is mm's answer, with the real `not_ready` of Multimem over a models folder of a test's own and this environment, which has no torch (the mm-gpu
group is not installed). Nothing is started or loaded."""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("multimem")

from fakes import FAKE  # noqa: E402
from model_folders import model_folder  # noqa: E402

from mmsb.cli import Registry  # noqa: E402
from mmsb.core.config import Config  # noqa: E402
from mmsb.systems.mm.system import Mm  # noqa: E402

D1 = Config("mm/d1-3B", "mm", "d1-3B", False)


@pytest.fixture
def no_torch():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed here (the mm-gpu group): this test is of an environment without it")


def test_mm_is_not_ready_for_a_model_that_is_not_downloaded_or_whose_environment_lacks_torch_and_says_why(tmp_path, no_torch):
    weights = tmp_path / "models"
    model_folder(weights, "d1-3B")
    mm = Mm(models_dir=weights)
    with pytest.raises(ValueError, match=r"d1-3B: its env is not ready: torch is not installed.*uv sync --group mm-gpu"):
        mm.check_ready("d1-3B")
    with pytest.raises(ValueError, match="d1-3B: not downloaded: mm model use d1-3B"):
        Mm(models_dir=tmp_path / "empty").check_ready("d1-3B")
    with pytest.raises(ValueError, match="mm has variants"):
        mm.check_ready(None)
    with pytest.raises(ValueError, match="nonsense: not a model mm knows"):
        mm.check_ready("nonsense")
    # what `describe` and `preview` need is the model's facts, which they have whether or not it can run
    assert mm.describe("d1-3B").needs_gpu


def test_mm_is_ready_for_a_decider_it_was_given_and_the_run_command_names_the_configuration_that_is_not(tmp_path, no_torch):
    assert Mm(models={"fake": FAKE}, models_dir=tmp_path).check_ready("fake") is None
    weights = tmp_path / "models"
    model_folder(weights, "d1-3B")
    mm = Mm(models={"fake": FAKE}, models_dir=weights)
    Registry.check_ready([Config("mm/fake", "mm", "fake", False)], {"mm": mm})
    Registry.check_ready([Config("blind", None, None, False)], {})
    with pytest.raises(ValueError, match=r"mm/d1-3B cannot run now: d1-3B: its env is not ready: torch is not installed"):
        Registry.check_ready([Config("mm/fake", "mm", "fake", False), D1], {"mm": mm})
    with pytest.raises(ValueError, match=r"mm/d1-3B\+files cannot run now: d1-3B: its env is not ready"):
        Registry.check_ready([Config("mm/d1-3B+files", "mm", "d1-3B", True)], {"mm": mm})
