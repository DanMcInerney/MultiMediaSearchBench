"""New tests for mmmb/core/config.py. The expected results are the plan's section 4.3 table (written by hand) and the folder names
of the recorded runs (`mm-d1-3B`, `files`, `blind`)."""

from __future__ import annotations

import pytest

from mmmb.core.config import Config, parse_config

DEFAULTS = {"files": None, "listing": None, "mm": "d1-3B", "other": "v1.2"}  # what the registry reads from the system.toml files


# written, Config(name, system, variant, own_tools), results folder
TABLE = [
    ("blind", Config("blind", None, None, False), "blind"),
    ("files", Config("files", "files", None, True), "files"),
    ("listing", Config("listing", "listing", None, False), "listing"),
    ("mm", Config("mm/d1-3B", "mm", "d1-3B", False), "mm-d1-3B"),
    ("mm/d1-3B", Config("mm/d1-3B", "mm", "d1-3B", False), "mm-d1-3B"),
    ("mm+files", Config("mm/d1-3B+files", "mm", "d1-3B", True), "mm-d1-3B+files"),
    ("mm/d1-3B+files", Config("mm/d1-3B+files", "mm", "d1-3B", True), "mm-d1-3B+files"),
    ("listing+files", Config("listing+files", "listing", None, True), "listing+files"),
    ("mm/d1-Omni-2B", Config("mm/d1-Omni-2B", "mm", "d1-Omni-2B", False), "mm-d1-Omni-2B"),
    ("other", Config("other/v1.2", "other", "v1.2", False), "other-v1-2"),  # the variant's dot is not a file name's
    ("  mm/d1-3B  ", Config("mm/d1-3B", "mm", "d1-3B", False), "mm-d1-3B"),
]


@pytest.mark.parametrize("written, config, folder", TABLE, ids=[t[0] for t in TABLE])
def test_a_written_configuration_is_the_config_and_the_results_folder_of_the_plans_table(written, config, folder):
    got = parse_config(written, DEFAULTS)
    assert got == config and got.dir == folder


REFUSED = [
    ("gold", "gold is not built yet (DESIGN section 4; not P0)"),
    ("gold+files", "gold is not built yet"),
    ("gold/x", "gold is not built yet"),
    ("embed", "not a configuration: blind, or one of the systems files, listing, mm, other"),
    ("", "not a configuration"),
    ("+files", "not a configuration"),
    ("blind+files", "blind has no tools to add to"),
    ("blind/x", "blind has no tools to add to"),
    ("files+files", "files is the agent's own tools already"),
    ("files/x", "files is the agent's own tools already"),
    ("listing/x", "listing has no variants"),
    ("mm/", "names no variant after the slash"),
    ("mm/+files", "names no variant after the slash"),
    ("mm+files+files", "the only + is a final +files"),
    ("mm/d1-3B+files+files", "the only + is a final +files"),
    ("mm+d1-3B", "the only + is a final +files"),
]


@pytest.mark.parametrize("written, message", REFUSED, ids=[r[0] or "empty" for r in REFUSED])
def test_a_configuration_that_is_not_one_is_refused_and_gold_says_it_is_not_built(written, message):
    with pytest.raises(ValueError) as refused:
        parse_config(written, DEFAULTS)
    assert message in str(refused.value)


def test_a_name_parses_back_to_its_config_and_configs_with_plain_variants_get_folders_of_their_own():
    # (two variants that differ only in characters a file name cannot hold, as v1.2 and v1-2, would share a folder: the runner checks)
    configs = {parse_config(w, DEFAULTS) for w in ["blind", "files", "listing", "listing+files", "mm", "mm+files", "mm/x", "mm/x+files",
                                                   "mm/x-files", "mm/x-files+files", "other", "other+files", "other/v1_2", "mm/files"]}
    assert len(configs) == 14
    for config in configs:
        assert parse_config(config.name, DEFAULTS) == config, config
    assert len({c.dir for c in configs}) == len(configs), "two configs in one results folder would overwrite each other"
