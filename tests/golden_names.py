"""The one place where the recorded goldens and the code that is held to them are allowed to differ: by a name, or by a line of the tool's
skill that it changed on purpose.

The goldens under tests/oracles/recorded (the exec goldens `exec/*`, the pilot's `execs.json` and instruction texts) were recorded from
the search tool when it was called Multimem (package `multimem`, command `mm`, commit fe0ece5 and, for the pilot, before). It is now
MultimediaSearch (package and command `mms`), and the recordings stay byte for byte as they were: they are not re-recorded to make a
test pass. What the code builds now differs from them in two ways, each listed here and nowhere else:

- `RENAMES`: the names the tool changed. Every entry is a rename with its own reason.
- `SKILL_CHANGES`: the lines the tool's skill gained or changed after the goldens, in its commit facd716. Each entry is the exact old text and
  the exact new text, with one reason for the list.

A test that holds the code to a golden reads the golden through this module: `golden_text`, `golden_bytes` and `golden_json` give the
recording with `RENAMES` and then `SKILL_CHANGES` applied, in order, and nothing else changed. The rest of each golden is still compared
byte for byte, and a difference that is on neither list (a line the tool's text gained since, say) still fails the test and says what
differs. tests/test_golden_names.py holds both lists to their use: every entry matches a golden, and the served text equals the mapped one.

What does not change, and so is not here, is the benchmark's own choice: its system id `mm` (the `mm/...` configurations and `mm-d1-3B`
folders), the MCP server's name `mm` in the agents' mcp.json (so the agents still see `mcp__mm__ask`, as every recorded init event and
replayed stream has them), the words "the mm tools" of the base line, and the folder name `no-mm-home` that the shell's `MMS_HOME` points at.
"""

from __future__ import annotations

import json
from pathlib import Path

# (as recorded, as built now, why). Applied in this order; each must match some golden (tests/test_golden_names.py).
RENAMES: tuple[tuple[str, str, str], ...] = (
    ("multimem-questions", "mms-questions",
     "the skill is called mms-questions: its frontmatter `name:` and the first line of the model's notes, both in the instruction texts"),
    ("multimem.toml", "mms.toml", "the config file of a home is mms.toml: the skill says the owner lowers a cap by editing it"),
    ("multimem.cli", "mms.cli", "the module the MCP server is started with (`python -m mms.cli ... mcp`): the package and its command are mms"),
    ("MULTIMEM_", "MMS_",
     "the variables are MMS_*: MMS_HOME (set to a folder that is never made in a shell with the agent's own tools), MMS_MODELS and MMS_DEBUG "
     "(the owner's, dropped from the shell by their prefix)"),
    (".multimem", ".mms", "the home folder is ~/.mms: a deny word of the isolation, from system.toml"),
    ("Multimem", "MultimediaSearch",
     "the tool is called MultimediaSearch: the server's own first sentence, the skill's description and its heading"),
    ("`mm approve ", "`mms approve ", "the command is mms: the server's instructions and the skill tell the owner to run `mms approve <name>`"),
    ("`mm model use", "`mms model use", "the command is mms: the skill tells the owner to run `mms model use` to switch to a model that hears"),
    ("what mm sends it", "what mms sends it", "the tool is called mms in the notes of d1-3B: what mms sends the model"),
    ("(mm accepts ", "(mms accepts ", "the tool is called mms in the notes of d1-3B: how many answers mms accepts"),
)

# Why the lines below differ: the tool's skill gained them in its commit facd716 (MultimediaSearch, then Multimem; in the public repository
# it is the history before 66db891, 7 November 2026), after the goldens were recorded at fe0ece5. They are the tool's own words, not renames.
SKILL_CHANGES_WHY = "the tool's skill gained these in facd716 (MultimediaSearch, 7 Nov 2026)"
# (as recorded, as built now). Applied after RENAMES; each must match exactly once in each instruction text. An insertion is the line before
# it (or the line after) and the same line with the new lines put in.
SKILL_CHANGES: tuple[tuple[str, str], ...] = (
    # `look` with `region`, after the bullet on sheets and before the bullet on files the model does not read
    ("  overview, to narrow down.\n- **A file the model does not read**",
     "  overview, to narrow down.\n"
     "- **`region` zooms in.** `look` with `region` (`top`, `left`, `bottom`, `right`: fractions of the picture from its top\n"
     "  left) shows only that part, enlarged to the full inline size: a small object, a face, a corner of the screen. For a\n"
     "  box from `ask`, widen its edges a little: they are rounded, and what surrounds the thing helps you judge it. A sheet\n"
     "  with `region` shows that part of each frame, and only what moves there makes a new tile: a door, a desk, a doorstep.\n"
     "- **A file the model does not read**"),
    # `best_first`, between the bullet on the order of sightings and the one on a narrower range
    ("- Newest first by default. `oldest_first: true` finds the first time instead.\n- A narrower range and fewer `sources`",
     "- Newest first by default. `oldest_first: true` finds the first time instead.\n"
     "- **`best_first: true`** reads the whole range and gives the `limit` sightings with the highest `p`, best first, instead\n"
     "  of the first `limit` found. Use it to find every time, to count, or to ask whether something happened at all over a\n"
     "  long range, and over sound, where many windows can pass the default `hit`. It costs the whole range (up to `max_s`).\n"
     "- A narrower range and fewer `sources`"),
    # the `limit` line gains what `best_first` does with it
    ('- `limit` stops the scan at that many sightings. `limit: 1` is "last seen", and ends as soon as it is found.\n',
     '- `limit` stops the scan at that many sightings. `limit: 1` is "last seen", and ends as soon as it is found. With\n'
     "  `best_first` the scan reads on and keeps the best `limit`.\n"),
    # the `stopped` line says `limit` never ends a `best_first` scan
    ("`stopped` says why a scan ended: `limit`, `range_end`, `max_s` or `cancelled`.",
     "`stopped` says why a scan ended: `limit` (never with `best_first`), `range_end`, `max_s` or `cancelled`."),
    # a new rule 5, and the old rule 5 is 6
    ("5. A model gives the same answer to the same window and questions every time. Rewording moves easy questions by a few",
     "5. **A look-alike is not the thing.** When no sighting or candidate holds up under `look` (a coyote for a dog, a person\n"
     "   carrying a box past the door for a parcel left there), the answer is that it is not there, or that you could not\n"
     "   confirm it: say what you saw. Never give the look-alikes' times as the answer.\n"
     "6. A model gives the same answer to the same window and questions every time. Rewording moves easy questions by a few"),
)


def renamed(text: str) -> str:
    """`text` as recorded, with the renames applied."""
    for old, new, _ in RENAMES:
        text = text.replace(old, new)
    return text


def changed(text: str) -> str:
    """`text` with the skill's changes put in (after the renames): an old text that is not there once is left, so the comparison fails on it."""
    for old, new in SKILL_CHANGES:
        if text.count(old) == 1:
            text = text.replace(old, new)
    return text


def golden_text(path: Path) -> str:
    """A golden's text, with the renames and the skill's changes applied."""
    return changed(renamed(Path(path).read_bytes().decode("utf-8")))


def golden_bytes(path: Path) -> bytes:
    """A golden's bytes, with the renames and the skill's changes applied."""
    return golden_text(path).encode("utf-8")


def golden_json(path: Path):
    """A golden's JSON, with the renames and the skill's changes applied to its text (so to its keys and its values)."""
    return json.loads(golden_text(path))
