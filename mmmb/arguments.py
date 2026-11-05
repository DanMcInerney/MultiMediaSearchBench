"""The `mmmb` command line (moved out of cli.py, which holds the commands and the order of their steps): what each command and
option is called and its default. `main` (cli.py) parses with `parser_of`, and a command added here is a subparser and a case in `main`."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmmb import scenarios
from mmmb.core import paths
from mmmb.core.answer import DEFAULT_FORMAT, FORMATS
from mmmb.core.clock import CLOCKS, DEFAULT_CLOCK
from mmmb.core.clockcheck import CASES
from mmmb.core.runfolder import JOBS_MAX
from mmmb.core.runner import AGENTS, JOBS, TIMEOUT_S
from mmmb.core.select import SUBSETS

OPTIONS = ("scale", "seed", "days", "source", "per_age", "speech")  # what a scenario's recipe may be given; `build` passes those it takes
DEFAULT_SEED = 1


def csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmmb", description="MultiMediaMemoryBench: how well a memory system for agents answers "
                                                              "questions about recordings, against the agent's own tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write a scenario's recordings, its questions and the stores, and probe for shortcuts")
    build.add_argument("--scenario", required=True, choices=tuple(scenarios.REGISTRY))
    build.add_argument("--scale", help="the scenario's scale (v7_screen: tiny, pilot, ...)")
    build.add_argument("--seed", type=int, help=f"the seed (default {DEFAULT_SEED})")
    build.add_argument("--days", type=csv, help="the days, oldest first, as ISO dates separated by commas (default: the latest)")
    build.add_argument("--source", type=Path, help="the download a public scenario reads (v8_egolife)")
    build.add_argument("--per-age", dest="per_age", type=int, help="questions of each age (v8_egolife)")
    build.add_argument("--speech", type=int, help="questions that need sound (v8_egolife)")
    build.add_argument("--out", type=Path, default=paths.RUNS, help=f"where runs go (default {paths.RUNS})")
    build.add_argument("--temp", type=Path, default=paths.TEMP, help="where the dumps (each mmb-<key>-data, found again by the next build "
                       "of the same recipe in the same folder) and the working folders go (default "
                       f"{paths.TEMP}): a folder for runs only, which an agent with its own tools can list; on the download's volume for a "
                       "scenario that links it (v8_egolife)")
    build.add_argument("--cache", type=Path, default=paths.CACHE, help="where the dumps' records (questions, entries, probe) and the ingested "
                       f"stores are kept to be found again (default {paths.CACHE}): on the volume of --out and --temp, because a store is "
                       "copied into the run with hard links")
    build.add_argument("--configs", default="files", help="the run's configurations, comma separated; the systems among them are "
                        "ingested (default files)")
    build.add_argument("--workers", type=int, default=3, help="files written or read at once (default 3)")
    build.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds an exec may take (kept in the run)")
    build.add_argument("--no-probe", action="store_true", help="do not run the shortcut probe over the data (the run and its report say so)")
    ingest = sub.add_parser("ingest", help="ingest the dump into the stores of the systems of some configurations")
    ingest.add_argument("run", help="a run folder, or its name under the runs folder")
    ingest.add_argument("--configs", help="configurations, comma separated (default the run's)")
    run = sub.add_parser("run", help="the execs of Claude and Codex over the questions (the plans' quota)")
    run.add_argument("run", help="a run folder, or its name under the runs folder")
    run.add_argument("--agents", default=",".join(AGENTS))
    run.add_argument("--configs", help="configurations, comma separated: blind, files, <system>[/<variant>][+files] (default the run's)")
    run.add_argument("--jobs", type=int, default=JOBS, help=f"execs at once, at most {JOBS_MAX} (a system may allow fewer)")
    run.add_argument("--subset", choices=SUBSETS, default="all", help="pilot: the media questions and 3 controls (text and records)")
    run.add_argument("--controls", help="the one configuration also asked the text and records questions (default: every one that reads them)")
    run.add_argument("--only", default="", help="these qids of the subset only, comma separated (a smoke)")
    run.add_argument("--scenarios", default="", help="only the questions of these scenarios (as V7), comma separated")
    run.add_argument("--kinds", default="", help="only these kinds of question (find, last, when, count, absent), comma separated")
    run.add_argument("--modalities", default="", help="only the questions about these data (image, video, audio, text, records), comma separated")
    run.add_argument("--splits", default="", help="only these splits (dev, test), comma separated")
    run.add_argument("--tag", action="append", metavar="NAME=VALUE", help="only the questions with this tag (repeat for more)")
    run.add_argument("--rung", help="only the questions of this rung of the size ladder: --tag rung=RUNG")
    run.add_argument("--limit", type=int, help="at most this many questions: a seeded sample that keeps the mix of kinds, canaries included")
    run.add_argument("--discriminating", default="", help="only the questions that separated the configurations in these runs (comma "
                     "separated; the agents of this command), and two that did not as canaries")
    run.add_argument("--timeout", type=float, help="seconds an exec may take (kept in the run; default the run's)")
    run.add_argument("--claude-effort", help="Claude's effort (default MMMB_CLAUDE_EFFORT, else high)")
    run.add_argument("--codex-effort", help="Codex's effort (default MMMB_CODEX_EFFORT, else high; the pilot ran it at medium)")
    run.add_argument("--claude-model", help="Claude's model (default MMMB_CLAUDE_MODEL, else sonnet)")
    run.add_argument("--codex-model", help="Codex's model (default MMMB_CODEX_MODEL, else the pinned one)")
    run.add_argument("--answer-format", choices=FORMATS, help="how the agent ends its final message: json, an answer with the evidence it cites "
                     f"(graded beside the verdict), or line, `ANSWER: <answer>` as the recorded runs did (default the run's own, else {DEFAULT_FORMAT}; "
                     "a run with results keeps its format)")
    run.add_argument("--side-by-side", action="store_true", help="run the cells beside each other (off by default): the ones that need no GPU while "
                     "one that does holds it, at most 4 execs at once in all and 4 of each agent (--jobs still bounds one cell), longest expected "
                     "case first; an agent with its own tools sees no CUDA device")
    run.add_argument("--clock", choices=CLOCKS, default=DEFAULT_CLOCK, help="what an exec's time limit is measured on: working, its wall time less "
                     "the seconds its system says it waited in the queue (a wall cap of 3 times the limit stands; a system that cannot say is "
                     "measured on the wall and is refused more than one exec at once), or wall, its wall time throughout "
                     f"(default {DEFAULT_CLOCK}; each result says which)")
    run.add_argument("--anyway", action="store_true", help="run a run whose shortcut probe failed (it is refused otherwise)")
    run.add_argument("--dry-run", action="store_true", help="say what the run will cost, write the first exec's files per agent and configuration, "
                     "run nothing")
    retrieve = sub.add_parser("retrieve", help="each system asked for the needles with no agent (the GPU, alone)")
    retrieve.add_argument("run", help="a run folder, or its name under the runs folder")
    retrieve.add_argument("--configs", help="configurations, comma separated (default the run's)")
    retrieve.add_argument("--dry-run", action="store_true", help="say what would be asked and about how long it takes, ask nothing")
    reports = sub.add_parser("report", help="the paired comparison; written in the first run's folder")
    reports.add_argument("runs", nargs="+", help="run folders; several runs of one scenario are pooled")
    reports.add_argument("--control", default="files", help="the configuration the others are compared with (pooled runs; default files)")
    compare = sub.add_parser("compare", help="two runs of the same questions, cell by cell: the flips, the counts, D1's band, McNemar")
    compare.add_argument("run_a", help="the run to compare with, usually the recorded one")
    compare.add_argument("run_b", help="the run compared, usually the repeat")
    compare.add_argument("--across-settings", action="store_true",
                         help="compare a cell made by another model or effort than the other run's (it is refused otherwise)")
    repeats = sub.add_parser("repeat", help="a recorded run again: its recipe, agents at their models and efforts, configurations and questions")
    repeats.add_argument("recorded", help="the recorded run's folder, or its name under the runs folder")
    repeats.add_argument("--into", help="go on in this run, which an earlier repeat built, instead of building another")
    repeats.add_argument("--out", type=Path, default=paths.RUNS, help=f"where the new run goes (default {paths.RUNS})")
    repeats.add_argument("--temp", type=Path, default=paths.TEMP, help=f"where its dump and working folders go (default {paths.TEMP}; as for build)")
    repeats.add_argument("--cache", type=Path, default=paths.CACHE,
                         help=f"where the dump's record and the stores are kept (default {paths.CACHE}; as for build)")
    repeats.add_argument("--workers", type=int, default=3, help="files written or read at once while building (default 3)")
    repeats.add_argument("--jobs", type=int, default=JOBS, help=f"execs at once, at most {JOBS_MAX} (a system may allow fewer)")
    repeats.add_argument("--answer-format", choices=FORMATS, help=f"as for run (default {DEFAULT_FORMAT}: D1 asks for json once, and for line only "
                         "when a cell falls outside the band)")
    repeats.add_argument("--clock", choices=CLOCKS, help=f"as for run (default {DEFAULT_CLOCK}); a system that cannot report its queue is refused "
                         "more than one exec at once on the working clock, and the refusal comes after the build")
    repeats.add_argument("--dry-run", action="store_true", help="say what it would do and cost, build and start nothing")
    check = sub.add_parser("clock-check", help="the working clock tried on a few cases: each asked alone, then queued with the others, and the answers, "
                           "tokens and charged seconds side by side (DESIGN 8.4; it spends the plans' quota)")
    check.add_argument("run", help="a run folder, or its name under the runs folder: built, with the system's store ingested")
    check.add_argument("--agents", default=",".join(AGENTS))
    check.add_argument("--config", required=True, help="the configuration to check: a system's, as mm/d1-3B")
    check.add_argument("--cases", type=int, default=CASES, help=f"how many cases (default {CASES}): a seeded sample that keeps the mix of kinds, "
                       "of the cases that do not hold the system alone")
    check.add_argument("--jobs", type=int, default=JOBS, help=f"execs at once in the queued half, at most {JOBS_MAX} (default {JOBS})")
    return parser
