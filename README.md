# MultiMediaMemoryBench (MMMB)

**MMMB tells people how well an AI agent can find things in their own large piles of video, pictures and sound, and which
search tools make it better, judged on the media people really keep and the questions they really ask.**

People keep footage from security cameras, phones, dashcams, meetings, labs and field recorders, and they rarely look at it
again, because finding one moment in it takes hours. An agent that can be told "find when the parcel was taken" can change
that. MMMB measures how close agents and their tools are to doing it well.

- **Neutral.** Any agent and any search system can be measured. The agent working alone with the files is always the baseline.
- **Right means shown to be right.** An answer counts as grounded when it points to where in the media the answer is. Every
  answer is known in advance and checked by fixed rules, not by another model.
- **"It isn't there" is an answer.** About a quarter of the questions ask for things that never happened.
- **Cost is part of the result:** time, tokens and dollars, the local GPU seconds, and the wait before a pile can be searched.
- **Honest numbers.** Results say how sure they are, and small differences are called unproven.

[VISION.md](VISION.md) says what MMMB is for. [DESIGN.md](DESIGN.md) says how it is built and what is measured.
[TESTING.md](TESTING.md) says how the build is checked and what has not been run. [MEDIA.md](MEDIA.md) says where the
media for the scenarios comes from.

## Status

P0 is built (5 November 2026). The benchmark that was copied from Multimem is now the `mmmb` package, and its core runs with
Multimem not installed.

- **Scenarios:** V7, a synthetic week of an owner's screen, phone, front door camera, microphone, notes and logs, with
  needles hidden among decoys; and V8, EgoLifeQA over the public EgoLife smart-glasses video.
- **Systems:** `files` (the agent's own tools, the baseline), `listing` (file names, sizes, times and metadata only: the floor
  and the leak filter) and `mm` (Multimem's local decision model).
- **Agents:** Claude Code and Codex, locked down.
- **Built:** the question file, grounded answers, choosing cases, cached dumps and stores, side-by-side scheduling (off unless
  asked), `mmmb compare` and `mmmb repeat`.
- **Working-time clock:** built (`--clock working|wall`, default `working`). mm reports no queue, so it runs on the wall clock and
  is refused more than one exec at once on the working clock. `mmmb clock-check` is built but not run, and for mm it needs a
  small change in Multimem first.
- **Not done, because it needs the GPU, a paid agent or Dan's go-ahead:** the pilot re-run through the new runner (at `--jobs 1`), and the
  first `mm+files` run. No GPU and no paid agent ran in P0. The commands, their cost and what they decide are in
  [DESIGN.md section 12](DESIGN.md#12-phases).
- **Next (P1):** the `embed` and `transcript` systems and the priority-1 scenarios.

Tested on Windows 11 only.

## Install

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). A real run also needs the Claude Code and Codex CLIs signed in (the
runs use your plans' quota), Git for Windows (Claude's shell tool needs it), and, for the `mm` system, an NVIDIA GPU with
Multimem's model installed.

```
uv sync                              # the core, V7 and V8, the files and listing systems, and pytest
uv sync --extra mm                   # also Multimem, for the mm system
uv sync --extra mm --group mm-gpu    # also torch and transformers, to run mm's real decision model
```

- `uv sync` is exact: it removes an extra you do not ask for. `uv run` does not remove what is installed.
- The `mm` extra is a path to Multimem's checkout (`C:/Users/danhm/tools/Multimem` in `pyproject.toml`), because Multimem
  has no git remote. Change it on another machine.
- The libraries that write the V7 week are pinned, because its bytes depend on them. ffmpeg comes with `imageio-ffmpeg`.

## Run it

```
uv run mmmb build --scenario v7_screen --scale tiny --configs files,listing
uv run mmmb run <run> --agents claude --configs files,listing --dry-run
uv run mmmb run <run> --agents claude --configs files,listing
uv run mmmb report <run>
```

- `build` writes the V7 week and its questions, ingests the dump into the stores of the systems among `--configs` (default
  `files`), and runs the shortcut probe. It prints the run folder, `D:\mmmb-runs\<date>-v7-tiny-s1`; `<run>` is that folder, or
  its name. The tiny week is 2 days and 19 questions, written in about 35 seconds. `--scale pilot` is the 3-day, 31-question
  week the pilot used; `standard` is 7 days.
- `run --dry-run` says what the run will cost and starts no agent. Without `--dry-run` it spends your plans' quota. Add
  `--jobs 3` for three execs at once (at most 4), and `--limit 12` or `--kinds last,absent` to run fewer cases.
- `report` pools the paired comparison into `report.md` and `report.json` in the run folder. `run` continues a run that
  stopped.
- `run` never ingests: a configuration whose system has no store in the run is refused, with the `mmmb ingest` command to
  make it. To use mm: `uv sync --extra mm`, then `--configs files,mm` on `build` and on `run`, with `--jobs 1` on `run`: mm gives
  no queue figure, so on the working clock more than one mm exec at once is refused (and the laptop's GPU runs one at a time).

Commands, each with `--help`:

| Command | Does |
| --- | --- |
| `mmmb build` | writes a scenario's dump and questions (or finds them cached), ingests the systems named, probes for shortcuts |
| `mmmb ingest RUN` | ingests the dump into the stores of more systems |
| `mmmb run RUN` | the execs of Claude and Codex over the questions; resumes; `--dry-run` prices it |
| `mmmb retrieve RUN` | each system asked for the needles with no agent (the GPU, alone) |
| `mmmb report RUN [RUN ...]` | figures and Markdown; several runs of one scenario are pooled |
| `mmmb compare A B` | two runs of the same questions, cell by cell: the flips, the counts, McNemar |
| `mmmb repeat RECORDED` | a recorded run again, at its agents' models and efforts, then `compare` |
| `mmmb clock-check RUN --config CONFIG` | the working clock tried on a few cases, each asked alone and then queued (not run yet; refused for mm until Multimem reports its queue) |

## Systems and configurations

A configuration is what an agent is given for one run of a question. It is written `blind`, or `<system>[/<variant>][+files]`.

| Written | The agent has |
| --- | --- |
| `blind` | no tools and no recordings: what the model already knows |
| `files` | the files and its own tools (shell, file reads, ffmpeg): the baseline |
| `listing` | the listing of the files' names, sizes, times and metadata: the floor, and a leak filter |
| `mm` (`mm/d1-3B`) | mm's tools only: `find`, `ask`, `look`, `label` |
| `mm+files` (`mm/d1-3B+files`) | its own tools and mm's: what a user who installs mm has |
| `gold` | refused: not built yet |

`mm` is the configuration the recorded pilot ran. `mm+files` is the headline for a system and has not been run yet.
The default variant of `mm` is `d1-3B`, which sees but does not hear, so `mm` is asked only the questions about pictures,
video, text and records; `mm+files` is asked them all.

## Where things are

| Path | What |
| --- | --- |
| `mmmb/core/` | the runner, the agents (`agents/`), grading, the report; knows no system |
| `mmmb/systems/` | `files/`, `listing/`, `mm/`, each with a `system.toml` |
| `mmmb/scenarios/` | `v7_screen/`, `v8_egolife/` |
| `tests/` | the tests; `tests/oracles/recorded/` holds the recorded pilot and EgoLife runs they are held to |
| `results/` | published reports: [the haystack pilot](results/2026-11-04-haystack-pilot/README.md) (4 November 2026) and [EgoLife](results/2026-11-03-egolife/report.md) (3 November 2026) |
| `D:\mmmb-runs`, `D:\mmmb-data`, `D:\mmmb-cache` | runs, prepared dumps, and the dumps' records with the ingested stores (change them with `--out`, `--temp`, `--cache`; keep them on one drive) |

The published pilot asked 24 questions about 3 synthetic days of mm and of the files alone. With mm, Codex was right on 23 of 24
against 22 without it, and Claude on 22 against 18, with fewer tokens; the gains are not yet significant. Read it in its own
[README](results/2026-11-04-haystack-pilot/README.md). Its commands and file names are those of the copied harness and no
longer apply.

## Tests

```
uv run pytest
```

runs the fast tests (about 1,000; about 8 minutes on this laptop, at 4 workers) with no agent, no GPU and no network.
Without the `mm` extra the tests in `tests/mm` skip, and say so in the summary. [TESTING.md](TESTING.md) has the markers, the
oracles, and the commands for the slow ones.
