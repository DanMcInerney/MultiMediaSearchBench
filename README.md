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

Built (6 November 2026): a standalone benchmark of four real piles of media, one for each shape, with every answer taken from
the source's own labels.

| Scenario | The pile | Questions (full size) |
| --- | --- | --- |
| `v1_security` | three outdoor security cameras, 75 minutes each, one MP4 for each unbroken stretch a camera recorded (MEVA) | 22: when an activity was first or last seen, how many times, and things that never happened |
| `i3_trailcam` | a trail camera's memory card: 4,256 photos over six months, most of them empty (Caltech Camera Traps) | 23: which days an animal came, how many, first and last visits, animals that never came |
| `a1_home_audio` | a voice recorder left running in a living room for 24 hours, one WAV (SINS) | 20: when the vacuum, the TV or a visitor started and stopped, how many times, things that did not happen |
| `m4_samples` | a folder of about 6,100 short sound clips under names that say nothing (FSD50K) | 21: find every clip of a sound, how many there are, sounds that are not there |

About a quarter of each scenario's questions are about things that never happened, and about 30% are `dev`.

- **Systems:** `files` (the agent alone with the files and its own tools: the baseline), `listing` (file names, sizes, times
  and metadata only: the floor and the leak check), and any plug-in search system. `mm` (Multimem's local model) is one,
  optional.
- **Agents:** Claude Code and Codex, locked down.
- **Checked with real agents:** a tiny run of `i3_trailcam` with Claude and Codex in `files` and `listing` (TESTING.md), and
  the baseline below: 5 questions of each scenario.
- **Extras:** V7, the synthetic week of a screen, phone, door camera, microphone and logs that mm was tuned on, and V8,
  EgoLifeQA over the EgoLife smart-glasses video. Both came from Multimem's work and stay as they were.
- **Not done:** full runs of the four scenarios with real agents (paid), and everything that needs the GPU (DESIGN.md
  section 12).

Tested on Windows 11 only.

## Baseline (6 November 2026)

The first paid run: each agent alone with the files (`files`), on 5 `test` questions of each scenario (`--splits test --limit 5`,
seed 1, full size), 900 s an exec. Claude Code ran Haiku at `xhigh` effort, Codex ran `gpt-6-luna` at `xhigh`.

| Scenario | The 5 questions | Claude (Haiku) | Codex (Luna) |
| --- | --- | --- | --- |
| `i3_trailcam` | 2 which-days, 1 count, 1 first, 1 absent | 4/5 | 3/5 (1 of them lucky) |
| `a1_home_audio` | 2 start times, 1 stop time, 1 count, 1 absent | 3/5 | 1/5 |
| `v1_security` | 2 first, 1 last, 1 count, 1 absent | 2/5 | 1/5 |
| `m4_samples` | 2 find-every-clip, 2 count, 1 absent | 1/5 | 0/5 (all 5 timed out) |
| **All four** | 20 | **10/20**, all grounded | **5/20**, 4 grounded |

- **Too few questions to rank the two.** With 5 questions a cell's 95% interval spans about 60 points, and 20 questions an
  agent still leave the gap unproven.
- **Codex ran out of time** on 7 of its 20 (all of M4, one count in A1, one absent in V1), its mean exec 615 s. Claude timed out
  once and hit its 60-turn cap twice. A full run should give more than 900 s (`--timeout`).
- **Claude's line was fixed during the run.** Claude Code refuses a Bash command over 10,000 characters, and refuses a
  redirect: Claude took the refusals for ffmpeg itself being off and gave up on 2 of M4's 5. Its files line now says both. M4's
  Claude figure is a rerun with the length named; its I3, A1 and V1 ran with the old line.
- **No leaks.** In `listing` (names, sizes and times only) both agents answered "unknown" to all 20: right only on the 4
  absent questions, as they should be.
- **Cost:** Claude's execs came to $8.07 at list price (on the plan: 69% to 72% of the week). Codex's cost is not recorded
  (on the plan: 0% to 1%), nor the tokens of an exec that timed out.

The four reports are in [results/2026-11-06-baseline](results/2026-11-06-baseline/): [I3](results/2026-11-06-baseline/i3.md),
[A1](results/2026-11-06-baseline/a1.md), [V1](results/2026-11-06-baseline/v1.md), [M4](results/2026-11-06-baseline/m4.md).

## Install

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). A real run also needs the Claude Code and Codex CLIs signed in (the
runs use your plans' quota), Git for Windows (Claude's shell tool needs it), and, for the `mm` system, an NVIDIA GPU with
Multimem's model installed.

```
uv sync                              # the core, the scenarios, the files and listing systems, and pytest
uv sync --extra mm                   # also Multimem, for the mm system
uv sync --extra mm --group mm-gpu    # also torch and transformers, to run mm's real decision model
```

- The four scenarios read their sources from the folder `MMMB_SOURCES` names, else `D:\bench-data\sources\`. It holds one
  folder per source: `meva`, `caltech-camera-traps`, `sins`, `fsd50k` ([MEDIA.md](MEDIA.md) says where each comes from). A
  build reads them and never writes there; it copies what it needs. Agents are refused any command or output that names the
  sources folder (or its parent, when it is called `sources`), so give it a name of 6 or more letters that no tool prints,
  such as `mmmb-sources`.
- The files the V1, I3 and M4 builds read (16.7 GB, unchanged copies) are the Hugging Face dataset
  [DanMcInerney/mmmb-sources](https://huggingface.co/datasets/DanMcInerney/mmmb-sources), private for now:
  `hf download DanMcInerney/mmmb-sources --repo-type dataset --local-dir D:/mmmb-sources`, then set `MMMB_SOURCES` to that
  folder. A pile built from it is the same as one built here, file for file. SINS (A1) is not in it, because its licence
  forbids passing it on: download it from Zenodo as [MEDIA.md](MEDIA.md) says.
- `uv sync` is exact: it removes an extra you do not ask for. `uv run` does not remove what is installed.
- The `mm` extra is a path to Multimem's checkout (`C:/Users/danhm/tools/Multimem` in `pyproject.toml`), because Multimem
  has no git remote. On a machine without it, set `UV_FROZEN=1`: `uv sync` and `uv run` then install from `uv.lock`
  without looking for Multimem, which they otherwise fail to find even when the extra is not asked for.
- The libraries that write the V7 week are pinned, because its bytes depend on them. ffmpeg comes with `imageio-ffmpeg`.

## Run it

```
uv run mmmb build --scenario i3_trailcam --scale standard --configs files,listing
uv run mmmb run <run> --agents claude,codex --configs files,listing --dry-run
uv run mmmb run <run> --agents claude,codex --configs files,listing
uv run mmmb report <run>
```

- `build` copies the scenario's files into a dump with their recording times, writes its questions, ingests the dump into the
  stores of the systems among `--configs` (default `files`), and runs the leak probe. It prints the run folder,
  `D:\mmmb-runs\<date>-i3-standard-s1`; `<run>` is that folder, or its name. `--scenario` is `v1_security`, `i3_trailcam`,
  `a1_home_audio` or `m4_samples` (or the extras `v7_screen`, `v8_egolife`); `--seed N` draws other questions from the same
  pile. A full-size dump is built in minutes and is found again by the next build of the same recipe.
- `--scale tiny` is a few files of the same pile and 6-8 questions, built in seconds: a smoke run, as in TESTING.md.
- `run --dry-run` says what the run will cost and starts no agent. Without `--dry-run` it spends your plans' quota, and stops
  when a plan is at 95% of its use. `--jobs` is how many execs of a configuration run at once (3 unless you say, at most 4). A
  configuration whose system needs the GPU (`mm`) runs one at a time whatever `--jobs` says, and only `--gpu-jobs N` asks for
  more. `--limit 12` or `--kinds count,absent` run fewer cases.
- `report` writes `report.md` and `report.json` in the run folder: per agent and configuration, what it got right with its
  interval, grounded and lucky answers, absent questions answered, seconds, tokens and dollars per right answer, the questions
  `listing` answered (leaks, left out of every other figure), and each other configuration beside `files`. Several runs of one
  real scenario (other seeds) are pooled; runs of several scenarios get a table over all of them. `run` continues a run that
  stopped.
- `run` never ingests: a configuration whose system has no store in the run is refused, with the `mmmb ingest` command to
  make it. To use mm: `uv sync --extra mm`, then `--configs files,mm` on `build` and on `run`. mm runs one exec at a time, because
  it needs the GPU and the laptop's driver crashed under three at once. `--gpu-jobs 2` asks for more, and until Multimem reports
  its queue the working clock refuses it (`--clock wall` lets it through, with a warning).

Commands, each with `--help`:

| Command | Does |
| --- | --- |
| `mmmb build` | writes a scenario's dump and questions (or finds them cached), ingests the systems named, probes for shortcuts |
| `mmmb ingest RUN` | ingests the dump into the stores of more systems |
| `mmmb run RUN` | the execs of Claude and Codex over the questions; resumes; `--dry-run` prices it |
| `mmmb retrieve RUN` | each system asked for the needles with no agent (the GPU, alone) |
| `mmmb report RUN [RUN ...]` | figures and Markdown; runs of one real scenario (or of V7) are pooled, V8's are reported one at a time, and runs of several scenarios get a table over all |
| `mmmb compare A B` | two runs of the same questions, cell by cell: the flips, the counts, McNemar |
| `mmmb repeat RECORDED` | a recorded run again, at its agents' models and efforts, then `compare` |
| `mmmb clock-check RUN --config CONFIG` | the working clock tried on a few cases, each asked alone and then queued (not run yet; for mm it needs `--gpu-jobs 2` or more and is refused until Multimem reports its queue) |

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
| `mmmb/core/` | the runner, the agents (`agents/`), grading, the reports (`summary.py` for the real scenarios); knows no system |
| `mmmb/systems/` | `files/`, `listing/`, `mm/`, each with a `system.toml` |
| `mmmb/scenarios/` | `realdump/` (the builder the real scenarios share), `v1_security/`, `i3_trailcam/`, `a1_home_audio/`, `m4_samples/`; the extras `v7_screen/`, `v8_egolife/` |
| `tests/` | the tests; `tests/oracles/recorded/` holds the recorded pilot and EgoLife runs they are held to |
| `results/` | published reports: [the baseline](results/2026-11-06-baseline/) (6 November 2026), [the haystack pilot](results/2026-11-04-haystack-pilot/README.md) (4 November 2026) and [EgoLife](results/2026-11-03-egolife/report.md) (3 November 2026) |
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
