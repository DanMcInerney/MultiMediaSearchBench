# MultiMediaSearchBench (MMSB): design

[VISION.md](VISION.md) says what MMSB is for and what it must stay true to. This file says how it is built and what it
measures. [CATALOGUE.md](CATALOGUE.md) lists every pile of media and gives example questions, [MEDIA.md](MEDIA.md) says where
each pile's media comes from and under what licence, and [TESTING.md](TESTING.md) says how the code is checked.

## 1. Overview

**The task.** A person hands an AI agent a large pile of their own media and asks for something in it ("when was the parcel
taken?", "find every clip of the cat"). The agent answers, and says where in the media the answer is.

**What is compared.** The baseline is the agent alone with the files and its own tools (a shell, file reads, ffmpeg). A
**search system** is anything that ingests a pile and gives the agent tools to search it: a local model that scans time
windows, an embedding index, a transcript search, a long-context model. MMSB measures the baseline, and each system beside it
on the same questions.

**What a run measures, for each question, agent and configuration:**

| Measure | Why |
| --- | --- |
| Right or wrong, by a fixed rule | no model grades another |
| Grounded or lucky: whether the evidence the answer cites (a file and a time) holds the answer | a right answer with wrong evidence is a guess |
| Seconds to answer | the person waits for it |
| Agent tokens and dollars | the person pays for them |
| The system's GPU seconds | local GPU time is free in money, not in time or heat |
| Ingest cost for each pile: seconds, GPU seconds, disk, files refused | it is paid once, before the first answer |

**Principles** (VISION.md has them in full):
- **Neutral.** The core names no system. A system is a plug-in behind one contract (3.1), and the agent alone is always the
  baseline.
- **Fixed rules.** Every answer is known in advance from the source's own labels and graded by a rule (6.2).
- **"It isn't there" is an answer.** About a quarter of the questions ask for things that never happened (5.4).
- **Cost is part of the result** (9).
- **Honest numbers.** Every proportion has an interval, a difference is called shown only when a test says so (10), nothing
  is tuned on the questions it is scored on (5.6), and leaks are hunted (7).

### 1.1 Vocabulary

| Word | Meaning |
| --- | --- |
| **Scenario** | one pile of media with its questions, as a module under `mmsb/scenarios/`. It has an ID (`I3`) and a name (`i3_trailcam`) |
| **Pile**, **shelf** | the media as a person keeps it: a memory card, a camera export, a folder of clips. The catalogue files each scenario on a shelf |
| **Source** | the public dataset a pile is built from, downloaded by `mmsb fetch` |
| **Dump** | the pile as written to disk for a run: real files under the names a device would give them, read-only, with a manifest of sizes, hashes and true start times |
| **Question** | one line of a scenario's question file: the text, its "now", the truth with evidence spans, the grading rule, tags (6.1) |
| **Case** | one question as a system sees it: times, sources and a working folder, never the text or the truth |
| **System** | a plug-in that ingests a dump into a store and serves tools to the agent (`files`, `listing`, `mm`) |
| **Variant** | one model or setting inside a system, written after a slash: `mm/d1-3B` |
| **Store** | what a system built from a dump at ingest |
| **Configuration** | what an agent is given for a question: `blind`, `files`, `listing`, `<system>`, `<system>+files` (4.1) |
| **Agent** | Claude Code or Codex, run headless and locked down (4.3) |
| **Exec** | one agent session answering one question in one configuration. Its record is a **result** |
| **Cell** | one agent in one configuration, over all its questions |
| **Run** | one built dump of one scenario with its questions, and every result made over it. It is a folder (8.7) |

## 2. How it is built

Six layers. Only the last knows any system.

| Layer | Holds | Knows about systems? |
| --- | --- | --- |
| **Dumps** | folders of real files as a person would hand them over, with a manifest | no |
| **Questions** | one JSONL file for each scenario (6.1) | no |
| **Grader** | fixed rules (6.2) and the grounded status (6.4) | no |
| **Agents** | locked-down Claude Code and Codex sessions, one for each exec | only through the tools a system hands over |
| **Runner** | case selection, scheduling, the clock, the cache, resume, statistics, the report | only through the contract |
| **Systems** | one plug-in each | each knows itself |

**The layout:**

```
mmsb/cli.py  arguments.py     the commands and the order of their steps; their names, options and defaults
mmsb/fetch.py  fetch_piles.py mmsb fetch: the files each scenario's build reads, from where each source is published
mmsb/catalogue.py             mmsb catalogue: CATALOGUE.md, made from the scenarios themselves
mmsb/core/                    the neutral half
  contract.py  config.py      the types a system and the runner exchange (3.1); a configuration's name (4.1)
  question.py  grade.py       the question file (6.1) and the value rules (6.2)
  answer.py  grounded.py      the final answer (6.3) and its grounded status (6.4)
  scenario.py  taxonomy.py    what a scenario gives the runner (5.1); shelves, categories and cues (5.2)
  build.py  cache.py  runfolder.py  fs.py  paths.py    making a run; the cache (8.6); a run's folder (8.7); files; where things live
  runner.py  schedule.py  atonce.py  select.py         the exec loop; side-by-side cells; execs at once (8.3); choosing cases (8.2)
  clock.py  clockcheck.py  gpu.py                      working time (8.4); the check of it; whether the GPU is free
  costing.py  dryrun.py       what a run will cost, said before it starts
  record.py  stats.py  summary.py  report.py  render.py   a result; the statistics; the reports (10)
  retrieval.py  compare.py  repeat.py  probe.py        runs without agents and run against run (8.8); the probe's outcome (7.1)
  agents/                     spec.py (one exec as an agent gets it), common.py, claude.py, codex.py
mmsb/systems/                 a registry that reads every system.toml; files/  listing/  mm/
mmsb/scenarios/               a registry; realdump/ (the builder the scenarios of real recordings share); one folder for each scenario
tests/                        the tests; tests/mm/ needs the mm extra
results/                      published reports
media/sources.jsonl           the sources in use, one record each (MEDIA.md is the same in prose)
```

**Where things live.** Everything MMSB makes is under one home, `MMSB_HOME` (default: the user's home folder).

| Folder | Holds | Changed with |
| --- | --- | --- |
| `<home>/mmsb-runs` | a run's folder | `--out` |
| `<home>/mmsb-data` | the dumps (`mmsb-<key10>-data`), each run's ffmpeg (`mmsb-<tag>-bin`) and working folders (`mmsb-<tag>-w<k>`) | `--temp` |
| `<home>/mmsb-cache` | the dumps' records and the ingested stores | `--cache` |
| `<home>/mmsb-temp` | the tests' runs | |
| `<home>/mmsb-sources` | the downloaded sources, one folder each; a build reads them and never writes there | `MMSB_SOURCES` |

- The first three must share a volume, because a store is copied into a run with hard links. The commands refuse folders on
  two volumes.
- The data folder may hold only `mmsb-*` folders, because an agent with its own tools can list it.
- The runs folder must not hold the data folder: an agent is refused the folder its run is in, so that it cannot read the
  runs beside its own.

**The core is neutral, and a test holds it to that** (`tests/test_neutral.py`):
- every module of `mmsb.core`, `mmsb.scenarios` and each system but `mm` imports with MultimediaSearch absent, and naming
  `mm` then answers "mm needs the mm extra: uv sync --extra mm";
- nothing outside `mmsb/systems/mm` and `tests/mm` imports MultimediaSearch;
- nothing in `mmsb/core` or `mmsb/scenarios` imports or names a system, apart from a short list that gives its reason for
  each (`files` as the name of a configuration).

Each system's dependencies are an optional extra of `pyproject.toml`. The `mm` extra is MultimediaSearch (`mms`), a direct
reference to its repository at one commit, which `uv.lock` pins as well.

## 3. Systems

### 3.1 The contract

A system is a folder under `mmsb/systems/` with a `system.toml` and a class. The types are in `mmsb/core/contract.py`.

**`system.toml`.** The registry reads every `system.toml` as TOML and imports a class only when a configuration names its
system. That is how the core knows what every agent shell must keep out while the system itself is not installed. A system
whose extra is missing is refused when it is named, with the extra's name. There is no fallback.

```toml
[system]
name = "mm"                        # the folder's name
class = "mmsb.systems.mm.system:Mm"
extra = "mm"                       # the pyproject extra it needs; omit for none
group = "mm-gpu"                   # the dependency group it needs as well; omit for none
license = "MIT"
default_variant = "d1-3B"          # omit for a system without variants
horizon = "cut"                    # or "view": see serve, below

[isolation]                        # applied to every agent with its own tools, whichever system is being run
env_drop_prefixes = ["MMS_"]       # environment variables dropped from the agent's shell
deny_words = [".mms/"]             # words no command of the agent may hold
deny_package_roots = ["mms"]       # installed packages whose repository no command may reach

[isolation.env_nowhere]            # variable -> a folder in the working folder that is never made
MMS_HOME = "no-mm-home"
```

**Two objects.** A daemon and a loaded model span many cases, so a system is not a set of calls for each case. The `System`
builds stores, and `open` returns a `Session` that lives across the cases of one agent and one configuration.

| `System` operation | Does | Rules |
| --- | --- | --- |
| `describe(variant)` | a `Description`: name, version, variant, the media it reads (image, video, audio, text, records), local or not, whether it needs a GPU, network, licence, `config_hash`, `store_hash`, horizon, whether it has an oracle, whether it is a leak filter | the version and `config_hash` are recorded with the results, and a configuration that has results in a run is refused when its `config_hash` has changed. `store_hash` covers only what ingest depends on, so the variants of one system can share a store |
| `ingest(dump, store)` | builds the system's own state in `store` from a read-only dump | sees the dump only, never the questions; timed. Returns seconds, GPU seconds, bytes written, each file refused with the reason, each file stored at a time other than its own (`misdated`), and figures it measured. A misdated file refuses every run over the store |
| `open(store, dump, context)` | a `Session` over the store | one session for each agent and configuration; `close()` ends it |
| `check_ready(variant)` | raises, saying why, when `open` would fail now (weights not downloaded, an environment not installed); starts nothing | asked of every configuration before the first agent starts, so that a run does not spend its first cell and then fail at the next |
| `preview(store, dump, context)` | a session that starts no process and changes no file | answers `max_jobs`, `estimate`, `serve` and `queue_s` from the store's files, for `--dry-run` and the estimates printed before a run |

| `Session` operation | Does | Rules |
| --- | --- | --- |
| `serve(case)` | a `Served`: the agent's tools (an MCP server's command, arguments, environment and tool names), the system's own instructions, the word the prompt uses for the system, folders the agent may open, and fields to add to the result | a `Case` never holds the question's text or truth. Only media before the case's horizon may be answered from. A system that cuts its own index to the horizon declares `horizon = "cut"`; one that cannot declares `"view"`, and the scenario hands the agent a folder cut to the horizon |
| `max_jobs(cases, requested)`, `admits(case, busy)` | how many cases run at once, and whether one may start beside those running | the system's own limits; they can only lower what the runner allows (8.3) |
| `estimate(sources, ranges)` | a pure cost function: the system's seconds for a range | asked about the question's `cost_ranges`, never its truth |
| `audit(case, calls)` | fields for the result, from the exec's tool calls (its GPU seconds among them) | may raise to end the command |
| `reset(case, calls, others)` | takes the exec's state out of the store | runs before the exec's working folder is emptied, so no exec can see another's |
| `queue_s(case)` | the seconds this case has waited in the system's queue so far | read live by the working clock (8.4); `None` means the system cannot say |
| `usage(case, calls)` | GPU seconds, queue seconds and API dollars of this case | for each case, not "since the last call", because cases overlap |
| `oracle(query)` | optional: the system's ranked moments for a plain query, with no agent | for the retrieval run (8.8) |
| `close()` | stops what the session started; returns named lists | each list that is not empty is written as `results/<agent>/<config>/<name>.json` |

A session with nothing to do (`PlainSession`) serves no server and costs nothing. `files` and `blind` use it.

**Rules every system follows:**
- **Ingest is blind.** It runs before any question is known. It may take as long as it likes, and the time is reported as
  the person's wait.
- **Systems are frozen before a test run.** The version and configuration hash go with the results. Tuning happens on the
  dev split only (5.6).
- **Instructions are general.** A system's instructions are short (about 2,000 tokens) and the same for every scenario. They
  may teach the agent the system's tools, never what is in a scenario.
- **No network,** unless the system declares itself hosted.
- **A case is charged its own work.** GPU seconds are reported for each case. A cache inside a system is allowed if it would
  also serve a real user, and across cases it may keep only what ingest built, so no case gains from another's.
- **One GPU, one user.** The runner gives the GPU to one cell at a time and, by default, to one exec at a time (8.3). A
  system does not start GPU work outside its sessions.

### 3.2 The systems

| System | What it is | Tools it gives the agent | Ingest |
| --- | --- | --- | --- |
| `files` | the baseline: no system | none beyond the agent's own (shell, file reads, ffmpeg) | none |
| `listing` | the floor, and the leak filter (7.2) | `list_files` and `file_info`: names, sizes, times, EXIF and container fields; no pixels and no sound | seconds |
| `mm` | [MultimediaSearch](https://github.com/DanMcInerney/MultimediaSearch): a small local decision model reads every window in a time range against the agent's question, at question time | `find`, `ask`, `look`, `label`, and its skill as instructions | CPU only; the GPU works for each question |

**`listing`.**
- `ingest` writes `listing.jsonl`, a row for each file: path, source, size, modified time, the EXIF tags Pillow reads, and
  PyAV's container fields (format, duration, width, height, fps, codec, creation time, sample rate). It reads what any tool
  on the files could read and copies nothing that only the manifest knows.
- The agent has the two tools and nothing else: no built-in tool, and the server is given no path to the dump.
- Its only time is the file's modified time. So `ingest` compares each file's modified time with the manifest's true time,
  read for that check only, and reports a file more than one second off as `misdated`, which refuses the run. A dump whose
  files carry a download's times cannot be listed.
- It declares every medium as read, so every question is asked of it. It is a filter for leaks, not a reader.

**`mm`.**
- Each variant is a decision model MultimediaSearch knows. `d1-3B`, the default, sees and does not hear; `d1-Omni-2B` sees
  and hears. A variant's description says what it reads, and that decides which questions `mm` alone is asked (4.2).
- The store is an ingested MultimediaSearch home. Its ingest does not depend on the variant, so the variants share one
  store, and each agent and variant gets its own working copy with its own daemon.
- It cuts its index to each case's horizon (`horizon = "cut"`), runs a case alone when its estimate is 300 GPU seconds or
  more, and takes each exec's labels and jobs out of the home when the exec ends.
- It keeps no wait time for each job, so `queue_s` is `None` and its execs are timed on the wall clock (8.4).
- When the machine's clock is more than an hour past a question's "now", its prompt adds a sentence telling the agent to
  give times as dates and not as ages.
- The model's weights are looked for in `MMS_MODELS`, else `~/.mms/models`. `uv sync --extra mm --group mm-gpu` installs
  what running the model needs.

### 3.3 Adding a system

1. Make `mmsb/systems/<name>/` with a `system.toml`. `mmsb/systems/listing/` is the smallest system with a server of its
   own. Put its dependencies in an extra of `pyproject.toml`.
2. Write the class: `describe`, `ingest`, `open` and `check_ready`; `preview` for dry runs; `oracle` if it can rank moments
   without an agent.
3. Run it: `--configs files,<name>` (or `<name>+files`) on `mmsb build` and on `mmsb run`.

Nothing else changes: the registry finds the folder, and the core never learns the name.

## 4. Configurations and agents

### 4.1 Configurations

Each exec is a fresh agent session in one configuration:

| Written | The agent has | Role |
| --- | --- | --- |
| `blind` | no tools and no recordings | the floor of what the model already knows of a public dataset |
| `listing` | the listing only | the floor of what the file list gives away; a question it answers is a leak (7.2) |
| `files` | the files and its own tools | the baseline, and the control every other configuration is set beside |
| `<system>+files`, as `mm+files` | its own tools and the system's | **the headline for a system**: what a person who installs it has |
| `<system>`, as `mm` | the system's tools only | a diagnostic: what the system's tools do alone |

- **How it is written.** `blind`, or `<system>[/<variant>][+files]`, with the system's default variant filled in:
  `mm+files` is `mm/d1-3B+files`.
- **Folders.** A configuration's results are in `results/<agent>/<dir>/` and its execs in `execs/<agent>/<dir>/<qid>/`, where
  `<dir>` is `blind`, `files`, `listing`, `mm-d1-3B` or `mm-d1-3B+files`. Two configurations of one run may not share a
  `<dir>`.
- **The headline compares `files` with `<system>+files`.** Taking the agent's shell away is not what any user does, so the
  system alone is a diagnostic.

### 4.2 Which questions a configuration is asked

- A configuration with its own tools, and `blind`, are asked every question.
- A configuration with a system's tools only is asked the questions about the media its system reads
  (`Description.reads`). `mm/d1-3B` does not hear, so it is not asked the sound questions; `mm/d1-3B+files` is, because its
  agent has ffmpeg.
- `--controls CONFIG` gives the text and records questions to that one configuration only, among those without their own
  tools.
- The run writes what each configuration was asked to `systems.json`, and the report counts only those.

### 4.3 Agents

Claude Code and Codex. Each exec is one headless session.

- **The same settings in every configuration.** Claude is `sonnet` at `high` effort with a cap of 60 turns; Codex is
  `gpt-6.1-terra` at `high`. `MMSB_CLAUDE_MODEL`, `MMSB_CLAUDE_EFFORT`, `MMSB_CODEX_MODEL`, `MMSB_CODEX_EFFORT` and the
  flags of `mmsb run` (`--claude-model`, `--claude-effort`, `--codex-model`, `--codex-effort`) change them. Each result
  records the model and effort that made it.
- **The prompt** is the scenario's: the date and time "now", what the pile is and where it is (or which system holds it),
  the seconds the exec is held to, the question, and the answer format (6.3). A system adds its own instructions and tool
  schemas. Nothing in the agents names a system: an agent builds its command line from what the system served and whether
  the configuration has its own tools.
- **The plan's quota.** The agents run on the user's plans, not an API key. An agent stops when its plan is at 95% of its
  weekly use, and the same command continues later.
- **Another agent** is a module in `mmsb/core/agents/`: a function from one `SessionSpec` to the common result.

### 4.4 How an agent is locked down

An agent must not reach the answers, the other questions' working folders, the network, or the owner's own setup.

**Four modes,** from whether the configuration has its own tools and whether its system serves a server:

| Mode | The agent may use |
| --- | --- |
| blind | nothing: any tool use ends the exec |
| a server | the one MCP server the system serves, and no built-in tool |
| own tools | file reads and a shell that runs ffmpeg and read-only commands, over the dump and its working folder |
| both | the server and its own tools |

**What every exec has:**
- **None of the owner's setup:** no user settings, hooks, plugins, skills, personal instructions, memory or other MCP
  servers. Claude runs with only the settings the run gives it; Codex ignores the user's configuration and rules.
- **No network.** Codex's shell runs in a sandbox with no network; Claude's shell is limited to ffmpeg and read-only
  commands and its web tools are off. No one looks up a public dataset's answers.
- **A clean shell.** Its `PATH` is the run's own ffmpeg (there is no ffprobe) and the system folders, so none of the
  owner's programs and no virtual environment. The variables each system's `[isolation]` names are dropped, or lead to a
  folder that is never made.
- **A working folder of its own,** emptied before and after the exec.

**The deny words.** No command of an agent with its own tools, and no output of one, may hold a deny word. They are:
- the run's folder, the folder it is in (so the runs beside it), the runs and cache folders, and the tests' folder: each
  holds questions with their answers;
- this repository's checkout (and the main checkout, from a worktree), which holds the scenarios' code;
- the agents' own homes (`.claude`, `.codex`);
- each system's `deny_words`, and the repositories of the packages it names;
- the scenario's words: its own code, the source's names, and the name of the sources folder;
- the working folders of the other execs, by path and by glob;
- the dump itself, when the agent reads a folder cut from it (a scenario with a horizon);
- `-hwaccel` and `nvidia-smi`, when the agent is kept off the GPU (8.3).

A word that begins `in:` is checked only in what the agent runs, not in what it reads: a listing may name a folder that a
command may not open. A run is refused when the path of its own data folder holds a deny word, because every command
naming the folder would be an escape.

**The escape rule.** Every event of the session is checked as it arrives. A tool the mode does not have, or a deny word in
a command or its output, ends the exec at once as `escaped`, which is graded wrong. The prompt of an agent with its own
tools says so: nothing outside the files' folder and its working folder is its to use. A dump that changed during a run is
also an escape, and stops that configuration.

## 5. Scenarios

### 5.1 What a scenario is

A scenario is a module under `mmsb/scenarios/` with the names of the `Scenario` protocol (`mmsb/core/scenario.py`) as
functions, and a line in the registry. It knows its media and its questions and never knows a system.

| Name | Gives |
| --- | --- |
| `name`, `code`, `generator` | `i3_trailcam`, `I3`, and a number bumped when the scenario makes something other than before |
| `shelf`, `filing` | how the catalogue files the pile and each question template (5.2) |
| `recipe(**options)` | what the dump is made from: the source, the scale, the seed, the days, the machine's UTC offsets |
| `hashes(recipe)` | the hashes of the plan's questions and files, which a run records and checks when it is opened again |
| `build(recipe, out, workers)` | the dump and its questions |
| `probe(...)` | the scenario's shortcut probe (7.1) |
| `prompt(question, access)` | the prompt, by how the agent reaches the recordings |
| `view(...)` | the folder an agent with its own tools reads: the dump, or a folder cut to the question's horizon |
| `deny_words(...)` | the scenario's deny words (4.4) |
| `after_exec`, `run_fields`, `release` | fields it adds to a result and to `run.json`; cleaning up |
| `report(runs, control)` | its report (10) |

- **One dump, many questions.** All a scenario's cases share the dump, so it is built, probed and ingested once for each
  system.
- **Scales.** A scenario built from a dataset has a `tiny` scale (a few files of the same pile, built in seconds, for tests
  and a smoke run) and a `standard` one (the full pile). `v7_screen` has scales of its own, by how many days it generates,
  and `v8_egolife` has one.
- **Seeds.** `--seed` draws other questions from the same pile.
- **A fixed "now".** Each question is asked as of a fixed time after the pile's last recording, so a run made later asks
  the same question.
- **A horizon,** where the source asks each question as of its own time (V8): nothing recorded after it may be answered
  from. A system cuts its index to it, or the scenario hands the agent a folder cut to it.

### 5.2 Shelves, categories and cues

The benchmark is organised the way a person thinks of it: a shelf of their own media, and a question of the kind people
ask. `CATALOGUE.md` is the list, made by `mmsb catalogue` from the scenarios themselves.

**A shelf** is a name, a medium (video, image, audio or mixed), a life area (home, work, school, fun) and a line describing
the pile.

**Eight categories, by what the person wants** (`mmsb/core/taxonomy.py`), whatever the media:

| Key | Category | A person says | The answer |
| --- | --- | --- | --- |
| `find` | Find it | "Find the photo of the parking-level sign." | one file or moment |
| `find_all` | Find every one | "Show me every clip with the cat." | a set of files, moments or dates |
| `when` | When | "What time did the van come? When did I last see it?" | a time or date |
| `did` | Did it happen | "Did the smoke alarm go off? Was there ever a bear?" | yes and when, or not there |
| `how_many` | How many, how long | "How many times did the dog bark?" | a number |
| `what` | What did it say or show | "What was the total? What deadline did they give?" | a short value, read or heard |
| `who` | Who or which | "Who suggested the price? Which camera saw it first?" | a person, file or place |
| `order` | Before and after | "Was the parcel before the rain?" | an order, or a neighbouring event |

**A cue** is what must be perceived to answer: sight, sound, speech, text or a mix. It tells a system with no speech apart
from one that misses the needle.

The filing is on each scenario (`shelf`, and `filing`, each template's category and cue), not on the question lines, so
refiling a question changes no question file and no run.

### 5.3 The scenarios

Twenty-three, covering video, images, long audio, many short audio files and one mixed pile. CATALOGUE.md has each pile's
size, source and question counts.

| Media | Scenarios |
| --- | --- |
| Video | `v1_security` (three security cameras), `v2_dashcam` (dashcam event clips), `v3_match` (football match clips), `v4_meetings` (work meetings), `v8_egolife` (smart-glasses days), `v9_lectures` (lecture recordings), `v10_home_clips` (phone clips at home) |
| Image | `i1_camera_roll` (a phone camera roll), `i2_screenshots` (phone screenshots), `i3_trailcam` (a trail camera's card), `i4_site_photos` (building inspection photos), `i7_receipts` (scanned receipts), `i8_course_pages` (course notes and slides), `i9_nature` (nature photos) |
| Audio | `a1_home_audio` (a day from a room recorder), `a3_meetings` (research group meetings), `a4_field` (a night of a field recorder), `a6_lecture_audio` (recorded lectures), `a7_podcast` (a podcast backlog), `a8_music` (a music library), `m3_voicemail` (voicemails), `m4_samples` (a sound-effects library) |
| Mixed | `v7_screen` (a developer's week: screen recording, notes, logs, phone photos, a porch camera and a microphone) |

- **Real recordings.** All but one are built from a public dataset by the shared builder (5.5).
- **`v7_screen` is generated.** The benchmark makes the whole week, with every needle planted among decoys. It has text and
  records questions as well as media ones, a time hint of varying vagueness on each question, and a report of its own.
- **`v8_egolife` is a public benchmark's questions** (EgoLifeQA, multiple choice), each asked as of its own time. The user
  downloads EgoLife and passes its folder with `--source`. It has no shortcut probe; `blind` is its floor.
- **Honest about what is staged.** Some sources are role-played, acted or scripted, and one pile's truth is a speech
  recogniser's transcripts. The catalogue says so beside each shelf.

### 5.4 Truth, and what a question avoids

**Every answer is computed from the source's own labels** by a question template: a dataset's activity spans, species
labels, sound labels, transcripts, receipts' fields. No model writes or checks a question, and no agent's hearing or sight
is the truth. Each question keeps its evidence: the spans (a file, and seconds into it) that hold the answer.

**Scenarios of talk** (lectures, meetings, a podcast) share `realdump/speech.py`: terms confirmed in the timed transcripts,
terms confirmed to be in none of them, and questions written by reading the transcripts, each resting on a phrase that the
build checks is in the transcript at its evidence.

**Absent questions.** About a quarter of the questions ask about something that never happened. They read exactly like the
others and are filed as the question they read as. An absent question is certain in one of two ways:
- a term in no complete label set (a word in no transcript, a shop on no receipt, an animal the camera never photographed);
- a **held-out needle**: a real question whose files, and anything easily taken for them, are left out of the dump.

**What a question avoids,** each checked in its scenario's tests by a plain reader of the labels:
- a label a careful annotator may have missed, or that the source marks unsure;
- a class easily taken for another in the dump (a night coyote for a fox; cooking, eating and dishwashing, which run into
  one another with no edge in the sound);
- a thing too small to be sure of (a bird among thousands of empty frames);
- a window whose edge cuts an event, or that spans a stop of the recorder;
- a moment the labels place loosely, unless the tolerance is widened to cover it;
- an answer the file list gives away (the first photo on the card as "the first visit").

### 5.5 The shared builder

`mmsb/scenarios/realdump` is what the scenarios of real recordings share. Each such scenario is then a reader of its source
and labels, a writer of its files, and its question templates.

- **Files as a device leaves them.** Each file is copied (never linked: the source stays read-only) into the dump under the
  name a camera, phone or recorder would give it, in nested folders where the device makes them, with the dataset's
  identifiers removed. Its modified time is set to its recording time, and the manifest records its size, hash and start.
- **Times are wall-clock times.** A source's labels give times with no zone, so the dump keeps them as local time on the
  machine that builds it: the file's time, its own metadata and the question's "now" read the same clock. The recipe
  records the machine's UTC offsets for its days, and a recipe made in another zone is refused.
- **Questions.** One helper makes a question with its grade, tags, scope and evidence spans, and puts the question's own
  numbers and times into the grade's `ignore`, so an answer may repeat them.
- **The prompt** says the time it is ("It is now Monday 23 December 2013, 09:00."), what the pile is and where it is (or
  which system holds it), how its files are named and timed, the rule about staying inside its folders, the seconds the
  exec is held to, the question, and the answer request.
- **The deny words:** the scenario's and the builder's code, the source's names in commands, and the name of the sources
  folder (its own name, or its parent's when it is called `sources`; at least 6 letters, so that ordinary output does not
  hold it).
- **The probe** (7.1) and **the report** (10).

**Fetching** (`mmsb fetch --scenario NAME|all`). It downloads what each build reads into the folder the build reads, with
no account, from where the publisher hosts it. MMSB publishes no media, so the user is bound by each source's licence.
- A file is renamed into place only when whole, so the same command continues a fetch that stopped.
- A count that is not what the build needs is refused before anything large is fetched: another count would be another
  dump.
- Label files are taken from fixed commits of their repositories where the publisher keeps them in one, so the answers do
  not change under a published result.
- It waits as a server asks (429, Retry-After) and asks for a few files at a time.
- A source whose licence forbids passing it on is fetched from its publisher for the user and never mirrored.

### 5.6 Dev and test

- **The split.** About 30% of a scenario's questions are `dev` and the rest `test`, by a draw seeded with the run's seed.
  `v7_screen` splits by seed instead (seeds 1 to 10 are `dev`), and all of `v8_egolife` is `dev`, because a system was
  tuned on it.
- **What may be tuned on.** Any system, its instructions and its settings are built and tuned on `dev` only. A `test`
  result is published only from a frozen system version, whose version and `config_hash` are in the run.
- **Selecting.** `mmsb run --splits dev` or `--splits test`.
- **What is published.** The catalogue shows examples from `dev` only. `mmsb catalogue --full FILE` writes every question
  with its answer and stays out of the repository.

### 5.7 Adding a scenario

1. Make `mmsb/scenarios/<name>/` with the names of 5.1 as functions. For real recordings, build on `realdump`: a reader of
   the source and its labels, a writer of the files, and question templates with their rules.
2. Give it a `shelf` and a `filing`, and add a line to the registry in `mmsb/scenarios/__init__.py`.
3. Add its source to `media/sources.jsonl` and MEDIA.md, and a fetch function for it.
4. Test that every answer follows from the labels, that the questions avoid what 5.4 lists, and that the probe passes on
   the built dump.

Bump `generator` whenever the scenario would write other files or questions for the same recipe: the dump's key and every
run's check depend on it.

## 6. Questions, answers and grading

### 6.1 The question file

One JSON object for each line, kept outside the dump and never visible to a system or an agent. `load_questions`
(`mmsb/core/question.py`) is the one place it is parsed: it refuses an unknown key, a missing key, a value of the wrong
type, an unknown rule or a repeated `qid`, and everything after it holds dataclasses.

```json
{"qid": "I3-first-dog", "scenario": "I3", "split": "dev", "now": "2014-03-01T09:00:00-08:00",
 "question": "When did my trail camera first photograph a dog?", "attachments": [], "kind": "when", "hint": "none",
 "truth": {"value": "2013-10-17 16:43:35", "evidence": [{"file": "DCIM/100RECNX/IMG_0412.JPG", "t0": 0.0, "t1": 0.0}], "wrong": []},
 "grade": {"rule": "time", "truth": null, "time": "16:43:35", "tol_s": 60.0, "wrong": [], "ignore": [], "date": "2013-10-17"},
 "tags": {"modality": "image", "template": "first", "level": "none", "absent": false, "shape": "<date and time>"},
 "options": null, "scope": {"sources": ["camera"], "since": 1378800000.0, "until": 1393600000.0},
 "group": "I3-first-dog", "horizon": null, "cost_ranges": [[1378800000.0, 1393600000.0]], "retrieval": null}
```

(The values are illustrative.)

| Key | Holds |
| --- | --- |
| `now` | the time the question is asked at, ISO with an offset |
| `kind` | the kind of question, a short word; `absent` for a question about something not there |
| `truth` | the value, the evidence spans (seconds into a file), and spans that are wrong answers |
| `grade` | the rule and what it needs (6.2) |
| `tags` | `modality`, `template`, `level`, `absent`, and what the scenario adds. Cases can be selected by any tag |
| `options` | the choices of a multiple-choice question, else null |
| `scope` | the sources and the time range the question is about |
| `group` | two cases of one group never run at once, because they would show each other the needle |
| `horizon` | the time after which nothing may be answered from, else null |
| `cost_ranges` | the ranges a system's `estimate` is asked about, so that no truth enters a `Case` |
| `retrieval` | what the retrieval run asks (8.8), else null |

The file is in the dump's cache record and copied into each run. Both places are on every agent's deny list, and it is
never in the data folder.

### 6.2 Grading rules

`mmsb/core/grade.py`: pure rules, no model. An answer is casefolded, number words become digits, and times become minutes
after midnight. `None` is wrong. **A hedge is wrong:** an answer with a second number, time or date that is neither the
truth nor the question's own.

| Rule | Right when |
| --- | --- |
| `number` | the truth is among the answer's numbers, and no other number and no listed wrong answer is |
| `count` | the answer's numbers are exactly the truth (`once`, `twice`, `thrice` count; the digits of a name such as `cam3` do not) |
| `time` | a time of the answer is within `tol_s` of the truth, and every other time in it is too, or is the question's own |
| `code_time` | the truth's digits and a time within `tol_s`, with no other number of as many digits |
| `absent` | the answer abstains: a word such as "unknown", "none", "never" or "not", and no number or time of its own |
| `choice` | the answer starts with the truth's letter, A to D, and names no second letter |
| `set` | the answer names every one of `items` (ISO dates, or file names) and nothing else of their kind |
| `text` | a short answer (at most 25 words) holds one of the accepted phrases as whole words, offers no second guess and names no wrong one |
| `amount` | the answer's one number is within the tolerance of the truth, in the units the question asks for |

- **A date as well.** A `time` or `code_time` grade may hold a `date`. The answer must then name that day, as a weekday, as
  "29 October", as "the 29th" or as `2026-10-29`, and no other. A time alone is wrong.
- **The question's own numbers.** `ignore` holds the numbers and clock times the question itself contains, which an answer
  may repeat. The digits of a name (`cam3`, `IMG_0003.JPG`, a path) and a month of a year are not numbers.
- **Abstaining with a range.** An answer to an absent question may say what it searched ("nothing between 9:00 and 17:00")
  when the range is an hour or more and the whole answer is negative. A short range is a sighting, not an abstention.
- **A set of files** is matched by base name, in any case, wherever the agent saw the file. The verdict needs the exact
  set, and the F1 of the items found is kept beside it as `items_f1`.
- **Two-way questions.** Where `text` has at most two wrong phrases (the other side of "before or after"), an explanation
  may name the other side, and the side named first is the answer.

### 6.3 The answer

`mmsb/core/answer.py`. A run asks for one of two formats, kept in `run.json`; a run with results keeps its format.

**`json`, the default.** The agent ends with one JSON object:

```json
{"answer": "17 October 2013, 16:43", "evidence": [{"file": "DCIM/100RECNX/IMG_0412.JPG", "t0": 0, "t1": 0}]}
```

- **Parsing.** The answer is that of the last complete JSON object in the final message; a code fence around it is just
  text. It needs a string `answer` and a list `evidence`. Anything else is a failed answer, counted as `unparsed`.
- **A citation names its moment in one of two ways,** because an agent that has only a system's tools never sees a file
  path:
  - `{"file", "t0", "t1"}`: seconds into the file, clock times (`H:MM[:SS]`), or ISO times. A file is its manifest path, or
    a longer path ending in it.
  - `{"source", "t0", "t1"}`: ISO times that carry an offset.

  A system that works in wall-clock time and an agent that works in file offsets are graded alike.
- **"Not there"** is the answer `unknown` with an empty `evidence`.

**`line`.** The agent ends with one line, `ANSWER: <answer>`. There is no evidence, so a `line` result has no grounded
status. `--answer-format` picks the format.

### 6.4 Grounded and lucky

`mmsb/core/grounded.py`. `passed` is the value rule with no problem in the exec. The grounded status is told beside it:

| The answer | Status |
| --- | --- |
| the value fails its rule | `wrong` |
| an absent question, the value passes, no evidence | `grounded` |
| an absent question, the value passes, evidence given | `wrong` |
| the value passes, at least one citation, and every citation hits a truth span | `grounded` |
| the value passes, and it cites nothing or a citation hits nothing | `lucky` |
| the `line` format | `not_recorded` |

- **A hit.** A citation is placed in absolute time and compared with the question's evidence. It hits a truth span of a
  video within 10 s, of a sound within 5 s, and of a line of a log or transcript within 5 s, always on the same source. It
  hits a picture, a note or a version of a file only by being that file, or by a span that holds the file's own time to the
  second.
- **A broad span is not penalised.** The rule is overlap within the tolerance, so a citation of a whole file that holds
  the truth hits it.
- **Unresolved.** A citation whose file the dump does not have hits nothing and is counted (`evidence_unresolved`).
- **Counts.** A count question's result also holds the event F1 of its citations with a 1 s and a 5 s collar (`f1_1s`,
  `f1_5s`), as DCASE scores sound events: a maximum matching, so one wide citation finds one moment, and citations that
  find nothing are false alarms.
- **Refused early.** A `json` run with a question that has something to find and no evidence span is refused before any
  agent starts.

## 7. Leaks and shortcuts

A score must come from doing the task. Three things guard that.

### 7.1 The shortcut probe

Every build probes its dump before any run, and the probe is the scenario's own. The core keeps the outcome (`passed`,
`failed`, `skipped`, or `none` for a scenario with no probe) in `run.json`, cached with the dump, and writes the probe's
findings to `shortcuts.md` in the run folder. `mmsb run` refuses a run whose probe failed unless `--anyway`;
`mmsb build --no-probe` skips it, and the report says so.

- **Real recordings** (`realdump.probe`). No file's path, and no text in its metadata, names a label: each label as a
  phrase and by its words of three letters or more (but the commonest, such as "person"), with the plain words a scenario
  adds. Metadata is read by suffix (a JPEG's segments before the image, a WAV's chunks but the samples), and a file of a
  suffix with no reader fails. File sizes are not checked: they are the recordings' own, and nothing can be done about them
  without changing the media.
- **The generated week** (`v7_screen/shortcuts.py`). For each media question it measures what cheap filters would give an
  agent: how many other frames a colour filter or a frame difference would shortlist, how many other sounds a band filter
  would, and whether the needle's file size stands out. Each must leave enough candidates that the filter does not find the
  needle alone.

### 7.2 The `listing` leak filter

A system declares itself a leak filter (`Description.leak_filter`; `listing` does). A configuration is a filter only if it
has no tools of its own, so `listing+files` is an ordinary tested configuration.

- A question with something to find that a filter configuration answers right is **answerable from metadata**: the file
  names, sizes, times or tags give it away, or the model already knew it. One agent's filter passing it is enough.
- The report names such questions and leaves them out of every other configuration's figures.
- **Absent questions are exempt.** An agent with nothing to read answers "unknown", which is right for every absent
  question, so none of them says anything about a leak.
- It catches what the probe cannot, such as a photo whose time alone gives the answer.

### 7.3 The rest

- **No network** for agents or local systems (4.4).
- **Files are named as a device names them,** and dataset identifiers are removed (5.5).
- **The deny words** keep the questions, the scenario's code and the sources away from an agent with a shell (4.4).
- **`blind`** shows what a model already knows of a public dataset, where that matters.

## 8. Running

### 8.1 The commands

| Command | Does |
| --- | --- |
| `mmsb fetch --scenario NAME` | downloads the sources a scenario's build reads |
| `mmsb catalogue` | prints CATALOGUE.md |
| `mmsb build --scenario NAME [--scale S --seed N --configs A,B]` | writes the dump and questions (or finds them cached), ingests the systems named, probes for shortcuts, and writes the run folder |
| `mmsb ingest RUN --configs A,B` | ingests the dump into the stores of more systems |
| `mmsb run RUN [--agents ... --configs ...]` | the execs; resumes; `--dry-run` prices it |
| `mmsb report RUN [RUN ...]` | the report (10) |
| `mmsb retrieve RUN`, `mmsb compare A B`, `mmsb repeat RUN`, `mmsb clock-check RUN --config C` | runs without agents, and run against run (8.8) |

`RUN` is a run folder, or its bare name under the runs folder. `run` never ingests: a configuration whose system has no
store in the run is refused, with the `mmsb ingest` command that makes it.

### 8.2 Choosing cases

Running faster means choosing fewer cases. Nothing else about a run changes.

- **Select** with `--scenarios`, `--kinds`, `--modalities`, `--splits`, `--tag NAME=VALUE` (repeatable) and `--only QID,..`.
  A value no question has is an error. A selection prints what each step left, and one that picks nothing is refused.
- **`--limit N`** draws N cases at random, stratified by scenario and kind, with a fixed seed. A small run keeps the mix
  of the full one and is the same small run each time.
- **`--discriminating RUN[,RUN]`** keeps only cases that separated configurations in earlier runs: right in some
  configuration of one agent and wrong in another. A case right everywhere or wrong everywhere tells a paired comparison
  nothing. Two such cases stay in as canaries.
- **Before it starts,** the runner prints what the selection will cost, and `--dry-run` stops there:
  - cases × agents × configurations, and how many run at once;
  - wall time, GPU seconds, tokens and dollars, from the medians of what the same cell took in earlier runs, with the mean
    beside them, because a few timeouts make a run longer than its median says. A cell with no earlier exec is counted at
    2.5 minutes an exec (1 for a blind one) and says so;
  - the limits only a system can state: execs at once, and cases whose work cannot finish inside an exec's time.

  `run --dry-run` also prepares the first exec of each cell in the run's `dry-run/` folder and prints its command line. It
  starts no agent and opens no session.

A small selection cannot show a difference (10), and its report says so.

### 8.3 How many at once

**One rule** (`mmsb/core/atonce.py`), asked by the runner, every dry run and estimate, and `clock-check`:

- A cell runs `--jobs` execs at once (default 3, at most 4).
- A cell whose system needs the GPU runs `--gpu-jobs` at once (default 1, at most 4). Several model execs on one GPU
  compete for its memory and can take down its driver, so more than one is only what the user asks for, and the command
  prints a warning when it will run several. `--jobs` and `--clock` never raise it.
- The system's own limit (`Session.max_jobs`, `Session.admits`) comes on top and only lowers it.
- Two cases of one group never run at once, in any configuration.
- At most 4 execs run at once in all, because each needs a working folder of its own and the deny words are made for four.

**The GPU runs alone.** Before a cell whose system needs the GPU starts, the run refuses if another program is computing on
the GPU (`MMSB_GPU_BESIDE` names process ids the user lets it share with). If the GPU is lost during an exec, the result is
a `setup` problem and is asked again.

**The order.** Without `--side-by-side` the cells of a run run one after another: the configurations with no variant first
(`blind`, `files`, `listing`), for each agent, then each system's variants in turn. Inside a cell, cases start in the order
of their `until`.

**`--side-by-side`** (off by default) gives each cell a thread, and the threads share three pools (`mmsb/core/schedule.py`):

| Pool | Size | Rule |
| --- | --- | --- |
| The GPU | 1 cell | a cell that needs the GPU holds it for its whole length, its session's `open` and `close` included; cells take it in their sequential order |
| Execs | 4 in all | across every cell and both agents |
| Each agent's sessions | 4 | quota and rate limits |

- Cells that need no GPU run beside the one that holds it.
- A free slot goes to the cell that needs the GPU first, then to the earlier cell in the sequential order.
- In a cell whose system does not cut its index to each case's horizon, pending cases start longest first, by the medians
  of earlier runs. A cell whose system cuts its index keeps the order of `until`, because it reads in time order.
- The dry run simulates the same policy and prints its estimate.

**Agents stay off the GPU.** An agent decoding on the GPU would be a second GPU user. An agent with its own tools is given
`CUDA_VISIBLE_DEVICES=-1`, and `-hwaccel` and `nvidia-smi` join its deny words, when its own system needs the GPU
(`mm+files` has a shell and a GPU job in one cell) or when the run is side by side.

### 8.4 The clock and timeouts

**Working time, not wall time.** When several cases share one system, a case's wait for it is an artefact of the
benchmark: a real user's system has the GPU to itself. So an exec is charged its own turns plus its own service time, never
its time in the system's queue (`mmsb/core/clock.py`).

```
working = wall - queue        over when working >= limit, or when wall >= 3 x limit, whatever the queue
```

- **The limit** is `--timeout` (900 s unless the run says other), kept in the run. The prompt tells the agent the seconds.
- **The queue** is what the exec's system reports through `Session.queue_s`, read at least once a second while the exec
  runs, so a queue that grows moves the end of the limit.
- **The wall cap,** three times the limit, ends an exec that waits for ever.
- **No queue figure.** A system whose `queue_s` is `None` leaves the clock on the wall, and the result says `clock: wall`.
  That is exact when nothing else shares the system.
- **The refusal.** More than one exec at once on the working clock, with a system that gives no queue figure, is refused
  before anything is spent: each exec would be charged its wait for the others. The message names one exec at a time.
- **`--clock working|wall`** on `run` and `repeat`, default `working`. `wall` charges wall time throughout.
- **Result fields.** `seconds` is always the wall time. `clock`, `working_s` and `queue_s` say what the limit was measured
  on.
- A timeout is a wrong answer: `timeout: no answer in 900 s`, with the queued seconds when there were any.

### 8.5 How an exec ends

| The exec's problem | Meaning | What the run does |
| --- | --- | --- |
| none | it answered | graded |
| `escaped` | a tool outside the mode, a deny word, or the data changed | a wrong answer |
| `timeout` | the clock ran out | a wrong answer |
| `capped` | the turn cap or the context window ran out | a wrong answer |
| `setup` | the session was not the one asked for, the system died, or the GPU was lost | asked again; the agent stops at once |
| `limit`, `auth` | the plan is out, or the login is | asked again; the agent stops at once |
| `error` | anything else | asked again; three in a row stop the agent |

- **Resume.** A run that stops for any reason continues where it left off when the same command is given again: a done
  result is never asked twice, and one to ask again is left out of every figure until it is.
- **A result** (`results/<agent>/<config>/<qid>.json`) holds the answer, `passed`, the grounded fields, the problem, the
  clock's figures, turns, tokens and cost, the tools called and how often, the agent's model and effort, what the system
  added, and the exec's folder. It holds no truth: the report grades from the question file.
- **An exec's folder** (`execs/<agent>/<config>/<qid>/`) keeps the prompt, the instructions, the session's events, the tool
  calls and the final message.

### 8.6 The cache and frozen dumps

`mmsb/core/cache.py`. What a recipe makes, and what a system ingests it into, is made once and found again.

- **A dump's key** is the sha256 of the scenario's name, its recipe and its generator as canonical JSON. The dump is
  `<temp>/mmsb-<key10>-data`. Its record is `<cache>/dumps/<key>/`: the manifest, the questions, the recipe, the seconds it
  took, the frozen file stats, and the probe's outcome.
- **A store's key** adds the system, its version and its `store_hash`. It is `<cache>/stores/<system>-<key10>/`, with
  `ingest.json` (the ingest report). A run's `stores/<system>` is a hard-link copy, so it costs no space.
- **Frozen.** A dump is made read-only and its files' stats are recorded. Every run checks them before it starts and after
  each exec of an agent with its own tools.
- **A dump that changed is an error, not a rebuild,** because it may be an escape or a failing disk, and what ran over it
  ran over other recordings. A dump that is gone is written again from the recipe and compared with the record: the same
  files, sizes and hashes, or an error.
- **Made whole or not at all.** A dump or a store is written into `<final>.part` and renamed. A `.part` left behind is
  removed, said aloud, and written again. A lock held by the operating system, released when its process ends however it
  ends, makes a second command that wants what the first is making stop at once and touch nothing.
- **What the keys do not cover.** The libraries a generator uses, and the probe's code. When a library change alters
  bytes, bump the scenario's generator or remove the cache. There is no eviction.
- **To force a rebuild,** remove `<temp>/mmsb-<key10>-data` and `<cache>/dumps/<key>`, and the stores if the bytes changed.

### 8.7 The run folder

`mmsb build` names it `<date>-<time>-<id>-<scale>-s<seed>` in the runs folder.

```
<run>/run.json              what the run is (below); written last, so a folder without it is a build that failed
      questions.jsonl       the questions, with their answers
      dump.json             the dump's entries
      data-manifest.json    the frozen stats of the dump's files
      shortcuts.md          the probe's findings
      stores/<system>/      each system's store, with ingest.json
      systems.json          what each configuration was asked, and its system's description
      results/<agent>/<config>/<qid>.json
      execs/<agent>/<config>/<qid>/
      results/retrieval/<config>/<qid>.json
      homes/                a system's working copies for each agent
      report.md  report.json
```

**`run.json`** holds the scenario and its generator, the recipe, the seed, scale and days, the dump's key, where the data,
temp and cache folders are, the timeout, the configurations, the hash of `questions.jsonl`, the hashes of the plan's
questions and files, the probe's outcome, when it was made, the benchmark's commit, and the seconds the build took.
`mmsb run` adds the answer format and each agent's model and effort.

**Opened again, a run is checked.** It is refused when the scenario's generator has changed, when `questions.jsonl` is not
the file `run.json` recorded, or when the plan or files the recipe makes now are not those it was built with. Its results
would otherwise be graded against other questions.

### 8.8 Runs without agents, and run against run

- **`mmsb retrieve RUN`** asks each system that declares an oracle for every needle, with no agent: the yes/no of the
  question's `retrieval` spec, over its sources and ranges. The ranked moments are scored against the truth within the
  spec's tolerance: found, in the top 5, the needle's best score, its rank, false sightings, and why a miss missed (`near`,
  `stopped`, `cut`, `missed`). It needs no quota and separates a system's search from the agent's use of it. It holds the
  GPU alone.
- **`mmsb compare A B`** sets two runs of the same questions side by side, cell by cell: the right counts, the questions
  whose verdict flipped with both answers, McNemar's p, and each run's problems. It says whether every cell's count is
  within 2 of the other run's and no p is under 0.05. That means no large loss, not that the runs agree: agents are
  stochastic, so the flips are printed to be read. Cells made by another model or effort are refused unless
  `--across-settings`.
- **`mmsb repeat RUN`** makes an earlier run again as it was made: the same recipe, the same agents at the models and
  efforts its results name, the same configurations and questions. It builds or finds the dump and stores, runs, and
  prints the `compare` command. `--dry-run` builds and starts nothing.
- **`mmsb clock-check RUN --config CONFIG`** tries the working clock before it is trusted. It asks a seeded sample of
  cases (10 unless `--cases`) of each agent twice, each alone and then all queued at once, and prints the answers, tokens
  and charged seconds of the two halves side by side. If the clock is right, a case is charged about the same working
  seconds either way while its wall seconds grow. It refuses, before any agent starts, a system that reports no queue or
  that would run the cases one at a time.

## 9. Costs

| Cost | Where it comes from | In the report |
| --- | --- | --- |
| Seconds | the exec's wall time (`seconds`), with `working_s` and `queue_s` beside it | seconds for each right answer; timeouts |
| Tokens | the agent's own count: input and output, and for Claude the cache it read and wrote | thousands of tokens for each right answer |
| Dollars | Claude reports a cost at list price; Codex reports none | total, and dollars for each right answer |
| The system's GPU seconds | the session's `audit` of the exec's calls, kept in the result's `work` | GPU seconds for each right answer in `v7_screen`'s and `v8_egolife`'s reports; the summary does not show them |
| Ingest | `ingest.json` of each store: seconds, GPU seconds, bytes written, files refused and misdated | printed by `mmsb build` and `mmsb ingest` |

- **Lower bounds.** An exec that is stopped may report no tokens or cost, so a cell's totals are lower bounds when it has
  timeouts.
- **For each right answer,** not for each question: a configuration that answers quickly and wrongly is not cheap.
- **The plan's use.** Each result records the share of the plan's week used, which is what stops a run at 95%.

## 10. Statistics and the report

**The statistics** (`mmsb/core/stats.py`):

- **Wilson's 95% interval** on every proportion right. Under 30 questions the report notes that the interval is wide.
- **Paired, question by question.** Each configuration is compared with the control (`files`, or `--control`) for each
  agent, over the questions both have a done result for. `b` counts the questions right only in the configuration, `c`
  those right only in the control.
- **McNemar's exact two-sided test** on `b` and `c`.
- **When a difference is shown.** The report calls a configuration better or worse only when McNemar's p is under 0.05.
  Otherwise it says "not shown (too few questions, or no difference)".
- **An interval of the difference** (Newcombe's hybrid score interval for paired proportions) is given only from 10
  discordant pairs. With fewer, an interval that excludes 0 is no sign.
- **Pooling over seeds.** Several runs of one scenario with other seeds ask other questions and are pooled. Two runs of
  the same scenario, seed and scale ask the same questions and are refused: pooled, each question would count twice.
- **Distinct questions, not repeats.** Repeating a question shrinks only its own run-to-run variance, never the spread
  between questions (Miller, "Adding Error Bars to Evals").

**How many questions a verdict needs.** Paired questions for each agent and comparison, at 80% power and α = 0.05 (Connor's
approximation for McNemar):

| Effect, and share of pairs right in one configuration only | Questions |
| --- | --- |
| +17 points, every discordant pair one way | about 45 |
| +15 points, 20% discordant | about 70 |
| +10 points, 15-20% discordant | about 115-155 |
| +4 points, 8% discordant | about 390 |

So a run that wants a verdict selects 70 to 150 questions across scenarios and aims at effects of 10 points or more. A
smaller effect, or a smaller selection, is reported as not shown, never implied.

**The report** (`mmsb report RUN [RUN ...]`, written as `report.md` and `report.json` in the first run's folder). The
summary (`mmsb/core/summary.py`) names no system and no scenario. For each scenario:

- **What each agent got right** in each configuration: right out of n with its interval; grounded, lucky and unparsed
  answers; absent questions answered; "unknown" on a real one; seconds, tokens and dollars for each right answer; the mean
  item F1 of the find-every-one questions; timeouts.
- **Each configuration beside the control:** the paired count, the difference, `b` and `c`, McNemar's p, and the verdict.
- **By kind and by category** of question.
- **Answerable from metadata:** the questions a leak filter answered, left out of every other figure (7.2).
- **To run again:** results that are not done, left out.
- **Every question,** with each cell's verdict.
- **Over all the scenarios,** when the runs are of several: right out of n and its interval for each agent and
  configuration.
- The header gives each run's scale, seed, commit and probe outcome.

The scenarios of real recordings report with the summary. `v7_screen` and `v8_egolife` have reports of their own with the
same statistics: the first also cuts by how vague the question's time hint is and adds the retrieval figures, the second
sets `blind`, `files` and a system side by side, by the question's age and category.

## 11. Related benchmarks

| Benchmark | Close because | Scale |
| --- | --- | --- |
| [LVBench](https://arxiv.org/abs/2406.08035), [HourVideo](https://arxiv.org/abs/2411.04998), [EgoLifeQA](https://arxiv.org/abs/2503.03803), [EgoMonth](https://arxiv.org/abs/2608.13113) | hour-to-month video QA | 117 h / 500 videos / 266 h / 300+ h |
| [Ego-R1 Bench](https://arxiv.org/abs/2506.13654) | an agent with search tools over a week of video | 300 questions, ~44 h each |
| [M3-Bench](https://arxiv.org/abs/2508.09736) | memory agents over long robot-view video; questions asked at a time | 100 scripted videos + 920 web; 4,490 questions |
| [LongVidSearch](https://arxiv.org/abs/2603.14468), [Deep Video Discovery](https://arxiv.org/abs/2505.18079) | agentic search, scored with its tool cost | 3,000 questions / LVBench |
| [camroll](https://arxiv.org/abs/2606.05275), [DISBench](https://arxiv.org/abs/2602.10809), [ATM-Bench](https://arxiv.org/abs/2603.01990), [PhotoBench](https://arxiv.org/abs/2603.01493) | a real person's camera roll, with evidence | 31.5K / 109K / 3.8K photos + 533 videos / 3.6K photos |
| [BLAB](https://arxiv.org/abs/2505.03054) | long audio, timestamps scored by F1 | 833 h, files up to 2 h |
| [Visual Haystacks](https://arxiv.org/abs/2407.13766) | accuracy against the size of the pile, 1 to 10K images | ~1K questions |
| [Lifelog Search Challenge](https://arxiv.org/abs/2506.06743), [Video Browser Showdown](https://videobrowsershowdown.org/about-vbs/) | timed known-item search over months of images and thousands of hours of video, many systems on one task | 725K images / ~2,300-3,800 h |
| [MIRIX ScreenshotVQA](https://arxiv.org/abs/2507.07957) | memory of one's own screen, owner-written questions | 5-18K screenshots per person |

The numbers are the papers' own.

**What none of them does, and MMSB does:**
- hand an agent a real pile at real size and in real mess;
- cover many small audio files as well as long ones;
- compare search systems behind one plug-in contract, each against the agent's own tools on the same files;
- report the ingest wait and GPU seconds beside tokens and dollars.

The Lifelog Search Challenge and the Video Browser Showdown compare many systems on one task, but with people at the
controls, not agents.

**Ideas taken:**

| Idea | From | In MMSB |
| --- | --- | --- |
| Grounded scoring: right only if the cited file and span hold the answer | CG-Bench, Ego4D NLQ, ATM-Bench | 6.4 |
| Absent questions, about a quarter | PhotoBench, MMNeedle, Deep Video Discovery | 5.4 |
| Leak filters: no media, or metadata only | HourVideo, LVBench, Daily-Omni | `blind`, `listing`, 7.2 |
| Find-all scored by distinct moments with a collar | DCASE, Video Browser Showdown | 6.4 |
| "As-of" questions: each has a now, so one dump serves many questions | TeleEgo, M3-Bench, EgoLifeQA | `now`, `horizon`, 6.1 |
| Score retrieval apart from the agent | DCASE PSDS, LV-Haystack | `oracle`, 8.8 |
| One plug-in contract, many systems, one scorer | Lifelog Search Challenge, Video Browser Showdown, TRECVID | 3.1 |
| Working-time limits, error classes, resume | Inspect AI, Harbor, SWE-bench | 8.4, 8.5 |
| Accuracy reported with its cost | AI Agents That Matter, HAL | 9 |

**Taken as warnings.** Benchmarks that build long audio or video by concatenating clips leave seams that are an artificial
shortcut. MMSB keeps a pile's files separate and as recorded.

## 12. Not built yet

Designed, and not in the code:

- **A `gold` configuration:** the agent given the evidence clips alone, so that a miss there is the agent's reading and not
  the search. `gold` is refused by name.
- **More search systems.** `mm` is the only one. An embedding index (frames, photos and sound windows), a transcript search
  and a hosted long-context model are the obvious next ones, each a folder under `mmsb/systems/`. Hosted systems would
  report API dollars for each case and run only on public scenarios.
- **A check of the instructions' size.** Nothing enforces the limit of 3.1, and `mm`'s skill is over it.
- **Rotated options for multiple-choice questions,** counted right only if every rotation is. Without it a leak filter
  passes about a quarter of them by chance, so `listing` is not meaningful on `v8_egolife`.
- **Sharing the GPU by lease:** a system asking the runner for the GPU around each piece of work, so the runner measures
  queue time itself. Today a cell holds the GPU whole, and queue time is what the system reports.
- **A queue figure from `mm`,** which would let several of its execs run at once on the working clock. `mmsb clock-check`
  has not been run on a system that reports one.
- **Time budgets sized to the pile** (300 s for an hour of media up to 900 s for a day). The function exists
  (`clock.budget_s`) and no scenario uses it; every exec has the run's one timeout.
- **A size ladder:** the same needles in smaller and larger piles of one source, to show from what size a system pays off.
  `--rung` selects by a `rung` tag, which only some questions carry.
- **A large-file track,** and scenarios of a person's own private media or of scripted recordings.
- **Query by example:** `attachments` is in the question file and always empty.
- **More of the report:** ingest cost amortised over 1, 10 and 100 questions, a time-to-correct curve, accuracy against
  cost as a Pareto plot, and the count of searches an absent question drew.
- **More of the statistics:** Holm's correction over the pre-registered pairs, bootstrap intervals for rankings between
  systems, a reliability set of repeated questions, and sequential stopping.
- **A regrade command.** `mmsb report` reads each result's recorded verdict; nothing grades stored answers again after a
  rule changes.
- **Session mode:** several questions about one pile in one agent session.
- **A guard for two cells sharing a system's working copy** under `--side-by-side`. A system that needs the GPU is
  serialised by the GPU pool; one that needs none and keeps state for each agent is not.
