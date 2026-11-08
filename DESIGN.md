# MultiMediaSearchBench (MMSB): design

Status: built (6 November 2026). [VISION.md](VISION.md) says what MMSB is for; this file says how. MMSB is a standalone
benchmark of real piles of media: four scenarios, one for each media shape, built from real recordings with every answer
taken from the source's own labels (5.7), on a shared builder (7.2), graded by fixed rules (6.1) and reported by a neutral
report (6.3). The benchmark first copied from MultimediaSearch, then called Multimem (P0, 5 November) is `mmsb` too: the V7 and V8 scenarios and the `mm`
system stay as extras, and the core runs with MultimediaSearch not installed. Not done: full runs of the four scenarios with real
agents, the pilot re-run, the first `mm+files` run, and the check of the working-time clock (section 12). A section says
where it describes what later work will build, and every estimate is marked as one. Section 8.1 is measured.
[TESTING.md](TESTING.md) says how the build was checked.

## 1. Overview

**The task:** a user hands an AI agent a large dump of their own media and asks for something in it ("when was the parcel
taken?", "find the voice memo where I said the gate code"). A **search system** is anything that ingests a dump and gives
the agent tools to search it:
- a local model that scans time windows (MultimediaSearch);
- an embedding index;
- a transcript search;
- a long-context model.

The baseline is the agent alone with the files and its own tools.

**A case** is one question about one dump. **A run** is a chosen set of cases, asked of chosen agents in chosen
configurations (section 4). There are no tiers. A quick run is a run with few cases (section 8.2).

**What a run measures, per case and configuration:**

| Measure | Why |
| --- | --- |
| Grounded accuracy: the answer is right and the evidence it cites (file and time) holds it | a right answer with wrong evidence is a guess |
| Seconds to answer | the user waits for it |
| Agent tokens and dollars | the user pays for them |
| System cost per case: GPU seconds, API dollars | local GPU time is free in money, not in time or heat |
| Ingest cost per dump: seconds, GPU seconds, disk, files refused or misdated | the user pays it once per dump, before the first answer |

## 2. How it is built

Six layers. Only the last knows any system.

| Layer | Holds | Knows about systems? |
| --- | --- | --- |
| **Dumps** | folders of real files as a user would hand them over, with a manifest of hashes and true start times | no |
| **Questions** | one JSONL file per scenario: the question, its "now", the truth with evidence spans, the grading rule, tags (section 6) | no |
| **Grader** | fixed rules; no model grades another | no |
| **Agents** | locked-down Claude Code and Codex sessions, one per case and configuration | only through the tools a system hands over |
| **Runner** | case selection, scheduling, resource pools, budgets, resume, statistics, report | only through the contract |
| **Systems** | one plug-in each: `files`, `listing`, `mm` built; `embed`, `transcript`, ... in P1 | each knows itself |

**The layout as built:**

```
MultiMediaSearchBench/
  VISION.md  DESIGN.md  TESTING.md  README.md  MEDIA.md  pyproject.toml  uv.lock  .python-version
  mmsb/cli.py       build, ingest, run, retrieve, report, compare, repeat, clock-check: the order of each command's steps
  mmsb/arguments.py   the commands' names, options and defaults
  mmsb/core/        the neutral half; one module for each part of the runner
    contract.py config.py       the types a system and the runner exchange (3.1); a configuration's name (4)
    question.py grade.py        the question file (6.1) and the value rules
    answer.py grounded.py       the final answer and its grounded score (6.2, 6.3)
    scenario.py build.py cache.py runfolder.py     a scenario's types; making a run; the dump and store cache (8.4); a run's folder
    runner.py schedule.py select.py costing.py dryrun.py clock.py clockcheck.py     the exec loop; the pools (8.3); choosing cases (8.2); working time (8.4)
    record.py report.py render.py compare.py repeat.py retrieval.py stats.py     results, figures and Markdown; two runs side by side; a run
                                again; the retrieval run (8.5); statistics
    summary.py                  the neutral report of the real scenarios (6.3)
    fs.py paths.py gpu.py probe.py     files and permissions; where things live; the GPU check; the probe's outcome
    agents/     spec.py (one exec as the agents get it), common.py, claude.py, codex.py
  mmsb/systems/     a registry that reads every system.toml; files/  listing/  mm/       # one folder per plug-in
  mmsb/scenarios/   a registry; realdump/ (the real scenarios' shared builder); v1_security/  i3_trailcam/  a1_home_audio/
                    m4_samples/; the extras v7_screen/  v8_egolife/                     # recipe, builder, questions, prompts
  tests/            unit tests; oracles/ (the recorded goldens and the code that made them); mm/ (needs the mm extra)
  results/          <date>-<selection>/                                                 # published reports
  media/            sources.jsonl (MEDIA.md)
```

Each system's dependencies are an optional extra of `pyproject.toml` (`mmsb[mm]`). The `mm` extra is `mms`, MultimediaSearch, a git source
pinned to one commit (D4).

**Where things live (correction C4).** Downloads stay in `D:\bench-data`, fetched once (V8 reads `D:\bench-data\egolife`).
A prepared dump cannot be there, because V8 denies that folder to its agents. A prepared dump is `D:\mmsb-data\mmb-<key10>-data`,
read-only; its record and each system's store are in `D:\mmsb-cache`; a run's folder is in `D:\mmsb-runs`. The three are
changed with `--temp`, `--cache` and `--out`. They must share a volume, because a store is copied into a run with hard
links, and the commands refuse folders on two drives. `D:\mmsb-data` may hold only `mmb-*` folders (`check_temp`), because an
agent with its own tools can list it. The tests keep theirs in `D:\mmsb-temp`. The runs recorded before MMSB (`D:\mm-bench`,
`D:\mm-bench-temp`) hold the pilot's answers, so they are on every agent's deny list, as are this repository and its main
checkout.

**The neutrality check (as built, `tests/test_neutral.py`).** The core, the `files` and `listing` systems and every
scenario install and run on a machine where MultimediaSearch has never been installed.
- Every run, fast: a subprocess whose import system refuses `mms`, the tests' helpers and the old root modules imports
  every module under `mmsb.core`, `mmsb.scenarios` and each system but `mm`, reads mm's `system.toml`, and sees opening `mm`
  give "mm needs the mm extra: uv sync --extra mm".
- Every run, two scans of the source: no import of `mms` outside `mmsb/systems/mm` and `tests/mm`; and no import or
  name of a system inside `mmsb/core` or `mmsb/scenarios`, apart from a short list that gives its reason for each
  (`files` as the name of a configuration, a text anchor of PIL that is spelled `mm`). Words in text are not scanned:
  the report's text still quotes the recorded pilot's words about mm.
- Slow (`-m slow`): a fresh environment made with `uv sync --frozen --no-default-groups` builds V7 tiny, runs its probe,
  runs a `files` cell with a stand-in agent, writes a report and sees `mm` refused.

## 3. Systems

### 3.1 The contract

A system is a folder under `mmsb/systems/` with a `system.toml` and a class. The registry reads every `system.toml` as
TOML and imports a class only when a configuration names that system. That is how the core knows what every agent shell
must keep out (`[isolation]`: environment prefixes to drop, words no command may hold, the repositories of installed
packages, variables that lead nowhere) while the system itself is absent. A system whose extra is not installed is refused
when it is named, with the extra's name. There is no fallback.

**Correction C2.** A daemon and a model load span many cases, so a system cannot be a set of per-case calls. The `System`
builds stores, and `open` returns a `Session` that lives across the cases of one agent and configuration. `serve`, `reset`,
`usage` and `oracle` are the session's. The types are in `mmsb/core/contract.py`.

| `System` operation | Does | Rules |
| --- | --- | --- |
| `describe(variant)` | a `Description`: name, version, variant, the media it reads (image, video, audio, text, records), local, needs a GPU and how much memory, network, licence, `config_hash`, `store_hash`, horizon, whether it has an oracle, whether it is a leak filter | the version and `config_hash` are recorded with every result, and a configuration that has results is refused when its `config_hash` changes. `store_hash` covers only what ingest depends on, so the variants of one system share a store. It needs no readiness |
| `ingest(dump, store)` | builds the system's own state in `store` from a read-only dump | sees the dump only, never the questions; timed; returns seconds, GPU seconds, bytes written, each file refused with the reason, each file stored at a time other than its own (`misdated`), and figures it measured. A misdated file refuses every run over the store |
| `open(store, dump, context)` | a `Session` over a copy of the store | one session for each agent and configuration; `close()` ends it |
| `check_ready(variant)` | raises, saying why, when `open` would fail now (weights missing, an environment not installed); starts nothing | asked of every configuration before the first agent starts, so a run does not spend its first cells and then fail at the next system's |
| `preview(store, dump, context)` | a session that starts nothing and changes nothing, for `--dry-run` and the estimates printed before a run | answers `max_jobs`, `estimate` and `serve` from the store's files |

| `Session` operation | Does | Rules |
| --- | --- | --- |
| `serve(case)` | the `Served`: the agent's tools (an MCP server's command, args, environment and tool names), the system's own instructions, and fields to add to the result | a `Case` never holds the question's text or truth. Only media before the case's horizon may be answered from. A system that cuts its own index to the horizon declares `horizon = "cut"`. One that cannot declares `"view"`, and the scenario hands the agent a folder cut to the horizon (V8 does this for `files`). Cutting a store's ingest per distinct `now` is not built |
| `max_jobs(cases, requested)`, `admits(case, busy)` | how many cases run at once, and whether one may start beside those running | the system's own limits (mm: one at a time when any case has a horizon or the model is slow; no second version-sensitive case at once; a heavy case alone). The runner adds two rules for every system: two cases of one group never run at once, and `requested` is `--jobs`, or `--gpu-jobs` (1 unless asked) for a system with `needs_gpu` (8.4) |
| `estimate(sources, ranges)` | a pure cost function: GPU seconds of a range | asked about the question's `cost_ranges`, never its truth |
| `audit(case, calls)`, `reset(case, calls, others)` | fields for the result from the exec's calls; then the exec's state is taken out of the store | `audit` may raise to end the command (V8's horizon leak) |
| `usage(case, calls)`, `queue_s(case)` | GPU seconds, queue seconds and API dollars of this case, not "since the last call": cases overlap; the seconds this case has waited so far | `queue_s` is read live by the working clock (8.4); `None` means the system cannot say. mm answers `None`: it keeps no per-job wait. The runner does not call `usage` yet |
| `oracle(query)` | optional: the system's ranked moments for a plain query, with no agent | for the retrieval run (section 8.5), which compares systems without paid agent sessions |
| `close()` | stops the daemon; returns named lists, each written as `results/<agent>/<config>/<name>.json` when not empty | mm returns `ambient-labels`, the labels no case owns |

A session with nothing to do (`PlainSession`) serves no server and costs nothing; `files` and `blind` use it.

**Rules every system follows:**
- **Ingest is blind.** It runs before any question is known. It may take as long as it likes, and the time is reported
  as the user's wait.
- **Systems are frozen before a test run.** Their version and configuration hash go with each result. Tuning happens on
  the dev split only (section 6.4).
- **Instructions are general.** A system's instructions text is at most about 2,000 tokens and the same for every
  scenario. It may teach the agent its tools, never what is in a scenario. (mm's skill and notes are about 26,000
  characters, over this limit; no check enforces it yet.)
- **No network,** unless the system declares itself hosted. Hosted systems run only on public scenarios, never on
  private ones, and their API dollars are reported.
- **GPU work goes through the runner.**
  - In `exclusive` mode (the default, and the only one built), the scheduler gives the GPU to one system at a time.
  - In `lease` mode, the system asks the runner for the GPU around each piece of work, through a small local endpoint named
    in its environment, and the runner knows queue time exactly. Not built: nothing needs it before `embed`.
  - The runner is meant to cross-check a system's GPU seconds against its own sampling of the GPU (C13). That waits for P1,
    when a second GPU system exists to compare.

### 3.2 The systems to ship

A benchmark with one contestant only tests that contestant. MMSB ships common, cheap baselines beside mm:

| System | What it is | Tools it gives the agent | Ingest |
| --- | --- | --- | --- |
| `files` | the baseline: no system | none beyond the agent's own (shell, file reads, ffmpeg) | none |
| `listing` | the floor and the leak filter (section 7.3) | a listing of names, sizes, times, EXIF and container fields; no pixels or sound | seconds |
| `embed` | the common "multimodal RAG" setup | `search(text or picture, since, until, k)` over frames at 1 fps and photos (SigLIP or CLIP) and 5 s sound windows (CLAP); `frame(file, t)` | GPU: every frame and window once |
| `transcript` | speech made searchable | `search(words, since, until)` over a timestamped transcript (Whisper) | GPU: all sound once |
| `mm` | MultimediaSearch (`mms`): a small local model reads every window in a range against the agent's question, at question time | `find`, `ask`, `look`, `label`, and mm's skill | CPU only; the GPU works per question |
| hosted (later) | a long-context model given the dump in chunks | `ask(question, files)` | none; API dollars per case |

`embed` and `transcript` pay the GPU once at ingest, and `mm` pays it on every case. The report will show both, amortised
over 1, 10 and 100 questions (not built), so a user can see which is cheaper for their use.

P0 builds `files`, `listing` and `mm`. `embed`, `transcript` and the hosted system are P1 or later.

**`listing` as built.**
- `ingest` writes `store/listing.jsonl`, a row for each file: path, source, size, `mtime`, the EXIF tags Pillow reads, and
  PyAV's container fields (format, duration, width, height, fps, codec, creation time, sample rate). It reads what any tool
  on the files could read, and copies nothing the manifest knows and a file does not (`t`, `kind`, `sha256`).
- Its server is a stdio MCP server with two tools, `list_files(source, since, until, limit, after)` and `file_info(path)`.
  The agent has no other tool and no path to the dump.
- Its only time is the file's `mtime`: `since`, `until`, the order and the horizon (`--until`) are all about it. That is
  right for V7, whose generator sets each file to the time it holds (`os.utime(path, (t, t))`; the recorded pilot's 1,572 files
  are exactly at it). V8's `mtime` is the download's (3 November 2026, against recordings of 29 to 31 October), so the horizon
  cut every clip from every question and the agent got an empty listing. So `ingest` compares each file's `mtime` with the
  manifest's `t`, read for that check only and never listed or served, and reports a file more than one second off as
  `misdated`; every command that would use the store then refuses the run, naming the files and both times. `listing` on V8
  is refused at the build or at `mmsb ingest`, before any agent. One second is the tolerance because it is mm's own for the time
  it stored a file at, and covers a volume that keeps whole seconds. A log is only required not to be before its first line,
  because its `mtime` is its last line's. A cached listing store is valid only for a dump with the same `mtime`s.
- It reads all five media, so every question is asked of it. It is a filter for leaks, not a reader (7.3).

Each model inside a system is a variant (`mm/d1-3B`, `mm/d1-Omni-2B`, `embed/siglip2-base`), and every variant is
reported apart.

## 4. Configurations and agents

Each case runs in a fresh agent session per configuration:

| Configuration | Written as | The agent has | Role |
| --- | --- | --- | --- |
| `blind` | `blind` | nothing: no tools, no recordings | the floor of what the model already knows of a public dataset; V8's report uses it. Not in the first design: it costs a few lines and the recorded EgoLife report needs it |
| `listing` | `listing` | the listing only | the floor; a case it answers is flagged as answerable from metadata or prior knowledge |
| `files` | `files` | the files and its own tools | the baseline |
| `S+files` | `mm+files` (`mm/d1-3B+files`) | its own tools and system S's tools | **the headline for each system**: what a user who installs S has |
| `S` | `mm` (`mm/d1-3B`) | S's tools only | diagnostic, on request: what S's tools do alone. It is the configuration the recorded pilot ran |
| `gold` | `gold` | the evidence clips alone | diagnostic, on request: a miss here is the agent's reading, not the search. Refused: "gold is not built yet" |

**How a configuration is written.** `blind`, or `<system>[/<variant>][+files]`, with the system's default variant filled
in. Its results are in `results/<agent>/<dir>/` and its execs in `execs/<agent>/<dir>/<qid>/`, where `<dir>` is `blind`,
`files`, `listing`, `mm-d1-3B` or `mm-d1-3B+files`. `listing+files` is an ordinary configuration. Two configurations of one
run may not share a `<dir>`.

**Which questions a configuration is asked.** One with no tools of its own is asked only the questions on media its system
reads (`Description.reads`): d1-3B does not hear, so `mm` was asked 24 of the pilot's 31. One with its own tools is asked
all of them, so `mm+files` gets 31: its agent has ffmpeg for the sound. `blind` is asked all. Of the configurations that
read the text and records questions, all are asked them by default; `--controls CONFIG` gives them to that one only, as the
copied code did.

**Agents:** Claude Code and Codex for now.
- Each runs headless and locked down: no network, no access outside its run, none of the owner's skills or plugins.
- Model, effort and turn cap are the same in every configuration. Claude is `sonnet` at `high` effort with a cap of 60
  turns; Codex is `gpt-6.1-terra` at `high`, and the pilot ran it at `medium`. The environment (`MMSB_CLAUDE_MODEL`,
  `MMSB_CLAUDE_EFFORT`, `MMSB_CODEX_MODEL`, `MMSB_CODEX_EFFORT`) and the flags of `mmsb run` set them; `mmsb repeat` uses what
  the recorded run's results say.
- The prompt skeleton is shared: the user's question, the date and time "now", where the dump is, and the answer format
  (section 6.2). A system adds its own instructions and tool schemas. The agent builds its command line from what the
  system `serve`d and whether it has its own tools, and nothing in the agents names a system.
- Other agents can be added through the same runner.

The headline compares `files` with `S+files` for each system S. Taking the agent's shell away is not what any user does,
so it is a diagnostic only. The `S+files` prompt was never run by the pilot and is pinned by a test of its literal text.

## 5. Scenarios

Each scenario is one dump and 15-40 questions about it. All its cases share the dump, so it is built, probed and ingested
once per system. The questions mix the kinds in section 5.6. The tables give one example question each.

[MEDIA.md](MEDIA.md) says where each scenario's media comes from, what is already downloaded, and what is still missing
(4 November 2026). Licences are recorded there but are not a filter (section 11.3).

P0 built V7 and V8 only; V1, I3, A1 and M4 followed on 6 November (5.7). A scenario is a module under `mmsb/scenarios/` named
so that it can be imported (C6): `v1_security` and `v7_screen`, not `V1-security`. Its questions keep the ID of the tables below (`Question.scenario` is `V7`, `V8`).
A scenario is the functions of the `Scenario` protocol (`mmsb/core/scenario.py`): `recipe`, `hashes`, `build`, `probe`,
`prompt`, `view`, `deny_words`, `after_exec`, `release`, `report`, and a line in the registry. It never knows a system.
Agents cannot hear, so the truth of every sound question comes from a mixing list, a script or a dataset's labels, never
from an agent listening. Speech-heavy scenarios (A3, M1-M3) are reported apart as well as in the totals, since some
systems do not handle speech at all.

### 5.1 Big video

| ID | Scenario | The user asks | The dump | Hard because | Built from | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | Doorbell or security export | "My parcel vanished Tuesday. When was it taken, and what was the person wearing?" | 3 cameras × 48 h, 1-hour or 24-hour MP4s, 4-8 GB each | night and infrared, many passers-by, the same event on two cameras | MEVA | 1 |
| V2 | Dashcam road trip | "Where did the cyclist cut in front of me?" "What exit did we take for fuel?" | 8 h in 4 GB GoPro-style chapters, the camera clock set wrong | everything moves; a sign is readable for a second; "what time" means the true time | a real daily dashcam archive with a per-second GPS track (Bristol VA); comma2k19 (GPS and CAN); no event labels | 3 |
| V3 | Kid's recital or match | "Find my daughter's solo." "Every time number 7 touched the ball." | one 3-hour 4K phone video, over 10 GB | one huge file; small faces | full-length recordings from the stands (YouTube); SoccerTrack v2 (gated on Hugging Face; a public Drive folder is untested) | 3 |
| V4 | Lecture, meeting or conference recording | "When did the speaker show the attention diagram?" "Who drew on the whiteboard?" | a 6-hour livestream; or 20-40 h of meetings | slides look alike; needs reading | AMI Meeting Corpus; CC-licensed lectures | 2 |
| V5 | Lab or robot camera | "Find the run where the arm dropped the cup." | 300 episodes of 2-3 cameras, or one 8-hour lab camera | near-identical runs; only the outcome differs | DROID | 1 |
| V6 | Bird-feeder or wildlife camera | "When did the woodpecker visit?" | 12 h from a fixed camera | long still stretches, a tiny subject | a feeder cam's daily 12-hour archive with the project's species logs (Birds Untold) | 3 |
| V7 | Developer screen | "When did the borrow-checker error first appear?" | days of screen recording in 1-minute files | text-sized changes among look-alike decoys | the copied synthetic week; AgentNet's screen recordings | kept |
| V8 | Egocentric days | EgoLifeQA's questions, as open answers where its labels allow | 3 days of smart-glasses video | many hours; much of it speech | EgoLife | kept |

### 5.2 Many images

| ID | Scenario | The user asks | The dump | Hard because | Built from | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | Phone camera roll | "Find the parking-level sign at the airport." "The receipt for the TV." | 10-30k photos over years: HEIC, bursts, WhatsApp copies without dates | small text, near-duplicates, wrong dates | ATM-Bench (one person's 3,759 photos, 1,013 questions in their voice by annotators); YFCC100M photostreams; DISBench (57 users, 122 verified queries); Dan's phone (private) | 1 |
| I2 | Screenshots folder | "Find the screenshot with the wifi password." | 5k screenshots | they all look alike; needs reading | Android in the Wild and Rico screenshots; Dan's own (private) | 2 |
| I3 | Trail camera SD card | "Which nights did deer come? Was there ever a bear?" | 20k triggers, 90% empty from wind | empty frames; night infrared; animals at the edge | Caltech Camera Traps (LILA BC) | 1 |
| I4 | Home inspection or insurance | "Show me every photo of water damage in the basement." | 2k photos | subtle; must find all of them | one damaged home's phone photos, building-defect close-ups, listing photos | 3 |
| I5 | Family archive scans | "Find grandma at the beach" (one reference photo attached) | 8k scanned prints, no dates | search by example; people age | one family's 1,376 scans (Internet Archive) with FG-NET's 82 people across ages | 3 |
| I6 | Construction time-lapse | "When was the crane taken down?" | 50k photos, one every 5 minutes | slow change | an hourly 4K building-site webcam, 2018-2021 (ANU) | 3 |

### 5.3 Long audio, one file

| ID | Scenario | The user asks | The dump | Hard because | Built from | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Home while away | "Did the smoke alarm go off? When did the dog start barking?" | 24 h from a room microphone in one file (about 2.7 GB of mono 16 kHz WAV) | the TV plays alarms too; the dog barks at everything | a real home's ambience (SINS) with real events (FSD50K) mixed in at a set SNR | 1 |
| A2 | Overnight recording | "Did I talk in my sleep? When did the baby cry?" | 8 h from a bedroom | mostly silence; faint events | a quiet-room recording with FSD50K events mixed in; or Dan's own | 2 |
| A3 | Meeting or interview | "When did Bob agree to the price?" | a 3-hour meeting; 6 h of interview tapes | it is speech | AMI Meeting Corpus | 1 |
| A4 | Field recorder | "When did the owl call? Find every frog chorus." | 3 nights outdoors | wind, rain, distant calls | three real nights of one BirdVox recorder, with owls and frogs mixed in from the annotated soundscapes and AnuraSet | 2 |
| A5 | Car or machine | "When does the rattle start?" | a 2-hour drive; 24 h of a factory floor | a small change in constant noise | MIMII machine sounds (clips only: no long base found) | 3 |

### 5.4 Many short audio files, hours in total

| ID | Scenario | The user asks | The dump | Hard because | Built from | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Voice memos | "Find the memo where I said the gate code." "Where I hummed that melody." | 400 memos over 2 years, 10 s to 10 min, M4A | speech; dates only from the files | Dan's own (private); or a scripted set Dan records | 2 |
| M2 | Messaging voice notes | "Find Mum's note with the lasagne recipe." | 1,000 Opus notes from a chat export | speech; meaningless names | a scripted set recorded by Dan or volunteers | 3 |
| M3 | Voicemails | "The voicemail from the dentist about rescheduling." | 300 calls | phone-quality sound | real business voicemails from litigation archives; FTC robocalls; answering-machine tapes | 3 |
| M4 | Sample library | "Find a door creak." "Every snare with reverb." | 10k WAVs | find-all by sound, no speech | FSD50K clips | 1 |
| M5 | Bat or bird recorder on a schedule | "Which recordings have owls in them?" | 3,000 one-minute files | many files, few hits | AnuraSet's real one-minute schedule files; or the bird soundscapes, cut on an AudioMoth-style schedule | 2 |
| M6 | Motion clips with sound | "Which clips have glass breaking?" | 2,000 thirty-second camera clips | the sound is inside video files | camera clips with FSD50K events mixed into their sound | 2 |

### 5.5 Mixed dumps, and the large-file track

| ID | Scenario | The user asks | The dump | Hard because | Built from | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | Scripted real day | "When did the courier come? Did I leave the hob on?" | a day of Dan's home on his own cameras, phone and microphone, with about 30 planted events | real light, noise and codecs | the script is the truth (M3-Bench's method); private | 2 |
| X2 | Whole phone dump | I1- and M1-style questions, plus the phone's videos | photos, videos and memos together, 40+ GB | mixed media and mixed dates | Dan's phone (private) | 2 |
| L | Large-file track | the same needles, in bigger files | one 4 GB video, one 10-hour sound file, a 50 MP panorama, a 30 GB folder | a needle a minute before the end of a huge file, or straddling a fixed cut (1 s, 5 s, 60 s); files over 2 GB; time only from the mtime | re-muxed from V1, A1, I1 | 1 |

### 5.6 Question kinds

| Kind | The user says | Answer | Graded by | From |
| --- | --- | --- | --- | --- |
| find and read | "What level did I park on at the airport?" | a value and its evidence | value rule and evidence overlap | CG-Bench |
| last seen | "When did I last have the blue backpack?" | a time and its evidence | tolerance; earlier needles are wrong answers | Ego4D NLQ |
| when | "What time did the white van arrive?" | a time | tolerance | |
| count | "How many times did the dog bark last night?" | a number, with evidence for each | exact (±1 reported apart) | |
| find all | "Show me every clip with the cat in it." | a list of moments | event F1 with a time collar | VBS ad-hoc search, DCASE |
| absent | "Did anyone open the side gate on Tuesday?" | "not there" | not there, and no evidence | PhotoBench, MMNeedle |
| query by example | "Find other photos with this bike." (a picture attached) | a list | F1 | Ego4D VQ2D, MomentSeeker |
| cross-source | "Which camera saw the man with the bag first?" | source and time | both | MEVA |
| cross-modal | "What was at the front door when the doorbell rang?" | a value | value rule, both spans | |
| order | "Did the parcel come before or after it started raining?" | before or after, with both spans | answer and both spans | |
| multi-hop | "How many times did the doorbell ring before the parcel came?" | a value | value rule, all spans | DISBench, LongVidSearch |

About a quarter of all questions are absent ones. Agents and systems both force matches.

### 5.7 The real scenarios, as built (6 November 2026)

One scenario for each media shape, each a module of the `Scenario` protocol over the shared builder (7.2): a reader of the
source and its labels, and question templates. Each has a `tiny` scale (a few files of the same pile, built in seconds, for the
tests and a smoke run) and a `standard` one. Every answer comes from the source's own labels; about a quarter of the questions
are absent, built to read exactly like a real question of their template; about 30% are `dev` by a seeded draw (6.4); `--seed`
draws other questions from the same pile. The question is asked at a fixed `now` after the pile's last recording.

| ID | Module | The dump (standard) | Questions (standard, seed 1) | Rules |
| --- | --- | --- | --- | --- |
| V1 | `v1_security` | MEVA, 12 March 2018, 10:00 to 11:15, the three visible cameras of one site: one MP4 per camera for each unbroken stretch it recorded (the recorder stopped 10:35 to 10:40), `cam1_20180312_100000-103500.mp4`, 6 files, 5.9 GB; tiny, the first ten minutes (3 files, 0.8 GB) | 22, 6 absent: when a car first turned left, or someone last closed a car door, on a camera; how many times someone closed a car door on cam1 between 10:15 and 10:30; the same of activities no camera shows, or in a window with none | `time` with the date, 30 s; `count`; `absent` |
| I3 | `i3_trailcam` | Caltech Camera Traps, location 23: the memory card, `DCIM/100RECNX/IMG_0001.JPG` on, each photo with RECONYX EXIF, 10 September 2013 to 28 February 2014, 4,256 photos, 1.9 GB; tiny, one week of December 2013 (199 photos) | 23, 6 absent: on which days in December 2013 a skunk came past (a set of dates); on how many days a pig came; when the camera first or last photographed a deer; the same of animals that never came, or in a month with none | `set` of dates; `count`; `time` with the date, 60 s; `absent` |
| A1 | `a1_home_audio` | SINS node 7, the living room of a person living alone: one WAV of 2 February 2017, midnight to midnight, one channel as recorded (its DC offset kept), `RECORD/REC_20170202_000000.wav`, 2.8 GB; tiny, 21:30 to 22:30 (115 MB) | 20, 5 absent: at what time, between two times, the vacuum cleaner was switched on or off, the TV watched, a visitor came or left, the resident left or got back; how many times; the same over a window with none | `time`, 60 s (the TV, 120 s), asked as HH:MM:SS; `count`; `absent` |
| M4 | `m4_samples` | FSD50K's evaluation split, but most clips of the sounds asked about and every clip of the sounds easily taken for them: `folder_01/rec_00001.wav` on, in 10 numbered folders, the clips as they are with no metadata, their times a seeded spread over April to September 2026 (the source has none), about 6,100 clips, 5.2 GB; tiny, 240 clips in 3 folders | 21, 5 absent: find every clip of a cat meowing (a set of file names); how many clips of someone chewing there are; the same of sounds the folder does not hold (a gong) | `set` of file names; `count`; `absent` |

**What a question avoids** (several found by the independent reviews of the build, and each checked in the scenario's tests by a
plain reader of the labels of its own):
- a label a careful annotator may have missed, or that the source marks unsure (MEVA's `unaudited` and `not_good`; a label
  within a second of a clip's edge);
- a class easily taken for one in the dump (V1's vehicle manoeuvres, reversing, U-turns and turns, are asked only where no other
  manoeuvre is within 30 s of the moment or in the window; A1's cooking, eating, dishwashing and working run into one another
  with no edge in the sound, and are not asked; I3 asks of no fox, which a night coyote in infrared could be taken for; M4
  leaves out of the folder every clip of a sound easily taken for one it asks about, such as bells and chimes beside a gong:
  `NEAR` in its questions.py, from FSD50K's rater FAQ, the review's lists and judgement);
- an animal too small to be sure of: I3 asks nothing of birds, which a labeller may miss among thousands of empty frames;
- a window whose edge cuts an event, or that spans a stop of the recorder;
- a moment the labels place loosely: A1's TV labels mark the watching, and the set's sound starts and stops up to a minute either
  side, so the TV's tolerance is 120 s;
- an answer the file list gives away: I3 asks for no first or last visit that is the card's first or last photo taken on
  motion (its other photos are time-lapse shots on the hour). The `listing` leak check runs at every run as well.

**The trail camera's midnight.** The dataset gives each photo's time with no AM or PM, so the hour after midnight reads as noon,
while the camera printed the right one on the photo. I3 reads the "A" or "P" from a fixed box of each hour-12 photo's strip
(checked on 1,213 photos, with no box between the two values; one that is neither is refused) and writes the 375 night photos
of the card at 00:xx, in EXIF, the modified time and the order of the names. It still asks no first or last question that
rests on an hour-12 photo.

**Departures from the plan.** V1 is one MP4 per unbroken stretch, not one per hour, and leaves out MEVA's infrared cameras and
its night clip, which have no official labels. A1 asks about labelled activities only (no sounds mixed in, 7.1). M4's dump is
not the whole split: see its row. Each file's modified time is its start (a real recorder leaves the time it closed the file);
the manifest's `t`, from which a citation's seconds are counted, is the start too.

**Tags on every question.** Cases can be selected by any tag, and the report cuts every measure by each:
- scenario, media shape, kind and modality;
- haystack size (hours, files) and its rung on the size ladder (section 7.5);
- certificate length (seconds of media needed to check the answer: EgoSchema);
- look-back (how long before "now" the evidence is: under 2 h, 2-6 h, 6-24 h, over 24 h);
- evidence pieces and how far apart they are;
- needs audio, needs speech, needs reading small text;
- needle salience (share of the frame, or SNR);
- how vague the time hint is (a part of a day, a day, none).

## 6. Questions, answers and grading

### 6.1 The question file

One JSON object per line, kept outside the dump and never visible to a system or an agent:

```json
{"qid": "V1-parcel-last-1", "scenario": "V1", "split": "test",
 "now": "2026-11-03T21:00:00+00:00", "question": "When was a parcel last left at the front door?",
 "attachments": [], "kind": "last", "hint": "none",
 "truth": {"value": "2026-11-02 07:30", "evidence": [{"file": "cam1/2026-11-02_07.mp4", "t0": 1788.0, "t1": 1802.0}],
           "wrong": [{"file": "cam1/2026-11-01_18.mp4", "t0": 410.0, "t1": 425.0}]},
 "grade": {"rule": "time", "tol_s": 120}, "tags": {"lookback_h": 37.5, "certificate_s": 14, "salience": 0.03, "rung": "48h"}}
```

Evidence spans are in seconds into the file. The dump's manifest gives each file's true start time, so the grader can
read a cited time either way.

**As built (`mmsb/core/question.py`).** The example is shortened. `load_questions` is the one place the file is parsed. It
refuses an unknown key, a missing key, a value of the wrong type, an unknown rule or a repeated `qid`, and everything after
it holds dataclasses. A line also has the keys the separation turned out to need:
- `options`: the choices of a multiple-choice question (V8), else null;
- `scope`: the sources and the `since` and `until` (epoch seconds) the question names;
- `group`: two cases of one group never run at once, because they would show each other the needle. V7's group is its
  template, V8's is the `qid`;
- `horizon`: the time after which nothing may be answered from (V8's), else null;
- `cost_ranges`: the ranges a system's `estimate` is asked about, so that no truth enters a `Case`;
- `retrieval`: what the retrieval run asks (the query, sources, ranges, `k`, tolerance, and the moments and other moments
  to score against), else null.

`grade.rule` is one of `number`, `count`, `time`, `code_time`, `absent`, `choice`, `set`, with `truth`, `time`, `tol_s`,
`wrong`, `ignore`, `date` and `items`. `set` is for "find every one": right only with the exact set of `items`, which are ISO
dates or file names (a file by its base name, in any case, wherever the agent saw it), and the F1 of the items found is kept
beside the verdict as `items_f1`; a line written before `items` existed reads as none. The loader refuses a `set` with no
items, items on another rule, a repeated item, dates mixed with file names, an item that is no date, and two files of one
base name. `ignore` holds the question's own numbers and clock times, which an answer may repeat (`grade.asked`); the digits
of a name (`cam3`, `IMG_0003.JPG`, a path) and a month of a year are no numbers, and a count the question also holds is still
the answer. The kinds written are `find`, `last`, `when`, `count` and `absent` (V7) and `choice` (V8). `now` is ISO
with an offset. V7's is noon on the day after its last day (C12): the pilot's prompts used the real clock, 4 November
11:27 to 15:01, and a fixed `now` makes a re-run ask the same question. mm's clock then reads later than `now`, so mm's
`clock_note` ("give since and until as dates and times, not as ages or now") is added when the wall clock is more than an
hour past it. That happens only on a V7 re-run made after the recorded day; V8's questions are always older, so V8 has the
note every time, as recorded. `split` is `dev` for V7 seeds 1-10 and for all of V8 so far, because
mm's skill was tuned on these runs (6.4). Tags written: V7 has `modality`, `template`, `level`, `last`, `absent`,
`lookback_h`, `certificate_s`, `rung`, `first_day`, `last_day`; V8 has `modality`, `template`, `level`, `last`, `absent`,
`age`, `category`, `lookback_h`, `rung`, `pilot`, `day`, `first_day`. The file is `<cache>/dumps/<key>/questions.jsonl`,
copied into each run; both are on every agent's deny list, and it is never in the data folder.

### 6.2 The answer

Every agent ends with one JSON object:

```json
{"answer": "Monday 07:30", "evidence": [{"file": "cam1/2026-11-02_07.mp4", "t0": "07:29:50", "t1": "07:30:05"}]}
```

A cited span may be wall-clock times or seconds into the file. A system that works in wall-clock time and an agent that
works in file offsets are graded alike.

**As built (`mmsb/core/answer.py`).**
- **Two formats.** `json` is the default. `line`, one final line `ANSWER: <answer>` with no evidence, is the recorded
  runs' format; it exists for the pilot re-run and is to go once Dan accepts that re-run (D1, section 12).
  `--answer-format` picks one; a run keeps its format in `run.json`, and changing it on a run that has results is refused. A
  run with results and no recorded format reads as `line`, which is what the converted pilot and EgoLife runs are.
- **Parsing.** The answer is that of the last complete JSON object in the final message; a code fence around it is just
  text. It needs a string `answer` and a list `evidence`. Anything else is a failed answer (`answer` null and the reason in
  `answer_error`), counted as `unparsed` in the report. It is never read as a `line` answer.
- **Correction C7.** An agent that has only a system's tools never sees a file path, so a citation names its moment in
  either of two ways: `{"file", "t0", "t1"}`, with seconds into the file or a clock time (`H:MM[:SS]`); or `{"source",
  "t0", "t1"}`, with full ISO times that carry an offset. A `file` is its manifest path, or a longer path ending in `/` and
  the manifest path (the agent may have read it in a folder). A clock time is the first moment at or after the file's start
  that the local clock reads it, so it cannot name the day within a file of several days: use `source` and ISO times.
  *As built since the smoke run (6 November 2026):* a `file` may also carry ISO times, as a photo's time is written: with an
  offset, the instant; with none, the dump's wall clock, as a clock time is. Real Claude cited a trail-camera photo so, and its
  right answer had been recorded as unparsed.
- **The prompt.** It ends with a sentence asking for the format (`answer_request`). V7 names what the agent holds there:
  `mm`, `the files`, or `mm and these files`. V8 has no "unknown" clause, because its questions are multiple choice. The
  `S+files` sentence ("unknown when mm and these files do not hold it") was never recorded. The real agents have not seen
  the `json` request yet.

### 6.3 Scores

- **Grounded accuracy** (the headline): the value passes its rule, and every cited span overlaps a truth span within the
  kind's tolerance (10 s for video, 5 s for sound, the same file for a picture). A right value with wrong evidence is
  counted as `lucky` and shown apart.
- **Find all and count:** event F1 by distinct moments, as DCASE scores sound events, with 1 s and 5 s collars in place of
  DCASE's 200 ms. A count is also graded exactly.
- **Absent:** right only with "not there" and no evidence. The report counts the searches each absent question drew.
  Deep Video Discovery's main failure is the "search trap": searching again and again for what is not there.
- **Time:** a time-to-correct curve per configuration (the share answered correctly within 30, 60, 180 and 600 s). Beside
  it, a score in the style of the Lifelog Search Challenge, max(0, 100 - 50·t/T - 10·wrong), with `wrong` counting cited
  spans that hold nothing. It is never shown instead of accuracy.
- **Cost:** agent tokens and dollars, and system GPU seconds and API dollars, per correct answer. Ingest cost is reported per
  dump and amortised over 1, 10 and 100 questions. A Pareto plot shows accuracy against seconds and dollars.

**As built (`mmsb/core/grounded.py`).** `passed` is what it always was: the value rule, and no problem. The grounded status
is told beside it, so that a re-run compares with the recorded verdicts on `passed` and reports grounded accuracy next to
them. The report's headline is still `passed`, because the recorded runs have no status. The statuses:

| The answer | Status |
| --- | --- |
| the `line` format, or a result made before the status existed | `not_recorded` |
| the value fails its rule | `wrong` |
| an absent question, the value passes, no evidence | `grounded` |
| an absent question, the value passes, evidence given | `wrong` |
| the value passes, at least one citation, and every citation hits a truth span | `grounded` |
| the value passes, and it cites nothing or a citation hits nothing | `lucky` |

- **Tolerances, and correction C8.** 6.3 named none for text and records. A citation hits a truth span of a video within
  10 s, of a sound within 5 s, and of a line of a log or a transcript within 5 s, always on the same source. It hits a
  picture, a note or a version of a config only by being that file or, in the `source` form, by a span that holds the
  file's own time to the second. A citation whose file the dump does not have hits nothing and is counted as unresolved.
- **A broad span is not penalised.** The rule is overlap within the tolerance, as 6.3 says, so a citation of a whole file
  that holds the truth hits it.
- **Count and find all.** The result also holds the event F1 with a 1 s collar and a 5 s collar (`f1_1s`, `f1_5s`). It is a
  maximum matching: a citation is matched to one truth moment and a truth moment to one citation, so one wide citation finds
  one moment. A further citation of a moment already found counts once, not as a false alarm. Citations that find nothing
  are false alarms, and those within the collar of each other are one.
- **New result fields:** `answer_format`, `evidence`, `grounded`, `evidence_unresolved`, `answer_error`, and `f1_1s`, `f1_5s`
  for a count. The report adds grounded, lucky and unparsed columns only when some result has a status.
- **Not built:** the time-to-correct curve, the Lifelog Search Challenge score, the Pareto plot, ingest cost amortised over
  1, 10 and 100 questions, and the count of searches an absent question drew. The report has seconds, tokens and GPU seconds
  per correct answer.

**The neutral report (`mmsb/core/summary.py`, 6 November 2026).** The core's first report (`report.py`, `render.py`) is V7's
mm report and stays where it is, because `compare`, `repeat`, the dry run and the recorded pilot's oracle read it there. The
real scenarios report with `summary`, which names no system. Per scenario, agent and configuration: right out of n with its
Wilson interval, grounded, lucky and unparsed answers, absent questions answered, `unknown` on a real one, seconds, thousands
of tokens and dollars per right answer, the mean `items_f1` of the `set` questions, timeouts, and right by kind. A question
that a leak filter (`listing`) answered right, and that has something to find, is listed as a leak and left out of every other
configuration's figures; a result that is not done (a limit, a setup fault) is counted under "to run again" and left out. Each
other configuration, filters aside, is set beside the control (`files`, `--control`) on the questions both answered, with
McNemar's counts. `mmsb report` gives a scenario's own report for its runs (the summary for the four real scenarios, which
pool runs of other seeds and refuse the same questions twice; V8 one run at a time), and the summary with a table over all
for runs of several scenarios.

### 6.4 Dev and test

Every scenario has a public **dev** split (about 30% of its questions) and a **test** split (about 70%).
- Any system, mm included, is built and tuned on dev only.
- Test results are published only from a frozen system version.
- For the synthetic screen week, seeds 1-10 are dev, and test seeds are drawn fresh for each release. mm's skill and the
  copied harness were adjusted after runs on pilot seed 1, so that seed is dev.
- For public datasets, which questions are in test stays private to the run.

As built, V7 writes `dev` for seeds 1-10 and `test` for others, and V8 writes `dev` for all 50 questions, because mm's skill
was tuned on those runs. `mmsb run --splits dev,test` selects by it. No test split exists yet.

## 7. Building the dumps

### 7.1 Ways to know the answer

1. **Annotated real data.** A dataset's own time spans become questions through templates in user wording: MEVA's
   activities, AMI's topics and decisions, DROID's episode instructions, camera-trap species, the soundscapes' calls,
   FSD50K's labels. Each question is checked by rule against the spans, and Dan spot-checks 10%.
2. **Real needles in real backgrounds,** where annotations do not cover what users ask.
   - Sound: real event clips mixed into real ambience at a set SNR, with exact onsets (the Scaper method).
   - Video: real clips spliced at natural cuts, or a real object composited with blending. The probe must show that a
     splice is not the most unusual moment in its scope.
3. **Owner-written questions about the owner's own data** (I1, I2, M1, X2 with Dan's own media). ScreenshotVQA, camroll
   and ATM-Bench all chose this over LLM-written questions as more natural. These stay private.
4. **Scripted recordings** (X1, and M1/M2 as scripted sets). A person follows a script of planted events or lines while
   real devices record. This gives real light, real noise and real codecs with the truth known by construction, and sound
   truth that nobody has to hear.
5. **Real footage on a plausible schedule** for the synthetic screen week and its camera (X-LeBench's method): real clips
   laid on the plan's days, carrying the plan's needles and decoys.

### 7.2 Real dumps

Dumps keep what real ones have:
- sizes from 1 MB photos to multi-GB videos and 24-hour sound files;
- codecs and containers: H.264, HEVC, MP4, MOV, MKV, M4A, MP3, WAV, FLAC, Opus;
- variable frame rate, rotation flags and HDR;
- HEIC, JPEG and PNG;
- EXIF present, stripped or wrong, time zones and daylight saving;
- duplicates, nested folders, and names like `IMG_4821.HEIC` and `VID_20250612_181502.mp4`.

**As built (`mmsb/scenarios/realdump`, 6 November 2026),** what the four real scenarios share:
- **Writing.** Each file is copied (never linked: the source stays read-only) into the dump under the name a device or folder would
  give it, in parallel, and its modified time is set to its recording time; the manifest entry measures it. A wall time with no
  zone is this machine's local time, so the recipe records the UTC offsets of its days and a dump built in another zone is
  refused (`check_offsets`).
- **Questions.** `question` makes one with its grade, tags, scope and evidence spans (`Span(file, t0, t1)`, seconds into the
  file), and puts the question's own numbers and times into the grade's `ignore`; `split` marks about 30% `dev` by the seed.
- **The prompt** says the time it is ("It is now Monday 23 December 2013, 09:00"), what the pile is and where it is (or which
  system holds it), how its files are named and timed, and the answer request (6.2).
- **The deny words:** the scenario's and the builder's code (`from mmsb`, `import mmsb`), the source's names in commands (as
  paths where a name is inside ordinary code: `/sins` and not `sins`, which is in `isinstance`), and `bench-data`, where the
  sources are, in a command and in what it prints.
- **The probe** (7.3).

### 7.3 The shortcut probe and leaks

**The shortcut probe** (from the copied `shortcuts.py`) checks every dump before any run. No file name, file size, EXIF or
container field, frame difference, splice seam, colour, loudness or frequency band may single out a needle.

**Correction C1.** The probe is the scenario's, not the core's. `shortcuts.py` is V7 throughout: its floors are by V7's
scale, and it reads V7's frame lists and the areas of its dialogs and boxes. It is `mmsb/scenarios/v7_screen/shortcuts.py`,
and each scenario brings its own probe. The core keeps the outcome (`passed`, `failed`, `skipped`, or `none` for a scenario
with no probe, such as V8, a public dataset) and says what to say of a run whose probe did not pass (`core/probe.py`). The
outcome is in `run.json`, the probe's files are `shortcuts.json` and `shortcuts.md` in the run folder, and the outcome is
cached with the dump. `mmsb run` refuses a run whose probe failed unless `--anyway`; `mmsb build --no-probe` skips the
probe, and the report says so.

**The real scenarios' probe (`realdump.probe`).** No file's path, and no text of its metadata, names a label: each label as a
phrase and by its words of three letters or more (but the commonest, such as "person"), with the plain words a scenario adds
("vacuum" for `vacuumcleaner`). Metadata is read by suffix: JPEG segments before the image, WAV chunks but the samples, and
V1's MP4 boxes; a file of a suffix with no reader fails, and only runs of four printable characters or more are text. File
sizes are not checked: they are the recordings' own (on the trail camera an animal's photo tends to be smaller than an empty
one, as on any card), and nothing can be done about them without changing the media. The `listing` leak check at run time
catches what the probe cannot, such as a photo whose time alone gives the answer.

**Leaks:**
- No network for agents or local systems, so no one looks up a public dataset's answers.
- Files get the names a phone, camera or recorder would give them, and dataset identifiers are removed.
- The `listing` configuration runs on every scenario. A case it answers is flagged and kept out of the headline. This is
  LVBench's and HourVideo's blind filter, and Daily-Omni's text-only filter, applied to what a dump really leaks.
  - **As built,** a system declares itself a leak filter (`Description.leak_filter`; `listing` does). A configuration is a
    filter only if it has no tools of its own, so `listing+files` is an ordinary tested configuration. The report names the
    questions a filter configuration answered right, as `answerable_from_metadata`, and leaves them out of the paired
    counts. A question counts if any agent's filter passed it, and the count is of one run.
  - **Correction C9.** Absent questions are exempt. An agent with nothing to read answers "unknown", which is right for
    every absent question, so none of them would say anything about a leak.
  - Its only time is `mtime` (3.2), which is why it is refused on a dump whose files are not at their time, V8 as downloaded.
  - **Open (P1): chance passes.** A multiple-choice question has four options, so a filter configuration passes about a
    quarter of them by chance (about 44% of them across two agents), and the filter would flag those and drop them from the
    paired counts at random. The rule for a chance pass (rotated options below, or a floor above chance) is not built, so
    `listing` is not to be run on a multiple-choice scenario until it is. V8's own report also counts every configuration
    with a system other than `files` as tested and never reads `leak_filter`; that is for the same decision.
- Multiple-choice questions (EgoLifeQA) are asked in 4 rotations of their options and count only if all 4 are right
  (VNBench). Not built (C14): the recorded V8 run has one rotation, re-running it is not P0, and its recorded report stands
  as it is. Do it when V8 is next asked.

### 7.4 The ingest report

Once per dump per system version:
- seconds to ingest, GB per minute, GPU seconds and disk used;
- files refused or misdated, and why;
- seconds from the end of the dump to the first answer.

Every system meets the large-file track (L) here. The copied pilot already found two limits in mm this way (section 10.3).

As built, a store's `ingest.json` holds the seconds, GPU seconds, bytes written, the files refused with the reason, the files
misdated, and the system's `store_hash`. `mmsb build` and `mmsb ingest` print them. A misdated file refuses the run and every
later command over that store, even for another run. GB per minute and the time to the first answer are not built.

### 7.5 The size ladder

The needles of V1, A1, I1 and M4 also sit in smaller and larger haystacks of the same source: 1 hour, 4 hours, 1 day and
3 days (or 200, 2,000 and 20,000 photos and clips). The ladder comes from Visual Haystacks and TimeScope. It shows, per
system and agent, from what size of dump a system pays off against `files`. A system that pays once at ingest and one
that pays per question cross at different sizes. The smallest rungs also give the quickest cases to select.

## 8. Running

### 8.1 Where the time went in the copied pilot (measured)

The pilot ran one system (mm with d1-3B) against `files`, on 3 synthetic days, seed 1
(`D:\mm-bench\20261104-0314-pilot-s1`):

| Step | Wall time | Bounded by |
| --- | --- | --- |
| build (write, load, probe) | 11 min 23 s | CPU, mostly one process |
| Codex `files`, 31 cases, 3 at a time | ~14 min (1,896 session-seconds) | the API |
| Claude `files`, 31 cases, 3 at a time | ~85 min (6,693 session-seconds; its longest hit the 900 s limit) | the API, and brute force over frames |
| Codex `mm`, 24 cases, 1 at a time | ~25 min: 1,513 session-s, of which **1,156 GPU-s (76%)** | the GPU |
| Claude `mm`, 24 cases, 1 at a time | ~49 min: 2,942 session-s, of which **2,499 GPU-s (85%)** | the GPU |

The run took about 3 hours of work, and 4 with a driver crash and a reboot. The laptop's NVIDIA driver has crashed under
one GPU job and under three, so **exactly one GPU job runs at a time**, whatever the system. In the code that is one rule
(`mmsb/core/atonce.py`, 8.4): a configuration whose system needs the GPU runs one exec at a time unless `--gpu-jobs N` says
more, and `--jobs` and `--clock` do not. A per-question GPU system
already keeps the GPU busy 76-85% of the time, so running more agents against it saves little. Time is saved by choosing
cases, running GPU-free work beside GPU work, and ingesting once.

### 8.2 Choosing cases

Running faster means choosing fewer cases. Nothing else about a run changes.

- **Select** by case ID, scenario, media shape, question kind, split, rung of the size ladder, or any tag:

  ```
  mmsb run --scenarios V1,A1,M4 --rung 1h --kinds last,absent --limit 12 --agents claude --configs files,mm+files
  ```

- **`--limit N`** draws N cases at random, stratified by scenario and kind, with a fixed seed. A small run keeps the mix of
  the full one. It is the largest-remainder share of each (scenario, kind), then a sample inside it; the canaries count
  toward N.
- **`--discriminating RUN[,RUN]`** keeps only cases that separated configurations in earlier runs: right in some
  configuration of one agent and wrong in another, in one run. A case right in every configuration, or wrong in every one,
  tells a paired comparison nothing. Two such cases stay in as canaries, one right and one wrong in turn. Only the agents
  named in `--agents` count.
- **As built,** the flags are `--scenarios` (as the question names it, `V7`), `--kinds`, `--modalities`, `--splits`,
  `--tag NAME=VALUE` (repeatable), `--rung X` (which is `--tag rung=X`; V7's rung is its scale and V8's is `3d`, and the
  size ladder waits for P2), `--only QID,..`, `--subset pilot`, `--limit` and `--discriminating`. A value no question has is an error. A
  selection prints what each step left (`selection: the questions: 31, then --kinds last: 4`) and one that picks nothing is
  refused.
- **Before it starts,** the runner prints what the selection will cost, and `--dry-run` stops there:
  - cases × agents × configurations;
  - the dumps and stores that must first be built or ingested, with the time the earlier ones took;
  - wall time and GPU seconds, estimated from the medians of what the same cell took in earlier runs (and the mean beside
    them, because a few 900 s timeouts make a run longer than its median says). A cell with no history is counted at 2.5
    minutes an exec (1 for a blind one) and says so.

  `run --dry-run` still writes the first exec's files for each agent and configuration, in the run's `dry-run/` folder, as the
  copied code did, and starts no agent. `repeat --dry-run` writes nothing.
- **The scale of a selection:** a dozen cases on the 1-hour rungs, with ingest cached, take minutes. Every scenario at full
  size takes a night, and the first ingest of a large dump can take longer than the cases themselves. (Estimates: no rung
  exists yet.)

A small selection cannot certify a difference (section 9), and its report says so.

### 8.3 The scheduler

**Three resource pools:**

| Pool | Size | Work |
| --- | --- | --- |
| CPU | at most 4 processes (the laptop's rule) | download, re-mux, generate, CPU ingest, probe |
| Agents | per CLI, up to 4 sessions (quota and rate limits) | every configuration's agent turns |
| GPU | exactly 1 job, for every system together | GPU ingest (`embed`, `transcript`), per-question GPU (`mm`), the retrieval run |

Configurations that need no GPU per question (`listing`, `files`, and `embed+files` once ingested) run beside those that
do. The longest expected cases start first. Ingest of the next dump fills the GPU's idle gaps. The speed research puts the
saving of running side by side at 25-35% of a run's wall time (an estimate: nothing here has measured it on real agents).

Two conditions:
- **Dan's go-ahead.** CPU-heavy work beside a GPU job goes against the laptop's current "GPU runs alone" rule. This is
  question 13.2, still unanswered, so running side by side is **off** unless `mmsb run --side-by-side` asks for it, and the
  run prints that it is on. Without the flag the cells of a run run one after another, as the copied `run_agents` did.
- **Agents stay off the GPU:** CUDA is hidden from their environment, and `-hwaccel` and `nvidia-smi` are on their deny
  list. An agent decoding on the GPU would be a second GPU user.

**As built (`mmsb/core/schedule.py`).** A cell is one agent and one configuration. With the flag, each cell gets a thread,
and the threads share the pools:
- **The GPU token.** A cell that needs the GPU holds its one token for its whole length, its session's `open` and `close`
  included. The cells that need it take it in the order they would have run one after another. Cells that need none start
  when a slot is free.
- **The execs.** At most 4 execs run at once in all, across both agents, not 4 for each: the working folders and the deny
  words are made for 4, and more crashed the laptop once. Each agent also has a gate of 4 sessions, and `--jobs` still
  bounds one cell, `--gpu-jobs` (1 unless asked) the cell that needs the GPU: the rule of 8.4 holds inside it. A free slot goes to the cell that needs the GPU first, then to the earlier cell in the sequential order.
- **Longest first.** In a cell that needs no GPU and whose system does not cut its index to each case's horizon, and in a
  cell with no system, pending cases start longest first, by the medians of earlier runs. A cell whose system cuts its index
  (`horizon = "cut"`, even `listing`, which needs no GPU) keeps the order of `until`, because the as-of order needs it. A
  cell with no history keeps its order.
- **Not built:** ingest of the next dump filling the GPU's gaps. It waits for a system with GPU ingest (P1).
- **On paper only.** `simulate` runs the same policy event by event with no sleeps, and the dry run prints its estimate. A
  side-by-side run is not always shorter than the sequential one: scheduling under a bound pool can come out longer than the
  sequential order (Graham's anomaly: about 1 in 3,000 seeded random cases, by up to 2.7%). It is never longer when no pool
  binds.

**Correction C11: the rule that keeps agents off the GPU.** The design kept agents off the GPU only when running beside GPU
work. But `mm+files` is one cell with a shell and a GPU job in it. So the rule (`agent_off_gpu`) applies to any agent with its
own tools when the run is side by side, or when its own system needs the GPU for its questions:

| Configuration | Without `--side-by-side` | With it |
| --- | --- | --- |
| `files` | no rule (as recorded) | off the GPU |
| `mm+files` | off the GPU | off the GPU |
| `mm`, `listing`, `blind` (no own tools) | no rule | no rule |
| `listing+files` | no rule | off the GPU |

"Off the GPU" is `CUDA_VISIBLE_DEVICES=-1` for the agent's shell (Codex) or process (Claude, so also its MCP server), and
`-hwaccel` and `nvidia-smi` in its deny words. Recorded `files` execs stay as they were by default, so the exec oracle holds.

**A hazard that is not guarded.** The sessions of one agent and one variant share a home (`mm/d1-3B` and `mm/d1-3B+files`
use one). Two cells that share a home would collide under `--side-by-side` if neither held the GPU token. Real mm variants
need the GPU, so the token serializes them; a variant with no GPU (the tests' `mm/fake` and `mm/fake+files`) has nothing
that stops two cells using one home. Guard it before a system without a GPU keeps per-agent state (P1).

The copied harness's error classes stay:
- `setup`, `limit`, `auth` and `error` results are asked again and left out. `setup`, `limit` and `auth` stop the agent at
  once; three `error` results in a row stop it too;
- `escaped`, `timeout` and `capped` are wrong answers;
- a system that dies, or a lost GPU, counts as `setup`.

A run that stops for any reason continues where it left off when it is started again.

### 8.4 Working time, ingest once, and honest GPU seconds

**Working time, not wall time.** When several cases share the GPU, waiting for it is an artefact of the benchmark. A real
user's system has the GPU to itself. So:
- A case is charged its agent's turns plus its own GPU service time, never queue time. This is Inspect AI's
  `working_limit`, paired with a wall-clock cap.
  - In `lease` mode the runner measures the queue itself (not built).
  - In `exclusive` mode the system reports it through `Session.queue_s(case)`, read live while the exec runs, as vLLM
    reports queue time apart from service time. (C10: the design said `usage()`; the clock needs the figure during the exec.)
- A tool that returns early with partial results counts that time from when its GPU work starts, not from when it was
  queued, so a queued request does not come back empty and invite polling.
- **Before trusting it,** run 10 cases both alone and queued, and compare answers, tokens and charged seconds
  (`mmsb clock-check`, below).

**Budgets sized to the haystack:** 300 working seconds for an hour of media, 900 for a day or more, with a wall cap of 3×
that. A case that the retrieval run shows no system can scan within its budget gets its hint narrowed, or is left out of
default selections.

**As built (P0), `mmsb/core/clock.py`.**
- **The clock.** An exec is held to `working = wall - queue` against its limit (`--timeout`, 900 s for V7 and V8), and to a wall
  cap of 3× the limit whatever the queue. The agents look at it at least once a second while they wait for their process, so a
  queue that grows moves the end of the limit.
- **No queue figure, `clock: wall`.** A system whose `queue_s` is `None` leaves the clock on the wall, and the result says so.
  That is exact when nothing else shares the system.
- **The texts.** The recorded `timeout: no answer in 900 s` is kept byte for byte, unless the exec queued: then it adds
  `of working time (1010 s on the wall clock, 120 s queued)`. The wall cap says `timeout: no answer in 2700 s on the wall
  clock`, with `(120 s of it queued)` when it did.
- **`--clock working|wall`,** on `run` and `repeat`, default `working`. `wall` charges wall time throughout.
- **Result fields.** `clock`, `working_s` (what the limit was measured on) and `queue_s` (`null` when not reported). `seconds`
  stays the wall time, so the report is unchanged. A result made before the clock has none of them and is still done.
- **How many at once (`mmsb/core/atonce.py`).** One rule, asked by the runner, every dry run and estimate, the refusal below and
  `clock-check`. A cell runs `--jobs` execs at once (default 3), unless its system needs the GPU (`Description.needs_gpu`):
  then it runs `--gpu-jobs` (default 1), because three decision-model execs at once crashed the NVIDIA driver on 2 November
  2026. Neither is raised by the other, nor by `--clock`. The system's own limit (`Session.max_jobs`: mm is one at a time
  for a case with a horizon, or for a slow model) comes on top and only lowers it. `repeat`, which has no store yet to ask
  the system, counts the rule without that limit. Several execs on the GPU are only what the owner asks for with `--gpu-jobs`,
  and a run, a dry run, a `repeat` and a `clock-check` print a `warning:` line when they will run several. `run` and `repeat`
  print a line of the counts (`execs at once: 3 (--jobs); 1 for those that need the GPU, mm/d1-3B (--gpu-jobs)`), and a cell
  whose count is not `--jobs` says so, so the dry run's wall estimate is the one the run is held to.
- **The refusal.** More than one exec at once on the working clock, with a system that gives no queue figure, is refused: each
  exec would be charged its wait for the others. "At once" is the rule above, which can be fewer than `--jobs`. It is checked
  from the preview sessions before anything is spent, so `run --dry-run` refuses too; a fresh `repeat --dry-run` builds nothing
  and has no store to ask. In a real `repeat` the refusal comes after the build, which it needs (at the defaults mm is one at a
  time, so the pilot's `repeat` is not refused). The message names one exec at a time by the option of the system, `--gpu-jobs 1`
  or, for a system that needs no GPU, `--jobs 1`. It never offers `--clock wall`, which would keep the several. `--clock wall`
  with `--gpu-jobs` above 1 is allowed, because the owner asked for them, and prints the warning.
- **`budget_s(media_hours)`** is linear from 300 s at 1 hour to 900 s at 24 hours (300 below and 900 above). It is unused until
  a scenario has an hour rung.
- **mm reports no queue (C10, decision D3).** MultimediaSearch keeps no per-job wait, so `NoQueue`, the base of both of mm's sessions,
  answers `None`: mm is `clock: wall` until D3. That is exact at one exec at a time, which is mm's default (`--gpu-jobs 1`), as
  the pilot ran its mm cells: three mm execs at once crashed the NVIDIA driver on 2 November 2026. So several are only for
  `--gpu-jobs`, and until D3 the working clock refuses them. "Partial results count from GPU start" is in mm's adapter
  too (its `ask` waits up to 45 s from the call). Both are changes to MultimediaSearch, about 30-50 lines. Not made.
- **`mmsb clock-check RUN --config CONFIG [--agents --cases N --jobs N --gpu-jobs N]`** is the check before trusting it, **not yet
  run.** It asks N cases (default 10) of each agent twice, alone and then all queued, the queued half at the execs at once of the
  rule above: `--gpu-jobs` for a system that needs the GPU, so for mm it needs `--gpu-jobs 2` or more (the default 1 queues
  nothing), else `--jobs` (default 3). It prints for each case its
  answer, tokens and charged (working) seconds in both halves and, queued, the seconds it waited and its wall seconds, then the
  totals. The halves are two folders, `<run>/clock-check/alone` and `<run>/clock-check/queued`, each with copies of the run's
  `run.json`, `questions.jsonl` and `data-manifest.json` and the run's own dump and stores; each resumes like a run. The cases
  are a seeded stratified sample of those the system does not mark `exclusive`; mm marks a case exclusive when it is heavy
  (`estimate_s >= HEAVY_S`), since a case that holds the system alone queues for nothing. A half denies the folder of the run it
  branched from as well as its own, as the run's `questions.jsonl` holds the answers. It refuses, before any agent starts, when: the
  system would run the cases fewer than two at a time (`--jobs 1`, or `--gpu-jobs 1`, the default, for a system that needs the GPU;
  a limit of the system's own is told apart), so nothing would queue; the system gives no queue figure and the queued half is over
  one exec at once (D3, which is every case of mm today); every case is exclusive; the system is not ready; or the configuration
  has no system. It has no model or effort flags: `MMSB_*` apply. Cost and command: 12.2.

**Ingest once (`mmsb/core/cache.py`, as built):**
- Downloads are fetched once and content-hashed (EgoLife stays in `D:\bench-data`).
- Each prepared dump is cached by the hash of its recipe. Its **key** is the sha256 of the scenario's name, its recipe and
  its generator as canonical JSON (reordering a dict changes nothing; a bump of the generator makes another dump). The dump is
  `<temp>/mmb-<key10>-data`, read-only. Its record is `<cache>/dumps/<key>/`: `manifest.jsonl`, `dump.json`, `questions.jsonl`,
  `recipe.json`, `built.json`, `data-manifest.json` (what `fs.freeze` saw), and the probe's outputs.
- Each system's store is cached by (system, version, `store_hash`, dump key) in `<cache>/stores/<system>-<key10>/`, with
  `ingest.json`. The variants of a system whose ingest does not depend on the variant share it: mm's do not depend on the
  model. A run's `stores/<system>` is a hard-link copy, so it costs no space, and each (agent, variant) gets its own working
  copy of mm's home.
- A system or recipe change rebuilds only what it touches.
- **Made whole or not at all.** A dump or a store is written into `<final>.part` and renamed. A `.part` left behind is
  removed, said aloud, and written again. A lock file (an operating-system byte lock, released a few milliseconds after a
  process dies by any means) means a second command that wants what the first is making stops at once with `Busy` and
  touches nothing.
- **A dump that changed is an error, not a rebuild,** because it may be an escape (an agent with its own tools wrote to it) or
  a failing disk, and what ran over it ran over other recordings. A dump that is gone (the temp folder is cleaned by Windows)
  is written again from the recipe and compared with the record: the same files, sizes and hashes, or an error.
- **Caveats.** The keys do not cover the libraries the generator uses: a library change that alters bytes needs a bump of the
  scenario's generator, or the cache removed. The probe's outcome is cached with the dump, and the key does not cover the
  probe's code: remove `<cache>/dumps/<key>/probe` after changing it. There is no eviction: the pilot's dump is about 465 MB,
  and each system's store is kept beside it.
- **To force a rebuild,** remove `<temp>/mmb-<key10>-data` and `<cache>/dumps/<key>`, and the stores
  (`<cache>/stores/<system>-<key10>`) if the bytes changed.
- `mmsb build` and `mmsb repeat` take `--temp` and `--cache`; `run`, `ingest` and `retrieve` read them from `run.json`. A run
  made before the cache has none and is refused.
- The synthetic generator writes with `--workers` files at once (default 3, the recorded build's).

**GPU seconds are the system's business, and the report records them honestly.**
- Caches inside a system (encoder outputs, answer caches) are allowed if they would also serve a real user.
- Across cases, a cache may only keep what ingest built, so no case gains from another's.
- Batching is the system's own choice, and the report records it.

### 8.5 Runs without agents

- **The retrieval run:** each system's `oracle()` per needle, with no agent. It reports recall at k, average precision and
  seconds. It is cheap, needs no quota, and separates a system's search from the agent's use of it. As built it is
  `mmsb retrieve RUN [--configs ...] [--dry-run]`, for the systems that declare an oracle (mm). Each needle question asks the
  yes/no its `retrieval` spec holds (`k` sightings, 5 for a `last` question and 50 for a count; 5 candidates besides; 1,800 s),
  and the ranked moments are scored
  against the truth within the spec's tolerance: found, top 5, the needle's best p, the first rank, false sightings, and why a
  miss missed (`near`, `stopped`, `cut`, `missed`). The records are `results/retrieval/<config>/<qid>.json` and `warm.json`.
  Recall at k and average precision as named here are not computed; the report's "recall" and "top 5" columns are these
  counts. It holds the GPU alone.
- **Re-grading:** recorded runs are graded and reported again after a grader or report change (Harbor's `regrade`), with no
  agent and no GPU. As built there is no command for it. `mmsb report` reads each result's `passed`, and the tests regrade the
  260 recorded answers through the question files (`Question.grade`) and check that they equal the recorded verdicts, 110 of
  110 and 150 of 150.
- **Replays for harness tests:** recorded tool results are replayed at their recorded latency, so the runner can be tested
  offline. As built (`tests/replay.py`): a stand-in process prints a recorded `events.jsonl`, and the real agents, watcher and
  classifier run over it inside the real `run_agents`, with no latency. It covers normal, timeout, capped, escaped (planted),
  setup (planted) and lost-GPU execs (TESTING.md).

### 8.6 Session mode

Optionally, one agent session gets 5 questions about one dump in a row, as a user asks them, so startup and learning the
dump are paid once. It is reported apart, since its questions are not independent.

## 9. Statistics

Each system is compared with `files` per agent, question by question: paired, with McNemar's test. The pre-registered
comparisons are those pairs, corrected with Holm's method. Rankings between systems carry bootstrap intervals over
questions.

**How many questions.** Paired questions per agent and comparison, at 80% power and α = 0.05 (Connor's approximation for
McNemar):

| Effect, and share of pairs right in one configuration only | Questions |
| --- | --- |
| +17 points, all discordant pairs one way (Claude's pilot), optimistic | about 45 |
| +15 points, 20% discordant | about 70 |
| +10 points, 15-20% discordant | about 115-155 |
| +4 points, 8% discordant (Codex's pilot) | about 390 |

So a run that wants a verdict selects 70-150 media questions across scenarios and aims at effects of 10 points or more. A
smaller effect, or a smaller selection, is reported as not certified, never implied.

- **Distinct questions, not repeats.** Repeating a question shrinks only its own run-to-run variance, never the spread
  between questions (Miller, "Adding Error Bars to Evals"). Repeats go to a reliability set of 10 questions × 3 runs,
  reported as pass^k (τ-bench).
- **Sequential stopping, pre-registered:** cases run in a random order, and a comparison stops when an anytime-valid rule
  (an e-process or a confidence sequence) decides it, capped at 150. It saves most when the effect is large.

## 10. Where the copied code stands, and how it becomes MMSB

### 10.1 What was tied to MultimediaSearch

The copied benchmark was nine files at the repository root, about 9,000 lines: `bench.py`, `claude_run.py`, `codex_run.py`,
`egolife.py`, `grade.py`, `haystack.py`, `haystack_media.py`, `shortcuts.py` and `egolife_oracles.json`. They reached into
MultimediaSearch for its reader, its config and ffmpeg lookup, and its test helpers (`day`, `fake_decider`, `conftest`); the `mm`
arm's server, tool names and prompt were written into the agents; and `bench.py` held mm's daemon, snapshot homes, scrubbing,
jobs, GPU estimate and model-only tier. All of it has moved and the old files are gone. Commit bb498a6 holds them, and
`tests/oracles/capture_exec.py` reads them from there. Where each part went is the layout in section 2. Three moves were not
a literal reading of the first design:
- **The probe (C1)** is the scenario's: 7.3.
- **Deny words (C3).** V7's are the old generator's eight words, because copies of `haystack.py` still sit in MultimediaSearch's
  `benchmarks/` on this machine, and words for the new modules (`v7_screen`, `from mmsb`, `import mmsb`). A bare `mmsb` is not
  one, because it would match the data folder's name.
- **The window count (C5).** mm's `windows_in` read V7's plan, which a system may not see. It is rebuilt at ingest from the
  dump (`store/windows.json`) and held to the recorded `gpu_estimate_s`, `exclusive` and `beyond_timeout` of all 48 pilot and
  50 EgoLife `mm` results. V8's clips count windows by a span rule, not by demuxing, which agreed with the recorded
  estimates on 50 of 50 results where a simulated demux agreed on 10 of 40. A sound counts one window for each of its windows,
  an upper bound on the old count; d1-3B does not hear, so no recorded `mm` case is affected.

The published results stay in `results/2026-11-04-haystack-pilot/` and `results/2026-11-03-egolife/`, as they were. The
repository's old README, which told the pilot, is now `results/2026-11-04-haystack-pilot/README.md` under a line saying its
commands and names no longer apply, and the repository README is new.

### 10.2 The order of the work

1. Copy in the few helpers from MultimediaSearch's `tests/` (font, sample rate, the shade and pitch constants), and give the
   synthetic week its own ffmpeg lookup. V7 then builds with no MultimediaSearch. **Done.**
2. Write the question-file format, and make V7 and V8 write it. **Done.**
3. Split `bench.py` into `core/runner` and `systems/mm`, and make the agent runners take tools from `serve()`. **Done.**
4. Re-run the copied pilot through the new runner. The same verdicts for `files` and `mm` (the pilot's mm-only
   configuration) show that nothing was lost. Then `mm+files` runs for the first time. **Not done:** it needs the GPU and
   Dan's go-ahead, and P0 ran no GPU and no paid agent. `mmsb repeat` and `mmsb compare` are built; what stands in place of
   the re-run is the offline proof in TESTING.md. The command and what "the same verdicts" means (D1) are in section 12.
5. Add case selection, `listing`, then `embed` and `transcript`. Case selection and `listing` are done; `embed` and
   `transcript` are P1.

### 10.3 What mm's plug-in will meet first

The ingest report and track L will exercise two limits of mm today:
- `remember` refuses any file over 2 GB (`remember_max_mb`).
- A folder source keeps a file over 50 MB in place, dated by when mm saw it rather than by its recording time.

These belong to mm's notes, not to MMSB's design. Track L is built so that any system's handling of big files shows the
same way.

The port found four more, all MultimediaSearch's:
- `remember` treats two files with identical bytes as unchanged, so a duplicate makes ingest fail with "not stored once".
- The window estimate depends on two private functions of MultimediaSearch's reader (`reader._times`, `reader._video`). A test
  compares the copies in `v7_screen/frames.py` with them and fails if they are renamed or change.
- `set_horizon` writes into the live index while the daemon runs, and relies on the daemon rereading it at each call.
- mm's skill changed in 5 lines (41 characters) between the pilot and the commit the goldens were captured at; a test pins
  the line map and fails when the skill changes again.

## 11. Ideas from other benchmarks, and datasets

From web research on 4 November 2026, in three sweeps: video, sound/pictures/lifelogs, and harness speed. Links are to the
papers. "Unverified" means only a secondary source was found. Several 2026 papers are recent, so treat their numbers as
theirs.

### 11.1 The closest benchmarks, and what none of them does

| Benchmark | Close because | Scale |
| --- | --- | --- |
| [LVBench](https://arxiv.org/abs/2406.08035), [HourVideo](https://arxiv.org/abs/2411.04998), [EgoLifeQA](https://arxiv.org/abs/2503.03803), [EgoMonth](https://arxiv.org/abs/2608.13113) | hour-to-month video QA | 117 h / 500 videos / 266 h / 300+ h |
| [Ego-R1 Bench](https://arxiv.org/abs/2506.13654) | an agent with search tools over a week of video | 300 questions, ~44 h each |
| [M3-Bench](https://arxiv.org/abs/2508.09736) | memory agents over long robot-view video; questions asked at a time | 100 scripted videos + 920 web; 4,490 questions |
| [LongVidSearch](https://arxiv.org/abs/2603.14468), [Deep Video Discovery](https://arxiv.org/abs/2505.18079) | agentic search, scored with its tool cost | 3,000 questions / LVBench |
| [camroll](https://arxiv.org/abs/2606.05275), [DISBench](https://arxiv.org/abs/2602.10809), [ATM-Bench](https://arxiv.org/abs/2603.01990), [PhotoBench](https://arxiv.org/abs/2603.01493) | a real person's camera roll, with evidence | 31.5K / 109K / 3.8K photos + 533 videos / 3.6K photos |
| [BLAB](https://arxiv.org/abs/2505.03054) | long audio, timestamps scored by F1 | 833 h, files up to 2 h |
| [Visual Haystacks](https://arxiv.org/abs/2407.13766) | accuracy against haystack size, 1 to 10K images | ~1K questions |
| [Lifelog Search Challenge](https://arxiv.org/abs/2506.06743), [Video Browser Showdown](https://videobrowsershowdown.org/about-vbs/) | timed known-item search over months of images and thousands of hours of video, many systems on one task | 725K images / ~2,300-3,800 h |
| [MIRIX ScreenshotVQA](https://arxiv.org/abs/2507.07957) | memory of one's own screen, owner-written questions (not released) | 5-18K screenshots per person |

**What none of them does, and MMSB does:**
- Hand an agent a real dump at real size and in real mess.
- Cover many small audio files as well as long ones.
- Compare search systems behind one plug-in contract, each against the agent's own tools on the same files.
- Report the ingest wait and GPU seconds beside tokens and dollars.

LSC and VBS compare many systems on one task, but with people at the controls, not agents. No benchmark of continuous
screen-video memory was found.

### 11.2 What to take

| # | Idea | From | In MMSB |
| --- | --- | --- | --- |
| 1 | Grounded scoring: correct only if the cited file and span hold the answer | CG-Bench, Ego4D NLQ, TVR, ATM-Bench (Joint@k) | 6.3 |
| 2 | Real footage on a plausible schedule, with real needles spliced at set depths | X-LeBench, TimeScope, MLVU | 7.1 |
| 3 | Real sounds mixed into real rooms at a set SNR, with exact onsets and near-miss decoys | Scaper, DESED, RealDESED | A1, A2, M6 |
| 4 | Scripted real capture, the truth known by construction | M3-Bench | X1, M1, M2 |
| 5 | A haystack-size ladder | TimeScope, Visual Haystacks, V-NIAH (by depth) | 7.5 |
| 6 | Absent questions, about a quarter; count the "search trap" | PhotoBench, MMNeedle, Deep Video Discovery | 5.6, 6.3 |
| 7 | Leak filters: no media, metadata only, rotated options | HourVideo, LVBench, Daily-Omni, VNBench | `listing`, 7.3 |
| 8 | Penalise time and wrong evidence | LSC, VBS | 6.3 |
| 9 | Find-all scored by distinct moments; graded relevance | VBS, MultiVENT 2.0, DCASE | 6.3 |
| 10 | Tags: certificate, look-back, needs-audio, evidence spread | EgoSchema, Video-MME, EgoLifeQA, EgoMemReason, TeleEgo | 5.6 |
| 11 | Multi-hop across files and days; aggregates | DISBench, LongVidSearch, TimelineQA, MVU-Eval | 5.6 |
| 12 | Query by example | Ego4D VQ2D, MomentSeeker | 5.6, I5 |
| 13 | Owner-written questions | ScreenshotVQA, camroll, ATM-Bench | 7.1 |
| 14 | A gold-evidence configuration | LongVidSearch (GPT-5 near-perfect with gold clips, 42% without) | 4 |
| 15 | "As-of" questions: each has a now, so one dump serves many questions | TeleEgo, M3-Bench, EgoLifeQA | `serve(now)`, 6.1 |
| 16 | Score retrieval apart from the agent | DCASE PSDS, LV-Haystack | 8.5, `oracle()` |
| 17 | One plug-in contract, many systems, one scorer | LSC, VBS, TRECVID AVS | 3 |
| 18 | Resource groups, working-time limits, error classes, re-grading | Harbor, Inspect AI, SWE-bench | 8.3, 8.4, 8.5 |
| 19 | Accuracy against cost on a Pareto plot | AI Agents That Matter, HAL | 6.3 |

**Taken as warnings:**
- AudioMarathon builds long audio by concatenating clips. Its seams are the artificial shortcut MMSB avoids.
- TV Needle concatenates videos into one. MMSB keeps files separate, as a user's are.

### 11.3 Datasets

| Dataset | Licence and gating (as found) | Size and annotations | For |
| --- | --- | --- | --- |
| [MEVA](https://registry.opendata.aws/mevadata/) | CC BY 4.0; open on AWS, no account | ~328 h multi-camera video; 37 activity types with spans for ~184 h | V1, M6, V7's camera |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC BY 4.0 for signals, transcripts and some annotations | 100 h; room video, close and far microphones, slides; transcripts | V4, A3 |
| [FSD50K](https://zenodo.org/records/4060432) | CC BY 4.0; per clip CC0, BY or BY-NC; no account | 51K clips, 200 classes (doorbell and alarm classes: check `vocabulary.csv`) | A1, A2, M4, M6 |
| [SINS](https://zenodo.org/records/2555143) | CC BY 4.0 on Zenodo, but its licence PDF says academic use only | a continuous week in a real home, 13 nodes × 4 microphones, activity labels | A1 |
| [DROID](https://arxiv.org/abs/2403.12945) | CC BY 4.0 | 76K episodes, 350 h, 3 cameras, an instruction each | V5 |
| YFCC100M, or [DISBench](https://arxiv.org/abs/2602.10809) on it | per-photo CC; original metadata under Yahoo's Webscope agreement (reports conflict) | real users' photos with EXIF; DISBench: 57 users, 109K photos, 122 verified queries | I1 |
| [Bird soundscapes](https://zenodo.org/records/7050014) | CC BY 4.0; no account | 33 one-hour recordings, 20,147 timed calls | A4, M5 |
| [Caltech Camera Traps](https://lila.science/) | CDLA-Permissive | 243K images, 140 places, `date_captured` to the second (checked) | I3 |
| [comma2k19](https://github.com/commaai/comma2k19) | MIT | 33 h of driving, ~100 GB, GPS and CAN, no event labels | V2 |
| [AgentNet / OpenCUA](https://huggingface.co/datasets/xlangai/AgentNet) | MIT per its card; gating unverified | 22.6K human screen-video tasks with input events | V7 |
| MIMII | CC BY-SA 4.0 (as found) | machine sounds (valves, pumps, fans, slide rails) | A5 |
| EPIC-KITCHENS-100 + EPIC-SOUNDS | CC BY-NC 4.0 | 100 h egocentric with sound; 78K timed sound events | a kitchen scenario, if non-commercial suits |
| Gated: VIRAT, Ego4D, LSC/NTCIR, BirdCLEF, V3C | an agreement, an application or an account | | not without Dan |

On 4 November 2026 Dan decided that licences are not a filter for use on this machine; [MEDIA.md](MEDIA.md) records the
licence each source states. No agent accepts an agreement or makes an account, so gated sources wait for Dan.

## 12. Phases

| Phase | What | Done when |
| --- | --- | --- |
| P0 | The separation (section 10.2, steps 1-4); the question-file format; grounded answers; case selection; `listing`; cached ingest; GPU-free configurations beside GPU ones; the working-time clock checked on 10 cases | the core runs without MultimediaSearch installed, and the copied pilot gives the same verdicts through `systems/mm` |
| P1 | `embed` and `transcript`; the priority-1 scenarios (V1, V5, I1, I3, A1, A3, M4) and track L | each has 15+ questions, a passing probe and a retrieval run for every system |
| P2 | The priority-2 scenarios (V4, I2, A2, A4, M1, M5, M6, X1, X2); the size ladder; session mode | a full run has 70+ discriminating questions per agent |
| P3 | The priority-3 scenarios, once sources are found; hosted systems if wanted | as P1 |

### 12.1 P0's state (5 November 2026), and the real scenarios (6 November)

**6 November 2026: the real scenarios.** MMSB was re-planned as a standalone benchmark of real piles of media (`PLAN.md`). Built
and checked offline: V1, I3, A1 and M4 (5.7) on the shared builder (7.2), the `set` rule (6.1), the neutral report (6.3), and
the real scenarios' probe (7.3). Each full-size dump was built once and probed (TESTING.md section 9). Two independent reviews
gated the work: one of the builder, the rule, the report and I3 before the other three copied them, and one of the whole. Real
Claude and Codex answered two I3 questions in `files` and through `listing` (TESTING.md 4.2): the run found one fault, a file
cited with an ISO time read as unparsed (6.2), which is fixed. Not done: a full run of any of the four with real agents (paid),
and the priority-1 scenarios not yet built (V5, I1, A3, the track L). P1's search systems (`embed`, `transcript`) are not built; `mm` is optional. The smoke run's
Claude (Sonnet) execs on the tiny card cost $0.37 and $0.51 each; a full-size exec is not measured.

**Built, and checked offline** (how: [TESTING.md](TESTING.md)). The core runs without MultimediaSearch: the neutrality tests import
it with MultimediaSearch refused, and in an environment made without the `mm` extra they build V7 tiny, probe it and run a `files` cell.
- The separation, steps 1-3 of 10.2: V7 and V8 build and ask with no MultimediaSearch, and `systems/mm` reproduces the old `mm` arm.
  Four oracles were held to the recorded pilot and EgoLife runs (TESTING.md): the V7 build (the whole week, 1,575 files,
  byte for byte); the report (figures and Markdown); the exec each agent builds (the pilot's 110 execs); and a replay of the
  recorded execs through the real run loop.
- The question-file format, grounded answers (`json`), case selection (`--limit`, `--discriminating`), `listing`, cached
  ingest, side-by-side scheduling (off by default), `mmsb compare` and `mmsb repeat`.
- The working-time clock (`--clock`, 8.4) and `mmsb clock-check`, which has not been run.

**Not done, because it needs the GPU, a paid agent or Dan's go-ahead.** No GPU and no paid agent ran in P0. Real Claude and
Codex have not seen the `json` request, the `S+files` prompt, the listing server, a second `--add-dir` for mm's `looks`
folder, or `--model claude-sonnet-5-5`. Nothing ran against the real V8 download or the real mm model.

**Decisions, and the defaults the work took.**

| # | Question | Default taken |
| --- | --- | --- |
| D1 | "The same verdicts" cannot mean identical per question: the agents are stochastic, and no repeat run exists to measure the noise. | Accept when each cell's correct count is within 2 of the recorded one (Claude `files` 23, Claude `mm` 22, Codex `files` 27, Codex `mm` 23); McNemar of the recorded run against the new one is not significant (p under 0.05) in any cell; the verdict lines read the same; no `setup` result; and a person has read every flip. Run once in `json`; run `line` too (another $19 and about 3 h) only if a cell falls outside the band. It is written once, in `mmsb/core/compare.py` (`BAND`, `ALPHA`). A pass means no large loss; it does not mean the runs agree (a band of 2 is about 6 points of 31 questions, and McNemar over a few discordant pairs has little power) |
| D2 | When may the re-run and the clock check spend? | Not until Stage 2 passed its review, and with the GPU free |
| D3 | mm cannot report queue time. The working clock and the 10-case check need a small MultimediaSearch change: per-job queue seconds from `gpu.py` and `jobs.py`, and `ask`'s early return counted from GPU start. About 30-50 lines there | Make it in MultimediaSearch as its own task before the clock check. P0's code is ready for it; until then mm is `clock: wall`, more than one mm exec at once on the working clock is refused, and the 10-case check stays open |
| D4 | Multimem had no remote, so `mmsb[mm]` was a path source with this machine's path in `pyproject.toml` | Resolved on 7 November 2026: the tool, now MultimediaSearch (`mms`), is public at github.com/DanMcInerney/MultimediaSearch, and `mmsb[mm]` is a git source pinned to a commit in `pyproject.toml` and `uv.lock`. Publishing MMSB (13.8) no longer needs a path |
| D5 | 13.2, CPU work beside a GPU job | Stays off; `--side-by-side` exists and is tested |
| D6 | What holds a GPU configuration to one exec at a time? (G2's blocker: only D3's refusal did, and the refusal offered `--clock wall`) | The owner's rule, built: a configuration whose system needs the GPU runs one exec at a time unless `--gpu-jobs N` (default 1) asks for more; `--jobs` and `--clock` do not. One rule in `core/atonce.py` for the runner, every dry run and estimate, the refusal and `clock-check` (8.4) |

**Open findings of the build.**
- `listing` is refused on a dump whose files are not at their time (3.2), V8 as downloaded included. To run it there the clips
  would need their recording times (copies, 41 GB, not links to the read-only download), the filter needs a rule for chance
  passes on multiple-choice questions, and V8's own report, which takes every configuration with a system other than `files` as
  tested and never reads `leak_filter`, would need to honour it (7.3). P1.
- mm's instructions are over the 2,000-token rule of 3.1, and nothing enforces it.
- Nothing guards two cells that share a home under `--side-by-side` (8.3).
- MultimediaSearch's four limits (10.3).
- The report's text still quotes the pilot's words about mm ("decider tier", "asks / finds / looks"), and the report reads the
  tool names `ask`, `find`, `look` and the file `ambient-labels.json` when they are there. The imports are neutral; the wording
  is not.

### 12.2 The commands that wait for Dan

Each needs his go-ahead and a free GPU, because the laptop's rule is that a GPU job runs alone:
`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` must print nothing.

**1. The pilot through the new runner (10.2 step 4).** Run it in US Eastern time: the pilot's times are local, and a recipe from
another zone is refused.

```
uv sync --extra mm --group mm-gpu
uv run mmsb repeat tests/oracles/recorded/pilot-s1/converted --dry-run
uv run mmsb repeat tests/oracles/recorded/pilot-s1/converted
uv run mmsb compare tests/oracles/recorded/pilot-s1/converted D:\mmsb-runs\<new run>
```

- **The defaults are the pilot's.** The `files` cells run `--jobs` (3) at once and `mm/d1-3B` runs one exec at a time
  (`--gpu-jobs`, default 1), as the pilot ran them (8.4). mm reports no queue, but one exec at a time needs none, so the repeat
  is not refused on the working clock. The dry run prints `(1 at once)` for the mm cells and says so. Do not pass `--gpu-jobs`
  above 1: the laptop's rule is one GPU exec at a time, after three mm execs at once crashed the NVIDIA driver on 2 November
  2026 (with it, the working clock refuses until D3, and `--clock wall` lets it through with a warning).
- `--group mm-gpu` installs torch 2.11.0+cu128, torchvision 0.26.0+cu128 and transformers 5.21.0 into this environment,
  because mm runs d1 in the python that started its daemon. It is several GB, and uv copies it, because uv's cache is on C: and
  the environment on D:; set `UV_CACHE_DIR` on D: first. It was not installed in P0.
- The path is `.../converted` (the raw `pilot-s1/run.json` names no scenario). `repeat` reads the converted run: the recipe,
  the configurations (`files`, `mm/d1-3B`), the questions of each cell, and each agent's model and effort (Codex
  `gpt-6.1-terra` at `medium`; Claude records no effort, so only its model is checked). It builds or finds the cached dump and
  store, runs `files` and `mm/d1-3B`, and prints the `compare` command. `--answer-format json` is the default.
  `--into D:\mmsb-runs\<new run>` continues a run that stopped. Read the dry run first: it builds and starts nothing.
- `mmsb compare` prints, for each cell, the correct counts of the recorded run and the new one against the band, the
  questions that flipped (with both answers), McNemar's p, the problems of each run, and the verdict lines, and then the four
  numeric conditions of D1. A person reads every flip.
- **Cost, from the pilot's measurements.** 110 execs, 13,044 session-seconds, 3,655 GPU-seconds (the dry run prints the same
  three). The tokens were 50.3M and the Claude cost $19.00 at list; the dry run prints 51.8M and $19.76, because three Claude
  timeouts report no cost and are counted at their cell's mean. Codex's use is on its plan, not in dollars. Wall time: the dry
  run prints about 2.0 hours at the means (0.7 at the medians), with `files` at 3 and `mm` at 1, and the build on top (11 minutes
  in the pilot, not measured in the new code). That estimate counts the `files` cells as perfectly parallel: it gives them
  8,589 session-seconds over 3, 48 minutes, and the pilot's took 99 (Codex 14, Claude 85). The `mm` cells take what the estimate
  says, 74 minutes (Codex 25, Claude 49). So the pilot took about 173 minutes, about 2.9 hours, and the re-run should take as long.
- If a cell falls outside the band, run it again with `--answer-format line` (another $19, about 3.6 hours), as D1 says.

**2. `mm+files` for the first time (10.2 step 4, last sentence).** After the re-run, in the same run folder:

```
uv run mmsb run D:\mmsb-runs\<new run> --agents codex,claude --configs mm/d1-3B+files --codex-effort medium
```

No cost was measured. A rough ceiling is the `files` cell plus the GPU: Claude about $16 and 2 hours, Codex about 45
minutes. Watch three things the offline tests could not show: that a real Claude accepts the second `--add-dir` for mm's
`looks` folder (if it refuses, a `Read` of a `look` picture is denied by Claude and not counted as an escape); that hiding CUDA
from the agent's environment (8.3) does not break mm's server; and how much slower `S+files` is than `files` alone.

**3. The clock check on 10 cases (8.4).**

```
$env:MMSB_CODEX_EFFORT = "medium"
uv run mmsb clock-check <new run> --agents codex,claude --config mm/d1-3B --cases 10 --gpu-jobs 2
```

(The first line is PowerShell. `clock-check` has no effort flag: Codex runs at `MMSB_CODEX_EFFORT`, else high, and the cost
below is the pilot's Codex at `medium`.)

It asks 10 cases of each agent alone and then queued, in the two folders of 8.4, and prints the answers, tokens and charged
seconds side by side. **It needs D3 first, and Dan's go-ahead.** With mm reporting no queue it refuses at once, naming D3, and
asks nothing; and without `--gpu-jobs 2` or more it refuses too, because mm runs one exec at a time and nothing would queue.
The flag is the go-ahead: the queued half is 2 mm execs at once, the only time the benchmark asks for more than one on the GPU
(three at once is what crashed the driver on 2 November 2026), and the command prints a warning when it starts. Watch the GPU
while it runs. Cost, from the pilot's means: 20 execs for each agent; Codex about 21 minutes, 960 GPU-seconds and 2.3M tokens;
Claude about 41 minutes, 2,080 GPU-seconds, about $2.25 and 3.3M tokens; together about 1 hour and 3,040 GPU-seconds. These are
lower bounds: queued execs take longer.

## 13. Open questions for Dan

1. **Licences:** decided on 4 November 2026: not a filter for use on this machine. Gated sources still need Dan
   ([MEDIA.md](MEDIA.md), "Decisions for Dan").
2. **CPU work beside a GPU job:** may GPU-free configurations and builds run while a system holds the GPU? Still
   unanswered. It stays off: `mmsb run --side-by-side` exists and is tested, and nothing turns it on by itself (8.3).
3. **Where mm's plug-in lives:** decided for P0: in MMSB as `systems/mm`, with MultimediaSearch as the optional `mm` extra, a git source
   pinned to a commit in `pyproject.toml` (D4, section 12.1).
4. **Which other systems first:** `embed` (SigLIP or CLIP, plus CLAP) and `transcript` (Whisper)? Both download models.
5. **Hosted systems:** may a paid long-context model run on the public scenarios?
6. **Dan's own media and scripted recordings:**
   - a phone dump, screenshots and voice memos (I1, I2, M1, X2);
   - a scripted day (X1);
   - a scripted set of voice notes (M2).

   All with questions he writes, kept private and never published.
7. **Sources:** on 4 November 2026 every public scenario got a starter source. Some fall short: V1 cannot reach 48 h
   per camera, A5 has no long base, and M5 has frogs, not owls. The gaps and the decisions they need are in
   [MEDIA.md](MEDIA.md).
8. **Publishing:** will MMSB be public? That decides the dev/test policy for the real-data scenarios, and the project's
   own licence.
