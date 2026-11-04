# MultiMediaMemoryBench (MMMB): design

Status: proposal, 4 November 2026. Nothing in it is built. This folder holds a copy of Multimem's benchmark, which tests one
system (mm) and cannot run without it. This document makes MMMB a general benchmark, and section 10 says how the copied code
gets there. Section 8.1 is measured; the rest is design, and every estimate is marked as one.

## 1. What MMMB is for

**The task:** a user hands an AI agent a large dump of their own media and asks for something in it. The dump might be an
NVR export, a phone's camera roll, a day of microphone recording, a folder of robot logs or a meeting archive. The question
might be "when did the courier come?", "what level did I park on at the airport?" or "did the smoke alarm go off while I was
out?".

**The question MMMB answers:** with each search system attached, how often does an agent answer right, with evidence, how
soon, and at what cost? A search system is anything that ingests a dump and gives the agent tools to search it, for example:
- a local decision model that scans windows (Multimem);
- an embedding index;
- a transcript search;
- a long-context model.

The baseline is the agent alone with the files and its own tools.

MMMB is neutral between systems. No system's code is in the core. Every system plugs in through one contract (section 3) and
is scored by the same rules.

**What it measures, per question and configuration:**

| Measure | Why |
| --- | --- |
| Grounded accuracy: the answer is right and the evidence it cites (file and time) holds it | a right answer with the wrong evidence is a guess |
| Seconds to answer | the user waits for it |
| Agent tokens and dollars | the user pays for them |
| System cost per question: GPU seconds, API dollars | local GPU time is free in money, not in time or heat |
| Ingest cost per dump: seconds, GPU seconds, disk, files refused or misdated | the user pays it once per dump, before the first answer |

## 2. How it is built

Six layers. Only the last one knows any system.

| Layer | Holds | Knows about systems? |
| --- | --- | --- |
| **Dumps** | folders of real files as a user would hand them over, with a manifest of hashes | no |
| **Questions** | one JSONL file per scenario: the question, its "now", the truth with evidence spans, the grading rule, tags (section 6) | no |
| **Grader** | rules only, no model grades another | no |
| **Agents** | locked-down Claude Code and Codex sessions, one per question and configuration | only through the tools a system hands over |
| **Runner** | scheduling, resource pools, budgets, resume, statistics, report | only through the contract |
| **Systems** | one plug-in per system: `files`, `listing`, `embed`, `transcript`, `mm`, ... | each knows itself |

**Proposed layout:**

```
MultiMediaMemoryBench/
  DESIGN.md  README.md  pyproject.toml          # mmmb[mm], mmmb[embed]: each system's dependencies are an optional extra
  mmmb/core/        runner, schedule, agents/claude.py, agents/codex.py, grade, probe, stats, report
  mmmb/systems/     files/  listing/  embed/  transcript/  mm/        # one folder per plug-in
  mmmb/scenarios/   synthetic_week/  egolife/  security/  household/  ...   # recipe, builder, questions
  results/          <date>-<scenario>/<agent>/<config>/                  # published reports
```

Dumps live in `D:\bench-data` (downloaded once, content-hashed). Runs go to `D:\mmmb-runs`.

**The test of neutrality:** the core, the `files` and `listing` systems and every scenario install and run on a machine where
Multimem has never been installed. A CI check imports `mmmb.core` and each scenario with no extras.

## 3. Systems

### 3.1 The contract

A system is a folder under `mmmb/systems/` with a `system.toml` and a class with these operations:

| Operation | Does | Rules |
| --- | --- | --- |
| `describe()` | name, version, media it reads (image, video, audio, text), whether it is local, what it needs (GPU, GPU memory, network), licence | the version is recorded with every result |
| `ingest(dump, store)` | builds the system's own state in `store` from a read-only dump | sees the dump only, never the questions; timed; returns seconds, GPU seconds, bytes written, and each file refused or given a time it could not believe, with the reason |
| `serve(store, now)` | returns the agent's tools: an MCP server's command, args and environment, plus the system's own instructions text | only media before `now` may be answered from. A system that cannot cut by time declares `horizon = "view"`, and the runner ingests a view of the dump up to each distinct `now` instead |
| `reset(store)` | returns the store to its state after ingest, between questions | or `reset = "copy"`: the runner hard-links a fresh copy of the store for each question |
| `usage()` | GPU seconds, GPU queue seconds and API dollars since the last call | the runner cross-checks GPU seconds against its own sampling of the GPU |
| `oracle(query, since, until, k)` | optional: the system's own ranked moments for a plain query, with no agent | lets the retrieval tier (section 8) compare systems without paid agent runs |

**Rules every system follows:**
- **Ingest is blind.** It runs before any question is known. It may take as long as it likes, and the time is reported as the
  user's wait.
- **Systems are frozen before a test run.** Their version and configuration hash go with each result. Tuning happens on the
  dev split only (section 6.4).
- **Instructions are general.** A system's instructions text is at most about 2,000 tokens and the same for every scenario.
  It may teach the agent how to use the tools, never what is in a scenario.
- **No network,** unless the system declares itself hosted. Hosted systems run only on public scenarios, never on
  private ones, and their API dollars are reported.
- **GPU work goes through the runner.**
  - In `exclusive` mode (the default), the scheduler gives the GPU to one system at a time.
  - In `lease` mode, the system asks the runner for the GPU around each piece of work, through a small local endpoint given
    in its environment. The runner then knows queue time exactly (section 8.2).

### 3.2 The systems to ship

A benchmark with one contestant tests that contestant. MMMB ships cheap, common baselines beside mm:

| System | What it is | Tools it gives the agent | Ingest |
| --- | --- | --- | --- |
| `files` | the baseline: no system | none beyond the agent's own (shell, file reads, ffmpeg) | none |
| `listing` | the floor and the leak filter (section 7.3) | a listing of names, sizes, times, EXIF and container fields; no pixels or sound | seconds |
| `embed` | the common "multimodal RAG" setup | `search(text or picture, since, until, k)` over frames at 1 fps and photos (SigLIP or CLIP) and 5 s sound windows (CLAP); `frame(file, t)` | GPU: every frame and window once |
| `transcript` | speech made searchable | `search(words, since, until)` over a timestamped transcript (Whisper); `listen` is not offered | GPU: all sound once |
| `mm` | Multimem: a small local model reads each window in a range against the agent's question at ask time | `find`, `ask`, `look` (`label` off) and mm's skill | CPU only (indexing); the GPU works at question time |
| hosted (later) | a long-context model given the dump in chunks | `ask(question, files)` | none; API dollars per question |

`embed` and `transcript` pay the GPU once at ingest, and `mm` pays it on every question. The report shows both, so a user can
see which is cheaper for one question or for fifty.

Each model inside a system is a variant: `mm/d1-3B`, `mm/d1-Omni-2B`, `embed/siglip2-base`. Every variant is reported
apart.

## 4. Configurations and agents

Each question runs in a fresh agent session per configuration:

| Configuration | The agent has | Role |
| --- | --- | --- |
| `listing` | the listing only | the floor; a question it answers is flagged as answerable from metadata or prior knowledge |
| `files` | the files and its own tools | the baseline |
| `S+files` | its own tools and system S's tools | **the headline for each system**: what a user who installs S has |
| `S` | S's tools only | diagnostic, smoke tier only: what S's tools do alone |
| `gold` | the evidence clips alone | diagnostic, smoke tier only: a miss here is the agent's reading, not the search |

**Agents:** Claude Code and Codex for now. Each runs headless, locked down (no network, no access outside the run, no skills
or plugins of the owner's), with the same model, effort and turn cap in every configuration. The prompt skeleton is shared:
the user's question, the date and time "now", where the dump is, and the answer format (section 6.2). A system adds its own
instructions and tool schemas. Other agents can be added through the same runner.

The headline is `files` against `S+files` for each system S. Taking the agent's shell away is not what any user does, so it
is a diagnostic only.

## 5. Scenarios

Each scenario is a **dump** plus 15-40 questions about it. Questions share the dump, so it is built, ingested and probed once
per system.

| # | Scenario | The user says | The dump | Built from | Priority |
| --- | --- | --- | --- | --- | --- |
| S1 | Home security export | "Someone took my parcel this week. Here's the export from my 3 cameras." | 3 cameras × 2-3 days, as an NVR exports them: 1-hour MP4s, or one 4-8 GB file per camera per day; day, night and IR | MEVA (328 h, overlapping cameras, 37 activity types with time spans) | 1 |
| S2 | Household sound | "Did the smoke alarm go off while I was away?" | 1-3 days of one room microphone, one file a day (24 h of mono 16 kHz WAV is about 2.7 GB) | a real home's week of ambience (SINS) with real events (FSD50K) mixed in at a set SNR; TV, music and near misses as decoys | 1 |
| L | Large-file track | (the above, bigger) | the same needles in one 4 GB video, one 10 h sound file, a 50 MP panorama, a 30 GB folder | re-muxed from S1, S2, S3 | 1 |
| S0 | Scripted real day | "When did the courier come? Did I leave the hob on?" | a day of Dan's home on his own cameras, phone and microphone, with about 30 planted events | the script is the truth (M3-Bench's method); private | 2 |
| S3 | Phone camera roll | "I dumped my phone. Find the photo of the parking sign at the airport." | 5-10k photos (HEIC, JPEG, PNG screenshots), 200 videos (rotated HEVC MOV, slow motion, a few over 2 GB), 50 voice memos, WhatsApp copies without EXIF, duplicates | YFCC100M single-user photostreams; DISBench (57 users, 122 verified queries); Dan's own phone (private) | 2 |
| S4 | Robot logs | "Find the episodes where the gripper dropped the cup." | a few hundred episodes, 2-3 cameras each, an MP4 per camera | DROID (an instruction per episode) | 2 |
| S5 | Meetings and lectures | "When was the cost slide shown? Who drew on the whiteboard?" | 20-40 h of meeting video and sound, 30-90 min files, slides | AMI Meeting Corpus (topics, decisions, slides) | 3 |
| S6 | Dashcam or road trip | "When did it start raining? Where did we stop for fuel?" | 8-10 h in 1-3 min chunks, camera clock set wrong | comma2k19 (GPS and CAN, no event labels) | 3 |
| S7 | Field recorder | "When did the fox first show up on the trap camera? When did the owl call?" | 20k trap images (90% empty, wind triggers), 3 nights of sound | Caltech Camera Traps; 33 annotated one-hour bird soundscapes | 3 |
| S8 | Developer screen | "When did the borrow-checker error first appear?" | a synthetic week of screen recording (the copied generator), plus real screen video | the copied `haystack.py`; AgentNet's human screen recordings | kept |
| S9 | Egocentric days | as EgoLifeQA | 3 days of glasses video | EgoLife, with open answers and evidence where its annotations allow | kept |

**What makes each one hard, on purpose:**
- S1: night and IR, rain, headlights, many people and cars that are not the one asked about, the same event on two cameras
  at different times, and a small subject.
- S2: needles from -5 to +10 dB around the ambience, TV and music that hold alarms and doorbells, and a dog that barks at
  everything. Agents cannot hear, so the sound needles come from a mix list or a script, never from a listen.
- L: a needle a minute before the end of a 10-hour file, needles that straddle any fixed cut (1 s, 5 s, 60 s), a file larger
  than 2 GB, and files whose only time is their mtime.
- S3: burst shots and near-duplicates, screenshots of text, and photos whose time must come from the file, not EXIF.
- S4: hundreds of near-identical episodes, where only the outcome differs.
- S5: speech, which some systems handle and some do not. Speech questions are reported apart.
- S6: a camera clock wrong by a known offset. "What time" means the true time.
- S7: thousands of empty frames, animals at the edge of the frame or at night.

### Question kinds

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

**Tags on every question,** and the report cuts every measure by each:
- haystack size (hours, files);
- certificate length (seconds of media needed to check the answer: EgoSchema);
- look-back (how long before "now" the evidence is: under 2 h, 2-6 h, 6-24 h, over 24 h);
- evidence pieces and how far apart they are;
- needs audio, needs speech, needs reading small text;
- needle salience (share of the frame, or SNR);
- kind, modality and how vague the time hint is (a part of a day, a day, none).

## 6. Questions, answers and grading

### 6.1 The question file

One JSON object per line, kept outside the dump and never visible to a system or an agent:

```json
{"qid": "security-parcel-last-1", "scenario": "security", "split": "test",
 "now": "2026-11-03T21:00:00+00:00", "question": "When was a parcel last left at the front door?",
 "attachments": [], "kind": "last", "hint": "week",
 "truth": {"value": "2026-11-02 07:30", "evidence": [{"file": "cam1/2026-11-02_07.mp4", "t0": 1788.0, "t1": 1802.0}],
           "wrong": [{"file": "cam1/2026-11-01_18.mp4", "t0": 410.0, "t1": 425.0}]},
 "grade": {"rule": "time", "tol_s": 120}, "tags": {"lookback_h": 37.5, "certificate_s": 14, "salience": 0.03}}
```

Evidence spans are in seconds into the file. The scenario's manifest maps each file to its true start time, so the
grader can read a cited time either way.

### 6.2 The answer

Every agent ends with one JSON object:

```json
{"answer": "Monday 07:30", "evidence": [{"file": "cam1/2026-11-02_07.mp4", "t0": "07:29:50", "t1": "07:30:05"}]}
```

A cited span may be wall-clock times or seconds into the file. Systems that work in wall-clock time and agents that work in
file offsets are graded alike.

### 6.3 Scores

- **Grounded accuracy** (the headline): the value passes its rule and every cited span overlaps a truth span within the
  kind's tolerance (10 s for video, 5 s for sound, the same file for a picture). A right value with wrong evidence is
  counted as `lucky` and shown apart.
- **Find all and count:** event F1 by distinct moments, as DCASE scores sound events (with 1 s and 5 s collars in place of
  DCASE's 200 ms). A count is also graded exactly.
- **Absent:** right only with "not there" and no evidence. The report counts the searches each absent question drew.
  Deep Video Discovery's main failure is the "search trap": searching again and again for what is not there.
- **Time:** a time-to-correct curve per configuration, the share answered correctly within 30, 60, 180 and 600 s. Beside it,
  a score in the style of the Lifelog Search Challenge, max(0, 100 - 50·t/T - 10·wrong), with `wrong` counting cited spans
  that hold nothing. It is never shown instead of accuracy.
- **Cost:** agent tokens and dollars, system GPU seconds and API dollars, per correct answer. Ingest cost is reported per
  dump, and amortised over 1, 10 and 100 questions. The report draws a Pareto plot of accuracy against seconds and dollars.

### 6.4 Dev and test

Every scenario has a public **dev** split and a **test** split, about 30% and 70% of its questions. Any system, mm included,
is built and tuned on dev only. Test results are published only from a frozen system version.
- For the synthetic week, seeds 1-10 are dev and the test seeds are drawn fresh per release. mm's skill and the copied
  harness were adjusted after runs on pilot seed 1, so that seed is dev.
- For public datasets, which questions are in test stays private to the run folder.

## 7. Building the dumps

### 7.1 Ways to know the answer

1. **Annotated real data.** A dataset's own time spans (MEVA's activities, AMI's topics and decisions, DROID's episode
   instructions, camera-trap species labels) become questions through templates in user wording. Each is checked by rule
   against the spans, and Dan spot-checks 10%.
2. **Real needles in real backgrounds,** where annotations do not cover what users ask.
   - Sound: real event clips mixed into real ambience at a set SNR, with exact onsets (the Scaper method).
   - Video: real clips spliced at natural cuts, or a real object composited with blending. The probe must show that a splice
     is not the most unusual moment in its scope.
3. **Owner-written questions about the owner's own data** (S3 with Dan's phone, S4 with his robot logs). ScreenshotVQA,
   camroll and ATM-Bench all chose this over LLM-written questions as more natural. These stay private.
4. **A scripted real day (S0):** Dan follows a script of about 30 planted events while his own devices record. This gives
   real light, real noise and real codecs, with the truth known by construction, and sound truth that nobody has to hear.
5. **Real footage on a plausible schedule** for the synthetic week (X-LeBench's method): its porch camera becomes real clips
   (MEVA, S0) laid on the plan's days, with the plan's needles and decoys.

### 7.2 Real dumps

Dumps keep what real ones have:
- sizes from 1 MB photos to multi-GB videos and 10-hour sound files;
- codecs and containers (H.264, HEVC, MP4, MOV, MKV, M4A, MP3, WAV, FLAC, Opus);
- variable frame rate, rotation flags and HDR;
- HEIC, JPEG and PNG;
- EXIF present, stripped or wrong, time zones and daylight saving;
- duplicates, nested folders, and names like `IMG_4821.HEIC` and `VID_20250612_181502.mp4`.

### 7.3 The shortcut probe and leaks

**The shortcut probe** (from the copied `shortcuts.py`) checks every dump before any run. No file name, file size, EXIF or
container field, frame difference, splice seam, colour, loudness or frequency band may single out a needle.

**Leaks:**
- No network for agents or local systems, so no one looks up a public dataset's answers.
- Files get the names a phone or NVR would give them, and dataset identifiers are removed.
- The `listing` configuration runs on every scenario. A question it answers is flagged and kept out of the headline. This is
  the blind filter of LVBench and HourVideo, and Daily-Omni's text-only filter, applied to what a dump really leaks.
- Multiple-choice questions (EgoLifeQA) are asked in 4 rotations and count only if all 4 are right (VNBench).

### 7.4 The ingest track

Once per scenario per system version: seconds to ingest, GB per minute, GPU seconds, disk, files refused or misdated and
why, and seconds from the end of the dump to the first answer. Every system meets large files here. The copied pilot already
found two limits in mm this way (section 10.3).

### 7.5 Scaling curves

Each S1, S2 and S8 needle also sits in haystacks of 1 h, 4 h, 1 day and 3 days of the same source (Visual Haystacks,
TimeScope). The curve shows, per system and agent, from what size of dump a system pays off against `files`. A system that
pays once at ingest and one that pays per question cross at different sizes.

## 8. Speed

### 8.1 Where the time went in the copied pilot (measured)

One system (mm with d1-3B) against `files`. 3 synthetic days, seed 1, `D:\mm-bench\20261104-0314-pilot-s1`:

| Step | Wall time | Bounded by |
| --- | --- | --- |
| build (write, load, probe) | 11 min 23 s | CPU, mostly one process |
| Codex `files`, 31 execs, 3 at a time | ~14 min (1,896 exec-s) | the API |
| Claude `files`, 31 execs, 3 at a time | ~85 min (6,693 exec-s; its longest exec hit the 900 s limit) | the API, and brute force over frames |
| Codex `mm`, 24 execs, 1 at a time | ~25 min: 1,513 exec-s, of which **1,156 GPU-s (76%)** | the GPU |
| Claude `mm`, 24 execs, 1 at a time | ~49 min: 2,942 exec-s, of which **2,499 GPU-s (85%)** | the GPU |

The run took about 3 hours of work, and 4 with a driver crash and a reboot. The laptop's NVIDIA driver has crashed under one
GPU job and under three, so **exactly one GPU job runs at a time**, whatever the system. A per-question GPU system already
keeps the GPU busy 76-85% of the time, so running more agents against it saves little. The large savings are in:
- running GPU-free work beside GPU work;
- ingesting once;
- not spending runs on questions that do not separate configurations.

### 8.2 The plan

**a. Three resource pools and one scheduler.**

| Pool | Size | Work |
| --- | --- | --- |
| CPU | at most 4 processes (the laptop's rule) | download, re-mux, generate, CPU ingest, probe |
| Agents | per CLI, up to 4 sessions (quota and rate limits) | every configuration's agent turns |
| GPU | exactly 1 job, for every system together | GPU ingest (`embed`, `transcript`), question-time GPU (`mm`), the retrieval tier |

Configurations that need no GPU at question time (`listing`, `files`, `embed+files` once ingested) run beside those that do.
The longest expected questions start first. Ingest of the next system or scenario fills the GPU's idle gaps. The speed
research puts the saving of running side by side at 25-35% of a run's wall time.

Two conditions:
- **Dan's go-ahead.** CPU-heavy work beside a GPU job goes against the laptop's current "GPU runs alone" rule.
- **Agents stay off the GPU:** CUDA is hidden from their environment, and `-hwaccel` and `nvidia-smi` are on their deny
  list. An agent decoding on the GPU would be a second GPU user.

The copied harness's error classes stay: `setup`, `limit` and `auth` results are retried and left out, and `escaped`,
`timeout` and `capped` are wrong answers. A system that dies, or a lost GPU, is `setup`.

**b. Working time, not wall time.** When several questions share the GPU, waiting for it is an artefact of the benchmark. A
real user's system has the GPU to itself. So:
- A question is charged its agent's turns plus its own GPU service time, never queue time. This is Inspect AI's
  `working_limit`, paired with a wall-clock cap: 900 working seconds and a 2-3× wall cap. In `lease` mode the runner
  measures the queue itself. In `exclusive` mode the system reports it through `usage()`, as vLLM reports queue time apart
  from service time.
- A system whose tool returns early with partial results counts that time from when its GPU work starts, not from when the
  request was queued, so a queued request does not come back empty and invite polling.
- Before trusting it, run 10 questions both alone and queued, and compare answers, tokens and charged seconds.

**c. Ingest once.**
- Downloads are fetched once and content-hashed.
- Each prepared dump is cached by the hash of its recipe.
- Each system's store is cached by (system, version, configuration, dump hash), and hard-linked per question.
- A system or recipe change rebuilds only what it touches.
- The synthetic generator writes days and sources in up to 4 processes.

**d. GPU seconds are the system's business, and the bench reports them honestly.** Caches inside a system (encoder outputs,
answer caches) are allowed if they would also serve a real user. Across questions, a cache may only keep what ingest built,
so no question gains from another's run. Batching inside a system is the system's own choice, and the report records it.

**e. Tiers.** Targets are estimates:

| Tier | What | Resources | Target |
| --- | --- | --- | --- |
| T0 | unit tests; recorded runs re-graded and re-reported (Harbor's `regrade`); recorded tool results replayed at their recorded latency | CPU | minutes |
| T1 | retrieval tier: each system's `oracle()` per needle, no agent; recall at k, AP, seconds | GPU | under 1 h per system per scenario, cached |
| T2 | smoke: one scenario's 1 h rung, 8 discriminating questions, one agent, every configuration | all | under 20 min |
| T3 | standard: every scenario, both agents, `files` and each `S+files`, one pass | all | one night |
| T4 | significance: T3 plus sequential top-up and repeats | all | a weekend |

**f. Budgets sized to the haystack:** 300 working seconds for an hour of media, 900 for a day or more. A question that
T1 shows no system can scan within its budget gets its hint narrowed or is dropped from T2 and T3.

**g. Session mode** (a second track): one agent session gets 5 questions about one dump in a row, as a user asks them, so
startup and learning the dump are paid once. Reported apart, since its questions are not independent.

## 9. Statistics

Each system is compared with `files` per agent, question by question (paired, McNemar). The pre-registered comparisons are
those pairs, corrected for multiplicity with Holm's method. Rankings between systems are given with bootstrap intervals over
questions.

**How many questions.** Paired questions per agent and comparison, at 80% power and α = 0.05 (Connor's approximation for
McNemar):

| Effect, and share of pairs right in one configuration only | Questions |
| --- | --- |
| +17 points, all discordant pairs one way (Claude's pilot), optimistic | about 45 |
| +15 points, 20% discordant | about 70 |
| +10 points, 15-20% discordant | about 115-155 |
| +4 points, 8% discordant (Codex's pilot) | about 390 |

So MMMB aims at effects of 10 points or more, from 70-150 media questions per run across scenarios. A smaller effect is
reported as not certified, not implied.

- **Distinct questions, not repeats.** Repeating a question shrinks only its own run-to-run variance, never the spread between
  questions (Miller, "Adding Error Bars to Evals"). Repeats go to a reliability set of 10 questions × 3 runs, reported as
  pass^k (τ-bench).
- **Drop questions that carry no information.** A question right in every configuration, or wrong in every one, in two
  earlier runs leaves the routine tiers. Two stay as canaries. Results are then labelled "on discriminating questions". The
  speed research puts the saving at 30-60% of runs.
- **Sequential stopping, pre-registered:** questions run in a random order, and a comparison stops when an anytime-valid rule
  (an e-process or a confidence sequence) decides it, capped at 150 questions. It saves most when the effect is large.

## 10. Where the copied code stands, and how it becomes MMMB

### 10.1 What is tied to Multimem today

| File | Lines | Tie to Multimem | Becomes |
| --- | --- | --- | --- |
| `grade.py` | 290 | none | `core/grade`, plus grounded scoring |
| `shortcuts.py` | 643 | `RATE` from Multimem's `tests/day.py` | `core/probe` |
| `haystack.py` | 1,814 | `multimem.reader` to check written frames and count mm's windows; `load` stores files through mm's `remember`; helpers from `tests/` (`day`, `fake_decider`, `conftest`) | `scenarios/synthetic_week`: the plan, files and questions. `load` and the window count move to `systems/mm`, frame checks use PyAV, and the helpers' constants are copied in |
| `haystack_media.py` | 1,357 | `multimem.config.ffmpeg_exe`; `tests/day` fonts | `scenarios/synthetic_week/media`, with its own ffmpeg lookup |
| `claude_run.py` | 511 | the `mm` arm's MCP server, tool names and prompt | `core/agents/claude`: tools and instructions come from `serve()` |
| `codex_run.py` | 636 | the same, mm's server instructions, and mm's skill read from the Multimem repo | `core/agents/codex`, likewise |
| `bench.py` | 2,390 | mm's daemon, snapshot homes, scrubbing mm's index between questions, cancelling mm's jobs, mm's GPU estimate, the decider tier, `tests/conftest` | split: `core/runner` (exec loop, slots, deny lists, resume, error classes, statistics, report) and `systems/mm` (everything about mm's daemon, which becomes `ingest`, `serve`, `reset`, `usage` and `oracle`) |
| `egolife.py` | 1,229 | `multimem` config, media, reader and `ask` | `scenarios/egolife`, with mm's parts in `systems/mm` |
| `README.md`, `results/` | | the published mm-against-files results | `results/2026-11-0x-*` labelled as system `mm`; a new README for MMMB |

### 10.2 The order of the work

1. Copy in the few helpers from Multimem's `tests/` (font, sample rate, the shade and pitch constants) and give
   `haystack_media` its own ffmpeg lookup. The synthetic week then builds with no Multimem.
2. Write the question-file format, and make the synthetic week and EgoLife write it.
3. Split `bench.py` into `core/runner` and `systems/mm`, and make the agent runners take tools from `serve()`.
4. Re-run the copied pilot through the new runner: the same verdicts for `files` and `mm` (the copied pilot's mm-only
   configuration) show that nothing was lost. Then `mm+files` runs for the first time.
5. Add `listing`, then `embed` and `transcript`.

### 10.3 What mm's plug-in will meet first

The ingest track (section 7.4) and track L will exercise two limits of mm today. These belong to mm's notes, not to MMMB's
design:
- `remember` refuses any file over 2 GB (`remember_max_mb`).
- A folder source keeps a file over 50 MB in place, dated when mm saw it rather than by its recording time.

Track L is built so that any system's handling of big files shows the same way.

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

**What none of them does, and MMMB does:**
- Hand an agent a real dump at real size and in real mess.
- Compare search systems behind one plug-in contract, each against the agent's own tools on the same files.
- Report the ingest wait and GPU seconds beside tokens and dollars.

LSC and VBS compare many systems on one task, but with people at the controls, not agents. No benchmark of continuous
screen-video memory was found.

### 11.2 What to take

| # | Idea | From | In MMMB |
| --- | --- | --- | --- |
| 1 | Grounded scoring: correct only if the cited file and span hold the answer | CG-Bench, Ego4D NLQ, TVR, ATM-Bench (Joint@k) | section 6.3 |
| 2 | Real footage on a plausible schedule, real needles spliced at set depths | X-LeBench, TimeScope, MLVU | 7.1 |
| 3 | Real sounds mixed into real rooms at a set SNR, with exact onsets; near-miss decoys | Scaper, DESED, RealDESED | S2 |
| 4 | Scripted real capture, the truth known by construction | M3-Bench | S0 |
| 5 | Haystack-size ladder; the short rungs are smoke runs | TimeScope, Visual Haystacks, V-NIAH (by depth) | 7.5, T2 |
| 6 | Absent questions, about a quarter; count the "search trap" | PhotoBench, MMNeedle, Deep Video Discovery | 5, 6.3 |
| 7 | Leak filters: no media, metadata only, rotated options | HourVideo, LVBench, Daily-Omni, VNBench | `listing`, 7.3 |
| 8 | Penalise time and wrong evidence | LSC, VBS | 6.3 |
| 9 | Find-all by distinct moments; graded relevance | VBS, MultiVENT 2.0, DCASE | 6.3 |
| 10 | Tags: certificate, look-back, needs-audio, evidence spread | EgoSchema, Video-MME, EgoLifeQA, EgoMemReason, TeleEgo | 5 |
| 11 | Multi-hop across files and days; aggregates | DISBench, LongVidSearch, TimelineQA, MVU-Eval | 5 |
| 12 | Query by example | Ego4D VQ2D, MomentSeeker | 5 |
| 13 | Owner-written questions | ScreenshotVQA, camroll, ATM-Bench | 7.1 |
| 14 | Gold-evidence configuration | LongVidSearch (GPT-5 near-perfect with gold clips, 42% without) | 4 |
| 15 | "As-of" questions: each has a now; one dump serves many questions | TeleEgo, M3-Bench, EgoLifeQA | 3.1 `serve(now)`, 6.1 |
| 16 | Score retrieval apart from the agent | DCASE PSDS, LV-Haystack | T1, `oracle()` |
| 17 | One plug-in contract, many systems, one scorer | LSC, VBS, TRECVID AVS | 3 |
| 18 | Resource groups, working-time limits, error classes, regrading | Harbor, Inspect AI, SWE-bench | 8.2 |
| 19 | Accuracy against cost on a Pareto plot | AI Agents That Matter, HAL | 6.3 |

**Taken as warnings:**
- AudioMarathon builds long audio by concatenating clips. Its seams are the artificial shortcut MMMB avoids.
- TV Needle concatenates videos into one. MMMB keeps files separate, as a user's are.

### 11.3 Datasets

| Dataset | Licence and gating (as found) | Size and annotations | For |
| --- | --- | --- | --- |
| [MEVA](https://registry.opendata.aws/mevadata/) | CC BY 4.0; open on AWS, no account | ~328 h multi-camera video; 37 activity types with spans for ~184 h | S1; real porch clips for S8 |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC BY 4.0 for signals, transcripts and some annotations | 100 h; room video, close and far microphones, slides; transcripts | S5 |
| [FSD50K](https://zenodo.org/records/4060432) | CC BY 4.0; per clip CC0, BY or BY-NC; no account | 51K clips, 200 classes (doorbell and alarm classes: check `vocabulary.csv`) | S2 events |
| [SINS](https://zenodo.org/records/2555143) | licence in a PDF, unread | a continuous week in a real home, 13 nodes × 4 microphones, activity labels | S2 ambience |
| [DROID](https://arxiv.org/abs/2403.12945) | CC BY 4.0 | 76K episodes, 350 h, 3 cameras, an instruction each | S4 |
| YFCC100M, or [DISBench](https://arxiv.org/abs/2602.10809) on it | per-photo CC; original metadata under Yahoo's Webscope agreement (reports conflict) | real users' photos with EXIF; DISBench: 57 users, 109K photos, 122 verified queries | S3 |
| [Bird soundscapes](https://zenodo.org/records/7050014) | CC BY 4.0; no account | 33 one-hour recordings, 20,147 timed calls | S7 sound |
| [Caltech Camera Traps](https://lila.science/) | CDLA-Permissive | 243K images, 140 places (time fields unverified) | S7 images |
| [comma2k19](https://github.com/commaai/comma2k19) | MIT | 33 h of driving, ~100 GB, GPS and CAN, no event labels | S6 |
| [AgentNet / OpenCUA](https://huggingface.co/datasets/xlangai/AgentNet) | MIT per its card; gating unverified | 22.6K human screen-video tasks with input events | S8 |
| EPIC-KITCHENS-100 + EPIC-SOUNDS | CC BY-NC 4.0 | 100 h egocentric with sound; 78K timed sound events | a kitchen scenario, if non-commercial suits |
| Gated: VIRAT, Ego4D, LSC/NTCIR, BirdCLEF, V3C | an agreement, an application or an account | | not without Dan |

Dan reads each licence before anything is downloaded. No agent accepts a licence or makes an account.

## 12. Phases

| Phase | What | Done when |
| --- | --- | --- |
| P0 | The separation (section 10.2, steps 1-4); the question-file format; grounded answers; `listing`; the cached build; GPU-free configurations beside GPU ones; the working-time clock checked on 10 questions | the core runs without Multimem installed, and the copied pilot gives the same verdicts through `systems/mm` |
| P1 | `embed` and `transcript`; track L; S2 household sound; S1 home security | each scenario has 30+ questions, a passing probe, and a T1 run for every system |
| P2 | S0 scripted day, S3 phone roll, S4 robot logs; scaling curves; session mode | the T3 run has 70+ discriminating questions per agent |
| P3 | S5 meetings, S6 dashcam, S7 field recorder; hosted systems if wanted | as P1 |

## 13. Open questions for Dan

1. **Licences:** each dataset in section 11.3, read and accepted by Dan.
2. **CPU work beside a GPU job:** may GPU-free configurations and builds run while a system holds the GPU?
3. **Where mm's plug-in lives:** in MMMB as `systems/mm` with Multimem as an optional dependency (proposed), or in Multimem?
4. **Which other systems:** `embed` (SigLIP or CLIP, plus CLAP) and `transcript` (Whisper) as the first two? Both download
   models.
5. **Hosted systems:** may a paid long-context model run on the public scenarios?
6. **Dan's own data:** a phone dump, robot logs and a scripted day, with questions he writes, kept private and never
   published.
7. **Publishing:** will MMMB be public? That decides the dev/test policy for the real-data scenarios, and the project's own
   licence.
