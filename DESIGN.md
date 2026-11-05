# MultiMediaMemoryBench (MMMB): design

Status: proposal, 4 November 2026. Nothing in it is built. [VISION.md](VISION.md) says what MMMB is for; this file says
how. This folder holds a copy of Multimem's benchmark, which tests one system (mm) and cannot run without it. Section 10
says how that code becomes MMMB. Section 8.1 is measured; the rest is design, and every estimate is marked as one.

## 1. Overview

**The task:** a user hands an AI agent a large dump of their own media and asks for something in it ("when was the parcel
taken?", "find the voice memo where I said the gate code"). A **search system** is anything that ingests a dump and gives
the agent tools to search it:
- a local model that scans time windows (Multimem);
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
| **Systems** | one plug-in each: `files`, `listing`, `embed`, `transcript`, `mm`, ... | each knows itself |

**Proposed layout:**

```
MultiMediaMemoryBench/
  VISION.md  DESIGN.md  README.md  pyproject.toml   # mmmb[mm], mmmb[embed]: each system's dependencies are an optional extra
  mmmb/core/        runner, select, schedule, agents/claude.py, agents/codex.py, grade, probe, stats, report
  mmmb/systems/     files/  listing/  embed/  transcript/  mm/            # one folder per plug-in
  mmmb/scenarios/   V1-security/  A1-home-audio/  M4-samples/  ...        # recipe, builder, questions
  results/          <date>-<selection>/                                   # published reports
```

Dumps live in `D:\bench-data` (downloaded once, content-hashed). Runs go to `D:\mmmb-runs`.

**The neutrality check:** the core, the `files` and `listing` systems and every scenario install and run on a machine
where Multimem has never been installed. A CI check imports `mmmb.core` and each scenario with no extras.

## 3. Systems

### 3.1 The contract

A system is a folder under `mmmb/systems/` with a `system.toml` and a class with these operations:

| Operation | Does | Rules |
| --- | --- | --- |
| `describe()` | name, version, media it reads (image, video, audio, speech, text), whether it is local, what it needs (GPU, GPU memory, network), licence | the version is recorded with every result |
| `ingest(dump, store)` | builds the system's own state in `store` from a read-only dump | sees the dump only, never the questions; timed; returns seconds, GPU seconds, bytes written, and each file refused or given a time it could not believe, with the reason |
| `serve(store, now)` | returns the agent's tools (an MCP server's command, args and environment) and the system's own instructions | only media before `now` may be answered from. A system that cannot cut by time declares `horizon = "view"`, and the runner ingests a view of the dump up to each distinct `now` instead |
| `reset(store)` | returns the store to its state after ingest, between cases | or `reset = "copy"`: the runner hard-links a fresh copy of the store for each case |
| `usage()` | GPU seconds, GPU queue seconds and API dollars since the last call | the runner cross-checks GPU seconds against its own sampling of the GPU |
| `oracle(query, since, until, k)` | optional: the system's ranked moments for a plain query, with no agent | for the retrieval run (section 8.5), which compares systems without paid agent sessions |

**Rules every system follows:**
- **Ingest is blind.** It runs before any question is known. It may take as long as it likes, and the time is reported
  as the user's wait.
- **Systems are frozen before a test run.** Their version and configuration hash go with each result. Tuning happens on
  the dev split only (section 6.4).
- **Instructions are general.** A system's instructions text is at most about 2,000 tokens and the same for every
  scenario. It may teach the agent its tools, never what is in a scenario.
- **No network,** unless the system declares itself hosted. Hosted systems run only on public scenarios, never on
  private ones, and their API dollars are reported.
- **GPU work goes through the runner.**
  - In `exclusive` mode (the default), the scheduler gives the GPU to one system at a time.
  - In `lease` mode, the system asks the runner for the GPU around each piece of work, through a small local endpoint named
    in its environment, and the runner knows queue time exactly.

### 3.2 The systems to ship

A benchmark with one contestant only tests that contestant. MMMB ships common, cheap baselines beside mm:

| System | What it is | Tools it gives the agent | Ingest |
| --- | --- | --- | --- |
| `files` | the baseline: no system | none beyond the agent's own (shell, file reads, ffmpeg) | none |
| `listing` | the floor and the leak filter (section 7.3) | a listing of names, sizes, times, EXIF and container fields; no pixels or sound | seconds |
| `embed` | the common "multimodal RAG" setup | `search(text or picture, since, until, k)` over frames at 1 fps and photos (SigLIP or CLIP) and 5 s sound windows (CLAP); `frame(file, t)` | GPU: every frame and window once |
| `transcript` | speech made searchable | `search(words, since, until)` over a timestamped transcript (Whisper) | GPU: all sound once |
| `mm` | Multimem: a small local model reads every window in a range against the agent's question, at question time | `find`, `ask`, `look`, and mm's skill | CPU only; the GPU works per question |
| hosted (later) | a long-context model given the dump in chunks | `ask(question, files)` | none; API dollars per case |

`embed` and `transcript` pay the GPU once at ingest, and `mm` pays it on every case. The report shows both, amortised
over 1, 10 and 100 questions, so a user can see which is cheaper for their use.

Each model inside a system is a variant (`mm/d1-3B`, `mm/d1-Omni-2B`, `embed/siglip2-base`), and every variant is
reported apart.

## 4. Configurations and agents

Each case runs in a fresh agent session per configuration:

| Configuration | The agent has | Role |
| --- | --- | --- |
| `listing` | the listing only | the floor; a case it answers is flagged as answerable from metadata or prior knowledge |
| `files` | the files and its own tools | the baseline |
| `S+files` | its own tools and system S's tools | **the headline for each system**: what a user who installs S has |
| `S` | S's tools only | diagnostic, on request: what S's tools do alone |
| `gold` | the evidence clips alone | diagnostic, on request: a miss here is the agent's reading, not the search |

**Agents:** Claude Code and Codex for now.
- Each runs headless and locked down: no network, no access outside its run, none of the owner's skills or plugins.
- Model, effort and turn cap are the same in every configuration.
- The prompt skeleton is shared: the user's question, the date and time "now", where the dump is, and the answer format
  (section 6.2). A system adds its own instructions and tool schemas.
- Other agents can be added through the same runner.

The headline compares `files` with `S+files` for each system S. Taking the agent's shell away is not what any user does,
so it is a diagnostic only.

## 5. Scenarios

Each scenario is one dump and 15-40 questions about it. All its cases share the dump, so it is built, probed and ingested
once per system. The questions mix the kinds in section 5.6. The tables give one example question each.

[MEDIA.md](MEDIA.md) says where each scenario's media comes from, what is already downloaded, and what is still missing
(4 November 2026). Licences are recorded there but are not a filter (section 11.3).
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

### 6.2 The answer

Every agent ends with one JSON object:

```json
{"answer": "Monday 07:30", "evidence": [{"file": "cam1/2026-11-02_07.mp4", "t0": "07:29:50", "t1": "07:30:05"}]}
```

A cited span may be wall-clock times or seconds into the file. A system that works in wall-clock time and an agent that
works in file offsets are graded alike.

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

### 6.4 Dev and test

Every scenario has a public **dev** split (about 30% of its questions) and a **test** split (about 70%).
- Any system, mm included, is built and tuned on dev only.
- Test results are published only from a frozen system version.
- For the synthetic screen week, seeds 1-10 are dev, and test seeds are drawn fresh for each release. mm's skill and the
  copied harness were adjusted after runs on pilot seed 1, so that seed is dev.
- For public datasets, which questions are in test stays private to the run.

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

### 7.3 The shortcut probe and leaks

**The shortcut probe** (from the copied `shortcuts.py`) checks every dump before any run. No file name, file size, EXIF or
container field, frame difference, splice seam, colour, loudness or frequency band may single out a needle.

**Leaks:**
- No network for agents or local systems, so no one looks up a public dataset's answers.
- Files get the names a phone, camera or recorder would give them, and dataset identifiers are removed.
- The `listing` configuration runs on every scenario. A case it answers is flagged and kept out of the headline. This is
  LVBench's and HourVideo's blind filter, and Daily-Omni's text-only filter, applied to what a dump really leaks.
- Multiple-choice questions (EgoLifeQA) are asked in 4 rotations of their options and count only if all 4 are right
  (VNBench).

### 7.4 The ingest report

Once per dump per system version:
- seconds to ingest, GB per minute, GPU seconds and disk used;
- files refused or misdated, and why;
- seconds from the end of the dump to the first answer.

Every system meets the large-file track (L) here. The copied pilot already found two limits in mm this way (section 10.3).

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
one GPU job and under three, so **exactly one GPU job runs at a time**, whatever the system. A per-question GPU system
already keeps the GPU busy 76-85% of the time, so running more agents against it saves little. Time is saved by choosing
cases, running GPU-free work beside GPU work, and ingesting once.

### 8.2 Choosing cases

Running faster means choosing fewer cases. Nothing else about a run changes.

- **Select** by case ID, scenario, media shape, question kind, split, rung of the size ladder, or any tag:

  ```
  mmmb run --scenarios V1,A1,M4 --rung 1h --kinds last,absent --limit 12 --agents claude --configs files,mm+files
  ```

- **`--limit N`** draws N cases at random, stratified by scenario and kind, with a fixed seed. A small run keeps the mix of
  the full one.
- **`--discriminating`** keeps only cases that separated configurations in earlier runs. A case right in every
  configuration, or wrong in every one, tells a paired comparison nothing. Two such cases stay in as canaries.
- **Before it starts,** the runner prints what the selection will cost, and `--dry-run` stops there:
  - cases × agents × configurations;
  - the dumps that must first be fetched, built or ingested, with their estimated time;
  - wall time and GPU seconds, estimated from earlier runs' medians.
- **The scale of a selection:** a dozen cases on the 1-hour rungs, with ingest cached, take minutes. Every scenario at full
  size takes a night, and the first ingest of a large dump can take longer than the cases themselves.

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
saving of running side by side at 25-35% of a run's wall time.

Two conditions:
- **Dan's go-ahead.** CPU-heavy work beside a GPU job goes against the laptop's current "GPU runs alone" rule.
- **Agents stay off the GPU:** CUDA is hidden from their environment, and `-hwaccel` and `nvidia-smi` are on their deny
  list. An agent decoding on the GPU would be a second GPU user.

The copied harness's error classes stay:
- `setup`, `limit` and `auth` results are asked again and left out;
- `escaped`, `timeout` and `capped` are wrong answers;
- a system that dies, or a lost GPU, counts as `setup`.

A run that stops for any reason continues where it left off when it is started again.

### 8.4 Working time, ingest once, and honest GPU seconds

**Working time, not wall time.** When several cases share the GPU, waiting for it is an artefact of the benchmark. A real
user's system has the GPU to itself. So:
- A case is charged its agent's turns plus its own GPU service time, never queue time. This is Inspect AI's
  `working_limit`, paired with a wall-clock cap.
  - In `lease` mode the runner measures the queue itself.
  - In `exclusive` mode the system reports it through `usage()`, as vLLM reports queue time apart from service time.
- A tool that returns early with partial results counts that time from when its GPU work starts, not from when it was
  queued, so a queued request does not come back empty and invite polling.
- **Before trusting it,** run 10 cases both alone and queued, and compare answers, tokens and charged seconds.

**Budgets sized to the haystack:** 300 working seconds for an hour of media, 900 for a day or more, with a wall cap of 2-3×
that. A case that the retrieval run shows no system can scan within its budget gets its hint narrowed, or is left out of
default selections.

**Ingest once:**
- Downloads are fetched once and content-hashed.
- Each prepared dump is cached by the hash of its recipe.
- Each system's store is cached by (system, version, configuration, dump hash) and hard-linked per case.
- A system or recipe change rebuilds only what it touches.
- The synthetic generator writes days and sources in up to 4 processes.

**GPU seconds are the system's business, and the report records them honestly.**
- Caches inside a system (encoder outputs, answer caches) are allowed if they would also serve a real user.
- Across cases, a cache may only keep what ingest built, so no case gains from another's.
- Batching is the system's own choice, and the report records it.

### 8.5 Runs without agents

- **The retrieval run:** each system's `oracle()` per needle, with no agent. It reports recall at k, average precision and
  seconds. It is cheap, needs no quota, and separates a system's search from the agent's use of it.
- **Re-grading:** recorded runs are graded and reported again after a grader or report change (Harbor's `regrade`), with no
  agent and no GPU.
- **Replays for harness tests:** recorded tool results are replayed at their recorded latency, so the runner can be tested
  offline.

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

## 10. Where the copied code stands, and how it becomes MMMB

### 10.1 What is tied to Multimem today

| File | Lines | Tie to Multimem | Becomes |
| --- | --- | --- | --- |
| `grade.py` | 290 | none | `core/grade`, plus grounded scoring |
| `shortcuts.py` | 643 | `RATE` from Multimem's `tests/day.py` | `core/probe` |
| `haystack.py` | 1,814 | `multimem.reader` to check written frames and count mm's windows; `load` stores files through mm's `remember`; helpers from `tests/` (`day`, `fake_decider`, `conftest`) | scenario V7: the plan, files and questions. `load` and the window count move to `systems/mm`, frame checks use PyAV, and the helpers' constants are copied in |
| `haystack_media.py` | 1,357 | `multimem.config.ffmpeg_exe`; `tests/day` fonts | V7's media, with its own ffmpeg lookup |
| `claude_run.py` | 511 | the `mm` arm's MCP server, tool names and prompt | `core/agents/claude`: tools and instructions come from `serve()` |
| `codex_run.py` | 636 | the same, mm's server instructions, and mm's skill read from the Multimem repo | `core/agents/codex`, likewise |
| `bench.py` | 2,390 | mm's daemon, snapshot homes, scrubbing mm's index between questions, cancelling mm's jobs, mm's GPU estimate, the model-only tier, `tests/conftest` | split: `core/runner` (session loop, slots, deny lists, resume, error classes, statistics, report) and `systems/mm` (mm's daemon, which becomes `ingest`, `serve`, `reset`, `usage` and `oracle`) |
| `egolife.py` | 1,229 | `multimem` config, media, reader and `ask` | scenario V8, with mm's parts in `systems/mm` |
| `README.md`, `results/` | | the published mm-against-files results | `results/2026-11-0x-*`, labelled as system `mm`; a new README for MMMB |

### 10.2 The order of the work

1. Copy in the few helpers from Multimem's `tests/` (font, sample rate, the shade and pitch constants), and give the
   synthetic week its own ffmpeg lookup. V7 then builds with no Multimem.
2. Write the question-file format, and make V7 and V8 write it.
3. Split `bench.py` into `core/runner` and `systems/mm`, and make the agent runners take tools from `serve()`.
4. Re-run the copied pilot through the new runner. The same verdicts for `files` and `mm` (the pilot's mm-only
   configuration) show that nothing was lost. Then `mm+files` runs for the first time.
5. Add case selection, `listing`, then `embed` and `transcript`.

### 10.3 What mm's plug-in will meet first

The ingest report and track L will exercise two limits of mm today:
- `remember` refuses any file over 2 GB (`remember_max_mb`).
- A folder source keeps a file over 50 MB in place, dated by when mm saw it rather than by its recording time.

These belong to mm's notes, not to MMMB's design. Track L is built so that any system's handling of big files shows the
same way.

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
- Cover many small audio files as well as long ones.
- Compare search systems behind one plug-in contract, each against the agent's own tools on the same files.
- Report the ingest wait and GPU seconds beside tokens and dollars.

LSC and VBS compare many systems on one task, but with people at the controls, not agents. No benchmark of continuous
screen-video memory was found.

### 11.2 What to take

| # | Idea | From | In MMMB |
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
- AudioMarathon builds long audio by concatenating clips. Its seams are the artificial shortcut MMMB avoids.
- TV Needle concatenates videos into one. MMMB keeps files separate, as a user's are.

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
| P0 | The separation (section 10.2, steps 1-4); the question-file format; grounded answers; case selection; `listing`; cached ingest; GPU-free configurations beside GPU ones; the working-time clock checked on 10 cases | the core runs without Multimem installed, and the copied pilot gives the same verdicts through `systems/mm` |
| P1 | `embed` and `transcript`; the priority-1 scenarios (V1, V5, I1, I3, A1, A3, M4) and track L | each has 15+ questions, a passing probe and a retrieval run for every system |
| P2 | The priority-2 scenarios (V4, I2, A2, A4, M1, M5, M6, X1, X2); the size ladder; session mode | a full run has 70+ discriminating questions per agent |
| P3 | The priority-3 scenarios, once sources are found; hosted systems if wanted | as P1 |

## 13. Open questions for Dan

1. **Licences:** decided on 4 November 2026: not a filter for use on this machine. Gated sources still need Dan
   ([MEDIA.md](MEDIA.md), "Decisions for Dan").
2. **CPU work beside a GPU job:** may GPU-free configurations and builds run while a system holds the GPU?
3. **Where mm's plug-in lives:** in MMMB as `systems/mm` with Multimem as an optional dependency (proposed), or in
   Multimem?
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
8. **Publishing:** will MMMB be public? That decides the dev/test policy for the real-data scenarios, and the project's
   own licence.
