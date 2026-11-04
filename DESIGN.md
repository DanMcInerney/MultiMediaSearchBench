# MultiMediaMemoryBench (MMMB): design for realism and speed

Status: proposal, 4 November 2026. Nothing in it is built. The starting point is the copy, in this folder, of Multimem's
`benchmarks/` (synthetic haystack generator 6, EgoLifeQA, the harness and the two published results). Sections 2 and 6.1 are
measured; the rest is design, and every estimate is marked as one.

## 1. What MMMB is for

**The point of mm:** a user dumps very large video, picture and sound files on their laptop (an NVR export, a phone's
camera roll, a day of microphone, a folder of robot logs) and asks their agent to find something in them. mm cuts the
files into time windows, and a small local decision model reads every window in the range the agent names and returns
the likeliest moments. The agent looks at those and answers. The model is the eyes and ears. The agent's job is to aim
it and check what it finds.

**The question MMMB answers:** when a user gives an agent a pile of large media and asks for something in it, does
attaching mm make the agent right more often, sooner and more cheaply?

**What it measures, per question and arm:**

| Measure | Why |
| --- | --- |
| Grounded accuracy: the answer is right and the evidence it cites (file and time) is where the answer is | a right answer with the wrong evidence is a guess (section 5) |
| Seconds to answer | the user waits for it |
| Tokens and dollars | the user pays for it |
| GPU seconds | local, free in money, not in time or heat |
| Dump cost: seconds from dump to first answer, disk used | a real user pays it once per dump; the current bench never measures it |

**Arms.** Each question runs in a fresh agent session per arm:

| Arm | The agent has | Role |
| --- | --- | --- |
| `listing` (was `blind`) | the folder's listing: names, sizes, times, EXIF; no pixels or sound | the floor, and the leak filter: a question it answers is flagged (section 8.4) |
| `files` | the files and its own tools (Read, shell, ffmpeg) | the baseline |
| `files+mm` (new) | its own tools and mm's | **the headline arm**: this is what a user who installs mm has |
| `mm` | mm's tools only | as now, for attributing a gain to mm's tools; smoke tier only |
| `gold` | the evidence clips alone | diagnostic, smoke tier only: a miss here is the agent's reading, not the search |

The headline comparison becomes `files` against `files+mm`. Today's `mm` arm takes the agent's shell away, which no user
would do. Removing it measures mm's tools in isolation, not mm as installed.

## 2. Where the copied bench stands (measured)

**Results** (the README has the full tables): on the 3-day synthetic pilot (24 paired questions, d1-3B), Codex went
22/24 → 23/24 with mm and Claude 18/24 → 22/24. mm never cost a right answer, and it halved Codex's tokens and cut
Claude's about 8×. Neither result is significant (McNemar p = 1.0 and 0.125). On EgoLifeQA (real egocentric video, 40
video questions), mm tied the files arm (34 vs 36) at twice the tokens per correct answer. The transcripts carried most
of the answers.

**Time** (pilot seed 1, `D:\mm-bench\20261104-0314-pilot-s1`, from the result files):

| Step | Wall time | What bounds it |
| --- | --- | --- |
| `build` (write, freeze, load, shortcut probe) | 11 min 23 s | CPU, one process for most of it |
| Codex files arm, 31 execs, 3 at a time | ~14 min (1,896 exec-seconds) | the API |
| Claude files arm, 31 execs, 3 at a time | ~85 min (6,693 exec-seconds; its longest exec hit the 900 s limit) | the API, and Claude's brute force over frames |
| Codex mm arm, 24 execs, 1 at a time | ~25 min: 1,513 exec-s, of which **1,156 GPU-s (76%)** | the GPU |
| Claude mm arm, 24 execs, 1 at a time | ~49 min: 2,942 exec-s, of which **2,499 GPU-s (85%)** | the GPU |

The mm arms run one exec at a time because the laptop's NVIDIA driver crashed with three concurrent asks, and has
crashed with one as well. The run took about 4 hours, including a driver crash and a reboot. **In the mm arms the GPU is
already busy 76-85% of the time**, so running agents in parallel against it can save at most the remaining 15-25%. The
large savings have to come from elsewhere (section 6):
- running the arms side by side rather than in phases;
- not rebuilding or re-reading what has not changed;
- not spending execs on questions that never tell the arms apart.

**What is not realistic yet:**

1. **Synthetic needles.** A drawn red dialog and a pure tone are easier to filter than a real error or a real alarm in a
   real room. The README already says this favours the files arm.
2. **No large files.** The biggest file is a 1-minute screen segment or a 30 s porch clip. mm's handling of big files is
   never exercised, and today it is the weakest part of the dump story:
   - `remember` refuses any file over `remember_max_mb` (2,000 MB by default; the bench's snapshot sets 500)
     (`src/multimem/ingest.py:433`). A 4 GB GoPro file or an NVR day export is `too_big`.
   - A folder source indexes a file over `copy_max_mb` (50 MB) **in place, at the time mm saw it**
     (`ingest.py:250`), not at the file's own `creation_time`. Per DESIGN.md §13.2 every window of such a file sits at
     that one time, with `at` its offset into the file. So `since` and `until` cannot narrow inside a 3-hour video, and
     its moments carry the ingest time, not the recording time.

   "mm chops large files" holds for reading (windows of 1 s of video, 5 s of sound) but not for storing or dating them.
   MMMB should measure this before anyone relies on it (section 4, track L).
3. **No messy dumps.** Real dumps mix HEIC, JPEG and PNG screenshots, rotated HEVC `.MOV`, variable frame rate,
   `.m4a` voice memos, WhatsApp copies with their EXIF stripped, duplicates, cameras with a wrong clock, and meaningless
   names (`IMG_4821.HEIC`).
4. **Narrow questions.** There are 10 templates, each with one needle and one value. Missing kinds: find-all, query by
   example ("this bike"), cross-camera ("which camera saw him first"), cross-modal ("what was at the door when the bell
   rang"), and order ("before or after the rain").
5. **Answers are not grounded.** The grader checks the value and, for some kinds, a time. It does not check that the
   agent's evidence is where the answer is.
6. **Too few discriminating questions.** 8 of the pilot's 24 (text and records) are right in both arms every time, and
   tell the arms apart in no run.
7. **The dump step is untimed.** Ingest happens at build time, so time-to-first-answer after a dump is not measured.
8. **EgoLifeQA is multiple choice** (the blind arm scores 40%) and transcript-heavy (agents asked mm to look at video in
   14 of 40 questions).
9. **It cannot run from this folder yet.** `bench.py`, `haystack.py`, `haystack_media.py` and `shortcuts.py` import
   `multimem` and add `../tests` to `sys.path` for `conftest`, `day` and `fake_decider`. `codex_run.ROOT` is
   `parents[1]`. Here, `parents[1]` is `D:\robotics`.

## 3. Principles

1. **Real pixels and real sound in the haystack.** The synthetic week stays as a control with every answer known. Each
   other scenario uses real footage, real photos and real recordings, with needles that are real objects and sounds.
2. **Every answer is known, checked by rule, and carries evidence.** No model grades another.
3. **Dumps look like real dumps:** real sizes, formats, timestamps and mess.
4. **Questions sound like users.** They are vague about time, name things rather than files, and sometimes ask about
   things that are not there.
5. **Build once, ask many.** Cache everything that does not change what is measured. Never cache what is measured.
6. **One model process on the GPU, never idle while work waits.** Exactly one GPU job at a time, as now.
7. **Spend execs where the arms can differ.** A question both arms always get right is a canary, not a data point.

## 4. Scenarios

Each scenario is a **dump** (a folder as the user would hand it over) plus 15-40 questions about it. Questions share a
dump, so it is built, ingested and probed once.

| # | Scenario | The user says | The dump | Real data to build it from | Priority |
| --- | --- | --- | --- | --- | --- |
| S1 | Home security export | "Someone took my parcel this week. Here's the export from my 3 cameras." | 3 cameras × 2-3 days, as an NVR exports them: 1-hour MP4s, or one 4-8 GB file per camera per day; day, night and IR | MEVA (328 h, overlapping cameras, 37 activity types with spans; open on AWS) | 1 |
| S2 | Household sound | "Did the smoke alarm go off while I was away?" | 1-3 days of one room microphone, one file a day (a 24 h mono 16 kHz WAV is about 2.7 GB) | a real home's week of ambience (SINS) with real events (FSD50K) mixed Scaper-style at a set SNR; TV, music and near misses as decoys | 1 |
| L | Large-file track | (all of the above, bigger) | the same needles in one 4 GB video, one 10 h audio file, a 50 MP panorama, a 30 GB folder | re-muxed from S1, S2, S3 | 1 |
| S3 | Phone camera roll | "I dumped my phone. Find the photo of the parking sign at the airport." | 5-10k photos (HEIC, JPEG, PNG screenshots), 200 videos (rotated HEVC MOV, slow motion, a few over 2 GB), 50 voice memos, WhatsApp copies without EXIF, duplicates | YFCC100M single-user photostreams (real EXIF times and cameras); DISBench's 57 users and 122 verified queries; Dan's own phone (private; the most realistic) | 2 |
| S4 | Robot logs | "Find the episodes where the gripper dropped the cup." | a few hundred episodes, 2-3 cameras each, MP4 per camera | DROID (language instructions per episode) | 2 |
| S0 | Scripted real day | "When did the courier come? Did I leave the hob on?" | a day of Dan's home, recorded on his own cameras, phone and microphone, with about 30 planted events from a script | the truth is the script (M3-Bench's method); private | 2 |
| S5 | Meetings and lectures | "When was the cost slide shown, and who drew on the whiteboard?" | 20-40 h of meeting video and audio, 30-90 min files, slides | AMI Meeting Corpus (room video, audio, slides, topic and decision annotations) | 3 |
| S6 | Dashcam or road trip | "When did it start raining? What exit did we take for the fuel stop?" | 8-10 h in 1-3 min chunks, camera clock set wrong | comma2k19 or BDD100K | 3 |
| S7 | Field recorder | "When did the fox first show up on the trap camera? When did the owl call?" | 20k trap images (90% empty, wind triggers), 3 nights of audio | Caltech Camera Traps (LILA BC); 33 annotated one-hour bird soundscapes | 3 |
| S8 | Developer screen | "When did the borrow-checker error first appear?" | the synthetic week (gen 6), plus real screen video | `haystack.py`; the calibration stream; AgentNet's human screen recordings | kept |
| S9 | Egocentric day | as EgoLifeQA | 3 days of glasses video | EgoLife (as now), turned into open answers with evidence where its annotations allow | kept |

Licences and gating are in section 8. Dan reads each licence. No agent accepts one or makes an account.

**What makes each one hard, on purpose:**
- S1: night and IR, rain, headlights, many people and cars that are not the one asked about, and the same event on two
  cameras at different times. The question's thing is small in the frame.
- S2: needles at -5 to +10 dB below or above the ambience, TV and music that contain alarms and doorbells, the dog that
  barks at everything. Agents cannot hear, so Dan checks the sound needles by ear before a run.
- L: the needle a minute before the end of a 10-hour file, and needles that straddle a window or segment boundary.
- S3: many near-duplicates (burst shots), screenshots of text, and photos whose time comes from the file, not EXIF.
- S4: hundreds of near-identical episodes. Only the outcome differs.
- S5: speech. mm is weak on long speech (VISION.md), so speech questions are reported apart from the rest.
- S6: a camera clock that is wrong by a known offset: "what time" means the true time.
- S7: thousands of empty frames, animals at the edge of the frame or at night.

### Question kinds

| Kind | The user says | Answer | Graded by | Borrowed from |
| --- | --- | --- | --- | --- |
| find and read | "What level did I park on at the airport?" | a value and its evidence | value rule, evidence overlap | CG-Bench (grounded answers) |
| last seen | "When did I last have the blue backpack?" | a time and its evidence | time within a tolerance; earlier needles are wrong answers | Ego4D NLQ |
| when | "What time did the white van arrive?" | a time | tolerance | |
| count | "How many times did the dog bark last night?" | a number, with evidence for each | exact (±1 reported apart) | |
| find all | "Show me every clip with the cat in it." | a list of (file, span) | event F1 with a time collar | VBS ad-hoc search, DCASE |
| absent | "Did anyone open the side gate on Tuesday?" | "not there" | not there, no evidence | as now |
| query by example | "Find other photos with this bike." (a picture attached) | a list | F1 | Ego4D VQ2D, MomentSeeker |
| cross-source | "Which camera saw the man with the bag first?" | source and time | both | MEVA's overlapping cameras |
| cross-modal | "What was at the front door when the doorbell rang?" | a value | value rule, both spans | |
| order | "Did the parcel arrive before or after it started raining?" | before or after, with both spans | answer and both spans | |
| vague time | "the morning of the trip", "last weekend" | as its kind | as its kind | |

About a quarter of all questions are absent ones. Agents and models force matches on them.

Every question records these tags, and the report cuts accuracy and cost by each:
- **haystack size:** hours, files, and the windows a full scan would read;
- **certificate length:** seconds of media needed to verify the answer (EgoSchema);
- **look-back:** how long before "now" the evidence is (EgoLifeQA: under 2 h, 2-6 h, 6-24 h, over 24 h);
- **evidence:** how many pieces, and how far apart (EgoMemReason);
- **needs audio:** whether a deaf model like d1-3B can answer it at all;
- **needle salience:** share of the frame, or SNR in dB;
- the modality and the time-hint level.

## 5. Answers and grading

The agent ends with one JSON object:

```json
{"answer": "level P3", "evidence": [{"file": "DCIM/100APPLE/IMG_4821.HEIC", "t": null},
                                    {"file": "cam2/2026-11-03_14.mp4", "t0": "14:31:05", "t1": "14:31:20"}]}
```

- **Grounded accuracy** (the headline): the value passes its rule and every cited span overlaps a truth span within
  the kind's tolerance (video 10 s, sound 5 s, a picture the same file). A right value with wrong evidence is counted
  as `lucky` and shown apart.
- **Find all, count:** event F1 by distinct moments, as DCASE scores sound events (DCASE's collars are 200 ms at the
  onset and max(200 ms, 20% of the event) at the offset; MMMB widens them to mm's window, 1 s of video and 5 s of
  sound). A count is also graded exactly.
- **Absent:** right only with "not there" and no evidence. The report counts the searches each absent question drew.
- **Time:** each correct answer's seconds feed a time-to-correct curve per arm (the share answered correctly within 30,
  60, 180 and 600 s), read from data the bench already logs. A secondary score in the style of the Lifelog Search
  Challenge, max(0, 100 - 50·t/T - 10·wrong), with `wrong` counting cited spans that hold nothing, is reported beside
  it, never instead of accuracy.
- **Cost:** tokens, dollars and GPU seconds per correct answer. The report shows a Pareto plot of accuracy against
  seconds and dollars per arm.

## 6. Speed

### 6.1 Where the time goes

See the table in section 2. In order of size:
1. **GPU seconds in the mm arms:** 76-85% of their exec time. One `ask` over 3 days of screen needs more than the 900 s a
   question gets.
2. **Claude's files arm:** brute force over frames, 6,693 exec-seconds for 31 questions.
3. **Build:** 11 min per seed and scale, repeated whenever the generator or mm's store changes.
4. **Ceiling questions:** 8 of 24 paired questions per agent, right in both arms every time.
5. **Serial phases:** the arms ran one after another, files arms first, then mm arms, one agent at a time.

### 6.2 The plan

**a. Three resource pools and one scheduler.** Every unit of work names the resource it uses:

| Pool | Size | Work |
| --- | --- | --- |
| CPU | at most 4 processes (the laptop rule) | download, re-mux, generate, ingest, shortcut probe |
| Agents | per CLI, up to 4 sessions (quota and rate limits) | every arm's agent turns |
| GPU | **exactly 1 job at a time** | every `ask`, the decider tier |

The scheduler runs the `listing` and `files` arms (agents only), and the next scenario's build, at the same time as the
mm arms (agents plus GPU). This is the largest single saving: the speed research estimates 25-35% of a run's wall
time. The scheduler starts the longest expected execs first, so one heavy ask does not set the tail. Harbor calls the
GPU pool a `concurrency_group`; Inspect AI uses `concurrency("gpu", 1)`.

Two conditions:
- **Dan's go-ahead.** Running ffmpeg-heavy files execs beside a GPU run goes against the current "GPU runs alone" rule.
  Without it, the arms stay in phases and this saving is lost.
- **The files arm is kept off the GPU:** CUDA hidden from its environment, and `-hwaccel` and `nvidia-smi` on its
  deny list. An agent that decodes with the GPU would be a second GPU user, and the driver crashes.

CPU contention may slow mm's decoding (2-35 ms of a window's 50-60), so the report gives decode time beside GPU time.
The harness already does what Harbor's error classes do: `setup`, `limit` and `auth` results are asked again and left
out, and `escaped`, `timeout` and `capped` are wrong answers. Keep that, and add "daemon died" to the first group.

**b. A GPU queue with a working-time clock.** Let 2-3 mm execs run at once against one daemon. Their asks wait in one FIFO
queue and reach the GPU one job at a time, the same load as `--jobs 1` today. Waiting in the queue is an artefact of the
benchmark, since a real user's agent has the GPU to itself, so neither the agent nor the clock should see it:
- an exec is charged its own turns plus its own GPU service time, never queue time. This is Inspect AI's
  `working_limit`: clock time minus waits for a shared resource, always paired with a wall-clock cap. For MMMB, 900
  working seconds and a 2-3× wall cap. mm's job record needs queued, started and ended times, as vLLM logs
  `request_queue_time` apart from service time. The harness enforces the working limit itself and sets the CLI's own
  timeout to the wall cap;
- `ask`'s "return within about 45 s with what it has" counts from the job's start, not its submission, so a queued ask
  does not come back empty and invite polling;
- **before trusting it,** run 10 questions both ways (alone and queued) and compare answers, tokens and charged seconds.

Expected saving: up to the 15-25% of mm-arm time the GPU now sits idle, plus the overlap of two agents' mm arms (not
measured). This is worth doing, but it is not the big lever.

**c. Fewer GPU seconds per question.** This is harder than it looks.
- **Caching each window's encoding is ruled out for d1.** vLLM's encoder cache and SGLang's prefix cache reuse an
  image's encoding across requests. DESIGN.md §16 lists "KV or vision-encoder caching" as not adopted, because under
  d1's set attention the image's state depends on the questions after it. A window costs its full 50-60 ms on every
  ask.
- **No window read twice in one exec.** DESIGN.md §16 defers an answer cache keyed by window, model, profile and
  question set ("built once re-asks are seen to repeat windows"). Claude made 5.2 asks per media exec in the pilot, many
  over overlapping ranges, so MMMB can settle this first: count the repeated windows in the pilot's `tools.json`.
  - In a scored run the cache is cleared with each exec's scrub, so no exec gains from another's.
  - In replays (T0) it is kept and charges the recorded service time.
  - Cross-agent hits would be rare anyway: the two CLIs word their questions differently.
- **Batch windows,** only after a check. On armature's probe, 4 windows in one call took 298 ms against 368 ms
  (-19%, at 0.4 GB more GPU memory), and that is about the ceiling, since window reads are prefill-bound. GPU kernels are
  not batch-invariant: Thinking Machines found 80 distinct completions in 1,000 at temperature 0 until they used
  batch-invariant kernels. So a batched verdict may differ from a single one, which would break the still gate's and
  the answer cache's reliance on d1 being deterministic. Batch only if batched and single verdicts agree on a few
  thousand real windows.
- **Real footage gates more:** mm's still gate keeps 98% of the synthetic screen's seconds but 71-86% of a real one's,
  and a still security camera keeps about 120 windows an hour rather than 3,600. Real scenarios will cost less GPU per
  hour than the synthetic screen. This is a fact to report, not a lever.

So the GPU's share is cut mostly by reading fewer windows, not by reading each one faster. The agent narrowing its range
is in mm's skill and the lint, and the bench measures it.

**d. Build and ingest once.** Downloads go once to `D:\bench-data`, content-hashed. Each scenario's prepared dump (renamed,
re-muxed, metadata set) is cached by its recipe's hash. Each scenario's mm snapshot home is cached by (dump hash, mm store
schema), since the index does not depend on the model, and is hard-linked into per-arm homes, as now. A rebuild happens
only when the recipe or mm's store changes. The synthetic generator writes days and sources in up to 4 processes.

**e. Spend execs where they count:**
- **Canaries, not data:** text and records questions, and any question both arms passed in two earlier runs, leave the
  routine tiers. Two stay as canaries.
- **Enough discriminating questions.** Paired questions per agent and model needed at 80% power and α = 0.05, by
  Connor's approximation for McNemar:

  | Effect, and share of pairs right in one arm only | Questions |
  | --- | --- |
  | Claude's pilot (+17 points, all 4 discordant pairs for mm), optimistic | about 45 |
  | +15 points, 20% discordant | about 70 |
  | +10 points, 15-20% discordant | about 115-155 |
  | Codex's pilot (+4 points, about 8% discordant) | about 390 |

  So MMMB aims at effects of 10 points or more, from 70-150 media questions per run drawn from several scenarios. A gain
  as small as Codex's will not be certified at any affordable size, and the report should say so rather than imply it.
- **Distinct questions, not repeats.** Miller's "Adding Error Bars to Evals" shows that repeating a question shrinks only
  its own run-to-run variance, never the spread between questions. Repeats go to a small reliability set: 3 runs of 10
  questions per arm, reported as pass^k as τ-bench does.
- **Drop questions that carry no information.** A pair right in both arms, or wrong in both, tells McNemar nothing.
  Questions concordant in two earlier runs leave the routine tiers, and the effect is then reported as "on
  discriminating questions". The speed research puts the saving at 30-60% of execs. Benchmark-compression work keeps
  tasks with 30-70% pass rates; for agent benchmarks it compresses only 2-3.5×.
- **Sequential stopping, pre-registered:** questions run in a random order, and a comparison stops when an
  anytime-valid rule (an e-process or confidence sequence) decides it, capped at 150. Peeking without such a rule
  inflates false positives. The saving is large when the effect is large and nothing when it is small. The
  SpatiotemporalBench trial stopped at 21 of 24 by hand for the same reason.

**f. Tiers.** Each tier has a wall-clock target (estimates, not measured):

| Tier | What | Resources | Target |
| --- | --- | --- | --- |
| T0 | unit tests, fake decider; recorded execs re-graded and re-reported (Harbor's `regrade`); recorded ask and look results replayed at their recorded latency for harness tests | CPU | minutes |
| T1 | decider tier: each needle's oracle yes/no per model, no agent | GPU | under 1 h per model per scenario, cached |
| T2 | smoke: one scenario, 8 discriminating questions, one agent, all arms | all | under 20 min |
| T3 | standard: every scenario, both agents, `files` and `files+mm`, one pass | all | one night |
| T4 | significance: T3 plus sequential top-up and repeats | all | a weekend |

**g. Budgets sized to the haystack.** A question's time budget scales with what a brute-force scan would read: 300 s for
an hour of media, 900 s for a day or more. The run flags any question whose oracle ask would exceed the budget before it
starts, as `beyond_timeout` does now, and the T1 tier drops it from T2 and T3 or narrows its hint.

**h. Session mode** (a second track, for realism): one agent session gets 5 questions about one dump in a row, as a user
would ask them. Startup and the agent's learning of what is recorded are paid once. Reported apart from the independent
questions, which keep the statistics clean.

### 6.3 What the pilot would cost under the plan

Estimates, not measured. The same 24 questions, both agents, one seed:

| | Today | Plan |
| --- | --- | --- |
| Build | 11 min | 0 on a rerun (cached); about 4 min fresh (4 processes) |
| Questions | 24 per agent | 16 media + 2 canaries (the controls are quick, so this saves quota more than time) |
| Files arms | about 85 min (Claude sets it) | beside the mm arms, at 4 jobs |
| mm arms | about 75 min, one agent after the other | about 60 min, both agents in one GPU queue (only the 15-25% idle is recovered) |
| **Total** | **about 3 h** | **about 1.5 h**, mostly from running the arms side by side |

What would cut it further is fewer GPU seconds, and d1's design rules out the usual caches (6.2c). The answer cache is
the one to measure. On a bigger run, dropping concordant questions and sequential stopping save more than any of
these.

## 7. Building the haystacks

**Ways to know the answer:**
1. **Annotated real data.** A dataset's own time spans (MEVA's activities, AMI's topics and decisions, DROID's episode
   instructions, camera-trap species labels) become questions through templates in plain user wording. Each question
   is checked by rule against the spans, and Dan spot-checks 10%.
2. **Real needles inserted into real backgrounds,** where annotations do not cover what users ask. Sound: real event
   clips mixed into real ambience at a set SNR (the Scaper method). Video: real clips spliced at a natural cut, or a real
   object composited with blending. The shortcut probe must show that the splice is not the most unusual moment in
   its scope.
3. **Owner-written questions about the owner's own data:** S3 with Dan's phone, S4 with his robot logs. ScreenshotVQA,
   camroll and ATM-Bench all chose this over LLM-written questions. These are the most realistic and stay private on the
   laptop.
4. **A scripted real day (S0):** Dan follows a script of about 30 planted events while his own cameras, phone and
   microphone record, so the truth is known by construction (M3-Bench's method). This is real light, real noise and
   real codecs, with known answers, which also gives the agents' sound answers a ground truth that nobody has to hear.
5. **Real footage on a plausible schedule** for the synthetic week (X-LeBench's method): the porch camera's clips
   become real ones (MEVA, S0), laid on the plan's days, with the plan's needles and decoys.

**The shortcut probe** extends to real data. No file name, file size, EXIF or container field, frame difference, splice
seam, colour, loudness or frequency band may single out a needle.

**Leaks:**
- No network in either arm, so no agent can look up a public dataset's answers.
- Files get the names a phone or NVR would give them, and dataset identifiers are removed.
- The `listing` arm runs on every public-data scenario. A question it gets right is answerable from metadata or from
  what the model already knows of the dataset, and is flagged.

**The dump step as its own track** (once per scenario per mm version, CPU only): seconds to ingest, GB per minute, disk
used, files refused or misdated (`too_big`, in-place, `time_from = now`), and seconds from the end of the dump to the
first answer.

**Scaling curves:** each S1, S2 and S8 needle sits in haystacks of 1 h, 4 h, 1 day and 3 days of the same source. This
shows from what size of dump mm pays off for each agent, the main claim a user cares about.

## 8. Ideas from other benchmarks, and datasets

From web research on 4 November 2026: three sweeps covering video, sound/pictures/lifelogs, and harness speed.
Links are to the papers. "Unverified" means only a secondary source was found. Several 2026 papers postdate what this
project has read before, so treat their numbers as theirs.

### 8.1 The closest benchmarks, and what none of them does

| Benchmark | Close because | Scale |
| --- | --- | --- |
| [LVBench](https://arxiv.org/abs/2406.08035), [HourVideo](https://arxiv.org/abs/2411.04998), [EgoLifeQA](https://arxiv.org/abs/2503.03803), [EgoMonth](https://arxiv.org/abs/2608.13113) | hour-to-month video QA | 117 h / 500 videos / 266 h / 300+ h |
| [Ego-R1 Bench](https://arxiv.org/abs/2506.13654) | an agent with tools much like mm's `find`, `ask` and `look` over a week of video | 300 questions, ~44 h each |
| [M3-Bench](https://arxiv.org/abs/2508.09736) | memory agents over long robot-view video; questions asked at a time | 100 scripted videos + 920 web; 4,490 questions |
| [LongVidSearch](https://arxiv.org/abs/2603.14468), [Deep Video Discovery](https://arxiv.org/abs/2505.18079) | agentic search, scored with its tool cost | 3,000 questions / LVBench |
| [camroll](https://arxiv.org/abs/2606.05275), [DISBench](https://arxiv.org/abs/2602.10809), [ATM-Bench](https://arxiv.org/abs/2603.01990), [PhotoBench](https://arxiv.org/abs/2603.01493) | a real person's camera roll, with evidence | 31.5K / 109K / 3.8K photos + 533 videos / 3.6K photos |
| [BLAB](https://arxiv.org/abs/2505.03054) | long audio, timestamps scored by F1 | 833 h, clips up to 2 h |
| [Visual Haystacks](https://arxiv.org/abs/2407.13766) | accuracy against haystack size, 1 to 10K images | ~1K questions |
| [Lifelog Search Challenge](https://arxiv.org/abs/2506.06743), [Video Browser Showdown](https://videobrowsershowdown.org/about-vbs/) | timed known-item search over months of images and thousands of hours of video | 725K images / ~2,300-3,800 h |
| [MIRIX ScreenshotVQA](https://arxiv.org/abs/2507.07957) | memory of one's own screen, owner-written questions (not released) | 5-18K screenshots per person |

**What none of them does, and MMMB should:**
- Pit an agent with its own shell against the same agent with a local search tool, on the same files.
- Use real dump files at real size and in real mess.
- Time the dump.
- Report GPU seconds beside tokens.

No benchmark of continuous screen-video memory (Screenpipe- or Recall-style) was found.

### 8.2 What to take

Ranked by what each does for MMMB:

| # | Idea | From | In MMMB | For |
| --- | --- | --- | --- | --- |
| 1 | **Grounded scoring.** Correct only if the cited file and span overlap the evidence. Report plain and grounded accuracy. | CG-Bench (white-box), Ego4D NLQ, TVR, ATM-Bench (Joint@k = answer × evidence recall) | section 5; mm already returns spans | realism |
| 2 | **Real footage on a plausible schedule**, real needles spliced in at set depths | X-LeBench (LLM-written days filled with real Ego4D clips), TimeScope, MLVU | the porch camera becomes real clips (MEVA, a scripted capture) laid on the week's plan; needles at controlled depth | realism |
| 3 | **Real sounds mixed into real rooms** at a set SNR, with exact onsets | Scaper, DESED, RealDESED | S2: FSD50K events into a real home's ambience (SINS), SNR swept down to 0 dB and below; near misses (phone ring vs doorbell, microwave vs smoke alarm) | realism |
| 4 | **Scripted real capture.** Act out a day with planted events, so the truth is known by construction | M3-Bench (actors with head cameras following scripts of 70+ events) | Dan records a scripted day at home (cameras, phone, microphone) with 30 planted events; kept private | realism |
| 5 | **Haystack-size ladder.** The same needle in 1 h, 8 h, 1 day, 1 week. The short rungs are the smoke runs. | TimeScope, Visual Haystacks, V-NIAH (by depth too) | section 7's scaling curves; T2 runs the 1 h rung | both |
| 6 | **Absent needles, 20-30% of questions.** Agents and models force matches; DVD's main failure is the "search trap", searching again and again for what is not there. | PhotoBench, MMNeedle, Deep Video Discovery | absent questions raised from 7 in 24 to a quarter; the report counts searches per absent question | realism |
| 7 | **Leak filters.** Drop or flag questions answered without the media, from names, times and EXIF alone. | HourVideo, LVBench (blind filter), Daily-Omni (text-only filter), VNBench (rotated options) | the `blind` arm becomes a `listing` arm (section 8.4); EgoLifeQA's choices are asked in 4 rotations, counted right only if all 4 are | realism |
| 8 | **Penalise time and wrong evidence**, not only reward a right answer | LSC: max(0, 100 - 50·t/T - 10·wrong); VBS: max(0, 50 + 50·decay - 10·wrong) | a secondary score beside grounded accuracy; `wrong` counts cited spans that hold nothing | realism |
| 9 | **Find-all by distinct moments**, with graded relevance | VBS ad-hoc search, MultiVENT 2.0, DCASE event F1 (onset collar 200 ms, offset max(200 ms, 20% of the event)) | find-all and count questions; collars widened to mm's window (1 s video, 5 s sound) | realism |
| 10 | **Tag every question:** certificate length, look-back, needs-audio, evidence pieces and how far apart | EgoSchema, Video-MME, EgoLifeQA, EgoMemReason, TeleEgo | section 4's tags; the report cuts by each (mm should gain most at long look-back and wide spans) | realism |
| 11 | **Multi-hop across files and days**, and aggregates | DISBench (inter-event), LongVidSearch (hop-k), TimelineQA, MVU-Eval | "how many times did the doorbell ring before the parcel came?" | realism |
| 12 | **Query by example** | Ego4D VQ2D, MomentSeeker | "here is my blue mug; when was it last on the porch camera?" | realism |
| 13 | **Owner-written questions about their own record.** ScreenshotVQA, camroll and ATM-Bench all turned down LLM-written questions as unnatural. | those three | Dan writes the questions for S3/S4/scripted day; an LLM may draft template wording for public sets, a person keeps or rewrites each | realism |
| 14 | **Gold-evidence arm** (diagnostic): the agent is handed the right clips | LongVidSearch (GPT-5 near-perfect with gold clips, 42% without) | on T2 only: a miss with gold evidence is the agent's reading; a miss without it is the search | both |
| 15 | **"As-of" questions.** Each question has a now, and only media before it counts. | TeleEgo, M3-Bench, EgoLifeQA | already in `egolife.py` (horizons); extend to every scenario so one dump serves many questions | speed |
| 16 | **Score the model apart from the agent.** Threshold-free AP or PSDS on mm's raw p per needle | DCASE PSDS, LV-Haystack (search F1 and FLOPs) | the decider tier (T1) reports AP and seconds per window per model | speed |

**Taken as warnings:**
- AudioMarathon builds long audio by concatenating clips. Its seams are the artificial shortcut MMMB avoids.
- TV Needle concatenates videos into one. MMMB keeps the files separate, as a user's are.

### 8.3 Datasets

| Dataset | Licence and gating (as found) | Size and annotations | For |
| --- | --- | --- | --- |
| [MEVA](https://registry.opendata.aws/mevadata/) | CC BY 4.0; open on AWS, no account | ~328 h of multi-camera video (+4.6 h drone); 37 activity types with spatio-temporal boxes for ~184 h | S1 home and building security; real porch clips for S8 |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC BY 4.0 for signals, transcripts and some annotations; direct download | 100 h; room and personal video, close and far microphones, slides; word-level transcripts, dialogue acts | S5 meetings |
| [FSD50K](https://zenodo.org/records/4060432) | CC BY 4.0 for the dataset, per-clip CC0, BY or BY-NC; no account | 51K clips, 108 h, 200 classes (doorbell and alarm classes: check `vocabulary.csv`) | S2 foreground events |
| [SINS](https://zenodo.org/records/2555143) | licence in a PDF, unread | one continuous week in a real home, 13 nodes × 4 microphones, activity labels | S2 real backgrounds |
| [DROID](https://arxiv.org/abs/2403.12945) | CC BY 4.0 | 76K episodes, 350 h, 3 cameras, an instruction per episode | S4 robot logs |
| YFCC100M photostreams, or [DISBench](https://arxiv.org/abs/2602.10809) built on them | per-photo CC licences; the original metadata is under Yahoo's Webscope agreement (reports conflict) | real users' photos with EXIF time and camera; DISBench: 57 users, 109K photos, 122 verified queries with exhaustive targets | S3 phone roll |
| [Annotated bird soundscapes](https://zenodo.org/records/7050014) | CC BY 4.0; no account | 33 one-hour recordings, 20,147 timed calls, 56 species | S7 night audio |
| [Caltech Camera Traps](https://lila.science/) (LILA BC) | CDLA-Permissive; public cloud | 243K images from 140 places (per-image time fields unverified) | S7 trap camera |
| [comma2k19](https://github.com/commaai/comma2k19) | MIT | 33 h of highway driving, ~100 GB, GPS and CAN, **no event labels** (questions from GPS and CAN: "when did we stop?") | S6 dashcam |
| [AgentNet / OpenCUA](https://huggingface.co/datasets/xlangai/AgentNet) | MIT per its card; gating unverified | 22.6K human screen-video tasks on Windows, macOS and Ubuntu, with input events | S8 real screen video |
| EPIC-KITCHENS-100 + EPIC-SOUNDS | CC BY-NC 4.0 | 100 h egocentric with sound, 78K timed sound events | a kitchen scenario, if non-commercial suits |
| Gated: VIRAT, Ego4D, LSC/NTCIR, BirdCLEF (Kaggle), V3C | a signed agreement, an application or an account | | not without Dan |

Dan reads each licence before anything is downloaded. Standing permission covers installing and paid runs of popular public
benchmarks, never accepting a licence or making an account.

### 8.4 Changes to the arms from this research

- **`blind` becomes `listing`.** The agent gets the folder's listing (names, sizes, times, EXIF and container fields) but
  no pixels or sound. A question it answers is flagged as metadata-answerable and kept out of the headline. This is
  Daily-Omni's and LVBench's leak filter, applied to what a dump really leaks. Execs are short, about 1 minute.
- **`gold` (T2 only):** the agent gets the evidence clips alone, to split search misses from reading misses.

## 9. Fairness

- Both arms run the same agent, model, effort and turn cap, with no network.
- The `files` arm should get what a real user's agent has: Claude its whole shell, not "Bash for ffmpeg only". Today's
  restriction exists to stop escapes. The deny list can do that job alone.
- mm's model gets no hint the agent does not write. The agent gets mm's skill and the model's notes, as an installed mm
  gives them.

## 10. Phases

| Phase | What | Done when |
| --- | --- | --- |
| P0 | MMMB runs from this folder (vendor `conftest`'s `Api` and `Daemon`, `day` and `fake_decider`, or point at a Multimem checkout by setting); grounded answers; the `files+mm` and `listing` arms; canaries; the cached build; arms side by side; the GPU queue with its working-time clock, checked on 10 questions; count the pilot's repeated windows | the pilot reruns in about 1.5 h with the same verdicts |
| P1 | Track L (large files), S2 household sound, S1 home security | each has 30+ questions, a passing probe and a T1 run |
| P2 | S0 scripted day, S3 phone roll, S4 robot logs, scaling curves, session mode | the T3 run has 70+ discriminating questions per agent |
| P3 | S5 meetings, S6 dashcam, S7 field recorder | as P1 |

## 11. Open questions for Dan

1. **Licences.** Each dataset in section 8: Dan reads and accepts. No agent does.
2. **CPU work beside a GPU run:** may `files` execs (ffmpeg on the CPU) run while an mm arm uses the GPU?
3. **The answer cache** (DESIGN.md §16, deferred): build it if the pilot's own `tools.json` shows agents re-reading
   windows? And may windows be batched once batched verdicts are shown to equal single ones?
4. **Headline arm:** `files` against `files+mm`, with `mm` alone only on the smoke tier?
5. **Big files in mm:** should `remember` cut a file over `remember_max_mb` into segments at its true times, and should a
   folder source date an in-place file by its `creation_time`? MMMB's track L will show what breaks first.
6. **Dan's own data:** a phone dump and robot logs, with questions he writes, kept private and never published.
7. **Where MMMB lives:** its own repository here, or Multimem's `benchmarks/` with this folder as a copy?
