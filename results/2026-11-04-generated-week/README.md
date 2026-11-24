# The generated week, with and without a search system (4 November 2026)

Does an agent answer questions about a person's recordings better with a search system than with the same files and its own
tools? 24 questions about 3 generated days (scenario `v7_screen`, scale `pilot`, seed 1), each asked of each agent in two
configurations, in a fresh agent session each time:

- **`files`:** the recordings are plain files in a folder (photos, screen recordings, camera clips, sound, notes, logs). The
  agent uses its own tools. Claude Code has Read, Glob, Grep and Bash for ffmpeg. Codex has its shell (in its Windows sandbox) and
  `view_image`.
- **`mm/d1-3B`:** the same recordings are ingested into [MultimediaSearch](https://github.com/DanMcInerney/MultimediaSearch).
  The agent has only its tools (`ask`, `find`, `look`, `label`), plus its skill and the model's notes. `ask` has a small local
  model (d1-3B) look at every window in a time range on the GPU and return the moments that match.

Answers are graded by fixed rules (numbers, codes, times within a tolerance), not by another model. [report.md](report.md) has
every table and a row for each question, and [shortcuts.md](shortcuts.md) the output of the shortcut probe.

The week was written by generator 6. The generator is now 7 and draws a different week from the same seed, so a new run asks
about other needles and is not a repeat of this one.

## Results

24 questions about 3 synthetic days, each asked of each agent in both configurations. 16 of them are media questions (photos and video). The
other 8 are text and records, which both configurations got right throughout.

| Agent | Right, `files` | Right, `mm` | Media only | Median time per question | Tokens | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| **Codex** (gpt-6.1-terra, medium effort) | 22/24 (92%) | **23/24 (96%)** | 14/16 → 15/16 | 28 s → **19 s** | 5.3M → **2.7M** | (not reported) |
| **Claude Code** (claude-sonnet-5-5) | 18/24 (75%) | **22/24 (92%)** | 10/16 → 14/16 | 58 s → **32 s** | 29.9M → **3.9M**¹ | $13.48 → **$2.70** |

¹ A lower bound: an exec that timed out records no tokens (two Claude mm execs did).

**What it says:**

- **mm never cost a right answer.** No question was right with the files alone and wrong with mm. Codex gained 1 question and Claude 4.
- **Video is where the gap is.** Claude went from 3/9 to 7/9 on video, Codex from 7/9 to 8/9. Text, records and photos were 100%
  in both configurations for both agents.
- **Faster and cheaper.** With mm, Codex used half the tokens and Claude about an eighth. Both answered in a third to a half less
  time. Claude's estimated cost fell from $13.48 to $2.70 for the 24 questions.
- **mm adds GPU time.** That cost is local, not tokens. Per correct media answer: Codex 77 GPU seconds, Claude 179.
- **Not yet significant.** With 24 paired questions, the gains are McNemar p = 1.0 (Codex, 1 to 0) and p = 0.125 (Claude, 4 to 0).
  A difference this size needs about 6 discordant questions all one way to reach p < 0.05. This is one seed of the pilot scale:
  read it as a strong sign, not a finding. `mmsb report` pools more seeds.

Cost per correct media answer, from the report:

| Agent | Configuration | Correct | Seconds per correct | Tokens per correct | GPU seconds per correct |
| --- | --- | --- | --- | --- | --- |
| Claude | `files` | 10/16 | 420 | 2.84M | - |
| Claude | `mm` | 14/16 | 201 | (lower bound, see ¹) | 179 |
| Codex | `files` | 14/16 | 85 | 353k | - |
| Codex | `mm` | 15/16 | 95 | 163k | 77 |

### Where both configurations failed

- **"When did the error screen last appear?" over all three days** (`video-screen-error-3`): wrong for both agents in both configurations.
  The answer is in about 47,600 screen frames, among decoys. With the files alone, Codex answered "unknown" and Claude hit its turn cap.
  With mm, one `ask` over three days of screen needs more GPU time than a question's 900 s limit allows (the run flags this
  before it starts). Claude timed out, and Codex answered "unknown". Narrowing the range first would have worked; neither agent
  did it in time.
- `video-absent-error-2` asks for the error code on Tuesday 3 November, when there was no error screen (the right answer says
  so). Codex got it right in both configurations. Claude hit its turn cap with the files alone and timed out with mm, on the other ask flagged as
  past the limit.

### What was not tested

- **Sound.** d1-3B, mm's default model, does not hear. The 7 sound questions (doorbell, smoke alarm) ran with the files alone only: Codex
  5/7, Claude 5/7. `d1-Omni-2B` hears, and was not in this run.
- **Other decision models.** Only d1-3B.

## How each configuration finds an answer: four questions side by side

With the files alone an agent does not read every frame. The agents are resourceful. They use file names as timestamps, tile frames into
contact sheets with ffmpeg (or System.Drawing in Codex's case), and write their own detectors: a colour counter for red pixels,
ffmpeg's `signalstats` to pick saturated frames, band filters and `silencedetect` to find a beep, spectrogram images to look at.
With mm, the pattern is `find` (what is recorded when), one `ask` (the model reads every window in the range and returns the
likeliest moments), then `look` at the best ones to check.

**1. "What number was on the last green sign that I took a photo of?"** (photos over three days; answer 585)

| | Codex, `files` | Codex, `mm` | Claude, `files` | Claude, `mm` |
| --- | --- | --- | --- | --- |
| Result | right | right | right | right |
| Time | 29 s | 19 s | 90 s | 50 s |
| Tokens | 111k | 53k | 671k | 274k |
| Cost | - | - | $0.38 | $0.19 |
| How | tiled the photos 60 to a sheet, newest first, and opened 2 sheets | 1 ask: 113 windows, 7 GPU s | 27 calls: 9 ffmpeg contact sheets and 2 photos opened | 4 asks: 292 windows, 19 GPU s |

**2. "On what day and at what time was a parcel last left at the front door?"** (59 porch-camera clips over three days;
answer Monday 07:30)

| | Codex, `files` | Codex, `mm` | Claude, `files` | Claude, `mm` |
| --- | --- | --- | --- | --- |
| Result | right | right | **wrong** (Tuesday 19:39) | right |
| Time | 77 s | 58 s | 114 s | 75 s |
| Tokens | 445k | 95k | 750k | 270k |
| Cost | - | - | $0.42 | $0.20 |
| How | one frame of each clip on a contact sheet, then strips of the likely clips: 16 calls, 6 pictures | 1 ask: 600 windows, 37 GPU s | 51 calls, 35 pictures opened | 2 asks: 626 windows, 42 GPU s |

**3. "What was the error code on the red error screen on Sunday 1 November, and at what time did it first appear?"** (one
day of screen recording: 390 one-minute files, about 25,600 frames, with red warning dialogs and banners as decoys; answer E-8755
at 14:01:22)

| | Codex, `files` | Codex, `mm` | Claude, `files` | Claude, `mm` |
| --- | --- | --- | --- | --- |
| Result | **wrong** ("unknown") | right | right | right |
| Time | 124 s | 253 s | 827 s | 170 s |
| Tokens | 576k | 416k | 5.5M | 400k |
| Cost | - | - | $2.36 | $0.23 |
| How | wrote a red-pixel scanner in C#, tried 3 thresholds, opened 5 frames, found nothing it trusted | 2 asks: 3,690 windows, 237 GPU s | 109 calls: ffmpeg's saturation measure (`signalstats`) to pick frames, 11 opened | 2 asks: 2,013 windows, 140 GPU s |

This is the clearest case. With the files alone, Claude got there by brute force, taking 5 times as long, 14 times the tokens and 10 times
the cost of its mm run.

**4. "At what time did the smoke alarm beep on Sunday 1 November, in the morning?"** (microphone; answer 07:18:57; with the files alone
only, since d1-3B does not hear)

| | Codex, `files` | Claude, `files` |
| --- | --- | --- |
| Result | right | right |
| Time | 43 s | 82 s |
| Tokens | 134k | 332k |
| Cost | - | $0.24 |
| How | ffmpeg band-pass (2 to 4 kHz) and `silencedetect`, then `astats` on the hits | 35 calls, 15 spectrogram images |

## What was asked: the generated week

The data is a generated week, written by the benchmark so that every answer is known. The `pilot` scale is 3 days, seed 1: 1,572 files, among them about 81,000 screen frames, 1,442 porch-camera frames, phone photos and clips,
a hallway microphone, notes, versions of a config file and the logs of three services.

- **The screen** changes like a real one. It has typing, scrolling, a moving cursor, a taskbar clock, idle and busy stretches, and
  up to 2 frames a second, cut into one-minute files as mm's recorder cuts them. Its rate of change was calibrated against two real
  hours of a developer's screen recording.
- **Each question has a needle** hidden among decoys that look or sound alike but mean something else: a red warning dialog or
  banner beside the red error dialog, a brown dog beside the parcel, other beeps beside the smoke alarm. Some questions ask, in the
  same words, about something that is not there; the right answer is that it is not there.
- **How vague the time hint is** varies: a part of a day ("on Sunday, in the morning"), a day, or no hint ("when did ... last
  ...?", over the whole week).
- **A shortcut probe** runs at build time. It checks that no cheap trick finds a needle alone: no colour filter, frame difference,
  band filter or file size puts it first. Its output for this run is in [shortcuts.md](shortcuts.md).
  A filter tuned to a needle's exact shade or pitch does find the rest, once an agent has found one. That is by design and
  reported, not checked.
- **The agent cannot cheat.** It cannot reach the generator, the answer key, mm's store (in `files`) or another
  question's working folder. A command that tries ends the question as `escaped`, which is graded wrong. So are timeouts (900 s
  per question) and Claude's turn cap (60).

### Read the results with these

- **Without a recorder, nothing would be recorded at all.** `files` reads a screen recording, camera clips and microphone audio
  that exist only because the benchmark wrote them. In real use MultimediaSearch is also the recorder: the comparison is between
  two ways of reading the same recordings, not between having it and having nothing.
- **Synthetic needles are easier to filter than real ones.** A drawn red dialog or a pure tone is easier to find with a colour
  or band filter than a real error or a real alarm in a noisy room. This favours `files`.
- **mm's screen asks cost a little more here than on a real screen.** mm's change gate keeps 98% of this screen's seconds as
  windows to read, against 71 to 86% on the real recording.
- **`look` is dearer than a contact sheet.** It returns separate pictures, up to 8 per call, where a 5x5 ffmpeg montage is one
  picture of about 1.5k tokens.
- **Tokens are a measure of work, not a bill.** They count input, output and, for Claude, cache reads and writes, and a cached
  token costs much less. The cost column is Claude's list-price estimate. Codex reports no cost.
- **The questions are not independent.** On the pilot scale, a template's part-of-day and day questions share a needle.
