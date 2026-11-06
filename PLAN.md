# MMMB: the plan to production readiness

Agreed with Dan on 5 November 2026, then re-planned the same day: MMMB is a standalone benchmark of realistic media, not
Multimem's benchmark separated out. Not started.

## The goal

MMMB measures how well an AI agent finds things in a real pile of media: long videos, lots of pictures, a long audio file, many
short sound files. Today the code runs without Multimem, but everything it can run came from Multimem's work: V7 (the synthetic
week mm was tuned on), V8 (EgoLife, run through mm), and mm as the only search system. None of the realistic piles has a
scenario, though their media is on disk in `D:\bench-data\sources\`.

Production-ready means:
- one realistic scenario for each media shape, built from real recordings, with every answer taken from the source's own labels;
- each runs with real Claude and Codex in the `files` baseline and the `listing` leak check, and any plug-in system (mm is one,
  optional);
- a report that reads right for any run, with no system named in the core;
- documents that say what is verified and what still needs a paid or GPU run.

V7, V8 and mm stay as they are, as extras.

## Limits

- No GPU job, no mm model, no `--group mm-gpu` install.
- No full benchmark runs and no long tests. End-to-end checks are the bare minimum, as fast as possible.
- Keep it simple: it is a multimedia memory benchmark.

## The four scenarios

| Shape | Scenario | Source on disk | Example question | Answers from |
| --- | --- | --- | --- | --- |
| Long videos | `v1_security` | MEVA: 3 overlapping security cameras, 75 min each, plus their infrared twins and a night clip (7.5 GB) | "When did someone open a car trunk on the parking-lot camera?" | activity spans, to the frame |
| Lots of pictures | `i3_trailcam` | Caltech Camera Traps, location 23: 4,256 photos over 6 months, about 80% empty (1.9 GB) | "Which nights in October did a deer come?" "Was there ever a bear?" | species and capture time per photo |
| One long audio file | `a1_home_audio` | SINS node 7: 24 real hours of one living room (4.2 GB) | "When did the vacuum run?" "When did the visitor arrive?" | activity spans (TV, vacuum, visit, away, cooking) |
| Many short sound files | `m4_samples` | FSD50K evaluation split: 10,231 clips, 200 classes (11 GB) | "Find every clip of glass breaking." | human-checked clip labels |

Each scenario:
- copies its files into a dump under the names a real device or folder would give them, with each file's time set to its
  recording time (copies, not links, so the source stays untouched; D: has 558 GB free). MEVA's 5-minute clips are joined into
  hourly MP4s per camera, SINS's minute files into one 24-hour WAV, and FSD50K's clips get neutral names;
- asks 20-30 questions at full size, about a quarter about things that never happened: a time window with no such label, a
  class left out of the dump, a species the camera never saw;
- has a `tiny` size of a few files for the tests, built in seconds from the same source;
- marks about 30% of its questions `dev` and 70% `test`, by a seeded draw.

Defaults taken, each easy to change later:
- A1 asks about labelled activities only. Mixing alarms, a doorbell or a dog into the room is later work.
- I3 keeps the camera's own clock, which writes midnight as 12:xx (real mess), and asks no question that rests on those
  frames.
- SINS keeps one channel as recorded, its DC offset included.

## What gets built

1. **A shared builder for real dumps** (`mmmb/scenarios/realdump/`), so the four scenarios stay small: copying and naming files
   with their times, the manifest, a simple shortcut probe (no file name or metadata field carries a label, and needle files are
   not singled out by size), the prompt, the view, the deny words. Each scenario is then a reader of its source and labels plus
   question templates.
2. **One new grading rule, `set`,** for "find every one" questions (clips, nights): right only with the exact set; the F1 is
   recorded beside it. Every other question uses the rules there are (`time`, `count`, `number`, `absent`).
3. **A neutral report in the core.** The core's report today is V7's mm report ("mm benchmark: with mm and without", "decider
   tier"), and it has nothing to say about a run of `files` alone. The new report gives, per scenario, agent and configuration:
   right out of n with its interval, grounded and lucky answers, wrong answers on absent questions, seconds, tokens and dollars
   per correct answer, the questions `listing` answered (leaks), and, when a system is in the run, its paired comparison with
   `files`. V7's report moves into `v7_screen` unchanged, so it still matches the recorded pilot byte for byte.

## How it is checked

- Each scenario's tests build its `tiny` size from the real source on disk (they skip loudly without it, as V8's do).
- Every question's answer is derived again in the tests by a separate, plain reader of the label files, never copied from the
  builder.
- The grader accepts each truth and rejects planted wrong answers: a wrong night, a missing clip, an extra clip, a time outside
  the tolerance, "unknown" on a needle.
- Each full dump is built once and probed (CPU and disk only, minutes each).
- One small real-agent test: `i3_trailcam` tiny, `files` and `listing`, Claude and Codex, 2 questions: 8 execs, about 10
  minutes, likely under $1 of Claude and a little Codex quota.
- The fast suite once at the end (about 8 minutes, 4 CPU workers, no GPU). Targeted tests only before that.

## Order of work

1. **Unit 1, done directly:** the shared builder, the `set` rule, the neutral report (and V7's moved), and `i3_trailcam`, the
   simplest scenario, as the first user of all three. Commit.
2. **Gate 1:** one independent review of unit 1 by a fresh agent, then at most one repair pass. The three other scenarios build
   on it, so a mistake here would be copied three times.
3. **Unit 2, three makers at once:** `v1_security`, `a1_home_audio` and `m4_samples`, each by a fresh agent on its own folder and
   tests only, each running its tests on one worker (three workers in all, within the laptop's limit of 4). I own the shared files
   (the registry, the core, the documents). Join, run their tests together, build each full dump once. Commit.
4. **Gate 2:** one independent review of the whole change, then at most one repair pass.
5. **The real-agent test** above; fix what it shows.
6. **Documents:** README (a standalone benchmark, the scenarios, how to run each), DESIGN (what is built), TESTING (the new
   tests, the real-agent test as a recipe to rerun after each CLI update), MEDIA (which sources are now used).
7. **The fast suite once, and the last commit.** No push without Dan's go-ahead.

## Not in this plan

| Left out | Why |
| --- | --- |
| Search systems besides mm (`embed`, `transcript`) | the `files` baseline needs none; each is a later plug-in |
| The pilot re-run, the first `mm+files` run, the clock check | they need the GPU or a full paid run; their commands are in DESIGN section 12.2 |
| The other scenarios (V2-V6, I1, I2, I4-I6, A2-A5, M1-M3, M5, M6, X1, X2) and the large-file track | one per shape first |
| Mixed-in sound needles (alarms, doorbells, dogs) | they need the mixing method of DESIGN 7.1; A1 starts from its labels |
| V8's four rotations of multiple-choice options | V8 is an extra now; do it when it is next run |
| The Pareto plot, the Lifelog Search Challenge score, Holm's correction, bootstrap rankings, session mode, `gold`, the size ladder | nothing needs them for a first release |
| Publishing, a licence and CI; mm's instructions over the 2,000-token rule; where mm's plug-in lives | Dan's decisions (DESIGN section 13) |
