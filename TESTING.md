# MMMB: testing

How the build is checked, what it is held to, and what has not been run. The design is in [DESIGN.md](DESIGN.md). Dated
notes are at the end, newest last; add to them whenever testing teaches something.

The suite has 1,078 tests: 854 under `tests/`, 93 under `tests/oracles/` and 131 under `tests/mm/` (counted after the G2 repair,
on 5 November 2026; five are slow). None starts a paid agent, uses the GPU, or needs the network.

## 1. Environment

```
uv sync                              # Python 3.12, the core, the dev group (pytest 9.2.0, pytest-xdist 3.8.0)
uv sync --extra mm                   # also Multimem, as an editable path: needed by tests/mm
uv sync --frozen --no-default-groups # what the neutrality test makes: no extra, no pytest
```

- **`--frozen`** uses `uv.lock` as it is and never locks again. The `mm` extra is a path to Multimem's checkout, which must
  exist for uv to lock, so a machine without Multimem cannot lock; `uv sync --frozen` works there. The lock holds torch
  2.11.0+cu128, torchvision 0.26.0+cu128 and transformers 5.21.0 for the `mm-gpu` group, which P0 did not install.
- **`uv sync` is exact:** it removes an extra you did not ask for. `uv run` does not. When several agents share the
  environment, run `.venv\Scripts\python.exe -m pytest`, so `uv run` does not sync it under them.
- **The pins matter.** `av==19.0.2`, `numpy==2.5.4`, `pillow==13.0.0` and `imageio-ffmpeg==0.7.0` are what the recorded
  pilot's files were written with; V7's bytes depend on them. ffmpeg is `imageio_ffmpeg.get_ffmpeg_exe()` alone: none is on
  PATH here, and the recorded sound files reproduce with it.
- **Time zone.** V7's plan is in local time, so the oracles on the recorded pilot hold only in US Eastern time. Daylight
  saving ended at 02:00 on 1 November 2026, so the UTC offsets at the midnights that start 1, 2 and 3 November and end the 3rd
  are -0400, -0500, -0500, -0500. Elsewhere those oracles skip, and say which offsets they need. Windows has no `tzset`, so a
  test cannot change zone itself.
- **Everything on D:.** Set `TEMP` and `TMP` to `D:/mmmb-temp/tmp`, give pytest `--basetemp=D:/mmmb-temp/pytest-<name>`, and
  `UV_CACHE_DIR=D:/mmmb-temp/uv-cache` for any uv call that could write its cache. C: swung between 1.8 GB and 23 GB free
  during the build because of other work on the machine. uv's cache is on C: and the environment on D:, so uv copies.
- **Test names.** None of `tests/`, `tests/oracles/` and `tests/mm/` has an `__init__.py`, and the test files import their helpers as
  siblings, so a test file's name must be unique across the three. That is why oracle 1a is `test_v7_hashes.py` and oracle 2 is
  `test_recorded_report.py`: `tests/test_v7_plan.py` and `tests/test_report.py` have the other names.

## 2. The limits

These hold for every test and every check:
- **No paid agent runs:** no real `claude -p`, no real `codex exec`. Agents in tests are stand-in scripts that print made-up or
  recorded events.
- **No GPU:** no decision model, no d1, no `-m gpu`. The laptop's NVIDIA driver has crashed under one GPU job and under three.
- **At most 4 pytest workers and 4 CPU processes.** A GPU run beside 12-worker suites hung the driver and crashed Windows once
  (Multimem, 30 October 2026). The default is `-n 4`.
- **Throwaway homes.** A session fixture points `HOME`, `USERPROFILE`, `CODEX_HOME`, `CLAUDE_CONFIG_DIR` and `MULTIMEM_HOME` into
  a temporary folder. An audit hook fails a test that writes the real `~/.claude`, `~/.codex`, `~/.multimem`, or the folders
  MMMB keeps its runs in (`D:\mmmb-runs`, `D:\mmmb-data`, `D:\mmmb-cache`), even when the code under test swallows the error. A
  test gives its commands an `--out`, `--temp` and `--cache` of its own. The same hook fails a test that starts a program named
  `claude` or `codex`, or any program inside those three folders (`subprocess`, `os.system`, `os.exec*`, `os.spawn*`, and the
  command a shell is given): a test that needs an agent puts a stand-in of another name in its place, and patches
  `claude.check_login`.
- **Read only:** the recorded runs in `D:\mm-bench` and `D:\mm-bench-temp`, Multimem, and `D:\bench-data`.
- **mm tests** start Multimem's daemon only with throwaway homes and the `python:` fake decider, never with a real model.

## 3. Markers and how to run

`pyproject.toml` makes the default `-n 4 --dist loadgroup -m "not slow and not paid and not gpu"`.

| Marker | Means | Tests |
| --- | --- | --- |
| `slow` | seconds to minutes: a byte-exact build, a fresh environment | 5, run with `-m slow` |
| `paid` | starts a real `claude` or `codex` and spends the plan's quota | none. The one old test of this kind was not ported |
| `gpu` | loads a model on the GPU | none |

```
uv run pytest                          # the fast suite: about 8 to 10 minutes (484 s for 1,048 tests at U14's end; 489 s for 1,073 after the G2 repair)
uv run pytest tests/mm --require-mm    # the tests that need the mm extra; a skip is a failure
uv run pytest -m slow <file>           # one slow test file (section 4)
```

**Without the mm extra,** every test under `tests/mm` skips itself, and the summary says so in a banner: "N mm tests SKIPPED: the
mm extra is not installed; run uv sync --extra mm". `--require-mm` turns that into a failure with exit status 1. In a fresh
environment with no extra, U8 saw 554 passed and 9 skipped, with the banner "88 mm tests SKIPPED". With the extra it saw 652
passed and 1 skipped in 261 s at 4 workers; the suite has grown since.

Tests that run pytest in a subprocess carry `@pytest.mark.xdist_group`.

## 4. The checks, and how to run each

Oracles 1, 2, 3 and 5 hold the code to the recorded pilot and EgoLife runs: the expected values are the recorded files', never
the code under test's, so a difference is a fault of the port, not of the expectation. Oracle 4 is the ported suite and N the
neutrality check.

| # | Checks | Command | Time |
| --- | --- | --- | --- |
| 1a | V7: the plan hash and files hash equal `run.json`'s; its 1,572 files and 3 logs are the 1,575 of the recorded manifest, each with the recorded path and `mtime_ns` | `uv run pytest tests/oracles/test_v7_hashes.py` | about 2 s |
| 1b | V7: 43 files written and compared by sha256, size and mtime. The rule: `random.Random(7)`, then in this order `sample(files of the kind, n)` for screen 12, photo 12, porch 6, mic 4, note 4, clip 3, config 2 | `uv run pytest -m slow tests/oracles/test_v7_sample.py` | 13 s (U13) |
| 1c | V7: the whole pilot week, 1,575 files, by sha256, size and mtime; the shortcut probe passes and writes `shortcuts.json` and `.md` equal to the recorded ones but for the seconds | `uv run pytest -m slow tests/oracles/test_v7_full.py --v7-full=D:/mmmb-temp/oracle1c --v7-workers=4 -n 0` | about 6 min to write at 4 workers, 1,377 worker-seconds, and 84 s to probe; about 8 min at the recorded build's 3 workers |
| 1 (V8) | V8: the plan and files hash equal the recorded run's; its 50 questions and its 2,592 clips and 3 transcript files are the recorded ones. Needs the download at `D:\bench-data\egolife`, and skips loudly without it. It builds no 41 GB view | `uv run pytest tests/oracles/test_v8_plan.py` (the clip table, about 1.5 min, is marked `slow`) | a few seconds |
| 2 | The report: the 110 pilot results (converted once) give, through `report.numbers` and `render.markdown`, the recorded `report.json` (through a short `legacy_view`) and `report.md` byte for byte. A planted flip of one `passed` makes it fail and moves `b` and `c` by one. Recorded answers regraded through `Question.grade` equal the recorded verdicts, 110 of 110 | `uv run pytest tests/oracles/test_recorded_report.py` | about 3 s |
| 2 (V8) | The same for EgoLife: 150 results and 50 retrieval records, regraded 150 of 150; the 150 recorded prompts are the expected V8 prompts. Needs no download and holds in any zone | `uv run pytest tests/oracles/test_recorded_egolife.py` | about 3 s |
| 3 | The exec the agents build: argv, environment, instructions, settings, `mcp.json` and catalog for the blind, files and server cells of Claude and Codex equal the goldens the old code wrote; for each of the 110 pilot execs, `settings.json`, instructions, `mcp.json`, the init event's tool list and the Codex catalog are what the agents write now. `files/text-note-code-1` is left out: `NOTES.txt` says two dry runs overwrote it. The catalog check reads `D:\mm-bench` and skips loudly without it | `uv run pytest tests/oracles/test_exec.py tests/oracles/test_goldens.py`; the `mm` cells: `uv run pytest tests/mm/test_exec_mm.py --require-mm` | a few seconds |
| 4 | The ported suite: the 245 tests of Multimem's `test_bench_rules`, `test_haystack`, `test_haystack_media`, `test_shortcuts`, `test_agents`, `test_codex`, `test_bench` and `test_egolife`, moved by what they test (section 7). Their expectations changed only by the intended differences of section 6 | `uv run pytest` | the whole fast suite, about 8 minutes |
| 5 | Replay: a stand-in prints a recorded `events.jsonl` and the real agents, watcher and classifier run over it inside the real `run_agents`. Classes: the 110 recorded execs when `D:\mm-bench` is there (104 normal, 3 timeouts that are killed, wrong and not asked again, 3 capped; 8 normal ones are checked in), a planted escape, a planted setup fault, and the 20 execs that ended on a lost GPU, which must come out as `setup` and be left to run again | `uv run pytest tests/oracles/test_replay.py` | about 1 min at 4 workers |
| N | Neutrality, fast: a process that refuses `multimem`, the tests' helpers and the old root modules imports the core, the scenarios and the systems but mm; two scans of the source. Slow: a fresh environment with no extra builds V7 tiny, probes it, runs a `files` cell with a stand-in, reports, and sees `mm` refused | `uv run pytest tests/test_neutral.py`; the slow one with `-m slow` | fast: seconds; slow: 36 s (U13) |

- **Agreement tests** (`tests/mm/test_agreement.py`, 16 tests, need the extra): what was copied from Multimem still agrees with
  it. They check the small PyAV reader in `v7_screen/frames.py` against Multimem's reader, the writer's constants against
  Multimem's and its fake decider's, the ffmpeg lookup, a clip's length, and the numbers the report quotes (`ASK_SHARE`,
  the decode second, the words `ask` refuses). They fail loudly when Multimem renames or changes one.
- **The recorded estimate** (`tests/mm/test_recorded_estimate.py`): `Session.estimate` rounds to the recorded `gpu_estimate_s`,
  and `exclusive` and `beyond_timeout` match, for all 48 pilot and 50 EgoLife `mm` results.
- **Seam tests** (`tests/mm/test_seam.py`): a stand-in `claude` or `codex` that speaks JSON-RPC to the real mm server through
  the real agent path (`tests/stand_in_agent.py` is the stand-in, shared with the listing tests).
- **How many execs run at once** (`tests/test_gpu_jobs.py`, 9 tests, and the cases of `test_repeat.py`, `test_clock_check.py` and
  `tests/mm/test_clock_mm.py`). The owner's rule is a table written out in the test, and the runs are read from the times the
  stand-in agent was called at: a system that needs the GPU runs one exec at a time at the defaults, `files` runs `--jobs`,
  `--gpu-jobs N` gives the GPU cell N and `--jobs` and `--clock wall` do not; the same in `--side-by-side`; the dry run prints
  the count each cell will run at and its wall estimate is the one worked out by hand from the counts the run was seen to run
  (for the recorded pilot, from its recorded seconds: `files` at 3 and `mm` at 1, and again at `--gpu-jobs 2`); a warning line is
  printed for several GPU execs and for no other; no refusal names `--clock wall`, each names one exec at a time by the option of its
  system; `clock-check` of a system that needs the GPU refuses at the defaults, naming `--gpu-jobs 2`, and its queued half runs at
  `--gpu-jobs` and warns. A half of `clock-check` denies the run it branched from (the run's `questions.jsonl`, `run.json` and the
  other half's results are reached by the deny list every exec of a half is given).
- **The working-time clock** (`tests/test_clock.py`, 23 tests; `tests/test_clock_check.py`, 13; `tests/mm/test_clock_mm.py`, 2).
  With `now` and `queue_s` injected, a case at 950 s of wall time with 100 s queued is not over, one at 901 s of working time
  is, and the wall cap fires whatever the queue. A system that gives no queue figure gives `clock: wall` and the recorded
  `timeout: no answer in 900 s`, byte for byte, in both agents. A queue read live moves the end of the limit. A real process is
  killed at a 0.2 s limit by each agent and the result holds `clock`, `working_s` and `queue_s`, while `seconds` stays the wall
  time. More than one exec at once on the working clock is refused for a session that cannot say its queue, by `run`, in a dry
  run too, and for both of mm's sessions. `budget_s` is checked at 300, 900 and between. For `clock-check`, over the stand-in
  system: each case is asked alone and then queued in two folders and the table is printed; the cases come from those that
  do not hold the system alone; and it refuses, before any agent starts and writing nothing, a system with no queue figure
  (naming D3), one that would run the cases one at a time (the option that asks for more is named), one that holds every case
  alone, one not ready, and a configuration that is not one system. A second command asks nothing, and a cut-off half goes on. None of this ran on a
  shared GPU: see section 8.
- **The listing's rule for a file's time** (`tests/test_listing_times.py`, 5 tests; `tests/oracles/test_listing_pilot_times.py`, 2).
  A file whose mtime is not within one second of the manifest's `t` (a log: not before its first line) is `misdated`, and the
  run is refused by every command. Over the fake V8 download (made today, of recordings of 29 and 30 October): every clip is
  refused and no transcript, at `build`, at `ingest` and at a later `run`, before any agent and naming both times. Over a planted
  dump, a file moved three days either way, or two seconds, is refused for that file and half a second is not. Over the recorded
  pilot's manifest, all 1,575 entries are at their time (1,572 files exactly, 3 logs 237,409 to 237,435 s after their first
  line), and a shift of 1.1 s refuses all 1,572 files.
- **Planted faults.** A suite counts only once it has been seen to fail on a planted fault: for the runner, two kinds swapped in
  `DONE_KINDS`; for `admits`, the heavy rule dropped; for deny, a word removed; for the report, a result flipped; for oracle 1,
  one colour constant changed. G1 planted 32 faults: 21 were caught and 11 escaped; the repair pass followed (section 9). The G2
  repair planted 24 more, one at a time and restored byte for byte, and all were caught (section 9).

**What waits for Dan** (the pilot re-run, the first `mm+files` run, the clock check) is in DESIGN.md section 12.2, with its
commands and costs.

## 5. The goldens

The recorded runs are the pilot (`D:\mm-bench\20261104-0314-pilot-s1`, 4 November 2026) and EgoLife
(`D:\mm-bench\20261103-0257-egolife-s1`, 3 November). Their small files are copied into `tests/oracles/recorded/` (2.7 MB), and
`recorded.json` there holds where each came from and the sha256 of every copied file. `tests/oracles/test_goldens.py` checks
the set as a whole before anything is held to it: that nobody edited a file, that there is an entry for each of the 110 execs,
that every file of the data folder is hashed once, and that the two sources of the exec goldens agree where they overlap.

**Capture commands.** Each refuses to write where its output exists. A golden is never regenerated to make a test pass: if one
is wrong, delete it on purpose and say why in the log below.

| Script | Wrote | Run |
| --- | --- | --- |
| `tests/oracles/capture_recorded.py` | `pilot-s1/`: byte copies of `run.json`, `data-manifest.json`, `shortcuts.json`, `shortcuts.md`, `report.json`, `report.md`; `sha256.json` (the sha256 of all 1,575 files of the data folder, hashed in one process); `execs.json` (what each of the 110 execs wrote before it ran); `instructions/<sha>.md`; `replay/` (8 small execs, 2 for each cell); `egolife-s1/{run.json,report.json,report.md}`; `recorded.json` | `uv run python tests\oracles\capture_recorded.py [--pilot RUN] [--egolife RUN] [--out recorded]` |
| `tests/oracles/capture_exec.py` | `exec/`: argv, environment, instructions, settings, `mcp.json` and catalog for Claude and Codex in the blind, files and mm modes, and `deny-v7.json`, `deny-v8.json`; `capture.json` says where it came from | `C:\Users\danhm\tools\Multimem\.venv\Scripts\python.exe tests\oracles\capture_exec.py`, with the old modules read by `git show bb498a6:<file>`. No CLI is started and nothing is asked |
| `tests/oracles/convert_recorded.py` | `pilot-s1/converted/` and `egolife-s1/converted/`: the results in the new format (`arm`, `model` become `config`, `system`, `variant`; nothing else changes), `questions.jsonl` from the new writers, `systems.json`, `run.json`; and `egolife-s1/prompts.json` | `uv run python tests\oracles\convert_recorded.py [pilot\|egolife\|egolife-prompts] [--run RUN] [--out DIR]`, in US Eastern time |

- **Why the exec goldens are captured and not read.** No recorded run holds an argv. Exec folders hold instructions, prompt,
  settings, `mcp.json`, catalog, events, tools, final and stderr, so the argv could only come from the code that built it.
- **Not checked in:** the recorded timeout, capped and GPU-lost replay sources (230 to 650 KB each), the 54 recorded Codex
  catalogs (648 KB each), and EgoLife's `clips.json` and `data-manifest.json`. Those checks read `D:\mm-bench` and skip loudly
  without it. The Codex catalogs have two sha256 values: `ac5d2260...` for the 30 `files` execs that were not overwritten, and
  `e67870b7...` for the 24 `mm` execs and the overwritten one.
- **Two things that are assumptions.** The golden Claude argv has `--effort high`, the module's default, because no recorded
  result holds Claude's effort (`agent_effort` is `None`). The golden for mm's instructions is built from Multimem's current skill,
  41 characters longer than the recorded one (section 6, difference 3).
- **Deny-list goldens** use the sentinels `D:\__RUN__`, `D:\__TEMP__`, `D:\__OUT__`, `D:\__ROOT__` and the tag `TAG`.

## 6. Intended differences

Each is asserted by a test, not ignored. The expected exec of a recorded run differs from the new one in these ways only.

| # | Difference | Reason |
| --- | --- | --- |
| 1 | The server command's python path | a new venv |
| 2 | Run, home, slot and data paths | a new run; the goldens use placeholders |
| 3 | 5 lines of mm's skill: "nervelet simulator" is "robot simulator", "this machine" is "the developer's machine", 41 characters in all | Multimem changed since the pilot. An exact line map in `tests/mm/test_exec_mm.py` fails, saying which lines, when the skill changes again |
| 4 | `MM_CLAUDE_MODEL`, `MM_CLAUDE_EFFORT`, `MM_CODEX_MODEL`, `MM_CODEX_EFFORT` are `MMMB_*` | the names should not say mm |
| 5 | V7's prompt stamp: the pilot used the real clock (4 November, 11:27 to 15:01), the new one a fixed `Question.now`, noon on the day after the last day. The test masks that first sentence for V7 only | DESIGN 6.1 wants a fixed now (C12) |
| 6 | V7's `clock_note` when the wall clock is over an hour past `now`. V8 recorded it always, so V8 asserts it | the sentence is true then; mm's `searched.now` shows the real clock |
| 7 | Deny words: `/mmmb-runs/`, `/mmmb-cache/` added; `/mm-bench/` stays, with `mm-bench-temp` and this repository's main checkout; module words added; `.multimem` comes from `system.toml` | the new layout; the old answers stay on the same disk (G1 F1) |
| 8 | `timeout_s` is in every result, not only V8's | the core records it |
| 9 | The `mm+files` cell has no golden: its argv, settings and instructions are written by hand from the design, and its `line` prompt ("unknown when mm and these files do not hold it") is pinned by a test of its literal text | it was never recorded |
| 10 | A grader spec reads `rule` and `tol_s` where the old read `kind` and `tol_min`; a spec with no `date` key is no longer possible | the question file (DESIGN 6.1) |
| 11 | The window count: V8's clips by a span rule, a sound by one window for each of its windows | see DESIGN 10.1 (C5). The estimate still equals the recorded one for every `mm` result |
| 12 | A retrieval record's `warnings` is a sorted set and `seconds` is rounded to 2 decimals | the contract returns them so |
| 13 | A skipped mm model is an error, "No agent results yet" is an error naming `systems.json`, `--controls all` is gone, the limits line says "its work" and "GPU seconds", and `show_agents` counts every configuration at `--jobs` (the old "mm at 1 job" is the system's to lower, and says so). There is no standalone `probe` command | the move to configurations and `Session` |
| 14 | The runner reads the answer from the agent's final message; the agents no longer return `answer` | grounded answers (DESIGN 6.2) |

## 7. Tests not ported

Multimem's 245 tests were moved by what they test, and a ported file's first line names its source and commit (Multimem at
`fe0ece5`). Their names do not map one to one: by name, 42 are not found in the new suite, because "arm" became
"configuration", or a test was split between the core and `tests/mm`, or its subject went. They were not checked one by one.
What the files themselves and the unit notes say was left out:

| Left out | Why |
| --- | --- |
| `test_codex_recalls_the_day` | a paid test: 19 questions to a real `codex exec` over the old recall tier. It and its fixtures went |
| `test_bench_rules`: a grader case with no `date` key | a `Grade` cannot lack one; the question loader refuses a line without it |
| `test_agents`: two refusals about a missing home or working folder | the spec always has them |
| `test_agents`, `test_codex`: `guidance="none"` | it went with the old recall CLI (`codex_run.main` and `prompt`), which nothing in this repository called |
| `test_the_haystacks_open_refuses_a_run_of_another_dataset` | the `dataset` key of `run.json` and the two commands (`bench.py`, `egolife.py`) are gone; `mmmb` opens any run by its `scenario` |
| `test_a_run_is_built_with_neutral_frozen_data` (V8) | a frozen dump is the cache's now, and `tests/test_cache.py` tests it |

Moved, not dropped: the junction test to `tests/test_fs.py`; the lost-GPU test to `tests/test_runner.py` (it is a rule of the core,
keyed on `Description.needs_gpu`); the real agents' timeout test to the agents' tests (the spec carries the timeout); the home,
daemon, scrub, window-estimate and model tests to `tests/mm` (128 tests); the V8 tests that need a daemon to
`tests/mm/test_v8_mm.py`; the dry-run print of asks that may not finish, to `tests/test_cli.py`. A test that mixed the two halves
(`test_a_run_is_built_copied_and_continued`, `test_a_models_family_its_jobs_and_the_pilot_come_from_names_abouts_and_question_fields`,
`test_models_are_resolved_the_gpu_is_checked_and_the_prompt_differs_only_in_the_arm`) was split, its neutral half in the core's
tests and its mm half in `tests/mm`. Two (`test_a_files_exec_may_open_only_its_own_slot_and_its_folders_have_no_name_that_tells_the_week`
and `test_a_changed_distractor_with_no_version_bump_is_caught_by_the_files_hash`) have successors with other names in
`tests/test_runfolder.py`.

## 8. Not verified

Nothing in P0 ran a real agent, the GPU, or the real model. So no test could show:
- real Claude and Codex answering the `json` request, the `mm+files` prompt, or the listing server; real Claude accepting a second
  `--add-dir` for mm's `looks` folder with `--restricted` (if it refuses, a `Read` is denied by Claude and not counted as an
  escape); real Claude with `--model claude-sonnet-5-5`;
- `mmmb repeat` end to end, and the times of the dump build and the ingest in the new code (none recorded);
- a V8 build of the whole dump from `D:\bench-data\egolife` (its plan is checked; its dump is not), and the refusal of `listing`
  there, which is seen only over the fake download (the real clips' mtimes are the download's, 3 November, against 29 to 31 October);
- the CLI end to end with the real mm and the GPU;
- the pilot-scale rename in the cache (antivirus could lock a folder during `os.replace`, and there is no retry), and the POSIX
  branch of the cache lock;
- the clock on a shared GPU, and the wall effect of its wait loop under load; Ctrl+C during `clock-check` or a side-by-side run;
- the wall-time saving of side-by-side (paper only, `simulate`); a POSIX daemon restart, which would inherit Claude's
  `CUDA_VISIBLE_DEVICES=-1` (it does not apply on this machine);
- `run --dry-run --agents codex`, which fails here because this machine's Codex does not list the pinned model `gpt-6.1-terra`
  (the old code did the same).

## 9. Log

### 2026-11-04: measured while planning

All offline, with Multimem's venv.

| Fact | Result |
| --- | --- |
| The copied modules against Multimem's current `benchmarks/` | no behaviour difference; differences were a path insert, docstrings, one default (`MODELS_DIR`) and unused names |
| Multimem since the pilot's commit | HEAD `fe0ece5` is 4 commits past `7895548` plus dirty; no functional scan change. The skill changed in 5 lines: Claude's instructions were 25,821 characters when recorded and 25,862 now; Codex's 27,364 and 27,405 |
| The V7 plan and files hash | reproduce exactly from the copy: plan `cfdb321d22e4...`, files `b1406b488821...`, in under 2 s |
| The V7 hash and the time zone | days 1 to 3 November cross the daylight-saving change (-0400 to -0500); Windows has no `tzset` |
| V7 files | 1,572 files and 3 logs, 1,575 manifest entries, 31 questions; the data folder is 465,254,840 bytes and matches the manifest in every file |
| V7 byte reproducibility | a 43-file sample rewrote sha256-identical to the recorded files. A full write is about 1,377 worker-seconds, about 5.7 min at 4 workers. The recorded build took 11:23 with load and probe, on a shared laptop |
| The probe | run again it gives no failures in 84 s; `shortcuts.json` identical but for `seconds`, `shortcuts.md` but for its seconds line |
| The old report code on the recorded runs | the pilot and EgoLife: 0 JSON differences, Markdown identical, about 3 s each, torch not imported. The repository's `results/*/report.json` is byte-equal to `D:\mm-bench`'s |
| Regrading the recorded answers | `grade(q.grade, r.answer) and problem is None` equals the recorded `passed` in 110 of 110 pilot and 150 of 150 EgoLife results |
| Argv | not recorded anywhere; captured from the old code |
| Multimem's remote | none (`git remote -v` prints nothing), so `mmmb[mm]` is a path source |
| Tooling | uv 0.9.10; `uv python list` shows 3.14, Multimem's venv has 3.12 |
| Fixture sizes | the pilot's results 105,580 bytes and EgoLife's 308,923 (its decider records 175 KB): small enough to check in |
| `mm` time arguments in the recorded runs | all 365 `since` and `until` are ISO; no relative ages; 2 `find` calls with no `until` |

The recorded totals the costs in DESIGN 12.2 rest on: the pilot's 110 execs took 13,044 session-seconds and 3,655 GPU-seconds
and cost $19.00 at list for 50.3M tokens, about 173 minutes of wall time plus 11 minutes to build. Per `mm` exec: Codex 63 s,
48 GPU-s, about 113k tokens; Claude 123 s, 104 GPU-s, $0.11, about 164k tokens.

### 2026-11-05: the P0 build

- **U0, the skeleton.** The environment (left without the `mm` extra, which U7 added), the pins, and the goldens captured from the
  recorded runs: 110 exec entries and 1,575 hashes. `D:\mm-bench` was only read.
- **Stage 1, U1 to U8.** The shared foundation, V7 without Multimem, the agents, the runner, the report, V8 without Multimem,
  `systems/mm`, the command and the deletion of the nine root files. At the end of U8: 652 passed and 1 skipped with
  `--require-mm -n 4` in 261 s; in a fresh environment with no extra, 554 passed and 9 skipped, the banner "88 mm tests
  SKIPPED", and `--require-mm` exit 1. U7 compared `systems/mm` with the old code function by function (an AST comparison): 37
  identical and 21 different on purpose.
- **G1, an independent review.** Every oracle passed in the reviewer's own hands. The whole pilot dump built by the command in an
  environment with no Multimem: 1,575 files, 0 differ. Ingested through `Mm.ingest`: 1,572 items and 6,029 records, 0 differ from
  the recorded index. The goldens, captured again from the old code, were genuine. 32 faults were planted: 21 caught and 11
  escaped. It made 14 findings (F1 to F14), among them: an agent could read the pilot's answers on the same disk (F1); a rewrite
  of the data froze other bytes into the run (F2); no file said which Multimem and skill a result ran with (F3); and opening
  the file `look` names counted as an escape (F4).
- **The G1 repair.** All of F1 to F14, a seam helper, and the masking of `close`'s error. The suite: 696 passed, 1 skipped.
- **Stage 2, U9 to U13.** Grounded answers; case selection, `compare` and `repeat`; `listing` (957 passed and 1 skipped in 11:33
  at `-n 2`; the slow neutrality test passed in a fresh environment); cached ingest; side-by-side scheduling (1,011 passed and 1
  skipped in 7:33). The dry run of the pilot repeat prints 110 execs, 13,044 session-seconds and 3,655 GPU-seconds, equal to the
  recorded ones.
- **U14, the working-time clock.** `--clock` on `run` and `repeat`, `mmmb clock-check`, and one wait loop for both agents. 1,048
  passed and 1 skipped in 484 s; the slow neutrality test passed in 55 s; 39 planted faults were all caught. No golden changed.
  Two of U14's own test runs called the real `claude auth status` (read only, in the tests' throwaway home) before the fixture
  patched `check_login`.
- **U15, these documents,** were written beside U14 and corrected from its notes. The suite then collected 1,054 tests
  (`pytest --collect-only -q -n 0 -m ""`).

Not run in P0: see section 8 and DESIGN.md section 12.

### 2026-11-05: the G2 repair

G2's review passed every oracle and found one blocker. The repair, in one pass, with one coder:
- **Finding 1, execs at once on the GPU.** `core/atonce.py` is the one rule (`--gpu-jobs`, default 1, for a system that needs the GPU;
  `--jobs` for the others), asked by the runner, `show_agents`, `show_limits`, `check_clocks`, `show_repeat`, `check_queue` and
  `clock-check`; `repeat` passes both options to its run. The refusals name `--gpu-jobs 1` or `--jobs 1`, never `--clock wall`.
- **Finding 2**, the listing refuses a dump whose files are not at their time (tolerance one second); DESIGN 3.2 and 7.3 say so.
  **Finding 3**, a clock-check half denies its parent run (`RunFolder.parent`). **Finding 4**, `EXCLUSIVE`, `GPU_ESTIMATE_S` and
  `BEYOND_TIMEOUT` are in `core/contract.py`, and mm imports them. **Finding 5**, `grounded.check_groundable` refuses a `json` run
  with a question that cannot be grounded, in `run_agents` and the dry run. **Finding 6**, the unused import and `resolve_models`
  are gone. **Finding 7**, the README and DESIGN 3.2, 12.1 and 12.2 were corrected. **Finding 8**, three printed wordings.
- **The guard.** The audit hook of `tests/conftest.py` now also refuses to start a real `claude` or `codex` or a program in the
  owner's folders. Before it, nothing did: two test runs in U14 had reached the real `claude auth status`.
- **Tests changed because the owner's rule did:** `test_clock_mm.py` (the refusal of mm at `--jobs 3` is now at `--gpu-jobs 2`; the
  dry run at `--jobs 3 --clock wall` and the message that named `--clock wall` are gone; clock-check at `--jobs 1` said "1 at a
  time, not 1"), `test_repeat_pilot.py` (`run_argv` takes `--gpu-jobs`; the dry run's counts are now checked against the
  recorded seconds), `test_repeat.py` (`run_argv`; the refusal of a system with no queue figure now names `--jobs 1` and not
  `--clock wall`, and two tests are new for a system that needs the GPU), `test_clock.py` (`check_queue` takes the description),
  `test_clock_check.py` (the "one at a time" message), `test_runner.py` (`show_agents` takes `gpu_jobs`), `test_cli.py` (`JOBS`
  moved to `atonce`), `listing_world.py` and `test_listing.py` (an entry's `t` is its file's mtime plus 0.4375 s, since the
  listing now compares them; the leak check still finds a copy of `t`), `test_mm_arm.py` (`resolve_models` was removed; its test asks
  `Mm.ready_model`), `test_conftest.py` and `conftest.py` (the guard).
- **Result, run in US Eastern time with `--require-mm -n 4`:** 1,072 passed and 1 skipped in 489 s (8:09). The recorded-report, recorded-EgoLife, exec, goldens,
  replay oracles and the slow neutrality and V7 sample tests passed, unchanged. No golden or recorded file changed.
- **24 faults were planted**, one at a time in the worktree and restored byte for byte (a hash of `mmmb/` and `tests/` was the
  same before and after), and all were caught: the rule ignoring `needs_gpu`; the runner, the dry runs, `show_repeat` and
  clock-check each asking `--jobs` of a GPU system; `run_argv` and `run_command` dropping `--gpu-jobs`; the refusal offering the
  wall clock or naming the wrong option; no warning; the listing reporting nothing misdated, a 3 s tolerance, a log held to a
  point; the half not naming its parent; two result keys changed in `contract.py` (the recorded report oracle caught both);
  `check_groundable` never raising or never called; three faults in the guard; the old wording of the questions. One planted fault
  was left in `mmmb/cli.py` for a minute when the first run of the script was stopped by hand, and put back from its exact text.
- Other projects' pytest runs (rtb-fleet) were running on the laptop beside these runs, so the times above are slower than U14's.
