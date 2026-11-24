# MMSB: testing

How MultiMediaSearchBench is tested, what the tests need, and what they do not cover. The design is in
[DESIGN.md](DESIGN.md).

## 1. Running the tests

```
uv sync
uv run pytest
```

`pyproject.toml` gives pytest the options `-n 4 --dist loadgroup -m 'not slow and not paid and not gpu'`, so a plain
`uv run pytest` runs the fast tests on 4 workers. None of them starts a real agent, uses the GPU or reaches the network.

- **Time:** about 6 minutes on Windows at 4 workers, and under 2 minutes on Linux at 8 (`uv run pytest -n 8`).
- **Why 4 workers:** a suite at 12 workers beside a GPU job once hung the GPU driver and crashed Windows. Ask for more with
  `-n` only when nothing else is using the GPU.
- **`--dist loadgroup`:** test files that share a costly fixture, or that run pytest or a command in a subprocess, carry
  `@pytest.mark.xdist_group`, and xdist hands out the tests of a group together.
- **CI** (`.github/workflows/test.yml`) runs `uv sync --locked` and `uv run --no-sync pytest` on Windows, Linux and macOS, for
  each push to `main` and each pull request. It installs no extra and downloads no source, so there the tests in `tests/mm`
  and the tests that read a real source skip.

Three markers are left out of the default run. A marker on the command line replaces the default selection.

| Marker | Means | Run with | Tests today |
| --- | --- | --- | --- |
| `slow` | seconds to minutes, such as a fresh environment made with uv | `uv run pytest -m slow` | one, in `tests/test_neutral.py` |
| `paid` | starts a real `claude` or `codex` and spends the plan's quota | `uv run pytest -m paid` | none |
| `gpu` | loads a model on the GPU | `uv run pytest -m gpu` | none |

One option is the suite's own: `--require-mm` fails, instead of skipping, the tests in `tests/mm` when the `mm` extra is not
installed.

```
uv run pytest tests/test_grade.py        # one file
uv run pytest tests/mm --require-mm      # the tests of the mm system; a missing extra is a failure
uv run pytest -m slow                    # the slow test
```

## 2. What the tests need

| Tests | Need | Without it |
| --- | --- | --- |
| most of `tests/` | `uv sync` alone | |
| the real scenarios (`tests/test_v1.py`, `test_i3.py`, `test_a1.py` and the others named after a scenario) | that scenario's source on disk, in the folder `MMSB_SOURCES` names (else `mmsb-sources` in the MMSB home); `uv run mmsb fetch --scenario <name>` downloads it | the tests that read the source skip and give the folder they looked in. The tests of the same files that work on made-up data still run |
| `tests/mm` | the `mm` extra: `uv sync --extra mm` | every file skips, and the summary ends with a banner: "N mm tests SKIPPED: the mm extra is not installed; run uv sync --extra mm" |
| the slow test | `uv` on PATH, and the network if uv's cache lacks a package | it skips |

- **Throwaway homes.** For the whole session `HOME`, `USERPROFILE`, `CODEX_HOME`, `CLAUDE_CONFIG_DIR` and `MMS_HOME` lead into
  a temporary folder, so code that looks for your agent setup finds an empty one.
- **The guard.** An audit hook in `tests/conftest.py` fails any test that changes your real `~/.claude`, `~/.codex` or
  `~/.mms`, or the folders MMSB keeps real runs, data and cache in (`mmsb-runs`, `mmsb-data` and `mmsb-cache` under
  `MMSB_HOME`, else your home folder). The test fails even when the code under test swallowed the error. A test gives its
  commands an `--out`, `--temp` and `--cache` under its own `tmp_path`.
- **No real agent.** The same hook fails a test that starts a program named `claude` or `codex`, or any program inside those
  three folders, whether directly or through a shell. A test that needs an agent uses a stand-in of another name.
- **Where the tests write.** Each test writes under pytest's `tmp_path`. The slow test makes its environment in `mmsb-temp`
  under `MMSB_HOME`; a run denies its agents that folder, because the tests' questions hold answers. To keep the tests'
  files off the system drive, pass `--basetemp` a folder under `mmsb-temp`.
- **The mm tests** start MultimediaSearch's daemon only over a folder under pytest's temp folder, with a fake decision model
  that needs no GPU. Fixtures in `tests/mm/conftest.py` keep them off Hugging Face and off a real `uv`.
- **Windows only.** Some tests skip elsewhere, each with its reason, among them directory junctions and Windows permissions
  (`tests/test_fs.py`), Git for Windows (`tests/test_agents.py`) and drive letters (`tests/test_cache_cli.py`).
- **Pinned packages.** `av`, `numpy`, `pillow` and `imageio-ffmpeg` are pinned because the bytes of the generated V7 week
  depend on them. ffmpeg comes from `imageio-ffmpeg`; none needs to be on PATH.

## 3. How the tests are laid out

`tests/` and `tests/mm/` have no `__init__.py`, and test files import their helpers as siblings. The two folders share one
namespace, so every file name must be unique across both (`tests/test_clock.py`, `tests/mm/test_clock_mm.py`).

### Stand-ins

No test runs a real agent or a real model. These take their places:

| File | What it is |
| --- | --- |
| `tests/stub_system.py` | a system, a session and an agent for the tests of the runner, with the folders of a run. It starts no process |
| `tests/echo_agent.py` | agents whose result says the model and effort they were given, as the real ones' results do |
| `tests/stand_in_agent.py` | a `claude` or `codex` process run through the real agent code: it finds the server as the agent does, starts it, speaks JSON-RPC to it and writes the stream the agent writes |
| `tests/agent_spec.py` | the session specs of the four modes of an exec (blind, files, a server, a server and files), written by hand |
| `tests/ghost_system.py` | a system whose package is not installed, for the refusal of a missing extra |
| `tests/cli_world.py` | a registry with the system `stub` and the scenario `toy`: V7's plan over a dump of three small files |
| `tests/cache_world.py` | a scenario that counts its builds and a system that counts its ingests |
| `tests/listing_world.py` | a small dump with secrets planted in its files, and a real MCP client to the listing's server |
| `tests/v8_fake.py` | a two-day fake EgoLife download with tiny clips |
| `tests/speech_check.py` | a second, plain reader of the transcripts of the speech scenarios, which derives each truth again |
| `tests/mm/fake_decider.py` | a decision model that answers from content the test knows (a word, a colour, a pitch), in the real model's answer shapes |
| `tests/mm/env_decider.py` | the fake decider, which also writes down the environment of the process it runs in |
| `tests/mm/fakes.py` | what the mm tests share: the fake model as a variant of the system, the tiny V7 week built and ingested once for a worker, a session over a copy of that store |
| `tests/mm/seam.py` | a stand-in agent process over the real mm session and its real server |
| `tests/mm/day.py`, `corpus.py`, `model_folders.py` | a made-up day of files, two PDF writers, and model folders without weights |

### What covers what

| Part | Tests |
| --- | --- |
| The run loop | `tests/test_runner.py`, `test_select.py` (which questions a configuration is asked), `test_config.py`, `test_ready.py` |
| How many execs run at once | `test_gpu_jobs.py`, `test_schedule.py`, `test_schedule_run.py`, `test_schedule_cli.py` |
| The working-time clock | `test_clock.py`, `test_clock_check.py` |
| The agents | `test_agents.py` (command lines, what a session may use, reading a stream), `test_codex.py` |
| Questions, answers and grading | `test_question.py`, `test_answer.py`, `test_answer_run.py`, `test_grade.py`, `test_set_rule.py`, `test_grounded.py`, `test_taxonomy.py` |
| Reports and statistics | `test_report.py`, `test_stats.py`, `test_summary.py`, `test_compare.py`, `test_repeat.py`, `test_costing.py` |
| The cache and the run folder | `test_cache.py`, `test_cache_cli.py`, `test_runfolder.py`, `test_fs.py` |
| The command line | `test_cli.py`, `test_cli_build.py`, `test_cli_v8.py`, `test_catalogue.py` |
| `mmsb fetch` | `test_fetch.py`, `test_fetch_piles.py`, against a local HTTP server; any other address fails the test |
| The shortcut probe | `test_probe.py`, `test_v7_probe.py` |
| The generated week (V7) | `test_v7_plan.py`, `test_v7_media.py`, `test_v7_frames.py`, `test_v7_questions.py`, `test_v7_scenario.py` |
| EgoLifeQA (V8) | `test_v8.py`, `test_v8_view.py`, `test_v8_run.py`, over the fake download |
| The real scenarios | `test_realdump.py`, `test_speech.py` and `test_speech_files.py` for what they share, and one file for each scenario: `test_v1.py` to `test_v4.py`, `test_v9.py`, `test_v10.py`, `test_i1.py` to `test_i4.py`, `test_i7.py` to `test_i9.py`, `test_a1.py`, `test_a3.py`, `test_a4.py`, `test_a6.py` to `test_a8.py`, `test_m3.py`, `test_m4.py` |
| The systems registry and `files` | `test_systems.py` |
| The `listing` system | `test_listing.py`, `test_listing_run.py`, `test_listing_times.py` |
| The `mm` system | `tests/mm/` |
| Neutrality | `test_neutral.py` |
| The suite's own setup | `test_conftest.py` |

- **A real scenario's tests** build its `tiny` dump from the source, derive every answer again with a plain reader of the
  labels that is the test's own, check that the grader takes each truth and refuses planted wrong answers, run the leak
  probe, and for most run `mmsb build`, `run` and `report` with a stand-in agent.
- **The mm tests** run the real system and its real daemon with the fake decider: what a session serves an agent, what keeps
  one exec's state out of the next, the cache, readiness, the dry run's preview, and that the small copies V7 holds of
  MultimediaSearch's reader and constants still agree with the installed package (`tests/mm/test_agreement.py`).
- **Neutrality** (`tests/test_neutral.py`): a process that refuses to import `mms` imports the whole core, every scenario and
  every system but `mm`; a scan of the syntax finds no import of `mms` outside `mmsb/systems/mm` and `tests/mm`; and a scan
  of `mmsb/core` and `mmsb/scenarios` finds no system named. The slow test makes a fresh environment with no extra, builds
  the tiny V7 week, runs a `files` cell with a stand-in agent, writes a report and sees `mm` refused.

## 4. What the tests do not cover

- **No test runs a real agent.** Every agent in the suite is a stand-in, so a change in the Claude Code or Codex command
  line shows only in a real run.
- **No test loads a real decision model.** The mm tests use the fake decider.
- **Real agent runs have been made only on Windows 11.** The tests pass on Linux and macOS, but Codex's sandbox and Claude's
  shell tool are untried there.
- **The markers `paid` and `gpu` have no tests yet.** They are there so that a test of either kind can be added without
  joining the default run.
- **No test builds a full-size dump.** The tests build the `tiny` scale and plan the larger ones from the labels.
- **Not run yet with real agents:** the working-time clock (`mmsb clock-check`) and more than one GPU exec at a time. Their
  tests use stand-ins.

To check the real agents by hand, run a small paid run. It spends a little of each agent's quota:

```
uv run mmsb fetch --scenario i3_trailcam
uv run mmsb build --scenario i3_trailcam --scale tiny --configs files,listing
uv run mmsb run <run> --agents claude,codex --configs files,listing --limit 2 --dry-run
uv run mmsb run <run> --agents claude,codex --configs files,listing --limit 2
uv run mmsb report <run>
```

## 5. Before a pull request

```
uv sync
uv run pytest
```

- All of the default run passes. If your change touches the `mm` system, also run `uv sync --extra mm` and
  `uv run pytest tests/mm --require-mm`.
- A change to a scenario is checked with its source on disk: read the skips in the summary, and fetch the source if the
  scenario's tests are among them.
- `mmsb/core/` imports and names no system: `tests/test_neutral.py` checks it.
- A new test file needs a name no other file in `tests/` or `tests/mm/` has, and writes only under its `tmp_path`.
- A test that needs a real agent or the GPU is marked `paid` or `gpu`, and one that takes more than a few seconds `slow`.

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest.
