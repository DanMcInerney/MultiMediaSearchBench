# The four real piles, with and without a search system (6 and 7 November 2026)

The first run of a search system beside the agent's own tools on real media. Each agent was asked 5 `test` questions of each
pile (`--splits test --limit 5`, seed 1, full size), with 900 s for each, in three configurations:

- `files`: the agent alone with the files and its own tools. These cells are the [baseline](../2026-11-06-baseline/) published
  on 6 November.
- `listing`: the files' names, sizes and times only. The floor, and the leak check.
- `mm/d1-Omni-2B+files`: the agent's own tools and [MultimediaSearch](https://github.com/DanMcInerney/MultimediaSearch)'s, with
  its `d1-Omni-2B` model, which sees and hears.

Claude Code ran Haiku at `xhigh` effort and Codex ran `gpt-6-luna` at `xhigh`, at commit e94911b.

| Pile | Claude, `files` | Claude, `mm+files` | Codex, `files` | Codex, `mm+files` | Report |
| --- | --- | --- | --- | --- | --- |
| `i3_trailcam` | 4/5 | 3/5 | 3/5 (1 lucky) | 4/5 (1 lucky) | [i3.md](i3.md) |
| `a1_home_audio` | 3/5 | 3/5 | 1/5 | 1/5 (lucky) | [a1.md](a1.md) |
| `v1_security` | 2/5 | 1/5 | 1/5 | 2/5 | [v1.md](v1.md) |
| `m4_samples` | 1/5 | 0/5 | 0/5 | 0/5 | [m4.md](m4.md) |
| **All four** | **10/20** | **7/20** | **5/20** | **7/20** | |

## What it says

- **No difference is shown.** In every pile the two configurations differ by at most one question, and every McNemar p is 1.0.
  Over the 20 questions Claude lost 3 with the search system and Codex gained 2: with 5 questions a pile, neither is evidence.
- **Time ran out.** With `mm+files`, 7 of each agent's 20 execs timed out, all 5 of `m4_samples` among them for both. With
  `files`, Codex timed out on 7 and Claude on 1 (and hit its 60-turn cap twice).
- **`listing` answered nothing it should not.** Both agents answered "unknown" to all 20, which is right only on the 4 absent
  questions.
- **Cost.** Claude's `files` execs came to $8.07 at list price and its `mm+files` execs to $2.93. The second is a lower bound:
  an exec that times out records no tokens. Codex reports no cost.

## The wait before a pile can be searched

| Pile | Ingest into MultimediaSearch | Store written |
| --- | --- | --- |
| `i3_trailcam` (4,256 photos) | 133 s | 1.9 GB |
| `a1_home_audio` (24 h, one file) | 20 s | 2.8 GB |
| `v1_security` (3.5 h of video) | 284 s | 5.8 GB |
| `m4_samples` (6,123 clips) | 276 s | 5.3 GB |

Ingest used no GPU: the model reads the media when it is asked, not before.

## Read it with these

- **5 questions a cell.** A cell's 95% interval spans about 60 points.
- **The agents were not told the 900 s.** The prompt has said the limit since 8 November, so a later run is not like for like.
- **Claude's `files` cell of `m4_samples` is a rerun**, made after its prompt was fixed to name two limits of its shell tool
  (a command over 10,000 characters, and a redirect), which it had taken for ffmpeg being off.
- **The GPU dropped off once** during the `mm+files` execs (6 November, on Codex's third `i3_trailcam` exec). The runner
  halted, and after a reboot the same command resumed the run and asked that exec again.
- **One search system, by this benchmark's author.** A second system is the check that the comparison is fair.
- The reports were written when the benchmark was called MultiMediaMemoryBench, and are headed "MMMB report".
