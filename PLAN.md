# MMMB: the plan to production readiness

Agreed with Dan on 5 November 2026. Not started. The state it starts from is in [DESIGN.md section 12](DESIGN.md#12-phases)
and [TESTING.md section 8](TESTING.md#8-not-verified).

## What "production-ready" means here

The benchmark as it is (scenarios V7 and V8; systems `files`, `listing` and `mm`; agents Claude Code and Codex):
- installs and runs with the real agents, and its report is right;
- has no known open defect;
- has documents that say what is verified and what still needs the GPU or a paid full run, with the commands.

New systems (`embed`, `transcript`) and new scenarios are growth, not readiness, and are not part of this plan.

## Limits

- Do not hog the GPU: no GPU job, no mm model, no `--group mm-gpu` install.
- No full benchmark runs and no long tests. End-to-end tests are the bare minimum, as fast as possible, and only when needed.
- Keep it simple: it is a multimedia memory benchmark.

## Facts found while planning (5 November 2026)

- The fast suite passed with the `mm` extra in US Eastern time: 1,072 passed, 1 skipped, in 7:40 at 4 workers.
- Codex 0.159.3 lists `gpt-6.1-terra` again, so TESTING.md's note that `run --dry-run --agents codex` fails here is out of date.
- V7 asks 31 questions at both the `pilot` (3-day) and `standard` (7-day) scales. Reaching the 70+ questions DESIGN section 9
  needs is a matter of pooling seeds (3 seeds give about 93), which `mmmb report` already does. It needs no code.

## Steps

1. **A small real-agent test, with no GPU.** It spends a little of the plans' quota, which Dan approved with this plan.
   - Build V7 tiny with `--configs files,listing` (about 35 s of CPU), in folders of its own on D:.
   - `mmmb run` with `--agents claude,codex --configs files,listing --limit 2`: 8 execs, about 10 minutes, likely under $1 of
     Claude and a little Codex quota. Then `mmmb report`.
   - One `claude -p` call that checks `--restricted` accepts a second `--add-dir`: the `mm+files` question, without mm.
   - It shows whether real agents accept the `json` answer request, the listing server and today's CLI flags, and whether the
     report holds on fresh results.
2. **Fix what step 1 breaks.** Reproduce each fault first; check with the affected test files only.
3. **Update the documents** (README status, TESTING.md, DESIGN section 12.1):
   - what is now verified;
   - step 1 as a repeatable recipe, to run after each CLI update;
   - V7 test seeds (11 and up) and pooling seeds for enough questions;
   - what still needs the GPU or a paid full run, with the commands.
4. **One independent review** by a fresh agent that made none of it, then at most one repair pass. Then the fast suite once (about
   8 minutes, 4 CPU workers, no GPU).
5. **Commit to this branch.** No push without Dan's go-ahead.

The work is done directly and in order: the units are small, and children running tests side by side would break the laptop's
rule of at most 4 CPU processes. The only child is the reviewer of step 4.

## Not in this plan

| Left out | Why |
| --- | --- |
| The pilot re-run, the first `mm+files` run, the clock check | they need the GPU or a full paid run; their commands are in DESIGN section 12.2 |
| `embed`, `transcript`, the P1 scenarios | growth, not readiness |
| V8's four rotations of multiple-choice options | V8 cannot run without its 41 GB view and a paid run; do it when V8 is next asked (DESIGN 7.3) |
| The Pareto plot, the Lifelog Search Challenge score, Holm's correction, bootstrap rankings, session mode, `gold`, the size ladder, the guard for two cells sharing a home, configurable data folders | nothing uses them yet, and a change to the report breaks its byte-for-byte oracle against the recorded pilot |
| Publishing, a licence and CI; mm's instructions over the 2,000-token rule; Multimem's missing remote | Dan's decisions (DESIGN section 13) |
