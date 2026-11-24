# Contributing

Thank you for looking. Issues and pull requests are welcome, most of all:

- **a second search system** (README, "Measuring another system"): the best check that the benchmark favours none;
- **results**: a run of any pile with any agent, with its `report.md` and `report.json`;
- **a new pile** of media people really keep, from a public dataset whose licence allows it ([MEDIA.md](MEDIA.md) shows what
  is recorded for each source);
- **Linux and macOS runs**: the tests pass there, but real agent runs have only been made on Windows 11.

## Before a pull request

```
uv sync
uv run pytest
```

The fast tests need no agent, no GPU and no network, and take about 6 minutes. CI runs them on Windows, Linux and macOS.

- [VISION.md](VISION.md) says what the benchmark must stay true to, and [DESIGN.md](DESIGN.md) how it is built. A change that
  alters what is measured belongs in DESIGN.md too.
- `mmsb/core/` imports and names no system: `tests/test_neutral.py` checks it.
- Answers are graded by fixed rules. A change must not add a model as a judge.
- Nothing is tuned on `test` questions: use the `dev` split.
- The files under `tests/oracles/recorded/` and `tests/mm/tables/` are compared byte for byte. Do not reformat them.
- Never commit media, and never put a dataset where its licence forbids it (SINS may not be passed on).

## Reporting a result

Say the commit, the scenario, scale and seed, each agent's model and effort, and the configurations. Include the absent
questions and the `listing` cells: a result without its floor cannot be read.
