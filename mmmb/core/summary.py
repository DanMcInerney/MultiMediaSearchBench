"""The summary report (DESIGN 6.3, 7.3, 9) of runs of any scenarios: for each scenario, what each agent got right in each configuration,
with its 95% interval, how its answers were grounded and what they cost; each other configuration set beside the control (`files`)
question by question; the questions a leak filter answered, which every other figure leaves out; and, for runs of several scenarios, the
same over all of them. It reads the runs' questions and results and nothing else, and names no system and no scenario. The scenarios
built from real recordings report with it (`Scenario.report`), and so does `mmmb report` of runs of several scenarios.

A result counts once it is done (`is_done`: an escaped, timed-out or capped exec is a wrong answer); one to run again (`setup`, `limit`,
`auth`, `error`) is counted under `to_rerun` and nowhere else. A question a leak filter answered right (`report.answerable`: the dump
leaks it to an agent with the listing alone) is left out of every configuration's figures but the filter's own."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from mmmb.core import fs
from mmmb.core.config import CONTROL, Config
from mmmb.core.probe import probe_of
from mmmb.core.record import kind_of
from mmmb.core.report import answerable, is_filter, paired, results_of, side
from mmmb.core.runfolder import RunFolder, is_done
from mmmb.core.stats import mean, wilson

ALPHA = 0.05  # a paired difference is told as shown only below this McNemar p (DESIGN 9: smaller ones are called unproven)
FEW = 30  # questions under which a configuration's interval is noted as wide


def configs_of(runs: Sequence[RunFolder], control: str) -> list[Config]:
    """The configurations of `runs`, the control first, then by name."""
    found = {c.name: c for run in runs for c in run.configs}
    return sorted(found.values(), key=lambda c: (c.name != control, c.name))


def filters_of(runs: Sequence[RunFolder]) -> set[str]:
    """The names of the configurations that are leak filters in any of `runs` (by their systems.json)."""
    out = set()
    for run in runs:
        noted = fs.read_json(run.folder / "systems.json")  # (a run that has been run has one: `mmmb report` refuses one that has not)
        out |= {c.name for c in run.configs if is_filter(c, noted.get(c.name, {}))}
    return out


def cell(rows: list[tuple]) -> dict:
    """The figures of one agent's done results in one configuration, `rows` of `(question, result)`: `report.side`'s, with the 95%
    Wilson interval of the share right, the questions by kind and by media, and the mean item F1 of the `set` questions."""
    rs = [r for _, r in rows]
    out = side(rs)
    out["interval"] = list(wilson(out["correct"], out["n"])) if rs else None
    right = out["correct"]
    out["usd_per_right"] = out["cost_usd"] / right if out["cost_usd"] is not None and right else None
    out["s_per_right"] = out["seconds"] / right if right else None
    out["tokens_per_right"] = out["tokens_all"] / right if out["tokens_all"] is not None and right else None
    out["by_kind"] = tally((q.kind, r) for q, r in rows)
    out["by_media"] = tally((str(q.tags["modality"]), r) for q, r in rows)
    out["items_f1"] = mean([r["items_f1"] for r in rs if r.get("items_f1") is not None])
    return out


def tally(pairs) -> dict[str, list[int]]:
    """`[right, n]` for each key of `(key, result)` pairs, by key."""
    out: dict[str, list[int]] = {}
    for key, r in pairs:
        right, n = out.get(key, [0, 0])
        out[key] = [right + bool(r["passed"]), n + 1]
    return dict(sorted(out.items()))


def section(runs: Sequence[RunFolder], control: str) -> dict:
    """The figures of the runs of one scenario (`numbers` checks that their questions do not repeat)."""
    loaded = [(run.questions(), results_of(run)) for run in runs]
    configs = configs_of(runs, control)
    filters = filters_of(runs)
    leaks = answerable([c for c in configs if c.name in filters], loaded)
    agents = sorted({a for _, results in loaded for a in results})
    cells: dict[str, dict[str, dict]] = {}
    pairs: dict[str, dict[str, dict]] = {}
    questions: list[dict] = []
    rerun = Counter()
    for agent in agents:
        done: dict[str, dict[tuple, tuple]] = {}  # configuration, then (run, qid), then (question, result)
        for c in configs:
            for run, (qs, results), leaked in zip(runs, loaded, leaks, strict=True):
                got = results.get(agent, {}).get(c.dir, {})
                for q in qs:
                    r = got.get(q.qid)
                    if r is None:
                        continue
                    if not is_done(r):
                        rerun[f"{agent} {c.name}"] += 1
                    elif c.name in filters or q.qid not in leaked:
                        done.setdefault(c.name, {})[run.name, q.qid] = (q, r)
            if c.name in done:
                cells.setdefault(agent, {})[c.name] = cell(list(done[c.name].values()))
        base = done.get(control, {})
        for c in configs:
            if c.name == control or c.name in filters or c.name not in done:
                continue
            rows = [(q, base[key][1], r) for key, (q, r) in done[c.name].items() if key in base]
            if rows:
                pairs.setdefault(agent, {})[c.name] = {k: v for k, v in paired(rows).items() if k not in ("control", "tested")}
    for run, (qs, results), leaked in zip(runs, loaded, leaks, strict=True):
        for q in qs:
            verdicts = {f"{agent} {c.name}": verdict(results.get(agent, {}).get(c.dir, {}).get(q.qid)) for agent in agents for c in configs}
            questions.append({"run": run.name, "qid": q.qid, "kind": q.kind, "media": q.tags["modality"], "split": q.split,
                              "leaked": leaked.get(q.qid, []), "verdicts": {k: v for k, v in verdicts.items() if v is not None}})
    return {"scenario": runs[0].scenario.name, "code": runs[0].scenario.code, "questions": sum(len(qs) for qs, _ in loaded),
            "configs": [c.name for c in configs], "filters": sorted(filters), "cells": cells, "pairs": pairs,
            "leaks": [{"run": run.name, "qid": qid, "agents": who} for run, leaked in zip(runs, leaks) for qid, who in leaked.items()],
            "to_rerun": dict(rerun), "rows": questions}


def verdict(r: dict | None) -> str | None:
    """What the per-question table shows of a result: `right`, `wrong`, the kind of problem that made it wrong, `rerun`; None for none."""
    if r is None:
        return None
    if not is_done(r):
        return "rerun"
    return "right" if r["passed"] else kind_of(r["problem"]) or "wrong"


def numbers(runs: Sequence[RunFolder], control: str = CONTROL) -> dict:
    """The summary's figures (report.json) over `runs`, scenario by scenario in the order first met, and `overall`, each agent and
    configuration over every scenario, when there are several. Runs of one scenario with the same seed and scale ask the same questions
    and are refused: pooled, each question would count twice."""
    if not any(control in {c.name for c in run.configs} for run in runs):
        raise ValueError(f"no run has the configuration {control!r} to set the others beside: name one they have (--control)")
    seen: dict[tuple, str] = {}
    for run in runs:
        key = (run.scenario.name, run.seed, run.scale)
        if key in seen:
            raise ValueError(f"{run.name} and {seen[key]} are the same scenario, seed and scale, so the same questions: pooling them would "
                             "count each question twice")
        seen[key] = run.name
    names = list(dict.fromkeys(run.scenario.name for run in runs))
    sections = [section([r for r in runs if r.scenario.name == name], control) for name in names]
    overall: dict[str, dict[str, list[int]]] = {}
    for s in sections:
        for agent, by_config in s["cells"].items():
            for name, c in by_config.items():
                right, n = overall.setdefault(agent, {}).get(name, [0, 0])
                overall[agent][name] = [right + c["correct"], n + c["n"]]
    return {"runs": [{"name": run.name, "scenario": run.scenario.name, "scale": run.scale, "seed": run.seed,
                      "commit": fs.read_json(run.folder / "run.json").get("mmmb_commit"), "probe": probe_of(run.folder)} for run in runs],
            "control": control, "scenarios": sections, **({"overall": overall} if len(sections) > 1 else {})}


# -- the Markdown ---------------------------------------------------------------------------------------------------------------------

def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)]


def accuracy_rows(s: dict) -> list[list]:
    rows = []
    for agent, by_config in s["cells"].items():
        for name in s["configs"]:
            c = by_config.get(name)
            if c is None:
                continue
            g = c.get("grounding")
            low, high = c["interval"]
            rows.append([agent, name + (" (leak check)" if name in s["filters"] else ""), pct(c["correct"], c["n"]),
                         f"{100 * low:.0f}-{100 * high:.0f}%", "-" if g is None else g["grounded"], "-" if g is None else g["lucky"],
                         "-" if g is None else g["unparsed"], f"{c['absent_answered']}/{c['absent']}", f"{c['abstained']}/{c['needles']}",
                         num(c["s_per_right"], 0), num(None if c["tokens_per_right"] is None else c["tokens_per_right"] / 1000, 0),
                         num(c["cost_usd"], 2), num(c["usd_per_right"], 2), num(c["items_f1"], 2), c["timeouts"]])
    return rows


def pair_rows(s: dict, control: str) -> list[list]:
    rows = []
    for agent, by_config in s["pairs"].items():
        for name, p in by_config.items():
            diff = "-" if p["diff"] is None else f"{100 * p['diff']:+.0f} pts"
            ci = "" if p["ci"] is None else f" [{100 * p['ci'][0]:+.0f}, {100 * p['ci'][1]:+.0f}]"
            shown = p["p"] is not None and p["p"] < ALPHA
            said = ("better" if p["diff"] > 0 else "worse") if shown else "not shown (too few questions, or no difference)"
            rows.append([agent, f"{name} vs {control}", p["n"], diff + ci, f"{p['b']} / {p['c']}", num(p["p"], 3), said])
    return rows


def section_lines(s: dict, control: str) -> list[str]:
    out = ["", f"## {s['code']} ({s['scenario']}): {s['questions']} questions", "", "### What each agent got right", ""]
    out += table(["agent", "configuration", "right", "95% interval", "grounded", "lucky", "unparsed", "answered an absent one",
                  "unknown on a real one", "s per right", "tokens k per right", "cost $", "$ per right", "find-all F1", "timed out"],
                 accuracy_rows(s))
    if few := [f"{a} {n}" for a, by in s["cells"].items() for n, c in by.items() if c["n"] < FEW]:
        out += ["", f"Under {FEW} questions the interval is wide: {', '.join(few)}."]
    if rows := pair_rows(s, control):
        out += ["", f"### Each configuration beside {control}, question by question", ""]
        out += table(["agent", "pair", "paired", "difference (95% CI)", "b / c", "McNemar p", "verdict"], rows)
        out += ["", (f"b is the questions right only in the configuration, c those right only in {control}. A difference is shown "
                     f"only when McNemar's p is under {ALPHA}.")]
    kinds = sorted({k for by in s["cells"].values() for c in by.values() for k in c["by_kind"]})
    out += ["", "### By kind of question", ""]
    out += table(["agent", "configuration", *kinds],
                 [[agent, name, *(pct(*c["by_kind"].get(k, [0, 0])) for k in kinds)] for agent, by in s["cells"].items()
                  for name, c in by.items()])
    if s["filters"]:
        out += ["", "### Answerable from metadata", ""]
        out += ([f"- {x['qid']} ({x['run']}): {', '.join(x['agents'])}" for x in s["leaks"]] or
                [f"None: {', '.join(s['filters'])} answered no question that has something to find."])
        if s["leaks"]:
            out += ["", "These are left out of every other figure."]
    if s["to_rerun"]:
        out += ["", "To run again (left out): " + ", ".join(f"{k} {v}" for k, v in sorted(s["to_rerun"].items())) + "."]
    columns = sorted({k for row in s["rows"] for k in row["verdicts"]})
    out += ["", "### Every question", ""]
    out += table(["qid", "kind", "media", *columns],
                 [[row["qid"] + (" (leaked)" if row["leaked"] else ""), row["kind"], row["media"],
                   *(row["verdicts"].get(k, "") for k in columns)] for row in s["rows"]])
    return out


def markdown(figures: dict) -> str:
    """report.md of `numbers`' figures."""
    out = ["# MMMB report", ""]
    out += [f"- {r['name']}: {r['scenario']}, scale {r['scale']}, seed {r['seed']}, commit {r['commit']}, shortcut probe "
            f"{(r['probe'] or {}).get('status', 'not recorded')}" for r in figures["runs"]]
    out += ["", (f"A right answer passed its rule. Grounded: it also cited where in the media the answer is; lucky: it did not. The "
                 f"control is {figures['control']}: the agent alone with the files and its own tools.")]
    for s in figures["scenarios"]:
        out += section_lines(s, figures["control"])
    if "overall" in figures:
        out += ["", "## Over all the scenarios", ""]
        rows, filters = [], {f for s in figures["scenarios"] for f in s["filters"]}
        for agent, by in figures["overall"].items():
            for name, (k, n) in by.items():
                low, high = wilson(k, n)
                rows.append([agent, name + (" (leak check)" if name in filters else ""), pct(k, n), f"{100 * low:.0f}-{100 * high:.0f}%"])
        out += table(["agent", "configuration", "right", "95% interval"], rows)
    return "\n".join(out) + "\n"


def report(runs: Sequence[RunFolder], control: str = CONTROL) -> tuple[dict, str]:
    """`Scenario.report` for a scenario that reports with the summary: its figures and Markdown."""
    figures = numbers(runs, control)
    return figures, markdown(figures)
