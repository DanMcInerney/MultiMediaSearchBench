"""The report's figures (moved from `numbers` and what it calls in bench.py of this repository at bb498a6): the paired comparison of
each tested configuration with the control, by agent, over the questions both have a done result for; by how vague the time hint is;
how each side worked and what a correct answer cost it; one row per question; and the retrieval tier (core/retrieval.py). `render.markdown`
turns the figures into report.md. Nothing here reads a result written by anything but `core/record.py`.

A paired row is `(question, control result, tested result)`. Every figure that compares two sides has them under `control` and
`tested`: a pooled level has several configurations on its tested side. A tested configuration is a run's configuration with a system
that is not the control (`blind` has none); its figures are named by `Config.name`, and `figures["frame"]` has the words the
Markdown uses for the two sides (`render`).

`systems.json` in a run folder says, by `Config.name`, which modalities (`asked`) a tested configuration was asked questions about: the
runner writes it when it selects the cases (`note_configs`).

A configuration whose system is a leak filter (`Description.leak_filter`, kept in systems.json, DESIGN 7.3) is a floor, not a tested
configuration: it is not compared with the control. A question with something to find that it answers right is answerable from
metadata, whichever agent it was asked of (the dump leaks it); the paired counts of every tested configuration leave such a question out,
and `answerable_from_metadata` names it and the agents that answered it. An absent question is exempt (C9): an agent with nothing to
read answers "unknown" to it, so every one would be flagged."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from mmmb.core import fs
from mmmb.core.config import Config
from mmmb.core.contract import BEYOND_TIMEOUT, EXCLUSIVE, GPU_ESTIMATE_S
from mmmb.core.grounded import NOT_RECORDED, status_of
from mmmb.core.probe import probe_of
from mmmb.core.record import kind_of
from mmmb.core.retrieval import FOLDER as RETRIEVAL_FOLDER
from mmmb.core.retrieval import LEVELS, TYPES, decider_figures  # the tier below owns them; core/select.py reads LEVELS here
from mmmb.core.runfolder import RunFolder, is_done
from mmmb.core.select import MEDIA
from mmmb.core.stats import interval, mcnemar, mean, median, pair_counts, ratio, token_total

CROSS_LEVELS = LEVELS[:3]  # the levels the crossover looks at, from the least vague
LEAD, HALF = 0.20, 0.5  # the crossover level: the tested side leads by this share of questions, or costs this share of the control's cost, or less
MIN_LEVEL_N = 10  # paired questions a level needs before its crossover is told
MIN_CORRECT = 5  # correct answers a side needs before its cost per correct answer is told
NOT_WORSE = 0.10  # the cost branch of the crossover lets the tested side be this much less accurate, when the interval's low end is above it


def grounding_of(rows: list[dict]) -> dict | None:
    """How the answers of a side were grounded (DESIGN 6.3), beside its verdict: the results by status, `unparsed` (an exec that ended
    with no problem and whose final message could not be read), and `accuracy`, the share that are `grounded` of those that recorded
    a status. None when no result of the side holds one (all `line`, or recorded before the status): then nothing of it is told."""
    if not any("grounded" in r for r in rows):
        return None
    counts = Counter(status_of(r) for r in rows)
    judged = len(rows) - counts[NOT_RECORDED]
    return {**{status: counts[status] for status in ("grounded", "lucky", "wrong", NOT_RECORDED)},
            "unparsed": sum(1 for r in rows if r.get("answer_error") and r["problem"] is None),
            "accuracy": counts["grounded"] / judged if judged else None}


def side(rows: list[dict]) -> dict:
    """The figures of one side's results over the paired questions; `grounding` too when its results hold a grounded status."""
    absent = [r for r in rows if r["absent"]]
    needles = [r for r in rows if not r["absent"]]
    tokens = [r["tokens"] for r in rows if r.get("tokens")]
    totals = [t for r in rows if (t := token_total(r)) is not None]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    grounding = grounding_of(rows)
    return {"correct": sum(r["passed"] for r in rows), "accuracy": sum(r["passed"] for r in rows) / len(rows) if rows else None,
            "n": len(rows), "absent": len(absent),
            "absent_answered": sum(1 for r in absent if r["answer"] is not None and not r["abstained"]),
            "needles": len(needles), "abstained": sum(1 for r in needles if r["abstained"]),
            "median_s": median([r["seconds"] for r in rows]), "median_calls": median([r["calls"] for r in rows]),
            "seconds": sum(r["seconds"] for r in rows), "timeouts": sum(1 for r in rows if kind_of(r["problem"]) == "timeout"),
            "tokens": {k: sum(t[k] for t in tokens) for k in ("input", "output", "cache_read", "cache_write")} if tokens else None,
            "tokens_mean": mean(totals), "tokens_all": sum(totals) if totals else None, "tokens_n": len(totals),
            "cost_usd": sum(costs) if costs else None, **({"grounding": grounding} if grounding else {})}


def results_of(run: RunFolder) -> dict[str, dict[str, dict[str, dict]]]:
    """Every result record of the run: agent, then configuration folder (`Config.dir`), then qid. A result is a file named after a
    question of the run: the other lists in the folder (a session's named lists, `Session.close`) are not read."""
    out: dict = {}
    root = run.folder / "results"
    qids = {q.qid for q in run.questions()}
    for agent in sorted(p for p in root.glob("*") if p.is_dir() and p.name != RETRIEVAL_FOLDER):
        out[agent.name] = {arm.name: {f.stem: fs.read_json(f) for f in sorted(arm.glob("*.json")) if f.stem in qids}
                           for arm in sorted(agent.iterdir()) if arm.is_dir()}
    return out


def paired(rows: list[tuple]) -> dict:
    """The comparison over paired questions `(question, control result, tested result)`: both sides' figures, the difference (the
    mean of the paired differences) with Newcombe's interval when `interval` gives one, and b (right only with the tested side), c
    (right only with the control) with McNemar's exact p. The p is what a verdict rests on; b and c are always told."""
    a, b, c, d = pair_counts(rows)
    control, tested = side([f for _, f, _ in rows]), side([m for _, _, m in rows])
    return {"n": len(rows), "control": control, "tested": tested, "diff": (b - c) / len(rows) if rows else None,
            "ci": interval(a, b, c, d) if rows else None, "b": b, "c": c, "p": mcnemar(b, c) if rows else None}


def lean(rows: list[tuple]) -> dict:
    """`paired` for a level: the same difference, interval and counts, with each side's figures. A control result is counted once
    however many tested results it is paired with (pooled over configurations, it is paired with one for each), so `n_control` can
    be below `n`; that is for display, the difference is the mean over the pairs."""
    control = list({id(f): f for _, f, _ in rows}.values())
    a, b, c, d = pair_counts(rows)
    return {"n": len(rows), "n_control": len(control), "control": side(control), "tested": side([m for _, _, m in rows]),
            "diff": (b - c) / len(rows) if rows else None, "ci": interval(a, b, c, d) if rows else None, "b": b, "c": c,
            "p": mcnemar(b, c) if rows else None}


def by_level(rows: list[tuple]) -> dict[str, dict]:
    """`lean` of the paired rows at each level of the question's hint that has any, from the least vague."""
    return {level: lean(of) for level in LEVELS if (of := [r for r in rows if r[0].hint == level])}


def cost_ratios(c: dict) -> tuple[float | None, float | None]:
    """The tested side's seconds and tokens per correct answer as a share of the control's, for a level of `by_level`; None where a
    side has fewer than MIN_CORRECT correct answers (a cost per correct answer of two or three is no figure) or no tokens."""
    f, m = c["control"], c["tested"]
    if min(f["correct"], m["correct"]) < MIN_CORRECT:
        return None, None
    seconds = ratio(m["seconds"] / m["correct"], f["seconds"] / f["correct"])
    tokens = (ratio(m["tokens_all"] / m["correct"], f["tokens_all"] / f["correct"])
              if f["tokens_all"] and m["tokens_all"] and f["tokens_n"] == f["n"] and m["tokens_n"] == m["n"] else None)
    return seconds, tokens


def crossover(levels: dict[str, dict], min_n: int | None = None, *,
              order: Sequence[str] = CROSS_LEVELS) -> dict | None:
    """The first of part, day and week (or of `order`) that has `min_n` paired questions (MIN_LEVEL_N) and where the tested side leads
    by LEAD, or costs HALF as much per correct answer or less (seconds, or tokens) without being less accurate (a difference of 0 or
    more, or an interval whose low end is above -NOT_WORSE). Its `by` says which of "accuracy" and "cost" held, with the tested
    side's lead and its cost per correct answer as a share of the control's; None when no level qualifies."""
    min_n = MIN_LEVEL_N if min_n is None else min_n
    for level in order:
        c = levels.get(level)
        if not c or c["n"] < max(1, min_n):
            continue
        lead, by = c["diff"], []
        seconds, tokens = cost_ratios(c)
        if lead is not None and lead >= LEAD - 1e-9:
            by.append("accuracy")
        not_worse = lead is not None and (lead >= 0 or (c.get("ci") is not None and c["ci"][0] > -NOT_WORSE))
        if not_worse and any(x is not None and x <= HALF for x in (seconds, tokens)):
            by.append("cost")
        if by:
            return {"level": level, "by": by, "lead": lead, "seconds": seconds, "tokens": tokens}
    return None


def thin_levels(levels: dict[str, dict], min_n: int | None = None, *,
                order: Sequence[str] = CROSS_LEVELS) -> dict[str, int]:
    """The levels of part, day and week (or of `order`) that have questions but fewer than `min_n` pairs (MIN_LEVEL_N), with how many."""
    min_n = MIN_LEVEL_N if min_n is None else min_n
    return {level: levels[level]["n"] for level in order
            if level in levels and 0 < levels[level]["n"] < max(1, min_n)}


def worked(rs: list[dict]) -> dict:
    """How a side worked over its results: the share that called `ask`, asks, finds and looks per exec (read by tool name from the
    tool counts every result has), and from `work` (`known` of the results have it) the windows and GPU seconds of the ask jobs per
    exec, the jobs stopped at `max_s`, and the pictures viewed and shell commands per exec."""
    tools = [r.get("tools") or {} for r in rs]
    got = [r["work"] for r in rs if r.get("work")]
    return {"n": len(rs), "ask_execs": sum(1 for t in tools if t.get("ask")),
            **{k: mean([t.get(name, 0) for t in tools]) for k, name in (("asks", "ask"), ("finds", "find"), ("looks", "look"))},
            "known": len(got), "windows": mean([w["windows"] for w in got]), "gpu_s": mean([w["seconds"] for w in got]),
            "estimate_s": mean([r[GPU_ESTIMATE_S] for r in rs if r.get(GPU_ESTIMATE_S) is not None]),
            "held_s": mean([r["held_s"] for r in rs if r.get("held_s") is not None]),
            "alone": sum(1 for r in rs if r.get(EXCLUSIVE)), "beyond": sum(1 for r in rs if r.get(BEYOND_TIMEOUT)),
            "max_s": sum(w["stopped"].get("max_s", 0) for w in got), "views": mean([w["views"] for w in got]),
            "shells": mean([w["shells"] for w in got])}


def per_correct(rs: list[dict]) -> dict:
    """What a correct answer cost a side: its results' seconds, tokens (k) and GPU seconds, each divided by the correct ones
    (`correct` is k: the report tells a cost only from MIN_CORRECT). The tokens are told only when every result gives them and the
    GPU seconds only when every result has `work` (else they would be too few)."""
    k = sum(r["passed"] for r in rs)
    totals = [t for r in rs if (t := token_total(r)) is not None]
    got = [r["work"] for r in rs if r.get("work")]
    return {"correct": k, "n": len(rs), "s": sum(r["seconds"] for r in rs) / k if k else None,
            "tokens_k": sum(totals) / 1000 / k if k and totals and len(totals) == len(rs) else None,
            "gpu_s": sum(w["seconds"] for w in got) / k if k and got and len(got) == len(rs) else None}


def compare(agent: str, tested: Config, types_asked: frozenset[str], control: Config, loaded: list[tuple],
            leaks: Sequence[Mapping[str, Sequence[str]]]) -> dict | None:
    """One tested configuration's figures for one agent, pooled over the runs: of the questions it is asked (those about
    `types_asked`, less those answerable from metadata: `leaks`, `answerable`'s, one for each run of `loaded`, and counted
    nowhere here), those with a done result with both configurations are paired (an escaped, timed-out or capped one is a wrong
    answer); one with a result to run again (`setup`, `limit`, `auth`, `error`) is left out, and one with none is pending. `all` is
    every paired question, `media` the pictures, video and sound, the part that depends on the system; `levels` and `crossover` are
    over the media by how vague the hint is, `work` and `cost` how each side worked and what a correct answer cost it on them.
    `loaded` has each run's questions and results (`results_of`). `rows` are the paired media questions, for `numbers` to pool. None
    when the agent has no result with the tested configuration."""
    rows, asked, left_out, seen = [], 0, 0, False
    kinds: dict[str, Counter] = {"control": Counter(), "tested": Counter()}
    for (questions, results), leaked in zip(loaded, leaks, strict=True):
        arms = {"control": results.get(agent, {}).get(control.dir, {}), "tested": results.get(agent, {}).get(tested.dir, {})}
        seen = seen or bool(arms["tested"])
        for q in questions:
            if q.tags["modality"] not in types_asked or q.qid in leaked:
                continue
            asked += 1
            f, m = arms["control"].get(q.qid), arms["tested"].get(q.qid)
            for arm, r in (("control", f), ("tested", m)):
                if r is not None and r["problem"]:
                    kinds[arm][kind_of(r["problem"])] += 1
            if is_done(f) and is_done(m):
                rows.append((q, f, m))
            elif (f is not None and not is_done(f)) or (m is not None and not is_done(m)):
                left_out += 1
    if not seen:
        return None
    types = {}
    for type_ in TYPES:
        of = [(f, m) for q, f, m in rows if q.tags["modality"] == type_]
        if of:
            types[type_] = {"n": len(of), "control": sum(f["passed"] for f, _ in of), "tested": sum(m["passed"] for _, m in of)}
    media = [r for r in rows if r[0].tags["modality"] in MEDIA]
    levels = by_level(media)
    return {"questions": asked, "left_out": left_out, "pending": asked - len(rows) - left_out,
            "kinds": {arm: dict(counts) for arm, counts in kinds.items()}, "all": paired(rows), "media": paired(media),
            "levels": levels, "crossover": crossover(levels), "thin": thin_levels(levels),
            "work": {"control": worked([f for _, f, _ in media]), "tested": worked([m for _, _, m in media])},
            "cost": {"control": per_correct([f for _, f, _ in media]), "tested": per_correct([m for _, _, m in media])},
            "scope_all": sum(m["scope_all"] for _, _, m in rows), "types": types, "rows": media}


def cell_of(r: dict | None) -> dict | None:
    """What the appendix keeps of one exec's result: whether it passed, its seconds and problem kind, the pictures it viewed
    (None when the result does not say) and the asks it made."""
    if r is None:
        return None
    return {"passed": bool(r["passed"]), "seconds": r["seconds"], "problem": kind_of(r["problem"]), "beyond": bool(r.get(BEYOND_TIMEOUT)),
            "views": (r.get("work") or {}).get("views"), "asks": (r.get("tools") or {}).get("ask", 0)}


def appendix_rows(runs: Sequence[RunFolder], loaded: list[tuple], tested: Sequence[Config], control: Config) -> dict[str, list[dict]]:
    """Per agent, a row for each question some side ran: qid, level, type, the control's cell and each tested configuration's."""
    out: dict[str, list[dict]] = {}
    for run, (questions, results) in zip(runs, loaded):
        for agent, arms in results.items():
            for q in questions:
                control_cell = arms.get(control.dir, {}).get(q.qid)
                cells = {c.name: cell_of(arms[c.dir].get(q.qid)) for c in tested if c.dir in arms and q.qid in arms[c.dir]}
                if control_cell is not None or cells:
                    out.setdefault(agent, []).append({"seed": run.seed, "qid": q.qid, "level": q.hint, "type": q.tags["modality"],
                                                      "control": cell_of(control_cell), "tested": cells})
    return out


def agent_models(loaded: list[tuple]) -> dict[str, dict[str, int]]:
    """Per agent, how many of its results each of the agent's models gave, with the effort when the agent records one (Codex), so
    results of two settings pooled by mistake show."""
    out: dict[str, Counter] = {}
    for _, results in loaded:
        for agent, arms in results.items():
            for rs in arms.values():
                for r in rs.values():
                    name = str(r.get("agent_model") or "unknown") + (f" at {r['agent_effort']} effort" if r.get("agent_effort") else "")
                    out.setdefault(agent, Counter())[name] += 1
    return {agent: dict(sorted(c.items())) for agent, c in sorted(out.items())}


def is_filter(config: Config, noted: dict) -> bool:
    """Whether `config` is a leak filter: no tools of its own, and its systems.json entry `noted` says its system is one. An entry
    of a recorded run has no description."""
    return not config.own_tools and bool((noted.get("description") or {}).get("leak_filter"))


def sides_of(runs: Sequence[RunFolder], control: str) -> tuple[Config, dict[Config, frozenset[str]], set[Config]]:
    """The control configuration named `control`, the tested ones over all `runs` (those with a system that is not the control:
    `blind` has none) with the kinds of data each is asked about, from the run's systems.json, and the leak filters among the others."""
    controls = {c for run in runs for c in run.configs if c.name == control}
    if len(controls) != 1 or any(control not in {c.name for c in run.configs} for run in runs):
        raise ValueError(f"every run needs the one configuration {control!r} to compare the others with")
    tested: dict[Config, frozenset[str]] = {}
    filters: set[Config] = set()
    for run in runs:
        systems = fs.read_json(run.folder / "systems.json")
        for c in run.configs:
            if c.system is not None and c.name != control:
                if c.name not in systems:
                    raise ValueError(f"{run.folder / 'systems.json'} does not say which questions {c.name} was asked")
                if is_filter(c, systems[c.name]):
                    filters.add(c)
                else:
                    tested[c] = frozenset(systems[c.name]["asked"])
    return controls.pop(), tested, filters


def answerable(filters: Sequence[Config], loaded: list[tuple]) -> list[dict[str, list[str]]]:
    """For each run of `loaded`, the questions answerable from metadata, each with the agents (sorted) whose result with a leak filter
    passed it. An absent question is never one (C9). A question one agent's filter passed is one for every agent: it says what the
    dump leaks, not what the agent knows."""
    out = []
    for _, results in loaded:
        found: dict[str, set[str]] = {}
        for agent, arms in results.items():
            for f in filters:
                for qid, r in arms.get(f.dir, {}).items():
                    if r["passed"] and not r["absent"]:
                        found.setdefault(qid, set()).add(agent)
        out.append({qid: sorted(agents) for qid, agents in sorted(found.items())})
    return out


def metadata_figures(runs: Sequence[RunFolder], loaded: list[tuple], filters: Sequence[Config], leaks: list[dict[str, list[str]]]) -> dict:
    """`answerable_from_metadata`: the filters, how many questions with something to find they were given a done result for (in all
    the runs, by any agent), and the questions they answered, with the agents."""
    given = sum(len({qid for arms in results.values() for f in filters for qid, r in arms.get(f.dir, {}).items()
                     if is_done(r) and not r["absent"]}) for _, results in loaded)
    return {"filters": sorted(f.name for f in filters), "asked": given,
            "questions": [{"run": run.name, "seed": run.seed, "qid": q.qid, "type": q.tags["modality"], "level": q.hint,
                           "agents": leaks_of_run[q.qid]}
                          for run, (questions, _), leaks_of_run in zip(runs, loaded, leaks, strict=True) for q in questions
                          if q.qid in leaks_of_run]}


def numbers(runs: Sequence[RunFolder], control: str = "files") -> dict:
    """The report's figures over `runs`, pooled (report.json; `render.markdown` renders them): per agent and tested configuration
    (the configurations of the runs other than `control`'s and `blind`) the paired comparison with `control`, by hint level and per
    type, how each side worked and what a correct answer cost, the same by level pooled over the configurations, the retrieval tier,
    one row per question, and the counts that always print. `frame` is what the Markdown's sentences say of the runs besides their
    tables: the words for the two sides (the control's name, and the system's when the tested configurations have one), the time
    limit, and the tested configurations that are asked the questions that do not depend on the system."""
    first: dict[tuple, str] = {}
    for run in runs:
        if (run.seed, run.scale) in first:
            raise ValueError(f"{run.name} and {first[run.seed, run.scale]} are the same seed ({run.seed}) and scale "
                             f"({run.scale}), so the same questions (the plan does not depend on the days): pooling them "
                             "would count each question twice")
        first[run.seed, run.scale] = run.name
    if len({run.timeout_s for run in runs}) > 1:
        raise ValueError("runs with other time limits are not pooled: " + ", ".join(f"{run.name} {run.timeout_s:g} s" for run in runs))
    loaded = [(run.questions(), results_of(run)) for run in runs]
    controlled, asked, filters = sides_of(runs, control)
    tested = sorted(asked, key=lambda c: c.name)
    floors = sorted(filters, key=lambda c: c.name)
    leaks = answerable(floors, loaded)
    agents: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    for agent in sorted({a for _, results in loaded for a in results}):
        rows: list[tuple] = []
        for c in tested:
            if (row := compare(agent, c, asked[c], controlled, loaded, leaks)) is not None:
                rows += row.pop("rows")
                agents.setdefault(agent, {})[c.name] = row
        if agent in agents:
            levels = by_level(rows)
            pooled[agent] = {"levels": levels, "crossover": crossover(levels), "thin": thin_levels(levels)}
    every = [r for _, results in loaded for arms in results.values() for rs in arms.values() for r in rs.values()]
    problems = Counter(kind_of(r["problem"]) for r in every if r["problem"])
    ambient = sum(len(fs.read_json(f)) for run in runs for f in (run.folder / "results").glob("*/*/ambient-labels.json"))
    systems = {c.system for c in tested}
    return {"runs": [{"name": run.name, "seed": run.seed, "scale": run.scale, "days": len(run.days),
                      "commit": fs.read_json(run.folder / "run.json")["mmmb_commit"], "probe": probe_of(run.folder)} for run in runs],
            "frame": {"control": control, "tested": systems.pop() if len(systems) == 1 else "tested", "timeout_s": runs[0].timeout_s,
                      "text_asked_of": [c.name for c in tested if asked[c] - set(MEDIA)]},
            "agents": agents, "agent_models": agent_models(loaded), "pooled": pooled,
            **({"answerable_from_metadata": metadata_figures(runs, loaded, floors, leaks)} if floors else {}),
            "retrieval": decider_figures(runs),
            "appendix": {a: rows for a, rows in appendix_rows(runs, loaded, tested, controlled).items() if a in agents},
            "totals": {"results": len(every), "problems": dict(problems),
                       "to_rerun": sum(1 for r in every if not is_done(r)), "ambient_labels": ambient}}
