"""The V8 report (moved from `duel` .. `markdown` of egolife.py of this repository at bb498a6): EgoLifeQA asked three ways, without the
recordings (`blind`, the floor), with the agent's own tools (the control, `files`) and with a system's tools (the tested configuration), as
paired comparisons over the questions with a done result in all three, as video and speech questions apart, by age and by category; how each
side worked and what a correct answer cost; the horizon's audit; and the retrieval tier (each tested configuration asked, with no agent, the
yes/no of every question's oracle). `markdown` renders the figures as report.md.

It reads the results the way core/report.py does (`report.results_of`, `report.paired`, `report.side`, ...), from the questions of the run
folder. A figure that the old report keyed by the arm (`files`, `mm`) is keyed by its role here: `control`, `tested`; the figure's `frame`
has the words the Markdown says for them (the control's configuration name and the tested system's), so a recorded run reads as it was
written. A tested configuration is shown by its variant (`d1-3B` of `mm/d1-3B`).

`run.json` holds, besides what a RunFolder needs, `pools` (the questions the bench could ask before choosing, from the download:
`v8_egolife.pool_counts`) and `decode_s` (seconds the reader decodes a glasses window in here, measured at ingest by the system)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from mmmb.core import fs, grade, render, report
from mmmb.core.question import Question
from mmmb.core.record import kind_of
from mmmb.core.retrieval import FOLDER as RETRIEVAL_FOLDER
from mmmb.core.retrieval import variant_of
from mmmb.core.runfolder import RunFolder, is_done
from mmmb.core.stats import median, wilson
from mmmb.scenarios.v8_egolife.choose import AGE_NAMES
from mmmb.scenarios.v8_egolife.source import CATEGORIES

# what mm assumes a glasses window costs to decode (Multimem's ask script: DECODE_S), which the recorded Markdown quotes beside the
# measured figure; tests/mm asserts that the two agree
ASSUMED_DECODE_S = 0.010

SPEECH_LABEL = {"records": "speech"}  # a question that needs sound is a records question to the core (the transcript is a log source)
P_LEVELS = (0.5, 0.3)  # the p from which a needle is counted as nearly found, besides the model's own hit
ROLES = ("blind", "control", "tested")  # the three configurations, in the order the tables list them


def duel(base: Sequence[dict], other: Sequence[dict], questions: Sequence[Question]) -> dict:
    """`report.paired` for two arms' results of the same questions: `base` and `other` figures, the difference `other - base`
    (the mean of the paired differences) with Newcombe's interval from MIN_DISCORDANT pairs, b (right only in `other`), c (right
    only in `base`) and McNemar's p."""
    got = report.paired(list(zip(questions, base, other)))
    return {"n": got["n"], "base": got["control"], "other": got["tested"], **{k: got[k] for k in ("diff", "ci", "b", "c", "p")}}


def unparsed(rs: Sequence[dict]) -> int:
    """The results with no problem whose answer has no letter A to D (they count as wrong)."""
    return sum(1 for r in rs if r["problem"] is None and grade.choice(r["answer"]) is None)


def arm_figures(rs: Sequence[dict]) -> dict:
    return {**report.side(list(rs)), "unparsed": unparsed(rs), "wilson": list(wilson(sum(r["passed"] for r in rs), len(rs)))}


def horizon_figures(questions: Sequence[Question], tested: Sequence[dict]) -> dict:
    """How much of each question's recording the horizon cut off (the seconds from H to the query), and what the tested results'
    audits say: recording times past H in their tool results, and asks started past it."""
    gaps = [datetime.fromisoformat(q.now).timestamp() - q.horizon for q in questions]
    audited = [r for r in tested if "after_horizon" in r]
    return {"median_s": median(gaps), "max_s": max(gaps) if gaps else None, "past_30s": sum(1 for g in gaps if g > 30),
            "audited": len(audited), "after_horizon": sum(r["after_horizon"] for r in audited),
            "asks": sum(r["asks"] for r in audited), "asks_past_horizon": sum(r["asks_past_horizon"] for r in audited)}


def group_figures(rows: list[tuple]) -> dict:
    """The figures of triples `(question, blind, control, tested)` of one set of questions: each arm, and the three pairings."""
    qs, blind, control, tested = ([r[i] for r in rows] for i in range(4))
    return {"n": len(rows), "arms": {"blind": arm_figures(blind), "control": arm_figures(control), "tested": arm_figures(tested)},
            "tested_control": duel(control, tested, qs), "control_blind": duel(blind, control, qs), "tested_blind": duel(blind, tested, qs)}


def agent_figures(questions: Sequence[Question], arms: dict, control: str, tested: str) -> dict | None:
    """One agent with one tested configuration: the questions with a done result in all three arms (a result to run again, or none,
    leaves a question out), as video and speech; the video ones by age and by category; how each arm worked; the crossover. `arms`
    are the agent's results by configuration folder; `control` and `tested` are the folders of those two."""
    qs = {q.qid: q for q in questions}
    blind, control_, tested_ = arms.get("blind", {}), arms.get(control, {}), arms.get(tested, {})
    if not tested_:
        return None
    rows = [(q, blind[qid], control_[qid], tested_[qid]) for qid, q in qs.items()
            if all(is_done(a.get(qid)) for a in (blind, control_, tested_))]
    video = [r for r in rows if r[0].tags["modality"] == "video"]
    speech = [r for r in rows if r[0].tags["modality"] == "records"]
    ages = {name: group_figures(of) for name in AGE_NAMES if (of := [r for r in video if r[0].tags["level"] == name])}
    levels = {name: report.lean([(q, f, m) for q, _, f, m in [r for r in video if r[0].tags["level"] == name]]) for name in ages}
    return {"questions": len(qs), "paired": len(rows), "video": group_figures(video) if video else None,
            "speech": group_figures(speech) if speech else None, "ages": ages,
            "crossover": report.crossover(levels, order=AGE_NAMES), "thin": report.thin_levels(levels, order=AGE_NAMES),
            "categories": {c: {"n": len(of), **{a: sum(r[i]["passed"] for r in of) for a, i in zip(ROLES, (1, 2, 3))}}
                           for c in CATEGORIES if (of := [r for r in video if r[0].tags["template"] == c])},
            "work": {"control": report.worked([r[2] for r in video]), "tested": report.worked([r[3] for r in video])},
            "cost": {a: report.per_correct([r[i] for r in video]) for a, i in zip(ROLES, (1, 2, 3))},
            "horizon": horizon_figures([r[0] for r in rows], [r[3] for r in rows]),
            "appendix": [{"qid": q.qid, "level": q.tags["level"], "template": q.tags["template"],
                          "type": SPEECH_LABEL.get(q.tags["modality"], q.tags["modality"]),
                          **{a: report.cell_of(r) for a, r in zip(ROLES, (b, f, m))}} for q, b, f, m in rows]}


def decider_recall(records: Sequence[dict]) -> dict:
    """A model's oracle records as recall: at its hit (`found`), at p of 0.5 and 0.3, in the top 5; the median needle p, the
    sightings away from every target, and the median seconds a window."""
    return {"asked": len(records), "found": sum(bool(r["found"]) for r in records),
            **{f"p{int(100 * x)}": sum((r["p"] or 0) >= x for r in records) for x in P_LEVELS},
            "top5": sum(bool(r["top5"]) for r in records), "p": median([r["p"] for r in records if r["p"] is not None]),
            "away": sum(r["away"] for r in records), "windows": sum(r["windows"] for r in records),
            "s_per_window": median([r["seconds"] / r["windows"] for r in records if r["windows"]]),
            "stopped": sum(1 for r in records if r["stopped"] == "max_s")}


def decider_figures(run: RunFolder) -> dict:
    """The retrieval tier per configuration: recall overall, as video and speech, and by category."""
    out = {}
    for folder in sorted((run.folder / "results" / RETRIEVAL_FOLDER).glob("*")):
        records = [fs.read_json(f) for f in sorted(folder.glob("*.json")) if f.stem != "warm"]
        warm_file = folder / "warm.json"
        if records:
            out[records[0]["config"]] = {
                "first_s": fs.read_json(warm_file)["seconds"] if warm_file.exists() else None, "all": decider_recall(records),
                "types": {SPEECH_LABEL.get(t, t): decider_recall(of) for t in ("video", "records")
                          if (of := [r for r in records if r["type"] == t])},
                "categories": {c: decider_recall(of) for c in CATEGORIES if (of := [r for r in records if r["template"] == c])}}
    return out


def numbers(run: RunFolder, control: str = "files") -> dict:
    """The report's figures (report.json; `markdown` renders them): per agent and tested configuration (the configurations of the
    run with a system that is not `control`) the three arms over the video questions and over the speech ones apart, by age and by
    category, how each arm worked and what a correct answer cost, the horizon's audit, and the retrieval tier. `frame` is what the
    Markdown calls the two sides: the control's name and the tested system's."""
    questions = run.questions()
    meta = fs.read_json(run.folder / "run.json")
    results = report.results_of(run)
    if len(controls := [c for c in run.configs if c.name == control]) != 1:
        raise ValueError(f"{run.name} needs the one configuration {control!r} to compare the others with")
    tested = sorted((c for c in run.configs if c.system is not None and c.name != control), key=lambda c: c.name)
    agents = {agent: {c.name: got for c in tested if (got := agent_figures(questions, arms, controls[0].dir, c.dir))}
              for agent, arms in results.items()}
    every = [r for arms in results.values() for rs in arms.values() for r in rs.values()]
    systems = {c.system for c in tested}
    return {"run": {"name": run.name, "seed": run.seed, "days": len(run.days), "commit": meta["mmmb_commit"],
                    "timeout_s": run.timeout_s, "decode_s": meta["decode_s"], "pools": meta["pools"],
                    "video": [sum(1 for q in questions if q.tags["modality"] == "video" and q.tags["level"] == a) for a in AGE_NAMES],
                    "speech": sum(1 for q in questions if q.tags["modality"] == "records"),
                    "categories": {c: sum(1 for q in questions if q.tags["template"] == c) for c in CATEGORIES},
                    "horizon": horizon_figures(questions, [])},
            "frame": {"control": control, "tested": systems.pop() if len(systems) == 1 else "tested"},
            "agents": {a: models for a, models in agents.items() if models}, "retrieval": decider_figures(run),
            "totals": {"results": len(every), "problems": dict(Counter(kind_of(r["problem"]) for r in every if r["problem"])),
                       "to_rerun": sum(1 for r in every if not is_done(r))}}


def right(k: int, n: int, ci: tuple[float, float] | None = None) -> str:
    """`k/n (60%)`, with the Wilson interval when given."""
    if not n:
        return "-"
    return f"{k}/{n} ({100 * k / n:.0f}%" + (f", {100 * ci[0]:.0f}-{100 * ci[1]:.0f}" if ci else "") + ")"


def diff_text(d: dict) -> str:
    """A difference in points, with its interval when there are pairs enough for one (b and c are in a column of their own)."""
    if d["diff"] is None:
        return "-"
    return render.pts(d["diff"]) + ("" if d["ci"] is None else f" [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}]")


def duel_row(label: str, d: dict) -> list[str]:
    return [label, str(d["n"]), diff_text(d), f"{d['b']} / {d['c']}", render.chance(d["p"])]


def arm_table(g: dict, frame: dict) -> list[str]:
    control, tested = frame["control"], frame["tested"]
    rows = []
    for role, arm in zip(ROLES, ("blind", control, tested)):
        a = g["arms"][role]
        rows.append([arm, right(a["correct"], a["n"], a["wilson"]), render.number(a["median_s"]), render.number(a["median_calls"], ".0f"),
                     render.kilo(a), str(a["timeouts"]), str(a["unparsed"])])
    return [render.table(["arm", "right (95% interval)", "median s", "median calls", "tokens k", "timed out", "no letter"], rows), "",
            render.table(["pair", "n", "difference (95% CI)", "b / c", "McNemar p"],
                         [duel_row(f"{tested} vs {control}", g["tested_control"]), duel_row(f"{control} vs blind", g["control_blind"]),
                          duel_row(f"{tested} vs blind", g["tested_blind"])]), "",
            "A difference is the second arm less the first, in points; b is the questions right only in the second, c only in the first."]


def age_lines(m: dict, frame: dict) -> list[str]:
    control, tested = frame["control"], frame["tested"]
    rows = []
    for name, g in m["ages"].items():
        a = g["arms"]
        d = g["tested_control"]
        rows.append([name, str(g["n"]), *(right(a[x]["correct"], g["n"]) for x in ROLES), diff_text(d), f"{d['b']} / {d['c']}",
                     render.chance(d["p"]), render.both(a["control"]["median_s"], a["tested"]["median_s"]),
                     render.both(render.tokens_k(a["control"]), render.tokens_k(a["tested"]), ".0f")])
    return [render.table(["age", "n", "blind (the floor)", control, tested, f"{tested} vs {control}", "b / c", "p",
                          f"median s ({control} / {tested})", f"tokens k ({control} / {tested})"], rows), "",
            f"The youngest age at which {tested} leads {control}: {render.crossover_text(m['crossover'], tested, m['thin'])} "
            "(n = 10 per age: a sign, not a finding)."]


def work_lines(m: dict, frame: dict) -> list[str]:
    rows = []
    for role in ("control", "tested"):
        w, c = m["work"][role], m["cost"][role]
        rows.append([frame[role], render.share(w["ask_execs"], w["n"]) if role == "tested" else "-",
                     " / ".join(render.number(w[k]) for k in ("asks", "finds", "looks")) if role == "tested" else "-",
                     render.known_of(w, "windows") if role == "tested" else "-", render.known_of(w, "gpu_s") if role == "tested" else "-",
                     str(w["max_s"]) if role == "tested" else "-", render.known_of(w, "views") if role == "control" else "-",
                     render.known_of(w, "shells") if role == "control" else "-", f"{c['correct']}/{c['n']}", render.cost_cell(c, "s"),
                     render.cost_cell(c, "tokens_k"), render.cost_cell(c, "gpu_s") if role == "tested" else "-"])
    return [render.table(["arm", "called ask", "asks / finds / looks", "ask windows", "ask GPU s", "asks stopped at max_s", "pictures viewed",
                          "shell commands", "correct", "s per correct", "tokens k per correct", "GPU s per correct"], rows)]


def decider_lines(tier: dict) -> list[str]:
    if not tier:
        return ["No decider results yet."]
    rows = []
    for name, d in tier.items():
        for label, r in [("all", d["all"]), *d["types"].items(), *d["categories"].items()]:
            rows.append([variant_of(name), label, str(r["asked"]), render.share(r["found"], r["asked"]), render.share(r["p50"], r["asked"]),
                         render.share(r["p30"], r["asked"]), render.share(r["top5"], r["asked"]), render.number(r["p"], ".2f"),
                         str(r["away"]), render.number(r["s_per_window"], ".3f"), str(r["stopped"])])
    return [render.table(["model", "set", "asked", "recall (at its hit)", "p >= 0.5", "p >= 0.3", "top 5", "median needle p",
                          "sightings away from every target", "median s per window", "stopped at max_s"], rows), "",
            "Recall and the top 5 are read first. A sighting away from every target is told beside them and is no fault: the "
            "event may recur."]


def horizon_rule(h: dict) -> str:
    return (f"Every arm saw the clips that ended by each question's horizon H (median {render.number(h['median_s'], '.0f')} s before "
            f"the question, at most {render.number(h['max_s'], '.0f')} s; {h['past_30s']} questions more than 30 s, which fell after a "
            "gap) and the transcript lines that ended by H.")


def audit_line(h: dict, tested: str) -> str:
    return (f"Recording times after H in {tested} tool results: {h['after_horizon']} in {h['audited']} audited execs; asks started past H "
            f"(the agent's confusion about {tested}'s clock, harmless): {h['asks_past_horizon']}/{h['asks']}.")


def markdown(figures: dict) -> str:
    """report.md: the tables of `numbers`."""
    r, frame = figures["run"], figures["frame"]
    control, tested = frame["control"], frame["tested"]
    arms = ("blind", control, tested)
    out = [f"# EgoLifeQA through the {tested} bench", "",
           f"Run {r['name']} (seed {r['seed']}, {r['days']} days, {tested} commit {r['commit']}). {sum(r['video'])} video questions "
           f"({', '.join(f'{n} aged {a}' for a, n in zip(AGE_NAMES, r['video']))}; by category "
           f"{', '.join(f'{c} {n}' for c, n in r['categories'].items())}) and {r['speech']} speech questions, asked as of each "
           f"question's horizon. The pool before choosing: {r['pools']['video']} video and {r['pools']['speech']} speech; dropped: "
           f"{', '.join(f'{n} {why}' for why, n in r['pools']['dropped'].items())}. Each exec had {r['timeout_s']:g} s. "
           f"A glasses window decodes in {r['decode_s'] * 1000:.0f} ms here ({tested} assumes {ASSUMED_DECODE_S * 1000:.0f}).",
           "", horizon_rule(r["horizon"]), "",
           "No shortcut probe: a public dataset, so the blind arm is the floor (what the agent knows without the recordings)."]
    for agent, models in figures["agents"].items():
        for config, m in models.items():
            name = variant_of(config)
            video = m["video"]
            out += ["", f"## {agent} with {name}: {video['n'] if video else 0} video questions of {m['questions']} (paired: done in "
                        "all three arms)", "", *(arm_table(video, frame) if video else ["None paired yet."]), "",
                    f"### {agent} with {name}: by age ({tested} vs {control}; blind is the floor)",
                    "", *age_lines(m, frame), "", f"### {agent} with {name}: by category (right / n)", "",
                    render.table(["category", "n", *arms], [[c, str(v["n"]), *(str(v[a]) for a in ROLES)]
                                                            for c, v in m["categories"].items()]),
                    "", "TaskMaster has few questions in each age: read it as a count, not a rate.", "",
                    f"### {agent} with {name}: the {m['speech']['n'] if m['speech'] else 0} speech questions, apart", ""]
            out += (arm_table(m["speech"], frame) + ["", f"Answered from the transcripts alone, since {name} does not hear."]
                    if m["speech"] else ["None paired yet."])
            out += ["", f"### {agent} with {name}: how each arm worked, and what a correct answer cost (the video questions)", "",
                    *work_lines(m, frame)]
            out += ["", f"### {agent} with {name}: the horizon, from the {tested} results", "", audit_line(m["horizon"], tested)]
    if not figures["agents"]:
        out += ["", "No agent results yet."]
    out += ["", "## The decider tier (each model's yes/no over each question's evidence, no agent)", "", *decider_lines(figures["retrieval"])]
    t = figures["totals"]
    out += ["", "## Read with these", "",
            "- The published scores (GPT-4o 36.2, Gemini-1.5-Pro 36.9 and EGAgent 57.5 in the paper) are on all 500 items and 7 days "
            "without tools: a scale, not a comparison.",
            "- Chance is 25%. With 40 paired questions p < 0.05 needs about 6 or more discordant pairs all one way; by age, 10 "
            "questions each, only a sign.",
            "- EgoLife has been public since 2025: the blind arm measures what the models know of it.",
            f"- The {control} and {tested} arms both get EgoLife's transcripts (a Chinese line and its English); {tested} reads them as "
            "records.",
            f"- A glasses window decodes in {r['decode_s'] * 1000:.0f} ms here and {tested} assumes {ASSUMED_DECODE_S * 1000:.0f} ms, so "
            "its estimates and lint are low for these clips; asks stopped at `max_s` and timed-out execs are counted above.",
            f"- Names are neutral in the {control} and {tested} arms: the clips are `glasses/<date>/HH.MM.SS.mp4`, with no person or "
            "dataset name.",
            "- A question is asked as of its horizon, so the seconds before it (above) are lost to every arm.",
            f"- Problems by kind: {render.kinds_text(t['problems'])}. Results to run again: {t['to_rerun']} of {t['results']}."]
    for agent, models in figures["agents"].items():
        for config, m in models.items():
            body = [[x["qid"], x["level"], x["template"], x["type"], *(render.cell_text(x[a], "asks") for a in ROLES)]
                    for x in m["appendix"]]
            out += ["", f"## Appendix: {agent} with {variant_of(config)}, one row per question", "",
                    render.table(["qid", "age", "category", "type", *arms], body)]
    return "\n".join(out) + "\n"
