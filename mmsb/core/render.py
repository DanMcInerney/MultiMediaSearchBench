"""The report as Markdown (moved from `table` .. `markdown` of bench.py of this repository at bb498a6): report.md, the tables of
`report.numbers` and the lines that always print. The words for the two sides come from the figures' `frame`: the control's name
(`files`) and the tested system's (`mm`), so a recorded run reads as it was written. A configuration is shown by its variant
(`d1-3B` of `mm/d1-3B`).

What the Markdown says of the data and of the tested system, in "Read with these" and in the tables' column names (the asks, finds
and looks, the GPU seconds, the decider tier), is the V7 benchmark's with mm: the recorded report.md says it, so it stays here
until a scenario or a system brings its own."""

from __future__ import annotations

from mmsb.core import report
from mmsb.core.probe import probe_line
from mmsb.core.retrieval import TYPES, family, variant_of
from mmsb.core.stats import MIN_DISCORDANT, mcnemar

# the share of an exec's time that mm's estimate lets an ask need and still finish (systems/mm/estimate.py has the same), which the
# sentence in "Read with these" quotes
ASK_SHARE = 0.75


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def share(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def number(x: float | None, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def chance(p: float | None) -> str:
    return "-" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}"


def kilo(arm: dict) -> str:
    """An arm's tokens in thousands (see `token_total`)."""
    return "-" if arm["tokens_all"] is None else f"{arm['tokens_all'] / 1000:.0f}"


def diff_cell(c: dict) -> str:
    """A difference in points with its interval; with fewer than MIN_DISCORDANT pairs right in one arm only, no interval, and
    b and c instead."""
    if c["diff"] is None:
        return "-"
    if c["ci"] is None:
        return f"{pts(c['diff'])} (too few discordant pairs: b {c['b']}, c {c['c']})"
    return f"{pts(c['diff'])} [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]"


def both(control, tested, spec: str = ".1f") -> str:
    return f"{number(control, spec)} / {number(tested, spec)}"


def kinds_text(counts: dict) -> str:
    return ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items())) or "none"


def pts(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:+.0f} pts"


def grounded_cells(f: dict, w: dict) -> list[str]:
    """A table's grounding columns for the two sides (`report.grounding_of`): the share grounded of those that recorded a status, the
    results that were right and not grounded (lucky), and the final messages that could not be read (unparsed). A side whose results
    hold no status shows `-`."""
    sides = [side.get("grounding") for side in (f, w)]
    return [" / ".join("-" if g is None else share(g["grounded"], g["grounded"] + g["lucky"] + g["wrong"]) for g in sides),
            *(" / ".join("-" if g is None else str(g[key]) for g in sides) for key in ("lucky", "unparsed"))]


def has_grounding(figures: dict) -> bool:
    """Whether any side of any table holds a grounded status: a report of recorded runs, or of the `line` format, has none."""
    return any("grounding" in c[side] for models in figures["agents"].values() for m in models.values() for c in (m["all"], m["media"])
               for side in ("control", "tested"))


def tokens_k(arm: dict) -> float | None:
    """A side's mean tokens per exec, in thousands."""
    return None if arm["tokens_mean"] is None else arm["tokens_mean"] / 1000


def known_of(w: dict, key: str, spec: str = ".1f") -> str:
    """A per-exec mean of `worked`, with how many of the results it comes from when that is not all of them."""
    if w[key] is None:
        return "-"
    return format(w[key], spec) + ("" if w["known"] == w["n"] else f" ({w['known']}/{w['n']})")


def crossover_text(c: dict | None, tested: str, thin: dict | None = None) -> str:
    """The crossover of a model or of all: the level and which branch held (accuracy, cost per correct answer, or both), or why
    there is none, naming the levels that have too few pairs. `tested` is the word for the tested side."""
    if c is None:
        few = ", ".join(f"{level} ({n})" for level, n in (thin or {}).items())
        return (f"none: at no level with {report.MIN_LEVEL_N} or more paired questions does {tested} lead by 20 points, or cost half as much "
                "per correct answer without being less accurate" + (f"; too few paired questions at {few}" if few else ""))
    bits = [f"accuracy {pts(c['lead'])}"]
    if "cost" in c["by"]:
        bits += [f"{c[k]:.2f}x the {k} per correct answer" for k in ("seconds", "tokens") if c[k] is not None]
    return f"`{c['level']}` by {' and '.join(c['by'])} ({', '.join(bits)})"


def level_lines(figures: dict) -> list[str]:
    """Per agent, accuracy with and without the tested system, seconds, tokens and timeouts at each level of the hint, per model
    and pooled over the models, and the crossover sentence for each."""
    out: list[str] = []
    control, tested = figures["frame"]["control"], figures["frame"]["tested"]
    for agent, models in figures["agents"].items():
        pooled = figures["pooled"][agent]
        groups = [(variant_of(name), m["levels"], m["crossover"], m["thin"]) for name, m in models.items()]
        if len(models) > 1:
            groups.append(("all models", pooled["levels"], pooled["crossover"], pooled["thin"]))
        rows, notes = [], []
        for label, levels, cross, thin in groups:
            for level, c in levels.items():
                f, w = c["control"], c["tested"]
                rows.append([label, level, str(c["n"]), share(f["correct"], c["n_control"]), share(w["correct"], c["n"]),
                             diff_cell(c), f"{c['b']} / {c['c']}", chance(c["p"]), both(f["median_s"], w["median_s"]),
                             both(tokens_k(f), tokens_k(w), ".0f"), f"{f['timeouts']} / {w['timeouts']}"])
            notes.append(f"- {label}: " + (crossover_text(cross, tested, thin) if levels else "no media question paired"))
        out += ["", f"## {agent}: by how vague the time hint is (the media questions, paired)", "",
                table(["model", "level", "paired", f"without {tested}", f"with {tested}", "difference (95% CI)", "b / c", "McNemar p",
                       f"median s ({control} / {tested})", f"tokens k per exec ({control} / {tested})",
                       f"timed out ({control} / {tested})"], rows),
                "", f"Crossover, the first of part, day and week with at least 10 paired questions where {tested} leads by 20 points "
                "or more, or costs half as much per correct answer or less (seconds, or tokens) without being less accurate:",
                "", *notes]
    return out


def worked_lines(figures: dict) -> list[str]:
    """How each side worked, per agent and model, per exec on the paired media questions."""
    rows = []
    control, tested = figures["frame"]["control"], figures["frame"]["tested"]
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            f, w = m["work"]["control"], m["work"]["tested"]
            rows.append([agent, variant_of(name), control, str(f["n"]), "-", "-", "-", "-", "-", "-", "-", "-", known_of(f, "views"),
                         known_of(f, "shells")])
            rows.append([agent, variant_of(name), tested, str(w["n"]), share(w["ask_execs"], w["n"]),
                         " / ".join(number(w[k]) for k in ("asks", "finds", "looks")), known_of(w, "windows"),
                         known_of(w, "gpu_s"), str(w["max_s"]) if w["known"] else "-", number(w["estimate_s"]),
                         f"{number(w['held_s'])} ({w['alone']} alone)" if w["held_s"] is not None else "-", str(w["beyond"]), "-", "-"])
    return ["", "## How each arm worked (the media questions, per exec)", "",
            table(["agent", "model", "arm", "execs", "called ask", "asks / finds / looks", "ask windows", "ask GPU s",
                   "asks stopped at max_s", "GPU s estimated", "held s (run alone)", "estimated past the timeout", "pictures viewed", "shell commands"], rows)]


def cost_cell(c: dict, key: str, spec: str = ".1f") -> str:
    """A cost per correct answer, told only from MIN_CORRECT correct answers (k is beside it)."""
    return number(c[key], spec) if c["correct"] >= report.MIN_CORRECT else "-"


def cost_lines(figures: dict) -> list[str]:
    """What a correct answer costs each side, per agent and model, on the paired media questions. Fewer than MIN_CORRECT correct
    answers (k) make no figure."""
    rows = []
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            for side in ("control", "tested"):
                c = m["cost"][side]
                rows.append([agent, variant_of(name), figures["frame"][side], f"{c['correct']}/{c['n']}", cost_cell(c, "s"),
                             cost_cell(c, "tokens_k"), cost_cell(c, "gpu_s") if side == "tested" else "-"])
    return ["", "## Cost per correct answer (the media questions)", "",
            table(["agent", "model", "arm", "correct (k)", "seconds", "tokens k", "GPU seconds"], rows), "",
            f"A cost per correct answer is told from {report.MIN_CORRECT} correct answers (k) in the arm; below that it is `-`."]


def verdict_lines(tier: dict) -> list[str]:
    """Per template: who found its needles, who nearly missed them (in the top 5 or with p of 0.5 or more), who missed them,
    whose scan ended before the needle (stopped on a false sighting, or cut at max_s), the needle's p, and the verdict on what a
    miss says."""
    rows = []
    for template, t in tier["templates"].items():
        models = t["models"]
        verdict = ("; ".join(f"{text}: {', '.join(variant_of(n) for n in v['models'])} ({', '.join(v['levels'])})"
                             for text, v in t["verdicts"].items())
                   or "found by every model asked")
        rows.append([template, str(t["questions"]),
                     ", ".join(f"{variant_of(n)} {m['found']}/{m['asked']}" for n, m in models.items() if m["found"]) or "-",
                     ", ".join(f"{variant_of(n)} {m['near']}" for n, m in models.items() if m["near"]) or "-",
                     ", ".join(f"{variant_of(n)} {m['missed']}" for n, m in models.items() if m["missed"]) or "-",
                     ", ".join(f"{variant_of(n)} {m['stopped'] + m['cut']}" for n, m in models.items()
                               if m["stopped"] + m["cut"]) or "-",
                     ", ".join(f"{variant_of(n)} {m['p']:.2f}" for n, m in models.items() if m["p"] is not None) or "-", verdict])
    return ["", table(["template", "needle questions", "found by", "nearly missed by (top 5 or p >= 0.5)", "missed by",
                       "scan ended before the needle by (false sighting, max_s)", "needle p (median)", "verdict"], rows)]


def summary_lines(figures: dict) -> list[str]:
    """One row per model: the decider tier's media figures beside each agent's accuracy and cost on the media with the tested
    system."""
    tier, agents = figures["retrieval"]["models"], figures["agents"]
    names = sorted({*tier, *(n for models in agents.values() for n in models)})
    if not names:
        return []
    header = ["model", "family", "recall (at its hit)", "recall at p >= 0.5", "top 5", "false sightings per ask",
              "median s per window"]
    for agent in agents:
        header += [f"{agent}: {figures['frame']['tested']} accuracy", f"{agent}: s / k tokens / GPU s per correct"]
    rows = []
    for name in names:
        d = tier.get(name)
        m = d["media"] if d else None
        row = [variant_of(name), family(name), *((share(m["found"], m["needles"]), share(m["p50"], m["needles"]),
                                     share(m["top5"], m["needles"]), number(m["false"] / m["asked"], ".2f") if m["asked"] else "-",
                                     number(m["s_per_window"], ".3f")) if m else ("-",) * 5)]
        for models in agents.values():
            a = models.get(name)
            c = a["cost"]["tested"] if a else None
            row += [share(a["media"]["tested"]["correct"], a["media"]["n"]) if a else "-",
                    f"{cost_cell(c, 's')} / {cost_cell(c, 'tokens_k')} / {cost_cell(c, 'gpu_s')} (k={c['correct']})" if c else "-"]
        rows.append(row)
    return ["", "## Summary per model (the media questions)", "", table(header, rows)]


def cell_text(c: dict | None, what: str) -> str:
    """One side's result of a question in the appendix: pass or FAIL (and the kind of problem), seconds, then the views (`v`) or
    asks (`a`) it made."""
    if c is None:
        return "-"
    n = c["views"] if what == "views" else c["asks"]
    problem = f" {c['problem']}" if c["problem"] else ""
    flag = " (est. past timeout)" if c.get("beyond") else ""
    return f"{'pass' if c['passed'] else 'FAIL'}{problem} {c['seconds']:.0f} s {'v' if what == 'views' else 'a'}{'-' if n is None else n}{flag}"


def appendix_lines(figures: dict) -> list[str]:
    """One row per question and agent, to read the failures: qid, level, type, the control's pass, seconds and pictures viewed
    (`v`), and each model's pass, seconds and asks (`a`)."""
    out: list[str] = []
    many = len(figures["runs"]) > 1
    control, tested = figures["frame"]["control"], figures["frame"]["tested"]
    for agent, rows in figures["appendix"].items():
        names = list(figures["agents"][agent])
        body = [[*([str(r["seed"])] if many else []), r["qid"], r["level"], r["type"], cell_text(r["control"], "views"),
                 *(cell_text(r["tested"].get(n), "asks") for n in names)] for r in rows]
        out += ["", f"## Appendix: {agent}, one row per question", "",
                table([*(["seed"] if many else []), "qid", "level", "type", control, *(f"{tested} {variant_of(n)}" for n in names)],
                      body)]
    return out


def answerable_lines(figures: dict) -> list[str]:
    """The questions a leak filter answered from the files' metadata (the figures' `answerable_from_metadata`), which no paired count
    above holds; nothing when the runs have no leak filter."""
    found = figures.get("answerable_from_metadata")
    if found is None:
        return []
    filters, questions = ", ".join(found["filters"]), found["questions"]
    out = ["", "## Answerable from metadata", "",
           f"The leak filter ({filters}) was asked {found['asked']} questions that have something to find (an absent question is exempt: "
           f"an agent with nothing to read answers \"unknown\" to it) and answered {len(questions) or 'none'} of them right. Those are "
           "left out of every paired count above, the questions as well as the results."]
    if questions:
        out += ["", table(["run", "qid", "type", "level", "answered by"],
                          [[q["run"], q["qid"], q["type"], q["level"], ", ".join(q["agents"])] for q in questions])]
    return out


def markdown(figures: dict) -> str:
    """report.md: the tables of `numbers`, and the lines that always print."""
    frame = figures["frame"]
    control, tested = frame["control"], frame["tested"]
    out = [f"# {tested} benchmark: with {tested} and without", ""]
    runs = figures["runs"]
    out.append("Runs pooled: " + "; ".join(f"{r['name']} (seed {r['seed']}, {r['scale']}, {r['days']} days, commit "
                                            f"{r['commit']})" for r in runs) + ".")
    if figures.get("agent_models"):
        out.append("Agents: " + "; ".join(f"{agent} " + ", ".join(f"{name} ({n} results)" for name, n in names.items())
                                          for agent, names in figures["agent_models"].items()) + ".")
    grounded = has_grounding(figures)
    for agent, models in figures["agents"].items():
        out += ["", f"## {agent}: accuracy with and without {tested}, on each model's questions", ""]
        rows = []
        for name, m in models.items():
            for label, c in (("all", m["all"]), ("media", m["media"])):
                diff = diff_cell(c)
                f, w = c["control"], c["tested"]
                headline = label == "all"  # the counts of what was left out and what the tested system was asked are of the whole model
                rows.append([variant_of(name), label, f"{c['n']}/{m['questions']}" if headline else str(c["n"]),
                             share(f["correct"], c["n"]), share(w["correct"], c["n"]), diff, f"{c['b']} / {c['c']}",
                             chance(c["p"]), f"{f['absent_answered']}/{f['absent']} / {w['absent_answered']}/{w['absent']}",
                             f"{f['abstained']}/{f['needles']} / {w['abstained']}/{w['needles']}",
                             both(f["median_s"], w["median_s"]), both(f["median_calls"], w["median_calls"], ".0f"),
                             f"{f['timeouts']} / {w['timeouts']}", f"{kilo(f)} / {kilo(w)}", both(f["cost_usd"], w["cost_usd"], ".2f"),
                             str(m["left_out"]) if headline else "-", str(m["scope_all"]) if headline else "-",
                             f"{kinds_text(m['kinds']['control'])} / {kinds_text(m['kinds']['tested'])}" if headline else "-",
                             *(grounded_cells(f, w) if grounded else [])])
        sides = f"({control} / {tested})"
        out.append(table(["model", "questions", "paired", f"without {tested}", f"with {tested}", "difference (95% CI)", "b / c",
                          "McNemar p", f"wrong on absent {sides}", f"abstained on needles {sides}",
                          f"median s {sides}", f"median calls {sides}", f"timed out {sides}",
                          f"tokens k {sides}", f"cost $ {sides}", "left out", "scope all", f"problems {sides}",
                          *([f"grounded {sides}", f"lucky {sides}", f"unparsed {sides}"] if grounded else [])], rows))
    if figures["agents"]:
        out += level_lines(figures) + worked_lines(figures) + cost_lines(figures)
        out += ["", f"## By data type: without {tested} -> with {tested}, right/paired", ""]
        rows = [[agent, variant_of(name), *(f"{c['control']}/{c['n']} -> {c['tested']}/{c['n']}" if (c := m["types"].get(type_)) else "-"
                                            for type_ in TYPES)]
                for agent, models in figures["agents"].items() for name, m in models.items()]
        out.append(table(["agent", "model", *TYPES], rows))
    else:
        out += ["", "No agent results yet."]
    out += answerable_lines(figures)
    out += ["", "## The decider tier (each model's yes/no over each needle's source and range, no agent)", ""]
    tier = figures["retrieval"]
    if tier["asks"]:
        rows = []
        for name, m in tier["models"].items():
            for type_, t in m["types"].items():
                rows.append([variant_of(name), type_, share(t["found"], t["needles"]), share(t["p50"], t["needles"]),
                             share(t["top5"], t["needles"]), share(t["absent_clean"], t["absent"]),
                             number(t["false"] / t["asked"], ".2f"), number(t["s_per_window"], ".3f"), number(m["first_s"])])
        out.append(table(["model", "type", "recall (at its hit)", "recall at p >= 0.5", "top 5",
                          "absent with no false sighting", "false sightings per ask", "median s per window", "first ask s"],
                         rows))
        out += ["", ("Templates no model found: " + ", ".join(tier["missed"]) + ". The verdicts below say whether that is the "
                     "haystack or the models.") if tier["missed"] else "Every template was found by some model."]
        out += verdict_lines(tier)
    else:
        out.append("No decider results yet.")
    out += summary_lines(figures)
    t = figures["totals"]
    rows_k = sum(len(models) for models in figures["agents"].values())
    low = sum(1 for models in figures["agents"].values() for m in models.values() for c in (m["all"], m["media"])
              if c["p"] is not None and c["p"] < 0.05)
    out += ["", "## Read with these", "",
            f"- Without {tested} the screen, the microphone and the cameras would not be recorded at all: the {control} arm reads a "
            "week that exists only because the bench wrote it.",
            "- The screen changes like a real one while the owner is active: typing (small changes), scrolling (large ones), a cursor that "
            "moves and a clock, up to 2 frames a second, with idle and busy stretches. Its totals and the size of its changes are calibrated "
            "against two real hours of a screen recording (TESTBED.md). A caret's blinks are not modelled, and a scroll wraps instead of "
            "bringing new lines; the reader's change gate turns 98% of the haystack's seconds with a frame into windows, where it turns "
            "71 to 86% of a real screen's.",
            "- `look` costs about 25 times more per frame than an ffmpeg montage: it returns separate pictures, 8 to a call, "
            "where a 5x5 montage is about 1.5k tokens. An agent that opens frames with `look` pays for it; one that tiles "
            "them does not.",
            f"- With about 20 paired questions, p < 0.05 needs at least 6 discordant pairs all one way (McNemar exact: "
            f"mcnemar(6, 0) = {mcnemar(6, 0):.3f}, mcnemar(5, 0) = {mcnemar(5, 0):.4f}). A run with another seed adds new "
            "needles, and `report` pools runs. By hint level there are a few questions each: pool seeds before reading it.",
            f"- The text and records questions do not depend on the model, so only one model's {tested} arm "
            f"({', '.join(variant_of(n) for n in frame['text_asked_of'])}) is asked them: the other models' rows hold the media questions.",
            "- The decider tier's recall at p >= 0.5 counts a needle whose best p within 10 s of it, as a sighting or a "
            "candidate, is 0.5 or more: where it is above the recall at the model's hit, that hit may be set too high.",
            "- Tokens are the input, the output and for Claude the cache it read and wrote (Codex's input already holds its cached "
            "tokens): a size of work, not a bill, since a cached token costs much less. Cost is Claude's list-price estimate; "
            "Codex reports none.",
            f"- The difference is told in points with Newcombe's hybrid score interval for paired data, only from {MIN_DISCORDANT} "
            "pairs right in one arm only (b + c); below that the interval would exclude 0 where McNemar's exact p does not, so "
            "b and c and the p are given alone. A verdict rests on McNemar's p. A timed-out exec is a wrong answer and is "
            "counted on its own too (the `timed out` columns).",
            f"- An {tested} exec whose ask may need more GPU seconds than {ASK_SHARE:.0%} of the exec's time ({ASK_SHARE * frame['timeout_s']:g} s of "
            f"{frame['timeout_s']:g}: the agent's own turns take the rest; the `held s` and `estimated past the timeout` columns, and the appendix) "
            "may time out because of the model's speed alone: that is not a miss of its eyes, and `agents` names those questions before it runs.",
            "- The rows are not independent: the part and day questions of a template share a needle on the `pilot` scale, a "
            "template's levels share its week, and pooled seeds repeat templates. Read a level or a model with a few dozen "
            "questions as a sign to follow up, not as a finding.",
            f"- Multiple comparisons: this report has k = {rows_k} agent-model rows, each tested on all its questions and on "
            f"the media alone ({2 * rows_k} p values). At p < 0.05 about {0.05 * rows_k:.2f} of k rows would fall below it by "
            f"chance alone; {low} of the {2 * rows_k} p values here are.",
            f"- Ask labels with no project (an exec whose questions could leak into another's scope): {t['ambient_labels']}. "
            f"Problems by kind: {kinds_text(t['problems'])}. Results to run again: {t['to_rerun']} of {t['results']}.",
            "- Paired: a question counts when both arms have a done result. An escaped, timed-out or capped exec is a wrong "
            "answer; a `setup`, `limit`, `auth` or `error` result did not test the question, is run again by the next "
            "command, and is left out (the `left out` column).",
            *(["- Grounded: the answer is right and the evidence it cites is at the moment the question is about (a video within 10 s, a "
               "sound or a line of a log within 5 s, a picture or a note by being that file); an absent question is grounded when it says "
               "unknown and cites nothing. Lucky: right, but it cites nothing or something elsewhere. Unparsed: the final message was no "
               "JSON answer of the form asked for, which is a failed answer. The columns hold the results that recorded a status; the "
               "verdict beside them (the without and with columns) is the value rule alone."] if grounded else []),
            f"- Runs pooled: {len(runs)}; seeds {', '.join(str(r['seed']) for r in runs)}."]
    out[-1:-1] = [f"- Shortcut probe: {line}." for r in runs if (line := probe_line(r["name"], r["probe"]))]
    out += appendix_lines(figures)
    return "\n".join(out) + "\n"
