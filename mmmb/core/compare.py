"""`mmmb compare RUN_A RUN_B`: a run against another of the same questions, usually a re-run against the recorded run it repeats (plan 8.1,
decision D1). Per agent and configuration (a cell): the questions whose verdict flipped, by qid and with both answers, the correct counts,
whether the count is inside D1's band, McNemar's p of A against B, the problems and the `setup` ones; then, per agent and tested
configuration, the report's verdict line of each run, and whether D1's conditions hold.

What D1 settles, and what it does not: inside the band means no cell lost or gained more than BAND answers, which is no large loss. It does not
mean the runs agree, because the agents are stochastic, BAND of 31 questions is about 7 points, and McNemar over a few discordant pairs
has little power. So the flips are printed to be read, one by one, by a person.

A cell whose results were made by another model or at another effort than the other run's is refused (its counts would be of another
experiment), unless `across_settings` says to compare it anyway, and then the cell is marked."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from mmmb.core import fs, render, report
from mmmb.core.question import Question
from mmmb.core.record import kind_of
from mmmb.core.runfolder import RunFolder, is_done
from mmmb.core.stats import mcnemar

# D1 (plan 9.3), the one place these are written. The pilot's recorded counts were Claude files 23, Claude mm 22, Codex files 27, Codex mm 23.
BAND = 2  # a cell's correct count is within this many of the other run's
ALPHA = 0.05  # McNemar's p of the two runs is under this in no cell
ANSWER_FORMAT_RULE = "run once in json; run `line` as well only if a cell falls outside the band"

MEANING = ("A pass of D1's numbers means no large loss, not that the runs agree: the agents are stochastic, a band of {band} is about {pts:.0f} "
           "points of {n} questions, and McNemar's p over so few discordant pairs has little power. Read every flip above, one by one, before "
           "calling the runs the same.")


@dataclass(frozen=True)
class Flip:
    """A question both runs have a done result for, right in one and wrong in the other."""

    qid: str
    want: str  # what the question grades for
    a: dict
    b: dict


@dataclass(frozen=True)
class Cell:
    """One agent's one configuration in the two runs. A refused cell (`refusal`) or a missing one (`only`) has no figures."""

    agent: str
    config: str
    dir: str
    settings_a: tuple[str, ...]
    settings_b: tuple[str, ...]
    refusal: str | None = None
    only: str | None = None  # "A" or "B": the one run that has the cell
    n_a: int = 0
    n_b: int = 0
    correct_a: int = 0
    correct_b: int = 0
    paired: int = 0
    b: int = 0  # right in B only
    c: int = 0  # right in A only
    p: float = 1.0
    flips: tuple[Flip, ...] = ()
    unpaired: tuple[str, ...] = ()  # qids with a result in one run that did not make a done pair
    problems_a: Mapping[str, int] = field(default_factory=dict)
    problems_b: Mapping[str, int] = field(default_factory=dict)

    @property
    def figures(self) -> bool:
        return self.refusal is None and self.only is None

    @property
    def difference(self) -> int:
        return self.correct_b - self.correct_a

    @property
    def inside(self) -> bool:
        return self.figures and abs(self.difference) <= BAND

    @property
    def significant(self) -> bool:
        return self.figures and self.p < ALPHA

    @property
    def setup_b(self) -> int:
        return (self.problems_b or {}).get("setup", 0)


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    cells: tuple[Cell, ...]
    verdicts: Mapping[tuple[str, str], tuple[str, str]]  # (agent, tested configuration) -> the report's verdict line in A, in B
    n: int  # questions

    @property
    def refused(self) -> list[Cell]:
        return [c for c in self.cells if c.refusal]

    @property
    def conditions(self) -> dict[str, bool]:
        """D1's four numeric conditions: every cell compared and inside the band, no cell significant, the verdict lines the same, no
        `setup` in B. (A person reads the flips: the fifth.)"""
        return {"every cell inside the band": bool(self.cells) and all(c.inside for c in self.cells),
                f"no cell with McNemar p under {ALPHA:g}": not any(c.significant for c in self.cells),
                "the verdict lines read the same": all(x == y for x, y in self.verdicts.values()),
                "no setup result in B": not any(c.setup_b for c in self.cells)}


def settings_of(results: Mapping[str, dict]) -> tuple[str, ...]:
    """The models and efforts a cell's results say they were made by, as text: `model` or `model at effort`. An agent that records no
    effort (Claude's results hold none) has the model only."""
    return tuple(sorted({str(r.get("agent_model") or "unknown") + (f" at {r['agent_effort']} effort" if r.get("agent_effort") else "")
                         for r in results.values()}))


def want_of(q: Question) -> str:
    g = q.grade
    if g.rule == "absent":
        return "absent (unknown)"
    return " at ".join(dict.fromkeys(x for x in (g.truth, g.time) if x)) + (f" on {g.date}" if g.date else "")


def cell_of(agent: str, folder: str, a: Mapping[str, dict] | None, b: Mapping[str, dict] | None, questions: Mapping[str, Question],
            across_settings: bool) -> Cell:
    """The cell `agent`/`folder` from the results of A and B (None or empty: the run has none)."""
    named = next((r["config"] for rs in (a, b) if rs for r in rs.values()), folder)
    sa, sb = settings_of(a) if a else (), settings_of(b) if b else ()
    if not a or not b:
        return Cell(agent, named, folder, sa, sb, only="B" if not a else "A")
    if sa != sb and not across_settings:
        return Cell(agent, named, folder, sa, sb, refusal=f"made by {' + '.join(sa)} in A and {' + '.join(sb)} in B: another experiment; "
                    "--across-settings compares it anyway")
    pairs = {qid: (a[qid], b[qid]) for qid in a.keys() & b.keys() if is_done(a[qid]) and is_done(b[qid])}
    flips = tuple(Flip(qid, want_of(questions[qid]), x, y) for qid, (x, y) in sorted(pairs.items()) if bool(x["passed"]) != bool(y["passed"]))
    right_b = sum(1 for f in flips if f.b["passed"])
    problems = [Counter(kind_of(r["problem"]) for r in rs.values() if r["problem"]) for rs in (a, b)]
    return Cell(agent, named, folder, sa, sb, n_a=len(a), n_b=len(b), correct_a=sum(bool(r["passed"]) for r in a.values()),
                correct_b=sum(bool(r["passed"]) for r in b.values()), paired=len(pairs), b=right_b, c=len(flips) - right_b,
                p=mcnemar(right_b, len(flips) - right_b), flips=flips, unpaired=tuple(sorted((a.keys() | b.keys()) - pairs.keys())),
                problems_a=dict(problems[0]), problems_b=dict(problems[1]))


def verdicts_of(a: RunFolder, b: RunFolder) -> dict[tuple[str, str], tuple[str, str]]:
    """The crossover line of the report of each run, for each agent and tested configuration both have (the verdict the report gives)."""
    lines = []
    for run in (a, b):
        figures = report.numbers([run])
        lines.append({(agent, name): render.crossover_text(m["crossover"], figures["frame"]["tested"], m["thin"])
                      for agent, models in figures["agents"].items() for name, m in models.items()})
    return {key: (lines[0][key], lines[1][key]) for key in lines[0].keys() & lines[1].keys()}


def compare_cells(a: RunFolder, b: RunFolder, *, across_settings: bool = False) -> tuple[Cell, ...]:
    """The two runs cell by cell. They must be of one scenario and one plan (the same questions); a cell made by other models or efforts
    is refused unless `across_settings`."""
    if a.scenario.name != b.scenario.name:
        raise ValueError(f"{a.name} is of {a.scenario.name} and {b.name} of {b.scenario.name}: there is nothing to compare")
    if (plans := [fs.read_json(r.folder / "run.json")["plan"] for r in (a, b)])[0] != plans[1]:
        raise ValueError(f"{a.name} and {b.name} do not ask the same questions (their plans differ): their results cannot be paired")
    questions = {q.qid: q for q in a.questions()}
    if questions.keys() != {q.qid for q in b.questions()}:
        raise ValueError(f"{a.name} and {b.name} have other qids, though their plans are one")
    ra, rb = report.results_of(a), report.results_of(b)
    return tuple(cell_of(agent, folder, ra.get(agent, {}).get(folder), rb.get(agent, {}).get(folder), questions, across_settings)
                 for agent in sorted(ra.keys() | rb.keys()) for folder in sorted(ra.get(agent, {}).keys() | rb.get(agent, {}).keys()))


def compare_runs(a: RunFolder, b: RunFolder, *, across_settings: bool = False) -> Comparison:
    """The cells of `compare_cells` and the verdict line of each run's report (which makes each run's figures: seconds for the pilot)."""
    cells = compare_cells(a, b, across_settings=across_settings)
    return Comparison(a.name, b.name, cells, verdicts_of(a, b), len(a.questions()))


# -- the text ---------------------------------------------------------------------------------------------------------------------

def answer_text(r: dict) -> str:
    return ("pass" if r["passed"] else "FAIL") + f" {r['answer']!r}" + (f" ({kind_of(r['problem'])})" if r["problem"] else "")


def cell_lines(c: Cell) -> list[str]:
    head = f"{c.agent} {c.config}"
    if c.only:
        return [f"{head}: only run {c.only} has it, so it is not compared"]
    if c.refusal:
        return [f"{head}: REFUSED, {c.refusal}"]
    marks = [f"{c.correct_a}/{c.n_a} in A, {c.correct_b}/{c.n_b} in B, {c.difference:+d}",
             f"{'inside' if c.inside else 'OUTSIDE'} the band of +-{BAND}"]
    lines = [f"{head}: " + ", ".join(marks),
             f"    {c.paired} paired, {len(c.flips)} flipped (right in B only {c.b}, in A only {c.c}), McNemar p {c.p:.3f}"
             + (" (under " + f"{ALPHA:g})" if c.significant else ""),
             f"    problems: A {kinds(c.problems_a)}; B {kinds(c.problems_b)}; setup in B: {c.setup_b}"]
    if c.unpaired:
        lines.append(f"    {len(c.unpaired)} without a done pair, not compared: {', '.join(c.unpaired[:6])}" + (" ..." if len(c.unpaired) > 6 else ""))
    if c.settings_a != c.settings_b:
        lines.append(f"    SETTINGS DIFFER: A {' + '.join(c.settings_a)}, B {' + '.join(c.settings_b)}")
    return lines


def kinds(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items())) or "none"


def render_comparison(c: Comparison) -> str:
    """The comparison as text, for a person to read."""
    out = [f"compare {c.a} (A) with {c.b} (B): {c.n} questions", "",
           f"D1 (plan 9.3): a cell is inside the band when its correct count is within +-{BAND} of A's; McNemar's p of A against B is under "
           f"{ALPHA:g} in none; the verdict lines read the same; B has no setup result; every flip is read by a person. {ANSWER_FORMAT_RULE}.", ""]
    for cell in c.cells:
        out += cell_lines(cell)
    plain = sorted({cell.agent for cell in c.cells if cell.figures and not any("effort" in s for s in cell.settings_a + cell.settings_b)})
    if plain:
        out += ["", f"{', '.join(plain)} records no effort in its results, so only its model is compared."]
    out += ["", "Flips, to be read (the answers of A and of B, and what the question wants):"]
    flips = [(cell, f) for cell in c.cells for f in cell.flips]
    out += [f"  {cell.agent} {cell.config} {f.qid}: wants {f.want}; A {answer_text(f.a)}, B {answer_text(f.b)}" for cell, f in flips] or ["  none"]
    out += ["", "Verdict lines (the report's crossover sentence), A then B:"]
    for (agent, name), (x, y) in sorted(c.verdicts.items()):
        out += [f"  {agent} {name}: " + ("same: " + x if x == y else f"DIFFER\n      A {x}\n      B {y}")]
    out += ["", *(f"{'yes' if ok else 'NO'}: {what}" for what, ok in c.conditions.items())]
    if c.refused:
        out += ["", f"{len(c.refused)} cells were refused for their settings: nothing was concluded for them."]
    out += ["", MEANING.format(band=BAND, pts=100 * BAND / c.n, n=c.n)]
    return "\n".join(out)
