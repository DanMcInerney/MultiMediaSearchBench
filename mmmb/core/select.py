"""Which questions a run asks and which a configuration is asked. The subset a run is asked and what each configuration without tools
of its own is asked, which is what its system reads, moved from bench.py of this repository at bb498a6: a configuration with tools of
its own reads every modality, so it is asked every question. The choosing of cases is new (DESIGN 8.2, plan 6.1): `select` narrows a
run's questions by scenario, kind, modality, split and tag; `discriminating` keeps those that separated configurations in earlier runs,
with canaries; `stratified` draws a seeded sample that keeps the mix of kinds. A selection names the qids it picks, and the run is
asked exactly those (`--only`)."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from mmmb.core.config import Config
from mmmb.core.contract import Description
from mmmb.core.question import Question
from mmmb.core.runfolder import RunFolder, is_done, load_result

MEDIA = ("image", "video", "audio")  # the modalities that depend on what a system reads: the rest is text and records
CONTROL_TYPES = ("text", "records")
MODALITIES = MEDIA + CONTROL_TYPES
SUBSETS = ("all", "pilot")
SEED = 1  # the seed of the sample `--limit` draws: fixed, so that a small run is the same small run each time
CANARIES = 2  # cases that did not separate the configurations, kept in a discriminating selection to show the harness still answers them


def asked_modalities(config: Config, description: Description | None, controls: str | None = None) -> list[str]:
    """The modalities `config` is asked about: all of them when it has tools of its own or no system, else what its system reads,
    and of those the text and records only when `controls` is None (every configuration) or names it."""
    if config.own_tools or description is None:
        return sorted(MODALITIES)
    reads = set(description.reads)
    return sorted(reads if controls is None or config.name == controls else reads & set(MEDIA))


def config_questions(questions: Sequence[Question], config: Config, description: Description | None,
                     controls: str | None = None) -> list[Question]:
    """The questions of `questions` that `config` is asked (see `asked_modalities`)."""
    kinds = asked_modalities(config, description, controls)
    return [q for q in questions if q.tags["modality"] in kinds]


def pick_controls(questions: Sequence[Question]) -> list[Question]:
    """Three controls from the text and records questions, by their fields: a needle of each, and an absent one. Of each the
    one with a retrieval run, then the least vague hint, then the qid; so any scale has them."""
    from mmmb.core.report import LEVELS  # the report owns the order of the hints; the runner need not import the report to run

    pool = [q for q in questions if q.tags["modality"] in CONTROL_TYPES]

    def first(of: list[Question]) -> list[Question]:
        return [min(of, key=lambda q: (q.retrieval is None, LEVELS.index(q.hint), q.qid))] if of else []

    return [*first([q for q in pool if q.tags["modality"] == "text" and not q.tags["absent"]]),
            *first([q for q in pool if q.tags["modality"] == "records" and not q.tags["absent"]]),
            *first([q for q in pool if q.tags["absent"]])]


def pilot_questions(questions: Sequence[Question], reads: Iterable[frozenset[str]]) -> list[Question]:
    """The pilot: the media questions of the modalities some system of the run reads (`reads`; pictures and video when there is
    none), and 3 controls, in the questions' order. A scenario that marks its own pilot questions (a `pilot` tag) has those."""
    if any("pilot" in q.tags for q in questions):
        return [q for q in questions if q.tags.get("pilot")]
    heard = {k for r in reads for k in r} & set(MEDIA) or {"image", "video"}
    keep = {q.qid for q in questions if q.tags["modality"] in heard} | {q.qid for q in pick_controls(questions)}
    return [q for q in questions if q.qid in keep]


def selected(questions: Sequence[Question], subset: str, descriptions: Mapping[Config, Description | None],
             only: Sequence[str] = ()) -> list[Question]:
    """The questions a run asks: all of them, or the pilot's; with `only`, just those qids of them (a smoke). `descriptions` are
    those of the run's configurations (None for one with no system): the pilot's media are what the systems read that have no
    tools of their own."""
    if subset not in SUBSETS:
        raise ValueError(f"a subset is one of {', '.join(SUBSETS)}, not {subset!r}")
    reads = [d.reads for c, d in descriptions.items() if d is not None and not c.own_tools]
    picked = list(questions) if subset == "all" else pilot_questions(questions, reads)
    if only:
        if unknown := sorted(set(only) - {q.qid for q in picked}):
            raise ValueError(f"--only names questions that are not in the {subset} questions: {', '.join(unknown)}")
        picked = [q for q in picked if q.qid in set(only)]
    return picked


# -- choosing cases ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    """What `--scenarios --kinds --modalities --splits --tag --limit --discriminating` ask for; an empty field asks for nothing. `agents`
    are the agents the run is for: only their results in the `discriminating` runs count. (The subset and `--only` are `selected`'s.)"""

    scenarios: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    tags: Mapping[str, str] = field(default_factory=dict)
    limit: int | None = None
    seed: int = SEED
    discriminating: tuple[RunFolder, ...] = ()
    agents: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.scenarios or self.kinds or self.modalities or self.splits or self.tags or self.discriminating) and self.limit is None


def stratified(questions: Sequence[Question], n: int, seed: int) -> tuple[Question, ...]:
    """`n` of `questions` drawn at random with `seed`, in the questions' order, so that each (scenario, kind) has its share: the
    largest-remainder quota of n, whole numbers first and the left-over places to the largest fractions (a tie to the stratum named
    first). A stratum is sampled by its own generator, so another stratum's size does not change which of its questions are drawn. `n` over
    the total gives them all."""
    if n < 0:
        raise ValueError(f"a sample of {n} questions")
    total = len(questions)
    if n >= total:
        return tuple(questions)
    strata: dict[tuple[str, str], list[Question]] = {}
    for q in questions:
        strata.setdefault((q.scenario, q.kind), []).append(q)
    take = {key: n * len(of) // total for key, of in strata.items()}
    fraction = {key: n * len(of) % total for key, of in strata.items()}  # (in units of 1/total: exact, so no tie is a rounding accident)
    for key in sorted(strata, key=lambda k: (-fraction[k], k))[:n - sum(take.values())]:
        take[key] += 1
    picked = {q.qid for key, of in strata.items() for q in random.Random(f"{seed}/{key[0]}/{key[1]}").sample(of, take[key])}
    return tuple(q for q in questions if q.qid in picked)


@dataclass(frozen=True)
class Separation:
    """How earlier runs' results sort questions, by qid, in the questions' order: `separating` ones were answered right by some
    configurations and wrong by others of one agent in one run; of the rest, `right` and `wrong` were answered so by every done result,
    `mixed` differently by different agents or runs, and `unrun` have no done result (a `setup`, `limit`, `auth` or `error` result is
    not one: it did not test the question)."""

    separating: tuple[str, ...]
    right: tuple[str, ...]
    wrong: tuple[str, ...]
    mixed: tuple[str, ...]
    unrun: tuple[str, ...]


def separate(questions: Sequence[Question], runs: Sequence[RunFolder], agents: Sequence[str] = ()) -> Separation:
    """The questions sorted by what the done results of `runs` (of `agents`, or of every agent) say of them."""
    from mmmb.core.retrieval import FOLDER  # (it imports this module for the modalities)

    seen: dict[str, set[bool]] = {q.qid: set() for q in questions}
    separating: set[str] = set()
    for run in runs:
        for agent_dir in sorted(p for p in (run.folder / "results").glob("*") if p.is_dir() and p.name != FOLDER):
            if agents and agent_dir.name not in agents:
                continue
            here: dict[str, set[bool]] = {}  # qid -> what this run's configurations of this agent did
            for config_dir in sorted(p for p in agent_dir.iterdir() if p.is_dir()):
                for qid in seen:
                    if is_done(result := load_result(config_dir / f"{qid}.json")):
                        here.setdefault(qid, set()).add(bool(result["passed"]))
            for qid, passes in here.items():
                seen[qid] |= passes
                if len(passes) > 1:
                    separating.add(qid)
    groups = (separating,
              {qid for qid, v in seen.items() if v == {True} and qid not in separating},
              {qid for qid, v in seen.items() if v == {False} and qid not in separating},
              {qid for qid, v in seen.items() if len(v) > 1 and qid not in separating},
              {qid for qid, v in seen.items() if not v})
    return Separation(*(tuple(q.qid for q in questions if q.qid in group) for group in groups))


def canaries_of(separation: Separation, count: int = CANARIES) -> tuple[str, ...]:
    """`count` qids that did not separate the configurations, one right and one wrong in turn (the first of each in the questions'
    order), or all of one kind when there are no more of the other: a case every configuration gets right shows the harness still
    works, one every configuration gets wrong that nothing there changed."""
    pools = [list(separation.right), list(separation.wrong)]
    picked: list[str] = []
    while len(picked) < count and any(pools):
        for pool in pools:
            if pool and len(picked) < count:
                picked.append(pool.pop(0))
    return tuple(picked)


def discriminating(questions: Sequence[Question], runs: Sequence[RunFolder], canaries: int = CANARIES,
                   agents: Sequence[str] = ()) -> tuple[Question, ...]:
    """The questions of `questions` that separated configurations in `runs` (see `separate`), and `canaries` that did not (see
    `canaries_of`), in the questions' order."""
    sep = separate(questions, runs, agents)
    keep = {*sep.separating, *canaries_of(sep, canaries)}
    return tuple(q for q in questions if q.qid in keep)


def narrow(questions: Sequence[Question], flag: str, wanted: Sequence[str], key) -> list[Question]:
    """The questions whose `key` is one of `wanted` (any case). Refused when a wanted value is no question's: a typing mistake."""
    held = sorted({str(key(q)).lower() for q in questions})
    if unknown := [w for w in wanted if w.lower() not in held]:
        raise ValueError(f"{flag} names {', '.join(unknown)}, which no question has; they have {', '.join(held)}")
    return [q for q in questions if str(key(q)).lower() in {w.lower() for w in wanted}]


def select(questions: Sequence[Question], selection: Selection, trail: list[tuple[str, int]] | None = None) -> tuple[Question, ...]:
    """The questions `selection` picks, in the questions' order: those with the scenarios, kinds, modalities, splits and tags asked
    for; of them, with `discriminating` runs, the ones that separated the configurations there and the canaries; of them, with
    `limit`, a stratified sample of that many in all (the canaries count and stay: the sample is of the others). Raises when a value is
    no question's or a step leaves nothing. `trail` gets each step's name and the number of questions it leaves."""
    if selection.limit is not None and selection.limit < 1:
        raise ValueError(f"--limit is 1 or more, not {selection.limit}")
    chosen, steps = list(questions), trail if trail is not None else []
    steps.append(("the questions", len(chosen)))
    for flag, wanted, key in (("--scenarios", selection.scenarios, lambda q: q.scenario), ("--kinds", selection.kinds, lambda q: q.kind),
                              ("--modalities", selection.modalities, lambda q: q.tags["modality"]),
                              ("--splits", selection.splits, lambda q: q.split)):
        if wanted:
            chosen = narrow(chosen, flag, wanted, key)
            steps.append((f"{flag} {','.join(wanted)}", len(chosen)))
    for name, value in selection.tags.items():
        if not any(name in q.tags for q in chosen):
            have = ", ".join(sorted({t for q in chosen for t in q.tags}))
            raise ValueError(f"--tag {name}={value}: no question has the tag {name}; they have {have}")
        chosen = narrow([q for q in chosen if name in q.tags], f"--tag {name}", [value], lambda q, name=name: q.tags[name])
        steps.append((f"--tag {name}={value}", len(chosen)))
    keep = {q.qid for q in chosen}
    if selection.discriminating:
        named = ",".join(r.name for r in selection.discriminating)
        sep = separate(chosen, selection.discriminating, selection.agents)
        pool = [q for q in chosen if q.qid in sep.separating]
        if not pool:
            raise ValueError(f"none of the {len(chosen)} questions separated the configurations in {named} ({len(sep.unrun)} have no done result there)")
        canaries = canaries_of(sep)[:selection.limit]
        if selection.limit is not None:
            pool = list(stratified(pool, max(0, selection.limit - len(canaries)), selection.seed))
        keep = {q.qid for q in pool} | set(canaries)
        steps.append((f"--discriminating {named} ({len(pool)} that separated them, {len(canaries)} canaries)", len(keep)))
    elif selection.limit is not None:
        keep = {q.qid for q in stratified(chosen, selection.limit, selection.seed)}
        steps.append((f"--limit {selection.limit}", len(keep)))
    return tuple(q for q in chosen if q.qid in keep)
