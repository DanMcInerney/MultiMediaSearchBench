"""The retrieval tier: each system asked, with no agent, the yes/no of a needle over its source and range, and the ranked moments it
gives scored against the question's truth (moved from `oracle`, `run_decider`, `span_of`, `meets`, `touched`, `needle_p`, `outcome`,
`verdicts` and `decider_figures` of bench.py of this repository at bb498a6, and `oracle_record` of egolife.py). The asking is the
system's (`Session.oracle`); the scoring and the records are here. One record holds the fields of both scenarios' old records, and
`decider_figures` turns the records of a run into the tier's figures for the report.

    results/retrieval/<Config.dir>/<qid>.json    one record for each question (`retrieval_record`)
    results/retrieval/<Config.dir>/warm.json     {"config", "seconds"}: how long the session took to open, loaded and warm

A record is of one configuration (`config`, `system`, `variant`, as an agent's result is). TYPES and LEVELS are shared with
`core/report.py`, which re-exports them: this module is below it."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from mmmb.core import fs
from mmmb.core.config import Config
from mmmb.core.contract import Moment, RetrievalQuery, RetrievalResult, Session
from mmmb.core.question import Question
from mmmb.core.runfolder import RunFolder
from mmmb.core.select import CONTROL_TYPES, MEDIA
from mmmb.core.stats import median

FOLDER = "retrieval"  # the folder of results/ that holds the tier's records, not an agent's
CANDIDATES, MAX_S = 5, 1800.0  # what every system is allowed an ask in this tier: candidates besides its sightings, and seconds
TYPES = CONTROL_TYPES + MEDIA  # what a question is about, in the order the report lists them (core/select.py has the modalities)
LEVELS = ("part", "day", "week", "time")  # how vague a question's time hint is; `time` is an exact time (config-asof)
NEAR_P = 0.5  # a needle's best p from which a miss is a near miss

# the wordings of a verdict on a needle question that a configuration did not find
SUSPECT = "haystack suspect"
CALIBRATION = "below its model's hit (calibration)"
MISS = "this model's miss"
STOPPED = "stopped on a false sighting before the needle"
CUT = "cut at max_s before the needle"
ONE_MODEL = "one model ran: the agent arm's look, or a second family, tells a miss from a haystack fault"
ONE_FAMILY = "one family ran: the agent arm's look, or a second family, tells a miss from a haystack fault"
OUTCOMES = ("found", "near", "stopped", "cut", "missed")


def span_of(found: Moment) -> tuple[float, float]:
    """The start and end of a sighting or candidate (epoch seconds): the same for one instant."""
    return found.t0, found.t1


def meets(a0: float, a1: float, moments: Sequence[Sequence[float]], tol: float) -> bool:
    """Whether the span a0..a1 comes within `tol` seconds of a moment."""
    return any(a0 <= m1 + tol and a1 >= m0 - tol for m0, m1 in moments)


def touched(spans: Sequence[tuple[float, float]], moments: Sequence[Sequence[float]], tol: float) -> list[int]:
    """The indexes of the moments some span comes within `tol` seconds of."""
    return [i for i, (m0, m1) in enumerate(moments) if any(a <= m1 + tol and b >= m0 - tol for a, b in spans)]


def needle_p(moments: Sequence[Sequence[float]], sightings: Sequence[Moment], candidates: Sequence[Moment], tol: float) -> float | None:
    """The needle's p: the best p of a sighting or a candidate within `tol` seconds of a moment of the truth, or None when
    the ask named none (the needle was neither a hit nor among its best windows)."""
    ps = [s.p for s in sightings if meets(*span_of(s), moments, tol)]
    ps += [c.p for c in candidates if meets(t := span_of(c)[0], t, moments, tol)]
    return max(ps) if ps else None


def query_of(q: Question) -> RetrievalQuery:
    """What the tier asks for question `q`: the words, sources, ranges and limit of its retrieval spec, with the allowances every
    system gets (CANDIDATES, MAX_S)."""
    spec = q.retrieval
    if spec is None:
        raise ValueError(f"{q.qid} has no retrieval run")
    return RetrievalQuery(q.qid, spec.query, spec.sources, spec.ranges, spec.k, CANDIDATES, spec.oldest_first, MAX_S)


def retrieval_record(q: Question, config: Config, query: RetrievalQuery, result: RetrievalResult) -> dict:
    """The record of one question's retrieval run: what `query` asked and what the system gave, scored against the moments of the
    question's truth within its spec's tolerance. Only a sighting that overlaps a range counts (`in_range`), since an earlier note or
    version is read too. A sighting of another needle of the template (`others`) is no false sighting. `found`: such a sighting
    touches a moment; `top5`: found, or a candidate that near; `p`: the needle's best p (`needle_p`); `rank`: the place of the first
    in-range sighting that touches the truth, newest first (by end), or None; `away`: sightings, in range or not, farther from every
    moment than the tolerance (the event may recur, so they are told beside the recall and not as a fault). An absent question has
    none of `found`, `top5`, `p` and `rank`."""
    spec = q.retrieval
    tol, moments, others = spec.tol_s, spec.moments, spec.others
    absent = bool(q.tags["absent"])
    spans = [span_of(s) for s in result.sightings]
    inside = [(a, b) for a, b in spans if any(a < until and b >= since for since, until in query.ranges)]
    false = sum(1 for a, b in inside if not meets(a, b, moments, tol) and not meets(a, b, others, tol))
    found = top5 = p = rank = None
    got: list[int] = []
    if not absent:
        got = touched(inside, moments, tol)
        found = bool(got)
        newest_first = sorted(inside, key=lambda s: s[1], reverse=True)
        rank = next((i for i, (a, b) in enumerate(newest_first, 1) if meets(a, b, moments, tol)), None)
        top5 = found or any(meets(t, t, moments, tol) for t in (span_of(c)[0] for c in result.candidates))
        p = needle_p(moments, result.sightings, result.candidates, tol)
    return {"qid": q.qid, "type": q.tags["modality"], "template": q.tags["template"], "level": q.tags["level"], "last": bool(q.tags.get("last")),
            "absent": absent, "config": config.name, "system": config.system, "variant": config.variant, "question": query.text,
            "source": ",".join(query.sources), "since": min(a for a, _ in query.ranges), "until": max(b for _, b in query.ranges),
            "ranges": [list(r) for r in query.ranges], "limit": query.k, "found": found, "top5": top5, "p": p, "rank": rank,
            "moments": len(moments), "moments_found": got, "others": len(others), "others_found": len(touched(inside, others, tol)),
            "in_range": len(inside), "false": false, "away": sum(1 for a, b in spans if not meets(a, b, moments, tol)),
            "windows": result.windows, "seconds": result.seconds, "wall_s": result.wall_s, "stopped": result.stopped,
            "warnings": list(result.warnings), "sightings": [dict(m.raw) for m in result.sightings],
            "candidates": [dict(m.raw) for m in result.candidates]}


def run_retrieval(run: RunFolder, config: Config, session: Session, questions: Sequence[Question], warm_s: float | None = None,
                  query: Callable[[Question], RetrievalQuery] = query_of) -> int:
    """The retrieval tier for one configuration: each of `questions` that has a retrieval run asked of `session` (opened for
    `config`), its record written to results/retrieval/<Config.dir>/<qid>.json, and the number asked returned. A question with a
    record is not asked again. `warm_s`, the seconds the session took to open, loaded and warm, is kept as warm.json once. `query`
    says what to ask for a question (a test asks the words its fake model answers to)."""
    folder = run.folder / "results" / FOLDER / config.dir
    todo = [q for q in questions if q.retrieval is not None and not (folder / f"{q.qid}.json").exists()]
    if not todo:
        return 0
    if warm_s is not None and not (folder / "warm.json").exists():
        fs.write_json(folder / "warm.json", {"config": config.name, "seconds": warm_s})
    for q in todo:
        asked = query(q)
        record = retrieval_record(q, config, asked, session.oracle(asked))
        fs.write_json(folder / f"{q.qid}.json", record)
        print(f"{config.name} {q.qid}: found={record['found']} false={record['false']} {record['windows']} windows in "
              f"{record['seconds']:g} s", flush=True)
    return len(todo)


def outcome(r: dict) -> str:
    """How a retrieval record scored its needle: `found` (a sighting); `near` (not found, but among the top 5 candidates or with
    p of NEAR_P or more: the scan reached it); `stopped` (the ask met its limit of sightings, all false or of earlier needles,
    before it reached the needle: a `last` ask stops after LAST_LIMIT sightings, newest first, before it reaches the needle);
    `cut` (stopped at `max_s`, so the needle may not have been reached); else `missed`, a scan that went over it and saw
    nothing."""
    if r["found"]:
        return "found"
    if r["top5"] or (r.get("p") or 0) >= NEAR_P:
        return "near"
    if r.get("stopped") == "limit" and r.get("false"):
        return "stopped"
    return "cut" if r.get("stopped") == "max_s" else "missed"


def verdicts(rs: list[dict]) -> dict[str, str]:
    """For one needle question and the records of each configuration asked it: the verdict, by `config`, on each that did not
    find it.
    - A near miss is below its model's hit (calibration).
    - A scan that ended before the needle (`stopped`, `cut`) says nothing of the model's eyes or the haystack, and is told as it
      is; it does not count as a family that ran.
    - A miss is the model's when any model found the needle, or a model of another family had it in its top 5.
    - Only when nobody found it and models of two or more families scanned it and none had it in the top 5 is the haystack
      suspect. With one family this tier cannot tell (the agent arm's look can)."""
    fair = [r for r in rs if outcome(r) in ("found", "near", "missed")]
    families = {family(r["config"]) for r in fair}
    found = any(outcome(r) == "found" for r in rs)
    out: dict[str, str] = {}
    for r in rs:
        got = outcome(r)
        if got == "found":
            continue
        if got in ("near", "stopped", "cut"):
            out[r["config"]] = {"near": CALIBRATION, "stopped": STOPPED, "cut": CUT}[got]
        elif found or any(outcome(x) == "near" for x in rs if family(x["config"]) != family(r["config"])):
            out[r["config"]] = MISS
        elif len(families) >= 2:
            out[r["config"]] = SUSPECT
        else:
            out[r["config"]] = ONE_MODEL if len(fair) == 1 else ONE_FAMILY
    return out


def decider_figures(runs: Sequence[RunFolder]) -> dict:
    """The tier's figures: per configuration and type, from the records of all runs, and per template who found it, nearly missed
    it or missed it, with the verdicts."""
    records: dict[str, list[dict]] = {}
    firsts: dict[str, list[float]] = {}
    for run in runs:
        for folder in sorted((run.folder / "results" / FOLDER).glob("*")):
            for file in sorted(folder.glob("*.json")):
                r = fs.read_json(file)
                if file.stem == "warm":
                    firsts.setdefault(r["config"], []).append(r["seconds"])
                    continue
                records.setdefault(r["config"], []).append({**r, "run": run.name})

    def near(r: dict) -> bool:
        return bool(r["found"]) or (r.get("p") or 0) >= NEAR_P

    def recall(of: list[dict]) -> dict:
        needles = [r for r in of if not r["absent"]]
        return {"asked": len(of), "needles": len(needles), "found": sum(bool(r["found"]) for r in needles),
                "p50": sum(near(r) for r in needles), "top5": sum(bool(r["top5"]) for r in needles),
                "false": sum(r["false"] for r in of),
                "s_per_window": median([r["seconds"] / r["windows"] for r in of if r["windows"]])}

    models = {}
    for name in sorted(records):
        types = {}
        for type_ in TYPES:
            of = [r for r in records[name] if r["type"] == type_]
            if of:
                absent = [r for r in of if r["absent"]]
                types[type_] = {**recall(of), "absent": len(absent), "absent_clean": sum(1 for r in absent if r["false"] == 0)}
        models[name] = {"family": family(name), "first_s": median(firsts.get(name, [])), "types": types,
                        "media": recall([r for r in records[name] if r["type"] in MEDIA])}
    asked: dict[tuple, list[dict]] = {}  # (run, qid) -> the records of each configuration asked that needle question
    for rs in records.values():
        for r in rs:
            if not r["absent"]:
                asked.setdefault((r["run"], r["qid"]), []).append(r)
    templates: dict[str, dict] = {}
    for _, rs in sorted(asked.items()):
        t = templates.setdefault(rs[0]["template"], {"questions": 0, "models": {}, "verdicts": {}})
        t["questions"] += 1
        said = verdicts(rs)
        for r in rs:
            m = t["models"].setdefault(r["config"], {"asked": 0, **{o: 0 for o in OUTCOMES}, "ps": []})
            m["asked"] += 1
            m[outcome(r)] += 1
            if r.get("p") is not None:
                m["ps"].append(r["p"])
            if r["config"] in said:
                v = t["verdicts"].setdefault(said[r["config"]], {"models": [], "levels": []})
                v["models"] = sorted({*v["models"], r["config"]})
                v["levels"] = sorted({*v["levels"], r["level"]}, key=lambda x: LEVELS.index(x) if x in LEVELS else len(LEVELS))
    for t in templates.values():
        for m in t["models"].values():
            m["p"] = median(m.pop("ps"))
        t["models"] = {n: t["models"][n] for n in sorted(t["models"])}
    found = {r["template"] for rs in records.values() for r in rs if r["found"]}
    return {"models": models, "templates": dict(sorted(templates.items())), "missed": sorted(set(templates) - found),
            "asks": sum(len(rs) for rs in records.values())}


def family(config: str) -> str:
    """A configuration's family, from its name: the variant's name up to its first `-`, `_` or `.` (`d1` for mm/d1-3B and
    mm/d1-Omni-2B, then d3, pplx and h2o), or the system's name for a configuration with no variant. Variants of one family are
    trained alike."""
    return re.split(r"[-_.]", variant_of(config), maxsplit=1)[0].lower()


def variant_of(config: str) -> str:
    """What follows the system in a configuration's name (`d1-3B` of `mm/d1-3B`), or the name itself when it has no variant."""
    return config.partition("/")[2] or config
