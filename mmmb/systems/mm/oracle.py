"""The retrieval run of mm: the questions asked of the decision model through the daemon, with no agent (moved from `ask`, `warm`,
`first_photo`, `oracle_params` and the asking half of `oracle` of bench.py, and of `oracle_record` of egolife.py, of this repository at
bb498a6). The scoring of what comes back and the records are the core's (core/retrieval.py); here a `RetrievalQuery` becomes asks over
the daemon's API and the answers become `Moment`s."""

from __future__ import annotations

import time

from multimem import config

from mmmb.core.contract import Dump, Moment, RetrievalQuery, RetrievalResult

ORACLE_PROJECT, WARM_PROJECT = "id:bench-oracle", "id:bench-warm"  # so the labels of these asks are in no default scope


def ask(api, params: dict, project: str) -> tuple[dict, float]:
    """One `ask` as a job, collected until it ends: its result and the wall seconds."""
    began = time.monotonic()
    job = api.call("jobs/start", script="ask", params=params, wait_s=50, caller={"project": project})
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    if job["state"] != "done":
        raise RuntimeError(f"the ask {params['question']!r} {job['state']}: {job['error']}")
    return job["result"], round(time.monotonic() - began, 2)


def warm(api, params: dict) -> float:
    """Seconds of the one-window ask `Run.warm_params` gives, so the model is loaded before the execs; the load is in the number."""
    _, seconds = ask(api, params, WARM_PROJECT)
    return seconds


def warm_params(dump: Dump) -> dict:
    """The ask that loads a model before its execs: one window of the dump's earliest picture, else of its earliest video (the
    earliest, so that no earlier one is read)."""
    for media in ("image", "video"):
        found = [e for e in dump.entries if e.media == media]
        if found:
            first = min(found, key=lambda e: (e.t, e.path))
            return {"question": "Is anything in view?", "sources": [first.source], "since": first.t, "until": first.t + 1, "limit": 1,
                    "candidates": 0}
    raise ValueError(f"{dump.scenario}'s dump has no picture and no video to load the model on")


def ask_params(query: RetrievalQuery, since: float, until: float) -> dict:
    """What an ask over one range of `query` passes: its words, sources, limit, candidates and seconds; `oldest_first` only when the
    query has it (a `last` question's newest-first scan is its own)."""
    return {"question": query.text, "sources": list(query.sources), "since": since, "until": until, "limit": query.k,
            "candidates": query.candidates, "max_s": query.max_s, **({"oldest_first": True} if query.oldest_first else {})}


def moment_of(found: dict) -> Moment:
    """A sighting or candidate of an ask result: `t` when it is one instant, else `t0` and `t1`."""
    return Moment(config.parse_when(found.get("t") or found["t0"]), config.parse_when(found.get("t") or found["t1"]), found["p"],
                  found["source"], found)


def retrieve(api, query: RetrievalQuery) -> RetrievalResult:
    """`query` asked as one ask over each of its ranges, each collected until it ends: the sightings and candidates of all of them,
    their windows and GPU seconds added, the wall seconds of the asks, how they stopped (`max_s` if any did, else the first that
    gave a reason) and their warnings."""
    results = [ask(api, ask_params(query, a, b), ORACLE_PROJECT) for a, b in query.ranges]
    stops = [r.get("stopped") for r, _ in results]
    return RetrievalResult(
        sightings=tuple(moment_of(s) for r, _ in results for s in r["sightings"]),
        candidates=tuple(moment_of(c) for r, _ in results for c in r["candidates"]),
        windows=sum(r["progress"]["windows"] for r, _ in results), seconds=round(sum(r["progress"]["seconds"] for r, _ in results), 2),
        wall_s=round(sum(w for _, w in results), 2), stopped="max_s" if "max_s" in stops else next((s for s in stops if s), None),
        warnings=tuple(sorted({w for r, _ in results for w in r.get("warnings", [])})))
