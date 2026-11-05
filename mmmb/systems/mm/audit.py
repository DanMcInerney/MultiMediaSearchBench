"""What an mm exec's calls show, and what the home shows of them (moved from `ambient_labels` and the ask half of `work_of` of bench.py,
and `tool_times`, `accepted_asks`, `audit`, `AUDIT_S` and `HorizonLeak` of egolife.py, of this repository at bb498a6): the ask jobs it ran,
the recordings past a horizon its results held (a leak of the harness, not an answer), and the labels the daemon holds that no project
owns."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from multimem import config

from mmmb.core.agents.common import result_json, result_jsons

AUDIT_S = 0.001  # a time in a tool result is past H when it is more than this after it (a result rounds to the millisecond)


class HorizonLeak(RuntimeError):
    """A tool result of the mm arm held a recording from after the question's horizon: the harness is wrong, not the answer."""


def work_of(tools: Sequence[dict]) -> dict:
    """The ask jobs an exec ran, from its calls (tools.json), counted by the last result seen of each (the `progress` it gave:
    `windows`, GPU `seconds`; how it `stopped`, `running` if it had not). A tool that gives no result, or one that is not JSON,
    counts as none. The core's `work_of` has the exec's views and shell commands, and these are merged over its zeros."""
    jobs: dict[str, dict] = {}
    for t in tools:
        args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
        if t.get("tool") == "ask":
            got = result_json(t.get("result"))
            if got is not None and (job := got.get("job") or args.get("job")):
                jobs[str(job)] = got
    progress = [(j.get("progress") if isinstance(j.get("progress"), dict) else {}) for j in jobs.values()]
    return {"jobs": len(jobs), "windows": sum(p.get("windows") or 0 for p in progress),
            "seconds": round(sum(p.get("seconds") or 0 for p in progress), 2),
            "stopped": dict(Counter(j.get("stopped") or "running" for j in jobs.values()))}


def tool_times(tool: dict, home: Path | None) -> list[float]:
    """The times of recordings in an mm tool's results (every JSON block of them: a `look` of several moments has one each): an
    ask's sightings and candidates, a find's rows (not its labels, which the agent made itself and mm dates by its own clock), the
    end of each item a look returned (from the home's index), and for a sheet the end of each item its tiles were cut from. A
    sheet's range and tile times are the times the agent asked for, as an ask's `until` is, so they are not counted."""
    name, times, items = tool.get("tool"), [], []
    when = lambda text: config.parse_when(text) if text else None  # noqa: E731
    for got in result_jsons(tool.get("result")):
        if name == "ask":
            times += [when(s.get(k)) for s in got.get("sightings") or [] for k in ("t", "t0", "t1")]
            times += [when(c.get(k)) for c in got.get("candidates") or [] for k in ("t", "t0", "t1")]
        elif name == "find":
            times += [when(r.get(k)) for r in got.get("results") or [] if r.get("type") != "label" for k in ("t0", "t1")]
        elif name == "look":
            for m in [m for m in got.get("moments") or [got] if isinstance(m, dict)]:
                items += [m["item"]] if isinstance(m.get("item"), int) else []
                for tile in [x for x in m.get("tiles") or [] if isinstance(x, dict)]:
                    items += [tile["look"]["item"]] if isinstance(tile.get("look"), dict) else []
    if items and home is not None:
        with closing(sqlite3.connect(f"file:{home / 'index.sqlite'}?mode=ro", uri=True, timeout=30)) as db:
            times += [row[0] for i in items for row in db.execute("SELECT t1 FROM items WHERE id = ?", (i,))]
    return [t for t in times if t is not None]


def accepted_asks(tools: Sequence[dict]) -> list[dict]:
    """The arguments of the asks mm took (their result holds a `job`): an ask with `question` or `questions`, not a call that
    collects or cancels a job, and not one mm refused."""
    return [t["arguments"] for t in tools if t.get("tool") == "ask" and isinstance(t.get("arguments"), dict)
            and ("question" in t["arguments"] or "questions" in t["arguments"])
            and any("job" in got for got in result_jsons(t.get("result")))]


def audit(tools: Sequence[dict], h: float, home: Path | None) -> dict:
    """What an mm exec's calls held past the horizon `h`: `after_horizon`, the distinct recording times after it in the results
    (any is a leak of the harness), and of the asks mm took (`asks`) those with an `until` past it (`asks_past_horizon`: the
    agent's confusion about mm's clock, which reads days later than the question, and harmless)."""
    late = {t for tool in tools for t in tool_times(tool, home) if t > h + AUDIT_S}
    starts = accepted_asks(tools)
    past = [a for a in starts if a.get("until") is None or config.parse_when(a["until"]) > h + AUDIT_S]
    return {"after_horizon": len(late), "asks": len(starts), "asks_past_horizon": len(past)}


def ambient_labels(api, since: float, until: float) -> list[dict]:
    """The labels `ask` made with no project (its labels carry the caller's project), between `since` and `until`. A pinned project
    makes none; any means an exec's ask had no project, so another question's default scope could hold its answer."""
    out: list[dict] = []
    after = None
    while True:
        page = api.call("find", types=["label"], scope="all", since=since, until=until, limit=200,
                        **({"after": after} if after else {}))
        out += [r for r in page["results"] if r.get("by") == "ask" and r.get("project") is None]
        after = page["next"]
        if after is None:
            return out
