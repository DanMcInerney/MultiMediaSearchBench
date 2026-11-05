"""What an exec leaves running in the daemon, and the waits that follow (moved from `open_jobs`, `cancel_open`, `settle`, `wait_idle`
and `quiesce` of bench.py of this repository at bb498a6). The calls are the daemon's API (`jobs/get`, `jobs/cancel`, `status`)."""

from __future__ import annotations

import time
from collections.abc import Sequence

from mmmb.core.agents.common import result_json

JOB_WAIT_S = 60.0  # in all, an exec that ended waits this long for its ask jobs (and the daemon's) to end
OTHERS_WAIT_S = 10.0  # ... but only this long when other execs are running, as it cannot tell their jobs from its orphans


def open_jobs(tools: Sequence[dict]) -> list[str]:
    """The ask jobs that an exec's calls started and left running or queued (by the last result seen of each)."""
    last: dict[str, str] = {}
    for t in tools:
        got = result_json(t.get("result")) if t.get("tool") == "ask" else None
        args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
        if got is not None and (job := got.get("job") or args.get("job")):
            last[str(job)] = str(got.get("state"))
    return [job for job, state in last.items() if state in ("queued", "running")]


def cancel_open(api, tools: Sequence[dict]) -> list[str]:
    """Cancel the ask jobs an exec left running, so that none goes on using the GPU or labels after the exec has ended (a cancelled
    ask still labels what it found, which `scrub_home` then takes out). A job already gone is passed over. The jobs it cancels."""
    out = []
    for job in open_jobs(tools):
        try:
            api.call("jobs/cancel", job=job)
        except Exception:  # noqa: BLE001 - the daemon says it is gone or ended: nothing to cancel
            continue
        out.append(job)
    return out


def settle(api, jobs: Sequence[str], limit_s: float | None = None) -> list[str]:
    """Wait for each of `jobs` to end (`jobs/get`), `limit_s` in all; the ids that had not ended by then. A job the daemon does
    not know (it ended long ago, or the daemon is gone) counts as ended."""
    until = time.monotonic() + (JOB_WAIT_S if limit_s is None else limit_s)
    unended = []
    for job in jobs:
        while True:
            left = until - time.monotonic()
            try:
                got = api.call("jobs/get", job=job, wait_s=max(0.0, min(left, 10.0)))
            except Exception:  # noqa: BLE001
                break
            if got.get("state") in ("done", "failed", "cancelled"):
                break
            if left <= 0:
                unended.append(job)
                break
    return unended


def wait_idle(api, allowed: int, limit_s: float) -> int | None:
    """Wait until the daemon's jobs (running and queued, from `status`) are `allowed` or fewer, `limit_s` at most; how many it
    still had over that, 0 when it came down, None when the count cannot be read. A job whose id was never logged, because its exec
    timed out inside the call that started it, is only in this count."""
    until = time.monotonic() + limit_s
    while True:
        try:
            counts = api.call("status").get("jobs") or {}
        except Exception:  # noqa: BLE001
            return None
        if counts.get("running") is None:
            return None
        over = counts["running"] + (counts.get("queued") or 0) - allowed
        if over <= 0:
            return 0
        if time.monotonic() >= until:
            return over
        time.sleep(0.25)


def quiesce(api, tools: Sequence[dict], others: int = 0, limit_s: float | None = None) -> dict:
    """What an mm exec leaves running when it ends, dealt with before its labels are scrubbed and its template released: the jobs
    its log shows open are cancelled and waited for (an ask labels what it found when it stops, a cancel included, so a label may
    land after the cancel), then the daemon's job count is waited down to `others`, one for each exec still running (a job whose id
    the log never held is only in the count; with others running that is a guess, so the wait is OTHERS_WAIT_S). Returns the
    cancelled jobs, those that had not ended (`unended`), the jobs still over the count (`orphans`, None when unreadable) and the
    seconds waited."""
    limit_s = JOB_WAIT_S if limit_s is None else limit_s
    began = time.monotonic()
    cancelled = cancel_open(api, tools)
    unended = settle(api, cancelled, limit_s)
    limit = max(0.0, (limit_s if not others else min(limit_s, OTHERS_WAIT_S)) - (time.monotonic() - began))
    orphans = wait_idle(api, others, limit)
    return {"cancelled": cancelled, "unended": unended, "orphans": orphans, "settle_s": round(time.monotonic() - began, 1)}
