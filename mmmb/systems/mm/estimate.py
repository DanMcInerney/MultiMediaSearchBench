"""What a range costs the decision model (moved from `window_s`, `arm_jobs`, `decider_minutes`, `minutes_text`, `HEAVY_S`, `SLOW_S`, `ASK_SHARE` and `gpu_estimate` of bench.py, `seconds_in`,
`measure_decode` and `EgoRun.estimate` of egolife.py, and `windows_in` of haystack.py, of this repository at bb498a6): the windows the reader will
cut a source's files into in a range, times what a window costs the model. The old counts read the scenario's plan, which a system may not see, so they
are counted from the dump when it is ingested (`table_of`, kept as store/windows.json) and from nothing else when a range is asked about (`windows_in`).

A file's windows, by what the dump says of it:
- a video with a `length` (a camera's clip: one frame in each second it plays): the seconds of its overlap with the range, rounded up, as
  Multimem's own index estimate counts a video;
- a video without one (a screen records only what changes): the seconds of the clock that hold a frame in the range, from the frames the
  reader's own demux finds, as the reader takes the first frame of each second;
- a picture: one, at its time; a sound: its windows (`audio_s` every `audio_hop_s`), as many as start in the range, which counts the quiet
  ones the reader skips too;
- a note: one, from its time on (a version is current from its time: an ask reads all of them); a file with a name (a config) is a version:
  the ones in the range and the one current at its start;
- the lines of a log source: the characters of their JSON and 26 for each, over the size of a window of records."""

from __future__ import annotations

import json
import math
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import av
from multimem import media, reader
from multimem.decider import About
from multimem.scripts import ask as ask_script

from mmmb.core.contract import Dump, ManifestEntry
from mmmb.core.question import Question
from mmmb.systems.mm.load import log_lines
from mmmb.systems.mm.models import Model

SLOW_S = 0.15  # a model that takes more seconds a window runs one exec at a time: parallel execs share its GPU queue
HEAVY_S = 300.0  # an mm exec that may need more GPU seconds than this runs alone on the GPU
ASK_SHARE = 0.75  # the share of an exec's time its ask can have: the agent's own turns take the rest, so an ask estimated past it cannot finish
LARGE_PX = 1280 * 720  # pictures of more pixels take the reader longer to decode than mm assumes (`ask.DECODE_S`): measured at ingest
VERSION = 1  # of windows.json
COST_KINDS = ("video", "audio", "text", "records")  # what window_s prices; a picture costs as a video window does


def window_s(about: About, kind: str = "video") -> float:
    """What mm estimates one window of `kind` costs this model at the reader's default profile (`ask.window_seconds`)."""
    return ask_script.window_seconds(kind, reader.profile_of(None, about), about)


def arm_jobs(model: Model | None, jobs: int) -> int:
    """The execs at once in an arm: one for a model that takes over SLOW_S a window (parallel execs share one GPU queue, so
    its timeouts would be the queue's fault), else `jobs`."""
    return 1 if model is not None and window_s(model.about) > SLOW_S else jobs


def decode_of(root: Path, entries: Sequence[ManifestEntry]) -> float:
    """Seconds one window of `entries` (video files under `root`, oldest first) takes the reader to decode at its default profile, over
    three of them (the first, a middle one and the last)."""
    store = types.SimpleNamespace(path_of=lambda item: item.path)
    spent, windows = 0.0, 0
    for e in (entries[0], entries[len(entries) // 2], entries[-1]):
        item = types.SimpleNamespace(kind="video", source=e.source, t0=e.t, path=root / e.path)
        began = time.perf_counter()
        windows += sum(1 for _ in reader._video(store, item, dict(reader.PROFILE), None, None, False))
        spent += time.perf_counter() - began
    return spent / windows


def picture_px(path: Path) -> int:
    """Pixels in a frame of the video at `path`."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return stream.codec_context.width * stream.codec_context.height


def audio_seconds(path: Path) -> float:
    """How long the sound at `path` plays, as Multimem's index counts it (from its first audio stream's packets)."""
    length = media.duration(path, sound=True)
    if length is None:
        raise ValueError(f"{path} has no sound to count windows of")
    return round(length, 3)


def audio_windows(length: float) -> list[tuple[float, float]]:
    """The windows the reader cuts a recording into: `audio_s` every `audio_hop_s`, the last one set back to the end."""
    size, hop = reader.PROFILE["audio_s"], reader.PROFILE["audio_hop_s"]
    starts = [k * hop for k in range(int(max(length - size, 0) // hop) + 1)]
    if starts[-1] + size < length - 1e-6:
        starts.append(length - size)
    return [(s, s + size) for s in starts]


def unit_of(root: Path, e: ManifestEntry) -> dict:
    """What a file of the dump counts as for the estimate (see the module's list). The reader's demux gives a video's frames."""
    if e.media == "image":
        return {"k": "image", "t": e.t}
    if e.media == "video":
        if e.length is not None:
            return {"k": "span", "t": e.t, "length": e.length}
        return {"k": "frames", "t": e.t, "frames": [round(sec, 3) for _, sec in reader._times(root / e.path)]}
    if e.media == "audio":
        return {"k": "audio", "t": e.t, "length": audio_seconds(root / e.path)}
    if e.media == "text":
        return {"k": "version" if e.name else "note", "t": e.t}
    raise ValueError(f"{e.path}: a {e.media} file is read as none of the media of the estimate")


def table_of(dump: Dump, decode=decode_of) -> dict:
    """The table of a dump (windows.json): for each source its files as `unit_of` counts them and its log lines as `[time, characters]`, the
    seconds one glasses-sized window takes to decode for each source whose pictures are larger than LARGE_PX (`decode(root, entries)`
    measures it; it needs the files), and the span of time the dump covers. Reads the files it counts; never a question."""
    sources: dict[str, list[dict]] = {}
    decode_s: dict[str, float] = {}
    for e in sorted((e for e in dump.entries if e.media != "records"), key=lambda e: (e.t, e.path)):
        sources.setdefault(e.source, []).append(unit_of(dump.root, e))
    for source in sources:
        videos = [e for e in dump.entries if e.source == source and e.media == "video"]
        if videos and picture_px(dump.root / videos[0].path) > LARGE_PX:
            decode_s[source] = round(decode(dump.root, sorted(videos, key=lambda e: (e.t, e.path))), 4)
    for log in dump.logs:
        lines = log_lines(dump, log)
        sources[log.name] = [{"k": "lines", "lines": [[when, len(json.dumps(line)) + 26] for when, line in lines]}]
    times = []
    for units in sources.values():
        for u in units:
            if "lines" in u:
                times += [when for when, _ in u["lines"]]
            else:
                times += [u["t"], u["t"] + (u["frames"][-1] if u.get("frames") else u.get("length", 0.0))]
    return {"version": VERSION, "span": [min(times), max(times) + 1], "decode_s": decode_s, "sources": sources}


def windows_in(table: Mapping, source: str, ranges: Sequence[tuple[float, float]]) -> dict[str, int]:
    """The windows of each cost kind (COST_KINDS) that the reader cuts `source` into over `ranges`: the ones that start in the range
    (see the module's list). A record window is counted over all the ranges together, as the size of the lines they hold. A source the
    dump does not have is an error."""
    if source not in table["sources"]:
        raise ValueError(f"a source is one of {', '.join(table['sources'])}, not {source!r}")
    counts, chars, versions = dict.fromkeys(COST_KINDS, 0), 0, []
    for unit in table["sources"][source]:
        kind = unit["k"]
        if kind == "version":
            versions.append(unit["t"])
        for since, until in ranges:
            if kind == "image":
                counts["video"] += since <= unit["t"] < until
            elif kind == "span":
                counts["video"] += math.ceil(max(0.0, min(unit["t"] + unit["length"], until) - max(unit["t"], since)))
            elif kind == "frames":
                if unit["t"] < until and unit["t"] + unit["frames"][-1] >= since:  # (a file outside the range has no frame in it)
                    counts["video"] += len({math.floor(unit["t"] + sec) for sec in unit["frames"] if since <= unit["t"] + sec < until})
            elif kind == "audio":
                counts["audio"] += sum(since <= unit["t"] + a < until for a, _ in audio_windows(unit["length"]))
            elif kind == "note":
                counts["text"] += unit["t"] < until
            elif kind == "lines":
                chars += sum(size for when, size in unit["lines"] if since <= when < until)
    for since, until in ranges:
        counts["text"] += sum(since <= t < until for t in versions) + (1 if any(t < since for t in versions) else 0)
    counts["records"] = math.ceil(chars / reader.PROFILE["records_chars"])
    return counts


def heavy(seconds: float) -> bool:
    """Whether an exec that may need `seconds` of the GPU holds it alone."""
    return seconds >= HEAVY_S


def beyond_timeout(seconds: float, timeout_s: float) -> bool:
    """Whether an ask that may need `seconds` of the GPU cannot finish inside an exec of `timeout_s`: that is the model's speed, not a miss."""
    return seconds > ASK_SHARE * timeout_s


def estimate_s(table: Mapping, about: About, sources: Sequence[str], ranges: Sequence[tuple[float, float]]) -> float:
    """The GPU seconds one ask over all of `sources` and `ranges` may need: their windows times what a window costs `about`. A window
    of video costs what the reader's decode really takes here when the source's pictures are large and it was measured."""
    total = 0.0
    for source in sources:
        counts = windows_in(table, source, ranges)
        each = {kind: window_s(about, kind) for kind in COST_KINDS}
        if source in table["decode_s"]:
            each["video"] = each["video"] - ask_script.DECODE_S + table["decode_s"][source]
        total += sum(counts[kind] * each[kind] for kind in COST_KINDS)
    return total


def decider_minutes(table: Mapping, about: About, questions: Sequence[Question]) -> float:
    """Minutes the retrieval asks of `questions` take a model: the windows of each ask's sources and ranges at the model's seconds a window."""
    return sum(estimate_s(table, about, q.retrieval.sources, q.retrieval.ranges) for q in questions) / 60


def minutes_text(minutes: float) -> str:
    return "under a minute" if minutes < 1 else f"about {minutes:.0f} min"
