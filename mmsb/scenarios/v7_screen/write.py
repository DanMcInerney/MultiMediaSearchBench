"""Writing the week: every file of a plan made and checked as it is made, the logs written, and the mtimes set to each file's time.

    write(plan, folder, workers)    every file and log of the plan under `folder`; the plan's own checks are asserted

`write` asserts the plan's own checks: each needle is what the fake decider sees after encoding, no distractor is, each needle's code is in
its own file only, and no two recordings are the same bytes (which `remember` would keep once). Nothing of the truth is written. Moved from
haystack.py of the repository at bb498a6; the checks that read a file back use `frames`, the copy of the reader's decoding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from mmsb.scenarios.v7_screen.constants import PITCH_RMS, PITCH_SHARE, PROFILE, RATE, pitch_share, shade_share
from mmsb.scenarios.v7_screen.frames import audio_span, decode_frames, fit, frame_times
from mmsb.scenarios.v7_screen.media_common import NEEDLE_RGB, SOURCE_NEEDLES
from mmsb.scenarios.v7_screen.media_porch import write_porch
from mmsb.scenarios.v7_screen.media_screen import DIALOG, SCREEN, ScreenFrame, screen_frame, screen_frames, screen_layers, write_screen
from mmsb.scenarios.v7_screen.media_sound import write_mic
from mmsb.scenarios.v7_screen.pictures import clip_card, photo_image, photo_needle, portrait_video
from mmsb.scenarios.v7_screen.plan import File, Plan

WORKERS = 3  # files written at once: ffmpeg and x264 run outside the interpreter for much of each

NEED_SHARE, OTHER_SHARE, MOST_SHARE = 0.03, 0.005, 0.5  # a needle's share of its shade after encoding; any other picture's; and no more than


def audio_windows(length: float) -> list[tuple[float, float]]:
    """The windows the reader cuts a recording into: `audio_s` every `audio_hop_s`, the last one set back to the end."""
    size, hop = PROFILE["audio_s"], PROFILE["audio_hop_s"]
    starts = [k * hop for k in range(int(max(length - size, 0) // hop) + 1)]
    if starts[-1] + size < length - 1e-6:
        starts.append(length - size)
    return [(s, s + size) for s in starts]


def check_shades(image: Image.Image, source: str, present: str | None, where: str) -> None:
    """The picture (as a decider reads it) has each needle shade of `source` in the share the fake decider needs: the one that is
    `present` NEED_SHARE or more (and under MOST_SHARE: no needle is its frame's dominant colour), no other OTHER_SHARE."""
    for kind in SOURCE_NEEDLES[source]:
        share = shade_share(image, NEEDLE_RGB[kind])
        if kind == present:
            if not NEED_SHARE <= share < MOST_SHARE:
                raise AssertionError(f"{where}: {share:.1%} of the {kind} shade, not {NEED_SHARE:.0%} to {MOST_SHARE:.0%}")
        elif share >= OTHER_SHARE:
            raise AssertionError(f"{where}: {share:.2%} of the {kind} shade, which only a needle has")


def decoded(path: Path, wanted: set[int] | None = None) -> dict[int, Image.Image]:
    """The frames of a video as the reader hands them to a decider (scaled to its `max_px`), by index (frame k is the k-th of the file); the
    `wanted` ones only."""
    times = frame_times(path)
    pick = [(pts, k) for k, (pts, _) in enumerate(times) if wanted is None or k in wanted]
    return {k: image for k, image in decode_frames(path, pick, PROFILE["max_px"])}


def check_screen(f: File, path: Path) -> None:
    """The file has the frames `screen_frames` lists, at their seconds; in the first frame of every state with a dialog or an accent and
    of every 4th, and in all of an error's (the cursor wherever it went), the needle's shade is where the state says; and a frame with
    the cursor on the header of an error dialog still has it."""
    states, frames = f.spec["states"], screen_frames(f.spec, f.t)
    times = frame_times(path)
    if len(times) != len(frames):
        raise AssertionError(f"{f.path}: {len(times)} frames, {len(frames)} in its spec ({len(states)} states)")
    if any(not math.isclose(seconds, frame.second, abs_tol=1e-3) for (_, seconds), frame in zip(times, frames)):
        raise AssertionError(f"{f.path}: its frames are not at the seconds of its spec")
    firsts = {frame.state: i for i, frame in reversed(list(enumerate(frames)))}
    pick = set()
    for i, frame in enumerate(frames):
        s = states[frame.state]
        if (s["overlay"] or {}).get("kind") == "error" or (firsts[frame.state] == i and (s["overlay"] or s["accent"] or frame.state % 4 == 0)):
            pick.add(i)
    for i, image in decoded(path, pick).items():
        s = states[frames[i].state]
        check_shades(image, "screen", "error" if (s["overlay"] or {}).get("kind") == "error" else None,
                     f"{f.path}: the frame at {frames[i].second} s, in the state at {s['from']} s")
    for s in states:
        if (s["overlay"] or {}).get("kind") == "error":  # the cursor on the dialog's header, as a frame at the size a decider reads
            w, h = SCREEN
            cx, cy = s["overlay"].get("at") or (0.5, 0.5)
            on_header = ScreenFrame(0, 0, (round(w * cx), round(h * (cy - DIALOG[1] / 2 + 0.05))), "00:00", "2026-01-01")
            image = screen_frame(screen_layers(s), on_header).resize(fit(w, h, PROFILE["max_px"]), Image.Resampling.BICUBIC)
            check_shades(image, "screen", "error", f"{f.path}: the error with the cursor on its header")
            break


def check_porch(f: File, path: Path) -> None:
    length, parcel = int(f.spec["length_s"]), f.spec["parcel"]
    if len(frame_times(path)) != length:
        raise AssertionError(f"{f.path}: {len(frame_times(path))} frames, a clip of {length} s")
    pick = set(range(length)) if parcel or any(e["kind"] in ("courier", "owner_pickup") for e in f.spec["events"]) else set(range(0, length, 4))
    for k, image in decoded(path, pick).items():
        check_shades(image, "porch", "parcel" if parcel and parcel[0] <= k < parcel[1] else None, f"{f.path}: second {k}")


def check_sounds(f: File, samples: np.ndarray) -> None:
    """Over the reader's windows of a file with a needle (decoded from it), the needle is heard in a window that overlaps it (the fake
    decider's `pitch:<hz>`: the doorbell's first note, the alarm's pitch), and no window that overlaps none of its kind hears the
    pitch (of either note of the chime)."""
    needles: dict[tuple[str, tuple[float, ...]], list[tuple[float, float]]] = {}
    for s in f.spec["sounds"]:
        if s["kind"] in ("doorbell", "alarm"):
            pitches = tuple(s["notes"]) if s["kind"] == "doorbell" else (float(s["hz"]),)
            needles.setdefault((s["kind"], pitches), []).append((s["at"], s["at"] + s["dur"]))
    heard = {(key, span): False for key, spans in needles.items() for span in spans}
    for a, b in audio_windows(f.spec["length_s"]):
        piece = samples[int(round(a * RATE)):int(round(b * RATE))]
        if float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) < PITCH_RMS:
            continue
        for key, spans in needles.items():
            kind, pitches = key
            overlapping = [span for span in spans if a < span[1] and b > span[0]]
            if not overlapping:
                for hz in pitches:
                    if pitch_share(piece, RATE, hz) >= PITCH_SHARE:
                        raise AssertionError(f"{f.path}: the window {a:g} to {b:g} s sounds like {hz:g} Hz and overlaps no such {kind}")
            elif pitch_share(piece, RATE, pitches[0]) >= PITCH_SHARE:
                for span in overlapping:
                    heard[key, span] = True
    for (key, span), yes in heard.items():
        if not yes:
            raise AssertionError(f"{f.path}: the {key[0]} at {span[0]:g} s ({key[1][0]:g} Hz) is heard in no window")


def exif_of(t: float) -> Image.Exif:
    """The EXIF of a phone's photo taken at `t` (local epoch): DateTimeOriginal as the wall clock then, and OffsetTimeOriginal, the local
    offset at that moment (`+01:00`), as a phone writes them. Both are what `mm remember` reads a picture's time from."""
    when = datetime.fromtimestamp(t)
    zone = when.astimezone().strftime("%z")
    exif = Image.Exif()
    inner = exif.get_ifd(ExifTags.IFD.Exif)
    inner[ExifTags.Base.DateTimeOriginal] = when.strftime("%Y:%m:%d %H:%M:%S")
    inner[ExifTags.Base.OffsetTimeOriginal] = f"{zone[:3]}:{zone[3:]}"
    return exif


def write_file(f: File, folder: Path) -> None:
    """One file of the week, checked as it is made; its mtime is its time. A photo says its time in its EXIF and a phone clip in its
    `creation_time` (UTC), as a phone's files do."""
    path = folder / f.path
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = f.spec
    stamped = datetime.fromtimestamp(f.t).strftime("%Y-%m-%d %H:%M:%S")
    with tempfile.TemporaryDirectory(prefix="haystack-") as work_dir:
        work = Path(work_dir)
        if f.kind == "photo":
            photo_image(spec).save(path, quality=90, exif=exif_of(f.t))
            with Image.open(path) as saved:  # as the reader hands it to a decider: the JPEG decoded at 1/2
                saved.draft("RGB", (640, 360))
                check_shades(saved.convert("RGB"), "phone", photo_needle(spec), f.path)
        elif f.kind == "clip":
            card = clip_card(spec["scene"], spec["seed"], stamped)
            check_shades(card, "phone", None, f.path)
            made = f"creation_time={datetime.fromtimestamp(f.t, timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            portrait_video(path, card, work, seconds=3, tags=("-metadata", made))
        elif f.kind == "screen":
            write_screen(path, spec, f.t, work)
            check_screen(f, path)
        elif f.kind == "porch":
            write_porch(path, spec, stamped, work)
            check_porch(f, path)
        elif f.kind == "mic":
            write_mic(path, spec, work)
            if any(s["kind"] in ("doorbell", "alarm") for s in spec["sounds"]):
                check_sounds(f, audio_span(path, 0, 1e9, RATE)[1])
        else:  # note, config
            path.write_bytes(spec["text"].encode("utf-8"))
    os.utime(path, (f.t, f.t))


def count(token: str, text: str) -> int:
    """How often `token` is in `text` with no digit beside it."""
    return len(re.findall(rf"(?<!\d){re.escape(token)}(?!\d)", text))


def check_text(plan: Plan, folder: Path) -> None:
    """Each needle's code or value is in its own file or lines and nowhere else, read back from what was written."""
    def read(*parts: str) -> str:
        return folder.joinpath(*parts).read_text(encoding="utf-8")

    notes = {f.path: read(f.path) for f in plan.files if f.kind == "note"}
    configs = {f.path: read(f.path) for f in plan.files if f.kind == "config"}
    log_text = "\n".join(read("logs", f"{service}.jsonl") for service in plan.logs)
    everything = "\n".join([*notes.values(), *configs.values(), log_text])
    for q in plan.questions:
        if q["template"] == "note-code":
            code = q["truth"]["value"]
            if count(code, everything) != 1 or sum(code in text for text in notes.values()) != 1:
                raise AssertionError(f"{q['qid']}: the code {code} is in {count(code, everything)} places, not in one note only")
        elif q["template"] == "absent-note" and re.search(r"wi-?fi|password", "\n".join(notes.values()), re.I):
            raise AssertionError(f"{q['qid']}: a note mentions a Wi-Fi password")
        elif q["template"] in ("log-count", "absent-log"):
            code = q["fake"]["question"].split(":", 1)[1]
            lines = sum(code in line for line in log_text.splitlines())
            if count(code[4:], everything) != lines:  # no other line, note or version has the code's digits as a number
                raise AssertionError(f"{q['qid']}: the digits of {code} are in {count(code[4:], everything)} places, the code in {lines} lines")
            others = lines - len(q["truth"]["moments"])  # the lines on other days
            if (lines if q["absent"] else not 1 <= others <= 3):
                raise AssertionError(f"{q['qid']}: {code} is on {lines} lines, {others} of them outside the slot")
    for value in (f.spec["retention_days"] for f in plan.files if f.kind == "config"):
        if sum(text.count(f"retention_days = {value}\n") for text in configs.values()) != 1:
            raise AssertionError(f"retention_days = {value} is not in exactly one version of app.toml")


def write(plan: Plan, folder: Path, workers: int = WORKERS) -> None:
    """Write every file and log of the plan under `folder`, set their mtimes, and assert the plan's own checks (each needle is
    what the fake decider sees after encoding, no distractor is, each needle's code is in its own file only, no two recordings are
    the same bytes, which `remember` would keep once: photos too, which are recordings now). Nothing of the truth is written. `workers`
    files are written at once."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(write_file, f, folder) for f in plan.files]:
            future.result()
    (folder / "logs").mkdir(exist_ok=True)
    for service, lines in plan.logs.items():
        path = folder / "logs" / f"{service}.jsonl"
        path.write_bytes("".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
        if lines:
            last = datetime.fromisoformat(lines[-1]["ts"]).timestamp()
            os.utime(path, (last, last))
    check_text(plan, folder)
    seen: dict[tuple[str, str], str] = {}
    for f in plan.files:
        if f.kind in ("photo", "clip", "screen", "porch", "mic"):
            key = (f.source, hashlib.sha256((folder / f.path).read_bytes()).hexdigest())
            if key in seen:
                raise AssertionError(f"{f.path} has the bytes of {seen[key]}: remember would keep one")
            seen[key] = f.path
