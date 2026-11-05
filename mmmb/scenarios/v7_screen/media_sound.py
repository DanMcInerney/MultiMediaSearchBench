"""The hallway microphone of the V7 week: `sound(spec)` is one sound and `mic_samples(spec)` a segment (bed, TV, sounds), float32 mono
at RATE; `write_mic` writes it as opus.

    mic segment: `{"length_s", "seed", "bed_dbfs", "tv": [[from, to], ..], "sounds": [{"kind", "at", "dur", "hz"?, "notes"?, "gain"?, "seed"}]}`

Sound is told by pitch as pictures are by shade: the alarm's pitch is in `ALARM_HZ`, the doorbell's notes are a pair of `DOORBELL_PAIRS`,
and every other sound's pitch is at least `PITCH_GAP` from both, which `mic_samples` checks for the sounds of its own segment. A bed of
-50 dBFS is under the reader's `quiet_rms` (-48 dBFS), so a window of a quiet room costs the reader its decoding and nothing more.
A doorbell has `notes` (a pair of DOORBELL_PAIRS, the first is the pitch to ask for) and `dur` CHIME_DUR; an alarm has `hz` (of ALARM_HZ)
and `dur` ALARM_DUR; a distractor sound names no `hz` (`pick_hz` takes its kind's own, by its `seed`) and may name a `gain` (1 by
default; a beeper or a timer at 2 is louder than the alarm and the chime, which a sort by loudness in the needle's band must not take
for the needle). Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from mmmb.scenarios.v7_screen.constants import RATE
from mmmb.scenarios.v7_screen.media_common import _finish, _rng, ffmpeg_exe, gap

PITCH_GAP = 0.08  # a sound's pitch is at least this share of the lower pitch away from a needle's
BED_DBFS = -50.0
ALARM_HZ = (3000, 3400)  # a house's smoke alarm beeps at a pitch of this range
DOORBELL_PAIRS = ((659.26, 523.25), (783.99, 659.26), (587.33, 392.00), (698.46, 523.25))  # a house's chime: first and second note
BEEPER_HZ = {"microwave": (2450.0, 3800.0), "oven": (2600.0, 4000.0), "watch": (2700.0, 4250.0), "truck": (2520.0, 3900.0)}
TIMER_HZ = (880.0, 940.0)  # a kitchen timer's, in the chime's band (400 to 1,000 Hz) but 8% or more from every note of a chime
DING_HZ = (987.77, 1174.66, 1567.98, 2093.0)  # a notification's, and a TV jingle's, pitches
RING_HZ = (440.0, 1250.0, 1900.0)  # a phone ring is two tones: this pitch and 480/440 of it (440 and 480 Hz)
ALARM_DUR, CHIME_DUR = 2.5, 1.8  # seconds: three beeps (0.5 s on, 0.5 s off), and a two-note chime
LOUD_GAIN = 2.0  # a beeper (oven, truck) or a timer at this gain has a higher level than the alarm (0.34) or any chime (0.25), whatever its pitch
MAX_PEAK = 0.98  # a sound with a `gain` may not peak above this: a clipped sound is not the pitch it was made at
ALARM_ON_S, ALARM_OFF_S = 0.5, 0.5


def _times(seconds: float) -> np.ndarray:
    return np.arange(int(round(seconds * RATE))) / RATE


def _ramp(n: int, edge_s: float = 0.01) -> np.ndarray:
    """An envelope of 1 with `edge_s` ramps in and out (so a tone does not click)."""
    edge = min(int(edge_s * RATE), n // 2)
    env = np.ones(n)
    if edge:
        env[:edge] = np.linspace(0, 1, edge)
        env[n - edge:] = np.linspace(1, 0, edge)
    return env


def _partials(hz: float, t: np.ndarray, amps: tuple[float, ...]) -> np.ndarray:
    """A tone of `hz` with partials (the k-th at k * hz with amplitude `amps[k - 1]`), the ones below the Nyquist rate only."""
    out = np.zeros(len(t))
    for k, amp in enumerate(amps, 1):
        if k * hz < RATE * 0.45:
            out += amp * np.sin(2 * np.pi * k * hz * t)
    return out


def _chime(notes, dur: float) -> np.ndarray:
    """A doorbell: its first note, then its second halfway through `dur`, each decaying in 0.35 s with two partials. The first is
    louder (amplitude 0.6, the second 0.45)."""
    out = np.zeros(int(round(dur * RATE)))
    for k, (hz, amp) in enumerate(zip(notes, (0.6, 0.45))):
        start = int(round(k * dur / 2 * RATE))
        t = _times(dur - k * dur / 2)
        piece = amp * np.exp(-t / 0.35) * _ramp(len(t), 0.004) * _partials(hz, t, (1.0, 0.3, 0.1))
        out[start:start + len(piece)] += piece[:len(out) - start]
    return out


def _beeps(hz: float, dur: float, on: float, off: float, amps: tuple[float, ...], per: int = 0, pause: float = 0.0) -> np.ndarray:
    """Beeps of `on` seconds `off` apart through `dur`, in groups of `per` with a `pause` after each (no groups when 0)."""
    out = np.zeros(int(round(dur * RATE)))
    at, count = 0.0, 0
    while at + on <= dur + 1e-9:
        t = _times(on)
        i = int(round(at * RATE))
        out[i:i + len(t)] += (_ramp(len(t)) * _partials(hz, t, amps))[:len(out) - i]
        at += on + off
        count += 1
        if per and count % per == 0:
            at += pause
    return out


def pick_hz(kind: str, seed: int) -> float:
    """A pitch for a sound that names none: from its kind's own (`BEEPER_HZ`, `TIMER_HZ`, `DING_HZ`, `RING_HZ`), which are all at least
    PITCH_GAP from every chime note and from the whole of ALARM_HZ."""
    pool = {**BEEPER_HZ, "timer": TIMER_HZ, "ding": DING_HZ, "ring": RING_HZ}[kind]
    return float(_rng(seed, f"hz-{kind}").choice(pool))


def sound(spec: dict) -> np.ndarray:
    """The samples (float32, mono, at RATE) of one sound of a microphone segment, `spec["dur"]` seconds long, times `spec["gain"]`
    (default 1: a louder or quieter copy of the same sound; a `ValueError` when the result would peak above MAX_PEAK, since the
    segment's file would clip it). The needles: `doorbell` (two notes, `notes`, default the first of DOORBELL_PAIRS; louder than 0.45)
    and `alarm` (three beeps of 0.5 s, 0.5 s apart, repeated every 4.5 s, at `hz`, default 3200, with a second partial; amplitude
    0.45). The distractors: `microwave`, `oven`, `watch` and `truck` (a pure beeper, its pitch from BEEPER_HZ unless `hz` says),
    `timer` (a kitchen timer, 0.4 s on and 0.3 s off, TIMER_HZ), `ding` (a notification, DING_HZ), `ring` (a phone ring, RING_HZ and
    480/440 of it), `kettle`, `slam`, `steps`, `vacuum` and `bark`. A beeper or a timer with a `gain` of 2 is louder than either
    needle, at its own pitch: a sort by loudness in the needle's band does not put the needle first."""
    out = _sound(spec)
    gain = float(spec.get("gain", 1.0))
    if gain == 1.0:
        return out
    if not gain > 0:
        raise ValueError(f"a sound's gain is above 0, not {gain:g}")
    out = (out * np.float32(gain)).astype(np.float32)
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > MAX_PEAK:
        raise ValueError(f"a {spec['kind']} at gain {gain:g} peaks at {peak:.2f}, over {MAX_PEAK}: it would clip")
    return out


def _sound(spec: dict) -> np.ndarray:
    kind, dur = spec["kind"], float(spec["dur"])
    seed = spec.get("seed", 0)
    rng = np.random.default_rng(int.from_bytes(f"{seed}-{kind}".encode(), "little") % (1 << 63))
    t = _times(dur)
    n = len(t)
    if kind == "doorbell":
        return _chime(spec.get("notes") or DOORBELL_PAIRS[0], dur).astype(np.float32)
    if kind == "alarm":
        return _beeps(float(spec.get("hz", 3200)), dur, ALARM_ON_S, ALARM_OFF_S, (0.45, 0.18), per=3, pause=1.5).astype(np.float32)
    if kind in BEEPER_HZ:
        hz = float(spec.get("hz") or pick_hz(kind, seed))
        on, off = {"microwave": (0.25, 0.2), "oven": (0.9, 0.5), "watch": (0.1, 0.15), "truck": (0.4, 0.4)}[kind]
        return _beeps(hz, dur, on, off, (0.35,)).astype(np.float32)
    if kind == "timer":
        return _beeps(float(spec.get("hz") or pick_hz("timer", seed)), dur, 0.4, 0.3, (0.35,)).astype(np.float32)
    if kind == "ding":
        hz = float(spec.get("hz") or pick_hz("ding", seed))
        return (0.4 * np.exp(-t / 0.18) * _ramp(n, 0.004) * _partials(hz, t, (1.0, 0.25))).astype(np.float32)
    if kind == "ring":
        hz = float(spec.get("hz") or pick_hz("ring", seed))
        burst = 0.5 + 0.5 * np.sin(2 * np.pi * 20 * t)
        gate = ((t % 1.2) < 0.4).astype(float) * _ramp(n, 0.01)
        return (0.25 * gate * burst * (np.sin(2 * np.pi * hz * t) + np.sin(2 * np.pi * hz * 480 / 440 * t))).astype(np.float32)
    if kind == "kettle":  # a whistle that rises from 2,000 to 2,400 Hz, over hiss
        phase = 2 * np.pi * np.cumsum(2000 + 400 * t / dur) / RATE
        return (0.3 * np.sin(phase) * np.minimum(1, np.minimum(t, dur - t) / 0.05) + 0.02 * rng.standard_normal(n)).astype(np.float32)
    if kind == "slam":
        return (rng.uniform(-1, 1, n) * 0.6 * np.exp(-t / 0.04)).astype(np.float32)
    if kind == "steps":  # a low thump every half second
        out = np.zeros(n)
        for k in range(int(dur / 0.5)):
            i = int(k * 0.5 * RATE)
            burst = np.convolve(rng.standard_normal(int(0.08 * RATE)), np.ones(12) / 12, "same") * np.exp(-_times(0.08) / 0.025)
            out[i:i + len(burst)] += 0.5 * burst[:n - i]
        return out.astype(np.float32)
    if kind == "vacuum":  # a roar of smoothed noise, with a hum at 120 Hz
        noise = np.convolve(rng.standard_normal(n), np.ones(3) / 3, "same")
        return (0.06 * noise * _ramp(n, 0.3) + 0.03 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    if kind == "bark":  # two or three barks of 0.18 s around 450 Hz, with many partials and noise
        out = np.zeros(n)
        for k in range(min(3, 2 + seed % 2)):
            i = int(k * 0.35 * RATE)
            bt = _times(0.18)
            f0 = 450 - 80 * bt / 0.18
            tone_ = sum(a * np.sin(2 * np.pi * np.cumsum(f0 * m) / RATE) for m, a in zip(range(1, 7), (1, 0.7, 0.5, 0.4, 0.3, 0.2)))
            part = 0.2 * np.exp(-bt / 0.08) * (tone_ + 0.5 * rng.standard_normal(len(bt))) * _ramp(len(bt), 0.005)
            out[i:i + len(bt)] += part[:max(0, n - i)]
        return out.astype(np.float32)
    raise ValueError(f"a sound is a doorbell, alarm, microwave, oven, watch, truck, timer, ding, ring, kettle, slam, steps, vacuum or bark, not {kind!r}")


def _tv(seconds: float, seed) -> np.ndarray:
    """A TV stretch: speech-like noise (about 4 kHz wide, a syllable every 0.25 s) at about -32 dBFS, and every 40 s or so a
    jingle of three notes from DING_HZ."""
    rng = np.random.default_rng(int.from_bytes(f"{seed}-tv".encode(), "little") % (1 << 63))
    n = int(round(seconds * RATE))
    half = rng.standard_normal(n // 2 + 2, dtype=np.float32)
    out = np.empty(n, np.float32)  # twice the length, a sample between each two: lowpassed to about 4 kHz
    out[0::2] = half[:(n + 1) // 2]
    out[1::2] = (half[:n // 2] + half[1:n // 2 + 1]) / 2
    k = np.arange(n // 40 + 1) / (RATE / 40)  # the syllables, as an envelope of 400 steps a second
    out *= np.repeat((0.045 * (0.35 + 0.65 * np.abs(np.sin(np.pi * 4 * k + 0.8 * np.sin(2 * np.pi * 0.3 * k))))).astype(np.float32), 40)[:n]
    at = rng.uniform(8, 25)
    while at + 1.5 < seconds:
        for j, hz in enumerate(rng.choice(DING_HZ, 3)):
            i, jt = int((at + 0.3 * j) * RATE), _times(0.3)
            out[i:i + len(jt)] += (0.12 * np.exp(-jt / 0.15) * _ramp(len(jt), 0.004) * _partials(float(hz), jt, (1.0, 0.2)))[:max(0, n - i)]
        at += rng.uniform(30, 50)
    return out


def check_pitches(sounds) -> None:
    """The sounds of one segment keep their pitches apart: a ding, a ring, a timer and a beeper are at least PITCH_GAP from the
    chime's notes, a ring's upper tone included, and from the alarm's `hz`, whichever the segment holds. (A `ValueError` names the
    pair.)"""
    needles = [(f"the doorbell's note {hz}", hz) for s in sounds if s["kind"] == "doorbell" for hz in (s.get("notes") or DOORBELL_PAIRS[0])]
    needles += [("the alarm's pitch", float(s.get("hz", 3200))) for s in sounds if s["kind"] == "alarm"]
    for s in sounds:
        if s["kind"] in (*BEEPER_HZ, "timer", "ding", "ring"):
            hz = float(s.get("hz") or pick_hz(s["kind"], s.get("seed", 0)))
            for what, needle in needles:
                if min(gap(hz * m, needle) for m in ((1, 480 / 440) if s["kind"] == "ring" else (1,))) < PITCH_GAP:
                    raise ValueError(f"a {s['kind']} at {hz:g} Hz (at {s['at']:g} s) is within {PITCH_GAP:.0%} of {what} ({needle:g} Hz)")


def mic_samples(spec: dict) -> np.ndarray:
    """A microphone segment: a bed of white noise at `bed_dbfs` (-50 by default), each `tv` stretch, then each of `sounds` at its
    `at`, cut where the segment ends. float32 in about -1..1, `length_s` long, the same for the same spec."""
    check_pitches(spec["sounds"])
    n = int(round(spec["length_s"] * RATE))
    rng = np.random.default_rng(int.from_bytes(f"{spec['seed']}-bed".encode(), "little") % (1 << 63))
    out = rng.standard_normal(n, dtype=np.float32)
    out *= np.float32(10 ** (spec.get("bed_dbfs", BED_DBFS) / 20))
    for a, b in spec.get("tv", ()):
        i = int(round(a * RATE))
        piece = _tv(min(b, spec["length_s"]) - a, f"{spec['seed']}-{a:g}")
        out[i:i + len(piece)] += piece[:n - i]
    for s in spec["sounds"]:
        i = int(round(s["at"] * RATE))
        piece = sound(s)
        out[i:i + len(piece)] += piece[:max(0, n - i)]
    return out


def write_mic(path: Path, spec: dict, work: Path) -> None:
    """A mic segment as opus at 24 kbit/s, its samples piped to ffmpeg as 16-bit PCM (never a WAV on disk). `compression_level 0` and
    60 ms frames make the encode about 6 times faster than ffmpeg's defaults, at the size and the sound the reader needs."""
    samples = mic_samples(spec)
    np.clip(samples, -1, 1, out=samples)
    samples *= 32767
    Path(work).mkdir(parents=True, exist_ok=True)
    part = Path(work) / Path(path).name
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
                           "-b:a", "24k", "-compression_level", "0", "-frame_duration", "60", "-fflags", "+bitexact", str(part)],
                          input=samples.astype("<i2").tobytes(), capture_output=True)
    if done.returncode:
        raise RuntimeError(f"ffmpeg could not encode {Path(path).name}: {done.stderr.decode(errors='replace').strip()[-400:]}")
    _finish(part, Path(path))
