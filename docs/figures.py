"""Draws the README's figures into docs/img.

    uv run --with matplotlib python docs/figures.py [--data D:/mmsb-data] [--only hero,piles]

The charts are drawn from the numbers below, which are copied from the published reports in results/ and from CATALOGUE.md.
The pictures of the piles (`hero`, `piles`) read the four built dumps (`mmsb build --scenario ... --scale standard`), so they
need the data folder; the others need nothing. What is shown of the data: thumbnails of Caltech Camera Traps photos
(CDLA-Permissive 1.0), frames of MEVA (CC BY 4.0), the loudness of the SINS day in 30-second steps and the outlines of FSD50K
clips. No sound of SINS or FSD50K can be rebuilt from them. Every example question is from the `dev` split.
"""

from __future__ import annotations

import argparse
import os
import textwrap
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).parent / "img"

BG, PANEL, PANEL2, LINE = "#0D1320", "#151D2E", "#1C2740", "#2A3857"
TEXT, MUTED, DIM = "#EAF0FA", "#9AA8C0", "#62708A"
AMBER, TEAL, VIOLET, CORAL, GREEN, BLUE = "#F6B445", "#3CC8B4", "#A192FF", "#FF7D6E", "#5BD38C", "#5AA9FF"
MEDIA = {"Video": AMBER, "Image": TEAL, "Audio": VIOLET, "Mixed": CORAL}

INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Segoe UI", "Inter", "Helvetica Neue", "Arial") if f in INSTALLED), "DejaVu Sans")
MONO = next((f for f in ("Cascadia Mono", "Consolas", "Menlo") if f in INSTALLED), "DejaVu Sans Mono")

# ---------------------------------------------------------------------------------------------------------------- the canvas


class Canvas:
    """A figure measured in pixels, y downwards, so that a layout is written as it is seen."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.axis("off")
        self.fig.patch.set_facecolor(BG)
        yy, xx = np.mgrid[0:40, 0:80]
        glow = np.zeros((40, 80, 4))
        glow[..., :3] = (0.24, 0.42, 0.85)
        glow[..., 3] = 0.16 * np.exp(-(((xx - 62) / 30) ** 2 + ((yy - 2) / 22) ** 2))
        self.ax.imshow(glow, extent=(0, w, h, 0), interpolation="bicubic", zorder=0, aspect="auto")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

    def text(self, x, y, s, size=24, color=TEXT, weight="normal", ha="left", va="top", font=None, zorder=5, **kw):
        return self.ax.text(x, y, s, fontsize=size * 0.72, color=color, fontweight=weight, ha=ha, va=va, fontfamily=font or FONT,
                            zorder=zorder, **kw)

    def width(self, t) -> float:
        return t.get_window_extent(self.fig.canvas.get_renderer()).width

    def box(self, x, y, w, h, fill=PANEL, edge=LINE, r=14, lw=1.2, alpha=None, zorder=2):
        r = min(r, w / 2, h / 2)
        self.ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", facecolor=fill,
                                         edgecolor=edge or "none", linewidth=lw if edge else 0, alpha=alpha, zorder=zorder))

    def rect(self, x, y, w, h, fill, zorder=3, alpha=1.0):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none", zorder=zorder, alpha=alpha))

    def frame(self, x, y, w, h, color=AMBER, lw=4, zorder=7):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=color, linewidth=lw, zorder=zorder))

    def image(self, im, x, y, w, h, zorder=3):
        self.ax.imshow(np.asarray(im), extent=(x, x + w, y + h, y), zorder=zorder, aspect="auto", interpolation="lanczos")

    def chip(self, x, y, s, color, size=20, pad=14, font=None):
        """A rounded label; returns its right edge."""
        t = self.text(x + pad, y + size * 0.36, s, size=size, color=color, weight="bold", font=font, zorder=6)
        width = self.width(t)
        self.box(x, y, width + 2 * pad, size * 1.9, fill=color + "22", edge=color + "88", r=size * 0.6, lw=1, zorder=4)
        return x + width + 2 * pad

    def arrow(self, x0, y0, x1, y1, color=MUTED, lw=2.4, zorder=4):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, color=color, linewidth=lw,
                                          zorder=zorder, shrinkA=0, shrinkB=0))

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{name}.png"
        self.fig.savefig(path, dpi=100, facecolor=BG)
        plt.close(self.fig)
        Image.open(path).convert("RGB").save(path, optimize=True)
        print(f"docs/img/{path.name}  {path.stat().st_size // 1024} KB")


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """The image cut to w:h about its middle and scaled to that size."""
    iw, ih = im.size
    if iw / ih > w / h:
        cut = int(ih * w / h)
        im = im.crop(((iw - cut) // 2, 0, (iw + cut) // 2, ih))
    else:
        cut = int(iw * h / w)
        im = im.crop((0, (ih - cut) // 2, iw, (ih + cut) // 2))
    return im.resize((w, h), Image.LANCZOS)


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width))


# ---------------------------------------------------------------------------------------------------------------- the piles

TRAILCAM = "mmsb-5b6356b8ca-data/DCIM/100RECNX"  # the standard dumps of seed 1, as `mmsb build` names them
SECURITY = "mmsb-74a64a7e31-data"
RECORDER = "mmsb-028a0ce1db-data/RECORD/REC_20170202_000000.wav"
SOUNDS = "mmsb-bf7133e280-data"
PHOTOS, DOG = 4256, 913  # the card's photos, and the first one of a dog (question first-dog, dev)
PEOPLE = {1: 55, 376: 432, 378: 324, 1994: 1940, 2003: 2048, 3717: 3755}  # photos of whoever tended the camera, among those the grids would show, and the empty one shown for each
TAPE = ("rec_00869.wav", "rec_01986.wav", "rec_03971.wav", "rec_04017.wav")  # the answer of find-packing_tape (dev)


def photo(data: Path, n: int, w: int, h: int) -> Image.Image:
    return fit(Image.open(data / TRAILCAM / f"IMG_{n:04d}.JPG").convert("RGB"), w, h)


def photo_grid(c: Canvas, data: Path, x, y, cols, rows, tw, th, gap, needle_at: int) -> tuple[float, float]:
    """Photos spread evenly over the card, with the dog's among them; returns the middle of the dog's."""
    count = cols * rows
    numbers = [1 + round(k * (PHOTOS - 1) / (count - 1)) for k in range(count)]
    numbers = [PEOPLE.get(n, n) for n in numbers]
    numbers[needle_at] = DOG
    spot = (0.0, 0.0)
    for k, n in enumerate(numbers):
        px, py = x + (k % cols) * (tw + gap), y + (k // cols) * (th + gap)
        c.image(photo(data, n, tw * 2, th * 2), px, py, tw, th)
        if n == DOG:
            c.frame(px - 3, py - 3, tw + 6, th + 6)
            spot = (px + tw / 2, py + th / 2)
        else:
            c.rect(px, py, tw, th, BG, zorder=4, alpha=0.35)
    return spot


def video_frame(path: Path, t: float) -> Image.Image:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int((t - 1) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time >= t:
                return frame.to_image()
    raise ValueError(f"no frame at {t} s in {path}")


def loudness(data: Path, step_s: int = 30) -> np.ndarray:
    """The day's loudness in dB, a value each `step_s` seconds."""
    with wave.open(str(data / RECORDER)) as w:
        rate, out = w.getframerate(), []
        while chunk := w.readframes(rate * step_s):
            out.append(float(np.frombuffer(chunk, dtype=np.int16).astype(np.float32).std()))  # about its mean: the recorder has a DC offset
    return 20 * np.log10(np.maximum(np.array(out), 1.0))


def outline(path: Path, bins: int) -> np.ndarray:
    """A clip as the peak of each of `bins` stretches, 0 to 1."""
    with wave.open(str(path)) as w:
        a = np.abs(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32))
    a = a[: len(a) // bins * bins].reshape(bins, -1).max(axis=1) if len(a) >= bins else np.zeros(bins)
    return a / max(float(a.max()), 1.0)


def qa(c: Canvas, x, y, w, tag, color, shelf, pile, question, answer, cite) -> None:
    """The words of a pile's panel: what it is, a question, its answer and where the answer is."""
    right = c.chip(x, y, tag, color, size=20, font=MONO)
    c.text(right + 16, y - 3, shelf, size=30, weight="bold")
    c.text(x, y + 54, pile, size=19, color=MUTED, linespacing=1.45)
    c.box(x, y + 128, w, 112, fill=PANEL2, edge=LINE, r=14)
    c.text(x + 20, y + 142, "ASKED", size=13, color=DIM, weight="bold")
    c.text(x + 20, y + 166, question, size=21, weight="bold", linespacing=1.35)
    c.box(x, y + 254, w, 90, fill="#12261F", edge=GREEN + "77", r=14)
    c.text(x + 20, y + 268, "THE ANSWER, AND WHERE IT IS", size=13, color=GREEN, weight="bold")
    c.text(x + 20, y + 296, answer, size=22, weight="bold", font=MONO)
    c.text(x + w - 20, y + 300, cite, size=16, color=MUTED, font=MONO, ha="right")


def piles(data: Path) -> None:
    W, ROW, TOP = 1600, 420, 130
    c = Canvas(W, TOP + 4 * ROW + 12)
    c.text(50, 30, "Four of the piles, as an agent finds them", size=40, weight="bold")
    c.text(50, 88, "Real media at its real size, the questions people ask of it, and answers taken from the source's own labels.", size=21, color=MUTED)
    TX, TW, VX, VW = 60, 500, 610, 930
    for k in range(4):
        c.box(30, TOP + k * ROW, W - 60, ROW - 18, fill=PANEL, edge=LINE, r=20)

    # the trail camera
    y = TOP + 28
    qa(c, TX, y, TW, "I3", TEAL, "Trail camera", "One camera's memory card: 4,256 photos over\nsix months, most of them empty.",
       "When did my trail camera first\nphotograph a dog?", "2013-10-17 16:43:35", "IMG_0913.JPG")
    tw, th, gap = 72, 54, 6
    spot = photo_grid(c, data, VX, y, 7, 5, tw, th, gap, needle_at=17)
    bx, bw = VX + 7 * (tw + gap) + 14, VW - 7 * (tw + gap) - 14
    bh = 5 * th + 4 * gap
    c.image(photo(data, DOG, bw * 2, bh * 2), bx, y, bw, bh, zorder=6)
    c.frame(bx, y, bw, bh)
    c.arrow(spot[0] + tw / 2 + 4, spot[1], bx - 4, spot[1], color=AMBER, lw=3, zorder=8)
    c.text(VX, y + 308, "35 of the 4,256 photos, evenly spread, and the one that answers.", size=17, color=MUTED)
    c.text(VX + VW, y + 310, "Photos: Caltech Camera Traps", size=14, color=DIM, ha="right")

    # the security cameras
    y = TOP + ROW + 28
    qa(c, TX, y, TW, "V1", AMBER, "Security cameras", "Three outdoor cameras of one site,\n75 minutes each, in six MP4 files.",
       "When did someone last close a car\ndoor on cam2?", "2018-03-12 11:01:53", "cam2, second file, 21:24")
    at_s = 1285.0  # seconds into each camera's second file; cam1's starts 2 s after the others'
    fw, fh = 302, 170
    frames = {}
    for i, (cam, name, t) in enumerate((("cam1", "cam1_20180312_104002-111502.mp4", at_s - 2), ("cam2", "cam2_20180312_104000-111500.mp4", at_s),
                                        ("cam3", "cam3_20180312_104000-111500.mp4", at_s))):
        frames[cam] = video_frame(data / SECURITY / cam / name, t)
        px = VX + i * (fw + 12)
        c.image(fit(frames[cam], fw * 2, fh * 2), px, y, fw, fh)
        c.box(px + 8, y + 8, 70, 28, fill="#000000", edge=None, r=6, alpha=0.6, zorder=6)
        c.text(px + 43, y + 11, cam, size=16, weight="bold", ha="center", font=MONO, zorder=7)
        if cam == "cam2":
            c.frame(px - 3, y - 3, fw + 6, fh + 6)
    ty, tx0, tx1 = y + 200, VX + 70, VX + 610  # the three cameras' recordings as bars, and the moment
    for i, cam in enumerate(("cam1", "cam2", "cam3")):
        c.text(VX, ty + i * 28 - 5, cam, size=16, color=MUTED, font=MONO)
        for a, b in ((0.0, 35 / 75), (40 / 75, 1.0)):
            c.box(tx0 + a * (tx1 - tx0), ty + i * 28, (b - a) * (tx1 - tx0), 14, fill=LINE, edge=None, r=5)
    at = tx0 + (61 + 53 / 60) / 75 * (tx1 - tx0)
    c.ax.plot([at, at], [ty - 8, ty + 80], color=AMBER, lw=3, zorder=6)
    c.ax.add_patch(Circle((at, ty + 35), 7, color=AMBER, zorder=7))
    for label, frac, ha in (("10:00", 0.0, "left"), ("10:35", 35 / 75, "right"), ("10:40", 40 / 75, "left"), ("11:15", 1.0, "right")):
        c.text(tx0 + frac * (tx1 - tx0), ty + 86, label, size=14, color=DIM, ha=ha, font=MONO)
    c.text(VX, y + 310, "3.5 hours of footage. The moment lasts two seconds.", size=17, color=MUTED)
    zw, zh = VW - 610 - 28, 150
    zoom = fit(frames["cam2"].crop((1480, 440, 1920, 720)), zw * 2, zh * 2)
    c.image(zoom, VX + VW - zw, ty - 12, zw, zh, zorder=8)
    c.frame(VX + VW - zw, ty - 12, zw, zh, zorder=9)
    c.text(VX + VW, y + 312, "Frames: MEVA", size=14, color=DIM, ha="right")

    # the room recorder
    y = TOP + 2 * ROW + 28
    qa(c, TX, y, TW, "A1", VIOLET, "Room recorder", "A voice recorder left running in a living\nroom for 24 hours: one WAV file.",
       "At what time yesterday, between 14:00\nand 15:30, did a visitor arrive?", "14:59:20", "14:58:20 to 15:00:20")
    db = loudness(data)
    lo, hi = np.percentile(db, 0.5), np.percentile(db, 99.7)
    level = np.clip((db - lo) / (hi - lo), 0.015, 1)
    ph, mid, amp = 268, y + 134, 118
    c.box(VX, y, VW, ph, fill=BG, edge=LINE, r=14)
    xs = np.linspace(VX + 6, VX + VW - 6, len(level))
    span = VW - 12
    c.ax.fill_between(xs, mid - level * amp, mid + level * amp, color=VIOLET, alpha=0.9, zorder=4, linewidth=0)
    a, b = VX + 6 + 14.0 / 24 * span, VX + 6 + 15.5 / 24 * span
    c.rect(a, y + 4, b - a, ph - 8, AMBER, zorder=3, alpha=0.2)
    at = VX + 6 + (14 + 59 / 60 + 20 / 3600) / 24 * span
    c.ax.plot([at, at], [y + 4, y + ph - 4], color=AMBER, lw=3, zorder=6)
    c.text(at + 12, y + 12, "a visitor arrives, 14:59:20", size=17, color=AMBER, weight="bold", zorder=7)
    c.text(a - 10, y + ph - 34, "the hours asked about", size=15, color=AMBER, ha="right", zorder=7)
    for hour in range(0, 25, 3):
        c.text(VX + 6 + hour / 24 * span, y + ph + 8, f"{hour:02d}:00", size=14, color=DIM, ha="center", font=MONO)
    c.text(VX, y + 310, "How loud the day was. Nothing in it says which sound is a visitor.", size=17, color=MUTED)
    c.text(VX + VW, y + 312, "Recording: SINS", size=14, color=DIM, ha="right")

    # the sound-effects library
    y = TOP + 3 * ROW + 28
    qa(c, TX, y, TW, "M4", VIOLET, "Sound-effects library", "About 6,100 short clips in numbered folders,\nunder names that say nothing.",
       "Find every clip of packing tape.", "4 clips", "rec_00869.wav and 3 more")
    cols, rows, gap = 12, 6, 6
    cw, ch = (VW - (cols - 1) * gap) / cols, 43
    clips = sorted((data / SOUNDS).glob("folder_*/*.wav"))
    wanted = [p for p in clips if p.name in TAPE]
    shown = [clips[round(k * (len(clips) - 1) / (cols * rows - 1))] for k in range(cols * rows)]
    for k, clip in zip((7, 29, 44, 62), wanted):
        shown[k] = clip
    for k, clip in enumerate(shown):
        px, py = VX + (k % cols) * (cw + gap), y + (k // cols) * (ch + gap)
        hit = clip in wanted
        c.box(px, py, cw, ch, fill="#3A2F18" if hit else BG, edge=AMBER if hit else LINE, r=6, lw=3 if hit else 1)
        peaks = outline(clip, 22)
        bx = px + 8 + np.arange(22) * (cw - 14) / 22
        c.ax.vlines(bx, py + ch / 2 - peaks * (ch / 2 - 6) - 1, py + ch / 2 + peaks * (ch / 2 - 6) + 1, color=AMBER if hit else VIOLET,
                    lw=1.8, alpha=1 if hit else 0.75, zorder=5)
    c.text(VX, y + 310, "72 of the 6,123 clips, as outlines. The four that answer are among them.", size=17, color=MUTED)
    c.text(VX + VW, y + 312, "Clips: FSD50K", size=14, color=DIM, ha="right")
    c.save("piles")


def hero(data: Path) -> None:
    W, H = 1600, 720
    c = Canvas(W, H)
    c.text(50, 40, "MMSB", size=20, color=AMBER, weight="bold")
    c.text(48, 66, "MultiMediaSearchBench", size=68, weight="bold")
    c.text(50, 172, "Can an AI agent find the one moment that matters in the video, photos and sound you keep?", size=27, color=MUTED)
    x = 50
    for label, color in (("23 piles of real media", TEAL), ("553 questions", AMBER), ("22% ask for things that never happened", CORAL),
                         ("graded by fixed rules, not by a model", GREEN)):
        x = c.chip(x, 232, label, color, size=19) + 14
    cols, rows, tw, th, gap = 16, 5, 88, 66, 6
    gx, gy = (W - cols * (tw + gap) + gap) / 2, 308
    spot = photo_grid(c, data, gx, gy, cols, rows, tw, th, gap, needle_at=42)
    c.box(110, 372, 650, 100, fill="#0A0F1A", edge=LINE, r=22, alpha=0.95, zorder=8)
    c.text(140, 388, "ASKED", size=13, color=DIM, weight="bold", zorder=9)
    c.text(140, 414, "When did my trail camera first photograph a dog?", size=24, weight="bold", zorder=9)
    c.box(870, 548, 610, 100, fill="#0C1F18", edge=GREEN, r=22, alpha=0.96, zorder=8)
    c.text(900, 564, "ANSWERED, WITH WHERE IT IS", size=13, color=GREEN, weight="bold", zorder=9)
    c.text(900, 592, "2013-10-17 16:43:35", size=25, weight="bold", font=MONO, zorder=9)
    c.text(1450, 598, "IMG_0913.JPG", size=18, color=MUTED, font=MONO, ha="right", zorder=9)
    c.arrow(spot[0], spot[1] + th / 2 + 5, spot[0], 546, color=AMBER, lw=3, zorder=9)
    c.text(W - 50, H - 14, "One trail camera's card: 80 of its 4,256 photos (Caltech Camera Traps)", size=14, color=DIM, ha="right", va="bottom")
    c.save("hero")


# ---------------------------------------------------------------------------------------------------------------- how it works


def pipeline() -> None:
    W, H = 1600, 650
    c = Canvas(W, H)
    c.text(50, 30, "How a question is run", size=40, weight="bold")
    c.text(50, 88, "A fresh agent session for each question and each configuration. No model grades an answer.", size=21, color=MUTED)
    steps = (("1", "A pile", "Media as people keep\nit: at its real size,\nunder the names a\ndevice gives it.", TEAL),
             ("2", "A question", "In a person's words.\nThe answer is known\nfrom the source's\nown labels.", AMBER),
             ("3", "An agent", "Claude Code or Codex,\nlocked to the pile,\nwith 15 minutes\nfor the question.", BLUE),
             ("4", "An answer", "A value, and where\nin the media it is.\n“It isn't there” is\nan answer too.", VIOLET),
             ("5", "A fixed rule", "Times to a tolerance,\nsets in full, counts\nexactly. The same\ngrade every time.", GREEN),
             ("6", "A report", "Right, grounded or\nlucky, with intervals.\nSeconds, tokens,\ndollars, GPU.", CORAL))
    bw, bh, gap, y0 = 232, 236, 21.6, 150
    for k, (n, title, body, color) in enumerate(steps):
        x = 50 + k * (bw + gap)
        c.box(x, y0, bw, bh, fill=PANEL, edge=color + "99", r=18, lw=1.6)
        c.ax.add_patch(Circle((x + 38, y0 + 40), 19, color=color, zorder=5))
        c.text(x + 38, y0 + 39, n, size=20, color=BG, weight="bold", ha="center", va="center", zorder=6)
        c.text(x + 68, y0 + 20, title, size=25, weight="bold")
        c.text(x + 22, y0 + 84, body, size=18.5, color=MUTED, linespacing=1.5)
        if k < len(steps) - 1:
            c.arrow(x + bw + 1, y0 + bh / 2, x + bw + gap - 1, y0 + bh / 2, color=TEXT, lw=2)
    y1 = 424
    c.text(50, y1, "What the agent is given is what is compared", size=25, weight="bold")
    configs = (("blind", "nothing", "what the model already knows", DIM), ("listing", "names, sizes and times", "the floor, and the leak check", MUTED),
               ("files", "the files and its own tools", "the baseline, always run", BLUE),
               ("<system>", "a search system's tools", "the system alone", VIOLET), ("<system>+files", "both", "as a user who installed it", GREEN))
    cw = (W - 100 - 4 * 16) / 5
    for k, (name, has, role, color) in enumerate(configs):
        x = 50 + k * (cw + 16)
        c.box(x, y1 + 52, cw, 132, fill=PANEL2, edge=color + "AA", r=14, lw=1.4)
        c.text(x + 18, y1 + 66, name, size=22, color=TEXT if color in (DIM, MUTED) else color, weight="bold", font=MONO)
        c.text(x + 18, y1 + 108, has, size=18.5)
        c.text(x + 18, y1 + 140, role, size=16.5, color=MUTED)
    c.save("how-it-works")


# ---------------------------------------------------------------------------------------------------------------- the catalogue

SHELVES = {  # CATALOGUE.md: ID, shelf, size at scale standard, questions
    "Video": (("V1", "Security cameras", "6 files, 3.5 h", 22), ("V2", "Dashcam event clips", "107 files, 0.4 h", 22),
              ("V3", "Football match clips", "525 files, 4.4 h", 23), ("V4", "Work meetings", "12 files, 6.6 h", 24),
              ("V8", "Smart-glasses days", "EgoLife, 21.6 h", 50), ("V9", "Lecture recordings", "26 files, 20.4 h", 23),
              ("V10", "Phone clips at home", "3,000 files, 24.0 h", 23)),
    "Image": (("I1", "Phone camera roll", "3,750 photos", 24), ("I2", "Phone screenshots", "3,071 screenshots", 23),
              ("I3", "Trail camera", "4,256 photos", 23), ("I4", "Building inspection photos", "577 photos", 24),
              ("I7", "Receipts and bills", "1,391 scans", 22), ("I8", "Course notes and slides", "439 pages", 23),
              ("I9", "Nature photos", "3,190 photos", 24)),
    "Audio": (("A1", "Room recorder", "1 file, 24.0 h", 20), ("A3", "Research group meetings", "10 files, 10.0 h", 24),
              ("A4", "Field recorder", "11 files, 10.9 h", 18), ("A6", "Recorded lectures", "15 files, 12.2 h", 23),
              ("A7", "Podcast backlog", "48 files, 18.2 h", 23), ("A8", "Music library", "2,461 files, 20.5 h", 20),
              ("M3", "Voicemails", "1,295 files, 8.9 h", 23), ("M4", "Sound-effects library", "6,123 files, 16.5 h", 21)),
    "Mixed": (("V7", "Developer's week", "generated: 1,572 files of six kinds", 31),),
}


def catalogue() -> None:
    W, H = 1600, 950
    c = Canvas(W, H)
    c.text(50, 30, "23 piles of the media people keep", size=40, weight="bold")
    c.text(50, 88, "Home, work, school and hobbies. Each is a public dataset laid out as the device or the person would have left it.", size=21, color=MUTED)
    cw, th, gap = 484, 74, 10
    for k, kind in enumerate(("Video", "Image", "Audio")):
        x = 50 + k * (cw + 24)
        columns = [(kind, 148)] + ([("Mixed", 148 + 50 + 7 * (th + gap) + 14)] if kind == "Video" else [])
        for name, y in columns:
            shelves, color = SHELVES[name], MEDIA[name]
            c.text(x, y, name.upper(), size=20, color=color, weight="bold")
            c.text(x + cw, y + 3, f"{len(shelves)} pile{'s' * (len(shelves) > 1)}, {sum(s[3] for s in shelves)} questions", size=17, color=MUTED, ha="right")
            c.rect(x, y + 34, cw, 3, color)
            for i, shelf in enumerate(shelves):
                ty = y + 50 + i * (th + gap)
                c.box(x, ty, cw, th, fill=PANEL, edge=LINE, r=12)
                c.box(x + 12, ty + 14, 62, 46, fill=color + "22", edge=color + "88", r=10, lw=1)
                c.text(x + 43, ty + 36, shelf[0], size=20, color=color, weight="bold", ha="center", va="center", font=MONO)
                c.text(x + 90, ty + 9, shelf[1], size=22, weight="bold")
                c.text(x + 90, ty + 42, shelf[2], size=17, color=MUTED)
                c.text(x + cw - 16, ty + 37, f"{shelf[3]} questions", size=15.5, color=DIM, ha="right", va="center")
    c.save("catalogue")


CATEGORIES = (("Find it", "“Find the photo of the parking-level sign.”", (15, 24, 14, 0)),  # CATALOGUE.md: video, image, audio, mixed
              ("Find every one", "“Show me every clip with the cat.”", (15, 40, 32, 0)),
              ("When", "“What time did the van come?”", (30, 23, 32, 12)),
              ("Did it happen", "“Did the smoke alarm go off?”", (21, 22, 30, 0)),
              ("How many, how long", "“How many times did the dog bark?”", (31, 25, 31, 5)),
              ("What did it say or show", "“What was the total on the receipt?”", (47, 18, 15, 14)),
              ("Who or which", "“Which camera saw it first?”", (19, 5, 11, 0)),
              ("Before and after", "“What happened right after the alarm?”", (9, 6, 7, 0)))
CUES = (("sight", 188, TEAL), ("speech", 136, VIOLET), ("text read", 90, AMBER), ("sound", 86, BLUE), ("several", 53, CORAL))


def questions() -> None:
    W, H = 1600, 770
    c = Canvas(W, H)
    c.text(50, 30, "553 questions, filed by what the person wants", size=40, weight="bold")
    c.text(50, 88, "Eight things people ask of their media. The same eight are asked of video, pictures and sound.", size=21, color=MUTED)
    x0, y0, rh, bar = 50, 160, 64, 520
    top = max(sum(v) for _, _, v in CATEGORIES)
    for i, (name, example, counts) in enumerate(CATEGORIES):
        y = y0 + i * rh
        c.text(x0, y, name, size=23, weight="bold")
        c.text(x0, y + 32, example, size=16, color=MUTED)
        x = x0 + 430
        for kind, n in zip(MEDIA, counts):
            w = n / top * bar
            if n:
                c.box(x, y + 8, max(w - 3, 8), 36, fill=MEDIA[kind], edge=None, r=5)
                if w > 30:
                    c.text(x + w / 2 - 1, y + 25, str(n), size=16, color=BG, weight="bold", ha="center", va="center")
            x += w
        c.text(x + 10, y + 25, str(sum(counts)), size=20, weight="bold", va="center")
    x = x0 + 430
    for kind, color in MEDIA.items():
        c.box(x, y0 + 8 * rh + 14, 18, 18, fill=color, edge=None, r=4)
        c.text(x + 26, y0 + 8 * rh + 9, kind.lower(), size=17, color=MUTED)
        x += 112
    px = 1130
    c.box(px, 150, 420, 320, fill=PANEL, edge=LINE, r=18)
    c.text(px + 24, 166, "What must be perceived", size=22, weight="bold")
    for i, (name, n, color) in enumerate(CUES):
        y = 222 + i * 47
        c.text(px + 24, y - 3, name, size=18, color=MUTED)
        c.box(px + 130, y, n / 188 * 215, 24, fill=color, edge=None, r=5)
        c.text(px + 140 + n / 188 * 215, y - 4, str(n), size=18, weight="bold")
    c.box(px, 488, 420, 222, fill=PANEL, edge=LINE, r=18)
    cx, cy, r = px + 100, 600, 68
    c.ax.add_patch(Wedge((cx, cy), r, 0, 360, width=24, color=LINE, zorder=4))
    c.ax.add_patch(Wedge((cx, cy), r, 270, 270 + 0.22 * 360, width=24, color=CORAL, zorder=5))  # y runs downwards: 270 is the top
    c.text(cx, cy - 2, "22%", size=28, weight="bold", ha="center", va="center")
    c.text(px + 196, 522, "“It isn't there”", size=23, weight="bold", color=CORAL)
    c.text(px + 196, 564, "122 questions ask for\nthings that never\nhappened, in the same\nwords as the others.", size=17, color=MUTED, linespacing=1.45)
    c.save("questions")


# ---------------------------------------------------------------------------------------------------------------- the results


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


# results/2026-11-07-search-tool: the questions right of 5, in the order I3, A1, V1, M4
REAL = (("Claude Code alone", (4, 3, 2, 1), BLUE, False), ("Claude Code with a search system", (3, 3, 1, 0), BLUE, True),
        ("Codex alone", (3, 1, 1, 0), AMBER, False), ("Codex with a search system", (4, 1, 2, 0), AMBER, True))
REAL_NAMES = (("I3", "Trail camera", "4,256 photos"), ("A1", "Room recorder", "24 h in one file"), ("V1", "Security cameras", "3.5 h of video"),
              ("M4", "Sound effects", "6,123 clips"))


def results() -> None:
    W, H = 1600, 810
    c = Canvas(W, H)
    c.text(50, 30, "Four real piles: today's agents find half, or less", size=40, weight="bold")
    c.text(50, 88, "5 test questions of each pile, 15 minutes a question (6 and 7 November 2026). Claude Code on Haiku and Codex on gpt-6-luna, at xhigh effort.",
           size=19.5, color=MUTED)
    x0, x1, yb, yt = 110, 1550, 640, 196

    def at(p):
        return yb - p * (yb - yt)

    for p in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        c.ax.plot([x0, x1], [at(p), at(p)], color=LINE, lw=1, zorder=2)
        c.text(x0 - 14, at(p), f"{p:.0%}", size=16, color=DIM, ha="right", va="center")
    c.ax.plot([x0, x1], [at(0.2), at(0.2)], color=CORAL, lw=1.8, ls=(0, (6, 5)), zorder=3)
    c.text(x0 + (x1 - x0) * 0.8 - 14, at(0.2) - 54, "the floor: file names,\nsizes and times only", size=15.5, color=CORAL, ha="right",
           linespacing=1.3)
    groups = [*REAL_NAMES, ("All", "All four", "20 questions")]
    gw, bw = (x1 - x0) / len(groups), 50
    for g, (tag, name, size) in enumerate(groups):
        cx = x0 + gw * (g + 0.5)
        if tag == "All":
            c.rect(x0 + gw * g + 6, yt - 26, gw - 6, yb - yt + 26, PANEL2, zorder=1, alpha=0.8)
            c.text(cx, yt - 22, "with 95% intervals", size=14, color=DIM, ha="center")
        for s, (_, right, color, searched) in enumerate(REAL):
            k, n = (sum(right), 20) if tag == "All" else (right[g], 5)
            bx = cx + (s - 2) * (bw + 6) + 3 + (8 if s >= 2 else -8)
            h = max(k / n * (yb - yt), 3)
            c.ax.add_patch(Rectangle((bx, yb - h), bw, h, facecolor=color + "2A" if searched else color, edgecolor=color, linewidth=2.2,
                                     hatch="///" if searched else None, zorder=4))
            top = yb - h
            if tag == "All":
                lo, hi = wilson(k, n)
                top = at(hi)
                c.ax.plot([bx + bw / 2] * 2, [at(lo), at(hi)], color=TEXT, lw=1.8, zorder=6)
                for e in (lo, hi):
                    c.ax.plot([bx + bw / 2 - 8, bx + bw / 2 + 8], [at(e), at(e)], color=TEXT, lw=1.8, zorder=6)
            c.text(bx + bw / 2, top - 8, f"{k}/{n}", size=17, weight="bold", ha="center", va="bottom")
        c.text(cx, yb + 14, name, size=21, weight="bold", ha="center")
        c.text(cx, yb + 46, size, size=16, color=MUTED, ha="center")
    x = 110
    for label, _, color, searched in REAL:
        c.ax.add_patch(Rectangle((x, 726), 30, 20, facecolor=color + "2A" if searched else color, edgecolor=color, linewidth=2,
                                 hatch="///" if searched else None, zorder=4))
        x += 40 + c.width(c.text(x + 40, 721, label, size=18)) + 36
    c.text(110, 766, "The search system is MultimediaSearch with its d1-Omni-2B model, used beside the agent's own tools. No difference between two bars is proven: 5 questions are too few.",
           size=15.5, color=MUTED)
    c.save("results")


def generated_week() -> None:
    W, H = 1600, 690
    c = Canvas(W, H)
    c.text(50, 30, "A generated week: as many right answers, for far less work", size=40, weight="bold")
    c.text(50, 88, "24 questions about 3 generated days of screen, door camera, phone and logs (4 November 2026). The files alone, against MultimediaSearch (d1-3B) alone.",
           size=19.5, color=MUTED)
    panels = (("Right, of 24", (("Codex", 22, 23), ("Claude Code", 18, 22)), 24, "{:.0f}", "The gains are not yet significant: McNemar p = 1.0 for Codex, 0.125 for Claude."),
              ("Tokens used, millions", (("Codex", 5.3, 2.7), ("Claude Code", 29.9, 3.9)), 30, "{:.1f}M", "Claude's figure with search is a lower bound: two execs timed out."),
              ("Median seconds a question", (("Codex", 28, 19), ("Claude Code", 58, 32)), 60, "{:.0f} s", "Search adds local GPU time: 77 to 179 s for each right media answer."))
    pw, ph, y0 = 484, 500, 150
    for k, (title, rows, top, fmt, note) in enumerate(panels):
        x = 50 + k * (pw + 24)
        c.box(x, y0, pw, ph, fill=PANEL, edge=LINE, r=18)
        c.text(x + 26, y0 + 20, title, size=25, weight="bold")
        for i, (agent, files, search) in enumerate(rows):
            y = y0 + 86 + i * 160
            c.text(x + 26, y, agent, size=20, color=MUTED, weight="bold")
            for j, (label, value, color) in enumerate((("files alone", files, DIM), ("with search", search, GREEN))):
                yy = y + 38 + j * 50
                w = max(value / top * (pw - 250), 6)
                c.box(x + 26, yy, w, 38, fill=color, edge=None, r=7)
                c.text(x + 38 + w, yy + 18, fmt.format(value), size=22, weight="bold", va="center", color=TEXT if j else MUTED)
                c.text(x + pw - 22, yy + 18, label, size=15, color=DIM, ha="right", va="center")
        c.text(x + 26, y0 + ph - 74, wrap(note, 50), size=16, color=MUTED, linespacing=1.4)
    c.save("generated-week")


def egolife() -> None:
    W, H = 1600, 450
    c = Canvas(W, H)
    c.text(50, 30, "Real smart-glasses video: the files alone are hard to beat", size=40, weight="bold")
    c.text(50, 88, "EgoLifeQA: 40 multiple-choice video questions over 21.6 hours of one wearer (3 November 2026). Codex on gpt-6.1-terra at high effort.",
           size=19.5, color=MUTED)
    rows = (("blind", "no recordings: what the model already knows", 16, DIM), ("files", "the clips, the transcripts and its own tools", 36, BLUE),
            ("search system", "MultimediaSearch (d1-3B) alone", 34, GREEN))
    x0, bar = 520, 900
    for i, (name, what, right, color) in enumerate(rows):
        y = 160 + i * 84
        c.text(50, y, name, size=24, weight="bold", font=MONO)
        c.text(50, y + 36, what, size=16.5, color=MUTED)
        c.box(x0, y + 6, bar, 44, fill=PANEL2, edge=None, r=8)
        c.box(x0, y + 6, right / 40 * bar, 44, fill=color, edge=None, r=8)
        lo, hi = wilson(right, 40)
        c.ax.plot([x0 + lo * bar, x0 + hi * bar], [y + 28, y + 28], color=TEXT, lw=2, zorder=6)
        c.text(x0 + bar + 16, y + 27, f"{right}/40", size=24, weight="bold", va="center")
    c.ax.plot([x0 + 0.25 * bar] * 2, [152, 400], color=CORAL, lw=1.6, ls=(0, (6, 5)), zorder=7)
    c.text(x0 + 0.25 * bar, 406, "chance: 25%", size=15, color=CORAL, ha="center")
    c.text(x0 + bar, 406, "lines: 95% intervals", size=15, color=DIM, ha="right")
    c.save("egolife")


FIGURES = {"hero": hero, "piles": piles, "how-it-works": pipeline, "catalogue": catalogue, "questions": questions,
           "results": results, "generated-week": generated_week, "egolife": egolife}
NEED_DATA = ("hero", "piles")


def main() -> None:
    home = Path(os.environ.get("MMSB_HOME") or Path.home())
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=home / "mmsb-data", help="the folder of the built dumps (default: <MMSB_HOME>/mmsb-data)")
    p.add_argument("--only", default=",".join(FIGURES), help="the figures to draw, comma separated: " + ", ".join(FIGURES))
    args = p.parse_args()
    for name in args.only.split(","):
        if name in NEED_DATA:
            FIGURES[name](args.data)
        else:
            FIGURES[name]()


if __name__ == "__main__":
    main()
