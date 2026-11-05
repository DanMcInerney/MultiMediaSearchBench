"""The owner's screen in the V7 week: the states, their frames as the recorder keeps them, and the files.

    screen_state(state) -> Image      one state of the screen at its start, 1280x720: an app card, an accent, an overlay (a dialog), the taskbar
    screen_layers(state)              the same in layers: the card, and the decorations over it with their exact mask
    screen_frames(spec, start)        the frames of a screen segment: when each is (half seconds), and the state, cursor, typing, scroll and clock it shows
    screen_frame(layers, frame) -> Image   a frame: the card scrolled and typed on, the decorations over it, the taskbar clock and the cursor
    write_screen(path, spec, start, work)  the file: VFR x264

A screen segment is `{"length_s", "states": [{"from", "to", "app", "overlay": None|{"kind", "rgb", "code", "msg", "at"?},
"accent": None|{"kind", "rgb"}, "seed", "cursor"?: [x, y], "acts"?: [[tick, "move", x, y] | [tick, "type", n] | [tick, "scroll", lines], ..]}]}`:
the cursor's place at the state's start (CURSOR_HOME without one) and what the owner does after it, at `tick` half seconds from the
state's start (from 1, increasing): a `move` of the cursor, a `type` (`n` characters are on the app's typing line, TYPING) or a `scroll`
(`lines` lines scrolled since the state's start; a scroll empties the typing line). The plan also names the state's `level` of activity
(idle, light or busy: the planner's, not drawn). Its frames are `screen_frames`'.

An error state has an `overlay` of kind `error`, `rgb` NEEDLE_RGB["error"] and a `code` such as `E-4417`; its distractors are the other
overlay kinds and the accents, whose `rgb` may be left out. A dialog does not always open in one place, so that a crop of one place
finds it: it opens at `overlay["at"]`, its centre (`dialog_at` draws one; DIALOG_AT says where, in the middle of the screen).
`DIALOG_AREA` is the union of every place a dialog can take, as fractions of the frame; the dialogs of the error's red are a confirm
(its header and its button), and a warning given a red header. A dialog's title is 20 px tall at the reader's 640x360, readable to the
decider but about 8 px in a contact sheet 5 frames across, so that an agent must open or crop a frame to tell an error from the red
"Confirm" and the red-headed warnings, as with a real dialog.

The screen follows the real recorder (DESIGN.md §6: ddagrab at about 2 fps with the cursor drawn, then `mpdecimate=max=119` and
`-fps_mode vfr`): a frame whenever the picture changes, at most one every half second (a tick, TICK_S), and at least one every 60 s. mm's
reader takes the first frame of each second, so a segment's windows are at most its seconds with a frame, and the files arm sees every
frame. A frame is a state's start, an act (a cursor move, typing or a scroll) or a minute change of the taskbar's clock; `screen_frames`
is the one place that says where they are, and the writer, the checks, the window counts and the probe all use it. Each frame differs from
the one before it, as the recorder's `mpdecimate` makes them: a typed character is a visible (not space) one (`typed_text`), and the plan
types no more than the typing line shows where a decoration covers its end (`typing_cap`, from `_decorations`).
A state is drawn in layers (`screen_layers`): the app card, and the decorations that stand over it and never move, the accent, the dialog
and the taskbar (TASKBAR_H px along the bottom, below every dialog and needle: dark, with the time and the date at the right as Windows shows
them), with an exact mask found by drawing them on a black and on a white background, so a dialog's white body over a white page is
opaque. A frame is the card with its content box rolled up by the lines scrolled (wrapping: the lines that leave at the top come back at the
bottom, so a scroll brings no new colour), the typed text on the app's typing line (TYPING: a box, a line height, a line that holds the
text's whole height, a size, an ink from the app's own palette, and a bold face a little larger than the card's text), the decorations
pasted over both, then the clock and the cursor (`stamp`). Layers made with no scroll or typing are `screen_state(state)` exactly. A frame
costs a copy, a roll, a short text and a masked paste.
The cursor is a white arrow with a black outline, 12x19 px at 1280x720 (about 6x9.5 at the reader's 640x360; on the dev laptop, 3840x2400 at
200% with pointer size 2, the owner's own pointer is 0.67 to 1 times that at 640 px). The reader's change gate (a cell of 16 px at 640x360
moving by more than 6) is sensitive to its size: at this size 87 to 99% of moves are a change, at 0.8 times it 19 to 89%, at 0.67 times
0 to 78%. `plan.py` plans what the owner does (`state_acts`: idle, light and busy states drawn as a chain, in runs of active
seconds with 1 or 2 acts each, in the mix of kinds of the app), and its constants are calibrated against two real hours of a screen
recording (TESTBED.md): the typed text is larger and bolder than the card's so that a frame of typing changes about as much of the frame
at 640x360 as a real one's frame (0.05 to 0.3%, the card's own size gives 0.025%), and any typed character is a change to the gate
(98% of the seconds with a frame are windows, 71 to 86% in the real hours).
Key frames are at a file's start and at scene cuts (x264 puts one at each state change, 1 to 4 a file), and none is forced every 10 s, so a
segment is cut at a state's start. Not modelled: the recorder forces a key frame every 10 s (DESIGN.md §6), so its cuts land mid-state; a scroll wraps instead of bringing new lines; there are no caret blinks (A-B-A returns are
0%, 2 to 5% in the real hours). Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

import bisect
import math
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from mmmb.scenarios.v7_screen.constants import font
from mmmb.scenarios.v7_screen.media_common import (
    DISTRACTOR_RGB, NEEDLE_RGB, SAFE, SHADE_GAP, _clear_of_needles, _fit, _ink, _rgb, _rng, _write_video, distance,
)

SCREEN = (1280, 720)

SCREEN_CRF = 30  # capture.py's screen preset

DIALOG = (0.45, 0.35)  # a dialog's share of the screen's width and height: 15.75% of its area
DIALOG_AT = ((0.38, 0.62), (0.40, 0.60))  # where a dialog's centre may be, as fractions of the screen: the range of x and of y (in the middle part)
DIALOG_AREA = (DIALOG_AT[0][0] - DIALOG[0] / 2, DIALOG_AT[1][0] - DIALOG[1] / 2, DIALOG_AT[0][1] + DIALOG[0] / 2, DIALOG_AT[1][1] + DIALOG[1] / 2)
# ^ every place a dialog can be, (x0, y0, x1, y1) as fractions of the screen: the union of its positions (its shadow, grey, is left out)
TASKBAR_H = 40  # px: the taskbar along the bottom of the screen, under the dialogs (DIALOG_AREA ends at 78% of the height) and every needle
TASKBAR_RGB, TASKBAR_LINE, TASKBAR_ICON, TASKBAR_LOGO, TASKBAR_INK = (32, 32, 40), (64, 64, 72), (96, 96, 108), (0, 120, 215), (235, 235, 240)
CURSOR_ARROW = ((0, 0), (0, 17), (4, 13), (7, 18), (9, 17), (6, 12), (11, 12))  # the pointer's outline from its tip: 12 px wide, 19 tall
CURSOR_HOME = (640, 360)  # where the cursor is when a state names no place
FLOOR_S = 60  # the recorder keeps a frame at least this often (mpdecimate max=119 at 2 fps)
TITLE_PX = 40  # a dialog's title, in 1280x720 pixels: 20 after the scaling to 640x360, 8 in a contact sheet 5 frames across


def dialog_at(r: random.Random) -> list[float]:
    """Where a dialog opens: the centre of one of the places of DIALOG_AREA, as fractions of the screen, from `r`."""
    return [round(r.uniform(*DIALOG_AT[0]), 3), round(r.uniform(*DIALOG_AT[1]), 3)]


APPS = ("editor", "browser", "terminal", "chat", "sheet", "slides")

ACCENTS = ("banner", "cells", "badge", "toast")
WORDS = ("main", "config", "parse", "render", "index", "value", "return", "import", "result", "total", "update", "queue",
         "stream", "handler", "buffer", "token", "client", "export", "merge", "cache")
FILES = ("app.py", "util.py", "server.py", "cli.py", "models.py", "tests.py", "README.md", "setup.cfg", "notes.txt")
CODE_RGB = ((86, 156, 214), (78, 201, 176), (206, 145, 120), (197, 134, 192), (220, 220, 170), (181, 206, 168))
THEMES = ((30, 50, 110), (20, 110, 120), (50, 50, 58), (240, 232, 215), (60, 40, 110), (30, 90, 60))


def _line(r: random.Random, lo: int = 2, hi: int = 6) -> str:
    return " ".join(r.choice(WORDS) for _ in range(r.randint(lo, hi)))


def screen_state(state: dict) -> Image.Image:
    """One screen state, 1280x720: the app card (`editor`, `browser`, `terminal`, `chat`, `sheet` or `slides`), then its accent
    (`banner`, `cells`, `badge` or `toast`, in a red that is not the error's), then its overlay: a dialog 45% by 35% of the screen,
    centred on the overlay's `at` (a place of DIALOG_AREA; the middle without one), with a header in the kind's colour. An `error`'s
    header is the error needle's shade, titled `ERROR <code>` in 40 px letters (as is every kind's title; the header stays 38% of the
    dialog's height); a warning is orange, an info blue, a save prompt grey and a confirm ("Delete 14 files?") another red. The
    taskbar is along the bottom, with no clock (`screen_frame` adds it, and the cursor)."""
    r = _rng(state.get("seed", 0), "screen")
    image = _APPS[state["app"]](r)
    _decorate(image, state, r)
    return image


def _decorate(image: Image.Image, state: dict, r: random.Random) -> None:
    """What stands on top of the app card and never moves: the state's accent, its overlay, the taskbar."""
    if accent := state.get("accent"):
        _accent(image, accent, r)
    if overlay := state.get("overlay"):
        _dialog(image, overlay, r)
    _taskbar(image)


def _chrome(draw: ImageDraw.ImageDraw, title: str, fill=(225, 228, 235), ink=(40, 40, 48)) -> None:
    draw.rectangle([0, 0, SCREEN[0], 40], fill=fill)
    draw.text((20, 6), title, fill=ink, font=font(24))
    for k in range(3):
        draw.ellipse([SCREEN[0] - 120 + k * 34, 12, SCREEN[0] - 102 + k * 34, 30], fill=(150, 152, 160))


def _editor(r: random.Random) -> Image.Image:
    dark = r.random() < 0.7
    bg, fg, side, bar = (((30, 30, 36), (212, 212, 212), (37, 37, 43), (52, 52, 60)) if dark
                         else ((252, 252, 252), (40, 40, 46), (240, 240, 243), (225, 228, 235)))
    image = Image.new("RGB", SCREEN, bg)
    draw = ImageDraw.Draw(image)
    name = r.choice(FILES)
    _chrome(draw, f"{name} - Visual Studio Code", bar, fg)
    draw.rectangle([0, 40, 56, SCREEN[1] - 28], fill=bar)
    draw.rectangle([56, 40, 300, SCREEN[1] - 28], fill=side)
    for k, file in enumerate(r.sample(FILES, 7)):
        draw.text((76, 62 + k * 38), file, fill=fg, font=font(24))
    draw.rectangle([300, 40, SCREEN[0], 80], fill=side)
    draw.text((320, 48), name, fill=fg, font=font(24))
    for k in range(15):
        y = 100 + k * 34
        draw.text((316, y), str(k + 1), fill=(120, 120, 130), font=font(24))
        draw.text((380 + 40 * r.randint(0, 3), y), _line(r), fill=r.choice(CODE_RGB) if dark else (40, 70, 130), font=font(26))
    draw.rectangle([0, SCREEN[1] - 28, SCREEN[0], SCREEN[1]], fill=(0, 122, 204))
    draw.text((16, SCREEN[1] - 26), "main  0 problems  Ln 12, Col 8  UTF-8", fill="white", font=font(20))
    return image


def _browser(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, SCREEN[0], 92], fill=(225, 228, 235))
    draw.rounded_rectangle([20, 6, 280, 40], radius=10, fill=(250, 250, 250))
    draw.text((36, 10), r.choice(("Docs", "News", "Shop", "Mail")), fill=(50, 50, 60), font=font(22))
    draw.rounded_rectangle([160, 44, SCREEN[0] - 160, 88], radius=18, fill="white")  # (the address that is typed at)
    draw.text((190, 54), f"https://example.com/{r.choice(('docs', 'news', 'shop', 'weather'))}", fill=(60, 60, 70), font=font(24))
    draw.text((80, 130), _line(r).title(), fill=(30, 30, 40), font=font(54))
    block = r.choice(SAFE)
    draw.rectangle([80, 220, 480, 440], fill=block)
    for k in range(7):
        draw.rounded_rectangle([520, 230 + k * 40, 520 + r.randint(400, 640), 254 + k * 40], radius=8, fill=(210, 210, 215))
    for k in range(4):
        draw.text((80, 470 + k * 44), _line(r, 5, 9), fill=(60, 60, 70), font=font(28))
    return image


def _terminal(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (14, 14, 16))
    draw = ImageDraw.Draw(image)
    _chrome(draw, "Terminal", (45, 45, 52), (220, 220, 225))
    for k in range(14):
        y = 56 + k * 40
        if k % 4 == 0:
            draw.text((30, y), "$ ", fill=(80, 220, 110), font=font(28))
            draw.text((70, y), _line(r, 1, 4), fill=(235, 235, 235), font=font(28))
        else:
            draw.text((30, y), _line(r), fill=r.choice(((200, 200, 205), (110, 190, 255), (230, 200, 90))), font=font(28))
    draw.text((30, 56 + 14 * 40), "$ ", fill=(80, 220, 110), font=font(28))  # the prompt that is typed at
    return image


def _chat(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (232, 232, 236))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 280, SCREEN[1]], fill=(60, 40, 90))
    for k, channel in enumerate(r.sample(WORDS, 8)):
        draw.text((24, 30 + k * 48), f"# {channel}", fill=(230, 225, 240), font=font(26))
    for k in range(6):
        mine = k % 2
        x0 = 640 if mine else 320
        draw.rounded_rectangle([x0, 40 + k * 100, x0 + 560, 110 + k * 100], radius=24, fill=(70, 140, 240) if mine else "white")
        draw.text((x0 + 24, 60 + k * 100), _line(r, 3, 6), fill="white" if mine else (40, 40, 50), font=font(28))
    draw.rounded_rectangle([320, SCREEN[1] - 104, SCREEN[0] - 40, SCREEN[1] - 56], radius=14, fill="white")  # the message that is typed in
    return image


def _sheet(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (252, 252, 252))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, SCREEN[0], 40], fill=(225, 235, 225))
    draw.text((20, 6), f"{r.choice(('budget', 'sales', 'plan', 'tracker'))}.xlsx", fill=(30, 60, 40), font=font(24))
    draw.text((14, 50), "fx", fill=(120, 120, 130), font=font(24))  # the formula bar (40 to 84) is typed in
    draw.rectangle([0, 84, SCREEN[0], 124], fill=(240, 240, 242))
    for x in range(0, SCREEN[0], 160):
        draw.line([x, 84, x, SCREEN[1]], fill=(200, 200, 205), width=2)
        draw.text((x + 70, 92), "ABCDEFGH"[x // 160], fill=(80, 80, 90), font=font(24))
    for y in range(124, SCREEN[1], 40):
        draw.line([0, y, SCREEN[0], y], fill=(200, 200, 205), width=2)
        for x in range(0, SCREEN[0], 160):
            draw.text((x + 10, y + 4), f"{r.randint(1, 999)}.{r.randint(10, 99)}", fill=(30, 30, 40), font=font(24))
    draw.rectangle([160, 164, 480, 244], outline=(30, 110, 220), width=4)
    return image


def _slides(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (236, 236, 240))
    draw = ImageDraw.Draw(image)
    _chrome(draw, "Slides", (225, 228, 235))
    theme = r.choice(THEMES)
    for k in range(4):
        draw.rectangle([24, 70 + k * 150, 220, 190 + k * 150], fill=theme if k == 0 else (250, 250, 252), outline=(160, 160, 170), width=2)
    draw.rectangle([260, 70, SCREEN[0] - 40, SCREEN[1] - 40], fill=theme)
    ink = _ink(theme)
    draw.text((320, 120), _line(r, 2, 3).title(), fill=ink, font=font(72))
    for k in range(4):
        draw.text((340, 270 + k * 70), "- " + _line(r, 3, 6), fill=ink, font=font(38))
    return image


_APPS = {"editor": _editor, "browser": _browser, "terminal": _terminal, "chat": _chat, "sheet": _sheet, "slides": _slides}


def _accent(image: Image.Image, accent: dict, r: random.Random) -> None:
    """A red that is not the error's: an alert banner, a block of red cells, a badge on the title bar or a toast."""
    kind = accent["kind"]
    rgb = _rgb(accent.get("rgb") or DISTRACTOR_RGB["error"][kind])
    if distance(rgb, NEEDLE_RGB["error"]) < SHADE_GAP:
        raise ValueError(f"the {kind} accent {rgb} is within {SHADE_GAP} of the error needle's shade {NEEDLE_RGB['error']}")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    if kind == "banner":
        draw.rectangle([0, 96, w, 226], fill=rgb)
        words = r.choice(("Limited offer: 50% off today", "Service alert: delays expected", "Sale ends tonight"))
        draw.text((80, 130), words, fill="white", font=font(48))
    elif kind == "cells":
        x0, y0 = 160 * r.randint(1, 4), 40 * r.randint(3, 9)
        draw.rectangle([x0, y0, x0 + 320, y0 + 120], fill=rgb)
        for k in range(3):
            draw.text((x0 + 12, y0 + 6 + k * 40), f"{r.randint(1, 999)}.{r.randint(10, 99)}", fill=(30, 30, 40), font=font(24))
    elif kind == "badge":
        draw.ellipse([w - 70, 4, w - 30, 38], fill=rgb)
        draw.text((w - 50, 21), str(r.randint(1, 9)), fill="white", font=font(22), anchor="mm")
    else:
        draw.rounded_rectangle([w - 440, h - 140, w - 24, h - 40], radius=14, fill=rgb)
        draw.text((w - 420, h - 112), r.choice(("Sync failed, retrying", "Upload paused", "Low disk warning")), fill="white", font=font(32))


def _dialog_box(overlay: dict) -> tuple[int, int, int, int]:
    """A dialog's left, top, width and height (its shadow is 12 px to the right and 14 below), where `overlay["at"]` puts it."""
    w, h = SCREEN
    dw, dh = round(w * DIALOG[0]), round(h * DIALOG[1])
    x0, y0 = (w - dw) // 2, (h - dh) // 2
    if overlay.get("at"):
        cx, cy = overlay["at"]
        if not (DIALOG_AT[0][0] <= cx <= DIALOG_AT[0][1] and DIALOG_AT[1][0] <= cy <= DIALOG_AT[1][1]):
            raise ValueError(f"a {overlay['kind']} dialog opens at {overlay['at']}, outside the places of DIALOG_AT {DIALOG_AT}")
        x0, y0 = round(w * cx - dw / 2), round(h * cy - dh / 2)
    return x0, y0, dw, dh


def _dialog(image: Image.Image, overlay: dict, r: random.Random) -> None:
    kind = overlay["kind"]
    want = NEEDLE_RGB["error"] if kind == "error" else DISTRACTOR_RGB["error"][kind]
    head = _rgb(overlay.get("rgb") or want)
    if kind == "error" and head != NEEDLE_RGB["error"]:
        raise ValueError(f"an error dialog's header is the error needle's shade {NEEDLE_RGB['error']}, not {head}")
    if kind != "error" and distance(head, NEEDLE_RGB["error"]) < SHADE_GAP:
        raise ValueError(f"a {kind} dialog's header {head} is within {SHADE_GAP} of the error needle's shade {NEEDLE_RGB['error']}")
    draw = ImageDraw.Draw(image)
    x0, y0, dw, dh = _dialog_box(overlay)
    top = round(dh * 0.38)
    code = overlay.get("code")
    title = {"error": f"ERROR {code}" if code else "ERROR", "warning": f"WARNING {code}" if code else "WARNING", "info": "INFO",
             "save": "Unsaved changes", "confirm": "Confirm"}[kind]
    draw.rectangle([x0 + 12, y0 + 14, x0 + dw + 12, y0 + dh + 14], fill=(70, 70, 76))
    draw.rectangle([x0, y0, x0 + dw, y0 + dh], fill=(248, 248, 250), outline=(150, 150, 155), width=2)
    draw.rectangle([x0, y0, x0 + dw, y0 + top], fill=head)
    _fit(draw, (x0 + 28, y0 + top // 2), title, "white", TITLE_PX, dw - 56, anchor="lm")
    message = overlay.get("msg") or {"error": "The operation failed", "warning": "Check before you go on", "info": "Done",
                                     "save": "Save changes to report.docx?", "confirm": "Delete 14 files?"}[kind]
    _fit(draw, (x0 + 28, y0 + top + 50), str(message), (30, 30, 36), 38, dw - 56, anchor="lm")
    buttons = {"error": (("OK", (225, 225, 230)),), "warning": (("OK", (225, 225, 230)),), "info": (("OK", (225, 225, 230)),),
               "save": (("Cancel", (225, 225, 230)), ("Save", (60, 110, 220))),
               "confirm": (("Cancel", (225, 225, 230)), ("Delete", _rgb(DISTRACTOR_RGB["error"]["confirm"])))}[kind]
    x = x0 + dw - 24
    for label, fill in reversed(buttons):
        bw = 150
        draw.rounded_rectangle([x - bw, y0 + dh - 76, x, y0 + dh - 24], radius=10, fill=fill)
        draw.text((x - bw // 2, y0 + dh - 50), label, fill=_ink(fill), font=font(30), anchor="mm")
        x -= bw + 16


TICK_S = 0.5  # a frame is at a tick: the recorder's 2 a second


class Typing(NamedTuple):
    """Where an app scrolls and is typed in (1280x720 px). `box` is what a scroll moves (up `line_h` px a line, wrapping), `rect` the
    typing line, filled with the card's background where text is typed, `at` where its text starts (after `prefix`), in `size` px
    letters, `cap` the most visible (not space) characters it holds, `ink` the letters' colour on a light and on a dark background (None: the slide's own)."""

    box: tuple[int, int, int, int]
    line_h: int
    rect: tuple[int, int, int, int]
    at: tuple[int, int]
    size: int
    cap: int
    ink: tuple | None
    prefix: str = ""
    stroke: int = 0


TYPING = {  # (the rect holds the text's whole height, descenders too, and is clear of the edge of what it is typed in)
    "editor": Typing((304, 96, 1280, 606), 34, (372, 610, 1272, 656), (380, 608), 36, 40, ((40, 40, 46), (212, 212, 212)), "", 1),
    "browser": Typing((60, 100, 1240, 672), 44, (186, 47, 1100, 85), (190, 43), 34, 44, ((60, 60, 70), (235, 235, 235)), "", 1),
    "terminal": Typing((0, 52, 1280, 612), 40, (66, 617, 1272, 665), (70, 614), 38, 52, ((20, 20, 24), (235, 235, 235)), "", 1),
    "chat": Typing((292, 0, 1280, 612), 34, (340, 618, 1224, 662), (346, 613), 38, 37, ((40, 40, 50), (235, 235, 235)), "", 1),
    "sheet": Typing((0, 124, 1280, 644), 40, (56, 42, 1272, 82), (60, 39), 34, 58, ((30, 30, 40), (235, 235, 235)), "", 1),
    "slides": Typing((260, 70, 1240, 560), 70, (336, 555, 1236, 610), (340, 550), 46, 30, None, "- ", 1),
}


class ScreenFrame(NamedTuple):
    """One frame of a screen segment: its second (from the segment's start, a multiple of TICK_S), the index of the state it is in, the
    cursor's tip, the taskbar's clock (`14:05`) and date (`2026-10-28`: ISO, since `01/11` reads as 11 January on a US machine and
    a week answer names its day), the characters typed on the app's typing line and the lines scrolled since the state's start."""

    second: float
    state: int
    cursor: tuple[int, int]
    clock: str
    date: str
    typed: int = 0
    lines: int = 0


def _taskbar(image: Image.Image) -> None:
    """The taskbar: a dark strip along the bottom with a logo and some icons in the middle; the clock is the frame's."""
    w, h = image.size
    top = h - TASKBAR_H
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, top, w, h], fill=TASKBAR_RGB)
    draw.line([0, top, w, top], fill=TASKBAR_LINE, width=1)
    x = w // 2 - 130
    for dx, dy in ((0, 0), (11, 0), (0, 11), (11, 11)):
        draw.rectangle([x + dx, top + 9 + dy, x + dx + 9, top + 9 + dy + 9], fill=TASKBAR_LOGO)
    for k in range(5):
        draw.rounded_rectangle([x + 48 + k * 44, top + 8, x + 48 + k * 44 + 24, top + 32], radius=6, fill=TASKBAR_ICON)


@lru_cache(maxsize=16)
def _cached_font(size: int):
    return font(size)


def typing_text(seed: int) -> str:
    """The text a state's owner types, from its seed: words with spaces between."""
    r = _rng(seed, "typing")
    return " ".join(r.choice(WORDS) for _ in range(120))


def typed_text(text: str, lines: int, n: int, prefix: str = "") -> str:
    """What is on the typing line when `n` characters are typed after `lines` lines have been scrolled: `prefix`, then `text` from a place
    that the lines give, up to its n-th visible (not space) character, so that every character typed changes the picture."""
    start, seen = lines * 41 % 300, 0
    for i in range(start, len(text)):
        seen += text[i] != " "
        if seen >= n:
            return prefix + text[start:i + 1]
    return prefix + text[start:]


def _decorations(state: dict) -> list[tuple[int, int, int, int]]:
    """Where the decorations of `state` are, (x0, y0, x1, y1) each, so far as they can hide a typing line: the taskbar, a toast, a banner, a
    badge, the box a block of red cells is in, and the dialog with its shadow."""
    w, h = SCREEN
    out = [(0, h - TASKBAR_H, w, h)]
    kind = (state.get("accent") or {}).get("kind")
    if kind == "toast":
        out.append((w - 440, h - 140, w - 24, h - 40))
    elif kind == "banner":
        out.append((0, 96, w, 226))
    elif kind == "badge":
        out.append((w - 70, 4, w - 30, 38))
    elif kind == "cells":
        out.append((160, 120, 1120 + 160, 480))  # (x0 is 160 to 640, y0 120 to 360)
    if overlay := state.get("overlay"):
        x0, y0, dw, dh = _dialog_box(overlay)
        out.append((x0, y0, x0 + dw + 12, y0 + dh + 14))
    return out


@lru_cache(maxsize=8)
def _rows(app: str) -> tuple[int, int]:
    """The first and last row of the app's typed text, whatever the letters (WORDS' ascenders and descenders, and the stroke)."""
    typing = TYPING[app]
    top, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(typing.at, " ".join(WORDS), font=_cached_font(typing.size), stroke_width=typing.stroke)[1::2]
    return top, bottom


def typing_limit(state: dict) -> int:
    """The x up to which the typing line of `state` shows: where the first decoration over the text's rows begins, else the line's end."""
    typing = TYPING[state["app"]]
    top, bottom = _rows(state["app"])
    limit = typing.rect[2]
    for x0, y0, x1, y1 in _decorations(state):
        if y0 < bottom and y1 > top and x1 > typing.at[0]:
            limit = min(limit, max(x0, typing.at[0]))
    return limit


def typing_cap(state: dict, lines: int) -> int:
    """The most visible characters the typing line of `state` can show after `lines` lines have been scrolled: the app's `cap`, or fewer
    where a decoration (a toast over the end of the line) hides the rest: a character typed under one would change nothing."""
    typing = TYPING[state["app"]]
    limit = typing_limit(state)
    if limit >= typing.rect[2]:
        return typing.cap
    text, font = typing_text(state.get("seed", 0)), _cached_font(typing.size)
    for n in range(1, typing.cap + 1):
        if typing.at[0] + font.getlength(typed_text(text, lines, n, typing.prefix)) + 2 * typing.stroke > limit:
            return n - 1
    return typing.cap


class ScreenLayers:
    """A state's picture in layers, so that a frame is a copy, a roll, a short text and a masked paste: the app card (`card`); what stands
    on top of it and never moves (`deco`, the accent, the overlay and the taskbar, drawn as `screen_state` draws them) with its exact
    `mask` (255 where a decoration pixel is, found by drawing the decorations on an all black and an all white background: a pixel that is
    the same on both is theirs, whatever the card shows there; a dialog's white body over a white page is in the mask); and the typing
    line's `bg` and `ink`, with the `text` the owner types (from the state's seed)."""

    def __init__(self, state: dict):
        seed = state.get("seed", 0)
        r = _rng(seed, "screen")
        self.card = _APPS[state["app"]](r)
        after = r.getstate()
        drawn = []
        for sentinel in ((0, 0, 0), (255, 255, 255)):
            image = Image.new("RGB", SCREEN, sentinel)
            r.setstate(after)
            _decorate(image, state, r)
            drawn.append(image)
        same = (np.asarray(drawn[0]) == np.asarray(drawn[1])).all(axis=2)
        self.deco = drawn[0]
        self.mask = Image.fromarray(same.astype(np.uint8) * 255, "L")
        rows = np.flatnonzero(same.any(axis=1))
        self.regions = []  # the decorations by bands of rows, each pasted by its own box: (xy, deco, mask), cheaper than the whole frame
        for band in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
            y0, y1 = int(band[0]), int(band[-1]) + 1
            cols = np.flatnonzero(same[y0:y1].any(axis=0))
            box = (int(cols[0]), y0, int(cols[-1]) + 1, y1)
            self.regions.append((box[:2], self.deco.crop(box), self.mask.crop(box)))
        self.typing = TYPING[state["app"]]
        self.bg = self.card.getpixel((self.typing.rect[2] - 6, self.typing.rect[1] + 3))
        light = 0.299 * self.bg[0] + 0.587 * self.bg[1] + 0.114 * self.bg[2] > 150
        self.ink = _ink(self.bg) if self.typing.ink is None else self.typing.ink[0 if light else 1]
        _clear_of_needles(self.ink, "the typing ink", "screen")
        self.text = typing_text(seed)
        self._rolled: tuple[int, Image.Image] = (0, self.card)

    def scrolled(self, lines: int) -> Image.Image:
        """The card with its scroll box rolled up by `lines` lines (wrapping: the lines that leave at the top come back at the bottom)."""
        if self._rolled[0] != lines:
            box = self.typing.box
            shift = lines * self.typing.line_h % (box[3] - box[1])
            image = self.card.copy()
            if shift:
                image.paste(ImageChops.offset(self.card.crop(box), 0, -shift), box[:2])
            self._rolled = (lines, image)
        return self._rolled[1]

    def typed(self, frame: ScreenFrame) -> str:
        """The text on the typing line of `frame`: its first `frame.typed` visible characters (`typed_text`)."""
        return typed_text(self.text, frame.lines, frame.typed, self.typing.prefix)


def screen_layers(state: dict) -> ScreenLayers:
    """`state` in layers, which `screen_frame` draws a frame from; with no scroll or typing they make `screen_state(state)` exactly."""
    return ScreenLayers(state)


def screen_frames(spec: dict, start: float) -> list[ScreenFrame]:
    """The frames of the screen segment `spec` that starts at `start` (local epoch, a whole second): what the recorder keeps, which is
    a frame whenever the screen changes and at least one every FLOOR_S, at most 2 a second (a tick is TICK_S). The reader reads the first of
    each second. A frame is at the start of each state (the cursor where it is), each of the states' `acts` (at `acts[k][0]` ticks from the
    state's start: a cursor `move`, a `type` or a `scroll`), each minute change of the taskbar's clock, then wherever more than FLOOR_S passes
    without one; in time order, all before `length_s`. The frame shows the cursor of the last `move`, the characters typed by the last
    `type` (none after a `scroll`, which starts the typing line afresh) and the lines scrolled by the last `scroll`. This is the one place
    that says where a frame is."""
    states, length = spec["states"], int(math.ceil(spec["length_s"]))
    starts = [s["from"] for s in states]
    marks = {2 * a for a in starts} | {2 * s["from"] + act[0] for s in states for act in s.get("acts", ())}
    marks |= {2 * sec for sec in range(1, length) if (int(start) + sec) % 60 == 0}
    out = []
    k, i, cursor, typed, lines = -1, 0, CURSOR_HOME, 0, 0
    for sec in with_floor([m * TICK_S for m in sorted(m for m in marks if 0 <= m < 2 * length)], length):
        now = bisect.bisect_right(starts, sec) - 1
        state = states[now]
        if now != k:
            k, i, cursor, typed, lines = now, 0, tuple(state.get("cursor", CURSOR_HOME)), 0, 0
        acts = state.get("acts", ())
        while i < len(acts) and state["from"] + acts[i][0] * TICK_S <= sec:
            kind, value = acts[i][1], acts[i][2:]
            if kind == "move":
                cursor = (value[0], value[1])
            elif kind == "type":
                typed = value[0]
            else:
                lines, typed = value[0], 0
            i += 1
        when = datetime.fromtimestamp(int(start) + int(sec))
        out.append(ScreenFrame(sec, k, (int(cursor[0]), int(cursor[1])), when.strftime("%H:%M"), when.strftime("%Y-%m-%d"), typed, lines))
    return out


def with_floor(times: list[float], length: float) -> list[float]:
    """`times` (sorted, seconds, before `length`) with a time added wherever more than FLOOR_S passes with no frame, from one frame to
    the next and to `length`: the recorder's `mpdecimate max=119`."""
    out: list[float] = []
    for sec in (*times, length):
        while out and sec - out[-1] > FLOOR_S:
            out.append(out[-1] + FLOOR_S)
        if sec < length:
            out.append(sec)
    return out


def screen_frame(layers: ScreenLayers, frame: ScreenFrame) -> Image.Image:
    """The state's `layers` as `frame` shows it: the card scrolled by the frame's lines, its typed text on the typing line, the
    decorations (accent, dialog, taskbar) pasted over both through their mask, then the taskbar's clock and date at the right and the
    cursor (a white arrow with a black outline, its tip at the frame's `cursor`) over everything."""
    image = layers.scrolled(frame.lines).copy()
    draw = ImageDraw.Draw(image)
    if frame.typed:
        typing = layers.typing
        draw.rectangle(typing.rect, fill=layers.bg)
        draw.text(typing.at, layers.typed(frame), fill=layers.ink, font=_cached_font(typing.size), stroke_width=typing.stroke, stroke_fill=layers.ink)
    for xy, deco, mask in layers.regions:
        image.paste(deco, xy, mask)
    stamp(image, frame)
    return image


def stamp(image: Image.Image, frame: ScreenFrame) -> None:
    """Draw the taskbar's clock and date at the right of `image` and the frame's cursor over everything."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.text((w - 16, h - TASKBAR_H + 3), frame.clock, fill=TASKBAR_INK, font=_cached_font(17), anchor="ra")
    draw.text((w - 16, h - TASKBAR_H + 21), frame.date, fill=TASKBAR_INK, font=_cached_font(15), anchor="ra")
    x, y = frame.cursor
    draw.polygon([(x + dx, y + dy) for dx, dy in CURSOR_ARROW], fill=(255, 255, 255), outline=(0, 0, 0))


def write_screen(path: Path, spec: dict, start: float, work: Path) -> None:
    """A screen segment that starts at `start` as a VFR video: a frame at each time of `screen_frames`, drawn from the state's layers (made
    once for the state) with the frame's scroll, typing, clock and cursor."""
    def frames():
        k, layers = -1, None
        for frame in screen_frames(spec, start):
            if frame.state != k:
                k, layers = frame.state, screen_layers(spec["states"][frame.state])
            yield frame.second, screen_frame(layers, frame)

    _write_video(path, frames(), spec["length_s"], SCREEN, SCREEN_CRF, work)
