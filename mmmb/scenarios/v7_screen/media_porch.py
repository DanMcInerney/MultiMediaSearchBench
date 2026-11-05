"""The front door camera of the V7 week: `porch_frame(spec, second, stamp)` is one second, 640x360, a scene and its events;
`write_porch` writes the clip as x264 at 1 frame a second.

    porch clip: `{"length_s", "events": [{"kind", "from", "to", "rgb"?, "drop_at"?, "x"?, "scale"?, "bag"?}], "parcel": [from, to]|None, "box"?, "seed"}`

A parcel is the clip's `parcel: [drop_at, pickup]`, with the courier's `drop_at` the same second; a courier whose `drop_at` is None (a
distractor) carries his box to the door and back out and leaves nothing on the step. A parcel is not always left in one place: it is
left at `box` (`parcel_box` draws one, on the step by the door), and `BOX_AREA` is the union of every place it can take, as fractions
of the frame. The distractors of the parcel's colour are in those places too: a `visitor` (close to the camera, in a brown coat, with a
carrier bag or without) has more brown in `BOX_AREA` than a parcel has, and the door of the porch is not brown (a brown door would put
brown in the parcel's place in every frame). Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from mmmb.scenarios.v7_screen.constants import font
from mmmb.scenarios.v7_screen.media_common import DISTRACTOR_RGB, NEEDLE_RGB, _clear_of_needles, _px, _rgb, _rng, _write_video

PORCH = (640, 360)

PORCH_CRF = 28  # capture.py's camera preset

BOX = (0.13, 0.43, 0.38, 0.69)  # the parcel's front face on the step, as fractions of the frame: 160x95 px, 6.6% of it
BOX_W, BOX_H = 0.25, 0.26  # its size
TOP_STRIP = 26  # px: its lighter top face
BOX_X0, BOX_Y1 = (0.04, 0.20), (0.67, 0.70)  # where it may be: the range of its left edge and of its bottom edge, on the step by the door
BOX_AREA = (BOX_X0[0], BOX_Y1[0] - BOX_H - TOP_STRIP / 360, BOX_X0[1] + BOX_W + 12 / 640, BOX_Y1[1])
# ^ every place it can be (x0, y0, x1, y1) as fractions of the frame: the union of its positions, face and top face
WALLS = ((205, 200, 190), (196, 198, 200), (210, 205, 195), (190, 196, 190))
JACKETS = ((40, 60, 120), (60, 60, 70), (200, 60, 70), (230, 230, 235), (60, 120, 90), (120, 60, 140), (240, 200, 40))


def parcel_box(r: random.Random) -> list[float]:
    """Where a parcel is left on the step: the face's `[x0, y0, x1, y1]` as fractions of the frame, one of the places of BOX_AREA, from `r`.
    A parcel stays where it was left, so the planner gives every clip that shows it the same one (`box` in the clip's spec)."""
    x0, y1 = round(r.uniform(*BOX_X0), 3), round(r.uniform(*BOX_Y1), 3)
    return [x0, round(y1 - BOX_H, 3), round(x0 + BOX_W, 3), y1]


def porch_frame(spec: dict, second: float, stamp: str) -> Image.Image:
    """Second `second` of the clip `spec`: the door, the step and the path, the clip's own props (a doormat, a planter, a bin), the
    events that are on at that second, then (so that no light touches it) the parcel on the step, a cardboard box with tape and a
    label in the parcel needle's shade, and the camera's time `stamp` along the bottom. A `courier` walks in carrying a box (in
    the shadow, a darker brown), puts it down at `drop_at` (the clip's `parcel` then shows it on the step) and walks out; with
    `"drop_at": None` he never puts it down: he carries it to the door and back out, and the step is empty in every frame (a
    distractor: nothing is delivered). A `visitor` (`x`, `scale` 3, `bag`) stands close to the camera in a brown coat, a third of the
    frame or more: more brown, in the parcel's place, than a parcel has. A `walker`, `cat`, `dog` or `car` crosses; an
    `owner_pickup` is the owner at the door, lifting the parcel at the `parcel`'s end; a `light` dims or brightens the scene."""
    image = _porch_base(spec.get("seed", 0)).copy()
    draw = ImageDraw.Draw(image)
    w, h = PORCH
    parcel = spec.get("parcel")
    for event in spec.get("events", ()):
        if event["kind"] == "light" and event["from"] <= second < event["to"]:
            rgb = _rgb(event.get("rgb") or (25, 30, 55))
            _clear_of_needles(rgb, "a light event's tint", "porch")
            image = Image.blend(image, Image.new("RGB", PORCH, rgb), 0.3)
            draw = ImageDraw.Draw(image)
    for event in spec.get("events", ()):
        if event["from"] <= second <= event["to"] and event["kind"] != "light":
            _event(draw, event, second, parcel, _rng(spec.get("seed", 0), f"{event['kind']}-{event['from']}"))
    if parcel and parcel[0] <= second < parcel[1]:
        _parcel(draw, spec.get("box"))
    draw.rectangle([0, h - 26, w, h], fill=(0, 0, 0))
    draw.text((12, h - 24), stamp, fill="white", font=font(18))
    return image


@lru_cache(maxsize=16)
def _porch_base(seed: int) -> Image.Image:
    """The door, step, path and props of a clip: a function of its seed."""
    r = _rng(seed, "porch")
    w, h = PORCH
    image = Image.new("RGB", PORCH, r.choice(WALLS))
    draw = ImageDraw.Draw(image)
    door = _rgb(DISTRACTOR_RGB["parcel"]["door"])
    draw.rectangle([36, 0, 236, 216], fill=(90, 90, 95))
    draw.rectangle([42, 6, 230, 216], fill=door)
    for x0, y0, x1, y1 in ((56, 20, 128, 100), (146, 20, 218, 100), (56, 116, 128, 200), (146, 116, 218, 200)):
        draw.rectangle([x0, y0, x1, y1], outline=(30, 45, 75), width=3)
    draw.ellipse([204, 118, 214, 128], fill=(215, 205, 160))
    draw.rectangle([0, 216, 300, 256], fill=(170, 168, 160))
    draw.rectangle([0, 256, 300, 276], fill=(140, 138, 130))
    draw.rectangle([0, 276, w, h], fill=(150, 148, 140))
    for x in (60, 190, 330, 470):
        draw.line([x, 276, x - 30, h], fill=(120, 118, 110), width=3)
    draw.rectangle([300, 216, w, 276], fill=(80, 125, 70))
    if r.random() < 0.8:  # a doormat
        draw.rectangle([70, 240, 200, 252], fill=_rgb(DISTRACTOR_RGB["parcel"]["mat"]))
    if r.random() < 0.7:  # a planter
        draw.polygon([(250, 196), (292, 196), (286, 250), (256, 250)], fill=_rgb(DISTRACTOR_RGB["parcel"]["planter"]))
        draw.ellipse([240, 150, 302, 206], fill=(60, 130, 60))
    if r.random() < 0.6:  # a bin
        draw.rectangle([330, 150, 392, 262], fill=(60, 90, 70))
        draw.rectangle([326, 142, 396, 156], fill=(45, 70, 55))
    return image


def _person(draw: ImageDraw.ImageDraw, x: int, feet: int, jacket, scale: float = 1.0, carrying=None) -> None:
    """A person standing at `x` with feet at `feet`: head, jacket, trousers; `carrying` is the colour of a box held in front."""
    body, head = int(46 * scale), int(12 * scale)
    draw.rectangle([x - body // 2, feet - int(46 * scale), x - 2, feet], fill=(45, 50, 70))
    draw.rectangle([x + 2, feet - int(46 * scale), x + body // 2, feet], fill=(45, 50, 70))
    draw.rectangle([x - body // 2, feet - int(100 * scale), x + body // 2, feet - int(44 * scale)], fill=jacket)
    draw.ellipse([x - head, feet - int(130 * scale), x + head, feet - int(104 * scale)], fill=(225, 185, 150))
    if carrying:
        draw.rectangle([x - int(36 * scale), feet - int(88 * scale), x + int(10 * scale), feet - int(58 * scale)], fill=carrying)
        draw.rectangle([x - int(36 * scale), feet - int(88 * scale), x + int(10 * scale), feet - int(82 * scale)], fill=(160, 130, 95))


def _event(draw: ImageDraw.ImageDraw, event: dict, second: float, parcel, r: random.Random) -> None:
    kind, a, b = event["kind"], event["from"], event["to"]
    w, h = PORCH
    u = (second - a) / max(b - a, 1e-9)
    rgb = event.get("rgb")
    if rgb is not None:
        _clear_of_needles(rgb, f"a {kind} event's colour", "porch")
        rgb = _rgb(rgb)
    flip = r.random() < 0.5
    across = (1 - u) if flip else u
    if kind == "courier":
        jacket = rgb or _rgb(DISTRACTOR_RGB["parcel"]["courier"])
        carried = _rgb(DISTRACTOR_RGB["parcel"]["carried"])
        if "drop_at" in event and event["drop_at"] is None:  # he keeps the box: in to the door by the middle of his time, and out with it
            x = int(w + 40 + (330 - w - 40) * (1 - abs(2 * u - 1)))
            _person(draw, x, 300, jacket, carrying=carried)
            return
        drop = event.get("drop_at", (a + b) / 2)
        if second < drop:
            x = int(w + 40 + (330 - w - 40) * (second - a) / max(drop - a, 1e-9))
            _person(draw, x, 300, jacket, carrying=carried)
        else:
            x = int(330 + (w + 40 - 330) * (second - drop) / max(b - drop, 1e-9))
            _person(draw, x, 300, jacket)
    elif kind == "walker":
        _person(draw, int(-30 + (w + 60) * across), 330, rgb or r.choice(JACKETS), 0.85)
    elif kind == "visitor":  # close to the camera, in a brown coat, waiting at the door (with a carrier bag, or without): more brown than any parcel
        scale = float(event.get("scale", 3.0))
        _person(draw, int(event.get("x", 190) + 5 * math.sin(second)), int(130 * scale + 14), rgb or _rgb(DISTRACTOR_RGB["parcel"]["courier"]), scale,
                carrying=_rgb(DISTRACTOR_RGB["parcel"]["carried"]) if event.get("bag") else None)
    elif kind == "cat":
        x = int(-60 + (w + 120) * across)
        fur = rgb or r.choice(((60, 60, 65), (235, 235, 230), (140, 140, 145), _rgb(DISTRACTOR_RGB["parcel"]["cat"])))
        draw.ellipse([x, 296, x + 56, 328], fill=fur)
        draw.polygon([(x + 38, 300), (x + 42, 284), (x + 50, 300)], fill=fur)
        draw.line([x, 310, x - 22, 296], fill=fur, width=5)
    elif kind == "dog":
        x = int(-90 + (w + 180) * across)
        fur = rgb or _rgb(DISTRACTOR_RGB["parcel"]["dog"])
        draw.ellipse([x, 286, x + 90, 322], fill=fur)
        draw.ellipse([x + 70 if not flip else x - 22, 276, x + 112 if not flip else x + 20, 306], fill=fur)
        for lx in (x + 8, x + 66):
            draw.rectangle([lx, 314, lx + 10, 340], fill=fur)
    elif kind == "car":
        x = int(-170 + (w + 340) * across)
        body = rgb or r.choice(((30, 60, 200), (245, 245, 245), (130, 130, 135), (240, 20, 20), (240, 210, 20)))
        draw.rounded_rectangle([x, 290, x + 160, 330], radius=10, fill=body)
        draw.polygon([(x + 28, 290), (x + 46, 266), (x + 118, 266), (x + 136, 290)], fill=body)
        for cx in (x + 36, x + 124):
            draw.ellipse([cx - 14, 316, cx + 14, 344], fill=(20, 20, 20))
    elif kind == "owner_pickup":
        jacket = rgb or r.choice(JACKETS)
        lifted = bool(parcel) and second >= parcel[1]
        _person(draw, 300, 262, jacket, 1.1, carrying=_rgb(DISTRACTOR_RGB["parcel"]["carried"]) if lifted else None)
    else:
        raise ValueError(f"a porch event is a courier, walker, visitor, cat, dog, car, owner_pickup or light, not {kind!r}")


def _parcel(draw: ImageDraw.ImageDraw, box=None) -> None:
    """The parcel on the step: its front face in the needle's shade with a strip of tape and a label, its top face lighter. `box` is
    where (BOX by default; a `ValueError` when it is not one of the places of BOX_AREA)."""
    if box is not None:
        bx0, by0, bx1, by1 = box
        if not (BOX_X0[0] <= bx0 <= BOX_X0[1] and BOX_Y1[0] <= by1 <= BOX_Y1[1] and abs(bx1 - bx0 - BOX_W) < 1e-6 and abs(by1 - by0 - BOX_H) < 1e-6):
            raise ValueError(f"a parcel's box {list(box)} is not one of the places of BOX_AREA: left edge in {BOX_X0}, bottom edge in {BOX_Y1}, {BOX_W}x{BOX_H}")
    x0, y0, x1, y1 = _px(PORCH, *(box or BOX))
    draw.polygon([(x0, y0), (x0 + 12, y0 - TOP_STRIP), (x1 + 12, y0 - TOP_STRIP), (x1, y0)], fill=(212, 170, 120))
    draw.rectangle([x0, y0, x1, y1], fill=NEEDLE_RGB["parcel"], outline=(150, 108, 62), width=2)
    mid = (x0 + x1) // 2
    draw.rectangle([mid - 7, y0 - TOP_STRIP + 2, mid + 7, y1], fill=(232, 214, 168))
    draw.rectangle([x0 + 14, y1 - 42, x0 + 62, y1 - 12], fill=(246, 246, 246))
    draw.line([x0 + 20, y1 - 32, x0 + 56, y1 - 32], fill=(60, 60, 60), width=2)
    draw.line([x0 + 20, y1 - 22, x0 + 46, y1 - 22], fill=(60, 60, 60), width=2)


def _stamp_text(stamp: str, second: float) -> str:
    """The camera's time at `second` of a clip that starts at `stamp` (`2026-10-28 15:35:44`; any other text is kept)."""
    try:
        return (datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stamp


def write_porch(path: Path, spec: dict, stamp: str, work: Path) -> None:
    """A porch clip, one frame a second, its stamp running from `stamp`."""
    frames = ((s, porch_frame(spec, s, _stamp_text(stamp, s))) for s in range(int(spec["length_s"])))
    _write_video(path, frames, spec["length_s"], PORCH, PORCH_CRF, work)
