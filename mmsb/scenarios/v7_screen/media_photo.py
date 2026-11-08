"""The phone's photos of the V7 week: `photo(spec) -> Image`, 1280x720, a `sign`, a `car`, an `object`, `room`, `food`, `plant` or
`street`.

    photo: `{"scene", "rgb": [..]|None, "word", "number", "box": [x0, y0, x1, y1] (fractions)|None, "seed"}`. `rgb` is the sign's fill,
    the car's body, or the colour of the scene's subject (an umbrella, a door, tomatoes, leaves, a shirt).

A needle photo is a `sign` with `rgb` NEEDLE_RGB["sign"] (its `box` 22 to 34% of the photo, or none: `sign_box` makes one) or a `car`
with `rgb` NEEDLE_RGB["car"]; every other photo's colour is one of DISTRACTOR_RGB's. The reader hands a decider pictures of at most
640x360 (its `max_px`), so the needles are sized for that; the shades keep 3% or more of the picture through that scaling and the
JPEG (tests/test_v7_media.py). Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from mmsb.scenarios.v7_screen.media_common import (
    DISTRACTOR_RGB, NEEDLE_RGB, SAFE, _clear_of_needles, _fit, _gradient, _ink, _px, _rgb, _rng,
)

PHOTO = (1280, 720)

SIGN_BOX = (0.22, 0.34)  # a needle sign's share of the photo
CAR_BOX = (0.04, 0.18, 0.96, 0.86)  # a car's box: its body is about 34% of the photo
OBJECTS = ("umbrella", "bag", "shirt")


def photo(spec: dict) -> Image.Image:
    """A phone photo of `spec["scene"]`: a `sign` (fill `rgb`, `word`, `number`, in `box`; 22 to 34% of the photo for a needle), a
    `car` (body `rgb`, in `box`; the body of a red one is 30% or more of the photo), or an `object`, `room`, `food`, `plant` or `street`
    with `rgb` the colour of its subject. A sign's fill and a car's body may be the needle's own shade; no other colour may be
    within SHADE_GAP of a needle's."""
    scene = spec["scene"]
    seed = spec.get("seed", 0)
    rgb = spec.get("rgb")
    if scene == "sign":
        return _sign_photo(spec, _rgb(rgb or NEEDLE_RGB["sign"]))
    if scene == "car":
        return _car_photo(spec, _rgb(rgb or NEEDLE_RGB["car"]))
    if rgb is not None:
        _clear_of_needles(rgb, f"the {scene} photo's colour", "phone")
    return {"object": _object_photo, "room": _room_photo, "food": _food_photo, "plant": _plant_photo,
            "street": _street_photo}[scene](seed, None if rgb is None else _rgb(rgb))


def _backdrop(r: random.Random) -> Image.Image:
    """A wall, a fence or a sky: muted, and no needle's shade."""
    kind = r.choice(("wall", "sky", "fence"))
    if kind == "sky":
        return _gradient(PHOTO, (150, 185, 225), (215, 228, 240))
    image = _gradient(PHOTO, *r.choice((((196, 188, 176), (170, 162, 150)), ((200, 205, 210), (176, 182, 190)), ((215, 205, 185), (190, 178, 158)))))
    if kind == "fence":
        draw = ImageDraw.Draw(image)
        for x in range(0, PHOTO[0], 64):
            draw.rectangle([x, 0, x + 6, PHOTO[1]], fill=(150, 142, 130))
    return image


def sign_box(spec: dict) -> tuple[float, ...]:
    """A sign's box, `(x0, y0, x1, y1)` as fractions of the photo: the spec's own, or one its seed makes, 22 to 34% of the photo."""
    if spec.get("box"):
        return tuple(spec["box"])
    r = _rng(spec.get("seed", 0), "sign-box")
    height = r.uniform(0.62, 0.85)
    width = r.uniform(*SIGN_BOX) / height
    x0, y0 = r.uniform(0.04, 0.96 - width), r.uniform(0.03, 0.95 - height)
    return x0, y0, x0 + width, y0 + height


def _sign_photo(spec: dict, fill: tuple[int, int, int]) -> Image.Image:
    _clear_of_needles(fill, "the sign's fill", "phone", allow="sign")
    image = _backdrop(_rng(spec.get("seed", 0), "sign"))
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = _px(PHOTO, *sign_box(spec))
    ink = _ink(fill)
    if y1 < PHOTO[1] - 4:
        draw.rectangle([(x0 + x1) // 2 - 12, y1, (x0 + x1) // 2 + 12, PHOTO[1]], fill=(95, 95, 100))
    draw.rounded_rectangle([x0 + 10, y0 + 10, x1 + 10, y1 + 10], radius=28, fill=(60, 60, 64))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=28, fill=fill, outline=ink, width=10)
    cx, height = (x0 + x1) // 2, y1 - y0
    _fit(draw, (cx, y0 + height * 0.24), str(spec.get("word", "GATE")), ink, int(height * 0.18), (x1 - x0) * 0.8)
    _fit(draw, (cx, y0 + height * 0.64), str(spec.get("number", "47")), ink, int(height * 0.46), (x1 - x0) * 0.8)
    return image


def _car_photo(spec: dict, body: tuple[int, int, int]) -> Image.Image:
    _clear_of_needles(body, "the car's body", "phone", allow="car")
    r = _rng(spec.get("seed", 0), "car")
    image = _backdrop(r)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.86, 1, 1), fill=(88, 88, 94))
    x0, y0, x1, y1 = _px(PHOTO, *(spec.get("box") or CAR_BOX))
    w, h = x1 - x0, y1 - y0
    draw.ellipse([x0 - 10, y1 - h // 14, x1 + 10, y1 + h // 10], fill=(70, 70, 76))
    draw.rounded_rectangle([x0, y0 + h * 38 // 100, x1, y0 + h * 88 // 100], radius=h // 10, fill=body)
    cabin = [(12, 40), (26, 0), (70, 0), (86, 40)]  # (percent of the width, percent of the height) from the car's top left
    draw.polygon([(x0 + w * px // 100, y0 + h * py // 100) for px, py in cabin], fill=body)
    window = (35, 50, 70)
    for pane in ([(17, 38), (28, 8), (48, 8), (48, 38)], [(52, 38), (52, 8), (67, 8), (80, 38)]):  # the two windows
        draw.polygon([(x0 + w * px // 100, y0 + h * py // 100) for px, py in pane], fill=window)
    for x in (0.2, 0.8):
        cx, cy, radius = x0 + int(x * w), y0 + h * 88 // 100, h * 14 // 100
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(20, 20, 20))
        draw.ellipse([cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2], fill=(170, 170, 175))
    draw.rectangle([x1 - w * 6 // 100, y0 + h * 56 // 100, x1 - w * 2 // 100, y0 + h * 63 // 100], fill=(250, 240, 190))
    draw.rectangle([x0 + w * 45 // 100, y0 + h * 54 // 100, x0 + w * 55 // 100, y0 + h * 57 // 100], fill=(60, 60, 60))
    if r.random() < 0.5:  # a plate
        draw.rectangle([x0 + w * 40 // 100, y0 + h * 72 // 100, x0 + w * 60 // 100, y0 + h * 80 // 100], fill=(240, 240, 235))
    return image


def _object_photo(seed: int, rgb) -> Image.Image:
    r = _rng(seed, "object")
    image = _backdrop(r)
    draw = ImageDraw.Draw(image)
    colour = rgb or r.choice(SAFE)
    what = r.choice(OBJECTS)
    w, h = PHOTO
    draw.rectangle(_px(PHOTO, 0, 0.84, 1, 1), fill=r.choice(((150, 110, 75), (120, 90, 70), (180, 150, 110))))
    cx = r.randint(w * 35 // 100, w * 65 // 100)
    if what == "umbrella":
        draw.pieslice([cx - 300, h * 8 // 100, cx + 300, h * 8 // 100 + 600], 180, 360, fill=colour)
        draw.line([cx, h * 8 // 100 + 300, cx, h * 80 // 100], fill=(60, 60, 64), width=14)
        draw.arc([cx - 50, h * 70 // 100, cx + 10, h * 90 // 100], 0, 180, fill=(60, 60, 64), width=14)
        for k in (-200, -100, 100, 200):
            draw.line([cx, h * 8 // 100, cx + k, h * 8 // 100 + 300 - abs(k) // 6], fill=(40, 40, 44), width=4)
    elif what == "bag":
        draw.arc([cx - 130, h * 12 // 100, cx + 130, h * 52 // 100], 180, 360, fill=(60, 60, 64), width=16)
        draw.rounded_rectangle([cx - 240, h * 32 // 100, cx + 240, h * 82 // 100], radius=36, fill=colour)
        draw.rectangle([cx - 240, h * 44 // 100, cx + 240, h * 47 // 100], fill=(240, 240, 240))
    else:
        outline = [(-120, 12), (-330, 30), (-260, 48), (-180, 40), (-180, 82), (180, 82), (180, 40), (260, 48), (330, 30), (120, 12)]  # (dx, percent of the height)
        draw.polygon([(cx + dx, h * py // 100) for dx, py in outline], fill=colour)
        draw.ellipse([cx - 70, h * 7 // 100, cx + 70, h * 22 // 100], fill=(225, 185, 150))
    return image


def _room_photo(seed: int, rgb) -> Image.Image:
    """A room: a sofa, a window, a picture, a lamp and a door (of `rgb`, or wood)."""
    r = _rng(seed, "room")
    image = Image.new("RGB", PHOTO, r.choice(((235, 228, 210), (222, 228, 238), (238, 232, 240), (236, 236, 236))))
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.68, 1, 1), fill=r.choice(((150, 110, 75), (120, 90, 70), (180, 150, 110))))
    draw.rectangle(_px(PHOTO, 0.74, 0.06, 0.92, 0.70), fill=rgb or (150, 110, 75), outline=(90, 70, 60), width=6)
    draw.ellipse(_px(PHOTO, 0.755, 0.38, 0.775, 0.43), fill=(230, 215, 150))
    draw.rounded_rectangle(_px(PHOTO, 0.06, 0.45, 0.58, 0.78), radius=40, fill=r.choice(((70, 90, 140), (110, 110, 115), (150, 120, 90))))
    draw.rectangle(_px(PHOTO, 0.10, 0.10, 0.34, 0.34), fill=(90, 70, 60))
    draw.rectangle(_px(PHOTO, 0.11, 0.12, 0.33, 0.32), fill=(205, 195, 175))
    draw.rectangle(_px(PHOTO, 0.40, 0.14, 0.62, 0.48), fill=(205, 228, 250), outline=(250, 250, 250), width=12)
    draw.ellipse(_px(PHOTO, 0.64, 0.50, 0.68, 0.62), fill=(240, 215, 150))
    return image


def _food_photo(seed: int, rgb) -> Image.Image:
    """A plate on a table: round items of `rgb` (tomatoes) among others."""
    r = _rng(seed, "food")
    image = Image.new("RGB", PHOTO, r.choice(((160, 110, 70), (190, 160, 120), (120, 90, 60))))
    draw = ImageDraw.Draw(image)
    draw.ellipse(_px(PHOTO, 0.25, 0.06, 0.75, 0.94), fill=(250, 250, 250), outline=(225, 225, 225), width=14)
    palette = ((225, 150, 40), (130, 85, 40), (240, 225, 150), (250, 200, 90))
    for _ in range(r.randint(4, 7)):
        x, y, radius = r.randint(420, 860), r.randint(160, 560), r.randint(40, 85)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=rgb or r.choice(palette) if r.random() < 0.6 else r.choice(palette))
    for _ in range(6):
        x, y = r.randint(450, 830), r.randint(200, 520)
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(70, 150, 70) if rgb is None else (240, 225, 150))
    return image


def _plant_photo(seed: int, rgb) -> Image.Image:
    """A potted plant, its leaves `rgb` (a green that is not the sign's)."""
    r = _rng(seed, "plant")
    leaf = rgb or DISTRACTOR_RGB["sign"]["plant"]
    image = Image.new("RGB", PHOTO, (238, 234, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.86, 1, 1), fill=(165, 130, 95))
    draw.polygon([_px(PHOTO, 0.40, 0.60), _px(PHOTO, 0.60, 0.60), _px(PHOTO, 0.56, 0.93), _px(PHOTO, 0.44, 0.93)], fill=(190, 100, 60))
    for _ in range(r.randint(8, 12)):
        x, y = r.randint(420, 860), r.randint(80, 420)
        shade = tuple(max(0, min(255, c + r.randint(-12, 12))) for c in leaf)
        draw.ellipse([x - 90, y - 34, x + 90, y + 34], fill=shade)
    return image


def _street_photo(seed: int, rgb) -> Image.Image:
    """A street: houses, a road, parked cars in other colours, a few small signs (never the sign's green) and passers-by, one
    of them in `rgb` (a green shirt)."""
    r = _rng(seed, "street")
    w, h = PHOTO
    image = _gradient(PHOTO, (170, 200, 230), (225, 232, 240))
    draw = ImageDraw.Draw(image)
    for k in range(5):
        x0 = k * w // 5
        wall = r.choice(((210, 190, 150), (130, 150, 175), (190, 190, 195), (200, 170, 150)))
        draw.rectangle([x0, h * r.randint(12, 30) // 100, x0 + w // 5 - 6, h * 66 // 100], fill=wall)
        for wx in range(x0 + 24, x0 + w // 5 - 40, 62):
            draw.rectangle([wx, h * 38 // 100, wx + 34, h * 38 // 100 + 54], fill=(80, 100, 130))
    draw.rectangle(_px(PHOTO, 0, 0.66, 1, 0.74), fill=(175, 175, 178))
    draw.rectangle(_px(PHOTO, 0, 0.74, 1, 1), fill=(85, 85, 90))
    for k in range(4):
        draw.rectangle(_px(PHOTO, 0.06 + k * 0.25, 0.88, 0.14 + k * 0.25, 0.90), fill=(235, 235, 235))
    for _ in range(r.randint(1, 2)):  # a parked car, in a colour that is no needle's
        x = r.randint(60, 900)
        body = _rgb(r.choice((DISTRACTOR_RGB["car"]["blue"], DISTRACTOR_RGB["car"]["grey"], DISTRACTOR_RGB["car"]["white"],
                              DISTRACTOR_RGB["sign"]["car"])))
        draw.rounded_rectangle([x, h * 70 // 100, x + 260, h * 80 // 100], radius=14, fill=body)
        draw.polygon([(x + 40, h * 70 // 100), (x + 70, h * 63 // 100), (x + 190, h * 63 // 100), (x + 220, h * 70 // 100)], fill=body)
        for cx in (x + 55, x + 205):
            draw.ellipse([cx - 22, h * 77 // 100, cx + 22, h * 77 // 100 + 44], fill=(20, 20, 20))
    for k in range(r.randint(2, 4)):  # small signs: a plate on a post
        x, y = r.randint(40, w - 120), r.randint(h * 40 // 100, h * 60 // 100)
        fill = _rgb(r.choice([DISTRACTOR_RGB["sign"][c] for c in ("yellow", "white", "blue", "grey")]))
        draw.rectangle([x + 18, y, x + 24, y + 150], fill=(90, 90, 95))
        draw.rounded_rectangle([x, y, x + 44, y + 40], radius=5, fill=fill, outline=_ink(fill), width=3)
    for k in range(r.randint(1, 3)):  # a passer-by
        x = r.randint(80, w - 120)
        shirt = rgb if (k == 0 and rgb) else r.choice(SAFE)
        draw.rectangle([x, h * 66 // 100, x + 36, h * 72 // 100 + 64], fill=(50, 60, 100))
        draw.rectangle([x - 4, h * 58 // 100, x + 40, h * 66 // 100 + 24], fill=shirt)
        draw.ellipse([x + 4, h * 52 // 100, x + 32, h * 59 // 100], fill=(225, 185, 150))
    return image
