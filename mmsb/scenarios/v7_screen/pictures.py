"""The phone's pictures that the files are made of: a wall or hedge photo, the card and stamp of a phone clip, and the portrait clip a
phone makes (frames coded sideways and a display matrix, which a player turns upright). `photo_image` draws a photo spec and
`photo_needle` says which needle a photo is. Moved from haystack.py of the repository at bb498a6 and, for `portrait_video` and its
helpers, from tests/day.py of Multimem at fe0ece5."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mmsb.scenarios.v7_screen.constants import font
from mmsb.scenarios.v7_screen.media_common import NEEDLE_RGB, _clear_of_needles, _gradient, ffmpeg_exe
from mmsb.scenarios.v7_screen.media_photo import PHOTO, photo

CLIP = (540, 960)


def box(size: tuple[int, int], *fractions: float) -> tuple[int, ...]:
    """Pixel corners from fractions of the picture's width and height: (x0, y0, x1, y1, ...)."""
    return tuple(int(f * size[i % 2]) for i, f in enumerate(fractions))


def wall_photo(spec: dict) -> Image.Image:
    """A facade or a hedge across 40 to 70% of the photo (`rgb`: a red wall, a green hedge), with its windows and door: more of a
    needle's named colour than the needle itself has (a sign is a quarter of the photo, a red car a third), and no needle's shade."""
    _clear_of_needles(spec["rgb"], "the wall's colour", "phone")
    r = random.Random(spec["seed"])
    w, h = PHOTO
    image = _gradient(PHOTO, (150, 185, 225), (215, 228, 240))
    draw = ImageDraw.Draw(image)
    width, height = r.uniform(0.75, 0.96), r.uniform(0.62, 0.86)
    left, top = r.uniform(0.02, 0.98 - width), r.uniform(0.04, 0.96 - height)
    x0, y0, x1, y1 = int(w * left), int(h * top), int(w * (left + width)), int(h * (top + height))
    draw.rectangle([x0, y0, x1, y1], fill=tuple(spec["rgb"]), outline=(90, 80, 75), width=6)
    columns = r.randint(3, 5)
    for row in range(2):
        for k in range(columns):
            wx = x0 + int((x1 - x0) * (0.06 + 0.88 * k / columns))
            wy = y0 + int((y1 - y0) * (0.08 + 0.36 * row))
            draw.rectangle([wx, wy, wx + int((x1 - x0) * 0.06), wy + int((y1 - y0) * 0.2)], fill=(205, 228, 250), outline=(245, 245, 245), width=5)
    mid = (x0 + x1) // 2
    draw.rectangle([mid - int((x1 - x0) * 0.06), y1 - int((y1 - y0) * 0.3), mid + int((x1 - x0) * 0.06), y1], fill=(90, 70, 60))
    return image


def photo_image(spec: dict) -> Image.Image:
    return wall_photo(spec) if spec["scene"] == "wall" else photo(spec)


def stamp(image: Image.Image, text: str, bar: int = 60) -> Image.Image:
    """The camera's time stamp, along the bottom: it makes every clip different, and `remember` keeps a recording once."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle([0, h - bar, w, h], fill=(0, 0, 0))
    draw.text((16, h - bar + bar // 6), text, fill="white", font=font(bar * 2 // 3))
    return image


def clip_card(scene: str, seed: int, when: str) -> Image.Image:
    """A portrait frame of a phone clip: a street, a dog or a shop front."""
    r = random.Random(seed)
    image = Image.new("RGB", CLIP, (170, 200, 230))
    draw = ImageDraw.Draw(image)
    w, h = CLIP
    if scene == "street":
        draw.rectangle(box(CLIP, 0, .65, 1, 1), fill=(85, 85, 90))
        for x in (0, .35, .7):
            draw.rectangle(box(CLIP, x, .15, x + .3, .65), fill=r.choice(((210, 190, 150), (130, 150, 175), (190, 190, 195))))
        for k in range(4):
            draw.rectangle(box(CLIP, .06 + k * .25, .78, .14 + k * .25, .80), fill=(235, 235, 235))
        draw.rectangle(box(CLIP, .45, .55, .53, .75), fill=(30, 50, 110))
        draw.ellipse(box(CLIP, .45, .50, .53, .56), fill=(220, 180, 150))
    elif scene == "dog":
        draw.rectangle(box(CLIP, 0, 0, 1, .65), fill=(225, 215, 195))
        draw.rectangle(box(CLIP, 0, .65, 1, 1), fill=(175, 150, 120))
        draw.ellipse(box(CLIP, .15, .45, .85, .75), fill=(140, 100, 60))
        draw.ellipse(box(CLIP, .55, .28, .90, .52), fill=(140, 100, 60))
        draw.ellipse(box(CLIP, .55, .26, .66, .40), fill=(100, 70, 40))
        draw.ellipse(box(CLIP, .74, .34, .78, .37), fill=(20, 20, 20))
        for x in (.2, .35, .6, .72):
            draw.rectangle(box(CLIP, x, .70, x + .06, .86), fill=(120, 85, 50))
    else:
        draw.rectangle(box(CLIP, 0, 0, 1, 1), fill=(60, 70, 100))
        for k in range(8):
            draw.rectangle(box(CLIP, k / 8, .12, (k + 1) / 8, .30), fill=(30, 80, 190) if k % 2 else (245, 245, 245))
        draw.rectangle(box(CLIP, .08, .36, .92, .66), fill=(240, 230, 170))
        draw.rectangle(box(CLIP, .38, .70, .62, .96), fill=(110, 80, 50))
        draw.text((w // 2, h * 6 // 100), r.choice(("OPEN", "CAFE", "BOOKS")), fill="white", font=font(70), anchor="mm")
    return stamp(image, when)


def photo_needle(spec: dict) -> str | None:
    """The needle a photo spec is: `sign`, `car`, or None."""
    if spec["scene"] in ("sign", "car") and tuple(spec["rgb"] or ()) == NEEDLE_RGB[spec["scene"]]:
        return spec["scene"]
    return None


def same(a: Image.Image, b: Image.Image, limit: float = 25.0) -> bool:
    """Whether two pictures show the same thing: their sizes match and their 48x48 thumbnails differ by under `limit`
    levels on average."""
    def thumb(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("RGB").resize((48, 48)), dtype=float)
    return a.size == b.size and float(np.abs(thumb(a) - thumb(b)).mean()) < limit


def ff(*args: str) -> None:
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"ffmpeg {' '.join(args)[:200]} failed: {done.stderr.strip()[-400:]}")


def upright_frame(path: Path, folder: Path) -> Image.Image:
    """The first frame of `path` as ffmpeg itself turns it (it applies the display matrix)."""
    ff("-i", str(path), "-frames:v", "1", str(folder / "first.png"))
    with Image.open(folder / "first.png") as frame:
        return frame.convert("RGB")


def portrait_video(path: Path, upright: Image.Image, work: Path, *, rotation: int = -90, codec: str = "libx264",
                   pix_fmt: str = "yuv420p", seconds: int = 2, fps: int = 5, tags: tuple[str, ...] = ()) -> None:
    """A phone's portrait clip: the frames coded sideways and a display matrix of `rotation` degrees, so a player turns
    them upright. `tags` (ffmpeg output options, e.g. `-metadata creation_time=...`) are on both passes: the copy keeps no
    creation time of its own. Checked against ffmpeg's own decode, which applies the matrix."""
    turn = Image.Transpose.ROTATE_90 if rotation == -90 else Image.Transpose.ROTATE_270
    upright.transpose(turn).save(work / "coded.png")
    ff("-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", str(work / "coded.png"), "-c:v", codec,
       *(["-x265-params", "log-level=error"] if codec == "libx265" else []), "-pix_fmt", pix_fmt, *tags,
       str(work / ("coded" + path.suffix)))
    ff("-display_rotation", str(rotation), "-i", str(work / ("coded" + path.suffix)), "-c", "copy", *tags, str(path))  # (a copy drops creation_time)
    first = upright_frame(path, work)
    assert same(first, upright, 40), f"{path.name}: a display matrix of {rotation} does not turn the frames upright"
