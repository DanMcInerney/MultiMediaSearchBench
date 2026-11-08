"""Ported from tests/test_bench_rules.py of MultimediaSearch at fe0ece5 (plan 7.4, oracle 4), in that file's order: the six tests of the fake decider's
`shade:R,G,B` (the other thirteen are the grader's, tests/test_grade.py). The rules the benchmark's harder haystack leans on, pinned with no agent and
no daemon: a part of a frame in one shade is found after JPEG and video coding, and not found in a shade 60 away. They are tests/mm's because the
fake decider is MultimediaSearch's (tests/mm/fake_decider.py, copied) and imports its facts (`mms.decider_d1.ABOUT`). Moved by script, unchanged."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("mms")

import av  # noqa: E402
import day  # noqa: E402
import fake_decider  # noqa: E402
from PIL import Image  # noqa: E402

from mms.decider_d1 import ABOUT  # noqa: E402


SIZE = (640, 360)  # what the reader hands a decider for a screen or a porch camera


BACKDROP = (70, 90, 130)  # 100 or more from every shade below


RED, BROWN, GREEN = (200, 40, 40), (125, 80, 45), (30, 160, 60)


def framed(rgb, share: float = 0.2) -> Image.Image:
    """A frame of faint noise with one box of exactly `rgb` over about `share` of it."""
    w, h = SIZE
    box_w, box_h = int(w * share ** 0.5), int(h * share ** 0.5)
    pixels = np.random.default_rng(1).integers(-6, 7, (h, w, 3)) + BACKDROP
    x, y = (w - box_w) // 3, (h - box_h) // 3
    pixels[y:y + box_h, x:x + box_w] = rgb
    return Image.fromarray(pixels.astype(np.uint8))


def jpeg(image: Image.Image) -> Image.Image:
    """The picture after a JPEG round trip at quality 90 (what a photo of the haystack is saved at)."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def x264(image: Image.Image, folder) -> Image.Image:
    """The picture after an x264 round trip, as the haystack's screen and porch files are written, decoded by PyAV as the
    reader does."""
    path = folder / "frame.mp4"
    day.vfr_video(path, folder, [(image, 2.0)])
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_image()


def says(instructions: str, *media) -> bool:
    """Whether the fake decider answers yes to a yes or no question over `media`."""
    answer = fake_decider.decide(None, {"q": {"type": "noul", "instructions": instructions}}, list(media))
    return answer["q"]["noul"] > 0.5


def shade_of(rgb) -> str:
    return "shade:" + ",".join(str(v) for v in rgb)


def test_a_shade_is_the_pixels_within_30_by_euclidean_distance():
    def flat(rgb) -> Image.Image:
        return Image.new("RGB", (64, 36), rgb)

    assert (fake_decider.SHADE_TOL, fake_decider.SHADE_SHARE) == (30, 0.02)
    assert fake_decider.shade_share(flat(RED), RED) == 1.0
    assert fake_decider.shade_share(flat((200, 69, 40)), RED) == 1.0  # 29 away
    assert fake_decider.shade_share(flat((200, 71, 40)), RED) == 0.0  # 31 away
    # 25 off in two channels is 35 away: within 30 in each channel is not within 30
    assert fake_decider.shade_share(flat((200, 65, 65)), RED) == 0.0
    assert fake_decider.shade_share(flat((200, 52, 52)), RED) == 1.0
    assert fake_decider.shade_share(Image.new("RGBA", (8, 8), RED + (255,)), RED) == 1.0, "any mode is read as RGB"
    assert fake_decider.shade_share(Image.new("RGB", (0, 0)), RED) == 0.0


def test_a_shaded_box_is_found_at_the_share_threshold_and_not_below_it():
    def with_pixels(n: int) -> Image.Image:
        """A frame with exactly `n` pixels of RED, the first ones in reading order."""
        pixels = np.full((SIZE[1] * SIZE[0], 3), BACKDROP, dtype=np.uint8)
        pixels[:n] = RED
        return Image.fromarray(pixels.reshape(SIZE[1], SIZE[0], 3))

    at = round(SIZE[0] * SIZE[1] * fake_decider.SHADE_SHARE)
    assert at == 4608
    assert not says(shade_of(RED), with_pixels(at - 100)) and says(shade_of(RED), with_pixels(at + 100))
    assert says(shade_of(RED), with_pixels(0), with_pixels(at + 100)), "one picture of several is enough"
    assert not says(shade_of(RED), with_pixels(0))
    assert not says(shade_of(RED))


@pytest.mark.parametrize("rgb", [RED, BROWN, GREEN])
def test_a_shaded_box_of_a_fifth_of_the_frame_survives_jpeg_and_x264(rgb, tmp_path):
    image = framed(rgb)
    assert fake_decider.shade_share(image, rgb) == pytest.approx(0.2, abs=0.01)
    for how, coded in (("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert coded.size == SIZE
        share = fake_decider.shade_share(coded, rgb)
        assert 0.17 <= share <= 0.23, f"{how}: the box is {share:.3f} of the frame after coding, from 0.2"
        assert says(shade_of(rgb), coded), how


def test_a_shaded_box_as_small_as_a_needle_may_be_still_clears_the_threshold_after_coding(tmp_path):
    image = framed(RED, 0.05)
    for how, coded in (("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert fake_decider.shade_share(coded, RED) >= 0.03, f"{how}: a needle's 3% floor must not fall under the 2% threshold"
        assert says(shade_of(RED), coded), how


def test_a_shade_60_away_does_not_match_before_or_after_coding(tmp_path):
    near = (140, 40, 40)  # 60 from RED
    assert float(np.linalg.norm(np.subtract(near, RED))) == 60.0
    image = framed(near)
    for how, picture in (("raw", image), ("jpeg", jpeg(image)), ("x264", x264(image, tmp_path))):
        assert fake_decider.shade_share(picture, RED) < 0.005, how
        assert not says(shade_of(RED), picture), how
        assert says(shade_of(near), picture), how


def test_shade_finds_a_box_anywhere_and_leaves_the_other_predicates_as_they_were():
    corner = Image.new("RGB", SIZE, (245, 245, 240))
    corner.paste(Image.new("RGB", (SIZE[0] // 2, SIZE[1] // 2), (255, 0, 0)), (0, 0))  # the top-left quarter in red
    lower = framed(RED)  # a box that is not in the top-left quarter and is not the mean colour
    assert says("corner:red", corner) and not says("corner:red", lower)
    assert not says("colour:red", lower) and says("colour:red", Image.new("RGB", SIZE, (255, 0, 0)))
    assert not says(shade_of(RED), corner), "255,0,0 is not within 30 of 200,40,40"
    assert says(shade_of(RED), lower) and says(shade_of((255, 0, 0)), corner)
    tone = np.sin(2 * np.pi * 659 * np.arange(ABOUT.audio_rate) / ABOUT.audio_rate).astype(np.float32) * 0.3
    sound = {"type": "audio", "audio": tone, "sampling_rate": ABOUT.audio_rate}
    assert says("pitch:659", sound) and not says("pitch:900", sound) and not says(shade_of(RED), sound)
