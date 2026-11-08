"""The rules the constants of the `media_*` modules keep against each other (a needle's shade is clear of every distractor's, a chime's
notes are clear of every other sound's pitch, nothing on the taskbar or in the typing lines is near a needle's shade), checked when the
package is imported: a wrong constant fails then, never in a run. It is a module of its own because it reads the constants of all the
others. Moved from haystack_media.py of the repository at bb498a6."""

from __future__ import annotations

from mmsb.scenarios.v7_screen.media_common import (
    DISTRACTOR_RGB, NEEDLE_RGB, SAFE, SHADE_GAP, SOURCE_NEEDLES, _clear_of_needles, distance, gap,
)
from mmsb.scenarios.v7_screen.media_porch import JACKETS, WALLS
from mmsb.scenarios.v7_screen.media_screen import TASKBAR_ICON, TASKBAR_INK, TASKBAR_LINE, TASKBAR_LOGO, TASKBAR_RGB, TYPING
from mmsb.scenarios.v7_screen.media_sound import ALARM_HZ, BEEPER_HZ, DING_HZ, DOORBELL_PAIRS, PITCH_GAP, RING_HZ, TIMER_HZ


def _check_constants() -> None:
    """The module's own constants meet its rules (a wrong one fails when the file is imported, never in a run)."""
    for source, kinds in SOURCE_NEEDLES.items():
        for kind in kinds:
            for group in kinds:
                for name, rgb in DISTRACTOR_RGB[group].items():
                    assert distance(rgb, NEEDLE_RGB[kind]) >= SHADE_GAP, f"the {source} distractor {name} {rgb} is near the {kind} shade"
    for (a, ka), (b, kb) in ((x, y) for x in NEEDLE_RGB.items() for y in NEEDLE_RGB.items() if x[0] < y[0]):
        assert distance(ka, kb) >= SHADE_GAP, f"the {a} and {b} shades are within {SHADE_GAP}"
    for rgb in SAFE:  # photos and the browser's picture
        assert all(distance(rgb, NEEDLE_RGB[k]) >= SHADE_GAP for k in ("sign", "car", "error")), f"SAFE {rgb} is near a needle's shade"
    for rgb in (TASKBAR_RGB, TASKBAR_LINE, TASKBAR_ICON, TASKBAR_LOGO, TASKBAR_INK, (255, 255, 255), (0, 0, 0)):  # the taskbar, its clock, the cursor
        assert distance(rgb, NEEDLE_RGB["error"]) >= SHADE_GAP, f"the taskbar or cursor colour {rgb} is near the error shade"
    for app, typing in TYPING.items():  # the text typed (a slide's ink is `_ink`'s)
        for rgb in typing.ink or ((255, 255, 255), (20, 20, 24)):
            _clear_of_needles(rgb, f"the {app}'s typing ink", "screen")
    for rgb in (*WALLS, *JACKETS):  # the porch
        assert distance(rgb, NEEDLE_RGB["parcel"]) >= SHADE_GAP, f"the porch colour {rgb} is near the parcel's shade"
    notes = [n for pair in DOORBELL_PAIRS for n in pair]
    lone = {**{f"{k} {i}": hz for k, pitches in BEEPER_HZ.items() for i, hz in enumerate(pitches)},
            **{f"timer {hz}": hz for hz in TIMER_HZ}, **{f"ding {hz}": hz for hz in DING_HZ}, **{f"ring {hz}": hz for hz in RING_HZ},
            **{f"ring {hz} (+9%)": hz * 480 / 440 for hz in RING_HZ}}
    for name, hz in lone.items():
        assert all(gap(hz, note) >= PITCH_GAP for note in notes), f"{name} is within {PITCH_GAP:.0%} of a chime note"
        near_alarm = ALARM_HZ[0] / (1 + PITCH_GAP) < hz < ALARM_HZ[1] * (1 + PITCH_GAP)
        assert not near_alarm, f"{name} is within {PITCH_GAP:.0%} of the alarm"
    for note in notes:
        assert gap(note, ALARM_HZ[0]) >= PITCH_GAP, f"a chime note {note} is near the alarm"


_check_constants()
