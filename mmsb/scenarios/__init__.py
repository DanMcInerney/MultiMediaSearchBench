"""The scenarios by name (DESIGN 5): a name stands for the module that is the scenario, which has the names of a `Scenario` as
functions and is imported when it is asked for. A scenario is added by its module and a line here."""

from __future__ import annotations

import importlib

from mmsb.core.scenario import Scenario

REGISTRY = {"v7_screen": "mmsb.scenarios.v7_screen", "v8_egolife": "mmsb.scenarios.v8_egolife",
            "v1_security": "mmsb.scenarios.v1_security", "i3_trailcam": "mmsb.scenarios.i3_trailcam",
            "a1_home_audio": "mmsb.scenarios.a1_home_audio", "m4_samples": "mmsb.scenarios.m4_samples",
            "i1_camera_roll": "mmsb.scenarios.i1_camera_roll", "i4_site_photos": "mmsb.scenarios.i4_site_photos", "i9_nature": "mmsb.scenarios.i9_nature",
            "v2_dashcam": "mmsb.scenarios.v2_dashcam", "v3_match": "mmsb.scenarios.v3_match", "v10_home_clips": "mmsb.scenarios.v10_home_clips",
            "v9_lectures": "mmsb.scenarios.v9_lectures", "a6_lecture_audio": "mmsb.scenarios.a6_lecture_audio", "a7_podcast": "mmsb.scenarios.a7_podcast",
            "v4_meetings": "mmsb.scenarios.v4_meetings", "a3_meetings": "mmsb.scenarios.a3_meetings",
            "m3_voicemail": "mmsb.scenarios.m3_voicemail", "a4_field": "mmsb.scenarios.a4_field", "a8_music": "mmsb.scenarios.a8_music",
            "i2_screenshots": "mmsb.scenarios.i2_screenshots", "i7_receipts": "mmsb.scenarios.i7_receipts", "i8_course_pages": "mmsb.scenarios.i8_course_pages"}


def get(name: str) -> Scenario:
    """The scenario `name` stands for, imported now. A ValueError names the scenarios there are for any other name."""
    if name not in REGISTRY:
        raise ValueError(f"{name!r} is not a scenario: {', '.join(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])  # type: ignore[return-value]  (a module with the Scenario's names)
