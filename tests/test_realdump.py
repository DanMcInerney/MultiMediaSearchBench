"""The builder the real scenarios share (scenarios/realdump): files written with their recording times and measured, the leak probe, the
split, the prompt in each way of reaching the files, and the recipe's time zone. Over small files made here."""

from __future__ import annotations

import os
import struct
import types
from datetime import datetime
from pathlib import Path

import pytest

from mmmb.core.contract import Served
from mmmb.core.question import Span
from mmmb.core.runfolder import IN_ONLY
from mmmb.core.scenario import Access
from mmmb.scenarios import realdump
from mmmb.scenarios.realdump import File

JPEG = b"\xff\xd8" + b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00deer!" + b"\xff\xd9"


INFO = b"INFO" + b"INAM" + struct.pack("<I", 14) + b"mountain_lion\0"  # a WAV's title, in a LIST chunk


def wav(extra: bytes = b"", samples: bytes = b"\x00\x00" * 8) -> bytes:
    """A WAV of `samples` with `extra` chunks before its data."""
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
    body = b"WAVE" + fmt + extra + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


def written(tmp_path: Path, contents: dict[str, bytes]):
    files = [File(path, "camera", datetime(2013, 12, 16, 8, n).timestamp(), "photo", "image", {"n": n}) for n, path in enumerate(contents)]
    entries = realdump.write(files, tmp_path / "dump", 2, lambda f, path: path.write_bytes(contents[f.path]))
    return realdump.dump_of("key", tmp_path / "dump", "x", entries)


def test_each_file_is_written_with_its_recording_time_and_measured(tmp_path):
    dump = written(tmp_path, {"DCIM/IMG_0001.JPG": JPEG, "DCIM/IMG_0002.JPG": JPEG + b"x"})
    first, second = dump.entries
    assert (first.path, first.size, second.size) == ("DCIM/IMG_0001.JPG", len(JPEG), len(JPEG) + 1)
    assert os.stat(tmp_path / "dump" / first.path).st_mtime == first.t == datetime(2013, 12, 16, 8, 0).timestamp()
    assert datetime.fromtimestamp(second.t) == datetime(2013, 12, 16, 8, 1), "a wall time is this machine's"
    assert first.mtime_ns == os.stat(tmp_path / "dump" / first.path).st_mtime_ns and len(first.sha256) == 64
    with pytest.raises(ValueError, match="not empty"):
        written(tmp_path, {"other.jpg": JPEG})


def test_the_probe_passes_a_clean_dump_and_finds_a_label_in_a_name_or_in_metadata_but_not_in_the_media(tmp_path):
    clean = written(tmp_path / "a", {"IMG_0001.JPG": JPEG, "rec.wav": wav(samples=b"deer" * 4)})
    assert realdump.probe(clean, tmp_path / "a", ["deer", "mountain_lion"]).status == "passed"
    assert "No file's name" in (tmp_path / "a" / "shortcuts.md").read_text(encoding="utf-8")
    comment = b"\xff\xfe\x00\x0dA DEER here"
    leaky = written(tmp_path / "b", {"deer_0001.jpg": JPEG, "IMG_0002.JPG": JPEG[:2] + comment + JPEG[2:],
                                     "rec.wav": wav(extra=b"LIST" + struct.pack("<I", len(INFO)) + INFO)})
    outcome = realdump.probe(leaky, tmp_path / "b", ["deer", "mountain_lion"])
    assert outcome.status == "failed"
    assert outcome.failures == ("deer_0001.jpg: its name says 'deer'", "IMG_0002.JPG: its metadata says 'deer'",
                                "rec.wav: its metadata says 'mountain lion'")


def test_the_split_marks_three_in_ten_dev_by_the_seed():
    qs = [realdump.question(code="X", qid=f"q{n}", now=1.7e9, text="?", kind="count", template="count", modality="image", shape="<n>",
                            grade=realdump.grade("count", truth="1"), evidence=[Span("a.jpg", 0, 0)], sources=("camera",), since=0,
                            until=1, value="1") for n in range(20)]
    one, again, other = realdump.split(qs, 1), realdump.split(qs, 1), realdump.split(qs, 2)
    assert sum(q.split == "dev" for q in one) == 6 and one == again
    assert {q.qid for q in one if q.split == "dev"} != {q.qid for q in other if q.split == "dev"}


def test_a_question_that_is_absent_has_no_evidence_and_only_it():
    with pytest.raises(ValueError, match="absent"):
        realdump.question(code="X", qid="q", now=1.7e9, text="?", kind="absent", template="absent", modality="image", shape="<n>",
                          grade=realdump.grade("absent"), evidence=[Span("a.jpg", 0, 0)], sources=("camera",), since=0, until=1, value=None)


def prompt_of(own_tools: bool, server: bool, fmt: str = "json") -> str:
    q = realdump.question(code="X", qid="q", now=datetime(2013, 12, 23, 9, 0).timestamp(), text="On how many days did a deer come?",
                          kind="count", template="count", modality="image", shape="<a number>", grade=realdump.grade("count", truth="2"),
                          evidence=[Span("a.jpg", 0, 0)], sources=("camera",), since=0, until=1, value="2")
    served = Served(types.SimpleNamespace(name="memo") if server else None, None, (), None, "memo" if server else None, None, {})
    return realdump.prompt(q, Access(own_tools, Path("D:/data") if own_tools else None, served, fmt), "the memory card of my trail camera",
                           "DCIM holds the photos.")


def test_the_prompt_says_the_time_how_the_files_are_reached_and_the_answer_shape():
    files = prompt_of(True, False)
    assert files.startswith("It is now Monday 23 December 2013, 09:00. The files of the memory card of my trail camera are in ")
    assert "DCIM holds the photos. Your working folder is yours to write in." in files and "On how many days did a deer come?" in files
    assert '{"answer": "<a number>", "evidence": [...]}' in files and 'when the files do not hold it' in files
    system = prompt_of(False, True)
    assert "memo holds the memory card of my trail camera." in system and "when memo does not hold it" in system and "DCIM" not in system
    both = prompt_of(True, True)
    assert "when memo and these files do not hold it" in both
    assert "You cannot see the memory card of my trail camera." in prompt_of(False, False)
    assert prompt_of(True, False, "line").endswith("End with one line ANSWER: <a number>, or ANSWER: unknown when the files do not hold it.")


def test_the_deny_words_hold_the_code_and_the_source_names_in_commands_and_the_source_folder_everywhere():
    words = realdump.deny_words("i9_thing", ("caltech",), ("bench-data",))
    assert set(words) == {IN_ONLY + w for w in ("i9_thing", "realdump", "from mmmb", "import mmmb", "caltech")} | {"bench-data"}


def test_a_recipe_made_in_another_time_zone_is_refused():
    days = realdump.span_days(datetime(2013, 12, 16).date(), datetime(2013, 12, 22).date())
    assert len(days) == 7 and days[-1] == "2013-12-22"
    realdump.check_offsets({"days": days, "utc_offsets": realdump.utc_offsets(days)})
    with pytest.raises(ValueError, match="another dump"):
        realdump.check_offsets({"days": days, "utc_offsets": ["+1400", "+1400"]})


def test_a_file_whose_metadata_the_probe_cannot_read_fails_it(tmp_path):
    dump = written(tmp_path, {"cam1/0001.mp4": b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12})
    outcome = realdump.probe(dump, tmp_path, ["deer"])
    assert outcome.failures == ("cam1/0001.mp4: its metadata is not read (no reader for .mp4)",)
    mp4 = {".mp4": lambda path: path.read_bytes()}
    assert realdump.probe(dump, tmp_path, ["deer"], mp4).status == "passed", "a scenario of other files passes its own reader"


def test_with_no_way_to_the_files_the_prompt_still_offers_unknown():
    assert prompt_of(False, False).endswith('Answer "unknown" with "evidence": [] when what you know does not hold it.')


def test_the_probe_finds_a_word_of_a_label_and_reads_only_the_text_of_metadata(tmp_path):
    """Gate 2: a label is caught by its words too ("microwave" for `Microwave_oven`, "coin" for `Coin_(dropping)`), in a name or in a
    WAV's title; in metadata only runs of four printable characters or more are text."""
    title = b"INFO" + b"INAM" + struct.pack("<I", 20) + b"coin drop on table\0\0"
    dump = written(tmp_path / "a", {"folder_01/microwave_beep.wav": wav(), "folder_01/0002.wav": wav(b"LIST" + struct.pack("<I", len(title)) + title),
                                    "folder_01/0003.wav": wav(b"junk" + struct.pack("<I", 6) + b"\x01ov\x02en")})
    outcome = realdump.probe(dump, tmp_path / "a", ["Microwave_oven", "Coin_(dropping)", "Other_vehicle"])
    assert outcome.failures == ("folder_01/microwave_beep.wav: its name says 'microwave'", "folder_01/0002.wav: its metadata says 'coin'")
    assert realdump.label_words(["Coin_(dropping)", "person_opens_trunk"])[:2] == ["person opens trunk", "coin (dropping)"]
    assert "person" not in realdump.label_words(["person_opens_trunk"]), "the commonest words are no label"


SOME_CODE = ('python -c "import wave, numpy as np; x = np.frombuffer(w.readframes(n), dtype=np.int16); print(isinstance(x, np.ndarray)); '
             'timevalue = somevar = namevalue = 0; mixedcase = 1"')


@pytest.mark.parametrize("name", ["v1_security", "i3_trailcam", "a1_home_audio", "m4_samples"])
def test_no_deny_word_of_a_real_scenario_is_in_ordinary_code(name, tmp_path):
    """Gate 2: "sins" is in `isinstance`, "meva" in `timevalue`; an agent's own script must not end its exec as an escape."""
    from mmmb.core.agents.common import reached
    import importlib
    words = tuple(w.lower() for w in importlib.import_module(f"mmmb.scenarios.{name}").deny_words((), tmp_path, "k"))
    assert reached({"command": SOME_CODE}, words) is None
    assert reached({"command": r"dir D:\bench-data\sources"}, words) == "bench-data"
