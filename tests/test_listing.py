"""New tests for the listing system (plan 6.2, DESIGN 3.2): ingest, the MCP server, the horizon, what a session serves, and what the system says
of itself. The report's treatment of its results and the runs through the runner and the real agents are in test_listing_run.py.

Expected values come from sources that do not run the code under test: `os.stat` for sizes and times (the server reads a file written at ingest),
the ffmpeg command for a video's and a sound's container fields (the system reads them with PyAV), the writer of the V7 week for what a photo
or a phone clip says of its time, what a test plants (a secret word in the bodies and the pixels, values only the manifest holds, a comment in an
EXIF), and the spec's lists of keys written out by hand. The client is the SDK's, over stdio, to the real server command a session serves.

The tiny V7 week of seed 7 is built once for the module (about 45 s at two workers), so the tests share a worker."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import pytest
from listing_world import (ALLOWED_KEYS, BLOB, CONTAINER_KEYS, COMMENT, KIND, ROW_KEYS, SECRET, SHA, T, T_OFFSET, called, case, ingested, keys_of, pages,
                           planted_dump, server_of, talk)
from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational
from test_neutral import modules_walked

from mmmb import scenarios, systems
from mmmb.core.config import parse_config
from mmmb.core.contract import Dump, ManifestEntry
from mmmb.systems.listing import read as reading
from mmmb.systems.listing import server, system
from mmmb.systems.listing.system import Listing, ListingSession

pytestmark = pytest.mark.xdist_group("listing")
MODALITIES = frozenset({"image", "video", "audio", "text", "records"})


@pytest.fixture(scope="module")
def week(tmp_path_factory):
    """The tiny V7 week and the listing of it: `(dump, listing.jsonl)`."""
    v7 = scenarios.get("v7_screen")
    root = tmp_path_factory.mktemp("week")
    dump, _ = v7.build(v7.recipe(scale="tiny", seed=7, days=["2026-10-30", "2026-10-31"]), root / "data", 2)
    return dump, ingested(dump, root / "store")


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """The planted dump (listing_world.py) and its listing: `(dump, listing.jsonl)`."""
    root = tmp_path_factory.mktemp("planted")
    (root / "data").mkdir()
    dump = planted_dump(root / "data")
    return dump, ingested(dump, root / "store")


def mtimes(dump: Dump) -> dict[str, float]:
    """The time the file system gives each file of the dump."""
    return {e.path: (dump.root / e.path).stat().st_mtime for e in dump.entries}


def ffmpeg_says(path: Path) -> dict:
    """A media file as the ffmpeg command describes it (`ffmpeg -i`, which exits 1 for want of an output): duration, the first video stream's
    size and frame rate, the first audio stream's sample rate. A second reader of the same facts PyAV gives."""
    text = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text).groups()
    video = re.search(r"Video: [^\n]*?, (\d{2,5})x(\d{2,5})[^\n]*?([\d.]+) fps", text)
    audio = re.search(r"Audio: [^\n]*?(\d+) Hz", text)
    return {"duration": 3600 * int(h) + 60 * int(m) + float(s), **({"width": int(video[1]), "height": int(video[2]), "fps": float(video[3])} if video else {}),
            **({"sample_rate": int(audio[1])} if audio else {})}


# -- the server, over the tiny week --------------------------------------------------------------------------------------------

def test_the_server_offers_exactly_two_tools_and_the_instructions_a_session_serves_are_its_own(week):
    dump, listing = week

    async def script(client):
        return (await client.list_tools()).tools, client.instructions

    tools, told = talk(script, listing)
    assert [t.name for t in tools] == ["list_files", "file_info"]
    assert all(t.description and t.input_schema["additionalProperties"] is False for t in tools)
    assert set(tools[0].input_schema["properties"]) == {"source", "since", "until", "limit", "after"}
    assert tools[1].input_schema["required"] == ["path"]
    served = ListingSession(listing).serve(case())
    assert served.server_instructions == told, "Codex is given in a file what Claude is told by the server"
    assert served.server.tools == ("list_files", "file_info")


def test_every_file_of_the_week_is_listed_with_the_size_and_time_the_file_system_gives(week):
    dump, listing = week
    seen = {}

    async def script(client):
        for entry in dump.entries:
            seen[entry.path] = await called(client, "file_info", path=entry.path)
        return await pages(client, limit=200)

    (only,) = talk(script, listing)
    times = mtimes(dump)
    assert [f["path"] for f in only["files"]] == [p for _, p in sorted((t, p) for p, t in times.items())], "oldest first, by the file system's time"
    assert only["matching"] == len(dump.entries) > 90 and "next" not in only
    by_path = {e.path: e for e in dump.entries}
    for row in only["files"]:
        info = (dump.root / row["path"]).stat()
        assert row["size"] == info.st_size and row["source"] == by_path[row["path"]].source
        assert abs(datetime.fromisoformat(row["mtime"]).timestamp() - info.st_mtime) < 0.001
        error, full = seen[row["path"]]
        assert not error and {k: full[k] for k in row} == row, "file_info says what list_files says, and more"


def test_a_photo_and_a_phone_clip_say_their_time_as_the_writer_put_it_and_media_say_what_ffmpeg_says(week):
    dump, listing = week
    times = mtimes(dump)
    photo = next(e for e in dump.entries if e.path.startswith("phone/") and e.path.endswith(".jpg"))
    clip = next(e for e in dump.entries if e.path.startswith("phone/") and e.path.endswith(".mp4"))
    mic = next(e for e in dump.entries if e.source == "mic")
    screen = next(e for e in dump.entries if e.source == "screen")

    async def script(client):
        return [(await called(client, "file_info", path=e.path))[1] for e in (photo, clip, mic, screen)]

    got = dict(zip(("photo", "clip", "mic", "screen"), talk(script, listing)))
    assert got["photo"]["exif"]["DateTimeOriginal"] == datetime.fromtimestamp(times[photo.path]).strftime("%Y:%m:%d %H:%M:%S"), "(write.py: exif_of)"
    assert got["photo"]["container"] == {"format": "JPEG", "width": 1280, "height": 720}
    assert got["clip"]["container"]["creation_time"].startswith(datetime.fromtimestamp(times[clip.path], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
    for name, entry in (("clip", clip), ("mic", mic), ("screen", screen)):
        want, found = ffmpeg_says(dump.root / entry.path), got[name]["container"]
        assert set(found) <= CONTAINER_KEYS and abs(found["duration"] - want.pop("duration")) < 0.05, name
        assert {k: found[k] for k in want} == want, name
    assert "exif" not in got["mic"] and "container" in got["mic"] and "sample_rate" in got["mic"]["container"]


def test_rows_past_until_are_in_neither_tool_and_a_row_at_until_is_in(week):
    dump, listing = week
    times = mtimes(dump)
    until = sorted(times.values())[len(times) // 2]  # a file's own time: that file is in
    inside = sorted((t, p) for p, t in times.items() if t <= until)
    past = sorted(p for p, t in times.items() if t > until)
    assert inside and past and any(t == until for t, _ in inside)

    async def script(client):
        return (await pages(client, limit=200), await called(client, "file_info", path=inside[-1][1]),
                [await called(client, "file_info", path=p) for p in past], await called(client, "list_files", after=past[0]))

    listed, last, gone, after = talk(script, listing, until)
    assert [f["path"] for page in listed for f in page["files"]] == [p for _, p in inside] and listed[0]["matching"] == len(inside)
    assert not last[0] and last[1]["path"] == inside[-1][1]
    assert all(error and got["error"]["code"] == "not_found" and "not in the listing" in got["error"]["message"] for error, got in gone), "as a path that never was"
    assert after[0] and after[1]["error"]["code"] == "not_found", "a row that is not there is no place to continue from"


def test_a_source_all_of_whose_files_are_past_until_is_not_a_source(week):
    dump, listing = week
    times, source_of = mtimes(dump), {e.path: e.source for e in dump.entries}
    late = max({s for s in source_of.values()}, key=lambda s: min(t for p, t in times.items() if source_of[p] == s))
    until = min(t for p, t in times.items() if source_of[p] == late) - 1

    async def script(client):
        return await called(client, "list_files", source=late), await called(client, "list_files")

    refused, everything = talk(script, listing, until)
    assert refused[0] and refused[1]["error"]["code"] == "bad_request" and late not in refused[1]["error"]["message"].split("sources are")[1]
    assert not everything[0] and {f["source"] for f in everything[1]["files"]} <= set(source_of.values()) - {late}


def test_paging_returns_each_row_once_in_order_and_a_filter_pages_too(week):
    dump, listing = week
    times, source_of = mtimes(dump), {e.path: e.source for e in dump.entries}
    ordered = [p for _, p in sorted((t, p) for p, t in times.items())]
    screen = [p for p in ordered if source_of[p] == "screen"]
    since = sorted(times.values())[30]
    from_since = [p for p in ordered if times[p] >= since]

    async def script(client):
        return (await pages(client, limit=7), await pages(client, source="screen", limit=6),
                await pages(client, since=datetime.fromtimestamp(since).isoformat(), limit=200),
                await pages(client, since=since, until=since + 1, limit=200))

    everything, one_source, iso_since, window = talk(script, listing)
    assert [f["path"] for page in everything for f in page["files"]] == ordered, "each row once, oldest first"
    assert [len(page["files"]) for page in everything] == [7] * (len(ordered) // 7) + [len(ordered) % 7] * bool(len(ordered) % 7)
    assert all(page["next"] == page["files"][-1]["path"] for page in everything[:-1]) and "next" not in everything[-1]
    assert [f["path"] for page in one_source for f in page["files"]] == screen and one_source[0]["matching"] == len(screen)
    assert [f["path"] for page in iso_since for f in page["files"]] == from_since, "an ISO time with no zone is local; since is included"
    assert [f["path"] for page in window for f in page["files"]] == [p for p in ordered if since <= times[p] <= since + 1]


def test_a_path_outside_the_listing_and_a_malformed_call_are_errors_with_a_code(week, tmp_path):
    dump, listing = week
    outside = tmp_path / "outside.txt"
    outside.write_text("not in the dump")
    real = dump.entries[0].path

    async def script(client):
        out = {"outside": await called(client, "file_info", path=str(outside)), "up": await called(client, "file_info", path="../" + real),
               "windows": await called(client, "file_info", path=real.replace("/", "\\")), "number": await called(client, "file_info", path=7),
               "none": await called(client, "file_info"), "extra": await called(client, "file_info", path=real, content=True)}
        for name, arguments in (("zero", {"limit": 0}), ("big", {"limit": 201}), ("text", {"limit": "5"}), ("flag", {"limit": True}),
                                ("time", {"since": "yesterday"}), ("nan", {"until": "nan"}), ("kind", {"kind": "photo"}),
                                ("source", {"source": "nowhere"}), ("after", {"after": "nowhere/x.jpg"})):
            out[name] = await called(client, "list_files", **arguments)
        out["tool"] = await called(client, "read_file", path=real)
        return out

    got = talk(script, listing)
    assert all(error for error, _ in got.values()), {k: v for k, v in got.items() if not v[0]}
    codes = {k: body["error"]["code"] for k, (_, body) in got.items()}
    assert {k for k, c in codes.items() if c == "not_found"} == {"outside", "up", "windows", "after", "tool"}
    assert {k for k, c in codes.items() if c == "bad_request"} == set(got) - {"outside", "up", "windows", "after", "tool"}


def test_the_server_imports_nothing_that_could_open_a_recording_and_starts_with_those_modules_refused_and_the_dump_gone(week):
    """It is given the listing file and no path to the dump: with the picture and media libraries and this package refused to its process it
    answers all the same."""
    dump, listing = week
    forbidden = {"PIL", "av", "numpy", "imageio_ffmpeg", "mmmb", "multimem"}
    tree = ast.parse(Path(server.__file__).read_text(encoding="utf-8"))
    imported = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in getattr(n, "names", [None])}
    assert not imported & forbidden, imported & forbidden
    program = textwrap.dedent(f"""\
        import runpy, sys
        class Refuse:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {sorted(forbidden)!r}:
                    raise ModuleNotFoundError(name, name=name)
        sys.meta_path.insert(0, Refuse())
        sys.argv = sys.argv[1:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        """)

    async def script(client):
        return [t.name for t in (await client.list_tools()).tools], await pages(client, limit=200)

    names, (only,) = talk(script, listing, program=["-c", program])
    assert names == ["list_files", "file_info"] and only["matching"] == len(dump.entries)


def test_the_week_lists_no_key_the_spec_does_not_name_and_no_manifest_time_kind_or_hash(week):
    dump, listing = week
    rows = [json.loads(line) for line in listing.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(dump.entries) and {r["path"] for r in rows} == {e.path for e in dump.entries}
    assert set().union(*map(set, rows)) <= ROW_KEYS and set().union(*(set(r.get("container", {})) for r in rows)) <= CONTAINER_KEYS
    text = listing.read_text(encoding="utf-8")
    assert not any(e.sha256 in text for e in dump.entries[:20]) and not BLOB.search(text)
    assert all(r["size"] == (dump.root / r["path"]).stat().st_size for r in rows)


# -- a planted dump --------------------------------------------------------------------------------------------------------------

def test_a_dump_of_a_picture_a_sound_and_files_of_text_lists_what_they_say_of_themselves_and_nothing_inside_them(planted):
    dump, listing = planted

    async def script(client):
        out = await pages(client, limit=2)
        return out, [(await called(client, "file_info", path=e.path))[1] for e in dump.entries]

    listed, infos = talk(script, listing)
    outputs = [*listed, *infos]
    everything = json.dumps(outputs) + listing.read_text(encoding="utf-8")
    assert SECRET not in everything and SECRET.lower() not in everything.lower(), "the word in a note, a config, a log, a picture's pixels and a sound's samples"
    assert not [e.t for e in dump.entries if str(e.t) in everything] and ".4375" not in everything and KIND not in everything and SHA not in everything, \
        "the manifest's t, kind and sha256"
    assert not BLOB.search(everything), "no run of base64: a file's bytes are not in any output"
    assert keys_of(outputs) <= ALLOWED_KEYS, keys_of(outputs) - ALLOWED_KEYS
    by_path = {i["path"]: i for i in infos}
    assert by_path["phone/IMG_birthday.jpg"]["exif"]["ImageDescription"] == COMMENT, "an EXIF comment is metadata: reported"
    assert by_path["phone/IMG_birthday.jpg"]["exif"]["UserComment"] == "user comment text"
    assert "phone/IMG_birthday.jpg" in [f["path"] for page in listed for f in page["files"]], "and so is a name"
    assert by_path["phone/IMG_planted.png"]["container"] == {"format": "PNG", "width": len(SECRET), "height": 1} and "exif" not in by_path["phone/IMG_planted.png"]
    wav = by_path["mic/mic-1.wav"]["container"]
    assert (wav["format"], wav["codec"], wav["sample_rate"]) == ("wav", "pcm_u8", 8000) and abs(wav["duration"] - 100 * len(SECRET) / 8000) < 0.001
    for path in ("notes/n.md", "config/app.toml", "logs/web.jsonl"):
        assert set(by_path[path]) == {"path", "source", "size", "mtime"} and by_path[path]["size"] == (dump.root / path).stat().st_size


def test_exif_tags_are_listed_by_name_as_numbers_and_text_and_binary_values_are_not_unless_they_are_text(tmp_path):
    """The tags the test puts in a picture come out as the values it put in: rationals as numbers, a GPS triple as a list, the text of a
    UserComment without its charset prefix, a UTF-16 XP tag as text, a long text cut at 256 characters; a maker's bytes and a binary tag are
    not listed."""
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Acme"
    exif[ExifTags.Base.Orientation] = 1
    exif[ExifTags.Base.XResolution] = IFDRational(72, 1)
    exif[0x9C9C] = "xp text".encode("utf-16-le")
    exif[50341] = b"\x00\x01\xff\xfe"
    inner = exif.get_ifd(ExifTags.IFD.Exif)
    inner[ExifTags.Base.ExposureTime] = IFDRational(1, 250)
    inner[ExifTags.Base.FNumber] = IFDRational(28, 10)
    inner[ExifTags.Base.ISOSpeedRatings] = 400
    inner[ExifTags.Base.MakerNote] = bytes(range(64, 128))
    inner[ExifTags.Base.UserComment] = b"ASCII\x00\x00\x00" + b"x" * 300
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps[1], gps[2] = "N", (IFDRational(40, 1), IFDRational(26, 1), IFDRational(46, 1))
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg", exif=exif)
    assert reading.exif_of(tmp_path / "a.jpg") == {
        "Make": "Acme", "Orientation": 1, "XResolution": 72.0, "XPComment": "xp text", "UserComment": "x" * 256 + "\u2026", "ISOSpeedRatings": 400,
        "ExposureTime": 0.004, "FNumber": 2.8, "GPSLatitudeRef": "N", "GPSLatitude": [40.0, 26.0, 46.0]}


def test_the_guard_that_finds_a_run_of_base64_finds_one():
    import base64
    import os

    assert BLOB.search(base64.b64encode(os.urandom(300)).decode()) and not BLOB.search("IMG_20261030_080029 and 2026-10-30T08:00:29.000-04:00")


def test_a_row_may_hold_only_the_keys_of_the_spec_of_the_right_types():
    """The check that `ingest` writes through and the server reads through. Expected: the spec's row, written out here."""
    good = {"path": "a/b.jpg", "source": "a", "size": 3, "mtime": 1.5, "exif": {"Make": "x"}, "container": {"format": "JPEG", "width": 3}}
    server.check_row(good)
    server.check_row({k: good[k] for k in ("path", "source", "size", "mtime")})
    for bad, why in (({**good, "t": 1.0}, "unknown key t"), ({**good, "kind": "photo"}, "unknown key kind"), ({**good, "sha256": SHA}, "unknown key sha256"),
                     ({k: v for k, v in good.items() if k != "size"}, "missing size"), ({**good, "size": "3"}, "integer"), ({**good, "size": True}, "integer"),
                     ({**good, "mtime": float("nan")}, "mtime a number"), ({**good, "container": {"kind": "x"}}, "not kind"), ({**good, "container": []}, "container is"),
                     ({**good, "exif": []}, "exif is"), ([], "JSON object")):
        with pytest.raises(ValueError, match=why):
            server.check_row(bad)


@pytest.mark.parametrize("key, value", [("t", T), ("kind", KIND), ("sha256", SHA), ("text", SECRET)])
def test_ingest_refuses_a_listing_that_would_hold_a_key_the_manifest_has_and_the_files_do_not(planted, tmp_path, monkeypatch, key, value):
    """The faults the listing is for: a copy of the manifest's time, kind or hash, or of a file's content, in a row."""
    dump, _ = planted
    real = reading.row_of

    def copying(root, entry):
        row, refused = real(root, entry)
        return {**row, key: value}, refused

    monkeypatch.setattr(reading, "row_of", copying)
    with pytest.raises(ValueError, match=f"unknown key {key}"):
        Listing().ingest(dump, tmp_path)


def test_a_file_that_will_not_read_is_listed_with_what_the_file_system_gives_and_a_refusal_and_a_missing_file_is_an_error(tmp_path):
    root, store = tmp_path / "data", tmp_path / "store"
    for rel, data in (("phone/bad.jpg", b"not a jpeg"), ("mic/bad.wav", b"RIFF"), ("notes/ok.md", b"fine")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    def entry(rel, media):  # at the time the file system gives it (plus what no file holds); a file that is not there has none
        t = (root / rel).stat().st_mtime + T_OFFSET if (root / rel).exists() else 0.0
        return ManifestEntry(rel, rel.split("/")[0], t, KIND, media, None, False, 1, 1, SHA)

    dump = Dump("k", root, "x", (entry("phone/bad.jpg", "image"), entry("mic/bad.wav", "audio"), entry("notes/ok.md", "text")), ())
    report = Listing().ingest(dump, store)
    assert sorted((r.path, r.reason.split(":")[0]) for r in report.refused) == [
        ("mic/bad.wav", "container not read"), ("phone/bad.jpg", "container not read"), ("phone/bad.jpg", "exif not read")]
    rows = {json.loads(line)["path"]: json.loads(line) for line in (store / "listing.jsonl").read_text().splitlines()}
    assert set(rows) == {"phone/bad.jpg", "mic/bad.wav", "notes/ok.md"}
    assert set(rows["phone/bad.jpg"]) == {"path", "source", "size", "mtime"} and rows["phone/bad.jpg"]["size"] == 10
    assert report.misdated == () and report.gpu_seconds == 0.0 and report.bytes_written == (store / "listing.jsonl").stat().st_size
    with pytest.raises(FileNotFoundError):
        Listing().ingest(Dump("k", root, "x", (entry("notes/gone.md", "text"),), ()), tmp_path / "store-ghost")


def test_a_store_is_made_once_and_a_path_in_the_dump_twice_is_refused(planted, tmp_path):
    dump, listing = planted
    with pytest.raises(ValueError, match="made once"):
        Listing().ingest(dump, listing.parent)
    twice = Dump(dump.key, dump.root, dump.scenario, (*dump.entries, dump.entries[0]), ())
    with pytest.raises(ValueError, match="listed twice"):
        Listing().ingest(twice, tmp_path)


# -- the system -----------------------------------------------------------------------------------------------------------------

def test_the_registry_has_listing_with_no_variants_and_a_session_serves_the_server_cut_to_the_horizon_and_nothing_of_the_case(planted):
    dump, listing = planted
    assert type(systems.load("listing")) is Listing and systems.defaults()["listing"] is None
    assert parse_config("listing", systems.defaults()).dir == "listing" and parse_config("listing+files", systems.defaults()).dir == "listing+files"
    a, b = ListingSession(listing).serve(case(None, "one")), ListingSession(listing).serve(case(None, "two"))
    assert a == b, "nothing of the case is in what is served, so cases run at once may share it"
    served = a.server
    assert (served.name, served.tools, dict(served.env)) == ("listing", ("list_files", "file_info"), {})
    assert Path(served.command).is_absolute() and served.args == (str(Path(server.__file__)), "--listing", str(listing)), "no --until: V7 has no horizon"
    assert (a.holder, a.parts, a.record, a.opens, a.clock_note) == ("listing", (), {}, (), None) and a.base_line
    cut = ListingSession(listing).serve(case(1793400000.5)).server
    assert cut.args == (*served.args, "--until", "1793400000.5") and server_of(listing, 7.0).args[-2:] == ("--until", "7.0")


def test_describe_says_what_the_system_is_and_its_hashes_cover_the_tools_and_the_fields(monkeypatch):
    d = Listing().describe(None)
    assert (d.name, d.variant, d.reads, d.local, d.needs_gpu, d.gpu_memory_gb, d.network) == ("listing", None, MODALITIES, True, False, None, False)
    assert (d.horizon, d.has_oracle, d.leak_filter, d.license) == ("cut", False, True, "none")
    assert Listing().describe(None) == d and d.config_hash != d.store_hash
    with pytest.raises(ValueError, match="no variants"):
        Listing().describe("d1-3B")
    assert Listing().check_ready(None) is None
    # the configuration hash moves with a tool's description or schema, the instructions, the base line; the store's with none of them
    for name, changed in (("DESCRIPTIONS", {**server.DESCRIPTIONS, "file_info": "other"}),
                          ("SCHEMAS", {**server.SCHEMAS, "file_info": {"type": "object"}}), ("INSTRUCTIONS", "other")):
        with monkeypatch.context() as patch:
            patch.setattr(server, name, changed)
            other = Listing().describe(None)
        assert other.config_hash != d.config_hash and other.store_hash == d.store_hash, name
    with monkeypatch.context() as patch:
        patch.setattr(system, "BASE", "other")
        assert Listing().describe(None).config_hash != d.config_hash
    # both move with the fields it lists and with what ingest writes
    for module, name, changed in ((server, "CONTAINER_KEYS", (*server.CONTAINER_KEYS, "bitrate")), (server, "FILE_KEYS", (*server.FILE_KEYS, "owner")),
                                  (reading, "TEXT_MAX", reading.TEXT_MAX + 1), (system, "INGEST_VERSION", system.INGEST_VERSION + 1)):
        with monkeypatch.context() as patch:
            patch.setattr(module, name, changed)
            other = Listing().describe(None)
        assert other.store_hash != d.store_hash and other.config_hash != d.config_hash, name


def test_the_fast_neutrality_check_walks_the_listing_with_multimem_refused():
    seen = modules_walked()
    assert {"mmmb.systems.listing", "mmmb.systems.listing.system", "mmmb.systems.listing.server", "mmmb.systems.listing.read"} <= set(seen)
