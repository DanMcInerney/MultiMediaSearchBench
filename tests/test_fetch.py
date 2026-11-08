"""`mmsb fetch` (mmsb/fetch.py) against a local HTTP server that serves small stand-ins of each source in the layout the real one has: the
retries and the waits they make (a stand-in for `sleep` notes the seconds, so no test waits), a download that resumes, a file that is
there already, the count each build needs, the labels and clips of each scenario's source, and SINS's minute files read out of zips by
Range and kept as FLAC. No test reaches the network: any address but the local server's fails the test."""

from __future__ import annotations

import email.utils
import io
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from mmsb import cli, fetch
from mmsb.scenarios import realdump
from mmsb.scenarios.a1_home_audio import source as sins
from mmsb.scenarios.i3_trailcam import source as cct
from mmsb.scenarios.m4_samples import source as fsd
from mmsb.scenarios.v1_security import source as meva


class Handler(BaseHTTPRequestHandler):
    site: Site

    def log_message(self, *args) -> None:
        pass

    def do_HEAD(self) -> None:
        self.answer(False)

    def do_GET(self) -> None:
        self.answer(True)

    def reply(self, status: int, body: bytes, headers: dict[str, str], send: bool = True) -> None:
        self.send_response(status)
        for key, value in {"Content-Length": str(len(body)), **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        if send:
            self.wfile.write(body)

    def answer(self, with_body: bool) -> None:
        url = urllib.parse.urlsplit(self.path)
        self.site.requests.append((self.command, url.path, self.headers.get("Range")))
        faults = self.site.faults.get(url.path)
        fault = faults.pop(0) if faults else None
        if fault and fault[0] != "cut":
            return self.reply(fault[0], b"", fault[1])
        if "list-type" in url.query:
            return self.reply(200, self.site.listing(url.path, url.query), {})
        body = self.site.files.get(url.path)
        if body is None:
            return self.reply(404, b"", {})
        start, end, status = 0, len(body) - 1, 200
        if wanted := self.headers.get("Range"):
            first, _, last = wanted.removeprefix("bytes=").partition("-")
            start, end, status = int(first), min(int(last), len(body) - 1) if last else len(body) - 1, 206
            if start >= len(body):
                return self.reply(416, b"", {"Content-Range": f"bytes */{len(body)}"})
        part = body[start:end + 1]
        headers = {"Content-Length": str(len(part)), **({"Content-Range": f"bytes {start}-{end}/{len(body)}"} if status == 206 else {})}
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if with_body:
            self.wfile.write(part[:fault[1]] if fault else part)  # a cut: fewer bytes than it said, then the connection closes


class Site:
    """The server: `files` by path, `faults` (a path's next answers: a status and its headers, or ("cut", bytes sent)) and every request."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.faults: dict[str, list] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.page = 2  # keys a listing gives at a time
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), type("Bound", (Handler,), {"site": self}))
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def listing(self, bucket: str, query: str) -> bytes:
        q = urllib.parse.parse_qs(query)
        keys = sorted(p[len(bucket):] for p in self.files if p.startswith(bucket + q["prefix"][0]))
        at = int(q.get("continuation-token", ["0"])[0])
        rows = "".join(f"<Contents><Key>{k}</Key><Size>{len(self.files[bucket + k])}</Size></Contents>" for k in keys[at:at + self.page])
        more = at + self.page < len(keys)
        return (f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{rows}'
                f"<IsTruncated>{str(more).lower()}</IsTruncated>" + (f"<NextContinuationToken>{at + self.page}</NextContinuationToken>" if more else "")
                + "</ListBucketResult>").encode()

    def gets(self, path: str) -> list[tuple[str, str, str | None]]:
        return [r for r in self.requests if r[0] == "GET" and r[1] == path]


@pytest.fixture(autouse=True)
def only_the_local_server(monkeypatch):
    real = urllib.request.urlopen

    def local(request, *args, **kwargs):
        host = urllib.parse.urlsplit(request.full_url).hostname
        assert host == "127.0.0.1", f"a test reached {request.full_url}"
        return real(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local)


@pytest.fixture
def site():
    server = Site()
    yield server
    server.server.shutdown()
    server.server.server_close()


@pytest.fixture
def waits() -> list[float]:
    return []


@pytest.fixture
def fetcher(waits):
    return fetch.Fetcher(sleep=waits.append, workers=2, tries=4)


@pytest.fixture
def sources(tmp_path) -> Path:
    return tmp_path / "mmsb-sources"


# -- the manner ---------------------------------------------------------------------------------------------------------------

def test_a_busy_server_is_waited_for_as_it_says_and_a_silent_one_by_doubling(site, fetcher, waits, tmp_path):
    site.files["/a"] = b"0123456789"
    site.faults["/a"] = [(429, {"Retry-After": "7"}), (503, {}), (503, {}), ]
    assert fetcher.download(site.url + "a", tmp_path / "a") == "fetched" and (tmp_path / "a").read_bytes() == b"0123456789"
    assert waits == [7.0, 2.0, 4.0], "the server's 7 seconds, then 2 and 4, as a try 2 and a try 3 that failed without a word"
    assert len(site.gets("/a")) == 4


def test_a_server_that_stays_busy_is_given_up_after_the_tries_and_says_so(site, fetcher, waits, tmp_path):
    site.faults["/a"] = [(503, {})] * 9
    site.files["/a"] = b"x"
    with pytest.raises(fetch.FetchError, match=r"/a: HTTP 503 .* on each of 4 tries"):
        fetcher.download(site.url + "a", tmp_path / "a")
    assert waits == [1.0, 2.0, 4.0] and not (tmp_path / "a").exists()


def test_a_retry_after_is_seconds_or_a_date_and_one_that_asks_too_long_stops_the_command():
    assert fetch.Fetcher.wait("12", 1) == 12.0 and fetch.Fetcher.wait(None, 1) == 1.0 and fetch.Fetcher.wait("not a time", 3) == 4.0
    assert fetch.Fetcher.wait(None, 20) == fetch.WAIT_MAX, "the doubling stops at a minute"
    assert 25 < fetch.Fetcher.wait(email.utils.formatdate(time.time() + 30, usegmt=True), 1) <= 30
    assert fetch.Fetcher.wait(email.utils.formatdate(time.time() - 30, usegmt=True), 1) == 0.0, "a date that has passed"
    with pytest.raises(fetch.FetchError, match="asks to wait 7200 s: run the same command again later"):
        fetch.Fetcher.wait("7200", 1)


def test_a_connection_that_drops_is_resumed_from_what_arrived(site, fetcher, waits, tmp_path):
    site.files["/a"] = bytes(range(100))
    site.faults["/a"] = [("cut", 30)]
    assert fetcher.download(site.url + "a", tmp_path / "a") == "fetched"
    assert (tmp_path / "a").read_bytes() == bytes(range(100)) and not (tmp_path / "a.part").exists()
    assert [r[2] for r in site.gets("/a")] == [None, "bytes=30-"], "the second try asked for the rest"
    assert waits == [1.0]


def test_a_part_left_by_an_earlier_run_is_continued_and_one_longer_than_the_file_is_started_again(site, fetcher, waits, tmp_path):
    site.files["/a"] = bytes(range(100))
    (tmp_path / "a.part").write_bytes(bytes(range(40)))
    fetcher.download(site.url + "a", tmp_path / "a")
    assert (tmp_path / "a").read_bytes() == bytes(range(100)) and [r[2] for r in site.gets("/a")] == ["bytes=40-"]
    site.files["/b"] = b"short"
    (tmp_path / "b.part").write_bytes(b"x" * 50)
    fetcher.download(site.url + "b", tmp_path / "b")
    assert (tmp_path / "b").read_bytes() == b"short" and [r[2] for r in site.gets("/b")] == ["bytes=50-", None]


def test_a_file_of_the_right_size_is_left_alone_and_one_of_another_size_is_fetched_again(site, fetcher, tmp_path):
    site.files["/a"] = b"0123456789"
    (tmp_path / "a").write_bytes(b"0123456789")
    assert fetcher.download(site.url + "a", tmp_path / "a") == "kept"
    assert [r[0] for r in site.requests] == ["HEAD"], "the server's size asked, no file fetched"
    assert fetcher.download(site.url + "a", tmp_path / "a", size=10) == "kept" and len(site.requests) == 1, "a size the listing gave asks nothing"
    (tmp_path / "a").write_bytes(b"0123")
    assert fetcher.download(site.url + "a", tmp_path / "a", size=10) == "fetched" and (tmp_path / "a").read_bytes() == b"0123456789"


def test_a_file_that_is_not_the_size_its_listing_said_is_refused_and_not_kept(site, fetcher, tmp_path):
    site.files["/a"] = b"0123456789"
    with pytest.raises(fetch.FetchError, match="came as 10 bytes, not the 12 its listing says"):
        fetcher.download(site.url + "a", tmp_path / "a", size=12)
    assert not (tmp_path / "a").exists() and not (tmp_path / "a.part").exists()


def test_a_missing_file_is_not_there_unless_it_is_optional_and_then_it_is_absent(site, fetcher, tmp_path):
    with pytest.raises(fetch.NotFound, match="/nothing is not there"):
        fetcher.download(site.url + "nothing", tmp_path / "nothing")
    tally = fetcher.fetch_all("x", tmp_path, [fetch.Job(site.url + "nothing", "n", optional=True)])
    assert dict(tally) == {"absent": 1} and not (tmp_path / "n").exists()
    with pytest.raises(fetch.NotFound):
        fetcher.fetch_all("x", tmp_path, [fetch.Job(site.url + "nothing", "n")])


# -- the sources --------------------------------------------------------------------------------------------------------------

CLIPS = {  # name -> (hour folder, is it a clip V1 reads)
    "2018-03-12.10-00-00.10-05-00.school.G336.r13.avi": ("10", True),
    "2018-03-12.10-00-09.10-05-09.hospital.G341.r13.avi": ("10", True),
    "2018-03-12.11-10-00.11-15-00.hospital.G436.r13.avi": ("11", True),
    "2018-03-12.11-15-00.11-20-00.school.G336.r13.avi": ("11", False),  # begins at the end of the window
    "2018-03-12.10-00-00.10-05-00.hospital.G474.r13.avi": ("10", False),  # an infrared camera
}


def serve_meva(site: Site, patch) -> None:
    for name, (hour, _) in CLIPS.items():
        site.files[f"/bucket/drops-123-r13/2018-03-12/{hour}/{name}"] = name.encode() * 3
    clip = lambda n: n.removesuffix(".r13.avi")  # noqa: E731
    first, second, third = list(CLIPS)[:3]
    site.files[f"/labels/kitware-meva-training/2018-03-12/10/{clip(first)}.activities.yml"] = b"- {meta: marked empty_37}\n"
    site.files[f"/labels/kitware/2018-03-12/10/{clip(second)}.activities.yml"] = b"- {meta: marked empty_37}\n"
    site.files[f"/labels/kitware/2018-03-12/11/{clip(third)}.activities.yml"] = b"- {meta: marked empty_37}\n"
    site.files["/labels/kitware/status.csv"] = b"person_opens_trunk, 12\n"
    patch.setattr(fetch, "MEVA_BUCKET", site.url + "bucket/")
    patch.setattr(fetch, "MEVA_LABELS", site.url + "labels/")
    patch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "v1_security": 3})


def test_v1_clips_come_from_the_listing_with_a_label_file_each_from_the_one_set_that_has_it(site, fetcher, sources, monkeypatch, capsys):
    serve_meva(site, monkeypatch)
    fetch.fetch(["v1_security"], sources, fetcher)
    folder = sources / "meva"
    names = sorted(p.name for p in (folder / "video" / "2018-03-12").rglob("*.avi"))
    assert names == sorted(n for n, (_, wanted) in CLIPS.items() if wanted), "the three clips of the window, not the other two"
    assert (folder / "video/2018-03-12/11" / list(CLIPS)[2]).read_bytes() == list(CLIPS)[2].encode() * 3
    labels = folder / meva.LABELS
    found = sorted(str(p.relative_to(labels)).replace("\\", "/") for p in labels.rglob("*") if p.is_file())
    first, second, third = (n.removesuffix(".r13.avi") for n in list(CLIPS)[:3])
    assert found == sorted([f"kitware-meva-training/2018-03-12/10/{first}.activities.yml", f"kitware/2018-03-12/10/{second}.activities.yml",
                            f"kitware/2018-03-12/11/{third}.activities.yml", "kitware/status.csv"])
    assert "v1_security: the 3 files its build needs are in" in capsys.readouterr().out
    assert all(meva.wanted(p.name) for p in (folder / "video").rglob("*.avi")), "what the builder reads"
    site.requests.clear()
    fetch.fetch(["v1_security"], sources, fetcher)
    assert not [r for r in site.requests if r[1].endswith(".avi")], "a clip that is there, of the size the listing gave, is not asked for again"


def test_v1_is_refused_before_a_clip_is_fetched_when_the_listing_is_not_what_the_build_needs(site, fetcher, sources, monkeypatch):
    serve_meva(site, monkeypatch)
    monkeypatch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "v1_security": 4})
    with pytest.raises(fetch.FetchError, match=r"v1_security: 3 clips, not the 4 its build needs"):
        fetch.fetch(["v1_security"], sources, fetcher)
    assert not any(r[1].endswith(".avi") for r in site.requests) and not sources.exists()


def test_v1_says_when_a_clip_has_no_label_file(site, fetcher, sources, monkeypatch):
    serve_meva(site, monkeypatch)
    del site.files[f"/labels/kitware/2018-03-12/11/{list(CLIPS)[2].removesuffix('.r13.avi')}.activities.yml"]
    with pytest.raises(fetch.FetchError, match="2 label files for 3 clips"):
        fetch.fetch(["v1_security"], sources, fetcher)


def test_the_listing_is_read_page_by_page(site, fetcher):
    for n in range(5):
        site.files[f"/bucket/p/{n}"] = b"x" * n
    site.files["/bucket/other/0"] = b"y"
    assert fetch.listing(fetcher, site.url + "bucket/", "p/") == [(f"p/{n}", n) for n in range(5)]
    assert len([r for r in site.requests if r[1] == "/bucket/"]) == 3, "five keys, two to a page"


def labels_zip(photos: list[tuple[str, str, str]]) -> bytes:
    data = {"categories": [{"id": 1, "name": "deer"}, {"id": 2, "name": "empty"}],
            "images": [{"id": f"i{n}", "file_name": name, "location": loc, "date_captured": f"{day} 10:00:0{n}"}
                       for n, (name, loc, day) in enumerate(photos)],
            "annotations": [{"image_id": f"i{n}", "category_id": 1 + n % 2} for n in range(len(photos))]}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(cct.MEMBER, json.dumps(data))
    return out.getvalue()


def serve_cct(site: Site, patch) -> list[str]:
    photos = [("a.jpg", "23", "2013-12-16"), ("b.jpg", "23", "2013-12-17"), ("c.jpg", "23", "2014-02-28"), ("far.jpg", "24", "2013-12-12"),
              ("before.jpg", "23", "2013-09-09"), ("after.jpg", "23", "2014-03-01")]  # the card: location 23, 10 September to 28 February
    site.files["/labels.zip"] = labels_zip(photos)
    for name, *_ in photos:
        site.files[f"/images/{name}"] = name.encode() * 50
    patch.setattr(fetch, "CCT_LABELS", site.url + "labels.zip")
    patch.setattr(fetch, "CCT_IMAGES", site.url + "images/")
    patch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "i3_trailcam": 3})
    return ["a.jpg", "b.jpg", "c.jpg"]


def test_i3_gets_the_labels_and_the_photos_of_the_card_only_and_is_counted_by_its_own_reader(site, fetcher, sources, monkeypatch):
    names = serve_cct(site, monkeypatch)
    fetch.fetch(["i3_trailcam"], sources, fetcher)
    folder = sources / "caltech-camera-traps"
    assert sorted(p.name for p in (folder / "images").iterdir()) == names and (folder / "images/b.jpg").read_bytes() == b"b.jpg" * 50
    assert [p.file for p in cct.read_labels(str(folder))[0]] == names
    site.requests.clear()
    fetch.fetch(["i3_trailcam"], sources, fetcher)
    assert not site.gets("/images/a.jpg") and [r[0] for r in site.requests if r[1] == "/images/a.jpg"] == ["HEAD"]
    assert not any(r[1] in ("/images/far.jpg", "/images/before.jpg", "/images/after.jpg") for r in site.requests), "another camera, other days"


def test_i3_is_refused_with_fewer_photos_than_its_labels_name_at_the_days_it_is_asked_for(site, fetcher, sources, monkeypatch):
    from mmsb.scenarios import i3_trailcam as I3

    serve_cct(site, monkeypatch)
    folder = sources / "caltech-camera-traps"
    fetch.fetch(["i3_trailcam"], sources, fetcher)
    (folder / "images" / "c.jpg").unlink()  # (of 28 February: in the standard card, not in the tiny week)
    photos = cct.read_labels(str(folder))[0]
    with pytest.raises(FileNotFoundError, match=r"1 of the 3 photos this card needs are not in .*mmsb fetch --scenario i3_trailcam"):
        cct.check_photos(str(folder), photos)
    with pytest.raises(FileNotFoundError, match=r"1 of the 3 photos this card needs"):
        I3.chosen_of(str(folder), photos, *cct.CARD_DAYS)
    assert [p.file for p in I3.chosen_of(str(folder), photos, *I3.SCALES["tiny"])] == ["a.jpg", "b.jpg"], "the tiny week is all there"
    fetch.fetch(["i3_trailcam"], sources, fetcher)
    assert (folder / "images" / "c.jpg").is_file(), "a fetch brings back the photo that was gone"
    monkeypatch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "i3_trailcam": 4})
    with pytest.raises(fetch.FetchError, match="i3_trailcam: 3 photos, not the 4"):
        fetch.fetch(["i3_trailcam"], sources, fetcher)


def serve_fsd(site: Site, patch) -> list[str]:
    ids = ["100", "250", "3"]
    site.files["/fsd/labels/eval.csv"] = ("fname,labels,mids\n" + "".join(f'{i},"Dog,Animal","/m/1,/m/2"\n' for i in ids)).encode()
    site.files["/fsd/labels/vocabulary.csv"] = b"0,Dog,/m/1\n1,Animal,/m/2\n"
    site.files["/fsd/metadata/pp_pnp_ratings_FSD50K.json"] = b"{}"
    for i in ids:
        site.files[f"/fsd/clips/eval/{i}.wav"] = b"RIFF" + bytes(76) + b"\x01\x00"  # 82 bytes: samples after the 78 of the header
    site.files["/fsd/clips/eval/9999.wav"] = b"not in the labels"
    patch.setattr(fetch, "FSD50K", site.url + "fsd/")
    patch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "m4_samples": 3})
    return ids


def test_m4_gets_the_labels_and_the_evaluation_clips_the_labels_name(site, fetcher, sources, monkeypatch):
    ids = serve_fsd(site, monkeypatch)
    fetch.fetch(["m4_samples"], sources, fetcher)
    folder = sources / "fsd50k"
    assert sorted(p.name for p in (folder / "clips" / "eval").iterdir()) == sorted(f"{i}.wav" for i in ids), "no clip the labels do not name"
    assert {c.id for c in fsd.read_labels(str(folder))[0]} == set(ids)
    assert not any(r[1].endswith("9999.wav") for r in site.requests)


def test_a_count_that_is_not_the_builds_stops_a_fetch_before_the_clips(site, fetcher, sources, monkeypatch):
    serve_fsd(site, monkeypatch)
    monkeypatch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "m4_samples": 10231})
    with pytest.raises(fetch.FetchError, match="m4_samples: 3 clips, not the 10231 its build needs"):
        fetch.fetch(["m4_samples"], sources, fetcher)
    assert not (sources / "fsd50k" / "clips").exists() and (sources / "fsd50k" / "labels" / "eval.csv").is_file()


def wav(samples: np.ndarray) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(4)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.astype("<i2").tobytes())
    return out.getvalue()


def zip_of(members: dict[str, bytes], method: int) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", method) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return out.getvalue()


PREFIX = "Original/audio/Node7/audio/"
SINS_MEMBERS = {  # member -> is it a minute file that plays in the day A1 uses (2 February 2017; one that began in the last minute before it too)
    "Node7_20170201_235925_385_audio.wav": True,
    "Node7_20170202_120000_000_audio.wav": True,
    "Node7_20170130_120000_000_audio.wav": False,
    "Node7_20170202_235916_439_audio.wav": True,
    "Node7_20170203_000016_439_audio.wav": False,
}


def serve_sins(site: Site, patch) -> dict[str, np.ndarray]:
    """Two zips (one deflated, one stored) of the members above and a readme; returns the samples of each minute file."""
    rng = np.random.default_rng(7)
    samples = {name: rng.integers(-20000, 20000, (1600, 4), dtype=np.int16) for name in SINS_MEMBERS}
    names = list(SINS_MEMBERS)
    site.files["/z/Node7_audio_01.zip/content"] = zip_of({PREFIX + n: wav(samples[n]) for n in names[:3]}, zipfile.ZIP_DEFLATED)
    site.files["/z/Node7_audio_02.zip/content"] = zip_of({**{PREFIX + n: wav(samples[n]) for n in names[3:]}, "readme.txt": b"hello"},
                                                         zipfile.ZIP_STORED)
    site.files["/labels/living_labels.csv"] = b"Class;Start time;Stop time\n"
    site.files["/labels/labels.csv"] = b"Class;Start time;Stop time\n"
    patch.setattr(fetch, "SINS_ZIPS", site.url + "z/")
    patch.setattr(fetch, "SINS_LABELS", site.url + "labels/")
    patch.setattr(fetch, "SINS_ZIP_NAMES", ("Node7_audio_01.zip", "Node7_audio_02.zip"))
    patch.setattr(fetch, "WINDOW", 64)  # so that the directory and each member are separate reads, as in a zip of gigabytes
    patch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "a1_home_audio": 3})
    return samples


def test_a1_reads_the_minute_files_of_the_day_out_of_the_zips_and_keeps_them_as_flac_with_all_four_channels(site, fetcher, sources, monkeypatch,
                                                                                                         capsys):
    samples = serve_sins(site, monkeypatch)
    fetch.fetch(["a1_home_audio"], sources, fetcher)
    out = capsys.readouterr().out
    assert "academic research only" in out and "do not copy it anywhere others can fetch it from" in out, "the licence is said first"
    folder = sources / "sins"
    want = sorted(n.replace(".wav", ".flac") for n, inside in SINS_MEMBERS.items() if inside)
    found = sorted(p.name for p in (folder / "node07").rglob("*.flac"))
    assert found == want and (folder / "node07/2017-02-01" / want[0]).is_file() and (folder / "node07/2017-02-02" / want[1]).is_file()
    for name, inside in SINS_MEMBERS.items():
        if inside:
            flac = next((folder / "node07").rglob(name.replace(".wav", ".flac")))
            assert np.array_equal(fetch.decoded(flac), samples[name]), "lossless, all four channels"
            assert np.array_equal(sins.first_channel(flac), samples[name][:, 0]), "and the file is what A1's builder reads"
    assert (folder / "labels/living_labels.csv").is_file() and (folder / "labels/labels.csv").is_file() and not list(folder.rglob("*.part"))
    assert "readme" not in str(list(folder.rglob("*")))
    before = {p: p.stat().st_mtime_ns for p in (folder / "node07").rglob("*.flac")}
    first_run = len(site.requests)
    fetch.fetch(["a1_home_audio"], sources, fetcher)
    assert before == {p: p.stat().st_mtime_ns for p in (folder / "node07").rglob("*.flac")}, "a file that is there is left alone"
    assert len(site.requests) - first_run < first_run, "and only the zips' directories are read again"


def test_a_minute_file_that_did_not_arrive_whole_is_refused_by_its_crc(site, fetcher, sources, monkeypatch):
    serve_sins(site, monkeypatch)
    body = bytearray(site.files["/z/Node7_audio_02.zip/content"])  # stored: the samples are in it as they are
    at = bytes(body).index(b"RIFF")
    body[at + 100] ^= 0xFF
    site.files["/z/Node7_audio_02.zip/content"] = bytes(body)
    with pytest.raises(fetch.FetchError, match="did not arrive whole: its size or CRC-32 is not the zip's"):
        fetch.fetch(["a1_home_audio"], sources, fetcher)


def test_a1_is_refused_when_the_zips_hold_fewer_minute_files_than_its_build_needs(site, fetcher, sources, monkeypatch):
    serve_sins(site, monkeypatch)
    monkeypatch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "a1_home_audio": 1441})
    with pytest.raises(fetch.FetchError, match="a1_home_audio: 3 minute files, not the 1441"):
        fetch.fetch(["a1_home_audio"], sources, fetcher)
    assert not list((sources / "sins").rglob("*.flac"))


# -- the command --------------------------------------------------------------------------------------------------------------

def test_the_command_fetches_every_scenario_with_all_and_names_none_but_the_four(site, sources, monkeypatch, capsys):
    serve_meva(site, monkeypatch)
    serve_cct(site, monkeypatch)
    serve_fsd(site, monkeypatch)
    serve_sins(site, monkeypatch)
    assert cli.main(["fetch", "--scenario", "all", "--sources", str(sources), "--workers", "2"]) == 0
    assert sorted(p.name for p in sources.iterdir()) == ["caltech-camera-traps", "fsd50k", "meva", "sins"]
    out = capsys.readouterr().out
    for scenario in fetch.SCENARIOS:
        assert f"{scenario}: the 3 files its build needs are in" in out
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--scenario", "v7_screen"])
    with pytest.raises(SystemExit):
        cli.main(["fetch"])


def test_the_command_takes_a_scenario_twice_once_and_says_what_stopped_it(site, sources, monkeypatch, capsys):
    serve_fsd(site, monkeypatch)
    monkeypatch.setattr(fetch, "EXPECTED", {**fetch.EXPECTED, "m4_samples": 5})
    assert cli.main(["fetch", "--scenario", "m4_samples", "--scenario", "m4_samples", "--sources", str(sources)]) == 1
    assert "error: m4_samples: 3 clips, not the 5 its build needs" in capsys.readouterr().err


def test_a_sources_folder_an_agent_could_not_be_kept_out_of_is_refused_before_anything_is_fetched(site, tmp_path, monkeypatch, capsys):
    serve_fsd(site, monkeypatch)
    assert cli.main(["fetch", "--scenario", "m4_samples", "--sources", str(tmp_path / "data")]) == 1
    assert "ordinary output could hold" in capsys.readouterr().err and not site.requests


def test_the_default_sources_folder_is_the_one_the_builders_read(monkeypatch, tmp_path):
    root = realdump.sources_root()
    assert [m.SOURCE for m in (meva, cct, sins, fsd)] == [root / n for n in ("meva", "caltech-camera-traps", "sins", "fsd50k")]
    assert cli.parser_of().parse_args(["fetch", "--scenario", "all"]).sources == root, "the folder the builders read under"
    monkeypatch.setenv("MMSB_SOURCES", str(tmp_path / "mine"))
    assert cli.parser_of().parse_args(["fetch", "--scenario", "all"]).sources == tmp_path / "mine" == realdump.sources_root()


def test_a_fetch_into_another_folder_says_that_a_build_reads_mmsb_sources(site, sources, monkeypatch, capsys):
    serve_fsd(site, monkeypatch)
    monkeypatch.setenv("MMSB_SOURCES", str(sources.parent / "elsewhere-sources"))
    assert cli.main(["fetch", "--scenario", "m4_samples", "--sources", str(sources)]) == 0
    assert f"note: a build reads MMSB_SOURCES, which is {sources.parent / 'elsewhere-sources'}; set it to {sources}" in capsys.readouterr().out
    monkeypatch.setenv("MMSB_SOURCES", str(sources))
    assert cli.main(["fetch", "--scenario", "m4_samples", "--sources", str(sources)]) == 0
    assert "note:" not in capsys.readouterr().out


# -- what the answers rest on, and what a listing may name ------------------------------------------------------------------------

def test_the_labels_and_the_mirror_are_read_at_a_commit_and_not_at_a_branch_that_moves():
    for url, revision in ((fetch.MEVA_LABELS, fetch.MEVA_REVISION), (fetch.SINS_LABELS, fetch.SINS_REVISION), (fetch.FSD50K, fetch.FSD50K_REVISION)):
        assert re.fullmatch(r"[0-9a-f]{40}", revision) and f"/{revision}/" in url and not re.search(r"/(master|main)/", url), url


@pytest.mark.parametrize("name", ["..", ".", "", "a/b", "../x", "..\\..\\x", "C:x", "x\\y", "/etc/x"])
def test_a_name_that_is_not_one_file_name_is_refused(name):
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch.plain(name)
    assert fetch.plain("100.wav") == "100.wav" and fetch.plain("59641d3e-23d2.jpg") == "59641d3e-23d2.jpg" and fetch.plain("a b.jpg") == "a b.jpg"


def test_a_listing_that_names_a_place_outside_the_folder_writes_nothing_there(site, fetcher, sources, monkeypatch):
    """A CCT file name, an FSD50K name and an S3 key that climb out of the sources folder."""
    outside = sources.parent / "evil"
    serve_cct(site, monkeypatch)
    site.files["/labels.zip"] = labels_zip([("..\\..\\evil.jpg", "23", "2013-12-16")])
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch.fetch(["i3_trailcam"], sources, fetcher)
    serve_fsd(site, monkeypatch)
    site.files["/fsd/labels/eval.csv"] = b'fname,labels,mids\n..\\..\\evil,"Dog","/m/1"\n'
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch.fetch(["m4_samples"], sources, fetcher)
    serve_meva(site, monkeypatch)
    name = list(CLIPS)[0]
    site.files[f"/bucket/drops-123-r13/2018-03-12/10/../../../evil/{name}"] = b"x"
    with pytest.raises(fetch.FetchError, match="is not drops-123-r13/2018-03-12/<hour>/<clip>"):
        fetch.fetch(["v1_security"], sources, fetcher)
    assert not outside.exists() and not any(r[1].endswith("evil.jpg") or r[1].endswith("evil.wav") for r in site.requests)
    assert not (sources / "fsd50k" / "clips").exists() and not (sources / "meva" / "video").exists()


def test_a_429_with_no_retry_after_is_waited_out_by_the_minute_so_six_tries_cover_a_five_minute_window(site, waits, tmp_path):
    fetcher = fetch.Fetcher(sleep=waits.append, workers=1)  # (its own tries)
    site.files["/a"] = b"0123456789"
    site.faults["/a"] = [(429, {})] * 5
    assert fetcher.download(site.url + "a", tmp_path / "a") == "fetched"
    assert waits == [fetch.WAIT_MAX] * 5 and sum(waits) >= 300
    site.faults["/b"] = [(503, {})] * 3
    site.files["/b"] = b"x"
    waits.clear()
    fetcher.download(site.url + "b", tmp_path / "b")
    assert waits == [1.0, 2.0, 4.0], "a server that is merely failing is still waited for by doubling"
