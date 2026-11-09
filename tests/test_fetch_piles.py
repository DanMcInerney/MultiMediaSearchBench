"""`mmsb fetch` for the piles of media (mmsb/fetch_piles.py) against a local HTTP server that serves small stand-ins of what each place
publishes, in the shape its API has: an archive.org item's metadata, a Hugging Face tree that comes in pages, a Zenodo record, a course's
download page, the iNaturalist API, and files and zips at fixed addresses. The politeness is `fetch.Fetcher`'s and is tested in
test_fetch.py; here are the listings, the counts and budgets that refuse before anything large is fetched, the layout each source is
written in, the labels that come before the media, and a second run that fetches nothing. No test reaches the network: any address but
the local server's fails the test."""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mmsb import cli, fetch, fetch_piles


class Handler(BaseHTTPRequestHandler):
    site: Site

    def log_message(self, *args) -> None:
        pass

    def do_HEAD(self) -> None:
        self.answer(False)

    def do_GET(self) -> None:
        self.answer(True)

    def answer(self, with_body: bool) -> None:
        url = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(url.path)
        self.site.requests.append((self.command, path + (f"?{url.query}" if url.query else ""), self.headers.get("Range")))
        if path in self.site.pages:  # an API: a body, and the headers it comes with
            body, headers = self.site.pages[path](urllib.parse.parse_qs(url.query))
            return self.send(200, body, headers, with_body)
        body = self.site.files.get(path)
        if body is None:
            return self.send(404, b"", {}, with_body)
        start, end, status = 0, len(body) - 1, 200
        if wanted := self.headers.get("Range"):
            first, _, last = wanted.removeprefix("bytes=").partition("-")
            start, end, status = int(first), min(int(last), len(body) - 1) if last else len(body) - 1, 206
        headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}"} if status == 206 else {}
        self.send(status, body[start:end + 1], headers, with_body)

    def send(self, status: int, body: bytes, headers: dict[str, str], with_body: bool) -> None:
        self.send_response(status)
        for key, value in {"Content-Length": str(len(body)), **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        if with_body:
            self.wfile.write(body)


class Site:
    """The server: `files` by path, `pages` (an API's path -> a function of the query that gives the body and its headers) and every request."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.pages: dict[str, Callable[[dict], tuple[bytes, dict[str, str]]]] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), type("Bound", (Handler,), {"site": self}))
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def json(self, path: str, value: object, headers: dict[str, str] | None = None) -> None:
        self.pages[path] = lambda query: (json.dumps(value).encode(), headers or {})

    def gets(self, prefix: str) -> list[str]:
        return [path for method, path, _ in self.requests if method == "GET" and path.startswith(prefix)]


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
    return fetch.Fetcher(sleep=waits.append, workers=2, tries=3)


@pytest.fixture
def sources(tmp_path) -> Path:
    return tmp_path / "mmsb-sources"


def tree_of(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)).replace("\\", "/"): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


# -- archive.org: lectures and a podcast ------------------------------------------------------------------------------------------------

def serve_archive(site: Site, patch, item: str = "mit14_x", videos: int = 3) -> dict[str, bytes]:
    """An item of `videos` lectures with a caption file each, and what the site makes of them (a thumbnail, an Ogg derivative, the
    metadata), which are not read."""
    files, listing = {}, []
    for n in range(1, videos + 1):
        for ext, kind in (("mp4", "MPEG4"), ("srt", "SubRip")):
            name = f"{item}_lec{n:02d}_360p.{ext}"
            files[name] = (f"{name} " * (40 if ext == "mp4" else 3)).encode()
            listing.append({"name": name, "size": str(len(files[name])), "format": kind, "source": "original"})
    listing += [{"name": f"{item}_lec01_360p.ogv", "size": "99", "format": "Ogg Video", "source": "derivative"},
                {"name": "__ia_thumb.jpg", "size": "9", "format": "Item Tile", "source": "original"},
                {"name": f"{item}_files.xml", "size": "9", "format": "Metadata", "source": "original"}]
    site.json(f"/metadata/{item}", {"files": listing})
    for name, body in files.items():
        site.files[f"/download/{item}/{name}"] = body
    patch.setattr(fetch_piles, "ARCHIVE", site.url)
    patch.setitem(fetch_piles.COUNTS, "ocw-14-01", videos)
    patch.setitem(fetch_piles.ARCHIVE_ITEMS, "ocw-14-01", item)
    return files


def test_an_archive_item_brings_its_videos_and_their_captions_and_nothing_the_site_made_of_them(site, fetcher, sources, monkeypatch):
    files = serve_archive(site, monkeypatch)
    assert fetch_piles.fetch_ocw_14_01(fetcher, sources) == 6
    assert tree_of(sources / "ocw-14-01") == files, "a folder of the item's own videos and captions, flat, as the publisher names them"
    site.requests.clear()
    assert fetch_piles.fetch_ocw_14_01(fetcher, sources) == 6
    assert not site.gets("/download/"), "a file that is there, of the size the listing gave, is not asked for again (not even for its size)"


def test_the_captions_come_before_the_videos_and_labels_only_stops_after_them(site, fetcher, sources, monkeypatch):
    serve_archive(site, monkeypatch)
    assert fetch_piles.fetch_ocw_14_01(fetcher, sources, labels_only=True) == 3
    assert sorted(p.suffix for p in (sources / "ocw-14-01").iterdir()) == [".srt"] * 3 and not site.gets("/download/mit14_x/mit14_x_lec01_360p.mp4")
    fetch_piles.fetch_ocw_14_01(fetcher, sources)
    assert len(list((sources / "ocw-14-01").glob("*.mp4"))) == 3


def test_an_item_with_another_count_of_videos_is_refused_before_one_is_fetched(site, fetcher, sources, monkeypatch):
    serve_archive(site, monkeypatch)
    monkeypatch.setitem(fetch_piles.COUNTS, "ocw-14-01", 26)
    with pytest.raises(fetch.FetchError, match=r"ocw-14-01: 3 videos, not the 26 found when it was chosen"):
        fetch_piles.fetch_ocw_14_01(fetcher, sources)
    assert not site.gets("/download/") and not sources.exists()


def test_a_listing_that_comes_to_more_bytes_than_the_source_was_chosen_at_is_refused(site, fetcher, sources, monkeypatch):
    serve_archive(site, monkeypatch)
    monkeypatch.setitem(fetch_piles.BUDGET, "ocw-14-01", 100)
    with pytest.raises(fetch.FetchError, match=r"ocw-14-01: .* GB to fetch, over the .* GB it was chosen at"):
        fetch_piles.fetch_ocw_14_01(fetcher, sources)
    assert not site.gets("/download/")


def test_a_file_name_that_climbs_out_of_the_folder_is_refused(site, fetcher, sources, monkeypatch):
    serve_archive(site, monkeypatch, videos=1)
    meta = json.loads(site.pages["/metadata/mit14_x"]({})[0])
    meta["files"][0]["name"] = "..\\..\\evil.mp4"
    site.json("/metadata/mit14_x", meta)
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch_piles.fetch_ocw_14_01(fetcher, sources)
    assert not (sources.parent / "evil.mp4").exists() and not site.gets("/download/")


def test_archive_orgs_busy_answer_460_is_waited_for_and_then_the_file_comes(site, sources, waits, monkeypatch):
    files = serve_archive(site, monkeypatch, videos=1)
    name = next(n for n in files if n.endswith(".mp4"))
    original = Handler.answer
    refused = []

    def busy(self, with_body):
        if self.path.endswith(name) and len(refused) < 2:
            refused.append(1)
            return self.send(460, b"", {}, with_body)
        return original(self, with_body)

    monkeypatch.setattr(Handler, "answer", busy)
    fetch_piles.fetch_ocw_14_01(fetch.Fetcher(sleep=waits.append, workers=1, tries=4), sources)
    assert (sources / "ocw-14-01" / name).read_bytes() == files[name] and waits == [1.0, 2.0], "two refusals, waited for by doubling"


# -- Hugging Face ---------------------------------------------------------------------------------------------------------------------

REVISION = "0123456789abcdef0123456789abcdef01234567"


def serve_tree(site: Site, repo: str, files: dict[str, bytes], page: int = 2) -> None:
    """The repository's tree at REVISION in pages of `page` entries (and a folder entry, which is not a file), and its files at
    their resolve addresses."""
    entries = [{"type": "directory", "path": "soccertrack"}] + [{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]

    def tree(query: dict) -> tuple[bytes, dict[str, str]]:
        assert query.get("recursive") == ["true"]
        at = int(query.get("cursor", ["0"])[0])
        link = {"Link": f'<{site.url}api/datasets/{repo}/tree/{REVISION}?recursive=true&cursor={at + page}>; rel="next"'} if at + page < len(entries) else {}
        return json.dumps(entries[at:at + page]).encode(), link

    site.pages[f"/api/datasets/{repo}/tree/{REVISION}"] = tree
    for path, body in files.items():
        site.files[f"/datasets/{repo}/resolve/{REVISION}/{path}"] = body


def st2_files() -> dict[str, bytes]:
    """Three matches (ids 9, 10 and 118, which sort by number: 9, 10, 118), two clips in the first two and one in the third; a README."""
    files = {"README.md": b"cc-by-4.0"}
    for match, clips in (("118", 1), ("10", 2), ("9", 2)):
        for n in range(clips):
            files[f"soccertrack/clips/st{match}_h1_c{n:03d}.mp4"] = f"clip {match} {n}".encode() * 30
            files[f"soccertrack/labels/st{match}_h1_c{n:03d}.Labels-ball.json"] = json.dumps([{"label": "pass"}]).encode()
    files["tokens/soccertrack/st118_h1_c000.npz"] = b"not read"
    return files


def serve_st2(site: Site, patch, matches: int = 2, count: int = 4) -> dict[str, bytes]:
    files = st2_files()
    serve_tree(site, "o/st2", files)
    patch.setattr(fetch_piles, "HUB", site.url)
    patch.setitem(fetch_piles.HUBS, "st2-followcam", dataclasses.replace(fetch_piles.HUBS["st2-followcam"], repo="o/st2", revision=REVISION))
    patch.setattr(fetch_piles, "ST2_MATCHES", matches)
    patch.setitem(fetch_piles.COUNTS, "st2-followcam", count)
    return files


def test_a_match_is_fetched_whole_with_its_labels_the_first_matches_by_sorted_id_and_nothing_else(site, fetcher, sources, monkeypatch):
    files = serve_st2(site, monkeypatch)
    fetch_piles.fetch_st2_followcam(fetcher, sources)
    want = {p: b for p, b in files.items() if p == "README.md" or re.search(r"st(9|10)_h1", p)}
    assert tree_of(sources / "st2-followcam") == want, "matches 9 and 10 (the first two in the order of their numbers, not of their digits), whole"
    assert not any("npz" in path or "st118" in path for path in site.gets("/datasets/")), "the tokens and a third match are not asked for"
    assert len(site.gets("/api/datasets/")) == 7, "thirteen entries (a folder, a README, ten clips and labels, the tokens), two to a page"
    assert all(f"/resolve/{REVISION}/" in path for path in site.gets("/datasets/")), "read at the commit, not at a branch that moves"


def test_the_labels_of_a_hub_source_come_first_and_the_clips_after(site, fetcher, sources, monkeypatch):
    serve_st2(site, monkeypatch)
    fetch_piles.fetch_st2_followcam(fetcher, sources, labels_only=True)
    found = tree_of(sources / "st2-followcam")
    assert sorted(found) == sorted(["README.md", *(p for p in found if p.endswith(".json"))]) and len(found) == 5 and not site.gets("/datasets/o/st2/resolve/" + REVISION + "/soccertrack/clips")
    site.requests.clear()
    fetch_piles.fetch_st2_followcam(fetcher, sources)
    assert len(tree_of(sources / "st2-followcam")) == 9 and len(site.gets("/datasets/")) == 4, "the four clips only: the labels are there already"


def test_a_match_that_is_missing_a_label_file_is_refused_before_a_clip_is_fetched(site, fetcher, sources, monkeypatch):
    files = serve_st2(site, monkeypatch)
    del files["soccertrack/labels/st9_h1_c001.Labels-ball.json"]
    serve_tree(site, "o/st2", files)
    with pytest.raises(fetch.FetchError, match=r"st2-followcam: the clips of matches 9, 10 and their label files are not the same set"):
        fetch_piles.fetch_st2_followcam(fetcher, sources)
    assert not site.gets("/datasets/")


def test_a_hub_source_with_another_count_than_it_was_chosen_at_is_refused(site, fetcher, sources, monkeypatch):
    serve_st2(site, monkeypatch, count=525)
    with pytest.raises(fetch.FetchError, match=r"st2-followcam: 4 files that make the pile, not the 525"):
        fetch_piles.fetch_st2_followcam(fetcher, sources)
    assert not site.gets("/datasets/")


def test_a_small_hub_source_comes_whole_at_once_with_every_file_but_its_gitattributes_and_in_the_folders_it_keeps(site, fetcher, sources,
                                                                                                                   monkeypatch):
    files = {".gitattributes": b"*.mp4 lfs", "README.md": b"cc-by-nc-4.0", "video_metadata.csv": b"video_name,classification\n",
             "Near Crash/20260106_x_001.mp4": b"near" * 40, "Crash/20260106_y_001.mp4": b"crash" * 40}
    serve_tree(site, "o/dash", files)
    monkeypatch.setattr(fetch_piles, "HUB", site.url)
    monkeypatch.setitem(fetch_piles.HUBS, "aus-dashcam", dataclasses.replace(fetch_piles.HUBS["aus-dashcam"], repo="o/dash", revision=REVISION))
    monkeypatch.setitem(fetch_piles.COUNTS, "aus-dashcam", 2)
    assert fetch_piles.fetch_aus_dashcam(fetcher, sources, labels_only=True) == 4, "labels_only holds nothing back from a small source"
    assert tree_of(sources / "aus-dashcam") == {p: b for p, b in files.items() if p != ".gitattributes"}


def test_a_tree_path_that_names_a_place_outside_the_folder_is_refused(site, fetcher, sources, monkeypatch):
    serve_tree(site, "o/dash", {"../evil.mp4": b"x"})
    monkeypatch.setattr(fetch_piles, "HUB", site.url)
    monkeypatch.setitem(fetch_piles.HUBS, "aus-dashcam", dataclasses.replace(fetch_piles.HUBS["aus-dashcam"], repo="o/dash", revision=REVISION))
    monkeypatch.setitem(fetch_piles.COUNTS, "aus-dashcam", 1)
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch_piles.fetch_aus_dashcam(fetcher, sources)
    assert not (sources.parent / "evil.mp4").exists()


def test_the_repositories_are_read_at_a_commit_and_the_atm_bench_selection_leaves_out_its_videos_and_emails():
    for revision in (fetch_piles.ST2_REVISION, fetch_piles.DASHCAM_REVISION, fetch_piles.SROIE_REVISION, fetch_piles.CORD_REVISION,
                     fetch_piles.ATM_REVISION):
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
    files = [("README.md", 1), ("data/atm-bench/atm-bench.json", 2), ("data/raw_memory/image/20220430_132212.jpg", 3),
             ("data/raw_memory/geocoding_cache/image/20220430_132212_location_name.json", 4), ("data/raw_memory/video/a.mp4", 5),
             ("data/raw_memory/geocoding_cache/video/a_location_name.json", 6), ("data/raw_memory/email/e.json", 7), ("ATM-Bench-Demo.png", 8),
             ("data/processed_memory/readme.md", 9)]
    assert [p for p, _ in fetch_piles.atm_files(files)] == [f[0] for f in files if f[0] not in (
        "data/raw_memory/video/a.mp4", "data/raw_memory/geocoding_cache/video/a_location_name.json", "data/raw_memory/email/e.json", "ATM-Bench-Demo.png")]


# -- Zenodo, the Edinburgh mirror, a course's download page ------------------------------------------------------------------------------

def test_a_zenodo_record_gives_the_files_by_name_and_the_audio_comes_after_the_labels(site, fetcher, sources, monkeypatch):
    names = ("a_annotations.csv", "a_audio.flac", "README.md")
    site.json("/rec/9", {"files": [{"key": n, "size": 40 if "flac" in n else 12} for n in (*names, "not wanted.zip")]})
    for n in names:
        site.files[f"/rec/9/files/{n}/content"] = b"x" * (40 if "flac" in n else 12)
    monkeypatch.setattr(fetch_piles, "ZENODO", site.url + "rec/")
    monkeypatch.setattr(fetch_piles, "BIRDVOX_RECORD", "9")
    monkeypatch.setattr(fetch_piles, "BIRDVOX_FILES", names)
    assert fetch_piles.fetch_birdvox_full_night(fetcher, sources, labels_only=True) == 2
    assert not (sources / "birdvox-full-night/a_audio.flac").exists()
    assert fetch_piles.fetch_birdvox_full_night(fetcher, sources) == 3
    monkeypatch.setattr(fetch_piles, "BIRDVOX_FILES", (*names, "missing.csv"))
    with pytest.raises(fetch.FetchError, match="missing.csv is not in Zenodo record 9"):
        fetch_piles.fetch_birdvox_full_night(fetcher, sources)


def serve_mirror(site: Site, patch, meetings: tuple[str, ...] = ("ES2015a", "IS1006a")) -> None:
    for m in meetings:
        site.files[f"/ami/{m}/audio/{m}.Mix-Headset.wav"] = f"audio {m}".encode() * 10
        if m.startswith("ES"):  # the mirror has a corner camera for the ES meetings only
            site.files[f"/ami/{m}/video/{m}.Corner.avi"] = f"video {m}".encode() * 10
    for name in fetch_piles.AMI_ANNOTATION_FILES:
        site.files[f"/ann/{name}"] = name.encode()
    patch.setattr(fetch_piles, "AMI", site.url + "ami/")
    patch.setattr(fetch_piles, "AMI_ANNOTATIONS", site.url + "ann/")
    patch.setattr(fetch_piles, "AMI_MEETINGS", meetings)


def test_ami_brings_the_annotations_and_for_each_meeting_its_mix_and_the_corner_camera_where_the_mirror_has_one(site, fetcher, sources, monkeypatch,
                                                                                                           capsys):
    serve_mirror(site, monkeypatch)
    assert fetch_piles.fetch_ami(fetcher, sources, labels_only=True) == 6 and not (sources / "ami/signals").exists()
    assert fetch_piles.fetch_ami(fetcher, sources) == 6 + 3
    folder = sources / "ami"
    assert sorted(p for p in tree_of(folder) if p.startswith("signals/")) == [
        "signals/ES2015a/audio/ES2015a.Mix-Headset.wav", "signals/ES2015a/video/ES2015a.Corner.avi", "signals/IS1006a/audio/IS1006a.Mix-Headset.wav"]
    assert "the corner camera is missing for 1 of them" in capsys.readouterr().out
    site.requests.clear()
    fetch_piles.fetch_ami(fetcher, sources)
    assert not [p for p in site.gets("/ami/")], "a second run fetches no signal again"


def test_ami_stops_when_a_mix_is_missing_from_the_mirror(site, fetcher, sources, monkeypatch):
    serve_mirror(site, monkeypatch)
    del site.files["/ami/IS1006a/audio/IS1006a.Mix-Headset.wav"]
    with pytest.raises(fetch.NotFound, match="IS1006a.Mix-Headset.wav is not there"):
        fetch_piles.fetch_ami(fetcher, sources)


def test_icsi_brings_the_annotations_and_the_mix_of_each_meeting(site, fetcher, sources, monkeypatch):
    for name in fetch_piles.ICSI_ANNOTATION_FILES:
        site.files[f"/ann/{name}"] = name.encode()
    for m in ("Bmr001", "Bmr005"):
        site.files[f"/sig/{m}.interaction.wav"] = m.encode() * 20
    monkeypatch.setattr(fetch_piles, "ICSI_ANNOTATIONS", site.url + "ann/")
    monkeypatch.setattr(fetch_piles, "ICSI_SIGNALS", site.url + "sig/")
    monkeypatch.setattr(fetch_piles, "ICSI_MEETINGS", ("Bmr001", "Bmr005"))
    assert fetch_piles.fetch_icsi(fetcher, sources) == 4
    assert sorted(tree_of(sources / "icsi")) == ["annotations/ICSI_original_transcripts.zip", "annotations/ICSI_plus_NXT.zip",
                                                 "signals/Bmr001.interaction.wav", "signals/Bmr005.interaction.wav"]


def serve_course(site: Site, patch, course: str, hrefs: list[str]) -> None:
    page = "<html>" + "".join(f'<a href="{h}">x</a><a class="resource-thumbnail" href="{h}" download>pdf</a>' for h in hrefs) + "</html>"
    site.files[f"/courses/{course}/download/"] = page.encode()
    patch.setattr(fetch_piles, "OCW", site.url + "courses/")


def test_the_pdfs_of_a_courses_download_page_are_each_fetched_once_into_the_courses_folder(site, fetcher, sources, monkeypatch):
    course = "6-006-x"
    serve_course(site, monkeypatch, course, [f"/courses/{course}/ab12_exam%201.pdf", f"/courses/{course}/ab12_exam%201.pdf",
                                             f"https://ocw.mit.edu/courses/{course}/cd34_notes.PDF", f"/courses/{course}/other-course.mp4",
                                             "/courses/9-00sc-y/zz_other.pdf", f"/courses/{course}/resources/page/"])
    for name in ("ab12_exam 1.pdf", "cd34_notes.PDF"):
        site.files[f"/courses/{course}/{name}"] = b"%PDF" + name.encode()
    monkeypatch.setattr(fetch_piles, "OCW_COURSES", (course,))
    assert fetch_piles.fetch_ocw_pages(fetcher, sources) == 2
    assert sorted(tree_of(sources / "ocw-pages")) == [f"{course}/ab12_exam 1.pdf", f"{course}/cd34_notes.PDF"]
    monkeypatch.setattr(fetch_piles, "OCW_COURSES", (course, "empty-course"))
    site.files["/courses/empty-course/download/"] = b"<html>no files</html>"
    with pytest.raises(fetch.FetchError, match=r"ocw-pages: no PDF is linked from .*empty-course/download/"):
        fetch_piles.fetch_ocw_pages(fetcher, sources)


# -- whole zips -----------------------------------------------------------------------------------------------------------------------------

def test_two_zips_are_kept_as_they_are_the_annotations_first_and_the_media_after_and_not_asked_for_again(site, fetcher, sources, monkeypatch):
    site.files["/charades/Charades.zip"] = b"PK-annotations" * 5
    site.files["/charades/Charades_v1_480.zip"] = b"PK-videos" * 500
    monkeypatch.setattr(fetch_piles, "CHARADES", site.url + "charades/")
    assert fetch_piles.fetch_charades(fetcher, sources, labels_only=True) == 1 and not (sources / "charades/Charades_v1_480.zip").exists()
    assert fetch_piles.fetch_charades(fetcher, sources) == 2
    assert tree_of(sources / "charades") == {"Charades.zip": b"PK-annotations" * 5, "Charades_v1_480.zip": b"PK-videos" * 500}
    site.requests.clear()
    fetch_piles.fetch_charades(fetcher, sources)
    assert [m for m, _, _ in site.requests] == ["HEAD", "HEAD"], "the server's size asked for each zip, no zip fetched"


def test_fma_is_its_metadata_zip_and_its_small_zip(site, fetcher, sources, monkeypatch):
    site.files["/fma/fma_metadata.zip"], site.files["/fma/fma_small.zip"] = b"meta" * 9, b"small" * 99
    monkeypatch.setattr(fetch_piles, "FMA", site.url + "fma/")
    assert fetch_piles.fetch_fma(fetcher, sources) == 2 and sorted(tree_of(sources / "fma")) == ["fma_metadata.zip", "fma_small.zip"]


# -- FTC robocalls ------------------------------------------------------------------------------------------------------------------------

def robocall_zip(recordings: int = 3, extra: dict[str, bytes] | None = None) -> bytes:
    import io
    import zipfile

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("LICENSE.md", "README.md", "metadata.csv"):
            z.writestr(f"robocall-audio-dataset-main/{name}", name * 3)
        for n in range(recordings):
            z.writestr(f"robocall-audio-dataset-main/audio-wav-16khz/call{n}_normalized.wav", b"RIFF" + bytes([n]) * 50)
        z.writestr("robocall-audio-dataset-main/pdf_files/case.pdf", b"%PDF")
        z.writestr("robocall-audio-dataset-main/audio-original/call0.mp3", b"not taken")
        z.writestr("robocall-audio-dataset-main/.gitignore", b"not taken")
        for name, body in (extra or {}).items():
            z.writestr(name, body)
    return out.getvalue()


def test_the_robocall_pile_is_written_out_of_the_repositorys_zip_as_the_repository_lays_it_out_and_the_zip_is_removed(site, fetcher, sources, monkeypatch):
    site.files["/repo.zip"] = robocall_zip()
    monkeypatch.setattr(fetch_piles, "ROBOCALL_ZIP", site.url + "repo.zip")
    monkeypatch.setitem(fetch_piles.COUNTS, "robocall-ftc", 3)
    assert fetch_piles.fetch_robocall_ftc(fetcher, sources) == 3 + 3 + 1
    assert sorted(tree_of(sources / "robocall-ftc")) == ["LICENSE.md", "README.md", "audio-wav-16khz/call0_normalized.wav", "audio-wav-16khz/call1_normalized.wav",
                                                          "audio-wav-16khz/call2_normalized.wav", "metadata.csv", "pdf_files/case.pdf"]
    assert not list(sources.glob("*.zip*")), "the zip is not kept, nor a part of it"
    site.requests.clear()
    assert fetch_piles.fetch_robocall_ftc(fetcher, sources) == 7 and not site.requests, "a folder that has them is not fetched again, nor asked about"


def test_a_zip_with_another_count_of_recordings_or_a_member_that_climbs_out_is_refused_and_writes_nothing(site, fetcher, sources, monkeypatch):
    monkeypatch.setattr(fetch_piles, "ROBOCALL_ZIP", site.url + "repo.zip")
    monkeypatch.setitem(fetch_piles.COUNTS, "robocall-ftc", 4)
    site.files["/repo.zip"] = robocall_zip()
    with pytest.raises(fetch.FetchError, match="robocall-ftc: 3 recordings, not the 4 found when it was chosen"):
        fetch_piles.fetch_robocall_ftc(fetcher, sources)
    assert not (sources / "robocall-ftc").exists() and not (sources / "robocall-ftc.zip").exists(), "a zip that is refused is not kept"
    monkeypatch.setitem(fetch_piles.COUNTS, "robocall-ftc", 3)
    site.files["/repo.zip"] = robocall_zip(extra={"robocall-audio-dataset-main/pdf_files/../../evil.pdf": b"x"})
    with pytest.raises(fetch.FetchError, match="is not a plain file name"):
        fetch_piles.fetch_robocall_ftc(fetcher, sources)
    assert not (sources.parent / "evil.pdf").exists() and not (sources / "robocall-ftc/audio-wav-16khz").exists()


# -- iNaturalist ------------------------------------------------------------------------------------------------------------------------

def observation(n: int, licence: str = "cc-by", photo_licence: str | None = None, ext: str = "jpg") -> dict:
    """What the API gives for an observation, with what the pile must not keep: a location, the observer's name and login."""
    return {"id": n, "taxon": {"name": f"Genus species{n}", "preferred_common_name": f"Bird {n}", "rank": "species", "id": 9},
            "observed_on": f"2023-05-{n % 28 + 1:02d}", "time_observed_at": f"2023-05-{n % 28 + 1:02d}T10:00:00-04:00", "license_code": licence,
            "location": "42.4,-76.5", "geojson": {"coordinates": [-76.5, 42.4]}, "place_guess": "Ithaca", "user": {"login": "someone", "name": "Some One"},
            "photos": [{"id": 5000 + n, "license_code": photo_licence or licence, "url": f"https://static.inaturalist.org/photos/{5000 + n}/square.{ext}"},
                       {"id": 6000 + n, "license_code": licence, "url": "https://static.inaturalist.org/photos/x/square.jpg"}]}


def serve_inat(site: Site, patch, observations: list[dict], page: int = 2) -> None:
    def api(query: dict) -> tuple[bytes, dict[str, str]]:
        assert query["user_id"] == ["5669063"] and query["quality_grade"] == ["research"] and query["photo_license"] == ["cc0,cc-by,cc-by-nc"]
        assert (query["d1"], query["d2"], query["per_page"], query["order_by"], query["order"]) == (
            ["2022-01-01"], ["2025-12-31"], [str(page)], ["id"], ["asc"]) and "page" not in query
        after = int(query["id_above"][0])
        results = [o for o in observations if o["id"] > after][:page]
        return json.dumps({"total_results": len(observations), "page": 1, "per_page": page, "results": results}).encode(), {}

    site.pages["/v1/observations"] = api
    patch.setattr(fetch_piles, "INAT_API", site.url + "v1/observations")
    patch.setattr(fetch_piles, "INAT_PHOTOS", site.url + "photos/")
    patch.setattr(fetch_piles, "INAT_PAGE", page)
    for o in observations:
        for ext in ("jpg", "JPG", "jpeg"):
            site.files[f"/photos/{5000 + o['id']}/large.{ext}"] = f"photo {o['id']}".encode() * 20


def test_the_observations_are_paged_by_id_each_request_a_second_after_the_last_and_only_the_kept_fields_are_written(site, fetcher, sources, waits,
                                                                                                                 monkeypatch, capsys):
    obs = [observation(1), observation(2, ext="JPG"), observation(3, licence="cc-by-nc"), observation(4, licence="cc-by", photo_licence="all rights reserved"),
           observation(5, ext="jpeg")]
    serve_inat(site, monkeypatch, obs)
    assert fetch_piles.fetch_inat_observer(fetcher, sources, labels_only=True) == 1
    rows = [json.loads(line) for line in (sources / "inat-observer/observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [1, 2, 3, 5], "the observation whose first photo is not openly licensed is left out"
    assert rows[1] == {"id": 2, "taxon_name": "Genus species2", "taxon_common_name": "Bird 2", "taxon_rank": "species", "observed_on": "2023-05-03",
                       "time_observed_at": "2023-05-03T10:00:00-04:00", "license": "cc-by", "photo_license": "cc-by", "photo_id": 5002, "photo_ext": "JPG"}
    assert not any(word in json.dumps(rows) for word in ("location", "geojson", "Ithaca", "someone", "Some One", "-76.5"))
    assert [p.split("id_above=")[1] for p in site.gets("/v1/observations")] == ["0", "2", "4"], "by the last id, three pages of two"
    assert waits == [1.0, 1.0], "a second between two requests of the API"
    assert "4 observations (1 left out" in capsys.readouterr().out


def test_the_first_photo_of_each_observation_is_fetched_from_the_open_bucket_and_a_second_run_asks_for_nothing(site, fetcher, sources, monkeypatch):
    obs = [observation(1), observation(2, ext="JPG"), observation(3, ext="jpeg")]
    serve_inat(site, monkeypatch, obs)
    assert fetch_piles.fetch_inat_observer(fetcher, sources) == 4
    assert sorted(tree_of(sources / "inat-observer")) == ["observations.jsonl", "photos/5001.jpg", "photos/5002.JPG", "photos/5003.jpeg"]
    assert (sources / "inat-observer/photos/5002.JPG").read_bytes() == b"photo 2" * 20 and not any("6001" in p for p in site.gets("/photos/"))
    assert sorted(site.gets("/photos/")) == ["/photos/5001/large.jpg", "/photos/5002/large.JPG", "/photos/5003/large.jpeg"]
    site.requests.clear()
    fetch_piles.fetch_inat_observer(fetcher, sources)
    assert not site.gets("/v1/") and [m for m, _, _ in site.requests] == ["HEAD"] * 3, "no page of the API again, and a photo's size asked, no photo"


def test_photos_are_asked_for_three_at_a_time_at_most():
    assert fetch_piles.INAT_WORKERS == 3
    assert fetch_piles.limited(fetch.Fetcher(workers=4), 3).workers == 3 and fetch_piles.limited(fetch.Fetcher(workers=1), 3).workers == 1
    assert fetch_piles.limited(fetch.Fetcher(workers=4), fetch_piles.AMI_SLOW_WORKERS).workers == 2


def test_an_observer_with_no_observation_is_an_error_and_writes_nothing(site, fetcher, sources, monkeypatch):
    serve_inat(site, monkeypatch, [])
    with pytest.raises(fetch.FetchError, match="the API has no observation of user 5669063"):
        fetch_piles.fetch_inat_observer(fetcher, sources)
    assert not (sources / "inat-observer/observations.jsonl").exists() and not list(sources.rglob("*.part"))


# -- the command ------------------------------------------------------------------------------------------------------------------------------

def test_every_scenario_of_the_piles_names_sources_that_have_a_fetcher_and_a_budget_where_it_is_scripted():
    assert set(fetch.PILES).isdisjoint(fetch.REAL) and set(fetch.SCENARIOS) == {*fetch.REAL, *fetch.PILES}
    for scenario, ids in fetch.PILES.items():
        assert all(source in fetch_piles.SOURCES for source in ids), scenario
    named = {source for ids in fetch.PILES.values() for source in ids}
    assert named == set(fetch_piles.SOURCES), "every source has a scenario and every scenario a source"
    assert fetch.PILES["v9_lectures"] == ("ocw-14-01",) and "ocw-18642" not in named | set(fetch_piles.ARCHIVE_ITEMS) | set(fetch_piles.COUNTS), (
        "MIT 18.642 was considered and dropped: archive.org answered HTTP 460 on every one of its videos")
    scripted = {source for source, function in fetch_piles.SOURCES.items() if function.__name__ != "fetch_nothing"}
    assert scripted <= set(fetch_piles.BUDGET) and set(fetch_piles.COUNTS) <= scripted
    assert {"v9_lectures", "a6_lecture_audio", "a7_podcast", "v4_meetings", "a3_meetings", "v3_match", "v2_dashcam", "v10_home_clips", "i7_receipts",
            "i8_course_pages", "i9_nature", "a8_music", "i1_camera_roll", "i2_screenshots", "i4_site_photos", "m3_voicemail", "a4_field"} == set(fetch.PILES)


def test_the_command_takes_the_scenarios_of_the_piles_and_fetches_the_sources_each_needs(site, sources, monkeypatch, capsys):
    serve_archive(site, monkeypatch)
    for name in ("Charades.zip", "Charades_v1_480.zip"):
        site.files["/charades/" + name] = name.encode() * 3
    monkeypatch.setattr(fetch_piles, "CHARADES", site.url + "charades/")
    assert cli.main(["fetch", "--scenario", "v9_lectures", "--scenario", "v10_home_clips", "--sources", str(sources), "--workers", "2"]) == 0
    assert sorted(p.name for p in sources.iterdir()) == ["charades", "ocw-14-01"]
    out = capsys.readouterr().out
    assert "v9_lectures: the 6 files its build needs are in" in out and "v10_home_clips: the 2 files its build needs are in" in out


def test_all_is_every_scenario_the_four_and_the_piles(site, sources, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(fetch, "FETCHERS", {name: (lambda f, root, name=name: called.append(name) or 1) for name in fetch.REAL})
    monkeypatch.setattr(fetch_piles, "SOURCES", {source: (lambda f, root, source=source: called.append(source) or 2)
                                                   for ids in fetch.PILES.values() for source in ids})
    fetch.fetch(["all"], sources, fetch.Fetcher())
    assert called[:4] == list(fetch.REAL) and called[4:] == [source for ids in fetch.PILES.values() for source in ids]
    assert "i7_receipts: the 4 files its build needs are in" in capsys.readouterr().out, "sroie's and cord-v2's"


def test_a_source_that_is_not_scripted_says_so_and_names_no_file_it_could_not_fetch(sources):
    with pytest.raises(fetch.FetchError, match=r"android-in-the-wild: not scripted yet .*came from an earlier survey"):
        fetch_piles.SOURCES["android-in-the-wild"](fetch.Fetcher(), sources)
    assert not sources.exists()
