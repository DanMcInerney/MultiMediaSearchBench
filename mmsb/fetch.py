"""`mmsb fetch`: the files the four real scenarios read, from where each source is published, into the folders the builders read
(`realdump.sources_root()`, one folder per source). No account is needed for any of it.

    mmsb fetch --scenario v1_security      MEVA: 42 clips of three cameras (5.8 GB) and their labels, one file at a time
    mmsb fetch --scenario i3_trailcam      Caltech Camera Traps: the labels and the 4,256 photos of one camera's card (1.9 GB)
    mmsb fetch --scenario a1_home_audio    SINS: node 7's minute files of one day (4.9 GB read from Zenodo, 3.3 GB written as FLAC)
    mmsb fetch --scenario m4_samples       FSD50K: the labels and the 10,231 clips of the evaluation split (8.8 GB)

Each source is a function that lists what its builder reads (from the source's own listing or labels), checks the count against what
the builder needs (`EXPECTED`: another count would be another dump, so it is refused before anything large is fetched, and again
when the files are there, by the builder's own reader where that can be run), and fetches it. Nothing is written but a file that is whole: a download goes to `<name>.part`, resumes
from it, and is renamed when its length is the server's. A file that is there already is left alone when its size is right (the
listing's, else the server's); a wrong one is fetched again. So the same command continues a fetch that stopped.

The manner is polite: a few requests at a time, a name that says who asks, and a wait before each retry (the server's Retry-After
when it sends one, as a 429 does; else 1, 2, 4 ... seconds up to a minute). A request is tried `RETRIES` times.

SINS is published in large Zenodo zips of WAV files, and its licence (academic research only, no passing on) forbids mirroring it, so
the minute files are read out of those zips by HTTP Range, each checked by its size and CRC-32, and kept as lossless FLAC with all
four channels, which is what A1's builder reads."""

from __future__ import annotations

import csv
import http.client
import json
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import av
import numpy as np

from mmsb.scenarios import a1_home_audio, i3_trailcam, m4_samples, v1_security
from mmsb.scenarios.a1_home_audio import source as sins
from mmsb.scenarios.i3_trailcam import source as cct
from mmsb.scenarios.m4_samples import source as fsd
from mmsb.scenarios.realdump import SOURCES_ENV, source_word, sources_root
from mmsb.scenarios.v1_security import source as meva

SCENARIOS = (v1_security.name, i3_trailcam.name, a1_home_audio.name, m4_samples.name)
# What each build needs, counted in the files its source gives: the clips, the photos, the minute files, the clips. Found in the
# source's listing, a different number is another dump.
EXPECTED = {v1_security.name: 42, i3_trailcam.name: 4256, a1_home_audio.name: 1441, m4_samples.name: 10231}

MEVA_BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"  # the clips: drops-123-r13/<day>/<hour>/<clip>.avi
MEVA_CLIPS = "drops-123-r13"
# The labels come from commits, not branches: the answers must not change under a published baseline. Each is the revision whose files
# are the ones the baseline ran on (their sha256 were compared on 8 November 2026): the labels repositories' `master` then, and the
# Hub's revision of the third-party FSD50K mirror. The clips of MEVA, LILA and Zenodo are in buckets that do not change.
MEVA_REVISION = "421841a75577b697c314e952e585aecbb1b99e17"
SINS_REVISION = "8ec915ddc49d7ff33d39115f7b72d2e2429e47fe"
FSD50K_REVISION = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
MEVA_LABELS = f"https://gitlab.kitware.com/meva/meva-data-repo/-/raw/{MEVA_REVISION}/annotation/DIVA-phase-2/MEVA/"  # (the raw files)
LILA = "https://storage.googleapis.com/public-datasets-lila/"
CCT_LABELS = LILA + "caltechcameratraps/labels/caltech_camera_traps.json.zip"
CCT_IMAGES = LILA + "caltech-unzipped/cct_images/"
FSD50K = f"https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/{FSD50K_REVISION}/"
SINS_LABELS = f"https://raw.githubusercontent.com/KULeuvenADVISE/SINS_database/{SINS_REVISION}/annotation/"
SINS_ZIPS = "https://zenodo.org/api/records/2547315/files/"  # <zip>/content
SINS_ZIP_NAMES = tuple(f"Node7_audio_{n:02d}.zip" for n in range(1, 11))
SINS_LICENCE = SINS_ZIPS + "license.pdf/content"
SINS_NOTICE = ("SINS is licensed for academic research only and may not be passed on to a third party (its license.pdf: "
               f"{SINS_LICENCE}). This downloads it from Zenodo for you, to use under that licence; do not copy it anywhere others can "
               "fetch it from.")

USER_AGENT = "mmsb-fetch (MultiMediaSearchBench)"
WORKERS = 4  # requests at once
RETRIES = 6  # tries of one request
WAIT_MAX = 60.0  # the longest wait of those that double
WAIT_LONGEST = 3600.0  # a server that asks for more than this is not waited for
CHUNK = 1 << 20
WINDOW = 1 << 16  # what a zip's directory is read in at a time
SLACK = 1024  # bytes read past a member's data: a local header's name and extra field may be longer than the directory says
TRANSIENT = {429, 500, 502, 503, 504}
CHANNELS = 4  # of a SINS minute file


class FetchError(RuntimeError):
    """A fetch that cannot go on: a count that is not the build's, a file that did not arrive whole, a server that asks too long."""


class NotFound(FileNotFoundError):
    """The server has no file at the address."""


@dataclass(frozen=True)
class Job:
    url: str
    rel: str  # where it goes, in its source's folder
    size: int | None = None  # its size, when a listing gives it
    optional: bool = False  # a 404 is no error: the file is in one place or another


class Fetcher:
    """The HTTP of a fetch: requests that wait and try again, whole files that resume, and the work shared among a few threads. `sleep`
    is what waits (a test gives a stand-in that only notes the time asked)."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep, workers: int = WORKERS, tries: int = RETRIES) -> None:
        self.sleep, self.workers, self.tries = sleep, workers, tries

    @staticmethod
    def wait(retry_after: str | None, attempt: int, limited: bool = False) -> float:
        """Seconds to wait after try number `attempt`: what the server's Retry-After says (a number of seconds or a date), else 1, 2, 4
        ... up to WAIT_MAX; but WAIT_MAX each time when the server `limited` the rate (429) and did not say when to come back, as Hugging
        Face does (3,000 requests in 5 minutes): five of them are the window."""
        seconds = None
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    seconds = (when - datetime.now(when.tzinfo)).total_seconds()
                except (TypeError, ValueError):
                    pass
        if seconds is None:
            return WAIT_MAX if limited else min(WAIT_MAX, 2.0 ** (attempt - 1))
        if seconds > WAIT_LONGEST:
            raise FetchError(f"the server asks to wait {seconds:.0f} s: run the same command again later")
        return max(seconds, 0.0)

    def retrying(self, url: str, attempt: Callable[[], object]):
        """`attempt()`, tried again after a wait when the server is busy (429, 5xx) or the connection fails or drops. A 404 is NotFound."""
        for n in range(1, self.tries + 1):
            try:
                return attempt()
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise NotFound(f"{url} is not there") from None
                if error.code not in TRANSIENT or n == self.tries:
                    raise FetchError(f"{url}: HTTP {error.code} {error.reason}" + (f" on each of {n} tries" if n > 1 else "")) from None
                self.sleep(self.wait(error.headers.get("Retry-After"), n, error.code == 429))
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as error:
                if n == self.tries:
                    raise FetchError(f"{url}: {error!r} on each of {n} tries") from None
                self.sleep(self.wait(None, n))

    def open(self, url: str, headers: dict[str, str] | None = None, method: str = "GET"):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})}, method=method)
        return urllib.request.urlopen(request, timeout=60)

    def read(self, url: str, start: int | None = None, end: int | None = None) -> bytes:
        """The body at `url`, or bytes `start` to `end` of it (inclusive; the server cuts `end` at the file's end)."""
        headers = {} if start is None else {"Range": f"bytes={start}-{end}"}

        def attempt() -> bytes:
            with self.open(url, headers) as response:
                return response.read()

        return self.retrying(url, attempt)

    def size_of(self, url: str) -> int:
        def attempt() -> int:
            with self.open(url, method="HEAD") as response:
                return int(response.headers["Content-Length"])

        return self.retrying(url, attempt)

    def download(self, url: str, dest: Path, size: int | None = None) -> str:
        """`dest` from `url`: "kept" when a file of the right size (`size`, else the server's) is there; else "fetched", into
        `<dest>.part` from where an earlier try or run stopped, and renamed into place when its length is the server's. A file of
        the wrong size is fetched again."""
        if dest.is_file():
            if dest.stat().st_size == (self.size_of(url) if size is None else size):
                return "kept"
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")

        def attempt() -> None:
            have = part.stat().st_size if part.exists() else 0
            try:
                response = self.open(url, {"Range": f"bytes={have}-"} if have else None)
            except urllib.error.HTTPError as error:
                if error.code == 416:  # the part is longer than the file, or all of it: either way not to build on
                    part.unlink()
                    raise ConnectionError(f"{part} is not a start of {url}") from None
                raise
            with response:
                resumed = response.status == 206
                length = response.headers.get("Content-Range" if resumed else "Content-Length")  # (the whole file's, or the rest's)
                with open(part, "ab" if resumed else "wb") as out:
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
            if length is not None and part.stat().st_size != int(length.rpartition("/")[2]):
                raise http.client.IncompleteRead(b"")

        self.retrying(url, attempt)
        if size is not None and part.stat().st_size != size:
            got = part.stat().st_size
            part.unlink()
            raise FetchError(f"{url} came as {got} bytes, not the {size} its listing says")
        part.replace(dest)
        return "fetched"

    def fetch_all(self, label: str, folder: Path, jobs: Sequence[Job]) -> Counter:
        """Every job into `folder`, `workers` at once, one line of progress at about each twentieth; the first that fails stops the rest
        and is raised. What each did: kept, fetched, or absent (an optional file the server has not got)."""
        def one(job: Job) -> str:
            try:
                return self.download(job.url, folder / job.rel, job.size)
            except NotFound:
                if job.optional:
                    return "absent"
                raise

        return self.each(label, jobs, one)

    def each(self, label: str, items: Sequence, work: Callable) -> Counter:
        """`work(item)` for each of `items` in the threads, the strings they return counted; the first error stops the rest and is raised."""
        tally, step = Counter(), max(1, len(items) // 20)
        pool = ThreadPoolExecutor(self.workers)
        try:
            futures = [pool.submit(work, item) for item in items]
            for done, future in enumerate(as_completed(futures), 1):
                tally[future.result()] += 1
                if done % step == 0 or done == len(items):
                    print(f"  {label}: {done}/{len(items)}", flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return tally


def plain(name: str) -> str:
    """`name` when it is one file name, as a listing or a label file gives it; else FetchError: a name that climbs out of the folder
    (`..\\..\\x`) or names another one (`a/b`, `C:x`) would write where the fetch was not asked to."""
    if not name or name in (".", "..") or PurePosixPath(name).name != name or PureWindowsPath(name).name != name:
        raise FetchError(f"{name!r} is not a plain file name: a listing must not name a place outside the sources folder")
    return name


def check(what: str, found: int, scenario: str) -> None:
    if found != EXPECTED[scenario]:
        raise FetchError(f"{scenario}: {found} {what}, not the {EXPECTED[scenario]} its build needs (the source has changed, or a fetch is "
                         "short): its questions are drawn from that many")


def listing(fetcher: Fetcher, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """The (key, size) of every object whose key starts with `prefix` in an S3 bucket that lists without an account."""
    found, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
        root = ET.fromstring(fetcher.read(f"{bucket}?{urllib.parse.urlencode(query)}"))
        found += [(c.findtext("{*}Key"), int(c.findtext("{*}Size"))) for c in root.iterfind("{*}Contents")]
        if root.findtext("{*}IsTruncated") != "true":
            return found
        token = root.findtext("{*}NextContinuationToken")


# -- MEVA ----------------------------------------------------------------------------------------------------------------------------

def fetch_meva(fetcher: Fetcher, root: Path) -> int:
    """The clips V1 reads, from the bucket's listing, and each one's label file (`<set>/<day>/<hour>/<clip>.activities.yml`, in one
    of the two sets, so a 404 in the other is no error) and the label names, from the labels repository's raw files. The clips'
    count is checked before any is fetched, and the label files' after: one for each clip. Returns the clips."""
    folder = root / meva.SOURCE.name
    clips = [(key, size) for hour in range(meva.FIRST.hour, meva.LAST.hour + 1)
             for key, size in listing(fetcher, MEVA_BUCKET, f"{MEVA_CLIPS}/{meva.DAY}/{hour:02d}/") if meva.wanted(key.rpartition("/")[2])]
    parts = [key.split("/") for key, _ in clips]
    if bad := [k for k, p in zip(clips, parts, strict=True) if len(p) != 4 or p[:2] != [MEVA_CLIPS, meva.DAY]]:
        raise FetchError(f"{v1_security.name}: the key {bad[0][0]!r} is not {MEVA_CLIPS}/{meva.DAY}/<hour>/<clip>")
    check("clips", len(clips), v1_security.name)
    named = [(plain(p[2]), plain(p[3]).removesuffix(".r13.avi")) for p in parts]  # (hour, clip)
    labels = [Job(f"{MEVA_LABELS}{s}/{meva.DAY}/{hour}/{name}.activities.yml", f"{meva.LABELS}/{s}/{meva.DAY}/{hour}/{name}.activities.yml",
                  optional=True) for hour, name in named for s in meva.SETS]
    print(f"{v1_security.name}: {len(clips)} clips and their labels from MEVA")
    fetcher.fetch_all("clips", folder, [Job(MEVA_BUCKET + key, f"{meva.VIDEO}/{hour}/{name}.r13.avi", size)
                                        for (key, size), (hour, name) in zip(clips, named, strict=True)])
    fetcher.fetch_all("label files", folder, [*labels, Job(f"{MEVA_LABELS}{meva.SETS[0]}/status.csv", f"{meva.LABELS}/{meva.SETS[0]}/status.csv")])
    if (have := sum((folder / j.rel).is_file() for j in labels)) != len(clips):
        raise FetchError(f"{v1_security.name}: {have} label files for {len(clips)} clips: each is in exactly one of the sets {', '.join(meva.SETS)}")
    return len(clips)


# -- Caltech Camera Traps ------------------------------------------------------------------------------------------------------------

def fetch_cct(fetcher: Fetcher, root: Path) -> int:
    """The labels, then the photos of the card that they name (location 23, taken in CARD_DAYS: the rest of the location is not read);
    then I3's own reader counts the card. Returns the photos."""
    folder = root / cct.SOURCE.name
    print(f"{i3_trailcam.name}: the labels and the photos of the card (location {cct.LOCATION}) from Caltech Camera Traps (LILA BC)")
    fetcher.fetch_all("labels", folder, [Job(CCT_LABELS, cct.LABELS)])
    with zipfile.ZipFile(folder / cct.LABELS) as z:
        names = [plain(i["file_name"]) for i in cct.card_images(json.loads(z.read(cct.MEMBER)))]
    check("photos", len(names), i3_trailcam.name)
    fetcher.fetch_all("photos", folder, [Job(CCT_IMAGES + name, f"{cct.IMAGES}/{name}") for name in names])
    photos = cct.read_labels(str(folder))[0]
    cct.check_photos(str(folder), photos)
    check("photos in the folder", len(photos), i3_trailcam.name)
    return len(photos)


# -- FSD50K --------------------------------------------------------------------------------------------------------------------------

def fetch_fsd50k(fetcher: Fetcher, root: Path) -> int:
    """The labels, the raters' judgements and the vocabulary, then the evaluation clips the labels name; then M4's own reader counts
    the clips. Returns the clips."""
    folder = root / fsd.SOURCE.name
    print(f"{m4_samples.name}: the labels and the evaluation clips from FSD50K (Hugging Face mirror)")
    fetcher.fetch_all("labels", folder, [Job(FSD50K + rel, rel) for rel in (fsd.LABELS, fsd.VOCABULARY, fsd.RATINGS)])
    with open(folder / fsd.LABELS, newline="", encoding="utf-8") as f:
        ids = [plain(row["fname"]) for row in csv.DictReader(f)]
    check("clips", len(ids), m4_samples.name)
    fetcher.fetch_all("clips", folder, [Job(f"{FSD50K}{fsd.CLIPS}/{i}.wav", f"{fsd.CLIPS}/{i}.wav") for i in ids])
    clips = fsd.read_labels(str(folder))[0]
    check("clips in the folder", len(clips), m4_samples.name)
    return len(clips)


# -- SINS ----------------------------------------------------------------------------------------------------------------------------

class RemoteFile:
    """A file on a server as `zipfile` reads it: each read is a Range request for a window of at least WINDOW bytes, which the next
    reads are served from."""

    def __init__(self, fetcher: Fetcher, url: str) -> None:
        self.fetcher, self.url, self.size, self.at, self.window = fetcher, url, fetcher.size_of(url), 0, (0, b"")

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.at

    def seek(self, offset: int, whence: int = 0) -> int:
        self.at = (offset, self.at + offset, self.size + offset)[whence]
        return self.at

    def read(self, n: int = -1) -> bytes:
        end = self.size if n < 0 else min(self.size, self.at + n)
        if end <= self.at:
            return b""
        first, data = self.window
        if not first <= self.at or end > first + len(data):
            first = self.at
            data = self.fetcher.read(self.url, first, min(self.size, max(end, first + WINDOW)) - 1)
            self.window = (first, data)
        out, self.at = data[self.at - first:end - first], end
        return out


def minute_name(info: zipfile.ZipInfo) -> tuple[Path, datetime] | None:
    """Where the minute file of a zip member goes in node 7's folder (`<day>/Node7_..._audio.flac`) and when it begins, else None for
    a member that is no minute file of the node."""
    flac = info.filename.rpartition("/")[2].removesuffix(".wav") + ".flac"
    m = sins.STAMP.fullmatch(flac)
    if m is None:
        return None
    start = datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S") + timedelta(milliseconds=int(m[3]))
    return Path(sins.NODE) / start.date().isoformat() / flac, start


def member_samples(fetcher: Fetcher, url: str, info: zipfile.ZipInfo) -> np.ndarray:
    """The samples (frames by channels, 16-bit) of the WAV member `info` of the zip at `url`, read by Range and checked by its size and
    CRC-32."""
    start = info.header_offset
    data = fetcher.read(url, start, start + 30 + len(info.filename.encode()) + SLACK + info.compress_size)
    name_len, extra_len = struct.unpack("<2H", data[26:30])
    body = data[30 + name_len + extra_len:30 + name_len + extra_len + info.compress_size]
    if data[:4] != b"PK\x03\x04" or len(body) != info.compress_size or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise FetchError(f"{info.filename} of {url} is not where its zip's directory says")
    raw = zlib.decompress(body, -15) if info.compress_type == zipfile.ZIP_DEFLATED else body
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise FetchError(f"{info.filename} of {url} did not arrive whole: its size or CRC-32 is not the zip's")
    at, fmt, samples = 12, None, None
    while at + 8 <= len(raw):
        kind, size = raw[at:at + 4], int.from_bytes(raw[at + 4:at + 8], "little")
        if kind == b"fmt ":
            fmt = struct.unpack("<HHIIHH", raw[at + 8:at + 24])
        elif kind == b"data":
            samples = raw[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    if raw[:4] != b"RIFF" or fmt is None or samples is None or (fmt[0] not in (1, 0xFFFE) or fmt[1:3] != (CHANNELS, sins.RATE) or fmt[5] != 16
                                                               or len(samples) % (2 * CHANNELS)):
        raise FetchError(f"{info.filename} of {url} is not a 16-bit, {sins.RATE} Hz, {CHANNELS}-channel WAV")
    return np.frombuffer(samples, "<i2").reshape(-1, CHANNELS)


def decoded(path: Path) -> np.ndarray:
    """The samples (frames by channels) of a 16-bit FLAC."""
    with av.open(str(path)) as container:
        return np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=1).reshape(-1, CHANNELS)


def write_flac(samples: np.ndarray, path: Path) -> None:
    """`samples` as a lossless FLAC at `path`, with all four channels: written beside it, decoded again and compared, then renamed."""
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(part), "w", format="flac") as container:
        stream = container.add_stream("flac", rate=sins.RATE, layout="quad")
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples).reshape(1, -1), format="s16", layout="quad")
        frame.sample_rate = sins.RATE
        for packet in (*stream.encode(frame), *stream.encode(None)):
            container.mux(packet)
    if not np.array_equal(decoded(part), samples):
        part.unlink()
        raise FetchError(f"{path.name} did not decode to the samples it was written from")
    part.replace(path)


def fetch_sins(fetcher: Fetcher, root: Path) -> int:
    """The labels A1 reads, and node 7's minute files that play in the day A1 uses: found by reading each Zenodo zip's directory, then
    read out of the zip by Range, written as FLAC (a file that is there is whole: it is renamed into place when it is); then A1's own
    reader counts the minute files that play in the day. Returns them."""
    folder = root / sins.SOURCE.name
    print(SINS_NOTICE)
    first = datetime.fromisoformat(a1_home_audio.DAY) + timedelta(seconds=a1_home_audio.SCALES["standard"][0])
    last = first + timedelta(seconds=a1_home_audio.SCALES["standard"][1])
    fetcher.fetch_all("labels", folder, [Job(f"{SINS_LABELS}{Path(rel).name}", rel) for rel in (sins.LIVING, sins.HOME)])
    members = []
    for name in SINS_ZIP_NAMES:
        url = f"{SINS_ZIPS}{name}/content"
        with zipfile.ZipFile(RemoteFile(fetcher, url)) as z:
            members += [(url, info, *found) for info in z.infolist() if (found := minute_name(info)) and first - sins.LONGEST < found[1] < last]
    check("minute files", len(members), a1_home_audio.name)
    print(f"{a1_home_audio.name}: {len(members)} minute files of {sins.NODE}, {sum(i.compress_size for _, i, *_ in members) / 1e9:.1f} GB "
          f"read from {len(SINS_ZIP_NAMES)} zips on Zenodo")

    def one(member: tuple) -> str:
        url, info, rel, _ = member
        if (folder / rel).is_file():
            return "kept"
        write_flac(member_samples(fetcher, url, info), folder / rel)
        return "fetched"

    fetcher.each("minute files", members, one)
    here = sins.minute_files(str(folder), first, last)
    check("minute files in the folder", len(here), a1_home_audio.name)
    return len(here)


FETCHERS = {v1_security.name: fetch_meva, i3_trailcam.name: fetch_cct, a1_home_audio.name: fetch_sins, m4_samples.name: fetch_fsd50k}


def fetch(scenarios: Sequence[str], sources: Path, fetcher: Fetcher | None = None) -> None:
    """The sources of `scenarios` ("all" is every one) into `sources`, each into the folder its builder reads; says what it did, and
    raises when what a source gives is not what its build needs."""
    wanted = SCENARIOS if "all" in scenarios else tuple(dict.fromkeys(scenarios))
    source_word(sources)  # a folder an agent could not be kept out of is refused now, not after the downloads
    fetcher = fetcher or Fetcher()
    for scenario in wanted:
        found = FETCHERS[scenario](fetcher, Path(sources))
        print(f"{scenario}: the {found} files its build needs are in {Path(sources)}")
    if Path(sources).resolve() != sources_root().resolve():
        print(f"note: a build reads {SOURCES_ENV}, which is {sources_root()}; set it to {Path(sources)} to build from what was fetched")
