"""`mmsb fetch` for the piles of media of the scenarios added after the first four (`fetch.PILES` says which scenario reads which source):
lectures, a podcast, meetings, a match, a dashcam, home clips, receipts, course pages, nature photos, music, and the sources an earlier
survey had put on disk (a camera roll, screenshots, site photos, voicemails, a field recording). One folder per source in the sources
folder, named by the source's id in media/sources.jsonl; MEDIA.md says where each comes from and under what licence.

Each source is a function `fetch_<source>(fetcher, root, labels_only=False)` that lists what its scenario reads from the place that
publishes it (an archive.org item's metadata, a Hugging Face repository's tree at a fixed commit, a Zenodo record, a course's download
page, the iNaturalist API, or a fixed list of names on a server), checks the count against what was found when the source was chosen
(`COUNTS`: another count is another pile, so it is refused before anything large is fetched) and the bytes against `BUDGET`, and fetches
it with `fetch.Fetcher`: the same politeness, resumes and `.part` files as the first four. A file that is there already with the size the
listing (else the server) gives is not fetched again, so the same command continues a fetch that stopped. `labels_only` brings the
labels and the small files first (the transcripts, the label files, the receipts and the dashcam, the annotation zips), the media after.

What the fetch brings is what the scenario reads, as the publisher keeps it: a zip stays a zip and a parquet stays a parquet. Only the
names a scenario sees are the publisher's, so a scenario that must hide them (a lecture's number, a guest's name) renames in its dump."""

from __future__ import annotations

import json
import re
import urllib.parse
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mmsb.fetch import FetchError, Fetcher, Job, NotFound, plain

ARCHIVE = "https://archive.org/"
HUB = "https://huggingface.co/"
ZENODO = "https://zenodo.org/api/records/"
AMI = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"  # <meeting>/<audio|video>/<meeting>.<signal>
AMI_ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
ICSI_SIGNALS = "https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/"
ICSI_ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/"
CHARADES = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/"
FMA = "https://os.unil.cloud.switch.ch/fma/"
OCW = "https://ocw.mit.edu/courses/"
INAT_API = "https://api.inaturalist.org/v1/observations"
INAT_PHOTOS = "https://inaturalist-open-data.s3.amazonaws.com/photos/"

# The Hugging Face repositories are read at the commit they were when each source was chosen (9 November 2026), as the first four's
# labels are: the listing, the labels and the clips are then the ones the questions were drawn from.
ST2_REPO, ST2_REVISION = "algo-dev99/soccertrack-v2-followcam-events", "5ca71d867ee073335abb24b771e0604ff22d1b88"
DASHCAM_REPO, DASHCAM_REVISION = "qutegocentric/Australian_Roads_Dashcam_Driving", "79300a79a4efac07f24b12eb3890a2977cba9e88"
SROIE_REPO, SROIE_REVISION = "jsdnrs/ICDAR2019-SROIE", "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
CORD_REPO, CORD_REVISION = "naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98"
ATM_REPO, ATM_REVISION = "Jingbiao/ATM-Bench", "78e826dc07e97466b2f54443831ef9a83ab8b27c"

GB = 10 ** 9
INAT_USER, INAT_FROM, INAT_TO = 5669063, "2022-01-01", "2025-12-31"
INAT_LICENCES = ("cc0", "cc-by", "cc-by-nc")  # the photos the open-data bucket holds
INAT_PAGE = 200
INAT_WAIT = 1.0  # seconds between two requests of the API
INAT_WORKERS = 3  # photos at once
AMI_SERIES = ("ES2002", "ES2015", "ES2016", "IS1006", "TS3005")  # a to d of each (ES2002 and ES2016 were on disk from the first survey)
AMI_MEETINGS = tuple(f"{series}{part}" for series in AMI_SERIES for part in "abcd")
AMI_ANNOTATION_FILES = ("DOME_documentation.pdf", "SocialRoleAnnotation.tar.gz", "ami_public_auto_1.5.1.zip", "ami_public_manual_1.6.2.zip",
                        "dome_annotations_M1.csv", "dome_dataset_M1.csv")
AMI_SLOW_WORKERS = 2  # the mirror is slow (0.06 to 0.5 MB/s): two files at once
# The first nine Bmr meetings of the mirror's list in order, and Bmr019 (the two on disk are Bmr005 and Bmr019): ten of the real
# research-group meetings of ICSI.
ICSI_MEETINGS = ("Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007", "Bmr008", "Bmr009", "Bmr010", "Bmr019")
ICSI_ANNOTATION_FILES = ("ICSI_original_transcripts.zip", "ICSI_plus_NXT.zip")
ST2_MATCHES = 3  # whole matches, the first by sorted match id (8.3 GB; the first five are 14.7 GB)
OCW_COURSES = ("6-006-introduction-to-algorithms-spring-2020", "9-00sc-introduction-to-psychology-fall-2011",
               "7-016-introductory-biology-fall-2018", "5-111sc-principles-of-chemical-science-fall-2014")
ROBOCALL_ZIP = "https://codeload.github.com/wspr-ncsu/robocall-audio-dataset/zip/refs/heads/main"  # a zip made on the fly: no length, no Range
ROBOCALL_ROOT = "robocall-audio-dataset-main/"
ROBOCALL_FILES = ("LICENSE.md", "README.md", "metadata.csv")
ROBOCALL_DIRS = ("audio-wav-16khz/", "pdf_files/")
BIRDVOX_RECORD = "1205569"
BIRDVOX_FILES = ("BirdVox-full-night_csv-annotations_unit01.csv", "BirdVox-full-night_csv-annotations_unit02.csv",
                 "BirdVox-full-night_csv-annotations_unit03.csv", "BirdVox-full-night_csv-annotations_unit05.csv",
                 "BirdVox-full-night_csv-annotations_unit07.csv", "BirdVox-full-night_csv-annotations_unit10.csv",
                 "BirdVox-full-night_flac-audio_unit03.flac", "BirdVox-full-night_gps-coordinates.csv",
                 "BirdVox-full-night_utc-start-times.csv", "README.md")

# What each listing held when the source was chosen, in the files the source gives (another count is another pile).
COUNTS = {
    "ocw-14-01": 26,  # lecture videos (and as many SRT files)
    "ocw-6-0002": 15,
    "chalk-radio": 60,
    "st2-followcam": 525,  # clips of the first three matches (174, 172 and 179), and as many label files
    "aus-dashcam": 128,  # clips
    "sroie": 2,  # parquet files
    "cord-v2": 6,
    "atm-bench": 3759,  # photos
    "robocall-ftc": 1427,  # recordings (WAV)
}
# The most a source may bring that is not on disk, in bytes: a little over the size it was chosen at. A listing that comes to more is
# not fetched ("the source has changed").
BUDGET = {"ocw-14-01": 3.1 * GB, "ocw-6-0002": 2.0 * GB, "chalk-radio": 3.6 * GB, "ami": 2.0 * GB, "icsi": 1.6 * GB, "st2-followcam": 9.0 * GB,
          "aus-dashcam": 0.7 * GB, "charades": 17.0 * GB, "sroie": 0.6 * GB, "cord-v2": 2.5 * GB, "ocw-pages": 0.7 * GB,
          "inat-observer": 1.7 * GB, "fma": 8.5 * GB, "atm-bench": 1.4 * GB, "birdvox-full-night": 0.95 * GB, "robocall-ftc": 1.4 * GB}


def safe_rel(path: str) -> str:
    """`path` of a listing (`Crash/20260106_x.mp4`) when each part of it is one plain file name, else FetchError."""
    return "/".join(plain(part) for part in path.split("/"))


def expect(source: str, what: str, found: int, wanted: int) -> None:
    if found != wanted:
        raise FetchError(f"{source}: {found} {what}, not the {wanted} found when it was chosen (the source has changed, or a fetch is short)")


def read_json(fetcher: Fetcher, url: str) -> tuple[object, str]:
    """The JSON at `url` and the Link header that came with it (a Hugging Face tree gives the next page there)."""
    def attempt() -> tuple[object, str]:
        with fetcher.open(url) as response:
            return json.load(response), response.headers.get("Link") or ""

    return fetcher.retrying(url, attempt)


def read_text(fetcher: Fetcher, url: str) -> str:
    return fetcher.read(url).decode("utf-8", errors="replace")


def limited(fetcher: Fetcher, workers: int) -> Fetcher:
    """`fetcher`, asking for at most `workers` files at once (a slow or small server is asked for less than the command's `--workers`)."""
    return Fetcher(fetcher.sleep, min(fetcher.workers, workers), fetcher.tries)


def sized(fetcher: Fetcher, jobs: list[Job]) -> list[Job]:
    """`jobs` with the size the server gives each (a HEAD request), those whose file the server has not got dropped when they are optional."""
    def one(job: Job) -> Job | None:
        try:
            return Job(job.url, job.rel, job.size if job.size is not None else fetcher.size_of(job.url), job.optional)
        except NotFound:
            if job.optional:
                return None
            raise

    with ThreadPoolExecutor(fetcher.workers) as pool:
        return [job for job in pool.map(one, jobs) if job is not None]


def within_budget(source: str, folder: Path, jobs: list[Job]) -> None:
    """FetchError when the jobs' files that are not there already come to more than the source's BUDGET. Sizes are the listing's."""
    new = sum(job.size or 0 for job in jobs if not (folder / job.rel).is_file())
    if new > BUDGET[source]:
        raise FetchError(f"{source}: {new / GB:.1f} GB to fetch, over the {BUDGET[source] / GB:.1f} GB it was chosen at: the source has changed, "
                         "or the listing is wrong")


def bring(fetcher: Fetcher, source: str, folder: Path, jobs: list[Job], label: str = "files") -> int:
    """The jobs into `folder` (smallest first), after the budget is checked; says what it did. Returns the files that are there."""
    within_budget(source, folder, jobs)
    tally = fetcher.fetch_all(label, folder, sorted(jobs, key=lambda job: job.size or 0))
    print(f"{source}: {label}: " + ", ".join(f"{n} {what}" for what, n in sorted(tally.items())) if tally else f"{source}: no {label}")
    return sum((folder / job.rel).is_file() for job in jobs)


def count_files(folder: Path) -> int:
    return sum(p.is_file() and not p.name.endswith(".part") for p in folder.rglob("*")) if folder.is_dir() else 0


# -- the places that list their files -------------------------------------------------------------------------------------------------

def archive_files(fetcher: Fetcher, item: str) -> list[tuple[str, int]]:
    """The (name, size) of the video and subtitle files an archive.org item published (its originals: the derivatives, thumbnails and
    the metadata the site makes are not read), from its metadata API."""
    meta, _ = read_json(fetcher, f"{ARCHIVE}metadata/{urllib.parse.quote(item)}")
    return [(f["name"], int(f["size"])) for f in meta["files"] if f.get("source") == "original" and f.get("format") in ("MPEG4", "SubRip")]


def hub_files(fetcher: Fetcher, repo: str, revision: str) -> list[tuple[str, int]]:
    """The (path, size) of every file of a Hugging Face dataset repository at `revision`, page by page (the tree API gives 1,000 entries
    and a Link header for the next page)."""
    url, found = f"{HUB}api/datasets/{repo}/tree/{revision}?recursive=true", []
    while url:
        page, link = read_json(fetcher, url)
        found += [(entry["path"], int(entry["size"])) for entry in page if entry["type"] == "file"]
        url = (m[1] if (m := re.search(r'<([^>]+)>\s*;\s*rel="next"', link)) else None)
    return found


def hub_job(repo: str, revision: str, path: str, size: int) -> Job:
    return Job(f"{HUB}datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}", safe_rel(path), size)


def zenodo_files(fetcher: Fetcher, record: str) -> list[tuple[str, int]]:
    """The (name, size) of the files of a Zenodo record, from its API."""
    meta, _ = read_json(fetcher, f"{ZENODO}{record}")
    return [(f["key"], int(f["size"])) for f in meta["files"]]


def ocw_pdfs(fetcher: Fetcher, course: str) -> list[str]:
    """The names of the PDFs linked from the course's download page (`<course>/download/`): the exams, notes, assignments and
    solutions OCW publishes as files, once each, in the page's order."""
    page = read_text(fetcher, f"{OCW}{course}/download/")
    names = re.findall(rf'href="(?:https://ocw\.mit\.edu)?/courses/{re.escape(course)}/([^"/?#]+\.pdf)"', page, re.IGNORECASE)
    return [plain(urllib.parse.unquote(name)) for name in dict.fromkeys(names)]


# -- archive.org: MIT OpenCourseWare lectures and Chalk Radio -----------------------------------------------------------------------

ARCHIVE_ITEMS = {"ocw-14-01": "mit14_01s23", "ocw-6-0002": "MIT6.0002F16", "chalk-radio": "chalk-radio"}


def fetch_archive(fetcher: Fetcher, root: Path, source: str, labels_only: bool = False) -> int:
    """The item's videos and the SRT beside each (the captions are the truth). The count of videos, and of captions, is checked first."""
    item, folder = ARCHIVE_ITEMS[source], root / source
    files = archive_files(fetcher, item)
    videos, captions = ([f for f in files if f[0].endswith(ext)] for ext in (".mp4", ".srt"))
    expect(source, "videos", len(videos), COUNTS[source])
    expect(source, "caption files", len(captions), COUNTS[source])
    def jobs(found: list[tuple[str, int]]) -> list[Job]:
        return [Job(f"{ARCHIVE}download/{urllib.parse.quote(item)}/{urllib.parse.quote(name)}", safe_rel(name), size) for name, size in found]

    captions, videos = jobs(captions), jobs(videos)  # (every name is checked before a file is fetched)
    print(f"{source}: {len(videos)} videos and their captions from archive.org ({item})")
    bring(fetcher, source, folder, captions, "captions")
    if not labels_only:
        bring(fetcher, source, folder, videos, "videos")
    return count_files(folder)


def fetch_ocw_14_01(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_archive(fetcher, root, "ocw-14-01", labels_only)


def fetch_ocw_6_0002(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_archive(fetcher, root, "ocw-6-0002", labels_only)


def fetch_chalk_radio(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_archive(fetcher, root, "chalk-radio", labels_only)


# -- the AMI and ICSI meetings, from the Edinburgh mirror ----------------------------------------------------------------------------

def fetch_ami(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """The annotations (words with times, topics, decisions) and, for each of the meetings of AMI_MEETINGS, the headset mix
    (`signals/<id>/audio/<id>.Mix-Headset.wav`) and the corner camera (`signals/<id>/video/<id>.Corner.avi`). The mirror has no corner
    camera for the IS and TS meetings (it films them from other cameras): those meetings have audio alone, and the fetch says so."""
    folder, slow = root / "ami", limited(fetcher, AMI_SLOW_WORKERS)
    bring(fetcher, "ami", folder, [Job(AMI_ANNOTATIONS + name, f"annotations/{name}") for name in AMI_ANNOTATION_FILES], "annotations")
    if labels_only:
        return count_files(folder)
    jobs = [Job(f"{AMI}{m}/{kind}/{m}.{signal}", f"signals/{m}/{kind}/{m}.{signal}", optional=signal == "Corner.avi")
            for m in AMI_MEETINGS for kind, signal in (("audio", "Mix-Headset.wav"), ("video", "Corner.avi"))]
    jobs = sized(slow, jobs)
    print(f"ami: {len(AMI_MEETINGS)} meetings, {len(jobs)} files (the corner camera is missing for "
          f"{len(AMI_MEETINGS) * 2 - len(jobs)} of them) from the Edinburgh mirror")
    bring(slow, "ami", folder, jobs, "signals")
    return count_files(folder)


def fetch_icsi(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """The annotations (the original transcripts and the NXT version) and the mix of each of ICSI_MEETINGS (`signals/<id>.interaction.wav`)."""
    folder, slow = root / "icsi", limited(fetcher, AMI_SLOW_WORKERS)
    bring(fetcher, "icsi", folder, [Job(ICSI_ANNOTATIONS + name, f"annotations/{name}") for name in ICSI_ANNOTATION_FILES], "annotations")
    if not labels_only:
        bring(slow, "icsi", folder, sized(slow, [Job(f"{ICSI_SIGNALS}{m}.interaction.wav", f"signals/{m}.interaction.wav") for m in ICSI_MEETINGS]),
              "signals")
    return count_files(folder)


# -- Hugging Face repositories ----------------------------------------------------------------------------------------------------------

def st2_matches(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The clips and labels of the first ST2_MATCHES matches by sorted match id (`soccertrack/clips/st<match>_h<half>_c<n>.mp4`, its labels
    `soccertrack/labels/st<match>_h<half>_c<n>.Labels-ball.json`), whole matches: every clip of each; and the README."""
    clip = re.compile(r"soccertrack/clips/st(\d+)_h\d_c\d+\.mp4")
    matches = sorted({m[1] for path, _ in files if (m := clip.fullmatch(path))}, key=int)[:ST2_MATCHES]
    keep = tuple(f"st{m}_" for m in matches)
    chosen = [(path, size) for path, size in files if path == "README.md" or (path.startswith(("soccertrack/clips/", "soccertrack/labels/"))
                                                                              and path.rpartition("/")[2].startswith(keep))]
    clips, labels = ([f for f in chosen if f"/{kind}/" in f[0]] for kind in ("clips", "labels"))
    if len(clips) != len(labels) or {p.rpartition("/")[2].removesuffix(".mp4") for p, _ in clips} != \
            {p.rpartition("/")[2].removesuffix(".Labels-ball.json") for p, _ in labels}:
        raise FetchError(f"st2-followcam: the clips of matches {', '.join(matches)} and their label files are not the same set")
    return chosen


def everything(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Every file of the repository but its `.gitattributes`."""
    return [(path, size) for path, size in files if path != ".gitattributes"]


def atm_files(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """What ATM-Bench's camera roll is read from: the README, the questions (`data/atm-bench`), the photos and their geocoding
    (`data/raw_memory/image`, `geocoding_cache/image`) and the processed memory files; not its videos, their geocoding and its emails."""
    return [(path, size) for path, size in files if path == "README.md" or path.startswith(
        ("data/atm-bench/", "data/raw_memory/image/", "data/raw_memory/geocoding_cache/image/", "data/processed_memory/"))]


@dataclass(frozen=True)
class Hub:
    repo: str
    revision: str
    select: Callable[[list[tuple[str, int]]], list[tuple[str, int]]]  # the files of the repository the scenario reads
    pile: str  # the suffix of the files that make the pile (counted against COUNTS)
    held: str | None = None  # the suffix of the files `labels_only` holds back; None: every file is small, and comes at once


HUBS = {
    "st2-followcam": Hub(ST2_REPO, ST2_REVISION, st2_matches, ".mp4", ".mp4"),
    "aus-dashcam": Hub(DASHCAM_REPO, DASHCAM_REVISION, everything, ".mp4"),
    "sroie": Hub(SROIE_REPO, SROIE_REVISION, everything, ".parquet"),
    "cord-v2": Hub(CORD_REPO, CORD_REVISION, everything, ".parquet"),
    "atm-bench": Hub(ATM_REPO, ATM_REVISION, atm_files, ".jpg", ".jpg"),
}


def fetch_hub(fetcher: Fetcher, root: Path, source: str, labels_only: bool = False) -> int:
    """The files of the source's repository that its scenario reads, at the fixed commit. The count of the files that make the pile
    (the clips, the photos, the parquet files) is checked before any is fetched. `labels_only` holds back the files ending in the
    source's `held` suffix; a source with none (the receipts, the dashcam: 0.6 GB or less of it) comes whole at once."""
    hub = HUBS[source]
    files = hub.select(hub_files(fetcher, hub.repo, hub.revision))
    expect(source, "files that make the pile", sum(path.endswith(hub.pile) for path, _ in files), COUNTS[source])
    jobs = [hub_job(hub.repo, hub.revision, path, size) for path, size in files]
    print(f"{source}: {len(jobs)} files from {hub.repo} on Hugging Face ({hub.revision[:8]})")
    folder, large = root / source, [job for job in jobs if hub.held and job.rel.endswith(hub.held)]
    bring(fetcher, source, folder, [job for job in jobs if job not in large], "labels" if large else "files")
    if large and not labels_only:
        bring(fetcher, source, folder, large, "media")
    return count_files(folder)


def fetch_st2_followcam(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_hub(fetcher, root, "st2-followcam", labels_only)


def fetch_aus_dashcam(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_hub(fetcher, root, "aus-dashcam", labels_only)


def fetch_sroie(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_hub(fetcher, root, "sroie", labels_only)


def fetch_cord_v2(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_hub(fetcher, root, "cord-v2", labels_only)


def fetch_atm_bench(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    return fetch_hub(fetcher, root, "atm-bench", labels_only)


# -- whole files at fixed addresses ------------------------------------------------------------------------------------------------------

def fetch_zips(fetcher: Fetcher, root: Path, source: str, base: str, small: str, large: str, labels_only: bool) -> int:
    """Two zips kept as zips: the small one (the annotations) first, the large one (the videos or the clips) after. Their sizes are the
    server's."""
    folder = root / source
    bring(fetcher, source, folder, sized(fetcher, [Job(base + small, small)]), "annotations")
    if not labels_only:
        bring(fetcher, source, folder, sized(fetcher, [Job(base + large, large)]), "media")
    return count_files(folder)


def fetch_charades(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """`Charades.zip` (the labels, scripts and licence) and `Charades_v1_480.zip` (the 9,848 videos at 480p), whole: a scenario reads the
    videos out of the zip, so it is not unpacked here."""
    return fetch_zips(fetcher, root, "charades", CHARADES, "Charades.zip", "Charades_v1_480.zip", labels_only)


def fetch_fma(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """`fma_metadata.zip` (tracks.csv, genres.csv, the audio features) and `fma_small.zip` (8,000 clips of 30 s), whole."""
    return fetch_zips(fetcher, root, "fma", FMA, "fma_metadata.zip", "fma_small.zip", labels_only)


def fetch_birdvox_full_night(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """BirdVox-full-night's record on Zenodo: the annotations of six units, the audio of unit 3 (10.9 hours, FLAC), the GPS and the UTC
    start times, and the README; by name, from the record's own listing."""
    folder, found = root / "birdvox-full-night", dict(zenodo_files(fetcher, BIRDVOX_RECORD))
    if missing := [name for name in BIRDVOX_FILES if name not in found]:
        raise FetchError(f"birdvox-full-night: {', '.join(missing)} is not in Zenodo record {BIRDVOX_RECORD}")
    jobs = [Job(f"{ZENODO}{BIRDVOX_RECORD}/files/{urllib.parse.quote(name)}/content", safe_rel(name), found[name]) for name in BIRDVOX_FILES]
    bring(fetcher, "birdvox-full-night", folder, [j for j in jobs if not j.rel.endswith(".flac")], "labels")
    if not labels_only:
        bring(fetcher, "birdvox-full-night", folder, [j for j in jobs if j.rel.endswith(".flac")], "audio")
    return count_files(folder)


# -- OCW course pages --------------------------------------------------------------------------------------------------------------------

def fetch_ocw_pages(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """The PDFs linked from the download page of each course of OCW_COURSES, one folder for each (`<course>/<name>.pdf`)."""
    folder, jobs = root / "ocw-pages", []
    for course in OCW_COURSES:
        names = ocw_pdfs(fetcher, course)
        if not names:
            raise FetchError(f"ocw-pages: no PDF is linked from {OCW}{course}/download/")
        jobs += [Job(f"{OCW}{course}/{urllib.parse.quote(name)}", f"{course}/{name}") for name in names]
    print(f"ocw-pages: {len(jobs)} PDFs of {len(OCW_COURSES)} courses from ocw.mit.edu")
    bring(fetcher, "ocw-pages", folder, sized(fetcher, jobs), "PDFs")
    return count_files(folder)


# -- iNaturalist: one observer's photos ----------------------------------------------------------------------------------------------

def inat_rows(fetcher: Fetcher) -> tuple[list[dict], int]:
    """The kept fields of every research-grade observation of INAT_USER in INAT_FROM to INAT_TO that has a photo of an open licence, by
    the API's `id_above` paging (not `page`, which stops at 10,000), INAT_PAGE at a time and a request every INAT_WAIT seconds. The
    location and the user's name are not kept. Returns the rows, and how many of the observations came with a first photo whose own
    licence is not open (its picture is not in the open-data bucket), which are left out."""
    rows, dropped, last, total = [], 0, 0, None
    while True:
        query = {"user_id": INAT_USER, "d1": INAT_FROM, "d2": INAT_TO, "quality_grade": "research", "photos": "true",
                 "photo_license": ",".join(INAT_LICENCES), "per_page": INAT_PAGE, "order_by": "id", "order": "asc", "id_above": last}
        page, _ = read_json(fetcher, f"{INAT_API}?{urllib.parse.urlencode(query)}")
        total = page["total_results"] if total is None else total
        for o in page["results"]:
            photo, taxon = (o.get("photos") or [{}])[0], o.get("taxon") or {}
            if photo.get("license_code") not in INAT_LICENCES:
                dropped += 1
                continue
            rows.append({"id": o["id"], "taxon_name": taxon.get("name"), "taxon_common_name": taxon.get("preferred_common_name"),
                         "taxon_rank": taxon.get("rank"), "observed_on": o.get("observed_on"), "time_observed_at": o.get("time_observed_at"),
                         "license": o.get("license_code"), "photo_license": photo["license_code"], "photo_id": photo["id"],
                         "photo_ext": plain(photo["url"].rpartition(".")[2])})
        if len(page["results"]) < INAT_PAGE:
            return rows, dropped
        last = page["results"][-1]["id"]
        fetcher.sleep(INAT_WAIT)


def fetch_inat_observer(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """`observations.jsonl` (written beside it and renamed when whole; one that is there is not asked for again) and the first photo of
    each observation (`photos/<photo id>.<ext>`, from the open-data bucket's `large` size), INAT_WORKERS at once."""
    folder = root / "inat-observer"
    path = folder / "observations.jsonl"
    if not path.is_file():
        rows, dropped = inat_rows(fetcher)
        if not rows:
            raise FetchError(f"inat-observer: the API has no observation of user {INAT_USER} in {INAT_FROM} to {INAT_TO}")
        folder.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        part.replace(path)
        print(f"inat-observer: {len(rows)} observations ({dropped} left out: a first photo that is not openly licensed)")
    if labels_only:
        return count_files(folder)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    jobs = [Job(f"{INAT_PHOTOS}{row['photo_id']}/large.{row['photo_ext']}", f"photos/{row['photo_id']}.{row['photo_ext']}") for row in rows]
    # a photo's size is the server's: a HEAD for each would be a request more than the file, and `download` checks the length it was
    # sent; the budget is counted at the 470 KB a large photo came to on average (3,240 photos, 1.5 GB)
    within_budget("inat-observer", folder, [Job(job.url, job.rel, 470_000) for job in jobs])
    tally = limited(fetcher, INAT_WORKERS).fetch_all("photos", folder, jobs)
    print("inat-observer: photos: " + ", ".join(f"{n} {what}" for what, n in sorted(tally.items())))
    return count_files(folder)


# -- FTC robocalls, from a GitHub zip ----------------------------------------------------------------------------------------------

def fetch_robocall_ftc(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
    """The recordings, their transcripts (`metadata.csv`), the case PDFs and the licence of the robocall-audio-dataset repository's main
    branch, as the repository lays them out. GitHub makes the zip as it is asked for (no length, no Range): the zip comes whole
    beside the folder, the members the pile reads are written out of it (each beside its name and renamed when whole), and the zip is
    removed. A folder that holds the recordings and the three files already is left alone, and nothing is asked for."""
    folder, zipped = root / "robocall-ftc", root / "robocall-ftc.zip"
    recordings = list((folder / "audio-wav-16khz").glob("*.wav"))
    if len(recordings) == COUNTS["robocall-ftc"] and all((folder / name).is_file() for name in ROBOCALL_FILES):
        print(f"robocall-ftc: {len(recordings)} recordings kept")
        return count_files(folder)
    if not zipped.is_file():
        fetcher.download(ROBOCALL_ZIP, zipped)
    with zipfile.ZipFile(zipped) as z:
        try:
            members = [(info, info.filename.removeprefix(ROBOCALL_ROOT)) for info in z.infolist()
                       if not info.is_dir() and info.filename.startswith(ROBOCALL_ROOT)]
            members = [(info, safe_rel(rel)) for info, rel in members if rel in ROBOCALL_FILES or rel.startswith(ROBOCALL_DIRS)]
            expect("robocall-ftc", "recordings", sum(rel.endswith(".wav") for _, rel in members), COUNTS["robocall-ftc"])
            if (size := sum(info.file_size for info, _ in members)) > BUDGET["robocall-ftc"]:
                raise FetchError(f"robocall-ftc: {size / GB:.1f} GB in the zip, over the {BUDGET['robocall-ftc'] / GB:.1f} GB it was chosen at")
        except FetchError:
            z.close()
            zipped.unlink()  # (a zip that is refused is not kept to be refused again: the next run asks GitHub again)
            raise
        for info, rel in members:
            if (folder / rel).is_file() and (folder / rel).stat().st_size == info.file_size:
                continue
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            part = (folder / rel).with_name((folder / rel).name + ".part")
            with z.open(info) as source, open(part, "wb") as out:
                while chunk := source.read(1 << 20):
                    out.write(chunk)
            part.replace(folder / rel)
    zipped.unlink()
    print(f"robocall-ftc: {len(members)} files written out of the repository's zip")
    return count_files(folder)


# -- sources an earlier survey fetched and converted, not scripted yet -----------------------------------------------------------------

def not_scripted(source: str, why: str) -> Callable[..., int]:
    def fetch_nothing(fetcher: Fetcher, root: Path, labels_only: bool = False) -> int:
        raise FetchError(f"{source}: not scripted yet ({why}); what is in {root / source} came from an earlier survey, see MEDIA.md")

    return fetch_nothing


SOURCES: dict[str, Callable[..., int]] = {
    "ocw-14-01": fetch_ocw_14_01,
    "ocw-6-0002": fetch_ocw_6_0002,
    "chalk-radio": fetch_chalk_radio,
    "ami": fetch_ami,
    "icsi": fetch_icsi,
    "st2-followcam": fetch_st2_followcam,
    "aus-dashcam": fetch_aus_dashcam,
    "charades": fetch_charades,
    "sroie": fetch_sroie,
    "cord-v2": fetch_cord_v2,
    "ocw-pages": fetch_ocw_pages,
    "inat-observer": fetch_inat_observer,
    "fma": fetch_fma,
    "atm-bench": fetch_atm_bench,
    "birdvox-full-night": fetch_birdvox_full_night,
    "robocall-ftc": fetch_robocall_ftc,
    "android-in-the-wild": not_scripted("android-in-the-wild", "its screenshots are cut from TFRecord shards of a Google bucket"),
    "bd3-building-defects": not_scripted("bd3-building-defects", "its photos are cut from a parquet file"),
}
