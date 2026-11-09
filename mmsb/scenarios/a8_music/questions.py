"""The questions of A8, every answer from the genres the tracks' uploaders gave them (source.py), each asked the morning after the library's
newest track was added:

| template | question                                                                         | rule                          |
| `genre`  | Find every folk track in the folder Album 023. (4 to 40)                          | `set` of file names           |
| `count`  | How many hip-hop tracks are in the folder Album 056? (3 to 40)                    | `count`                       |
| `first`  | Is there any rock in the folder Album 012? If so, which track is the first one?   | `set` of one file name        |
| `most`   | Which folder has the most electronic tracks?                                      | `text`: the folder's name     |
| `absent` | `genre`, `count` or `first` of a genre that is asked about as present, in a folder  | `absent`                      |
|          | that has none of it; `most` of the held-out genre, which the library has none of    |                               |

**The genre is the uploader's**, a coarse label that a listener may not share (a folk track may be taken for a pop one): the questions ask of
four genres that have a sound of their own, hip-hop, folk, rock and electronic (ASKED), and a track is of the genre its uploader gave it. Of
every genre a listener may take for one asked about (NEAR: international and instrumental for folk, pop and instrumental for rock,
experimental and instrumental for electronic) the library holds no track while that genre is asked about, and instrumental never, as it is
taken for all; so a track is only taken for one of another genre when its uploader and a listener differ.

Which genres are asked about is drawn with the seed: one of the four is **held out**, a genre the library has not a track of (the question
`most` asks of it, once, and no question of a folder does, or the genre would be asked of only as absent); two are asked about as present; the
fourth is in the library, never asked about. The tracks are whole albums of the dataset's own (`album` in `tracks.csv`), at least three of each in the `small` subset, three or four
albums to a folder and a folder of at least 24 tracks, in the order of the albums, each album in the order of its tracks, named
`Album 001` onwards and `track_0001.mp3` onwards through the library: a folder has the mix of genres of the albums it was made of. Every file's
modified time is the time its track was uploaded to the archive (real, and no more than a time a person would read from the file).

A folder question is asked only of a folder with the tracks it asks for (a `genre` question of four to forty, a `count` of three to forty,
a `first` of two or more), and `most` only where the folder has at least three more tracks of the genre than any other. An absent folder question
is of one of the two genres asked about as present, in a folder that has none of it (so it reads as the questions that have an answer do)."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.a8_music.source import Track

CODE = "A8"
SOURCE_NAME = "library"
ASKED = ("Hip-Hop", "Folk", "Rock", "Electronic")
SAID = {"Hip-Hop": "hip-hop", "Folk": "folk", "Rock": "rock", "Electronic": "electronic"}  # how a person says a genre of a track
NEAR = {"Hip-Hop": (), "Folk": ("International", "Instrumental"), "Rock": ("Pop", "Instrumental"),
        "Electronic": ("Experimental", "Instrumental")}  # the genres easily taken for one asked about, which the library has no track of
ALWAYS_OUT = ("Instrumental",)
MIN_ALBUM = 2  # the tracks of an album the library takes (of the genres it has)
FOLDER_MIN = {"tiny": 15, "standard": 24}  # the tracks a folder has at least
TARGET = {"tiny": 150, "standard": 3000}  # the tracks the library has, about
GENRE_FILES = (4, 40)  # a `genre` question's answer
COUNT_FILES = (3, 40)  # a `count` question's answer
FIRST_FILES = 2  # a `first` question's folder has at least this many tracks of the genre
MARGIN = 3  # the folder `most` asks for has at least this many more tracks of the genre than any other
QUOTAS = {"tiny": {"genre": 1, "count": 1, "first": 1, "most": 1, "absent_genre": 1, "absent_count": 0, "absent_first": 0, "absent_most": 1},
          "standard": {"genre": 6, "count": 4, "first": 3, "most": 2, "absent_genre": 2, "absent_count": 1, "absent_first": 1, "absent_most": 1}}
FILES = "<the file names, as in track_0042.mp3, separated by commas>"
FILE = "<the file name, as in track_0042.mp3>"
NUMBER = "<a number>"
FOLDER = "<the folder name, as in Album 012>"


@dataclass(frozen=True)
class Roles:
    held: str  # the genre the library has no track of
    asked: tuple[str, str]  # the two genres asked about as present
    other: str  # the fourth, in the library and never asked about


def roles_of(seed: int, scale: str) -> Roles:
    order = list(ASKED)
    random.Random(f"a8:{seed}:{scale}:genres").shuffle(order)
    return Roles(order[0], (order[1], order[2]), order[3])


def viable(folders: Sequence[Sequence[Track]], roles: Roles, scale: str) -> bool:
    """Whether the questions of `scale` can all be asked of `folders`: a folder with the margin for each `most`, and enough folders and genres
    that have what a `genre`, `count` and `first` question asks for (each folder and genre once) and none of what an absent one asks for."""
    quota = QUOTAS[scale]
    if len(folders) < 2:
        return False
    counts = [{g: sum(t.genre == g for t in folder) for g in ASKED} for folder in folders]
    for g in roles.asked[:quota["most"]]:
        ranked = sorted(c[g] for c in counts)
        if ranked[-1] - ranked[-2] < MARGIN:
            return False
    pairs = [(k, g) for k in range(len(folders)) for g in roles.asked]
    have = [sum(low <= counts[k][g] <= high for k, g in pairs) for low, high in (GENRE_FILES, COUNT_FILES, (FIRST_FILES, 10 ** 6))]
    empty = sum(counts[k][g] == 0 for k, g in pairs)
    return (have[0] >= quota["genre"] and have[1] >= quota["count"] and have[2] >= quota["first"]
            and sum(have[:3]) >= quota["genre"] + quota["count"] + quota["first"] and empty >= quota["absent_genre"] + quota["absent_count"] + quota["absent_first"])


def choose(tracks: Sequence[Track], scale: str, seed: int) -> tuple[Roles, list[list[Track]]]:
    """The genres the questions of `scale` ask about, drawn with `seed`, and the folders of the library: the albums of the tracks of the
    genres the library has, in a seeded order, three or four to a folder, each folder at least FOLDER_MIN tracks, until the library has about
    TARGET[scale] tracks; each album's tracks in their order. The order is drawn again, deterministically, where the folders it makes cannot
    be asked everything of (`viable`)."""
    roles = roles_of(seed, scale)
    out = {*ALWAYS_OUT, roles.held, *(n for g in (roles.held, *roles.asked) for n in NEAR[g])}
    albums: dict[int, list[Track]] = {}
    for t in tracks:
        if t.genre not in out:
            albums.setdefault(t.album, []).append(t)
    kept = [sorted(v, key=lambda t: (t.number, t.id)) for _, v in sorted(albums.items()) if len(v) >= MIN_ALBUM]
    for attempt in range(100):
        random.Random(f"a8:{seed}:{scale}:albums:{attempt}").shuffle(kept)
        folders: list[list[Track]] = [[]]
        total = 0
        for album in kept:
            if total >= TARGET[scale]:
                break
            if len(folders[-1]) >= FOLDER_MIN[scale]:
                folders.append([])
            folders[-1] += album
            total += len(album)
        if len(folders) > 1 and len(folders[-1]) < FOLDER_MIN[scale]:
            last = folders.pop()
            folders[-1] += last
        if viable(folders, roles, scale):
            return roles, folders
    raise ValueError(f"no order of the albums at {scale}, seed {seed}, makes folders that every question can be asked of")


def slug(genre: str) -> str:
    return re.sub(r"[^a-z]+", "", genre.casefold())


def folder_name(path: str) -> str:
    return path.split("/")[1]


def questions_of(folders: Sequence[Sequence[tuple[Track, realdump.File]]], roles: Roles, scale: str, seed: int, now: float) -> tuple[Question, ...]:
    """The questions of the library `folders` (each a list of its tracks with their files), over `roles`; `now` is epoch seconds."""
    rng = random.Random(f"a8:{seed}:{scale}:questions")
    quota = QUOTAS[scale]
    files = [f for folder in folders for _, f in folder]
    since, until = min(f.t for f in files), max(f.t + f.length for f in files)
    names = [folder_name(folder[0][1].path) for folder in folders]
    of = [{g: [f for t, f in folder if t.genre == g] for g in ASKED} for folder in folders]  # the files of each genre in each folder
    used: set[tuple[int, str]] = set()  # a folder and a genre are asked about once
    out: list[Question] = []

    def basename(f: realdump.File) -> str:
        return f.path.rsplit("/", 1)[-1]

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[realdump.File], value: str | None, absent: bool = False) -> Question:
        return realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text,
                                 kind="absent" if absent else template, template="absent_" + template if absent else template, modality="audio",
                                 shape=shape, grade=grade, evidence=[Span(f.path, 0.0, f.length) for f in found], sources=(SOURCE_NAME,), since=since,
                                 until=until, value=value)

    def pick(candidates: Sequence[tuple[int, str]], n: int, role: str) -> list[tuple[int, str]]:
        order = [c for c in candidates if c not in used]
        rng.shuffle(order)
        chosen: list[tuple[int, str]] = []
        for c in order:
            if len(chosen) < n and c not in used:
                chosen.append(c)
                used.add(c)
        if len(chosen) < n:
            raise ValueError(f"{role}: only {len(chosen)} of the {n} folders and genres it asks about are in the library")
        return chosen

    pairs = [(k, g) for k in range(len(folders)) for g in roles.asked]
    # most: the folder with the most of a genre, where no other is within the margin
    for g in roles.asked[:quota["most"]]:
        ranked = sorted(range(len(folders)), key=lambda k: -len(of[k][g]))
        if len(of[ranked[0]][g]) - len(of[ranked[1]][g]) < MARGIN:
            raise ValueError(f"no folder has {MARGIN} more {g} tracks than any other")  # (`choose` makes folders that do)
        top = ranked[0]
        name = names[top]
        accepted = f"{name.casefold()}|album {int(name[-3:])}"
        wrong = [names[k] for k in ranked[1:3]]
        out.append(ask(f"most-{slug(g)}", f"Which folder has the most {SAID[g]} tracks?", "most", FOLDER, realdump.grade("text", truth=accepted, wrong=wrong),
                       of[top][g], name))
    # genre, count and first, in a folder that has the tracks
    for template, low, high, role in (("genre", GENRE_FILES[0], GENRE_FILES[1], "genre"), ("count", COUNT_FILES[0], COUNT_FILES[1], "count"),
                                      ("first", FIRST_FILES, 10 ** 6, "first")):
        for k, g in pick([(k, g) for k, g in pairs if low <= len(of[k][g]) <= high], quota[role], role):
            found = of[k][g]
            if template == "genre":
                items = [basename(f) for f in found]
                out.append(ask(f"genre-{slug(g)}-{names[k][-3:]}", f"Find every {SAID[g]} track in the folder {names[k]}.", "genre", FILES,
                               realdump.grade("set", items=items), found, ", ".join(items)))
            elif template == "count":
                out.append(ask(f"count-{slug(g)}-{names[k][-3:]}", f"How many {SAID[g]} tracks are in the folder {names[k]}?", "count", NUMBER,
                               realdump.grade("count", truth=str(len(found))), found, str(len(found))))
            else:
                out.append(ask(f"first-{slug(g)}-{names[k][-3:]}", f"Is there any {SAID[g]} in the folder {names[k]}? If so, which track is the first one?",
                               "first", FILE, realdump.grade("set", items=[basename(found[0])]), found[:1], basename(found[0])))
    # absent: a folder with none of a genre that is asked about as present
    present = [(k, g) for k, g in pairs if not of[k][g]]
    for template, role in (("genre", "absent_genre"), ("count", "absent_count"), ("first", "absent_first")):
        for _ in range(quota[role]):
            for k, g in pick(present, 1, role):
                text = {"genre": f"Find every {SAID[g]} track in the folder {names[k]}.", "count": f"How many {SAID[g]} tracks are in the folder {names[k]}?",
                        "first": f"Is there any {SAID[g]} in the folder {names[k]}? If so, which track is the first one?"}[template]
                out.append(ask(f"{template}-{slug(g)}-{names[k][-3:]}", text, template, {"genre": FILES, "count": NUMBER, "first": FILE}[template],
                               realdump.grade("absent"), [], None, absent=True))
    for _ in range(quota["absent_most"]):
        out.append(ask(f"most-{slug(roles.held)}", f"Which folder has the most {SAID[roles.held]} tracks?", "most", FOLDER, realdump.grade("absent"), [], None,
                       absent=True))
    return realdump.split(out, seed)
