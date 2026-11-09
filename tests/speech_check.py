"""A second reader of the transcripts of the speech scenarios (V9, A6, A7), for their tests: the SRT read here by a plain regular expression and the
words of a phrase searched in the running text of a file, never with scenarios/realdump/speech.py. `check_questions` takes each question's truth
from it and checks it against the question's grade, evidence and the files' times."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

from mmsb.core.grade import grade
from mmsb.scenarios import realdump

BLOCK = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+) --> (\d+):(\d\d):(\d\d)[,.](\d+)\r?\n(.*?)(?=\r?\n\r?\n|\Z)", re.DOTALL)


def blocks(path: Path) -> list[tuple[float, float, str]]:
    """`(start, end, text)` of each caption of an SRT, its notes of sounds (`[MUSIC]`) and names of speakers (`NAME:`) taken out."""
    out = []
    for m in BLOCK.finditer(Path(path).read_text(encoding="utf-8-sig")):
        g = m.groups()
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", g[8].replace("\n", " "))
        text = re.sub(r"^\s*[A-Z][A-Z .'-]{1,40}:\s*", "", text)
        out.append((int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 10 ** len(g[3]),
                    int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 10 ** len(g[7]), text))
    return out


def stream(cues: list[tuple[float, float, str]]) -> tuple[list[str], list[float]]:
    """The words of the captions, and the start of the caption each is in."""
    words, at = [], []
    for t0, _, text in cues:
        found = re.findall(r"[a-z0-9']+", text.casefold())
        words += found
        at += [t0] * len(found)
    return words, at


def hits(cues: list[tuple[float, float, str]], phrase: str) -> list[float]:
    """The start of the caption in which each saying of `phrase` begins: its words one after the other, the last with or without an s."""
    words, at = stream(cues)
    want = re.findall(r"[a-z0-9']+", phrase.casefold())
    n = len(want)
    return [at[i] for i in range(len(words) - n + 1) if words[i:i + n - 1] == want[:-1] and words[i + n - 1] in (want[-1], want[-1] + "s", want[-1] + "es")]


def run_on(cues: list[tuple[float, float, str]], phrase: str) -> bool:
    """Whether the letters and digits of `phrase` run on in the captions, whatever the spacing."""
    flat = "".join(re.sub(r"[^a-z0-9]", "", text.casefold()) for _, _, text in cues)
    return re.sub(r"[^a-z0-9]", "", phrase.casefold()) in flat


def check_questions(questions, files, srt_of, terms, never, tol_s: float = 60.0) -> None:
    """Each question's truth as this reader finds it. `files` are the plan's files in the order of the term, `srt_of(file)` the path of the SRT of
    each, `terms` and `never` the scenario's (`search` words by the key a qid carries)."""
    cues = [blocks(srt_of(f)) for f in files]
    start = [datetime.fromtimestamp(f.t) for f in files]
    name = [f.path.rsplit("/", 1)[-1].casefold() for f in files]
    by_key = {"-".join(re.findall(r"[a-z0-9']+", t.search.casefold())): t.search for t in (*terms, *never)}
    by_said = {t.said: t.search for t in (*terms, *never)}
    for q in questions:
        template = q.tags["template"]
        if template in ("what", "who", "before"):
            continue  # written, and checked against the cues by the scenario's build and by the tests of the curated questions
        absent = template.startswith("absent_")
        if template == "order":
            said_a, said_b = re.search(r"bring up (.+) before or after (.+)\?$", q.question).groups()
            a, b = by_said[said_a], by_said[said_b]
            named = re.search(r"of (\w+ \d+ \w+ \d{4}),", q.question)[1]  # the day tells the file: `Wednesday 26 June 2024` (an English locale)
            day = [i for i in range(len(files)) if f"{start[i]:%A} {start[i].day} {start[i]:%B %Y}" == named]
            both = [i for i in day if len(hits(cues[i], a)) == 1 and len(hits(cues[i], b)) == 1]
            assert len(both) == 1, f"{q.qid}: not both said once, in one of the files of {named}: {day}"
            assert all(not hits(cues[i], x) for i in day if i != both[0] for x in (a, b)), f"{q.qid}: said in another file of {named} too"
            day = both
            at_a, at_b = hits(cues[day[0]], a)[0], hits(cues[day[0]], b)[0]
            truth = "before" if at_a < at_b else "after"
            assert q.grade.rule == "text" and q.grade.truth == truth and abs(at_a - at_b) >= 120, q.qid
            assert grade(q.grade, truth) and not grade(q.grade, "after" if truth == "before" else "before"), q.qid
            continue
        search = by_key[q.qid.removeprefix(template.replace("_", "-") + "-")]
        found = [(i, hits(cues[i], search)) for i in range(len(files)) if hits(cues[i], search)]
        if absent:
            assert not found and not any(run_on(c, search) for c in cues), f"{q.qid}: {search!r} is said"
            assert q.grade.rule == "absent" and not q.truth.evidence
            assert grade(q.grade, "No, they never mention it.") and not grade(q.grade, "Yes, at 10:15.")
            continue
        assert found, f"{q.qid}: {search!r} is said nowhere"
        if template in ("first", "did"):
            i, at = found[0][0], found[0][1][0]
            when = start[i] + timedelta(seconds=at)
            assert (q.grade.date, q.grade.time) == (when.date().isoformat(), when.strftime("%H:%M:%S")), q.qid
            assert q.grade.tol_s == tol_s
            assert grade(q.grade, f"{when:%Y-%m-%d %H:%M:%S}") and not grade(q.grade, f"{when + timedelta(minutes=3):%Y-%m-%d %H:%M:%S}"), q.qid
            assert not grade(q.grade, f"{when + timedelta(days=1):%Y-%m-%d %H:%M:%S}"), q.qid
        elif template == "which":
            assert len(found) == 1 and q.grade.items == (name[found[0][0]],), q.qid
            assert grade(q.grade, name[found[0][0]]) and not grade(q.grade, name[(found[0][0] + 1) % len(name)]), q.qid
        elif template == "all":
            items = tuple(sorted(name[i] for i, _ in found))
            assert 2 <= len(items) and q.grade.items == items, q.qid
            assert grade(q.grade, ", ".join(items)) and not grade(q.grade, ", ".join(items[1:])), q.qid
        elif template == "count":
            assert q.grade.truth == str(len(found)) and 2 <= len(found) < len(files), q.qid
            assert grade(q.grade, str(len(found))) and not grade(q.grade, str(len(found) + 1)), q.qid
        elif template == "long":
            assert len(found) == 1 and abs(float(q.grade.truth) - files[found[0][0]].length / 60) <= 0.05, q.qid
            assert q.grade.tol_s == 1.0 and grade(q.grade, f"{float(q.grade.truth) + 0.5:.1f}") and not grade(q.grade, f"{float(q.grade.truth) + 3:.1f}"), q.qid
        else:
            raise AssertionError(f"{q.qid}: a template this reader does not know")
        spans = {s.file for s in q.truth.evidence}
        assert spans <= {f.path for f in files} and q.truth.evidence, q.qid


def check_plan(scenario, files, questions, low: int, high: int, names: str) -> None:
    """What every plan of a speech scenario holds: a count of questions in [`low`, `high`] with about a quarter absent and about 30% dev, each filed
    under a category (four at least), qids that are file names and not repeated, files named as `names` says and timed in order, and each question asked
    after the last file."""
    from mmsb.core.question import QID
    from mmsb.core.taxonomy import check as check_filing
    from mmsb.core.taxonomy import filed

    check_filing(scenario.shelf, scenario.filing)
    assert low <= len(questions) <= high, len(questions)
    absent = sum(q.tags["absent"] for q in questions)
    assert 0.15 <= absent / len(questions) <= 0.35, (absent, len(questions))
    assert abs(sum(q.split == "dev" for q in questions) / len(questions) - realdump.DEV_SHARE) < 0.1
    assert len({q.qid for q in questions}) == len(questions) and all(QID.fullmatch(q.qid) for q in questions)
    assert len({filed(scenario, q.tags["template"])[0] for q in questions}) >= 4
    assert not [q.qid for q in questions if filed(scenario, q.tags["template"])[0] == "unfiled"]
    assert all(re.fullmatch(names, f.path) for f in files), [f.path for f in files if not re.fullmatch(names, f.path)]
    assert [f.t for f in files] == sorted(f.t for f in files) and len({f.t for f in files}) == len(files)
    last = max(f.t + float(f.length) for f in files)
    assert all(datetime.fromisoformat(q.now).timestamp() > last for q in questions)
    assert all(q.scope.since <= files[0].t and q.scope.until >= last for q in questions)
    asked = [q.qid.split("-", 1)[1] for q in questions if q.tags["template"] in ("first", "did", "which", "all", "count", "long")]
    assert len(asked) == len(set(asked)), "a term is asked about once"


def check_notes(notes, files, srt_of, stems: list[str]) -> None:
    """Each curated note of a scenario: the phrase it rests on is in the captions of its recording within 5 s of its evidence (as this reader reads them),
    its value is graded right, and nothing in its question names a file, a path or a time of day."""
    for n in notes:
        (i,) = [k for k, s in enumerate(stems) if n.file in s]
        near = [c for c in blocks(srt_of(files[i])) if c[1] >= n.t0 - 5 and c[0] <= n.t1 + 5]
        assert hits(near, n.check), f"{n.key}: {n.check!r} is not said in {files[i].path} from {n.t0} to {n.t1}"
        assert grade(n.grade, n.value), n.key
        assert not re.search(r"\.(mp4|mp3|m4a)|\d{1,2}:\d\d", n.text), n.key


def check_terms(terms, never, srts, at_least: int = 25, never_at_least: int = 6) -> None:
    """The scenario's terms as this reader finds them: at least `at_least` of them are said in some caption of `srts`, and at least `never_at_least`
    of `never` are said in none (nor run on in any spacing), and no term is in both lists."""
    cues = [blocks(p) for p in srts]
    said = [t.search for t in terms if any(hits(c, t.search) for c in cues)]
    assert len(said) >= at_least, f"{len(said)} terms are said: {sorted(set(t.search for t in terms) - set(said))} are not"
    assert len(terms) == len({t.search for t in terms}) and not {t.search for t in terms} & {t.search for t in never}
    unsaid = [t.search for t in never if not any(hits(c, t.search) or run_on(c, t.search) for c in cues)]
    assert len(never) >= never_at_least and unsaid == [t.search for t in never], f"said after all: {set(t.search for t in never) - set(unsaid)}"
