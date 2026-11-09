"""The questions of M3, every answer from the transcripts of the dataset (source.py: Whisper's, a machine's, so what is "said" is what the
transcript has), each asked the morning after the box's last message. The English messages are the pile (the Mandarin ones, 3.8% of the
dataset, are left out: a scheme is the same in both languages and a transcript in characters would not say so).

| template  | question                                                                              | rule                              |
| `scheme`  | Find every voicemail that mentions Wells Fargo. (3 to 40; or the same between Monday  | `set` of file names               |
|           | 14 September and Sunday 20 September, of a scheme that has more)                      |                                   |
| `count`   | How many voicemails mention Spectrum? (the same two forms)                            | `count`                           |
| `first`   | Did anyone leave me a voicemail that mentions PayPal? If so, when was the first one?  | `time` with the date, 60 s        |
| `single`  | Find the voicemail that mentions Walmart. (a name that one message has)               | `set` of one file name            |
| `price`   | What dollar amount did the voicemail I got on Thursday 1 October at 14:07 mention?    | `amount`, a dollar                |
| `percent` | What percentage did the voicemail I got on ... at ... mention?                        | `amount`, a point                 |
| `caller`  | Who did the voicemail I got on ... at ... say it was from?                            | `text`, the names it is called by |
| `absent`  | `scheme`, `count`, `first` or `single` of a name no message has                       | `absent`                          |

A message's time is invented (a seeded spread over nine weeks, 7 September to 8 November 2026, in the hours of a day; the source has none), so
no answer is only a time: `first` asks which message came first among those about a scheme, and the three per-message questions name their
message by the minute it came in and ask what it said. Every message is at least two minutes from the next, and a question takes the time
as the file's modified time.

A scheme is a name or a few words a person says (SCHEMES) with the pattern that finds it in a transcript (`strict`), whole words, in any case,
and the words of a transcript that are easily taken for it (`near`: its products for a brand, its neighbours in the business, the same
scheme said another way). A message that has a near word but not the pattern is a message a listener may count with the scheme and the
labels do not, so it is left out of the box while the scheme is asked about; so is every message that has a scheme that is asked as absent
(`held out`), whatever the transcript has of it. A name that no message of the dataset has (ABSENT: Netflix, Venmo, a lottery) is checked
against every transcript and every case name at the build. No two asked schemes are such that one's near words are the other's messages.

The per-message questions take a message only where its transcript is plain: `price` and `percent`, where it names one amount (a dollar
amount, or a percentage) that the same script has in at least three messages (one mishearing is not a label to trust), and `caller`, where the
first 30 words have one of CALLERS as the caller ("calling you from AT&T Direct TV", and not "your Amazon account", which a warning is about).
A `first` scheme is one whose first message is not among the box's first ten (the list of the files would give its time), and a `tiny` box
holds at least the messages the per-message questions are asked of. None asks for a phone number or the name of a person, which the
transcripts have: they are neither question nor answer."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from mmsb.core.question import Question, Span
from mmsb.scenarios import realdump
from mmsb.scenarios.m3_voicemail.source import Call

CODE = "M3"
SOURCE_NAME = "voicemail"  # the dump's source
TOL_S = 60.0  # a message's time is told to the second; a person reads it to the minute
MAX_FILES = 40  # a `scheme` question lists no more than this
WEEK_FILES = (3, 36)  # the messages of a week that a windowed `scheme` or `count` question has (a week is one that has some, and not too many)
PLAIN_COUNT = (3, 40)  # the answer of a `count` question over the whole box
FIRST_COUNT = (3, 60)  # the messages a `first` question's scheme has: more, and the first is likely to be among the box's first
EARLIEST = 10  # the first message about a scheme is not among the box's first so many (the list of the files would give its time)
SEEN_BY = 3  # the messages that must have a dollar amount or a percentage for it to be asked about
CLAIMED = 30  # the words at the start of a message that a caller's name is in
QUOTAS = {"tiny": {"scheme": 1, "count": 1, "first": 1, "single": 1, "absent_first": 1, "absent_single": 1, "price": 1, "percent": 0,
                   "caller": 1, "week_scheme": 0, "week_count": 0, "absent_scheme": 0, "absent_count": 0},
          "standard": {"scheme": 2, "count": 2, "first": 3, "single": 2, "absent_first": 2, "absent_single": 1, "price": 2, "percent": 1,
                       "caller": 3, "week_scheme": 2, "week_count": 1, "absent_scheme": 1, "absent_count": 1}}
TINY_CALLS = 90  # the messages of the `tiny` box, the asked-about ones among them
TINY_MOST = 12  # and no scheme asked about at `tiny` has more messages than this
FILES = "<the file names, as in msg_0042.wav, separated by commas>"
FILE = "<the file name, as in msg_0042.wav>"
NUMBER = "<a number>"
WHEN = "<the date and time, as YYYY-MM-DD HH:MM>"
DOLLARS = "<a number of dollars>"
PERCENT = "<a percentage>"
COMPANY = "<the company or agency>"


@dataclass(frozen=True)
class Scheme:
    said: str  # as a person says it: "mentions {said}"
    strict: str  # the pattern a transcript has
    near: str = ""  # the pattern of what is easily taken for it, whose messages are left out while it is asked about


SCHEMES = {  # the schemes that a message of the dataset has: how many messages have each is in the comment, over the English ones
    "amazon": Scheme("Amazon", r"\bamazon\b", r"\bamz\b|\balexa\b|\bprime\b"),  # 331
    "apple": Scheme("Apple", r"\bapple\b", r"\biphones?\b|\bicloud\b|\bmac ?books?\b|\bair ?pods?\b|\bearpods?\b|\bimac\b|\bipad\b"),  # 186
    "ssa": Scheme("Social Security", r"\bsocial security\b", r"\bssa\b|\bssn\b|\bsupplemental security\b|\bsocial\b"),  # 279
    "student": Scheme("my student loans", r"\bstudent\b|\bnelnet\b|\bnavient\b|\bgreat lakes\b|\bloan forgiveness\b",
                      r"\bfed loans?\b|\bfederal loans?\b|\bloans?\b"),  # 70
    "warranty": Scheme("my car's warranty", r"\bwarrant(?:y|ies)\b", r"\bvehicle\b|\bdealer\b|\bmanufacturer\b|\bservice contract\b|\bcoverage\b|\bextended\b"),  # 55
    "electric": Scheme("my electric company", r"\belectric(?:ity)?\b", r"\benergy\b|\bpower\b|\butility\b|\butilities\b|\bdisconnect\w*"),  # 65
    "airpods": Scheme("AirPods", r"\bair ?pods?\b", r"\bearpods?\b|\bearbuds?\b|\bheadphones?\b"),  # 60
    "macbook": Scheme("a MacBook", r"\bmac ?books?\b", r"\blaptop\b|\bmac\b"),  # 66
    "credit": Scheme("my credit cards", r"\binterest rates?\b|\bcredit cards?\b", r"\bmastercard\b|\bvisa\b|\bdiscover\b|\bbalance\b"),  # 69
    "customs": Scheme("customs", r"\bcustoms\b", r"\bcbp\b|\bborder protection\b|\bseized\b|\bparcel\b|\bpackage\b"),  # 27
    "directv": Scheme("AT&T or DirecTV", r"\bat&t\b|\bdirect ?tv\b|\bdirecttv\b", r"\bsatellite\b|\bdish\b|\btv\b"),  # 27
    "irs": Scheme("the IRS", r"\birs\b", r"\binternal revenue\b|\bback tax(?:es)?\b|\btax(?:es)?\b|\bfederal reserve\b"),  # 15
    "wells": Scheme("Wells Fargo", r"\bwells fargo\b", r"\bwells\b|\bfargo\b"),  # 10
    "marriott": Scheme("Marriott", r"\bmarriott\b", r"\bhilton\b|\bhotels?\b|\bresorts?\b|\bvacations?\b|\btimeshare\b"),  # 8
    "hilton": Scheme("Hilton", r"\bhilton\b", r"\bmarriott\b|\bhotels?\b|\bresorts?\b|\bvacations?\b|\btimeshare\b"),  # 6
    "spectrum": Scheme("Spectrum", r"\bspectrum\b", r"\bcharter\b|\bxfinity\b|\bcomcast\b|\bcable\b"),  # 11
    "comcast": Scheme("Comcast or Xfinity", r"\bcomcast\b|\bxfinity\b", r"\bcable\b|\bspectrum\b"),  # 6
    "paypal": Scheme("PayPal", r"\bpaypal\b", r"\bcash app\b|\bvenmo\b|\bzelle\b"),  # 4
    "cashapp": Scheme("Cash App", r"\bcash app\b", r"\bpaypal\b|\bvenmo\b|\bzelle\b"),  # 2
    "microsoft": Scheme("Microsoft", r"\bmicrosoft\b", r"\bwindows\b|\bteamviewer\b|\btech support\b|\btechnical support\b"),  # 4
    "fedex": Scheme("FedEx", r"\bfedex\b", r"\bups\b|\busps\b|\bdhl\b|\bcourier\b|\bparcel\b"),  # 4
    "medicare": Scheme("Medicare", r"\bmedicare\b", r"\bmedicaid\b|\bhealth\b|\binsurance\b|\bsenior\b"),  # 4
    "capone": Scheme("Capital One", r"\bcapital one\b", r"\bcapital\b"),  # 5
    "chase": Scheme("Chase", r"\bchase\b", r"\bjpmorgan\b|\bj\.p\. morgan\b"),  # 5
    "boa": Scheme("Bank of America", r"\bbank of america\b", r"\bamerica\b"),  # 6
    "xbox": Scheme("an Xbox or a PlayStation", r"\bxbox\b|\bplaystation\b", r"\bgaming\b|\bgame\b|\bconsole\b"),  # 4
    "walmart": Scheme("Walmart", r"\bwal-?mart\b", r"\bwal\b"),  # 1
    "fourseasons": Scheme("Four Seasons", r"\bfour seasons\b", r"\bcomplimentary stay\b|\bhotels?\b"),  # 1
    "oge": Scheme("Oklahoma Gas and Electric", r"\boklahoma gas\b", r"\bgas and electric\b"),  # 1
}
ABSENT = {  # names that no transcript and no case name of the dataset has (checked at the build, `choose`)
    "netflix": Scheme("Netflix", r"\bnetflix\b|\bhulu\b|\bdisney\b"),
    "lottery": Scheme("a lottery prize", r"\blotter(?:y|ies)\b|\bsweepstakes\b|\bjackpot\b|\braffle\b"),
    "westernunion": Scheme("Western Union", r"\bwestern union\b|\bmoneygram\b"),
    "venmo": Scheme("Venmo", r"\bvenmo\b|\bzelle\b"),
    "verizon": Scheme("Verizon", r"\bverizon\b|\bt-mobile\b|\bsprint\b"),
    "bitcoin": Scheme("bitcoin", r"\bbitcoin\b|\bcrypto\w*\b|\bcoinbase\b"),
    "uber": Scheme("Uber", r"\buber\b|\blyft\b"),
    "facebook": Scheme("Facebook", r"\bfacebook\b|\binstagram\b|\btiktok\b"),
    "costco": Scheme("Costco", r"\bcostco\b|\bsam's club\b|\bbest buy\b|\bhome depot\b"),
}
SINGLES = ("walmart", "fourseasons", "oge")  # the schemes that exactly one message has (`single`)
CALLERS = {  # the callers a `caller` question takes: key: (the names a right answer has, `|` between them; the words in which a transcript has it
    # as the caller: "calling you from AT&T Direct TV", and not "your Amazon account")
    "amazon": ("amazon", r"\b(?:from|with|this is) amazon\b"), "att": ("at t|att|directv|direct tv", r"\bfrom at&t\b"),
    "wells": ("wells fargo|fargo", r"\b(?:from|dear) wells fargo\b"), "spectrum": ("spectrum", r"\bfrom spectrum\b"),
    "marriott": ("marriott", r"\b(?:from|choosing) marriott\b"), "hilton": ("hilton", r"\b(?:from|choosing) hilton\b"),
    "paypal": ("paypal", r"\bfrom paypal\b"), "fedex": ("fedex", r"\bfedex international express\b|\bfrom fedex\b"),
    "comcast": ("comcast|xfinity", r"\bfrom (?:comcast|xfinity)\b"), "capone": ("capital one", r"\bfrom capital one\b"),
    "chase": ("chase", r"\bfrom chase\b"), "boa": ("bank of america", r"\bfrom bank of america\b"),
    "ssa": ("social security", r"\b(?:from|calling) the (?:department of )?social security (?:administration|department|office)\b"),
    "cbp": ("customs|border protection|cbp", r"\b(?:from|call is from) (?:the )?(?:department of )?(?:u\.?s\.? )?customs and border protection\b"),
    "jcpl": ("jersey central", r"\bwith jersey central\b")}
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
MONEY_WORDS = re.compile(r"\bdollars?\b|\bbucks?\b|\bcents?\b|\bthousand\b|\bhundred\b|\bmillion\b|\bbillion\b", re.I)
PERCENTS = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent)", re.I)
RANGE = re.compile(r"\d\s*(?:to|-|and)\s*\d+(?:\.\d+)?\s?(?:%|percent)", re.I)  # "0 to 5 percent": two numbers, no one


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


@dataclass(frozen=True)
class Roles:
    scheme: tuple[str, ...]  # schemes (keys of SCHEMES) a `scheme` question asks about, over the whole box
    count: tuple[str, ...]
    week_scheme: tuple[str, ...]  # a scheme a `scheme` question asks about in a week
    week_count: tuple[str, ...]
    first: tuple[str, ...]
    single: tuple[str, ...]
    absent_scheme: tuple[str, ...]  # keys of ABSENT, or of SCHEMES: a scheme the dump holds no message of (every one is left out)
    absent_count: tuple[str, ...]
    absent_first: tuple[str, ...]
    absent_single: tuple[str, ...]

    def asked(self) -> tuple[str, ...]:
        """Every scheme the dump is made around: the real ones asked about, and those held out."""
        return tuple(dict.fromkeys(k for v in (self.scheme, self.count, self.week_scheme, self.week_count, self.first, self.single,
                                               self.absent_scheme, self.absent_count, self.absent_first, self.absent_single) for k in v))


def table(key: str) -> Scheme:
    return SCHEMES.get(key) or ABSENT[key]


def sets_of(calls: Sequence[Call]) -> dict[str, tuple[frozenset[int], frozenset[int]]]:
    """For each scheme of SCHEMES, the numbers (in `calls`) of the messages that have it (`strict`), and of those that have a near word and
    not it."""
    out = {}
    for key, s in SCHEMES.items():
        strict, near = rx(s.strict), rx(s.near) if s.near else None
        has = frozenset(i for i, c in enumerate(calls) if strict.search(c.transcript))
        out[key] = (has, frozenset(i for i, c in enumerate(calls) if i not in has and near and near.search(c.transcript)))
    return out


def check_absent(calls: Sequence[Call], cases: Sequence[str]) -> None:
    """Refuses the build when a name of ABSENT is in a transcript (of any message, in any language) or a case name."""
    for key, s in ABSENT.items():
        found = rx(s.strict)
        if hit := next((c for c in calls if found.search(c.transcript)), None) or next((x for x in cases if found.search(x.replace("-", " "))), None):
            raise ValueError(f"{key} is asked about as a name no message has, and {hit if isinstance(hit, str) else hit.file} has it")


def classes_of(english: Sequence[Call]) -> tuple[dict[float, list[int]], dict[float, list[int]], dict[str, list[int]]]:
    """The per-message questions' pools, as numbers in `english`: the messages with each dollar amount (`price`: a dollar or more) and each
    percentage (`percent`) that SEEN_BY messages or more have, and those with each caller (`caller`)."""
    amounts, percents, callers = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, c in enumerate(english):
        if (v := money_of(c)) is not None:
            amounts[v].append(i)
        if (p := percent_of(c)) is not None:
            percents[p].append(i)
        if (k := caller_of(c)) is not None:
            callers[k].append(i)
    return ({v: ix for v, ix in amounts.items() if len(ix) >= SEEN_BY and v >= 1}, {v: ix for v, ix in percents.items() if len(ix) >= SEEN_BY},
            dict(callers))


def per_message(english: Sequence[Call], candidates: set[int], rng: random.Random, quota: dict[str, int]) -> list[int]:
    """Of the `candidates` (numbers in `english`), the messages of the tiny box that its per-message questions are asked of: for each of
    `price`, `percent` and `caller`, as many of different amounts, percentages or callers as the quota asks, one message of each."""
    out: list[int] = []
    for role, pools in zip(("price", "percent", "caller"), classes_of(english), strict=True):
        keys = sorted(k for k, ix in pools.items() if candidates & set(ix))
        rng.shuffle(keys)
        if len(keys) < quota[role]:
            raise ValueError(f"{role}: only {len(keys)} of the {quota[role]} it asks about are among the messages the tiny box may have")
        out += [rng.choice(sorted(candidates & set(pools[k]))) for k in keys[:quota[role]]]
    return list(dict.fromkeys(out))


def choose(calls: Sequence[Call], cases: Sequence[str], scale: str, seed: int) -> tuple[Roles, list[Call]]:
    """The schemes the questions of `scale` ask about, drawn with `seed`, and the messages of the box in the order they came in (the
    dataset's English ones in a seeded order, but those that the schemes asked about leave out, and at `tiny` a seeded few of the rest). A
    scheme is drawn only where what it leaves out is no message another asked scheme needs; and a `first` scheme only where its first
    message is not among the box's first (`EARLIEST`): it is drawn again, deterministically, when it is."""
    check_absent(calls, cases)
    english = [c for c in calls if c.language == "en"]
    sets = sets_of(english)
    quota = QUOTAS[scale]
    limit = TINY_MOST if scale == "tiny" else None  # at `tiny` every message of an asked scheme is in the box: a scheme of more is not asked

    def size(key: str) -> int:
        return len(sets[key][0])

    def within(low: int, high: int) -> list[str]:
        return sorted(k for k in SCHEMES if low <= size(k) <= min(high, limit or high))

    pools = {"single": [k for k in SINGLES if size(k) == 1], "scheme": within(*PLAIN_COUNT), "count": within(*PLAIN_COUNT),
             "week_scheme": within(MAX_FILES + 1, len(english)), "week_count": within(MAX_FILES + 1, len(english))}
    held_pool = [k for k in within(2, 12) if k not in SINGLES]
    for attempt in range(50):
        rng = random.Random(f"m3:{seed}:{scale}:{attempt}")
        taken: dict[str, tuple[frozenset[int], frozenset[int]]] = {}  # a scheme: (the messages it needs in the box, those it leaves out)

        def draw(role: str, pool: Sequence[str], held: bool = False) -> tuple[str, ...]:
            order = list(pool)
            rng.shuffle(order)
            out: list[str] = []
            for key in order:
                if len(out) == quota[role]:
                    break
                if key in taken:
                    continue
                strict, near = sets[key] if key in SCHEMES else (frozenset(), frozenset())
                needs, leaves = (frozenset(), strict | near) if held else (strict, near)
                if all(not (leaves & n) and not (l & needs) for n, l in taken.values()):
                    taken[key] = (needs, leaves)
                    out.append(key)
            if len(out) < quota[role]:
                raise LookupError(role)
            return tuple(out)

        try:
            roles = {role: draw(role, pool) for role, pool in pools.items()}
            roles["first"] = draw("first", [k for k in within(*FIRST_COUNT) if k not in SINGLES])
            for role in ("absent_scheme", "absent_count", "absent_first"):
                roles[role] = draw(role, sorted(ABSENT) + held_pool, held=True)
            roles["absent_single"] = draw("absent_single", sorted(ABSENT) + [k for k in SINGLES if size(k) == 1], held=True)
        except LookupError:
            continue  # the schemes drawn leave no room for another of its role: draw again
        needed = set().union(*(n for n, _ in taken.values()))
        gone = set().union(*(l for _, l in taken.values()))
        rest = [i for i in range(len(english)) if i not in gone and i not in needed]
        if scale == "tiny":
            sample = random.Random(f"m3:{seed}:{attempt}:others")
            forced = per_message(english, set(rest), sample, quota)
            rest = [*forced, *sample.sample([i for i in rest if i not in forced], TINY_CALLS - len(needed) - len(forced))]
        order = sorted({*rest, *needed})
        random.Random(f"m3:{seed}:{scale}:{attempt}:numbers").shuffle(order)
        place = {i: n for n, i in enumerate(order)}
        if all(min(place[i] for i in sets[k][0]) >= EARLIEST for k in roles["first"]):
            return Roles(**roles), [english[i] for i in order]
    raise ValueError(f"no draw of the schemes at {scale}, seed {seed} fits together, with every `first` scheme's first message after the box's first {EARLIEST}")


def clock_text(t: float) -> str:
    d = datetime.fromtimestamp(t)
    return f"{realdump.WEEKDAYS[d.weekday()]} {d.day} {realdump.MONTHS[d.month - 1]} at {d:%H:%M}"


def day_text(day: date) -> str:
    return f"{realdump.WEEKDAYS[day.weekday()]} {day.day} {realdump.MONTHS[day.month - 1]}"


def money_of(call: Call) -> float | None:
    """The one dollar amount a transcript names, or None when it names none, two, or one in words as well."""
    found = {float(m.replace(",", "")) for m in MONEY.findall(call.transcript)}
    return next(iter(found)) if len(found) == 1 and not MONEY_WORDS.search(call.transcript) else None


def percent_of(call: Call) -> float | None:
    """The one percentage a transcript names, or None when it names none, two, or a range."""
    found = {float(m) for m in PERCENTS.findall(call.transcript)}
    return next(iter(found)) if len(found) == 1 and not RANGE.search(call.transcript) else None


def caller_of(call: Call) -> str | None:
    """The key of CALLERS the transcript has as its caller, in its first CLAIMED words, when it has one only."""
    first = " ".join(call.transcript.split()[:CLAIMED])
    hits = [k for k, (_, claim) in CALLERS.items() if rx(claim).search(first)]
    return hits[0] if len(hits) == 1 else None


def questions_of(everything: Sequence[Call], kept: Sequence[Call], files: Sequence[realdump.File], roles: Roles, scale: str, seed: int,
                 now: float) -> tuple[Question, ...]:
    """The questions of the box `files` (each made from the message of `kept` at its place), over `roles`; `everything` is the dataset's
    messages (for the amounts and percentages that more than one message has); `now` is epoch seconds."""
    since, until = min(f.t for f in files), max(f.t + f.length for f in files)
    rng = random.Random(f"m3:{seed}:{scale}:questions")
    quota = QUOTAS[scale]
    english = [c for c in everything if c.language == "en"]
    where = {c.file: i for i, c in enumerate(kept)}
    sets, whole = sets_of(kept), sets_of(english)  # the messages that have each scheme, in the box and in the dataset
    absent = {k for role in (roles.absent_scheme, roles.absent_count, roles.absent_first, roles.absent_single) for k in role}
    for key in roles.asked():
        if key in absent and key in SCHEMES and (sets[key][0] or sets[key][1]):
            raise ValueError(f"{key} is asked about as absent, and the box holds a message of it")
        if key in absent and key in ABSENT and any(rx(ABSENT[key].strict).search(c.transcript) for c in kept):
            raise ValueError(f"{key} is asked about as absent, and the box holds a message of it")
        if key not in absent and (len(whole[key][0]) != len(sets[key][0]) or sets[key][1]):
            raise ValueError(f"{key} is asked about, and the box does not hold all of its messages and none of those taken for it")

    def ask(qid: str, text: str, template: str, shape: str, grade, found: Sequence[int], value: str | None, absent: bool = False) -> Question:
        return realdump.question(code=CODE, qid=("absent-" if absent else "") + qid, now=now, text=text, kind="absent" if absent else template,
                                 template="absent_" + template if absent else template, modality="audio", shape=shape, grade=grade,
                                 evidence=[Span(files[n].path, 0.0, files[n].length) for n in found], sources=(SOURCE_NAME,), since=since,
                                 until=until, value=value)

    def name_of(n: int) -> str:
        return files[n].path.rsplit("/", 1)[-1]

    def weeks() -> list[tuple[date, date]]:
        first = datetime.fromtimestamp(files[0].t).date()
        monday = first - timedelta(days=first.weekday())
        last = datetime.fromtimestamp(files[-1].t).date()
        return [(monday + timedelta(days=7 * n), monday + timedelta(days=7 * n + 6)) for n in range((last - monday).days // 7 + 1)]

    def in_week(found: Sequence[int], week: tuple[date, date]) -> list[int]:
        return [n for n in found if week[0] <= datetime.fromtimestamp(files[n].t).date() <= week[1]]

    out: list[Question] = []
    # scheme and count: the messages that have a scheme, over the box or in a week
    for key in roles.scheme:
        found = sorted(sets[key][0])
        names = [name_of(n) for n in found]
        out.append(ask(f"scheme-{key}", f"Find every voicemail that mentions {SCHEMES[key].said}.", "scheme", FILES,
                       realdump.grade("set", items=names), found, ", ".join(names)))
    for key in roles.count:
        found = sorted(sets[key][0])
        out.append(ask(f"count-{key}", f"How many voicemails mention {SCHEMES[key].said}?", "count", NUMBER,
                       realdump.grade("count", truth=str(len(found))), found, str(len(found))))
    for template, role, shape in (("scheme", roles.week_scheme, FILES), ("count", roles.week_count, NUMBER)):
        for key in role:
            options = [w for w in weeks() if WEEK_FILES[0] <= len(in_week(sorted(sets[key][0]), w)) <= WEEK_FILES[1]]
            if not options:
                raise ValueError(f"no week has {WEEK_FILES[0]} to {WEEK_FILES[1]} messages of {key}")
            week = rng.choice(options)
            found = in_week(sorted(sets[key][0]), week)
            span = f"between {day_text(week[0])} and {day_text(week[1])}"
            if template == "scheme":
                names = [name_of(n) for n in found]
                out.append(ask(f"scheme-{key}-week", f"Find every voicemail that mentions {SCHEMES[key].said} {span}.", "scheme", shape,
                               realdump.grade("set", items=names), found, ", ".join(names)))
            else:
                out.append(ask(f"count-{key}-week", f"How many voicemails {span} mention {SCHEMES[key].said}?", "count", shape,
                               realdump.grade("count", truth=str(len(found))), found, str(len(found))))
    # first: the first message that has a scheme, not among the box's first
    for key in roles.first:
        found = sorted(sets[key][0])
        n = found[0]
        if n < EARLIEST:
            raise ValueError(f"the first message of {key} is the box's number {n + 1}: the list of the files would give its time")
        when = datetime.fromtimestamp(files[n].t)
        out.append(ask(f"first-{key}", f"Did anyone leave me a voicemail that mentions {SCHEMES[key].said}? If so, when was the first one?",
                       "first", WHEN, realdump.grade("time", time=when.strftime("%H:%M:%S"), tol_s=TOL_S, date=when.date().isoformat()), [n],
                       when.strftime("%Y-%m-%d %H:%M")))
    # single: the one message that has a name
    for key in roles.single:
        (n,) = sorted(sets[key][0])
        out.append(ask(f"single-{key}", f"Find the voicemail that mentions {SCHEMES[key].said}.", "single", FILE,
                       realdump.grade("set", items=[name_of(n)]), [n], name_of(n)))
    # the absent ones: a name no message has
    for template, role, text in (("scheme", roles.absent_scheme, "Find every voicemail that mentions {}."),
                                 ("count", roles.absent_count, "How many voicemails mention {}?"),
                                 ("first", roles.absent_first, "Did anyone leave me a voicemail that mentions {}? If so, when was the first one?"),
                                 ("single", roles.absent_single, "Find the voicemail that mentions {}.")):
        for key in role:
            out.append(ask(f"{template}-{key}", text.format(table(key).said), template,
                           {"scheme": FILES, "count": NUMBER, "first": WHEN, "single": FILE}[template], realdump.grade("absent"), [], None,
                           absent=True))
    # per message: named by the minute it came in
    amounts, percents, callers = classes_of(english)
    by_caller = {k: [where[english[i].file] for i in ix if english[i].file in where] for k, ix in callers.items()}
    by_caller = {k: ns for k, ns in by_caller.items() if ns}
    for template, role_n, pools, text, shape, unit in (
            ("price", quota["price"], {v: [where[english[i].file] for i in ix if english[i].file in where] for v, ix in amounts.items()},
             "What dollar amount did the voicemail I got on {} mention?", DOLLARS, 1.0),
            ("percent", quota["percent"], {v: [where[english[i].file] for i in ix if english[i].file in where] for v, ix in percents.items()},
             "What percentage did the voicemail I got on {} mention?", PERCENT, 0.0)):
        classes = sorted(v for v, ns in pools.items() if ns)
        rng.shuffle(classes)
        for v in classes[:role_n]:
            n = rng.choice(pools[v])
            clean = f"{v:g}" if v == int(v) else f"{v:.2f}"
            out.append(ask(f"{template}-{n + 1:04d}", text.format(clock_text(files[n].t)), template, shape,
                           realdump.grade("amount", truth=clean, tol_s=unit), [n], clean))
        if len(classes) < role_n:
            raise ValueError(f"{template}: only {len(classes)} of the {role_n} amounts it asks about are in the box")
    classes = sorted(by_caller)
    rng.shuffle(classes)
    if len(classes) < quota["caller"]:
        raise ValueError(f"caller: only {len(classes)} of the {quota['caller']} callers it asks about are in the box")
    for k in classes[:quota["caller"]]:
        n = rng.choice(by_caller[k])
        accepted, _ = CALLERS[k]
        wrong = [w for o, (a, _) in CALLERS.items() if o != k for w in a.split("|") if w not in accepted.split("|")]
        out.append(ask(f"caller-{n + 1:04d}", f"Who did the voicemail I got on {clock_text(files[n].t)} say it was from?", "caller", COMPANY,
                       realdump.grade("text", truth=accepted, wrong=wrong), [n], accepted.split("|")[0]))
    return realdump.split(out, seed)
