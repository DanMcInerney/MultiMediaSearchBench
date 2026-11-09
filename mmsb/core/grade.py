"""The graders of the agents' answers: pure rules, no model (moved from grade.py of the MultimediaSearch repo, fe0ece5). An answer is
casefolded, number words become digits, times become minutes after midnight. `grade(g, answer)` grades one answer by a
question's `Grade`: its rule, truth, time and tolerance in seconds, wrong answers to refuse, `ignore` (the numbers and clock times
the question itself holds, which an answer may repeat) and `date` (the day a time must be on, or None). An answer that hedges (a
second number, time or date the question did not ask) is wrong."""

from __future__ import annotations

import re
from datetime import date

from mmsb.core.question import ISO_DAY, RULES, Grade

UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen " \
        "eighteen nineteen".split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ONCE = {"once": "1", "twice": "2", "thrice": "3"}
# a name with digits in it, which is no number: one led by letters (cam3, IMG_0003.JPG, cam3_20180312_100000-103500.mp4), or a
# path that has a letter in it (DCIM/100RECNX)
NAMED = re.compile(r"\b[^\W\d]+\d[\w.-]*|(?<!\S)(?=\S*[^\W\d_])\S*[\\/]\S*", re.IGNORECASE)
WORD = re.compile(r"\b(?:(" + "|".join(TENS) + r")(?:[ -](one|two|three|four|five|six|seven|eight|nine))?|("
                  + "|".join(UNITS) + r"))\b")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([ap])\.?m\b\.?)?")
MONTH = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul"
         r"|aug|sept|sep|oct|nov|dec)\b\.?")
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}[t ](\d{1,2}:\d{2}(?::\d{2})?)(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?(?!\d)")
DATES = (re.compile(rf"\b(?:{DAY}\s+(?:of\s+)?{MONTH}|{MONTH}\s+{DAY})(?:,?\s+\d{{4}})?(?!\d)"),
         re.compile(rf"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)"
                    rf"\.?,?\s+{DAY}(?!\d)"),
         re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
         re.compile(rf"\b{MONTH}\s+(?:of\s+)?\d{{4}}(?!\d)"))  # a month of a year, "October 2013"
WEEKDAYS = "mon tue wed thu fri sat sun".split()
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
# a date an answer may name, as `date_mentions` reads it; a clock time ("21:04") is never the day of a month or a year
NOT_CLOCK = r"(?![\d:])"
ISO_DATE = re.compile(rf"(?<!\d)(\d{{4}})-(\d{{2}})-(\d{{2}}){NOT_CLOCK}")
DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH})(?:,?\s+(\d{{4}}))?{NOT_CLOCK}")
MONTH_DAY = re.compile(rf"\b({MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?{NOT_CLOCK}")
# a weekday, with the day of the month after it when it has one ("Thursday 29", "Thu the 29th"); mon, wed, sat and sun are
# words too, so they count only before a full stop, a comma or a number
WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tues?|thu(?:rs?)?|fri"
                     r"|(?:mon|wed|sat|sun)(?=[.,]|\s+\d))\b\.?"
                     rf"(?:,?\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?{NOT_CLOCK}(?!\s*[ap]\.?m\b))?")
# "on the 29th", "on 29th", "the 29th at 21:04": a day of the month by its ordinal, when a date and not "the 2nd alarm" (the
# ordinal ends the sentence, or a comma, a bracket, "at" or a time follows it)
BARE_DAY = re.compile(r"\b(?:on\s+(?:the\s+)?|the\s+)(\d{1,2})(?:st|nd|rd|th)\b"
                      r"(?=\s*(?:$|[,.;:()!?]|(?:at|around|about|@)\b|\d{1,2}:\d{2}))")
# a year in brackets, "(2026)", which is no number of an answer (unless it is the answer: a code can look like a year)
YEAR = re.compile(r"\(\s*(?:in\s+)?((?:19|20)\d{2})\s*\)")
# an abstaining or negating word (and a lone digit 0, which is not the start of 0.5 or 0:30)
NEGATION = re.compile(r"\b(?:unknown|none|never|nothing|nobody|nowhere|zero|no|not|nor|cannot|unable)\b|n/a|\b\w+n't\b"
                      r"|(?<![\w.:])0(?![\w:]|[.,]\d)")
# the time range an answer to an absent question may say it searched: "No, nothing (I checked 9:00-17:00)", "None found from
# 9:00 to 17:00", "I looked between 9:00 and 17:00 and saw nothing". `searched` takes it out of the answer's times only when it is
# SEARCHED_MIN minutes long or more (a sighting is a moment: "No; I checked 10:05-10:06" is not an abstention), and the rest of
# the answer is negative or says what was searched (each clause holds a negation, or only NEUTRAL words), with no "yes", no word
# of CONTRAST and none of PRESENT (a thing that went on: "it never closed")
CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\b\.?)?"
RANGE = re.compile(rf"\bbetween\s+({CLOCK})\s+and\s+({CLOCK})|\b({CLOCK})\s*(?:-|–|—|to|until|till|through|thru)\s*({CLOCK})")
NEUTRAL = re.compile(r"\b(?:i|we|my|the|a|an|from|between|in|over|during|within|for|at|on|of|to|through|all|whole|entire|full|"
                     r"footage|recording|recordings|camera|range|window|period|today|also|again|carefully|thoroughly|went|"
                     r"(?:check|search|look|scann?|cover|review|watch|listen|inspect|examin|scour)\w*)\b")
CONTRAST = re.compile(r"\b(?:but|however|although|though|except|yet|apart|besides|aside|other than)\b")
PRESENT = re.compile(r"\byes\b|\b(?:still|remain\w*|stay\w*|kept|continu\w*|stopp\w*|clos\w*|whole time|entire time|all along|"
                     r"all the time|throughout|nonstop)\b")
SEARCHED_MIN = 60
# an option letter at the start of an answer ("B", "(b)", "B) Alice", "B - Alice"): then its end, a space or one of `.):,-`
CHOICE = re.compile(r"\(?([A-D](?=$|[\s.):,-])|[a-d](?=$|[.):,-]))")  # a lower-case letter alone or before a mark: "a screwdriver" picks none
WRAPS = ("<>", "[]", '""', "''", "``")  # what an answer may be wrapped in whole: `<B>`, `[B]`, `"B"`
# a file name an answer may give for a `set` of files: a name, a dot and an extension that starts with a letter ("IMG_0042.JPG",
# "clip-0007.wav"; not "1.25"); only the extensions of the set's own files count
FILE_NAME = re.compile(r"[\w][\w.-]*\.([a-z][a-z0-9]{1,3})\b", re.IGNORECASE)
# a second letter named after "or", "and", "/", "&" or a comma: "B or C", "B/C", "B, C" are hedges
SAID_WORDS = 25  # the longest answer the rule `text` takes: a short value, not a list of guesses
HEDGE_WORD = re.compile(r" (?:or|maybe|perhaps|possibly) ")  # in `words_of`'s form: a second guess
TWO_WAY = 2  # at most this many `wrong` phrases are the other side of a two-way question ("before or after")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # "1,250.00"
DOT_THOUSANDS = re.compile(r"(?<=\d)\.(?=\d{3}(?!\d))")  # "67.000" of a whole amount of 67000
CURRENCY = re.compile(r"\b(?:rm|rp|usd|eur|gbp|myr|idr)(?=\d)")  # "rm91.50" is an amount, not a name as `cam3` is
HEDGE = re.compile(r"(?i:\bor\b|\band\b)\s*\(?[A-D]\b(?!')|[/&,]\s*\(?[A-D]\b(?!')")


def norm(text: str) -> str:
    """Casefolded, number words as digits ("four eight two one" is "4 8 2 1", "thirty" is "30")."""
    def number(m: re.Match) -> str:
        if m.group(3):
            return str(UNITS.index(m.group(3)))
        return str(TENS[m.group(1)] + (UNITS.index(m.group(2)) if m.group(2) else 0))
    return WORD.sub(number, text.casefold())


def clocks(text: str) -> str:
    """`norm(text)` with each ISO date and time left as its time alone: `2026-10-28T14:03:10-04:00` is ` 14:03:10 `."""
    return ISO.sub(r" \1 ", norm(text))


def mentions(text: str) -> list[list[float]]:
    """Each time of day in `text` as the minutes after midnight it may be: "2:03" with no am or pm is both 2:03 and 14:03. An
    ISO time (`2026-10-28T14:03:10-04:00`) is read as written: 14:03."""
    out = []
    for m in TIME.finditer(clocks(text)):
        hour, minute, second, half = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), m.group(4)
        if minute > 59:
            continue
        if half:
            hours = [hour % 12 + (12 if half == "p" else 0)]
        else:
            hours = [hour, hour + 12] if hour < 12 else [hour]
        out.append([h * 60 + minute + second / 60 for h in hours])
    return out


def minutes(text: str) -> list[float]:
    """Every time of day in `text` as minutes after midnight (see `mentions`)."""
    return [t for one in mentions(text) for t in one]


def digits_in(plain: str, keep=()) -> list[str]:
    """The numbers written in digits in `plain` (already casefolded; times and dates left out: `28 October`, `October 28`,
    `Tuesday 28`, `on the 28th`, `2026-10-28`, and a year in brackets, `(2026)`, unless it is one of `keep`), a run of single
    digits being one number: "4 8 2 1" is 4821."""
    plain = ISO.sub(r" \1 ", plain)
    for dates in DATES:
        plain = dates.sub(" ", plain)
    plain = BARE_DAY.sub(" ", plain)
    plain = YEAR.sub(lambda m: m.group(0) if m.group(1) in keep else " ", plain)
    plain = TIME.sub(" ", plain)
    plain = re.sub(r"\b\d(?: \d)+\b", lambda m: m.group(0).replace(" ", ""), plain)
    return re.findall(r"\d+(?:\.\d+)?", plain)


def numbers(text: str, keep=()) -> list[str]:
    """The numbers in `text`, number words as digits (see `norm`, `digits_in`)."""
    return digits_in(norm(text), keep)


def at(clock: str) -> float:
    """Minutes after midnight of "12:07" or "10:00:35"."""
    h, m, *s = (int(p) for p in clock.split(":"))
    return h * 60 + m + (s[0] / 60 if s else 0)


def near(text: str, clock: str, minutes_: float) -> bool:
    """Whether `text` has a time within `minutes_` of `clock` ("12:07" or "10:00:35")."""
    truth = at(clock)
    return any(abs(t - truth) <= minutes_ for t in minutes(text))


def unknown(text: str) -> bool:
    return re.match(r"\W*unknown\b", norm(text)) is not None


def code(text: str, n: str) -> bool:
    return re.search(rf"\b(?:err[- ]?)?{n}\b", norm(text)) is not None


def asked(question: str) -> list[str]:
    """The numbers and the clock times ("HH:MM") a question holds itself, for a grade's `ignore`: an answer may repeat them
    ("ERR-731", the hours of "in the evening (18:00 to 24:00)"; dates and the digits of a name such as `cam3` are not numbers)."""
    out = list(dict.fromkeys(numbers(NAMED.sub(" ", question))))
    for m in TIME.finditer(clocks(question)):
        if int(m.group(2)) < 60:
            out.append(f"{int(m.group(1)):02d}:{m.group(2)}")
    return list(dict.fromkeys(out))


def split(ignore) -> tuple[set[str], list[float]]:
    """An `ignore` list as its numbers and its clock times (minutes after midnight)."""
    return {x for x in ignore if ":" not in x}, [at(x) for x in ignore if ":" in x]


def timed(answer: str, ignore_times: list[float], truth: float | None = None, tol: float = 0.0) -> bool:
    """Whether every time of day in `answer` is within `tol` minutes of `truth`, or one of `ignore_times` (to 30 s)."""
    return all(any((truth is not None and abs(t - truth) <= tol) or any(abs(t - i) <= 0.5 for i in ignore_times) for t in one)
               for one in mentions(answer))


def span(start: list[float], end: list[float]) -> float | None:
    """The shortest time from a reading of `start` to a later reading of `end` (each as `mentions` gives it: "10:05" with no am
    or pm is 10:05 and 22:05), or None when no reading of `end` is after one of `start`. Two times with no am or pm are read
    in the same half of the day: "10:05-10:06" is a minute, never twelve hours; "8:00 to 20:00" is twelve, not none."""
    pairs = zip(start, end) if len(start) == len(end) == 2 else ((a, b) for a in start for b in end)
    gaps = [b - a for a, b in pairs if b > a]
    return min(gaps) if gaps else None


def searched(plain: str) -> str:
    """`plain` with the time ranges it says it searched taken out, so that they are no times of a sighting. A range counts
    when it is SEARCHED_MIN minutes long or more (`span`), and the whole answer is one that abstains: no "yes", no word of
    CONTRAST or PRESENT, and every clause holds a negation or says only what was searched (NEUTRAL words), as in "no, nothing (I
    checked 9:00-17:00)" and "none found from 9:00 to 17:00". Any other range, and any single time, stays."""
    plain = ISO.sub(r" \1 ", plain)  # an ISO time's offset ("14:03:10-04:00") is no range
    if CONTRAST.search(plain) or PRESENT.search(plain):
        return plain

    def blank(m: re.Match) -> str:
        bounds = [mentions(t) for t in (m[1] or m[3], m[2] or m[4])]  # a bound that is no time, such as 12:75, has none
        gap = span(bounds[0][0], bounds[1][0]) if all(bounds) else None
        return " " if gap is not None and gap >= SEARCHED_MIN else m.group(0)

    blanked = RANGE.sub(blank, plain)
    clauses = (c.strip() for c in re.split(r"[,;.:!?()\n]|\band\b", blanked))
    clean = all(NEGATION.search(c) or not NEUTRAL.sub("", c).strip() for c in clauses if c)
    return blanked if clean else plain


def abstained(text: str | None, ignore=()) -> bool:
    """Whether an answer says the thing is not there: it holds a word of abstaining or negation (`unknown`, `none`, `never`,
    `nothing`, `nobody`, `no`, `not` (so `did not`, `could not`), `cannot`, `unable`, `n/a`, `zero`, `didn't` and the like), and
    every number in digits (but a name's, as `cam3`'s) and every time in it is in `ignore` (the question's own: "Not sure, maybe 47" is a guess; a number
    word such as "one" is not a number here: "No, not a single one" abstains), or is the bound of a range it says it searched
    in an answer that is negative throughout (`searched`)."""
    if text is None:
        return False
    plain = searched(text.casefold().replace("’", "'"))
    digits, times = split(ignore)
    return (NEGATION.search(plain) is not None and all(n == "0" or n in digits for n in digits_in(NAMED.sub(" ", plain)))
            and timed(plain, times))


def choice(answer: str | None) -> str | None:
    """The option an answer picks: the letter A to D it starts with (any case, in brackets or not, then its end, a space or
    one of `.):,-`), upper case; None for any other start ("Alice", "The answer is B") and for a hedge, an answer that
    names a second letter after the first ("B or C"). One pair of `<>`, `[]`, quotes or backticks around the whole answer
    (`<B>`, `"B) Alice"`) is taken off first."""
    if answer is None:
        return None
    text = answer.strip()
    if len(text) > 2 and text[0] + text[-1] in WRAPS:
        text = text[1:-1].strip()
    found = CHOICE.match(text)
    return None if found is None or HEDGE.search(text[found.end():]) else found[1].upper()


def date_mentions(text: str) -> list[tuple]:
    """Each date `text` names, as `(year, month, day, weekday)` with None for what it does not say (the weekday is 0 for
    Monday): `2026-10-29`, `29 October` and `October 29` with or without a year, `Thursday` (or `Thu`) alone or with its
    day of the month (`Thursday 29`), or the day of the month by its ordinal alone (`on the 29th`)."""
    plain, found = norm(text), []
    for pattern, read in ((ISO_DATE, lambda m: (int(m[1]), int(m[2]), int(m[3]), None)),
                          (DAY_MONTH, lambda m: (int(m[3]) if m[3] else None, MONTHS.index(m[2][:3]) + 1, int(m[1]), None)),
                          (MONTH_DAY, lambda m: (int(m[3]) if m[3] else None, MONTHS.index(m[1][:3]) + 1, int(m[2]), None)),
                          (WEEKDAY, lambda m: (None, None, int(m[2]) if m[2] else None, WEEKDAYS.index(m[1][:3]))),
                          (BARE_DAY, lambda m: (None, None, int(m[1]), None))):
        found += [read(m) for m in pattern.finditer(plain)]
        plain = pattern.sub(" ", plain)  # a date is read once: "Thursday 29 October" is a weekday and a day and month only
    return found


def on_date(text: str, day: str) -> bool:
    """Whether `text` names the date `day` (ISO) and no other: every date in it agrees with `day` on what it says."""
    truth = date.fromisoformat(day)
    wanted = (truth.year, truth.month, truth.day, truth.weekday())
    named = date_mentions(text)
    return bool(named) and all(all(m is None or m == w for m, w in zip(one, wanted)) for one in named)


def base_name(path: str) -> str:
    """The last part of a path, with either slash, in lower case: a `set` of files is matched by name, wherever the agent saw it."""
    return re.split(r"[\\/]", path)[-1].casefold()


def named_items(g: Grade, answer: str) -> tuple[set[str], int]:
    """The items of a `set` grade that `answer` names, and how many other things of their kind it names. Dates: each date the answer
    names by its month and day (`date_mentions`; a year, when given, must agree; a weekday alone names no date; a day of the month
    named alone, as "the 4th", is another date unless it is the day of an item), with the dates of `g.ignore` (the question's own
    bounds) passed over unless they are items. Files: each file name with an extension of the items'
    (`FILE_NAME`), matched by name (`base_name`), each counted once."""
    if all(ISO_DAY.fullmatch(x) for x in g.items):
        wanted = {date.fromisoformat(x): x for x in g.items}
        ignored = {date.fromisoformat(x) for x in g.ignore if ISO_DAY.fullmatch(x)}
        found: set[str] = set()
        others: set[tuple] = set()
        days = {d.day for d in wanted}
        for year, month, day, _ in date_mentions(answer):
            if month is None and day is not None and day not in days:  # "and maybe the 4th": a day named alone that is none of them
                others.add((None, None, day))
            if month is None or day is None:
                continue
            hits = [x for d, x in wanted.items() if (d.month, d.day) == (month, day) and year in (None, d.year)]
            if hits:
                found.update(hits)
            elif not any((d.month, d.day) == (month, day) and year in (None, d.year) for d in ignored):
                others.add((year, month, day))
        return found, len(others)
    wanted_names = {base_name(x): x for x in g.items}
    suffixes = {name.rsplit(".", 1)[-1] for name in wanted_names}
    named = {m.group(0).casefold() for m in FILE_NAME.finditer(answer) if m.group(1).casefold() in suffixes}
    return {wanted_names[n] for n in named if n in wanted_names}, len(named - set(wanted_names))


def items_f1(g: Grade, answer: str | None) -> float | None:
    """For a `set` grade, the F1 of the items the answer names against the grade's (`named_items`): 0 for no answer; None for another
    rule. It is told beside the verdict, which needs the exact set."""
    if g.rule != "set":
        return None
    if answer is None:
        return 0.0
    found, others = named_items(g, answer)
    precision = len(found) / (len(found) + others) if found or others else 0.0
    recall = len(found) / len(g.items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def words_of(text: str) -> str:
    """`norm(text)` as its words between single spaces, with a space at each end: a phrase is found in it as whole words."""
    return " " + " ".join(re.findall(r"[^\W_]+", norm(text))) + " "


def said(g: Grade, answer: str) -> bool:
    """Whether the short `answer` holds one of the phrases of `g.truth` (separated by `|`) as whole words, offers no second guess and names none of
    `g.wrong`. A second guess is a word of `HEDGE_WORD` or, when the truth is a number, another number (the question's own left out). A `wrong` of at most
    `TWO_WAY` phrases is the other side of a two-way question, which an explanation may name: there the side named first is the answer."""
    plain = words_of(answer)
    phrases = [words_of(x) for x in g.truth.split("|")]
    found = [at for p in phrases if (at := plain.find(p)) >= 0]
    rest = plain
    for phrase in phrases:
        rest = rest.replace(phrase, " ")  # an accepted phrase may hold "or"
    if len(plain.split()) > SAID_WORDS or not found or HEDGE_WORD.search(rest):
        return False
    allowed = {w for p in phrases for w in p.split() if w.isdigit()}
    if allowed and any(w.isdigit() and w not in allowed | set(split(g.ignore)[0]) for w in plain.split()):
        return False
    against = [at for x in g.wrong if (at := plain.find(words_of(x))) >= 0]
    return not against or (len(g.wrong) <= TWO_WAY and min(found) < min(against))


def amount(g: Grade, answer: str) -> bool:
    """Whether the one number in `answer` (the question's own left out) is within `g.tol_s` of `g.truth`."""
    digits, _ = split(g.ignore)
    plain = THOUSANDS.sub("", NAMED.sub(" ", CURRENCY.sub(" ", norm(answer))))
    if "." not in g.truth and float(g.truth) >= 1000:
        plain = DOT_THOUSANDS.sub("", plain)
    found = [float(n) for n in digits_in(plain) if n == g.truth or n not in digits]
    return len(found) == 1 and abs(found[0] - float(g.truth)) <= g.tol_s + 1e-9


def grade(g: Grade, answer: str | None) -> bool:
    """Whether `answer` is right by `g`: None is wrong, and so is a hedge, an answer with a number or a time that is neither the
    truth nor in `g.ignore` (the question's own numbers and times).
    - `number`: the truth is among the answer's numbers, no other number is, and no wrong answer is;
    - `count`: the answer's numbers, with the `ignore` ones other than the truth taken out (an error code's digits) and names such
      as `cam3` or `IMG_0003.JPG` left out, are exactly the truth (`once`, `twice` and `thrice` are 1, 2 and 3);
    - `time`: a time of the answer is within `tol_s` of `time`, every time of it is, or is ignored, and none is a wrong time;
    - `code_time`: the truth's digits (`E-4417`, `e4417` and `4417` all pass), a time within `tol_s` of `time` (every time of
      the answer within it or ignored), and no other number of as many digits;
    - `absent`: the answer abstains (`abstained`);
    - `choice`: the answer's letter (`choice`) is the truth, one of A to D ("unknown", an option's words and a hedge are wrong);
    - `set`: the answer names every one of `items` (dates or file names: `named_items`) and nothing else of their kind;
    - `text`: a short answer holds one of the accepted phrases of the truth, offers no second guess and names no wrong one (`said`);
    - `amount`: the answer's one number (`amount`) is within `tol_s` of the truth, in the units the question asks in.
    A `time` or `code_time` grade may hold a `date` (ISO; None or empty is none): the answer must also name that date, by its
    weekday, as "29 October", "October 29" or "29 Oct" (with a year or not), as "the 29th" or as `2026-10-29`, and name no other
    date: a time alone is wrong (`on_date`). A year in brackets, "(2026)", is no number of an answer (a `number` answer keeps
    one that is its truth or a wrong answer), and nor is the year of the `date` in a `code_time` answer."""
    if answer is None:
        return False
    rule = g.rule
    digits, times = split(g.ignore)
    tol = None if g.tol_s is None else g.tol_s / 60  # minutes, as `near` and `timed` take them
    if rule in ("time", "code_time") and g.date and not on_date(answer, g.date):
        return False
    if rule == "number":
        found = set(numbers(answer, keep=[g.truth, *g.wrong]))
        return g.truth in found and not set(g.wrong) & found and found <= {g.truth} | digits
    if rule == "count":
        text = re.sub(r"\b(?:once|twice|thrice)\b", lambda m: ONCE[m.group(0)], NAMED.sub(" ", norm(answer)))
        for ignored in digits - {g.truth}:
            text = re.sub(rf"\b{re.escape(ignored)}\b", " ", text)
        return set(numbers(text)) == {g.truth}
    if rule == "time":
        return (near(answer, g.time, tol) and timed(answer, times, at(g.time), tol)
                and not any(near(answer, w, tol) for w in g.wrong))
    if rule == "code_time":
        wanted = re.sub(r"\D", "", g.truth)
        year = (g.date or "")[:4]  # the year of the day it names is part of the date, not another code
        others = [n for n in numbers(answer, keep=[wanted])
                  if len(n) == len(wanted) and n not in (wanted, year) and n not in digits]
        return (re.search(rf"(?<!\d){wanted}(?!\d)", norm(answer)) is not None and not others
                and near(answer, g.time, tol) and timed(answer, times, at(g.time), tol))
    if rule == "absent":
        return abstained(answer, g.ignore)
    if rule == "choice":
        return choice(answer) == g.truth
    if rule == "set":
        found, others = named_items(g, answer)
        return len(found) == len(g.items) and not others
    if rule == "text":
        return said(g, answer)
    if rule == "amount":
        return amount(g, answer)
    raise ValueError(f"a question's grade is one of {', '.join(RULES)}, not {rule!r}")
