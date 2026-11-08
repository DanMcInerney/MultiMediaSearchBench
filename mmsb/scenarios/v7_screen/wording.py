"""The words of the V7 week: what the questions say (`WORDS`, `WEEK_WORDS`, `ORACLES`), the notes' and log lines' text, and the small parts of
a question that are made from them (`grade_of`, `shade_of`, `apart`). Moved from haystack.py of the repository at bb498a6."""

from __future__ import annotations

import random
from collections.abc import Sequence

from mmsb.core import grade
from mmsb.scenarios.v7_screen.media_common import NEEDLE_RGB

WORDS = {"note-code": "What is the {thing} that I noted {hint}?",
         "config-asof": "What was retention_days set to in app.toml {hint}?",
         "log-count": "How many times did the payments service log {code} {hint}?",
         "log-last": "At what time did the last deploy of the web service finish {hint}?",
         "photo-sign": "What number was on the green sign that I took a photo of {hint}?",
         "photo-count": "How many photos of a red car did I take {hint}?",
         "screen-error": "What was the error code on the red error screen {hint}, and at what time did it first appear?",
         "porch-parcel": "At what time was a parcel left at the front door {hint}?",
         "mic-doorbell": "At what time did the doorbell ring {hint}?",
         "mic-alarm": "At what time did the smoke alarm beep {hint}?"}
WEEK_WORDS = {  # no hint: the prompt names the span. A time answer must name the day too (`grade.date`)
    "photo-sign": "What number was on the last green sign that I took a photo of?",
    "photo-count": "How many photos of a red car did I take?",
    "screen-error": "What was the error code on the red error screen the last time one appeared, and on what day and at what time "
                    "did it first appear?",
    "porch-parcel": "On what day and at what time was a parcel last left at the front door?",
    "mic-alarm": "On what day and at what time did the smoke alarm last beep?"}
ORACLES = {"note-code": "Does the text mention {topic}?", "config-asof": None,
           "log-count": "Does a record have the error code {code}?",
           "log-last": "Does a record say that a deploy of the web service finished?",
           "photo-sign": "Is there a green sign in the picture?", "photo-count": "Is there a red car in the picture?",
           "screen-error": "Is an error dialog shown on the screen?", "porch-parcel": "Is there a parcel on the step by the door?",
           "mic-doorbell": "Is a doorbell ringing?", "mic-alarm": "Is a smoke alarm beeping?"}
TOPICS = (("code of the spare key box", "the spare key box", "Spare key box: it now opens with {code}.", 4),
          ("combination of my bike lock", "a bike lock", "Bike lock: the new combination is {code}.", 4),
          ("code of my gym locker", "a gym locker", "Gym locker: the code is {code} this month.", 4),
          ("PIN of the parcel locker", "a parcel locker", "Parcel locker: pick up with PIN {code}.", 6))
ERRORS = ("disk full", "out of memory", "connection lost", "license expired", "checksum mismatch", "database locked",
          "backup failed")
WARNINGS = ("low battery", "update available", "slow network", "certificate expires soon", "high temperature")
SIGN_WORDS = ("GATE", "PLATFORM", "ROOM", "BAY", "LEVEL")
OTHER_SIGNS = ("yellow", "white", "blue", "grey")
OTHER_CARS = ("blue", "white", "grey", "yellow")


def grade_of(kind: str, truth: str | None = None, time: str | None = None, tol_min: float | None = None,
             wrong: Sequence[str] = (), ignore: Sequence[str] = (), date_: str | None = None) -> dict:
    return {"kind": kind, "truth": truth, "time": time, "tol_min": tol_min, "wrong": list(wrong), "ignore": list(ignore),
            "date": date_}


def shade_of(kind: str) -> str:
    """The fake decider's question for a needle's shade (`shade:R,G,B`)."""
    return "shade:" + ",".join(str(c) for c in NEEDLE_RGB[kind])


def apart(clocks: Sequence[str], truth: str, minutes: float = 10) -> list[str]:
    """The clock times of `clocks` that are not within `minutes` of `truth`: the earlier needles of a week question that an answer
    must not give, with the true time's own neighbours left out (a right answer is within the tolerance of it)."""
    return [c for c in clocks if abs(grade.at(c) - grade.at(truth)) > minutes]


REASONS = ("gateway timeout", "card declined", "insufficient funds", "fraud check failed", "issuer unavailable")
PATHS = ("orders", "users", "items", "cart", "health", "search", "checkout")
NAMES = ("ana", "ben", "chloe", "dev", "eli", "fay", "gus", "hana", "ivan", "june")


def background_line(r: random.Random, service: str) -> tuple[str, str, str | None]:
    x, order = r.random(), r.randrange(10000, 99999)
    if service == "payments":
        if x < 0.82:
            return "info", f"payment ok order={order} amount={r.randrange(5, 500)}.{r.randrange(100):02d} EUR", None
        if x < 0.92:
            return "warning", f"card retry order={order} attempt={r.randrange(1, 4)}", None
        return "error", f"{r.choice(REASONS)} order={order}", f"ERR-{r.randrange(100, 700)}"
    path = f"/api/{r.choice(PATHS)}"
    if service == "web":
        if x < 0.88:
            return "info", f"request ok path={path} {r.choice((200, 200, 200, 304))} in {r.randrange(5, 650)} ms", None
        if x < 0.96:
            return "warning", f"slow request path={path} in {r.randrange(400, 650)} ms", None
        return "error", f"upstream error path={path} 502", f"ERR-{r.randrange(100, 700)}"
    user = r.choice(NAMES)
    if x < 0.85:
        return "info", r.choice((f"login ok user={user}", f"token refreshed user={user}")), None
    if x < 0.96:
        return "warning", f"login failed user={user} reason=bad credentials", None
    return "error", f"account locked user={user}", f"ERR-{r.randrange(100, 700)}"


GROCERIES = ("milk", "bread", "eggs", "apples", "rice", "pasta", "coffee", "tomatoes", "cheese", "oranges", "oats", "butter")
PEOPLE = ("Priya", "Marcus", "Elena", "Tom", "Sofia", "Daniel", "Aiko", "Omar")
BOOKS = (("The Left Hand of Darkness", "Le Guin"), ("Piranesi", "Clarke"), ("Station Eleven", "Mandel"),
         ("The Dispossessed", "Le Guin"), ("Klara and the Sun", "Ishiguro"))
IDEAS = ("a small script that renames photos by date", "a shelf above the desk", "a weekly review on Sunday evening",
         "moving the plants to the window", "a list of the trips for next year", "a plan for the garden beds")


def distractor_note(r: random.Random) -> str:
    """A note of the owner's life, with small numbers only (none is a needle's code) and none of a needle's topics."""
    kind = r.choice(("groceries", "meeting", "reminder", "book", "idea"))
    if kind == "groceries":
        a, b, c = r.sample(GROCERIES, 3)
        return f"Groceries: {a}, {b} and {c}. Get {r.randrange(2, 9)} of the first."
    if kind == "meeting":
        return f"Meeting with {r.choice(PEOPLE)} at {r.randrange(9, 18)}:{r.choice(('00', '15', '30', '45'))}, room {r.randrange(2, 30)}."
    if kind == "reminder":
        return f"Reminder: call {r.choice(PEOPLE)} about the {r.choice(('invoice', 'holiday', 'dentist', 'garden'))}, before {r.randrange(10, 29)}."
    if kind == "book":
        title, author = r.choice(BOOKS)
        return f"Book: {title} by {author}, now on page {r.randrange(20, 400)}."
    return f"Idea: {r.choice(IDEAS)}."
