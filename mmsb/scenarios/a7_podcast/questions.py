"""The questions of A7, every answer from the transcripts (source.py), each asked the morning after the last episode. The templates are
speech.py's (its docstring has the table), over the terms below, with the person calling the recordings episodes and whoever talks "they":

| template              | question                                                                 | category | rule            |
| `first`               | When did they first mention the cloud chamber?                           | when     | `time`, 30 s    |
| `did`                 | Did they ever mention Wikipedia? If so, when was the first time?         | did      | `time`, 30 s    |
| `which`               | In which episode did they talk about the Navajo?                         | find     | `set` of one    |
| `all`                 | Which episodes mention the Cold War?                                     | find_all | `set`           |
| `count`               | In how many episodes did they mention social media?                      | how_many | `count`         |
| `long`                | How long, in minutes, was the episode in which they talked about fusion? | how_many | `amount`, 1 min |
| `order`               | In the episode of Wednesday 8 March 2023, did they bring up X before Y? | order    | `text`          |
| `who`                 | Who was the guest on the episode about searching for the oldest stars?   | who      | `text`          |
| `what`                | Since what year had the guest been teaching at MIT?                      | what     | `text`, `number`|
| `before`              | Which did the child ask Alexa for first, the lights off or on?           | order    | `text`          |
| `absent_first`, ...   | `first`, `did` or `which` of a term in no transcript                     | as above | `absent`        |

The guests are named teachers in a published podcast, and a `who` answer is one of their names, the full name or (where it is no common word)
the surname; its `wrong` holds the names of every other guest of the pile (not the host, whom a right answer may name). The episode is told by what the
host says it is about
(each clue is said in that episode alone: tests/test_a7.py). Not asked about: the trailer, the specials and the promos (they are not in the
pile), the line that opens every episode ("Welcome to Chalk Radio, a podcast about inspired teaching at MIT") and the OCW credit that closes it
(a phrase said in every episode has no `which`, `all` or `count`). A term that was never said is plausible for a podcast of teachers and in no
caption, even set as run-on letters."""

from __future__ import annotations

from collections.abc import Sequence

from mmsb.core.question import Grade, Question
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech, speech_files
from mmsb.scenarios.realdump.speech import Say, Talk, Term
from mmsb.scenarios.realdump.speech_files import Note

CODE = "A7"
SOURCE_NAME = "podcasts"  # the dump's source
SAY = Say(CODE, SOURCE_NAME, "audio", "they", "episode", "episodes")
QUOTAS = {"tiny": {"first": 1, "which": 1, "absent_did": 1},
          "standard": {"first": 2, "did": 2, "which": 2, "all": 1, "count": 1, "long": 1, "order": 1, "absent_first": 2, "absent_did": 2,
                       "absent_which": 2}}

TERMS = (
    Term("nuclear weapons", "nuclear weapons"), Term("cloud chamber", "the cloud chamber"), Term("ionizing radiation", "ionizing radiation"),
    Term("Creole", "Creole"), Term("poetry", "poetry"), Term("reparations", "reparations"), Term("fusion", "fusion"),
    Term("large language model", "large language models"), Term("Einstein", "Einstein"), Term("cold war", "the Cold War"),
    Term("health insurance", "health insurance"), Term("Terrascope", "Terrascope"), Term("Navajo", "the Navajo"),
    Term("flight school", "flight school"), Term("Python", "Python"), Term("mental health", "mental health"),
    Term("online learning", "online learning"), Term("MicroMasters", "MicroMasters"), Term("social media", "social media"),
    Term("climate change", "climate change"), Term("App Inventor", "App Inventor"), Term("dry ice", "dry ice"), Term("big bang", "the Big Bang"),
    Term("telescope", "telescopes"), Term("face recognition", "face recognition"), Term("pre registration", "pre-registration"),
    Term("credit card", "credit cards"), Term("fintech", "fintech"), Term("history of science", "the history of science"), Term("Zoom", "Zoom"),
    Term("Alexa", "Alexa"), Term("astronaut", "astronauts"), Term("space shuttle", "the space shuttle"), Term("sustainability", "sustainability"),
    Term("vaccine", "vaccines"), Term("YouTube", "YouTube"), Term("Legos", "Legos"), Term("tumor", "tumors"), Term("COVID", "COVID"),
    Term("blockchain", "blockchain"), Term("Nobel", "the Nobel Prize"), Term("Twitter", "Twitter"), Term("Instagram", "Instagram"),
    Term("Facebook", "Facebook"), Term("Texas", "Texas"), Term("Hawaii", "Hawaii"), Term("Harvard", "Harvard"), Term("Stanford", "Stanford"),
    Term("Berkeley", "Berkeley"), Term("Mars", "Mars"), Term("Kenya", "Kenya"),
)
NEVER = (
    Term("Wikipedia", "Wikipedia"), Term("iPhone", "the iPhone"), Term("virtual reality", "virtual reality"), Term("dark matter", "dark matter"),
    Term("string theory", "string theory"), Term("Antarctica", "Antarctica"), Term("bitcoin", "Bitcoin"), Term("neural network", "neural networks"),
)

# The guests of the pile (the full names the captions give): no `who` answer is another's. The host is not among them: an answer may name her ("Eric
# Grimson, interviewed by Sarah Hansen") and be right.
HOST = "Sarah Hansen"
GUESTS = ("Mike Short", "Michael Short", "Amah Edoh", "Gil Strang", "Gilbert Strang", "Anjali Sastry", "David Thorburn", "Chris Terman",
          "Christopher Terman", "Michel DeGraff", "Philip Greenspun", "Tina Srivastava", "Jonathan Gruber", "Jon Gruber", "Gary Gensler",
          "Anna Frebel", "Jeff Grossman", "Jeffrey Grossman", "Jeff Hoffman", "Jeffrey Hoffman", "Garnette Cadogan", "Dava Newman",
          "Brandon Leshchinskiy", "Ceasar McDowell", "Nancy Kanwisher", "Sanjay Sarma", "John Dolhun", "Sarah Hewett", "Dennis McLaughlin",
          "Curt Newton", "Catherine D'Ignazio", "Jacob Andreas", "Harini Suresh", "Daniel Jackson", "Serena Booth", "Cynthia Breazeal",
          "Gigliola Staffilani", "Robert Redwine", "Bob Redwine", "Jim Walsh", "Frank Schilbach", "Cecile Chazot", "Jessica Sandland", "Eric Grimson",
          "Haynes Miller", "Paige Bright", "Summer Morrill", "Liz Potter-Nelson", "Sarah Meyers", "Anne White", "David Hsu", "David Kaiser",
          "Esther Duflo", "Sara Ellison", "Ari Epstein", "Hal Abelson", "Chris Capozzola", "Christopher Capozzola", "Rebecca Saxe",
          "Joshua Bennett", "Andrew Lo", "Ana Bell", "Felice Frankel", "Mary Rowe", "Elizabeth Siler", "Vivien Sansour")


def guest(name: str, *accepted: str) -> Grade:
    """The grade of a `who` answer: the guest's `name` (and the surnames in `accepted`), none of the other names of GUESTS that are no part of it."""
    others = [g for g in GUESTS if g != name and not {w.casefold() for w in g.split()} & {w.casefold() for w in name.split()}]
    return realdump.grade("text", truth="|".join((name, *accepted)), wrong=tuple(others))


CURATED = {"tiny": 1, "standard": 7}  # how many of NOTES a scale asks (a tiny pile has a few questions)
OPEN = "<a short answer>"
NAME = "<the guest's name>"
NUMBER = "<a number>"
NOTES = (
    Note("S02E02", "oldest-stars", "Who was the guest on the episode about searching for the oldest stars?", "who", NAME,
         guest("Anna Frebel", "Frebel"), 35, 40, "Anna Frebel", "Anna Frebel"),
    Note("S04E04", "four-decades", "Who was the guest who had taught computer science for almost four decades?", "who", NAME,
         guest("Eric Grimson", "Grimson"), 45, 66, "Eric Grimson", "Eric Grimson"),
    Note("S05E01", "cloud-chamber", "Who was the guest on the episode about the cloud chamber?", "who", NAME,
         guest("Anne White"), 21, 26, "Anne White", "Anne White"),
    Note("S02E04", "astronaut", "Who was the guest on the episode with the former NASA astronaut?", "who", NAME,
         guest("Jeff Hoffman", "Hoffman"), 35, 40, "Jeff Hoffman", "Jeff Hoffman"),
    Note("S01E01", "toenails", "In the episode about ionizing radiation, what did the guest tell his students to analyze instead of a theoretical "
         "problem?", "what", OPEN, realdump.grade("text", truth="toenails|toenail"), 3.5, 9, "toenails", "your toenails"),
    Note("S05E06", "teaching-since", "In the episode about App Inventor, since what year had the guest been teaching at MIT?", "what", NUMBER,
         realdump.grade("number", truth="1972", wrong=("1969",)), 25, 30, "since 1972", "1972"),
    Note("S03E07", "alexa-lights", "In the episode that opens with a child talking to Alexa, did she ask first for the lights to go off or on?",
         "before", OPEN, realdump.grade("text", truth="off", wrong=("on",)), 1, 11, "turn off the lights", "off"),
)


def questions_of(talks: Sequence[Talk], stems: Sequence[str], scale: str, seed: int) -> tuple[Question, ...]:
    """The questions of `talks` (`stems` the source's names of them), drawn with `seed`."""
    speech_files.check_never(talks, NEVER)
    terms = speech_files.screened(talks, TERMS, others=())
    return speech.questions_of(talks, terms, NEVER, SAY, QUOTAS[scale], seed, realdump.local_t, speech_files.curated(NOTES, stems)[:CURATED[scale]])
