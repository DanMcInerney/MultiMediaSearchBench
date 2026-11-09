"""The questions of V9, every answer from the transcripts (source.py), each asked the morning after the term's last lecture. The templates are
speech.py's (its docstring has the table), over the terms below, with the person calling the recordings lectures and the speaker the professor:

| template                  | question                                                               | category | rule            |
| `first`                   | When did the professor first mention comparative advantage?            | when     | `time`, 60 s    |
| `did`                     | Did the professor ever mention LeBron? If so, when was the first time? | did      | `time`, 60 s    |
| `which`                   | In which lecture did the professor talk about monopsony?               | find     | `set` of one    |
| `all`                     | Which lectures mention tariffs?                                        | find_all | `set`           |
| `count`                   | In how many lectures did the professor mention moral hazard?           | how_many | `count`         |
| `long`                    | How long, in minutes, was the lecture in which the professor talked    | how_many | `amount`, 1 min |
|                           | about the Giffen goods?                                                |          |                 |
| `order`                   | In the lecture of Wednesday 25 October 2023, did the professor bring   | order    | `text`          |
|                           | up Cheerios before or after Amazon?                                    |          |                 |
| `what`                    | What did the professor say the bidding for a kidney on eBay started at | what     | `text`, `number`, |
|                           |                                                                        |          | `amount`        |
| `before`                  | Which cereal did the professor name right after Cheerios?              | order    | `text`          |
| `absent_first`, ...       | `first`, `did` or `which` of a term in no transcript                   | as above | `absent`        |

A term is one the lectures say (the build checks it): a concept, a company or an example a student remembers, in the words the professor uses for it.
Not one with a plural that is another word ("externality" and "externalities"), nor one with the word "free", the name of a box of every MP4 that the
leak probe would read as a leak. A term a student in the room says is not asked, nor one first said in the first minute of its file
(`speech_files.screened`). A term that was never said is one a course of economics could have covered and did not, and no cousin of a phrase that was
said (not "rent control", for the lectures on price ceilings); it is in no caption, even set as run-on letters."""

from __future__ import annotations

from collections.abc import Sequence

from mmsb.core.question import Question
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech, speech_files
from mmsb.scenarios.realdump.speech import Say, Talk, Term
from mmsb.scenarios.realdump.speech_files import Note

CODE = "V9"
SOURCE_NAME = "lectures"  # the dump's source
SAY = Say(CODE, SOURCE_NAME, "video", "the professor", "lecture", "lectures")
QUOTAS = {"tiny": {"first": 1, "did": 1, "which": 1, "absent_did": 1, "absent_which": 1},
          "standard": {"first": 2, "did": 2, "which": 2, "all": 1, "count": 2, "long": 1, "order": 1, "absent_first": 2, "absent_did": 2,
                       "absent_which": 2}}

TERMS = (
    Term("comparative advantage", "comparative advantage"), Term("Giffen good", "Giffen goods"),
    Term("moral hazard", "moral hazard"), Term("adverse selection", "adverse selection"), Term("Cournot", "Cournot"),
    Term("monopsony", "monopsony"), Term("oligopoly", "oligopoly"), Term("cartel", "cartels"),
    Term("social welfare function", "the social welfare function"), Term("Affordable Care Act", "the Affordable Care Act"),
    Term("minimum wage", "the minimum wage"), Term("social security", "Social Security"), Term("global warming", "global warming"),
    Term("Cheerios", "Cheerios"), Term("Colombia", "Colombia"), Term("isoquant", "isoquants"), Term("isocost", "isocost"),
    Term("tariff", "tariffs"), Term("autarky", "autarky"), Term("game theory", "game theory"), Term("present value", "present value"),
    Term("expected utility", "expected utility"), Term("risk averse", "risk averse"), Term("self control", "self control"),
    Term("behavioral economics", "behavioral economics"), Term("medallion", "medallions"), Term("kidney", "kidneys"),
    Term("insulin", "insulin"), Term("Amazon", "Amazon"), Term("Pepsi", "Pepsi"), Term("ramen", "ramen"), Term("Uber", "Uber"),
    Term("Starbucks", "Starbucks"), Term("sunk cost", "sunk costs"), Term("barriers to entry", "barriers to entry"), Term("patent", "patents"),
    Term("Medicare", "Medicare"), Term("Nash equilibrium", "Nash equilibrium"), Term("Elon Musk", "Elon Musk"), Term("Tesla", "Tesla"), Term("OPEC", "OPEC"),
    Term("Adam Smith", "Adam Smith"), Term("deadweight loss", "deadweight loss"), Term("consumer surplus", "consumer surplus"),
    Term("producer surplus", "producer surplus"), Term("price ceiling", "price ceilings"), Term("indifference curve", "indifference curves"),
    Term("inflation", "inflation"), Term("COVID", "COVID"), Term("pizza", "pizza"), Term("roses", "roses"), Term("Federal Reserve", "the Federal Reserve"),
    Term("Karl Marx", "Karl Marx"), Term("Walmart", "Walmart"), Term("Medicaid", "Medicaid"), Term("TikTok", "TikTok"),
)
NEVER = (
    Term("bitcoin", "Bitcoin"), Term("ChatGPT", "ChatGPT"), Term("Brexit", "Brexit"), Term("Keynes", "Keynes"), Term("Phillips curve", "the Phillips curve"),
    Term("recession", "a recession"), Term("Netflix", "Netflix"), Term("Facebook", "Facebook"), Term("Twitter", "Twitter"),
    Term("Silicon Valley", "Silicon Valley"),
)

CURATED = {"tiny": 1, "standard": 6}  # how many of NOTES a scale asks (a tiny pile has a few questions)
OPEN = "<a short answer>"
NUMBER = "<a number>"
NOTES = (
    Note("lec01_", "kidney-bid", "In the first lecture the professor told a story of someone who auctioned a kidney on eBay. What price, in dollars, did the "
         "bidding start at?", "what", NUMBER, realdump.grade("amount", truth="25000", tol_s=0.0), 1325, 1331, "bidding start at 25 000", "25000"),
    Note("lec14_", "cereal", "In the lecture on competition between firms, which new cereal did the professor say gives a company a monopoly for a while?",
         "what", OPEN, realdump.grade("text", truth="apple cinnamon"), 2386, 2397, "Apple Cinnamon Cheerios", "Apple Cinnamon Cheerios"),
    Note("lec14_", "cereal-types", "In the lecture on competition between firms, how many kinds of breakfast cereal did the professor say there were around "
         "World War II?", "what", NUMBER, realdump.grade("number", truth="3"), 2402, 2412, "three types of cereal", "three"),
    Note("lec10_", "twenty-years", "In the lecture on welfare economics, how many years back did the professor ask us to cast our minds, to the taxicabs "
         "before Uber and Lyft?", "what", NUMBER, realdump.grade("number", truth="20"), 1455, 1465, "20 years ago", "20"),
    Note("lec14_", "after-cheerios", "In the lecture on competition between firms, which cereal did the professor name right after Cheerios when he "
         "listed the three there were around World War II?", "before", OPEN,
         realdump.grade("text", truth="cornflakes|corn flakes", wrong=("quaker", "oats")), 2402, 2412, "Cornflakes", "Cornflakes"),
    Note("lec01_", "positive-first", "In the first lecture, after the kidney story, did the professor take up the positive question or the normative "
         "question first?", "before", OPEN, realdump.grade("text", truth="positive", wrong=("normative",)), 1340, 1349, "positive question is", "the positive one"),
)


def questions_of(talks: Sequence[Talk], stems: Sequence[str], scale: str, seed: int) -> tuple[Question, ...]:
    """The questions of `talks` (`stems` the source's names of them), drawn with `seed`."""
    speech_files.check_never(talks, NEVER)
    terms = speech_files.screened(talks, TERMS)
    return speech.questions_of(talks, terms, NEVER, SAY, QUOTAS[scale], seed, realdump.local_t, speech_files.curated(NOTES, stems)[:CURATED[scale]])
