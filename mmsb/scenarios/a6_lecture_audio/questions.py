"""The questions of A6, every answer from the transcripts (source.py), each asked the morning after the term's last lecture. The templates are
speech.py's (its docstring has the table), over the terms below, with the person calling the recordings lectures and the speaker the professor:

| template              | question                                                                | category | rule              |
| `first`               | When did the professor first mention the knapsack problem?              | when     | `time`, 30 s      |
| `did`                 | Did the professor ever mention memoization? If so, when was the first?  | did      | `time`, 30 s      |
| `which`               | In which lecture did the professor talk about the Titanic?              | find     | `set` of one      |
| `all`                 | Which lectures mention the empirical rule?                              | find_all | `set`             |
| `count`               | In how many lectures did the professor mention confidence intervals?    | how_many | `count`           |
| `long`                | How long, in minutes, was the lecture in which the professor talked     | how_many | `amount`, 1 min   |
|                       | about hierarchical clustering?                                          |          |                   |
| `order`               | In the lecture of Monday 7 November 2016, did the professor bring up X   | order    | `text`            |
|                       | before or after Y?                                                      |          |                   |
| `what`                | How many steps north does the masochistic drunk take?                   | what     | `text`, `number`, |
|                       |                                                                         |          | `amount`          |
| `before`              | Which kind of model did the professor talk about first?                 | order    | `text`            |
| `absent_first`, ...   | `first`, `did` or `which` of a term in no transcript                    | as above | `absent`          |

A term is one the lectures say (the build checks it): an algorithm, a method, an example a student remembers. A term a student in the room says
is not asked, nor one first said in the first minute of its file (`speech_files.screened`), and nothing asks about the OCW credit that opens every
file. A term that was never said is one the course could have covered and did not (the build checks that it is in no caption, even set as run-on
letters)."""

from __future__ import annotations

from collections.abc import Sequence

from mmsb.core.question import Question
from mmsb.scenarios import realdump
from mmsb.scenarios.realdump import speech, speech_files
from mmsb.scenarios.realdump.speech import Say, Talk, Term
from mmsb.scenarios.realdump.speech_files import Note

CODE = "A6"
SOURCE_NAME = "recordings"  # the dump's source
SAY = Say(CODE, SOURCE_NAME, "audio", "the professor", "lecture", "lectures")
QUOTAS = {"tiny": {"first": 1, "which": 1, "absent_did": 1},
          "standard": {"first": 2, "did": 2, "which": 2, "all": 1, "count": 2, "long": 1, "order": 1, "absent_first": 2, "absent_did": 2, "absent_which": 2}}

TERMS = (
    Term("knapsack problem", "the knapsack problem"), Term("greedy algorithm", "greedy algorithms"), Term("dynamic programming", "dynamic programming"),
    Term("memoization", "memoization"), Term("Fibonacci", "Fibonacci"), Term("shortest path", "the shortest path"),
    Term("depth first search", "depth-first search"), Term("breadth first search", "breadth-first search"), Term("random walk", "random walks"),
    Term("masochistic drunk", "the masochistic drunk"), Term("roulette", "roulette"), Term("birthday problem", "the birthday problem"),
    Term("Monte Carlo", "Monte Carlo"), Term("central limit theorem", "the central limit theorem"), Term("empirical rule", "the empirical rule"),
    Term("standard error", "standard error"), Term("confidence interval", "confidence intervals"), Term("normal distribution", "the normal distribution"),
    Term("linear regression", "linear regression"), Term("logistic regression", "logistic regression"), Term("overfitting", "overfitting"),
    Term("cross validation", "cross-validation"), Term("spring constant", "the spring constant"), Term("nearest neighbor", "nearest neighbors"),
    Term("k means", "k-means"), Term("hierarchical clustering", "hierarchical clustering"), Term("centroid", "centroids"),
    Term("Titanic", "the Titanic"), Term("reptile", "reptiles"), Term("alligator", "alligators"), Term("false positive", "false positives"),
    Term("pylab", "pylab"), Term("NumPy", "NumPy"), Term("R squared", "R squared"), Term("Las Vegas", "Las Vegas"), Term("casino", "a casino"),
    Term("pigeonhole", "the pigeonhole principle"), Term("merge sort", "merge sort"), Term("lambda", "lambda"),
)
NEVER = (
    Term("Bayes", "Bayes' theorem"), Term("bootstrap", "bootstrapping"), Term("p value", "p-values"), Term("null hypothesis", "the null hypothesis"),
    Term("random forest", "random forests"), Term("gradient descent", "gradient descent"), Term("traveling salesman", "the traveling salesman problem"),
    Term("TensorFlow", "TensorFlow"), Term("Monty Hall", "the Monty Hall problem"), Term("binary search", "binary search"),
)

CURATED = {"tiny": 1, "standard": 6}  # how many of NOTES a scale asks (a tiny pile has a few questions)
OPEN = "<a short answer>"
NUMBER = "<a number>"
NOTES = (
    Note("lec01_", "finger-exercises", "In the first lecture the professor described the finger exercises of the problem sets. If one takes you longer "
         "than how many minutes should you let the staff know?", "what", NUMBER, realdump.grade("number", truth="15"), 91, 100, "more than 15 minutes", "15"),
    Note("lec11_", "first-reptile", "In the lecture on picking features, which animal did the professor start with as his single example of a reptile?",
         "what", OPEN, realdump.grade("text", truth="cobra", wrong=("boa",)), 1706, 1716, "cobra", "a cobra"),
    Note("lec13_", "titanic-died", "In the lecture on predicting who survived the Titanic, how many passengers did the professor say died?",
         "what", NUMBER, realdump.grade("number", truth="812"), 953, 960, "812 died", "812"),
    Note("lec05_", "drunk-steps", "In the lecture on random walks, how many steps north does the masochistic drunk take when he chooses to head north?",
         "what", NUMBER, realdump.grade("amount", truth="1.1", tol_s=0.05), 660, 668, "1.1 steps north", "1.1"),
    Note("lec04_", "birthday-367", "In the lecture on the birthday problem, how many people did the professor say there have to be in a group to be "
         "certain that two of them share a birthday?", "what", NUMBER, realdump.grade("number", truth="367"), 1888, 1912, "367 people", "367"),
    Note("lec01_", "models-first", "In the first lecture the professor named three kinds of models. Which one did he talk about first?", "before", OPEN,
         realdump.grade("text", truth="optimization", wrong=("statistical", "simulation")), 338, 352, "talk first about optimization models", "optimization"),
)


def questions_of(talks: Sequence[Talk], stems: Sequence[str], scale: str, seed: int) -> tuple[Question, ...]:
    """The questions of `talks` (`stems` the source's names of them), drawn with `seed`."""
    speech_files.check_never(talks, NEVER)
    terms = speech_files.screened(talks, TERMS)
    return speech.questions_of(talks, terms, NEVER, SAY, QUOTAS[scale], seed, realdump.local_t, speech_files.curated(NOTES, stems)[:CURATED[scale]])
