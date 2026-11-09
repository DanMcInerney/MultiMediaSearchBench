"""The terms of I8: what a student remembers a course's notes by, and how the text layer of a page is searched for them.

`TERMS` are the words and names of the four courses that the notes use, chosen by reading their text layers (the vocabulary of each course's pages, and
what the lectures are about); the script confirms each one on the pages (`mentions`) and asks only of those whose count suits a question (questions.py).
`NEVER` are terms of the same subjects that a student might expect the notes to hold and that no page of them does, which the script confirms too: they
are the terms of the questions about what is not there."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from mmsb.scenarios.i8_course_pages.source import Page


@dataclass(frozen=True)
class Term:
    course: str  # the course whose subject it is
    search: str  # the words looked for; the last may have an s or es
    said: str  # as a person says it, after "mention": `the photoelectric effect`


def term(course: str, search: str, said: str | None = None) -> Term:
    return Term(course, search, said or search)


TERMS = (
    *(term("chemistry", s, d) for s, d in (
        ("photoelectric effect", "the photoelectric effect"), ("quantum number", "quantum numbers"), ("hydrogen atom", "the hydrogen atom"),
        ("electromagnetic radiation", None), ("classical mechanics", None), ("ground state", "the ground state"), ("wavelength", None),
        ("frequency", None), ("Rutherford", None), ("Thomson", None), ("cathode ray", None), ("Pauli exclusion", "Pauli exclusion"),
        ("Planck", None), ("emission spectrum", None), ("de Broglie", None), ("uncertainty principle", "the uncertainty principle"),
        ("significant figures", None), ("SI prefixes", None), ("orbital", None), ("photon", None), ("Bohr", None), ("ionization energy", None),
        ("wave function", "the wave function"), ("nucleus", "the nucleus"))),
    *(term("algorithms", s, d) for s, d in (
        ("hash table", "hash tables"), ("binary search", None), ("selection sort", None), ("insertion sort", None), ("merge sort", None),
        ("linked list", None), ("binary tree", None), ("direct access array", None), ("worst case", "the worst case"), ("heap", "heaps"),
        ("priority queue", None), ("AVL", "AVL"), ("radix sort", None), ("counting sort", None), ("dynamic array", None),
        ("amortized", "amortized"), ("recursion", None), ("recurrence", None), ("master theorem", "the master theorem"),
        ("graph", None), ("breadth-first search", None), ("depth-first search", None), ("Dijkstra", "Dijkstra"),
        ("Bellman-Ford", None), ("topological", "topological"), ("shortest path", None), ("dynamic programming", None),
        ("sequence interface", "the sequence interface"), ("set interface", "the set interface"), ("queue", None), ("permutation sort", None),
        ("comparison model", "the comparison model"), ("decision tree", None), ("hashing", None), ("collision", None), ("chaining", None),
        ("universal hash", "universal hash"), ("DAG", None), ("rotation", None), ("subtree", None), ("successor", None), ("pigeonhole", "the pigeonhole"))),
    *(term("biology", s, d) for s, d in (
        ("amino acid", None), ("covalent bond", None), ("eukaryotic cell", "eukaryotic cells"), ("template strand", "the template strand"), ("base pair", None), ("stop codon", "the stop codon"),
        ("cell division", None), ("enzyme", None), ("ribozyme", None), ("trypsin", None), ("activation energy", None), ("transition state", "the transition state"),
        ("peptide bond", None), ("ribosome", None), ("mRNA", None), ("tRNA", None), ("DNA polymerase", None), ("RNA polymerase", None),
        ("transcription", None), ("translation", None), ("replication", None), ("mitosis", None), ("meiosis", None), ("chromosome", None),
        ("nucleotide", None), ("ATP", None), ("membrane", None), ("cytoskeleton", "the cytoskeleton"), ("organelle", None), ("mitochondria", None),
        ("golgi", "the Golgi"), ("endoplasmic reticulum", "the endoplasmic reticulum"), ("hydrogen bond", None), ("hydrophobic", None), ("allosteric", None),
        ("inhibitor", None), ("substrate", None), ("promoter", None), ("codon", None), ("anticodon", None), ("polypeptide", None),
        ("start codon", "the start codon"), ("primer", None), ("helicase", None), ("mutation", None), ("Mendel", None))),
    *(term("psychology", s, d) for s, d in (
        ("classical conditioning", None), ("visual cortex", "the visual cortex"), ("primary visual cortex", "the primary visual cortex"),
        ("Phineas Gage", None), ("spatial resolution", None), ("object recognition", None), ("fMRI", None), ("cognitive neuroscience", None),
        ("EEG", None), ("amygdala", "the amygdala"), ("hippocampus", "the hippocampus"), ("frontal lobe", "the frontal lobe"),
        ("temporal lobe", "the temporal lobe"), ("occipital lobe", "the occipital lobe"), ("cerebellum", "the cerebellum"), ("neuron", None),
        ("synapse", None), ("neurotransmitter", None), ("dopamine", None), ("long-term memory", None), ("short-term memory", None), ("Pavlov", None),
        ("operant conditioning", None), ("reinforcement", None), ("Skinner", None), ("attention", None), ("blindsight", None), ("neglect", None),
        ("retina", "the retina"), ("consciousness", None), ("Broca", "Broca"), ("lesion", None), ("PET", "PET"), ("MEG", None),
        ("TMS", None), ("placebo", None), ("random assignment", None), ("Stroop", "Stroop"), ("priming", None), ("encoding", None),
        ("extinction", None), ("bottom-up", "bottom-up"), ("top-down", "top-down"), ("gestalt", None), ("thalamus", "the thalamus"),
        ("cortex", "the cortex"))),
)
NEVER = (
    term("chemistry", "periodic table", "the periodic table"), term("chemistry", "Lewis structure"), term("chemistry", "hybridization"), term("chemistry", "VSEPR"), term("chemistry", "Hund's rule"),
    term("algorithms", "memoization"), term("algorithms", "knapsack"), term("algorithms", "spanning tree"), term("algorithms", "big O"),
    term("biology", "glycolysis"), term("biology", "lac operon"), term("biology", "CRISPR"), term("biology", "photosynthesis"),
    term("biology", "PCR"), term("biology", "plasmid"), term("biology", "Okazaki fragments"),
    term("psychology", "cognitive dissonance"), term("psychology", "double-blind"), term("psychology", "serotonin"),
    term("psychology", "Wernicke"), term("psychology", "habituation"), term("psychology", "working memory"),
    term("psychology", "depth perception"),
)


ANYWHERE = frozenset((  # the `search` of the terms that general knowledge places in more than one of the four subjects, which a question of which course asks of
    "membrane", "mutation", "hydrogen bond", "covalent bond", "activation energy", "transition state", "substrate", "inhibitor", "ATP", "photon",
    "electromagnetic radiation", "neuron", "synapse", "neurotransmitter", "extinction", "translation", "collision", "decision tree", "cortex", "lesion",
    "placebo", "reinforcement"))


def fold(text: str) -> str:
    """`text` as lower-case words of letters and digits with one space between, a space at each end: a phrase is found in it as whole words."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())) + " "


def texts_of(page: Page) -> tuple[str, str]:
    """A page's text folded twice: a word hyphenated across the end of a line joined (`photo-` and `electric` as `photoelectric`) and not (`photo
    electric`), since the text layer does not say which the hyphen was."""
    return fold(re.sub(r"-\s*[\r\n]+\s*", "", page.text)), fold(re.sub(r"-\s*[\r\n]+\s*", " ", page.text))


def mentions(page: Page, search: str) -> bool:
    """Whether the page's text has the words `search` as whole words, the last with an s or es perhaps, in either folding of it."""
    wanted = fold(search)
    return any(wanted in text or wanted[:-1] + "s " in text or wanted[:-1] + "es " in text for text in texts_of(page))
