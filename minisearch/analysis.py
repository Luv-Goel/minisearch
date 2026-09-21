"""Text analysis for minisearch: tokenization, stop-words, porter-lite stemmer.

Pipeline: lowercase -> unicode word split -> stop-word removal -> stemming.
"""

from __future__ import annotations

import re

__all__ = [
    "STOPWORDS",
    "TOKEN_RE",
    "tokenize_raw",
    "stem",
    "analyze",
    "analyze_with_positions",
]

STOPWORDS = frozenset(
    """
    a an and are as at be but by for from had has have he her his in into is it
    its of on or that the their then there these they this to was were will with
    you your not no nor so than too very can just over under out about
    """.split()
)

# Unicode letters (no digits/underscore) OR digit runs.
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize_raw(text: str) -> list[str]:
    """Lowercase unicode word split (letter runs + digit runs)."""
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def _truncate_double(word: str) -> str:
    """Strip one of a doubled trailing consonant: running -> run."""
    if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Porter-lite stemmer (~30 lines, zero dependencies).

    Handles plurals (cats -> cat, boxes -> box, stories -> story),
    past/progressive (-ed/-ing with double-consonant fix: running -> run,
    connected -> connect) and adverbs (quickly -> quick). Deliberately
    small: irregular forms (went/go, better/good) are NOT conflated.
    """
    if len(word) <= 2:
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:                       # stories -> story
        w = w[:-3] + "y"
    elif w.endswith("sses"):                                   # classes -> class
        w = w[:-2]
    elif w.endswith(("xes", "zes", "ches", "shes")) and len(w) > 4:
        w = w[:-2]                                             # boxes -> box
    elif w.endswith("s") and not w.endswith(("us", "ss", "is")) and len(w) > 3:
        w = w[:-1]                                             # cats -> cat
    if w.endswith("ied") and len(w) > 4:                       # cried -> cry
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5:                     # running -> run
        w = _truncate_double(w[:-3])
    elif w.endswith("ed") and len(w) > 4:                      # connected -> connect
        w = _truncate_double(w[:-2])
    if w.endswith("ly") and len(w) > 4:                        # quickly -> quick
        w = w[:-2]
    return w


def _normalize_token(tok: str) -> str:
    if tok[:1].isdigit():
        return tok
    return stem(tok)


def analyze(text: str) -> list[str]:
    """Analyze text into index/query terms (stop-words removed)."""
    if not text:
        return []
    out: list[str] = []
    for tok in tokenize_raw(text):
        if tok in STOPWORDS:
            continue
        out.append(_normalize_token(tok))
    return out


def analyze_with_positions(text: str) -> list[tuple[str, int]]:
    """Analyze text into (term, position) pairs.

    Positions are sequential over *kept* tokens (stop-words leave no gap),
    so the phrase "quick brown" also matches "quick the brown".
    """
    if not text:
        return []
    out: list[tuple[str, int]] = []
    pos = 0
    for tok in tokenize_raw(text):
        if tok in STOPWORDS:
            continue
        out.append((_normalize_token(tok), pos))
        pos += 1
    return out
