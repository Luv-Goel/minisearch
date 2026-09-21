"""minisearch: a from-scratch full-text search engine with BM25 ranking.

Stdlib only. See :class:`minisearch.index.Index` for the main API.
"""

from __future__ import annotations

from .analysis import analyze, stem
from .index import Hit, Index
from .query import parse_query
from .scorer import BM25

__version__ = "0.1.0"

__all__ = [
    "Index",
    "Hit",
    "BM25",
    "parse_query",
    "analyze",
    "stem",
    "__version__",
]
