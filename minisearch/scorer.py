"""BM25 ranking (the Lucene/Elasticsearch variant with +0.5 IDF smoothing).

score(q, d) = sum over query terms t of
    IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

with IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1).

Defaults k1=1.2, b=0.75 match Elasticsearch out of the box.
"""

from __future__ import annotations

import math

__all__ = ["BM25"]


class BM25:
    """BM25 scorer with tunable saturation (k1) and length-norm (b)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)

    def idf(self, num_docs: int, doc_freq: int) -> float:
        """IDF with +0.5 smoothing; always >= 0, even when df == N."""
        n = max(0, int(num_docs))
        if n == 0:
            return 0.0
        df = min(max(0, int(doc_freq)), n)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(
        self,
        tf: int,
        doc_freq: int,
        num_docs: int,
        doc_len: int,
        avg_len: float,
    ) -> float:
        """Score one term in one document."""
        if tf <= 0:
            return 0.0
        avg = float(avg_len) if avg_len and avg_len > 0 else 1.0
        dl = float(doc_len) if doc_len and doc_len >= 0 else 0.0
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avg)
        if denom <= 0:
            return 0.0
        return self.idf(num_docs, doc_freq) * (tf * (self.k1 + 1.0)) / denom

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BM25(k1={self.k1}, b={self.b})"
