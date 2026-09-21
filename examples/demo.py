"""Demo: index ~10 sample docs and run 6+ queries with scores/snippets."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minisearch import Index

DOCS = [
    ("d1", {"title": "Introduction to search engines",
            "body": "A search engine builds an inverted index over documents "
                    "and ranks results with BM25 scoring."}),
    ("d2", {"title": "BM25 ranking explained",
            "body": "BM25 combines term frequency saturation with document "
                    "length normalization and smoothed inverse document "
                    "frequency."}),
    ("d3", {"title": "Python for data science",
            "body": "Python is a popular programming language for data "
                    "analysis, machine learning and scripting."}),
    ("d4", {"title": "Cooking pasta at home",
            "body": "Boil pasta until al dente, then toss with tomato sauce, "
                    "garlic and fresh basil."}),
    ("d5", {"title": "Quick brown fox",
            "body": "The quick brown fox jumps over the lazy dog near the "
                    "river bank."}),
    ("d6", {"title": "Elasticsearch in action",
            "body": "Elasticsearch stores JSON documents and supports full "
                    "text search, phrases and fuzzy matching."}),
    ("d7", {"title": "Hiking the Alps",
            "body": "Mountain trails in the Alps offer stunning views, "
                    "glacier lakes and cozy huts."}),
    ("d8", {"title": "Database indexing basics",
            "body": "A B+ tree index speeds up equality and range lookups in "
                    "relational databases."}),
    ("d9", {"title": "Fuzzy search with typos",
            "body": "Fuzzy matching finds search results even when the "
                    "query contains spelling mistakes."}),
    ("d10", {"title": "Night sky photography",
             "body": "Photographing the night sky needs a tripod, long "
                     "exposure and a fast wide lens."}),
]

QUERIES = [
    ("BM25 ranking", "bm25", {}),
    ("phrase query", '"quick brown"', {}),
    ("boolean AND/NOT", "search AND engine NOT database", {}),
    ("fielded + boost", "search", {"title": 3.0}),
    ("prefix query", "search*", {}),
    ("fuzzy query", "serch~1", {}),
    ("fielded phrase", 'title:"night sky"', {}),
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="minisearch-demo-"))
    idx = Index.create(tmp)
    for doc_id, fields in DOCS:
        idx.add_document(doc_id, fields)
    idx.commit()
    print(f"indexed {len(DOCS)} docs in {tmp} (seq={idx.commit()})")
    print("=" * 70)
    for label, query, boosts in QUERIES:
        print(f"\n[{label}] query={query!r} boosts={boosts or '-'}")
        hits = idx.search(query, top_k=3, fields=boosts or None)
        if not hits:
            print("  (no hits)")
            continue
        for h in hits:
            print(f"  {h.doc_id}  score={h.score:.4f}")
            print(f"    snippet: {h.snippet}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
