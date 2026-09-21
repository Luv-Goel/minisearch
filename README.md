# minisearch

[![CI](https://github.com/Luv-Goel/minisearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Luv-Goel/minisearch/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![zero-deps](https://img.shields.io/badge/deps-zero-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

A from-scratch full-text search engine with **BM25 ranking** in pure Python
(stdlib only) — an incorporates-everything-learning-project inspired by
Elasticsearch/Lucene. Sibling infra to SQL engines (minidb-style B+ trees):
minisearch is **inverted-index + BM25 + persistence**, no SQL anywhere.

## Features

| Area | What you get |
|---|---|
| Analysis | lowercase, unicode word split, built-in stop-words, porter-lite stemmer (~30 lines, no NLTK) |
| Index | per-field postings with term frequencies + positions, arbitrary text fields, doc replace |
| Ranking | BM25 (k1=1.2, b=0.75, +0.5 IDF smoothing), per-field boosts, phrase bonus |
| Queries | terms, `"phrases"`, `AND`/`OR`/`NOT`, `+`/`-` prefixes, `field:term`, `term*`, `term~1`/`~2`, parens |
| Persistence | JSON snapshot segments + atomic manifest, fsync on commit, crash recovery, delete tombstones |
| Server | stdlib `ThreadingHTTPServer` JSON API + urllib client + argparse CLI |
| Tests | 59 unittest tests, stdlib only |

## Quickstart

Library:

```python
from minisearch import Index

idx = Index.create("./my-index")
idx.add_document("1", {"title": "BM25 ranking",
                       "body": "full text search with an inverted index"})
idx.add_document("2", {"title": "Cooking pasta",
                       "body": "tomato sauce with garlic and basil"})
idx.commit()

for hit in idx.search("search engine", top_k=5, fields={"title": 2.0}):
    print(hit.doc_id, round(hit.score, 4), hit.snippet)
```

Server:

```bash
minisearch serve --port 8000 --data-dir ./data
curl -X PUT localhost:8000/indexes/books/docs \
  -d '{"id":"1","fields":{"title":"python guide","body":"search engines"}}'
curl -X POST localhost:8000/indexes/books/search \
  -d '{"query":"python","top_k":5}'
```

CLI:

```bash
minisearch index ./idx 1 --field title=hello --field body="hello world"
minisearch search ./idx "hello" --top-k 5 --boost title=2.0
minisearch stats ./idx
minisearch delete ./idx 1
```

## Scoring (BM25)

```
score(q, d) = SUM_t IDF(t) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * |d| / avgdl))
IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1),   k1 = 1.2, b = 0.75
```

Term frequency saturates (the 10th occurrence matters less than the 1st),
long documents are penalised via `|d| / avgdl`, and rare terms score higher
through IDF. Scores are computed per field-scope and multiplied by the
field boost; phrase matches earn a small bonus. Higher score = ranked first;
ties break by doc id for determinism.

## Query syntax

| Syntax | Meaning | Example |
|---|---|---|
| `a b` | `a` OR `b` | `python cooking` |
| `a AND b` | must contain both | `python AND search` |
| `a OR b` | either (explicit) | `cooking OR java` |
| `NOT x`, `-x` | exclude docs with `x` | `search NOT pasta`, `search -pasta` |
| `+x` | must contain `x` | `+python +search` |
| `"a b"` | phrase (adjacent positions) | `"quick brown"` |
| `field:q` | search one field | `title:python`, `title:"night sky"` |
| `pre*` | prefix / wildcard | `search*` |
| `w~1`, `w~2` | fuzzy, edit distance <= 1 or 2 | `serch~1` |
| `(...)` | grouping | `(a OR b) AND c` |

## Architecture

```
                +-------------------+
                |    CLI / client   |
                +--------+----------+
                         | HTTP/JSON (urllib)
                +--------v----------+
                |  server.py (thin) |
                +--------+----------+
                         |
              +----------v-----------+
              | index.py             |
              |  query -> postings   |
              |  -> BM25 -> hits     |
              +--+------+------+----+
                 |      |      |     |
          analysis scorer query store
          (token (BM25) (bool (JSON+
           +stem)        AST)  fsync)
```

`server.py` holds no search logic — it parses HTTP and delegates to
`Index`. `store.py` owns durability (atomic manifest + fsync + recovery).

## Layout

```
minisearch/
  minisearch/__init__.py   exports Index, Hit, BM25, parse_query, __version__
  minisearch/analysis.py   tokenizer, stop-words, porter-lite stemmer
  minisearch/index.py      inverted index, doc store, BM25 search
  minisearch/scorer.py     BM25
  minisearch/query.py      parser: boolean/phrase/fielded/prefix/fuzzy
  minisearch/store.py      persistence, fsync, recovery
  minisearch/server.py     thin HTTP API
  minisearch/client.py     urllib client
  minisearch/cli.py        argparse CLI
  tests/                   test_analysis.py, test_search.py,
                           test_persistence.py, test_server.py
  examples/demo.py         indexes 10 docs, runs 7 queries
  README.md  LICENSE  .gitignore  pyproject.toml
  .github/workflows/ci.yml
```

## Tests

```bash
python -m unittest discover -s tests -v   # 59 tests, stdlib only
python examples/demo.py                    # end-to-end demo
```

## Deliberate limits

Single-process library (no sharding/replication), full snapshot per commit
(no incremental segments/merging), JSON storage (not a compact binary
format), English-centric analysis, fuzzy/prefix expand against the term
dictionary (fine at this scale, not at web scale).
