"""Inverted index: doc store, per-field postings, deletes, BM25 search.

Postings key layout: ``"{field}\\x00{term}"`` for fielded terms plus a
merged ``"\\x00{term}"`` entry so unfielded queries search every field
at once. Each posting maps doc_id -> sorted position list.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Optional

from .analysis import analyze_with_positions
from .query import Bool, Fuzzy, Phrase, Prefix, Term, edit_distance, parse_query
from .scorer import BM25
from .store import Store

__all__ = ["Hit", "Index"]

_SNIP_WINDOW = 60


@dataclass
class Hit:
    """One ranked search result."""

    doc_id: str
    score: float
    fields: dict[str, Any]
    snippet: str = ""
    highlights: list[str] = _dc_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "score": self.score,
            "fields": dict(self.fields),
            "snippet": self.snippet,
            "highlights": list(self.highlights),
        }


def _postings_key(fld: str, term: str) -> str:
    return f"{fld}\x00{term}"


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_field_text(v) for v in value)
    return str(value)


def _highlight(text: str, terms: set[str]) -> str:
    if not text or not terms:
        return text[: 2 * _SNIP_WINDOW] if text else ""
    lowered = {t.lower() for t in terms}
    parts = re.split(r"(\W+)", text, flags=re.UNICODE)
    for i, part in enumerate(parts):
        if part and part[0].isalnum() and part.lower() in lowered:
            parts[i] = f"<em>{part}</em>"
    return "".join(parts)


def _snippet(text: str, terms: set[str], width: int = _SNIP_WINDOW) -> str:
    if not text:
        return ""
    if not terms:
        short = text[: 2 * width]
        return short + ("..." if len(text) > 2 * width else "")
    low = text.lower()
    best = -1
    for t in terms:
        idx = low.find(t.lower())
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    if best == -1:
        short = text[: 2 * width]
        return short + ("..." if len(text) > 2 * width else "")
    start = max(0, best - width)
    end = min(len(text), best + width)
    suffix = "..." if end < len(text) else ""
    prefix = "..." if start > 0 else ""
    return prefix + text[start:end] + suffix


class Index:
    """A full-text index living in a directory on disk."""

    def __init__(self, path: str | Path, k1: float = 1.2, b: float = 0.75,
                 _load: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._scorer = BM25(k1=k1, b=b)
        self._store = Store(self.path)
        self._docs: dict[str, dict[str, Any]] = {}
        self._postings: dict[str, dict[str, list[int]]] = {}
        self._deletes: set[str] = set()
        self._seq = 0
        if _load:
            self._reload()

    @classmethod
    def create(cls, path: str | Path) -> "Index":
        idx = cls(path, _load=False)
        idx._reload()
        idx.commit()
        return idx

    @classmethod
    def open(cls, path: str | Path) -> "Index":
        return cls(path, _load=True)

    def _reload(self) -> None:
        state = self._store.load()
        self._docs = {str(k): v for k, v in state["docs"].items()}
        self._postings = {
            str(k): {str(d): list(p) for d, p in v.items()}
            for k, v in state["postings"].items()
        }
        self._deletes = {str(d) for d in state["deletes"]}
        self._seq = int(state.get("seq", 0))
        for dead in self._deletes:
            self._docs.pop(dead, None)

    # -- writes ------------------------------------------------------
    def add_document(self, doc_id: str | int, fields: dict[str, Any]) -> None:
        """Add (or replace) a document. Visible now; commit() persists."""
        if fields is None or not isinstance(fields, dict):
            raise ValueError("fields must be a dict of field -> value")
        doc_id = str(doc_id)
        if not doc_id:
            raise ValueError("doc_id must be non-empty")
        text_fields = {k: _field_text(v) for k, v in fields.items()}
        if not any(t.strip() for t in text_fields.values()):
            raise ValueError("document has no indexable text")
        with self._lock:
            if doc_id in self._docs:
                self._remove_postings(doc_id)
            self._deletes.discard(doc_id)
            self._docs[doc_id] = dict(fields)
            for fname, text in text_fields.items():
                fld = fname.lower()
                for term, pos in analyze_with_positions(text):
                    slot = self._postings.setdefault(_postings_key(fld, term), {})
                    slot.setdefault(doc_id, []).append(pos)
                    all_slot = self._postings.setdefault(_postings_key("", term), {})
                    all_slot.setdefault(doc_id, []).append(pos)

    def delete_document(self, doc_id: str | int) -> bool:
        """Delete a doc (tombstone). Returns True if the doc existed."""
        doc_id = str(doc_id)
        with self._lock:
            if doc_id not in self._docs:
                return False
            self._remove_postings(doc_id)
            del self._docs[doc_id]
            self._deletes.add(doc_id)
            return True

    def _remove_postings(self, doc_id: str) -> None:
        for key in list(self._postings.keys()):
            slot = self._postings[key]
            if doc_id in slot:
                del slot[doc_id]
                if not slot:
                    del self._postings[key]

    def commit(self) -> int:
        """Persist a snapshot (JSON + fsync). Returns the new sequence."""
        with self._lock:
            self._seq = self._store.commit_state(
                self._docs, self._postings, self._deletes)
            return self._seq

    # -- reads --------------------------------------------------------
    @property
    def seq(self) -> int:
        return self._seq

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def num_docs(self) -> int:
        return len(self._docs)

    def get(self, doc_id: str | int) -> Optional[dict[str, Any]]:
        with self._lock:
            doc = self._docs.get(str(doc_id))
            return dict(doc) if doc is not None else None

    def doc_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._docs.keys())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "num_docs": len(self._docs),
                "num_terms": len(self._postings),
                "num_postings": sum(len(v) for v in self._postings.values()),
                "num_deletes": len(self._deletes),
                "seq": self._seq,
                "path": str(self.path),
            }


    # -- search --------------------------------------------------------
    def search(self, query: str, top_k: int = 10,
               fields: Optional[dict[str, float]] = None) -> list[Hit]:
        """Ranked search returning up to top_k hits (BM25, desc score)."""
        if top_k <= 0:
            return []
        boosts = self._norm_boosts(fields)
        with self._lock:
            if not self._docs or not query or not query.strip():
                return []
            tree = parse_query(query)
            if tree.is_empty():
                return []
            doc_scores: dict[str, float] = {}
            doc_terms: dict[str, set[str]] = {}
            matched = self._eval_bool(tree, boosts, doc_scores, doc_terms)
            if not matched:
                return []
            hits = [self._make_hit(d, doc_scores.get(d, 0.0),
                                   doc_terms.get(d, set()))
                    for d in matched]
            hits.sort(key=lambda h: (-h.score, h.doc_id))
            return hits[:top_k]

    def count(self, query: str, fields: Optional[dict[str, float]] = None) -> int:
        """Return the number of documents matching ``query``.

        Cheaper than ``len(search(query))`` because it does not materialize
        Hit objects, snippets, or highlights. Same matching semantics as
        ``search`` (same query syntax, same field boosts).
        """
        boosts = self._norm_boosts(fields)
        with self._lock:
            if not self._docs or not query or not query.strip():
                return 0
            tree = parse_query(query)
            if tree.is_empty():
                return 0
            doc_scores: dict[str, float] = {}
            doc_terms: dict[str, set[str]] = {}
            matched = self._eval_bool(tree, boosts, doc_scores, doc_terms)
            return len(matched)

    @staticmethod
    def _norm_boosts(fields: Optional[dict[str, float]]) -> dict[str, float]:
        if not fields:
            return {}
        out: dict[str, float] = {}
        for k, v in fields.items():
            try:
                w = float(v)
            except (TypeError, ValueError):
                continue
            if w > 0:
                out[str(k).lower()] = w
        return out

    def _boost_of(self, fld: Optional[str], boosts: dict[str, float]) -> float:
        if fld and fld in boosts:
            return boosts[fld]
        return 1.0

    # -- query evaluation ------------------------------------------------
    def _eval_bool(self, node: Bool, boosts: dict[str, float],
                   scores: dict[str, float],
                   terms: dict[str, set[str]]) -> set[str]:
        if node.is_empty():
            return set()
        excluded: set[str] = set()
        for sub in node.must_not:
            excluded |= self._eval_clause(sub, boosts, None, None)
        if node.must:
            result: Optional[set[str]] = None
            for sub in node.must:
                got = self._eval_clause(sub, boosts, scores, terms)
                result = got if result is None else (result & got)
                if not result:
                    break
            result = result or set()
            return result - excluded
        result = set()
        for sub in node.should:
            got = self._eval_clause(sub, boosts, scores, terms)
            result |= got
        return result - excluded

    def _eval_clause(self, clause, boosts: dict[str, float],
                     scores: Optional[dict[str, float]],
                     terms: Optional[dict[str, set[str]]]) -> set[str]:
        if isinstance(clause, Bool):
            if scores is None:
                tmp_s: dict[str, float] = {}
                tmp_t: dict[str, set[str]] = {}
                return self._eval_bool(clause, boosts, tmp_s, tmp_t)
            return self._eval_bool(clause, boosts, scores, terms)
        if isinstance(clause, Term):
            return self._score_term(clause.term, clause.field, boosts,
                                    scores, terms)
        if isinstance(clause, Phrase):
            return self._score_phrase(clause.terms, clause.field, boosts,
                                      scores, terms)
        if isinstance(clause, Prefix):
            return self._score_prefix(clause.prefix, clause.field, boosts,
                                      scores, terms)
        if isinstance(clause, Fuzzy):
            return self._score_fuzzy(clause, boosts, scores, terms)
        return set()  # pragma: no cover - unknown node

    # -- posting helpers ---------------------------------------------------
    def _postings_for(self, fld: Optional[str], term: str):
        if fld:
            return self._postings.get(_postings_key(fld, term), {})
        return self._postings.get(_postings_key("", term), {})

    def _doc_len(self, doc_id: str, fld: Optional[str]) -> int:
        total = 0
        prefix = (fld + "\x00") if fld else ""
        for key, slot in self._postings.items():
            if fld and not key.startswith(prefix):
                continue
            plist = slot.get(doc_id)
            if plist:
                total += len(plist)
        return total

    def _avg_len(self, fld: Optional[str]) -> float:
        if not self._docs:
            return 1.0
        total = sum(self._doc_len(d, fld) for d in self._docs)
        return (total / len(self._docs)) or 1.0

    def _add_score(self, doc_id: str, term: str, fld: Optional[str],
                   boost: float, scores, terms) -> None:
        postings = self._postings_for(fld, term)
        plist = postings.get(doc_id)
        if not plist:
            return
        n = len(self._docs)
        s = self._scorer.score(len(plist), len(postings), n,
                               self._doc_len(doc_id, fld),
                               self._avg_len(fld))
        if scores is not None:
            scores[doc_id] = scores.get(doc_id, 0.0) + boost * s
        if terms is not None:
            terms.setdefault(doc_id, set()).add(term)

    def _field_names(self) -> list[str]:
        names: set[str] = set()
        for key in self._postings:
            fld, _, _ = key.partition("\x00")
            if fld:
                names.add(fld)
        return sorted(names)

    def _score_term(self, term: str, fld: Optional[str],
                    boosts: dict[str, float], scores, terms) -> set[str]:
        if fld is None and boosts:
            matched: set[str] = set()
            for name, weight in sorted(boosts.items()):
                got = self._score_term(term, name, {name: weight},
                                       scores, terms)
                matched |= got
            return matched
        boost = self._boost_of(fld, boosts)
        postings = self._postings_for(fld, term)
        matched = {d for d in postings if d in self._docs}
        for d in matched:
            self._add_score(d, term, fld, boost, scores, terms)
        return matched

    def _score_phrase(self, qterms: list[str], fld: Optional[str],
                      boosts: dict[str, float], scores, terms) -> set[str]:
        if not qterms:
            return set()
        if fld is None and boosts:
            boosted: set[str] = set()
            for name, weight in sorted(boosts.items()):
                boosted |= self._score_phrase(qterms, name, {name: weight},
                                              scores, terms)
            return boosted
        if len(qterms) == 1:
            return self._score_term(qterms[0], fld, boosts, scores, terms)
        boost = self._boost_of(fld, boosts)
        lists = [self._postings_for(fld, t) for t in qterms]
        common = set(lists[0])
        for post in lists[1:]:
            common &= set(post)
        common &= set(self._docs)
        matched: set[str] = set()
        for d in common:
            pos_sets = [set(lists[i].get(d, [])) for i in range(len(qterms))]
            for p in pos_sets[0]:
                ok = True
                for i in range(1, len(qterms)):
                    if (p + i) not in pos_sets[i]:
                        ok = False
                        break
                if ok:
                    matched.add(d)
                    break
        for d in matched:
            for t in qterms:
                self._add_score(d, t, fld, boost, scores, terms)
            if scores is not None:
                scores[d] = scores.get(d, 0.0) + 0.5 * boost
        return matched

    def _expand_prefix(self, prefix: str,
                       fld: Optional[str]) -> list[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        if fld:
            want = fld + "\x00"
            for key in self._postings:
                if key.startswith(want) and \
                        key[len(want):].startswith(prefix):
                    out.add((fld, key[len(want):]))
        else:
            for key in self._postings:
                if key.startswith("\x00") and key[1:].startswith(prefix):
                    out.add(("", key[1:]))
        return sorted(out)

    def _score_prefix(self, prefix: str, fld: Optional[str],
                      boosts: dict[str, float], scores, terms) -> set[str]:
        if fld is None and boosts:
            matched: set[str] = set()
            for name, weight in sorted(boosts.items()):
                matched |= self._score_prefix(prefix, name, {name: weight},
                                              scores, terms)
            return matched
        boost = self._boost_of(fld, boosts)
        matched = set()
        for efld, term in self._expand_prefix(prefix, fld):
            scope = efld or None
            postings = self._postings_for(scope, term)
            for d in [d for d in postings if d in self._docs]:
                matched.add(d)
                n = len(self._docs)
                s = self._scorer.score(
                    len(postings[d]), len(postings), n,
                    self._doc_len(d, scope), self._avg_len(scope))
                if scores is not None:
                    scores[d] = scores.get(d, 0.0) + boost * s
                if terms is not None:
                    terms.setdefault(d, set()).add(term)
        return matched

    def _expand_fuzzy(self, wanted: str, max_dist: int,
                      fld: Optional[str]) -> list[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        if fld:
            want = fld + "\x00"
            pairs = [(fld, k[len(want):]) for k in self._postings
                     if k.startswith(want)]
        else:
            pairs = [("", k[1:]) for k in self._postings
                     if k.startswith("\x00")]
        for scope, t in pairs:
            if abs(len(t) - len(wanted)) > max_dist:
                continue
            if edit_distance(t, wanted, max_dist) <= max_dist:
                out.add((scope, t))
        return sorted(out)

    def _score_fuzzy(self, node: Fuzzy, boosts: dict[str, float],
                     scores, terms) -> set[str]:
        if node.field is None and boosts:
            matched: set[str] = set()
            for name, weight in sorted(boosts.items()):
                sub = Fuzzy(term=node.term, max_dist=node.max_dist,
                            field=name, analyzed=node.analyzed)
                matched |= self._score_fuzzy(sub, {name: weight},
                                             scores, terms)
            return matched
        boost = self._boost_of(node.field, boosts)
        wanted = node.analyzed or node.term
        matched = set()
        for efld, term in self._expand_fuzzy(wanted, node.max_dist,
                                             node.field):
            scope = efld or None
            postings = self._postings_for(scope, term)
            for d in [d for d in postings if d in self._docs]:
                matched.add(d)
                n = len(self._docs)
                s = self._scorer.score(
                    len(postings[d]), len(postings), n,
                    self._doc_len(d, scope), self._avg_len(scope))
                if scores is not None:
                    scores[d] = scores.get(d, 0.0) + boost * s
                if terms is not None:
                    terms.setdefault(d, set()).add(term)
        return matched

    # -- snippets ------------------------------------------------------------
    def _make_hit(self, doc_id: str, score: float,
                  matched_terms: set[str]) -> Hit:
        stored = dict(self._docs.get(doc_id, {}))
        texts = [(k, _field_text(v)) for k, v in stored.items()
                 if _field_text(v).strip()]
        snippet = ""
        highlights: list[str] = []
        if texts:
            if matched_terms:
                scored = []
                for k, t in texts:
                    low = t.lower()
                    c = sum(low.count(m.lower()) for m in matched_terms)
                    scored.append((c, len(t), k, t))
                scored.sort(key=lambda r: (-r[0], r[1]))
                best_text = scored[0][3]
            else:
                best_text = texts[0][1]
            snippet = _snippet(best_text, matched_terms)
            highlights = [_highlight(t, matched_terms) for _, t in texts]
        return Hit(doc_id=doc_id, score=float(score), fields=stored,
                   snippet=snippet, highlights=highlights)



