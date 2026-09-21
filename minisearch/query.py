"""Query parser: boolean, fielded search, phrases, prefix, fuzzy.

Syntax::

    quick brown                implicit OR of terms
    quick AND brown             explicit AND (must contain both)
    "quick brown"               phrase (positional match)
    title:elasticsearch         fielded term (search one field only)
    title:"quick brown"         fielded phrase
    search*                     prefix / wildcard (terms starting with "search")
    title:search*               fielded prefix
    elastik~1 / elastik~2       fuzzy (Levenshtein distance <= 1 or 2)
    NOT danger / -danger        exclusion (docs matching it are dropped)
    +required -excluded         +/- prefixes for MUST / MUST-NOT
    (a OR b) AND c              parentheses for grouping

Semantics:
  * Adjacent clauses combine with OR (Elasticsearch simple-query style).
  * Explicit AND binds tighter than OR and merges clauses into one
    conjunctive group; unary NOT / - excludes a clause from the group.
  * A pure exclusion query ("-foo") matches nothing (like Elasticsearch).
  * Terms/phrases/fuzzy/prefix are AND-ed within AND groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .analysis import analyze, tokenize_raw

__all__ = [
    "Term",
    "Phrase",
    "Prefix",
    "Fuzzy",
    "Bool",
    "parse_query",
    "edit_distance",
]


def edit_distance(a: str, b: str, max_dist: int = 2) -> int:
    """Levenshtein distance with early exit above max_dist (returns >max)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


# -- AST nodes ------------------------------------------------------------
@dataclass
class Term:
    term: str
    field: Optional[str] = None


@dataclass
class Phrase:
    terms: list[str]
    field: Optional[str] = None

# -- lexer -----------------------------------------------------------------
def _lex(text: str) -> list[str]:
    """Split a query into tokens, keeping quotes/parens/operators.

    A field prefix glued to a quoted phrase (``title:"quick brown"``)
    stays one token so the field is not lost.
    """
    toks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c)
            i += 1
        elif c == '"':
            j = text.find('"', i + 1)
            if j == -1:
                j = n
            toks.append(text[i:j + 1] if j < n else text[i:])
            i = (j + 1) if j < n else n
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                if text[j] == '"':
                    k = text.find('"', j + 1)
                    if k == -1:
                        j = n
                        break
                    j = k + 1
                else:
                    j += 1
            toks.append(text[i:j])
            i = j
    return toks


def _analyze_word(word: str) -> list[str]:
    return analyze(word)


# -- token -> leaf -----------------------------------------------------------
def _leaf(tok: str, default_field: Optional[str]) -> Clause | None:
    fld = default_field
    rest = tok
    if ":" in tok and not tok.startswith(":"):
        name, _, after = tok.partition(":")
        if name and all(ch.isalnum() or ch in "_-." for ch in name) and after:
            fld = name.lower()
            rest = after
    if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
        terms = analyze(rest[1:-1])
        if not terms:
            return None
        return Phrase(terms=terms, field=fld)
    if rest.endswith("*") and len(rest) > 1:
        raw = rest[:-1].lower()
        raws = tokenize_raw(raw)
        if not raws:
            return None
        from .analysis import stem as _stem
        prefix = raw if raws[0][:1].isdigit() else _stem(raws[0])
        return Prefix(prefix=prefix, field=fld)
    if "~" in rest and len(rest) > 2:
        base, _, dist_s = rest.rpartition("~")
        if base:
            dist = 1
            if dist_s.isdigit():
                dist = max(1, min(2, int(dist_s)))
            elif dist_s != "":
                return _word_leaf(rest, fld, default_field)
            analyzed = _analyze_word(base)
            if not analyzed:
                return None
            return Fuzzy(term=base.lower(), max_dist=dist, field=fld,
                         analyzed=analyzed[0])
    return _word_leaf(rest, fld, default_field)


def _word_leaf(word: str, fld: Optional[str],
               default_field: Optional[str]) -> Clause | None:
    terms = _analyze_word(word)
    if not terms:
        return None
    if len(terms) == 1:
        return Term(term=terms[0], field=fld)
    node = Bool()
    for t in terms:
        node.must.append(Term(term=t, field=fld))
    return node



@dataclass
class Prefix:
    prefix: str
    field: Optional[str] = None



# -- parser -------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens: list[str], default_field: Optional[str]):
        self.toks = tokens
        self.pos = 0
        self.default_field = default_field

    def peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def next(self) -> Optional[str]:
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self) -> Bool:
        if not self.toks:
            return Bool()
        return self.parse_or()

    def parse_or(self) -> Bool:
        node = self.parse_and()
        while True:
            tok = self.peek()
            if tok is None or tok == ")":
                return node
            if tok.upper() == "OR":
                self.next()
                node = _or(node, self.parse_and())
            elif tok.upper() == "AND":
                self.next()
                node = _or(node, self.parse_and())
            elif tok.upper() == "NOT":
                self.next()
                inner = self.parse_unary()
                excl = Bool()
                if inner is not None:
                    excl.must_not.append(inner)
                node = _and(node, excl)
            elif tok.startswith("-") and len(tok) > 1:
                self.next()
                inner = _leaf(tok[1:], self.default_field)
                excl = Bool()
                if inner is not None:
                    excl.must_not.append(inner)
                node = _and(node, excl)
            else:
                # adjacent required ("+a +b") clauses conjoin, else union
                right = self.parse_and()
                if getattr(node, "required", False) or \
                        getattr(right, "required", False):
                    node = _and(node, right)
                else:
                    node = _or(node, right)
        return node

    def parse_and(self) -> Bool:
        node = self._bool_of(self.parse_unary())
        while (tok := self.peek()) is not None and tok.upper() == "AND":
            self.next()
            # "a AND NOT b" excludes b from a (Elasticsearch semantics)
            if (tok2 := self.peek()) is not None and tok2.upper() == "NOT":
                self.next()
                inner = self.parse_unary()
                excl = Bool()
                if inner is not None:
                    excl.must_not.append(inner)
                node = _and(node, excl)
            else:
                right = self.parse_unary()
                node = _and(node, self._bool_of(right))
        return node

    def parse_unary(self) -> Clause | None:
        tok = self.peek()
        if tok is None or tok == ")":
            return None
        if tok.upper() == "NOT":
            self.next()
            inner = self.parse_unary()
            b = Bool()
            if inner is not None:
                b.must_not.append(inner)
            return b
        if tok.startswith("-") and len(tok) > 1:
            self.next()
            inner = _leaf(tok[1:], self.default_field)
            b = Bool()
            if inner is not None:
                b.must_not.append(inner)
            return b
        if tok.startswith("+") and len(tok) > 1:
            self.next()
            inner = _leaf(tok[1:], self.default_field)
            # "+term" means MUST (like an implicit AND with the neighbours).
            if inner is None:
                return None
            if isinstance(inner, Bool):
                return inner
            b = Bool()
            b.must.append(inner)
            b.required = True  # type: ignore[attr-defined]
            return b
        if tok == "(":
            self.next()
            node = self.parse_or()
            if self.peek() == ")":
                self.next()
            return node
        self.next()
        if tok.upper() in ("AND", "OR"):
            return None
        return _leaf(tok, self.default_field)

    @staticmethod
    def _bool_of(clause: Clause | None) -> Bool:
        if clause is None:
            return Bool()
        if isinstance(clause, Bool):
            return clause
        b = Bool()
        b.must.append(clause)
        return b


def _or(a: Bool, b: Bool) -> Bool:
    if a.is_empty():
        return b
    if b.is_empty():
        return a
    # pure exclusions merge straight into the accumulated node, so
    # "python NOT cooking" and "search -java" filter instead of union.
    if not b.must and not b.should and b.must_not:
        out = Bool()
        out.must = list(a.must)
        out.should = list(a.should)
        out.must_not = list(a.must_not) + list(b.must_not)
        return out
    if not a.must and not a.should and a.must_not:
        return b
    if a.must and not a.should and b.must and not b.should \
            and not a.must_not and not b.must_not:
        out = Bool()
        out.should = [a, b]
        return out
    out = Bool()
    for node in (a, b):
        if isinstance(node, Bool) and not node.must_not:
            if node.must and not node.should:
                out.should.extend(node.must)
                continue
            if node.should and not node.must:
                out.should.extend(node.should)
                continue
        out.should.append(node)
    out.must_not = list(a.must_not) + list(b.must_not)
    return out


def _and(a: Bool, b: Bool) -> Bool:
    if a.is_empty():
        return b
    if b.is_empty():
        return a
    out = Bool()
    for node in (a, b):
        if isinstance(node, Bool):
            out.must.extend(node.must)
            out.must_not.extend(node.must_not)
            if node.should:
                if len(node.should) == 1 and not node.must:
                    out.must.append(node.should[0])
                else:
                    out.must.append(node)
        else:  # pragma: no cover - callers always pass Bool
            out.must.append(node)
    return out


def parse_query(text: str, default_field: Optional[str] = None) -> Bool:
    """Parse query text into a Bool AST (never raises on bad input)."""
    if not text or not text.strip():
        return Bool()
    try:
        parser = _Parser(_lex(text), default_field)
        node = parser.parse()
        return node if isinstance(node, Bool) else Bool()
    except Exception:
        return Bool()

@dataclass
class Fuzzy:
    term: str
    max_dist: int = 1
    field: Optional[str] = None
    analyzed: Optional[str] = None


@dataclass
class Bool:
    must: list = field(default_factory=list)       # all must match
    should: list = field(default_factory=list)     # at least one (if no must)
    must_not: list = field(default_factory=list)   # none may match

    def is_empty(self) -> bool:
        return not (self.must or self.should or self.must_not)


Clause = Term | Phrase | Prefix | Fuzzy | Bool
