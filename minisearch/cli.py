"""Command-line interface: index / search / serve / stats."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .index import Index

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minisearch",
                                description="minisearch full-text search")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="add a document to an index")
    s.add_argument("path", help="index directory")
    s.add_argument("doc_id", help="document id")
    s.add_argument("--field", action="append", default=[],
                   metavar="name=text", help="repeatable field assignment")
    s.add_argument("--json-fields", default=None,
                   help="JSON object of fields (alternative to --field)")

    s = sub.add_parser("search", help="search an index")
    s.add_argument("path", help="index directory")
    s.add_argument("query", help="query string")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--boost", action="append", default=[],
                   metavar="field=weight", help="field boost (repeatable)")

    s = sub.add_parser("serve", help="run the HTTP API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default="data")

    s = sub.add_parser("stats", help="print index stats")
    s.add_argument("path", help="index directory")

    s = sub.add_parser("delete", help="delete a document")
    s.add_argument("path", help="index directory")
    s.add_argument("doc_id", help="document id")
    return p


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected name=value, got {pair!r}")
        k, _, v = pair.partition("=")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            fields: dict = {}
            if args.json_fields:
                fields.update(json.loads(args.json_fields))
            fields.update(_parse_kv(args.field))
            idx = Index.open(args.path)
            idx.add_document(args.doc_id, fields)
            idx.commit()
            print(f"indexed {args.doc_id}")
        elif args.command == "search":
            boosts: dict[str, float] = {}
            for pair in args.boost:
                k, _, v = pair.partition("=")
                boosts[k] = float(v)
            idx = Index.open(args.path)
            for hit in idx.search(args.query, top_k=args.top_k,
                                  fields=boosts or None):
                print(f"{hit.doc_id}\t{hit.score:.4f}\t{hit.snippet}")
        elif args.command == "serve":
            from .server import serve
            serve(args.host, args.port, args.data_dir)
        elif args.command == "stats":
            print(json.dumps(Index.open(args.path).stats(), indent=2))
        elif args.command == "delete":
            idx = Index.open(args.path)
            print("deleted" if idx.delete_document(args.doc_id)
                  else "not found")
            idx.commit()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def console_main() -> None:
    """Console-script entry point (exits with main()'s return code)."""
    raise SystemExit(main())
