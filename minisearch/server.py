"""Thin HTTP API over the core Index (stdlib ThreadingHTTPServer).

Endpoints::

    GET  /health
    PUT  /indexes/{name}/docs            {"id": ..., "fields": {...}}
    DELETE /indexes/{name}/docs/{id}
    POST /indexes/{name}/search          {"query": ..., "top_k": 10,
                                          "fields": {"title": 2.0}}
    GET  /indexes/{name}/stats
    POST /indexes/{name}/commit          persist to disk

Indexes live under a root data dir (one subdir per index name).
"""

from __future__ import annotations

import json
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .index import Index

__all__ = ["MiniSearchHandler", "serve", "build_server", "INDEX_REGISTRY"]

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

INDEX_REGISTRY: dict[str, Index] = {}
_REGISTRY_LOCK = threading.RLock()
_DATA_ROOT = Path("data")


def _json(handler: BaseHTTPRequestHandler, status: int,
          payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return None
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise _BadRequest(f"invalid JSON body: {exc}") from exc


class _BadRequest(Exception):
    pass


class _NotFound(Exception):
    pass


def get_index(name: str) -> Index:
    """Return (creating/lazily opening) the Index for *name*."""
    if not _NAME_RE.match(name):
        raise _BadRequest(f"invalid index name: {name!r}")
    with _REGISTRY_LOCK:
        idx = INDEX_REGISTRY.get(name)
        if idx is None:
            idx = Index.open(_DATA_ROOT / name)
            INDEX_REGISTRY[name] = idx
        return idx


class MiniSearchHandler(BaseHTTPRequestHandler):
    """HTTP handler; keep thin -- all search logic lives in Index."""

    server_version = "MiniSearch/0.1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        pass

    # -- routing ---------------------------------------------------------
    def _route(self, method: str):
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if method == "GET" and path == "/health":
            return ("health", {})
        m = re.fullmatch(r"/indexes/([^/]+)/docs/([^/]+)", path)
        if m and method == "DELETE":
            return ("delete_doc", {"index": m.group(1), "doc_id": m.group(2)})
        m = re.fullmatch(r"/indexes/([^/]+)/docs", path)
        if m and method == "PUT":
            return ("put_doc", {"index": m.group(1)})
        m = re.fullmatch(r"/indexes/([^/]+)/search", path)
        if m and method == "POST":
            return ("search", {"index": m.group(1)})
        m = re.fullmatch(r"/indexes/([^/]+)/stats", path)
        if m and method == "GET":
            return ("stats", {"index": m.group(1)})
        m = re.fullmatch(r"/indexes/([^/]+)/commit", path)
        if m and method == "POST":
            return ("commit", {"index": m.group(1)})
        return (None, {})

    def _dispatch(self, method: str) -> None:
        action, params = self._route(method)
        try:
            if action == "health":
                _json(self, 200, {"ok": True, "service": "minisearch"})
            elif action == "put_doc":
                body = _read_json(self)
                if not isinstance(body, dict):
                    raise _BadRequest("body must be a JSON object")
                if "id" not in body or "fields" not in body:
                    raise _BadRequest("body needs 'id' and 'fields'")
                if not isinstance(body["fields"], dict):
                    raise _BadRequest("'fields' must be an object")
                idx = get_index(urllib.parse.unquote(params["index"]))
                idx.add_document(str(body["id"]), body["fields"])
                _json(self, 200, {"ok": True, "id": str(body["id"])})
            elif action == "delete_doc":
                idx = get_index(urllib.parse.unquote(params["index"]))
                ok = idx.delete_document(urllib.parse.unquote(params["doc_id"]))
                if not ok:
                    raise _NotFound("document not found")
                _json(self, 200, {"ok": True})
            elif action == "search":
                body = _read_json(self)
                if not isinstance(body, dict) or "query" not in body:
                    raise _BadRequest("body needs 'query'")
                top_k = body.get("top_k", 10)
                try:
                    top_k = int(top_k)
                except (TypeError, ValueError) as exc:
                    raise _BadRequest("'top_k' must be int") from exc
                flds = body.get("fields")
                if flds is not None and not isinstance(flds, dict):
                    raise _BadRequest("'fields' must be an object")
                idx = get_index(urllib.parse.unquote(params["index"]))
                hits = idx.search(str(body["query"]), top_k=top_k, fields=flds)
                _json(self, 200, {"hits": [h.to_dict() for h in hits],
                                  "total": len(hits)})
            elif action == "stats":
                idx = get_index(urllib.parse.unquote(params["index"]))
                _json(self, 200, idx.stats())
            elif action == "commit":
                idx = get_index(urllib.parse.unquote(params["index"]))
                seq = idx.commit()
                _json(self, 200, {"ok": True, "seq": seq})
            else:
                _json(self, 404, {"error": "not found", "path": self.path})
        except _BadRequest as exc:
            _json(self, 400, {"error": str(exc)})
        except _NotFound as exc:
            _json(self, 404, {"error": str(exc)})
        except ValueError as exc:
            _json(self, 400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            _json(self, 500, {"error": f"internal error: {exc}"})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")


def build_server(host: str = "127.0.0.1", port: int = 8000,
                 data_dir: str | Path = "data") -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server."""
    global _DATA_ROOT
    _DATA_ROOT = Path(data_dir)
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), MiniSearchHandler)
    server.daemon_threads = True
    return server


def serve(host: str = "127.0.0.1", port: int = 8000,
          data_dir: str | Path = "data") -> None:
    """Serve forever (Ctrl-C to stop)."""
    server = build_server(host, port, data_dir)
    print(f"minisearch serving on http://{host}:{server.server_port} "
          f"(data={Path(data_dir)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
