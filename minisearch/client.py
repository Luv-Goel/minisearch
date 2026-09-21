"""urllib-based client for the minisearch HTTP API (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

__all__ = ["MiniSearchClient", "MiniSearchError"]


class MiniSearchError(Exception):
    """Raised when the server returns a 4xx/5xx response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class MiniSearchClient:
    """Thin JSON client. ``base_url`` like ``http://127.0.0.1:8000``."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    # -- low level -----------------------------------------------------
    def _request(self, method: str, path: str,
                 payload: Any = None) -> Any:
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = exc.reason
            raise MiniSearchError(exc.code, detail) from exc

    # -- API ------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def put_doc(self, index: str, doc_id: str | int,
                fields: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PUT", f"/indexes/{urllib.parse.quote(str(index), safe='')}/docs",
            {"id": str(doc_id), "fields": fields})

    def delete_doc(self, index: str, doc_id: str | int) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/indexes/{urllib.parse.quote(str(index), safe='')}/docs/"
            f"{urllib.parse.quote(str(doc_id), safe='')}")

    def search(self, index: str, query: str, top_k: int = 10,
               fields: dict[str, float] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if fields:
            payload["fields"] = fields
        return self._request(
            "POST",
            f"/indexes/{urllib.parse.quote(str(index), safe='')}/search",
            payload)

    def stats(self, index: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/indexes/{urllib.parse.quote(str(index), safe='')}/stats")
