"""Persistence: JSON snapshot segments + atomic manifest + fsync.

Layout inside the index directory::

    manifest.json          {"version": 1, "seq": N, "segments": ["seg_00007.json"]}
    seg_00007.json         {"docs": {...}, "postings": {...}, "deletes": [...]}

Commit protocol (crash-safe): write segment tmp + fsync, replace;
write manifest tmp + fsync, replace; fsync dir; GC old segments.
A crash leaves only .tmp orphans (cleaned on open) or the last good
manifest -- the index always reopens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["Store", "MANIFEST_NAME"]

MANIFEST_NAME = "manifest.json"
SEGMENT_PREFIX = "seg_"
SEGMENT_SUFFIX = ".json"
TMP_SUFFIX = ".tmp"
STORE_VERSION = 1


def _fsync_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return  # e.g. Windows: cannot open a directory
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name("." + path.name + TMP_SUFFIX)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True)
        _fsync_file(f)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class Store:
    """File store for one index directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.segments: list[str] = []

    # -- load ----------------------------------------------------------
    def load(self) -> dict[str, Any]:
        """Load merged state; safe on fresh dirs and after crashes."""
        self._clean_tmp_orphans()
        manifest = self._read_manifest()
        if manifest is None:
            segs = sorted(self.root.glob(SEGMENT_PREFIX + "*" + SEGMENT_SUFFIX))
            if not segs:
                return {"docs": {}, "postings": {}, "deletes": [], "seq": 0}
            return self._merge([p.name for p in segs])
        self.seq = int(manifest.get("seq", 0))
        self.segments = list(manifest.get("segments", []))
        return self._merge(self.segments)

    def _read_manifest(self) -> dict[str, Any] | None:
        path = self.root / MANIFEST_NAME
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            return None
        return data

    def _merge(self, segments: list[str]) -> dict[str, Any]:
        docs: dict[str, Any] = {}
        postings: dict[str, dict[str, list[int]]] = {}
        deletes: list[str] = []
        seq = 0
        ordered = sorted(segments)
        for name in ordered:
            path = self.root / name
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    seg = json.load(f)
            except (OSError, ValueError):
                continue  # skip corrupt segment, keep the rest
            seq = max(seq, int(seg.get("seq", 0)))
            for doc_id, record in seg.get("docs", {}).items():
                docs[str(doc_id)] = record
            for key, plist in seg.get("postings", {}).items():
                slot = postings.setdefault(key, {})
                for doc_id, positions in plist.items():
                    slot[str(doc_id)] = list(positions)
            for doc_id in seg.get("deletes", []):
                doc_id = str(doc_id)
                if doc_id not in deletes:
                    deletes.append(doc_id)
                docs.pop(doc_id, None)
        if deletes:
            dead = set(deletes)
            for key in list(postings.keys()):
                slot = postings[key]
                for doc_id in [d for d in slot if d in dead]:
                    del slot[doc_id]
                if not slot:
                    del postings[key]
        self.seq = max(self.seq, seq)
        if ordered:
            self.segments = [n for n in ordered if (self.root / n).exists()]
        return {"docs": docs, "postings": postings, "deletes": deletes,
                "seq": self.seq}

    # -- commit --------------------------------------------------------
    def commit_state(self, docs: dict[str, Any],
                     postings: dict[str, dict[str, list[int]]],
                     deletes: list[str] | set[str]) -> int:
        """Persist a full snapshot as a new segment; returns new seq."""
        self.seq += 1
        seg_name = f"{SEGMENT_PREFIX}{self.seq:05d}{SEGMENT_SUFFIX}"
        segment = {
            "version": STORE_VERSION,
            "seq": self.seq,
            "docs": docs,
            "postings": postings,
            "deletes": sorted(str(d) for d in deletes),
        }
        _atomic_write_json(self.root / seg_name, segment)
        manifest = {"version": STORE_VERSION, "seq": self.seq,
                    "segments": [seg_name]}
        _atomic_write_json(self.root / MANIFEST_NAME, manifest)
        self.segments = [seg_name]
        self._collect_garbage(keep={seg_name})
        return self.seq

    def _collect_garbage(self, keep: set[str]) -> None:
        for path in self.root.glob(SEGMENT_PREFIX + "*" + SEGMENT_SUFFIX):
            if path.name not in keep:
                try:
                    path.unlink()
                except OSError:
                    pass
        self._clean_tmp_orphans()
        _fsync_dir(self.root)

    def _clean_tmp_orphans(self) -> None:
        for path in self.root.glob(".*" + TMP_SUFFIX):
            try:
                path.unlink()
            except OSError:
                pass

