"""Tiny disk cache: one JSON file per request."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(os.environ.get("PITWALL_CACHE", Path.home() / ".pitwall_cache"))


class DiskCache:
    """Store and fetch JSON-serialisable data on disk, keyed by a string."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def get(self, key: str, max_age: float | None = None) -> Any | None:
        """Return cached data, or None if missing, corrupt or older than max_age seconds."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # treat a broken file as a cache miss
        if max_age is not None and time.time() - entry["saved_at"] > max_age:
            return None
        return entry["data"]

    def set(self, key: str, data: Any) -> None:
        """Save data atomically (write a temp file, then rename it)."""
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        entry = {"key": key, "saved_at": time.time(), "data": data}
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> int:
        """Delete every cached file; returns how many were removed."""
        files = list(self.directory.glob("*.json"))
        for f in files:
            f.unlink()
        return len(files)
