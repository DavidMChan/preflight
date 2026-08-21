"""A small SQLite cache for reference verdicts.

Verifying a bibliography is mostly network latency, and the same references
recur across drafts of the same paper, so a warm cache turns a minute into a
second. Positive and negative results get different lifetimes: "this exists" is
stable, "I could not find this" often just means the index had not caught up.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing, suppress
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(
    os.environ.get("PREFLIGHT_CACHE", "~/.cache/preflight")
).expanduser() / "refcheck.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    positive   INTEGER NOT NULL,
    stored_at  REAL NOT NULL
);
"""


class Store:
    """Best-effort persistence. Every failure degrades to "no cache"."""

    def __init__(
        self,
        path: Path | str | None = DEFAULT_PATH,
        positive_ttl: float = 30 * 86400,
        negative_ttl: float = 2 * 86400,
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.positive_ttl = positive_ttl
        self.negative_ttl = negative_ttl
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        if self.path is not None:
            with suppress(OSError, sqlite3.Error):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(self.path, timeout=5.0)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(_SCHEMA)
                self._conn.commit()

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def get(self, key: str) -> dict[str, Any] | None:
        if self._conn is None or not key:
            return None
        try:
            with closing(self._conn.execute(
                "SELECT payload, positive, stored_at FROM verdicts WHERE key = ?", (key,)
            )) as cursor:
                row = cursor.fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            self.misses += 1
            return None
        payload, positive, stored_at = row
        ttl = self.positive_ttl if positive else self.negative_ttl
        if time.time() - stored_at > ttl:
            self.misses += 1
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        self.hits += 1
        return data if isinstance(data, dict) else None

    def put(self, key: str, payload: dict[str, Any], positive: bool) -> None:
        if self._conn is None or not key:
            return
        with suppress(sqlite3.Error, TypeError, ValueError):
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts (key, payload, positive, stored_at) VALUES (?,?,?,?)",
                (key, json.dumps(payload), 1 if positive else 0, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            with suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
