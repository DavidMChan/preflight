"""Coalescing lookups for hosts that ask for requests to be spaced out.

arXiv's API asks for one request every three seconds over a single connection,
and DBLP's SPARQL endpoint asks for a ten-second crawl delay. Looking references
up one at a time against either would take minutes. Both accept many lookups in
one request, so lookups are queued here and sent together: one worker per host,
one request in flight, and the host's spacing between requests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from .core import Candidate

#: Fetches one batch of keys of a single kind. Returns results by key; a key
#: absent from the result was looked up and not found. Raising means the lookup
#: itself failed, which is reported as "unknown", never as "not found".
BatchFetch = Callable[[str, list[str]], Awaitable[dict[str, list[Candidate]]]]


class Batcher:
    def __init__(self, fetch: BatchFetch, *, interval: float, size: int, window: float = 0.3) -> None:
        self._fetch = fetch
        self.interval = interval
        self.size = size
        self.window = window
        self._futures: dict[tuple[str, str], asyncio.Future[list[Candidate]]] = {}
        self._pending: list[tuple[str, str]] = []
        self._worker: asyncio.Task[None] | None = None
        self._last = 0.0
        self.requests = 0

    def submit(self, kind: str, key: str) -> asyncio.Future[list[Candidate]]:
        """Queue a lookup, or join the one already queued for the same key."""
        slot = (kind, key)
        future = self._futures.get(slot)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._futures[slot] = future
            self._pending.append(slot)
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run())
        return future

    async def get(self, kind: str, key: str) -> list[Candidate]:
        return await asyncio.shield(self.submit(kind, key))

    async def _run(self) -> None:
        await asyncio.sleep(self.window)         # let concurrent lookups join the batch
        while self._pending:
            kind = self._pending[0][0]
            batch = [slot for slot in self._pending if slot[0] == kind][: self.size]
            for slot in batch:
                self._pending.remove(slot)
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            keys = [key for _, key in batch]
            try:
                self.requests += 1
                results = await self._fetch(kind, keys)
            except Exception as exc:          # a failed batch is unknown, never "not found"
                for slot in batch:
                    future = self._futures[slot]
                    if not future.done():
                        future.set_exception(LookupError(str(exc) or type(exc).__name__))
            else:
                for slot in batch:
                    future = self._futures[slot]
                    if not future.done():
                        future.set_result(results.get(slot[1], []))
            finally:
                self._last = time.monotonic()

    def close(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        for future in self._futures.values():
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()            # mark retrieved, so asyncio does not warn
