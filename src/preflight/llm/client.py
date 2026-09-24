"""A small, deliberately boring OpenAI wrapper.

Only the fuzzy checks talk to a model. Everything geometric stays deterministic,
so an API outage degrades the report rather than invalidating it.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMUnavailable(LLMError):
    """No API key, or the openai package is not installed."""


def _extract(response: Any, model: str) -> str:
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise LLMError(f"{model} returned an empty response")
    return content


@dataclass(slots=True)
class LLMClient:
    model: str = "gpt-5.6-luna"
    api_key: str | None = None
    timeout: float = 120.0
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self) -> Any:
        if not self.api_key:
            raise LLMUnavailable(
                "No OpenAI API key. Put OPENAI_API_KEY in your .env, or pass --openai-key."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("The `openai` package is not installed.") from exc
        return OpenAI(api_key=self.api_key, timeout=self.timeout)

    def complete(self, system: str, user: str) -> str:
        client = self._client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            raise LLMError(f"{self.model}: {type(exc).__name__}: {exc}") from exc
        return _extract(response, self.model)

    def json(self, system: str, user: str, *, schema_hint: str = "") -> dict[str, Any]:
        """Ask for JSON and parse it, tolerating fenced code blocks."""
        instruction = system
        if schema_hint:
            instruction += (
                "\n\nRespond with a single JSON object and nothing else. "
                f"It must match this shape:\n{schema_hint}"
            )
        raw = self.complete(instruction, user)
        return _parse_json(raw, self.model)


@dataclass(slots=True)
class AsyncLLMClient:
    """The async twin of :class:`LLMClient`.

    The checks are almost entirely I/O: nineteen independent model calls that
    each wait on the network. Running them on the async client turns the
    checklist from a serial minute-and-a-half into one round trip's worth of
    wall clock.
    """

    model: str = "gpt-5.6-luna"
    api_key: str | None = None
    timeout: float = 120.0
    max_output_tokens: int = 4096
    max_concurrency: int = 8
    # The SDK retries 429s and 5xx with exponential backoff, honouring the
    # server's retry-after. Its default of two gives up quickly when a
    # bibliography's parse and web-search calls share one rate budget.
    max_retries: int = 5
    _sem_obj: Any = None
    _client_obj: Any = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self) -> Any:
        """One client for the whole run, so connections are pooled across calls."""
        if self._client_obj is not None:
            return self._client_obj
        if not self.api_key:
            raise LLMUnavailable(
                "No OpenAI API key. Put OPENAI_API_KEY in your .env, or pass --openai-key."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("The `openai` package is not installed.") from exc
        object.__setattr__(self, "_client_obj", AsyncOpenAI(api_key=self.api_key, timeout=self.timeout,
                                                            max_retries=self.max_retries))
        return self._client_obj

    async def aclose(self) -> None:
        if self._client_obj is not None:
            try:
                await self._client_obj.close()
            except Exception:  # pragma: no cover - shutdown is best effort
                pass
            object.__setattr__(self, "_client_obj", None)

    @property
    def _semaphore(self) -> asyncio.Semaphore:
        # Created lazily so the client can be built outside a running loop.
        existing = getattr(self, "_sem_obj", None)
        if existing is None:
            existing = asyncio.Semaphore(self.max_concurrency)
            object.__setattr__(self, "_sem_obj", existing)
        return existing

    async def complete(self, system: str, user: str) -> str:
        client = self._client()
        async with self._semaphore:
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                raise LLMError(f"{self.model}: {type(exc).__name__}: {exc}") from exc
        return _extract(response, self.model)

    async def json(self, system: str, user: str, *, schema_hint: str = "") -> dict[str, Any]:
        instruction = system
        if schema_hint:
            instruction += (
                "\n\nRespond with a single JSON object and nothing else. "
                f"It must match this shape:\n{schema_hint}"
            )
        return _parse_json(await self.complete(instruction, user), self.model)

    async def structured(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any],
        name: str = "result",
    ) -> dict[str, Any]:
        """Schema-enforced JSON via the Responses API.

        Unlike :meth:`json`, the shape is guaranteed by the provider rather than
        requested in prose, so a parse failure cannot silently become a missing
        field — which for reference parsing would become a false accusation.
        """
        client = self._client()
        async with self._semaphore:
            try:
                response = await client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text={"format": {"type": "json_schema", "name": name,
                                     "schema": schema, "strict": True}},
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                raise LLMError(f"{self.model}: {type(exc).__name__}: {exc}") from exc
        return _parse_json(response.output_text or "", self.model)

    async def web_search(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any],
        name: str = "result",
        allowed_domains: list[str] | None = None,
        context_size: str = "low",
    ) -> tuple[dict[str, Any], list[str]]:
        """Search the live web, then answer in a fixed schema.

        Returns (parsed answer, cited URLs). This is the tier a database-only
        checker has no answer for: a citation that is real but simply not indexed
        anywhere, which is the single largest source of false accusations.
        """
        client = self._client()
        tool: dict[str, Any] = {"type": "web_search", "search_context_size": context_size}
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains[:100]}
        async with self._semaphore:
            try:
                response = await client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tools=[tool],
                    text={"format": {"type": "json_schema", "name": name,
                                     "schema": schema, "strict": True}},
                    max_output_tokens=self.max_output_tokens,
                )
            except Exception as exc:
                raise LLMError(f"{self.model}: {type(exc).__name__}: {exc}") from exc

        citations: list[str] = []
        for item in getattr(response, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                for annotation in getattr(block, "annotations", []) or []:
                    url = getattr(annotation, "url", None)
                    if url and url not in citations:
                        citations.append(url)
        return _parse_json(response.output_text or "", self.model), citations


def _parse_json(raw: str, model: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"{model} did not return JSON: {raw[:200]}") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"{model} returned malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"{model} returned {type(parsed).__name__}, expected a JSON object")
    return parsed
