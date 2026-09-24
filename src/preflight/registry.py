"""Check registration.

Checks are plain functions decorated with :func:`register`. The runner asks the
registry which checks apply to a profile, so adding a check never means editing
a dispatch table.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .context import CheckContext
from .models import LLM, NETWORK, Finding, Severity, mode_label

CheckFn = Callable[[CheckContext], Any]
"""``(ctx) -> Finding | Iterable[Finding] | None``, or a coroutine returning one."""


@dataclass(slots=True)
class Check:
    id: str
    title: str
    category: str
    module: str
    fn: CheckFn
    description: str = ""
    requires: tuple[str, ...] = ()      # settings flags that must be truthy
    order: int = 100
    aggregate: bool = False             # runs last; reads what the others left behind
    concurrent: bool | None = None      # None = auto (True for async checks)
    uses: tuple[str, ...] = ()          # NETWORK, LLM; empty for a check that reads only the PDF

    @property
    def mode(self) -> str:
        return mode_label(self.uses)

    def uses_in(self, ctx: CheckContext) -> tuple[str, ...]:
        """What this check can have reached in this run.

        A model the check can do without (the reference checker parses and
        searches with one when it may) is not counted when the run has none.
        """
        has_model = ctx.settings.enable_llm and bool(ctx.settings.openai_api_key)
        if LLM in self.uses and "enable_llm" not in self.requires and not has_model:
            return tuple(u for u in self.uses if u != LLM)
        return self.uses

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.fn)

    @property
    def runs_concurrently(self) -> bool:
        """Whether this check should overlap with others.

        Async checks do by default. A synchronous check that blocks on I/O (the
        bibliographic lookups, for instance) can opt in and will be handed to a
        worker thread.
        """
        return self.is_async if self.concurrent is None else self.concurrent

    def applies(self, ctx: CheckContext) -> bool:
        if not ctx.profile.module_enabled(self.module):
            return False
        if not ctx.profile.check_enabled(self.id):
            return False
        return all(getattr(ctx.settings, flag, False) for flag in self.requires)

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Run synchronously. Only safe outside a running event loop."""
        if self.is_async:
            return asyncio.run(self.arun(ctx))
        try:
            result = self.fn(ctx)
        except Exception as exc:  # a broken check must not sink the report
            return [self._crash(ctx, exc)]
        return self._finish(ctx, result)

    async def arun(self, ctx: CheckContext) -> list[Finding]:
        """Run inside an event loop, off-thread when the check is blocking."""
        try:
            if self.is_async:
                result = await self.fn(ctx)
            elif self.runs_concurrently:
                result = await asyncio.to_thread(self.fn, ctx)
            else:
                result = self.fn(ctx)
        except Exception as exc:
            return [self._crash(ctx, exc)]
        return self._finish(ctx, result)

    def _crash(self, ctx: CheckContext, exc: Exception) -> Finding:
        return Finding(
            check_id=self.id,
            title=self.title,
            severity=Severity.SKIPPED,
            category=self.category,
            message=f"Check failed to run: {type(exc).__name__}: {exc}",
            uses=self.uses_in(ctx),
        )

    def _finish(self, ctx: CheckContext, result: Any) -> list[Finding]:
        if result is None:
            return []
        findings = [result] if isinstance(result, Finding) else list(result)
        for f in findings:
            # A venue may soften or harden a shared check without forking it.
            f.severity = ctx.profile.severity_for(f.check_id, f.severity)
            # Every finding says whether it came from the PDF alone, a network
            # lookup or a model. A check that knows more precisely has said so.
            if f.uses is None:
                f.uses = self.uses_in(ctx)
        return findings


@dataclass
class Registry:
    checks: dict[str, Check] = field(default_factory=dict)

    def add(self, check: Check) -> None:
        if check.id in self.checks:
            raise ValueError(f"duplicate check id: {check.id}")
        self.checks[check.id] = check

    def __iter__(self) -> Iterator[Check]:
        return iter(sorted(self.checks.values(), key=lambda c: (c.order, c.id)))

    def __len__(self) -> int:
        return len(self.checks)

    def for_context(self, ctx: CheckContext) -> list[Check]:
        return [c for c in self if c.applies(ctx)]

    def modules(self) -> dict[str, list[Check]]:
        out: dict[str, list[Check]] = {}
        for c in self:
            out.setdefault(c.module, []).append(c)
        return out


REGISTRY = Registry()


def register(
    check_id: str,
    title: str,
    *,
    module: str,
    category: str = "general",
    description: str = "",
    requires: tuple[str, ...] = (),
    order: int = 100,
    aggregate: bool = False,
    concurrent: bool | None = None,
    uses: tuple[str, ...] | None = None,
) -> Callable[[CheckFn], CheckFn]:
    """Register a check as part of ``module`` (e.g. ``"acl.geometry"``).

    Profiles switch whole modules on and off, so a module is the unit of reuse
    between conferences.

    ``uses`` says what the check reaches beyond the PDF (:data:`~.models.NETWORK`,
    :data:`~.models.LLM`). Left out, it follows from ``requires``: a check that
    needs ``enable_llm`` calls a model, one that needs the reference checker
    makes network lookups, and anything else reads only the PDF.
    """

    def decorator(fn: CheckFn) -> CheckFn:
        REGISTRY.add(
            Check(
                id=check_id,
                title=title,
                category=category,
                module=module,
                fn=fn,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                requires=requires,
                order=order,
                aggregate=aggregate,
                concurrent=concurrent,
                uses=uses if uses is not None else _uses_from(requires),
            )
        )
        return fn

    return decorator


#: The settings flag a check requires, and what it therefore reaches.
_FLAG_USES = {"enable_llm": LLM, "enable_scores": LLM, "enable_refcheck": NETWORK,
              "enable_hallucinator": NETWORK}


def _uses_from(requires: tuple[str, ...]) -> tuple[str, ...]:
    reached = {_FLAG_USES[flag] for flag in requires if flag in _FLAG_USES}
    return tuple(u for u in (NETWORK, LLM) if u in reached)


def load_builtin_checks() -> Registry:
    """Import every check module so decorators populate the registry."""
    import importlib
    import pkgutil

    from . import checks

    for info in pkgutil.iter_modules(checks.__path__):
        importlib.import_module(f"{checks.__name__}.{info.name}")
    return REGISTRY


def describe() -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "title": c.title,
            "module": c.module,
            "category": c.category,
            "description": c.description,
            "requires": list(c.requires),
            "uses": list(c.uses),
            "mode": c.mode,
        }
        for c in REGISTRY
    ]
