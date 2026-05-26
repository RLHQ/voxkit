"""Structured LLM provider boundary.

The real proofread/translation stages need an LLM dependency, but the pipeline
must remain testable without network access. This module defines the narrow
provider interface plus a deterministic mock provider that validates structured
outputs with Pydantic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, Sequence, TypeVar

from pydantic import TypeAdapter, ValidationError

__all__ = [
    "LLMProviderError",
    "LLMProviderUnavailable",
    "LLMSchemaError",
    "LLMUsage",
    "LLMResult",
    "StructuredLLMProvider",
    "MockStructuredLLMProvider",
    "complete_structured_with_retry",
    "validate_structured_output",
]


T = TypeVar("T")
OutputFactory = Callable[[Sequence[dict[str, Any]], dict[str, Any], int], Any]


class LLMProviderError(RuntimeError):
    """Base class for structured LLM provider failures."""


class LLMProviderUnavailable(LLMProviderError):
    """Raised when a provider cannot be used in the current environment."""


class LLMSchemaError(LLMProviderError):
    """Raised when provider output does not match the requested schema."""

    def __init__(self, message: str, *, raw_output: Any, cause: Exception | None = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.cause = cause


@dataclass(frozen=True)
class LLMUsage:
    """Token/cost audit surface shared by real and mock providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    """Structured completion result with audit metadata."""

    provider: str
    model: str
    request_id: str
    parsed: Any
    raw_output: Any
    usage: LLMUsage
    elapsed_secs: float
    attempts: int = 1


class StructuredLLMProvider(Protocol):
    """Minimal interface used by proofread/translation stages."""

    provider: str
    model: str

    def complete_structured(
        self,
        batch: Sequence[dict[str, Any]],
        schema: Any,
        *,
        params: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Return a schema-validated structured completion for ``batch``."""


def validate_structured_output(raw_output: Any, schema: Any) -> Any:
    """Validate provider output against a Pydantic-compatible schema.

    ``schema`` can be a ``BaseModel`` subclass, a ``list[Model]`` style type, or
    any type accepted by :class:`pydantic.TypeAdapter`.
    """
    value = raw_output
    if isinstance(raw_output, str):
        try:
            value = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise LLMSchemaError(
                "provider output is not valid JSON",
                raw_output=raw_output,
                cause=exc,
            ) from exc

    try:
        return TypeAdapter(schema).validate_python(value)
    except ValidationError as exc:
        raise LLMSchemaError(
            "provider output failed schema validation",
            raw_output=raw_output,
            cause=exc,
        ) from exc


class MockStructuredLLMProvider:
    """Deterministic provider for local tests and offline pipeline work.

    ``outputs`` may be a list/tuple of raw outputs or a callable. Each call is
    validated against the requested schema exactly like a real provider result.
    """

    def __init__(
        self,
        outputs: Sequence[Any] | OutputFactory,
        *,
        provider: str = "mock",
        model: str = "mock-structured-llm",
        unavailable: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self._outputs = outputs
        self._unavailable = unavailable
        self.calls = 0

    def complete_structured(
        self,
        batch: Sequence[dict[str, Any]],
        schema: Any,
        *,
        params: dict[str, Any] | None = None,
    ) -> LLMResult:
        if self._unavailable:
            raise LLMProviderUnavailable(f"{self.provider} provider is unavailable")

        self.calls += 1
        started = time.monotonic()
        params = params or {}
        raw_output = self._next_output(batch, params)
        parsed = validate_structured_output(raw_output, schema)
        elapsed = time.monotonic() - started
        usage = LLMUsage(
            prompt_tokens=_estimate_tokens(batch),
            completion_tokens=_estimate_tokens(raw_output),
        )
        usage = replace(
            usage,
            total_tokens=usage.prompt_tokens + usage.completion_tokens,
        )
        return LLMResult(
            provider=self.provider,
            model=self.model,
            request_id=f"{self.provider}-{self.calls:06d}",
            parsed=parsed,
            raw_output=raw_output,
            usage=usage,
            elapsed_secs=elapsed,
        )

    def _next_output(
        self,
        batch: Sequence[dict[str, Any]],
        params: dict[str, Any],
    ) -> Any:
        if callable(self._outputs):
            return self._outputs(batch, params, self.calls)

        index = self.calls - 1
        if index >= len(self._outputs):
            raise LLMProviderError("mock provider has no output for this call")
        return self._outputs[index]


def complete_structured_with_retry(
    provider: StructuredLLMProvider,
    batch: Sequence[dict[str, Any]],
    schema: Any,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = 2,
) -> LLMResult:
    """Run a structured completion, retrying schema failures only."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_schema_error: LLMSchemaError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = provider.complete_structured(batch, schema, params=params)
            return replace(result, attempts=attempt)
        except LLMSchemaError as exc:
            last_schema_error = exc

    assert last_schema_error is not None
    raise last_schema_error


def _estimate_tokens(value: Any) -> int:
    """Cheap deterministic token estimate for mock audit metadata."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, len(text) // 4)
