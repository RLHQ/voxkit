"""Tests for the structured LLM provider boundary."""

from __future__ import annotations

from typing import Any, Sequence

import pytest
from pydantic import BaseModel

from voxkit.core.llm_provider import (
    LLMProviderError,
    LLMProviderUnavailable,
    LLMSchemaError,
    MockStructuredLLMProvider,
    complete_structured_with_retry,
    validate_structured_output,
)


class CueEdit(BaseModel):
    cue_id: str
    corrected_text: str


class CueEditBatch(BaseModel):
    edits: list[CueEdit]


def test_validate_structured_output_accepts_dict_and_json_string() -> None:
    data = {"edits": [{"cue_id": "cue_000001", "corrected_text": "Hello"}]}
    parsed = validate_structured_output(data, CueEditBatch)
    assert parsed.edits[0].cue_id == "cue_000001"

    parsed_json = validate_structured_output(
        '{"edits":[{"cue_id":"cue_000002","corrected_text":"Hi"}]}',
        CueEditBatch,
    )
    assert parsed_json.edits[0].corrected_text == "Hi"


def test_validate_structured_output_rejects_invalid_json() -> None:
    with pytest.raises(LLMSchemaError, match="not valid JSON") as excinfo:
        validate_structured_output("{nope", CueEditBatch)
    assert excinfo.value.raw_output == "{nope"


def test_mock_provider_returns_audited_result() -> None:
    provider = MockStructuredLLMProvider(
        [{"edits": [{"cue_id": "cue_000001", "corrected_text": "Hello"}]}],
        model="mock-proofread",
    )

    result = provider.complete_structured(
        [{"cue_id": "cue_000001", "text": "helo"}],
        CueEditBatch,
        params={"editLevel": "light"},
    )

    assert result.provider == "mock"
    assert result.model == "mock-proofread"
    assert result.request_id == "mock-000001"
    assert result.parsed.edits[0].corrected_text == "Hello"
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == (
        result.usage.prompt_tokens + result.usage.completion_tokens
    )
    assert result.elapsed_secs >= 0.0


def test_mock_provider_callable_receives_batch_params_and_call_number() -> None:
    seen: dict[str, Any] = {}

    def output_factory(
        batch: Sequence[dict[str, Any]],
        params: dict[str, Any],
        call_number: int,
    ) -> dict[str, Any]:
        seen["batch"] = batch
        seen["params"] = params
        seen["call_number"] = call_number
        return {
            "edits": [
                {
                    "cue_id": batch[0]["cue_id"],
                    "corrected_text": params["prefix"] + batch[0]["text"],
                }
            ]
        }

    provider = MockStructuredLLMProvider(output_factory)
    result = provider.complete_structured(
        [{"cue_id": "cue_000010", "text": "text"}],
        CueEditBatch,
        params={"prefix": "ok: "},
    )

    assert seen["call_number"] == 1
    assert seen["params"] == {"prefix": "ok: "}
    assert result.parsed.edits[0].corrected_text == "ok: text"


def test_complete_structured_with_retry_retries_schema_errors_only() -> None:
    provider = MockStructuredLLMProvider(
        [
            {"edits": [{"cue_id": "cue_000001"}]},
            {"edits": [{"cue_id": "cue_000001", "corrected_text": "fixed"}]},
        ]
    )

    result = complete_structured_with_retry(
        provider,
        [{"cue_id": "cue_000001", "text": "fixd"}],
        CueEditBatch,
        max_attempts=2,
    )

    assert provider.calls == 2
    assert result.attempts == 2
    assert result.parsed.edits[0].corrected_text == "fixed"


def test_complete_structured_with_retry_surfaces_final_schema_error() -> None:
    provider = MockStructuredLLMProvider(
        [
            {"edits": [{"cue_id": "cue_000001"}]},
            {"edits": [{"cue_id": "cue_000001"}]},
        ]
    )

    with pytest.raises(LLMSchemaError, match="schema validation"):
        complete_structured_with_retry(
            provider,
            [{"cue_id": "cue_000001", "text": "fixd"}],
            CueEditBatch,
            max_attempts=2,
        )
    assert provider.calls == 2


def test_provider_unavailable_is_not_retried_or_switched() -> None:
    provider = MockStructuredLLMProvider([], unavailable=True)

    with pytest.raises(LLMProviderUnavailable):
        complete_structured_with_retry(provider, [], CueEditBatch, max_attempts=3)
    assert provider.calls == 0


def test_mock_provider_exhaustion_is_explicit() -> None:
    provider = MockStructuredLLMProvider([])

    with pytest.raises(LLMProviderError, match="no output"):
        provider.complete_structured([], CueEditBatch)
