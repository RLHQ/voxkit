"""LLMClient 行为测试。

为了让重试相关测试在毫秒级跑完，统一使用注入式 ``ProviderSpec``，把
``RetryPolicy`` 调到极小的 base_delay / max_delay。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from voxkit.llm.client import LLMClient
from voxkit.llm.errors import LLMError, LLMRateLimit, LLMRefusal
from voxkit.llm.providers import ProviderSpec, RetryPolicy


def _fast_spec(max_attempts: int = 2) -> ProviderSpec:
    """生成测试用 spec：快速退避 + 假 base_url + 假 env 名。"""
    return ProviderSpec(
        name="testprov",
        base_url="https://example.test",
        api_key_env="TESTPROV_KEY_NEVER_SET",  # 不应被实际读取
        default_model="test-model",
        max_context_tokens=8000,
        retry=RetryPolicy(
            max_attempts=max_attempts,
            base_delay_s=0.001,
            max_delay_s=0.002,
        ),
    )


# ── happy path ─────────────────────────────────────────────────────────────


@respx.mock
def test_chat_success_returns_parsed_result() -> None:
    payload = {
        "model": "test-model",
        "choices": [
            {"message": {"role": "assistant", "content": '{"ok": true}'}}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }
    route = respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with LLMClient(_fast_spec(), api_key="sk-fake") as llm:
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])

    assert route.called
    assert result.text == '{"ok": true}'
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5
    assert result.model == "test-model"
    assert result.raw["usage"]["prompt_tokens"] == 12


# ── 429 limit ──────────────────────────────────────────────────────────────


@respx.mock
def test_rate_limit_parses_retry_after() -> None:
    respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "7.5"}, text="slow down")
    )
    with LLMClient(_fast_spec(), api_key="sk-fake") as llm:
        with pytest.raises(LLMRateLimit) as exc:
            llm.chat(messages=[{"role": "user", "content": "x"}])
    assert exc.value.retry_after_secs == pytest.approx(7.5)


@respx.mock
def test_rate_limit_without_header_has_none_retry_after() -> None:
    respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="too many")
    )
    with LLMClient(_fast_spec(), api_key="sk-fake") as llm:
        with pytest.raises(LLMRateLimit) as exc:
            llm.chat(messages=[{"role": "user", "content": "x"}])
    assert exc.value.retry_after_secs is None


# ── 5xx retry ──────────────────────────────────────────────────────────────


@respx.mock
def test_5xx_retries_then_succeeds() -> None:
    success = {
        "model": "test-model",
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    route = respx.post("https://example.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=success),
        ]
    )
    with LLMClient(_fast_spec(max_attempts=2), api_key="sk-fake") as llm:
        result = llm.chat(messages=[{"role": "user", "content": "x"}])
    assert route.call_count == 2
    assert result.text == "ok"


@respx.mock
def test_5xx_exhausted_raises_llm_error() -> None:
    respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="still bad")
    )
    with LLMClient(_fast_spec(max_attempts=2), api_key="sk-fake") as llm:
        with pytest.raises(LLMError) as exc:
            llm.chat(messages=[{"role": "user", "content": "x"}])
    # 错误消息应当点出 status code 与尝试次数。
    msg = str(exc.value)
    assert "503" in msg
    assert "2" in msg


# ── env var missing ────────────────────────────────────────────────────────


def test_missing_env_var_raises_with_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _fast_spec()
    monkeypatch.delenv(spec.api_key_env, raising=False)
    with pytest.raises(LLMError) as exc:
        LLMClient(spec)  # 不传 api_key，应当读 env 失败
    assert spec.api_key_env in str(exc.value)


# ── context manager closes client ──────────────────────────────────────────


def test_context_manager_closes_underlying_client() -> None:
    llm = LLMClient(_fast_spec(), api_key="sk-fake")
    underlying = llm._client  # 受测私有引用：验证 close 语义
    assert not underlying.is_closed
    with llm:
        pass
    assert underlying.is_closed


# ── content policy refusal ─────────────────────────────────────────────────


@respx.mock
def test_4xx_with_content_policy_keyword_raises_refusal() -> None:
    respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            text='{"error":"request violates content policy"}',
        )
    )
    with LLMClient(_fast_spec(), api_key="sk-fake") as llm:
        with pytest.raises(LLMRefusal):
            llm.chat(messages=[{"role": "user", "content": "x"}])
