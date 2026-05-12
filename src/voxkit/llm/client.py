"""OpenAI 兼容 chat/completions 客户端。

最小可执行：sync httpx + 手写指数退避，不引 tenacity/openai/litellm。
调用形态：

    with LLMClient("deepseek") as llm:
        result = llm.chat(
            messages=[{"role": "user", "content": "..."}],
            response_format="json_object",
            temperature=0.0,
        )
        print(result.text, result.prompt_tokens, result.completion_tokens)

错误分流见 :mod:`voxkit.llm.errors`：超时/5xx 重试由 client 内部托管，耗尽
后抛 ``LLMTimeout``/``LLMError``；429 透传 ``LLMRateLimit``（含 Retry-After）；
内容拒答抛 ``LLMRefusal`` 不重试。
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx

from voxkit.llm.errors import (
    LLMError,
    LLMRateLimit,
    LLMRefusal,
    LLMTimeout,
)
from voxkit.llm.providers import ProviderSpec, get_provider


@dataclass
class ChatResult:
    """``chat`` 的返回值：解析后的文本 + token 统计 + 原始响应（审计用）。"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)


# 内容拒答的启发式关键词；DeepSeek 等 provider 偶发返回 4xx + 政策文案，
# 这里粗匹配避免把拒答当 transient 错误重试浪费 quota。
_REFUSAL_KEYWORDS = ("policy", "refuse", "content")


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """解析 Retry-After header（仅秒数形式；HTTP-date 形式忽略即可，provider
    用得极少且 voxkit 不需要精度）。
    """
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


class LLMClient:
    """同步 OpenAI 兼容客户端。

    线程模型：单实例非线程安全（内部 ``httpx.Client`` 复用）。voxkit 当前
    pipeline 全 sync 单线程，不暴露 async 形态。
    """

    def __init__(
        self,
        provider: Union[str, ProviderSpec],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        # provider 可传字符串名或已构造的 spec；后者用于测试注入快速 retry 策略。
        spec = get_provider(provider) if isinstance(provider, str) else provider
        self._spec: ProviderSpec = spec
        self._model: str = model or spec.default_model

        # credentials 始终从 env 取，避免序列化泄漏；显式 ``api_key`` 仅供测试覆盖。
        key = api_key if api_key is not None else os.environ.get(spec.api_key_env)
        if not key:
            raise LLMError(
                f"env {spec.api_key_env} not set; export it before calling LLMClient"
            )
        self._api_key: str = key
        self._timeout_s: float = timeout_s
        self._client: httpx.Client = httpx.Client(timeout=timeout_s)

    # ── context manager ────────────────────────────────────────────────────
    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """释放底层 httpx.Client 的连接池。"""
        self._client.close()

    # ── public API ─────────────────────────────────────────────────────────
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: str = "json_object",
        temperature: float = 0.0,
        **extra: Any,
    ) -> ChatResult:
        """调用 ``POST {base_url}/v1/chat/completions`` 并解析。

        ``extra`` 透传到 body（例如 ``max_tokens`` / ``top_p``），方便上层
        在不修改 client 的前提下传 provider-specific 参数。
        """
        url = f"{self._spec.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": {"type": response_format},
            "temperature": temperature,
        }
        # extra 放后面是有意：允许调用方覆盖默认 model/temperature（极少用，
        # 但调试时方便）。
        body.update(extra)

        return self._request_with_retry(url, headers=headers, body=body)

    # ── internals ──────────────────────────────────────────────────────────
    def _request_with_retry(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        body: Dict[str, Any],
    ) -> ChatResult:
        retry = self._spec.retry
        last_exc: Optional[BaseException] = None
        last_status: Optional[int] = None

        for attempt in range(1, retry.max_attempts + 1):
            try:
                resp = self._client.post(url, headers=headers, json=body)
            except httpx.TimeoutException as e:
                last_exc = e
                last_status = None
                self._sleep_backoff(attempt)
                continue
            except httpx.HTTPError as e:
                # 连接重置 / DNS 等可恢复 transport 错误：和 5xx 一类处理。
                last_exc = e
                last_status = None
                self._sleep_backoff(attempt)
                continue

            status = resp.status_code

            if 200 <= status < 300:
                return self._parse_success(resp)

            if status == 429:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                # 限流不在 client 内部重试：让上层根据 retry_after 决定排队节奏。
                raise LLMRateLimit(
                    f"rate limited by {self._spec.name}",
                    retry_after_secs=retry_after,
                )

            if 400 <= status < 500:
                # 区分拒答 vs 普通 4xx：拒答看响应文本里的政策类关键词。
                text = resp.text or ""
                if any(kw in text.lower() for kw in _REFUSAL_KEYWORDS):
                    raise LLMRefusal(
                        f"{self._spec.name} refused: HTTP {status}: {text[:200]}"
                    )
                raise LLMError(f"HTTP {status} from {self._spec.name}: {text[:200]}")

            # 5xx：可恢复，进入退避。
            last_exc = None
            last_status = status
            self._sleep_backoff(attempt)

        # 退避耗尽：超时类抛 LLMTimeout，其它（5xx / transport error）抛 LLMError。
        if isinstance(last_exc, httpx.TimeoutException):
            raise LLMTimeout(
                f"timed out talking to {self._spec.name} after {retry.max_attempts} attempts"
            ) from last_exc
        if last_status is not None and 500 <= last_status < 600:
            raise LLMError(
                f"{self._spec.name} returned HTTP {last_status} after "
                f"{retry.max_attempts} attempts"
            )
        raise LLMError(
            f"failed to call {self._spec.name} after {retry.max_attempts} attempts: "
            f"{last_exc!r}"
        )

    def _sleep_backoff(self, attempt: int) -> None:
        """指数退避 + 抖动；最后一次尝试后不睡（外层会抛错）。"""
        retry = self._spec.retry
        if attempt >= retry.max_attempts:
            return
        # 2^(attempt-1) * base，capped；jitter 50%-100% 区间，避免雷鸣群惊。
        raw = retry.base_delay_s * (2 ** (attempt - 1))
        capped = min(raw, retry.max_delay_s)
        delay = capped * (0.5 + random.random() * 0.5)
        time.sleep(delay)

    def _parse_success(self, resp: httpx.Response) -> ChatResult:
        try:
            data = resp.json()
        except ValueError as e:
            raise LLMError(f"non-JSON 200 response from {self._spec.name}") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(
                f"unexpected response shape from {self._spec.name}: {data!r}"
            ) from e

        usage = data.get("usage") or {}
        return ChatResult(
            text=text if isinstance(text, str) else str(text),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=str(data.get("model", self._model)),
            raw=data,
        )


__all__ = ["ChatResult", "LLMClient"]
