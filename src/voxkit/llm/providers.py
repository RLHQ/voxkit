"""LLM provider 注册表。

每个 provider 由 :class:`ProviderSpec` 描述：base_url + api_key_env + 默认模型
+ 上下文上限 + 重试策略。voxkit 通过 OpenAI 兼容协议统一调用，所以这里只需
要登记入口元信息，不需要 per-provider 适配代码。

首批只登记 ``deepseek``；未来 vLLM/Ollama/OpenAI/Qwen 都按同一形状追加即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from voxkit.llm.errors import LLMError


@dataclass(frozen=True)
class RetryPolicy:
    """指数退避参数。``max_attempts`` 含首次请求，例如 5 = 1 次主调 + 最多 4 次重试。"""

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


@dataclass(frozen=True)
class ProviderSpec:
    """单个 OpenAI 兼容 provider 的接入参数。

    ``api_key_env`` 是环境变量名（非 key 本身），credentials 始终走 env，
    避免误把 key 序列化进 manifest 或日志。
    """

    name: str
    base_url: str
    api_key_env: str
    default_model: str
    max_context_tokens: int
    retry: RetryPolicy = field(default_factory=RetryPolicy)


# 注册表：首批仅 DeepSeek。未来追加 provider 时仅需在此处加一行 + 写注释。
PROVIDERS: Dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        max_context_tokens=64000,
    ),
}


def get_provider(name: str) -> ProviderSpec:
    """按名查 provider。命中返回 spec；未命中抛 :class:`LLMError`，错误信息里
    列出当前已注册的 provider，避免拼写错误调试成本。
    """
    spec = PROVIDERS.get(name)
    if spec is None:
        available = ", ".join(sorted(PROVIDERS.keys())) or "(none)"
        raise LLMError(
            f"unknown LLM provider {name!r}; available: {available}"
        )
    return spec


__all__ = ["RetryPolicy", "ProviderSpec", "PROVIDERS", "get_provider"]
