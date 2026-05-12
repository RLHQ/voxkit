"""voxkit LLM 客户端 + provider 注册（OpenAI 兼容）。

模块结构：

* :mod:`voxkit.llm.client` — :class:`LLMClient` / :class:`ChatResult`
* :mod:`voxkit.llm.providers` — :class:`ProviderSpec` / :func:`get_provider`
* :mod:`voxkit.llm.errors` — 错误类型层级
* :mod:`voxkit.llm.prompts` — :func:`load_prompt` + 模板文件
"""

from __future__ import annotations

from voxkit.llm.client import ChatResult, LLMClient
from voxkit.llm.errors import (
    LLMError,
    LLMRateLimit,
    LLMRefusal,
    LLMSchemaError,
    LLMTimeout,
)
from voxkit.llm.prompts import load_prompt
from voxkit.llm.providers import ProviderSpec, get_provider

__all__ = [
    "LLMClient",
    "ChatResult",
    "ProviderSpec",
    "get_provider",
    "load_prompt",
    "LLMError",
    "LLMTimeout",
    "LLMRateLimit",
    "LLMSchemaError",
    "LLMRefusal",
]
