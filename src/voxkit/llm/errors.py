"""LLM 客户端错误类型层级。

设计意图：把传输层 / 协议层 / 内容层错误分开，让 proofread/translate 这类
上层可以做不同处置：

* :class:`LLMTimeout` / 5xx → 由 client 内部退避重试；耗尽后抛给调用方
* :class:`LLMRateLimit` → 调用方可读 ``retry_after_secs`` 决定排队策略
* :class:`LLMSchemaError` → JSON 解析或 Pydantic 校验失败，调用方走 repair 流程
* :class:`LLMRefusal` → provider 拒答（内容政策），**不重试**，标人工 review
"""

from __future__ import annotations

from typing import Optional


class LLMError(Exception):
    """所有 LLM 相关错误的基类。"""


class LLMTimeout(LLMError):
    """请求超时（含 client 重试耗尽后的 timeout 终态）。"""


class LLMRateLimit(LLMError):
    """HTTP 429 限流。``retry_after_secs`` 若 provider 在 Retry-After header
    给出，则透传；否则为 ``None``，由调用方自行决定退避。
    """

    def __init__(self, message: str = "", *, retry_after_secs: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_secs = retry_after_secs


class LLMSchemaError(LLMError):
    """LLM 返回的内容无法解析成期望 schema（JSON 残缺 / 字段缺失等）。
    ``raw_text`` 保留原始响应，供 repair 提示或人工排查使用。
    """

    def __init__(self, message: str = "", *, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


class LLMRefusal(LLMError):
    """Provider 以内容政策为由拒答；**不重试**。调用方应将批次标 needsHumanReview。"""


__all__ = [
    "LLMError",
    "LLMTimeout",
    "LLMRateLimit",
    "LLMSchemaError",
    "LLMRefusal",
]
