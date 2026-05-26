"""LLM 价格表 + cost 估算。

中心化所有 (provider, model) → USD/M-token 的单价，让 proofread/translate
summary 和 ``--dry-run`` 都能算出"这批跑下来要烧多少美刀"。

设计原则：

  - **纯函数 + 静态数据**：不依赖 LLM client、不读盘、不发请求。`run_*`
    pipeline 调一调就能拿到 cost 字符串。
  - **未知 (provider, model) 不要硬错**：返回 ``None``，由调用方决定打不打
    "(unknown rate)" 提示——避免把"我没维护这个 model 的价格"升级成
    pipeline 错误。
  - **价格单位明确**：表里写 "USD per **million** tokens"，与 DeepSeek /
    OpenAI / Anthropic 公开页面单位一致，避免做 1000 / 1_000_000 错位换算。

加新 provider 直接在 :data:`PRICING` 里追加 ``(provider, model): (prompt, completion)``
即可，附 source URL 注释。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = [
    "PRICING",
    "estimate_cost",
    "format_cost",
    "lookup_rates",
]


#: ``(provider, model) → (prompt_rate, completion_rate)``，单价单位 USD per
#: **million** input/output tokens（与各家公开定价页对齐）。
#:
#: Sources：
#:
#:   - deepseek-v4-flash：https://api-docs.deepseek.com/quick_start/pricing
#:     （deepseek-chat 系列；具体型号名以 :mod:`voxkit.llm.providers` 注册为准）
#:
#: 未列出的 (provider, model) → :func:`estimate_cost` 返回 ``None``。
PRICING: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("deepseek", "deepseek-v4-flash"): (0.27, 1.10),
}


def lookup_rates(provider: str, model: str) -> Optional[Tuple[float, float]]:
    """返回 (prompt_rate, completion_rate) USD/M-token；未知组合返回 ``None``。"""
    return PRICING.get((provider, model))


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    """按 :data:`PRICING` 算总成本（USD）。

    未知 (provider, model) → ``None``（让调用方区分"未知 rate"和"已知但 = $0"）。
    零 token 输入合法，返回 ``0.0``。负 token 视作 0（粗暴 clamp，避免 caller
    误传导致负成本）。
    """
    rates = lookup_rates(provider, model)
    if rates is None:
        return None
    p_rate, c_rate = rates
    pt = max(0, int(prompt_tokens))
    ct = max(0, int(completion_tokens))
    return (pt / 1_000_000) * p_rate + (ct / 1_000_000) * c_rate


def format_cost(usd: Optional[float]) -> str:
    """把 :func:`estimate_cost` 结果格式化成 "``~$0.04``" 或 "``(unknown rate)``"。

    精度策略：
      - >= $1：保留 2 位小数（``~$1.23``）
      - >= $0.01：保留 2 位小数（``~$0.04``）
      - >= $0.0001：保留 4 位小数（``~$0.0042``）
      - 更小：``~$<0.0001``（避免一堆 0 噪音）
    """
    if usd is None:
        return "(unknown rate)"
    if usd >= 1:
        return f"~${usd:.2f}"
    if usd >= 0.01:
        return f"~${usd:.2f}"
    if usd >= 0.0001:
        return f"~${usd:.4f}"
    if usd <= 0:
        return "~$0.00"
    return "~$<0.0001"
