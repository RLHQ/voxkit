"""``voxkit.core.pricing`` 单元测试。

覆盖：已知 model / 未知 model / 零 token / format_cost 多档分支。
"""

from __future__ import annotations

import pytest

from voxkit.core.pricing import (
    PRICING,
    estimate_cost,
    format_cost,
    lookup_rates,
)


# ── lookup_rates ────────────────────────────────────────────────────────────


def test_lookup_rates_known() -> None:
    rates = lookup_rates("deepseek", "deepseek-v4-flash")
    assert rates == (0.27, 1.10)


def test_lookup_rates_unknown_returns_none() -> None:
    assert lookup_rates("openai", "gpt-4o") is None
    assert lookup_rates("deepseek", "definitely-not-a-model") is None


# ── estimate_cost ───────────────────────────────────────────────────────────


def test_estimate_cost_known_model() -> None:
    # 1M prompt tokens * 0.27 + 1M completion * 1.10 = 1.37
    usd = estimate_cost("deepseek", "deepseek-v4-flash", 1_000_000, 1_000_000)
    assert usd == pytest.approx(1.37)


def test_estimate_cost_partial_tokens() -> None:
    # 100k + 50k tokens
    # 0.1 * 0.27 + 0.05 * 1.10 = 0.027 + 0.055 = 0.082
    usd = estimate_cost("deepseek", "deepseek-v4-flash", 100_000, 50_000)
    assert usd == pytest.approx(0.082)


def test_estimate_cost_zero_tokens() -> None:
    assert estimate_cost("deepseek", "deepseek-v4-flash", 0, 0) == 0.0


def test_estimate_cost_unknown_model_returns_none() -> None:
    assert estimate_cost("openai", "gpt-4o", 1000, 1000) is None
    assert estimate_cost("deepseek", "未知模型", 1000, 1000) is None


def test_estimate_cost_negative_tokens_clamped_to_zero() -> None:
    """负数 token 被 clamp 成 0，避免负 cost。"""
    assert estimate_cost("deepseek", "deepseek-v4-flash", -100, -50) == 0.0


# ── format_cost ─────────────────────────────────────────────────────────────


def test_format_cost_none() -> None:
    assert format_cost(None) == "(unknown rate)"


def test_format_cost_zero() -> None:
    assert format_cost(0.0) == "~$0.00"


def test_format_cost_pennies() -> None:
    assert format_cost(0.04) == "~$0.04"
    assert format_cost(0.10) == "~$0.10"


def test_format_cost_dollars() -> None:
    assert format_cost(1.234) == "~$1.23"
    assert format_cost(12.5) == "~$12.50"


def test_format_cost_sub_penny() -> None:
    assert format_cost(0.0042) == "~$0.0042"


def test_format_cost_tiny_clamped() -> None:
    """小于 $0.0001 → 显示 "<0.0001" 避免一长串 0。"""
    assert format_cost(0.00001) == "~$<0.0001"


# ── 表里所有条目都至少有合理形状（防止表里塞错数据） ────────────────────────


def test_pricing_table_shape() -> None:
    """每个条目必须是 (provider:str, model:str): (prompt:float, completion:float)。"""
    for key, val in PRICING.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], str) and isinstance(key[1], str)
        assert isinstance(val, tuple) and len(val) == 2
        p_rate, c_rate = val
        assert isinstance(p_rate, (int, float)) and p_rate >= 0
        assert isinstance(c_rate, (int, float)) and c_rate >= 0
