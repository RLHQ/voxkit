"""``voxkit.core.proofread_risk`` 纯函数单测。"""

from __future__ import annotations

import pytest

from voxkit.core.proofread_risk import (
    estimate_tokens,
    grade_risk,
    infer_edit_level,
    is_cjk_char,
)


# ── is_cjk_char ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ch", ["中", "文", "你", "好", "あ", "カ", "！", "，"])
def test_is_cjk_char_true_for_cjk(ch: str) -> None:
    assert is_cjk_char(ch) is True


@pytest.mark.parametrize("ch", ["a", "Z", "1", " ", ".", ",", ""])
def test_is_cjk_char_false_for_others(ch: str) -> None:
    assert is_cjk_char(ch) is False


# ── estimate_tokens ──────────────────────────────────────────────────────────


def test_estimate_tokens_empty_returns_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_cjk_uses_05_coef() -> None:
    # 10 个 CJK 字符 → ~5 tokens（保守上限，不要求精确）
    assert estimate_tokens("中" * 10) == 5


def test_estimate_tokens_latin_uses_025_coef() -> None:
    # 16 个 ASCII → 4 tokens
    assert estimate_tokens("a" * 16) == 4


def test_estimate_tokens_mixed_is_sum() -> None:
    # 4 CJK (2 tok) + 8 ASCII (2 tok) = 4
    assert estimate_tokens("中文中文" + "a" * 8) == 4


def test_estimate_tokens_min_one() -> None:
    # 单字符也至少 1 token，防止 batch 估算除零
    assert estimate_tokens("a") == 1


# ── grade_risk ───────────────────────────────────────────────────────────────


def test_grade_risk_identical_low() -> None:
    risk, notes = grade_risk("hello world", "hello world")
    assert risk == "low"
    assert notes == []


def test_grade_risk_punctuation_only_is_low() -> None:
    risk, notes = grade_risk("hello world", "Hello, world.")
    assert risk == "low"
    assert "numeric_change" not in notes


def test_grade_risk_numeric_change_is_medium() -> None:
    risk, notes = grade_risk("收入 100 元", "收入 200 元")
    assert risk == "medium"
    assert "numeric_change" in notes


def test_grade_risk_empty_or_deleted_is_high() -> None:
    risk, notes = grade_risk("有内容", "")
    assert risk == "high"
    assert "empty_or_deleted" in notes


def test_grade_risk_empty_to_empty_is_low() -> None:
    # 源就是空，不应误报
    risk, notes = grade_risk("", "")
    assert risk == "low"
    assert notes == []


def test_grade_risk_protected_term_change_is_high() -> None:
    risk, notes = grade_risk(
        "use Claude for this",
        "use ChatGPT for this",
        protected_terms=frozenset({"Claude"}),
    )
    assert risk == "high"
    assert any("protected_term_change:Claude" in n for n in notes)


def test_grade_risk_protected_term_preserved_no_flag() -> None:
    risk, notes = grade_risk(
        "use Claude for this",
        "Use Claude for this.",
        protected_terms=frozenset({"Claude"}),
    )
    assert risk == "low"
    assert not any("protected_term_change" in n for n in notes)


def test_grade_risk_large_text_delta_is_medium() -> None:
    # 长度差 >30%
    risk, notes = grade_risk("a" * 100, "a" * 60)
    assert risk == "medium"
    assert "large_text_delta" in notes


def test_grade_risk_priority_high_over_medium() -> None:
    # 同时触发 numeric_change (medium) + empty_or_deleted (high) → high
    risk, notes = grade_risk("有 5 个", "")
    assert risk == "high"
    assert "empty_or_deleted" in notes


# ── infer_edit_level ─────────────────────────────────────────────────────────


def test_infer_edit_level_identical_is_none() -> None:
    assert infer_edit_level("hello", "hello") == "none"


def test_infer_edit_level_small_change_is_minor() -> None:
    # 只改一个字符（标点）
    assert infer_edit_level("hello world", "hello world.") == "minor"


def test_infer_edit_level_large_change_is_major() -> None:
    assert infer_edit_level("hello", "completely different sentence") == "major"
