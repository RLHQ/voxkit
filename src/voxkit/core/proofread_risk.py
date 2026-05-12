"""Proofread 阶段的本地风险评级 + token 估算。纯函数，无 I/O 无网络。

风险评级不依赖 LLM 二判：所有规则都基于源文本 vs 校对文本的 diff 特征，可单测、
可解释、可复现。覆盖范围（按文档 §校对风险）：

  - ``numeric_change``：数字 / 日期 / 金额变化 → medium
  - ``protected_term_change``：glossary protected 词被改写 → high
  - ``empty_or_deleted``：非空源文本被改为空 → high
  - ``large_text_delta``：长度差异 >30% → medium
  - ``content_unchanged``：correctedText == sourceText（不算风险，仅用于编辑等级判定）

named-entity-change 在 v1 不实现（NER 不上线），留待未来。

token 估算：CJK 0.5 / Latin 0.25（保守），主要给 batching 用，不要求精确。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from voxkit.io.schema import EditLevel, RiskLevel

__all__ = [
    "estimate_tokens",
    "grade_risk",
    "infer_edit_level",
    "is_cjk_char",
]


# ── token 估算 ─────────────────────────────────────────────────────────────


def is_cjk_char(ch: str) -> bool:
    """是否落在常见 CJK / 假名范围。

    覆盖：CJK Unified Ideographs (U+4E00–U+9FFF)、Hiragana (U+3040–U+309F)、
    Katakana (U+30A0–U+30FF)、中日韩兼容（含中文标点常用区 U+3000–U+303F）。
    """
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3000 <= code <= 0x303F   # CJK 标点
        or 0x3040 <= code <= 0x309F   # Hiragana
        or 0x30A0 <= code <= 0x30FF   # Katakana
        or 0x4E00 <= code <= 0x9FFF   # CJK Unified
        or 0xFF00 <= code <= 0xFFEF   # 全角 ASCII / 半角 Katakana
    )


def estimate_tokens(text: str) -> int:
    """估算 LLM tokens。CJK 0.5 token/char、其它 0.25 token/char（即 4 chars/token）。

    意图是 *保守上限*（不至于把 batch 切得太小）。真实 DeepSeek BPE 对 CJK 大致是
    1 token/char，对 Latin 大致是 0.25 token/char；这里 CJK 系数取 0.5 兼顾"留
    余量"和"别把 batch 切得太小拖低成本效率"。
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if is_cjk_char(c))
    other = len(text) - cjk
    # 至少 1 token，避免 batch 估算时整批为 0 触发死循环。
    return max(1, int(cjk * 0.5 + other * 0.25))


# ── 风险评级 ───────────────────────────────────────────────────────────────

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "blocking": 3}


def _bump(current: RiskLevel, candidate: RiskLevel) -> RiskLevel:
    """取较高风险等级。"""
    return candidate if _RISK_ORDER[candidate] > _RISK_ORDER[current] else current


# 匹配整数 / 小数 / 千分位 / 百分号 / 货币简写。
_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _numeric_tokens(text: str) -> List[str]:
    """提取文本中所有数字字面量；保留原顺序便于做 multiset 比较。"""
    return _NUMERIC_RE.findall(text)


def grade_risk(
    source_text: str,
    corrected_text: str,
    *,
    protected_terms: frozenset[str] | set[str] = frozenset(),
) -> Tuple[RiskLevel, List[str]]:
    """对单条 cue 的 (source, corrected) 评级 → (risk, notes)。

    决策顺序按"最严重优先"：blocking > high > medium > low。``notes`` 是触发的
    规则名称列表（去重保留顺序），用于 UI 展示和审计。
    """
    notes: List[str] = []
    risk: RiskLevel = "low"

    src = source_text.strip()
    cor = corrected_text.strip()

    # 1. 空字符串：非空 → 空属于 high；本来就是空则不报。
    if src and not cor:
        notes.append("empty_or_deleted")
        risk = _bump(risk, "high")

    # 2. Protected term：源出现但校对后不出现。大小写敏感（glossary 默认 smart 但
    #    保护判定仍按字面，避免漏报）。
    for term in protected_terms:
        if term in source_text and term not in corrected_text:
            notes.append(f"protected_term_change:{term}")
            risk = _bump(risk, "high")

    # 3. 数字变化：multiset 不一致即触发。
    src_nums = _numeric_tokens(source_text)
    cor_nums = _numeric_tokens(corrected_text)
    if sorted(src_nums) != sorted(cor_nums):
        notes.append("numeric_change")
        risk = _bump(risk, "medium")

    # 4. 长度差异 >30%（以较长的为分母，避免短文本基数太小）。
    longer = max(len(source_text), len(corrected_text))
    if longer > 0:
        delta_ratio = abs(len(source_text) - len(corrected_text)) / longer
        if delta_ratio > 0.30:
            notes.append("large_text_delta")
            risk = _bump(risk, "medium")

    return risk, notes


def infer_edit_level(source_text: str, corrected_text: str) -> EditLevel:
    """根据 diff 量度推断编辑等级 ``none`` / ``minor`` / ``major``。

      - 完全相同 → none
      - 长度比 ≤10% 且改动只在标点 / 空白 / 短词替换 → minor
      - 其余 → major

    实现选 *字符级 Levenshtein 距离归一化* 的简化版（不需要绝对精确）。
    """
    if source_text == corrected_text:
        return "none"

    # 简化：纯字符集差异 + 长度差异联合判定。
    longer = max(len(source_text), len(corrected_text), 1)
    char_delta = sum(1 for a, b in zip(source_text, corrected_text) if a != b)
    char_delta += abs(len(source_text) - len(corrected_text))
    ratio = char_delta / longer
    return "minor" if ratio <= 0.10 else "major"
