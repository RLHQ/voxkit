"""whisper.cpp entry → TranscriptSegment 重组器。

这是 Remixr ``services/utils/whisper-segmenter.ts`` 的 Python 端口，行为对齐：

  - 英文 word 模式：whisper.cpp 在 ``--max-len 1 --split-on-word`` 下为英文输出
    每个 word 一个 entry（含前导空格），按 4 个边界条件聚合成 segment：
      1. 末尾标点（``.!?。！？``）
      2. 与下一 entry 的 gap > 500ms
      3. 段累计时长 > 5s
      4. 段累计字符 > 100
  - 中文短语模式：whisper.cpp 不为 CJK 输出 word-level 时间戳，每个 entry
    本身已是短语级，直接 1:1 映射到 segment，``words`` 留空。

模式判定优先级：
  1. ``language_hint`` ∈ {zh, ja, yue, ko} → chinese_phrase
  2. ``language_hint`` == "en" → english_word
  3. 否则按 entry 前导空格比例：≥ 0.5 → english_word，否则 chinese_phrase
  4. 空 entry 列表 → english_word（默认，但会立即返回 []）

输出 ``TranscriptSegment`` 的 ``id`` 是 ``f"seg_{i+1:03d}"`` 格式占位符；
下游 (``asr_merge`` 拼接 chunk 后) 会重新统一编号。
"""

from __future__ import annotations

import re
from typing import Literal

from voxkit.core.constants import CJK_LANGUAGES
from voxkit.core.types import Entry
from voxkit.io.schema import TranscriptSegment, Word

# ---------------------------------------------------------------------------
# 边界触发常量（与 Remixr whisper-segmenter.ts 对齐）
# ---------------------------------------------------------------------------

#: 末尾标点正则。允许标点后跟引号/空白等非字母字符（覆盖 ``"hello."`` / ``hello.``）。
PUNCT_END_RE: re.Pattern[str] = re.compile(r"[.!?。！？]\W*$")

#: 与下一 entry 的 ``offsets.from - offsets.to`` 超过该值（毫秒）→ 切 segment。
GAP_MS: int = 500

#: 段累计时长上限（毫秒）。
MAX_DUR_MS: int = 5_000

#: 段累计字符上限（防止无标点长句无限增长）。
MAX_CHARS: int = 100

#: 判定走"英文 word 模式"的最小占比：≥ 该比例的 entry 含前导空格。
#: 阈值用 ``>=`` 比较，0.5 归到 english_word（与 Remixr TS 一致：``>= 0.5``）。
ENGLISH_LIKELIHOOD_RATIO: float = 0.5


Mode = Literal["english_word", "chinese_phrase"]


# ---------------------------------------------------------------------------
# 模式检测
# ---------------------------------------------------------------------------


def detect_mode(entries: list[Entry], language_hint: str | None = None) -> Mode:
    """决定走 word 模式还是 phrase 模式。

    决策顺序：

      1. ``language_hint`` 命中 :data:`CJK_LANGUAGES` → ``"chinese_phrase"``
      2. ``language_hint == "en"`` → ``"english_word"``
      3. 否则按 entry 前导空格比例：

         - ``ratio >= 0.5`` → ``"english_word"``
         - 否则 → ``"chinese_phrase"``

      4. 空列表 → ``"english_word"``（默认，下游 segment_entries 会立即返回 []）

    Args:
        entries: 已过滤掉空文本/特殊 token 的 entry 列表。
        language_hint: 来自 whisper.cpp ``result.language`` 或 CLI ``--language``。

    Returns:
        ``"english_word"`` 或 ``"chinese_phrase"``。
    """
    if language_hint:
        lang = language_hint.lower()
        if lang in CJK_LANGUAGES:
            return "chinese_phrase"
        if lang == "en":
            return "english_word"
        # 其他语言（fr / de / es 等）目前都按英文 word 模式处理（whisper.cpp
        # 对它们也输出含前导空格的 word entry）。落到比例判定。

    if not entries:
        return "english_word"

    leading = sum(1 for e in entries if e.text.startswith(" "))
    ratio = leading / len(entries)
    return "english_word" if ratio >= ENGLISH_LIKELIHOOD_RATIO else "chinese_phrase"


# ---------------------------------------------------------------------------
# 内部聚合工具
# ---------------------------------------------------------------------------


def _agg_optional(values: list[float | None]) -> float | None:
    """对一组可能为 None 的浮点数求均值；全 None 返回 None。"""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _ms_to_secs(ms: int) -> float:
    """毫秒 → 秒，保留 3 位小数（避免浮点尾巴污染 JSON 输出）。"""
    return round(ms / 1000.0, 3)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def segment_entries(
    entries: list[Entry],
    language: str | None = None,
) -> list[TranscriptSegment]:
    """把 :class:`Entry` 列表聚合成 :class:`TranscriptSegment` 列表。

    内部先调 :func:`detect_mode` 确定模式，再走对应分支：

      - english_word：滑动窗口聚合（4 边界条件 + EOF flush）
      - chinese_phrase：1 entry 1 segment（words 留空）

    Args:
        entries: 已过滤掉空文本/特殊 token 的 entry 列表。一般由
            :func:`voxkit.core.whisper_exec.parse_whisper_json` 产生。
        language: 可选 language hint（whisper.cpp ``result.language``）。

    Returns:
        :class:`TranscriptSegment` 列表。``id`` 为占位符
        ``"seg_001"``、``"seg_002"`` …，下游 ``asr_merge`` 会重新编号。
    """
    if not entries:
        return []

    mode = detect_mode(entries, language)
    if mode == "chinese_phrase":
        return _segment_chinese_phrase(entries)
    return _segment_english_word(entries)


# ---------------------------------------------------------------------------
# 中文短语模式
# ---------------------------------------------------------------------------


def _segment_chinese_phrase(entries: list[Entry]) -> list[TranscriptSegment]:
    """每个 entry 直接当一个 segment，``words`` 留空。"""
    segments: list[TranscriptSegment] = []
    for entry in entries:
        text = entry.text.strip()
        if not text:
            continue
        seg_id = f"seg_{len(segments) + 1:03d}"
        segments.append(
            TranscriptSegment(
                id=seg_id,
                start=_ms_to_secs(entry.t_from_ms),
                end=_ms_to_secs(entry.t_to_ms),
                text=text,
                words=[],
                no_speech_prob=entry.no_speech_prob,
                avg_confidence=entry.confidence,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# 英文 word 模式
# ---------------------------------------------------------------------------


def _segment_english_word(entries: list[Entry]) -> list[TranscriptSegment]:
    """滑动窗口按 4 边界条件聚合。

    边界条件（在「append 当前 entry 之后」检查，与 Remixr TS 对齐）：

      1. ``PUNCT_END_RE`` 命中累积文本末尾
      2. ``last_word.t_to - first_word.t_from > MAX_DUR_MS``
      3. ``len(cur_text.strip()) > MAX_CHARS``
      4. **下一个 entry** 的 gap > GAP_MS（注意 gap 用「下一 entry.from -
         当前 entry.to」判断，不是回头看）

    EOF：循环结束 flush 残留窗口。
    """
    segments: list[TranscriptSegment] = []
    buf_entries: list[Entry] = []
    buf_text_parts: list[str] = []  # 含前导空格，用于拼回自然分词

    def flush() -> None:
        if not buf_entries:
            return
        text = "".join(buf_text_parts).strip()
        if not text:
            buf_entries.clear()
            buf_text_parts.clear()
            return
        words = [
            Word(
                word=e.text.strip(),
                start=_ms_to_secs(e.t_from_ms),
                end=_ms_to_secs(e.t_to_ms),
            )
            for e in buf_entries
        ]
        seg_id = f"seg_{len(segments) + 1:03d}"
        segments.append(
            TranscriptSegment(
                id=seg_id,
                start=_ms_to_secs(buf_entries[0].t_from_ms),
                end=_ms_to_secs(buf_entries[-1].t_to_ms),
                text=text,
                words=words,
                no_speech_prob=_agg_optional([e.no_speech_prob for e in buf_entries]),
                avg_confidence=_agg_optional([e.confidence for e in buf_entries]),
            )
        )
        buf_entries.clear()
        buf_text_parts.clear()

    for i, entry in enumerate(entries):
        # 防御性：trim 后空文本（不该到这里，但 Agent X 的过滤未必覆盖所有边角）
        if not entry.text.strip():
            continue

        buf_entries.append(entry)
        buf_text_parts.append(entry.text)

        is_last = i == len(entries) - 1
        nxt = entries[i + 1] if not is_last else None

        cur_text = "".join(buf_text_parts)
        first_entry = buf_entries[0]
        cur_dur_ms = entry.t_to_ms - first_entry.t_from_ms
        cur_chars = len(cur_text.strip())

        reached_punct = bool(PUNCT_END_RE.search(cur_text))
        reached_dur = cur_dur_ms > MAX_DUR_MS
        reached_chars = cur_chars > MAX_CHARS
        reached_gap = nxt is not None and (nxt.t_from_ms - entry.t_to_ms) > GAP_MS

        if is_last or reached_punct or reached_dur or reached_chars or reached_gap:
            flush()

    return segments


__all__ = [
    "Mode",
    "PUNCT_END_RE",
    "GAP_MS",
    "MAX_DUR_MS",
    "MAX_CHARS",
    "ENGLISH_LIKELIHOOD_RATIO",
    "CJK_LANGUAGES",
    "detect_mode",
    "segment_entries",
]
