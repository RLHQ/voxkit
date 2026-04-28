"""分段转录合并工具 —— port from CutFlow services/utils/asr-merge.ts。

负责把多个 chunk 的 ASR 结果在全局时间线上拼起来：
  1. 把每个 chunk 的 segment / word 时间戳偏移到绝对时间
  2. 在 chunk 拼接处去掉重复 segment（chunk 边界 overlap 区域）
  3. 重新按全局顺序生成 seg_NNN id
  4. 校验合并后时间线连续性，产出 MergeNote 给上游做日志/审计

Public API：
  - offset_segment(seg, delta_secs)
  - validate_timeline_continuity(segments)
  - merge_chunks(chunks)
  - write_merge_log(chunks, merged, drops_per_chunk, path)

设计注释：
  - chunk_start_secs 约定 = 该 chunk 在全局时间线上的起始秒数（绝对）。
    chunk 内部 segment.start / word.start 是 chunk-relative（从 0 起）。
    merge_chunks 会把 chunk-relative 偏成 absolute。
  - 所有函数都返回新对象，不就地修改输入；原 TS 实现是就地改的，
    Python 这里换成 pydantic immutable 风格更安全。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from voxkit.io.schema import TranscriptSegment, Word

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OVERLAP_DEDUP_TOLERANCE_SECS: float = 0.5
"""chunk 接缝处 overlap 去重的时间容差（±0.5s）。"""

TIMELINE_OUT_OF_ORDER_THRESHOLD_SECS: float = 0.0
"""任何 curr.start < prev.end 都算 out_of_order（含 0 即抹平边界，0 不触发）。"""

TIMELINE_BIG_OVERLAP_THRESHOLD_SECS: float = 1.0
"""超过 1s 的 overlap 才视为异常 overlap。"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkResult:
    """单个 chunk 的转录结果，merge 输入。

    Attributes:
        chunk_index: chunk 序号（0-based）
        segments: post-segmenter 的 TranscriptSegment 列表，**尚未** offset
                  （即 segment.start 是 chunk-relative，从 0 起）。
        chunk_start_secs: 该 chunk 在全局时间线上的起始秒数（绝对时间）。
                          merge_chunks 会用这个值把 segments 偏到 absolute。
    """

    chunk_index: int
    segments: list[TranscriptSegment]
    chunk_start_secs: float


@dataclass(frozen=True)
class MergeNote:
    """合并过程中发现的时间线异常，warn-only。"""

    kind: Literal["out_of_order", "overlap"]
    seg_id: str
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
# offset_segment
# ─────────────────────────────────────────────────────────────────────────────


def _round_ms(x: float) -> float:
    """Round to millisecond precision (3 decimals)."""
    return round(x, 3)


def offset_segment(seg: TranscriptSegment, delta_secs: float) -> TranscriptSegment:
    # ─────────────────────────────────────────────────────────────
    # PORT NOTE — THE 6-MONTH BUG REGRESSION
    # Remixr originally only offset segment.{start,end}, leaving
    # word-level timestamps frozen at chunk-relative 0. This produced
    # nonsensical word timings after merge. ALWAYS offset words too.
    # See tests/test_asr_merge.py::test_word_offset_regression
    # ─────────────────────────────────────────────────────────────
    """Return a NEW TranscriptSegment with start/end and EVERY word.start/word.end shifted.

    Round outputs to 3 decimals (millisecond precision). Never mutate input.
    """
    new_words = [
        Word(
            word=w.word,
            start=_round_ms(w.start + delta_secs),
            end=_round_ms(w.end + delta_secs),
        )
        for w in seg.words
    ]
    return TranscriptSegment(
        id=seg.id,
        start=_round_ms(seg.start + delta_secs),
        end=_round_ms(seg.end + delta_secs),
        text=seg.text,
        words=new_words,
        no_speech_prob=seg.no_speech_prob,
        avg_confidence=seg.avg_confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# validate_timeline_continuity
# ─────────────────────────────────────────────────────────────────────────────


def validate_timeline_continuity(segments: list[TranscriptSegment]) -> list[MergeNote]:
    """Walk segments in order. For each pair (prev, curr):
      - If curr.start < prev.end: emit MergeNote(kind="out_of_order", ...)
      - If curr.start < prev.end - 1.0 (>1s overlap): emit MergeNote(kind="overlap", ...)

    Note: an out-of-order item could ALSO be a big overlap if it's > 1s. Emit BOTH notes
    in that case (doesn't matter for correctness, just for logging completeness).

    Returns a list of notes; warn-only — never raises.
    """
    notes: list[MergeNote] = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        # out_of_order: any negative gap
        if curr.start < prev.end - TIMELINE_OUT_OF_ORDER_THRESHOLD_SECS:
            notes.append(
                MergeNote(
                    kind="out_of_order",
                    seg_id=curr.id,
                    detail=(
                        f"segments[{i}].start={curr.start:.3f} "
                        f"< segments[{i - 1}].end={prev.end:.3f}"
                    ),
                )
            )

        # overlap: > 1s
        if curr.start < prev.end - TIMELINE_BIG_OVERLAP_THRESHOLD_SECS:
            notes.append(
                MergeNote(
                    kind="overlap",
                    seg_id=curr.id,
                    detail=(
                        f"segments[{i}].start={curr.start:.3f} "
                        f"< segments[{i - 1}].end={prev.end:.3f} "
                        f"(overlap={prev.end - curr.start:.3f}s)"
                    ),
                )
            )
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# merge_chunks
# ─────────────────────────────────────────────────────────────────────────────


def _segment_signal(seg: TranscriptSegment) -> int:
    """Quality signal — word count（首选）或 text 字符数（fallback）。

    word_timestamps 模式下每个英文单词独立 entry，word count 直接反映 chunk
    实际产出的"颗粒度"——whisper 早截断会让 word count 显著低。CJK 短语
    模式下 ``words`` 为空，回落到 text 字符数（同样反映产出量）。
    """
    if seg.words:
        return len(seg.words)
    return len(seg.text.strip())


def _overlap_score(
    segs: list[TranscriptSegment], lo: float, hi: float
) -> int:
    """Sum of ``_segment_signal`` over segments overlapping ``[lo, hi)``."""
    return sum(
        _segment_signal(s)
        for s in segs
        if s.end > lo and s.start < hi
    )


def _append_with_tolerance(
    merged: list[TranscriptSegment],
    incoming: list[TranscriptSegment],
    tolerance_secs: float,
) -> None:
    """Append ``incoming`` into ``merged`` in-place, skipping segments whose
    start is before ``merged[-1].end - tolerance_secs`` (already covered).

    Mutates ``merged``. 复用三处 merge-arbitration 分支共享的 last_end 守卫。
    """
    last_end = merged[-1].end if merged else float("-inf")
    for seg in incoming:
        if seg.start >= last_end - tolerance_secs:
            merged.append(seg)
            last_end = seg.end


def _reid_sequential(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Re-id segments to seg_001, seg_002, ... (zero-pad width 3)."""
    out: list[TranscriptSegment] = []
    for i, s in enumerate(segments, start=1):
        out.append(
            TranscriptSegment(
                id=f"seg_{i:03d}",
                start=s.start,
                end=s.end,
                text=s.text,
                words=list(s.words),
                no_speech_prob=s.no_speech_prob,
                avg_confidence=s.avg_confidence,
            )
        )
    return out


def merge_chunks(
    chunks: list[ChunkResult],
    *,
    overlap_tolerance_secs: float = OVERLAP_DEDUP_TOLERANCE_SECS,
) -> tuple[list[TranscriptSegment], list[MergeNote]]:
    """Merge per-chunk segments into a single global segment list.

    Strategy: **signal-aware overlap arbitration** — 在每个 chunk 接缝处，
    把重叠区视作小区域，分别用 prev 和 chunk i 的产出打分（word count；
    fallback 到 text 字符数）。**胜方占据整个 overlap 区**，败方拿 overlap
    之外的部分。

    Why signal-aware:
      A/B 实验（tmp/synth/）显示两种简单策略都有缺陷：
      - chunk-i-priority（旧）：chunk 0 末尾完整产出（如 "differentiation"）
        被 dedup 误删——chunk i 暖机后跳过了同样的内容。
      - prev-priority：保住 chunk 0 末尾长词，但失去 chunk i 同区间的标点 /
        断句（chunk i 在中段处理，模型注意力更稳）。
      smart 策略：拼前比较两边在 overlap 区的实际产出量——whisper 早截断的
      一边 word count 必然低，胜方自然是产出更全的那边。"differentiation"
      场景：prev 含 7 词，chunk i 在同区间含 3 词 → prev 赢，长词保住。

    Algorithm:
      1. Offset 每 chunk 的 segments 到 absolute time。
      2. For i = 1..N-1（chunk i 非空）：
         a. overlap_lo = chunk_i.first_seg.start
         b. overlap_hi = max(prev_last.end, overlap_lo)
         c. 若 hi <= lo → 无真重叠，直接 append chunk_i。
         d. 否则比 prev 与 chunk_i 在 [lo, hi) 内的 ``_segment_signal`` 总和。
         e. 胜方占 overlap；败方仅保留 overlap 外的 segments。
      3. Re-id + validate_timeline_continuity。

    Why prev-priority not chunk-i-priority (kept for context):
      原 Remixr 实现是 "chunk i 优先"——把 prev 末尾在 ``[chunk_start_i -
      tolerance, +∞)`` 内的 segments 全删让 chunk i 接管。这假设 chunk i
      会在 overlap 区重新产出同等质量的内容。A/B 实验（tmp/synth/）证伪了这
      个假设：whisper.cpp 在 chunk 末尾 5s 实际上**完整产出**到边界（如
      "differentiation" @ 595.28）；而 chunk i 第一个 segment 因为暖机往往
      从 chunk_start + 几百 ms 起，跳过了边界附近的内容。chunk-i-priority
      下 "differentiation" 整词消失。改用 prev-priority 后：prev 的 5s
      overlap 产出全部保留，chunk i 只在 prev 真正用尽后接管，不再丢词。
    """
    if not chunks:
        return [], []

    # Step 1: pre-offset each chunk's segments
    offset_per_chunk: list[list[TranscriptSegment]] = [
        [offset_segment(s, ch.chunk_start_secs) for s in ch.segments] for ch in chunks
    ]

    # Step 2: signal-aware overlap arbitration
    merged: list[TranscriptSegment] = list(offset_per_chunk[0])

    for i in range(1, len(chunks)):
        chunk_i_segs = offset_per_chunk[i]
        if not chunk_i_segs:
            continue
        if not merged:
            merged.extend(chunk_i_segs)
            continue

        overlap_lo = chunk_i_segs[0].start
        overlap_hi = max(merged[-1].end, overlap_lo)

        # 三分支结构：先决定 prev overlap 区是否需要 trim，再统一调
        # _append_with_tolerance 把 chunk_i 接上去（守卫 merged[-1].end）。
        no_real_overlap = overlap_hi <= overlap_lo + 1e-6
        if not no_real_overlap:
            score_prev = _overlap_score(merged, overlap_lo, overlap_hi)
            score_chunk_i = _overlap_score(chunk_i_segs, overlap_lo, overlap_hi)
            if score_chunk_i > score_prev:
                # chunk_i 信号更强：trim prev 在 overlap 区的 segments
                merged[:] = [
                    s for s in merged
                    if s.start < overlap_lo - overlap_tolerance_secs
                ]
            # else: prev 信号 >= chunk_i（含并列）→ 不动 prev，由 last_end 守卫自然剪枝

        _append_with_tolerance(merged, chunk_i_segs, overlap_tolerance_secs)

    # Step 3: re-id
    merged = _reid_sequential(merged)

    # Step 4: validate
    notes = validate_timeline_continuity(merged)

    return merged, notes


# ─────────────────────────────────────────────────────────────────────────────
# write_merge_log
# ─────────────────────────────────────────────────────────────────────────────


def write_merge_log(
    chunks: list[ChunkResult],
    merged: list[TranscriptSegment],
    drops_per_chunk: Optional[dict[int, list[str]]],
    path: Path,
) -> None:
    """Write merge.json (single JSON object, NOT NDJSON) recording the merge plan.

    Schema:
      {
        "chunks": [
          {"index": N, "chunk_start_secs": ..., "segments_in": M, "segments_dropped": [<seg_ids>]}
        ],
        "merged_segments": <total>,
        "tolerance_secs": 0.5
      }

    Pretty-printed (indent=2). Used for audit.
    """
    drops_per_chunk = drops_per_chunk or {}
    payload = {
        "chunks": [
            {
                "index": ch.chunk_index,
                "chunk_start_secs": ch.chunk_start_secs,
                "segments_in": len(ch.segments),
                "segments_dropped": list(drops_per_chunk.get(ch.chunk_index, [])),
            }
            for ch in chunks
        ],
        "merged_segments": len(merged),
        "tolerance_secs": OVERLAP_DEDUP_TOLERANCE_SECS,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "OVERLAP_DEDUP_TOLERANCE_SECS",
    "TIMELINE_OUT_OF_ORDER_THRESHOLD_SECS",
    "TIMELINE_BIG_OVERLAP_THRESHOLD_SECS",
    "ChunkResult",
    "MergeNote",
    "offset_segment",
    "validate_timeline_continuity",
    "merge_chunks",
    "write_merge_log",
]
