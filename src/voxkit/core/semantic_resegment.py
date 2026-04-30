"""字幕语义重切：把 ASR segment 按"句子优先 + 字幕物理兜底"重新分段。

设计契约（与 segmenter.py 的关系）：

  - segmenter.py 是 chunk-internal 的"原始切分"，输出每个 chunk 的 ASR segment
  - asr_merge.py 把多 chunk 拼成全局 timeline
  - **本模块是 SRT/VTT 渲染前的可选 post-processor**：
    输入 RemixrSegment[]（已带 speaker、word 时间戳），输出 SubtitleCue[]
  - **只影响字幕，不动 transcript.{voxkit,raw}.json**——JSON 是 ASR 真实产出，
    重切是字幕渲染优化，两者语义不同不应混用

算法概要：

  1. 拍平所有 word 到 (text, start, end, speaker)；speaker 继承自 segment
  2. 按 speaker 分块（连续同 speaker 归一组），避免跨说话人合并
  3. 每个 speaker 块：拼文本 → pysbd 切句 → 字符 span 反查到 word 区间
  4. 句子超物理上限 → split_long 在标点 / 韵律 gap 处贪婪切分
  5. 短 cue 合并到同 speaker 邻居（避免闪现 0.5s 字幕）
  6. enforce_monotonic 钳住跨 cue 时间倒挂（whisper word ts 偶尔重叠）

CJK 自动降级：CJK 语言（zh/ja/yue/ko）下 whisper.cpp 不输出 word 时间戳，
直接 pass-through——每个原 segment 包装成 1 个 cue。

参数（:class:`ResegmentParams`）经实验在 EN podcast 上调校：
综合分 baseline 48.5 → resegment 67.5，物理分 31.4 → 83.0
（详见 docs/semantic-resegment.md / tmp/pysbd-experiment 历史）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

from voxkit.core.constants import CJK_LANGUAGES
from voxkit.io.schema import RemixrSegment


__all__ = [
    "ResegmentParams",
    "SubtitleCue",
    "resegment_for_subtitles",
]


# ---------------------------------------------------------------------------
# 参数（rule-set v1）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResegmentParams:
    """语义重切的可调参数；冻结以保证 manifest 可序列化复现。"""

    # 字幕物理上限
    max_dur_s: float = 7.0
    min_dur_s: float = 1.5
    max_chars: int = 84              # ≈ 2 行 × 42（Netflix 主流上限）
    soft_max_chars: int = 75
    max_cps: float = 22.0

    # 韵律
    prosody_gap_s: float = 0.25
    prosody_gap_weight: int = 7

    # 软切点权重表：值越高越优先在该位置切
    # 关键调校：逗号 / "like," 在口语里满天飞，权重要压低；分号 / 韵律 gap 优先
    soft_break_weights: dict[str, int] = field(
        default_factory=lambda: {
            ";": 10, ":": 8, "—": 8, "–": 8,
            ",": 3,
            # 连词（在它前面切）
            "and": 2, "but": 4, "or": 2, "so": 3, "because": 4,
            "however": 5, "though": 3, "although": 4, "while": 2, "yet": 3,
        }
    )


@dataclass(frozen=True)
class SubtitleCue:
    """单条字幕；不持有 word 时间戳，下游渲染只读 start/end/speaker/text。"""

    start: float
    end: float
    speaker: str | None
    text: str


# ---------------------------------------------------------------------------
# 内部数据结构（仅算法层使用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    word: str
    start: float
    end: float
    speaker: str


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def resegment_for_subtitles(
    segments: Sequence[RemixrSegment],
    *,
    language: str,
    params: ResegmentParams | None = None,
) -> list[SubtitleCue]:
    """重切 segment[] → 字幕 cue[]。

    Pre-conditions:
        - segments 已按时间排序、speaker 字段已 inject（diarization 后）
        - 非 CJK 语言下 segments[*].words 必须有 word-level (start, end)

    Behaviour:
        - CJK 语言（zh/ja/yue/ko）→ pass-through（每 segment 1 cue）
        - 非 CJK 但 segments[0].words 为空 → 同样 pass-through（防御）
        - 否则走 pysbd 重切

    pysbd import 失败 → 抛 ImportError，由 caller 决定 warn-once + fallback
    （本模块不吞异常，避免静默降级）。
    """
    p = params or ResegmentParams()

    if not segments:
        return []

    if language.lower() in CJK_LANGUAGES:
        return _passthrough(segments)

    if not segments[0].words:
        return _passthrough(segments)

    # 仅在真正需要时才 import pysbd——保持 voxkit 主路径无依赖
    import pysbd

    segmenter = pysbd.Segmenter(language=language.lower(), clean=False, char_span=True)
    words = _flatten_words(segments)
    cues = _resegment_word_level(words, segmenter, p)
    cues = _enforce_monotonic(cues)
    return cues


# ---------------------------------------------------------------------------
# 实现细节
# ---------------------------------------------------------------------------


def _passthrough(segments: Sequence[RemixrSegment]) -> list[SubtitleCue]:
    return [
        SubtitleCue(start=s.start, end=s.end, speaker=s.speaker, text=s.text.strip())
        for s in segments
        if s.text.strip()
    ]


def _flatten_words(segments: Sequence[RemixrSegment]) -> list[_Word]:
    out: list[_Word] = []
    for seg in segments:
        spk = seg.speaker or "Speaker A"
        for w in seg.words:
            txt = w.word.strip()
            if not txt:
                continue
            out.append(_Word(word=txt, start=float(w.start), end=float(w.end), speaker=spk))
    return out


def _group_by_speaker(words: list[_Word]) -> list[list[_Word]]:
    """Avoid pysbd merging across speaker turns into a single sentence."""
    blocks: list[list[_Word]] = []
    cur: list[_Word] = []
    for w in words:
        if cur and cur[-1].speaker != w.speaker:
            blocks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        blocks.append(cur)
    return blocks


def _words_to_text_with_offsets(
    words: list[_Word],
) -> tuple[str, list[tuple[int, int]]]:
    """单空格连接所有 word；同时记录每个 word 在拼接文本中的 [start, end) 字符区间。"""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, w in enumerate(words):
        if i > 0:
            parts.append(" ")
            cursor += 1
        offsets.append((cursor, cursor + len(w.word)))
        parts.append(w.word)
        cursor += len(w.word)
    return "".join(parts), offsets


def _char_range_to_word_range(
    char_start: int, char_end: int, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    """把字符区间 [char_start, char_end) 映射回 word 索引区间（半开）。"""
    w_start: int | None = None
    w_end: int | None = None
    for i, (cs, ce) in enumerate(offsets):
        if w_start is None and ce > char_start:
            w_start = i
        if cs < char_end:
            w_end = i + 1
        else:
            break
    if w_start is None:
        return (0, 0)
    return (w_start, w_end if w_end is not None else w_start + 1)


# ── 长句切分 ────────────────────────────────────────────────────────────────


# 软切点字符——故意不含 .!?。！？，因为那些是句末标点，pysbd 上游已经按它们
# 把整段切成独立 sentence；split_long 只在 sentence 内部找次级切点。
_PUNCT_CHARS = frozenset({";", ":", "—", "–", ","})
_CONJUNCTIONS = frozenset(
    {"and", "but", "or", "so", "because", "however", "though",
     "although", "while", "yet"}
)


def _chunk_metrics(words: list[_Word]) -> tuple[float, int]:
    if not words:
        return (0.0, 0)
    dur = words[-1].end - words[0].start
    chars = sum(len(w.word) for w in words) + max(0, len(words) - 1)
    return dur, chars


def _need_split(words: list[_Word], p: ResegmentParams) -> bool:
    dur, chars = _chunk_metrics(words)
    if dur <= 0:
        return False
    cps = chars / dur
    return dur > p.max_dur_s or chars > p.max_chars or cps > p.max_cps


def _compute_break_weights(words: list[_Word], p: ResegmentParams) -> list[float]:
    """每个 word 之后的可切性权重；底分用 gap 实数让最长 gap 也能脱颖而出。"""
    n = len(words)
    weights = [0.0] * n
    wmap = p.soft_break_weights
    for i in range(n - 1):
        last_char = words[i].word[-1] if words[i].word else ""
        if last_char in _PUNCT_CHARS:
            weights[i] = max(weights[i], float(wmap.get(last_char, 5)))
        nxt_clean = re.sub(r"[^\w]", "", words[i + 1].word.lower())
        if nxt_clean in _CONJUNCTIONS:
            weights[i] = max(weights[i], float(wmap.get(nxt_clean, 3)))
        gap = max(0.0, words[i + 1].start - words[i].end)
        if gap >= p.prosody_gap_s:
            scaled = p.prosody_gap_weight * (1 + min(1.5, gap) / 1.5)
            weights[i] = max(weights[i], float(scaled))
        if gap > 0:
            # 底分；远小于标点权重，仅在完全无标点段落里救场
            weights[i] = max(weights[i], gap * 1.0)
    return weights


# 评分系数（实验中 sweet-spot；改动需重跑评分）
_WEIGHT_BONUS = 25.0       # 1 分权重 ≈ 容忍 25 字符偏离 target
_OFFSET_TOL = 15           # target ± 这么多字符内零惩罚
_OVER_HARD_PENALTY = 60.0  # 软惩罚；允许在没更好选择时切到稍超 hard 的位置
_HARD_CHAR_RATIO = 1.3     # 绝不超 hard_chars × 这个比例
_HARD_DUR_RATIO = 1.2      # 同上


def _split_long(words: list[_Word], p: ResegmentParams) -> list[list[_Word]]:
    """把超物理上限的 word 序列贪婪切分到合规。

    策略：先估算需要切几段（按 chars / dur 谁更紧），再依次找最接近"目标位置"
    的高权重切点。绝不在词组内部硬切——除非整段一个 gap / 标点都没有。
    """
    if not _need_split(words, p) or len(words) < 4:
        return [words]

    n = len(words)
    chars_total = sum(len(w.word) for w in words) + max(0, n - 1)
    dur_total = words[-1].end - words[0].start

    n_by_chars = -(-chars_total // p.max_chars)  # ceil
    n_by_dur = -(-int(dur_total * 1000) // int(p.max_dur_s * 1000))
    n_chunks = max(2, n_by_chars, n_by_dur)

    cum_chars = [0] * n
    c = 0
    for i, w in enumerate(words):
        c += len(w.word) + (1 if i > 0 else 0)
        cum_chars[i] = c

    weights = _compute_break_weights(words, p)
    target_size = chars_total / n_chunks

    chunks: list[list[_Word]] = []
    start = 0
    for _ in range(n_chunks - 1):
        target_end_chars = (cum_chars[start - 1] if start > 0 else 0) + target_size
        best_idx = -1
        best_score = -1e9
        for i in range(start, n - 1):
            chunk_chars = cum_chars[i] - (cum_chars[start - 1] if start > 0 else 0)
            chunk_dur = words[i].end - words[start].start
            if (
                chunk_chars > p.max_chars * _HARD_CHAR_RATIO
                or chunk_dur > p.max_dur_s * _HARD_DUR_RATIO
            ):
                # 绝对硬上限：超过这个 score 几何无穷负，直接跳
                continue
            over_hard = chunk_chars > p.max_chars or chunk_dur > p.max_dur_s
            offset_raw = abs(cum_chars[i] - target_end_chars)
            offset = max(0.0, offset_raw - _OFFSET_TOL)
            penalty = _OVER_HARD_PENALTY if over_hard else 0.0
            score = weights[i] * _WEIGHT_BONUS - offset - penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1 or best_idx >= n - 1:
            break
        chunks.append(words[start : best_idx + 1])
        start = best_idx + 1
    if start < n:
        chunks.append(words[start:])

    # 万一某段仍超 hard（极少情况）→ 递归
    final: list[list[_Word]] = []
    for chunk in chunks:
        if _need_split(chunk, p) and len(chunk) >= 4:
            final.extend(_split_long(chunk, p))
        else:
            final.append(chunk)
    return [c for c in final if c]


# ── 短 cue 合并 ─────────────────────────────────────────────────────────────


def _can_merge(a: SubtitleCue, b: SubtitleCue, p: ResegmentParams) -> bool:
    if a.speaker != b.speaker:
        return False
    merged_dur = b.end - a.start
    merged_chars = len(a.text) + 1 + len(b.text)
    return merged_dur <= p.max_dur_s and merged_chars <= p.max_chars


def _pick_merge_neighbour(
    *,
    prev_ok: bool,
    next_ok: bool,
    prev_dur: float,
    next_dur: float,
) -> Literal["prev", "next", "skip"]:
    """Decide which neighbour to merge a too-short cue into.

    Prefer the shorter neighbour to keep cue durations balanced; skip when
    neither side is mergeable (different speaker / would exceed physical limit).
    """
    if not prev_ok and not next_ok:
        return "skip"
    if next_ok and (not prev_ok or next_dur <= prev_dur):
        return "next"
    return "prev"


def _merge_too_short(cues: list[SubtitleCue], p: ResegmentParams) -> list[SubtitleCue]:
    """Bi-directionally merge cues shorter than ``min_dur_s`` into a same-speaker
    neighbour, preferring the shorter side so durations stay balanced.
    """
    out: list[SubtitleCue] = list(cues)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            cur = out[i]
            if cur.end - cur.start >= p.min_dur_s:
                i += 1
                continue
            prev_ok = i > 0 and _can_merge(out[i - 1], cur, p)
            next_ok = i + 1 < len(out) and _can_merge(cur, out[i + 1], p)
            decision = _pick_merge_neighbour(
                prev_ok=prev_ok,
                next_ok=next_ok,
                prev_dur=(out[i - 1].end - out[i - 1].start) if prev_ok else 0.0,
                next_dur=(out[i + 1].end - out[i + 1].start) if next_ok else 0.0,
            )
            if decision == "skip":
                i += 1
                continue
            if decision == "next":
                nxt = out[i + 1]
                merged = SubtitleCue(
                    start=cur.start, end=nxt.end, speaker=cur.speaker,
                    text=(cur.text + " " + nxt.text).strip(),
                )
                out[i : i + 2] = [merged]
            else:  # "prev"
                prv = out[i - 1]
                merged = SubtitleCue(
                    start=prv.start, end=cur.end, speaker=cur.speaker,
                    text=(prv.text + " " + cur.text).strip(),
                )
                out[i - 1 : i + 1] = [merged]
                i -= 1
            changed = True
            i += 1
    return out


# ── 主流程 ─────────────────────────────────────────────────────────────────


def _resegment_word_level(
    words: list[_Word],
    segmenter,  # pysbd.Segmenter
    p: ResegmentParams,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for block in _group_by_speaker(words):
        full_text, offsets = _words_to_text_with_offsets(block)
        for span in segmenter.segment(full_text):
            ws, we = _char_range_to_word_range(span.start, span.end, offsets)
            sent_words = block[ws:we]
            if not sent_words:
                continue
            for chunk in _split_long(sent_words, p):
                if not chunk:
                    continue
                cues.append(
                    SubtitleCue(
                        start=chunk[0].start,
                        end=chunk[-1].end,
                        speaker=chunk[0].speaker,
                        text=" ".join(w.word for w in chunk).strip(),
                    )
                )
    return _merge_too_short(cues, p)


def _enforce_monotonic(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    """钳住 cue.start ≥ 前一 cue.end（whisper word 时间戳偶尔倒挂）。"""
    out: list[SubtitleCue] = []
    last_end = 0.0
    for c in cues:
        s = max(c.start, last_end)
        e = max(c.end, s + 0.05)
        out.append(SubtitleCue(start=s, end=e, speaker=c.speaker, text=c.text))
        last_end = e
    return out
