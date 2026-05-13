"""字幕语义重切：把 ASR segment 按"句子优先 + 字幕物理兜底"重新分段。

设计契约（与 segmenter.py 的关系）：

  - segmenter.py 是 chunk-internal 的"原始切分"，输出每个 chunk 的 ASR segment
  - asr_merge.py 把多 chunk 拼成全局 timeline
  - **本模块是 SRT/VTT 渲染前的可选 post-processor**：
    输入 RemixrSegment[]（已带 speaker、word 时间戳），输出 SubtitleCue[]
  - **只影响字幕，不动 transcript.{voxkit,raw}.json**——JSON 是 ASR 真实产出，
    重切是字幕渲染优化，两者语义不同不应混用

英文路径（有 word-level timestamp）：

  1. 拍平所有 word 到 (text, start, end, speaker)；speaker 继承自 segment
  2. 按 speaker 分块（连续同 speaker 归一组），避免跨说话人合并
  3. 每个 speaker 块：拼文本 → pysbd 切句 → 字符 span 反查到 word 区间
  4. 句子超物理上限 → split_long 在标点 / 韵律 gap 处贪婪切分
  5. 短 cue 合并到同 speaker 邻居（避免闪现 0.5s 字幕）
  6. enforce_monotonic 钳住跨 cue 时间倒挂（whisper word ts 偶尔重叠）

CJK 路径（whisper.cpp 不输出 word-level timestamp）：

  1. 把每个 Whisper phrase segment 当作字幕打包原子，避免切进词/专名内部
  2. 在 segment 内已有句末标点 / 分号时拆成更细 phrase atom
  3. 按 speaker 连续块处理，永不跨 speaker 合并
  4. 根据 CJK 专用字符数 / 时长 / CPS 阈值打包相邻 phrase
  5. 只有单个 phrase 自己超物理上限时，才在 phrase 内做字符时间插值拆分
  6. 短 cue 合并到同 speaker 邻居 + enforce_monotonic
  7. 若输入时间或文本无法可靠插值，退回 passthrough + 短合并

参数（:class:`ResegmentParams`）经实验在 EN podcast 上调校：
综合分 baseline 48.5 → resegment 67.5，物理分 31.4 → 83.0
（详见 docs/semantic-resegment.md / tmp/pysbd-experiment 历史）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

from voxkit.core.constants import CJK_LANGUAGES
from voxkit.core.word_classes import is_trailing_bad
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


@dataclass(frozen=True)
class _CjkChar:
    char: str
    start: float
    end: float
    speaker: str | None
    segment_index: int


@dataclass(frozen=True)
class _CjkAtom:
    text: str
    start: float
    end: float
    speaker: str | None
    segment_index: int


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
        - CJK 语言（zh/ja/yue/ko）→ phrase-aware 打包；过长 phrase 才字符级拆分
        - 非 CJK 但 segments[0].words 为空 → 同样 pass-through（防御）
        - 否则走 pysbd 重切

    pysbd import 失败 → 抛 ImportError，由 caller 决定 warn-once + fallback
    （本模块不吞异常，避免静默降级）。
    """
    p = params or ResegmentParams()

    if not segments:
        return []

    # CJK 路径没有 word-level timestamp。先尊重 Whisper phrase 边界做字幕打包；
    # 只有单个 phrase 自己超过硬约束时，才在该 phrase 内做字符时间插值。
    if language.lower() in CJK_LANGUAGES:
        cues = _resegment_cjk_phrase_level(segments, p)
        return _enforce_monotonic(cues)

    # no-word-timestamp 防御路径：1:1 包装 + 同 speaker 短合并 + 单调钳位。
    if not segments[0].words:
        cues = _passthrough(segments)
        cues = _merge_too_short(cues, p)
        return _enforce_monotonic(cues)

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


def _passthrough_merged(segments: Sequence[RemixrSegment], p: ResegmentParams) -> list[SubtitleCue]:
    """Fallback for inputs where character interpolation would be misleading.

    The public API has no diagnostics channel, so keeping the fallback isolated
    makes the reason auditable in code: invalid duration / empty interpolatable
    text / no generated char tokens.
    """
    return _merge_too_short(_passthrough(segments), p)


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


# ── CJK 字符级重切 ──────────────────────────────────────────────────────────


_CJK_SENTENCE_END = frozenset("。！？!?")
_CJK_STRONG_BREAK = frozenset("；;")
_CJK_MEDIUM_BREAK = frozenset("，、：:")

_CJK_DEFAULT_MAX_CHARS = 42
# 软上限：触发 cue flush 的字符阈值。调到 18 是为了 vlog/短视频风格的中文
# 无标点输入——whisper.cpp 中文 ASR 不带标点，原 28 字符的目标会把多个气口
# 合并成一个 cue（小宁子样本 avg 23 char/cue vs 人工金标 11）。
# 详见 docs/eval-baseline-observations.md。
_CJK_DEFAULT_SOFT_MAX_CHARS = 18
_CJK_DEFAULT_MAX_CPS = 18.0


def _cjk_limits(p: ResegmentParams) -> tuple[int, int, float]:
    """Use tighter default reading limits for CJK than for Latin subtitles."""
    max_chars = min(p.max_chars, _CJK_DEFAULT_MAX_CHARS)
    soft_max_chars = min(p.soft_max_chars, _CJK_DEFAULT_SOFT_MAX_CHARS, max_chars)
    max_cps = min(p.max_cps, _CJK_DEFAULT_MAX_CPS)
    return max_chars, soft_max_chars, max_cps


def _estimate_char_time(
    seg_start: float,
    seg_end: float,
    text_len: int,
    char_index: int,
) -> float:
    return seg_start + (char_index / text_len) * (seg_end - seg_start)


def _build_cjk_atoms(segments: Sequence[RemixrSegment]) -> list[_CjkAtom] | None:
    """Build phrase atoms, splitting at sentence-end / strong / medium punctuation.

    把 ``_CJK_MEDIUM_BREAK = ，、：:`` 也作为切点的根因：whisper.cpp 中文 ASR
    本身不输出标点，第一 pass reseg 时 segment 内没有 medium 标点可切，行为
    与「只切句末」一致；但当输入已带标点时（proofread 后的双 pass 场景、或
    带 prompt 引导的 ASR），逗号承载了 ~80% 气口边界，全部忽略会让 cue 仍
    被打包过粗。详见 docs/eval-baseline-observations.md §2.2 与 Phase 3 实验。
    """
    atoms: list[_CjkAtom] = []
    for seg_idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        start = float(seg.start)
        end = float(seg.end)
        if end <= start:
            return None

        span_start = 0
        text_len = len(text)
        for i, ch in enumerate(text):
            if (
                ch not in _CJK_SENTENCE_END
                and ch not in _CJK_STRONG_BREAK
                and ch not in _CJK_MEDIUM_BREAK
            ):
                continue
            part = text[span_start : i + 1].strip()
            if part:
                atoms.append(
                    _CjkAtom(
                        text=part,
                        start=_estimate_char_time(start, end, text_len, span_start),
                        end=_estimate_char_time(start, end, text_len, i + 1),
                        speaker=seg.speaker,
                        segment_index=seg_idx,
                    )
                )
            span_start = i + 1

        tail = text[span_start:].strip()
        if tail:
            atoms.append(
                _CjkAtom(
                    text=tail,
                    start=_estimate_char_time(start, end, text_len, span_start),
                    end=end,
                    speaker=seg.speaker,
                    segment_index=seg_idx,
                )
            )
    return atoms


def _group_cjk_atoms_by_speaker(atoms: list[_CjkAtom]) -> list[list[_CjkAtom]]:
    blocks: list[list[_CjkAtom]] = []
    cur: list[_CjkAtom] = []
    for atom in atoms:
        if cur and cur[-1].speaker != atom.speaker:
            blocks.append(cur)
            cur = []
        cur.append(atom)
    if cur:
        blocks.append(cur)
    return blocks


def _cjk_atom_text(atoms: list[_CjkAtom]) -> str:
    text = ""
    for atom in atoms:
        text = _join_cue_text(text, atom.text)
    return text


def _cjk_atoms_to_cue(atoms: list[_CjkAtom]) -> SubtitleCue:
    return SubtitleCue(
        start=atoms[0].start,
        end=atoms[-1].end,
        speaker=atoms[0].speaker,
        text=_cjk_atom_text(atoms),
    )


def _cjk_atom_needs_split(atom: _CjkAtom, p: ResegmentParams) -> bool:
    max_chars, _soft_max_chars, max_cps = _cjk_limits(p)
    dur = atom.end - atom.start
    if dur <= 0:
        return False
    chars = len(atom.text)
    return chars > max_chars or dur > p.max_dur_s or chars / dur > max_cps


def _split_cjk_atom(atom: _CjkAtom, p: ResegmentParams) -> list[SubtitleCue]:
    chars = _build_cjk_chars(
        [
            RemixrSegment(
                id=f"cjk_atom_{atom.segment_index}",
                speaker=atom.speaker or "Speaker A",
                start=atom.start,
                end=atom.end,
                text=atom.text,
                words=[],
            )
        ]
    )
    if chars is None or not chars:
        return [SubtitleCue(atom.start, atom.end, atom.speaker, atom.text)]
    return [_cjk_chars_to_cue(chunk) for chunk in _split_cjk_long(chars, p)]


def _cjk_atoms_fit(atoms: list[_CjkAtom], p: ResegmentParams) -> bool:
    max_chars, _soft_max_chars, max_cps = _cjk_limits(p)
    dur = atoms[-1].end - atoms[0].start
    if dur <= 0:
        return False
    chars = len(_cjk_atom_text(atoms))
    return chars <= max_chars and dur <= p.max_dur_s and chars / dur <= max_cps


def _cjk_atoms_exceed_soft(atoms: list[_CjkAtom], p: ResegmentParams) -> bool:
    _max_chars, soft_max_chars, _max_cps = _cjk_limits(p)
    return len(_cjk_atom_text(atoms)) > soft_max_chars


def _is_short_latin_atom(atom: _CjkAtom) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+", atom.text)) and len(atom.text) <= 4


def _should_soft_flush(atoms: list[_CjkAtom], p: ResegmentParams) -> bool:
    if not atoms:
        return False
    _max_chars, soft_max_chars, _max_cps = _cjk_limits(p)
    dur = atoms[-1].end - atoms[0].start
    text = _cjk_atom_text(atoms)
    if text and text[-1] in _CJK_SENTENCE_END and dur >= p.min_dur_s:
        return True
    return dur >= p.min_dur_s and len(text) >= soft_max_chars


def _pack_cjk_atom_block(atoms: list[_CjkAtom], p: ResegmentParams) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    cur: list[_CjkAtom] = []

    def flush_cur() -> None:
        nonlocal cur
        if cur:
            cues.append(_cjk_atoms_to_cue(cur))
            cur = []

    i = 0
    while i < len(atoms):
        atom = atoms[i]
        nxt = atoms[i + 1] if i + 1 < len(atoms) else None

        if _cjk_atom_needs_split(atom, p):
            flush_cur()
            cues.extend(_split_cjk_atom(atom, p))
            i += 1
            continue

        if not cur:
            cur = [atom]
            if _should_soft_flush(cur, p):
                flush_cur()
            i += 1
            continue

        if (
            nxt is not None
            and _is_short_latin_atom(atom)
            and atom.speaker == nxt.speaker
            and _cjk_atoms_fit([atom, nxt], p)
            and not _cjk_atoms_fit(cur + [atom, nxt], p)
        ):
            flush_cur()
            cur = [atom]
            i += 1
            continue

        candidate = cur + [atom]
        if not _cjk_atoms_fit(candidate, p):
            flush_cur()
            cur = [atom]
        elif _cjk_atoms_exceed_soft(candidate, p) and (
            cur[-1].end - cur[0].start
        ) >= p.min_dur_s:
            flush_cur()
            cur = [atom]
        else:
            cur = candidate

        if _should_soft_flush(cur, p):
            flush_cur()
        i += 1

    flush_cur()
    return cues


def _resegment_cjk_phrase_level(
    segments: Sequence[RemixrSegment],
    p: ResegmentParams,
) -> list[SubtitleCue]:
    """Pack CJK Whisper phrases into readable cues without splitting words."""
    atoms = _build_cjk_atoms(segments)
    if atoms is None or not atoms:
        return _passthrough_merged(segments, p)
    if not any(_looks_cjk_text(atom.text) for atom in atoms):
        return _passthrough_merged(segments, p)

    cues: list[SubtitleCue] = []
    for block in _group_cjk_atoms_by_speaker(atoms):
        cues.extend(_pack_cjk_atom_block(block, p))
    if not cues:
        return _passthrough_merged(segments, p)
    return _merge_too_short(cues, p)


def _build_cjk_chars(segments: Sequence[RemixrSegment]) -> list[_CjkChar] | None:
    """Build subtitle-only char tokens with linear time inside each segment.

    Return ``None`` when interpolation would be misleading: a non-empty text
    segment must have finite, positive duration. Blank segments are ignored.
    """
    chars: list[_CjkChar] = []
    for seg_idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        start = float(seg.start)
        end = float(seg.end)
        dur = end - start
        if dur <= 0:
            return None
        n = len(text)
        if n <= 0:
            continue
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            ch_start = start + (i / n) * dur
            ch_end = start + ((i + 1) / n) * dur
            chars.append(
                _CjkChar(
                    char=ch,
                    start=ch_start,
                    end=ch_end,
                    speaker=seg.speaker,
                    segment_index=seg_idx,
                )
            )
    return chars


def _group_cjk_chars_by_speaker(chars: list[_CjkChar]) -> list[list[_CjkChar]]:
    """Speaker turns are hard boundaries for CJK just like for word-level EN."""
    blocks: list[list[_CjkChar]] = []
    cur: list[_CjkChar] = []
    for ch in chars:
        if cur and cur[-1].speaker != ch.speaker:
            blocks.append(cur)
            cur = []
        cur.append(ch)
    if cur:
        blocks.append(cur)
    return blocks


def _cjk_chunk_metrics(chars: list[_CjkChar]) -> tuple[float, int]:
    if not chars:
        return (0.0, 0)
    return (chars[-1].end - chars[0].start, len(chars))


def _cjk_need_split(chars: list[_CjkChar], p: ResegmentParams) -> bool:
    dur, count = _cjk_chunk_metrics(chars)
    if dur <= 0:
        return False
    return dur > p.max_dur_s or count > p.max_chars or count / dur > p.max_cps


def _is_latin_word_interior(chars: list[_CjkChar], i: int) -> bool:
    """切在 char[i] 之后是否落在一个连续拉丁词的内部。

    用于阻止 `_split_cjk_long` 把 'Steam' 这类词切成 'S t' + 'eam'。
    判定：char[i] 是 ASCII 字母 **且** char[i+1] 是 ASCII 字母。
    """
    if i < 0 or i + 1 >= len(chars):
        return False
    a = chars[i].char
    b = chars[i + 1].char
    return a.isascii() and a.isalpha() and b.isascii() and b.isalpha()


def _cjk_break_weight(chars: list[_CjkChar], i: int, p: ResegmentParams) -> float:
    """Weight for a potential boundary after char ``i``."""
    if i < 0 or i >= len(chars) - 1:
        return 0.0

    cur = chars[i]
    nxt = chars[i + 1]
    if cur.speaker != nxt.speaker:
        return 1e6

    weight = 0.0
    if cur.char in _CJK_SENTENCE_END:
        weight = max(weight, 100.0)
    elif cur.char in _CJK_STRONG_BREAK:
        weight = max(weight, 70.0)
    elif cur.char in _CJK_MEDIUM_BREAK:
        weight = max(weight, 45.0)

    gap = max(0.0, nxt.start - cur.end)
    if cur.segment_index != nxt.segment_index and gap >= p.prosody_gap_s:
        scaled = p.prosody_gap_weight * (1 + min(1.5, gap) / 1.5)
        weight = max(weight, 60.0 + scaled)
    elif gap > 0:
        weight = max(weight, gap)

    return weight


def _split_cjk_long(chars: list[_CjkChar], p: ResegmentParams) -> list[list[_CjkChar]]:
    """Split a CJK char run using punctuation/gap first, physical limits second."""
    if not _cjk_need_split(chars, p) or len(chars) < 2:
        return [chars]

    total_chars = len(chars)
    dur_total = chars[-1].end - chars[0].start
    n_by_chars = -(-total_chars // p.max_chars)
    n_by_dur = -(-int(dur_total * 1000) // int(p.max_dur_s * 1000))
    n_by_cps = -(-total_chars // max(1, int(p.max_cps * p.max_dur_s)))
    n_chunks = max(2, n_by_chars, n_by_dur, n_by_cps)
    target_size = total_chars / n_chunks

    chunks: list[list[_CjkChar]] = []
    start = 0
    for chunk_no in range(n_chunks - 1):
        remaining_chunks = n_chunks - chunk_no
        min_remaining = remaining_chunks - 1
        if start >= total_chars - min_remaining:
            break

        target_idx = start + target_size - 1
        best_idx = -1
        best_score = -1e9
        search_end = total_chars - min_remaining - 1
        for i in range(start, search_end + 1):
            # 永不在拉丁词内部切（防 'Steam' → 'S t' + 'eam'）
            if _is_latin_word_interior(chars, i):
                continue
            chunk = chars[start : i + 1]
            dur, count = _cjk_chunk_metrics(chunk)
            if dur <= 0:
                continue
            over_hard = dur > p.max_dur_s or count > p.max_chars or count / dur > p.max_cps
            if count > p.max_chars * _HARD_CHAR_RATIO or dur > p.max_dur_s * _HARD_DUR_RATIO:
                continue
            weight = _cjk_break_weight(chars, i, p)
            offset = max(0.0, abs(i - target_idx) - _OFFSET_TOL)
            penalty = _OVER_HARD_PENALTY if over_hard else 0.0
            score = weight * _WEIGHT_BONUS - offset - penalty
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            # No weighted boundary fits the hard ratios. Cut by the tightest
            # physical constraint, still on a char boundary.
            max_by_cps = max(1, int(p.max_cps * max(0.05, p.max_dur_s)))
            width = max(1, min(p.max_chars, max_by_cps, int(target_size)))
            best_idx = min(start + width - 1, search_end)
            # Fallback 也不能切到拉丁词内部：往后移到最近的非拉丁边界
            while (
                best_idx < search_end and _is_latin_word_interior(chars, best_idx)
            ):
                best_idx += 1

        chunks.append(chars[start : best_idx + 1])
        start = best_idx + 1

    if start < total_chars:
        chunks.append(chars[start:])

    final: list[list[_CjkChar]] = []
    for chunk in chunks:
        if _cjk_need_split(chunk, p) and len(chunk) >= 2:
            final.extend(_split_cjk_long(chunk, p))
        else:
            final.append(chunk)
    return [c for c in final if c]


def _split_cjk_block(chars: list[_CjkChar], p: ResegmentParams) -> list[list[_CjkChar]]:
    chunks: list[list[_CjkChar]] = []
    start = 0
    for i in range(len(chars) - 1):
        weight = _cjk_break_weight(chars, i, p)
        cur = chars[start : i + 1]
        should_break = weight >= 40.0 or _cjk_need_split(cur, p)
        if should_break:
            chunks.extend(_split_cjk_long(cur, p))
            start = i + 1
    if start < len(chars):
        chunks.extend(_split_cjk_long(chars[start:], p))
    return chunks


def _cjk_chars_to_cue(chars: list[_CjkChar]) -> SubtitleCue:
    return SubtitleCue(
        start=chars[0].start,
        end=chars[-1].end,
        speaker=chars[0].speaker,
        text="".join(ch.char for ch in chars).strip(),
    )


def _resegment_cjk_char_level(
    segments: Sequence[RemixrSegment],
    p: ResegmentParams,
) -> list[SubtitleCue]:
    """CJK subtitle-only semantic resegmentation using char interpolation.

    This path deliberately does not create or mutate ``RemixrWord`` objects.
    When interpolation is not defensible, return the old passthrough behaviour
    plus short-cue merging.
    """
    chars = _build_cjk_chars(segments)
    if chars is None or not chars:
        return _passthrough_merged(segments, p)

    cues: list[SubtitleCue] = []
    for block in _group_cjk_chars_by_speaker(chars):
        for chunk in _split_cjk_block(block, p):
            if chunk:
                cues.append(_cjk_chars_to_cue(chunk))
    if not cues:
        return _passthrough_merged(segments, p)
    return _merge_too_short(cues, p)


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
    """每个 word 之后的可切性权重；底分用 gap 实数让最长 gap 也能脱颖而出。

    `trailing-bad token`（介词 / 冠词 / 连词 / 助动词 / 缩略 will/would 等）
    后面的切点会被打 0.2× 折扣——不完全禁止（偶尔仍是最优），但优先选别处。
    标点 / 句末 gap 等高分切点不受此折扣影响（它们走 `max()` 路径，折扣只压
    "底分 gap 救场" 一类弱切点）。
    """
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
            weights[i] = max(weights[i], gap * 1.0)
        # trailing-bad 折扣：词 i 若是介词/连词/助词等，切到 i 之后会让当前
        # cue 以 "of/the/is/will" 这种半截结尾，体感破碎。压 0.2×。
        if is_trailing_bad(words[i].word):
            weights[i] *= 0.2
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


def _looks_cjk_text(text: str) -> bool:
    return any(
        "\u3400" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def _join_cue_text(a: str, b: str) -> str:
    a = a.strip()
    b = b.strip()
    if not a:
        return b
    if not b:
        return a
    if _looks_cjk_text(a) or _looks_cjk_text(b):
        return a + b
    return a + " " + b


#: 极短 cue 阈值：低于这个时长用户读不到，必须合并；触发后放宽合并物理上限。
_FLASH_CUE_DUR_S = 0.5
_FLASH_RELAX_CPS = 1.5    # cps 上限放宽到 1.5×
_FLASH_RELAX_CHARS = 1.2  # chars 上限放宽到 1.2×


def _can_merge(a: SubtitleCue, b: SubtitleCue, p: ResegmentParams) -> bool:
    """允许相邻 cue 合并的物理可行性。

    跨 speaker 一律拒绝（speaker 不变量比物理舒适度高）。

    极短 cue (< 0.5s) 必须合并——用户根本读不到 0.5s 闪屏，让合并后字幕略超
    cps/chars 远好于继续闪屏。所以 ``a`` 或 ``b`` 任一为 flash 时放宽限。
    `max_dur_s` 不放宽，否则可能合出 10s+ 的长字幕；同时 0.5s + 邻居也几乎
    不会撞到 max_dur_s 真实边界。
    """
    if a.speaker != b.speaker:
        return False
    merged_dur = b.end - a.start
    merged_chars = len(_join_cue_text(a.text, b.text))
    merged_cps = merged_chars / merged_dur if merged_dur > 0 else float("inf")

    flash_present = (a.end - a.start) < _FLASH_CUE_DUR_S or (b.end - b.start) < _FLASH_CUE_DUR_S
    cps_limit = p.max_cps * _FLASH_RELAX_CPS if flash_present else p.max_cps
    chars_limit = int(p.max_chars * _FLASH_RELAX_CHARS) if flash_present else p.max_chars

    return (
        merged_dur <= p.max_dur_s
        and merged_chars <= chars_limit
        and merged_cps <= cps_limit
    )


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
                    text=_join_cue_text(cur.text, nxt.text),
                )
                out[i : i + 2] = [merged]
            else:  # "prev"
                prv = out[i - 1]
                merged = SubtitleCue(
                    start=prv.start, end=cur.end, speaker=cur.speaker,
                    text=_join_cue_text(prv.text, cur.text),
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
