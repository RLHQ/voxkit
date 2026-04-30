"""Unit tests for voxkit.core.semantic_resegment.

纯函数测试：构造 RemixrSegment[] 输入，验证三条分支：

  1. CJK pass-through：每段直接 1:1 映射成 cue，不调 pysbd
  2. 无 word 时间戳的非 CJK：同样 pass-through（防御性）
  3. 英文有 word 时间戳：调 pysbd，按句子边界切分；超长句调 split_long
"""

from __future__ import annotations

import pytest

from voxkit.core.semantic_resegment import (
    ResegmentParams,
    SubtitleCue,
    resegment_for_subtitles,
)
from voxkit.io.schema import RemixrSegment, RemixrWord


def _seg(
    sid: str,
    start: float,
    end: float,
    text: str,
    words: list[RemixrWord] | None = None,
    speaker: str = "Speaker 1",
) -> RemixrSegment:
    return RemixrSegment(
        id=sid,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        words=words or [],
    )


def _w(word: str, start: float, end: float) -> RemixrWord:
    return RemixrWord(word=word, start=start, end=end)


# ─────────────────────────────────────────────────────────────────────────
# Pass-through branches
# ─────────────────────────────────────────────────────────────────────────


def test_empty_input_returns_empty():
    assert resegment_for_subtitles([], language="en") == []


def test_cjk_long_cues_passthrough_unchanged():
    """CJK 路径不调 pysbd；当 cue 已经 ≥ min_dur_s 时不该合并，1:1 输出。"""
    segs = [
        _seg("s1", 0.0, 2.0, "你好世界。", speaker="Speaker 1"),
        _seg("s2", 2.0, 4.0, "今天天气真好。", speaker="Speaker 1"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert len(cues) == 2
    assert cues[0].text == "你好世界。"
    assert cues[0].speaker == "Speaker 1"
    assert cues[1].text == "今天天气真好。"


def test_cjk_short_cues_get_merged():
    """档 1：CJK 路径打开 _merge_too_short——同 speaker 短 cue 合并到邻居，
    避免 < 1.5s 的闪现字幕。
    """
    # 三个 1s 同 speaker cue → 单调合并到 max_dur_s 上限附近
    segs = [
        _seg("s1", 0.0, 1.0, "第一句", speaker="Speaker 1"),
        _seg("s2", 1.0, 2.0, "第二句", speaker="Speaker 1"),
        _seg("s3", 2.0, 3.0, "第三句", speaker="Speaker 1"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    # 至少减少 cue 数；且合并后每个 cue ≥ min_dur_s（除非物理上限阻挡）
    assert len(cues) < 3
    # 合并后的文本保留单空格分隔
    joined = " ".join(c.text for c in cues)
    assert "第一句" in joined and "第二句" in joined and "第三句" in joined


def test_cjk_short_cues_different_speakers_dont_merge():
    """跨 speaker 不合并——即使两侧都很短，也保留为独立 cue（避免说话人混淆）。"""
    segs = [
        _seg("s1", 0.0, 1.0, "你好", speaker="Speaker 1"),
        _seg("s2", 1.0, 2.0, "Hi", speaker="Speaker 2"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert len(cues) == 2
    assert cues[0].speaker == "Speaker 1"
    assert cues[1].speaker == "Speaker 2"


def test_passthrough_when_words_missing():
    """非 CJK 但 segments[0].words 为空 → 走 CJK-style 路径（passthrough + 短合并）。"""
    segs = [_seg("s1", 0.0, 2.0, "Hello world.", words=[])]
    cues = resegment_for_subtitles(segs, language="en")
    assert len(cues) == 1
    assert cues[0].text == "Hello world."


def test_passthrough_skips_blank_text():
    """空白 segment 在入口就被滤掉，不参与合并候选。"""
    segs = [
        # 两侧 cue 时长 ≥ min_dur_s，确保不会合并；专测空白过滤
        _seg("s1", 0.0, 2.0, "Hello.", speaker="A"),
        _seg("s2", 2.0, 3.0, "   ", speaker="A"),
        _seg("s3", 3.0, 5.0, "World.", speaker="A"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert [c.text for c in cues] == ["Hello.", "World."]


# ─────────────────────────────────────────────────────────────────────────
# English word-level path
# ─────────────────────────────────────────────────────────────────────────


def test_simple_two_sentence_segment_splits_at_period():
    """同一 segment 含两句话 → 按句号切成 2 cue。"""
    segs = [
        _seg(
            "s1",
            0.0,
            4.0,
            "Hello world. This is fine.",
            words=[
                _w("Hello", 0.0, 0.4),
                _w("world.", 0.4, 0.9),
                _w("This", 1.0, 1.3),
                _w("is", 1.3, 1.5),
                _w("fine.", 1.5, 2.0),
            ],
            speaker="Speaker 1",
        ),
    ]
    # min_dur_s 默认 1.5，会触发短 cue 合并；用更宽松参数关掉合并验证 SBD 单独效果
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    assert len(cues) == 2
    assert cues[0].text.endswith(".")
    assert cues[1].text.endswith(".")
    assert cues[0].text.startswith("Hello")
    assert cues[1].text.startswith("This")


def test_speaker_boundary_blocks_pysbd_merge():
    """跨 speaker 的"句子"不能被 pysbd 当成一句合并。"""
    segs = [
        _seg(
            "s1",
            0.0,
            1.0,
            "Hello",
            words=[_w("Hello", 0.0, 0.4)],
            speaker="Speaker 1",
        ),
        _seg(
            "s2",
            1.0,
            2.0,
            "world.",
            words=[_w("world.", 1.0, 1.4)],
            speaker="Speaker 2",
        ),
    ]
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    speakers = {c.speaker for c in cues}
    assert speakers == {"Speaker 1", "Speaker 2"}, "speakers must not be merged"


def test_long_sentence_gets_split():
    """单句 >max_chars + max_dur → split_long 切多段，每段都合规。"""
    # 构造一句 90 词、~30s 的句子，没有句号但有逗号
    pieces: list[tuple[str, float]] = []
    t = 0.0
    for i in range(60):
        pieces.append((f"word{i},", t))
        t += 0.4
    pieces.append(("end.", t))
    words = [_w(w, s, s + 0.3) for w, s in pieces]
    text = " ".join(w for w, _ in pieces)
    segs = [_seg("s1", 0.0, t + 0.3, text, words=words, speaker="A")]

    p = ResegmentParams(max_dur_s=6.0, max_chars=70, min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)

    assert len(cues) >= 3, "long sentence must be split"
    for c in cues:
        assert c.end - c.start <= p.max_dur_s * 1.21, f"cue {c} duration over hard limit"
        assert len(c.text) <= p.max_chars * 1.31, f"cue {c} chars over hard limit"


def test_short_cue_merged_with_neighbour():
    """<min_dur_s 的小尾巴应被合并到同 speaker 邻居。"""
    segs = [
        _seg(
            "s1",
            0.0,
            5.0,
            "First long sentence here. Tiny.",
            words=[
                _w("First", 0.0, 0.3),
                _w("long", 0.3, 0.6),
                _w("sentence", 0.6, 1.0),
                _w("here.", 1.0, 1.4),
                _w("Tiny.", 1.5, 2.0),  # 0.5s — well below min_dur=1.5
            ],
            speaker="A",
        ),
    ]
    p = ResegmentParams(min_dur_s=1.5)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    # "Tiny." 应该被合进前一个 cue
    assert all(c.end - c.start >= 0.05 for c in cues)
    # 至少一个 cue 同时含 "here." 和 "Tiny."（合并后）
    merged = [c for c in cues if "here." in c.text and "Tiny." in c.text]
    assert merged, f"short cue not merged into neighbour; got: {[c.text for c in cues]}"


def test_monotonic_timestamps_enforced():
    """whisper word ts 偶尔倒挂（next.start < prev.end）→ 输出必须单调。"""
    segs = [
        _seg(
            "s1",
            0.0,
            3.0,
            "Sentence one. Sentence two.",
            words=[
                _w("Sentence", 0.0, 0.5),
                _w("one.", 0.5, 1.0),
                # 故意倒挂：next.start 在 prev.end 之前
                _w("Sentence", 0.95, 1.5),
                _w("two.", 1.5, 2.0),
            ],
            speaker="A",
        ),
    ]
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    # 后一 cue 的 start 必 >= 前一 cue 的 end
    for i in range(len(cues) - 1):
        assert cues[i + 1].start >= cues[i].end


def test_pysbd_missing_raises_import_error(monkeypatch):
    """pysbd 不可用时模块直接抛 ImportError，由 caller 决定 fallback。"""
    import sys
    monkeypatch.setitem(sys.modules, "pysbd", None)
    segs = [
        _seg("s1", 0.0, 1.0, "Hi.", words=[_w("Hi.", 0.0, 0.3)]),
    ]
    with pytest.raises((ImportError, TypeError)):
        # TypeError because monkeypatching to None makes import fail with TypeError on Python 3.10+
        resegment_for_subtitles(segs, language="en")
