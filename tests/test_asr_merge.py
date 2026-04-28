"""Tests for voxkit.core.asr_merge — chunk transcript merge + word-offset regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxkit.core.asr_merge import (
    OVERLAP_DEDUP_TOLERANCE_SECS,
    ChunkResult,
    MergeNote,
    merge_chunks,
    offset_segment,
    validate_timeline_continuity,
    write_merge_log,
)
from voxkit.io.schema import TranscriptSegment, Word


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _seg(id_, start, end, text, words=None) -> TranscriptSegment:
    """Build a TranscriptSegment with optional words."""
    return TranscriptSegment(id=id_, start=start, end=end, text=text, words=words or [])


def _word(w, s, e) -> Word:
    return Word(word=w, start=s, end=e)


# ─────────────────────────────────────────────────────────────────────────────
# offset_segment
# ─────────────────────────────────────────────────────────────────────────────


def test_offset_segment_basic_no_words():
    """1. Segment with no words: start/end shift, words still []."""
    out = offset_segment(_seg("x", 0.0, 5.0, "hi"), 600.0)
    assert out.start == 600.0
    assert out.end == 605.0
    assert out.words == []
    assert out.text == "hi"
    assert out.id == "x"


def test_offset_segment_with_single_word():
    """2. Segment with one word: word also shifted."""
    out = offset_segment(
        _seg("x", 0.0, 5.0, "hi", [_word("hi", 0.0, 5.0)]),
        600.0,
    )
    assert out.words[0].word == "hi"
    assert out.words[0].start == 600.0
    assert out.words[0].end == 605.0


def test_word_offset_regression():
    """3. THE 6-MONTH BUG: words must be offset alongside segment timestamps.

    If this fails, somebody removed the word-offset line in offset_segment.
    """
    seg = _seg(
        "x",
        0.0,
        5.0,
        "hello world",
        [_word("hello", 0.0, 1.0), _word("world", 1.5, 5.0)],
    )
    offset = offset_segment(seg, 600.0)
    assert offset.start == 600.0
    assert offset.end == 605.0
    # The bug: words might stay at 0..5 instead of 600..605
    assert offset.words[0].start == 600.0, "WORD-OFFSET BUG: word.start not shifted"
    assert offset.words[0].end == 601.0
    assert offset.words[1].start == 601.5
    assert offset.words[1].end == 605.0


def test_offset_segment_round_trip():
    """4. offset by +5 then -5 returns to (approximately) original."""
    seg = _seg(
        "x",
        1.234,
        9.876,
        "round trip",
        [_word("round", 1.234, 5.0), _word("trip", 5.5, 9.876)],
    )
    plus = offset_segment(seg, 5.0)
    back = offset_segment(plus, -5.0)
    assert back.start == pytest.approx(seg.start, abs=1e-3)
    assert back.end == pytest.approx(seg.end, abs=1e-3)
    assert back.words[0].start == pytest.approx(seg.words[0].start, abs=1e-3)
    assert back.words[1].end == pytest.approx(seg.words[1].end, abs=1e-3)


def test_offset_segment_negative_delta():
    """5. Negative delta works (caller may subtract)."""
    out = offset_segment(_seg("x", 100.0, 105.0, "hi", [_word("hi", 100.0, 105.0)]), -50.0)
    assert out.start == 50.0
    assert out.end == 55.0
    assert out.words[0].start == 50.0
    assert out.words[0].end == 55.0


def test_offset_segment_does_not_mutate_input():
    """offset_segment must never mutate its input segment or words."""
    seg = _seg("x", 0.0, 5.0, "hi", [_word("hi", 0.0, 5.0)])
    _ = offset_segment(seg, 600.0)
    assert seg.start == 0.0
    assert seg.end == 5.0
    assert seg.words[0].start == 0.0
    assert seg.words[0].end == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# validate_timeline_continuity
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_empty_list():
    """6. Empty list → []."""
    assert validate_timeline_continuity([]) == []


def test_validate_single_segment():
    """7. Single segment → []."""
    assert validate_timeline_continuity([_seg("a", 0.0, 5.0, "x")]) == []


def test_validate_end_aligned_clean():
    """8. Two segments end-aligned (prev.end == curr.start) → no notes."""
    segs = [_seg("a", 0.0, 5.0, "x"), _seg("b", 5.0, 10.0, "y")]
    assert validate_timeline_continuity(segs) == []


def test_validate_small_out_of_order():
    """9. Out-of-order by 0.1s → exactly 1 note kind=out_of_order (not overlap)."""
    segs = [_seg("a", 0.0, 5.0, "x"), _seg("b", 4.9, 10.0, "y")]
    notes = validate_timeline_continuity(segs)
    assert len(notes) == 1
    assert notes[0].kind == "out_of_order"
    assert notes[0].seg_id == "b"


def test_validate_big_overlap_emits_both():
    """10. Big overlap (1.5s) → 2 notes (out_of_order AND overlap)."""
    segs = [_seg("a", 0.0, 5.0, "x"), _seg("b", 3.5, 8.0, "y")]
    notes = validate_timeline_continuity(segs)
    kinds = sorted(n.kind for n in notes)
    assert kinds == ["out_of_order", "overlap"]
    # both reference the same offending seg_id
    assert all(n.seg_id == "b" for n in notes)


# ─────────────────────────────────────────────────────────────────────────────
# merge_chunks
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_empty_chunks():
    """11. Empty chunks → ([], [])."""
    merged, notes = merge_chunks([])
    assert merged == []
    assert notes == []


def test_merge_single_chunk_zero_offset():
    """12. Single chunk with chunk_start_secs=0 → segments re-IDed, no notes."""
    chunk = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a", 0.0, 5.0, "one"),
            _seg("b", 5.0, 10.0, "two"),
            _seg("c", 10.0, 15.0, "three"),
        ],
        chunk_start_secs=0.0,
    )
    merged, notes = merge_chunks([chunk])
    assert len(merged) == 3
    assert [s.id for s in merged] == ["seg_001", "seg_002", "seg_003"]
    assert merged[0].start == 0.0
    assert merged[2].end == 15.0
    assert notes == []


def test_merge_single_chunk_with_offset():
    """13. Single chunk with chunk_start_secs=600 → segments offset by 600."""
    chunk = ChunkResult(
        chunk_index=0,
        segments=[_seg("a", 0.0, 5.0, "x"), _seg("b", 5.0, 10.0, "y")],
        chunk_start_secs=600.0,
    )
    merged, notes = merge_chunks([chunk])
    assert merged[0].start == 600.0
    assert merged[0].end == 605.0
    assert merged[1].start == 605.0
    assert merged[1].end == 610.0
    assert notes == []


def test_merge_two_chunks_with_overlap_dedup():
    """14. Two overlapping chunks: prev-priority — chunk 0 retained, chunk 1
    appends only segments past prev's last_end.

    chunk0: 0..600 abs, segments ending at 600 (a3 covers 597-600 = chunk1 区).
    chunk1: 595..1195 abs, 4 segs each 100s.
    Expected (prev-priority): a0, a1, a2, a3 全保留；chunk 1 first b0 abs=595
    < a3.end=600 → 不 append；b1 abs=695 >= 600-0.5 → append；b2/b3 同理。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 200.0, "early"),
            _seg("a1", 200.0, 400.0, "middle"),
            _seg("a2", 400.0, 597.0, "late"),
            _seg("a3", 597.0, 600.0, "tail"),  # 在 prev-priority 下保留
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            _seg("b0", 0.0, 100.0, "c1-first"),  # absolute 595..695，被 prev 覆盖
            _seg("b1", 100.0, 300.0, "c1-second"),
            _seg("b2", 300.0, 500.0, "c1-third"),
            _seg("b3", 500.0, 600.0, "c1-fourth"),
        ],
        chunk_start_secs=595.0,
    )
    merged, notes = merge_chunks([chunk0, chunk1])

    # Expected: a0, a1, a2, a3, b1, b2, b3 = 7 segments
    assert len(merged) == 7
    assert [s.text for s in merged] == [
        "early", "middle", "late", "tail",
        "c1-second", "c1-third", "c1-fourth",
    ]
    # b1 abs.start = 595 + 100 = 695
    assert merged[4].start == 695.0
    assert merged[6].end == 1195.0
    # No out-of-order: a3.end=600, b1.start=695 → clean
    assert notes == []


def test_merge_word_offset_across_merge():
    """15. End-to-end word offset: chunk1's words must be at absolute time."""
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg(
                "a",
                0.0,
                3.0,
                "hello world",
                [_word("hello", 0.0, 1.0), _word("world", 1.5, 3.0)],
            ),
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            _seg(
                "b",
                2.0,
                5.0,
                "goodbye now",
                [_word("goodbye", 2.0, 3.5), _word("now", 4.0, 5.0)],
            ),
        ],
        chunk_start_secs=600.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    # chunk1's segment is at absolute 602..605, words 602..603.5 / 604..605
    last = merged[-1]
    assert last.start == 602.0
    assert last.end == 605.0
    assert last.words[0].start == 602.0  # the bug check
    assert last.words[0].end == 603.5
    assert last.words[1].start == 604.0
    assert last.words[1].end == 605.0


def test_merge_keeps_chunk0_overlap_segments_under_prev_priority():
    """Prev-priority: chunk0 末尾的 overlap 区 segment 必须保留，chunk1 只补
    chunk0 last_end 之后的内容。

    Background: A/B 实验证实 whisper.cpp 在 chunk 末尾经常完整产出（如
    "differentiation" @ 595.28），chunk 1 暖机后从 chunk_start + 几百 ms 起，
    跳过了 [chunk_start, chunk_1_first_seg.start] 区间。chunk-i-priority 下
    这些内容会被 dedup 误删。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 100.0, "before"),
            _seg("a1", 596.0, 599.0, "tail-in-overlap"),  # prev-priority 保留
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[_seg("b0", 0.0, 100.0, "after-overlap")],  # abs 595..695
        chunk_start_secs=595.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    # a0 + a1 + b0 (b0.start=595 < a1.end=599 → 实际应被跳过；b0 实际 abs=595
    # 在 a1.end=599 之内 → 不 append。这里换个能 append 的例子更清晰。)
    # 此处 b0 abs.start=595 < a1.end=599 - 0.5=598.5 → 不 append
    # 结果: [a0, a1] 两段
    assert texts == ["before", "tail-in-overlap"]


def test_merge_keeps_chunk0_tail_when_chunk1_warmup_loses_text():
    """Regression: chunk 1 warmup-loss 时 chunk 0 末尾合理产出必须保留。

    Background: A/B 实验证实 whisper.cpp 在 chunk 末尾 ~3-5s 经常 early-truncate，
    chunk 1 开头同样可能暖机损失。旧 dedup 用 chunks[i].chunk_start_secs 作切点，
    会把 chunk 0 在 [chunk_start - tol, chunk_end] 的合理产出删光，但 chunk 1
    暖机后才开始产出 → 净损失。修复：用 chunk i 第一个实际产出的 segment.start
    作切点，保留 chunk 0 直到 chunk 1 真正接管的位置。

    场景：chunk 0 [0, 60) 在 abs 56-58 还有合理产出；chunk 1 [55, 115) 暖机后
    从 abs 57.5 才开始产出。新逻辑应保留 chunk 0 的 56-58 segment。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 50.0, "early content"),
            _seg("a1", 56.0, 58.0, "register differentiation"),  # chunk 末尾合理产出
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            # chunk-relative 2.5 → abs 57.5（chunk 1 暖机损失了 [55, 57.5)）
            _seg("b0", 2.5, 5.0, "and language change"),
            _seg("b1", 5.0, 60.0, "rest of chunk one"),
        ],
        chunk_start_secs=55.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    assert "early content" in texts
    assert "register differentiation" in texts, (
        "chunk 0 tail must survive when chunk 1 warmup-loses the overlap zone"
    )
    assert "and language change" in texts
    assert "rest of chunk one" in texts


def test_merge_handles_empty_chunk_i():
    """Defensive: chunk i 完全没产出（极端 silence）时不挂掉，prev 保留。"""
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 50.0, "speech"),
            _seg("a1", 56.0, 58.0, "tail"),
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[],  # 空 chunk
        chunk_start_secs=55.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    # prev-priority：chunk0 全保留
    assert texts == ["speech", "tail"]


# ─────────────────────────────────────────────────────────────────────────────
# merge_chunks — signal-aware overlap arbitration (3 new branches)
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_chunk_i_wins_overlap_when_signal_stronger():
    """Smart overlap: chunk_i 在 overlap 区 word count > prev → chunk_i 接管，
    prev overlap 区被 trim。

    场景：chunk0 末尾 early-truncate（overlap 区只有 1 词 "weak"）；chunk1 在
    同 overlap 区有 4 词 "strong full sentence here"。signal-aware 仲裁应判
    chunk_i 赢，删掉 prev 的 "weak"，保留 chunk_i 的完整产出。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 50.0, "warmup", words=[_word("warmup", 0.0, 50.0)]),
            _seg("a1", 55.0, 60.0, "weak", words=[_word("weak", 55.0, 60.0)]),
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            # chunk-relative 0..5 → abs 55..60，4 词覆盖整个 overlap 区
            _seg(
                "b0", 0.0, 5.0, "strong full sentence here",
                words=[
                    _word("strong", 55.0, 56.0),
                    _word("full", 56.0, 57.0),
                    _word("sentence", 57.0, 58.0),
                    _word("here", 58.0, 60.0),
                ],
            ),
            _seg("b1", 5.0, 60.0, "rest", words=[_word("rest", 60.0, 115.0)]),
        ],
        chunk_start_secs=55.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    assert "warmup" in texts, "prev overlap 区外的 segment 必须保留"
    assert "weak" not in texts, (
        "chunk_i 信号强 (4 words > 1 word) 时 prev overlap 区应被 trim"
    )
    assert "strong full sentence here" in texts, "chunk_i 应接管 overlap 区"
    assert "rest" in texts


def test_merge_prev_wins_overlap_when_scores_tied():
    """Smart overlap: 信号并列 → prev 赢（保守 default），与旧 prev-priority 行为一致。

    场景：chunk0 / chunk1 在 overlap 区 [55, 60) 各有 1 词。并列时不应该来回翻转，
    而是稳定走 prev → chunk_i overlap 区被 last_end 守卫剪枝。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 50.0, "early"),
            _seg("a1", 55.0, 60.0, "tail", words=[_word("tail", 55.0, 60.0)]),
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            _seg("b0", 0.0, 5.0, "alt", words=[_word("alt", 55.0, 60.0)]),
            _seg("b1", 5.0, 60.0, "rest"),
        ],
        chunk_start_secs=55.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    assert "tail" in texts, "并列时 prev 必须保留"
    assert "alt" not in texts, "并列时 chunk_i overlap 区应被 last_end 守卫跳过"
    assert "rest" in texts


def test_merge_no_overlap_appends_directly():
    """Smart overlap: chunk_i.first_seg.start > prev.last.end → 无真重叠分支，
    跳过 score 比较直接 append。等同旧 prev-priority append 行为。
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[_seg("a0", 0.0, 50.0, "before")],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        # b0 chunk-relative 5..15 → abs 60..70，prev 末尾 50 < 60 → 无重叠
        segments=[_seg("b0", 5.0, 15.0, "after")],
        chunk_start_secs=55.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    texts = [s.text for s in merged]
    assert texts == ["before", "after"]


# ─────────────────────────────────────────────────────────────────────────────
# write_merge_log
# ─────────────────────────────────────────────────────────────────────────────


def test_write_merge_log_single_chunk(tmp_path: Path):
    """16. Single chunk → file written, JSON parses, has correct shape."""
    chunk = ChunkResult(
        chunk_index=0,
        segments=[_seg("a", 0.0, 5.0, "x"), _seg("b", 5.0, 10.0, "y")],
        chunk_start_secs=0.0,
    )
    merged, _ = merge_chunks([chunk])
    out = tmp_path / "merge.json"
    write_merge_log([chunk], merged, None, out)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["merged_segments"] == 2
    assert data["tolerance_secs"] == OVERLAP_DEDUP_TOLERANCE_SECS
    assert len(data["chunks"]) == 1
    entry = data["chunks"][0]
    assert entry["index"] == 0
    assert entry["chunk_start_secs"] == 0.0
    assert entry["segments_in"] == 2
    assert entry["segments_dropped"] == []


def test_write_merge_log_multi_chunk_with_drops(tmp_path: Path):
    """17. Multi-chunk with drops_per_chunk dict → drops listed in output."""
    chunks = [
        ChunkResult(
            chunk_index=0,
            segments=[_seg("a", 0.0, 5.0, "x")],
            chunk_start_secs=0.0,
        ),
        ChunkResult(
            chunk_index=1,
            segments=[_seg("b", 0.0, 5.0, "y"), _seg("c", 5.0, 10.0, "z")],
            chunk_start_secs=600.0,
        ),
    ]
    drops = {1: ["b"]}  # chunk1 lost segment "b" at the seam
    merged_dummy: list[TranscriptSegment] = []
    out = tmp_path / "merge.json"
    write_merge_log(chunks, merged_dummy, drops, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    by_idx = {c["index"]: c for c in data["chunks"]}
    assert by_idx[0]["segments_dropped"] == []
    assert by_idx[1]["segments_dropped"] == ["b"]
    assert by_idx[1]["chunk_start_secs"] == 600.0
    assert by_idx[1]["segments_in"] == 2
    assert data["merged_segments"] == 0


def test_write_merge_log_pretty_printed(tmp_path: Path):
    """write_merge_log should produce indent=2 JSON for human review."""
    chunk = ChunkResult(chunk_index=0, segments=[], chunk_start_secs=0.0)
    out = tmp_path / "merge.json"
    write_merge_log([chunk], [], None, out)
    text = out.read_text(encoding="utf-8")
    # indent=2 means we should see nested newlines and 2-space indent
    assert "\n  " in text


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: MergeNote is a frozen dataclass
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_note_is_frozen():
    note = MergeNote(kind="overlap", seg_id="seg_001", detail="hi")
    with pytest.raises(Exception):
        note.kind = "out_of_order"  # type: ignore[misc]
