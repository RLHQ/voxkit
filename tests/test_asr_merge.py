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
    """14. Two overlapping chunks: chunk1 takes over the overlap region.

    chunk0: 0..600 abs, contains seg ending at 597 and seg ending at 600
    chunk1: 595..1195 abs (chunk_start_secs=595), 4 segs spanning 0..600 relative
    Expect: chunk0's segments whose start >= 594.5 are dropped, then chunk1
    is appended.
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 200.0, "early"),
            _seg("a1", 200.0, 400.0, "middle"),
            _seg("a2", 400.0, 597.0, "late"),
            _seg("a3", 597.0, 600.0, "tail"),  # falls in chunk1's territory
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[
            _seg("b0", 0.0, 100.0, "c1-first"),  # absolute 595..695
            _seg("b1", 100.0, 300.0, "c1-second"),
            _seg("b2", 300.0, 500.0, "c1-third"),
            _seg("b3", 500.0, 600.0, "c1-fourth"),
        ],
        chunk_start_secs=595.0,
    )
    merged, notes = merge_chunks([chunk0, chunk1])

    # chunk0's tail (start=597) >= 595 - 0.5 → dropped.
    # chunk0 retains a0/a1/a2 (3 segments) ending at 597.
    # chunk1 first segment start=595 >= 597 - 0.5 = 596.5? No, 595 < 596.5,
    # so b0 is dropped. b1 abs=695 >= 596.5 ✓
    # Actually: lastEnd = 597 (a2.end). b0 abs.start=595 < 597 - 0.5 → dropped.
    # b1 abs.start=695 >= 596.5 → kept. Then b2 abs.start=895 ≥ 795-0.5? yes. b3 too.
    # Expected: a0, a1, a2, b1, b2, b3 = 6 segments
    assert len(merged) == 6
    assert [s.id for s in merged] == [
        "seg_001",
        "seg_002",
        "seg_003",
        "seg_004",
        "seg_005",
        "seg_006",
    ]
    # Verify chunk1 segments are absolute-offset
    assert merged[3].start == 695.0  # b1 abs.start
    assert merged[5].end == 1195.0  # b3 abs.end
    # Timeline should be clean (no out_of_order / overlap notes)
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


def test_merge_drops_chunk0_segments_inside_chunk1_territory():
    """If a chunk0 segment starts inside chunk1's region, chunk1 wins.

    Defends the dedup direction (chunk i is "primary" for [chunk_start_i, chunk_end_i)).
    """
    chunk0 = ChunkResult(
        chunk_index=0,
        segments=[
            _seg("a0", 0.0, 100.0, "before"),
            _seg("a1", 596.0, 599.0, "tail-in-overlap"),  # start >= 595 → drop
        ],
        chunk_start_secs=0.0,
    )
    chunk1 = ChunkResult(
        chunk_index=1,
        segments=[_seg("b0", 0.0, 100.0, "primary")],
        chunk_start_secs=595.0,
    )
    merged, _ = merge_chunks([chunk0, chunk1])
    # a0 kept, a1 dropped (>= 594.5), b0 kept (abs 595, lastEnd=100, 595>=99.5)
    assert len(merged) == 2
    assert merged[0].text == "before"
    assert merged[1].text == "primary"
    assert merged[1].start == 595.0


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
