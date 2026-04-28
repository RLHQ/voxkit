"""Unit tests for ``voxkit.core.align_speakers.assign_speakers``.

Pure-function tests — no subprocess, no IO. Validates the maximum-overlap
speaker assignment, tie-break, label-mode switch, unmatched detection, and
input-immutability contract.
"""

from __future__ import annotations

from voxkit.core.align_speakers import assign_speakers
from voxkit.io.schema import (
    AudioInfo,
    DiarizationOutput,
    Segment,
    SpeakerInfo,
    TranscriptSegment,
)


def _make_diarization(segments: list[Segment]) -> DiarizationOutput:
    """Tiny helper — constructs a DiarizationOutput with placeholder header
    fields and the supplied segment list. ``num_speakers`` is set from the
    distinct ``raw_speaker`` values present in ``segments``.
    """
    raw_ids = sorted({s.raw_speaker for s in segments})
    speakers = [
        SpeakerInfo(
            id=f"Speaker {i + 1}",
            raw_id=raw,
            total_duration_secs=0.0,
        )
        for i, raw in enumerate(raw_ids)
    ]
    return DiarizationOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=60.0),
        device="cpu",
        model="pyannote/speaker-diarization-3.1",
        rtf=0.5,
        elapsed_secs=30.0,
        num_speakers=len(speakers),
        speakers=speakers,
        segments=segments,
    )


def _seg(seg_id: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(id=seg_id, start=start, end=end, text="x")


# ── basics ───────────────────────────────────────────────────────────────


def test_empty_segments_yields_empty_dict():
    dia = _make_diarization(
        [Segment(start=0.0, end=10.0, speaker="Speaker 1", raw_speaker="SPEAKER_00")]
    )
    speaker_by_id, unmatched = assign_speakers([], dia)
    assert speaker_by_id == {}
    assert unmatched == []


def test_empty_diarization_marks_all_unmatched():
    dia = DiarizationOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=10.0),
        device="cpu",
        model="m",
        rtf=0.0,
        elapsed_secs=0.0,
        num_speakers=0,
        speakers=[],
        segments=[],
    )
    segs = [_seg("a", 0.0, 1.0), _seg("b", 1.0, 2.0)]
    speaker_by_id, unmatched = assign_speakers(segs, dia)
    assert speaker_by_id == {}
    assert unmatched == ["a", "b"]


# ── single-overlap mapping ───────────────────────────────────────────────


def test_single_overlap_assigns_speaker():
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
    ])
    segs = [_seg("s1", 1.0, 2.0)]
    speaker_by_id, unmatched = assign_speakers(segs, dia)
    assert speaker_by_id == {"s1": "Speaker 1"}
    assert unmatched == []


def test_no_overlap_segment_is_unmatched():
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
    ])
    segs = [_seg("s_off", 100.0, 105.0)]
    speaker_by_id, unmatched = assign_speakers(segs, dia)
    assert speaker_by_id == {}
    assert unmatched == ["s_off"]


def test_partial_unmatched_partial_matched():
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
    ])
    segs = [
        _seg("a", 1.0, 2.0),    # matched
        _seg("b", 100.0, 101.0),  # unmatched
        _seg("c", 3.0, 4.0),    # matched
    ]
    speaker_by_id, unmatched = assign_speakers(segs, dia)
    assert speaker_by_id == {"a": "Speaker 1", "c": "Speaker 1"}
    assert unmatched == ["b"]


# ── overlap arithmetic ───────────────────────────────────────────────────


def test_longer_overlap_wins_over_shorter():
    """Transcript [4,9] overlaps Speaker 1 [0,5] for 1s and Speaker 2 [5,10]
    for 4s — Speaker 2 should win.
    """
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
        Segment(start=5.0, end=10.0, speaker="Speaker 2", raw_speaker="SPEAKER_02"),
    ])
    segs = [_seg("s", 4.0, 9.0)]
    speaker_by_id, _ = assign_speakers(segs, dia)
    assert speaker_by_id == {"s": "Speaker 2"}


def test_equal_overlap_lower_index_wins():
    """Transcript [0,10] overlaps Speaker A [0,5] = 5s and Speaker B [5,10] = 5s.
    Tie → lower-indexed (first) diarization segment wins.
    """
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
        Segment(start=5.0, end=10.0, speaker="Speaker 2", raw_speaker="SPEAKER_02"),
    ])
    segs = [_seg("s", 0.0, 10.0)]
    speaker_by_id, _ = assign_speakers(segs, dia)
    assert speaker_by_id == {"s": "Speaker 1"}


def test_zero_overlap_treated_as_no_match():
    """Touching but not overlapping (end == next start) yields zero overlap."""
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
    ])
    segs = [_seg("s", 5.0, 6.0)]  # starts exactly where dia ends
    speaker_by_id, unmatched = assign_speakers(segs, dia)
    # 0-overlap means no positive match → unmatched.
    assert speaker_by_id == {}
    assert unmatched == ["s"]


# ── label-mode switch ────────────────────────────────────────────────────


def test_speaker_labels_ranked_uses_speaker_field():
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 7", raw_speaker="SPEAKER_99"),
    ])
    speaker_by_id, _ = assign_speakers(
        [_seg("s", 1.0, 2.0)], dia, speaker_labels="ranked"
    )
    assert speaker_by_id == {"s": "Speaker 7"}


def test_speaker_labels_raw_uses_raw_speaker_field():
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 7", raw_speaker="SPEAKER_99"),
    ])
    speaker_by_id, _ = assign_speakers(
        [_seg("s", 1.0, 2.0)], dia, speaker_labels="raw"
    )
    assert speaker_by_id == {"s": "SPEAKER_99"}


# ── purity ────────────────────────────────────────────────────────────────


def test_input_segments_are_not_mutated():
    """The function must not touch its inputs."""
    dia = _make_diarization([
        Segment(start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
    ])
    seg = _seg("s", 1.0, 2.0)
    original_dump = seg.model_dump()
    assign_speakers([seg], dia, speaker_labels="ranked")
    assert seg.model_dump() == original_dump


def test_input_diarization_is_not_mutated():
    dia_seg = Segment(
        start=0.0, end=5.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"
    )
    dia = _make_diarization([dia_seg])
    original_dia_dump = dia.model_dump()
    assign_speakers([_seg("s", 1.0, 2.0)], dia, speaker_labels="ranked")
    assert dia.model_dump() == original_dia_dump
