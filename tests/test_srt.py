"""Unit tests for ``voxkit.io.srt`` — SRT and WebVTT generators.

Covers:
  - Basic SRT / VTT structure (cue numbers, time format, speaker prefix).
  - Speaker-prefix opt-out.
  - Edge: 1-hour boundary (HH digit cascade).
  - Edge: sub-millisecond rounding does not produce malformed timestamps.
  - format_srt_time / format_vtt_time helpers as standalone units.
"""

from __future__ import annotations

import re

from voxkit.io.schema import AudioInfo, TranscriptionOutput, TranscriptSegment
from voxkit.io.srt import (
    format_srt_time,
    format_vtt_time,
    to_subtitles_srt,
    to_subtitles_vtt,
)


def _make_output(segments: list[TranscriptSegment]) -> TranscriptionOutput:
    return TranscriptionOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=10.0),
        asr_backend="whisper-cpp",
        asr_model="ggml-large-v3-turbo",
        language="en",
        word_timestamps=True,
        rtf=0.05,
        elapsed_secs=10.0,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Time formatters
# ---------------------------------------------------------------------------


def test_format_srt_time_basic():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(1.5) == "00:00:01,500"
    assert format_srt_time(61.123) == "00:01:01,123"
    assert format_srt_time(3661.001) == "01:01:01,001"


def test_format_srt_time_one_hour_boundary():
    assert format_srt_time(3600.0) == "01:00:00,000"


def test_format_srt_time_subms_rounding_no_overflow():
    # 59.9999s rounds to 60_000 ms total → must cascade cleanly to 1 minute,
    # not produce ``00:00:59,1000`` or similar malformed output.
    out = format_srt_time(59.9999)
    assert out == "00:01:00,000"


def test_format_srt_time_clamps_negative():
    assert format_srt_time(-0.5) == "00:00:00,000"


def test_format_vtt_time_uses_period():
    assert format_vtt_time(0.0) == "00:00:00.000"
    assert format_vtt_time(1.5) == "00:00:01.500"
    assert format_vtt_time(3600.0) == "01:00:00.000"


# ---------------------------------------------------------------------------
# Document generators
# ---------------------------------------------------------------------------


def _two_seg_fixture() -> TranscriptionOutput:
    return _make_output(
        [
            TranscriptSegment(id="seg_001", start=0.0, end=1.5, text="hello"),
            TranscriptSegment(id="seg_002", start=2.5, end=4.05, text="world"),
        ]
    )


def test_to_subtitles_srt_basic_structure():
    doc = to_subtitles_srt(_two_seg_fixture())

    # Trailing newline
    assert doc.endswith("\n")

    lines = doc.splitlines()
    # Cue 1
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert lines[2] == "Speaker A: hello"
    assert lines[3] == ""
    # Cue 2
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,500 --> 00:00:04,050"
    assert lines[6] == "Speaker A: world"
    assert lines[7] == ""


def test_to_subtitles_srt_cue_regex_match():
    """A regex pass over the doc confirms valid cue blocks."""
    doc = to_subtitles_srt(_two_seg_fixture())
    cue_re = re.compile(
        r"^(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.+)$",
        re.MULTILINE,
    )
    matches = cue_re.findall(doc)
    assert len(matches) == 2
    assert matches[0][0] == "1"
    assert matches[1][0] == "2"


def test_to_subtitles_srt_no_speaker_prefix():
    doc = to_subtitles_srt(_two_seg_fixture(), speaker_prefix=False)
    assert "Speaker A:" not in doc
    # The text body should still be present
    assert "hello" in doc
    assert "world" in doc


def test_to_subtitles_srt_one_hour_boundary():
    out = _make_output(
        [TranscriptSegment(id="seg_001", start=0.0, end=3600.0, text="long")]
    )
    doc = to_subtitles_srt(out)
    assert "00:00:00,000 --> 01:00:00,000" in doc


def test_to_subtitles_srt_empty_segments():
    out = _make_output([])
    assert to_subtitles_srt(out) == ""


def test_to_subtitles_vtt_header_and_periods():
    doc = to_subtitles_vtt(_two_seg_fixture())
    assert doc.startswith("WEBVTT\n\n")
    # Periods, not commas
    assert "00:00:00.000 --> 00:00:01.500" in doc
    assert "," not in doc.splitlines()[2]  # cue time line uses periods
    # Speaker prefix preserved
    assert "Speaker A: hello" in doc
    assert "Speaker A: world" in doc


def test_to_subtitles_vtt_cue_regex_match():
    doc = to_subtitles_vtt(_two_seg_fixture())
    cue_re = re.compile(
        r"^(\d+)\n(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n",
        re.MULTILINE,
    )
    matches = cue_re.findall(doc)
    assert len(matches) == 2


def test_to_subtitles_vtt_no_speaker_prefix():
    doc = to_subtitles_vtt(_two_seg_fixture(), speaker_prefix=False)
    assert "Speaker A:" not in doc
