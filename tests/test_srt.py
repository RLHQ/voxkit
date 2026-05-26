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
    """v0.7.2 review #1：segment path 默认 'auto' 不再带 Speaker A 前缀。
    传 ``speaker_prefix='always'`` 才恢复旧行为。"""
    doc = to_subtitles_srt(_two_seg_fixture(), speaker_prefix="always")

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


def test_to_subtitles_srt_default_auto_strips_placeholder():
    """v0.7.2 review #1：默认 'auto' 在 segment path（schema 无 speaker）下不加前缀。"""
    doc = to_subtitles_srt(_two_seg_fixture())
    assert "Speaker A:" not in doc
    assert "hello" in doc and "world" in doc


def test_to_subtitles_srt_cue_regex_match():
    """A regex pass over the doc confirms valid cue blocks."""
    doc = to_subtitles_srt(_two_seg_fixture(), speaker_prefix="always")
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
    """Back-compat：``speaker_prefix=False``（bool）等同 'never'。"""
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
    doc = to_subtitles_vtt(_two_seg_fixture(), speaker_prefix="always")
    assert doc.startswith("WEBVTT\n\n")
    # Periods, not commas
    assert "00:00:00.000 --> 00:00:01.500" in doc
    assert "," not in doc.splitlines()[2]  # cue time line uses periods
    # Speaker prefix preserved
    assert "Speaker A: hello" in doc
    assert "Speaker A: world" in doc


def test_to_subtitles_vtt_cue_regex_match():
    doc = to_subtitles_vtt(_two_seg_fixture(), speaker_prefix="always")
    cue_re = re.compile(
        r"^(\d+)\n(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n",
        re.MULTILINE,
    )
    matches = cue_re.findall(doc)
    assert len(matches) == 2


def test_to_subtitles_vtt_no_speaker_prefix():
    doc = to_subtitles_vtt(_two_seg_fixture(), speaker_prefix=False)
    assert "Speaker A:" not in doc


# ---------------------------------------------------------------------------
# Cue path — `to_subtitles_srt_from_cues` speaker_prefix policy (B1 fix)
# ---------------------------------------------------------------------------


class _CueFx:
    """Minimal duck-typed cue: only start/end/speaker/text are read by renderer."""

    def __init__(self, start: float, end: float, text: str, speaker=None):
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker


def test_cues_auto_skips_prefix_for_single_speaker_placeholder():
    """B1: 单人讲座 + 上游塞了 'Speaker A' 占位符时，auto 不加前缀。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [
        _CueFx(0.0, 1.0, "hello", "Speaker A"),
        _CueFx(1.0, 2.0, "world", "Speaker A"),
    ]
    doc = to_subtitles_srt_from_cues(cues)
    assert "Speaker A:" not in doc
    assert "hello" in doc and "world" in doc


def test_cues_auto_keeps_prefix_for_multi_speaker():
    """auto 在 ≥2 个非空 speaker 时仍渲染前缀（保留 diarization 信号）。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [
        _CueFx(0.0, 1.0, "hello", "Speaker 1"),
        _CueFx(1.0, 2.0, "hi", "Speaker 2"),
    ]
    doc = to_subtitles_srt_from_cues(cues)
    assert "Speaker 1: hello" in doc
    assert "Speaker 2: hi" in doc


def test_cues_always_forces_prefix_even_single_speaker():
    """always = 旧行为（v0.7.1 以前），单人也强加 placeholder 前缀。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [_CueFx(0.0, 1.0, "hello", "Speaker A")]
    doc = to_subtitles_srt_from_cues(cues, speaker_prefix="always")
    assert "Speaker A: hello" in doc


def test_cues_never_strips_prefix_even_multi_speaker():
    """never = 强制不加，即使 diarization 有多人。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [
        _CueFx(0.0, 1.0, "hello", "Speaker 1"),
        _CueFx(1.0, 2.0, "hi", "Speaker 2"),
    ]
    doc = to_subtitles_srt_from_cues(cues, speaker_prefix="never")
    assert "Speaker 1:" not in doc and "Speaker 2:" not in doc
    assert "hello" in doc and "hi" in doc


def test_cues_vtt_auto_matches_srt_policy():
    """VTT 与 SRT 共享同一套 speaker_prefix 策略。"""
    from voxkit.io.srt import to_subtitles_vtt_from_cues

    cues = [_CueFx(0.0, 1.0, "hello", "Speaker A")]
    doc = to_subtitles_vtt_from_cues(cues)
    assert "Speaker A:" not in doc
    assert doc.startswith("WEBVTT")


def test_should_show_prefix_helper_handles_none():
    """None speaker 既不算入 distinct 也不被当成"有 speaker 信息"。"""
    from voxkit.io.srt import should_show_speaker_prefix

    assert not should_show_speaker_prefix([None, None, None], "auto")
    assert not should_show_speaker_prefix(["Speaker A", None], "auto")
    assert should_show_speaker_prefix(["Speaker 1", "Speaker 2"], "auto")
    # explicit overrides bypass distinct check
    assert should_show_speaker_prefix([None], "always")
    assert not should_show_speaker_prefix(["A", "B"], "never")


def test_should_show_prefix_filters_placeholders():
    """v0.7.2 review #5：'Speaker A' / 'Speaker ?' 不计入 distinct count。"""
    from voxkit.io.srt import should_show_speaker_prefix

    # 全占位符 → 0 informative → no prefix
    assert not should_show_speaker_prefix(["Speaker A", "Speaker A"], "auto")
    assert not should_show_speaker_prefix(["Speaker ?", "Speaker ?"], "auto")
    # 1 real + 1 unmatched placeholder → 1 informative → no prefix (单人 + 漏标)
    assert not should_show_speaker_prefix(["Speaker 1", "Speaker ?"], "auto")
    # 2 real + 1 unmatched → 2 informative → prefix on
    assert should_show_speaker_prefix(
        ["Speaker 1", "Speaker 2", "Speaker ?"], "auto"
    )


def test_cues_auto_skips_placeholder_cue_in_multi_speaker_mix():
    """v0.7.2 review #5：多 speaker 模式下未匹配 'Speaker ?' cue 不写 'Speaker ?: ...'。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [
        _CueFx(0.0, 1.0, "hello", "Speaker 1"),
        _CueFx(1.0, 1.5, "[cough]", "Speaker ?"),
        _CueFx(1.5, 2.5, "hi", "Speaker 2"),
    ]
    doc = to_subtitles_srt_from_cues(cues)
    assert "Speaker 1: hello" in doc
    assert "Speaker 2: hi" in doc
    # 关键：未匹配 cue 不应渲染 "Speaker ?:" 前缀
    assert "Speaker ?:" not in doc
    assert "[cough]" in doc


def test_cues_always_keeps_placeholder_prefix_for_compat():
    """always 模式回退到旧行为：占位符 cue 也加前缀（即使有 'Speaker ?'）。"""
    from voxkit.io.srt import to_subtitles_srt_from_cues

    cues = [
        _CueFx(0.0, 1.0, "hello", "Speaker 1"),
        _CueFx(1.0, 1.5, "[cough]", "Speaker ?"),
    ]
    doc = to_subtitles_srt_from_cues(cues, speaker_prefix="always")
    assert "Speaker 1: hello" in doc
    assert "Speaker ?: [cough]" in doc


def test_is_informative_speaker_helper():
    """``is_informative_speaker`` 暴露给 unit tests + 文档示例。"""
    from voxkit.io.srt import is_informative_speaker

    assert is_informative_speaker("Speaker 1")
    assert is_informative_speaker("Alice")
    assert not is_informative_speaker(None)
    assert not is_informative_speaker("")
    assert not is_informative_speaker("Speaker A")
    assert not is_informative_speaker("Speaker ?")
