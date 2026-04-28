"""align 子命令测试：用合成的 transcript + diarization 验证 SRT 输出。"""

from __future__ import annotations

import json
from pathlib import Path

from voxkit.commands import align as A
from voxkit.io.schema import (
    AudioInfo,
    DiarizationOutput,
    Segment,
    SpeakerInfo,
)


def _make_diarization() -> DiarizationOutput:
    return DiarizationOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=30.0),
        device="cpu",
        model="pyannote/speaker-diarization-3.1",
        rtf=0.5,
        elapsed_secs=15.0,
        num_speakers=2,
        speakers=[
            SpeakerInfo(id="Speaker 1", raw_id="SPEAKER_01", total_duration_secs=20.0),
            SpeakerInfo(id="Speaker 2", raw_id="SPEAKER_00", total_duration_secs=10.0),
        ],
        segments=[
            Segment(start=0.0, end=10.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
            Segment(start=10.0, end=20.0, speaker="Speaker 2", raw_speaker="SPEAKER_00"),
            Segment(start=20.0, end=30.0, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
        ],
    )


def test_format_srt_time():
    assert A._format_srt_time(70) == "00:00:00,070"
    assert A._format_srt_time(13_100) == "00:00:13,100"
    assert A._format_srt_time(3_661_500) == "01:01:01,500"


def test_parse_remixr_format(tmp_path: Path):
    """Remixr transcript.raw.json 格式（segments 数组，秒）。"""
    p = tmp_path / "transcript.raw.json"
    p.write_text(json.dumps({
        "segments": [
            {"start": 0.07, "end": 13.10, "text": "Hello world"},
            {"start": 13.20, "end": 18.50, "text": "Second turn"},
        ]
    }))
    segs = A._parse_transcript(p)
    assert len(segs) == 2
    assert segs[0].start_ms == 70
    assert segs[0].end_ms == 13_100
    assert segs[0].text == "Hello world"


def test_parse_whisper_cpp_format(tmp_path: Path):
    """whisper.cpp --output-json-full 格式（transcription，毫秒）。"""
    p = tmp_path / "whisper.json"
    p.write_text(json.dumps({
        "transcription": [
            {"offsets": {"from": 70, "to": 13100}, "text": " Hello"},
        ]
    }))
    segs = A._parse_transcript(p)
    assert len(segs) == 1
    assert segs[0].start_ms == 70
    assert segs[0].text == "Hello"  # 已 strip


def test_align_produces_srt(tmp_path: Path):
    """端到端：合成 transcript + diarization → 写出 SRT 含 ranked label。"""
    t_path = tmp_path / "transcript.raw.json"
    t_path.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 12.0, "end": 18.0, "text": "I see"},
            {"start": 22.0, "end": 28.0, "text": "Right"},
        ]
    }))
    out = tmp_path / "aligned.srt"
    A.align_to_srt(
        transcript_path=t_path,
        diarization=_make_diarization(),
        out_srt=out,
        speaker_labels="ranked",
    )
    text = out.read_text()
    # 三个 segment 各一条，speaker label 来自重叠最大的 diarization 段
    assert "Speaker 1: Hello" in text   # 0-5 落在 Speaker 1 段（0-10）
    assert "Speaker 2: I see" in text   # 12-18 落在 Speaker 2 段（10-20）
    assert "Speaker 1: Right" in text   # 22-28 落在 Speaker 1 段（20-30）
    # 时间戳格式
    assert "00:00:00,000 --> 00:00:05,000" in text


def test_align_raw_labels(tmp_path: Path):
    """speaker_labels=raw → 输出 SPEAKER_xx 而非 Speaker N。"""
    t_path = tmp_path / "transcript.raw.json"
    t_path.write_text(json.dumps({
        "segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}]
    }))
    out = tmp_path / "aligned.srt"
    A.align_to_srt(
        transcript_path=t_path,
        diarization=_make_diarization(),
        out_srt=out,
        speaker_labels="raw",
    )
    text = out.read_text()
    assert "SPEAKER_01: Hello" in text
    assert "Speaker 1: Hello" not in text


def test_align_no_overlap_marks_unknown(tmp_path: Path):
    """transcript 段落超出 diarization 范围 → 标记 Speaker ?。"""
    t_path = tmp_path / "transcript.raw.json"
    t_path.write_text(json.dumps({
        "segments": [{"start": 100.0, "end": 105.0, "text": "Off the end"}]
    }))
    out = tmp_path / "aligned.srt"
    A.align_to_srt(
        transcript_path=t_path,
        diarization=_make_diarization(),
        out_srt=out,
        speaker_labels="ranked",
    )
    assert "Speaker ?: Off the end" in out.read_text()
