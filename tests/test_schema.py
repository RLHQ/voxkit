"""schema 稳定契约测试：camelCase 别名 + round-trip + 关键字段存在。"""

from __future__ import annotations

import json

from voxkit.io.schema import (
    AudioInfo,
    DiarizationOutput,
    Segment,
    SpeakerInfo,
)


def _make_sample() -> DiarizationOutput:
    return DiarizationOutput(
        schema_version="1",
        audio=AudioInfo(
            path="/tmp/a.wav",
            duration_secs=3821.6,
            extracted_from="/tmp/a.mp4",
        ),
        device="mps",
        model="pyannote/speaker-diarization-3.1",
        rtf=0.0772,
        elapsed_secs=295.0,
        num_speakers=2,
        speakers=[
            SpeakerInfo(id="Speaker 1", raw_id="SPEAKER_01", total_duration_secs=1450.3),
            SpeakerInfo(id="Speaker 2", raw_id="SPEAKER_00", total_duration_secs=980.5),
        ],
        segments=[
            Segment(start=0.04, end=8.32, speaker="Speaker 1", raw_speaker="SPEAKER_01"),
            Segment(start=8.4, end=10.0, speaker="Speaker 2", raw_speaker="SPEAKER_00"),
        ],
        warnings=[],
    )


def test_round_trip_camel_case():
    """dump → JSON → 回读保持 camelCase 字段名。"""
    sample = _make_sample()
    dumped = sample.model_dump_json(by_alias=True)
    payload = json.loads(dumped)
    # 顶层 camelCase
    assert payload["schemaVersion"] == "1"
    assert payload["elapsedSecs"] == 295.0
    assert payload["numSpeakers"] == 2
    # 嵌套 camelCase
    assert payload["audio"]["durationSecs"] == 3821.6
    assert payload["audio"]["extractedFrom"] == "/tmp/a.mp4"
    assert payload["speakers"][0]["rawId"] == "SPEAKER_01"
    assert payload["speakers"][0]["totalDurationSecs"] == 1450.3
    assert payload["segments"][0]["rawSpeaker"] == "SPEAKER_01"
    # warnings 即使为空也存在
    assert payload["warnings"] == []


def test_validate_back():
    """JSON 回读能 validate 成原模型。"""
    sample = _make_sample()
    dumped = sample.model_dump_json(by_alias=True)
    parsed = DiarizationOutput.model_validate_json(dumped)
    assert parsed.audio.duration_secs == sample.audio.duration_secs
    assert parsed.segments[0].speaker == "Speaker 1"


def test_extracted_from_optional():
    """纯音频输入 extractedFrom=None 也合法。"""
    audio = AudioInfo(path="/tmp/a.wav", duration_secs=10.0)
    assert audio.extracted_from is None
    payload = json.loads(audio.model_dump_json(by_alias=True))
    assert payload["extractedFrom"] is None


def test_warnings_default_empty_list():
    """warnings 字段缺省时是空 list。"""
    out = DiarizationOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=10.0),
        device="cpu",
        model="m",
        rtf=0.5,
        elapsed_secs=5.0,
        num_speakers=0,
        speakers=[],
        segments=[],
    )
    assert out.warnings == []
    assert out.schema_version == "1"
