"""Pydantic schema：voxsplit diarize 的稳定输出契约。

字段命名与 Remixr (CutFlow) ASRProvider 接口风格保持一致：
- camelCase
- 时间字段全部用 `xxxSecs`（避免 ms / s 混用）
- schemaVersion 是首要稳定契约，禁止 breaking change 不升版本
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AudioInfo(BaseModel):
    """输入音频元信息。视频输入时记录 extracted_from。"""

    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(..., description="实际跑 diarization 的 wav 绝对路径")
    duration_secs: float = Field(..., alias="durationSecs")
    extracted_from: Optional[str] = Field(
        None,
        alias="extractedFrom",
        description="若输入是视频，记录原始视频路径；纯音频输入则为 None",
    )


class SpeakerInfo(BaseModel):
    """ranked 重映射后的 speaker 元信息。"""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description='ranked 标签，例如 "Speaker 1"')
    raw_id: str = Field(..., alias="rawId", description='pyannote 原始标签 "SPEAKER_01"')
    total_duration_secs: float = Field(..., alias="totalDurationSecs")


class Segment(BaseModel):
    """单个 speaker turn。"""

    model_config = ConfigDict(populate_by_name=True)

    start: float
    end: float
    speaker: str = Field(..., description='ranked 标签 "Speaker N"')
    raw_speaker: str = Field(..., alias="rawSpeaker")


class DiarizationOutput(BaseModel):
    """voxsplit diarize 的最终 stdout JSON。"""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    audio: AudioInfo
    device: str = Field(..., description='"mps" / "cuda" / "cpu"')
    model: str = Field(..., description='例如 "pyannote/speaker-diarization-3.1"')
    rtf: float = Field(..., description="real-time factor = elapsed / duration")
    elapsed_secs: float = Field(..., alias="elapsedSecs")
    num_speakers: int = Field(..., alias="numSpeakers")
    speakers: List[SpeakerInfo]
    segments: List[Segment]
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "AudioInfo",
    "SpeakerInfo",
    "Segment",
    "DiarizationOutput",
]
