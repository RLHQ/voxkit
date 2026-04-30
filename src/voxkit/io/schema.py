"""Pydantic schema：voxkit diarize 的稳定输出契约。

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
    """voxkit diarize 的最终 stdout JSON。"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Transcribe — voxkit-native rich schema (transcript.voxkit.json)
# ─────────────────────────────────────────────────────────────────────────────


class Word(BaseModel):
    """单词级时间戳（英文 word 模式才会有；CJK phrase 模式 words 为空）。"""

    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    """voxkit-native segment。是 Remixr `RemixrSegment` 的超集。

    与 Remixr 的差异：
      - 无 ``speaker``：voxkit 自身不做 diarization，pre-diarization 阶段
        所有段统一占位，由 ``remixr_adapter`` 在导出时填 ``"Speaker A"``。
      - 多了 ``no_speech_prob`` / ``avg_confidence``：审计字段，camelCase 别名。
      - 无 ``rawText`` / ``subtitles``：那两个字段是 Remixr proofread 阶段产物。
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    start: float
    end: float
    text: str
    words: List[Word] = Field(default_factory=list)
    no_speech_prob: Optional[float] = Field(None, alias="noSpeechProb")
    avg_confidence: Optional[float] = Field(None, alias="avgConfidence")


class ChunkStat(BaseModel):
    """单个 chunk 的处理统计；写到 ``perChunk`` 数组里。"""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    start_secs: float = Field(..., alias="startSecs")
    duration_secs: float = Field(..., alias="durationSecs")
    elapsed_secs: float = Field(..., alias="elapsedSecs")
    rtf: float
    cached: bool = False  # True if loaded from chunk_NNN.json checkpoint


class TranscriptionOutput(BaseModel):
    """Voxkit-native rich transcript（写到 ``transcript.voxkit.json``）。

    这是 voxkit 自己审计用的丰富格式：含 RTF / elapsed / perChunk / 幻觉丢弃数 /
    warnings。Remixr-compatible 输出由 ``remixr_adapter.to_remixr_transcript``
    从这个对象映射出去。
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    audio: AudioInfo
    asr_backend: str = Field(..., alias="asrBackend")  # "whisper-cpp"
    asr_model: str = Field(..., alias="asrModel")
    language: str
    word_timestamps: bool = Field(..., alias="wordTimestamps")
    rtf: float
    elapsed_secs: float = Field(..., alias="elapsedSecs")
    per_chunk: List[ChunkStat] = Field(default_factory=list, alias="perChunk")
    hallucination_drops: int = Field(0, alias="hallucinationDrops")
    segments: List[TranscriptSegment]
    warnings: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Remixr — Zod-mirroring schema (transcript.raw.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# These class field names ARE the on-disk JSON keys — no aliases. Source of
# truth for the contract is:
#   /Users/xsharp/Workspace/3Craft/CutFlow/packages/shared/src/types/transcript.ts
# Do not add fields here without updating the Zod schema.


class RemixrWord(BaseModel):
    """Remixr `WordSchema`：扁平的 word 时间戳。"""

    word: str
    start: float
    end: float


class RemixrSegment(BaseModel):
    """Remixr `SegmentSchema`。

    设计要点：
      - ``speaker`` 默认 ``"Speaker A"``：pre-diarization 占位符。
      - ``subtitles`` 默认 ``[]``：Remixr 的 proofread agent 后续填充。
      - 故意 **不** 暴露 ``rawText``：那是 Remixr proofread 阶段才设置的字段，
        voxkit 必须不写，否则会污染 Remixr 的「未校对」判定。
    """

    id: str
    speaker: str = "Speaker A"
    start: float
    end: float
    text: str
    subtitles: List[str] = Field(default_factory=list)
    words: List[RemixrWord] = Field(default_factory=list)


class RemixrTranscript(BaseModel):
    """Remixr `TranscriptSchema`，对应 ``transcript.raw.json``。

    Remixr 的 Zod schema 不限制额外 key，所以 ``_metadata`` 在序列化层叠加，
    不在这里建模。
    """

    sourceId: str
    segments: List[RemixrSegment]


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle cues — render-layer artifact (subtitles.cues.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# 这是 ``--resegment semantic`` 的机读出口：与 ``subtitles.srt/vtt`` 同源（同一
# 份 ``SubtitleCue[]``），但保留浮点精度 + 显式 speaker 字段。下游消费者（如
# Remixr）想要语义重切的成果时读这份文件，而不是反解 SRT 文本——也不应该把它
# 错当成 ASR ground truth：``transcript.raw.json`` 永远是 ASR 真实产出，本文件
# 是渲染期决策，混用会污染 Remixr 的 proofread 判定。


class SubtitleCueOut(BaseModel):
    """单条字幕 cue 的可序列化形态；与 ``core.semantic_resegment.SubtitleCue``
    同形（start/end/speaker/text），但走 Pydantic 以确保 JSON 输出格式稳定。

    ``speaker`` 用 ``None`` 表示未跑 diarization（区别于占位 ``"Speaker A"``）。
    """

    start: float
    end: float
    speaker: Optional[str] = None
    text: str


class SubtitleCuesOutput(BaseModel):
    """``subtitles.cues.json`` 的顶层 schema。

    与 ``RemixrTranscript`` 平行的渲染层契约：

      - ``schemaVersion``：独立计数器，初始 ``"1"``；下游消费者据此判断兼容性
      - ``sourceId``：与 ``transcript.raw.json`` 中的 sourceId 一致，便于关联
      - ``resegment``：产出 cues 的策略（``"semantic"`` / ``"none"`` 等），方便
        审计 cue 是怎么来的；当前只 ``"semantic"`` 模式真的会写这个文件
      - ``params``：重切参数快照，可复现；缺省字段不写入 JSON
      - ``cues``：扁平的 ``SubtitleCueOut[]``
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    source_id: str = Field(..., alias="sourceId")
    resegment: str
    params: Optional[dict] = None
    cues: List[SubtitleCueOut]


__all__ = [
    "AudioInfo",
    "SpeakerInfo",
    "Segment",
    "DiarizationOutput",
    # transcribe — voxkit-native
    "Word",
    "TranscriptSegment",
    "ChunkStat",
    "TranscriptionOutput",
    # transcribe — Remixr-compatible
    "RemixrWord",
    "RemixrSegment",
    "RemixrTranscript",
    # subtitle cues — render-layer artifact
    "SubtitleCueOut",
    "SubtitleCuesOutput",
]
