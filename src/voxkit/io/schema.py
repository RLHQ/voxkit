"""Pydantic schema：voxkit diarize 的稳定输出契约。

字段命名与 Remixr (CutFlow) ASRProvider 接口风格保持一致：
- camelCase
- 时间字段全部用 `xxxSecs`（避免 ms / s 混用）
- schemaVersion 是首要稳定契约，禁止 breaking change 不升版本
"""

from __future__ import annotations

from typing import List, Literal, Optional

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

    ``id``：稳定 cue id（``"cue_NNNNNN"`` 6 位零填充顺序号），由 ``cues_json``
    在序列化时按 1-based enumerate 顺序赋值。下游 proofread/translate 产物用
    ``cueId`` 反引这里的 ``id``，**不要靠时间戳浮点匹配**。

    ``speaker`` 用 ``None`` 表示未跑 diarization（区别于占位 ``"Speaker A"``）。
    """

    id: str = Field(..., description='稳定 cue id，例如 "cue_000001"')
    start: float
    end: float
    speaker: Optional[str] = None
    text: str


class SubtitleCuesOutput(BaseModel):
    """``subtitles.cues.json`` 的顶层 schema。

    与 ``RemixrTranscript`` 平行的渲染层契约：

      - ``schemaVersion``：独立计数器，**当前 "2"**（v2 引入了 ``cues[].id``）。
        本项目内部使用，不写 v1 → v2 兼容读，旧 workdir 重新生成即可
      - ``sourceId``：与 ``transcript.raw.json`` 中的 sourceId 一致，便于关联
      - ``resegment``：产出 cues 的策略（``"semantic"`` / ``"none"`` 等），方便
        审计 cue 是怎么来的；当前只 ``"semantic"`` 模式真的会写这个文件
      - ``params``：重切参数快照，可复现；缺省字段不写入 JSON
      - ``metrics``：字幕质量统计；缺省字段不写入 JSON
      - ``cues``：扁平的 ``SubtitleCueOut[]``，每条带稳定 ``id``
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("2", alias="schemaVersion")
    source_id: str = Field(..., alias="sourceId")
    resegment: str
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    cues: List[SubtitleCueOut]


# ─────────────────────────────────────────────────────────────────────────────
# Proofread — LLM text-enhancement artifact (subtitles.proofread.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# 这是 LLM 校对阶段的产物：以 ``subtitles.cues.json`` (schemaVersion=2) 为输入，
# 按 ``cueId`` 反引源 cue，**只改文本，不改时间戳/speaker**。生命周期 state 落
# 在 artifact 顶层 (``draft`` / ``reviewed`` / ``final``)，manifest 镜像一份用于
# 快速索引。``inputHash`` 用于推导 stale：上游 cues.json 重建后 hash 不一致即过期。
#
# 风险等级（``risk``）：
#   - ``low``：普通修改，自动接受
#   - ``medium``：可能影响理解，UI 标记
#   - ``high``：可能引入事实错误，默认人工复核
#   - ``blocking``：违反 schema/不变量，不应写稳定产物
#
# 编辑强度（``edit_level``）：``none`` / ``minor`` / ``major``。


RiskLevel = Literal["low", "medium", "high", "blocking"]
EditLevel = Literal["none", "minor", "major"]
ArtifactState = Literal["draft", "reviewed", "final"]


class ProofreadCueOut(BaseModel):
    """单条校对后的 cue。``cueId`` 必须对应 ``subtitles.cues.json`` 中的 id。

    时间字段 ``sourceStart`` / ``sourceEnd`` 是源 cue 时间的副本，写入这里只为
    artifact 自洽（不依赖上游也能展示）；**LLM 不得修改**。
    """

    model_config = ConfigDict(populate_by_name=True)

    cue_id: str = Field(..., alias="cueId")
    source_start: float = Field(..., alias="sourceStart")
    source_end: float = Field(..., alias="sourceEnd")
    speaker: Optional[str] = None
    source_text: str = Field(..., alias="sourceText")
    corrected_text: str = Field(..., alias="correctedText")
    edit_level: EditLevel = Field(..., alias="editLevel")
    risk: RiskLevel = "low"
    needs_human_review: bool = Field(False, alias="needsHumanReview")
    notes: List[str] = Field(default_factory=list)


class ProofreadParams(BaseModel):
    """proofread 阶段输入参数快照（写入 artifact + manifest 用于复现）。"""

    model_config = ConfigDict(populate_by_name=True)

    edit_level: str = Field(..., alias="editLevel")  # punctuation/light/standard/strict
    allow_retiming: bool = Field(False, alias="allowRetiming")
    glossary_hash: Optional[str] = Field(None, alias="glossaryHash")


class ProofreadMetrics(BaseModel):
    """proofread 完成后的聚合指标。"""

    model_config = ConfigDict(populate_by_name=True)

    cue_count: int = Field(..., alias="cueCount")
    changed_cue_rate: float = Field(..., alias="changedCueRate")
    review_cue_rate: float = Field(..., alias="reviewCueRate")
    prompt_tokens_total: int = Field(0, alias="promptTokensTotal")
    completion_tokens_total: int = Field(0, alias="completionTokensTotal")


class ProofreadOutput(BaseModel):
    """``subtitles.proofread.json`` 顶层 schema。

    生命周期：
      - ``state="draft"``：机器生成，未经人工确认（``voxkit proofread`` 出口）
      - ``state="reviewed"``：人工/规则确认（``voxkit review confirm``）
      - ``state="final"``：锁定发布（``voxkit review lock``）

    ``inputHash`` 是上游 ``subtitles.cues.json`` 的 sha256；下游/UI 可用它推导
    "是否 stale"（``cues.json`` 重建后 hash 不一致即过期）。
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    state: ArtifactState = "draft"
    source_id: str = Field(..., alias="sourceId")
    input_artifact: str = Field(..., alias="inputArtifact")
    input_hash: str = Field(..., alias="inputHash")
    language: str
    provider: str
    model: str
    prompt_version: str = Field(..., alias="promptVersion")
    prompt_hash: str = Field(..., alias="promptHash")
    params: ProofreadParams
    cues: List[ProofreadCueOut]
    metrics: ProofreadMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Translation — LLM cross-language artifact (subtitles.<lang>.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# v1 限制：``cueMappingPolicy = "one-to-one"``。目标 cue 与源 cue 一对一，时间
# 范围直接继承源 cue。group-within-speaker rewrap 留到后续版本。
#
# 输入选择：默认 ``subtitles.proofread.json``（state ∈ {draft, reviewed, final}
# 都可接受），缺失则回落 ``subtitles.cues.json``。``inputArtifact`` 字段记录
# 实际使用的输入。
#
# ``id`` 形如 ``trg_000001``：与源 cue id 区分（避免误以为它们等价）。


class TranslationCueOut(BaseModel):
    """单条目标语言 cue。"""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description='目标 cue id，例如 "trg_000001"')
    source_cue_ids: List[str] = Field(..., alias="sourceCueIds")
    start: float
    end: float
    speaker: Optional[str] = None
    text: str  # 目标语言文本
    mapping: Literal["one-to-one", "merged", "split"] = "one-to-one"
    risk: RiskLevel = "low"
    needs_human_review: bool = Field(False, alias="needsHumanReview")
    notes: List[str] = Field(default_factory=list)


class TranslationParams(BaseModel):
    """translate 阶段输入参数快照。"""

    model_config = ConfigDict(populate_by_name=True)

    style: str  # literal / natural / subtitle / technical
    length_policy: str = Field("preserve", alias="lengthPolicy")
    cue_mapping_policy: str = Field("one-to-one", alias="cueMappingPolicy")
    glossary_hash: Optional[str] = Field(None, alias="glossaryHash")


class TranslationMetrics(BaseModel):
    """translation 完成后的聚合指标。"""

    model_config = ConfigDict(populate_by_name=True)

    cue_count: int = Field(..., alias="cueCount")
    over_char_limit_rate: float = Field(0.0, alias="overCharLimitRate")
    over_cps_rate: float = Field(0.0, alias="overCpsRate")
    glossary_miss_rate: float = Field(0.0, alias="glossaryMissRate")
    prompt_tokens_total: int = Field(0, alias="promptTokensTotal")
    completion_tokens_total: int = Field(0, alias="completionTokensTotal")


class TranslationOutput(BaseModel):
    """``subtitles.<lang>.json`` 顶层 schema。

    与 ``ProofreadOutput`` 平行，但 ``sourceLanguage`` / ``targetLanguage`` 分离，
    ``cues[]`` 用 ``sourceCueIds`` 反引源 cue（v1 始终 1 个，未来 merged/split
    时可能多个）。
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    state: ArtifactState = "draft"
    source_id: str = Field(..., alias="sourceId")
    input_artifact: str = Field(..., alias="inputArtifact")
    input_hash: str = Field(..., alias="inputHash")
    source_language: str = Field(..., alias="sourceLanguage")
    target_language: str = Field(..., alias="targetLanguage")
    provider: str
    model: str
    prompt_version: str = Field(..., alias="promptVersion")
    prompt_hash: str = Field(..., alias="promptHash")
    params: TranslationParams
    cues: List[TranslationCueOut]
    metrics: TranslationMetrics


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
    # proofread — LLM text-enhancement artifact
    "RiskLevel",
    "EditLevel",
    "ArtifactState",
    "ProofreadCueOut",
    "ProofreadParams",
    "ProofreadMetrics",
    "ProofreadOutput",
    # translation — LLM cross-language artifact
    "TranslationCueOut",
    "TranslationParams",
    "TranslationMetrics",
    "TranslationOutput",
]
