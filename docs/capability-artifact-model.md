# Voxkit 能力与产物模型

本文从产品与工程边界出发，梳理 voxkit 当前和未来可扩展的原子能力、参数、产物，以及这些能力之间的关系。目标是在加入 LLM 校对、翻译、更多 ASR provider 或在线处理能力之前，先把系统的语义边界固定下来。

核心原则：

- **原始事实不可变**：ASR 的原始识别结果和时间戳是 ground truth，后续字幕切分、校对、翻译都不能反向覆盖它。
- **时间轴优先由确定性算法维护**：LLM 可以建议文本修正、术语归一、翻译和少量重排，但不直接掌控 `start` / `end`。
- **能力可组合，产物可审计**：每个阶段都应该有明确输入、输出、参数快照、事件和指标。
- **语言策略分流**：英文等有 word-level timestamp 的语言走 word-aware 路径；中文、日文、韩文等 CJK 走 phrase/char-aware 路径。
- **面向产品的产物不反解展示格式**：下游系统应读取 JSON 产物，不应从 SRT/VTT 反解结构化信息。

## 建模维度

最初可以从四个问题切入：有哪些原子能力、有哪些参数、每个能力生成什么产物、产物之间是什么关系。要让它支撑长期产品化，还需要补充几个横切维度：

| 维度 | 要回答的问题 | 为什么重要 |
|---|---|---|
| 原子能力 | 系统能独立执行哪些动作？ | 决定 CLI/API 边界、测试边界和重跑粒度。 |
| 控制参数 | 哪些行为可配置？默认值是什么？ | 决定可复现性、A/B 实验和产品 preset。 |
| 阶段产物 | 每个能力写出什么稳定文件或事件？ | 决定下游读取契约和缓存边界。 |
| 依赖关系 | 哪些能力依赖哪些产物？ | 决定 pipeline DAG、失败恢复和 stale 判断。 |
| 系统不变量 | 哪些事情永远不能发生？ | 防止 LLM 校对、翻译和字幕渲染污染 raw transcript。 |
| 产物生命周期 | draft、reviewed、final、stale 如何流转？ | 支撑人工审核、重复运行和产品状态展示。 |
| 质量指标 | 怎么判断一个产物好不好？ | 支撑自动复核、模型升级和策略选择。 |
| 产品工作流 | 用户真正想完成哪些任务？ | 把底层能力组合成可理解的默认模式。 |
| 成本与可复现性 | 怎么控制 LLM 成本、缓存和审计？ | 让批量处理、团队协作和回归排查可控。 |

## 术语

| 术语 | 含义 |
|---|---|
| 原子能力 | 可以独立运行、复用、测试和记录参数的处理步骤，例如 ASR、diarization、semantic resegment、proofread。 |
| 产物 | 某个能力写出的稳定文件或流，例如 `transcript.raw.json`、`subtitles.cues.json`、`events.ndjson`。 |
| 原始事实 | 来自媒体和 ASR/diarization 模型的一手输出，允许过滤和合并，但不允许被 LLM 美化后覆盖。 |
| 渲染层产物 | 面向播放器或字幕 UI 的产物，例如 cue 流、SRT、VTT。它们是展示决策，不等同于原始 transcript。 |
| 文本增强产物 | 校对、术语归一、翻译等 LLM 或人工处理后的文本层结果。 |
| 时间 authority | 对某段文本 `start` / `end` 的来源。英文通常是 word timestamp；CJK 可能是 phrase timestamp 或字符插值。 |
| draft | 机器生成但尚未人工确认的产物状态。 |
| reviewed | 人工或高置信规则确认过的产物状态。 |
| final | 被产品/项目锁定用于发布或下游消费的产物状态。 |
| stale | 上游输入或参数变化后，已经不再与当前 pipeline 一致的产物状态。 |
| preset | 面向产品场景的一组默认能力组合和参数，例如快速转录、双语字幕。 |

## 能力地图

### 现有能力

| 能力 | 输入 | 主要参数 | 产物 | 备注 |
|---|---|---|---|---|
| 媒体探测 | 音频/视频文件 | input path | duration、媒体类型 | 通过 ffprobe 获取时长。 |
| 音频准备 | 音频/视频文件 | 采样率 16kHz、mono | `work/input.16khz.mono.wav` 或原音频引用 | 视频抽音，音频直用。 |
| chunk 规划 | 音频时长 | `chunk_threshold_secs`、`chunk_secs`、`chunk_overlap_secs` | `ChunkPlan` | 当前为固定时间网格。 |
| chunk 抽取 | `ChunkPlan` + 音频 | chunk start/duration | `work/chunks/chunk_NNN.wav` | 长音频 checkpoint 基础。 |
| ASR 转录 | chunk wav | `model`、`language`、`word_timestamps`、`vad`、`logprob_thold`、timeout | `work/chunks/chunk_NNN.json` | whisper.cpp 后端；CJK 自动不依赖 word timestamps。 |
| 幻觉过滤 | whisper entries | blocklist、no-speech/logprob 阈值 | filtered entries、`hallucinations.log` | 过滤静音水印、ghost loop 等。 |
| ASR segment 重组 | filtered entries | 语言模式、句末标点、gap、max duration/chars | `TranscriptSegment[]` | 英文 word 聚合；CJK phrase 1:1 映射。 |
| 多 chunk 合并 | chunk segments | overlap tolerance | merged `TranscriptSegment[]`、`work/merge.json` | signal-aware overlap arbitration。 |
| 说话人切分 | 音频 | `diarize_model`、`num/min/max_speakers`、device、speaker label style | `work/diarization.json` | pyannote 后端。 |
| speaker 对齐 | merged segments + diarization | overlap / nearest fallback、label style | 带 speaker 的 Remixr segments | 写入 `transcript.raw.json`。 |
| 语义字幕重切 | Remixr segments | `resegment`、`ResegmentParams` | `subtitles.cues.json` | 只影响渲染层。英文 word-aware；CJK phrase-aware。 |
| 字幕渲染 | segments 或 cues | `emit_srt`、`emit_vtt` | `subtitles.srt`、`subtitles.vtt` | 展示格式，不作为结构化 source of truth。 |
| 字幕质量统计 | cues | `ResegmentParams` | manifest subtitle metrics、`subtitles.cues.json.metrics` | cue count、时长、闪现率、CPS 等。 |
| 事件流 | pipeline phases | `json_events` | `events.ndjson`、stderr NDJSON | 用于实时 UI、进度条、审计。 |
| manifest 汇总 | 全部阶段 | source id、参数快照 | `manifest.json` | 产物索引、耗时、warnings、metrics。 |

### 拟新增能力

| 能力 | 输入 | 主要参数 | 产物 | 备注 |
|---|---|---|---|---|
| LLM 校对 | `subtitles.cues.json` 或 `transcript.raw.json` | provider/model、language、domain glossary、edit policy、batch size、risk threshold | `subtitles.proofread.json`、diff、风险标记 | 默认只改文本，不改时间轴。 |
| 术语归一 | raw/proofread text + glossary | glossary、case policy、protected terms | 可并入 proofread 产物或单独 `terms.applied.json` | 适合产品名、人名、技术词。 |
| LLM 翻译 | proofread cues 或 raw cues | source/target language、style、glossary、length policy、subtitle-fit policy | `subtitles.<lang>.json`、`subtitles.<lang>.srt/vtt` | 翻译可以在同一时间范围内二次排版。 |
| 人工校对回写 | proofread/translation draft + 人工编辑 | reviewer、review policy | `subtitles.reviewed.json` | 产品 UI 后续可接入。 |
| 在线/实时转录 | 音频流 | window size、partial/final policy | partial events、final segments | 实时数据应先进入 event stream，再沉淀为稳定产物。 |
| 多 provider ASR | audio/chunk | provider、model、timestamps mode、cost policy | provider-specific raw、normalized transcript | 需要统一 normalize schema。 |
| 质量评估 | raw/proofread/translation | metrics profile、sampling policy | `quality.report.json` | 对比不同策略和模型。 |

## 参数面

参数需要分层管理，避免一个 CLI 或 API 请求变成无结构的大杂烩。

### 1. 任务与工作区参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| 输入路径 | `input` | 原始媒体文件。 |
| 工作目录 | `workdir` | 所有产物落盘目录。 |
| source 标识 | `source_id` | 下游系统关联键。 |
| resume/force | `resume`、`force` | 控制 checkpoint 复用和幂等。 |
| 保留中间文件 | `keep_work` | 失败调试和审计。 |

### 2. 音频与 chunk 参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| chunk 触发阈值 | `chunk_threshold_secs` | 短音频不分块。 |
| chunk 目标时长 | `chunk_secs` | 控制 ASR 单次处理长度。 |
| chunk overlap | `chunk_overlap_secs` | 接缝去重和防截断。 |
| chunk 策略 | `chunk_strategy` | 未来可扩展为 `fixed-grid` / `vad-aligned`。 |
| 边界搜索窗口 | `boundary_search_secs` | VAD 对齐 chunk 边界时使用。 |

### 3. ASR 参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| ASR provider | `asr_provider` | 当前隐含为 `whisper-cpp`，未来可扩展。 |
| 模型 | `model` / `asr_model` | whisper.cpp 模型别名或 provider 模型名。 |
| 语言 | `language` | `auto` 或具体语言代码。 |
| 词级时间戳 | `word_timestamps` | 英文等非 CJK 重切的重要输入；CJK 自动忽略。 |
| VAD | `vad`、`vad_model` | 降低静音幻觉。 |
| 置信阈值 | `logprob_thold` | 过滤低置信 ASR。 |
| 超时 | `timeout_ms` | 控制单 chunk 执行时长。 |
| whisper 路径 | `whisper_bin` | 本地后端发现覆盖。 |

### 4. 清洗与合并参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| 幻觉黑名单 | `blocklist` | 过滤水印、社区字幕、静音循环文本。 |
| overlap 容忍 | `overlap_tolerance_secs` | chunk merge 决策。 |
| 合并策略 | `merge_strategy` | 当前为 signal-aware；未来可暴露用于 A/B。 |

### 5. 说话人参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| 是否启用 | `with_diarization` | ASR 后追加 speaker labels。 |
| 模型 | `diarize_model` | `sd-3.1` / `community-1`。 |
| speaker 数 | `num_speakers`、`min_speakers`、`max_speakers` | pyannote 聚类提示。 |
| label 策略 | `speaker_labels` | `ranked` / `raw`。 |
| 对齐策略 | `speaker_align_strategy` | 当前基于最大 overlap；可扩展 nearest fallback。 |

### 6. 字幕重切参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| 重切策略 | `resegment` | `none` / `semantic`。 |
| 最大时长 | `max_dur_s` | 单 cue 时长上限。 |
| 最小时长 | `min_dur_s` | 避免闪现字幕。 |
| 最大字符数 | `max_chars` | 两行字幕物理约束。 |
| 软字符上限 | `soft_max_chars` | 优先切分/flush 阈值。 |
| CPS 上限 | `max_cps` | 阅读速度约束。 |
| 韵律 gap | `prosody_gap_s` | word-aware 切分软边界。 |
| CJK timebase | `timebase` | `phrase` / `char-interpolated`。 |

### 7. 校对参数

| 参数 | 建议名称 | 作用 |
|---|---|---|
| 是否启用 | `proofread` | 启用 LLM 校对。 |
| provider/model | `proofread_provider`、`proofread_model` | 控制 LLM 后端和成本质量。 |
| 输入层 | `proofread_input` | `raw-segments` / `semantic-cues`。默认建议 `semantic-cues`。 |
| 编辑强度 | `proofread_edit_level` | `punctuation` / `light` / `standard` / `strict`。 |
| 语言 | `proofread_language` | 可继承 ASR language。 |
| 术语表 | `glossary_path` | 专名、产品名、固定译法。 |
| 保护规则 | `protected_terms` | 不允许模型改写的 token。 |
| batch size | `proofread_batch_cues` | 控制上下文和成本。 |
| 风险阈值 | `review_threshold` | 高风险 cue 标记人工复核。 |
| 是否允许重切 | `proofread_allow_resegment` | 默认 false；如开启必须输出建议而非直接改时间。 |

### 8. 翻译参数

| 参数 | 建议名称 | 作用 |
|---|---|---|
| 是否启用 | `translate` | 启用翻译。 |
| 目标语言 | `target_language` | 如 `zh-CN`、`en`。 |
| 输入层 | `translation_input` | 默认 `proofread-cues`，无校对时退回 `semantic-cues`。 |
| 风格 | `translation_style` | `literal` / `natural` / `subtitle` / `technical`。 |
| 长度策略 | `length_policy` | `preserve` / `subtitle-fit`。 |
| cue 关系 | `cue_mapping_policy` | `one-to-one` / `group-within-speaker` / `split-to-fit`。 |
| 术语表 | `glossary_path` | 目标语言术语。 |
| 是否输出 SRT/VTT | `emit_translated_srt`、`emit_translated_vtt` | 翻译字幕渲染。 |

### 9. 输出与事件参数

| 参数 | 当前/建议名称 | 作用 |
|---|---|---|
| SRT | `emit_srt` | 输出源语言 SRT。 |
| VTT | `emit_vtt` | 输出源语言 VTT。 |
| JSON events | `json_events` | stderr NDJSON 事件流。 |
| 产物 profile | `artifact_profile` | 未来可区分 `minimal` / `debug` / `product`。 |

## 产物目录

### 当前稳定产物

| 产物 | 来源能力 | 语义层级 | 是否 source of truth | 说明 |
|---|---|---|---|---|
| `work/input.16khz.mono.wav` | 音频准备 | 媒体中间层 | 否 | 转录/diarization 实际输入。 |
| `work/chunks/chunk_NNN.wav` | chunk 抽取 | 媒体中间层 | 否 | 单 chunk ASR 输入。 |
| `work/chunks/chunk_NNN.json` | ASR | provider raw | 是，针对单 chunk | whisper.cpp 原始 JSON。 |
| `work/chunks/hallucinations.log` | 幻觉过滤 | 审计 | 否 | 被过滤条目的 NDJSON 日志。 |
| `work/merge.json` | 多 chunk 合并 | 审计 | 否 | overlap 保留/丢弃决策。 |
| `work/diarization.json` | 说话人切分 | speaker fact | 是，针对 speaker turns | pyannote 输出和 speaker 统计。 |
| `transcript.voxkit.json` | ASR pipeline | voxkit 原生 transcript | 是 | 丰富审计字段，voxkit 内部主产物。 |
| `transcript.raw.json` | Remixr adapter | Remixr transcript | 是 | 下游兼容主产物；不写 proofread 字段。 |
| `subtitles.cues.json` | 语义重切 | 渲染层结构化 cue | 是，针对字幕展示 | SRT/VTT 的结构化同源产物。 |
| `subtitles.srt` | 字幕渲染 | 展示格式 | 否 | 给播放器/人工查看。 |
| `subtitles.vtt` | 字幕渲染 | 展示格式 | 否 | Web 播放器友好。 |
| `events.ndjson` | 事件流 | 运行时观测 | 否 | 实时 UI 和 debug。 |
| `manifest.json` | 汇总 | 索引与审计 | 否 | artifacts、warnings、metrics、参数快照。 |

### 建议新增产物

| 产物 | 来源能力 | 语义层级 | 是否 source of truth | 说明 |
|---|---|---|---|---|
| `subtitles.proofread.json` | LLM/人工校对 | 源语言文本增强 | 是，针对校对文本 | 引用 cue id，保留原时间轴。 |
| `subtitles.proofread.diff.json` | 校对 diff | 审计 | 否 | raw/proofread 差异、风险分级。 |
| `subtitles.<lang>.json` | 翻译 | 目标语言 cue 流 | 是，针对目标语言字幕 | 可 one-to-one，也可在同 speaker 时间范围内重排。 |
| `subtitles.<lang>.srt` | 翻译渲染 | 展示格式 | 否 | 目标语言 SRT。 |
| `subtitles.<lang>.vtt` | 翻译渲染 | 展示格式 | 否 | 目标语言 VTT。 |
| `quality.report.json` | 质量评估 | 审计 | 否 | 多阶段指标、风险 cue、抽样建议。 |
| `glossary.applied.json` | 术语归一 | 审计 | 否 | 哪些术语被保护或替换。 |

## 产物生命周期

产物不只有“存在/不存在”，还应有状态。状态用于产品 UI、重跑策略、人工审核和缓存判断。

| 状态 | 含义 | 典型产物 | 可被机器覆盖吗 |
|---|---|---|---|
| `partial` | 实时或批处理中间结果，尚未稳定。 | ASR partial events、LLM batch partial events | 可以。 |
| `draft` | 机器生成的完整候选结果。 | `subtitles.proofread.json`、`subtitles.<lang>.json` | 可以，但应保留旧版本或 diff。 |
| `reviewed` | 人工确认或高置信规则确认过。 | `subtitles.reviewed.json` | 默认不覆盖，除非显式 force。 |
| `final` | 被锁定用于发布、交付或外部系统消费。 | 发布版字幕、最终翻译 | 不允许自动覆盖。 |
| `stale` | 上游输入、参数、模型、prompt 或 glossary 变了。 | 任意下游产物 | 不直接删除，但 UI 应提示过期。 |
| `failed` | 阶段失败并留下错误信息。 | `manifest.json` 中的 stage status | 不应写半成品冒充成功。 |

生命周期规则：

1. 上游 source of truth 变化时，下游 draft/reviewed/final 必须重新计算 freshness。
2. `reviewed` 和 `final` 产物应有人工操作记录，例如 reviewer、reviewedAt、baseArtifact hash。
3. 机器重跑默认只覆盖 `draft`；覆盖 `reviewed` / `final` 需要显式参数。
4. `stale` 不是错误状态，它表示产物仍可查看，但不再代表当前输入和参数。
5. 实时 `partial` 只服务 UI，不参与长期缓存和最终导出。

## 产物关系

```mermaid
flowchart TD
    input["input media"] --> prep["audio prep"]
    prep --> audio["work/input.16khz.mono.wav"]
    audio --> chunk["chunk planning/extraction"]
    chunk --> chunkwav["chunk_NNN.wav"]
    chunkwav --> asr["ASR"]
    asr --> chunkjson["chunk_NNN.json"]
    chunkjson --> filter["hallucination filter"]
    filter --> segment["segment entries"]
    segment --> merge["merge chunks"]
    merge --> voxkit["transcript.voxkit.json"]
    voxkit --> raw["transcript.raw.json"]

    audio --> dia["diarization"]
    dia --> diajson["work/diarization.json"]
    diajson --> align["speaker alignment"]
    raw --> align
    align --> raw_spk["speaker-aware transcript.raw.json"]

    raw_spk --> reseg["semantic resegment"]
    reseg --> cues["subtitles.cues.json"]
    cues --> render_src["source SRT/VTT"]
    cues --> proofread["proofread"]
    proofread --> proofjson["subtitles.proofread.json"]
    proofjson --> translate["translate"]
    cues --> translate
    translate --> targetjson["subtitles.<lang>.json"]
    targetjson --> render_target["translated SRT/VTT"]

    reseg --> metrics["subtitle metrics"]
    proofread --> quality["quality report"]
    translate --> quality
    metrics --> manifest["manifest.json"]
    quality --> manifest
```

关系规则：

1. `transcript.voxkit.json` 和 `transcript.raw.json` 是 ASR/transcript 层产物，不承载校对后的文本。
2. `subtitles.cues.json` 是源语言渲染层 source of truth；SRT/VTT 只是它的展示格式。
3. `subtitles.proofread.json` 只引用 cue id，不覆盖 `subtitles.cues.json`。
4. `subtitles.<lang>.json` 是目标语言字幕 source of truth，不覆盖源语言 cue。
5. `manifest.json` 只做索引和审计，不应成为唯一数据来源。
6. `events.ndjson` 可以包含实时 partial 数据，但稳定文件写出后以稳定文件为准。

## 系统不变量

这些规则优先级高于任何单个能力的实现细节。

### 数据层不变量

- `transcript.voxkit.json` 和 `transcript.raw.json` 只表达 ASR/transcript 层，不写入校对后的文本。
- `subtitles.cues.json` 只表达源语言渲染层 cue，不被 LLM 校对结果覆盖。
- `subtitles.proofread.json` 引用源 cue id，保存校对文本和风险标记，不反写 raw/cues。
- `subtitles.<lang>.json` 是目标语言字幕产物，不覆盖源语言字幕。
- SRT/VTT 永远是渲染结果，不作为结构化 source of truth。

### 时间轴不变量

- LLM 默认不得修改 `start` / `end` / `speaker`。
- 如未来允许 retiming，必须输出 retiming proposal，经过确定性校验后由独立阶段应用。
- 翻译 cue 不得跨 speaker 合并。
- 翻译 cue 的时间范围应被其 `sourceCueIds` 覆盖；任何扩展都必须记录原因。
- CJK 字符插值是字幕层 timebase，不应伪装成 word-level timestamp。

### 可审计性不变量

- 每个 LLM 产物必须记录 provider、model、prompt/schema version、参数、输入 artifact hash。
- 每个高风险修改必须能定位到原 cue 和原文本。
- 同一次运行内，稳定产物写出应是原子性的：要么完整成功，要么不替换旧文件。
- 参数默认值变化需要在 changelog 或 manifest 中可追踪。

## 校对产物建议 schema

校对默认以 `subtitles.cues.json` 为输入，因为它已经是适合人类阅读的字幕单元。LLM 不直接处理 SRT，不直接改时间戳。

```jsonc
{
  "schemaVersion": "1",
  "sourceId": "YTVSwOY19Qs",
  "inputArtifact": "subtitles.cues.json",
  "language": "zh",
  "provider": "openai",
  "model": "example-model",
  "params": {
    "editLevel": "standard",
    "allowRetiming": false,
    "glossaryVersion": "2026-05-10"
  },
  "cues": [
    {
      "cueId": "cue_000001",
      "sourceStart": 0.1,
      "sourceEnd": 4.2,
      "speaker": "Speaker 1",
      "sourceText": "原始字幕文本",
      "correctedText": "校对后的字幕文本",
      "editLevel": "minor",
      "risk": "low",
      "needsHumanReview": false,
      "notes": []
    }
  ],
  "metrics": {
    "cueCount": 1000,
    "changedCueRate": 0.37,
    "reviewCueRate": 0.04
  }
}
```

校对 invariants：

- `cueId` 必须来自输入 cue，且不重复。
- 默认 `sourceStart` / `sourceEnd` 必须等于输入 cue 的时间。
- `correctedText` 不得为空，除非原文为空或 cue 被标记为删除建议。
- 数字、日期、人名、产品名、URL、代码片段变更应标记更高风险。
- LLM 无法确定时应保守输出 `needsHumanReview=true`，而不是编造。

## 翻译产物建议 schema

翻译建议以校对后的 cue 为输入；如果没有校对产物，可以回退到 `subtitles.cues.json`。翻译允许在同一 speaker 的连续时间范围内重新拆分，以满足目标语言阅读速度。

```jsonc
{
  "schemaVersion": "1",
  "sourceId": "YTVSwOY19Qs",
  "inputArtifact": "subtitles.proofread.json",
  "sourceLanguage": "zh",
  "targetLanguage": "en",
  "provider": "openai",
  "model": "example-model",
  "params": {
    "style": "subtitle",
    "lengthPolicy": "subtitle-fit",
    "cueMappingPolicy": "group-within-speaker"
  },
  "cues": [
    {
      "id": "trg_000001",
      "sourceCueIds": ["cue_000001", "cue_000002"],
      "start": 0.1,
      "end": 5.8,
      "speaker": "Speaker 1",
      "text": "Translated subtitle text.",
      "mapping": "merged",
      "risk": "low"
    }
  ],
  "metrics": {
    "cueCount": 920,
    "overCpsRate": 0.02,
    "overCharLimitRate": 0.01
  }
}
```

翻译 invariants：

- `sourceCueIds` 必须能追溯到源语言 cue。
- 翻译 cue 不得跨 speaker 合并。
- 翻译 cue 的时间范围默认必须被其 `sourceCueIds` 覆盖；如需扩展时间，必须显式标记原因。
- 目标语言 SRT/VTT 必须从 `subtitles.<lang>.json` 渲染，不从 LLM 直接生成的 SRT 读取。
- 翻译后的 cue 也要跑字幕物理指标：duration、chars、CPS、闪现率。

## 质量指标与风险分级

质量指标分两类：一类是物理指标，能自动计算；另一类是语义风险，通常用于抽样或人工复核。

### 字幕物理指标

| 指标 | 适用产物 | 含义 |
|---|---|---|
| `cueCount` | cues/proofread/translation | cue 总数。 |
| `avgCueDurS` / `p50CueDurS` / `p90CueDurS` | cues/proofread/translation | 字幕展示时长分布。 |
| `flashCueRate` | cues/proofread/translation | 低于 `min_dur_s` 的闪现字幕比例。 |
| `longCueRate` | cues/proofread/translation | 超过 `max_dur_s` 的长字幕比例。 |
| `avgChars` | cues/proofread/translation | 平均字符数。 |
| `overCharLimitRate` | cues/proofread/translation | 超过字符上限的比例。 |
| `overCpsRate` | cues/proofread/translation | 超过阅读速度上限的比例。 |
| `speakerSwitchCueRate` | cues/translation | cue 是否异常跨 speaker 或 speaker 变化过密。 |

### ASR 与时间轴风险

| 风险 | 触发条件 | 建议处理 |
|---|---|---|
| `low_confidence` | `avg_confidence` 低或 logprob 低。 | 校对时给更多上下文，或标人工复核。 |
| `hallucination_drop_nearby` | cue 附近有 blocklist 过滤记录。 | 标记高风险。 |
| `chunk_boundary_nearby` | cue 距 chunk 接缝很近。 | 检查截断、重复、漏词。 |
| `speaker_unmatched` | speaker 为 `Speaker ?` 或 diarization 无 overlap。 | UI 提醒 speaker 不确定。 |
| `timebase_interpolated` | CJK 字符插值生成时间。 | 不用于精确剪辑点，只用于字幕展示。 |

### 校对风险

| 风险 | 触发条件 | 建议处理 |
|---|---|---|
| `numeric_change` | 数字、日期、金额、百分比变化。 | 默认人工复核。 |
| `named_entity_change` | 人名、机构、产品、地名变化。 | 结合 glossary 或人工复核。 |
| `large_text_delta` | 文本长度或编辑距离超阈值。 | 标记 major edit。 |
| `empty_or_deleted` | 非空源文本被改为空。 | 默认拒绝或人工确认。 |
| `protected_term_change` | protected terms 被改写。 | 自动回滚该 cue 或标 high risk。 |
| `uncertain_model_output` | LLM 自报不确定或输出 schema 校验失败。 | 重试或人工复核。 |

### 翻译风险

| 风险 | 触发条件 | 建议处理 |
|---|---|---|
| `length_expansion` | 目标语言显著长于源语言，CPS 超限。 | 触发 subtitle-fit 重排或二次压缩。 |
| `glossary_miss` | 目标文本没有使用指定术语。 | 自动修正或复核。 |
| `source_coverage_gap` | source cue 没有对应翻译。 | 重试该 batch。 |
| `source_coverage_duplicate` | source cue 被多个目标 cue 重复覆盖且非 split。 | 校验 mapping。 |
| `speaker_crossing` | 目标 cue 跨 speaker。 | 自动拒绝。 |
| `style_violation` | 不符合指定 style，例如过度意译。 | 低优先级复核或模型重试。 |

建议风险等级：

| 等级 | 含义 | 默认动作 |
|---|---|---|
| `low` | 普通修改，自动接受。 | 进入 draft。 |
| `medium` | 可能影响理解。 | UI 标记，可抽样复核。 |
| `high` | 可能引入事实错误。 | 默认人工复核。 |
| `blocking` | 违反 schema、不变量或覆盖关系。 | 在 cue 层强制 `needsHumanReview=true`。**当前实现仍写入 stable draft（cue 仍可被人工修复），但 `quality.report.json` 风险直方图会单独凸显** —— 严格 "拒绝写产物" 只在批级 transport 失败时触发（写 `pending` marker，整批 artifact 不落盘）。 |

> **未知/缺失 risk** 在 `quality.report.json` 聚合时会被强制归入 `blocking` 桶（保守路线），避免 malformed LLM 输出悄悄按 `low` 通过审核。

## 实时数据与稳定产物

实时能力应该使用事件流承载中间状态，稳定产物只在阶段完成后写入。

建议事件类型：

| 事件 | 含义 |
|---|---|
| `audio.prep.start/done` | 音频准备开始/完成。 |
| `chunk.plan.done` | chunk 计划生成。 |
| `asr.chunk.start/partial/done` | 单 chunk ASR 进度；partial 只供 UI 展示。 |
| `asr.merge.done` | 多 chunk 合并完成。 |
| `diarization.start/done` | 说话人切分开始/完成。 |
| `resegment.done` | 字幕 cue 重切完成。 |
| `proofread.batch.start/done` | 校对批次完成。 |
| `translate.batch.start/done` | 翻译批次完成。 |
| `artifact.write` | 某个稳定产物写出。 |
| `quality.done` | 指标和风险报告生成。 |

事件规则：

- partial 事件不能被下游当成最终 transcript。
- 稳定产物写出后，manifest 记录路径、hash、参数快照和 warnings。
- 如果某阶段失败，已有上游稳定产物仍可复用；下游产物必须标记缺失或失败，不应写半成品覆盖旧文件。

## 成本、缓存与可复现性

ASR、diarization 和 LLM 阶段的成本结构不同。设计上应允许每个阶段单独缓存、重跑和审计。

### 缓存键

| 阶段 | 建议缓存键组成 |
|---|---|
| audio prep | input file path、mtime/hash、ffmpeg 参数。 |
| chunk ASR | chunk audio hash、ASR provider、model、language、word timestamp mode、VAD 参数、logprob 阈值。 |
| diarization | audio hash、diarize model、speaker hints、device 无关模型参数。 |
| semantic resegment | input transcript hash、language、`ResegmentParams`。 |
| proofread | `contentHash` = sha256(id, text, start, end, speaker)；`policyHash` = sha256(provider, model, promptVersion, promptHash, editLevel, glossaryHash, cacheSchema)；命中需两者都相等且 cacheSchema 与当前实现一致。 |
| translation | `contentHash` 与 proofread 同形；`policyHash` 含 (provider, model, promptVersion, promptHash, style, lengthPolicy, cueMappingPolicy, glossaryHash, sourceLanguage, targetLanguage, cacheSchema)。 |

> **content hash 必须含 start/end/speaker**：上游 source cue 改了时间或 speaker 但 id+text 没变时，cache 必须失效——否则下游 proofread/translation 会按旧时间轴回写，污染"时间轴不变量"。

### Manifest 应记录的信息

实际实现采用**顶层** `proofread` 与 `translations.<lang>` 两个段（不是 `stages.<name>` 命名空间），用于减少嵌套深度。下游消费者（含 `voxkit review` 镜像写入路径）必须从这两个 key 读：

```jsonc
{
  "proofread": { "state": "...", "provider": "...", "model": "...",
                 "promptVersion": "proofread.v1", "promptHash": "...",
                 "inputArtifact": "subtitles.cues.json", "inputHash": "sha256:...",
                 "outputArtifact": "subtitles.proofread.json", "outputSchemaVersion": "1",
                 "freshPromptTokens": ..., "cachedPromptTokens": ...,
                 "promptTokens": ..., "completionTokens": ..., ... },
  "translations": {
    "zh": { "state": "...", "sourceLanguage": "...", "targetLanguage": "zh",
            "outputArtifact": "subtitles.zh.json", "outputSchemaVersion": "1",
            "style": "...", "lengthPolicy": "...", "cueMappingPolicy": "one-to-one",
            "freshPromptTokens": ..., "cachedPromptTokens": ..., ... }
  }
}
```

| 字段 | 作用 |
|---|---|
| artifact path/hash | 判断 freshness 和复现输入。 |
| provider/model | 排查模型升级影响。 |
| prompt/schema version | LLM 行为变化可追踪。 |
| glossary version/hash | 术语变化可追踪。 |
| elapsed/rtf/token usage | 评估性能和成本（fresh vs cached 拆分）。 |
| warnings/errors | 产品 UI 和自动复核入口。 |
| metrics | 策略比较和质量门禁。 |

### 重跑策略

- 默认重跑只更新缺失或 stale 的 draft 产物。
- `--force` 三档（与 `voxkit proofread` / `voxkit translate` CLI 同名）：
  - `--force`：只覆盖 draft；遇 reviewed/final 拒绝。
  - `--force-reviewed`：允许覆盖 reviewed；隐含 `--force`。
  - `--force-final`：允许覆盖 final；销毁人工 lock 元数据，慎用。
- 任一 force 档都**只清空 `work/proofread/` 或 `work/translate.<lang>/` checkpoint 目录**；旧 stable artifact **不预先 unlink**，仅在新批次全部完成后通过 `os.replace` 原子替换。LLM 中途失败时旧 artifact 完整保留。
- LLM batch 失败分两类：
  - 内容层（`LLMSchemaError` / `LLMRefusal`）：本批 fallback 写 risk=blocking + needsHumanReview，落 checkpoint，run 继续。
  - 传输/限流（`LLMTimeout` / `LLMRateLimit` / 5xx 耗尽）：本批写 `batch_NNN.pending.json` marker，run 继续；末尾若有任何 pending → **拒绝写稳定 artifact** + 抛 `LLMError("incomplete")`。rerun（无需 --force）只重做 pending 批，已完成 checkpoint 自动复用。
- 旧版本产物应至少保留 hash 和 manifest 记录；是否保留完整文件可由 artifact retention policy 控制。
- 当 provider/model 不可用时，应明确失败，不要静默切换到另一个模型生成不可比较的产物。
- 成本审计：manifest 中 `freshPromptTokens` / `freshCompletionTokens` 记录本轮真的花掉的 token；`cachedPromptTokens` / `cachedCompletionTokens` 来自 checkpoint。两者之和（`promptTokens` / `completionTokens`）保留兼容旧消费者。

## 产品视角的能力组合

| 产品模式 | 推荐能力链 | 默认关注点 | 主要产物 |
|---|---|---|---|
| 快速转录 | audio prep -> ASR -> merge -> raw transcript -> SRT/VTT | 速度、resume、低成本 | `transcript.raw.json`、`subtitles.srt/vtt` |
| 播放器字幕 | 快速转录 -> semantic resegment -> cues -> SRT/VTT | 阅读体验、CPS、闪现率 | `subtitles.cues.json` |
| 访谈/多人内容 | ASR -> diarization -> speaker alignment -> resegment | speaker 准确性、跨 speaker 不合并 | `work/diarization.json`、speaker-aware cues |
| 高质量源语言字幕 | 播放器字幕 -> proofread -> risk report -> optional review | 术语、标点、事实安全 | `subtitles.proofread.json`、diff |
| 双语字幕 | proofread -> translate -> subtitle-fit -> target render | 目标语言阅读速度、术语一致 | `subtitles.<lang>.json`、target SRT/VTT |
| 精修交付 | proofread/translation draft -> human review -> final lock | 生命周期、权限、版本冻结 | `subtitles.reviewed.json`、final artifacts |
| 数据分析/剪辑 | raw transcript + speaker turns + cues + proofread text | 时间精度、可追溯、检索 | raw transcript、speaker turns、cue index |
| 实时预览 | audio stream -> partial ASR events -> final stabilization | latency、partial/final 区分 | events、final segments |

建议第一批 preset：

| Preset | 能力组合 | 推荐默认值 |
|---|---|---|
| `fast-transcript` | ASR + merge + raw export | `resegment=none`，不启用 LLM。 |
| `subtitle` | ASR + semantic resegment + SRT/VTT | `resegment=semantic`，输出 `subtitles.cues.json`。 |
| `interview` | ASR + diarization + semantic resegment | `with_diarization=true`，`speaker_labels=ranked`。 |
| `proofread-subtitle` | subtitle + proofread + diff/risk | `proofread_input=semantic-cues`，`allowRetiming=false`。 |
| `bilingual-subtitle` | proofread-subtitle + translation | `length_policy=subtitle-fit`，允许同 speaker group 重排。 |
| `review-ready` | bilingual-subtitle + quality report | high/blocking 风险进入人工审核队列。 |

## 推荐实现顺序

1. **冻结能力模型和不变量**：确认本文的层级和产物命名，尤其是 `raw` / `cues` / `proofread` / `translation` 的边界。
2. **给 cue 补稳定 id**：`subtitles.cues.json` 当前只有 `start/end/speaker/text`，加入 `id` 会让 proofread/translation 更稳。
3. **加入 artifact hash 与 freshness**：manifest 记录输入/输出 hash，用于 stale 判断和缓存。
4. **实现 proofread draft**：从 `subtitles.cues.json` 读入，输出 `subtitles.proofread.json`，严格 schema 校验，不改时间。
5. **实现 proofread diff/risk**：本地对比 source/corrected，标记数字、专名、长度异常、空文本等风险。
6. **实现 translation draft**：优先从 proofread 输入，输出 `subtitles.<lang>.json`，再渲染目标语言 SRT/VTT。
7. **统一 metrics**：raw cues、proofread cues、translated cues 都跑字幕物理指标。
8. **把参数写入 manifest**：每个能力记录 provider、model、prompt/schema version、glossary version、metrics。
9. **实现 preset 层**：把常用能力链封装成 `fast-transcript`、`subtitle`、`bilingual-subtitle` 等模式。
10. **再考虑 UI/人工审核**：产品侧读取 proofread/translation JSON，人工编辑后写 reviewed/final 产物。

## 待定问题

- 校对输入默认用 `subtitles.cues.json` 还是 `transcript.raw.json`？建议默认 cues，必要时可回看 raw words。
- proofread 是否允许合并/拆分 cue？建议第一版不允许，只输出文本修正；重切建议单独产物。
- 翻译是否强制 one-to-one？建议不要强制；默认允许同 speaker 连续 cue 在同一时间范围内重排。
- `subtitles.cues.json` 是否升 schema version 来加入 cue id？建议升到 `"2"` 或允许兼容读取缺省 id。
- glossary 是全局项目配置，还是每次任务输入？建议两者都支持，并把版本/hash 写 manifest。
- 多 provider ASR 的 provider raw 是否落盘？建议落盘，方便审计和 normalize bug 排查。
- reviewed/final 产物的覆盖权限放在 CLI 层、产品层，还是 workspace lock 层？建议至少在 workspace 层防误覆盖。
- LLM 成本预算是按任务设置，还是按 workspace/project 设置？建议两者都支持，任务级覆盖项目默认值。
- preset 是作为 CLI 参数 `--preset` 暴露，还是作为产品 API 的 job template 暴露？建议底层同一套配置，入口不同。
