# `voxkit transcribe` — 深入

本文是 `voxkit transcribe` 的完整参考。CLI 表层用法见
[`README.md`](../README.md#上手--transcribe)。

## 数据流

```
input.mp4
    │
    ▼  ffmpeg -ar 16000 -ac 1
work/input.16khz.mono.wav
    │
    ▼  plan_chunks(threshold=900s, chunk=600s, overlap=5s)
ChunkPlan: [ChunkSpec(0, 0..600), ChunkSpec(1, 595..1195), …]
    │
    ▼  per chunk:  ffmpeg -ss <start> -t <dur>  →  whisper-cli (-mc 0 -lpt -0.8 --vad ...)
work/chunks/chunk_NNN.{wav,json}              (checkpoint：--resume 命中即跳过)
    │
    ▼  parse_whisper_json → list[Entry]       (chunk-relative 时间)
    │
    ▼  filter_entries(blocklist)              (watermark / standalone / ghost-loop)
work/chunks/hallucinations.log                (NDJSON 审计日志)
    │
    ▼  detect_mode + segment_entries          (英文 word 模式 / CJK phrase 模式)
list[TranscriptSegment]                       (chunk-relative)
    │
    ▼  merge_chunks(overlap_tolerance=0.5s)   (offset_segment 同步偏移 segment 与 words[])
list[TranscriptSegment]                       (绝对时间)
work/merge.json                               (每 chunk 保留/丢弃 segment id)
    │
    ▼  TranscriptionOutput (Pydantic, schemaVersion="1")
    │
    ▼  optional: --with-diarization → pyannote → speaker labels
    │
    ├──► transcript.voxkit.json               (voxkit 原生)
    ├──► transcript.raw.json                  (Remixr-shaped, exclusive write)
    ├──► subtitles.srt / subtitles.vtt
    ├──► subtitles.cues.json                  (--resegment=semantic 才写; 渲染层 cues 的机读出口)
    └──► manifest.json                        (含 perChunk + warnings)
```

整套编排实现在 `src/voxkit/core/transcribe_pipeline.py::run_pipeline`，CLI 入口
`src/voxkit/commands/transcribe.py`。每个箭头对应一个 NDJSON 事件写入 `events.ndjson`
（`--json-events` 同时镜像到 stderr）。

## 关键阈值

| 常量 | 值 | 来源 |
|---|---|---|
| `CHUNK_THRESHOLD_SECS` | 900 | 短于 15 min 的输入不分块 |
| `CHUNK_DURATION_SECS` | 600 | 每 chunk 10 min |
| `CHUNK_OVERLAP_SECS` | 5 | 相邻 chunk 5 s 重叠（dedup 用） |
| `OVERLAP_DEDUP_TOLERANCE_SECS` | 0.5 | merge 时容忍 ±500 ms |
| `--logprob-thold` | -0.8 | whisper.cpp 默认 -1.0，本工具收紧 |
| `dynamic timeout` | `max(30 min, duration*0.3)` | per-chunk |

均移植自 Remixr `services/whisper.ts`，是长音频转录实战中沉淀下来的数值。
chunk 相关默认值可用 `VOXKIT_CHUNK_*` 环境变量覆盖，也可以通过
`--chunk-threshold-secs` / `--chunk-secs` / `--chunk-overlap-secs` 在 CLI
单次运行中覆盖；这主要用于短音频测试和切分策略 A/B。

## 输出文件 schema

### `transcript.raw.json`（Remixr 兼容）

字节级对应 [`packages/shared/src/types/transcript.ts`](https://github.com/3Craft/CutFlow/blob/main/packages/shared/src/types/transcript.ts)
的 Zod schema。可作为 Remixr `storage/projects/<projectId>/sources/<sourceId>/transcript.raw.json`
的 drop-in 替换。

```json
{
  "sourceId": "YTVSwOY19Qs",
  "segments": [
    {
      "id": "seg_001",
      "speaker": "Speaker A",
      "start": 0.1,
      "end": 5.48,
      "text": "Since last year, I've just had this existential dread and also, like, hope",
      "subtitles": [],
      "words": [
        { "word": "Since", "start": 0.1, "end": 0.3 },
        { "word": "last",  "start": 0.3, "end": 0.46 }
      ]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `sourceId` | string | Remixr 的 source 标识，由 CLI `--source-id` 控制（默认 input 文件 stem） |
| `segments[].id` | string | `seg_NNN`（1-based，3 位零填充） |
| `segments[].speaker` | string | 默认 `"Speaker A"`；传 `--with-diarization` 后写入 `Speaker 1/2/...` 或 `SPEAKER_00` 等真实标签 |
| `segments[].start` / `.end` | number (s) | 绝对时间 |
| `segments[].text` | string | 已合并的句段文本（segmenter 4 优先级边界产出） |
| `segments[].subtitles` | string[] | **总是 `[]`** —— Remixr proofread agent 后续填充 |
| `segments[].words` | array | 英文 word 模式有内容；CJK phrase 模式为 `[]` |
| `segments[].words[].word` | string | 单词原文（可能带尾标点，如 `"year,"`） |
| `segments[].words[].start` / `.end` | number (s) | 绝对时间 |

注意：voxkit **故意不写** `rawText` 字段——那是 Remixr proofread 阶段产物，写了会污染
Remixr 的「未校对」判定。

#### `--with-diarization` 对 raw JSON 的影响

不开启时，`transcript.raw.json` 保持 ASR-only 形态，`segments[].speaker` 使用 Remixr 兼容占位
`"Speaker A"`。开启 `--with-diarization` 后，pipeline 会在 ASR merge 完成后运行 pyannote，
按时间重叠把 speaker 注入每个 segment：

- `--speaker-labels ranked`：输出 `Speaker 1/2/...`，按说话总时长排序，适合产品 UI。
- `--speaker-labels raw`：输出 `SPEAKER_00` 等 pyannote 原始标签，适合调试或算法对比。
- 未匹配到 diarization 的 segment 会标成 `Speaker ?`，并在 manifest / `_metadata.warnings` 中留下审计信息。

`transcript.voxkit.json` 仍然不承载 speaker 字段；speaker 属于 Remixr 形态和字幕渲染层。

可选 `_metadata` 顶层字段（Remixr 忽略未知 key，安全）：

```json
{
  "_metadata": {
    "voxkitVersion": "0.4.0",
    "asrBackend": "whisper-cpp",
    "asrModel": "ggml-large-v3-turbo.bin",
    "language": "en",
    "sourceDurationSecs": 3821.61,
    "processedAt": "2026-04-28T04:32:32.646537+00:00",
    "whisperBin": "/opt/homebrew/bin/whisper-cli",
    "vadModel": "/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin",
    "warnings": []
  }
}
```

### `subtitles.cues.json`（语义重切的机读出口）

仅当 `--resegment=semantic` 且重切真的产出了 cues 时才写（diarized fallback 不算）。
与 `subtitles.srt/vtt` 同源——同一份 `SubtitleCue[]` 三处渲染：SRT、VTT、JSON。

**英文路径**：pysbd 句子边界 → 长句 split_long → 短 cue 合并 → 单调钳位。
**CJK 路径**：whisper.cpp 不输出 word timestamp 故跳过 pysbd，但 **短 cue 合并仍然生效**——
将 < `min_dur_s`（默认 1.5s）的同 speaker 相邻 cue 合并以消除闪现字幕。实测某 106 min
中文播客：4426 → 2721 cues（−38.5%），平均时长 1.43s → 2.33s，闪现率 58.8% → 0%。
长 segment 在 CJK 不做拆分（segmenter 的 5s/100chars 上限已实务封顶）。

```jsonc
{
  "schemaVersion": "1",
  "sourceId": "YTVSwOY19Qs",
  "resegment": "semantic",
  "params": {
    "max_dur_s": 7.0,
    "min_dur_s": 1.5,
    "max_chars": 84,
    "soft_max_chars": 75,
    "max_cps": 22.0,
    "prosody_gap_s": 0.25,
    "prosody_gap_weight": 7,
    "soft_break_weights": { ";": 10, ":": 8, ",": 3, "but": 4 }
  },
  "cues": [
    { "start": 0.10, "end": 5.48, "speaker": "Speaker A", "text": "Since last year..." },
    { "start": 5.50, "end": 9.12, "speaker": "Speaker B", "text": "Yeah, exactly." }
  ]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `schemaVersion` | string | 独立版本计数器，初始 `"1"` |
| `sourceId` | string | 与 `transcript.raw.json` 中一致 |
| `resegment` | string | 当前固定 `"semantic"`；预留扩展（pysbd 之外的策略） |
| `params` | object \| 缺省 | `ResegmentParams` 快照，可复现 |
| `cues[].start` / `.end` | number (s) | 绝对时间，浮点（不是 SRT 的 ms 取整） |
| `cues[].speaker` | string \| null | diarization 跑过则填 ranked label，否则 `null` 或 `"Speaker A"` |
| `cues[].text` | string | 已合并的子句文本 |

**与 `transcript.raw.json` 的关系**：完全解耦。`raw.json` 是 ASR ground truth（segmenter 的粗粒度切分 + word 时间戳），`cues.json` 是渲染层决策（pysbd 句子边界 + 物理上限切分）。下游想要语义重切的成果就读这份；想要原始 ASR 还是去读 `raw.json`。**禁止**把 cues 反向塞回 `raw.json.segments[].subtitles[]`——那会触发 Remixr proofread agent 的「已校对」误判。

**Remixr 集成示例**（TS 伪码）：

```ts
import { z } from "zod";

const SubtitleCueZ = z.object({
  start: z.number(),
  end:   z.number(),
  speaker: z.string().nullable().optional(),
  text:  z.string(),
});
const CuesFileZ = z.object({
  schemaVersion: z.literal("1"),
  sourceId: z.string(),
  resegment: z.string(),
  cues: z.array(SubtitleCueZ),
});

const cuesPath = `storage/projects/${pid}/sources/${sid}/subtitles.cues.json`;
const cues = CuesFileZ.parse(JSON.parse(await fs.readFile(cuesPath, "utf8")));
// cues.cues 即可直接喂播放器/字幕预览，无需反解 SRT
```

manifest 里通过 `artifacts.subtitle_cues_json` 声明该文件是否产出。

### `transcript.voxkit.json`（voxkit 原生）

`raw.json` 的超集，含审计信息：

```jsonc
{
  "schemaVersion": "1",
  "audio": {
    "path": ".../work/input.16khz.mono.wav",
    "durationSecs": 3821.609813,
    "extractedFrom": "tmp/YTVSwOY19Qs.mp4"   // 视频输入才有
  },
  "asrBackend": "whisper-cpp",
  "asrModel": "ggml-large-v3-turbo.bin",
  "language": "en",
  "wordTimestamps": true,
  "rtf": 0.0476,
  "elapsedSecs": 181.73,
  "perChunk": [ /* ChunkStat[] */ ],
  "hallucinationDrops": 0,
  "segments": [ /* TranscriptSegment[] */ ],
  "warnings": []
}
```

与 `raw.json` 的差异：

- 无 `speaker` / `subtitles` / `rawText`（这些是 Remixr 形态）
- 多 `audio` / `asrBackend` / `asrModel` / `language` / `rtf` / `elapsedSecs` / `perChunk`
- `segments[]` 多 `noSpeechProb` / `avgConfidence`（whisper.cpp 直出的置信度，可选）

Pydantic 模型见 `src/voxkit/io/schema.py::TranscriptionOutput`。

### `manifest.json`

每次运行的元数据 + 完整的 `perChunk` 统计 + warnings + artifact 路径表：

```jsonc
{
  "voxkitVersion": "0.4.0",
  "schemaVersion": "1",
  "startedAt": "2026-04-28T04:29:30.819719+00:00",
  "finishedAt": "2026-04-28T04:32:32.646537+00:00",
  "input": "tmp/YTVSwOY19Qs.mp4",
  "sourceId": "YTVSwOY19Qs",
  "workdir": "/abs/path/to/out",
  "asrBackend": "whisper-cpp",
  "asrModel": "ggml-large-v3-turbo.bin",
  "whisperBin": "/opt/homebrew/bin/whisper-cli",
  "vadModel": "/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin",
  "language": "en",
  "wordTimestamps": true,
  "vad": true,
  "logprobThold": -0.8,
  "resume": true,
  "elapsedSecs": 181.73,
  "rtf": 0.0476,
  "durationSecs": 3821.61,
  "chunkCount": 7,
  "perChunk": [
    { "index": 0, "startSecs": 0.0,    "durationSecs": 600.0, "elapsedSecs": 26.93, "rtf": 0.0449, "cached": false },
    { "index": 1, "startSecs": 595.0,  "durationSecs": 600.0, "elapsedSecs": 26.87, "rtf": 0.0448, "cached": false }
  ],
  "hallucinationDrops": 0,
  "mergeNotes": [
    { "kind": "out_of_order", "segId": "seg_008", "detail": "segments[7].start=27.710 < segments[6].end=27.790" }
  ],
  "warnings": [ "merge note: out_of_order at seg_008 (...)" ],
  "artifacts": {
    "raw_json":           "/abs/.../transcript.raw.json",
    "voxkit_json":        "/abs/.../transcript.voxkit.json",
    "manifest":           "/abs/.../manifest.json",
    "events":             "/abs/.../events.ndjson",
    "srt":                "/abs/.../subtitles.srt",
    "vtt":                "/abs/.../subtitles.vtt",
    "subtitle_cues_json": "/abs/.../subtitles.cues.json"
  }
}
```

`mergeNotes` 是 warn-only —— `out_of_order` 是相邻 segment 时间轴的 ms 级回溯，
不影响下游消费（whisper.cpp 偶尔吐出来的，4-character 重叠在播放器里完全不可见）。

### `events.ndjson`

每行一个 JSON 对象，记录全流程关键节点：

```jsonc
{"event": "start",           "stage": "pipeline", "input": "...", "workdir": "...", "voxkit_version": "0.4.0", "started_at": "..."}
{"event": "discover",        "whisper_cli": "...", "model": "...", "vad_model": "..."}
{"event": "audio.normalize.start", "input": "..."}
{"event": "audio.normalize.done",  "master_wav": "...", "duration_secs": 3821.61}
{"event": "plan",            "chunk_count": 7, "thresholds": {...}, "total_secs": 3821.61}
{"event": "progress",        "stage": "whisper.chunk", "chunk": 0, "percent": 5}
{"event": "progress",        "stage": "whisper.chunk", "chunk": 0, "percent": 10}
…
{"event": "chunk.done",      "chunk": 0, "elapsed_secs": 26.93, "rtf": 0.0449}
{"event": "merge.done",      "segments": 909, "merge_notes": 16}
{"event": "write.artifact",  "kind": "raw_json", "path": "..."}
{"event": "done",            "elapsed_secs": 181.73, "rtf": 0.0476}
```

`--json-events` 让 stderr 同时收到这些行（机器消费）；不传时 stderr 输出人读
进度，`events.ndjson` 文件**始终**写满，便于事后 grep。

### `work/merge.json`

每 chunk 输入了多少 segment、丢了哪些（overlap dedup 区间）：

```jsonc
{
  "chunks": [
    { "index": 0, "chunk_start_secs":    0.0, "segments_in": 140, "segments_dropped": [] },
    { "index": 1, "chunk_start_secs":  595.0, "segments_in": 154, "segments_dropped": [] },
    { "index": 2, "chunk_start_secs": 1190.0, "segments_in": 139, "segments_dropped": [] }
  ]
}
```

### `work/chunks/hallucinations.log`

NDJSON，每行一个被丢的 entry：

```jsonc
{"chunk_index": 3, "entry_index": 42, "text": "请订阅本频道", "rule": "standalone_match",   "matched_pattern": "请订阅本频道"}
{"chunk_index": 5, "entry_index":  7, "text": "啦啦啦啦啦啦啦啦啦啦啦啦", "rule": "ghost_cjk_loop", "matched_pattern": "啦啦啦啦啦啦"}
```

3 种 `rule`：

- `watermark_prefix` — 文本 NFC 归一化后 `startswith()` 命中黑名单前缀
- `standalone_match` — 归一化后 `==` 命中黑名单短语
- `ghost_cjk_loop` — 同 entry 内 ≥6 字 CJK 子串重复 ≥2 次（结构性，无关键字）

## 调试技巧

### 1. 用 `tmp/voxkit-real/` 做 schema 参考

仓库里 [`tmp/voxkit-real/`](../tmp/voxkit-real/) 是一次真实跑完 64 min 英文播客的完整工作目录，
manifest / events / work/ 子树齐全，可作 schema diff 的 ground truth：

```bash
jq '.segments[0]' tmp/voxkit-real/transcript.raw.json
jq '.perChunk | length' tmp/voxkit-real/manifest.json   # 7
jq '.warnings | length' tmp/voxkit-real/manifest.json   # 16（mergeNotes 副本）
wc -l tmp/voxkit-real/events.ndjson
```

### 2. Resume 行为验证

```bash
# 第一次：从零跑
voxkit transcribe long.mp4 --workdir out/ --json-events

# 立刻再跑一次：所有 chunk_NNN.json 命中，应当只见到 chunk.cached 事件，wall clock 秒级返回
voxkit transcribe long.mp4 --workdir out/ --json-events 2>&1 \
  | grep -E '"event":"chunk\.(start|cached|done)"'

# --force：清空 work/ 重跑，所有 chunk 重算
voxkit transcribe long.mp4 --workdir out/ --force
```

### 3. 严格幂等触发

```bash
voxkit transcribe input.mp4 --workdir out/      # OK
voxkit transcribe input.mp4 --workdir out/      # error: transcript.raw.json already exists
```

修复方式：换 `--workdir`，或先 `rm out/transcript.raw.json`。

### 4. CJK 模式自检

`--language=auto` 时，segmenter 按 whisper 实际输出特征判断模式：
**leading-space ratio ≥ 0.5 走英文 word 模式，否则走 CJK phrase 模式**。
强制覆盖用 `--language en` / `--language zh`。

CJK 模式下 `--word-timestamps` flag 会**自动忽略**（whisper.cpp 的 `--max-len 1 --split-on-word`
对中文产出很碎，没有意义），`segments[].words` 为 `[]`。

```bash
# 验证 CJK 模式 words 为空
jq '.segments[0].words | length' out/transcript.raw.json   # 0（CJK）
jq '.segments[0].words | length' out/transcript.raw.json   # > 0（英文）
```

### 5. 自定义反幻觉黑名单

bundled blocklist 见 `src/voxkit/data/hallucination_blocklist.json`。覆盖时整文件替换：

```jsonc
{
  "version": 1,
  "watermark_prefixes": ["你的频道水印"],
  "standalone_matches":  ["请关注我"],
  "ghost_loop": { "min_substring_chars": 6, "min_repeats": 2 },
  "normalize":  { "strip_chars": " \t\n\r.,。，、！？!?…-—　「」" }
}
```

```bash
voxkit transcribe input.mp4 --workdir out/ --blocklist /path/to/custom.json
```

### 6. whisper-cli 版本兼容自检

`voxkit doctor --profile transcribe` 启动期 grep 5 个关键 flag（`--output-json-full` / `--max-context` /
`--vad` / `--split-on-word` / `--logprob-thold`）；transcribe 入口再 re-check 一次。
缺任何 flag 都 fast-fail，不会跑到一半才报错。

```bash
voxkit doctor --profile transcribe 2>&1 | grep whisper
```

如果你的 whisper-cli 太旧：

```bash
brew upgrade whisper-cpp
```

## 已知行为

- **out_of_order merge note 是 warn-only**：相邻 segment ms 级时间轴回溯由 whisper.cpp
  本身产生，不影响下游消费，写到 manifest.warnings 仅供审计
- **bundle 不含 whisper.cpp 模型**：License 边界 + 体积考虑，`voxkit fetch-bundle` 只拉
  pyannote 4 个 repo + silero VAD，whisper 模型走 brew/HF 单独装
- **失败时 `work/` 强制保留**：`--no-keep-work` 仅在成功收尾时生效，失败的时候审计需求最强
- **PID lock 写在 `manifest.json`**：同一 workdir 并发 voxkit 进程会被第二个 fast-fail；
  PID 已死则 warning 接管

## 进一步阅读

- 实现入口：`src/voxkit/core/transcribe_pipeline.py::run_pipeline`
- Pydantic schema：`src/voxkit/io/schema.py`
- Remixr 适配器：`src/voxkit/io/remixr_adapter.py`
- 反幻觉过滤器：`src/voxkit/core/hallucination_filter.py`
- 双模式 segmenter：`src/voxkit/core/segmenter.py`
- 长音频合并：`src/voxkit/core/asr_merge.py`
