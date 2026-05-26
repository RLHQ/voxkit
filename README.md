# voxkit

把音频/视频一次性转成可审计、可恢复、可直接进产品工作流的结构化语音数据：
转录、字幕、说话人、语义重切、Remixr 兼容 JSON。

底层基于 [whisper.cpp](https://github.com/ggerganov/whisper.cpp)（ASR）+
[pyannote.audio](https://github.com/pyannote/pyannote-audio)（speaker diarization）。

## 几句话说清楚

**这个是干什么的？**
voxkit 是一个本地优先的「语音转结构化数据」CLI，把音频/视频处理成 transcript、SRT/VTT 字幕、说话人标签和可供下游系统直接读取的 JSON。

**有什么优势？**
它把 whisper.cpp 的高速本地转录、pyannote 的说话人切分、长音频 checkpoint、反幻觉过滤、语义字幕重切和 Remixr 兼容输出封装成一条可重复、可审计的流水线。

**解决什么痛点？**
它解决长音视频转录里最麻烦的工程问题：静音幻觉、长文件中断重跑、字幕切得碎/闪、说话人标签难对齐、SRT/JSON 反复转换，以及 ASR 结果很难安全接入产品数据结构。

## 故事

voxkit 把音频/视频里的语音处理成结构化、可消费的数据：转录、说话人切分、字幕、对齐——
一个 CLI 搞定，输出格式与 [Remixr (CutFlow)](https://github.com/3Craft/CutFlow) 直接兼容。

> **与 voxsplit 的关系**：voxkit 是 voxsplit 的 v0.3.0 重命名扩展。voxsplit 0.2.x 仅做 diarization；
> v0.3.0 起加入 `transcribe`，定位升格为 toolkit，故改名 voxkit。
> 旧用户的迁移见 [v0.2.x → v0.3.0 迁移](#v02x--v030-迁移)。

## Quick start — 一份双语字幕完整流水线

直接复制粘贴；4 条命令把 mp4 / mp3 跑成英文校对 + 中文翻译 + 质量报告。
需要 `DEEPSEEK_API_KEY` 环境变量（或换其它 provider）。

```bash
voxkit transcribe video.mp4 --workdir vk/ --language en --resegment=semantic
voxkit proofread vk/ --language en --glossary glossary.json
voxkit translate vk/ --target-language zh --glossary glossary.json
voxkit quality vk/                                              # 质量报告
```

每步产物：

| 文件 | 来自 | 内容 |
|---|---|---|
| `vk/subtitles.cues.json` | `transcribe --resegment=semantic` | 语义切分后的 cue 流（id + start/end + text） |
| `vk/subtitles.srt` / `.vtt` | `transcribe` | 英文字幕 |
| `vk/subtitles.proofread.json` | `proofread` | LLM 校对后的 cue（含 corrected / risk / needsHumanReview） |
| `vk/subtitles.zh.json` / `.zh.srt` / `.zh.vtt` | `translate` | 中文翻译产物 |
| `vk/quality.report.json` | `quality` | 字幕覆盖率 / CPS / 标点 / over-char 等指标聚合 |

**常用增量命令**：

```bash
voxkit needs-review vk/                                         # 列待人工 review 的 cue
voxkit reseg vk/                                                # 双 pass：用 corrected 的标点重切
voxkit translate vk/ --target-language zh --dry-run             # 跑前算 token / cost，不调 LLM
voxkit translate vk/ --target-language zh --render-only \
  --speaker-prefix never                                        # 改字幕格式但不重花 LLM token
voxkit review confirm vk/                                       # 锁 proofread 为 reviewed
```

完整版本变化和新 flag 见 [CHANGELOG.md](./CHANGELOG.md)。安装与依赖见下面 "安装" 节。

## 子命令

| 命令 | 用途 |
|---|---|
| `voxkit doctor` | 按目标自检依赖；支持 `--profile transcribe` / `--profile diarize` |
| `voxkit setup` | 显式创建 worker venv + 安装 pyannote.audio |
| `voxkit transcribe` | ★ whisper.cpp 转录 → `transcript.raw.json` + SRT/VTT；可选说话人注入与语义字幕重切 |
| `voxkit diarize` | pyannote 说话人切分 → `DiarizationOutput` JSON |
| `voxkit align` | transcript + diarization → 带 Speaker N 的 SRT |
| `voxkit proofread` | LLM 校对 `subtitles.cues.json` → `subtitles.proofread.json` |
| `voxkit translate` | LLM 翻译 → `subtitles.<lang>.json` + SRT/VTT |
| `voxkit reseg` | 用 corrected 标点再切一次（双 pass） |
| `voxkit quality` | 字幕质量报告（cue 覆盖 / CPS / 标点等） |
| `voxkit needs-review` | 列出 proofread / translate artifact 中需要人工复核的 cue |
| `voxkit review` | confirm / lock subtitle artifact 生命周期 |
| `voxkit eval` | 对照人类金标 SRT 评估 reseg 质量 |
| `voxkit build-bundle` | 打包模型为 tar.gz（4 个 pyannote HF repo + silero VAD） |
| `voxkit fetch-bundle` | 从 GitHub Release 拉模型 bundle |

## 安装

### 5 分钟跑通：只转录 + 出字幕

如果你第一次接触 voxkit，建议先跑通最小路径：**音频/视频 → transcript + SRT/VTT**。
这条路径不需要 Hugging Face token，也不需要 pyannote。

```bash
# 1. 安装 CLI（本地源码）
uv tool install --editable .

# 2. 安装本地转录依赖
brew install whisper-cpp ffmpeg-full

# 3. 下载默认 whisper.cpp 模型
huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin \
  --local-dir ~/.cache/voxkit/models

# 4. 只检查转录链路，避免被 diarization 依赖干扰
voxkit doctor --profile transcribe

# 5. 跑你的第一个文件
voxkit transcribe input.mp4 --workdir out/ --language auto
```

成功后看 `out/subtitles.srt` 和 `out/transcript.raw.json`。这就是第一次“点亮”。

> `huggingface-cli` 来自 `huggingface_hub`；如果本机没有，可先运行
> `uv tool install huggingface_hub`，或用你自己的方式安装 Hugging Face CLI。

### 想要说话人标签

如果你需要 `Speaker 1/2/...`，再准备 pyannote 模型。3Craft 内部或有 bundle 权限的用户推荐：

```bash
gh auth login
voxkit fetch-bundle
voxkit doctor --profile diarize

voxkit transcribe input.mp4 --workdir out/ \
  --with-diarization \
  --resegment=semantic
```

没有 bundle 权限时，走 Hugging Face token + gated model accept，见
[diarize 上手](#上手--diarizev02x-旧功能保留)。

### 开发者安装

```bash
uv venv
uv pip install -e ".[dev]"

# 需要跑 diarize / with-diarization 时再装 worker extra
uv pip install -e ".[worker,dev]"
voxkit setup
```

`voxkit doctor` 默认检查全部依赖；第一次跑通时更推荐按目标检查：

```bash
voxkit doctor --profile transcribe   # whisper.cpp + ffmpeg + ASR 模型
voxkit doctor --profile diarize      # pyannote / HF bundle / ffmpeg
voxkit doctor                        # 全量检查
```

## 上手 — `transcribe`

### 基本用法

```bash
voxkit transcribe input.mp4 --workdir out/ --language en --json-events
```

一站式产出带说话人标签、且更适合播放器展示的字幕：

```bash
voxkit transcribe input.mp4 --workdir out/ \
  --language en \
  --with-diarization \
  --resegment=semantic
```

- 位置参数 `<input>`：单个音频或视频文件（不是目录；批处理用 shell 组合）
- `--workdir` 必填，所有产物落在这一棵子树下，是数据正交的边界
- `--json-events` 把 stderr 切到 NDJSON 事件流（机器消费），同时镜像写到 `events.ndjson`
- `--with-diarization` 会在 ASR 后追加 pyannote 说话人切分，把 `Speaker 1/2/...` 写进 `transcript.raw.json` 与字幕
- `--resegment=semantic` 只重切渲染层字幕，保留 ASR ground truth，同时写出机读的 `subtitles.cues.json`

### 工作目录布局（数据正交、过程产物可审计）

```
out/
├── manifest.json              # 运行元数据 + perChunk 统计 + warnings
├── transcript.raw.json        # Remixr 兼容（drop-in）
├── transcript.voxkit.json     # voxkit 原生丰富格式
├── subtitles.srt
├── subtitles.vtt
├── subtitles.cues.json        # --resegment=semantic 时输出，机读字幕 cue 流
├── events.ndjson              # 全流程 NDJSON 事件流
└── work/                      # 中间产物，--no-keep-work 成功时清理
    ├── input.16khz.mono.wav   # ffmpeg 归一化主音频
    ├── diarization.json       # --with-diarization 时输出，说话人切分审计文件
    ├── chunks/
    │   ├── chunk_000.wav
    │   ├── chunk_000.json     # whisper.cpp --output-json-full 原始
    │   └── …
    └── merge.json             # 每 chunk 保留/丢弃 segment 的合并报告
```

幂等契约：`transcript.raw.json` 用 **exclusive write (`open(path, "x")`)**，已存在则 fail-fast；
顶层 `transcript.voxkit.json` / `subtitles.*` / `manifest.json` 每次重写；
`work/chunks/chunk_NNN.json` 是 append-only checkpoint，`--resume`（默认）下命中即跳过。

### 实测性能（64 分钟英文播客，Apple M-series）

```
端到端 wall clock: 181.7 s   (3:02)
RTF:               0.0476
分块:              7 chunks × 600 s + 5 s overlap
最终 segments:     909
hallucinations:    0          (英文素材不触发中文黑名单)
```

### CLI 标志

| 标志 | 默认 | 说明 |
|---|---|---|
| `<input>` | (必填) | 音频/视频文件 |
| `--workdir DIR` | (必填) | 数据正交工作目录 |
| `--model NAME` | `large-v3-turbo` | whisper.cpp 模型别名或 `.bin` 绝对路径 |
| `--language CODE` | `auto` | `en` / `zh` / 任意 ISO code |
| `--word-timestamps` / `--no-word-timestamps` | on | 词级时间戳；CJK 自动忽略 |
| `--vad` / `--no-vad` | on | 启用 silero VAD（缺模型时 warn-once 降级） |
| `--logprob-thold FLOAT` | `-0.8` | logprob 阈值（whisper 默认 `-1.0`，本工具收紧） |
| `--source-id ID` | input 文件名 stem | Remixr 的 `sourceId` |
| `--keep-work` / `--no-keep-work` | keep | 失败时强制 keep，与标志无关 |
| `--json-events` | off | stderr NDJSON + 镜像 `events.ndjson` |
| `--timeout MS` | dynamic | 覆写 `max(30 min, duration*0.3)*1000` |
| `--chunk-threshold-secs FLOAT` | env / built-in | 超过该时长才分块，便于测试短音频 chunk |
| `--chunk-secs FLOAT` | env / built-in | 每个转录 chunk 的目标时长 |
| `--chunk-overlap-secs FLOAT` | env / built-in | 相邻 chunk 的重叠时长，必须小于 `--chunk-secs` |
| `--whisper-bin PATH` | 自动发现 | 覆盖 `which whisper-cli` |
| `--vad-model PATH` | env / brew | silero VAD bin 路径 |
| `--resume` / `--no-resume` | on | 命中 `chunk_NNN.json` 即跳过 |
| `--force` | off | 等同 `--no-resume`，先清 `work/` 再跑 |
| `--blocklist PATH` | bundled JSON | 覆盖默认中文幻觉黑名单 |
| `--emit-srt` / `--no-emit-srt` | on | 是否输出 SRT |
| `--emit-vtt` / `--no-emit-vtt` | on | 是否输出 VTT |
| `--with-diarization` / `--no-with-diarization` | off | ASR 后追加 pyannote diarization，把真实 speaker 标签注入 raw JSON 与字幕 |
| `--speaker-labels ranked\|raw` | `ranked` | `ranked` 输出 `Speaker 1/2/...`；`raw` 保留 `SPEAKER_00` 等 pyannote 原始标签 |
| `--resegment none\|semantic` | `none` | 语义字幕重切；仅影响 SRT/VTT/`subtitles.cues.json`，不改 `transcript.raw.json` |

### 反幻觉策略（来自 Remixr 实战沉淀）

1. **VAD**（`--vad --vad-model X`）— silero v5.1.2 屏蔽静音段，砍掉 whisper 在静默处的胡言
2. **`--max-context 0`** — 切断 chunk 间 KV-cache 串扰，避免一段错的把后面带歪
3. **`--logprob-thold -0.8`** — 低置信度段过滤（whisper 默认 `-1.0` 太松）
4. **中文水印黑名单** — JSON 配置 7 个 watermark prefix + 19 个 standalone 短语 +
   ghost CJK loop（≥6 字子串重复 ≥2 次的结构性丢弃）

```bash
# 用自定义黑名单
voxkit transcribe input.mp4 --workdir out/ \
  --blocklist /path/to/custom-blocklist.json
```

被丢弃的条目会写到 `out/work/hallucinations.log`（NDJSON），可审计。

### 说话人注入与语义字幕重切

`--with-diarization` 把原本分开的 ASR 和 diarization 串成一条流水线：

```bash
voxkit transcribe interview.mp4 --workdir out/ \
  --with-diarization \
  --speaker-labels ranked
```

产物变化：

- `transcript.raw.json` 的 `segments[].speaker` 从占位的 `Speaker A` 变成 `Speaker 1/2/...`
- `subtitles.srt` / `subtitles.vtt` 使用真实 speaker 前缀
- `out/work/diarization.json` 保留 pyannote 原始切分结果，方便审计
- `manifest.json` 记录 diarization 模型、设备、耗时、说话人数和 speaker label 策略

`--resegment=semantic` 专门解决字幕显示问题：英文按句子/子句边界重切，长 cue 拆短，过短 cue 合并；
CJK 输入走 phrase-aware 打包，按标点、speaker、字符数、时长和 CPS 约束整理字幕，必要时只在单个超长 phrase 内做字符时间插值。

```bash
voxkit transcribe podcast.mp4 --workdir out/ \
  --resegment=semantic
```

语义重切只影响渲染层产物：`subtitles.srt`、`subtitles.vtt` 和 `subtitles.cues.json`。
`transcript.raw.json`（Remixr 适配视图）与 `transcript.voxkit.json`（voxkit 主产物）保持 ASR
层语义，避免把播放器字幕决策反向污染 transcript。

### Resume / Force / 长视频 checkpoint

```bash
# 默认 resume：第二次跑同一个 workdir，命中 chunk_NNN.json 即跳过
voxkit transcribe long.mp4 --workdir out/

# --force：清空 work/ 重跑（等同 --no-resume）
voxkit transcribe long.mp4 --workdir out/ --force
```

> 严格幂等：`transcript.raw.json` 已存在时直接报错，调用方需显式换 `--workdir` 或先删旧文件。
> 与 Remixr 的「raw 不可变」契约一致。

## 上手 — `diarize`（v0.2.x 旧功能保留）

### 推荐：从自托管 bundle 拉模型（开箱即用）

适合 3Craft 内部机器或 voxkit 已发布 bundle 的场景：

```bash
gh auth login                       # 一次性，对 3Craft/voxkit 有读权限即可
voxkit fetch-bundle                 # 拉 latest release 中的模型 bundle
voxkit doctor --profile diarize     # 全绿（离线模式）
voxkit diarize input.mp4 -o out.json
```

无需 HF 账号 / token / 4 个 Accept 点击 / 500 MB 模型下载。

### 备选：从 Hugging Face 上游拉模型

如果没 bundle 可拉（外部用户 / 全新 repo）：

1. 在 https://huggingface.co/settings/tokens 创建 token，写入 `~/.cache/huggingface/token`
2. 在以下 3 个 gated 模型页点 **Accept**（wespeaker 已开放，无需 accept）：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. macOS 用户：`brew install ffmpeg-full`

### diarize 用法速查

```bash
voxkit doctor --profile diarize                        # 只检查 diarize 相关依赖
voxkit setup                                           # 显式触发 venv + pyannote 安装
voxkit diarize input.mp4 -o out.json                   # 主命令（自动从视频抽音频）
voxkit diarize a.wav --model community-1 -o out.json   # 指定模型
voxkit align transcript.raw.json out.json -o aligned.srt
voxkit build-bundle --bundle-version v1 --output-dir ./release-staging
voxkit fetch-bundle --release v1                       # 默认拉 latest
```

## 模型选择

### diarize：`--model`

| 模型 | 默认 | License | 适合场景 |
|---|---|---|---|
| `sd-3.1` | yes | MIT | 播客 / 谈话类 / 1-on-1 访谈（清晰录音、3 人以内） |
| `community-1` |  | CC-BY-4.0 | 多人会议 / 远场 / 嘈杂 / 重叠语音多 |

依据：[官方 benchmark](https://huggingface.co/pyannote/speaker-diarization-community-1#benchmark)
上 `community-1` 在 AliMeeting / AMI / Ego4D 等会议数据集上比 `sd-3.1` 低 2-5 个 DER 点；
但电视谈话类（REPERE）反而 +1.0 DER。本喵实测在 1 小时英文播客上两者桶级一致性
97.78% vs 97.65%，几乎打平 —— 默认保持 `sd-3.1`，会议场景显式 `--model community-1` 即可。
`community-1` 段切得更细（+43% segments），对快速插话场景也更敏感。

### transcribe：`--model`

默认 `large-v3-turbo`（whisper.cpp 官方推荐的速度/质量平衡点；本仓库实测 RTF ≈ 0.05 on Apple Silicon）。
也接受任意 whisper.cpp `.bin` 绝对路径。

## 5 个 diarize 已固化坑

| 坑 | 由 voxkit 自动处理 |
|---|---|
| HF token 缺失 / 4 个 gated 未 accept | doctor + diarize 入口 fast-fail |
| torchcodec 找不到 ffmpeg lib | 自动 export `DYLD_LIBRARY_PATH=/opt/homebrew/lib` |
| pyannote 3.x → 4.x API 变更 | 多版本兼容 try/fallback |
| HEAD metadata 200 不代表 accept | 检查实际权重文件 HEAD |
| ffmpeg 版本不兼容 | ffprobe 探测 major 版本，区间外告警 |

## v0.2.x → v0.3.0 迁移

```bash
# 旧用户的 venv + cache 路径需要手工 mv 一次
mv ~/.local/share/voxsplit ~/.local/share/voxkit
mv ~/.cache/voxsplit       ~/.cache/voxkit

# 或者直接重装（在新路径创建 venv）
voxkit setup
```

兼容契约：

- `DiarizationOutput.schemaVersion` 仍为 `"1"` —— 已有 JSON 消费者无需改动
- 新增 `TranscriptionOutput.schemaVersion = "1"`（独立计数器）
- GitHub bundle release repo：v0.2.x 历史归档在 `3Craft/voxsplit`；
  v0.3.0 起新 bundle 发布到 `3Craft/voxkit`，`fetch-bundle` 默认指向新 repo

## Remixr 集成

voxkit 的 `transcript.raw.json` 与 Remixr (CutFlow) 的 Zod schema
（`packages/shared/src/types/transcript.ts`）字节级兼容：

```bash
voxkit transcribe input.mp4 --workdir out/ --source-id src_xxxx
cp out/transcript.raw.json \
   /path/to/Remixr/storage/projects/<projectId>/sources/<sourceId>/transcript.raw.json
```

`_metadata` 字段（`voxkitVersion` / `asrBackend` / `asrModel` / `rtf` / `perChunk` / `warnings` 等）
Remixr 会忽略未知字段，对 voxkit 自己审计有用。

如果下游需要播放器级字幕 cue，不建议反解 SRT；直接读取 `subtitles.cues.json`：

```bash
voxkit transcribe input.mp4 --workdir out/ \
  --with-diarization \
  --resegment=semantic

cp out/subtitles.cues.json \
   /path/to/Remixr/storage/projects/<projectId>/sources/<sourceId>/subtitles.cues.json
```

更深入的 transcribe 文档（数据流图、字段一览、调试技巧）见
[`docs/transcribe.md`](docs/transcribe.md)。

## Roadmap

- `vox-asr` provider（火山引擎云端 ASR），与 whisper.cpp 在 `whisper_exec.py`
  内部接口隔离
- Remixr 端切流到 `voxkit-adapter`，删除 `services/whisper.ts` 等约 1400 行 TS 代码
- VAD/静音对齐 chunk 边界，减少固定网格切到句中或词中的概率
- 更多 subtitle resegment 参数外露；当前 chunk 阈值、时长和 overlap 已支持 CLI 覆盖
