# voxkit

「语音 → 结构化数据」工具集。基于 [whisper.cpp](https://github.com/ggerganov/whisper.cpp)（ASR）+
[pyannote.audio](https://github.com/pyannote/pyannote-audio)（speaker diarization）。

## 故事

voxkit 把音频/视频里的语音处理成结构化、可消费的数据：转录、说话人切分、字幕、对齐——
一个 CLI 搞定，输出格式与 [Remixr (CutFlow)](https://github.com/3Craft/CutFlow) 直接兼容。

> **与 voxsplit 的关系**：voxkit 是 voxsplit 的 v0.3.0 重命名扩展。voxsplit 0.2.x 仅做 diarization；
> v0.3.0 起加入 `transcribe`，定位升格为 toolkit，故改名 voxkit。
> 旧用户的迁移见 [v0.2.x → v0.3.0 迁移](#v02x--v030-迁移)。

## 子命令

| 命令 | 用途 |
|---|---|
| `voxkit doctor` | 自检 10 项依赖（uv / Python / models offline / HF token / 4 gated / ffmpeg / venv / whisper-cli / whisper model / VAD model） |
| `voxkit setup` | 显式创建 venv + 装 pyannote.audio + 预下载模型 |
| `voxkit transcribe` | ★ whisper.cpp 转录 → `transcript.raw.json` + SRT/VTT |
| `voxkit diarize` | pyannote 说话人切分 → `DiarizationOutput` JSON |
| `voxkit align` | transcript + diarization → 带 Speaker N 的 SRT |
| `voxkit build-bundle` | 打包模型为 tar.gz（4 个 pyannote HF repo + silero VAD） |
| `voxkit fetch-bundle` | 从 GitHub Release 拉模型 bundle |

## 安装

```bash
# 本地开发
uv venv && uv pip install -e ".[worker,dev]"

# 或 pipx 全局（v0.3.0 暂未发 PyPI，使用本地源码路径）
pipx install /path/to/voxkit
```

`transcribe` 子命令额外需要本机的 whisper.cpp 二进制 + 模型：

```bash
brew install whisper-cpp ffmpeg-full
huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin \
  --local-dir ~/.cache/voxkit/models
```

`voxkit doctor` 会一次性把上述 10 项依赖全部探测一遍并给出修复提示，缺啥补啥即可。

## 上手 — `transcribe`（v0.3.0 新增）

### 基本用法

```bash
voxkit transcribe input.mp4 --workdir out/ --language en --json-events
```

- 位置参数 `<input>`：单个音频或视频文件（不是目录；批处理用 shell 组合）
- `--workdir` 必填，所有产物落在这一棵子树下，是数据正交的边界
- `--json-events` 把 stderr 切到 NDJSON 事件流（机器消费），同时镜像写到 `events.ndjson`

### 工作目录布局（数据正交、过程产物可审计）

```
out/
├── manifest.json              # 运行元数据 + perChunk 统计 + warnings
├── transcript.raw.json        # Remixr 兼容（drop-in）
├── transcript.voxkit.json     # voxkit 原生丰富格式
├── subtitles.srt
├── subtitles.vtt
├── events.ndjson              # 全流程 NDJSON 事件流
└── work/                      # 中间产物，--no-keep-work 成功时清理
    ├── input.16khz.mono.wav   # ffmpeg 归一化主音频
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
| `--whisper-bin PATH` | 自动发现 | 覆盖 `which whisper-cli` |
| `--vad-model PATH` | env / brew | silero VAD bin 路径 |
| `--resume` / `--no-resume` | on | 命中 `chunk_NNN.json` 即跳过 |
| `--force` | off | 等同 `--no-resume`，先清 `work/` 再跑 |
| `--blocklist PATH` | bundled JSON | 覆盖默认中文幻觉黑名单 |
| `--emit-srt` / `--no-emit-srt` | on | 是否输出 SRT |
| `--emit-vtt` / `--no-emit-vtt` | on | 是否输出 VTT |

### 反幻觉策略（基于 Remixr 6 个月血泪）

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
voxkit doctor                       # 全绿（离线模式）
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
voxkit doctor                                          # 10 项自检（自动识别离线/在线模式）
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

更深入的 transcribe 文档（数据流图、字段一览、调试技巧）见
[`docs/transcribe.md`](docs/transcribe.md)。

## Roadmap

- **Phase 2**：`voxkit transcribe --with-diarization` 一站式产出带 `Speaker N` 的 transcript
  （链接 transcribe + diarize + align）
- **Phase 2**：`vox-asr` provider（火山引擎云端 ASR），与 whisper.cpp 在 `whisper_exec.py`
  内部接口隔离
- **Phase 2**：Remixr 端切流到 `voxkit-adapter`，删除 `services/whisper.ts` 等约 1400 行 TS 代码
