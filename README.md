# voxsplit

独立 speaker diarization CLI，基于 pyannote.audio 4.x。

## 设计目标

- 输入音频或视频，输出 speaker segments JSON（schemaVersion=1）
- 把 5 类常踩坑固化为 `voxsplit doctor` 自检
- 主进程零重依赖：pyannote / torch 走独立 venv（lazy install）

## 安装

```bash
# 本地开发
uv venv && uv pip install -e ".[worker,dev]"

# 或 pipx 全局（v0.2.0 暂未发 PyPI，使用本地源码路径）
pipx install /path/to/voxsplit
```

## 上手（两条路径）

### 推荐：从自托管 bundle 拉模型（开箱即用）

适合 3Craft 内部机器或 voxsplit 已发布 bundle 的场景：

```bash
gh auth login                       # 一次性，对 3Craft/voxsplit 有读权限即可
voxsplit fetch-bundle               # 拉 latest release 中的模型 bundle
voxsplit doctor                     # ✅ 全绿（离线模式）
voxsplit diarize input.mp4 -o out.json
```

无需 HF 账号 / token / 4 个 Accept 点击 / 500MB 模型下载。

### 备选：从 Hugging Face 上游拉模型

如果没 bundle 可拉（外部用户 / 全新 repo）：

1. 在 https://huggingface.co/settings/tokens 创建 token，写入 `~/.cache/huggingface/token`
2. 在以下 3 个 gated 模型页点 **Accept**（wespeaker 已开放，无需 accept）：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. macOS 用户：`brew install ffmpeg-full`

## 用法

```bash
voxsplit doctor                                 # 7 项自检（自动识别离线/在线模式）
voxsplit setup                                  # 显式触发 venv + pyannote 安装
voxsplit diarize input.mp4 -o out.json          # 主命令（自动从视频抽音频）
voxsplit diarize a.wav --model community-1 -o out.json  # 指定模型
voxsplit align transcript.raw.json out.json -o aligned.srt
voxsplit build-bundle --bundle-version v1 --output-dir ./release-staging
voxsplit fetch-bundle --release v1              # 默认拉 latest
```

## 模型选择（`--model`）

| 模型 | 默认 | License | 适合场景 |
|---|---|---|---|
| `sd-3.1` | ✅ | MIT | 播客 / 谈话类 / 1-on-1 访谈（清晰录音、3 人以内） |
| `community-1` |  | CC-BY-4.0 | 多人会议 / 远场 / 嘈杂 / 重叠语音多 |

依据：[官方 benchmark](https://huggingface.co/pyannote/speaker-diarization-community-1#benchmark) 上 `community-1` 在 AliMeeting / AMI / Ego4D 等会议数据集上比 `sd-3.1` 低 2-5 个 DER 点；但电视谈话类（REPERE）反而 +1.0 DER。本喵实测在 1 小时英文播客上两者桶级一致性 97.78% vs 97.65%，几乎打平 —— 默认保持 `sd-3.1`，会议场景显式 `--model community-1` 即可。`community-1` 段切得更细（+43% segments），对快速插话场景也更敏感。

## 5 个常踩坑（已固化）

| 坑 | 由 voxsplit 自动处理 |
|---|---|
| HF token 缺失 / 4 个 gated 未 accept | doctor + diarize 入口 fast-fail |
| torchcodec 找不到 ffmpeg lib | 自动 export DYLD_LIBRARY_PATH=/opt/homebrew/lib |
| pyannote 3.x → 4.x API 变更 | 多版本兼容 try/fallback |
| HEAD metadata 200 不代表 accept | 检查实际权重文件 HEAD |
| ffmpeg 版本不兼容 | ffprobe 探测 major 版本，区间外告警 |

详细背景见 `docs/known-pitfalls.md`（TBD）。
