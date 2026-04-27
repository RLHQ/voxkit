# voxsplit

独立 speaker diarization CLI，基于 pyannote.audio 4.x。

## 设计目标

- 输入音频或视频，输出 speaker segments JSON（schemaVersion=1）
- 把 5 类常踩坑固化为 `voxsplit doctor` 自检
- 主进程零重依赖：pyannote / torch 走独立 venv（lazy install）

## 安装

```bash
# 推荐
pipx install voxsplit  # 0.1 暂未发版，使用本地源码
# 或本地开发
uv venv && uv pip install -e ".[worker,dev]"
```

## 前置（首次必做）

1. 在 https://huggingface.co/settings/tokens 创建 token，写入 `~/.cache/huggingface/token`
2. 在以下 4 个 gated 模型页点 **Accept**：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
3. macOS 用户：确保安装了 `ffmpeg-full`（`brew install ffmpeg-full`）

## 用法

```bash
voxsplit doctor                                 # 6 项自检
voxsplit setup                                  # 显式触发依赖与模型预下载
voxsplit diarize input.mp4 -o out.json          # 主命令（自动从视频抽音频）
voxsplit align transcript.raw.json out.json -o aligned.srt
```

## 5 个常踩坑（已固化）

| 坑 | 由 voxsplit 自动处理 |
|---|---|
| HF token 缺失 / 4 个 gated 未 accept | doctor + diarize 入口 fast-fail |
| torchcodec 找不到 ffmpeg lib | 自动 export DYLD_LIBRARY_PATH=/opt/homebrew/lib |
| pyannote 3.x → 4.x API 变更 | 多版本兼容 try/fallback |
| HEAD metadata 200 不代表 accept | 检查实际权重文件 HEAD |
| ffmpeg 版本不兼容 | ffprobe 探测 major 版本，区间外告警 |

详细背景见 `docs/known-pitfalls.md`（TBD）。
