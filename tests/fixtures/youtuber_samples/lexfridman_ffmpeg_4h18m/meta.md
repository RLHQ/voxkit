# lexfridman_ffmpeg_4h18m

**频道**：Lex Fridman Podcast（@lexfridman）
**视频**：Lex Fridman Podcast #496 – FFmpeg: The Incredible Technology Behind Video
**URL**：<https://www.youtube.com/watch?v=nepKKz-MzFM>
**视频时长**：15502 秒（4:18:22）
**音频**：按需 `./fetch.sh` 拉取完整 4h18m（约 496MB，不入仓）
**字幕**：全程 en（覆盖整 4h18m）
**下载日期**：2026-05-13

## 字幕轨道

| 文件 | 来源轨道 | 说明 |
|---|---|---|
| `sub.en.srt` | en | 官方人工字幕，**带说话人切换标记**：每次说话人变化时新条目以 `- ` 开头 |

## 为什么入选

- **长 podcast 的 diarize/align 压力测试**：完整字幕覆盖 4h18m 多人对话，字幕里的 `- ` 标记可作为 diarize 输出的对照（"哪一秒切换说话人"）。
- **保持完整音频**：用 voxkit 跑前先 `./fetch.sh` 拉完整 WAV；如只想测前 N 分钟，用 ffmpeg `-t` 自行切片，不在 fixture 中冻结某个截段（避免比较口径不一致）。

## voxkit 用途映射

- `diarize + align` → 字幕里的 `- ` 切换点 vs voxkit diarize 输出的说话人边界做 IoU 对照；
- `transcribe`（en，长 podcast）→ 完整 ASR 与 sub.en.srt 做 WER；
- `proofread`（en，技术访谈）→ FFmpeg 相关技术术语写法基准。

## 音频规格

WAV / 16000 Hz / mono / 15502 秒（完整未截）。按需用同目录 `./fetch.sh` 拉取。

## 切片示例（按需）

```bash
# 取前 5 分钟做快速回归
ffmpeg -i audio.wav -t 300 -c copy audio_0-5min.wav

# 取 1h00m–1h05m 这一段
ffmpeg -i audio.wav -ss 3600 -t 300 -c copy audio_1h-1h05m.wav
```
