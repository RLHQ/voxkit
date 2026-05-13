# lexfridman_ffmpeg_5min

**频道**：Lex Fridman Podcast（@lexfridman）
**视频**：Lex Fridman Podcast #496 – FFmpeg: The Incredible Technology Behind Video
**URL**：<https://www.youtube.com/watch?v=nepKKz-MzFM>
**原视频时长**：8884 秒（2:28:04）
**音频截取**：前 300 秒（5:00）
**字幕**：全程 en（覆盖整 2h28m）
**下载日期**：2026-05-13

## 字幕轨道

| 文件 | 来源轨道 | 说明 |
|---|---|---|
| `sub.en.srt` | en | 官方人工字幕，**带说话人切换标记**：每次说话人变化时新条目以 `- ` 开头 |

## 为什么入选

- **长 podcast 的 diarize/align 压力测试**：完整字幕覆盖 2h28m 多人对话，字幕里的 `- ` 标记可作为 diarize 输出的对照（"哪一秒切换说话人"）。
- **音频前 5min 即可验证**：voxkit `diarize` 在前几分钟的 turn-taking 切换密度足够测稳定性；如需跑完整长度，按 meta 重新拉完整音频。

## voxkit 用途映射

- `diarize + align` → 字幕里的 `- ` 切换点 vs voxkit diarize 输出的说话人边界做 IoU 对照；
- `transcribe`（en，长 podcast）→ 前 5min ASR 与 sub.en.srt 前 5min 切片做 WER；
- `proofread`（en，技术访谈）→ FFmpeg 相关技术术语写法基准。

## 音频规格

WAV / 16000 Hz / mono / 300.0 秒（原视频 0:00–5:00 截段）

## 重新拉完整音频

```bash
yt-dlp -x --audio-format wav \
  --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
  -o audio_full.%(ext)s \
  "https://youtube.com/watch?v=nepKKz-MzFM"
```

完整音频约 270 MB（2h28m WAV 16kHz mono），不建议入仓。
