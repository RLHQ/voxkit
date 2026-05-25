# kurzgesagt_germany_14min

**频道**：Kurzgesagt – In a Nutshell（@kurzgesagt）
**视频**：GERMANY IS OVER
**URL**：<https://www.youtube.com/watch?v=n-gYFcVx-8Y>
**时长**：845 秒（14:05）
**音频**：按需 `./fetch.sh` 拉取（约 27MB，不入仓）
**字幕**：en（入仓）
**下载日期**：2026-05-13

## 字幕轨道

| 文件 | 来源轨道 | 说明 |
|---|---|---|
| `sub.en.srt` | en | 官方人工字幕，标点完整，断句尊重旁白气口 |

> 探测时**只发现 en 一轨手动字幕**。早年传说 Kurzgesagt "提供官方多语 CC" 的说法当前不成立——可能是历史上某些视频有，但近期视频没有。如需多语对照样本，看 `3blue1brown_logarithm_subs_only/`。

## 为什么入选

- **英文短篇高密度科普标杆**：14 分钟视频里政治、经济、人口学术语密集，是 voxkit `proofread` 在科普领域的术语一致性参考。
- **音视频质量稳定**：旁白配音录音棚级别，ASR 干净，适合做 `transcribe` 的基线。
- **VTT 中保留 `\h`（硬空格）**：原始 VTT 转 SRT 时未规范化，是验证 voxkit 字幕清洗管线的边角样本。

## voxkit 用途映射

- `transcribe`（en）→ 与 sub.en.srt 做 WER 对照；
- `reseg`（en）→ 与人工断句对照；
- `proofread`（en，科普术语）→ 术语写法基准；
- `translate` en→zh → 用本样本测中文产出，跨样本对照 xnzxnz 的英→中风格。

## 音频规格

WAV / 16000 Hz / mono / 844.0 秒。按需用同目录 `./fetch.sh` 拉取。
