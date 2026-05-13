# 3blue1brown_logarithm_subs_only

**频道**：3Blue1Brown（@3blue1brown）
**视频**：How (and why) to take a logarithm of an image
**URL**：<https://www.youtube.com/watch?v=ldxFjLJ3rVY>
**时长**：2692 秒（44:52）
**下载日期**：2026-05-13

> **本目录不含音频**：原视频 44 分钟，16kHz mono WAV 约 80MB，按 fixture 体积纪律不入仓。如需音频跑 `./fetch.sh`（基于实拉验证过的 yt-dlp 命令，落到 `audio.wav`）。

## 字幕轨道（11 种语言）

| 文件 | 来源轨道 | 性质 | 条目数 |
|---|---|---|---|
| `sub.en.srt` | en | **作者人工字幕**（Grant Sanderson 原文） | 722 |
| `sub.ar.srt` | ar | AI 翻译 + criblate.com 社区校对 | 578 |
| `sub.de.srt` | de | 同上 | 816 |
| `sub.es.srt` | es | 同上 | 763 |
| `sub.fr.srt` | fr | 同上 | 782 |
| `sub.hi.srt` | hi | 同上 | 722 |
| `sub.hr.srt` | hr | 同上 | 687 |
| `sub.ko.srt` | ko | 同上 | 407 |
| `sub.pt-BR.srt` | pt-BR | 同上 | 757 |
| `sub.ru.srt` | ru | 同上 | 749 |
| `sub.vi.srt` | vi | 同上 | 719 |

> ⚠️ **两条关键事实**：
> 1. 除 en 外的 10 种语言**不是译者人工翻译**，是 AI 翻译再经 criblate.com 社区平台收集校对意见。3Blue1Brown 这套流程恰好就是 voxkit translate+proofread 想要服务的客户场景——"机器翻译 + 人工校对"。
> 2. 11 个语种**各自独立切分**，条目数差异显著（ko 407 vs de 816），**不是逐条对齐的平行语料**。每条 cue 的时间码和切分边界都是各语种译者自行决定的，**做 BLEU/CER 这类逐句对照前必须先做时间窗口对齐**。

## 为什么入选

- **多语风格对照**：11 种语言来自同一原片但各自切分；适合做译文**风格、术语写法、断句策略**的横向对照，不适合做严格的逐条 BLEU 对齐。
- **客户场景对照**：AI 翻译 + 校对的工作流就是 voxkit translate→proofread 的目标产出形态；可以把 voxkit 跑出的中文译文与某种已有的 AI+校对版本（如 ru 或 ko）做风格对照。
- **学术/数学术语**：图像处理、对数、傅里叶等术语，是 `proofread` 学术领域的高难样本。

## voxkit 用途映射

- `translate` en→zh → 译文风格可与 sub.ko.srt / sub.ru.srt 这种"同流程不同语种"产出对照；
- `reseg`（en）→ 用作长视频（44min）reseg 稳定性测试；
- `proofread`（en，学术术语）→ 数学/图像处理术语基准。

## 音频规格

按需下载——跑 `./fetch.sh`，产出 `audio.wav`（16kHz mono，约 80MB）。fetch.sh 已固化 yt-dlp 命令，无须手敲。

## 字幕预处理说明

入仓时已用 `sed '/criblate/d'` **剥离首条的 `[... criblate.com ...]` 头注**（原始字幕**只有首条**带这个注释，标明轨道来源——不是每条都带）。如需还原原始字幕，重新跑 `fetch.sh`。

剥离后首条示例（`sub.ko.srt`）：

```
1
00:00:00,000 --> 00:00:06,949
내가 이런 영상들 중 하나를 만들 때마다, ...
```
