# tests/fixtures/youtuber_samples/

来自「认真做字幕的 YouTuber」名单的实拉样本：音频 + 人工字幕，作为 voxkit
`transcribe / reseg / proofread / translate / diarize` 的人类金标对照。

频道筛选标准与原始名单见 `docs/认真做字幕的YouTuber.md`。

## 入选样本（4 个，下载日期 2026-05-13）

| 目录 | 频道 | 时长 | 音频 | 字幕语种 | 主要用途 |
|---|---|---|---|---|---|
| `xnzxnz_first_look_10min/` | 小宁子 XNZ | 10:06 | ✅ | zh-Hans, zh-Hant, en-US（**zh↔en 同帧对齐**） | 中文 reseg + zh↔en translate 金标 |
| `kurzgesagt_germany_14min/` | Kurzgesagt | 14:05 | ✅ | en（**作者人工**） | 英文短篇科普 proofread / reseg |
| `3blue1brown_logarithm_subs_only/` | 3Blue1Brown | 44:52 | 按需 `fetch.sh` | 11 语（en 人工 + 10 语 AI+校对，各自独立切分） | 多语**风格**对照 + 客户场景对照（非逐条平行） |
| `lexfridman_ffmpeg_4h18m/` | Lex Fridman | 4:18:22（完整） | 按需 `fetch.sh` | en（带说话人切换 `- `） | 长 podcast diarize/align |

入仓总体积约 45 MB（仅含 `xnzxnz` / `kurzgesagt` 两个音频；3B1B ≈ 82MB 与 Lex ≈ 496MB 按需 `./fetch.sh` 拉取，不入仓）。

## 目录结构约定

```
youtuber_samples/<handle>_<topic>_<duration>/
├── meta.md              # 视频来源 + 字幕轨道性质（人工/AI翻译）+ voxkit 用途映射
├── audio.wav            # 16kHz mono WAV（可选）
└── sub.<lang>.srt       # 每种语种一份；语言代码取 yt-dlp/YouTube 原始标识
```

## 字幕重要警告

YouTube 的 "Available subtitles" 一栏里出现的字幕**不全是译者人工翻译**：

- **小宁子 zh-Hans / en-US**：作者人工双语，同帧对齐——真金标；
- **小宁子 zh-Hant**：YouTube 从 zh-CN **机器繁体转换**，断句相同，仅字形差异；
- **3Blue1Brown en**：作者人工字幕；
- **3Blue1Brown 非英文 10 语**：AI 翻译 + criblate.com 社区校对，**不是译者从零翻译**，**也不是逐条与 en 对齐的平行语料**（各语种条目数 407–816 不等，独立切分）；
- **Kurzgesagt en / Lex Fridman en**：官方人工字幕。

在用作 BLEU/对照集前，**先打开对应 meta.md 看字幕性质**——把 AI 翻译误当人工金标会高估 voxkit 的表现；把 3B1B 多语字幕当逐条平行语料会算错对齐。

> 注：fixture 入仓时**已剥离**各 srt 首条的 `[... criblate.com ...]` 头注，原始字幕首条带这个标注表明轨道来源；如需还原可重新跑 `fetch.sh`。

## 复现方式

```bash
# 1) 探测某频道最新视频的字幕情况
yt-dlp --skip-download --list-subs "https://youtube.com/watch?v=<ID>"
# 看 "Available subtitles for ..." 下面的列表（不是 "Available automatic captions"）

# 2) 下载音频（16kHz mono WAV）+ 指定语种字幕
yt-dlp -x --audio-format wav \
  --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
  --write-sub --sub-langs "<langs>" --sub-format vtt --convert-subs srt \
  -o "audio.%(ext)s" \
  -o "subtitle:sub.%(language)s.%(ext)s" \
  "https://youtube.com/watch?v=<ID>"

# 3) 长视频只截前 N 秒
... --download-sections "*0-300" --force-keyframes-at-cuts ...
```

## 已探测但**无外挂字幕**的频道（不入仓）

下列频道当前最新视频只有 YouTube auto-generated CC，不能作为人工金标：

- 影视飓风（@mediastorm6801）— 中文硬字幕烧录在画面内
- Huberman Lab（@hubermanlab）
- Rich Roll（@richroll）
- Every Frame a Painting（@everyframeapainting）
- CGP Grey、Nerdwriter1、Tim Ferriss
- 回形针 PaperClip（@papercliptv）近期只发 Shorts

这些频道**字幕本身做得好**（硬字幕节奏、官网 transcript 等），但**无法直接 yt-dlp 拉取**，用作 fixture 需先 OCR 或抓官网 transcript，本批未做。
