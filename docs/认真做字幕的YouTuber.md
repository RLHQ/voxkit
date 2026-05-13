# 认真做字幕的 YouTuber

> 这不是「推荐订阅」清单，而是 voxkit 字幕质量工程的**人类金标参考样本库**。
>
> 用途：
> - 给 `reseg`（语义重切）提供「人是怎么断句」的对照基准；
> - 给 `proofread`（LLM 校对）提供领域术语写法的参考；
> - 给 `translate`（中↔英）提供双语对齐样片；
> - 给 `diarize + align` 提供多人长对话的说话人切换标注；
> - 给 `quality` 报告提供可校的 fixture 源。
>
> 本文档基于 **2026-05-13 实拉探测数据**编写。每个频道的字幕状况都用
> `yt-dlp --list-subs` 核实过，没有想当然。

---

## 1. 入选标准

「认真做字幕」根据**字幕的获取方式**分两类，两类都有价值，但适配的 voxkit 能力不同。

### A 类：有外挂手动字幕轨道（可 `yt-dlp` 直接拉取）

硬性条件：
1. **YouTube 字幕开关里能看到**「中文（中国）」「English」等具名轨道，**不是**「中文（自动生成）/ English (auto-generated)」。判别命令：
   ```bash
   yt-dlp --skip-download --list-subs "<video_url>" | grep -A 30 "Available subtitles"
   ```
   有非空行 = A 类。
2. **断句服务于阅读节奏**：句末有标点、不会把术语腰斩、跨镜头不硬切。
3. **更新有规模**：≥30 期稳定输出。

> ⚠️ 警告：YouTube 的 "Available subtitles" 不等于「译者人工翻译」。
> 3Blue1Brown 的非英文轨道是 **AI 翻译 + 社区校对**（每条带 `[AI translated]` 头注），
> 当 fixture 时按 **AI+校对** 性质处理，不要当作人工译者金标。

### B 类：硬字幕烧录 / 仅官网 transcript（YouTube 端拿不到外挂轨道）

字幕做得认真，但**烧在画面里**或**只在频道官网/独立 transcript 站**，无法 `yt-dlp` 直接拉。
适合做：

- **风格观察样本**（看人是怎么断句、术语写法）；
- **OCR 试验素材**（硬字幕 + 视频流，提取出来再清洗）；
- **官网 transcript 抓取目标**（如 lexfridman.com、hubermanlab.com 提供完整 transcript）。

B 类频道列在 §3，但 voxkit fixture 优先用 A 类。

---

## 2. A 类频道（5 个，全部实拉验证）

每条按以下模板列：链接、视频形态、字幕轨道实拉结果、对 voxkit 的价值、是否已落到 `tests/fixtures/youtuber_samples/`。

### 1. 小宁子 XNZ —— 中文金标

- 链接：<https://www.youtube.com/@xnzxnz>
- 视频形态：财经/科技/消费产品短中长度解说，主要中文。
- 字幕实拉结果（最新一期 `2_FpQj_f69g`，10 min）：
  - `zh-CN`、`zh-Hans`（byte-identical，作者上传的简体）
  - `zh-Hant`（YouTube 机器繁体转换）
  - `en-US`（作者人工英译，**与 zh-Hans 每条同时间窗口对齐**）
- voxkit 价值：
  - **中文 `reseg` 唯一金标频道**（A 类里唯一中文）；
  - **`translate` zh↔en 双向对照真金标**（双语同帧对齐）；
  - **`proofread` 中文术语写法基准**。
- 已下载 fixture：`tests/fixtures/youtuber_samples/xnzxnz_first_look_10min/`

### 2. 3Blue1Brown —— 多语对照样本

- 链接：<https://www.youtube.com/@3blue1brown>
- 视频形态：数学/计算机视觉化讲解长视频，典型 30–50 分钟。
- 字幕实拉结果（最新一期 `ldxFjLJ3rVY`，44 min）：
  - `en`：作者人工原文，**首条**带 `[Submit subtitle corrections at criblate.com]` 头注标明轨道来源；
  - `ar / de / es / fr / hi / hr / ko / pt-BR / ru / vi`（10 种）：**AI 翻译 + criblate.com 社区校对**，**首条**带 `[AI translated. Submit corrections at criblate.com]`（本地语言版）头注标明轨道性质。
- voxkit 价值：
  - **`translate` 多语风格对照**：11 种语言**各自独立切分**（条目数 407–816 差异较大，**不是逐条平行语料**），适合做译文风格对照，不适合做严格 BLEU 对齐；
  - **客户场景对照**：3B1B 的工作流（AI 翻译 → 社区校对）就是 voxkit `translate + proofread` 想服务的产品形态；
  - 局限：非英文轨道**不是译者人工翻译，是 AI+校对的天花板**，把它当 BLEU 金标会把 voxkit 校到 AI 译本水平为止。
- 已下载 fixture（仅字幕，44min 音频按需 `fetch.sh` 拉）：`tests/fixtures/youtuber_samples/3blue1brown_logarithm_subs_only/`

### 3. Kurzgesagt – In a Nutshell

- 链接：<https://www.youtube.com/@kurzgesagt>
- 视频形态：动画式科普短片，8–14 分钟。
- 字幕实拉结果（最新一期 `n-gYFcVx-8Y`，14 min）：
  - **仅 `en` 一轨**（官方人工，旁白气口断句）。
  > 此处修正一个常见误解：传说 Kurzgesagt "官方提供多语 CC" 在 2026-05 探测时**不成立**，至少近期视频只有 en 一轨。
- voxkit 价值：
  - **`proofread` 英文科普术语**基准（人口学、政治、经济密集）；
  - **`transcribe` 英文 ASR baseline**（动画配音棚级别录音）；
  - VTT 中保留 `\h` 硬空格，可测 voxkit 字幕清洗。
- 已下载 fixture：`tests/fixtures/youtuber_samples/kurzgesagt_germany_14min/`

### 4. Lex Fridman Podcast

- 链接：<https://www.youtube.com/@lexfridman>
- 视频形态：1–4 小时双人长访谈。
- 字幕实拉结果（最新一期 `nepKKz-MzFM`，2h28m）：
  - **仅 `en` 一轨**，但**带说话人切换标注**：每次说话人变化，新条目以 `- ` 开头（例：`- The important is, is your code good?`）。
- voxkit 价值：
  - **`diarize + align` 长对话压力测试**：字幕里的 `- ` 切换点可作为 diarize 输出说话人边界的对照；
  - **`proofread` 跨小时术语一致性测试**。
- 已下载 fixture：`tests/fixtures/youtuber_samples/lexfridman_ffmpeg_5min/`（音频截前 5min + en 全程字幕）

### 5. Veritasium

- 链接：<https://www.youtube.com/@veritasium>
- 视频形态：科学实验/物理科普长视频，15–30 分钟。
- 字幕实拉结果（最新一期 `SVTPv4sI_Jc`，21 min）：**仅 `en` 一轨**（官方人工）。
- voxkit 价值：
  - **`proofread` 英文科学术语**基准（量子、生物物理等）；
  - 与 Kurzgesagt 互为对照（同英文科普但风格不同：Veritasium 实地实验、Kurzgesagt 动画解说）。
- 已下载 fixture：暂无（本批没拉，按需补）。

### 备选 A 类（探测过但未入精选）

- **TED-Ed**（<https://www.youtube.com/@TEDEd>）— 有 my/en/ko 等手动字幕，教育类视频，可作 §2 补位。

---

## 3. B 类频道（字幕认真但 YouTube 端无外挂轨道）

这些频道字幕**做得非常认真**，但获取方式不是 `yt-dlp --list-subs`：

| 频道 | 链接 | 字幕形态 | 字幕长处 | 获取方式 |
|---|---|---|---|---|
| 影视飓风 Mediastorm | <https://www.youtube.com/@mediastorm6801> | 中英双语**硬字幕烧录** | 节奏跟镜头切换 + 配音气口 | OCR 视频流 |
| Huberman Lab | <https://www.youtube.com/@hubermanlab> | 官网 transcript | 神经科学术语稳定 | hubermanlab.com 抓 transcript |
| Rich Roll | <https://www.youtube.com/@richroll> | 官网 transcript（部分期） | 多人长访谈说话人标注 | richroll.com 抓 transcript |
| Every Frame a Painting | <https://www.youtube.com/@everyframeapainting> | 视频内置（已停更） | 影评解说断句教科书 | OCR / 手抄 |
| 回形针 PaperClip | <https://www.youtube.com/@papercliptv> | 硬字幕烧录（停更） | 高信息密度术语稳定 | OCR |

> 这些频道是「字幕质量学习」的好教材，但**不要直接挪到 `tests/fixtures/`**——
> 当前没有抓取脚本，硬塞进去会变成不可复现的 fixture。

### 不确定 / 已剔除

以下频道在初版文档里出现过，本次复核后剔除或降级：

- **何同学**：YouTube 端字幕情况不稳定，主战场在 B 站，剔除。
- **林亦 LYi**：纯硬字幕烧录，无外挂轨道（用户反馈确认）。如做 OCR 试验可考虑，但本批不入。
- **罗翔说刑法 / 小Lin说 / 所长林超**：handle 不确定 + 字幕状况未实拉，本批不入。如需补，先用 §1 命令探测后再决策。

---

## 4. 已下载 fixture 索引

详见 `tests/fixtures/youtuber_samples/README.md`。一目了然版：

| 目录 | 频道 | 音频 | 字幕语种 | 总用途 |
|---|---|---|---|---|
| `xnzxnz_first_look_10min/` | 小宁子 | 10min WAV | zh-Hans / zh-Hant / en-US | 中文 reseg + zh↔en translate |
| `kurzgesagt_germany_14min/` | Kurzgesagt | 14min WAV | en | 英文 proofread / reseg |
| `3blue1brown_logarithm_subs_only/` | 3Blue1Brown | 按需 `fetch.sh` | en + 10 种 AI 翻译（各自独立切分） | translate 多语**风格**对照（非逐条对齐）|
| `lexfridman_ffmpeg_5min/` | Lex Fridman | 5min WAV（截） + 全程字幕 | en（带说话人 `- ` 切换） | diarize/align |

总体积约 55 MB。

---

## 5. 使用建议

| 评估能力 | 优先用 |
|---|---|
| `reseg`（中文断句） | 小宁子 `xnzxnz_first_look_10min/sub.zh-Hans.srt` |
| `reseg`（英文断句） | Kurzgesagt `kurzgesagt_germany_14min/sub.en.srt` |
| `translate` zh→en | 小宁子（同帧对齐金标） |
| `translate` 多语**风格**对照 | 3Blue1Brown（注意：非 en 是 AI+校对的天花板；各语种独立切分，非逐条平行） |
| `proofread`（英文科普术语） | Kurzgesagt + Veritasium |
| `proofread`（英文学术术语） | 3Blue1Brown en 原文 |
| `diarize + align` | Lex Fridman（字幕 `- ` 切换标注） |
| `transcribe` WER baseline | 任意 A 类（音频干净 + 字幕 = 参考转录） |

## 6. 扩展这份名单

发现新候选频道时按这个流程走，**别再凭印象写**：

```bash
# 1. 取频道最新一期非 Shorts 视频
yt-dlp --no-warnings --flat-playlist --playlist-end 5 \
  --print "%(id)s|%(duration)s|%(title)s" \
  "https://www.youtube.com/@<handle>/videos" \
  | awk -F'|' '$2 > 120 {print; exit}'

# 2. 用上面拿到的 video id 探测字幕
yt-dlp --skip-download --list-subs "https://youtube.com/watch?v=<ID>"

# 3. 看输出里 "Available subtitles for ..." 块：
#    - 有非空行 → A 类候选，按 §2 模板写
#    - 只有 "Available automatic captions" → B 类，按 §3 写并标明获取方式
```

字幕的**真实性质**（人工 vs AI+校对 vs 机器繁体转换）必须打开 SRT 文件看头几条才能确认——
**不要相信轨道名字**。3Blue1Brown 的 `ko` 轨道叫得跟人工译者的轨道一模一样，但点开是 AI 翻译。
