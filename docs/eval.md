# voxkit eval（设计稿）

> 状态：**草案 v0**，未实现。本文档对应任务：用「认真做字幕的 YouTuber」实拉字幕作为人类金标，跑 voxkit 后做回归对照，持续改进 transcribe / reseg / translate / proofread 各阶段。

## 1. 为什么不直接扩 `quality`

`voxkit quality`（`src/voxkit/commands/quality.py`）现在做的是**纯内在质量**：吃 workdir 里 `subtitles.cues.json` / `subtitles.proofread.json` / `subtitles.<lang>.json`，算 CPS、字符上限违例率、闪屏率、滞留率、speaker 切换率、风险直方图等指标——**完全不需要外部参考**。

`eval` 做的是**有金标对照下**的回归打分：「voxkit 这次跑出来的 SRT，比上次更接近人类版本了吗？」两件事职责不同：

| 维度 | `quality`（已有） | `eval`（拟新增） |
|---|---|---|
| 输入 | workdir 内 artifact | workdir + **外部参考字幕** |
| 问题 | 字幕「能不能用」 | 字幕「像不像人」 |
| 何时跑 | 每条流水线产出后 | 有金标 fixture 时（CI 回归 / 本地调试） |
| 失败的含义 | 字幕不达可发布标准 | 模型 / 提示词回退了 |

两者独立、可叠加。`eval` **复用** `quality_metrics` 的物理指标计算器（重用而非平行实现）。

## 2. CLI 接口

```bash
voxkit eval <workdir> \
  --reference <path/to/sub.<lang>.srt> \
  --lang <bcp47> \
  [--stage transcribe|translate|proofread]   # 默认根据 lang 自动挑 artifact
  [--report <path>]                          # 默认 <workdir>/eval.report.json
  [--metrics cer,boundary_f1,chrf]           # 默认全开（chrF 而非 BLEU——见 §4）
```

多金标场景（小宁子 zh + en 双向）通过两次调用串：

```bash
voxkit eval out/xnzxnz --reference fixtures/.../sub.zh-Hans.srt --lang zh --stage transcribe
voxkit eval out/xnzxnz --reference fixtures/.../sub.en-US.srt    --lang en --stage translate
```

注册位置：`src/voxkit/cli.py`（与 `quality`、`review` 同层 subparser）。

## 3. 时间窗口对齐（核心难点）

参考字幕和 voxkit 输出**切分边界几乎不会一致**——人按气口断，voxkit 按 VAD 或 LLM 断。所以不能逐条比，必须先把两侧 cue 流按时间窗口配对。

策略（first cut，刻意简单）：

1. 用 cue 时间区间的 **IoU**（intersection-over-union）做二部图匹配；
2. 阈值 `IoU >= 0.3` 视为同一句的对应 cue（可能 1-N / N-1）；
3. 多对一情况下，把对侧文本拼接后再算字符级距离；
4. 没匹配上的视为「漏切」或「多切」，单独计入 boundary metric。

这套办法不完美（漂移大的视频会塌），但足以让首版跑起来。后续可换 Needleman–Wunsch（cue-level DP 对齐）。

## 4. 第一批指标

| 指标 | 含义 | 用在哪 |
|---|---|---|
| **CER**（char error rate） | 字符级 Levenshtein / 参考长度 | transcribe（CJK 友好）、translate |
| **WER**（word error rate） | 词级 Levenshtein，**仅英文 / 拉丁** | transcribe en |
| **boundary F1** | cue 切分边界的精确率/召回率（IoU≥0.5 算命中） | reseg / transcribe |
| **chrF**（character F-score） | 译文相似度，BLEU 的替代——CJK 上更稳，无需 tokenizer | translate |
| **CPS drift** | voxkit 与参考的 CPS 中位数差 | reseg 健康度 |

依赖：`python-Levenshtein` 或 `editdistance`（任选轻量包，<100KB）。**不引** sacrebleu / nltk 等重型 NLP——chrF 自实现（5 行字符 n-gram 即可）。

## 5. 报告产物

`eval.report.json` 结构（与 `quality.report.json` 形态对齐）：

```jsonc
{
  "schemaVersion": 1,
  "workdir": "out/xnzxnz",
  "reference": "tests/fixtures/.../sub.zh-Hans.srt",
  "lang": "zh",
  "stage": "transcribe",
  "alignment": {
    "ref_cues": 302,
    "hyp_cues": 318,
    "matched_pairs": 281,
    "unmatched_ref": 21,
    "unmatched_hyp": 37,
    "median_iou": 0.71
  },
  "metrics": {
    "cer": 0.087,
    "boundary_f1": 0.84,
    "chrf": 0.78,
    "cps_drift": -0.4
  },
  "deltas_vs_baseline": null   // 可选：若给了 --baseline <prev_report>
}
```

`--baseline prev.report.json` 时多输出 `deltas_vs_baseline`，即回归差值。

## 6. 首批挂载 fixture

按金标可靠度排序，先跑最稳的：

| Fixture | 阶段 | 参考 | 期望首跑结论 |
|---|---|---|---|
| `xnzxnz_first_look_10min/` | transcribe (zh) | `sub.zh-Hans.srt` | 建立中文 ASR CER 基线 |
| `xnzxnz_first_look_10min/` | translate (zh→en) | `sub.en-US.srt`（同帧对齐） | 建立 zh→en chrF 基线 |
| `kurzgesagt_germany_14min/` | transcribe (en) | `sub.en.srt` | 建立英文 ASR WER 基线 |
| `lexfridman_ffmpeg_4h18m/` | transcribe (en) | `sub.en.srt`（全程 4h18m） | 长对话场景压力；按需 `fetch.sh` 拉音频后用 ffmpeg 自切片回归 |
| `3blue1brown_logarithm_subs_only/` | — | — | **暂缓**——10 种非英文是 AI+校对天花板，不当金标；en 需要先跑 `fetch.sh` 拉音频 |

3B1B 非英文字幕**永远不**作为 `eval --reference`，但可作为「voxkit 译文 vs 现有 AI+校对译文」的**风格比较**素材，归到另一个工具或 ad-hoc 脚本里。

## 7. 怎么挂上回归

- **本地**：`make eval` 或 `just eval` 跑全部 fixture，diff 上次报告；
- **CI**：跑稳定 fixture（先排除需 fetch 音频的 3B1B），失败条件设为「任一指标比 `main` baseline 退化 > 5%」；
- **历史趋势**：在 `tests/fixtures/youtuber_samples/<fixture>/baseline.eval.json` 里保留上一次绿色构建的报告，commit 跟着代码走。模型 / prompt 改动需要同步更新 baseline，强制人工 review。

## 8. 不在本期范围内

- 多 ASR 模型横向对比（whisper-large vs medium vs distil）；
- 说话人 diarization 评分（DER）——等 voxkit diarize 落地再说；
- 半自动 fixture 扩张（脚本化抓更多 YouTuber）；
- Web UI 看报告。

## 9. 实现工单（待开）

1. `src/voxkit/core/alignment.py`：cue 时间窗口对齐（IoU + 二部图）；
2. `src/voxkit/core/eval_metrics.py`：CER / WER / chrF / boundary F1；
3. `src/voxkit/commands/eval.py`：CLI 入口 + 报告序列化；
4. `tests/test_eval_alignment.py`、`tests/test_eval_metrics.py`：算法单测；
5. `tests/test_eval_fixtures.py`：e2e 跑 §6 表里的 fixture（音频缺失时 skip）；
6. `cli.py` 注册子命令。

预估首版（仅 §4 的指标 + §6 前两条 fixture）2–3 天。
