# voxkit eval 评估方法学

> 状态：**反思 + 设计稿 v0**（2026-05-13），不含实现。
>
> 本文档诚实质疑当前 `voxkit eval` 指标（boundary precision/recall/F1）的局限性，
> 并设计 **L3 LLM 多维评估** 作为补充。是否实施由下一轮决策。

## 1. 当前 eval 指标实际测什么？

`voxkit eval`（v0.7.0）输出 3 类指标：

1. **boundary precision/recall/F1**：voxkit cue 时间边界 vs 金标边界 ±tol（默认 0.3s）的对齐率；
2. **density_ratio + chars_drift + dur_drift**：cue 数量与平均长度对齐度；
3. **broken_latin_words**：跨 cue 切断拉丁词的启发式 catch（CJK only）。

**核心是 1**——boundary 对齐。但 boundary 对齐 ≠ 字幕质量。

## 2. boundary metrics 的 4 个盲点

### 2.1 语义完整性盲点

voxkit 可以切在「Steam」+「手柄是一个对称布局的」两个 cue 里——
boundary 时间对了就算 hit，但人眼一看就别扭（"Steam" 单独成 cue 信息量太低）。

反过来，voxkit 切在语义完美的气口，但凑巧落在金标 ±0.4s 外，
就被判为 miss，哪怕实际切法**更**符合阅读体验。

### 2.2 内容质量盲点

错别字 / 漏字 / 多字 / 标点错误 **完全不查**。
proofread 漏改一个错字、translate 翻错术语、CJK 半角空格漏加——eval 0 反应。

具体例子（v0.7.0 真实数据，hubermanlab.com）：
- voxkit："如果你是这一波瞄准的目标用户"
- 金标：  "如果你是 G 胖瞄准的目标用户"

「这一波」vs「G 胖」是 ASR 识别错误，但 boundary 计算时这两条 cue 时间对齐 → **算 hit**，质量损失被掩盖。

### 2.3 「同义切法」被惩罚

字幕切分有多种合理方式：

- 按气口切：「是个手柄 / 卖 700 块」
- 按句法切：「是个手柄卖 700 块」
- 按镜头切：还可能切在镜头转换处

金标只是其中一种切法（这位 YouTuber 的偏好）。
voxkit 不同但同样合理的切法会被扣分——「金标 = 唯一正确」是过强假设。

### 2.4 阅读体验只测一半

`chars_drift` 部分反映 readability，但 `voxkit quality` 里的
CPS / 闪屏率 / 滞留率 / 字符上限违例 **没融入 eval**。
两个产物各自报告，没合并视图。

## 3. precision=0.906 ≠ 字幕质量 90.6%

明确这点：boundary F1 = 0.720 只说明 **voxkit 的 cue 边界位置与金标对齐的平衡点是 72%**。
它不说：

- 字幕用着舒不舒服
- 内容对不对
- 节奏合不合适

把这个数字当**字幕总体质量分**会严重误导。它只是**一个维度**。

## 4. 推荐的分层评估架构

不是 LLM 替代 boundary，而是**多层组合**：

| Layer | 性质 | 跑频率 | 成本 | 当前状态 |
|---|---|---|---|---|
| **L1 客观对齐** | boundary precision/recall, density, broken_latin | 每次 commit | 秒级，零 LLM | ✅ `voxkit eval` |
| **L2 物理可读性** | CPS, 闪屏率, 滞留率, 字符上限违例, trailing-bad | 每次 commit | 秒级，零 LLM | ✅ `voxkit quality` |
| **L3 LLM 多维评分** | 语义保留 / 术语 / 标点 / 节奏 / 整体 | release / PR review | 分钟级，~¥0.1 | ❌ 待实施 |
| **L4 人工 spot-check** | 抽样 cue 人眼对照 | 重大改动后 | 人时 | ✅ HTML 对照报告（手工） |

**L3 是最值得新增的**，因为它能 catch L1/L2 完全看不到的问题——错别字、术语错、标点风格。

L4 已经通过 HTML 对照报告半工程化（见 `/tmp/voxkit-samples/<fixture>/<version>/comparison.html`），下一步可作为 `voxkit eval --html` 子命令固化。

## 5. L3 LLM eval 设计稿

### 5.1 CLI 接口

```bash
voxkit eval --llm <workdir> \
  --reference <path/to/gold.srt> \
  --lang <bcp47> \
  [--provider deepseek]              # 默认复用 proofread provider 选择
  [--model MODEL]
  [--max-cues-per-batch 20]
  [--output eval-llm.report.json]
```

### 5.2 流程

1. **时间窗口对齐**：复用 `voxkit.core.eval_metrics.boundary_metrics` 的对齐算法，把 voxkit cue 与金标 cue 配成 N-M 对（IoU >= 0.3 视为同一组）。
2. **批量 prompt**：每组对喂给 LLM，让它对 voxkit 输出打分（**金标作参考但非唯一答案**）。
3. **聚合**：每对返回结构化 JSON 评分 + 文字解释。
4. **落产物**：`eval-llm.report.json` + stdout 摘要。

### 5.3 评分维度（5 维 + 整体）

| 维度 | 0-10 评分定义 |
|---|---|
| **语义保留度** | voxkit cue 内容是否传达了对应金标的所有关键信息（不要求字字相同）|
| **术语准确性** | 专有名词 / 数字 / 英文术语 / 中英混排是否正确 |
| **断句自然度** | 阅读节奏是否流畅，不机械对齐金标——voxkit 的不同切法也可被高分 |
| **标点规范** | 中文全角标点 / 半角空格 / 句末标点使用是否符合主流字幕风格 |
| **整体可读性** | 综合（CPS、cue 长度、字幕滞留时间感受）|

每维度 0-10 分 + 一句话解释。整体可读性是**加权平均**（非简单平均，重点维度权重高）。

### 5.4 Prompt 草案

```
你是中文字幕质量评审员。下面给你一组 voxkit 自动产生的字幕 cue 和对应人工金标 cue
（同一时间窗口）。请评估 voxkit 的输出质量。

【关键原则】
- 金标只是「一种合理切法」，不是唯一正确答案
- voxkit 的不同切法只要语义完整、阅读自然，应该高分
- 重点 catch 错别字、术语错、标点风格不符、阅读体验差

【时间窗口】
voxkit cues: [...]
gold cues: [...]

【输出 JSON】
{
  "scores": {
    "semantic": 0-10,
    "terminology": 0-10,
    "segmentation": 0-10,
    "punctuation": 0-10,
    "readability": 0-10
  },
  "overall": 0-10,
  "issues": ["..."],   // 高风险问题列表（漏译、错字、断句怪）
  "explanation": "..."  // 一句话总评
}
```

### 5.5 产物 schema

```jsonc
{
  "schemaVersion": 1,
  "workdir": "...",
  "reference": "...",
  "language": "zh",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "promptHash": "sha256:...",
  "alignment": {
    "voxkit_groups": 198,
    "gold_groups": 198,
    "alignment_iou_median": 0.71
  },
  "scores_aggregate": {
    "semantic": {"mean": 8.4, "p50": 9, "p10": 5},
    "terminology": {...},
    "segmentation": {...},
    "punctuation": {...},
    "readability": {...},
    "overall": {"mean": 8.2, "p50": 8, "p10": 4}
  },
  "high_risk_cues": [
    {"voxkit_idx": 8, "gold_idx": 7, "overall": 4, "issues": ["ASR 错字：这一波→G 胖"]}
  ],
  "tokens": {"prompt": 9800, "completion": 4200, "cost_estimate_usd": 0.02}
}
```

`high_risk_cues` 是**最有价值的人工 review 入口**——人不需要看 200 条，只看 LLM 标出的 10-20 条问题 cue 即可。

### 5.6 实施成本估算

| 项 | 估计 |
|---|---|
| 工程化代码量 | ~200 行 Python（`commands/eval_llm.py` + `core/llm_eval.py`）+ ~100 行测试 |
| Prompt 调优 | 1-2 天迭代到稳定 |
| 单次评估 cost | 10min 视频（~200 cues）→ ~10k input + 5k output tokens → ~¥0.05-0.1 (DeepSeek) |
| 单次评估耗时 | ~1-2 min（10 个并发 LLM 请求）|

## 6. 实施前要解决的疑虑

| 疑虑 | 应对 |
|---|---|
| **LLM 评分不可重复**（同字幕两次跑分不一）| 设 temperature=0 + 锁 prompt hash；接受 ±0.5 分 variance |
| **LLM 偏向认可"看起来合理"的输出** | prompt 强调「catch 错误」+ 加 few-shot 反例 |
| **prompt 决定一切** | 用 `voxkit/v0.7.0` 跑一遍人工 review 校准，比 LLM 评分 vs 人工评分 |
| **CI 频繁跑成本高** | L3 不进 CI，只在 release/PR 跑 |
| **金标作 reference 矛盾**（"金标不是唯一答案"但 prompt 给 LLM 看）| 显式告诉 LLM「金标是参考非答案，可以打高分给不同但合理的切法」 |

## 7. 决策路径

不立即实施。下一轮可选：

- **A. 直接做 L3 voxkit eval --llm**：~2-3 天工程化 + 调优；产出新维度数据
- **B. 先把 reseg F1 推到 0.78+**（候选 A 改 packing 强制 flush medium-break atom），稳定基础再上 L3
- **C. 跑多 fixture 验证 boundary metrics 一致性**：在 Lex Fridman / 3B1B en 上跑，看 boundary F1 与人工感受是否还相关；如果不相关，更优先 L3

我的倾向：**B+C 并行**。继续推 boundary 数据到 ≥ 0.78，同时跑多 fixture 验证 boundary metrics 是否在多样 fixture 上仍可信。如果 boundary 仍可信，L3 是锦上添花；如果不可信（某个 fixture F1 高但人觉得字幕烂），L3 立刻变成必需。

## 8. 当前可用的 L4 人工对照工具

不等 L3 实施，**现在就能用 L4 抽样查质量**：

```bash
# 生成 HTML 左右对照报告
.venv/bin/python tools/render_comparison.py \
  /tmp/voxkit-samples/<fixture>/<version>/subtitles.reseg2.srt \
  tests/fixtures/youtuber_samples/<fixture>/<gold>.srt \
  -o /tmp/voxkit-samples/<fixture>/<version>/comparison.html
```

（注：当前 `render_comparison.py` 是 ad-hoc 脚本未固化；下一轮如做 L3，顺手把它落到 `tools/` 或新增 `voxkit eval --html`。）
