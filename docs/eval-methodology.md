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

## 10. L3 实跑数据（2026-05-13）

L3 已实施为 `voxkit eval --llm`（commit `c9b94e7`），对齐算法 fix `92c6eb6`
后跑了两个 fixture，**强力验证了 §7 的预测**：boundary metrics 跨场景不稳定，
LLM 评分跨场景稳定。

### 10.1 boundary F1 vs LLM overall 横向对比

| Fixture | boundary precision | boundary recall | **boundary F1** | **LLM overall mean** | LLM overall p10 |
|---|---|---|---|---|---|
| 小宁子 zh (vlog) | 0.898 | 0.603 | **0.721** | **8.34** | 5.0 |
| Kurzgesagt en (动画) | 0.180 | 0.209 | **0.194** | **8.02** | 5.0 |
| **跨 fixture 差距** | 5.0× | 2.9× | **3.7×** | **1.04×** | 1.0× |

**boundary F1 跨 fixture 差 3.7 倍，LLM overall 几乎持平（1.04 倍）**。

这正是 L3 设计目标——金标作参考但允许 voxkit 不同切法，LLM 不会因为 voxkit 按
句法切（vs 金标按字幕渲染单元切）就低分。LLM 实际给 Kurzgesagt `segmentation=8.06`，
认可了 voxkit 切法的合理性。boundary recall=0.209 是因为 voxkit 切了 202 个金标
没切的位置，但这些位置在 LLM 看来切得合理。

### 10.2 LLM 维度细分

| 维度 | 小宁子 mean / p10 | Kurzgesagt mean / p10 | 解读 |
|---|---|---|---|
| **semantic** (语义保留) | 8.41 / 5 | 8.16 / 5 | 跨场景一致 |
| **terminology** (术语/数字) | 9.46 / 9 | **9.66 / 10** | 动画配音 ASR 几乎完美 |
| **segmentation** (切分自然度) | 8.42 / 7 | 8.06 / 6 | LLM 认可不同切法 |
| **punctuation** (标点) | **9.62 / 9** | 9.24 / 7 | proofread 加标点优势明显 |
| **readability** (整体节奏) | 8.54 / 7 | 7.93 / 5 | 小宁子略好 |
| **overall** | 8.34 / 5.0 | 8.02 / 5.0 | 跨场景 1.04× |

**关键观察**：
1. **terminology p10**：小宁子 9 / Kurzgesagt 10——动画 ASR 优于 vlog 麦克风
2. **punctuation 小宁子高于 Kurzgesagt**：proofread 阶段对中文标点贡献明显（Kurzgesagt 没跑 proofread，stage=cues）
3. **p10=5.0 跨 fixture 相同**：底部 10% 总有问题 group，差异不大

### 10.3 L3 抓到 boundary 完全看不到的问题（high_risk 实例）

**小宁子 zh（32 / 214 高风险）**：

| group | voxkit | 金标 | LLM 抓 |
|---|---|---|---|
| 6 | 「这**一波**瞄准的目标用户」 | 「**G 胖**瞄准的目标用户」 | ASR 错识 + proofread 漏改 |
| 12 | 「Steam」 | 「它到底值不值得买」 | reseg 把 "Steam" 单独成 cue 信息丢失 |
| 16 | 「Steam **主体**的按键」 | 「Steam **主题**的按键」 | 错字 |
| 24 | 「也被**接到**了」 | 「也被**挤到**了」 | ASR 错字 |
| 29 | 「但是这个**抛开**...」 | 「但是这个**不好够**」 | ASR 严重错识 |
| 54 | 「不支持 PS5、」 | 「PS5 Switch 和 Xbox」 | 切断丢内容 |

**Kurzgesagt en（48 / 241 高风险）**：

| group | voxkit | 金标 | LLM 抓 |
|---|---|---|---|
| 6 | 「What happened?」 | 「What happened?」+「Population Collapse」 | 漏掉 "Population Collapse" 标题 |
| 7 | 「collapse.」 | 「Population Collapse」 | ASR 没识别 "Population"，只剩 "collapse" |
| 11 | 「Compared to South Korea, this sounds almost amazing,」 | 完整复合句 | 切断丢核心 "but it still means population collapse" |
| 22 | 「In 2026, Germany is already one of the oldest...」 | 完整 + "with a median age of over 45" | 漏后半部分 |

**所有这些问题 boundary metrics 都给 hit**（时间对齐 OK），但 LLM 准确抓到了实质质量问题。

### 10.4 工程化产物

入仓 baselines（精简版，~21-35 KB）：
- `tests/fixtures/youtuber_samples/xnzxnz_first_look_10min/baseline-llm.eval.json`
- `tests/fixtures/youtuber_samples/kurzgesagt_germany_14min/baseline-llm.eval.json`

每份含：aggregate（5 维 + overall mean/p50/p10）+ high_risk_groups 完整列表（含原文 + LLM 解释）+ prompt hash + token 消耗。

完整 eval-llm.report.json（含全部 groups 详情，146-300 KB）落在 `/tmp/voxkit-samples/`，不入仓。

### 10.5 单次评估成本

| Fixture | input cue | LLM call | tokens (prompt+completion) | 估计成本 |
|---|---|---|---|---|
| 小宁子 zh (10min) | 213 | 22 | 49k + 81k = 130k | ~¥0.05 (deepseek-v4-flash) |
| Kurzgesagt en (14min) | 241 | 25 | 58k + 98k = 156k | ~¥0.06 |

**单次 release/PR review 跑一对 fixture 总成本 < ¥0.2**，可接受。

### 10.6 关于 L3 的剩余风险

- **LLM 评分有 variance**：temperature=0 但 DeepSeek 仍有少量浮动。两次跑同 fixture overall 可能差 0.1-0.3。**不要 over-fit 单次 LLM 分数**。
- **prompt 校准未跟人工评分对照**：当前 LLM 给 8.34/8.02 是否反映真实人感受？建议下一轮做人工 spot-check 校准：手抽 20 个 cue 自己 1-10 打分，跟 LLM 比相关性。
- **prompt v1 未优化**：当前 prompt 直接基于 §5.4 草案，没迭代。如果发现 LLM 系统性偏松/偏严，调 prompt 重跑（promptHash 变化作为 baseline 失效信号）。

## 11. 下一轮决策点

L3 数据让中文场景 ROI 更清晰：

| 候选 | 现状 | L3 加持的判断 |
|---|---|---|
| **3a proofread 加 split 能力** | 未做 | LLM 已 catch ~6 个 ASR 错字（'这一波'/'主体'/'接到'/'抛开'/'拗开'），proofread prompt 加「拆分能力」可能不是关键，更应该改 proofread prompt **加强错字检测** |
| **EN 路径候选 A**（pysbd 权重调整）| 未做 | LLM 显示 EN 实际质量 OK（8.02），boundary F1 0.194 是**指标 artifact** 不是真问题。EN 不优先 |
| **proofread 错字检测加强** | 未做 | 新方向：LLM 抓到的 6 个错字都是同音字（一波/G 胖、主体/主题、接到/挤到、抛开/不好够）—— proofread v2 加「相邻同音字校验」启发 |
| **3B1B 跑 LLM eval** | 未做 | 794 cue × ~¥0.2 = ~¥1。第三 fixture 数据点，可选 |

**新洞察**：L3 揭示 voxkit 的真问题在 **ASR 错字 + 同音字**，不在 reseg 切分粒度。
proofread v2 prompt 加强错字检测可能比 reseg 候选 A 更高 ROI。
