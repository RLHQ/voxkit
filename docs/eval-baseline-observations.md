# voxkit baseline observations: 小宁子 10min zh

> 实跑日期：2026-05-13  ·  voxkit 0.5.1  ·  whisper large-v3-turbo  ·  proofread provider=deepseek edit-level=standard

实拉数据决定 `voxkit eval` 第一批指标的优先级。这份报告专门用来**校正** `docs/eval.md` v0 设计稿——指标做对了，才能驱动 reseg / proofread 的迭代改进。

## 1. 整体对比（一张表）

| 维度 | voxkit (proofread 后) | 人工金标 | 差值 |
|---|---|---|---|
| Cue 总数 | **145** | **302** | **-52%** 🔴 |
| Avg 字符 / cue | 23.1 | 11.1 | **2.1×** 🔴 |
| Avg 时长 / cue | 4.01s | 1.81s | **2.2×** 🔴 |
| 整体字符数（去标点） | 3073 | 3130 | -1.8% ✅ |
| voxkit 边界命中金标率（±0.3s） | 88.4% | — | ✅ |
| 金标边界被 voxkit 覆盖率（±0.3s） | 39.1% | — | 🔴 |
| 跨 cue 词切断 bug | 1 (Steam→"S t"+"eam") | 0 | 🔴 |
| ASR 错字纠正（proofread 阶段） | "狂爱刷新件"→"狂按刷新键" 等 | n/a | ✅ |
| 标点 + 半角空格 + 「的/地」(proofread) | 加得到位 | n/a | ✅ |

## 2. 三层差距分析

### 2.1 ASR 层（whisper transcribe）— 差距小

- 整体字符数仅差 1.8%，**几乎所有内容都抓到了**；
- 个别识别错误（"狂爱刷新件" "这么货" "完整的"）**全部被 proofread 阶段纠正**；
- 结论：ASR 层不是改进重点。可以放到 Phase 2 之后。

### 2.2 reseg 层（`--resegment semantic`）— **主要矛盾**

**问题 A：切分粒度严重不足（P0）**

voxkit 145 cues / avg 23 字符 / avg 4 秒 vs 金标 302 cues / avg 11 字符 / avg 1.8 秒。

边界数据揭示根因：**voxkit 切的地方都对（88% 命中金标），但是切得太少**（金标 39% 边界没被 voxkit 覆盖）。

例子：voxkit cue 1（00:00:00–00:00:04, "Steam 出新硬件了，是个手柄，卖 700 块。"）对应金标 3 条 cue：
```
1   00:00:00,000 → 00:00:01,791   Steam 出新硬件了
2   00:00:01,791 → 00:00:02,625   是个手柄
3   00:00:02,625 → 00:00:04,000   卖 700 块
```

人工断句是「一气口一行」，voxkit 是「一长气段一行」。后者**字幕滞留过长**，读者来不及看 → 主观感受是「不像人工字幕」。

可能根因（待验证）：
- `src/voxkit/core/semantic_resegment.py` 用 pysbd 做句子边界切分，pysbd 对中文偏向「按句号切」，对逗号/顿号等子句标点不敏感；
- CJK phrase-aware 打包阶段（见 transcribe.py:221 的注释）可能也偏宽松。

**问题 B：拉丁词跨 cue 切断（P1）**

cue 8 末尾 `... 它到底值不值得买？S t`，cue 9 开头 `eam 手柄是 ...`。

reseg 在 cue 边界切断了 "Steam"。本次 fixture 只命中 1 处，但任何 1 处都是字幕级别的破坏性 bug——必须在 reseg 阶段把拉丁词当原子单位。

### 2.3 proofread 层（DeepSeek `standard`）— 接近天花板

**做得对的（保持）**：
- ASR 错字纠正：`狂爱刷新件 → 狂按刷新键`、`这么货 → 这么火`；
- 中英 / 中数字之间自动加半角空格：`700块 → 700 块`、`Steam一直 → Steam 一直`；
- 末尾标点：`。，？` 添加到位；
- 「的/地」区分：`完整的告诉你 → 完整地告诉你`；
- 93% cues changed / 10% 高风险——比例正常。

**唯一小问题**：proofread 没有「合并/拆分」能力——它在原 cue 上做内容校正，**不能解决问题 A 的粒度问题**。这是设计决策（保持时间码不动），但意味着 reseg 的粒度问题必须在 reseg 阶段解决，不能指望 proofread 兜底。

**Speaker A 前缀**：proofread 后已不显示在 correctedText 里，但 cue 字段保留 `"speaker": "Speaker A"`——单说话人场景可考虑跳过 diarize 默认行为。

## 3. 对 `docs/eval.md` v0 设计稿的校正

v0 设计稿打算第一批做 4 个指标（CER / WER / boundary F1 / chrF）。**基于实际数据，应该收敛到 3 个真正反映问题的指标**：

| v0 指标 | 调整 | 理由 |
|---|---|---|
| ~~CER~~ | **降级**到 P2 | ASR + proofread 后字符总数差 1.8%，CER 改进空间太小，先不做 |
| ~~WER~~ | **删除** | 中文场景无意义；英文 fixture 再说 |
| boundary F1 | **保留为 P0** | 唯一能量化「切分密度不足」的指标 |
| chrF | **降级**到 P2 | 主要用于 translate，本轮不评估 translate |
| **新增：cue 密度比** | **P0** | voxkit_cues / gold_cues，直接读懂粒度差 |
| **新增：avg 字符/时长漂移** | **P0** | 读 readability 风险 |
| **新增：跨 cue 拉丁词切断数** | **P1** | 直接 catch Steam 这类 bug，正则可识别 |

## 4. 推荐：Phase 1 eval 最小实现范围

只做 3 个指标，1 个命令：

```bash
voxkit eval <workdir> --reference <gold.srt> --lang zh
```

输出 `eval.report.json`：

```jsonc
{
  "schemaVersion": 1,
  "alignment": {
    "vk_cues": 145,
    "gold_cues": 302,
    "density_ratio": 0.48          // < 0.8 报警
  },
  "metrics": {
    "boundary_f1": 0.54,           // tol=±0.3s
    "boundary_precision": 0.88,    // voxkit 切的对不对
    "boundary_recall": 0.39,       // 金标该切的有没有切
    "avg_chars_drift": 12.0,       // voxkit - gold
    "avg_dur_drift_s": 2.2,
    "broken_latin_words": 1        // 跨 cue 切断
  }
}
```

**precision vs recall 分开报，是这次的关键洞察**——voxkit 现状是高 precision 低 recall（切的都对，但漏切多），改进 reseg 是要**提 recall**，要能从指标上看出这种结构。

## 5. Phase 2 改进方向草案（数据驱动版）

### 2a. reseg 提粒度（P0）

候选方案，按尝试成本排序：

1. **改 pysbd 子句切分参数**：把逗号、顿号、感叹号纳入二级切点；
2. **加长度上限强制切**：单 cue 超过 2.5s 或 18 字符触发二次切分（基于词级时间戳）；
3. **CJK phrase-aware 打包参数**：当前打包可能过于积极，调小目标长度；
4. **LLM 辅助切分**：在 proofread 之前加一个轻量 LLM "splitter" 步骤——成本高，留作 fallback。

每个方案跑一次小宁子 fixture，看 boundary recall 提升。目标：recall 从 39% 提到 70%+。

### 2b. 拉丁词原子化（P1）

在 `semantic_resegment.py` 切分前预扫，把所有连续拉丁字符序列标记为不可切原子单位。代码量小。

## 6. 决策点

下一步建议（你拍板）：

- **A. 先把 Phase 1 eval 命令写出来**，把当前 baseline 数据**自动化**（这样后面每次改 reseg 都能机器跑分）；
- **B. 跳过 eval 命令，直接动手改 reseg**：用本报告里的人工统计当 baseline，改完再人工对比；
- **C. 先并行做：A 由我推进，你同时定 2a 的方案选 1-3 哪个先试**。

我的建议是 **A**——没有自动化打分，改 reseg 没法快速迭代验证（每次都得手工统计），心智负担太重。
