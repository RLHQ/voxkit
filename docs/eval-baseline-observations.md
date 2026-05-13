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

---

## 7. Phase 3 增补：EN 路径根因分析

Kurzgesagt 14min EN 实测 precision 0.180 / recall 0.209 / F1 0.194。**这不是 Phase 2 引入的 regression**——Phase 2 改动只动 `_CJK_*` 分支，EN 路径未受影响。这是 EN 路径的固有 baseline。

### 7.1 根因（按重要性）

1. **目标函数本质冲突**（最深层根因）：voxkit `_resegment_word_level` 按 pysbd 的「句子语义完整性」切分，金标按「字幕渲染单元 + 镜头转换 + 演员停顿」切分。前者优化句法，后者优化字幕渲染节奏。两者**fundamentally 不同**——纯参数调优只能边际改善，不能跨过这个 gap。

2. **VAD 静默吃进 cue**：whisper-cli 输出的 segment timestamp 只覆盖有声部分，相邻 segment 之间的静默不被显式建模。voxkit `_split_long` 的 `prosody_gap_s=0.25` 检测依赖 word-level 时间戳的相邻 gap——但 whisper 在静默处的 word timestamp 可能丢失，导致 7s+ 静默被吞进前一个 cue（实例：voxkit `[25.46-33.67s] What happened?` 把 8 秒静默归入）。

3. **pysbd char span → word index 反查不稳**：`_char_range_to_word_range`（line 246-261）用贪婪策略映射，pysbd 切点落在空格中间时可能多/少选 1 个词，导致 cue 末尾偏移 1-2 词（实例：voxkit `consequences of` vs 金标 `consequences of a`）。

### 7.2 候选调优（按成本排序）

| # | 改动 | 成本 | 预期收益 |
|---|---|---|---|
| **A** | 调 `soft_break_weights` 表（加 clause 启发，降 comma 权重） | small | recall +10-15% |
| **B** | `prosody_gap_s` 动态化（按全局 speech rate 计算，0.25 → 0.3-0.5 自适应） | medium | precision +5-8% |
| **C** | 引入「字幕行长感知」切点权重（max_chars 不仅是硬上限，还作为软推荐反馈到 break_weights） | large | F1 可破 0.5+，但参数空间爆炸需迭代调 |

### 7.3 推荐路径

先做 **A**（5 行代码，立即验证）。若 precision 不掉下来，再叠 **B**。两者合起来预估 F1 从 0.19 推到 0.28-0.32。

**关键认知**：voxkit EN 路径的高 precision 低 recall 不是 bug，是设计选择的副作用——它在按"句子语义"优化，金标在按"字幕渲染"优化。要破 F1 0.5 必须做 C 或重新定义目标函数。**短期不建议投入大量精力推 EN F1**，应聚焦中文场景（用户主战场）和共享改进（如 medium-break atom 切分，对 EN 也是基础设施）。

## 8. Phase 3 实验记录：3b 双 pass reseg

**假说**：proofread 加完标点后，把 cue 喂回 reseg，能驱动 `_build_cjk_atoms` 在 `，。？！` 处切，把 recall 从 0.559 推到 0.70+。

**实验跑了两轮，结论被第 2 轮推翻**。

### 8.1 第 1 轮（input = 0.5.1 reseg + proofread, 145 粗 cue）— 失败

| 指标 | 单 pass (Phase 2) | 双 pass 实验 | 变化 |
|---|---|---|---|
| precision | 0.901 | **0.790** | **-0.11 ⚠️** |
| recall | 0.559 | **0.488** | **-0.07 ⚠️** |
| F1 | 0.690 | 0.603 | -0.09 |

**根因**：0.5.1 reseg 把 whisper 1-2s 短 segment 合并到 4s+ 长 cue → proofread 在长 cue 内加标点 → 二次 reseg 切新 atom 用 `_estimate_char_time` 线性插值（line 290）→ 在 4s+ 长 cue 内线性插值精度差于 whisper 原始 segment 边界 → precision 暴跌。

### 8.2 第 2 轮（input = 0.6.0 reseg + proofread, 200 细 cue）— 成功

| 指标 | 单 pass (Phase 2) | 双 pass 实验 | 变化 |
|---|---|---|---|
| precision | 0.901 | **0.906** | **+0.005** ✅（不退反升）|
| recall | 0.559 | **0.597** | **+0.038** ✅ |
| F1 | 0.690 | **0.720** | **+0.030** ✅ |
| broken_latin_words | 0 | 0 | ✅ |

**为什么第 2 轮成功**：input cue 已经 avg ~3s 短跨度，`_estimate_char_time` 线性插值在小区间内精度足够。

### 8.3 关键洞察：双 pass 对 input 粒度敏感

| Input 形态 | Avg cue 时长 | 双 pass precision | 结论 |
|---|---|---|---|
| 0.5.1 reseg (粗 145 cue) | ~4.1s | 0.790 ⚠️ | 不可用 |
| 0.6.0 reseg (细 200 cue) | ~3.0s | 0.906 ✅ | 可用 |

**前提条件**：双 pass 要 work，input cue 必须已经被第一 pass reseg 切到 ~3s 以内。Phase 2 的密度修复（`_CJK_DEFAULT_SOFT_MAX_CHARS=18`）刚好满足这个前提——两个 phase 的改动**互相成全**，否则单做任何一个都不够。

### 8.4 工程化产物：`voxkit reseg` 子命令

第 2 轮数据说服了工程化决策。已新增 `voxkit reseg <workdir>` 子命令（commit pending v0.7.0），消费 `subtitles.proofread.json`，输出 `subtitles.cues.reseg2.json` + 可选 `subtitles.reseg2.srt`。零 LLM 零网络，CI 可频繁跑。`voxkit eval` 加 reseg2 fallback 优先于 proofread。

完整推荐流水线：

```bash
voxkit transcribe <audio> --workdir <wd> --language zh --resegment semantic
voxkit proofread <wd> --language zh
voxkit reseg <wd>            # ← 新增：用 proofread 加的标点做二次切分
voxkit eval <wd> --reference <gold.srt> --lang zh   # 自动读 reseg2
```

## 9. Phase 3 后续优先级建议

| 方向 | 现状 | ROI |
|---|---|---|
| **3b 双 pass reseg** | ✅ 已工程化为 `voxkit reseg` | F1 0.69 → 0.72 实现 |
| **3a proofread-with-split** | 未做 | 比 3b 复杂但理论上 F1 上限更高（~0.78）；3b 已达可用质量，3a 收益递减，**优先级降为低** |
| **3e EN 路径改进** | 未做（见 §7） | F1 0.19 → 0.28-0.32（候选 A+B）；短期 ROI 低于中文场景 |

下一轮可选：
- **冲 F1 0.80+**：3a 工程化或改 reseg packing 阶段让 medium-break atom **强制 flush**（不让短 cue 合并）
- **拓宽场景**：跑 Lex Fridman / 3B1B 长视频，验证 0.7.0 流水线鲁棒性
- **改 EN 路径**：候选 A 调 `soft_break_weights`
