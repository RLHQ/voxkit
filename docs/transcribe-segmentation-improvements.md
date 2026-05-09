# Whisper 转录切分改进方案

本文整理 `voxkit transcribe` 在 Whisper 转录后切分、合并和字幕重切上的现状、风险点与后续改进方向。

目标不是追求单一“最聪明”的切分算法，而是在三类产物之间保持清晰边界：

- `transcript.voxkit.json` / `transcript.raw.json`：ASR ground truth，尽量忠实保留 Whisper 输出和时间戳。
- `subtitles.srt` / `subtitles.vtt`：面向播放器的人类可读字幕。
- `subtitles.cues.json`：语义重切后的机读 cue 流，供下游播放器或预览 UI 直接消费。

## 当前方案

### 1. 长音频物理分块

实现位置：`src/voxkit/core/audio.py`

当前规则：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `CHUNK_THRESHOLD_SECS` | 900 | 小于等于 15 分钟不分块 |
| `CHUNK_DURATION_SECS` | 600 | 每个 chunk 10 分钟 |
| `CHUNK_OVERLAP_SECS` | 5 | 相邻 chunk 重叠 5 秒 |

分块只用于控制 Whisper 处理长音频的稳定性和 checkpoint 粒度。切分点目前是固定时间网格：

```text
chunk 0: [0, 600)
chunk 1: [595, 1195)
chunk 2: [1190, 1790)
...
```

### 2. 单 chunk 内 ASR segment 重组

实现位置：`src/voxkit/core/segmenter.py`

英文和其他非 CJK 语言尽量启用 word timestamp：

- whisper.cpp 参数：`--max-len 1 --split-on-word`
- 每个 Whisper entry 约等于一个 word
- 按 4 个条件聚合成 `TranscriptSegment`

边界条件：

| 条件 | 当前阈值 |
|---|---:|
| 句末标点 | `. ! ? 。 ！ ？` |
| 下一个 entry 的停顿 | `> 500ms` |
| segment 累计时长 | `> 5s` |
| segment 累计字符 | `> 100` |

CJK 语言当前走 phrase 模式：

- whisper.cpp 不输出可靠 word-level timestamp
- 每个 Whisper phrase entry 直接变成一个 `TranscriptSegment`
- `words=[]`

### 3. 多 chunk 合并

实现位置：`src/voxkit/core/asr_merge.py`

当前使用 signal-aware overlap arbitration：

1. 把每个 chunk 内部的 segment 和 word 时间戳偏移到全局时间。
2. 在 chunk 接缝处计算重叠区。
3. 比较上一块和当前块在重叠区的“信号量”。
4. 信号更强的一侧保留重叠区，另一侧只保留 overlap 之外的内容。

信号量规则：

- 有 `words` 时用 word count。
- 无 `words` 时回退到 `text.strip()` 字符数。

这比简单的“后一块优先”或“前一块优先”更稳。后一块优先可能删掉上一块末尾完整识别出的长词；前一块优先可能错过当前块在 overlap 区更完整的标点和句子。

### 4. 字幕语义重切

实现位置：`src/voxkit/core/semantic_resegment.py`

`--resegment=semantic` 只影响字幕层，不反写 `transcript.raw.json`。

英文路径：

1. 展平所有 word，保留 `start` / `end` / `speaker`。
2. 按连续 speaker 分块，避免跨说话人合并。
3. 使用 `pysbd` 做句子边界识别。
4. 通过字符 span 映射回 word 区间。
5. 长句按标点、连词、韵律 gap 贪婪拆分。
6. 过短 cue 合并到同 speaker 邻居。
7. 钳住时间线单调递增。

当前字幕参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `max_dur_s` | 7.0 | 单条字幕最大时长 |
| `min_dur_s` | 1.5 | 小于该值尝试合并 |
| `max_chars` | 84 | 约 2 行 x 42 字符 |
| `soft_max_chars` | 75 | 软上限 |
| `max_cps` | 22.0 | 每秒字符数上限 |
| `prosody_gap_s` | 0.25 | 可作为软切点的停顿 |

CJK 路径：

- 不做 pysbd 句子级重切。
- 仍然会合并同 speaker 的过短 cue，减少闪现字幕。
- 长 segment 暂不拆分。

## 当前方案的优点

1. ASR 和字幕渲染解耦。

   `transcript.raw.json` 保留原始 ASR 粒度，`subtitles.cues.json` 承载字幕层语义重切。这样不会把播放器展示策略反向污染 transcript，也不会误触发下游 proofread 状态。

2. 边界丢词已有防护。

   固定 overlap、`--no-speech-thold 0.85`、signal-aware merge 共同降低 chunk 接缝处被截断或重复的概率。

3. 英文字幕语义质量较好。

   `pysbd + word timestamp + 物理约束` 是一个轻量但可审计的组合，避免直接依赖 LLM 对全文重新切句。

4. 有清晰的降级路径。

   没有 word timestamp、没有 `pysbd`、CJK 输入等情况都会退回可解释的 pass-through 或短 cue 合并。

## 已知不足

### 1. 物理 chunk 边界仍是固定网格

固定 `[0, 600), [595, 1195)` 这种切法简单可靠，但不理解音频内容。切点可能落在一句话中间、一个长词中间、一次快速插话中间。

当前 overlap 和 merge 能补救一部分问题，但它们属于事后修复。更理想的做法是在切 chunk 前就尽量选择静音或低能量点。

### 2. CJK 语义重切弱于英文

CJK 缺少 word-level timestamp，当前无法像英文那样做“句子边界 -> word 时间反查”。因此 CJK 只做短 cue 合并，不做完整句子级重切。

这能解决闪现字幕，但不能解决这些问题：

- 一条 cue 内含多个中文句子。
- 句子被 Whisper phrase 边界切开。
- 长 cue 内部没有更自然的阅读断点。

### 3. 语义重切默认关闭

当前 `--resegment` 默认是 `none`。这对保守兼容有利，但如果用户目标是播放器字幕，默认体验不一定最好。

### 4. overlap 参数不可通过 CLI 调整

`VOXKIT_CHUNK_*` 环境变量只作为诊断 hatch 存在。生产 CLI 不暴露 chunk 参数，意味着用户遇到特殊音频时只能通过环境变量 A/B。

### 5. 缺少切分质量指标

目前有功能测试和部分实验说明，但 pipeline 输出里还没有稳定记录字幕质量统计，例如：

- cue 数量
- 平均 cue 时长
- `<1.5s` 闪现率
- `>7s` 长 cue 比例
- 字符数超限比例
- CPS 超限比例
- chunk 接缝附近 merge notes 数量

缺少这些指标时，很难系统比较不同切分策略。

## 改进方案

### 方案 A：VAD/静音对齐 chunk 边界

优先级：高

核心思路：

仍以 600s 为目标 chunk 长度，但不直接在 600s 处切。对每个目标边界，在一个搜索窗口内寻找最适合的音频切点。

建议参数：

| 参数 | 建议值 |
|---|---:|
| 目标 chunk 长度 | 600s |
| 搜索窗口 | `target_boundary ± 20s` |
| 最小静音时长 | 300ms |
| 最大允许偏移 | 30s |
| fallback overlap | 5s 或 8s |

候选切点来源：

1. 优先使用 Silero VAD 的 non-speech 区间。
2. 如果没有 VAD 模型，使用 ffmpeg `silencedetect` 或简单 RMS 能量扫描。
3. 如果窗口内找不到静音，回退到固定网格。

切点评分：

```text
score = silence_duration_bonus
      - abs(candidate - target_boundary) * distance_penalty
      - speech_overlap_penalty
```

预期收益：

- 从源头减少句子踩 chunk 边界的概率。
- 减少 merge 阶段的重复/裁剪压力。
- 对中英文都有效。

实现建议：

1. 新增 `BoundaryPlan` 或扩展 `ChunkPlan`，记录每个边界的选择原因。
2. 在 `manifest.json` 里记录 `chunking.strategy`、`targetBoundarySecs`、`actualBoundarySecs`、`reason`。
3. 保持默认策略可回滚：`fixed-grid` 和 `vad-aligned` 两种模式并存。

验收指标：

- 接缝附近 `mergeNotes` 不增加。
- 接缝前后 3s 内的重复文本率下降。
- 人工抽样中边界截断明显减少。
- 相同音频多次运行切点稳定。

### 方案 B：CJK 字符级语义重切

优先级：高

核心思路：

CJK 没有 word timestamp，但可以在 segment 级时间内做字符级近似插值，再结合中文标点和停顿进行字幕重切。

输入：

- `RemixrSegment[]`
- 每个 segment 的 `start` / `end` / `text` / `speaker`
- 相邻 segment 间的 gap

切分边界：

| 边界类型 | 优先级 |
|---|---:|
| `。！？!?` | 最高 |
| `；;` | 高 |
| `，、：:` | 中 |
| 相邻 segment gap `>= 250ms` | 中高 |
| 字符数/时长/CPS 超限 | 兜底 |

时间插值方式：

```text
char_time(i) = segment.start + (i / len(text)) * (segment.end - segment.start)
```

如果一个 cue 跨多个原始 segment：

- cue start = 第一个 segment/字符估算 start
- cue end = 最后一个 segment/字符估算 end
- speaker 必须一致，否则强制断开

推荐约束：

| 参数 | 建议值 |
|---|---:|
| `max_dur_s` | 7.0 |
| `min_dur_s` | 1.5 |
| `max_chars` | 42 到 56 |
| `max_cps` | 14 到 18 |
| `prosody_gap_s` | 0.25 |

预期收益：

- 中文字幕从“Whisper phrase 粒度”提升到“句子/子句粒度”。
- 继续减少闪现字幕。
- 对中文播客、访谈、课程类内容尤其明显。

风险：

- 字符级时间是估算，不是 ASR 原生 word timestamp。
- 快速语速或长停顿混在同一 segment 中时，局部时间可能偏差。
- 只能用于字幕层，不应写回 `transcript.raw.json.words`。

实现建议：

1. 在 `semantic_resegment.py` 内新增 `_resegment_cjk_char_level`。
2. 保留当前 CJK pass-through 作为 fallback。
3. 在 `subtitles.cues.json.params` 中标记 `timebase: "char-interpolated"`。
4. 文档明确：CJK 字符级时间仅用于字幕渲染，不是精确 word alignment。

验收指标：

- `<1.5s` cue 比例保持接近 0。
- `>7s` cue 比例下降。
- 平均 cue 字符数落在 12 到 32 个中文字符附近。
- 人工抽样中不出现明显“字幕提前/滞后整句”的情况。

### 方案 C：将 semantic resegment 作为字幕默认策略

优先级：中

核心思路：

保持 transcript 默认保真，但字幕默认走语义重切。

可选路径：

1. 保守路径：CLI 默认仍是 `--resegment none`，但 README 和示例全部推荐 `--resegment semantic`。
2. 产品路径：将 `voxkit transcribe` 默认改为 `--resegment semantic`，提供 `--resegment none` 回退。
3. 分层路径：`transcript.raw.json` 永远不重切，`subtitles.*` 默认重切，`manifest.subtitle.resegment` 明确记录。

推荐采用第 3 种。它符合当前架构边界，也给播放器用户更好的默认体验。

迁移注意：

- 这会改变默认 SRT/VTT 的 cue 数量和边界。
- 下游 snapshot test 可能需要更新。
- 必须保证 `subtitles.cues.json` 一并产出，避免下游反解 SRT。

验收指标：

- 默认命令输出更适合直接播放。
- `transcript.raw.json` byte-level 行为不被改变。
- `manifest.subtitle.resegment` 能解释每次运行采用的策略。

### 方案 D：外露 chunk 参数和 resegment 参数

优先级：中

核心思路：

把当前只能通过环境变量或代码修改的参数，以受控方式暴露给 CLI，方便 A/B 和特殊音频调优。

建议新增参数：

```bash
voxkit transcribe input.mp4 \
  --chunk-strategy vad-aligned \
  --chunk-secs 600 \
  --chunk-overlap-secs 8 \
  --resegment semantic \
  --subtitle-max-dur 7 \
  --subtitle-min-dur 1.5 \
  --subtitle-max-chars 84
```

设计原则：

- 保留当前默认值。
- 参数必须写入 `manifest.json`。
- 对明显危险的值做校验，例如 overlap 必须小于 chunk。
- 不把太多实验参数一次性暴露。先开放稳定且用户能理解的几个。

验收指标：

- 不设置参数时行为完全兼容。
- 设置参数后 manifest 可复现。
- CLI help 能说明参数只影响 chunk 或字幕层。

### 方案 E：字幕质量统计与回归评估

优先级：高

核心思路：

在每次运行中输出可比较的切分统计，先不改算法，也能让后续优化有尺子。

建议统计：

| 指标 | 说明 |
|---|---|
| `cueCount` | 字幕 cue 数 |
| `avgCueDurS` | 平均 cue 时长 |
| `p50CueDurS` / `p90CueDurS` | 时长分位数 |
| `flashCueRate` | `< min_dur_s` 的 cue 比例 |
| `longCueRate` | `> max_dur_s` 的 cue 比例 |
| `avgChars` | 平均字符数 |
| `overCharLimitRate` | 超字符上限比例 |
| `overCpsRate` | 超 CPS 比例 |
| `speakerBoundaryMerges` | 尝试跨 speaker 合并被阻止的次数 |
| `monotonicClamps` | 时间线钳位次数 |

输出位置：

- `manifest.subtitle.metrics`
- 可选：`subtitles.cues.json.metrics`

实现建议：

1. 新增纯函数 `compute_subtitle_metrics(cues, params)`。
2. 在 `semantic_resegment.py` 或 `io/cues_json.py` 附近放置，避免 pipeline 变胖。
3. 增加 fixtures，覆盖英文、CJK、短 cue、长 cue、倒挂时间戳。

验收指标：

- 任意策略都能输出同一组 metrics。
- metrics 不影响字幕内容。
- 后续改算法时可以用 metrics 做自动回归。

### 方案 F：LLM/标点恢复辅助重切

优先级：低，实验性

核心思路：

对标点少、句子边界弱的内容，可以用 LLM 或轻量标点恢复模型辅助判断语义边界。

适用场景：

- Whisper 输出缺少标点。
- 口语长句很多。
- 需要更接近人工字幕的断句。

不建议作为默认方案，原因：

- 成本和延迟更高。
- 可复现性弱于规则算法。
- 需要额外隐私和离线/在线策略。
- 容易把“修正文案”和“切字幕”混在一起。

如果做实验，应限制输出：

- 只允许返回边界位置，不允许改写文本。
- 所有边界必须映射回原始 token/字符 index。
- 必须保留 deterministic fallback。

## 推荐路线

### 第一阶段：建立尺子

先做方案 E。

原因：

- 风险最低。
- 不改变用户产物。
- 后续所有算法改进都有可比较指标。

交付：

- `compute_subtitle_metrics`
- `manifest.subtitle.metrics`
- 单元测试

### 第二阶段：减少源头截断

做方案 A。

原因：

- chunk 边界对所有语言都有影响。
- 能减少接缝处 merge 的压力。
- 与 semantic resegment 解耦，容易 A/B。

交付：

- `fixed-grid` / `vad-aligned` chunk strategy
- boundary 选择审计
- 回归测试和少量真实音频验证

### 第三阶段：补齐 CJK 字幕体验

做方案 B。

原因：

- 当前 CJK 是最明显的语义短板。
- 字符级估算足够用于字幕层，但必须明确不能污染 ASR ground truth。

交付：

- CJK char-level resegment
- `timebase: "char-interpolated"`
- 中文 fixtures 和人工抽样

### 第四阶段：调整默认体验

做方案 C 和 D 的稳定子集。

原因：

- 等 metrics 和 CJK 行为稳定后，再考虑默认打开 semantic。
- CLI 参数外露应服务于真实调优需求，而不是把内部参数全部甩给用户。

## 非目标

以下事情不建议放进本轮切分优化：

- 不把 semantic cues 回写到 `transcript.raw.json.segments[].subtitles`。
- 不把 CJK 字符插值伪装成 word timestamp。
- 不让 LLM 改写 transcript 文本。
- 不为了字幕好看而改变 ASR segment 的 ground truth。
- 不在没有 metrics 的情况下盲目调默认阈值。

## 决策摘要

当前方案已经是可用且稳健的基线，尤其是 overlap merge 和英文 semantic resegment。但它仍有两个明显升级空间：

1. chunk 边界应该尽量对齐静音，而不是固定时间网格。
2. CJK 字幕应该进入字符级语义重切，而不是只做短 cue 合并。

推荐先补 metrics，再做 VAD-aligned chunk，最后做 CJK char-level resegment。这样每一步都有回滚点，也能保持 `transcript.raw.json` 的保真边界不被打破。
