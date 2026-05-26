# Changelog

All notable changes to voxkit. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project does NOT follow strict semver until 1.0.0; minor versions may include breaking
changes (with migration notes).

## [Unreleased]

### Added

- **`--max-cue-duration` 暴露到 `transcribe` 与 `reseg`**（F3）：透传到
  `ResegmentParams.max_dur_s` 作为语义切分的 trigger 阈值——仍走语义切分，
  仅调紧/调松触发条件，不走硬性时长强切。`<=0` 由 CLI + pipeline 双层 reject。
  默认沿用 `ResegmentParams()` 的 `7.0s`。
- **超长 cue 双 pass 引导**（F3）：transcribe / reseg 渲染完 cue 后检测
  `max(cue.dur) > max_cue_duration × 1.5`：
  - 普通模式 stderr 输出引导文案（`transcribe` 推荐 `voxkit proofread` +
    `voxkit reseg` 双 pass workflow；`reseg` 推荐调紧 `--max-cue-duration`
    或检查 proofread 标点质量）。
  - `--json-events` 模式发 `long_cues_detected` NDJSON 事件（`count` /
    `longestSecs` / `thresholdSecs`），同时镜像写入 `events.ndjson`。
  - warning 文案进入 `voxkit_out.warnings` 与 manifest `warnings`。
  - 阈值 = `max_cue_duration × 1.5`（与切分算法内的 `_HARD_DUR_RATIO=1.2`
    刻意区分；这里是用户提示阈值，那里是算法硬上限）。

## [0.7.4] — 2026-05-26

3 个并行 sub-agent 一次性消化 v0.7.1 反馈剩下的 F2 / U1 / F4 / U3 4 项。每个
agent 在独立 git worktree 跑，main 上顺序 cherry-pick。无 schema 变更。

### Added

- **`voxkit needs-review <workdir>` 子命令** — 列出 proofread / translate
  artifact 中需要人工复核的 cue。默认过滤 `needsHumanReview=True` 或
  `risk ∈ {high, blocking}`；支持 `--target <lang>`（读
  `subtitles.<lang>.json`，缺失回落 `subtitles.proofread.json`）、
  `--format text|json`、`--include-risk` 覆盖默认 risk 集合。read-only，
  不写盘 / 不 lock workspace。stderr 写 `N cue(s) flagged out of M total`
  summary，stdout 输出队列。回应下游反馈 U3：之前需要 `jq` 自己挖
  `cues[] | select(.needsReview)`。
- **U1：proofread / translate summary 行加 token 拆分 + cost 估算**。
  老格式 `64158 + 74140 tokens` 不分输入输出、也不打 USD。新增格式：

  ```
  proofread done: cues=9% changed, 3% need review
    tokens: prompt=64158, completion=74140 (total=138298)
    est cost: ~$0.10 (deepseek/deepseek-v4-flash @ $0.27 + $1.10 per M)
  ```

  未知 (provider, model) 组合显示 `est cost: (unknown rate for <p>/<m>)`，
  不阻塞 pipeline。translate 同形。
- **F4：`voxkit proofread --dry-run` / `voxkit translate --dry-run`**。
  跑完 batch 切分 + token / cost 估算后退出，**不调 LLM、不获 workspace lock、
  不写盘**。专为批量跑视频前先看一眼"这 47 分钟的会议大概要烧多少美刀"。
  优先级高于 `--render-only` / `--force*`（dry-run 是只读操作，stale `.lock`
  也不会挡）。
- 新增 `voxkit.core.pricing` 模块：中心化 `(provider, model) → USD/M-token`
  价目表（当前只 `deepseek/deepseek-v4-flash`），公开 `estimate_cost()` /
  `format_cost()` 纯函数。加新 provider 只需在 `PRICING` dict 里追加一行。
- **`voxkit transcribe --initial-prompt <text>` / `--initial-prompt-file <path>`**
  — 透传 whisper-cli `--prompt`，作为上下文先验抑制专名同音词 typo
  （F2 反馈：whisper 把 "Claude" 反复听成 "Cloud"、"Anthropic" 听成
  "anthropoid"）。两个 flag 互斥（argparse mutex group）；string 直接传，
  file 读 UTF-8 文本（适合 JSON glossary 衍生出的长 prompt）。
  - **chunk 透传**：whisper-cli 是 stateless 子进程，每个 chunk 都重传 prompt。
  - **长度守护**：超过 1000 char 自动截断 + warn（不 hard fail），对齐
    whisper-cli ~224 token 的内部上限。
  - **manifest 审计**：只写 `initialPromptUsed: bool` + `initialPromptChars: int`，
    不落明文（隐私 + manifest 体积）。
  - 用法示例：
    ```bash
    voxkit transcribe talk.mp4 --workdir /tmp/wd \
      --initial-prompt "Claude, Anthropic, MCP, Sonnet, Opus, Haiku."
    # 或从 glossary 衍生
    voxkit transcribe talk.mp4 --workdir /tmp/wd \
      --initial-prompt-file ./glossary.prompt.txt
    ```

## [0.7.3] — 2026-05-26

针对 v0.7.1 下游反馈的 P1 UX 改进（U2 / U4 / U6 / U7），全部是非破坏性
ergonomic 调整。无 schema 变更。

### Added

- **U2：proofread / translate 普通模式下输出 per-batch 进度行**。每个
  batch.done 写一行 "proofread: batch 12/45 done (480/1820 cues, 26%; 9%
  changed so far) [cache hit]"。json_events 模式下不打（NDJSON 已覆盖）。
- **U4：4 个核心命令完成后输出 "next steps:" 导览**。
  - `voxkit transcribe`：提示 `quality`，以及（带 `--resegment=semantic` 时）
    `proofread` / `translate`。
  - `voxkit proofread`：提示 `reseg` 双 pass、`translate`、`quality`、
    `review confirm`。
  - `voxkit translate`：提示 `quality`、`--render-only --speaker-prefix`、
    `review confirm --target`。
  - `voxkit reseg`：提示 `quality`、`translate`、`eval`。
  - json_events 模式跳过。
- **U6：proofread / translate `--help` epilog 列出 LLM provider 注册表**。
  从 `voxkit.llm.providers.PROVIDERS` 自动渲染（当前只 `deepseek`），含
  env var 名 + default model；以后追加 provider 不用手动同步 help。
- **U7：`--help` epilog 多了 --force 三档对应表**。让"reviewed 该用哪个 flag"
  这种问题 5 秒能查到，不必翻 CLAUDE.md。

## [0.7.2] — 2026-05-26

针对 v0.7.1 下游反馈（`code-with-claude/voxkit-feedback.md`）的 P0 4 项 bug
全部修复 + code-review 暴露的 6 个补强一并搞定，无 schema 变更，下游 reader
不用改。详细回应见 `code-with-claude/voxkit-feedback-response.md`。

### Added

- **`voxkit transcribe` / `voxkit reseg` 新增 `--speaker-prefix {auto, always,
  never}`** 与 `voxkit translate` 对齐（review #2）。三个命令共享同一套语义，
  下游想全局回退到 v0.7.1 行为只需统一传 `always`。
- **`voxkit translate --render-only`**（review #3）：跳过 LLM 与 cache，仅根
  据现有 `subtitles.<lang>.json` 重渲染 SRT/VTT。专为"只想换 `--speaker-prefix`
  / 格式参数"场景，避免被迫 `--force` 重 LLM 浪费 token。
- 公共导出 `voxkit.io.srt.is_informative_speaker` + `PLACEHOLDER_SPEAKERS`：
  下游想做自己的占位符过滤可以复用。

### Fixed

- **B1：SRT 不再无脑加 "Speaker A:" 前缀**（修复扩到所有命令）。
  - `voxkit translate`：新增 `--speaker-prefix {auto, always, never}`，默认
    `auto` 仅在 cue 实际含 ≥2 个不同**信息性** speaker 时渲染前缀。
  - `voxkit transcribe`：**segment path** 默认 `auto`（segment schema 无 speaker
    信息，等同"全占位符"，auto = 不渲染）；**cue path**
    （`--resegment=semantic`）同样默认 `auto`。 → 修补了 0.7.2 review #1。
  - `voxkit reseg`：reseg2.srt 同样默认 `auto`。
  - **`Speaker A` / `Speaker ?` 公认为占位符**：不计入 distinct count，per-cue
    渲染时单独跳过（修补 0.7.2 review #5；多 speaker 场景下未匹配 cue 不会再
    漏出 `"Speaker ?: [cough]"`）。
  - `always` 等同 0.7.1 之前的旧行为。
- **B2：silero VAD 吃开场不再静默**。`voxkit transcribe` 在 VAD 实际生效
  且首条 segment 起点 > 15s 时，把 warning 写入 manifest + stderr。**文案
  弱化因果断言**（review #4）：从 "VAD trimmed first {N}s as non-speech"
  改成 "first transcribed segment starts at {N}s with VAD on; if real speech
  starts earlier, silero VAD may have trimmed it — rerun with --no-vad to
  verify"，避免在 intro music / 真实静默场景下误导用户。
- **B3：`voxkit translate` 在缺 cues.json 时报错带修复命令**。错误信息现在
  明确指向 `voxkit transcribe <input> --workdir <dir> --resegment=semantic`。
- **B5：`voxkit doctor` whisper-cli 探测超时 5s → 15s**。macOS Metal 初始化
  典型 ~8s，旧的 5s 超时在 brew-installed whisper-cpp + Metal 后端上会假阴性。
- 删除 `transcribe_pipeline.py` 里 VAD warning 分支的冗余 `import sys`（review
  #6；模块顶层已 import）。

## [0.7.1] — 2026-05-25

### Fixed

- **`semantic_resegment` 死循环修复**：`_split_long` 与 `_split_cjk_long`
  在递归保护处增加 shrinkage 守卫（`len(chunk) < n`）。原算法在子 chunk
  没真正变短时仍会递归，导致两类病态输入触发 `RecursionError`：
  - **EN 路径**：whisper-cli 偶发把长尾静音锁进单词的 `end`，使单 word
    `dur > max_dur_s × _HARD_DUR_RATIO`（默认 8.4s）→ 主循环所有候选切点
    被 hard-ratio 全数拒绝 → `chunks=[整段]` → 无限递归。
  - **CJK 路径**：极短文本 + 极长 dur（如 `n_by_dur > total_chars`）→
    主循环 `start >= total_chars - min_remaining` 立即 break → 同上。
  - 新增两条回归测试覆盖两条路径。

## [0.7.0] — 2026-05-13

**双 pass reseg：`voxkit reseg` 子命令** + CJK atom 切分覆盖 medium 标点
（，、：:）。基于小宁子 10min 实拉对照人工金标：

| 指标            | 0.6.0   | 0.7.0   | 变化           |
|----------------|---------|---------|---------------|
| vk_cues        | 200     | 210     | +5%           |
| precision      | 0.901   | **0.906** | +0.005（升） |
| recall         | 0.559   | **0.597** | **+0.038**   |
| F1             | 0.690   | **0.720** | **+0.030**   |
| chars_drift    | +4.37   | +4.73   | ≈            |
| broken_latin   | 0       | 0       | ✅            |

### Added

- **`voxkit reseg` 子命令** — 读 `subtitles.proofread.json`，把 corrected
  cue 当带标点 ASR segment 喂回 `semantic_resegment`，输出
  `subtitles.cues.reseg2.json` + 可选 `subtitles.reseg2.srt`。零 LLM 零
  网络，CI 可频繁跑。完整推荐流水线：
  ```bash
  voxkit transcribe ... --resegment semantic
  voxkit proofread <wd>
  voxkit reseg <wd>                  # ← 新增
  voxkit eval <wd> --reference ...   # 自动读 reseg2
  ```
- **`voxkit eval` 加 reseg2 fallback**：load_voxkit_cues 优先级
  `reseg2 > proofread > cues`，自动消费 `voxkit reseg` 产物。
- **9 个 reseg 命令单测** 覆盖：含逗号长 cue 切分、SRT 渲染、speaker 保留、
  错误路径（缺 proofread / 缺 workdir / 拒覆盖）、eval fallback 优先级。

### Changed

- **`_build_cjk_atoms` 加入 `_CJK_MEDIUM_BREAK = ，、：:` 切点**
  （commit 8a9ce2a，v0.7.0 前置）。whisper 中文 ASR 输出无标点时无影响，
  但带标点输入（proofread 后 / prompt 引导 ASR）下逗号承载 ~80% 气口
  边界，是双 pass reseg 工作的前提。1 个新单测覆盖。

### Design rationale（详见 docs/eval-baseline-observations.md §8）

双 pass reseg 对 input cue 粒度敏感：

| Input 形态 | Avg cue 时长 | 双 pass precision |
|---|---|---|
| 0.5.1 reseg (粗 145 cue) | ~4.1s | 0.790 ⚠️（不可用）|
| 0.6.0 reseg (细 200 cue) | ~3.0s | 0.906 ✅（可用）|

`_estimate_char_time` 线性插值在 4s+ 长 cue 内会丢精度，但在 ~3s 短 cue
内足够。Phase 2（v0.6.0 收紧 `_CJK_DEFAULT_SOFT_MAX_CHARS=18`）和本期
（v0.7.0 双 pass）**互相成全**——单做任何一个都不够。

### Breaking change

无。`voxkit reseg` 是可选新命令，不改现有 `transcribe / proofread / eval`
行为。`subtitles.cues.json` / `subtitles.proofread.json` schema 未变。

## [0.6.0] — 2026-05-13

中文 reseg 切分粒度修复 + 拉丁词原子化。基于
`tests/fixtures/youtuber_samples/xnzxnz_first_look_10min/`（小宁子 10min）
对照人工金标实测：

- cue 密度 145 → 200（金标 302）：density_ratio 0.480 → **0.662**（+38%）
- 边界 recall 0.391 → **0.559**（+43%），precision 0.884 → 0.901（不退反升）
- 边界 F1 0.542 → **0.690**（+27%）
- 字符 / 时长漂移 +10.25 chars → +4.37、+2.21s → +1.10s（-50% 以上）
- 跨 cue 拉丁词切断 1 → **0**（'Steam' 不再被切成 'S t' + 'eam'）

### Added

- **`voxkit eval` 子命令** — 对照人类金标 SRT 评估 voxkit 输出 reseg 质量。
  输出 `eval.report.json`，含 cue 密度比、边界 precision/recall/F1、
  字符/时长 drift、跨 cue 拉丁词切断数。纯计算零 LLM，CI 可频繁跑。
- **`tests/fixtures/youtuber_samples/`** — 4 个「认真做字幕的 YouTuber」
  实拉样本（小宁子 zh+en 同帧对齐、Kurzgesagt en、Lex Fridman en、
  3Blue1Brown 11 语对照）作为人类金标 fixture。
- **`tests/fixtures/youtuber_samples/xnzxnz_first_look_10min/baseline.eval.json`**
  — voxkit 0.6.0 在该 fixture 上的 eval 基线，后续改动可 diff 看进退。
- `docs/eval.md` 设计稿、`docs/eval-baseline-observations.md` 基线观察、
  `docs/认真做字幕的YouTuber.md` 频道筛选标准。
- 2 个 CJK reseg 回归测试（拉丁词原子化 + vlog 风格无标点中文密度）。

### Changed

- **`semantic_resegment._CJK_DEFAULT_SOFT_MAX_CHARS`** 28 → **18**。
  whisper.cpp 中文 ASR 不带标点，原 28 字符目标会把多个气口合并成一个
  长 cue（小宁子样本 avg 23 char/cue vs 人工金标 11）。**这是 breaking
  change**：所有现有中文 fixture 的 reseg 输出会变得更细切，cue 数典型
  +30~40%；字幕渲染观感更贴近人工。
- **`semantic_resegment._split_cjk_long`** — 候选切点循环跳过拉丁词内部，
  fallback 物理切点也会往后调整到最近非拉丁边界。修复 7s+ 长 CJK segment
  做字符级切分时把 'Steam' 切成 'S t' + 'eam' 的 P1 bug。

### Migration notes

- 用户如希望保留 0.5.1 的旧粒度，可在 `transcribe` 调用时构造
  `ResegmentParams(soft_max_chars=28)` 显式覆盖——但建议先用 `voxkit eval`
  对照金标看新默认是否更接近你的字幕风格。
- 现有 `subtitles.cues.json` artifact schema **未变**；重跑 transcribe 即可
  获得新粒度产物。

## [0.5.1] — 2026-05-12

字幕切分质量修复。基于 `tmp/e2e_test/` 90 秒样本实测：
trailing-bad 收尾 40% → 12%、闪屏 cue 4% → 0%、跨 cue "Is it" 重复 1 → 0。

### Added

- **`voxkit.core.word_classes`** — 共享词集 `ENGLISH_TRAILING_BAD`（~150 词，
  涵盖介词/冠词/连词/助动词/缩略 will-would/弱代词/疑问代词），配合
  `is_trailing_bad(token)` helper。任何带停顿标点（`.!?,;:`）的 token 都自动
  豁免（标点 = 合法停顿点）。
- **`SubtitlePhysicalMetrics` 新增 3 个切分质量指标**：
  - `trailingBadWordRate` — 末尾停在介词/连词的 cue 比例（CJK 主体跳过）
  - `singleWordCueRate` — 单 token cue 比例
  - `crossCueRepeatRate` — 相邻 cue 末尾 1-3 词与下一 cue 开头 1-3 词重复的比例
    （proofread 错误闭合切坏边界的典型征兆）
- 7 个回归测试覆盖切分点偏好、闪屏合并、跨 cue 重复检测、CJK 路径不受影响。

### Changed

- **`semantic_resegment._compute_break_weights`** — 给 trailing-bad token 后的
  软切点打 0.2× 折扣；标点/句末等强切点不受影响。结果：长句 split_long 优先
  避开介词/连词后切分。
- **`semantic_resegment._can_merge`** — 极短 cue (< 0.5s) 触发物理上限放宽
  (cps × 1.5、chars × 1.2)。修复 e2e cue_000008 "I'll" 0.17s 闪屏因 cps 22.4
  > 22.0 被拒合并的根因。`max_dur_s` 不放宽。
- **`proofread.v1.md`** 加 3 条硬约束：保留切坏边界（不补造句末标点）、不在
  相邻 cue 复制重叠词（修 "Is it" 重复）、不给被切断的子句各自加问号。
  promptHash 自动失效旧 checkpoint，rerun 会全 batch 重做（属期望行为）。
- **`translate.v1.md`** 加 2 条硬约束：保留源端切坏的不完整尾部、跨 cue 连读
  自然（不强行画句号让两段割裂）。仍保持 `cueMappingPolicy=one-to-one`，
  group-within-speaker 留 v0.6+。

### 不在本版本

- `cueMappingPolicy=group-within-speaker`（v0.6+，doc §未来扩展能力）
- 未引入 spaCy / 重型 NLP（用户选定手写词表路线，覆盖 80% 真实场景）

## [0.5.0] — 2026-05-12

LLM 安全性硬化：cache 失效完整化、批级 transport 容错、reviewed/final 防误覆盖。
端到端跑通 deepseek-v4-flash（默认 model）。Codex 独立审查 + 工程师 review 合并修复。

### Added

- **`voxkit.core.lifecycle.gate_force_overwrite` + `ForceLevel`** — proofread/translate
  共用的 force-gate；按 artifact `state` 分三档拒覆盖（draft/reviewed/final）。
- **`--force-reviewed` / `--force-final`** CLI flag —— 必须显式声明才能覆盖人工
  confirm/lock 的产物（`--force` 默认只覆盖 draft）。
- **批级 transport 错误处理** —— `LLMTimeout` / `LLMRateLimit` 单批失败写
  `batch_NNN.pending.json` marker；run 末尾若有 pending → 拒绝写稳定 artifact，
  rerun 只补失败批，已完成 checkpoint 自动复用。
- **manifest cost 拆分** —— 新增 `freshPromptTokens` / `freshCompletionTokens` /
  `cachedPromptTokens` / `cachedCompletionTokens`，区分本轮真的花掉 vs checkpoint
  复用；`promptTokens` / `completionTokens` 总和保留兼容旧消费者。
- **manifest `outputArtifact` / `outputSchemaVersion`** — proofread / translations.<lang>
  各自补上输出 artifact 路径 + schema 版本，便于 freshness 判断。
- **`SYSTEM_OVERHEAD_TOKENS = 600`** — `_build_batches` 在切批时扣掉 system prompt /
  context cue / completion 余量，防止 batch 撞 context window。
- 4 处回归测试覆盖 force gate / cache key 失效条件 / 空文本拒收 / transport 续跑。

### Changed

- **proofread/translate cache key 改成 `(contentHash, policyHash, cacheSchema)`** —
  `contentHash` 入键 `(id, text, start, end, speaker)`（之前只有 id+text，会让 stale
  时间轴/speaker 被旧 cache 覆盖）；`policyHash` 集中所有影响 LLM 输出的策略
  （provider/model/promptVersion/promptHash/editLevel 或 style/lengthPolicy/
  cueMappingPolicy/glossaryHash/sourceLanguage/targetLanguage）。任一变化即让
  checkpoint 失效。`cacheSchema=2` 老 checkpoint 自动作废。
- **`--force` 不再预先 unlink 旧 artifact** —— 只清 `work/proofread/` 或
  `work/translate.<lang>/` checkpoint 目录；新批次全部完成后才 `os.replace` 原子
  替换。LLM 中途失败时旧 stable artifact 完整保留。
- **token 估算保守化** —— CJK 0.5 → 1.0 token/char（贴近 DeepSeek BPE 实际值），
  Latin 0.25 → 0.3。`_build_batches` 把 `context_prev/next` 实际 token 也算进预算
  （之前只看 cue 数量，长 CJK context 会撑爆）。
- **`quality.report.json` 的未知/缺失 `risk` → `blocking` 桶**（之前默认 `low`，
  malformed LLM 输出会悄悄通过审核）。
- **`subtitles.<lang>.srt` / `.vtt` 改原子写**（同 JSON 一致），失败时旧字幕保留。
- **proofread / translate 默认 LLM model** → `deepseek-v4-flash`（`deepseek-chat`
  2026-07-24 deprecated）。
- **`docs/capability-artifact-model.md`** —— 缓存键 / manifest 顶层布局 /
  blocking 语义 / `--force` 三档全部对齐当前实现。

### Fixed

- **schema validator 拒收空白 `correctedText` / `translatedText`** —— LLM 返回 `""`
  或纯空白会被 Pydantic 拒收 → 触发一次 repair → 仍空白则 fallback 标 `risk=blocking`
  + `needsHumanReview`，不再悄悄落到 stable draft。
- **`peek_artifact_state`** 用 try/except 替代 stat 预检查，消除 TOCTOU。
- **批级 transport except 收窄到 `(LLMTimeout, LLMRateLimit)`** —— 之前 catch 了
  `LLMError` 基类，会吞掉未来其它 LLM 子类异常。

### 已知遗留

- proofread/translate batch 主循环仍存在 ~110 行高度近似复制；`llm_batch_runner.py`
  通用化留作后续 PR（工程量较大）。
- `_atomic_write_text` / `_atomic_write_json` / `write_manifest` / `write_quality_report`
  应整合到 `voxkit/io/atomic.py`（Codex M1）；本轮未做。

### Added

- **`voxkit doctor --profile {transcribe,diarize,all}`** — first-run checks can now
  focus on the user's goal. `transcribe` treats whisper-cli, ffmpeg, and the
  ASR model as required while hiding pyannote/HF noise; `diarize` focuses on
  pyannote model readiness, HF/bundle state, ffmpeg, and the worker venv.
- **`subtitles.cues.json`** — render-layer machine-readable mirror of the
  semantic resegmenter output. Written only when `--resegment=semantic` and
  the resegment path actually produced cues (the diarized 1-cue-per-segment
  fallback is excluded). Schema:
  `{schemaVersion, sourceId, resegment, params, cues[{start,end,speaker,text}]}`.
  Lets downstream consumers (e.g. Remixr) ingest semantic cues directly
  instead of reverse-parsing SRT text. `transcript.raw.json` stays untouched
  — it is ASR ground truth; cues are render-layer derivatives, see
  `docs/transcribe.md` for the rationale.
- New events `write.subtitle_cues` (path + cue_count) and the existing
  `resegment.done` now bracket cues.json emission.
- `Workspace.cues_json_path`; `manifest.artifacts.subtitle_cues_json`
  populated when written.
- `voxkit.io.cues_json` module + `SubtitleCueOut` / `SubtitleCuesOutput`
  Pydantic models in `io/schema.py`.
- **CJK short-cue merge** — `--resegment=semantic` now also benefits CJK
  inputs: short cues (< `min_dur_s`, default 1.5s) merge into a same-speaker
  neighbour to eliminate flicker subtitles. Empirically validated on a
  106-min Mandarin podcast: 4426 → 2721 cues (−38.5%), avg duration
  1.43s → 2.33s, sub-1.5s cues 58.8% → 0%, no over-7s cues introduced.
  Implementation: opens the existing `_merge_too_short` to the CJK passthrough
  path; pysbd is still skipped (CJK has no word-level timestamps), so the
  passthrough → merge → monotonic chain is the entire CJK pipeline.
  Long-segment splitting in CJK remains unimplemented (YAGNI: segmenter's
  5s/100chars upper bound already gates this in practice).

### Changed

- `_ensure_raw_json_writable` also unlinks `subtitles.cues.json` on `--force`
  so the exclusive-create write does not collide on rerun.
- **CJK `--resegment=semantic`** is no longer a strict no-op — it now applies
  short-cue merging. Output `cue_count` may be lower than `segment_count`
  (previously they were equal). `transcript.raw.json` is unaffected.

## [0.3.0] — 2026-04-28

This release renames the project from `voxsplit` to `voxkit` and adds a `transcribe` subcommand
backed by whisper.cpp, repositioning voxkit as a "speech-to-structured-data toolkit" rather than
a single-purpose diarization CLI. The whisper.cpp invocation stack (chunking, anti-hallucination,
dual-mode segmentation, overlap dedup, checkpoint resume) is ported from Remixr's TypeScript
implementation, replacing ~1400 lines of `services/whisper.ts` with a Python equivalent that any
caller can hit via the CLI.

### Added

- **`voxkit transcribe`** subcommand — whisper.cpp ASR with the full anti-hallucination stack:
  - 3-layer defense: VAD (silero) + `--max-context 0` + `--logprob-thold -0.8`.
  - Long-audio chunking: 15-min threshold, 10-min chunks, 5s overlap, 0.5s overlap-dedup
    tolerance.
  - Word-level timestamps in English mode (`--max-len 1 --split-on-word`); CJK languages
    (`zh` / `ja` / `yue` / `ko`) auto-drop those flags and emit phrase-level segments with
    empty `words[]`.
  - Chinese hallucination blocklist (7 watermark prefixes + 19 standalone phrases + ghost CJK
    loop detection: ≥6-char CJK substring repeated ≥2 times).
  - NFC normalization before blocklist matching (whisper.cpp occasionally emits NFD).
  - Checkpoint resume via `work/chunks/chunk_NNN.json` cache; `--force` clears `work/`.
  - Dynamic per-chunk timeout: `max(30 min, duration * 0.3)`.
  - VAD model 3-level fallback: `--vad-model` flag → `WHISPER_VAD_MODEL_PATH` env →
    `/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin`; warn-once and disable VAD if
    none found.
  - whisper-cli discovery: `--whisper-bin` → `$WHISPER_BIN` → `which whisper-cli` →
    `/opt/homebrew/bin/whisper-cli`.
- **Workdir-based artifact layout** — data orthogonal, fully auditable, concurrent-safe via
  per-workdir PID lock:
  - `transcript.raw.json` — Remixr Zod-compatible (drop-in for
    `storage/projects/{projectId}/sources/{sourceId}/transcript.raw.json`); written with
    exclusive `wx` mode (re-running the same workdir errors loudly).
  - `transcript.voxkit.json` — rich native format with RTF, elapsed, perChunk stats,
    hallucinationDrops, warnings.
  - `subtitles.srt` + `subtitles.vtt` — segment-level cues (`--emit-srt` / `--emit-vtt`,
    both on by default).
  - `manifest.json` — input, args, voxkit version, whisper-cli version, start/end times,
    PID lock.
  - `events.ndjson` — mirror of stderr NDJSON event stream (always written).
  - `work/input.16khz.mono.wav` — ffmpeg-normalized master (`-ar 16000 -ac 1`).
  - `work/chunks/chunk_NNN.{wav,json,entries.json}` — per-chunk audio + raw whisper output +
    post-blocklist filtered entries.
  - `work/hallucinations.log` — NDJSON record of every dropped entry + reason code.
  - `work/merge.json` — per-chunk segment-id keep/drop decisions.
  - `work/timeline_validation.log` — warn-only timeline-continuity check output.
- **Remixr adapter** (`io/remixr_adapter.py`) — single point of truth for `transcript.raw.json`
  schema mapping. Embeds optional `_metadata` (voxkit version, asrBackend, asrModel, language,
  sourceDurationSecs, processedAt, whisperBin, vadModel, RTF, perChunk, warnings); Remixr
  ignores unknown fields, safe for audit.
- **SRT/VTT generators** (`io/srt.py`) — segment-level cues, `"Speaker A: "` placeholder until
  diarization is chained (Phase 2).
- **`Workspace`** (`core/workspace.py`) — frozen dataclass exposing every workdir path +
  `EventMirror` context manager (tees NDJSON to `events.ndjson`) + PID lock via independent
  `<workdir>/.lock` file (`O_CREAT | O_EXCL` atomic create); stale-PID detection downgrades
  to warning.
- **Doctor checks** (3 new) — all WARN-only so diarize-only users don't regress:
  - `check_whisper_cli()` — verifies `whisper-cli` on PATH and greps required flags
    (`--output-json-full`, `--max-context`, `--vad`, `--split-on-word`, `--logprob-thold`).
  - `check_whisper_model()` — discovers the configured ggml model.
  - `check_vad_model()` — discovers the silero VAD bin.
- **Bundle aux files** — `BUNDLE_AUX_FILES` extends bundle to include the silero VAD bin
  (~885 KB). The whisper.cpp ggml model itself stays OUT (license boundary + 1.5 GB size);
  `voxkit doctor` directs users to `brew install whisper-cpp` and
  `huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin`.
- **Pydantic models** (`io/schema.py` extended): `TranscriptionOutput`, `TranscriptSegment`,
  `Word`, `ChunkStat`, `RemixrTranscript`, `RemixrSegment`, `RemixrWord`. camelCase aliases
  via `populate_by_name=True`. New `TranscriptionOutput.schemaVersion = "1"` (independent
  counter from `DiarizationOutput.schemaVersion`).
- **Internal `Entry` type** (`core/types.py`) — frozen dataclass for whisper.cpp transcription
  rows; bridge between `whisper_exec` / `segmenter` / `hallucination_filter`.
- **Pipeline orchestration** (`core/transcribe_pipeline.py`) — `run_pipeline(req, progress)`
  drives audio prep → chunk plan → per-chunk whisper (resume-aware) → blocklist filter →
  segmenter → ASR merge → write voxkit / raw / SRT / VTT / manifest.
- **Audio extensions** (`core/audio.py`):
  - `plan_chunks(duration, work_dir, *, threshold=900, chunk=600, overlap=5)` — chunk plan
    builder.
  - `normalize_to_wav_16k_mono(input, out_wav)` — ffmpeg normalization.
  - `extract_chunk(master_wav, spec)` — `-ss` before `-i` for accurate input-seek.
- **Dual-mode segmenter** (`core/segmenter.py`) — `detect_mode()` uses leading-space ratio
  ≥0.5 to pick English word mode vs Chinese phrase mode (`--language` overrides). 4-priority
  segment boundary: punctuation-end > 500 ms gap > 5 s duration > 100 chars.
- **ASR merge** (`core/asr_merge.py`) — overlap-dedup + offset.
  `offset_segment(seg, delta)` shifts BOTH `segment.{start,end}` AND `words[].{start,end}` —
  this synchronization is the regression fix for a 6-month bug in Remixr's TS implementation,
  enforced by a hard regression test in `tests/test_asr_merge.py`.
- **Hallucination filter** (`core/hallucination_filter.py` + `data/hallucination_blocklist.yaml`)
  — three rules in order: watermark prefix (NFC-normalized startswith), standalone exact match,
  ghost CJK loop. Drops are logged as NDJSON to `hallucinations.log`.
- **Whisper exec** (`core/whisper_exec.py`) — pure-function `build_argv()` (snapshot-testable),
  `Popen`-based `run_whisper()` that streams stderr and parses `progress\s*=\s*(\d+)%` to emit
  `{event: "progress", stage: "whisper.chunk", chunk, percent}`.
- 200+ new unit + integration tests (`test_transcribe_e2e` gated by `requires_whisper`
  marker for environments without whisper.cpp installed).

### Changed

- **BREAKING — package rename**: `voxsplit` → `voxkit`. All Python imports change.
- **BREAKING — CLI rename**: `voxsplit` command → `voxkit`. The old name is NOT aliased;
  user must update scripts.
- **BREAKING — user data paths**:
  - `~/.local/share/voxsplit/venv` → `~/.local/share/voxkit/venv`
  - `~/.cache/voxsplit/.installed` → `~/.cache/voxkit/.installed`
- **BREAKING — bundle filename**: `voxsplit-models.tar.gz` → `voxkit-models.tar.gz`,
  `voxsplit-models.manifest.json` → `voxkit-models.manifest.json`.
- **BREAKING — bundle GitHub repo**: `3Craft/voxsplit` → `3Craft/voxkit`. The old repo
  retains v0.2.x bundles for archival; v0.3.0+ bundles publish to the new repo only.
- **Worker subprocess module path**: `python -m voxsplit.core.pipeline` →
  `python -m voxkit.core.pipeline`.
- **Worker stdout sentinel**: `__VOXSPLIT_JSON__` → `__VOXKIT_JSON__`.
- **`pyproject [project.scripts]` entry**: `voxsplit = "voxsplit.cli:main"` →
  `voxkit = "voxkit.cli:main"`.
- **`cli.py` `prog=`**: `prog="voxsplit"` → `prog="voxkit"`.

### Stable / non-breaking

- **`DiarizationOutput.schemaVersion` stays `"1"`** — existing JSON consumers do not need
  updates.
- All v0.2.x subcommands preserved with identical CLI contracts (post-rename of the prog name):
  `diarize`, `align`, `doctor`, `setup`, `build-bundle`, `fetch-bundle`.
- Worker venv lazy-install mechanism (`uv venv` + `pyannote.audio>=4.0.4,<5`) unchanged;
  whisper.cpp + ffmpeg + silero VAD model stay OUT of venv (native binaries discovered via
  PATH).
- `commands/align.py` already reads voxkit-style transcript JSON (`segments[].start/end/text`);
  no logic change needed.

### Compatibility & dependencies

- whisper.cpp 1.7+ recommended (verified against 1.8.4); requires flags
  `--output-json-full`, `--max-context`, `--vad`, `--split-on-word`, `--logprob-thold`.
  `voxkit doctor` warn-fails (non-fatal) if any are missing.
- Default whisper model: `large-v3-turbo` (FP16, 1.5 GB). Alternative:
  `large-v3-turbo-q5_0` (547 MB, slightly lower quality). User installs via `brew` or
  `huggingface-cli`.
- pyannote / torch remain in the lazy-install venv (only used by `diarize`).

### Migration notes

For users upgrading from voxsplit 0.2.x:

```bash
# 1. Rename user data (or just re-run `voxkit setup` to recreate fresh).
mv ~/.local/share/voxsplit ~/.local/share/voxkit
mv ~/.cache/voxsplit ~/.cache/voxkit

# 2. Reinstall the new entrypoint (uv tool / pipx / pip — pick one).
uv tool install --force voxkit
# or: pipx install --force voxkit
# or: pip install --upgrade voxkit

# 3. Fetch the v0.3.0+ bundle from the new repo.
voxkit fetch-bundle              # pulls from 3Craft/voxkit, not 3Craft/voxsplit

# 4. (For transcribe) Install whisper.cpp + the ggml model.
brew install whisper-cpp ffmpeg-full
huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin \
  --local-dir ~/.cache/voxkit/models

# 5. Verify.
voxkit doctor                    # 10 checks (7 inherited + 3 new whisper-related)
```

Existing `DiarizationOutput` JSON consumers do not need code changes — `schemaVersion` is
still `"1"`. Scripts referencing the `voxsplit` command must be updated to `voxkit`.

### Known issues

None known at release.

### Verification

- 263+ tests passing (`pytest tests/ -q`).
- End-to-end real-world test: 64-min English podcast → 7 chunks, RTF 0.0476, 909 segments,
  0 hallucinations, full Remixr Zod conformance.
- Concurrency: two `voxkit transcribe` runs against different workdirs do not interfere
  (data orthogonal); same workdir is rejected by `wx` exclusive write on
  `transcript.raw.json`.
- Resume: re-running `voxkit transcribe` on an existing workdir keeps `chunk_NNN.json`
  mtimes unchanged; `--force` updates them all.

## [0.2.x] — voxsplit (archived)

Released as the `voxsplit` package. See the `3Craft/voxsplit` GitHub repo for v0.2.x release
history. voxkit 0.3.0 is a renamed continuation; the functional content of voxsplit 0.2.x
(`diarize`, `align`, `doctor`, `setup`, `build-bundle`, `fetch-bundle` commands) is preserved
verbatim.
