"""Quality metrics — 字幕物理指标 + 风险分布；不依赖 LLM，纯计算。

输入：``<workdir>`` 内已存在的字幕 artifact（任意子集）：

  - ``subtitles.cues.json`` (schemaVersion=2)
  - ``subtitles.proofread.json`` (schemaVersion=1)
  - ``subtitles.<lang>.json`` (schemaVersion=1，可多份)

输出 ``quality.report.json``，包含：

  - 每个可读 artifact 的物理指标（cue 时长 / 字符 / cps / speaker 切换 …）
  - proofread / translation 的风险直方图 + note tag 频次
  - prompt / completion token 合计（来自各 artifact 自带 metrics 字段）

设计原则：

  - **零网络、零 LLM**：所有计算从已落盘的 JSON 推导。
  - **缺失即跳过**：哪个 artifact 不存在就不写对应字段（``exclude_none=True``）。
  - CJK 检测复用 ``voxkit.core.proofread_risk.is_cjk_char``，**禁止重复实现**。
  - CJK 比例 > 50%（含标点）即认定为 CJK 主体；CJK 主体的字符限制为 25、cps 上限 9。
"""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from voxkit.core.proofread_risk import is_cjk_char

__all__ = [
    "SubtitlePhysicalMetrics",
    "ProofreadAggregateMetrics",
    "TranslationAggregateMetrics",
    "QualityReport",
    "compute_physical_metrics",
    "aggregate_proofread",
    "aggregate_translation",
    "build_quality_report",
    "write_quality_report",
]


# ── thresholds（默认值；compute_physical_metrics 可覆盖） ──────────────────

# 字符上限：Latin 行默认 42，CJK 主体减半到 25（中文字幕业界 ~ 13–17，留余量）。
_DEFAULT_CHAR_LIMIT_LATIN = 42
_DEFAULT_CHAR_LIMIT_CJK = 25

# CPS 上限：英文常见 17，CJK 阅读速度慢取 9。
_DEFAULT_CPS_LIMIT_LATIN = 17.0
_DEFAULT_CPS_LIMIT_CJK = 9.0

# 单条 cue 时长边界：< 1s 闪屏；> 7s 滞留。
_FLASH_THRESHOLD_S = 1.0
_LONG_THRESHOLD_S = 7.0

# CJK 主体判定阈值：含 CJK 字符（标点也算）的比例 > 0.5。
_CJK_MAJORITY_RATIO = 0.5

# 风险等级桶；保证 histogram 即使一条 cue 也不会缺 key。
_RISK_BUCKETS = ("low", "medium", "high", "blocking")


# ── pydantic 模型 ──────────────────────────────────────────────────────────


class SubtitlePhysicalMetrics(BaseModel):
    """单一字幕轨的物理指标（cue 时长 / 字符 / cps / speaker 切换）。

    所有 rate 字段都是 0..1 之间的分数（cue 总数为分母）。``cueCount=0`` 时所有
    rate 取 0.0，避免除零。
    """

    model_config = ConfigDict(populate_by_name=True)

    cue_count: int = Field(..., alias="cueCount")
    avg_cue_dur_s: float = Field(..., alias="avgCueDurS")
    p50_cue_dur_s: float = Field(..., alias="p50CueDurS")
    p90_cue_dur_s: float = Field(..., alias="p90CueDurS")
    flash_cue_rate: float = Field(..., alias="flashCueRate")
    long_cue_rate: float = Field(..., alias="longCueRate")
    avg_chars: float = Field(..., alias="avgChars")
    over_char_limit_rate: float = Field(..., alias="overCharLimitRate")
    over_cps_rate: float = Field(..., alias="overCpsRate")
    speaker_switch_cue_rate: float = Field(..., alias="speakerSwitchCueRate")


class ProofreadAggregateMetrics(BaseModel):
    """proofread artifact 的聚合指标：风险分布 + note 频次 + 已有 rate 字段透传。"""

    model_config = ConfigDict(populate_by_name=True)

    cue_count: int = Field(..., alias="cueCount")
    changed_cue_rate: float = Field(..., alias="changedCueRate")
    review_cue_rate: float = Field(..., alias="reviewCueRate")
    risk_histogram: Dict[str, int] = Field(..., alias="riskHistogram")
    note_histogram: Dict[str, int] = Field(..., alias="noteHistogram")
    prompt_tokens_total: int = Field(0, alias="promptTokensTotal")
    completion_tokens_total: int = Field(0, alias="completionTokensTotal")


class TranslationAggregateMetrics(BaseModel):
    """translation artifact 的聚合指标：长度溢出 / glossary miss / 风险分布。"""

    model_config = ConfigDict(populate_by_name=True)

    cue_count: int = Field(..., alias="cueCount")
    over_char_limit_rate: float = Field(..., alias="overCharLimitRate")
    glossary_miss_rate: float = Field(..., alias="glossaryMissRate")
    risk_histogram: Dict[str, int] = Field(..., alias="riskHistogram")
    note_histogram: Dict[str, int] = Field(..., alias="noteHistogram")
    prompt_tokens_total: int = Field(0, alias="promptTokensTotal")
    completion_tokens_total: int = Field(0, alias="completionTokensTotal")


class QualityReport(BaseModel):
    """``quality.report.json`` 顶层 schema。

    每个二级字段可选：仅在对应 artifact 可读时填充；``exclude_none`` 序列化时
    会把缺失字段从 JSON 中剥离，便于消费者用 ``key in report`` 判断。
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    source_id: str = Field(..., alias="sourceId")
    generated_at: str = Field(..., alias="generatedAt")
    inputs: Dict[str, Any]
    cues_metrics: Optional[SubtitlePhysicalMetrics] = Field(None, alias="cuesMetrics")
    proofread_metrics: Optional[ProofreadAggregateMetrics] = Field(
        None, alias="proofreadMetrics"
    )
    proofread_cue_metrics: Optional[SubtitlePhysicalMetrics] = Field(
        None, alias="proofreadCueMetrics"
    )
    translations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# ── 物理指标计算 ───────────────────────────────────────────────────────────


def _is_cjk_majority(texts: Sequence[str]) -> bool:
    """判断一组文本是否 CJK 主体。CJK 字符（含 CJK 标点）占比 > 0.5 即认定。"""
    total = 0
    cjk = 0
    for t in texts:
        for ch in t:
            if ch.isspace():
                continue
            total += 1
            if is_cjk_char(ch):
                cjk += 1
    if total == 0:
        return False
    return (cjk / total) > _CJK_MAJORITY_RATIO


def _percentile(values: List[float], q: float) -> float:
    """简单百分位（无插值）：返回排序后 ceil(q * n) 位置（1-based）的值。

    用 statistics 库的话需要 Python 3.8+ 的 ``quantiles``，但其行为依赖 ``method``
    参数；这里直接用最朴素的 nearest-rank 法，便于单测可复现。
    """
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    idx = int(round(q * (len(s) - 1)))
    return s[idx]


def compute_physical_metrics(
    cues: Sequence[Dict[str, Any]],
    *,
    flash_threshold_s: float = _FLASH_THRESHOLD_S,
    long_threshold_s: float = _LONG_THRESHOLD_S,
) -> SubtitlePhysicalMetrics:
    """计算单一字幕轨的物理指标。

    ``cues`` 每条期望含 ``start`` / ``end`` / ``text``；``speaker`` 可选。CJK 主体
    自动启用 25-char / 9-cps 限值；否则用 42-char / 17-cps。
    """
    n = len(cues)
    if n == 0:
        return SubtitlePhysicalMetrics(
            cueCount=0,
            avgCueDurS=0.0,
            p50CueDurS=0.0,
            p90CueDurS=0.0,
            flashCueRate=0.0,
            longCueRate=0.0,
            avgChars=0.0,
            overCharLimitRate=0.0,
            overCpsRate=0.0,
            speakerSwitchCueRate=0.0,
        )

    texts = [str(c.get("text", "")) for c in cues]
    durations = [max(0.0, float(c["end"]) - float(c["start"])) for c in cues]
    char_counts = [len(t) for t in texts]

    cjk_majority = _is_cjk_majority(texts)
    char_limit = _DEFAULT_CHAR_LIMIT_CJK if cjk_majority else _DEFAULT_CHAR_LIMIT_LATIN
    cps_limit = _DEFAULT_CPS_LIMIT_CJK if cjk_majority else _DEFAULT_CPS_LIMIT_LATIN

    flash = sum(1 for d in durations if d < flash_threshold_s)
    long_ = sum(1 for d in durations if d > long_threshold_s)
    over_char = sum(1 for c in char_counts if c > char_limit)

    # cps：dur=0 时跳过分母（认定无穷大就直接计为 over）。
    over_cps = 0
    for chars, dur in zip(char_counts, durations):
        if dur <= 0:
            if chars > 0:
                over_cps += 1
            continue
        if chars / dur > cps_limit:
            over_cps += 1

    # speaker switch：第 i 条与 i-1 不同。第 0 条不计入分子，但分母仍是 n。
    switch = 0
    prev: Any = None
    for i, c in enumerate(cues):
        spk = c.get("speaker")
        if i > 0 and spk != prev:
            switch += 1
        prev = spk

    return SubtitlePhysicalMetrics(
        cueCount=n,
        avgCueDurS=statistics.fmean(durations),
        p50CueDurS=_percentile(durations, 0.5),
        p90CueDurS=_percentile(durations, 0.9),
        flashCueRate=flash / n,
        longCueRate=long_ / n,
        avgChars=statistics.fmean(char_counts),
        overCharLimitRate=over_char / n,
        overCpsRate=over_cps / n,
        speakerSwitchCueRate=switch / n,
    )


# ── 聚合：proofread / translation ──────────────────────────────────────────


def _empty_risk_histogram() -> Dict[str, int]:
    return {k: 0 for k in _RISK_BUCKETS}


def _accumulate_risk_and_notes(
    cues: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, int], Dict[str, int]]:
    """遍历 cue 列表，累加 ``risk`` 桶与 ``notes`` 出现次数。

    未知 risk 值默认归入 ``low``（最保守），保证 histogram key 集稳定。同一条 cue
    出现重复 note tag 视为多次（这个频次能反映"密集疑点"）。
    """
    risk_hist = _empty_risk_histogram()
    note_hist: Dict[str, int] = {}
    for cue in cues:
        risk = cue.get("risk") or "low"
        if risk not in risk_hist:
            risk = "low"
        risk_hist[risk] += 1
        for note in cue.get("notes") or []:
            note_hist[note] = note_hist.get(note, 0) + 1
    return risk_hist, note_hist


def aggregate_proofread(proofread_doc: Dict[str, Any]) -> ProofreadAggregateMetrics:
    """从 proofread artifact dict 聚合风险 + note 频次。

    复用 artifact 自带的 ``metrics.changedCueRate`` / ``reviewCueRate`` /
    ``promptTokensTotal`` / ``completionTokensTotal``，不重新计算（避免与上游口径
    分叉）。
    """
    cues = proofread_doc.get("cues") or []
    metrics = proofread_doc.get("metrics") or {}
    risk_hist, note_hist = _accumulate_risk_and_notes(cues)
    return ProofreadAggregateMetrics(
        cueCount=int(metrics.get("cueCount", len(cues))),
        changedCueRate=float(metrics.get("changedCueRate", 0.0)),
        reviewCueRate=float(metrics.get("reviewCueRate", 0.0)),
        riskHistogram=risk_hist,
        noteHistogram=note_hist,
        promptTokensTotal=int(metrics.get("promptTokensTotal", 0)),
        completionTokensTotal=int(metrics.get("completionTokensTotal", 0)),
    )


def aggregate_translation(
    translation_doc: Dict[str, Any],
) -> TranslationAggregateMetrics:
    """从 translation artifact dict 聚合长度溢出 / glossary miss / 风险分布。"""
    cues = translation_doc.get("cues") or []
    metrics = translation_doc.get("metrics") or {}
    risk_hist, note_hist = _accumulate_risk_and_notes(cues)
    return TranslationAggregateMetrics(
        cueCount=int(metrics.get("cueCount", len(cues))),
        overCharLimitRate=float(metrics.get("overCharLimitRate", 0.0)),
        glossaryMissRate=float(metrics.get("glossaryMissRate", 0.0)),
        riskHistogram=risk_hist,
        noteHistogram=note_hist,
        promptTokensTotal=int(metrics.get("promptTokensTotal", 0)),
        completionTokensTotal=int(metrics.get("completionTokensTotal", 0)),
    )


# ── 主入口：扫描 workdir + 构建 report ──────────────────────────────────────


# 语言 token：开头字母，余下允许字母数字 / 下划线 / 连字符；不会匹配到 "cues" /
# "proofread" 因为我们显式排除这两个 stem。
_LANG_RE = re.compile(r"^subtitles\.([a-zA-Z][a-zA-Z0-9_-]*)\.json$")
_EXCLUDED_LANG_STEMS = {"cues", "proofread"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _proofread_cues_to_physical_input(
    proofread_doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """把 proofread cue 转成 ``compute_physical_metrics`` 的输入形态。

    用 ``correctedText`` 作为字符基础（这是真实落屏的文本），时间用源 cue 时间。
    """
    out: List[Dict[str, Any]] = []
    for c in proofread_doc.get("cues") or []:
        out.append({
            "start": float(c.get("sourceStart", 0.0)),
            "end": float(c.get("sourceEnd", 0.0)),
            "text": c.get("correctedText", ""),
            "speaker": c.get("speaker"),
        })
    return out


def _translation_cues_to_physical_input(
    translation_doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """translation artifact 已经是扁平 start/end/text/speaker，直接复用。"""
    return list(translation_doc.get("cues") or [])


def build_quality_report(workdir: Path) -> QualityReport:
    """扫描 ``workdir`` 内的字幕 artifact，构建 ``QualityReport``。

    至少需要 cues / proofread / 任一 translation 之一存在；全无则抛
    ``ValueError``（CLI 层捕获并以非零退出码报错）。
    """
    workdir = Path(workdir)
    if not workdir.is_dir():
        raise ValueError(f"workdir not a directory: {workdir}")

    inputs: Dict[str, Any] = {}
    source_id: Optional[str] = None

    cues_path = workdir / "subtitles.cues.json"
    proofread_path = workdir / "subtitles.proofread.json"

    cues_metrics: Optional[SubtitlePhysicalMetrics] = None
    proofread_metrics: Optional[ProofreadAggregateMetrics] = None
    proofread_cue_metrics: Optional[SubtitlePhysicalMetrics] = None
    translations: Dict[str, Dict[str, Any]] = {}

    # ── cues.json ─────────────────────────────────────────────────────────
    if cues_path.is_file():
        doc = _load_json(cues_path)
        inputs["cues"] = cues_path.name
        source_id = source_id or doc.get("sourceId")
        cues_metrics = compute_physical_metrics(doc.get("cues") or [])

    # ── proofread.json ────────────────────────────────────────────────────
    if proofread_path.is_file():
        doc = _load_json(proofread_path)
        inputs["proofread"] = proofread_path.name
        source_id = source_id or doc.get("sourceId")
        proofread_metrics = aggregate_proofread(doc)
        proofread_cue_metrics = compute_physical_metrics(
            _proofread_cues_to_physical_input(doc)
        )

    # ── translations: subtitles.<lang>.json ───────────────────────────────
    translation_inputs: Dict[str, str] = {}
    for p in sorted(workdir.glob("subtitles.*.json")):
        m = _LANG_RE.match(p.name)
        if not m:
            continue
        lang = m.group(1)
        if lang in _EXCLUDED_LANG_STEMS:
            continue
        doc = _load_json(p)
        source_id = source_id or doc.get("sourceId")
        translation_inputs[lang] = p.name
        agg = aggregate_translation(doc)
        phys = compute_physical_metrics(_translation_cues_to_physical_input(doc))
        translations[lang] = {
            "aggregate": agg.model_dump(by_alias=True, exclude_none=True),
            "physical": phys.model_dump(by_alias=True, exclude_none=True),
        }
    if translation_inputs:
        inputs["translations"] = translation_inputs

    if not inputs:
        raise ValueError(
            f"no subtitle artifacts found in {workdir}; "
            f"expected at least one of: subtitles.cues.json, "
            f"subtitles.proofread.json, subtitles.<lang>.json"
        )

    if source_id is None:
        # 理论上不会到这（任一 artifact 都应有 sourceId）；兜底用目录名。
        source_id = workdir.name

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return QualityReport(
        schemaVersion="1",
        sourceId=source_id,
        generatedAt=generated_at,
        inputs=inputs,
        cuesMetrics=cues_metrics,
        proofreadMetrics=proofread_metrics,
        proofreadCueMetrics=proofread_cue_metrics,
        translations=translations,
    )


def write_quality_report(path: Path, report: QualityReport) -> None:
    """原子写 ``quality.report.json``：tmp 同目录 → ``os.replace`` 改名。

    ``model_dump(by_alias=True, exclude_none=True)`` 输出 camelCase 且省略 None
    字段；缩进 2、``ensure_ascii=False`` 保留中文、文件末尾留 newline。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(by_alias=True, exclude_none=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
