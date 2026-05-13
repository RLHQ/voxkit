"""voxkit eval 指标计算：对照人类金标字幕评估 voxkit 输出。

设计要点（详见 `docs/eval-baseline-observations.md`）：
* 指标精简到 3 类：cue 密度比、边界 precision/recall/F1、跨 cue 拉丁词切断；
* `boundary_precision` 与 `boundary_recall` **分开报**——voxkit 现状是高
  precision 低 recall（切的对但漏切），不分开看不到结构性差距；
* 纯计算、零 LLM 网络，stdlib + pydantic 即可。
"""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict

from voxkit.core.constants import CJK_LANGUAGES


# ── 数据加载 ────────────────────────────────────────────────────────────────


_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(path: Path) -> List[Dict[str, Any]]:
    """把 SRT 文件解析成 ``[{start, end, text}, ...]``。

    宽松地容忍 trailing 空行、CRLF、`,` 或 `.` 作为毫秒分隔符。
    """
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    cues: List[Dict[str, Any]] = []
    for block in re.split(r"\n{2,}", text):
        lines = [ln for ln in block.strip().split("\n") if ln.strip() != ""]
        if len(lines) < 3:
            continue
        # lines[0] 是序号；lines[1] 是时间；lines[2:] 是文本
        m = _SRT_TIME_RE.match(lines[1])
        if not m:
            continue
        g = list(map(int, m.groups()))
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        cues.append({"start": start, "end": end, "text": "\n".join(lines[2:])})
    return cues


def load_voxkit_cues(workdir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """优先读 ``subtitles.proofread.json``，缺则回退 ``subtitles.cues.json``。

    返回 ``(cues, source_artifact_name)``，``cues`` 字段统一为 ``start / end /
    text``，方便上层与金标对齐。
    """
    workdir = Path(workdir)
    proof = workdir / "subtitles.proofread.json"
    if proof.is_file():
        doc = json.loads(proof.read_text(encoding="utf-8"))
        cues = [
            {
                "start": float(c.get("sourceStart", 0.0)),
                "end": float(c.get("sourceEnd", 0.0)),
                "text": c.get("correctedText", "") or "",
            }
            for c in (doc.get("cues") or [])
        ]
        return cues, "proofread"

    raw = workdir / "subtitles.cues.json"
    if raw.is_file():
        doc = json.loads(raw.read_text(encoding="utf-8"))
        cues = [
            {
                "start": float(c.get("start", 0.0)),
                "end": float(c.get("end", 0.0)),
                "text": c.get("text", "") or "",
            }
            for c in (doc.get("cues") or [])
        ]
        return cues, "cues"

    raise ValueError(
        f"no voxkit subtitle artifact in {workdir} "
        "(expected subtitles.proofread.json or subtitles.cues.json)"
    )


# ── 指标：纯函数 ────────────────────────────────────────────────────────────


def density_ratio(vk: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> float:
    """``len(vk) / len(gold)``；金标为空时返回 0.0（防御除零）。"""
    if not gold:
        return 0.0
    return len(vk) / len(gold)


def _collect_bounds(cues: List[Dict[str, Any]]) -> List[float]:
    """收集 cue 的所有时间边界（start ∪ end），去重后排序。"""
    bs = set()
    for c in cues:
        bs.add(round(float(c["start"]), 3))
        bs.add(round(float(c["end"]), 3))
    return sorted(bs)


def _near(b: float, sorted_arr: List[float], tol: float) -> bool:
    """二分查找：``sorted_arr`` 里是否存在与 ``b`` 距离 ≤ ``tol`` 的元素。"""
    if not sorted_arr:
        return False
    i = bisect.bisect_left(sorted_arr, b)
    for j in (i - 1, i):
        if 0 <= j < len(sorted_arr) and abs(sorted_arr[j] - b) <= tol:
            return True
    return False


def boundary_metrics(
    vk: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    tol_s: float = 0.3,
) -> Dict[str, Any]:
    """切分边界 precision / recall / F1。

    * precision = voxkit 边界命中金标 / voxkit 边界总数（切的对不对）
    * recall    = 金标边界被 voxkit 覆盖 / 金标边界总数（漏不漏切）
    """
    vk_b = _collect_bounds(vk)
    gold_b = _collect_bounds(gold)

    if not vk_b or not gold_b:
        return {
            "tolerance_s": tol_s,
            "vk_bounds_count": len(vk_b),
            "gold_bounds_count": len(gold_b),
            "vk_hits": 0,
            "gold_covered": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    vk_hits = sum(1 for b in vk_b if _near(b, gold_b, tol_s))
    gold_covered = sum(1 for b in gold_b if _near(b, vk_b, tol_s))

    precision = vk_hits / len(vk_b)
    recall = gold_covered / len(gold_b)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tolerance_s": tol_s,
        "vk_bounds_count": len(vk_b),
        "gold_bounds_count": len(gold_b),
        "vk_hits": vk_hits,
        "gold_covered": gold_covered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def avg_drift(
    vk: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
) -> Dict[str, float]:
    """平均字符数 / 平均时长漂移：voxkit - gold。

    voxkit 比金标粗时漂移为正——直接读出 readability 风险。
    """
    def _avg_chars(cues: List[Dict[str, Any]]) -> float:
        return sum(len(c.get("text", "")) for c in cues) / len(cues) if cues else 0.0

    def _avg_dur(cues: List[Dict[str, Any]]) -> float:
        return (
            sum(float(c["end"]) - float(c["start"]) for c in cues) / len(cues)
            if cues
            else 0.0
        )

    vk_c, gold_c = _avg_chars(vk), _avg_chars(gold)
    vk_d, gold_d = _avg_dur(vk), _avg_dur(gold)
    return {
        "vk_avg_chars": vk_c,
        "gold_avg_chars": gold_c,
        "chars_drift": vk_c - gold_c,
        "vk_avg_dur_s": vk_d,
        "gold_avg_dur_s": gold_d,
        "dur_drift_s": vk_d - gold_d,
    }


# 启发式：检测 reseg 把拉丁词切到两个 cue 的情况。
#
# 命中条件：
#   * 当前 cue 末尾（可后跟标点空白）是 1–3 个孤立拉丁字母，且前面有空白；
#   * 下一 cue 开头是 1+ 个**小写**拉丁字母（新词通常大写，故小写多为词尾）。
#
# 这套启发式是为 **CJK 语种**设计的——中文上下文里出现孤立拉丁字母大概率
# 是被切断的词；但英文场景下「cue 末尾介词 of/to/in + 下条小写起头」是
# 正常 ASR 切分，会被严重误报。因此 ``language`` 非 CJK 时直接返回 0。
_END_LATIN_SCRAP = re.compile(r"\s[A-Za-z]{1,3}[\s，,。.\?\!？！]*$")
_START_LOWER_LATIN = re.compile(r"^[a-z]+")


def broken_latin_words(
    cues: List[Dict[str, Any]],
    language: str | None = None,
) -> int:
    """统计相邻 cue 跨边界切断拉丁词的对数（启发式，详见正则上方注释）。

    ``language`` 非 CJK（且非 ``None``）时跳过检测——避免在英文等场景把
    正常句末介词误判为切断。``None`` 时保留 CJK 行为以兼容旧调用。
    """
    if language is not None and language.lower() not in CJK_LANGUAGES:
        return 0
    n = 0
    for i in range(len(cues) - 1):
        t = (cues[i].get("text") or "").rstrip()
        nxt = (cues[i + 1].get("text") or "").lstrip()
        if _END_LATIN_SCRAP.search(t) and _START_LOWER_LATIN.match(nxt):
            n += 1
    return n


# ── 顶层报告 ────────────────────────────────────────────────────────────────


class EvalReport(BaseModel):
    """``voxkit eval`` 顶层产物 schema。形态与 ``QualityReport`` 风格对齐。"""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: int = 1
    workdir: str
    reference: str
    language: str
    sourceArtifact: str  # "proofread" | "cues"
    tolerance_s: float
    alignment: Dict[str, Any]
    metrics: Dict[str, Any]


def build_eval_report(
    workdir: Path,
    reference: Path,
    language: str,
    tolerance_s: float = 0.3,
) -> EvalReport:
    """串联：加载产物 → 算指标 → 装 ``EvalReport``。"""
    workdir = Path(workdir)
    reference = Path(reference)
    vk_cues, source_artifact = load_voxkit_cues(workdir)
    gold_cues = parse_srt(reference)

    drift = avg_drift(vk_cues, gold_cues)
    bm = boundary_metrics(vk_cues, gold_cues, tol_s=tolerance_s)
    broken = broken_latin_words(vk_cues, language=language)
    dr = density_ratio(vk_cues, gold_cues)

    return EvalReport(
        workdir=str(workdir),
        reference=str(reference),
        language=language,
        sourceArtifact=source_artifact,
        tolerance_s=tolerance_s,
        alignment={
            "vk_cues": len(vk_cues),
            "gold_cues": len(gold_cues),
            "density_ratio": dr,
            "vk_avg_chars": drift["vk_avg_chars"],
            "gold_avg_chars": drift["gold_avg_chars"],
            "vk_avg_dur_s": drift["vk_avg_dur_s"],
            "gold_avg_dur_s": drift["gold_avg_dur_s"],
        },
        metrics={
            "boundary_precision": bm["precision"],
            "boundary_recall": bm["recall"],
            "boundary_f1": bm["f1"],
            "boundary_vk_bounds": bm["vk_bounds_count"],
            "boundary_gold_bounds": bm["gold_bounds_count"],
            "boundary_vk_hits": bm["vk_hits"],
            "boundary_gold_covered": bm["gold_covered"],
            "chars_drift": drift["chars_drift"],
            "dur_drift_s": drift["dur_drift_s"],
            "broken_latin_words": broken,
        },
    )


def write_eval_report(path: Path, report: EvalReport) -> None:
    """落到 ``path``，UTF-8，pretty-printed。"""
    Path(path).write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
