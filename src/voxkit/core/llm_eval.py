"""LLM 多维评分（L3 评估层）：补 boundary metrics 看不到的语义 / 术语 / 标点 / 节奏。

设计动机（详见 docs/eval-methodology.md）：

- L1 ``boundary_metrics`` 只测时间边界对齐，盲点 4 类（语义、内容质量、同义切法、阅读体验）
- L2 ``quality_metrics`` 只看物理可读性（CPS / 闪屏 / 滞留），不对照金标
- **L3 ``llm_eval`` 通过 LLM 跨这两层**：金标作参考但允许 voxkit 不同切法、5 维独立评分

零依赖新增（复用 voxkit.llm.client）。LLM 调用慢且有成本，**不进 CI**——
release / PR review 时跑，输出 ``eval-llm.report.json``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from voxkit.core.eval_metrics import load_voxkit_cues, parse_srt
from voxkit.llm.client import LLMClient
from voxkit.llm.prompts import load_prompt


# ── Pydantic 数据模型 ───────────────────────────────────────────────────────


class LlmCueScores(BaseModel):
    """单组评分（5 维）。"""

    semantic: int = Field(ge=0, le=10)
    terminology: int = Field(ge=0, le=10)
    segmentation: int = Field(ge=0, le=10)
    punctuation: int = Field(ge=0, le=10)
    readability: int = Field(ge=0, le=10)


class LlmGroupResult(BaseModel):
    """LLM 对单个对齐组的评分结果 + 输入快照。"""

    model_config = ConfigDict(extra="forbid")

    group_id: int
    time_start: float
    time_end: float
    voxkit_texts: List[str]
    gold_texts: List[str]
    scores: LlmCueScores
    overall: float = Field(ge=0, le=10)
    issues: List[str] = Field(default_factory=list)
    explanation: str = ""


class LlmEvalReport(BaseModel):
    """``voxkit eval --llm`` 的顶层产物。"""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: int = 1
    workdir: str
    reference: str
    language: str
    sourceArtifact: str  # "reseg2" | "proofread" | "cues"
    provider: str
    model: str
    promptVersion: str = "v1"
    promptHash: str
    alignment: Dict[str, Any]
    scores_aggregate: Dict[str, Dict[str, float]]
    high_risk_groups: List[LlmGroupResult]
    groups: List[LlmGroupResult]
    tokens: Dict[str, int]


# ── Cue 级对齐：把时间重叠的 vk + gold cue 收集成 group ─────────────────────


@dataclass
class _AlignedGroup:
    group_id: int
    time_start: float
    time_end: float
    voxkit: List[Dict[str, Any]]
    gold: List[Dict[str, Any]]


def align_cue_groups(
    vk: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
) -> List[_AlignedGroup]:
    """每个 voxkit cue 作 anchor，关联所有时间相交的 gold cue。

    策略原因：早期版本用「时间相交贪婪 cluster」会在 cue 几乎连续的真实字幕
    （vlog/podcast 中 cue 间 gap 接近 0）上链式膨胀，把整个视频合成 ~10 个
    超大 group——LLM 无法精确指认问题。
    新策略：voxkit cue 是评估 anchor，每个 cue 单独评分，附带跟它时间相交的
    所有 gold cue 作参考。未被任何 voxkit cue 覆盖的 gold cue 作独立
    gold_only group，让 LLM 评估「voxkit 是否漏切了关键内容」。

    返回按 ``time_start`` 排序的 ``_AlignedGroup`` 列表，``group_id`` 顺序赋值。
    """
    raw_groups: List[_AlignedGroup] = []
    used_gold_idx: set[int] = set()
    gn = len(gold)

    # 1. 每个 voxkit cue 一个 group + 时间相交 gold
    gi_start = 0
    for v in vk:
        # 推进 gi_start 跳过完全在 v.start 之前的 gold
        while gi_start < gn and gold[gi_start]["end"] <= v["start"]:
            gi_start += 1
        related: List[Dict[str, Any]] = []
        gj = gi_start
        while gj < gn and gold[gj]["start"] < v["end"]:
            related.append(gold[gj])
            used_gold_idx.add(gj)
            gj += 1
        raw_groups.append(
            _AlignedGroup(
                group_id=-1,  # 占位，最后排序后赋值
                time_start=float(v["start"]),
                time_end=float(v["end"]),
                voxkit=[v],
                gold=related,
            )
        )

    # 2. 未被覆盖的 gold cue 作独立 group（vk 漏切的关键内容）
    for j, g in enumerate(gold):
        if j in used_gold_idx:
            continue
        raw_groups.append(
            _AlignedGroup(
                group_id=-1,
                time_start=float(g["start"]),
                time_end=float(g["end"]),
                voxkit=[],
                gold=[g],
            )
        )

    # 3. 按时间排序 + 重新赋 group_id
    raw_groups.sort(key=lambda gp: gp.time_start)
    return [
        _AlignedGroup(
            group_id=i,
            time_start=g.time_start,
            time_end=g.time_end,
            voxkit=g.voxkit,
            gold=g.gold,
        )
        for i, g in enumerate(raw_groups)
    ]


# ── LLM 调用（批处理）─────────────────────────────────────────────────────


def _render_prompt(template: str, *, language: str, batch_json: str) -> str:
    """把 prompt 模板的 ``{language}`` / ``{batch}`` placeholder 填上。"""
    return template.replace("{language}", language).replace("{batch}", batch_json)


def _group_to_llm_payload(g: _AlignedGroup) -> Dict[str, Any]:
    """精简 group 给 LLM，省 token。"""
    return {
        "group_id": g.group_id,
        "time_start": round(g.time_start, 2),
        "time_end": round(g.time_end, 2),
        "voxkit": [
            {
                "start": round(c["start"], 2),
                "end": round(c["end"], 2),
                "text": c.get("text", ""),
            }
            for c in g.voxkit
        ],
        "gold": [
            {
                "start": round(c["start"], 2),
                "end": round(c["end"], 2),
                "text": c.get("text", ""),
            }
            for c in g.gold
        ],
    }


def _parse_llm_response(text: str, expected_ids: List[int]) -> List[Dict[str, Any]]:
    """解析 LLM 返回的 JSON 数组；缺失 group_id 用 fallback 分数填。

    LLM 可能少返回某些组（被截断 / 拒答），这里做防御性补齐。
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "results" in parsed:
            parsed = parsed["results"]
        if not isinstance(parsed, list):
            raise ValueError("expected JSON array")
    except (json.JSONDecodeError, ValueError):
        # 全 fallback
        return [_fallback_score(gid) for gid in expected_ids]

    by_id = {int(r.get("group_id", -1)): r for r in parsed if isinstance(r, dict)}
    out = []
    for gid in expected_ids:
        if gid in by_id:
            out.append(by_id[gid])
        else:
            out.append(_fallback_score(gid))
    return out


def _fallback_score(group_id: int) -> Dict[str, Any]:
    """LLM 无返回时给中性默认值（5/10），不让单组失败拖垮整份报告。"""
    return {
        "group_id": group_id,
        "scores": {
            "semantic": 5,
            "terminology": 5,
            "segmentation": 5,
            "punctuation": 5,
            "readability": 5,
        },
        "overall": 5.0,
        "issues": ["LLM 未返回该组评分（已用中性默认值）"],
        "explanation": "fallback default (no LLM response)",
    }


def score_groups_with_llm(
    groups: List[_AlignedGroup],
    *,
    client: LLMClient,
    language: str,
    prompt_template: str,
    max_groups_per_batch: int = 10,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """批量调用 LLM 给所有 groups 打分。

    返回 ``(raw_results, prompt_tokens_total, completion_tokens_total)``。
    """
    raw_results: List[Dict[str, Any]] = []
    pt_total = ct_total = 0

    for i in range(0, len(groups), max_groups_per_batch):
        batch = groups[i : i + max_groups_per_batch]
        batch_payload = [_group_to_llm_payload(g) for g in batch]
        batch_ids = [g.group_id for g in batch]
        prompt = _render_prompt(
            prompt_template,
            language=language,
            batch_json=json.dumps(batch_payload, ensure_ascii=False),
        )

        result = client.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format="json_object",
            temperature=0.0,
        )
        pt_total += result.prompt_tokens
        ct_total += result.completion_tokens

        parsed = _parse_llm_response(result.text, batch_ids)
        raw_results.extend(parsed)

    return raw_results, pt_total, ct_total


# ── 聚合 + 顶层 build ───────────────────────────────────────────────────────


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[idx])


def _aggregate_scores(
    results: List[LlmGroupResult],
) -> Dict[str, Dict[str, float]]:
    """每个维度（+overall）的 mean / p50 / p10。"""
    if not results:
        return {}

    dims = ["semantic", "terminology", "segmentation", "punctuation", "readability"]
    agg: Dict[str, Dict[str, float]] = {}
    for d in dims:
        vals = [float(getattr(r.scores, d)) for r in results]
        agg[d] = {
            "mean": sum(vals) / len(vals),
            "p50": _percentile(vals, 0.5),
            "p10": _percentile(vals, 0.1),
        }
    overall_vals = [r.overall for r in results]
    agg["overall"] = {
        "mean": sum(overall_vals) / len(overall_vals),
        "p50": _percentile(overall_vals, 0.5),
        "p10": _percentile(overall_vals, 0.1),
    }
    return agg


def _build_group_results(
    groups: List[_AlignedGroup],
    raw: List[Dict[str, Any]],
) -> List[LlmGroupResult]:
    """把对齐 group + LLM raw scores 合成 ``LlmGroupResult``。"""
    by_id = {int(r["group_id"]): r for r in raw}
    out = []
    for g in groups:
        r = by_id.get(g.group_id) or _fallback_score(g.group_id)
        scores = LlmCueScores(**r["scores"])
        out.append(
            LlmGroupResult(
                group_id=g.group_id,
                time_start=g.time_start,
                time_end=g.time_end,
                voxkit_texts=[c.get("text", "") for c in g.voxkit],
                gold_texts=[c.get("text", "") for c in g.gold],
                scores=scores,
                overall=float(r.get("overall", 5.0)),
                issues=list(r.get("issues") or []),
                explanation=str(r.get("explanation", "")),
            )
        )
    return out


def build_llm_eval_report(
    workdir: Path,
    reference: Path,
    language: str,
    *,
    client: LLMClient,
    provider: str,
    model: str,
    max_groups_per_batch: int = 10,
    high_risk_threshold: float = 6.0,
) -> LlmEvalReport:
    """串联：load cues → align → LLM 批评分 → 聚合 → 顶层 ``LlmEvalReport``。"""
    workdir = Path(workdir)
    reference = Path(reference)

    vk_cues, source_artifact = load_voxkit_cues(workdir)
    gold_cues = parse_srt(reference)

    groups = align_cue_groups(vk_cues, gold_cues)
    template, prompt_hash = load_prompt("eval", "v1")

    raw, pt, ct = score_groups_with_llm(
        groups,
        client=client,
        language=language,
        prompt_template=template,
        max_groups_per_batch=max_groups_per_batch,
    )

    group_results = _build_group_results(groups, raw)
    aggregate = _aggregate_scores(group_results)

    high_risk = [
        r for r in group_results
        if r.overall < high_risk_threshold
        or any(getattr(r.scores, d) < high_risk_threshold for d in
               ("semantic", "terminology", "segmentation", "punctuation", "readability"))
    ]

    # 对齐摘要
    vk_only = sum(1 for g in groups if g.voxkit and not g.gold)
    gold_only = sum(1 for g in groups if g.gold and not g.voxkit)
    both = sum(1 for g in groups if g.voxkit and g.gold)

    return LlmEvalReport(
        workdir=str(workdir),
        reference=str(reference),
        language=language,
        sourceArtifact=source_artifact,
        provider=provider,
        model=model,
        promptVersion="v1",
        promptHash=prompt_hash,
        alignment={
            "vk_cues": len(vk_cues),
            "gold_cues": len(gold_cues),
            "groups": len(groups),
            "groups_with_both": both,
            "groups_vk_only": vk_only,
            "groups_gold_only": gold_only,
        },
        scores_aggregate=aggregate,
        high_risk_groups=high_risk,
        groups=group_results,
        tokens={
            "prompt": pt,
            "completion": ct,
            "total": pt + ct,
        },
    )


def write_llm_eval_report(path: Path, report: LlmEvalReport) -> None:
    """落到 ``path``，UTF-8，pretty-printed JSON。"""
    Path(path).write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
