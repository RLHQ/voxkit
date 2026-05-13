"""voxkit.core.llm_eval 单测。

策略：FakeLLMClient 返回预制 JSON 响应，断言对齐分组 / 批处理 / 聚合 /
高风险提取 / 落盘形态。零网络。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.llm_eval import (
    LlmCueScores,
    LlmEvalReport,
    _aggregate_scores,
    _build_group_results,
    _fallback_score,
    _parse_llm_response,
    align_cue_groups,
    build_llm_eval_report,
    score_groups_with_llm,
    write_llm_eval_report,
)
from voxkit.llm.client import ChatResult


# ── FakeLLMClient（复用 test_translate_pipeline.py 同款形态）────────────


class FakeLLMClient:
    def __init__(self, responses: List[str], *, model: str = "deepseek-v4-flash") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        if not self._responses:
            raise AssertionError("FakeLLMClient 响应不够")
        resp = self._responses.pop(0)
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        return ChatResult(
            text=resp,
            prompt_tokens=100,
            completion_tokens=50,
            model=self._model,
            raw={},
        )

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _score_resp(group_id: int, **scores: int) -> Dict[str, Any]:
    """构造一个 LLM 返回的单组评分 dict。"""
    base = {
        "semantic": 8,
        "terminology": 9,
        "segmentation": 7,
        "punctuation": 9,
        "readability": 8,
    }
    base.update(scores)
    overall = round(sum(base.values()) / 5, 1)
    return {
        "group_id": group_id,
        "scores": base,
        "overall": overall,
        "issues": [],
        "explanation": "ok",
    }


# ── align_cue_groups ────────────────────────────────────────────────────────


def test_align_cue_groups_pairs_overlapping_cues() -> None:
    """voxkit 1 条 cue 覆盖金标 2 条时间相交 cue → 1 个组 (1+2)。"""
    vk = [{"start": 0.0, "end": 4.0, "text": "你好世界"}]
    gold = [
        {"start": 0.0, "end": 2.0, "text": "你好"},
        {"start": 2.0, "end": 4.0, "text": "世界"},
    ]
    groups = align_cue_groups(vk, gold)
    assert len(groups) == 1
    assert len(groups[0].voxkit) == 1
    assert len(groups[0].gold) == 2
    assert groups[0].time_start == 0.0
    assert groups[0].time_end == 4.0


def test_align_cue_groups_separates_non_overlapping() -> None:
    """两段时间不相交的 cue → 2 个独立组。"""
    vk = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 5.0, "end": 6.0, "text": "b"},
    ]
    gold = [
        {"start": 0.0, "end": 1.0, "text": "x"},
        {"start": 5.0, "end": 6.0, "text": "y"},
    ]
    groups = align_cue_groups(vk, gold)
    assert len(groups) == 2


def test_align_cue_groups_vk_only_or_gold_only() -> None:
    """某侧空白时间段 → 独立组 (vk_only / gold_only)。"""
    vk = [{"start": 0.0, "end": 1.0, "text": "voxkit-only"}]
    gold = [{"start": 5.0, "end": 6.0, "text": "gold-only"}]
    groups = align_cue_groups(vk, gold)
    assert len(groups) == 2
    g_vk = next(g for g in groups if g.voxkit and not g.gold)
    g_gold = next(g for g in groups if g.gold and not g.voxkit)
    assert g_vk.voxkit[0]["text"] == "voxkit-only"
    assert g_gold.gold[0]["text"] == "gold-only"


def test_align_cue_groups_assigns_sequential_ids() -> None:
    vk = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 5.0, "end": 6.0, "text": "b"}]
    gold = []
    groups = align_cue_groups(vk, gold)
    assert [g.group_id for g in groups] == [0, 1]


# ── _parse_llm_response ─────────────────────────────────────────────────────


def test_parse_llm_response_valid_array() -> None:
    text = json.dumps([_score_resp(0), _score_resp(1)])
    parsed = _parse_llm_response(text, [0, 1])
    assert len(parsed) == 2
    assert parsed[0]["scores"]["semantic"] == 8


def test_parse_llm_response_object_wrapper() -> None:
    """LLM 可能返回 {"results": [...]} 形式。"""
    text = json.dumps({"results": [_score_resp(0)]})
    parsed = _parse_llm_response(text, [0])
    assert len(parsed) == 1


def test_parse_llm_response_fills_missing_groups() -> None:
    """LLM 只返回部分 group → 缺失的填 fallback 中性分。"""
    text = json.dumps([_score_resp(0)])  # 只有 0，缺 1, 2
    parsed = _parse_llm_response(text, [0, 1, 2])
    assert len(parsed) == 3
    assert parsed[0]["scores"]["semantic"] == 8
    assert parsed[1]["scores"]["semantic"] == 5  # fallback
    assert parsed[2]["scores"]["semantic"] == 5


def test_parse_llm_response_invalid_json() -> None:
    """JSON 解析失败 → 全 fallback。"""
    parsed = _parse_llm_response("not a json", [0, 1])
    assert all(r["scores"]["semantic"] == 5 for r in parsed)


# ── score_groups_with_llm 批处理 ───────────────────────────────────────────


def test_score_groups_batches_requests() -> None:
    """7 组 + batch=3 → 应触发 3 次 LLM 调用。"""
    groups = align_cue_groups(
        [{"start": i, "end": i + 0.5, "text": f"v{i}"} for i in range(7)],
        [{"start": i, "end": i + 0.5, "text": f"g{i}"} for i in range(7)],
    )
    assert len(groups) == 7

    responses = [
        json.dumps([_score_resp(0), _score_resp(1), _score_resp(2)]),
        json.dumps([_score_resp(3), _score_resp(4), _score_resp(5)]),
        json.dumps([_score_resp(6)]),
    ]
    client = FakeLLMClient(responses)

    raw, pt, ct = score_groups_with_llm(
        groups,
        client=client,
        language="zh",
        prompt_template="lang={language}\nbatch={batch}",
        max_groups_per_batch=3,
    )
    assert len(client.calls) == 3
    assert len(raw) == 7
    assert pt == 300 and ct == 150  # 3 × (100 + 50)


def test_score_groups_passes_language_and_batch_to_prompt() -> None:
    """prompt 模板的 {language} 和 {batch} placeholder 必须被替换。"""
    groups = align_cue_groups(
        [{"start": 0.0, "end": 1.0, "text": "你好"}],
        [{"start": 0.0, "end": 1.0, "text": "你好"}],
    )
    client = FakeLLMClient([json.dumps([_score_resp(0)])])
    score_groups_with_llm(
        groups,
        client=client,
        language="zh",
        prompt_template="lang={language}\nbatch={batch}",
        max_groups_per_batch=10,
    )
    msg = client.calls[0]["messages"][0]["content"]
    assert "lang=zh" in msg
    assert "你好" in msg  # batch 里 cue 文本透传


# ── _aggregate_scores ──────────────────────────────────────────────────────


def test_aggregate_scores_means_p50_p10() -> None:
    """5 组评分 → 各维度 mean / p50 / p10 计算正确。"""
    groups = align_cue_groups(
        [{"start": i, "end": i + 0.5, "text": f"v"} for i in range(5)],
        [{"start": i, "end": i + 0.5, "text": f"g"} for i in range(5)],
    )
    # 5 组 semantic 分数 [10, 8, 6, 4, 2]，mean=6.0, p50=6, p10=2
    raw = [
        _score_resp(0, semantic=10),
        _score_resp(1, semantic=8),
        _score_resp(2, semantic=6),
        _score_resp(3, semantic=4),
        _score_resp(4, semantic=2),
    ]
    results = _build_group_results(groups, raw)
    agg = _aggregate_scores(results)
    assert agg["semantic"]["mean"] == pytest.approx(6.0)
    assert agg["semantic"]["p50"] == pytest.approx(6.0)
    assert agg["semantic"]["p10"] == pytest.approx(2.0)


# ── high_risk 提取 + e2e ───────────────────────────────────────────────────


def _write_srt(path: Path, cues: List[Dict[str, Any]]) -> None:
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{fmt(c['start'])} --> {fmt(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_llm_eval_report_e2e_extracts_high_risk(tmp_path: Path) -> None:
    """端到端：cues.json + gold SRT + Fake LLM → 报告含高风险组。"""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps({"cues": [
            {"start": 0.0, "end": 1.0, "text": "good"},
            {"start": 1.0, "end": 2.0, "text": "bad terminology cue"},
        ]}),
        encoding="utf-8",
    )
    ref = tmp_path / "gold.srt"
    _write_srt(ref, [
        {"start": 0.0, "end": 1.0, "text": "good"},
        {"start": 1.0, "end": 2.0, "text": "correct"},
    ])

    # 一组高分 + 一组高风险（terminology=2 触发 high_risk）
    responses = [json.dumps([
        _score_resp(0),
        _score_resp(1, terminology=2),
    ])]
    client = FakeLLMClient(responses)

    report = build_llm_eval_report(
        workdir=workdir,
        reference=ref,
        language="en",
        client=client,
        provider="deepseek",
        model="fake-model",
    )
    assert isinstance(report, LlmEvalReport)
    assert report.schemaVersion == 1
    assert report.provider == "deepseek"
    assert len(report.groups) == 2
    # high_risk: terminology=2 < 6 触发
    assert len(report.high_risk_groups) == 1
    assert report.high_risk_groups[0].group_id == 1
    assert report.high_risk_groups[0].scores.terminology == 2
    # alignment 摘要
    assert report.alignment["groups"] == 2
    assert report.alignment["groups_with_both"] == 2


def test_write_llm_eval_report_roundtrip(tmp_path: Path) -> None:
    """写盘后 JSON 可解析 + schemaVersion 保留。"""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps({"cues": [{"start": 0.0, "end": 1.0, "text": "x"}]}),
        encoding="utf-8",
    )
    ref = tmp_path / "g.srt"
    _write_srt(ref, [{"start": 0.0, "end": 1.0, "text": "x"}])

    client = FakeLLMClient([json.dumps([_score_resp(0)])])
    report = build_llm_eval_report(
        workdir=workdir,
        reference=ref,
        language="en",
        client=client,
        provider="deepseek",
        model="fake",
    )
    out = tmp_path / "eval-llm.json"
    write_llm_eval_report(out, report)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 1
    assert payload["language"] == "en"
    assert "promptHash" in payload and len(payload["promptHash"]) == 64
