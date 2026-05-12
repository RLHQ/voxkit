"""``voxkit.core.quality_metrics`` 单测。

策略：tmp_path 写最小 JSON fixture，断言每个指标 / 风险桶 / 输出形态。固定不引
LLM mock（这个模块本来就纯计算，零网络）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.quality_metrics import (
    QualityReport,
    aggregate_proofread,
    aggregate_translation,
    build_quality_report,
    compute_physical_metrics,
    write_quality_report,
)


# ── 物理指标 ────────────────────────────────────────────────────────────────


def test_compute_physical_metrics_basic() -> None:
    cues = [
        {"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "A"},
        {"start": 2.0, "end": 4.0, "text": "foo bar", "speaker": "A"},
        {"start": 4.0, "end": 5.0, "text": "x", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    assert m.cue_count == 3
    # 平均时长 = (2+2+1)/3 ≈ 1.6667
    assert abs(m.avg_cue_dur_s - (5.0 / 3)) < 1e-6
    assert m.p50_cue_dur_s == 2.0
    assert m.p90_cue_dur_s == 2.0
    # 一条 1s 不算 flash（严格 <），其余也 >= 1s
    assert m.flash_cue_rate == 0.0
    assert m.long_cue_rate == 0.0
    # 平均字符 = (11 + 7 + 1) / 3
    assert abs(m.avg_chars - 19 / 3) < 1e-6
    # Latin 字符限制 42 / cps 17，都不超
    assert m.over_char_limit_rate == 0.0
    assert m.over_cps_rate == 0.0
    # 全 A：无 speaker 切换
    assert m.speaker_switch_cue_rate == 0.0


def test_physical_metrics_trailing_bad_word_rate() -> None:
    """末尾停在介词/连词的 cue 比例（仅 Latin 主体启用）。"""
    cues = [
        {"start": 0.0, "end": 2.0, "text": "we built some", "speaker": "A"},
        {"start": 2.0, "end": 4.0, "text": "great products and", "speaker": "A"},
        {"start": 4.0, "end": 6.0, "text": "shipped them globally", "speaker": "A"},
        {"start": 6.0, "end": 8.0, "text": "for our customers and partners.", "speaker": "A"},
        {"start": 8.0, "end": 10.0, "text": "Then it broke.", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    # cue 1 ("some")、cue 2 ("and") 命中 → 2/5
    assert m.trailing_bad_word_rate == 0.4


def test_physical_metrics_single_word_cue_rate() -> None:
    """单 token cue 的比例（典型闪屏症状）。"""
    cues = [
        {"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "A"},
        {"start": 2.0, "end": 2.2, "text": "I'll", "speaker": "A"},  # 单词 + 闪屏
        {"start": 2.2, "end": 4.0, "text": "be back soon", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    assert m.single_word_cue_rate == pytest.approx(1 / 3)


def test_physical_metrics_cross_cue_repeat_rate() -> None:
    """相邻 cue 末尾词与下一 cue 开头词重复（proofread 错误闭合切坏边界）。"""
    cues = [
        # cue 1 末尾 "is it" + cue 2 开头 "Is it" → 重复（大小写不敏感）
        {"start": 0.0, "end": 2.0, "text": "Cash App versus Square. Is it", "speaker": "A"},
        {"start": 2.0, "end": 5.0, "text": "Is it a good idea to be CEO?", "speaker": "A"},
        # cue 3 末尾 "and" + cue 4 开头 "we" → 不重复
        {"start": 5.0, "end": 7.0, "text": "Then we shipped it and", "speaker": "A"},
        {"start": 7.0, "end": 9.0, "text": "we celebrated.", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    # 4 cue → 3 个相邻对；1 对重复 → 1/4
    assert m.cross_cue_repeat_rate == 0.25


def test_physical_metrics_cjk_skips_trailing_bad() -> None:
    """CJK 主体路径下 trailing/single-word 应永远为 0（中文无词性概念）。"""
    cues = [
        {"start": 0.0, "end": 2.0, "text": "我们做了一些", "speaker": "A"},
        {"start": 2.0, "end": 4.0, "text": "产品", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    assert m.trailing_bad_word_rate == 0.0
    assert m.single_word_cue_rate == 0.0


def test_compute_physical_metrics_cjk_threshold() -> None:
    """CJK 主体启用 25-char 限制；30 字必触发 overCharLimit。"""
    long_zh = "中" * 30
    cues = [
        {"start": 0.0, "end": 5.0, "text": long_zh, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "text": "短", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    # 30 > 25 → 第一条超限；第二条不超
    assert m.over_char_limit_rate == 0.5
    # 30 字符 / 5 秒 = 6 cps < 9，不算超速
    assert m.over_cps_rate == 0.0


def test_compute_physical_metrics_speaker_switch_rate() -> None:
    cues = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "A"},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "B"},
        {"start": 2.0, "end": 3.0, "text": "c", "speaker": "B"},
        {"start": 3.0, "end": 4.0, "text": "d", "speaker": "A"},
    ]
    m = compute_physical_metrics(cues)
    # 切换：A→B、B→A 两次；分母 4 → 0.5
    assert m.speaker_switch_cue_rate == 0.5


def test_compute_physical_metrics_flash_and_long_buckets() -> None:
    cues = [
        {"start": 0.0, "end": 0.5, "text": "a", "speaker": "A"},  # flash
        {"start": 1.0, "end": 9.0, "text": "b" * 5, "speaker": "A"},  # long (8s)
        {"start": 10.0, "end": 12.0, "text": "c", "speaker": "A"},  # normal
    ]
    m = compute_physical_metrics(cues)
    assert m.flash_cue_rate == pytest.approx(1 / 3)
    assert m.long_cue_rate == pytest.approx(1 / 3)


# ── proofread 聚合 ─────────────────────────────────────────────────────────


def _proofread_doc(cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "src1",
        "inputArtifact": "subtitles.cues.json",
        "inputHash": "sha256:abc",
        "language": "zh",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "promptVersion": "proofread.v1",
        "promptHash": "x" * 64,
        "params": {"editLevel": "standard", "allowRetiming": False},
        "cues": cues,
        "metrics": {
            "cueCount": len(cues),
            "changedCueRate": 0.5,
            "reviewCueRate": 0.25,
            "promptTokensTotal": 100,
            "completionTokensTotal": 50,
        },
    }


def test_aggregate_proofread_risk_histogram() -> None:
    cues = [
        {"cueId": "cue_000001", "sourceStart": 0, "sourceEnd": 1,
         "sourceText": "a", "correctedText": "a", "editLevel": "none",
         "risk": "low", "notes": []},
        {"cueId": "cue_000002", "sourceStart": 1, "sourceEnd": 2,
         "sourceText": "b", "correctedText": "B", "editLevel": "minor",
         "risk": "medium", "notes": ["numeric_change"]},
        {"cueId": "cue_000003", "sourceStart": 2, "sourceEnd": 3,
         "sourceText": "c", "correctedText": "", "editLevel": "major",
         "risk": "high", "notes": ["empty_or_deleted"]},
        {"cueId": "cue_000004", "sourceStart": 3, "sourceEnd": 4,
         "sourceText": "d", "correctedText": "?", "editLevel": "major",
         "risk": "blocking", "notes": ["schema_fail"]},
    ]
    agg = aggregate_proofread(_proofread_doc(cues))
    assert agg.cue_count == 4
    assert agg.risk_histogram == {"low": 1, "medium": 1, "high": 1, "blocking": 1}
    assert agg.changed_cue_rate == 0.5
    assert agg.review_cue_rate == 0.25
    assert agg.prompt_tokens_total == 100
    assert agg.completion_tokens_total == 50


def test_aggregate_proofread_unknown_risk_goes_to_blocking_bucket() -> None:
    """Codex P3 修正：缺失/未知 risk → blocking（不是悄悄归 low）。"""
    cues = [
        {"cueId": "cue_000001", "sourceStart": 0, "sourceEnd": 1,
         "sourceText": "a", "correctedText": "a", "editLevel": "none",
         "notes": []},  # risk 缺失
        {"cueId": "cue_000002", "sourceStart": 1, "sourceEnd": 2,
         "sourceText": "b", "correctedText": "B", "editLevel": "minor",
         "risk": "weird-value", "notes": []},  # risk 未知值
        {"cueId": "cue_000003", "sourceStart": 2, "sourceEnd": 3,
         "sourceText": "c", "correctedText": "C", "editLevel": "minor",
         "risk": "low", "notes": []},  # 已知合法值
    ]
    agg = aggregate_proofread(_proofread_doc(cues))
    assert agg.risk_histogram["blocking"] == 2
    assert agg.risk_histogram["low"] == 1
    assert agg.risk_histogram["medium"] == 0


def test_aggregate_proofread_note_histogram_counts_duplicates() -> None:
    cues = [
        {"cueId": "cue_000001", "sourceStart": 0, "sourceEnd": 1,
         "sourceText": "x", "correctedText": "y", "editLevel": "minor",
         "risk": "medium",
         "notes": ["numeric_change", "protected_term_change:Claude"]},
        {"cueId": "cue_000002", "sourceStart": 1, "sourceEnd": 2,
         "sourceText": "x", "correctedText": "z", "editLevel": "minor",
         "risk": "medium", "notes": ["numeric_change"]},
        {"cueId": "cue_000003", "sourceStart": 2, "sourceEnd": 3,
         "sourceText": "x", "correctedText": "w", "editLevel": "minor",
         "risk": "low", "notes": []},
    ]
    agg = aggregate_proofread(_proofread_doc(cues))
    assert agg.note_histogram == {
        "numeric_change": 2,
        "protected_term_change:Claude": 1,
    }
    # 未触发的桶仍为 0，确保 key 集稳定
    assert agg.risk_histogram["blocking"] == 0


# ── translation 聚合 ───────────────────────────────────────────────────────


def _translation_doc(
    cues: List[Dict[str, Any]],
    *,
    glossary_miss_rate: float = 0.0,
    over_char_limit_rate: float = 0.0,
) -> Dict[str, Any]:
    return {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "src1",
        "inputArtifact": "subtitles.cues.json",
        "inputHash": "sha256:abc",
        "sourceLanguage": "zh",
        "targetLanguage": "en",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "promptVersion": "translate.v1",
        "promptHash": "y" * 64,
        "params": {"style": "subtitle", "lengthPolicy": "preserve",
                   "cueMappingPolicy": "one-to-one"},
        "cues": cues,
        "metrics": {
            "cueCount": len(cues),
            "overCharLimitRate": over_char_limit_rate,
            "overCpsRate": 0.0,
            "glossaryMissRate": glossary_miss_rate,
            "promptTokensTotal": 200,
            "completionTokensTotal": 80,
        },
    }


def test_aggregate_translation_glossary_miss() -> None:
    cues = [
        {"id": "trg_000001", "sourceCueIds": ["cue_000001"], "start": 0.0,
         "end": 2.0, "speaker": "A", "text": "use AI here",
         "mapping": "one-to-one", "risk": "medium",
         "needsHumanReview": False, "notes": ["glossary_miss:Claude"]},
        {"id": "trg_000002", "sourceCueIds": ["cue_000002"], "start": 2.0,
         "end": 4.0, "speaker": "A", "text": "ok",
         "mapping": "one-to-one", "risk": "low", "notes": []},
    ]
    agg = aggregate_translation(
        _translation_doc(cues, glossary_miss_rate=0.5, over_char_limit_rate=0.0)
    )
    assert agg.cue_count == 2
    assert agg.glossary_miss_rate == 0.5
    assert agg.risk_histogram == {"low": 1, "medium": 1, "high": 0, "blocking": 0}
    assert agg.note_histogram == {"glossary_miss:Claude": 1}
    assert agg.prompt_tokens_total == 200


# ── build_quality_report 集成 ──────────────────────────────────────────────


def _cues_doc(
    cues: List[Dict[str, Any]] | None = None,
    *,
    source_id: str = "src1",
) -> Dict[str, Any]:
    cues = cues or [
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "hi"},
        {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "A", "text": "ok"},
    ]
    return {
        "schemaVersion": "2",
        "sourceId": source_id,
        "resegment": "semantic",
        "cues": cues,
    }


def test_build_quality_report_with_only_cues(tmp_path: Path) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps(_cues_doc()), encoding="utf-8"
    )

    report = build_quality_report(workdir)
    assert report.source_id == "src1"
    assert report.inputs == {"cues": "subtitles.cues.json"}
    assert report.cues_metrics is not None
    assert report.cues_metrics.cue_count == 2
    assert report.proofread_metrics is None
    assert report.proofread_cue_metrics is None
    assert report.translations == {}


def test_build_quality_report_with_proofread_and_two_translations(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps(_cues_doc()), encoding="utf-8"
    )
    proofread_cues = [
        {"cueId": "cue_000001", "sourceStart": 0.0, "sourceEnd": 2.0,
         "speaker": "A", "sourceText": "hi", "correctedText": "Hi.",
         "editLevel": "minor", "risk": "medium",
         "needsHumanReview": False, "notes": ["numeric_change"]},
        {"cueId": "cue_000002", "sourceStart": 2.0, "sourceEnd": 4.0,
         "speaker": "A", "sourceText": "ok", "correctedText": "ok",
         "editLevel": "none", "risk": "low",
         "needsHumanReview": False, "notes": []},
    ]
    (workdir / "subtitles.proofread.json").write_text(
        json.dumps(_proofread_doc(proofread_cues)), encoding="utf-8"
    )

    def _trans_cue(idx: int) -> Dict[str, Any]:
        return {
            "id": f"trg_{idx:06d}", "sourceCueIds": [f"cue_{idx:06d}"],
            "start": float(idx - 1) * 2.0, "end": float(idx) * 2.0,
            "speaker": "A", "text": "hello",
            "mapping": "one-to-one", "risk": "low",
            "needsHumanReview": False, "notes": [],
        }

    (workdir / "subtitles.en.json").write_text(
        json.dumps(_translation_doc([_trans_cue(1), _trans_cue(2)])),
        encoding="utf-8",
    )
    (workdir / "subtitles.ja.json").write_text(
        json.dumps(_translation_doc([_trans_cue(1)])),
        encoding="utf-8",
    )

    report = build_quality_report(workdir)
    assert set(report.inputs.keys()) == {"cues", "proofread", "translations"}
    assert report.inputs["translations"] == {
        "en": "subtitles.en.json",
        "ja": "subtitles.ja.json",
    }
    assert report.cues_metrics is not None
    assert report.proofread_metrics is not None
    assert report.proofread_metrics.risk_histogram == {
        "low": 1, "medium": 1, "high": 0, "blocking": 0,
    }
    assert report.proofread_cue_metrics is not None
    assert report.proofread_cue_metrics.cue_count == 2
    assert set(report.translations.keys()) == {"en", "ja"}
    assert report.translations["en"]["aggregate"]["cueCount"] == 2
    assert report.translations["ja"]["physical"]["cueCount"] == 1


def test_build_quality_report_no_inputs_raises(tmp_path: Path) -> None:
    workdir = tmp_path / "empty"
    workdir.mkdir()
    with pytest.raises(ValueError):
        build_quality_report(workdir)


def test_build_quality_report_ignores_unrelated_subtitle_json(
    tmp_path: Path,
) -> None:
    """``subtitles.proofread.json`` 和 ``subtitles.cues.json`` 不能被语言正则误识。"""
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps(_cues_doc()), encoding="utf-8"
    )
    (workdir / "subtitles.proofread.json").write_text(
        json.dumps(_proofread_doc([])), encoding="utf-8"
    )
    report = build_quality_report(workdir)
    assert report.translations == {}
    assert "translations" not in report.inputs


def test_write_quality_report_atomic_and_camelcase(tmp_path: Path) -> None:
    workdir = tmp_path / "ws"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps(_cues_doc()), encoding="utf-8"
    )
    report = build_quality_report(workdir)

    out_path = workdir / "quality.report.json"
    write_quality_report(out_path, report)

    text = out_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    # 无 tmp 残留
    leftovers = list(workdir.glob("quality.report.json.*.tmp"))
    assert leftovers == []

    doc = json.loads(text)
    # camelCase 别名落盘
    assert doc["schemaVersion"] == "1"
    assert doc["sourceId"] == "src1"
    assert "generatedAt" in doc
    assert doc["cuesMetrics"]["cueCount"] == 2
    assert "avgCueDurS" in doc["cuesMetrics"]
    assert "speakerSwitchCueRate" in doc["cuesMetrics"]
    # None 字段被剥离
    assert "proofreadMetrics" not in doc
    assert "proofreadCueMetrics" not in doc
