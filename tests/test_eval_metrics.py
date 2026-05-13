"""voxkit.core.eval_metrics 单测。

策略：tmp_path 写小型 SRT/JSON fixture，手算可验证。零 LLM、零网络。
fixture 设计见模块顶部 `_make_voxkit_cues` / `_make_gold_cues`，每个指标
都对应可推演的预期值。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.eval_metrics import (
    EvalReport,
    avg_drift,
    boundary_metrics,
    broken_latin_words,
    build_eval_report,
    density_ratio,
    load_voxkit_cues,
    parse_srt,
    write_eval_report,
)


# ── 共用 fixture ────────────────────────────────────────────────────────────


def _make_voxkit_cues() -> List[Dict[str, Any]]:
    """5 条 voxkit cue，包含 Hello/W + orld 跨 cue 切断。"""
    return [
        {"start": 0.0, "end": 1.0, "text": "你好世界"},
        {"start": 1.0, "end": 2.0, "text": "我叫张三"},
        {"start": 2.0, "end": 4.0, "text": "今天天气 真不错"},
        {"start": 4.0, "end": 5.0, "text": "Hello W"},      # 末尾拉丁字母碎
        {"start": 5.0, "end": 6.0, "text": "orld 再见"},     # 开头拉丁词尾
    ]


def _make_gold_cues() -> List[Dict[str, Any]]:
    """7 条金标 cue。"""
    return [
        {"start": 0.0, "end": 0.5, "text": "你好"},
        {"start": 0.5, "end": 1.0, "text": "世界"},
        {"start": 1.0, "end": 2.0, "text": "我叫张三"},
        {"start": 2.0, "end": 3.0, "text": "今天天气"},
        {"start": 3.0, "end": 4.0, "text": "真不错"},
        {"start": 4.0, "end": 5.5, "text": "Hello world"},
        {"start": 5.5, "end": 6.0, "text": "再见"},
    ]


def _write_srt(path: Path, cues: List[Dict[str, Any]]) -> None:
    """把 cues 写成 SRT。"""
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt(c['start'])} --> {fmt(c['end'])}")
        lines.append(c["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── parse_srt ───────────────────────────────────────────────────────────────


def test_parse_srt_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.srt"
    _write_srt(p, _make_gold_cues())
    parsed = parse_srt(p)
    assert len(parsed) == 7
    assert parsed[0]["start"] == 0.0 and parsed[0]["end"] == 0.5
    assert parsed[0]["text"] == "你好"
    assert parsed[5]["text"] == "Hello world"


def test_parse_srt_handles_trailing_blank(tmp_path: Path) -> None:
    p = tmp_path / "x.srt"
    p.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhi\n\n\n\n2\n00:00:01,000 --> 00:00:02,000\nbye\n",
        encoding="utf-8",
    )
    parsed = parse_srt(p)
    assert [c["text"] for c in parsed] == ["hi", "bye"]


# ── load_voxkit_cues ────────────────────────────────────────────────────────


def test_load_voxkit_prefers_proofread(tmp_path: Path) -> None:
    (tmp_path / "subtitles.cues.json").write_text(
        json.dumps({"cues": [{"start": 0, "end": 1, "text": "raw"}]}),
        encoding="utf-8",
    )
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps({
            "cues": [
                {"sourceStart": 0, "sourceEnd": 1, "correctedText": "校"}
            ]
        }),
        encoding="utf-8",
    )
    cues, src = load_voxkit_cues(tmp_path)
    assert src == "proofread"
    assert cues[0]["text"] == "校"


def test_load_voxkit_fallback_cues(tmp_path: Path) -> None:
    (tmp_path / "subtitles.cues.json").write_text(
        json.dumps({"cues": [{"start": 0, "end": 1, "text": "raw"}]}),
        encoding="utf-8",
    )
    cues, src = load_voxkit_cues(tmp_path)
    assert src == "cues"
    assert cues[0]["text"] == "raw"


def test_load_voxkit_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no voxkit subtitle artifact"):
        load_voxkit_cues(tmp_path)


# ── density_ratio ───────────────────────────────────────────────────────────


def test_density_ratio() -> None:
    assert density_ratio(_make_voxkit_cues(), _make_gold_cues()) == pytest.approx(5 / 7)


def test_density_ratio_zero_gold() -> None:
    # 防御除零
    assert density_ratio([{"start": 0, "end": 1, "text": "x"}], []) == 0.0


# ── boundary_metrics ────────────────────────────────────────────────────────


def test_boundary_metrics_tol_03() -> None:
    """手算（tol=0.3s）：
    vk_bounds  = [0, 1, 2, 4, 5, 6]               (6 个)
    gold_bounds = [0, 0.5, 1, 2, 3, 4, 5.5, 6]     (8 个)
    vk 命中 gold: 0✓ 1✓ 2✓ 4✓ 5(距 5.5=0.5)✗ 6✓ → 5/6
    gold 被 vk 覆盖: 0✓ 0.5(距 0/1 都 =0.5)✗ 1✓ 2✓ 3(距 2/4 都 =1)✗ 4✓ 5.5(距 5/6 都 =0.5)✗ 6✓ → 5/8
    """
    m = boundary_metrics(_make_voxkit_cues(), _make_gold_cues(), tol_s=0.3)
    assert m["vk_bounds_count"] == 6
    assert m["gold_bounds_count"] == 8
    assert m["vk_hits"] == 5
    assert m["gold_covered"] == 5
    assert m["precision"] == pytest.approx(5 / 6)
    assert m["recall"] == pytest.approx(5 / 8)
    expected_f1 = 2 * (5 / 6) * (5 / 8) / (5 / 6 + 5 / 8)
    assert m["f1"] == pytest.approx(expected_f1)


def test_boundary_metrics_tol_05_loosens() -> None:
    """tol=0.5s 比 tol=0.3 更宽：gold 3.0 仍不命中（vk 最近 2.0/4.0 都距 1.0），
    其余 7 个金标边界全中（0/0.5/1/2/4/5.5/6）→ recall = 7/8。
    """
    m = boundary_metrics(_make_voxkit_cues(), _make_gold_cues(), tol_s=0.5)
    assert m["gold_covered"] == 7
    assert m["recall"] == pytest.approx(7 / 8)
    # 与 tol=0.3 (5/8) 相比 recall 确实提升
    m03 = boundary_metrics(_make_voxkit_cues(), _make_gold_cues(), tol_s=0.3)
    assert m["recall"] > m03["recall"]


def test_boundary_metrics_empty_inputs() -> None:
    m = boundary_metrics([], [], tol_s=0.3)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ── avg_drift ───────────────────────────────────────────────────────────────


def test_avg_drift() -> None:
    """vk avg 字符 = (4+4+8+7+7)/5 = 6.0  (cue 3 "今天天气 真不错" 含 1 空格 = 8)
    gold avg 字符 = (2+2+4+4+3+11+2)/7 = 4.0
    vk avg dur = (1+1+2+1+1)/5 = 1.2；gold = (0.5+0.5+1+1+1+1.5+0.5)/7 = 6/7
    """
    d = avg_drift(_make_voxkit_cues(), _make_gold_cues())
    assert d["vk_avg_chars"] == pytest.approx(6.0)
    assert d["gold_avg_chars"] == pytest.approx(4.0)
    assert d["chars_drift"] == pytest.approx(2.0)
    assert d["vk_avg_dur_s"] == pytest.approx(1.2)
    assert d["gold_avg_dur_s"] == pytest.approx(6 / 7)
    assert d["dur_drift_s"] == pytest.approx(1.2 - 6 / 7)


# ── broken_latin_words ──────────────────────────────────────────────────────


def test_broken_latin_words_detects_split() -> None:
    """voxkit fixture 里 cue 4 末尾 'Hello W' + cue 5 开头 'orld 再见'
    应识别为 1 处切断。
    """
    n = broken_latin_words(_make_voxkit_cues())
    assert n == 1


def test_broken_latin_words_clean_text_returns_zero() -> None:
    """完整词不应误报。"""
    cues = [
        {"start": 0, "end": 1, "text": "Hello world"},
        {"start": 1, "end": 2, "text": "再见"},
    ]
    assert broken_latin_words(cues) == 0


def test_broken_latin_words_capital_start_not_broken() -> None:
    """下一 cue 开头大写字母通常是新词起点，不算切断。"""
    cues = [
        {"start": 0, "end": 1, "text": "买了 A"},
        {"start": 1, "end": 2, "text": "Tesla 真不错"},
    ]
    assert broken_latin_words(cues) == 0


def test_broken_latin_words_skipped_for_non_cjk() -> None:
    """启发式是为 CJK 设计的；EN 句末介词 + 下条小写起头属正常英文结构，不算切断。"""
    cues = [
        {"start": 0, "end": 1, "text": "consequences of"},
        {"start": 1, "end": 2, "text": "a fertility crisis"},
        {"start": 2, "end": 3, "text": "made worse by the"},
        {"start": 3, "end": 4, "text": "mismanagement"},
    ]
    # 默认 language=None 时保持 CJK-friendly 行为（仍会匹配——为不破坏现有 API）
    # 显式传 EN 时跳过整个检测
    assert broken_latin_words(cues, language="en") == 0


def test_broken_latin_words_still_active_for_cjk_explicit() -> None:
    """显式 language=zh 时启发式照常生效。"""
    cues = [
        {"start": 0, "end": 1, "text": "买了 S"},
        {"start": 1, "end": 2, "text": "team 真不错"},
    ]
    assert broken_latin_words(cues, language="zh") == 1


# ── build_eval_report + write ───────────────────────────────────────────────


def test_build_eval_report_e2e(tmp_path: Path) -> None:
    """端到端：写 voxkit cues.json + 金标 SRT → build_eval_report → 校所有顶层字段。"""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps({"cues": _make_voxkit_cues(), "sourceId": "test"}),
        encoding="utf-8",
    )
    ref = tmp_path / "gold.srt"
    _write_srt(ref, _make_gold_cues())

    report = build_eval_report(
        workdir=workdir,
        reference=ref,
        language="zh",
        tolerance_s=0.3,
    )
    assert isinstance(report, EvalReport)
    assert report.schemaVersion == 1
    assert report.language == "zh"
    assert report.sourceArtifact == "cues"
    assert report.tolerance_s == 0.3
    assert report.alignment["vk_cues"] == 5
    assert report.alignment["gold_cues"] == 7
    assert report.alignment["density_ratio"] == pytest.approx(5 / 7)
    assert report.metrics["boundary_precision"] == pytest.approx(5 / 6)
    assert report.metrics["boundary_recall"] == pytest.approx(5 / 8)
    assert report.metrics["broken_latin_words"] == 1


def test_write_eval_report_roundtrip(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "subtitles.cues.json").write_text(
        json.dumps({"cues": _make_voxkit_cues()}),
        encoding="utf-8",
    )
    ref = tmp_path / "gold.srt"
    _write_srt(ref, _make_gold_cues())

    report = build_eval_report(workdir, ref, "zh", 0.3)
    out = tmp_path / "eval.report.json"
    write_eval_report(out, report)
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 1
    assert payload["alignment"]["vk_cues"] == 5
