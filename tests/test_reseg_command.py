"""voxkit reseg 子命令单测。

策略：tmp_path 准备 proofread.json fixture（带逗号 / 句末标点），跑命令，
断言 reseg2.json 形态 + 切分行为 + 错误路径。零 LLM 零网络。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.commands.reseg import run as run_reseg


def _proof_doc(cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "state": "draft",
        "sourceId": "test",
        "language": "zh",
        "cues": cues,
    }


def _proof_cue(
    cid: str, start: float, end: float, text: str, speaker: str = "Speaker A"
) -> Dict[str, Any]:
    return {
        "cueId": cid,
        "sourceStart": start,
        "sourceEnd": end,
        "sourceText": text,
        "correctedText": text,
        "speaker": speaker,
        "editLevel": "minor",
        "risk": "low",
        "needsHumanReview": False,
        "notes": [],
    }


def _ns(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace with sensible defaults for reseg.run."""
    return argparse.Namespace(
        **{
            "language": None,
            "emit_srt": True,
            "force": False,
            "max_cue_duration": None,
            **kwargs,
        }
    )


# ── happy path ──────────────────────────────────────────────────────────────


def test_reseg_splits_proofread_cue_at_comma(tmp_path: Path) -> None:
    """proofread cue 含逗号 → reseg2 切成多条短 cue。

    需要 input cue 超过 soft_max_chars=18，packing 阶段才不会把切好的
    atom 重新合并回单条 cue（这正是真实 vlog 长 cue 的场景）。
    """
    workdir = tmp_path
    # 30+ 字符长 cue，含 2 逗号 + 1 句号
    long_text = "我今天去了一家很好吃的餐厅，点了几个招牌菜，味道真的特别棒。"
    assert len(long_text) > 18  # 确保超 soft_max 触发 packing flush
    (workdir / "subtitles.proofread.json").write_text(
        json.dumps(
            _proof_doc([
                _proof_cue("c1", 0.0, 6.0, long_text),
            ])
        ),
        encoding="utf-8",
    )
    rc = run_reseg(_ns(workdir=str(workdir)))
    assert rc == 0

    out = json.loads(
        (workdir / "subtitles.cues.reseg2.json").read_text(encoding="utf-8")
    )
    assert out["schemaVersion"] == 2
    assert out["sourceId"] == "test"
    cues = out["cues"]
    # 30+ 字符长 cue 含 2 逗号 + 1 句号 → reseg2 应切成多条
    assert len(cues) >= 2, f"长 cue 未被切: {[c['text'] for c in cues]}"
    # 文本完整保留
    assert "".join(c["text"] for c in cues) == long_text


def test_reseg_emits_srt_by_default(tmp_path: Path) -> None:
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    assert run_reseg(_ns(workdir=str(tmp_path))) == 0
    srt = tmp_path / "subtitles.reseg2.srt"
    assert srt.is_file()
    body = srt.read_text(encoding="utf-8")
    assert "测试" in body
    assert "-->" in body


def test_reseg_no_emit_srt_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    assert run_reseg(_ns(workdir=str(tmp_path), emit_srt=False)) == 0
    assert not (tmp_path / "subtitles.reseg2.srt").exists()


def test_reseg_preserves_speakers(tmp_path: Path) -> None:
    """speaker 跟着 cue 走，跨 speaker 不合并 reseg。"""
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(
            _proof_doc([
                _proof_cue("c1", 0.0, 2.0, "你好。", speaker="A"),
                _proof_cue("c2", 2.0, 4.0, "你好。", speaker="B"),
            ])
        ),
        encoding="utf-8",
    )
    assert run_reseg(_ns(workdir=str(tmp_path))) == 0
    cues = json.loads(
        (tmp_path / "subtitles.cues.reseg2.json").read_text(encoding="utf-8")
    )["cues"]
    speakers = [c.get("speaker") for c in cues]
    assert "A" in speakers and "B" in speakers


# ── error paths ─────────────────────────────────────────────────────────────


def test_reseg_fails_without_proofread(tmp_path: Path) -> None:
    rc = run_reseg(_ns(workdir=str(tmp_path)))
    assert rc != 0
    assert not (tmp_path / "subtitles.cues.reseg2.json").exists()


def test_reseg_fails_when_workdir_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    rc = run_reseg(_ns(workdir=str(missing)))
    assert rc != 0


def test_reseg_refuses_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    assert run_reseg(_ns(workdir=str(tmp_path))) == 0
    # 第二次跑应失败
    rc = run_reseg(_ns(workdir=str(tmp_path)))
    assert rc != 0


def test_reseg_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    assert run_reseg(_ns(workdir=str(tmp_path))) == 0
    # --force 强制覆盖
    assert run_reseg(_ns(workdir=str(tmp_path), force=True)) == 0


# ── F3: --max-cue-duration 透传 ──────────────────────────────────────────────


def test_reseg_max_cue_duration_passes_through_to_params(
    tmp_path: Path, monkeypatch
) -> None:
    """--max-cue-duration 透传到 ResegmentParams.max_dur_s。"""
    captured: Dict[str, Any] = {}

    import voxkit.commands.reseg as reseg_mod

    real_reseg = reseg_mod.resegment_for_subtitles

    def fake_reseg(segments, *, language=None, params=None):
        captured["max_dur_s"] = params.max_dur_s if params is not None else None
        return real_reseg(segments, language=language, params=params)

    monkeypatch.setattr(reseg_mod, "resegment_for_subtitles", fake_reseg)
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    rc = run_reseg(_ns(workdir=str(tmp_path), max_cue_duration=4.5))
    assert rc == 0
    assert captured["max_dur_s"] == pytest.approx(4.5)
    # params 也镜像到 out_doc["params"]["maxDurS"]
    out = json.loads(
        (tmp_path / "subtitles.cues.reseg2.json").read_text(encoding="utf-8")
    )
    assert out["params"]["maxDurS"] == pytest.approx(4.5)


def test_reseg_default_max_cue_duration_is_dataclass_default(
    tmp_path: Path, monkeypatch
) -> None:
    """不传 --max-cue-duration → ResegmentParams() 的默认值（7.0s）。"""
    from voxkit.core.semantic_resegment import ResegmentParams
    captured: Dict[str, Any] = {}

    import voxkit.commands.reseg as reseg_mod

    real_reseg = reseg_mod.resegment_for_subtitles

    def fake_reseg(segments, *, language=None, params=None):
        captured["max_dur_s"] = params.max_dur_s if params is not None else None
        return real_reseg(segments, language=language, params=params)

    monkeypatch.setattr(reseg_mod, "resegment_for_subtitles", fake_reseg)
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    rc = run_reseg(_ns(workdir=str(tmp_path)))
    assert rc == 0
    assert captured["max_dur_s"] == pytest.approx(ResegmentParams().max_dur_s)


def test_reseg_rejects_nonpositive_max_cue_duration(tmp_path: Path, capsys) -> None:
    """--max-cue-duration <= 0 必须 reject，不写出 reseg2.json。"""
    (tmp_path / "subtitles.proofread.json").write_text(
        json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
        encoding="utf-8",
    )
    for bad in (0.0, -1.0):
        # 每次创建新 workdir 避免 reseg2.json 残留
        sub = tmp_path / f"wd_{bad}"
        sub.mkdir()
        (sub / "subtitles.proofread.json").write_text(
            json.dumps(_proof_doc([_proof_cue("c1", 0.0, 2.0, "测试。")])),
            encoding="utf-8",
        )
        rc = run_reseg(_ns(workdir=str(sub), max_cue_duration=bad))
        assert rc != 0
        assert "max-cue-duration" in capsys.readouterr().err
        assert not (sub / "subtitles.cues.reseg2.json").exists()


# ── eval_metrics fallback ───────────────────────────────────────────────────


def test_eval_metrics_prefers_reseg2_over_proofread(tmp_path: Path) -> None:
    """load_voxkit_cues 优先级：reseg2 > proofread > cues。"""
    from voxkit.core.eval_metrics import load_voxkit_cues

    # 三者都存在
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
    (tmp_path / "subtitles.cues.reseg2.json").write_text(
        json.dumps({"cues": [{"start": 0, "end": 1, "text": "切"}]}),
        encoding="utf-8",
    )
    cues, source = load_voxkit_cues(tmp_path)
    assert source == "reseg2"
    assert cues[0]["text"] == "切"
