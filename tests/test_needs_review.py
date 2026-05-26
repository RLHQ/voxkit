"""voxkit needs-review 子命令单测。

策略：tmp_path 写最小但 schema-合法的 subtitles.proofread.json 与
subtitles.zh.json，捕获 stdout/stderr 断言。read-only，零 LLM 零网络。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.commands.needs_review import run as run_needs_review


# ─── fixture helpers ────────────────────────────────────────────────────────


def _proof_cue(
    cid: str,
    text: str,
    *,
    risk: str = "low",
    needs: bool = False,
    start: float = 0.0,
    end: float = 1.0,
) -> Dict[str, Any]:
    return {
        "cueId": cid,
        "sourceStart": start,
        "sourceEnd": end,
        "speaker": "Speaker A",
        "sourceText": text,
        "correctedText": text,
        "editLevel": "minor",
        "risk": risk,
        "needsHumanReview": needs,
        "notes": [],
    }


def _proof_doc(cues: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "test",
        "inputArtifact": "subtitles.cues.json",
        "inputHash": "deadbeef",
        "language": "zh",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "promptVersion": "v1",
        "promptHash": "cafebabe",
        "params": {
            "editLevel": "standard",
            "allowRetiming": False,
            "glossaryHash": None,
        },
        "cues": cues,
        "metrics": {
            "cueCount": len(cues),
            "changedCueRate": 0.0,
            "reviewCueRate": 0.0,
            "promptTokensTotal": 0,
            "completionTokensTotal": 0,
        },
    }


def _trans_cue(
    cid: str,
    text: str,
    *,
    risk: str = "low",
    needs: bool = False,
    start: float = 0.0,
    end: float = 1.0,
) -> Dict[str, Any]:
    return {
        "id": cid,
        "sourceCueIds": [f"src_{cid}"],
        "start": start,
        "end": end,
        "speaker": "Speaker A",
        "text": text,
        "mapping": "one-to-one",
        "risk": risk,
        "needsHumanReview": needs,
        "notes": [],
    }


def _trans_doc(cues: List[Dict[str, Any]], target: str = "zh") -> Dict[str, Any]:
    return {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "test",
        "inputArtifact": "subtitles.proofread.json",
        "inputHash": "deadbeef",
        "sourceLanguage": "en",
        "targetLanguage": target,
        "provider": "deepseek",
        "model": "deepseek-chat",
        "promptVersion": "v1",
        "promptHash": "cafebabe",
        "params": {
            "style": "natural",
            "lengthPolicy": "preserve",
            "cueMappingPolicy": "one-to-one",
            "glossaryHash": None,
        },
        "cues": cues,
        "metrics": {
            "cueCount": len(cues),
            "overCharLimitRate": 0.0,
            "overCpsRate": 0.0,
            "glossaryMissRate": 0.0,
            "promptTokensTotal": 0,
            "completionTokensTotal": 0,
        },
    }


def _ns(**overrides: Any) -> argparse.Namespace:
    defaults: Dict[str, Any] = {
        "workdir": None,
        "target": None,
        "format": "text",
        "include_risk": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_proofread(workdir: Path, cues: List[Dict[str, Any]]) -> Path:
    p = workdir / "subtitles.proofread.json"
    p.write_text(json.dumps(_proof_doc(cues), ensure_ascii=False), encoding="utf-8")
    return p


# ─── happy path: text format ───────────────────────────────────────────────


def test_text_format_lists_flagged_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """5 cue 输入，2 个 needsHumanReview=True → text 输出 2 行 + stderr summary。"""
    cues = [
        _proof_cue("cue_000001", "low risk one"),
        _proof_cue("cue_000002", "needs review one", needs=True),
        _proof_cue("cue_000003", "low risk two"),
        _proof_cue("cue_000004", "needs review two", needs=True, risk="medium"),
        _proof_cue("cue_000005", "low risk three"),
    ]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(_ns(workdir=str(tmp_path)))
    cap = capsys.readouterr()

    assert rc == 0
    lines = [ln for ln in cap.out.splitlines() if ln.strip()]
    assert len(lines) == 2, f"expected 2 flagged lines, got: {cap.out!r}"
    assert "cue_000002" in cap.out
    assert "cue_000004" in cap.out
    assert "cue_000001" not in cap.out
    assert "2 cue(s) flagged out of 5" in cap.err


def test_text_format_includes_high_risk_even_without_needs_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """risk=high 即便 needsHumanReview=False 也应该被列出（默认 risk 过滤）。"""
    cues = [
        _proof_cue("cue_000001", "low"),
        _proof_cue("cue_000002", "high risk", risk="high", needs=False),
        _proof_cue("cue_000003", "blocking risk", risk="blocking", needs=False),
    ]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(_ns(workdir=str(tmp_path)))
    cap = capsys.readouterr()

    assert rc == 0
    assert "cue_000002" in cap.out
    assert "cue_000003" in cap.out
    assert "cue_000001" not in cap.out
    assert "2 cue(s) flagged out of 3" in cap.err


def test_text_format_renders_timestamp_and_preview(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """text 行应含 SRT 时间戳与截断的 text preview。"""
    long_text = "x" * 80  # > 60 char preview cutoff
    cues = [
        _proof_cue(
            "cue_000099",
            long_text,
            needs=True,
            start=323.4,  # 5:23.400 → 00:05:23,400
            end=325.0,
        ),
    ]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(_ns(workdir=str(tmp_path)))
    cap = capsys.readouterr()

    assert rc == 0
    assert "00:05:23,400" in cap.out
    assert "cue_000099" in cap.out
    # 80 char text 应被截断为 60 + "..."
    assert "x" * 60 + "..." in cap.out


# ─── empty queue ────────────────────────────────────────────────────────────


def test_empty_queue_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """0 flagged 仍然 exit 0，stdout 空，stderr summary 提示。"""
    cues = [_proof_cue("cue_000001", "fine"), _proof_cue("cue_000002", "fine")]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(_ns(workdir=str(tmp_path)))
    cap = capsys.readouterr()

    assert rc == 0
    assert cap.out.strip() == ""
    assert "0 cue(s) flagged out of 2" in cap.err


# ─── json format ────────────────────────────────────────────────────────────


def test_json_format_returns_parseable_cue_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--format json 输出可解析回 cue list，字段名保持 camelCase（by_alias）。"""
    cues = [
        _proof_cue("cue_000001", "low"),
        _proof_cue("cue_000002", "needs", needs=True),
    ]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(_ns(workdir=str(tmp_path), format="json"))
    cap = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(cap.out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["cueId"] == "cue_000002"
    assert parsed[0]["needsHumanReview"] is True
    assert "1 cue(s) flagged out of 2" in cap.err


# ─── --target & fallback ────────────────────────────────────────────────────


def test_target_reads_translation_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--target zh 直接读 subtitles.zh.json（translation schema，cue 字段 id）。"""
    trans_cues = [
        _trans_cue("trg_000001", "translated low"),
        _trans_cue("trg_000002", "translated needs", needs=True),
    ]
    (tmp_path / "subtitles.zh.json").write_text(
        json.dumps(_trans_doc(trans_cues, target="zh"), ensure_ascii=False),
        encoding="utf-8",
    )

    rc = run_needs_review(_ns(workdir=str(tmp_path), target="zh"))
    cap = capsys.readouterr()

    assert rc == 0
    assert "trg_000002" in cap.out
    assert "trg_000001" not in cap.out
    assert "1 cue(s) flagged out of 2" in cap.err


def test_target_falls_back_to_proofread_when_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--target zh 但 subtitles.zh.json 不存在 → fallback 到 subtitles.proofread.json。"""
    cues = [_proof_cue("cue_000001", "needs", needs=True)]
    _write_proofread(tmp_path, cues)
    # 注意：故意不写 subtitles.zh.json

    rc = run_needs_review(_ns(workdir=str(tmp_path), target="zh"))
    cap = capsys.readouterr()

    assert rc == 0
    assert "cue_000001" in cap.out
    assert "1 cue(s) flagged out of 1" in cap.err


# ─── error: missing file & bad workdir ──────────────────────────────────────


def test_missing_proofread_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """workdir 存在但 subtitles.proofread.json 不存在 → exit 2 + stderr error。"""
    rc = run_needs_review(_ns(workdir=str(tmp_path)))
    cap = capsys.readouterr()

    assert rc == 2
    assert "error" in cap.err.lower()
    assert "subtitles.proofread.json" in cap.err


def test_missing_workdir_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """workdir 路径不是目录 → exit 2。"""
    bogus = tmp_path / "does_not_exist"
    rc = run_needs_review(_ns(workdir=str(bogus)))
    cap = capsys.readouterr()

    assert rc == 2
    assert "workdir" in cap.err


def test_missing_target_with_no_fallback_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--target zh 且 subtitles.zh.json 与 subtitles.proofread.json 都不存在 → exit 2。"""
    rc = run_needs_review(_ns(workdir=str(tmp_path), target="zh"))
    cap = capsys.readouterr()

    assert rc == 2
    assert "error" in cap.err.lower()


# ─── --include-risk override ────────────────────────────────────────────────


def test_include_risk_override_widens_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--include-risk medium,high 把 medium cue 也纳入（默认只 high/blocking）。"""
    cues = [
        _proof_cue("cue_000001", "low"),
        _proof_cue("cue_000002", "medium", risk="medium"),
        _proof_cue("cue_000003", "high", risk="high"),
    ]
    _write_proofread(tmp_path, cues)

    rc = run_needs_review(
        _ns(workdir=str(tmp_path), include_risk="medium,high")
    )
    cap = capsys.readouterr()

    assert rc == 0
    assert "cue_000002" in cap.out
    assert "cue_000003" in cap.out
    assert "cue_000001" not in cap.out
    assert "2 cue(s) flagged out of 3" in cap.err


def test_include_risk_invalid_value_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--include-risk 含非法 risk 等级 → exit 2。"""
    _write_proofread(tmp_path, [_proof_cue("c1", "x")])
    rc = run_needs_review(
        _ns(workdir=str(tmp_path), include_risk="bogus")
    )
    cap = capsys.readouterr()

    assert rc == 2
    assert "unknown risk" in cap.err.lower()
