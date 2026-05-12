"""``voxkit.core.lifecycle`` 单元测试。

覆盖：
  - artifact 类型识别
  - 自洽性校验（含重复 id）
  - 状态机的合法转换 / 反向 / 跨级 / 缺 reviewer
  - manifest 镜像 proofread / translation（翻译语言从文件名推断）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.lifecycle import (
    LifecycleError,
    detect_artifact_kind,
    mirror_to_manifest,
    transition_state,
    validate_self_consistency,
)
from voxkit.core.workspace import open_workspace, read_manifest


# ── fixture helpers ─────────────────────────────────────────────────────────


def _proofread_raw(cues: List[Dict[str, Any]] | None = None, **overrides: Any) -> Dict[str, Any]:
    cues = cues if cues is not None else [
        {
            "cueId": "cue_000001",
            "sourceStart": 0.0,
            "sourceEnd": 2.0,
            "speaker": "Speaker 1",
            "sourceText": "你好",
            "correctedText": "你好。",
            "editLevel": "minor",
            "risk": "low",
            "needsHumanReview": False,
            "notes": [],
        }
    ]
    data: Dict[str, Any] = {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "fake_src",
        "inputArtifact": "subtitles.cues.json",
        "inputHash": "sha256:abc",
        "language": "zh",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "promptVersion": "proofread.v1",
        "promptHash": "x" * 64,
        "params": {"editLevel": "standard", "allowRetiming": False, "glossaryHash": None},
        "cues": cues,
        "metrics": {
            "cueCount": len(cues),
            "changedCueRate": 0.0,
            "reviewCueRate": 0.0,
            "promptTokensTotal": 0,
            "completionTokensTotal": 0,
        },
    }
    data.update(overrides)
    return data


def _translation_raw(cues: List[Dict[str, Any]] | None = None, **overrides: Any) -> Dict[str, Any]:
    cues = cues if cues is not None else [
        {
            "id": "trg_000001",
            "sourceCueIds": ["cue_000001"],
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "text": "Hello.",
            "mapping": "one-to-one",
            "risk": "low",
            "needsHumanReview": False,
            "notes": [],
        }
    ]
    data: Dict[str, Any] = {
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": "fake_src",
        "inputArtifact": "subtitles.proofread.json",
        "inputHash": "sha256:abc",
        "sourceLanguage": "zh",
        "targetLanguage": "en",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "promptVersion": "translate.v1",
        "promptHash": "x" * 64,
        "params": {
            "style": "subtitle",
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
    data.update(overrides)
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── detect_artifact_kind ────────────────────────────────────────────────────


def test_detect_kind_proofread(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    _write_json(p, _proofread_raw())
    assert detect_artifact_kind(p) == "proofread"


def test_detect_kind_translation(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.en.json"
    _write_json(p, _translation_raw())
    assert detect_artifact_kind(p) == "translation"


def test_detect_kind_bad(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    _write_json(p, {"cues": [{"foo": "bar"}]})
    with pytest.raises(LifecycleError):
        detect_artifact_kind(p)

    # 文件不存在
    with pytest.raises(LifecycleError):
        detect_artifact_kind(tmp_path / "missing.json")

    # 空 cues
    empty = tmp_path / "empty.json"
    _write_json(empty, {"cues": []})
    with pytest.raises(LifecycleError):
        detect_artifact_kind(empty)


# ── validate_self_consistency ───────────────────────────────────────────────


def test_validate_self_consistency_ok() -> None:
    validate_self_consistency("proofread", _proofread_raw())
    validate_self_consistency("translation", _translation_raw())


def test_validate_self_consistency_duplicate_cue_id_raises() -> None:
    raw = _proofread_raw(cues=[
        {
            "cueId": "cue_000001",
            "sourceStart": 0.0, "sourceEnd": 1.0, "speaker": "Speaker 1",
            "sourceText": "a", "correctedText": "a", "editLevel": "none",
            "risk": "low", "needsHumanReview": False, "notes": [],
        },
        {
            "cueId": "cue_000001",  # 重复
            "sourceStart": 1.0, "sourceEnd": 2.0, "speaker": "Speaker 1",
            "sourceText": "b", "correctedText": "b", "editLevel": "none",
            "risk": "low", "needsHumanReview": False, "notes": [],
        },
    ])
    with pytest.raises(LifecycleError, match="duplicate"):
        validate_self_consistency("proofread", raw)

    raw_t = _translation_raw(cues=[
        {"id": "trg_000001", "sourceCueIds": ["c1"], "start": 0.0, "end": 1.0,
         "speaker": None, "text": "a", "mapping": "one-to-one", "risk": "low",
         "needsHumanReview": False, "notes": []},
        {"id": "trg_000001", "sourceCueIds": ["c2"], "start": 1.0, "end": 2.0,
         "speaker": None, "text": "b", "mapping": "one-to-one", "risk": "low",
         "needsHumanReview": False, "notes": []},
    ])
    with pytest.raises(LifecycleError, match="duplicate"):
        validate_self_consistency("translation", raw_t)


# ── transition_state ────────────────────────────────────────────────────────


def test_transition_draft_to_reviewed_sets_metadata(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    _write_json(p, _proofread_raw())

    updated = transition_state(p, to="reviewed", reviewer="Alice")
    assert updated["state"] == "reviewed"
    assert updated["reviewedBy"] == "Alice"
    assert updated["reviewedAt"].endswith("Z")

    # 写盘后再读一次，确认持久化
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["state"] == "reviewed"
    assert on_disk["reviewedBy"] == "Alice"


def test_transition_reviewed_to_final_preserves_reviewer(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    raw = _proofread_raw(state="reviewed", reviewedBy="Bob", reviewedAt="2026-01-01T00:00:00Z")
    _write_json(p, raw)

    updated = transition_state(p, to="final")
    assert updated["state"] == "final"
    assert updated["reviewedBy"] == "Bob"
    assert updated["reviewedAt"] == "2026-01-01T00:00:00Z"


def test_transition_invalid_skip_draft_to_final_raises(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    _write_json(p, _proofread_raw())
    with pytest.raises(LifecycleError, match="cannot transition"):
        transition_state(p, to="final")


def test_transition_backward_raises(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    # reviewed → 'reviewed' 重复也算非法 +0
    _write_json(p, _proofread_raw(state="reviewed", reviewedBy="X", reviewedAt="Y"))
    with pytest.raises(LifecycleError, match="cannot transition"):
        transition_state(p, to="reviewed", reviewer="Z")

    # final → reviewed（反向）
    _write_json(p, _proofread_raw(state="final", reviewedBy="X", reviewedAt="Y"))
    with pytest.raises(LifecycleError, match="cannot transition"):
        transition_state(p, to="reviewed", reviewer="Z")


def test_transition_confirm_requires_reviewer(tmp_path: Path) -> None:
    p = tmp_path / "subtitles.proofread.json"
    _write_json(p, _proofread_raw())
    with pytest.raises(LifecycleError, match="reviewer"):
        transition_state(p, to="reviewed", reviewer="")
    with pytest.raises(LifecycleError, match="reviewer"):
        transition_state(p, to="reviewed", reviewer="   ")
    with pytest.raises(LifecycleError, match="reviewer"):
        transition_state(p, to="reviewed", reviewer=None)


# ── mirror_to_manifest ──────────────────────────────────────────────────────


def test_mirror_to_manifest_proofread(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path)
    p = ws.proofread_json_path
    _write_json(p, _proofread_raw())

    updated = transition_state(p, to="reviewed", reviewer="Alice")
    mirror_to_manifest(tmp_path, p, updated)

    manifest = read_manifest(ws)
    assert manifest is not None
    assert manifest["proofread"]["state"] == "reviewed"
    assert manifest["proofread"]["reviewedBy"] == "Alice"
    assert manifest["proofread"]["reviewedAt"].endswith("Z")


def test_mirror_to_manifest_translation_lang_inferred(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path)
    p = tmp_path / "subtitles.en.json"
    _write_json(p, _translation_raw())

    updated = transition_state(p, to="reviewed", reviewer="Carol")
    mirror_to_manifest(tmp_path, p, updated)

    manifest = read_manifest(ws)
    assert manifest is not None
    assert manifest["translations"]["en"]["state"] == "reviewed"
    assert manifest["translations"]["en"]["reviewedBy"] == "Carol"

    # 文件名不匹配则报错
    bad = tmp_path / "subtitles.json"
    _write_json(bad, _translation_raw())
    with pytest.raises(LifecycleError, match="infer translation language"):
        mirror_to_manifest(tmp_path, bad, updated)
