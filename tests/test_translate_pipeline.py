"""``voxkit.core.translate_pipeline`` 集成测试。

策略与 ``test_proofread_pipeline`` 同：mock LLM + 写最小 fixture，断言：

  - artifact 形状（包含 sourceCueIds / targetLanguage / state）
  - SRT/VTT 渲染
  - manifest.translations[lang] 段
  - 输入优先级：proofread.json 优先于 cues.json
  - glossary_miss 风险评级与 metrics
  - schema 失败 → fallback 标人工，text=source
  - speaker 边界禁止跨批
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.translate_pipeline import (
    TranslateRequest,
    _build_batches,
    run_translate,
)
from voxkit.core.translate_pipeline import _SrcCue
from voxkit.core.workspace import open_workspace
from voxkit.llm.client import ChatResult


# ── fake LLM client ─────────────────────────────────────────────────────────


@dataclass
class _FakeCall:
    messages: List[Dict[str, Any]]


class FakeLLMClient:
    def __init__(self, responses: List[str], *, model: str = "deepseek-v4-flash") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: List[_FakeCall] = []

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of canned responses")
        resp = self._responses.pop(0)
        self.calls.append(_FakeCall(messages=list(messages)))
        return ChatResult(
            text=resp, prompt_tokens=80, completion_tokens=40,
            model=self._model, raw={},
        )

    def close(self) -> None:
        pass


# ── fixture helpers ─────────────────────────────────────────────────────────


def _write_cues(path: Path, cues: List[Dict[str, Any]] | None = None) -> None:
    cues = cues or [
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "你好"},
        {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "Speaker 1", "text": "世界"},
        {"id": "cue_000003", "start": 4.0, "end": 6.0, "speaker": "Speaker 2", "text": "再见"},
    ]
    path.write_text(json.dumps({
        "schemaVersion": "2",
        "sourceId": "fake_src",
        "resegment": "semantic",
        "params": {"language": "zh"},
        "cues": cues,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_proofread(path: Path, *, source_id: str = "fake_src") -> None:
    """写一个最小 proofread.json（来自上一阶段）。"""
    path.write_text(json.dumps({
        "schemaVersion": "1",
        "state": "draft",
        "sourceId": source_id,
        "inputArtifact": "subtitles.cues.json",
        "inputHash": "sha256:abc",
        "language": "zh",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "promptVersion": "proofread.v1",
        "promptHash": "x" * 64,
        "params": {"editLevel": "standard", "allowRetiming": False, "glossaryHash": None},
        "cues": [
            {"cueId": "cue_000001", "sourceStart": 0.0, "sourceEnd": 2.0, "speaker": "Speaker 1",
             "sourceText": "你好", "correctedText": "你好。", "editLevel": "minor", "risk": "low",
             "needsHumanReview": False, "notes": []},
        ],
        "metrics": {"cueCount": 1, "changedCueRate": 1.0, "reviewCueRate": 0.0,
                    "promptTokensTotal": 0, "completionTokensTotal": 0},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_translate_for(cue_ids: List[str], texts: List[str] | None = None) -> str:
    texts = texts or [f"translated:{cid}" for cid in cue_ids]
    return json.dumps({
        "cues": [
            {"cueId": cid, "translatedText": text, "needsHumanReview": False}
            for cid, text in zip(cue_ids, texts)
        ]
    }, ensure_ascii=False)


# ── batching ────────────────────────────────────────────────────────────────


def _mk(id_: str, speaker: str | None, text: str = "x") -> _SrcCue:
    return _SrcCue(id=id_, start=0.0, end=1.0, speaker=speaker, text=text)


def test_build_batches_respects_speaker() -> None:
    cues = [_mk("c1", "A"), _mk("c2", "A"), _mk("c3", "B")]
    batches = _build_batches(cues, max_tokens=10000, max_cues=10, context_prev=0, context_next=0)
    assert len(batches) == 2
    assert batches[0].target_idxs == [0, 1]
    assert batches[1].target_idxs == [2]


# ── pipeline integration ────────────────────────────────────────────────────


def test_run_translate_with_cues_input(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)

    fake = FakeLLMClient([
        _mock_translate_for(["cue_000001", "cue_000002"], ["hello", "world"]),
        _mock_translate_for(["cue_000003"], ["bye"]),
    ])
    req = TranslateRequest(workdir=ws.root, target_language="en")
    summary = run_translate(req, llm_client=fake)

    json_path = ws.root / "subtitles.en.json"
    srt_path = ws.root / "subtitles.en.srt"
    vtt_path = ws.root / "subtitles.en.vtt"
    assert json_path.exists() and srt_path.exists() and vtt_path.exists()

    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    assert artifact["schemaVersion"] == "1"
    assert artifact["state"] == "draft"
    assert artifact["targetLanguage"] == "en"
    assert artifact["sourceLanguage"] == "zh"
    assert artifact["inputArtifact"] == "subtitles.cues.json"
    assert artifact["inputHash"].startswith("sha256:")
    assert artifact["params"]["cueMappingPolicy"] == "one-to-one"

    cues = artifact["cues"]
    assert len(cues) == 3
    assert [c["text"] for c in cues] == ["hello", "world", "bye"]
    assert [c["sourceCueIds"] for c in cues] == [["cue_000001"], ["cue_000002"], ["cue_000003"]]
    assert [c["id"] for c in cues] == ["trg_000001", "trg_000002", "trg_000003"]
    # 时间从源 cue 继承
    assert cues[0]["start"] == 0.0 and cues[0]["end"] == 2.0
    assert cues[2]["speaker"] == "Speaker 2"

    # SRT 形态
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,000\nSpeaker 1: hello" in srt_text

    # VTT header
    vtt_text = vtt_path.read_text(encoding="utf-8")
    assert vtt_text.startswith("WEBVTT")

    # manifest
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["translations"]["en"]["state"] == "draft"
    assert manifest["translations"]["en"]["batchCount"] == 2
    assert manifest["artifacts"]["subtitle_translation_en_json"].endswith("subtitles.en.json")
    assert summary["state"] == "draft"


def test_run_translate_prefers_proofread_over_cues(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)
    _write_proofread(ws.proofread_json_path)

    fake = FakeLLMClient([_mock_translate_for(["cue_000001"], ["hello."])])
    req = TranslateRequest(workdir=ws.root, target_language="en")
    run_translate(req, llm_client=fake)

    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    assert artifact["inputArtifact"] == "subtitles.proofread.json"
    # proofread 用的是 correctedText "你好。"，所以源给 LLM 的是带句号版本
    # 这里通过 cue 数量验证（proofread 只 1 个 cue）
    assert len(artifact["cues"]) == 1


def test_run_translate_glossary_miss_flag(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "use Claude here"},
    ])
    glossary = tmp_path / "g.json"
    glossary.write_text(json.dumps({
        "version": 1,
        "terms": [{"source": "Claude", "target": "Claude AI", "casePolicy": "smart"}],
    }), encoding="utf-8")

    # LLM 忘了用指定译法
    fake = FakeLLMClient([_mock_translate_for(["cue_000001"], ["use AI here"])])
    req = TranslateRequest(workdir=ws.root, target_language="en", glossary_path=glossary)
    run_translate(req, llm_client=fake)

    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["risk"] == "medium"
    assert any("glossary_miss:Claude" in n for n in cue["notes"])
    assert artifact["metrics"]["glossaryMissRate"] == 1.0


def test_run_translate_invalid_json_fallback(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "你好"},
    ])
    fake = FakeLLMClient(["bad json", "still bad"])
    req = TranslateRequest(workdir=ws.root, target_language="en")
    run_translate(req, llm_client=fake)

    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["risk"] == "blocking"
    assert cue["needsHumanReview"] is True
    assert cue["text"] == "你好"  # fallback：源文本透传
    assert "schema_fail" in cue["notes"]


def test_run_translate_resume_uses_checkpoint(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)

    fake1 = FakeLLMClient([
        _mock_translate_for(["cue_000001", "cue_000002"]),
        _mock_translate_for(["cue_000003"]),
    ])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake1)

    # 删 artifact 保留 checkpoint
    (ws.root / "subtitles.en.json").unlink()
    fake2 = FakeLLMClient([])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake2)
    assert len(fake2.calls) == 0


def test_run_translate_force_wipes_checkpoint(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    fake1 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v1"])])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake1)

    fake2 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v2"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", force=True),
        llm_client=fake2,
    )
    assert len(fake2.calls) == 1
    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    assert artifact["cues"][0]["text"] == "v2"


def test_run_translate_existing_artifact_refuses(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    fake = FakeLLMClient([_mock_translate_for(["cue_000001"], ["y"])])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)

    with pytest.raises(FileExistsError):
        run_translate(
            TranslateRequest(workdir=ws.root, target_language="en"),
            llm_client=FakeLLMClient([]),
        )


def test_run_translate_no_input_raises(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    with pytest.raises(FileNotFoundError):
        run_translate(
            TranslateRequest(workdir=ws.root, target_language="en"),
            llm_client=FakeLLMClient([]),
        )


def test_run_translate_emit_flags_skip_outputs(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    fake = FakeLLMClient([_mock_translate_for(["cue_000001"], ["y"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", emit_srt=False, emit_vtt=False),
        llm_client=fake,
    )
    assert (ws.root / "subtitles.en.json").exists()
    assert not (ws.root / "subtitles.en.srt").exists()
    assert not (ws.root / "subtitles.en.vtt").exists()


# ── 回归测试：Codex 审查暴露的不变量 ──────────────────────────────────────


def test_run_translate_force_refuses_final_without_explicit_flag(tmp_path: Path) -> None:
    """Codex P1: --force 默认只覆盖 draft；final 状态需要 --force-final。"""
    from voxkit.core.lifecycle import transition_state

    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    fake1 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v1"])])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake1)

    json_path = ws.root / "subtitles.en.json"
    transition_state(json_path, to="reviewed", reviewer="alice")
    transition_state(json_path, to="final")
    assert json.loads(json_path.read_text(encoding="utf-8"))["state"] == "final"

    # 默认 --force 应被拒
    with pytest.raises(FileExistsError) as exc:
        run_translate(
            TranslateRequest(workdir=ws.root, target_language="en", force=True),
            llm_client=FakeLLMClient([_mock_translate_for(["cue_000001"], ["v2"])]),
        )
    assert "final" in str(exc.value)
    assert json.loads(json_path.read_text(encoding="utf-8"))["cues"][0]["text"] == "v1"

    # --force-reviewed 也应被拒（覆盖等级不够）
    with pytest.raises(FileExistsError):
        run_translate(
            TranslateRequest(workdir=ws.root, target_language="en", force_level="reviewed"),
            llm_client=FakeLLMClient([_mock_translate_for(["cue_000001"], ["v2"])]),
        )

    # --force-final 才能覆盖
    fake2 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v2"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", force_level="final"),
        llm_client=fake2,
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["cues"][0]["text"] == "v2"


def test_run_translate_checkpoint_invalidates_on_style_change(tmp_path: Path) -> None:
    """Codex P1: style 变化必须让 cache 失效。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    fake1 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v1"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", style="subtitle"),
        llm_client=fake1,
    )

    (ws.root / "subtitles.en.json").unlink()
    fake2 = FakeLLMClient([_mock_translate_for(["cue_000001"], ["v2"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", style="literal"),
        llm_client=fake2,
    )
    assert len(fake2.calls) == 1, "style 变了 cache 必须失效"
    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    assert artifact["cues"][0]["text"] == "v2"


def test_run_translate_empty_text_marks_blocking(tmp_path: Path) -> None:
    """Codex P2: 空 translatedText → repair → fallback blocking。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "你好"},
    ])
    empty_response = json.dumps({
        "cues": [{"cueId": "cue_000001", "translatedText": "", "needsHumanReview": False}]
    }, ensure_ascii=False)
    fake = FakeLLMClient([empty_response, empty_response])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)

    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["risk"] == "blocking"
    assert cue["needsHumanReview"] is True
    assert cue["text"] == "你好"
    assert "schema_fail" in cue["notes"]


def test_run_translate_transport_failure_writes_pending_marker(tmp_path: Path) -> None:
    """Codex P2: transport 错误 → pending marker，不写 artifact / SRT / VTT。"""
    from voxkit.llm.errors import LLMError, LLMRateLimit

    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
        {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "B", "text": "y"},
    ])

    class _RateLimitOnSecondBatch:
        def __init__(self) -> None:
            self._batch = 0
            self._model = "deepseek-v4-flash"
            self.calls = 0

        def chat(self, messages, **kw):  # type: ignore[no-untyped-def]
            self.calls += 1
            self._batch += 1
            if self._batch == 1:
                return ChatResult(
                    text=_mock_translate_for(["cue_000001"], ["hello"]),
                    prompt_tokens=10, completion_tokens=5,
                    model=self._model, raw={},
                )
            raise LLMRateLimit("rate limited", retry_after_secs=1.0)

        def close(self) -> None:
            pass

    fake = _RateLimitOnSecondBatch()
    work_dir = ws.work / "translate.en"
    with pytest.raises(LLMError, match="incomplete"):
        run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)

    assert (work_dir / "batch_000.json").exists()
    pending = work_dir / "batch_001.pending.json"
    assert pending.exists()
    assert not (ws.root / "subtitles.en.json").exists()
    assert not (ws.root / "subtitles.en.srt").exists()

    # rerun：第一批命中 cache，第二批 LLM
    fake2 = FakeLLMClient([_mock_translate_for(["cue_000002"], ["world"])])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake2)
    assert len(fake2.calls) == 1
    assert not pending.exists()
    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    assert [c["text"] for c in artifact["cues"]] == ["hello", "world"]
