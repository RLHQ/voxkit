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


# ── B1 regression: SRT speaker prefix should be context-aware ───────────────


def test_translate_srt_single_speaker_placeholder_no_prefix(tmp_path: Path) -> None:
    """B1 修复：单 speaker（"Speaker A" 占位符）→ SRT 不再强加 "Speaker A:" 前缀。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "Speaker A", "text": "你好"},
        {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "Speaker A", "text": "世界"},
    ])
    fake = FakeLLMClient([_mock_translate_for(["cue_000001", "cue_000002"], ["hello", "world"])])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)

    srt = (ws.root / "subtitles.en.srt").read_text(encoding="utf-8")
    assert "Speaker A:" not in srt, "auto 模式不该把占位符渲染成前缀"
    assert "hello" in srt and "world" in srt

    # JSON 里 speaker 字段仍然保留（供下游消费）
    artifact = json.loads((ws.root / "subtitles.en.json").read_text(encoding="utf-8"))
    assert artifact["cues"][0]["speaker"] == "Speaker A"


def test_translate_srt_multi_speaker_keeps_prefix(tmp_path: Path) -> None:
    """多 speaker 时 auto 仍保留前缀，保护 diarization 信号不被一刀切掉。"""
    ws = open_workspace(tmp_path / "ws")
    # 默认 fixture 是 Speaker 1 / Speaker 2 双人
    _write_cues(ws.cues_json_path)
    fake = FakeLLMClient([
        _mock_translate_for(["cue_000001", "cue_000002"], ["hello", "world"]),
        _mock_translate_for(["cue_000003"], ["bye"]),
    ])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)

    srt = (ws.root / "subtitles.en.srt").read_text(encoding="utf-8")
    assert "Speaker 1: hello" in srt
    assert "Speaker 2: bye" in srt


def test_translate_speaker_prefix_always_forces_placeholder(tmp_path: Path) -> None:
    """speaker_prefix='always' 等同 v0.7.1 之前的旧行为，单人也强加前缀。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "Speaker A", "text": "你好"},
    ])
    fake = FakeLLMClient([_mock_translate_for(["cue_000001"], ["hello"])])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", speaker_prefix="always"),
        llm_client=fake,
    )
    srt = (ws.root / "subtitles.en.srt").read_text(encoding="utf-8")
    assert "Speaker A: hello" in srt


def test_translate_speaker_prefix_never_strips_even_multi(tmp_path: Path) -> None:
    """speaker_prefix='never' 强制移除前缀，即使是多 speaker。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)
    fake = FakeLLMClient([
        _mock_translate_for(["cue_000001", "cue_000002"], ["hello", "world"]),
        _mock_translate_for(["cue_000003"], ["bye"]),
    ])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en", speaker_prefix="never"),
        llm_client=fake,
    )
    srt = (ws.root / "subtitles.en.srt").read_text(encoding="utf-8")
    assert "Speaker 1:" not in srt
    assert "Speaker 2:" not in srt
    assert "hello" in srt and "bye" in srt


def test_translate_no_input_error_mentions_resegment(tmp_path: Path) -> None:
    """B3 修复：缺 cues.json 时报错应明确指向 transcribe --resegment=semantic。"""
    ws = open_workspace(tmp_path / "ws")
    with pytest.raises(FileNotFoundError) as exc:
        run_translate(
            TranslateRequest(workdir=ws.root, target_language="en"),
            llm_client=FakeLLMClient([]),
        )
    msg = str(exc.value)
    assert "--resegment=semantic" in msg
    assert "voxkit transcribe" in msg


# ── v0.7.2 review #3: --render-only short-circuit ──────────────────────────


def test_render_only_reuses_existing_artifact_no_llm(tmp_path: Path) -> None:
    """切换 --speaker-prefix 时 --render-only 应跳过 LLM，复用现有 JSON。"""
    ws = open_workspace(tmp_path / "ws")
    # 先用 auto 产出多 speaker fixture（auto → 渲染前缀）
    _write_cues(ws.cues_json_path)
    fake = FakeLLMClient([
        _mock_translate_for(["cue_000001", "cue_000002"], ["hello", "world"]),
        _mock_translate_for(["cue_000003"], ["bye"]),
    ])
    run_translate(TranslateRequest(workdir=ws.root, target_language="en"), llm_client=fake)
    srt_path = ws.root / "subtitles.en.srt"
    # 多 speaker auto → 默认有前缀
    assert "Speaker 1: hello" in srt_path.read_text(encoding="utf-8")

    # render-only 改成 never：不该再调 LLM，但 SRT 内容应变化
    no_llm = FakeLLMClient([])  # 空 canned response — 任何 chat() 调用都会 AssertionError
    summary = run_translate(
        TranslateRequest(
            workdir=ws.root,
            target_language="en",
            speaker_prefix="never",
            render_only=True,
        ),
        llm_client=no_llm,
    )
    assert no_llm.calls == [], "render-only 不应触发任何 LLM call"
    assert summary["renderOnly"] is True
    assert summary["speakerPrefix"] == "never"

    srt2 = srt_path.read_text(encoding="utf-8")
    assert "Speaker 1:" not in srt2
    assert "hello" in srt2 and "bye" in srt2


def test_render_only_fails_when_no_existing_artifact(tmp_path: Path) -> None:
    """缺 subtitles.<lang>.json 时 --render-only 应明确报错。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)
    with pytest.raises(FileNotFoundError) as exc:
        run_translate(
            TranslateRequest(
                workdir=ws.root, target_language="en", render_only=True
            ),
            llm_client=FakeLLMClient([]),
        )
    msg = str(exc.value)
    assert "render-only" in msg
    assert "subtitles.en.json" in msg


def test_render_only_rejects_force_combo(tmp_path: Path) -> None:
    """--render-only + --force 应是 ValueError（语义冲突）。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    # 先产出
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en"),
        llm_client=FakeLLMClient([_mock_translate_for(["cue_000001"], ["y"])]),
    )
    with pytest.raises(ValueError, match="render-only"):
        run_translate(
            TranslateRequest(
                workdir=ws.root,
                target_language="en",
                render_only=True,
                force_level="draft",
            ),
            llm_client=FakeLLMClient([]),
        )


def test_render_only_preserves_artifact_state(tmp_path: Path) -> None:
    """--render-only 不应修改 subtitles.<lang>.json 的 state / metrics。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path, cues=[
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
    ])
    run_translate(
        TranslateRequest(workdir=ws.root, target_language="en"),
        llm_client=FakeLLMClient([_mock_translate_for(["cue_000001"], ["y"])]),
    )
    json_path = ws.root / "subtitles.en.json"
    original_bytes = json_path.read_bytes()
    run_translate(
        TranslateRequest(
            workdir=ws.root,
            target_language="en",
            speaker_prefix="never",
            render_only=True,
        ),
        llm_client=FakeLLMClient([]),
    )
    assert json_path.read_bytes() == original_bytes, "render-only 不应改 JSON artifact"


# ── dry-run (F4) ────────────────────────────────────────────────────────────


def test_run_translate_dry_run_skips_llm_and_writes_nothing(tmp_path: Path) -> None:
    """``dry_run=True`` 不调 LLM、不写 artifact / work dir / SRT / VTT。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)

    fake = FakeLLMClient([])
    req = TranslateRequest(
        workdir=ws.root, target_language="en", dry_run=True
    )
    summary = run_translate(req, llm_client=fake)

    assert summary["dryRun"] is True
    assert summary["cueCount"] == 3
    assert summary["batchCount"] >= 1
    assert summary["estPromptTokens"] > 0
    assert summary["estCompletionTokens"] > 0
    assert summary["targetLanguage"] == "en"
    assert summary["estCostUsd"] is None  # 未指定 model
    assert len(fake.calls) == 0, "dry-run 严禁调 LLM"
    # 无 artifact / SRT / VTT / manifest
    assert not (ws.root / "subtitles.en.json").exists()
    assert not (ws.root / "subtitles.en.srt").exists()
    assert not (ws.root / "subtitles.en.vtt").exists()
    assert not ws.manifest_path.exists()


def test_run_translate_dry_run_known_model_returns_cost(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)

    req = TranslateRequest(
        workdir=ws.root,
        target_language="en",
        dry_run=True,
        model="deepseek-v4-flash",
    )
    summary = run_translate(req, llm_client=FakeLLMClient([]))
    assert summary["estCostUsd"] is not None
    assert summary["estCostUsd"] > 0
    assert summary["model"] == "deepseek-v4-flash"


def test_run_translate_dry_run_missing_input_raises(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    req = TranslateRequest(
        workdir=ws.root, target_language="en", dry_run=True
    )
    with pytest.raises(FileNotFoundError):
        run_translate(req, llm_client=FakeLLMClient([]))


def test_run_translate_dry_run_overrides_render_only(tmp_path: Path) -> None:
    """dry-run 优先级高于 render_only：哪怕没 subtitles.<lang>.json 也能 dry-run。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)
    # 没有 subtitles.en.json → render_only 单独会 FileNotFound；dry_run 拦截后正常
    req = TranslateRequest(
        workdir=ws.root,
        target_language="en",
        render_only=True,
        dry_run=True,
    )
    summary = run_translate(req, llm_client=FakeLLMClient([]))
    assert summary["dryRun"] is True


def test_run_translate_dry_run_not_blocked_by_stale_lock(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues(ws.cues_json_path)
    ws.root.mkdir(parents=True, exist_ok=True)
    lock_file = ws.root / ".lock"
    lock_file.write_text("stale", encoding="utf-8")

    summary = run_translate(
        TranslateRequest(
            workdir=ws.root, target_language="en", dry_run=True
        ),
        llm_client=FakeLLMClient([]),
    )
    assert summary["dryRun"] is True
    assert lock_file.read_text(encoding="utf-8") == "stale"
