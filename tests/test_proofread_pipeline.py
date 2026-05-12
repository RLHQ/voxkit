"""``voxkit.core.proofread_pipeline`` 集成测试。

策略：直接构造一个最小可用的 ``subtitles.cues.json`` (schemaVersion=2) 写入
workspace，再用一个 fake LLMClient 喂确定的 JSON 响应，断言：

  - artifact 形状（schemaVersion / state / inputHash / cueId 覆盖）
  - manifest 镜像段写入
  - events.ndjson 含 batch.start/done 与 proofread.done
  - per-batch checkpoint 命中后跳过 LLM
  - schema 失败 → 整批 fallback 标人工
  - speaker 边界禁止跨批
  - glossary 注入 protected_terms 后会触发 risk
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

from voxkit.core.proofread_pipeline import (
    ProofreadRequest,
    _build_batches,
    run_proofread,
)
from voxkit.core.workspace import open_workspace
from voxkit.io.schema import SubtitleCueOut
from voxkit.llm.client import ChatResult


# ── fake LLM client ──────────────────────────────────────────────────────────


@dataclass
class _FakeCall:
    messages: List[Dict[str, Any]]
    response: str


class FakeLLMClient:
    """记录调用 + 按预设序列返回。"""

    def __init__(
        self,
        responses: List[str],
        *,
        model: str = "deepseek-chat",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        self._responses = list(responses)
        self._model = model
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.calls: List[_FakeCall] = []

    def chat(self, messages, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of canned responses")
        resp = self._responses.pop(0)
        self.calls.append(_FakeCall(messages=list(messages), response=resp))
        return ChatResult(
            text=resp,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            model=self._model,
            raw={},
        )

    def close(self) -> None:
        pass


# ── fixture helpers ──────────────────────────────────────────────────────────


def _write_cues_v2(
    path: Path,
    *,
    source_id: str = "fake_src",
    cues: List[Dict[str, Any]] | None = None,
) -> None:
    cues = cues or [
        {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "你好 世界"},
        {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "Speaker 1", "text": "我喜欢 coding"},
        {"id": "cue_000003", "start": 4.0, "end": 6.0, "speaker": "Speaker 2", "text": "really cool stuff"},
    ]
    doc = {
        "schemaVersion": "2",
        "sourceId": source_id,
        "resegment": "semantic",
        "params": {"language": "zh"},
        "cues": cues,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_response_for(cue_ids: List[str], *, suffix: str = "") -> str:
    """构造一条覆盖给定 cueId 的 LLM JSON 响应。"""
    return json.dumps(
        {
            "cues": [
                {"cueId": cid, "correctedText": f"corrected:{cid}{suffix}", "needsHumanReview": False}
                for cid in cue_ids
            ]
        },
        ensure_ascii=False,
    )


# ── batching unit tests ──────────────────────────────────────────────────────


def _mk(id_: str, speaker: str | None, text: str) -> SubtitleCueOut:
    return SubtitleCueOut(id=id_, start=0.0, end=1.0, speaker=speaker, text=text)


def test_build_batches_respects_speaker_boundary() -> None:
    cues = [
        _mk("cue_000001", "A", "x"),
        _mk("cue_000002", "A", "y"),
        _mk("cue_000003", "B", "z"),
        _mk("cue_000004", "B", "w"),
    ]
    batches = _build_batches(cues, max_tokens=10000, max_cues=10, context_prev=0, context_next=0)
    assert len(batches) == 2
    assert batches[0].target_idxs == [0, 1]
    assert batches[1].target_idxs == [2, 3]


def test_build_batches_respects_cue_count_cap() -> None:
    cues = [_mk(f"cue_{i+1:06d}", "A", "x") for i in range(7)]
    batches = _build_batches(cues, max_tokens=10000, max_cues=3, context_prev=0, context_next=0)
    assert [len(b.target_idxs) for b in batches] == [3, 3, 1]


def test_build_batches_context_within_speaker() -> None:
    cues = [_mk(f"cue_{i+1:06d}", "A", "x") for i in range(5)]
    batches = _build_batches(cues, max_tokens=10000, max_cues=2, context_prev=1, context_next=1)
    # 第二批 targets=[2,3]，context_prev=[1]、context_next=[4]
    assert batches[1].target_idxs == [2, 3]
    assert batches[1].prev_idxs == [1]
    assert batches[1].next_idxs == [4]


# ── pipeline integration tests ──────────────────────────────────────────────


def test_run_proofread_writes_artifact_and_manifest(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    # 3 个 cue 跨 2 个 speaker → 2 个 batch
    fake = FakeLLMClient([
        _mock_response_for(["cue_000001", "cue_000002"]),
        _mock_response_for(["cue_000003"]),
    ])

    req = ProofreadRequest(workdir=ws.root, max_cues_per_batch=40)
    summary = run_proofread(req, llm_client=fake)

    # ── artifact ────────────────────────────────────────────
    assert ws.proofread_json_path.exists()
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert artifact["schemaVersion"] == "1"
    assert artifact["state"] == "draft"
    assert artifact["sourceId"] == "fake_src"
    assert artifact["inputArtifact"] == "subtitles.cues.json"
    assert artifact["inputHash"].startswith("sha256:")
    assert len(artifact["cues"]) == 3
    assert [c["cueId"] for c in artifact["cues"]] == [
        "cue_000001", "cue_000002", "cue_000003",
    ]
    # corrected != source → changed
    assert artifact["cues"][0]["correctedText"].startswith("corrected:")
    assert artifact["cues"][0]["editLevel"] in ("minor", "major")
    assert artifact["metrics"]["cueCount"] == 3
    assert artifact["metrics"]["changedCueRate"] == 1.0

    # ── manifest ────────────────────────────────────────────
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["proofread"]["state"] == "draft"
    assert manifest["proofread"]["batchCount"] == 2
    assert manifest["proofread"]["cachedBatchCount"] == 0
    assert manifest["proofread"]["inputHash"] == artifact["inputHash"]
    assert manifest["artifacts"]["subtitle_proofread_json"] == str(ws.proofread_json_path)

    # ── events ──────────────────────────────────────────────
    events = [
        json.loads(line) for line in ws.events_path.read_text(encoding="utf-8").splitlines() if line
    ]
    event_names = [e["event"] for e in events]
    assert "proofread.start" in event_names
    assert event_names.count("proofread.batch.start") == 2
    assert event_names.count("proofread.batch.done") == 2
    assert "proofread.done" in event_names

    # ── checkpoint files ────────────────────────────────────
    assert (ws.proofread_work_dir / "batch_000.json").exists()
    assert (ws.proofread_work_dir / "batch_001.json").exists()


def test_run_proofread_resume_skips_cached_batches(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    fake1 = FakeLLMClient([
        _mock_response_for(["cue_000001", "cue_000002"]),
        _mock_response_for(["cue_000003"]),
    ])
    req = ProofreadRequest(workdir=ws.root)
    run_proofread(req, llm_client=fake1)

    # 第一次：2 个 batch 都调了 LLM
    assert len(fake1.calls) == 2

    # 删掉 artifact 但保留 work/proofread/，再跑一次
    ws.proofread_json_path.unlink()
    fake2 = FakeLLMClient([])  # 不应被调用
    run_proofread(req, llm_client=fake2)

    assert len(fake2.calls) == 0, "checkpoint 命中应跳过所有 LLM 调用"
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["proofread"]["cachedBatchCount"] == 2


def test_run_proofread_force_wipes_checkpoints(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    fake1 = FakeLLMClient([
        _mock_response_for(["cue_000001", "cue_000002"]),
        _mock_response_for(["cue_000003"]),
    ])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake1)

    # force=True 应该全部重跑
    fake2 = FakeLLMClient([
        _mock_response_for(["cue_000001", "cue_000002"], suffix=":v2"),
        _mock_response_for(["cue_000003"], suffix=":v2"),
    ])
    run_proofread(ProofreadRequest(workdir=ws.root, force=True), llm_client=fake2)

    assert len(fake2.calls) == 2
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert "v2" in artifact["cues"][0]["correctedText"]


def test_run_proofread_existing_artifact_refuses_without_force(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    fake = FakeLLMClient([
        _mock_response_for(["cue_000001", "cue_000002"]),
        _mock_response_for(["cue_000003"]),
    ])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake)

    fake2 = FakeLLMClient([])
    with pytest.raises(FileExistsError):
        run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake2)


def test_run_proofread_missing_cues_raises(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    with pytest.raises(FileNotFoundError):
        run_proofread(ProofreadRequest(workdir=ws.root), llm_client=FakeLLMClient([]))


def test_run_proofread_invalid_json_response_fallback_marks_review(tmp_path: Path) -> None:
    """LLM 返回坏 JSON → repair 也坏 → 整批 fallback 标 needsHumanReview。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
        ],
    )
    fake = FakeLLMClient([
        "not json at all",   # 第一次失败
        "still not json",    # repair 也失败
    ])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake)

    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["needsHumanReview"] is True
    assert cue["risk"] == "blocking"
    assert cue["correctedText"] == cue["sourceText"]
    assert any("schema_fail" in n for n in cue["notes"])


def test_run_proofread_glossary_protects_terms(tmp_path: Path) -> None:
    """glossary protected term 在 LLM 改写后会触发 high risk + needsHumanReview。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "use Claude here"},
        ],
    )
    glossary = tmp_path / "g.json"
    glossary.write_text(
        json.dumps({
            "version": 1,
            "terms": [{"source": "Claude", "protected": True}],
        }),
        encoding="utf-8",
    )

    # LLM 改写了 protected 词
    fake = FakeLLMClient([
        json.dumps({
            "cues": [{"cueId": "cue_000001", "correctedText": "use ChatGPT here", "needsHumanReview": False}]
        }),
    ])
    run_proofread(
        ProofreadRequest(workdir=ws.root, glossary_path=glossary),
        llm_client=fake,
    )

    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["risk"] == "high"
    assert cue["needsHumanReview"] is True
    assert any("protected_term_change:Claude" in n for n in cue["notes"])
    assert artifact["params"]["glossaryHash"] is not None
