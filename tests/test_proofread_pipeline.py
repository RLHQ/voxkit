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
        model: str = "deepseek-v4-flash",
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


# ── 回归测试：Codex 审查暴露的不变量 ──────────────────────────────────────


def test_run_proofread_force_refuses_reviewed_without_explicit_flag(tmp_path: Path) -> None:
    """Codex P1: --force 默认只覆盖 draft；reviewed 状态需要 --force-reviewed。"""
    from voxkit.core.lifecycle import transition_state

    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
        ],
    )
    fake1 = FakeLLMClient([_mock_response_for(["cue_000001"])])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake1)

    # 推到 reviewed
    transition_state(ws.proofread_json_path, to="reviewed", reviewer="alice")
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert artifact["state"] == "reviewed"

    # 默认 --force（force_level="draft"）应被拒绝
    with pytest.raises(FileExistsError) as exc:
        run_proofread(
            ProofreadRequest(workdir=ws.root, force=True),
            llm_client=FakeLLMClient([_mock_response_for(["cue_000001"], suffix=":v2")]),
        )
    assert "reviewed" in str(exc.value)
    # 旧 artifact 仍存在且未被改
    after = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert after["state"] == "reviewed"
    assert after["cues"][0]["correctedText"] == artifact["cues"][0]["correctedText"]

    # --force-reviewed 应通过
    fake2 = FakeLLMClient([_mock_response_for(["cue_000001"], suffix=":v2")])
    run_proofread(
        ProofreadRequest(workdir=ws.root, force_level="reviewed"),
        llm_client=fake2,
    )
    after = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert "v2" in after["cues"][0]["correctedText"]
    assert after["state"] == "draft"


def test_run_proofread_checkpoint_invalidates_on_policy_change(tmp_path: Path) -> None:
    """Codex P1: edit_level 变化必须让 checkpoint 失效（policyHash 入键）。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
        ],
    )
    fake1 = FakeLLMClient([_mock_response_for(["cue_000001"], suffix=":v1")])
    run_proofread(
        ProofreadRequest(workdir=ws.root, edit_level="standard"),
        llm_client=fake1,
    )
    assert len(fake1.calls) == 1

    # 删 artifact 保留 checkpoint，改 edit_level → 应该重新调 LLM
    ws.proofread_json_path.unlink()
    fake2 = FakeLLMClient([_mock_response_for(["cue_000001"], suffix=":v2")])
    run_proofread(
        ProofreadRequest(workdir=ws.root, edit_level="strict"),
        llm_client=fake2,
    )
    assert len(fake2.calls) == 1, "policy 变了 checkpoint 必须失效"
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert ":v2" in artifact["cues"][0]["correctedText"]


def test_run_proofread_checkpoint_invalidates_on_source_time_change(tmp_path: Path) -> None:
    """Codex P1: source cue 时间/speaker 改变必须让 checkpoint 失效（contentHash 入键）。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "hello"},
        ],
    )
    fake1 = FakeLLMClient([_mock_response_for(["cue_000001"])])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake1)

    # 删 artifact，把 cue 的 start/end/speaker 都改了，text 不变
    ws.proofread_json_path.unlink()
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 5.0, "end": 7.0, "speaker": "B", "text": "hello"},
        ],
    )
    fake2 = FakeLLMClient([_mock_response_for(["cue_000001"])])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake2)
    assert len(fake2.calls) == 1, "source time/speaker 变了 cache 必须失效"
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert artifact["cues"][0]["sourceStart"] == 5.0
    assert artifact["cues"][0]["speaker"] == "B"


def test_run_proofread_empty_corrected_text_marks_blocking(tmp_path: Path) -> None:
    """Codex P2: LLM 返回空 correctedText → repair → 仍空 → fallback blocking。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
        ],
    )
    empty_response = json.dumps({
        "cues": [{"cueId": "cue_000001", "correctedText": "", "needsHumanReview": False}]
    }, ensure_ascii=False)
    fake = FakeLLMClient([empty_response, empty_response])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake)

    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    cue = artifact["cues"][0]
    assert cue["risk"] == "blocking"
    assert cue["needsHumanReview"] is True
    assert cue["correctedText"] == cue["sourceText"]
    assert "schema_fail" in cue["notes"]


def test_run_proofread_transport_failure_writes_pending_marker(tmp_path: Path) -> None:
    """Codex P2: LLM transport 错误 → pending marker，不写稳定 artifact，rerun 续做。"""
    from voxkit.llm.errors import LLMError, LLMRateLimit

    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(
        ws.cues_json_path,
        cues=[
            {"id": "cue_000001", "start": 0.0, "end": 2.0, "speaker": "A", "text": "x"},
            {"id": "cue_000002", "start": 2.0, "end": 4.0, "speaker": "B", "text": "y"},
        ],
    )

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
                    text=_mock_response_for(["cue_000001"]),
                    prompt_tokens=10, completion_tokens=5,
                    model=self._model, raw={},
                )
            raise LLMRateLimit("rate limited", retry_after_secs=1.0)

        def close(self) -> None:
            pass

    fake = _RateLimitOnSecondBatch()
    with pytest.raises(LLMError, match="incomplete"):
        run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake)

    assert (ws.proofread_work_dir / "batch_000.json").exists()
    pending = ws.proofread_work_dir / "batch_001.pending.json"
    assert pending.exists()
    pdata = json.loads(pending.read_text(encoding="utf-8"))
    assert pdata["errorKind"] == "LLMRateLimit"
    # 关键：稳定 artifact 不应被写出（旧 artifact 也未被预先 unlink）
    assert not ws.proofread_json_path.exists()

    # rerun 无 force：第一批命中 cache，第二批走 LLM
    fake2 = FakeLLMClient([_mock_response_for(["cue_000002"])])
    run_proofread(ProofreadRequest(workdir=ws.root), llm_client=fake2)
    assert len(fake2.calls) == 1
    assert not pending.exists(), "成功后清理 pending marker"
    artifact = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
    assert len(artifact["cues"]) == 2


# ── dry-run (F4) ────────────────────────────────────────────────────────────


def test_run_proofread_dry_run_skips_llm_and_writes_nothing(tmp_path: Path) -> None:
    """``dry_run=True`` 不调 LLM、不写 artifact / work dir。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    fake = FakeLLMClient([])  # 任何调用都会触发 AssertionError
    req = ProofreadRequest(workdir=ws.root, dry_run=True)
    summary = run_proofread(req, llm_client=fake)

    assert summary["dryRun"] is True
    assert summary["cueCount"] == 3
    assert summary["batchCount"] >= 1
    assert summary["estPromptTokens"] > 0
    assert summary["estCompletionTokens"] > 0
    # provider 默认 deepseek + 未指定 model → 走 unknown rate 分支
    assert summary["provider"] == "deepseek"
    assert summary["estCostUsd"] is None
    assert len(fake.calls) == 0, "dry-run 严禁调 LLM"
    # 不写 artifact、不写 work/proofread/、不写 manifest
    assert not ws.proofread_json_path.exists()
    assert not ws.manifest_path.exists()
    assert not ws.proofread_work_dir.exists()


def test_run_proofread_dry_run_with_known_model_returns_cost(tmp_path: Path) -> None:
    """显式指定已注册 model → estCostUsd 不为 None。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)

    req = ProofreadRequest(
        workdir=ws.root, dry_run=True, model="deepseek-v4-flash"
    )
    summary = run_proofread(req, llm_client=FakeLLMClient([]))

    assert summary["estCostUsd"] is not None
    assert summary["estCostUsd"] > 0
    assert summary["model"] == "deepseek-v4-flash"


def test_run_proofread_dry_run_missing_input_raises(tmp_path: Path) -> None:
    """dry-run 也该对缺 cues.json 报 FileNotFoundError。"""
    ws = open_workspace(tmp_path / "ws")
    req = ProofreadRequest(workdir=ws.root, dry_run=True)
    with pytest.raises(FileNotFoundError):
        run_proofread(req, llm_client=FakeLLMClient([]))


def test_run_proofread_dry_run_not_blocked_by_stale_lock(tmp_path: Path) -> None:
    """dry-run 不该被 stale .lock 文件挡住（只读操作）。"""
    ws = open_workspace(tmp_path / "ws")
    _write_cues_v2(ws.cues_json_path)
    # 模拟"上次跑挂留下的 .lock"
    ws.root.mkdir(parents=True, exist_ok=True)
    lock_file = ws.root / ".lock"
    lock_file.write_text("stale", encoding="utf-8")

    req = ProofreadRequest(workdir=ws.root, dry_run=True)
    summary = run_proofread(req, llm_client=FakeLLMClient([]))
    assert summary["dryRun"] is True
    # lock 文件没被动
    assert lock_file.read_text(encoding="utf-8") == "stale"
