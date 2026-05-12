"""``voxkit proofread`` 主流水线。

输入：``<workdir>/subtitles.cues.json`` (schemaVersion=2)
输出：``<workdir>/subtitles.proofread.json`` (state="draft")

阶段拆解：

  1. 读 cues + 算 ``inputHash``（上游字节 sha256）
  2. 可选加载 glossary，算 ``glossaryHash``
  3. 加载 prompt 模板 + 算 ``promptHash``
  4. 按 token / cue / speaker 切 batch
  5. 逐 batch：检查 ``work/proofread/batch_NNN.json`` 命中则跳过；否则调 LLM →
     Pydantic 校验 → 单次 repair → fallback 标人工 → 落 checkpoint
  6. 合并所有 batch 结果 → 风险评级 → 写最终 artifact (exclusive write)
  7. manifest 镜像 state/cost/metrics

并发：单 workdir 同时只允许一个 proofread，靠 workspace ``.lock`` 保证。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ValidationError

from voxkit.core.proofread_risk import (
    estimate_tokens,
    grade_risk,
    infer_edit_level,
)
from voxkit.core.workspace import (
    EventMirror,
    Workspace,
    acquire_lock,
    open_workspace,
    read_manifest,
    release_lock,
    write_manifest,
)
from voxkit.io.glossary import Glossary, glossary_hash, load_glossary, protected_terms
from voxkit.io.schema import (
    ProofreadCueOut,
    ProofreadMetrics,
    ProofreadOutput,
    ProofreadParams,
    SubtitleCueOut,
    SubtitleCuesOutput,
)
from voxkit.llm import ChatResult, LLMClient
from voxkit.llm.errors import LLMError, LLMRefusal, LLMSchemaError
from voxkit.llm.prompts import load_prompt

__all__ = [
    "ProofreadRequest",
    "run_proofread",
]


# ── public request ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProofreadRequest:
    """CLI / 程序化调用都用这一个不可变 dataclass。"""

    workdir: Path
    provider: str = "deepseek"
    model: Optional[str] = None  # None → 用 provider.default_model
    language: Optional[str] = None  # None → 沿用 cues.params 中的 language（缺则 "auto"）
    edit_level: str = "standard"
    glossary_path: Optional[Path] = None
    max_input_tokens: int = 6000
    max_cues_per_batch: int = 40
    context_prev: int = 8
    context_next: int = 4
    force: bool = False
    json_events: bool = False
    timeout_s: float = 60.0


# ── internal types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BatchSpec:
    """单个 LLM 调用的输入约束。indices 都是相对 `all_cues` 的位置。"""

    index: int
    target_idxs: List[int]
    prev_idxs: List[int]
    next_idxs: List[int]


@dataclass
class _BatchResult:
    out_cues: List[ProofreadCueOut]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class _CorrectedCue(BaseModel):
    """LLM 返回的单条 cue 的最小契约。"""

    cueId: str
    correctedText: str
    needsHumanReview: bool = False


class _CorrectedBatch(BaseModel):
    cues: List[_CorrectedCue]


# ── batching ────────────────────────────────────────────────────────────────


def _build_batches(
    cues: Sequence[SubtitleCueOut],
    *,
    max_tokens: int,
    max_cues: int,
    context_prev: int,
    context_next: int,
) -> List[_BatchSpec]:
    """按 token / cue / speaker 边界把 cues 切成 batch。

    规则：
      - target 段是连续区间；不跨 speaker
      - token 预算只算 targets（context 用 cue 数量界）
      - 单 cue 即使超 token 预算也独占一个 batch（边界保底）
    """
    n = len(cues)
    if n == 0:
        return []

    batches: List[_BatchSpec] = []
    i = 0
    batch_idx = 0
    while i < n:
        speaker = cues[i].speaker
        target_idxs = [i]
        tokens = estimate_tokens(cues[i].text)
        j = i + 1
        while j < n:
            if cues[j].speaker != speaker:
                break
            cue_tok = estimate_tokens(cues[j].text)
            if tokens + cue_tok > max_tokens:
                break
            if len(target_idxs) >= max_cues:
                break
            target_idxs.append(j)
            tokens += cue_tok
            j += 1

        prev_idxs = list(range(max(0, i - context_prev), i))
        next_idxs = list(range(j, min(n, j + context_next)))
        batches.append(
            _BatchSpec(
                index=batch_idx,
                target_idxs=target_idxs,
                prev_idxs=prev_idxs,
                next_idxs=next_idxs,
            )
        )
        batch_idx += 1
        i = j

    return batches


# ── prompt rendering ────────────────────────────────────────────────────────


def _render_prompt(template: str, *, edit_level: str, protected: Sequence[str]) -> str:
    """把 ``{edit_level}`` / ``{protected_terms}`` 占位符替换成最终系统消息。

    用 ``str.replace`` 而非 ``str.format`` 是为了避开模板里的花括号（JSON 示例）。
    """
    protected_text = "（无）" if not protected else "\n".join(f"- {t}" for t in protected)
    return (
        template
        .replace("{edit_level}", edit_level)
        .replace("{protected_terms}", protected_text)
    )


# ── LLM call w/ repair ──────────────────────────────────────────────────────


def _call_llm_with_repair(
    client: LLMClient,
    *,
    system: str,
    user: str,
    expected_ids: List[str],
) -> tuple[_CorrectedBatch, ChatResult]:
    """单 batch：调 LLM → Pydantic 校验 → 一次 repair → 仍失败抛 LLMSchemaError。"""

    def _try_parse(raw: str) -> _CorrectedBatch:
        # response_format=json_object 通常返回纯 JSON，但有些 provider 偶发会包裹
        # ```json ... ```；保留 raw 同时容错 markdown fence。
        body = raw.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:]
            body = body.strip()
        parsed = _CorrectedBatch.model_validate_json(body)
        got_ids = [c.cueId for c in parsed.cues]
        if got_ids != expected_ids:
            raise ValueError(
                f"cueId order/coverage mismatch: expected {expected_ids[:3]}…, "
                f"got {got_ids[:3]}…"
            )
        return parsed

    # 第一次
    result = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    try:
        return _try_parse(result.text), result
    except (ValidationError, ValueError, json.JSONDecodeError) as e1:
        first_err = str(e1)

    # repair 单次：把原输出 + 错误信息回灌给模型
    repair_user = (
        "Your previous JSON response failed validation. Error:\n"
        f"{first_err}\n\n"
        "Previous response:\n"
        f"{result.text}\n\n"
        "Re-emit ONLY the corrected JSON object, no prose. Keep the cueId order and "
        "include exactly these ids in this order: "
        f"{json.dumps(expected_ids, ensure_ascii=False)}."
    )
    repair_result = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": repair_user},
        ]
    )
    try:
        parsed = _try_parse(repair_result.text)
    except (ValidationError, ValueError, json.JSONDecodeError) as e2:
        raise LLMSchemaError(
            f"proofread batch failed after repair: {e2}"
        ) from e2

    # token 计入两次调用的总和
    repair_result = ChatResult(
        text=repair_result.text,
        prompt_tokens=result.prompt_tokens + repair_result.prompt_tokens,
        completion_tokens=result.completion_tokens + repair_result.completion_tokens,
        model=repair_result.model,
        raw=repair_result.raw,
    )
    return parsed, repair_result


# ── batch composition ───────────────────────────────────────────────────────


def _compose_batch(
    batch: _BatchSpec,
    all_cues: Sequence[SubtitleCueOut],
    parsed: _CorrectedBatch,
    *,
    protected: frozenset[str],
) -> List[ProofreadCueOut]:
    """把 LLM 返回的 ``correctedText`` 套回源 cue + 跑本地风险评级。"""
    out: List[ProofreadCueOut] = []
    parsed_by_id = {c.cueId: c for c in parsed.cues}
    for idx in batch.target_idxs:
        src = all_cues[idx]
        cor = parsed_by_id[src.id]
        corrected_text = cor.correctedText
        risk, notes = grade_risk(src.text, corrected_text, protected_terms=protected)
        edit_level = infer_edit_level(src.text, corrected_text)
        needs_review = cor.needsHumanReview or risk in ("high", "blocking")
        out.append(
            ProofreadCueOut(
                cueId=src.id,
                sourceStart=src.start,
                sourceEnd=src.end,
                speaker=src.speaker,
                sourceText=src.text,
                correctedText=corrected_text,
                editLevel=edit_level,
                risk=risk,
                needsHumanReview=needs_review,
                notes=notes,
            )
        )
    return out


def _fallback_batch(
    batch: _BatchSpec,
    all_cues: Sequence[SubtitleCueOut],
    *,
    reason: str,
) -> List[ProofreadCueOut]:
    """LLM 失败（schema / refusal）时把 batch 整体标人工，corrected = source。"""
    out: List[ProofreadCueOut] = []
    for idx in batch.target_idxs:
        src = all_cues[idx]
        out.append(
            ProofreadCueOut(
                cueId=src.id,
                sourceStart=src.start,
                sourceEnd=src.end,
                speaker=src.speaker,
                sourceText=src.text,
                correctedText=src.text,
                editLevel="none",
                risk="blocking",
                needsHumanReview=True,
                notes=[reason],
            )
        )
    return out


# ── checkpoint ──────────────────────────────────────────────────────────────


def _content_hash(cues: Sequence[SubtitleCueOut]) -> str:
    """对一组 cue 的 id+text 计算稳定 hash，用作 checkpoint resume key。"""
    payload = json.dumps(
        [(c.id, c.text) for c in cues],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(ws: Workspace, batch_index: int) -> Path:
    return ws.proofread_work_dir / f"batch_{batch_index:03d}.json"


def _try_load_checkpoint(
    path: Path,
    *,
    expect_content_hash: str,
    expect_prompt_version: str,
    expect_model: str,
) -> Optional[_BatchResult]:
    """命中 checkpoint 时返回 cached ``_BatchResult``，不一致则忽略（不删旧文件）。"""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("contentHash") != expect_content_hash
        or data.get("promptVersion") != expect_prompt_version
        or data.get("model") != expect_model
    ):
        return None
    cues = [ProofreadCueOut.model_validate(c) for c in data.get("cues", [])]
    return _BatchResult(
        out_cues=cues,
        prompt_tokens=int(data.get("promptTokens", 0)),
        completion_tokens=int(data.get("completionTokens", 0)),
        cached=True,
    )


def _write_checkpoint(
    path: Path,
    *,
    batch_index: int,
    content_hash: str,
    prompt_version: str,
    model: str,
    result: _BatchResult,
) -> None:
    payload = {
        "batchIndex": batch_index,
        "contentHash": content_hash,
        "promptVersion": prompt_version,
        "model": model,
        "promptTokens": result.prompt_tokens,
        "completionTokens": result.completion_tokens,
        "cues": [c.model_dump(by_alias=True) for c in result.out_cues],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ── main entry ──────────────────────────────────────────────────────────────


def _emit(em: EventMirror, payload: Dict[str, Any], *, to_stderr: bool) -> None:
    em.emit(payload)
    if to_stderr:
        import sys
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stderr.flush()


def run_proofread(
    req: ProofreadRequest,
    *,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """主入口。返回 manifest 增量字典（同时已写盘）。"""

    ws = open_workspace(req.workdir)
    acquire_lock(ws)
    started = time.monotonic()
    try:
        # 1. 输入
        if not ws.cues_json_path.is_file():
            raise FileNotFoundError(
                f"missing input: {ws.cues_json_path}; run `voxkit transcribe "
                f"--resegment=semantic` first"
            )
        if ws.proofread_json_path.exists() and not req.force:
            raise FileExistsError(
                f"refusing to overwrite {ws.proofread_json_path}; pass --force "
                "to rebuild (also wipes work/proofread/)"
            )

        raw_bytes = ws.cues_json_path.read_bytes()
        input_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        cues_doc = SubtitleCuesOutput.model_validate(json.loads(raw_bytes))
        if cues_doc.schema_version != "2":
            raise ValueError(
                f"unsupported cues schemaVersion={cues_doc.schema_version}; "
                "rebuild with current voxkit transcribe"
            )

        # 2. force：清空 work/proofread/，再确保目录在
        if req.force and ws.proofread_work_dir.exists():
            shutil.rmtree(ws.proofread_work_dir)
        ws.proofread_work_dir.mkdir(parents=True, exist_ok=True)
        if req.force and ws.proofread_json_path.exists():
            ws.proofread_json_path.unlink()

        # 3. glossary
        gloss: Optional[Glossary] = None
        gloss_hash: Optional[str] = None
        protected_set: frozenset[str] = frozenset()
        if req.glossary_path is not None:
            gloss = load_glossary(req.glossary_path)
            gloss_hash = glossary_hash(gloss)
            protected_set = frozenset(protected_terms(gloss))

        # 4. prompt
        prompt_text, prompt_hash = load_prompt("proofread", "v1")
        prompt_version = "proofread.v1"
        system_msg = _render_prompt(
            prompt_text,
            edit_level=req.edit_level,
            protected=sorted(protected_set),
        )

        # 5. LLM client
        owns_client = llm_client is None
        client = llm_client or LLMClient(
            req.provider, model=req.model, timeout_s=req.timeout_s
        )
        used_model = client._model  # noqa: SLF001 — 内部字段，文档化访问点

        # 6. 切 batch
        batches = _build_batches(
            cues_doc.cues,
            max_tokens=req.max_input_tokens,
            max_cues=req.max_cues_per_batch,
            context_prev=req.context_prev,
            context_next=req.context_next,
        )

        language = req.language or (cues_doc.params or {}).get("language") or "auto"

        with EventMirror(ws) as emit:
            _emit(emit, {
                "event": "proofread.start",
                "workdir": str(ws.root),
                "provider": req.provider,
                "model": used_model,
                "promptVersion": prompt_version,
                "editLevel": req.edit_level,
                "cueCount": len(cues_doc.cues),
                "batchCount": len(batches),
                "glossaryHash": gloss_hash,
            }, to_stderr=req.json_events)

            # 7. 处理 batch
            all_out: List[ProofreadCueOut] = []
            prompt_tok_total = 0
            completion_tok_total = 0
            cached_count = 0

            try:
                for batch in batches:
                    target_cues = [cues_doc.cues[i] for i in batch.target_idxs]
                    chash = _content_hash(target_cues)
                    cp_path = _checkpoint_path(ws, batch.index)

                    _emit(emit, {
                        "event": "proofread.batch.start",
                        "batchIndex": batch.index,
                        "targetCount": len(target_cues),
                    }, to_stderr=req.json_events)

                    cached = _try_load_checkpoint(
                        cp_path,
                        expect_content_hash=chash,
                        expect_prompt_version=prompt_version,
                        expect_model=used_model,
                    )
                    if cached is not None:
                        all_out.extend(cached.out_cues)
                        prompt_tok_total += cached.prompt_tokens
                        completion_tok_total += cached.completion_tokens
                        cached_count += 1
                        _emit(emit, {
                            "event": "proofread.batch.done",
                            "batchIndex": batch.index,
                            "cached": True,
                        }, to_stderr=req.json_events)
                        continue

                    # 构造用户消息
                    user_obj = {
                        "language": language,
                        "context_prev": [
                            {"cueId": cues_doc.cues[i].id, "speaker": cues_doc.cues[i].speaker, "text": cues_doc.cues[i].text}
                            for i in batch.prev_idxs
                        ],
                        "targets": [
                            {"cueId": c.id, "speaker": c.speaker, "text": c.text}
                            for c in target_cues
                        ],
                        "context_next": [
                            {"cueId": cues_doc.cues[i].id, "speaker": cues_doc.cues[i].speaker, "text": cues_doc.cues[i].text}
                            for i in batch.next_idxs
                        ],
                    }
                    user_msg = json.dumps(user_obj, ensure_ascii=False, indent=2)
                    expected_ids = [c.id for c in target_cues]

                    try:
                        parsed, llm_result = _call_llm_with_repair(
                            client,
                            system=system_msg,
                            user=user_msg,
                            expected_ids=expected_ids,
                        )
                        out_cues = _compose_batch(
                            batch, cues_doc.cues, parsed, protected=protected_set
                        )
                        result_pt = llm_result.prompt_tokens
                        result_ct = llm_result.completion_tokens
                    except (LLMSchemaError, LLMRefusal) as e:
                        # fallback：整批标人工，不抛
                        reason = "provider_refusal" if isinstance(e, LLMRefusal) else "schema_fail"
                        out_cues = _fallback_batch(batch, cues_doc.cues, reason=reason)
                        result_pt = 0
                        result_ct = 0

                    br = _BatchResult(
                        out_cues=out_cues,
                        prompt_tokens=result_pt,
                        completion_tokens=result_ct,
                    )
                    _write_checkpoint(
                        cp_path,
                        batch_index=batch.index,
                        content_hash=chash,
                        prompt_version=prompt_version,
                        model=used_model,
                        result=br,
                    )
                    all_out.extend(br.out_cues)
                    prompt_tok_total += br.prompt_tokens
                    completion_tok_total += br.completion_tokens

                    _emit(emit, {
                        "event": "proofread.batch.done",
                        "batchIndex": batch.index,
                        "cached": False,
                        "promptTokens": br.prompt_tokens,
                        "completionTokens": br.completion_tokens,
                    }, to_stderr=req.json_events)
            finally:
                if owns_client:
                    client.close()

            # 8. 聚合 metrics + 写 artifact
            cue_count = len(all_out)
            changed = sum(1 for c in all_out if c.corrected_text != c.source_text)
            need_review = sum(1 for c in all_out if c.needs_human_review)

            metrics = ProofreadMetrics(
                cueCount=cue_count,
                changedCueRate=(changed / cue_count) if cue_count else 0.0,
                reviewCueRate=(need_review / cue_count) if cue_count else 0.0,
                promptTokensTotal=prompt_tok_total,
                completionTokensTotal=completion_tok_total,
            )

            params = ProofreadParams(
                editLevel=req.edit_level,
                allowRetiming=False,
                glossaryHash=gloss_hash,
            )

            artifact = ProofreadOutput(
                state="draft",
                sourceId=cues_doc.source_id,
                inputArtifact="subtitles.cues.json",
                inputHash=input_hash,
                language=language,
                provider=req.provider,
                model=used_model,
                promptVersion=prompt_version,
                promptHash=prompt_hash,
                params=params,
                cues=all_out,
                metrics=metrics,
            )

            payload = artifact.model_dump(by_alias=True, exclude_none=False)
            tmp = ws.proofread_json_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # exclusive-create 语义靠前置检查（force/已存在删除）保证；os.replace 是原子的
            tmp.replace(ws.proofread_json_path)

            elapsed = time.monotonic() - started

            # 9. manifest 镜像（追加 `proofread` 段 + artifacts 入口）
            existing = read_manifest(ws) or {}
            existing.setdefault("artifacts", {})
            existing["artifacts"]["subtitle_proofread_json"] = str(ws.proofread_json_path)
            existing["proofread"] = {
                "state": "draft",
                "provider": req.provider,
                "model": used_model,
                "promptVersion": prompt_version,
                "promptHash": prompt_hash,
                "editLevel": req.edit_level,
                "glossaryHash": gloss_hash,
                "inputArtifact": "subtitles.cues.json",
                "inputHash": input_hash,
                "batchCount": len(batches),
                "cachedBatchCount": cached_count,
                "batchSize": {
                    "tokensMax": req.max_input_tokens,
                    "cuesMax": req.max_cues_per_batch,
                },
                "changedCueRate": metrics.changed_cue_rate,
                "reviewCueRate": metrics.review_cue_rate,
                "promptTokens": prompt_tok_total,
                "completionTokens": completion_tok_total,
                "elapsedSecs": elapsed,
            }
            write_manifest(ws, existing)

            _emit(emit, {
                "event": "proofread.done",
                "cueCount": cue_count,
                "changedCueRate": metrics.changed_cue_rate,
                "reviewCueRate": metrics.review_cue_rate,
                "promptTokens": prompt_tok_total,
                "completionTokens": completion_tok_total,
                "cachedBatchCount": cached_count,
                "elapsedSecs": elapsed,
            }, to_stderr=req.json_events)

            return existing["proofread"]

    finally:
        release_lock(ws)
