"""``voxkit translate`` 主流水线。

输入选择优先级：
  1. ``<workdir>/subtitles.proofread.json`` (任何 state)
  2. 缺失则回落 ``<workdir>/subtitles.cues.json`` (schemaVersion=2)

输出：
  - ``<workdir>/subtitles.<lang>.json``（state="draft"）
  - 可选 ``subtitles.<lang>.srt`` / ``subtitles.<lang>.vtt``

v1 限制（按规划 §Step 3）：cueMappingPolicy 仅 ``one-to-one``，时间范围直接
继承源 cue。group-within-speaker rewrap 在 v1 不实现。

阶段拆解与 proofread 同形：load → batch → call LLM (repair once) → fallback
mark review → checkpoint → assemble → write + manifest。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ValidationError

from voxkit.core.proofread_risk import estimate_tokens, is_cjk_char
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
    ArtifactState,
    ProofreadOutput,
    SubtitleCuesOutput,
    TranslationCueOut,
    TranslationMetrics,
    TranslationOutput,
    TranslationParams,
)
from voxkit.io.srt import format_srt_time, format_vtt_time
from voxkit.llm import ChatResult, LLMClient
from voxkit.llm.errors import LLMError, LLMRefusal, LLMSchemaError
from voxkit.llm.prompts import load_prompt

__all__ = [
    "TranslateRequest",
    "run_translate",
]


# ── public request ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranslateRequest:
    workdir: Path
    target_language: str
    source_language: Optional[str] = None  # None → 从输入 artifact 推断
    style: str = "subtitle"
    length_policy: str = "preserve"
    provider: str = "deepseek"
    model: Optional[str] = None
    glossary_path: Optional[Path] = None
    max_input_tokens: int = 6000
    max_cues_per_batch: int = 40
    context_prev: int = 4
    context_next: int = 2
    emit_srt: bool = True
    emit_vtt: bool = True
    force: bool = False
    json_events: bool = False
    timeout_s: float = 60.0


# ── normalized source cue (input-agnostic) ──────────────────────────────────


@dataclass(frozen=True)
class _SrcCue:
    """从 proofread 或 cues 标准化出来的源 cue。"""

    id: str
    start: float
    end: float
    speaker: Optional[str]
    text: str


@dataclass(frozen=True)
class _BatchSpec:
    index: int
    target_idxs: List[int]
    prev_idxs: List[int]
    next_idxs: List[int]


@dataclass
class _BatchResult:
    out_cues: List[TranslationCueOut]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class _TranslatedCue(BaseModel):
    cueId: str
    translatedText: str
    needsHumanReview: bool = False


class _TranslatedBatch(BaseModel):
    cues: List[_TranslatedCue]


# ── input loading ──────────────────────────────────────────────────────────


def _load_source(ws: Workspace) -> Tuple[List[_SrcCue], str, str, Dict[str, Any]]:
    """优先读 proofread，回落 cues。返回 (cues, source_lang, input_artifact, raw_dict)。"""
    if ws.proofread_json_path.is_file():
        raw_bytes = ws.proofread_json_path.read_bytes()
        doc = ProofreadOutput.model_validate(json.loads(raw_bytes))
        src_cues = [
            _SrcCue(
                id=c.cue_id,
                start=c.source_start,
                end=c.source_end,
                speaker=c.speaker,
                text=c.corrected_text,
            )
            for c in doc.cues
        ]
        return src_cues, doc.language, "subtitles.proofread.json", {"bytes": raw_bytes}

    if ws.cues_json_path.is_file():
        raw_bytes = ws.cues_json_path.read_bytes()
        doc2 = SubtitleCuesOutput.model_validate(json.loads(raw_bytes))
        if doc2.schema_version != "2":
            raise ValueError(
                f"unsupported cues schemaVersion={doc2.schema_version}; "
                "rebuild with current voxkit transcribe"
            )
        lang = (doc2.params or {}).get("language") or "auto"
        src_cues = [
            _SrcCue(id=c.id, start=c.start, end=c.end, speaker=c.speaker, text=c.text)
            for c in doc2.cues
        ]
        return src_cues, lang, "subtitles.cues.json", {"bytes": raw_bytes}

    raise FileNotFoundError(
        f"no translation input in {ws.root}: need subtitles.proofread.json "
        "or subtitles.cues.json"
    )


# ── batching (mirror of proofread_pipeline._build_batches) ──────────────────


def _build_batches(
    cues: Sequence[_SrcCue],
    *,
    max_tokens: int,
    max_cues: int,
    context_prev: int,
    context_next: int,
) -> List[_BatchSpec]:
    n = len(cues)
    if n == 0:
        return []
    batches: List[_BatchSpec] = []
    i = 0
    bi = 0
    while i < n:
        speaker = cues[i].speaker
        target_idxs = [i]
        tokens = estimate_tokens(cues[i].text)
        j = i + 1
        while j < n:
            if cues[j].speaker != speaker:
                break
            t = estimate_tokens(cues[j].text)
            if tokens + t > max_tokens:
                break
            if len(target_idxs) >= max_cues:
                break
            target_idxs.append(j)
            tokens += t
            j += 1
        prev_idxs = list(range(max(0, i - context_prev), i))
        next_idxs = list(range(j, min(n, j + context_next)))
        batches.append(_BatchSpec(index=bi, target_idxs=target_idxs, prev_idxs=prev_idxs, next_idxs=next_idxs))
        bi += 1
        i = j
    return batches


# ── prompt rendering ────────────────────────────────────────────────────────


def _render_prompt(
    template: str,
    *,
    source_language: str,
    target_language: str,
    style: str,
    length_policy: str,
    glossary: Optional[Glossary],
) -> str:
    protected = sorted(protected_terms(glossary)) if glossary else []
    protected_text = "（无）" if not protected else "\n".join(f"- {t}" for t in protected)

    if glossary:
        mappings = [t for t in glossary.terms if t.target]
        if mappings:
            mapping_text = "\n".join(f"- `{t.source}` → `{t.target}`" for t in mappings)
        else:
            mapping_text = "（无）"
    else:
        mapping_text = "（无）"

    return (
        template
        .replace("{source_language}", source_language)
        .replace("{target_language}", target_language)
        .replace("{style}", style)
        .replace("{length_policy}", length_policy)
        .replace("{protected_terms}", protected_text)
        .replace("{glossary_mappings}", mapping_text)
    )


# ── LLM call w/ repair ──────────────────────────────────────────────────────


def _call_llm_with_repair(
    client: LLMClient,
    *,
    system: str,
    user: str,
    expected_ids: List[str],
) -> Tuple[_TranslatedBatch, ChatResult]:
    def _parse(raw: str) -> _TranslatedBatch:
        body = raw.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:]
            body = body.strip()
        parsed = _TranslatedBatch.model_validate_json(body)
        got_ids = [c.cueId for c in parsed.cues]
        if got_ids != expected_ids:
            raise ValueError(
                f"cueId mismatch: expected {expected_ids[:3]}…, got {got_ids[:3]}…"
            )
        return parsed

    result = client.chat(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    try:
        return _parse(result.text), result
    except (ValidationError, ValueError, json.JSONDecodeError) as e1:
        first_err = str(e1)

    repair_user = (
        "Your previous JSON response failed validation. Error:\n"
        f"{first_err}\n\nPrevious response:\n{result.text}\n\n"
        "Re-emit ONLY the corrected JSON object, no prose. Keep cueId order: "
        f"{json.dumps(expected_ids, ensure_ascii=False)}."
    )
    repair = client.chat(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": result.text},
        {"role": "user", "content": repair_user},
    ])
    try:
        parsed = _parse(repair.text)
    except (ValidationError, ValueError, json.JSONDecodeError) as e2:
        raise LLMSchemaError(f"translate batch failed after repair: {e2}") from e2

    merged = ChatResult(
        text=repair.text,
        prompt_tokens=result.prompt_tokens + repair.prompt_tokens,
        completion_tokens=result.completion_tokens + repair.completion_tokens,
        model=repair.model,
        raw=repair.raw,
    )
    return parsed, merged


# ── translation risk grading ───────────────────────────────────────────────


def _grade_translation_risk(
    src_text: str,
    trg_text: str,
    *,
    glossary: Optional[Glossary],
) -> Tuple[str, List[str], Dict[str, bool]]:
    """返回 (risk, notes, flags)。flags 用于聚合 metrics：

      - flags["over_char"]：目标长度 > 字符上限（按目标语言粗略门槛 42 字符）
      - flags["over_cps"]：目标长度 / duration 估算超 17 chars/sec（先不读
        duration，留 placeholder False，避免假报警）—— v1 不读时间
      - flags["glossary_miss"]：glossary 指定了 target 但翻译里未出现
    """
    notes: List[str] = []
    risk = "low"
    flags = {"over_char": False, "over_cps": False, "glossary_miss": False}

    if not trg_text.strip():
        notes.append("empty_translation")
        risk = "high"

    # 长度膨胀：目标长度 >1.5 倍源长度（粗略门槛，CJK ↔ Latin 不直接可比，但
    # 翻译同语对时仍有意义）
    src_eff = max(1, len(src_text))
    if len(trg_text) > src_eff * 1.5:
        notes.append("length_expansion")
        risk = max(risk, "medium", key={"low": 0, "medium": 1, "high": 2, "blocking": 3}.get)

    # 字符上限：目标语言为 CJK 时 25 字符是常见门槛；Latin 42 字符
    is_target_cjk = any(is_cjk_char(c) for c in trg_text)
    char_limit = 25 if is_target_cjk else 42
    if len(trg_text) > char_limit:
        flags["over_char"] = True

    # glossary miss
    if glossary:
        for term in glossary.terms:
            if term.target and term.source in src_text and term.target not in trg_text:
                notes.append(f"glossary_miss:{term.source}")
                flags["glossary_miss"] = True
                risk = max(risk, "medium", key={"low": 0, "medium": 1, "high": 2, "blocking": 3}.get)

    return risk, notes, flags


# ── batch composition ──────────────────────────────────────────────────────


def _compose_batch(
    batch: _BatchSpec,
    src_cues: Sequence[_SrcCue],
    parsed: _TranslatedBatch,
    *,
    glossary: Optional[Glossary],
    start_id: int,
) -> Tuple[List[TranslationCueOut], List[Dict[str, bool]]]:
    out: List[TranslationCueOut] = []
    flag_list: List[Dict[str, bool]] = []
    parsed_by_id = {c.cueId: c for c in parsed.cues}
    next_id = start_id
    for idx in batch.target_idxs:
        src = src_cues[idx]
        trans = parsed_by_id[src.id]
        risk, notes, flags = _grade_translation_risk(
            src.text, trans.translatedText, glossary=glossary
        )
        needs_review = trans.needsHumanReview or risk in ("high", "blocking")
        out.append(
            TranslationCueOut(
                id=f"trg_{next_id:06d}",
                sourceCueIds=[src.id],
                start=src.start,
                end=src.end,
                speaker=src.speaker,
                text=trans.translatedText,
                mapping="one-to-one",
                risk=risk,  # type: ignore[arg-type]
                needsHumanReview=needs_review,
                notes=notes,
            )
        )
        flag_list.append(flags)
        next_id += 1
    return out, flag_list


def _fallback_batch(
    batch: _BatchSpec,
    src_cues: Sequence[_SrcCue],
    *,
    reason: str,
    start_id: int,
) -> Tuple[List[TranslationCueOut], List[Dict[str, bool]]]:
    out: List[TranslationCueOut] = []
    flag_list: List[Dict[str, bool]] = []
    next_id = start_id
    for idx in batch.target_idxs:
        src = src_cues[idx]
        out.append(
            TranslationCueOut(
                id=f"trg_{next_id:06d}",
                sourceCueIds=[src.id],
                start=src.start,
                end=src.end,
                speaker=src.speaker,
                text=src.text,  # fallback 保留源文本，避免空 cue
                mapping="one-to-one",
                risk="blocking",
                needsHumanReview=True,
                notes=[reason],
            )
        )
        flag_list.append({"over_char": False, "over_cps": False, "glossary_miss": False})
        next_id += 1
    return out, flag_list


# ── checkpoint ─────────────────────────────────────────────────────────────


def _translate_work_dir(ws: Workspace, lang: str) -> Path:
    return ws.work / f"translate.{lang}"


def _translation_paths(ws: Workspace, lang: str) -> Tuple[Path, Path, Path]:
    return (
        ws.root / f"subtitles.{lang}.json",
        ws.root / f"subtitles.{lang}.srt",
        ws.root / f"subtitles.{lang}.vtt",
    )


def _content_hash(cues: Sequence[_SrcCue]) -> str:
    payload = json.dumps([(c.id, c.text) for c in cues], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(work_dir: Path, batch_index: int) -> Path:
    return work_dir / f"batch_{batch_index:03d}.json"


def _try_load_checkpoint(
    path: Path,
    *,
    expect_content_hash: str,
    expect_prompt_version: str,
    expect_model: str,
    expect_target_language: str,
) -> Optional[_BatchResult]:
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
        or data.get("targetLanguage") != expect_target_language
    ):
        return None
    cues = [TranslationCueOut.model_validate(c) for c in data.get("cues", [])]
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
    target_language: str,
    result: _BatchResult,
) -> None:
    payload = {
        "batchIndex": batch_index,
        "contentHash": content_hash,
        "promptVersion": prompt_version,
        "model": model,
        "targetLanguage": target_language,
        "promptTokens": result.prompt_tokens,
        "completionTokens": result.completion_tokens,
        "cues": [c.model_dump(by_alias=True) for c in result.out_cues],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ── SRT/VTT rendering ──────────────────────────────────────────────────────


def _render_translated(cues: Sequence[TranslationCueOut], *, time_fmt) -> str:
    parts: List[str] = []
    for i, c in enumerate(cues, 1):
        parts.append(str(i))
        parts.append(f"{time_fmt(c.start)} --> {time_fmt(c.end)}")
        body = c.text.strip()
        parts.append(f"{c.speaker}: {body}" if c.speaker else body)
        parts.append("")
    return "\n".join(parts) + "\n" if parts else ""


# ── main entry ──────────────────────────────────────────────────────────────


def _emit(em: EventMirror, payload: Dict[str, Any], *, to_stderr: bool) -> None:
    em.emit(payload)
    if to_stderr:
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stderr.flush()


def run_translate(
    req: TranslateRequest,
    *,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """主入口。返回 manifest 增量字典。"""
    ws = open_workspace(req.workdir)
    acquire_lock(ws)
    started = time.monotonic()
    work_dir = _translate_work_dir(ws, req.target_language)
    json_path, srt_path, vtt_path = _translation_paths(ws, req.target_language)
    try:
        # 1. 拒覆盖
        if json_path.exists() and not req.force:
            raise FileExistsError(
                f"refusing to overwrite {json_path}; pass --force"
            )

        # 2. 输入
        src_cues, src_lang, input_artifact, raw_meta = _load_source(ws)
        input_hash = "sha256:" + hashlib.sha256(raw_meta["bytes"]).hexdigest()
        source_language = req.source_language or src_lang or "auto"

        # 3. force：清空目标语言 work + 删旧 artifact / srt / vtt
        if req.force and work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if req.force:
            for p in (json_path, srt_path, vtt_path):
                if p.exists():
                    p.unlink()

        # 4. glossary
        gloss: Optional[Glossary] = None
        gloss_hash: Optional[str] = None
        if req.glossary_path is not None:
            gloss = load_glossary(req.glossary_path)
            gloss_hash = glossary_hash(gloss)

        # 5. prompt
        prompt_text, prompt_hash = load_prompt("translate", "v1")
        prompt_version = "translate.v1"
        system_msg = _render_prompt(
            prompt_text,
            source_language=source_language,
            target_language=req.target_language,
            style=req.style,
            length_policy=req.length_policy,
            glossary=gloss,
        )

        # 6. LLM client
        owns_client = llm_client is None
        client = llm_client or LLMClient(
            req.provider, model=req.model, timeout_s=req.timeout_s
        )
        used_model = client._model  # noqa: SLF001

        # 7. batches
        batches = _build_batches(
            src_cues,
            max_tokens=req.max_input_tokens,
            max_cues=req.max_cues_per_batch,
            context_prev=req.context_prev,
            context_next=req.context_next,
        )

        with EventMirror(ws) as emit:
            _emit(emit, {
                "event": "translate.start",
                "workdir": str(ws.root),
                "inputArtifact": input_artifact,
                "sourceLanguage": source_language,
                "targetLanguage": req.target_language,
                "provider": req.provider,
                "model": used_model,
                "style": req.style,
                "cueCount": len(src_cues),
                "batchCount": len(batches),
            }, to_stderr=req.json_events)

            all_out: List[TranslationCueOut] = []
            all_flags: List[Dict[str, bool]] = []
            prompt_tok_total = 0
            completion_tok_total = 0
            cached_count = 0

            try:
                next_trg_id = 1
                for batch in batches:
                    target_cues = [src_cues[i] for i in batch.target_idxs]
                    chash = _content_hash(target_cues)
                    cp_path = _checkpoint_path(work_dir, batch.index)

                    _emit(emit, {
                        "event": "translate.batch.start",
                        "batchIndex": batch.index,
                        "targetCount": len(target_cues),
                    }, to_stderr=req.json_events)

                    cached = _try_load_checkpoint(
                        cp_path,
                        expect_content_hash=chash,
                        expect_prompt_version=prompt_version,
                        expect_model=used_model,
                        expect_target_language=req.target_language,
                    )
                    if cached is not None:
                        all_out.extend(cached.out_cues)
                        # cache 不带 flags；要重算才能 metrics 准确
                        for cue in cached.out_cues:
                            risk, notes, flags = _grade_translation_risk(
                                "", cue.text, glossary=gloss
                            )
                            all_flags.append(flags)
                        prompt_tok_total += cached.prompt_tokens
                        completion_tok_total += cached.completion_tokens
                        cached_count += 1
                        next_trg_id += len(cached.out_cues)
                        _emit(emit, {
                            "event": "translate.batch.done",
                            "batchIndex": batch.index,
                            "cached": True,
                        }, to_stderr=req.json_events)
                        continue

                    user_obj = {
                        "context_prev": [
                            {"cueId": src_cues[i].id, "speaker": src_cues[i].speaker, "text": src_cues[i].text}
                            for i in batch.prev_idxs
                        ],
                        "targets": [
                            {"cueId": c.id, "speaker": c.speaker, "text": c.text}
                            for c in target_cues
                        ],
                        "context_next": [
                            {"cueId": src_cues[i].id, "speaker": src_cues[i].speaker, "text": src_cues[i].text}
                            for i in batch.next_idxs
                        ],
                    }
                    user_msg = json.dumps(user_obj, ensure_ascii=False, indent=2)
                    expected_ids = [c.id for c in target_cues]

                    try:
                        parsed, lr = _call_llm_with_repair(
                            client, system=system_msg, user=user_msg, expected_ids=expected_ids
                        )
                        out_cues, flags = _compose_batch(
                            batch, src_cues, parsed, glossary=gloss, start_id=next_trg_id
                        )
                        result_pt = lr.prompt_tokens
                        result_ct = lr.completion_tokens
                    except (LLMSchemaError, LLMRefusal) as e:
                        reason = "provider_refusal" if isinstance(e, LLMRefusal) else "schema_fail"
                        out_cues, flags = _fallback_batch(
                            batch, src_cues, reason=reason, start_id=next_trg_id
                        )
                        result_pt = 0
                        result_ct = 0

                    next_trg_id += len(out_cues)
                    br = _BatchResult(out_cues=out_cues, prompt_tokens=result_pt, completion_tokens=result_ct)
                    _write_checkpoint(
                        cp_path,
                        batch_index=batch.index,
                        content_hash=chash,
                        prompt_version=prompt_version,
                        model=used_model,
                        target_language=req.target_language,
                        result=br,
                    )
                    all_out.extend(br.out_cues)
                    all_flags.extend(flags)
                    prompt_tok_total += br.prompt_tokens
                    completion_tok_total += br.completion_tokens

                    _emit(emit, {
                        "event": "translate.batch.done",
                        "batchIndex": batch.index,
                        "cached": False,
                        "promptTokens": br.prompt_tokens,
                        "completionTokens": br.completion_tokens,
                    }, to_stderr=req.json_events)
            finally:
                if owns_client:
                    client.close()

            # 8. metrics
            cue_count = len(all_out)
            over_char = sum(1 for f in all_flags if f.get("over_char"))
            over_cps = sum(1 for f in all_flags if f.get("over_cps"))
            gloss_miss = sum(1 for f in all_flags if f.get("glossary_miss"))

            metrics = TranslationMetrics(
                cueCount=cue_count,
                overCharLimitRate=(over_char / cue_count) if cue_count else 0.0,
                overCpsRate=(over_cps / cue_count) if cue_count else 0.0,
                glossaryMissRate=(gloss_miss / cue_count) if cue_count else 0.0,
                promptTokensTotal=prompt_tok_total,
                completionTokensTotal=completion_tok_total,
            )
            params = TranslationParams(
                style=req.style,
                lengthPolicy=req.length_policy,
                cueMappingPolicy="one-to-one",
                glossaryHash=gloss_hash,
            )
            artifact = TranslationOutput(
                state="draft",
                sourceId=_resolve_source_id(ws),
                inputArtifact=input_artifact,
                inputHash=input_hash,
                sourceLanguage=source_language,
                targetLanguage=req.target_language,
                provider=req.provider,
                model=used_model,
                promptVersion=prompt_version,
                promptHash=prompt_hash,
                params=params,
                cues=all_out,
                metrics=metrics,
            )

            # 9. 写 JSON
            payload = artifact.model_dump(by_alias=True, exclude_none=False)
            tmp = json_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(json_path)

            # 10. 渲染 SRT/VTT
            if req.emit_srt:
                srt_path.write_text(_render_translated(all_out, time_fmt=format_srt_time), encoding="utf-8")
            if req.emit_vtt:
                vtt_path.write_text("WEBVTT\n\n" + _render_translated(all_out, time_fmt=format_vtt_time), encoding="utf-8")

            elapsed = time.monotonic() - started

            # 11. manifest 镜像
            existing = read_manifest(ws) or {}
            existing.setdefault("artifacts", {})
            existing["artifacts"][f"subtitle_translation_{req.target_language}_json"] = str(json_path)
            if req.emit_srt:
                existing["artifacts"][f"subtitle_translation_{req.target_language}_srt"] = str(srt_path)
            if req.emit_vtt:
                existing["artifacts"][f"subtitle_translation_{req.target_language}_vtt"] = str(vtt_path)

            translations = existing.setdefault("translations", {})
            translations[req.target_language] = {
                "state": "draft",
                "sourceLanguage": source_language,
                "inputArtifact": input_artifact,
                "inputHash": input_hash,
                "provider": req.provider,
                "model": used_model,
                "promptVersion": prompt_version,
                "promptHash": prompt_hash,
                "style": req.style,
                "lengthPolicy": req.length_policy,
                "cueMappingPolicy": "one-to-one",
                "glossaryHash": gloss_hash,
                "batchCount": len(batches),
                "cachedBatchCount": cached_count,
                "batchSize": {"tokensMax": req.max_input_tokens, "cuesMax": req.max_cues_per_batch},
                "overCharLimitRate": metrics.over_char_limit_rate,
                "overCpsRate": metrics.over_cps_rate,
                "glossaryMissRate": metrics.glossary_miss_rate,
                "promptTokens": prompt_tok_total,
                "completionTokens": completion_tok_total,
                "elapsedSecs": elapsed,
            }
            write_manifest(ws, existing)

            _emit(emit, {
                "event": "translate.done",
                "targetLanguage": req.target_language,
                "cueCount": cue_count,
                "overCharLimitRate": metrics.over_char_limit_rate,
                "glossaryMissRate": metrics.glossary_miss_rate,
                "promptTokens": prompt_tok_total,
                "completionTokens": completion_tok_total,
                "cachedBatchCount": cached_count,
                "elapsedSecs": elapsed,
            }, to_stderr=req.json_events)

            return translations[req.target_language]

    finally:
        release_lock(ws)


def _resolve_source_id(ws: Workspace) -> str:
    """优先从 proofread 拿 sourceId，否则 cues。"""
    if ws.proofread_json_path.is_file():
        try:
            doc = json.loads(ws.proofread_json_path.read_text(encoding="utf-8"))
            sid = doc.get("sourceId")
            if sid:
                return sid
        except (OSError, json.JSONDecodeError):
            pass
    if ws.cues_json_path.is_file():
        try:
            doc = json.loads(ws.cues_json_path.read_text(encoding="utf-8"))
            sid = doc.get("sourceId")
            if sid:
                return sid
        except (OSError, json.JSONDecodeError):
            pass
    return "unknown"
