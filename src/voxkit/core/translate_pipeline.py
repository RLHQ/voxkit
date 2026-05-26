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

from pydantic import BaseModel, ValidationError, field_validator

from voxkit.core.lifecycle import ForceLevel, gate_force_overwrite
from voxkit.core.proofread_risk import SYSTEM_OVERHEAD_TOKENS, estimate_tokens, is_cjk_char
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
from voxkit.io.srt import (
    SpeakerPrefixPolicy,
    format_srt_time,
    format_vtt_time,
    is_informative_speaker,
    should_show_speaker_prefix,
)
from voxkit.llm import ChatResult, LLMClient
from voxkit.llm.errors import LLMError, LLMRateLimit, LLMRefusal, LLMSchemaError, LLMTimeout
from voxkit.llm.prompts import load_prompt

__all__ = [
    "TranslateRequest",
    "run_translate",
]


# ── public request ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranslateRequest:
    """``force_level`` 与 :class:`ProofreadRequest.force_level` 同义。"""

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
    # speaker_prefix:
    #   "auto"  → 仅在出现 ≥2 个非空 speaker 时给 SRT/VTT 每条 cue 加 "Speaker X: "
    #   "always"→ 旧行为（>=v0.7.1 之前）：始终加前缀；单人也会被强加 "Speaker A:"
    #   "never" → 永不加前缀
    speaker_prefix: SpeakerPrefixPolicy = "auto"
    # render_only（v0.7.2 review #3）：跳过 LLM / checkpoint，直接读现有
    # subtitles.<lang>.json 重渲染 SRT/VTT。专为"只想换 --speaker-prefix"这类
    # 纯格式化重渲染场景，避免被迫 --force 重 LLM 浪费 token。
    render_only: bool = False
    force_level: ForceLevel = None
    json_events: bool = False
    timeout_s: float = 60.0

    # 兼容旧调用：force=True ⇔ force_level="draft"
    force: bool = False

    def __post_init__(self) -> None:
        if self.force and self.force_level is None:
            object.__setattr__(self, "force_level", "draft")


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
    """LLM 译文契约。``translatedText`` 必须含非空白字符；空译文走 repair 兜底。"""

    cueId: str
    translatedText: str
    needsHumanReview: bool = False

    @field_validator("translatedText")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("translatedText must contain non-whitespace characters")
        return v


class _TranslatedBatch(BaseModel):
    cues: List[_TranslatedCue]


# ── input loading ──────────────────────────────────────────────────────────


def _load_source(
    ws: Workspace,
) -> Tuple[List[_SrcCue], str, str, Dict[str, Any], str]:
    """优先读 proofread，回落 cues。返回 (cues, source_lang, input_artifact, raw_meta, source_id)。"""
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
        return src_cues, doc.language, "subtitles.proofread.json", {"bytes": raw_bytes}, doc.source_id

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
        return src_cues, lang, "subtitles.cues.json", {"bytes": raw_bytes}, doc2.source_id

    raise FileNotFoundError(
        f"no translation input in {ws.root}: need subtitles.proofread.json "
        "or subtitles.cues.json. "
        "Run `voxkit transcribe <input> --workdir <dir> --resegment=semantic` "
        "to produce subtitles.cues.json (optionally followed by `voxkit proofread <dir>`)."
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
    """与 :func:`voxkit.core.proofread_pipeline._build_batches` 同形规则：

      - 不跨 speaker
      - 预算 ``max_tokens − SYSTEM_OVERHEAD_TOKENS`` 含 context_prev/next cue tokens
      - 单 cue 超预算独占一批
    """
    n = len(cues)
    if n == 0:
        return []
    effective_budget = max(max_tokens - SYSTEM_OVERHEAD_TOKENS, 1)
    tok = [estimate_tokens(c.text) for c in cues]

    batches: List[_BatchSpec] = []
    i = 0
    bi = 0
    while i < n:
        speaker = cues[i].speaker
        prev_start = max(0, i - context_prev)
        prev_idxs = list(range(prev_start, i))
        prev_tokens = sum(tok[prev_start:i])

        target_idxs = [i]
        tokens = tok[i]
        j = i + 1
        while j < n:
            if cues[j].speaker != speaker:
                break
            next_window_end = min(n, j + 1 + context_next)
            next_tokens = sum(tok[j + 1:next_window_end])
            if prev_tokens + tokens + tok[j] + next_tokens > effective_budget:
                break
            if len(target_idxs) >= max_cues:
                break
            target_idxs.append(j)
            tokens += tok[j]
            j += 1
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


# Cache key 与 proofread 同形（详见 proofread_pipeline._content_hash 注释）：
#   contentHash = (id, text, start, end, speaker)
#   policyHash  = (provider, model, promptVersion, promptHash, style, lengthPolicy,
#                  cueMappingPolicy, glossaryHash, sourceLanguage, targetLanguage,
#                  cacheSchema)
#   命中要求 contentHash AND policyHash 都相等，cacheSchema=current。

#: cache schema 版本；语义改了就 bump，老 checkpoint 自动作废。
TRANSLATE_CACHE_SCHEMA = 2


def _content_hash(cues: Sequence[_SrcCue]) -> str:
    payload = json.dumps(
        [
            {
                "id": c.id,
                "text": c.text,
                "start": round(float(c.start), 6),
                "end": round(float(c.end), 6),
                "speaker": c.speaker,
            }
            for c in cues
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_hash(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    prompt_hash: str,
    style: str,
    length_policy: str,
    cue_mapping_policy: str,
    glossary_hash: Optional[str],
    source_language: str,
    target_language: str,
) -> str:
    payload = json.dumps(
        {
            "cacheSchema": TRANSLATE_CACHE_SCHEMA,
            "provider": provider,
            "model": model,
            "promptVersion": prompt_version,
            "promptHash": prompt_hash,
            "style": style,
            "lengthPolicy": length_policy,
            "cueMappingPolicy": cue_mapping_policy,
            "glossaryHash": glossary_hash,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(work_dir: Path, batch_index: int) -> Path:
    return work_dir / f"batch_{batch_index:03d}.json"


def _pending_path(work_dir: Path, batch_index: int) -> Path:
    """传输/限流失败 batch 的占位文件。"""
    return work_dir / f"batch_{batch_index:03d}.pending.json"


def _try_load_checkpoint(
    path: Path,
    *,
    expect_content_hash: str,
    expect_policy_hash: str,
) -> Optional[_BatchResult]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if int(data.get("cacheSchema", 1)) != TRANSLATE_CACHE_SCHEMA:
        return None
    if data.get("contentHash") != expect_content_hash:
        return None
    if data.get("policyHash") != expect_policy_hash:
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
    policy_hash: str,
    result: _BatchResult,
) -> None:
    payload = {
        "cacheSchema": TRANSLATE_CACHE_SCHEMA,
        "batchIndex": batch_index,
        "contentHash": content_hash,
        "policyHash": policy_hash,
        "promptTokens": result.prompt_tokens,
        "completionTokens": result.completion_tokens,
        "cues": [c.model_dump(by_alias=True) for c in result.out_cues],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_pending_marker(
    path: Path,
    *,
    batch_index: int,
    error_kind: str,
    error_message: str,
) -> None:
    payload = {
        "batchIndex": batch_index,
        "errorKind": error_kind,
        "errorMessage": error_message[:500],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ── SRT/VTT rendering ──────────────────────────────────────────────────────


def _atomic_write_text(path: Path, body: str) -> None:
    """tmp + ``Path.replace`` 原子覆写文本文件；写失败时旧文件保留。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _render_translated(
    cues: Sequence[TranslationCueOut],
    *,
    time_fmt,
    speaker_prefix: SpeakerPrefixPolicy = "auto",
) -> str:
    show_prefix = should_show_speaker_prefix(
        [c.speaker for c in cues], speaker_prefix
    )
    parts: List[str] = []
    for i, c in enumerate(cues, 1):
        parts.append(str(i))
        parts.append(f"{time_fmt(c.start)} --> {time_fmt(c.end)}")
        body = c.text.strip()
        # 与 voxkit.io.srt._render_cues 同形的 per-cue 占位符过滤（review #5）：
        # auto 模式下 'Speaker A' / 'Speaker ?' 不写前缀；always 保留旧行为。
        write_prefix = bool(show_prefix and c.speaker) and (
            speaker_prefix != "auto" or is_informative_speaker(c.speaker)
        )
        if write_prefix:
            parts.append(f"{c.speaker}: {body}")
        else:
            parts.append(body)
        parts.append("")
    return "\n".join(parts) + "\n" if parts else ""


# ── main entry ──────────────────────────────────────────────────────────────


def _emit(em: EventMirror, payload: Dict[str, Any], *, to_stderr: bool) -> None:
    em.emit(payload)
    if to_stderr:
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stderr.flush()


def _run_render_only(
    req: TranslateRequest,
    *,
    json_path: Path,
    srt_path: Path,
    vtt_path: Path,
) -> Dict[str, Any]:
    """``--render-only`` short-circuit：读现有 subtitles.<lang>.json，按当前
    ``speaker_prefix`` 重渲染 SRT/VTT。不动 LLM、不动 work checkpoints、不动
    artifact 的 ``state`` / ``reviewedBy`` / ``provider`` / cost metrics。

    返回最小 dict（仅含本次重渲染信息），manifest **不更新**——既然 LLM 没跑、
    cost 没变，没必要污染 audit 字段。
    """
    if not json_path.is_file():
        raise FileNotFoundError(
            f"--render-only requires existing {json_path.name}; run "
            f"`voxkit translate <workdir> --target-language {req.target_language}` "
            f"first to produce it"
        )
    doc = TranslationOutput.model_validate_json(json_path.read_text(encoding="utf-8"))
    cues = doc.cues
    if req.emit_srt:
        _atomic_write_text(
            srt_path,
            _render_translated(
                cues, time_fmt=format_srt_time, speaker_prefix=req.speaker_prefix
            ),
        )
    if req.emit_vtt:
        _atomic_write_text(
            vtt_path,
            "WEBVTT\n\n"
            + _render_translated(
                cues, time_fmt=format_vtt_time, speaker_prefix=req.speaker_prefix
            ),
        )
    return {
        "renderOnly": True,
        "targetLanguage": req.target_language,
        "cueCount": len(cues),
        "speakerPrefix": req.speaker_prefix,
        "state": doc.state,
    }


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
        # 0. --render-only short-circuit：纯重渲染，不动 LLM / checkpoint。
        # 在 gate_force_overwrite 之前处理——重渲染不算"覆盖 artifact"。
        if req.render_only:
            if req.force_level is not None:
                raise ValueError(
                    "--render-only is incompatible with --force / --force-reviewed / "
                    "--force-final (render-only never touches the existing JSON artifact)"
                )
            return _run_render_only(
                req, json_path=json_path, srt_path=srt_path, vtt_path=vtt_path
            )

        # 1. 拒覆盖（reviewed/final 必须显式 --force-reviewed/--force-final）
        gate_force_overwrite(
            json_path,
            force_level=req.force_level,
            artifact_label=f"subtitles.{req.target_language}.json",
        )

        # 2. 输入
        src_cues, src_lang, input_artifact, raw_meta, source_id = _load_source(ws)
        input_hash = "sha256:" + hashlib.sha256(raw_meta["bytes"]).hexdigest()
        source_language = req.source_language or src_lang or "auto"

        # 3. force：清空目标语言 work checkpoints。
        # **不要** 预先 unlink 旧 artifact / srt / vtt —— 改成最后 atomic replace；
        # LLM 失败时旧 artifact 仍然完整可用。
        if req.force_level is not None and work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

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

        # 6a. policy hash：所有影响 LLM 输出的策略集中计 hash。
        policy_h = _policy_hash(
            provider=req.provider,
            model=used_model,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            style=req.style,
            length_policy=req.length_policy,
            cue_mapping_policy="one-to-one",
            glossary_hash=gloss_hash,
            source_language=source_language,
            target_language=req.target_language,
        )

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

            # cost 拆 fresh vs cached（同 proofread 形态）：
            all_out: List[TranslationCueOut] = []
            all_flags: List[Dict[str, bool]] = []
            fresh_pt = 0
            fresh_ct = 0
            cached_pt = 0
            cached_ct = 0
            cached_count = 0
            pending_batches: List[Dict[str, Any]] = []

            try:
                next_trg_id = 1
                for batch in batches:
                    target_cues = [src_cues[i] for i in batch.target_idxs]
                    chash = _content_hash(target_cues)
                    cp_path = _checkpoint_path(work_dir, batch.index)
                    pending_path = _pending_path(work_dir, batch.index)

                    _emit(emit, {
                        "event": "translate.batch.start",
                        "batchIndex": batch.index,
                        "targetCount": len(target_cues),
                    }, to_stderr=req.json_events)

                    cached = _try_load_checkpoint(
                        cp_path,
                        expect_content_hash=chash,
                        expect_policy_hash=policy_h,
                    )
                    if cached is not None:
                        all_out.extend(cached.out_cues)
                        # cache 不带 flags；要重算才能 metrics 准确
                        for cue in cached.out_cues:
                            risk, notes, flags = _grade_translation_risk(
                                "", cue.text, glossary=gloss
                            )
                            all_flags.append(flags)
                        cached_pt += cached.prompt_tokens
                        cached_ct += cached.completion_tokens
                        cached_count += 1
                        next_trg_id += len(cached.out_cues)
                        pending_path.unlink(missing_ok=True)
                        _emit(emit, {
                            "event": "translate.batch.done",
                            "batchIndex": batch.index,
                            "cached": True,
                        }, to_stderr=req.json_events)
                        if not req.json_events:
                            sys.stderr.write(
                                f"translate {req.target_language}: batch "
                                f"{batch.index + 1}/{len(batches)} done "
                                f"({len(all_out)}/{len(src_cues)} cues) [cache hit]\n"
                            )
                            sys.stderr.flush()
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
                        # 内容层失败：fallback 标 blocking + needsHumanReview，
                        # 视为本批已完结（写 checkpoint）。
                        reason = "provider_refusal" if isinstance(e, LLMRefusal) else "schema_fail"
                        out_cues, flags = _fallback_batch(
                            batch, src_cues, reason=reason, start_id=next_trg_id
                        )
                        result_pt = 0
                        result_ct = 0
                    except (LLMTimeout, LLMRateLimit) as e:
                        # 仅捕获传输层失败；其它 LLMError 子类视为 bug 向上抛。
                        kind = type(e).__name__
                        _write_pending_marker(
                            pending_path,
                            batch_index=batch.index,
                            error_kind=kind,
                            error_message=str(e),
                        )
                        pending_batches.append({
                            "batchIndex": batch.index,
                            "errorKind": kind,
                            "errorMessage": str(e)[:200],
                        })
                        _emit(emit, {
                            "event": "translate.batch.failed",
                            "batchIndex": batch.index,
                            "errorKind": kind,
                            "errorMessage": str(e)[:200],
                            "willRetryOnRerun": True,
                        }, to_stderr=req.json_events)
                        continue

                    next_trg_id += len(out_cues)
                    br = _BatchResult(out_cues=out_cues, prompt_tokens=result_pt, completion_tokens=result_ct)
                    _write_checkpoint(
                        cp_path,
                        batch_index=batch.index,
                        content_hash=chash,
                        policy_hash=policy_h,
                        result=br,
                    )
                    pending_path.unlink(missing_ok=True)
                    all_out.extend(br.out_cues)
                    all_flags.extend(flags)
                    fresh_pt += br.prompt_tokens
                    fresh_ct += br.completion_tokens

                    _emit(emit, {
                        "event": "translate.batch.done",
                        "batchIndex": batch.index,
                        "cached": False,
                        "promptTokens": br.prompt_tokens,
                        "completionTokens": br.completion_tokens,
                    }, to_stderr=req.json_events)
                    if not req.json_events:
                        sys.stderr.write(
                            f"translate {req.target_language}: batch "
                            f"{batch.index + 1}/{len(batches)} done "
                            f"({len(all_out)}/{len(src_cues)} cues, "
                            f"+{br.prompt_tokens + br.completion_tokens} tokens)\n"
                        )
                        sys.stderr.flush()
            finally:
                if owns_client:
                    client.close()

            # 7a. pending batch 存在 → 拒绝写稳定 artifact，让 rerun 补做
            if pending_batches:
                _emit(emit, {
                    "event": "translate.partial",
                    "completedBatches": len(batches) - len(pending_batches),
                    "pendingBatches": len(pending_batches),
                    "details": pending_batches,
                }, to_stderr=req.json_events)
                first_kind = pending_batches[0]["errorKind"]
                raise LLMError(
                    f"translate incomplete: {len(pending_batches)}/{len(batches)} "
                    f"batches failed (first error: {first_kind}). "
                    f"Completed batches were checkpointed; rerun without --force "
                    f"to retry only the pending batches."
                )

            prompt_tok_total = fresh_pt + cached_pt
            completion_tok_total = fresh_ct + cached_ct

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
                sourceId=source_id,
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

            # 9. 写 JSON（atomic replace；旧 artifact 未删，失败时不丢）
            payload = artifact.model_dump(by_alias=True, exclude_none=False)
            tmp = json_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(json_path)

            # 10. 渲染 SRT/VTT（同样 atomic，避免半截字幕被播放器读到）
            if req.emit_srt:
                _atomic_write_text(
                    srt_path,
                    _render_translated(
                        all_out,
                        time_fmt=format_srt_time,
                        speaker_prefix=req.speaker_prefix,
                    ),
                )
            if req.emit_vtt:
                _atomic_write_text(
                    vtt_path,
                    "WEBVTT\n\n"
                    + _render_translated(
                        all_out,
                        time_fmt=format_vtt_time,
                        speaker_prefix=req.speaker_prefix,
                    ),
                )

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
                "outputArtifact": f"subtitles.{req.target_language}.json",
                "outputSchemaVersion": "1",
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
                "freshPromptTokens": fresh_pt,
                "freshCompletionTokens": fresh_ct,
                "cachedPromptTokens": cached_pt,
                "cachedCompletionTokens": cached_ct,
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


