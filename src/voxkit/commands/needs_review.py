"""``voxkit needs-review`` 子命令：列出 proofread / translate 的 review 队列。

针对下游反馈 U3：proofread / translate 在 cue 上标 ``needsHumanReview`` 与
``risk`` 但没有便捷的 listing 入口，使用者需要 ``jq`` 自己挖。本命令是 read-only
扫描器——读 artifact、过滤 → stdout 输出（text / json），stderr 写 1 行 summary。

不写盘、不 lock workspace、不调用 LLM；用 Pydantic schema 验证后再过滤，避免
手解 dict 漏字段。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from voxkit.io.schema import (
    ProofreadOutput,
    TranslationOutput,
)
from voxkit.io.srt import format_srt_time

# RiskLevel = Literal["low", "medium", "high", "blocking"]
_RISK_CHOICES: Tuple[str, ...] = ("low", "medium", "high", "blocking")
_DEFAULT_RISK_FILTER: Tuple[str, ...] = ("high", "blocking")
_TEXT_PREVIEW_CHARS = 60


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "needs-review",
        help=(
            "列出 proofread / translate artifact 中需要人工复核的 cue（"
            "needsHumanReview=True 或 risk ∈ {high, blocking}）。read-only。"
        ),
    )
    p.add_argument(
        "workdir",
        help=(
            "voxkit workdir。默认读 subtitles.proofread.json；若指定 --target "
            "则读 subtitles.<target>.json，缺失时回落到 proofread"
        ),
    )
    p.add_argument(
        "--target",
        default=None,
        help=(
            "目标语言代码（如 zh / en）。指定后优先读 subtitles.<target>.json；"
            "缺省直接读 subtitles.proofread.json"
        ),
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="输出格式：text（一行一 cue，默认）/ json（完整 cue 数组）",
    )
    p.add_argument(
        "--include-risk",
        default=None,
        help=(
            "逗号分隔的 risk 等级列表（low / medium / high / blocking）。"
            "缺省按 'high,blocking' 过滤——配合 needsHumanReview=True 逻辑或。"
            "传入则用该列表覆盖默认 risk 过滤集合（needsHumanReview 仍然或入）"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_risk_filter(raw: Optional[str]) -> Tuple[str, ...]:
    """``--include-risk`` 解析。空 → 默认 {high, blocking}。

    无效 risk 等级抛 ``ValueError``，由 :func:`run` 转 exit 2 + stderr 错误。
    """
    if raw is None:
        return _DEFAULT_RISK_FILTER
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        return _DEFAULT_RISK_FILTER
    bad = [p for p in parts if p not in _RISK_CHOICES]
    if bad:
        raise ValueError(
            f"unknown risk level(s): {','.join(bad)}; "
            f"valid choices: {','.join(_RISK_CHOICES)}"
        )
    # 去重保序
    seen: List[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return tuple(seen)


def _resolve_artifact_path(workdir: Path, target: Optional[str]) -> Path:
    """决议读哪个 artifact 文件。

    - ``--target zh`` 指定时优先 ``subtitles.zh.json``；不存在则回落 proofread
    - 缺省直接 ``subtitles.proofread.json``
    """
    if target:
        candidate = workdir / f"subtitles.{target}.json"
        if candidate.is_file():
            return candidate
        # fallback
        fallback = workdir / "subtitles.proofread.json"
        if fallback.is_file():
            return fallback
        # 都没有 → 返回 target 路径让上层报「文件不存在」（更明确）
        return candidate
    return workdir / "subtitles.proofread.json"


def _load_artifact(
    path: Path,
) -> Tuple[str, Sequence[Any], int]:
    """读 + Pydantic validate artifact。

    返回 ``(kind, cues, total)``：
      - ``kind``: "proofread" / "translation"
      - ``cues``: Pydantic 模型列表（``ProofreadCueOut`` 或 ``TranslationCueOut``）
      - ``total``: 总 cue 数（=len(cues)，给 summary 用）

    通过 ``sourceLanguage`` 字段是否存在判断 kind——translation schema 必有，
    proofread schema 没有。
    """
    doc: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "sourceLanguage" in doc or "targetLanguage" in doc:
        model = TranslationOutput.model_validate(doc)
        return "translation", model.cues, len(model.cues)
    model = ProofreadOutput.model_validate(doc)
    return "proofread", model.cues, len(model.cues)


def _cue_id(cue: Any) -> str:
    """proofread cue 用 ``cue_id``（alias cueId），translation cue 用 ``id``。"""
    # ProofreadCueOut.cue_id（field alias cueId）
    cid = getattr(cue, "cue_id", None)
    if cid:
        return cid
    # TranslationCueOut.id
    return getattr(cue, "id", "")


def _cue_start(cue: Any) -> float:
    """proofread cue 的开始时间是 ``source_start``，translation cue 是 ``start``。"""
    if hasattr(cue, "source_start"):
        return float(cue.source_start)
    return float(getattr(cue, "start", 0.0))


def _cue_text(cue: Any) -> str:
    """proofread cue 用 ``corrected_text``，translation cue 用 ``text``。"""
    if hasattr(cue, "corrected_text"):
        return cue.corrected_text or ""
    return getattr(cue, "text", "") or ""


def _filter_cues(
    cues: Iterable[Any], risk_filter: Tuple[str, ...]
) -> List[Any]:
    """needsHumanReview=True OR risk ∈ risk_filter。"""
    out: List[Any] = []
    rf = set(risk_filter)
    for c in cues:
        risk = str(getattr(c, "risk", "low"))
        needs = bool(getattr(c, "needs_human_review", False))
        if needs or risk in rf:
            out.append(c)
    return out


def _emit_text(cues: Sequence[Any]) -> str:
    """每行：``cue_id  HH:MM:SS,mmm  risk=...  needsReview=...  "text 前 60 char"``。

    text 中的换行折成空格，长文本截断后加 ``...``。
    """
    lines: List[str] = []
    for c in cues:
        cid = _cue_id(c)
        ts = format_srt_time(_cue_start(c))
        risk = str(getattr(c, "risk", "low"))
        needs = bool(getattr(c, "needs_human_review", False))
        text = _cue_text(c).replace("\n", " ").replace("\r", " ").strip()
        if len(text) > _TEXT_PREVIEW_CHARS:
            preview = text[:_TEXT_PREVIEW_CHARS] + "..."
        else:
            preview = text
        # risk 字段左侧对齐到 8 字符宽，方便目测扫读
        lines.append(
            f"{cid}  {ts}  risk={risk:<8}  needsReview={needs}  \"{preview}\""
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _emit_json(cues: Sequence[Any]) -> str:
    """完整 cue 数组 JSON dump（by_alias=True，保持外部 schema 字段名）。"""
    payload: List[Dict[str, Any]] = [
        c.model_dump(by_alias=True, mode="json") for c in cues
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# entry
# ─────────────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    if not workdir.is_dir():
        sys.stderr.write(f"error: workdir not a directory: {workdir}\n")
        return 2

    try:
        risk_filter = _parse_risk_filter(args.include_risk)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    artifact_path = _resolve_artifact_path(workdir, args.target)
    if not artifact_path.is_file():
        sys.stderr.write(
            f"error: artifact not found: {artifact_path}\n"
        )
        return 2

    try:
        _kind, cues, total = _load_artifact(artifact_path)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error: invalid JSON in {artifact_path}: {exc}\n")
        return 2
    except Exception as exc:  # pydantic ValidationError 等
        sys.stderr.write(
            f"error: failed to parse {artifact_path.name}: {exc}\n"
        )
        return 2

    flagged = _filter_cues(cues, risk_filter)

    if args.format == "json":
        sys.stdout.write(_emit_json(flagged))
    else:
        sys.stdout.write(_emit_text(flagged))

    sys.stderr.write(
        f"needs-review: {len(flagged)} cue(s) flagged out of {total} total\n"
    )
    return 0


