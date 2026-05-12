"""Artifact state machine — 仅 ``draft → reviewed → final``，反向 / 跨级转换默认拒绝。``stale`` 是计算态不持久化。

提供两件事：

  1. 识别 artifact 类型（``proofread`` / ``translation``）并用 Pydantic 校验。
  2. 在状态机允许的边上修改 ``state`` 字段（含 ``reviewedBy`` / ``reviewedAt`` 元数据），
     原子写盘，并把变更镜像到 ``manifest.json``。

仅 ``voxkit review`` 子命令使用；不直接接触 LLM / 网络。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

# Re-exported below for tooling/IDE completion.

from pydantic import ValidationError

from voxkit.core.workspace import open_workspace, read_manifest, write_manifest
from voxkit.io.schema import ProofreadOutput, TranslationOutput

__all__ = [
    "LifecycleError",
    "LIFECYCLE_ORDER",
    "ForceLevel",
    "detect_artifact_kind",
    "load_artifact",
    "validate_self_consistency",
    "transition_state",
    "mirror_to_manifest",
    "peek_artifact_state",
    "gate_force_overwrite",
]


# ── 常量 ────────────────────────────────────────────────────────────────────

#: 生命周期顺序；数值越大越靠后。``stale`` 不在此处（计算态不持久化）。
LIFECYCLE_ORDER: dict[str, int] = {"draft": 0, "reviewed": 1, "final": 2}

#: ``--force`` 等级；高级隐含覆盖低级。``None`` = 不覆盖任何已存在的 artifact。
ForceLevel = Optional[Literal["draft", "reviewed", "final"]]

#: ``subtitles.<lang>.json`` 文件名里抽取语言代码。
_TRANSLATION_NAME_RE = re.compile(r"^subtitles\.([A-Za-z][A-Za-z0-9_-]*)\.json$")


# ── 错误类型 ────────────────────────────────────────────────────────────────


class LifecycleError(Exception):
    """artifact 不合法 / 不允许的状态转换 / 自洽性校验失败时抛出。"""


# ── 识别 + 加载 ─────────────────────────────────────────────────────────────


def detect_artifact_kind(path: Path) -> Literal["proofread", "translation"]:
    """通过 ``cues[0]`` 字段名识别 artifact 类型。

    - 含 ``cueId`` → proofread
    - 含 ``sourceCueIds`` → translation
    - 其他情况抛 :class:`LifecycleError`。

    仅看第一条 cue；空 cue 列表也判失败。
    """
    if not path.is_file():
        raise LifecycleError(f"artifact not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot parse JSON at {path}: {exc}") from exc

    cues = raw.get("cues")
    if not isinstance(cues, list) or len(cues) == 0:
        raise LifecycleError(
            f"artifact {path} has no cues; cannot detect kind"
        )
    head = cues[0]
    if not isinstance(head, dict):
        raise LifecycleError(f"artifact {path} cues[0] is not an object")
    if "cueId" in head:
        return "proofread"
    if "sourceCueIds" in head:
        return "translation"
    raise LifecycleError(
        f"artifact {path} cues[0] has neither 'cueId' nor 'sourceCueIds'; "
        "cannot detect kind"
    )


def load_artifact(path: Path) -> Tuple[str, dict]:
    """识别 + Pydantic 校验，返回 ``(kind, raw_dict)``。

    校验失败抛 :class:`LifecycleError`。返回的是原始 dict（保留所有未知字段），
    Pydantic 仅用作合约校验。
    """
    kind = detect_artifact_kind(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        if kind == "proofread":
            ProofreadOutput.model_validate(raw)
        else:
            TranslationOutput.model_validate(raw)
    except ValidationError as exc:
        raise LifecycleError(
            f"artifact {path} fails {kind} schema: {exc}"
        ) from exc
    return kind, raw


# ── 轻量 state peek（force gate 用） ────────────────────────────────────────


def peek_artifact_state(path: Path) -> Optional[str]:
    """读 artifact 顶层 ``state`` 字段；文件不存在或损坏返回 ``None``。

    **故意不走 Pydantic 全量校验**：半损坏 artifact 也要能被读出 state，否则
    ``--force`` 永远救不回来。直接 try/except 替代预 stat（避免 TOCTOU）。
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    state = raw.get("state")
    if isinstance(state, str):
        return state
    return None


def gate_force_overwrite(
    path: Path,
    *,
    force_level: ForceLevel,
    artifact_label: str,
) -> None:
    """Reviewed/Final artifact 防误覆盖；被 proofread/translate pipeline 共用。

    规则：
      - artifact 不存在 → 直通
      - force_level=None → 拒绝
      - force_level 等级必须 ≥ existing.state（未知 state 视为 ``draft``）

    raise FileExistsError 当被拒绝时；消息里告诉用户该传哪个 flag。
    """
    existing_state = peek_artifact_state(path)
    if existing_state is None:
        return  # 不存在 / 损坏到读不出 state；force gate 直通
    existing_rank = LIFECYCLE_ORDER.get(existing_state, 0)
    if force_level is None:
        raise FileExistsError(
            f"refusing to overwrite {path} (state={existing_state!r}); "
            f"pass --force / --force-reviewed / --force-final"
        )
    force_rank = LIFECYCLE_ORDER.get(force_level, 0)
    if force_rank < existing_rank:
        next_flag = {
            "reviewed": "--force-reviewed",
            "final": "--force-final",
        }.get(existing_state, "--force")
        raise FileExistsError(
            f"refusing to overwrite {artifact_label} in state {existing_state!r}; "
            f"current --force level only covers ≤ {force_level!r}. "
            f"Pass {next_flag} to acknowledge the destructive intent."
        )


# ── 自洽性校验 ──────────────────────────────────────────────────────────────


def validate_self_consistency(kind: str, raw: dict) -> None:
    """检查 artifact 自身一致性（不与"原版"对比，因为 confirm 只有单文件）。

    断言：
      - ``cues`` 至少一条
      - cue id 字段（proofread=``cueId``, translation=``id``）无重复

    其他结构性字段已经在 :func:`load_artifact` 里通过 Pydantic 校验过。
    """
    cues = raw.get("cues") or []
    if not isinstance(cues, list) or len(cues) == 0:
        raise LifecycleError("artifact has empty cues list")

    if kind == "proofread":
        id_key = "cueId"
    elif kind == "translation":
        id_key = "id"
    else:
        raise LifecycleError(f"unknown artifact kind: {kind!r}")

    seen: set[str] = set()
    dups: list[str] = []
    for c in cues:
        cid = c.get(id_key) if isinstance(c, dict) else None
        if cid is None:
            raise LifecycleError(f"cue missing {id_key}: {c!r}")
        if cid in seen:
            dups.append(cid)
        else:
            seen.add(cid)
    if dups:
        raise LifecycleError(
            f"duplicate {id_key} values in artifact: {sorted(set(dups))!r}"
        )


# ── 状态转换 ────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    """UTC ISO-8601 时间戳（秒精度，``Z`` 结尾）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp 文件 + ``os.replace``，与项目其他原子写一致。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def transition_state(
    path: Path,
    *,
    to: Literal["reviewed", "final"],
    reviewer: str | None = None,
) -> dict:
    """把 artifact 的 ``state`` 推进一档，并原子写盘。

    规则：
      - ``draft → reviewed`` 需要非空 ``reviewer``；写入 ``reviewedBy``/``reviewedAt``。
      - ``reviewed → final`` 不写 reviewer 字段，保留 confirm 阶段的元数据。
      - 任何反向 / 跨级 / 重复转换都拒绝。

    返回写盘后的完整 dict（用于 manifest 镜像或日志）。
    """
    if to not in ("reviewed", "final"):
        raise LifecycleError(f"unsupported target state: {to!r}")

    kind, raw = load_artifact(path)
    validate_self_consistency(kind, raw)

    current = raw.get("state", "draft")
    if current not in LIFECYCLE_ORDER:
        raise LifecycleError(f"unknown current state: {current!r}")

    # 必须严格 +1 步
    expected_prev = {"reviewed": "draft", "final": "reviewed"}[to]
    if current != expected_prev:
        raise LifecycleError(
            f"cannot transition state {current!r} → {to!r}; "
            f"expected current state {expected_prev!r}. "
            "Backward / skip-level transitions are not allowed."
        )

    if to == "reviewed":
        if not reviewer or not reviewer.strip():
            raise LifecycleError(
                "transition draft → reviewed requires a non-empty reviewer name"
            )
        raw["state"] = "reviewed"
        raw["reviewedBy"] = reviewer.strip()
        raw["reviewedAt"] = _utc_now_iso()
    else:  # to == "final"
        # 保留既有 reviewedBy/reviewedAt，不覆盖。
        raw["state"] = "final"

    _atomic_write_json(path, raw)
    return raw


# ── manifest 镜像 ───────────────────────────────────────────────────────────


def _infer_translation_language(artifact_path: Path) -> str:
    """从 ``subtitles.<lang>.json`` 文件名提取 ``<lang>``。"""
    m = _TRANSLATION_NAME_RE.match(artifact_path.name)
    if not m:
        raise LifecycleError(
            f"cannot infer translation language from filename: {artifact_path.name!r}; "
            "expected pattern 'subtitles.<lang>.json'"
        )
    return m.group(1)


def mirror_to_manifest(workdir: Path, artifact_path: Path, updated: dict) -> None:
    """把 artifact 顶层 ``state`` + reviewer 元数据写回 ``<workdir>/manifest.json``。

    - proofread → ``manifest["proofread"]``
    - translation → ``manifest["translations"][<lang>]``（lang 从文件名推断）

    缺失对应段时会原地建空 dict，再写回；不删除其他字段。
    """
    ws = open_workspace(workdir)
    manifest = read_manifest(ws) or {}

    kind = detect_artifact_kind(artifact_path)
    state = updated.get("state")
    reviewed_by = updated.get("reviewedBy")
    reviewed_at = updated.get("reviewedAt")

    if kind == "proofread":
        section: dict[str, Any] = manifest.setdefault("proofread", {})
        section["state"] = state
        if reviewed_by is not None:
            section["reviewedBy"] = reviewed_by
        if reviewed_at is not None:
            section["reviewedAt"] = reviewed_at
    else:
        lang = _infer_translation_language(artifact_path)
        translations: dict[str, Any] = manifest.setdefault("translations", {})
        section = translations.setdefault(lang, {})
        section["state"] = state
        if reviewed_by is not None:
            section["reviewedBy"] = reviewed_by
        if reviewed_at is not None:
            section["reviewedAt"] = reviewed_at

    write_manifest(ws, manifest)
