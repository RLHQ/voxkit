"""Writer: ``SubtitleCue[]`` → ``subtitles.cues.json``.

This is the render-layer counterpart to :mod:`voxkit.io.remixr_adapter`. It
serializes the cues produced by :mod:`voxkit.core.semantic_resegment` into a
machine-readable JSON document so downstream consumers (Remixr, etc.) can
ingest the resegmented output directly instead of re-parsing SRT text.

Design notes:

* The on-disk shape is :class:`~voxkit.io.schema.SubtitleCuesOutput` — keep all
  Pydantic-driven contract decisions there, this module only handles the
  cue → model adaptation and exclusive-write semantics.
* ``write_cues_json`` uses ``open(path, "x")`` to match the same exclusive-
  create idempotency contract as :func:`voxkit.io.remixr_adapter.write_remixr_json`
  (``transcript.raw.json`` is immutable per workdir; cues mirror that).
* ``params`` and ``metrics`` are opaque ``dict[str, Any]`` values here so this
  module stays independent of :class:`~voxkit.core.semantic_resegment.ResegmentParams`.
  The pipeline is responsible for snapshotting params via ``dataclasses.asdict``
  and computing render-layer quality metrics.
* This file MUST NOT be confused with ``transcript.raw.json``: the former is
  a render-layer derivative; the latter is a Remixr-shaped adapter view of
  the ASR transcript (downstream contract, not whisper raw — that lives at
  ``work/chunks/chunk_NNN.json``; the voxkit-native merged transcript lives at
  ``transcript.voxkit.json``). See ``docs/capability-artifact-model.md`` for
  the full layering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from voxkit.io.schema import SubtitleCueOut, SubtitleCuesOutput

if TYPE_CHECKING:
    # Same trick as io/srt.py — only the SubtitleCue *shape*
    # (start / end / speaker / text) is read here.
    from voxkit.core.semantic_resegment import SubtitleCue

__all__ = ["to_cues_output", "write_cues_json"]


def to_cues_output(
    cues: Sequence["SubtitleCue"],
    *,
    source_id: str,
    resegment: str,
    params: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> SubtitleCuesOutput:
    """Build a :class:`SubtitleCuesOutput` from in-memory cues.

    No filesystem side effects; useful in tests and when the pipeline wants to
    embed the cues elsewhere (e.g. inside ``transcript.voxkit.json``).
    """
    # cue id 在序列化边界赋值，不污染内部 SubtitleCue 数据流；格式与 schemaVersion=2
    # 契约绑定（见 SubtitleCueOut docstring）。1-based 6 位零填充，最高支持百万 cue。
    cue_models = [
        SubtitleCueOut(
            id=f"cue_{i + 1:06d}",
            start=float(c.start),
            end=float(c.end),
            speaker=c.speaker,
            text=c.text,
        )
        for i, c in enumerate(cues)
    ]
    return SubtitleCuesOutput(
        sourceId=source_id,
        resegment=resegment,
        params=params,
        metrics=metrics,
        cues=cue_models,
    )


def write_cues_json(
    cues: Sequence["SubtitleCue"],
    path: Path,
    *,
    source_id: str,
    resegment: str,
    params: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
    indent: int = 2,
) -> None:
    """Serialize ``cues`` to ``path`` exclusively.

    Raises :class:`FileExistsError` on collision so re-running into the same
    workdir fails loudly (mirrors the ``transcript.raw.json`` contract).
    """
    out = to_cues_output(
        cues,
        source_id=source_id,
        resegment=resegment,
        params=params,
        metrics=metrics,
    )
    payload = out.model_dump(by_alias=True, exclude_none=True)
    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
