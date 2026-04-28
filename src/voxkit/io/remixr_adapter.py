"""Adapter: voxkit-native ``TranscriptionOutput`` → Remixr ``transcript.raw.json``.

This is the *single point of truth* for how voxkit's rich transcript maps onto
Remixr's Zod schema (``packages/shared/src/types/transcript.ts``). All Remixr
compatibility decisions live here so the rest of the pipeline can stay in
voxkit-native types.

Design notes (every choice has a Remixr-side reason):

* ``sourceId``: required, validated non-empty and slash-free. Remixr stores
  transcripts under ``storage/projects/{projectId}/sources/{sourceId}/`` so a
  ``/`` in ``sourceId`` would either path-traverse or break Remixr's lookups.
* ``segments[i].id``: defensively re-numbered ``seg_NNN`` (1-indexed, width-3)
  on export. We don't trust upstream IDs because pipeline merging can produce
  gaps.
* ``segments[i].speaker = "Speaker A"``: pre-diarization placeholder; Remixr
  has a separate diarization pipeline that overwrites this. Hardcoding here
  matches the prototype output exactly.
* ``segments[i].subtitles = []``: filled later by Remixr's proofread agent.
* ``rawText`` is never set: that field is what Remixr uses to detect
  "proofread happened" — emitting it from voxkit would falsely mark a raw
  transcript as proofread.
* ``write_remixr_json`` uses ``open(..., "x")``: matches the user-decided
  exclusive-write idempotency contract for ``transcript.raw.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from voxkit.io.schema import (
    RemixrSegment,
    RemixrTranscript,
    RemixrWord,
    TranscriptionOutput,
)

__all__ = ["to_remixr_transcript", "write_remixr_json"]


def _validate_source_id(source_id: str) -> str:
    """Reject empty or path-traversal-flavoured source ids."""
    if not isinstance(source_id, str):
        raise TypeError(f"source_id must be a str, got {type(source_id).__name__}")
    if not source_id:
        raise ValueError("source_id must be non-empty")
    if "/" in source_id or "\\" in source_id:
        raise ValueError(
            f"source_id must not contain path separators, got {source_id!r}"
        )
    return source_id


def to_remixr_transcript(
    out: TranscriptionOutput,
    *,
    source_id: str,
) -> RemixrTranscript:
    """Map voxkit-native ``TranscriptionOutput`` → Remixr ``RemixrTranscript``.

    Rules:
      - ``sourceId`` = ``source_id`` (validated non-empty, no path separators).
      - ``segments[i].id`` = ``f"seg_{i+1:03d}"`` (defensive zero-pad re-id).
      - ``segments[i].speaker`` = ``"Speaker A"`` (pre-diarization placeholder).
      - ``segments[i].subtitles`` = ``[]`` (Remixr proofread fills later).
      - ``segments[i].words`` is a field-for-field passthrough.
      - ``rawText`` is never written.
    """
    _validate_source_id(source_id)

    remixr_segments: list[RemixrSegment] = []
    for i, seg in enumerate(out.segments):
        words = [
            RemixrWord(word=w.word, start=w.start, end=w.end) for w in seg.words
        ]
        remixr_segments.append(
            RemixrSegment(
                id=f"seg_{i + 1:03d}",
                speaker="Speaker A",
                start=seg.start,
                end=seg.end,
                text=seg.text,
                subtitles=[],
                words=words,
            )
        )

    return RemixrTranscript(sourceId=source_id, segments=remixr_segments)


def write_remixr_json(
    t: RemixrTranscript,
    path: Path,
    *,
    metadata: Optional[dict] = None,
    indent: int = 2,
) -> None:
    """Write a ``RemixrTranscript`` to ``path`` exclusively.

    Uses ``open(path, "x")`` so a colliding file raises ``FileExistsError``;
    this matches the user-decided "raw is immutable, switch workdir or delete"
    contract.

    Serialization:
      - ``payload = t.model_dump(exclude_none=True)`` — these field names ARE
        the final on-disk keys, do not pass ``by_alias``.
      - if ``metadata`` is given, it is attached as ``payload["_metadata"]``.
        Remixr's Zod schema ignores unknown keys, so this is forward-safe.
      - ``json.dumps(..., ensure_ascii=False, indent=indent)`` plus a trailing
        newline. ``ensure_ascii=False`` keeps CJK readable in storage.

    Args:
        t: Remixr transcript to write.
        path: Destination ``transcript.raw.json`` path.
        metadata: Optional voxkit-side audit metadata (e.g.
            ``{"voxkitVersion": "0.3.0"}``).
        indent: JSON indent level (default 2 to match Remixr prototype).
    """
    payload = t.model_dump(exclude_none=True)
    if metadata is not None:
        payload["_metadata"] = metadata

    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"

    # "x" → exclusive create; raises FileExistsError on collision.
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
