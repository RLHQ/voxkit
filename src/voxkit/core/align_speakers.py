"""Pure speaker-assignment for transcript segments.

Given a list of :class:`~voxkit.io.schema.TranscriptSegment` and a
:class:`~voxkit.io.schema.DiarizationOutput`, produce a mapping
``{segment_id: speaker_label}`` by maximum-overlap match against the
diarization timeline.

Design rationale (matches Phase 2 spec):

* Returns ``(dict, list[unmatched_ids])`` instead of mutating the input
  segments — voxkit-native :class:`TranscriptSegment` deliberately has no
  ``speaker`` field (see ``io/schema.py``), so the caller injects the labels
  into the Remixr-shaped output (which DOES carry ``speaker``) and into the
  subtitle generators.
* Tie-break: equal overlap → lower-indexed diarization segment wins. This is
  deterministic and matches how the legacy ``commands/align.py::_assign_speaker``
  loop behaves (first-seen at equal overlap).
* Pure: never mutates ``segments`` or ``diarization``.
"""

from __future__ import annotations

from typing import Literal

from voxkit.io.schema import DiarizationOutput, Segment, TranscriptSegment

__all__ = ["assign_speakers", "SpeakerLabelMode"]


SpeakerLabelMode = Literal["ranked", "raw"]


def _best_diarization_match(
    seg_start: float,
    seg_end: float,
    dia_segments: list[Segment],
) -> tuple[int, float] | tuple[None, float]:
    """Return ``(idx, overlap_secs)`` of the diarization segment with the
    longest overlap, or ``(None, 0.0)`` if no positive overlap exists.

    Tie-break: lower index wins (we use strict ``>`` when comparing, so the
    first encountered maximum is kept).
    """
    best_idx: int | None = None
    best_overlap = 0.0
    for i, d in enumerate(dia_segments):
        overlap = min(seg_end, d.end) - max(seg_start, d.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    return best_idx, best_overlap


def assign_speakers(
    segments: list[TranscriptSegment],
    diarization: DiarizationOutput,
    *,
    speaker_labels: SpeakerLabelMode = "ranked",
) -> tuple[dict[str, str], list[str]]:
    """Assign a speaker label to each transcript segment via maximum overlap.

    Args:
        segments: voxkit-native transcript segments (chunk-merged, absolute
            timeline). Not mutated.
        diarization: pyannote diarization output with ranked + raw labels.
        speaker_labels: ``"ranked"`` → use ``diarization.segments[i].speaker``
            (e.g. ``"Speaker 1"``); ``"raw"`` → use ``raw_speaker``
            (e.g. ``"SPEAKER_01"``).

    Returns:
        Tuple of:

        * ``speaker_by_seg_id`` — ``{segment.id: speaker_label}``. Only contains
          entries for segments that had at least one positive-overlap match;
          callers default unmatched segments to ``"Speaker ?"`` (or whatever
          placeholder fits their context).
        * ``unmatched_seg_ids`` — list of ``segment.id`` values (in input order)
          for segments with NO overlapping diarization segment. Callers can
          surface this as a warning count.

    Notes:
        * Pure function: does not mutate ``segments`` or ``diarization``.
        * Empty ``segments`` → ``({}, [])``.
        * Empty ``diarization.segments`` → all input segments are unmatched.
    """
    speaker_by_id: dict[str, str] = {}
    unmatched: list[str] = []

    if not diarization.segments:
        return speaker_by_id, [s.id for s in segments]

    for seg in segments:
        idx, _overlap = _best_diarization_match(
            seg.start, seg.end, diarization.segments
        )
        if idx is None:
            unmatched.append(seg.id)
            continue
        d = diarization.segments[idx]
        label = d.speaker if speaker_labels == "ranked" else d.raw_speaker
        speaker_by_id[seg.id] = label

    return speaker_by_id, unmatched
