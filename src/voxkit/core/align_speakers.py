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

Optional robustness knobs (opt-in; defaults preserve historical behaviour):

* ``min_dia_duration_s``: drop diarization segments shorter than this before
  computing overlaps. pyannote occasionally emits 17ms / 80ms phantom
  alt-speaker bursts (overlap detection or clustering jitter); leaving them
  in poisons the per-segment majority vote. Recommended 0.5s for typical
  podcasts. ``0.0`` (default) = no filter.
* ``fallback_to_nearest``: if a transcript segment falls entirely in a
  between-segment gap (zero overlap), assign it to the nearest diarization
  segment instead of marking it unmatched. Prevents phantom speaker switches
  on sub-second words. ``False`` (default) = old behaviour.
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


def _nearest_diarization(
    seg_start: float,
    seg_end: float,
    dia_segments: list[Segment],
) -> int | None:
    """Find the diarization segment whose interval is closest in time to
    ``[seg_start, seg_end)``. Returns ``None`` only when ``dia_segments`` is empty.

    "Closest" = minimum gap between the segment midpoint and either endpoint
    of the dia interval. Ties (rare) → lower index.
    """
    if not dia_segments:
        return None
    mid = (seg_start + seg_end) / 2
    best_idx = 0
    best_dist = min(abs(mid - dia_segments[0].start), abs(mid - dia_segments[0].end))
    for i in range(1, len(dia_segments)):
        d = dia_segments[i]
        dist = min(abs(mid - d.start), abs(mid - d.end))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def assign_speakers(
    segments: list[TranscriptSegment],
    diarization: DiarizationOutput,
    *,
    speaker_labels: SpeakerLabelMode = "ranked",
    min_dia_duration_s: float = 0.0,
    fallback_to_nearest: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Assign a speaker label to each transcript segment via maximum overlap.

    Args:
        segments: voxkit-native transcript segments (chunk-merged, absolute
            timeline). Not mutated.
        diarization: pyannote diarization output with ranked + raw labels.
        speaker_labels: ``"ranked"`` → use ``diarization.segments[i].speaker``
            (e.g. ``"Speaker 1"``); ``"raw"`` → use ``raw_speaker``
            (e.g. ``"SPEAKER_01"``).
        min_dia_duration_s: filter out diarization segments shorter than this
            before computing overlaps. Prevents pyannote's sub-100ms phantom
            alt-speaker bursts from misattributing segments. ``0.0`` = off.
        fallback_to_nearest: when a transcript segment has zero overlap with
            any (filtered) diarization segment, assign it to the nearest
            diarization segment instead of marking unmatched. ``False`` = off
            (old behaviour: segment goes to ``unmatched``).

    Returns:
        Tuple of:

        * ``speaker_by_seg_id`` — ``{segment.id: speaker_label}``. Contains an
          entry for every segment that resolved to a label (via overlap or
          fallback). Callers default unmatched segments to ``"Speaker ?"``.
        * ``unmatched_seg_ids`` — segment ids that had neither overlap nor a
          usable fallback (only happens when no diarization segments survive
          filtering, or when ``fallback_to_nearest=False``).

    Notes:
        * Pure function: does not mutate ``segments`` or ``diarization``.
        * Empty ``segments`` → ``({}, [])``.
        * Empty effective dia (after filter) → all input segments unmatched.
    """
    speaker_by_id: dict[str, str] = {}
    unmatched: list[str] = []

    dia_effective: list[Segment] = (
        [d for d in diarization.segments if (d.end - d.start) >= min_dia_duration_s]
        if min_dia_duration_s > 0
        else list(diarization.segments)
    )

    if not dia_effective:
        return speaker_by_id, [s.id for s in segments]

    def _label_of(d: Segment) -> str:
        return d.speaker if speaker_labels == "ranked" else d.raw_speaker

    for seg in segments:
        idx, _overlap = _best_diarization_match(seg.start, seg.end, dia_effective)
        if idx is not None:
            speaker_by_id[seg.id] = _label_of(dia_effective[idx])
            continue
        # No overlap.
        if fallback_to_nearest:
            nearest_idx = _nearest_diarization(seg.start, seg.end, dia_effective)
            if nearest_idx is not None:
                speaker_by_id[seg.id] = _label_of(dia_effective[nearest_idx])
                continue
        unmatched.append(seg.id)

    return speaker_by_id, unmatched
