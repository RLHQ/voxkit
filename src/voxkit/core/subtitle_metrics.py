"""Subtitle quality metrics for semantic resegmentation outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor
from typing import Sequence

from voxkit.core.semantic_resegment import ResegmentParams, SubtitleCue

__all__ = [
    "SubtitleMetrics",
    "compute_subtitle_metrics",
]


@dataclass(frozen=True)
class SubtitleMetrics:
    """Manifest-serializable summary of subtitle cue quality."""

    cueCount: int
    avgCueDurS: float
    p50CueDurS: float
    p90CueDurS: float
    flashCueRate: float
    longCueRate: float
    avgChars: float
    overCharLimitRate: float
    overCpsRate: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a plain JSON/manifest-friendly mapping."""

        return asdict(self)


def compute_subtitle_metrics(
    cues: Sequence[SubtitleCue],
    params: ResegmentParams,
) -> SubtitleMetrics:
    """Compute subtitle quality metrics without mutating inputs.

    Durations are clamped at zero for aggregate duration statistics so malformed
    or reversed cue timestamps cannot produce surprising negative averages.
    Cues with non-positive duration and non-empty text are counted as over-CPS
    when ``params.max_cps`` is non-negative, because they cannot be rendered at
    a finite reading speed.
    """

    cue_count = len(cues)
    if cue_count == 0:
        return SubtitleMetrics(
            cueCount=0,
            avgCueDurS=0.0,
            p50CueDurS=0.0,
            p90CueDurS=0.0,
            flashCueRate=0.0,
            longCueRate=0.0,
            avgChars=0.0,
            overCharLimitRate=0.0,
            overCpsRate=0.0,
        )

    durations = [_duration_s(cue) for cue in cues]
    char_counts = [len(cue.text or "") for cue in cues]

    flash_count = sum(1 for dur in durations if dur < params.min_dur_s)
    long_count = sum(1 for dur in durations if dur > params.max_dur_s)
    over_char_count = sum(1 for chars in char_counts if chars > params.max_chars)
    over_cps_count = sum(
        1
        for dur, chars in zip(durations, char_counts, strict=True)
        if _is_over_cps(dur, chars, params.max_cps)
    )

    return SubtitleMetrics(
        cueCount=cue_count,
        avgCueDurS=sum(durations) / cue_count,
        p50CueDurS=_percentile(durations, 0.50),
        p90CueDurS=_percentile(durations, 0.90),
        flashCueRate=flash_count / cue_count,
        longCueRate=long_count / cue_count,
        avgChars=sum(char_counts) / cue_count,
        overCharLimitRate=over_char_count / cue_count,
        overCpsRate=over_cps_count / cue_count,
    )


def _duration_s(cue: SubtitleCue) -> float:
    return max(0.0, float(cue.end) - float(cue.start))


def _is_over_cps(duration_s: float, chars: int, max_cps: float) -> bool:
    if chars <= 0:
        return False
    if duration_s <= 0.0:
        return max_cps >= 0.0
    return (chars / duration_s) > max_cps


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = floor(pos)
    upper = ceil(pos)
    if lower == upper:
        return sorted_values[lower]

    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
