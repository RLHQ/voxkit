"""SRT / VTT subtitle generators for voxkit-native ``TranscriptionOutput``.

``format_srt_time`` / ``format_vtt_time`` are the single source of truth for
subtitle timestamp formatting; ``commands/align.py`` delegates here so the
two surfaces never drift.

These functions accept ``TranscriptionOutput`` (the rich form) only; they do
not accept ``RemixrTranscript`` because that would defeat the speaker-prefix
contract (Remixr segments always carry ``"Speaker A"`` as a placeholder).
"""

from __future__ import annotations

from voxkit.io.schema import TranscriptionOutput

__all__ = [
    "format_srt_time",
    "format_vtt_time",
    "to_subtitles_srt",
    "to_subtitles_vtt",
]


def _split_hms_ms(seconds: float) -> tuple[int, int, int, int]:
    """Split a non-negative float-seconds value into (h, m, s, ms).

    Negative inputs are clamped to zero (subtitles never go before t=0).
    Sub-millisecond fractions are rounded; if rounding overflows the
    millisecond field we cascade the carry up cleanly (so 59.9999s never
    renders as ``00:00:59,1000``).
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return h, m, s, ms


def format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm`` (comma)."""
    h, m, s, ms = _split_hms_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp ``HH:MM:SS.mmm`` (period)."""
    h, m, s, ms = _split_hms_ms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _segment_text(text: str, speaker_prefix: bool) -> str:
    """Apply optional speaker prefix.

    ``TranscriptSegment`` does not carry speaker info in v1 of the schema, so
    this prefix is unconditionally ``"Speaker A: "`` when enabled — matching
    the Remixr placeholder used by ``remixr_adapter``.
    """
    body = text.strip()
    if speaker_prefix:
        return f"Speaker A: {body}"
    return body


def to_subtitles_srt(
    out: TranscriptionOutput, *, speaker_prefix: bool = True
) -> str:
    """Render the rich transcript as an SRT document.

    One cue per ``TranscriptSegment``. Cues are 1-indexed. Returns the full
    document as a single string with a trailing newline (so writing to disk
    via ``Path.write_text`` produces a POSIX-friendly file).
    """
    parts: list[str] = []
    for i, seg in enumerate(out.segments, 1):
        parts.append(str(i))
        parts.append(
            f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}"
        )
        parts.append(_segment_text(seg.text, speaker_prefix))
        parts.append("")  # blank line separates cues
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def to_subtitles_vtt(
    out: TranscriptionOutput, *, speaker_prefix: bool = True
) -> str:
    """Render the rich transcript as a WebVTT document.

    Same iterator as ``to_subtitles_srt`` but with the ``WEBVTT`` header and
    period-separated millisecond timestamps. WebVTT cues do not require an
    integer cue number, but emitting one stays compatible with both specs and
    matches user expectation.
    """
    parts: list[str] = ["WEBVTT", ""]
    for i, seg in enumerate(out.segments, 1):
        parts.append(str(i))
        parts.append(
            f"{format_vtt_time(seg.start)} --> {format_vtt_time(seg.end)}"
        )
        parts.append(_segment_text(seg.text, speaker_prefix))
        parts.append("")
    return "\n".join(parts) + "\n"
