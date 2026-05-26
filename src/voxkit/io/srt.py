"""SRT / VTT subtitle generators.

Two render paths:

  - **Segment path** (``to_subtitles_srt`` / ``to_subtitles_vtt``):
    one cue per ``TranscriptSegment``; the voxkit-native schema has no
    per-segment speaker field, so the renderer treats every segment as
    carrying the "Speaker A" placeholder.

  - **Cue path** (``to_subtitles_srt_from_cues`` / ``to_subtitles_vtt_from_cues``):
    one cue per ``SubtitleCue``; speaker prefix is per-cue (already carries
    diarization label). Used by the optional semantic resegment post-processor
    (:mod:`voxkit.core.semantic_resegment`).

Both paths share the same ``speaker_prefix`` policy
(:data:`SpeakerPrefixPolicy`). ``format_srt_time`` / ``format_vtt_time`` are
the single source of truth for subtitle timestamp formatting;
``commands/align.py`` delegates here so the two surfaces never drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from voxkit.io.schema import TranscriptionOutput

if TYPE_CHECKING:
    # SubtitleCue lives in core.semantic_resegment; only its attribute shape
    # (start / end / speaker / text) is read here, so keep the import in
    # TYPE_CHECKING to skip a runtime dep on a sibling layer.
    from voxkit.core.semantic_resegment import SubtitleCue

#: 渲染期 speaker 前缀策略。
#:   - "auto"  : 仅在 cue 序列含 ≥2 个不同**信息性** speaker 时加前缀（默认）
#:   - "always": 永远加前缀；speaker 为 None 时退回 "Speaker A"（旧行为）
#:   - "never" : 永远不加前缀
#: 单人讲座 + 未跑 diarize 时上游会塞 "Speaker A" 占位符——"auto" 用 distinct
#: 计数把这种 placeholder 噪声挡掉，避免 SRT 每条字幕都长 "Speaker A: ..."。
SpeakerPrefixPolicy = Literal["auto", "always", "never"]

#: 公认的"非信息性"占位符 speaker 标签。
#:   - "Speaker A" : ``TranscriptSegment`` / ``semantic_resegment`` 在没有
#:                   diarization 时的默认占位符
#:   - "Speaker ?" : ``align_speakers`` 对未匹配到 diarization turn 的 segment
#:                   的 fallback 标签
#: 这些标签不携带"谁在说"的信息，``auto`` 模式下不计入 distinct count，渲染层
#: 也不会把它们写进单条 cue 前缀（即使全局 show_prefix=True）。
PLACEHOLDER_SPEAKERS: "frozenset[str]" = frozenset({"Speaker A", "Speaker ?"})

__all__ = [
    "format_srt_time",
    "format_vtt_time",
    "to_subtitles_srt",
    "to_subtitles_vtt",
    "to_subtitles_srt_from_cues",
    "to_subtitles_vtt_from_cues",
    "SpeakerPrefixPolicy",
    "PLACEHOLDER_SPEAKERS",
    "should_show_speaker_prefix",
    "is_informative_speaker",
]


def is_informative_speaker(speaker: str | None) -> bool:
    """``speaker`` 是否携带可用的"谁在说"信息。

    非 None、非空白、非 :data:`PLACEHOLDER_SPEAKERS` 之一 → True。
    渲染层用这个判定来决定是否给单条 cue 加前缀。
    """
    if not speaker:
        return False
    if speaker in PLACEHOLDER_SPEAKERS:
        return False
    return True


def _coerce_policy(value: "SpeakerPrefixPolicy | bool") -> SpeakerPrefixPolicy:
    """Back-compat：旧调用方传 ``bool``（True/False），映射到 ``"always"/"never"``。"""
    if value is True:
        return "always"
    if value is False:
        return "never"
    return value


def should_show_speaker_prefix(
    speakers: Sequence[str | None],
    policy: "SpeakerPrefixPolicy | bool",
) -> bool:
    """根据策略 + 实际 speaker 分布决定是否渲染前缀。

    ``auto`` 的关键判定：去重后**信息性** speaker ≥ 2 个才显示。
    :data:`PLACEHOLDER_SPEAKERS` 中的标签（"Speaker A" / "Speaker ?"）不计入，
    避免：
      - 全 "Speaker A" 占位符场景被误判成 1 个 speaker → 加前缀（v0.7.1 B1）
      - "Speaker 1" + 单条 "Speaker ?" 未匹配 cue 被误判成 2 个 speaker → 把
        "Speaker ?:" 漏到那条 cue 上（v0.7.2 review #5）
    """
    policy = _coerce_policy(policy)
    if policy == "always":
        return True
    if policy == "never":
        return False
    distinct = {s for s in speakers if is_informative_speaker(s)}
    return len(distinct) >= 2


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


def _segment_text(text: str, show_placeholder_prefix: bool) -> str:
    """Apply optional ``"Speaker A: "`` placeholder prefix to a segment line.

    ``TranscriptSegment`` does not carry speaker info in v1 of the schema, so
    when the renderer needs to add a label at all it must be the placeholder.
    ``show_placeholder_prefix`` is the already-resolved policy decision (i.e.
    ``should_show_speaker_prefix`` was called by the caller with a synthetic
    placeholder list).
    """
    body = text.strip()
    if show_placeholder_prefix:
        return f"Speaker A: {body}"
    return body


def _segment_show_prefix(
    out: TranscriptionOutput,
    policy: "SpeakerPrefixPolicy | bool",
) -> bool:
    """Segment 路径的 "show prefix" 决策。
    Segments 没有 per-cue speaker，渲染层把它们当作全部 ``"Speaker A"`` 占位符。
    所以在 ``auto`` 下永不渲染（distinct informative = 0），在 ``always`` 下永远
    渲染，在 ``never`` 下永远不渲染。把决策路由到统一的 ``should_show_speaker_prefix``
    保持两条路径行为一致。
    """
    placeholders = ["Speaker A"] * len(out.segments)
    return should_show_speaker_prefix(placeholders, policy)


def to_subtitles_srt(
    out: TranscriptionOutput,
    *,
    speaker_prefix: "SpeakerPrefixPolicy | bool" = "auto",
) -> str:
    """Render the rich transcript as an SRT document.

    One cue per ``TranscriptSegment``. Cues are 1-indexed. Returns the full
    document as a single string with a trailing newline (so writing to disk
    via ``Path.write_text`` produces a POSIX-friendly file).

    ``speaker_prefix`` defaults to ``"auto"``——segment schema 没有 speaker 信息，
    auto 永远视为 "1 个非信息性 speaker" → 不加前缀。要恢复 v0.7.1 之前的
    "每条都长 Speaker A:" 行为，显式传 ``"always"`` 或 ``True``。
    """
    show_prefix = _segment_show_prefix(out, speaker_prefix)
    parts: list[str] = []
    for i, seg in enumerate(out.segments, 1):
        parts.append(str(i))
        parts.append(
            f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}"
        )
        parts.append(_segment_text(seg.text, show_prefix))
        parts.append("")  # blank line separates cues
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def to_subtitles_vtt(
    out: TranscriptionOutput,
    *,
    speaker_prefix: "SpeakerPrefixPolicy | bool" = "auto",
) -> str:
    """Render the rich transcript as a WebVTT document.

    Same iterator as ``to_subtitles_srt`` but with the ``WEBVTT`` header and
    period-separated millisecond timestamps. WebVTT cues do not require an
    integer cue number, but emitting one stays compatible with both specs and
    matches user expectation.
    """
    show_prefix = _segment_show_prefix(out, speaker_prefix)
    parts: list[str] = ["WEBVTT", ""]
    for i, seg in enumerate(out.segments, 1):
        parts.append(str(i))
        parts.append(
            f"{format_vtt_time(seg.start)} --> {format_vtt_time(seg.end)}"
        )
        parts.append(_segment_text(seg.text, show_prefix))
        parts.append("")
    return "\n".join(parts) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Cue path — for semantic_resegment post-processor output
# ─────────────────────────────────────────────────────────────────────────────


def _cue_text(text: str, speaker: str | None, *, show_prefix: bool) -> str:
    """Render one cue's text line, respecting both global ``show_prefix`` and
    per-cue placeholder filtering.

    Two layers of filtering:
      1. ``show_prefix`` (global): caller already decided whether any cue
         should carry a prefix (via :func:`should_show_speaker_prefix`).
      2. :func:`is_informative_speaker` (per-cue): even when global is True,
         a cue whose speaker is "Speaker A" / "Speaker ?" must not render the
         placeholder — that's noise, not signal.

    In ``always`` mode (the legacy back-compat path), per-cue placeholder
    filtering is bypassed via ``show_prefix=True`` AND the policy passing
    through; see ``_render_cues``.
    """
    body = text.strip()
    if show_prefix and speaker:
        return f"{speaker}: {body}"
    return body


def _render_cues(
    cues: Sequence["SubtitleCue"],
    *,
    time_fmt,
    header: list[str],
    speaker_prefix: "SpeakerPrefixPolicy | bool" = "auto",
) -> str:
    policy = _coerce_policy(speaker_prefix)
    show_prefix = should_show_speaker_prefix(
        [getattr(c, "speaker", None) for c in cues], policy
    )
    parts: list[str] = list(header)
    for i, c in enumerate(cues, 1):
        parts.append(str(i))
        parts.append(f"{time_fmt(c.start)} --> {time_fmt(c.end)}")
        # 全局 show_prefix=True 时，per-cue 决策：
        #   - "always" 模式：尊重旧行为，即使是占位符也加（用户显式要回退）
        #   - "auto" 模式：占位符 cue 跳过（避免 "Speaker ?:" 漏到未匹配 cue）
        cue_speaker = c.speaker
        if show_prefix and policy == "auto" and not is_informative_speaker(cue_speaker):
            cue_speaker = None
        parts.append(_cue_text(c.text, cue_speaker, show_prefix=show_prefix))
        parts.append("")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def to_subtitles_srt_from_cues(
    cues: Sequence["SubtitleCue"],
    *,
    speaker_prefix: "SpeakerPrefixPolicy | bool" = "auto",
) -> str:
    """Render :class:`SubtitleCue` sequence as an SRT document.

    Each cue already carries its own speaker label (or ``None`` when no
    diarization ran). The resegment module is responsible for monotonic
    timestamps and physical limits. ``speaker_prefix`` controls whether the
    ``"Speaker N: "`` prefix is rendered; default ``"auto"`` skips the prefix
    when there is only one distinct speaker (the common single-presenter case).
    """
    return _render_cues(
        cues, time_fmt=format_srt_time, header=[], speaker_prefix=speaker_prefix
    )


def to_subtitles_vtt_from_cues(
    cues: Sequence["SubtitleCue"],
    *,
    speaker_prefix: "SpeakerPrefixPolicy | bool" = "auto",
) -> str:
    """Render :class:`SubtitleCue` sequence as a WebVTT document."""
    return _render_cues(
        cues,
        time_fmt=format_vtt_time,
        header=["WEBVTT", ""],
        speaker_prefix=speaker_prefix,
    )
