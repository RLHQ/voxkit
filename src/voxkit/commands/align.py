"""voxkit align — 把 transcript.raw.json + diarization JSON 对齐成 SRT。

策略：对每个 transcript segment，找时间重叠最大的 diarization 段，把那个
speaker 标签打在该 segment 上。

支持两种 transcript 输入格式：
  1. Remixr transcript.raw.json：含 segments 数组，每条 {start, end, text}（秒）
  2. whisper.cpp --output-json-full：含 transcription 数组，每条 {offsets: {from, to}, text}（毫秒）

speaker 标号：
  - speaker_labels=ranked → 用 diarization JSON 里 speakers[].id（已 ranked）
  - speaker_labels=raw    → 用 segments[].rawSpeaker

输出 SRT 格式：
    1
    00:00:00,070 --> 00:00:13,100
    Speaker 1: text...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

from voxkit.core.align_speakers import SpeakerLabelMode, assign_speakers
from voxkit.io.schema import DiarizationOutput, Segment, TranscriptSegment
from voxkit.io.srt import format_srt_time


@dataclass
class TranscriptSeg:
    """统一格式：start/end 单位是毫秒。"""
    start_ms: int
    end_ms: int
    text: str


def _parse_transcript(path: Path) -> List[TranscriptSeg]:
    """自动识别 Remixr / whisper.cpp 两种格式。"""
    data = json.loads(path.read_text())
    segs: List[TranscriptSeg] = []

    # whisper.cpp --output-json-full
    if isinstance(data, dict) and "transcription" in data:
        for t in data["transcription"]:
            segs.append(TranscriptSeg(
                start_ms=int(t["offsets"]["from"]),
                end_ms=int(t["offsets"]["to"]),
                text=str(t["text"]).strip(),
            ))
        return segs

    # Remixr transcript.raw.json：约定字段是 segments[]，时间单位秒
    if isinstance(data, dict) and "segments" in data:
        for s in data["segments"]:
            start = s.get("start") or s.get("startSecs") or 0
            end = s.get("end") or s.get("endSecs") or 0
            text = s.get("text", "")
            segs.append(TranscriptSeg(
                start_ms=int(round(float(start) * 1000)),
                end_ms=int(round(float(end) * 1000)),
                text=str(text).strip(),
            ))
        return segs

    raise ValueError(
        "无法识别 transcript 格式：期望 Remixr transcript.raw.json"
        "（{segments: [...]）或 whisper.cpp --output-json-full（{transcription: [...]}）"
    )


def _format_srt_time(ms: int) -> str:
    """毫秒 → ``HH:MM:SS,mmm``。委托给 io/srt 的浮点秒版本，保持唯一来源。"""
    return format_srt_time(ms / 1000.0)


def _assign_speaker(seg: TranscriptSeg, dia_segments: List[Segment]) -> Optional[Segment]:
    """重叠最大的 diarization 段；无重叠返回 None。

    Thin wrapper that mirrors the legacy signature for any external caller —
    internally delegates to :mod:`voxkit.core.align_speakers` for the overlap
    math (single source of truth). Returns the matching ``Segment`` or
    ``None``.
    """
    from voxkit.core.align_speakers import _best_diarization_match

    seg_s = seg.start_ms / 1000.0
    seg_e = seg.end_ms / 1000.0
    idx, _overlap = _best_diarization_match(seg_s, seg_e, dia_segments)
    if idx is None:
        return None
    return dia_segments[idx]


def _to_transcript_segments(transcript_segs: List[TranscriptSeg]) -> List[TranscriptSegment]:
    """Promote the lightweight ``TranscriptSeg`` (ms-based, id-less) into
    Pydantic :class:`TranscriptSegment` (sec-based, id-bearing) so we can call
    :func:`assign_speakers`. ``id`` is a synthetic 1-indexed string — only used
    as a key into the returned mapping; never serialised.
    """
    out: List[TranscriptSegment] = []
    for i, w in enumerate(transcript_segs):
        out.append(
            TranscriptSegment(
                id=str(i),
                start=w.start_ms / 1000.0,
                end=w.end_ms / 1000.0,
                text=w.text,
            )
        )
    return out


def align_to_srt(
    *,
    transcript_path: Path,
    diarization: DiarizationOutput,
    out_srt: Path,
    speaker_labels: str = "ranked",
) -> Path:
    """主对齐函数：被 diarize 子命令和独立 align 子命令共用。

    返回写入的 SRT 路径。

    Implementation note — overlap logic delegates to
    :func:`voxkit.core.align_speakers.assign_speakers`. This module owns:

    * transcript JSON parsing (Remixr / whisper.cpp formats)
    * SRT emission
    * the "Speaker ?" placeholder for unmatched segments

    The pure overlap-math lives in ``align_speakers`` so the transcribe
    pipeline can reuse it without touching this CLI handler.
    """
    transcript_segs = _parse_transcript(transcript_path)

    # Promote to Pydantic shape and delegate overlap assignment.
    pyd_segments = _to_transcript_segments(transcript_segs)
    speaker_by_id, unmatched_ids = assign_speakers(
        pyd_segments,
        diarization,
        speaker_labels=cast(SpeakerLabelMode, speaker_labels),
    )

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    miss = 0
    for i, w in enumerate(transcript_segs, 1):
        seg_id = str(i - 1)  # matches _to_transcript_segments's 0-indexed ids
        label = speaker_by_id.get(seg_id)
        if label is None:
            miss += 1
            label = "Speaker ?"

        lines.append(str(i))
        lines.append(
            f"{_format_srt_time(w.start_ms)} --> {_format_srt_time(w.end_ms)}"
        )
        lines.append(f"{label}: {w.text}")
        lines.append("")

    # Defensive: miss count should always equal len(unmatched_ids).
    assert miss == len(unmatched_ids), (
        f"miss={miss} != unmatched={len(unmatched_ids)} (alignment bug)"
    )

    out_srt.write_text("\n".join(lines))
    print(
        f"[align] {out_srt} ({len(transcript_segs)} segs, "
        f"{diarization.num_speakers} speakers, {miss} 段无 speaker)"
    )
    return out_srt


def run(args: argparse.Namespace) -> int:
    """voxkit align 子命令入口。"""
    transcript = Path(args.transcript).expanduser().resolve()
    dia_path = Path(args.diarization).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    if not transcript.is_file():
        print(f"❌ transcript 不存在: {transcript}", file=sys.stderr)
        return 1
    if not dia_path.is_file():
        print(f"❌ diarization 不存在: {dia_path}", file=sys.stderr)
        return 1

    diarization = DiarizationOutput.model_validate_json(dia_path.read_text())
    align_to_srt(
        transcript_path=transcript,
        diarization=diarization,
        out_srt=out,
        speaker_labels=args.speaker_labels,
    )
    return 0


__all__ = ["run", "align_to_srt", "_parse_transcript", "_format_srt_time"]
