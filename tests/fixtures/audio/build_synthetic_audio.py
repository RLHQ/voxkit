#!/usr/bin/env python3
"""Build deterministic synthetic WAV fixtures for chunk-boundary experiments.

The generated files are intentionally *not* real speech. They encode
"speech-like" regions as low-volume tone/noise and silence as zero samples.
This lets tests validate chunk planning, VAD/silence detection, and subtitle
timing plumbing without shipping large real recordings.

Default scenarios use a short target boundary (12s) so local tests stay fast.
Use ``--scale 50`` to produce ~600s-boundary variants for manual experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
MAX_I16 = 32767

Kind = Literal["tone", "noise", "silence"]


@dataclass(frozen=True)
class SegmentSpec:
    kind: Kind
    start: float
    end: float
    label: str
    frequency: float = 440.0
    amplitude: float = 0.18


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    suggested_chunk_threshold_secs: float
    suggested_chunk_secs: float
    suggested_chunk_overlap_secs: float
    target_boundary_secs: float
    expected_boundary_window_secs: tuple[float, float]
    segments: list[SegmentSpec]


def _scaled(x: float, scale: float) -> float:
    return round(x * scale, 6)


def _seg(
    kind: Kind,
    start: float,
    end: float,
    label: str,
    *,
    frequency: float = 440.0,
    amplitude: float = 0.18,
    scale: float,
) -> SegmentSpec:
    return SegmentSpec(
        kind=kind,
        start=_scaled(start, scale),
        end=_scaled(end, scale),
        label=label,
        frequency=frequency,
        amplitude=amplitude,
    )


def build_scenarios(scale: float = 1.0) -> list[Scenario]:
    """Return deterministic short-form scenarios.

    With the default scale:
      - target chunk length is 12s
      - overlap is 2s
      - threshold is 18s

    These values mirror the production relationship (threshold > chunk,
    overlap < chunk) without generating multi-hundred-MB fixtures.
    """
    chunk = _scaled(12.0, scale)
    overlap = _scaled(2.0, scale)
    threshold = _scaled(18.0, scale)

    return [
        Scenario(
            name="boundary_silence_near_target",
            description=(
                "Two silence candidates around the target boundary. A "
                "VAD-aligned planner should choose one of them instead of "
                "cutting through the middle tone region."
            ),
            suggested_chunk_threshold_secs=threshold,
            suggested_chunk_secs=chunk,
            suggested_chunk_overlap_secs=overlap,
            target_boundary_secs=chunk,
            expected_boundary_window_secs=(_scaled(10.8, scale), _scaled(13.3, scale)),
            segments=[
                _seg("tone", 0.0, 10.8, "speech_before_boundary", scale=scale),
                _seg("silence", 10.8, 11.4, "pre_boundary_silence", scale=scale),
                _seg("tone", 11.4, 12.8, "speech_crossing_grid_cut", scale=scale),
                _seg("silence", 12.8, 13.3, "post_boundary_silence", scale=scale),
                _seg("tone", 13.3, 26.0, "speech_after_boundary", frequency=523.25, scale=scale),
            ],
        ),
        Scenario(
            name="boundary_no_silence",
            description=(
                "Continuous tone through the boundary search window. The "
                "planner should fall back to fixed-grid chunking."
            ),
            suggested_chunk_threshold_secs=threshold,
            suggested_chunk_secs=chunk,
            suggested_chunk_overlap_secs=overlap,
            target_boundary_secs=chunk,
            expected_boundary_window_secs=(_scaled(10.0, scale), _scaled(14.0, scale)),
            segments=[
                _seg("tone", 0.0, 26.0, "continuous_speech_surrogate", scale=scale),
            ],
        ),
        Scenario(
            name="boundary_multi_candidate",
            description=(
                "Multiple silence candidates with different distance and "
                "duration tradeoffs. This locks down the boundary scoring rule."
            ),
            suggested_chunk_threshold_secs=threshold,
            suggested_chunk_secs=chunk,
            suggested_chunk_overlap_secs=overlap,
            target_boundary_secs=chunk,
            expected_boundary_window_secs=(_scaled(9.8, scale), _scaled(14.6, scale)),
            segments=[
                _seg("tone", 0.0, 9.8, "speech_a", frequency=392.0, scale=scale),
                _seg("silence", 9.8, 10.1, "short_far_silence", scale=scale),
                _seg("tone", 10.1, 11.75, "speech_b", frequency=440.0, scale=scale),
                _seg("silence", 11.75, 12.15, "short_near_silence", scale=scale),
                _seg("tone", 12.15, 14.0, "speech_c", frequency=493.88, scale=scale),
                _seg("silence", 14.0, 14.6, "long_far_silence", scale=scale),
                _seg("tone", 14.6, 26.0, "speech_d", frequency=523.25, scale=scale),
            ],
        ),
        Scenario(
            name="cjk_phrase_cadence",
            description=(
                "Short tone phrases separated by punctuation-like pauses. "
                "Pair this audio with synthetic CJK segment fixtures when "
                "smoke-testing char-level subtitle resegmentation."
            ),
            suggested_chunk_threshold_secs=threshold,
            suggested_chunk_secs=chunk,
            suggested_chunk_overlap_secs=overlap,
            target_boundary_secs=chunk,
            expected_boundary_window_secs=(_scaled(0.0, scale), _scaled(26.0, scale)),
            segments=[
                _seg("tone", 0.0, 1.0, "phrase_1", frequency=330.0, scale=scale),
                _seg("silence", 1.0, 1.25, "comma_pause_1", scale=scale),
                _seg("tone", 1.25, 2.5, "phrase_2", frequency=349.23, scale=scale),
                _seg("silence", 2.5, 3.0, "sentence_pause_1", scale=scale),
                _seg("tone", 3.0, 4.1, "phrase_3", frequency=392.0, scale=scale),
                _seg("silence", 4.1, 4.35, "comma_pause_2", scale=scale),
                _seg("tone", 4.35, 5.8, "phrase_4", frequency=440.0, scale=scale),
                _seg("silence", 5.8, 6.4, "sentence_pause_2", scale=scale),
                _seg("tone", 6.4, 8.0, "phrase_5", frequency=493.88, scale=scale),
            ],
        ),
    ]


def _sample_for_segment(seg: SegmentSpec, absolute_index: int, rng: random.Random) -> int:
    if seg.kind == "silence":
        return 0
    if seg.kind == "noise":
        value = rng.uniform(-1.0, 1.0) * seg.amplitude
    else:
        t = absolute_index / SAMPLE_RATE
        value = math.sin(2.0 * math.pi * seg.frequency * t) * seg.amplitude
    return max(-MAX_I16, min(MAX_I16, int(value * MAX_I16)))


def write_wav(path: Path, segments: list[SegmentSpec]) -> None:
    """Write mono PCM16 WAV for ``segments``."""
    if not segments:
        raise ValueError("segments must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(max(s.end for s in segments) * SAMPLE_RATE))
    by_frame: list[SegmentSpec | None] = [None] * total_frames
    for seg in segments:
        start = int(round(seg.start * SAMPLE_RATE))
        end = int(round(seg.end * SAMPLE_RATE))
        for i in range(max(0, start), min(total_frames, end)):
            by_frame[i] = seg

    rng = random.Random(1337)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for i, seg in enumerate(by_frame):
            sample = 0 if seg is None else _sample_for_segment(seg, i, rng)
            frames.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(bytes(frames))


def write_manifest(path: Path, scenario: Scenario, wav_path: Path) -> None:
    payload = {
        "schemaVersion": "1",
        "kind": "voxkit.synthetic-audio-fixture",
        "sampleRate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sampleWidthBytes": SAMPLE_WIDTH_BYTES,
        "wav": wav_path.name,
        **asdict(scenario),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_all(out_dir: Path, *, scale: float = 1.0) -> list[Path]:
    """Generate every scenario and return written WAV paths."""
    written: list[Path] = []
    for scenario in build_scenarios(scale):
        wav_path = out_dir / f"{scenario.name}.wav"
        meta_path = out_dir / f"{scenario.name}.json"
        write_wav(wav_path, scenario.segments)
        write_manifest(meta_path, scenario, wav_path)
        written.append(wav_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic WAV fixtures for voxkit tests."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
        help="Directory for generated .wav/.json files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale all durations. Use 50 for ~600s-boundary manual fixtures.",
    )
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be > 0")

    written = build_all(args.out_dir, scale=args.scale)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
