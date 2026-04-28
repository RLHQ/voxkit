"""Internal dataclass types shared across the transcribe pipeline.

These types stay close to whisper.cpp's `--output-json-full` shape so that
downstream modules (segmenter, hallucination_filter, asr_merge) can operate on
a single stable Python type without each re-parsing the raw JSON.

They are *internal* — not part of the on-disk schema contract. The on-disk
contract lives in `voxkit.io.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    """One row of whisper.cpp `transcription[]` after parsing.

    Fields stay close to whisper.cpp's `--output-json-full` shape so other
    modules (segmenter, hallucination_filter) operate on a stable type.

    Attributes:
        text: word/phrase text. Leading space is preserved (whisper.cpp emits
            English words with a leading space; CJK phrases without).
        t_from_ms: ``offsets.from`` from whisper.cpp output (milliseconds).
        t_to_ms: ``offsets.to`` from whisper.cpp output (milliseconds).
        no_speech_prob: per-entry no-speech probability, when available.
        confidence: per-entry confidence (avg logprob converted), when
            available.
        raw: passthrough of the original dict for debug / audit.
    """

    text: str
    t_from_ms: int
    t_to_ms: int
    no_speech_prob: float | None = None
    confidence: float | None = None
    raw: dict = field(default_factory=dict)


__all__ = ["Entry"]
