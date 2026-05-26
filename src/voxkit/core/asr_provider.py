"""ASR provider boundary primitives.

Current production transcription still runs through ``whisper_exec`` directly.
This module defines the provider/raw/normalize contract needed before adding
more ASR backends, without changing existing pipeline behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from voxkit.core.constants import CJK_LANGUAGES
from voxkit.core.segmenter import segment_entries
from voxkit.core.whisper_exec import parse_whisper_json
from voxkit.io.schema import TranscriptSegment

__all__ = [
    "TimestampMode",
    "ASRProviderRequest",
    "ASRProviderResult",
    "ProviderRawArtifact",
    "NormalizedASRTranscript",
    "ASRProvider",
    "provider_raw_path",
    "write_provider_raw",
    "resolve_timestamp_mode",
    "normalize_whisper_cpp_raw",
]


TimestampMode = Literal["word", "phrase", "char-interpolated"]


@dataclass(frozen=True)
class ASRProviderRequest:
    """Provider-neutral request metadata for a single ASR chunk."""

    provider: str
    model: str
    language: str
    word_timestamps: bool
    chunk_index: int
    audio_path: Path
    cost_policy: str | None = None


@dataclass(frozen=True)
class ProviderRawArtifact:
    """Auditable provider-native raw JSON written under ``work/asr``."""

    provider: str
    model: str
    chunk_index: int
    path: Path
    sha256: str
    timestamp_mode: TimestampMode


@dataclass(frozen=True)
class NormalizedASRTranscript:
    """Provider raw normalized into voxkit's stable transcript segment type."""

    provider: str
    model: str
    language: str
    timestamp_mode: TimestampMode
    segments: list[TranscriptSegment]
    raw_artifact: ProviderRawArtifact | None = None


@dataclass(frozen=True)
class ASRProviderResult:
    """Provider call result before downstream filtering/merging."""

    raw_json: dict[str, Any]
    raw_artifact: ProviderRawArtifact
    elapsed_secs: float


class ASRProvider(Protocol):
    """Minimal interface future ASR backends should implement."""

    provider: str
    model: str

    def transcribe_chunk(self, request: ASRProviderRequest) -> ASRProviderResult:
        """Transcribe one chunk and persist provider-native raw output."""


def provider_raw_path(workdir: Path, provider: str, chunk_index: int) -> Path:
    """Return ``work/asr/<provider>/chunk_NNN.raw.json`` for a provider chunk."""
    provider_dir = _safe_provider_dir(provider)
    return workdir / "asr" / provider_dir / f"chunk_{chunk_index:03d}.raw.json"


def write_provider_raw(
    workdir: Path,
    *,
    provider: str,
    model: str,
    language: str,
    word_timestamps: bool,
    chunk_index: int,
    raw_json: dict[str, Any],
    indent: int = 2,
) -> ProviderRawArtifact:
    """Persist provider-native raw JSON and return its audit metadata."""
    path = provider_raw_path(workdir, provider, chunk_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(raw_json, ensure_ascii=False, indent=indent) + "\n"
    path.write_text(text, encoding="utf-8")
    return ProviderRawArtifact(
        provider=provider,
        model=model,
        chunk_index=chunk_index,
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        timestamp_mode=resolve_timestamp_mode(language, word_timestamps),
    )


def resolve_timestamp_mode(language: str, word_timestamps: bool) -> TimestampMode:
    """Classify the timebase exposed by provider raw output.

    CJK ASR entries are phrase-level even when a caller asks for word timestamps.
    ``char-interpolated`` is reserved for subtitle-layer splitting and should
    not be reported as provider raw ASR timebase.
    """
    if language.lower() in CJK_LANGUAGES:
        return "phrase"
    return "word" if word_timestamps else "phrase"


def normalize_whisper_cpp_raw(
    raw_json: dict[str, Any],
    *,
    model: str,
    language: str,
    word_timestamps: bool,
    raw_artifact: ProviderRawArtifact | None = None,
) -> NormalizedASRTranscript:
    """Normalize whisper.cpp raw JSON into ``TranscriptSegment[]``.

    This is a thin adapter around the existing parser/segmenter so future
    provider work can compare against today's behavior before migrating the
    production pipeline.
    """
    entries = parse_whisper_json(raw_json)
    segments = segment_entries(entries, language=language)
    return NormalizedASRTranscript(
        provider="whisper-cpp",
        model=model,
        language=language,
        timestamp_mode=resolve_timestamp_mode(language, word_timestamps),
        segments=segments,
        raw_artifact=raw_artifact,
    )


def _safe_provider_dir(provider: str) -> str:
    value = provider.strip().lower()
    if not value:
        raise ValueError("provider must not be empty")
    return re.sub(r"[^a-z0-9_.-]+", "_", value).strip("_") or "provider"
