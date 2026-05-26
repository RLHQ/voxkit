"""Tests for ASR provider raw/normalize boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from voxkit.core.asr_provider import (
    normalize_whisper_cpp_raw,
    provider_raw_path,
    resolve_timestamp_mode,
    write_provider_raw,
)


def _english_raw() -> dict:
    return {
        "transcription": [
            {
                "text": " Hello",
                "offsets": {"from": 0, "to": 500},
                "no_speech_prob": 0.01,
            },
            {
                "text": " world.",
                "offsets": {"from": 500, "to": 1000},
                "no_speech_prob": 0.01,
            },
        ]
    }


def _cjk_raw() -> dict:
    return {
        "transcription": [
            {
                "text": "你好世界。",
                "offsets": {"from": 0, "to": 1200},
                "no_speech_prob": 0.02,
            }
        ]
    }


def test_provider_raw_path_sanitizes_provider_name(tmp_path: Path) -> None:
    path = provider_raw_path(tmp_path / "work", "vendor/asr v1", 3)
    assert path == tmp_path / "work" / "asr" / "vendor_asr_v1" / "chunk_003.raw.json"


def test_provider_raw_path_rejects_empty_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        provider_raw_path(tmp_path / "work", "  ", 0)


def test_write_provider_raw_persists_auditable_json(tmp_path: Path) -> None:
    work = tmp_path / "work"
    raw = _english_raw()
    artifact = write_provider_raw(
        work,
        provider="whisper-cpp",
        model="large-v3-turbo",
        language="en",
        word_timestamps=True,
        chunk_index=2,
        raw_json=raw,
    )

    assert artifact.path == work / "asr" / "whisper-cpp" / "chunk_002.raw.json"
    assert artifact.path.exists()
    assert artifact.provider == "whisper-cpp"
    assert artifact.model == "large-v3-turbo"
    assert artifact.chunk_index == 2
    assert artifact.timestamp_mode == "word"

    text = artifact.path.read_text(encoding="utf-8")
    assert json.loads(text) == raw
    assert artifact.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_resolve_timestamp_mode_keeps_asr_and_subtitle_timebases_distinct() -> None:
    assert resolve_timestamp_mode("en", True) == "word"
    assert resolve_timestamp_mode("en", False) == "phrase"
    assert resolve_timestamp_mode("zh", True) == "phrase"
    assert resolve_timestamp_mode("ja", True) == "phrase"


def test_normalize_whisper_cpp_raw_preserves_english_word_timestamps() -> None:
    normalized = normalize_whisper_cpp_raw(
        _english_raw(),
        model="large-v3-turbo",
        language="en",
        word_timestamps=True,
    )

    assert normalized.provider == "whisper-cpp"
    assert normalized.timestamp_mode == "word"
    assert len(normalized.segments) == 1
    segment = normalized.segments[0]
    assert segment.text == "Hello world."
    assert [(w.word, w.start, w.end) for w in segment.words] == [
        ("Hello", 0.0, 0.5),
        ("world.", 0.5, 1.0),
    ]


def test_normalize_whisper_cpp_raw_preserves_cjk_phrase_timebase(
    tmp_path: Path,
) -> None:
    artifact = write_provider_raw(
        tmp_path / "work",
        provider="whisper-cpp",
        model="large-v3-turbo",
        language="zh",
        word_timestamps=True,
        chunk_index=0,
        raw_json=_cjk_raw(),
    )
    normalized = normalize_whisper_cpp_raw(
        _cjk_raw(),
        model="large-v3-turbo",
        language="zh",
        word_timestamps=True,
        raw_artifact=artifact,
    )

    assert normalized.timestamp_mode == "phrase"
    assert normalized.raw_artifact == artifact
    assert len(normalized.segments) == 1
    assert normalized.segments[0].text == "你好世界。"
    assert normalized.segments[0].words == []
