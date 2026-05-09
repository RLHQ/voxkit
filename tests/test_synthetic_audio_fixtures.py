"""Tests for synthetic audio fixture generation."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path


SCRIPT = Path(__file__).parent / "fixtures" / "audio" / "build_synthetic_audio.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("voxkit_synthetic_audio_builder", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_synthetic_audio_writes_wavs_and_manifests(tmp_path: Path) -> None:
    builder = _load_builder()

    written = builder.build_all(tmp_path)
    assert {p.name for p in written} == {
        "boundary_silence_near_target.wav",
        "boundary_no_silence.wav",
        "boundary_multi_candidate.wav",
        "cjk_phrase_cadence.wav",
    }

    for wav_path in written:
        meta_path = wav_path.with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["schemaVersion"] == "1"
        assert meta["kind"] == "voxkit.synthetic-audio-fixture"
        assert meta["sampleRate"] == 16000
        assert meta["channels"] == 1
        assert meta["suggested_chunk_secs"] == 12.0
        assert meta["suggested_chunk_overlap_secs"] == 2.0
        assert meta["segments"]

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0


def test_build_synthetic_audio_scale_changes_suggested_chunk_size(tmp_path: Path) -> None:
    builder = _load_builder()

    builder.build_all(tmp_path, scale=2.0)
    meta = json.loads(
        (tmp_path / "boundary_silence_near_target.json").read_text(encoding="utf-8")
    )

    assert meta["suggested_chunk_secs"] == 24.0
    assert meta["suggested_chunk_threshold_secs"] == 36.0
    assert meta["suggested_chunk_overlap_secs"] == 4.0
    assert meta["target_boundary_secs"] == 24.0
