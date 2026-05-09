"""End-to-end smoke tests using generated synthetic audio fixtures.

The synthetic WAVs are deterministic tone/silence files, not real speech. To
validate subtitle segmentation deterministically, these tests run the real
pipeline orchestration and artifact writers while stubbing only the Whisper
subprocess output.
"""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from typing import Any

from tests.fixtures.audio.build_synthetic_audio import build_all
from voxkit.core.transcribe_pipeline import TranscribeRequest, run_pipeline
from voxkit.core.whisper_exec import WhisperRunResult
from voxkit.core.workspace import open_workspace


def _wav_duration_secs(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def test_synthetic_cjk_audio_e2e_writes_char_level_cues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Generated CJK cadence audio exercises the full render artifact chain."""
    generated = build_all(tmp_path / "generated_audio")
    audio_path = next(p for p in generated if p.name == "cjk_phrase_cadence.wav")

    import voxkit.core.transcribe_pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod,
        "find_whisper_cli",
        lambda override=None: Path("/fake/whisper-cli"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "find_whisper_model",
        lambda name: Path("/fake/ggml-base.bin"),
    )
    monkeypatch.setattr(pipeline_mod, "find_vad_model", lambda override=None: None)
    monkeypatch.setattr(pipeline_mod, "find_ffmpeg", lambda: "/fake/ffmpeg")

    def _fake_normalize(input_path: Path, out_wav: Path, *, ffmpeg_bin=None) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, out_wav)

    def _fake_extract(master_wav: Path, spec, *, ffmpeg_bin=None) -> None:
        spec.out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master_wav, spec.out_wav)

    raw_whisper: dict[str, Any] = {
        "result": {"language": "zh"},
        "params": {"language": "zh"},
        "transcription": [
            {
                "text": "第一句说完了。第二句继续讲！第三句也结束？",
                "offsets": {"from": 0, "to": 6000},
                "no_speech_prob": 0.01,
            }
        ],
    }

    def _fake_run_whisper(
        audio: Path,
        out_json: Path,
        flags,
        *,
        whisper_bin: Path,
        timeout_secs: float,
        env=None,
        progress_cb=None,
    ) -> WhisperRunResult:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(raw_whisper, ensure_ascii=False), encoding="utf-8")
        return WhisperRunResult(
            raw_json=raw_whisper,
            entries=[],
            elapsed_secs=0.25,
        )

    monkeypatch.setattr(pipeline_mod, "normalize_to_wav_16k_mono", _fake_normalize)
    monkeypatch.setattr(pipeline_mod, "extract_chunk", _fake_extract)
    monkeypatch.setattr(pipeline_mod, "probe_duration", _wav_duration_secs)
    monkeypatch.setattr(pipeline_mod, "run_whisper", _fake_run_whisper)
    monkeypatch.setattr(pipeline_mod.core_env, "patched_env", lambda extra=None: {})

    ws = open_workspace(tmp_path / "ws")
    req = TranscribeRequest(
        input_path=audio_path,
        workspace=ws,
        model="base",
        language="zh",
        word_timestamps=True,
        vad=False,
        logprob_thold=-0.8,
        source_id="synthetic_cjk",
        keep_work=True,
        json_events=False,
        timeout_ms=60_000,
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=True,
        emit_srt=True,
        emit_vtt=True,
        resegment="semantic",
    )

    result = run_pipeline(req)

    assert result.voxkit_output.language == "zh"
    assert result.voxkit_output.word_timestamps is False
    assert ws.raw_json_path.exists()
    assert ws.voxkit_json_path.exists()
    assert ws.srt_path.exists()
    assert ws.vtt_path.exists()
    assert ws.cues_json_path.exists()

    cues_payload = json.loads(ws.cues_json_path.read_text(encoding="utf-8"))
    cue_texts = [cue["text"] for cue in cues_payload["cues"]]
    assert cue_texts == [
        "第一句说完了。",
        "第二句继续讲！",
        "第三句也结束？",
    ]
    assert cues_payload["params"]["timebase"] == "char-interpolated"
    assert cues_payload["metrics"]["cueCount"] == 3
    assert cues_payload["metrics"]["flashCueRate"] == 0.0

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sourceId"] == "synthetic_cjk"
    assert manifest["subtitle"]["resegment"] == "semantic"
    assert manifest["subtitle"]["metrics"] == cues_payload["metrics"]
    assert manifest["artifacts"]["subtitle_cues_json"] == str(ws.cues_json_path)
