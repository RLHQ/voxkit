"""Tests for :mod:`voxkit.core.transcribe_pipeline`.

These tests do NOT require ``whisper-cli`` or ffmpeg. They use heavy
``monkeypatch`` stubs to exercise the orchestration logic itself: discovery
fan-out, resume cache hit, artifact write order, NDJSON event flow, exclusive
write semantics for ``transcript.raw.json``.

Real-binary smoke is in ``tests/test_transcribe_e2e.py`` (Agent T2).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from voxkit.core.constants import ExitCode
from voxkit.core.transcribe_pipeline import (
    PipelineError,
    TranscribeRequest,
    TranscribeResult,
    run_pipeline,
)
from voxkit.core.types import Entry
from voxkit.core.whisper_exec import WhisperRunResult
from voxkit.core.workspace import open_workspace
from voxkit.io.schema import RemixrTranscript


# ─────────────────────────────────────────────────────────────────────────
# Dataclass / exception sanity
# ─────────────────────────────────────────────────────────────────────────


def test_transcribe_request_is_frozen(tmp_path: Path) -> None:
    """TranscribeRequest is a frozen dataclass — fields are typed and immutable."""
    ws = open_workspace(tmp_path / "ws")
    req = TranscribeRequest(
        input_path=tmp_path / "in.wav",
        workspace=ws,
        model="base",
        language="en",
        word_timestamps=True,
        vad=False,
        logprob_thold=-0.8,
        source_id="src",
        keep_work=True,
        json_events=False,
        timeout_ms=None,
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=True,
        emit_srt=True,
        emit_vtt=True,
    )
    assert req.model == "base"
    assert req.workspace is ws
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        req.model = "large"  # type: ignore[misc]


def test_transcribe_result_shape() -> None:
    """TranscribeResult exposes the four documented public fields."""
    fields = TranscribeResult.__dataclass_fields__  # type: ignore[attr-defined]
    assert {"voxkit_output", "artifacts", "warnings", "elapsed_secs", "rtf"} <= set(
        fields.keys()
    )


def test_pipeline_error_carries_exit_code() -> None:
    """PipelineError exposes ``exit_code`` and renders message via ``str()``."""
    err = PipelineError("boom", exit_code=int(ExitCode.ENV_PROBLEM))
    assert str(err) == "boom"
    assert err.exit_code == int(ExitCode.ENV_PROBLEM)
    # Default exit code = generic fail
    assert PipelineError("oops").exit_code == int(ExitCode.GENERIC_FAIL)


# ─────────────────────────────────────────────────────────────────────────
# Mock fixtures: patch out every external touch (ffmpeg, whisper, ffprobe).
# ─────────────────────────────────────────────────────────────────────────


def _stub_entries() -> list[Entry]:
    """A short, deterministic list of English-style word-mode entries."""
    return [
        Entry(text=" Hello", t_from_ms=0, t_to_ms=400),
        Entry(text=" world.", t_from_ms=400, t_to_ms=900),
        Entry(text=" This", t_from_ms=1200, t_to_ms=1500),
        Entry(text=" is", t_from_ms=1500, t_to_ms=1700),
        Entry(text=" voxkit.", t_from_ms=1700, t_to_ms=2300),
    ]


def _stub_raw_json(entries: list[Entry]) -> dict:
    """Wrap entries back into the whisper.cpp -ojf transcription[] shape."""
    return {
        "transcription": [
            {
                "text": e.text,
                "offsets": {"from": e.t_from_ms, "to": e.t_to_ms},
                "no_speech_prob": e.no_speech_prob,
            }
            for e in entries
        ]
    }


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install monkey-patches across every external dependency of the pipeline.

    Returns a dict the test can use to introspect call counts / inject custom
    behaviour for individual pipeline runs.
    """
    state: dict[str, Any] = {
        "whisper_calls": 0,
        "extract_calls": 0,
        "normalize_calls": 0,
        "vad_present": False,
    }
    fake_entries = _stub_entries()
    fake_raw = _stub_raw_json(fake_entries)

    # Discovery layer
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
    monkeypatch.setattr(
        pipeline_mod,
        "find_vad_model",
        lambda override=None: (
            Path("/fake/ggml-silero.bin") if state["vad_present"] else None
        ),
    )

    # Audio layer
    def _fake_normalize(input_path: Path, out_wav: Path, *, ffmpeg_bin=None) -> None:
        state["normalize_calls"] += 1
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(b"\0")  # 1-byte sentinel so existence check passes

    def _fake_probe(path: Path) -> float:
        return 30.0  # single-chunk path (well below 900s threshold)

    def _fake_extract(master_wav: Path, spec, *, ffmpeg_bin=None) -> None:
        state["extract_calls"] += 1
        spec.out_wav.parent.mkdir(parents=True, exist_ok=True)
        spec.out_wav.write_bytes(b"\0")

    def _fake_find_ffmpeg() -> str:
        return "/fake/ffmpeg"

    monkeypatch.setattr(pipeline_mod, "normalize_to_wav_16k_mono", _fake_normalize)
    monkeypatch.setattr(pipeline_mod, "probe_duration", _fake_probe)
    monkeypatch.setattr(pipeline_mod, "extract_chunk", _fake_extract)
    monkeypatch.setattr(pipeline_mod, "find_ffmpeg", _fake_find_ffmpeg)

    # Whisper layer
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
        state["whisper_calls"] += 1
        # Persist the synthetic JSON to disk so resume sees a valid checkpoint.
        json_path = (
            Path(str(out_json))
            if str(out_json).endswith(".json")
            else Path(str(out_json) + ".json")
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(fake_raw, ensure_ascii=False), encoding="utf-8"
        )
        # Optionally emit a progress beat so the event stream gets that event.
        if progress_cb is not None:
            try:
                progress_cb(50)
            except Exception:
                pass
        return WhisperRunResult(
            raw_json=fake_raw, entries=list(fake_entries), elapsed_secs=0.42
        )

    monkeypatch.setattr(pipeline_mod, "run_whisper", _fake_run_whisper)

    # core_env.patched_env() touches DYLD_LIBRARY_PATH; harmless but pin it.
    monkeypatch.setattr(pipeline_mod.core_env, "patched_env", lambda extra=None: {})

    return state


def _make_request(ws, *, resume: bool = True, json_events: bool = False) -> TranscribeRequest:
    return TranscribeRequest(
        input_path=Path("/tmp/fake-input.wav"),
        workspace=ws,
        model="base",
        language="en",
        word_timestamps=True,
        vad=False,
        logprob_thold=-0.8,
        source_id="fake_src",
        keep_work=True,
        json_events=json_events,
        timeout_ms=60_000,
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=resume,
        emit_srt=True,
        emit_vtt=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Discovery: missing whisper-cli / model raise PipelineError
# ─────────────────────────────────────────────────────────────────────────


def test_missing_whisper_cli_is_env_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxkit.core.transcribe_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "find_whisper_cli", lambda override=None: None)
    monkeypatch.setattr(
        pipeline_mod,
        "find_whisper_model",
        lambda name: Path("/fake/ggml-base.bin"),
    )

    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    with pytest.raises(PipelineError) as excinfo:
        run_pipeline(req)
    assert excinfo.value.exit_code == int(ExitCode.ENV_PROBLEM)
    assert "whisper-cli not found" in str(excinfo.value)


def test_missing_model_is_env_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voxkit.core.transcribe_pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "find_whisper_cli", lambda override=None: Path("/fake/wcli")
    )
    monkeypatch.setattr(pipeline_mod, "find_whisper_model", lambda name: None)

    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    with pytest.raises(PipelineError) as excinfo:
        run_pipeline(req)
    assert excinfo.value.exit_code == int(ExitCode.ENV_PROBLEM)
    assert "whisper model not found" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end mock pipeline
# ─────────────────────────────────────────────────────────────────────────


def test_run_pipeline_writes_all_artifacts(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Mock pipeline run produces every documented artifact."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    result = run_pipeline(req)

    assert isinstance(result, TranscribeResult)
    assert ws.raw_json_path.exists()
    assert ws.voxkit_json_path.exists()
    assert ws.srt_path.exists()
    assert ws.vtt_path.exists()
    assert ws.manifest_path.exists()
    assert ws.events_path.exists()

    # Pipeline ran whisper exactly once (single chunk for 30s audio).
    assert patched_pipeline["whisper_calls"] == 1
    assert patched_pipeline["normalize_calls"] == 1


def test_remixr_round_trips(tmp_path: Path, patched_pipeline: dict[str, Any]) -> None:
    """transcript.raw.json must validate against the Pydantic Remixr schema."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    raw_text = ws.raw_json_path.read_text(encoding="utf-8")
    raw_obj = json.loads(raw_text)
    # _metadata is a voxkit-side audit field; strip before strict validation.
    raw_obj.pop("_metadata", None)
    parsed = RemixrTranscript.model_validate(raw_obj)
    assert parsed.sourceId == "fake_src"
    assert all(seg.id.startswith("seg_") for seg in parsed.segments)
    assert all(seg.speaker == "Speaker A" for seg in parsed.segments)


def test_voxkit_json_uses_camel_case(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """transcript.voxkit.json renders camelCase keys via ``by_alias``."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    payload = json.loads(ws.voxkit_json_path.read_text(encoding="utf-8"))
    # camelCase aliased fields:
    for key in (
        "schemaVersion",
        "asrBackend",
        "asrModel",
        "wordTimestamps",
        "elapsedSecs",
        "perChunk",
        "hallucinationDrops",
    ):
        assert key in payload, f"missing camelCase key {key!r} in transcript.voxkit.json"
    assert payload["schemaVersion"] == "1"
    assert payload["asrBackend"] == "whisper-cpp"
    # sourceId is attached as a sidecar key — not part of the Pydantic schema
    # but still required for downstream symmetry.
    assert payload["sourceId"] == "fake_src"


def test_events_ndjson_is_well_formed(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """events.ndjson has ≥5 lines and every line parses as JSON."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    lines = [
        line
        for line in ws.events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(lines) >= 5, f"expected ≥5 events, got {len(lines)}: {lines}"
    parsed = [json.loads(line) for line in lines]
    events = {p.get("event") for p in parsed}
    # Spot-check a sampling of mandatory events.
    for required in ("start", "discover", "plan", "chunk.done", "merge.done", "done"):
        assert required in events, (
            f"events.ndjson missing required event {required!r}; saw {events}"
        )


def test_json_events_forwards_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patched_pipeline: dict[str, Any],
) -> None:
    """``json_events=True`` mirrors every emit to stderr as one JSON / line."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws, json_events=True)
    run_pipeline(req)

    captured = capsys.readouterr()
    err_lines = [line for line in captured.err.splitlines() if line.strip()]
    # Each line must be JSON.
    parsed = [json.loads(line) for line in err_lines]
    assert any(p.get("event") == "done" for p in parsed)


def test_vad_missing_emits_warning(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """VAD requested but model absent → warning recorded in TranscriptionOutput."""
    ws = open_workspace(tmp_path / "ws")
    # vad_present remains False per fixture default
    req = TranscribeRequest(
        input_path=Path("/tmp/fake.wav"),
        workspace=ws,
        model="base",
        language="en",
        word_timestamps=True,
        vad=True,  # request VAD ...
        logprob_thold=-0.8,
        source_id="src",
        keep_work=True,
        json_events=False,
        timeout_ms=60_000,
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=True,
        emit_srt=False,
        emit_vtt=False,
    )
    result = run_pipeline(req)
    assert any(
        "VAD model not found" in w for w in result.voxkit_output.warnings
    ), result.voxkit_output.warnings


# ─────────────────────────────────────────────────────────────────────────
# Resume cache hit: pre-existing chunk_NNN.json is reused
# ─────────────────────────────────────────────────────────────────────────


def test_resume_skips_whisper_when_checkpoint_present(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Pre-seed a chunk_000.json + resume=True → run_whisper is never called."""
    ws = open_workspace(tmp_path / "ws")
    # Pre-seed chunk_000.json with synthetic content so resume detects it.
    seeded = ws.chunks / "chunk_000.json"
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(
        json.dumps(_stub_raw_json(_stub_entries()), ensure_ascii=False),
        encoding="utf-8",
    )
    # Pre-seed master wav so prepare doesn't re-normalize.
    ws.master_wav.parent.mkdir(parents=True, exist_ok=True)
    ws.master_wav.write_bytes(b"\0")

    req = _make_request(ws, resume=True)
    result = run_pipeline(req)

    assert patched_pipeline["whisper_calls"] == 0, (
        "whisper-cli should NOT be called when chunk JSON is cached"
    )
    # Cached chunk reflected in perChunk stat.
    chunk_stats = result.voxkit_output.per_chunk
    assert len(chunk_stats) == 1
    assert chunk_stats[0].cached is True


def test_resume_handles_invalid_utf8_in_checkpoint(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Regression: chunk_NNN.json may contain invalid UTF-8 bytes inside
    ``tokens[].text`` for CJK audio (whisper.cpp BPE token boundary cuts a
    multi-byte char in half). Strict utf-8 decode would fall through to a
    cache miss and re-run whisper — wasting work and breaking idempotence.
    Resume must read with ``errors="replace"`` and treat the checkpoint as
    valid since segment-level text is well-formed."""
    ws = open_workspace(tmp_path / "ws")
    seeded = ws.chunks / "chunk_000.json"
    seeded.parent.mkdir(parents=True, exist_ok=True)
    # Hand-crafted bytes mirror real whisper.cpp -ojf output for CJK audio:
    # segment.text 合法（"去做"），tokens[].text 含 \xe9\x80（被切到字节中间）。
    bad_bytes = (
        b'{\n'
        b'  "transcription": [\n'
        b'    {\n'
        b'      "text": "\xe5\x8e\xbb\xe5\x81\x9a",\n'
        b'      "offsets": {"from": 0, "to": 200},\n'
        b'      "no_speech_prob": 0.01,\n'
        b'      "tokens": [\n'
        b'        {"text": "\xe9\x80", "offsets": {"from": 0, "to": 100}}\n'
        b'      ]\n'
        b'    }\n'
        b'  ]\n'
        b'}\n'
    )
    seeded.write_bytes(bad_bytes)
    ws.master_wav.parent.mkdir(parents=True, exist_ok=True)
    ws.master_wav.write_bytes(b"\0")

    req = _make_request(ws, resume=True)
    result = run_pipeline(req)

    # Cache hit: whisper-cli must NOT have been invoked.
    assert patched_pipeline["whisper_calls"] == 0
    chunk_stats = result.voxkit_output.per_chunk
    assert len(chunk_stats) == 1
    assert chunk_stats[0].cached is True
    # Guard against silent-fail: cache hit must yield real segment text from
    # the seeded JSON, not an empty/garbled result.
    assert any(
        "去做" in seg.text for seg in result.voxkit_output.segments
    ), "seeded segment text 去做 should reach voxkit output via cache"


def test_force_invalidates_checkpoint(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """resume=False forces whisper to run even when chunk_000.json exists."""
    ws = open_workspace(tmp_path / "ws")
    seeded = ws.chunks / "chunk_000.json"
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(json.dumps({"transcription": []}), encoding="utf-8")

    req = _make_request(ws, resume=False)
    run_pipeline(req)
    assert patched_pipeline["whisper_calls"] == 1


# ─────────────────────────────────────────────────────────────────────────
# Idempotence: a second run on the same workspace succeeds with cached chunks
# ─────────────────────────────────────────────────────────────────────────


def test_second_run_blocks_on_resume_strict_wx(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Strict wx semantics (Plan §user-decision #1): resume + raw.json exists
    must FAIL with PipelineError. The user is expected to pass --force or
    pick a fresh --workdir.
    """
    from voxkit.core.transcribe_pipeline import PipelineError

    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws, resume=True)

    run_pipeline(req)
    assert ws.raw_json_path.exists()

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(req)
    assert "already exists" in str(exc_info.value)
    assert "--force" in str(exc_info.value)


def test_force_overwrites_existing_raw_json(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """--force (resume=False) is the explicit opt-out: pipeline unlinks the
    stale raw.json and rewrites cleanly.
    """
    ws = open_workspace(tmp_path / "ws")
    first_req = _make_request(ws, resume=True)
    run_pipeline(first_req)
    first_inode = ws.raw_json_path.stat().st_ino

    forced_req = _make_request(ws, resume=False)
    run_pipeline(forced_req)
    assert ws.raw_json_path.exists()
    # File was rewritten (different inode after unlink+create).
    assert ws.raw_json_path.stat().st_ino != first_inode


# ─────────────────────────────────────────────────────────────────────────
# keep_work cleanup
# ─────────────────────────────────────────────────────────────────────────


def test_no_keep_work_removes_work_dir_on_success(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), keep_work=False)
    run_pipeline(req)
    assert not ws.work.exists(), "work/ must be removed when keep_work=False"
    # But user-facing artifacts still exist.
    assert ws.raw_json_path.exists()
    assert ws.voxkit_json_path.exists()


# ─────────────────────────────────────────────────────────────────────────
# Speaker prefix in subtitles
# ─────────────────────────────────────────────────────────────────────────


def test_srt_contains_speaker_prefix(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """SRT renders the Remixr-aligned ``Speaker A:`` prefix."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    srt = ws.srt_path.read_text(encoding="utf-8")
    assert "Speaker A:" in srt
    assert "-->" in srt
