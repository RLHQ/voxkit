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
        "raw_result_language": None,
        # F2: capture WhisperFlags passed to each run_whisper call so tests
        # can assert --prompt plumbing without parsing argv.
        "flags_history": [],
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
        state["flags_history"].append(flags)
        raw = dict(fake_raw)
        raw_result_language = state.get("raw_result_language")
        if raw_result_language is not None:
            raw["result"] = {"language": raw_result_language}
            raw["params"] = {"language": raw_result_language}

        # Persist the synthetic JSON to disk so resume sees a valid checkpoint.
        json_path = (
            Path(str(out_json))
            if str(out_json).endswith(".json")
            else Path(str(out_json) + ".json")
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )
        # Optionally emit a progress beat so the event stream gets that event.
        if progress_cb is not None:
            try:
                progress_cb(50)
            except Exception:
                pass
        return WhisperRunResult(
            raw_json=raw, entries=list(fake_entries), elapsed_secs=0.42
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

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    records = manifest["artifactRecords"]
    by_kind = {record["kind"]: record for record in records}
    assert {"raw_json", "voxkit_json", "events", "srt", "vtt"} <= set(by_kind)
    assert all(len(record["hash"]) == 64 for record in records)
    assert by_kind["raw_json"]["sourceArtifacts"] == ["voxkit_json"]
    assert by_kind["raw_json"]["sourceArtifactHashes"] == {
        "voxkit_json": by_kind["voxkit_json"]["hash"]
    }
    assert by_kind["srt"]["sourceArtifacts"] == ["raw_json"]
    assert by_kind["vtt"]["sourceArtifacts"] == ["raw_json"]
    assert by_kind["raw_json"]["path"] == "transcript.raw.json"

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


def test_srt_default_auto_skips_placeholder_prefix(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """v0.7.2 review #1：segment path（无 diarization、无 resegment）默认 auto，
    SRT 不应再含 'Speaker A:' 占位符前缀。"""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    srt = ws.srt_path.read_text(encoding="utf-8")
    assert "Speaker A:" not in srt
    assert "-->" in srt
    assert "Hello world." in srt


def test_srt_speaker_prefix_always_keeps_legacy_format(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """显式 ``--speaker-prefix always`` 恢复 v0.7.1 之前的 ``Speaker A:`` 前缀。"""
    from dataclasses import replace

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), speaker_prefix="always")
    run_pipeline(req)

    srt = ws.srt_path.read_text(encoding="utf-8")
    assert "Speaker A:" in srt


# ─────────────────────────────────────────────────────────────────────────
# VAD warm-up warning (B2 fix)
# ─────────────────────────────────────────────────────────────────────────


def test_vad_warmup_warning_helper_thresholds():
    """``_vad_warmup_warning`` 纯函数：覆盖触发与抑制条件。"""
    from voxkit.core.transcribe_pipeline import _vad_warmup_warning

    # 未开 VAD → 永不 warn
    assert _vad_warmup_warning(
        vad_effectively_on=False, first_segment_start=30.0, duration_secs=120.0
    ) is None

    # 无 segment → 不 warn
    assert _vad_warmup_warning(
        vad_effectively_on=True, first_segment_start=None, duration_secs=120.0
    ) is None

    # 开场就开口（< 阈值）→ 不 warn
    assert _vad_warmup_warning(
        vad_effectively_on=True, first_segment_start=5.0, duration_secs=120.0
    ) is None

    # 首条 cue 推迟 > 阈值且音频足够长 → warn
    warn = _vad_warmup_warning(
        vad_effectively_on=True, first_segment_start=30.0, duration_secs=120.0
    )
    assert warn is not None
    assert "--no-vad" in warn
    assert "30.0s" in warn
    # 文案不应做"VAD 一定吃了"的因果断言（我们无法证明）；只描述观察。
    assert "trimmed first" not in warn, "应避免无依据的因果断言"

    # 总时长 ≤ 首 segment 起点（坏数据兜底）→ 不 warn
    assert _vad_warmup_warning(
        vad_effectively_on=True, first_segment_start=30.0, duration_secs=10.0
    ) is None


def test_vad_warmup_warning_threshold_param():
    """自定义 threshold 应被尊重，便于上游按场景下调。"""
    from voxkit.core.transcribe_pipeline import _vad_warmup_warning

    assert _vad_warmup_warning(
        vad_effectively_on=True,
        first_segment_start=20.0,
        duration_secs=120.0,
        threshold_secs=30.0,
    ) is None
    assert _vad_warmup_warning(
        vad_effectively_on=True,
        first_segment_start=40.0,
        duration_secs=120.0,
        threshold_secs=30.0,
    ) is not None


# ─────────────────────────────────────────────────────────────────────────
# subtitles.cues.json — render-layer machine-readable output
# ─────────────────────────────────────────────────────────────────────────


def test_resegment_none_skips_cues_json(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Default ``resegment=none`` must not produce ``subtitles.cues.json`` —
    that file is the semantic-resegment artifact, not a generic export.
    """
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    assert req.resegment == "none"
    run_pipeline(req)
    assert not ws.cues_json_path.exists()
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["subtitle"]["metrics"]["cueCount"] == manifest["subtitle"]["cueCount"]
    assert manifest["subtitle"]["metrics"]["avgCueDurS"] > 0


def test_resegment_semantic_writes_cues_json(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """``resegment=semantic`` produces a parseable ``subtitles.cues.json`` whose
    cue count matches the manifest and whose sourceId matches the request.
    """
    pytest.importorskip("pysbd")

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic")
    run_pipeline(req)

    assert ws.cues_json_path.exists(), "subtitles.cues.json must be written"
    payload = json.loads(ws.cues_json_path.read_text(encoding="utf-8"))

    # Schema invariants (schemaVersion bumped to "2" — cue id required)
    assert payload["schemaVersion"] == "2"
    assert payload["sourceId"] == "fake_src"
    assert payload["resegment"] == "semantic"
    assert isinstance(payload["cues"], list)
    # Every cue carries a stable ``cue_NNNNNN`` id, 1-based and unique.
    cue_ids = [c["id"] for c in payload["cues"]]
    if cue_ids:
        assert cue_ids[0] == "cue_000001"
        assert len(set(cue_ids)) == len(cue_ids)
    # Params snapshot is present so downstream can audit how cues were sliced.
    assert "params" in payload and isinstance(payload["params"], dict)
    assert "max_dur_s" in payload["params"]

    # Cross-check against manifest.
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["subtitle"]["resegment"] == "semantic"
    assert manifest["subtitle"]["cueCount"] == len(payload["cues"])
    assert manifest["subtitle"]["metrics"] == payload["metrics"]
    assert payload["metrics"]["cueCount"] == len(payload["cues"])
    assert "subtitle_cues_json" in manifest["artifacts"]
    assert manifest["artifacts"]["subtitle_cues_json"] == str(ws.cues_json_path)
    records = {record["kind"]: record for record in manifest["artifactRecords"]}
    assert records["subtitle_cues_json"]["path"] == "subtitles.cues.json"
    assert records["subtitle_cues_json"]["sourceArtifacts"] == ["raw_json"]
    assert records["srt"]["sourceArtifacts"] == ["subtitle_cues_json"]
    assert records["vtt"]["sourceArtifacts"] == ["subtitle_cues_json"]
    assert records["srt"]["sourceArtifactHashes"] == {
        "subtitle_cues_json": records["subtitle_cues_json"]["hash"]
    }


def test_language_auto_resegment_uses_detected_language(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """``--language auto --resegment semantic`` must not pass ``auto`` to pysbd."""
    pytest.importorskip("pysbd")
    patched_pipeline["raw_result_language"] = "en"

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), language="auto", resegment="semantic")
    result = run_pipeline(req)

    assert result.voxkit_output.language == "en"
    assert ws.cues_json_path.exists()
    payload = json.loads(ws.cues_json_path.read_text(encoding="utf-8"))
    assert payload["resegment"] == "semantic"
    assert payload["params"]["timebase"] == "word"


def test_cjk_resegment_marks_char_interpolated_timebase(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """CJK semantic cues use estimated char timing, so the artifact must say so."""
    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), language="zh", resegment="semantic")
    run_pipeline(req)

    payload = json.loads(ws.cues_json_path.read_text(encoding="utf-8"))
    assert payload["resegment"] == "semantic"
    assert payload["params"]["timebase"] == "char-interpolated"
    assert payload["metrics"]["cueCount"] == len(payload["cues"])


def test_resegment_semantic_force_rerun_unlinks_cues_json(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """``--force`` (resume=False) must clear stale subtitles.cues.json so the
    next exclusive-create write does not collide with the previous artifact.
    """
    pytest.importorskip("pysbd")

    ws = open_workspace(tmp_path / "ws")
    first = replace(_make_request(ws, resume=True), resegment="semantic")
    run_pipeline(first)
    first_inode = ws.cues_json_path.stat().st_ino

    forced = replace(_make_request(ws, resume=False), resegment="semantic")
    run_pipeline(forced)
    assert ws.cues_json_path.exists()
    assert ws.cues_json_path.stat().st_ino != first_inode


# ─────────────────────────────────────────────────────────────────────────
# F3: --max-cue-duration 透传 + 超长 cue warning + 引导文案
# ─────────────────────────────────────────────────────────────────────────


def test_max_cue_duration_passes_through_to_resegment_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_pipeline: dict[str, Any]
) -> None:
    """req.max_cue_duration → ResegmentParams(max_dur_s=…)，
    透到 resegment_for_subtitles 的 params。"""
    pytest.importorskip("pysbd")

    captured: dict[str, Any] = {}
    import voxkit.core.semantic_resegment as sem_mod
    real_fn = sem_mod.resegment_for_subtitles

    def fake_reseg(segments, *, language=None, params=None):
        captured["max_dur_s"] = params.max_dur_s if params is not None else None
        return real_fn(segments, language=language, params=params)

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic", max_cue_duration=4.0)
    run_pipeline(req)
    assert captured["max_dur_s"] == pytest.approx(4.0)


def test_max_cue_duration_default_uses_dataclass_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_pipeline: dict[str, Any]
) -> None:
    """req.max_cue_duration=None → ResegmentParams() 默认 max_dur_s。"""
    pytest.importorskip("pysbd")
    from voxkit.core.semantic_resegment import ResegmentParams

    captured: dict[str, Any] = {}
    import voxkit.core.semantic_resegment as sem_mod
    real_fn = sem_mod.resegment_for_subtitles

    def fake_reseg(segments, *, language=None, params=None):
        captured["max_dur_s"] = params.max_dur_s if params is not None else None
        return real_fn(segments, language=language, params=params)

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic")
    assert req.max_cue_duration is None
    run_pipeline(req)
    assert captured["max_dur_s"] == pytest.approx(ResegmentParams().max_dur_s)


def test_pipeline_rejects_nonpositive_max_cue_duration(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Defense in depth: pipeline 层也 reject <=0。"""
    ws = open_workspace(tmp_path / "ws")
    for bad in (0.0, -1.5):
        sub_ws = open_workspace(tmp_path / f"ws_{bad}")
        req = replace(
            _make_request(sub_ws), resegment="semantic", max_cue_duration=bad
        )
        with pytest.raises(PipelineError) as exc_info:
            run_pipeline(req)
        assert "max_cue_duration" in str(exc_info.value)


def test_long_cue_warning_recorded_when_cue_exceeds_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_pipeline: dict[str, Any]
) -> None:
    """合成 cue 含 > max_dur_s × 1.5 的超长 cue → warnings 含
    'exceed soft duration limit' 文案 + voxkit_out.warnings 同步。"""
    pytest.importorskip("pysbd")

    from voxkit.core.semantic_resegment import SubtitleCue
    import voxkit.core.semantic_resegment as sem_mod

    def fake_reseg(segments, *, language=None, params=None):
        # 制造一条 12s cue（> 7 × 1.5 = 10.5s）
        return [
            SubtitleCue(start=0.0, end=12.0, speaker="Speaker A", text="long one"),
            SubtitleCue(start=12.0, end=14.0, speaker="Speaker A", text="ok"),
        ]

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic")
    result = run_pipeline(req)

    assert any(
        "exceed soft duration limit" in w for w in result.voxkit_output.warnings
    ), result.voxkit_output.warnings
    # manifest warnings 也同步
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert any(
        "exceed soft duration limit" in w for w in manifest["warnings"]
    )


def test_long_cue_warning_stderr_guidance_non_json_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    patched_pipeline: dict[str, Any],
) -> None:
    """普通模式（非 json_events）→ stderr 输出引导文案
    （proofread + reseg 双 pass workflow）。"""
    pytest.importorskip("pysbd")

    from voxkit.core.semantic_resegment import SubtitleCue
    import voxkit.core.semantic_resegment as sem_mod

    def fake_reseg(segments, *, language=None, params=None):
        return [SubtitleCue(start=0.0, end=12.0, speaker="Speaker A", text="x")]

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws, json_events=False), resegment="semantic")
    run_pipeline(req)

    err = capsys.readouterr().err
    assert "exceed soft duration limit" in err
    assert "voxkit proofread" in err
    assert "voxkit reseg" in err


def test_long_cue_json_events_emits_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    patched_pipeline: dict[str, Any],
) -> None:
    """json_events 模式 → 发 long_cues_detected NDJSON 事件 + events.ndjson 镜像。"""
    pytest.importorskip("pysbd")

    from voxkit.core.semantic_resegment import SubtitleCue
    import voxkit.core.semantic_resegment as sem_mod

    def fake_reseg(segments, *, language=None, params=None):
        return [
            SubtitleCue(start=0.0, end=12.0, speaker="Speaker A", text="x"),
            SubtitleCue(start=12.0, end=24.0, speaker="Speaker A", text="y"),
        ]

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws, json_events=True), resegment="semantic")
    run_pipeline(req)

    # events.ndjson 镜像必须有事件
    events = [
        json.loads(line)
        for line in ws.events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    long_events = [e for e in events if e.get("event") == "long_cues_detected"]
    assert len(long_events) == 1
    evt = long_events[0]
    assert evt["count"] == 2
    assert evt["longestSecs"] == pytest.approx(12.0)
    # threshold = ResegmentParams().max_dur_s * 1.5 = 7.0 * 1.5 = 10.5
    assert evt["thresholdSecs"] == pytest.approx(10.5)

    # stderr 也得有同样的 NDJSON 行（json_events 模式 forward）
    err_lines = [
        line for line in capsys.readouterr().err.splitlines() if line.strip()
    ]
    parsed = []
    for line in err_lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert any(p.get("event") == "long_cues_detected" for p in parsed)


def test_no_long_cue_warning_when_all_under_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_pipeline: dict[str, Any]
) -> None:
    """所有 cue dur <= threshold → 不产生 warning。"""
    pytest.importorskip("pysbd")

    from voxkit.core.semantic_resegment import SubtitleCue
    import voxkit.core.semantic_resegment as sem_mod

    def fake_reseg(segments, *, language=None, params=None):
        return [
            SubtitleCue(start=0.0, end=5.0, speaker="Speaker A", text="ok1"),
            SubtitleCue(start=5.0, end=9.0, speaker="Speaker A", text="ok2"),
        ]

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic")
    result = run_pipeline(req)
    assert not any(
        "exceed soft duration limit" in w for w in result.voxkit_output.warnings
    )


def test_long_cue_threshold_scales_with_user_max_cue_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_pipeline: dict[str, Any]
) -> None:
    """用户传 --max-cue-duration 5 → threshold = 5 × 1.5 = 7.5s。
    cue.dur=8s 触发 warning（默认 7×1.5=10.5 不触发）。"""
    pytest.importorskip("pysbd")

    from voxkit.core.semantic_resegment import SubtitleCue
    import voxkit.core.semantic_resegment as sem_mod

    def fake_reseg(segments, *, language=None, params=None):
        return [SubtitleCue(start=0.0, end=8.0, speaker="Speaker A", text="x")]

    monkeypatch.setattr(sem_mod, "resegment_for_subtitles", fake_reseg)

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic", max_cue_duration=5.0)
    result = run_pipeline(req)
    assert any(
        "exceed soft duration limit" in w for w in result.voxkit_output.warnings
    )
    assert any(
        "max_cue_duration=5.0" in w for w in result.voxkit_output.warnings
    )


def test_resegment_semantic_emits_write_event(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """events.ndjson records ``resegment.done`` and ``write.subtitle_cues``."""
    pytest.importorskip("pysbd")

    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), resegment="semantic")
    run_pipeline(req)

    events = [
        json.loads(line)
        for line in ws.events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    seen = {e.get("event") for e in events}
    assert "resegment.done" in seen
    assert "write.subtitle_cues" in seen
    write_evt = next(e for e in events if e.get("event") == "write.subtitle_cues")
    assert write_evt["path"] == str(ws.cues_json_path)
    assert isinstance(write_evt["cue_count"], int)


# ─────────────────────────────────────────────────────────────────────────
# F2: --initial-prompt 透传到 whisper-cli flags + manifest audit
# ─────────────────────────────────────────────────────────────────────────


def test_initial_prompt_none_by_default(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """No prompt set → WhisperFlags.initial_prompt is None; manifest reflects."""
    ws = open_workspace(tmp_path / "ws")
    req = _make_request(ws)
    run_pipeline(req)

    assert patched_pipeline["whisper_calls"] == 1
    flags = patched_pipeline["flags_history"][0]
    assert flags.initial_prompt is None

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["initialPromptUsed"] is False
    assert manifest["initialPromptChars"] == 0


def test_initial_prompt_plumbed_to_whisper_flags(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """req.initial_prompt → flags.initial_prompt verbatim; manifest records meta."""
    ws = open_workspace(tmp_path / "ws")
    prompt = "Claude, Anthropic, MCP, Sonnet, Opus, Haiku."
    req = replace(_make_request(ws), initial_prompt=prompt)
    run_pipeline(req)

    flags = patched_pipeline["flags_history"][0]
    assert flags.initial_prompt == prompt

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["initialPromptUsed"] is True
    assert manifest["initialPromptChars"] == len(prompt)
    # Body must NOT leak into the manifest (privacy + size).
    assert prompt not in ws.manifest_path.read_text(encoding="utf-8")


def test_initial_prompt_blank_normalised_to_none(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Whitespace-only prompt collapses to None (don't emit --prompt for noise)."""
    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), initial_prompt="   \n\t  ")
    run_pipeline(req)

    flags = patched_pipeline["flags_history"][0]
    assert flags.initial_prompt is None

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["initialPromptUsed"] is False
    assert manifest["initialPromptChars"] == 0


def test_initial_prompt_long_text_is_truncated_with_warning(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Prompts > 1000 chars are truncated (whisper-cli ~224 token limit) + warned.

    No hard failure — the user might paste a glossary that's too long; we want
    to keep going, not abort the whole transcribe.
    """
    ws = open_workspace(tmp_path / "ws")
    long_prompt = "Claude. " * 200  # 1600 chars
    req = replace(_make_request(ws), initial_prompt=long_prompt)
    result = run_pipeline(req)

    flags = patched_pipeline["flags_history"][0]
    assert flags.initial_prompt is not None
    assert len(flags.initial_prompt) == 1000

    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert manifest["initialPromptUsed"] is True
    assert manifest["initialPromptChars"] == 1000

    assert any(
        "initial_prompt" in w and "truncating" in w
        for w in result.voxkit_output.warnings
    ), result.voxkit_output.warnings


def test_initial_prompt_survives_resume_cache_miss(
    tmp_path: Path, patched_pipeline: dict[str, Any]
) -> None:
    """On a cold run (no chunk JSON cache), prompt reaches every chunk's WhisperFlags.

    Regression guard: chunk transcription is in a per-chunk loop; if the prompt
    pass-through is hoisted to the wrong scope, only chunk 0 gets it.
    """
    ws = open_workspace(tmp_path / "ws")
    req = replace(_make_request(ws), initial_prompt="Claude is a model.")
    run_pipeline(req)

    # 30s audio → single chunk in the fixture; assert that single call carries
    # the prompt. The structural plumb-through is the load-bearing claim.
    assert all(
        f.initial_prompt == "Claude is a model."
        for f in patched_pipeline["flags_history"]
    )


def test_initial_prompt_build_argv_emits_prompt_flag() -> None:
    """build_argv contract: --prompt only appears when initial_prompt is non-empty.

    This is an integration check against the actual whisper_exec module to make
    sure the pipeline's flags translate to whisper-cli's argv form.
    """
    from voxkit.core.whisper_exec import WhisperFlags, build_argv

    f_none = WhisperFlags(
        model_path=Path("/m.bin"),
        language="en",
        vad=False,
        vad_model_path=None,
        initial_prompt=None,
    )
    argv = build_argv(f_none, Path("/a.wav"), Path("/o.json"), whisper_bin=Path("/wc"))
    assert "--prompt" not in argv

    f_with = WhisperFlags(
        model_path=Path("/m.bin"),
        language="en",
        vad=False,
        vad_model_path=None,
        initial_prompt="Claude, Anthropic.",
    )
    argv = build_argv(f_with, Path("/a.wav"), Path("/o.json"), whisper_bin=Path("/wc"))
    assert "--prompt" in argv
    i = argv.index("--prompt")
    assert argv[i + 1] == "Claude, Anthropic."
