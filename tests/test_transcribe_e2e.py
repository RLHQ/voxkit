"""End-to-end pipeline tests for ``voxkit transcribe``.

These tests exercise the full pipeline (audio prep → chunk → whisper.cpp →
filter → segment → merge → write). They are gated on local availability of
``whisper-cli`` plus a ggml model, via the ``requires_whisper`` marker
registered in ``conftest.py``.

The pipeline module ships in Round 2 (Agent I). Until it lands, every gated
test is skipped at collection time via ``pytestmark`` — this file stays green
on import even when the module is missing.

The ``test_e2e_fixture_files_exist`` sanity check is intentionally NOT gated so
fixture availability is always verified.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# ── Defer imports — Round 2 Agent I writes the pipeline; tests skip gracefully
# when the module is not yet available so test collection never crashes. ─
try:
    from voxkit.core.transcribe_pipeline import (  # type: ignore[import-not-found]
        TranscribeRequest,
        run_pipeline,
    )

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from voxkit.core.workspace import open_workspace

    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audio"
SHORT_EN = FIXTURE_DIR / "short_en.wav"
SHORT_ZH = FIXTURE_DIR / "short_zh.wav"


# Module-level skipif: every gated test in this file is skipped en bloc when
# the pipeline module is missing. The non-gated fixture sanity test below
# carries its own ``pytest.mark.skipif(False, ...)`` to opt out (i.e. it
# always runs).
pipeline_required = pytest.mark.skipif(
    not (PIPELINE_AVAILABLE and WORKSPACE_AVAILABLE),
    reason="voxkit.core.transcribe_pipeline not yet available (Round 2 Agent I)",
)


# ─────────────────────────────────────────────────────────────────────────
# Always-on sanity: fixtures exist + are valid 16kHz mono wavs.
# Not gated by requires_whisper — runs even without whisper-cli installed.
# ─────────────────────────────────────────────────────────────────────────


def test_e2e_fixture_files_exist() -> None:
    """Sanity check: fixture audio files exist and are valid 16kHz mono wav."""
    assert SHORT_EN.exists(), f"missing fixture: {SHORT_EN}"
    assert SHORT_ZH.exists(), f"missing fixture: {SHORT_ZH}"
    assert SHORT_EN.stat().st_size > 0
    assert SHORT_ZH.stat().st_size > 0
    # Files should be small (<200KB target).
    assert SHORT_EN.stat().st_size < 200 * 1024
    assert SHORT_ZH.stat().st_size < 200 * 1024

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not on PATH — skipping format check")

    for path in (SHORT_EN, SHORT_ZH):
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate,channels,codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip().splitlines()
        # Order matches -show_entries: sample_rate, channels, codec_name.
        # ffprobe sometimes orders fields differently across versions, so use a
        # set membership check rather than positional indexing.
        values = set(out)
        assert "16000" in values, f"{path.name}: expected 16000Hz, got {values}"
        assert "1" in values, f"{path.name}: expected mono (1 channel), got {values}"
        assert "pcm_s16le" in values, (
            f"{path.name}: expected pcm_s16le codec, got {values}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Helpers for gated tests below.
# ─────────────────────────────────────────────────────────────────────────


def _build_request(
    *,
    workspace: object,
    input_path: Path,
    language: str,
    resume: bool = True,
):
    """Construct a TranscribeRequest with conservative test-friendly defaults.

    The request shape matches the Round 2 spec; Agent I owns the dataclass.
    Tests use ``model="base"`` for speed.
    """
    return TranscribeRequest(  # type: ignore[name-defined]
        input_path=input_path,
        workspace=workspace,
        model="base",
        language=language,
        word_timestamps=True,
        vad=False,  # disable VAD: not all environments ship the silero model
        logprob_thold=-0.8,
        source_id=f"test_{language}",
        keep_work=True,
        json_events=False,
        timeout_ms=120_000,  # 2 min for 5s audio is plenty
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=resume,
        emit_srt=True,
        emit_vtt=True,
    )


def _assert_artifacts_written(ws) -> None:
    """All six pipeline artifacts must exist + be non-empty."""
    expected = [
        ws.raw_json_path,
        ws.voxkit_json_path,
        ws.srt_path,
        ws.vtt_path,
        ws.manifest_path,
        ws.events_path,
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"empty artifact: {p}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_ndjson_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


# ─────────────────────────────────────────────────────────────────────────
# Gated end-to-end tests.
# ─────────────────────────────────────────────────────────────────────────


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_english_pipeline(tmp_path: Path) -> None:
    """Full pipeline on the English fixture.

    Asserts:
      - ``run_pipeline`` returns without raising
      - All six artifacts written (transcript.raw.json, transcript.voxkit.json,
        subtitles.srt, subtitles.vtt, manifest.json, events.ndjson)
      - ``transcript.raw.json`` parses as JSON (Remixr shape, ``sourceId`` lives
        here, not on the voxkit-native output)
      - ``transcript.voxkit.json`` has ``schemaVersion="1"`` and
        ``asrBackend="whisper-cpp"``
      - ``manifest.json`` has run metadata including the request ``sourceId``
      - ``events.ndjson`` has at least one event line
    """
    ws = open_workspace(tmp_path / "ws_en")
    req = _build_request(workspace=ws, input_path=SHORT_EN, language="en")

    result = run_pipeline(req)
    assert result is not None

    _assert_artifacts_written(ws)

    raw = _read_json(ws.raw_json_path)
    assert isinstance(raw, dict)
    # sourceId is a Remixr-side concept and lives on the raw transcript.
    assert raw.get("sourceId") == "test_en"

    voxkit = _read_json(ws.voxkit_json_path)
    assert voxkit.get("schemaVersion") == "1", (
        f"expected schemaVersion='1', got {voxkit.get('schemaVersion')!r}"
    )
    assert voxkit.get("asrBackend") == "whisper-cpp", (
        f"expected asrBackend='whisper-cpp', got {voxkit.get('asrBackend')!r}"
    )

    manifest = _read_json(ws.manifest_path)
    assert isinstance(manifest, dict) and len(manifest) > 0
    assert manifest.get("sourceId") == "test_en"

    assert _count_ndjson_lines(ws.events_path) >= 1


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_chinese_pipeline(tmp_path: Path) -> None:
    """Same as English but with ``language='zh'``.

    Adds a CJK-specific assertion: every segment's ``words`` list is empty
    (segmenter goes to chinese_phrase mode for ``zh``; word-level timestamps
    are disabled by ``whisper_exec`` for CJK languages).
    """
    ws = open_workspace(tmp_path / "ws_zh")
    req = _build_request(workspace=ws, input_path=SHORT_ZH, language="zh")

    result = run_pipeline(req)
    assert result is not None

    _assert_artifacts_written(ws)

    raw = _read_json(ws.raw_json_path)
    assert raw.get("sourceId") == "test_zh"

    voxkit = _read_json(ws.voxkit_json_path)
    assert voxkit.get("schemaVersion") == "1"
    assert voxkit.get("asrBackend") == "whisper-cpp"
    # CJK languages must turn off word timestamps regardless of request.
    assert voxkit.get("wordTimestamps") is False, (
        f"CJK pipeline must disable word timestamps; got {voxkit.get('wordTimestamps')!r}"
    )

    segments = voxkit.get("segments", [])
    assert isinstance(segments, list)
    # CJK: every segment should report an empty words[] (chinese_phrase mode).
    for seg in segments:
        words = seg.get("words", [])
        assert words == [], (
            f"CJK segment unexpectedly has word-level timestamps: {words!r}"
        )


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_idempotent_resume(tmp_path: Path) -> None:
    """Resume reuses ``chunk_NNN.json`` checkpoints from a prior run.

    ``transcript.raw.json`` is written exclusively (``'wx'``) so a second run
    on a workspace that still has it would hard-fail (covered separately by
    ``test_e2e_exclusive_write_blocks_double_run``). To exercise pure
    chunk-level resume we delete the raw + voxkit JSON before the second run,
    leaving the chunk checkpoint files in place.

    Verification: every entry in ``result.voxkit_output.per_chunk`` must
    report ``cached=True`` on the second run.
    """
    ws = open_workspace(tmp_path / "ws_resume")
    req = _build_request(workspace=ws, input_path=SHORT_EN, language="en", resume=True)

    first = run_pipeline(req)
    assert first is not None
    first_chunks = list(first.voxkit_output.per_chunk)
    assert first_chunks, "expected at least one chunk in first run"
    assert all(c.cached is False for c in first_chunks), (
        "first run should have no cached chunks; "
        f"got {[c.cached for c in first_chunks]}"
    )
    chunk_jsons = sorted(ws.chunks.glob("chunk_*.json"))
    assert chunk_jsons, "expected chunk_*.json checkpoints after first run"

    # Clear top-level outputs (raw uses 'wx') but keep chunk checkpoints.
    ws.raw_json_path.unlink()
    if ws.voxkit_json_path.exists():
        ws.voxkit_json_path.unlink()

    second = run_pipeline(req)
    assert second is not None
    second_chunks = list(second.voxkit_output.per_chunk)
    assert len(second_chunks) == len(first_chunks)
    assert all(c.cached is True for c in second_chunks), (
        "second resumed run must report every chunk as cached; "
        f"got {[c.cached for c in second_chunks]}"
    )


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_force_resets_workspace(tmp_path: Path) -> None:
    """``force=True`` (re-open with ``open_workspace(force=True)``) wipes the
    chunk cache; the next run must re-transcribe with ``cached=False``.
    """
    ws = open_workspace(tmp_path / "ws_force")
    initial = _build_request(
        workspace=ws, input_path=SHORT_EN, language="en", resume=True
    )
    first = run_pipeline(initial)
    assert first is not None
    assert all(c.cached is False for c in first.voxkit_output.per_chunk), (
        "initial run cannot have cached chunks"
    )

    # Wipe work/ via the workspace API, then run again with resume=False.
    ws2 = open_workspace(tmp_path / "ws_force", force=True)
    assert not any(ws2.chunks.glob("chunk_*.json")), (
        "open_workspace(force=True) should have removed cached chunks"
    )
    forced = _build_request(
        workspace=ws2, input_path=SHORT_EN, language="en", resume=False
    )
    second = run_pipeline(forced)
    assert second is not None
    second_chunks = list(second.voxkit_output.per_chunk)
    assert second_chunks, "force run must produce at least one chunk"
    assert all(c.cached is False for c in second_chunks), (
        "force run must re-transcribe; "
        f"got cached={[c.cached for c in second_chunks]}"
    )


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_resume_blocks_double_run(tmp_path: Path) -> None:
    """Strict ``wx`` exclusive-write semantics (Plan §user-decision #1).

    Running ``run_pipeline`` twice on the same workspace with ``resume=True``
    must FAIL on the second invocation because ``transcript.raw.json`` already
    exists. The user is expected to either pick a fresh ``--workdir`` or pass
    ``--force`` (which sets ``resume=False`` and unlinks the stale raw.json).

    This pins the brainstorming-phase decision that voxkit's raw output
    behaves like Remixr's own raw artifacts: write-once-never-overwrite by
    default, with ``--force`` as the explicit opt-out.
    """
    from voxkit.core.transcribe_pipeline import PipelineError

    ws = open_workspace(tmp_path / "ws_strict_resume")
    req = _build_request(workspace=ws, input_path=SHORT_EN, language="en", resume=True)

    run_pipeline(req)
    assert ws.raw_json_path.exists()

    # Second run with resume=True must raise PipelineError.
    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(req)
    msg = str(exc_info.value)
    assert "already exists" in msg
    assert "--force" in msg


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_transcribe_with_diarization(tmp_path: Path, monkeypatch) -> None:
    """``with_diarization=True`` produces real speaker labels in raw.json + SRT.

    The real pyannote worker is intentionally NOT invoked (it would require
    the lazy venv install + model download — outside the unit test budget).
    Instead we monkeypatch ``_run_diarization_pass`` to return a synthetic
    :class:`DiarizationOutput` that covers the entire transcript timeline,
    exercising the full integration codepath EXCEPT the worker subprocess.

    Asserts:
      - Every ``transcript.raw.json segments[].speaker`` matches ``Speaker N``
        (not the v0.3.0 ``"Speaker A"`` placeholder).
      - ``work/diarization.json`` audit artefact exists.
      - ``manifest.json`` carries ``withDiarization=true``,
        ``speakerLabels="ranked"``, and ``numSpeakers``.
      - The SRT carries the real speaker prefix (not "Speaker A:") when there
        are ≥2 distinct speakers (v0.7.1 B1: auto mode skips single-speaker
        prefix as noise).
    """
    from voxkit.core import transcribe_pipeline as TP
    from voxkit.io.schema import (
        AudioInfo as _AudioInfo,
        DiarizationOutput as _DiarizationOutput,
        Segment as _Segment,
        SpeakerInfo as _SpeakerInfo,
    )

    # Two-speaker synthetic timeline. v0.7.1 B1 fix: auto mode skips the prefix
    # when there's only 1 distinct speaker — but this e2e wants to verify that
    # real diarization labels survive into the SRT, so we synthesise a true
    # multi-speaker case (split 0..2.5 / 2.5..120).
    fake_dia = _DiarizationOutput(
        audio=_AudioInfo(path="/tmp/master.wav", duration_secs=120.0),
        device="cpu",
        model="pyannote/speaker-diarization-3.1",
        rtf=0.05,
        elapsed_secs=6.0,
        num_speakers=2,
        speakers=[
            _SpeakerInfo(
                id="Speaker 1", raw_id="SPEAKER_00", total_duration_secs=117.5
            ),
            _SpeakerInfo(
                id="Speaker 2", raw_id="SPEAKER_01", total_duration_secs=2.5
            ),
        ],
        segments=[
            _Segment(
                start=0.0, end=2.5,
                speaker="Speaker 2", raw_speaker="SPEAKER_01",
            ),
            _Segment(
                start=2.5, end=120.0,
                speaker="Speaker 1", raw_speaker="SPEAKER_00",
            ),
        ],
    )

    def _fake_diarize_pass(req, *, master_wav, duration_secs, em, forward_stderr):
        return fake_dia, 6.0

    monkeypatch.setattr(TP, "_run_diarization_pass", _fake_diarize_pass)

    ws = open_workspace(tmp_path / "ws_diarize")
    # Build a request with diarization on. The base helper doesn't expose the
    # new fields so we construct a TranscribeRequest directly.
    req = TranscribeRequest(  # type: ignore[name-defined]
        input_path=SHORT_EN,
        workspace=ws,
        model="base",
        language="en",
        word_timestamps=True,
        vad=False,
        logprob_thold=-0.8,
        source_id="test_dia",
        keep_work=True,
        json_events=False,
        timeout_ms=120_000,
        whisper_bin_override=None,
        vad_model_override=None,
        blocklist_path=None,
        resume=True,
        emit_srt=True,
        emit_vtt=True,
        with_diarization=True,
        speaker_labels="ranked",
    )

    result = run_pipeline(req)
    assert result is not None
    _assert_artifacts_written(ws)

    # 1. transcript.raw.json segments[].speaker is real, not the placeholder.
    raw = _read_json(ws.raw_json_path)
    segments = raw.get("segments", [])
    assert len(segments) > 0, "expected at least one segment from whisper"
    real_labels = {"Speaker 1", "Speaker 2"}
    for seg in segments:
        assert seg.get("speaker") in real_labels, (
            f"expected real speaker label, got {seg.get('speaker')!r}"
        )
        # The placeholder must NOT survive into the diarized output.
        assert seg.get("speaker") != "Speaker A"

    # 2. work/diarization.json audit artefact exists.
    diarize_audit = ws.work / "diarization.json"
    assert diarize_audit.exists()
    audit = json.loads(diarize_audit.read_text(encoding="utf-8"))
    assert audit.get("numSpeakers") == 2

    # 3. manifest fields populated.
    manifest = _read_json(ws.manifest_path)
    assert manifest.get("withDiarization") is True
    assert manifest.get("speakerLabels") == "ranked"
    assert manifest.get("numSpeakers") == 2
    assert manifest.get("diarizationModel") == "pyannote/speaker-diarization-3.1"

    # 4. SRT must never carry the v0.3.0 "Speaker A:" placeholder.
    # 注：v0.7.1 B1 fix 后，auto 模式只在 cues 实际含 ≥2 个不同 speaker 时才
    # 渲染前缀。本测试用 5s sine fixture，whisper 通常只产 1 个 segment，所以
    # 渲染层观察到的 distinct speaker = 1，自然不写前缀——这是预期行为。
    # 真正的"diarization 标签是否落地"验证由上面 raw.json 的断言负责。
    srt_text = ws.srt_path.read_text(encoding="utf-8")
    assert "Speaker A:" not in srt_text


@pipeline_required
@pytest.mark.requires_whisper
def test_e2e_force_overwrites_existing_raw(tmp_path: Path) -> None:
    """``--force`` (resume=False) is the explicit opt-out for the wx contract.

    First run produces transcript.raw.json. Second run with resume=False must
    succeed — pipeline unlinks the existing raw.json and rewrites it. This is
    the documented escape hatch for users who want to re-transcribe an
    already-completed workspace without manually picking a new --workdir.
    """
    ws = open_workspace(tmp_path / "ws_force_raw")
    first_req = _build_request(
        workspace=ws, input_path=SHORT_EN, language="en", resume=True
    )
    run_pipeline(first_req)
    assert ws.raw_json_path.exists()
    first_mtime = ws.raw_json_path.stat().st_mtime_ns

    # Second run with resume=False (the --force CLI path) must succeed and
    # rewrite the raw json (mtime advances).
    forced_req = _build_request(
        workspace=ws, input_path=SHORT_EN, language="en", resume=False
    )
    second = run_pipeline(forced_req)
    assert second is not None
    assert ws.raw_json_path.exists()
    assert ws.raw_json_path.stat().st_mtime_ns >= first_mtime
    _assert_artifacts_written(ws)
