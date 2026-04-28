"""Unit tests for ``voxkit.core.diarize_runner``.

Mocks ``subprocess.run`` so we never spawn the real pyannote worker. Covers:

* argv assembly (``build_worker_argv``)
* sentinel JSON extraction
* happy path → returns parsed ``DiarizationOutput``
* non-zero exit → ``DiarizeFailed``
* timeout → ``DiarizeTimeout``
* missing sentinel → ``ValueError``
* invalid sentinel JSON → ``ValueError``
* progress callback fires on stderr ``[stage] N%`` lines
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voxkit.core import diarize_runner as DR
from voxkit.core.constants import WORKER_JSON_SENTINEL
from voxkit.io.progress import ProgressEmitter
from voxkit.io.schema import (
    AudioInfo,
    DiarizationOutput,
    Segment,
    SpeakerInfo,
)


def _sample_diarization_payload() -> dict:
    """Minimum valid DiarizationOutput JSON shape (camelCase, by_alias)."""
    return DiarizationOutput(
        audio=AudioInfo(path="/tmp/audio.wav", duration_secs=10.0),
        device="cpu",
        model="pyannote/speaker-diarization-3.1",
        rtf=0.42,
        elapsed_secs=4.2,
        num_speakers=2,
        speakers=[
            SpeakerInfo(id="Speaker 1", raw_id="SPEAKER_00", total_duration_secs=6.0),
            SpeakerInfo(id="Speaker 2", raw_id="SPEAKER_01", total_duration_secs=4.0),
        ],
        segments=[
            Segment(start=0.0, end=6.0, speaker="Speaker 1", raw_speaker="SPEAKER_00"),
            Segment(start=6.0, end=10.0, speaker="Speaker 2", raw_speaker="SPEAKER_01"),
        ],
    ).model_dump(by_alias=True)


# ── build_worker_argv ─────────────────────────────────────────────────────


def test_build_worker_argv_minimal():
    argv = DR.build_worker_argv(
        venv_python=Path("/tmp/venv/bin/python"),
        audio_path=Path("/tmp/a.wav"),
        duration_secs=12.0,
    )
    # First arg is the venv python; second/third are -m voxkit.core.pipeline.
    assert argv[0] == "/tmp/venv/bin/python"
    assert argv[1:3] == ["-m", "voxkit.core.pipeline"]
    assert "--audio" in argv and "/tmp/a.wav" in argv
    assert "--audio-duration-secs" in argv and "12.000000" in argv
    assert "--model" in argv and "sd-3.1" in argv
    assert "--device" in argv and "auto" in argv
    assert "--speaker-labels" in argv and "ranked" in argv
    # Optional flags absent when not set
    assert "--num-speakers" not in argv
    assert "--min-speakers" not in argv
    assert "--max-speakers" not in argv
    assert "--extracted-from" not in argv
    assert "--json-events" not in argv


def test_build_worker_argv_with_optionals():
    argv = DR.build_worker_argv(
        venv_python=Path("/v/p"),
        audio_path=Path("/x.wav"),
        duration_secs=7.5,
        num_speakers=3,
        min_speakers=2,
        max_speakers=4,
        extracted_from=Path("/orig.mp4"),
        json_events=True,
        speaker_labels="raw",
        model="community-1",
        device="mps",
    )
    assert "--num-speakers" in argv and "3" in argv
    assert "--min-speakers" in argv and "2" in argv
    assert "--max-speakers" in argv and "4" in argv
    assert "--extracted-from" in argv and "/orig.mp4" in argv
    assert "--json-events" in argv
    assert "--speaker-labels" in argv and "raw" in argv
    assert "--model" in argv and "community-1" in argv
    assert "--device" in argv and "mps" in argv


# ── extract_sentinel_json ─────────────────────────────────────────────────


def test_extract_sentinel_json_finds_payload():
    stdout = (
        "torch noise line\n"
        "another noise\n"
        f"{WORKER_JSON_SENTINEL}{{\"foo\": 1}}\n"
        "trailing line\n"
    )
    payload = DR.extract_sentinel_json(stdout)
    assert payload == '{"foo": 1}'


def test_extract_sentinel_json_returns_none_when_missing():
    stdout = "no sentinel here\nstill nothing\n"
    assert DR.extract_sentinel_json(stdout) is None


def test_extract_sentinel_json_picks_first_match():
    stdout = (
        f"{WORKER_JSON_SENTINEL}first\n"
        f"{WORKER_JSON_SENTINEL}second\n"
    )
    assert DR.extract_sentinel_json(stdout) == "first"


# ── run_diarize: happy path ───────────────────────────────────────────────


def test_run_diarize_happy_path(monkeypatch, tmp_path):
    """Successful run returns parsed DiarizationOutput."""
    payload = _sample_diarization_payload()
    sentinel_line = WORKER_JSON_SENTINEL + json.dumps(payload)
    fake_stdout = "boot\n" + sentinel_line + "\n"
    fake_stderr = ""

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=fake_stdout,
        stderr=fake_stderr,
    )
    captured: dict = {}

    def _fake_run(argv, *, env=None, capture_output=True, text=True, timeout=None):
        captured["argv"] = argv
        captured["env"] = env
        captured["timeout"] = timeout
        return completed

    monkeypatch.setattr(DR.subprocess, "run", _fake_run)

    result = DR.run_diarize(
        Path("/tmp/audio.wav"),
        duration_secs=10.0,
        venv_python=tmp_path / "py",
        forward_stderr=False,
    )

    assert isinstance(result, DiarizationOutput)
    assert result.num_speakers == 2
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "Speaker 1"
    # argv smoke-check: contains the audio path
    assert str(Path("/tmp/audio.wav")) in captured["argv"]


# ── run_diarize: failure paths ────────────────────────────────────────────


def test_run_diarize_nonzero_exit_raises_DiarizeFailed(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=4,
        stdout="",
        stderr="oh no\nsomething went wrong\n",
    )
    monkeypatch.setattr(
        DR.subprocess, "run",
        lambda *a, **kw: completed,
    )
    with pytest.raises(DR.DiarizeFailed) as exc_info:
        DR.run_diarize(
            Path("/tmp/x.wav"),
            duration_secs=1.0,
            venv_python=tmp_path / "py",
            forward_stderr=False,
        )
    err = exc_info.value
    assert err.returncode == 4
    assert "something went wrong" in err.stderr_tail


def test_run_diarize_timeout_raises_DiarizeTimeout(monkeypatch, tmp_path):
    def _raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=0.5)

    monkeypatch.setattr(DR.subprocess, "run", _raise_timeout)
    with pytest.raises(DR.DiarizeTimeout):
        DR.run_diarize(
            Path("/tmp/x.wav"),
            duration_secs=1.0,
            venv_python=tmp_path / "py",
            forward_stderr=False,
            timeout_secs=0.5,
        )


def test_run_diarize_missing_sentinel_raises_ValueError(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="lots of noise\nbut no sentinel\n",
        stderr="",
    )
    monkeypatch.setattr(DR.subprocess, "run", lambda *a, **kw: completed)
    with pytest.raises(ValueError, match="sentinel"):
        DR.run_diarize(
            Path("/tmp/x.wav"),
            duration_secs=1.0,
            venv_python=tmp_path / "py",
            forward_stderr=False,
        )


def test_run_diarize_invalid_sentinel_json_raises_ValueError(monkeypatch, tmp_path):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=WORKER_JSON_SENTINEL + "not valid json {{\n",
        stderr="",
    )
    monkeypatch.setattr(DR.subprocess, "run", lambda *a, **kw: completed)
    with pytest.raises(ValueError, match="invalid sentinel JSON"):
        DR.run_diarize(
            Path("/tmp/x.wav"),
            duration_secs=1.0,
            venv_python=tmp_path / "py",
            forward_stderr=False,
        )


# ── progress callbacks via stderr ─────────────────────────────────────────


def test_progress_callback_fires_on_stage_pct_lines(monkeypatch, tmp_path):
    """Stderr lines like ``[diarize] 50%`` invoke ``progress.progress``."""
    payload = _sample_diarization_payload()
    sentinel_line = WORKER_JSON_SENTINEL + json.dumps(payload)
    fake_stderr = (
        "[model_load] 0%\n"
        "[model_load] 100%\n"
        "[diarize] 0%\n"
        "[diarize] 50%\n"
        "[diarize] 100%\n"
        "device=cpu model=...\n"
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=sentinel_line + "\n",
        stderr=fake_stderr,
    )
    monkeypatch.setattr(DR.subprocess, "run", lambda *a, **kw: completed)

    emitter = MagicMock(spec=ProgressEmitter)

    DR.run_diarize(
        Path("/tmp/x.wav"),
        duration_secs=1.0,
        venv_python=tmp_path / "py",
        progress=emitter,
        forward_stderr=False,
    )

    # All five [stage] N% lines should have fired the progress callback.
    assert emitter.progress.call_count == 5
    # Spot-check the diarize 50% call
    emitter.progress.assert_any_call("diarize", 50)
    emitter.progress.assert_any_call("model_load", 100)


def test_no_progress_callback_when_progress_is_none(monkeypatch, tmp_path):
    """progress=None must not blow up even with parseable stderr lines."""
    payload = _sample_diarization_payload()
    sentinel_line = WORKER_JSON_SENTINEL + json.dumps(payload)
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=sentinel_line + "\n",
        stderr="[diarize] 99%\n",
    )
    monkeypatch.setattr(DR.subprocess, "run", lambda *a, **kw: completed)
    # Should not raise
    DR.run_diarize(
        Path("/tmp/x.wav"),
        duration_secs=1.0,
        venv_python=tmp_path / "py",
        progress=None,
        forward_stderr=False,
    )


def test_progress_callback_exception_does_not_break_run(monkeypatch, tmp_path):
    """A misbehaving emitter must not abort an otherwise successful run."""
    payload = _sample_diarization_payload()
    sentinel_line = WORKER_JSON_SENTINEL + json.dumps(payload)
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=sentinel_line + "\n",
        stderr="[diarize] 50%\n",
    )
    monkeypatch.setattr(DR.subprocess, "run", lambda *a, **kw: completed)

    class _Boom:
        def progress(self, *a, **kw):
            raise RuntimeError("boom")

    # Should not raise; result should be produced
    result = DR.run_diarize(
        Path("/tmp/x.wav"),
        duration_secs=1.0,
        venv_python=tmp_path / "py",
        progress=_Boom(),
        forward_stderr=False,
    )
    assert isinstance(result, DiarizationOutput)
