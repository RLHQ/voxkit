"""测试 voxkit transcribe 子命令的 argparse 形态 + run() stub 行为。

Round 1 范围：parser 结构 + run() 参数处理 + stub 退出码。
不涉及 subprocess、whisper-cli、真实音频处理（那是 Round 2 Agent I）。
"""

from __future__ import annotations

import pytest

from voxkit.cli import _build_parser
from voxkit.commands.transcribe import run
from voxkit.core.constants import ExitCode


# ── parser 结构 ──────────────────────────────────────────────


def _parse(*extra: str):
    """便捷：构造 parser 并用 transcribe + 必需参数解析剩余 flag。"""
    parser = _build_parser()
    return parser.parse_args(
        ["transcribe", "/tmp/in.mp4", "--workdir", "/tmp/out", *extra]
    )


def test_basic_positional_and_workdir():
    args = _parse()
    assert args.cmd == "transcribe"
    assert args.input == "/tmp/in.mp4"
    assert args.workdir == "/tmp/out"


def test_default_flag_values():
    args = _parse()
    assert args.model == "large-v3-turbo"
    assert args.language == "auto"
    assert args.word_timestamps is True
    assert args.vad is True
    assert args.logprob_thold == pytest.approx(-0.8)
    assert args.source_id is None
    assert args.keep_work is True
    assert args.json_events is False
    assert args.timeout is None
    assert args.whisper_bin is None
    assert args.vad_model is None
    assert args.resume is True
    assert args.force is False
    assert args.blocklist is None
    assert args.emit_srt is True
    assert args.emit_vtt is True


def test_no_word_timestamps_flips_false():
    args = _parse("--no-word-timestamps")
    assert args.word_timestamps is False


def test_no_vad_flips_false():
    args = _parse("--no-vad")
    assert args.vad is False


def test_logprob_thold_parses_float():
    args = _parse("--logprob-thold", "-0.5")
    assert isinstance(args.logprob_thold, float)
    assert args.logprob_thold == pytest.approx(-0.5)


def test_source_id_override():
    args = _parse("--source-id", "custom_id")
    assert args.source_id == "custom_id"


def test_language_override():
    args = _parse("--language", "zh")
    assert args.language == "zh"


def test_force_flag_independent_of_resume():
    args = _parse("--force")
    assert args.force is True
    # --resume 默认仍是 True；force 是独立 flag
    assert args.resume is True


def test_no_resume_flag():
    args = _parse("--no-resume")
    assert args.resume is False
    assert args.force is False


def test_blocklist_path():
    args = _parse("--blocklist", "/path/to/file")
    assert args.blocklist == "/path/to/file"


def test_no_emit_srt_and_vtt():
    args = _parse("--no-emit-srt", "--no-emit-vtt")
    assert args.emit_srt is False
    assert args.emit_vtt is False


def test_json_events_flag():
    args = _parse("--json-events")
    assert args.json_events is True


def test_keep_work_negation():
    args = _parse("--no-keep-work")
    assert args.keep_work is False


def test_timeout_int():
    args = _parse("--timeout", "60000")
    assert args.timeout == 60000


def test_whisper_bin_and_vad_model_paths():
    args = _parse("--whisper-bin", "/usr/local/bin/whisper-cli",
                  "--vad-model", "/opt/silero.bin")
    assert args.whisper_bin == "/usr/local/bin/whisper-cli"
    assert args.vad_model == "/opt/silero.bin"


# ── run() 行为 ───────────────────────────────────────────────


def test_run_missing_input_returns_generic_fail(tmp_path, capsys):
    parser = _build_parser()
    args = parser.parse_args(
        ["transcribe", str(tmp_path / "does-not-exist.mp4"),
         "--workdir", str(tmp_path / "ws")]
    )
    rc = run(args)
    assert rc == int(ExitCode.GENERIC_FAIL)
    captured = capsys.readouterr()
    assert "input file not found" in captured.err.lower()


# ── CLI → pipeline contract tests (mock-based) ─────────────────────────────
# Round 2 Agent I wired the stub up to a real pipeline. These tests now mock
# `run_pipeline` so they verify CLI argument parsing + request construction
# without invoking ffmpeg/whisper-cli. For the real end-to-end tests see
# tests/test_transcribe_e2e.py (gated by @pytest.mark.requires_whisper).


def _patch_run_pipeline(monkeypatch):
    """Replace voxkit.core.transcribe_pipeline.run_pipeline with a recorder.

    Returns a list that captures every TranscribeRequest the CLI builds.
    The patched function returns a dummy success result so run() can reach
    its 0-exit code path.
    """
    from types import SimpleNamespace

    captured: list = []

    def fake_run_pipeline(req):
        captured.append(req)
        return SimpleNamespace(
            voxkit_output=SimpleNamespace(segments=[], rtf=0.0, elapsed_secs=0.0),
            artifacts={},
            warnings=[],
            elapsed_secs=0.0,
            rtf=0.0,
        )

    # Patch both the source location AND the import re-export inside transcribe.py.
    import voxkit.core.transcribe_pipeline as pipeline_mod
    import voxkit.commands.transcribe as cmd_mod

    monkeypatch.setattr(pipeline_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cmd_mod, "run_pipeline", fake_run_pipeline)
    return captured


def test_run_invokes_pipeline_with_parsed_args(tmp_path, monkeypatch):
    """CLI builds a TranscribeRequest and hands it to run_pipeline (mocked)."""
    inp = tmp_path / "x.mp4"
    inp.write_bytes(b"fake")
    captured = _patch_run_pipeline(monkeypatch)

    parser = _build_parser()
    args = parser.parse_args(
        ["transcribe", str(inp), "--workdir", str(tmp_path / "ws")]
    )
    rc = run(args)
    assert rc == int(ExitCode.OK)
    assert len(captured) == 1, "run() must invoke run_pipeline exactly once"
    req = captured[0]
    assert req.input_path == inp
    assert req.source_id == "x"  # default = input stem


def test_run_source_id_default_is_input_stem(tmp_path, monkeypatch):
    """Without --source-id, request.source_id defaults to Path(input).stem."""
    inp = tmp_path / "my-recording.wav"
    inp.write_bytes(b"")
    captured = _patch_run_pipeline(monkeypatch)

    parser = _build_parser()
    args = parser.parse_args(
        ["transcribe", str(inp), "--workdir", str(tmp_path / "ws")]
    )
    rc = run(args)
    assert rc == int(ExitCode.OK)
    assert captured[0].source_id == "my-recording"


def test_run_explicit_source_id_wins(tmp_path, monkeypatch):
    """--source-id overrides the input-stem default."""
    inp = tmp_path / "x.wav"
    inp.write_bytes(b"")
    captured = _patch_run_pipeline(monkeypatch)

    parser = _build_parser()
    args = parser.parse_args(
        ["transcribe", str(inp), "--workdir", str(tmp_path / "ws"),
         "--source-id", "explicit_one"]
    )
    rc = run(args)
    assert rc == int(ExitCode.OK)
    assert captured[0].source_id == "explicit_one"
