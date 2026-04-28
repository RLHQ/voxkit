"""voxkit diarize — 主命令。

流程：
  1. 验证输入文件 → 抽音频（视频）/ 直接用（音频）
  2. ensure_venv 触发 lazy install（首次自动）
  3. spawn `<venv>/bin/python -m voxkit.core.pipeline ...`，把 worker 的 stdout
     当作 DiarizationOutput JSON
  4. 写到用户指定的 -o 路径
  5. 如果给了 --transcript / --emit-aligned-srt，调用 align 子命令逻辑
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from voxkit.commands import align as align_cmd
from voxkit.core import audio as core_audio
from voxkit.core import env as core_env
from voxkit.core import lazy_install
from voxkit.core.constants import ExitCode
from voxkit.core.diarize_runner import (
    DiarizeFailed,
    DiarizeTimeout,
    run_diarize,
)
from voxkit.io.progress import ProgressEmitter
from voxkit.io.schema import DiarizationOutput


def _run_worker(
    *,
    venv_python: Path,
    audio_path: Path,
    duration_secs: float,
    extracted_from: Path | None,
    args: argparse.Namespace,
    progress: ProgressEmitter,
) -> DiarizationOutput:
    """Thin CLI-side wrapper: build params from ``argparse.Namespace`` and
    delegate to :func:`voxkit.core.diarize_runner.run_diarize`.

    Maintains the original side-effect of forwarding worker stderr to the
    parent terminal (the runner does that when ``forward_stderr=True``).
    Maps runner exceptions to the existing ``progress.error`` + ``sys.exit``
    contract so CLI behaviour is byte-identical to v0.3.0.
    """
    try:
        return run_diarize(
            audio_path,
            duration_secs=duration_secs,
            venv_python=venv_python,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            model=args.model,
            device=args.device,
            speaker_labels=args.speaker_labels,
            extracted_from=extracted_from,
            progress=None,  # CLI relies on stderr forwarding, not callback
            env=core_env.patched_env(),
            timeout_secs=None,
            forward_stderr=True,
            json_events=args.json_events,
        )
    except DiarizeFailed as exc:
        # worker 已 emit error 事件；这里仅给非 0 信号
        progress.error(
            "WORKER_EXIT",
            f"worker 退出码 {exc.returncode}",
            fix="查看上方 stderr 输出",
        )
        sys.exit(exc.returncode)
    except DiarizeTimeout as exc:
        progress.error("WORKER_TIMEOUT", str(exc))
        sys.exit(int(ExitCode.WORKER_FAILED))
    except ValueError as exc:
        # sentinel missing or bad JSON — both surface here
        msg = str(exc)
        if "sentinel" in msg:
            progress.error("WORKER_NO_OUTPUT", msg)
        else:
            progress.error("WORKER_BAD_JSON", msg)
        sys.exit(int(ExitCode.WORKER_FAILED))


def run(args: argparse.Namespace) -> int:
    progress = ProgressEmitter(json_events=args.json_events)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        progress.error("INPUT_NOT_FOUND", f"输入文件不存在: {input_path}")
        return int(ExitCode.GENERIC_FAIL)

    if args.emit_aligned_srt and not args.transcript:
        progress.error("BAD_ARGS", "--emit-aligned-srt 需要同时传 --transcript")
        return int(ExitCode.GENERIC_FAIL)

    # ── Step 1: ensure venv（先于 prepare_audio：避免装失败时浪费抽音 30s）──
    try:
        venv_info = lazy_install.ensure_venv(verbose=not args.json_events)
    except lazy_install.SetupError as e:
        progress.error("LAZY_INSTALL_FAILED", str(e),
                       fix="手动跑 voxkit setup 排查；或 rm -rf ~/.local/share/voxkit 重来")
        return int(ExitCode.GENERIC_FAIL)

    # ── Step 2: prepare audio ───────────────────────────────
    progress.progress("audio_extract", 0)
    try:
        prep = core_audio.prepare_audio(input_path)
    except Exception as e:
        progress.error("AUDIO_PREP_FAILED", str(e))
        return int(ExitCode.ENV_PROBLEM)
    progress.progress("audio_extract", 100)
    progress.info(
        f"audio={prep.audio_path}  duration={prep.duration_secs:.1f}s"
        + (f"  (extracted from {prep.extracted_from.name})" if prep.extracted_from else "")
    )

    # ── Step 3: run worker ──────────────────────────────────
    try:
        result = _run_worker(
            venv_python=venv_info.venv_python,
            audio_path=prep.audio_path,
            duration_secs=prep.duration_secs,
            extracted_from=prep.extracted_from,
            args=args,
            progress=progress,
        )
    finally:
        if prep.cleanup and prep.cleanup.exists():
            try:
                prep.cleanup.unlink()
            except OSError:
                pass

    # ── Step 4: write output ────────────────────────────────
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(by_alias=True, indent=2))
    progress.info(f"diarization JSON → {out_path}")

    # ── Step 5: optional alignment ──────────────────────────
    if args.transcript and args.emit_aligned_srt:
        srt_path = align_cmd.align_to_srt(
            transcript_path=Path(args.transcript).expanduser().resolve(),
            diarization=result,
            out_srt=Path(args.emit_aligned_srt).expanduser().resolve(),
            speaker_labels=args.speaker_labels,
        )
        progress.info(f"aligned SRT → {srt_path}")

    progress.done(elapsed_secs=result.elapsed_secs)
    return int(ExitCode.OK)


__all__ = ["run"]
