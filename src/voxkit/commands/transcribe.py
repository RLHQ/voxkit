"""voxkit transcribe — whisper.cpp 后端 ASR 转录子命令。

CLI 入口 + 参数验证。pipeline 编排见 voxkit.core.transcribe_pipeline。

Round 2 完成：``run()`` 现在调用 :func:`run_pipeline` 跑完整流程。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxkit.core.constants import ExitCode

# Round 1 — workspace primitive (Agent W)
try:
    from voxkit.core.workspace import open_workspace, WorkspaceLockError
    _WORKSPACE_AVAILABLE = True
except ImportError:
    _WORKSPACE_AVAILABLE = False

# Round 2 — pipeline orchestrator (this Agent I)
try:
    from voxkit.core.transcribe_pipeline import (
        PipelineError,
        TranscribeRequest,
        run_pipeline,
    )
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """注册 `transcribe` 子命令。

    与 voxkit/commands/build_bundle.py + fetch_bundle.py 风格保持一致。
    """
    p = sub.add_parser(
        "transcribe",
        help="对音频/视频跑 whisper.cpp 转录 → transcript.raw.json + SRT/VTT",
    )

    p.add_argument(
        "input",
        help="音频或视频文件 (wav/mp3/m4a/flac/mp4/mov/mkv/webm)",
    )
    p.add_argument(
        "--workdir",
        required=True,
        help="数据正交工作目录（所有产物落此）",
    )
    p.add_argument(
        "--model",
        default="large-v3-turbo",
        help="模型别名或 .bin 绝对路径（默认 large-v3-turbo）",
    )
    p.add_argument(
        "--language",
        default="auto",
        help="语种代码 (auto/en/zh/ja/...)，默认 auto",
    )
    p.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="word_timestamps",
        help="词级时间戳（CJK 自动忽略）",
    )
    p.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 VAD（VAD 模型缺失时 warn-once 并降级）",
    )
    p.add_argument(
        "--logprob-thold",
        type=float,
        default=-0.8,
        dest="logprob_thold",
        help="logprob 阈值（whisper 默认 -1.0；本工具收紧到 -0.8）",
    )
    p.add_argument(
        "--source-id",
        default=None,
        dest="source_id",
        help="Remixr sourceId（默认取 input 文件名 stem）",
    )
    p.add_argument(
        "--keep-work",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="keep_work",
        help="失败时强制保留 work/ 目录（默认 keep）",
    )
    p.add_argument(
        "--json-events",
        action="store_true",
        dest="json_events",
        help="stderr NDJSON 事件流 + events.ndjson 镜像",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="毫秒；不传走 dynamic timeout",
    )
    p.add_argument(
        "--whisper-bin",
        default=None,
        dest="whisper_bin",
        help="whisper-cli 路径（默认自动发现）",
    )
    p.add_argument(
        "--vad-model",
        default=None,
        dest="vad_model",
        help="silero VAD model 路径（默认 env/brew 发现）",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用 chunk_NNN.json 检查点（默认 on）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="清空 work/ 重跑（等同 --no-resume）",
    )
    p.add_argument(
        "--blocklist",
        default=None,
        help="覆盖中文幻觉黑名单 JSON 路径（默认使用 bundled）",
    )
    p.add_argument(
        "--emit-srt",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="emit_srt",
        help="输出 SRT（默认 on）",
    )
    p.add_argument(
        "--emit-vtt",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="emit_vtt",
        help="输出 VTT（默认 on）",
    )


def run(args: argparse.Namespace) -> int:
    """验证参数 → 构造 :class:`TranscribeRequest` → 调用 :func:`run_pipeline`。

    Exit codes:
      * ``ExitCode.OK`` (0)              — success
      * ``ExitCode.GENERIC_FAIL`` (1)    — pipeline-level error / IO / merge / lock
      * ``ExitCode.ENV_PROBLEM`` (3)     — whisper-cli / model / ffmpeg discovery failed
    """
    # 1. Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"error: input file not found: {input_path}\n")
        return int(ExitCode.GENERIC_FAIL)

    workdir = Path(args.workdir).expanduser().resolve()

    # 2. Resolve --force vs --resume（任一触发都视为 force=True）
    force = args.force or not args.resume
    effective_resume = not force

    # 3. Open workspace（如模块可用）
    if not _WORKSPACE_AVAILABLE:
        sys.stderr.write(
            "error: voxkit.core.workspace not available; voxkit install is incomplete\n"
        )
        return int(ExitCode.GENERIC_FAIL)

    try:
        ws = open_workspace(workdir, force=force)
    except WorkspaceLockError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return int(ExitCode.GENERIC_FAIL)
    except Exception as exc:  # noqa: BLE001 — surface filesystem failures cleanly
        sys.stderr.write(f"error: failed to open workspace: {exc}\n")
        return int(ExitCode.GENERIC_FAIL)
    sys.stderr.write(f"voxkit transcribe: workspace ready at {ws.root}\n")

    # 4. source_id 默认值 = input.stem
    source_id = args.source_id or input_path.stem
    sys.stderr.write(f"voxkit transcribe: source_id={source_id}\n")

    if not _PIPELINE_AVAILABLE:
        sys.stderr.write(
            "voxkit transcribe: pipeline module unavailable; install voxkit completely\n"
        )
        return int(ExitCode.GENERIC_FAIL)

    # 5. Build request + run pipeline
    req = TranscribeRequest(
        input_path=input_path,
        workspace=ws,
        model=args.model,
        language=args.language,
        word_timestamps=args.word_timestamps,
        vad=args.vad,
        logprob_thold=args.logprob_thold,
        source_id=source_id,
        keep_work=args.keep_work,
        json_events=args.json_events,
        timeout_ms=args.timeout,
        whisper_bin_override=Path(args.whisper_bin) if args.whisper_bin else None,
        vad_model_override=Path(args.vad_model) if args.vad_model else None,
        blocklist_path=Path(args.blocklist) if args.blocklist else None,
        resume=effective_resume,
        emit_srt=args.emit_srt,
        emit_vtt=args.emit_vtt,
    )

    try:
        result = run_pipeline(req)
    except PipelineError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return int(exc.exit_code)
    except KeyboardInterrupt:
        sys.stderr.write("error: interrupted\n")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:  # noqa: BLE001 — last-ditch barrier
        sys.stderr.write(f"error: unexpected pipeline failure: {exc}\n")
        return int(ExitCode.GENERIC_FAIL)

    # 6. Success summary
    sys.stderr.write(
        f"transcribe complete: {len(result.voxkit_output.segments)} segments, "
        f"RTF={result.rtf:.3f}, elapsed={result.elapsed_secs:.1f}s\n"
    )
    return int(ExitCode.OK)


__all__ = ["run", "add_subparser"]
