"""voxsplit CLI 入口：argparse 路由 4 个子命令。

主进程仅依赖 stdlib + pydantic（轻）。pyannote 调用代码由 lazy venv 内的
worker 子进程承担，保持 CLI 启动的快速与最小依赖。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxsplit",
        description="独立 speaker diarization CLI（pyannote.audio 后端）",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    # ── doctor ──────────────────────────────────────────────
    sub.add_parser(
        "doctor",
        help="自检 6 项依赖与环境配置（uv/Python/HF token/4 gated/ffmpeg/venv）",
    )

    # ── setup ───────────────────────────────────────────────
    sub.add_parser(
        "setup",
        help="显式创建 venv + 装 pyannote.audio + 预下载模型",
    )

    # ── diarize ─────────────────────────────────────────────
    p_diarize = sub.add_parser("diarize", help="对音频/视频跑 speaker diarization")
    p_diarize.add_argument("input", help="音频或视频文件 (wav/mp3/m4a/flac/mp4/mov/mkv/webm)")
    p_diarize.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    p_diarize.add_argument("--transcript", help="Remixr transcript.raw.json，启用对齐输出")
    p_diarize.add_argument("--num-speakers", type=int, default=None)
    p_diarize.add_argument("--min-speakers", type=int, default=None)
    p_diarize.add_argument("--max-speakers", type=int, default=None)
    from voxsplit.core.constants import DEFAULT_MODEL, MODEL_CHOICES

    p_diarize.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    p_diarize.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
    )
    p_diarize.add_argument("--emit-aligned-srt", help="输出 SRT 路径（需 --transcript）")
    p_diarize.add_argument(
        "--json-events",
        action="store_true",
        help="stderr 改为 NDJSON 事件协议（机器消费）",
    )
    p_diarize.add_argument(
        "--speaker-labels",
        choices=["ranked", "raw"],
        default="ranked",
    )
    p_diarize.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略 ~/.cache/voxsplit 中已有的结果",
    )

    # ── align ───────────────────────────────────────────────
    p_align = sub.add_parser("align", help="把已有的 transcript+diarization 对齐成 SRT")
    p_align.add_argument("transcript", help="Remixr transcript.raw.json 路径")
    p_align.add_argument("diarization", help="voxsplit diarize 输出 JSON 路径")
    p_align.add_argument("-o", "--output", required=True, help="输出 SRT 路径")
    p_align.add_argument(
        "--speaker-labels",
        choices=["ranked", "raw"],
        default="ranked",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 延迟导入：保持 --help / 错误参数路径轻量
    if args.cmd == "doctor":
        from voxsplit.commands.doctor import run as run_doctor
        return run_doctor()
    if args.cmd == "setup":
        from voxsplit.commands.setup import run as run_setup
        return run_setup()
    if args.cmd == "diarize":
        from voxsplit.commands.diarize import run as run_diarize
        return run_diarize(args)
    if args.cmd == "align":
        from voxsplit.commands.align import run as run_align
        return run_align(args)

    parser.error(f"未知子命令: {args.cmd}")
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
