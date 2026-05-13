"""voxkit CLI 入口：argparse 路由 4 个子命令。

主进程仅依赖 stdlib + pydantic（轻）。pyannote 调用代码由 lazy venv 内的
worker 子进程承担，保持 CLI 启动的快速与最小依赖。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from voxkit import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxkit",
        description="语音 → 结构化数据 toolkit（whisper.cpp + pyannote.audio）",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="显示版本号并退出",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    # ── doctor ──────────────────────────────────────────────
    p_doctor = sub.add_parser(
        "doctor",
        help="按目标自检依赖与环境配置",
    )
    p_doctor.add_argument(
        "--profile",
        choices=["all", "transcribe", "diarize"],
        default="all",
        help=(
            "检查目标：transcribe 只看转录必需项；diarize 只看说话人切分必需项；"
            "all 检查全部（默认）"
        ),
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
    from voxkit.core.constants import DEFAULT_MODEL, MODEL_CHOICES

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
        help="忽略 ~/.cache/voxkit 中已有的结果",
    )

    # ── align ───────────────────────────────────────────────
    p_align = sub.add_parser("align", help="把已有的 transcript+diarization 对齐成 SRT")
    p_align.add_argument("transcript", help="Remixr transcript.raw.json 路径")
    p_align.add_argument("diarization", help="voxkit diarize 输出 JSON 路径")
    p_align.add_argument("-o", "--output", required=True, help="输出 SRT 路径")
    p_align.add_argument(
        "--speaker-labels",
        choices=["ranked", "raw"],
        default="ranked",
    )

    # ── build-bundle / fetch-bundle（模型自托管）─────────────
    from voxkit.commands import build_bundle, fetch_bundle
    build_bundle.add_subparser(sub)
    fetch_bundle.add_subparser(sub)

    # ── transcribe（whisper.cpp 后端 ASR）─────────────────────
    from voxkit.commands import transcribe
    transcribe.add_subparser(sub)

    # ── proofread（LLM 字幕校对）─────────────────────────────
    from voxkit.commands import proofread
    proofread.add_subparser(sub)

    # ── translate（LLM 字幕翻译）─────────────────────────────
    from voxkit.commands import translate
    translate.add_subparser(sub)

    # ── review（artifact lifecycle）─────────────────────────
    from voxkit.commands import review
    review.add_subparser(sub)

    # ── quality（聚合指标报告）──────────────────────────────
    from voxkit.commands import quality
    quality.add_subparser(sub)

    # ── eval（对照金标的回归评估）───────────────────────────
    from voxkit.commands import eval as eval_cmd
    eval_cmd.add_subparser(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 延迟导入：保持 --help / 错误参数路径轻量
    if args.cmd == "doctor":
        from voxkit.commands.doctor import run as run_doctor
        return run_doctor(args)
    if args.cmd == "setup":
        from voxkit.commands.setup import run as run_setup
        return run_setup()
    if args.cmd == "diarize":
        from voxkit.commands.diarize import run as run_diarize
        return run_diarize(args)
    if args.cmd == "align":
        from voxkit.commands.align import run as run_align
        return run_align(args)
    if args.cmd == "build-bundle":
        from voxkit.commands.build_bundle import run as run_build
        return run_build(args)
    if args.cmd == "fetch-bundle":
        from voxkit.commands.fetch_bundle import run as run_fetch
        return run_fetch(args)
    if args.cmd == "transcribe":
        from voxkit.commands.transcribe import run as run_transcribe
        return run_transcribe(args)
    if args.cmd == "proofread":
        from voxkit.commands.proofread import run as run_proofread
        return run_proofread(args)
    if args.cmd == "translate":
        from voxkit.commands.translate import run as run_translate
        return run_translate(args)
    if args.cmd == "review":
        from voxkit.commands.review import run as run_review
        return run_review(args)
    if args.cmd == "quality":
        from voxkit.commands.quality import run as run_quality
        return run_quality(args)
    if args.cmd == "eval":
        from voxkit.commands.eval import run as run_eval
        return run_eval(args)

    parser.error(f"未知子命令: {args.cmd}")
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
