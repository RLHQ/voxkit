"""``voxkit eval`` 子命令入口。

对照人类金标 SRT 字幕，评估 ``<workdir>`` 内 voxkit 输出的 reseg 质量。
零 LLM 零网络，纯计算——可在 CI 频繁跑。指标语义详见 ``docs/eval.md`` 与
``docs/eval-baseline-observations.md``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxkit.core.eval_metrics import build_eval_report, write_eval_report


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval",
        help=(
            "对照人类金标 SRT 评估 voxkit reseg 质量，输出 eval.report.json。"
            "纯计算，零 LLM 零网络"
        ),
    )
    p.add_argument("workdir", help="voxkit transcribe/proofread 的 workdir")
    p.add_argument(
        "--reference",
        required=True,
        help="人工金标 SRT 路径（与 workdir 内字幕语种一致）",
    )
    p.add_argument(
        "--lang",
        required=True,
        help="语种代码（zh/en/...），仅做记录，不影响计算",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.3,
        help="边界对齐容忍度（秒，默认 0.3）",
    )
    p.add_argument(
        "--output",
        default=None,
        help="覆盖输出路径（默认 <workdir>/eval.report.json）",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    reference = Path(args.reference)
    output = Path(args.output) if args.output else workdir / "eval.report.json"

    if not workdir.is_dir():
        sys.stderr.write(f"error: workdir not a directory: {workdir}\n")
        return 2
    if not reference.is_file():
        sys.stderr.write(f"error: reference SRT not found: {reference}\n")
        return 2

    try:
        report = build_eval_report(
            workdir=workdir,
            reference=reference,
            language=args.lang,
            tolerance_s=args.tolerance,
        )
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    write_eval_report(output, report)

    a = report.alignment
    m = report.metrics
    sys.stdout.write(
        f"eval report written: {output}\n"
        f"  artifact={report.sourceArtifact} lang={report.language} tol=±{report.tolerance_s}s\n"
        f"  cues: vk={a['vk_cues']} gold={a['gold_cues']} "
        f"density={a['density_ratio']:.3f}\n"
        f"  boundary: precision={m['boundary_precision']:.3f} "
        f"recall={m['boundary_recall']:.3f} f1={m['boundary_f1']:.3f}\n"
        f"  drift: chars={m['chars_drift']:+.2f} dur={m['dur_drift_s']:+.2f}s\n"
        f"  broken_latin_words={m['broken_latin_words']}\n"
    )
    return 0
